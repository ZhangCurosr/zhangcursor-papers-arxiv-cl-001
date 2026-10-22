# DATA SCHEDULES AND LEARNING RATE SCHEDULES SHOULD BE DESIGNED TOGETHER

Haodong Wen<sup>1,∗</sup> Kairong Luo<sup>1,\*</sup> Tingkai Yan<sup>1,\*</sup> Wenguang Chen<sup>1,2</sup> Kaifeng Lyu<sup>1</sup>

<sup>1</sup>Tsinghua University

<sup>2</sup>Peng Cheng Laboratory

{whd25,lkr24,ytk26}@mails.tsinghua.edu.cn

{cwg,klyu}@mail.tsinghua.edu.cn

## ABSTRACT

Mid-training (MT) is a training phase in LLM training pipelines that continues pretraining (PT) on smaller datasets of higher value. Taking the PT and MT phases as a whole, training usually starts with only PT data and then transitions completely to MT data. In this work, we view this complete transition from PT to MT data as a special case of a broader class of data mixing schedules that specify how the proportions of PT and MT tokens in each batch change over training. Under fixed total training compute and MT token budgets, we ask: Is this complete transition to MT data optimalfor minimizingfinal MT loss? We show that the answer depends on the learning rate (LR) schedule: one data schedule may outperform another under one LR schedule, but changing only the LR schedule can reverse the outcome. Specifically, when the LR decays throughout continued training, introducing MT data earlier and mixing it with PT data lowers the final MT loss compared with placing all MT data at the end of training, but this result is reversed if we use constant LR. We therefore study how to jointly choose when to introduce MT data and when to decay the LR through ablation studies, and find that it is better to decay the LR only after MT data enters training. Finally, we develop a theoretical model that explains why the optimal data schedule for a fixed MT token budget depends on the LR schedule.

## 1 INTRODUCTION

Pretraining (PT) and mid-training (MT) are two consecutive phases in large language model (LLM) training pipelines. PT uses broad web corpora, while MT uses smaller, curated mixtures enriched with mathematics, code, and other targeted data (DeepSeek-AI et al., 2024; Yang et al., 2025; Team OLMo et al., 2025; Abdin et al., 2024; GLM-4.5 Team, 2025; LLM-Core Xiaomi, 2025; K2 Team et al., 2025). In the simplest form of this pipeline, training starts with PT data alone and then transitions completely to MT data.

However, reserving all MT tokens for the final phase is not the only way to allocate the MT token budget over training. From a data mixing perspective, the two phases described above form a particular data schedule. Let γ(t) denote the fraction of MT tokens in the batch at step t, which we call the data schedule function. The complete transition to MT data as described above corresponds $\mathbf { t o } \gamma ( t ) = \mathbf { 1 } \{ t > \tau _ { 0 } \}$ , where $\tau _ { 0 }$ is the transition time. In general, different allocations of MT tokens over training correspond to different data schedules γ(t). This motivates a natural question: with the total training compute and MT token budgetfixed, is this complete transition to MT data optimalfor minimizingfinal MT loss?

We study this question by comparing two schedule families that introduce MT data either abruptly or gradually. Both schedule families begin with PT data alone and end at a chosen MT ratio $\gamma _ { \mathrm { e n d } } \in ( 0 , 1 ]$ but they differ in how MT data enters training. The Two-Stage (TS) schedule jumps directly to this ratio and holds it fixed until training ends. The Zero-then-Linear (ZL) schedule instead increases the

![](images/77ca9219a6b089c0dcc44da2d824bb88dc0a37f22048a2ddff22f4e667dc656d.jpg)

![](images/5486bef935743f96db46fc528d1477e386e366717828d1df7a7b1d1bbcd75dba.jpg)

(a) Left: Linear Decay LR & Two-Stage (TS) data schedule. Right: PT-MT losses under different Two-Stage (TS) data schedules with FDR.  
![](images/8e588d04a282af44a551ae5cc0f3e4649149997b0e6a4f89487b8a492d71522b.jpg)

![](images/6fe64afdbf5529fbd97327fbb699652b93ed9df68ea61a44defaff46ddbbb667.jpg)  
(b) Left: Constant LR & Two-Stage (TS) data schedule. Right: PT-MT losses under different Two-Stage (TS) data schedules with constant LR.

Figure 1: Replaying PT data can be both beneficial and harmful under different LR schedules. The compute and MT data budgets are fixed (the shaded areas in the left panels show the MT budgets). With Full LR Decay Range throughout continued training (FDR, top), mixing PT and MT data in the second TS stage gives lower final MT loss than using MT data alone. With constant LR (Constant LR, bottom), the pure MT stage instead gives the lowest tested MT loss. Results use the dense model at a 10% MT budget. Training-step coordinates in the schedule panels are schematic: the PT checkpoint is drawn at 100k steps, rather than its actual step 95,368.

ratio gradually:

$$
\gamma ( t ) = \left\{ \begin{array} { l l } { \displaystyle 0 , } & { 0 \leq t \leq \tau _ { 1 } , } \\ { \displaystyle \gamma _ { \mathrm { e n d } } \frac { t - \tau _ { 1 } } { T - \tau _ { 1 } } , } & { \tau _ { 1 } < t \leq T , } \end{array} \right. \quad \quad 0 \leq \tau _ { 1 } < T .
$$

Here, T is the number of steps in the continued training period after the initial PT checkpoint, and $\tau _ { 1 }$ is the time when MT data first enters training. At a fixed batch size, the area $\begin{array} { r } { \int _ { 0 } ^ { T } \gamma ( t ) } \end{array}$ dt is proportional to the number of MT tokens used. We hold this area fixed across schedules to compare different ways of allocating the same MT budget over time.

We first revisit the debate over whether we should replay PT data during continued training or transition to MT data alone. At first glance, introducing MT data earlier while replaying PT data could produce larger updates while the learning rate (LR) is still high, but it also leaves more subsequent training that may weaken its effect. This tradeoff suggests that whether replaying PT data is beneficial may depend on the LR schedule, and we test this by comparing the same set of data schedules under different LR schedules.

We start by considering LR decay throughout the continued training period. Our results support the findings of Kotha & Liang (2026), who found that mixing PT and MT data in the second stage lowers final MT loss compared with using MT data alone under the same compute budget. In our experiments, we fix both the total training compute and MT token budget. Mixing PT data into the second stage therefore requires introducing MT data earlier and spreading it over more steps. We find that this earlier mixing yields lower final MT loss than placing all MT data in a final pure MT stage, consistently across dense and MoE models, multiple MT budgets, and different lengths of continued training (Figs. 2 and 5).

However, we find that replaying PT data may not be beneficial if we change the LR schedule. Specifically, holding the data schedules and all other settings fixed, if we replace LR decay with a constant LR, then the same PT replay now increases the final MT loss. Among TS schedules, placing all MT data in a final stage containing only MT data gives the lowest tested MT loss. Figure 1 illustrates this phenomenon at a 10% MT data budget, showing the data and LR schedules alongside their resulting PT and MT losses. The reversal persists across all our experimental setups (Section 2). These results show that the preferred data schedule cannot be determined independently of the LR schedule. In general, we identify the following ranking reversal phenomenon of data schedules: the ranking of data schedules by final MT loss can shift or even reverse under different LR schedules.

The ranking reversal phenomenon above suggests that the data and LR schedules should be designed together. We therefore seek joint scheduling of LR and data that minimizes final MT loss under fixed compute and MT token budgets. To explore what the optimal joint schedule of LR and data looks like, we hold each data schedule fixed and vary when LR decay begins relative to the time when MT data enters training, which we call the MT entry time.

We find that LR should decay after MT data enters training. Across all 19 sweeps for different LR and data schedules, the best tested data and LR schedule pairs follow the above pattern and achieve lower final MT loss than constant LR or decay beginning at MT entry (Section 2.3). For the dense model, the best tested LR decay covers the last 40% of the MT data window for TS and the last 20% for ZL (Fig. 3). See Appendix D for the complete dense and MoE sweeps.

Finally, we theoretically explain why the optimal data schedule depends on the LR schedule. We develop a tractable model of SGD on noisy quadratic objectives and optimize the data schedule under a fixed MT budget. In this model, placing all MT data at the end is optimal under a constant LR, but can lose optimality under sufficiently strong linear decay. Under additional spectral assumptions and for sufficiently long training, we derive a unique optimum that begins with PT data, passes through a mixture of PT and MT data, and ends with MT data (Section 3).

## 2 DATA SCHEDULES AND LEARNING RATE SCHEDULES SHOULD BE DESIGNED TOGETHER

In this section, we first reproduce the counterintuitive finding, inspired by Kotha & Liang (2026), that introducing MT data earlier and replaying PT data later during mid-training can improve MT loss. We then diagnose the role of the learning rate (LR) in this phenomenon. In particular, when using a constant LR during mid-training, placing MT data later instead leads to better MT loss. This observation aligns with Luo et al. (2026) and motivates us to design data and LR schedules jointly. We conduct over 250 runs across dense and MoE models. Across our LR decay sweeps, we obtain the lowest final MT loss by keeping the LR high after MT data enters and decaying only near the end of training (Section 2.3).

## 2.1 EXPERIMENTAL SETUP

In the following experiments, we use DCLM-Baseline as the PT data (Li et al., 2024) and the finemath-4plus subset of FineMath as the MT data (Ben Allal et al., 2025) to construct our data schedules. We conduct the data schedule experiments on both dense and MoE models. Our dense model has 630M parameters and mainly follows the Qwen2.5-0.5B architecture, but uses untied input and output embeddings (Qwen et al., 2024). It is first pretrained from scratch on 200B DCLM tokens during the PT phase and then trained on another 20.97B tokens during the MT phase. In the MT phase, MT tokens account for 10%, 25%, or 47.75% of the total tokens (2.10B, 5.24B, or 10.01B MT tokens), while replayed PT tokens make up the remainder. Since FineMath contains approximately 10B tokens in total, the MT data is consumed for at most one epoch. Our MoE model follows the OLMoE design (Muennighoff et al., 2024), with 2.99B total parameters, 64 experts per layer, and top-8 routing. The PT phase consists of 40B PT tokens, while the MT phase contains either 4B or 10B tokens in two experimental settings. More setup details are provided in Appendix B.

Data and LR schedule configurations. To study the interplay between data and LR schedules during MT, we consider two representative data schedules under a fixed MT token budget. Specifically, let T denote the number of continual pretraining steps, with t = 0 corresponding to a fixed PT checkpoint. Let γ(t) be the fraction of MT tokens in the training batch at training step t. We fix the overall MT token fraction α and the batch size, so the budget constraint on MT tokens is $\begin{array} { r } { \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t = \alpha T } \end{array}$

![](images/ea432de765dc8e60fbfcf874f856a09e59a28e59c07cebbf2b529d5b0a920bb4.jpg)

![](images/37d73954adc97518a28a9d7615bfa49c2e74326e66ba0c3bd8e99872d9c18c4e.jpg)

![](images/630d41fb683c1bd5f2ed000786339ad596b54a2177170a3ec0f79793901e6d1d.jpg)  
Figure 2: Across all three MT budgets, replaying PT data improves MT loss under FDR, but not under Const or MDR. The three panels show final MT loss as the final MT ratio in the second TS stage varies. Under the FDR regime, the lowest tested loss occurs when this stage mixes PT and MT data. Under the Const and MDR regimes, it occurs with pure MT data (100%). Large circles mark these minima, while other hollow markers show repeated runs. See Section 2.2 and Appendix C for the setup and full results.

Two Stage (TS) uses only PT data until $\tau _ { \mathrm { T S } } .$ , then switches to a fixed mixture with MT ratio $\gamma _ { \mathrm { e n d } } \colon$

$$
\gamma _ { \mathrm { T S } } ( t ) = \left\{ \begin{array} { l l } { 0 , } & { 0 \leq t < \tau _ { \mathrm { T S } } , } \\ { \gamma _ { \mathrm { e n d } } , } & { \tau _ { \mathrm { T S } } \leq t \leq T , } \end{array} \right. \quad \tau _ { \mathrm { T S } } = T \left( 1 - \frac { \alpha } { \gamma _ { \mathrm { e n d } } } \right) .
$$

Zero-then-Linear (ZL) uses only PT data until $\tau _ { \mathrm { Z L } }$ , then increases the MT ratio linearly from zero to $\gamma _ { \mathrm { e n d } } \mathbf { \hat { . } }$

$$
\gamma _ { \mathrm { z L } } ( t ) = \left\{ \begin{array} { l l } { \displaystyle 0 , } & { 0 \leq t < \tau _ { \mathrm { Z L } } , } \\ { \displaystyle \gamma _ { \mathrm { e n d } } \frac { t - \tau _ { \mathrm { Z L } } } { T - \tau _ { \mathrm { Z L } } } , } & { \tau _ { \mathrm { Z L } } \leq t \leq T , } \end{array} \right. \qquad \tau _ { \mathrm { Z L } } = T \left( 1 - \frac { 2 \alpha } { \gamma _ { \mathrm { e n d } } } \right) .
$$

The budget requires $\gamma _ { \mathrm { e n d } } \geq \alpha$ for $\mathrm { T S }$ and $\gamma _ { \mathrm { e n d } } \geq 2 \alpha$ for ZL.

When comparing different data schedules, we consider three training regimes: Const LR regime, where the peak LR remains constant throughout MT; Full LR Decay Range (FDR) regime, where the LR decays throughout the entire continual training period regardless of the data schedule; Matching the Decay Range (MDR) regime, where the LR decay starts when MT data is introduced and reaches the final LR at the end of MT. Figure 7 illustrates all six combinations of LR and data schedules. Appendix B gives the exact definitions and experimental details. Specifically, in this paper, all the learning rate decays are linear.

## 2.2 DIFFERENT LR SCHEDULES PREFER DIFFERENT DATA SCHEDULES

In practice, a TS data schedule typically sets $\gamma _ { \mathrm { e n d } } = 1$ , which places all MT data at the end of training. A more general TS, however, allows $\gamma _ { \mathrm { e n d } } < 1$ , which replays some PT data in the second stage. In the following paragraphs, we show that replaying PT data can improve MT loss in the FDR regime, but that this advantage disappears in the Const LR regime. The above ranking reversal phenomenon indicates that the timing of LR decay is crucial for MT loss, and that the benefit of a data schedule cannot be assessed independently of the LR schedule.

Replaying PT data can improve MT loss in the FDR regime. Modern training pipelines commonly place most MT data late in training to make better use of it (DeepSeek-AI et al., 2024; Yang et al., 2025; Team OLMo et al., 2025; Abdin et al., 2024). However, some recent work shows that retaining PT data in later stages can improve MT loss (Kotha & Liang, 2026). We observe that this counterintuitive pattern also appears across our experimental setups. As shown in Figure 1a, we adopt a TS data schedule and vary $\gamma _ { \mathrm { e n d } } . ~ \mathbf { A s } \ \gamma _ { \mathrm { e n d } }$ increases, more MT data is placed later, and the range over which MT data is used narrows. We find that both PT loss and MT loss generally increase with $\gamma _ { \mathrm { e n d } } .$ A similar trend also holds in Figures 5 and 8 to 10, across architectures and MT budgets. These results confirm that the observation in prior work holds across a range of settings (Kotha & Liang, 2026).

Replaying PT data is not beneficial in the Const LR regime. As shown in Figure 1b, the advantage is reversed when we use a constant LR with the same type of TS data schedule. $\mathrm { \bf A s } \gamma _ { \mathrm { e n d } }$ increases, MT loss gradually decreases while PT loss gradually increases. This observation reveals that small learning rates in the later steps of training lead to underutilization of MT data, consistent with the observation in prior work on data curricula (Luo et al., 2026). These ranking reversal phenomena are also observed in Figures 5, 8 and 9, across architectures and MT budgets; details are given in Appendices C and F. Therefore, different LR schedules favor different data schedules, and the benefit of a data schedule cannot be separated from the LR schedule.

![](images/2ad1f2391a08f6675d512d13ea6ff9c59758908d1aaf8801502d2bc420862e4b.jpg)  
Figure 3: LR should decay well after MT enters training for TS and ZL data schedules. The bottom row shows how final MT loss changes with relative LR decay duration for TS and ZL across three MT budgets. ZL reaches its lowest MT loss at 20%, and TS at 40%, improving on both Const (0%) and MDR (100%). The top row illustrates TS (left) and ZL (right) at a 10% MT budget, paired with LR schedules of different relative LR decay durations, measured as percentages of the time from MT entry to training completion. See Section 2.3 and Appendix D for the setup and further discussion.

The timing of LR decay is crucial for MT loss. Since small learning rates after LR annealing can lead to underutilization of MT data, the timing of LR decay is crucial for MT loss. For example, matching the LR decay range to the MT data range (MDR) can improve MT data efficiency over a full-range LR decay (FDR). As shown in Figures 2 and 5, MDR can improve MT loss at the same γ across both dense and MoE models and MT data budgets. $\mathrm { A s \gamma \gamma _ { e n d } }$ increases, MT data is placed later, and the benefit of matching LR decay also increases. This observation shows that the LR schedule regime is important for optimizing the data schedule: even after tuning $\gamma _ { \mathrm { e n d } }$ under FDR, its final MT loss is higher than that under MDR at the same $\gamma _ { \mathrm { e n d } }$ . In addition, our results show that MDR lowers final MT loss relative to Const at all three dense budgets but raises it in all three main MoE settings (Figures 2 and 5). This motivates delaying LR decay to maintain a higher average LR over the MT data range. We next test whether keeping the LR high after MT data is introduced, followed by a shorter final cooldown, can improve on both Const and MDR.

## 2.3 LEARNING RATE DECAY SHOULD BEGIN WELL AFTER MT DATA ENTERS TRAINING

Section 2.2 shows that the data and LR schedules must be chosen together, and matching the LR decay range to the MT data range can improve MT loss. Moreover, the timing of LR decay affects MT performance, and starting LR decay at MT entry time is not always beneficial. We now ask how long LR decay should last to minimize final MT loss for each fixed data schedule.

Delaying LR decay until after MT entry improves model performance over both Const and MDR. Keeping the LR high allows larger updates from MT data, while a final LR decay can reduce the effect of gradient noise. To balance these effects, we keep each data schedule fixed and vary when the LR begins to decrease. The LR remains at its peak before decaying linearly to the same final value. We define the relative LR decay duration as the number of LR decay steps divided by the number of steps from MT entry time to training completion, expressed as a percentage. It is zero for Const and 100% for MDR. Across all 19 scans, the best tested relative LR decay durations lie between 10% and 40%, giving lower MT loss than both baselines for dense and MoE models (Appendix D). Figure 3 shows this pattern for dense models: the lowest losses occur when the LR stays high through most of the MT data window and decreases only near the end. This delayed LR decay design achieves a better balance and outperforms both Const and MDR.

The optimal relative LR decay duration for ZL is about half that for TS. The results above show that a delayed LR decay gives lower MT loss. This holds for both TS and ZL, and the two data schedules favor different relative LR decay durations. Additionally, we find an interesting result that across dense and MoE experiments, the best tested relative LR decay duration for ZL is about half that for TS (Figures 3, 15, 16 and 21). Each schedule uses its own MT data window, from MT entry time to training completion, to measure the relative LR decay duration. TS uses a fixed MT ratio during this interval, while ZL increases it gradually from zero. To use the same MT budget and reach the same final MT ratio, ZL therefore needs an interval twice as long. Its smaller relative LR decay duration therefore corresponds to approximately the same number of LR decay steps as TS. We also repeat the LR decay sweep for dense TS schedules with different final MT ratios and find that the best tested relative LR decay duration remains unchanged (Figures 11 and 13).

The preferred relative LR decay duration is stable within the model family. We further conduct extensive sweeps on the optimal relative LR decay duration for the MoE model. We find that the above optimal values (10% for ZL and 20% for TS) remain unchanged across different setups. Specifically, in MoE, the best tested relative LR decay durations remain the same across continued training lengths, and TS retains its preference at earlier PT checkpoints (Figures 19 and 24). We also conduct an ablation on the MT dataset. We find that replacing FineMath with StarCoderData (Li et al., 2023) preserves the preferred relative LR decay durations for both TS and ZL (Figure 25). Appendix F describes the dataset and comparison setup. Appendix D summarizes all 19 scans.

## 3 THEORETICAL ANALYSIS

In this section, we provide a theoretical analysis to understand how the ranking reversal phenomenon between data schedules for different learning rate schedules might arise. Specifically, we consider a simple setting where SGD optimizes a block diagonal quadratic loss function. Though this model does not fully capture the complexity of data mixing and learning rate schedules, we believe it offers insight into how these two can be coupled together in the optimization landscape. Further, under a certain spectral assumption used in previous scaling law analyses, we can give a closed form of the optimal data schedule under practical learning rate schedules with decay.

## 3.1 SETUP

Given a data distribution D, we consider a quadratic loss $\begin{array} { r } { \mathcal { L } _ { \mathcal { D } } ( \boldsymbol { \theta } ) = \frac { 1 } { 2 } \boldsymbol { \theta } ^ { \top } \boldsymbol { M } \boldsymbol { \theta } } \end{array}$ , where $\pmb \theta \in \mathbb { R } ^ { d }$ is the trainable parameter vector, and M is the covariance matrix of the distribution D. During training, we consider a mixture of PT and MT data, drawn from the distributions D and D , respectively. We denote their covariance matrices by $\pmb { G } = \mathbb { E } _ { \pmb { D } _ { \mathrm { P T } } } \left[ \pmb { x } \pmb { x } ^ { \top } \right]$ and $\pmb { H } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { M T } } } \left[ \pmb { x } \pmb { x } ^ { \top } \right]$ . We evaluate our model using the MT loss $\begin{array} { r } { { \mathcal { L } } ( \pmb { \theta } ) = \frac { 1 } { 2 } \pmb { \theta } ^ { \top } \pmb { H } \pmb { \theta } } \end{array}$

PT and MT data. In practice, PT and MT data share some foundational knowledge, while each also contains knowledge that is specific to its own domain. For example, broad web PT data and mathematical MT data both contain general language patterns, while covering different specialized features. Motivated by this structure, we model shared and domain specific features through block diagonal covariance matrices, with a structure similar to those studied in transfer learning and data mixing theory (Ju et al., 2023; Mallinar et al., 2024; Pan et al., 2026; Dai & Zheng, 2026) as ${ \cal H } = \mathrm { d i a g } \big ( H _ { 0 } , H _ { 1 } , { \bf 0 } _ { d _ { 2 } \times d _ { 2 } } \big ) , G = \mathrm { d i a g } \big ( H _ { 0 } , { \bf 0 } _ { d _ { 1 } \times d _ { 1 } } , H _ { 2 } \big )$ . Here $H _ { 0 } , H _ { 1 }$ , and $H _ { 2 }$ represent the covariance blocks for shared, MT-specific, and PT-specific features, respectively, similar to Dai & Zheng (2026). For $l \in \{ 0 , 1 , 2 \}$ , we assume $\pmb { H } _ { l } = \mathrm { d i a g } ( \lambda _ { 1 } ^ { ( l ) } , \dots , \lambda _ { d _ { l } } ^ { ( l ) } )$ with $\lambda _ { i } ^ { ( l ) } > 0$ and $d = d _ { 0 } + d _ { 1 } + d _ { 2 }$

![](images/d6aea944a33e1db37ca56506309fe3471f71298bc903213fc8acfd0aee65baa8.jpg)  
Figure 4: The optimal data schedule can have three stages under LR decay. The first two columns show an abrupt switch from pure PT to pure MT. With stronger LR decay in the last two columns, the teal curves instead include a mixing stage in which the MT ratio increases continuously. This change occurs at both MT budgets shown. See Theorem 3.1 for the theoretical result and Appendix M for the numerical setup.

Data mixing and SGD training dynamics. We analyze SGD on mixtures of PT and MT data using a continuous-time SDE approximation (Jastrz˛ebski et al., 2017; Li et al., 2017; 2021; Malladi et al., 2022). Let T denote the training horizon in continuous time, and let ${ \boldsymbol \gamma } ( t ) \in [ 0 , 1 ]$ denote the probability of sampling MT data at time $t \in [ 0 , T ]$ ]. At time $t ,$ the training objective is given by $\begin{array} { r } { \mathcal { L } _ { t } ( \pmb { \theta } ) = \frac { 1 } { 2 } \pmb { \theta } ^ { \top } \big ( \gamma ( t ) \pmb { H } + ( 1 - \gamma ( t ) ) \pmb { G } \big ) \pmb { \theta } } \end{array}$ . We model each stochastic gradient as $\nabla { \mathcal { L } } _ { t } ( \pmb { \theta } )$ plus independent additive noise with mean zero and covariance Σ. The continuous-time approximation describes the average gradient update through a deterministic term and the accumulated gradient noise through Brownian motion. For a learning rate schedule $\eta ( t )$ , the training dynamics of $\theta _ { t }$ are given by the following SDE:

$$
\mathrm { d } \pmb { \theta } _ { t } = - \eta ( t ) \nabla \mathcal { L } _ { t } ( \pmb { \theta } _ { t } ) \mathrm { d } t + \eta ( t ) \sqrt { h \Sigma } \mathrm { d } \pmb { B } _ { t } .\tag{1}
$$

Here $\mathbf { } _ { \mathbf { } \mathbf { } \mathbf { } B _ { t } }$ is a standard d-dimensional Brownian motion, and $h > 0$ is the interval in continuous time between consecutive SGD updates. Σ is the covariance matrix of the additive gradient noise. See Appendix H for the discrete SGD update and the derivation of its SDE approximation. Appendix M.2 compares the final MT loss predicted by the SDE with exact calculations and Monte Carlo simulations of discrete SGD.

In our analysis, we focus on a regime where the quantity of MT data is constrained:

Condition 3.1. For somefixed $\alpha \in ( 0 , 1 ]$ , MT data accounts for at most an α fraction of the total training data.

Under this budget constraint, we seek a data schedule that minimizes the expected final MT loss for a fixed training horizon $T$ and LR schedule η:

$$
\operatorname* { m i n } _ { \gamma : [ 0 , T ] \to [ 0 , 1 ] } \mathbb { E } [ \mathcal { L } ( \pmb { \theta } _ { T } ) ] \qquad \mathrm { s . t . } \qquad \frac { 1 } { T } \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t \leq \alpha .\tag{2}
$$

The solution is unique up to changes on a set of measure zero, as established in the last paragraph of Appendix J.1. We therefore refer to this solution as the optimal data schedule. For additional theoretical settings, see Appendix H for more details.

## 3.2 OPTIMAL DATA SCHEDULES DEPEND ON THE LR SCHEDULE

In Section 2.2, placing all MT data in a final stage where only MT data is used gives the lowest MT loss among TS schedules under Const, whereas FDR favors earlier MT entry with PT replay. We now

recover this LR dependence in theory. Furthermore, under some power-law spectral assumptions (Lin et al., 2024; Luo et al., 2025; Yan et al., 2026), we give a closed form of the optimal data schedule under the FDR regime.

Theorem 3.1 (Informal). Under Condition 3.1, let $\eta _ { \mathrm { m a x } } > 0 .$ . The following statements hold:

(i) Const. $I f \eta ( t ) \equiv \eta _ { \mathrm { m a x } }$ , the optimal data schedule $\gamma ^ { * }$ is given by $\gamma ^ { * } ( t ) = \bar { \gamma } ( t ) : = \mathbb { 1 } _ { \{ t \geq ( 1 - \alpha ) T \} }$ namely the TS schedule with $\gamma _ { \mathrm { e n d } } = 1 $

(ii) Linear decay (FDR). $I f \eta ( t ) = \eta _ { \mathrm { m a x } } - c t / T ,$ , where $( 1 - 4 ^ { - 1 / 3 } ) \eta _ { \mathrm { m a x } } < c < \eta _ { \mathrm { m a x } } , 0 < \alpha <$ $1 - ( 1 - 4 ^ { - 1 / 3 } ) \eta _ { \mathrm { m a x } } / c .$ , then γ¯ is not optimal.

(iii) Three stage optimum under FDR. In the FDR regime, under an additional power-law spectral assumption and for sufficiently large $T ,$ the optimal data schedule $\gamma ^ { * }$ is given by:

$$
\gamma ^ { * } ( t ) = \left\{ { \begin{array} { l l } { 0 , } & { 0 \leq t < t _ { 1 } , } \\ { C \eta ( t ) ^ { - { \frac { 2 ( b - 1 ) } { b + 1 } } } , } & { t _ { 1 } \leq t < t _ { 2 } , } \\ { 1 , } & { t _ { 2 } \leq t \leq T , } \end{array} } \right.
$$

where $0 < t _ { 1 } < t _ { 2 } < T , C > 0$ is constant in time, and $b > 1$ is determined by the power-law spectral assumption.

Theorem 3.1 shows that different learning rate schedules induce different optimal data schedules, and the optimal data schedule under the FDR regime has a time range when both PT and MT data are used in training under the power-law spectral assumption. We then give some intuition on how this theorem works. The formal statements of our theoretical results are given in Appendix I, and proof are given in Appendices J to L.

Interpretation of Theorem 3.1. For a fixed LR schedule, only the contribution from features specific to MT depends on the data schedule, since PT and MT train the shared features in the same way. For these MT features, training may leave a nonzero loss even without gradient noise; we call this the bias term. Gradient noise introduces an additional contribution to the expected final loss, which we call the variance term. The linear SDE and quadratic loss allow us to separate these contributions exactly. Writing $\begin{array} { r } { R ( s ) = \int _ { s } ^ { T } \eta ( t ) \gamma ( t ) } \end{array}$ dt for the cumulative LR of MT updates after time s, Equation (8) gives

$$
\mathbb { E } [ \mathcal { L } ( \pmb { \theta } _ { T } ) ] = A + \underbrace { \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { ( 1 ) } \theta _ { i } ^ { ( 1 ) } ( 0 ) ^ { 2 } e ^ { - 2 \lambda _ { i } ^ { ( 1 ) } R ( 0 ) } } _ { \mathrm { b i a s t e r m } } + \underbrace { \frac { h } { 2 } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { ( 1 ) } \nu _ { i } ^ { ( 1 ) } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } e ^ { - 2 \lambda _ { i } ^ { ( 1 ) } R ( s ) } \mathrm { d } s } _ { \mathrm { v a r i a r c e t e r m } } ,
$$

where $A$ is the expected final loss from shared features, independent of the data schedule, and $\nu _ { i } ^ { ( 1 ) }$ is the gradient noise variance for the corresponding MT feature. The bias term decreases with $\dot { R } ( 0 )$ For noise introduced at time $s ,$ the factor $e ^ { - 2 \lambda _ { i } ^ { ( 1 ) } R ( s ) }$ describes how subsequent MT updates reduce its contribution to the final loss.

With a constant LR, using the full MT budget fixes $R ( 0 )$ , so all such schedules have the same bias term. Placing all MT data in a final pure MT stage maximizes $R ( s )$ at every time s and therefore minimizes the variance term. This explains the optimality of the abrupt switch in part (i).

Under LR decay, consider moving a small amount of MT data from a later time to an earlier one while keeping the MT budget fixed. The larger LR at the earlier time increases $R ( 0 )$ , reducing the bias term, and more strongly suppresses noise introduced before that time. However, removing the corresponding later MT updates leaves less MT training to suppress noise introduced between the two times, increasing its contribution to the variance term. Under the conditions of part (ii), moving some MT data earlier lowers the final loss, so the constant LR optimum is no longer optimal.

To determine the optimal data schedule, we compare the benefit of using the same amount of MT data at different times. If moving a small amount of MT data earlier or later lowers the final loss, the original allocation cannot be optimal. The optimal schedule must therefore balance the benefits of earlier and later MT updates so that no such reallocation can further reduce the loss. Under the assumptions of part (iii), solving for this balance gives the three-stage schedule in the theorem. During the middle stage, the optimal MT ratio satisfies $\gamma ^ { * } ( t ) = C \eta ( t ) ^ { \breve { - } 2 ( b - 1 ) / ( b + 1 ) }$ . Since $b > 1$ this ratio increases as the LR decreases: the optimal mixture contains progressively more MT data before the final pure MT stage. See Appendix L for the derivation.

## 3.3 LR SHOULD DECAY AFTER MT DATA ENTERS TRAINING

In Section 2.3, we observed that to minimize the final MT loss, the optimal learning rate decay should begin after the MT data enters training. Under additional conditions, we also recover this finding theoretically.

Setup. We use the SDE in Equation (1) and fix the data schedule to TS with a final MT ratio of 1:

$$
\gamma ( t ) = \mathbb { 1 } _ { \{ t \geq T - L \} } , \qquad L = \alpha T .
$$

Thus, training uses only PT data before $( 1 - \alpha ) T$ and only MT data during the remaining interval of length L. We consider stable-decay (SD) LR schedules, which keep the LR at M before decreasing it linearly to m over the final W units of time:

$$
\eta _ { W } ( t ) = \left\{ \begin{array} { l l } { M , } & { 0 \leq t < T - W , } \\ { m + \Delta ( T - t ) / W , } & { T - W \leq t \leq T , } \end{array} \right. \quad \quad 0 < W \leq T .
$$

Here $0 < m < M$ are fixed and $\Delta : = M - m$ . We include Const by defining $\eta _ { 0 } ( t ) = M ; W = L$ gives MDR and $W = T$ gives FDR. As in part (iii) of Theorem 3.1, we average the final MT loss over a power-law spectral model as well as training noise, and denote it by $J _ { T } ( W )$ . Our goal is to minimize $J _ { T } ( W )$ over $W \in [ 0 , T ]$

Next we state the main theorem, which shows that the optimal learning rate decay begins after the MT data enters training.

Theorem 3.2 (Informal). For the dynamics in Equation (1), consider TS and the SD LR family with

$$
\gamma ( t ) = \mathbb { 1 } _ { \{ t \geq ( 1 - \alpha ) T \} } , \qquad 0 < m < M .
$$

Under suitable power-law spectral conditions,for every sufficiently smallfixed $\alpha > 0$ and sufficiently large T, any optimal decay duration $W ^ { * }$ satisfies $0 < W ^ { * } < \alpha T$ . Thus, optimal LR decay begins strictly after MT entry.

See Appendix N for the formal statement (Theorem N.1) and proof.

Interpretation of Theorem 3.2. Unlike changing the data schedule, changing the LR affects both the shared and MT-specific contributions to the final loss. Under the power-law spectral model, the contribution of earlier gradient noise decreases as a power of the cumulative LR of subsequent updates. The exponents are $b _ { 0 }$ for shared features and $b _ { 1 }$ for MT-specific features; a smaller exponent means slower decay. Thus, in our analysis, we focus on the scenario $b _ { 0 } \geq b _ { 1 }$ , which describes a regime in which earlier noise in MT-specific features decreases no faster in the long run than in shared features. This captures a natural difficulty in MT adaptation: the new domain requires training specialized features, while the shared features already receive updates throughout PT.

Keeping the LR high after MT entry helps reduce the variance accumulated during PT. Under the training dynamics in Equation (1), gradient noise affects MT-specific features even during pure PT training. PT updates do not train these features, so only MT updates can reduce the contribution of this noise to the final variance term. The contribution of noise introduced during pure PT training to the variance term of the final MT loss is

$$
V _ { 1 } \frac { \displaystyle \int _ { 0 } ^ { T - L } \eta _ { W } ( s ) ^ { 2 } ~ \mathrm { d } s } { [ g _ { 1 } + R ( 0 ) ] ^ { b _ { 1 } } } ,
$$

where $R ( 0 )$ is the cumulative LR of MT updates defined above, and $V _ { 1 } , g _ { 1 } > 0$ are fixed constants from the spectral model (Equation (30)). Starting LR decay before MT entry can reduce the noise accumulated during PT, represented by the numerator, but also weakens the MT updates that suppress it, represented by the denominator. With a small MT budget, noise accumulates over a long PT stage and must be reduced within a short MT stage. Keeping the LR high for the initial MT updates increases $R ( 0 )$ , reducing this variance contribution as well as the bias term.

![](images/68a30a10728a72fdb746660299094321266313cc751eca97614dcbb61334d918.jpg)

![](images/6e22418a102c486653ec185b67ffa33eabdf0f9de6abbf6603ab759c642a992b.jpg)

![](images/664e8a93f5df45263794fab526fca290c2b18ba76e0c698159001f143b6c26ad.jpg)  
Figure 5: LR-dependent data-schedule preferences also occur in MoE. Left to right: 4B continuation tokens with a 25% MT budget; 10B tokens with 25% MT; and 10B tokens with 47.75% MT. All three settings use the same 2.99B-parameter model. At a final MT ratio of 100%, Const can outperform MDR. See Section 2.2 and Appendix F for the setup and full results.

A final LR decay reduces the variance term. Keeping the LR high until the end would continue to introduce noise with little time for later updates to reduce it. Under $b _ { 0 } \geq b _ { 1 } .$ , a sufficiently small MT budget, and a sufficiently long training horizon, the benefit of stronger initial MT updates outweighs the noise reduction from starting decay at or before MT entry. Decaying later still reduces the final variance compared with Const. Together, these effects place the optimal decay strictly inside the MT stage.

## 4 ADDITIONAL EXPERIMENTS

In this section, we present additional experiments following the discussion in Section 2.

## 4.1 JOINT SCHEDULING ALSO SHAPES THE PT–MT LOSS FRONTIER

In addition to MT loss, PT loss also matters in practice, and we usually need to balance the two. As discussed in Section 2.2 and shown in Figures 8 and 9, with FDR, both MT and PT loss decrease as the ending MT ratio decreases under both TS and ZL data schedules. However, in most cases where LR decay does not suppress learning from MT data, including Constant LR and MDR, PT loss usually increases as MT loss decreases. Therefore, we need to strike a balance between the PT and MT loss in these cases. From this perspective, we find that the ZL data schedule usually yields a narrower range of MT and PT losses across different ending MT ratios, whereas the TS data schedule exhibits much larger variations, especially in PT loss. Thus, in practice, using TS requires more careful tuning. In addition, given a TS or ZL data schedule, we find that the MDR regime dominates both Const and FDR regimes for dense models mostly (Figures 8 and 9). Moreover, under MDR, we find that TS dominates ZL in terms of both PT and MT loss (Figures 8, 9, 17 and 18). But under Constant LR, ZL exhibits advantages over TS for the MoE model, which may be attributed to the load-balancing benefits of ZL in the MoE setting discussed in the following section.

## 4.2 CONTINUOUS DATA SCHEDULES REDUCE LOAD IMBALANCE CAUSED BY DISTRIBUTION SHIFTS

Continuous data schedules suppress the load balancing loss spike. One additional training requirement for MoE models compared with dense models is that MoE must also balance the training tokens assigned to different experts. A common approach is to add an auxiliary load balancing term to the training loss, discouraging the router from sending most tokens to a few experts while leaving others underused (Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 2022). In our experiments, the abrupt data distribution shift under TS produces a sharp spike in the load balancing loss, indicating routing instability during the data shift. However, introducing MT data gradually through a continuous schedule such as ZL largely avoids this spike (Figure 6). For the 10B runs with a final MT ratio of 100%, ZL also ends with a slightly lower load balancing loss than TS under both Const and MDR. We also provide further ablations across training horizons and LR schedules in Appendix G (Figure 26).

![](images/8d8812055832b48b9a31e982b725de6fe7aca77a4cd90244f924c53ee02f5361.jpg)  
Figure 6: Gradual MT entry reduces router imbalance under data distribution shift. TS produces sharp spikes when the MT ratio jumps, while ZL avoids comparable spikes. Under Const (left), both schedules use the same LR trajectory, so this difference comes from the data schedule. MDR (right) shows a similar pattern, although LR decay starts at different times for TS and ZL. These MoE runs use 10B continuation tokens and a fixed 25% MT budget. Dotted lines mark MT entry. See Appendix G for setup and diagnostic details.

ZL also better preserves the rankings of expert loads across the data shift. Beyond the loss, we further rank experts by how many MT tokens they receive in held-out evaluations before MT entry time and at the end of training. Under ZL, experts that receive more tokens before the shift tend to remain more heavily used afterwards. Under TS, expert loads become more balanced after the switch, but some experts that received more tokens before the switch drop in the load ranking, while previously less used experts move up. Figure 30 compares both the final load balance and these ranking changes under TS+Const and ZL+MDR. This substantial reshuffling of expert loads may make subsequent training less stable. Appendix G gives the full comparison, with the ranking statistics reported in Table 14.

## 5 CONCLUSION

In this work, we show that data schedules and LR schedules should be designed together by combining experiments on dense and MoE models with a quadratic theoretical model. There are several promising directions for future work: 1) Our results provide an initial understanding of this coupling, but determining the form of jointly optimal schedules under fixed compute and MT token budgets remains theoretically challenging. 2) We also focus primarily on minimizing final MT loss. Future work should consider PT and MT losses jointly and potentially incorporate downstream task metrics when evaluating and choosing schedules.

## 6 ACKNOWLEDGEMENT

We thank Jianxun Zhou and Chenyi Dang for their early efforts on this project. We would like to thank Zixuan Wang, Kaiyue Wen, Huaqing Zhang, Zhixuan Pan, Binghui Li, and Zhenbo Sun for helpful discussions and suggestions. This work was supported by the Major Key Project of PCL under Grant PCL2025A12.

## AI USE STATEMENT

In this work, we used generative AI tools to assist with checking mathematical claims, theoretical derivations, and proofs, and with analyzing and interpreting experimental results. We did not use generative AI tools to design experiments or implement methods. Additionally, we used these tools for manuscript drafting and editing, figure preparation, literature searches, and LAT<sub>E</sub>X formatting. We reviewed all AI-assisted work, checking the text for accuracy, comparing experimental analyses with the original results, and examining theoretical derivations and proofs for correctness. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REFERENCES

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sebastien Bubeck, Ronen Eldan, et al. Phi-4 technical report. CoRR, abs/2412.08905, 2024. URL https://arxiv.org/abs/2412.08905.

Alon Albalak, Liangming Pan, Colin Raffel, and William Yang Wang. Efficient online data mixing for language model pre-training. CoRR, abs/2312.02406, 2023. URL https://arxiv.org/ abs/2312.02406.

Christina Baek, Ricardo Pio Monti, David Schwab, Amro Abbas, Rishabh Adiga, Cody Blakeney, Maximilian Böther, Paul Burstein, Aldo Gael Carranza, Alvin Deng, et al. The finetuner’s fallacy: When to pretrain with your finetuning data. arXiv preprint arXiv:2603.16177, 2026. URL https://arxiv.org/abs/2603.16177.

Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1):183–202, 2009. doi: 10.1137/080716542. URL https://doi.org/10.1137/080716542.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, et al. SmolLM2: When smol goes big – data-centric training of a small language model. CoRR, abs/2502.02737, 2025. URL https://arxiv.org/abs/2502.02737.

Mayee F. Chen, Michael Y. Hu, Nicholas Lourie, Kyunghyun Cho, and Christopher Ré. Aioli: A unified optimization framework for language model data mixing. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 71cc3acb2578bdd1e52bf9f85bbb1af0-Abstract-Conference.html.

Mayee F. Chen, Tyler Murray, David Heineman, Matt Jordan, Hannaneh Hajishirzi, Christopher Ré, Luca Soldaini, and Kyle Lo. Olmix: A framework for data mixing throughout LM development. CoRR, abs/2602.12237, 2026. URL https://arxiv.org/abs/2602.12237.

Rui Dai and Shuran Zheng. Explaining data mixing scaling laws, 2026. URL https://arxiv. org/abs/2606.08167.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, et al. Deepseek-v3 technical report. CoRR, abs/2412.19437, 2024. URL https://arxiv.org/abs/2412.19437.

Meng Ding, Kaiyi Ji, Di Wang, and Jinhui Xu. Understanding forgetting in continual learning with linear regression. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 10978–11001. PMLR, 2024. URL https://proceedings.mlr.press/v235/ding24c.html.

Aleksandr Dremov, Alexander Hägele, Atli Kosson, and Martin Jaggi. Training dynamics of the cooldown stage in warmup-stable-decay learning rate scheduler. Transactions on Machine Learning Research, 2025. URL https://arxiv.org/abs/2508.01483.

Itay Evron, Edward Moroshko, Rachel Ward, Nathan Srebro, and Daniel Soudry. How catastrophic can catastrophic forgetting be in linear regression? In Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings ofMachine Learning Research, pp. 4028–4079. PMLR, 2022. URL https://proceedings.mlr.press/v178/evron22a.html.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23(120):1–39, 2022. URL https://jmlr.org/papers/v23/21-0998.html.

Ce Ge, Zhijian Ma, Daoyuan Chen, Yaliang Li, and Bolin Ding. BiMix: A bivariate data mixing law for language model pretraining. CoRR, abs/2405.14908, 2024. URL https://arxiv.org/ abs/2405.14908.

GLM-4.5 Team. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. CoRR, abs/2508.06471, 2025. URL https://arxiv.org/abs/2508.06471.

Daniel Goldfarb and Paul Hand. Analysis of catastrophic forgetting for random orthogonal transformation tasks in the overparameterized regime. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pp. 2975–2993. PMLR, 2023. URL https://proceedings.mlr.press/v206/ goldfarb23a.html.

Daniel Goldfarb, Itay Evron, Nir Weinberger, Daniel Soudry, and Paul Hand. The joint effect of task similarity and overparameterization on catastrophic forgetting an analytical model. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 6cb00ce1a21a090a3dae04cebebd8341-Abstract-Conference.html.

Jacopo Graldi, Alessandro Breccia, Giulia Lanzillotta, Thomas Hofmann, and Lorenzo Noci. The importance of being lazy: Scaling limits of continual learning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 20326–20367. PMLR, 2025. URL https://proceedings.mlr.press/ v267/graldi25a.html.

Jiawei Gu, Zacc Yang, Chuanghao Ding, Rui Zhao, and Fei Tan. CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 16143–16162. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.903. URL https://aclanthology.org/2024.emnlp-main.903/.

Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro von Werra, and Martin Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-2427. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 8b970e15a89bf5d12542810df8eae8fc-Abstract-Conference.html.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. MiniCPM: Unveiling the potential of small language models with scalable training strategies. In First Conference on Language Modeling, 2024. URL https://arxiv.org/abs/2404.06395.

Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, and Irina Rish. Simple and scalable strategies to continually pretrain large language models. Transactions on Machine Learning Research, 2024. URL https: //arxiv.org/abs/2403.08763.

Stanisław Jastrz˛ebski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. Three factors influencing minima in sgd. arXiv preprint arXiv:1711.04623, 2017.

Peizhong Ju, Sen Lin, Mark S. Squillante, Yingbin Liang, and Ness B. Shroff. Generalization performance of transfer learning: Overparameterized and underparameterized regimes, 2023. URL https://arxiv.org/abs/2306.04901.

Hyunji Jung, Hanseul Cho, and Chulhee Yun. Convergence and implicit bias of gradient descent on continual linear classification. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 94ab02a30b0e4a692a42ccd0b4c55399-Abstract-Conference.html.

K2 Team, Zhengzhong Liu, Liping Tang, Linghao Jin, Haonan Li, Nikhil Ranjan, Desai Fan, Shaurya Rohatgi, Richard Fan, Omkar Pangarkar, Huijuan Wang, et al. K2-V2: A 360-open, reasoningenhanced LLM. arXiv preprint arXiv:2512.06201, 2025. URL https://arxiv.org/abs/ 2512.06201.

Suhas Kotha and Percy Liang. Replaying pre-training data improves fine-tuning. CoRR, abs/2603.04964, 2026. URL https://arxiv.org/abs/2603.04964.

Sebastian Lee, Sebastian Goldt, and Andrew Saxe. Continual learning in the teacher-student setup: Impact of task similarity. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 6109–6119. PMLR, 2021. URL https://proceedings.mlr.press/v139/lee21e.html.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb.

Binghui Li, Fengling Chen, Zixun Huang, Lean Wang, and Lei Wu. Functional scaling laws in kernel regression: Loss dynamics and learning rate schedules. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2509.19189.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, et al. DataComp-LM: In search of the next generation of training sets for language models. CoRR, abs/2406.11794, 2024. URL https://arxiv.org/abs/2406.11794.

Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and adaptive stochastic gradient algorithms. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pp. 2101–2110. PMLR, 2017. URL https://proceedings.mlr.press/v70/li17f.html.

Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. Journal of Machine Learning Research, 20(40): 1–47, 2019. URL https://jmlr.org/papers/v20/17-526.html.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, et al. StarCoder: May the source be with you! CoRR, abs/2305.06161, 2023. URL https://arxiv.org/ abs/2305.06161.

Zhiyuan Li, Kaifeng Lyu, and Sanjeev Arora. Reconciling modern deep learning with traditional optimization analyses: The intrinsic learning rate. In Advances in Neural Information Processing Systems, volume 33, pp. 14544–14555, 2020. URL https://proceedings.neurips.cc/ paper/2020/hash/a7453a5f026fb6831d68bdc9cb0edcae-Abstract.html.

Zhiyuan Li, Sadhika Malladi, and Sanjeev Arora. On the validity of modeling sgd with stochastic differential equations (sdes). Advances in Neural Information Processing Systems, 34:12712– 12725, 2021.

Zhiyuan Li, Tianhao Wang, and Sanjeev Arora. What happens after SGD reaches zero loss? a mathematical framework. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2110.06914.

Licong Lin, Jingfeng Wu, Sham M Kakade, Peter L Bartlett, and Jason D Lee. Scaling laws in linear regression: Compute, parameters, and data. Advances in Neural Information Processing Systems, 37:60556–60606, 2024.

Emmy Liu, Graham Neubig, and Chenyan Xiong. Midtraining bridges pretraining and posttraining distributions. CoRR, abs/2510.14865, 2025a. URL https://arxiv.org/abs/2510.14865.

Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. RegMix: Data mixture as regression for language model pre-training. In International Conference on Learning Representations, 2025b. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 5f67d864aae6115374fed7beddd119e0-Abstract-Conference.html.

LLM-Core Xiaomi. Mimo: Unlocking the reasoning potential of language model – from pretraining to posttraining. CoRR, abs/2505.07608, 2025. URL https://arxiv.org/abs/2505.07608.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017. URL https://arxiv.org/ abs/1608.03983.

Kairong Luo, Haodong Wen, Shengding Hu, Zhenbo Sun, Zhiyuan Liu, Maosong Sun, Kaifeng Lyu, and Wenguang Chen. A multi-power law for loss curve prediction across learning rate schedules. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 63e88a6ebfb1f68808a058a32b947e71-Abstract-Conference.html.

Kairong Luo, Zhenbo Sun, Haodong Wen, Xinyu Shi, Jiarui Cui, Chenyi Dang, Kaifeng Lyu, and Wenguang Chen. How learning rate decay wastes your best data in curriculum-based LLM pretraining. In International Conference on Learning Representations, 2026. URL https: //arxiv.org/abs/2511.18903.

Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora. On the sdes and scaling rules for adaptive gradient algorithms. Advances in Neural Information Processing Systems, 35: 7697–7711, 2022.

Neil Mallinar, Austin Zane, Spencer Frei, and Bin Yu. Minimum-norm interpolation under covariate shift. arXiv preprint arXiv:2404.00522, 2024.

Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A. Raffel. Scaling data-constrained language models. In Advances in Neural Information Processing Systems, volume 36, pp. 50358–50376, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, et al. OLMoE: Open mixture-of-experts language models. CoRR, abs/2409.02060, 2024. URL https://arxiv. org/abs/2409.02060.

Brendan O’Donoghue and Emmanuel Candès. Adaptive restart for accelerated gradient schemes. Foundations ofComputational Mathematics, 15:715–732, 2015. doi: 10.1007/s10208-013-9150-3.

Antonio Orvieto and Aurelien Lucchi. Continuous-time models for stochastic optimization algorithms. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://papers.nips.cc/paper\_files/paper/2019/hash/ 9cd78264cf2cd821ba651485c111a29a-Abstract.html.

Zhixuan Pan, Shaowen Wang, Liao Pengfei, and Jian Li. Understanding llm behaviors via compression: Data generation, knowledge acquisition and scaling laws. Advances in Neural Information Processing Systems, 38:168404–168454, 2026.

Scott Pesme, Loucas Pillaud-Vivien, and Nicolas Flammarion. Implicit bias of SGD for diagonal linear networks: A provable benefit of stochasticity. In Advances in Neural Information Processing Systems, volume 34, pp. 29218–29230, 2021. URL https://proceedings.neurips.cc paper/2021/hash/f4661398cb1a3abd3ffe58600bf11322-Abstract.html.

Haoran Que, Jiaheng Liu, Ge Zhang, Chenchen Zhang, Xingwei Qu, Yinghao Ma, Feiyu Duan, Zhiqi Bai, Jiakai Wang, Yuanxing Zhang, Xu Tan, Jie Fu, Jiamang Wang, Lin Qu, Wenbo Su, and Bo Zheng. D-cpt law: Domain-specific continual pre-training scaling law for large language models. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/a4628e9fbd3002a554923642f74d5d6b-Abstract-Conference.html.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024. URL https://arxiv.org/abs/2412.15115.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017. URL https://arxiv.org/ abs/1701.06538.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, et al. Megatron-LM: Training multi-billion parameter language models using model parallelism. CoRR, abs/1909.08053, 2019. URL https://arxiv.org/abs/1909.08053.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, et al. 2 olmo 2 furious. CoRR, abs/2501.00656, 2025. URL https://arxiv.org/abs/2501.00656.

Howe Tissue, Venus Wang, and Lu Wang. Scaling law with learning rate annealing. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-3044. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 830b1abc6d2da85f23d41169fa44d185-Abstract-Conference.html.

Jiapeng Wang, Changxin Tian, Kunlong Chen, Ziqi Liu, Jiaxin Mao, Wayne Xin Zhao, Zhiqiang Zhang, and Jun Zhou. MergeMix: Optimizing mid-training data mixtures via learnable model merging. CoRR, abs/2601.17858, 2026. URL https://arxiv.org/abs/2601.17858.

Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, and Daniel Dajun Zeng. Learning dynamics in continual pre-training for large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 64525–64548. PMLR, 2025. URL https://proceedings.mlr.press/ v267/wang25cx.html.

Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, and Tengyu Ma. Understanding warmup-stable-decay learning rates: A river valley loss landscape view. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 6a1fe80a9e2dcda0b3e5fd0fd87eb097-Abstract-Conference.html.

Jingfeng Wu, Difan Zou, Vladimir Braverman, Quanquan Gu, and Sham M. Kakade. The power and limitation of pretraining-finetuning for linear regression under covariate shift. In Advances in Neural Information Processing Systems, volume 35, pp. 33041–33053, 2022. doi: 10.52202/068431-2394. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ d5c04aa72b92c53bda5b525b60958295-Abstract-Conference.html.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing data mixtures speeds up language model pretraining. In Advances in Neural Information Processing Systems, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html.

Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, and Yang You. To repeat or not to repeat: Insights from scaling LLM under token-crisis. In Advances in Neural Information Processing Systems, volume 36, pp. 59304–59322, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html.

Tingkai Yan, Haodong Wen, Binghui Li, Kairong Luo, Wenguang Chen, and Kaifeng Lyu. Larger datasets can be repeated more: A theoretical analysis of multi-epoch scaling in linear regression. In International Conference on Learning Representations, 2026. URL https://arxiv.org/ abs/2511.13421.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. Qwen3 technical report. CoRR, abs/2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Kailai Yang, Xiao Liu, Lei Ji, Hao Li, Xiao Liang, Zhiwei Liu, Yeyun Gong, Peng Cheng, and Mao Yang. Data mixing agent: Learning to re-weight domains for continual pre-training. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, pp. 9447–9473. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.427. URL https://aclanthology.org/2026.acl-long.427/.

Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, and Xipeng Qiu. Data mixing laws: Optimizing data mixtures by predicting language modeling performance. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html.

Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, and Sham Kakade. How does critical batch size scale in pretraining? In International Conference on Learning Representations, pp. 66756–66782, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ a6f14f95d9c9443927638bcd5d917a7a-Abstract-Conference.html.

## CONTENTS

1 Introduction 1   
2 Data Schedules and Learning Rate Schedules Should Be Designed Together 3   
2.1 Experimental Setup 3   
2.2 Different LR Schedules Prefer Different Data Schedules 4   
2.3 Learning Rate Decay Should Begin Well After MT Data Enters Training . 5   
3 Theoretical Analysis 6   
3.1 Setup 6   
3.2 Optimal Data Schedules Depend on the LR Schedule 7   
3.3 LR Should Decay After MT Data Enters Training 9   
4 Additional Experiments 10   
4.1 Joint scheduling also shapes the PT–MT loss frontier 10   
4.2 Continuous data schedules reduce load imbalance caused by distribution shifts . . 10   
5 Conclusion 11   
6 Acknowledgement 11   
A Related Works 20   
B Training Setup and Schedule Definitions 21   
C Complete Data Schedule Results for the Dense Model 25   
D Learning Rate Decay Duration Scans 30   
E Data Stream Replicas and Noise Estimates 39   
F MoE Results and Additional Ablations 39   
G Router Dynamics and Interpretation 51   
H Additional Theoretical Setup 58   
I Formal Statements of Theoretical Results 58   
J Proof of Theorem I.1 60   
J.1 Derivation of the Terminal Loss 60   
J.2 KKT Condition for Equation (2) 61   
J.3 Proof of Theorem I.1 62   
K Proof of Theorem I.2 63   
L Proof of Theorem I.3 64   
L.1 Derivations of the Minimization Objective 64   
L.2 KKT Condition 65   
L.3 The Shape of the Data Schedule when M (t) is a Constant . 66   
L.4 Proof of Theorem I.3 67   
M Numerical Validation of the Theoretical Model 70   
M.1 Additional Details on the Simulation Experiments . 71   
M.2 Checks of the SDE Approximation . 72   
M.3 Numerical Validation of Theoretical Predictions 72   
M.4 How the Optimal Data Schedule Changes with the Learning Rate Schedule 72   
M.5 Verification of the KKT Condition of the Optimal Data Schedule 73   
N Optimal Learning Rate Decays after MT Data Enters 75   
N.1 Setup and Formal Statement 75   
N.2 Derivation of the Expected Validation Loss 75   
N.3 Auxiliary Lemmas 76   
N.4 Proof of Theorem N.1 81

## A RELATED WORKS

Midtraining and continual pretraining. Modern language model recipes increasingly include a later training phase that shifts the data distribution from broad pretraining corpora toward more curated or capability-focused data. Recent technical reports describe this pattern through late stage training on curated mixtures and multistage data curricula (Team OLMo et al., 2025; Abdin et al., 2024). In Qwen3, the reasoning stage both raises the proportions of STEM, coding, reasoning, and synthetic data and accelerates learning rate decay (Yang et al., 2025). This practice couples the data and learning rate schedules but does not establish whether that coupling is optimal. A closely related literature studies continual pretraining, where a model adapts to a new distribution while retaining performance on its source distribution. Existing work examines source data replay, learning rate rewarming and redecay, and mixture ratios for balancing adaptation and retention (Ibrahim et al., 2024; Que et al., 2024; Wang et al., 2025). Recent controlled studies make the timing question more explicit. Liu et al. (2025a) find that the introduction time and mixture weight of specialized data interact strongly in midtraining. Kotha & Liang (2026) show that source data replay can improve target data efficiency and that its benefit depends on how much target data is already present during pretraining. They also study LR cooldown and report that cosine decay can make uniform mixing outperform placing target data last. Baek et al. (2026) find that repeating domain data during pretraining can improve domain performance and preserve general capabilities after finetuning. A separate line of work studies how loss changes when a finite corpus is reused over multiple epochs (Muennighoff et al., 2023; Xue et al., 2023; Yan et al., 2026). These studies vary the amount of reuse rather than the temporal allocation of a fixed target data budget. Together, prior results establish that replay, timing, and reuse all matter. We build on these observations by comparing the same data schedules across LR schedules while fixing total training compute and the MT token budget.

Learning rate scheduling in LLM training. Cosine decay is widely used in language model pretraining, but it ties the decay horizon to a predetermined training length (Loshchilov & Hutter, 2017; Luo et al., 2025). Warmup Stable Decay separates a long stable phase from a final cooldown, which supports continued training and reusable decay branches (Hu et al., 2024; Hägele et al., 2024). A constant LR with exponential weight averaging can also match the optimization efficiency of cosine decay and Warmup Stable Decay, particularly at large batch sizes (Zhang et al., 2025). Several studies model how the full learning rate trajectory affects loss rather than treating the schedule as a fixed experimental detail. Schedule aware scaling laws quantify the effects of accumulated learning rate and annealing on loss curves (Tissue et al., 2025; Luo et al., 2025). Complementary analyses study the optimization dynamics of the stable and cooldown phases directly (Wen et al., 2025; Dremov et al., 2025). In continual pretraining, learning rate rewarming and redecay are often coupled with source data replay (Ibrahim et al., 2024), while Wang et al. (2025) jointly model distribution shift and learning rate annealing in the loss curve. Most directly related to our motivation, Luo et al. (2026) show that strong decay can reduce the benefit of high quality examples placed late in a curriculum ordered by data quality. Their remedies use a milder final decay or model averaging. We instead vary the timing of domain mixtures under a fixed target data budget and ask how the preferred data schedule changes across learning rate schedules.

Data mixture ratios and temporal scheduling. Prior work on data mixing primarily optimizes how much data to sample from each domain, whereas our question concerns how a fixed target data budget should be allocated over training time. Proxy based methods learn mixture weights through robust optimization or regression over smaller training runs (Xie et al., 2023; Liu et al., 2025b). Scaling law methods predict performance at unseen mixture ratios and data scales (Ge et al., 2024; Ye et al., 2025; Que et al., 2024; Gu et al., 2024). In particular, Gu et al. (2024) characterize the tradeoff between general and domain performance during continual pretraining, but treat the mixture ratio as a variable for the full training run rather than as a temporal schedule. Other methods allow the proportions to change during training. Albalak et al. (2023) use a bandit algorithm to adapt domain sampling probabilities online, while learned and adaptive frameworks optimize the ratios directly or use a learned policy (Chen et al., 2025; Yang et al., 2026). Recent frameworks address efficient mixture search and mixture reuse as the available domains change (Wang et al., 2026; Chen et al., 2026). These methods broaden how mixture proportions can be selected and adapted. Our comparisons keep both total training compute and the cumulative MT token budget fixed, then vary the LR schedule to test whether the preferred temporal allocation changes.

Theory of data scheduling under distribution shift. A complementary theoretical literature studies how prescribed task sequences affect transfer and forgetting. Analyses of overparameterized linear regression relate forgetting to task geometry, task order, and repeated exposure (Evron et al., 2022; Goldfarb & Hand, 2023; Goldfarb et al., 2024). Teacher student and feature learning models show that task similarity and representation dynamics shape interference (Lee et al., 2021; Graldi et al., 2025). Repeated gradient descent on linear classification tasks can converge to a joint solution under suitable conditions (Jung et al., 2025). Closer to our setting, Wu et al. (2022) derive target risk bounds when online SGD trains first on a source distribution and then on a target distribution under covariate shift. Ding et al. (2024) analyze noisy linear regression across multiple tasks and show that covariance spectra, task order, and step size jointly shape forgetting. These works prescribe the task sequence and then evaluate transfer, forgetting, or target risk. Our analysis instead treats the within training mixture as the decision variable, fixes its cumulative target data budget, and characterizes how the optimal temporal allocation changes with the learning rate schedule.

Comparison with Kotha & Liang (2026). A closely related work is Kotha & Liang (2026), which demonstrates the benefits of replaying pretraining data during mid-training and fine-tuning. They also investigate how pretraining data and target data should be scheduled. However, they focus on a two-stage training scenario consisting of pretraining followed by mid-training or fine-tuning, with a fixed mixing ratio between the two data sources within each stage. In contrast, we use a stepwise data schedule within MT, which allows the proportions of PT and MT data to vary over the training process. More importantly, while Kotha & Liang (2026) observe that the learning rate schedule can affect the effectiveness of data ordering, we systematically study the interaction between the learning rate schedule and the data schedule. Across a wide range of settings, we show that changing only the LR schedule can reverse which data schedule achieves lower final MT loss. Our results therefore show that the effectiveness of a data schedule cannot be considered independently of the LR schedule. Furthermore, we provide a simple guideline for jointly designing the LR schedule and the data schedule: LR decay should begin only after MT data enters training.

## B TRAINING SETUP AND SCHEDULE DEFINITIONS

This section describes the training configuration and explicitly defines the data and LR schedules used in Section 2.

Losses and plot conventions. PT loss denotes DCLM validation loss, and MT loss denotes validation loss on FineMath, or on StarCoderData Python in the code-domain ablation. In the data-schedule and LR-decay scans, curves connect primary runs, and rings mark their lowest tested losses. Hollow overlays show repeated runs using different subsets or orders of training examples (Appendix E). The data schedule comparison figures plot final PT loss horizontally and final MT loss vertically, with point labels giving the final MT ratio. Top panels separate TS and ZL; the remaining panels group the same results by LR rule.

Data and preprocessing. PT uses DCLM-Baseline (Li et al., 2024), and MT uses the finemath-4plus subset of FineMath released with SmolLM2 (Ben Allal et al., 2025). We tokenize the same source corpora for the dense model and the MoE model using their respective tokenizers. An EOS token is appended to each document, and best-fit packing produces fixed-length rows of 4,097 tokens, padded when necessary and shuffled within each shard. The first 4,096 tokens are inputs and the last 4,096 are prediction targets. Attention is causal across the packed row, without a document-level attention mask. The loss mask excludes positions whose input token equals EOS. The tokenized pools contain approximately 206B DCLM and 10.0B FineMath tokens for the dense model, and 212.6B DCLM and 9.75B FineMath tokens for the MoE model. DCLM data used for continued training comes from shards not consumed during PT, and both validation sets are held out from training.

Models and optimization. Tables 1 and 2 give the full configurations of the dense model and MoE model used in this paper. Both models are pretrained on DCLM before any continued training comparison. The dense model uses the layer dimensions of Qwen2.5-0.5B (Qwen et al., 2024), with 630M parameters under the specified untied embedding configuration. The MoE model scales an OLMoE configuration (Muennighoff et al., 2024) to 2.99B total parameters. Within a model family, all continued training runs start from the same final PT weights, with a freshly initialized optimizer and no LR warmup. The implementation for the dense model follows the original MindSpore optimizer convention of omitting Adam bias correction and applying weight decay to every parameter. MoE training uses Megatron-LM v0.15.3 (Shoeybi et al., 2019), whose default decay exclusions leave normalization weights unregularized. These conventions are held fixed within each family.

The final PT checkpoint has DCLM and FineMath validation losses of 2.8147 and 2.1472 for the dense model, and 2.6824 and 2.1625 for the MoE model.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Architecture</td><td></td></tr><tr><td>Reference architecture</td><td>Qwen2.5-0.5B</td></tr><tr><td>Total parameters</td><td>630M</td></tr><tr><td>Transformer layers</td><td>24</td></tr><tr><td>Hidden size</td><td>896</td></tr><tr><td>Attention heads (query / KV)</td><td>14 /2</td></tr><tr><td>Attention head dimension</td><td>64</td></tr><tr><td>Q/K/V biases</td><td>Enabled</td></tr><tr><td>FFN activation / width</td><td>SwiGLU / 4,864</td></tr><tr><td>Normalization</td><td>RMSNorm,  $\epsilon = 1 0 ^ { - 6 }$ </td></tr><tr><td>Positional encoding</td><td>RoPE, base 10,000</td></tr><tr><td>Vocabulary size</td><td>151,936</td></tr><tr><td>Input/output embeddings</td><td>Untied</td></tr><tr><td>Tokenizer</td><td>Qwen2.5 151643 / 151656</td></tr><tr><td>EOS / pad token IDs</td><td></td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W } , \beta = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ </td></tr><tr><td>Adam bias correction</td><td>Disabled</td></tr><tr><td>Weight decay</td><td>0.1 on all parameters</td></tr><tr><td>Gradient-norm clipping</td><td>1.0</td></tr><tr><td>Precision</td><td>BF16 autocast; FP32 master weights and optimizer states</td></tr><tr><td>Weight initialization</td><td> $\mathcal { N } ( 0 , 0 . 0 1 ^ { 2 } )$ </td></tr><tr><td>Bias / norm initialization</td><td>Zero / one</td></tr><tr><td>Training budgets and learning rates</td><td></td></tr><tr><td>Sequence length</td><td>4,096 tokens</td></tr><tr><td>Global batch</td><td>512 sequences (2,097,152 tokens/step)</td></tr><tr><td>PT budget</td><td>95,368 steps; approximately 200B DCLM tokens</td></tr><tr><td>PT learning rate</td><td>5,000-step linear warmup to  $5 \times 1 0 ^ { - 4 }$  , then constant</td></tr><tr><td>Continuation budget</td><td>10,000 steps; 20.97152B tokens</td></tr><tr><td>MT fraction</td><td>10%, 25%, or 47.75%</td></tr><tr><td>MT token budget</td><td> $2 . 1 0 \mathbf { B } , 5 . 2 4 \mathbf { B } .$  or 10.01B, respectively</td></tr><tr><td>Continuation peak / final LR</td><td> $5 \times 1 0 ^ { - 4 } / 1 0 ^ { - 5 }$ </td></tr></table>

Table 1: Architecture and training settings for the 630M dense model. The final LR applies to schedules with decay; Const retains the peak LR.

Fixed data budgets. Let $T$ be the number of continued training steps, α the fraction of its token budget allocated to MT, and $\gamma _ { \mathrm { e n d } }$ the final MT ratio, as in the main text. Write $\gamma ( t )$ for the MT fraction, with $\begin{array} { r } { \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t = \alpha \dot { T } } \end{array}$ . For Two-Stage (TS),

$$
\gamma _ { \mathrm { T S } } ( t ) = \left\{ \begin{array} { l l } { 0 , } & { 0 \leq t < t _ { 0 } , } \\ { \gamma _ { \mathrm { e n d } } , } & { t _ { 0 } \leq t \leq T , } \end{array} \right. \quad \quad \frac { t _ { 0 } } { T } = 1 - \frac { \alpha } { \gamma _ { \mathrm { e n d } } } , \quad \alpha \leq \gamma _ { \mathrm { e n d } } \leq 1 .
$$

Zero-then-Linear (ZL) linearly increases the MT fraction after an initial PT-only interval:

$$
\gamma _ { \mathrm { Z L } } ( t ) = \left\{ \begin{array} { l l } { 0 , } & { 0 \leq t < t _ { 0 } , } \\ { \gamma _ { \mathrm { e n d } } \frac { t - t _ { 0 } } { T - t _ { 0 } } , } & { t _ { 0 } \leq t \leq T , } \end{array} \right. \quad \frac { t _ { 0 } } { T } = 1 - \frac { 2 \alpha } { \gamma _ { \mathrm { e n d } } } .
$$

ZL is feasible when $\alpha \leq \gamma _ { \mathrm { e n d } } / 2$ , equivalently 2α $\leq \gamma _ { \mathrm { e n d } } \leq 1$ . At a shared $( \alpha , \gamma _ { \mathrm { e n d } } )$ , its MT data window $L : = T - t _ { 0 }$ is twice the TS window. Changing $\gamma _ { \mathrm { e n d } }$ therefore changes both the final mixture and the time when MT data first enter. At the respective minimum feasible final MT ratio, $t _ { 0 } = 0$ and the initial PT-only interval disappears.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Architecture</td><td></td></tr><tr><td>Reference architecture</td><td>OLMoE</td></tr><tr><td>Total parameters</td><td>2.99B</td></tr><tr><td>Active non-embedding parameters</td><td>454M</td></tr><tr><td>Input/output embedding parameters</td><td>155M combined</td></tr><tr><td>Transformer layers</td><td>12</td></tr><tr><td>Hidden size</td><td>1,536</td></tr><tr><td>Attention heads</td><td>12, multi-head attention</td></tr><tr><td>Attention head dimension</td><td>128</td></tr><tr><td>Linear biases</td><td>Disabled</td></tr><tr><td>Experts per layer / active per token</td><td>64 / 8; dropless, no shared experts</td></tr><tr><td>Expert FFN activation / width</td><td>SwiGLU / 768</td></tr><tr><td>Normalization</td><td>RMSNorm and QK-norm</td></tr><tr><td>Positional encoding</td><td>RoPE, base 10,000</td></tr><tr><td>Vocabulary size</td><td>50,304 (padded)</td></tr><tr><td>Input/output embeddings</td><td>Untied</td></tr><tr><td>Tokenizer</td><td>OLMoE-1B-7B-0924</td></tr><tr><td>EOS / pad token IDs</td><td>50279 / 1</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W } , \beta = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ </td></tr><tr><td>Weight decay</td><td>0.1, excluding one-dimensional parameters</td></tr><tr><td>Gradient-norm clipping</td><td>1.0</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>Weight initialization</td><td> $\mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } )$ </td></tr><tr><td>Router</td><td>Softmax in FP32; balancing coefficient 0.01; z-loss coefficient</td></tr><tr><td>Implementation</td><td>0.001 Megatron-LM v0.15.3; local transformer, FlashAttention, Py- Torch norms, sequential experts</td></tr><tr><td>Training budgets and learning rates</td><td></td></tr><tr><td>Sequence length</td><td>4,096 tokens</td></tr><tr><td>Global batch</td><td>1,024 sequences (4,194,304 tokens/step)</td></tr><tr><td>PT budget</td><td>9,536 steps; approximately 40B DCLM tokens</td></tr><tr><td>PT learning rate</td><td>238-step warmup to  $4 \times 1 0 ^ { - 4 }$  , then constant</td></tr><tr><td>Continuation budget</td><td>954 steps (4B tokens) or 2,384 steps (10B tokens)</td></tr><tr><td>MT fraction</td><td>25% in the base configuration</td></tr><tr><td>Continuation peak / final LR</td><td> $4 \times 1 0 ^ { - 4 } / 8 \times 1 0 ^ { - 6 }$ </td></tr></table>

Table 2: Architecture and training settings for the 2.99B-parameter MoE model. Unlike the original OLMoE recipe, PT retains a constant LR after warmup, and normalization weights have no weight decay. The final LR applies only to schedules with decay.

To determine each batch’s composition, we calculate the total number of PT sequences prescribed by the schedule up to that batch and round it up. The increase from the previous batch gives the PT count for the current batch, with the remaining sequences drawn from MT data. Sequences are assigned to workers in a fixed repeating order. This makes batch composition deterministic, avoid random fluctuations in the mixing ratio, and allows the allocation to be reconstructed from the current training step. Reported switch times and LR decay durations use the actual integer step counts.

Learning rate windows. For a positive decay length W, LR stays at $\eta _ { \mathrm { m a x } }$ until $T - W$ and then decreases linearly to $\eta _ { \mathrm { m i n } } = \eta _ { \mathrm { m a x } } / 5 0$ :

$$
\eta _ { W } ( t ) = \left\{ \begin{array} { l l } { \eta _ { \operatorname* { m a x } } , } & { 0 \leq t < T - W , } \\ { \eta _ { \operatorname* { m i n } } + ( \eta _ { \operatorname* { m a x } } - \eta _ { \operatorname* { m i n } } ) \frac { T - t } { W } , } & { T - W \leq t \leq T . } \end{array} \right.
$$

Const keeps $\eta _ { \mathrm { m a x } }$ throughout; it is plotted at zero window length and does not decay to η<sub>min</sub>. Full-Decay-Range (FDR) uses $W = T$ . Matched-Decay-Range (MDR) uses $W = L { \dot { = } } T - { t _ { 0 } }$ beginning decay when MT data first enter. The scans vary $r : = \breve { W } / L$ with the data schedule fixed. The denominator is the entire interval from first MT exposure to training end, including the low-MTdensity part of a ZL ramp; it is neither T nor the MT budget expressed in training steps, αT. A 20% window means $W = 0 . { \dot { 2 } } L$ , and a 120% window begins decay before MT exposure. When $t _ { 0 } = 0$ MDR and FDR coincide and use one run.

![](images/e6f345e5387b2601f4eb848cef8f6fa958a23dc5a6811576d2787b4adf4e5a38.jpg)  
Figure 7: Illustration of different LR schedules and data schedules. Rows show TS and ZL, and columns show Const, FDR, and MDR. Solid lines and shading show the MT ratio on the left axes. Dash dotted lines show LR relative to its peak on the right axes. Both data schedules use a 25% MT budget and end at 100% MT, so ZL introduces MT earlier. MDR therefore begins LR decay earlier for ZL than for TS. Time starts at the shared PT checkpoint.

Validation and comparability. All final PT/MT loss comparisons use the average cross entropy per token on the validation set, evaluated at the final checkpoint. Final evaluation for the dense model uses the full held-out sets, approximately 560M DCLM and 160M FineMath tokens; intermediate evaluations use 32M tokens every 500 steps. MoE final evaluation uses a separate skip-training pass on each domain, with $3 2 \times 1 0 2 4 \times 4 0 9 6 = 1 3 4 , 2 1 7 , 7 2 8$ tokens per domain. Because the two families use different tokenizers and training implementations, their absolute loss values are not directly comparable. Cross-family comparisons concern schedule ordering and the shape of the response to LR changes.

## C COMPLETE DATA SCHEDULE RESULTS FOR THE DENSE MODEL

Table 3 lists all tested final MT ratios, each crossed with Const, FDR, and MDR, with identical boundary schedules run only once. The continued training token budget remains 20.97152B across all three MT fractions. The largest fraction is 47.75%, rather than 50%, to use approximately one pass through FineMath: 2,444,800 of 2,445,076 packed rows. At this fraction, TS requires $\gamma _ { \mathrm { e n d } } \geq 4 7 . 7 5 \%$ and ZL requires $\gamma _ { \mathrm { e n d } } \geq 9 5 . 5 \%$ . For ZL, we test five final MT ratios across its narrow feasible range, from 95.5% to 100%.

<table><tr><td>MT budget</td><td>TS final MT ratios (%)</td><td>ZL final MT ratios (%)</td></tr><tr><td>10%</td><td>10, 20, 40, 80, 100</td><td>20, 40, 60, 80, 100</td></tr><tr><td>25%</td><td>25, 40, 50, 70, 75, 85, 100</td><td>50, 60, 70, 80, 90, 100</td></tr><tr><td>47.75%</td><td>48, 50, 60, 70, 80, 90, 95, 100</td><td>95.5, 97, 98, 99, 100</td></tr></table>

Table 3: Complete data schedule runs for the dense model. Total and MT token budgets are fixed within each group.

Figures 8 to 10 show the final PT and MT losses for all tested configurations. Table 4 gives the best MT loss in each sampled data/LR family. For TS, the best tested final MT ratio under Const and MDR is 100% at every budget, whereas FDR selects 20%, 40%, and 70% as the MT budget increases. The FDR loss range contracts from approximately 0.068 to 0.031 to 0.007. Lowering γ<sub>end</sub> does not always improve the FDR loss: at 10%, loss rises from 1.6108 at $\gamma _ { \mathrm { e n d } } = 2 0 \%$ to 1.6197 at $\gamma _ { \mathrm { e n d } } = 1 0 \%$ ; at 25%, it rises from 1.5024 at $\gamma _ { \mathrm { e n d } } = 4 0 \%$ to 1.5097 at $\gamma _ { \mathrm { e n d } } = 2 5 \%$ . At 47.75%, the FDR minimum at $\gamma _ { \mathrm { e n d } } = 7 0 \%$ is only 0.0016 below the loss at $\gamma _ { \mathrm { e n d } } = 6 0 \%$ , so the broad flattening is more informative than a precise preferred endpoint.
<table><tr><td>MT budget</td><td>Data schedule</td><td>Const</td><td>FDR</td><td>MDR</td></tr><tr><td>10%</td><td>TS</td><td>1.5420 (100)</td><td>1.6108 (20)</td><td>1.5394 (100)</td></tr><tr><td></td><td>ZL</td><td>1.5485 (100)</td><td>1.6144 (20)</td><td>1.5682 (100)</td></tr><tr><td>25%</td><td>TS</td><td>1.4717 (100)</td><td>1.5024 (40)</td><td>1.4475 (100)</td></tr><tr><td></td><td>ZL</td><td>1.4810 (100)</td><td>1.5064 (50)</td><td>1.4802 (100)</td></tr><tr><td>47.75%</td><td>TS</td><td>1.4217 (100)</td><td>1.4194 (70)</td><td>1.3826 (100)</td></tr><tr><td></td><td>ZL</td><td>1.4335 (100)</td><td>1.4246 (97)</td><td>1.4221 (100)</td></tr></table>

Table 4: Lowest final MT loss among the tested final MT ratios for each MT budget, data schedule, and LR schedule. Parentheses give the final MT ratio $\gamma _ { \mathrm { e n d } }$ in percent.

TS versus ZL depends on the LR rule and final ratio. Table 5 compares the shapes at the same final MT ratio of 100%. Here ZL gives lower observed MT loss under FDR, while TS gives lower loss under MDR and Const. This differs from selecting each shape’s best final ratio: TS then gives lower FDR loss at all three MT budgets for the dense model (Table 4). Selecting each shape’s best final ratio also favors TS under FDR in all three main MoE settings (Table 11 and Fig. 20). The points labeled 100% in Figures 8 and 9 show the first comparison, while each curve’s minimum gives the second. Under MDR, changing from TS to ZL also starts LR decay earlier, so this comparison does not isolate data shape at a fixed LR trajectory. MT gains can also come at a PT cost. At the 25% budget under Const, TS has lower MT loss than ZL but substantially higher PT loss.

To quantify the LR applied to MT data, define $\begin{array} { r } { E = \int _ { 0 } ^ { T } \eta ( t ) \gamma ( t ) } \end{array}$ dt for an LR trajectory $\eta ( t )$ . This quantity weights each MT token by the LR at which it is used. For TS and ZL with the same budget α and final MT ratio $\gamma _ { \mathrm { e n d } } .$ , let $u = \alpha / \gamma _ { \mathrm { e n d } }$ and $\rho = \eta _ { \mathrm { m i n } } / \eta _ { \mathrm { m a x } }$ . Integrating the schedules in Section 2.1 gives

$$
\frac { E _ { \mathrm { Z L } } } { E _ { \mathrm { T S } } } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { C o n s t } , } \\ { \displaystyle \frac { \rho + \frac { 2 } { 3 } ( 1 - \rho ) u } { \rho + \frac { 1 } { 2 } ( 1 - \rho ) u } , } & { \mathrm { F D R } , } \\ { \displaystyle \frac { 2 ( 1 + 2 \rho ) } { 3 ( 1 + \rho ) } , } & { \mathrm { M D R } . } \end{array} \right.
$$

Both models use $\rho = 0 . 0 2 .$ , giving an MDR ratio of approximately 0.68 and FDR ratios of approximately 1.24 to 1.31 across settings where TS and ZL have the same MT budget and final MT ratio. These are continuous time calculations, with small differences possible from discrete step rounding. They describe the LR assigned to MT tokens, but omit the effects of later updates and optimization noise and do not by themselves predict final loss. In particular, equal exposure under Const does not imply equal loss, and ZL can have higher FDR loss despite greater exposure, as in the dense model at a 10% MT budget with $\gamma _ { \mathrm { e n d } } = 2 0 \%$ . The role of LR in learning from late MT data is consistent with Luo et al. (2026), who find that strong LR decay can reduce the benefit of valuable examples encountered late in training.

<table><tr><td>MT budget</td><td>Const</td><td>FDR</td><td>MDR</td></tr><tr><td>10%</td><td>+0.0065</td><td>-0.0185</td><td>+0.0288</td></tr><tr><td>25%</td><td>+0.0093</td><td>-0.0138</td><td>+0.0327</td></tr><tr><td>47.75%</td><td>+0.0118</td><td>-0.0006</td><td>+0.0395</td></tr></table>

Table 5: MT loss difference in the dense model, ZL minus TS, with both schedules ending at 100% MT. Negative values favor ZL. Differences use the reported final losses and are not significance estimates.

![](images/9bbc81590239605994411b496bf439277816f5edb5742eb55fd97d8afe3866c7.jpg)

![](images/d8fa113dcdcc656ec881eaf3983c987a22ce3b31518a04ce450172076d4ebdd2.jpg)

![](images/e78d3caa6b6d51f4cebc9674530c30d7c564ac099012c40b7567766f22ffe01f.jpg)

![](images/cdcf90f50a7a144fbe616ec30a7c1c4fe83accf8c4ba87c33cb6bd7e8300e56c.jpg)

![](images/896e72873e28ed969adae0e5c6f4421108d34245ac089ae219d8631fc61c497e.jpg)  
Figure 8: FDR favors earlier MT mixing, while Const and MDR favor ending with MT data alone. For both TS and ZL, the lowest tested MT loss occurs at a final MT ratio of 20% under FDR and 100% under Const and MDR. These dense model runs use a 10% MT budget over 20.97B continued training tokens, including 2.10B FineMath tokens.

![](images/99f4c6b95f5bc2aab66a23e8f41a851fc832393f9091e6211edfb21a1f346b06.jpg)

![](images/c2e282eb8bc5ffabb4bbbd69bad2d72fd517f6a3e95075538546c1c9608e9a85.jpg)

![](images/fa396652658877a57f61ce6c2c8e33d1b508489f78bad2078efccfd225ec655b.jpg)

![](images/ac2bdf73f182b96b4658a1f07d03a1c46f71d456a0c4c74245f43267921e2786.jpg)

![](images/934d55b6648c7400a6fa989cbb1cdcad713c27782bf9b409bb883b25cf6c78d9.jpg)  
Figure 9: Changing the LR schedule changes the preferred final MT ratio. FDR gives the lowest tested MT loss at 40% MT for TS and 50% for ZL; Const and MDR favor 100% for both. These dense model runs use a 25% MT budget over 20.97B continued training tokens, including 5.24B FineMath tokens.

![](images/7a02fd5a58f54fbdc7e941444e8e9c568df7ee8bab657060dee8d77c55411c57.jpg)

![](images/56a4a87c6b1bcf3f72b35b2fa2a787d79213b257a9ef3fb13e41b92e65782cec.jpg)

![](images/29652cb79bfe04f09b33ba301439ef7b43d7b694c77b5b3fec088868fbfb48da.jpg)

![](images/c9aefa9d4d85f832c232a7b371023e360cd537e684ad6ff80118d3f03329a893.jpg)

![](images/37da3e30a2fd959976303f0bc7aa7f7e59eca2c18a2407fb26b9db954fd1b28c.jpg)  
Figure 10: ZL losses under FDR change little across its tested final MT ratios. They vary by only 0.001, while Const and MDR favor a final MT ratio of 100%. TS covers eight final MT ratios; ZL covers five from 95.5% to 100%. These dense model runs use a 47.75% MT budget over 20.97B continued training tokens, including 10.01B FineMath tokens. Hollow overlays show selected data sampling repeats.

## D LEARNING RATE DECAY DURATION SCANS

Following Section 2.3, we keep each data schedule fixed and vary when LR decay begins. The LR remains at the same peak until decay begins, then decreases linearly to the same final value at training completion. The relative LR decay duration expresses this decay duration as a percentage of the MT data window, from MT entry time to training completion. For example, a relative duration of 20% means that LR decay covers the final 20% of this interval. Const corresponds to 0%, and MDR to 100%. Values above 100% start LR decay before MT entry time.

Table 6 lists the tested relative durations for all 19 scans, and Table 7 summarizes their lowest observed losses. Each scan includes Const and MDR as baselines. We also test FDR, which decays the LR throughout continued training, except for StarCoder ZL.

A delayed LR decay gives the lowest tested loss in all nineteen scans. For the dense model, the best tested relative durations are 40% for TS and 20% for ZL. For the MoE model, they are 20% for TS and 10% for ZL. They improve on both Const and MDR, with reductions in MT loss relative to MDR ranging from 0.0094 to 0.0660. These results support keeping the LR high after MT entry time and delaying its decay, as discussed in Section 2.3. The preferred relative durations refer to the tested values. In particular, the scans from the two earlier MoE PT checkpoints test only 20% and 40% in addition to Const, MDR, and FDR, so they do not locate the optimum more precisely.

<table><tr><td>MT budget Schedule</td><td></td><td>γend</td><td>MT window (steps)</td><td>Additional relative LR decay durations (%)</td></tr><tr><td colspan="5">Dense: 200B-token PT; 20.97B-token continuation; FineMath</td></tr><tr><td>10%</td><td>TS</td><td>100%</td><td>1,000</td><td>20,40, 50, 80, 120</td></tr><tr><td></td><td>TS</td><td>80%</td><td>1,250</td><td>20, 40, 80, 90, 110, 120</td></tr><tr><td></td><td>ZL</td><td>100%</td><td>2,000</td><td>5, 10, 20, 40, 80, 90, 110, 120</td></tr><tr><td>25%</td><td>TS</td><td>100%</td><td>2,500</td><td>20, 40, 80, 120</td></tr><tr><td>47.75%</td><td>ZL</td><td>100%</td><td>5,000</td><td>5, 10, 20, 40, 80</td></tr><tr><td></td><td>TS</td><td>100%</td><td>4,775</td><td>20, 40, 80, 120</td></tr><tr><td></td><td>TS</td><td>80%</td><td>5,969</td><td>20, 40, 80, 120</td></tr><tr><td></td><td>TS</td><td>60%</td><td>7,958</td><td>20,40,80</td></tr><tr><td></td><td>ZL</td><td>100%</td><td>9,550</td><td>5, 10, 20, 40, 80</td></tr><tr><td colspan="5">MoE: 40B-token PT; 4B-token continuation; FineMath</td></tr><tr><td>25%</td><td>TS</td><td>100%</td><td>238</td><td>5, 10, 20, 40, 80, 120</td></tr><tr><td></td><td>ZL</td><td>100%</td><td>477</td><td>5, 10, 20, 40,80</td></tr><tr><td colspan="5">MoE: 40B-token PT; 10B-token continuation; FineMath</td></tr><tr><td>25%</td><td>TS</td><td>100%</td><td>596</td><td>5,10, 20,40,80,120</td></tr><tr><td></td><td>ZL</td><td>100%</td><td>1,192</td><td>5, 10, 20, 40, 80</td></tr><tr><td>47.75%</td><td>TS</td><td>100%</td><td>1,138</td><td>5, 10, 20, 40, 80, 120</td></tr><tr><td></td><td>ZL</td><td>100%</td><td>2,277</td><td>5, 10, 20, 40, 80</td></tr><tr><td colspan="5">MoE: 20B-token PT; 4B-token continuation; FineMath</td></tr><tr><td>25% MoE: 30B-token PT; 4B-token continuation; FineMath</td><td>TS</td><td>100%</td><td>238</td><td>20,40</td></tr><tr><td colspan="5">25%</td></tr><tr><td></td><td>TS</td><td>100%</td><td>238 20,40</td><td></td></tr><tr><td colspan="5">MoE: 40B-token PT; 10B-token continuation; StarCoderData Python</td></tr><tr><td>25%</td><td>TS</td><td>100%</td><td></td><td>596 5, 10, 20, 40, 80, 120</td></tr><tr><td></td><td>ZL</td><td>100%</td><td>1,192</td><td>10, 20, 40,80</td></tr></table>

Table 6: LR decay scan settings. Relative LR decay durations are percentages of each schedule’s MT data window. Const (0%) and MDR (100%) are tested in every scan; FDR is also tested except for StarCoderData ZL.

Table 8 reports every sampled MT loss for four further MoE configurations: TS and ZL over 10B continuation tokens with a 47.75% MT budget, and TS over 4B continuation tokens with a 25% MT budget starting from 20B-token or 30B-token PT checkpoints. The best tested relative LR decay durations remain 20% for TS and 10% for ZL. The two earlier PT checkpoints confirm that delaying

<table><tr><td colspan="5">Best relative LR</td><td colspan="2">Gain over MDR</td></tr><tr><td>MT budget</td><td>Schedule, γend Dense: 200B-token PT; 20.97B-token continuation; FineMath</td><td>decay duration</td><td>Best loss</td><td>Const</td><td>MDR</td><td></td></tr><tr><td colspan="9"></td></tr><tr><td>10%</td><td>TS, 100% TS, 80%</td><td>40% 40%</td><td>1.5156 1.5215</td><td>1.5420 1.5529</td><td>1.5394 1.5454</td><td>0.0238 0.0239</td></tr><tr><td></td><td>ZL, 100%</td><td>20%</td><td>1.5203</td><td>1.5485</td><td>1.5682</td><td>0.0479</td></tr><tr><td>25%</td><td>TS, 100%</td><td>40%</td><td>1.4275</td><td>1.4717</td><td>1.4475</td><td>0.0200</td></tr><tr><td>47.75%</td><td>ZL, 100%</td><td>20%</td><td>1.4392</td><td>1.4810</td><td>1.4802</td><td>0.0410</td></tr><tr><td></td><td>TS, 100%</td><td>40%</td><td>1.3690</td><td>1.4217</td><td>1.3826</td><td>0.0136</td></tr><tr><td></td><td>TS, 80%</td><td>40%</td><td>1.3831</td><td>1.4413</td><td>1.3959</td><td>0.0128</td></tr><tr><td></td><td>TS, 60%</td><td>40%</td><td>1.4039</td><td>1.4672</td><td>1.4133</td><td>0.0094</td></tr><tr><td></td><td>ZL, 100%</td><td>20%</td><td>1.3857</td><td>1.4335</td><td>1.4221</td><td>0.0364</td></tr><tr><td colspan="5">MoE: 40B-token PT; 4B-token continuation; FineMath</td><td></td><td></td></tr><tr><td>25% ZL, 100%</td><td>TS, 100%</td><td>20%</td><td>1.6080</td><td>1.6176</td><td>1.6486</td><td>0.0406</td></tr><tr><td></td><td></td><td>10%</td><td>1.6000</td><td>1.6131</td><td>1.6586</td><td>0.0586</td></tr><tr><td colspan="5">MoE: 40B-token PT; 10B-token continuation; FineMath</td><td></td><td></td></tr><tr><td>25%</td><td>TS, 100%</td><td>20%</td><td>1.4853</td><td>1.5110</td><td>1.5225</td><td>0.0372</td></tr><tr><td>47.75%</td><td>ZL, 100%</td><td>10%</td><td>1.4872</td><td>1.5101</td><td>1.5508</td><td>0.0636</td></tr><tr><td></td><td>TS, 100%</td><td>20%</td><td>1.4007</td><td>1.4292</td><td>1.4390</td><td>0.0383</td></tr><tr><td></td><td>ZL, 100%</td><td>10%</td><td>1.4110</td><td>1.4424</td><td>1.4770</td><td>0.0660</td></tr><tr><td colspan="5">MoE: 20B-token PT; 4B-token continuation; FineMath 25%</td><td></td><td></td></tr><tr><td>MoE: 30B-token PT; 4B-token continuation; FineMath</td><td>TS, 100%</td><td>20%</td><td>1.6562</td><td>1.6666</td><td>1.6969</td><td>0.0407</td></tr><tr><td colspan="5">25% TS, 100% 20% 1.6260</td><td>1.6409 1.6673</td><td>0.0413</td></tr><tr><td>MoE: 40B-token PT; 10B-token continuation; StarCoderData Python 25%</td><td>TS, 100%</td><td>20%</td><td>0.9317</td><td>0.9468</td><td></td><td></td></tr><tr><td colspan="5"></td><td>0.9594</td><td>0.0277</td></tr><tr><td></td><td>ZL, 100%</td><td>10%</td><td>0.9332</td><td>0.9476</td><td>0.9762</td><td>0.0430</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: Final MT losses in all 19 scans. Best loss and its relative LR decay duration refer to the lowest-loss primary run among the tested schedules. Gain over MDR is MDR minus best loss, calculated from the displayed values. Group headings specify the model, PT budget, continuation budget, and MT dataset.

LR decay improves on Const and MDR, although testing only two additional durations leaves the shape of the loss curve near its minimum uncertain.
<table><tr><td rowspan="2">Relative LR decay</td><td colspan="2">MoE: 10B continuation tokens 47.75%MT</td><td colspan="2">MoE: 4B continuation tokens 25% MT, TS</td></tr><tr><td>TS</td><td>ZL</td><td>PT 20B</td><td>PT 30B</td></tr><tr><td>Const (0%)</td><td>1.4292</td><td>1.4424</td><td>1.6666</td><td>1.6409</td></tr><tr><td>5%</td><td>1.4091</td><td>1.4131</td><td></td><td></td></tr><tr><td>10%</td><td>1.4046</td><td>1.4110</td><td></td><td></td></tr><tr><td>20%</td><td>1.4007</td><td>1.4137</td><td>1.6562</td><td>1.6260</td></tr><tr><td>40%</td><td>1.4040</td><td>1.4268</td><td>1.6578</td><td>1.6303</td></tr><tr><td>80%</td><td>1.4252</td><td>1.4609</td><td></td><td></td></tr><tr><td>MDR (100%)</td><td>1.4390</td><td>1.4770</td><td>1.6969</td><td>1.6673</td></tr><tr><td>120%</td><td>1.4534</td><td></td><td></td><td></td></tr><tr><td>FDR</td><td>1.4994</td><td>1.4806</td><td>1.7899</td><td>1.7552</td></tr></table>

Table 8: Final MT losses at every tested relative LR decay duration for the four additional scans, including the reference LR schedules. All data schedules end at 100% MT. The 10B-token continuations start from the 40B-token PT checkpoint, while the last two columns use earlier checkpoints from the same PT trajectory. Bold entries mark the lowest tested loss in each column. Dashes indicate durations that were not tested.

Figures 11 to 16 show the LR trajectories alongside the final MT loss curves. In each paired scan, the left panel shows the fixed data schedule and candidate LR trajectories. The right panel plots final MT loss against decay duration in steps; point labels give the duration as a percentage of that schedule’s MT data window. Lines connect evaluated configurations. These figures cover the twelve primary scans, including the complete LR/data and loss panels for TS at a 47.75% MT budget in the dense model, as highlighted in the main text. For the dense model, the additional TS scan at a 10% MT budget and a final MT ratio of 100% appears in Figure 3. Its MT data window lasts 1,000 steps, and the lowest tested loss is 1.5156 at a relative LR decay duration of 40%. The 120% relative duration, outside the displayed range, gives 1.5512, while FDR gives 1.6790. Table 8 supplies the complete sampled losses for the four further configurations. Their LR trajectories and loss curves are shown in Figures 21 and 23 in Appendix F. The two StarCoder scans are compared with their FineMath counterparts in Figure 25.

![](images/7aa374a903e5c8882222794530631c0637bdeb0bfef092e0a18d79a9829dbf6d.jpg)

![](images/c40763cd47815f661d3e83d8e4635b896184c52afe0f40511788d82e6730302c.jpg)

![](images/b324c89e4b1f066c9d54fddaa81be652bde51cf6a1aafeea4cd091ac26bb7535.jpg)

![](images/64253f58e4ec20dc40a423a6f559d0cbece2b2ac95d3cfd3f21928a069f03ebd.jpg)  
Figure 11: A short final LR decay improves on Const and MDR. The lowest tested MT losses are 1.5215 at a relative LR decay duration of 40% for TS (top) and 1.5203 at 20% for ZL (bottom). These dense model runs use a 10% MT budget. TS ends at 80% MT with a 1,250-step MT window; ZL ends at 100% MT with a 2,000-step window. Hollow TS markers show repeats at 20%; the 40% minimum was not repeated.

![](images/5bde9b8c8e43ed1c0bf5e1cced7f13b7c006de937fcf8faeff11fdbc8d5d8121.jpg)

![](images/71a7f4ae1fb7bebff5cdf8a169f8a6f16b0eecf52d9270ed8696425847693e18.jpg)

![](images/02936350039f0b4a7e4d45612f7b21b1d147e0b361546de62b7eda8a1c748f9f.jpg)

![](images/5b8f22cffe4ddb199e8da824709343d65c31a9dbbb246c9618f2763fc2edbacf.jpg)  
Figure 12: TS and ZL favor different relative LR decay durations. TS (top) reaches its lowest tested loss of 1.4275 at 40%, and ZL (bottom) reaches 1.4392 at 20%. Nearby short decay durations give similar losses. Both dense model runs use a 25% MT budget and end at 100% MT, with MT windows of 2,500 steps for TS and 5,000 for ZL. These configurations have no repeated runs.

![](images/d53a9ad9e0c3149f0eacdb7c216e7f64d076270ae577b5a1811ff414c984a9ad.jpg)

![](images/b6cab654edb4c3ee8cb67e74a8a1372e0fb493cf0edde086910270adf9253fd4.jpg)

![](images/5484a080f5afc3f82a706d2e75f693ae7c8d8fc85dca4e937533af16d9859b48.jpg)

![](images/9a54d52a147ce06e7961de79d2835859bca3ef782a98a3b64ed94a4c0746ff80.jpg)  
Figure 13: The best tested relative LR decay duration remains 40% when TS ends at lower MT ratios. At a final MT ratio of 80% (top), this gives a loss of 1.3831, with repeats at 1.3834 and 1.3846. At 60% MT (bottom), it gives 1.4039, with similar losses at relative LR decay durations between 20% and 80%. These dense model runs use a 47.75% MT budget and MT windows of 5,969 and 7,958 steps, respectively. The scan ending at 100% MT appears in Figure 3.

![](images/9f5569deb5934b74492cbab48a3dc8a2bc048af2f646acd10d98936241aacfe7.jpg)

![](images/682ffd38daf5c87e2bce7de048ab456465241cfaf0f5cfed8abb4d43dacdd536.jpg)

![](images/11e0cfeb2f0273983654c3cf5cd3c27fc229e61436acba072c08318a88650079.jpg)

![](images/daf742599c36a37bcd87fe79c89e750839e922806a1db5afe6be6c55dd8cf7b9.jpg)  
Figure 14: TS favors a relative LR decay duration of 40%, while ZL favors 20%. Their lowest tested losses are 1.3690 for TS (top) and 1.3857 for ZL (bottom). For ZL, MDR and FDR differ by only 450 decay steps and yield similar losses of 1.4221 and 1.4249, respectively. Both dense model runs use a 47.75% MT budget and end at 100% MT, with MT windows of 4,775 steps for TS and 9,550 for ZL. This ZL configuration has no repeated runs.

![](images/e1568977c101c5e7bc58361448aa334f970bc4852f81b97670e06f8ce4428f11.jpg)

![](images/db6faa07af73366faf03e3186e6db166ababa84ea086cbfc96151ac73d3e5dcb.jpg)

![](images/0fab675df5c2cc2c3f110f0d47cc7ac0d8c2948534752d87804de9b7c8280b48.jpg)

![](images/89314fef363dd5a273b72c2a4a9576eb5096883390db40102e4e7c068e043138.jpg)  
Figure 15: The best tested TS and ZL decays span nearly the same number of training steps. TS (top) reaches 1.6080 at a relative LR decay duration of 20% (47 steps), while ZL (bottom) reaches 1.6000 at 10% (48 steps). These MoE runs use 4B continued training tokens, a 25% MT budget, and a final MT ratio of 100%. Their MT windows last 238 and 477 steps, respectively. Hollow TS markers repeat MDR; neither minimum was repeated.

![](images/0668720fec31fe05bb108b619ab27431f93c9449953ca2b1a6f181ab5d6c998e.jpg)

![](images/937f8b05aa64fa8005418f130c9c37e8c53a2b4a73786e22044faa643ac9642f.jpg)

![](images/f3a16eda13870df46f6fc881e8faa8cdf33074e5ebd2f228f7f2db28f873b79e.jpg)

![](images/fd2697cee788e08e94943de699cb85a54e3a7f5f179caa6962a869d607b2e693.jpg)  
Figure 16: The best tested TS and ZL decays both last 119 steps. TS (top) reaches 1.4853 at a relative LR decay duration of 20%, while ZL (bottom) reaches 1.4872 at 10%. These MoE runs use 10B continued training tokens, a 25% MT budget, and a final MT ratio of 100%. Their MT windows last 596 and 1,192 steps, respectively. Hollow TS markers repeat MDR; the best TS and ZL configurations have no matched data repetitions.

## E DATA STREAM REPLICAS AND NOISE ESTIMATES

For six selected configurations, we repeat training twice using different training examples or a different order of the same examples (Table 9). All three runs start from the same PT checkpoint and use the same training setup. For the dense model, the two repeats begin one third and two thirds of the way through each dataset, respectively, returning to the beginning when they reach the end. At the 47.75% MT budget, each run uses almost the entire FineMath dataset, so the runs mainly differ in the order of MT examples. Each repeat shares about 36% of its DCLM examples with the original run. For the MoE model, the three runs instead use separate subsets of both PT and MT data, with no overlap between runs.

<table><tr><td colspan="6">Relative LR decay</td></tr><tr><td>MT budget</td><td>γend</td><td>duration or rule</td><td>Primary</td><td>Repeat 1</td><td>Repeat 2</td></tr><tr><td colspan="6">Dense: 200B-token PT; 20.97B-token continuation</td></tr><tr><td>10%</td><td>80% 20%</td><td></td><td>1.5227</td><td>1.5213 1.5212</td><td>0.000706</td></tr><tr><td>47.75%</td><td>100%</td><td>MDR</td><td>1.3826</td><td>1.3828</td><td>1.3830 0.000149</td></tr><tr><td>47.75%</td><td>70%</td><td>FDR</td><td>1.4194</td><td>1.4217</td><td>1.4198 0.001008</td></tr><tr><td>47.75%</td><td>80% 40%</td><td></td><td>1.3831</td><td>1.3834</td><td>1.3846 0.000620</td></tr><tr><td colspan="6">MoE: 40B-token PT; 4B-token continuation</td></tr><tr><td>25%</td><td>100% MDR</td><td>1.6486</td><td>1.6531</td><td>1.6497</td><td>0.001916</td></tr><tr><td colspan="6">MoE: 40B-token PT; 10B-token continuation</td></tr><tr><td>25%</td><td>100% MDR</td><td>1.5225</td><td>1.5231</td><td>1.5221</td><td>0.000421</td></tr></table>

Table 9: Final MT losses for the six repeated TS configurations, all using FineMath. The descriptive dispersion $\begin{array} { r } { s _ { 3 } = \sqrt { \frac { 1 } { 3 } \sum _ { j = 1 } ^ { 3 } ( \ell _ { j } - \bar { \ell } ) ^ { 2 } } } \end{array}$ is computed from the original losses before rounding; its denominator is three, rather than the denominator-two sample standard deviation.

Repeated runs are shown as hollow points. We use only the original runs to fit the curves and identify the best tested configuration. These repeats show how the loss changes when we change the training examples or their order at the six selected configurations. All runs start from the same PT checkpoint, so they do not capture variation across independently pretrained models. The repeats for the dense model also share many training examples. Repeating an MDR run does not tell us how much the loss would vary at other relative LR decay durations or with other data schedules. We therefore do not treat the small loss differences between nearby relative LR decay durations for ZL at a 25% MT budget for the dense model, or between the best TS and ZL MoE settings with a 10B-token continuation, as evidence that one setting reliably outperforms another.

## F MOE RESULTS AND ADDITIONAL ABLATIONS

Our main MoE experiments compare approximately 4B and 10B tokens of continued training, with MT data accounting for 25% in both settings. These runs last 954 and 2,384 steps and use approximately 1.0B and 2.5B MT tokens, respectively, corresponding to 244,224 and 610,304 packed FineMath sequences. The 10B-token continuation uses approximately 25.6% of the FineMath dataset. In both settings, continued training draws DCLM data from shards not consumed during PT. Table 10 lists all tested final MT ratios and the training step at which MT data first enters for each schedule. Figure 5 also shows results for 10B tokens of continued training with a 47.75% MT budget.

LR dependent preferences persist, but MDR is not uniformly better than Const. Figures 17 and 18 show the final PT and MT losses for all tested configurations. At both horizons, TS+Const favors $\gamma _ { \mathrm { e n d } } = 1 0 0 \%$ , whereas TS+FDR favors $\gamma _ { \mathrm { e n d } } = 2 5 \%$ . At 4B, the best TS+MDR loss is worse than the best TS+Const loss, and MDR’s own minimum occurs at $\gamma _ { \mathrm { e n d } } = 5 0 \%$ . This does not assert pointwise dominance: at the $\gamma _ { \mathrm { e n d } } = 2 5 \%$ boundary, MDR and FDR coincide. At 10B, MDR is nearly flat for the non-boundary endpoints: its loss is 1.5225 at $\gamma _ { \mathrm { e n d } } = 1 0 0 \%$ , 1.5236 at 75%, and 1.5245 at 50%. The MDR-minus-Const difference at $\gamma _ { \mathrm { e n d } } = 1 0 0 \%$ shrinks from +0.0310 at 4B to +0.0115 at 10B; at $\gamma _ { \mathrm { e n d } } = 5 0 \%$ in the 10B experiments, MDR is 0.0014 below Const. The approximately 63% reduction of the gap at 100% MT is consistent with sensitivity to adaptation time, but extending the horizon increases total PT and MT tokens together and does not isolate a single mechanism.

<table><tr><td colspan="2"></td><td colspan="2">First MT step</td></tr><tr><td>Data schedule</td><td>Final MT ratio</td><td>4B-token continuation</td><td>First MT step 10B-token continuation</td></tr><tr><td>TS</td><td>100%</td><td>716</td><td>1,788</td></tr><tr><td rowspan="5"></td><td>75%</td><td>636</td><td>1,589</td></tr><tr><td>50%</td><td>477</td><td>1,192</td></tr><tr><td>40%</td><td>358</td><td>894</td></tr><tr><td>25%</td><td>0</td><td>0</td></tr><tr><td>100%</td><td>477</td><td>1,192</td></tr><tr><td rowspan="3">ZL</td><td>75%</td><td>318</td><td>795</td></tr><tr><td>50%</td><td>0</td><td>0</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 10: Tested data schedules for the MoE model at a 25% MT budget. The first MT step denotes the TS switch or ZL ramp start. The TS $\gamma _ { \mathrm { e n d } } = 1 0 0 \%$ data windows are 238 and 596 steps.
<table><tr><td>Continuation tokens</td><td>Data schedule</td><td>Const</td><td>FDR</td><td>MDR</td></tr><tr><td>4B</td><td>TS</td><td>1.6176 (100)</td><td>1.6349 (25)</td><td>1.6343 (50)</td></tr><tr><td></td><td>ZL</td><td>1.6131 (100)</td><td>1.6548 (50)</td><td>1.6548 (50)</td></tr><tr><td>10B</td><td>TS</td><td>1.5110 (100)</td><td>1.5343 (25)</td><td>1.5225 (100)</td></tr><tr><td></td><td>ZL</td><td>1.5101 (100)</td><td>1.5548 (50)</td><td>1.5508 (100)</td></tr></table>

Table 11: Lowest final MT loss among the tested final MT ratios for each training duration, data schedule, and LR schedule in the MoE model at a 25% MT budget. Parentheses give the final MT ratio $\gamma _ { \mathrm { e n d } }$ in percent. Both MDR minima after 4B continued training tokens occur at $\gamma _ { \mathrm { e n d } } = 5 0 \%$ , so MDR does not select $\gamma _ { \mathrm { e n d } } = 1 0 0 \%$ at both continued training token budgets.

TS versus ZL. At matched endpoints, ZL improves MT loss under FDR and worsens it under MDR in both horizons (Table 12). Const gives smaller differences, with the 4B advantage of ZL largely disappearing at 10B. After separately selecting each shape’s best scanned window, ZL gives 1.6000 versus TS’s 1.6080 at 4B, whereas TS gives 1.4853 versus ZL’s 1.4872 at 10B. This ordering change prevents a general claim that either shape is preferable for MoE training. The scans appear in Appendix D; Figure 19 compares the TS horizon response directly.
<table><tr><td colspan="4">Continuation tokens</td></tr><tr><td>and LR rule</td><td> $\gamma _ { \mathrm { e n d } } = 1 0 0 \%$ </td><td> $\gamma _ { \mathrm { e n d } } = 7 5 \%$ </td><td> $\gamma _ { \mathrm { e n d } } = 5 0 \%$ </td></tr><tr><td>4B Const</td><td>-0.0045</td><td>-0.0029</td><td>-0.0028</td></tr><tr><td>4B FDR</td><td>-0.0298</td><td>-0.0282</td><td>-0.0233</td></tr><tr><td>4B MDR</td><td>+0.0101</td><td>+0.0155</td><td>+0.0206</td></tr><tr><td>10B Const</td><td>-0.0010</td><td>+0.0010</td><td>+0.0015</td></tr><tr><td>10B FDR</td><td>-0.0264</td><td>-0.0251</td><td>-0.0183</td></tr><tr><td>10B MDR</td><td>+0.0283</td><td>+0.0282</td><td>+0.0303</td></tr></table>

Table 12: MoE MT loss difference, ZL minus TS, at matched final MT ratios. Differences are computed from unrounded losses and reported to four decimal places. Negative values favor ZL. Each comparison uses one primary run per schedule, without significance testing.

![](images/576fa6bdc2af3ed9ac2cc895eae6b0196d295a2b455d70d223226f49c9d55f9a.jpg)

![](images/9ab9ff3ebc11cb11cbaa9242c27fbd404953ee38a74ab242c58390cad7594897.jpg)

![](images/af204131fb7ff5b4d556c07fff2a3b472a741f7c646da3ef1230b7f36f26dba0.jpg)

![](images/54d1f021019659e94dd64b9b290eb67a6089c724a46fb647943a03b4cfc0049b.jpg)

![](images/9ffc37ffee0e6cbdf3d5bd17f10fb48f344bbf431987be80435655294ad0be2f.jpg)  
Figure 17: Const gives the lowest tested MT loss among the three baseline LR schedules. MDR favors a final MT ratio of 50% for both TS and ZL. These MoE runs use 4B continued training tokens with a 25% MT budget. Hollow overlays repeat TS+MDR at a final MT ratio of 100%.

![](images/d42a16f8db315678a1bba30cdee29a7525a0d64462151bfbc319082c841d3ced.jpg)

![](images/e5dae713c05d9ba20ffe7f8dc2fc8027f057356c4877ad87bec9cd1be66fbab6.jpg)

![](images/dc2df0f51ba063c627a8a0ace75646129300890e6ba9b0ba9ceb300892361aec.jpg)

![](images/1d2ae6598da19e21e6ec4545546d347578d1f463eec97b02a4d0d23d3928a3a3.jpg)

![](images/91a3594da707f814d38a69c37ddf65e040287f8a0f0020cb8579898c6c28a18e.jpg)  
Figure 18: Longer MoE continued training narrows the TS loss gap between MDR and Const. At a final MT ratio of 100%, TS+MDR is closer to TS+Const than after 4B tokens, while ZL+MDR remains worse than ZL+Const. These runs use 10B continued training tokens with a 25% MT budget. Hollow overlays repeat TS+MDR at 100% MT.

![](images/09d68a4dbff31ae03fdd3abc4ffea07bfee088577a0d757f2e29291802b34e04.jpg)

![](images/5549eb0f345ad313e35deb2f0439b89b9eab3bc5195510929517a9c004bdd23f.jpg)

![](images/dd854fb13fef660e665067b9b70dddb1a6c410a56423d03cf951fa52519fb41a.jpg)

![](images/6d36dc42e6b58b1dac27b20a6f05cad705a7b4546216e82cb4d5a78722fa984b.jpg)

![](images/c530a65fcde664e323c90e8e95ca78c240c25099d9f3dba148adce497092b178.jpg)  
Figure 19: Longer MoE continued training narrows the MDR–Const loss gap. For TS, extending training from 4B to 10B tokens reduces the primary-run gap from 0.0310 to 0.0115 at a final MT ratio of 100%, while the best tested relative LR decay duration remains 20%. The upper panels show final MT loss and its change from each horizon’s Const baseline; the lower panels vary the final MT ratio. All runs use a 25% MT budget. In the difference panel, circles and local error bars at MDR show the three-run mean and $\pm s _ { 3 }$ , computed from unrounded losses relative to the primary Const loss; they do not estimate uncertainty at other durations or in the Const baseline.

A larger MT budget at the 10B horizon. We increase the MT budget from 25% to 47.75% while retaining the 40B PT checkpoint and the 10B continuation horizon of 2,384 steps. This setting uses approximately 4.77B FineMath tokens. The optimizer, data sources, batch size, and peak and final LRs are unchanged. TS is tested at final MT ratios of 100%, 90%, 80%, 60%, and 47.75%. ZL is tested at $\gamma _ { \mathrm { e n d } } = 1 0 0 \%$ . The group contains 28 runs across the basic LR rules and the additional decay windows. $\mathrm { A t \gamma _ { e n d } = 1 0 0 \% }$ , the TS switch occurs at step 1,246 and the ZL ramp starts at step 107, giving MT data windows of 1,138 and 2,277 steps, respectively.

The ranking reversal phenomenon between Const and FDR persists at this larger budget (Figure 20). Const favors a final MT ratio of 100%, with MT loss 1.4292, while FDR favors the lowest tested final MT ratio, 47.75%, with loss 1.4528. The range of FDR losses across TS endpoints decreases from 0.094 at the 25% budget to 0.047 at the 47.75% budget. $\mathrm { A t \gamma _ { e n d } = 1 0 0 \% }$ , MDR gives 1.4390, remaining 0.0098 above Const despite the longer MT data window. At this endpoint, ZL improves on TS under FDR by 0.0188, but has higher MT loss under Const and MDR by 0.0132 and 0.0380, respectively. Thus, increasing the MT budget preserves the dependence on the LR rule without making either MDR or a smooth data transition uniformly preferable.

A short final decay improves both data shapes (Figure 21). The best tested TS window covers 20% of its MT data window and reaches 1.4007, while the best tested ZL window covers 10% and reaches 1.4110. Both selected windows span 228 steps because ZL has an MT data window approximately twice as long. Relative to their own Const baselines, these losses are lower by 0.0285 and 0.0314, respectively. Table 8 reports all sampled values, and Table 7 places these minima alongside the other sweeps. No data stream replicas were run for this budget, so the comparisons describe the observed runs rather than uncertainty across repetitions.

![](images/071229ffb0af96ce832d6b68e3b44d14d8e95aa6445eda8f43279e69d0f2fa34.jpg)

![](images/a7850978d19527f03996980971278bdd35c024b7ddae843f0b45339bbb164db5.jpg)

![](images/78ebe43f9f4f4cba09c5f91940693e0edf0c1664f5cafbe66ea9efbb1240a3fe.jpg)

![](images/ce5a1d72ac9351270fc2929d914241200686985ac12858e1f1442e2315da1c3f.jpg)

![](images/b92c1acfabf28cc979f1c4d0237c1962240eb1f2d0b0d659904440bb42590187.jpg)  
Figure 20: Const and FDR favor opposite TS endpoints at a larger MoE MT budget. Const favors a final MT ratio of 100%, while FDR favors 47.75%. At 100% MT, ZL improves on TS under FDR but has higher MT loss under Const and MDR. These runs use 10B continued training tokens with a 47.75% MT budget, starting from the 40B-token PT checkpoint. ZL is tested only at 100% MT. Each point is one run.

![](images/7ae36b957e9b26f0dae733bf24172666522ce972c132533869a66532ac1556f4.jpg)

![](images/09857bb7fd95738f2e48666c4ae525e81649f5e4db9d09308636a1d0e881c86d.jpg)

![](images/b64f29ebd1404345eb25da191c3f18755c42be918f150575275893c5d6827402.jpg)

![](images/26b86a86904441eb08a96d185252e415c1f3d343901b7df9f61e26703042928e.jpg)  
Figure 21: Increasing the MoE MT budget preserves the best tested relative LR decay durations. TS (top) favors 20%, and ZL (bottom) favors 10%; both correspond to 228 decay steps. These runs use 10B continued training tokens with a 47.75% MT budget and end at 100% MT. Their MT windows last 1,138 and 2,277 steps, respectively. These configurations have no data stream repeats.

Earlier checkpoints from the same PT trajectory. We repeat a subset of the 4B-token MoE continuations from checkpoints saved after 4,800 and 7,200 PT steps, corresponding to approximately 20.1B and 30.2B tokens. We call these the 20B and 30B starting points and compare them with the main 40B checkpoint at step 9,536. All three checkpoints come from the same PT run. Each continuation uses 954 steps, a fixed 25% MT budget, and the same training configuration as the main 4B-token MoE continuations, including optimizer reinitialization and no LR warmup. At each earlier checkpoint, we test nine TS runs: Const and FDR at final MT ratios of 100%, 50%, and 25%, plus MDR and 20% and 40% decay windows at a final MT ratio of 100%. At a final MT ratio of 25%, mixing begins at step zero, so MDR and FDR share the same run. These groups do not include ZL runs or data stream replicas.

When we compare the same final MT ratios, changing the PT checkpoint preserves the ranking reversal phenomenon (Figure 22). At all three checkpoints, Const favors a final MT ratio of 100%, while FDR favors 25%. Longer PT lowers the final MT loss at each shared configuration. For example, Const at a final MT ratio of 100% gives 1.6666, 1.6409, and 1.6176 after 20B, 30B, and 40B PT tokens, respectively. The corresponding FDR losses exceed Const by 0.1233, 0.1143, and 0.1162. At a final MT ratio of 25%, FDR instead improves on Const by 0.0072, 0.0049, and 0.0021. These comparisons show that the ranking reversal phenomenon is present across the sampled checkpoints, without requiring that the model first completes the full 40B PT run.

The preference for a short final decay also persists (Figures 23 and 24). At a final MT ratio of 100%, MT data enters at step 716 and occupies the last 238 steps. The lowest tested MT loss occurs at a 20% window for all three checkpoints, improving on Const by 0.0104, 0.0149, and 0.0096 for the 20B, 30B, and 40B starting points, respectively. MDR instead increases MT loss by 0.0303, 0.0264, and 0.0310. The earlier checkpoints test only the 20% and 40% windows in addition to Const, MDR, and FDR. They therefore support the same ordering among these choices, but do not locate a continuous optimum or establish the shape of the curve near its minimum. Full values for the earlier checkpoints appear in Table 8.

![](images/d3aabf8a095f1d80de9aa09174cf2c1073d6a778027061abe8957657aafc6ee9.jpg)

![](images/8b3bda0ca0431645c9e431cea10f4521da731b626a889e1b0d66d7b1c84ec6a2.jpg)  
Figure 22: Const and FDR favor opposite TS endpoints even from earlier PT checkpoints. After either 20B PT tokens (left) or 30B (right), Const gives its lowest tested MT loss when TS ends at 100% MT, while FDR favors 25%. Both use 4B continued training tokens and a 25% MT budget. The checkpoints come from the same PT run.

Changing the MT dataset. StarCoderData is a collection of source code from GitHub released for training StarCoder (Li et al., 2023). It is derived from The Stack, with additional cleaning, removal of near duplicates, and filtering of personal information. We use its Python subset as MT data to test whether the preferred relative LR decay durations persist when the target domain changes from mathematics to code.

We replace FineMath with StarCoderData Python, starting from the same 40B PT checkpoint and continuing training for 10B tokens with a 25% MT budget. Both TS and ZL end at a final MT ratio of 100%. PT data, tokenizer, batch size, and LR settings remain unchanged. These two scans add

![](images/85b4ec025dd1049a131d3a82f261e19ac6066172498d52161c4d83fb23d278bd.jpg)

![](images/86f1c21f4381ccd14af4cdc6ad233a5c84383a9e9f0ee7fe8009af76d1e0aced.jpg)

![](images/c874a8b571ff9bc912b83584167b3d48a68af8f5580d00344e89ac338d1227e2.jpg)

![](images/845fa8fda2ec307a4ac233df6464aeb148071031106d6ad25cdf0a93c1cd4235.jpg)  
Figure 23: Delaying LR decay improves on Const and MDR from both earlier PT checkpoints. The lowest tested MT losses are 1.6562 after 20B PT tokens (top) and 1.6260 after 30B (bottom), both at a relative LR decay duration of 20% (47 steps). Each MoE TS run uses 4B continued training tokens, a 25% MT budget, and a final MT ratio of 100%, giving a 238-step MT window. Only 20% and 40% are tested beyond Const, MDR, and FDR.

![](images/5c903b26c942a54745b1b0f49bf81734ad72573a28db3c7f5ca1cc28b55f8ed0.jpg)

![](images/e507bff57614e4f83cf31aa52e4d99787fe4bf250b694e153e1adb19241d63ea.jpg)  
Figure 24: Changing the PT checkpoint preserves the preferred data and LR schedules among the shared configurations. Const and FDR favor opposite TS endpoints (left). A relative LR decay duration of 20% gives the lowest tested loss, while MDR remains worse than Const (right). The right panel shows loss changes relative to each checkpoint’s own Const baseline at a final MT ratio of 100%. These MoE runs start from 20B, 30B, and 40B checkpoints of the same PT trajectory and use 4B continued training tokens with a 25% MT budget. The 40B checkpoint has additional sampled durations.

15 runs. Both include Const and MDR, while FDR is tested only for TS. ZL does not include a 5% window, and neither scan includes data sampling replicas.

Changing MT data from mathematics to code preserves the best tested relative LR decay duration. Figure 25 compares each scan with its FineMath counterpart after subtracting its own Const loss. For both datasets, the lowest tested loss occurs at 20% for TS and 10% for ZL, corresponding to the same 119 steps of LR decay. The improvement over Const differs between datasets, but the best relative durations among those tested remain unchanged. MT loss is evaluated on each dataset separately, so absolute losses should not be compared across the two datasets.

![](images/c13a7a71102586f6facb5a5c14941882d39957e4e68526ab149bd8d1ee1965f4.jpg)

![](images/80ea8f8bc5f263e3b270af5a1a15aeec1e44018badfeac1fd2ff2a1412acafcb.jpg)  
Figure 25: Changing MT data from mathematics to code preserves the best tested relative LR decay duration for both TS (left) and ZL (right). Both datasets use the same MoE model and 10B-token continuation with a 25% MT budget and final MT ratio 100%. The horizontal axis gives decay duration as a percentage of each schedule’s MT data window, with zero denoting Const and 100% denoting MDR. The vertical axis gives final MT loss minus the corresponding Const loss for that dataset and data schedule. Points show measured values, and rings mark the lowest tested losses at 20% for TS and 10% for ZL. Lines connect tested values.

## G ROUTER DYNAMICS AND INTERPRETATION

Measurement. We use training logs to track changes during training and separate checkpoint evaluations to examine individual expert loads. The training curves in Figures 6 and 26 to 28 show the auxiliary load balancing loss on the current mixture of PT and MT data, with its coefficient removed before logging. For a training microbatch, the value for layer ℓ is

$$
B _ { \ell } = 6 4 \sum _ { j = 1 } ^ { 6 4 } f _ { \ell j } P _ { \ell j } ,
$$

where $f _ { \ell j }$ is the fraction of routing assignments sent to expert $j ,$ counting all eight assignments per token, and $P _ { \ell j }$ is the mean router softmax probability for that expert before selecting the top eight experts. Uniform routing gives $B _ { \ell } = 1$ . The training objective multiplies this loss by the auxiliary loss coefficient, but Megatron divides out the coefficient before recording load\_balancing\_loss. Each plotted value is the unweighted mean across the 12 MoE layers, also averaged over microbatches, data parallel workers, and the 20 training steps in the logging interval. We plot these logged values without further rescaling.

For the checkpoint evaluations, we measure expert loads directly from routing assignments on heldout DCLM and FineMath, using approximately 34M tokens per domain at each checkpoint. For each of the 12 layers, let $p _ { \ell j }$ be the number of assignments sent to expert $j$ divided by the total number of assignments in that layer, so that the fractions sum to one across the 64 experts. We measure expert load imbalance using

$$
I _ { \ell } = 6 4 \sum _ { j = 1 } ^ { 6 4 } p _ { \ell j } ^ { 2 } .\tag{3}
$$

Uniform routing gives $I _ { \ell } = 1$ , and larger values indicate greater expert load imbalance. The heatmaps and regressions express expert shares relative to uniform routing as $6 4 p _ { \ell j }$ , so a value of one denotes the uniform share. The training statistic combines assignment fractions with router probabilities, whereas $I _ { \ell }$ uses only assignment counts. The two quantities are therefore different, even though both equal one under uniform routing. We use the training curves to locate transient spikes and the checkpoint measurements to examine how expert loads change on each domain, without directly comparing their numerical values.

Abrupt data distribution shifts under TS produce transient imbalance. At both 4B and 10B continuation horizons, the TS data switch produces a sharp load balancing spike that subsides over approximately 80 steps (Figures 27 and 28). Larger jumps in the MT ratio produce larger spikes, whereas ZL avoids a comparable peak as MT data enters gradually. The effect is therefore concentrated around the change in training distribution.

ZL reduces load balancing spikes under the same LR schedule. To separate the effect of the data transition from that of LR decay, we compare TS and ZL under Const. Both schedules use the same LR at every training step, yet TS produces a sharp load balancing spike while ZL avoids a comparable spike (Figures 6 and 26). This difference persists across both training lengths and all three tested final MT ratios, supporting the benefit of gradually introducing MT data.

Expert loads change as the router adapts to MT data. To understand the load balancing spike under TS, we examine checkpoints before the data switch, during MT training, and at training completion in the MoE TS+Const run with 10B continuation tokens and a final MT ratio of 100%. At each checkpoint, we separately evaluate how the router distributes DCLM (PT) and FineMath (MT) tokens across experts. Before the switch, expert loads are nearly balanced on DCLM. However, evaluating the same checkpoint on FineMath shows that a few experts receive many more tokens than the others, even though MT data has not yet entered training. This suggests that the spike occurs because the routing learned on PT data initially assigns MT tokens unevenly across experts. As MT training proceeds, the router distributes FineMath tokens more evenly, and the measured imbalance falls substantially (Table 13). The FineMath heatmaps in Figure 29 show this change directly: experts that receive a large share of tokens before the switch become less dominant by training completion. Meanwhile, expert loads on DCLM become somewhat less balanced, showing that adaptation to MT data also changes how the router distributes PT tokens.

![](images/9be7d8e9c7397e8c5f744e28c67ea76a9ca2c4e86e29d90db03844f858b98f47.jpg)  
Figure 26: Gradual MT entry reduces load balancing spikes. Under Const (left), TS and ZL use the same LR at every step, allowing us to compare the data schedules without also changing the LR. TS produces a sharp spike when the MT ratio suddenly increases, while ZL avoids a comparable spike by introducing MT data gradually. MDR (right) shows a similar pattern, although LR decay begins at different times for TS and ZL. Rows show MoE experiments with 4B (top) and 10B (bottom) tokens of continued training. Curves compare final MT ratios of 100%, 75%, and 50%. Dotted lines mark when MT data first enters. The 10B results also appear in Figure 6.

<table><tr><td>Checkpoint (step)</td><td>DCLM imbalance</td><td>FineMath imbalance</td></tr><tr><td>1,750 (before switch)</td><td>1.005</td><td>1.216</td></tr><tr><td>2,250 (462 steps after switch)</td><td>1.029</td><td>1.006</td></tr><tr><td>2,384 (final)</td><td>1.043</td><td>1.015</td></tr></table>

Table 13: Expert load imbalance on DCLM and FineMath for the MoE TS+Const run with 10B continuation tokens and a final MT ratio of 100%. Each value is the average of $I _ { \ell }$ across the 12 layers, as defined in Equation (3). The first checkpoint is 38 steps before the data switch. These measurements evaluate each domain separately on validation data, whereas the training logs summarize routing on the changing mixture of PT and MT data.

![](images/67962ef8e5e6a09f236897c789bd1f5697da6d46bd82a91dc5fb2b5dbd8e031b.jpg)  
Figure 27: TS produces a transient load balancing spike, while gradual MT entry avoids a comparable peak. The upper panels compare TS at several final MT ratios, including Const and MDR at 100% MT; the lower panels compare TS and ZL under MDR at final MT ratios of 100%, 75%, and 50%. These MoE runs use 4B continued training tokens. Training loss (left) and the logged load balancing statistic (right) are shown every 20 training steps.

![](images/5a09f51400398f30c77e1b474beb6fd6a303a2ea534fc0e7ede4ab0d48c08191.jpg)  
Figure 28: Smaller TS data shifts produce smaller load balancing spikes, while ZL avoids a comparable peak. The upper panels compare TS switches at steps 1,788, 1,589, and 1,192 for final MT ratios of 100%, 75%, and 50%; the lower panels compare TS and ZL under MDR at the same endpoints. These MoE runs use 10B continued training tokens. Training step is horizontal, with training loss on the left and logged load balancing on the right. Final validation losses are reported separately in Tables 11 and 12.

router @1750 (pre-switch) on DCLM 一 router @2250 (switch+462) on FineMath router @1750 (pre-switch) on FineMath router @2384 (switch+596) on DCLM router @2250 (switch+462) on DCLM router @2384 (switch+596) on FineMath

MoE: 10B-token continuation, final MT 100%, Const Expert-load imbalance per layer  
![](images/4fe82a9cf29dd1b40b37b79c504962203da879c39fe3839a5a159bd10dcdeee5.jpg)

![](images/bfd2ad8ffa2698e3418654947043dc6af7be8d6f7df59642e359fbb26bfc9c7c.jpg)

![](images/47b3a9ac9d6edb4a9bfd5d0b30f1b26520fd4f68fb8f3178d6a01b08fd84fc8c.jpg)

![](images/d20fb5fdb15e9b3a2fb5377e74b0d1ae3573ee506b2792c5fcd12a5d613af1cc.jpg)  
Figure 29: The router initially assigns MT tokens unevenly across experts and becomes more balanced during MT training. Checkpoint routing diagnostics for the MoE TS+Const run with 10B continuation tokens and a final MT ratio of 100%. Top: imbalance by layer on DCLM and FineMath at steps 1,750, 2,250, and 2,384. Bottom: FineMath expert shares before the switch, at training end, and their difference, with expert and layer indices as coordinates. Shares are relative to uniform routing. These diagnostics describe one run, without replica averaging.

How expert loads change under TS and ZL. The preceding analysis shows that expert loads become more balanced after the TS switch. We next examine which experts receive more or fewer FineMath assignments after adaptation and compare these changes under TS and ZL. In panel (b) of Figure 30, the horizontal axis shows how much each expert’s initial assignment share differs from the uniform share, and the vertical axis shows how its share changes by training completion. The negative trend means that experts receiving more than the uniform share tend to receive less after adaptation, while those receiving less tend to receive more. To compare TS and ZL, we examine both the size of these changes and whether experts that were initially more heavily used remain so afterwards.

We use three statistics to compare the above changes (Table 14). For routing shares p before MT entry time and q at training completion, total variation (TV) is

$$
\mathrm { T V } ( p , q ) = { \frac { 1 } { 1 2 } } \sum _ { \ell = 1 } ^ { 1 2 } { \frac { 1 } { 2 } } \sum _ { j = 1 } ^ { 6 4 } | p _ { \ell j } - q _ { \ell j } | .
$$

It measures the net change in the routing distribution, averaged across layers. For example, TV of 0.15 means that matching the final shares requires redistributing at least 15% of the assignment mass, on average across layers. It does not measure cumulative movement during training. The regression residual standard deviation measures how far each expert’s share change deviates from the fitted relationship with its initial excess. Spearman correlation compares the expert rankings by assignment share before and after adaptation within each layer, then averages across layers.

TS+Const and ZL+MDR show comparable net redistribution, but ZL+MDR has substantially less scatter around the fitted trend and preserves more of the initial expert ordering (Table 14 and Fig. 30). In other words, ZL+MDR shows a more consistent contraction of the initial load differences, while TS+Const shows more reshuffling of which experts receive the most assignments. These checkpoint statistics compare different LR schedules and similar but nonidentical starting routers, so they describe the redistribution pattern rather than isolating the effect of data shape alone.
<table><tr><td>Schedule</td><td>Final imbalance FineMath / DCLM</td><td>Net routing change FineMath TV</td><td>Regression residual std.</td><td>Rank retention Spearman</td></tr><tr><td>TS+Const</td><td>1.015 / 1.043</td><td>0.156</td><td>0.119</td><td>0.30</td></tr><tr><td>ZL+MDR</td><td>1.008 / 1.039</td><td>0.140</td><td>0.042</td><td>0.74</td></tr></table>

Table 14: MoE routing comparison over 10B continuation tokens with a 25% MT budget and final MT ratio 100%. TS+Const compares steps 1,750 and 2,384, and ZL+MDR compares steps 1,150 and 2,384, from before each schedule’s MT entry time to training completion. TV reports the net change in FineMath assignment shares. Regression residuals use shares relative to uniform routing, and Spearman correlations are averaged across layers. Figure 6 provides the separate control with identical LR trajectories.

![](images/9a5cd77461a44c0a2b79a0aac1fd9aa8cb0d5afc71de70ef274656e4819e6c22.jpg)  
(a) Final expert load imbalance.

![](images/68c4bd0f61e4c4c47b01af7cff4e454fe0b088893e57058ce4ab0ff68f244dd3.jpg)

(b) Changes in the fraction of FineMath assignments sent to each expert.  
![](images/4c0f2a492e3b62ec705ac813cf295796d379a353cfbd67470030561558d20349.jpg)  
(c) Preservation of expert load rankings.

Figure 30: ZL+MDR preserves expert load rankings better than TS+Const, although both reach nearly balanced MT loads. (a) Both schedules bring FineMath (MT) loads close to uniform in most layers; one denotes equal assignment shares across experts. (b) Initially more heavily used experts lose share, while less used experts gain it. Each point represents one expert in one layer. The horizontal axis shows its share before MT entry time minus the uniform share, and the vertical axis shows its final minus initial share, both in units of 1/64. Points on the dotted line end at the uniform share. (c) Despite similar final balance, the ordering of experts changes more under TS+Const. The higher rank correlations under ZL+MDR mean that experts receiving relatively more FineMath assignments initially tend to remain more heavily used at the end. All checkpoint measurements use held-out data. These 10B-token MoE continuations use different LR schedules, so the comparison does not isolate the effect of the data schedule. See Table 14 for the setup.

## H ADDITIONAL THEORETICAL SETUP

Original SGD training algorithm with data mixing. Let K denote the total number of training steps, and at each training step $k \in [ 0 , K - 1 ]$ , we sample MT data with probability $\gamma ( k )$ and sample PT data with probabilit $1 - \gamma ( k )$ . Thus, for the $k ^ { \mathrm { { t h } } }$ data point ${ \bf { x } } _ { k } .$ , its covariance matrix is $\mathbb { E } \pmb { x } _ { k } \pmb { x } _ { k } ^ { \top } \ = \ \gamma ( \hat { k } ) \pmb { H } + ( \bar { 1 } - \gamma ( k ) ) \pmb { G }$ At the $k ^ { \mathrm { { t h } } }$ step, we therefore have $\mathcal { L } _ { k } ( \pmb \theta )$ ${ \textstyle \frac { 1 } { 7 } } \pmb { \theta } ^ { \top } \left( \gamma ( k ) \pmb { H } + \ddot { ( 1 - \gamma ( k ) ) } \pmb { G } \right) \theta$ . We train our model using online stochastic gradient descent (SGD) with learning rate schedule $E = \{ \eta _ { 0 } , \cdot \cdot \cdot , \eta _ { K - 1 } \}$ . The update rule can be written as

$$
\pmb { \theta } _ { k + 1 } = \pmb { \theta } _ { k } - \eta _ { k } \left( \nabla \mathcal { L } _ { k } ( \pmb { \theta } _ { k } ) + \xi _ { k } \right) .\tag{4}
$$

Here $\xi _ { k } \sim \mathcal { N } ( 0 , \Sigma )$ is additional noise independent of the signal. We let $\pmb { \Sigma } = \mathrm { d i a g } ( \pmb { \Sigma } _ { 0 } , \pmb { \Sigma } _ { 1 } , \pmb { \Sigma } _ { 2 } )$ where $\pmb { \Sigma } _ { l } = \mathrm { d i a g } ( \nu _ { 1 } ^ { ( l ) } , \dots , \nu _ { d _ { l } } ^ { ( l ) } ) \in \mathbb { R } ^ { d _ { l } \times d _ { l } } \mathrm { f o r } l = 0 , 1 , 2$ . We assume that $\nu _ { i } ^ { ( l ) } > 0$ for all $0 < i \leq d _ { l }$ and $l = 0 , 1 , 2$ . Our goal is to minimize the expected validation loss $\mathbb { E } [ \mathcal { L } ( \pmb { \theta } _ { K } ) ]$ , where the expectation is taken over the randomness in the gradient noise and the data sampling.

SDE approximation. Motivated by Li et al. (2017; 2019; 2020; 2021); Pesme et al. (2021); Li et al. (2022; 2025), we analyze the continuous time limit of Equation (4), which makes the analysis more tractable. We fix a discretization step size $h > 0$ , set $t _ { k } = k h$ for $k = 0 , \ldots , K$ , and let $T = K h$ We also define $\eta ( t _ { k } ) = \eta _ { k } / h$ and $\gamma ( t _ { k } ) = \gamma ( k )$ as the continuous-time learning rate schedule and data schedule, respectively. The corresponding continuous-time loss function is defined as

$$
\mathcal { L } _ { t } ( \pmb \theta ) : = \frac { 1 } { 2 } \pmb \theta ^ { \top } \left( \gamma ( t ) \pmb H + \left( 1 - \gamma ( t ) \right) \pmb G \right) \pmb \theta .
$$

Using $\eta _ { k } = h \eta ( t _ { k } )$ , Equation (4) becomes

$$
\pmb { \theta } _ { k + 1 } - \pmb { \theta } _ { k } = - \eta ( t _ { k } ) \nabla \mathcal { L } _ { t _ { k } } ( \pmb { \theta } _ { k } ) h - \eta ( t _ { k } ) h \xi _ { k } .
$$

Following Li et al. (2019); Orvieto & Lucchi (2019), we approximate the iteration above with the Itô SDE given in Equation (1). The minimization of the expected validation loss $\mathbb { E } [ \mathcal { L } ( \pmb { \theta } _ { K } ) ]$ is then approximated by the minimization of $\mathbb { E } [ \mathcal { L } ( \pmb { \theta } _ { T } ) ]$ , where the expectation is taken over the randomness in the Brownian motion and the data sampling.

## I FORMAL STATEMENTS OF THEORETICAL RESULTS

In this section, we provide the formal statement of Theorem 3.1 and its related assumptions. We restate Theorem 3.1 as Theorems I.1 to I.3, corresponding to parts (i), (ii) and (iii), respectively. The proofs of Theorems I.1 to I.3 are given in Appendices J to L respectively.

We assume that $\textstyle \sum _ { i = 1 } ^ { d _ { 1 } } \theta _ { i } ^ { ( 1 ) } ( 0 ) ^ { 2 } > 0$ in subsequent results. We first show the optimal data schedule under a constant learning rate.

Theorem I.1 (Optimal Data Schedule under a Constant Learning Rate). Suppose that $\eta ( t ) \equiv \eta _ { \mathrm { m a x } } >$ 0. Under Condition 3.1, the optimal data schedule is given by thefollowing schedule:

$$
\gamma ^ { * } ( t ) = \left. \begin{array} { l l } { 0 , } & { t \in [ 0 , ( 1 - \alpha ) T ) , } \\ { 1 , } & { t \in [ ( 1 - \alpha ) T , T ] , } \end{array} \right.
$$

which we refer to as the 0-1 data schedule.

We next replace the constant learning rate with a learning rate schedule with linear decay, while keeping all other settings fixed. The following theorem gives sufficient conditions on the decay strength of the learning rate and the MT budget, under which the 0-1 data schedule in Theorem I.1 is no longer optimal.

Theorem I.2 (Optimal Schedule under Linear Learning Rate Decay). Suppose that the learning rate decays linearly according to

$$
\eta ( t ) = \eta _ { \mathrm { m a x } } - \frac { c } { T } t , \qquad ( 1 - 4 ^ { - \frac { 1 } { 3 } } ) \eta _ { \mathrm { m a x } } < c < \eta _ { \mathrm { m a x } } .
$$

Then, under Condition $3 . I ,$ when $\begin{array} { r } { 0 < \alpha < 1 - \frac { ( 1 - 4 ^ { - \frac { 1 } { 3 } } ) \eta _ { \mathrm { m a x } } } { c } , \gamma ^ { \ast } ( t ) } \end{array}$ in Theorem I.1 is not the optimal schedule.

Theorem I.2 does not characterize the exact form of the optimal data schedule under a linearly decaying learning rate schedule. However, it proves that changing only the learning rate schedule from constant to linear decay can change the corresponding optimal data schedule with everything else fixed.

Theorem I.1 and Theorem I.2 show that different learning rate schedules induce different optimal data schedules. Nevertheless, the optimal data schedule under a linear decay learning rate is not explicitly solvable. Motivated by the multi-power law in Luo et al. (2025), we introduce a power-law spectral model that allows us to derive the optimal data schedule under a linear decay learning rate in closed form.

Assumption I.1 (Power-Law Spectral Model). For each block $l \in \{ 0 , 1 , 2 \}$ , assume that the pairs $\{ ( \lambda _ { i } ^ { ( l ) } , \nu _ { i } ^ { ( l ) } ) \} _ { i = 1 } ^ { d _ { l } }$ are independent and identically distributed across i, with

$$
\lambda _ { i } ^ { ( l ) } \sim p _ { l } ( \lambda ) = D _ { l } \lambda ^ { \alpha _ { l } } \exp ( - K _ { l } \lambda ) , \qquad \lambda \in [ 0 , \infty ) , \qquad i = 1 , \dots , d _ { l } ,
$$

$$
\nu _ { i } ^ { ( l ) } \mid \lambda _ { i } ^ { ( l ) } \sim q _ { l } ( \nu \mid \lambda _ { i } ^ { ( l ) } ) , \qquad \nu \in [ 0 , \infty ) , \qquad i = 1 , \ldots , d _ { l } ,
$$

where $- 1 < \alpha _ { l } < 0 , K _ { l } > 0 ,$ , and

$$
D _ { l } = \frac { K _ { l } ^ { \alpha _ { l } + 1 } } { \Gamma ( \alpha _ { l } + 1 ) } .
$$

Moreover, for some $\rho _ { l } > 0 ,$

$$
\mathbb { E } _ { q _ { l } ( . | \lambda ) } [ \nu \mid \lambda ] = F _ { l } \lambda ^ { \rho _ { l } } \exp ( - G _ { l } \lambda ) ,
$$

where $F _ { l } , G _ { l } > 0$ are constants independent ofthe learning rate schedule.

The spectral assumption describes a collection of directions that are learned at different speeds. A smaller eigenvalue λ means that the loss changes more weakly along the corresponding direction, so the error in that direction is corrected more slowly. The density $p _ { l }$ describes how common different eigenvalues are. Since $- 1 < \alpha _ { l } < 0 .$ , the factor $\lambda ^ { \alpha _ { l } }$ places more mass near zero, representing many slowly learned directions, while $e ^ { - K _ { l } \lambda }$ makes very large eigenvalues increasingly rare. Together, these factors give a Gamma density with shape $\alpha _ { l } + 1$ and rate $K _ { l } .$ . Luo et al. (2025) use an eigenvalue density with a power-law expression to explain how Multi-Power Law arises. The two models differ in how they describe large eigenvalues: Luo et al. (2025) impose a finite upper bound on the eigenvalues, and our model uses a density on $[ 0 , \infty )$ with an exponentially decaying tail.

The noise assumption describes how the strength of the additional gradient noise varies with eigenvalues. Here, $\nu _ { i } ^ { ( l ) }$ is the variance of noise in direction i of block l: a larger value means stronger random perturbations to the gradient along that direction. Thus, $\mathbb { E } [ \nu \mid \bar { \lambda ] } = F _ { l } \lambda ^ { \rho _ { l } } e ^ { - G _ { l } \lambda }$ specifies the average noise variance among directions with eigenvalue λ. The parameter $F _ { l }$ sets the overall noise scale. In directions with small eigenvalues, where the exponential factor $e ^ { - G _ { l } \bar { \lambda } }$ is close to one, $\rho _ { l } > 0$ describes weaker noise in those directions. The factor $\stackrel { \cdot } { e } ^ { - G _ { l } \lambda }$ makes the average noise variance decay in directions with very large eigenvalues. This functional form follows Luo et al. (2025), who use such a relationship between eigenvalues and noise variance to derive the additional loss reduction caused by learning rate decay in a quadratic model.

Under Assumption I.1, we consider minimizing the expected validation loss, where the expectation is taken with respect to the training process and the distribution of the eigenvalues. Under this setup, we show the shape of the optimal data schedule under a linearly decaying learning rate in closed form.

Theorem I.3 (Optimal Data Schedule under the Power-Law Spectral Model). Suppose that $0 < \rho _ { 1 } <$ 1 and that $\eta ( t ) = \eta _ { \mathrm { m a x } } - c t / T$ , where $\eta _ { \mathrm { m a x } }$ and c are fixed constants satisfying $\left( 1 - 4 ^ { - 1 / 3 } \right) \eta _ { \mathrm { m a x } } <$ $c < \eta _ { \mathrm { m a x } } .$ Define $x _ { \star } : = c ^ { - 1 } \left( 4 ^ { - 1 / 3 } \eta _ { \operatorname* { m a x } } - \left( \eta _ { \operatorname* { m a x } } - c \right) \right) \in ( 0 , 1 )$ , and suppose that the data fraction $0 < \alpha < x _ { \star }$ isfixed. Then, when T is sufficiently large, the optimal data schedule under Condition 3.1 and Assumption I.1 can be given by

$$
\gamma ^ { * } ( t ) = \left\{ \begin{array} { l l } { 0 , } & { t \in [ 0 , t _ { 1 } ) , } \\ { \left( G _ { \mathrm { e f f } } + \displaystyle \int _ { t _ { 2 } } ^ { T } \eta ( \tau ) \mathrm { d } \tau \right) \displaystyle \frac { 4 c } { T ( b + 1 ) } \left( \eta ( t _ { 2 } ) \right) ^ { - \frac { 4 } { b + 1 } } \left( \eta ( t ) \right) ^ { - \frac { 2 ( b - 1 ) } { b + 1 } } , } & { t \in [ t _ { 1 } , t _ { 2 } ) , } \\ { 1 , } & { t \in [ t _ { 2 } , T ] , } \end{array} \right.
$$

where $\textstyle b : = \alpha _ { 1 } + \rho _ { 1 } + 2 , G _ { \mathrm { e f f } } : = { \frac { K _ { 1 } + G _ { 1 } } { 2 } }$ , and $t _ { 1 } , t _ { 2 }$ are two time points that satisfy $0 < t _ { 1 } < t _ { 2 } < T ,$ $t _ { 1 } = \Theta ( T )$ and $t _ { 2 } = \Theta ( T )$ ).

## J PROOF OF THEOREM I.1

In this section, we derive the evaluation loss from the SDE in Equation (1) and then prove the optimality of the 0–1 data schedule in Theorem I.1.

## J.1 DERIVATION OF THE TERMINAL LOSS

Before we begin our proof, we first define some additional notation. We denote

$$
a _ { i } ^ { ( l ) } ( t ) : = \left\{ \begin{array} { l l } { \lambda _ { i } ^ { ( 0 ) } , } & { l = 0 , } \\ { \gamma ( t ) \lambda _ { i } ^ { ( 1 ) } , } & { l = 1 , } \\ { \big ( 1 - \gamma ( t ) \big ) \lambda _ { i } ^ { ( 2 ) } , } & { l = 2 , } \end{array} \right. \quad i = 1 , \ldots , d _ { l } .\tag{5}
$$

For $0 \ \leq \ s \ \leq \ t \ \leq \ T$ , we define $\begin{array} { r } { R _ { i } ^ { ( l ) } ( s , t ) : = \exp \left( - \int _ { s } ^ { t } \eta ( u ) a _ { i } ^ { ( l ) } ( u ) \mathrm { d } u \right) } \end{array}$ , and $E _ { i } ^ { ( l ) } ( s , t ) : =$ $\begin{array} { r } { R _ { i } ^ { ( l ) } ( s , t ) ^ { 2 } = \exp \left( - 2 \int _ { s } ^ { t } \eta ( u ) a _ { i } ^ { ( l ) } ( u ) \mathrm { d } u \right) } \end{array}$ . Because H, G, and Σ are diagonal, the ith coordinate of Equation (1) satisfies

$$
\mathrm { d } \theta _ { i } ^ { ( l ) } ( t ) = - \eta ( t ) a _ { i } ^ { ( l ) } ( t ) \theta _ { i } ^ { ( l ) } ( t ) \mathrm { d } t + \eta ( t ) \sqrt { h \nu _ { i } ^ { ( l ) } } \mathrm { d } B _ { i } ^ { ( l ) } ( t ) .\tag{6}
$$

Solving Equation (6) gives

$$
\theta _ { i } ^ { ( l ) } ( T ) = R _ { i } ^ { ( l ) } ( 0 , T ) \theta _ { i } ^ { ( l ) } ( 0 ) + \sqrt { h \nu _ { i } ^ { ( l ) } } \int _ { 0 } ^ { T } \eta ( s ) R _ { i } ^ { ( l ) } ( s , T ) \mathrm { d } B _ { i } ^ { ( l ) } ( s ) .
$$

The stochastic integral has zero mean. Since $R _ { i } ^ { ( l ) } ( 0 , T ) \theta _ { i } ^ { ( l ) } ( 0 )$ is deterministic, its cross term with the stochastic integral vanishes after taking expectations. The Itô isometry then gives

$$
\mathbb { E } \Big [ \theta _ { i } ^ { ( l ) } ( T ) ^ { 2 } \Big ] = \theta _ { i } ^ { ( l ) } ( 0 ) ^ { 2 } E _ { i } ^ { ( l ) } ( 0 , T ) + h \nu _ { i } ^ { ( l ) } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } E _ { i } ^ { ( l ) } ( s , T ) \mathrm { d } s .\tag{7}
$$

We denote $\begin{array} { r } { \mathcal { L } _ { l } ( \pmb { \theta } ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d _ { l } } \lambda _ { i } ^ { ( l ) } \big ( \theta _ { i } ^ { ( l ) } \big ) ^ { 2 } } \end{array}$ . Since the model is evaluated on the MT loss ${ \mathcal { L } } ( \pmb { \theta } ) =$ $\begin{array} { r } { \sum _ { l = 0 } ^ { 1 } \mathcal { L } _ { l } ( \pmb { \theta } ) = \frac { 1 } { 2 } \sum _ { l = 0 } ^ { 1 } \sum _ { i = 1 } ^ { d _ { l } } \lambda _ { i } ^ { ( l ) } \big ( \theta _ { i } ^ { ( l ) } \big ) ^ { 2 } } \end{array}$ , substituting Equation (7) yields

$$
\mathcal { I } ( \gamma , \eta ) : = \mathbb { E } [ \mathcal { L } ( \theta _ { T } ) ] = \frac { 1 } { 2 } \sum _ { l = 0 } ^ { 1 } \sum _ { i = 1 } ^ { d _ { l } } \lambda _ { i } ^ { ( l ) } \left[ \theta _ { i } ^ { ( l ) } ( 0 ) ^ { 2 } E _ { i } ^ { ( l ) } ( 0 , T ) + h \nu _ { i } ^ { ( l ) } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } E _ { i } ^ { ( l ) } ( s , T ) \mathrm { d } s \right]
$$

$$
= \frac { 1 } { 2 } \sum _ { l = 0 } ^ { 1 } \sum _ { i = 1 } ^ { d _ { l } } \lambda _ { i } ^ { ( l ) } \theta _ { i } ^ { ( l ) } ( 0 ) ^ { 2 } E _ { i } ^ { ( l ) } ( 0 , T ) + \frac { 1 } { 2 } \sum _ { l = 0 } ^ { 1 } \sum _ { i = 1 } ^ { d _ { l } } \lambda _ { i } ^ { ( l ) } h \nu _ { i } ^ { ( l ) } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } E _ { i } ^ { ( l ) } ( s , T ) \mathrm { d } s .\tag{8}
$$

We call the first term the bias term, which arises because the initialization of the parameters is different from the global optimum. We call the second term the variance term, which is induced by the accumulation of gradient noise. Note that for all $i , l ,$ and $s < t , E _ { i } ^ { ( l ) } ( s , t )$ is convex up to a set of measure zero. Thus $\mathcal { I } ( \gamma , \eta )$ is a convex functional of the data schedule $\gamma ( t )$ up to a set of measure zero. Furthermore, the feasible set $\begin{array} { r } { \{ \gamma ( t ) : [ 0 , T ]  [ 0 , 1 ] , \frac { 1 } { T } \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t \leq \alpha \} } \end{array}$ is convex. Thus the optimization problem in Equation (2) is a convex optimization problem. The feasible set also satisfies the Slater condition, so the KKT conditions are necessary and sufficient for optimality.

Next we prove that if the optimal data schedule exists, then the optimal data schedule is unique up to a set of measure zero. We proceed by contradiction and assume that there exist two different optimal data schedules $\gamma _ { 1 }$ and $\gamma _ { 2 } .$ , and we define $M = \mathcal { I } ( \gamma _ { 1 } , \eta )$ . By the convexity of $\mathcal { I } ( \gamma , \eta )$ , we have

$$
\mathcal { I } ( x \gamma _ { 1 } + ( 1 - x ) \gamma _ { 2 } , \eta ) \leq x \mathcal { I } ( \gamma _ { 1 } , \eta ) + ( 1 - x ) \mathcal { I } ( \gamma _ { 2 } , \eta ) = M .
$$

Equality must hold; otherwise, $x \gamma _ { 1 } + ( 1 - x ) \gamma _ { 2 }$ would achieve a smaller objective value, contradicting the optimality of $\gamma _ { 1 }$ and $\gamma _ { 2 }$ . Notice that the summands for $l = 0$ are independent of the data schedule. Thus by the strict convexity of the exponential function, the equality holds if and only if for all i and for all $\dot { 0 } \leq s \leq T$

$$
- 2 \int _ { s } ^ { T } \eta ( u ) \gamma _ { 1 } ( u ) \lambda _ { i } ^ { ( 1 ) } \mathrm { d } u = - 2 \int _ { s } ^ { T } \eta ( u ) \gamma _ { 2 } ( u ) \lambda _ { i } ^ { ( 1 ) } \mathrm { d } u .
$$

Taking the derivative, we have $\gamma _ { 1 } ( u ) = \gamma _ { 2 } ( u )$ , a.e., which is a contradiction.

## J.2 KKT CONDITION FOR EQUATION (2)

We denote the bracketed term in Equation (8) by

$$
Q _ { i } ^ { ( l ) } ( t ) : = \theta _ { i } ^ { ( l ) } ( 0 ) ^ { 2 } E _ { i } ^ { ( l ) } ( 0 , T ) + h \nu _ { i } ^ { ( l ) } \int _ { 0 } ^ { t } \eta ( s ) ^ { 2 } E _ { i } ^ { ( l ) } ( s , T ) \mathrm { d } s .
$$

Note that $\gamma ( t )$ does not affect $Q _ { i } ^ { ( 0 ) } ( t )$ , so we only need to minimize $\textstyle { \frac { 1 } { 2 } } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { ( 1 ) } Q _ { i } ^ { ( 1 ) } ( T )$ . From now on, we omit the superscript (1) for simplicity.

We rewrite the MT data constraint as

$$
\int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t - \alpha T \leq 0 ,
$$

and define the Lagrangian functional

$$
\mathcal { F } ( \gamma , \kappa ; \eta ) : = \mathcal { I } ( \gamma , \eta ) + \kappa \left( \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t - \alpha T \right) .
$$

Now we derive the first variation of $\mathcal { F } ( \gamma , \kappa ; \eta )$ . For a bounded perturbation $v ( t )$ , let $\gamma _ { \varepsilon } = \gamma + \varepsilon v$ We use a subscript ε to denote quantities induced by the perturbed schedule and define

$$
\begin{array} { c l l } { \displaystyle a _ { i , \varepsilon } ( u ) : = \lambda _ { i } \gamma _ { \varepsilon } ( u ) = a _ { i } ( u ) + \varepsilon \lambda _ { i } v ( u ) , } \\ { \displaystyle E _ { i , \varepsilon } ( s , T ) : = \exp \left( - 2 \int _ { s } ^ { T } \eta ( t ) a _ { i , \varepsilon } ( t ) \mathrm { d } t \right) } \\ { \displaystyle = E _ { i } ( s , T ) \exp \left( - 2 \varepsilon \lambda _ { i } \int _ { s } ^ { T } \eta ( t ) v ( t ) \mathrm { d } t \right) . } \end{array}
$$

In particular, $a _ { i , 0 } = a _ { i }$ and $E _ { i , 0 } = E _ { i }$ , so the notation without an ε subscript always refers to the original schedule γ. Differentiating the second identity at $\varepsilon = 0$ gives

$$
\frac { \mathrm { d } } { \mathrm { d } \varepsilon } E _ { i , \varepsilon } ( s , T ) \bigg \vert _ { \varepsilon = 0 } = - 2 \lambda _ { i } E _ { i } ( s , T ) \int _ { s } ^ { T } \eta ( t ) v ( t ) \mathrm { d } t .\tag{9}
$$

To differentiate the terminal loss, we substitute Equation (9) into Equation (8) and obtain

$$
\begin{array} { l } { { \displaystyle { \cal D } { \mathcal I } ( \gamma ) [ v ] = - \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \theta _ { i } ( 0 ) ^ { 2 } E _ { i } ( 0 , T ) \int _ { 0 } ^ { T } \eta ( t ) v ( t ) \mathrm { d } t } } \\ { { \displaystyle ~ - \ h \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \nu _ { i } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } E _ { i } ( s , T ) \left[ \int _ { s } ^ { T } \eta ( t ) v ( t ) \mathrm { d } t \right] \mathrm { d } s . } } \end{array}
$$

The second term integrates over the triangular region $\{ ( s , t ) : 0 \leq s \leq t \leq T \}$ . By Fubini’s theorem,

$$
\begin{array} { l } { \displaystyle \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } E _ { i } ( s , T ) \left[ \int _ { s } ^ { T } \eta ( t ) v ( t ) \mathrm { d } t \right] \mathrm { d } s } \\ { \displaystyle \qquad = \int _ { 0 } ^ { T } \eta ( t ) v ( t ) \left[ \int _ { 0 } ^ { t } \eta ( s ) ^ { 2 } E _ { i } ( s , T ) \mathrm { d } s \right] \mathrm { d } t . } \end{array}
$$

Both parts of the objective variation can therefore be collected into a single integral. Including the variation of the constraint gives

$$
\begin{array} { r l r } {  { D _ { \gamma } \mathcal { F } ( \gamma , \kappa ; \eta ) [ v ] = D \mathcal { I } ( \gamma ) [ v ] + \kappa \int _ { 0 } ^ { T } v ( t ) \mathrm { d } t } } \\ & { } & { = \int _ { 0 } ^ { T } \big ( \kappa - M _ { \gamma } ( t ) \big ) v ( t ) \mathrm { d } t . } \end{array}
$$

where the marginal value of MT data at time t is

$$
M _ { \gamma } ( t ) : = \eta ( t ) \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \left[ \theta _ { i } ( 0 ) ^ { 2 } E _ { i } ( 0 , T ) + h \nu _ { i } \int _ { 0 } ^ { t } \eta ( s ) ^ { 2 } E _ { i } ( s , T ) \mathrm { d } s \right] .\tag{10}
$$

Consequently, the pointwise KKT conditions for the box constraint $0 \leq \gamma ( t ) \leq 1$ are

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { M _ { \gamma ^ { \star } } ( t ) \leq \kappa , } & { \gamma ^ { \star } ( t ) = 0 , } \\ { M _ { \gamma ^ { \star } } ( t ) = \kappa , } & { 0 < \gamma ^ { \star } ( t ) < 1 , } \\ { M _ { \gamma ^ { \star } } ( t ) \geq \kappa , } & { \gamma ^ { \star } ( t ) = 1 , } \end{array} \right. } \end{array}\tag{11}
$$

together with primal feasibility, $\kappa \geq 0$ , and complementary slackness

$$
\kappa \left( \int _ { 0 } ^ { T } \gamma ^ { \star } ( t ) \mathrm { d } t - \alpha T \right) = 0 .
$$

## J.3 PROOF OF THEOREM I.1

Proof. Recall that

$$
Q _ { i } ( t ) : = \theta _ { i } ( 0 ) ^ { 2 } E _ { i } ( 0 , T ) + h \nu _ { i } \int _ { 0 } ^ { t } \eta ( s ) ^ { 2 } E _ { i } ( s , T ) \mathrm { d } s .
$$

The fundamental theorem of calculus gives $Q _ { i } ^ { \prime } ( t ) = h \nu _ { i } \eta ( t ) ^ { 2 } E _ { i } ( t , T )$ . Hence, for a differentiable learning-rate schedule,

$$
M _ { \gamma } ^ { \prime } ( t ) = \eta ^ { \prime } ( t ) \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } Q _ { i } ( t ) + h \eta ( t ) ^ { 3 } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \nu _ { i } E _ { i } ( t , T ) .
$$

Under a constant learning rate $\eta _ { \mathrm { m a x } } ,$ we have $\eta ^ { \prime } ( t ) = 0$ , and therefore

$$
M _ { \gamma } ^ { \prime } ( t ) = h \eta _ { \mathrm { m a x } } ^ { 3 } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \nu _ { i } E _ { i } ( t , T ) > 0 .
$$

The strict inequality follows from $h > 0 , \eta _ { \mathrm { m a x } } > 0$ , and $\lambda _ { i } , \nu _ { i } > 0$ in our setup.

We now give a global comparison, which also proves uniqueness. For a feasible schedule $\gamma _ { \mathrm { : } }$ define its remaining MT mass after time s by

$$
C _ { \gamma } ( s ) : = \int _ { s } ^ { T } \gamma ( u ) \mathrm { d } u .
$$

Because $M _ { \gamma } ( t ) > 0$ for almost every $t > 0 ,$ , adding MT mass on any positive-measure set on which $\gamma < 1$ strictly decreases the objective. Thus, except for the immediate boundary cases $\alpha = 0$ and $\alpha = 1$ , every optimal schedule exhausts the budget:

$$
\int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t = \alpha T .\tag{12}
$$

For any schedule satisfying (12), the box constraint gives

$$
C _ { \gamma } ( s ) \leq \operatorname* { m i n } \{ \alpha T , T - s \} , \qquad 0 \leq s \leq T .
$$

The schedule $\bar { \gamma }$ in Theorem I.1 attains this upper bound for every s. Moreover, under the constant learning rate,

$$
E _ { i } ^ { \gamma } ( s , T ) = \exp [ - 2 \eta _ { \mathrm { m a x } } \lambda _ { i } C _ { \gamma } ( s ) ] .
$$

Consequently, $E _ { i } ^ { \bar { \gamma } } ( s , T ) \leq E _ { i } ^ { \gamma } ( s , T )$ for every i and $s .$ The initialization term in Equation (8) is the same for all schedules satisfying (12), while the variance term of $\bar { \gamma }$ is no larger coordinate-wise. It follows that

$$
\mathcal { I } ( \bar { \gamma } , \eta _ { \mathrm { m a x } } ) \leq \mathcal { I } ( \gamma , \eta _ { \mathrm { m a x } } ) .
$$

If equality holds, the positivity of the variance weights forces $C _ { \gamma } ( s ) = C _ { \bar { \gamma } } ( s )$ for almost every s. Taking weak derivatives then gives $\gamma = \bar { \gamma }$ almost everywhere. The cases $\dot { \alpha } = 0$ and $\alpha = 1$ follow directly from the box constraint and the strict decrease of the objective when MT mass is added.

## K PROOF OF THEOREM I.2

Proof. We prove the theorem by contradiction. Let

$$
t _ { \alpha } : = ( 1 - \alpha ) T , \qquad { \bar { \gamma } } ( t ) : = \mathbf { 1 } \{ t \geq t _ { \alpha } \}
$$

denote the transition time and the 0-1 schedule from Theorem I.1, respectively. Suppose that γ¯ is optimal under the linear decay learning rate schedule. The KKT conditions derived above imply

$$
M _ { \bar { \gamma } } ( t ) \leq \kappa \quad ( t < t _ { \alpha } ) , \qquad M _ { \bar { \gamma } } ( t ) \geq \kappa \quad ( t > t _ { \alpha } ) .
$$

Because $M _ { \bar { \gamma } }$ is continuously differentiable, it follows that $M _ { \bar { \gamma } } ( t _ { \alpha } ) = \kappa$ and

$$
M _ { \bar { \gamma } } ^ { \prime } ( t _ { \alpha } ) \geq 0 .\tag{13}
$$

We now calculate this derivative explicitly. Define the integrated learning rate over an interval by

$$
\Phi ( a , b ) : = \int _ { a } ^ { b } \eta ( u ) \mathrm { d } u , \qquad 0 \leq a \leq b \leq T .
$$

The two intervals before and after the transition occur repeatedly below, so we introduce the mnemonic abbreviations

$$
\Phi _ { \alpha } ^ { - } ( s ) : = \Phi ( s , t _ { \alpha } ) , \qquad \Phi _ { \alpha } ^ { + } : = \Phi ( t _ { \alpha } , T ) ,
$$

where the superscripts − and + indicate accumulation before and after $t _ { \alpha }$ , respectively. For $\eta ( t ) =$ $\eta _ { \mathrm { m a x } } - c t / T$ , these quantities are

$$
\begin{array} { r l } & { \eta _ { \alpha } : = \eta ( t _ { \alpha } ) = \eta _ { \mathrm { m a x } } - c ( 1 - \alpha ) , } \\ & { \Phi _ { \alpha } ^ { + } = \eta _ { \mathrm { m a x } } ( T - t _ { \alpha } ) - \displaystyle \frac { c } { 2 T } \big ( T ^ { 2 } - t _ { \alpha } ^ { 2 } \big ) } \\ & { \qquad = \alpha T \left[ \eta _ { \mathrm { m a x } } - c \left( 1 - \displaystyle \frac { \alpha } { 2 } \right) \right] , } \\ & { \Phi _ { \alpha } ^ { - } ( s ) = \eta _ { \mathrm { m a x } } ( t _ { \alpha } - s ) - \displaystyle \frac { c } { 2 T } \big ( t _ { \alpha } ^ { 2 } - s ^ { 2 } \big ) , \qquad 0 \le s \le t _ { \alpha } . } \end{array}
$$

Under $\bar { \gamma } ,$ coordinate i has zero curvature before $t _ { \alpha }$ and curvature $\lambda _ { i }$ afterwards. Let $\bar { E } _ { i }$ denote the squared propagator induced by this candidate schedule. Consequently,

$$
\begin{array} { r } { \bar { E } _ { i } ( s , T ) = \exp \left( - 2 \lambda _ { i } \Phi _ { \alpha } ^ { + } \right) , \qquad 0 \leq s \leq t _ { \alpha } . } \end{array}\tag{14}
$$

In particular, the quantity in Equation (10) at the transition point is

$$
\begin{array} { l } { { \displaystyle \bar { Q } _ { i } ( t _ { \alpha } ) = \theta _ { i } ( 0 ) ^ { 2 } \exp \bigl ( - 2 \lambda _ { i } \Phi _ { \alpha } ^ { + } \bigr ) } } \\ { { \displaystyle \qquad + h \nu _ { i } \exp \bigl ( - 2 \lambda _ { i } \Phi _ { \alpha } ^ { + } \bigr ) \int _ { 0 } ^ { t _ { \alpha } } \eta ( s ) ^ { 2 } \mathrm { d } s . } } \end{array}
$$

Since $\eta ^ { \prime } ( t ) = - c / T$ , substituting this identity and Equation (14) into the general expression for $M _ { \gamma } ^ { \prime } ( t )$ gives

$$
\begin{array} { l } { \displaystyle M _ { \widetilde { \gamma } } ^ { \prime } ( t _ { \alpha } ) = - \frac { c } { T } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \theta _ { i } ( 0 ) ^ { 2 } \exp \bigl ( - 2 \lambda _ { i } \Phi _ { \alpha } ^ { + } \bigr ) } \\ { \displaystyle \quad + { h \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \nu _ { i } \exp \bigl ( - 2 \lambda _ { i } \Phi _ { \alpha } ^ { + } \bigr ) } \left[ \eta _ { \alpha } ^ { 3 } - \frac { c } { T } \int _ { 0 } ^ { t _ { \alpha } } \eta ( s ) ^ { 2 } \mathrm { d } s \right] } \\ { \displaystyle = - \frac { c } { T } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \theta _ { i } ( 0 ) ^ { 2 } \exp \bigl ( - 2 \lambda _ { i } \Phi _ { \alpha } ^ { + } \bigr ) } \\ { \displaystyle \quad + { h \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { 2 } \nu _ { i } \exp \bigl ( - 2 \lambda _ { i } \Phi _ { \alpha } ^ { + } \bigr ) } \left[ \frac { \eta _ { \alpha } } { ( 1 - \alpha ) T } \int _ { 0 } ^ { t _ { \alpha } } \eta _ { \alpha } ^ { 2 } \mathrm { d } s - \frac { c } { T } \int _ { 0 } ^ { t _ { \alpha } } \eta ( s ) ^ { 2 } \mathrm { d } s \right] . } \end{array}\tag{15}
$$

Because η is decreasing, $\eta ( s ) \geq \eta _ { \alpha }$ for $0 \leq s \leq t _ { \alpha }$ . Therefore, the bracket in Equation (15) satisfies

$$
\begin{array} { r l r } & { } & { \eta _ { \alpha } ^ { 3 } - \cfrac { c } { T } \int _ { 0 } ^ { t _ { \alpha } } \eta ( s ) ^ { 2 } \mathrm { d } s \leq \eta _ { \alpha } ^ { 3 } - \cfrac { 1 } { 3 } \left( \eta _ { \operatorname* { m a x } } ^ { 3 } - \eta _ { \alpha } ^ { 3 } \right) } \\ & { } & \\ & { } & { \qquad = \cfrac { 4 } { 3 } \left( \eta _ { \operatorname* { m a x } } - c ( 1 - \alpha ) \right) ^ { 3 } - \cfrac { 1 } { 3 } \eta _ { \operatorname* { m a x } } ^ { 3 } < 0 , } \end{array}
$$

where the last inequality uses $\begin{array} { r } { \alpha < 1 - \frac { ( 1 - 4 ^ { - \frac { 1 } { 3 } } ) \eta _ { \mathrm { m a x } } } { c } } \end{array}$ . The first term of Equation (15) is nonpositive, and the positivity of $h , \lambda _ { i }$ , and $\nu _ { i }$ makes the second contribution strictly negative. Hence $M _ { \tilde { \gamma } } ^ { \prime } ( \dot { t } _ { \alpha } ) < 0 .$ contradicting Equation (13). □

## L PROOF OF THEOREM I.3

## L.1 DERIVATIONS OF THE MINIMIZATION OBJECTIVE

We begin from the final loss expression in Equation (8). Recall that only the shared block $l = 0$ and the MT-specific block $l = 1$ contribute to the final loss. Since the $l = 0$ term is independent of $\gamma _ { : }$ , we only need to minimize

$$
\begin{array} { r l r }   { \mathcal { L } _ { 1 } ( \pmb { \theta } ) = \underbrace { \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { ( 1 ) } \theta _ { i } ^ { ( 1 ) } ( 0 ) ^ { 2 } \exp { ( - 2 \lambda _ { i } ^ { ( 1 ) } \int _ { 0 } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t ) } } } \\ & { } & { \qquad + \underbrace { \frac { 1 } { 2 } \sum _ { i = 1 } ^ { d _ { 1 } } \lambda _ { i } ^ { ( 1 ) } h \nu _ { i } ^ { ( 1 ) } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } \exp { ( - 2 \lambda _ { i } ^ { ( 1 ) } \int _ { s } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t ) } } _ { \qquad \mathscr { L } _ { 1 } ^ { \mathrm { v a r } } ( \theta ) } \mathrm { d } s . } \end{array}
$$

We now average $\mathcal { L } _ { 1 } ( \pmb { \theta } )$ over the i.i.d. spectral pairs $\{ ( \lambda _ { i } ^ { ( 1 ) } , \nu _ { i } ^ { ( 1 ) } ) \} _ { i = 1 } ^ { d _ { 1 } }$ specified in Assumption I.1. In subsequent analysis, we will repeatedly use the Gamma-integral identity

$$
\int _ { 0 } ^ { \infty } \lambda ^ { r - 1 } e ^ { - x \lambda } \mathrm { d } \lambda = { \frac { \Gamma ( r ) } { x ^ { r } } } , \qquad r , x > 0 .
$$

Because the $\lambda _ { i } ^ { ( 1 ) }$ are identically distributed, the same spectral expectation multiplies every initialization coefficient. Thus

$$
\begin{array} { l } { \displaystyle \mathbb { E } \mathcal { L } _ { 1 } ^ { \mathrm { b i a s } } ( \theta ) = \frac { D _ { 1 } } { 2 } \left( \sum _ { i = 1 } ^ { d _ { 1 } } \theta _ { i } ^ { ( 1 ) } ( 0 ) ^ { 2 } \right) \int _ { 0 } ^ { \infty } \lambda ^ { \alpha _ { 1 } + 1 } \exp \left( - \left( K _ { 1 } + 2 \int _ { 0 } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t \right) \lambda \right) \mathrm { d } \lambda } \\ { \displaystyle \qquad = \frac { D _ { 1 } } { 2 } \left( \sum _ { i = 1 } ^ { d _ { 1 } } \theta _ { i } ^ { ( 1 ) } ( 0 ) ^ { 2 } \right) \frac { \Gamma \left( \alpha _ { 1 } + 2 \right) } { \left( K _ { 1 } + 2 \int _ { 0 } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t \right) ^ { \alpha _ { 1 } + 2 } } . } \end{array}\tag{16}
$$

We next turn to the variance term. We define the remaining accumulated learning rate after time s by

$$
R _ { \gamma } ( s ) : = \int _ { s } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t .
$$

For a representative spectral pair $( \lambda , \nu )$ , we first condition on λ. Then Assumption I.1 gives

$$
\begin{array} { r } { {  { \mathbb E } } _ { \nu | \lambda } \Big [ \lambda \nu e ^ { - 2 \lambda R _ { \gamma } ( s ) } \Big | \lambda \Big ] = \lambda e ^ { - 2 \lambda R _ { \gamma } ( s ) } {  { \mathbb E } } _ { \nu | \lambda } [ \nu \mid \lambda ] = F _ { 1 } \lambda ^ { \rho _ { 1 } + 1 } \exp \big [ - \big ( 2 R _ { \gamma } ( s ) + G _ { 1 } \big ) \lambda \big ] . } \end{array}
$$

We then average over $\lambda .$ Using $p _ { 1 } ( \lambda ) = D _ { 1 } \lambda ^ { \alpha _ { 1 } } e ^ { - K _ { 1 } \lambda }$ , the law of total expectation yields

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \lambda , \nu } \Big [ \lambda \nu e ^ { - 2 \lambda R _ { \gamma } ( s ) } \Big ] = \mathbb { E } _ { \lambda } \Big [ \mathbb { E } _ { \nu | \lambda } \Big [ \lambda \nu e ^ { - 2 \lambda R _ { \gamma } ( s ) } \Big | \lambda \Big ] \Big ] } \\ { \displaystyle = D _ { 1 } F _ { 1 } \int _ { 0 } ^ { \infty } \lambda ^ { \alpha _ { 1 } + \rho _ { 1 } + 1 } \exp \big [ - \big ( 2 R _ { \gamma } ( s ) + G _ { 1 } + K _ { 1 } \big ) \lambda \big ] \mathrm { d } \lambda . } \end{array}
$$

Tonelli’s theorem permits exchanging the expectation with respect to $\{ ( \lambda _ { i } ^ { ( 1 ) } , \nu _ { i } ^ { ( 1 ) } ) \} _ { i = 1 } ^ { d _ { 1 } }$ and the time integral, so

$$
\begin{array} { r l r } {  { \mathbb { E } [ \mathcal { L } _ { 1 } ^ { \mathrm { v a r } } ( \pmb { \theta } ) ] = \frac { h d _ { 1 } } { 2 } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } \mathbb { E } _ { \lambda , \nu } \Big [ \lambda \nu e ^ { - 2 \lambda R _ { \gamma } ( s ) } \Big ] \mathrm { d } s } } \\ & { } & { \quad = \frac { h d _ { 1 } D _ { 1 } F _ { 1 } \Gamma ( \alpha _ { 1 } + \rho _ { 1 } + 2 ) } { 2 ^ { \alpha _ { 1 } + \rho _ { 1 } + 3 } } \int _ { 0 } ^ { T } \frac { \eta ( s ) ^ { 2 } } { ( R _ { \gamma } ( s ) + ( K _ { 1 } + G _ { 1 } ) / 2 ) ^ { \alpha _ { 1 } + \rho _ { 1 } + 2 } } \mathrm { d } s . } \end{array}\tag{17}
$$

Here we require $\alpha _ { 1 } + \rho _ { 1 } + 2 > 0$ so that the spectral integral is finite at the origin. We define $C _ { 0 } : = \mathbb { E } \mathcal { L } _ { 0 } ( \mathbf { \dot { \theta } } )$

$$
C : = D _ { 1 } \left( \sum _ { i = 1 } ^ { d _ { 1 } } \theta _ { i } ^ { ( 1 ) } ( 0 ) ^ { 2 } \right) \Gamma ( \alpha _ { 1 } + 2 ) 2 ^ { - \alpha _ { 1 } - 3 } , \qquad C ^ { \prime } : = h d _ { 1 } D _ { 1 } F _ { 1 } \Gamma ( \alpha _ { 1 } + \rho _ { 1 } + 2 ) 2 ^ { - \alpha _ { 1 } - \rho _ { 1 } - 3 } .
$$

Combining Equations (16) and (17) gives the following minimization objective:

$$
\begin{array} { r l r } {  { \mathbb { E } [ \mathcal { I } ( \gamma , \eta ) ] = C _ { 0 } + \frac { C } { \Big ( K _ { 1 } / 2 + \int _ { 0 } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t \Big ) ^ { \alpha _ { 1 } + 2 } } } } \\ & { } & { + C ^ { \prime } \int _ { 0 } ^ { T } \frac { \eta ( s ) ^ { 2 } } { \Big ( ( K _ { 1 } + G _ { 1 } ) / 2 + \int _ { s } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t \Big ) ^ { \alpha _ { 1 } + \rho _ { 1 } + 2 } } \mathrm { d } s . } \end{array}\tag{18}
$$

Both schedule-dependent terms in Equation (18) are nonincreasing when MT data is added. Therefore, for $0 < \alpha < 1$ , an optimal schedule exhausts the MT data budget:

$$
\int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t = \alpha T .
$$

Equation (18), together with this active constraint and $0 \leq \gamma ( t ) \leq 1$ , is the functional optimized in the next subsection.

By a similar argument to that in Appendix J.1, the KKT conditions are necessary and sufficient for optimality, and the optimal data schedule is unique if it exists.

## L.2 KKT CONDITION

For compactness, define

$$
a : = \alpha _ { 1 } + 2 , \quad \quad b : = \alpha _ { 1 } + \rho _ { 1 } + 2 , \quad \quad G _ { \mathrm { e f f } } : = \frac { K _ { 1 } + G _ { 1 } } { 2 } ,
$$

and introduce the two schedule-dependent accumulated learning rates

$$
A _ { \gamma } : = \frac { K _ { 1 } } { 2 } + \int _ { 0 } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t , \qquad B _ { \gamma } ( s ) : = G _ { \mathrm { e f f } } + \int _ { s } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t .
$$

Thus, after dropping terms independent of γ, the objective in Equation (18) is

$$
J ( \gamma ) = C A _ { \gamma } ^ { - a } + C ^ { \prime } \int _ { 0 } ^ { T } \eta ( s ) ^ { 2 } B _ { \gamma } ( s ) ^ { - b } \mathrm { d } s .\tag{19}
$$

Let $v \in L ^ { \infty } ( [ 0 , T ] )$ be an admissible perturbation. Differentiating Equation (19) in the direction v gives

$$
\begin{array} { l } { { \displaystyle { \cal D } J ( \gamma ) [ v ] = - \frac { a C } { A _ { \gamma } ^ { a + 1 } } \int _ { 0 } ^ { T } \eta ( t ) v ( t ) \mathrm { d } t } } \\ { { \displaystyle ~ - b C ^ { \prime } \int _ { 0 } ^ { T } \frac { \eta ( s ) ^ { 2 } } { B _ { \gamma } ( s ) ^ { b + 1 } } \left( \int _ { s } ^ { T } \eta ( t ) v ( t ) \mathrm { d } t \right) \mathrm { d } s } . } \end{array}
$$

The second term is integrated over $\{ ( s , t ) : 0 \leq s \leq t \leq T \}$ . Applying Fubini’s theorem therefore yields

$$
D J ( \gamma ) [ v ] = - \int _ { 0 } ^ { T } M _ { \gamma } ( t ) v ( t ) \mathrm { d } t ,
$$

where the marginal value of MT data at time t is

$$
M _ { \gamma } ( t ) : = \eta ( t ) \left[ \frac { a C } { A _ { \gamma } ^ { a + 1 } } + b C ^ { \prime } \int _ { 0 } ^ { t } \frac { \eta ( s ) ^ { 2 } } { B _ { \gamma } ( s ) ^ { b + 1 } } \mathrm { d } s \right] .\tag{20}
$$

We now solve the optimization problem. Let $\kappa \geq 0$ be the multiplier of $\begin{array} { r } { \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t - \alpha T \leq 0 } \end{array}$ , and let $u _ { 0 } ( t ) , u _ { 1 } ( t ) \geq 0$ be the pointwise multipliers of $\dot { } - \gamma ( t ) \leq 0 \mathrm { a n d } \gamma ( t ) \dot { } - 1 \leq 0$ , respectively. The Lagrangian is

$$
\begin{array} { l } { \displaystyle \mathfrak { L } ( \gamma , \kappa , u _ { 0 } , u _ { 1 } ) : = J ( \gamma ) + \kappa \left( \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t - \alpha T \right) } \\ { \displaystyle - \int _ { 0 } ^ { T } u _ { 0 } ( t ) \gamma ( t ) \mathrm { d } t + \int _ { 0 } ^ { T } u _ { 1 } ( t ) \big ( \gamma ( t ) - 1 \big ) \mathrm { d } t . } \end{array}
$$

The KKT conditions are, almost everywhere on $[ 0 , T ]$

$$
\begin{array} { r l } & { 0 \leq \gamma ( t ) \leq 1 , \qquad \displaystyle \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t \leq \alpha T , } \\ & { \kappa \geq 0 , \qquad u _ { 0 } ( t ) \geq 0 , \qquad u _ { 1 } ( t ) \geq 0 , } \\ & { - M _ { \gamma } ( t ) + \kappa - u _ { 0 } ( t ) + u _ { 1 } ( t ) = 0 , } \\ & { \kappa \left( \displaystyle \int _ { 0 } ^ { T } \gamma ( t ) \mathrm { d } t - \alpha T \right) = 0 , \quad u _ { 0 } ( t ) \gamma ( t ) = 0 , \quad u _ { 1 } ( t ) \big ( 1 - \gamma ( t ) \big ) = 0 . } \end{array}\tag{21}
$$

(22)

Because $M _ { \gamma } ( t ) > 0$ whenever $\eta ( t ) > 0$ , adding MT data strictly decreases the objective. Hence, for $0 < \alpha < 1$ , the optimal data schedule must satisfy

$$
\int _ { 0 } ^ { T } \gamma ^ { \star } ( t ) \mathrm { d } t = \alpha T .
$$

Eliminating $u _ { 0 } ( t )$ and $u _ { 1 } ( t )$ from Equations (21) and (22) gives the equivalent conditions

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { M _ { \gamma ^ { \star } } ( t ) \leq \kappa , } & { \gamma ^ { \star } ( t ) = 0 , } \\ { M _ { \gamma ^ { \star } } ( t ) = \kappa , } & { 0 < \gamma ^ { \star } ( t ) < 1 , } \\ { M _ { \gamma ^ { \star } } ( t ) \geq \kappa , } & { \gamma ^ { \star } ( t ) = 1 . } \end{array} \right. } \end{array}
$$

Since $x \mapsto x ^ { - a }$ and $x \mapsto x ^ { - b }$ are convex on $( 0 , \infty )$ , J is convex in $\gamma .$ The feasible set is also convex, so these KKT conditions are sufficient as well as necessary.

## L.3 THE SHAPE OF THE DATA SCHEDULE WHEN $M _ { \gamma } ( t )$ IS A CONSTANT

In this section, we assume that $\eta ^ { \prime } ( t ) < 0$ for $t \in [ 0 , T ]$ , and $M _ { \gamma } ( t )$ is constant for $t \in [ t _ { 1 } , t _ { 2 } ]$ Differentiation of Equation (20) gives

$$
M _ { \gamma } ^ { \prime } ( t ) = \frac { \eta ^ { \prime } ( t ) } { \eta ( t ) } M _ { \gamma } ( t ) + \frac { b C ^ { \prime } \eta ( t ) ^ { 3 } } { B _ { \gamma } ( t ) ^ { b + 1 } } .\tag{23}
$$

Thus when $M _ { \gamma } ( t ) = \kappa$ is constant,

$$
\kappa \eta ^ { \prime } ( t ) + \frac { b C ^ { \prime } \eta ( t ) ^ { 4 } } { B _ { \gamma } ( t ) ^ { b + 1 } } = 0 ,
$$

which implies

$$
B _ { \gamma } ( t ) = \left( - \frac { b C ^ { \prime } \eta ( t ) ^ { 4 } } { \kappa \eta ^ { \prime } ( t ) } \right) ^ { 1 / ( b + 1 ) } .
$$

When $\begin{array} { r } { \eta ( t ) = \eta _ { \mathrm { m a x } } - \frac { c } { T } t , \eta ^ { \prime } ( t ) = - c / T } \end{array}$ is constant. Thus for a fixed time $t _ { 2 }$ , we have

$$
\frac { B _ { \gamma } ( t ) } { B _ { \gamma } ( t _ { 2 } ) } = \left( \frac { \eta ( t ) ^ { 4 } } { \eta ( t _ { 2 } ) ^ { 4 } } \right) ^ { 1 / ( b + 1 ) } ,
$$

and thus

$$
B _ { \gamma } ( t ) = B _ { \gamma } ( t _ { 2 } ) \left( \frac { \eta ( t ) ^ { 4 } } { \eta ( t _ { 2 } ) ^ { 4 } } \right) ^ { 1 / ( b + 1 ) } .\tag{24}
$$

We define

$$
B _ { 2 } : = B _ { \gamma } ( t _ { 2 } ) = G _ { \mathrm { e f f } } + \int _ { t _ { 2 } } ^ { T } \eta ( \tau ) \gamma ( \tau ) \mathrm { d } \tau , \qquad K _ { 2 } : = B _ { 2 } \eta ( t _ { 2 } ) ^ { - 4 / ( b + 1 ) } , \qquad \kappa _ { 2 } : = \frac { b C ^ { \prime } T } { c K _ { 2 } ^ { b + 1 } } .
$$

A direct corollary is that

$$
B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) = K _ { 2 } \eta ( t ) ^ { 4 / ( b + 1 ) } .
$$

Then we take the derivative of Equation (24) and substitute $B _ { \gamma } ^ { \prime } ( t ) = - \eta ( t ) \gamma ( t )$ . We get

$$
\gamma ( t ) = \frac { 4 c K _ { 2 } } { T ( b + 1 ) } \eta ( t ) ^ { - \frac { 2 ( b - 1 ) } { b + 1 } } .\tag{25}
$$

## L.4 PROOF OF THEOREM I.3

Proof. We define a family of data schedules $\gamma _ { t _ { 1 } , t _ { 2 } } ( t )$ parameterized by two time steps $0 < t _ { 1 } < t _ { 2 } <$ T. The expression of $\gamma _ { t _ { 1 } , t _ { 2 } } ( t )$ is given by

$$
\gamma _ { t _ { 1 } , t _ { 2 } } ( t ) : = \left\{ \begin{array} { l l } { 0 , } & { 0 \leq t < t _ { 1 } , } \\ { \frac { 4 c K _ { 2 } } { T ( b + 1 ) } \eta ( t ) ^ { - \frac { 2 ( b - 1 ) } { b + 1 } } , } & { t _ { 1 } \leq t < t _ { 2 } , } \\ { 1 , } & { t _ { 2 } \leq t \leq T . } \end{array} \right.
$$

We first need to verify that $\gamma _ { t _ { 1 } , t _ { 2 } } ( t )$ is a valid data schedule, i.e., $0 \leq \gamma _ { t _ { 1 } , t _ { 2 } } ( t ) \leq 1$

Lemma L.1. There exists a constant $T _ { 0 } > 0$ such that for any $T > T _ { 0 } , \gamma _ { t _ { 1 } , t _ { 2 } } ( t )$ is a valid data schedule for any $0 < t _ { 1 } < t _ { 2 } < T$

Proof. The assumptions $- 1 < \alpha _ { 1 } < 0$ and $0 < \rho _ { 1 } < 1$ imply $1 < b < 3$ . Therefore, $\gamma _ { t _ { 1 } , t _ { 2 } } ( t )$ is monotonically increasing in $t \in ( t _ { 1 } , t _ { 2 } )$ . So we only need to verify that the left limit satisfies $\gamma _ { t _ { 1 } , t _ { 2 } } ( t _ { 2 } ^ { - } ) \leq 1$ . By definition, we have

$$
\gamma _ { t _ { 1 } , t _ { 2 } } ( t _ { 2 } ^ { - } ) = \frac { 4 c B _ { 2 } } { T ( b + 1 ) \eta ( t _ { 2 } ) ^ { 2 } } .
$$

Lemma L.2. When $t \geq t _ { 2 } ,$ we have

$$
B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) \leq G _ { \mathrm { e f f } } + \frac { T \eta ( t ) ^ { 2 } } { 2 c } .
$$

Proof. We conclude the proof through a direct calculation:

$$
\begin{array} { l } { \displaystyle B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) = G _ { \mathrm { e f f } } + \int _ { t } ^ { T } \eta ( \tau ) \mathrm { d } \tau = G _ { \mathrm { e f f } } + \int _ { t } ^ { T } \left( \eta _ { \mathrm { m a x } } - \frac c T \tau \right) \mathrm { d } \tau } \\ { \displaystyle \qquad \le G _ { \mathrm { e f f } } + \int _ { t } ^ { c ^ { - 1 } T \eta _ { \mathrm { m a x } } } \left( \eta _ { \mathrm { m a x } } - \frac c T \tau \right) \mathrm { d } \tau } \\ { \displaystyle \qquad \le G _ { \mathrm { e f f } } + \frac { T \eta ( t ) ^ { 2 } } { 2 c } . } \end{array}
$$

By Lemma L.2, we have

$$
B _ { 2 } = B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t _ { 2 } ) \leq G _ { \mathrm { e f f } } + \frac { T \eta ( t _ { 2 } ) ^ { 2 } } { 2 c } .
$$

Substituting the upper bound on $B _ { 2 }$ into the expression for $\gamma _ { t _ { 1 } , t _ { 2 } } ( t _ { 2 } ^ { - } )$ gives an upper bound on $\gamma _ { t _ { 1 } , t _ { 2 } } ( t _ { 2 } ^ { - } )$ :

$$
\gamma _ { t _ { 1 } , t _ { 2 } } ( t _ { 2 } ^ { - } ) \leq \frac { 4 c } { T ( b + 1 ) \eta ( t _ { 2 } ) ^ { 2 } } \cdot \left( \frac { T \eta ( t _ { 2 } ) ^ { 2 } } { 2 c } + G _ { \mathrm { e f f } } \right) \leq \frac { 2 } { b + 1 } + \frac { 4 c G _ { \mathrm { e f f } } } { T ( b + 1 ) \eta ( T ) ^ { 2 } } .
$$

Thus a sufficient condition for $\gamma _ { t _ { 1 } , t _ { 2 } } ( t _ { 2 } ^ { - } ) \leq 1$ is

$$
\frac { 2 } { b + 1 } + \frac { 4 c G _ { \mathrm { e f f } } } { T ( b + 1 ) \eta ( T ) ^ { 2 } } \leq 1 ,
$$

which is equivalent to

$$
T \geq \frac { 4 c G _ { \mathrm { e f f } } } { ( b + 1 ) \left( 1 - \frac { 2 } { b + 1 } \right) \eta ( T ) ^ { 2 } } = \frac { 4 c G _ { \mathrm { e f f } } } { ( b - 1 ) \eta ( T ) ^ { 2 } } = : T _ { 0 } .
$$

Lemma L.3. There exists a constant $T _ { 1 } > 0$ such that for any $T > T _ { 1 } , M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { \prime \prime } ( t ) < 0$ for all $0 < t < t _ { 1 }$ and $M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { \prime \prime } ( t ) > 0$ for all $t _ { 2 } < t < T$

Proof. Equation (23) gives

$$
\begin{array} { r l } & { M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { \prime } ( t ) = \frac { \eta ^ { \prime } ( t ) } { \eta ( t ) } M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) + \frac { b C ^ { \prime } \eta ( t ) ^ { 3 } } { B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) ^ { b + 1 } } } \\ & { \qquad = \eta ^ { \prime } ( t ) \left[ \frac { a C } { A _ { \gamma } ^ { a + 1 } } + b C ^ { \prime } \int _ { 0 } ^ { t } \frac { \eta ( s ) ^ { 2 } } { B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( s ) ^ { b + 1 } } \mathrm { d } s \right] + \frac { b C ^ { \prime } \eta ( t ) ^ { 3 } } { B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) ^ { b + 1 } } } \\ & { \qquad = - \frac { c } { T } \frac { a C } { A _ { \gamma } ^ { a + 1 } } + b C ^ { \prime } \left[ - \frac { c } { T } \int _ { 0 } ^ { t } \frac { \eta ( s ) ^ { 2 } } { B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( s ) ^ { b + 1 } } \mathrm { d } s + \frac { \eta ( t ) ^ { 3 } } { B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) ^ { b + 1 } } \right] . } \end{array}\tag{26}
$$

Notice that $B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) \equiv B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t _ { 1 } )$ when $t \in [ 0 , t _ { 1 } ]$ , and we denote this constant by $B _ { 1 }$ . We write it explicitly as

$$
B _ { 1 } : = B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t _ { 1 } ) = G _ { \mathrm { e f f } } + \int _ { t _ { 1 } } ^ { T } \eta ( \tau ) \gamma _ { t _ { 1 } , t _ { 2 } } ( \tau ) \mathrm { d } \tau .
$$

Then we have

$$
M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { \prime \prime } ( t ) = - \frac { 4 b c C ^ { \prime } } { T } \frac { \eta ( t ) ^ { 2 } } { B _ { 1 } ^ { b + 1 } } < 0 , \qquad 0 < t < t _ { 1 } .
$$

When $\begin{array} { r } { t _ { 2 } < t < T , B _ { \gamma } ( t ) = G _ { \mathrm { e f f } } + \int _ { t } ^ { T } \eta ( \tau ) } \end{array}$ dτ and $\gamma ( t ) = 1$ . Direct differentiation gives

$$
M _ { \gamma } ^ { \prime \prime } ( t ) = \frac { b C ^ { \prime } \eta ( t ) ^ { 2 } } { B _ { \gamma } ( t ) ^ { b + 2 } } \left[ - \frac { 4 c } { T } B _ { \gamma } ( t ) + ( b + 1 ) \eta ( t ) ^ { 2 } \right] .
$$

By Lemma L.2, we have

$$
M _ { \gamma } ^ { \prime \prime } ( t ) \geq \frac { b C ^ { \prime } \eta ( t ) ^ { 2 } } { B _ { \gamma } ( t ) ^ { b + 2 } } \left[ - \frac { 4 c } { T } G _ { \mathrm { e f f } } + ( b - 1 ) \eta ( T ) ^ { 2 } \right] > 0
$$

when $\begin{array} { r } { T > \frac { 4 c G _ { \mathrm { e f f } } } { ( b - 1 ) \eta ( T ) ^ { 2 } } } \end{array}$

Theorem I.3 holds if and only if there exists $( t _ { 1 } , t _ { 2 } )$ such that $\gamma _ { t _ { 1 } , t _ { 2 } } ( t )$ satisfies the KKT conditions and the constraints for $\gamma$ . For this candidate, the pointwise KKT conditions and the active budget

require

$$
\begin{array} { c c } { { { \cal M } _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) \le \kappa _ { 2 } , } } & { { \qquad 0 \le t < t _ { 1 } , } } \\ { { { \cal M } _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) = \kappa _ { 2 } , } } & { { \qquad t _ { 1 } \le t < t _ { 2 } , } } \\ { { { \cal M } _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t ) \ge \kappa _ { 2 } , } } & { { \qquad t _ { 2 } \le t \le T , } } \\ { { 0 \le \gamma _ { t _ { 1 } , t _ { 2 } } ( t ) \le 1 , } } & { { \qquad 0 \le t \le T , } } \\ { { \displaystyle \int _ { 0 } ^ { T } \gamma _ { t _ { 1 } , t _ { 2 } } ( t ) \mathrm { d } t = \alpha T . } } & { { } } \end{array}
$$

By Lemma L.3 and Lemma L.1, sufficient conditions for the conditions above are

$$
\begin{array} { c } { { \displaystyle M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { \prime } \left( t _ { 1 } \right) = 0 , } } \\ { { \displaystyle \int _ { 0 } ^ { T } \gamma _ { t _ { 1 } , t _ { 2 } } ( t ) \mathrm { d } t = \alpha T . } } \end{array}
$$

Before we proceed, we define some additional notation. Let $\begin{array} { r } { x = { \frac { T - t } { T } } } \end{array}$ be the reversed-and-normalized time variable. Then $\eta ( t ) = \eta _ { \mathrm { m a x } } - c ( 1 - x )$ . We define

$$
\epsilon : = \eta ( T ) = \eta _ { \mathrm { m a x } } - c > 0 , \quad P ( x ) : = \eta ( t ) = \epsilon + c x , \quad L ( x ) : = \frac { 1 } { T } \int _ { t } ^ { T } \eta ( s ) \mathrm { d } s = \epsilon x + \frac { c } { 2 } x ^ { 2 } ,
$$

and write

$$
x _ { j } : = \frac { T - t _ { j } } { T } , \qquad j \in \{ 1 , 2 \} .
$$

We first rewrite Equation (26) in terms of the new variables:

$$
M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { \prime } ( t ) = \frac { b C ^ { \prime } } { B _ { 1 } ^ { b + 1 } } \left[ P ( x ) ^ { 3 } - c \int _ { x } ^ { 1 } P ( u ) ^ { 2 } \mathrm d u - \frac { c a C } { b C ^ { \prime } T } \frac { B _ { 1 } ^ { b + 1 } } { A _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { a + 1 } } \right] .
$$

Define

$$
F ( x ) : = P ( x ) ^ { 3 } - c \int _ { x } ^ { 1 } P ( u ) ^ { 2 } \mathrm { d } u .
$$

Then $M _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { \prime } ( t _ { 1 } ) = 0$ is equivalent to

$$
F ( x _ { 1 } ) = \frac { c a C } { b C ^ { \prime } T } \frac { B _ { 1 } ^ { b + 1 } } { A _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ^ { a + 1 } } .\tag{27}
$$

It remains to solve for x<sub>2</sub> using the budget. From Equation (25), we have

$$
\gamma _ { t _ { 1 } , t _ { 2 } } \left( T ( 1 - x ) \right) = \frac { 4 c B _ { 2 } } { T ( b + 1 ) } P ( x _ { 2 } ) ^ { - 4 / ( b + 1 ) } P ( x ) ^ { - 2 ( b - 1 ) / ( b + 1 ) } .
$$

Recall that $\begin{array} { r } { B _ { 2 } = B _ { \gamma _ { t _ { 1 } , t _ { 2 } } } ( t _ { 2 } ) = G _ { \mathrm { e f f } } + \int _ { t _ { 2 } } ^ { T } \eta ( \tau ) \mathrm { d } \tau = G _ { \mathrm { e f f } } + T L ( x _ { 2 } ) } \end{array}$ . Then we have

$$
\begin{array} { l } { \displaystyle \alpha = \frac { 1 } { T } \int _ { 0 } ^ { T } \gamma _ { t _ { 1 } , t _ { 2 } } ( t ) \mathrm { d } t } \\ { \displaystyle = \frac { 1 } { T } \left( \int _ { t _ { 1 } } ^ { t _ { 2 } } \gamma _ { t _ { 1 } , t _ { 2 } } ( t ) \mathrm { d } t + ( T - t _ { 2 } ) \right) } \\ { \displaystyle = x _ { 2 } + \frac { 4 \big ( L ( x _ { 2 } ) + T ^ { - 1 } G _ { \mathrm { e f f } } \big ) } { 3 - b } P ( x _ { 2 } ) ^ { - 4 / ( b + 1 ) } \left( P ( x _ { 1 } ) ^ { ( 3 - b ) / ( b + 1 ) } - P ( x _ { 2 } ) ^ { ( 3 - b ) / ( b + 1 ) } \right) . } \end{array}\tag{28}
$$

We denote $f _ { \epsilon , b } ( x _ { 1 } , x _ { 2 } )$ as the right-hand side of the above equation. The term involving $T ^ { - 1 } G _ { \mathrm { e f f } }$ is of order $T ^ { - 1 }$ , uniformly on compact subsets of $0 < x _ { 2 } < x _ { 1 } < 1$

Lemma L.4. Under the assumptions of Theorem I.3, the equation

$$
F ( x _ { 1 } ) = 0 ,
$$

has a unique solution $x _ { 1 } = x _ { \ast }$ <sub>⋆</sub> satisfying $0 < x _ { \star } < 1$ . This solution is independent ofT.

Proof. Since $P ^ { \prime } ( x ) = c$ and $P ( 1 ) = \eta _ { \mathrm { m a x } }$ , direct integration gives

$$
F ( x ) = P ( x ) ^ { 3 } - c \int _ { x } ^ { 1 } P ( u ) ^ { 2 } \mathrm { d } u = { \frac { 4 P ( x ) ^ { 3 } - \eta _ { \mathrm { m a x } } ^ { 3 } } { 3 } } .
$$

Consequently, $F ^ { \prime } ( x ) = 4 c P ( x ) ^ { 2 } > 0$ , so F has at most one zero. Solving $F ( x ) = 0$ gives

$$
x = x _ { \star } = \frac { 4 ^ { - 1 / 3 } \eta _ { \mathrm { \scriptsize { m a x } } } - \epsilon } { c } .
$$

The assumption $c > ( 1 - 4 ^ { - 1 / 3 } ) \eta _ { \mathrm { m a x } }$ implies $x _ { \star } > 0$ , while $4 ^ { - 1 / 3 } \eta _ { \mathrm { m a x } } - \epsilon < \epsilon$ implies $x _ { \star } < 1$ Thus the first equation has exactly one solution in (0, 1).

For some fixed $\begin{array} { r } { \frac { \alpha + x _ { \star } } { 2 } \leq x _ { 1 } \leq 1 } \end{array}$ , we have $f _ { \epsilon , b } ( x _ { 1 } , \alpha ) > \alpha .$ , and

$$
\begin{array} { l l l } { f _ { \epsilon , b } ( x _ { 1 } , x _ { 2 } ) \le x _ { 2 } + \displaystyle \frac { 4 \bigl ( \epsilon x _ { 2 } + \frac { c } { 2 } x _ { 2 } ^ { 2 } + T ^ { - 1 } G _ { \mathrm { e f f } } \bigr ) } { 3 - b } P ( x _ { 2 } ) ^ { - 4 / ( b + 1 ) } P ( x _ { 1 } ) ^ { ( 3 - b ) / ( b + 1 ) } } \\ { \le x _ { 2 } + \displaystyle \frac { 4 \bigl ( \epsilon x _ { 2 } + \frac { c } { 2 } x _ { 2 } + T ^ { - 1 } G _ { \mathrm { e f f } } \bigr ) } { 3 - b } P ( 0 ) ^ { - 4 / ( b + 1 ) } P ( 1 ) ^ { ( 3 - b ) / ( b + 1 ) } } \\ { \le \left( 1 + \displaystyle \frac { 4 \bigl ( \epsilon + \frac { c } { 2 } \bigr ) \eta _ { \mathrm { m a x } } ^ { ( 3 - b ) / ( b + 1 ) } } { ( 3 - b ) \epsilon ^ { 4 / ( b + 1 ) } } \right) x _ { 2 } + \displaystyle \frac { 4 G _ { \mathrm { e f f } } \eta _ { \mathrm { m a x } } ^ { ( 3 - b ) / ( b + 1 ) } } { T ( 3 - b ) \epsilon ^ { 4 / ( b + 1 ) } } . } \end{array}
$$

for all $0 < x _ { 2 } < x _ { \star }$ . We denote $\begin{array} { r } { D = \left( 1 + \frac { 4 \left( \epsilon + \frac { c } { 2 } \right) \eta _ { \mathrm { m a x } } ^ { ( 3 - b ) / ( b + 1 ) } } { ( 3 - b ) \epsilon ^ { 4 / ( b + 1 ) } } \right) \geq 1 , E = \frac { 4 G _ { \mathrm { e f f } } \eta _ { \mathrm { m a x } } ^ { ( 3 - b ) / ( b + 1 ) } } { ( 3 - b ) \epsilon ^ { 4 / ( b + 1 ) } } } \end{array}$ , and $\begin{array} { r } { T _ { 2 } = \frac { 3 E } { \alpha } } \end{array}$ . Then when $T > T _ { 2 }$ , we have

$$
f _ { \epsilon , b } ( x _ { 1 } , \frac { \alpha } { 3 D } ) \leq D \times \frac { \alpha } { 3 D } + \frac { E } { T _ { 2 } } \leq \frac { 2 \alpha } { 3 } < \alpha .
$$

Hence, for sufficiently large T and any fixed $\begin{array} { r } { x _ { 1 } \in \left[ \frac { \alpha + x _ { \star } } { 2 } , 1 \right] } \end{array}$ , there exists a unique solution $x _ { 2 } ( x _ { 1 } ) =$ Θ(1).

We now return to Equation (27). As $x _ { 2 } ( x _ { 1 } ) = \Theta ( 1 )$ for all $\begin{array} { r } { x _ { 1 } \in \left\lceil \frac { \alpha + x _ { \star } } { 2 } , 1 \right\rceil } \end{array}$ , we have

$$
\int _ { 0 } ^ { T } \eta ( s ) \gamma ( s ) \mathrm { d } s \geq \frac { 1 } { 2 } \eta ( t _ { 2 } ) ( T - t _ { 2 } ) \geq \frac { 1 } { 2 } \times x _ { 2 } \eta _ { \operatorname* { m a x } } \times x _ { 2 } T = \Theta ( T ) .
$$

Therefore, $A _ { \gamma _ { t _ { 1 } , t _ { 2 } } } = \Theta ( T )$ and $B _ { 1 } = \Theta ( T )$ . Since $b - a = \rho _ { 1 } < 1$ , the right-hand side of the smooth-pasting equation is $O ( T ^ { \rho _ { 1 } - 1 } ) = o ( 1 )$ , uniformly on $\begin{array} { r } { x _ { 1 } \in \left\lceil \frac { \alpha + x _ { \star } } { 2 } , 1 \right\rceil } \end{array}$ . So there exists $T _ { 3 } > 0$ such that for all $T > T _ { 3 }$ , the right-hand side of Equation (27) satisfies

$$
0 = F ( x ^ { * } ) < \frac { c a C } { b C ^ { \prime } T } \frac { B _ { 1 } ^ { b + 1 } } { A _ { \gamma _ { t _ { 1 } } , t _ { 2 } } ^ { a + 1 } } < F ( 1 ) .
$$

As $F ( x )$ is increasing, when $T > \operatorname* { m a x } \{ T _ { 0 } , T _ { 1 } , T _ { 2 } , T _ { 3 } \}$ , there exists a unique solution $x _ { 1 } ( T ) \in$ $( x _ { \star } , 1 )$ to Equation (27). This concludes our proof of the existence of a solution $( x _ { 1 } ( T ) , x _ { 2 } ( T ) )$ that satisfies the KKT conditions and the constraints.

## M NUMERICAL VALIDATION OF THE THEORETICAL MODEL

This section provides the details of the simulation and additional experiments supporting the theoretical predictions in Theorem 3.1. In Appendix M.1, we provide additional details on the simulation setup. In Appendix M.2, we provide a check on the validity of the SDE approximation. In Appendix M.3, we numerically solve the same scheduling problem analyzed in Section 3, thereby checking the closed form solution given in Theorem I.3. In Appendix M.4, we intuitively explain how the optimal data schedule changes with the learning rate schedule as shown in Theorems I.1 and I.3. In Appendix M.5, we verify that the data schedule solved by numerical methods satisfies the KKT condition.

## M.1 ADDITIONAL DETAILS ON THE SIMULATION EXPERIMENTS

Simulation setup. We use the same quadratic model, structure of PT and MT data, power-law spectrum, linearly decaying learning rate and constrained MT data budget as in the theorem. We use the closed form of the SDE approximation to calculate the expected validation loss. Regarding why this approximation is reasonable, see Appendix M.2 for more details.

Calculation of the validation loss. In our simulation, we use two methods to numerically calculate the expected validation loss of the SDE approximation: one is to sample all the eigenvalues of G and H according to their corresponding distributions, and the other one is to calculate the expected validation loss with respect to the distributions of eigenvalues directly. For each numerical setup and MT data budget, we first discretize the interval $[ 0 , T ]$ into $M = 4 0 0$ identical time bins $\{ [ \tau _ { i - 1 } , \tau _ { i } ] \} _ { i = 1 } ^ { M }$ and approximate $\gamma ( t )$ with $\gamma ( \tau _ { j } )$ for $\tau _ { j } \leq t \leq \tau _ { j + 1 }$

Hyperparameter selection. For the hyperparameters of the MPL spectral model, we take $\alpha _ { l } =$ $- 0 . 5 , \bar { K } _ { l } = 1 , \rho _ { l } = 0 . 5 , F _ { l } = 1 , G _ { l } = 1$ . The training horizon is set to $T = 2 0 0$ . We take $\eta _ { \mathrm { m a x } } = 1$ as the initial learning rate and initialize $\theta _ { i } ^ { l } = 1$ for all $l = 0 , 1 , 2$ and $1 \leq i \leq d _ { l }$ . We also set the discretization step size $h = 0 . 1$ . Note that by the derivations in Appendix J.1, the minimization objective is determined only by the contribution of knowledge specific to the MT domain, so the minimization objective depends on $d _ { 1 }$ and is independent of $d _ { 0 }$ and $d _ { 2 }$ . In our simulation experiments, we set $d _ { 1 } = 4 0 0 0$

Numerical optimization. We discretize $[ 0 , T ]$ into $K = 4 0 0$ equal-width bins and minimize the resulting objective $J ( y )$ over the feasible set

$$
\mathcal { Y } = \left\{ \boldsymbol { y } \in [ 0 , 1 ] ^ { K } : \sum _ { k = 1 } ^ { K } \boldsymbol { y } _ { k } = \alpha K \right\} .
$$

We minimize J over $\mathcal { V }$ using an accelerated projected-gradient method based on FISTA (Beck & Teboulle, 2009). Starting from a feasible point $\boldsymbol { y } ^ { ( 0 ) } \in \mathcal { V }$ , we initialize an auxiliary point $z ^ { ( 0 ) } = y ^ { ( 0 ) }$ At iteration n, we take a gradient step from $z ^ { ( n ) }$ and project the result onto $y { : }$

$$
\begin{array} { r } { v ^ { ( n ) } = z ^ { ( n ) } - \eta _ { n } \nabla J \Big ( z ^ { ( n ) } \Big ) , \qquad y _ { k } ^ { ( n + 1 ) } = \mathrm { c l i p } \Big ( v _ { k } ^ { ( n ) } - \tau _ { n } , 0 , 1 \Big ) . } \end{array}
$$

The scalar threshold $\tau _ { n }$ is determined by the bisection method to enforce $\begin{array} { r } { \sum _ { k } y _ { k } ^ { ( n + 1 ) } = \alpha K } \end{array}$ . We choose the step size $\eta _ { n }$ by backtracking. For each trial step size, we compute $y ^ { ( n + 1 ) }$ using the gradient and projection steps above and check whether

$$
J \Big ( y ^ { ( n + 1 ) } \Big ) \leq J \Big ( z ^ { ( n ) } \Big ) + \Big \langle \nabla J \Big ( z ^ { ( n ) } \Big ) , y ^ { ( n + 1 ) } - z ^ { ( n ) } \Big \rangle + \frac { \big \| y ^ { ( n + 1 ) } - z ^ { ( n ) } \big \| _ { 2 } ^ { 2 } } { 2 \eta _ { n } } .
$$

If the inequality fails, we halve $\eta _ { n }$ and recompute $y ^ { ( n + 1 ) }$ , repeating until the condition is satisfied. After an accepted step, the next line search is initialized with $1 . 2 \eta _ { n }$

The next auxiliary point incorporates momentum along the latest update direction:

$$
z ^ { ( n + 1 ) } = y ^ { ( n + 1 ) } + \beta _ { n } \left( y ^ { ( n + 1 ) } - y ^ { ( n ) } \right) ,
$$

where $\beta _ { n }$ is the standard FISTA momentum coefficient. Whenever $J ( y ^ { ( n + 1 ) } ) > J ( y ^ { ( n ) } )$ , we restart the momentum and set $z ^ { ( n + 1 ) } = y ^ { ( n + 1 ) }$ (O’Donoghue & Candès, 2015). Iterations terminate when the absolute objective change is below $1 0 ^ { - 1 5 }$ max $\{ 1 , | J ( y ^ { ( n ) } ) | \}$ , or after 20,000 iterations.

Calculation of the closed form schedule predicted by Theorem I.3. For a fixed parameter setting, the schedule in Theorem I.3 is determined by its two transition times $t _ { 1 }$ and $t _ { 2 }$ . We compute them from the two scalar conditions used in the proof. The first one, given in Equation (27), is a sufficient condition for satisfying the KKT conditions. The second is the active budget condition in Equation (28), which states that the total amount of MT data should equal the MT data budget. After solving these two equations, we calculate the interior part of the schedule using the expression in Theorem I.3, and set the schedule to zero before $t _ { 1 }$ and to one after $t _ { 2 }$

![](images/80f82c754783a5d4b98b631a697b1181845a8da46f36caf8572c338a968a3dde.jpg)

![](images/3ae842671af606d19d71d9dfb6e493dc5c127556b105de88b1ee1341b4be1ae6.jpg)  
Figure 31: Consistency of the SDE calculation, the exact discrete SGD second moment recursion, and Monte Carlo simulation for $\alpha = 0 . 2 5$ . Results are shown for a constant learning rate (left) and linear decay with $c = 0 . 8$ (right), across four representative data schedules. Error bars give two standard errors of the Monte Carlo mean.

## M.2 CHECKS OF THE SDE APPROXIMATION

To show the accuracy of the SDE approximation, for coordinate i in block l, we denote $m _ { i , k } ^ { ( l ) } : =$ $\mathbb { E } \left[ \left( \boldsymbol { \theta } _ { i , k } ^ { ( l ) } \right) ^ { 2 } \right]$ , where k indexes the training step. The discrete SGD update in Equation (4) gives

$$
m _ { i , k + 1 } ^ { ( l ) } = \left( 1 - \eta _ { k } a _ { i } ^ { ( l ) } ( t _ { k } ) \right) ^ { 2 } m _ { i , k } ^ { ( l ) } + \eta _ { k } ^ { 2 } \nu _ { i } ^ { ( l ) }\tag{29}
$$

through a direct calculation, where $t _ { k } = k h$ and $a _ { i } ^ { ( l ) } ( t )$ is defined in Equation (5). Iterating (29) yields the exact expected terminal loss of the discrete model. As a further check, we simulate the stochastic recursion itself and average the terminal loss across all the trajectories. We plot the results of the three methods in Figure 31. We also report the uncertainty of the Monte Carlo estimates using error bars, which indicate ±2 standard errors of the mean across simulated trajectories. Figure 31 shows that the results of the three methods are almost identical, which supports the accuracy of the SDE approximation.

## M.3 NUMERICAL VALIDATION OF THEORETICAL PREDICTIONS

In this section, we verify numerically that the closed form solution in Theorem I.3 agrees with the optimum of Equation (2).

Figure 32 shows that the solutions of the two numerical methods closely agree with the closed form solution given in Theorem I.3 under different MT data budgets. In particular, the numerical solutions recover the three stage structure predicted by the theorem: an initial PT only phase, an intermediate region in which the MT ratio increases continuously, and a final MT only phase. The agreement also extends to the locations of the two transition points. These results confirm that the schedule structure derived in the theorem is also observed when the expected validation loss is optimized directly.

## M.4 HOW THE OPTIMAL DATA SCHEDULE CHANGES WITH THE LEARNING RATE SCHEDULE

Theorems I.1 and I.3 together show that the shape of the optimal data schedule differs when the learning rate changes from a constant learning rate to a linearly decaying learning rate schedule. To test the predictions of Theorem I.1 and Theorem I.3 regarding how the optimal data schedule changes with the learning rate, we numerically optimize the expected validation loss over the data schedule. We use the same setting as in Theorem I.3, and the expectation is taken with respect to the gradient noise and the eigenvalues of G and H. We take two data budgets $\alpha \in \{ 0 . 1 , 0 . 2 5 \}$ and vary the learning rate decay strength c from a constant learning rate to a learning rate schedule with a strong linear decay.

![](images/9d1a667748895bb51a90e290e1dd477baae5c70db1144dfbb9f5be7f4a1e21f9.jpg)  
Figure 32: Numerical validation of Theorem I.3. Solid curves show the schedules predicted by the closed form solution, while markers show the optima obtained by directly optimizing the expected validation loss using different numerical methods. The two solutions agree for both MT data budgets considered.

The numerical results are shown in Figure 4. With a constant learning rate, the optimum is the two stage schedule from Theorem I.1: the first stage is to use only PT data, and the second stage is to use only MT data. The conclusion also holds for a learning rate schedule with a weak decay, namely $c = 0 . 2$ , in these instances. Once the decay becomes sufficiently strong, namely $c = 0 . 5$ and $c = 0 . 8 .$ the optimal data schedule changes to a three stage schedule. In particular, the optimal schedule begins with PT data only, then transitions to a mixture of PT and MT data, and ends with MT data only. During the second stage, the proportion of MT data gradually increases. These observations are consistent with the predictions of Theorem I.3.

## M.5 VERIFICATION OF THE KKT CONDITION OF THE OPTIMAL DATA SCHEDULE

To further demonstrate the optimality of the data schedule shown in Theorems I.1 to I.3, we additionally evaluate the KKT conditions given by Equation (10) and Equation (11).

In the left panel of Figure 33, we fix the data schedule to the two-stage schedule given by Theorem I.1, and select two learning rate schedules: a constant learning rate and a learning rate schedule with linear decay. Using Equation (10), we plot the ratio of $M _ { \gamma } ( t )$ to $M _ { \gamma } ( ( 1 - \alpha ) T )$ under each of the two learning rate schedules. In the right panel of Figure 33, we select a learning rate schedule with linear decay and its corresponding optimal data schedule. For easier visual comparison, we rescale $M _ { \gamma } ( t )$ by a positive constant. This rescaling only changes the overall scale of $\bar { M } _ { \gamma } ( t )$ , but does not change its shape over time. We then plot $\bar { M _ { \gamma } } ( t )$ divided by this rescaling factor.

If the 0-1 data schedule given by Theorem I.1 is optimal, then by Equation (11), we have $M _ { \gamma } ( t ) \leq \kappa$ for $0 \leq t \leq ( 1 - \alpha ) T$ , and $M _ { \gamma } ( t ) \geq \kappa$ for $( 1 - \dot { \alpha } ) T \leq t \leq \dot { T }$ . The left panel of Figure 33 shows that under a constant learning rate, the 0-1 data schedule satisfies this condition, but under a learning rate schedule with linear decay, the 0-1 data schedule violates this condition. This demonstrates that the 0-1 data schedule is optimal for a constant learning rate, but is not optimal for a linear decay learning rate schedule, which aligns with the predictions of Theorems I.1 and I.2.

The right panel of Figure 33 shows that $M _ { \gamma } ( t )$ also has a three-stage structure and is divided into the same three stages as the data schedule. Specifically, in the first stage where only PT data is used, $M _ { \gamma } ( t )$ is monotonically increasing; in the second stage where both PT and MT data are used, $M _ { \gamma } ( t )$ is constant; and in the third stage where only MT data is used, $M _ { \gamma } ( t )$ also monotonically increases. We take the Lagrange multiplier κ and the rescaling factor as the value of $M _ { \gamma } ( t )$ in the second stage. Then by Equation (11), the optimal data schedule given by Theorem I.3 satisfies the KKT conditions, which further confirms the optimality of the data schedule.

![](images/74b7839ffff12bb081d4dc8788824d4fdc3001a33ee8885d99edaaad49a74da1.jpg)

![](images/0eb2e978ccc89164f5b11d812ddc2b6cf4ae32de4eeb7431fb550731d56d7588.jpg)  
Figure 33: Numerical illustration of the KKT mechanism. Left: At the two stage schedule with $\alpha = 0 . 2 5$ , the marginal value of MT data is increasing under a constant learning rate, whereas linear decay with $c = 0 . 8$ violates the switching condition around $( 1 - \alpha ) T$ . Right: At the numerical optimum under linear decay, the normalized marginal value equals the KKT multiplier throughout the interior mixing phase and satisfies the appropriate inequalities on the two boundary phases.

## N OPTIMAL LEARNING RATE DECAYS AFTER MT DATA ENTERS

## N.1 SETUP AND FORMAL STATEMENT

We give the formal version of Theorem 3.2. Throughout this appendix, we assume Condition 3.1 and Assumption I.1 and use the dynamics in Equation (1). The MT-specific block is nonempty, and the initialization, dimensions, noise parameters, and spectral parameters are fixed independently of T. For a fixed MT budget $\alpha \in ( 0 , 1 )$ , set $L = \alpha T$ and

$$
\gamma ( t ) = \mathbb { 1 } _ { \{ t \geq T - L \} } .
$$

Fix $0 < m < M$ and $\Delta = M - m$ . We optimize over $W \in [ 0 , T ]$ in the SD family

$$
\eta _ { W } ( t ) = \left\{ \begin{array} { l l } { M , } & { 0 \leq t < T - W , } \\ { m + \Delta ( T - t ) / W , } & { T - W \leq t \leq T , } \end{array} \right. \quad W > 0 ,
$$

with $\eta _ { 0 } ( t ) = M$ . Let $J _ { T } ( W )$ denote the expected final MT loss, with expectation over both training noise and the spectral distribution.

For $l \in \{ 0 , 1 \}$ , we define

$$
a _ { l } : = \alpha _ { l } + 2 \in ( 1 , 2 ) , \qquad b _ { l } : = \alpha _ { l } + \rho _ { l } + 2 , \qquad g _ { l } : = \frac { K _ { l } + G _ { l } } { 2 } > 0 ,
$$

and

$$
B _ { l } : = \frac { D _ { l } \Gamma ( a _ { l } ) } { 2 ^ { a _ { l } + 1 } } \sum _ { i = 1 } ^ { d _ { l } } \theta _ { i } ^ { ( l ) } ( 0 ) ^ { 2 } , \qquad V _ { l } : = \frac { h d _ { l } D _ { l } F _ { l } \Gamma ( b _ { l } ) } { 2 ^ { b _ { l } + 1 } } .
$$

The indices 0 and 1 refer to shared and MT-specific features, respectively. In particular, $V _ { 1 } > 0 ;$ if the shared block is absent, its bias and variance contributions are zero.

Theorem N.1 (Optimal decay begins after MT data enters training). Under Condition 3.1 and Assumption I.1, consider the dynamics in Equation (1) with the TS data schedule $\gamma ( t ) = \mathbb { 1 } _ { \left\{ t \geq ( 1 - \alpha ) T \right\} }$ and the SD LRfamily η<sub>W</sub>, withfixed $0 < m < M$ . The initialization and all model parameters are fixed independently of T. Assume

$$
1 < b _ { 1 } < 2 , \qquad b _ { 0 } \geq b _ { 1 } , \qquad b _ { l } : = \alpha _ { l } + \rho _ { l } + 2 , \quad l \in \{ 0 , 1 \} .
$$

For every fixed $\alpha \in ( 0 , \alpha _ { c } )$ , where $\alpha _ { c }$ is given by Equation (37) with $b = b _ { 1 }$ and $r = m / M ,$ , there exists a finite $T _ { 0 } ,$ depending on α and the other fixed parameters, such that for every $T > \dot { T } _ { 0 } , J _ { T } ( W )$ attains its global minimum on $[ 0 , T ]$ , and every global minimizer $W ^ { * }$ satisfies

$$
0 < W ^ { * } < \alpha T , \qquad ( 1 - \alpha ) T < T - W ^ { * } < T .
$$

The second inequality states that LR decay begins strictly after MT entry. The proof is given in Appendix N.4.

## N.2 DERIVATION OF THE EXPECTED VALIDATION LOSS

In particular, $B _ { 1 } = C$ and $V _ { 1 } = C ^ { \prime }$ in Equation (18). Note that the contribution from the shared block was omitted from the expression of Equation (18) because it is independent of the data schedule. Here, we retain this contribution because the contribution from the shared block depends on the learning rate schedule. Following a derivation similar to that of Equation (18), we obtain

$$
\begin{array} { l } { { J _ { T } ( W ) = C _ { 0 } ^ { \prime } + \displaystyle \frac { B _ { 0 } } { \left( K _ { 0 } / 2 + \int _ { 0 } ^ { T } \eta ( t ) \mathrm { d } t \right) ^ { \alpha _ { 0 } + 2 } } } } \\ { { \displaystyle ~ + V _ { 0 } \int _ { 0 } ^ { T } \displaystyle \frac { \eta ( s ) ^ { 2 } } { \left( \left( K _ { 0 } + G _ { 0 } \right) / 2 + \int _ { s } ^ { T } \eta ( t ) \mathrm { d } t \right) ^ { \alpha _ { 0 } + \rho _ { 0 } + 2 } } \mathrm { d } s } } \\ { { \displaystyle ~ + \frac { B _ { 1 } } { \left( K _ { 1 } / 2 + \int _ { 0 } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t \right) ^ { \alpha _ { 1 } + 2 } } } } \\ { { \displaystyle ~ + V _ { 1 } \int _ { 0 } ^ { T } \displaystyle \frac { \eta ( s ) ^ { 2 } } { \left( \left( K _ { 1 } + G _ { 1 } \right) / 2 + \int _ { s } ^ { T } \eta ( t ) \gamma ( t ) \mathrm { d } t \right) ^ { \alpha _ { 1 } + \rho _ { 1 } + 2 } } \mathrm { d } s . } } \end{array}\tag{30}
$$

Furthermore, we define some additional notation for simplicity. We write

$$
A _ { W } ( u ) : = \int _ { 0 } ^ { u } \eta _ { W } ( T - v ) \mathrm { d } v = \left\{ \begin{array} { l l } { m u + \Delta u ^ { 2 } / ( 2 W ) , } & { 0 \leq u < W , } \\ { M u - \Delta W / 2 , } & { W \leq u \leq T . } \end{array} \right.\tag{31}
$$

The quantity $A _ { W } ( u )$ captures the cumulative effect from time $T - u$ to $T$ . For $0 < H \leq T , \beta > 1$ and $g > 0$ , write

$$
I _ { \beta , g } ( H , W ) : = \int _ { 0 } ^ { H } \frac { \eta _ { W } ( T - u ) ^ { 2 } } { [ g + A _ { W } ( u ) ] ^ { \beta } } \mathrm { d } u , \qquad P ( W ) : = \frac { \int _ { L } ^ { T } \eta _ { W } ( T - u ) ^ { 2 } \mathrm { d } u } { [ g _ { 1 } + A _ { W } ( L ) ] ^ { b _ { 1 } } } .\tag{32}
$$

Using this notation, we can rewrite the minimization objective as

$$
\begin{array} { c } { { J _ { T } ( W ) = C _ { 0 } ^ { \prime } + \displaystyle { \frac { B _ { 0 } } { [ K _ { 0 } / 2 + A _ { W } ( T ) ] ^ { a _ { 0 } } } + \frac { B _ { 1 } } { [ K _ { 1 } / 2 + A _ { W } ( L ) ] ^ { a _ { 1 } } } } } } \\ { { + V _ { 0 } I _ { b _ { 0 } , g _ { 0 } } ( T , W ) + V _ { 1 } I _ { b _ { 1 } , g _ { 1 } } ( L , W ) + V _ { 1 } P ( W ) . } } \end{array}\tag{33}
$$

## N.3 AUXILIARY LEMMAS

We first give a lemma on the expression of the budget threshold $\alpha _ { c } .$ . For $1 < b < 2$ , define $r : = m / \bar { M } \in ( 0 , 1 )$ and $\kappa : = ( b - \bar { 1 } ) / b \in ( 0 , 1 )$ ). We define some additional quantities that will be used in the main theorem and its proof:

$$
\delta _ { b } ( r ) : = \operatorname* { m i n } \left\{ \left( { \frac { 1 + r } { 2 } } \right) ^ { - b } , { \frac { 1 + r + r ^ { 2 } } { 3 r ^ { b } } } \right\} - \left[ 1 - { \frac { ( 1 - r ) \kappa } { 2 } } \right] ^ { - b } ,\tag{34}
$$

$$
H _ { b } ( r ) : = r ^ { - b } + \frac { b ( 1 - r ) ( 1 + r + r ^ { 2 } ) } { 6 r ^ { b + 1 } } ,\tag{35}
$$

$$
K _ { b } ( r ) : = \frac { ( 1 - r ) r ^ { 1 - b } \kappa ^ { 1 - b } } { ( b - 1 ) ( 2 - b ) } + \frac { r ^ { 2 - b } } { b - 1 } ,\tag{36}
$$

$$
\chi : = { \left\{ \begin{array} { l l } { V _ { 0 } / V _ { 1 } , } & { { \mathrm { i f ~ t h e ~ s h a r e d ~ b l o c k ~ i s ~ p r e s e n t ~ a n d ~ } } b _ { 0 } = b , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

$$
\alpha _ { c } : = \operatorname* { m i n } \left\{ \frac { 1 } { 2 } , \frac { \delta _ { b } ( r ) } { 2 [ H _ { b } ( r ) + ( 1 + \chi ) K _ { b } ( r ) ] } \right\} .\tag{37}
$$

Lemma N.1 (Positivity of the budget threshold). For every $1 < b < 2$ and $0 < r < 1$ , the quantity $\delta _ { b } ( r )$ defined in Equation (34) is strictly positive. Consequently, the budget threshold $\alpha _ { c }$ defined in Equation (37) is strictly positive.

Proof. To prove that $\delta _ { b } ( r ) > 0$ , we show that $\begin{array} { r } { \left( \frac { 1 + r } { 2 } \right) ^ { - b } - \left\lceil 1 - \frac { ( 1 - r ) \kappa } { 2 } \right\rceil ^ { - b } > 0 } \end{array}$ and $\textstyle { \frac { 1 + r + r ^ { 2 } } { 3 r ^ { b } } } -$ $\left\lceil 1 - \frac { ( 1 - r ) \kappa } { 2 } \right\rceil ^ { - b } > 0$ separately. We first show that $\begin{array} { r } { \left( \frac { 1 + r } { 2 } \right) ^ { - b } - \left\lceil 1 - \frac { ( 1 - r ) \kappa } { 2 } \right\rceil ^ { - b } > 0 } \end{array}$ . Notice that when $0 < r \bar { < } 1$ and $\kappa < 1$ , we have

$$
1 - { \frac { ( 1 - r ) \kappa } { 2 } } - { \frac { 1 + r } { 2 } } = { \frac { ( 1 - r ) ( 1 - \kappa ) } { 2 } } > 0 .
$$

Since $b > 0 .$ , this implies

$$
\left( \frac { 1 + r } { 2 } \right) ^ { - b } > \left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] ^ { - b } .
$$

Next we prove that $\begin{array} { r } { \frac { 1 + r + r ^ { 2 } } { 3 r ^ { b } } - \left\lceil 1 - \frac { ( 1 - r ) \kappa } { 2 } \right\rceil ^ { - b } > 0 } \end{array}$ . We use the weighted arithmetic–geometric mean inequality: for positive numbers $x , y$ and a weight $\lambda \in [ 0 , 1 ]$ , it states that

$$
( 1 - \lambda ) x + \lambda y \geq x ^ { 1 - \lambda } y ^ { \lambda } .
$$

This inequality follows directly from the concavity of the logarithm:

$$
\log \bigl ( ( 1 - \lambda ) x + \lambda y \bigr ) \geq ( 1 - \lambda ) \log x + \lambda \log y .
$$

We choose $x = 1 , y = r ,$ , and $\lambda = \kappa / 2 \in ( 0 , 1 )$ , and by the arithmetic–geometric mean inequality, we have

$$
1 - \frac { ( 1 - r ) \kappa } { 2 } = \left( 1 - \frac { \kappa } { 2 } \right) \cdot 1 + \frac { \kappa } { 2 } \cdot r \geq 1 ^ { 1 - \kappa / 2 } r ^ { \kappa / 2 } = r ^ { \kappa / 2 } .
$$

Since $b \kappa = b - 1$ , this yields

$$
\left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] ^ { b } \geq r ^ { b \kappa / 2 } = r ^ { ( b - 1 ) / 2 } > r ^ { b - 1 } .
$$

The last inequality holds because $0 < r < 1$ and $0 < ( b - 1 ) / 2 < b - 1$ . Consequently,

$$
\left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] ^ { - b } < r ^ { 1 - b } .
$$

On the other hand,

$$
\frac { 1 + r + r ^ { 2 } } { 3 } - r = \frac { ( 1 - r ) ^ { 2 } } { 3 } > 0 .
$$

We combine the two inequalities above and get that

$$
\frac { 1 + r + r ^ { 2 } } { 3 r ^ { b } } > \frac { r } { r ^ { b } } = r ^ { 1 - b } > \left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] ^ { - b } .
$$

Therefore, we prove that $\delta _ { b } ( r ) > 0$ . Since $H _ { b } ( r ) > 0 , K _ { b } ( r ) > 0$ , and $\chi \geq 0$ by their definitions, $\delta _ { b } ( r ) > 0$ implies that $\alpha _ { c } > 0$ □

Next, we give three lemmas on the properties of some terms appearing in Equation (33), namely $A _ { W } ( u ) , \bar { I _ { \beta , g } } ( H , W )$ , and $P ( W )$

Lemma N.2. Fix $T > 0 , 0 < H \leq T , \beta > 1$ , and $u > 0 .$ . Then $I _ { \beta , g } ( H , W )$ and $A _ { W } ( u )$ are nonincreasing in $W \in ( 0 , T ]$

Proof. We calculate $I _ { \beta , g } ( H , W )$ by changing the variable to $\begin{array} { r } { x = A _ { W } ( u ) = \int _ { 0 } ^ { u } \eta _ { W } ( T - v ) d v } \end{array}$ . Therefore, we have

$$
\mathrm { d } x = \eta _ { W } ( T - u ) \mathrm { d } u , \qquad \eta _ { W } ( T - u ) ^ { 2 } \mathrm { d } u = \eta _ { W } \bigl ( T - A _ { W } ^ { - 1 } ( x ) \bigr ) \mathrm { d } x .
$$

We substitute the above equations into Equation (32), and get

$$
\begin{array} { c l l } { { { \cal I } _ { \beta , g } ( H , W ) = \displaystyle \int _ { 0 } ^ { A _ { W } ( H ) } \frac { \eta _ { W } \bigl ( T - A _ { W } ^ { - 1 } ( x ) \bigr ) ^ { 2 } } { ( g + x ) ^ { \beta } } \frac { \mathrm { d } x } { \eta _ { W } \bigl ( T - A _ { W } ^ { - 1 } ( x ) \bigr ) } } } \\ { { { } } } \\ { { { } } } \end{array}
$$

Next, to calculate $I _ { \beta , g } ( H , W )$ , we need to calculate $A _ { W } ^ { - 1 } ( x )$ . When $0 \leq u < W$ , we have that $x = m u + { \Delta u ^ { 2 } } / { ( 2 W ) }$ , and thus $\begin{array} { r } { 0 \le x \le A _ { W } ( W ) = \frac { ( M + m ) W } { 2 } } \end{array}$ . A direct calculation gives

$$
\frac { 2 \Delta x } { W } = \frac { 2 m \Delta u } { W } + \frac { \Delta ^ { 2 } u ^ { 2 } } { W ^ { 2 } } , \qquad m ^ { 2 } + \frac { 2 \Delta x } { W } = \left( m + \frac { \Delta u } { W } \right) ^ { 2 } .
$$

Therefore, $u = ( W / \Delta ) ( \sqrt { m ^ { 2 } + 2 \Delta x / W } - m )$ . When $W \le u \le T , x = M u - \Delta W / 2 .$ so $u = ( x + \Delta W / 2 ) / \dot { M }$ . Thus

$$
A _ { W } ^ { - 1 } ( x ) = \left\{ \begin{array} { l l } { \displaystyle \frac { W } { \Delta } \left( \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W } } - m \right) , } & { 0 \leq x \leq A _ { W } ( W ) , } \\ { \displaystyle \frac { x + \Delta W / 2 } { M } , } & { A _ { W } ( W ) \leq x \leq A _ { W } ( T ) . } \end{array} \right.
$$

We substitute the expression of $A _ { W } ^ { - 1 } ( x )$ into the expression of the learning rate, and get

$$
\begin{array} { l l } { \eta _ { W } \bigl ( T - A _ { W } ^ { - 1 } ( x ) \bigr ) = \left\{ \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W } } , 0 \leq x \leq A _ { W } ( W ) , \right. } \\ { \qquad \left. \vphantom { \sum _ { w } ^ { 2 } } A _ { W } ( W ) \leq x \leq A _ { W } ( T ) , \right. } \\ { \qquad = \operatorname* { m i n } \left\{ M , \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W } } \right\} . } \end{array}
$$

For simplicity, we define

$$
\xi _ { W } ( x ) : = \operatorname* { m i n } \left\{ M , \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W } } \right\} ,\tag{38}
$$

and rewrite $I _ { \beta , g } ( H , W )$ as

$$
I _ { \beta , g } ( H , W ) = \int _ { 0 } ^ { A _ { W } ( H ) } \xi _ { W } ( x ) ( g + x ) ^ { - \beta } \mathrm { d } x .
$$

We now prove that $I _ { \beta , q } ( H , W )$ is nonincreasing in W. Given $0 < W _ { 1 } < W _ { 2 } \leq T$ , we need to show that ${ I _ { \beta , g } ( H , W _ { 1 } ) \geq I _ { \beta , g } ( H , W _ { 2 } ) }$ . We calculate the difference of the two integrals:

$$
\begin{array} { l } { \displaystyle I _ { \beta , g } ( H , W _ { 1 } ) - I _ { \beta , g } ( H , W _ { 2 } ) = \int _ { 0 } ^ { A _ { W _ { 1 } } ( H ) } \xi _ { W _ { 1 } } ( x ) ( g + x ) ^ { - \beta } \mathrm { d } x - \int _ { 0 } ^ { A _ { W _ { 2 } } ( H ) } \xi _ { W _ { 2 } } ( x ) ( g + x ) ^ { - \beta } \mathrm { d } x } \\ { \displaystyle \qquad = \int _ { 0 } ^ { A _ { W _ { 2 } } ( H ) } [ \xi _ { W _ { 1 } } ( x ) - \xi _ { W _ { 2 } } ( x ) ] ( g + x ) ^ { - \beta } \mathrm { d } x + \int _ { A _ { W _ { 2 } } ( H ) } ^ { A _ { W _ { 1 } } ( H ) } \xi _ { W _ { 1 } } ( x ) ( g + x ) ^ { - \beta } \mathrm { d } x . } \end{array}
$$

For each fixed $x \geq 0$ , by Equation (38), $\xi _ { W } ( x )$ is nonincreasing in W. For each fixed $u > 0 .$ , direct differentiation of Equation (31) gives

$$
\frac { \partial A _ { W } ( u ) } { \partial W } = \left\{ { - \Delta / 2 , } 0 < W \le u , \quad \right.
$$

which is negative. Hence, $A _ { W } ( u )$ is nonincreasing in $W .$ . Combining these two observations, we have $\xi _ { W _ { 1 } } ( x ) \ge \xi _ { W _ { 2 } } ( x )$ for all $x \geq 0$ and $A _ { W _ { 1 } } ( H ) \supset A _ { W _ { 2 } } ( H )$ . Therefore, both integrals in the last expression are nonnegative, which proves that $I _ { \beta , g } ( H , W )$ is nonincreasing in W.

Finally, we prove that the bias term in Equation (33) is nondecreasing in W. This follows directly since $\dot { B } _ { l } \ge \bar { 0 }$ for all $l , a _ { l } > 0$ , and $A _ { W } ( \grave { H } )$ decreases with W. 口

Lemma N.3. $I f b _ { 1 } > 1$ , then

$$
\operatorname* { m i n } _ { L \leq W \leq T } P ( W ) = \operatorname* { m i n } \{ P ( L ) , P ( T ) \} .
$$

Proof. For simplicity, we define some additional notation. We define $z = L / W \in [ \alpha , 1 ]$ , and consider W as a function of z. We further define $N ( z ) = \int _ { t } ^ { t } \eta _ { W } ( T - u ) ^ { 2 } \mathrm { d } u , Y ( z ) = g _ { 1 } + A _ { W } ( L )$ and $b = b _ { 1 }$ . Then by definition, $P ( W ) = P ( L / z ) = N ( z ) / Y ( z ) ^ { b }$ , and $A _ { W } ( L ) = L ( m + \Delta z / 2 )$

To prove our claim, we calculate the expressions for $N ( z )$ and $Y ( z )$ . By definition, $Y ( z ) : =$ $g _ { 1 } + L ( m + \Delta z / 2 )$ , and

$$
\begin{array} { r l } & { { \mathrm { N } } ( z ) : = \int _ { L } ^ { L } { \eta _ { L , 2 } ( T - u ) ^ { 2 } \mathrm { d } u } } \\ & { \quad - \int _ { L } ^ { L / 2 \lambda } { \left( m + \frac { \Delta z } { L } \right) ^ { 2 } \mathrm { d } u } + M ^ { 3 } ( T - L / z ) } \\ & { \quad = \int _ { L } ^ { L / 2 \lambda } { \left( m ^ { 2 } + { \frac { 2 m \lambda z } { L } } \right) ^ { 2 } \mathrm { d } u } + M ^ { 2 } ( T - L / z ) } \\ & { \quad = \left[ { \frac { { \lambda ^ { 2 } \eta _ { L } } } { L } } + { \frac { m \lambda \Delta z } { L } } + { \frac { { \lambda ^ { 2 } \eta _ { L } } } { { 3 2 L ^ { 2 } } } } \right] \mathrm { d } u + M ^ { 2 } ( T - L / z ) } \\ & { \quad = \left[ { \frac { { \lambda ^ { 2 } \eta _ { L } } } { L } } + { \frac { m \lambda \Delta z } { L } } \right] ^ { 2 } - { \frac { { \lambda ^ { 2 } \eta _ { L } } } { { 3 2 L ^ { 2 } } } } { \frac { { \lambda ^ { 2 } \eta _ { L } } } { { 3 2 L } } } + M ^ { 2 } ( T - L / z ) } \\ & { \quad = m ^ { 2 } L \left( { \frac { 1 } { z } } - 1 \right) + m \Delta L \left( { \frac { 1 } { z } } - \bar { z } \right) } \\ & { \quad \quad + { \frac { { \lambda ^ { 2 } \eta _ { L } } } { 3 } } \left( { \frac { 1 } { z } } - z ^ { 2 } \right) + M ^ { 2 } \left( T - { \frac { L } { z } } \right) } \\ & { \quad = M ^ { 2 } T + { \frac { L } { z } } \left( m ^ { 2 } + m \Delta + { \frac { { \Delta } ^ { 2 } z } { 3 } } - M ^ { 2 } \right) - L \left( m ^ { 2 } + m \Delta z + { \frac { { \Delta } ^ { 2 } z ^ { 2 } } { 3 } } \right) . } \end{array}
$$

We substitute $M = m + \Delta$ into the coefficient of $L / z$ to simplify its expression:

$$
\begin{array} { c } { { M ^ { 2 } - m ^ { 2 } - m \Delta - \displaystyle \frac { \Delta ^ { 2 } } { 3 } = ( m + \Delta ) ^ { 2 } - m ^ { 2 } - m \Delta - \displaystyle \frac { \Delta ^ { 2 } } { 3 } } } \\ { { = m \Delta + \displaystyle \frac { 2 \Delta ^ { 2 } } { 3 } = \displaystyle \frac { \Delta ( 3 m + 2 \Delta ) } { 3 } = \displaystyle \frac { \Delta ( 2 M + m ) } { 3 } . } } \end{array}
$$

We substitute this identity into the expression of $N ( z )$ and get

$$
N ( z ) = M ^ { 2 } T - L \left[ \frac { \Delta ( 2 M + m ) } { 3 z } + m ^ { 2 } + m \Delta z + \frac { \Delta ^ { 2 } z ^ { 2 } } { 3 } \right] .
$$

We observe that both $N ( z )$ and $Y ( z )$ are nonnegative, since $\begin{array} { r } { N ( z ) = \int _ { L } ^ { T } \eta _ { L / z } ( T - u ) ^ { 2 } } \end{array}$ du, and $Y ( z ) = g _ { 1 } + L ( m + \Delta z / 2 )$ . Furthermore, $N ( z )$ is concave in z, because we have

$$
N ^ { \prime \prime } ( z ) = - \frac { 2 L \Delta ( 2 M + m ) } { 3 z ^ { 3 } } - \frac { 2 L \Delta ^ { 2 } } { 3 } < 0
$$

by direct calculation. The function $Y ( z )$ is positive and affine, so $Y ^ { b }$ is convex. Let $p : =$ min $\{ N ( \alpha ) / Y ( \alpha ) ^ { b } , N ( 1 ) / Y ( 1 ) ^ { b } \} \ge 0$ . For $z = ( 1 - \lambda ) \alpha + \lambda \cdot 1$ with $0 \leq \lambda \leq 1$ , by the concavity of N and convexity of $\check { Y } ^ { b }$ , we have

$$
\begin{array} { l } { { N ( z ) \geq ( 1 - \lambda ) N ( \alpha ) + \lambda N ( 1 ) } } \\ { { \qquad \geq p \big [ ( 1 - \lambda ) Y ( \alpha ) ^ { b } + \lambda Y ( 1 ) ^ { b } \big ] \geq p Y ( z ) ^ { b } . } } \end{array}
$$

Thus $P ( L / z ) \geq p ,$ , and we conclude our proof.

Lemma N.4. Fix $0 < \kappa < 1$ , let $W ^ { \circ } = \kappa L ,$ , and suppose $L \leq H \leq T$ . For $\beta > 1$ and $g > 0$

$$
\begin{array} { l } { 0 \leq I _ { \beta , g } ( H , W ^ { \circ } ) - I _ { \beta , g } ( H , T ) } \\ { \leq \displaystyle \int _ { 0 } ^ { \infty } \Delta \operatorname* { m i n } \left. 1 , \displaystyle \frac { x } { m W ^ { \circ } } \right. ( g + x ) ^ { - \beta } \mathrm { d } x + \displaystyle \frac { m } { \beta - 1 } [ g + A _ { T } ( H ) ] ^ { 1 - \beta } . } \end{array}\tag{39}
$$

$H 1 < \beta < 2$ , this implies

$$
I _ { \beta , g } ( H , W ^ { \circ } ) - I _ { \beta , g } ( H , T ) \leq \frac { \Delta ( m W ^ { \circ } ) ^ { 1 - \beta } } { ( \beta - 1 ) ( 2 - \beta ) } + \frac { m ( m H ) ^ { 1 - \beta } } { \beta - 1 } .\tag{40}
$$

When $T \to \infty , I _ { \beta , g } ( H , W ^ { \circ } ) - I _ { \beta , g } ( H , T )$ can be bounded by

$$
\left\{ \begin{array} { l l } { O ( T ^ { 1 - \beta } ) , } & { 1 < \beta < 2 , } \\ { O ( T ^ { - 1 } \log T ) , } & { \beta = 2 , } \\ { O ( T ^ { - 1 } ) , } & { \beta > 2 . } \end{array} \right.\tag{41}
$$

Proof. Since $W ^ { \circ } \ = \ \kappa L \ < \ T$ , Lemma N.2 gives both $I _ { \beta , g } ( H , W ^ { \circ } ) \ : - \ : I _ { \beta , g } ( H , T ) \ : \geq \ : 0$ and $A _ { W ^ { \circ } } ( H ) \geq A _ { T } ( H )$ . We next derive the upper bound. For either $W = \tilde { W ^ { \circ } } \mathrm { o r } W = T$ , we first split $I _ { \beta , g } ( H , W )$ using $\xi _ { W } ( x ) = m + [ \bar { \xi _ { W } ( x ) } - m ]$

$$
I _ { \beta , g } ( H , W ) = m \int _ { 0 } ^ { A _ { W } ( H ) } ( g + x ) ^ { - \beta } \mathrm { d } x + \int _ { 0 } ^ { A _ { W } ( H ) } [ \xi _ { W } ( x ) - m ] ( g + x ) ^ { - \beta } \mathrm { d } x .
$$

Then we subtract the two decompositions and get

$$
\begin{array} { l } { { \displaystyle I _ { \beta , g } ( H , W ^ { \circ } ) - I _ { \beta , g } ( H , T ) } } \\ { { \displaystyle \quad = \int _ { 0 } ^ { A _ { W ^ { \circ } } ( H ) } [ \xi _ { W ^ { \circ } } ( x ) - m ] ( g + x ) ^ { - \beta } \mathrm { d } x } } \\ { { \displaystyle \qquad - \int _ { 0 } ^ { A _ { T } ( H ) } [ \xi _ { T } ( x ) - m ] ( g + x ) ^ { - \beta } \mathrm { d } x + m \int _ { A _ { T } ( H ) } ^ { A _ { W ^ { \circ } } ( H ) } ( g + x ) ^ { - \beta } \mathrm { d } x } } \\ { { \displaystyle \qquad \le \int _ { 0 } ^ { A _ { W ^ { \circ } } ( H ) } [ \xi _ { W ^ { \circ } } ( x ) - m ] ( g + x ) ^ { - \beta } \mathrm { d } x + m \int _ { A _ { T } ( H ) } ^ { A _ { W ^ { \circ } } ( H ) } ( g + x ) ^ { - \beta } \mathrm { d } x } . } \end{array}
$$

The inequality uses the fact that $\xi _ { W } ( x ) - m \ge 0$ for all $x \ge 0$ and for all $W$ , and the fact that $A _ { W ^ { \circ } } ( H ) \geq \dot { A } _ { T } ( H )$

We next bound the two integrals separately. For the second integral, since $\beta > 1$ , we have

$$
\begin{array} { l } { \displaystyle \int _ { A _ { T } ( H ) } ^ { A _ { W ^ { \circ } } ( H ) } ( g + x ) ^ { - \beta } \mathrm { d } x \leq \int _ { A _ { T } ( H ) } ^ { \infty } ( g + x ) ^ { - \beta } \mathrm { d } x } \\ { \displaystyle \qquad = \left[ \frac { ( g + x ) ^ { 1 - \beta } } { 1 - \beta } \right] _ { x = A _ { T } ( H ) } ^ { x = \infty } = \frac { [ g + A _ { T } ( H ) ] ^ { 1 - \beta } } { \beta - 1 } . } \end{array}
$$

Here the upper-endpoint value is zero because $1 - \beta < 0$ . Combining these two bounds proves Equation (39).

We now bound $\begin{array} { l } { \displaystyle { \int _ { 0 } ^ { A _ { W ^ { \circ } } ( H ) } [ \xi _ { W ^ { \circ } } ( x ) - m ] ( g + x ) ^ { - \beta } } } \end{array}$ dx. By the definition of $\xi _ { W ^ { \circ } }$ in Equation (38), we have

$$
\xi _ { W ^ { \circ } } ( x ) - m = \operatorname* { m i n } \left\{ \Delta , \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W ^ { \circ } } } - m \right\} .
$$

Note that

$$
\begin{array} { c } { \displaystyle { \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W ^ { \circ } } } - m = \frac { \left( \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W ^ { \circ } } } - m \right) \left( \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W ^ { \circ } } } + m \right) } { \sqrt { m ^ { 2 } + \frac { 2 \Delta x } { W ^ { \circ } } } + m } } } \\ { \displaystyle { = \frac { m ^ { 2 } + 2 \Delta x / W ^ { \circ } - m ^ { 2 } } { \sqrt { m ^ { 2 } + 2 \Delta x / W ^ { \circ } } + m } = \frac { 2 \Delta x / W ^ { \circ } } { \sqrt { m ^ { 2 } + 2 \Delta x / W ^ { \circ } } + m } } } \\ { \displaystyle { \leq \frac { 2 \Delta x / W ^ { \circ } } { 2 m } = \frac { \Delta x } { m W ^ { \circ } } . } } \end{array}
$$

Therefore

$$
0 \leq \xi _ { W ^ { \circ } } ( x ) - m \leq \Delta \operatorname* { m i n } \left\{ 1 , { \frac { x } { m W ^ { \circ } } } \right\} ,
$$

and we conclude the first part of the proof.

To conclude the second part of the proof, we further calculate a bound on $\begin{array} { r } { \int _ { 0 } ^ { \infty } \Delta \operatorname* { m i n } \left\{ 1 , \frac { x } { m W ^ { \circ } } \right\} \left( g + \right. } \end{array}$ $x ) ^ { - \beta }$ dx. For $1 < \beta < 2$ , we have

$$
\begin{array} { l l l } & { \displaystyle \int _ { 0 } ^ { \infty } \operatorname* { m i n } \{ 1 , x / a \} x ^ { - \beta } \mathrm { d } x = \int _ { 0 } ^ { a } \frac { x } { a } x ^ { - \beta } \mathrm { d } x + \int _ { a } ^ { \infty } x ^ { - \beta } \mathrm { d } x = \frac { 1 } { a } \int _ { 0 } ^ { a } x ^ { 1 - \beta } \mathrm { d } x + \int _ { a } ^ { \infty } x ^ { - \beta } \mathrm { d } x } \\ & { = a ^ { 1 - \beta } \left( \frac { 1 } { 2 - \beta } + \frac { 1 } { \beta - 1 } \right) = a ^ { 1 - \beta } \displaystyle \frac { ( \beta - 1 ) + ( 2 - \beta ) } { ( \beta - 1 ) ( 2 - \beta ) } = \frac { a ^ { 1 - \beta } } { ( \beta - 1 ) ( 2 - \beta ) } . } \end{array}
$$

We now apply this identity with $a = m W ^ { \circ }$ , and get that

$$
\int _ { 0 } ^ { \infty } \Delta \operatorname* { m i n } \left\{ 1 , \frac { x } { m W ^ { \circ } } \right\} ( g + x ) ^ { - \beta } \mathrm { d } x \leq \int _ { 0 } ^ { \infty } \Delta \operatorname* { m i n } \left\{ 1 , \frac { x } { m W ^ { \circ } } \right\} x ^ { - \beta } \mathrm { d } x = \frac { \Delta ( m W ^ { \circ } ) ^ { 1 - \beta } } { ( \beta - 1 ) ( 2 - \beta ) } .
$$

Also, we have $[ g + A _ { T } ( H ) ] ^ { 1 - \beta } \leq ( m H ) ^ { 1 - \beta }$ since $g + A _ { T } ( H ) \geq m H > 0$ and $1 - \beta < 0$ . We substitute the above two upper bounds into Equation (39) and get Equation (40).

Evaluating this last tail, and using $A _ { T } ( H ) \geq m H$ for the tail in Equation (39), gives the explicit bound

$$
\begin{array} { l } { { \displaystyle I _ { \beta , g } ( H , W ^ { \circ } ) - I _ { \beta , g } ( H , T ) } } \\ { { \le \displaystyle \frac \Delta { m W ^ { \circ } } \int _ { 0 } ^ { m W ^ { \circ } } x ( g + x ) ^ { - \beta } \mathrm { d } x + \Delta \int _ { m W ^ { \circ } } ^ { \infty } ( g + x ) ^ { - \beta } \mathrm { d } x + \frac m { \beta - 1 } ( g + m H ) ^ { 1 - \beta } } } \\ { { \le \displaystyle \frac \Delta { m W ^ { \circ } } \int _ { 0 } ^ { m W ^ { \circ } } x ( g + x ) ^ { - \beta } \mathrm { d } x + \frac \Delta { \beta - 1 } ( g + m W ^ { \circ } ) ^ { 1 - \beta } + \frac m { \beta - 1 } ( g + m H ) ^ { 1 - \beta } . } } \end{array}
$$

We now analyze the first integral for $\beta = 2$ and $\beta > 2$ separately. For $\beta = 2$ , we have

$$
\begin{array} { l l l } { { \displaystyle \int _ { 0 } ^ { m W ^ { \circ } } { x ( g + x ) } ^ { - 2 } \mathrm { d } x = \int _ { g } ^ { g + m W ^ { \circ } } ( y ^ { - 1 } - g y ^ { - 2 } ) \mathrm { d } y = \left[ \log y + \frac { g } { y } \right] _ { y = g } ^ { y = g + m W ^ { \circ } } } } \\ { { \displaystyle = \log \left( 1 + \frac { m W ^ { \circ } } { g } \right) + \frac { g } { g + m W ^ { \circ } } - 1 \leq \log \left( 1 + \frac { m W ^ { \circ } } { g } \right) . } } \end{array}
$$

For $\beta > 2 ,$ we have instead

$$
\begin{array} { r l } { \displaystyle \int _ { 0 } ^ { m W ^ { \diamond } } x ( g + x ) ^ { - \beta } \mathrm { d } x \leq \int _ { 0 } ^ { \infty } x ( g + x ) ^ { - \beta } \mathrm { d } x = \int _ { g } ^ { \infty } ( y ^ { 1 - \beta } - g y ^ { - \beta } ) \mathrm { d } y } & { } \\ { \displaystyle \qquad = \frac { g ^ { 2 - \beta } } { \beta - 2 } - \frac { g ^ { 2 - \beta } } { \beta - 1 } = \frac { g ^ { 2 - \beta } } { ( \beta - 1 ) ( \beta - 2 ) } . } & { } \end{array}
$$

Thus, with $W ^ { \circ } = \kappa \alpha T$ and fixed positive $\alpha , \kappa ,$ , the first term in the explicit bound is $O ( T ^ { - 1 } \log T )$ when $\beta = 2$ and $O ( T ^ { - 1 } )$ when $\beta > 2$ . The second term is $O ( T ^ { 1 - \beta } )$ , and the third has the same rate because $H \geq \alpha T$ . For $\dot { \beta } = 2$ these two terms are $O ( T ^ { - 1 } )$ ; for $\beta > 2$ they decay faster than $T ^ { - 1 }$ We combine these observations with Equation (40) and prove Equation (41). □

## N.4 PROOF OF THEOREM N.1

Proof of Theorem N.1. Throughout the proof, $b : = b _ { 1 } \in ( 1 , 2 ) , \kappa = ( b - 1 ) / b , \mathrm { a n d } W ^ { \circ } = \kappa L$

Step 1: $J _ { T } ( W )$ attains its minimum on $[ 0 , T ]$ . We only need to show that the function $J _ { T } ( W )$ is continuous on $[ 0 , T ]$ ]. Since

$$
m \le \eta _ { W } ( T - u ) \le M , \mathrm { ~ a n d } 0 \le \frac { \eta _ { W } ( T - u ) ^ { 2 } } { [ g + A _ { W } ( u ) ] ^ { \beta } } \le M ^ { 2 } g ^ { - \beta } ,
$$

by Equation (32) and Equation $( 3 3 ) , J _ { T } ( W )$ only depends on the integrals of bounded continuous functions of $\eta _ { W } ( u )$ . Thus we only need to show that all the integrals of bounded continuous functions of $\eta _ { W } ( u )$ are continuous in W. By the dominated convergence theorem, it suffices to show that for all $W \in [ 0 , T ] , \eta _ { W } ( u )$ is continuous in W for almost every $u \in [ 0 , T ]$

Fix T and let $\{ W _ { n } \} _ { n \ge 1 }$ be any sequence in $\left[ 0 , T \right]$ . Assume that $\{ W _ { n } \}$ has a limit $W \in [ 0 , T ]$ Note that when $W ^ { ' } \bar { \neq } ~ 0 , \eta _ { W _ { n } } ( T \bar { - } ~ u ) \ :  \ : \bar { \eta _ { W } } ( \bar { T } - u )$ for all $u \in [ 0 , T ] .$ , and when $W \bar { } = 0$ η<sub>Wn</sub> $( T - u ) \to \eta _ { 0 } ( T - u )$ when u $; \ne 0$ . This concludes the first part of the proof.

Step $2 \colon J _ { T } ( W )$ cannot attain its minimum on $\left[ L , T \right]$ . We take the reference point $W ^ { \circ } = \kappa L$ where $\kappa = ( b - 1 ) / b$ , and show that for all $W \in [ \dot { L } , T ] , ^ { \cdot } J _ { T } ( W ) > J _ { T } ( W ^ { \circ } )$ ). By Equation (33), we

have

$$
\begin{array} { l } { { J _ { T } ( W ) - J _ { T } ( W ^ { \circ } ) = \displaystyle \left( \frac { B _ { 0 } } { [ K _ { 0 } / 2 + A _ { W } ( T ) ] ^ { a _ { 0 } } } - \frac { B _ { 0 } } { [ K _ { 0 } / 2 + A _ { W ^ { \circ } } ( T ) ] ^ { a _ { 0 } } } \right) } } \\ { { \displaystyle ~ + \left( \frac { B _ { 1 } } { [ K _ { 1 } / 2 + A _ { W } ( L ) ] ^ { a _ { 1 } } } - \frac { B _ { 1 } } { [ K _ { 1 } / 2 + A _ { W ^ { \circ } } ( L ) ] ^ { a _ { 1 } } } \right) } } \\ { { \displaystyle ~ + V _ { 1 } \left[ P ( W ) - P ( W ^ { \circ } ) \right] + V _ { 1 } \left[ I _ { b _ { 1 } , g _ { 1 } } ( L , W ) - I _ { b _ { 1 } , g _ { 1 } } ( L , W ^ { \circ } ) \right] } } \\ { { \displaystyle ~ + V _ { 0 } \left[ I _ { b _ { 0 } , g _ { 0 } } ( T , W ) - I _ { b _ { 0 } , g _ { 0 } } ( T , W ^ { \circ } ) \right] } . } \end{array}\tag{42}
$$

Note that $A _ { W } ( T ) \leq A _ { W ^ { \circ } } ( T )$ and $A _ { W } ( L ) \leq A _ { W ^ { \circ } } ( L )$ for all $W$ by Lemma $\mathrm { N } . 2 ,$ and by Lemma $\mathrm { N } . 3 ,$ we have $P ( W ) \geq \operatorname* { m i n } \{ P ( L ) , P ( T ) \}$ . So

$$
J _ { T } ( W ) - J _ { T } ( W ^ { \circ } ) \geq V _ { 1 } \left[ \operatorname* { m i n } \{ P ( L ) , P ( T ) \} - P ( W ^ { \circ } ) - D _ { 1 } - \frac { V _ { 0 } } { V _ { 1 } } D _ { 0 } \right] ,
$$

where $D _ { 1 } : = I _ { b _ { 1 } , g _ { 1 } } ( L , W ^ { \circ } ) - I _ { b _ { 1 } , g _ { 1 } } ( L , T ) , D _ { 0 } : = I _ { b _ { 0 } , g _ { 0 } } ( T , W ^ { \circ } ) - I _ { b _ { 0 } , g _ { 0 } } ( T , T )$ . Therefore, the second part of the proof is complete if we can show that min $\{ P ( L ) , P ( T ) \} - P ( W ^ { \circ } ) - D _ { 1 } - \frac { V _ { 0 } } { V _ { 1 } } D _ { 0 } >$ 0 for all sufficiently large $T .$ To prove this, we calculate $P ( L ) , P ( T ) , P ( W ^ { \circ } ) , D _ { 1 }$ and $D _ { 0 }$ in turn. We define

$$
S _ { T } : = \frac { M ^ { 2 } T } { ( M L ) ^ { b } } = M ^ { 2 - b } \alpha ^ { - b } T ^ { 1 - b } .
$$

We first compute $P ( L ) , P ( T )$ and $P ( W ^ { \circ } )$ . For $W = L$ , we have $\eta _ { W ^ { \circ } } ( T - u ) = M$ for all $u \in [ L , T ]$ , so the numerator of $P ( L )$ is given by

$$
\int _ { L } ^ { T } \eta _ { L } ( T - u ) ^ { 2 } \mathrm { d } u = M ^ { 2 } ( T - L ) = M ^ { 2 } T ( 1 - \alpha ) ,
$$

and by Equation (31), the denominator of $P ( L )$ is

$$
\left( g _ { 1 } + A _ { L } ( L ) \right) ^ { b } = \left( g _ { 1 } + M L - \frac { \Delta L } { 2 } \right) ^ { b } = \left( g _ { 1 } + M L \frac { 1 + r } { 2 } \right) ^ { b } .
$$

The numerator for $P ( W ^ { \circ } )$ is the same as that of $P ( L )$ , since again $\eta _ { W ^ { \circ } } ( T - u ) = M$ for all $u \in [ L , T ]$ . As for the denominator of $P ( W ^ { \circ } )$ ), we have

$$
\left( g _ { 1 } + A _ { W ^ { \circ } } ( L ) \right) ^ { b } = \left( g _ { 1 } + M L - \frac { \Delta W ^ { \circ } } { 2 } \right) ^ { b } = \left( g _ { 1 } + M L \left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] \right) ^ { b } .
$$

For $W = T$ , we have

$$
\int _ { L } ^ { T } \eta _ { T } ( T - u ) ^ { 2 } \mathrm { d } u = \int _ { L } ^ { T } \left( m + \frac { \Delta u } { T } \right) ^ { 2 } \mathrm { d } u = M ^ { 2 } T \int _ { \alpha } ^ { 1 } [ r + ( 1 - r ) v ] ^ { 2 } \mathrm { d } v .
$$

We denote $\begin{array} { r } { n ( \alpha ) = \int _ { \alpha } ^ { 1 } [ r + ( 1 - r ) v ] ^ { 2 } } \end{array}$ dv for simplicity. Through a direct calculation, we have

$$
\begin{array} { l } { { n ( \alpha ) = \left[ r ^ { 2 } v + r ( 1 - r ) v ^ { 2 } + \frac { ( 1 - r ) ^ { 2 } v ^ { 3 } } { 3 } \right] _ { v = \alpha } ^ { v = 1 } } } \\ { { \mathrm { ~ } = r ^ { 2 } ( 1 - \alpha ) + r ( 1 - r ) ( 1 - \alpha ^ { 2 } ) + \frac { ( 1 - r ) ^ { 2 } } { 3 } ( 1 - \alpha ^ { 3 } ) } } \\ { { \mathrm { ~ } = \displaystyle \frac { 1 + r + r ^ { 2 } } { 3 } - r ^ { 2 } \alpha - r ( 1 - r ) \alpha ^ { 2 } - \frac { ( 1 - r ) ^ { 2 } \alpha ^ { 3 } } { 3 } . } } \end{array}
$$

The last equality uses $r ^ { 2 } + r ( 1 - r ) + ( 1 - r ) ^ { 2 } / 3 = ( 1 + r + r ^ { 2 } ) / 3 . { \mathrm { A l s o } }$

$$
\left( g _ { 1 } + A _ { T } ( L ) \right) ^ { b } = \left( g _ { 1 } + M L \left[ r + \frac { ( 1 - r ) \alpha } { 2 } \right] \right) ^ { b } .
$$

Combining the above calculations, we have

$$
\begin{array} { r l } & { \frac { P ( L ) } { S _ { T } } = \frac { 1 - \alpha } { [ g _ { 1 } / ( M L ) + ( 1 + r ) / 2 ] ^ { b } } , } \\ & { \frac { P ( T ) } { S _ { T } } = \frac { n ( \alpha ) } { [ g _ { 1 } / ( M L ) + r + ( 1 - r ) \alpha / 2 ] ^ { b } } , } \\ & { \frac { P ( W ^ { \circ } ) } { S _ { T } } = \frac { 1 - \alpha } { [ g _ { 1 } / ( M L ) + 1 - ( 1 - r ) \kappa / 2 ] ^ { b } } , } \end{array}
$$

and consequently,

$$
\frac { \operatorname * { m i n } \{ P ( L ) , P ( T ) \} - P ( W ^ { \circ } ) } { S _ { T } } \longrightarrow G _ { b } ( \alpha , r ) ,\tag{43}
$$

where

$$
G _ { b } ( \alpha , r ) : = \operatorname* { m i n } \left\{ ( 1 - \alpha ) \left( \frac { 1 + r } { 2 } \right) ^ { - b } , \frac { n ( \alpha ) } { [ r + ( 1 - r ) \alpha / 2 ] ^ { b } } \right\} - ( 1 - \alpha ) \left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] ^ { - b } .
$$

A direct observation is that $G _ { b } ( 0 , r ) = \delta _ { b } ( r )$ . We next control the difference $G _ { b } ( \alpha , r ) - G _ { b } ( 0 , r )$ for sufficiently small α. Let $f ( \alpha )$ denote the second entry in the minimum:

$$
f ( \alpha ) = n ( \alpha ) [ r + ( 1 - r ) \alpha / 2 ] ^ { - b } .
$$

Through a direct calculation, we have

$$
\begin{array} { c } { { n ^ { \prime } ( \alpha ) = - r ^ { 2 } - 2 r ( 1 - r ) \alpha - ( 1 - r ) ^ { 2 } \alpha ^ { 2 } } } \\ { { { } } } \\ { { = - [ r + ( 1 - r ) \alpha ] ^ { 2 } . } } \end{array}
$$

Thus $n ( \alpha )$ is nonincreasing on $[ 0 , 1 ]$ . Since $n ( 0 ) = ( 1 + r + r ^ { 2 } ) / 3$ and $n ( 1 ) = 0$ , we have

$$
0 \leq n ( \alpha ) \leq { \frac { 1 + r + r ^ { 2 } } { 3 } } , \qquad 0 \leq \alpha \leq 1 .
$$

We next differentiate $f ( \alpha )$ , substitute the derivative of $n ( \alpha )$ , and get

$$
\begin{array} { l } { { f ^ { \prime } ( \alpha ) = n ^ { \prime } ( \alpha ) [ r + ( 1 - r ) \alpha / 2 ] ^ { - b } - \displaystyle \frac { b ( 1 - r ) } { 2 } n ( \alpha ) [ r + ( 1 - r ) \alpha / 2 ] ^ { - b - 1 } } } \\ { { \mathrm { } = - \displaystyle \frac { [ r + ( 1 - r ) \alpha ] ^ { 2 } } { [ r + ( 1 - r ) \alpha / 2 ] ^ { b } } - \displaystyle \frac { b ( 1 - r ) n ( \alpha ) } { 2 [ r + ( 1 - r ) \alpha / 2 ] ^ { b + 1 } } . } } \end{array}
$$

Note that both terms in the last line are nonpositive. Consequently,

$$
\begin{array} { l } { | f ^ { \prime } ( \alpha ) | = \displaystyle \frac { [ r + ( 1 - r ) \alpha ] ^ { 2 } } { [ r + ( 1 - r ) \alpha / 2 ] ^ { b } } + \frac { b ( 1 - r ) n ( \alpha ) } { 2 [ r + ( 1 - r ) \alpha / 2 ] ^ { b + 1 } } \le \frac { 1 } { r ^ { b } } + \frac { b ( 1 - r ) } { 2 r ^ { b + 1 } } \frac { 1 + r + r ^ { 2 } } { 3 } } \\ { = r ^ { - b } + \displaystyle \frac { b ( 1 - r ) ( 1 + r + r ^ { 2 } ) } { 6 r ^ { b + 1 } } = H _ { b } ( r ) , } \end{array}
$$

where the inequality holds since $r \leq r + ( 1 - r ) \alpha \leq 1 , r + ( 1 - r ) \alpha / 2 \geq r .$ and $0 \leq n ( \alpha ) \leq$ $( 1 + r + r ^ { 2 } ) / 3$ . Recall that $H _ { b } ( r )$ is defined as in Equation (35). In particular, $f ^ { \prime } ( \alpha ) \geq - H _ { b } ( r )$ for every $\alpha \in [ 0 , 1 ]$ . Therefore, by the fundamental theorem of calculus, we have

$$
f ( \alpha ) - f ( 0 ) = \int _ { 0 } ^ { \alpha } f ^ { \prime } ( s ) \mathrm { d } s \geq - \alpha H _ { b } ( r ) .
$$

The first entry satisfies the same lower bound, because $( ( 1 + r ) / 2 ) ^ { - b } \leq r ^ { - b } \leq H _ { b } ( r ) \quad$

$$
( 1 - \alpha ) \left( \frac { 1 + r } { 2 } \right) ^ { - b } \geq \left( \frac { 1 + r } { 2 } \right) ^ { - b } - \alpha H _ { b } ( r ) .
$$

We apply the lower bounds of $( 1 - \alpha ) ( ( 1 + r ) / 2 ) ^ { - b }$ and $f ( \alpha )$ and obtain the following lower bound:

$$
\operatorname* { m i n } \left\{ \left( 1 - \alpha \right) \left( \frac { 1 + r } { 2 } \right) ^ { - b } , f ( \alpha ) \right\} \geq \operatorname* { m i n } \left\{ \left( \frac { 1 + r } { 2 } \right) ^ { - b } , f ( 0 ) \right\} - \alpha H _ { b } ( r ) .
$$

Finally, note that

$$
- ( 1 - \alpha ) \left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] ^ { - b } \geq - \left[ 1 - \frac { ( 1 - r ) \kappa } { 2 } \right] ^ { - b } .
$$

We combine the two inequalities above, and therefore, we have

$$
{ \cal G } _ { b } ( \alpha , r ) \geq \delta _ { b } ( r ) - { \cal H } _ { b } ( r ) \alpha , \qquad 0 \leq \alpha \leq 1 .\tag{44}
$$

Here, we recall that $f ( 0 ) = ( 1 + r + r ^ { 2 } ) / ( 3 r ^ { b } )$ , and $\delta _ { b } ( r )$ is defined as in Equation (34).

We next bound $D _ { 0 }$ and $D _ { 1 }$ . For $D _ { 1 }$ , we take $H = L$ in Equation (40), and get that

$$
\begin{array} { r l } & { D _ { 1 } \leq \frac { \Delta ( m W ^ { 0 } ) ^ { 1 - b } } { ( b - 1 ) ( 2 - b ) } + \frac { m ( m L ) ^ { 1 - b } } { b - 1 } } \\ & { \quad = \frac { ( 1 - r ) M ( r M \kappa \alpha T ) ^ { 1 - b } } { ( b - 1 ) ( 2 - b ) } + \frac { r M ( r M \alpha T ) ^ { 1 - b } } { b - 1 } } \\ & { \quad = M ^ { 2 - b } \alpha ^ { 1 - b } T ^ { 1 - b } \left[ \frac { ( 1 - r ) r ^ { 1 - b } \kappa ^ { 1 - b } } { ( b - 1 ) ( 2 - b ) } + \frac { r ^ { 2 - b } } { b - 1 } \right] } \\ & { \quad = \alpha S _ { T } K _ { b } ( r ) . } \end{array}
$$

Here we used $S _ { T } = M ^ { 2 - b } \alpha ^ { - b } T ^ { 1 - b }$ , and $K _ { b } ( r )$ is defined as in Equation (36).

For $D _ { 0 } .$ , we distinguish cases according to the value of $b _ { 0 }$ relative to $b .$ When $b _ { 0 } = b ,$ , we apply Equation (40) with $H = T$ instead:

$$
\begin{array} { l } { \displaystyle { D _ { 0 } \leq \frac { ( 1 - r ) M ( r M \kappa \alpha T ) ^ { 1 - b } } { ( b - 1 ) ( 2 - b ) } + \frac { r M ( r M T ) ^ { 1 - b } } { b - 1 } , } } \\ { \displaystyle { \frac { D _ { 0 } } { S _ { T } } \leq \alpha \frac { ( 1 - r ) r ^ { 1 - b } \kappa ^ { 1 - b } } { ( b - 1 ) ( 2 - b ) } + \alpha ^ { b } \frac { r ^ { 2 - b } } { b - 1 } } } \\ { \displaystyle { \quad \leq \alpha \left[ \frac { ( 1 - r ) r ^ { 1 - b } \kappa ^ { 1 - b } } { ( b - 1 ) ( 2 - b ) } + \frac { r ^ { 2 - b } } { b - 1 } \right] = \alpha K _ { b } ( r ) } . } \end{array}
$$

The last step uses $0 < \alpha < 1$ and $b > 1$ , which imply $\alpha ^ { b } = \alpha \alpha ^ { b - 1 } \leq \alpha .$

If $b _ { 0 } > b ,$ Equation (41) applies with $\beta = b _ { 0 } , g = g _ { 0 } , H = T$ , and $W ^ { \circ } = \kappa \alpha T$ . Its constants can depend on the fixed parameters, including α and $\kappa ,$ but not on T. To see the normalization explicitly, $1 / S _ { T }$ is a fixed positive multiple of $T ^ { b - { \tilde { 1 } } }$ , and hence

$$
\begin{array} { r l r } & { } & { \frac { \mathrm { r a n g e ~ o f } b _ { 0 } } { b < b _ { 0 } < 2 } \left| \begin{array} { c } { D _ { 0 } } \\ { O ( T ^ { 1 - b _ { 0 } } ) } \end{array} \right| \frac { D _ { 0 } / S _ { T } } { O ( T ^ { b - b _ { 0 } } ) } } \\ & { } & { b _ { 0 } = 2 \left| O ( T ^ { - 1 } \log T ) \right| \left. O ( T ^ { b - 2 } \log T ) \right. } \\ & { } & { b _ { 0 } > 2 \left. \qquad O ( T ^ { - 1 } ) \right. \qquad O ( T ^ { b - 2 } ) } \end{array}
$$

In the first row, $b - b _ { 0 } < 0 ;$ in the last two rows, $b - 2 < 0$ . Moreover, log $T / T ^ { 2 - b } \to 0$ because $2 - b > 0$ . Thus every row gives $D _ { 0 } / S _ { T }  0$

Consequently, combining the bounds for $D _ { 1 }$ and $D _ { 0 }$ gives

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { s u p } _ { \mathbf { \mathscr { D } _ { 1 } } + \mathbf { \mathscr { ( } V _ { 0 } / V _ { 1 } ) } D _ { 0 } } \leq ( 1 + \chi ) \alpha K _ { b } ( r ) .
$$

We now combine this estimate with Equations (43) and (44) and get

$$
\begin{array} { r l r } {  { \operatorname* { l i m i n f } _ { T \to \infty } \frac { \operatorname* { m i n } \{ P ( L ) , P ( T ) \} - P ( W ^ { \circ } ) - D _ { 1 } - ( V _ { 0 } / V _ { 1 } ) D _ { 0 } } { S _ { T } } \geq \delta _ { b } ( r ) - \alpha [ H _ { b } ( r ) + ( 1 + \chi ) K _ { b } ( r ) ] } } \\ & { } & \\ & { } & { > \frac { \delta _ { b } ( r ) } { 2 } > 0 , } \end{array}
$$

because $\alpha < \alpha _ { c }$ . This finishes the second part of the proof.

Step 3: $J _ { T } ( W )$ cannot attain its minimum at $W = 0 .$ . We prove that $J _ { T } ( W )$ cannot attain its minimum at $W = 0$ when $T$ is sufficiently large by showing that $J _ { T } ( W ^ { \circ } ) < j _ { T } ( 0 )$ for sufficiently large T. Recall that by Equation (33), we have

$$
\begin{array} { c } { { J _ { T } ( W ) = C _ { 0 } + \displaystyle \frac { B _ { 0 } } { [ K _ { 0 } / 2 + A _ { W } ( T ) ] ^ { a _ { 0 } } } + \displaystyle \frac { B _ { 1 } } { [ K _ { 1 } / 2 + A _ { W } ( L ) ] ^ { a _ { 1 } } } } } \\ { { + V _ { 0 } I _ { b _ { 0 } , g _ { 0 } } ( T , W ) + V _ { 1 } I _ { b _ { 1 } , g _ { 1 } } ( L , W ) + V _ { 1 } P ( W ) , } } \end{array}
$$

and we analyze the last five terms separately. First, we evaluate $I _ { \beta , q } ( H , W ^ { \circ } )$ and $I _ { \beta , q } ( H , 0 )$ . Here, $H$ can be either L or $T ,$ , and β can be either $b _ { 1 }$ or $b _ { 0 } ,$ , respectively. Recall that $W ^ { \circ } = \kappa \alpha T$ , and for $H = L$ or $H = T$ , we have $H \geq \alpha T > \kappa \alpha T = W ^ { \circ }$ . So for all $\begin{array} { r } { 0 < W \le H ; } \end{array}$

$$
I _ { \beta , g } ( H , W ) = \int _ { 0 } ^ { W } \frac { ( m + \Delta u / W ) ^ { 2 } } { [ g + m u + \Delta u ^ { 2 } / ( 2 W ) ] ^ { \beta } } \mathrm { d } u + \int _ { W } ^ { H } \frac { M ^ { 2 } } { [ g + M u - \Delta W / 2 ] ^ { \beta } } \mathrm { d } u .
$$

When $0 \le u \le W , A _ { W } ^ { \prime } ( u ) = m + \Delta u / W$ , so

$$
\frac { \mathrm { d } } { \mathrm { d } u } \left[ ( m + \Delta u / W ) [ g + A _ { W } ( u ) ] ^ { 1 - \beta } \right] = \frac { \Delta } { W } [ g + A _ { W } ( u ) ] ^ { 1 - \beta } - ( \beta - 1 ) \frac { ( m + \Delta u / W ) ^ { 2 } } { \left[ g + A _ { W } ( u ) \right] ^ { \beta } } .
$$

Therefore, by integration by parts, we have

$$
\begin{array} { l } { I _ { \beta , g } ( H , W ) = \displaystyle \frac { m g ^ { 1 - \beta } - M [ g + ( M + m ) W / 2 ] ^ { 1 - \beta } } { \beta - 1 } + \frac { \Delta } { ( \beta - 1 ) W } \int _ { 0 } ^ { W } [ g + A _ { W } ( u ) ] ^ { 1 - \beta } \mathrm { d } u } \\ { \displaystyle \qquad + \frac { M } { \beta - 1 } \left[ [ g + ( M + m ) W / 2 ] ^ { 1 - \beta } - [ g + M H - \Delta W / 2 ] ^ { 1 - \beta } \right] } \\ { \displaystyle = \frac { m g ^ { 1 - \beta } } { \beta - 1 } - \frac { M [ g + M H - \Delta W / 2 ] ^ { 1 - \beta } } { \beta - 1 } + \frac { \Delta } { ( \beta - 1 ) W } \int _ { 0 } ^ { W } [ g + m u + \Delta u ^ { 2 } / ( 2 W ) ] ^ { 1 - \beta } \mathrm { d } u . } \end{array}
$$

Because $1 - \beta < 0 ,$ , its integral satisfies

$$
\begin{array} { c l } { \displaystyle 0 \le \frac 1 W \int _ { 0 } ^ { W } [ g + m u + \Delta u ^ { 2 } / ( 2 W ) ] ^ { 1 - \beta } \mathrm { d } u \le \frac 1 W \int _ { 0 } ^ { W } ( g + m u ) ^ { 1 - \beta } \mathrm { d } u } \\ { \displaystyle \qquad \le \left\{ \frac { ( g + m W ) ^ { 2 - \beta } } { m ( 2 - \beta ) W } = O ( W ^ { 1 - \beta } ) , \qquad 1 < \beta < 2 , \right. } \\ { \displaystyle \frac { \log ( 1 + m W / g ) } { m W } = O ( W ^ { - 1 } \log W ) , } & { \displaystyle \beta = 2 , } \\ { \displaystyle \frac { g ^ { 2 - \beta } } { m ( \beta - 2 ) W } = O ( W ^ { - 1 } ) , } & { \displaystyle \beta > 2 . } \end{array}
$$

We take $T \to \infty$ , and consequently $W = W ^ { \circ } = \kappa \alpha T  \infty , H \geq \alpha T  \infty$ . We obtain

$$
I _ { \beta , g } ( H , W ^ { \circ } ) \longrightarrow \frac { m g ^ { 1 - \beta } } { \beta - 1 } .
$$

For the constant learning rate $\eta _ { 0 } ( t ) = M$ , direct integration gives

$$
I _ { \beta , g } ( H , 0 ) = \int _ { 0 } ^ { H } \frac { M ^ { 2 } } { ( g + M u ) ^ { \beta } } \mathrm { d } u = \frac { M } { \beta - 1 } \left[ g ^ { 1 - \beta } - ( g + M H ) ^ { 1 - \beta } \right] .
$$

Consequently, $I _ { \beta , g } ( H , 0 ) \to M g ^ { 1 - \beta } / ( \beta - 1 )$ when $T \to \infty$

For both $W = 0$ and $W = W ^ { \circ }$ , and for all $H \ge \alpha T , A _ { W } ( H ) \ge m H \ge m \alpha T$ . Hence for both $l = 0$ and $l = 1$ , and for all $H \geq \alpha T$ , we have

$$
0 \leq \frac { B _ { l } } { [ K _ { l } / 2 + A _ { W } ( H ) ] ^ { a _ { l } } } \leq \frac { B _ { l } } { ( m \alpha T ) ^ { a _ { l } } } \longrightarrow 0 , T  \infty .
$$

We can bound $P ( W )$ similarly:

$$
0 \leq P ( W ) = \frac { \int _ { L } ^ { T } \eta _ { W } ( T - u ) ^ { 2 } \mathrm { d } u } { [ g _ { 1 } + A _ { W } ( L ) ] ^ { b } } \leq \frac { M ^ { 2 } T } { ( m \alpha T ) ^ { b } } = \frac { M ^ { 2 } } { ( m \alpha ) ^ { b } } T ^ { 1 - b } \longrightarrow 0 , \quad T  \infty .
$$

Substituting these limits into Equation (42) gives

$$
J _ { T } ( 0 ) - J _ { T } ( W ^ { \circ } ) \longrightarrow ( M - m ) \sum _ { l = 0 } ^ { 1 } \frac { V _ { l } g _ { l } ^ { 1 - b _ { l } } } { b _ { l } - 1 } > 0 , \quad T  \infty .
$$

Therefore, there exists a threshold $T _ { 0 }$ such that for all $T > T _ { 0 }$ , we have

$$
J _ { T } ( 0 ) - J _ { T } ( W ^ { \circ } ) > \frac { M - m } { 2 } \sum _ { l = 0 } ^ { 1 } \frac { V _ { l } g _ { l } ^ { 1 - b _ { l } } } { b _ { l } - 1 } > 0 , \quad T > T _ { 0 } .
$$

This concludes the proof.