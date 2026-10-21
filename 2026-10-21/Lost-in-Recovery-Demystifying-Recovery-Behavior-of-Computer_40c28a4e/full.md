# Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay

Byeonghun Bang<sup>1</sup>\* Sangwoo Park<sup>1</sup>\* Jinheon Baek<sup>1</sup> Kangsan Kim<sup>1</sup> Youngwan Lee<sup>1,2</sup> Sung Ju Hwang<sup>1,3</sup>

<sup>1</sup>KAIST <sup>2</sup>ETRI <sup>3</sup>DeepAuto.ai

{bh.bang, jackson0530, jinheon.baek, kangsan.kim, sungju.hwang}@kaist.ac.kr yw.lee@etri.re.kr

## Abstract

Recent advances in large (vision) language models have enabled computer-use agents that automate digital workflows via fine-grained keyboard and mouse operations. However, such agents remain fragile in long-horizon settings, where errors accumulate and executed actions leave unintended side effects, making the ability to recover from failures critical. Yet, existing remedies remain inadequate: environment resets are infeasible in proprietary settings, action replay is prohibitively expensive over long horizons, and API-based rollback is limited to web environments, leaving model-based recovery largely unexplored. To address this gap, we present RETRACE, a scalable evaluation framework for systematically studying recovery behavior in computer-use agents. We formalize Transition-Consistent Recovery: identical downstream actions from the recovered and target states must yield identical outcomes. We operationalize this criterion through goal-oriented action replay following controlled error injection. Using RE-TRACE, we present the first systematic study of autonomous, model-based recovery, revealing that (i) ideal recovery can unlock model’s latent task completion capabilities, (ii) current computer-use agents fail to exploit their latent recovery capabilities, and (iii) accurate model-based recovery yields measurable endto-end gains. Together, these findings establish recovery as a fundamental capability for building reliable computer-use agents. Our code and dataset is available at https:// github.com/BangProx/ReTRACE.

## 1 Introduction

The digital computing environment has become the primary interface through which users interact with software, supporting a wide spectrum of tasks ranging from simple actions (e.g., website logins) to complex workflows (document editing). Recent advances in large (vision) language models (Bai et al., 2025; Singh et al., 2025; Guo et al., 2025; Anthropic, 2026) have made it possible to automate such workflows through computer-use agents, which act in digital environments via fine-grained keyboard and mouse inputs (Gur et al., 2023; Furuta et al., 2023; Sager et al., 2025).

![](images/cbf120d34c24536d89cf93c79e6f751019efea9fde148786ef2d069402133207.jpg)  
Figure 1: Comparison of error-handling strategies in computer-use agents. (A) Reflection proceeds from the erroneous state, (B) rollback requires often-infeasible environment resets, while (C) recovery restores a prior intact state and resumes the task.

However, computer-use agents remain fragile in long-horizon and sophisticated tasks, where early mistakes accumulate and propagate across subsequent steps (Wang et al., 2026a), and where executed actions may leave unintended side effects that should not have occurred (e.g., an agent organizing a directory may complete the task but silently overwrite files that the user wanted to keep). As a result, the ability to revert the environment to a previously intact state is critical for reliable operation. To address this, prior work explores two directions:

inducing self-reflective behaviors through textual feedback or parameter updates from reward signals that refine forward actions (Chae et al., 2025a,b; Wu et al., 2025b; Wang et al., 2025a; Zhang et al., 2026a; Xue et al., 2026; Wang et al., 2026c), and explicit test-time rollback through full environment resets followed by re-executing all prior actions (i.e., action replay) (Koh et al., 2024b; He et al., 2025) or API-based undo operations in web environments (e.g., browser back) (Hu et al., 2025; Zhang et al., 2026b), utilized as a heuristic component within an inference pipeline rather than subjected to systematic study.

Yet, these heuristic remedies fall short of providing general-purpose recovery. Environment resets are often infeasible in proprietary settings such as production websites and operating systems (Regatti et al., 2021; Nguyen and Cheng, 2023; Wang et al., 2026b), action replay (Koh et al., 2024b; He et al., 2025) inherits this requirement and becomes costly over long horizons, and API-based rollback (Hu et al., 2025; Zhang et al., 2026b) is largely confined to web environments, leaving broader desktop settings uncovered. Consequently, autonomous model-level recovery, in which the agent itself restores a prior state without relying on such ad hoc mechanisms, emerges as an essential capability. However, whether current agents possess recovery capabilities, and how it affects overall agent performance remain largely unexplored. Therefore, we investigate several fundamental research questions:

RQ1. Is recovery necessary to unlock models’ latent capabilities?

RQ2. Do current computer-use agents possess recovery capabilities under erroneous conditions, and are such capabilities latent or entirely absent? RQ3. Does accurate model-level recovery translate into measurable improvements in task completion?

To address this, we introduce RETRACE, a pipeline that conducts the first systematic study of autonomous, model-based recovery behavior in computer-use agents. Motivated by predictive state representation theory (Littman and Sutton, 2001), which represents states by the formalization of predicted futures, we formalize successful recovery as Transition-Consistent Recovery: a recovered state matches its target if executing identical downstream actions from both states yields identical outcomes. This formulation captures latent changes beyond visual similarity by leveraging behavioral equivalence, operationalized via goaloriented action replay. Specifically, we replay each on-policy trajectory multiple times and retain only those with consistent successful outcomes, ensuring deterministic environment transitions. From these trajectories, we sample intermediate states as recovery targets, perturb them into erroneous states, and evaluate whether the original task outcome can be reproduced by replaying the remaining actions.

We then instantiate RETRACE on OSWorld (Xie et al., 2024) and WindowsAgentArena (Bonatti et al., 2024) across closed- and open-source computer-use agents, and find that recovery plays a critical role in reliable agent operation. We first show that ideal recovery, in which agents are returned to an oracle prior state, boosts performance over naive environment resets (Section 4), identifying recovery as the key leverfor unlocking agents’ latent task-solving capacity (RQ1). Yet current agents leave this lever unused, lacking both the tendency and capability to recover autonomously (Section 5) (RQ2). When equipped with explicit model-based recovery, however, recovery success climbs by more than an order of magnitude, revealing that the underlying capability is latent rather than absent (Section 5) (RQ2). Moreover, recovery quality in turn translates into downstream task success: accurate model-level recovery delivers measurable end-to-end gains by unlocking otherwise latent task-solving capacity, while weak recovery can actively derail the trajectory (RQ3) (Section 6). These findings position accurate model-based recovery, together with reliable continuation from the recovered state, as a key step toward stronger computer-use agents.

## 2 Related Works

Computer-Use Agents Computer-use agents aim to complete diverse user tasks across webbased environments (Gur et al., 2023; Furuta et al., 2023; OpenAI, 2025; Awadallah et al., 2025), desktop operating systems such as Linux and Windows (Wang et al., 2025b; Yang et al., 2025; Anthropic, 2026), and mobile environments (Ye et al., 2025a; Xu et al., 2026). Pioneering works leverage LLMs over HTML DOM content (Liu et al., 2018; Kim et al., 2023; Zhou et al., 2023), while recent studies operate LVLMs over screenshot observations (Koh et al., 2024a; Bai et al., 2025; Kim et al., 2026; Wang et al., 2026c) and further reinforce backbone capabilities through reinforcement learning (Wang et al., 2025a; Xu et al., 2025; Xue et al.,

![](images/7b1853c07d0e31a5e45d8c89fbe6e169cea27c0917a9ced9942522ee59e4008c.jpg)  
Figure 2: Overview of RETRACE. (A) On-Policy Trajectory Rollout collects successful trajectories and replays each action sequence N times to build a transition-consistent dataset. (B) Trajectory Drift via Perturbation selects a recovery target and perturbs the agent into an erroneous state. (C) Goal-Oriented Action Replay replays the remaining actions from the recovered state to test whether the agent reproduces the original task outcome.

## 3 Preliminary

2026) or external memories (Zheng et al., 2023; Fu et al., 2024). However, these efforts concentrate on strengthening theforward decision policy, leaving long-horizon brittleness, where early mistakes accumulate and propagate downstream (Wang et al., 2026a), largely unaddressed. In contrast, we focus on the systematic analysis of recovery behavior in computer-use agents, a complementary but underexplored axis.

Agent Behaviors under Failure Prior work explores diverse error-handling strategies across complex interactive tasks. For example, Lightman et al. (2024); Shinn et al. (2023); Xi et al. (2024) use outcome and process-level feedback to refine actions at test time in mathematical and code tasks (Chen et al., 2022). In long-horizon embodied settings, Huang et al. (2022); Wang et al. (2023a,b) leverage external rewards to revise plans or acquire new skills, and Li et al. (2025) perform rollback through resetting the environment and executing identical actions. This problem has also been explored in computer-use agents: Wang et al. (2024); Chae et al. (2025b); Wu et al. (2025b) use external verifiers to induce reflective behaviors (Hu et al., 2025) for effective user task completion, Koh et al. (2024b); He et al. (2025) employ action replay to revert to prior states upon failure, and Hu et al. (2025); Zhang et al. (2026b) utilize explicit rollback via go-back operations in Playwright, while Lu et al. (2026) introduce semantic actions for state recovery within the DFS algorithm. However, previous works used these mechanisms as heuristics in inference pipelines rather than systematically analyzing their behavior and effects. In contrast, we uncover key findings on model-level recovery in computer-use agents using RETRACE, a scalable pipeline enabling the first systematic analysis of recovery behavior.

Computer-Use Task We formulate a computeruse agent (CUA) task as a Partially Observable Markov Decision Process (Kaelbling et al., 1998), $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { O } , \mathcal { E } , \mathcal { T } , \Omega )$ , where a policy $\pi _ { \theta }$ interacts with a digital environment E (e.g., Linux terminals, PowerPoint) to complete a user query q. Each state $s _ { t } = ( h _ { t } , o _ { t } ) \in \mathcal { S }$ decomposes into a hidden component $h _ { t }$ and an observation $o _ { t } = \Omega ( s _ { t } ) \in \mathcal { O } _ { \left. \right.} $ satisfying the Markov property. At each step, π<sub>θ</sub> selects an action $a _ { t } \in \ A .$ , spanning high-level (CLICK) to low-level (pyautogui.click) operations, yielding $s _ { t + 1 } \sim T ( \cdot \mid s _ { t } , a _ { t } )$

Forward Policy and Trajectory Since E is only partially observable, $\pi _ { \theta }$ conditions on the observation–action history, $\pi _ { \boldsymbol { \theta } } ( a _ { t } \ | \ o _ { 0 } , a _ { 0 } , \ldots , o _ { t } )$ , yielding a trajectory ${ \boldsymbol { \tau } } = \{ ( o _ { t } , a _ { t } ) \} _ { t = 0 } ^ { T }$ . However, this forward formulation leaves unspecified how the agent should behave once early actions drive τ into an erroneous state from which q can no longer be completed, the setting we study next.

## 4 Accurate Recovery Matters

Here, we show through a preliminary experiment that accurate recovery yields measurable gains, establishing it as a powerful lever for reliable agents. For more details, please see Appendices B.1 and F.

Expected Gains from Ideal Recovery Agents exhibit diverse trajectories across repeated rollouts of the same task, and this variance may imply substantial latent task-solving capability that any single rollout fails to exploit, motivating our hypothesis: reverting to an appropriate past state (recovery step) can convert it into task success. To validate this, we measure Success@5, the success rate over 5 independent continuations, when the forward policy resumes from different recovery steps.

![](images/de386c5b2ef884c9b996b960c046fe614c19a9a03612d112dc4fc905695e357c.jpg)  
Figure 3: Success@k across different recovery steps R under ideal recovery, using EvoCUA-32B as the forward policy on OSWorld.

Setup We focus on tasks that the forward policy fails to solve under standard rollouts, the cases where recovery can yield measurable gains, with EvoCUA-32B (Xue et al., 2026) as the forward policy on the OSWorld (Xie et al., 2024) benchmark. For each failed trajectory, we annotate the error step E (where the agent first goes off track) using an LLM-based detector, and set the recovery step R, from which the forward policy resumes, to one of: the initial state $( \mathrm { R } = 0$ , equivalent to a full environment reset), one step before the error $( \mathrm { R } = \mathrm { E } \mathrm { - } 1 )$ , a uniformly sampled past step (R = E-random), or a step identified by an LLM-based recovery step detector $\mathrm { ( R = R _ { L L M } ) }$ . We use Qwen3.6-Plus (Qwen Team, 2026b) as the backbone for both detectors.

Results As shown in Figure 3, every non-zero recovery strategy outperforms complete environment resets $( \mathrm { R } = 0 )$ , with the largest advantage observed in the low-K regime. This demonstrates that, under the same retry budget, recovering from the previous state utilizes each retry substantially more efficiently than restarting the task from scratch (Finding 1), resulting in significantly higher Success@K. As K increases, the gap naturally diminishes because repeated restarts also accumulate additional opportunities for success. These findings therefore highlight the efficiency of accurate recovery in utilizing model’s internal capabilities under constrained retry budgets.

## 5 How Well Can Agents Recover?

Here, we analyze the recovery behavior of current computer-use agents using the empirical pipeline illustrated in Figure 2. For more details, please see Appendices A, B.3, C, and F.

## 5.1 Transition-Consistent Recovery

Given a trajectory ${ \boldsymbol { \tau } } = \{ ( s _ { t } , a _ { t } ) \} _ { t = 0 } ^ { T }$ produced by a policy π, the recovery task is defined as a function f that maps an arbitrary state $s _ { t }$ back to a past target state, $f ( s _ { t } ) = s _ { \mathrm { t a r g e t } } = s _ { t - \mathrm { d } }$ , where d is the recovery distance. Whether this mapping succeeds is judged by a predefined recovery oracle $\mathcal { R }$ , which returns a binary indicator of recovery success or failure, ${ \mathcal { R } } ( f ( s _ { t } ) , s _ { \mathrm { t a r g e t } } ) \in \{ 0 , 1 \}$ Recalling from Section 3 that each state decomposes as $s _ { t } ~ = ~ ( h _ { t } , o _ { t } )$ into a hidden component $h _ { t }$ and an observation $o _ { t }$ , R should verify that $f ( s _ { t } )$ matches $s _ { \mathrm { t a r g e t } }$ in both components. A natural instantiation of R is to compare the observations of $f ( s _ { t } )$ and $s _ { \mathrm { t a r g e t } }$ directly, e.g., via pixelor feature-level matching. However, this surface criterion is unreliable, since latent components of the environment may be implicitly altered without any visible change (Shani and Brafman, 2004) $( \mathrm { e . g . }$ , a hidden configuration file modified while the screenshot remains identical), causing observationonly oracles to falsely declare success on states that nonetheless diverge in their downstream effects. To capture such latent discrepancies, we draw on the theory of predictive state representations (Littman and Sutton, 2001), which defines state equivalence through identical transition dy namics, and formalize transition-consistent recovery. Specifically, in a controlled environment $\mathcal { E }$ with a deterministic transition function $\mathcal { T } _ { \mathcal { E } }$ , we declare a recovery successful when executing the same downstream action $a _ { t - \mathrm { d } }$ from the recovered state reproduces the next state along the original trajectory, defined as $\mathcal { R } _ { \mathrm { T C } } = \mathbb { I } \big [ \mathcal { T } _ { \mathcal { E } } ( f ( s _ { t } ) , a _ { t - \mathrm { d } } )$ $s _ { t - \mathrm { d + 1 } } ]$ for $\left( s _ { t - \mathrm { d } } , a _ { t - \mathrm { d } } , s _ { t - \mathrm { d } + 1 } \right) \in \tau$ , which could extend to the multi-step setting by recursively chaining transitions along τ, as $\mathcal { R } _ { \mathrm { T C } } ~ =$ $\mathbb { I } \big [ \mathcal { T } _ { \mathcal { E } } ( \cdot \cdot \cdot \mathcal { T } _ { \mathcal { E } } ( f ( s _ { \mathrm { t } } ) , a _ { \mathrm { t - d } } ) \cdot \cdot \cdot , a _ { \mathrm { t - 1 } } ) = s _ { \mathrm { t } } \big ]$

To complement this hidden-state check, we additionally enforce observation-level agreement to guard against cases where the visible interface itself diverges from the target. Formally, we define $\mathcal { R } _ { \mathrm { O } } = \mathbb { I } [ o = \hat { o } ]$ , where $( h , o ) = s _ { \mathrm { t a r g e t } }$ and $( \hat { h } , \hat { o } ) = f ( s _ { t } )$ . The final recovery oracle then combines both, $\mathcal { R } = \mathcal { R } _ { \mathrm { T C } } \wedge \mathcal { R } _ { \mathrm { O } }$ , jointly evaluating recovery on the hidden and observable axes.

## 5.2 Recovery Evaluation

RETRACE evaluates recovery in three stages (Figure 2): the first two construct a transition-consistent evaluation set, and the third measures recovery on it via goal-oriented action replay.

On-Policy Trajectory Rollout As the first stage of RETRACE (Figure 2 (A)), evaluating transitionconsistent recovery requires a target state $s _ { \mathrm { t a r g e t } }$ and an error state $s _ { \mathrm { e r r o r } }$ for a trajectory τ , together with a deterministic transition function $\tau .$ To probe each policy on its own behavior, τ must remain on-policy with the evaluated $\pi \theta$ . We thus begin by rolling out each $\pi _ { \boldsymbol { \theta } } \in \Pi$ on every user query $\mathrm { q } \in \mathcal { Q }$ , collecting ground-truth trajectories $\tau _ { \mathrm { g t } } = \{ ( s _ { t } , a _ { t } ) \} _ { t = 0 } ^ { T }$ whose success is determined by an oracle $\Phi _ { \mathrm { q } }$ . For each $\tau _ { \mathrm { g t } }$ , we sample seed time steps $t _ { \mathrm { s e e d } } \in \{ 1 , \dots , T \}$ and retain only successful trajectories with $\Phi _ { \mathrm { q } } ( \tau _ { \mathrm { g t } } ) = 1$ , forming an evaluation set D for each π .

We further require a reliable transition function, grounded in the formal definition of state recovery in Section 5.1. To this end, inspired by prior work that approximates the latent model distribution through iterative sampling of model outputs (Wu et al., 2025a), we construct an amortized deterministic transition function $\hat { \mathcal { T } }$ for a given action trajectory $\tau _ { \mathrm { g t } }$ . Specifically, we replay the identical action sequence N times and retain only those that yield N successful outcomes, thereby ensuring reliable state transitions. This finally yields a transition-consistent trajectory dataset $\mathcal { D } _ { \mathrm { T C } }$ for each policy $\pi _ { \theta } \in \Pi$

Trajectory Drift via Perturbation Given the transition-consistent dataset, for $( \tau _ { \mathrm { g t } } , t _ { \mathrm { s e e d } } , \hat { T } ) \in$ $\mathcal { D } _ { \mathrm { T C } }$ we next require a ground-truth error state $s _ { \mathrm { e r r o r } }$ $\left( \mathrm { F i g u r e } \ 2 \ \left( \mathbf { B } \right) \right)$ . Taking $s _ { t _ { \mathrm { s e e d } } } \in \tau _ { \mathrm { g t } }$ as the target state $s _ { \mathrm { t a r g e t } } .$ , we obtain $s _ { \mathrm { e r r o r } }$ by drifting the trajectory into erroneous conditions with respect to the user query $\mathbf { q }$ via a perturbation policy $\pi _ { \theta } ^ { \mathrm { p } }$ Concretely, we first replay the action sequence of $\tau _ { \mathrm { g t } }$ in $\mathcal { E }$ using $\hat { \mathcal { T } }$ up to step $t _ { \mathrm { s e e d } }$ , with the LLM verifier $h _ { \theta }$ confirming that $s _ { t _ { \mathrm { s e e d } } }$ is reproduced. From this state, $\pi _ { \theta } ^ { \mathrm { p } }$ rolls out a perturbation trajectory $\tau _ { \mathrm { p } } ~ = ~ \left\{ \big ( s _ { t _ { \mathrm { s e e d } } } , ~ a _ { t _ { \mathrm { s e e d } } } \big ) , ~ . ~ . ~ . , ~ \big ( s _ { t _ { \mathrm { s e e d } } + t _ { \mathrm { p } } } \big ) \right\}$ up to the perturbation limit $t _ { \mathrm { p } }$ . The oracle $\Phi _ { \mathrm { q } }$ then assigns the final state $s _ { t _ { \mathrm { s e e d } } + t _ { \mathrm { p } } }$ as $s _ { \mathrm { e r r o r } }$ when it is erroneous, yielding the transition-consistent recovery evaluation set $\mathcal { D } _ { \mathrm { R } }$ for each policy $\pi _ { \theta } \in \Pi$ . By construction, every instance in $\mathcal { D } _ { \mathrm { R } }$ pairs a verified off-track error state $s _ { \mathrm { e r r o r } }$ with its reachable target s<sub>target</sub>, enabling a controlled evaluation of agent recovery under genuine failure.

Goal-Oriented Action Replay Given the recovery evaluation set $\mathcal { D } _ { \mathrm { R } }$ , we initiate recovery from each $s _ { \mathrm { e r r o r } } \in \mathcal { D } _ { \mathrm { R } }$ , which serves as the starting state for the recovery policy $\pi _ { \theta } ^ { \mathrm { R } }$ (Figure 2 (C)). Since $s _ { \mathrm { e r r o r } }$ is conditioned on $s _ { \mathrm { t a r g e t } }$ , we first replay $\tau _ { \mathrm { g t } }$ up to $s _ { \mathrm { t a r g e t } }$ and then $\tau _ { \mathrm { p } }$ for $t _ { \mathrm { p } }$ steps, using the LLM verifier $h _ { \theta }$ to confirm faithful reconstruction of $s _ { \mathrm { t a r g e t } }$ and $s _ { \mathrm { e r r o r } } ,$ respectively. With $s _ { \mathrm { e r r o r } }$ thus reconstructed, the recovery policy $\pi _ { \theta } ^ { \mathrm { R } }$ executes from it for at most r steps, producing the recovery trajectory $\tau _ { \mathrm { R } } = \{ ( s _ { t } , a _ { t } ) \} _ { t = 0 } ^ { T }$ and the recovered state $s _ { T } .$ However, single-step transition-consistent recovery is impractical, as defining oracle annotations for numerous intermediate states is infeasible. Therefore, following the multi-step formulation of transitionconsistent recovery in Section 5.1, we instead leverage the pre-configured oracle Φ for user-query completion, extending the objective to test whether replaying the remaining actions of $\tau _ { \mathrm { g t } }$ from the recovered state reproduces the original task outcome. The recovery oracle is thus defined as $\mathcal { R } _ { \mathrm { T C } }$ = $\mathbb { I } \big [ \Phi _ { \mathrm { q } } ( \mathcal { T } _ { \mathcal { E } } ( \cdot \cdot \cdot \mathcal { T } _ { \mathcal { E } } ( f ( s _ { t } ) , a _ { \mathrm { t - d } } ) , \dots , a _ { T } ) ) \ : = \ : 1 \big ]$ , together with $\mathcal { R } _ { \mathrm { O } }$ for observation-level matching.

## 5.3 Recovery Policy

Given an error state, a recovery policy acts to restore a past target state. We study two instantiations, differing in whether recovery is elicited from the forward agent itself or from a dedicated policy.

Implicit Recovery The implicit recovery policy reuses the forward decision policy as is, i.e., $\pi _ { \theta } ^ { \mathrm { { \tiny { R } } } } = \pi _ { \theta }$ , applying no recovery-specific instruction or parameter update. The same agent simply continues to act from the error state, which probes whether it spontaneously produces restorative actions toward prior states as an intrinsic behavior.

Explicit Recovery In contrast, the explicit recovery policy $\pi _ { \theta } ^ { \mathrm { R } }$ is an external policy specifically designed to revert the agent to a target recovery state $s _ { \mathrm { t a r g e t } }$ . Formally, this recovery process is defined as $\pi _ { \theta } ^ { \mathrm { R } } ( a _ { t } \mid s _ { t } , s _ { \mathrm { t a r g e t } } )$ , which, following the POMDP formulation in Section 3, can be expressed as $\pi _ { \boldsymbol { \theta } } ^ { \mathrm { R } } ( a _ { t } \mid o _ { \mathrm { t a r g e t } } , ( o _ { 0 } , a _ { 0 } , \ldots , o _ { t } ) )$ Unlike the forward decision policy that begins from a fixed starting point, recovery may start from an arbitrary state along the forward trajectory. Moreover, since the target state itself lies within that trajectory, its hidden component cannot be inferred from $O _ { \mathrm { { t a r g e t } } }$ alone; we thus introduce an additional recovery context C to represent the target hidden state $h _ { \mathrm { t a r g e t } } \ \in \ s _ { \mathrm { t a r g e t } } .$ leading to the final explicit recovery formulation $\pi _ { \boldsymbol { \theta } } ^ { \mathrm { R } } ( a _ { t } \mid o _ { \mathrm { t a r g e t } } , \mathcal { C } , ( o _ { 0 } , a _ { 0 } , \ldots , o _ { t } ) )$

![](images/0c6ddc2b72b8d6f221db856fd3e33133773d1f87ec3f16883d61e951a4755e4a.jpg)  
Figure 4: Comparison of short-horizon recovery strategies across computer-use agents on OSWorld.

![](images/9bfacd9dc39f494c3d77574ecb3cf6b06750ba5370d9113b459ba2a94eac42bd.jpg)  
Figure 5: Comparison of shorthorizon recovery strategies across models on WindowsAgentArena.

![](images/b900e4961893a3a02c62e6673ae3189811e7cefe7a6e2a85204edde0719ce621.jpg)  
Figure 6: Recovery intention under implicit vs explicit recovery, measured by the similarity between agent rationales and recovery prompts.

<table><tr><td>Baselines</td><td>Win Rate (%)</td></tr><tr><td>w/o  $\scriptstyle { \mathcal { R } } _ { \mathrm { O } }$ </td><td>82.0%</td></tr><tr><td>w/o  $\mathcal { R } _ { \mathrm { T C } }$ </td><td>63.0%</td></tr><tr><td>GPT-5.4 Evaluator</td><td>54.0%</td></tr></table>

Table 1: Average win rate of RE-TRACE against alternative evaluation strategies in matching human judgments, where human annotations serve as the ground truth. For more details, see Appendix B.3.

## 5.4 Experiments

## 5.4.1 Setup

We now describe the experimental setup, including datasets, models, and inference configurations.

Dataset We evaluate recovery on two computeruse benchmarks, OSWorld (Xie et al., 2024) and WindowsAgentArena (Bonatti et al., 2024), each deployed in an isolated environment and providing a user query together with an oracle evaluator that determines whether its execution succeeds.

Models We evaluate a range of state-of-the-art computer-use agents. As closed-source models, we use Claude Sonnet 4.6 (Anthropic, 2026) and GPT-5.4-Computer-Use (OpenAI, 2026), which achieve top performance on OSWorld. For open-source models, we consider two post-trained agentic models, EvoCUA-32B (Xue et al., 2026) and UI-TARS-1.5-7B (Qin et al., 2025), together with Qwen3.5- 9B-Thinking (Qwen Team, 2026a), a strong opensource model without task-specific post-training.

Inference Configurations We run experiments under short horizon and long horizon settings, with recovery distance d set to 1 and 5, respectively.

We further vary the maximum recovery step budget $\mathrm { { r } } \in \{ 1 , 3 , 1 0 \}$

Recovery Context We consider several recovery policy configurations with different contexts C. For the implicit recovery policy, we provide the observation–action history from the first step up to the error step, denoted 0-E. For the explicit recovery policy, we evaluate four variants: 0-E; 0-R, which instead provides the observation–action history up to the target recovery step $s _ { \mathrm { t a r g e t } } ;$ w/o History, which excludes the history from the context; and w/ Recovery Target Contextualization, which contextualizes the transition history leading up to the target recovery with an LLM summarizer to reduce context (Zhou et al., 2025; Ye et al., 2025b).

## 5.4.2 Results and Analysis

We now present and analyze the recovery evaluation results across both benchmarks and all agents.

Agents Fail to Recover Prior States Our results in Figures 4 and 5 show that even the strongest computer-use agents, Claude-Sonnet-4.6 and GPT-5.4-CUA, have limited recovery capability, while the strongest open-source model, EvoCUA-32B, exhibits a near-complete inability to restore previous states, with an average recovery rate below 1%. This suggests that current forward policies lack both the intrinsic capability and the behavioral tendency to autonomously perform restorative actions under failure (Finding 2). The analysis in Figure 6 reinforces this: rationales generated during implicit recovery show substantially lower similarity to the ground-truth recovery prompts, indicating that agents rarely even attempt recovery.

![](images/a6478e1f3d0d4b7656df07b048ce954f71c7bf272c911fdf9a18dfead3d15914.jpg)  
Figure 7: Maximum long-horizon recovery success rate among four different explicit recovery strategies on OSWorld.

<table><tr><td>Recovery Context</td><td> $\mathbf { S R } \left( \% \right) \uparrow$ </td><td> $\Delta \uparrow$ </td></tr><tr><td>Baseline</td><td> $3 9 . 3 8 \pm 1 . 1 7$ </td><td></td></tr><tr><td>Contextualizer</td><td> $4 1 . 0 6 \pm 3 . 0 5$ </td><td> $+ 1 . 6 8$ </td></tr><tr><td>0~R</td><td> ${ \bf 4 4 . 2 6 \pm 1 . 9 0 }$ </td><td> $+ 4 . 8 8$ </td></tr><tr><td>0~E</td><td> $4 0 . 4 9 \pm 0 . 9 5$ </td><td> $+ 1 . 1 1$ </td></tr><tr><td>w/o History</td><td> $4 0 . 2 \pm 1 . 8 4$ </td><td> $+ 0 . 8 2$ </td></tr></table>

<table><tr><td>Model</td><td>Setting</td><td>SR (%) ↑</td></tr><tr><td>Sonnet-4.6</td><td>Baseline w/ Recovery</td><td> $7 0 . 2 7 \pm 0 . 7 0$   $7 1 . 8 1 \pm 1 . 5 1$ </td></tr><tr><td>Qwen3.5-9B</td><td>Baseline w/ Recovery</td><td> $3 8 . 7 6 \pm 0 . 6 4$   $4 1 . 5 6 \pm 4 . 1 4$ </td></tr><tr><td>UI-TARS-1.5-7B</td><td>Baseline w/ Recovery</td><td> $1 7 . 2 5 \pm 1 . 5 5$   $1 5 . 7 4 \pm 0 . 6 5$ </td></tr><tr><td>EvoCUA-32B</td><td>Baseline w/ Recovery</td><td> $5 4 . 8 8 \pm 4 . 3 9$   $5 1 . 5 5 \pm 2 . 5 4$ </td></tr></table>

Table 2: End-to-end success rate (SR) across recovery strategies, where ∆ is the change over the no-recovery baseline, with three independent runs.  
Table 3: End-to-end task success rate (SR) on OSWorld across different models with the Recovery Target Contextualization strategy, over two independent runs.

Analysis on Explicit Recovery The results in Figures 4 and 5 further show that explicit modelbased recovery substantially outperforms implicit recovery, by an average margin of 17.86% across the Linux and Windows environments. In addition, the largest improvements are observed for generalist computer-use agents, such as Claude-Sonnet-4.6, GPT-5.4-CUA, and Qwen-3.5-9B. These results suggest that the recovery capability of current computer-use agents is latent rather than absent, but such capabilities are not effectively elicited during standardforward task execution (Finding 3). In contrast, the gains are considerably smaller for post-trained open-source agents, suggesting that extensive post-training on one-directional taskcompletion trajectories may weaken their inherent recovery capability (Finding 4). Figure 7 shows the same trend in long-horizon settings: generalist agents exhibit substantially greater recovery potential, whereas post-trained specialists consistently struggle with explicit recovery. Nevertheless, even the most capable models combined with basic explicit recovery strategies still fail to achieve complete recovery under our evaluation settings, suggesting that substantial research remains in developing more robust recovery mechanisms for computer-use agents.

![](images/8aae26d8d3344d3b116636883932e3ad594d78f8305acbafb29aee299b63310e.jpg)  
Figure 8: Performance change (∆) across recovery iterations under $R e \mathrm { - }$ covery Target Contextualization.

Efficacy of RETRACE To assess how reliably RETRACE judges recovery success, we conduct a human evaluation study against three baselines: two ablations of our oracle R with either the imagewise matching component $\mathcal { R } _ { \mathit { O } }$ or the transitionconsistent component $\mathcal { R } _ { \mathrm { T C } }$ removed, and GPT-5.4, a state-of-the-art model with extensive computeruse knowledge. We recruit 12 annotators to judge whether each recovered state aligns with the target state, with 50 samples per variant and each annotator rating 25 of them. As shown in Table 1, RE-TRACE agrees with human judgments more often than all baselines, including GPT-5.4, confirming the reliability of its recovery oracle R. We further report an average inter-annotator agreement of 82% and a Cohen’s kappa (Cohen, 1960) of 0.45, indicating reliable agreement.

## 6 End-to-End Evaluation of Recovery

In this section, we analyze how recovery quality affects the end-to-end task performance of computeruse agents. For more details, please see Appendices B.2 and F.

## 6.1 Setup

We follow the evaluation protocols of Sections 4 and 5.4. Specifically, the forward and recovery

![](images/6d90c675932dc93e56d8a19d7d3cffa69d82775c2be9f62430bb73b0caf04457.jpg)  
Table 4: Qualitative analysis of erroneous agent behaviors induced by imperfect state restoration.

agents share the same backbone model, on which we run the explicit recovery strategies.

## 6.2 Results and Analysis

Potential Gains from Model-Based Recovery To validate the potential benefits of model-based recovery, we analyze its performance gains across diverse recovery settings. As shown in Table 2, model-based recovery improves Qwen3.5-9B over the no-recovery baseline under every recovery setting, with an average gain of 2.12% in the endto-end setting. Notably, these gains hold despite the suboptimal recovery success rates observed in Figures 4 and 5, indicating that even imperfect recovery can begin to unlock the latent capabilities ofcurrent computer-use agents, and that more reliable recovery mechanisms hold substantial promise for improving the overall performance and robustness (Finding 5).

Recovery Performance Effects on Task Completion The results in Table 3 show that models with weaker recovery capabilities (UI-TARS-1.5- 7B, EvoCUA-32B) suffer performance degradation during end-to-end inference, underscoring the importance of accurate recovery behavior. In addition, Figure 8 demonstrates that allowing more recovery attempts yields further gains for models with comparably better recovery performance (Qwen-3.5-9B, UI-TARS-1.5-7B). EvoCUA-32B, in contrast, exhibits severely limited explicit recovery capability: while even a single successful recovery can substantially improve downstream task performance, its consistently low recovery success rate means that additional attempts more often derail the trajectory into increasingly erroneous states. This suggests that accurate model-level recovery delivers performance gains, while weak recovery can actively derail the trajectory (Finding 6).

## 6.3 Qualitative Analysis

In Section 6.2, we showed that poor recovery performance can severely degrade downstream task completion, in contrast to the idealized settings of Section 4. To understand why, we examine the rationales that forward agents generate after recovering to incorrect states, and categorize the resulting behaviors into two major patterns (Table 4). Additional examples are provided in Appendix D.

Spurious Perception Our first observation is that agents fixate on elements distinct from the target state and rely on them when choosing their next action. For example, in Table 4, the agent immediately clicks the right-alignment button because the cursor happens to hover over it in the restored state, skipping the drag operation on the text that should have come first and thereby deviating from the trajectory needed for correct task completion.

State Tracking Failure Meanwhile, agents also hallucinate actions by assuming the restored state matches the target, failing to register the discrepancy when the two diverge. For example, as shown in Table 4, although the restored state is masked by the terminal interface and no longer shows the fontselection UI, the agent acts as if that interface were still open, revealing the side effects of imperfect state restoration.

## 7 Conclusion

In this paper, we conducted the first systematic study of autonomous, model-level recovery behavior in computer-use agents through RETRACE, a scalable pipeline that formalizes successful restoration as Transition-Consistent Recovery, where a recovered state matches its target if identical downstream actions yield identical outcomes, and operationalizes this via goal-oriented action replay. Empirical evaluations across closed and open-source agents reveal that current agents lack both the tendency and capability to recover autonomously, while accurate model-based recovery substantially closes this gap and translates into measurable endto-end gains, despite characteristic failure modes such as spurious perception and state-tracking errors. We believe these findings position accurate model-based recovery as a key step toward reliable computer-use agents.

## Limitations

RETRACE is designed to measure and analyze recovery capabilities and behaviors rather than to directly improve them. Our explicit recovery settings can only elicit capabilities already present in the underlying model and cannot compensate for fundamental deficiencies in GUI understanding, state tracking, or action planning. Consequently, inaccurate recovery by specialist agents with weaker recovery capabilities may induce additional trajectory drift and reduce task success. Potential directions for addressing this gap include recovery-aware post-training on erroneous states, target states, and corresponding corrective action trajectories, as well as auxiliary objectives for tracking latent environment states and predicting actioninduced transitions. RETRACE can provide the evaluation protocol and controlled recovery scenarios needed to develop and assess such methods, but designing and validating them is beyond the scope of the present work.

## Ethical Considerations

Our work aims to advance the reliability of computer-use agents by systematically studying their recovery behavior, contributing positively to the safe and dependable deployment of digital automation. However, we acknowledge the dual-use nature of such research, as more capable agents could in principle be repurposed for harmful automation. To mitigate such risks, we conduct all experiments in sandboxed virtual environments using publicly available benchmarks in accordance with their licenses, without interaction with real user accounts or personal data, and recommend pairing deployments with appropriate safeguards.

## References

Anthropic. 2026. Claude-sonnet-4.6. https://www.anthropic.com/news/ claude-sonnet-4-6.

Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, et al. 2025. Fara-7b: An efficient agentic model for computer use. arXiv preprint arXiv:2511.19663.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3- vl technical report. arXiv preprint arXiv:2511.21631.

Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, et al. 2024. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264.

Hyungjoo Chae, Namyoung Kim, Kai Tzu iunn Ong, Minju Gwak, Gwanwoo Song, Jihoon Kim, Sunghwan Kim, Dongha Lee, and Jinyoung Yeo. 2025a. Web agents with world models: Learning and leveraging environment dynamics in web navigation. In The Thirteenth International Conference on Learning Representations.

Hyungjoo Chae, Sunghwan Kim, Junhee Cho, Seungone Kim, Seungjun Moon, Gyeom Hwangbo, Dongha Lim, Minjin Kim, Yeonjun Hwang, Minju Gwak, et al. 2025b. Web-shepherd: Advancing prms for reinforcing web agents. arXiv preprint arXiv:2505.15277.

Bei Chen, Fengji Zhang, A. Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2022. Codet: Code generation with generated tests. ArXiv, abs/2207.10397.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1):37–46.

Yao Fu, Dong-Ki Kim, Jaekyeom Kim, Sungryull Sohn, Lajanugen Logeswaran, Kyunghoon Bae, and Honglak Lee. 2024. Autoguide: Automated generation and selection of context-aware guidelines for large language model agents. Advances in Neural Information Processing Systems, 37:119919–119948.

Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. 2023. Multimodal web navigation with instruction-finetuned foundation models. arXiv preprint arXiv:2305.11854.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. 2023. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856.

Shiqi He, Yue Cui, Xinyu Ma, Yaliang Li, Bolin Ding, and Mosharaf Chowdhury. 2025. Branch-andbrowse: Efficient and controllable web exploration with tree-structured reasoning and action memory. arXiv preprint arXiv:2510.19838.

Minda Hu, Tianqing Fang, Jianshu Zhang, Jun-Yu Ma, Zhisong Zhang, Jingyan Zhou, Hongming Zhang, Haitao Mi, Dong Yu, and Irwin King. 2025. Web-CoT: Enhancing web agent reasoning by reconstructing chain-of-thought in reflection, branching, and rollback. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 5155– 5173, Suzhou, China. Association for Computational Linguistics.

Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. 2022. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608.

Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. 1998. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134.

Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. Language models can solve computer tasks. Advances in Neural Information Processing Systems, 36:39648–39677.

Suji Kim, Kangsan Kim, and Sung Ju Hwang. 2026. Learn from weaknesses: Automated domain specialization for small computer-use agents. arXiv preprint arXiv:2605.28775.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. 2024a. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 881–905.

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. 2024b. Tree search for language model agents. arXiv preprint arXiv:2407.01476.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Xingzuo Li, Kehai Chen, Yunfei Long, Xuefeng Bai, Yong Xu, and Min Zhang. 2025. Generator-assistant stepwise rollback framework for large language model agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17683–17700, Suzhou, China. Association for Computational Linguistics.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601.

Michael Littman and Richard S Sutton. 2001. Predictive representations of state. Advances in neural information processing systems, 14.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802.

Ziyu Lu, Tengjin Weng, Yiying Yang, Yuhang Zhao, Xinxin Huang, and Wenhao Jiang. 2026. Beap-agent: Backtrackable execution and adaptive planning for gui agents. arXiv preprint arXiv:2601.21352.

Hoai-An Nguyen and Ching-An Cheng. 2023. Provable reset-free reinforcement learning by no-regret reduction. In International Conference on Machine Learning, pages 25939–25955. PMLR.

OpenAI. 2025. Operator. https://openai.com/ ko-KR/index/introducing-operator/.

OpenAI. 2026. Gpt-5.4-computer use. https://openai.com/ko-KR/index/ introducing-gpt-5-4/.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. 2025. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326.

Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.

Qwen Team. 2026b. Qwen3.6-Plus: Towards real world agents.

Jayanth Reddy Regatti, Aniket Anand Deshmukh, Frank Cheng, Young Hun Jung, Abhishek Gupta, and Urun Dogan. 2021. Offline rl with resource constrained online deployment. arXiv preprint arXiv:2110.03165.

Pascal J Sager, Benjamin Meyer, Peng Yan, Rebekka von Wartburg-Kottler, Layan Etaiwi, Aref Enayati, Gabriel Nobel, Ahmed Abdulkadir, Benjamin F Grewe, and Thilo Stadelmann. 2025. Ai agents for computer use: A review of instruction-based computer control, gui automation, and operator assistants. arXiv preprint arXiv:2501.16150, pages 1–64.

Guy Shani and Ronen Brafman. 2004. Resolving perceptual aliasing in the presence of noisy sensors. Advances in Neural Information Processing Systems, 17.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Aaditya K. Singh, Adam Fry, Adam Perelman, Adam Tart, Adithya Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr A Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, and 464 others. 2025. Openai gpt-5 system card.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023a. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, et al. 2025a. Uitars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544.

Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multiagent collaboration. Advances in Neural Information Processing Systems, 37:2686–2710.

Xinyu Jessica Wang, Haoyue Bai, Yiyou Sun, Haorui Wang, Shuibai Zhang, Wenjie Hu, Mya Schroder, Bilge Mutlu, Dawn Song, and Robert D Nowak. 2026a. The long-horizon task mirage? diagnosing where and why agentic systems break. arXiv preprint arXiv:2604.11978.

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, et al. 2025b. Opencua: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123.

Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, and Yuxiong He. 2026b. Agent world model: Infinity synthetic environments for agentic reinforcement learning. arXiv preprint arXiv:2602.10090.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612.

Zihao Wang, Shaofei Cai, Anji Liu, Xiaojian Ma, and Yitao Liang. 2023b. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560.

Ziwei Wang, Leyang Yang, Xiaoxuan Tang, Sheng Zhou, Dajun Chen, Wei Jiang, and Yong Li. 2026c. History-aware reasoning for gui agents. In Proceedings of the AAAI Conference on Artificial Intelligence.

Fang Wu, Weihao Xuan, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, and Yejin Choi. 2025a. The invisible leash: Why rlvr may or may not escape its origin. arXiv preprint arXiv:2507.14843.

Qinzhuo Wu, Pengzhi Gao, Wei Liu, and Jian Luan. 2025b. Backtrackagent: Enhancing gui agent with error detection and backtracking mechanism. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 4250– 4272.

Zhiheng Xi, Dingwen Yang, Jixuan Huang, Jiafu Tang, Guanyu Li, Yiwen Ding, Wei He, Boyang Hong, Shihan Do, Wenyu Zhan, Xiao Wang, Rui Zheng, Tao Ji, Xiaowei Shi, Yitao Zhai, Rongxiang Weng, Jingang Wang, Xunliang Cai, Tao Gui, and 5 others. 2024. Enhancing llm reasoning via critique models with test-time and training-time supervision. Preprint, arXiv:2411.16579.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. 2024. Osworld: Benchmarking multimodal agents for openended tasks in real computer environments. Preprint, arXiv:2404.07972.

Haiyang Xu, Xi Zhang, Haowei Liu, Junyang Wang, Zhaozai Zhu, Shengjie Zhou, Xuhao Hu, Feiyu Gao, Junjie Cao, Zihua Wang, et al. 2026. Mobile-agentv3. 5: Multi-platform fundamental gui agents. arXiv preprint arXiv:2602.16855.

Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, and Yuxiao Dong. 2025. Mobilerl: Online agentic reinforcement learning for mobile gui agents. arXiv preprint arXiv:2509.18119.

Taofeng Xue, Chong Peng, Mianqiu Huang, Linsen Guo, Tiancheng Han, Haozhe Wang, Jianing Wang, Xiaocheng Zhang, Xin Yang, Dengchang Zhao, et al. 2026. Evocua: Evolving computer use agents via learning from scalable synthetic experience. arXiv preprint arXiv:2601.15876.

Bowen Yang, Kaiming Jin, Zhenyu Wu, Zhaoyang Liu, Qiushi Sun, Zehao Li, JingJing Xie, Zhoumianze Liu, Fangzhi Xu, Kanzhi Cheng, Qingyun Li, Yian Wang, Yu Qiao, Zun Wang, and Zichen Ding.

2026. Os-symphony: A holistic framework for robust and generalist computer-using agent. Preprint, arXiv:2601.07779.

Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Amrita Saha, Zeyuan Chen, et al. 2025. Gta1: Gui test-time scaling agent. arXiv preprint arXiv:2507.05791.

Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. 2025a. Mobile-agentv3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144.

Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang, Zile Qiao, Xinyu Wang, et al. 2025b. Agentfold: Long-horizon web agents with proactive context management. arXiv preprint arXiv:2510.24699.

Yao Zhang, Shijie Tang, Zeyu Li, Zhen Han, and Volker Tresp. 2026a. Webarbiter: A principle-guided reasoning process reward model for web agents. arXiv preprint arXiv:2601.21872.

Zhisong Zhang, Tianqing Fang, Kaixin Ma, Wenhao Yu, Hongming Zhang, Haitao Mi, and Dong Yu. 2026b. Webrollback: Enhancing web agents with explicit rollback mechanisms. In Proceedings of the 19th Conference ofthe European Chapter ofthe Association for Computational Linguistics (Volume 2: Short Papers), pages 187–197.

Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. 2023. Synapse: Trajectory-as-exemplar prompting with memory for computer control. arXiv preprint arXiv:2306.07863.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. 2025. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841.

## A RETRACE Details

Here, we provide implementation details on the evaluation set construction and recovery evaluation in RETRACE framework. We provide statistics of the filtered trajectories in Table 6, and distribution of domains within our recovery evaluation set in Table 7.

## A.1 Evaluation Set Construction

Benchmarks We primarily leverage two publicly available benchmarks consisting of both web tasks and non-web tasks: OSWorld <sup>1</sup> (Xie et al., 2024), released under the Apache License 2.0, and WindowsAgentArena (Bonatti et al., 2024), released under the MIT License. Both benchmarks consist of non-web tasks with significantly higher ratio compared to web-based tasks, which we demonstrate such ratio in Table 5 that underscores the importance of model-level recovery, where web based apis such as Playwright is not supported. To accelerate evaluation, we execute multiple environments in parallel, and for WindowsAgentArena, we utilize the processed environments provided by OS-Symphony<sup>2</sup> (Yang et al., 2026), released under the Apache License 2.0, to enable efficient large-scale evaluation set construction. Our use of these artifacts is consistent with their intended use as public benchmark and evaluation resources. We deploy all environments within isolated Docker containers and apply a 10-second post-action settle window to ensure that pending UI updates are fully rendered before capturing the next observation. For screenshot resolution, we use (1920, 1080) for OS-World and (1280, 800) for WindowsAgentArena. Finally, we restrict our evaluation to tasks that do not require proxy or Google Drive settings, due to constraints on proxy server budget and account configuration, respectively.

Models To construct the recovery evaluation set, we use five models: Claude-Sonnet-4.6 (Anthropic, 2026), GPT-5.4-CUA (OpenAI, 2026), Qwen-3.5-9B-Thinking (Qwen Team, 2026a), EvoCUA-32B (Xue et al., 2026), and UI-TARS-1.5-7B (Qin et al., 2025). Claude-Sonnet-4.6 and GPT-5.4-CUA are configured with thinking mode and xhigh reasoning mode, respectively. We use the pyautogui action space for all models except Claude-Sonnet-

<table><tr><td>Benchmark</td><td colspan="2">Web Tasks (%) Non-Web Tasks (%)</td></tr><tr><td>OSWorld</td><td>12.66</td><td>87.34</td></tr><tr><td>WindowsAgentArena</td><td>21.28</td><td>78.72</td></tr></table>

Table 5: Ratio of web and non-web tasks in the OSWorld and WindowsAgentArena benchmarks.

4.6, which uses the proprietary claude-computer use action space, while providing raw screenshots as observations. We adopt the default prompts from OSWorld and adapt them to Windows environments for WindowsAgentArena evaluation (Figures 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26). All open-source models are served locally using vLLM (Kwon et al., 2023) with bfloat16 precision through OpenAI-compatible HTTP endpoints. Experiments are conducted using a combination of one NVIDIA H200 GPU (141GB) and two RTX A6000 GPUs.

On-Policy Rollout For each model, we construct trajectories for the corresponding tasks in each benchmark. We then create an annotated dataset by selecting potential starting steps within each trajectory, while filtering out trajectories containing multiple actions within a single step or erroneous trajectories in which actions are missing from the agent output, as such cases can complicate the annotation process. Subsequently, the action sequences from each trajectory are replayed consecutively, and we retain only the trajectories that achieve a successful oracle evaluation in three consecutive replays (N=3) to minimize the effect from environment stochasticity (Transition-Consistent Filtering). This process empirically ensures deterministic transition dynamics, resulting in a filtered evaluation set in which the starting step of each trajectory is guaranteed to exhibit consistent transitions while retaining 64.4% of recovery trajectories (Table 6). For OsWorld short-horizon evaluation settings using GPT-5.4-CUA, we randomly sample 600 samples for perturbation after transitionconsistent filtering to keep the API computational cost manageable prior to subsequent filtering stage (Trajectory Perturbation).

Trajectory Perturbation Using the filtered evaluation set consisting of annotated starting steps from each trajectory, we replay the action sequences until reaching the designated starting step, while verifying that the resulting observation matches the original state. Subsequently, a perturbation agent sharing the same backbone as the forward agent is prompted (Figures 27, 28) to intentionally drift the trajectory under a maximum step budget identical to the recovery distance d. For short-horizon recovery evaluation, we use a perturbation budget of 1 for all models, except GPT-5.4-CUA, for which we use a budget of 2 due to its strong tendency to generate a “take a screenshot” action as the initial step which maps to time.sleep(), even under explicit instructions. We then retain only trajectories that result in oracle failure according to the benchmark evaluators (while excluding samples that were crashed due to environmental failures or gpt verification failures). This process constructs a recovery-annotated dataset, where the original starting state serves as the target recovery state, and the final perturbed state serves as the error state.

<table><tr><td></td><td>Dataset Filtering Step</td><td>Claude-Sonnet-4.6</td><td>GPT-5.4-CUA</td><td>QWEN-3.5-9B</td><td>UI-Tars-1.5-7B</td><td>EvoCUA-32B</td></tr><tr><td rowspan="4">Osrld</td><td># of traj. after formatting</td><td>2733</td><td>2749</td><td>790</td><td>1277</td><td>1259</td></tr><tr><td># of traj. after transition-consistent filtering</td><td>1921</td><td>1912 (600)</td><td>574</td><td>227</td><td>924</td></tr><tr><td># of traj. after perturbation</td><td>1335</td><td>334</td><td>397</td><td>133</td><td>314</td></tr><tr><td># of traj. after final set construction</td><td>120</td><td>120</td><td>120</td><td>60</td><td>120</td></tr><tr><td rowspan="5">Aena Wws</td><td># of traj. after formatting</td><td>919</td><td>867</td><td>211</td><td>N/A</td><td>280</td></tr><tr><td># of traj. after transition-consistent filtering</td><td>779</td><td>407</td><td>153</td><td>N/A</td><td>201</td></tr><tr><td># of traj. after perturbation</td><td>506</td><td>200</td><td>88</td><td>N/A</td><td>122</td></tr><tr><td># of traj. after final set construction</td><td>70</td><td>70</td><td>50</td><td>N/A</td><td>50</td></tr></table>

Table 6: Dataset statistics after each filtering step for the short-horizon settings.

Final Evaluation Set Construction Finally, using the recovery-annotated dataset, we exclude problematic samples whose perturbation trajectories generate multiple actions within a single step or fail to produce actions at arbitrary steps, as such cases complicate annotation. For short-horizon recovery evaluation, we additionally remove samples without meaningful perturbation actions, such as time.sleep() or WAIT. For GPT-5.4-CUA, we exclusively retain samples where the first action is “take a screenshot” and the second action corresponds to an actual perturbation action, due to its strong tendency to begin trajectories with screenshot actions. Furthermore, we remove samples where the error state and target recovery state show significantly high similarity, while also assuring that duplicate samples with both same error state observation and target recovery state observation do not exist, adopting SSIM as core mechanism. After filtering, we randomly sample evaluation subsets. For short-horizon evaluation in

OSWorld, we sample 120 instances for all models except UI-TARS-1.5-7B, for which we sample 60 due to the limited number of remaining trajectories. For long-horizon evaluation, we sample 80 instances for Qwen-3.5-9B and EvoCUA-32B, 60 for Claude-Sonnet-4.6, and UI-TARS-1.5-7B. Finally for WindowsAgentArena, we sample 70 instances for Claude-Sonnet-4.6 and GPT-5.4-CUA, and 50 for EvoCUA-32B and Qwen-3.5-9B, due to the different number of remaining samples after the filtering process.

## A.2 Recovery Evaluation

Benchmarks We use the evaluation sets constructed in Section A.1 to measure the recovery success rates of both implicit and explicit recovery policies, where the annotated evaluation sets are instantiated in both OSWorld and WindowsAgentArena, where we set the environment settings identical with Section A.1.

Models We evaluate both implicit and explicit recovery using the five models described in Section A.1. For Claude-Sonnet-4.6 and GPT-5.4- CUA, we use a temperature of 1.0 with thinking mode and xhigh reasoning mode enabled, respectively. For Qwen-3.5-9B and EvoCUA-32B, we use a temperature of 0.7, while UI-TARS-1.5-7B is evaluated with a temperature of 0.0. As the recovery target contextualizer, we use Claude-Sonnet-4.6 as the backbone state-transition summarizer with a temperature of 0.0 and a maximum token budget of 1024. All open-source backbones are served locally using vLLM within the same computing infrastructure described in Section A.1. Finally, we use GPT-5-Mini (Singh et al., 2025) as the state-verification model with default OpenAI hyperparameters.

Recovery Context For implicit recovery strategies, we provide the full history of observation–action pairs up to the error state, allowing the forward agent to infer the underlying state transitions, similar to real-world settings where agents interact continuously throughout the trajectory. In contrast, for explicit recovery strategies, we provide policies with various recovery contexts, including the observation–action history up to the error state and the target recovery state, which we denote as 0-E and 0-R, respectively. We additionally consider settings where such observations are not provided, as well as settings where the observation–action pairs up to the goal state are summarized by an LLM contextualizer that generates a compact description of the target recovery state and the transitions leading to it. When constructing observation–action histories, we maintain aligned observation–action pairs and exclude the final observation resulting from the last action, since new observation is input as current observation. The only exception is GPT-5.4-CUA, for which the final observation is included in the history due to proprietary API technical constraints.

<table><tr><td colspan="2">Domain Composition (%)</td><td>Original</td><td>Claude-Sonnet-4.6</td><td>GPT-5.4-CUA</td><td>QWEN-3.5-9B</td><td>UI-Tars-1.5-7B</td><td>EvoCUA-32B</td></tr><tr><td colspan="2">chrome</td><td>6.64</td><td>4.17</td><td>2.50</td><td>6.67</td><td>25.00</td><td>0.00</td></tr><tr><td rowspan="10">Osrild</td><td>gimp</td><td>8.23</td><td>0.83</td><td>6.67</td><td>8.33</td><td>1.67</td><td>6.67</td></tr><tr><td>libreoffice_calc</td><td>14.87</td><td>26.67</td><td>14.17</td><td>7.50</td><td>0.00</td><td>17.50</td></tr><tr><td>libreoffice_impress</td><td>14.87</td><td>29.17</td><td>15.83</td><td>21.67</td><td>13.33</td><td>38.33</td></tr><tr><td>libreoffice_writer</td><td>7.28</td><td>5.83</td><td>8.33</td><td>5.83</td><td>13.33</td><td>14.17</td></tr><tr><td>multi_apps</td><td>23.73</td><td>16.67</td><td>35.00</td><td>27.50</td><td>3.33</td><td>9.17</td></tr><tr><td>OS</td><td>7.59</td><td>4.17</td><td>1.67</td><td>2.50</td><td>11.67</td><td>8.33</td></tr><tr><td>thunderbird</td><td>4.75</td><td>4.17</td><td>1.67</td><td>11.67</td><td>11.67</td><td>0.00</td></tr><tr><td>vlc</td><td>5.38</td><td>3.33</td><td>5.00</td><td>0.83</td><td>15.00</td><td>3.33</td></tr><tr><td>vscode</td><td>6.65</td><td>5.00</td><td>9.17</td><td>7.50</td><td>5.00</td><td>2.50</td></tr><tr><td></td><td>13.48</td><td>14.29</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">Wina</td><td>file_explorer</td><td>2.13</td><td>4.29</td><td>17.14</td><td>24.00</td><td>N/A</td><td>48.00</td></tr><tr><td>microsoft_paint libreoffice_writer</td><td>12.77</td><td>22.86</td><td>0.00</td><td>0.00</td><td>N/A</td><td>0.00</td></tr><tr><td>libreoffice_calc</td><td>17.02</td><td>11.43</td><td>22.86 18.57</td><td>4.00</td><td>N/A</td><td>8.00</td></tr><tr><td>settings</td><td>3.55</td><td>1.43</td><td>0.00</td><td>0.00 16.00</td><td>N/A</td><td>6.00</td></tr><tr><td>vs_code</td><td>13.48</td><td>10.00</td><td>4.29</td><td>40.00</td><td>N/A</td><td>14.00</td></tr><tr><td>msedge</td><td>9.22</td><td>7.14</td><td>7.14</td><td></td><td>N/A</td><td>14.00</td></tr><tr><td></td><td>vlc</td><td></td><td>8.57</td><td></td><td>0.00</td><td>N/A</td><td>0.00</td></tr><tr><td></td><td>clock</td><td>9.92 2.84</td><td>1.43</td><td>21.43 0.00</td><td>8.00 8.00</td><td>N/A</td><td>0.00</td></tr><tr><td></td><td>windows_calc</td><td>2.13</td><td>0.00</td><td>4.29</td><td>0.00</td><td>N/A N/A</td><td>4.00 0.00</td></tr><tr><td></td><td>notepad</td><td>1.42</td><td>5.71</td><td>1.43</td><td>0.00</td><td>N/A</td><td>0.00</td></tr><tr><td></td><td>chrome</td><td>12.06</td><td>12.86</td><td>2.86</td><td>0.00</td><td>N/A</td><td>6.00</td></tr></table>

Table 7: Domain Composition ratio per recovery evaluation dataset (short horizon).

Prompts We construct the system prompts for explicit recovery by concatenating the recovery prefix prompts (Figures 29, 31) from Section F with the default system prompts used for each model in their respective benchmark settings. Similarly, we construct explicit recovery instructions by combining the model-specific instructions provided in the default OSWorld benchmark with the recovery instruction prefixes from Figures 30 and 32. In contrast, for implicit recovery policies, we use the original default system prompts and instructions without any recovery-specific prefix prompts or instructions. Meanwhile, we use prompts in Figures 34 and 35 for recovery target contextualizer, while leveraging prompts in Figure 33 for LLM -based state verification during recovery evaluation.

Recovery Verification In RETRACE, we evaluate recovery success along two axes: latent state recovery success and observation-wise state recovery success. For latent state recovery, we leverage our goal-oriented action replay framework, where recovery is considered unsuccessful if the benchmark oracle evaluator outputs 0 after executing the remaining actions from the ground-truth trajectory starting from the recovered state. For observationwise recovery, we use SSIM (Wang et al., 2004) with a threshold of 0.99 to detect visual mismatches that may interfere with accurate evaluation. Furthermore, in our main recovery evaluation experiments, we exclusively sample tasks that are consistently executable across all variants without docker failures or state verification failures, thereby ensuring a fair and reliable comparison of recovery success rates.

## B Additional Experiment Details

Here, we explain the experimental settings and implementation details for experiments introduced in Sections 4 and 6, along with details on Human Evaluation introduced in Table 1.

## B.1 Ideal Gains from Recovery

This experiment estimates how much end-to-end performance can improve when an agent is allowed to resume from an appropriate past state after a failure. It measures the potential gain from recovery under a controlled replay setting, where the target past state is reconstructed by action replay before the forward agent resumes execution.

Benchmark and model We run the analysis on OSWorld (Xie et al., 2024) using EvoCUA-32B (Xue et al., 2026) as the forward model, following the setup from Section A.1. We set the temperature to 0.7, max\_tokens to 32768, and max\_steps to 50. We first collect standard forward rollouts and keep only failed instances whose result.txt score is 0. For each failed instance, we identify an error step and select a corresponding recovery step using the recovery-step selection procedure described in Section 4. The environment at the selected recovery step is reconstructed by replaying the saved action prefix up to that step. We measure Success@5 on 83 samples after excluding instances where action replay fails.

Fixed annotations For each failed instance, we annotate an error step E once before running any recovery-step experiment. The annotation stage reads the saved trajectory, reconstructs the observation-action history, and calls the LLM errorstep detector. When evaluating the LLM-selected recovery step, we also annotate a recovery step $\mathrm { R } _ { \mathrm { L L M } }$ once in the same preparation stage. These annotations are then materialized into a fixed run plan, so different recovery-step conditions are evaluated against the same underlying failed trajectory rather than re-detecting E or $\mathrm { R } _ { \mathrm { L L M } }$ at runtime.

Recovery Steps We compare four starting steps. R = 0 restarts from the initial environment state. R = E − 1 resumes from the step immediately before the detected error. R = E − random samples a previous step with a fixed seed and reuses the same sampled step across runs for that instance. $\mathrm { R } = \mathrm { R } _ { \mathrm { L L M } }$ uses the recovery step predicted by the recovery-step detector. For $\mathrm { R } > 0$ , the environment is reset and the saved action prefix up to R is replayed before the forward agent resumes. For $\mathrm { R } = 0$ , no prefix is replayed.

Metric For each instance and each condition, we draw five independent continuations from the selected recovery state, and report Success@K for $\mathrm { ~ K ~ } \in \ \{ 1 , \ldots , 5 \}$ . Each continuation begins with action-prefix replay to reconstruct the recovery state; because replay can occasionally desynchronize from the saved trajectory (e.g., due to UI timing), we retry replay up to three times per continuation. Success@K measures whether at least one of the K successfully reconstructed continuations from the same recovery state completes the task, and reflects how much of the agent’s latent task-solving ability is exposed by recovery.

## B.2 End-to-End Evaluation

Benchmark We evaluate on the OSWorld (Xie et al., 2024) benchmark, where we follow a consistent setting as A.1.

Forward agents We use three open-source models as the forward agent: Qwen3.5-9B (Qwen Team, 2026a) with thinking mode, UI-TARS-1.5-7B (Qin et al., 2025), and EvoCUA-32B (Xue et al., 2026), following the setup in Section A.1, where default system prompts and instructions in Section A.1 are provided. Within each end-to-end experiment, the recovery agent shares the same backbone as the forward agent; the two differ only in their prompts and step budgets. For all forward agents we set max\_steps to 50. Temperatures are chosen from each model’s official inference recipe except for EvoCUA-32B: its default temperature of 0.0 produces deterministic outputs, which would cause the forward agent to repeat the same action when it resumes from a past state, so we set it to 0.7 to preserve action diversity after recovery. The resulting values are 0.7 for EvoCUA-32B, 0.7 for Qwen3.5- 9B-Thinking, and 1.0 for UI-TARS-1.5-7B. The max\_tokens budget is 32768 for EvoCUA-32B (to accommodate its long reasoning trace), 3000 for Qwen3.5-9B-Thinking, and 500 for UI-TARS-1.5-7B.

Detectors Both the error-step detector and the recovery-step detector are LLM-based and share Qwen3.6-Plus (Qwen Team, 2026b) as their backbone, accessed through the OpenRouter $\mathrm { A P I } ^ { 3 }$ with a temperature of 0.7 and custom system prompts (Figure 36, 37). The error detector is queried once per forward step. Once a recovery is triggered, the error detector is frozen for the next 5 steps; this prevents the detector from re-interrupting the forward agent before any recovery-induced changes have a chance to take effect.

Recovery agent The recovery agent is given a recovery step budget of 10 steps per attempt and is triggered at most max\_recovery\_iteration times per episode, with max\_recovery\_iteration $\in \ \{ 1 , 3 , 5 \}$ , using the identical system prompt and instructions as Section A.2. Recovery steps are tracked separately from the forward agent’s step budget. When all retries are exhausted, the forward agent continues without further recovery activation.

Post-recovery context After recovery completes, the forward agent resumes from the recovery-step context where the message history is truncated to the recovery boundary, and a short transition string informing the agent that the environment has been recovered is prepended before the next forward decision.

## B.3 Human Evaluation

When constructing the human evaluation sets, we sample recovery results from three representative models, including Claude-Sonnet-4.6, Qwen3.5- 9B, and EvoCUA-32B, across both short- and longhorizon evaluation settings in OSWorld, as well as the short-horizon setting in WindowsAgentArena. We compare RETRACE against three baselines: variants of our oracle evaluator with either $\mathcal { R } _ { O }$ or $\mathcal { R } _ { \mathrm { T C } }$ removed, and GPT-5.4, a state-of-the-art model with extensive computer-use knowledge. We collect cases where the evaluation decisions produced by the oracle implemented in RETRACE differ from those of the alternative evaluators, and randomly sample 50 instances for each variant, resulting in a total of 150 evaluation samples in Table 1. For the image-wise matching evaluator, we provide only the recovery target screenshot and the screenshot after the final recovery step. For the Transition-Consistent evaluator, success is determined solely based on whether replaying the remaining actions from the recovered state successfully reaches the original task goal. For GPT-5.4, we provide the history of observation-action pairs leading up to the error step, the recovery trajectory generated by the recovery agent, the recovery target screenshot, the error-step screenshot, and the final recovered-state screenshot. GPT-5.4 is then prompted to determine whether the semantic meaning of the recovered state aligns with the target recovery state, following the same context and inputs provided to human annotators for a fair comparison (See Figure 38 for prompt details).

After constructing the evaluation set, we split the 50 samples for each variant into two subsets and assigned two human annotators to each subset. In total, we recruited 12 annotators (4 annotators for each variant) who were researchers based in South Korea and highly proficient in computer-use tasks and voluntarily consented to participate in the study, with each annotator compensated approximately 20 USD for their participation, provided with detailed guideline for objective evaluation (Figure 11). Each pair of annotators independently evaluated whether the recovery target state had been successfully restored, labeling each sample in a binary success-or-failure manner consistent with our evaluators, while optionally indicating whether their decision was made despite stochastic environment dynamics (e.g., different timestamps or pop-ups) when the recovery process was believed to be successful, allowing us to analyze the effect of such environment dynamics on oracle judgments in RETRACE (Table 11).

Using these annotations as reference labels, we compared the decisions from our oracle evaluator against those from the alternative evaluators and measured their alignment with human judgments. Specifically, we independently computed the win rate of our oracle evaluator against each baseline for every annotator pair, and report the average win rate across all four subsets per variant in Table 1. Across annotator pairs, we additionally observed an average agreement rate of 82% and a Cohen’s kappa coefficient of 0.45 (Cohen, 1960).

## C Supplementary Experiments

Here, we provide additional experiments to further support the results and analysis in the main paper.

Additional Results on WindowsAgentArena In Figure 9, we present the full results on WindowsAgentArena, reporting the individual recovery success rates of each policy model. These results complement the average recovery success rates of the implicit and explicit recovery variants reported in Figure 5. The results are consistent with the overall trend observed across our experiments: current computer-use agents possess latent recovery capabilities rather than lacking them entirely, but they do not reliably invoke these capabilities during inference (Finding 2, Finding 3). Moreover, the results further support our Finding 4, where generalist agents exhibit stronger recovery capabilities overall. Specifically, Claude-Sonnet-4.6, GPT-5.4- CUA, and Qwen-3.5-9B achieve higher recovery success rates under implicit recovery and generally exhibit larger performance gains when explicitly instructed to perform recovery. These additional analyses further substantiate the claims and major findings presented in our original paper.

![](images/80a2728d88a505b91bdbb7c9e30cd8bb7dc1e6451d79a935bc7b61f3469db552.jpg)  
Figure 9: Expanded Results on WindowsAgentArena from Figure 5.

<table><tr><td>Model</td><td>Max 2</td><td>Max 4</td><td>Max 5</td><td>Max 10</td></tr><tr><td>Claude-Sonnet-4.6</td><td>26.98</td><td>13.34</td><td>8.76</td><td>6.37</td></tr><tr><td>GPT-5.4-CUA</td><td>17.85</td><td>6.30</td><td>5.81</td><td>5.15</td></tr><tr><td>Qwen3.5-9B</td><td>12.04</td><td>7.29</td><td>6.12</td><td>5.29</td></tr><tr><td>EvoCUA-32B</td><td>0.86</td><td>2.00</td><td>0.86</td><td>1.44</td></tr><tr><td>UI-TARS-1.5-7B</td><td>7.26</td><td>7.81</td><td>12.31</td><td>7.82</td></tr></table>

Table 8: Implicit recovery success rate (%) across diverse max recovery step budgets in OSWorld benchmark. We report the mean success rate across three different trials from three independently sampled evaluation sets.

Robust Suboptimal Implicit Recovery The results in Table 8 show that current computer-use agents struggle to perform restorative actions across different recovery budgets. Moreover, the recovery success rates of most agents consistently decrease as the maximum number of recovery steps increases, suggesting that simply allowing more steps does not resolve their recovery limitations. Notably, however, success rates do not drop to zero, suggesting that agents retain some latent ability to perform restorative actions but fail to reliably leverage it under failure scenarios. These findings further support our main-paper conclusion that current computer-use agents struggle to effectively realize their restorative capabilities when recovering from failures (Finding 2, Finding 3).

Hyperparameter Ablation Study To further validate the robustness of our findings, we provide an additional analysis of the recovery capabilities of state-of-the-art computer-use agents under different hyperparameter settings in Table 9, using the Recovery Target Contextualization setting. The results consistently show that EvoCUA-32B exhibits limited recovery capability even under explicit recovery prompting, compared to UI-TARS-1.5-7B across a wide range of hyperparameter configurations. Furthermore, the recovery success rate also remains robust across diverse hyper parameter settings for EvoCUA in WindowsAgentArena (Table 10). This finding highlights the inherent limitation of the EvoCUA-32B post-trained model, despite being extensively trained on offline trajectory data. In other words, even with explicit recovery guidance, the model remains unable to reliably restore previous states regardless of the hyper parameter settings. These observations further reinforce the results presented in Figure 4 and Figure 7.

<table><tr><td></td><td>Temperature</td><td>Max Tokens</td><td>Short Horizon</td><td>Long Horizon</td></tr><tr><td rowspan="4">EVCUA 2</td><td>1.0</td><td>32768</td><td>0.00</td><td>7.94</td></tr><tr><td>1.0</td><td>16384</td><td>1.75</td><td>4.76</td></tr><tr><td>0.0</td><td>32768</td><td>3.51</td><td>20.63</td></tr><tr><td>0.0</td><td>16384</td><td>4.39</td><td>6.35</td></tr><tr><td rowspan="4">UI-r- 1.5.-B</td><td>1.0</td><td>1000</td><td>31.67</td><td>25.00</td></tr><tr><td>1.0</td><td>500</td><td>26.67</td><td>26.67</td></tr><tr><td>0.0</td><td>1000</td><td>26.67</td><td>21.67</td></tr><tr><td>0.0</td><td>500</td><td>26.67</td><td>13.33</td></tr></table>

Table 9: Recovery Success Rate (%) across two different models with different hyper parameters using Recovery Target Contextualization in OSWorld benchmark.

<table><tr><td>Temperature</td><td></td><td>Max Tokens Top_P</td><td>Success Rate (%)</td></tr><tr><td rowspan="5">EVCUA -32B</td><td>0.7</td><td>32768 0.95</td><td>18.00</td></tr><tr><td>1.0</td><td>32768 0.95</td><td>14.00</td></tr><tr><td>1.0</td><td>16384</td><td>0.95 20.00</td></tr><tr><td>0.0</td><td>32768</td><td>0.95 20.00 0.95</td></tr><tr><td>0.0</td><td>16384</td><td>18.00</td></tr></table>

Table 10: Recovery Success Rate (%) of EvoCUA-32B with different hyper parameters using Recovery Target Contextualization in WindowsAgentArena benchmark.

![](images/fd8326cde479eb42faa4fcd6b8ac14758f2d1fec2635d731766aff09bdf5f11b.jpg)  
Figure 10: Three-run average recovery success rates for each recovery variant across three randomly sampled recovery evaluation sets in RETRACE.

Dataset Robustness Analysis In Figure 10, we show that the trend observed in Figure 4 remains consistent across different randomly sampled recovery evaluation sets. We randomly sample two additional recovery evaluation sets for respective policy model, each containing the same number of samples as the original set, and measure the recovery success rate of each policy model on all three sets (including one used in our main experiments). We then compute the macro-average success rate across the three evaluation sets for each recovery variant of each policy model. The results closely mirror the trend observed in Figure 4: explicit recovery instructions yield substantial gains over implicit recovery, demonstrating that current computer-use agents possess recovery capabilities but fail to reliably invoke them during inference (Finding 2, Finding 3). Furthermore, post-trained models such as EvoCUA exhibit nearly no recovery capability, while UI-TARS shows only modest improvements with explicit recovery instructions. In contrast, generalist agents (Claude, GPT-5.4, and Qwen) exhibit substantial gains from explicit recovery instructions. These results further support our claim, suggesting that extensive post-training on one-directional task-completion trajectories may suppress or weaken the inherent recovery capabilities of computer-use agents (Finding 4). Furthermore, in Table 8, we show that across three randomly sampled recovery evaluation sets, current computer-use agents rarely have restorative behavior (Finding 2), and its performance is robust regardless of evaluation set.

<table><tr><td>Oracle</td><td>Unaffected Rate (%)</td></tr><tr><td>RETRACE (ALL)</td><td>86.0 (%)</td></tr></table>

Table 11: Analysis on the rate of oracle evaluations unaffected by stochastic environmental dynamics.

Effect of Environment Dynamics on RETRACE In addition, we investigate the impact of stochastic transition dynamics, such as pop-ups, rendering variations, and dynamic timestamps, on the judgments of our oracle evaluation framework. Specifically, we conduct a human study following the same protocol as in Table 1 and Appendix B.3. Annotators are asked to determine whether a successfully recovered state differs from the target state only due to ignorable environmental variations. Since recovery failures identified by human annotators are not attributed to environmental dynamics (see the evaluation guidelines in Figure 11), this evaluation quantifies how often our oracle incorrectly judges a successful recovery as failure due to stochastic environmental variations. To obtain a reliable estimate, we consider only cases where both annotators agree on a judgment on both recovery success and environmental effect. From these cases, we sample instances where the human judgment disagrees with RETRACE, and both agree that there exists stochastic environmental dynamics. Finally, we eliminate the same recovery evaluation set distributed across the independent evaluation sets (GPT-5.4 evaluator, w/o $\mathcal { R } _ { \mathrm { O } }$ , w/o $\mathcal { R } _ { \mathrm { T C } } )$ from Table 1 and compute micro-average ratio. The results in Table 11 show that RETRACE is largely robust to stochastic environmental dynamics: 86.0% of the RETRACE judgments were favorable decisions unaffected by stochastic environmental dynamics. Furthermore, since this analysis is performed on a challenging subset consisting exclusively of disagreement cases between different oracles, the overall adverse effect of stochastic environmental dynamics on the judgments of RE-TRACE is expected to be even smaller in practice.

<table><tr><td>Oracle</td><td>Alignment Rate (%)</td></tr><tr><td> $\mathcal { R } _ { \mathrm { T C } }$ </td><td>38.8 (%)</td></tr><tr><td> $\mathcal { R } _ { \mathrm { { O } } }$ </td><td>61.2 (%)</td></tr><tr><td>RETRACE</td><td>81.2 (%)</td></tr></table>

Table 12: Analysis on the alignment rate of oracle judgments with human annotations using samples under $\scriptstyle { \mathcal { R } } _ { \mathrm { O } }$ and $\mathcal { R } _ { \mathrm { T C } }$ disagreement.

Analysis on the $\mathcal { R } _ { \mathrm { O } }$ and $\mathcal { R } _ { \mathrm { T C } }$ disagreements Furthermore, we analyze the quality of the judgments made by RETRACE in cases where the two oracles, $\scriptstyle { \mathcal { R } } _ { \mathrm { O } }$ and $\mathcal { R } _ { \mathrm { T C } }$ , disagree. Specifically, we sample disagreement cases between $\mathcal { R } _ { \mathrm { O } }$ and R<sub>TC</sub> from the human evaluation sets (Tables 1 and 11). Following the protocol in Table 11, we consider only samples for which both human annotators within each evaluation subset agree on a recovery success evaluation. We then measure the agreement rate between human judgments and oracle decisions when using $\mathcal { R } _ { \mathrm { O } }$ , and $\mathcal { R } _ { \mathrm { T C } }$ alone, and both oracles jointly, among such samples using the micro-average protocol demonstrated in Table 11. The results in Table 12 demonstrate that the joint use of two oracles provides complementary strength and that the combination of them yields higher agreement with human judgments than either oracle alone, compared to utilizing each oracle in isolation, further indicating that even in disagreement situations, RETRACE can provide objective evaluation results.

Analysis of RETRACE’s Sensitivity to the SSIM Threshold Finally, we analyze the sensitivity of $\mathcal { R } _ { \mathrm { O } }$ to the SSIM threshold, which is set to 0.99 by default. Table 13 reports the alignment between the judgments of $\mathcal { R } _ { \mathrm { O } }$ and human annotations under different SSIM thresholds. Following the protocol used in Tables 11 and 12, we sample a common set of instances for which both human annotators agree on whether the recovery is successful, and measure the proportion of agreement between $\mathcal { R } _ { \mathrm { O } }$ and the corresponding human judgments across different thresholds among such samples. The results show that lowering the SSIM threshold causes $\scriptstyle { \mathcal { R } } _ { \mathrm { O } }$ to overestimate recovery success, substantially reducing its alignment with human annotations. These findings justify our design choice of adopting a strict SSIM threshold of 0.99 in RETRACE to ensure high-fidelity screenshot-level matching.

<table><tr><td>Threshold</td><td>Alignment Rate (%)</td></tr><tr><td>0.99</td><td>56.9 (%)</td></tr><tr><td>0.95</td><td>33.9 (%)</td></tr><tr><td>0.91</td><td>26.6 (%)</td></tr></table>

Table 13: Analysis of the Effect of the SSIM Threshold on the Alignment Between $\mathcal { R } _ { \mathrm { O } }$ and Human Judgments.

## D Qualitative Analyses

Here, we provide additional qualitative examples that explain the behavior pattern of computer agents under suboptimal recovery demonstrated in Table 14.

![](images/1b0b079555d73b45061abdfd50e97a3a1a0ab08a7c8aefaeb8dcf251796e17eb.jpg)  
Table 14: Additional Qualitative analysis of erroneous agent behaviors induced by imperfect state restoration.

## E Human Evaluation Guidelines

In this section, we provide the guidelines for human evaluation provided to the participants in Table 1.

Human Evaluation Guideline   
You will be provided with:   
- Goal Screenshot   
Error-Step Screenshot   
Final Recovery Screenshot   
Original Instruction   
- Recovery Trajectory   
- Original Execution Trajectory   
- Your task is to determine whether   
the recovery process successfully   
restores the environment to the   
intended goal state.   
Evaluation Objective   
- Evaluate whether the Final   
Recovery Screenshot is functionally   
equivalent to the Goal Screenshot.   
- Success (1): the environment is   
correctly restored   
- Failure (0): the environment is   
not properly restored   
Evaluation Principles   
1. Prioritize Visual Similarity   
The primary signal is the visual   
similarity between the Final   
Recovery Screenshot and the Goal   
Screenshot. Please carefully   
compare the screenshots at a   
fine-grained, segment-level detail,   
including interaction-dependent UI   
elements. Pay close attention to   
whether:   
- active windows or pages differ,   
- selected content or tabs change,   
- focused text boxes differ,   
- dropdowns or menus are opened   
differently,   
- cursor positions or hover states   
change,   
- highlighted or selected UI   
elements differ,   
- drag-and-drop states change, or   
other interaction-sensitive regions   
are inconsistent.   
Mark as Failure (0) if important   
visual or interaction-level   
differences remain.   
Important Minor differences   
caused by dynamic environments   
(e.g., notifications, pop-ups,   
timestamps, loading indicators,   
rendering variations) can be ignored   
if the underlying functional state   
remains equivalent.   
2. Verify Functional Recovery   
The recovered state should allow the   
original task to continue correctly.   
Even if screenshots appear visually   
similar, mark as Failure (0) if:-   
- important interaction context is   
missing,

- unintended changes remain,   
- incorrect files/pages/windows are   
active,   
or the operational state is not   
properly restored.   
Use the recovery trajectory and   
original trajectory to understand:   
- what actions changed the   
environment,   
- whether erroneous states were   
actually reverted,   
- and whether the recovery process   
is consistent with the intended goal   
state.   
Trajectory information is especially   
important when screenshots appear   
visually similar but the actual   
operational state may still differ.   
Final Output   
Binary Label: 1 for Success   
Binary Label: 0 for Failure   
Final Output (Optional If Recovery   
is Successful)   
If you believe that the recovery   
is successful, please determine   
whether the successful recovery   
decision was despite ignorable   
observation differences caused by   
minor dynamic environmental changes   
such as notifications, pop-ups,   
timestamps, loading indicators, and   
rendering variations.   
Binary Label: 1 for YES   
Binary Label: 0 for NO  
Figure 11: Human Evaluation Guideline

## F Prompts

Here, we provide the prompts in our experiments.

```csv
System Prompt for Claude-Sonnet-4.6 For
ward Agent in OsWorld
<SYSTEM_CAPABILITY>
You are utilising an Ubuntu
virtual machine using x86_64
architecture with internet access.
<sub>*</sub> You can feel free to install
Ubuntu applications with your bash
tool. Use curl instead of wget.
<sub>*</sub> To open browser, please just
click on the Chrome icon. Note,
Chrome is what is installed on your
system.
<sub>*</sub> Using bash tool you can start
GUI applications, but you need to
set export DISPLAY=:1 and use a
subshell. For example "(DISPLAY=:1
xterm &)". GUI apps run with
bash tool will appear within your
desktop environment, but they may
take some time to appear. Take a
screenshot to confirm it did.
<sub>*</sub> When using your bash tool with
```

commands that are expected to   
output very large quantities of   
text, redirect into a tmp file and   
use str\_replace\_editor or ‘grep -n   
-B <lines before> -A <lines after>   
<query> <filename>‘ to confirm   
output.   
<sub>\*</sub> When viewing a page it can be   
helpful to zoom out so that you can   
see everything on the page. Either   
that, or make sure you scroll down   
to see everything before deciding   
something isn’t available.   
DO NOT ask users for   
clarification during task execution.   
DO NOT stop to request more   
information from users. Always   
take action using available tools.   
<sub>\*</sub> When using your computer function   
calls, they take a while to run   
and send back to you. Where   
possible/feasible, try to chain   
multiple of these calls all into   
one function calls request.   
<sub>\*</sub> TASK FEASIBILITY: You can   
declare a task infeasible at   
any point during execution -   
whether at the beginning after   
taking a screenshot, or later   
after attempting some actions and   
discovering barriers. Carefully   
evaluate whether the task is   
feasible given the current system   
state, available applications,   
and task requirements. If you   
determine that a task cannot be   
completed due to:   
- Missing required applications   
or dependencies that cannot be   
installed   
- Insufficient permissions or   
system limitations   
- Contradictory or impossible   
requirements   
- Any other fundamental barriers   
that make completion impossible   
Then you MUST output exactly   
"[INFEASIBLE]" (including the   
square brackets) anywhere in your   
response to trigger the fail action.   
The system will automatically   
detect this pattern and terminate   
the task appropriately.   
<sub>\*</sub> The current date is   
{current\_date}.   
<sub>\*</sub> Home directory of this Ubuntu   
system is ’/home/user’.   
<sub>\*</sub> If you need a password for sudo,   
the password of the computer is   
{password}.   
</SYSTEM\_CAPABILITY>   
<IMPORTANT>   
<sub>\*</sub> If the item you are looking at   
is a pdf, if after taking a single   
screenshot of the pdf it seems   
that you want to read the entire   
document instead of trying to

continue to read the pdf from your   
screenshots + navigation, determine   
the URL, use curl to download the   
pdf, install and use pdftotext to   
convert it to a text file, and then   
read that text file directly with   
your StrReplaceEditTool.   
</IMPORTANT>  
Figure 12: System Prompt for Claude-Sonnet-4.6 Forward Agent in OsWorld

```csv
System Prompt for Claude-Sonnet-4.6 For
ward Agent in WindowsAgentArena
<SYSTEM_CAPABILITY>
You are utilising a Windows
virtual machine using x86_64
architecture with internet access.
To open browser, please just
click on the Chrome icon. Note,
Chrome is what is installed on your
system.
<sub>*</sub> When viewing a page it can be
helpful to zoom out so that you can
see everything on the page. Either
that, or make sure you scroll down
to see everything before deciding
something isn’t available.
<sub>*</sub> When you want to open some
applications on Windows, please
use Double Click on it instead of
clicking once.
</SYSTEM_CAPABILITY>
```  
Figure 13: System Prompt for Claude-Sonnet-4.6 Forward Agent in WindowsAgentArena

System Prompt for Evocua-32B forward   
agent in OsWorld   
# Tools   
You may call one or more functions   
to assist with the user query.   
You are provided with function   
signatures within <tools></tools>   
XML tags:   
<tools>   
{tools\_xml}   
</tools>   
For each function call, return   
a json object with function   
name and arguments within   
<tool\_call></tool\_call> XML tags:   
<tool\_call>   
{"name": <function-name>,   
"arguments": <args-json-object>}   
</tool\_call>   
The current date is {date}

If you need a password for sudo,   
the password of the computer is   
{client\_password}   
# Response format   
Response format for every step:   
1) Action: a short imperative   
describing what to do in the UI.   
2) A single   
<tool\_call>...</tool\_call> block   
containing only the JSON: {"name":   
<function-name>, "arguments":   
<args-json-object>}.   
Rules:   
- Output exactly in the order:   
Action, <tool\_call>.   
- Be brief: one sentence for   
Action.   
- Do not output anything else   
outside those parts.   
- If finishing, use   
action=terminate in the tool call.

## Figure 14: System Prompt for Evocua-32B forward agent in OsWorld

System Prompt used for Evocua-32B for  
ward agent in WindowsAgentArena   
# Tools   
You may call one or more functions   
to assist with the user query.   
You are provided with function   
signatures within <tools></tools>   
XML tags:   
<tools>   
{tools\_xml}   
</tools>   
For each function call, return   
a json object with function   
name and arguments within   
<tool\_call></tool\_call> XML tags:   
<tool\_call>   
{"name": <function-name>,   
"arguments": <args-json-object>}   
</tool\_call>   
# Response format   
Response format for every step:   
1) Action: a short imperative   
describing what to do in the UI.   
2) A single   
<tool\_call>...</tool\_call> block   
containing only the JSON: {"name":   
<function-name>, "arguments":   
<args-json-object>}.   
Rules:   
- Output exactly in the order:   
Action, <tool\_call>.   
- Be brief: one sentence for

Action.   
- Do not output anything else   
outside those parts.   
- If finishing, use   
action=terminate in the tool call.

## Figure 15: System Prompt used for Evocua-32B forward agent in WindowsAgentArena

Tool description prompt used for Evocua-  
32B agent   
{ "type": "function",   
"function": {   
"name\_for\_human":   
"computer\_use",   
"name": "computer\_use",   
"description":   
description\_prompt,   
"parameters": {   
"properties": {   
"action": {   
"description":   
S2\_ACTION\_DESCRIPTION,   
"enum": ["key", "type",   
"mouse\_move", "left\_click",   
"left\_click\_drag",   
"right\_click",   
"middle\_click", "double\_click",   
"triple\_click",   
"scroll", "wait",   
"terminate", "key\_down", "key\_up"],   
"type": "string"   
},   
"keys": {   
"description": "Required   
only by ‘action=key‘.",   
"type": "array"   
},   
"text": {   
"description": "Required   
only by ‘action=type‘.",   
"type": "string"   
},   
"coordinate": {   
"description": "The x,y   
coordinates for mouse actions.",   
"type": "array"   
},   
"pixels": {   
"description": "The   
amount of scrolling.",   
"type": "number"   
},   
"time": {"description":   
"The seconds to wait.", "type":   
"number"},   
"status": {   
"description": "The   
status of the task.",   
"type": "string",   
"enum": ["success",   
"failure"]   
}   
},

"required": ["action"],   
"type": "object"   
},   
"args\_format": "Format the   
arguments as a JSON object."   
}   
}  
Figure 16: Tool description prompt used for Evocua-32B agent

Description Prompt used for Evocua-32B   
agent   
Use a mouse and keyboard to   
interact with a computer, and take   
screenshots.   
This is an interface to a desktop   
GUI. You must click on desktop   
icons to start applications.   
<sub>\*</sub> Some applications may take time   
to start or process actions, so   
you may need to wait and take   
successive screenshots to see the   
results of your actions. E.g. if   
you click on Firefox and a window   
doesn’t open, try wait and taking   
another screenshot.   
{resolution\_info}   
<sub>\*</sub> Whenever you intend to move the   
cursor to click on an element   
like an icon, you should consult   
a screenshot to determine the   
coordinates of the element before   
moving the cursor.   
<sub>\*</sub> If you tried clicking on a   
program or link but it failed   
to load even after waiting, try   
adjusting your cursor position so   
that the tip of the cursor visually   
falls on the element that you want   
to click.   
<sub>\*</sub> Make sure to click any buttons,   
links, icons, etc with the cursor   
tip in the center of the element.   
Don’t click boxes on their edges   
unless asked.

## Figure 17: Description Prompt used for Evocua-32B agent

Action description prompt used for Evocua-  
32B agent   
<sub>\*</sub> ‘key‘: Performs key down presses   
on the arguments passed in order,   
then performs key releases in   
reverse order.   
<sub>\*</sub> ‘key\_down‘: Press and HOLD the   
specified key(s) down in order (no   
release). Use this for stateful   
holds like holding Shift while   
clicking.

```erb
<sub>*</sub> ‘key_up‘: Release the specified
key(s) in reverse order.
<sub>*</sub> ‘type‘: Type a string of text on
the keyboard.
‘mouse_move‘: Move the cursor to
a specified (x, y) pixel coordinate
on the screen.
‘left_click‘: Click the left
mouse button at a specified (x, y)
pixel coordinate on the screen.
‘left_click_drag‘: Click and
drag the cursor to a specified (x,
y) pixel coordinate on the screen.
<sub>*</sub> ‘right_click‘: Click the right
mouse button at a specified (x, y)
pixel coordinate on the screen.
<sub>*</sub> ‘middle_click‘: Click the middle
mouse button at a specified (x, y)
pixel coordinate on the screen.
<sub>*</sub> ‘double_click‘: Double-click the
left mouse button at a specified
(x, y) pixel coordinate on the
screen.
<sub>*</sub> ‘triple_click‘: Triple-click the
left mouse button at a specified
(x, y) pixel coordinate on the
screen.
<sub>*</sub> ‘scroll‘: Performs a scroll of
the mouse scroll wheel.
‘hscroll‘: Performs a horizontal
scroll (mapped to regular scroll).
‘wait‘: Wait specified seconds
for the change to happen.
‘terminate‘: Terminate the
current task and report its
completion status. ‘answer‘:
Answer a question.
```

## Figure 18: Action description prompt used for Evocua-32B agent

Action space used for UI-Tars-1.5-7B for  
ward agent   
click(start\_box=’<|box\_start|>   
(x1,y1)<|box\_end|>’)   
left\_double(start\_box=’<|box\_start|>   
(x1,y1)<|box\_end|>’)   
right\_single(start\_box=’<|box\_start|>   
(x1,y1)<|box\_end|>’)   
drag(start\_box=’<|box\_start|>   
(x1,y1)<|box\_end|>’,   
end\_box=’<|box\_start|>   
(x3,y3)<|box\_end|>’) hotkey(key=”)   
type(content=”) #If you want to   
submit your input, use "   
n" at the end of ‘content‘.   
scroll(start\_box=’<|box\_start|>   
(x1,y1)<|box\_end|>’,   
direction=’down or up or right   
or left’)   
wait() #Sleep for 5s and take   
a screenshot to check for any   
changes.   
finished(content=’xxx’) # Use   
escape characters

", and   
n in content part to ensure we can   
parse the content in normal python   
string format.  
Figure 19: Action space used for UI-Tars-1.5-7B forward agent

System prompt used for GPT-5.4-CUA agent   
in OsWorld benchmark   
Here are some helpful tips:   
- My computer password is   
"{CLIENT\_PASSWORD}" when sudo is   
needed.   
- Stick to the website or   
application already opened for   
the task when possible.   
- Prefer Chrome over   
Firefox/Chromium unless the task   
says otherwise.   
- You can act without asking for   
confirmation.   
- If the task is infeasible, say so   
explicitly.  
Figure 20: System prompt used for GPT-5.4-CUA agent in OsWorld benchmark

System prompt used for QWEN-3.5-9B for  
ward agent in OsWorld   
You are a multi-purpose intelligent   
assistant. Based on my requests,   
you can use tools to help me   
complete various tasks.   
# Tools   
You have access to the following   
functions:   
<tools>   
{json.dumps(tools\_def)}   
</tools>   
If you choose to call a function   
ONLY reply in the following format   
with NO suffix:   
<tool\_call>   
<function=example\_function\_name>   
<parameter=example\_parameter\_1>   
value\_1   
</parameter>   
<parameter=example\_parameter\_2>   
This is the value for the second   
parameter   
that can span   
multiple lines   
</parameter>   
</function>   
</tool\_call>   
<IMPORTANT>   
Reminder:

- Function calls MUST follow   
the specified format: an   
inner <function=...></function>   
block must be nested within   
<tool\_call></tool\_call> XML tags   
- Required parameters MUST be   
specified   
- You may provide optional   
reasoning for your function call   
in natural language BEFORE the   
function call, but NOT after   
- If there is no function call   
available, answer the question like   
normal with your current knowledge   
and do not tell the user about   
function calls   
- The current date is   
{datetime.today().strftime(’%A,   
%B %d, %Y’)}.   
- Collapsed screenshots appear as   
text: {self.collapse\_text}   
If you need a password for sudo,   
the password of the computer is   
{self.client\_password}   
</IMPORTANT>   
# Response format   
Response format for every step:   
1) Action: a short imperative   
describing what to do in the UI.   
2) A single   
<tool\_call>...</tool\_call> block.   
Rules:   
- Output exactly in the order:   
Action, <tool\_call>.   
- Be brief: one sentence for   
Action.   
- Do not output anything else   
outside those parts.   
- If finishing, use   
action=terminate in the tool call.  
Figure 21: System prompt used for QWEN-3.5-9B forward agent in OsWorld

System prompt used for GPT-5.4-CUA agent   
in WindowsAgentArena   
Here are some helpful tips:   
- Stick to the website or   
application already opened for   
the task when possible.   
- Prefer Chrome over   
Firefox/Chromium unless the task   
says otherwise.   
- You can act without asking for   
confirmation.   
- If the task is infeasible, say so   
explicitly.  
Figure 22: System prompt used for GPT-5.4-CUA agent in WindowsAgentArena

## System prompt used for Qwen-3.5-9B agent in WindowsAgentArena

You are a multi-purpose intelligent   
assistant. Based on my requests,   
you can use tools to help me   
complete various tasks.   
# Tools   
You have access to the following   
functions:   
<tools>   
{json.dumps(tools\_def)}   
</tools>   
If you choose to call a function   
ONLY reply in the following format   
with NO suffix:   
<tool\_call>   
<function=example\_function\_name>   
<parameter=example\_parameter\_1>   
value\_1 </parameter>   
<parameter=example\_parameter\_2>   
This is the value for the second   
parameter   
that can span   
multiple lines   
</parameter>   
</function>   
</tool\_call>   
<IMPORTANT>   
Reminder:   
- Function calls MUST follow   
the specified format: an   
inner <function=...></function>   
block must be nested within   
<tool\_call></tool\_call> XML tags   
- Required parameters MUST be   
specified   
- You may provide optional   
reasoning for your function call   
in natural language BEFORE the   
function call, but NOT after   
- If there is no function call   
available, answer the question like   
normal with your current knowledge   
and do not tell the user about   
function calls   
- Collapsed screenshots appear as   
text: {self.collapse\_text}   
</IMPORTANT>   
# Response format   
Response format for every step:   
1) Action: a short imperative   
describing what to do in the UI.   
2) A single   
<tool\_call>...</tool\_call> block.   
Rules:   
- Output exactly in the order:   
Action, <tool\_call>.   
- Be brief: one sentence for

Action.   
- Do not output anything else   
outside those parts.   
- If finishing, use   
action=terminate in the tool call.

## Figure 23: System prompt used for Qwen-3.5-9B agent in WindowsAgentArena

Action description prompt used for   
QWEN3.5-9B agent   
‘key‘: Performs key down presses   
on the arguments passed in order,   
then performs key releases in   
reverse order.   
<sub>\*</sub> ‘type‘: Type a string of text on   
the keyboard.   
‘mouse\_move‘: Move the cursor to   
a specified (x, y) pixel coordinate   
on the screen.   
‘left\_click‘: Click the left   
mouse button at a specified (x, y)   
pixel coordinate on the screen.   
Optional ‘text‘ parameter can   
specify modifier keys (e.g.,   
"ctrl", "shift", "ctrl+shift")   
that will be held during the click.   
‘left\_click\_drag‘: Click and   
drag the cursor to a specified (x,   
y) coordinate.   
‘right\_click‘: Click the right   
mouse button at a specified (x, y)   
pixel coordinate on the screen.   
Optional ‘text‘ parameter can   
specify modifier keys that will   
be held during the click.   
‘middle\_click‘: Click the middle   
mouse button at a specified (x, y)   
pixel coordinate on the screen.   
Optional ‘text‘ parameter can   
specify modifier keys that will   
be held during the click.   
<sub>\*</sub> ‘double\_click‘: Double-click the   
left mouse button at a specified   
(x, y) pixel coordinate on the   
screen. Optional ‘text‘ parameter   
can specify modifier keys that will   
be held during the click.   
<sub>\*</sub> ‘triple\_click‘: Triple-click the   
left mouse button at a specified   
(x, y) pixel coordinate on the   
screen (simulated as double-click   
since it’s the closest action).   
Optional ‘text‘ parameter can   
specify modifier keys that will   
be held during the click.   
<sub>\*</sub> ‘scroll‘: Performs a scroll of   
the mouse scroll wheel. Optional   
‘text‘ parameter can specify a   
modifier key (e.g., "shift",   
"ctrl") that will be held during   
scrolling.   
<sub>\*</sub> ‘hscroll‘: Performs a horizontal   
scroll (mapped to regular scroll).

Optional ‘text‘ parameter can   
specify a modifier key that will   
be held during scrolling.   
<sub>\*</sub> ‘wait‘: Wait specified seconds   
for the change to happen.   
<sub>\*</sub> ‘terminate‘: Terminate the   
current task and report its   
completion status.   
<sub>\*</sub> ‘answer‘: Answer a question.

Figure 24: Action description prompt used for QWEN3.5-9B agent

```markdown
User prompt used for UI-Tars-1.5-7B for
ward agent
You are a GUI agent. You are given
a task and your action history,
with screenshots. You need to
perform the next action to complete
the task.
## Output Format
11
Thought: ...
Action:
11
## Action Space
{action_space}
## Note
- Use {language} in ‘Thought‘ part.
- Write a small plan and finally
summarize your next action (with
its target element) in one sentence
in ‘Thought‘ part.
## User Instruction
{instruction}
```

Figure 25: User prompt used for UI-Tars-1.5-7B forward agent

Tool description prompt used for QWEN3.5-   
9B agent   
"type": "function",   
"function": {   
"name": "computer\_use",   
"description":   
description\_prompt,   
"parameters": {   
"type": "object",   
"required": ["action"],   
"properties": {   
"action": {   
"type": "string",   
"description":   
action\_description\_prompt,   
"enum": [   
"key",   
"type",

```csv
"mouse_move",
"left_click",
"left_click_drag",
"right_click",
"middle_click",
"double_click",
"triple_click",
"scroll",
"hscroll",
"wait",
"terminate",
"answer",
],
},
"keys": {
"type": "array",
"description": "Required
only by ‘action=key‘."},
"text": {
"type": "string",
"description":
"Required by ‘action=type‘
and ‘action=answer‘.
Optional for
click actions (left_click,
right_click, middle_click,
double_click,
triple_click) to specify
modifier keys (e.g., ’ctrl’,
’shift’, ’ctrl+shift’).
Optional for scroll actions
(scroll, hscroll) to specify
a modifier key (e.g., ’shift’,
’ctrl’) to hold during scrolling.",
},
"coordinate": {
"type": "array",
"description": "(x, y)
coordinates." },
"pixels": {
"type": "number",
"description": "Scroll amount."},
"time": {"type":
"number", "description": "Seconds
to wait."},
"status": {
"type": "string",
"description": "Task
status for terminate.",
"enum": ["success",
"failure"],
},
},
},
},
```

## Figure 26: Tool description prompt used for QWEN3.5- 9B agent

Prefix prompt used for perturbation agent   
<Perturbation\_AGENT\_ROLE>   
<sub>\*</sub> You are a Perturbation Agent.   
Your primary objective is to   
generate series of INCORRECT   
actions that causes a significant

DRIFT from the user’s original instruction, such that the observation is completely different.

<sub>\*</sub> Your series of generated actions should lead to a completely unrelated state to the current instruction, forcing the system state into an unexpected or divergent path, where there it shows completely different scenery. You will be provided with the current state and a "correct" action (the ground truth) from the given state.

<sub>\*</sub> Your initially generated action must be DIFFERENT from the provided action and should actively disrupt the flow of the task, leading to a completely different state. Try to make the path diverge as much as possible until you believe that the scenery is completely different. <sub>\*</sub> Do not make a minor mistake; perform an active, significantly deviant steps that has nothing to do with the original goal. <sub>\*</sub> While trying to create a significant drift, try your best effort to signficantly drift the current state via a single action, instead of doing a series of small mistakes.

</Perturbation\_AGENT\_ROLE>

<Perturbation\_COMPLETION> <sub>\*</sub> To complete your task, you must output a series of actions that satisfies the following criteria, which you will be also given the history of disruptive flow of observation and actions you have performed:

1. It is DIFFERENT from the provided correct action. 2. It is a valid, executable action in the current state. 3. It causes a SIGNIFICANT DRIFT from the intended goal (completely unrelated or divergent), to the point where the observation completely changes to different scenery.

4. Since the given correct action is from the initial state, try to make the action only different from the given action when there are no past perturbation history, meaning that the correct action only applies to the inital state, and not for the rest of the perturbation process. <sub>\*</sub> Once you have generated this single unrelated action, that constitutes the completion of your turn, which you must perform multiple times until the system

state is completely different from the original instruction. <sub>\*</sub> While trying to create a significant drift, try your best effort to signficantly drift the current state via a single action, instead of doing a series of small mistakes. You should not try to complete the original task at all, and you must make a mistake or go off-track intentionally. </Perturbation\_COMPLETION>

Figure 27: Prefix prompt used for perturbation agent

## Prefix prompt used for GPT-5.4-CUA perturbation agent

<sub>\*</sub> Your generated actions must lead to a completely unrelated state, forcing the system into an unexpected or divergent path where the screen shows completely different scenery.

<sub>\*</sub> You will be provided with the current screenshot, the user’s original instruction, and the ’correct’ action (the ground truth) from the initial state.

<sub>\*</sub> Your initially generated action must be DIFFERENT from the provided correct action and should actively disrupt the flow of the task, leading to a completely different state. Make the path diverge as much as possible until the scenery is completely different.

<sub>\*</sub> Do not output waiting actions that do not alter anything, such as "wait". You must generate an active, significantly deviant step that has nothing to do with the original goal, instead of making a minor mistake, meaning that the observation and the state should drasticially change.

</Perturbation\_AGENT\_ROLE>   
<Perturbation\_COMPLETION>   
<sub>\*</sub> To complete your task, you must   
output a series of computer-use   
actions that satisfy the following   
criteria:   
1. The action is DIFFERENT from   
the provided correct action.   
2. The action is valid and   
executable in the current state.   
3. The action causes a SIGNIFICANT   
DRIFT from the intended goal   
(completely unrelated or   
divergent), to the point where   
the observation completely changes   
to different scenery.   
4. Since the given correct action   
is from the initial state, only the   
initial action must differ from the   
given correct action; subsequent   
actions only need to keep diverging   
from the original task.   
<sub>\*</sub> Once the system state is   
completely different from the   
original instruction’s expected   
scenery, you may stop emitting   
actions.   
<sub>\*</sub> You should NOT try to complete   
the original task at all. You MUST   
go off-track intentionally.   
</Perturbation\_COMPLETION>

## Figure 28: Prefix prompt used for GPT-5.4-CUA perturbation agent

Prefix prompt used for recovery agent (0-E)   
<RECOVERY\_AGENT\_ROLE>   
<sub>\*</sub> You are a Recovery Agent. Your   
primary objective is to navigate   
from the current Error State to the   
Goal State. <sub>\*</sub> The Goal State image   
is labeled with "This is the GOAL   
STATE you need to return to" - this   
is your target destination.   
<sub>\*</sub> The current state (Error Step)   
is where an error or deviation   
occurred, and you must find the   
path back to the Goal State.   
<sub>\*</sub> Analyze the differences between   
the current state and Goal State,   
then execute actions to bridge that   
gap.   
Focus on efficiency: take the   
most direct path to recovery while   
ensuring accuracy.   
</RECOVERY\_AGENT\_ROLE>   
<RECOVERY\_COMPLETION>   
<sub>\*</sub> When you believe you have   
successfully returned to the Goal   
State:   
1. Take a screenshot to verify   
the current state matches the Goal   
State

2. Compare visual elements   
carefully:   
- Window positions and sizes   
- UI component states (buttons,   
menus, dialogs)   
- Text content and input fields   
- Application state and open   
windows   
- Cursor position if relevant   
3. If the states match, output   
the "done" action to complete the   
recovery   
4. If they don’t match, identify   
remaining discrepancies and   
continue taking recovery actions   
<sub>\*</sub> Be thorough in your comparison   
- minor differences may indicate   
incomplete recovery   
<sub>\*</sub> Do not declare completion until   
you have verified through a fresh   
screenshot   
</RECOVERY\_COMPLETION>

## Figure 29: Prefix prompt used for recovery agent (0-E)

```markdown
Recovery Instruction for recovery agent (0-
E)
## RECOVERY TASK
You are a Recovery Agent. Your
task is to navigate the system back
to the Goal State.
### Goal State
The Goal State image is labeled
with "This is the GOAL STATE you
need to return to".
Use the available tools to navigate
the system back to that state.
### Current Status
- Current step: Error Step
{error_step}
- Target: Return to Goal State
(Step {goal_step})
### Your Task
Analyze the difference between the
current state and the Goal State,
then take the necessary actions
step by step to reach the Goal
State.
```

## Figure 30: Recovery Instruction for recovery agent (0- E)

Prefix prompt used for recovery agent (0-R)   
<RECOVERY\_AGENT\_ROLE>   
<sub>\*</sub> You are a Recovery Agent. Your   
primary objective is to navigate   
from the current Error State to the   
Goal State.   
<sub>\*</sub> The Goal State image is labeled

with "This is the GOAL STATE you   
need to return to" - this is your   
target destination.   
<sub>\*</sub> The current state (Error Step)   
is where an error or deviation   
occurred, and you must find the   
path back to the Goal State.   
Analyze the differences between   
the current state and Goal State,   
then execute actions to bridge that   
gap.   
Focus on efficiency: take the   
most direct path to recovery while   
ensuring accuracy.   
<sub>\*</sub> You will be provided with the   
description of the goal state for   
more detailed information to assist   
in your recovery process.   
<sub>\*</sub> You will also be provided   
with information regarding the   
transition from the initial state   
to the goal state. Use such   
information to gain knowledge of   
the goal state.   
</RECOVERY\_AGENT\_ROLE>   
<RECOVERY\_COMPLETION>   
<sub>\*</sub> When you believe you have   
successfully returned to the Goal   
State:   
1. Take a screenshot to verify   
the current state matches the Goal   
State   
2. Compare visual elements   
carefully:   
- Window positions and sizes   
- UI component states (buttons,   
menus, dialogs)   
- Text content and input fields   
- Application state and open   
windows   
- Cursor position if relevant   
3. If the states match, output   
the "done" action to complete the   
recovery   
4. If they don’t match, identify   
remaining discrepancies and   
continue taking recovery actions   
<sub>\*</sub> Be thorough in your comparison   
- minor differences may indicate   
incomplete recovery   
Do not declare completion until   
you have verified through a fresh   
screenshot   
</RECOVERY\_COMPLETION>

Figure 31: Prefix prompt used for recovery agent (0-R)  
Recovery Instruction for recovery agent (0-   
R)   
## RECOVERY TASK   
You are a Recovery Agent. Your   
task is to navigate the system back   
to the Goal State.

```markdown
### Goal State
The Goal State image is labeled
with "This is the GOAL STATE you
need to return to". Use the
available tools to navigate the
system back to that state.
You will be provided with the
description of the goal state for
more detailed information to assist
in your recovery process.
You will also be provided with
information regarding the
transition from the initial state
to the goal state. Use such
information to gain knowledge of
the goal state.
### Current Status
- Current step: Error Step
{error_step}
- Target: Return to Goal State
(Step {goal_step})
### Your Task
Analyze the difference between the
current state and the Goal State,
then take the necessary actions
step by step to reach the Goal
State.
```

## Figure 32: Recovery Instruction for recovery agent (0- R)

State-verification prompt for RETRACE   
You are an expert UI-state verifier.   
Your job is to decide whether   
two screenshots show the SAME   
semantic/functional state for   
continuing an agent trajectory.   
I will show you two screenshots:   
1. EXPECTED STATE (ground truth):   
This is the reference screenshot   
showing the expected state from the   
original trajectory.   
2. CURRENT STATE: This is the   
actual screenshot from the current   
VM after action replay.   
Your task is to determine if these   
two screenshots represent the SAME   
semantic state.   
Definition of "SAME semantic   
state":   
- An agent could continue the task   
from CURRENT STATE exactly as they   
would from EXPECTED STATE, without   
extra recovery steps.   
- The same page/view is open (same   
site/app and same logical screen).   
- The same critical UI state is   
present (same modal/popup/overlay

presence, same selected   
tab/filter/sort, same form   
values/selection states, same   
login/account state).   
- Minor cosmetic differences are   
allowed.   
Important considerations:   
- Ignore minor differences like   
exact timestamps, cursor positions,   
simple popups, information banners   
or minor visual variations.   
- Focus on the FUNCTIONAL state:   
Are the same windows/applications   
open? Are the same elements   
visible? Is the same content   
displayed?   
- Consider the overall layout and   
which UI components are active.   
Respond in the following JSON   
format:   
{   
"are\_same": true/false,   
"confidence": 0.0-1.0,   
"explanation": "Detailed   
explanation of what is the same or   
different between the two states"   
}   
Be precise about what differs   
if the states are not the same.   
Return ONLY valid JSON. No markdown.   
No extra keys.  
Figure 33: State-verification prompt for RETRACE

State transition summarizer system prompt   
for Recovery Target Contextualization   
<GOAL\_CONTEXTUALIZER\_ROLE>   
You are a Goal State Contextualizer.   
Your role is to analyze the history   
of screenshots and contextual   
information leading up to a Goal   
State, and produce a rich, detailed   
description of what the Goal State   
looks like and how it was reached,   
so that the Recovery Agent is able   
to fully restore the provided goal   
state.   
Your task is to describe the <sub>\*\*</sub>Goal   
State itself<sub>\*\*</sub> in sufficient detail   
so that it fully explains the state   
in detail.   
For the provided trajectory leading   
to the Goal State, you must:   
1. Describe the Goal State :   
Provide a comprehensive visual   
description of the Goal State   
screenshot -- what windows are   
open,   
what content is displayed, what UI   
elements are visible, what values

or data are shown.   
2. Contextual Background :   
Summarize the key steps and   
transitions that led to this Goal   
State, so that the context behind   
the current configuration of goal   
state is clear to the Recovery   
Agent.   
3. Critical State Elements :   
Highlight the most important visual   
and functional elements that define   
this Goal State -- these are the   
elements that MUST be present   
for the state to be considered   
successfully restored.   
</GOAL\_CONTEXTUALIZER\_ROLE>   
<OUTPUT\_GUIDELINES>   
<sub>\*</sub> Be precise and detailed -- the   
Recovery Agent will rely on your   
description to understand the   
target state.   
Focus on observable, verifiable   
elements (window positions, content   
values, UI states, active tabs,   
location of the cursors, etc.).   
Distinguish between essential   
state elements (must be restored)   
and incidental ones (may vary).   
Structure your response clearly   
with labeled sections.   
Do NOT describe actions or   
commands -- focus on the resulting   
state.   
</OUTPUT\_GUIDELINES>

## Figure 34: State transition summarizer system prompt for Recovery Target Contextualization

```markdown
State transition summarizer Instruction for
Recovery Target Contextualization
## GOAL STATE CONTEXTUALIZATION
TASK
You are provided with:
1. <sub>**</sub>Goal State History<sub>**</sub>: A
sequence of screenshots and
contextual information showing
the trajectory of actions and state
changes that led to the Goal State.
2. <sub>**</sub>Goal State Screenshot<sub>**</sub>: The
screenshot of the Goal State --
this is the target state that a
Recovery Agent needs to restore.
### Your Task
Analyze the history and the Goal
State screenshot to produce
a detailed, contextualized
description of the Goal State.
Focus on:
- <sub>**</sub>Visual Description<sub>**</sub>: What
```

```prolog
exactly is shown on screen -- open
windows, active applications,
displayed content, UI element
states (buttons, menus, checkboxes,
selections), file contents, browser
tabs, etc.
- Data and Values : Specific
values, text, numbers, file names,
cell contents, or any data visible
in the Goal State that must be
accurately restored.
- Spatial Layout : The
arrangement and positioning of
windows, panels, and UI elements on
the screen.
- <sub>**</sub>Contextual History<sub>**</sub>: A brief
summary of the key transitions
that led to this state, providing
context for why the state looks the
way it does.
### Output Format
Provide a structured description
with:
1. Goal State Overview : A
concise summary of the Goal State
(1-2 sentences).
2. <sub>**</sub>Detailed Visual
Description : Comprehensive
description of all visible elements
and their states.
3. <sub>**</sub>Critical Elements for
Recovery : A prioritized list
of the most important elements
that must be present for successful
recovery to this state.
4. <sub>**</sub>Context Summary<sub>**</sub>: Brief
description of how this state was
reached (what key steps led here).
```

## Figure 35: State transition summarizer Instruction for Recovery Target Contextualization

System prompt used for the Error Detector   
You are an expert Error Detector   
for GUI automation agents. Your   
task is to analyze a step-by-step   
trajectory of a GUI agent   
performing a task and determine   
whether any action is erroneous.   
# Step Indexing Convention   
- Step indices are 0-based.   
- If the trajectory is observation   
0, action 0, observation 1, action   
1, observation 2, and error is   
observed in ’observation 2’, then   
error step is 2.   
- Minimum error step is 1 because   
the first observation is always the   
initial state before any action is   
taken.   
# Input Format

You will receive:   
1. A goal instruction describing   
what the agent should accomplish.   
2. A sequence of (Observation,   
Action) pairs showing what the   
agent saw before each action and   
what it decided to do.   
3. A current observation after the   
most recent executed step. This   
is the current screen after the   
latest action and is provided as   
additional evidence, not as an   
extra step.   
# Your Task   
After reviewing the trajectory   
up to the current step, determine:   
- Whether any action in the   
trajectory is an error (i.e.,   
an action that moves the   
agent away from completing   
the goal, is incorrect, or is   
counterproductive).   
- If an error is detected, identify   
which step contain the error.   
# Retrospective Detection   
You may recognize that an earlier   
step was erroneous based on later   
observations. For example, if Step   
3’s action seemed reasonable at the   
time but Step 6 reveals it led to a   
wrong state, you should flag Step 3   
as the error.   
# Output Format   
You MUST respond with a valid JSON   
object and nothing else. Do not   
include any text before or after   
the JSON.   
If NO error is detected:   
{"is\_error": false, "error\_step":   
null, "reasoning": ""}   
If an error IS detected:   
{"is\_error": true, "error\_step":   
<integer>, "reasoning": "Detailed   
explanation of what went wrong and   
why this step is erroneous."}   
Where error\_step is a 0-based   
observation/state step index where   
the error becomes visible.   
# Important Guidelines   
- Be conservative: only flag   
clear errors, not merely suboptimal   
actions. You got only one chance   
to detect errors, so you should be   
careful.   
- Consider the full context of the   
goal when evaluating actions.   
- An action that seems wrong in

![](images/bc645fc1530c21efc6e2914ee9827fbb5f0eb36830a286f218929e1870aee99c.jpg)  
Figure 37: System prompt used for the Recovery Step Detector

Prompt used for GPT-5.4 evaluator in Table   
You are a strict and robust   
verifier responsible for   
determining whether the given   
target state has been successfuly   
reached from an erroneous starting   
state.   
Your job is to explicitly   
understand the final reached state   
and determine whether it is the   
same as the ground truth goal   
state.   
You will be provided with:   
1) The ground truth goal state that   
must be reached.   
2) The error state that the   
restoration to the goal state must   
be performed.   
3) Final state of the trajectory   
from a computer agent. 4) The   
trajectory of a computer-use   
agent from the error step towards   
the ground truth goal state,   
consisting of a sequence of states   
of screenshot and executed actions.   
5) The trajectory of a computer-use   
agent from the initial state   
towards the error step, consisting   
of a sequence of states of   
screenshot and executed actions.   
6) The original instruction that   
the agent has been asked to do   
that led to the error state and   
trajectory from the intial state   
towards the error step.   
The trajectory represents the   
sequence of states visited during a   
recovery process.   
Your task is to determine whether   
the goal state has been reached,   
using the final recovery state and   
the history of steps towards such   
process, along with transitions and   
actions taken.   
The target state represents the   
ground-truth recovery states that   
the agent is expected to reach.   
Rules:   
1. Target states are provided   
as visual screenshots of the   
environment.   
2. Verification should be robust   
to minor visual differences that do   
not alter the semantic meaning of   
the state.   
3. Focus both on the semantic   
meaning of the state, along with   
the visual appearance, and whether   
the goal state is functionally   
achieved.

4. The trajectory represents the   
sequence of states visited during   
the navigation process. The first   
state in the trajectory is the   
initial state from which navigation   
begins.   
Output format (STRICT): Output only   
true or false.   
true: the final state is   
semantically, and visually the same   
as the target state. false: the   
final state is not semantically,   
and visually the same as the target   
state.   
Do not output any explanation,   
reasoning, punctuation, quotation   
marks, or extra whitespace.

## Figure 38: Prompt used for GPT-5.4 evaluator in Table 1