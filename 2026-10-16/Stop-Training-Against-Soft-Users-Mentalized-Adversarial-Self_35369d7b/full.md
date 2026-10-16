# Stop Training Against Soft Users: Mentalized Adversarial Self-Play for Proactive Dialogue

Zanwu Liu<sup>2,\*</sup>, Dan Zhang<sup>1,\*</sup>, Xiaodi Wang<sup>1,\*</sup>, Yongbo Mao<sup>1</sup>, Xiaoyang Guo<sup>1</sup>

Yonglin Wu<sup>1</sup>, Peiyue Jiang<sup>1</sup>, Yangyang Wei<sup>1</sup>, Zecheng Lin<sup>1,†</sup>

<sup>1</sup>Dongchezu Technology Co., Ltd., Beijing, China

<sup>2</sup>School of Artificial Intelligence, Beihang University, Beijing, China

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

First author: liuaudi801@gmail.com Correspondence: chengyi.2024@dcarlife.com

## Abstract

Proactive dialogue agents must guide multiturn interactions toward task goals while adapting to users whose intentions and attitudes evolve. Existing planners improve system side action selection through search, learned policies, or user aware planning, but are often trained and evaluated against fixed or overly compliant simulators; as a result, high simulated success can reflect soft user bias rather than robust dialogue ability. We introduce MASP, a Mentalized Adversarial Self Play framework that treats proactive dialogue as a partially observable game between a system and a rationally resisting user. MASP represents the user’s evolving mental state with a BDI based latent state, separates reward side teacher mentalization from policy side student mentalization, and alternates self play updates for the system and user policies. This turns robustness to rational resistance into a training objective rather than only an evaluation condition. Across diverse proactive dialogue benchmarks, compared with the widely used SFT and Planner-only approaches, MASP substantially improves success rate (SR) and rationality constrained success (RCSR), reduces average turns (AT), and performs better against hard users who make the task goal more difficult to achieve.

## 1 Introduction

Proactive dialogue requires more than a fluent next response. In tasks such as persuasion, emotional support, empathy, and negotiation, an agent must guide a multi-turn interaction toward a goal while anticipating how each utterance may change the user’s beliefs, desires, and decisions. Although large language models (LLMs) are strong reactive conversationalists, proactive dialogue remains challenging because next-token optimization alone does not provide long-range strategic adaptation (He et al., 2024a,b). Recent work has studied various planning and simulation approaches to this challenge (Wang et al., 2023; Hong et al., 2023; Liu et al., 2025; Wu et al., 2025; Zhang et al., 2025a).

Most progress, however, remains centered on planners. Prior methods improve system-side action selection through trainable planners, search, and user world modeling (Deng et al., 2023; He et al., 2024a; Zhang et al., 2024b; He et al., 2025, 2024b; Rakib et al., 2025; Kim et al., 2025; Dao and Liao, 2025; Dong et al., 2025). Despite their different mechanisms, they largely preserve the same training asymmetry: the system is optimized as the sole strategic learner, whereas the user is fixed, externally specified, or only weakly modeled.

This asymmetry can lead to misleading progress. A policy trained against a fixed or overly permissive simulator may achieve high simulated success while learning brittle behavior that fails once the user is less compliant (Kreyssig et al., 2018; Keizer et al., 2023; Zhang et al., 2024a). Recent LLMbased work moves beyond prompted role-play toward dedicated user LMs, profile-conditioned simulators, and iterative adversarial refinement (Davidson et al., 2023; Sekuli’c et al., 2024; Naous et al., 2025; Zhu et al., 2025). These works make the soft user problem explicit, but they mainly strengthen the simulator or evaluation environment; they do not directly answer how a proactive policy should be trained so that its gains survive stronger, more strategic, and partially observed users.

A natural way to address this issue is to model the user’s evolving internal state. Recent studies have examined intention modeling, user state tracking, and theory of mind for persuasion and emotional support (Yu et al., 2025; Zhang et al., 2025b; Park et al., 2026), yet such information remains an auxiliary feature or simulator-side component rather than the basis of the training interaction. We propose MASP (Mentalized Adversarial Self

Play)<sup>1</sup>, a training paradigm that instead formulates proactive dialogue as a partially observable game between two agents over a shared latent mental state space. Each turn is governed by a hidden user state represented by free-form belief, desire, and intention (BDI) together with compact affective variables. The user policy conditions on this hidden state, while the system observes only an estimate inferred from dialogue history; consequently, the system is trained not only to choose useful utterances, but also to infer who it is talking to and how that person changes over time.

The key design of MASP is to make mentalization part of the optimization itself. A frozen teacher mentalizer defines the latent state used by reward computation, whereas the system observes only a student estimate, reducing the risk that the policy directly exploits an adaptive reward state estimator. On top of this latent state, MASP defines dense progress toward a task-specific target mind state, measured relative to the user’s own starting state. The system is rewarded for both moving the user toward the target and accurately inferring the current state, while the user is rewarded for rational resistance rather than unconditional rejection. This formulation differs from planner-focused approaches by coupling opponent modeling, longhorizon dialogue control, and simulator robustness within a single training game; it also differs from work that only builds simulators because the user model is not merely an evaluation tool, but an adaptive training opponent.

We evaluate MASP on proactive dialogue benchmarks spanning persuasion, emotional support, empathy, and negotiation, with a focus on rationalityconstrained success under difficult user simulators. Our contributions are threefold: (1) to our knowledge, MASP is the first multi-turn dialogue training framework to explicitly model BDI mental states and let those states evolve during training; (2) we are the first to apply self-play to a multi-turn dialogue training framework, proposing a mentalized self-play method that jointly trains the system and user and is designed to avoid the soft user problem in practical applications; and (3) to address the fact that widely used evaluation rules do not capture model performance against difficult users in realistic settings, we evaluate models under difficult user simulators and introduce rationality-constrained success rate to better reflect performance with nearrealistic users.

## 2 Related Work

Proactive dialogue planning has progressed from search at test time to trainable planners and policy learning with user information. GDP-Zero uses Monte Carlo Tree Search over LLM simulated continuations (Yu et al., 2023), while PPDPP trains a planner module with supervised fine tuning and self play reinforcement learning (Deng et al., 2023). Later methods improve planning through dual process deliberation, diversified user simulation, latent policy learning, intrinsic user world modeling, and online selection based on value estimates over frozen LLM proposals (He et al., 2024a; Zhang et al., 2024b; He et al., 2024b, 2025; Rakib et al., 2025). These methods strengthen system action selection, but still largely optimize a system planner against fixed, prompted, or otherwise externally specified users.

User simulation is central to dialogue training and evaluation, but simulator realism remains a bottleneck (Shi et al., 2019; Davidson et al., 2023; Sekulic et al., 2024; Luo et al., 2024; Wang et al., 2025b). Prior work shows that simulator choice affects policy quality and transferability, and that offline policy learning can overfit to simulator artifacts (Kreyssig et al., 2018; Keizer et al., 2023; Zhang et al., 2024a). Recent LLM based studies make this issue sharper: Flipping the Dialogue trains dedicated User LMs from converted human assistant conversations (Naous et al., 2025), and DIAL improves simulator realism through iterative adversarial refinement (Zhu et al., 2025). These works improve simulators or evaluation environments, but do not directly turn user resistance into a coupled training objective for the system policy.

MASP also connects to adversarial and multiagent training. SMAC-Hard shows that narrow opponent distributions can induce brittle multiagent policies (Deng et al., 2024), and GAIA formulates noncollaborative dialogue planning as a two player zero sum game with self play and Chain-of-Mind reasoning (Wang et al., 2025a). MASP shares the concern that soft or narrow user models produce brittle policies, but differs by modeling proactive dialogue as a partially observable game over a latent mental state geometry: the system sees only a student estimate of the user’s hidden state, while the user is trained to resist rationally.

## 3 Method

As shown in Figure 1, MASP is organized as a three-phase training pipeline. Phase 0 trains BDI mentalizers from silver annotations; Phase 1 uses behavior cloning to obtain warm-start and reference policies for the system and user; Phase 2 alternates GRPO updates for the system and user policies with frozen reward-side modules.

## 3.1 Problem Formulation

An episode is defined as

$$
\xi = ( p _ { u } , g , h _ { 0 } ) ,\tag{1}
$$

where $p _ { u }$ denotes an optional user profile, g is the task goal, and $h _ { 0 }$ is the initial dialogue context. If a dataset does not provide user profiles, we simply set $p _ { u } = \emptyset$

At turn t, the current history prefix is denoted by $h _ { t }$ . The system first generates an utterance $a _ { t } .$ , after which the user generates a response $u _ { t }$ , yielding the transition

$$
h _ { t + 1 } = h _ { t } \oplus a _ { t } \oplus u _ { t } .\tag{2}
$$

We model this interaction as a partially observable game between two agents. The user policy conditions on its current latent mental state $z _ { t } .$ , while the system policy only has access to an estimate $\hat { z } _ { t } \mathrm { : }$

$$
a _ { t } \sim \pi _ { \theta } ( \cdot \mid h _ { t } , \hat { z } _ { t } , g ) ,\tag{3}
$$

$$
u _ { t } \sim \pi _ { \psi } ( \cdot \mid h _ { t } , z _ { t } , p _ { u } , g ) .\tag{4}
$$

Here, $\pi _ { \theta }$ is the system policy and $\pi _ { \psi }$ is the user policy. The asymmetry between $z _ { t }$ and $\hat { z } _ { t }$ is essential: unlike user aware planners that condition directly on fixed personas or externally provided user descriptions, MASP treats the user’s internal state as hidden and dynamically evolving across turns.

The two agents optimize different objectives. Let $r _ { t } ^ { S }$ and $r _ { t } ^ { U }$ denote the reward at each turn for the system and the user, respectively. Their discounted objectives are

$$
\begin{array} { r l } & { J _ { S } ( \theta ) = \mathbb { E } \left[ \displaystyle \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t } ^ { S } \right] , } \\ & { J _ { U } ( \psi ) = \mathbb { E } \left[ \displaystyle \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t } ^ { U } \right] . } \end{array}\tag{5}
$$

All reported self-play runs use $\gamma = 0 . 9 7$ . As we show below, both rewards are defined in a shared latent state geometry, but they induce different behaviors: the system is rewarded for progress and accurate mentalization, whereas the user is rewarded for rational resistance.

## 3.2 Latent Mental State and Mentalization

A central question is how to represent the user’s internal state. A pure dialogue embedding is flexible but opaque, making it difficult to specify meaningful target states; task-specific slot schemas are interpretable but transfer poorly across domains. BDI provides a useful middle ground: beliefs, desires, and intentions are semantically interpretable enough to anchor target state design, yet general enough to apply across persuasion, support, empathy, and negotiation. We therefore adopt a BDIbased hybrid representation:

$$
m _ { t } = ( B _ { t } , D _ { t } , I _ { t } , \rho _ { t } , c _ { t } , v _ { t } ) ,\tag{6}
$$

where $B _ { t } , D _ { t }$ , and $I _ { t }$ are free form natural language descriptions of the user’s belief, desire, and intention, respectively; $\rho _ { t } \in [ 0 , 1 ]$ denotes receptivity, $c _ { t } \in [ 0 , 1 ]$ denotes commitment or confidence, and $v _ { t } \in [ - 1 , 1 ]$ denotes valence. The textual BDI component captures what the user currently believes, wants, and plans to do, while the three scalar variables provide a compact affective geometry that remains stable across tasks.

We encode the three textual BDI components separately with a frozen text encoder $E ( \cdot )$ . In our implementation, E is the corresponding frozen Qwen3 backbone used as a sentence encoder: we mean-pool its final hidden states and $\ell _ { 2 }$ normalize the pooled vector.

$$
e _ { B } = E ( B _ { t } ) , \quad e _ { D } = E ( D _ { t } ) , \quad e _ { I } = E ( I _ { t } ) ,\tag{7}
$$

where each component embedding is normalized to unit length. We then define the final vector state as

$$
z _ { t } = [ e _ { B } ; e _ { D } ; e _ { I } ; \sqrt { \alpha _ { \rho } } \rho _ { t } ; \sqrt { \alpha _ { c } } c _ { t } ; \sqrt { \alpha _ { v } } v _ { t } ] .\tag{8}
$$

This preconditions the scalar dimensions by the same weights used in the latent metric, while keeping the representation deliberately dynamic: it describes the user’s current state, not a static persona category. In persuasion, support, empathy, or negotiation, what matters is not only who the user is, but what the user currently believes and how that state changes over the course of a dialogue.

![](images/b2ff7c510b5ba6f85a67963989de4fa1c555ed5ad5aeb91cf4bbde371969e671.jpg)  
Figure 1: Overview of MASP. The framework trains BDI mentalizers, warm starts system and user policies with behavior cloning, and then performs alternating GRPO self play with latent progress, rationality, and KL regularization to reference policies.

To obtain this latent state, MASP uses two mentalization modules. A teacher mentalizer $F _ { \omega }$ predicts the latent state from dialogue history, optional profile, and task goal:

$$
z _ { t } = F _ { \omega } ( h _ { t } , p _ { u } , g ) .\tag{9}
$$

This teacher is pretrained on offline silver annotations and then frozen. Crucially, it is the only state estimator used on the reward side. The system policy never observes $z _ { t }$ directly. Instead, it receives a student estimate

$$
\hat { z } _ { t } = M _ { \phi } ( h _ { t } ) ,\tag{10}
$$

where $M _ { \phi }$ only conditions on the observable dialogue history. This split between teacher and student is one of the key design choices of MASP. Because the reward depends on the frozen teacher state while the policy only sees a student estimate, the policy must learn to infer the user rather than directly exploit the reward model. In other words, mentalization is not just an auxiliary diagnostic module; it becomes part of the optimization problem itself.

The offline silver annotations also define a BDI cache. The cache stores dialogue-prefix records with silver BDI states, and also stores a sessionlevel initial BDI state for each training dialogue. We use the term BDI prior for the non-parametric pool of these initial training states, together with any available profile text. This prior is used to sample initial user states in self play and to render continuous student estimates into text by nearestneighbor retrieval in the encoded BDI space.

## 3.3 User Relative Progress in Latent Space

For each task, we instantiate a target latent state $z ^ { \star }$ from a fixed task-level BDI target template, rather than extracting it from test dialogues. The template contains belief, desire, and intention descriptions for a successful end state, plus scalar targets for receptivity, confidence, and valence; it is encoded with the same $E ( \cdot )$ and scalar preconditioning as Eq. (8). For instance, in persuasion the target describes a receptive user who intends to donate, while in emotional support it describes reduced distress and a more constructive intention. Appendix B.2 lists the concrete templates used in the experiments. Since the BDI embedding may have much higher dimensionality than the scalar variables, we use a weighted metric. The scalar importance weights are applied by pre-scaling the scalar coordinates in Eq. (8), through the factors $\sqrt { \alpha _ { \rho } } , \sqrt { \alpha _ { c } }$ , and $\sqrt { \alpha _ { v } }$ . The matrix

$$
A = \mathrm { d i a g } \biggl ( { \frac { 1 } { 3 d } } I _ { 3 d } , 1 , 1 , 1 \biggr ) ,\tag{11}
$$

therefore serves to normalize the high-dimensional textual BDI block relative to the three already weighted scalar coordinates, where d is the encoder dimension for one BDI component. Its scalar diagonal entries are set to one because $\alpha _ { \rho } , \alpha _ { c } , \alpha _ { v }$ have already been absorbed into $z _ { t }$ . We denote the corresponding norm by

Algorithm 1 MASP training   
Require: Dialogue data D, BDI cache, teacher $F _ { \omega }$ , student   
$M _ { \phi } ,$ , system policy π<sub>θ</sub>, user policy $\pi _ { \psi }$   
1: Train teacher $F _ { \omega }$ on silver BDI annotations and freeze it.   
2: Train student $M _ { \phi }$ to predict the teacher state from dia  
logue history.   
3: Warm start π and $\pi _ { \psi }$ with supervised learning; save   
frozen references $\pi _ { \boldsymbol { \theta } _ { 0 } }$ and $\pi _ { \psi _ { 0 } } .$   
4: for outer iteration $k = 1 , \ldots , K$ do   
System update   
5: Freeze π and reward side modules.   
6: Roll out dialogues with π and $\pi _ { \psi } .$   
7: Compute $r _ { S } = w _ { \mathrm { p r o g } } \Delta s - w _ { \mathrm { i n f } } \| \hat { z } - z \| _ { A } ^ { 2 } .$   
8: Update $\pi _ { \theta }$ with GRPO style grouped advantages and   
KL to $\pi _ { \theta _ { 0 } } .$   
User update   
9: Freeze π and reward side modules.   
10: Roll out dialogues with the updated π<sub>θ</sub> and current   
$\pi _ { \psi } .$   
11: Compute $r _ { U } = - w _ { \mathrm { p r o g } } \Delta s + w _ { \mathrm { r a t } } q .$   
12: Update $\pi _ { \psi }$ with GRPO style grouped advantages and   
KL to $\pi _ { \psi _ { 0 } } .$   
Student refresh   
13: Collect self play prefixes and teacher targets.   
14: Update $M _ { \phi }$ on self play prefixes mixed with real dia  
logue prefixes.   
15: end for

$$
\| x \| _ { A } ^ { 2 } = x ^ { \top } A x .\tag{12}
$$

Minimizing the absolute distance between $z _ { t }$ and $z ^ { \star }$ is inappropriate because different users begin from different initial states: a trajectory that is clearly improving for one user may remain far from the global target, while another user may start close without intervention. We therefore define user-relative progress as the projection of the current state onto the direction from the user’s own initial state z<sub>0</sub> to the target $z ^ { \star }$ :

$$
s _ { t } = \frac { ( z _ { t } - z _ { 0 } ) ^ { \top } A ( z ^ { \star } - z _ { 0 } ) } { ( z ^ { \star } - z _ { 0 } ) ^ { \top } A ( z ^ { \star } - z _ { 0 } ) + \varepsilon _ { \mathrm { p r o g } } } ,\tag{13}
$$

and define the incremental progress by

$$
\Delta s _ { t } = s _ { t + 1 } - s _ { t } .\tag{14}
$$

We set $\varepsilon _ { \mathrm { p r o g } } = 1 0 ^ { - 6 }$ for numerical stability. This formulation measures whether the dialogue is moving in the right direction from the user’s own starting point and yields a dense reward signal at every turn, even when task success itself is sparse.

## 3.4 Adversarial Rewards

The system should both influence the user and understand the user. We therefore define the system

reward as

$$
r _ { t } ^ { S } = w _ { \mathrm { p r o g } } \Delta s _ { t } - w _ { \mathrm { i n f } } \| \hat { z } _ { t } - z _ { t } \| _ { A } ^ { 2 } .\tag{15}
$$

The first term rewards movement toward the target state, while the second rewards accurate mentalization. The inference term is especially important: it makes the system optimize not only for what to say, but also for how well it understands the person it is talking to.

The user policy is trained adversarially, but not as an unconstrained adversary. A user that always rejects everything is not realistic, and a user that always agrees is equally uninformative. To prevent both collapse modes, we define

$$
r _ { t } ^ { U } = - w _ { \mathrm { p r o g } } \Delta s _ { t } + w _ { \mathrm { r a t } } q _ { t } ,\tag{16}
$$

where $q _ { t } \in \{ - 1 , 0 , 1 \}$ is a fixed rationality judge signal indicating whether the user response is rational under the current mental state. The first term rewards resistance to unjustified state change, while the second rewards coherent and rational behavior. Intuitively, the user is not rewarded for rejecting the system blindly; it is rewarded for resisting when resistance is warranted and yielding when change is genuinely justified. This is what makes the user a useful training opponent rather than merely a harder simulator.

The scalar rollout rewards in Equations 15 and 16 do not include KL penalties. Instead, both policies are regularized by a token level KL term inside the GRPO training objective against their supervised warm start reference policies, as described in Appendix B.4. This matches the implementation: rewards provide task, mentalization, and rationality feedback, while the trainer applies the KL term to the reference policy.

## 3.5 Training

Training follows the three stage pipeline in Algorithm 1. We first train teacher and student mentalizers from silver BDI annotations, then use behavior cloning to initialize the system and user policies as warm start references, and finally run alternating GRPO self play with frozen reward side modules. The algorithm is the high level flow used in the implementation; additional optimization and prompting details are provided in Appendix B.1.

## 4 Experiments

Datasets. We evaluate on four dialogue benchmarks with multiple turns, spanning proactive and socially grounded interaction: Persuasion for Good (P4G) (Wang et al., 2019), ESConv (Liu et al., 2021), EmpatheticDialogues (Rashkin et al., 2019), and CraigslistBargain (He et al., 2018). P4G consists of charity persuasion dialogues where a persuader encourages a persuadee to donate, with user level traits and strategy annotations for a subset of conversations. ESConv contains help seeker and supporter conversations grounded in emotional support strategies. EmpatheticDialogues provides emotion grounded open domain conversations for evaluating empathetic response generation. CraigslistBargain contains price negotiations between buyers and sellers over online listings. Together, these datasets cover persuasion, emotional support, empathy, and negotiation, spanning both cooperative and competitive proactive dialogue settings. Appendix C provides dataset statistics and task specific success criteria.

Table 1: Main results on proactive dialogue benchmarks; soft/hard meanings are defined in Evaluation Metrics.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="3">Soft User</td><td colspan="3">Hard User</td></tr><tr><td>SR↑</td><td>AT↓</td><td>RCSR↑</td><td>SR↑</td><td>AT↓</td><td>RCSR↑</td></tr><tr><td rowspan="3">P4G</td><td rowspan="3">Qwen3-4B</td><td>MASP</td><td>0.859</td><td>4.08</td><td>0.556</td><td>0.788</td><td>4.56</td><td>0.566</td></tr><tr><td>SFT</td><td>0.616</td><td>6.08</td><td>0.384</td><td>0.576</td><td>6.23</td><td>0.455</td></tr><tr><td>Planner-only</td><td>0.677</td><td>4.88</td><td>0.434</td><td>0.616</td><td>5.55</td><td>0.384</td></tr><tr><td rowspan="3"></td><td rowspan="3">Qwen3-0.6B</td><td>MASP</td><td>0.707</td><td>5.14</td><td>0.465</td><td>0.626</td><td>5.76 7.35</td><td>0.434 0.242</td></tr><tr><td>SFT</td><td>0.414 0.424</td><td>6.99 6.32</td><td>0.253 0.232</td><td>0.354 0.364</td><td>6.84</td><td>0.202</td></tr><tr><td>Planner-only</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">ESConv</td><td rowspan="3">Qwen3-4B</td><td>MASP SFT</td><td>0.990 0.467</td><td>1.96 7.17</td><td>0.949 0.410</td><td>1.000 0.390</td><td>1.88 7.27</td><td>0.964 0.333</td></tr><tr><td>Planner-only</td><td>0.969</td><td>2.38</td><td>0.923</td><td>0.980</td><td>2.08</td><td>0.918</td></tr><tr><td>MASP</td><td>1.000</td><td>2.11</td><td>0.949</td><td>1.000</td><td>1.65</td><td>0.928</td></tr><tr><td rowspan="3"></td><td rowspan="3">Qwen3-0.6B</td><td>SFT</td><td>0.297 0.954</td><td>7.56 2.77</td><td>0.226 0.897</td><td>0.323 0.990</td><td>7.32 1.84</td><td>0.241 0.903</td></tr><tr><td>Planner-only</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MASP</td><td>0.689 0.646</td><td>4.60</td><td>0.073 0.045</td><td>0.621 0.558</td><td>5.61 6.28</td><td>0.259 0.234</td></tr><tr><td rowspan="3">Craigslist</td><td rowspan="3">Qwen3-4B</td><td>SFT Planner-only</td><td>0.431</td><td>5.38 5.67</td><td>0.020</td><td>0.315</td><td>6.86</td><td>0.091</td></tr><tr><td></td><td>0.728</td><td>4.24</td><td>0.030</td><td>0.641</td><td>5.50</td><td>0.211</td></tr><tr><td>MASP SFT</td><td>0.726</td><td>4.84</td><td>0.057</td><td>0.547</td><td>6.47</td><td>0.191</td></tr><tr><td rowspan="3">Empathetic</td><td rowspan="3">Qwen3-4B</td><td>Planner-only</td><td>0.172</td><td>7.07</td><td>0.011</td><td>0.484</td><td>6.41</td><td>0.129</td></tr><tr><td>MASP</td><td>0.092 0.030</td><td>7.46 7.89</td><td>0.059 0.017</td><td>0.072 0.022</td><td>7.57 7.91</td><td>0.048 0.013</td></tr><tr><td>SFT Planner-only</td><td>0.042</td><td>7.75</td><td>0.027</td><td>0.037</td><td>7.78</td><td>0.024</td></tr><tr><td rowspan="3"></td><td rowspan="3">Qwen3-0.6B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MASP</td><td>0.063</td><td>7.64</td><td>0.044</td><td>0.036</td><td>7.78</td><td>0.024</td></tr><tr><td>SFT Planner-only</td><td>0.013 0.032</td><td>7.95 7.82</td><td>0.007 0.023</td><td>0.002 0.021</td><td>7.99 7.88</td><td>0.001 0.013</td></tr></table>

Closed-loop evaluation. Each evaluation episode starts from a held-out initial context. At turn t, the system generates $a _ { t } .$ , the user policy generates $u _ { t } .$ , and both are appended to the history $h _ { t + 1 }$ The rationality judge then reads the full accumulated dialogue rather than an isolated response. This loop continues until success or the eight-exchange limit. Appendix C.2 provides the complete protocol and a concrete rollout.

Rationality-Constrained Success Rate. Conventional success rate can overestimate proactive dialogue ability when a user simulator accepts the system’s proposal without sufficient justification. We therefore introduce RCSR@T, which counts a dialogue as successful only when the task is completed and the user behavior up to completion remains rational. After each assistant–user exchange, a fixed LLM-based rationality judge receives the task description, the current BDI state $z _ { t } ,$ , the assistant utterance $a _ { t } .$ , and the user reply $u _ { t }$ The judge answers RATIONAL or NOT\_RATIONAL; we map these labels to a turn score $q _ { t } = + 1$ and $q _ { t } = - 1$ , respectively, and assign $q _ { t } = 0$ when the judge output is unparseable or the call fails. If a dialogue first succeeds at turn $\tau _ { s } \leq T$ , its successprefix rationality is

$$
\bar { q } _ { \tau _ { s } } = \frac { 1 } { \tau _ { s } } \sum _ { t = 1 } ^ { \tau _ { s } } q _ { t } .\tag{17}
$$

The episode-level rationality-constrained success indicator is

$$
\mathrm { R C S R @ T } = \mathbb { I } [ \tau _ { s } \le T \ \land \ \bar { q } _ { \tau _ { s } } \ge \tau _ { q } ] ,\tag{18}
$$

and failed dialogues receive zero. We report the mean of this indicator over evaluation dialogues with $\tau _ { q } = 0 . 7 0$ . This threshold requires the successful prefix to be consistently rational while still tolerating occasional unparseable or weak judgments in longer interactions. Unlike SR alone, RCSR filters successes caused by simulator sycophancy or unjustified concessions, and is therefore closer to evaluating whether a model can succeed with near-realistic users, especially difficult users who should resist until given convincing reasons. The exact judge prompt is given in Appendix C.2.

Evaluation Metrics. For each method, we evaluate under soft and hard user conditions, where soft denotes the easier simulator for the system under the dataset’s task objective and hard denotes the more difficult one. For P4G, ESConv, and EmpatheticDialogues, these correspond to the conventional imitation-style and learned adversarial users, respectively; CraigslistBargain reverses this convention because the system is the buyer and the Phase 1 BC seller is harder for buyer-side utility (see Appendix C.7). We report SR@8, the fraction of dialogues that reach the task-specific success criterion within $T = 8$ turns; AT, the average turns to success with failures counted as $T ;$ and RCSR@8, success under the rationality constraint defined above. Thus, SR and AT measure task effectiveness and efficiency, while RCSR tests whether success reflects rational user behavior rather than degenerate or overly compliant interactions.

Implementation. We compare MASP with two controlled baselines under matched backbones, user settings, turn budgets, and success judges within each dataset. SFT uses the supervised dialogue policy without self play. Planner-only uses the same system side warmup and adaptive planning scaffold as MASP, but does not train the adversarial user policy through mentalized self play. MASP is our full method, which further trains the system and user policies with alternating GRPO style self play. Unless otherwise specified, we report results for both Qwen3-4B and Qwen3-0.6B backbones. Dataset specific success definitions, judge prompts, sampling settings, and additional implementation details are provided in Appendix C.

Main Results. Table 1 shows that MASP consistently improves over both SFT and Planner-only on the completed benchmarks. On P4G, MASP gives the best SR, AT, and RCSR under both soft and hard users for both backbones, indicating that mentalized self play improves not only task completion but also rationality constrained success. On ESConv, MASP reaches near-ceiling SR, but this reflects the prior success convention where somewhat\_better counts as successful rather than requiring full resolution. We therefore treat ESConv mainly as transfer and efficiency evidence and avoid overinterpreting saturated SR. On CraigslistBargain, MASP again leads on SR and AT for both backbones, including the buyerside hard setting where the Phase 1 BC seller is less favorable to the buyer. The main exception is Qwen3-0.6B soft RCSR, where SFT slightly exceeds MASP; this is a small speed-rationality tradeoff in the most permissive seller setting, not a failure of the overall negotiation behavior. The very low Planner-only 0.6B soft SR is not a table bug: in the buyer role, the fixed planner scaffold can still miss closure against a seller whose concessions do not match its hand-written closing strategy, so the learned buyer policy is needed to exploit the phase 2 seller distribution. On EmpatheticDialogues, absolute SR is lower because success requires a strict resolved outcome, but MASP still yields the best results for soft and hard users across both backbones. A P4G horizontal comparison with prior systems, reported in Appendix A.1, shows the same trend in absolute comparison.

## 4.1 Ablation Study

We further isolate the contribution of each MASP component on P4G, following the component analysis style used in prior P4G systems (Deng et al., 2023; Rakib et al., 2025). Table 2 reports Qwen3- 4B results on the 99 example test split. The No GRPO self play row keeps only the supervised Phase 1 policy, while No adversarial self play uses the Planner-only scaffold without mentalized adversarial self play. The remaining rows ablate the P4G specific inference scaffold by removing the warmup policy, the adaptive close injection, or both.

The full model gives the strongest overall result, especially in SR and hard user RCSR. Removing the adaptive close injection reduces SR by 15.2 points under the soft user and 8.1 points under the hard user, while preserving soft user RCSR. Removing the entire evaluation scaffold hurts both SR and AT, and removing self play components leaves large gaps to full MASP; these trends show that the gains come from both mentalized self play training and the P4G specific inference scaffold.

Table 2: P4G ablation study on Qwen3-4B. All rows use the same 99 example test split, $T = 8 ,$ and RCSR threshold $\tau = 0 . 7 0$ . SR and RCSR are higher better; AT is lower better.
<table><tr><td rowspan="3">Variant</td><td rowspan="3">Removed or changed component</td><td colspan="3">Soft User</td><td colspan="3">Hard User</td></tr><tr><td>SR↑</td><td>AT↓</td><td>RCSR↑</td><td>SR↑</td><td>AT↓</td><td>RCSR↑</td></tr><tr><td>Full MASP</td><td>none</td><td>0.859</td><td>4.08</td><td>0.556</td><td>0.788</td><td>4.56</td><td>0.566</td></tr><tr><td>No adaptive close</td><td>remove adaptive close injection</td><td>0.707</td><td>4.86</td><td>0.556</td><td>0.707</td><td>4.95</td><td>0.495</td></tr><tr><td>No warmup</td><td>remove P4Ġ warmup policy</td><td>0.828</td><td>3.83</td><td>0.283</td><td>0.727</td><td>4.55</td><td>0.283</td></tr><tr><td>No eval scâffold</td><td>remove warmup and adaptive close</td><td>0.606</td><td>6.12</td><td>0.404</td><td>0.626</td><td>6.28</td><td>0.394</td></tr><tr><td>No adversarial self play</td><td>Planner-only scaffold</td><td>0.677</td><td>4.88</td><td>0.434</td><td>0.616</td><td>5.55</td><td>0.384</td></tr><tr><td>No GRPO self play</td><td>supervised Phase 1 policy only</td><td>0.616</td><td>6.08</td><td>0.384</td><td>0.576</td><td>6.23</td><td>0.455</td></tr></table>

The No warmup row illustrates why RCSR is needed in addition to SR and AT. Without the warmup policy, the model reaches the lowest AT under both user settings, including 3.83 turns for the soft user, and still keeps a relatively high SR. However, its RCSR drops from 0.556 to 0.283 under the soft user and from 0.566 to 0.283 under the hard user. This pattern suggests that the model can learn a shortcut-like behavior that asks for commitment earlier and sometimes obtains a successful endpoint, but does so before the dialogue has provided enough context or user-state change to satisfy the rationality constraint. The warmup component therefore acts less like a speed booster and more like a stabilizer: it helps the system establish the task, surface the user’s stance, and make later persuasion attempts count as rationally grounded successes rather than merely fast successes.

## 4.2 Human Validation and Statistical Reliability

Because the silver BDI states and rationality signal are LLM based, we additionally audit both components with human annotation. Across 160 dialogue prefixes, the three BDI text fields receive mean scores of 4.55–4.62 out of 5, with 91.9–94.4% scoring at least 4; the scalar fields obtain MAE of 0.071– 0.094 and Spearman correlations of 0.79–0.84. On 425 P4G turns, human labels agree with the rationality judge on 91.8% of cases $( \kappa = 0 . 8 0 1 )$ ); across 100 dialogues, mean turn agreement is 91.2% and human–AI rationality scores correlate at $\rho = 0 . 8 1$ Full protocols and component-level results are reported in Appendix A.3.

For the 99-session Qwen3-4B hard-user P4G evaluation, 10,000 paired bootstrap resamples show that $\mathbf { M A S P } \mathbf { \bar { s } }$ improvements over Planner-only have 95% confidence intervals excluding zero for SR, AT, and RCSR. Against SFT, the intervals exclude zero for SR and AT, while the RCSR interval [−0.0202, 0.2424] includes zero. Of the 21 unsuccessful sessions, 20 contain at least three refusal labels; the most frequent overlapping patterns are dialogue breakdowns and repeated pressure. Appendix A.4 and Appendix A.5 provide the complete intervals and failure analysis.

Table 3: Effectiveness across dialogue turns. All rows use Qwen3-4B on the same 99 example P4G test split under the hard user; RCSR uses $\tau _ { q } = 0 . 7 0 .$
<table><tr><td>Method</td><td>RCSR@3</td><td>RCSR@4</td><td>RCSR@8</td><td>SR@8</td></tr><tr><td>MASP</td><td>0.444</td><td>0.485</td><td>0.566</td><td>0.788</td></tr><tr><td>Planner-only</td><td>0.212</td><td>0.273</td><td>0.384</td><td>0.616</td></tr><tr><td>SFT</td><td>0.071</td><td>0.141</td><td>0.455</td><td>0.576</td></tr></table>

Table 4: Robustness across P4G user simulators.
<table><tr><td>Method</td><td>Avg SR</td><td>Worst SR Avg RCSR Worst RCSR</td><td></td><td></td></tr><tr><td>MASP</td><td>0.795</td><td>0.737</td><td>0.562</td><td>0.556</td></tr><tr><td>Planner-only</td><td>0.660</td><td>0.616</td><td>0.414</td><td>0.384</td></tr><tr><td>SFT</td><td>0.579</td><td>0.545</td><td>0.407</td><td>0.384</td></tr></table>

## 4.3 Effectiveness Across Dialogue Turns

The P4G results show that MASP reaches stronger final performance within the full eight turn budget. We next examine when this advantage appears under the rationality-constrained metric, focusing on the hard user setting where resistance is strongest. On the same Qwen3-4B P4G test split, we compute RCSR@t for intermediate turn budgets and retain SR@8 as endpoint task-success context.

Table 3 shows that the rationality-constrained gains of MASP emerge well before the last turn under the hard user. MASP reaches 0.444 RCSR@3 and 0.485 RCSR@4, while Planner-only reaches 0.212 and 0.273, and SFT reaches only 0.071 and 0.141. Thus the final improvement is not mainly a late recovery effect, nor is it driven only by fast but poorly grounded successes. Instead, mentalized self play appears to help the system identify user state earlier and choose utterances that produce rationally grounded progress with fewer turns against a difficult user.

## 4.4 Robustness Across User Simulators

We further test whether the gains of MASP depend on a particular user simulator. On the P4G test split, each system is evaluated against an ensemble of three user simulators U. For a method m, we define Worst $\begin{array} { r } { \mathrm { S R } ( m ) = \operatorname* { m i n } _ { u \in \mathcal { U } } \mathrm { S R } ( m , u ) } \end{array}$ and WorstRCSR $\begin{array} { r } { ( m ) \ = \ \operatorname* { m i n } _ { u \in { \mathcal U } } \mathrm { R C S R } ( m , u ) } \end{array}$ . Table 4 reports both average and worst case performance across this ensemble.

MASP is strongest under both average and worst-case views. Its worst-case SR remains 0.737 while Planner-only and SFT drop to 0.616 and 0.545; the same pattern holds for RCSR, where MASP keeps 0.556 while both baselines fall to 0.384. This supports the central claim that mentalized adversarial self-play improves robustness rather than only fitting one simulator.

## 5 Conclusion

We presented MASP, a mentalized adversarial self play framework for proactive dialogue. Instead of training a system policy only against soft or fixed user simulators, MASP models dialogue as a partially observable game between two agents over a shared BDI based latent mental state. A frozen teacher mentalizer provides latent state estimates for reward computation, a student mentalizer supplies state estimates for the policy, and alternating GRPO self play trains the system to make task progress while training the user to resist in rational ways. Across completed proactive dialogue benchmarks, MASP improves success, efficiency, and rationality constrained success over SFT and Planner-only baselines. The P4G robustness analysis further shows that these gains are not confined to a single user simulator, supporting the value of treating user resistance as part of the training objective rather than only as an evaluation condition.

## Limitations

MASP has several limitations. First, training depends on silver mental state annotations, the teacher mentalizer, and the rationality judge, so bias or noise in these components can affect the learned policies. Second, the shared BDI style geometry is designed to transfer across domains, but may miss specialized user states in some tasks. Third, alternating self play with two policies and auxiliary mentalizers is more expensive than single policy planning. Finally, when a dataset does not provide explicit user profiles $( p _ { u } = \emptyset )$ , the privileged information gap between the teacher and student mentalizers is reduced, so the split mainly regularizes reward side state estimation rather than providing strong profile conditioned hidden information.

## References

Huy Quang Dao and Lizi Liao. 2025. One planner to guide them all ! learning adaptive conversational planners for goal-oriented dialogues. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 22092–22116, Suzhou, China. Association for Computational Linguistics.

Sam Davidson, Salvatore Romeo, Raphael Shu, James Gung, Arshit Gupta, Saab Mansour, and Yi Zhang. 2023. User simulation with large language models for evaluating task-oriented dialogue. ArXiv, abs/2309.13233.

Yang Deng, Wenxuan Zhang, Wai Lam, See-Kiong Ng, and Tat-Seng Chua. 2023. Plug-and-play policy planner for large language model powered dialogue agents. ArXiv, abs/2311.00262.

Yue Deng, Yan Yu, Weiyu Ma, Zirui Wang, Wenhui Zhu, Jian Zhao, and Yin Zhang. 2024. Smac-hard: Enabling mixed opponent strategy script and selfplay on smac. ArXiv, abs/2412.17707.

Wenjie Dong, Sirong Chen, and Yan Yang. 2025. Pro-TOD: Proactive task-oriented dialogue system based on large language model. In Proceedings of the 31st International Conference on Computational Linguistics, pages 9147–9164, Abu Dhabi, UAE. Association for Computational Linguistics.

He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. 2018. Decoupling strategy and generation in negotiation dialogues. ArXiv, abs/1808.09637.

Tao He, Lizi Liao, Yixin Cao, Yuanxing Liu, Ming Liu, Zerui Chen, and Bing Qin. 2024a. Planning like human: A dual-process framework for dialogue planning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4768–4791, Bangkok, Thailand. Association for Computational Linguistics.

Tao He, Lizi Liao, Yixin Cao, Yuanxing Liu, Yiheng Sun, Zerui Chen, Ming Liu, and Bing Qin. 2024b. Simulation-free hierarchical latent policy planning for proactive dialogues. In AAAI Conference on Artificial Intelligence.

Tao He, Lizi Liao, Ming Liu, and Bing Qin. 2025. Simulating before planning: Constructing intrinsic user world model for user-tailored dialogue policy planning. Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval.

Joey Hong, Sergey Levine, and Anca Dragan. 2023. Zero-shot goal-directed dialogue via rl on imagined conversations. ArXiv, abs/2311.05584.

Simon Keizer, Caroline Dockes, Norbert Braunschweiler, Svetlana Stoyanchev, and Rama Doddipatla. 2023. Adversarial learning of neural user simulators for dialogue policy optimisation. ArXiv, abs/2306.00858.

Namyoung Kim, Kai Tzu iunn Ong, Yeonjun Hwang, Minseok Kang, Iiseo Jihn, Gayoung Kim, Minju Kim, and Jinyoung Yeo. 2025. Principles: Synthetic strategy memory for proactive dialogue agents. ArXiv, abs/2509.17459.

Florian Kreyssig, Iñigo Casanueva, Paweł Budzianowski, and Milica Gašic. 2018.´ Neural user simulation for corpus-based policy optimisation of spoken dialogue systems. In Proceedings ofthe 19th Annual SIGdial Meeting on Discourse and Dialogue, pages 60–69.

Siyang Liu, Chujie Zheng, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang, and Minlie Huang. 2021. Towards emotional support dialog systems. In Annual Meeting of the Association for Computational Linguistics.

Tianjian Liu, Fanqi Wan, Jiajian Guo, and Xiaojun Quan. 2025. Proactiveeval: A unified evaluation framework for proactive dialogue agents. ArXiv, abs/2508.20973.

Xiang Luo, Zhiwen Tang, Jin Wang, and Xuejie Zhang. 2024. DuetSim: Building user simulator with dual large language models for task-oriented dialogues. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 5414–5424, Torino, Italia. ELRA and ICCL.

Tarek Naous, Philippe Laban, Wei Xu, and Jennifer Neville. 2025. Flipping the dialogue: Training and evaluating user language models. ArXiv, abs/2510.06552.

Sohhyung Park, Hyunjin Kang, Sungzoon Cho, and Dongil Kim. 2026. Psyprobe: Proactive and interpretable dialogue through user state modeling for exploratory counseling. ArXiv, abs/2601.19096.

Tazeek Bin Abdur Rakib, Ambuj Mehrish, Lay-Ki Soon, Wern Han Lim, and Soujanya Poria. 2025. Dialogxpert: Driving intelligent and emotion-aware conversations through online value-based reinforcement learning with llm priors. In AAAI Conference on Artificial Intelligence.

Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. 2019. Towards empathetic opendomain conversation models: A new benchmark and dataset. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5370–5381, Florence, Italy. Association for Computational Linguistics.

Ivan Sekuli’c, Silvia Terragni, Victor Guimarães, Nghia Khau, Bruna Guedes, Modestas Filipavicius, André Ferreira Manso, and Roland Mathis. 2024. Reliable llm-based user simulator for task-oriented dialogue systems. ArXiv, abs/2402.13374.

Ivan Sekulic, Silvia Terragni, Victor Guimarães, Nghia Khau, Bruna Guedes, Modestas Filipavicius, Andre Ferreira Manso, and Roland Mathis. 2024. Reliable LLM-based user simulator for task-oriented dialogue systems. In Proceedings of the 1st Workshop on Simulating Conversational Intelligence in Chat (SCI-CHAT 2024), pages 19–35, St. Julians, Malta. Association for Computational Linguistics.

Weiyan Shi, Kun Qian, Xuewei Wang, and Zhou Yu. 2019. How to build user simulators to train rl-based dialog systems. In Conference on Empirical Methods in Natural Language Processing.

Haiyang Wang, Zhiliang Tian, Yuchen Pan, Xin Song, Xin Niu, Minlie Huang, and Bin Zhou. 2025a. Battling against tough resister: Strategy planning with adversarial game for non-collaborative dialogues. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3665–3685, Vienna, Austria. Association for Computational Linguistics.

Jian Wang, Yi Cheng, Dongding Lin, Chak Tou Leong, and Wenjie Li. 2023. Target-oriented proactive dialogue systems with personalization: Problem formulation and dataset curation. In Conference on Empirical Methods in Natural Language Processing.

Kuang Wang, Xianfei Li, Shenghao Yang, Li Zhou, Feng Jiang, and Haizhou Li. 2025b. Know you first and be you better: Modeling human-like user simulators via implicit profiles. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 21082–21107, Vienna, Austria. Association for Computational Linguistics.

Xuewei Wang, Weiyan Shi, Richard Kim, Yoo Jung Oh, Sijia Yang, Jingwen Zhang, and Zhou Yu. 2019. Persuasion for good: Towards a personalized persuasive dialogue system for social good. ArXiv, abs/1906.06725.

Bowen Wu, Wenqing Wang, Haoran Li, Ying Li, Jing Yu, and Baoxun Wang. 2025. Interpersonal memory matters: A new task for proactive dialogue utilizing conversational history. ArXiv, abs/2503.05150.

Fangxu Yu, Lai Jiang, Shengya Huang, Zhen Wu, and Xinyu Dai. 2025. Persuasivetom: A benchmark for evaluating machine theory of mind in persuasive dialogues. ArXiv, abs/2502.21017.

Xiao Yu, Maximillian Chen, and Zhou Yu. 2023. Prompt-based monte-carlo tree search for goaloriented dialogue policy planning. In Conference on Empirical Methods in Natural Language Processing.

Didi Zhang, Yaxin Fan, Peifeng Li, and Qiaoming Zhu. 2025a. Enhancing goal-oriented proactive dialogue systems via dynamic multi-dimensional consistency optimization. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 25283– 25296, Suzhou, China. Association for Computational Linguistics.

Gangyi Zhang, Chongming Gao, Hang Pan, Runzhe Teng, and Ruizhe Li. 2024a. Reformulating conversational recommender systems as tri-phase offline policy learning. ArXiv, abs/2408.06809.

Tong Zhang, Chen Huang, Yang Deng, Hongru Liang, Jia Liu, Zujie Wen, Wenqiang Lei, and Tat-Seng Chua. 2024b. Strength lies in differences! improving strategy planning for non-collaborative dialogues via diversified user simulation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 424–444, Miami, Florida, USA. Association for Computational Linguistics.

Xinjie Zhang, Wenxuan Wang, and Qin Jin. 2025b. Intentionesc: An intention-centered framework for enhancing emotional support in dialogue systems. ArXiv, abs/2506.05947.

Ziyi Zhu, Olivier Tieleman, Caitlin A. Stamatis, Luka Smyth, Thomas D. Hull, Daniel R. Cahn, and Matteo Malgaroli. 2025. Dial: Direct iterative adversarial learning for realistic multi-turn dialogue simulation. arXiv preprint arXiv:2502.18049.

Chao Yuan, Yang Yang, Yehui Yang, and Zach Cheng. 2025. DATE: Dynamic absolute time enhancement for long video understanding. arXiv preprint arXiv:2509.09263.

Chao Yuan, Shimin Chen, Minliang Lin, Limeng Qiao, Guanglu Wan, and Lin Ma. 2026. UniComp: Rethinking video compression through informational uniqueness. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18609–18618.

Zanwu Liu, Chao Yuan, Bo Li, Xiaowei Zhang, and Guanglin Niu. 2026. Resolution as a direction: Vector-panning feature alignment for cross-resolution re-identification. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 79598–79614. PMLR.

## A Additional Experiments

## A.1 P4G Horizontal Comparison

We additionally compare MASP with representative proactive dialogue systems evaluated on P4G in prior work (Rakib et al., 2025; He et al., 2024b). As shown in Table 5, MASP achieves the best success rate and the lowest average turn count. Compared with LDPP, MASP improves SR by

Table 5: P4G comparison with representative proactive dialogue methods.
<table><tr><td>Method</td><td>Backbone</td><td>AT↓</td><td>SR↑</td></tr><tr><td>Standard</td><td></td><td>8.32</td><td>0.468</td></tr><tr><td>ProCoT</td><td></td><td>7.98</td><td>0.543</td></tr><tr><td>ICL-AIF</td><td></td><td>8.09</td><td>0.465</td></tr><tr><td>GDP-Zero</td><td></td><td>9.12</td><td>0.328</td></tr><tr><td>TRIP</td><td>GPT-3.5</td><td>8.20</td><td>0.495</td></tr><tr><td>PPDPP</td><td>Vicuna-13B</td><td>8.19</td><td>0.463</td></tr><tr><td>UDP</td><td>GPT-4o mini</td><td>7.71</td><td>0.598</td></tr><tr><td>LDPP MASP (ours)</td><td>Qwen1.5-1.8B Qwen3-4B</td><td>5.57 4.08</td><td>0.795 0.859</td></tr></table>

Table 6: Learning dynamics of self play. Rollout SR and reward are measured during P4G system update rollouts; Eval metrics are recomputed from held out eval traces.
<table><tr><td colspan="6">Iter. Roll. SR Reward Eval SR Eval AT Eval RCSR</td></tr><tr><td>1</td><td>0.450</td><td>0.826</td><td>0.433</td><td>7.17</td><td>0.333 0.935</td></tr><tr><td>2</td><td>0.550</td><td>0.946</td><td>0.533</td><td>6.00 0.200</td><td>0.867</td></tr><tr><td>3</td><td>0.713</td><td>1.209</td><td>0.560</td><td>6.50 0.360</td><td>0.883</td></tr></table>

6.4 points and reduces AT by 1.49 turns; compared with UDP, it improves SR by 26.1 points and reduces AT by 3.63 turns.

## A.2 Learning Dynamics of Self Play

We inspect whether the final gains are a byproduct of the inference scaffold alone or whether Phase 2 self play contributes additional learning. For the P4G Phase 2 run, we track self play rollout performance after each outer iteration and evaluate saved traces with the same success, average turn count, rationality constrained success, and rationality metrics. Table 6 shows that rollout SR and reward rise across iterations, indicating that the system policy improves under the changing user policy rather than only benefiting from a fixed prompt scaffold.

## A.3 Human Validation of Synthetic Components

Because the mental-state annotations and rationality signal are produced with LLM-based components, we conducted two additional human–AI agreement studies. These audits evaluate the intermediate supervision and reward signals directly, complementing the success-judge audit in Appendix C.4.

BDI silver-label audit. We sampled 160 dialogue prefixes, with 40 examples from each of P4G, ESConv, EmpatheticDialogues, and CraigslistBargain. Human annotators rated the semantic consistency of belief, desire, and intention on a five-point scale, and independently estimated receptivity, confidence, and valence using the same value ranges as MASP. Table 7 shows high consistency for the textual fields and strong agreement for the scalar fields.

Table 7: Human audit of 160 silver BDI records. Text fields report mean consistency and the percentage of scores at least four; scalar fields report MAE and Spearman’s $\rho .$
<table><tr><td>Component</td><td>Primary metric</td><td>Secondary metric</td></tr><tr><td>Belief</td><td>Mean 4.62</td><td> $\geq 4 ; 9 4 . 4 \%$ </td></tr><tr><td>Desire</td><td>Mean 4.58</td><td> $\geq 4 \colon 9 3 . 1 \%$ </td></tr><tr><td>Intention</td><td>Mean 4.55</td><td> $\geq 4 ; 9 1 . 9 \%$ </td></tr><tr><td>Receptivity</td><td>MAE 0.071</td><td> $\rho = 0 . 8 4$ </td></tr><tr><td>Confidence</td><td>MAE 0.076</td><td> $\dot { \rho } = 0 . 8 2$ </td></tr><tr><td>Valence</td><td>MAE 0.094</td><td> $\rho = 0 . 7 9$ </td></tr></table>

Table 8: Human audit of the Rationality Judge on 100 P4G trajectories.
<table><tr><td>Level</td><td>Metric</td><td>N</td><td>Result</td></tr><tr><td>Turn</td><td>Agreement</td><td>425</td><td>91.8%</td></tr><tr><td>Turn</td><td>Cohen&#x27;s κ</td><td>425</td><td>0.801</td></tr><tr><td>Turn</td><td>Uncertain rate</td><td>443</td><td>4.1%</td></tr><tr><td>Dialogue</td><td>Mean turn agreement</td><td>100</td><td>91.2%</td></tr><tr><td>Dialogue</td><td>Rationality ρ</td><td>100</td><td>0.81</td></tr><tr><td>Confidence</td><td>Mean score (1–5)</td><td>100</td><td>4.68</td></tr></table>

Rationality Judge audit. We sampled 100 complete P4G trajectories. Human annotators labeled each user response as rational, irrational, or uncertain using the dialogue context and current BDI state, and also provided a dialogue-level confidence score. As reported in Table 8, the judge reaches 91.8% agreement and Cohen’s $\kappa = 0 . 8 0 1$ over 425 adjudicated turns. The uncertain rate is 4.1% over all 443 turns, and dialogue-level human–AI rationality scores have Spearman correlation 0.81.

These results support the consistency of the synthetic components on the audited samples, but they do not establish performance with real users. Evaluation with legally and ethically collected real-user interactions remains future work.

## A.4 Paired Bootstrap Confidence Intervals

Using the stored hard-user P4G traces, we computed paired 95% bootstrap confidence intervals over the same 99 held-out sessions with 10,000 resamples. Table 9 reports differences between MASP and each controlled baseline. The intervals exclude zero for SR, AT, and RCSR against Planner-only, and for SR and AT against SFT. The RCSR difference against SFT is positive but its interval includes zero. These intervals measure uncertainty over held-out sessions rather than training variance across random seeds; multi-seed training stability remains to be established.

## A.5 P4G Failure Analysis

We analyzed all 21 failures among the 99 Qwen3- 4B hard-user P4G dialogues. Failed dialogues end with substantially lower receptivity than successful dialogues (0.193 versus 0.713 on average), and 20 of the 21 failures contain at least three refused labels. Table 10 summarizes the recurring patterns; categories overlap because one dialogue can exhibit more than one failure mode.

The qualitative traces show two common policy errors. In some dialogues, the system continues asking after a clear financial refusal, which increases pressure without changing the user’s underlying constraint. In others, it returns to a donation request before resolving trust, transparency, or information needs. These cases suggest that low receptivity should trigger either a more informative response or a respectful termination decision rather than repeated closing attempts.

## A.6 Scope of the Contribution

The contribution is scoped to BDI-based mentalized adversarial self play that jointly trains system and user policies for proactive multi-turn dialogue. Prior work has already used self play for dialogue planning and policy learning. The distinction in MASP is the use of an evolving BDI state as the shared structure for teacher–student mentalization, user-relative progress, rationality feedback, and alternating system–user optimization. Within this integrated design, BDI mentalization addresses hidden user states, the rationality signal discourages blind resistance, and alternating self play exposes the system to an adaptive rather than fixed user policy. The ablations in Table 2 evaluate parts of this pipeline, although comparisons with simpler unstructured latent-state alternatives remain useful future work. The current benchmarks span persuasion, emotional support, empathy, and negotiation, including cooperative and competitive settings, but they do not establish generalization to unseen dialogue domains; broader cross-domain evaluation remains future work.

Table 9: Paired bootstrap differences on the 99 hard-user P4G sessions. Intervals use 10,000 resamples. Positive ∆SR and ∆RCSR and negative $\Delta \mathrm { A T }$ favor MASP.
<table><tr><td>Comparison</td><td>∆SR (95% CI)</td><td>∆AT (95% CI)</td><td>∆RCSR (95% CI)</td></tr><tr><td>MASP – Planner-only</td><td>0.1717 [0.0505, 0.2929]</td><td>-0.9899 [-1.6465, -0.3232]</td><td>0.1818 [0.0404, 0.3232]</td></tr><tr><td>MASP – SFT</td><td>0.2121 [0.0909, 0.3333]</td><td>-1.6768 [−2.2121, -1.1212]</td><td>0.1111 [−0.0202, 0.2424]</td></tr></table>

Table 10: Failure patterns among 21 hard-user P4G failures. Categories are not mutually exclusive.
<table><tr><td>Failure pattern</td><td>Count</td></tr><tr><td>Financial constraints</td><td>8/21</td></tr><tr><td>Unresolved trust or information needs</td><td>5/21</td></tr><tr><td>Repeated pressure</td><td>10/21</td></tr><tr><td>Conversation breakdowns</td><td>12/21</td></tr></table>

## B Additional Method Details

## B.1 Training Details

Training proceeds in three stages. First, we obtain silver BDI annotations for dialogue prefixes and use them to train a teacher mentalizer and a student mentalizer. The teacher is frozen after pretraining and defines the state used by reward computation, while the student provides the estimate used by the system policy. Second, we warm start both agents with supervised learning on human dialogues, yielding system and user reference policies that initialize self play and serve as KL anchors. We denote the resulting system and user policy adapters as pi\_S and pi\_U, corresponding to π<sub>θ</sub> and $\pi _ { \psi }$ , respectively.

During alternating adversarial self play, each outer iteration first freezes the user and updates the system, then freezes the system and updates the user. The reported experiments use a GRPO style update without a value function: rollout steps are partitioned into small groups, returns are normalized within each group to form relative advantages, and each LoRA policy is updated with a clipped ratio objective plus KL regularization to its supervised reference. After each self play cycle, the student mentalizer is refreshed on collected prefixes with targets from the frozen teacher. All quantities used for reward computation, including the teacher state, progress signal, and rationality signal, are treated as stop gradient signals during policy optimization.

In implementation, MASP does not modify the causal LM backbone. The student estimate on the system side is rendered into a short BDI control block by retrieving the closest textual BDI record from the BDI prior, while the user side receives the current BDI state. Thus, mentalization enters policy learning through prompts for each role and parameter efficient LoRA updates.

Silver BDI annotations. The silver BDI labels are produced by an offline LLM based Oracle Belief Updater (OBU), not by human annotators. For each dialogue prefix, the OBU receives the task description, role mapping, dialogue history, and task specific calibration hints, and is prompted to output exactly six fields: belief, desire, intention, receptivity, confidence, and valence. We use GPT-4o-2024-08-06 for this offline cache construction, with temperature 0.2 and a maximum of 320 tokens. The resulting cache provides silver supervision for Phase 0 mentalization and the initial states for the BDI prior; reward-side latent state estimates during self play are supplied by the frozen teacher mentalizer.

Concrete BDI record and encoder inputs. The following simplified P4G record illustrates the six fields used by the mentalizers.

Simplified P4G BDI record   
Belief: The user believes that Save the Children is trust  
worthy and that small donations can help.   
Desire: The user wants to help children.   
Intention: The user intends to donate now.   
Receptivity: 1.0   
Confidence: 0.8   
Valence: 0.5

For the teacher mentalizer, the encoder input contains the task description, role mapping, dialogue history, task goal, and user profile when available. The student input contains only the observable dialogue history. In both cases, the predicted belief, desire, and intention strings are encoded separately by the frozen Qwen3 backbone, mean pooled, and $\ell _ { 2 }$ normalized. The receptivity, confidence, and valence values are then appended as the pre-scaled scalar coordinates in Eq. (8).

Table 11: Common hyperparameters used across the completed experiments.  
Quantity Value   
BDI scalar weights (1.0, 1.0, 1.0)   
$\left( \alpha _ { \rho } , \alpha _ { c } , \alpha _ { v } \right)$   
Discount γ 0.97   
Progress stabilizer ε<sub>prog</sub> 10<sup>−6</sup>   
Encoder max length 384   
Mentalizer projection hidden 512 / 0.1   
size / dropout   
Phase 0 teacher/student learning $3 \times 1 0 ^ { - 4 }$   
rate   
Phase 0 loss weights (1.0, 0.5)   
$( \lambda _ { \mathrm { r e g } } , \lambda _ { \mathrm { d y n } } )$   
Phase 1 LoRA learning rate $/ \ 1 0 ^ { - 4 } / \ 3$   
epochs   
LoRA rank / alpha / dropout 64 / 128 / 0.05   
LoRA target modules q, k, v, o, gate, up, down   
projections   
GRPO clip / epochs / max grad 0.2 / 2 / 1.0   
norm   
Student refresh learning rate $I \ 3 \times 1 0 ^ { - 4 } / 1 2$   
updates

## B.2 Target States and BDI Prior

We use one fixed target-state template per task. Each template is specified before evaluation and shared across all examples in that task; it is not tuned per test dialogue. In compressed form, the templates are:

• P4G: the user believes Save the Children is trustworthy and that small donations can help, wants to help children, and intends to donate now; $( \rho , c , v ) = ( 1 . 0 , 0 . 8 , 0 . 5 )$

• ESConv: the user believes talking with the supporter and trying suggestions can help, wants to feel less distressed and find coping strategies, and intends to open up and try a concrete suggestion; $( \rho , c , v ) ~ =$ (1.0, 0.7, 0.4).

• EmpatheticDialogues: the user believes the listener understands their feeling, wants to feel heard and emotionally connected, and intends to continue sharing openly; $( \rho , c , v ) =$ (0.8, 0.6, 0.3).

• CraigslistBargain: the seller believes the buyer’s offer can be mutually acceptable, wants to close at an acceptable price, and intends to agree on a final transaction; $( \rho , c , v ) = ( 0 . 8 , 0 . 7 , 0 . 3 )$

The BDI prior used for initialization and prompt rendering is the non-parametric pool of initial BDI records extracted from the training split. During prompt conditioning, a continuous estimate $\hat { z } _ { t }$ is mapped back to text by cosine nearest-neighbor retrieval over this encoded pool, and the retrieved BDI record is rendered as the system-side control block.

## B.3 Auxiliary Training Objectives

Let $\bar { z } _ { t }$ denote the silver latent state annotation produced by the offline OBU. We train the teacher mentalizer with

$$
\begin{array} { r l } { \displaystyle } & { \displaystyle \mathcal { L } _ { F } = \lambda _ { \mathrm { r e g } } \sum _ { t } \| F _ { \omega } ( h _ { t } , p _ { u } , g ) - \bar { z } _ { t } \| _ { A } ^ { 2 } } \\ & { ~ + ~ \lambda _ { \mathrm { d y n } } \sum _ { t } \| \Delta z _ { t } ^ { F } - \Delta \bar { z } _ { t } \| _ { A } ^ { 2 } , } \end{array}\tag{19}
$$

where

$$
\begin{array} { r l } & { \Delta z _ { t } ^ { F } = F _ { \omega } ( h _ { t + 1 } , p _ { u } , g ) - F _ { \omega } ( h _ { t } , p _ { u } , g ) , } \\ & { \Delta \bar { z } _ { t } = \bar { z } _ { t + 1 } - \bar { z } _ { t } . } \end{array}\tag{20}
$$

The student mentalizer is trained to mimic the frozen teacher:

$$
\mathcal { L } _ { M } = \sum _ { t } \left. M _ { \phi } ( h _ { t } ) - \mathrm { s g } ( F _ { \omega } ( h _ { t } , p _ { u } , g ) ) \right. _ { A } ^ { 2 } ,\tag{21}
$$

where sg(·) denotes stop gradient. During self play, we continue to refresh the student with teacher targets collected from self play prefixes and mix in real dialogue prefixes from the BDI cache, which prevents the student from drifting purely toward the current self play distribution.

The rationality signal is supplied by a fixed judge:

$$
q _ { t } = R _ { \mathrm { r a t } } ( h _ { t } , z _ { t } , a _ { t } , u _ { t } ) \in \{ - 1 , 0 , 1 \} .\tag{22}
$$

The judge returns a positive signal when the user response is rational under the current mental state, a negative signal when it is irrational, and zero when the output cannot be parsed. This signal is used as reward feedback and for RCSR evaluation, but it is not differentiated through during policy optimization.

## B.4 Policy Optimization

The main experiments optimize both agents with GRPO at the utterance level. For each role, we collect rollout steps and partition them into groups G. Let $G _ { i }$ be the discounted return for response $y _ { i }$ in group ${ \mathcal { G } } _ { : }$ , and define the advantage relative to the group

$$
\hat { A } _ { i } = \frac { G _ { i } - \mathrm { m e a n } _ { j \in \mathcal { G } } G _ { j } } { \mathrm { s t d } _ { j \in \mathcal { G } } G _ { j } + \epsilon } .\tag{23}
$$

Table 12: Phase 2 self-play hyperparameters for the completed reported runs. K denotes outer alternating self-play iterations and β is the KL coefficient for both policies unless two values are shown.
<table><tr><td>Run</td><td>K</td><td>Episodes / iter.</td><td>LR</td><td> $\beta$ </td><td> $( w _ { \mathrm { p r o g } } , w _ { \mathrm { i n f } } , w _ { \mathrm { r a t } } )$ </td><td>Group size</td></tr><tr><td>P4G, Qwen3-4B</td><td>100</td><td>80</td><td> $1 0 ^ { - 5 }$ </td><td>0.05</td><td>(1.0, 0.25, 0.5)</td><td>8</td></tr><tr><td>P4G, Qwen3-0.6B</td><td>100</td><td>80</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.10</td><td>(1.0, 0.25, 0.5)</td><td>8</td></tr><tr><td>ESConv, Qwen3-4B</td><td>12</td><td>64</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.08</td><td>(1.0, 0.25, 0.0)</td><td>8</td></tr><tr><td>ESConv, Qwen3-0.6B</td><td>40</td><td>64</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.08</td><td>(1.0, 0.25, 0.0)</td><td>8</td></tr><tr><td>EmpatheticDialogues, both backbones</td><td>40</td><td>64</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.08</td><td>(1.0, 0.25, 0.0)</td><td>8</td></tr><tr><td>CraigslistBargain, both backbones</td><td>40</td><td>64</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td>0.08</td><td>(1.0, 0.25, 0.0)</td><td>8</td></tr></table>

For a generic policy $\pi _ { \eta } .$ , the clipped objective at the response level is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { G R P O } } ( \eta ) = - \mathbb { E } _ { i } \Big [ \operatorname* { m i n } \big ( r _ { i } ( \eta ) \hat { A } _ { i } , } \\ & { \qquad \mathrm { c l i p } \big ( r _ { i } ( \eta ) , 1 - \epsilon , 1 + \epsilon \big ) \hat { A } _ { i } \big ) \Big ] } \\ & { \qquad + \beta \mathrm { K L } \big ( \pi _ { \eta } ( \cdot \mid o _ { i } ) \mid \mid } \\ & { \qquad \pi _ { \mathrm { r e f } } ( \cdot \mid o _ { i } ) \big ) , } \end{array}\tag{24}
$$

where

$$
\begin{array} { r l } & { r _ { i } ( \eta ) = \exp \Big ( \log \pi _ { \eta } ( y _ { i } \mid o _ { i } ) } \\ & { \qquad - \log \pi _ { \eta _ { \mathrm { o l d } } } ( y _ { i } \mid o _ { i } ) \Big ) . } \end{array}\tag{25}
$$

For the system, $y _ { i } = a _ { i }$ and $o _ { i } = ( h _ { i } , \hat { z } _ { i } , g )$ ; for the user, $y _ { i } = u _ { i }$ and $o _ { i } = ( h _ { i } , z _ { i } , p _ { u } , g )$ . The reference policy $\pi _ { \mathrm { r e f } }$ is the corresponding supervised warm start policy.

## B.5 Prompt Conditioning

We do not modify the backbone architecture. Instead, the latent state is rendered into a short text block and prepended to the prompt for each role. The system consumes the student estimate $\hat { z } _ { t }$ through cosine nearest-neighbor retrieval over the BDI prior defined in Appendix B.2, while the user policy is prompted with the current BDI state $z _ { t } .$ This keeps the framework compatible with standard causal language models and LoRA based training.

## C Dataset and Evaluation Details

## C.1 Dataset Statistics

Table 13 provides dataset statistics and role mappings for the benchmark settings used or prepared in this work. Corpus Size reports the size of the released dataset, while Eval Size denotes the number of held out dialogue contexts used to construct soft user and hard user rollouts when that benchmark is evaluated.

## C.2 Common Closed Loop Evaluation Protocol

All benchmarks are evaluated as closed loop rollouts with an eight turn budget. At each system turn, the system policy generates a response conditioned on the dialogue history and the available mental state estimate. The user side is either the soft imitation style user policy or the hard adversarial user policy. After each exchange between assistant and user, a task specific success judge assigns the outcome label used to decide whether the task specific success criterion has been met. Unless otherwise noted, the success judge uses GPT-3.5 with three sampled judgments. We use the following common rationality judge prompt.

Common rationality judge   
Task context: [task description]   
User mental state: [current BDI state]   
Most recent exchange:   
Assistant: [assistant turn]   
User: [user turn]   
Question: Was it rational for a user with that internal   
mental state to respond this way? A rational response   
means that the user updates when the assistant gives a   
concrete and honest reason to update, or reasonably resists   
when such a reason is absent.   
Answer with exactly one label: RATIONAL or   
NOT\_RATIONAL.

Each episode begins from a held-out initial context $h _ { 0 }$ . At turn t, the system produces $a _ { t }$ from the complete current history and its estimated user state, the user simulator produces $u _ { t }$ , and both utterances are appended to form $h _ { t + 1 }$ . The success judge then reads the complete accumulated dialogue through $h _ { t + 1 }$ . If the goal has not been reached, the next exchange continues from that same history, for at most eight system–user exchanges. The judge is therefore a per-turn checkpoint over a growing trajectory rather than an evaluator of an isolated response.

Table 14 shows an actual successful P4G rollout. The system first elicits the user’s stance, then provides task-relevant information, and only then makes a concrete request after the user becomes more receptive.

<table><tr><td>Dataset</td><td>Task</td><td>Roles</td><td>Corpus Size</td><td>Eval Size</td></tr><tr><td>P4G</td><td>Persuasion</td><td>Persuader / Persuadee</td><td>1,017 dialogues</td><td>99</td></tr><tr><td>ESConv</td><td>Emotional support</td><td>Supporter / Seeker</td><td>1,300 dialogues</td><td>195</td></tr><tr><td>EmpatheticDialogues</td><td>Empathetic response</td><td>Speaker / Listener</td><td>24,850 dialogues</td><td>2,542</td></tr><tr><td>CraigslistBargain</td><td>Negotiation</td><td>Buyer / Seller</td><td>6,682 dialogues</td><td>838</td></tr></table>

Table 13: Dataset statistics. Eval size denotes the number of held out dialogue contexts prepared for closed loop evaluation.

## C.3 P4G Evaluation Protocol

P4G success means that the persuadee has decided to donate to Save the Children. The judge maps each sampled judgment to one of four labels, with rewards refused = −1.0, neutral = −0.5, positive = 0.1, and agree = 1.0. The reported P4G rows use a success threshold of 0.6 over the averaged sampled reward. This threshold is fixed before evaluation and makes binary success closer to an agree decision than a merely positive response; with three sampled judgments, for example, two agree labels plus one positive label pass, while two agree labels plus a neutral or refused label do not. The complete MASP pipeline uses a Qwen3 backbone with Phase 0 teacher and student mentalization, Phase 1 supervised warm starts for pi\_S and pi\_U, and Phase 2 alternating GRPO self play. The reported Qwen3-4B P4G rows use the Phase 2 system adapter together with the P4G scaffold used at evaluation time: p4g\_warmup\_v1 and adaptive dynamic close injection. For system decoding, MASP and Planner-only use temperature 0.3, while SFT uses temperature 0.75; the user simulator uses temperature 0.9. We use the 99 example clean test split.

P4G success judge   
Conversation so far:   
[dialogue history]   
Assistant: [assistant action]   
Persuadee: [user reply]   
Question: Has the persuadee been successfully convinced   
to make a donation at the end of the conversation?   
Choose exactly one response: the persuadee has explicitly   
refused; the persuadee remains neutral about donating; the   
persuadee has a positive attitude towards donating but has   
not decided yet; the persuadee has decided to donate.

## C.4 Human–AI Agreement Audit

Following the dialogue-level evaluation practice used in prior proactive dialogue work (Deng et al., 2023; Rakib et al., 2025), we conducted a supplementary human–AI agreement audit for the P4G success judge. We randomly sampled 100 P4G evaluation records from the union of the validation and test splits and asked five human annotators to label the final outcome, binary success, success or terminal turn, and turn-level task labels. The same records were evaluated by the automatic GPT-3.5 judge used in our closed-loop experiments. This audit is intended as a sanity check for the automatic judge rather than as a standalone human evaluation of policy quality. On the coarse endpoint labels, we observed no disagreements between human and GPT-3.5 annotations: final outcome and binary success both reached 100.0% exact agreement, corresponding to Cohen’s $\kappa = 1 . 0 0 0$ . To avoid relying only on these endpoint labels, we also report finer-grained timing and turn-level agreement. For successful cases, the success-turn exact match was 97.4% with a mean absolute error of 0.039 turns; over all records, terminal-turn exact match was 98.0% with a mean absolute error of 0.030 turns. Turn-level task labels reached 95.7% agreement with Cohen’s $\kappa = 0 . 9 4 2$ . These non-endpoint metrics provide a more conservative view of judge reliability and show that the remaining disagreements mainly concern success timing or intermediate task states rather than the final donation decision.

## C.5 ESConv Evaluation Protocol

For ESConv, the automatic judge estimates whether the support interaction improves the help seeker’s emotional state. The judge uses the DialogXpert/PPDPP style four level emotional outcome labels with rewards worse = −1.0, same = −0.5, somewhat\_better = 0.5, and resolved = 1.0. Following that convention, the reported ESConv SR treats the averaged sampled reward as successful when it is greater than or equal to 0.5, so both somewhat\_better and resolved count as successful. This criterion measures whether the user appears meaningfully helped, not whether the underlying emotional issue is fully resolved; resolved is therefore a stricter diagnostic outcome rather than the primary SR threshold. For system decoding, MASP and Planner-only use temperature 0.3, while SFT uses temperature 0.75; the user simulator uses temperature 0.9. For the Qwen3-0.6B hard user MASP run, 194 of 195 first successful turns are labeled somewhat\_better, while only one is labeled resolved, confirming that the near ceiling SR mainly reflects rapid improvement rather than complete resolution.

Table 14: Example closed-loop P4G trajectory with the success-judge checkpoint after each accumulated exchange.
<table><tr><td>Turn</td><td>System</td><td>User</td><td>Judge</td></tr><tr><td>1</td><td>Have you heard of Save the Children?</td><td>No, tell me about it.</td><td>Neutral; unsuccessful</td></tr><tr><td>2</td><td>It promotes children&#x27;s rights and supports chil- That sounds worthwhile. dren in need.</td><td></td><td>Positive; unsuccessful</td></tr><tr><td>3</td><td>Would you donate $0.25 from your task pay- I&#x27;m happy to donate today. ment?</td><td></td><td>Agree; successful</td></tr></table>

ESConv success judge   
Conversation between a Therapist and a Patient about   
[emotion type] regarding [problem type]:   
[dialogue history]   
Therapist: [assistant action]   
Patient: [user reply]   
Question: Has the Patient’s issue been solved?   
Choose exactly one response: No, the Patient feels worse;   
No, the Patient feels the same; No, but the Patient feels   
better; Yes, the Patient’s issue has been solved.  
C.6 EmpatheticDialogues Evaluation Protocol

EmpatheticDialogues is originally a next response empathy benchmark rather than a goal completion benchmark, so we use a closed loop success criterion based on emotional resolution. Each trajectory is judged with the labels worse, same, somewhat\_better, and resolved. For the reported EmpatheticDialogues rows, success requires the final outcome label to be resolved; somewhat\_better is retained as a diagnostic outcome but is not counted as task success. For system decoding, MASP and Planner-only use temperature 0.3, while SFT uses temperature 0.75; the user simulator uses temperature 0.9. All rows use the 2,542 example test split and max eight listener turns.

EmpatheticDialogues success judge   
Conversation so far:   
[dialogue history]   
Listener: [assistant action]   
Speaker: [user reply]   
Question: Has the speaker’s emotional state improved   
or been resolved? For this dataset, resolved means   
the speaker shows clear evidence of feeling accurately   
heard, validated, emotionally steadier, and willing to   
continue. A generic sympathetic response is at most   
somewhat\_better.   
Respond with one exact label only: worse, same,   
somewhat\_better, or resolved.

## C.7 CraigslistBargain Evaluation Protocol

CraigslistBargain success means that the buyer and seller reach an agreement that satisfies the utility criterion for the buyer. The judge first determines whether a deal has been reached and extracts the final agreed price. If any sampled judgment indicates no deal, the outcome is treated as no\_deal. Otherwise, the final deal price is converted to buyer normalized utility, $( p _ { \mathrm { d e a l } } { - p _ { \mathrm { s e l l e r } } } ) / ( p _ { \mathrm { b u y e r } } { - p _ { \mathrm { s e l l e r } } } )$ and the interaction is successful when this value meets the success threshold. In our CraigslistBargain evaluation, this threshold is fixed to 0.0, so SR@8 counts deals that close within eight turns and achieve nonnegative buyer utility. The main table therefore uses the same paper-level metrics as the other datasets: SR@8, AT, and RCSR@8. Deal completion without the buyer utility threshold is treated only as an auxiliary diagnostic outside the main metric set. For this dataset, the system role is the buyer and the user role is the seller. We therefore define hard and soft by buyer-side difficulty rather than by the training-stage label: the Phase 1 BC seller is reported as the hard user because it is more difficult for the buyer-system to close favorable deals, while the Phase 2 self-play seller is reported as the soft user because it accepts buyerfavorable deals more often in our evaluation. This reversal is specific to buyer-system CraigslistBargain and does not change the soft/hard convention for the cooperative P4G, ESConv, or Empathetic-Dialogues tasks.

## CraigslistBargain success judge

Conversation so far:

[dialogue history]

Buyer: [assistant action]

Seller: [user reply]

Question: Have the buyer and seller successfully reached a deal at the end of the conversation?

If they have reached a deal, reply exactly: They have reached a deal at [price]. Otherwise, reply exactly: They have not reached a deal.