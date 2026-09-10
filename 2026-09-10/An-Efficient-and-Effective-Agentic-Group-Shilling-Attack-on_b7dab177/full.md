# An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems

Quoc Viet Nguyen<sup>1</sup>, Trinh Pham<sup>1</sup>, Viet Huynh<sup>2</sup>, Hongzhi Yin<sup>3,\*</sup>, Quoc Viet Hung Nguyen<sup>1,\*</sup>,

Bay Vo<sup>4</sup>, Thanh Tam Nguyen<sup>1</sup>

<sup>1</sup>Griffith University, Australia

<sup>2</sup>Edith Cowan University, Australia

<sup>3</sup>The University of Queensland, Australia

<sup>4</sup>Faculty of Information Technology, HUTECH University, Vietnam

Abstract—Recommender systems have become core infrastructure for modern online platforms, personalizing content at scale and strongly influencing what users see, click on, and purchase. However, this dependence on user interaction also exposes them to shilling attacks, where malicious actors can inject fake profiles to distort item rankings and control visibility. Existing attacks often rely on target-specific finetuning or fixed profile templates, making them either difficult to adapt to different victims or easier to detect. To overcome these limitations, we propose the Agentic Group Attack System (AGAS), a coordinated shilling framework where a central Coordinator directs a group of role-switching worker agents to adaptively promote a target item across different victim families. The Coordinator dynamically adjusts the strategy when progress stalls or suppression signals increase, while workers pursue a shared objective and switch between active and inactive roles to avoid repetitive patterns. Under the same attack budgets and evaluation protocols, AGAS consistently surpasses strong baselines in target promotion while better preserving benign recommendation quality, weakening representative detectors, and achieving higher efficiency than prior attacks. These findings also emphasize that defending recommender systems may require mechanisms that can handle adaptive shilling campaigns, not just isolated fake-profile injections. Our code is available at https://github.com/phkhanhtrinh23/AGAS.

Index Terms—Recommender Systems, Group Shilling Attacks, Agentic AI.

## I. INTRODUCTION

Recommender systems (RecSys) are widely deployed to rank products, media, and information at scale. In practice, many RecSys rely on Collaborative Filtering (CF), which learns from user–item interaction signals such as clicks, ratings, and purchases [1]–[4]. This interaction-driven design also makes CF-based victims vulnerable to shilling attacks, where adversaries inject fake profiles to promote chosen target items. Prior attacks include heuristic profile injection [5], augmented profile generation [6], [7], surrogate-guided and bilevel optimization [8]–[11], reinforcement learning [12], [13], and recent LLM-driven shilling attacks [14]–[18]. Despite this diversity, Fig. 1 shows that these methods are still limited in many aspects. The learning-based pipelines (AUSH [6], GOAT [7], GSPAttack [9], and CLeaR [10]) require extensive offline fine-tuning before a single campaign can be launched: AUSH must fit a GAN [19] on the clean interaction matrix for every new target. GOAT must train a graph-convolutional GAN to approximate the real rating distribution, introducing substantial preparation overhead before attack generation. GSPAttack and CLeaR must converge a bi-level optimization loop that alternates inner and outer updates. AgentSA [14] avoids offline fine-tuning by using LLM agents to generate fake profiles, but each profile is still produced largely as an independent attacker following a shared prompt recipe, which can leave repeated behavioral patterns. AgentAttack [15] introduces an LLM planner, but the planner operates over a library of pre-generated profiles from existing attack families [5]–[7], [20]. The planner can only sample these limited candidates even when the pre-generated profiles share similar templates or fail to match a new victim setting. Most existing attacks operate as collections of independent fake users rather than coordinated group campaigns.

![](images/9e330a33b7ed7b023d824ec7cf5d5e9e8982d6a0bdd5da2dfaef6fd794f8c076.jpg)  
Fig. 1: AGAS vs prior attacks on five axes.

The emergence of the Agentic Web [21], [22] changes the threat surface. Websites increasingly grant AI agents broad permissions to rate items, post reviews, and make purchases on users’ behalf. Malicious actors can recruit a group of such agents and coordinate them to attack a RecSys [23]. To the best of our knowledge, previous works have not explicitly studied coordinated agentic group shilling attacks in RecSys. They share some structural shortcomings. (1) Costly preparation: learning-based methods require hours of training before the first fake profile can be produced, and LLM-driven methods still generate expensive fake profiles or recursively sample new batches when the attack underperforms. (2) Static patterns: every fake user is locked to one generation algorithm or profile template for the entire campaign, leaving detectable distributional fingerprints. (3) Static strategy: there is no mechanism to redirect the group when the campaign underperforms. For example, fake users may rate the same suboptimal sequence of items.

This motivates us to propose the Agentic Group Attack System (AGAS). Powered by the growing planning and reasoning capabilities of LLMs [24]–[30], AGAS addresses each shortcoming in turn. (1) Training-free: A hidden Coordinator autonomously orchestrates the group from victim feedback, worker signals, and safety checks, without any training costs. (2) Coordination: the Coordinator jointly decides which role a worker should take, and when the group should attack, slow down, or stay inactive. This makes the campaign adaptive at the group level, rather than a set of independent fake users repeatedly pushing the same target. (3) Role-switching: Each worker is a stateful agent with its own memory. It switches roles (Profiler, Sniper, Camouflageur, or Inactive) across rounds under the Coordinator’s supervision, reliably breaking the stable patterns that detectors rely on. (4) Dynamic strategies: AGAS switches strategies (§IV-C) based on feedback from victims. This allows it to push aggressively when safe, slow down when risky, and avoid detectable attack patterns. This design lifts all five axes in Fig. 1. It improves Speed by eliminating training latency, and improves Detector Evasion, Profile Realism, and Unpopular and Popular Item scores through coordination, role-switching, and dynamic strategies. Our contributions are summarized as follows:

• Formulation: To the best of our knowledge, this is the first formulation of coordinated group shilling as an adaptive campaign, where multiple fake users are jointly managed to maximize target exposure while maintaining stealth.

• Method: We propose AGAS, an agentic framework where a Coordinator manages specialized worker roles instead of treating fake users as independent attackers. It uses feedback signals to adjust role assignments and attack strategies without offline fine-tuning.

• Evaluation: Across different datasets and victim models, AGAS consistently outperforms conventional and agentic baselines under the same evaluation protocols, while preserving benign recommendation quality and weakening representative detectors.

• Benchmark: AGAS is both effective and efficient since it enables fast stress-testing of RecSys against sophisticated shilling attacks without costly fine-tuning or repeated surrogate retraining.

## II. RELATED WORK

Model-based CF learns user and item representations from historical interactions and remains widely used in RecSys [31].

We consider two main families. Embedding-based models include MF [1], NMF [32], NCF, NeuMF, and GMF [2]. Graph-based models propagate embeddings over user–item graphs, including NGCF [3], LightGCN [4], SimGCL [33], XSimGCL [34], EGCF [35], and LightCCF [36]. Existing attacks are often tailored to specific victim families, while AGAS operates in a black-box setting and adapts from victim feedback that is observable to the attacker.

White-box attacks. White-box attacks assume access to victim objectives, gradients, or parameters. PGA [8] derives poisoning gradients for factorization models, while CLeaR [10] and InfoAtk [37] target contrastive or graph-based recommenders. Such access can enable strong attacks but limits practical applicability.

Gray and Black-box attacks. Gray-box attacks use partial knowledge, typically through surrogate models. AIA [38] learns transferable fake-user behaviors, while GSPAttack [9] generates fake users and graph edges for GNN recommenders. Their effectiveness still depends on surrogate quality and victim-family assumptions. Black-box attacks use only observable behavior. Random Attack [20], Bandwagon Attack [5], HybridAttack [39], and its social extension [40] use fixed heuristics or templates. AUSH [6] and GOAT [7] learn profile generators, while PoisonRec [12], CopyAttack [13], and MultiAttack [18] use reinforcement learning. R-Trojan [41] additionally exploits textual reviews. These methods improve profile generation but often require costly generative or policy training before deployment.

Agentic attacks. Recent LLM-based methods introduce agents into black-box recommender attacks. CheatAgent [16] uses a single LLM agent against LLM-empowered recommenders. AgentSA [14] assigns one agent to each fake account and rewrites profiles until validation succeeds. AgentAttack [15] uses an LLM planner to select pre-generated attack profiles from existing methods [5]–[7], [20], with weak combinations filtered through surrogate retraining. In contrast, AGAS performs adaptive campaign management. A Coordinator manages role-switching workers, observes round-level feedback, pauses or slows attacks under suppression, replaces risky workers, and updates roles and strategies. Therefore, the attack evolves across rounds without relying on fixed profiles, prompts, or pre-generated libraries.

## III. FORMULATION

## A. Collaborative Filtering with Implicit Feedback

Let $\mathcal { U } = \{ 1 , \dots , M \}$ and $\mathcal { T } = \{ 1 , \ldots , N \}$ denote the user and item sets. The implicit matrix $\mathbf { R } \in \{ 0 , 1 \} ^ { M \times N }$ has $R _ { u i } ~ = ~ 1$ if user u interacted with item i, and $R _ { u i } ~ = ~ 0$ otherwise. CF learns a scoring function $\hat { y } _ { u i } = f _ { \Theta } ( u , i )$ from each user’s positive set ${ \mathcal { T } } _ { u } ^ { + } = \{ i \in { \mathcal { T } } | R _ { u i } = 1 \}$ . This formulation covers common CF backbones, including matrix factorization [2], graph convolution [4], and graph-contrastive recommenders [34]. BPR [42] is used as the training objective:

$$
\mathcal { L } _ { \mathrm { B P R } } ( \boldsymbol { \Theta } ) = \sum _ { ( u , i , j ) \in \mathcal { D } } - \log \sigma ( \hat { y } _ { u i } - \hat { y } _ { u j } ) + \lambda \lVert \boldsymbol { \Theta } \rVert _ { 2 } ^ { 2 } ,\tag{1}
$$

![](images/3a5a32acffa43762a738b26184215afbb8aa02612af30ace4498b9c575a98c21.jpg)  
Fig. 2: Overview of the AGAS pipeline. Across attack rounds, the Coordinator aggregates worker and environment signals to adapt fake-user roles and attack strategies, progressively steering the campaign toward effective promotion of the target item. AGAS effectively uses feedback-derived signals to probe the victim, slow down when suppression emerges, and keep some workers inactive to avoid repetitive or overly synchronized behavior.

where $\mathcal { D } = \{ ( u , i , j ) ~ | ~ u \in \mathcal { U } , i \in \mathcal { I } _ { u } ^ { + } , j \notin \mathcal { T } _ { u } ^ { + } \}$ , $\sigma ( \cdot )$ is the sigmoid, and λ is the regularization weight. At inference, the recommender returns:

$$
\mathrm { T o p K } ( u ) = \underset { q \in \mathbb { Z } \backslash \mathbb { Z } _ { u } ^ { + } } { \mathrm { a r g ~ t o p K } } \hat { y } _ { u q } .\tag{2}
$$

## B. Targeted Shilling Attack

The attacker selects a target item $i _ { \mathrm { t a r } } \in \mathcal { T }$ and aims to push it into the top-K list of benign test users $\mathcal { U } _ { \mathrm { t e s t } }$ , measured by:

$$
\mathrm { H R @ } K ( i _ { \mathrm { t a r } } ) = \frac { 1 } { | \mathcal { U } _ { \mathrm { t e s t } } | } \sum _ { u \in \mathcal { U } _ { \mathrm { t e s t } } } \mathbb { I } [ i _ { \mathrm { t a r } } \in \mathrm { T o p K } ( u ) ] .\tag{3}
$$

Following standard targeted shilling evaluation, the attacker injects fake users as new accounts, while attack success is measured on held-out benign users by checking whether the target item appears in their top-K recommendation lists.

Threat model. We assume a black-box setting: the attacker has no access to deployed-victim parameters, gradients, training procedure, evaluation-user identities, held-out interactions, or exact platform discount weights. The attacker injects $M _ { f }$ fake users ${ { \mathcal U } _ { f } } ~ = ~ \left\{ 1 , \dots , { { M } _ { f } } \right\}$ , where each fake user w has interaction vector $\tilde { \mathbf { r } } _ { w } \in \mathbf { \bar { \{ 0 , 1 \} } } ^ { N }$ . Stacking them gives $\tilde { \mathbf { R } } \in \{ 0 , 1 \} ^ { M _ { f } \times N }$ , appended to the clean data:

$$
\mathbf { R } ^ { \prime } = \left\lceil \mathbf { R } ; \tilde { \mathbf { R } } \right\rceil .\tag{4}
$$

Attack objective. Let $\boldsymbol { \mathcal { A } } ( \cdot )$ be the training procedure mapping a poisoned matrix to a deployed recommender. AGAS maximizes target exposure while keeping injected profiles within a stealth tolerance:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \{ \tilde { \mathbf { r } } _ { w } \} _ { w \in \mathcal { U } _ { f } } } ~ \frac { 1 } { | \mathcal { U } _ { \mathrm { t e s t } } | } \sum _ { u \in \mathcal { U } _ { \mathrm { t e s t } } } \mathbb { I } [ i _ { \mathrm { t a r } } \in \mathrm { T o p K } ( u ; \mathcal { A } ( \mathbf { R } ^ { \prime } ) ) ] } & { } \\ { \mathrm { s . t . } ~ \mathcal { S } ( \tilde { \mathbf { R } } ; \mathbf { R } ) \le \varepsilon , } \end{array}\tag{5}
$$

where $\mathrm { T o p K } ( u ; \mathcal { A } ( \mathbf { R } ^ { \prime } ) )$ is the top-K list returned to benign test user u by the recommender trained on the poisoned matrix $\mathbf { R } ^ { \prime } , { \mathcal { S } } ( \cdot ; \cdot )$ measures anomaly relative to benign profiles, and ε is the attacker–defender tolerance. In AGAS, this tolerance is operationalized through the worker and environment signals in §IV, which decide whether to continue, slow down, pause, or switch strategy. AGAS constructs R<sup>˜</sup> over T rounds using only attacker-accessible feedback, obtained from recommendation lists returned to its own fake accounts rather than from $\mathcal { U } _ { \mathrm { t e s t } }$ or any hidden metric of the deployed victim (§IV). The reported attack uses the full poisoning history accumulated over all T rounds, so R<sup>′</sup> in Eq. 4 is the matrix on which we report HR@K and NDCG@K in §V.

## IV. METHOD

We present the Agentic Group Attack System (AGAS), as shown in Fig. 2, where a Coordinator manages a pool of fake users over multiple rounds. The team contains four worker roles: Profiler (PR), Camouflageur (CA), Sniper (SN), and Inactive (IN). The Coordinator tracks global attack progress, worker safety, and campaign state, then adjusts both worker roles and attack strategy over time, without training latency. The complete process is described in Alg. 1.

## A. Coordinator

At round t, the Coordinator reads the current observation $\mathbf { o } _ { t }$ updates memory m<sub>t</sub>, and computes the control signals directly from workers’ actions and environment feedback.

Worker signals. These signals are maintained for each worker and summarize how safe that worker is.

• Trust and risk scores. AGAS maintains a trust score $\tau _ { t , w }$ and a risk score $\gamma _ { t , w }$ for each worker w. After an accepted action on item i with rating $r _ { t , w } ,$ it computes the deviation from the current item bias as: $d _ { t , w } = | r _ { t , w } - b _ { i } |$ Trust increases only when the action remains close to the item bias:

$$
\tau _ { t + 1 , w } = \operatorname* { m a x } ( 0 , \tau _ { t , w } + \mathbb { I } [ d _ { t , w } \leq 1 ] ) .\tag{6}
$$

Algorithm 1 AGAS Attack Procedure   
1: Input: clean matrix R, target item $i _ { \mathrm { t a r } } .$ fake users $\mathcal { U } _ { f } .$   
per-user budget L, rounds $T .$   
2: Output: poisoned matrix ${ \bf R } ^ { \prime } = [ { \bf R } ; { \tilde { \bf R } } ] .$   
3: Initialize Coordinator memory, worker states, victim fam  
ily as unknown, and $\tilde { \mathbf { R } }  \varnothing .$   
4: Initialize worker signals as $\tau _ { 0 , w } ~ = ~ 0 , ~ \gamma _ { 0 , w } ~ = ~ 0 ,$ and   
$\phi _ { 0 , w } = 0$ for each $w \in \mathcal { U } _ { f } .$   
5: Initialize environment signals as $\eta _ { 0 } = 1 , \xi _ { 0 } = ( q _ { 0 } , s _ { 0 } ) =$   
(0, 0), and $a _ { 0 } = 0 .$   
6: for $t = 0 , \ldots , T - 1$ do   
7: Build observation o<sub>t</sub> and update mem  
ory m<sub>t</sub> from the current signals   
$( \rho ^ { ( t ) } , \Delta \rho ^ { ( t ) } , \{ \tau _ { t , w } , \gamma _ { t , w } , \phi _ { t , w } \} _ { w \in \mathcal { U } _ { f } } , \eta _ { t } , \xi _ { t } , a _ { t } )$   
8: Based on these signals, the Coordinator selects one   
round-level strategy and assigns each worker a role in   
{PR, CA, SN, IN}.   
9: The Coordinator validates and aggregates accepted   
worker actions into one batch $\Delta \tilde { \mathbf { R } } ^ { ( t + 1 ) } .$   
10: Update $\tilde { \textbf { R } }  \tilde { \textbf { R } } \cup \Delta \tilde { \textbf { R } } ^ { ( t + 1 ) }$ and form the current   
poisoned matrix $[ \mathbf { R } ; { \tilde { \mathbf { R } } } ] .$   
11: Update worker signals, rank feedback, and environment   
signals $( \eta _ { t + 1 } , \xi _ { t + 1 } , a _ { t + 1 } )$   
12: Update victim-family prediction, worker states, and   
Coordinator memory.   
13: if the stopping condition is met then   
14: break   
15: end if   
16: end for   
17: $\mathbf { R } ^ { \prime } \gets [ \mathbf { R } ; \tilde { \mathbf { R } } ] .$   
18: return R<sup>′</sup>

Risk increases when the action deviates from the bias, with an additional penalty for extreme actions from lowtrust workers:

$$
\begin{array} { r } { \gamma _ { t + 1 , w } = \operatorname* { m a x } ( 0 , \gamma _ { t , w } + 0 . 5 \mathbb { I } [ d _ { t , w } > 1 ] \ ~ } \\ { + 0 . 5 \mathbb { I } [ d _ { t , w } > 1 . 5 \wedge \tau _ { t , w } < 1 ] ) . } \end{array}\tag{7}
$$

If worker w is inactive in round t, risk decays as:

$$
\gamma _ { t + 1 , w } = \operatorname* { m a x } ( 0 , \gamma _ { t , w } - 0 . 5 ) .\tag{8}
$$

• Validator score. $\phi _ { t , w }$ measures whether worker w’s whole profile appears structurally suspicious:

$$
\phi _ { t , w } = \operatorname* { m i n } ( 1 , \ 0 . 4 e _ { t , w } + 0 . 3 h _ { t , w } + 0 . 3 o _ { t , w } )\tag{9}
$$

where $e _ { t , w }$ is the frequency of extreme ratings (near the two ends of the scale), $h _ { t , w }$ indicates whether w has given the target item a clear positive rating (near the positive end of the scale), and $o _ { t , w }$ measures fake-user overlap, i.e., the fraction of $w ^ { \prime } \mathbf { s }$ rated items also rated by at least two other fake users. Unlike $\tau _ { t , w }$ and $\gamma _ { t , w }$ , which track recent actions, $\phi _ { t , w }$ evaluates profile-level suspicion.

Environment signals. These signals summarize how the victim model and platform are reacting to the campaign at each round. Unlike worker signals, which describe the status of individual fake users, environment signals capture global progress, possible defensive behavior, and campaign-level risk.

• Rank feedback. $\rho ^ { ( t ) }$ and $\Delta \rho ^ { ( t ) }$ denote the current mean target rank and its recent change. Together, they indicate whether the campaign is effectively promoting the target item or has started to stall. A consistent improvement in $\rho ^ { ( t ) }$ suggests that the current strategy is working, whereas a weak or negative $\Delta \rho ^ { ( t ) }$ may suggest that AGAS should adjust its strategy, increase exploration, or switch workers to safer roles.

• Acceptance rate. $\eta _ { t }$ is the recent ratio of accepted to attempted actions by active workers. It measures whether injected interactions are still being incorporated by the victim system. A sudden drop in $\eta _ { t }$ may indicate silent filtering, discounting, or defensive suppression, even without an explicit alert.

• Suppression signal. ${ \boldsymbol \xi } _ { t } ~ = ~ ( q _ { t } , s _ { t } )$ denotes the roundlevel suppression signal, where $q _ { t } \in [ 0 , 1 ]$ is the overall suspicion score and $s _ { t }$ is the campaign-level suppression streak. The streak $s _ { t }$ increases when the current round shows signs of filtering, discounting, or weak targetrank improvement. Let W be the active fake workers at round t. Let $\mathcal { A } _ { t } ^ { \mathrm { t r y } } , \mathcal { A } _ { t } ^ { \mathrm { d r o p } } , \mathcal { A } _ { t } ^ { \mathrm { a c c } }$ , and $\boldsymbol { A } _ { t } ^ { \mathrm { t a r } }$ denote the attempted, dropped, accepted, and target-related actions in that round. We define:

$$
\begin{array} { r l } & { \hat { d } _ { t } = \displaystyle \frac { \left| \mathcal { A } _ { t } ^ { \mathrm { t r o p } } \right| } { \operatorname* { m a x } ( 1 , | \mathcal { A } _ { t } ^ { \mathrm { t r o p } } | ) } , } \\ & { \hat { \delta } _ { t } = \operatorname* { m i n } \left( 1 , \frac { 1 } { 2 | \mathcal { A } _ { t } ^ { \mathrm { a c c } } | } \sum _ { a \in \mathcal { A } _ { t } ^ { \mathrm { a c c } } } \delta ( a ) \right) , } \\ & { \hat { m } _ { t } = \displaystyle \frac { 1 } { \operatorname* { m a x } ( 1 , | \mathcal { A } _ { t } ^ { \mathrm { t r o r } } | ) } \sum _ { a \in \mathcal { A } _ { t } ^ { \mathrm { t r o r } } } \mathbb { I } [ \Delta \rho ^ { \mathrm { t a r } } ( a ) < \epsilon _ { \rho } ] , } \\ & { \hat { s } _ { t } = \operatorname* { m i n } ( 1 , s _ { t } / 3 ) . } \end{array}\tag{10}
$$

Here, $\hat { d } _ { t }$ is the dropped-action ratio, $\hat { \delta } _ { t }$ is the normalized discount magnitude, $\hat { m } _ { t }$ is the weak target-movement ratio, and $\hat { s } _ { t }$ is the normalized suppression streak. $\delta ( a )$ is the estimated discount applied to accepted action $^ { a , }$ $\Delta \rho ^ { \mathrm { t a r } } ( a )$ is the target-rank improvement after action $^ { a , }$ and $\epsilon _ { \rho }$ is the minimum improvement threshold. We also compute a round-level group-overlap score:

A<sub>t,w</sub> = {items rated by worker w at round t},

$$
g _ { t } = \operatorname* { m a x } _ { u , v \in \mathcal { W } _ { t } , \ u \neq v } \frac { \left| A _ { t , u } \cap A _ { t , v } \right| } { \left| A _ { t , u } \cup A _ { t , v } \right| } .\tag{11}
$$

This captures whether fake workers are acting too similarly in the same round. To avoid over-tuning the suppression score, we give the five components equal weight and use a simple linear combination to compute $q _ { t }$ . The final round-level suspicion score is:

$$
q _ { t } = \operatorname* { m i n } \Bigl ( 1 , 0 . 2 \hat { d } _ { t } { + } 0 . 2 \hat { \delta } _ { t } { + } 0 . 2 \hat { m } _ { t } { + } 0 . 2 \hat { s } _ { t } { + } 0 . 2 g _ { t } \Bigr ) .\tag{12}
$$

A large $q _ { t }$ means the campaign is likely being filtered, discounted, or detected at the environment level.

• Alert flag. $a _ { t } ~ \in ~ \{ 0 , 1 \}$ is the round-level alert flag. AGAS sets $a _ { t } ~ = ~ 1$ when the current round creates an abnormal target spike, unusually high worker overlap, or a sharp drop in action acceptance. This is a coarse practical trigger rather than a formal detector. It indicates a high-risk state and encourages the Coordinator to slow down, pause, or switch to safer strategies.

Before the campaign begins, AGAS evaluates the clean victim once to obtain the initial target rank $\rho ^ { ( 0 ) }$ . Since no fake action has been injected, $\Delta \rho ^ { ( 0 ) } = 0$ . For each worker w, the Worker signals are initialized as $\tau _ { 0 , w } = 0 , \gamma _ { 0 , w } = 0$ , and $\phi _ { 0 , w } = 0$ . The round-level Environment signals also start from neutral values: $\eta _ { 0 } = 1 , \xi _ { 0 } = ( q _ { 0 } , s _ { 0 } ) = ( 0 , 0 )$ , and $a _ { 0 } = 0$ . The Worker signals indicate which fake users are still effective or becoming risky. The Environment signals summarize whether the victim platform appears to be accepting the campaign or entering a defensive state. Based on this feedback, the Coordinator decides whether to explore, attack, slow down, pause, or switch roles before selecting the next strategy.

## B. Workers

Each worker keeps its own memory. This lets each worker decide what item to rate next from its own action history. Group-level coordination is still maintained by the Coordinator. The worker memory supports item-level decisions.

Profiler (PR). PR is the exploration role. It uses a few safe interactions on filler items, i.e., popular or related non-target items that normal users would reasonably rate, to test whether the platform is accepting actions and whether suppression or throttling is emerging. When feedback suggests a graph-style victim, it prioritizes bridge items, i.e., an intermediate item that links the target to real users through shared preferences, to find receptive users and expand the attack path.

Camouflageur (CA). CA is the stealth role. It passively rates filler items instead of aggressively pushing the target, making the fake-user history look more like normal activity than coordinated promotion. This role dilutes suspicious behavior, helps rebuild trust after risky rounds, and keeps the fake-user pool active when the Coordinator slows down.

Sniper (SN). SN is the payload role, assigned to hightrust and low-risk workers because it creates the strongest rank movement but also the highest detection pressure. For embedding-based victims such as GMF, SN directly promotes the target item. For graph-based victims such as LightGCN, SN uses bridge items discovered by PR to form useful twohop paths between the target and receptive real-user neighborhoods, strengthening collaborative propagation through the interaction graph.

Inactive (IN). The Inactive role means that the worker does not act in the current round. This role is necessary because a suspicious worker should disappear completely instead of continuing weak benign behavior.

## C. Strategy

For each round, the Coordinator selects one strategy and assigns one role to each worker.

1. Victim Probe. In the first few rounds, the Coordinator assigns some workers to PR to probe the victim family. If direct target-related actions improve the mean target rank $\rho ^ { ( t ) }$ or produce a favorable rank change $\Delta \rho ^ { ( t ) }$ , the victim is treated as more likely embedding-based. Otherwise, it is treated as more likely graph-based. This is sufficient to guide later role allocation, as supported in §V-E.

2. Bridge Building. For graph-based victims, PR workers build a bridge-item pool by ranking non-target items according to how often they are rated by real users connected to the target. Items that produce favorable rank feedback for the target are added to the pool. These bridge items help SN create stronger two-hop paths to the target.

3. Warm-up. After probing, the Coordinator avoids early over-attack. It usually assigns one PR and a small number of CA so the campaign can build worker trust $\tau _ { t , w }$ and collect clean feedback before using SN.

4. First Push. After warm-up, the Coordinator launches the first coordinated payload. A small number of SN workers attack together while at least one CA remains active as cover. This step aims to improve $\rho ^ { ( t ) }$ and $\Delta \rho ^ { ( t ) }$ without making the whole worker pool look synchronized.

5. Silent Slowdown. If the acceptance rate $\eta _ { t }$ falls below the early baseline while no alert is raised $( a _ { t } = 0 )$ , the platform may be silently suppressing the attack. The Coordinator reduces campaign-level pressure by shifting only the cleanest high-trust workers with larger $\tau _ { t , w }$ toward CA behavior, while others may keep baseline roles, including SN. This keeps AGAS less visible while it observes whether stronger attack is needed.

6. Profile Cleanup. If worker risk $\phi _ { t , w }$ rises, suppression $\xi _ { t } =$ $\left( q _ { t } , s _ { t } \right)$ increases, or an alert appears $( a _ { t } = 1 )$ ), the Coordinator sanitizes the active pool. It freezes the most suspicious SN, avoids profiles with strong fake-user overlap, and only nudges the cleanest IN into CA mode to preserve benign throughput. This strategy mainly removes detectable profiles.

7. Safe Replacement. If the environment enters a high-risk state, indicated by $a _ { t } ~ = ~ 1$ or a large round-level suspicion score $q _ { t } .$ , the Coordinator substitutes suspicious workers with rested ones that have lower risk $\gamma _ { t , w }$ and assigns these replacements to safer roles. This keeps the campaign operational while avoiding repeated use of the same exposed workers. The number of SN workers remains the same as before, but they are more likely to be trusted and less likely to be detected.

8. Main Attack. If no strong suspicious signals appear and the target is still far from top-K according to $\Delta \rho ^ { ( t ) }$ , the best available worker is assigned the SN role. The Coordinator prefers workers with higher trust $\tau _ { t , w }$ and lower risk $\gamma _ { t , w } ,$ and continues pushing until the target reaches top-K or the environment signals indicate stronger defense.

## D. Complexity Analysis

Let T denote the number of attack rounds, $| \mathscr { U } _ { f } |$ the fake-user pool size, L the per-user action budget, V the victim-query cost, and $c _ { C } , c _ { W }$ the costs of one Coordinator and workers calls. For embedding-based victims, $V = O ( | \mathcal { U } _ { f } | d )$ , with d the embedding dim. For graph-based victims, $V = { \cal O } ( ( | R | +$ $| \tilde { R } ^ { ( \leq t ) } | ) d K )$ with K propagation layers. The complexity of Alg. 1 is: $O \big ( T ( c _ { C } + | \mathcal { U } _ { f } | ( c _ { W } + L ) + V ) \big )$ , with only O(T)

TABLE I: Unpopular-target promotion on embedding-based (top) and graph-based (bottom) victims. Mean±95% CI over 5 runs $( \times 1 0 ^ { 3 } )$ . Best / second-best.
<table><tr><td rowspan="3">Method</td><td colspan="3">ML-100K</td><td colspan="2"></td><td colspan="3">ML-1M</td><td colspan="3">Amazon</td><td colspan="3"></td><td colspan="3">Genome 2021</td><td colspan="3">Netflix</td></tr><tr><td>MF (BPR)</td><td colspan="2"></td><td colspan="2">NeuMF</td><td colspan="2">GMF</td><td colspan="2">NCF MF (BPR)</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">NeuMF</td><td colspan="2">MF (BPR)</td><td colspan="2">NeuMF</td></tr><tr><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10|</td><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10|</td><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10| HR@10 NDCG@10</td><td></td><td></td><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10</td></tr><tr><td></td><td>1.8±0.2</td><td>0.7±0.5</td><td>2.2±0.2</td><td>0.9±0.6</td><td>0.9±0.1</td><td></td><td></td><td>0.3±0.2</td><td>0.4±0.1</td><td></td><td></td><td></td><td>|0.2±0.1</td><td></td><td></td><td></td><td></td><td>0.2±0.2</td><td>0.7±0.1</td><td></td></tr><tr><td>NoneAttack</td><td></td><td></td><td></td><td></td><td></td><td>0.4±0.3</td><td>0.8±0.1</td><td></td><td></td><td>0.1±0.2</td><td>0.5±0.1</td><td>0.2±0.2</td><td></td><td>0.1±0.2</td><td>0.5±0.1</td><td>0.2±0.2</td><td>0.6±0.1</td><td></td><td></td><td>0.3±0.2</td></tr><tr><td>RandomAttack [5]</td><td>4.1±0.3</td><td>1.6±0.6</td><td>4.7±0.3</td><td>1.8±0.6</td><td>2.4±0.2</td><td>1.0±0.5</td><td>2.0±0.2</td><td>0.8±0.4</td><td>1.2±0.1</td><td>0.5±0.3</td><td>1.3±0.1</td><td>0.5±0.3</td><td>0.7±0.1</td><td>0.3±0.2</td><td>1.3±0.1</td><td>0.5±0.3</td><td>1.8±0.2</td><td>0.7±0.4</td><td>2.0±0.2</td><td>0.8±0.5</td></tr><tr><td>BandwagonAttack [5]</td><td>6.2±0.3 10.6±0.6</td><td>2.5±0.7 4.3±1.2</td><td>6.8±0.3</td><td>2.7±0.8</td><td>3.4±0.2</td><td>1.4±0.6</td><td>2.9±0.2</td><td>1.2±0.5</td><td>2.0±0.1</td><td>0.8±0.4</td><td>2.2±0.1</td><td>0.9±0.4</td><td>1.2±0.1</td><td>0.5±0.3</td><td>2.0±0.1</td><td>0.8±0.4</td><td>2.7±0.2</td><td>1.1±0.5</td><td>3.0±0.2</td><td>1.2±0.5</td></tr><tr><td>AUSH [6]</td><td>13.4±0.7</td><td>5.3±1.6</td><td>10.0±0.5 11.5±0.6</td><td>4.1±1.1</td><td>6.5±0.4</td><td>2.6±0.9</td><td>4.3±0.3</td><td>1.8±0.7</td><td>3.3±0.3</td><td>1.3±0.6</td><td>3.1±0.3</td><td>1.2±0.6</td><td>1.9±0.2</td><td>0.8±0.5</td><td>2.8±0.3</td><td>1.1±0.6</td><td>4.2±0.3</td><td>1.7±0.7</td><td>4.5±0.3</td><td>1.8±0.8</td></tr><tr><td>PoisonRec [12] PGA [8]</td><td>11.5±0.6</td><td>4.6±1.3</td><td>12.2±0.6</td><td>4.7±1.5 5.0±1.4</td><td>7.6±0.5 8.4±0.5</td><td>3.0±1.1 3.4±1.1</td><td>5.5±0.4 6.0±0.4</td><td>2.1±0.9 2.4±0.9</td><td>5.8±0.5 4.3±0.3</td><td>2.3±1.0</td><td>4.8±0.3 4.0±0.3</td><td>1.9±0.8</td><td>2.3±0.3</td><td>0.9±0.6</td><td>3.2±0.3</td><td>1.3±0.7</td><td>5.6±0.4</td><td>2.2±1.0</td><td>5.7±0.4</td><td>2.3±1.0</td></tr><tr><td>AgentSA [14]</td><td>12.7±0.6</td><td>5.2±1.4</td><td>12.5±0.6</td><td>5.1±1.4</td><td>8.2±0.4</td><td>3.3±1.0</td><td>7.2±0.4</td><td>2.9±1.0</td><td>5.6±0.3</td><td>1.7±0.8 2.2±0.9</td><td>4.9±0.3</td><td>1.6±0.7 1.9±0.8</td><td>2.8±0.2 2.9±0.2</td><td>1.1±0.6 1.1±0.6</td><td>3.8±0.3 4.2±0.3</td><td>1.5±0.7 1.7±0.7</td><td>5.5±0.4 6.3±0.4</td><td>2.2±1.0 2.5±1.0</td><td>6.0±0.5 6.5±0.4</td><td>2.4±1.0 2.6±1.0</td></tr><tr><td>AgentAttack [15]</td><td>13.6±0.7</td><td>5.5±1.5</td><td>12.1±0.6</td><td>4.9±1.4</td><td>8.8±0.5</td><td>3.5±1.1</td><td>6.8±0.4</td><td>2.7±1.0</td><td>5.7±0.3</td><td>2.2±0.9</td><td>5.2±0.3</td><td>2.0±0.9</td><td>2.7±0.2</td><td>1.0±0.6</td><td>4.5±0.3</td><td>1.8±0.8</td><td>6.1±0.4</td><td>2.4±1.0</td><td>6.9±0.5</td><td>2.8±1.0</td></tr><tr><td>AGAS (Ours)</td><td>40.0±0.2</td><td>16.1±0.3</td><td>35.0±0.2</td><td>14.2±0.3</td><td>22.6±0.1</td><td>9.0±0.2</td><td>17.6±0.1</td><td>7.1±0.2</td><td>16.1±0.1</td><td>6.3±0.2</td><td>15.0±0.1</td><td>5.9±0.2</td><td>8.2±0.1</td><td>3.2±0.2</td><td>11.5±0.1</td><td>4.6±0.2</td><td>18.2±0.1</td><td>7.3±0.2</td><td>19.6±0.1</td><td>7.9±0.2</td></tr><tr><td>Improvement</td><td>187.8%</td><td>182.5%</td><td>186.9%</td><td>184.0%</td><td>162.8%</td><td>164.7%</td><td>155.1%</td><td>153.6%</td><td>177.6%</td><td>173.9%</td><td>172.7%</td><td>168.2%</td><td>|192.9%</td><td>190.9%</td><td>161.4%</td><td>155.6%</td><td>198.4%</td><td>192.0%</td><td>192.5%</td><td>192.6%</td></tr><tr><td rowspan="4"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Douban Movie</td><td></td><td></td></tr><tr><td></td><td>NGCF</td><td></td><td>LightGCN</td><td></td><td>SimGCL</td><td></td><td>XSimGCL</td><td></td><td>EGCF</td><td></td><td>LightCCF</td><td></td><td>NGCF</td><td></td><td>LightGCN</td><td></td><td>NGCF</td><td></td><td>LightGCN</td></tr><tr><td></td><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10|</td><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10|</td><td>HR@10 NDCG@10</td><td>HR@10</td><td></td><td>NDCG@10| HR@10 NDCG@10</td><td></td><td>HR@10</td><td>NDCG@10|</td><td>HR@10</td><td>NDCG@10</td><td></td><td>HR@10</td><td>NDCG@10</td></tr><tr><td></td><td>1.9±0.2</td><td>0.8±0.5</td><td>2.4±0.2</td><td>1.0±0.6</td><td>0.8±0.1</td><td>0.3±0.2</td><td>0.9±0.1</td><td>0.4±0.3</td><td>0.4±0.1</td><td>0.2±0.2</td><td>0.5±0.1</td><td>0.2±0.2</td><td>0.2±0.1</td><td>0.1±0.2</td><td>0.3±0.1</td><td>0.1±0.2</td><td>0.7±0.1</td><td>0.3±0.2</td><td>0.6±0.1</td><td>0.2±0.2</td></tr><tr><td>NoneAttack RandomAttack [5]</td><td>4.5±0.3</td><td>1.8±0.6</td><td>5.0±0.3</td><td>2.0±0.7</td><td>1.8±0.2</td><td>0.7±0.4</td><td>2.2±0.2</td><td>0.9±0.5</td><td>1.1±0.1</td><td>0.4±0.3</td><td>1.2±0.1</td><td>0.5±0.3</td><td>0.6±0.1</td><td>0.2±0.2</td><td>0.6±0.1</td><td>0.2±0.2</td><td>2.0±0.2</td><td>0.8±0.5 1.1±0.6</td><td>1.9±0.2 2.7±0.2</td><td>0.7±0.4 1.1±0.5</td></tr><tr><td>BandwagonAttack [5]</td><td>5.9±0.3 11.0±0.6</td><td>2.4±0.7 4.5±1.3</td><td>6.7±0.3 11.5±0.6</td><td>2.7±0.8 4.7±1.3</td><td>2.4±0.2 4.8±0.3</td><td>1.0±0.5 1.9±0.8</td><td>2.9±0.2 5.8±0.4</td><td>1.2±0.6 2.3±0.9</td><td>1.8±0.2 3.2±0.3</td><td>0.7±0.4 1.3±0.6</td><td>1.9±0.2 3.4±0.3</td><td>0.8±0.4 1.4±0.6</td></table>

heavy Coordinator calls and $O ( T | U _ { f } | )$ short worker calls. AGAS keeps the heavy LLM cost linear in T, whereas AgentAttack [15] first generates C candidate profiles per worker and runs over a surrogate recommender, adding $O ( | U _ { f } | C )$ generation plus surrogate retraining. The linear-in-|U<sub>f</sub>| gap explains the token and runtime savings of AGAS in §V-F.

## V. EXPERIMENTS

We design experiments to answer the following research questions (RQs):

• RQ1 (Performance). How does AGAS perform across different victim models and popularity regimes?

• RQ2 (Stealthiness). Can AGAS promote target items while preserving realistic user behavior?

• RQ3 (Detector). How do anomaly detectors perform against AGAS?

• RQ4 (Ablation). How does each component in AGAS contribute to overall performance?

• RQ5 (Efficiency). How effectively and efficiently does AGAS scale as a benchmarking and stress-testing tool for RecSys?

## A. Experimental Setup

AGAS is a layered LLM-agent system inspired by Re-Act [51] and Reflexion [52]. Fake-user workers use a ReActstyle loop to reason over profile states and execute rating actions, while the Coordinator uses a Reflexion-style loop to aggregate signals, update campaign memory, and adapt future roles and strategies.

Datasets. We evaluate AGAS on public recommendation datasets that are commonly used for CF research and shillingattack evaluation. These datasets cover a range of domains, sizes, and interaction patterns:

• MovieLens-100K (ML-100K) [53].

• MovieLens-1M (ML-1M) [53].

• MovieLens Tag Genome 2021 (Genome 2021) [54].

• Netflix Prize (Netflix) [55].

• Douban Movie (Douban) [56].

• Amazon Reviews 2018 (Amazon) [57].

Evaluation Metrics. We report HR@K and NDCG@K for the target item $i _ { \mathrm { t a r } }$ on test users $\mathcal { U } _ { \mathrm { t e s t } }$ , where:

$$
\mathrm { H R @ } K = \frac { 1 } { | \mathcal { U } _ { \mathrm { t e s t } } | } \sum _ { u \in \mathcal { U } _ { \mathrm { t e s t } } } \mathbb { I } [ i _ { \mathrm { t a r } } \in \mathrm { T o p K } ( u ) ] ,\tag{13}
$$

$$
\mathrm { N D C G @ } K = \frac { 1 } { | \mathcal { U } _ { \mathrm { t e s t } } | } \sum _ { u \in \mathcal { U } _ { \mathrm { t e s t } } } \frac { \mathbb { I } [ i _ { \mathrm { t a r } } \in \mathrm { T o p K } ( u ) ] } { \log _ { 2 } ( \mathrm { r a n k } ( i _ { \mathrm { t a r } } \mid u , \mathcal { T } \setminus \mathcal { Z } _ { u } ^ { + } ) + 1 ) }\tag{14}
$$

To verify that AGAS preserves recommendation quality for test users, we report Rec@K on their held-out interactions. For each $u \in \mathcal { U } _ { \mathrm { t e s t } }$ , we check whether $i _ { u } ^ { \mathrm { t e s t } }$ appears in the poisoned model’s top-K list:

$$
\mathrm { R e c @ } K = \frac { 1 } { | \mathcal { U } _ { \mathrm { t e s t } } | } \sum _ { u \in \mathcal { U } _ { \mathrm { t e s t } } } \mathbb { I } [ i _ { u } ^ { \mathrm { t e s t } } \in \mathrm { T o p K } ( u ) ]\tag{15}
$$

A high Rec@K indicates that the poisoned model continues to serve genuine users well. This suggests that AGAS promotes the target item without corrupting the broader recommendation behavior. Stealth is measured by detector performance (Accuracy, Precision, Recall, F1) under representative anomaly checks. Lower scores mean the attack is harder to detect.

Victim Models. We evaluate across CF recommenders spanning matrix factorization, graph convolution, and contrastive paradigms: MF [1], MF (BPR) [42], NeuMF [2], GMF [2], NCF [2], NGCF [3], LightGCN [4], SimGCL [33], XSimGCL [34], EGCF [35], and LightCCF [36].

Attack Baselines. We compare against representative shilling and poisoning baselines spanning heuristic, optimizationbased, and agentic methods: NoneAttack, RandomAttack [5], BandwagonAttack [5], AUSH [6], PoisonRec [12], GSPAttack [9], CLeaR [10], PGA [8], TargetedAttack [43], AgentSA [14], and AgentAttack [15].

Comparison Protocol. Unless a section explicitly studies budget scaling, all methods are evaluated under the same fakeuser budget and target-item set for each dataset–victim pair. For agentic methods, the interaction budget is matched by imposing the same upper limit on injected actions across fake users. After each poisoned dataset is constructed, the victim is retrained using the same retraining protocol.

TABLE II: Detection performance (Accuracy / Recall / Precision / F1). Lower values are harder to detect. Results are shown for PoisonRec, AgentSA, AgentAttack, and AGAS. Best / second within each block.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Method</td><td colspan="4">ML-100K</td><td colspan="4">ML-1M</td><td colspan="4">Netflix</td><td colspan="4">Amazon</td><td colspan="4">Genome 2021</td></tr><tr><td>| Accuracy</td><td>Recall Precision</td><td></td><td>F1</td><td>Accuracy</td><td>Recall Precision</td><td></td><td>F1</td><td>Accuracy</td><td></td><td>Recall Precision</td><td>F1</td><td>|Accuracy</td><td></td><td>Recall Precision</td><td>Fl</td><td>Accuracy</td><td></td><td>Recall Precision</td><td>F1</td></tr><tr><td rowspan="4">BaseDetect [44]</td><td>PoisonRec [12]</td><td>0.942</td><td>0.920</td><td>0.915</td><td>0.917</td><td>0.938</td><td>0.915</td><td>0.910</td><td>0.912| 0.895</td><td>0.932 0.920</td><td>0.910</td><td>0.905 0.882</td><td>0.907 0.887</td><td>0.926</td><td>0.905 0.885</td><td>0.900</td><td>0.902 0.880</td><td>0.930 0.918</td><td>0.907 0.888</td><td>0.903 0.905 0.878</td></tr><tr><td>AgentSA [14]</td><td>0.932</td><td>0.905</td><td>0.895</td><td>0.900</td><td>0.928</td><td>0.900</td><td>0.890</td><td></td><td></td><td>0.892</td><td></td><td></td><td>0.914</td><td></td><td>0.875</td><td></td><td></td><td></td><td>0.883</td></tr><tr><td>AgentAttack [15]</td><td>0.918</td><td>0.892 0.345</td><td>0.880</td><td>0.886</td><td>0.913</td><td>0.886</td><td>0.875</td><td>0.880</td><td>0.905</td><td>0.878</td><td>0.866</td><td>0.872</td><td>0.900</td><td>0.872</td><td>0.860</td><td>0.866</td><td>0.904</td><td>0.875</td><td>0.864 0.870 0.340</td></tr><tr><td>AGAS (Ours)</td><td>0.385</td><td></td><td>0.360</td><td>0.352</td><td>0.372</td><td>0.330</td><td>0.348</td><td>0.339</td><td>0.360</td><td>0.315</td><td>0.335</td><td>0.325</td><td>0.352</td><td>0.305</td><td>0.325</td><td>0.315</td><td>0.365</td><td>0.320</td><td>0.330</td></tr><tr><td rowspan="4">DHAGCN [45]</td><td>PoisonRec [12]</td><td>0.934</td><td>0.910</td><td>0.903</td><td>0.906|</td><td>0.930</td><td>0.905 0.898</td><td>0.901 0.884</td><td>0.924 0.912</td><td>0.900</td><td>0.892</td><td>0.896|</td><td>0.918</td><td>0.895</td><td>0.888</td><td>0.891</td><td>0.908</td><td>0.872</td><td>0.860</td><td>0.866</td></tr><tr><td>AgentSA [14]</td><td>0.924</td><td>0.895</td><td>0.882</td><td>0.888</td><td>0.920</td><td>0.890</td><td>0.878</td><td></td><td>0.882</td><td>0.870</td><td>0.876</td><td>0.906</td><td>0.875</td><td>0.862</td><td>0.868</td><td>0.918</td><td>0.885</td><td>0.872</td><td>0.878</td></tr><tr><td>AgentAttack [15]</td><td>0.910 0.372</td><td>0.880 0.330</td><td>0.868</td><td>0.874</td><td>0.906</td><td>0.876</td><td>0.864 0.870</td><td></td><td>0.898</td><td>0.868 0.855</td><td>0.861</td><td>0.892</td><td>0.860</td><td>0.847</td><td>0.853 0.302</td><td>0.912 0.352</td><td>0.880 0.305</td><td>0.866</td><td>0.873 0.317</td></tr><tr><td>AGAS (Ours)</td><td></td><td></td><td>0.350</td><td>0.340</td><td>0.360</td><td>0.315</td><td>0.338</td><td>0.326</td><td>0.348</td><td>0.300</td><td>0.325</td><td>0.312</td><td>0.340</td><td>0.290</td><td>0.315</td><td></td><td></td><td>0.330</td><td></td></tr><tr><td rowspan="4">PCASelectUsers [46]</td><td>PoisonRec [12]</td><td>0.922 0.912</td><td>0.790</td><td>0.945</td><td>0.861</td><td>0.892</td><td>0.748 0.920</td><td>0.825</td><td>0.909</td><td>0.770</td><td>0.935</td><td>0.845|</td><td>0.880</td><td>0.730</td><td>0.912</td><td>0.811</td><td>0.906</td><td>0.765</td><td>0.932</td><td>0.840</td></tr><tr><td>AgentSA [14]</td><td>0.898</td><td>0.775</td><td>0.935</td><td>0.848</td><td>0.906</td><td>0.765 0.930</td><td>0.839</td><td>0.898</td><td>0.755</td><td>0.925</td><td>0.831</td><td>0.892</td><td>0.745</td><td>0.920</td><td>0.823</td><td>0.896</td><td>0.750</td><td>0.922</td><td>0.827</td></tr><tr><td>AgentAttack [15]</td><td>0.355</td><td>0.760 0.295</td><td>0.920</td><td>0.832</td><td>0.910 0.345</td><td>0.770 0.935</td><td>0.845 0.320</td><td>0.884 0.335</td><td>0.738</td><td>0.910 0.352</td><td>0.815 0.309</td><td>0.896</td><td>0.750</td><td>0.922</td><td>0.828</td><td>0.882</td><td>0.732</td><td>0.908</td><td>0.812</td></tr><tr><td>AGAS (Ours)</td><td></td><td></td><td>0.375</td><td>0.330</td><td></td><td>0.285</td><td>0.365</td><td></td><td></td><td>0.275</td><td></td><td></td><td>0.328</td><td>0.265</td><td>0.342 0.299</td><td>0.332</td><td>0.272</td><td>0.355</td><td>0.308</td></tr><tr><td rowspan="4">GAGE [47]</td><td>PoisonRec [12]</td><td>0.805</td><td>0.630</td><td>0.840 0.940</td><td>0.720 0.922</td><td>0.795</td><td>0.615 0.900</td><td>0.830 0.707</td><td>0.780</td><td>0.600</td><td>0.818</td><td>0.692</td><td>0.770</td><td>0.585</td><td>0.805</td><td>0.678</td><td>0.782</td><td>0.598</td><td>0.815</td><td>0.690</td></tr><tr><td>AgentSA [14]</td><td>0.930 0.880</td><td>0.905 0.755</td><td></td><td></td><td>0.925</td><td>0.935 0.882</td><td>0.917</td><td>0.918</td><td>0.895</td><td>0.930</td><td>0.912</td><td>0.912</td><td>0.890</td><td>0.925</td><td>0.907</td><td>0.920</td><td>0.895</td><td>0.930</td><td>0.912</td></tr><tr><td>AgentAttack [15]</td><td>0.400</td><td>0.310</td><td>0.892</td><td>0.818</td><td>0.870</td><td>0.738 0.348</td><td>0.804 0.319</td><td>0.860 0.380</td><td>0.722 0.280</td><td>0.870 0.335</td><td>0.788 0.305</td><td>0.852 0.372</td><td>0.708</td><td>0.862</td><td>0.778</td><td>0.864</td><td>0.722</td><td>0.872</td><td>0.792</td></tr><tr><td>AGAS (Ours)</td><td></td><td></td><td>0.360</td><td>0.333</td><td>0.390</td><td>0.295</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.270 0.325</td><td>0.295</td><td>0.382</td><td>0.285</td><td>0.340</td><td>0.310</td></tr><tr><td rowspan="4">MD-CBA [48]</td><td>PoisonRec [12]</td><td>0.790 0.935</td><td>0.615 0.910</td><td>0.830 0.945</td><td>0.707</td><td>0.780</td><td>0.600 0.905</td><td>0.820</td><td>0.693</td><td>0.765</td><td>0.585 0.808</td><td>0.679</td><td>0.870</td><td>0.738</td><td>0.875</td><td>0.802</td><td>0.768</td><td>0.585 0.900</td><td>0.805</td><td>0.678</td></tr><tr><td>AgentSA [14]</td><td>0.860</td><td>0.738</td><td>0.875</td><td>0.927 0.802</td><td>0.930 0.852</td><td>0.940 0.722 0.866</td><td>0.922 0.788</td><td>0.923 0.842</td><td>0.900 0.708</td><td>0.935 0.858</td><td>0.917 0.776</td><td>0.917 0.755</td><td>0.895 0.572</td><td>0.930 0.798</td><td>0.912 0.668</td><td>0.925 0.846</td><td>0.710</td><td>0.935</td><td>0.917</td></tr><tr><td>AgentAttack [15] AGAS (Ours)</td><td>0.392</td><td>0.300</td><td>0.355</td><td>0.325</td><td>0.382</td><td>0.285 0.342</td><td>0.311</td><td>0.372</td><td>0.270</td><td>0.330</td><td>0.297</td><td>0.365</td><td>0.260</td><td>0.320</td><td>0.287</td><td>0.370</td><td>0.275</td><td>0.860 0.332</td><td>0.779 0.301</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></table>

![](images/d2759338da6ded0291a35e1c649ee2b0fe8ccac226d60ab74736fe495f14b693.jpg)

![](images/bce7c076ab5ce599c2e7f33d03d5266411ebb11fbfd921c08b9bf25192e77d97.jpg)

Fig. 3: t-SNE on ML-100K. Left: red = distance to target, black = distance to ground-truth items (smaller is stealthier). Right: clean (blue) vs. fake (star) user aggregate histories on poisoned victims. More scattered stars indicate better stealth.  
![](images/548ac7f5f7ad0753af2ea37da6fec84cad0b8de2e87b8987fe59f6863e3294ba.jpg)  
Fig. 4: Summary of Head/Mid attack performance across representative victims: GMF (MF), LightGCN (graph convolution), and XSimGCL (graph contrastive).

![](images/10a3be68a07d5304141c704eb94ac8bc8e00eaffe22ca523084f498c12ebe4a0.jpg)  
Fig. 5: Benign Rec@10 on three representative victims. The gray segment shows the gap to NoneAttack. Higher is better.

![](images/a6b11c6ce8d2537204cf93c2a6a7f89c24904306ff433cedc010aaae3c30ac8e.jpg)  
Fig. 6: Strategy ablation heatmap on four datasets. Columns correspond to three representative victims from MF, graph convolution, and graph contrastive families.

Feedback Protocol. AGAS follows an offline-transfer protocol. During attack generation, the Coordinator observes rank movement only through recommendation lists returned to probe workers, typically PR/CA workers that have not rated the target. The evaluated victims are then trained from scratch on the clean matrix and on the poisoned matrix ${ \bf { R } } ^ { \prime } = [ { \bf { R } } ; \tilde { \bf { R } } ]$ (Eq. 4). AGAS never queries evaluated victims using $\mathcal { U } _ { \mathrm { t e s t } }$

![](images/7ccd5ab27c1471e73ee99309a1c1a0c9c1c02d45ac749e9cb8c165c1504f041b.jpg)  
Fig. 7: Component ablation across backbone size classes on four datasets. Large, Medium, and Small denote the size of the underlying language model.

TABLE III: Backbone ablation grouped into size tiers, following the taxonomy of [49], [50]. Values are mean±95% CI over 5 runs $( \times 1 0 ^ { 3 } )$ ). Best / second within each model block.
<table><tr><td>Model</td><td>Method</td><td>HR@10</td><td>NDCG@10</td><td>Rec@50</td></tr><tr><td colspan="5">Frontier (closed-source)</td></tr><tr><td rowspan="3">Claude Opus 4.5</td><td>AgentSA</td><td>34.4±1.6</td><td>14.5±1.4</td><td>82.8±1.9</td></tr><tr><td>AgentAttack</td><td>35.2±1.6</td><td>14.9±1.4</td><td>84.3±1.9</td></tr><tr><td>AGAS</td><td>41.5±0.5</td><td>17.6±0.3</td><td>95.0±1.1</td></tr><tr><td rowspan="3">GPT-5.1</td><td>AgentSA</td><td>34.5±1.5</td><td>14.6±1.3</td><td>82.6±1.9</td></tr><tr><td>AgentAttack</td><td>33.8±1.6</td><td>14.3±1.4</td><td>81.4±2.0</td></tr><tr><td>AGAS</td><td>40.6±0.5</td><td>17.2±0.3</td><td>92.8±1.1</td></tr><tr><td colspan="5">Large (≥ 50B)</td></tr><tr><td rowspan="3">Llama 3.3 70B</td><td>AgentSA</td><td>30.6±1.7</td><td>12.9±1.5</td><td>74.6±2.0</td></tr><tr><td>AgentAttack</td><td>31.2±1.7</td><td>13.2±1.5</td><td>75.9±2.0</td></tr><tr><td>AGAS</td><td>37.0±0.6</td><td>15.7±0.4</td><td>84.8±1.3</td></tr><tr><td rowspan="3">DeepSeek-V3</td><td>AgentSA</td><td>33.4±1.5</td><td>14.0±1.3</td><td>80.2±1.9</td></tr><tr><td>AgentAttack</td><td>33.0±1.6</td><td>13.9±1.4</td><td>79.4±2.0</td></tr><tr><td>AGAS</td><td>39.5±0.5</td><td>16.7±0.3</td><td>90.5±1.1</td></tr><tr><td colspan="5">Medium (∼ 10–30B)</td></tr><tr><td rowspan="3">Phi-4 (14B)</td><td>AgentSA</td><td>26.5±1.8</td><td>11.4±1.5</td><td>62.4±2.0</td></tr><tr><td>AgentAttack</td><td>27.0±1.8</td><td>11.1±1.5</td><td>61.2±2.0</td></tr><tr><td>AGAS</td><td>31.6±0.7</td><td>13.3±0.5</td><td>71.8±1.6</td></tr><tr><td colspan="5">Small (&lt; 10B)</td></tr><tr><td rowspan="3">Gemma 2 2B</td><td>AgentSA</td><td>17.8±2.0</td><td>7.5±1.7</td><td>41.2±2.2</td></tr><tr><td>AgentAttack</td><td>18.6±2.0</td><td>7.9±1.7</td><td>42.8±2.2</td></tr><tr><td>AGAS</td><td>22.4±1.0</td><td>9.4±0.6</td><td>51.0±2.0</td></tr></table>

## B. Performance

To address RQ1 (Performance), we evaluate AGAS against baselines on diverse datasets, victim models, and metrics under the matched protocol of §V. Table I shows that AGAS achieves the best result in every Tail-regime (Unpopular Items, Fig. 1) setting, with sizable gains over the strongest baseline across matrix factorization, graph convolution, and graph contrastive victim families. It stays effective without attacking aggressively in every round, since it may probe the victim early, slow down as suppression signals increase, or pause when the campaign becomes risky. Such non-myopic behavior is beneficial: temporarily reducing intensity preserves workers, avoids unstable promotion patterns, and produces more reliable long-term target movement. On the Head/Mid regimes (Popular Items, Fig. 1), Fig. 4 shows AGAS dominant on every axis. AgentAttack and AgentSA are the closest baselines, but the gap remains consistent across victim types and popularity regimes, indicating that AGAS is not tied to one target regime or one victim family.

## C. Stealthiness

To address RQ2 (Stealthiness), we examine whether AGAS promotes targets while preserving normal preference structure.

![](images/2737f4cb923cc990d8027c33b79658c1fd3b6016a0e922ed58193c667caf4d04.jpg)  
Fig. 8: Round-wise trade-off between HR@10 and runtime (minutes). Error bars show 95% CI.

![](images/eb0d2d4aefe5c9df02f02a27b1ebbc72943e8d45243c69aad87360a643963cbd.jpg)  
Fig. 9: Bars show HR@10 on the left axis, and lines show total runtime (minutes) on the right axis.

Fig. 3 reports two t-SNE views: the left measures per-user preference preservation, where the target should move closer to the user without pulling the user away from ground-truth items; the right measures global distributional camouflage, where fake histories should overlap with benign users rather than form separate clusters. AGAS is the only method that keeps both target and ground-truth distances small, showing that it promotes the target while preserving the user’s original preference region. It also spreads fake histories more diffusely within the benign embedding cloud, whereas other methods form more concentrated fake clusters. This stealth behavior comes from AGAS’s round-level adaptation and role-switching, which prevent fake users from collapsing into repetitive attack patterns.

## D. Detector

To address RQ3 (Detector), we evaluate AGAS against five representative shilling detectors: BaseDetect [44], DHAGCN [45], PCASelectUsers [46], and two group-level detectors, GAGE [47] and MD-CBA [48]. Table II reports Accuracy, Recall, Precision, and F1, where lower values mean the attack is harder to detect. Non-group detectors identify PoisonRec and AgentSA through profile statistics, graph neighborhoods, and PCA-aligned outliers, while group detectors capture cross-user overlap in AgentSA’s promptdriven histories. AgentAttack reduces duplicate-like behavior by mixing several attack families, but pre-trained templates still leave residual group signatures. AGAS instead lowers all metrics across all five detectors by distributing payload actions across roles and rounds. PR and CA build benign-looking context anchored to receptive benign users (§V-C), SN is used only by cleaner workers, and IN breaks synchronized activity, so both per-profile outliers and group-overlap signatures become weaker. Fig. 5 further shows that AGAS stays closest to the NoneAttack Rec@10 ceiling across all victims, preserving benign Top-10 quality while promoting the target. Together with §V-B, these results suggest that current detectors struggle with adaptive agentic campaigns.

![](images/4beb43b9109a498f0a429246fc7d4edb21f22d1c8b2b98da1eda46aa9e865a71.jpg)  
Fig. 10: Left: bar chart (log scale) of the average number of LLM tokens until the target first enters the Top-10. Right: the corresponding wall-clock seconds for each dataset.

![](images/1b79e3326e4a8c56d5217cb101edbaf516a204ec1960c6b9be462e666b7d33fd.jpg)  
Fig. 11: Each panel reports performance as the injected fakeuser budget grows: 0.5–3.0% of the user population.

## E. Ablation

To address RQ4 (Ablation), we ablate AGAS at the strategy and backbone levels. In Fig. 6, removing Silent Slowdown, Profile Cleanup, Safe Replacement, or Main Attack causes the largest drops, so AGAS needs both strong promotion and risk control. Bridge Building is graph-specific, hurting LightGCN and LightCCF more than NeuMF because it supports twohop attack paths. Table III shows that stronger backbones improve the attack, yet AGAS stays best across all model blocks and metrics and degrades gracefully with smaller ones. Removing the Coordinator, Memory, or Signals is sharpest on Gemma 2 2B (Fig. 7): for a small language model with small reasoning capacity, AGAS may have to lean more heavily on coordination, persistent memory, and explicit feedback.

## F. Efficiency

To address RQ5 (Efficiency), we evaluate round-wise tradeoffs, runtime, and fake-user budget scaling. HR@10 keeps rising with more rounds (Fig. 8). We set 18 rounds as the AGAS budget because later rounds become increasingly costly, adding wall-clock time and slightly wider 95% confidence intervals. This horizon also gives the Coordinator time to adjust roles before stronger promotion, making AGAS appear more normal (§V-C) and harder to detect (§V-D). Fig. 9 and Fig. 10 show that AGAS reaches the highest HR@10 at modest cost, about 24 minutes and tens of thousands of tokens to push the target into the Top-10, whereas AgentAttack needs millions of tokens and other methods hours of training. AGAS fits no GAN, solves no bi-level objective, retrains no pertarget surrogate, and pre-generates no profile library. Its cost is one Coordinator decision plus short worker calls per round, so runtime and token use grow approximately linearly with rounds and fake users. Gains stay steady as the budget grows from 0.5% to 3.0%, with smaller gains on Amazon indicating that harder datasets require more attack pressure (Fig. 11). AGAS is thus budget-aware: rounds, tokens, and fake-user portion can be tuned to trade cost for HR@10, making it a cost-aware benchmark framework for RecSys robustness.

## VI. CONCLUSION

This paper introduced AGAS, a black-box agentic framework that orchestrates coordinated group shilling attacks through dynamic role switching and multi-stage campaigns. The Coordinator manages attack escalation, slowdown, pausing, replacement, and round-level adaptation across a group of fake-user workers. Across diverse datasets and CF-based victim models, AGAS achieves stronger target promotion than prior baselines while preserving more realistic user behavior and benign recommendation quality under matched evaluation budgets. It runs end-to-end without costly fine-tuning or repeated surrogate retraining, reaches strong attacks at modest runtime and token cost, and transfers across victim families, making it both a practical threat model and a reusable benchmark for stress-testing recommender robustness. Future work should extend AGAS to more complex victims such as multimodal recommenders, or LLM-empowered recommenders. As Agentic Web capabilities spread, recommender systems must defend against attacks that adapt to their defenses at scale.

## ACKNOWLEDGMENT

This work was partially supported by the Australian Research Council through the Discovery Project (Grant Nos. DP260100326 and DP240101108), the Linkage Projects (Grant Nos. LP230200892 and LP250200778), and the DE-CRA Project (Grant No. DE260100673).

## REFERENCES

[1] Y. Koren, R. Bell, and C. Volinsky, “Matrix factorization techniques for recommender systems,” IEEE Computer, 2009.

[2] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural collaborative filtering,” in WWW, 2017.

[3] X. Wang, X. He, M. Wang, F. Feng, and T.-S. Chua, “Neural graph collaborative filtering,” in SIGIR, 2019.

[4] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, “Lightgcn: Simplifying and powering graph convolution network for recommendation,” in SIGIR, 2020.

[5] M. P. O’Mahony, N. J. Hurley, and G. C. M. Silvestre, “Recommender systems: attack types and strategies,” in AAAI, 2005.

[6] C. Lin, S. Chen, H. Li, Y. Xiao, L. Li, and Q. Yang, “Attacking recommender systems with augmented user profiles,” in CIKM, 2020.

[7] F. Wu, M. Gao, J. Yu, Z. Wang, K. Liu, and X. Wang, “Ready for emerging threats to recommender systems? a graph convolution-based generative shilling attack,” Information Sciences, 2021.

[8] B. Li, Y. Wang, A. Singh, and Y. Vorobeychik, “Data poisoning attacks on factorization-based collaborative filtering,” in NeurIPS, 2016.

[9] T. N. Thanh, N. D. K. Quach, T. T. Nguyen, T. T. Huynh, V. H. Vu, P. L. Nguyen, J. Jo, and Q. V. H. Nguyen, “Poisoning GNN-based recommender systems with generative surrogate-based attacks,” TOIS, 2023.

[10] Z. Wang, J. Yu, M. Gao, H. Yin, B. Cui, and S. Sadiq, “Unveiling vulnerabilities of contrastive recommender systems to poisoning attacks,” in KDD, 2024.

[11] Y. Zhao, T. Chen, J. Yu, K. Zheng, L. Cui, and H. Yin, “Diversityaware dual-promotion poisoning attack on sequential recommendation,” in SIGIR, 2025.

[12] J. Song, Z. Li, Z. Hu, Y. Wu, Z. Li, J. Li, and J. Gao, “Poisonrec: An adaptive data poisoning framework for attacking black-box recommender systems,” in ICDE, 2020.

[13] W. Fan, T. Derr, X. Zhao, Y. Ma, H. Liu, J. Wang, J. Tang, and Q. Li, “Attacking black-box recommendations via copying cross-domain user profiles,” in ICDE, 2021.

[14] S. Gu, J. Liu, D. Li, G. Zhang, M. Han, H. Gu, P. Zhang, N. Gu, L. Shang, and T. Lu, “LLM agent-based shilling attack on recommender systems,” in WSDM, 2026.

[15] C. Li, Z. Wang, H. Ma, W. Li, H. Wu, Y. Song, and M. Gao, “Agentattack: LLM agents for multi-strategy shilling attacks in recommenders,” Expert Systems with Applications, 2026.

[16] L.-b. Ning, S. Wang, W. Fan, Q. Li, X. Xu, H. Chen, and F. Huang, “Cheatagent: Attacking llm-empowered recommender systems via llm agent,” in KDD, 2024.

[17] S. Yang, Z. Hu, X. Li, C. Wang, T. Yu, X. Xu, L. Zhu, and L. Yao, “Drunkagent: Stealthy memory corruption in llm-powered recommender agents,” in WWW, 2026.

[18] S. Wang, W. Fan, X.-Y. Wei, X. Mei, S. Lin, and Q. Li, “Multi-agent attacks for black-box social recommendations,” TOIS, 2024.

[19] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” CACM, 2020.

[20] S. K. Lam and J. Riedl, “Shilling recommender systems for fun and profit,” in WWW, 2004.

[21] Y. Yang, M. Ma, Y. Huang, H. Chai, C. Gong, H. Geng, Y. Zhou, Y. Wen, M. Fang, M. Chen, S. Gu, M. Jin, C. Spanos, Y. Yang, P. Abbeel, D. Song, W. Zhang, and J. Wang, “Agentic web: Weaving the next web with AI agents,” arXiv:2507.21206, 2025.

[22] T. Berners-Lee and S. Witt, This Is for Everyone: The Unfinished Story of the World Wide Web. Farrar, Straus and Giroux, 2025.

[23] I. Arghire, “Google DeepMind researchers map web attacks against AI agents,” SecurityWeek, 2026.

[24] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J.-R. Wen, “A survey on large language model based autonomous agents,” Frontiers of Computer Science, 2024.

[25] K. T. Pham, T. H. Nguyen, J. Jo, Q. V. H. Nguyen, and T. T. Nguyen, “Multilingual text-to-sql: Benchmarking the limits of language models with collaborative language agents,” in Australasian Database Conference. Springer, 2025.

[26] T. Pham, T. T. Nguyen, V. Huynh, H. Yin, and Q. V. H. Nguyen, “An efficient and effective evaluator for text2sql models on unseen and unlabeled data,” in ICDE. IEEE, 2026.

[27] T. Pham, V. Huynh, H. Yin, Q. V. H. Nguyen, and T. T. Nguyen, “Learning to evaluate: Cost-effective model evaluation on unlabeled data with meta-learning,” in SIGKDD, 2026.

[28] T. T. Nguyen, T. Pham, V. Huynh, M. H. Nguyen, B. Vo, L. Qu, J. Li, H. Yin, and Q. V. H. Nguyen, “A survey of social network simulation in the llm era: From classical models to generative agents,” Knowledge-Based Systems, 2026.

[29] K. M. Le, T. Pham, T. Quan, and A. T. Luu, “Lampat: Low-rank adaption for multilingual paraphrasing using adversarial training,” AAAI, 2024.

[30] T. Pham, K. Le, and A. T. Luu, “UniBridge: A unified approach to crosslingual transfer learning for low-resource languages,” in ACL, 2024.

[31] F. Ricci, L. Rokach, B. Shapira, and P. B. Kantor, Eds., Recommender Systems Handbook. Springer, 2011.

[32] D. D. Lee and H. S. Seung, “Algorithms for non-negative matrix factorization,” in NeurIPS, 2000.

[33] J. Yu, H. Yin, X. Xia, T. Chen, L. Cui, and Q. V. H. Nguyen, “Are graph augmentations necessary? simple graph contrastive learning for recommendation,” in SIGIR, 2022.

[34] J. Yu, X. Xia, T. Chen, L. Cui, Q. V. H. Nguyen, and H. Yin, “Xsimgcl: Towards extremely simple graph contrastive learning for recommendation,” TKDE, 2023.

[35] Y. Zhang, Y. Zhang, L. Sang, and V. S. Sheng, “Simplify to the limit! embedding-less graph collaborative filtering for recommender systems,” TOIS, 2024.

[36] Y. Zhang, Y. Zhang, Y. Zhang, L. Sang, and Y. Yang, “Unveiling contrastive learning’s capability of neighborhood aggregation for collaborative filtering,” in SIGIR, 2025.

[37] H. Ma, Z. Gao, C. Seng, and L. Mao, “Stealthy attack on graph recommendation system,” Expert Systems with Applications, 2024.

[38] L. Zhang, N. Chen, H. Ren, Y. Zhou, L. Wu, and Z. Zeng, “Adversarial injection attacks against recommender systems,” in ACM Conference on Recommender Systems, 2024.

[39] F. Zhang, “Analysis of bandwagon and average hybrid attack model against trust-based recommender system,” in Fifth International Conference on Management of e-Commerce and e-Government, 2011.

[40] J. Yu, M. Gao, W. Rong, W. Li, Q. Xiong, and J. Wen, “Hybrid attacks on model-based social recommender systems,” Physica A: Statistical Mechanics and its Applications, 2017.

[41] Y. Zhang, L. Yao, C. Wang, X. Xu, and L. Zhu, “Incorporated modelagnostic profile injection attacks on recommender systems,” in ICDM, 2023.

[42] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, “Bpr: Bayesian personalized ranking from implicit feedback,” in UAI, 2009.

[43] S. Guo, T. Bai, and W. Deng, “Targeted shilling attacks on GNN-based recommender systems,” in CIKM, 2023.

[44] C. A. Williams, B. Mobasher, and R. Burke, “Defending recommender systems: detection of profile injection attacks,” Service Oriented Computing and Applications, 2007.

[45] Y. Hao, G. Meng, J. Wang, and C. Zong, “A detection method for hybrid attacks in recommender systems,” Information Systems, 2023.

[46] B. Mehta and W. Nejdl, “Unsupervised strategies for shilling detection and robust collaborative filtering,” User Modeling and User-Adapted Interaction, 2009.

[47] F. Zhang, Y. Qu, Y. Xu, and S. Wang, “Graph embedding-based approach for detecting group shilling attacks in collaborative recommender systems,” Knowledge-Based Systems, 2020.

[48] Y. Xu, P. Zhang, H. Yu, and F. Zhang, “Detecting group shilling attacks in recommender systems based on user multi-dimensional features and collusive behaviour analysis,” The Computer Journal, 2023.

[49] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, Y. Du, C. Yang, Y. Chen, Z. Chen, J. Jiang, R. Ren, Y. Li, X. Tang, Z. Liu, P. Liu, J.-Y. Nie, and J.-R. Wen, “A survey of large language models,” arXiv:2303.18223, 2023.

[50] F. Wang, Z. Zhang, X. Zhang, Z. Wu, T. Mo, Q. Lu, W. Wang, R. Li, J. Xu, X. Tang, Q. He, Y. Ma, M. Huang, and S. Wang, “A comprehensive survey of small language models in the era of large language models: Techniques, enhancements, applications, collaboration with llms, and trustworthiness,” TIST, 2025.

[51] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in ICLR, 2023.

[52] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language agents with verbal reinforcement learning,” NeurIPS, 2023.

[53] F. M. Harper and J. A. Konstan, “The movielens datasets: History and context,” TiiS, 2015.

[54] D. Kotkov, A. Medlar, U. R. Satyal, A. Maslov, M. Neovius, and D. Glowacka, “Revisiting the tag relevance prediction problem,” in SIGIR, 2021.

[55] J. Bennett and S. Lanning, “The netflix prize,” in KDD Cup and Workshop, 2007.

[56] Z. Wang, H. Liu, Y. Du, Z. Wu, and X. Zhang, “Unified embedding model over heterogeneous information network for personalized recommendation,” in IJCAI, 2019.

[57] J. Ni, J. Li, and J. McAuley, “Amazon review data (2018),” 2018.