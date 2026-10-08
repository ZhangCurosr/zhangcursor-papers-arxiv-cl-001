# Decoupling Exploration from Optimization in RLVR

Saif Punjwani<sup>1,∗</sup> Micah Goldblum<sup>1,2,∗</sup>

Columbia University

## Abstract

Modern language models undergo reinforcement learning with verifiable rewards (RLVR) on top of already-trained checkpoints. A key promise of RLVR is the discovery of new reasoning strategies. In principle, a model can sample novel ideas absent from its prior training data. In practice, however, augmenting RLVR with strong novelty incentives has seen limited success and can degrade model quality. Because verifiable rewards supervise only a narrow slice of the model’s knowledge and behavior, such degradations are difficult to recover from. Instead, we decouple exploration from optimization in a framework we call Exploration-Distillation (ExpDis). We train one or more explorer policies with a novelty bonus in the reward, filter their trajectories for correctness and quality, and distill them into a separate student policy. The student policy is then trained without a novelty bonus. We repeat the above procedure for several rounds, alternating between exploration and optimization. This decoupling allows us to aggressively scale exploration without degrading the student policy. Across seven mathematical reasoning benchmarks and two model families, ExpDis outperforms DAPO at the same wall-clock budget. Moreover, we observe improved pass@k scaling, indicating that ExpDis produces models that generate more diverse correct solutions.<sup>3</sup>

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has become the dominant recipe for improving the reasoning abilities of language models [19, 15, 33, 26, 44]. RLVR samples solutions to problems, scores each solution with a programmatic verifier, and shifts probability toward those that pass. By iteratively sampling new solutions and reinforcing successful ones, RLVR can in principle discover and then reinforce new solutions that were absent from the model’s training data. In practice, however, it can only reinforce what the model samples, so discovery depends on exploration.

A natural way to boost exploration is to add an explicit novelty bonus to the reward, following a long line of intrinsic-motivation methods in RL [30, 2, 3, 10]. When the model explores new solution strategies, some of them may succeed but often most of them fail, leading to a temporarily degraded model as we now are left with a model that mostly generates failed solutions. Though our initial policy is a pretrained LLM with vast knowledge including virtually all of Wikipedia, the reward supervises only a tiny slice of its knowledge. This means an LLM could forget most of Wikipedia and still achieve high reward. The reward therefore cannot detect damage that exploration causes outside that slice, let alone repair such degradations. We argue that this failure is not a property of novelty as a signal, but of where the novelty bonus is applied.

We decouple exploration and optimization by giving them separate model instances, both trained from the same base checkpoint. An explorer policy explores aggressively under a novelty bonus whose weight λ sets how strongly it is pushed toward unfamiliar solutions relative to correctness. A separate student policy never sees a novelty reward. It learns from the explorer only through filtered trajectories, and is then trained with standard RLVR. The two models communicate through filtered trajectories rather than parameters, so the student policy absorbs the explorer’s discoveries without inheriting its degradations induced by exploration. We call this framework Exploration-Distillation.

![](images/7efb191477907dbe5c75797ade044e28dfe5b9e69208f17c694c0dee131840e2.jpg)  
correct region correct incorrect filtered out

(a) Math reasoning performance (Qwen3-4B)  
![](images/654e2d2fa0685f5ded80f20d399b748a583fb429e4ab13e76b567cbc098666b7.jpg)

(b) Knowledge not supervised by the reward can degrade  
![](images/075589dba505f7540bda5d3a08fffc74638bbac9c9759ef2638a01ddaa15346c.jpg)  
Figure 1: Decoupling exploration from optimization improves math accuracy without degrading prior capabilities. Left: Rather than coupling exploration and optimization in a single RLVR run, ExpDis assigns these roles to separate policies. We first train an explorer µ with a novelty bonus in its reward, then distill a diverse pool of its reasoning traces to cold start a student, and finally train the student with standard RLVR using only a correctness reward. As depicted in the figure, this allows us to explore a diverse set of reasoning strategies and preserve this diversity in our student policy. (a) ExpDis outperforms DAPO in both average accuracy and pass@k, even when DAPO is trained for 4× as long (Qwen3-4B, mean pass@k over five math benchmarks). (b) Adding a novelty bonus (RND) directly to the DAPO reward degrades performance on prior capabilities relative to the base model. Because the reward supervises only a narrow slice of the model’s knowledge (here, mathematical reasoning), achieving high reward does not prevent general capability degradation. By decoupling the explorer from the student policy, we avoid this degradation.

Decoupling makes exploration safe to scale aggressively, since the student policy never receives the novelty bonus and the filter keeps failed trajectories out of its training data. We can scale breadth by running several explorers in parallel and distilling their pooled trajectories into one student policy. We can also scale depth by repeating the pipeline for multiple rounds that each start from the previous student. We scale both of these axes independently and find that their composition enables sustained performance gains even when all experiments are wall-clock matched.

We evaluate ExpDis on mathematical reasoning benchmarks with Qwen3-1.7B, Qwen3-4B, and Ministral-3-3B-Instruct-2512 as base models. At the same amount of compute, ExpDis outperforms DAPO in both pass@1 and pass@64. Furthermore, spreading this fixed budget over multiple parallel explorers and several rounds of the ExpDis procedure yields further gains. Adding a novelty bonus directly to DAPO, in contrast, increases diversity but degrades performance on general knowledge benchmarks. In ExpDis, the explorer can be pushed to far higher diversity without harming the student, which improves in both diversity and accuracy and avoids degradation.

In principle, ExpDis is compatible with any exploration mechanism, since the student learns only from the explorer’s filtered trajectories. Since the explorer is never deployed, it can also use mechanisms too aggressive to apply to the deployed model directly. Exploration methods that would destabilize standard RLVR may therefore become viable under ExpDis.

## 2 Related Work

Exploration in LLM reasoning. A number of works ask whether standard RLVR algorithms discover new solutions or only sharpen a model’s pre-RL generative distribution. Shao et al. [33] observe that RL improves majority-vote accuracy but not pass@k, and Chen et al. [5] find that base models overtake their RLVR-trained counterparts at large k, suggesting that RLVR narrows the set of problems a model can solve. Others find that RL can expand this set through prolonged training [25], or that the model learns new skills by composing existing ones [45]. A related line of work studies entropy collapse: policy entropy falls sharply early in training [7], and the gains from RLVR rely on a small set of high-entropy tokens to allow sampling diverse solutions [40]. Many methods counteract this entropy collapse during training. For example, DAPO [44] raises the upper clipping bound so that low-probability tokens with positive advantage can grow faster, whereas He et al. [16] directly regulate the policy’s entropy. Other approaches add exploration or novelty bonuses to the reward [36, 11, 38, 21]. These methods draw on a long line of exploration bonuses in classical RL, including count-based bonuses [2], intrinsic curiosity [30], and random network distillation [3]. ExpDis is compatible with a broad spectrum of exploration bonuses, and we try plugging several example bonuses into our framework, observing benefits of exploration in all cases.

Reasoning trace distillation. A classical precursor that trains a policy to imitate a separate process that finds better solutions is expert iteration [1], in which a tree-search expert generates improved moves and a neural apprentice learns to imitate them. Today, distilling reasoning traces from a stronger model is standard practice: DeepSeek-R1 [15] is distilled into smaller and faster open models, and several works study how to curate reasoning traces to instill reasoning in fine-tuned models [43, 13]. When no stronger model is available, a model can instead learn from its own filtered samples, as in STaR [46], ReST [14], and ReST<sup>EM</sup> [35]. DeepSeek-R1 fine-tunes its base model on correct traces rejection-sampled from an RL checkpoint before running RL again [15], and multi-teacher on-policy distillation methods [28] distill separately trained specialist models into one model. We similarly distill trajectories from one model into another, but we train our explorer policies from the same base model as our student with the distinct goal of exploration.

Decoupling exploration in RL. Separating exploration from exploitation has a long history in RL: E<sup>3</sup> [18] keeps separate exploration and exploitation policies and explicitly switches between them, ¸Sim¸sek and Barto [34] explore with a policy rewarded for improving a separate task value function that never sees this reward, and Go-Explore [9] first finds high-reward trajectories and then turns them into a robust policy via imitation learning. In reinforcement learning for LLMs, R<sup>2</sup>PO [39] trains a residual logit head to diversify within-group outcomes, uses it to sample rollouts for the policy’s GRPO updates, and discards it at inference. In concurrent work, ICE [8] explores with instruction-conditioned prompts and self-distills correct rollouts into the unconditioned policy. In both, the exploring policy and the student share weights. We instead train the explorer as a separate network and pass only filtered trajectories to the student. This design choice insulates the student from exploration pressures and thus avoids degrading the student.

## 3 Method

Exploration-Distillation trains an explorer policy µ with a novelty bonus to generate diverse trajectories, filters them for quality and correctness, and distills the resulting trajectories into a student policy $\pi _ { 1 }$ . We present a single iteration below and describe how we scale it in Section 3.1 and Section 3.2. We study the importance of each stage in Appendix E.

Explorer policy training. We initialize an explorer policy µ from the base policy $\pi _ { 0 }$ with DAPO [44] and augment its reward with a novelty reward $r _ { \mathrm { n o v e l t y } } \doteq \bar { 0 }$

$$
r _ { \mathrm { e x p l o r e r } } ( \tau ) ~ = ~ r _ { \mathrm { c o r r e c t } } ( \tau ) ~ + ~ \lambda \cdot r _ { \mathrm { n o v e l t y } } ( \tau ) \cdot { \bf 1 } [ r _ { \mathrm { c o r r e c t } } ( \tau ) = + 1 ] ~ + ~ r _ { \mathrm { o v e r l o n g } } ( \tau ) ,\tag{1}
$$

where $r _ { \mathrm { c o r r e c t } } \in \{ - 1 , + 1 \}$ and $\lambda \geq 0$ . As introduced in $\mathrm { D A P O } , r _ { \mathrm { o v e r l o n g } } \in [ - 1 , 0 ]$ is a soft penalty on completions that approach the length budget. Apart from the novelty term, the explorer is identical to our DAPO baseline. This reward formulation ensures that every correct rollout receives a higher reward than any incorrect one for any value of λ. We find that applying the bonus to all rollouts leads to training instability as a model can achieve increased reward simply by generating novel garbage.

Rejection sampling. We now need a method for collecting trajectories generated by the explorer policy to be distilled into the student. We remove any incorrect trajectories, since the point of this rejection sampling step is to insulate the student from any degradations caused to the explorer by novelty pressure. A naive choice would be to take all correct explorer trajectories or a random subset and hand them off, but we find that filtering not only for correctness but also for quality strongly improves performance. A trajectory passes the filter only if (1) its final \boxed{} answer is correct, (2) it terminates within a fixed token budget, and (3) it contains no repetition loops.

![](images/c931b0d40c667ac31d586ea4d8febb9d36119d309834fc3f7817e8286e8ac307.jpg)  
Figure 2: ExpDis scales across parallel explorers and iterations. Within a round, several explorers are trained in parallel with a novelty bonus, and their traces are pooled to distill a single student. Across rounds, that student initializes both the next round’s explorers and the next round’s student, so every round starts from a stronger model. Each round trains on its own shard of the training prompts, and the novelty weight anneals from $\lambda _ { 1 } { = } 0 . 7 5$ to $\lambda _ { 4 } { = } 0 . 2 5$ . We study how to allocate a fixed RL budget across explorers and rounds in Section 5.

After this filtering, we keep only a single shortest resulting trajectory per problem. Problems that the explorer solves correctly many times therefore do not dominate the student’s training data. We cap the resulting set at 500 trajectories per round. In Appendix E, we ablate our quality filtering procedure.

Student policy training. We distill the base policy on the resulting trajectories simply by performing supervised finetuning (SFT) on those trajectories with the standard cross-entropy loss. Following SFT, we train the distilled model with DAPO and the standard correctness-only reward (i.e., no novelty bonus) to obtain the student policy $\pi _ { 1 }$ , which inherits none of the degradation from exploration pressure. The student both receives the benefit of exploration via distillation, which is off-policy, and also the benefit of on-policy training during its DAPO stage.

## 3.1 Parallel Explorer Policies

Rather than train a single explorer policy $\mu ,$ we can train K independent explorer policies and pool their trajectories before filtering and performing distillation. Compared to the single-explorer variant, training parallel explorers can increase the diversity of trajectories we curate for the student. This diversity increases coverage over solutions and boosts our probability of discovering successful ones.

We set the novelty bonus weight λ identically for all peer explorer policies, as we find little benefit from varying them. In our experiments, we divide compute among explorers, requiring no additional wall-clock time over a single explorer. The more compute a training run uses, the more parallel explorers we expect to be optimal.

## 3.2 Iterative Exploration-Distillation

We can scale the above method by performing many rounds of alternating exploration and distillation. After producing a student policy, we can use this student as a strong initialization for both the next explorer policy and subsequent student policy. Formally: to train student $\pi _ { t }$ , we first train one or more explorers $\mu$ from $\pi _ { t - 1 }$ , filter the trajectories, distill those trajectories onto $\pi _ { t - 1 }$ via SFT and then perform standard RLVR on the distilled checkpoint without an exploration bonus to produce $\pi _ { t }$

We partition the training prompts into R non-overlapping shards $\mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { R }$ of equal size. As shown in Figure 2, round r draws its training prompts from $\mathcal { D } _ { r }$ . Hence, each round pairs a stronger initialization with prompts it has not yet trained on.

Annealing novelty bonus. Because exploration is split across several rounds, we can set its strength per round through the novelty-bonus weight λ, which we anneal on a fixed schedule from 0.75 to 0.25 over four rounds. This schedule follows the common practice of decreasing exploration pressure as training progresses [37]. Early rounds, where correct trajectories are still undiscovered, get strong novelty pressure whereas later rounds get lower novelty pressure to consolidate what has been found.

## 4 Experimental Setup

Base models. We evaluate ExpDis on Qwen3-1.7B, Qwen3-4B [42], and Ministral-3-3B-Instruct-2512 [24]. The Qwen3 models are distilled from larger RL-trained Qwen3 models rather than trained with RL directly [42], whereas Ministral-3-3B-Instruct is instruction-tuned and undergoes online preference optimization [24]. We describe all sampling parameters in Appendix A.

Novelty bonus. In principle, ExpDis is compatible with any novelty bonus. In our primary experiments, we choose a Random Network Distillation (RND) bonus [3] as we find that this formulation is especially effective. Following Burda et al. [3], we train one small predictor MLP $f _ { \theta , \ell }$ per layer to match a frozen random target MLP $f _ { \ell } ^ { \star }$ on features extracted from the language model’s own hidden states, and use the prediction error as the novelty signal. We extract features from a fixed set of intermediate layers ${ \mathcal { L } } ,$ , mean-pool over completion tokens (the completion is encoded without the prompt), and average across layers:

$$
r _ { \mathrm { n o v e l t y } } ( \tau ) ~ = ~ { \frac { 1 } { | { \mathcal { L } } | } } { \sum _ { \ell \in { \mathcal { L } } } } { \frac { 1 } { \sqrt { d } } } \left\| f _ { \ell } ^ { \star } \big ( h _ { \ell } ( \tau ) \big ) - f _ { \theta , \ell } \big ( h _ { \ell } ( \tau ) \big ) \right\| _ { 2 } .\tag{2}
$$

Here $h _ { \ell } ( \tau )$ is the pooled layer-ℓ feature of trajectory τ and d is the output dimension of the predictor and target networks, so each layer contributes a root-mean-square prediction error. The predictor updates continuously on the same features that drive the bonus, while the target stays frozen for the entire run. In preliminary experiments, features from several intermediate layers gave a cleaner signal than last-layer features alone (layers and network sizes in Appendix $\mathbf { A } )$ . We ablate the choice of $r _ { \mathrm { n o v e l t y } }$ in Appendix C.1 and observe a consistent improvement over DAPO across bonuses.

Reward. We use a standard math verifier for $r _ { \mathrm { c o r r e c t } } .$ . In Appendix C, we sweep the weight λ on $r _ { \mathrm { n o v e l t y } }$ and find that even very large weights leave the student model outperforming the base model. We find that performance roughly peaks at λ=0.5 and conduct all experiments with this value of λ.

Training dataset. We sample random prompts without replacement from DAPO-Math-17K [44], a collection of 17K mathematical reasoning problems, until reaching a cap of 300 steps (discussed below). We perform additional experiments on the DeepScaleR [27] dataset in Appendix E.

Evaluation. We report performance on seven mathematical reasoning benchmarks: AIME24, AIME25, AIME26 [29], MATH500 [17, 22], AMC23 [29], Minerva-Math [20], and GSM8K [6]. Following the Qwen3 technical report [42], sample 64 completions per problem (32 for AMC23 and 8 for GSM8K) and report the unbiased pass@k estimator for $k \in \{ \bar { 1 } , 2 , \bar { 4 } , \dots , 6 4 \}$ . GSM8K is close to saturation for all three models, and AMC23 is close to saturation for the Qwen3 models, which leaves little room to compare methods. We therefore focus on the other five benchmarks and report their mean unless noted otherwise.

We additionally study the model’s degradation on a suite of general capability benchmarks: MMLU-Pro [41], MMLU-Redux [12], and GPQA-Diamond [32] for knowledge, ZebraLogic [23] for logical reasoning, and IFEval [47] for instruction following. This serves as a proxy for capabilities not directly supervised by our training dataset and reward to study the extent to which exploration pressures can degrade a model’s knowledge over time.

Baselines. We compare against three RLVR baselines with a standard correctness-only reward: GRPO [33], Dr. GRPO [26], and DAPO [44]. We find empirically that DAPO is the strongest of these baselines and design two extensions: (1) adding a novelty bonus to the reward with λ=0.5 to isolate the effect of decoupling exploration, and (2) running DAPO for 4× as many steps.

![](images/268e19a90e9f63414e25cf25e34ca8ad48ca2d9db5c1407cf92a63c44df176dd.jpg)

![](images/52e840f298b48aec121fb3a16a2d7da351f889c17ce9a2f55657e547d96f0404.jpg)

![](images/65bccd4b244ec6606a7c209986d9d6fd108f4406397da4001685cbb0abf1953b.jpg)  
Base DAPO DAPO (4 × steps) DAPO (+novelty) ExpDis (single-round) ExpDis  
Figure 3: ExpDis outperforms all baselines under the same RL compute budget. By partitioning this budget into alternating rounds of exploration and optimization (as described in Section 3.2), we can surpass DAPO and even a single round of ExpDis. We report full benchmark results in Appendix B.

Compute matching. We perform 300 steps of RL with 4 prompts per step, each having a group size of 16 and a 32,768 max generation limit. We also study the effect of training DAPO for 4× the number of steps to determine if prolonged training allows baselines to catch up [25].

ExpDis splits these updates between the two policies: 200 for the explorer and 100 for the student. When we scale across explorers or rounds, each share is divided evenly among them, while we keep the SFT stage of the student policy two epochs on at most 500 filtered explorer trajectories per round.

## 5 Results

Across all three models, ExpDis outperforms the DAPO baseline in mean accuracy on mathematical reasoning benchmarks (Figure 3). A single round of ExpDis already outperforms DAPO trained for 4× as many steps. By spreading the same compute over several rounds of alternating exploration and optimization (Section 3.2), we observe further gains. All ExpDis runs outperform this prolonged DAPO run in both average performance and pass@k up to 64 generations per problem.

![](images/fb0a8a10121f86c2efe138df69a79bd7662b31f513baaf3abbe2e9c78c1a02a0.jpg)

![](images/cf35d67de748fb270492ecf5b2085672c107716200a5afbb127058ea3ad8cc43.jpg)

![](images/30585670454f5e127f576ba3aa831619b5ff02b41fd60ec8e409fa8f8cfa052e.jpg)  
Figure 4: Under a fixed RL compute budget, how we allocate compute impacts downstream student policy performance. We vary the number of explorers and rounds under a fixed compute budget, and find complementary impacts of each scaling axis. Under a fixed budget, adding rounds improves pass@1 more than adding explorers, and combining both gives the largest gains in pass@1 and pass@64. All results are reported on Qwen3-1.7B.

Scaling exploration. Decoupling lets us scale exploration along two axes: running several explorers in parallel within a round and repeating the ExpDis procedure over several rounds. In Figure 4, we vary both on Qwen3-1.7B while holding the total budget fixed at 300 RL updates. We observe that depth contributes more to pass@1 than breadth, though a combination of both improves pass@k over either. One interpretation is that the two axes play different roles: each additional round reinforces a progressively stronger student, whereas additional explorers widen the set of problems with correct trajectories and the variety of trajectories that get distilled into the student.

At our budget, gains saturate and begin to show signs of regression after three explorers and four rounds. We expect this optimum to shift as the budget grows: with a higher compute budget, longer rounds and larger batch sizes per explorer will have diminishing marginal returns, and we expect the optimal number of rounds and explorers to increase accordingly. Note that we vary breadth and depth at a fixed compute budget; explorers run in parallel, and spreading compute across more explorers adds no wall-clock time. In principle, this parallelization allows practitioners with vast compute access to ramp up the number of explorers at no time cost.

Diversity analysis. RLVR is known to collapse entropy and sharpen the sampling distribution as training proceeds [7, 5]. Counteracting this collapse within a single model requires a delicate balance, since the same model must both explore diverse reasoning strategies but must simultaneously be optimized for correctness and avoiding incorrect solutions. Adding a novelty bonus to DAPO illustrates this trade-off: it increases both lexical and answer diversity relative to standard DAPO, yet slightly reduces mathematical reasoning performance.

![](images/b439a4440617db3ba535f1e4dfed142054fb21fdfdf727d4922af8049cd7ff1d.jpg)

![](images/e7308a566924dad225d14dd3a6c5d600a62ece78360dfef44f5d5b91a0d2eb9a.jpg)

![](images/6e5d0adea1af42823810695bc5e66f2f389086dfd6f647bcd1505a75e7e1f3c7.jpg)

![](images/beab24f835428b51a0193354887a89ceda9fffaed65535ac4ce5da87dc44048d.jpg)  
Base DAPO DAPO (4× steps) DAPO (+novelty, λ = 0.5) ExpDis (single-explorer) ExpDis

Figure 5: Explorer policy can be driven to more diversity without harming the student. Left: We stud how explorer policy diversity impacts the downstream student in ExpDis. In general, we find that we can push the diversity of the explorer policy without corrupting the downstream student policy. Since we filter for correctness and quality before distilling, a moderate increase in explorer diversity (for example, lexical diversity) brought on by tuning the novelty bonus weight λ does not negatively impact the student policy. Similarly, we can increase the explorer token entropy far beyond that of DAPO (+novelty) without incurring downstream reasoning degradation. Right: We find that ExpDis generations contain more semantically diverse mathematical reasoning strategies. We define each diversity metric in Appendix D.

Diversity and performance are not always complementary; a model may, for example, learn to rephrase the same solution strategy, thereby increasing lexical diversity or entropy without improving pass@k. As shown in Figure 5, DAPO + novelty exhibits higher lexical diversity and token entropy, but regresses on mathematical reasoning compared to the DAPO model. Broadly, we find that we can push the explorer toward far greater diversity than a single model could tolerate while maintaining or improving on downstream accuracy.

Stage ablations. We run ablations on Qwen3-1.7B to understand whether ExpDis reduces to either (1) distilling unfiltered explorer trajectories and then running RL, or (2) distilling filtered trajectories without running RL afterward. We find that both variants score slightly below DAPO on AIME24, whereas the full procedure scores above it (Appendix E). Furthermore, keeping only the shortest valid trajectory per problem (i.e., quality filtering) outperforms keeping every correct trajectory. Finally, we replace RND with a kNN-based or an elliptical novelty bonus (the latter adapted from Tuyls et al. [38]) and find that mean accuracy drops by at most 1.6 points, and both variants still outperform DAPO (Appendix C.1).

Novelty bonus degradation. A verifiable reward, typically applied to math and code tasks, directly supervises only a narrow slice of a model’s capabilities. In our experiments, DAPO with the RND novelty bonus degrades below the base model on prior capability benchmarks (Section 4) despite improving on math (right). Conversely, ExpDis gets the benefits of exploration without degradation. We report results for each general capability benchmark in Figure 12.

While the drops we observe are modest at our scale, we speculate that larger-scale RLVR runs that apply exploration bonuses directly to the trained model may accumulate substantially larger degradations over time, whereas decoupling avoids this risk of corrupting the deployed model.

![](images/41c0a75c87542e3f29b8135c491f3e4a88d519d99549105c358e12064efd481e.jpg)  
Figure 6: Adding RND novelty bonus to DAPO degrades prior capabilities.

Policy evolution. We study how the explorer and student policies evolve across several iterations of ExpDis. Figure 7 suggests a simple picture: the explorers repeatedly expand the space of reasoning strategies available to the student, and the student keeps a growing share of that expansion while leaving its costs (reduced pass@k and reasoning faithfulness) behind. Within each round, the novelty bonus pushes the explorer entropy and semantic diversity well above the previous student, and subsequent distillation on the shortest correct trajectories contracts them again. We find that both the student’s diversity and accuracy ratchet upward together across subsequent rounds.

![](images/c6d8973e41afdd4176af7c26acc2d3eb1f5694c5b14d860d0d9b3c424e2b20bc.jpg)  
Figure 7: In successive ExpDis iterations, the student captures more of the explorer’s diversity while avoiding its degradations. We track the explorers and the student as they evolve during several ExpDis rounds and report the final baseline policies to the right of each panel. (a) Explorer RL causes a rise in token entropy often at the expense of accuracy, whereas the student’s entropy generally rises together with its accuracy across rounds. (b) The explorer policy semantic diversity stays roughly constant, and the student closes most of the gap to it by the final round. (c) We find that the student does not retain the lexical diversity of its explorers: it declines over rounds even as the student’s accuracy improves. Together with Figure 5, where large λ harms the student, this suggests that lexical diversity is not a useful target for exploration. (d) Reasoning faithfulness, as defined by Rahman et al. [31], measures whether a response’s reasoning supports its final answer. We find that novelty bonus typically causes faithfulness to decline, though the impact is mild by the final round. Results are averaged over three models (breakdown in Figure 14).

Furthermore, explorer policies reach roughly the same semantic diversity in every round, but the gap between them and the student they produce nearly closes by the fourth. One explanation is that each round’s explorers start from a stronger student and therefore solve more of the training problems, so more problems contribute a correct trajectory to the distillation set and the student inherits a broader range of the explorer policy’s solutions.

## 6 Discussion

In standard RLVR, exploration is limited by the need to remain correct. ExpDis removes this constraint by moving exploration into a separate policy. Because the explorer is never deployed, it can be rewarded for novelty, pushed to extremes, or discarded after a single round, with only filtered reasoning traces passing to the student. Our work is inspired by humans who experiment with bold new ideas when failure is permitted. We hypothesize that models too need a space to explore, try new ideas, and be allowed to fail.

Our results suggest that decoupling exploration from optimization improves RLVR even at a small budget, and we expect its importance and the value of exploration to grow as RL is scaled. As models improve, they become proficient at employing the solution strategies they have seen before, and the problems worth training on are increasingly those where correct solutions are rare or depart from the solution strategies the model has already mastered. A model optimized only for correctness may avoid sampling such novel strategies. This problem may be especially true beyond mathematical reasoning, for example in long-horizon agentic tasks, theorem proving, and open-ended scientific problems. Decoupling lets explorers be aggressive, and this aggressive exploration might be necessary in order for models to discover completely new ideas.

In our work, we focus our effort primarily on a single novelty bonus and explorers that share the student’s architecture. Under ExpDis, explorers could use different novelty signals or sampling strategies to specialize in different parts of the solution distribution. Because an explorer only has to produce useful traces, the best exploration method under ExpDis may look quite different from the best one under standard RLVR. We speculate that exploration methods previously too destabilizing to apply directly to a trained model may become usable in this setting. ExpDis may unlock new and more aggressive exploration mechanisms.

## 7 Acknowledgments

This project was generously supported by the Google TPU Builders Program, by IBM, and by the NVIDIA Academic Grant Program. We thank Ang Li, Rohun Agrawal, Kaiwen Zha, Andrea Zanette, and He He for helpful discussions.

## References

[1] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. Advances in neural information processing systems, 30, 2017.

[2] Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. Advances in neural information processing systems, 29, 2016.

[3] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.

[4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[5] Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? Advances in Neural Information Processing Systems, 38:57654–57689, 2026.

[6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[7] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

[8] Jim Dilkes, Vahid Yazdanpanah, and Sebastian Stein. Instruction-conditioned exploration with asymmetric reinforcement learning and self-distillation. arXiv preprint arXiv:2608.02087, 2026.

[9] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O Stanley, and Jeff Clune. First return, then explore. Nature, 590(7847):580–586, 2021.

[10] Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.

[11] Jingtong Gao, Ling Pan, Yejing Wang, Rui Zhong, Chi Lu, Maolin Wang, Qingpeng Cai, Peng Jiang, and Xiangyu Zhao. Navigate the unknown: Enhancing llm reasoning with intrinsic motivation guided exploration. arXiv preprint arXiv:2505.17621, 2025.

[12] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with mmlu? In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, 2025.

[13] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. In International Conference on Learning Representations, volume 2026, pages 108059–108130, 2026.

[14] Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998, 2023.

[15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[16] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner series. https://capricious-hydrogen-41c.notion.site Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680, 2025. Notion Blog.

[17] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

[18] Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine learning, 49(2):209–232, 2002.

[19] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

[20] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

[21] Tianjian Li, Yiming Zhang, Ping Yu, Swarnadeep Saha, Daniel Khashabi, Jason Weston, Jack Lanchantin, and Tianlu Wang. Jointly reinforcing diversity and quality in language model generations. arXiv preprint arXiv:2509.02534, 2025.

[22] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601, 2024.

[23] Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. ZebraLogic: On the scaling limits of LLMs for logical reasoning. In International conference on machine learning, pages 37889–37905. PMLR, 2025.

[24] Alexander H Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, et al. Ministral 3. arXiv preprint arXiv:2601.08584, 2026.

[25] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. Advances in Neural Information Processing Systems, 38:17998–18031, 2026.

[26] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025. URL https://arxiv. org/abs/2503.20783, 1, 2024.

[27] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. DeepScaleR: Surpassing O1-Preview with a 1.5B model by scaling RL, 2025. Notion Blog.

[28] Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, et al. Mopd: Multi-teacher on-policy distillation for capability integration in llm post-training. arXiv preprint arXiv:2606.30406, 2026.

[29] Mathematical Association of America. American invitational mathematics examination (aime), 2026.

[30] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pages 2778–2787. PMLR, 2017.

[31] Salman Rahman, Jingyan Shen, Anna Mordvina, Hamid Palangi, Saadia Gabriel, and Pavel Izmailov. When can LLMs learn to reason with weak supervision? arXiv preprint arXiv:2604.18574, 2026.

[32] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

[33] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[34] Özgür ¸Sim¸sek and Andrew G Barto. An intrinsic reward mechanism for efficient exploration. In Proceedings ofthe 23rd international conference on Machine learning, pages 833–840, 2006.

[35] Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, et al. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

[36] Yuda Song, Julia Kempe, and Remi Munos. Outcome-based exploration for llm reasoning. arXiv preprint arXiv:2509.06941, 2025.

[37] Richard S Sutton and Andrew Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

[38] Jens Tuyls, Dylan Foster, Akshay Krishnamurthy, and Jordan Ash. Representation-based exploration for language models: From test-time to post-training. In International Conference on Learning Representations, volume 2026, pages 120621–120645, 2026.

[39] Jingchu Wang, Bingbing Xu, Yige Yuan, Bin Xie, Xiaoqian Sun, and Huawei Shen. R<sup>2</sup>PO: Decoupling training trajectories from inference responses for llm reasoning. arXiv preprint arXiv:2601.11960, 2026.

[40] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. Advances in Neural Information Processing Systems, 38: 115452–115486, 2026.

[41] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

[42] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[43] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.

[44] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

[45] Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu, Maosong Sun, and Hao Peng. From f (x) and g (x) to f (g (x)): Llms learn new skills in rl by composing old ones. In International Conference on Learning Representations, volume 2026, pages 147547–147574, 2026.

[46] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

[47] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

## A Training & Hyperparameters

We report all hyperparameters shared by all methods in Table 1. As in the standard DAPO [44] algorithm, we perform dynamic sampling, overlong filtering, clip-higher, and do not include a KL penalty term. We use the Dr. GRPO [26] loss normalization, which divides the token-summed loss by a fixed constant instead of the number of tokens.

Table 1: Hyperparameters. Shared by all models and all runs unless noted.
<table><tr><td colspan="2">RL (explorer, student, baselines)</td></tr><tr><td>Objective Prompts per step, group size</td><td>DAPO clipped objective,  $\varepsilon _ { \mathrm { l o w } } = 0 . 2 , \varepsilon _ { \mathrm { h i g h } } = 0 . 2 8 , \mathrm { K L } \beta = 0$   $B { = } 4 , G { = } 1 6$ </td></tr><tr><td>Steps</td><td>explorer 200, student 100, baselines 300, DAPO (4× steps) 1,200 total</td></tr><tr><td>Max completion length</td><td>32,768</td></tr><tr><td>Soft overlong penalty</td><td>linear over the final 6,554 tokens</td></tr><tr><td>Optimizer</td><td>AdamW  $( \beta = ( 0 . 9 , 0 . 9 5 )$  , no weight decay)</td></tr><tr><td>Learning rate (constant)</td><td>explorer and DAPO baselines  $5 \times 1 0 ^ { - 6 } ;$  student</td></tr><tr><td>SFT (student)</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td></td></tr><tr><td>Loss</td><td>masked cross-entropy loss (completion-only)</td></tr><tr><td>Epochs, max examples</td><td>2 epochs, 500 per round</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 6 }$  (constant)</td></tr><tr><td>Novelty bonus (RND)</td><td></td></tr><tr><td>Layers</td><td> $\{ \lfloor L / 4 \rfloor , \lfloor L / 2 \rfloor , \lfloor 3 L / 4 \rfloor \}$ </td></tr><tr><td>Predictor and target</td><td>3-layer MLP (Linear-ReLU-Linear–ReLU-Linear), width and output 512</td></tr><tr><td>Predictor optimizer</td><td>Adam, learning rate  $1 0 ^ { - 4 }$  , one step per policy update; predictor and target re-</td></tr><tr><td>Features</td><td>initialized for every explorer and round live policy&#x27;s hidden states, mean-pooled over completion tokens</td></tr><tr><td>Bonus</td><td></td></tr><tr><td></td><td>raw (no normalization), credited on verifier-correct completions only</td></tr></table>

RL update. All RL stages use entirely on-policy steps: vLLM servers reload the trainer’s weights after every step and each batch of 4 prompts × 16 samples contributes to the following gradient step.  
![](images/28687653813023af528127622f6794580b2eae86afe07e8132aea5cb6c2fa8d6.jpg)

![](images/2bd3a2afc858997e967c51790027385c515085f06db27f4a3d36f7a682d710c1.jpg)

![](images/4cf36211247853909af842878c4e311c8195a2d26082a6ecec80030c85eb50ad.jpg)

![](images/e173359f1b5831ae62fc2d4b2608533cec5d480d577d59cdcac4a427d8099c57.jpg)  
Figure 8: Training dynamics of continued training on Qwen3-1.7B with our ExpDis setup $( \lambda \mathrm { = } 0 . 5 )$

Table 2: Update allocation per configuration. MR-ME (multi-round, multi-explorer) combines both axes: R rounds with K explorers per round. Every ExpDis run uses 200 explorer and 100 student updates (19,200 selected rollouts); single-model baselines spend all 300 on one model. Parallel explorers run concurrently, so wall-clock time is equal across rows. DAPO (4× steps) is the only exception.
<table><tr><td>Configuration</td><td> $R \times K$  Explorer updates</td><td>Student updates per round</td></tr><tr><td>Single-Explorer</td><td>1 × 1 200</td><td>100</td></tr><tr><td>Breadth K=2</td><td>1 × 2 100 each</td><td>100</td></tr><tr><td>Breadth K=3</td><td>1 × 3 67/67/66</td><td>100</td></tr><tr><td>Breadth K=5</td><td>1 × 5 40 each</td><td>100</td></tr><tr><td>Breadth  $K { = } 7$ </td><td>1 × 7 29/29/29/29/28/28/28</td><td>100</td></tr><tr><td>Depth R=4</td><td>4 × 1 50 per round</td><td>25</td></tr><tr><td>Depth R=5 5 × 1</td><td>40 per round</td><td>20</td></tr><tr><td>MR-ME R=4, K=3 4× 3</td><td>17/17/16 per round</td><td>25</td></tr><tr><td>MR-ME R=5, K=3 5 × 3</td><td>14/13/13 per round</td><td>20</td></tr><tr><td>MR-ME R=4, K=5</td><td>4 ×5 10 per explorer per round</td><td>25</td></tr></table>

## A.1 Compute budget

Hardware. Each run uses one TPU v5litepod-64 slice (16 hosts): 4 hosts train the policy with FSDP and 12 serve rollouts with vLLM in bf16 with a 40,960-token context.

Table 3: Sampling parameters. <sup>∗</sup>32 for AMC23 and 8 for GSM8K.
<table><tr><td></td><td>Training rollouts</td><td>Evaluation</td></tr><tr><td>Temperature</td><td>1.0</td><td>0.6</td></tr><tr><td>top-p / top-k / min-p</td><td>0.95 / 20 / –</td><td>0.95 / 20 / 0</td></tr><tr><td>Max completion tokens</td><td>32,768</td><td>32,768</td></tr><tr><td>Samples per problem</td><td>16</td><td>64*</td></tr></table>

## A.2 pass@k estimator

With n samples per problem, of which $c _ { i }$ are correct on problem $i ,$ we use the unbiased estimator of Chen et al. [4] averaged over N problems:

$$
\mathrm { p a s s @ } k \ : = \ : \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ 1 - \frac { { \binom { n - c _ { i } } { k } } } { { \binom { n } { k } } } \right] , \qquad k \le n .\tag{3}
$$

## B Extended Results

We report pass@k for all models and methods in Figure 9 and Table 4.

![](images/70499e45ee889b0d28c266de297cb5ac4c98b0f1f87f153decea577aab5ddda2.jpg)  
Base DAPO DAPO (4 × steps) DAPO (+novelty) ExpDis (single-round) ExpDis  
Figure 9: Full pass@k results.

## C Novelty Bonus Analysis

Impact on DAPO (+novelty). In the DAPO (+novelty) baseline, we augment the standard correctness-only reward with a novelty bonus as in (1). We study how the method performs with various novelty bonus weights λ from 0.25 to 2. We perform three training runs over different seeds and find that, with the RND novelty bonus, the mean mathematical reasoning performance degrades in comparison to DAPO.

![](images/9f4459c02e5142729cf42f77d0b9ab4d11f4d4d7779f8e8283c9a54a42ec0845.jpg)  
Figure 10: Adding a novelty bonus directly to the reward in DAPO degrades accuracy on math reasoning.

Table 4: Mean pass@k over the five primary benchmarks.
<table><tr><td>Model</td><td>Method</td><td>@1</td><td>@2</td><td>@4</td><td>@8</td><td>@16</td><td>@32</td><td>@64</td></tr><tr><td>Qwen3-1.7B</td><td>Base</td><td>43.42</td><td>52.18</td><td>58.26</td><td>63.02</td><td>66.95</td><td>70.63</td><td>73.82</td></tr><tr><td>Qwen3-1.7B</td><td>DAPO</td><td>45.63</td><td>54.31</td><td>60.67</td><td>65.54</td><td>69.16</td><td>72.14</td><td>74.64</td></tr><tr><td>Qwen3-1.7B</td><td>DAPO (4× steps)</td><td>47.28</td><td>56.44</td><td>62.83</td><td>67.41</td><td>70.80</td><td>73.62</td><td>76.24</td></tr><tr><td>Qwen3-1.7B</td><td>DAPO (+novelty)</td><td>45.08</td><td>53.78</td><td>60.07</td><td>64.91</td><td>68.61</td><td>71.77</td><td>74.43</td></tr><tr><td>Qwen3-1.7B</td><td>ExpDis (single-round)</td><td>48.93</td><td>58.55</td><td>64.98</td><td>69.27</td><td>72.43</td><td>75.10</td><td>77.83</td></tr><tr><td>Qwen3-1.7B</td><td>ExpDis</td><td>51.81</td><td>61.40</td><td>68.03</td><td>72.94</td><td>76.97</td><td>80.29</td><td>83.04</td></tr><tr><td>Qwen3-4B</td><td>Base</td><td>61.06</td><td>68.90</td><td>72.83</td><td>75.43</td><td>77.41</td><td>78.90</td><td>79.86</td></tr><tr><td>Qwen3-4B</td><td>DAPO</td><td>62.58</td><td>69.97</td><td>73.84</td><td>76.46</td><td>78.41</td><td>79.88</td><td>81.14</td></tr><tr><td>Qwen3-4B</td><td>DAPO (4× steps)</td><td>63.64</td><td>71.15</td><td>75.05</td><td>77.65</td><td>79.62</td><td>81.20</td><td>82.82</td></tr><tr><td>Qwen3-4B</td><td>DAPO (+novelty)</td><td>62.20</td><td>69.70</td><td>73.59</td><td>76.20</td><td>78.16</td><td>79.64</td><td>80.82</td></tr><tr><td>Qwen3-4B</td><td>ExpDis (single-round)</td><td>64.69</td><td>72.32</td><td>76.24</td><td>78.83</td><td>80.82</td><td>82.51</td><td>84.50</td></tr><tr><td>Qwen3-4B</td><td>ExpDis</td><td>66.50</td><td>74.80</td><td>79.13</td><td>81.82</td><td>83.82</td><td>85.53</td><td>87.63</td></tr><tr><td>Ministral-3-3B</td><td>Base</td><td>27.95</td><td>34.54</td><td>40.52</td><td>45.34</td><td>49.36</td><td>53.00</td><td>56.50</td></tr><tr><td>Ministral-3-3B</td><td>DAPO</td><td>29.63</td><td>36.63</td><td>43.21</td><td>48.79</td><td>53.45</td><td>57.19</td><td>60.12</td></tr><tr><td>Ministral-3-3B</td><td>DAPO (4× steps)</td><td>30.90</td><td>38.19</td><td>45.05</td><td>50.86</td><td>55.74</td><td>59.63</td><td>62.36</td></tr><tr><td>Ministral-3-3B</td><td>DAPO (+novelty)</td><td>29.21</td><td>36.11</td><td>42.54</td><td>47.93</td><td>52.43</td><td>56.15</td><td>59.22</td></tr><tr><td>Ministral-3-3B</td><td>ExpDis (single-round)</td><td>32.16</td><td>39.74</td><td>46.87</td><td>52.93</td><td>58.01</td><td>62.05</td><td>64.59</td></tr><tr><td>Ministral-3-3B</td><td>ExpDis</td><td>34.76</td><td>43.22</td><td>51.31</td><td>58.35</td><td>64.37</td><td>69.24</td><td>72.56</td></tr></table>

Impact on student. We additionally study the impact that sweeping the novelty bonus weight (λ) on the explorer policy has on the downstream student policy in a single-round ExpDis setup. Though any amount of novelty bonus hurts the explorer policy, even at high weights (such as λ=2) where pass@k regresses, the downstream student policy does not inherit this degradation.

![](images/29322cc2b730143b38e4f6e932fdcd3cefce619b0de44c79086bbf4697e67803.jpg)  
Figure 11: On Qwen3-1.7B, we find that a novelty bonus weight of λ=0.5 on the explorer policy roughly maximizes downstream student policy performance, though several values of λ outperform the DAPO baseline.

## C.1 Novelty metric $( r _ { \mathrm { n o v e l t y } } )$

To test whether our gains depend on RND, we train two single-round explorers on Qwen3-1.7B with a different novelty term based on the mean-pooled, L2-normalized final-layer hidden states of each trajectory.

1. kNN novelty scores each correct completion by its mean cosine distance to its $k { = } 1 6$ nearest neighbors in a replay buffer of the 4,096 most recent correct completions.

DAPO DAPO (4× steps) ExpDis DAPO (+novelty) Qwen3-1.7B Qwen3-4B Ministral-3B  
![](images/54040c552585ad1303e5e0a5811730ee95b70ca58f685b23935affa3c820b870.jpg)  
Figure 12: We report the degradation from adding a novelty bonus directly to the DAPO reward.

2. Elliptical novelty, adapted from the elliptical bonus of Tuyls et al. [38], uses the same features ϕ and the bonus $\sqrt { \phi ^ { \top } \Sigma ^ { - 1 } \phi } ,$ , where Σ is a ridge-regularized running covariance, $\begin{array} { r } { \Sigma = I + \sum _ { i } \phi _ { i } \phi _ { i } ^ { \intercal } } \end{array}$ over past correct completions.

We use $\lambda { = } 0 . 5$ for both and report results in Table 5.

Table 5: We ablate the choice of novelty metric used as our novelty bonus.
<table><tr><td>Method</td><td> $r _ { \mathrm { n o v e l t y } }$ </td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>MATH500</td><td>Minerva</td><td>Mean</td></tr><tr><td>DAPO</td><td>一</td><td>50.05</td><td>36.93</td><td>37.66</td><td>74.24</td><td>29.27</td><td>45.63</td></tr><tr><td>DAPO (4× steps)</td><td>一</td><td>51.41</td><td>38.05</td><td>41.96</td><td>75.02</td><td>29.98</td><td>47.28</td></tr><tr><td>ExpDis (ours)</td><td>RND</td><td>52.76</td><td>39.17</td><td>46.25</td><td>75.80</td><td>30.69</td><td>48.93</td></tr><tr><td>ExpDis</td><td>kNN</td><td>50.83</td><td>37.50</td><td>43.75</td><td>75.00</td><td>29.75</td><td>47.37</td></tr><tr><td>ExpDis</td><td>Elliptical</td><td>51.56</td><td>38.23</td><td>44.79</td><td>75.30</td><td>30.40</td><td>48.06</td></tr></table>

![](images/bda462a02c9dee7e0932f4c975cc1de8ad3c659254c3b49e2244340922cb7fbf.jpg)

![](images/d0402bb82a1be7c368bbb6ebcb12aa62218987106d641d5729376cccb92a60ec.jpg)

![](images/f253a78d6a2cc7cdca0654997681b0944a3fe667705c5b3c9867d82743f0d2f3.jpg)  
Qwen3-1.7B, annealed λ: K= 3 explorers K= 1 explorer

![](images/f528294bf4b863785efae241c153bab22bf021635639c1c3e91c94571dceba1f.jpg)

![](images/99e9b452c617a63da5fd4ad24ce7dcedf75a1e03dd220fe0b78c5431a4953cd7.jpg)  
Figure 13: Impact of increasing rounds given a fixed number of explorers per round.

## D Diversity Analysis

We define the diversity metrics used in the paper.

InterDistinct-4. Given n generations $\{ g _ { 1 } , \ldots , g _ { n } \}$ for a single problem, let $4 \mathrm { g r a m s } ( g _ { i } )$ be the multiset of 4-grams in $g _ { i }$ and $G _ { 4 } = \mathrm { U } _ { i } 4 \mathrm { g r a m s } ( g _ { i } )$ the set of distinct 4-grams in the pool. Then

$$
\mathrm { I n t e r D i s t i n c t { - } } 4 = \frac { \left| { \cal G } _ { 4 } \right| } { \sum _ { i } \left| 4 \mathrm { g r a m s } ( g _ { i } ) \right| } ,
$$

the ratio of distinct 4-grams to total 4-grams in the pool. We average across problems.

AnswerDistinct@n. For each problem, count the number of distinct extracted final answers across the n generations, then average across problems.

Token entropy. For each model we sample 8 rollouts for each of the 30 AIME24 problems with the evaluation sampler and record the mean per-token entropy of the sampled tokens.

Semantic diversity. We follow Li et al. [21], who finetune Qwen3-Embedding-4B as a classifier of whether two mathematical reasoning traces use the same approach. We use their checkpoint to produce pairwise similarity scores between all generations for a single prompt. We then cluster the generations and report the resulting number of clusters C normalized by the number of generations.

Table 6: We report diversity metrics for Qwen3-1.7B models on AIME24 generations.
<table><tr><td>Model</td><td>Entropy</td><td></td><td>Semantic Ans. entropy</td><td>Distinct ans.</td><td>InterDistinct-4</td></tr><tr><td>Base</td><td>0.237</td><td>0.100</td><td>1.42</td><td>8.94</td><td>0.311</td></tr><tr><td>DAPO</td><td>0.251</td><td>0.096</td><td>1.37</td><td>8.50</td><td>0.308</td></tr><tr><td>DAPO (4× steps)</td><td>0.265</td><td>0.091</td><td>1.35</td><td>8.46</td><td>0.302</td></tr><tr><td>ExpDis (single-explorer)</td><td>0.306</td><td>0.120</td><td>1.41</td><td>9.93</td><td>0.357</td></tr><tr><td>ExpDis</td><td>0.391</td><td>0.143</td><td>1.30</td><td>9.44</td><td>0.339</td></tr></table>

Table 7: We report InterDistinct-4 (Inter4) and AnswerDistinct@n (Ans@n) for the explorer and the student on AIME24 across explorer weights λ.
<table><tr><td colspan="2"></td><td colspan="2">Explorer</td><td colspan="2">Student</td></tr><tr><td>Model</td><td>Method</td><td>Inter4</td><td>Ans@n</td><td>Inter4</td><td>Ans@n</td></tr><tr><td>Qwen3-1.7B</td><td>Base</td><td></td><td></td><td>0.311</td><td>8.94</td></tr><tr><td>Qwen3-1.7B</td><td>DAPO</td><td></td><td></td><td>0.308</td><td>8.50</td></tr><tr><td>Qwen3-1.7B</td><td>DAPO (+novelty, λ=0.5)</td><td></td><td></td><td>0.342</td><td>9.39</td></tr><tr><td>Qwen3-1.7B</td><td>ExpDis (λ=0.25)</td><td>0.403</td><td>11.60</td><td>0.346</td><td>9.63</td></tr><tr><td>Qwen3-1.7B</td><td>ExpDis (λ=0.5)</td><td>0.422</td><td>12.61</td><td>0.357</td><td>9.93</td></tr><tr><td>Qwen3-1.7B</td><td>ExpDis (λ=0.75)</td><td>0.478</td><td>14.44</td><td>0.363</td><td>10.02</td></tr><tr><td>Qwen3-1.7B</td><td>ExpDis (λ=1.0)</td><td>0.534</td><td>16.28</td><td>0.335</td><td>9.30</td></tr><tr><td>Qwen3-1.7B</td><td>ExpDis (λ=2.0)</td><td>0.550</td><td>23.61</td><td>0.283</td><td>7.87</td></tr><tr><td>Qwen3-4B</td><td>Base</td><td></td><td></td><td>0.290</td><td>4.02</td></tr><tr><td>Qwen3-4B</td><td>DAPO</td><td></td><td></td><td>0.287</td><td>3.82</td></tr><tr><td>Qwen3-4B</td><td>DAPO (+novelty, λ=0.5)</td><td></td><td></td><td>0.319</td><td>4.22</td></tr><tr><td>Qwen3-4B</td><td>ExpDis (λ=0.5)</td><td>0.394</td><td>5.67</td><td>0.333</td><td>4.47</td></tr></table>

## E Ablations

Quality filtering. With three explorers, keeping every correct trajectory from the pooled explorers (NaivePool) gives a mean of 48.95%, no better than a single explorer (48.93%). Applying the quality filter increases performance to 49.47%.

Student distillation vs student RL. On Qwen3-1.7B, distilling unfiltered explorer traces and then running RL reaches 49.79% on AIME24, and distilling filtered traces without RL reaches 49.84%. Both variants underperform DAPO (at 50.05%). Only filtering followed by RL reaches 52.76% (Table 8).

Annealing λ. We find that annealing λ during multiple rounds (0.75, 0.5, 0.35, 0.25) outperforms a fixed λ=0.5 at every round (see Figure 15).

Training data. Every run samples the same number of prompts (about 1,200), so these runs differ only in the pool they sample from (Figure 16, Table 9). Sampling from DAPO-Math-17K or a 17K subset of DeepScaleR gives similar results, with neither better on every benchmark. Sampling ExpDis prompts from the full 40K DeepScaleR pool is better on every benchmark. All main results use DAPO-Math-17K.

![](images/e36bf66aeff7aeb3e2a25dea7fe535cc1771e95baf88f3c0d1fb7862c346d163.jpg)  
Figure 14: We study the evolution of student and teacher policies during several stages of ExpDis.

Table 8: Ablating student distillation, student RL, or both.
<table><tr><td>Configuration</td><td>AIME24</td></tr><tr><td>DAPO</td><td>50.05</td></tr><tr><td>Unfiltered  $\mathrm { S F T } + \mathrm { R L }$ </td><td>49.79</td></tr><tr><td>Filtered SFT, no RL</td><td>49.84</td></tr><tr><td>Filtered  $\mathrm { S F T + R L \left( E x p D i s \right) }$ </td><td>52.76</td></tr></table>

![](images/70c3aa035ff5b6b81736cdf06ca447c20b56bbf760fb089151b1adc6572941c7.jpg)  
Figure 15: Fixed versus annealed λ across four rounds (Qwen3-1.7B, single-explorer). The annealed schedule uses $\lambda { = } 0 . 7 5 , 0 . { \bar { 5 } } , 0 . 3 5 .$ , and 0.25.

Table 9: We ablate the choice of training dataset. We see roughly the same performance across all datasets.
<table><tr><td>Method</td><td>Dataset</td><td>AIME24</td><td>AIME25</td><td>AIME26</td><td>MATH500</td><td>AMC23</td><td>Minerva</td><td>GSM8K</td></tr><tr><td>GRPO</td><td>DAPO-Math-17K</td><td>47.81</td><td>35.36</td><td>35.68</td><td>72.84</td><td>83.75</td><td>28.02</td><td>89.89</td></tr><tr><td>GRPO</td><td>DeepScaleR-17K</td><td>47.96</td><td>35.52</td><td>35.68</td><td>72.71</td><td>84.09</td><td>27.86</td><td>89.81</td></tr><tr><td>Dr. GRPO</td><td>DAPO-Math-17K</td><td>48.49</td><td>35.99</td><td>36.46</td><td>73.34</td><td>84.38</td><td>28.45</td><td>90.04</td></tr><tr><td>Dr. GRPO</td><td>DeepScaleR-17K</td><td>48.83</td><td>36.26</td><td>36.51</td><td>73.18</td><td>84.78</td><td>28.20</td><td>89.95</td></tr><tr><td>DAPO</td><td>DAPO-Math-17K</td><td>50.05</td><td>36.93</td><td>37.66</td><td>74.24</td><td>85.16</td><td>29.27</td><td>90.25</td></tr><tr><td>DAPO</td><td>DeepScaleR-17K</td><td>49.68</td><td>37.21</td><td>37.76</td><td>74.08</td><td>85.62</td><td>28.90</td><td>90.14</td></tr><tr><td>ExpDis λ=0.5</td><td>DAPO-Math-17K</td><td>52.76</td><td>39.17</td><td>46.25</td><td>75.80</td><td>86.17</td><td>30.69</td><td>90.68</td></tr><tr><td>ExpDis λ=0.5</td><td>DeepScaleR-17K</td><td>51.77</td><td>38.33</td><td>44.84</td><td>75.16</td><td>86.48</td><td>30.44</td><td>90.59</td></tr><tr><td>ExpDis λ=0.5</td><td>DeepScaleR-full</td><td>52.89</td><td>39.24</td><td>47.66</td><td>76.08</td><td>86.84</td><td>31.37</td><td>90.78</td></tr></table>

![](images/cc4cbb403673f782538098375c78c183d9733e861e749a55094a360674e6af92.jpg)

![](images/41e72fa220cbf53b2e215d2edcbb94ca22d20c499f45ccba9af878e2b8c85411.jpg)

![](images/ee16a4d3cf9ffd63bedaf2bd01aefc79cd2f33fe7caa60105978d1b43c51e890.jpg)

![](images/9ca20257db413736bea6ea8e1f506298b5818dfef5dfbb0060d5ae06f1b4e42f.jpg)

![](images/2ec6f753b3f019f129793de1505903403a77aa062e0f4d0abb20c292976e0515.jpg)

![](images/bf0dec9dc951e68700c72cc689f8ca1793ba6f7e74b3d947b333ab15e0d68dd3.jpg)

![](images/a4440547c487ad9fa810f1857c1c7fa32a29df651f433b2e462628d678a8ef9c.jpg)  
Qwen3-1.7B GRPO Dr.GRPO DAPO ExpDis λ = 0.5  
Figure 16: We ablate the choice of training dataset, sampling a fixed prompt budget from each and training with a single-explorer ExpDis approach. We find a consistent gain over several RLVR baselines.