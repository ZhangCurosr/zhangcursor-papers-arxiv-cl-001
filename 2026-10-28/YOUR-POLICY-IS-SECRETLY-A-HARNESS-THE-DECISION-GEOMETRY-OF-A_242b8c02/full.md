# YOUR POLICY IS SECRETLY A HARNESS: THE DECISION GEOMETRY OF AGENT INTERFACES

Jian Mu The Hong Kong University of Science and Technology (Guangzhou)

Ruibo Shen The Hong Kong University of Science and Technology (Guangzhou)

Yao Shu<sup>∗</sup> The Hong Kong University of Science and Technology (Guangzhou) The Hong Kong University of Science and Technology

## ABSTRACT

Language model agents are post-trained and deployed through a harness, the prompts, tools, and control logic that present a task to the model and execute its outputs. Harnesses are often revised and judged by how an already trained model performs under them, which assumes that what the model learned under one harness carries over to the next. We ask what a policy learns from its harness and what survives when the harness changes. Under KL-regularized control, we show that the log odds of an optimal policy recover how strongly its harness favors each action over the others, so a trained policy carries the preferences of its harness. Deployed return then equals the ceiling of the harness, the best return it offers, minus the mismatch of the policy, the return lost to its own suboptimal decisions. Training changes only the mismatch, while changing the harness moves both terms. Comparing harnesses with a fixed model therefore mixes what each harness offers with how well that model uses it. In MiniWoB++, post-training makes a model’s preferences specific to its training harness. Reversing the order of the observation fields leaves every optimal decision unchanged yet lowers a trained model’s success from 85.0% to 19.2%. Conversely, every new harness that a fixed-model comparison rejects but that raises the ceiling outperforms the original agent after further training under it, and ceiling and mismatch measured beforehand predict these gains. For coding-agent harnesses on Terminal-Bench Lite, ceiling minus mismatch estimated with a stronger model as a stand-in for the optimum reproduces the accuracy ranking, while either term alone does not. Code is available at https://github.com/2020-qqtcg/HRP.

## 1 INTRODUCTION

Language model agents are post-trained and deployed through a harness, the prompts, tools, and control logic that present a task to the model and execute its outputs. Interface design and prompt formatting change agent performance (Yang et al., 2024; Sclar et al., 2024). Automated harness design revises prompts, tools, and workflows and scores each candidate by how a fixed model performs under it (Hu et al., 2025; Zhang et al., 2025b; Fernando et al., 2023; Zhang et al., 2026). Such a score assumes that what the model learned under earlier harnesses carries over to each candidate, yet the same interface change can raise one model’s accuracy and lower that of a fine-tuned model (Mu et al., 2025). At the same time, training can move behavior that a prompt or a harness supplies into the weights (Snell et al., 2022; Ye et al., 2026), and recent methods adapt harnesses and weights together (Hebbar et al., 2026; Chen et al., 2026; Kim et al., 2026). These practices raise the question we study:

What does a policy learn from its harness, and what survives when the harness changes?

Two experiments make the question concrete. In the first, we take a web agent post-trained under one harness and reverse the order of the fields in its observation, a JSON object with fields such as the task instruction, the page state, and the available actions. The fields keep their contents. We also hold fixed the actions, their execution, the rewards, and the reference policy that regularizes training, so every optimal decision stays the same. The model still changes its most probable action at about half of the states, and its task success falls from 85.0% to 19.2%. Keeping the decisions a model should make does not keep the decisions it makes.

![](images/1e0fc184583fc400a95fd87a834db4d88f4fc3bf5d1f8c5c6600d61e7aaaf2db.jpg)  
Figure 1: Adaptation reverses a frozen comparison. The left group evaluates the model trained under Harness 0, the source checkpoint, under both harnesses with its weights frozen. The right group evaluates the models after equal further training under each harness. The dotted line marks the source checkpoint’s success under Harness 0, and colors mark the evaluation harness.

In the second, a new harness adds batched clicks, which execute two clicks in one decision. The new actions raise the best achievable return, yet the trained model’s success drops when we move it to the new harness without changing its weights. A frozen comparison, which evaluates candidate harnesses with the same trained weights, would therefore reject the new harness. We then train the model further for the same budget under each harness, starting from the same trained weights, which we call the source checkpoint. We call further training under the new harness adaptation, and further training under the original harness continuation. The adapted model succeeds more often than both the source checkpoint and the continued one. The harness that first hurt the model gives the best agent. Figure 1 shows the task success at each step, with the new harness as Harness 1.

We explain both results in the control problem that a harness induces over semantic actions, the

operations an agent executes together with their arguments. Under KL-regularized control, the optimal policy is a softmax over action scores. Each score adds an action’s long-term value to its log probability under the reference policy, so an optimal policy’s log odds between two actions are proportional to the difference of their scores. A policy trained to optimality under a harness therefore carries the relative action scores that the harness induces, even though it contains none of the harness’s code, tools, or execution machinery. In this sense, a policy is secretly a harness. In our controlled study, real post-training moves a model’s action scores toward the optimum of its training harness, and each harness leaves its own preferences in the model.

Action scores also show why a trained policy can fail after it is moved to a new harness. The scores a policy carries are the ones it expresses through its training interface, which we call its readout through that interface, and the same weights can express different scores through another interface. A policy is thus secretly its training harness, as read through that harness. A new harness can therefore change which actions are optimal, by moving the optimal scores, and separately change which actions the model picks, by moving the readout. Reversing the observation fields leaves the optimal scores in place and moves only the readout.

Scores fix decisions but not how much return a harness offers, since raising every reward by the same constant changes no decision. To convert score changes into return, we therefore add this value back by applying the regularized performance-gap identity (Geist et al., 2019; O’Donoghue, 2022) to the harness-induced problem. Deployed return equals the harness’s ceiling C, the best return any policy can reach under it, minus the policy’s mismatch M, the return the policy loses by departing from the optimal policy. The mismatch is the KL divergence from the optimal policy, accumulated along the policy’s own trajectories. The ceiling does not depend on the model’s weights, so training under a fixed harness changes only M, while a harness change can move both terms. A frozen comparison of harnesses therefore mixes what each harness offers with how well the current checkpoint uses it. In the batched-click example, the new harness raises the ceiling slightly but raises the frozen model’s mismatch about ten times as much, so return falls. Adaptation then lowers the mismatch, and once the mismatch falls below a threshold we derive, the adapted model beats the original model under the original harness. Figure 2 illustrates both steps. Our contributions are as follows.

![](images/701194c8438c04778ecd07e068e87ef06c9cb4f7308756f9afcfa702a7b7499a.jpg)  
Figure 2: A harness revision and a weight update act on different terms. Summit heights are the ceilings, the robot’s altitude is its return, and the gap between them is the mismatch. Moving the trained model to the new harness with frozen weights (① to ②) raises the ceiling but lowers the return. Updating the weights under the new harness (② to ③) lowers only the mismatch. The adapted model beats the source checkpoint under the original harness once its mismatch M<sup>′</sup> falls below T, the new ceiling minus the source checkpoint’s return under the original harness.

1. What a policy learns under a harness. The log odds of an optimal policy identify exactly the relative action scores that a harness induces. These scores fix every decision but not how much return the harness offers, which Contribution 3 adds back as the ceiling. On MiniWoB++, post-training moves the model’s action preferences toward the training harness’s optimum and makes them specific to that harness.

2. Why transfer fails. We split the score error of a policy moved to a new harness into three parts. The target shift is the change in the optimal scores, the training residual is the error that training under the original harness left behind, and the readout drift is the change in the scores that the model expresses. Presentation changes that keep every optimal decision isolate the readout drift, and they alone can make a trained model fail on most tasks.

3. What transfer costs and how to measure it. Deployed return equals ceiling minus mismatch exactly. From this identity we derive the mismatch threshold below which adaptation turns a harness revision that hurt the frozen model into a gain. Adaptation to batched clicks crosses this threshold and outperforms continued training under the original harness. We also move trained checkpoints between six harnesses. Among the moves that a frozen comparison rejects, ceiling and mismatch measured before adaptation predict which ones adaptation makes beneficial. Mismatch estimated from sampled actions predicts return changes when the optimum is known. For coding agents, where no optimum is available, estimates that put a stronger teacher model in the optimum’s place order five harnesses by Terminal-Bench Lite accuracy.

## 2 HARNESSES INDUCE SEMANTIC CONTROL PROBLEMS

A harness turns model outputs into executed operations. We call an executed operation together with its arguments a semantic action $a \in A .$ , such as clicking a particular element or running a query. Different outputs can execute the same action. For example, a JSON object and a function-call string may issue the same parameterized query. Let $s \in S$ denote the semantic decision state. An interface $\mathcal { I } = ( c _ { \mathcal { T } } , d _ { \mathcal { T } } , e _ { \mathcal { T } } )$ formalizes a harness. It encodes the state into a context, decodes a model response into an action, and executes that action. For a token policy $\pi _ { \theta } .$ , the induced semantic policy sums the probabilities of the outputs that decode to the same action:

$$
\Pi _ { \theta , \mathcal { Z } } ( a \mid s ) = \sum _ { y : d _ { \mathcal { Z } } ( y ; s ) = a } \pi _ { \theta } ( y \mid c _ { \mathcal { Z } } ( s ) ) .\tag{1}
$$

Both the context $c _ { \mathcal { T } }$ and the decoder $d _ { \mathcal { I } }$ enter Eq. (1), so fixed weights do not fix the semantic policy. Together with a fixed task environment, I induces a transition law $P _ { \mathcal { T } } .$ , a utility $r _ { \mathcal { T } }$ that includes execution costs, and a semantic reference policy $\pi _ { \mathcal { T } } ^ { \mathrm { r e f } }$ . Every call to the evaluated model is a decision, including calls for verification or retry, and the state includes the harness’s controller memory, such as tool results and remaining budget. Appendix B specifies this construction and the conditions under which training over tokens yields semantic optimality.

Across comparisons, we fix the initial distribution $\rho _ { 0 } .$ , discount $0 \leq \gamma < 1$ , and regularization coefficient $\tau > 0$ , and measure rewards on a common scale. Each interface’s reference is part of its control problem and does not change during training. For finite $s , A ,$ bounded rewards, and $\pi _ { \mathcal { T } } ^ { \mathrm { r e f } } ( a \mid s ) \dot { > } 0$ , the regularized return of a semantic policy Π is

$$
J _ { \mathcal { T } } ( \Pi ) = \mathbb { E } _ { \rho _ { 0 } , \Pi , P _ { \tau } } \left[ \sum _ { t \geq 0 } \gamma ^ { t } \left( r _ { \mathcal { T } } ( s _ { t } , a _ { t } ) - \tau \log \frac { \Pi ( a _ { t } \mid s _ { t } ) } { \pi _ { \mathcal { T } } ^ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right) \right] .\tag{2}
$$

Write $\Pi _ { \boldsymbol { \mathcal { I } } } ^ { * }$ for the optimal policy and $\begin{array} { r } { d _ { \mathcal { T } } ^ { \Pi } ( s ) = ( 1 - \gamma ) \sum _ { t \geq 0 } \gamma ^ { t } \mathrm { P r } _ { \mathcal { T } } ^ { \Pi } ( s _ { t } = s ) } \end{array}$ for the discounted occupancy of Π.

Controlled study. In our mechanism experiments, every quantity above can be computed exactly. Six harnesses act on three MiniWoB++ task families (Liu et al., 2018), and we compile each induced control problem into a finite graph. H0 executes one native action per decision, and H1–H5 add batched clicks, automatic checks, verification tools, or extra instructions. Each prompt lists the complete action menu with one token per action, so the model’s next-token distribution gives $\Pi _ { \theta , \mathcal { Z } }$ exactly. The reward is binary task success. We post-train Qwen3.5-2B with exact policy gradients, and the frozen base model serves as the reference. A soft Bellman solve gives each harness’s optimum. Evaluations use tasks held out from training, and Appendix F.1 gives the full protocol.

## 3 WHAT A POLICY LEARNS UNDER A HARNESS

We now ask what a model’s action probabilities reveal about the control problem it was trained on.

## 3.1 OPTIMAL POLICIES ENCODE RELATIVE ACTION SCORES

The optimal action value $Q _ { \mathcal { I } } ^ { \ast }$ depends on the harness’s utility and transition law through the soft Bellman equations, and the optimal policy takes the Gibbs form $\Pi _ { \mathcal { T } } ^ { * } ( a \mid s ) \propto \pi _ { \mathcal { T } } ^ { \mathrm { r e f } } ( a \mid s ) \exp ( Q _ { \mathcal { T } } ^ { * } ( s , a ) / \tau )$ (Haarnoja et al., 2017; Geist et al., 2019). An action’s optimal probability therefore depends on its long-term value and its reference probability. We combine the two into the optimal action score

$$
F _ { \overline { { L } } } ^ { * } ( s , a ) = Q _ { \overline { { L } } } ^ { * } ( s , a ) + \tau \log \pi _ { \overline { { L } } } ^ { \mathrm { r e f } } ( a \mid s ) , \qquad \Pi _ { \overline { { L } } } ^ { * } ( \cdot \mid s ) = \mathrm { s o f t m a x } \big ( F _ { \overline { { L } } } ^ { * } ( s , \cdot ) / \tau \big ) .\tag{3}
$$

For instance, an optimal policy may favor a batch of clicks because the batch leads to better outcomes or because the reference places more weight on it. The score records the combined effect.

Adding the same constant to every action’s score at a state leaves the policy unchanged, so only score differences within a state affect decisions. We write $[ F ]$ for the class of score fields related by such statewise constants. Two interfaces have equivalent optimal decisions when $\Pi _ { \mathcal { T } _ { 1 } } ^ { * } ( \cdot \mid s ) = \Pi _ { \mathcal { T } _ { 2 } } ^ { * } ( \cdot \mid s )$ at every aligned state, that is, every semantic state that the two interfaces share.

Proposition 3.1 (Behavioral identification). Two interfaces have equivalent optimal decisions if and only $i f [ F _ { \mathcal { T } _ { 1 } } ^ { * } ] = [ \dot { F } _ { \mathcal { T } _ { 2 } } ^ { * } ]$ . Ifa checkpoint is optimalfor its training interface at every semantic state, then

$$
[ \tau \log \Pi _ { \theta , \mathcal { T } _ { \mathrm { t r } } } ] = [ F _ { \mathcal { T } _ { \mathrm { t r } } } ^ { * } ] , \qquad \tau \log \frac { \Pi _ { \theta , \mathcal { T } _ { \mathrm { t r } } } ( a \mid s ) } { \Pi _ { \theta , \mathcal { T } _ { \mathrm { t r } } } ( b \mid s ) } = F _ { \mathcal { T } _ { \mathrm { t r } } } ^ { * } ( s , a ) - F _ { \mathcal { T } _ { \mathrm { t r } } } ^ { * } ( s , b ) .\tag{4}
$$

Behavior determines exactly these relative scores. Different reference policies, rewards, and even transition laws can share one optimal score field and therefore one optimal policy, in line with partial identifiability results for reward learning (Skalse et al., 2023). Behavior also leaves absolute value undetermined, since adding a constant to every reward raises the optimal return and changes no decision. A policy can thus carry the decisions a harness induces without its code, tools, or execution machinery. The value a harness offers is a separate quantity, which Section 5 adds back. Appendix C proves the proposition from the Gibbs form and constructs such mechanisms.

A real checkpoint departs from the optimum. To compare the two, we represent each class by its centered member, subtracting the mean score across actions at each state: $\bar { \boldsymbol { F } } = \mathcal { P } _ { \perp } \boldsymbol { F }$ with $P _ { \perp } = I - \mathbf { 1 1 } ^ { \top } / n$ and $n = | { \mathcal { A } } |$ . For positive semantic probabilities, the checkpoint’s centered scores are $\widehat { F } _ { \theta , \mathcal { Z } } = \tau P _ { \perp }$ log $\Pi _ { \theta , { \mathcal { I } } } $ , the scores whose softmax in Eq. (3) reproduces the checkpoint’s own policy. We call them the model’s readout through I, the action preferences it expresses through the interface. Its score error under the training interface, $e _ { \mathrm { t r } } = \widehat { F } _ { \theta , \mathcal { T } _ { \mathrm { t r } } } - \bar { F } _ { \mathcal { T } _ { \mathrm { t r } } } ^ { * }$ , vanishes exactly when the checkpoint is optimal.

## 3.2 POST-TRAINING MAKES PREFERENCES HARNESS-SPECIFIC

Proposition 3.1 concerns optimal policies. We ask whether real post-training moves a model toward the optimum of its training harness, and whether what it learns depends on that harness. We posttrain Qwen3.5-2B under each of H0–H5. The geometry error of a checkpoint is the mean squared difference between its centered scores and the optimal ones, averaged over a fixed bank of reachable states, so that every checkpoint is scored on the same decisions.

(a) Task success (%)  
(b) Geometry error $( \times 1 0 ^ { - 2 } )$  
![](images/4d16bacec3717d3622d2313b5097e177862c6aa2135ca40fde71cc96db3f8a2a.jpg)

(c) Geometry error under transfer $( \times 1 0 ^ { - 2 } )$  
![](images/275bf6c20fd5c69a08c6b7e78aecfc80cd9ce67788a6db1286e0bc3fcbca23c9.jpg)  
Figure 3: Post-training moves action preferences toward the optimum of the training harness and makes them specific to it. Rows denote training harnesses. (a) Task success and (b) geometry error relative to the training harness’s optimum during post-training. Colored curves show the harness of each row, gray curves the other harnesses, and dotted lines the error at initialization. (c) Geometry error of each final checkpoint under each deployment harness. Outlined entries mark the lowest error in each column, attained by the checkpoint trained under that harness.

Under every harness, post-training raises task success, and the geometry error falls well below its initial level, by 69% to 90% by the end of training. It falls as much on held-out tasks that play no role in the updates as on training tasks. The model learns effective task behavior while its action scores approach the optimum of its training harness.

To test whether the learned preferences depend on the training harness, we deploy each final checkpoint under all six harnesses on test tasks. A fixed deployment harness fixes the dynamics, reference, and optimal scores, so the six checkpoints are compared against the same target. Under every deployment harness, the checkpoint trained under that harness has the lowest error. Post-training thus makes the model’s preferences specific to its training harness. Figure 3 shows the learning curves in its first two panels and this comparison in the third.

## Takeaways: What a policy learns

A policy trained to optimality under a harness carries that harness’s relative action scores. Real post-training moves a model’s action scores toward these optimal scores and makes them specific to the training harness.

## 4 WHY POLICY TRANSFER FAILS

Section 3 shows that a trained policy carries the relative action scores of its training harness. It carries them in its readout through the training interface rather than independently of that interface, so the same weights need not express the same preferences under another harness. A harness change can therefore break a policy that has learned its training harness well.

## 4.1 A HARNESS CHANGE MOVES THE TARGET AND THE READOUT

Let interfaces 1 and 2 be the training and deployment interfaces. A change in execution can change which actions are optimal, and a change in presentation can change which actions the frozen model selects. On aligned states and actions, the deployed score error splits into three terms:

$$
\underbrace { \widehat { F } _ { \theta , 2 } - \bar { F } _ { 2 } ^ { * } } _ { \mathrm { d e p l o y e d \ e r r o r } } = \underbrace { \bar { F } _ { 1 } ^ { * } - \bar { F } _ { 2 } ^ { * } } _ { g _ { 1 2 } : \mathrm { \ t a r g e t \ s h i f t } } + \underbrace { \widehat { F } _ { \theta , 1 } - \bar { F } _ { 1 } ^ { * } } _ { e _ { 1 } : \mathrm { \ t r a i n i n g \ r e s i d u a l } } + \underbrace { \widehat { F } _ { \theta , 2 } - \widehat { F } _ { \theta , 1 } } _ { \eta _ { 1 2 } : \mathrm { \ r e a d o u t \ d r i f t } } .\tag{5}
$$

The training residual exists before the change, and the target shift and the readout drift are its two effects. They are vectors, so they can add up or cancel.

## 4.2 EQUIVALENT OPTIMAL DECISIONS, DIFFERENT BEHAVIOR

When two interfaces have equivalent optimal decisions, $g _ { 1 2 } = 0$ , and any change in the model’s error comes from readout drift. We test this with the H0-trained checkpoint. We reverse the order of the observation fields, render the action menu as a list, or reassign the action codes, and keep the transitions, rewards, and semantic reference fixed, so neither the target nor the reference can move. The optimal policy and the optimal return thus stay the same, as numerical checks confirm.

Equivalent optimal decisions do not produce equivalent behavior. Reversing the field order has the largest effect. It changes the model’s most probable action at about half of the states, and most tasks

Table 1: Equivalent optimal decisions, different choices. Choice change is the share of states whose most probable action changes from the original. M and J are the mismatch and regularized return of Section 5.2.
<table><tr><td>Presentation</td><td>Choice change</td><td>M</td><td>J</td><td>Task success</td></tr><tr><td>Original</td><td>0.0%</td><td>0.015</td><td>0.423</td><td>85.0%</td></tr><tr><td>Reversed fields</td><td>52.0%</td><td>0.484</td><td>-0.046</td><td>19.2%</td></tr><tr><td>Menu as list</td><td>9.2%</td><td>0.023</td><td>0.415</td><td>86.0%</td></tr><tr><td>Shifted codes</td><td>27.7%</td><td>0.148</td><td>0.289</td><td>57.1%</td></tr><tr><td>Reversed codes</td><td>30.0%</td><td>0.130</td><td>0.308</td><td>68.2%</td></tr><tr><td>Permuted codes</td><td>25.3%</td><td>0.110</td><td>0.328</td><td>67.6%</td></tr></table>

then fail. Reassigning the action codes has a moderate effect, and the list format has a small one. Unlike earlier studies of prompt-format sensitivity (Sclar et al., 2024), the control problem here certifies that the optimal policy is unchanged, so the entire effect comes from the model’s readout. Table 1 lists the results, Appendix F.2 gives the protocol, and Appendix D.4 bounds readout drift by the model’s sensitivity to corresponding outputs.

## Takeaways: Why transfer fails

A harness change can move the optimal decisions and the model’s choices separately. A new presentation alone can make a trained model fail on most tasks.

## 5 WHAT TRANSFER COSTS AND WHEN ADAPTATION RECOVERS IT

Separating the two changes shows what moved. What they cost in return depends on which decisions they affect and how often the policy reaches them.

## 5.1 FROM SCORE ERRORS TO LOST RETURN

Consider a single decision at state s in a fixed interface I. If the policy Π makes this decision and the optimal policy makes every later one, the return lost relative to acting optimally throughout is $\tau \mathrm { K L } \bar { ( } \Pi ( \cdot \mid \bar { s ) } \| \Pi _ { \mathcal { T } } ^ { * } ( \cdot \mid s ) \rangle$ ). If Π also makes the later decisions, each of them adds its own loss, and summing along the trajectories of Π gives the total gap (O’Donoghue, 2022).

Proposition 5.1 (Regularized performance gap). For any semantic policy Π in a fixed interface,

$$
J _ { \mathcal { T } } ( \Pi _ { \mathcal { T } } ^ { * } ) - J _ { \mathcal { T } } ( \Pi ) = \frac { \tau } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { \mathcal { T } } ^ { \Pi } } \mathrm { K L } \big ( \Pi ( \cdot \mid s ) \| \Pi _ { \mathcal { T } } ^ { * } ( \cdot \mid s ) \big ) .\tag{6}
$$

The occupancy weights each local loss by how often the evaluated policy reaches the state, so errors on the policy’s common paths cost the most. In terms of scores, the local loss is the Bregman divergence $D _ { \Phi } ( F , G )$ of $\begin{array} { r } { { \bf { \dot { \Phi } } } ( F ) = \tau \log \sum _ { a } e ^ { F _ { a } / \tau } } \end{array}$ , a smooth maximum of the scores. $D _ { \Phi } ( F , G )$ is the gap between $\Phi ( F )$ and the linear approximation of Φ at G. For positive semantic action probabilities,

$$
\tau \mathrm { K L } ( \Pi _ { \theta , \mathcal { T } } \Vert \Pi _ { \mathcal { T } } ^ { * } ) = D _ { \Phi } ( F _ { \mathcal { T } } ^ { * } , \widehat { F } _ { \theta , \mathcal { T } } ) .\tag{7}
$$

The scores of the actions at a state locate the model’s decision, and $D _ { \Phi }$ measures how much return the decision loses when the model’s scores sit at $\widehat { F } _ { \theta , \mathcal { I } }$ instead of $F _ { \mathcal { I } } ^ { * }$ . We call these scores and this cost together the decision geometry of a harness. Averaging $D _ { \Phi }$ under the deployed policy’s occupancy turns the score error of Eq. (5) into lost return. For small errors, $D _ { \Phi }$ grows roughly quadratically with the summed error, so the three terms of Eq. (5) cost return through their sum rather than one by one, and Proposition 6.1 below turns this into a bound. Appendix D proves both identities.

## 5.2 CEILING AND MISMATCH

Relative scores fix the decisions and leave the value of a harness undetermined. We add that value back. Define the ceiling $C ( \mathcal { T } ) = J _ { \mathcal { T } } ( \Pi _ { \mathcal { T } } ^ { * } )$ , the optimal regularized return of the harness and thus the

best that any policy can reach under it, and the mismatch $M ( \theta , { \mathcal { T } } )$ , the return the checkpoint leaves unrealized. Proposition 5.1 gives

$$
J _ { \mathcal { T } } ( \Pi _ { \theta , \mathcal { T } } ) = C ( \mathcal { T } ) - M ( \theta , \mathcal { Z } ) , \qquad M ( \theta , \mathcal { Z } ) = \frac { \tau } { 1 - \gamma } \mathbb { E } _ { d _ { \tau } ^ { \Pi _ { \theta , \mathcal { T } } } } \mathrm { K L } ( \Pi _ { \theta , \mathcal { Z } } \| \Pi _ { \mathcal { Z } } ^ { * } ) .\tag{8}
$$

The ceiling depends on the harness and the objective, and the mismatch depends on the harness and the checkpoint, including errors inherited from training. With the harness fixed, a weight update changes return only through $M ,$ and a harness change can move both terms. In the summit illustration of Figure 2, the step from ① to ② is a harness revision, and the step from ② to ③ is a weight update. We call a checkpoint together with the harness it is deployed under a pair. For a frozen checkpoint, write $J _ { j } , C _ { j }$ , and $M _ { j }$ for the quantities under interface j. Then

$$
J _ { 2 } - J _ { 1 } = \left( C _ { 2 } - C _ { 1 } \right) - ( M _ { 2 } - M _ { 1 } ) ,\tag{9}
$$

so a revision helps a frozen model only when its ceiling gain exceeds the added mismatch. In the presentation study, the ceiling is fixed, so each increase in mismatch appears as an equal loss in return. Table 1 reports both for each presentation.

## 5.3 WHEN ADAPTATION MAKES A REVISION BENEFICIAL

When a revision raises the ceiling but lowers the frozen model’s return, adapting to it can realize the gain, since every unit of mismatch removed adds a unit of return. Write $J ( \bar { \boldsymbol { \theta } } , \mathcal { T } ) = J _ { \mathcal { T } } ( \Pi _ { \boldsymbol { \theta } , \mathcal { T } } )$

Corollary 5.2 (Adaptation after a harmful revision). Let $\theta _ { 0 }$ be the source checkpoint, trained under $\mathcal { T } _ { 1 }$ , let $\theta _ { 1 }$ be the checkpoint adapted from it under $\mathcal { T } _ { 2 } ,$ , and let $\Delta C = C _ { 2 } - C _ { 1 }$ . Suppose the revision harms thefrozen source checkpoint, ${ \mathrm { \Delta } } \dot { J } ( \theta _ { 0 } , { \mathcal { T } } _ { 2 } ) < J ( \theta _ { 0 } , { \mathcal { T } } _ { 1 } )$ . The adapted pair outperforms the source pair, $J ( \theta _ { 1 } , \mathcal { T } _ { 2 } ) > J ( \theta _ { 0 } , \mathcal { T } _ { 1 } )$ , ifand only if

$$
M ( \theta _ { 1 } , \mathcal { Z } _ { 2 } ) < C _ { 2 } - J ( \theta _ { 0 } , \mathcal { Z } _ { 1 } ) = \Delta C + M ( \theta _ { 0 } , \mathcal { Z } _ { 1 } ) .\tag{10}
$$

The threshold $T = \Delta C + M ( \theta _ { 0 } , \mathcal { T } _ { 1 } )$ is the new ceiling minus the source pair’s return. In words, adaptation pays off once the mismatch left under the new harness is smaller than the ceiling gain plus the mismatch that the source pair already had. Three comparisons describe the outcome. Recovery compares the adapted pair with the source pair, harness ranking compares the two harnesses at the adapted checkpoint, and gain over continuation compares adaptation with continued training under the source harness at the same budget, which tests whether the benefit comes from the revision or from further training. Appendix D.6 proves the corollary, and Appendix D.5 illustrates reversal and adaptation in a two-action example.

Batched clicks. From the H0-trained checkpoint, we create two branches with identical weights, update budgets, and task draws. One continues training under H0, and the other adapts to H1, which adds batched clicks. H1 raises the ceiling slightly, but the frozen checkpoint carries its H0 preferences into H1 and adds far more mismatch, so return and task success fall. Adaptation under H1 removes most of this mismatch, while the H0 branch, which never trains under H1, keeps it. During adaptation, the checkpoint’s return under H1 rises and its return under H0 falls. Its ranking of the two harnesse therefore reverses first, and only later does it beat the source pair. At the end, the adapted pair beats both the source pair and the H0 branch in task success and in return, and the paired intervals of these differences exclude zero. Figure 4 shows both branches in its first panel, and Appendix F.3 gives the endpoint measurements.

Across harness swaps. Batched clicks are one of many possible swaps among H0–H5. A swap moves the checkpoint post-trained under a source harness to a different target harness. We adapt every swap that a frozen comparison rejects and whose T is positive. When $\bar { T } \le 0$ , no policy under the target harness can beat the source pair, since mismatch is never negative. Nine swaps meet both conditions. After adaptation, the seven that raise the ceiling all beat the source pair, and the two that lower it gain nothing. Three of the seven switch to H5, which only adds instructions, so a revision that changes no action can still raise the ceiling through the reference it induces.

The frozen loss says little about this final gain. H0→H5 and H4→H3 lose the same small amount when frozen, yet H0→H5 ties for the largest gain and H4→H3 gains nothing. H0→H1 has the largest frozen loss and still gains. The frozen return change, $T - M ( \mathbf { \breve { { \boldsymbol { \theta } } } } _ { 0 } , \mathbf { \mathcal { T } } _ { 2 } )$ , and the gain, $T - M ( \theta _ { 1 } , \mathbf { \overline { { { Z } } } } _ { 2 } )$

(a) Return, mismatch, and ceiling  
![](images/abc5444a4d3a9db8c06c9e5e21a90c420d56cb934eecde5a62fa398bb91b6a13.jpg)  
Regularized return

(b) Frozen comparison  
![](images/92875756a765b39cac8c1b363e30ce8202590be840bb890511a4363b6cc5543d.jpg)  
Frozen return change

(c) Prediction before adaptation  
![](images/d88cc30394e9ceb9553c9e81932621679b0fa079c7bb275c7c5f7134ce539836.jpg)  
Predicted gain T − r<sub>2</sub>  
Figure 4: Harmful revisions become beneficial after adaptation when they raise the ceiling. (a) Return $^ { J , }$ ceiling C, and mismatch M of the H0-trained checkpoint, its H0 continuation, and its H1 adaptation under each harness. The dotted line marks the source pair’s return. Gains of the adapted pair over the source pair in the nine swaps that a frozen comparison rejects and that have $T { \bar { > } } 0 .$ plotted against (b) the frozen return change and (c) the predicted gain $T - r _ { 2 }$ . In (b) and (c), dark points raise the ceiling, light points lower it, and the red point is H0→H1.

differ only in the mismatch under the target harness, the target mismatch. Before adaptation, the target mismatch reflects preferences learned under the source harness and varies widely across swaps. Adaptation removes most of it, so the gain follows T, which differs across swaps mainly through the ceiling gain.

Ceiling and mismatch can therefore predict each gain before adaptation. Let $r _ { 2 }$ be the mismatch of the checkpoint post-trained directly under the target harness, measured under that harness. If adaptation brings the target mismatch down to this same level, the gain is $T - r _ { 2 }$ . This prediction needs only the oracle ceilings and the six post-training runs of Section 3.2. We recorded it before adaptation for every swap except H0→H1. It gets the sign of every gain right and orders the eight recorded gains with Spearman $\rho = 0 . 9 5$ , while the frozen return change orders them with $\rho = 0 . 1 7$ Figure 4 plots the gains against the frozen return change in its second panel and against $T - r _ { 2 }$ in its third, and Appendix F.4 lists all nine swaps with their rank correlations.

## Takeaways: What transfer costs

Deployed return is what the harness offers (the ceiling) minus what the policy leaves unrealized (the mismatch). A harness change moves both terms, and training moves only the mismatch. A frozen comparison therefore penalizes a new harness for mismatch that adaptation later removes. All seven rejected swaps that raise the ceiling outperform the source pair after adaptation.

## 6 MEASURING MISMATCH IN DEPLOYMENT

The results so far use full action distributions and a solved optimum. A deployed agent exposes only sampled actions and may have no known optimum, so we estimate mismatch from action samples, first against a known optimal policy and then against a teacher model in its place.

## 6.1 ESTIMATING MISMATCH FROM ACTION SAMPLES

Measuring mismatch requires the optimal policy, or a stand-in for it, and action samples that the deployed model draws through its unchanged interface at states on its own trajectories, since mismatch weights each decision by how often the model reaches it.

Pilot rollouts supply these states. Stopping each rollout at a random time t drawn with probability $( 1 - \gamma ) \gamma ^ { t }$ gives an exact draw from discounted occupancy. At each of m such states, we sample $\dot { N }$ actions through the deployed interface and smooth the counts into an action distribution $\widetilde { p } ( \cdot \mid s )$ . The mismatch estimate averages the KL divergence over the sampled states:

$$
\widehat { M } = \frac { \tau } { ( 1 - \gamma ) m } \sum _ { i = 1 } ^ { m } \mathrm { K L } ( \widetilde { p } ( \cdot  { | } s _ { i } )  { | } | \Pi _ { \cal Z } ^ { * } ( \cdot  { | } s _ { i } ) ) .\tag{11}
$$

![](images/f6beab2c566ce1d240caff0cfdff4a0c3b40e79119e80ed4b68836a68f3a5d55.jpg)

Figure 5: Sampled mismatch tracks the small return differences that decide adaptation. Curves show exact values and the two predictions at each saved H1-adapted checkpoint. (a) Recovery, the checkpoint’s return under H1 minus the source H0 pair’s return. (b) Harness ranking, the checkpoint’s return under H1 minus its return under H0.

A second route needs the probabilities of only the queried actions. The scaled log odds between two actions, τ log $[ \Pi _ { \theta , { \mathcal { T } } } ( a \mid s ) ^ { \top } \Pi _ { \theta , { \mathcal { T } } } ( b \mid s ) ]$ ], equal the difference of their scores in the model’s readout. If the queried pairs link every action to every other through a chain of pairs, these differences determine all centered scores. Comparing them with the optimal scores gives the centered score error, which bounds the mismatch.

Proposition 6.1 (From score error to mismatch). Let $x _ { s } ~ = ~ { \widehat F } _ { \theta , \tt T } ( s , \cdot ) - { \bar { F } } _ { \tt T } ^ { * } ( s , \cdot )$ and $\varepsilon _ { d } ^ { 2 } \ =$ $\mathbb { E } _ { d _ { \mathcal { T } } ^ { \Pi _ { \theta , \tau } } } \left\| x _ { s } \right\| _ { 2 } ^ { 2 } .$ . For positive semantic action probabilities,

$$
M ( \theta , \mathcal { T } ) \leq \frac { \varepsilon _ { d } ^ { 2 } } { 4 \tau ( 1 - \gamma ) } .\tag{12}
$$

Once this bound falls below the threshold in Corollary 5.2, the adapted pair is guaranteed to outperform the source pair. Appendix E gives the proof and a lower bound on mismatch. Section 6.2 tests the sampled estimate Mc against a known optimum, and Section 6.3 applies the score-error bound with a teacher model in the optimum’s place.

## 6.2 FINITE MEASUREMENTS PREDICT RETURN CHANGES

Section 5 computed every mismatch exactly. We now ask whether the sampled estimate $\widehat { M }$ of Eq. (11) can replace it, both for large return changes and for the small differences that decide whether adaptation pays off. An oracle supplies the ceiling and the optimal action distributions, and the deployed model is accessed only through sampled actions. The predicted return change of a candidate relative to its baseline is $\widehat { \Delta J } = \Delta C - \Delta \widehat { M }$ , and exact returns on tasks and configurations held out from calibrating the estimator serve as ground truth. We compare this prediction with two baselines at the same query budget. The ceiling alone sets $\Delta \widehat { M } = 0 .$ A frequency rollout estimates each return directly by running the policy and estimating each selected action’s log probability from repeated queries.

Large changes across configurations. We deploy the final H0-trained and H1-adapted checkpoints through H0’s native actions and vary one aspect of the deployment at a time, namely how reliably a click executes, what a click costs, or how the prompt is presented. We predict each configuration’s return change relative to a fixed baseline configuration. The ceiling alone fails, because presentation changes move the mismatch and leave the ceiling fixed. Subtracting the sampled mismatch reduces the prediction error about a hundredfold and gets the direction of every change not close to zero. The frequency rollout is about as accurate, so on large changes the two estimators perform alike.

Small differences during adaptation. Recovery and harness ranking from Section 5.3 are differences of a few hundredths in return, and their signs decide whether adaptation pays off. We predict both at each saved H1-adapted checkpoint. Here the mismatch-based prediction is about ten times more accurate than the frequency rollout. It takes the ceiling difference from the oracle and estimates only the mismatch, whereas the rollout estimates the full return of each pair. The predicted ranking reverses at the same checkpoint as the observed one, and the predicted recovery comes one saved checkpoint later. The frequency rollout overestimates both differences and predicts the reversal too early. Figure 5 shows both predictions, and Appendices F.5 and F.6 give the sampling budget and all numerical results.

## 6.3 CEILING AND MISMATCH OF CODING-AGENT HARNESSES

We deploy a frozen Qwen3.5-9B through the native configurations of five coding-agent harnesses: DeepAgents, Pi coding-agent, DeepSeek Harness, Terminus-2, and mini-SWE-agent. With the model fixed, the harness moves Terminal-Bench Lite accuracy from 47.9% to 64.1%. We ask whether the ceiling and the mismatch explain this ordering.

Table 2: Ceiling and mismatch of five coding-agent harnesses for a frozen Qwen3.5-9B. Ce and Mf are teacher-based estimates. The last row gives each column’s Spearman correlation with accuracy.
<table><tr><td>Harness</td><td>č M č -M</td><td>Accuracy (%)</td></tr><tr><td>Terminus-2</td><td>0.431 0.134</td><td>0.297  $6 4 . 1 \pm 5 . 1$ </td></tr><tr><td>mini-SWE-agent</td><td>0.631 0.335</td><td>0.296  $5 9 . 8 \pm 7 . 8$ </td></tr><tr><td>DeepSeek Harness</td><td>0.590 0.382</td><td>0.208  $5 4 . 7 \pm 3 . 9$ </td></tr><tr><td>Pi coding-agent</td><td>0.383 0.206</td><td>0.177  $5 1 . 3 \pm 2 . 6$ </td></tr><tr><td>DeepAgents</td><td>0.261 0.429</td><td>-0.168  $4 7 . 9 \pm 3 . 9$ </td></tr><tr><td>Spearman ρ</td><td>0.7 -0.7</td><td>1.0</td></tr></table>

![](images/1122dbb115266544c956c65c563b217c6dd9e6935c799b75e1ac35e6c923dfb7.jpg)  
(b) Mismatch accumulates over decisions<sub>Figure 6: Mismatch accumulates over decisions.</sub> Each harness’s mismatch splits into the number of decisions the evaluated model makes per trajectory and the mismatch per decision. Dashed curves have constant Mf.

Teacher-based estimates. Coding agents have open action spaces and no solvable optimum, so a stronger teacher model, GPT-5.6- luna, takes the optimum’s place. Below, the evaluated model means the frozen Qwen3.5-9B. On three Terminal-Bench Lite tasks, we record every decision of the evaluated model, sample outputs from it and from the teacher through the unchanged harness, and map each output to an action category shared by all harnesses, such as inspecting files, modifying them, running commands, or planning. We use the evaluated model’s own category distribution as the reference. Its KL penalty is then zero, so its regularized return equals its success probability, and by Eq. (8) so does its exact ceiling minus mismatch. We estimate the ceiling by the teacher’s regularized return, ${ \widetilde { C } } .$ . We estimate the mismatch, ${ \widetilde { M } } .$ , with the bound of Proposition 6.1, using the teacher’s log odds in place of the optimal scores and summing along the evaluated model’s trajectories. Because the teacher is not optimal, we test whether the estimates order the harnesses by accuracy. Appendix F.7 gives the protocol and estimators.

$\widetilde { C } - \widetilde { M }$ orders the harnesses. Computed without the evaluated model’s task outcomes, $\widetilde { C } - \widetilde { M }$ orders all five harnesses as accuracy does. Either term alone misorders them. The ceiling estimate alone ranks mini-SWE-agent first, and the mismatch estimate alone ranks Pi coding-

agent second, while their difference ranks both correctly. Terminus-2 has a middle ceiling and leads through the smallest mismatch. DeepSeek Harness and mini-SWE-agent have the two highest ceilings and large mismatch, the profile of the batched-click harness before adaptation. Table 2 lists the estimates.

The same pattern appears when both terms are exact. On MiniWoB++, where they can be computed, we let a frozen Qwen3.5-9B serve as its own reference under H0–H5. The ceiling alone then ranks the most successful harness last, and the mismatch alone ranks H0 below two less successful harnesses, while only their difference orders all six correctly. Appendix F.8 gives this comparison.

Where mismatch comes from. By Proposition 5.1, mismatch accumulates along the model’s own trajectories, so it grows with the number of decisions a harness asks of the model. DeepAgents has a mismatch per decision close to that of Terminus-2, but the evaluated model makes nearly three times as many model calls under it and accumulates the largest mismatch. DeepSeek Harness has the largest mismatch per decision. Under it, the teacher plans much more often than the evaluated model, and the evaluated model inspects the environment much more often. Figure 6 separates the two factors for each harness.

## 7 DISCUSSION AND CONCLUSION

A policy trained under a harness learns the harness’s relative action scores and expresses them through its interface. Post-training makes these preferences harness-specific, and a new interface can change the model’s choices even when every optimal decision stays the same. Because deployed return is the ceiling minus the mismatch, a harness that offers more can hurt a trained model and become the best choice after adaptation. Mismatch estimated from sampled actions predicts these changes when the optimum is known, and for coding agents, teacher-based estimates of ceiling minus mismatch order harnesses by accuracy. Harness evaluations should therefore report the ceiling next to the frozen return, evaluate several checkpoints under each harness rather than a single one, and check how presentation changes move the model’s choices. Harness distillation and joint adaptation of harnesses and weights (Ye et al., 2026; Chen et al., 2026; Kim et al., 2026) can use sampled-action measurements to track the ceiling and the mismatch separately. Appendix A discusses related work.

## REFERENCES

Huan ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Qihan Ren, Cheng Qian, Zhenhailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, and Mengdi Wang. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. Transactions on Machine Learning Research, 2026. URL https://arxiv. org/abs/2507.21046.

Haoyang Cao, Samuel N. Cohen, and Lukasz Szpruch. Identifiability in inverse reinforcement learning. In Advances in Neural Information Processing Systems, volume 34, pp. 12362–12373, 2021. URL https://arxiv.org/abs/2106.03498.

Zhengyu Chen, Teng Xiao, Huaisheng Zhu, Yige Yuan, Luan Zhang, and Jingang Wang. Co-Harness: Co-evolving harnesses and model weights for LLM agents. arXiv preprint arXiv:2607.22688, 2026. URL https://arxiv.org/abs/2607.22688.

Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, Zhaochun Ren, Nikos Aletras, Xi Wang, Han Zhou, and Zaiqiao Meng. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems, 2025. URL https://arxiv.org/abs/2508.07407.

Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. Promptbreeder: Self-referential self-improvement via prompt evolution, 2023. URL https: //arxiv.org/abs/2309.16797.

Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized Markov decision processes. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2160–2169. PMLR, 2019. URL https: //proceedings.mlr.press/v97/geist19a.html.

Adam Gleave, Michael Dennis, Shane Legg, Stuart Russell, and Jan Leike. Quantifying differences in reward functions, 2021. URL https://arxiv.org/abs/2006.13900.

Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies, 2017. URL https://arxiv.org/abs/1702.08165.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, 2018. URL https: //arxiv.org/abs/1801.01290.

Prannay Hebbar, Yogendra Manawat, Samuel Verboomen, Alesia Ivanova, Selvam Palanimalai, Kunal Bhatia, and Vignesh Baskaran. SIA: Self improving AI with harness & weight updates. arXiv preprint arXiv:2605.27276, 2026. URL https://arxiv.org/abs/2605.27276.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2408. 08435.

Zixuan Ke, Austin Xu, Yifei Ming, Xuan-Phi Nguyen, Caiming Xiong, and Shafiq Joty. MAS-ZERO: Designing multi-agent systems with zero supervision. In NeurIPS 2025 Workshop on Scaling Environments for Agents, 2025. URL https://openreview.net/forum?id= j5VXzWyoyW.

Haechan Kim, Yoonho Lee, Gisang Lee, Chelsea Finn, and Kangwook Lee. WHALE: A simple recipe for joint harness-weight optimization. arXiv preprint arXiv:2609.00196, 2026. URL https://arxiv.org/abs/2609.00196.

Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018. URL https://arxiv.org/abs/1805.00909.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. In International Conference on Learning Representations, 2018. URL https://arxiv.org/abs/1802.08802.

Jian Mu, Chaoyun Zhang, Chiming Ni, Lu Wang, Bo Qiao, Kartik Mathur, Qianhui Wu, Yuhang Xie, Xiaojun Ma, Mengyu Zhou, Si Qin, Liqun Li, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. GUI-360<sup>◦</sup>: A comprehensive dataset and benchmark for computerusing agents. arXiv preprint arXiv:2511.04307, 2025. URL https://arxiv.org/abs/ 2511.04307.

Andrew Y. Ng and Stuart J. Russell. Algorithms for inverse reinforcement learning. In Proceedings of the Seventeenth International Conference on Machine Learning (ICML 2000), pp. 663–670. Morgan Kaufmann, 2000.

Andrew Y. Ng, Daishi Harada, and Stuart J. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999), pp. 278–287. Morgan Kaufmann, 1999.

Brendan O’Donoghue. On the connection between Bregman divergence and value in regularized Markov decision processes. arXiv preprint arXiv:2210.12160, 2022. URL https://arxiv. org/abs/2210.12160.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2305. 18290.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From r to Q<sup>∗</sup>: Your language model is secretly a Q-function. In Conference on Language Modeling, 2024. URL https://arxiv. org/abs/2404.12358.

Maxime Robeyns, Martin Szummer, and Laurence Aitchison. A self-improving coding agent, 2025. URL https://arxiv.org/abs/2504.15228.

Stuart Russell. Learning Agents for Uncertain Environments (Extended Abstract). In Annual Conference on Computational Learning Theory, pp. 101–103, 1998. doi: 10.1145/279943.279964. URL https://mlanthology.org/colt/1998/russell1998colt-learning/.

Andreas Schlaginhaufen and Maryam Kamgarpour. Towards the transferability of rewards recovered via regularized inverse reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pp. 21461–21501, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/2628d4d3b054c2d7ad33ab03435204f4-Abstract-Conference.html.

John Schulman, Xi Chen, and Pieter Abbeel. Equivalence between policy gradients and soft Qlearning. arXiv preprint arXiv:1704.06440, 2017. URL https://arxiv.org/abs/1704. 06440.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models’ sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting. In International Conference on Learning Representations, 2024. URL https://arxiv.org/ abs/2310.11324.

Joar Skalse, Matthew Farrugia-Roberts, Stuart Russell, Alessandro Abate, and Adam Gleave. Invariance in policy optimisation and partial identifiability in reward learning. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. URL https: //proceedings.mlr.press/v202/skalse23a.html.

Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context. arXiv preprint arXiv:2209.15189, 2022. URL https://arxiv.org/abs/2209.15189.

Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J. Andrew Bagnell. All roads lead to likelihood: The value of reinforcement learning in fine-tuning. arXiv preprint arXiv:2503.01067, 2025. URL https://arxiv.org/abs/2503.01067.

Hongliang Wei, Xiaobing Tu, Yinggui Wang, Zhengxi Liu, Rongkun Xue, Jinkui Ren, Xiantao Zhang, Debin Zhao, and Xiaopeng Fan. HarnessBandit: Joint learnability-transferability scheduling for multi-harness agentic reinforcement learning. arXiv preprint arXiv:2609.13739, 2026. URL https://arxiv.org/abs/2609.13739.

Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen Arandjelovic, and Jakob Nikolas Kather. LLM agents making agent tools. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26092–26130, Vienna, Austria, July 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.1266. URL https://aclanthology.org/2025.acl-long.1266/.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/ abs/2405.15793.

Haoran Ye, Yuxing Lu, Haonan Dong, Zhaochen Su, and Guojie Song. Harness-Zero: Harness distillation via agent-as-harness. arXiv preprint arXiv:2609.24974, 2026. URL https://arxiv. org/abs/2609.24974.

Zhou Yu, Bin Bi, Shiva Kumar Pentyala, Shubham Mehrotra, Sougata Chaudhuri, Shilpa Bhagavath, Zeyuan Chen, Ran Xu, Phil Mui, James Zhu, and Sitaram Asur. Co-evolving harnesses and models: On-policy correction helps weaker models catch up where imitation fails. arXiv preprint arXiv:2609.09134, 2026. URL https://arxiv.org/abs/2609.09134.

Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. UFO2: The desktop agentOS. arXiv preprint arXiv:2504.14603, 2025a. URL https://arxiv.org/ abs/2504.14603.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin gödel machine: Openended evolution of self-improving agents. In The International Conference on Learning Representations (ICLR), 2026. URL https://iclr.cc/virtual/2026/poster/10007327.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. AFlow: Automating agentic workflow generation. In International Conference on Learning Representations, 2025b. URL https://arxiv.org/abs/2410.10762.

Regularized control and decision geometry. Control as inference and RL with a maximum entropy objective express optimal policies through rewards, action values, and reference distributions (Levine, 2018; Haarnoja et al., 2017; 2018). Regularized control supplies the Gibbs policy and the geometry of the value gap (Geist et al., 2019; O’Donoghue, 2022). The relation between policy gradients and soft Q-learning provides another view of these coordinates (Schulman et al., 2017). DPO and its sequential extension relate language-model policies to rewards and action values during post-training (Rafailov et al., 2023; 2024).

Our optimal action score combines the optimal semantic action value with the semantic reference prior. These coordinates let us compare the decisions induced by different interfaces and measure a deployed policy’s gap to each optimum. The analysis of fine-tuning objectives by Swamy et al. (2025) likewise uses a common geometric description to distinguish ideal equivalence from differences that emerge in practice. Here the distinction concerns the interface’s optimal decisions and the choices that a particular checkpoint makes through it.

Behavioral identification and reward transfer. Inverse RL recovers rewards from behavior subject to ambiguities (Ng & Russell, 2000; Russell, 1998). Reward shaping using a potential function preserves optimal policies (Ng et al., 1999). Analyses of identifiability under entropy regularization and of partial identifiability characterize the corresponding reward classes (Cao et al., 2021; Skalse et al., 2023). EPIC compares reward functions while accounting for transformations that preserve policies (Gleave et al., 2021). Work on the transferability of rewards recovered through regularized inverse RL further studies changes in transition laws (Schlaginhaufen & Kamgarpour, 2024).

Our equivalence class for a single interface removes statewise constants from the scores. Appendix C distinguishes this behavioral object from the nonunique decomposition into prior, reward, and transition. The transfer analysis then compares the target energy with the actual deployed readout, weighting their discrepancy by deployment occupancy.

Interface and agent design. Interface design and prompt formatting affect agent behavior (Yang et al., 2024; Sclar et al., 2024; Zhang et al., 2025a). GUI-360<sup>◦</sup> reports higher action prediction accuracy for GPT-4o and lower accuracy for fine-tuned Qwen after adding accessibility metadata and replacing coordinate arguments with element IDs (Mu et al., 2025). ADAS and AFlow search over agent implementations and workflows (Hu et al., 2025; Zhang et al., 2025b). Darwin Gödel Machine and Self-Improving Coding Agent revise scaffolds and assess them through execution (Zhang et al., 2026; Robeyns et al., 2025). Promptbreeder, ToolMaker, and MAS-ZERO respectively explore prompts, tools, and designs involving multiple agents (Fernando et al., 2023; Wölflein et al., 2025; Ke et al., 2025). Surveys place these approaches within agent systems with mechanisms for self improvement (Fang et al., 2025; ang Gao et al., 2026).

These design procedures can change action opportunities, realizations, or both. Our analysis separates each candidate’s optimal return from the evaluated checkpoint’s gap to that optimum. This distinction connects execution feedback to two possible sources of improvement: a more useful decision problem and a policy that better realizes its available decisions.

Context and harness distillation. Context distillation trains a model to reproduce, without the prompt, the behavior that the prompt elicits (Snell et al., 2022). Harness-Zero distills a specialized harness into the weights by letting a harnessing agent correct the student’s responses in the target harness’s action space (Ye et al., 2026). Our identification result states which decision information such training can carry, the relative action scores of the training harness. The readout study shows that the learned preferences are still expressed through the deployment interface, so a distilled policy can change its choices when the interface changes.

Joint adaptation of harnesses and model weights. Joint adaptation takes several forms: SIA selects harness and weight updates, Co-Harness trains from trajectories collected under revised interfaces, and WHALE alternates training with harness search (Hebbar et al., 2026; Chen et al., 2026; Kim et al., 2026). Yu et al. (2026) find that imitating expert trajectories can hurt a weaker model under an evolved harness while on-policy correction helps, and HarnessBandit schedules training across harnesses by learnability and transferability (Wei et al., 2026).

These methods can use the ceiling and the mismatch to tell whether a candidate harness offers too little or is not yet used well. Our adaptation threshold specifies the mismatch reduction needed to make a revision with a higher ceiling beneficial. Eq. (24) isolates how a weight update changes a revision’s relative value. Comparisons across checkpoints and interfaces therefore track both recovery after a revision and changes in interface rankings. The controlled protocol measures this interaction separately from gains under either interface alone.

## B SEMANTIC CONTROL: OBJECTS, ABSTRACTION, AND OBJECTIVES

## B.1 NOTATION AND STANDING CONDITIONS

Table 3: Notation. Energies and vector operations are statewise unless an expectation is shown.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal { T } = ( c \tau , d _ { \mathcal { T } } , e _ { \mathcal { T } } )$ </td><td>Context encoder, semantic decoder, and fixed execution protocol.</td></tr><tr><td> $\pi _ { \boldsymbol { \theta } } , \Pi _ { \boldsymbol { \theta } , \boldsymbol { \mathcal { I } } }$ </td><td>Policy over token sequences and its executed semantic pushforward.</td></tr><tr><td> $P _ { \mathcal { T } } , r _ { \mathcal { T } } , \pi _ { \mathcal { T } } ^ { \mathrm { r e f } }$ </td><td>Induced dynamics, task reward including execution costs, and semantic reference prior.</td></tr><tr><td> $\rho _ { 0 } , \gamma , \tau$ </td><td>Initial state distribution, discount, and KL regularization coefficient.</td></tr><tr><td> $F _ { \mathcal { T } } ^ { * } , \ \bar { F } _ { \mathcal { T } } ^ { * }$ </td><td>Optimal action score and its centered representative.</td></tr><tr><td> $\widehat { F } _ { \theta , \mathcal { I } }$ </td><td>Readout, the centered action scores  $\tau P _ { \perp }$  log  $\Pi _ { \theta , \pmb { \tau } } .$ </td></tr><tr><td> $d _ { \mathcal { Z } } ^ { \Pi }$ </td><td>Normalized discounted state occupancy of the evaluated policy.</td></tr><tr><td> $C ( { \mathcal { T } } ) , M ( \theta , { \mathcal { T } } )$ </td><td>Ceiling (optimal regularized return) and mismatch.</td></tr><tr><td> $e _ { 1 } , g _ { 1 2 } , \eta _ { 1 2 }$ </td><td>Training residual, target shift, and readout drift.</td></tr><tr><td> $\Phi , D _ { \Phi }$ </td><td>Log partition function and its Bregman divergence.</td></tr></table>

The formal results use finite state and action spaces, bounded rewards, $0 \leq \gamma < 1 , \tau > 0 .$ , and a reference prior that is positive on every available action. The initial distribution, reward units, and regularization coefficient are shared in a comparison across interfaces. State/action alignments are fixed before the comparison. Results involving log $\Pi _ { \theta , \mathcal { I } }$ additionally require positive probabilities for the actual policy on the probed actions. The gap identity for a fixed interface itself allows zero policy probabilities, with $0 \log 0 = 0$

We use a reference checkpoint distinct from the evolving evaluated checkpoint. When its sequence distribution is $\pi _ { \mathrm { r e f } } .$ , a natural semantic reference is

$$
\pi _ { \mathcal { Z } } ^ { \mathrm { r e f } } ( a \mid s ) = \sum _ { y : d _ { \mathcal { Z } } ( y ; s ) = a } \pi _ { \mathrm { r e f } } ( y \mid c _ { \mathcal { Z } } ( s ) ) .\tag{13}
$$

Consequently a new encoding can change both the target induced by the reference and the evaluated model’s readout. Experiments that hold the target fixed must verify or explicitly control this reference.

## B.2 DECISION EPOCHS AND CONTROLLER STATE

A semantic state includes every variable needed to predict the state at the next decision: the task state, visible history or a sufficient summary, tool results, budget counters, and controller memory. An action denotes a fully specified operation, such as invoking a particular tool with particular arguments, or submitting a particular candidate. Two output strings belong to one semantic action only when the decoder and executor execute the same operation at that state.

After a semantic action is selected, $e \tau$ can execute a fixed tool, perform a deterministic check, or apply a stochastic transition independent of the policy. Its expected costs enter $r _ { \mathbb { Z } } .$ , and its outcomes enter $P _ { \mathcal { T } }$ . If a retry, verifier, or stopping rule calls the evaluated model for another decision, that call is another decision epoch in the expanded state space. This keeps the transition kernel fixed while the policy varies.

An invalid string is mapped to an explicit failure or recovery action. Similarly, when an empirical action vocabulary groups outputs into an “other” category, that category must have a defined execution interpretation. Discarding unmodeled outputs changes the policy being measured.

## B.3 WHEN TRAINING OVER TOKENS INDUCES SEMANTIC OPTIMALITY

Let $Y _ { a } ( s )$ be the set of complete output realizations decoded as action a. Write a sequence policy and reference as

$$
\begin{array} { r } { p ( \boldsymbol { y } \mid \boldsymbol { s } ) = p _ { A } ( \boldsymbol { a } \mid \boldsymbol { s } ) p ( \boldsymbol { y } \mid \boldsymbol { s } , \boldsymbol { a } ) , \qquad \rho ( \boldsymbol { y } \mid \boldsymbol { s } ) = \rho _ { A } ( \boldsymbol { a } \mid \boldsymbol { s } ) \rho ( \boldsymbol { y } \mid \boldsymbol { s } , \boldsymbol { a } ) , \quad \boldsymbol { y } \in Y _ { \boldsymbol { a } } ( \boldsymbol { s } ) . } \end{array}
$$

For any such distributions, the KL chain rule gives

$$
\mathrm { K L } ( p ( \cdot  { | } s ) | | \rho ( \cdot  { | } s ) ) = \mathrm { K L } ( p _ { A } ( \cdot  { | } s ) | | \rho _ { A } ( \cdot  { | } s ) ) + \mathbb { E } _ { a \sim p _ { A } } \mathrm { K L } ( p ( \cdot  { | } s , a ) | | \rho ( \cdot  { | } s , a ) ) .\tag{14}
$$

Suppose reward, transition, and discount are identical within each fiber. Then the return from choosing y depends only on a, while the second term in Eq. (14) is minimized by retaining the reference conditional distribution inside that fiber. Optimization over complete sequence actions therefore reduces exactly to the semantic control problem regularized by KL divergence relative to $\rho _ { A }$

For an autoregressive model, probabilities of complete sequences include termination. The expected sum of token KL divergences equals sequence KL when the same terminating sequence measure and matching token conditionals are used. Rewards that depend on tokens, discounts that depend on sequence length, or additional model decisions within a proposed temporally extended action require the expanded model above.

## C BEHAVIORAL IDENTIFICATION AND MECHANISM AMBIGUITY

## C.1 SOFT BELLMAN OPTIMALITY

For a fixed interface, the optimal Bellman operator is

$$
( T V ) ( s ) = \tau \log \sum _ { a } \pi ^ { \mathrm { r e f } } ( a \mid s ) \exp \left( \frac { r ( s , a ) + \gamma \mathbb { E } _ { P ( \cdot \mid s , a ) } V ( s ^ { \prime } ) } { \tau } \right) .
$$

Monotonicity of the log partition function and its translation identity imply $\| T V - T W \| _ { \infty } \ \leq$ $\gamma \| V - W \| _ { \infty }$ . Thus it has a unique fixed point $V ^ { * }$ , and $Q ^ { * } = r + \gamma { \dot { P } } V ^ { * }$ is uniquely determined. At each state, maximizing $\begin{array} { r } { \sum _ { a } \pi _ { a } \overset { \cdot } { Q } _ { a } ^ { * } - \tau \sum _ { a } \pi _ { a } \log ( \pi _ { a } / \rho _ { a } ) } \end{array}$ over the simplex gives

$$
\pi _ { a } ^ { * } = \frac { \rho _ { a } e ^ { Q _ { a } ^ { * } / \tau } } { \sum _ { b } \rho _ { b } e ^ { Q _ { b } ^ { * } / \tau } } .
$$

Strict concavity gives uniqueness. At the fixed point, $\begin{array} { r } { V ^ { * } ( s ) = \tau \log \sum _ { b } \rho _ { b } e ^ { Q _ { b } ^ { * } / \tau } } \end{array}$ , so the optimal policy takes the Gibbs form

$$
\Pi ^ { * } ( a \mid s ) = \pi ^ { \operatorname { r e f } } ( a \mid s ) \exp \biggl ( \frac { Q ^ { * } ( s , a ) - V ^ { * } ( s ) } { \tau } \biggr ) .\tag{15}
$$

Substituting $F _ { a } ^ { * } = Q _ { a } ^ { * } + \tau \log \rho _ { a }$ produces the softmax representation in the main text.

## C.2 PROOF OF PROPOSITION 3.1

If $F ( s , \cdot ) = G ( s , \cdot ) + c ( s ) \mathbf { 1 }$ , the factor $e ^ { c ( s ) / \tau }$ cancels in softmax. Conversely, equal softmax probabilities imply

$$
F ( s , a ) - \Phi ( F ( s , \cdot ) ) = G ( s , a ) - \Phi ( G ( s , \cdot ) )
$$

for every action. Their difference is therefore constant across actions. Applying this equivalence to optimal action scores proves the first assertion. Under $\Pi _ { \theta , \mathcal { T } _ { \mathrm { t r } } } = \Pi _ { \mathcal { T } _ { \mathrm { t r } } } ^ { * }$ , the Gibbs identity gives

$$
\tau \log \Pi _ { \theta , \mathcal { Z } _ { \mathrm { t r } } } ( a \mid s ) = F _ { \mathcal { Z } _ { \mathrm { t r } } } ^ { * } ( s , a ) - V _ { \mathcal { Z } _ { \mathrm { t r } } } ^ { * } ( s ) .
$$

Projection onto $\mathbf { 1 } ^ { \perp }$ , or subtraction between two actions, removes $V _ { \mathcal { T } _ { \mathrm { t r } } } ^ { * } ( s )$ and proves Eq. (4).

## C.3 MECHANISM NONIDENTIFIABILITY

Proposition C.1 (Mechanism nonidentifiability). A fixed optimal score field can be realized by a continuous family of different reference priors and rewards, and even by different transition laws. Consequently, optimal behavior alone does not uniquely determine the interface mechanism.

Proof. Fix a finite score field F, any reference $\rho ^ { \prime }$ with full support, and any stochastic kernel $P ^ { \prime }$ Define

$$
V ( s ) = \Phi ( F ( s , \cdot ) ) , \qquad Q ^ { \prime } ( s , a ) = F ( s , a ) - \tau \log \rho ^ { \prime } ( a \mid s ) , \qquad r ^ { \prime } = Q ^ { \prime } - \gamma P ^ { \prime } V .
$$

Then $Q ^ { \prime } = r ^ { \prime } + \gamma P ^ { \prime } V$ , and

$$
\tau \log \sum _ { a } \rho ^ { \prime } ( a \mid s ) e ^ { Q ^ { \prime } ( s , a ) / \tau } = \tau \log \sum _ { a } e ^ { F ( s , a ) / \tau } = V ( s ) .
$$

By uniqueness of the Bellman fixed point, these are the optimal values of the constructed control problem. Its optimal policy is softmax ${ ( F / \tau ) }$ . Varying $\rho ^ { \prime }$ or $P ^ { \prime }$ gives a continuous family of mechanisms with the same scores and policy. □

## D PERFORMANCE GEOMETRY AND READOUT ACROSS INTERFACES

## D.1 PROOF OF PROPOSITION 5.1

Fix an interface and suppress its subscript. For a policy Π, write

$$
( \mathcal T ^ { \Pi } V ) ( s ) = \mathbb E _ { a \sim \Pi ( \cdot \vert s ) } \left[ r ( s , a ) + \gamma P V ( s , a ) - \tau \log \frac { \Pi ( a \mid s ) } { \pi ^ { \mathrm { r e f } } ( a \mid s ) } \right] .
$$

Rearranging the optimal-policy expression in Eq. (15) gives $Q ^ { * } - V ^ { * } = \tau \log ( \Pi ^ { * } / \pi ^ { \mathrm { r e f } } )$ . Substituting this relation into the regularized Bellman update yields the statewise gap:

$$
V ^ { * } ( s ) - ( \mathcal { T } ^ { \Pi } V ^ { * } ) ( s ) = \tau \mathrm { K L } ( \Pi ( \cdot \ | \ s ) | | \Pi ^ { * } ( \cdot \ | \ s ) ) = : k _ { \Pi } ( s ) .\tag{16}
$$

Subtracting the policy Bellman equation $V ^ { \Pi } = \mathcal { T } ^ { \Pi } V ^ { \Pi }$ yields

$$
V ^ { * } - V ^ { \mathrm { I I } } = k _ { \mathrm { I I } } + \gamma P ^ { \mathrm { I I } } ( V ^ { * } - V ^ { \mathrm { I I } } ) = ( I - \gamma P ^ { \mathrm { I I } } ) ^ { - 1 } k _ { \mathrm { I I } } .
$$

The normalized occupancy satisfies $( d ^ { \Pi } ) ^ { \top } = ( 1 - \gamma ) \rho _ { 0 } ^ { \top } ( I - \gamma P ^ { \Pi } ) ^ { - 1 }$ . Multiplication by $\rho _ { 0 } ^ { \top }$ proves Eq. (6), in which both the KL direction and the occupancy belong to the evaluated policy. □

## D.2 BREGMAN REPRESENTATION OF THE DECISION LOSS

For $\begin{array} { r } { \Phi ( F ) = \tau \log \sum _ { a } e ^ { F _ { a } / \tau } } \end{array}$ , the following identity relates score differences to the policy divergence. Lemma D.1 (Bregman–KL identity). For any score vectors F and G, the Bregman divergence generated by Φ satisfies

$$
D _ { \Phi } ( F , G ) = \Phi ( F ) - \Phi ( G ) - \langle \nabla \Phi ( G ) , F - G \rangle = \tau \mathrm { K L } \big ( \mathrm { s o f t m a x } ( G / \tau ) \| \mathrm { s o f t m a x } ( F / \tau ) \big ) .\tag{17}
$$

Proof. Let $p _ { F } = \mathrm { s o f t m a x } ( F / \tau )$ , so that $\nabla \Phi ( F ) = p _ { F }$ and τ log $p _ { F } = F - \Phi ( F ) \mathbf { 1 }$ . Then

$$
\begin{array} { l } { \tau \mathrm { K L } ( p _ { G } \| p _ { F } ) = \displaystyle \sum _ { a } ( p _ { G } ) _ { a } \big [ G _ { a } - \Phi ( G ) - F _ { a } + \Phi ( F ) \big ] } \\ { = \Phi ( F ) - \Phi ( G ) - \langle p _ { G } , F - G \rangle = D _ { \Phi } ( F , G ) . } \end{array}
$$

For any constants $c , d , \Phi ( F + c { \bf 1 } ) = \Phi ( F ) + c$ and $\nabla \Phi ( G + d { \bf 1 } ) = \nabla \Phi ( G )$ . These identities show $D _ { \Phi } ( \dot { F } + c { \bf 1 } , G + d { \bf 1 } ) = D _ { \Phi } ( F , G )$ . For a positive policy, let $F _ { \Pi } = \tau P _ { \perp }$ log Π. Substitution in Proposition 5.1 gives the performance gap in centered scores:

$$
J _ { \mathcal { T } } ( \Pi _ { \mathcal { T } } ^ { * } ) - J _ { \mathcal { T } } ( \Pi ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { \mathcal { T } } ^ { \Pi } } D _ { \Phi } ( F _ { \mathcal { T } } ^ { * } , F _ { \Pi } ) .\tag{18}
$$

For $\Pi = \Pi _ { \theta , \mathcal { T } }$ , its centered score vector is $\widehat { F } _ { \theta , \mathbb { Z } } ,$ giving the score form of mismatch through Eq. (7).

## D.3 GENERAL TRANSFER AND THE CASE WITH ZERO DRIFT

Eq. (5) follows by adding and subtracting the source target and source readout in the same centered coordinates. Substitution into the exact gap gives

$$
M ( \theta , 2 ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { 2 } ^ { \Pi _ { \theta , 2 } } } D _ { \Phi } \big ( \bar { F } _ { 2 } ^ { * } , \bar { F } _ { 1 } ^ { * } + e _ { 1 } + \eta _ { 1 2 } \big ) .\tag{19}
$$

For an optimal source policy whose readout is preserved by the swap, $e _ { 1 } = \eta _ { 1 2 } = 0$ , so

$$
M ( \theta , 2 ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { 2 } ^ { \Pi _ { \theta , 2 } } } D _ { \Phi } ( F _ { 2 } ^ { * } , F _ { 1 } ^ { * } ) \qquad ( e _ { 1 } = 0 , \eta _ { 1 2 } = 0 ) .\tag{20}
$$

The model continues to express the source preferences, whose discrepancy from the new optimum determines the deployment gap. For a quantitative perturbation bound, define $\varepsilon _ { g } , \varepsilon _ { e } , \varepsilon _ { \eta }$ as the respective root mean square norms of $g _ { 1 2 } , e _ { 1 } , \eta _ { 1 2 }$ under the same occupancy of the deployed policy in the target interface. Proposition 6.1 and Minkowski’s inequality give

$$
M ( \theta , 2 ) \leq \frac { ( \varepsilon _ { g } + \varepsilon _ { e } + \varepsilon _ { \eta } ) ^ { 2 } } { 4 \tau ( 1 - \gamma ) } .\tag{21}
$$

## D.4 CORRESPONDENCE CONTROLS READOUT DRIFT

To relate readout stability to model behavior, suppose T<sub>s</sub> bijectively matches output realizations of each semantic action across interfaces. Define the model’s sensitivity to this correspondence by

$$
\delta _ { \mathrm { t o k } } ( s ) = \operatorname* { s u p } _ { y } \left| \log \pi _ { \theta } ( T _ { s } y \mid c _ { 2 } ( s ) ) - \log \pi _ { \theta } ( y \mid c _ { 1 } ( s ) ) \right| .\tag{22}
$$

Proposition D.2 (Correspondence controls readout drift). For a bijection that preserves semantic actionfibers, withfinite $\delta _ { \mathrm { t o k } } ( s )$

$$
| \eta _ { 1 2 } ( s , a ) - \eta _ { 1 2 } ( s , b ) | \leq 2 \tau \delta _ { \mathrm { t o k } } ( s ) , \qquad \| \eta _ { 1 2 } ( s , \cdot ) \| _ { 2 } \leq \tau \sqrt { n } \delta _ { \mathrm { t o k } } ( s ) .\tag{23}
$$

Proof. At a fixed state, let $p _ { i } ( y ) = \pi _ { \theta } ( y \mid c _ { i } ( s ) )$ and $\delta = \delta _ { \mathrm { t o k } } ( s )$ . The bijection that preserves action fibers gives

$$
\Pi _ { \theta , 2 } ( a \mid s ) = \sum _ { y \in Y _ { a } ^ { ( 1 ) } ( s ) } p _ { 2 } ( T _ { s } y ) .
$$

Since $e ^ { - \delta } p _ { 1 } ( y ) \leq p _ { 2 } ( T _ { s } y ) \leq e ^ { \delta } p _ { 1 } ( y )$ , summing over the fiber yields

$$
e ^ { - \delta } \Pi _ { \theta , 1 } ( a \mid s ) \leq \Pi _ { \theta , 2 } ( a \mid s ) \leq e ^ { \delta } \Pi _ { \theta , 1 } ( a \mid s ) .
$$

Write v<sub>a</sub> = log $\Pi _ { \theta , 2 } ( a \mid s ) - \log \Pi _ { \theta , 1 } ( a \mid s )$ . Then $| v _ { a } | \leq \delta ,$ , and $\eta _ { 1 2 } = \tau P _ { \perp }$ v. Pairwise subtraction gives $| \eta _ { a } - \eta _ { b } | \leq 2 \tau \delta$ . Orthogonal projection is nonexpansive, so $\left. \eta _ { 1 2 } \right. _ { 2 } \leq \tau \left. v \right. _ { 2 } \leq \tau \sqrt { n } \delta$ □

## D.5 A TWO-ACTION ILLUSTRATION OF REVERSAL AND ADAPTATION

Consider a bandit with two actions and a uniform reference. The source execution protocol gives rewards (1, 0), and its optimal policy chooses the first action with probability $p = e ^ { 1 / \tau } / ( 1 + e ^ { 1 / \tau } )$ A revised protocol lowers the first action’s utility to 0 and raises the second action’s utility to $u > 0$ Holding the semantic policy fixed isolates the effect of this change in action values.

The two ceilings are

$$
C _ { 1 } = \tau \log \frac { e ^ { 1 / \tau } + 1 } { 2 } , \qquad C _ { 2 } = \tau \log \frac { 1 + e ^ { u / \tau } } { 2 } .
$$

Hence $C _ { 2 } > C _ { 1 }$ if and only if $u > 1$ . Holding the actual policy fixed also holds its reference KL fixed. Its returns satisfy

$$
J _ { 2 } - J _ { 1 } = u ( 1 - p ) - p .
$$

The policy gains u when it takes the second action and loses 1 when it takes the first. Its return therefore improves only when $u > p / ( 1 - p ) = e ^ { 1 / \tau }$ . Throughout $1 < u < e ^ { 1 / \tau }$ , the revised interface has a higher ceiling and a lower deployed return: the unchanged policy still favors the action preferred under the source optimum.

Because the source policy is optimal, Corollary 5.2 makes the revision beneficial once the remaining mismatch is smaller than $C _ { 2 } - C _ { 1 }$ . Reaching the new optimum realizes this entire gain, positive for every $u > 1$ . Figure 7 plots the frozen policy’s return change and the ceiling gain at $\tau = 0 . 5$ , with the vertical distance between the curves giving the mismatch under the revised interface. All quantities here are analytical.

![](images/97adca32613e22e2ae1fcb48c260aef5a594f8f343183f5a04d83fc331c9117d.jpg)  
Figure 7: Adaptation can realize an initially harmful interface improvement. Analytical curves for the two-action bandit with $\tau = 0 . 5$ . The frozen policy exceeds the original pair only for $u > e ^ { 2 }$ (dashed), while the ceiling gain is positive for $u > 1$ . Reaching the new optimum realizes the ceiling gain $C _ { 2 } - C _ { 1 }$ (solid). Corollary 5.2 gives the mismatch threshold for a benefit before full adaptation.

## D.6 POLICY ADAPTATION AND INTERFACE RANKINGS

Proof of Corollary 5.2. Keep the evaluation objective and reference fixed for each interface. Substituting $J ( \theta , \bar { \mathcal { T } _ { j } } ) = C _ { j } - \dot { M } ( \theta , \bar { \mathcal { T } _ { j } } )$ gives

$$
\begin{array} { r l } & { J ( \theta _ { 0 } , \mathcal { T } _ { 2 } ) - J ( \theta _ { 0 } , \mathcal { T } _ { 1 } ) = \Delta C + M ( \theta _ { 0 } , \mathcal { T } _ { 1 } ) - M ( \theta _ { 0 } , \mathcal { T } _ { 2 } ) , } \\ & { J ( \theta _ { 1 } , \mathcal { T } _ { 2 } ) - J ( \theta _ { 0 } , \mathcal { T } _ { 1 } ) = \Delta C + M ( \theta _ { 0 } , \mathcal { T } _ { 1 } ) - M ( \theta _ { 1 } , \mathcal { T } _ { 2 } ) . } \end{array}
$$

The initially harmful revision makes the first difference negative. Since $\Delta C + M ( \theta _ { 0 } , \mathcal { T } _ { 1 } ) = C _ { 2 } -$ $J ( \theta _ { 0 } , \mathcal { T } _ { 1 } )$ , the second is positive exactly under Eq. (10). Combining the initial harm and subsequent improvement gives the equivalent chain

$$
M ( \theta _ { 1 } , \mathcal { T } _ { 2 } ) < \Delta C + M ( \theta _ { 0 } , \mathcal { T } _ { 1 } ) < M ( \theta _ { 0 } , \mathcal { T } _ { 2 } ) .
$$

The source pair’s mismatch $M ( \theta _ { 0 } , { \mathcal { T } } _ { 1 } )$ allows for a source checkpoint that has not reached its optimum. For an optimal source policy, the threshold equals the ceiling gain $\Delta C$ , and reaching the new optimum realizes the entire gain. □

Interaction between weight and interface updates. For $j \in \{ 1 , 2 \}$ , define the gain from the weight update in interface j by

$$
A _ { j } = J ( \theta _ { 1 } , \mathscr { T } _ { j } ) - J ( \theta _ { 0 } , \mathscr { T } _ { j } ) = M ( \theta _ { 0 } , \mathscr { T } _ { j } ) - M ( \theta _ { 1 } , \mathscr { T } _ { j } ) .
$$

Write $\Delta J ( \theta ) = J ( \theta , \mathcal { T } _ { 2 } ) - J ( \theta , \mathcal { T } _ { 1 } )$ for the relative value of the revision. The interaction across the four combinations is

$$
\Omega = \Delta J ( \theta _ { 1 } ) - \Delta J ( \theta _ { 0 } ) = A _ { 2 } - A _ { 1 } .\tag{24}
$$

The ceiling terms cancel because the objectives are fixed across checkpoints. A positive Ω means the revision becomes relatively more valuable after the weight update. If $\Delta J ( \theta _ { 0 } ) < 0$ , its ranking reverses exactly when $\Omega > - \dot { \Delta } J ( \theta _ { 0 } )$ . This compares the two interfaces at the updated checkpoint, while Corollary 5.2 compares the adapted pair with the source pair. Reporting $A _ { 1 }$ and $A _ { 2 }$ also identifies the source of the interaction: deterioration under the source interface $( A _ { 1 } < 0 )$ can increase Ω even when $A _ { 2 } = 0$

## E MEASUREMENT THEORY

## E.1 CURVATURE ON THE DECISION QUOTIENT

For $p = \mathrm { s o f t m a x } ( F / \tau )$

$$
\nabla ^ { 2 } \Phi ( F ) = \frac { 1 } { \tau } \big ( \mathrm { d i a g } ( p ) - p p ^ { \top } \big ) .\tag{25}
$$

The categorical covariance annihilates 1. Its spectral norm is at most $1 / 2 { : }$ : the absolute row sum of row i is $\bar { 2 } p _ { i } ( 1 - p _ { i } ) \leq 1 / 2$ , and the matrix is symmetric. If $p _ { i } \geq p _ { \mathrm { m i n } }$ and $z \perp { \bf 1 }$ , then

$$
\begin{array} { l } { { \displaystyle z ^ { \top } \left( \mathrm { d i a g } ( p ) - p p ^ { \top } \right) z = \operatorname* { m i n } _ { c } \sum _ { i } p _ { i } ( z _ { i } - c ) ^ { 2 } } \ ~ } \\ { { \displaystyle \qquad \geq p _ { \mathrm { m i n } } \operatorname* { m i n } _ { c } \sum _ { i } ( z _ { i } - c ) ^ { 2 } = p _ { \mathrm { m i n } } \left\| z \right\| _ { 2 } ^ { 2 } } . } \end{array}
$$

Thus the upper Hessian constant is $1 / ( 2 \tau )$ and the lower constant on the centered subspace is $p _ { \mathrm { m i n } } / \tau$

## E.2 BOUNDS ON THE PERFORMANCE GAP

Proposition 6.1 bounds mismatch through centered score error. An upper bound below the threshold in Corollary 5.2 guarantees recovery of the source pair’s return. Under a condition on probability overlap, the same score error also provides a lower bound on mismatch. We state the complementary bound and prove both results below.

Corollary E.1 (Complementary lower bound). With $x _ { s }$ and $\varepsilon _ { d }$ as in Proposition $6 . I ,$ let $p _ { \operatorname* { m i n } } > 0$ be a uniform lower bound on action probabilities along the score interpolation between the actual and optimal policies. Then

$$
\frac { p _ { \operatorname* { m i n } } \varepsilon _ { d } ^ { 2 } } { 2 \tau ( 1 - \gamma ) } \leq M ( \theta , \mathcal { T } ) \leq \frac { \varepsilon _ { d } ^ { 2 } } { 4 \tau ( 1 - \gamma ) } .\tag{26}
$$

Proof of both bounds. At a fixed state, let $x = \widehat { F } - \bar { F } ^ { * }$ , with both arguments centered. Taylor’s integral remainder gives

$$
D _ { \Phi } ( \bar { F } ^ { * } , \widehat { F } ) = \int _ { 0 } ^ { 1 } ( 1 - t ) x ^ { \top } \nabla ^ { 2 } \Phi \big ( \widehat { F } + t ( \bar { F } ^ { * } - \widehat { F } ) \big ) x \mathrm { d } t .
$$

The curvature bounds above and $\textstyle \int _ { 0 } ^ { 1 } ( 1 - t ) \mathrm { d } t = 1 / 2$ imply

$$
\frac { p _ { \mathrm { m i n } } } { 2 \tau } \left\| x \right\| _ { 2 } ^ { 2 } \leq D _ { \Phi } ( \bar { F } ^ { * } , \widehat { F } ) \leq \frac { 1 } { 4 \tau } \left\| x \right\| _ { 2 } ^ { 2 } .
$$

Taking the actual occupancy expectation and dividing by $1 - \gamma$ proves both bounds.

A lower probability bound at both endpoint policies is sufficient along their score interpolation: the logarithm of softmax is concave in the score vector, so each interpolated log probability is at least the linear interpolation of its endpoint log probabilities.

The quadratic upper rate and constant are locally tight. For two actions around the uniform distribution, the categorical covariance has eigenvalue $1 / 2 \ : \mathrm { o n } \ : \mathbf { 1 } ^ { \perp }$ , so $D _ { \Phi } ( F , F + z ) = \| z \| _ { 2 } ^ { 2 } / ( 4 \tau ) + o ( \| z \| _ { 2 } ^ { 2 } )$ for centered z. No positive quadratic lower constant independent of the action probabilities is possible: near a policy concentrated on one action, the nonconstant curvature can approach zero.

## F EXPERIMENTAL DETAILS

We describe the experiments in the order of the main text: the MiniWoB++ setup and post-training study, the readout study, the H0-to-H1 adaptation study, the adaptation across harness swaps, the calibration of finite measurements, the independent prediction study, the coding-agent harness study, and the exact comparison of harness scores with task success.

## F.1 MINIWOB++ SETUP AND POST-TRAINING

The controlled study pairs language-model post-training with a finite control model of each harness. This gives every checkpoint a fixed optimal policy to compare against, while retaining task instructions, parameterized actions, and controller feedback in the model’s input.

We compare six harnesses, H0–H5, using H0 as the direct baseline.

Tasks and observations. We use three tasks from MiniWoB++ (Liu et al., 2018). In click-checkboxes, the agent toggles checkboxes to match the requested labels and submits the selection. In click-checkboxes-soft, the requested words and checkbox labels are matched by synonym meaning. In ascending-numbers, the agent clicks five tiles in numerical order, with the remaining numbers hidden after the first click. We retain the initially visible layout in the public observation, together with spatial tile IDs, the remaining tiles, and the number of completed clicks. Checkbox observations contain the instruction, each label and its checked status. All actor observations also expose the remaining decision budget and the harness’s controller state.

The reward is one for exact task completion and zero otherwise, including partial completion. All explicit action and tool costs are zero. Each episode allows at most ten actor decisions, with discount γ = 0.95 per decision. We compile the physical transitions, remaining budget, and controller memory into a finite acyclic graph. For every retained task, we verify a successful run in the native browser and separately compare the compiled transitions and termination rules with the page behavior. Training and exhaustive policy evaluation use these compiled graphs.

Harnesses. Table 4 specifies the six interfaces. The program verifier approves an action when it avoids an incorrect terminal outcome, using the public instruction, page state, and the benchmark’s fixed synonym vocabulary. Verification targets are checkbox submission and each of the five tile clicks. H4 uses a separate frozen Qwen3.5-2B reviewer whose prompt contains the public task and physical page state together with the proposed action, omitting the actor’s remaining budget and controller memory. It greedily selects between single-token approve/reject codes. For H3 and H4, verification feedback becomes part of the next observation and is cleared by a native action. The H1 batch executes two ordered clicks within one actor decision and stops at native termination, so it receives only one discount step. All ordered pairs of click IDs are available, including repeated IDs.

Table 4: MiniWoB++ harnesses. Every menu includes the complete parameters of each action. We measure the common budget in actor decisions.
<table><tr><td>ID</td><td>Interface</td><td>Actions, execution, and controller state</td></tr><tr><td>HO</td><td>Direct</td><td>One native click or checkbox submission per decision</td></tr><tr><td>H1</td><td>Batched clicks</td><td>Adds every ordered pair of clicks as a single menu option, executed without an intervening actor call.</td></tr><tr><td>H2</td><td>Automatic gate</td><td>Checks a commit action before execution. Rejection leaves the page unchanged, consumes the decision, and disables subsequent checks. Memory stores whether the gate is enabled.</td></tr><tr><td>H3</td><td>Verify tool</td><td>Adds an optional program check of a proposed native action. Memory stores the checked action and its advisory verdict.</td></tr><tr><td>H4</td><td>Reviewer tool</td><td>Uses H3&#x27;s action menu and memory, with advice supplied by a frozen language-model reviewer.</td></tr><tr><td>H5</td><td>Revised instruc- tions</td><td>Uses H0&#x27;s actions, observations, dynamics, and budget, with additional system instructions for matching labels, tracking tile IDs, correcting selections, and deciding when to submit.</td></tr></table>

Data splits. We generate tasks with the native pages and deduplicate them by full public task content before assigning splits. Table 5 gives the fixed counts. Checkbox instances contain fewer than five boxes, and in the number task the first required number lies in one of the first three spatial positions. The 72-task training probe is fixed before training and is a subset of the training set. The six harnesses share the same task contents and splits. Test tasks are held out from parameter updates.

All MiniWoB++ experiments reported in this paper use the same fixed test set of 576 tasks generated from MiniWoB++, with 192 tasks per family. Calibration and training-probe tasks remain separate.

Table 5: MiniWoB++ task allocation. Splits have disjoint public contents, except that the training probe comes from training. Metrics average tasks within each family and then weight the families equally.
<table><tr><td>Split</td><td>Per family</td><td>Total</td><td>Use in post-training</td></tr><tr><td>Training</td><td>480</td><td>1,440</td><td>Parameter updates</td></tr><tr><td>Calibration</td><td>96</td><td>288</td><td>Full learning curves</td></tr><tr><td>Training probe</td><td>24</td><td>72</td><td>Fixed training subset for learning curves</td></tr><tr><td>Test</td><td>192</td><td>576</td><td>Transfer evaluation</td></tr></table>

Table 6: Post-training settings. All harnesses share the same update and task-draw budgets, with computation depending on the number of reachable states and menu entries.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Base / reference / H4 re- viewer</td><td>Qwen3.5-2B with frozen base, reference, and reviewer weights</td></tr><tr><td>Trainable adapter</td><td>LoRA rank 16, scale α = 32, dropout 0, no bias updates</td></tr><tr><td rowspan="2">LoRA projection targets</td><td>Attention:  $\mathrm { q / k / v / o }$  _proj Linear attention:  $\mathrm { i n \_ p r o j \mathrm { \_ q k v , i n \mathrm { \_ p r o j \mathrm { \_ } z , i n \mathrm { \_ p r o j \mathrm { \_ b , } } } } }$ </td></tr><tr><td> $\mathrm { i n \_ p r o j \_ s u t \_ p r o j }$  MLP: gate/up/down_proj</td></tr><tr><td>Optimizer</td><td>AdamW with a constant learning rate  $1 0 ^ { - 5 } , \beta = ( 0 . 9 , 0 . 9 9 9 ) , \epsilon = 1 0 ^ { - 8 }$  weight decay 0, gradient norm clipping at 1</td></tr><tr><td>Training budget</td><td>1,024 updates per harness at four tasks per update (4,096 task draws)</td></tr><tr><td>Checkpoint schedule Numerical precision</td><td>Initialization and every 64 updates (17 checkpoints at the full budget)</td></tr><tr><td></td><td>BF16 for model forward passes FP32 for trainable parameters and optimizer states</td></tr><tr><td rowspan="2"></td><td>FP64 for categorical log-softmax and dynamic programming</td></tr><tr><td>Eight states grouped by prompt length and menu size</td></tr><tr><td>State microbatch Prompt length</td><td>At most 2,048 tokens without truncation</td></tr></table>

Policy readout and reference. The actor receives a system message specifying the task and harness rules and a JSON observation listing every available action. Each complete action, including its arguments, maps bijectively to one case-sensitive token from $\bar { \mathrm { A } } { - } \ m \mathrm { Z } / \bar { \mathrm { a } } { - } \ m \mathrm { Z }$ . We normalize the next-token logits over this full menu at sampling temperature one, using the model’s chat template with thinking disabled. This categorical distribution defines $\Pi _ { \theta , \pmb { \tau } _ { h } }$ throughout training and evaluation. The base model and tokenizer are Qwen3.5-2B. An independent frozen copy, without the trained adapter, reads the same messages to define $\pi _ { \mathcal { T } _ { h } } ^ { \mathrm { r e f } }$ at each state. The reference stays fixed across checkpoints. Because H5 changes the reference model’s input as well as the actor’s, we solve its regularized optimum separately from H0’s even though their execution dynamics coincide.

Training distribution and objective. Each harness starts from the same base weights and initial LoRA adapter (Hu et al., 2022). To retain correction states as successful behavior becomes more frequent, training uses

$$
\begin{array} { r } { \mu _ { h } = \frac { 1 } { 2 } \delta _ { s _ { 0 } , h } + \frac { 1 } { 2 } \nu _ { h } , \qquad J _ { \mu _ { h } } ( \theta ) = \mathbb { E } _ { \mu _ { h } , \Pi _ { \theta , T _ { h } } , P _ { h } } \displaystyle \sum _ { t \geq 0 } \gamma ^ { t } \left[ r _ { h } ( s _ { t } , a _ { t } ) - \tau \log \frac { \prod _ { \theta , T _ { h } } ( a _ { t } \mid s _ { t } ) } { \pi _ { T _ { h } } ^ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right] = } \end{array}\tag{27}
$$

with $\tau = 0 . 1$ . Here $s _ { 0 , h }$ is the native task start. We form $\nu _ { h }$ from all other reachable nonterminal states, stratifying by correction category or completion progress, remaining-budget bin, and controller memory. Checkbox categories are correct, missing selections only, extra selections only, and both missing and extra selections. For the number task, categories are defined by the number of completed clicks. Budget bins are $1 - 3 , 4 - 6 .$ , and 7–9 remaining decisions. We assign equal mass to nonempty strata and then to states within each stratum. Each start has a legal action prefix from the native state, and the objective begins at that start with its remaining budget. This gives every reachable decision state positive training mass while leaving the statewise dynamics, rewards, and optimal action preferences fixed.

Exact-PG updates. Each update samples four tasks with replacement, first choosing a task family uniformly and then an instance uniformly within that family. For each task, dynamic programming evaluates the current policy’s regularized value and advantage, and propagates discounted visitation from $\mu _ { h } .$ . Exact-PG sums the policy gradient over this task’s reachable states and complete action menu. Gradients for the four tasks are evaluated at the same parameters and averaged once per optimizer step. The objective thus weights tasks equally regardless of graph size. A separate soft Bellman solve provides $\Pi _ { \mathcal { T } _ { h } } ^ { * }$ and $F _ { \mathcal { T } _ { h } } ^ { * }$ as evaluation targets. Table 6 gives the optimization settings.

Each training run uses one RTX 4090 GPU.

Geometry and return measurements. At initialization and each saved checkpoint, we read the complete action distribution at every reachable nonterminal state of all calibration and training-probe tasks. For a task $u ,$ let $S _ { h , u }$ denote this fixed state bank. Its geometry error is

$$
\mathrm { M S E } _ { h , u } ( \theta ) = \frac { 1 } { \vert S _ { h , u } \vert } \sum _ { s \in S _ { h , u } } \frac { \left. \tau P _ { \perp } \left[ \log \Pi _ { \theta , \mathcal { T } _ { h } } ( \cdot \vert s ) - \log \Pi _ { \mathcal { T } _ { h } } ^ { * } ( \cdot \vert s ) \right] \right. _ { 2 } ^ { 2 } } { \vert A _ { h , u } ( s ) \vert } .\tag{28}
$$

We average this quantity within task families and then across the three families. The fixed bank makes the trajectory of this error comparable across checkpoints, including decisions that become rare under the learned policy. Alongside it, we compute native-start success probability, $J , C ,$ , and the squared score error weighted by the evaluated policy’s discounted occupancy. Success is the undiscounted probability of exact completion, evaluated by dynamic programming. Mismatch is computed independently as $\tau / ( 1 - \gamma )$ times the occupancy-weighted $\mathrm { K } \bar { \mathrm { L } } ( \Pi _ { \theta , \mathcal { T } _ { h } } | | \bar { \Pi } _ { \mathcal { T } _ { h } } ^ { * } )$ , allowing comparison with $C - \dot { J }$ . We sum over nonterminal states without renormalizing their occupancy mass.

Transfer and uncertainty. For transfer, we evaluate the six trained actors at the common final update 1,024 under all six deployment harnesses on the fixed test set, which gives the $6 \times 6$ matrix in Figure 3. Every cell uses its deployment harness’s state graph, menu, frozen reference, and optimal target. Comparisons within a column therefore hold the deployment control problem fixed.

The learning curves in Figure 3 use all 288 calibration tasks at initialization and every 64 updates through update 1,024. For both success and geometry MSE, we form pointwise 95% percentile intervals from 2,000 bootstrap resamples of tasks within each family. The same task draws are shared across metrics and checkpoints, with equal weight on each family. Changes from initialization use paired differences under these same draws. All measured checkpoints are plotted without smoothing. We make the same measurements on the 72-task training probe. From initialization to update 1,024, the geometry error falls by 69% to 90% across the six harnesses, both on the training probe and on the calibration tasks, which play no role in parameter updates. For each harness, the two reductions differ by less than one percentage point. Post-training thus improves the geometry on held-out and training tasks alike.

## F.2 READOUT STUDY

Tasks and frozen checkpoint. The main-text comparison uses the H0-trained Qwen3.5-2B checkpoint at post-training update 1,024. Its weights remain frozen throughout evaluation. We evaluate it on the fixed test set in Table 5, the same $\bar { 5 } 7 6$ tasks used by the adaptation and prediction studies below. R0 therefore reproduces the source checkpoint’s H0 results in Table 8.

Presentation variants. R0 uses the canonical H0 system message, JSON observation, and actioncode mapping. R1 reverses the order of top-level observation fields while preserving their contents and the nested action dictionary. R2 replaces the action dictionary with a list of records containing the same code and action fields. R3 shifts the action-code assignment cyclically by one position, R4 reverses it, and R5 uses a fixed random permutation for each menu size. These three variants preserve the order and full arguments of the semantic actions and update the displayed code assignment and decoder together. We align every probability table by semantic action before comparison. The system message and all task information are unchanged, and inputs are evaluated without truncation.

Shared reference and exact evaluation. We fix the semantic reference $q _ { 0 } ( a \mid s )$ to the frozen base model’s distribution under R0 and assign this same probability to the corresponding code in each variant. The evaluated actor reads each variant normally. All conditions use H0’s native actions and dynamics, ten model decisions, $\gamma = 0 . 9 5 , \tau = 0 . 1$ , and zero explicit costs. The full legal single-token menu is evaluated at sampling temperature one. Model forward passes use BF16 with state microbatches of eight, followed by FP64 probability normalization and control calculations. We enumerate reachable states, solve the optimum separately in each code ordering, and compute return and success from the measured semantic policy by dynamic programming. Mismatch is computed independently from occupancy-weighted $\mathrm { { \dot { K } L ( \dot { I I } \parallel \dot { I I } ^ { * } ) } }$ . The largest ceiling difference is $2 . 2 \times \dot { 1 } 0 ^ { - 1 6 }$ and the largest optimal-action probability difference is $8 . 3 \times \bar { 1 0 } ^ { - 1 6 }$ across variants.

Measurements and uncertainty. Choice change is the fraction of states where the most probable semantic action differs from R0. Readout RMS is the square root of the mean squared entries of $\eta = \tau P _ { \perp } ( \log \Pi _ { \theta , R _ { v } } - \log \Pi _ { \theta , R _ { 0 } } )$ . We average state-level measurements within each task, then average tasks within each family and give the three families equal weight. All variants are evaluated on the same tasks. We form percentile intervals from 2,000 paired task bootstrap resamples within families. These intervals quantify task variation at the fixed checkpoint. Table 7 reports the readout and performance measurements.

Table 7: H0 readout results, with 95% paired-task bootstrap intervals below each mean. RMS intervals apply the square root to the mean-squared drift interval endpoints. The target drift is numerically zero in every presentation.
<table><tr><td></td><td>Variant Readout RMS</td><td>Mismatch M</td><td>Return J</td><td>Success (%)</td></tr><tr><td>R0</td><td>0.000 [0.000, 0.000] [0.010, 0.021]</td><td>0.015</td><td>0.423 [0.409, 0.436]</td><td>85.0 [84.0, 86.1]</td></tr><tr><td>R1</td><td>0.200</td><td>0.484 [0.198, 0.203] [0.460, 0.507][-0.072, -0.020] [17.0, 21.4]</td><td>-0.046</td><td>19.2</td></tr><tr><td>R2</td><td>0.054</td><td>0.023 [0.053, 0.054] [0.018, 0.028]</td><td>0.415 [0.401,0.428]</td><td>86.0 [85.1, 86.9]</td></tr><tr><td>R3</td><td>0.181</td><td>0.148 [0.179, 0.183] [0.143, 0.155]</td><td>0.289 [0.274, 0.303]</td><td>57.1 [55.9, 58.3]</td></tr><tr><td>R4</td><td>0.181</td><td>0.130 [0.179, 0.182] [0.124, 0.136]</td><td>0.308 [0.292, 0.323]</td><td>68.2 [66.9, 69.5]</td></tr><tr><td>R5</td><td>0.168</td><td>0.110 [0.166, 0.170] [0.104, 0.116]</td><td>0.328 [0.312, 0.342]</td><td>67.6 [66.2, 68.9]</td></tr></table>

## F.3 ADAPTATION AFTER AN INTERFACE REVISION

Revision and source checkpoint. We study H0-to-H1 adaptation using the model, task families, action decoder, and harness definitions of Appendix F.1. H1 retains $\mathrm { H 0 } ^ { \bar { \cdot } } \mathrm { s }$ native actions and adds ordered pairs of clicks executed within one actor decision, stopping at native termination. The source checkpoint $\theta _ { 0 }$ is the H0-trained Qwen3.5-2B model at post-training update 1,024.

Tasks and splits. Training reuses the 1,440 training tasks and the coverage mixture $\mu _ { h }$ in $\operatorname { E q . }$ (27). The evaluation reuses all 576 tasks in Table 5, with 192 instances per family. These tasks are held out from parameter updates and checkpoint selection. We reuse the same task graphs, action menus, and reference probabilities as the post-training study. All deployment metrics start from the native task starts $\rho _ { 0 }$

Training branches and budget. Both branches inherit the same source LoRA weights and reset AdamW’s first and second moments. One continues under H0, giving $\theta _ { s } ,$ and the other adapts under H1, giving $\theta _ { t }$ . Both use paired task draws. Each receives 1,024 additional Exact-PG updates with four tasks per update, for 4,096 task draws. We retain the post-training settings in Table 6, including LoRA rank 16 and scale 32, constant learning rate $1 0 ^ { - 5 }$ , BF16 forward passes, and state microbatches of eight. Each interface’s dynamics, costs, frozen base-model reference, $\gamma = 0 . 9 5$ , and $\tau = 0 . 1$ stay fixed throughout training and evaluation.

Evaluation schedule and measurements. We evaluate at additional updates 0, 32, 64, 128, 256, 512, and 1,024. The source checkpoint is evaluated under both interfaces, and each later checkpoint from both branches is evaluated under both interfaces, giving 26 complete evaluations. The budget and checkpoint schedule were fixed before adaptation. The endpoint comparisons use update 1,024, and the curves include every scheduled checkpoint.

The oracle solves the soft Bellman equations for $C _ { j }$ and $\Pi _ { { \mathcal { T } } _ { j } } ^ { * }$ . Complete action probabilities and the evaluated policy’s forward discounted visitation determine M through Eq. (6). Terminal states contribute zero loss, and nonterminal visitation mass is retained without renormalization. Independently, Bellman policy evaluation computes J, while undiscounted policy evaluation gives task success. The residual in $J = C - M$ checks numerical consistency. These measurements use the compiled task graphs and full action distributions.

Comparisons. For each branch, we compare the target-interface mismatch with the fixed recovery threshold $C _ { 2 } - J ( \theta _ { 0 } , { \mathcal { I } } _ { 1 } )$ . We separately evaluate recovery beyond the source pair, interface ranking for the updated checkpoint, and the return difference between the H1-adapted pair and continued training under H0. Evaluating both branches under H1 also measures the effect of the training interface at a common deployment interface. We report the component gains $A _ { 1 } , A _ { 2 }$ and their difference Ω from Appendix D.6, alongside regularized return and task success.

Uncertainty and computation. Metrics average tasks within each family and weight the families equally. We form pointwise 95% percentile intervals from 2,000 paired bootstrap resamples within families. The same resampling indices are shared across checkpoints, interfaces, metrics, and paired differences. Each branch trains on one RTX 4090 GPU, and a third GPU evaluates checkpoints.

Adaptation results. H1 has a higher ceiling than H0, and both stay fixed across checkpoints. The frozen swap increases mismatch enough to outweigh this gain and lower return. H1 adaptation then reduces mismatch below the recovery threshold, and the final adapted pair exceeds both the source H0 pair and H0 continuation under H0. Table 8 gives the measurements of the source checkpoint and both final checkpoints. Rising return under H1 and falling return under H0 together reverse the ranking along the adaptation branch.

The interface ranking reverses before the adapted pair recovers the source pair’s return. At the first recorded checkpoint with a positive mean recovery difference, its paired interval still includes zero. By the final budget, the paired intervals for recovery and improvement over H0 continuation both lie above zero. Table 9 reports these endpoint comparisons, and Figure 5 shows the recovery and ranking trajectories. In the notation of Appendix $\mathrm { D } . \bar { 6 } , \mathcal { T } _ { 1 }$ is H0 and $\breve { \overline { { \mathcal { I } } } _ { 2 } }$ is H1.

Table 8: Deployment outcomes of the source checkpoint and both final checkpoints. Both training branches receive 1,024 additional updates. J is regularized return, M is occupancy-weighted decision loss, and success is the undiscounted probability of completing the task.
<table><tr><td>Checkpoint</td><td>Deployment</td><td>J</td><td>M</td><td>Success (%)</td></tr><tr><td>Source</td><td>HO</td><td>0.42289</td><td>0.01485</td><td>85.05</td></tr><tr><td>Source</td><td>H1</td><td>0.31285</td><td>0.13737</td><td>72.90</td></tr><tr><td>HO continuation</td><td>HO</td><td>0.42294</td><td>0.01481</td><td>83.37</td></tr><tr><td>HO continuation</td><td>H1</td><td>0.30404</td><td>0.14619</td><td>75.66</td></tr><tr><td>H1 adaptation</td><td>HO</td><td>0.38642</td><td>0.05132</td><td>78.31</td></tr><tr><td>H1 adaptation</td><td>H1</td><td>0.43133</td><td>0.01889</td><td>88.17</td></tr></table>

## F.4 ADAPTATION ACROSS HARNESS SWAPS

Among the 30 ordered swaps between H0–H5, a frozen comparison rejects 22, and 13 of those have $T \leq 0 ,$ , which leaves nine. Each of the nine starts from the checkpoint post-trained under its source harness in Appendix F.1 and adapts under the target harness with the protocol of Appendix F.3. We recorded the predictions for the eight swaps other than H0→H1 before adapting them.

Table 10 lists the nine swaps. For the eight swaps recorded in advance, $T - r _ { 2 }$ orders the gains with Spearman $\rho = 0 . 9 5$ , the ceiling gain alone with $\rho = 0 . 8 6$ , and the frozen return change with $\rho = 0 . 1 7$ . Subtracting the frozen return change and the gain from T gives the target mismatch before and after adaptation. It falls from between 0.015 and 0.146 to at most 0.019. The observed gain exceeds $T - r _ { 2 }$ in every swap, by 0.004 on average. The ranking predicted from the checkpoint post-trained under the target has the correct sign in all eight recorded swaps and orders the observed rankings with Spearman $\rho = 0 . 9 1$ . After adapting from H1 to H0, the checkpoint prefers H0 by 0.041 and still falls short of the source pair, so a ranking reversal alone does not show a net gain. Adapting from H0 to H5 also beats continued training under H0 by 0.085 [0.078, 0.092]. The 13 swaps with $T \leq 0$ include all five swaps from H5, with T between −0.136 and −0.016.

Table 9: Paired comparisons with 95% task bootstrap intervals. Final comparisons use the H1-adapted checkpoint at 1,024 additional updates. Recovery compares it under H1 with the source H0 pair. Interface ranking compares H1 and H0 at the same adapted checkpoint. Gain over H0 continuation compares the two branches at the same update budget, each under its training harness.
<table><tr><td>Quantity</td><td>Mean [95% CI]</td></tr><tr><td>Ceiling gain  $C _ { 2 } - C _ { 1 }$ </td><td>0.01248 [0.00698, 0.01754]</td></tr><tr><td>Frozen mismatch increase</td><td>0.12252 [0.11487, 0.12961]</td></tr><tr><td>Frozen return change</td><td>-0.11004 [-0.11558, -0.10438]</td></tr><tr><td>Recovery threshold T</td><td>0.02733 [0.01984, 0.03520]</td></tr><tr><td>Final recovery difference</td><td>0.00844 [0.00135, 0.01447]</td></tr><tr><td>Final interface ranking</td><td>0.04491 [0.03653, 0.05254]</td></tr><tr><td>Final gain over HO continuation</td><td>0.00840 [0.00254, 0.01411]</td></tr><tr><td> $A _ { 1 } \colon$  change under HO</td><td>-0.03647 [-0.04033, -0.03295]</td></tr><tr><td> $A _ { 2 } { \mathrm { : } }$  change under H1</td><td>0.11848 [0.10964, 0.12658</td></tr><tr><td> $\Omega = A _ { 2 } - A _ { 1 }$ </td><td>0.15495 [0.14430, 0.16493]</td></tr></table>

Table 10: The nine swaps that a frozen comparison rejects and that have $T > 0 ,$ , ordered by the observed gain. The ceiling gain $\Delta C$ , the frozen return change, $T ,$ , and the predicted gain $T - r _ { 2 }$ come from the post-trained checkpoints before adaptation. The gain is the adapted pair’s return minus the source pair’s return after 1,024 updates. Ranking is the adapted checkpoint’s return under the target minus its return under the source. Success is the change in task success from the source pair, in percentage points. The prediction for H0→H1 was computed after its adaptation.
<table><tr><td>Swap</td><td> $\Delta C$ </td><td>Frozen</td><td> $T$ </td><td> $T - r _ { 2 }$ </td><td>Gain</td><td>Ranking</td><td>Success</td></tr><tr><td>H1→H5</td><td>0.068</td><td>-0.054</td><td>0.092</td><td>0.081</td><td>0.085</td><td>0.167</td><td>+9.0</td></tr><tr><td>H0→H5</td><td>0.081</td><td>-0.004</td><td>0.095</td><td>0.085</td><td>0.085</td><td>0.111</td><td>+8.7</td></tr><tr><td>H3→H1</td><td>0.078</td><td>-0.040</td><td>0.099</td><td>0.075</td><td>0.083</td><td>0.169</td><td>+11.7</td></tr><tr><td>H4→H1</td><td>0.075</td><td>-0.040</td><td>0.089</td><td>0.066</td><td>0.071</td><td>0.164</td><td>+12.6</td></tr><tr><td>H2→H5</td><td>0.038</td><td>-0.018</td><td>0.053</td><td>0.042</td><td>0.045</td><td>0.084</td><td>-0.5</td></tr><tr><td>H1→H2</td><td>0.030</td><td>-0.055</td><td>0.054</td><td>0.039</td><td>0.042</td><td>0.082</td><td>+9.8</td></tr><tr><td>H0→H1</td><td>0.012</td><td>-0.110</td><td>0.027</td><td>0.003</td><td>0.008</td><td>0.045</td><td>+3.1</td></tr><tr><td>H4→H3</td><td>-0.003</td><td>-0.004</td><td>0.011</td><td>-0.009</td><td>-0.001</td><td>-0.001</td><td>+0.1</td></tr><tr><td>H1→H0</td><td>-0.012</td><td>-0.060</td><td>0.011</td><td>-0.003</td><td>-0.002</td><td>0.041</td><td>-2.5</td></tr></table>

## F.5 CALIBRATION OF FINITE MISMATCH MEASUREMENTS

Estimator. For each task and deployed pair, draw m independent pilot trajectories. For each trajectory, draw $T$ with $\mathrm { P r } ( T = t ) \stackrel { \cdot } { = } ( 1 - \gamma ) \gamma ^ { t }$ and retain its state $s _ { T }$ . Retaining one state per independent trajectory gives independent occupancy samples. A draw after termination contributes zero to the average over all m draws. At each nonterminal sampled state with K actions, draw N independent actions and form

$$
\widetilde { p } ( a \mid s ) = \frac { n _ { a } + \alpha } { N + K \alpha } .\tag{29}
$$

These smoothed distributions enter the mismatch estimate in Eq. (11). Each nonterminal state draw uses the complete deployed action menu. Pilot and action-sampling queries are both counted.

Candidate budgets and smoothing. The grid uses $m \in \{ 6 4 , 2 5 6 \}$ and $N \in \{ 1 6 , 6 4 , 2 5 6 , 1 0 2 4 \}$ Four smoothing rules use a pseudocount of 0.1 or 0.5 per action, or a total pseudocount $\beta \in \{ 0 . 5 , 1 \}$ divided across the K actions. This gives 32 candidates. Calibration uses 32 independent measurement pools, with shared nested prefixes across candidates. For each candidate, the calibration scorer computes the expected absolute-mismatch bias using exact occupancy and the binomial marginal distribution of each action count. The predictor receives only the sampled observations.

Selection criteria. A candidate must meet three criteria over H0 and H1 at every adaptation node: absolute expected mismatch bias at most 0.005 averaged over families, bias at most 0.01 within every family, and aggregate return-difference RMSE at most 0.01 for recovery and ranking across measurement pools. It must also reach RMSE at most 0.03 on the configuration calibration set. Among passing candidates, we select the lowest mean realized actor-query cost. Costs within 1% tie and are resolved by worst-node difference $\mathsf { R M S E } , m , N .$ , and the fixed smoothing-rule order.

Four candidates pass, all with $m = 2 5 6$ and $N = 1 0 2 4$ . The selected rule uses $\alpha = 0 . 5$ per action and has configuration calibration RMSE 0.00887. Each test prediction uses a single measurement pool. Table 11 compares the initial and selected budgets, and Figure 8 shows the full sweep.

Table 11: Calibration of the measurement budget. Both columns use a pseudocount of 0.5 per action. Bias refers to the magnitude of the expected mismatch estimation error. Query counts include pilot actions and frequency samples at nonterminal states.
<table><tr><td>Calibration quantity</td><td></td><td> $m = 6 4 , N = 1 6 \quad m = 2 5 6 , N = 1 0 2 4$ </td></tr><tr><td>Largest condition-level absolute bias in M</td><td>0.24805</td><td>0.00287</td></tr><tr><td>Largest family-level absolute bias in M</td><td>0.40320</td><td>0.00442</td></tr><tr><td>Worst-node return-difference RMSE</td><td>0.20259</td><td>0.00672</td></tr><tr><td>Mean actor queries per task and condition</td><td>356.9</td><td>42,553.3</td></tr></table>

Why the budget changes the estimate. Eq. (29) mixes the empirical action distribution with a uniform distribution of weight $K \alpha / ( N + K \alpha )$ . For a 30-action menu with $\alpha = 0 . 5$ , this weight is 48.4% at $N = 1 6$ and 1.44% at $\dot { N } = 1 0 2 4$ . Larger action budgets reduce this distortion, while more occupancy samples stabilize the aggregate estimate. On the test configurations of Table 12, increasing the budget reduces prediction RMSE from 0.01132 to 0.00230. The selected budget uses about 119 times as many actor queries on calibration.

![](images/c30877ebbdd642f5bac7ec785edd1b9583b84472f19f9ae6c9b2d9087c3fd5ec.jpg)

![](images/0082e3c4e5a4b17f97f32d48e7408b234b60bd0173d5105dba4ca7ead059935f.jpg)  
Figure 8: Measurement accuracy depends on the query budget and smoothing rule. Each point is a calibration candidate. (a) Largest absolute mismatch bias across deployment conditions. (b) Largest aggregate return-difference RMSE across adaptation nodes. Dashed lines mark calibration thresholds. The cross marks the initial budget and the star marks the selected estimator.

## F.6 INDEPENDENT PREDICTION OF DEPLOYMENT OUTCOMES

Tasks and checkpoints. The prediction study uses the fixed test set in Table 5, with 576 tasks divided equally across the three MiniWoB++ families. Estimator calibration uses a separate set of 72 tasks, 24 per family, generated in the same way as the splits in Table 5 and disjoint from all of them. This set is fixed before estimator selection and plays no role in the reported test results. We use the H0 checkpoint at post-training update 1,024 and the H1-adapted checkpoints from Appendix F.3. All weights are frozen for this study. The control objective uses $\gamma = 0 . 9 5 , \tau = 0 . 1$ , and a horizon of ten model decisions.

Execution and encoding sweep. Both final checkpoints are deployed through $\mathrm { H 0 ^ { \circ } s }$ native action menu. We cross execution settings with three encodings: the canonical presentation R0, reversed top-level observation fields R1, and a cyclic permutation of action codes R3. The encodings preserve the semantic actions and their execution. The reference is the frozen base model’s distribution under R0, aligned to the same semantic actions in all three presentations. Prompts are held fixed acros execution settings and omit numerical reliability and cost values.

The baseline has click reliability $p = 0 . 8 ,$ click cost $c = 0 . 0 1$ , and encoding R0. One sweep varies $p \ \in \ \{ 0 . 5 , 0 . 6 , 0 . 7 , 0 . 7 5 , 0 . 8 , 0 . 8 5 , 0 . 9 , 0 . 9 5 , 1 \}$ at fixed $c = 0 . 0 1$ . A second varies $c \in$ $\{ 0 , 0 . 0 0 2 5 , 0 . 0 0 5 , 0 . 0 0 7 5 , 0 . 0 1 , 0 . 0 1 5 , 0 . 0 2 , 0 . 0 3 , 0 . 0 5 \}$ at fixed $p = 0 . 8 .$ . Failed clicks leave the physical page unchanged and consume the decision and its cost. Submission retains the native task semantics. Calibration uses $p ~ \in ~ \{ 0 . 5 , 0 . 8 , 1 \}$ and $c \in \{ 0 , 0 . 0 1 , 0 . 0 5 \}$ in their respective sweeps. Crossing these settings with the encodings gives 15 calibration and 36 test configurations per checkpoint. The two checkpoints yield 72 test comparisons with their respective baselines.

Prediction and scoring. For each task and configuration, an oracle solves the control problem for C and the queried optimal action distributions. The predictor obtains independent pilot trajectories and finite categorical samples from the deployed policy. Eq. (11) gives $\widehat { \overline { { M } } }$ for the candidate and baseline. Their predicted return difference is

$$
\widehat { \Delta J } = \Delta C - \Delta \widehat { M } .\tag{30}
$$

Appendix F.5 specifies the estimator and calibration. All test predictions are saved before scoring starts. The scoring process computes exact J and occupancy-weighted KL from full policy probabilities. The predictor’s data access is limited to samples, pilot states, the reference, oracle ceilings, and queried optimal targets.

Adaptation comparisons. We evaluate the saved H1-adapted checkpoints at 0, 32, 64, 128, 256, 512, and 1,024 updates under native H0 and H1. Each interface retains its frozen base-model reference. At every node, the predictor estimates the recovery difference $J ( \theta _ { t } , \mathrm { H 1 } ) - J ( \theta _ { 0 } , \mathrm { H 0 } )$ and the ranking difference $J ( \theta _ { t } , \dot { \mathrm { H 1 } } ) - J ( \theta _ { t } , \mathrm { H 0 } )$ ). The source H0 baseline receives an independent measurement whose cost is included.

Baselines and access. Ceiling-only prediction sets $\Delta \widehat { M } = 0 .$ . The frequency rollout baseline estimates regularized returns directly, using repeated queries at each visited state to estimate the selected action’s log probability. Calibration selects 64 such samples per decision with additive smoothing $\alpha \ = \ 0 . 5 .$ This baseline receives the same actor-query allowance as the diagnostic, including its pilot queries and baseline measurement. Reference computation and oracle construction are recorded separately from actor queries.

Table 12: Prediction error on the fixed test tasks. Configuration RMSE averages over 72 checkpoint and configuration comparisons. Recovery and ranking RMSE average over seven saved adaptation checkpoints. No estimator reads exact action probabilities.
<table><tr><td>Estimator</td><td>Configuration RMSE</td><td>Recovery RMSE</td><td>Ranking RMSE</td></tr><tr><td>Ceiling minus measured mismatch</td><td>0.00230</td><td>0.00245</td><td>0.00286</td></tr><tr><td>Ceiling alone</td><td>0.23470</td><td>0.05744</td><td>0.05342</td></tr><tr><td>Frequency rollout, same query budget</td><td>0.00240</td><td>0.02672</td><td>0.02667</td></tr></table>

Prediction results. Figure 9 compares predicted and observed return changes across the configuration sweep. Table 12 reports errors in aggregate return differences, where each comparison first averages returns equally across task families. In the configuration sweep, the mismatch-based prediction identifies the direction of all 70 changes outside a 0.01 tie region and orders all 1,195 configuration pairs outside that region within each checkpoint. Classifying changes as harmful, tied, or beneficial gives 71 correct predictions out of 72.

In the adaptation study, at the final update 1,024, the exact recovery and ranking differences are 0.00844 and 0.04491, with intervals in Table 9. Their predictions are 0.00579 with 95% interval [−0.00116, 0.01228] and 0.04195 with interval [0.03371, 0.05015]. Using the same 0.01 tie region, three-class predictions agree with the exact values at six of seven recovery nodes and all seven ranking nodes. Figure 5 in Section 6.2 shows the predicted and observed trajectories.

![](images/5951828c2289991bbb1168e08380514c69c3a5221f4f31817b0d207e692c89dc.jpg)

![](images/ea472e0ca1df057e4272a9fec4f6cb49368bed3b598e163aa5e74abcf6f7d213.jpg)

![](images/44c3d420d218d9a1c961b0e427062bad9d2c37f95a463aca4d0c58bc61c5e609.jpg)  
Figure 9: Measured mismatch predicts the return realized under a harness revision. Each point compares a candidate with its baseline on the fixed test tasks. The diagonal marks agreement between predicted and observed return changes. Circles and squares denote the H0-trained and H1-adapted checkpoints. (a) Ceiling minus measured mismatch. (b) Direct rollouts with action probabilities estimated from frequencies at the same actor-query budget. (c) Ceiling alone.

Uncertainty and resources. We average within each family and give equal weight to the three families. Intervals use 2,000 paired task bootstrap resamples within families. They describe tasksampling uncertainty for the fixed checkpoints. Finite action samples are simulated from cached distributions of the actual language-model checkpoints. The new probability measurements took 2.19 GPU-hours.

## F.7 CODING-AGENT HARNESS STUDY

Harnesses and benchmark. The study uses the native configurations of DeepAgents, Pi codingagent, DeepSeek Harness, Terminus-2, and mini-SWE-agent. All file and command tools run inside the benchmark task container. The framework owns its loop, tool feedback, delegation where configured, compression, retries, and termination. Terminus-2 retains tmux, completion confirmation, and summarization. Mini-SWE-agent uses its official tool-call profile. Terminal-Bench Lite contains 39 tasks, scored by the official verifier. The agent never sees scoring data or gold patches.

Accuracy. Each harness has three independent runs. Accuracy is the mean of the three single-run success rates over all planned tasks, and the reported spread is the sample standard deviation across runs. Timeouts and failures remain in the denominator as unresolved tasks.

Models and budgets. The target is Qwen3.5-9B with thinking enabled, temperature 0.6, an output limit of 32,768 tokens, and a context window of 65,536 tokens. The teacher is GPT-5.6-luna with the same output and context limits and high reasoning effort. Each trajectory is capped at 50 top-level decisions and uses the task’s own time limit.

Recorded states and sampling. At each model call, we record the request context, tool definitions, controller state, and remaining budgets. Summary, subagent, and retry calls count as decisions. To sample a decision, we resend the recorded request and decode each sample with the harness’s parser, without executing tools. Samples measure the next action distribution at the recorded state.

Probe tasks and trajectories. The probes use three Terminal-Bench Lite tasks: acl-permissions-inheritance, analyze-access-logs, and bank-trans-filter. For each harness, the target and the teacher each contribute three trajectories per task, taken from separate runs of their own. A trajectory enters an estimate only when every recorded state in it has a complete probe. The teacher estimates use 7 of 9 trajectories for Pi coding-agent and all 9 for the other harnesses. The target estimates use 7, 6, and 8 of 9 trajectories for Pi coding-agent, DeepAgents, and DeepSeek Harness, and all 9 for Terminus-2 and mini-SWE-agent.

Common action categories. Each decoded output maps to one of 13 categories: inspect, modify, verify, setup, run, plan, delegate, wait, stop, context update, return, opaque, and invalid. File reads, listings, and searches map to inspect, and writes, edits, and deletions to modify. Todo and planning operations map to plan, and subagent calls to delegate. Shell commands, including Terminus-2 keystrokes at a shell prompt, are parsed, and a command maps to its dominant operation. Commands whose program the parser cannot resolve map to opaque, and responses without a countable native action map to invalid. The categories merge actions with different arguments and effects, so the estimates compare decisions at the level of these categories.

![](images/632674edd4065539aad7f7c7a18040cd651efccba397cbf05e75fad93ed8c942.jpg)  
Figure 10: Ceiling and mismatch of five coding-agent harnesses for a frozen Qwen3.5-9B. Estimates of Table 2 on three Terminal-Bench Lite probe tasks with $\tau = 0 . 1$ . Each harness sits at its accuracy over all 39 tasks. The open circle marks ${ \dot { \tilde { C } } } ,$ , the filled circle marks $\widetilde { C } - \widetilde { M }$ , and the bar between them has length $\widetilde { M }$

Estimators. At a recorded state s under harness $h ,$ , let $p _ { h } ( \cdot \mid s )$ and $q _ { h } ( \cdot \mid s )$ be the target’s and the teacher’s category distributions, estimated from 128 and 64 samples with a pseudocount of 0.5 per category. The objective uses the reference $p _ { h }$ , a binary terminal reward $Y$ , and no discount. The ceiling estimate averages the teacher’s regularized return over its complete trajectories $s _ { 0 } , \ldots , s _ { L - 1 } \colon$

$$
\widetilde { C } _ { h } = \mathbb { E } _ { q _ { h } } \left[ Y - \tau \sum _ { t = 0 } ^ { L - 1 } \mathrm { K L } \big ( q _ { h } ( \cdot \vert s _ { t } ) \big \vert \big \vert p _ { h } ( \cdot \vert s _ { t } ) \big ) \right] .\tag{31}
$$

The mismatch estimate applies Proposition 6.1 with $q _ { h }$ in place of the optimum. Without discounting, the factor $1 / ( 1 - \gamma )$ becomes a sum over the decisions of each complete target trajectory:

$$
\widetilde { M } _ { h } = \frac { 1 } { 4 \tau } \mathbb { E } _ { p _ { h } } \Big [ \sum _ { t = 0 } ^ { L - 1 } \Big \lVert \tau P _ { \perp } \log \frac { p _ { h } ( \cdot \mid s _ { t } ) } { q _ { h } ( \cdot \mid s _ { t } ) } \Big \rVert _ { 2 } ^ { 2 } \Big ] .\tag{32}
$$

We use $\tau = 0 . 1$ . With $p _ { h }$ as the reference, the target’s regularized return equals its success probability, so the exact ceiling minus mismatch also equals it. The estimates put a teacher that need not be optimal in place of the optimum, so $\widetilde { C } _ { h }$ estimates a lower bound on the ceiling and $\widetilde { M _ { h } }$ measures distance to the teacher. We therefore compare the ordering of $\widetilde { C } _ { h } - \widetilde { M } _ { h }$ with the ordering of accuracy. The mismatch per decision in Figure 6 is $\widetilde { M _ { h } }$ divided by the mean number of decisions per complete target trajectory. Table 13 lists the three estimates and the decisions per trajectory, and Figure 10 plots the estimates against accuracy.

Action shares. Table 14 gives the mean share of each category at the recorded states of complete target trajectories, for the target and for the teacher sampled at the same states. The remaining categories each hold at most 7.2% for either model. Across all five harnesses, the target inspects more often than the teacher. It also marks the task complete less often, except under Terminus-2, where the two do so at similar rates.

Table 13: Terminal-Bench Lite probe results at $\tau = 0 . 1$ . Decisions are the mean number of model calls per complete target trajectory.
<table><tr><td>Harness</td><td>Decisions</td><td> $\widetilde { C } _ { h }$ </td><td> $\widetilde { M } _ { h }$ </td><td> $\widetilde { C } _ { h } - \widetilde { M } _ { h }$ </td></tr><tr><td>Terminus-2</td><td>4.9</td><td>0.431</td><td>0.134</td><td>0.297</td></tr><tr><td>mini-SWE-agent</td><td>8.6</td><td>0.631</td><td>0.335</td><td>0.296</td></tr><tr><td>DeepSeek Harness</td><td>8.6</td><td>0.590</td><td>0.382</td><td>0.208</td></tr><tr><td>Pi coding-agent</td><td>6.6</td><td>0.383</td><td>0.206</td><td>0.177</td></tr><tr><td>DeepAgents</td><td>14.2</td><td>0.261</td><td>0.429</td><td>-0.168</td></tr></table>

Table 14: Action category shares (%) at the states of complete target trajectories on the Terminal-Bench Lite probe tasks, target / teacher.
<table><tr><td>Harness</td><td>States</td><td>Inspect</td><td>Modify</td><td>Plan</td><td>Stop</td><td>Opaque</td><td>Invalid</td></tr><tr><td>Terminus-2</td><td>44</td><td>37.7 / 30.3</td><td>15.8 / 20.8</td><td>0.0 / 0.0</td><td>38.3 / 35.1</td><td>8.2 / 6.4</td><td>0.0 / 0.1</td></tr><tr><td>mini-SWE-agent</td><td>77</td><td>59.7 / 43.5</td><td>14.1 / 11.3</td><td>0.0 / 0.0</td><td>11.7 / 22.4</td><td>14.1 / 18.9</td><td>0.1 / 0.0</td></tr><tr><td>DeepSeek Harness</td><td>69</td><td>33.3 / 20.0</td><td>12.1 / 15.0</td><td>15.1 / 26.5</td><td>11.2 / 14.8</td><td>27.7 / 18.6</td><td>0.1 / 2.7</td></tr><tr><td>Pi coding-agent</td><td>46</td><td>48.3 / 37.0</td><td>20.4 / 22.4</td><td>0.0 / 0.0</td><td>12.9 / 21.0</td><td>18.0 / 16.6</td><td>0.0 / 0.4</td></tr><tr><td>DeepAgents</td><td>85</td><td>27.9 / 24.8</td><td>11.8 / 16.8</td><td>11.1 / 11.4</td><td>3.9 / 8.4</td><td>38.2 / 32.4</td><td>3.5 / 0.0</td></tr></table>

## F.8 EXACT HARNESS SCORES AND TASK SUCCESS

Tasks and harnesses. We evaluate H0–H5 from Table 4 on the fixed test set of 576 tasks, with 192 tasks from each of the three MiniWoB++ families in Appendix F.1. We reuse their compiled execution graphs, whose transitions and successful task completion were checked against the native browser. The horizon is ten actor decisions, with remaining budget included in the state. H4 retains the archived decisions of its separate frozen Qwen3.5-2B reviewer.

Frozen model and semantic policy. The actor is off-the-shelf Qwen3.5-9B, with fixed weights and no project adapter. Each prompt exposes the harness’s complete legal action menu, with a single-token code for every parameterized semantic action. We obtain the action distribution by normalizing the corresponding token logits at temperature one, using BF16 inference, FP64 normalization, and a state microbatch of eight. Thinking is disabled. The objective instruction asks the model to maximize exact task completion within its decision budget. Actor probabilities are computed anew for this model under every harness. We set each harness’s reference to this same frozen semantic distribution and hold it fixed when solving for the optimal policy.

Objective and exact computation. We use the finite-horizon counterpart of Eq. (2), with $\gamma = 1$ $\tau = 0 . 1$ , binary terminal success reward, and zero explicit execution costs. This objective and reference define a new control problem for each harness, so the ceilings differ from those of the posttraining study in Section 3.2. Backward induction on the acyclic graph gives $\Pi _ { \boldsymbol { \mathcal { I } } } ^ { * }$ and the initial-state optimal value C. Let $\begin{array} { r } { \nu _ { \mathbb { Z } } ^ { \Pi } ( s ) = \sum _ { t = 0 } ^ { 9 } \operatorname* { P r } _ { \mathbb { Z } } ^ { \Pi } ( s _ { t } = s ) } \end{array}$ be the expected number of visits to a nonterminal state. Forward propagation under the evaluated policy supplies these visitation weights, giving

$$
M ( \theta , \mathcal { T } ) = \tau \sum _ { s } \nu _ { \mathcal { Z } } ^ { \Pi _ { \theta , \mathcal { Z } } } ( s ) \mathrm { K L } ( \Pi _ { \theta , \mathcal { Z } } ( \cdot  { | } s )  { | | \Pi _ { \mathcal { Z } } ^ { * } ( \cdot  { | } s ) ) } .\tag{33}
$$

The Bellman residual telescopes over the finite horizon to give $C - J = M$ . Since the evaluated policy equals its fixed reference, its regularization penalty is zero, and J equals task success probability. We compute J by forward reward accumulation and success by a separate backward recursion. These independent computations agree with $C - M$ to numerical precision. The largest observed identity residual is $7 . 2 2 \times \dot { 1 } 0 ^ { - 1 6 }$

Rollout evaluation and aggregation. For each task and harness, we draw 256 independent trajectories from the measured actor distribution on the compiled graph. The fraction that reaches native task success supplies the vertical coordinate in Figure 11.

Scores and success are averaged within each family and then equally across families. Each point therefore compares the same task distribution. The three selection rules maximize $C , - M$ , and

![](images/f62c46b00359483f4b545823b8da73526349484ffed045ab4c54f1dbe1ab15db.jpg)  
Figure 11: Harness scores compared with task success. Each point is one harness evaluated with the same frozen Qwen3.5-9B on a shared task set. The panels compare $C , - M$ , and $C - M$ with rollout success. Dashed lines show linear fits.

$C - M ,$ respectively. The dashed lines are unweighted linear fits through the six displayed harness means. All actor evaluations run on four RTX 4090 GPUs.

For these deterministic task graphs with binary terminal reward, an individual task’s reference success probability S gives $C = \tau \log ( 1 + S ( e ^ { 1 / \tau } - 1 ) )$ . This is increasing in S, but averaging this nonlinear quantity across tasks can order harnesses differently from averaging success. In the aggregate comparison, ceiling selects H0, while both negative mismatch and $\bar { C } - M$ select H3. Negative mismatch ranks H1 above H0, whereas exact and sampled success rank H0 above H1. Combining ceiling and mismatch gives the complete ordering of the exact mean success probabilities.