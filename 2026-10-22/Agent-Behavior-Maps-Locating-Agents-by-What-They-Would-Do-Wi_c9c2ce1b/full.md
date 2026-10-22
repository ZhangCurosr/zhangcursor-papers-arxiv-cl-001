# Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them

Shota Fujikawa<sup>∗†</sup> The University of Tokyo fjkwsht@is.s.u-tokyo.ac.jp

Kosuke Akimoto NEC Corporation kosuke\_a@nec.com

Daichi Haraguchi NEC Corporation daichi-haraguchi@nec.com

Takuya Tamura<sup>∗</sup> NEC Corporation tamura-takuya@nec.com

Masafumi Oyamada NEC Corporation oyamada@nec.com

## Abstract

LLM agents with similar success rates can follow diferent action sequences. An Agent Behavior Map represents LLM–harness combinations using centered action log-likelihoods on shared reference trajectories, without requiring new environment rollouts from the target agent. We derive error bounds relating squared coordinate distances to trajectory KL under explicit assumptions, and across eight cross-scored models the distances track the directly measured trajectory KL in rank order. Across 100 LLMs and four environments, ALFWorld-based coordinates support performance prediction and identify behavioral similarities. In ALFWorld, these similarities persist after controlling for success rates and model size, including across model families. As an extension, harness-component displacements approximately compose, with accuracy varying across components and environments.

## 1 Introduction

Comparing LLM agents is a matter not only of how often they succeed but also of how they act. Agents with similar success rates can difer in the tools they choose, the order of their actions, and how often they repeat an operation. Such behavior depends on the combination of the underlying LLM and the harness that manages history, planning, and reasoning (Zhang et al., 2026; Yao et al., 2026; Vats and Golev, 2026; Fan et al., 2026). Running every combination in the environment reveals both performance and behavior, but each new agent to compare requires collecting new trajectories. This paper asks whether a representation that captures similarity of run-time behavior can be built without new environment runs of the target agent.

When agents are compared by their loglikelihoods on shared reference trajectories, the difficulty is that the distribution of the histories on which the likelihoods are conditioned itself depends on the agent’s policy. The histories contained in the reference trajectories need not coincide with those that an agent reaches through its own actions. Whether similarity of action probabilities measured on shared histories corresponds to similarity of trajectory distributions and behavior at run time is therefore not obvious. How much of these on-policy properties can be captured by of-policy observations on fixed reference trajectories is the central question of this work.

Language model maps, which vectorize loglikelihoods on shared texts, represent the relations between models in a common coordinate space (Oyama et al., 2025). Their extension to prompt–response distributions has examined performance prediction from conditional log-likelihoods and the approximate additivity of the coordinate displacements caused by prompt changes (Oyama et al., 2026). This paper applies the framework to agents that interact sequentially with an environment and answers the question above with theory and experiments.

An Agent Behavior Map treats each agent as a pair of an LLM and a harness. Each agent rebuilds its input with its own harness from the shared external history in the reference trajectories, namely the task, the observations, and the past actions. The recorded actions are teacher-forced under this input, and the resulting log-likelihood vector is centered to give the agent’s coordinates. Because the reference trajectories are reused, the coordinates of a new agent can be computed without collecting any additional environment trajectories from it. Figure 1 shows an example in which agents that are close in these coordinates take similar action sequences when run. We validate the representation from two angles.

• Theoretical and empirical relation to the trajectory distribution (§2, §3.1). Under assumptions on the locality of the two policies and on the mismatch with the reference distribution, we derive error bounds that relate squared coordinate distances to the KL divergence between trajectory distributions. The error decomposes into local approximation, reference mismatch, centering, and finite samples, and the decomposition shows how the step-level and trajectory-level representations difer in their dependence on trajectory length. In a cross-scoring experiment with eight models, we compare the coordinate distances with the directly measured trajectory KL, the target itself, and find a high rank correlation.

![](images/6f8f4997433e76f2c81f1e621dae3d81c4f7e8dbb48a0a34ada3bbc6f8473f08.jpg)

![](images/6bd75418dbe06321b2701f33a3d3345fca4d913b3c7383400d17c922f4fc62a7.jpg)

![](images/396129df60c67ffaff311d11e7669d8924888fc1f707aa7d0c8db68f17ad91d3.jpg)

![](images/f8716fd10bcda36854c8e4eaadd5cb381a1d9947d5222951d56ca9d5c84484ff.jpg)  
Figure 1: Agents close on the map act alike when run. (a) 100 LLMs with the BASE harness, scored on 500 ALFWorld reference trajectories (t-SNE layout; neighbors computed in the original space). ⋆: gemma4-31b; green: its five nearest models on the map; orange rings: the five closest in parameter count. (b) Action sequences of these 11 models on one ALFWorld task. All succeed, but four of the five map neighbors share the anchor’s short route, whereas three of the five size-matched models take longer ones. (c) Efect size of the behavioral distance to the k-th nearest model on the map (solid) or in parameter count (dashed), averaged over four aspects and 100 models (§3.3); positive means more similar than the average pair.

• Validation on performance and behavior (§3.2, §3.3). We apply a common harness to 100 LLMs and evaluate coordinates built from ALF-World reference trajectories alone on ALFWorld, TextWorld, BFCL, and MiniWoB++. Besides predicting the success rates of unevaluated models from the evaluation results of the others, we ask whether neighbors on the map generate similar action sequences. In ALFWorld in particular, the similarity of action sequences remains when compared with neighbors chosen by success rate or by size, and when the candidates are restricted to models of other families. These results support the view that the coordinates carry behavioral information that neither performance level nor model family explains. The accuracy of performance prediction and the strength of behavioral similarity among neighbors, however, depend on the environment and the measure.

Finally, we represent the addition of a harness component as a displacement on the map and ask whether such displacements add up within an LLM (§4). Comparison with measured coordinates shows approximate additivity, with an accuracy that depends on the component and the environment.

## 2 Agent Behavior Maps

## 2.1 Problem: comparing agents by what they do when run

On task $x ,$ an agent interacts with the environment and produces a trajectory $\begin{array} { r l } { \tau } & { { } = } \end{array}$ $( x , o _ { 0 } , a _ { 0 } , o _ { 1 } , \dots , a _ { T - 1 } , o _ { T } )$ of observations $o _ { t }$ and actions $a _ { t }$ . Its length $\begin{array} { r } {  { T } \ = \ T ( \tau ) } \end{array}$ varies and is capped by a known limit $T _ { \mathrm { m a x } }$ The external history before the action at time t is $h _ { t } \ =$ $( x , o _ { 0 } , a _ { 0 } , \ldots , o _ { t } )$ ; it excludes the generator’s internal state, such as its system prompt, reasoning, plans, and summaries. An agent $\begin{array} { r l } { i } & { { } = } \end{array}$ (LLM, harness) rebuilds its input from $h _ { t }$ by its own rule $C _ { i }$ (history window, retrieval, summarization) and generates text under the action grammar; $\pi _ { i } ( a \mid h )$ denotes the resulting probability of action a (Appendix A.1 covers the grammar and tokenization). Given a task distribution w, an initial observation $p _ { 0 } .$ and environment dynamics $E _ { t } = E ( o _ { t + 1 } \mid h _ { t } , a _ { t } )$ shared by all agents, agent i induces the trajectory distribution

$$
P _ { i } ( \tau ) = w ( x ) p _ { 0 } ( o _ { 0 } \mid x ) \prod _ { t < T } \pi _ { i } ( a _ { t } \mid h _ { t } ) E _ { t } .\tag{1}
$$

The environment terms cancel in the log-likelihood ratio of two agents. We compare agents by the trajectory KL $K _ { i j } : = \mathrm { K L } ( P _ { i } \| P _ { j } )$ and its symmetrization $K _ { \mathrm { s y m } } : = ( K _ { i j } + K _ { j i } ) / 2 . \ P _ { i }$ is the distribution of trajectories when agent i is actually run, so the properties we care about, such as success rates and action frequencies, are functionals of $P _ { i } { \mathrm { : } }$ : on-policy quantities.

Writing $\mathbb { E } _ { i } = \mathbb { E } _ { P _ { i } }$ , the chain rule of KL (Cover and Thomas, 2006; Ke et al., 2020) gives

$$
\begin{array} { r } { K _ { i j } = \mathbb { E } _ { i } \sum _ { t < T } \mathrm { K L } \big ( \pi _ { i } ( \cdot \mid h _ { t } ) \| \pi _ { j } ( \cdot \mid h _ { t } ) \big ) , } \end{array}\tag{2}
$$

an expectation over the histories that agent i visits when it runs, so measuring it directly takes n rollouts and $O ( n ^ { 2 } )$ cross-scorings for n agents. We ask how much of this quantity can be read, without running the agents, from distances between coordinates obtained by forward passes alone.

## 2.2 Scoring agents on shared reference trajectories

We prepare N reference trajectories $\tau _ { 1 } , \ldots , \tau _ { N }$ once, from an unknown reference distribution $Q .$ The distances use neither $Q ( \tau )$ nor the generators or their mixing weights, so the trajectories can come from anywhere. We use four sources: logs of the agents under study (ALFWorld, Appendix C), traces published by others (τ-bench (Yao et al., 2025) and SWE-Bench (Jimenez et al., 2024) from HAL (Kapoor et al., 2026)), rollouts of two models (the 100-model experiment, §3), and an equalweight pool of eight scored models (§3.1). Let $T _ { r } = T ( \tau _ { r } )$ be the lengths and $M = \textstyle \sum _ { r } T _ { r }$ the total number of steps. For each agent i, teacher forcing (Williams and Zipser, 1989) reads $\ell _ { i , r t } : =$ log $\pi _ { i } ( a _ { r t } \mid h _ { r t } )$ and $\begin{array} { r } { A _ { i , r } : = \sum _ { t < T _ { r } } \ell _ { i , r t } } \end{array}$ without running the environment. Sharing the trajectories has two costs, the mismatch between Q and each agent’s on-policy distribution and the finite number of trajectories, which §2.3 and §2.4 address.

## 2.3 Step and trajectory embeddings

Let $\bar { \ell } _ { i }$ and $\bar { A _ { i } }$ be the means of $\ell _ { i , r t }$ over all M steps and of $A _ { i , r }$ over all N trajectories. Subtracting each agent’s own mean, we define two centered embeddings

$$
z _ { i } ^ { S } : = { \cal M } ^ { - \frac { 1 } { 2 } } ( \ell _ { i , r t } - \bar { \ell } _ { i } ) _ { r , t } , z _ { i } ^ { \tau } : = { \cal N } ^ { - \frac { 1 } { 2 } } ( A _ { i , r } - \bar { A } _ { i } ) _ { r } .
$$

A step coordinate asks whether an action is natural in its situation; a trajectory coordinate asks whether a trajectory is natural from its initial state. Only existing steps $t < T _ { r }$ are included. The distances are $\hat { D _ { S } } = \hat { \| } z _ { i } ^ { S } - z _ { j } ^ { S } \|$ and $\hat { D } _ { \tau } = \Vert z _ { i } ^ { \tau } - z _ { j } ^ { \tau } \Vert$

Table 1: Error bounds by source, with constants: approximation error (Theorem 1, top three rows) and sampling error (Corollary 1, bottom row, which defines $u _ { N } )$ . The trajectory bound is about $T$ times the step bound for every source.
<table><tr><td>Source</td><td>Step</td><td>Trajectory</td><td>Ratio</td></tr><tr><td>Local expansion Ref. mismatch</td><td> $\begin{array} { r } { { \frac { 1 } { 6 } } \varepsilon ^ { 3 } \mathbb { E } _ { i } [ T ] } \end{array}$   $\scriptstyle { \frac { 1 } { 2 } } \lambda T _ { \mathrm { m a x } } \varepsilon ^ { 2 }$ </td><td> $\begin{array} { r } { \frac { 1 } { 4 } \varepsilon ^ { 3 } \mathbb { E } _ { i } [ T ^ { 2 } ] } \end{array}$   $\scriptstyle \frac { 1 } { 2 } \lambda T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 }$ </td><td> $\frac { 3 \mathbb { E } _ { i } [ T ^ { 2 } ] } { 2 \mathbb { E } _ { i } [ T ] }$   $T _ { \mathrm { m a x } }$ </td></tr><tr><td>Centering</td><td> $\bar { \mu } ^ { 2 } / ( 2 \bar { T } )$ </td><td> $\bar { \mu } ^ { 2 } / 2$ </td><td>T</td></tr><tr><td>Sampling</td><td> $\frac { 6 T _ { \mathrm { m a x } } - 1 } { 2 } \varepsilon ^ { 2 } u _ { N }$ </td><td> $\frac { 5 } { 2 } T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 } u _ { N }$ </td><td>≈  $\frac { 5 } { 6 } T _ { \mathrm { m a x } }$ </td></tr></table>

For a pair $( i , j )$ , let $\begin{array} { r } { d _ { t } = \log \frac { \pi _ { i } ( a _ { t } | h _ { t } ) } { \pi _ { j } ( a _ { t } | h _ { t } ) } } \end{array}$ be the step log-ratio, $\begin{array} { r } { X \ = \ \sum _ { t < T } d _ { t } \ = \ \log { \frac { P _ { i } ( \tau ) } { P _ { j } ( \tau ) } } } \end{array}$ its sum, $\mu ~ = ~ \mathbb { E } _ { Q } X$ its mean under the reference, and $\bar { T } = \mathbb { E } _ { Q } [ T ]$ the mean length. The population distances are then $\begin{array} { r } { D _ { S } ^ { 2 } = \frac { 1 } { \bar { T } } \mathbb { E } _ { Q } \sum _ { t < T } ( \bar { d } _ { t } - \mu / \bar { T } ) ^ { 2 } } \end{array}$ and $D _ { \tau } ^ { 2 } = \operatorname { V a r } _ { Q } ( X )$ . Centering removes each agent’s overall confidence, its mean log-likelihood, from the coordinates (Appendices $\mathrm { A . 5 }$ and B.2 treat the uncentered variants).

Distances estimate the trajectory KL. Fix a pair $( i , j )$ . Assumption 1 (locality): for every history h that occurs under $P _ { i } , P _ { j }$ , or $Q , \pi _ { i } ( \cdot \ | \ h )$ and $\pi _ { j } ( \cdot ~ | ~ h )$ have the same support and |log $\pi _ { i } ( a \mid$ $h ) / \pi _ { j } ( a \mid h ) | \leq \varepsilon$ on it. Assumption $2 ~ ( r e f e$ rence mismatch): $\mathrm { T V } ( P _ { i } , Q ) \leq \lambda$ and $\mathrm { T V } ( P _ { j } , Q ) \leq \lambda$ Lengths satisfy $1 \leq T \leq T _ { \mathrm { m a x } } ;$ we assume neither independence across steps nor independence between $T$ and the actions.

Theorem 1 (Approximation error). Under Assumptions 1 and $\begin{array} { r } { 2 , \ | K _ { i j } - \frac { \bar { T } } { 2 } D _ { S } ^ { 2 } | \ \le \ R _ { S } } \end{array}$ and $| K _ { i j } -$ $\begin{array} { r } { \frac { 1 } { 2 } D _ { \tau } ^ { 2 } | \leq R _ { \tau } , } \end{array}$ where, up to absolute constants,

$$
R _ { S } = O \big ( \varepsilon ^ { 3 } \mathbb { E } _ { i } [ T ] + \lambda T _ { \operatorname* { m a x } } \varepsilon ^ { 2 } + \mu ^ { 2 } / \bar { T } \big ) ,\tag{3}
$$

$$
R _ { \tau } = O \big ( \varepsilon ^ { 3 } \mathbb { E } _ { i } [ T ^ { 2 } ] + \lambda T _ { \operatorname* { m a x } } ^ { 2 } \varepsilon ^ { 2 } + \mu ^ { 2 } \big ) .\tag{4}
$$

Analogous bounds holdfor $K _ { \mathrm { s y m } }$ (Appendix A.2).

The factor $T _ { \mathrm { m a x } }$ on the λ term, i.e., statedistribution mismatch accumulating over steps, is an error source that LM maps do not have (constants in Table 1, proofs in Appendix A.2).

## 2.4 Total error: approximation plus sampling

Theorem 1 concerns the population distances. In practice, $\hat { D } _ { S }$ and $\hat { D } _ { \tau }$ are computed from N reference trajectories, so the total error adds sampling error to the approximation error. Bounding $\mathbb { E } _ { i } [ T ] \leq T _ { \operatorname* { m a x } }$ and $\mathbb { E } _ { i } [ T ^ { 2 } ] \leq T _ { \operatorname* { m a x } } ^ { 2 }$ in Table 1 gives the following.

Corollary 1 (Total error). Let $\tau _ { 1 } , \dots , \tau _ { N } \ \sim \ Q$ be i.i.d., and let $\hat { \bar { T } } ~ : = ~ M / N , ~ \hat { K } _ { S } ~ : = ~ \textstyle { \frac { \hat { \bar { T } } } { 2 } } \hat { D } _ { S } ^ { 2 }$ $\begin{array} { r } { \hat { K } _ { \tau } : = \frac { 1 } { 2 } \hat { D } _ { \tau } ^ { 2 } } \end{array}$ , and $u _ { N } : = \sqrt { \log ( 8 / \delta ) / ( 2 N ) }$ . Under Assumptions 1 and 2, with probability at least $1 - \delta ,$ simultaneously

$$
\begin{array} { r l } & { | K _ { i j } - \hat { K } _ { S } | \le T _ { \operatorname* { m a x } } \varepsilon ^ { 2 } \bigl ( \frac \varepsilon 4 + \frac \lambda 2 + 3 u _ { N } \bigr ) + \frac { \mu ^ { 2 } } { 2 \bar { T } } , } \\ & { | K _ { i j } - \hat { K } _ { \tau } | \le T _ { \operatorname* { m a x } } ^ { 2 } \varepsilon ^ { 2 } \bigl ( \frac \varepsilon 4 + \frac \lambda 2 + 3 u _ { N } \bigr ) + \frac { \mu ^ { 2 } } { 2 } . } \end{array}
$$

The trajectory bound carries an extra factor of $T$ on every term. Since $K _ { i j } \ \leq \ \varepsilon ^ { 2 } \mathbb { E } _ { i } [ T ] / 2$ by Lemma 1(iii) and the chain rule, the step bound grows with T like the KL it estimates, whereas the trajectory bound grows as $T ^ { 2 }$ : summing log-ratios before squaring picks up cross terms between steps (Appendix A.4). Sampling error shares the parentheses, so it keeps this factor; it decays as $N ^ { - 1 / 2 }$ ， while the other terms do not. The proof applies Hoefding’s inequality (Hoefding, 1963) to four sample means (Appendix A.3).

Scope of the guarantees. The guarantees cover only close pairs, with small ε and λ. We therefore test empirically whether distances preserve the order of the KL outside this regime (§3.1) and whether close pairs behave alike when run (§3.3).

## 3 What does the map capture about an agent?

This section fixes the harness to a standard configuration (BASE: a history window of the last four steps, no planning, direct action; §4.1) and places each LLM as one point. The question is what the coordinates capture about an agent. LM maps showed a correlation between coordinates and downstream performance (Oyama et al., 2025). We answer with three quantities: the target itself, the KL between the trajectory distributions of the agents when run (§3.1); one projection of $\mathbf { i t } ,$ the success rate (§3.2); and the behavior itself when run (§3.3).

## 3.1 Does distance track the on-policy divergence?

Theorem 1 applies only in the local regime of small ε, that is, to nearly identical policies, whereas actual pairs of models may be far apart. We therefore check empirically whether the distances track the trajectory KL outside that regime, and which version, step or trajectory, tracks it better. In the bounds, the step version is favored by a factor of order T (Table 1); we ask whether this advantage shows in the data. Measuring the trajectory KL directly requires rollouts of every model and $O ( n ^ { 2 } )$ cross-scoring, so we use eight models: each solves a common set of ALFWorld (Shridhar et al., 2021) train tasks, every model scores every trajectory, and the cross-scoring gives $K _ { \mathrm { s y m } }$ for the ${ \binom { 8 } { 2 } } = 2 8 \mathrm { { p a i r s } } .$ The distances use the pooled trajectories of the eight models as the common reference (Appendix B.1).

![](images/a981af48f8f24bad92305b074c644ec5a63c019fda844cd75943fe668bc3d69c.jpg)  
Figure 2: Map estimates, step version $\textstyle { \frac { \bar { T } } { 2 } } { \hat { D } } _ { S } ^ { 2 }$ and trajectory version ${ \scriptstyle \frac { 1 } { 2 } } \hat { D } _ { \tau } ^ { 2 }$ , against the directly measured symmetric trajectory KL $K _ { \mathrm { s y m } } \mathrm { . }$ 8 models, 28 pairs, centered coordinates, log–log axes; error bars are standard errors over the 988 shared ALFWorld train tasks. Both track $K _ { \mathrm { s y m } }$ , the step version more closely.

Both correlate with the KL, the step version more strongly. Figure 2 shows the result (numbers in Table 6, Appendix B.2). The centered step estimate $\textstyle { \frac { \bar { T } } { 2 } } \hat { D } _ { S } ^ { 2 }$ correlates with $K _ { \mathrm { s y m } }$ at Pearson $r = 0 . 9 0$ and Spearman $\rho = 0 . 9 5$ , and the trajectory estimate $\scriptstyle { \frac { 1 } { 2 } } { \hat { D } } _ { \tau } ^ { 2 }$ at 0.86 / 0.94: distances measured without running the agents preserve the order of the trajectory KL measured by running them. Two trends are consistent with the bounds (Appendix B.2): the RMSE of the trajectory version is an order of magnitude larger than that of the step version, as the orders $T$ versus $T ^ { 2 }$ predict, and adding reference trajectories shrinks only the sampling error while the approximation error remains. Centering, although it adds a term to the bound, raises the correlation and lowers the RMSE for both versions. Reading the distances between the eight models from the 100-model map of §3.2 instead, with diferent reference trajectories and no grammar constraint, gives $r = 0 . 8 6$ and $\rho = 0 . 8 8$ (Appendix B.2). Both versions overestimate the measured KL in level, by a median factor of 7.5 for the step version and 74 for the trajectory version (Appendix B.2), so in what follows we use the distances only as a proxy for the order of the KL, and

we use the step version.

## 3.2 Performance: predicting without running

Setup. From here on we use 100 open-weight LLMs. The reference trajectories are 500 trajectories with 10,388 steps, generated by two models (Qwen3.8-27B (Qwen Team, 2026b) and Gemma4- E4B (Gemma Team, 2026)) on 250 ALFWorld train tasks; every model scores them, and its centered step coordinates are computed from these scores. Only the two generators ran the reference trajectories; the other models only score them. Success rates and action sequences are obtained by running each model itself on diferent tasks, so the material for the coordinates and the material for the evaluation do not overlap. Scoring uses raw loglikelihoods without the grammar constraint; rescoring with the constraint barely moves the coordinates (Appendix B.1).

Question. Can the downstream success rate of a model that has only computed likelihoods on the reference trajectories, and has never been run in the environment, be predicted from its coordinates? We hold out one model at a time and predict it from the remaining 99. The downstream tasks are 50 ALFWorld valid\_unseen tasks, the environment of the coordinates, and 50 tasks each from diferent environments, TextWorld (Côté et al., 2018), BFCL (Patil et al., 2025), and MiniWoB++ (Shi et al., 2017; Liu et al., 2018), plus the averages over the four benchmarks and over the three other environments (Appendix B.3). The baselines use only information available without running: a regression on the log parameter count (size), the mean success rate of the same family (family), and a regression on the mean log-likelihood $\bar { \ell }$ (the component removed by centering). On the coordinate side we use kNN, Ridge (Hoerl and Kennard, 1970), and Ridge with size as a covariate, with hyperparameters chosen by inner leave-one-out (Appendix B.3).

Results. Table 2 reports Pearson r. In ALFWorld, the environment of the coordinates, kNN reaches r = 0.88 against 0.63 for size (the bootstrap interval of the diference excludes zero): the success rate of a model that has only run forward passes on the 500 reference trajectories is predicted from the results of the other 99 models with an RMSE of 9 points on the 50 valid\_unseen tasks. In the other environments, all three coordinate methods beat size on BFCL (0.65–0.69 versus 0.48) and TextWorld (0.76–0.78 versus 0.71); on MiniWoB++ size is stronger (0.64 versus 0.56 for Ridge), although a quantity built from likelihoods alone still captures this much. Combining size and coordinates matches size on MiniWoB++ (0.63) and improves the four-benchmark average to 0.83 (size 0.74) and the three-environment average to 0.81 (0.72). Since $\bar { \ell }$ is weak and the coordinates are strong, what matters is not the overall level of the likelihood but its shape: which actions a model finds natural in which states. A success rate, however, is only one projection of behavior when run; we ask next whether that is all the coordinates capture.

Table 2: Predicting success rates without running the agents: Pearson r under leave-one-model-out over 100 models. Baselines use size (log parameter count), the family mean, or $\bar { \ell }$ (mean log-likelihood); coordinate methods are kNN, Ridge, and Ridge with size as a covariate (+size). Bold: higher than size, with a bootstrap 95% interval of the diference excluding zero. RMSE and all predictors in Table 7.
<table><tr><td></td><td colspan="3">Baselines</td><td colspan="3">Coordinates</td></tr><tr><td>Downstream task Size</td><td></td><td>Family</td><td>é</td><td>kNN</td><td>Ridge</td><td>+size</td></tr><tr><td>ALFWorld</td><td>.63</td><td>.46</td><td>.20</td><td>.88</td><td>.81</td><td>.82</td></tr><tr><td>TextWorld</td><td>.71</td><td>.44</td><td>.33</td><td>.76</td><td>.77</td><td>.78</td></tr><tr><td>BFCL</td><td>.48</td><td>.34</td><td>.32</td><td>.69</td><td>.69</td><td>.65</td></tr><tr><td>MiniWoB++</td><td>.64</td><td>.24</td><td>.22</td><td>.28</td><td>.56</td><td>.63</td></tr><tr><td>4-bench avg.</td><td>.74</td><td>.42</td><td>.34</td><td>.74</td><td>.78</td><td>.83</td></tr><tr><td>3 other envs.</td><td>.72</td><td>.40</td><td>.34</td><td>.72</td><td>.75</td><td>.81</td></tr></table>

## 3.3 Behavior: do neighbors act alike?

In §3.2, models that are close in coordinates had similar success rates. Do neighbors also share their behavior when run? Behavior can only be measured as a distance between a pair of models, so the question becomes whether the behavioral distance from a model to its coordinate neighbors is smaller than its distance to the remaining models. We look at neighbors rather than at the correlation over all pairs because the guarantee covers only close pairs (§2.4). Coordinate neighbors are compared with two kinds of neighbors. The first are neighbors chosen by size, family, or $\bar { \ell } ,$ which, as in Table $^ { 2 , }$ use only information available without running. The second are neighbors chosen by the measured success rate on that benchmark, a reference that is available only after running. Agents with the same success rate do not necessarily solve tasks in the same way, so the gap to the second kind shows what the coordinates capture beyond closeness in performance.

Setup. We use the action sequences that each model itself produced with the BASE harness: ALF-World valid\_unseen (50 tasks), the environment of the coordinates, and the other environments TextWorld (50), BFCL multi\_turn (25), and Mini-WoB++ (50), all for the 100 models. The behavioral distance between two models is the distance between their trajectories on the same task, averaged over tasks. Since similar behavior is not a single number, we measure it in four aspects from coarse to fine (definitions in Appendix B.4):

• Composition (which actions are used how often, ignoring order): Jensen–Shannon distance between the distributions of action types (Endres and Schindelin, 2003) (type JS)

• Global (agreement of the whole action sequence): normalized Levenshtein distance (Levenshtein, 1966; Rabanser et al., 2026) (edit distance)

• Local (what follows a given action): Jaccard distance between the sets of action 2-grams (2-gram)

• Early (whether the same first move is made from the same state): disagreement rate of the first action from the same initial state (Bouchard and Chauhan, 2026; Gonuguntla, 2026) (first action)

The diference in success rate is measured in the same form, which puts the results of §3.2 on the same scale. The efect size for a model’s five coordinate neighbors is the mean behavioral distance to the remaining models minus the mean distance to the neighbors, divided by the standard deviation over all pairs (positive: neighbors more similar). Figure 1(c) applies the same definition to the k-th nearest model alone, in coordinates and in size, averaged over the 100 models and the four aspects.

Models that behave alike can be found without running them. The five coordinate neighbors not only have similar success rates (0.39–0.61) but also behave alike on all four benchmarks (Table 3). The efect is roughly twice that of size neighbors (edit distance on ALFWorld 0.88 versus 0.40; 2-gram on MiniWoB++ 0.50 versus 0.27), and <sup>¯</sup>ℓ neighbors are weak (0.03–0.20). Family is a stronger competitor than size: coordinates exceed it on ALFWorld, TextWorld, and MiniWoB++ (except for the first action on TextWorld) and tie with it on BFCL. The efect remains when the coordinate neighbors are restricted to other families (edit distance on ALF-World 0.60), so it is not produced by same-family dependence. Similarity decreases with rank: the first coordinate neighbor is the most similar and the efect returns to the overall mean around rank 30, whereas size neighbors are nearly flat beyond the first (Figure 1(c)). The divergence position and the profile of idiosyncrasies show the same pattern, while for the success pattern, neighbors chosen by family or size are about as close as or closer than coordinate neighbors (Appendix B.4).

Table 3: Efect sizes of the behavioral similarity of each model’s five neighbors (§3.3; positive: neighbors more similar than the remaining models), for the diference in success rate and the four aspects of §3.3. Bold: largest of the four columns other than the reference. ${ } ^ { * } \colon p < 0 . 0 1 .$ <sup>+</sup>: $p < 0 . 0 5 ; { \mathrm { ? } } \colon$ significantly closer than success-rate neighbors $( p < 0 . 0 5 )$ . The success-rate row is omitted for success-rate neighbors. All measures in Table 8.
<table><tr><td></td><td colspan="3">Available without running</td><td>Map</td><td>Reference</td></tr><tr><td>Benchmark / measure</td><td>Size</td><td>Family</td><td>ē</td><td>Coord.</td><td>Success</td></tr><tr><td>ALFWorld</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success rate</td><td>.41*</td><td>.49*</td><td>.20*</td><td>.60*</td><td></td></tr><tr><td>Composition</td><td>.23*</td><td>.28*</td><td>.05</td><td> $. 5 2 ^ { * \dagger }$ </td><td> $. 3 0 ^ { * }$ </td></tr><tr><td>Global</td><td>.40*</td><td>.62*</td><td>.18+</td><td> $. 8 8 ^ { * \dagger }$ </td><td>.65*</td></tr><tr><td>Local</td><td>.41*</td><td>.68*</td><td>.20*</td><td> $\mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } } } \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } } } \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } } } \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } } }$ </td><td>.74*</td></tr><tr><td>Early</td><td>.30*</td><td>.49*</td><td>.11*</td><td> $\mathbf { \delta _ { \delta } } \mathbf { 1 } ^ { * \dagger }$ </td><td>.41*</td></tr><tr><td>TextWorld</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success rate</td><td>.58*</td><td>.56*</td><td>.27*</td><td>.61*</td><td></td></tr><tr><td>Composition</td><td>.10</td><td>.27*</td><td>.06</td><td>.33*†</td><td>.22*</td></tr><tr><td>Global</td><td>.25*</td><td>.42*</td><td>.14</td><td>.52*</td><td>.50*</td></tr><tr><td>Local</td><td>.27+</td><td>.44*</td><td>.14</td><td>.54*</td><td>.52*</td></tr><tr><td>Early</td><td>.14*</td><td>.25*</td><td>.04</td><td>.21*</td><td>.16*</td></tr><tr><td>BFCL (multi_turn)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success rate</td><td>.27*</td><td>.30*</td><td>.15*</td><td>.47*</td><td></td></tr><tr><td>Composition</td><td> $. 1 0 ^ { + }$ </td><td>.21*</td><td>.03</td><td>.18*</td><td>.23*</td></tr><tr><td>Global</td><td>.13*</td><td>.26*</td><td>.06</td><td>.26*</td><td>.30*</td></tr><tr><td>Local</td><td>.21*</td><td>.32*</td><td>.09+</td><td> $\mathbf { 3 8 ^ { * } }$ </td><td>.39*</td></tr><tr><td>Early</td><td>.19*</td><td>.25*</td><td>.09*</td><td> $. 3 2 ^ { * }$ </td><td>.21*</td></tr><tr><td>MiniWoB++</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success rate</td><td>.42*</td><td> $. 3 7 ^ { * }$ </td><td> $. 2 0 ^ { * }$ </td><td> $. 3 9 ^ { * }$ </td><td></td></tr><tr><td>Composition</td><td> $. 1 2 ^ { * }$ </td><td> $. 2 2 ^ { * }$ </td><td> $. 0 8 ^ { * }$ </td><td> $. 2 4 ^ { * \dagger }$ </td><td>.13*</td></tr><tr><td>Global</td><td>.20*</td><td> $. 3 2 ^ { * }$ </td><td> $. 1 0 ^ { + }$ </td><td> $\mathbf { 3 8 ^ { * } }$ </td><td> $. 3 7 ^ { * }$ </td></tr><tr><td>Local</td><td> $. 2 7 ^ { * }$ </td><td> $. 4 7 ^ { * }$ </td><td> $. 1 6 ^ { + }$ </td><td> $\boldsymbol { . 5 0 } ^ { * }$ </td><td> $. 4 2 ^ { * }$ </td></tr><tr><td>Early</td><td> $. 3 5 ^ { * }$ </td><td> $. 3 5 ^ { * }$ </td><td> $. 0 9 ^ { + }$ </td><td> $. 5 0 ^ { * \dagger }$ </td><td> $. 2 8 ^ { * }$ </td></tr></table>

Beyond closeness in performance. In ALF-World, the environment of the coordinates, coordinate neighbors behave more alike than neighbors chosen by the measured success rate in all four aspects († in Table 3); in the other environments the two are roughly on a par, with coordinates significantly closer only for composition on TextWorld and MiniWoB++ and for the first action on Mini-WoB++. We also regress out performance and size (Appendix B.4): because strong models may converge on the shortest procedures, we regress each pair’s behavioral distance on the diference in success rate, the pair’s mean success rate and its square, and the diference in log parameter count, and measure the neighbor efect on the residuals. In ALF-World the efect remains (edit distance 0.61, 2-gram

![](images/01fddd8135a5aeea71a3d63dec3f57e1f8945e8943746c6aa8fc6f35cac96e98.jpg)  
Figure 3: Harness displacements add up within an LLM. The 144 τ-bench agents (12 LLMs × 12 harnesses) in their first two principal components (PC1 separates LLMs; axes scaled independently). Squares: BASE; solid arrows: single-component displacements; dashed: their sum, ending at the predicted position of retrieve + think (open circles); filled circles: its measured position. Cosine and norm ratio: medians over the 12 LLMs in the full space.

0.68) and stays positive when the candidates are restricted to other families (0.31, 0.31); in the other environments it is weak, remains mainly in the local and early measures, and for some measures vanishes or changes sign. This linear control shows only that an association remains that the observed success rates and sizes do not explain. That the coordinates are built from the likelihood of the next action in a fixed state is consistent with the efect remaining in the local and early measures.

## 4 Composing Harness Efects in Agent Behavior Maps

Section 3 fixed the harness; this section fixes the LLM, varies the harness, and asks what structure the map has with respect to harnesses. A harness combines components for context management, planning, and acting. If adding a component appears as a displacement within an LLM and these displacements add up, the map reads as the position of the LLM plus a sum of component vectors, and the coordinates of an unscored combination follow from the scores of single components. We test whether the approximate additivity observed for prompt changes in LM maps (Oyama et al., 2026) also holds for agent harnesses (§4.2).

## 4.1 Harness interventions and setup

Harnesses. A harness splits the rule $C _ { i }$ that builds the LLM’s input from the external history $h _ { t }$

(§2.1) into three slots.

• Context management (which part of the history is shown): windowed passes the initial observation and the last four steps verbatim; retrieve adds past steps retrieved for their relevance; summarize replaces old history with a structured summary.

• Planning (whether a plan is rewritten before acting): none / replan.

• Acting (whether free-form reasoning is written before the action): direct / think.

System instructions, tool definitions, and the initial observation are shared by all harnesses; plans and reasoning enter only the current step’s input, so the external history consists of actions and observations only and every harness builds its input from the same $h _ { t } .$ , both when running and when scoring. The $3 \times 2 \times 2 = 1 2$ combinations include BASE (windowed/none/direct), four singlecomponent harnesses that add one component to BASE, and seven composed harnesses that add two or more (specifications in Appendix C.1).

Harness vectors. For the same LLM m, the difference between the coordinates of BASE and of the harness that adds component A, $\Delta _ { A } ^ { ( m ) } : = $ $z ( m , A ) - z ( m , \mathrm { b a s e } )$ , is the harness vector, and the sum of harness vectors predicts the coordinates of a composition, ${ \hat { z } } ( m , A { + } B ) : = z ( m , \mathrm { b a s e } ) +$ $\Delta _ { A } ^ { ( m ) } + \Delta _ { B } ^ { ( m ) }$ . Since the coordinates are centered log-likelihoods on the common reference trajectories, $\Delta _ { A }$ represents how component A changes the relative likelihood of the reference actions; the change in the mean log-likelihood is reported separately as a scalar (Appendix C.1).

Setup. All 12 configurations are scored, but only the coordinates of the five configurations BASE and single components enter the composition; the coordinates of the remaining seven are constructed as their sums. With 12 LLMs (listed in Appendix C.1), the coordinates of 60 agents thus yield composed coordinates for 84. The measured coordinates of the composed harnesses are used only for validation. The coordinates are obtained by teacher forcing on 359 HAL (Kapoor et al., 2026) traces of τ-bench (Yao et al., 2025); ALFWorld and SWE-Bench (6 LLMs) are in Appendix C.

## 4.2 Predicting coordinates of unscored compositions

Does the sum of the single-component displacements approach the measured displacement $z ( m , A { + } B ) { - } z ( m , \mathrm { b a s e } )$ of the composition? Figure 3 shows one composition, retrieve + think, as an example; all seven are in Figure 5 (Appendix C.2). For all compositions, Table 12 in the appendix reports, within each LLM, the cosine between the measured displacement and the predicted displacement $\Delta _ { A } + \Delta _ { B }$ , the norm ratio (measured / predicted), and the relative reconstruction error ∥measured−predicted∥/∥measured∥, as medians over LLMs. We compare displacements from BASE because absolute coordinates share a large component common to the LLM, which alone would make them look close. The arrows of the same component also look parallel across LLMs in Figure 3; this cross-LLM agreement of directions is not analyzed in this paper.

For two-component compositions, the median cosine over the 60 cases (12 LLMs × 5 compositions) is 0.90 and the median norm ratio 0.98, so the sum roughly matches in direction and length. The agreement varies by composition: retrieve with planning or reasoning agrees best (cosine 0.97 / 0.94, relative error 0.25 / 0.34), compositions with summarize keep the direction but deviate more (cosine 0.91 / 0.87), and replan + think is weakest (cosine 0.81, norm ratio 0.74: the measured displacement is shorter than predicted). Three-component compositions reach cosine 0.82 and norm ratio 0.87 (median over 2 × 12 cases). On ALFWorld and SWE-Bench (6 LLMs) the agreement weakens to median two-component cosines of 0.63 and 0.81 (Table 12). Additivity thus depends on the components and the environment, but for context management combined with planning or reasoning, the direction and magnitude of the composed displacement can be approximated from single-component displacements.

## 5 Related Work

Maps of language models and policies. LM maps (Oyama et al., 2025) place models in a common coordinate system by their log-likelihoods on fixed texts and have been extended to prompt– response distributions (Oyama et al., 2026); other works embed models by their outputs or correctness (Wu et al., 2026; Yax et al., 2025; Zhuang et al., 2025). All of them address static LMs. In reinforcement learning, policies are compared by their actions on a fixed set of states (Parker-Holder et al., 2020; Pacchiano et al., 2020; Konyushkova et al., 2021). We extend LM maps to closed-loop agents and quantify the cost of measuring on fixed trajectories as the term λ, which accumulates over steps like the compounding error of imitation learning (Ross and Bagnell, 2010; Ross et al., 2011). Predicting performance from scores on cheaper benchmarks (Ruan et al., 2024; Song et al., 2026) shares our goal but needs the target model’s scores, whereas our coordinates need neither scores nor environment runs.

Harness efects and agent behavior. Harness efects have been measured as changes in success rate by running every LLM–harness combination (Zhang et al., 2026; Yao et al., 2026; Fan et al., 2026; Cho, 2026; Chen et al., 2026) or decomposed into LLM and scafold abilities with IRT (Ge et al., 2026); we represent a component as a displacement on the map, obtained without running the composed agent, and ask whether displacements add (§4). Evaluations increasingly report consistency across trials, cost, and behaviors found in logs alongside success rates (Yao et al., 2025; Kapoor et al., 2025, 2026); harnesses with similar success rates can difer greatly in token use and failure patterns (Vats and Golev, 2026), and agent behavior carries fingerprints of the underlying LLM (Sun et al., 2025; Wang et al., 2026; Oderinwale, 2026; Lugoloobi et al., 2026). All of these works run the agents and compare the resulting behavior through aggregate statistics or classifiers. We measure behavioral diferences by distances obtained without running the agents, relate them to the trajectory KL with explicit error bounds (§2), and show that pairs at small distance behave alike when run (§3.3).

## 6 Conclusion

Agent Behavior Maps, obtained by teacher forcing on common reference trajectories, locate agents without running them. Their coordinates predict the success rates of models that were never run, on the benchmark the map was built from and, combined with size, in other environments. Neighbors on the map take similar action sequences when run. The theory decomposes the gap between the distances and the trajectory KL into four error sources and limits the guarantee to close pairs. Harness components appear as displacements within an LLM whose sums approximate the coordinates of unscored combinations, with an accuracy that depends on the component and the environment. The map represents not only a ranking by performance but also how close agents are in what they do.

## Limitations

Locality. Both the theoretical guarantee and the empirical correspondence are limited to close pairs. The KL can be read of the distance only for pairs in the local regime of small ε and λ, and pairs of diferent LLMs are not guaranteed to fall in it: the estimates exceed the measured KL by a median factor of 7.5, although their order agrees (§3.1). The correspondence with behavioral distance holds for neighbors and fades beyond about the 30th neighbor (Figure 1(c)); we have not shown that the magnitude of the distance between far-apart pairs measures how diferent their behavior is. The map answers who is similar to whom, not by how much.

Dependence on the reference trajectories. Differences between policies in regions that Q never visits cannot be measured, and the reference mismatch λ remains a sensitivity parameter that the likelihood matrix alone does not determine. A general theory of the reference policy is left for future work.

Estimation of the KL. The trajectory loglikelihood ratios in §3.1 are based on the probability of the canonical token sequence and difer from action probabilities that marginalize over all tokenizations. In environments where the candidate set can be enumerated, this can be resolved by renormalization, but in general it is a limitation.

Access to log-likelihoods. The coordinates require the target model’s log-likelihood of a given action sequence. Models that are available only through an API that does not return the logprobabilities of given text cannot be placed on the map, so this study is limited to open-weight models.

Sample size of behavior. The behavioral comparisons are based on one rollout per task, and the behavior of the same model varies from run to run (Bjarnason et al., 2026; Gonuguntla, 2026). We have not compared the size of this variation with the behavioral distance to neighbors.

Scope of the harness experiment. The harnesses in §4 are limited to the 12 configurations of our own three-slot design (context management, planning, acting) and to 12 LLMs; the main additivity results are on τ -bench, and ALFWorld and SWE-Bench cover 6 LLMs. Whether the same additivity holds for other scafolds with diferently designed components has not been tested.

## References

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. 2026. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. In International Conference on Machine Learning (ICML).

Bjarni Haukur Bjarnason, André Silva, and Martin Monperrus. 2026. On randomness in agentic evals. arXiv preprint arXiv:2602.07150.

Dylan Bouchard and Mohit Singh Chauhan. 2026. Beyond single-turn confidence: Trajectory-adapted uncertainty quantification for LLM agents. arXiv preprint arXiv:2608.11552.

Jixuan Chen, Jianzhi Shen, Haoqiang Kang, Zhi Hong, Qingyi Jiang, Soham Bose, Yiming Zhang, Leon Leng, Amit Vyas, Lingjun Mao, Siru Ouyang, Kun Zhou, and Lianhui Qin. 2026. AgentSpec: Understanding embodied agent scafolds through controlled composition. Preprint, arXiv:2606.14674.

Yong-eun Cho. 2026. It’s not the capability: Harness sensitivity is non-monotone across LLM agent tiers. Preprint, arXiv:2605.26731.

Marc-Alexandre Côté, Ákos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Ruo Yu Tao, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. 2018. TextWorld: A learning environment for textbased games. In Computer Games Workshop at IJCAI 2018.

Thomas M. Cover and Joy A. Thomas. 2006. Elements of Information Theory, 2nd edition. Wiley-Interscience.

Dominik M. Endres and Johannes E. Schindelin. 2003. A new metric for probability distributions. IEEE Transactions on Information Theory, 49(7):1858– 1860.

Run-Ze Fan, Zihao Zhang, Simin Ma, Yebowen Hu, Shouju Wang, Kaiqiang Song, Fei Liu, Hamed Zamani, and Xiaoyang Wang. 2026. An empirical study of harness design for coding agents. Preprint, arXiv:2609.20804.

Chris Ge, Daria Kryvosheieva, Daniel Fried, Uzay Girit, and Kaivalya Hariharan. 2026. Agent psychometrics: Task-level performance prediction in agentic coding benchmarks. In Conference on Language Modeling (COLM).

Gemma Team. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Luca Gioacchini, Giuseppe Siracusano, Davide Sanvito, Kiril Gashteovski, David Friede, Roberto Bifulco, and Carolin Lawrence. 2024. AgentQuest: A modular benchmark framework to measure progress and improve LLM agents. In Proceedings of the 2024 Conference of the North American Chapter of the

Associationfor Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations), pages 185–193, Mexico City, Mexico. Association for Computational Linguistics.

Ashritha Gonuguntla. 2026. The replay gap: Static evaluation of model switching in LLM agents scores the wrong world. In Eficient Reasoning Workshop at the Conference on Language Modeling (COLM). ArXiv:2608.08239.

Wassily Hoefding. 1963. Probability inequalities for sums of bounded random variables. Journal ofthe American Statistical Association, 58(301):13–30.

Arthur E. Hoerl and Robert W. Kennard. 1970. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR).

Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S. Siegel, Boyi Wei, Tianci Xue, Ziru Chen, Felix Chen, Saiteja Utpala, Franck Ndzomga, Dheeraj Oruganty, Sophie Luskin, Kangheng Liu, Botao Yu, Amit Arora, Dongyoon Hahm, Harsh Trivedi, Huan Sun, and 12 others. 2026. Holistic agent leaderboard: The missing infrastructure for AI agent evaluation. In International Conference on Learning Representations (ICLR).

Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. 2025. AI agents that matter. Transactions on Machine Learning Research.

Liyiming Ke, Sanjiban Choudhury, Matt Barnes, Wen Sun, Gilwoo Lee, and Siddhartha Srinivasa. 2020. Imitation learning as f-divergence minimization. In International Workshop on the Algorithmic Foundations ofRobotics (WAFR).

Ksenia Konyushkova, Yutian Chen, Tom Le Paine, Caglar Gulcehre, Cosmin Paduraru, Daniel J. Mankowitz, Misha Denil, and Nando de Freitas. 2021. Active ofline policy selection. In Advances in Neural Information Processing Systems (NeurIPS), volume 34.

Vladimir I. Levenshtein. 1966. Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8):707–710.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. In International Conference on Learning Representations (ICLR).

William Lugoloobi, Samuelle Marro, Jabez Magomere, Joss Wright, and Chris Russell. 2026. Known by their actions: Fingerprinting LLM browser agents via UI traces. Preprint, arXiv:2605.14786.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. AgentBoard: An analytical evaluation board of multi-turn LLM agents. In Advances in Neural Information Processing Systems (Datasets and Benchmarks). ArXiv:2401.13178.

Hamidah Oderinwale. 2026. Agent trajectories as programs: Fingerprinting and programming codingagent behavior. Preprint, arXiv:2606.16988.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

Momose Oyama, Yusuke Takase, and Hidetoshi Shimodaira. 2026. Language model maps for promptresponse distributions via log-likelihood vectors. Preprint, arXiv:2603.18593. To appear in Proceedings of EMNLP 2026.

Momose Oyama, Hiroaki Yamagiwa, Yusuke Takase, and Hidetoshi Shimodaira. 2025. Mapping 1,000+ language models via the log-likelihood vector. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 32983–33038, Vienna, Austria. Association for Computational Linguistics.

Aldo Pacchiano, Jack Parker-Holder, Yunhao Tang, Anna Choromanska, Krzysztof Choromanski, and Michael I. Jordan. 2020. Learning to score behaviors for guided policy optimization. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7445–7454.

Jack Parker-Holder, Aldo Pacchiano, Krzysztof Choromanski, and Stephen Roberts. 2020. Efective diversity in population based reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 33.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. 2025. The Berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 48371–48392.

Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id= qwen3.5.

Qwen Team. 2026b. Qwen3.8. https://github. com/QwenLM/Qwen3.8.

Stephan Rabanser, Sayash Kapoor, Peter Kirgis, Kangheng Liu, Saiteja Utpala, and Arvind Narayanan. 2026. Towards a science of AI agent reliability.

In International Conference on Machine Learning (ICML).

Abhinav Rastogi et al. 2025. Devstral: Fine-tuning language models for coding agent applications. Preprint, arXiv:2509.25193.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389.

Stéphane Ross and Drew Bagnell. 2010. Eficient reductions for imitation learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings ofMachine Learning Research, pages 661–668.

Stéphane Ross, Geofrey Gordon, and Drew Bagnell. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 627–635.

Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto. 2024. Observational scaling laws and the predictability of language model performance. In Advances in Neural Information Processing Systems (NeurIPS), volume 37.

Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. World of bits: An open-domain platform for web-based agents. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3135–3144.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations (ICLR).

Yueqi Song, Lintang Sutawika, Jiarui Liu, Lindia Tjuatja, Jiayi Geng, Yunze Xiao, Daniel Lee, Aditya Bharat Soni, Vincent Lo, Xiang Yue, and Graham Neubig. 2026. PACE: A proxy for agentic capability evaluation. Preprint, arXiv:2607.02032.

Giovanni Strona, Domenico Nappo, Francesco Boccacci, Simone Fattorini, and Jesus San-Miguel-Ayanz. 2014. A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals. Nature Communications, 5:4114.

Mingjie Sun, Yida Yin, Zhiqiu Xu, J. Zico Kolter, and Zhuang Liu. 2025. Idiosyncrasies in large language models. In Proceedings of the 42nd International Conference on Machine Learning (ICML).

Laurens van der Maaten and Geofrey Hinton. 2008. Visualizing data using t-SNE. Journal ofMachine Learning Research, 9(86):2579–2605.

Naman Vats and Oleg Golev. 2026. The scafold efect in coding agents: Harness choice as a hidden variable in coding-agent evaluation. Preprint, arXiv:2607.22585.

Chuyi Wang, Xiaohui Xie, Tongze Wang, Fangchen Luo, and Yong Cui. 2026. Who is behind the harness? fingerprinting LLMs through agentic behavior. Preprint, arXiv:2609.28559.

Frank Wilcoxon. 1945. Individual comparisons by ranking methods. Biometrics Bulletin, 1(6):80–83.

Ronald J. Williams and David Zipser. 1989. A learning algorithm for continually running fully recurrent neural networks. Neural Computation, 1(2):270–280.

Zhaomin Wu, Haodong Zhao, Ziyang Wang, Jizhou Guo, Qian Wang, and Bingsheng He. 2026. LLM DNA: Tracing model evolution via functional representations. In International Conference on Learning Representations (ICLR).

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), volume 37.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2025. τ-bench: A benchmark for toolagent-user interaction in real-world domains. In International Conference on Learning Representations (ICLR).

Yilun Yao, Xinyu Tan, Chao-Hsuan Liu, Yaoming Li, Zhengyang Wang, Wenhan Yu, Zhewen Tan, Yuxuan Tian, Guangxiang Zhao, Lin Sun, Xiangzheng Zhang, and Tong Yang. 2026. Harness-Bench: Measuring harness efects across models in realistic agent workflows. Preprint, arXiv:2605.27922.

Nicolas Yax, Pierre-Yves Oudeyer, and Stefano Palminteri. 2025. PhyloLM: Inferring the phylogeny of large language models and predicting their performances in benchmarks. In International Conference on Learning Representations (ICLR).

Z.ai. 2026. GLM-4.7-Flash. https://huggingface. co/zai-org/GLM-4.7-Flash.

Yunbei Zhang, Janet Wang, Yingqiang Ge, Weijie Xu, Jihun Hamm, and Chandan K. Reddy. 2026. Stop comparing LLM agents without disclosing the harness. Preprint, arXiv:2605.23950.

Richard Zhuang, Tianhao Wu, Zhaojin Wen, Andrew Li, Jiantao Jiao, and Kannan Ramchandran. 2025. EmbedLLM: Learning compact representations of large language models. In International Conference on Learning Representations (ICLR).

## A Theory: proofs and scoring details

This appendix supplements §2. Appendix A.1 defines the policy and the scoring convention; Appendices A.2–A.5 prove Theorem 1, Corollary 1, and the cross-term identity (30). Notation follows §2.3.

## A.1 Grammar-constrained action probabilities

We define action probabilities under grammarconstrained decoding. A deployed agent can execute only outputs that match the environment’s action format, so its outputs are typically constrained to an action grammar, e.g., by constrained decoding. Our grammar constraint reflects this setting: the constrained policy is the distribution of actions the agent actually sends to the environment.

Logged actions are serialized by a common normalization map $\mathcal { N }$ that preserves their efect on the environment. Let $\boldsymbol { \mathcal { A } } ( \boldsymbol { h } )$ be the set of action strings that satisfy the grammar under history h. Agent i reconstructs its input $C _ { i } ( h )$ from h with its rule $C _ { i }$ (§2). Let $q _ { i } ( v \mid C _ { i } ( h ) , v _ { < k } )$ be the model’s nexttoken probability given this input and a prefix $v _ { < k }$ and let $\scriptstyle \gamma ( h , v < k )$ be the set of tokens allowed by the grammar after $v _ { < k }$ . Grammar-constrained decoding sets the probabilities of disallowed tokens to zero and renormalizes the rest: for $v _ { k } \in \mathcal { V } ( h , v _ { < k } )$

$$
q _ { i } ^ { \mathcal { G } } ( v _ { k } \mid h , v _ { < k } ) = \frac { q _ { i } ( v _ { k } \mid C _ { i } ( h ) , v _ { < k } ) } { \displaystyle \sum _ { v \in \mathcal { V } ( h , v _ { < k } ) } q _ { i } ( v \mid C _ { i } ( h ) , v _ { < k } ) } .\tag{5}
$$

The probability of an action a is the sum over all token sequences v that normalize to a,

$$
\pi _ { i } ( a \mid h ) = \sum _ { v : \mathcal { N } ( \operatorname* { d e c o d e } ( v ) ) = a } \ \prod _ { k } q _ { i } ^ { \mathcal { G } } ( v _ { k } \mid h , v _ { < k } ) ,\tag{6}
$$

which satisfies $\begin{array} { r } { \sum _ { a \in \mathcal { A } ( h ) } \pi _ { i } ( a \mid h ) = 1 } \end{array}$

The trajectory-KL comparison in §3.1 uses grammar-constrained scoring; the experiments from §3.2 onward score without the grammar mask.

## A.2 Proof of Theorem 1

Fix a pair $( i , j )$ . By the chain rule (2), $K _ { i j }$ is the expected sum of per-step local KLs. Its gap to $\textstyle { \frac { \bar { T } } { 2 } } D _ { S } ^ { 2 }$ and ${ \scriptstyle \frac { 1 } { 2 } } D _ { \tau } ^ { 2 }$ has three sources:

(a) the remainder of the quadratic approximation of the local KL (Lemma 1);

(b) taking the quadratic term under the reference Q instead of the on-policy $P _ { i }$ (Lemma 2);

(c) centering.

We prove the two lemmas and then the theorem with explicit constants.

Lemma 1 (Quadratic approximation of the local KL). Let p, q be distributions with a common support, let $Y = \log ( p / q )$ with $| Y | \leq \varepsilon ,$ , and write $Y _ { \pm } = \operatorname* { m a x } ( \pm Y , 0 )$ . Then the following hold.

(i) Two-sided bound:

$$
\begin{array} { r } { - \frac 1 6 \mathbb { E } _ { p } [ Y ^ { 3 } ] \le \mathrm { K L } ( p \| q ) - \frac 1 2 \mathbb { E } _ { p } [ Y ^ { 2 } ] \le - \frac 1 6 \mathbb { E } _ { q } [ Y ^ { 3 } ] . } \end{array}\tag{7}
$$

(ii) Size ofthe remainder:

$$
\begin{array} { r l r } & { } & { \left| { \mathrm { K L } } ( p \| q ) - \frac { 1 } { 2 } { \mathbb E } _ { p } [ Y ^ { 2 } ] \right| \leq \frac { 1 } { 6 } \operatorname* { m a x } \{ { \mathbb E } _ { p } [ Y _ { + } ^ { 3 } ] , { \mathbb E } _ { q } [ Y _ { - } ^ { 3 } ] \} } \\ & { } & { \qquad \quad ( 8 ) } \end{array}
$$

(iii) Size of the local $K L \colon 0 \leq \mathrm { K L } ( p \| q ) \leq \varepsilon ^ { 2 } / 2 .$ Proof. Let $g ( y ) = e ^ { - y } - 1 + y - y ^ { 2 } / 2$ . Since $\begin{array} { r } { \mathbb { E } _ { p } [ e ^ { - Y } ] = \sum q = 1 } \end{array}$

$$
\begin{array} { r } { \mathrm { K L } ( p \| q ) - \frac { 1 } { 2 } \mathbb { E } _ { p } [ Y ^ { 2 } ] = \mathbb { E } _ { p } [ g ( Y ) ] , } \end{array}\tag{9}
$$

so it sufices to bound the right-hand side. By Taylor’s theorem, $\begin{array} { r } { g ( y ) = - \frac { y ^ { 3 } } { 6 } e ^ { - \xi } } \end{array}$ for some $\xi$ between 0 and $y . \mathrm { H } y \geq 0$ then $e ^ { - \check { \xi } } \in [ e ^ { - y } , 1 ]$ , and if $y < 0$ then $e ^ { - \xi } \in [ 1 , e ^ { - y } ]$ ; in either case

$$
\begin{array} { r } { - \frac { y ^ { 3 } } { 6 } \le g ( y ) \le - \frac { y ^ { 3 } } { 6 } e ^ { - y } . } \end{array}\tag{10}
$$

Averaging over p and applying the change of measure $\mathbb { E } _ { p } [ e ^ { - Y } f ( Y ) ] = \mathbb { E } _ { q } [ f ( Y ) ]$ to the right-hand side, (9) gives (7).

Next, split (10) by the sign of Y. For $Y \geq 0$ $- { \textstyle \frac { 1 } { 6 } } Y _ { + } ^ { 3 } \leq g ( Y ) \leq 0 ;$ for $Y < 0 , 0 \leq g ( Y ) \leq$ ${ \textstyle \frac { 1 } { 6 } } e ^ { - Y } Y _ { - } ^ { 3 } .$ . Hence, with $A ~ = ~ \mathbb { E } _ { p } [ g ( Y ) \mathbf { 1 } \{ Y ~ <$ 0}] and $B ~ = ~ - \mathbb { E } _ { p } [ g ( Y ) \mathbf { 1 } \{ Y ~ \geq ~ 0 \} ]$ , we have $\mathbb { E } _ { p } [ g ( Y ) ] = A - B$ and

$$
\begin{array} { r l } & { 0 \leq A \leq \frac { 1 } { 6 } \mathbb { E } _ { p } [ e ^ { - Y } Y _ { - } ^ { 3 } ] = \frac { 1 } { 6 } \mathbb { E } _ { q } [ Y _ { - } ^ { 3 } ] , } \\ & { 0 \leq B \leq \frac { 1 } { 6 } \mathbb { E } _ { p } [ Y _ { + } ^ { 3 } ] . } \end{array}\tag{11}
$$

As a diference of two nonnegative terms, $\left| A - B \right| \leq$ max $\{ A , B \}$ , which is the first inequality in (8); the second follows from $| Y | \le \varepsilon$ . Splitting by sign, rather than bounding $e ^ { - \xi }$ uniformly by $e ^ { \varepsilon }$ , avoids a factor $e ^ { \varepsilon }$

Finally, we bound the local KL. Let $F ( s ) =$ log $\mathbb { E } _ { p } [ e ^ { - s Y } ]$ Then $\begin{array} { r l r } { F ( 0 ) } & { { } = } & { 0 , F ( 1 ) } \end{array}$ log $\begin{array} { r } { \mathbb { E } _ { p } [ e ^ { - Y } ] ~ = ~ 0 } \end{array}$ , and $\begin{array} { r } { { \cal F } ^ { \prime } ( 0 ) ~ = ~ - \mathbb { E } _ { p } [ { \cal Y } ] } \end{array}$

$\cdot \mathrm { K L } ( p \| q )$ . Moreover, $F ^ { \prime \prime } ( s )$ is the variance of $Y$ under the distribution proportional to $p e ^ { - s Y }$ , so

$0 \leq F ^ { \prime \prime } ( s ) \leq \varepsilon ^ { 2 }$ . By Taylor’s theorem, for some $s _ { 0 } \in ( 0 , 1 )$

$$
\begin{array} { r } { 0 = F ( 1 ) = - \mathrm { { K L } } ( p \| q ) + \frac { 1 } { 2 } F ^ { \prime \prime } ( s _ { 0 } ) , } \end{array}\tag{12}
$$

which gives $0 \leq \mathrm { K L } ( p \| q ) \leq \varepsilon ^ { 2 } / 2$

Lemma 2 (Diference of expectations of a bounded function). $I f L \leq f \leq U$ , then $| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \leq$ $\left( U - L \right) \mathrm { T V } ( P , Q )$

Proof. Trajectories take values in a countable set, so $\begin{array} { r } { \mathrm { T V } ( P , Q ) = \frac { 1 } { 2 } \sum _ { \tau } \lvert P ( \tau ) - Q ( \tau ) \rvert } \end{array}$ . Let $c =$ $( L + U ) / 2 ;$ then $| \bar { \boldsymbol { f } } - \boldsymbol { c } | \leq ( U - L ) / 2$ , and since $\begin{array} { r } { \sum _ { \tau } \{ P ( \tau ) - Q ( \tau ) \} = 0 } \end{array}$

$$
\begin{array} { r l } { | \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | = \Big | \sum _ { \tau } ( f - c ) ( P - Q ) ( \tau ) \Big | } & { } \\ { \le \frac { U - L } { 2 } \sum _ { \tau } | P ( \tau ) - Q ( \tau ) | } & { } \\ { = ( U - L ) \operatorname { T V } ( P , Q ) . \quad { \scriptstyle ( 1 ; } } \end{array}\tag{3}
$$

Proof of Theorem 1. We show, with explicit constants, that $\begin{array} { r } { | K _ { i j } - \frac { \bar { T } } { 2 } D _ { S } ^ { 2 } | \le R _ { S } } \end{array}$ and $| K _ { i j } - { \textstyle \frac { 1 } { 2 } } D _ { \tau } ^ { 2 } | \le$ $R _ { \tau }$ , where

$$
\begin{array} { r } { R _ { S } = \frac { \varepsilon ^ { 3 } } { 6 } \mathbb { E } _ { i } [ T ] + \frac { \lambda T _ { \mathrm { m a x } } \varepsilon ^ { 2 } } { 2 } + \frac { \mu ^ { 2 } } { 2 \bar { T } } , } \end{array}\tag{14}
$$

$$
\begin{array} { r l } & { R _ { \tau } = \frac { \varepsilon ^ { 3 } } { 6 } \mathbb { E } _ { i } [ T ] + \frac { \varepsilon ^ { 3 } } { 4 } \mathbb { E } _ { i } [ T ( T - 1 ) ] } \\ & { \quad \quad + \frac { \lambda T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 } } { 2 } + \frac { \mu ^ { 2 } } { 2 } . } \end{array}\tag{15}
$$

The sum of the first two terms of (15) is $\frac { \varepsilon ^ { 3 } } { 4 } \mathbb { E } _ { i } [ T ^ { 2 } ] -$ $\begin{array} { r } { \frac { \varepsilon ^ { 3 } } { 1 2 } \mathbb { E } _ { i } [ T ] \leq \frac { \varepsilon ^ { 3 } } { 4 } \mathbb { E } _ { i } [ T ^ { 2 } ] } \end{array}$ , which gives the form in (4) and Table 1.

For $K _ { i j }$ only the parts of Assumptions 1 and 2 concerning $P _ { i }$ and $Q$ are used; the parts concerning $P _ { j }$ enter through the symmetric KL below. We set $d _ { t } = 0$ after termination, and write $\begin{array} { r } { Z = \sum _ { t < T } d _ { t } ^ { 2 } } \end{array}$ and $\mathcal { F } _ { t } = \sigma ( h _ { t } )$ . Whether the episode is still running at time t is determined by $h _ { t } .$ , so $\{ t < T \}$ is $\mathcal { F } _ { t }$ -measurable.

Step version. At each step $t \ < \ T ,$ , set $p \ =$ $\pi _ { i } ( \cdot ~ | ~ h _ { t } )$ and $q ~ = ~ \pi _ { j } ( \cdot ~ \mid ~ h _ { t } )$ ; then $Y ~ = ~ d _ { t }$ and $\mathbb { E } _ { p } [ Y ^ { 2 } ] ~ = ~ \mathbb { E } _ { i } [ d _ { t } ^ { 2 } ~ | ~ { \mathcal F } _ { t } ]$ . By Lemma 1(ii), $\begin{array} { r } { \mathrm { K L } ( \pi _ { i } ( \cdot  { \mathbf { \varepsilon } } | h _ { t } ) \| \pi _ { j } ( \cdot  { \mathbf { \varepsilon } } | h _ { t } ) ) = \frac { 1 } { 2 } \mathbb { E } _ { i } [ d _ { t } ^ { 2 }  { \mathbf { \varepsilon } } | \mathcal { F } _ { t } ] + r _ { t } } \end{array}$ with $| r _ { t } | \le \varepsilon ^ { 3 } / 6$ . Substituting this into the chain rule (2) and using the tower property gives

$$
\begin{array} { r } { K _ { i j } = \frac 1 2 \mathbb { E } _ { i } Z + r _ { * } , \qquad | r _ { * } | \le \frac { \varepsilon ^ { 3 } } { 6 } \mathbb { E } _ { i } [ T ] . } \end{array}\tag{16}
$$

On the other hand, $\begin{array} { r c l } { \bar { T } D _ { S } ^ { 2 } } & { = } & { \mathbb { E } _ { Q } \sum _ { t < T } ( d _ { t } } \end{array}$ $\begin{array} { r } { \mu / \bar { T } ) ^ { 2 } = \mathbb { E } _ { Q } Z - 2 \frac { \mu } { \bar { T } } \mathbb { E } _ { Q } X + \frac { \mu ^ { 2 } } { \bar { T } ^ { 2 } } \mathbb { E } _ { Q } [ T ] = \mathbb { E } _ { Q } Z - } \end{array}$ $\mu ^ { 2 } / \bar { T }$ , so

$$
\begin{array} { r } { K _ { i j } - \frac { \bar { T } } { 2 } D _ { S } ^ { 2 } = r _ { * } + \frac { 1 } { 2 } ( \mathbb { E } _ { i } Z - \mathbb { E } _ { Q } Z ) + \frac { \mu ^ { 2 } } { 2 \bar { T } } . } \end{array}\tag{17}
$$

Lemma 2 with $0 \leq Z \leq T _ { \mathrm { m a x } } \varepsilon ^ { 2 }$ bounds the second term by $\scriptstyle { \frac { 1 } { 2 } } \lambda T _ { \mathrm { m a x } } \varepsilon ^ { 2 }$ , and the triangle inequality gives (14).

Trajectory version. Let $k _ { t } = \mathbb { E } _ { i } [ d _ { t } \ | \ { \mathcal F } _ { t } ]$ . For $t < T , k _ { t } = \mathrm { K L } ( \pi _ { i } ( \cdot \mid h _ { t } ) \| \pi _ { j } ( \cdot \mid h _ { t } ) )$ , and for $t \geq T , k _ { t } = 0 ;$ by Lemma 1(iii), $0 ~ \leq ~ k _ { t } ~ \leq$ ${ \bf 1 } \{ t < T \} \varepsilon ^ { 2 } / 2$ . For $s < t , d _ { s }$ is F<sub>t</sub>-measurable, so $\mathbb { E } _ { i } [ d _ { s } d _ { t } ] = \mathbb { E } _ { i } [ d _ { s } k _ { t } ]$ ; together with $| d _ { s } | \le \varepsilon$

$$
\begin{array} { r } { \left| \sum _ { s < t } \mathbb { E } _ { i } [ d _ { s } d _ { t } ] \right| \le \frac { \varepsilon ^ { 3 } } { 2 } \mathbb { E } _ { i } \left[ \binom { T } { 2 } \right] = \frac { \varepsilon ^ { 3 } } { 4 } \mathbb { E } _ { i } [ T ( T - 1 ) ] . } \end{array}\tag{18}
$$

Since $\begin{array} { r } { X ^ { 2 } = Z + 2 \sum _ { s < t } d _ { s } d _ { t } } \end{array}$ , we have $\begin{array} { r l } { { \frac { 1 } { 2 } } { \mathbb E } _ { i } Z = } & { { } } \end{array}$ $\begin{array} { r } { \frac 1 2 \mathbb { E } _ { i } X ^ { 2 } - \sum _ { s < t } \mathbb { E } _ { i } [ \overline { { d _ { s } } } \breve { d } _ { t } ] ; } \end{array}$ ; combining this with (16) and $D _ { \tau } ^ { 2 } = \mathbb { E } _ { Q } \breve { X } ^ { 2 } - \mu ^ { 2 }$ gives

$$
\begin{array} { r l } & { K _ { i j } - \frac { 1 } { 2 } D _ { \tau } ^ { 2 } = r _ { * } - \sum _ { s < t } \mathbb { E } _ { i } [ d _ { s } d _ { t } ] } \\ & { \qquad + \frac { 1 } { 2 } ( \mathbb { E } _ { i } X ^ { 2 } - \mathbb { E } _ { Q } X ^ { 2 } ) + \frac { \mu ^ { 2 } } { 2 } . } \end{array}\tag{19}
$$

Since $0 \leq X ^ { 2 } \leq T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 }$ , Lemma 2 bounds the third term by $\scriptstyle { \frac { 1 } { 2 } } \lambda T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 }$ , giving (15). The proof assumes neither independence across steps nor independence between the episode length and the actions.

Symmetric KL. Swapping $( i , j )$ maps $d _ { t } \mapsto$ $- d _ { t } , X \ \mapsto \ - X$ , and $\mu \mapsto - \mu$ , and leaves $Z .$ $X ^ { 2 } , { \bar { T } }$ , and hence $D _ { S } ^ { 2 }$ and $D _ { \tau } ^ { 2 }$ unchanged (the same $Q$ is used in both directions). The proof holds with $\lambda ~ = ~ \mathrm { T V } ( P _ { i } , Q )$ , and the bound for $( j , i )$ replaces $\mathbb { E } _ { i } [ T ]$ and $\mathrm { T V } ( P _ { i } , Q )$ by $\mathbb { E } _ { j } [ T ]$ and $\operatorname { T V } ( P _ { j } , Q )$ . Averaging the two bounds gives, with $\lambda _ { \mathrm { s y m } } : = \{ \mathrm { T V } ( P _ { i } , Q ) + \mathrm { T V } ( P _ { j } , Q ) \} / 2 \leq \lambda .$

$$
\begin{array} { r } { \left. K _ { \mathrm { s y m } } - \frac { \bar { T } } { 2 } D _ { S } ^ { 2 } \right. \leq \frac { \varepsilon ^ { 3 } } { 1 2 } ( \mathbb { E } _ { i } [ T ] + \mathbb { E } _ { j } [ T ] ) } \\ { + \frac { \lambda _ { \mathrm { s y m } } T _ { \mathrm { m a x } } \varepsilon ^ { 2 } } { 2 } + \frac { \mu ^ { 2 } } { 2 \bar { T } } , } \end{array}\tag{20}
$$

and the trajectory version is analogous. □

## A.3 Proof of Corollary 1

Both $\hat { K } _ { S }$ and $\hat { K } _ { \tau }$ are functions of four sample means over the reference trajectories. We bound these means simultaneously by Hoefding’s inequality, propagate the errors to the two distances, and add the approximation error of Theorem 1. As in Appendix A.2, Assumption 1 is assumed on the supports of both $P _ { i }$ and $Q$ .

The estimators as sample means. For each reference trajectory $\tau _ { r } .$ let $\begin{array} { r } { X _ { r } \ = \ \sum _ { t < T _ { r } } d _ { r t } } \end{array}$ and $\begin{array} { r } { Z _ { r } = \sum _ { t < T _ { r } } d _ { r t } ^ { 2 } } \end{array}$ , let an overline denote the sample mean over $r ,$ and let $\begin{array} { r } { { \hat { \bar { T } } } = N ^ { - 1 } \sum _ { r } T _ { r } } \end{array}$ . The diference of the two agents’ centering means is ${ \hat { c } } = { \overline { { X } } } / { \hat { \bar { T } } }$ , so the distances in $\ S 2 . 3$ are

$$
\hat { \bar { T } } \hat { D } _ { S } ^ { 2 } = \overline { { { Z } } } - \overline { { { X } } } ^ { 2 } / \hat { \bar { T } } , \qquad \hat { D } _ { \tau } ^ { 2 } = \overline { { { X ^ { 2 } } } } - \overline { { { X } } } ^ { 2 } .\tag{21}
$$

Likewise, in the population,

$$
\bar { T } D _ { S } ^ { 2 } = \mathbb { E } _ { Q } Z - \mu ^ { 2 } / \bar { T } , \qquad D _ { \tau } ^ { 2 } = \mathbb { E } _ { Q } X ^ { 2 } - \mu ^ { 2 }\tag{22}
$$

(Appendix A.2). It thus sufices to bound the sample means of $F \in \{ Z , X , X ^ { 2 } , T \}$ simultaneously.

Simultaneous bound on the four sample means. Since $| d _ { t } | \leq \varepsilon \ \mathrm { a n d } \ 1 \leq T \leq T _ { \operatorname* { m a x } }$ , the statistics have range widths

$$
\begin{array} { r l } { w _ { Z } = T _ { \mathrm { m a x } } \varepsilon ^ { 2 } , \ } & { { } w _ { X } = 2 T _ { \mathrm { m a x } } \varepsilon , } \\ { w _ { X ^ { 2 } } = T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 } , \ } & { { } w _ { T } = T _ { \mathrm { m a x } } - 1 . } \end{array}\tag{23}
$$

The independent unit is a whole reference trajectory; steps within a trajectory are not treated as independent samples. As the reference trajectories are i.i.d., Hoefding’s inequality gives, for each $F$ with probability at least $1 - \delta / 4$

$$
| \overline { { F } } - \mathbb { E } _ { Q } F | \leq b _ { F } : = w _ { F } u _ { N } ,\tag{24}
$$

with $u _ { N } = \sqrt { \log ( 8 / \delta ) / ( 2 N ) }$ as in Corollary 1. By the union bound, all four hold with probability at least $1 - \delta ,$ , without independence among the statistics. We work on this event.

Step version. By (21) and (22),

$$
\hat { \bar { T } } \hat { D } _ { S } ^ { 2 } - \bar { T } D _ { S } ^ { 2 } = ( \overline { { { Z } } } - \mathbb { E } _ { Q } Z ) - \Big ( \frac { \overline { { { X } } } ^ { 2 } } { \hat { \bar { T } } } - \frac { \mu ^ { 2 } } { \bar { T } } \Big ) .\tag{25}
$$

The second term decomposes by the identity

$$
\begin{array} { l } { \displaystyle { \frac { \overline { { { X } } } ^ { 2 } } { \hat { \bar { T } } } - \frac { \mu ^ { 2 } } { \bar { T } } = \Big ( \frac { \overline { { { X } } } } { \hat { \bar { T } } } + \frac { \mu } { \bar { T } } \Big ) ( \overline { { { X } } } - \mu ) } } \\ { \displaystyle { - \frac { \overline { { { X } } } \mu } { \hat { \bar { T } } \bar { T } } ( \hat { \bar { T } } - \bar { T } ) } . } \end{array}\tag{26}
$$

Since $| \overline { { X } } | \leq \varepsilon \hat { T }$ and $| \mu | \leq \varepsilon { \bar { T } }$ , the two coeficients are at most 2ε and $\varepsilon ^ { 2 }$ , so (24) gives

$$
| \hat { \bar { T } } \hat { D } _ { S } ^ { 2 } - \bar { T } { D } _ { S } ^ { 2 } | \leq b _ { Z } + 2 \varepsilon { b } _ { X } + \varepsilon ^ { 2 } { b } _ { T } .\tag{27}
$$

Trajectory version. Similarly, $\hat { D } _ { \tau } ^ { 2 } \ - \ D _ { \tau } ^ { 2 } \ =$ $( \overline { { { X ^ { 2 } } } } - \mathbb { E } _ { Q } X ^ { 2 } ) - ( \overline { { { X } } } ^ { 2 } - \mu ^ { 2 } )$ , and $| \overline { { { X } } } ^ { 2 } - \mu ^ { 2 } | =$ $| \overline { { X } } + \mu | | \overline { { X } } - \mu | \leq 2 T _ { \operatorname* { m a x } } \varepsilon b _ { X }$ , so

$$
| \hat { D } _ { \tau } ^ { 2 } - D _ { \tau } ^ { 2 } | \leq b _ { X ^ { 2 } } + 2 T _ { \mathrm { m a x } } \varepsilon b _ { X } .\tag{28}
$$

Conclusion. Since $\hat { K } _ { S } ~ = ~ \textstyle { \frac { 1 } { 2 } } \hat { \bar { T } } \hat { D } _ { S } ^ { 2 }$ and $\hat { K } _ { \tau }$ ${ \scriptstyle \frac { 1 } { 2 } } \hat { D } _ { \tau } ^ { 2 }$ , the triangle inequality, Theorem 1, (27), and (28) give $| K _ { i j } - \hat { K } _ { S } | \le R _ { S } + B _ { S } u _ { N }$ and $| K _ { i j } - \hat { K } _ { \tau } | \le R _ { \tau } + B _ { \tau } u _ { N }$ . Substituting (23),

$$
\begin{array} { r l } & { B _ { S } = \frac { 1 } { 2 } w _ { Z } + \varepsilon w _ { X } + \frac { 1 } { 2 } \varepsilon ^ { 2 } w _ { T } = \frac { 6 T _ { \mathrm { m a x } } - 1 } { 2 } \varepsilon ^ { 2 } , } \\ & { B _ { \tau } = \frac { 1 } { 2 } w _ { X ^ { 2 } } + T _ { \mathrm { m a x } } \varepsilon w _ { X } = \frac { 5 } { 2 } T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 } . \quad \quad ( 2 } \end{array}\tag{9}
$$

Finally, $\begin{array} { r } { \mathbb { E } _ { i } [ T ] \le T _ { \mathrm { { m a x } } } , \mathbb { E } _ { i } [ T ^ { 2 } ] \le T _ { \mathrm { { m a x } } } ^ { 2 } , \frac { \varepsilon } { 6 } \le \frac { \varepsilon } { 4 } } \end{array}$ $\frac { 6 T _ { \mathrm { m a x } } - 1 } { 2 } \leq 3 T _ { \mathrm { m a x } }$ , and ${ \frac { 5 } { 2 } } \leq 3$ give

$$
\begin{array} { r } { R _ { S } + B _ { S } u _ { N } \le T _ { \mathrm { m a x } } \varepsilon ^ { 2 } \big ( \frac \varepsilon 4 + \frac \lambda 2 + 3 u _ { N } \big ) + \frac { \mu ^ { 2 } } { 2 \bar { T } } , } \\ { R _ { \tau } + B _ { \tau } u _ { N } \le T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 } \big ( \frac \varepsilon 4 + \frac \lambda 2 + 3 u _ { N } \big ) + \frac { \mu ^ { 2 } } { 2 } , } \end{array}
$$

which is Corollary 1. Swapping $( i , j )$ leaves $Z , X ^ { 2 }$ $T ,$ , and $| { \overline { { X } } } - \mu |$ unchanged, so the same bound holds for $K _ { \mathrm { s y m } }$ □

## A.4 Cross terms in the trajectory distance

The step distance squares the step log-ratios before summing them; the trajectory distance sums them first. The following identity shows exactly what this adds. Let $c = \mu / \bar { T } , \tilde { d } _ { t } = d _ { t } - c$ for $t < T$ , and $\textstyle S = \sum _ { t < T } { \tilde { d } } _ { t }$ . Then

$$
\begin{array} { r l } & { D _ { \tau } ^ { 2 } = \bar { T } D _ { S } ^ { 2 } + 2 \mathbb { E } _ { Q } \sum _ { s < t < T } \tilde { d } _ { s } \tilde { d } _ { t } } \\ & { \qquad + 2 c \mathrm { C o v } _ { Q } ( S , T ) + c ^ { 2 } \mathrm { V a r } _ { Q } ( T ) . } \end{array}\tag{30}
$$

The right-hand side consists of the step distance, the cross term between time steps, and two terms due to variable trajectory length. The cross term has up to $\binom { T } { 2 }$ summands per trajectory and accounts for the extra factor of T in Corollary 1.

Proof. From $\begin{array} { r } { X = \sum _ { t < T } d _ { t } = S + c T } \end{array}$ and $\mu =$ $c { \bar { T } } .$

$$
X - \mu = S + c ( T - { \bar { T } } ) .\tag{31}
$$

Since $D _ { \tau } ^ { 2 } = \operatorname { V a r } _ { Q } ( X ) = \mathbb { E } _ { Q } ( X - \mu ) ^ { 2 }$ , squaring (31) and averaging over Q gives

$$
\begin{array} { r l } & { D _ { \tau } ^ { 2 } = \mathbb { E } _ { Q } S ^ { 2 } + 2 c \mathbb { E } _ { Q } [ S ( T - \bar { T } ) ] } \\ & { \qquad + c ^ { 2 } \mathbb { E } _ { Q } ( T - \bar { T } ) ^ { 2 } . } \end{array}\tag{32}
$$

Here $\mathbb { E } _ { Q } S = \mathbb { E } _ { Q } X - c \mathbb { E } _ { Q } [ T ] = \mu - \mu = 0 ,$ so $\mathbb { E } _ { Q } [ S ( T - \bar { T } ) ] = \mathrm { C o v } _ { Q } ( S , T )$ and $\begin{array} { r } { \mathbb { E } _ { Q } ( T - \bar { T } ) ^ { 2 } = } \end{array}$ $\mathrm { V a r } _ { Q } ( T )$ . The first term expands as

$$
\begin{array} { r } { \mathbb { E } _ { Q } S ^ { 2 } = \mathbb { E } _ { Q } \sum _ { t < T } \tilde { d } _ { t } ^ { 2 } + 2 \mathbb { E } _ { Q } \sum _ { s < t < T } \tilde { d } _ { s } \tilde { d } _ { t } , } \end{array}\tag{33}
$$

and by the definition of $D _ { S } ^ { 2 }$ the first term on the right equals $\hat { T } D _ { S } ^ { 2 }$ . Substituting these into (32) gives (30). □

For example, suppose each reference trajectory has a constant step log-ratio $d _ { t } = b$ , where b varies across trajectories with $\mathbb { E } _ { Q } b = 0$ and is independent of $T$ . Then $\mu = c = \dot { 0 } , D _ { S } ^ { 2 } = \mathbb { E } _ { Q } [ T b ^ { 2 } ] / \bar { T } =$ ${ \mathrm { V a r } } _ { Q } ( b )$ , and $\begin{array} { r c l } { D _ { \tau } ^ { 2 } } & { = } & { \mathbb { E } _ { Q } [ \tilde { T ^ { 2 } } ] \operatorname { V a r } _ { Q } ( b ) } \end{array}$ : with strongly correlated steps, the trajectory estimate $\scriptstyle \frac { 1 } { 2 } D _ { \tau } ^ { 2 } \ \mathrm { i s } \ \mathbb { E } _ { Q } [ T ^ { 2 } ] / { \bar { T } }$ times the step estimate $\textstyle { \frac { \bar { T } } { 2 } } D _ { S } ^ { 2 }$ Since the cross term can also be negative, the trajectory version does not always overestimate.

## A.5 Uncentered variants

The uncentered embeddings are $\begin{array} { r l } { z _ { i , u } ^ { S } } & { { } = } \end{array}$ $M ^ { - 1 / 2 } ( \ell _ { i , r t } ) _ { r , t }$ and $\begin{array} { r l r } { z _ { i , u } ^ { \tau } } & { { } = } & { N ^ { - 1 / 2 } ( A _ { i , r } ) _ { r } } \end{array}$ with distances $\hat { D } _ { S , u }$ and $\hat { D } _ { \tau , u }$ . In the notation of Appendix A.3,

$$
\hat { \bar { T } } \hat { D } _ { S , u } ^ { 2 } = \overline { { { Z } } } , \qquad \hat { D } _ { \tau , u } ^ { 2 } = \overline { { { X ^ { 2 } } } } ,\tag{34}
$$

and the population distances are $\bar { T } D _ { S , u } ^ { 2 } = \mathbb { E } _ { Q } Z$ and $D _ { \tau , u } ^ { 2 } = \mathbb { E } _ { Q } X ^ { 2 }$ . The estimators are $\hat { K } _ { S , u } =$ $\begin{array} { r } { \frac { 1 } { 2 } \hat { \bar { T } } \hat { D } _ { S , u } ^ { 2 } = \frac { 1 } { 2 } \overline { { Z } } } \end{array}$ and $\begin{array} { r } { \hat { K } _ { \tau , u } = \frac { 1 } { 2 } \hat { D } _ { \tau , u } ^ { 2 } = \frac { 1 } { 2 } \overline { { X ^ { 2 } } } } \end{array}$ . They difer from the centered versions only by the $\mu ^ { 2 }$ terms in (22).

Proposition 1 (Error of the uncentered variants). Under the assumptions of Theorem $I , \ | K _ { i j }$ $\begin{array} { r } { \frac { \bar { T } } { 2 } D _ { S , u } ^ { 2 } | \leq R _ { S , u } } \end{array}$ and $| K _ { i j } - { \textstyle \frac { 1 } { 2 } } D _ { \tau , u } ^ { 2 } | \le R _ { \tau , u }$ , where

$$
\begin{array} { r } { R _ { S , u } = \frac { \varepsilon ^ { 3 } } { 6 } \mathbb { E } _ { i } [ T ] + \frac { \lambda T _ { \mathrm { m a x } } \varepsilon ^ { 2 } } { 2 } , } \end{array}\tag{35}
$$

$$
\begin{array} { r l } & { R _ { \tau , u } = \frac { \varepsilon ^ { 3 } } { 6 } \mathbb { E } _ { i } [ T ] + \frac { \varepsilon ^ { 3 } } { 4 } \mathbb { E } _ { i } [ T ( T - 1 ) ] } \\ & { \quad \quad \quad + \frac { \lambda T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 } } { 2 } . } \end{array}\tag{36}
$$

Furthermore, under the assumptions of Corollary 1, with $v _ { N } = \sqrt { \log ( 4 / \delta ) / ( 2 N ) }$ , with probability at least 1 − δ simultaneously,

$$
\begin{array} { r } { | K _ { i j } - \hat { K } _ { S , u } | \leq R _ { S , u } + \frac 1 2 T _ { \operatorname* { m a x } } \varepsilon ^ { 2 } v _ { N } , } \\ { | K _ { i j } - \hat { K } _ { \tau , u } | \leq R _ { \tau , u } + \frac 1 2 T _ { \operatorname* { m a x } } ^ { 2 } \varepsilon ^ { 2 } v _ { N } . } \end{array}\tag{37}
$$

Proof. The approximation error follows $\mathsf { A p - }$ pendix A.2. For the step version, (16) gives

$$
\begin{array} { r l } & { K _ { i j } - \frac { \bar { T } } { 2 } D _ { S , u } ^ { 2 } = K _ { i j } - \frac { 1 } { 2 } \mathbb { E } _ { Q } Z } \\ & { \qquad = r _ { * } + \frac { 1 } { 2 } ( \mathbb { E } _ { i } Z - \mathbb { E } _ { Q } Z ) } \end{array}\tag{38}
$$

with $\begin{array} { r } { | r _ { * } | \le \frac { \varepsilon ^ { 3 } } { 6 } \mathbb { E } _ { i } [ T ] } \end{array}$ , and Lemma 2 bounds the second term by $\begin{array} { r } { \frac { 1 } { 2 } \lambda \dot { T } _ { \mathrm { m a x } } \varepsilon ^ { 2 } } \end{array}$ , giving (35). For the trajectory version, $\begin{array} { r } { \bar { \mathbb { \frac { 1 } { 2 } } } \mathbb { E } _ { i } Z = \frac { 1 } { 2 } \bar { \mathbb { E } } _ { i } X ^ { 2 } - \sum _ { s < t } \mathbb { E } _ { i } [ d _ { s } d _ { t } ] } \end{array}$ gives

$$
\begin{array} { r l } & { K _ { i j } - \frac { 1 } { 2 } D _ { \tau , u } ^ { 2 } = r _ { * } - \sum _ { s < t } \mathbb { E } _ { i } [ d _ { s } d _ { t } ] } \\ & { \qquad + \frac { 1 } { 2 } ( \mathbb { E } _ { i } X ^ { 2 } - \mathbb { E } _ { Q } X ^ { 2 } ) . } \end{array}\tag{39}
$$

The cross term is at most $\frac { \varepsilon ^ { 3 } } { 4 } \mathbb { E } _ { i } [ T ( T - 1 ) ]$ by Lemma 1(iii), and the third term at most $\scriptstyle \frac { 1 } { 2 } \lambda T _ { \mathrm { m a x } } ^ { 2 } \varepsilon ^ { 2 }$ by Lemma 2, giving (36). Both bounds are the centered ones without the centering terms $\mu ^ { 2 } / ( 2 \bar { T } )$ and $\mu ^ { 2 } / 2$

For the sampling error, $\hat { \bar { T } }$ cancels in (34), so each version depends on a single sample mean, Z or $\overline { { X ^ { 2 } } }$ . By Hoefding’s inequality, $| \overline { { Z } } - \mathbb { E } _ { Q } Z | \le$ w<sub>Z</sub>v<sub>N</sub> and $| \overline { { { X ^ { 2 } } } } - \mathbb { E } _ { Q } X ^ { 2 } | \le w _ { X ^ { 2 } } v _ { N }$ each hold with probability at least $1 - \delta / 2$ , and both together with probability at least $1 - \delta$ . The triangle inequality then gives (37). Since swapping $( i , j )$ leaves Z and $X ^ { 2 }$ unchanged, the bounds also hold for the symmetric KL. □

## B Mapping and predicting: details and robustness

This appendix supplements §3. Appendix B.1 describes the setup of the two experiments, and $\mathsf { A p - }$ pendices B.2, B.3, and B.4 give details of the trajectory-KL comparison (§3.1), performance prediction (§3.2), and behavior (§3.3).

## B.1 Setup of the two experiments

Table 4 summarizes the two experiments of §3; the multi-harness experiment of §4 is summarized in Table 11.

Models. We use 100 open-weight LLMs, all with the BASE harness on ALFWorld; the full list with parameter counts $| \theta _ { i } |$ is in Table 5; parameter counts are total counts, including for MoE models, and the size baselines use their logarithm. Model families, used only as a baseline in §3.2–3.3, are assigned from the model name: 19 families, the largest being Qwen with 20 models.

Efect of the grammar mask. The 100-model experiment scores without the grammar mask of Appendix A.1. To check that this choice does not change the map, we use the scoring runs of ten LLMs (Qwen3-8B, Qwen3.8-27B, Qwen3.5- 2B/9B, Gemma4-E4B/12B, GPT-OSS-20B, GLM-4.7-Flash, Devstral-Small-2-24B, Granite-3.3-2B) that recorded both the masked and the raw loglikelihood of each reference action. Six of them (Qwen3-8B, Qwen3.8-27B, Qwen3.5-9B, Gemma4-E4B, GPT-OSS-20B, GLM-4.7-Flash) share 3,412 reference steps with both scores. On these steps, the mask changes the mean loglikelihood per action by −0.01 to +0.03 nats, the per-step values correlate at $r \geq 0 . 9 9$ , and the pairwise squared distances between the six models correlate at 1.000 with the same ordering of all pairs, with and without centering. For each of the ten LLMs separately, we then replace its raw scores in the 100-model map by the masked scores on the steps where both exist (591–10,304 of the 10,388 steps, depending on the LLM). This moves its centered coordinate by 0.05–0.21 times the distance to its nearest neighbor (median 0.07 over the ten LLMs); the nearest neighbor is unchanged for all ten, the five nearest neighbors for nine (four of five for Granite-3.3-2B), and the rank correlation of the distances to the other 99 models is at least 0.997.

Table 4: The two experiments of §3. Both fix the BASE harness. The multi-harness experiment of §4 is in Table 11.
<table><tr><td></td><td>Trajectory KL (§3.1)</td><td>100 LLMs (§3.2, 3.3)</td></tr><tr><td>LLMs</td><td>8</td><td>100</td></tr><tr><td>Reference</td><td>Each model&#x27;s own rollouts on the 988 of 1,000 ALFWorld train tasks completed by all 8 models</td><td>Rollouts of Qwen3.8-27B and Gemma4-E4B on 250 ALFWorld train tasks (500 trajectories, 10,388</td></tr><tr><td>trajectories</td><td>(7,904 trajectories, 187,961 steps) Equal-weight pool of the 8 models</td><td>steps) Equal-weight mixture of the 2 generators</td></tr><tr><td>Reference distribution Q Scoring</td><td>Grammar-constrained, canonical tokens; every</td><td>No grammar mask; each model&#x27;s chat template</td></tr><tr><td></td><td>model scores every trajectory (8 × 8)</td><td>Success rates and action sequences on ALFWorld</td></tr><tr><td>What is compared</td><td>Directly measured  $K _ { \mathrm { s y m } }$  (28 pairs)</td><td>valid_unseen, TextWorld, BFCL, MiniWoB++ (50 tasks each; behavior on BFCL uses multi_turn 25);</td></tr></table>

Definition of the trajectory KL. The eight models are Qwen3-8B, Gemma4-E4B, GPT-OSS-20B, Qwen3.8-27B, GPT-OSS-120B, GLM-4.7-Flash, Qwen3.5-9B, and Gemma4-12B. $K _ { i j }$ is the sample mean of log $P _ { i } / P _ { j }$ over the trajectories generated by i, and $K _ { \mathrm { s y m } }$ averages the two directions. Negative log-ratios would be kept, but none occur in this set. Since the reference is the equal-weight pool of the eight models, $d P _ { i } / d Q \leq 8$ by construction.

Compute and reproducibility. All likelihood computations and rollouts were run on ten NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB each) served through vLLM. The complete set of experiments, including the scoring of 10,388 reference steps by 100 LLMs, the downstream rollouts of 100 LLMs on four benchmarks, and the $1 2 \times 1 2$ harness experiment, took approximately two weeks of wall-clock time on this cluster; we did not record per-experiment GPU hours. Each downstream task was run once per agent (Table 4), and the hyperparameters of the predictors are listed in Appendix B.3 and those of the harnesses in Appendix C.1. Figure 1(a) is a t-SNE layout (van der Maaten and Hinton, 2008) of the centered coordinates with perplexity 15; all neighbors in the paper are computed in the original coordinate space. The τ -bench and SWE-Bench reference trajectories are the publicly released HAL traces (Kapoor et al., 2026).

## B.2 Trajectory KL: centering, step vs. trajectory, and sample size

Using the eight-model KL data of §3.1, we examine the representation choices fixed in §2.3 and the dependence on the number of reference trajectories.

Efect of centering. Table 6 compares agreement with the KL with and without centering. On these eight models, centering gives a slightly higher correlation and a smaller RMSE. In performance prediction (Table 7), uncentered Ridge is about as good as or slightly better than centered Ridge, while for behavioral neighbors (Table 9) centering tends to give larger efects. Since neither choice wins on every measure, the main text reports centered coordinates throughout and the appendix also reports uncentered results.

Step version vs. trajectory version. The step version agrees better with the KL in both correlation and RMSE (Table 6). Both overestimate the measured $K _ { \mathrm { s y m } } .$ , by a median factor of 7.5 for the step version and 74 for the trajectory version; the ratio of the two factors, about 10, is of the order of the mean reference trajectory length of 23.8 steps (Table 4).

Distances on the 100-model map. The eight models are among the 100 LLMs of §3.2, so their 28 pairwise distances can also be read from the 100-model map, whose reference trajectories come from two of the eight models and are scored without the grammar mask. Against the same $K _ { \mathrm { s y m } } ,$ the centered step estimate $\textstyle { \frac { \bar { T } } { 2 } } \hat { D } _ { S } ^ { 2 }$ from that map gives $r = 0 . 8 6$ and $\rho = 0 . 8 8$ (uncentered 0.80 / 0.92;

Table 5: The 100 open-weight LLMs of §3.2–3.3, identified by their Hugging Face repository, with parameter counts in billions (total parameters for MoE models) and the family labels used as a baseline. ⋄: the eight models of the trajectory-KL comparison (§3.1); ⋆: the twelve LLMs of the harness experiment (§4).
<table><tr><td>Model</td><td>B</td><td>Family</td><td>Model</td><td>B</td><td>Family</td></tr><tr><td>CohereForAI/c4ai-command-r7b-12-2024</td><td>8.0</td><td>Cohere</td><td>allenai/Llama-3.1-Tulu-3-8B-SFT</td><td>8.0</td><td>Llama</td></tr><tr><td>deepseek-ai/deepseek-coder-1.3b-instruct</td><td>1.3</td><td>DeepSeek</td><td>codellama/CodeLlama-13b-Instruct-hf</td><td>13.0</td><td>Llama</td></tr><tr><td>deepseek-ai/deepseek-coder-6.7b-instruct</td><td>6.7</td><td>DeepSeek</td><td>codellama/CodeLlama-34b-Instruct-hf</td><td>33.7</td><td>Llama</td></tr><tr><td>deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct</td><td>15.7</td><td>DeepSeek</td><td>mistralai/Ministral-3-3B-Instruct-2512</td><td>3.8</td><td>Mistral</td></tr><tr><td>deepseek-ai/DeepSeek-V2-Lite-Chat</td><td>15.7</td><td>DeepSeek</td><td>mistralai/Mistral-7B-Instruct-v0.2</td><td>7.2</td><td>Mistral</td></tr><tr><td>deepseek-ai/deepseek-coder-33b-instruct</td><td>33.3</td><td>DeepSeek</td><td>mistralai/Mistral-7B-Instruct-v0.3</td><td>7.2</td><td>Mistral</td></tr><tr><td>LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct</td><td>2.4</td><td>EXÃONE</td><td>mistralai/Ministral-3-8B-Instruct-2512</td><td>8.9</td><td>Mistral</td></tr><tr><td>LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct</td><td>7.8</td><td>EXAONE</td><td>mistralai/Mistral-Nemo-Instruct-2407</td><td>12.2</td><td>Mistral</td></tr><tr><td>LGAI-EXAONE/EXAONE-3.5-32B-Instruct</td><td>32.0</td><td>EXAONE</td><td>mistralai/Ministral-3-14B-Instruct-2512</td><td>13.9</td><td>Mistral</td></tr><tr><td>tiiuae/Falcon-H1-1.5B-Instruct</td><td>1.6</td><td>Falcon</td><td>mistralai/Mistral-Small-24B-Instruct-2501</td><td>23.6</td><td>Mistral</td></tr><tr><td>tiiuae/Falcon3-3B-Instruct</td><td></td><td>3.2 Falcon</td><td>mistralai/Devstral-Small-2505</td><td>23.6</td><td>Mistral</td></tr><tr><td>tiiuae/Falcon3-7B-Instruct</td><td>7.5</td><td>Falcon</td><td>mistralai/Mistral-Small-3.1-24B-Instruct-2503</td><td>24.0</td><td>Mistral</td></tr><tr><td>tiiuae/Falcon-H1-7B-Instruct</td><td>7.6</td><td>Falcon</td><td>mistralai/Devstral-Small-2-24B-Instruct-2512*</td><td>24.0</td><td>Mistral</td></tr><tr><td>tiiuae/Falcon3-10B-Instruct</td><td>10.3</td><td>Falcon</td><td>nvidia/NVIDIA-Nemotron-Nano-9B-v2</td><td>8.9</td><td>Nemotron</td></tr><tr><td>google/gemma-3-1b-it</td><td>1.0</td><td>Gemma</td><td>nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16</td><td>31.6</td><td>Nemotron</td></tr><tr><td>google/gemma-3-4b-it</td><td>4.3</td><td>Gemma</td><td>allenai/Olmo-3-7B-Instruct</td><td>7.3</td><td>OLMo</td></tr><tr><td>google/gemma-4-E2B-it</td><td>5.1</td><td>Gemma</td><td>allenai/Olmo-3-7B-Instruct-SFT</td><td>7.3</td><td>OLMo</td></tr><tr><td>google/gemma-3n-E4B-it</td><td>7.8</td><td>Gemma</td><td>allenai/Olmo-3-7B-Instruct-DPO</td><td>7.3</td><td>OLMo</td></tr><tr><td>google/gemma-4-E4B-it*</td><td>8.0</td><td>Gemma</td><td>allenai/Olmo-3.1-32B-Instruct</td><td>32.2</td><td>OLMo</td></tr><tr><td>google/gemma-4-12b-it*</td><td>12.0</td><td>Gemma</td><td>microsoft/Phi-3-mini-128k-instruct</td><td>3.8</td><td>Phi</td></tr><tr><td>google/gemma-3-12b-it</td><td>12.2</td><td>Gemma</td><td>microsoft/Phi-3.5-mini-instruct</td><td>3.8</td><td>Phi</td></tr><tr><td>google/gemma-4-26B-A4B-it</td><td>25.8</td><td>Gemma</td><td>microsoft/Phi-4-mini-instruct</td><td>3.8 Phi</td><td></td></tr><tr><td>google/gemma-3-27b-it</td><td>27.4</td><td>Gemma</td><td>microsoft/Phi-4-mini-reasoning</td><td>3.8 Phi</td><td></td></tr><tr><td>google/gemma-4-31b-it*</td><td>31.3</td><td>Gemma</td><td>microsoft/Phi-3-small-128k-instruct</td><td>7.4 Phi</td><td></td></tr><tr><td>THUDM/glm-4-9b-chat</td><td>9.4</td><td>GLM</td><td>microsoft/Phi-3-medium-128k-instruct</td><td>14.0 Phi</td><td></td></tr><tr><td>zai-org/GLM-4.7-Flash*</td><td>31.2</td><td>GLM</td><td>microsoft/phi-4</td><td>14.7</td><td>Phi</td></tr><tr><td>openai/gpt-oss-20b*</td><td>20.9</td><td>gpt-oss</td><td>Qwen/Qwen2.5-0.5B-Instruct</td><td>0.5</td><td>Qwen</td></tr><tr><td>openai/gpt-oss-120b*</td><td>116.8</td><td>gpt-oss</td><td>Qwen/Qwen3-0.6B</td><td>0.8</td><td>Qwen</td></tr><tr><td>ibm-granite/granite-3.1-1b-a400m-instruct</td><td>1.3</td><td>Granite</td><td>Qwen/Qwen2.5-1.5B-Instruct</td><td>1.5</td><td>Qwen</td></tr><tr><td>ibm-granite/granite-3.1-2b-instruct</td><td>2.5</td><td>Granite</td><td>Qwen/Qwen3-1.7B</td><td>2.0</td><td>Qwen</td></tr><tr><td>ibm-granite/granite-3.3-2b-instruct</td><td>2.5</td><td>Granite</td><td>Qwen/Qwen3.5-2B*</td><td>2.3</td><td>Qwen</td></tr><tr><td>ibm-granite/granite-3.1-3b-a800m-instruct</td><td>3.3</td><td>Granite</td><td>Qwen/Qwen2.5-3B-Instruct</td><td>3.1</td><td>Qwen</td></tr><tr><td>ibm-granite/granite-4.0-h-tiny</td><td>6.9</td><td>Granite</td><td>Qwen/Qwen3-4B</td><td>4.0</td><td>Qwen</td></tr><tr><td>ibm-granite/granite-3.1-8b-instruct</td><td>8.2</td><td>Granite</td><td>Qwen/Qwen3.5-4B</td><td>4.7</td><td>Qwen</td></tr><tr><td>ibm-granite/granite-3.3-8b-instruct</td><td>8.2</td><td>Granite</td><td>Qwen/Qwen2.5-7B-Instruct</td><td>7.6</td><td>Qwen</td></tr><tr><td>tencent/Hunyuan-0.5B-Instruct</td><td>0.5</td><td>Hunyuan</td><td>Qwen/Qwen3-8B*</td><td>8.2</td><td>Qwen</td></tr><tr><td>tencent/Hunyuan-1.8B-Instruct</td><td>1.8</td><td>Hunyuan</td><td>Qwen/Qwen3.5-9B*</td><td>9.7</td><td>Qwen</td></tr><tr><td>tencent/Hunyuan-7B-Instruct</td><td>7.5</td><td>Hunyuan</td><td>Qwen/Qwen3-14B</td><td>14.8</td><td>Qwen</td></tr><tr><td>internlm/internlm2_5-1_8b-chat</td><td></td><td>1.9 InternLM</td><td>Qwen/Qwen2.5-14B-Instruct</td><td>14.8</td><td>Qwen</td></tr><tr><td>internlm/internlm2_5-7b-chat</td><td>7.7</td><td>InternLM</td><td>Qwen/Qwen2.5-Coder-14B-Instruct</td><td>14.8</td><td>Qwen</td></tr><tr><td>internlm/internlm3-8b-instruct</td><td>8.8</td><td>InternLM</td><td>Qwen/Qwen3.8-27B*</td><td>27.8</td><td>Qwen</td></tr><tr><td>LiquidAI/LFM2.5-8B-A1B</td><td>8.5</td><td>LFM</td><td>Qwen/Qwen3.5-27B*</td><td>27.8</td><td>Qwen</td></tr><tr><td>meta-llama/Llama-3.2-1B-Instruct</td><td></td><td>1.2 Llama</td><td>Qwen/Qwen3-30B-A3B</td><td>30.5</td><td>Qwen</td></tr><tr><td>meta-llama/Llama-3.2-3B-Instruct</td><td>3.2</td><td>Llama</td><td>Qwen/Qwen3-32B</td><td>32.8</td><td>Qwen</td></tr><tr><td>codellama/CodeLlama-7b-Instruct-hf</td><td></td><td>6.7 Llama</td><td>Qwen/Qwen2.5-32B-Instruct</td><td>32.8</td><td>Qwen</td></tr><tr><td>meta-llama/Llama-3.1-8B-Instruct</td><td></td><td>8.0 Llama</td><td>Qwen/Qwen3.6-35B-A3B</td><td>36.0</td><td>Qwen</td></tr><tr><td>NousResearch/Hermes-3-Llama-3.1-8B</td><td></td><td>8.0 Llama</td><td>HuggingFaceTB/SmolLM3-3B</td><td>3.1</td><td>SmolLM</td></tr><tr><td>nvidia/Llama-3.1-Nemotron-Nano-8B-v1</td><td></td><td>8.0 Llama</td><td>01-ai/Yi-Coder-1.5B-Chat</td><td>1.5</td><td>Yi</td></tr><tr><td>allenai/Llama-3.1-Tulu-3-8B</td><td></td><td>8.0 Llama</td><td>01-ai/Yi-1.5-9B-Chat-16K</td><td>8.8 Yi</td><td></td></tr><tr><td>allenai/Llama-3.1-Tulu-3-8B-DPO</td><td></td><td>8.0 Llama</td><td>01-ai/Yi-1.5-34B-Chat-16K</td><td>34.4 Yi</td><td></td></tr></table>

Table 6: Agreement between the estimates read from the map and the directly measured symmetric trajectory KL $K _ { \mathrm { s y m } } \colon$ 8 models, 28 pairs, 988 shared ALFWorld train tasks. The estimates are $\textstyle { \frac { \bar { T } } { 2 } } { \hat { D } } _ { S } ^ { 2 }$ for the step version and ${ \scriptstyle \frac { 1 } { 2 } } \hat { D } _ { \tau } ^ { 2 }$ for the trajectory version. RMSE is in nats.
<table><tr><td></td><td>r</td><td>ρ</td><td>RMSE</td></tr><tr><td colspan="4">Step version</td></tr><tr><td>Centered</td><td>0.90</td><td>0.95</td><td>2,056</td></tr><tr><td>Uncentered</td><td>0.83</td><td>0.92</td><td>3,773</td></tr><tr><td colspan="4">Trajectory version</td></tr><tr><td>Centered</td><td>0.86</td><td>0.94</td><td>29,792</td></tr><tr><td>Uncentered</td><td>0.77</td><td>0.92</td><td>72,014</td></tr></table>

centered trajectory version 0.75 / 0.89), and the 28 estimates from the two embeddings have rank correlation 0.97. The lower correlation reflects both the diferent reference distribution Q and the scoring convention.

Dependence on the number of reference trajectories. We resample N reference trajectories from the pool of 7,904 without replacement, 50 times for each N from 100 to 3,200, and recompute the estimates. The resampling standard deviation of the step estimate shrinks roughly as $N ^ { - 1 / 2 }$ (median coeficient of variation over the 28 pairs 0.074 at N = 100 and 0.010 at $N = 3 { , } 2 0 0 \mathrm { { ; } }$ ; log–log slope −0.48 after the finite-population correction, −0.52 for the trajectory version), consistent with

Corollary 1. The median ratio of the estimate to the measured $K _ { \mathrm { s y m } } .$ , by contrast, stays between 7.5 and 7.6 for the step version and between 72 and 74 for the trajectory version at every N: most of the gap to the measured KL is approximation error and does not shrink with N.

## B.3 Performance prediction: protocol and full results

This section gives the details of §3.2 and the results of all predictors. Models, reference trajectories, and scoring are described in Appendix B.1.

Tasks and average success rates. A success rate is the fraction of tasks solved; run conditions are listed in Table 4. No values are missing across the 4 benchmarks, 50 tasks each, and 100 models. The tasks are 50 from ALFWorld valid\_unseen, 25 each from TextWorld treasure hunter easy and medium, 25 each from BFCL simple\_python and multi\_turn, and 25 each from MiniWoB++ navigate-tree and form-sequence. The four-benchmark average is the success rate over all 200 tasks, and the average over the three other environments is the success rate over the 150 tasks of TextWorld, BFCL, and Mini-WoB++.

Predictors and intervals. We hold out one model at a time and predict it from the remaining 99. kNN uses Euclidean distance in centered coordinates, and the number of neighbors k is chosen from {1, 3, 5, 10} by inner leave-one-out on the 99 training models. Ridge uses a linear kernel, with the penalty α chosen from five values, $\{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , \hat { 1 } 0 ^ { - 1 } , \hat { 1 } , 1 0 \}$ times the trace scale of the centered kernel. Size plus coordinates is a Ridge regression with $\log _ { 1 0 } | \theta _ { i } |$ as an unpenalized covariate. All hyperparameters are chosen by inner leaveone-out on the 99 training models, and predictions are clipped to [0, 1]. For diferences from the size baseline, 95% intervals are the 2.5th and 97.5th percentiles of 2,000 bootstrap resamples of models. We compute them for both the diference in Pearson $^ { r , }$ shown in bold in the main-text table, and the difference in RMSE, shown by arrows in the appendix table.

Results of all predictors. Table 7 and Figure 4 cover the training mean, size, family, <sup>¯</sup>ℓ, their combinations, and predictors on centered and uncentered coordinates. Two trends stand out. Adding <sup>¯</sup>ℓ to size barely changes the predictions. Predictions from uncentered coordinates are about as good as those from centered coordinates. Using Spearman rank correlation instead of Pearson correlation does not change the ordering of the methods.

## B.4 Behavior: data, metrics, and additional results

This section defines the action sequences, distances, neighbors, and tests of §3.3, and reports measures, controls, and robustness checks not shown in the main-text table.

## B.4.1 Definitions

Action sequences. We use one trajectory per model and task; run conditions are listed in Table 4. For the 50 ALFWorld valid\_unseen tasks, the trajectories come from the same runs as the performance labels. An action sequence is the sequence of commands for the 50 TextWorld treasure-hunter tasks, the sequence of function names extracted from model outputs for the 25 BFCL multi\_turn tasks, and the sequence of action strings for the 50 MiniWoB++ tasks. The action type is the leading verb, function name, or tool name. An action is judged invalid by the environment’s set of admissible actions in ALFWorld, by the environment’s feedback that it could not parse the command in TextWorld, by a turn without a function call in BFCL, and by a tool error in MiniWoB++.

Behavioral distances. The four aspects in the main text are the base-2 Jensen–Shannon distance between action-type frequency distributions (Endres and Schindelin, 2003), the Levenshtein distance divided by the longer length (Levenshtein, 1966; Rabanser et al., 2026), the Jaccard distance between sets of action 2-grams, and the disagreement rate of the first action from the same initial state (Bouchard and Chauhan, 2026; Gonuguntla, 2026). The first three are computed per task and averaged over the tasks, at least 10, recorded for both models. The appendix tables add four more measures. The divergence position is 1 − $k / \operatorname* { m i n } ( T _ { i } , T _ { j } )$ , where k is the number of steps before the first difering action. The conditional disagreement counts disagreement at step t only for pairs whose action sequences agree up to step t − 1, summed over $t = 2 , \ldots , 5$ ; only pairs with at least 10 such cases are used. For the first action, conditional disagreement, and divergence position, two invalid actions are not counted as agreeing. The profile distance is the Euclidean distance between per-model vectors standardized within each benchmark; each vector contains task averages of the repetition rate $\frac { T - | \mathrm { d i s t i n c t a c t i o n s } | } { T - 1 }$ (Gioacchini et al., 2024), the consecutive repetition rate, exploration breadth measured as distinct actions/T, the invalidaction rate, which is the complement of grounding accuracy (Ma et al., 2024), the share of informationgathering actions, the format-error rate, steps/limit, the rate of hitting the step limit, and the earlytermination rate. For BFCL we also include the rate of empty turns. The success pattern is the fraction of tasks on which the two models’ outcomes difer, $\begin{array} { r } { \frac { 1 } { | \mathcal { T } | } \sum _ { t } | s _ { i t } - s _ { j t } | } \end{array}$

![](images/f71e06680fa36974f2724ef8c47d9b729a65301506a890a19b522578ea5f828b.jpg)  
Figure 4: Predicting success rates without running the agents, with leave-one-model-out over 100 models. The horizontal axis is the prediction and the vertical axis the measured success rate. Open gray circles show the size baseline, a regression on log $| \theta _ { i } | ;$ blue points show coordinates alone, a Ridge regression on centered coordinates (the Ridge column of Table 2). Panels from left: ALFWorld, TextWorld, BFCL, MiniWoB++, and the four-benchmark average.

Table 7: All predictors for performance prediction, with leave-one-model-out over 100 models. Each cell gives RMSE in points / Pearson r. ↓ and ↑ mark a 95% bootstrap interval of the RMSE diference from the size baseline that lies below or above zero. The training mean is a constant prediction, so no r is shown.
<table><tr><td>Predictor</td><td>ALFWorld</td><td>TextWorld</td><td>BFCL</td><td>MiniWoB++</td><td>4-bench avg.</td><td>3 other envs.</td></tr><tr><td colspan="7">Information available without running</td></tr><tr><td>Training mean</td><td> $1 6 . 4 / - \stackrel { \uparrow } { }$ </td><td> $2 9 . 0 / - ^ { \uparrow }$ </td><td> $1 6 . 1 / - \stackrel { \uparrow } { }$ </td><td> $2 7 . 5 / - ^ { \uparrow }$ </td><td> $1 9 . 2 / - \stackrel { \uparrow } { }$ </td><td> $2 1 . 4 / - ^ { \uparrow }$ </td></tr><tr><td>Size log |θi|</td><td>12.6 / 0.63</td><td> $2 0 . 2 / 0 . 7 1$ </td><td>13.9 / 0.48</td><td>20.9 / 0.64</td><td> $1 2 . 9 / 0 . 7 4$ </td><td> $1 4 . 6 / 0 . 7 2$ </td></tr><tr><td>Family mean</td><td>14.7 / 0.46</td><td> $2 6 . 8 / 0 . 4 4 ^ { \uparrow }$ </td><td>15.5 / 0.34</td><td>27.6 / 0.24↑</td><td> $1 7 . 7 / 0 . 4 2 ^ { \uparrow }$ </td><td> $1 9 . 9 / 0 . 4 0 ^ { \uparrow }$ </td></tr><tr><td>Mean log-likelihood ē</td><td>16.0 / 0.20↑</td><td> $2 7 . 2 / 0 . 3 3 ^ { \uparrow }$ </td><td>15.1 / 0.32</td><td>26.6 / 0.22↑</td><td>18.0 / 0.34↑</td><td>19.9 / 0.34↑</td></tr><tr><td>Size + ē</td><td>12.4 / 0.65</td><td>19.2 / 0.74</td><td>13.4 / 0.54</td><td>20.7 / 0.65</td><td>12.2 / 0.77</td><td>13.8 / 0.76</td></tr><tr><td colspan="7">Centered coordinates</td></tr><tr><td>kNN</td><td> $9 . 0 / 0 . 8 8 ^ { \downarrow }$ </td><td>20.3 / 0.76</td><td>11.8 /0.69↓</td><td>28.2 / 0.28↑</td><td>13.5 / 0.74</td><td>15.3 / 0.72</td></tr><tr><td>Ridge</td><td> $9 . 6 / 0 . 8 1 ^ { \downarrow }$ </td><td>18.3 / 0.77</td><td>11.7 / 0.69↓</td><td>22.9 / 0.56</td><td>12.1 / 0.78</td><td>14.0 / 0.75</td></tr><tr><td>Size + Ridge</td><td> $9 . 4 / 0 . 8 2 ^ { \downarrow }$ </td><td>18.0/0.78↓</td><td>12.1 / 0.65↓</td><td>21.1 / 0.63</td><td>10.9/ 0.83↓</td><td>12.5 / 0.81↓</td></tr><tr><td>Size + ¯ + Ridge</td><td> $9 . 2 / 0 . 8 3 ^ { \downarrow }$ </td><td>17.1/0.80↓</td><td>11.0 / 0.72↓</td><td>20.3 / 0.67</td><td>10.3 / 0.84↓</td><td>12.3 /0.81↓</td></tr><tr><td colspan="7">Uncentered coordinates</td></tr><tr><td>kNN</td><td> $9 . 2 / 0 . 8 3 ^ { \downarrow }$ </td><td>19.5 / 0.73</td><td>12.5 / 0.62</td><td>23.2 / 0.52</td><td>12.8 / 0.77</td><td>14.8 / 0.72</td></tr><tr><td>Ridge</td><td> $9 . 0 / 0 . 8 3 ^ { \downarrow }$ </td><td>19.4 / 0.74</td><td>11.0/0.72↓</td><td>22.8 / 0.55</td><td>12.1 / 0.78</td><td>14.2 / 0.75</td></tr><tr><td>Size + Ridge</td><td> $8 . 9 / 0 . 8 4 ^ { \downarrow }$ </td><td>16.5 / 0.82↓</td><td>10.9/0.73↓</td><td>20.2 / 0.67</td><td>10.3 / 0.85↓</td><td> $1 1 . 6 / 0 . 8 4 ^ { \downarrow }$ </td></tr></table>

Choosing neighbors and testing. Each model’s five neighbors are the closest models in centered coordinates by Euclidean distance, in size by the diference in log |θ|, in <sup>¯</sup>ℓ by the diference in mean log-likelihood on the reference trajectories, in family by taking models of the same family in order of size, and in success rate by the diference in measured success rate on that benchmark. For families with fewer than five other members, the remaining neighbors are taken from other families in order of size. The efect size is the mean behavioral distance to the remaining models minus the mean distance to the five neighbors, averaged over models and divided by the standard deviation of the behavioral distance over all pairs. Figure 1(c) applies the same definition to the k-th nearest model alone; ranks $k \leq 5$ are shown individually and larger ranks as bin averages. Significance is assessed by a twosided Wilcoxon signed-rank test (Wilcoxon, 1945) paired by model. The comparison between coordinate neighbors and success-rate neighbors, marked † in the main-text table, uses the same paired test: it asks whether the distance to success-rate neighbors minus the distance to coordinate neighbors, divided by the all-pairs standard deviation, is positive. These are paired tests with models as the unit; pairs share models and no correction for multiple comparisons is applied, so the significance markers are descriptive guides rather than exact error rates. The main conclusions rest on efects that are large (at least 0.5 for coordinate neighbors in ALF-World) and consistent across measures rather than on individual markers; small diferences marked as significant, such as those between coordinate and success-rate neighbors in the other environments, should be read as suggestive.

## B.4.2 Robustness of the neighbor efect

The main conclusion is that coordinate neighbors behave alike when run. We check that this does not depend on the choice of behavioral distance, on how performance is controlled, or on how the coordinates are built.

Does it hold for other behavioral distances? Part A of Table 8 adds the divergence position, conditional disagreement, profile, and success pattern to the four distances of the main text. The divergence position and conditional disagreement, which measure divergence early in the episode, follow the same pattern as the first action. For the profile, which aggregates repetition and invalid-action rates, and for the success pattern, neighbors chosen by success rate are closer than coordinate neighbors. These measures reflect how often an agent fails and are therefore tied closely to its success rate.

Does it survive regressing out performance? Instead of comparing with neighbors chosen by success rate, part B of Table 8 removes the efect of performance by regression. We regress each behavioral distance on the diference in success rate, the pair’s mean success rate and its square, and the diference in log |θ|, and compute the efect of coordinate neighbors on the residuals. We control for the mean success rate as well because strong models converge on the shortest solutions and therefore act alike. For the success pattern, we instead subtract its expectation under a null distribution generated by the curveball algorithm (Strona et al., 2014), which shufles outcomes while keeping each model’s success rate and each task’s dificulty fixed. After this control, the efect of coordinate neighbors remains on every aspect in ALFWorld, the environment of the coordinates, and mainly on the local and early-step measures in the other environments.

Does it depend on how the coordinates are built or on families? The top half of Table 9 chooses neighbors in uncentered coordinates. The efect is weaker than with centered coordinates, presumably because the overall likelihood level then influences which models count as neighbors. The bottom half excludes models of the same family from the coordinate neighbors. The efect remains on all four benchmarks, so coordinate neighbors do not behave alike merely because they belong to the same family.

## B.4.3 Reliability of the behavioral distances

Are the distances reproducible across tasks? Table 10 shows the correlation between distance matrices computed on two random halves of the tasks. The composition, global, and local measures reach 0.85 or higher, so diferent task sets give nearly the same distances. The early-step measures, such as the first action, and the success pattern are somewhat lower, especially on BFCL, which has only 25 tasks.

## C Composing harnesses: details

This appendix supplements §4. Appendix C.1 gives the setup and Appendix C.2 the full additivity table and figure.

## C.1 Harness implementations, models, and references

Input construction. The input is assembled in the order fixed part, history selected by context management, plan and reasoning for the current step, and action instruction. The fixed part consists of the system instructions, tool definitions, and initial observation (including the task instruction) and is shared by all harnesses. One block of the external history is an executed action and the resulting observation; past plans and reasoning are not included. The budget for the history part is 8,192 tokens, and the whole input is at most 16,384 tokens including the reserve for the output; when the budget is exceeded, the oldest blocks are dropped first and, if necessary, the latest observation is truncated to its head and tail.

## Components.

• Context management: windowed passes the initial observation and the last four blocks verbatim. retrieve additionally selects, from the older history, blocks relevant to the current observation by BM25 (Robertson and Zaragoza, 2009) (weight

Table 8: Neighbor efect sizes for all behavioral distances, including those not in the main text; definitions as in Table 3, 100 models. A: raw behavioral distances, comparing how neighbors are chosen; bold marks the largest value among the four neighbor types available without running. B: efect of coordinate neighbors on residuals after regressing out success rate and size. Unmarked values have $p < 0 . 0 1 , { } ^ { + }$ marks $p < 0 . 0 5$ , and gray values are not significant.
<table><tr><td rowspan="2"></td><td colspan="2">Composition</td><td>Global</td><td>Local</td><td colspan="3">Early steps</td><td colspan="2">Outcome</td></tr><tr><td>Type JS</td><td>Profile</td><td>Edit</td><td>2-gram</td><td>First</td><td>Cond.</td><td>Diverg.</td><td>Success</td><td>Pattern</td></tr><tr><td colspan="10">A Choice of neighbors</td></tr><tr><td colspan="10">ALFWorld</td></tr><tr><td>Coordinates</td><td>0.52</td><td>0.63</td><td>0.88</td><td>0.96</td><td>0.61</td><td>0.51</td><td>0.70</td><td>0.60</td><td>0.32</td></tr><tr><td>Size</td><td>0.23</td><td>0.40</td><td>0.40</td><td>0.41</td><td>0.30</td><td>0.26</td><td>0.34</td><td>0.41</td><td>0.27</td></tr><tr><td>Family</td><td>0.28</td><td>0.48</td><td>0.62</td><td>0.68</td><td>0.49</td><td>0.43</td><td>0.62</td><td>0.49</td><td>0.35</td></tr><tr><td>é</td><td>0.05</td><td>0.23</td><td>0.18+</td><td>0.20</td><td>0.11</td><td>0.15</td><td>0.16+</td><td>0.20</td><td>0.17</td></tr><tr><td>Reference: success rate</td><td>0.30</td><td>0.78</td><td>0.65</td><td>0.74</td><td>0.41</td><td>0.31</td><td>0.52</td><td>一</td><td>0.61</td></tr><tr><td colspan="10">TextWorld</td></tr><tr><td>Coordinates</td><td>0.33</td><td>0.52</td><td>0.52</td><td>0.54</td><td>0.21</td><td>0.51</td><td>0.43</td><td>0.61</td><td>0.65</td></tr><tr><td>Size</td><td>0.10</td><td>0.35</td><td>0.25</td><td>0.27+</td><td>0.14</td><td>0.32</td><td>0.17</td><td>0.58</td><td>0.56</td></tr><tr><td>Family</td><td>0.27</td><td>0.47</td><td>0.42</td><td>0.44</td><td>0.25</td><td>0.46</td><td>0.34</td><td>0.56</td><td>0.61</td></tr><tr><td>ē</td><td>0.06</td><td>0.20</td><td>0.14</td><td>0.14</td><td>0.04</td><td>0.16</td><td>0.10</td><td>0.27</td><td>0.29</td></tr><tr><td>Reference: success rate</td><td>0.22</td><td>0.80</td><td>0.50</td><td>0.52</td><td>0.16</td><td>0.47</td><td>0.31+</td><td>一</td><td>1.06</td></tr><tr><td colspan="10">BFCL</td></tr><tr><td>Coordinates</td><td>0.18</td><td>0.36</td><td>0.26</td><td>0.38</td><td>0.32</td><td>0.22</td><td>0.46</td><td>0.47</td><td>0.28</td></tr><tr><td>Size</td><td>0.10+</td><td>0.20</td><td>0.13</td><td>0.21</td><td>0.19</td><td>0.21</td><td>0.30</td><td>0.27</td><td>0.25</td></tr><tr><td>Family</td><td>0.21</td><td>0.29</td><td>0.26</td><td>0.32</td><td>0.25</td><td>0.28</td><td>0.40</td><td>0.30</td><td>0.29</td></tr><tr><td>é</td><td>0.03</td><td>0.08</td><td>0.06</td><td>0.09+</td><td>0.09</td><td>0.04</td><td>0.11+</td><td>0.15</td><td>0.17</td></tr><tr><td>Reference: success rate</td><td>0.23</td><td>0.49</td><td>0.30</td><td>0.39</td><td>0.21</td><td>0.26</td><td>0.50</td><td></td><td>0.66</td></tr><tr><td colspan="10">MiniWoB++</td></tr><tr><td>Coordinates</td><td>0.24</td><td>0.34</td><td>0.38</td><td>0.50</td><td>0.50</td><td>0.11+</td><td>0.48</td><td>0.39</td><td>0.40</td></tr><tr><td>Size</td><td>0.12</td><td>0.16</td><td>0.20</td><td>0.27</td><td>0.35</td><td>-0.02</td><td>0.30</td><td>0.42</td><td>0.44</td></tr><tr><td>Family</td><td>0.22</td><td>0.23</td><td>0.32</td><td>0.47</td><td>0.35</td><td>0.08</td><td>0.36</td><td>0.37</td><td>0.37</td></tr><tr><td>é</td><td>0.08</td><td>0.10</td><td>0.10+</td><td>0.16+</td><td>0.09+</td><td>0.10+</td><td>0.11+</td><td>0.20</td><td>0.21</td></tr><tr><td>Reference: success rate</td><td>0.13</td><td>0.63</td><td>0.37</td><td>0.42</td><td>0.28</td><td>0.19+</td><td>0.29</td><td>一</td><td>0.98</td></tr><tr><td colspan="10">B Coordinate neighbors, residuals after removing success rate and size</td></tr><tr><td>ALFWorld</td><td>0.22</td><td>0.09+</td><td>0.61</td><td>0.68</td><td>0.31</td><td>0.20</td><td>0.34</td><td></td><td>0.24</td></tr><tr><td>TextWorld</td><td>0.04+</td><td>0.03+</td><td>0.18</td><td>0.23</td><td>-0.13</td><td>0.15</td><td>0.14</td><td></td><td>0.22</td></tr><tr><td>BFCL</td><td>0.04</td><td>0.09</td><td>0.09</td><td>0.20</td><td>0.13</td><td>0.10+</td><td>0.24</td><td></td><td>-0.15+</td></tr><tr><td>MiniWoB++</td><td>0.03</td><td>0.11</td><td>0.09</td><td>0.31</td><td>0.31</td><td>0.01</td><td>0.28</td><td></td><td>0.20</td></tr></table>

0.7) and recency (0.3) and inserts up to 1,536 tokens of them. summarize splits the older history, summarizes the parts, and merges them hierarchically into a JSON memory of at most 1,024 tokens that replaces the history. If the history has at most four blocks and uses at most 75% of the budget, neither retrieval nor summarization is applied and the history is passed raw.

• Planning: replan rewrites from scratch, at every step and in at most 256 tokens, the confirmed progress, the next subgoal, and the observation that would confirm its achievement, and adds them to the input of the current step.

• Acting: think generates, for the next single action, step-by-step reasoning of at most 512 tokens that organizes the goal, constraints, assumptions, and expected efect of the action, and adds it to the input of the current step (the plan, if any, is also included in the input).

Plans and reasoning are not carried over to the next step.

Generation and scoring. Generation (rollouts) and scoring (teacher forcing) use the same construction; for scoring, each harness builds its input from the same reference history $h _ { t }$ , and the raw loglikelihood of the fixed reference action is computed. The action being scored, future observations, and rewards are not used in the input.

LLMs. The 12 LLMs are Qwen3-8B (Yang et al., 2025), Qwen3.8-27B (Qwen Team, 2026b), Qwen3.5-27B/9B/2B (Qwen Team, 2026a), Gemma4-E4B/12B/31B (Gemma Team,

Table 9: Variants of coordinate neighbors, with definitions as in Table 8. Top: neighbors in uncentered coordinates. Bottom: coordinate neighbors restricted to models from other families. Neighbors in centered coordinates are in Table 8. Unmarked values have $p < 0 . 0 1 , { } ^ { + }$ marks $p < 0 . 0 5 .$ , and gray values are not significant.
<table><tr><td rowspan="2"></td><td colspan="2">Composition</td><td>Global</td><td>Local</td><td colspan="3">Early steps</td><td colspan="2">Outcome</td></tr><tr><td>Type JS</td><td>Profile</td><td>Edit</td><td>2-gram</td><td>First</td><td>Cond.</td><td>Diverg.</td><td>Success</td><td>Pattern</td></tr><tr><td colspan="10">Neighbors in uncentered coordinates</td></tr><tr><td colspan="10">Raw behavioral distance</td></tr><tr><td>ALFWorld</td><td>0.32</td><td>0.51</td><td>0.65</td><td>0.73</td><td>0.45</td><td>0.38</td><td>0.52</td><td>0.64</td><td>0.45</td></tr><tr><td>TextWorld</td><td>0.19</td><td>0.49</td><td>0.37</td><td>0.40</td><td>0.12+</td><td>0.39</td><td>0.27+</td><td>0.66</td><td>0.68</td></tr><tr><td>BFCL</td><td>0.12</td><td>0.31</td><td>0.21</td><td>0.31</td><td>0.22</td><td>0.22</td><td>0.45</td><td>0.47</td><td>0.35</td></tr><tr><td>MiniWoB++</td><td>0.14</td><td>0.28</td><td>0.25</td><td>0.40</td><td>0.38</td><td>0.12+</td><td>0.34</td><td>0.38</td><td>0.38</td></tr><tr><td colspan="10">Residuals after removing success rate and size</td></tr><tr><td>ALFWorld</td><td>0.10</td><td>-0.03</td><td>0.47</td><td>0.54</td><td>0.21</td><td>0.14</td><td>0.28</td><td></td><td>0.29</td></tr><tr><td>TextWorld</td><td>0.06+</td><td>0.11</td><td>0.15</td><td>0.19</td><td>-0.06+</td><td>0.14</td><td>0.14</td><td></td><td>0.20</td></tr><tr><td>BFCL</td><td>0.05</td><td>0.07</td><td>0.10</td><td>0.20</td><td>0.07+</td><td>0.16</td><td>0.28</td><td></td><td>0.04</td></tr><tr><td>MiniWoB++</td><td>0.03</td><td>0.16</td><td>0.11+</td><td>0.31</td><td>0.28</td><td>0.07</td><td>0.26</td><td></td><td>0.16</td></tr><tr><td colspan="10">Coordinate neighbors restricted to other families</td></tr><tr><td colspan="10">Raw behavioral distance</td></tr><tr><td>ALFWorld</td><td>0.47</td><td>0.56</td><td>0.60</td><td>0.61</td><td>0.45</td><td>0.36</td><td>0.41</td><td>0.54</td><td>0.26</td></tr><tr><td>TextWorld</td><td>0.31</td><td>0.44</td><td>0.45</td><td>0.44</td><td>0.13</td><td>0.43</td><td>0.34</td><td>0.52</td><td>0.54</td></tr><tr><td>BFCL</td><td>0.08</td><td>0.24</td><td>0.15+</td><td>0.24</td><td>0.23</td><td>0.06</td><td>0.26</td><td>0.40</td><td>0.19</td></tr><tr><td>MiniWoB++</td><td>0.21</td><td>0.31</td><td>0.26</td><td>0.28</td><td>0.42</td><td>0.02</td><td>0.36</td><td>0.35</td><td>0.30</td></tr><tr><td colspan="10">Residuals after removing success rate and size</td></tr><tr><td>ALFWorld</td><td>0.21</td><td>0.08+</td><td>0.31</td><td>0.31</td><td>0.16</td><td>0.13</td><td>0.09</td><td></td><td>0.04</td></tr><tr><td>TextWorld</td><td>0.07</td><td>0.01</td><td>0.17</td><td>0.17</td><td>-0.18</td><td>0.12</td><td>0.06</td><td></td><td>0.18</td></tr><tr><td>BFCL</td><td>-0.01</td><td>0.02</td><td>0.04</td><td>0.12</td><td>0.08</td><td>-0.01</td><td>0.10</td><td></td><td>-0.28</td></tr><tr><td>MiniWoB++</td><td>0.02</td><td>0.08+</td><td>-0.06</td><td>0.08</td><td>0.24</td><td>-0.05</td><td>0.17</td><td></td><td>0.05</td></tr></table>

Table 10: Reliability of the behavioral distances: correlation between distance matrices computed on two random halves of the tasks, averaged over 30 splits; all standard deviations are at most 0.08. The profile and conditional disagreement cannot be split by task and are omitted.
<table><tr><td colspan="4">ALFWorld TextWorld BFCL MiniWoB++</td></tr><tr><td>Reliability</td><td colspan="4">correlation across task halves</td></tr><tr><td>Type JS</td><td>0.97</td><td>0.89 0.89</td><td>0.95</td><td></td></tr><tr><td>Edit distance</td><td>0.91</td><td>0.92 0.89</td><td></td><td>0.92</td></tr><tr><td>2-gram</td><td>0.88</td><td>0.90</td><td>0.87</td><td>0.85</td></tr><tr><td>First action</td><td>0.78</td><td>0.82</td><td>0.70</td><td>0.91</td></tr><tr><td>Divergence</td><td>0.76</td><td>0.90</td><td>0.62</td><td>0.91</td></tr><tr><td>Success pattern</td><td>0.77</td><td>0.80</td><td>0.68</td><td>0.84</td></tr></table>

Table 11: Setup of the multi-harness experiment: 12 $\mathrm { L L M s } \times 1 2$ harnesses, scored on the benchmarks that have reference trajectories (ALFWorld and SWE-Bench use 6 of the 12 LLMs). N is the number of reference trajectories and “steps” the number of reference steps scored by all agents.
<table><tr><td></td><td>Reference trajectories (source) N / steps</td><td></td><td>LLMs scored</td></tr><tr><td>τ-bench</td><td>HAL airline</td><td>359 / 4,213 12</td><td></td></tr><tr><td>ALFWorld</td><td>own, 3 LLMs × 12 harnesses</td><td>210 / 2,2996</td><td></td></tr><tr><td></td><td>SWE-Bench HAL SWE-agent</td><td>100 / 3,9166</td><td></td></tr></table>

2026), GPT-OSS-20B/120B (OpenAI, 2025),

GLM-4.7-Flash (Z.ai, 2026), and Devstral-Small-2-24B (Rastogi et al., 2025), all among the 100 models of §3.2. SWE-Bench was scored with 6 of them (Qwen3-8B, Qwen3.8-27B, Gemma4-E4B, GPT-OSS-20B/120B, GLM-4.7-Flash).

Reference trajectories and coordinates. τ- bench: 359 HAL (Kapoor et al., 2026) airline traces mapped to the 30 tasks of the train split of $\tau ^ { 2 } \mathrm { - }$ bench (Barres et al., 2026); of their 4,236 steps, the 4,213 scored by all 144 agents are used for the coordinates. SWE-Bench: 100 HAL traces of SWEagent (Yang et al., 2024), 3,916 steps. ALFWorld: 210 train rollouts of our own $3 { \mathrm { ~ L L M s ~ } } \times 1 2$ harnesses, of which the 2,299 steps scored by all 72 agents (6 LLMs × 12 harnesses) are used. All coordinates are the centered step version of §2.3.

Change in the mean log-likelihood. The component removed from the coordinates by centering, $\bar { \ell } ( m , A ) - \bar { \ell } ( m , \mathrm { b a s e } )$ , is +9.6 for retrieve, +3.5 for summarize, −4.2 for replan, and −6.2 for think on τ -bench, in nats per step (medians over LLMs), and its sign is shared by at least 11 of the 12 LLMs. Context management makes the reference actions more natural, whereas planning and reasoning move the agent away from them.

Table 12: Full additivity table (medians over LLMs: cosine, norm ratio, relative reconstruction error). τ-bench uses 12 LLMs; ALFWorld and SWE-Bench use 6 of the 12 LLMs (ALFWorld uses only the 2,299 reference steps scored by all 72 agents). The pooled rows take the median over all LLM–harness cases: the five two-component harnesses (60 cases on τ-bench, 30 on ALFWorld and SWE-Bench) and the two three-component harnesses (24 / 12 / 12 cases).
<table><tr><td></td><td colspan="3">τ-bench</td><td colspan="3">ALFWorld</td><td colspan="3">SWE-Bench</td></tr><tr><td></td><td>cos</td><td>Norm ratio</td><td>Rel. error</td><td>COS</td><td>Norm ratio</td><td>Rel. error</td><td>cOs</td><td>Norm ratio</td><td>Rel. error</td></tr><tr><td>retrieve + replan</td><td>0.97</td><td>0.99</td><td>0.25</td><td>0.62</td><td>0.95</td><td>0.94</td><td>0.97</td><td>0.97</td><td>0.25</td></tr><tr><td>retrieve + think</td><td>0.94</td><td>1.00</td><td>0.34</td><td>0.61</td><td>0.95</td><td>0.95</td><td>0.96</td><td>0.96</td><td>0.28</td></tr><tr><td>summarize + replan</td><td>0.91</td><td>0.96</td><td>0.45</td><td>0.60</td><td>0.89</td><td>1.00</td><td>0.78</td><td>0.89</td><td>0.72</td></tr><tr><td>summarize + think</td><td>0.87</td><td>0.98</td><td>0.51</td><td>0.63</td><td>0.91</td><td>0.93</td><td>0.77</td><td>0.86</td><td>0.76</td></tr><tr><td>replan + think</td><td>0.81</td><td>0.74</td><td>0.77</td><td>0.70</td><td>0.73</td><td>1.03</td><td>0.74</td><td>0.73</td><td>0.95</td></tr><tr><td>Two components, pooled</td><td>0.90</td><td>0.98</td><td>0.46</td><td>0.63</td><td>0.89</td><td>0.97</td><td>0.81</td><td>0.90</td><td>0.65</td></tr><tr><td>Three components, pooled</td><td>0.82</td><td>0.87</td><td>0.68</td><td>0.52</td><td>0.72</td><td>1.25</td><td>0.77</td><td>0.76</td><td>0.84</td></tr></table>

![](images/5be476900f49bfb9b8a5793c7ece23c1480057f33c34d149eaba2314cd17cdba.jpg)  
Figure 5: Additivity for all seven compositions on τ-bench (12 LLMs), in the PC1–PC2 projection of Figure 3; axes are scaled independently. The last panel identifies the LLMs by their BASE positions (gray: the other harnesses). Two-component compositions are drawn as parallelograms, three-component ones as chains $\mathrm { B A S E } + \Delta _ { A } + \Delta _ { B } + \Delta _ { C }$ Titles give the cosine and norm ratio in the full coordinate space (medians over LLMs).

## C.2 Additivity: full tables

Table 12 and Figure 5 give the full results. On ALF-World and SWE-Bench (6 LLMs) the direction of the composed displacement also agrees with the prediction, but more weakly than on τ-bench (median cosines over all LLMs × 5 two-component compositions of 0.63 and 0.81), and on ALFWorld the relative reconstruction error reaches about 1. Each column is a median of per-LLM values, so the columns do not satisfy the identity below exactly. The ALFWorld reference trajectories are rollouts of the agents under study themselves, and only 2,299 of their steps are scored by all 72 agents.

Identity between norm ratio and cosine. With $v = z ( A + B ) - z ( \mathrm { b a s e } ) , \hat { v } = \Delta _ { A } + \Delta _ { B } , e = \hat { v } - v ,$ norm ratio $r = \| v \| / \| \hat { v } \|$ , and $c = \cos ( \hat { v } , v )$ , the relative reconstruction error is $\| e \| ^ { 2 } / \| v \| ^ { 2 } = 1 +$ $r ^ { - 2 } - 2 c / r$ . Even when the norms agree, an error remains whenever $c < 1$