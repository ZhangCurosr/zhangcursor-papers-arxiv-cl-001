# LLM-Based Exogenous World Modeling for Dynamic Scheduling

Ni Zhang<sup>1</sup>, Haofeng Yuan<sup>2</sup>, Jianan Zhou<sup>3</sup>, Shiji Song<sup>2</sup>, Zhiguang Cao<sup>1</sup>

<sup>1</sup>Singapore Management University <sup>2</sup>Tsinghua University <sup>3</sup>Nanyang Technological University

ni.zhang.2025@phdcs.smu.edu.sg, yhf22@mails.tsinghua.edu.cn

jianan004@e.ntu.edu.sg, shijis@mail.tsinghua.edu.cn, zgcao@smu.edu.sg

## Abstract

Dynamic scheduling requires dispatching online arrivals while anticipating future jobs, as poor decisions can compound downstream costs under nonlinear tardiness objectives. We show that a pretrained large language model (LLM) can serve as the exogenous component of a structured world model, forecasting uncertain future job arrivals. Specifically, we exploit an afterstate decomposition, which separates known deterministic shop transitions from uncertain exogenous job arrivals, enabling the LLM to forecast only the latter through a compact belief representation. Building on this, we develop Afterstate World-Model Planning (AWMP), which combines an evolved, belief-conditioned dispatching strategy with short scenario-based rollouts. A surprise-triggered refresh mechanism queries the LLM only when newly observed arrivals deviate suficiently from previous forecast, and online residual self-calibration further conditions the LLM on its recent forecast errors to handle nonstationary arrivals. Across diverse arrival distributions, AWMP reduces weighted tardiness by 6-21% on dynamic job-shop and 22-46% on flexible job-shop benchmarks, relative to the best of eight classical priority-dispatching rules.

## 1 Introduction

Scheduling determines how a collection of jobs should be processed on limited machine resources while satisfying operational constraints such as precedence, machine capacity, and delivery commitments (Pinedo and Hadavi 1992; Smit et al. 2025b). Classical scheduling formulations assume that all jobs and their processing requirements are known in advance, allowing optimization over a fixed problem instance. Real manufacturing systems, however, are inherently dynamic (Vieira, Herrmann, and Lin 2003; Ouelhadj and Petrovic 2009). New jobs arrive dynamically over time while machines are already processing previously released work, and the scheduler must repeatedly dispatch operations without knowing the future workload.

Under such uncertainty, scheduling performance is typically evaluated by due-date-oriented objectives. A widely adopted objective is total weighted tardiness, which incurs no cost before a job’s due date but imposes a growing penalty once the due date is missed. This nonlinear objective creates sharp due-date thresholds: a dispatching decision that appears locally beneficial may occupy critical machine capacity, delay a later high-priority job past its due date, and incur a disproportionate downstream penalty. Consequently, efective dynamic scheduling requires more than reacting to the currently available operations, but also anticipating how future job arrivals will reshape shop congestion and incorporating those expectations into decisions.

Existing approaches address this challenge either by reacting to the current workload or by explicitly modeling the future. 1) Classical priority-dispatching rules (PDRs), such as weighted shortest processing time (WSPT) and apparent tardiness cost (ATC), are fast and training-free (Panwalkar and Iskander 1977; Haupt 1989), but they rank only the currently available operations and carry no explicit representation of future arrivals. 2) Anticipatory optimization methods, including rolling-horizon replanning, sample-average approximation (SAA), and multiple-scenario approaches (MSA), evaluate decisions against future scenarios sampled from a prespecified stochastic model through repeated reoptimization (Powell 2007; Kleywegt, Shapiro, and Homem-de Mello 2002; Bent and Van Hentenryck 2004; Li et al. 2025). Their performance relies heavily on the fidelity of the assumed model, and the repeated reoptimization can be extremely expensive. 3) Recent neural methods learn dispatching policies from simulated trajectories with stochastic arrivals (Chang et al. 2022; Liu, Piplani, and Toro 2022, 2023; Liu et al. 2024b; Tang et al. 2026). While capable ofcapturing complex dynamic patterns, they typically entangle the exogenous arrival process with dispatching decisions and require task- and distribution-specific training, with retraining needed once either changes. These limitations motivate an anticipatory scheduler that can exploit observed arrival patterns without presupposing a parametric stochastic model, and that adapts without heavy retraining or reoptimization.

In this paper, we leverage a pretrained large language model (LLM) to model the exogenous component of a structured world model by forecasting future job arrivals, while explicitly accounting for known scheduling dynamics. Specifically, we employ an afterstate decomposition to separate the deterministic efects of dispatch actions from the uncertainty associated with future job arrivals. At each decision step, a dispatch action transforms the current state into an afterstate, representing the resulting shop configuration before subsequent arrivals are incorporated. This afterstate captures endogenous changes, including machine availability, operation progress, and precedence status, governed by known deterministic dynamics, leaving future job arrivals as the sole source of exogenous uncertainty. Accordingly, the LLM estimates a compact, decision-relevant belief over future arrival pressure, represented by the expected numbers of jobs arriving in the early and late portions of a fixed prediction horizon. The belief summarizes anticipated arrival intensity without specifying individual job identities, machine requirements, or operation-level dynamics. This compact interface makes the forecast directly usable by a dispatching planner while avoiding the need to learn a high-dimensional transition model.

![](images/d65a32c9e471802387deb17e00230bf08a559fbe98e2e49863ffc682e07402bf.jpg)  
Figure 1: Illustration of dynamic scheduling with online job arrivals and dispatching decisions under unknown future arrivals. At the current time $c _ { t } ,$ the scheduler observes the released jobs and current shop state and selects the next operation to dispatch. Here, $J _ { i }$ denotes job $i , J _ { i } – O _ { j }$ denotes the j-th operation of job i, and “due” indicates the corresponding job due date. Future job arrivals remain unknown.

Building on this decomposition, we develop AWMP, an Afterstate World-Model Planning framework for dynamic scheduling. At each decision step, AWMP maps every feasible dispatching action to its deterministic afterstate, and scores the action using both its afterstate features and the belief-derived features that capture future arrival pressure. A scenario generator then converts the early-late belief into concrete arrival scenarios, under which the retained action candidates are evaluated through short rollouts, and the action with the best aggregated evaluation is selected for dispatching.

Moreover, AWMP makes the LLM forecast adaptive and eficient for online planning through three mechanisms. First, online residual self-calibration feeds recent predictionoutcome pairs and forecast errors into subsequent prompts, enabling the LLM to adapt its arrival forecasts to nonstationary patterns through in-context learning. Second, belief amortization caches the forecast across all candidate evaluations and rollouts within a decision step, while the cached belief is also reused across decision steps, where the LLM is queried only when realized arrivals deviate suficiently from the current belief. This surprise-triggered refresh reduces the number of LLM calls from one per rollout to a handful per episode. Third, meta-evolution searches over compact, human-readable strategy objects, comprising the scoring and value functions, rollout parameters, and forecast-refresh threshold, to automatically discover an effective belief-conditioned planning strategy.

Our main contributions are as follows:

• We formalize dynamic scheduling through an afterstate decomposition that separates the uncertain, exogenous job arrivals from known, deterministic scheduling dynamics.

• We instantiate the exogenous arrival-belief component of the structured world model with an LLM to forecast a compact arrival belief, and make this forecast adaptive and eficient through online residual self-calibration and belief reuse with surprise-triggered refresh.

• We develop AWMP, a belief-conditioned afterstate planner integrating coarse action screening and short-rollout evaluation under fixed scenario generation, with the planning strategy discovered ofline via meta-evolution.

• Experiments across four arrival distributions show that AWMP discovers strategies that outperform the best of eight classical PDRs, reducing total tardiness by 6-21% on dynamic job-shop instances and 22-46% on dynamic flexible job-shop instances, while running in seconds.

## 2 Related Work

Classical dynamic scheduling methods. Classical methods for dynamic scheduling broadly fall into reactive dispatching and anticipatory optimization. Reactive PDRs, such as WSPT and ATC, re-rank eligible operations whenever the shop state changes (Panwalkar and Iskander 1977; Haupt 1989). They are fast and training-free, but evaluate only the current observable workload with no explicit representation of future arrivals. Anticipatory methods, including rollinghorizon optimization, SAA, and MSA, instead evaluate decisions against expected or sampled future workloads (Powell 2007; Kleywegt, Shapiro, and Homem-de Mello 2002; Bent and Van Hentenryck 2004; Li et al. 2025). Their lookahead quality depends on the assumed arrival model, while repeatedly solving the resulting subproblems can be computationally expensive. AWMP occupies a middle ground: it retains low-latency online dispatching but augments it with a compact arrival forecast inferred by an LLM from the observed arrival history, without requiring the arrival pattern to be specified as a parametric stochastic process.

Neural dynamic scheduling methods. Neural combinatorial optimization has introduced graph and attention encoders combined with policy-gradient or value-based reinforcement learning (RL) for routing and scheduling (Kool, van Hoof, and Welling 2019; Zhang et al. 2020). Recent methods extend learned dispatching policies to dynamic job-shop scheduling problems (JSSPs) and flexible job-shop scheduling problems (FJSPs) with random job arrivals (Luo 2020; Liu, Piplani, and Toro 2022, 2023; Smit et al. 2025a; Yu et al. 2026). These methods can capture complex interactions among jobs, operations, and machines, but they generally learn from specific simulated trajectories and may require retraining when the arrival process or optimization objective changes. In contrast, AWMP trains no neural policy, but uses a frozen pretrained LLM to estimate aggregate arrival pressure, and converts it into dispatching decisions through a compact planning strategy that is evolved ofline and frozen before evaluation.

Learned world models for planning. Model-based RL learns a transition model of the environment to evaluate and select future actions. Representative approaches include MuZero (Schrittwieser et al. 2020; Antonoglou et al. 2022), Dreamer (Hafner et al. 2021), and TD-MPC (Hansen, Wang, and Su 2022). These methods are designed for settings in which substantial environment dynamics must be learned from interaction. Dynamic scheduling has a diferent structure: once an action is specified, the transitions in shop status are deterministic and known. Our afterstate decomposition exploits this structure by retaining exact endogenous dynamics and isolating the future arrival stream as the only exogenous uncertainty. AWMP forms a structured world model from two parts: a low-dimensional predictor of exogenous arrival pressure and the deterministic scheduling simulator.

LLMs for optimization and heuristic discovery. LLMs have recently been used to generate and refine algorithms, heuristics, and environment dynamics simulators (Romera-Paredes et al. 2024; Liu et al. 2024a; Ye et al. 2024). AWMP uses LLMs in two separate roles. During online scheduling, a pretrained LLM instantiates the exogenous component of the structured world model by forecasting aggregate arrival pressure. During ofline meta-evolution, a proposer LLM evolves a compact, human-readable planning strategy incorporating the scoring rule, value estimator, and rollout parameters, rather than an independent solver program.

## 3 Methodology

Our methodology is organized around three questions: what to forecast, how to forecast, and how to plan with a structured world model, and AWMP addresses them through a modular design, as illustrated in Fig. 2. First, we formalize dynamic scheduling through an afterstate decomposition, which separates the known endogenous shop dynamics from unknown exogenous job arrivals. This decomposition identifies future arrivals, rather than the complete shop transition, as the only object that requires forecasting (Section 3.1). Then, we instantiate the exogenous component with a frozen, pretrained LLM to forecast a low-dimensional belief of early-late future arrival pressure from the observed arrival history, and introduce residual self-calibration to adapt the forecast in an online manner, especially under nonstationary arrivals (Section 3.2). Building on this, we develop AWMP, a belief-conditioned afterstate planner that first identifies promising actions with a compact scoring rule, and then evaluates them through short rollouts under arrival scenarios generated from the forecast belief (Section 3.3). Intra- and inter-step belief caching and surprise-triggered belief refresh amortize LLM calls without repeatedly forecasting at every rollout or decision step. Finally, meta-evolution discovers a compact, human-readable planning strategy ofline, which is then frozen for held-out evaluation (Section 3.4).

## 3.1 Afterstate Decomposition

We consider dynamic JSSPs and FJSPs, where jobs arrive online and dispatching decisions must be made before observing future workload. Let t index decision steps and $c _ { t }$ denote the corresponding physical shop time. At decision step $t ,$ the scheduler observes the current shop state $s _ { t }$ and selects a feasible dispatch action $a _ { t } \in \mathcal { A } ( s _ { t } )$ , where $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } _ { t } )$ is the set of all feasible operation-machine assignments. In JSSPs, the machine for each operation is fixed, while in FJSPs, machine selection is part of the action. The state $s _ { t }$ contains the current shop configuration, including machine availability, job progress, precedence status, and observed arrival history $h _ { t }$ . The efect of an action can be separated from that of subsequent arrivals. Applying $a _ { t }$ first produces an afterstate

$$
\sigma _ { t } = T ( s _ { t } , a _ { t } ) ,\tag{1}
$$

where $T$ is the known action-to-afterstate transition. The afterstate captures the updated machine availability, operation progress, and, for FJSPs, the selected machine assignment before any new arrivals are incorporated. These changes are endogenous: once $s _ { t }$ and $a _ { t }$ are specified, they are fully deterministic.

We define the set of future job arrivals as:

$$
\mathcal { Z } _ { t } = \{ ( \rho _ { j } , \zeta _ { j } ) : \rho _ { j } > c _ { t } \} \sim P _ { \beta } ( \cdot \mid h _ { t } ) ,\tag{2}
$$

where $\rho _ { j }$ is the release time of future job $j , \zeta _ { j }$ collects its attributes, such as its processing time, due date, and weight, and $\beta$ controls the arrival pattern such as intensity. The arrival process is exogenous: dispatching actions do not afect which jobs arrive or their release times.

Nevertheless, an action may change the physical time ofthe next decision step, and hence which portion of $\mathcal { Z } _ { t }$ becomes observable before that decision. The next state is therefore determined by an event-advance function $g \colon$

$$
s _ { t + 1 } = g ( \sigma _ { t } , \mathcal { Z } _ { t } ) ,\tag{3}
$$

which is likewise known and deterministic given the afterstate $\sigma _ { t }$ and a realization of arrival stream $\mathcal { Z } _ { t }$ . The function $g$ advances shop time, executes the deterministic scheduling dynamics, and incorporates exactly the jobs released during the transition. Consequently, $T$ and $g$ need not be learned, and the only unknown component required for exact anticipatory planning is the conditional arrival process $P _ { \beta } ( \cdot \mid h _ { t } )$

![](images/70c227be4c5a615cb325a69e21ee207eae1cff0cabfae130d2902b2459861de6.jpg)  
Figure 2: Overview of our AWMP framework. 1) The dynamic scheduling environment provides the current shop state and the observed arrival history. 2) The afterstate decomposition separates the known, deterministic shop transition from the exogenous job-arrival process. 3) A pretrained LLM forms the exogenous component of the structured world model and, with online residual self-calibration, forecasts a compact early-late arrival belief. 4) The belief-conditioned planner first scores feasible actions, retains the top-K candidates, and evaluates them through short scenario-based rollouts. 5) Meta-evolution optimize planning strategies ofline. 6) The best strategy is frozen and evaluated on held-out dynamic JSSP and FJSP benchmarks.

This decomposition provides direct insight into how a world model can be structured for dynamic scheduling. Because the endogenous shop transitions are known exactly, AWMP retains them in the scheduling simulator and models the uncertain exogenous arrival process separately. A pretrained LLM estimates the decision-relevant efect of $\mathcal { Z } _ { t }$ as a low-dimensional, compact belief from the observed arrival history $h _ { t } .$ , while the planner combines this belief with the exact transitions T and g for rollout evaluation. Together, the exogenous predictor and deterministic simulator constitute the structured world model used by AWMP.

## 3.2 LLM-Based Exogenous World Modeling

As discussed above, reliably predicting the complete arrival stream $\mathcal { Z } _ { t }$ , such as the exact release times and job attributes, is dificult and unnecessary. We instead forecast a compact projection, referred to as the belief, of the future arrival stream $\mathcal { Z } _ { t }$ that reflects the arrival pressure of $P _ { \beta } ( \cdot \ | \ h _ { t } )$ over a temporal horizon relevant to dispatching, without requiring a high-dimensional prediction of individual future jobs. We formulate this compact prediction as the exogenous arrival-belief component of the structured world model and instantiate it with a pretrained LLM.

Once a belief forecast is triggered at decision step t (time $c _ { t } )$ , we divide the prediction horizon $H _ { p }$ into two intervals:

$$
I _ { t } ^ { \mathrm { e a r l y } } = [ c _ { t } , c _ { t } + H _ { p } / 2 ) , I _ { t } ^ { \mathrm { l a t e } } = [ c _ { t } + H _ { p } / 2 , c _ { t } + H _ { p } ] .\tag{4}
$$

The early interval captures near-term arrival pressure that may immediately afect capacity allocation, while the late interval provides a coarser indication of subsequent congestion. The LLM-based arrival predictor receives the observed arrival history $h _ { t }$ and a residual-calibration record $\mathcal { C } _ { t }$ (introduced below), and returns the belief $b _ { t } ,$ , a forecast of the numbers of job arrivals in the early and late intervals:

$$
b _ { t } = \phi ( h _ { t } , \mathcal { C } _ { t } ) = ( \widehat { n } _ { t } ^ { \mathrm { e a r l y } } , \widehat { n } _ { t } ^ { \mathrm { l a t e } } ) ,\tag{5}
$$

where $\phi$ denotes the exogenous arrival-belief predictor, and $\widehat { n } _ { t } ^ { \mathrm { e a r l y } }$ and $\widehat { n } _ { t } ^ { \mathrm { l a t e } }$ are the predicted numbers of arrivals in $I _ { t } ^ { \mathrm { e a r l y } }$ and $I _ { t } ^ { \mathrm { l a t e } }$ , respectively.

The belief $b _ { t }$ provides a shared interface between the arrival predictor and the downstream afterstate planner, and is used in two ways. First, it directly conditions the evolved scoring and value functions, providing side information beyond the currently observable shop status for long-sighted dispatching decisions. Second, a fixed scenario generator, introduced in Section 3.3, converts this aggregate early–late arrival pressure into concrete rollout scenarios by sampling release times within the corresponding intervals. The resulting scenarios allow the planner to evaluate candidate actions through short rollouts while retaining the known deterministic transitions $T$ and $g$ exactly. The interface of the LLMbased arrival predictor is detailed in Appendix B.

Online residual self-calibration. Observed arrival patterns can drift over the scheduling horizon, making recent forecast errors informative for subsequent predictions. We therefore introduce online residual self-calibration, which provides prompt-level closed-loop feedback through recent prediction-outcome residuals.

For a completed previous forecast $b _ { i }$ , let $\widehat { n } _ { i }$ be the predicted number of arrivals over its monitored interval (with “early” and “late” superscripts omitted for brevity) and $n _ { i }$ the realized arrival count. At decision step t, we maintain a rolling calibration record containing the $L$ most recent forecasts whose prediction horizons have fully elapsed:

$$
{ \mathcal { C } } _ { t } = \mathrm { r e c e n t } _ { L } \left\{ \left( n _ { i } , { \widehat { n } } _ { i } \right) : c _ { i } + H _ { p } \leq c _ { t } \right\} .\tag{6}
$$

When $\mathcal { C } _ { t }$ is nonempty, we summarize the recent forecast bias by the mean signed residual:

$$
e _ { t } = \sum _ { ( n _ { i } , \widehat { n } _ { i } ) \in \mathcal { C } _ { t } } \left( n _ { i } - \widehat { n } _ { i } \right) / | \mathcal { C } _ { t } | ,\tag{7}
$$

where a positive $e _ { t }$ indicates underestimation, and a negative value indicates overestimation. When $\left| e _ { t } \right|$ exceeds a calibration threshold, the subsequent prompt explicitly instructs the LLM-based exogenous predictor to correct this bias. Calibration occurs entirely at the online prompt level: the LLM remains frozen and receives no gradient or parameter update. This mechanism thus provides online adaptation to nonstationary arrivals without fitting a new temporal stochastic model or retraining a neural forecaster network. Section 4.3 shows that the residual self-calibration raises the forecast correlation from $r = 0 . 1 2$ to $r = 0 . 9 3$ , improving the online arrival belief and downstream scheduling performance.

## 3.3 Belief-Conditioned Afterstate Planner

The arrival belief provides compact information about future congestion, but it does not by itself determine which action should be dispatched. To this end, AWMP converts the belief $b _ { t }$ into dispatching decisions through a coarse-to-fine planner: a lightweight belief-conditioned scorer first identifies a small set of promising candidate actions, and short scenariobased rollouts then compare and select dispatching actions. The planner is specified by a compact planning strategy:

$$
\Theta = \{ r ( \cdot ) , v ( \cdot ) , K , d , n , \tau \} ,\tag{8}
$$

where $r ( \cdot )$ is a candidate-action scoring rule, $v ( \cdot )$ is a terminal state-value estimator, K is the number of actions retained for rollouts, d is the rollout depth, n is the number of generated scenarios, and $\tau$ is the surprise threshold to trigger belief refresh. Both r and v are compact, human-readable functions rather than neural networks. Their structures, together with the planning parameters, are discovered through meta-evolution introduced in Section 3.4.

Candidate screening. For each feasible action $\begin{array} { r l } { a _ { t } } & { { } \in } \end{array}$ $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } _ { t } )$ , AWMP computes a belief-conditioned score from its deterministic afterstate $\sigma _ { t } = T ( s _ { t } , a _ { t } )$ and the belief $b _ { t } .$ and retains the K best-ranked actions:

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } _ { t } ^ { \mathrm { T o p K } } = \mathrm { T o p K } _ { a _ { t } \in \boldsymbol { \mathcal { A } } ( s _ { t } ) } \boldsymbol { r } \big ( \boldsymbol { \sigma } _ { t } , b _ { t } \big ) . } \end{array}\tag{9}
$$

This screening can be viewed as a PDR conditioned on the afterstate transition and the arrival belief. It serves as a coarsegrained filter that concentrates the subsequent simulation budget on promising candidates rather than rolling out the complete feasible action set.

Scenario generation. Rollout simulation requires concrete future arrivals, whereas $b _ { t }$ provides only aggregate early–late arrival counts. AWMP bridges this gap through a fixed, non-learned scenario generator. Unlike anticipatory optimization methods such as SAA and MSA, which sample scenarios from a prespecified stochastic model, the generator is fully driven by the LLM-forecast belief inferred from the observed arrival history online:

$$
\mathcal { \widetilde Z } _ { t } ^ { ( m ) } = \Gamma \left( b _ { t } , h _ { t } , \omega _ { m } \right) , \quad m = 1 , \dots , n ,\tag{10}
$$

where $\omega _ { m }$ is the random seed for scenario $m .$ . The generator allocates the forecast arrivals to the corresponding early and late intervals, samples release times within these intervals, and draws the remainingjob attributes from a reference distribution fixed before evaluation. Thus, the LLM-forecast belief determines temporal arrival pressure, while Γ supplies the remaining details required to instantiate arrival streams for simulation. The scenario generator is trivial, neither learned nor modified during meta-evolution or evaluation.

Short-rollout evaluation. Each retained action $\begin{array} { r l } { a _ { t } } & { { } \in \mathbf { \Gamma } } \end{array}$ $\mathcal { A } _ { t } ^ { \mathrm { T o p K } }$ is evaluated under all n generated scenarios. The first simulated dispatch is fixed to $a _ { t } .$ , and subsequent dispatches are selected by the scoring rule r. Given the arrival stream $\widetilde { \mathcal { Z } } _ { t } ^ { ( m ) }$ , every transition is computed exactly through $T$ and $g .$ Note that all rollouts can share the cached root belief $b _ { t } \colon$ once simulated arrivals are realized, their counts can be subtracted from the corresponding intervals to obtain a scenario-specific residual belief. This bookkeeping is fully deterministic and requires no additional LLM call.

Let $s _ { t + d } ^ { ( m , a _ { t } ) }$ denote the state reached after a d-step rollout of candidate $a _ { t }$ under scenario m. The rollout evaluation of $a _ { t }$ is calculated as the objective cost accumulated during the rollout plus the terminal value estimate $v ( s _ { t + d } ^ { ( m , a _ { t } ) } , b _ { t , d } ^ { ( m ) } )$ where $b _ { t , d } ^ { ( m ) }$ denotes the residual belief at the terminal simulated state. This yields a lightweight d-step bootstrap without requiring a long-horizon rollout or repeated optimization. Finally, the action with the best evaluation averaged across the n scenarios is selected for dispatching.

Intra- and inter-step belief reuse. As discussed above, the cached root belief $b _ { t }$ can be shared across all rollouts within a decision step. Reusing it across the K retained actions, n scenarios, and d rollout steps reduces the number of LLM calls from $O ( K n d )$ to at most one per decision step.

Moreover, we note that consecutive decision steps often fall within the same prediction horizon, making it unnecessary to query the LLM-based exogenous predictor at every step. AWMP therefore applies a similar belief reuse principle that carries the most recent belief forward by subtracting newly observed arrivals from the corresponding early or late count. A new forecast is triggered only when the discrepancy between the cumulative observed arrivals and those implied by the cached belief exceeds the surprise threshold τ, or when the cached prediction horizon no longer covers the planner’s lookahead. Intra-step sharing is exact since all rollouts are conditioned on the same observed history, whereas inter-step reuse is an intentional approximation that trades forecast freshness for fewer LLM calls. Section 4.4 reports the resulting reduction in LLM queries.

## 3.4 Meta-Evolution of Planning Strategy

Beyond forecast accuracy, final scheduling performance depends critically on the planning strategy Θ that translates the forecast belief into specific dispatching decisions. Manually specifying such a strategy is dificult and prone to brittle performance. We therefore formulate the discovery of planning strategies as an LLM-driven meta-evolution loop that automatically evolves strategies to efectively leverage arrival beliefs for producing better dispatching decisions.

Meta-evolution maintains a population of strategies and produces new candidates through selection, crossover, and multi-point mutation, with restart-on-stagnation to preserve diversity. In each iteration, a proposer LLM revises the scoring rule r(·), value estimator $v ( \cdot )$ , rollout parameters $K , d , n ,$ and refresh threshold τ within predefined safety constraints (we force $K \leq 6 , d \leq 3 .$ , and $n \leq 8$ to avoid unrestricted search) and a structural schema to keep the resulting strategies compact, executable, and human-readable. Candidate fitness is evaluated by weighted tardiness on the training instances, with instance-level costs aggregated in the log domain to reduce domination by a few large-cost instances. Within the meta-evolution loop, the proposer LLM modifies only the strategy object Θ. It neither generates arrival forecasts nor alters the forecasting prompt or scenario generator Γ. The LLM-based exogenous predictor remains frozen and supplies belief $b _ { t }$ independently during strategy evaluation.

After meta-evolution, the best strategy is selected and frozen. During final evaluation, the scoring rule, value estimator, rollout parameters, refresh threshold, forecasting prompt, scenario generator, and the LLM-based exogenous predictor all remain fixed, with only the residual-calibration record updated online as new arrivals are observed.

## 4 Experiments

We evaluate AWMP through four questions: 1) Does AWMP improve scheduling performance across diferent shop configurations and arrival distributions? 2) Does residual selfcalibration improve both forecast accuracy and downstream scheduling? 3) How efectively does beliefreuse reduce LLM calls? 4) Can meta-evolution improve the planning strategy and generalize well to held-out instances?

## 4.1 Experiment Setup

Benchmarks. We construct two dynamic scheduling settings from standard static benchmarks: dynamic JSSPs, based on 80 Taillard instances (Taillard 1993), and dynamic FJSPs, based on 15 Brandimarte and 66 Hurink instances (Brandimarte 1993; Hurink, Jurisch, and Thole 1994). For each benchmark instance, we preserve its original shop configuration and processing data (e.g., operation precedence, processing times, and machine eligibility), while jobs are released online according to one of four exogenous arrival distributions: pulse train, late-ramp, cyclic, and step-up. These distributions capture periodic bursts, a monotonically increasing arrival rate, cyclic variation, and a mid-horizon rate increase, respectively, as illustrated in Appendix A. We control the arrival intensity β to scale the scheduling horizon by a factor of five relative to the base benchmarks to avoid trivial instances. Scheduling performance is measured by total weighted tardiness, where jobs completed before their due dates incur no penalty, while unfinished jobs are charged at the end of the scheduling horizon, as detailed in Appendix C.

Baselines. We compare AWMP with two groups of baselines. The first contains eight classical PDRs: WSPT, ATC, shortest processing time (SPT), earliest due date (EDD), most work remaining (MWKR), first-in-first-out (FIFO), minimum slack (MS), and cost over time (COVERT). They re-rank currently feasible operations at every decision step whenever the shop state changes or new job arrivals occur. The second group comprises five optimization-based online schedulers: Myopic, Rolling-horizon, MSA, SAA, and expected-value optimization (EV). Myopic optimizes over currently released jobs only; Rolling-horizon periodically reoptimizes over a moving planning window; MSA selects decisions through multiple-scenario consensus; SAA optimizes a sample-average objective over sampled arrival scenarios; and EV solves a deterministic approximation under expected future arrivals. All five methods use OR-Tools (Perron and Furnon 2023) to solve their subproblems, and are allocated computational budgets at least as large as AWMP. We use Claude Haiku 4 as the frozen LLM for online arrival-belief prediction and Claude Sonnet 4 as the proposer LLM for ofline meta-evolution.

## 4.2 Overall Performance

We evaluate AWMP, i.e., the evolved planning strategy that leverages the belief produced by the LLM-based exogenous predictor, on dynamic JSSPs and FJSPs under four arrival distributions. We report the average objective (total weighted tardiness) over non-trivial instances, defined as those on which at least one of the eight PDRs incurs positive tardiness. Table 1 shows that AWMP achieves the best objective in all eight benchmark-distribution combinations, reducing total weighted tardiness relative to the best PDR by 5.8%– 20.7% on dynamic JSSPs and by 21.9%–46.3% on dynamic FJSPs, and outperforming all five optimization-based baselines in every setting. These gains require only a handful of LLM queries per episode (Section 4.4), giving runtimes comparable to or below Rolling-horizon and approximately one order of magnitude below other anticipatory baselines.

Table 1: Average objective (total weighted tardiness) on dynamic JSSP and FJSP instances under four arrival distributions (lower is better). Best PDR reports the best result among the eight baseline PDRs.
<table><tr><td colspan="2">Benchmark Distribution</td><td>Myopic</td><td>Rolling</td><td>MSA</td><td>SAA</td><td>EV</td><td>Best PDR</td><td>AWMP (Ours)</td></tr><tr><td rowspan="4">Dynamic JSSP</td><td>Pulse train</td><td>79931</td><td>67930</td><td>66509</td><td>66568</td><td>66178</td><td>32923</td><td>29238 (-11.2%)</td></tr><tr><td>Late-ramp</td><td>70584</td><td>59654</td><td>58544</td><td>58957</td><td>58380</td><td>34841</td><td>27619 (-20.7%)</td></tr><tr><td>Cyclic</td><td>82708</td><td>70243</td><td>69530</td><td>69561</td><td>68841</td><td>34486</td><td>32483 (-5.8%)</td></tr><tr><td>Step-up</td><td>78631</td><td>66590</td><td>65719</td><td>65885</td><td>65388</td><td>32686</td><td>29630 (-9.4%)</td></tr><tr><td rowspan="4">Dynamic FJSP</td><td>Pulse train</td><td>836</td><td>369</td><td>384</td><td>342</td><td>315</td><td>212</td><td>163 (-23.1%)</td></tr><tr><td>Late-ramp</td><td>1543</td><td>559</td><td>912</td><td>591</td><td>561</td><td>456</td><td>258 (-43.4%)</td></tr><tr><td>Cyclic</td><td>1164</td><td>383</td><td>512</td><td>395</td><td>372</td><td>300</td><td>161 (-46.3%)</td></tr><tr><td>Step-up</td><td>1130</td><td>386</td><td>582</td><td>528</td><td>497</td><td>265</td><td>207 (-21.9%)</td></tr></table>

Importantly, unlike neural scheduling methods that typically require distribution-specific retraining, AWMP uses a single evolved strategy that is frozen and evaluated across all four arrival distributions without per-distribution retuning. Its improvements therefore extend beyond the arrival process encountered during strategy evolution, covering bursty, increasing, cyclic, and step-changing arrival patterns. This suggests that the combination of history-based arrival beliefs and the evolved afterstate planner generalizes across diferent forms of arrival dynamics.

We also observe that the optimization-based baselines do not consistently outperform classical PDRs. In dynamic scheduling, anticipatory performance depends on both optimization strength and how well the assumed future workload reflects the arrivals that are eventually realized. Methods such as SAA, MSA, and EV optimize decisions with respect to sampled or expected future arrivals; when these assumptions deviate from the realized temporal arrival pattern, stronger optimization may still produce decisions that are suboptimal for the actual future workload. This efect is particularly consequential under weighted tardiness, where an early decision can occupy critical capacity and lead to substantial downstream penalties.

In contrast, classical PDRs make no explicit assumptions about future arrivals and continuously re-rank available operations as the shop state changes, thereby avoiding errors induced by an inaccurate explicit future model. AWMP combines the responsiveness of online dispatching with explicit anticipation by inferring future arrival pressure from the observed history and incorporating it into short-horizon planning.

## 4.3 Efect of Residual Self-Calibration

We evaluate the forecast accuracy and the efect of residual self-calibration on dynamic JSSP instances under a sinusoidally drifting arrival intensity. The near-term forecast correlation increases from r = 0.12 using only the observed arrival history to r = 0.93 after incorporating recent prediction–outcome residuals (Fig. 3). We compute the latter from 342 prediction–outcome pairs sampled from the same dynamic JSSP instances and arrival distribution, after centering the predictions and outcomes within each instance. This closed-loop calibration enables the arrival belief to more closely track the evolving arrival dynamics without parameter updates.

The improved forecast also leads to better scheduling performance. Under the same planning strategy, replacing the raw belief with the calibrated belief reduces the objective from 6771 to 4011, corresponding to a 41% reduction. The sampled prediction–outcome pairs are obtained from a separate diagnostic run and are not included in the main evaluation results.

## 4.4 LLM Call Eficiency

We compare three querying regimes under the same running setting: 1) naive per-rollout querying, 2) w/ intra-step belief caching, and 3) w/ inter-step reuse and surprise-triggered refresh. Naive querying requires 7191 LLM calls per episode. Intra-step belief caching reduces this count to 393, leading to an 18× reduction, while incorporating inter-step reuse and surprise-triggered refresh further reduces it to only 18, approximately 400× fewer calls than naive querying. The reduction comes from sharing the same root belief across candidate actions and rollout scenarios within a decision step, thereby avoiding repeated forecasts for identical observed histories. Across consecutive decision steps, the cached belief is carried forward and refreshed only when realized arrivals deviate suficiently from the current belief or when the prediction horizon is no longer adequate. At the measured latency of approximately 0.84 s per call, the naive regime requires well over an hour per episode due to the latency of LLM calls, whereas AWMP with inter- and intra-step reuse finishes its queries within seconds. These results show that belief reuse substantially amortizes the cost of online LLM forecasting for repeated dispatching decisions, while preserving the same belief-conditioned planning procedure.

## 4.5 Meta-evolution

Figure 4 tracks the meta-evolution process under a strict train-test split. We use TA01 and TA02 exclusively for strategy evolution, where the scoring rule, value estimator, rollout parameters, and refresh threshold are iteratively refined according to the training objective. After meta-evolution terminates, we select the single best-performing strategy, freeze the resulting planning strategy, and evaluate it directly on the disjoint held-out instances TA03 and TA04 without any further strategy adaptation or instance-specific tuning. As shown in Figure 4, the objective decreases substantially over the course of meta-evolution, indicating that the search process progressively discovers more efective ways of translating the arrival belief into dispatching decisions. Relative to the initial seed strategy, the evolved strategy reduces the objective by 57% on the training instances and by 62% on the held-out instances. Notably, the improvement on the unseen instances is comparable to that observed on the training instances. This suggests that meta-evolution does not merely fit the few instances used during strategy search, but discovers planning rules that remain efective beyond the instances used for evolution. Since the final planning strategy is frozen before evaluation, the gain on TA03 and TA04 reflects the transferability of the evolved belief-conditioned decision logic rather than further strategy adaptation at test time. Overall, these results provide evidence that a compact strategy evolved from only a small number of training instances can remain efective on previously unseen dynamic scheduling instances.

![](images/5e8b236852105e3a10bfd127070e84ddc95638120c9a487f36c0cb4fa2111584.jpg)  
(a)

![](images/751294c4af838c6760d4686e9cafe33e717f521eeb02631dc7713422191c6d84.jpg)  
(b)

Figure 3: Forecast correlation (a) without and (b) with residual self-calibration, increasing from $r = 0 . 1 2$ to $r = 0 . 9 3$  
![](images/a4cba56b4476e03b2bbc52ec88ddd8971ada4ccc712b0b99a3a36510671f910e.jpg)  
(a)

![](images/bdb75a28d65e2065fa9385b834e75ac6e84663b1c783e380ede86e61134d9fa8.jpg)  
(b)  
Figure 4: (a) Meta-evolution trajectory; (b) seed vs. evolved strategy performance on training and held-out instances.

## 5 Conclusion

We introduced AWMP, an afterstate world-model planning framework for dynamic scheduling under online arrivals. The afterstate decomposition separates known deterministic shop dynamics from the exogenous uncertain arrival stream, forming a structured world model in which a pretrained LLM models the exogenous component while the scheduling simulator retains the known endogenous transitions exactly. Residual self-calibration adapts the belief forecast online under nonstationary arrivals, while a meta-evolved, belief-conditioned planner converts it into dispatching decisions through candidate screening and short rollouts under a fixed, non-learned scenario generator. Experiments show that combining a lowdimensional LLM-based forecast of exogenous uncertainty with exact scheduling dynamics supports efective anticipatory scheduling without learning the complete shop transition or training a task-specific neural dispatching policy. Future work includes enriching the arrival belief with more job-level information and evaluating AWMP on real-world and more diverse arrival processes. We will also explore more eficient forecasting models to further reduce online inference latency. More broadly, the structured world-model framework may extend to robust optimization and stochastic programming settings.

Our study has several limitations. First, AWMP represents future arrivals through a compact early–late workload belief rather than detailed job-level predictions. This design keeps the forecasting problem low-dimensional and supports stable online calibration, avoiding the dificulty of predicting the complete future arrival stream. However, richer representations that additionally capture job attributes, such as processing routes, priorities, or machine requirements, may provide more informative guidance when future workloads are highly heterogeneous. Second, our experiments construct dynamic instances from standard JSSP and FJSP benchmarks under four controlled arrival patterns. These patterns cover several representative forms of temporal variation and allow controlled evaluation across diferent arrival dynamics. However, real manufacturing systems may exhibit more diverse arrival behaviors and operational conditions, and evaluation on real scheduling traces would provide a broader assessment of the framework. Finally, AWMP uses a pretrained LLM for online belief updates. Belief reuse and surprise-triggered refresh substantially limit the associated inference overhead. Nevertheless, the runtime of belief updates is still afected by the latency of the underlying model, and more eficient forecasting models could further improve online deployment.

## References

Antonoglou, I.; Schrittwieser, J.; Ozair, S.; Hubert, T.; and Silver, D. 2022. Planning in stochastic environments with a learned model. In International Conference on Learning Representations (ICLR).

Bent, R. W.; and Van Hentenryck, P. 2004. Scenario-based planning for partially dynamic vehicle routing with stochastic customers. Operations Research, 52(6): 977–987.

Brandimarte, P. 1993. Routing and scheduling in a flexible job shop by tabu search. Annals of Operations Research, 41(3): 157–183.

Chang, J.; Yu, D.; Hu, Y.; He, W.; and Yu, H. 2022. Deep reinforcement learning for dynamic flexible job shop scheduling with random job arrival. Processes, 10(4): 760.

Hafner, D.; Lillicrap, T. P.; Norouzi, M.; and Ba, J. 2021. Mastering Atari with discrete world models. In International Conference on Learning Representations (ICLR).

Hansen, N.; Wang, X.; and Su, H. 2022. Temporal diference learning for model predictive control. In International Conference on Machine Learning (ICML).

Haupt, R. 1989. A survey of priority rule-based scheduling. OR Spektrum, 11(1): 3–16.

Hurink, J.; Jurisch, B.; and Thole, M. 1994. Tabu search for the job-shop scheduling problem with multi-purpose machines. OR Spektrum, 15(4): 205–215.

Kleywegt, A. J.; Shapiro, A.; and Homem-de Mello, T. 2002. The sample average approximation method for stochastic discrete optimization. SIAM Journal on Optimization, 12(2): 479–502.

Kool, W.; van Hoof, H.; and Welling, M. 2019. Attention, learn to solve routing problems! In International Conference on Learning Representations (ICLR).

Li, S.; Ouyang, W.; Ma, Y.; and Wu, C. 2025. Learningguided rolling horizon optimization for long-horizon flexible job-shop scheduling. In International Conference on Learning Representations (ICLR).

Liu, F.; Tong, X.; Yuan, M.; Lin, X.; Luo, F.; Wang, Z.; Lu, Z.; and Zhang, Q. 2024a. Evolution of heuristics: Towards eficient automatic algorithm design using large language model. In International Conference on Machine Learning (ICML).

Liu, R.; Piplani, R.; and Toro, C. 2022. Deep reinforcement learning for dynamic scheduling of a flexible job shop. International Journal of Production Research, 60(13): 4049– 4069.

Liu, R.; Piplani, R.; and Toro, C. 2023. A deep multi-agent reinforcement learning approach to solve dynamic job shop scheduling problem. Computers & Operations Research, 159: 106294.

Liu, Z.; Mao, H.; Sa, G.; Liu, H.; and Tan, J. 2024b. Dynamic job-shop scheduling using graph reinforcement learning with auxiliary strategy. Journal of Manufacturing Systems, 73: 1– 18.

Luo, S. 2020. Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning. Applied Soft Computing, 91: 106208.

Ouelhadj, D.; and Petrovic, S. 2009. A survey of dynamic scheduling in manufacturing systems. Journal of Scheduling, 12(4): 417–431.

Panwalkar, S. S.; and Iskander, W. 1977. A survey of scheduling rules. Operations Research, 25(1): 45–61.

Perron, L.; and Furnon, V. 2023. Operations research tools developed at Google (OR-Tools). https://developers.google. com/optimization.

Pinedo, M.; and Hadavi, K. 1992. Scheduling: theory, algorithms and systems development. In Operations research proceedings 1991: Papers of the 20th annual meeting/vorträge der 20. Jahrestagung, 35–42. Springer.

Powell, W. B. 2007. Approximate Dynamic Programming: Solving the Curses of Dimensionality, volume 703. John Wiley & Sons.

Romera-Paredes, B.; Barekatain, M.; Novikov, A.; Balog, M.; Kumar, M. P.; Dupont, E.; Ruiz, F. J. R.; Ellenberg,

J. S.; Wang, P.; Fawzi, O.; Kohli, P.; and Fawzi, A. 2024. Mathematical discoveries from program search with large language models. Nature, 625(7995): 468–475.

Schrittwieser, J.; Antonoglou, I.; Hubert, T.; Simonyan, K.; Sifre, L.; Schmitt, S.; Guez, A.; Lockhart, E.; Hassabis, D.; Graepel, T.; Lillicrap, T. P.; and Silver, D. 2020. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588(7839): 604–609.

Smit, I. G.; Wu, Y.; Troubil, P.; Zhang, Y.; and Nuijten, W. P. M. 2025a. Neural combinatorial optimization for stochastic flexible job shop scheduling problems. In AAAI Conference on Artificial Intelligence (AAAI).

Smit, I. G.; Zhou, J.; Reijnen, R.; Wu, Y.; Chen, J.; Zhang, C.; Bukhsh, Z.; Zhang, Y.; and Nuijten, W. 2025b. Graph neural networks for job shop scheduling problems: A survey. Computers & Operations Research, 176: 106914.

Taillard, E. 1993. Benchmarks for basic scheduling problems. European Journal of Operational Research, 64(2): 278–285.

Tang, Y.; Zakwan, M.; Balta, E.; Lygeros, J.; and Rupenyan, A. 2026. Deep reinforcement learning for flexible job shop scheduling with random job arrivals. arXiv preprint arXiv:2605.22773.

Vieira, G. E.; Herrmann, J. W.; and Lin, E. 2003. Rescheduling manufacturing systems: a framework of strategies, policies, and methods. Journal ofScheduling, 6(1): 39–62.

Ye, H.; Wang, J.; Cao, Z.; Berto, F.; Hua, C.; Kim, H.; Park, J.; and Song, G. 2024. ReEvo: Large language models as hyper-heuristics with reflective evolution. In Advances in Neural Information Processing Systems (NeurIPS).

Yu, H.; Gu, W.; Tang, N.; and Guo, Z. 2026. A deep reinforcement learning approach for dynamic job-shop scheduling problem considering time variable and new job arrivals. Computers & Operations Research, 185: 107263.

Zhang, C.; Song, W.; Cao, Z.; Zhang, J.; Tan, P. S.; and Xu, C. 2020. Learning to dispatch for job shop scheduling via deep reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS).

## A Arrival Distributions

We construct each dynamic instance by assigning release times to the jobs of its underlying static benchmark. Job releases are restricted to an arrival window of length $H _ { \mathrm { a r r } } =$ $0 . 6 H$ , where H is the episode horizon. The first job is available at $t = 0 ,$ , while the release times of the remaining jobs are sampled from one of four distribution families. The intensity parameter $\beta$ scales the peak arrival rate. Each family exhibits a temporal structure that can, in principle, be inferred from the observed arrival history. Fig. 5 shows the empirical arrival-intensity profiles aggregated over 200 random seeds.

• Pulse train. Arrivals are concentrated around $K _ { \mathrm { p u l s e } } = 4$ evenly spaced burst centers, with Gaussian temporal jitter scaled by $0 . 0 2 H _ { \mathrm { a r r } }$ . This produces a sequence of short, periodic workload surges.

• Late-ramp. Arrival times are generated as:

$$
t _ { j } = \sqrt { U _ { j } } H _ { \mathrm { a r r } } , \mathrm { w h e r e } U _ { j } \sim \mathrm { U n i f } ( 0 , 1 ) ,
$$

resulting in an arrival rate that increases monotonically over the window. The expected normalized release time is $2 / 3 ,$ , producing a back-loaded workload.

• Cyclic. The arrival intensity follows two sinusoidal cycles over the arrival window:

$$
\lambda ( t ) = \lambda _ { 0 } \left( 1 + 0 . 9 \sin { \left( 4 \pi t / H _ { \mathrm { a r r } } \right) } \right) .
$$

This distribution alternates between recurring high- and low-intensity periods.

• Step-up. One third of the jobs are released uniformly during the first half of the window, while the remaining two thirds are released during the second half. Consequently, the expected arrival rate doubles at the midpoint.

![](images/4c2e199d0c67501616b613f407f056f53064629c5cdf966bb4ae653624fc1fe8.jpg)  
Figure 5: Empirical arrival-rate profile of each distribution family over the arrival window (200 seeds, $\beta = 1 . 5 )$ .

## B World-Model Interface

World-model input and output. Whenever a belief refresh is triggered at decision step t, the world model receives the observed arrival history $h _ { t } .$ , including the current physical time, prediction horizon, number of jobs released so far, and their release-time log. It returns the arrival belief:

$$
b _ { t } = ( \widehat { n } _ { t } ^ { \mathrm { e a r l y } } , \widehat { n } _ { t } ^ { \mathrm { l a t e } } ) ,
$$

where $\widehat { n } _ { t } ^ { \mathrm { e a r l y } }$ and $\widehat { n } _ { t } ^ { \mathrm { l a t e } }$ are nonnegative integer estimates of the numbers of newjobs arriving in the early and late portions of the prediction horizon, respectively. The LLM predicts only temporally aggregated arrival pressure; it does not predict individual job identities, exact release times, processing routes, or other job attributes. The concrete release times and job attributes required for rollout simulation are instantiated by the fixed scenario generator described in Section 3.3.

Prompt. Each world-model query combines a fixed forecasting instruction, the observed arrival history, the residualcalibration record, and a predefined JSON output schema. The model is instructed to return only

$$
\{ { \bf " e a r l y " : e } , \quad { \bf " } \mathrm { l a t e " : } \quad \mathrm { l } \} ,
$$

where e and l are the predicted arrival counts. The instruction, output schema, and all other prompt components remain fixed during meta-evolution and final evaluation.

## C Objective

The objective is total weighted tardiness:

$$
\mathrm { C o s t } = \sum _ { j } w _ { j } \operatorname* { m a x } \left( 0 , \widetilde { C } _ { j } - d _ { j } \right) ,
$$

where

$$
d _ { j } = \rho _ { j } + \mu p _ { j } , \quad w _ { j } = \frac { p _ { j } } { \bar { p } } .
$$

Here, $\rho _ { j }$ is the release time of job $j , p _ { j }$ is its total processing workload, $\bar { p }$ is the mean workload across all jobs in the instance, and $\mu$ controls due-date tightness. For a job completed within the episode, ${ \widetilde { C } } _ { j }$ is the completion time of its final operation. An unfinishedjob is assigned $\begin{array} { r } { \widetilde { C } _ { j } = H . } \end{array}$ , where H is the episode horizon, ensuring that unfinished work contributes a terminal tardiness penalty rather than being omitted from the objective. The hinge term max $0 , \widetilde { C } _ { j } - d _ { j } )$ creates a due-date threshold: a job incurs no penalty before its due date and a linearly increasing penalty thereafter. The due date accounts for thejob’s own processing workload through $\mu p _ { j }$ while $w _ { j } = p _ { j } / \bar { p }$ assigns proportionally greater importance to processing-intensive jobs.