# FGSS-BO: Feedback-Guided Hypothesis Management for LLM-Based Bayesian Optimization

Hao Chen<sup>1</sup>, Ziyang Ling<sup>1</sup>, Ronald X. Xu<sup>1,2,\*</sup>, Mingzhai Sun<sup>1,\*</sup>

<sup>1</sup>School of Biomedical Engineering, <sup>2</sup>School of Engineering Science

University of Science and Technology of China, Suzhou, China

{chenhao0873,zaneling}@mail.ustc.edu.cn

Corresponding authors: xux@ustc.edu.cn, mingzhai@ustc.edu.cn

## Abstract

Large language model (LLM)-guided Bayesian optimization (BO) uses past observations to guide search, but proposed directions and their revision can remain implicit. We propose Feedback-Guided Search State (FGSS), a structured hypothesis-management interface connecting language-based guidance to observed evidence, executable parameter interventions, and comparative support updates. An LLM reflector proposes and revises hypotheses; a deterministic curator validates them and maintains identities and test records. The evolving state guides candidate construction and priorities. Instantiated on LLAMBO, FGSS-BO retains its surrogate and default expected improvement selection, with a bounded testing policy. On two retrospective protocol/media tasks with nearest-neighbor response oracles, it attains higher sample mean final best and search AUC than the matched LLAMBO baseline and evaluated non-LLM baselines. Prose-reflection and EI-only controls show mixed component effects. Six direct-training hyperparameteroptimization tasks show task-dependent transfer, strongest on diabetes SVM; exact-grid and reaction controls examine other settings. FGSS exposes hypotheses, their execution, and crossround revisions for inspection while supporting search gains in selected settings.

## 1 Introduction

Bayesian optimization (BO) addresses black-box search when evaluations are expensive, including hyperparameter tuning and scientific experimentation (Jones et al., 1998; Shahriari et al., 2016; Frazier, 2018). Large language models (LLMs) have been incorporated into BO to interpret task descriptions, sample candidates, and predict outcomes from historical examples (Liu et al., 2024). These capabilities offer a flexible interface to experimental parameters whose meanings and admissible values differ across tasks.

LLAMBO maintains configuration–outcome observations and conditions its LLM calls on that history (Liu et al., 2024). It does not separately maintain identified search hypotheses with evidence references and later test outcomes. This distinction matters under a limited evaluation budget: a suggestion to reduce one parameter may produce a candidate that also changes other coordinates, so its outcome need not directly test the suggested direction. How can an LLM optimizer turn such suggestions into revisable guidance linked to the decisions they influence?

Inspired by reflection and evolving-memory approaches (Shinn et al., 2023; Suzgun et al., 2026; Zhang et al., 2026a), we propose Feedback-Guided Search State (FGSS) for structured, auditable hypothesis management in BO. Its continually updated guidance consists of search hypotheses with evidence references, an anchor, a parameter intervention, and an expected outcome direction. An LLM reflector proposes and revises these hypotheses; a deterministic curator validates operations and maintains identities and lifecycle records. Accepted hypotheses guide candidate construction and priorities, while eligible matched tests update comparative support for their primary hypothesis. The base surrogate continues to use the observation history.

We instantiate FGSS on LLAMBO and call the resulting optimizer FGSS-BO. LLAMBO separates candidate sampling, surrogate prediction, and acquisition selection, allowing us to introduce hypothesis management into the search loop. FGSS-BO and its matched baseline use the same feasibility-preserving candidate interface, while retaining the LLAMBO surrogate and default EI selection. FGSS-BO additionally permits a bounded exception for hypothesis-test selection.

We assess search performance alongside this hypothesis-management interface. On retrospective iPSC-RPE protocol search and C2C12 culturemedia search (Kanda et al., 2022; Cosenza et al., 2022), FGSS-BO attains higher sample mean final best and search AUC than its matched baseline. Prose-reflection and testing ablations show mixed component effects. Six preselected hyperparameter-optimization (HPO) tasks use direct model training, with the clearest gains on diabetes SVM; exact-grid, reaction, and chromatography controls examine other settings. Our contributions are:

• A structured search-hypothesis representation linking evidence, parameter interventions, and revision and test records through persistent identities.

• A search-control mechanism translating LLMproposed hypotheses into feasible candidate priorities and matched tests through deterministic curation and comparative support updates.

• Paired offline protocol/media comparisons with LLM and non-LLM baselines, component ablations, and direct-training HPO transfer checks, supplemented by candidate and oracle diagnostics.

## 2 Related Work

Bayesian optimization for scientific discovery. BO combines surrogate updating with acquisitionbased query selection (Jones et al., 1998; Shahriari et al., 2016; Frazier, 2018). Applications include hyperparameter optimization (Snoek et al., 2012), robotic iPSC-RPE protocol search, and culturemedia development (Kanda et al., 2022; Narayanan et al., 2025). LLM-guided methods add a language interface to parameter semantics and experimental knowledge within this numerical search loop.

LLM-guided optimization. LLM-guided optimizers complement statistical optimization with contextual reasoning and pretrained knowledge. LLAMBO frames BO components in natural language and uses LLMs for warm-starting, candidate sampling, and surrogate prediction (Liu et al., 2024); related systems address genetic perturbation design, scientific optimization, and molecular lead optimization (Roohani et al., 2025; Yang et al., 2026; Zhang et al., 2026b). Gupta et al. (2025) use randomized-feedback controls to diagnose outcome sensitivity in BioDiscoveryAgent. Whereas these controls measure dependence on feedback content, FGSS addresses how an optimizer represents and revises search hypotheses and turns them into search actions.

Feedback, memory, and reflection in LLM agents. Self-Refine iteratively revises outputs using self-feedback, while Reflexion retains reflective experience across trials (Madaan et al., 2023; Shinn et al., 2023). Dynamic Cheatsheet accumulates reusable guidance, and Agentic Context Engineering (ACE) organizes evolving context through reflection and curation (Suzgun et al., 2026; Zhang et al., 2026a). FGSS adopts these memory and curation principles for numerical BO. Its hypothesis records specify an observed anchor, a parameter intervention, and an expected outcome direction, making the guidance executable. Candidate lineage connects eligible numerical comparisons to the corresponding hypothesis, while the surrogate retains the observation history. Our prose-summary control tests a simpler continuously updated memory interface; FGSS additionally maintains individual hypotheses and their test records.

## 3 Background and Problem Setup

Bayesian optimization and LLM-guided optimization. We consider maximization of an expensive black-box objective $f : \mathcal { X }  \mathbb { R }$ under a limited evaluation budget. Starting from K initial observations $\mathcal { D } _ { 0 }$ , after t subsequent search evaluations the history is

$$
\mathcal { D } _ { t } = \mathcal { D } _ { 0 } \cup \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { t } , \quad y _ { i } = f ( x _ { i } ) .\tag{1}
$$

BO then selects an unevaluated configuration by maximizing an acquisition criterion over the remaining search space. LLAMBO augments this loop with LLM-based candidate sampling and surrogate prediction while conditioning prompts on task descriptions and previous evaluations (Liu et al., 2024).

Explicit search-hypothesis management. We augment $\mathcal { D } _ { t }$ with a state $S _ { t }$ of search hypotheses, evidence references, revisions, and eligible test outcomes. Unlike the observations themselves, these records specify proposed actions whose guidance can change across rounds.

Optimization settings. The method accepts numerical, ordinal, and categorical variables. The primary retrospective tasks use a historical-table oracle $\widehat { f } _ { \mathrm { N N } }$ , not new laboratory measurements. Exactgrid, emulator/simulator, and direct-training HPO tasks test other interfaces. We present the method for maximization; minimization reverses outcome comparisons and uses the corresponding improvement criterion.

## 4 Method

## 4.1 Framework Overview

Figure 1 summarizes the loop: evidence $e _ { t }$ is extracted from $\mathcal { D } _ { t }$ , the reflector and curator revise $S _ { t }$ , and the state guides candidate set $\mathcal { C } _ { t }$ . For example, a hypothesis may propose reducing one parameter at an observed anchor. An exact intervention changes that parameter alone; comparison with the anchor updates the parent hypothesis, informing later retention, revision, or retirement. Appendix A.4 provides the prompt contract and a recorded state example. The active set is bounded, with archived and retired records retained.

## 4.2 Evidence and Feedback State

The extractor converts recent and global observations into a compact evidence object,

$$
e _ { t } = E ( \mathcal { D } _ { t } ) .\tag{2}
$$

It contains ranked high- and low-performing configurations, recent improvement, stagnation, perparameter coverage, duplicate regions, and local contrasts between nearby configurations with different outcomes. Local contrasts provide an observed basis for proposing or revising hypotheses.

At round $t , S _ { t } ^ { - }$ is the carried state after t search evaluations, with $S _ { 0 } ^ { - } = \emptyset ; S _ { t }$ is its curated revision used for selection. The reflector and curator update it using $\textstyle e _ { t } \colon$

$$
\begin{array} { l } { h _ { t } = R ( e _ { t } , S _ { t } ^ { - } ) , } \\ { S _ { t } = C ( S _ { t } ^ { - } , h _ { t } , e _ { t } ) . } \end{array}\tag{3}
$$

Each hypothesis contains a persistent identifier, parameter scope, anchor configuration or region, intervention, expected outcome direction, evidence references, uncertainty, a heuristic support score, and lifecycle status. Direct tests target one or two variables. The intervention specifies a testable prediction, and its recorded lineage links a later observation to the primary hypothesis. The curator validates additions and revisions, merges duplicates, manages lifecycle, and caps the active set. For main-effect and interaction hypotheses, support and lifecycle updates retain the ID; changed interventions require new cards.

## 4.3 Comparative Support and State Evolution

Each candidate records the hypotheses contributing to its construction or priority. Let $J _ { t } ( x )$ index these contributors within the active state. This lineage is distinct from direct-test attribution:

$$
A _ { t } ( x ) = \{ H _ { j } : j \in J _ { t } ( x ) \} .\tag{4}
$$

Only an eligible direct test updates its primary hypothesis, with $m _ { j , t } \in \{ 0 , 1 \}$ and $\begin{array} { r } { \sum _ { j } m _ { j , t } \ \leq \ 1 } \end{array}$ Its evidence weight $w _ { j , x } \in [ 0 , 1 ]$ discounts uncertainty, imperfect matching, and control reuse. For control outcome $y _ { j , t } ^ { \mathrm { c t r l } }$ , define, for maximization,

$$
\begin{array} { r l } & { \delta _ { j , t } = y _ { t + 1 } - y _ { j , t } ^ { \mathrm { c t r l } } , } \\ & { u _ { j , t } = \eta _ { + } \mathbf { 1 } [ \delta _ { j , t } > \epsilon _ { j , t } ] } \\ & { \qquad - \eta _ { - } \mathbf { 1 } [ \delta _ { j , t } < - \epsilon _ { j , t } ] , } \\ & { \qquad s _ { j , t } = m _ { j , t } w _ { j , x _ { t + 1 } } u _ { j , t } , } \\ & { q _ { j , t + 1 } = \mathrm { c l i p } ( \bar { q } _ { j , t } + s _ { j , t } , 0 , 1 ) . } \end{array}\tag{5}
$$

Here $\bar { q } _ { j , t } \in [ 0 , 1 ]$ is the curated heuristic support score in $S _ { t }$ , including initialization of new cards; it is not a calibrated probability. Post-evaluation records, including $q _ { j , t + 1 }$ , enter $S _ { t + 1 } ^ { - } .$ . The increments $\eta _ { + } , \eta _ { - }$ are fixed, and $\epsilon _ { j , t }$ combines an empirical effect threshold with a numerical floor, not a significance test. Staleness subtracts $\lambda ~ \geq ~ 0$ once per verification epoch, separately from outcome updates. Minimization reverses $\delta _ { j , t }$ . Invalid matches receive no outcome update; globalincumbent progress is tracked separately. Openspace tests hold non-target variables fixed; tablebacked tests allow bounded nuisance differences and receive discounted weight. Thus $\delta _ { j , t }$ <sub>t</sub> supplies comparative support under the oracle, not an identified experimental causal effect. Test counts, control reuse, and failures govern subsequent weakening or retirement (Appendix A.1).

## 4.4 Regime Control and Candidate Generation

The reflector proposes an exploration–exploitation regime, while a deterministic controller constrains the final decision:

$$
\begin{array} { r } { \widetilde { r } _ { t } = R _ { \mathrm { r e g i m e } } ( e _ { t } , S _ { t } ^ { - } ) , } \\ { r _ { t } = G ( \widetilde { r } _ { t } , b _ { t } , g _ { t } , c _ { t } ) . } \end{array}\tag{6}
$$

Here $b _ { t }$ is the current best outcome, $g _ { t }$ describes recent progress or stagnation, and $c _ { t }$ summarizes coverage. The controller constrains pure exploitation, preserves diversity, and triggers recovery under repeated low-yield search. During recovery,

![](images/b061d543c90fa6f1107095e45223d3429dd757dfde46fe8d086b547692fe0db6.jpg)  
Figure 1: Overview of FGSS-BO. The extractor, LLM reflector, and deterministic curator maintain evidence-linked hypotheses; a regime gate controls candidate allocation. LLAMBO retains its history-conditioned surrogate and default EI selection, with a bounded exception for eligible matched tests. Solid arrows show current-round flow; dashed arrows show cross-round memory and eligible comparative support updates.

FGSS-BO proposes evidence-linked alternatives to failed directions, for example an interaction hypothesis under which a single-parameter effect depends on another parameter.

For typed parameter spaces, FGSS-BO samples feasible, previously unevaluated configurations and biases their priority using $S _ { t }$ and $r _ { t } .$

$$
\mathcal { C } _ { t } = G _ { \mathrm { c a n d } } ( \mathcal { X } \setminus \mathcal { X } _ { t } ; S _ { t } , r _ { t } ) .\tag{7}
$$

Here $\mathcal { X } _ { t }$ is the set of configurations evaluated in $\mathcal { D } _ { t }$ . Domain specifications enforce variable types and admissible ranges. The generator allocates candidates among global diversity, local refinement, hypothesis testing near anchors, and recovery. Hypothesis scopes, directions, and anchor contexts guide construction and priority; the global branch preserves proposals outside active hypotheses.

## 4.5 Integration with LLAMBO

FGSS-BO retains the LLAMBO surrogate and its expected improvement criterion. With LLMpredicted mean $\mu _ { t } ( x )$ and standard deviation $\sigma _ { t } ( x )$ floored at $1 0 ^ { - 5 }$ in the inherited implementation, EI over $\mathcal { C } _ { t }$ is

$$
\mathrm { E I } _ { t } ( \boldsymbol { x } ) = \Delta _ { t } ( \boldsymbol { x } ) \Phi ( \boldsymbol { z } _ { t } ) + \sigma _ { t } ( \boldsymbol { x } ) \phi ( \boldsymbol { z } _ { t } ) ,
$$

$$
z _ { t } = \frac { \Delta _ { t } ( x ) } { \sigma _ { t } ( x ) } ,\tag{8}
$$

(9)

where Φ and ϕ are the standard normal CDF and density, and $\Delta _ { t } ( x ) = \mu _ { t } ( x ) - b _ { t }$ for maximization. Let ${ \mathcal { C } } _ { t } ^ { \mathrm { t e s t } } \subseteq { \mathcal { C } } _ { t }$ contain eligible matched interven-

tions, and define

$$
\begin{array} { r l } & { x _ { t + 1 } ^ { \mathrm { E I } } = \arg \underset { x \in \mathcal { C } _ { t } } { \operatorname* { m a x } } \mathrm { E I } _ { t } ( x ) , } \\ & { x _ { t + 1 } ^ { \mathrm { t e s t } } = \arg \underset { x \in \mathcal { C } _ { t } ^ { \mathrm { t e s t } } } { \operatorname* { m a x } } P _ { t } ( x ) . } \end{array}
$$

The latter is the highest-priority eligible matched intervention. The deterministic priority $P _ { t }$ is lexicographic: adaptive test priority, EI, supportweighted match quality, challenger status, and distance/index tie-breaks (Appendix A.1). FGSS-BO selects

$$
x _ { t + 1 } = { \left\{ \begin{array} { l l } { x _ { t + 1 } ^ { \mathrm { t e s t } } , } & { a _ { t } = 1 , } \\ { x _ { t + 1 } ^ { \mathrm { E I } } , } & { a _ { t } = 0 , } \end{array} \right. }\tag{10}
$$

where $a _ { t }$ can be one only when an eligible matched test exists and the testing budget and minimum interval permit it. For $T = 2 0$ , the policy has four base test slots and up to two additional challenger slots during stagnation, with a minimum interval of three rounds. These constrain policy overrides, not all direct tests: EI can also select a test candidate, which consumes base-test usage. The bounded exception supplies comparisons linked to specific hypotheses; realized overrides and match quality are reported in Appendix A.2.

The matched LLAMBO baseline shares the candidate interface and surrogate but omits state conditioning and test-policy selection. Experimental Setup specifies the remaining shared settings.

## 5 Experimental Setup

## 5.1 Datasets and Tasks

The primary protocol/media tasks expose configuration–outcome feedback under limited budgets with different variable types.

iPSC-RPE. The task concerns differentiation of induced pluripotent stem cells into retinal pigment epithelial cells, maximizing pigmentation score (Kanda et al., 2022). We use the 253G1\_results\_r5r7r10 table. The seven settings cover preconditioning concentration and duration, detachment duration and pipetting, and differentiation supplement schedules. Ordered finite concentration/detachment levels, integer durations and speed, and a binary pipetting-length setting define a discrete configuration space.

C2C12. We use a 209-row processed table from the upstream DBO archive associated with multi-information-source culture-media optimization (Cosenza et al., 2022). Fourteen media coordinates are continuous on [0, 1] in the supplied representation. All methods maximize the archived response score on its supplied scale, without additional standardization. The source query label selects among four measurement sources for response lookup rather than specifying a physical culture intervention. Media and source-evaluation costs are outside the objective. Appendix A.3 details the table provenance and response interpretation.

Response and initialization protocol. Both tasks return the score of a nearest historical configuration in normalized parameter coordinates; C2C12 matching is restricted by source. This oracle permits new parameter combinations but does not validate their physical response. Initialization uses the ten lowest-scoring historical observations, shared across methods, followed by twenty search evaluations. This outcome-conditioned low-start protocol is a retrospective stress test, not label-free prospective initialization. The exact-grid iPSC control restricts candidates to observed configurations and removes nearest-neighbor assignment.

Cross-domain checks include six preselected Bayesmark HPO tasks from the LLAMBO repository (Liu et al., 2024): breast, digits, and diabetes, each paired with RandomForest and SVM. These train models directly without a nearest-neighbor oracle. Classification maximizes five-fold CV accuracy; diabetes minimizes positive CV MSE. Initialization uses ten label-free public configurations, shared within each pair, followed by twenty search evaluations. Held-out test scores never enter optimization feedback (Appendix A.15). We also retain all five Summit reaction tasks (Felton et al.,

2021; Hone et al., 2017; Reizman et al., 2016) and Olympus HPLC (Häse et al., 2021) in the appendix.

## 5.2 Baselines

We evaluate a feasibility-preserving adaptation of LLAMBO (LLAMBO-Feasible, hereafter LLAMBO). It retains the LLAMBO surrogate and EI, with the shared typed sampling/refinement interface but no FGSS state or test-policy overrides. The free-form variant, LLAMBO-Freeform, is evaluated separately for candidate-interface diagnostics; it is an implementation-level variant rather than a strict reproduction of original LLAMBO.

We also compare Random Search over the legal domains, HEBO (Cowen-Rivers et al., 2022), and GP-EI implemented in BoTorch (Balandat et al., 2020). GP-EI standardizes outcomes and normalizes numerical coordinates, using categorical kernels for pipetting length or measurement source. It optimizes EI over legal discrete settings for iPSC and continuous coordinates with enumerated sources for C2C12. All five methods share the actual initial observations, admissible domains, evaluation budget, and response oracle. Baseline fitting and acquisition settings are detailed in Appendix A.6.

The iPSC comparison adds Reflective Summary: each round, the same extractor’s evidence and previous prose summary are used to produce a replacement summary for the shared candidate refiner. It has no hypothesis curation or tests. FGSS w/o Tests retains structured reflection and ordinary state guidance but disables matched-test generation, test-policy overrides, and direct support updates. No-Override retains test candidates and support updates but always selects by EI. Reflection settings and evaluation budgets are shared, not realized token totals. HPO compares the two complete LLM pipelines.

## 5.3 Metrics

We report final best-observed objective value and search-stage best-so-far area under the curve (AUC). Let K be the initialization count and $T$ the subsequent search budget. For maximization,

$$
\begin{array} { r } { B _ { t } = \underset { ( x , y ) \in \mathcal { D } _ { t } } { \operatorname* { m a x } } y , } \\ { \mathrm { A U C _ { s e a r c h } } = \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } B _ { t } . } \end{array}\tag{11}
$$

Final best is $B _ { T } ;$ for minimization, replace max by min. Search AUC averages attained performance after initialization, not ROC AUC. Tables report means and sample standard deviations over six paired runs; bands are standard deviations. Paired bootstrap intervals are exploratory and unadjusted for multiple comparisons (Appendices A.7, A.11, and A.15). Metrics always use true oracle outcomes. Accuracy and MSE are analyzed separately.

## 5.4 Implementation Details

The primary tasks use K = 10, T = 20, and batch size one. The shared LLAMBO configuration uses GPT-4o mini, ten candidates, two templates, and ten generations. We use this backend for its practical balance of instruction following and API cost. FGSS adds a reflection call each round at temperature zero with an output cap of 1500 tokens and one configured retry, increasing total LLM compute. Table 8 reports its recorded per-run token use and elapsed time. Regime cutoffs use supplied taskresponse units (Table 5). Optional domain-prior cards and injections are disabled; task descriptions and pretrained LLM knowledge remain available. The appendix specifies settings, prompts, state examples, and provenance.

## 6 Results

## 6.1 Main Results on Protocol and Media Search

Table 1 compares five methods under the shared low-start, nearest-neighbor protocol. FGSS-BO has the highest sample mean final best and search AUC in both primary tasks. Relative to LLAMBO, the mean differences are +0.123/+0.147 for iPSC final best/AUC and +0.445/ + 0.141 for C2C12.

On iPSC, final-best margins over GP-EI and Random Search are modest (+0.020 and +0.033), while the AUC margin over GP-EI is +0.135. The paired iPSC interval versus GP-EI includes zero for final best but lies above zero for AUC (Table 9). C2C12 intervals versus GP-EI cross zero on both metrics, as does its AUC interval versus LLAMBO.

Figure 2 shows the paired LLM trajectories; Figure 4 compares all five methods (Appendix A.6).

## 6.2 Component Ablations

Table 2 places prose reflection and EI-only selection alongside full FGSS. Reflective Summary already raises both sample means relative to LLAMBO. Full FGSS exceeds it by 0.0285 in final best but is 0.0061 lower in search AUC. No-Override also retains higher means than LLAMBO:

its AUC exceeds full FGSS’s, while final best is slightly lower, with both paired intervals crossing zero (Figure 6). Against FGSS w/o Tests, full FGSS’s differences are +0.0088/ − 0.0275, again with intervals crossing zero (Appendix A.11). These controls show mixed effects on objective progress despite the differences in hypothesis management. Figure 7 reports the immediate opportunity cost of full FGSS’s 18 selection changes (Appendix A.13).

State-only feedback permutation gives mixed effects: aligned feedback yields a higher final-best mean and lower final-result variability, whereas permutation yields a higher search AUC. Both paired difference intervals include zero (Appendix A.12).

## 6.3 Direct-Training HPO Transfer

Tables 14 and 16 report the complete six-task HPO results and paired differences. Diabetes SVM provides the clearest gain: mean final CV MSE falls from 0.57768 to 0.51711 (10.48%), and search AUC from 0.61046 to 0.55636 (8.86%). Breast SVM improves modestly; breast RF and digits SVM are nearly tied. Digits RF and diabetes RF have mixed final/AUC results. These direct evaluations show task-dependent transfer beyond the table-backed oracle.

## 6.4 Candidate-Sampling Diagnostics

Figure 3 diagnoses earlier free-form duplication and rejection. On iPSC, 18.4% of raw proposals are unique and 3.4% survive validity/novelty checks, motivating legal-space construction. The six-seed interface controls in Figure 8 show that Freeform remains viable on C2C12, with sample means close to Feasible (Appendix A.14). Both main LLM methods use the same feasible interface across tasks, holding candidate-interface handling fixed in the FGSS comparison.

## 6.5 Oracle and Cross-Domain Controls

Exact-grid iPSC restricts candidates to observed configurations and removes nearest-neighbor assignment. Final-best means are close (0.840 LLAMBO, 0.848 FGSS-BO), while mean search AUC favors LLAMBO (0.750 versus 0.710; Appendix A.9). This pattern suggests protocoldependent performance; the control changes both the candidate domain and oracle assignment. On all five Summit tasks, HEBO and GP-EI outperform the two LLM methods in sample mean final best and AUC; FGSS-BO exceeds LLAMBO on four final-best means and three AUC means, but not Reizman Case 4 (Appendix A.8). HPLC yields higher FGSS-BO means on both metrics (Appendix A.10).

<table><tr><td>Task</td><td>Metric</td><td>Random Search</td><td>HEBO</td><td>GP-EI</td><td>LLAMBO</td><td>FGSS-BO</td></tr><tr><td></td><td>iPSC-RPE Final best</td><td> $0 . 7 8 0 \pm 0 . 0 7 8$ </td><td> $0 . 5 6 9 \pm 0 . 1 6 9$ </td><td> $0 . 7 9 3 \pm 0 . 0 2 6$ </td><td> $0 . 6 9 1 \pm 0 . 1 1 7$ </td><td> $0 . 8 1 3 \pm 0 . 0 2 3$ </td></tr><tr><td>iPSC-RPE Search AUC</td><td></td><td> $0 . 5 5 7 \pm 0 . 1 5 0$ </td><td> $0 . 4 6 5 \pm 0 . 1 0 0$ </td><td> $0 . 5 0 5 \pm 0 . 1 4 5$ </td><td> $0 . 4 9 4 \pm 0 . 0 5 7$ </td><td> $0 . 6 4 0 \pm 0 . 1 1 7$ </td></tr><tr><td>C2C12</td><td></td><td> $0 . 7 9 0 \pm 0 . 3 4 0$ </td><td> $1 . 0 2 6 \pm 0 . 6 2 3$ </td><td> $1 . 5 1 6 \pm 0 . 9 0 4$ </td><td> $1 . 2 8 1 \pm 0 . 2 2 0$ </td><td> $1 . 7 2 5 \pm 0 . 4 1 1$ </td></tr><tr><td>C2C12</td><td>Final best Search AUC</td><td> $0 . 5 5 2 \pm 0 . 2 8 9$ </td><td> $0 . 7 2 2 \pm 0 . 4 0 4$ </td><td> $0 . 8 6 8 \pm 0 . 6 5 3$ </td><td> $1 . 0 2 5 \pm 0 . 2 6 4$ </td><td> $1 . 1 6 7 \pm 0 . 2 5 6$ </td></tr></table>

Table 1: Primary offline protocol/media search over six paired runs: mean ± sample standard deviation, higher is better. All methods share ten low-score initial observations and twenty search evaluations under the same task-specific nearest-neighbor response oracle. LLAMBO uses the shared feasibility-preserving interface.

![](images/605777741731b1de52b65ab5d2a127e8005c8a3267deb3b4ac0a72a7ea750ad1.jpg)

(b) C2C12  
![](images/0b0dcc265c888801c551d3c1275a7938201a6f853e6087bba6367ef8d024dba6.jpg)  
Figure 2: Primary-task mean best-so-far objective over six paired runs. The dashed line separates ten shared low-score initialization observations from twenty search evaluations; shading shows one sample standard deviation. LLAMBO and FGSS-BO share the feasible-candidate interface and nearest-neighbor oracle. C2C12 scores refer to the retained configuration–source query task.

<table><tr><td>Method</td><td>Final best</td><td>Search AUC</td></tr><tr><td>LLAMBO</td><td> $. 6 9 1 \pm . 1 1 7$ </td><td> $. 4 9 4 \pm . 0 5 7$ </td></tr><tr><td>Reflective Summary</td><td> $. 7 8 5 \pm . 0 6 1$ </td><td> $. 6 4 6 \pm . 0 5 4$ </td></tr><tr><td>FGSS w/o Tests</td><td> $. 8 0 4 \pm . 0 7 0$ </td><td> $. 6 6 8 \pm . 0 5 1$ </td></tr><tr><td>FGSS No-Override</td><td> $. 8 0 2 \pm . 1 1 0$ </td><td> $. 6 9 0 \pm . 0 9 8$ </td></tr><tr><td>Full FGSS</td><td> $. 8 1 3 \pm . 0 2 3$ </td><td> $. 6 4 0 \pm . 1 1 7$ </td></tr></table>

Table 2: iPSC component comparison, six paired runs; mean ± sample standard deviation. All variants share the low-start nearest-neighbor protocol. No-Override retains test candidates and support updates but always selects by EI. LLAMBO uses the shared feasibilitypreserving interface.

## 7 Discussion

FGSS organizes hypothesis management around inspectable records: a proposed direction identifies parameters, an anchor, and an intervention whose later comparison can revise the same record. The

LLM interprets evidence and proposes changes; fixed rules govern admissibility and numerical support updates. Prose reflection and EI-only selection also achieve higher sample means than LLAMBO, so the component results separate this record-andaction interface from a uniform optimization advantage for full FGSS.

A candidate can outperform its matched control without improving the incumbent. Local comparative support and global progress therefore answer different questions, and FGSS tracks them separately. The higher AUC of No-Override and FGSS w/o Tests suggests a trade-off between early objective progress and obtaining targeted comparisons under this protocol.

HPO extends the evaluation to directly trained objectives, with the strongest gains on diabetes SVM. Exact-grid iPSC’s lower mean FGSS AUC shows that state-guided priorities need not improve progress in a restricted candidate domain. Together with the reaction controls, this motivates adapting guidance and testing to the search interface and response landscape. Future refinements could preserve more global exploration when local hypotheses dominate and use observed test utility to allocate the testing budget.

![](images/7ad8f6039e353246b2cbd30400e3cc69a6aeb879925988f29eb367576874d557.jpg)

![](images/46486f9cf364d1ddf7bd7ef7f021c0b4bed170a7d6e1f0cca2098f1db1f90dcd.jpg)  
Figure 3: Proposal-level candidate diagnostics from earlier runs, comparing free-form LLM proposals with feasibility-preserving construction. Severe iPSC-RPE duplication motivates a shared feasible-candidate interface.

## 8 Conclusion

FGSS provides structured hypothesis management for LLM-guided BO, turning proposed directions into executable parameter interventions and linking eligible comparisons to later revisions. FGSS-BO retains LLAMBO’s surrogate and default EI selection, with a bounded testing policy. Its records expose the path from language-based hypotheses to search actions and updates. The complete system achieves higher sample means on the two retrospective protocol/media tasks and yields gains on some direct-training HPO tasks, while component and oracle controls show mixed effects across settings.

## 9 Limitations

The primary tasks use historical-table nearestneighbor responses and outcome-conditioned lowscore initialization. Their results describe that retrospective protocol, not label-free prospective laboratory search. C2C12 retains a measurement-source variable, so the query score should not be interpreted as an assay-independent biological optimum. Prospective validation is needed to assess measurement noise, batch effects, experimental failures, and temporal drift.

Evaluation uses one LLM backend and base optimizer; FGSS adds reflection cost. The HPO suite compares the two complete LLM pipelines on directly trained objectives. Future evaluation could extend paired runs to ten seeds and examine further backends, initialization policies, budgets, and optimizers; noise-aware support updates and test prioritization could refine the balance of local hypotheses and exploration.

## References

Maximilian Balandat, Brian Karrer, Daniel R. Jiang, Samuel Daulton, Benjamin Letham, Andrew Gordon Wilson, and Eytan Bakshy. 2020. BoTorch: A framework for efficient monte-carlo bayesian optimization. In Advances in Neural Information Processing Systems, volume 33, pages 21524–21538.

Zachary Cosenza, Raul Astudillo, Peter I. Frazier, Keith Baar, and David E. Block. 2022. Multi-information source bayesian optimization of culture media for cellular agriculture. Biotechnology and Bioengineering, 119(9):2447–2458.

Alexander I. Cowen-Rivers, Wenlong Lyu, Rasul Tutunov, Zhi Wang, Antoine Grosnit, Ryan Rhys Griffiths, Alexandre Max Maraval, Jianye Hao, Jun Wang, Jan Peters, and Haitham Bou-Ammar. 2022. HEBO: Pushing the limits of sample-efficient hyperparameter optimisation. Journal ofArtificial Intelligence Research, 74:1269–1349.

Kobi C. Felton, Jan G. Rittig, and Alexei A. Lapkin. 2021. Summit: Benchmarking machine learning methods for reaction optimisation. Chemistry– Methods, 1(2):116–122.

Peter I. Frazier. 2018. A tutorial on bayesian optimization. Preprint, arXiv:1807.02811.

Rushil Gupta, Jason Hartford, and Bang Liu. 2025. LLMs for bayesian optimization in scientific domains: Are we there yet? In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 15482–15510.

Florian Häse, Matteo Aldeghi, Riley J. Hickman, Loïc M. Roch, Melodie Christensen, Elena Liles, Jason E. Hein, and Alán Aspuru-Guzik. 2021. Olympus: A benchmarking framework for noisy optimization and experiment planning. Machine Learning: Science and Technology, 2(3):035021.

Christopher A. Hone, Nicholas Holmes, Geoffrey R. Akien, Richard A. Bourne, and Frans L. Muller. 2017. Rapid multistep kinetic model generation from transient flow data. Reaction Chemistry & Engineering, 2(2):103–108.

Donald R. Jones, Matthias Schonlau, and William J. Welch. 1998. Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13(4):455–492.

Genki N. Kanda, Taku Tsuzuki, Motoki Terada, Noriko Sakai, Naohiro Motozawa, Tomohiro Masuda, Mitsuhiro Nishida, Chihaya T. Watanabe, Tatsuki Higashi, Shuhei A. Horiguchi, Taku Kudo, Motohisa Kamei, Genshiro A. Sunagawa, Kenji Matsukuma, Takeshi Sakurada, Yosuke Ozawa, Masayo Takahashi, Koichi Takahashi, and Tohru Natsume. 2022. Robotic search for optimal cell culture in regenerative medicine. eLife, 11:e77007.

Tennison Liu, Nicolás Astorga, Nabeel Seedat, and Mihaela van der Schaar. 2024. Large language models to enhance bayesian optimization. In International Conference on Learning Representations.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36.

Harini Narayanan, Joshua A. Hinckley, Rachel Barry, Brendan Dang, Lenna A. Wolffe, Adel Atari, Yuen-Yi Tseng, and J. Christopher Love. 2025. Accelerating cell culture media development using bayesian optimization-based iterative experimental design. Nature Communications, 16:6055.

Brandon J. Reizman, Yi-Ming Wang, Stephen L. Buchwald, and Klavs F. Jensen. 2016. Suzuki–Miyaura cross-coupling optimization enabled by automated feedback. Reaction Chemistry & Engineering, 1(6):658–666.

Yusuf Roohani, Andrew Lee, Qian Huang, Jian Vora, Zachary Steinhart, Kexin Huang, Alexander Marson, Percy Liang, and Jure Leskovec. 2025. BioDiscoveryAgent: An AI agent for designing genetic perturbation experiments. In International Conference on Learning Representations.

Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas. 2016. Taking the human out of the loop: A review of bayesian optimization. Proceedings ofthe IEEE, 104(1):148–175.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36.

Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. 2012. Practical bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems, volume 25.

Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. 2026. Dynamic cheatsheet: Test-time learning with adaptive memory. In Proceedings ofthe 19th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7080–7106. Association for Computational Linguistics.

Zhuo Yang, Daolang Wang, Lingli Ge, Beilun Wang, Tianfan Fu, and Yuqiang Li. 2026. Reasoning BO: Enhancing bayesian optimization with the reasoning power of LLMs. Materials Genome Engineering Advances, 4(3):e70086.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. 2026a. Agentic context engineering: Evolving contexts for self-improving language models. In International Conference on Learning Representations.

Yiming Zhang, Jun Jin Choong, Kaushalya Madhawa, and Keisuke Ozawa. 2026b. AutoLead: An LLMguided bayesian optimization framework for multiobjective lead optimization. Journal of Chemical Information and Modeling, 66(9):5091–5100.

## A Additional Experimental Details

Algorithm settings (Appendix A.1) and prompt/state examples (Appendix A.4) precede the detailed experimental controls below.

## A.1 Algorithm Details

The following specifies the parameterized, sequential full-FGSS implementation. Scientific-task objectives are maximized; diabetes HPO minimizes positive MSE, reversing improvement and supportupdate signs. S supplements the observation history with identified hypotheses, revisions, and test records, including persistent archived/retired memory; only the active set is capped. The reflector receives compact cards for all three groups, so its input can grow with retained memory. Candidate guidance includes at most four active cards.

Full FGSS loop. Input: typed search space X , oracle f, paired initial observations $\mathcal { D } _ { 0 }$ , search budget T, candidate count N, and empty carried state $S _ { 0 } ^ { - }$

1. For each search round $t = 0 , \ldots , T - 1$ , extract $e _ { t }$ from all observed configurations and outcomes. Rank the top/bottom three and retain the latest five; compute coverage, stagnation, and local contrasts.

2. Ask the reflector for evidence-linked add/confirm/revise/weaken/retire operations and a proposed regime using $e _ { t }$ and $S _ { t } ^ { - }$ Apply validation and semantic deduplication, execute operations, apply eligible staleness decay, and govern active capacity to obtain $S _ { t }$ . Resolve the regime with the deterministic gate.

3. Construct N legal, unseen candidates from global diversity, state guidance, local exploitation, and recovery. Reserve up to two slots when N = 10 for eligible matched interventions. Protect these tests from the shared LLM candidate refiner; validate and replenish the ordinary candidates.

4. Query the shared LLAMBO surrogate on the current history and pool; compute EI. Select its maximizer unless a valid test-policy slot is available. An override requires an eligible test, support score at least 0.55, and the budget/interval checks below.

5. Evaluate the selected configuration with the true oracle and append its outcome to obtain $\mathcal { D } _ { t + 1 }$ . If its lineage passes direct-test eligibility, compare its outcome with the identified observed control and update only the parent hypothesis. Record global-incumbent progress separately and carry the resulting state as $S _ { t + 1 } ^ { - } .$

6. Store the selected configuration, true outcome, candidate lineage, state transition, and call metadata; continue until T evaluations are completed.

The matched baseline uses the same typed sampling/refinement substrate and surrogate but omits steps involving state, direct tests, and test-policy overrides. Optional domain-prior cards are disabled in both methods; task descriptions and pretrained LLM knowledge remain available.

Local evidence and control matching. For numerical variables, differences are normalized by the specified range; unordered categories use a zero/one mismatch. For a target parameter, an observational pair must have target difference at least 0.10 and mean nuisance difference at most 0.30. Pairs are ordered by increasing nuisance difference, decreasing target difference, and decreasing absolute outcome difference, retaining at most four per parameter and twelve overall. Median directional differences and agreement summarize support; at most five context-dependent interaction records are retained. These are observational contrasts, not identified causal effects.

A direct test has one parent hypothesis, one or two target variables, and an observed anchor. Only the primary hypothesis receives its outcomebased support update, even if other hypotheses contributed to candidate guidance. In open spaces, non-target coordinates are held fixed and targeted coordinates are changed according to the intervention. Numeric steps are 10% of the range per depth (log-range for log variables); integer steps are rounded with a minimum of one, and ordered finite supports advance by their index. Depths one through six are attempted. Unordered levels change without a numerical direction. All changes are clipped/canonicalized, and unchanged or previously evaluated configurations are rejected. Interactions change the specified coordinates jointly; this is not an exhaustive factorial design. For tablebacked tasks, only label-free covariate matches to observed rows can receive a direct support update, with root-mean-square and per-coordinate nuisance limits of 0.25 and distinct oracle identities. Exactgrid candidates must be observed configurations. Approximate matches can differ in non-target variables; nearest-neighbor assignment can also change the underlying experimental row. These comparisons are not estimates of an isolated biological causal effect.

Ordinary candidate policies. Scientific-task global candidates use independent uniform sampling of numerical bounds or legal finite levels; HPO uses the stated linear/log/logit sampling scales (Appendix A.15). Local exploitation chooses a top-evidence configuration uniformly, then samples each coordinate locally with probability 0.8 (globally otherwise). Continuous local sampling is uniform within 12% of the full range around the center, clipped to bounds; integers use the corresponding rounded radius, ordered supports use the center and adjacent levels, and unordered categories retain the center with probability 0.8 or sample a legal level. State-guided candidates prioritize testable anchor interventions using adaptivepriority sampling, with a small nuisance perturbation for ordinary guided proposals; these are not automatically eligible for support updates. Recovery samples up to 128 global proposals to find normalized distance at least 0.25 from repeated low-gain centers, then falls back to global sampling. Challenger proposals add evidence-linked alternative parameter/direction/context hypotheses; they do not replace every recovery draw.

Ratios are normalized and integer quotas use largest-remainder allocation. Unusable guided quotas return to global search or recovery rather than generating ungrounded tests.

Support score and lifecycle. The LLMproposed initial support score is a validated heuristic, not a calibrated posterior probability or a test-success frequency. For new falsifiable cards, uncertainty caps it at 0.60/0.72/0.85 for high/medium/low uncertainty. Let $v _ { j }$ be 1, 0.75, or 0.5 for low, medium, or high uncertainty, $M _ { j , x } \in \left[ 0 , 1 \right]$ the recorded match quality, $a _ { j , x } = 0 . 7 5$ for a transferred anchor and 1 otherwise, and $n _ { j , \mathrm { c t r l } }$ its previous uses. The effective evidence weight is

$$
w _ { j , x } = \mathrm { c l i p } \left( \frac { v _ { j } M _ { j , x } a _ { j , x } } { 1 + n _ { j , \mathrm { c t r l } } } , 0 , 1 \right) .\tag{12}
$$

Exact open-space interventions have $\begin{array} { r l } { M _ { j , x } } & { { } = } \end{array}$ 1; table-backed match quality is $\mathrm { e x p } [ - ( d _ { \mathrm { r m s } } \cdot +$ $d _ { \mathrm { m a x } } ) ]$ Approximate matches receive discounted weight. The threshold $\epsilon _ { j , t }$ is the larger of the extracted effect threshold (0.02 times the observed score span, floor $1 0 ^ { - 1 2 } )$ and max $\cdot ( 1 0 ^ { - 1 0 } , 1 0 ^ { - 8 } \operatorname* { m a x } ( | y _ { j , t } ^ { \mathrm { c t r l } } | , 1 ) )$ . This empirical tolerance is not a noise-calibrated significance threshold. Support changes by +0.12w, 0, or −0.08w. Staleness applies when rounds since verification exceed the card’s window (two for cards created at the early stage, three subsequently), and subtracts 0.08 once per last-verified epoch. Active and weakened bands use 0.55 and 0.22; lowsupport cards can be archived without being retired. Failure-based retirement requires at least two consecutive failures and cumulative consecutive failure weight at least two. Historical controls are reused under a fixed response oracle; this does not control temporal drift or measurement noise in prospective experiments.

Adaptive priority and override accounting. Eligible test candidates are ordered lexicographically by adaptive priority, EI, support score times uncertainty weight and match quality, challenger flag, negative target distance, and negative candidate index. Adaptive priority is

$$
\begin{array} { r l } & { p _ { j } = ( 0 . 5 + 0 . 5 q _ { j } ) U _ { j } B _ { j } R _ { j } } \\ & { \qquad \times ( 1 + n _ { j } ) ^ { - 1 / 2 } ( 1 + n _ { \mathcal { P } _ { j } } ) ^ { - 1 / 2 } } \\ & { \qquad \times ( 1 + n _ { \mathrm { a n c h o r } } ) ^ { - 1 / 2 } } \\ & { \qquad \times \displaystyle \frac { 1 } { | \mathcal { P } _ { j } | } \sum _ { p \in \mathcal { P } _ { j } } ( 1 + n _ { p } ) ^ { - 1 / 2 } . } \end{array}\tag{13}
$$

Here $n _ { j } , ~ n _ { \mathcal { P } _ { j } } , ~ n _ { \mathrm { a n c h o r } } ,$ and $n _ { p }$ are direct-test counts for the hypothesis, parameter set, anchor, and individual parameter. $U _ { j }$ is $0 . 6 5 / 0 . 8 5 / 1$ for low/medium/high uncertainty, $B _ { j } ~ = ~ 1$ .15 for challengers and 1 otherwise, and $R _ { j }$ $\sqrt { ( d + 1 ) / ( d + 2 ) }$ for d rounds since the more recent of the hypothesis and parameter-set tests (1 if untested). Saturated/ineligible contexts receive zero priority. This is a deterministic heuristic, not a learned acquisition function. The uncertainty factor in this priority differs deliberately from the conservative evidence weight.

Positive budget fractions define caps max(1, ⌊0.20T⌋) and max(1, ⌊0.10T⌋) for base and additional challenger slots. For $T = 2 0$ these are four and two. Base usage counts trials receiving direct support updates, including tests selected by EI; extra slots require stagnation and exhausted base usage. Overrides have a minimum interval of three rounds. EI-selected tests are not excluded when the override budget is exhausted. The direct candidate quota is max(1, round(0.20N)), preserving at least one ordinary candidate when $N > 1$

Threshold roles and configuration sources. The normalized-distance, support-score, and budget constants in Table 3 are shared development settings. Table 5 gives fixed absolute response cutoffs τ on each task’s supplied scale. These are FGSS algorithm settings, rather than biological success criteria adopted from the source studies (Kanda et al., 2022; Cosenza et al., 2022). iPSC uses a CLI override; C2C12 uses its task-runner default. All six primary FGSS runs per task record the same cutoff. The effect tolerance instead adapts to the observed score span.

Strong reliability requires either improvement over the initial best or an absolute-cutoff hit, together with at least five search observations and three observations near the current best. The nearbest margin is max(0.10 span(y), 10<sup>−8</sup>) over observed outcomes. Cutoff hits mean $b _ { t } ~ \geq ~ \tau$ for maximization or $b _ { t } \leq \tau$ for minimization. Positive MSE therefore reaches the strong regime through improvement and supporting observations rather than HPO’s zero cutoff.

<table><tr><td>Setting</td><td>Value</td><td>Scope/provenance</td></tr><tr><td>Initialization/search evaluations</td><td>Primary/HPO 10/20; Summit 8/20</td><td>Six paired seeds, batch size one</td></tr><tr><td>Candidate pool; templates; generations</td><td>10; 2; 10</td><td>Shared main-run configuration</td></tr><tr><td>Reflector output/temperature/retry</td><td>1500 tokens; 0; 1</td><td>FGSS extra call; retries logged</td></tr><tr><td>Shared refiner output/temperature</td><td>1000 tokens; 0</td><td>Top five plus recent five observations</td></tr><tr><td>Evidence top/bottom/recent</td><td>3/3/5</td><td>Outcome-ranked top/bottom</td></tr><tr><td>Stagnation windows and thresholds</td><td>3/5; 0.01/0.02</td><td>Short/long progress heuristics</td></tr><tr><td>Numeric bins; near-duplicate tolerance</td><td>5; 0.05</td><td>Normalized configuration coordinates</td></tr><tr><td>Contrast target/nuisance thresholds</td><td>0.10/0.30</td><td>Observational pair selection</td></tr><tr><td>Pair limits; interactions</td><td>4 per parameter; 12 total; 5</td><td>Extractor defaults</td></tr><tr><td>Contrast effect threshold</td><td>0.02 times observed score span</td><td>Floor  $1 0 ^ { - 1 2 }$ </td></tr><tr><td>Active/per-parameter/type capacity</td><td>4/1; explore/exploit 2/2</td><td>Active guidance only</td></tr><tr><td>Support bands; stale penalty</td><td>0.55/0.22; 0.08</td><td>Not posterior probabilities</td></tr><tr><td>Success/failure increments</td><td> $+ 0 . 1 \dot { 2 } / - 0 . 0 8$  times evidence weight</td><td>Neutral increment zero</td></tr><tr><td>Retirement failure count/weight</td><td>2/2</td><td>Both conditions required</td></tr><tr><td>Test pool/base/challenger fractions</td><td> $0 . { \dot { 2 } } 0 / 0 . 2 0 / 0 . 1 0$ </td><td>Pool quota vs override caps</td></tr><tr><td>Test interval/support score</td><td>3 rounds; 0.55</td><td>Additional eligibility checks</td></tr><tr><td>Direct nuisance bounds</td><td>0.25/0.25</td><td>Aggregate and each coordinate</td></tr><tr><td>Optional domain prior</td><td>Disabled</td><td>No retrieved/static prior card</td></tr></table>

Table 3: Implementation settings inherited from development defaults and used in the recorded main runs, without a hyperparameter sweep.

<table><tr><td>Regime</td><td>Global</td><td>Guided</td><td>Local</td><td>Recovery</td></tr><tr><td>Explore</td><td>.55</td><td>.25</td><td>.10</td><td>.10</td></tr><tr><td>Mixed</td><td>.40</td><td>.30</td><td>.20</td><td>.10</td></tr><tr><td>Exploit</td><td>.25</td><td>.30</td><td>.35</td><td>.10</td></tr><tr><td>Recover</td><td>.45</td><td>.15</td><td>.15</td><td>.25</td></tr></table>

Table 4: Nominal candidate-allocation presets, not fixed realized ratios. The gate preserves a global floor of .30 (.55 before reliable support, .45 under weak support); plateau recovery enforces global/recovery floors .45/.25 and guided cap .20.

<table><tr><td>Task</td><td>Cutoff</td><td>Configuration source</td></tr><tr><td>iPSC-RPE</td><td>0.12</td><td>Recorded CLI override</td></tr><tr><td>C2C12</td><td>0.05</td><td> Task-runner default</td></tr><tr><td>SnAr</td><td></td><td>0.10 Summit task preset</td></tr><tr><td>Reizman 1-4</td><td></td><td>50 Summit task preset</td></tr><tr><td>HPLC</td><td></td><td>1000 Task-runner default</td></tr><tr><td>HPO</td><td></td><td>0 Task-bridge setting</td></tr></table>

Table 5: Absolute regime-gate cutoffs and their configuration sources. iPSC uses the reported override rather than the generic runner default of 0.01.

For maximization, the same cutoff also replaces the refiner’s hypothesis-guidance text with a broadexploration instruction when both the best outcome and observed span are below τ, with at most two distinct positive outcomes after six-decimal rounding. State records and the regime-controlled candidate policy remain enabled.

<table><tr><td>Seed</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>iPSC overrides</td><td>2</td><td>4</td><td>2</td><td>4</td><td>3</td><td>3</td></tr><tr><td>iPSC EI tests</td><td>6</td><td>0</td><td>6</td><td>2</td><td>1</td><td>4</td></tr><tr><td>C2C12 overrides</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>C2C12 EI tests</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 6: Actual selection changes and EI-selected matched tests; each run has twenty search evaluations. EI tests receive a recorded support update without changing EI selection.

## A.2 Realized Test Policy and Matching

An actual override changes the selected candidate index from the recorded EI choice, rather than merely activating arbitration. Table 6 audits all six full-FGSS primary runs. iPSC has 18 overrides out of 120 search evaluations (15%), plus 19 EIselected tests receiving support updates. C2C12 has neither overrides nor direct support updates: its observed gain cannot be attributed to the matchedtest path. Its structured reflection and ordinary state-guided search remain enabled.

All 37 iPSC updates use approximate covariate matches and one primary hypothesis. Their match quality has median 0.748, interquartile range [0.729, 0.784], and range [0.664, 0.918]. Normalized nuisance RMS distance has median 0.093 and maximum 0.159; maximum-coordinate distance has median 0.188 and maximum 0.250.

Generation logs contain 482 iPSC validationcheck events, including 155 explicitly rejected events (32.2%: 152 same-oracle and three alreadyobserved configurations). C2C12 records 476 events, including 329 same-oracle rejections (69.1%). The denominators include intermediate deferred checks and repeated proposal attempts; these are logged check-event rates, not rejection rates over unique tests or all internal generation attempts. Subsequent reanchoring or observed-row matching can still succeed.

FGSS w/o Tests disables test generation, override selection, and outcome support updates together. A separate EI-only selection ablation and post-hoc fixed-oracle cost audit appear in Appendix A.13.

## A.3 Search Spaces and Initialization

Summit initializes continuous coordinates with seeded Latin hypercube sampling and balances catalyst categories; initial points are identical within each paired seed. The primary iPSC-RPE and C2C12 tasks use outcome-conditioned low-score initialization and nearest-neighbor response lookup for retrospective evaluation. Exact-grid iPSC restricts candidate configurations to observed rows. HPLC initializes uniformly from observed configurations without labels and uses the reported offline lookup. C2C12’s source indicator denotes measurement provenance, so the optimizer queries configuration–source pairs.

C2C12 table provenance and response scale. The processed table copies all 209 rows of the upstream DBO archive: the first fourteen columns of BO\_data.txt provide media coordinates already on [0, 1], and its fifteenth column supplies source. The two columns of BO\_outputs.txt are retained verbatim as score and score\_var. An all-row numerical audit confirms this mapping exactly. The source study measures C2C12 growth through cell counting and correlated assay responses; we report the saved response-score scale rather than reconstructing absolute cell counts or an assay-specific inverse transformation. Maximization seeks a larger archived response at a fixed evaluation budget. Media costs, source-specific evaluation costs, and source-bias calibration are outside this objective.

All rows define a fixed response lookup, with no C2C12 train/test split or learned emulator fitted in this study. Beyond the disclosed low-score initialization, surrogate and reflector inputs contain queried outcomes only. Matching unobserved configurations and interventions consults covariates, not their response labels. The oracle returns a saved scalar deterministically; score\_var is retained as metadata rather than used to sample additional observation noise.

## A.4 Reflector Prompt and State Schema

The reflector receives current evidence and the previous state, and returns an assessment, a proposed search regime, and evidence-linked operations. The following is an abbreviated output contract, not a claimed model response:

{   
"global\_assessment": {   
"stage": "early|mid|late",   
"search\_mode\_suggestion": "balanced",   
"uncertainty\_note": "..."   
},   
"operations": [   
{   
"op": "add",   
"type": "exploit",   
"parameter": "<legal parameter>",   
"parameters": ["<legal parameter>"],   
"hypothesis\_kind": "main\_effect",   
"anchor\_observation\_id": "obs\_...",   
"intervention": {"changes": {...},   
"controlled\_parameters": [...]},   
"expected\_outcome": {   
"objective": "improve",   
"direction": "increase"},   
"evidence\_pair\_ids": ["pair\_..."],   
"evidence\_refs": ["contrastive\_evidence"],   
"confidence": 0.63   
}   
],   
"search\_phase": {   
"mode": "explore|mixed|exploit|recover",   
"target\_parameters": [...]   
}   
}

Existing-card operations use target\_id; the early prompt permits only add operations, while later prompts also permit confirm, revise, weaken, and retire. The validator requires evidence references and executable ranges. Invalid responses can trigger a rule-based fallback; deterministic repair and challenger operations can supplement valid responses, and these paths are recorded. A falsifiable test specification cannot be silently changed in place; a different intervention requires a new evidence-linked card. The LLM does not supply measured test counts or invent tested candidate IDs: these are populated by the deterministic evaluation/support-update path. Authentic schema and log keys retain confidence for the heuristic support score and credit for comparative evidence updates; neither denotes a calibrated probability or an identified causal effect.

Recorded state example. The following fields are selected from the pre-selection state in Reizman

<table><tr><td>Task</td><td>Variable</td><td>Type</td><td>Admissible domain</td></tr><tr><td>SnAr</td><td>Residence time Pyrrolidine equivalents</td><td>Continuous Continuous</td><td>[0.5, 2.0] min [1, 5] equivalents</td></tr><tr><td></td><td>DFNB concentration</td><td>Continuous</td><td>[0.1, 0.5] mol/L</td></tr><tr><td></td><td>Temperature</td><td>Continuous</td><td>[30, 120] degrees Celsius</td></tr><tr><td>Reizman 1-4</td><td></td><td></td><td></td></tr><tr><td></td><td>Residence time</td><td>Continuous</td><td>[60, 600] s</td></tr><tr><td></td><td>Temperature</td><td>Continuous</td><td>[30, 110] degrees Celsius</td></tr><tr><td></td><td>Catalyst loading</td><td>Continuous</td><td>[0.5, 2.5] mol%</td></tr><tr><td></td><td>Catalyst identity</td><td>Categorical</td><td>P1-L1, P2-L1, P1-L2, P1-L3, P1-L4, P1-L5, P1- L6, P1-L7</td></tr><tr><td>iPSC-RPE</td><td>PC</td><td>Ordered finite support</td><td>54 finite levels, 12.47–505.62</td></tr><tr><td></td><td>PP; DS; KP; 3P</td><td>Integer</td><td>[1, 6]; [10, 100]; [1, 19]; [3, 19]</td></tr><tr><td></td><td>DP; DL</td><td>Ordered finite support</td><td>{5, 8, 11, 14, 17, 20, 23}; {0, 1}</td></tr><tr><td>C2C12</td><td>Media components 1–14</td><td>Continuous</td><td>Each [0, 1] in adapter-normalized coordinates</td></tr><tr><td></td><td>Information source</td><td>Finite support</td><td>{0, 1, 2, 3}; retained from the historical offline</td></tr><tr><td></td><td>Sample/additional/tubing</td><td></td><td>protocol</td></tr><tr><td>HPLC</td><td>volume Sample flow; push speed;</td><td>Continuous</td><td>[0, .08]; [0, .06]; [.1, .9] ml</td></tr></table>

Table 7: Domains used by the actual adapters. Summit bounds and catalyst levels follow benchmark definitions (Felton et al., 2021); HPLC bounds follow the Olympus dataset configuration (Häse et al., 2021). iPSC concentration levels follow the original task-code definition; the adapter domains are not assumed exhaustive physical domains. C2C12 coordinates follow its adapter preprocessing.

Case 3, seed 0, search round 8 (zero-based). Values and IDs are retained from the saved log.

```jsonl
{
"id": "llm_h_6_14",
"parameters": ["temperature"],
"anchor_observation_id": "obs_000000",
"evidence_pair_ids": [
"pair_temperature_0000_0011"],
"confidence": 0.6399999999999999,
"validation_status": "partially_supported",
"tested_count": 1,
"success_count": 1,
"last_test_credit_weight": 0.5,
"last_test_improvement": 5.174800627842362,
"global_improvement_count": 0,
"hypothesis_value_status":
"locally_supported"
}
```

The same round proposed exploitation but resolved to mixed under weak reliability. The identified card and its recorded test outcomes illustrate the state objects that guide search and retain comparative support; local support remains distinct from global optimization progress.

## A.5 API Settings and Recorded Usage

The requested backend is gpt-4o-mini; the reflector submits its assembled instruction and evidence as a user message, at temperature zero with a 1500- token output cap and one configured retry. It does not explicitly set top\_p or an API sampling seed. Optimization seeds and the feedback-permutation seed are separate from API randomness. Temperature zero does not guarantee identical cloud responses; deterministic curation and validation refer to fixed recorded inputs, not the entire optimizer.

<table><tr><td>Task</td><td>Seed</td><td>Prompt tokens</td><td>Completion tokens</td><td>Elapsed (s)</td></tr><tr><td>iPSC</td><td>0</td><td>347576</td><td>10226</td><td>177.5</td></tr><tr><td></td><td>1</td><td>332143</td><td>10664</td><td>144.0</td></tr><tr><td></td><td>2</td><td>368290</td><td>10319</td><td>144.7</td></tr><tr><td></td><td>3</td><td>346650</td><td>10845</td><td>145.7</td></tr><tr><td></td><td>4</td><td>346886</td><td>10709</td><td>127.8</td></tr><tr><td></td><td>5</td><td>337694</td><td>9570</td><td>113.4</td></tr><tr><td>C2C12</td><td>0</td><td>412771</td><td>11030</td><td>175.7</td></tr><tr><td></td><td>1</td><td>365428</td><td>12230</td><td>178.8</td></tr><tr><td></td><td>2</td><td>370151</td><td>10958</td><td>156.0</td></tr><tr><td></td><td>3</td><td>361268</td><td>10461</td><td>161.2</td></tr><tr><td></td><td>4</td><td>457476</td><td>11976</td><td>168.6</td></tr><tr><td></td><td>5</td><td>436654</td><td>12364</td><td>174.1</td></tr></table>

Table 8: Recorded full-FGSS reflector usage per primary-task run, with twenty final call records each. Elapsed is the sum of saved reflector-call latencies, not total optimizer wall-clock time.

Each primary task has 120 final reflector records marked as successful LLM parses, with no logged fallback. Summed recorded reflector usage is 2,079,239 prompt and 62,333 completion tokens for iPSC, and 2,403,748 prompt and 69,019 completion tokens for C2C12. These are reflector-only usage totals, not total optimizer compute or billing. Final call records do not supply a complete count of transient failed requests or retries.

Historical records do not identify exact served model snapshots or per-request timestamps; temperature-zero calls do not guarantee bitwise cloud-API replay.

## A.6 Non-LLM Baseline Implementations and Trajectories

Random Search samples numerical bounds or finite legal levels uniformly. HEBO uses version 0.3.5, with the imported initialization count determining its startup size. In the primary tasks its GP fitter uses Adam instead of the default pSGLD after a numerical failure in C2C12 smoke; other model settings are retained. Ordered finite levels are encoded by index and unordered labels as categories. This fitting change is not applied to the Summit runs, which retain the package defaults. Cross-domain differences therefore cannot be attributed solely to task domain.

GP-EI uses BoTorch 0.8.5 with standardized outcomes and range-normalized numerical coordinates. It uses a categorical covariance component for iPSC pipetting length, C2C12 source, and Reizman catalyst identity. Acquisition optimization uses ten restarts and 256 raw samples; continuous optimization has a maximum of 200 iterations. Fully discrete iPSC uses legal discrete local search, C2C12 enumerates source labels, and Summit uses continuous or mixed optimization as appropriate. No wrapper-level random fallback replaces failed fitting. Repeated evaluations consume budget, and all methods reuse the actual paired initialization.

## A.7 Paired Primary-Task Differences

Table 9 reports FGSS-BO minus the matched LLM and GP-EI references. Intervals use 10,000 paired bootstrap resamples of the six run-level differences (seed 42) and are exploratory, without multiplicity correction.

## A.8 Cross-Domain Summit Evaluation

We evaluate five Summit tasks. SnAr is a continuous kinetic-flow simulator; Reizman Cases 1–4 are continuous/categorical reaction emulators with catalyst choices. Each method shares eight initial observations from the LLM comparison, followed by twenty search evaluations, over six paired seeds. Table 10 compares all five methods. HEBO and GP-EI have higher sample mean final best and search AUC than FGSS-BO and LLAMBO on every task. FGSS-BO exceeds its matched LLM reference on four final-best means and three AUC means, but underperforms it on Case 4.

Objectives and oracle implementation. All five objectives are maximized. SnAr uses $\mathrm { S T Y / 1 0 ^ { 4 } }$ $E _ { \mathrm { f a c t o r } } / 1 0 0$ , the sign reversal of the benchmark tutorial’s minimization objective. Space-time yield (STY) is measured in $\mathrm { k g \ m ^ { - 3 } h ^ { - 1 } }$ ; E-factor is the waste-to-product mass ratio. The scalar score combines throughput and waste, rather than reporting reaction yield. Reizman Cases 1–4 maximize predicted yield in percent, with responses clipped to [0, 100]. They share three continuous controls and an eight-level catalyst variable; names, units, and bounds appear in Table 7.

The downloaded Summit source distribution declares version 0.8.9. Our adapter implements the zero-noise SnAr kinetics with SciPy’s solve\_ivp and reads the official JSON preprocessing statistics and five fixed PyTorch predictors for each reizman\_suzuki\_case\_1–4 model directly. No emulator is retrained or additional observation noise sampled. Run seeds control the paired Latin hypercube initialization, category balancing, and candidate sampling; the oracle remains fixed, and cloud LLM randomness is separate. An offline replay of all 1680 saved responses from the 60 LLM runs agrees with these assets within $1 . 5 \times 1 0 ^ { - 1 4 }$ Metrics are reported per task without treating declared output bounds as verified global optima.

## A.9 Exact-Grid iPSC-RPE Control

The exact-grid variant restricts candidates to observed experimental configurations and does not assign new configurations nearest-neighbor responses. LLAMBO obtains final best 0.840±0.026 and search $\operatorname { A U C } 0 . 7 5 0 \pm 0 . 0 0 9 ; \operatorname { F G S S - B O }$ obtains $0 . 8 4 8 \pm 0 . 0 6 4$ and $0 . 7 1 0 \pm 0 . 0 9 6$ . The final-best means are close and the AUC advantage is not preserved. This control changes the admissible candidate set as well as removing nearest-neighbor assignment, so it does not isolate oracle effects from candidate-space restriction.

## A.10 Olympus HPLC Control

The continuous-parameter Olympus HPLC benchmark (Häse et al., 2021) provides a further experimental-design task. Across six paired runs, LLAMBO obtains final best $1 8 0 3 . 7 2 0 \pm 5 2 1 . 8 2 7$ and search AUC $1 6 7 7 . 3 6 5 \pm 4 6 5 . 7 8 3 $ ; FGSS-BO obtains $1 9 6 2 . 6 5 1 \pm 2 3 7 . 1 3 1$ and $1 7 6 5 . 8 4 6 \pm$ 381.965. FGSS-BO has higher sample means on both metrics; this comparison covers the two LLM pipelines.

![](images/92e6d88b5458ec2c0a0d91e6ce1b6b852e600cd79f8d3ba79607b11d066063b2.jpg)  
Seeds 0-5; shared 10-point initialization + 20 search evaluations; bands show sample standard deviation.

Figure 4: Five-method primary-task trajectories, six paired runs with ten shared low-score initialization observations and twenty search evaluations. Lines are means and bands are sample standard deviations. All methods use the same nearest-neighbor response oracle; bands are not confidence intervals. LLAMBO uses the shared feasibility preserving interface.
<table><tr><td>Task</td><td>Reference</td><td>Metric</td><td>Mean difference</td><td>Bootstrap 95% interval</td></tr><tr><td>iPSC-RPE</td><td>GP-EI</td><td>Final best</td><td>0.020</td><td>[-0.009, 0.050]</td></tr><tr><td>iPSC-RPE</td><td>LLAMBO</td><td>Final best</td><td>0.123</td><td>[0.046, 0.205]</td></tr><tr><td>iPSC-RPE</td><td>GP-EI</td><td>Search AUC</td><td>0.135</td><td>[0.014, 0.254]</td></tr><tr><td>iPSC-RPE</td><td>LLAMBO</td><td>Search AUC</td><td>0.147</td><td>[0.054, 0.252]</td></tr><tr><td>C2C12</td><td>GP-EI</td><td>Final best</td><td>0.210</td><td>[-0.271,0.691]</td></tr><tr><td>C2C12</td><td>LLAMBO</td><td>Final best</td><td>0.445</td><td>[0.110, 0.959]</td></tr><tr><td>C2C12</td><td>GP-EI</td><td>Search AUC</td><td>0.299</td><td>[-0.248,0.779]</td></tr><tr><td>C2C12</td><td>LLAMBO</td><td>Search AUC</td><td>0.141</td><td>[−0.134,0.419]</td></tr></table>

Table 9: Paired primary-task differences, FGSS-BO minus the reference; positive favors FGSS-BO. LLAMBO uses the shared feasibility-preserving interface.

## A.11 Component-Ablation Protocol and Paired Differences

All four iPSC variants share ten low-score initial observations, twenty search evaluations, the legal parameter domains, and the nearest-neighbor response oracle. The shared LLM settings are GPT-4o mini, ten candidates, two templates, and ten generations. The two reflective controls and full FGSS use a reflection output cap of 1500 tokens, temperature zero, and one configured retry. This matches call settings, not actual tokens or total compute with LLAMBO.

Reflective Summary. Each round, the same structured evidence extractor supplies current evidence alongside the previous prose summary. The LLM replaces that summary with updated guidance on promising directions, contradictions, uncertainty, coverage, and exploration. The summary conditions the shared candidate refiner, without hypothesis cards, deterministic hypothesis curation, matched-test generation, direct support updates, or test-policy overrides. Both this control and FGSS maintain cross-round memory under the shared reflection settings.

FGSS w/o Tests. Structured reflection, curation, search-regime gating, and ordinary state-guided candidates remain enabled. Matched-test generation, test-policy overrides of EI, and direct-test support updates are disabled together. This is a testing-bundle ablation, not an isolation of selection overrides alone. All twenty selections use the shared EI path. Realized full-FGSS selection changes and control-match quality appear in Appendix A.2.

Reference reuse and execution audit. The LLAMBO and full-FGSS references reuse the primary iPSC runs. A source-compatibility audit checked the operative extractor, curator, reflector instructions, phase gate, candidate policies, surrogate, and oracle paths; subsequent additions were optional interfaces and logging. Paired initialization, objective outcomes, and direct-control scores were also checked. All runs completed the required thirty evaluations and remain in the analysis, including the run with one FGSS w/o Tests reflection (seed 4, search round 19, one-based) that used the recorded rule fallback after a connection failure.

<table><tr><td>Task</td><td>Metric</td><td>Random Search</td><td>HEBO</td><td>GP-EI</td><td>LLAMBO</td><td>FGSS-BO</td></tr><tr><td>SnAr</td><td>Final best</td><td> $0 . 5 9 3 \pm 0 . 0 7 4$ </td><td> $1 . 0 5 1 \pm 0 . 0 0 5$ </td><td> $1 . 0 5 8 \pm 0 . 0 0 1$ </td><td> $0 . 6 4 1 \pm 0 . 1 0 7$ </td><td> $0 . 6 8 0 \pm 0 . 0 5 7$ </td></tr><tr><td>SnAr</td><td>Search AUC</td><td> $0 . 5 7 2 \pm 0 . 0 7 8$ </td><td> $0 . 9 6 7 \pm 0 . 0 4 8$ </td><td> $0 . 9 9 0 \pm 0 . 0 4 9$ </td><td> $0 . 5 7 9 \pm 0 . 1 0 1$ </td><td> $0 . 5 7 3 \pm 0 . 1 2 3$ </td></tr><tr><td>Reizman Case 1 Final best Reizman Case 1 Search AUC</td><td></td><td> $6 7 . 5 0 5 \pm 1 0 . 4 4 3$   $6 4 . 0 3 3 \pm 9 . 6 7 2$ </td><td> $8 6 . 4 8 0 \pm 6 . 8 8 7$   $7 4 . 6 7 3 \pm 9 . 3 1 7$ </td><td> $9 1 . 4 2 4 \pm 0 . 0 1 9$   $8 3 . 3 1 5 \pm 5 . 2 9 1$ </td><td> $6 0 . 0 7 1 \pm 1 5 . 7 9 2$   $5 5 . 0 8 7 \pm 1 3 . 7 5 1$ </td><td> $6 9 . 7 5 1 \pm 1 4 . 7 6 4$   $6 5 . 7 3 8 \pm 1 5 . 4 8 4$ </td></tr><tr><td>Reizman Case 2 Final best Reizman Case 2 Search AUC</td><td></td><td> $2 1 . 6 0 2 \pm 6 . 8 8 2$   $1 9 . 9 1 6 \pm 6 . 7 1 9$ </td><td> $3 7 . 1 5 1 \pm 6 . 8 1 7$   $3 2 . 1 6 6 \pm 6 . 3 0 7$ </td><td> $4 0 . 0 8 5 \pm 0 . 0 0 0$   $3 2 . 7 6 2 \pm 4 . 7 7 7$ </td><td> $2 0 . 9 1 0 \pm 7 . 1 2 4$   $1 9 . 1 6 4 \pm 6 . 7 2 1$ </td><td> $2 2 . 2 8 6 \pm 9 . 0 1 6$   $2 1 . 2 6 2 \pm 8 . 6 6 2$ </td></tr><tr><td>Reizman Case 3 Final best</td><td></td><td> $7 9 . 5 4 3 \pm 5 . 8 8 7$ </td><td> $8 9 . 5 3 6 \pm 2 . 6 0 4$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $9 2 . 7 7 0 \pm 0 . 2 4 1$ </td><td> $7 3 . 9 6 4 \pm 8 . 4 5 2$ </td><td> $7 6 . 6 0 6 \pm 4 . 9 2 1$ </td></tr><tr><td>Reizman Case 3 Search AUC</td><td></td><td> $7 2 . 0 9 4 \pm 3 . 8 0 0$ </td><td> $8 5 . 1 1 5 \pm 3 . 1 1 4$ </td><td> $8 4 . 7 2 3 \pm 4 . 5 8 9$ </td><td> $6 8 . 1 5 2 \pm 3 . 4 6 6$ </td><td> $7 1 . 7 2 6 \pm 2 . 1 8 3$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reizman Case 4 Final best</td><td></td><td> $9 0 . 2 1 6 \pm 3 . 3 6 6$ </td><td> $9 5 . 5 4 1 \pm 2 . 6 1 5$ </td><td> $9 6 . 6 8 8 \pm 0 . 0 3 4$ </td><td> $9 2 . 0 0 0 \pm 4 . 1 2 4$ </td><td> $8 9 . 8 6 6 \pm 3 . 5 6 1$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reizman Case 4 Search AUC</td><td></td><td> $8 8 . 0 7 9 \pm 2 . 9 9 2$ </td><td> $9 3 . 9 1 0 \pm 2 . 0 9 3$ </td><td> $9 4 . 2 4 3 \pm 1 . 4 5 0 $ </td><td> $8 9 . 9 3 9 \pm 2 . 6 5 1$ </td><td> $8 8 . 5 0 5 \pm 3 . 5 9 7$ </td></tr></table>

Table 10: Cross-domain Summit results: five methods over six paired runs, with eight initialization and twenty search evaluations. Values are mean ± sample standard deviation; higher is better. LLAMBO uses the shared feasibility-preserving interface.

Table 11 uses 50,000 paired bootstrap resamples with analysis seed 20261012. The intervals are exploratory percentile intervals without multiplicity correction. Full FGSS has a higher observed final-best mean than the summary, but no AUC advantage; removing the test bundle has mixed effects. The contrast compares complete prose-summary and structured pipelines, which differ in state representation, candidate control, and test management.

## A.12 State-Feedback Permutation on iPSC-RPE

separately under the same experimental settings. The audit verifies paired initialization and true surrogate-history routing, and all 30 direct support updates agree with their single-view comparisons.

This control retains the full FGSS mechanism but permutes historical outcomes across configurations only for state construction and updates. Configurations, initialization, domains, and evaluation budgets are unchanged. LLAMBO’s surrogate and the shared candidate refiner retain true observation history; the refiner also receives the perturbed state guidance. The permutation is refreshed at preand post-evaluation stages, with feedback seed 42. Each eligible support comparison uses the same post-evaluation view for both intervention and control. All reported performance metrics use true oracle outcomes.

Figure 5 includes all six paired seeds 0–5. The true-FGSS and LLAMBO references reuse the primary runs; the permutation runs were completed

State-feedback permutation achieves faster early progress and a higher mean search AUC, whereas aligned feedback achieves a higher mean final objective with lower across-seed variability in final outcomes (Table 12; Figure 5). This pattern is consistent with a trade-off between early progress and final-result consistency. The true-minus-permuted mean differences are 0.0275 for final best and −0.0349 for AUC; their exploratory paired bootstrap 95% intervals are $[ - 0 . 0 2 3 2 , 0 . 0 8 5 0 ]$ and $\left[ - 0 . 1 2 3 6 , 0 . 0 6 3 1 \right]$ , respectively. Both include zero. The surrogate retains true observation history in both conditions.

## A.13 Test-Selection Ablation and Immediate Opportunity Cost

EI-only selection control. FGSS No-Override retains the full reflector, curator, state-guided candidate construction, matched-test candidates, and direct support updates. Only the final selection policy changes to ei\_only; the test-generation quota is retained. All six seeds 0–5 complete the same ten low-score initialization observations and twenty search evaluations. They share initialization, legal domains, oracle, and LLM settings with the references in Table 13. The audit checks all 120 selected configurations against the CSVs and confirms that each maximizes saved predicted EI. The pools contain 99 test-candidate appearances, and 22 selected tests receive a single-parent support update; all 22 control outcomes match prior observations and use distinct oracle identities. All 120 final reflector records are successful LLM calls without logged fallback.

<table><tr><td>Contrast</td><td>Metric</td><td>Mean difference</td><td>Bootstrap 95% interval</td></tr><tr><td rowspan="2">Full FGSS — Reflective Summary</td><td>Final best</td><td>0.0285</td><td>[0.0050, 0.0617]</td></tr><tr><td>Search AUC</td><td>-0.0061</td><td>-0.0702, 0.0605</td></tr><tr><td rowspan="2">Full FGSS – FGSS w/o Tests</td><td>Final best</td><td>0.0088</td><td>-0.0370, 0.0730]</td></tr><tr><td>Search AUC</td><td>-0.0275</td><td>[-0.1264, 0.0651]</td></tr><tr><td rowspan="2">FGSS w/o Tests – Reflective Summary</td><td>Final best</td><td>0.0197</td><td>[-0.0527,0.0910]</td></tr><tr><td>Search AUC</td><td>0.0214</td><td>[-0.0310, 0.0822]</td></tr></table>

Table 11: Paired iPSC component differences over six runs. Positive favors the first method.

iPSC: State Feedback Control  
![](images/76cf828c5bcd7121f7db237eb4d5c14c72c56e419d7c9f16ff461bf705250a5f.jpg)  
Six paired seeds; lines: mean; shading: sample standard deviation; initialization: 10; search: 20.

Figure 5: iPSC-RPE with true or permuted state feedback, alongside the LLAMBO reference. Lines show mean best-so-far true objective over six paired runs; shading shows one sample standard deviation. The dashed boundary separates ten initialization observations from twenty search evaluations. Only FGSS-BO state feedback is permuted; the surrogate retains true observation history.
<table><tr><td>Condition</td><td>Final best</td><td>Search AUC</td></tr><tr><td>True</td><td> $. 8 1 3 2 \pm . 0 2 2 5$ </td><td> $. 6 4 0 1 \pm . 1 1 7 5$ </td></tr><tr><td>Permuted</td><td> $. 7 8 5 7 \pm . 0 7 2 6$ </td><td> $. 6 7 5 1 \pm . 0 6 5 4$ </td></tr></table>

Table 12: iPSC-RPE state-feedback control: mean ± sample standard deviation over six paired runs.

Full FGSS minus No-Override has mean paired differences +0.0117 in final best and −0.0495 in search AUC. The respective exploratory bootstrap 95% intervals are [−0.0712, 0.0965] and [−0.1343, 0.0324], from 50,000 paired-seed resamples with analysis seed 20261012. Both intervals include zero. Full FGSS has lower observed final-best dispersion, while No-Override has higher mean search AUC (Figure 6). No-Override’s higher sample means than LLAMBO show that gains occur without forced test selection in these runs. Collection periods and cloud API randomness remain uncontrolled.

Same-history, same-pool opportunity cost. For each of full FGSS’s 18 actual selection changes, we query the original fixed oracle after the fact on the recorded EI choice and selected test, without changing history or generating a new pool. Let $v _ { t } ^ { \mathrm { E I } } = \bar { f } _ { \mathrm { N N } } ( x _ { t + 1 } ^ { \mathrm { E I } } )$ and $v _ { t } ^ { \mathrm { t e s t } } = \widehat { f } _ { \mathrm { N N } } ( x _ { t + 1 } ^ { \mathrm { t e s t } } )$ , with $b _ { t }$ the incumbent before that evaluation. Define

$$
\begin{array} { r l } & { g _ { t } ^ { \mathrm { r e s p } } = v _ { t } ^ { \mathrm { E I } } - v _ { t } ^ { \mathrm { t e s t } } , } \\ & { g _ { t } ^ { \mathrm { b e s t } } = \operatorname* { m a x } ( b _ { t } , v _ { t } ^ { \mathrm { E I } } ) } \\ & { \quad \quad - \operatorname* { m a x } ( b _ { t } , v _ { t } ^ { \mathrm { t e s t } } ) . } \end{array}\tag{14}
$$

<table><tr><td>Method</td><td>Final best</td><td>Search AUC</td><td>Selection changes</td><td>Direct support updates</td></tr><tr><td>LLAMBO</td><td> $0 . 6 9 0 5 \pm 0 . 1 1 6 8$ </td><td> $0 . 4 9 3 6 \pm 0 . 0 5 7 1$ </td><td>0</td><td>0</td></tr><tr><td>Full FGSS</td><td> $0 . 8 1 3 2 \pm 0 . 0 2 2 5$ </td><td> $0 . 6 4 0 1 \pm 0 . 1 1 7 5$ </td><td>18</td><td>37</td></tr><tr><td>FGSS No-Override</td><td> $0 . 8 0 1 5 \pm 0 . 1 1 0 3$ </td><td> $0 . 6 8 9 6 \pm 0 . 0 9 8 0$ </td><td>0</td><td>22</td></tr></table>

Table 13: iPSC selection-policy control over six paired runs: $\mathrm { m e a n } \pm \mathrm { s a m p l e }$ standard deviation. Counts are totals over 120 search evaluations per method. Selection changes mean departures from the recorded EI choice, not tests naturally selected by EI. Full FGSS and LLAMBO reuse the audited primary references and shared feasibility-preserving interface.

![](images/dc9a52da5a9f4a0ef0bd8064d540ebde44495eaee83db7374f0244bc342e0a58.jpg)  
All seeds 0-5; shared 10-point initialization and 20 search evaluations. Bands/bars: sample SD; red intervals: exploratory paired bootstrap.  
Figure 6: iPSC No-Override ablation, all six paired seeds. Curves and bars show means; shading and bar error bars show sample standard deviations. The right panel reports full FGSS minus No-Override: numbered points are individual seeds, and red intervals are exploratory paired bootstrap 95% intervals. No-Override retains hypothesistest generation and support updates but always selects by EI. LLAMBO uses the shared feasibility-preserving interface.

Positive values favor the EI alternative; negative values favor the test. Replaying all 180 recorded responses (initialization included) reproduces the historical scores, and all selected-test scores and saved EI maxima agree with their logs. No LLM calls or new laboratory evaluations are performed.

Across all 18 events, mean response gap is 0.0563: EI has higher response in 11 events and the test in seven. Mean one-step incumbent gap is 0.0854: EI is higher in eight events, the test in five, with five ties (Figure 7). These are eventweighted means; whole-seed cluster bootstrap, rather than independent-event resampling, gives exploratory 95% intervals [−0.0433, 0.1932] and [0.0001, 0.1964], respectively. Intervals use all events and six seed clusters, without multiplicity adjustment.

As a post-hoc descriptive partition, the six first-search-round changes all favor EI immediately, with mean gap 0.3540 for both measures. The remaining twelve events have mean response/incumbent gaps − $\cdot 0 . 0 9 2 6 / - 0 . 0 4 8 9$ . All 18 events remain included in the aggregates. This audit compares one-step outcomes under a fixed oracle; it does not estimate downstream effects.

## A.14 Freeform and Feasible Candidate Interfaces

Figure 8 compares LLAMBO-Freeform with the existing LLAMBO-Feasible references on iPSC-RPE and C2C12, using all seeds 0–5. All 24 runs complete ten initialization and twenty search evaluations with batch size one. Paired initial configurations and outcomes, parameter domains, and task-specific nearest-neighbor protocols agree. The Freeform runs use GPT-4o mini with ten candidates, two templates, and ten generations; neither method uses FGSS state or a reflector.

On iPSC, Freeform obtains 0.346 in both final best and search AUC, versus Feasible’s 0.691 ± 0.117 and $0 . 4 9 4 \pm 0 . 0 5 7$ . All 120 Freeform search rounds exhaust the free-form proposal path and select recovery-fallback candidates; the six evaluation CSVs are identical. Its zero dispersion reflects coincident fallback trajectories, not reliable LLM optimization. On C2C12, no search round invokes recovery fallback: Freeform obtains $1 . 2 8 7 \pm 0 . 6 7 3$ and $1 . 0 5 8 { \pm } 0 . 5 8 3$ , versus Feasible’s $1 . 2 8 1 \pm 0 . 2 2 0$ and $1 . 0 2 5 \pm 0 . 2 6 4$ . Their sample means are close, with larger Freeform dispersion. Search AUC excludes initialization.

![](images/741c90c1f94b003c6ade26548f07962f559c25628ae38de8926618f870e1a298.jpg)  
Positive gaps favor EI; negative gaps favor the test. Same recorded history and pool; no counterfactual continuation. xN denotes overlapping response pairs.  
Figure 7: Immediate opportunity costs of all 18 full-FGSS selection changes under the fixed iPSC oracle. The left panel compares the alternative response scores; the right reports EI minus test response and one-step incumbent gaps, indexed by seed and one-based search round. Positive gaps favor EI. Each comparison uses the same recorded history and candidate pool.

These controls compare complete candidate interfaces. Feasible includes candidate refinement; Freeform retains validity checks, deduplication, retries, and recovery. The latter is an implementationlevel free-form variant, not a strict reproduction of original LLAMBO. The new Freeform runs record their settings, whereas older Feasible references lack complete per-run CLI provenance; collection periods and cloud API responses are not controlled by initialization pairing. Feasibilitypreserving construction addresses iPSC’s severe proposal rejection, while C2C12’s viable Freeform path yields sample means close to Feasible. Both main LLM methods use the same feasible interface across tasks; Freeform is reported separately as a candidate-interface control.

## A.15 Direct-Training Bayesmark HPO Transfer

The suite was fixed before evaluation: breast cancer, handwritten digits, and diabetes datasets, each paired with RandomForest and SVM. All six tasks and both methods are retained. There are 72 completed runs and 2160 configuration evaluations, with six paired seeds, ten initialization points, twenty search evaluations, and batch size one. Table 14 reports CV-based optimization metrics. The suite compares the two complete LLM pipelines.

Objective and preprocessing. Each configuration trains a scikit-learn model directly with fivefold cross-validation. Classification maximizes accuracy, using stratified folds; regression minimizes positive MSE. We retain the downloaded

LLAMBO train/test splits and feature preprocessing. Diabetes targets are standardized once using the training split’s mean and standard deviation before CV, following that runner; this is not a foldwise target-scaling protocol. The held-out test set uses the same training-target scaling and is used only for reporting, never for candidate generation, surrogate prediction, or state updates.

RandomForest uses ten trees, random state zero, and no fixed maximum leaf count. Classification SVM uses an RBF SVC with probability estimation enabled and random state zero; regression uses an RBF SVR. Folds and training randomness are fixed. The optimization seeds index initialization configurations and candidate randomness, not independent training splits.

Initialization and representation. Both methods use the first ten public initialization configurations for their model/seed from the LLAMBO repository, canonicalized to eight decimal places without inspecting outcomes. The HPO spaces are copied from its Bayesmark configuration, with sampling scales in Table 15. Global and local candidate sampling respect linear, log, or logit coordinates; evidence distances remain raw range-normalized. Log-variable test steps use log-range increments. Both methods serialize candidate floats to eight decimals and outcomes to six, preventing small impurity values from collapsing under the legacy formatter. The shared interface follows the feasibilitypreserving baseline, not an exact replication of freeform LLAMBO.

Both methods use the main shared LLM settings; full FGSS additionally uses the LLM reflector and bounded test policy. Feedback is true throughout, and no optional domain-prior card is enabled. CSV/JSON metrics, objective directions, budgets, and paired initialization were checked. The recorded independent audit retrained the 72 best configurations and reproduced their CV and held-out scores to floating-point precision.

![](images/aaccd84b4db53946eeebf01cf3982e4bda8ac96f57cabaf3f7e9240b5f3b5673.jpg)

Six paired seeds; lines: mean; shading: sample standard deviation; initialization: 10; search: 20.  
![](images/711f572e450ed27782eefc7aeeeafb0d1005a18a4a524a11b5215af459711af3.jpg)

Figure 8: Candidate-interface controls on both primary tasks over all six paired seeds. Lines are means; shading is sample standard deviation. The dashed line separates ten shared low-score initial observations from twenty search evaluations. iPSC Freeform uses recovery fallback in all 120 search rounds, producing six identical evaluation trajectories; C2C12 has no recovery fallback. Both methods omit FGSS state.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Direction</td><td colspan="2">Final best</td><td colspan="2">Search AUC</td></tr><tr><td>LLAMBO</td><td>FGSS-BO</td><td>LLAMBO</td><td>FGSS-BO</td></tr><tr><td>Breast RF</td><td>Accuracy ↑</td><td> $. 9 3 1 8 7 \pm . 0 0 3 4 0$ </td><td> $. 9 3 1 8 7 \pm . 0 0 3 6 8$ </td><td> $. 9 3 1 6 1 \pm . 0 0 3 6 5$ </td><td> $. 9 3 1 5 2 \pm . 0 0 3 8 8$ </td></tr><tr><td>Breast SVM</td><td>Accuracy ←</td><td> $. 9 4 2 8 6 \pm . 0 0 1 9 7$ </td><td> $. 9 4 4 3 2 \pm . 0 0 2 6 6$ </td><td> $. 9 4 2 8 2 \pm . 0 0 2 0 2$ </td><td> $. 9 4 3 7 0 \pm . 0 0 2 5 6$ </td></tr><tr><td>Digits RF</td><td>Accuracy ↑</td><td> $. 5 3 5 5 0 \pm . 2 5 6 0 7$ </td><td> $. 5 2 7 2 7 \pm . 2 9 3 3 9$ </td><td> $. 4 6 0 3 3 \pm . 3 0 3 7 6$ </td><td> $. 4 7 3 7 6 \pm . 3 1 5 0 6$ </td></tr><tr><td>Digits SVM</td><td>Accuracy ↑</td><td> $. 9 9 2 4 6 \pm . 0 0 0 5 2$ </td><td> $. 9 9 2 5 8 \pm . 0 0 0 3 6$ </td><td> $. 9 9 2 3 6 \pm . 0 0 0 4 5$ </td><td> $. 9 9 2 3 5 \pm . 0 0 0 3 6$ </td></tr><tr><td>Diabetes RF</td><td>MSE↓</td><td> $. 5 8 5 5 8 \pm . 0 3 1 0 2$ </td><td> $. 5 7 1 0 9 \pm . 0 4 2 7 0$ </td><td> $. 6 1 0 2 4 \pm . 0 4 1 6 3$ </td><td> $. 6 1 3 3 5 \pm . 0 7 3 9 4$ </td></tr><tr><td>Diabetes SVM</td><td>MSE↓</td><td> $. 5 7 7 6 8 \pm . 0 5 2 1 9$ </td><td> $. 5 1 7 1 1 \pm . 0 0 9 8 4$ </td><td> $. 6 1 0 4 6 \pm . 0 8 6 8 4$ </td><td> $. 5 5 6 3 6 \pm . 0 5 0 8 0$ </td></tr></table>

Table 14: Complete six-task HPO transfer comparison: means ± sample standard deviations over six paired runs per task. Accuracy is maximized; both diabetes metrics are minimized. RF denotes RandomForest. Raw accuracy and MSE are not aggregated. LLAMBO uses the shared feasibility-preserving interface.

<table><tr><td>Parameter</td><td>Domain</td><td>Scale</td></tr><tr><td>RandomForest</td></tr><tr><td>Max. depth [1, 15] [.01, .99]</td><td>Linear Logit</td></tr><tr><td>Max. feâture fraction</td></tr><tr><td>Min. impurity decrease [0, .5]</td></tr><tr><td>Linear Min. leaf sample fraction [.01, .49] Logit Min. split sample fraction</td></tr><tr><td>.01, .99] Logit Min. weighted leaf fraction [.01, .49] Logit</td></tr><tr><td>SVM</td></tr><tr><td>[1, 1000] Log</td></tr><tr><td></td></tr><tr><td>[.0001, .001] Log Tolerance</td></tr></table>

Table 15: HPO domains and sampling scales inherited from LLAMBO’s Bayesmark configuration, shared by both methods. Maximum depth is integer; all other parameters are continuous.

Diabetes SVM reduces mean final MSE/search

<table><tr><td>Task</td><td>Metric</td><td>Gain Bootstrap 95% interval</td></tr><tr><td rowspan="2">Breast RF</td><td>Final</td><td>0.00000</td></tr><tr><td>AUC -.00009</td><td>[-.00110,.00110] -.00066, .00038</td></tr><tr><td>Breast SVM</td><td>Final</td><td>.00147 [.00037, .00293] [-.00051, .00242</td></tr><tr><td rowspan="3">Digits RF</td><td>AUC</td><td>.00088</td></tr><tr><td>Final -.00824</td><td>-.13083, .09440</td></tr><tr><td>AUC .01343</td><td>-.06695, .07959</td></tr><tr><td rowspan="2">Digits SVM</td><td>Final</td><td>.00012</td></tr><tr><td>AUC</td><td>-.00023, .00046] -.00001 -.00026,.00026] -.01098,.03750]</td></tr><tr><td rowspan="2">Diabetes RF</td><td>Final</td><td>.01448</td></tr><tr><td>AUC -.00312</td><td>-.03414, .02294]</td></tr><tr><td rowspan="2">Diabetes SVM</td><td>Final</td><td>.06057 [.02708, .09707</td></tr><tr><td>AUC</td><td>.05411 [.02304, .08215]</td></tr></table>

Table 16: Paired HPO mean gains use unrounded perseed metrics. Final denotes final best; AUC denotes search AUC. Positive favors FGSS-BO: FGSS minus LLAMBO for accuracy, reversed for MSE. Exploratory intervals are task-specific and unadjusted for multiple comparisons. LLAMBO uses the shared feasibilitypreserving interface.

AUC by 10.48%/8.86%; other tasks show small, tied, or mixed effects. High initialization scores constrain gains on breast RF and digits SVM. Table 16 reports all paired differences with unadjusted exploratory percentile intervals (20,000 bootstrap resamples, analysis seed 42).