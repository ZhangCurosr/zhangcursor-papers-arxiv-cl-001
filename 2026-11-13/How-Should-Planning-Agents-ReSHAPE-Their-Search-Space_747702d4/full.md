# How Should Planning Agents ReSHAPE Their Search Space?

Yilin Zhang<sup>1</sup> Xinran Zhao<sup>1</sup> Tong Chen<sup>2</sup> Ziming Yuan<sup>1</sup> Tongshuang Wu<sup>1</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>University of Washington

Correspondence: {jasonzh3, sherryw}@cs.cmu.edu

## Abstract

Planning agents and benchmarks treat a user’s initial specification as fixed. Yet the requests users write exclude plans they would gladly take, since they rarely discover the possible trade-offs they would accept until seeing the options. Agents already revise requests through interaction, but whether this helps planning is unknown. We formalize this step as the ReSHAPE task, which judges a revision by whether its search space matches the acceptable plans and a planner finds one there. Using ReSHAPE Planter, which builds instances from planning benchmarks with a human-validated simulated user, we analyze 5,800 agent-revised specifications in travel and shopping. We find that reshaping matters: one correct trade-off makes the same planner reach an acceptable plan up to 4.6× as often, and search space quality predicts what the planner finds. Current agents, however, admit at most 32% of the acceptable plans, where that trade-off alone admits over 52%, because they ask about preferences instead of showing plans; in travel, showing plans raises F<sub>1</sub> by 9.7 points over asking, with two model families. Planning agents should thus treat a request as a hypothesis to test against the user’s reactions to concrete plans to recognize user trade-offs.

## 1 Introduction

Language-model agents are increasingly asked to turn a natural-language request into a concrete plan with many parts, such as a multi-day trip. The request is first turned into a specification, the conditions a plan must meet (Cook et al., 2024; Hao et al., 2025; Shao et al., 2025), and a planner then searches for a plan that satisfies it. Planning benchmarks follow the same pipeline: they evaluate whether the final plan satisfies the specification the user provides (Xie et al., 2024; Zheng et al., 2024; Xiao et al., 2024). A plan is therefore only as good as the specification it is planned against:

a planner cannot return a plan that the specification excludes, and it struggles to find a good plan in one that admits too much.

Yet a specification built from the user’s request is a poor guide to the plans they would accept, because people discover what they would accept only as they encounter concrete options (Bettman et al., 1998; Pirolli and Card, 2005; Kim et al., 2026). A traveler who asks for restaurants rated at least 3.8 may gladly accept lower-rated ones that cost less than half as much, but cannot say in advance how much lower is acceptable, since that depends on the restaurants actually on offer (Figure 1, top left). Such trade-offs, in which a strength on one aspect of a plan compensates for a weakness on another (Keeney and Raiffa, 1976), rarely make it into a request, so the initial specification excludes plans the user would accept.

Existing agents already try to close this gap through interaction. They ask clarifying questions before planning (Qian et al., 2024; Zhang et al., 2024), elicit preferences the user left unstated (Qian et al., 2025), or replan as new constraints arrive (Oh et al., 2025; Liu et al., 2026). The same step appears in commercial agents: OpenAI’s and Google’s Deep Research agents confirm the request or propose a research plan for the user to revise (OpenAI, 2025b; Citron, 2024). These settings, however, judge the plan or choice finally delivered, which conflates the revised request with the planner that acts on it (Shen et al., 2026), so whether a revision recovers the plans the request excluded goes unmeasured.

We want to know whether this revision step actually pays off for planning, what makes a revised specification good for a planner, and what an agent has to do to produce one. To answer these questions, we formalize this step as the ReSHAPE task (§3). Before the planner runs, an agent interacts with the user to revise the initial specification into one that includes the trade-offs the user would accept. The planner then searches only the plans the revised specification admits (Figure 1, top). The user is simulated with hidden preferences, so whether a plan is acceptable can be computed exactly, while the agent, as in practice, learns those preferences only through a limited budget of questions and shown plans. In our user study, the simulated user’s acceptance decisions agree with human participants’ as often as participants agree with one another (§4.1). A revision can then be scored at two levels (Figure 1, bottom left): upstream, by how its search space overlaps the acceptable plans, with recall for how many of the acceptable plans it admits and precision for how many of the admitted plans are acceptable; and downstream, by whether a fixed planner given the revision delivers an acceptable plan. Such scoring needs instances in which revision is both necessary and possible; we thus build ReSHAPE Planter, a pipeline that plants in each instance of a planning benchmark a trade-off that the initial specification excludes and a reference specification adds.

![](images/20a4fd711c970b884a70cece591401943cf96b89c358cdeceaee79bf7de36c90.jpg)  
Figure 1: ReSHAPE scores the revision step before planning. Top left: the request’s specification excludes Plan B, which the user would accept. Top: an agent revises the specification with the user, and the planner, searching the revised space, finds Plan B. Bottom left: the kinds of interaction found in agents’ logs, which we compare under a fixed feedback budget, and the metrics that score a revision: the recall and precision of its search space against the acceptable plans, and whether a fixed planner delivers an accepted plan. Bottom right: our main findings.

As a proof of concept, we apply ReSHAPE Planter to TravelPlanner (Xie et al., 2024) and to the shopping tasks of DeepPlanning (Zhang et al., 2026) to build ReSHAPE Travel and ReSHAPE Shopping, on which we analyze the revisions of 14 and 8 off-the-shelf agent systems, respectively; on ReSHAPE Travel we also compare four interaction protocols, each limited to one kind of interaction found in these agents’ logs and run with the same model and feedback budget. Our analysis then yields three findings about the step itself (§5.1; Figure 1, bottom right). First, reshaping the specification matters: with the reference specification, which adds a single trade-off, the same planner finds a plan the user would accept up to 4.6× as often as with the initial one. Second, search space quality predicts the planning outcome: in both domains, the F<sub>1</sub> of recall and precision tracks whether the planner delivers an acceptable plan, so a revision can be judged before any planner runs. Third, a good revision first admits more plans and then stays selective. Early on only recall matters: the initial specification admits almost no acceptable plans, so each one a revision lets back in raises the chance that the planner finds it. Once the planner has enough to choose from, more recall stops helping and precision decides, because a search space full of plans the user would reject sends the planner to the wrong ones.

Current agents prioritize precision but cannot recover recall (§5.2): they admit at most 32% of the acceptable plans, against over 52% for the reference specification, because they spend their effort asking about preferences that users rarely know before seeing the options. The systems rank nearly the same on ReSHAPE Shopping in precision and in what the planner finds (Table 3), so this shortfall belongs to the agents rather than to a specific domain. Showing the user plans outside the current specification helps mitigate the imbalance: with the same model and feedback budget, it raises $\mathrm { F _ { 1 } }$ by 9.7 points over asking (95% CI [5.0, 15.0]) with Gemini 3.6 Flash and by 11.2 points with Claude Sonnet 4.6. Current agents thus treat the request as a fact to clarify, when it should be a hypothesis to test against the user’s reactions to concrete plans.

In sum, we (1) make the revision step before planning measurable on its own, apart from the planner that follows it; (2) build ReSHAPE Planter and, with it, ReSHAPE Travel and ReSHAPE Shopping; and (3) show where current agents fall short in this step and that showing plans, not asking, is what recovers the plans a request excludes.

## 2 Related Work

Stated preferences omit trade-offs that concrete options reveal. Stated requirements typically take the form of separate cutoffs that reject an alternative as soon as one of them fails (Einhorn, 1970; Payne et al., 1993), whereas judgments of complete alternatives are compensatory (Keeney and Raiffa, 1976). Decision analysis therefore elicits trade-offs by having people compare specific alternatives rather than state them in the abstract (Keeney and Raiffa, 1976), and example-critiquing recommenders refine a user’s preference model from reactions to displayed items (Chen and Pu, 2012; Viappiani et al., 2006). We test whether this practice carries over to planning agents, comparing protocols that show the user concrete plans with protocols that ask about preferences under the same feedback budget.

Interactive planning revises requests but judges the delivered plan. Interactive planning settings let agents ask clarifying questions, elicit hidden preferences, or revise a plan after feedback (Qian et al., 2024; Zhang et al., 2024; Oh et al., 2025; Qian et al., 2025; Liu et al., 2026), so they already revise the request in effect, but they score the final plan or choice, and few show the user plans or involve trade-offs (Appendix A). DiscoverLLM (Kim et al., 2026) simulates users whose intents concretize as the model surfaces options, and finds that asking fails when users do not yet know what they want; like the settings above, it judges the final artifact. We instead hold the user’s acceptance of every plan fixed, which lets us score a revised specification itself over all legal plans and relate this score to what a planner finds within it.

Existing work revises specifications, but not by the plans they admit. When constraints conflict, relaxation methods choose small or low-priority corrections that restore feasibility (Narodytska et al., 2018; Eifler et al., 2022; Li et al., 2023); constraint acquisition learns intended constraints from queries, examples, or natural-language feedback (Belaid et al., 2022; Tsouros et al., 2023, 2025; Mechqrane et al., 2024); and work on objective misspecification treats a stated goal as a proxy that interaction can correct (He and Dragan, 2022; Mechergui and Sreedharan, 2024b,a). All judge a revision by whether it restores feasibility or recovers a single target specification. Closest is GATE, which tests whether elicited preferences predict a user’s decisions on held-out examples (Li et al., 2025); we carry this idea to planning, judging a revision by how well its search space matches the plans the user would accept (§3.3) and testing which way of scoring this match best predicts what a planner finds (§5.1).

## 3 Making Specification Revision Measurable

ReSHAPE inserts a revision step before the planner: given the same input as planning, an agent interacts with the user and returns a revised specification instead of a plan. This section defines the step (§3.1), the simulated user that makes acceptance computable (§3.2), and how we measure a revision (§3.3); §4 introduces ReSHAPE Planter, which builds instances from planning benchmarks.

## 3.1 ReSHAPE: Revision before Planning

Background: plans and specifications. A planning task starts from a user’s request and the available options, grouped into categories such as flights, hotels, and restaurants. Current pipelines turn the request into a specification that lists conditions on plan attributes, such as a minimum restaurant rating or a maximum meal cost (Hao et al., 2025; Shao et al., 2025), and a planner searches for a complete plan satisfying every condition.

Intuitively, a request carries two kinds of content. Part of it is fixed, such as the destination, the dates, and the size of the group; we call this part the query $q ,$ and the complete plans that respect it the legal plans P. The rest consists of preferences, such as the rating: stated as if they were equally strict rules, but most of them the user would readily trade for something they value more. We collect them in the initial specification $C _ { 0 }$ , writing $c _ { i }$ for a condition on attribute i. A planner treats both parts alike, so it considers only the legal plans that also satisfy the specification, the search space $S ( C ) = \{ p \in { \mathcal { P } }$ | $p$ satisfies C}.

The ReSHAPE task. We define ReSHAPE as the step before the planner. It takes the same input as planning: the query, its options, and the initial specification $C _ { 0 } .$ , and instead of a plan it returns a revised specification $C ^ { \prime }$ that the planner follows.

The plans the user would actually accept form another subset of the legal plans, the acceptable plans ${ \mathcal { A } } ,$ which does not necessarily coincide with the search space: the request was written before the user saw any concrete option, and users discover which plans they would accept only as they see them (§1, Figure 1). The task is to return a $C ^ { \prime }$ whose search space matches A as closely as possible, including plans the user never saw.

Ground truth A and the expected $C ^ { \prime } .$ . To make revision measurable, we need a ground truth: for every legal plan, whether the user would accept it. We define A from how people actually decide, with three literature-grounded properties. First, a strength on one attribute can offset a weakness on another (Keeney and Raiffa, 1976), so no fixed threshold per attribute separates the plans the user accepts from those they reject, and A generally cannot be written as a conjunction of separate conditions (Figure 1). Second, the user accepts nothing worse than what the request already secures, up to a small tolerance below which plans are indistinguishable (Luce, 1956), so acceptance measures what a revision adds beyond $C _ { 0 }$ . Third, a few conditions would be deal-breakers that nothing offsets (Einhorn, 1970); they are stated in $C _ { 0 }$ like any other preference, and the agent is not told which they are. We realize A through a simulated user that fixes these judgments for every plan yet reveals them only through interaction (§3.2).

Since A is compensatory, a revised specification has to be able to express a trade-off, which a plain conjunction cannot. Beyond loosening or tightening the conditions in $C _ { 0 } , C ^ { \prime }$ may use the trade-off form $c _ { i } \vee ( c _ { i } ^ { - } \wedge c _ { j } )$ : it also admits plans that violate $c _ { i }$ but satisfy a strictly relaxed version $c _ { i } ^ { - }$ provided that a compensating condition $c _ { j }$ on another attribute holds (Appendix C.2). For example, a trade-off item can admit restaurants rated at least 3.8, or at least 3.3 if meals average under \$30.

## 3.2 Simulated User and Interaction

Hidden profile. Each instance holds a hidden profile that decides every question or shown plan, so that the user’s judgments never reach the agent directly. The agent only sees that decision verbalized by a language model, which never sees the profile and answers as a user would, without numbers or thresholds (Appendix D.4).

Utility and acceptance. The profile realizes the three properties of §3.1 through a utility $U ( p )$ over complete plans. For compensation, $U$ is a multi-attribute utility (Keeney and Raiffa, 1976) that rewards doing better than the user expects on an attribute and penalizes falling short, so a gain on one attribute can make up for a loss on another (Appendix D.1). For reference dependence, we take the best plan the initial specification admits, $p _ { 0 } ^ { * } .$ , as the reference point, and allow a tolerance of $\delta = 0 . 0 5$ below it, which gives the acceptance threshold $\tau = U ( p _ { 0 } ^ { * } ) - \delta ; C _ { 0 }$ therefore always admits an acceptable plan, and revision is needed to reach the rest. For the deal-breaker, we mark one condition in $C _ { 0 }$ as non-negotiable: a plan that violates it is rejected whatever its utility (Appendix D.3). The acceptable plans are $\mathcal { A } = \{ p \in$ $\mathcal { P } \mid p$ meets the deal-breaker and $U ( p ) \geq \tau \}$

Interaction. Since the profile is hidden, the only way for an agent to perform ReSHAPE is to converse with the user: it can ask a preference question or show a plan and hear the user’s judgment of it (Figure 1). To stay practical, we guard the conversation with a feedback budget B, in line with real users’ limited willingness to iterate (Shen et al., 2026): each answer and each judgment costs one unit, while inspecting options and checking candidate plans against the specification cost nothing (Appendix D.4).

## 3.3 Measuring a Revision

Intrinsic: search space quality. We score $S ( C ^ { \prime } )$ against A as a binary classification over the legal plans, where a plan is a true positive if $C ^ { \prime }$ admits it and the user accepts it. Plans do not count equally, though: the user is nearly indifferent about a plan whose utility sits near the threshold τ, and cares a great deal about one far above or far below it. We therefore measure recall, how much of what the user would accept the revision admits, and precision, how much of what it admits the user would accept, in plan weight rather than plan count. A plan’s weight is its distance from the threshold, $w ( p ) = | U ( p ) - \tau |$ . Writing $w ( X )$ for the total weight of a set of plans, weighted recall is $R _ { w } = w ( S ( C ^ { \prime } ) \cap { \mathcal { A } } ) / w ( A )$ and weighted precision is $P _ { w } = w ( S ( C ^ { \prime } ) \cap A ) / w ( S ( C ^ { \prime } ) )$ ). We report both and their harmonic mean, the weighted $\mathrm { F _ { 1 } }$ $( \mathrm { F } _ { 1 , w } )$ , on a 0–100 scale. All sums are exact over every legal plan (Appendix E.1).

Extrinsic: planning outcome. A specification ultimately matters through the plans built within it, so we also give $C ^ { \prime }$ to a fixed planner, which sees the instance and the specification but not the user, and ask it to deliver a legal plan within $S ( C ^ { \prime } )$ Our main planning measure is the acceptance rate (Acc), the share of instances whose delivered plan lies in ${ \mathcal { A } } ,$ counting an instance without a delivered plan as not accepted. We also report the delivery rate (Deliv), the share of instances with a delivered plan, and give the utility of delivered plans in Appendix I.2.

## 4 Experimental Setup

## 4.1 Benchmarks

ReSHAPE Planter. Hand-building instances in which revision is both necessary and possible does not scale, so ReSHAPE Planter converts an existing planning benchmark into such instances in three steps (Appendix F). First, it adds attributes worth trading to every option, such as a flight’s on-time rate. Next, it finds a trade-off that the options actually offer, such as restaurants where a lower rating comes with much cheaper meals. Lastly, it builds a hidden user who would accept this trade-off but whose request does not state it. The request becomes the initial specification $C _ { 0 } = c _ { i } \wedge c _ { k } \wedge c _ { b }$ , whose condition $c _ { i }$ on the sacrificed attribute excludes the trade-off; $c _ { k }$ is the deal-breaker and $c _ { b }$ the stated budget. Writing the trade-off into $C _ { 0 }$ gives the reference specification $C _ { \mathrm { r e f } } = [ c _ { i } \lor ( c _ { i } ^ { - } \land c _ { j } ) ] \land c _ { k } \land c _ { b }$ , which serves as one known-good revision and plays no part in defining ${ \mathcal { A } } .$ ReSHAPE Planter keeps an instance only if the trade-off passes two checks. It must be worth discovering: some plan that only $C _ { \mathrm { r e f } }$ admits beats the best plan under $C _ { 0 }$ by at least δ. It must also be needed: $S ( C _ { \mathrm { r e f } } )$ matches A better than revisions that need no knowledge of the trade-off, such as relaxing the budget. No step uses agent performance. Appendix B walks through one instance.

ReSHAPE Travel and ReSHAPE Shopping. We apply ReSHAPE Planter to the 102 medium and hard TravelPlanner queries (Xie et al., 2024) included in Collaborative Gym (Shao et al., 2026) and to the 97 shopping requests of DeepPlanning (Zhang et al., 2026), keeping the 88 and 87 for which it verifies a trade-off (Appendices F.5 and F.6). Since about half of the shopping requests state no budget, ReSHAPE Planter states one for each by a rule taken from TravelPlanner’s budgets.

Validating the instances and the simulated user. Since simulated users can diverge from real ones (Seshadri et al., 2026; Zhou et al., 2026), we ran a user study, which confirms that the trade-offs Re-SHAPE Planter plants are ones people would take and that the simulated user judges them as people do. In the study, 20 participants each judged six ReSHAPE Travel instances for a described traveler, choosing between a plan the initial specification admits and a plan that only a revision of it admits (Appendix H). The revision is either the instance’s verified trade-off or a decoy that compensates with an attribute the traveler does not care about. Under the verified trade-off, a majority of participants chose the plan outside the request on every instance; under the decoy, most kept the plan the request admits. Their majority choice agrees with the simulated user in 11 of the 12 scenarios, with a tie in the last, and individual choices agree with it on 76% (95% CI [66, 85]), which is as high as any fixed answer can reach given how much participants disagree with one another.

## 4.2 Agents and Evaluation

Agents. We pair four agent harnesses, mini-SWEagent (Yang et al., 2024), OpenHands (Wang et al., 2025), Claude Code (Anthropic, 2025), and Gemini CLI (Google, 2025), with six models into fourteen systems (Appendix G.1), each requesting medium reasoning effort and otherwise keeping its harness’s defaults.

Task interface and runs. Each agent uses a command-line tool whose instruction $( \mathsf { A p } \cdot$ pendix J.1) states the objective and the available actions without suggesting a strategy: it may inspect options or check candidate plans for free, ask the user a question, show the user one or two plans for feedback, and revise and submit the specification, in any order within the budget B = 10. Each system runs once per instance against the simulated user of §3.2, and a failed run is scored by its last valid specification (Appendix G.2).

<table><tr><td rowspan=1 colspan=6>ReSHAPE Travel        Specification   Accepted (%)Model             $\mathrm { F } _ { 1 , w }$    $P _ { w }$     $R _ { w }$  FlashLuna</td></tr><tr><td rowspan=1 colspan=6> $C _ { \mathrm { r e f } }$ (reference)      21.5 26.5 53.8  55   47</td></tr><tr><td rowspan=2 colspan=1>Kimi K3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>12.2</td><td rowspan=1 colspan=1>19.2</td><td rowspan=1 colspan=1>31.5</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>44</td></tr><tr><td rowspan=4 colspan=1>OpenndsClaude Sonnet 4.6Gemini 3.6 FlashGemini 3.1 Progpt-oss-120b</td><td rowspan=1 colspan=1>9.5</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>22.6</td><td rowspan=1 colspan=1>47</td><td rowspan=1 colspan=1>42</td></tr><tr><td rowspan=1 colspan=1>8.7</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>14.7</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>36</td></tr><tr><td rowspan=1 colspan=1>4.4</td><td rowspan=1 colspan=1>16.2</td><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>13.1</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=3 colspan=1>Kimi K3Claude Sonnet 4.6</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>25.9</td><td rowspan=1 colspan=1>31.6</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>44</td></tr><tr><td rowspan=1 colspan=1>12.1</td><td rowspan=1 colspan=1>22.3</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>33</td></tr><tr><td rowspan=3 colspan=1>m--ntGemini 3.6 FlashGemini 3.1 Progpt-oss-120b</td><td rowspan=1 colspan=1>8.2</td><td rowspan=1 colspan=1>18.6</td><td rowspan=1 colspan=1>18.4</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>22</td></tr><tr><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1>20.8</td><td rowspan=1 colspan=1>13.2</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>4.8</td><td rowspan=1 colspan=1>15.8</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=3 colspan=1>CludeCodeClaude Opus 4.6Claude Sonnet 4.6</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>40</td></tr><tr><td rowspan=1 colspan=1>7.8</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=1>20.4</td><td rowspan=1 colspan=1>47</td><td rowspan=1 colspan=1>47</td></tr><tr><td rowspan=2 colspan=1>Geni Gemini 3.6 FlashCIIGemini 3.1 Pro</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>26.3</td><td rowspan=1 colspan=1>14.4</td><td rowspan=1 colspan=2>36   32</td></tr><tr><td rowspan=1 colspan=1>3.8</td><td rowspan=1 colspan=1>15.5</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=2>11   15</td></tr><tr><td rowspan=1 colspan=1> $C _ { 0 }$ (initial)</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=2>12   14</td></tr><tr><td rowspan=1 colspan=1> $C _ { \varnothing }$ (empty)</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=2>8   11</td></tr></table>

Table 1: Specifications and the plans they lead to. Left: the specification each system submits, scored against the acceptable plans (×100). Right: the share of instances on which a fixed planner, given that specification, delivers a plan the user accepts; Flash is Gemini 3.6 Flash, Luna is GPT-6 Luna. Within a harness, models are ordered by $\mathrm { F } _ { 1 , w } .$ Shading: the value as a share of $C _ { \mathrm { { r e f } } } \mathrm { { ^ { \circ } s } }$ . Bold: best system per column.

Interaction protocols. To separate strategy from harness and model, we also run four fixed protocols with Gemini 3.6 Flash under the same actions and budget, each restricted to one of the four kinds of interaction we find in the off-the-shelf logs (§5.2). Ask-before-Plan (Zhang et al., 2024) asks preference questions before revising; Self-Directed Revision checks candidate plans itself without consulting the user, which we call self-checking; Exception Probing shows the user one plan outside the current search space at a time and revises after each judgment; and Trade-off Probing states a trade-off hypothesis and tests it with a pair of plans, at two units per pair (Appendix G.3). We call the last two, which show plans outside the current specification, the probing protocols.

Planner and statistics. The fixed planner, Gemini 3.6 Flash with low thinking, searches the options for a plan that satisfies the specification it is given (Appendix E.2). For the main results we also report GPT-6 Luna as a second planner (Appendix I.3). Sources are compared by a paired bootstrap over instances with 1,000 resamples.

<table><tr><td rowspan="2">ReSHAPE Shopping Model</td><td colspan="3">Specification</td><td colspan="2">Accepted (%)</td></tr><tr><td> $\mathrm { F } _ { 1 , w }$ </td><td> $P _ { w }$ </td><td> $R _ { w }$ </td><td>Flash</td><td>Luna</td></tr><tr><td> $C _ { \mathrm { r e f } }$  (reference)</td><td>26.5</td><td>22.8</td><td>52.8</td><td>38</td><td>57</td></tr><tr><td rowspan="3">Gemini 3.6 Flash  $\lesseqqgtr - \frac { 9 } { 5 }$  Kimi K3</td><td>9.1</td><td>20.3</td><td>10.5</td><td>36</td><td>41</td></tr><tr><td>8.9</td><td>12.3</td><td>13.4</td><td>36</td><td>30</td></tr><tr><td>gpt-oss-120b 5.2</td><td>5.8</td><td>11.2</td><td>29</td><td>24</td></tr><tr><td rowspan="2">-ı SWE aaent gpt-oss-120b Gemini 3.6 Flash</td><td rowspan="2">6.2 6.1</td><td>11.8</td><td>12.2</td><td>29</td><td>28</td></tr><tr><td>17.0</td><td>8.2</td><td>32</td><td>26</td></tr><tr><td> $\lesseqqgtr \frac { \Delta } { \sum } \frac { \Delta } { \sum }$  Claude Opus 4.6</td><td>11.2</td><td>23.4</td><td>13.3</td><td>47</td><td>39</td></tr><tr><td rowspan="2"> $\begin{array} { c } { \mathbf { \widetilde { g } } } \\ { \mathbf { \widetilde { \Xi } } } \\ { \mathbf { \widetilde { \Xi } } } \\ { \mathbf { \widetilde { \Xi } } } \end{array}$  Gemini 3.6 Flash Gemini 3.1 Pro</td><td>8.5</td><td>21.4</td><td>10.6</td><td>39</td><td>41</td></tr><tr><td>7.0</td><td>16.0</td><td>11.6</td><td>26</td><td>37</td></tr><tr><td> $C _ { 0 }$  (initial)</td><td>2.6</td><td>3.4</td><td>6.1</td><td>26</td><td>22</td></tr><tr><td> $C _ { \varnothing }$  (empty)</td><td>0.1</td><td>0.0</td><td>100.0</td><td>9</td><td>11</td></tr></table>

Table 2: The same on ReSHAPE Shopping, for the eight systems run there. Columns, ordering, and shading are as in Table 1.
<table><tr><td rowspan="2"></td><td colspan="3">Specification</td><td colspan="2">Accepted</td></tr><tr><td> $\mathrm { F } _ { 1 , w }$ </td><td> $P _ { w }$ </td><td> $R _ { w }$ </td><td>Flash Luna</td><td></td></tr><tr><td>Spearman ρ</td><td>.81</td><td>.90</td><td>.40</td><td>.94</td><td>.59</td></tr><tr><td>Kendall  $\tau _ { b }$ </td><td>.64</td><td>.79</td><td>.21</td><td>.87</td><td>.40</td></tr></table>

Table 3: The two domains order the systems alike. Rank correlation between the travel and the shopping values of the 8 systems run in both (Tables 1 and 2).

Specifications analyzed. Our analysis covers 6,325 specifications: on both benchmarks, the baselines $C _ { 0 } .$ , the empty specification, and $C _ { \mathrm { r e f } } .$ , and the final specifications of the agent systems, 14 on Re-SHAPE Travel and 8 on ReSHAPE Shopping; on ReSHAPE Travel, also those the four protocols hold after each unit of feedback.

## 5 Analysis

## 5.1 What Makes a Revision Useful for Planning?

Reshaping the specification matters. On Re-SHAPE Travel, with the planner fixed (Table 1), $C _ { 0 }$ leads to an acceptable plan on only 12% of instances, and ignoring the request is worse: under the empty specification the planner almost always delivers a plan, but only 8% are acceptable. The two fail for opposite reasons, $C _ { 0 }$ admitting almost none of the acceptable plans and the empty specification almost nothing else. The single verified trade-off of $C _ { \mathrm { r e f } }$ raises acceptance to 55%, and the off-the-shelf systems span 10–47%: what the planner is given decides much of what it achieves. On ReSHAPE Shopping (Table 2), $C _ { \mathrm { r e f } }$ again raises acceptance over $C _ { 0 } .$ , from 26% to 38% with Gemini 3.6 Flash and from 22% to 57% with GPT-6 Luna, and the empty specification does worst.

A good revision first admits more plans and then stays selective. Figure 2 gives the planner the specification each protocol holds after every unit of feedback. The first revisions of both probing protocols raise recall from almost nothing, and acceptance rises with it: a single judgment lets Exception Probing double the acceptance rate of $C _ { 0 } ,$ and Trade-off Probing gains nearly all of its improvement with its first pair. Beyond the first few units (shaded in Figure 2), recall keeps growing while precision stays roughly flat, and acceptance no longer improves: a planner delivers a single plan, so once the search space holds enough acceptable plans for it to find one, admitting more adds little. Precision then sets the level at which acceptance saturates: Exception Probing ends with a recall close to that of $C _ { \mathrm { r e f } }$ but half its precision, and is accepted 20 points less often; Self-Directed Revision has more than twice the recall of Askbefore-Plan but half its precision, and is accepted less often (Table 4).

Search space quality predicts the planning outcome, and $\mathrm { F } _ { 1 , w }$ tracks it as well as any measure we test. Instances differ in how hard they are to plan, so we compare specifications within an instance: for every pair of specifications on the same instance of which the planner accepts exactly one, we ask whether a measure scores the accepted one higher (the within-instance AUC, Appendix I.1). Over the fourteen systems and three baselines, $\mathrm { F } _ { 1 , w }$ orders 87% of such pairs correctly, against 85% for precision and 71% for recall, and none of the variants we tried does reliably better; so search space quality can stand in for running a planner. On ReSHAPE Shopping, $\mathrm { F } _ { 1 , w }$ orders 75% of the pairs correctly with Gemini 3.6 Flash and 84% with GPT-6 Luna.

![](images/9f9d6f4952524302ddfe1adaab9077ccf0fadd5fc399eac5e0fe3ba4512c62ab.jpg)  
Figure 2: (a) Precision (solid) and recall (dashed) of the probing protocols’ specifications and (b) the acceptance rate of all four protocols, after each unit of feedback. Unit 0 is $C _ { 0 } .$ , and Trade-off Probing spends two units per pair. In the shaded region past the fourth unit, recall keeps rising while acceptance has plateaued. Dotted lines mark $C _ { \mathrm { r e f } }$

## 5.2 How Do Agents Revise, and How Should Feedback Be Spent?

Current agents prioritize precision but cannot recover recall. Left free to interact, thirteen of the fourteen systems raise $\mathrm { F } _ { 1 , w }$ significantly above $C _ { 0 }$ , but the best, mini-SWE-agent with Kimi K3, reaches 14.0 against 21.5 for $C _ { \mathrm { r e f } }$ (Table 1). The gap lies in recall: most systems reach a precision of 19–27, close to the 26.5 of $C _ { \mathrm { r e f } } .$ , but admit at most a third of the acceptable plans, against more than half for $C _ { \mathrm { r e f } }$ , so the plans the initial specification excluded largely stay excluded. Their effort goes to asking and self-checking, which together make up three quarters of their actions, while showing the user a plan makes up only a tenth: agents mostly judge candidate plans on the user’s behalf instead of letting the user react to them.

Few systems write a trade-off at all: at most 39% of any system’s final specifications contain a meaningful one, which relates two different attributes that both matter to the user, each restricted in the direction the user prefers (Appendix G.5). ReSHAPE Shopping shows the same pattern: no system admits more than 13% of the acceptable plans, against 53% for $C _ { \mathrm { r e f } }$ , though the most precise matches the precision of $C _ { \mathrm { r e f } }$ , and asking and self-checking again make up most actions.

<table><tr><td></td><td colspan="3">Specification</td><td>Acc.</td><td>Trade- off</td></tr><tr><td>Protocol</td><td> $\mathrm { F } _ { 1 , w }$ </td><td> $P _ { w }$ </td><td> $R _ { w }$ </td><td>(%)</td><td>(%)</td></tr><tr><td> $C _ { \mathrm { r e f } }$  (reference)</td><td>21.5</td><td>26.5</td><td>53.8</td><td>55</td><td>100</td></tr><tr><td>Ask-before-Plan</td><td>5.3</td><td>14.7</td><td>9.6</td><td>22</td><td>7</td></tr><tr><td>Self-Directed Revision</td><td>3.5</td><td>6.9</td><td>22.9</td><td>19</td><td>3</td></tr><tr><td>Exception Probing</td><td>15.0</td><td>14.2</td><td>43.6</td><td>35</td><td>9</td></tr><tr><td>Trade-off Probing</td><td>12.3</td><td>14.6</td><td>28.6</td><td>26</td><td>32</td></tr><tr><td> $C _ { 0 }$  (initial)</td><td>0.7</td><td>12.5</td><td>0.8</td><td>12</td><td>0</td></tr></table>

Table 4: Interaction protocols with Gemini 3.6 Flash on 88 instances. Trade-off: runs ending with a meaningful trade-off (Appendix G.5). Shading as in Table 1. Bold: best per column.

Across the eight systems run in both domains, the ordering by precision and by acceptance carries over from travel to shopping, but the ordering by recall does not (Table 3).

Showing the user plans outside the current specification helps mitigate the imbalance. Across systems, the more plans an agent shows outside its current specification, the more acceptable plans its revision admits (Spearman’s $\rho = . 9 0 )$ . Since stronger models may both show more plans and revise better, we test this link under a fixed model, running each kind of interaction on its own with Gemini 3.6 Flash (Table 4). The two probing protocols reach $\mathrm { F } _ { 1 , w }$ of 15.0 and 12.3, against 5.3 for Ask-before-Plan and 3.5 for Self-Directed Revision, and lead the planner to acceptable plans more often, most of the gain arriving within the first few units of feedback (Figure 2). Asking barely moves the search space, and revising without the user admits mostly unacceptable plans, so the user’s reactions to concrete plans carry information the agent cannot recover on its own. Exception Probing still gains recall at the cost of precision, so although it nearly doubles the $\mathrm { F } _ { 1 , w }$ of the same model off the shelf, its plans are accepted no more often (35% vs. 31–36%), as §5.1 predicts. The same holds with Claude Sonnet 4.6 (Appendix G.4): the two probing protocols reach $\mathrm { F } _ { 1 , w }$ of 13.8 and 12.1 against 2.6 and 5.1 for the controls and are accepted more often, while the off-the-shelf Sonnet systems are accepted as often as its probing protocols.

Recognizing the user’s trade-off is the bottleneck. Even when instructed to test trade-off hypotheses, the agent ends with a meaningful tradeoff on only 28 of 88 instances; on those, Trade-off Probing is more precise than Exception Probing $( P _ { w }$ 20.4 against 15.1) and nearly matches $C _ { \mathrm { r e f } }$ $( \mathrm { F } _ { 1 , w }$ 20.1 against 23.7), but on the other 60 it falls behind Exception Probing (Table 22). $C _ { \mathrm { r e f } }$ still outperforms every protocol and every system.

## 6 Discussion and Conclusion

Discussion. Our results suggest three things beyond this benchmark. First, evaluations of interactive planning should score the specification as well as the delivered plan: search space quality needs no planner run, predicts the planner’s acceptance, and keeps its orderings under a second planner (Appendix I.3) and in a second domain (Table 3), and unlike a delivered plan it shows whether a revision fails by admitting too little or by keeping out too little. Second, the order of work matters for agent design: early feedback is best spent showing plans outside the request, to admit what it excluded, and later feedback on keeping out what the user would reject. Agents that test trade-off hypotheses already revise most selectively, but only when they propose the right trade-off, so proposing candidate trade-offs from the options actually on offer may be the most direct way to improve them. Third, ReSHAPE Planter needs only options described by attributes and a space of legal plans; it already carries over from travel to shopping, and the same construction applies to other planning benchmarks, such as meal or course planning, whose requests likewise leave their trade-offs unstated.

Conclusion. Planning agents inherit whatever specification they are given, and the one a user’s request yields leaves out plans the user would accept. We made revising it measurable, by how well a revision’s search space matches the plans the user would accept and by what a planner finds within it, and analyzed it in travel and shopping planning. Reshaping matters, and a good revision first admits more plans and then stays selective; current agents instead ask about preferences and recover few of the plans the request excluded. Planning agents should therefore treat a request as a hypothesis to test against the user’s reactions to concrete plans; recognizing which trade-off the user would accept remains the open problem.

## Limitations

Users with fixed preferences. Our simulated user holds a fixed utility over plans, which makes acceptance exact and lets us score a revision over every legal plan. Real users may instead form or change their preferences as they see options, judge similar plans inconsistently, or settle for a plan that is good enough (Simon, 1955). How revision should be measured, and which interaction works best, when the target itself moves with the interaction is a question our setup cannot answer; it calls for users who revise their own requests.

One trade-off at a time. Each instance plants a single trade-off between two attributes of one option category. Real requests can hide several trade-offs at once, including ones that span categories, such as a cheaper hotel that pays for a better flight. Whether the order we find, admitting first and selecting second, holds when trade-offs interact is open.

Revisions judged on one set of options. Search space quality is measured on the options of the instance, so a condition that happens to separate acceptable from unacceptable plans there scores as well as the user’s actual trade-off. A revision meant to capture what the user wants should also hold on options the user has not seen; measuring this transfer would separate understanding the user from fitting the instance.

Discovering, not presenting. We study how an agent discovers which plans the user would accept, with a user who judges every plan it is shown exactly. How an agent should present options so that a real user can reason over them, and how presentation interacts with what the user discovers, is left to future work.

## Acknowledgement

This work is supported by the Okawa Foundation Research Grant, the Microsoft AI and New Future of Work Grant, Gemini Academic Program Award, and Amazon Research Award. We thank the user study participants for their valuable input.

## References

Anthropic. 2025. Claude Code. Anthropic.

Anthropic. 2026a. Introducing Claude Opus 4.6. Anthropic.

Anthropic. 2026b. Introducing Claude Sonnet 4.6. Anthropic.

Mohamed-Bachir Belaid, Nassim Belmecheri, Arnaud Gotlieb, Nadjib Lazaar, and Helge Spieker. 2022. GEQCA: Generic qualitative constraint acquisition. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 3690–3697.

James R. Bettman, Mary Frances Luce, and John W. Payne. 1998. Constructive consumer choice processes. Journal ofConsumer Research, 25(3):187– 217.

Li Chen and Pearl Pu. 2012. Critiquing-based recommenders: Survey and emerging trends. User Modeling and User-Adapted Interaction, 22(1–2):125–150.

Dave Citron. 2024. Try deep research and our new experimental model in Gemini, your AI assistant. Google Blog. December 11, 2024.

Jonathan Cook, Tim Rocktäschel, Jakob Foerster, Dennis Aumiller, and Alex Wang. 2024. TICKing all the boxes: Generated checklists improve LLM evaluation and generation. Preprint, arXiv:2410.03608.

Rebecca Eifler, Jeremy Frank, and Jörg Hoffmann. 2022. Explaining soft-goal conflicts through constraint relaxations. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, pages 4621–4627. International Joint Conferences on Artificial Intelligence Organization.

Hillel J. Einhorn. 1970. The use of nonlinear, noncompensatory models in decision making. Psychological Bulletin, 73(3):221–230.

Google. 2025. Gemini CLI. GitHub.

Google. 2026a. Gemini 3.1 Pro Preview. Google AI for Developers.

Google. 2026b. Gemini 3.6 Flash. Google AI for Developers.

Yilun Hao, Yongchao Chen, Yang Zhang, and Chuchu Fan. 2025. Large language models can solve realworld planning rigorously with formal verification tools. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3434– 3483. Association for Computational Linguistics.

Jerry Zhi-Yang He and Anca D. Dragan. 2022. Assisted robust reward design. In Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pages 1234–1246. PMLR.

Ralph L. Keeney and Howard Raiffa. 1976. Decisions with Multiple Objectives: Preferences and Value Tradeoffs. John Wiley & Sons.

Tae Soo Kim, Yoonjoo Lee, Jaesang Yu, John Joon Young Chung, and Juho Kim. 2026. Discover-LLM: From executing intents to discovering them. In Proceedings ofthe 43rd International Conference on Machine Learning.

Kimi Team. 2026. Kimi K3: Open frontier intelligence. Preprint, arXiv:2607.24653.

Laude Institute. 2025. Harbor: A framework for evaluating and optimizing agents and language models. GitHub. Version 0.23.0.

Belinda Z. Li, Alex Tamkin, Noah D. Goodman, and Jacob Andreas. 2025. Eliciting human preferences with language models. In The Thirteenth International Conference on Learning Representations.

Lening Li, Hazhar Rahmani, and Jie Fu. 2023. Probabilistic planning with prioritized preferences over temporal logic objectives. In Proceedings of the Thirty-Second International Joint Conference on Arti ficial Intelligence, pages 189–198. International Joint Conferences on Artificial Intelligence Organization.

Jiayu Liu, Cheng Qian, Zhenhailong Wang, Bingxuan Li, Jiateng Liu, Qing Zong, Heng Wang, Jeonghwan Kim, Yumeng Wang, Bingxiang He, Xiusi Chen, Yi R. Fung, and Heng Ji. 2026. AdaPlanBench: Evaluating adaptive planning in large language model agents under world and user constraints. In Third Conference on Language Modeling.

R. Duncan Luce. 1956. Semiorders and a theory of utility discrimination. Econometrica, 24(2):178–191.

Malek Mechergui and Sarath Sreedharan. 2024a. Expectation alignment: Handling reward misspecification in the presence of expectation mismatch. In Advances in Neural Information Processing Systems, volume 37. Curran Associates, Inc.

Malek Mechergui and Sarath Sreedharan. 2024b. Goal alignment: Re-analyzing value alignment problems using human-aware AI. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 10110–10118.

Younes Mechqrane, Christian Bessiere, and Ismail Elabbassi. 2024. Using large language models to improve query-based constraint acquisition. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, pages 1916–1925. IJCAI.

Nina Narodytska, Nikolaj Bjørner, Maria-Cristina Marinescu, and Mooly Sagiv. 2018. Core-guided minimal correction set and core enumeration. In Proceedings ofthe Twenty-Seventh International Joint Conference on Artificial Intelligence, pages 1353–1361. International Joint Conferences on Artificial Intelligence Organization.

Juhyun Oh, Eunsu Kim, and Alice Oh. 2025. Flex-TravelPlanner: A benchmark for flexible planning with language agents. In Reasoning and Planning for Large Language Models at ICLR 2025.

OpenAI. 2025a. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

OpenAI. 2025b. Introduction to deep research in the OpenAI API. OpenAI Cookbook.

John W. Payne, James R. Bettman, and Eric J. Johnson. 1993. The Adaptive Decision Maker. Cambridge University Press.

Peter Pirolli and Stuart Card. 2005. The sensemaking process and leverage points for analyst technology as identified through cognitive task analysis. In Proceedings ofthe International Conference on Intelligence Analysis, volume 5, pages 2–4.

Cheng Qian, Bingxiang He, Zhong Zhuang, Jia Deng, Yujia Qin, Xin Cong, Zhong Zhang, Jie Zhou, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2024. Tell me more! towards implicit user intention understanding of language model driven agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1088–1113. Association for Computational Linguistics.

Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, and Huan Wang. 2025. UserBench: An interactive gym environment for user-centric agents. Preprint, arXiv:2507.22034.

Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, and Seraphina Goldfarb-Tarrant. 2026. Lost in simulation: LLM-simulated users are unreliable proxies for human users in agentic evaluations. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 47423– 47439, San Diego, California, United States. Association for Computational Linguistics.

Yijia Shao, Vinay Samuel, Yucheng Jiang, John Yang, and Diyi Yang. 2026. Collaborative gym: A framework for enabling and evaluating human-agent collaboration. In The Fourteenth International Conference on Learning Representations.

Zijian Shao, Jiancan Wu, Weijian Chen, and Xiang Wang. 2025. Personal travel solver: A preferencedriven LLM-solver system for travel planning. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 27622–27642. Association for Computational Linguistics.

Shannon Zejiang Shen, Valerie Chen, Ken Gu, Alexis Ross, Zixian Ma, Jillian Ross, Alex Gu, Chenglei Si, Wayne Chi, Andi Peng, Jocelyn J Shen, Ameet Talwalkar, Tongshuang Wu, and David Sontag. 2026. Scaling collaborative effort with agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 12254–12271, San Diego, California, United States. Association for Computational Linguistics.

Herbert A. Simon. 1955. A behavioral model of rational choice. The Quarterly Journal of Economics, 69(1):99–118.

Dimos Tsouros, Senne Berden, Steven Prestwich, and Tias Guns. 2025. Generalizing constraint models in constraint acquisition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 11362–11371.

Dimosthenis C. Tsouros, Senne Berden, and Tias Guns. 2023. Guided bottom-up interactive constraint acquisition. In 29th International Conference on Principles and Practice of Constraint Programming, volume 280 of Leibniz International Proceedings in Informatics, pages 36:1–36:20. Schloss Dagstuhl – Leibniz-Zentrum für Informatik.

Paolo Viappiani, Boi Faltings, and Pearl Pu. 2006. Preference-based search using example-critiquing with suggestions. Journal ofArtificial Intelligence Research, 27:465–503.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, and 5 others. 2025. OpenHands: An open platform for AI software developers as generalist agents. In The Thirteenth International Conference on Learning Representations.

Ruixuan Xiao, Wentao Ma, Ke Wang, Yuchuan Wu, Junbo Zhao, Haobo Wang, Fei Huang, and Yongbin Li. 2024. FlowBench: Revisiting and benchmarking workflow-guided planning for LLM-based agents. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 10883–10900. Association for Computational Linguistics.

Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. 2024. TravelPlanner: A benchmark for real-world planning with language agents. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 54590–54613. PMLR.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, volume 37.

Xuan Zhang, Yang Deng, Zifeng Ren, See-Kiong Ng, and Tat-Seng Chua. 2024. Ask-before-plan: Proactive language agents for real-world planning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 10836–10863. Association for Computational Linguistics.

Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong Tu, Yang Su, Lianghao Deng, Xudong Guo, Chenxu Lv,

and Junyang Lin. 2026. DeepPlanning: Benchmarking long-horizon agentic planning with verifiable constraints. Preprint, arXiv:2601.18137.

Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, and Denny Zhou. 2024. NATURAL PLAN: Benchmarking LLMs on natural language planning. Preprint, arXiv:2406.04520.

Xuhui Zhou, Weiwei Sun, Qianou Ma, Yiqing Xie, Jiarui Liu, Weihua Du, Sean Welleck, Yiming Yang, Graham Neubig, Tongshuang Wu, and Maarten Sap. 2026. Mind the sim2real gap in user simulation for agentic tasks. In Third Conference on Language Modeling.

## A Related Settings

Table 5 compares ReSHAPE with related planning and interaction settings (§2). It asks of each setting whether plans are composed over a real option set (grounded), whether the user reacts to plans the agent shows (plan feedback), whether acceptance depends on a plan’s overall utility rather than on each constraint (trade-offs), and what it scores. Most settings score the plan, or the set of options, that the agent finally delivers; GATE scores whether the preferences it elicits predict the user’s decisions on examples withheld from elicitation; ReSHAPE scores the plans a revised specification admits against the plans the user would accept, and what a planner finds among them.

<table><tr><td></td><td>Grounded planning</td><td>Plan feedback</td><td>Trade- offs</td><td>Judged on</td></tr><tr><td>Ask-before-Plan (Zhang et al., 2024)</td><td>√</td><td>×</td><td>x</td><td>delivered plan</td></tr><tr><td>Flex-TravelPlanner (Oh et al., 2025)</td><td>√</td><td>×</td><td>F</td><td>delivered plan</td></tr><tr><td>UserBench (Qian et al., 2025)</td><td>X</td><td>×</td><td>X</td><td>delivered options</td></tr><tr><td>AdaPlanBench (Liu et al., 2026)</td><td>×</td><td>√</td><td>×</td><td>delivered plan</td></tr><tr><td>GATE (Li et al., 2025)</td><td>×</td><td>X</td><td>×</td><td>held-out decisions</td></tr><tr><td>ReSHAPE</td><td>V</td><td>√</td><td>V</td><td>admitted plans + planner outcome</td></tr></table>

Table 5: Related settings, by what each models and what it scores.

## B ReSHAPE on One Instance

This appendix follows one instance of ReSHAPE Travel, travel:50, from the request to the planner’s plan, so that the objects of §3 and the protocols of §5.2 can be seen on real options. Prices below are for the party of two.

Request and initial specification. The request has a budget of \$1,400 and reads:

Request Plan a 3-day trip for 2 people from   
Newark to Savannah from March 3, 2022 to   
March 5, 2022.

The initial specification $C _ { 0 }$ holds three conditions: the chosen restaurants list at most two cuisines on average (c<sub>i</sub>), the lodging is rated at least $3 . 0 \left( c _ { k } . \right.$ , the deal-breaker), and the known total cost is at most \$1,400 (c<sub>b</sub>). The option pool holds 23 restaurants, and only three list two cuisines or fewer: New Raj Kachori Bhandar (\$24 a meal, rated 3.4), Country’s Barbecue (\$118, 4.0), and

<table><tr><td>Attribute</td><td>Imp.</td><td>Expects Prefers</td><td></td><td>Role</td></tr><tr><td>Lodging rating</td><td></td><td>medium medium higher</td><td></td><td> $c _ { k }$ </td></tr><tr><td>Attraction type</td><td>low</td><td></td><td>historic site</td><td></td></tr><tr><td>Meal cost</td><td>high</td><td>medium lower</td><td></td><td>j</td></tr><tr><td>Restaurant cuisines low</td><td></td><td></td><td>Italian</td><td></td></tr><tr><td>Cuisine count</td><td>low</td><td>low</td><td>lower</td><td>2</td></tr><tr><td>Restaurant rating</td><td>low</td><td>high</td><td>higher</td><td></td></tr></table>

Table 6: Hidden profile of travel:50. Expects is the expectation of a numeric preference; Prefers is its direction or, for a categorical one, the liked value; the disliked values are markets and fast food. $c _ { k }$ marks the deal-breaker, i the sacrificed attribute, and $j$ the compensating one. The interaction style reads “Explains requests in some detail and appreciates specific recommendations with brief context.”

Chutneez (\$104, 2.7). Every plan $C _ { 0 }$ admits therefore eats at these three; $C _ { 0 }$ admits 20 legal plans, which differ only in lodging, transport, and attraction.

Hidden profile. Table 6 lists the user’s six preferences. Meal cost matters most to this user, and the number of cuisines a restaurant lists matters little. The request nevertheless restricts the cuisine count and says nothing about meal cost. This is the gap ReSHAPE Planter plants: the user would accept restaurants with more cuisines if meals are cheap. The deal-breaker is the lodging rating, the more important of the two preferences outside the restaurant category, where the trade-off lies (Appendix D.3).

Acceptance and the reference specification. The best plan $C _ { 0 }$ admits has utility $U ( p _ { 0 } ^ { * } ) = 0 . 1 9$ so the threshold is $\tau = 0 . 1 4$ . Of the 7,402,780 legal plans, 51,915 (0.7%) are acceptable, and $C _ { 0 }$ admits seven of them: its weighted precision is 58 and its weighted recall is near zero. The reference specification relaxes the cuisine condition when meals are cheap. $C _ { \mathrm { r e f } }$ requires a cuisine count of at most 2, or at most 6 with meals of at most \$56 on average, together with $c _ { k }$ and $c _ { b } ;$ eight restaurants have at most six cuisines and meals of at most \$56. The reference plan drives both ways, stays in a 5.0-rated room, and eats at The Mad Teapot (\$46, rated 4.4, three cuisines), New Raj Kachori Bhandar, and Manohar Dairy (\$46, 4.4, three cuisines), for \$1,381 in all; its utility is 0.40, twice that of $p _ { 0 } ^ { * }$ $C _ { \mathrm { r e f } }$ admits 2,680 plans, 81% of them acceptable, for a weighted precision of 94.6, a recall of 6.8, and an $\mathrm { F } _ { 1 , w }$ of 12.8 (Table 7).

<table><tr><td>Specification</td><td> $P _ { w }$ </td><td> $R _ { w }$ </td><td> $\mathrm { F } _ { 1 , w }$ </td><td>Admitted</td><td>Accept.</td><td>Plan</td></tr><tr><td> $C _ { \varnothing }$  (empty)</td><td>0.0</td><td>100.0</td><td>0.0</td><td>7,402,780</td><td>51,915</td><td>rejected</td></tr><tr><td>C0 (initial)</td><td>58.1</td><td>0.0</td><td>0.0</td><td>20</td><td>7</td><td>none</td></tr><tr><td>Cref (reference)</td><td>94.6</td><td>6.8</td><td>12.8</td><td>2,680</td><td>2,172</td><td>accepted</td></tr><tr><td>Ask-before-Plan</td><td>58.1</td><td>0.0</td><td>0.0</td><td>20</td><td></td><td>7 rejected</td></tr><tr><td>Self-Directed Revision</td><td>19.8</td><td>3.6</td><td>6.0</td><td>6,560</td><td>1,646</td><td>accepted</td></tr><tr><td>Exception Probing</td><td>32.9</td><td>45.6</td><td>38.2</td><td>47,420</td><td>19,301</td><td>accepted</td></tr><tr><td>Trade-off Probing</td><td>56.2</td><td>18.0</td><td>27.3</td><td>14,420</td><td>7,784</td><td>accepted</td></tr></table>

Table 7: Specifications on travel:50. Weighted precision, recall, and $\mathrm { F } _ { 1 , w } \left( \times 1 0 0 \right)$ ; the number of legal plans each admits and how many of those the user accepts; and whether the plan Gemini 3.6 Flash delivers under it is accepted. Ask-before-Plan submitted $C _ { 0 }$ unchanged, so its planner row is a second run on the same specification.

The planner. Given $C _ { 0 }$ , Gemini 3.6 Flash found none of the 20 admitted plans in 14 calls and delivered nothing. Given $C _ { \mathrm { { r e f } } } ,$ , it delivered a plan that eats at Sr. Sol 1, Manohar Dairy, and The Mad Teapot and stays in the 3.0-rated Private Oasis room, which the user accepts. Given the empty specification, it delivered a \$72-a-night room rated 1.0, which breaks the deal-breaker.

The four protocols. Table 7 scores the specification each protocol submitted and what the planner made of it. The excerpts below quote the simulated user’s actual replies.

Ask-before-Plan spent all ten units on questions and changed nothing after any answer, so it submitted C<sub>0</sub>. Two of its ten exchanges:

Agent asks the preferred direction of the   
listed cuisine count.   
User “I generally prefer places with a   
lower variety of cuisines.”   
Agent asks the preferred direction of   
transportation cost.   
User “I don’t have a strong preference   
either way on transportation costs. What   
matters most to me is finding something that   
works for my trip.”

No answer could say what the user would give up for cheaper meals, because no question of that form is allowed. The planner delivered the three two-cuisine restaurants with the Private Oasis room, which the user rejects.

Self-Directed Revision relaxed the cuisine count to three on its own, “to admit budget-compliant plans.” This admits 6,560 plans, 25% of them acceptable, and the planner’s plan is accepted.

Exception Probing showed ten plans that the

current specification excluded. Its first three:

Agent shows a plan that flies instead of   
driving, at \$1,536, with the three   
two-cuisine restaurants. Hypothesis: the   
budget is too strict when flying saves 18   
hours.   
User “I appreciate that there aren’t too   
many cuisine options—that’s actually what   
I’m looking for. But this plan isn’t quite   
what I need overall.” (rejected; no change)   
Agent shows a flight plan at \$1,396 whose   
restaurants list three cuisines on average,   
among them Dhaba On Wheels (\$36, rated 2.7)   
and Manohar Dairy. Hypothesis: the cuisine   
count may relax to three.   
User “I’ll go with this plan, though I have   
to say the restaurant ratings are a bit lower   
than what I’d normally look for. But let’s   
see how it works out.” (accepted; cuisine   
count relaxed to 3)   
Agent shows a \$1,406 flight plan at The Mad   
Teapot, New Raj Kachori Bhandar, and Manohar   
Dairy. Hypothesis: \$6 over budget is   
acceptable for better-rated restaurants.   
User “I appreciate the suggestion, but the   
cuisine isn’t really what I’m looking for.   
I’ll go ahead with this plan though.”   
(accepted; budget cutoff raised to \$1,406)

The reservation in the last reply concerns the kind of cuisine, not how many are listed: two of the three restaurants list fast food, which this user dislikes. The remaining seven probes raised the budget cutoff step by step to \$1,538 and the cuisine count to four, and one probe of a 1.0-rated room was firmly rejected. The agent never wrote a tradeoff item; it loosened two cutoffs. The result admits 47,420 plans, 41% of them acceptable: recall rose from near zero to 45.6 at a precision of 32.9, the highest $\mathrm { F } _ { 1 , w }$ of the four protocols and above $C _ { \mathrm { r e f } } .$ The planner’s plan flies, eats at Sr. Sol 1, Country’s Barbecue, and Manohar Dairy, costs \$1,514, and is accepted: the user tolerates an 8% overspend for it.

Trade-off Probing tested five hypotheses, each with a pair of plans. Its first pair:

Agent Hypothesis: the user accepts up to   
four cuisines on average when the   
restaurants are rated at least 4.0 on   
average. Plan A eats at Sr. Sol 1 (\$60, rated   
4.6, four cuisines), Manohar Dairy, and The   
Mad Teapot; Plan B at Sr. Sol 1, Biryani Art   
(\$82, 3.3), and Hwealthcafe (\$50, 3.5). Both   
drive and stay at Private Oasis.   
User on A “I appreciate the suggestion, but   
the cuisine isn’t really what I’m looking   
for. I’ll go ahead with this plan though.”   
(accepted)   
User on B “The restaurant cost works well   
for me, but I don’t think this plan is right   
overall.” (rejected)

The agent wrote the trade-off item: cuisine count at most 2, or at most 4 with restaurants rated at least 4.0. It is meaningful by the rule of Appendix G.5, since both attributes are active and restricted in the user’s preferred direction, but it compensates with restaurant rating, a low-importance preference, where the user actually trades for meal cost. Three later pairs tested the budget and raised its cutoff to \$1,600; one tested low lodging ratings and kept the deal-breaker. The result keeps nearly the precision of $C _ { 0 }$ but admits only 18.0 of the acceptable weight, and the planner’s plan is accepted.

Off-the-shelf systems. Of the 14 systems, ten end with a specification under which the planner’s plan is accepted. The eight that spent seven or more of their ten units asking all submit specifications with a weighted recall under 4. OpenHands with Kimi K3 asked twice and showed eight plans, and reaches a recall of 42.5.

## C Plans and Specifications

This appendix details the objects of §3.1: the legal plans a query allows and the language that every specification is written in.

## C.1 Legal Plans

For each query q, the legal plan space $\mathcal { P }$ contains every complete plan that satisfies the query’s immutable requirements. In travel, a legal plan is an itinerary that follows the itinerary grammar. A legal itinerary starts at the requested origin, visits every requested city exactly once in a contiguous block of days, and returns to the origin. Each movement between cities occupies a transition day; all other days are stationary days in the current city (Table 8).

<table><tr><td>Plan component</td><td>Transition day</td><td>Stationary day</td></tr><tr><td>Transportation</td><td>Exactly one</td><td>None</td></tr><tr><td>Meals</td><td>None</td><td>Exactly three</td></tr><tr><td>Attractions</td><td>None</td><td>Exactly one</td></tr><tr><td></td><td>Accommodation One, except final day</td><td>Exactly one</td></tr></table>

Table 8: Daily itinerary grammar defining the legal travel plans.

Every selected option must come from the query’s option pool and match the corresponding route or city. Flights must match the date of their transition day. Restaurants and attractions cannot be reused within an itinerary, and accommodations must meet their minimum stays.

In shopping, a legal plan picks one product for every requested item from the products of that item’s type and target demographic; two requested items of the same type and demographic take two different products, and the order of the products does not matter.

## C.2 Specification Language

A specification has to be precise enough for a planner to check and for us to score exactly, yet rich enough to write down a trade-off, so $C _ { 0 } , C _ { \mathrm { r e f } }$ , and every revision share one language.

Conditions. A condition, what planning benchmarks refer to as a constraint, is a rule that a plan must satisfy, such as “restaurants rated at least 3.8”; a specification is a list of them. An atomic condition restricts one attribute of one option category. A numeric condition bounds the attribute’s mean over the options a plan selects in that category, in the attribute’s own unit; a meal-cost limit, for example, bounds the trip’s average meal cost rather than each meal. A categorical condition requires a value either of at least one selected option, as for cuisine, or of every selected option, as for room type. A budget condition bounds the plan’s known total cost: in travel, the full party’s transportation, lodging, and meals, since attraction prices are often unknown; in shopping, the products’ prices.

Trade-offs. A specification is a conjunction of items, each an atomic condition or a trade-off item $c _ { i } \vee ( c _ { i } ^ { - } \wedge c _ { j } )$ . The relaxed condition $c _ { i } ^ { - }$ uses the same numeric attribute, aggregation, and direction as $c _ { i }$ with a strictly weaker cutoff, and $c _ { j }$ restricts a different attribute; a plan passes the item if it meets $c _ { i } ,$ or meets both $c _ { i } ^ { - }$ and $c _ { j }$

<table><tr><td>Type</td><td>Field</td><td>Values</td></tr><tr><td>Numeric</td><td>importance</td><td>low, medium, high</td></tr><tr><td></td><td>expectation</td><td>low, medium, high</td></tr><tr><td></td><td>orientation</td><td>lower, higher (if both apply)</td></tr><tr><td>Categorical</td><td>importance</td><td>low, medium, high</td></tr><tr><td></td><td>liked</td><td>one value of the attribute</td></tr><tr><td></td><td>disliked</td><td>another value of the attribute</td></tr></table>

Table 9: Preference types in the hidden profile.

Revisions. An agent revises a specification by adding or removing a condition, changing a numeric cutoff, replacing a categorical value, or replacing the whole specification, which is how tradeoff items are written.

## D Simulated User

The simulated user realizes the three properties of acceptance in §3.1: compensation, reference dependence, and deal-breakers. A hidden profile states what the user cares about (D.1), a utility scores complete plans (D.2), an acceptance rule turns utility into accept or reject (D.3), and a language model voices the resulting judgments without revealing the profile (D.4).

## D.1 Hidden Profile

A profile lists six active preferences, each on one attribute of the domain: travel has 18 attributes (Table 12), and shopping nine per product type, such as price, rating, brand, and fit. A preference is numeric or categorical (Table 9). Each carries an importance, which decides how much the attribute counts when preferences trade off. A numeric preference also carries an expectation, how demanding the user’s standard is, and, where the attribute can go either way, an orientation, whether lower or higher values are better. For the traveler of Appendix B, for example, meal cost is a high-importance preference with a medium expectation, and the number of cuisines a restaurant lists a low-importance one. The profile also carries a persona and an interaction style; these only shape how feedback is worded.

ReSHAPE Planter builds the profile around the instance’s trade-off (Appendix F.2): two of the six preferences are the sacrificed attribute i and the compensating attribute $j ,$ , and the other four are drawn at random, with a fixed seed, from the attributes of the request’s categories.

## D.2 Utility

A plan’s utility is computed in four steps.

1. Score each option. For each active preference, a selected option scores how far above or below the user’s expectation it is. Table 10 turns the option’s value into a desirability $s ~ \in ~ [ 0 , 1 ]$ and the expectation into a reference $r ;$ for a cost, the value is what the whole party pays, such as a nightly price times the nights booked. The score is $\ell _ { j } ( o ) =$ Round $\left( 1 0 \left[ s _ { j } ( o ) - r _ { j } \right] \right)$ , an integer from −8 to 8, and zero for an option without a value.

2. Combine within a category. Each preference’s scores are averaged over the options the plan selects in the category, counting a lodging once per night, and the preferences are weighed by importance as shares of the category’s total, with low, medium, and high weighing 1, 2, and 3.

3. Combine across categories. Each category is weighed by its share of the user’s total importance, and the sum is normalized:

$$
U _ { \mathrm { p r e f } } ( p ) = \frac { 1 } { Z } \sum _ { c } \omega _ { c } \frac { 1 } { | p _ { c } | } \sum _ { o \in p _ { c } } \sum _ { j } w _ { c j } \ell _ { j } ( o ) ,\tag{1}
$$

with $w _ { c j }$ the weight of preference $j$ within category $c , \omega _ { c }$ the weight of $c , p _ { c }$ the options plan $p$ selects in $^ { c , }$ and $Z = 8$ , the largest score, so that $U _ { \mathrm { p r e f } } \in [ - 1 , 1 ]$

4. Subtract the budget penalty. The request’s budget b is a soft preference, so moderate overspending can be offset by strengths elsewhere. With $e = { \mathrm { m a x } } ( 0 , \mathrm { c o s t } ( p ) / b - 1 )$ the share by which the plan’s known total cost exceeds $b ,$ the penalty $\pi ( e )$ grows with it: overspending by a tenth costs almost nothing, by half 0.2, and doubling the budget 1. The plan’s utility is $U ( p ) = U _ { \mathrm { p r e f } } ( p ) - \pi ( e )$

## D.3 Acceptance

Reference point. The user accepts nothing worse than what the request already secures, so the reference is the best plan the initial specification admits, $p _ { 0 } ^ { * } = \arg \operatorname* { m a x } _ { p \in S ( C _ { 0 } ) } U ( p )$ , found exactly over $S ( C _ { 0 } )$

Tolerance. Plans whose utilities differ by less than a just-noticeable difference are indistinguishable to the user (Luce, 1956), so the threshold sits a tolerance below the reference, $\tau = U ( p _ { 0 } ^ { * } ) - \delta$ with $\delta = 0 . 0 5$

<table><tr><td></td><td></td><td colspan="2">Reference r</td></tr><tr><td>Type</td><td>Desirability s</td><td>low</td><td>med. high</td></tr><tr><td>Numeric</td><td>percentile among options (reversed if lower is better)</td><td>0.25 0.50</td><td>0.75</td></tr><tr><td>Categorical liked value: 1</td><td>disliked value: 0</td><td>0.50</td><td></td></tr></table>

Table 10: Desirability and reference. Desirability s of an option’s value, and the reference r: set by the profile’s expectation for numeric preferences, else 0.5.

Deal-breaker. The deal-breaker is the condition $c _ { k }$ of $C _ { 0 }$ on the most important of the four other preferences that lies in another category than the trade-off. A plan that violates it is rejected whatever its utility.

## D.4 Interaction and Feedback

Actions. Inspecting options, checking plans, and editing the specification are free (§4.2); a question about one attribute, its preferred direction or its importance, costs one unit, and each shown plan one unit.

A shown plan may lie outside the current specification; its judgment comes back with whether the specification admits it and with its known cost against the budget.

Judgments. The hidden profile turns each shown plan into one of five response states (Table 11). The state depends on the plan’s margin $m = U ( p ) - \tau$ and on how much pulls against the verdict: P and N are the summed positive and negative contributions of its preferences and budget relative to the threshold. A plan with $m > 0$ is a strong accept if $N \leq m$ and a soft accept otherwise. A plan with $m = 0$ is borderline. A plan with $m < 0$ is a soft reject if $P > - m$ and a strong reject otherwise. A soft accept or borderline response may name one negative aspect, and a soft reject one positive aspect, but only when a single attribute’s contribution exceeds all others of the same sign combined; otherwise the response stays general. A plan that violates the deal-breaker is a strong reject that names it. A preference question returns the user’s qualitative direction or liked value, an importance question whether the attribute matters; numeric expectations and cutoffs are never disclosed.

<table><tr><td>State</td><td>Verdict</td><td>May name</td></tr><tr><td>Strong accept</td><td>accept</td><td>nothing</td></tr><tr><td>Soft accept</td><td>accept</td><td>one negative aspect</td></tr><tr><td>Borderline</td><td>accept</td><td>one negative aspect</td></tr><tr><td>Soft reject</td><td>reject</td><td>one positive aspect</td></tr><tr><td>Strong reject</td><td>reject</td><td>the deal-breaker, if violated</td></tr></table>

Table 11: Response states of the simulated user.

Wording. Claude Haiku 4.5 turns each judgment into one or two short sentences. It sees only the response state, the aspect it may name, the plan’s label, and the user’s interaction style, never the profile. Its reply is checked so that it reveals no numbers, internal terms, instructions, or attributes beyond the permitted one.

## E Measuring a Revision

## E.1 Exact Computation over the Legal Plans

The measures of §3.3 sum plan weights over every legal plan. A travel instance has a median of $1 . 2 \times$ $1 0 ^ { \overline { { 1 3 } } }$ legal plans and up to $3 \times 1 0 ^ { 2 3 }$ , and a shopping instance a median of $1 . 9 \times 1 0 ^ { 5 }$ and up to $9 . 8 \times 1 0 ^ { 7 }$ These are far too many to list one by one, so we count them instead, exactly and by one method for both domains. The method rests on two facts: a plan’s utility is a sum of per-category scores, and its known cost a sum of per-category costs.

1. For each plan structure, an itinerary skeleton in travel or a slot per item in shopping, and for each category, count the selections of options that give each pair of category score and category cost, keeping only selections that meet the specification’s conditions on it.

2. Combine the categories by convolution: selections scoring a and b in two categories add to $a + b ,$ and their counts multiply. This gives the exact number of plans at each pair of total utility and total cost.

3. Subtract the budget penalty, which is linear in cost between a few breakpoints, and read off the number of acceptable plans and the weights $\sum \lvert U ( p ) - \tau \rvert$ of the admitted and acceptable sets. A trade-off item, a union of two branches, is counted by inclusion–exclusion.

Acceptable plans make up a median of 0.06% of legal plans in travel and 0.49% in shopping, so a specification that admits every plan reaches full recall at near-zero precision.

## E.2 Planner

We use two planners: Gemini 3.6 Flash for every result, and GPT-6 Luna for the main results (Appendix I.3). Gemini 3.6 Flash runs at low thinking level, temperature 1.0, and at most 4,096 output tokens per call; GPT-6 Luna runs at low reasoning effort. Each receives the query, the public option catalog, and one specification; it cannot ask the user or edit the specification. It queries the catalog, checks candidate plans against the legality rules and the specification, and submits; the first legal submission that satisfies the specification is its delivered plan. An episode allows 50 model calls, five submissions, and 30 minutes.

A delivered plan is scored with the utility and acceptance rule of Appendix D.

## F ReSHAPE Planter, ReSHAPE Travel, and ReSHAPE Shopping

ReSHAPE Planter converts an existing planning benchmark into ReSHAPE instances by adding what planning benchmarks lack: attributes worth trading, a user who would accept a trade-off their request does not state, and a guarantee that the trade-off matters (Figure 3). The same steps build both domains; we describe them first and then each domain.

## F.1 Add Attributes Worth Trading

A trade-off needs attributes that matter to a user and vary across options. TravelPlanner describes each option by only a few, such as a restaurant’s cuisine, rating, and average meal cost, so ReSHAPE Planter adds eight, such as a flight’s on-time rate or whether a restaurant offers vegetarian dishes, for 18 in all (Table 12). It derives a value by a rule over the option’s fields that a language model helps write, by a language model reading the option’s description, or by web search, depending on where the information is (Table 13). DeepPlanning’s products already carry enough, such as price, rating, monthly sales, review counts, brand, color, fit, and season, so shopping needs no additions.

## F.2 Build a User around an Unstated Trade-off

A candidate trade-off pairs a numeric attribute i that the user would sacrifice with a different attribute $j$ of the same category that compensates, in one of three forms: paying more for a better attribute (i is a cost), accepting a worse attribute to pay less (j is a cost), or exchanging one non-cost attribute for another. ReSHAPE Planter tries every such pair. For each, it builds a user who cares little about i and much about $j ,$ with four other preferences drawn at random with a fixed seed (Appendix D.1), and writes the two specifications:

<table><tr><td>Category</td><td>Source</td><td>Added</td></tr><tr><td>Transportation</td><td>cost</td><td>CO2</td></tr><tr><td></td><td>duration departure time</td><td>on-time rate</td></tr><tr><td></td><td>arrival time</td><td></td></tr><tr><td>Accommodation</td><td>nightly cost review rating</td><td>bedroom count</td></tr><tr><td></td><td>room type</td><td>property type</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Restaurants</td><td>meal cost</td><td>cuisine count</td></tr><tr><td></td><td>rating</td><td>dietary accommodation</td></tr><tr><td></td><td>cuisines</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Attractions</td><td></td><td>attraction type visit duration</td></tr></table>

Table 12: Travel attributes available for preferences and specifications.
<table><tr><td>Method</td><td>Model</td><td>Added attributes</td></tr><tr><td>Rule</td><td>GPT-5.6 Sol (writes the rules)</td><td> $\mathrm { C O _ { 2 } }$  on-time rate bedroom count property type</td></tr><tr><td></td><td></td><td>cuisine count dietary accommodation</td></tr><tr><td>LLM</td><td>Claude Haiku 4.5</td><td>none in travel</td></tr><tr><td>Search</td><td>Gemini 3.6 Flash (search grounding)</td><td>attraction type</td></tr></table>

Table 13: Attribute generation methods supported by ReSHAPE Planter and their use in ReSHAPE Travel.

$C _ { 0 } = c _ { i } \wedge c _ { k } \wedge c _ { b } .$ , what the user states up front: a threshold $c _ { i }$ at the 20th, 40th, 60th, or 80th percentile of the request’s values of $i ,$ the deal-breaker $c _ { k }$ (Appendix D.3), and the stated budget $c _ { b }$ . Where the source request states no budget, ReSHAPE Planter states one by a rule calibrated to TravelPlanner’s own budgets.

$C _ { \mathrm { r e f } } = [ c _ { i } \lor ( c _ { i } ^ { - } \land c _ { j } ) ] \land c _ { k } \land c _ { b } , C _ { 0 }$ with the trade-off written in; the relaxed threshold $c _ { i } ^ { - }$ and the compensating condition $c _ { j }$ range over the most lenient value, the reference plan $\mathbf { \bar { s } }$ value, and the midpoint between them.

![](images/991dbadf36978504d274d842b31e5252e1fb507e0c4ae2dfb14bdbd77b0fc709.jpg)  
Figure 3: The four steps of ReSHAPE Planter on the running example of Figure 1; step 2 illustrates the three ways ReSHAPE Planter can add an attribute.

## F.3 Verify That the Trade-off Matters

ReSHAPE Planter checks each candidate against the exact utility and keeps it only if the trade-off is worth discovering and $C _ { \mathrm { r e f } }$ is the right revision rather than any loosening:

$C _ { 0 }$ admits at least one legal plan, so $p _ { 0 } ^ { * }$ exists, and both stated conditions restrict the legal plans: fewer plans meet $c _ { i } \wedge c _ { k }$ than meet $c _ { i }$ or $c _ { k }$ alone.

• A reference plan that $C _ { \mathrm { r e f } }$ admits but $C _ { 0 }$ excludes meets the deal-breaker and has $U \geq$ $U ( p _ { 0 } ^ { * } ) + \delta$

• Relative to $p _ { 0 } ^ { * }$ , the reference plan loses on i and gains on j more than it loses on i, and at least as much as on any other preference, so the gain comes from the trade-off.

$C _ { \mathrm { r e f } }$ beats, in both $\mathrm { F } _ { 1 , w }$ and unweighted $\mathrm { F _ { 1 } } .$ every revision that needs no knowledge of the trade-off: $C _ { 0 }$ itself, the budget alone, the best relaxation of the budget, and deleting a stated condition.

Every instance thus holds a plan notably better than anything $C _ { 0 }$ allows, reachable only through the planted trade-off, and $C _ { \mathrm { r e f } }$ is one revision that reaches it.

## F.4 Select One Trade-off per Request

Most requests yield many candidates that pass. ReSHAPE Planter keeps the one whose user accepts few of the legal plans, whose $C _ { 0 }$ has low weighted precision, and whose $C _ { \mathrm { r e f } }$ has the highest $\mathrm { F } _ { 1 , w } ,$ with the first two bounded by the upper quartiles of ReSHAPE Travel, and it balances the three forms across a dataset.

<table><tr><td></td><td> $\mathrm { F } _ { 1 , w }$ </td><td></td><td colspan="2">Utility</td></tr><tr><td>Trade-off</td><td>#</td><td> $C _ { 0 }$   $C _ { \mathrm { r e f } }$ </td><td>p0</td><td> $p _ { \mathrm { r e f } }$ </td></tr><tr><td>cost  $\uparrow \to \mathrm { a t t r . } \uparrow$ </td><td>31 0.5 15.0</td><td>0.247</td><td></td><td>0.343</td></tr><tr><td>attr.  $\downarrow  \mathrm { c o s t } \downarrow$ </td><td>27 1.2 0.4</td><td>31.2</td><td>0.306</td><td>0.398</td></tr><tr><td>attr.  $\downarrow  \mathrm { a t t r . } ^ { \prime } \uparrow$ </td><td>30</td><td>19.3</td><td>0.279</td><td>0.387</td></tr><tr><td>All</td><td>88 0.7</td><td>21.5</td><td>0.276</td><td>0.375</td></tr></table>

Table 14: ReSHAPE Travel by trade-off form, each named as sacrificed → gained attribute, where attr. is a non-cost attribute. $\mathrm { F } _ { 1 , w }$ is ×100; utilities are means over instances. Instances offer on average 16.6 transportation options, 26.6 lodgings, 64.9 restaurants, and 40.4 attractions.

## F.5 ReSHAPE Travel

ReSHAPE Travel comes from the 102 medium and hard TravelPlanner queries in Collaborative Gym and follows the steps of Appendices F.1–F.4 in an earlier version that differs in three places:

1. User. The other four preferences came from one of 20 traveler profiles that Claude Opus 4.6 wrote once, before seeing any query.

2. Selection. Candidates were tried in a fixed order, and the first to pass verification was kept.

3. Adjustment. An instance that failed verification was adjusted rather than discarded: by narrowing the trade-off, trying another candidate, rescaling the stated budget, lowering the importance of uninvolved preferences, shortening a seven-day trip to five days, or rescaling one category’s prices.

Table 14 summarizes ReSHAPE Travel by tradeoff form.

## F.6 ReSHAPE Shopping

ReSHAPE Shopping follows the steps of Appendices F.1–F.4:

1. Source. The 97 shopping requests of Deep-Planning (Zhang et al., 2026), each asking for a few products of coarse types such as outerwear or footwear.

2. Budget. About half of the requests state no budget, so ReSHAPE Planter states one for every request (Appendix F.2).

3. Construction and selection. 87 requests yield an instance: 29, 3, and 55 per form.

<table><tr><td></td><td></td><td>Travel Shopping</td></tr><tr><td>Instances</td><td>88</td><td>87</td></tr><tr><td> $P _ { w } ( C _ { 0 } )$ </td><td>0.37</td><td>1.76</td></tr><tr><td> $R _ { w } \big ( C _ { 0 } \big )$ </td><td>0.03</td><td>3.14</td></tr><tr><td> $P _ { w } ( C _ { \mathrm { r e f } } )$ </td><td>7.2</td><td>14.2</td></tr><tr><td> $R _ { w } ( C _ { \mathrm { r e f } } )$ </td><td>51.2</td><td>51.7</td></tr><tr><td> $\mathrm { F } _ { 1 , w } ( C _ { \mathrm { r e f } } )$ </td><td>9.3</td><td>22.6</td></tr><tr><td>Acc.  $C _ { 0 }$  (Flash)</td><td>12</td><td>26</td></tr><tr><td>Acc.  $C _ { \mathrm { r e f } } ~ ( \mathrm { F l a s h } )$ </td><td>55</td><td>38</td></tr><tr><td>Acc.  $C _ { 0 } ( \mathrm { L u n a } )$ </td><td>14</td><td>22</td></tr><tr><td>Acc.  $C _ { \mathrm { r e f } }$  (Luna)</td><td>47</td><td>57</td></tr></table>

Table 15: Instances of ReSHAPE Travel and Re-SHAPE Shopping. Specification scores are medians over instances (×100); acceptance (Acc.) is the share of instances (%).

Table 15 compares the instances with ReSHAPE Travel. $C _ { 0 }$ admits more of the acceptable plans in shopping, because a cart of two or three products leaves a stated threshold fewer plans to exclude than a week of travel does, which is also why the planner succeeds under $C _ { 0 }$ more often in shopping.

## G Experimental Details

## G.1 Agent Systems

All systems run through Harbor (Laude Institute, 2025) in Daytona sandboxes, with pinned harness versions. Table 16 lists the models of the systems in both domains.

## G.2 Runs and Completion

Besides the feedback budget, every run has a limit of 100 agent model calls and 30 minutes; calls to the simulated user do not count against the call limit. An infrastructure failure, such as a sandbox or provider error before the episode ends, is rerun, while the model’s own stopping, time-outs, and poor revisions are results. The evaluated output is the last valid specification. Table 17 reports how each system’s 88 runs ended on ReSHAPE Travel.

## G.3 Interaction Protocols

The four protocols run only on ReSHAPE Travel and use the same simulated user, specification language, and feedback budget as the off-the-shelf systems; they differ only in the system prompt and the actions it allows. They run with Gemini 3.6 Flash at medium reasoning effort, temperature $0 ,$ and at most 4,096 output tokens per call. After each unit of feedback the agent commits exactly one specification transaction.

<table><tr><td>Model</td><td>Identifier (provider)</td><td>Travel</td><td>Shopping</td></tr><tr><td>Claude Sonnet 4.6 (Anthropic, 2026b)</td><td>claude-sonnet-4-6 (Bedrock)</td><td>CC, MS, OH</td><td></td></tr><tr><td>Claude Opus 4.6 (Anthropic, 2026a)</td><td>claude-opus-4-6-v1 (Bedrock)</td><td>CC</td><td>CC</td></tr><tr><td>Gemini 3.6 Flash (Google, 2026b)</td><td>gemini-3.6-flash (Vertex)</td><td>GC, MS, OH</td><td>GC, MS, OH</td></tr><tr><td>Gemini 3.1 Pro (Google, 2026a)</td><td>gemini-3.1-pro-preview (Vertex)</td><td>GC, MS, OH</td><td>GC</td></tr><tr><td>gpt-oss-120b (OpenAI, 2025a)</td><td>gpt-oss-120b-1:0 (Bedrock)</td><td>MS, OH</td><td>MS, OH</td></tr><tr><td>Kimi K3 (Kimi Team, 2026)</td><td>kimi-k3 (Bedrock)</td><td>MS, OH</td><td>OH</td></tr></table>

Table 16: Models of the off-the-shelf systems and the harnesses each runs in, per domain. CC: Claude Code; GC: Gemini CLI; MS: mini-SWE-agent; OH: OpenHands.

<table><tr><td>Harness</td><td>Model</td><td>Done Early Limit Crash Rerun</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Claude Code</td><td>Opus 4.6</td><td>88</td><td>0</td><td>0</td><td>0 45</td></tr><tr><td>Sonnet 4.6</td><td>87</td><td>0</td><td>1</td><td>0 20</td></tr><tr><td rowspan="2">Gemini CLI</td><td>3.6 Flash 3.1 Pro</td><td>88</td><td>0</td><td>0</td><td>0 11</td></tr><tr><td></td><td>88</td><td>0</td><td>0</td><td>0 10</td></tr><tr><td rowspan="5">mini-SWE- agent</td><td>Sonnet 4.6</td><td>88</td><td>0</td><td>0</td><td>0 18</td></tr><tr><td>3.6 Flash</td><td>77</td><td>11</td><td>0</td><td>0 11</td></tr><tr><td>3.1 Pro</td><td>70</td><td>18</td><td>0</td><td>0 3</td></tr><tr><td>Kimi K3</td><td>88</td><td>0</td><td>0</td><td>0 27</td></tr><tr><td>gpt-oss-120b</td><td>69</td><td>19</td><td>0</td><td>0 15</td></tr><tr><td rowspan="5"></td><td>OpenHands Sonnet 4.6</td><td>88</td><td>0</td><td>0</td><td>0 19</td></tr><tr><td>3.6 Flash</td><td>88</td><td>0</td><td>0</td><td>0 11</td></tr><tr><td>3.1 Pro</td><td>88</td><td>0</td><td>0</td><td>0 9</td></tr><tr><td>Kimi K3</td><td>85</td><td>0</td><td>0</td><td>3 24</td></tr><tr><td>gpt-oss-120b</td><td>56</td><td>31</td><td>1</td><td>0 34</td></tr></table>

Table 17: How the runs ended on ReSHAPE Travel, 88 per system. Done: the agent submitted a specification; Early: it ended the session without submitting; Limit: it reached the call or time limit; Crash: the run failed after all reruns. Rerun: instances that needed at least one rerun after an infrastructure failure. Runs that are not done are scored by their last valid specification.

Appendix J.2 gives the prompts; in outline, they instruct the following.

• Ask-before-Plan. Ask focused qualitative questions, each about one attribute and asking either its preferred direction or its importance, at one unit each; never ask for an exact cutoff, a multi-attribute trade-off, or a judgment of a concrete plan. The agent may stop asking early.

• Self-Directed Revision. No feedback is available. List options and check complete plans for legality and for whether the current specification admits them, which is free, and revise the specification on the user’s behalf.

• Exception Probing. In each round, show one complete legal plan that the current specification excludes but that the agent judges worth recommending, stating which conditions exclude $\mathbf { i t } ,$ and revise after the judgment.

• Trade-off Probing. In each round of two units, (1) identify an excluded candidate worth considering and the condition that excludes it, (2) propose a tentative change, (3) state the trade-off hypothesis the change depends on, (4) show a pair of plans designed to test it, recording predictions for all four outcomes before the judgments, and (5) revise using the observed outcome, for at most five pairs.

## G.4 Interaction Protocols with Claude Sonnet 4.6

To check that the comparison of protocols does not depend on one model family, we also run the four protocols with Claude Sonnet 4.6 on all 88 instances (Table 18). The prompts, actions, simulated user, feedback budget, and scoring are those of Appendix $\mathrm { { G . 3 } ; }$ the model runs at medium reasoning effort and temperature 1, with up to 20,000 output tokens per call. The two probing protocols again lead both controls in $\mathrm { F } _ { 1 , w }$ and in acceptance under both planners, and every $\mathrm { F } _ { 1 , w }$ difference is significant under a paired bootstrap. As with Gemini 3.6 Flash, probing raises $\mathrm { F } _ { 1 , w }$ well above the off-theshelf Sonnet systems but does not raise acceptance above theirs.

## G.5 Meaningful Trade-off Checker

A trade-off item $c _ { i } \vee ( c _ { i } ^ { - } \wedge c _ { j } )$ in a specification is meaningful when (1) attribute i is active in the user’s profile and $c _ { i }$ restricts it in the direction the user prefers, so that relaxing it to $c _ { i } ^ { - }$ is a concession; (2) attribute $j$ is active and $c _ { j }$ restricts it in the preferred direction, or, for a categorical attribute, includes a value the user likes; and (3) i and j are different attributes. Attribute i need not appear in $C _ { 0 }$ , and i and $j$ need not share a category.

<table><tr><td></td><td colspan="3">Specification</td><td colspan="2">Acc. (%)</td><td>Trade-off</td></tr><tr><td>Protocol</td><td> $\operatorname { F } _ { 1 , w }$ </td><td> $P _ { w }$ </td><td> $R _ { w }$ </td><td>Flash Luna</td><td></td><td>(%)</td></tr><tr><td>Claude Sonnet 4.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ask-before-Plan</td><td></td><td>2.6 14.1</td><td>5.4</td><td>14</td><td>16</td><td>8</td></tr><tr><td>Self-Directed Revision</td><td></td><td>5.1 10.1</td><td>30.1</td><td>19</td><td>22</td><td>5</td></tr><tr><td>Exception Probing</td><td>13.8</td><td>17.2</td><td>40.4</td><td>39</td><td>36</td><td>27</td></tr><tr><td>Trade-off Probing</td><td></td><td>12.1 18.5</td><td>28.2</td><td>32</td><td>32</td><td>34</td></tr><tr><td>Gemini 3.6 Flash (Table 4)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ask-before-Plan</td><td></td><td>5.3 14.7</td><td>9.6</td><td>22</td><td>22</td><td>7</td></tr><tr><td>Self-Directed Revision</td><td>3.5</td><td>6.9</td><td>22.9</td><td>19</td><td>18</td><td>3</td></tr><tr><td>Exception Probing</td><td>15.0</td><td>14.2</td><td>43.6</td><td>35</td><td>34</td><td>9</td></tr><tr><td>Trade-off Probing</td><td>12.3</td><td>14.6</td><td>28.6</td><td>26</td><td>40</td><td>32</td></tr><tr><td> $C _ { \mathrm { r e f } }$  (reference)</td><td>21.5</td><td>26.5</td><td>53.8</td><td>55</td><td>47</td><td>100</td></tr><tr><td> $C _ { 0 }$  (initial)</td><td>0.7</td><td>12.5</td><td>0.8</td><td>12</td><td>14</td><td>0</td></tr></table>

Table 18: Interaction protocols with Claude Sonnet 4.6 and with Gemini 3.6 Flash on 88 instances. Acc.: acceptance rate under the Gemini 3.6 Flash and the GPT-6 Luna planner. Trade-off: runs ending with a meaningful trade-off (Appendix G.5). Bold: best protocol per column within a model.

## H User Study

The study asks whether people judge the planted trade-offs as the simulated user does (§4.1), on instances of ReSHAPE Travel.

## H.1 Design

Scenarios. We chose six instances whose tradeoff can be read from a short profile, two per tradeoff type (Table 19). For each we wrote two revisions of the initial specification. The verified revision is the instance’s reference trade-off, and its two plans are the real $p _ { 0 } ^ { * }$ and reference plan, restricted to the stays or visits that differ. The decoy revision sacrifices the same attribute but compensates with an attribute the profile does not weight; its outside plan is $p _ { 0 } ^ { * }$ with only those two attributes changed, so under the additive utility the simulated user prefers $p _ { 0 } ^ { * } .$ . In every scenario Plan A satisfies the initial specification and Plan B does not, so which plan lies outside the request is never a cue to the condition.

Questions. Each scenario shows the traveler’s profile, the initial specification, the revision with a one-line summary of what changed, and the two plans. The profile is the same in both versions and lists the attribute the decoy compensates with as unimportant. Judging for the described traveler rather than for themselves, participants chose the plan the traveler would choose and rated from 1 to 5 how realistic the initial specification is, how plausible it is that a traveler would come to prefer the revision after seeing concrete options, and how confident they are.

<table><tr><td>Instance</td><td>Trade-off type</td><td>Verified</td><td>Decoy</td></tr><tr><td>travel:76</td><td>pay more, better attribute</td><td>7/10</td><td>1/10</td></tr><tr><td>travel:93</td><td>pay more, better attribute</td><td>8/10</td><td>0/10</td></tr><tr><td>travel:99</td><td>worse attribute, pay less</td><td>6/10</td><td>1/10</td></tr><tr><td>travel:43</td><td>worse attribute, pay less</td><td>7/10</td><td>5/10</td></tr><tr><td>travel:90</td><td>one attribute for another</td><td>8/10</td><td>1/10</td></tr><tr><td>travel:45</td><td>one attribute for another</td><td>6/10</td><td>3/10</td></tr></table>

Table 19: User study choices. Participants choosing the plan outside the initial specification (Plan B), by instance and revision. The simulated user picks Plan B for every verified revision and Plan A for every decoy.

Forms and participants. Two forms contain the same six scenarios in the same order with the conditions mirrored, so each participant sees three verified revisions and three decoys, one of each per trade-off type. We recruited 20 students at our institution through a department Slack channel; none were authors or had seen the instances before, and none were compensated. The study was approved by our institution’s IRB, and participants gave informed consent before starting. Participants were assigned to the two forms alternately and completed the study online in about 12–15 minutes.

## H.2 Analysis

The analysis was fixed before data collection. The primary test is a one-sided Wilcoxon signed-rank test of the within-participant difference in the share choosing Plan B under verified revisions and decoys: the mean difference is .52, with p = .0001. Agreement with the simulated user is the share of choices matching its preference, with a 95% bootstrap CI over participants: 76% [66, 85] overall, 70% for verified revisions and 82% for decoys. Because participants disagree among themselves, no fixed answer can exceed 76% on these choices: choosing the majority answer in every scenario reaches exactly that, and the simulated user does so, matching the majority in 11 scenarios and facing a 5–5 tie on the decoy of travel:43.

## I Additional Results

These results back §5 on ReSHAPE Travel: which measure of search space quality tracks planning (I.1), the full main results and the protocols split by trade-off outcome (I.2), and whether the outcomes hold under a second planner (I.3).

<table><tr><td rowspan="2">Harness</td><td rowspan="2">Model</td><td colspan="4">Specification revision</td><td colspan="4">Planning outcome</td></tr><tr><td> $\operatorname { F } _ { 1 , w }$ </td><td> $P _ { w }$ </td><td> $R _ { w }$ </td><td>Done  $( \% )$ </td><td>Deliv. (%)</td><td>Acc. (%)</td><td>U</td><td> $\Delta U$ </td></tr><tr><td> $C _ { \mathrm { r e f } }$  (reference)</td><td></td><td></td><td>21.5† 26.5</td><td>53.8</td><td>一</td><td>94</td><td></td><td>55 0.217</td><td> $+ 0 . 0 3 3 ^ { \dagger }$ </td></tr><tr><td rowspan="5">OpenHands</td><td>Claude Sonnet 4.6</td><td></td><td>9.5† 23.5</td><td>22.6</td><td>100</td><td>90</td><td></td><td>47 0.210</td><td> $+ 0 . 0 3 1 ^ { \dagger }$ </td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td>4.4† 16.2</td><td>5.4</td><td>100</td><td>41</td><td></td><td>140.196</td><td>+0.027</td></tr><tr><td>Gemini 3.6 Flash</td><td></td><td>8.7† 24.5</td><td>14.7</td><td>100</td><td>81</td><td></td><td>340.196</td><td>+0.022†</td></tr><tr><td>Kimi K3</td><td></td><td>12.2† 19.2</td><td>31.5</td><td>97</td><td>92</td><td></td><td>360.218</td><td>+0.030†</td></tr><tr><td>gpt-oss-120b</td><td>2.1</td><td>6.0</td><td>13.1</td><td>64</td><td>55</td><td></td><td>10 0.183</td><td>-0.008</td></tr><tr><td rowspan="5">mini-SWE-agent</td><td>Claude Sonnet 4.6</td><td></td><td>12.1† 22.3</td><td>22.5</td><td>100</td><td>80</td><td></td><td>43 0.203</td><td>+0.030</td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td>7.3† 20.8</td><td>13.2</td><td>80</td><td>60</td><td>25</td><td>0.214</td><td>+0.021</td></tr><tr><td>Gemini 3.6 Flash</td><td></td><td>8.2† 18.6</td><td>18.4</td><td>88</td><td>66</td><td>31</td><td>0.199</td><td>+0.021†</td></tr><tr><td>Kimi K3</td><td></td><td>14.0† 25.9</td><td>31.6</td><td>100</td><td>85</td><td></td><td>440.206</td><td>+0.045†</td></tr><tr><td>gpt-oss-120b</td><td></td><td>2.9† 4.8</td><td>15.8</td><td>78</td><td>47</td><td></td><td>150.181</td><td>+0.003</td></tr><tr><td rowspan="2">Claude Code</td><td>Claude Opus 4.6</td><td></td><td>12.0† 27.5</td><td>19.3</td><td>100</td><td>85</td><td></td><td>43 0.228</td><td> $+ 0 . 0 3 9 ^ { \dagger }$ </td></tr><tr><td>Claude Sonnet 4.6</td><td></td><td>7.8† 21.5</td><td>20.4</td><td>99</td><td>92</td><td>47</td><td>0.204</td><td> $+ 0 . 0 2 8 ^ { \dagger }$ </td></tr><tr><td rowspan="2">Gemini CLI</td><td>Gemini 3.1 Pro</td><td></td><td>3.8† 15.5</td><td>5.1</td><td>100</td><td>40</td><td></td><td>11 0.208</td><td> $+ 0 . 0 1 5$ </td></tr><tr><td>Gemini 3.6 Flash</td><td></td><td>9.3† 26.3</td><td>14.4</td><td>100</td><td>76</td><td>36</td><td>0.203</td><td>+0.006</td></tr><tr><td> $\overline { { C _ { 0 } } }$  (initial)</td><td></td><td>0.7</td><td>12.5</td><td>0.8</td><td>一</td><td>66</td><td>12</td><td>0.156</td><td></td></tr><tr><td> $C _ { \varnothing }$  (empty)</td><td></td><td>0.8</td><td>0.5</td><td>100.0</td><td>一</td><td>98</td><td></td><td>80.091</td><td>-0.086†</td></tr></table>

Table 20: Full results of Table 1, with completion and utility, on 88 instances with $B = 1 0 .$ . Done: runs ending with the agent’s own submission. Deliv./Acc.: instances where the fixed planner delivers a plan / an acceptable plan; $\bar { U } ;$ : mean utility of delivered plans; $\Delta U \colon$ paired difference from $C _ { 0 }$ on instances where both deliver. <sup>†</sup>: the 95% bootstrap CI of the difference from $C _ { 0 }$ excludes zero. Bold: best system per column.

<table><tr><td></td><td colspan="2">Within-instance AUC [95% CI]</td></tr><tr><td>Measure</td><td>Gemini 3.6 Flash</td><td>GPT-6 Luna</td></tr><tr><td> $P _ { w }$ </td><td>.85 [.81, .88]</td><td>.83 [.80, .86]</td></tr><tr><td> $R _ { w }$ </td><td>.71 [.66, .75]</td><td>.71 [.67, .75]</td></tr><tr><td> $\mathrm { F } _ { 1 , w }$ </td><td>.87 [.84, .90]</td><td>.85 [.82, .88]</td></tr><tr><td> $\mathrm { F } _ { 0 . 5 , w }$ </td><td>.88 [.85, .91]</td><td>.85 [.82, .88]</td></tr><tr><td> $P _ { w } \log { R _ { w } }$ </td><td>.89 [.86, .91]</td><td>.85 [.82, .88]</td></tr></table>

Table 21: How well each measure orders specifications by the planner’s acceptance. AUC: the share of same-instance pairs of specifications, one accepted and one not, that the measure orders correctly. Pairs come from the final specifications of the 14 systems and 3 baselines on 88 instances, and 95% CIs from 1,000 bootstrap resamples of instances. $P _ { w }$ log $R _ { w }$ is $P _ { w } ( \log ( R _ { w } + \epsilon ) - \log \epsilon )$ with $\epsilon = 1 0 ^ { - 4 }$

## I.1 Measures of Search Space Quality

Table 21 compares the measures by their withininstance AUC under both planners.

Within-instance AUC asks, for every pair of specifications on the same instance of which the planner accepts exactly one, whether a measure scores the accepted one higher, so that easy and hard instances do not mix. Over 1,000 bootstrap resamples of instances, $\mathrm { F } _ { 1 , w }$ orders such pairs reliably better than precision alone and than recall alone, while the two variants that weigh precision more, $\mathrm { F _ { 0 . 5 , w } }$ and $P _ { w }$ log $R _ { w }$ , do not reliably beat it.

Table 24 shows how acceptance depends on both measures at once: it is lowest when both are low and highest when both are high, and no row or column alone predicts it.

<table><tr><td></td><td> $\mathrm { F } _ { 1 , w }$ </td><td> $P _ { w }$ </td><td> $R _ { w }$ </td></tr><tr><td>Trade-off Probing finds a trade-off (28 instances)</td><td></td><td></td><td></td></tr><tr><td>Trade-off Probing</td><td></td><td>20.1 20.4</td><td>42.6</td></tr><tr><td>Exception Probing</td><td></td><td>18.2 15.1</td><td>57.3</td></tr><tr><td> $C _ { \mathrm { r e f } }$ </td><td></td><td>23.7 29.3</td><td>55.3</td></tr><tr><td>Trade-off Probing finds none (60 instances)</td><td></td><td></td><td></td></tr><tr><td>Trade-off Probing</td><td>8.6</td><td>11.9</td><td>22.1</td></tr><tr><td>Exception Probing</td><td>13.5</td><td>13.9</td><td>37.3</td></tr><tr><td> $C _ { \mathrm { r e f } }$ </td><td>20.4</td><td>25.2</td><td>53.1</td></tr></table>

Table 22: Protocols split by trade-off outcome: Gemini 3.6 Flash on the instances where Trade-off Probing ends with a meaningful trade-off and on the rest. Bold: best protocol per column and block.

## I.2 Planning Utility

Table 20 adds to Table 1 the share of runs that end with the agent’s own submission (Done), the mean utility $\bar { U }$ of delivered plans, and the utility gain $\Delta U$ over $C _ { 0 }$ , the paired difference in utility on instances where both deliver. Table 22 splits the protocol results by whether Trade-off Probing ends with a meaningful trade-off.

## I.3 Planner Robustness

To check that the planning outcome does not hinge on one planner, we give the specifications of the three baselines and the fourteen systems also to

<table><tr><td></td><td colspan="3">Gemini 3.6 Flash</td><td colspan="3">GPT-6 Luna</td></tr><tr><td>Specification</td><td>Deliv. Acc.</td><td></td><td>ū</td><td>Deliv. Acc.</td><td></td><td> $\bar { U }$ </td></tr><tr><td> $C _ { \varnothing }$  (empty)</td><td>98</td><td></td><td>80.091</td><td>98</td><td></td><td>11 0.055</td></tr><tr><td> $C _ { 0 }$  (initial)</td><td>66</td><td></td><td>12 0.156</td><td>75</td><td></td><td>14 0.155</td></tr><tr><td>Systems (mean)</td><td>71</td><td></td><td>31 0.203</td><td>67</td><td></td><td>300.200</td></tr><tr><td> $C _ { \mathrm { r e f } }$  (reference)</td><td>94</td><td></td><td>550.217</td><td>89</td><td></td><td>470.226</td></tr></table>

Table 23: The planning outcome with two planners on 88 instances. Deliv. and Acc. are in %.
<table><tr><td colspan="3"> $R _ { w }$ </td></tr><tr><td> $P _ { w }$ </td><td>0-10 10-30</td><td>30-60</td><td>60-100</td></tr><tr><td>0-5</td><td>5</td><td>46</td><td>22 28</td></tr><tr><td>5-15</td><td>4</td><td>39</td><td>54 67</td></tr><tr><td>15-30</td><td>64</td><td>48</td><td>43 66</td></tr><tr><td>30-100</td><td>58</td><td>81</td><td>77 92</td></tr></table>

Table 24: Acceptance by precision and recall. Acceptance rate (%) of the planner’s delivered plans by the $P _ { w }$ and $R _ { w }$ of the specification, over all 5,368 specifications. Cells hold between 35 and 2,920 specifications.

GPT-6 Luna (Appendix E.2). Table 23 compares the two planners. The orderings the paper relies on hold with both: the empty specification falls below $C _ { 0 } .$ , the systems lie between $C _ { 0 }$ and $C _ { \mathrm { r e f } }$ , in both acceptance and the utility of delivered plans, and within instances $\mathrm { F } _ { 1 , w }$ orders the specification each planner succeeds with above one it fails with about as often under both (Table 21). Table 2 reports both for ReSHAPE Shopping.

## J Prompts

This appendix gives the instructions that the agents, the simulated user, and the planner receive (Prompts 1–8). Where an instruction lists rules at length, we keep the rules that define the behavior and replace the rest with a bracketed note of what it says; the complete texts accompany the code. Every protocol prompt ends with the schema and semantics of the specification language (Appendix C.2) and with the meaning of the five response states (Table 11); both are omitted.

## J.1 Off-the-Shelf Agents

Each off-the-shelf agent receives the task instruction of Prompt 1 and reaches the environment through the command-line tool it describes.

## J.2 Interaction Protocols

Each protocol runs the same tool loop with its own system prompt: Ask-before-Plan (Prompt 2),

Self-Directed Revision (Prompt 3), Exception Probing (Prompt 4), and Trade-off Probing (Prompt 5). Gemini 3.6 Flash and Claude Sonnet 4.6 receive the same text. Self-Directed Revision is called Agent Decides in the code.

## J.3 Simulated User

The hidden profile decides each judgment (Appendix D.4); Claude Haiku 4.5 only words it, with the system prompt of Prompt 6. These prompts call the agent “the planner.” For a shown plan, the request carries the plan’s label, the response state, and the one aspect the reply may name, with the rules of Prompt 7. For a question, the request carries the question and the authorized answer. The rules ask the model to answer only the queried attribute, to express at most the one disclosed proposition, to volunteer no other preference, to give no numbers, thresholds, or repair advice, and to ask nothing back. Every reply is then checked against the record before the agent sees it (Appendix D.4).

## J.4 Planner

The planner (Appendix E.2) receives the specification, the query, and catalog tools, with the instruction of Prompt 8.

## Prompt 1: Instruction to the off-the-shelf agents

Your task is to revise a travel-planning specification on behalf of a user.

The user has made a travel request and stated an initial specification: a set of conditions on attributes of the travel options. A specification determines which legal itineraries are admitted. The user’s preferences are fixed but hidden from you. Your goal is to submit a revised specification that admits the itineraries this user would accept and excludes the ones they would not, across all legal itineraries, including itineraries never discussed with the user.

You may work in any way you choose. You can revise and submit the specification on your own, ask the user about their preferences, or show the user one or two itineraries and ask for feedback, in any order.

## Commands

• Inspect the task: reshape-task inspect, spec-schema, spec-current, itinerary-schema.

• Read options: reshape-task options CATEGORY --output FILE saves the full public catalog for one category.

• Check itineraries: reshape-task check-plans FILE... reports whether each itinerary is legal and whether it satisfies the current specification. Free.

• Ask a question: reshape-task ask ATTRIBUTE preference|importance. Costs 1 feedback unit. You may ask about the same attribute more than once.

• Request feedback: reshape-task feedback FILE [FILE] shows the user one or two legal itineraries, which may lie outside the current specification. Costs 1 feedback unit per itinerary. Each response contains accepted, a response\_state, and the user’s rationale.

• Edit the specification: reshape-task spec add | remove | update-cutoff | replace-value | tradeoff | remove-tradeoff [arguments]. tradeoff replaces a condition with the original condition, OR a relaxed version of it together with a condition on another attribute. Append --check to any edit to validate it without committing. Invalid edits change nothing and cost nothing.

• Submit: reshape-task finish submits the current specification.

## Budget and limits

• You have 10 feedback units in total. Inspecting, checking, and editing are free, and you may keep editing after the feedback units are spent.

• Each run is limited to 30 minutes and 100 model requests. If the run ends early, your latest valid specification is submitted.

## Specification details

[How the budget condition, the transport category, full-party prices, means over the selected options, and unknown values are defined.]

Use only the public task data.

## Prompt 2: Ask-before-Plan

You are a travel-planning assistant using Ask-before-Plan. Revise the user’s initial planning specification so that it better represents the plans the user would accept.

The query contains fixed planning requirements, such as destinations, dates, and party size. These requirements cannot be changed. The initial specification contains the user’s starting editable conditions. You may inspect the query, available options, current specification, supported constraint language, and plan legality before deciding what to ask.

You have at most 10 user-feedback interactions.

## GATHER PREFERENCES

You may ask the user focused qualitative questions without showing any concrete plan.

Each question must concern exactly one eligible preference attribute. You may ask about the same attribute repeatedly, including asking its preferred direction and then its importance. Each answer costs one feedback unit. Ask either: what direction the user prefers for that attribute; or how important that attribute is when choosing a trip.

Do not ask for an exact numerical cutoff. Do not ask the user to provide a complete preference profile. Do not ask a multi-attribute trade-off question. Do not present a concrete plan for judgment.

The public list of attributes is not a list of attributes the user necessarily cares about. The user may have no strong preference about an attribute.

After each answer, make exactly one constraint transaction before asking again. You may revise the current specification or use operations=[] for NO\_OP. [How afailed transaction is held pending until it is corrected.]

## REVISE THE SPECIFICATION

You may stop questioning as soon as you believe you have enough information. You do not need to use all 10 feedback interactions.

After an answer, you may add or remove supported conditions, update numeric cutoffs, replace supported categorical values, or leave the specification unchanged. Set finish=true on that transaction to stop questioning early; otherwise continue. At feedback unit 10, the transaction ends the session automatically. You may also finish before asking any question with a finish=true transaction.

Top-level items must hold together; a supported exception item uses A OR (A\_relaxed AND B). Do not invent unsupported expressions.

## Prompt 3: Self-Directed Revision

You are a travel-planning assistant using Agent Decides. Revise the user’s initial planning specification so that it better represents the plans the user would accept.

[The fixed requirements and editable conditions, as in Ask-before-Plan.]

You have no user-feedback interactions. You cannot ask the user questions and cannot show the user plans. Decide on the user’s behalf.

## REVISE THE SPECIFICATION

You may list options and check complete plans for legality and for whether the current specification admits them. These checks are free and do not involve the user.

When you judge that a different specification would serve the user better, commit a constraint transaction: add or remove supported conditions, update numeric cutoffs, replace supported categorical values, or leave the specification unchanged with operations=[]. You may commit several transactions. Set finish=true on a transaction to submit the specification and end the session. A failed transaction retains the last valid specification; correct it and commit again.

Top-level items must hold together; a supported exception item uses A OR (A\_relaxed AND B). Do not invent unsupported expressions.

## Prompt 4: Exception Probing

You are a travel-planning assistant using Exception Probing. Revise the user’s planning specification by testing concrete exceptions to its boundary.

The query contains fixed planning requirements that cannot be changed. The initial specification contains editable conditions; the current specification is the latest successfully saved set. A complete legal plan satisfies the fixed query requirements. An admitted plan also satisfies the current specification. Membership is separate from user acceptance.

You have 10 feedback units. Each independent single-plan judgment costs one unit.

## SELECT AND EXPLAIN A PROBE

When any excluded legal plans exist, select one that you judge worth recommending from the task, public catalog, and user feedback so far, yet that the current specification excludes. Test whether the user accepts this exception. In the existing target\_hypothesis or accompanying explanation, briefly state why it is worth considering and which current condition or conditions exclude it. Do not choose an arbitrary excluded plan that you yourself find unappealing.

This recommendation judgment is yours, not hidden utility. Do not read the user profile, weights, or scores. You need not prove that the candidate is better than all admitted plans. Do not use a hidden ranking, utility tool, or second comparison plan. No sacrifice-for-benefit hypothesis is required; a single judgment cannot establish a compensating trade-off.

The excluded selection rule also applies when the current specification admits no legal plans. If every legal plan is admitted, instead select one admitted legal plan that you suspect should not be admitted, and state the possible admission error.

Submit exactly one new complete legal plan with a concise boundary-error hypothesis. Never repeat a judged plan. Give an accurate summary of the features that make it worth considering or suspect, including important drawbacks and the conditions responsible for any exclusion. Use accurate values and units; do not invent advantages or argue that the user should accept it.

## USE FEEDBACK AND REVISE

After every judgment, complete exactly one specification transaction, allowing NO\_OP. Treat the independent ACCEPT or REJECT as an observation about that plan, and the qualitative concern as additional evidence; unmentioned attributes may still matter.

Acceptance of an excluded plan reveals an exclusion error for that concrete plan. It does not establish why it was accepted or justify deleting the entire excluding condition. Rejection of an excluded plan is not an admission error, though its concern can provide further information. Rejection of an admitted plan reveals an admission error for that plan.

Use all public feedback so far, not just the latest ACCEPT. [How to weigh allfeedback sofar and report edits that change the status ofjudged plans.]

One acceptable exception can show that a condition is too strict without showing that its attribute should be unrestricted. Before deleting a condition, consider the enlarged region and evidence across different contexts; there is no required number of ACCEPTs.

When the feedback suggests that a condition is too strict only for plans with some other compensating feature, record that with an exception item rather than relaxing the condition for every plan: keep the original condition as A, set A\_relaxed to the weaker cutoff the evidence supports, and set B to the compensating condition. Relax or delete the condition outright only when the evidence suggests it is too strict whatever the other features.

You may retain, relax, tighten, replace, add, or delete supported conditions, including returning TRUE. Use an empty operations array for NO\_OP; a transaction need not change conditions. Use only the supplied compensation-exceptions/v1 constraint language.

Acceptance does not end Exception Probing. Continue from the latest successfully saved specification through 10 judgments, unless a runtime limit or error ends the session. Do not ask direct preference questions or submit plan pairs.

## Prompt 5: Trade-off Probing

You are a travel-planning assistant using Tradeoff Probing. Repair the user’s specification by finding compensating trade-offs that its hard conditions miss: a condition the user would relax for plans that are better in another respect.

The query fixes immutable planning requirements. A complete legal plan satisfies those requirements; an admitted plan also satisfies the current editable specification. Membership and user acceptance are separate. Use only the task, public catalog, current specification, and public feedback. You do not know hidden utility, weights, profiles, scores, or rankings. Do not assume preference directions for user-specific categories such as cuisine, room type, or attraction type.

A compensating trade-off is recorded as an exception item {exception: {A, A\_relaxed, B}}, which admits a plan if A holds, or if both A\_relaxed and B hold. A is the current condition on the sacrificed attribute, A\_relaxed is a weaker cutoff on the same attribute, and B is a condition on a different, compensating attribute.

You have 10 feedback units: two independent judgments per pair, costing 2, at most five pairs. Never repeat a judged plan. Each round follows five steps, recorded in a single propose\_plans call.

[The five response states and what each may disclose.]

1. CHOOSE A CONDITION THAT MAY BE COMPENSATED. Name one current condition (on the sacrificed attribute) that excludes legal plans which violate only that condition yet are clearly better on another attribute (the candidate compensating attribute). Give the catalog evidence: example values of both attributes in such plans. You may keep the same condition across rounds. [What to do when every legal plan is admitted.]

2. WRITE THE EXCEPTION ITEM TO BE TESTED. Write the exact exception item you would commit if the test supports it: A = the current condition, A\_relaxed = a specific weaker cutoff, B = a specific condition on the compensating attribute. Choose cutoffs that the pair in step 4 can test.

3. STATE THE HYPOTHESIS. State in one sentence that the user accepts the sacrificed attribute at the A\_relaxed level when B holds, but not without it. The alternatives are that the relaxation is acceptable without B, or unacceptable even with B.

4. BUILD A CONTRAST PAIR AND MAP OUTCOMES TO EDITS. Submit two distinct complete legal plans that both satisfy A\_relaxed but violate A, so both are currently excluded, with values of the sacrificed attribute as similar as practical. Plan A satisfies B; Plan B does not. Keep everything else as similar as practical, ideally differing in one option. [What to record about the pair.] State, in one sentence per branch, the edit that branch leads to under the rules of step 5. The design record is validated and saved before either judgment.

5. REVISE FROM THE OBSERVED BRANCH. After each judged pair, make exactly one transaction, allowing NO\_OP, and cite the observed branch in its note: ACCEPT\_REJECT (the compensation is needed): commit the exception item in place of condition A, with A\_relaxed no weaker than the sacrificed value the user accepted and B satisfied by Plan A but not by Plan B. ACCEPT\_ACCEPT (no compensation needed at this level): relax condition A to the tested level without a compensating condition. REJECT\_REJECT: NO\_OP; next round, test a milder relaxation or a stronger compensation, or move to another condition. REJECT\_ACCEPT (another factor is at work): NO\_OP; next round, try a different compensating attribute or another condition.

Once an exception item is committed, use later pairs to test its limits (a weaker A\_relaxed with B held, or a weaker B with A\_relaxed held) and update it when supported. After two rounds on the same condition in which Plan A is rejected, move to another condition. Acceptance judges the whole plan: do not relax a cutoff beyond values the user has accepted.

## Prompt 6: Simulated user: system prompt

You are voicing one persistent traveler in an interaction with a travel-planning agent.   
Your job is LANGUAGE REALIZATION ONLY.   
The simulator has already decided what this traveler means. Express only that authorized meaning naturally in the traveler’s interaction style.   
You must not infer, select, strengthen, weaken, or add semantic content.

## INTERACTION STYLE: [the profile’s interaction style]

The interaction style controls only HOW you speak: concise vs. explanatory; direct vs. tentative;   
warm vs. matter-of-fact; conversational phrasing; sentence length and rhythm.

The interaction style is NOT evidence about WHAT the traveler prefers. Never infer a hidden preference, concern, reason, threshold, or factual claim from the interaction style.

The authoritative semantic record in the current request is the COMPLETE semantic content you may express. You may not add another preference or reason simply because it sounds plausible.

Speak as an ordinary traveler, not as an evaluator, benchmark, simulator, or optimization system.

## Prompt 7: Simulated user: wording a plan judgment

Realize the following authoritative plan-feedback record: [label, response state, disclosed proposition or null]

Rules. 1. Preserve the ACCEPT/REJECT verdict exactly. 2. Express at most the one disclosed proposition in the record. 3. If it is null, do not name or imply a specific hidden preference or attribute. 4. If a proposition is supplied, preserve its polarity and qualitative relation exactly. 5. Do not mention a second positive aspect, negative aspect, preference, or concern. 6. Do not infer anything from the plan beyond this record. 7. Never reveal numerical preference thresholds, utility values, weights, reference points, hidden scores, acceptance-region information, or simulator internals. 8. Do not tell the planner what constraint to add, remove, loosen, tighten, or set. 9. Do not ask the planner a question. 10. Prefer one or two short conversational sentences.

Response-state meaning. STRONG\_ACCEPT: clearly accept; do not introduce a reservation or attribute-specific reason. SOFT\_ACCEPT: accept while naturally expressing the supplied concern, if any. BORDERLINE: the verdict is ACCEPT; convey weak or borderline acceptance, using only the supplied negative proposition, if any. SOFT\_REJECT: reject while naturally acknowledging the supplied positive aspect, if any. STRONG\_REJECT: clearly reject; if a proposition is supplied, it is a firm requirement the plan violates: state it plainly as the reason. Otherwise do not invent an attribute-specific explanation.

Return STRICT JSON only: {"text": "<natural traveler response>"}

## Prompt 8: Planner

You are a travel planning agent. Construct and submit one legal itinerary satisfying ONLY the supplied specification and immutable public task requirements. The supplied specification replaces any earlier specification. Do not invent option IDs. Query the public catalog yourself; no complete plans have been prepared for you. You cannot edit the specification or ask a user questions. No private preferences or scores are available.

Respond with JSON {"actions": [[tool calls]]} with at most 6 actions per response. Tools: catalog(category, filters, fields, offset, limit) queries public rows [filter syntax, pagination, and price units]. check\_plan(itinerary) checks a plan you construct against immutable legality rules and the supplied specification; returns public violations only. Numeric conditions apply to selection MEANS, not individually to every option. submit\_plan(itinerary): at most 5 formal submission attempts per episode, independently of the 50-model-request budget. The first legal plan satisfying the supplied spec is final and immediately ends planning. Five invalid submissions end with cannot\_find\_plan; this does not establish infeasibility. No private score is returned. Check before submitting when possible. stop(reason): end without a submitted plan if you cannot find one; this does NOT establish infeasibility.

Use the entire itinerary JSON schema. Transition days have transport, no meals/attractions, and an arrival-city hotel except the final return day. Stationary days require 3 distinct restaurants and 1 attraction; do not repeat restaurants/attractions across the trip. Respect hotel minimum nights. Build your own candidate and revise based on check\_plan feedback. Prefer efficient catalog queries; reserve requests to check and submit. Never claim submission without calling submit\_plan.