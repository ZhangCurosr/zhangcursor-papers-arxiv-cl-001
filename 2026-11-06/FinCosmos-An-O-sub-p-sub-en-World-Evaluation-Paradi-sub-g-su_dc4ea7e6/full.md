# FinCosmos: An O<sub>p</sub>en-World Evaluation Paradi<sub>g</sub>m for A<sub>g</sub>ents in Finance

Yupeng Cao<sup>1</sup>, Xueqing Peng<sup>1,†</sup>, Lingfei Qian<sup>1,†</sup>, Haohang Li<sup>2</sup>, Yangyang Yu<sup>2</sup>, Anke Xu<sup>3</sup>, Wenbo Cao<sup>3</sup>, Yan Wang<sup>1</sup>, Yuechen Jiang<sup>4</sup>, Yuyan Wang<sup>4</sup>, Zhuohan Xie<sup>5</sup>, Yankai Chen<sup>5</sup>, Zehan Li<sup>6</sup>, Jimin Huang<sup>1</sup>, Xue Liu<sup>5</sup>, Jian-Yun Nie<sup>7</sup>, Sophia Ananiadou<sup>4</sup>

<sup>1</sup>The Fin AI <sup>2</sup>Stevens Institute of Technology <sup>3</sup>Independent Researcher <sup>4</sup>University of Manchester <sup>5</sup>MBZUAI <sup>6</sup>DeepKin <sup>7</sup>University of Montreal

Language agents increasingly work for users on the open Web, and finance is where trusting them is hardest. The data must come from the Web, the strategy lives in the user’s head, and correctness depends on time, so an agent can fail silently: a strategy the user never asked for, unverified data, or a backtest that uses the future yields a report as convincing as a correct one. Existing benchmarks close the world somewhere, fixing the task, the data, or the timeline, and thereby remove these failures instead of measuring them. We introduce FinCosmos, an open-world evaluation paradigm for agents doing financial work. A simulated user holds a hidden strategy sampled from a compositional grammar and reveals it only when asked. The agent receives no data, so it must acquire data from the Web, build its own backtest under a data cutof, and deliver verifiable results. Grading needs no reference answer. Programs verify point-in-time correctness and recompute every reported metric, the delivered strategy is checked against the hidden specification, and a run is accepted only when the user is satisfied and none of 15 veto checks fails. Across four coding agents and a model without tools in 150 sessions, no session is accepted: every session that the simulated user signed of fails a program check, and agents raise only 54–64% of the details they were not told. Human experts attribute 88.33% of failures to the agents. With FinCosmos feedback, an agent with persistent memory learned all ten tasks.

<sup>z</sup> Date: November 6, 2026

<sup></sup> Leaderboard: leaderboard.fincosmos.com.co

The Fin AI

## 1 Introduction

Language agents no longer just answer questions. They work for users in the open world, where the Web is both their source of information and their place of work [8, 38, 69]. Given a request, agents search for sources, call data APIs, write and run code, and hand back results that the user acts on but rarely checks. The question is no longer whether an agent can solve a task, but whether it can be trusted with one. Finance is where this question is hardest to answer and where a wrong answer costs the most. The data lives on the Web, not in the model. Prices, filings, and news are vast, numeric, and change every day, so an agent that relies on memory hallucinates the figures its results rest on, and even with retrieval, models still miscompute them [5]. The task intent lives in the user’s head. Few investors want quite the same strategy, and its details surface only when the agent asks [7, 25, 60]. Correctness lives in the point-in-time. A decision may use only what was public on its date, yet the Web keeps revising itself, so today’s data carries the future into the past. Each of these produces a silentfailure, in which nothing crashes and the report looks as convincing as a correct one (Figure 1).

Existing benchmarks each close the real world somewhere, and each closure removes a silent failure instead of measuring it. An agent given curated data, explicit user intent, and an enforced time cutof never encounters these failures. As a result, none of them verifies what an agent delivers when the data, the user’s intent, and the timeline all stay open (Table 1). Closing the intent removes the strategy the user never asked for. Nearly every benchmark states the task in full at the start, whether it is a question over given filings [6, 19], a strategy to implement [53, 65], a research query [70], a fact to find on the live Web [36, 54], or a shopper profile [23]. Simulated-user benchmarks do require elicitation, but over a closed database in customer-service workflows [2, 60]. Closing the data removes the figure the agent never retrieved, since most financial benchmarks hand the agent curated documents or price series [39, 58]. Closing the timeline removes the backtest that peeks ahead. Trading benchmarks replay history in their own environment or evaluate agents only in live markets [26, 27, 42], so the agent never builds a backtest of its own, and they score realized returns, which are biased [28] and say little about whether the agent built what the user wanted [43]. Only an evaluation that leaves all three open can tell whether an agent is ready to work for a real user.

![](images/85859d0f4ec6f386626e96730e1603f464478392d115877f4ef5104110cb7220.jpg)  
Figure 1: A silent failure. The user is satisfied, yet the delivery gets the intent and the data wrong; only time is right.

We introduce FinCosmos, an open-world evaluation paradigm and the first framework that preserves the uncertainties of real financial work (implicit user intent, independently sourced Web data, and point-in-time constraints) and evaluates whether agents can deliver trustworthy results under these conditions (Figure 2). A simulated user, played in one of three personas, holds a strategy specification and reveals each detail only when the agent asks, so eliciting the request becomes part of what is tested. Because a task needs only a specification and no reference answer, specifications are sampled from a compositional grammar over six strategy components, which yields about 3.8 million distinct strategies before parameter values are chosen, so new tasks can be drawn whenever a released set saturates or leaks. Rather than working with curated data, the agent retrieves the data it needs from the open Web, and a fixed data cutof turns respecting time into a property that can be checked rather than assumed. To distinguish trustworthy deliveries from convincing but incorrect ones, the evaluation verifies the delivery itself rather than matching a reference. Each run is graded on 45 dimensions, combining programmatic verification with checks of strategy fidelity against the hidden specification. Programs check data provenance, point-in-time correctness, and whether reported metrics can be independently recomputed from the delivered trade log. Fifteen dimensions serve as vetoes: a run is accepted only when the user is satisfied and none of these critical veto checks fails, so a silent failure cannot pass.

Table 1: Comparison of FinCosmos with related benchmarks.
<table><tr><td></td><td colspan="3">Open real world</td><td colspan="3">Tasks and evaluation</td></tr><tr><td>Benchmark</td><td>Hidden intent</td><td>Self-src. data</td><td>Point-in- time</td><td>Gen. tasks</td><td>Exec. deliv.</td><td>Ref.- free</td></tr><tr><td>FinBen, MultiFinBen [39, 58]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>X</td><td></td></tr><tr><td>FinDeepResearch [70]</td><td>×</td><td>V</td><td>x</td><td>x</td><td>×</td><td></td></tr><tr><td>AlphaForgeBench [65]</td><td></td><td></td><td>x</td><td>× ×</td><td></td><td>××&gt;</td></tr><tr><td>CLQT [43]</td><td></td><td>××</td><td>x</td><td></td><td></td><td>V</td></tr><tr><td>InvestorBench [27]</td><td></td><td>××</td><td>x</td><td></td><td></td><td></td></tr><tr><td>AMA [42]</td><td></td><td></td><td>x</td><td>× ×</td><td></td><td>× ×</td></tr><tr><td>Herculean [40]</td><td>×××××</td><td>x</td><td>x</td><td>x</td><td>&gt;×××××</td><td>V</td></tr><tr><td>τ-banking [46]</td><td>V</td><td>x</td><td>x</td><td>x</td><td></td><td>x</td></tr><tr><td>FINCosmos (ours)</td><td>V</td><td>V</td><td>V</td><td>✓</td><td>V</td><td></td></tr></table>

Open real world: as in practice, the user leaves the intent vague, and the agent gets neither data nor a clean timeline. Hidden intent: the agent must elicit the task from a user. Self-src. data: no data is provided, and the agent finds it on the Web or data APIs. Point-in-time: the agent must keep its own backtest point-in-time, and the benchmark checks it. Gen. tasks: new tasks are produced on demand. Exec. deliv.: the agent delivers code and results rather than an answer or a decision. Ref.-free: grading verifies the delivery itself, without a reference answer, a target state, or realized returns.

We evaluate four general-purpose agents and a model without tools on ten frozen tasks, with three independent sessions per task and system (150 sessions total). FinCosmos is far from saturated: no session is accepted, and pre-veto quality ranges only from 20.2 to 37.2 out of 100. Its grades are diagnostic. They locate where each agent breaks along the workflow: of the 120 agent sessions, 25 deliver the investor’s strategy, 15 of these also keep their data within the cutof, 14 also backtest without look-ahead, and none delivers trades that reproduce its reported returns. Agents raise only 54–64% of the details they were not told and get 66–83% of the rest wrong. Its design is necessary. The investor signed of in 23% of sessions, and the vetoes an LLM judge decides would accept 11%, yet every one of these sessions fails a program check; grading by sign-of or by an LLM judge would also reorder the agents. Human experts attribute 88% of failed runs to the agent and agree with the LLM judge at � = 0.87. Finally, FinCosmos is solvable. With its leak-free feedback, an agent with persistent memory learned all ten tasks, none of which it passed at the first attempt. Our contributions are as follows:

• We introduce FinCosmos, the first open-world evaluation framework for agents doing financial work. A simulated investor holds a hidden, grammar-generated strategy, the agent acquires its own data from the Web and builds its own backtest, and programs verify the delivery without a reference answer. We release a benchmark of 10 frozen tasks, together with the task generator, the evaluator, and a public live leaderboard<sup>1</sup>.

• We show why each part of the design is needed: closed benchmarks remove silent failures instead of measuring them, and grading by user satisfaction or by an LLM judge counts them as successes.

• We evaluate four coding agents and find that none completes the workflow. They break most often at its two ends, pinning down the investor’s strategy and delivering verifiable results, and they can learn the tasks from its feedback.

## 2 Related Work

## 2.1 Financial A<sub>g</sub>ents and Benchmarks

LLM-based agents are increasingly used in finance [18], building on financial LLMs [57]. They range from trading agents with memory, tools, and multi-agent roles [56, 59, 61, 62, 64] to research agents that mine alphas, automate factor research, write reports [22, 31, 32, 50], and audit XBRL filings [14, 52]. Each is evaluated in a setting its designers choose, and look-ahead errors slip in even there [32]. Benchmarks for these agents have moved from answers to workflows, but they still fix the request when the session starts. Question-answering benchmarks test reasoning over given reports and filings, across languages and modalities [6, 19, 39, 57, 58], with results tracked on public leaderboards [33]. Their questions stay fixed even when they come from real users [13]. Research benchmarks open the data but pose fully specified questions, graded by accuracy or against expert ground truth [3, 70]. Strategy benchmarks hand the agent a specification and curated data [53, 65], and AlphaOpsBench [21] starts from coarse hypotheses but ofers no user to ask. Agent environments for enterprise decisions likewise pose fixed goals [15]. Trading benchmarks evaluate decisions in historical or live markets [26, 27, 42], and some enforce point-in-time data inside their own pipelines [40, 43], so the agent never keeps time on its own. FinTrace [4] evaluates tool calls along the trajectory, but on predefined tasks. Each opens at most one of intent, data, and time (Table 1).

## 2.2 O<sub>p</sub>en-World A<sub>g</sub>ent Evaluation

Outside finance, benchmarks open either the intent or the data, rarely both. Simulated users open the intent. �-bench [60] pairs a tool-using agent with an LLM-simulated user, and its successors add users who act on a shared environment [2], full-duplex voice [44], and a banking domain with a knowledge base and discoverable tools [46]. In all of them the user’s intent is hidden, but the world is closed: the agent works over a fixed database and is graded against a target state [25, 41, 49]. In finance, benchmarks evaluate advisors and recommenders over multi-turn dialogue, including against a user’s latent utility [12, 51], but they grade decisions rather than verifiable research deliverables. The fidelity of simulated users is limited: success rates shift with the simulator model [45], stronger assistants make weaker simulators [37], and no LLM faithfully reproduces the decisions of individual investors [16]. FinCosmos therefore uses its simulator as an examiner that discloses a hidden strategy under fixed rules, and validates it with human experts (Section 4.3). Web benchmarks open the environment instead. Agents are evaluated on sandboxed sites [69], on recorded or live websites [8], and on open-Web tasks with checkable answers [36, 38, 54], but the task is stated in full, even when it is personalized [23]. Deep research benchmarks keep the output open and grade it with LLM judges [10, 47], which have known biases and can be gamed [35, 66, 67], and flaws in tasks and grading can shift agent scores substantially [71]. Fixed task sets also go stale once models train on them, which LiveBench counters by refreshing its questions [55] and GDPevo by regenerating tasks [68]. FinCosmos keeps both the user and the Web open, grades by program-verified invariants wherever possible, and samples new tasks from a grammar rather than a fixed pool.

## 2.3 Look-Ahead Bias in Financial Evaluation

Backtests have long been known to overstate performance [1], and LLMs add new routes for leakage. Models memorize financial values from before their training cutof [11, 34], LLM trading agents can profit from leaked outcomes [30], and efective cutofs can lie months before the dates vendors report [20]. One line of remedies acts on the model, training it only on data before each date [9, 17] or dampening memorized outcomes at inference [29]. Another targets the backtest: switching one convention at a time measures how much each inflates results [63], LLM auditors flag clean backtests as rigged [48], short evaluation windows overstate LLM strategies [28], and few financial LLM papers discuss these biases at all [24]. These studies audit models or fixed pipelines. FinCosmos instead checks the data and backtest that each agent builds and delivers, with programs rather than LLM auditors, and fails any run that violates point-in-time correctness.

![](images/bd3f93419dd34c7d926fcd5dfb74466222e388373d40162a5ee70aae0da29cd8.jpg)  
Figure 2: Overview of FinCosmos. Left: its three components hide a strategy � from the agent and grade what it delivers. Right: the agent works in the open real world, finds its own data, and must stay before the cutof. Red badge mark where each silent failure arises and the check that catches it.

## 3 FinCosmos

FinCosmos has three components (Figure 2). The Strategy Generator samples a hidden strategy and turns it into a task (Section 3.2). The User Simulator plays the simulated user, who holds the strategy and reveals it only when asked (Section 3.3). The Evaluator grades what the agent delivers and returns diagnostic feedback (Section 3.4). The hidden strategy is sealed: only the User Simulator and the Evaluator can see it, and no feedback to the agent reveals it.

## 3.1 Problem Formulation

Tasks. A task is a triple ${ \boldsymbol { \tau } } = ( p , b , c )$ . The persona � defines who the user is and how the user speaks. The belief � is the hidden strategy that the user wants (Section 3.2). The test conditions ${ \boldsymbol { c } } = ( t _ { \mathrm { c u t } } , \boldsymbol { \mathcal { A } } , T , \Delta )$ fix the data cutof $t _ { \mathrm { c u t } }$ , the channels A through which the agent may reach data (but no data itself), the maximum number of agent turns �, and the time limit Δ per turn. They are frozen with the task, so that scores stay comparable. A deployment may set � and Δ to its own needs; Section 4.1 gives the values we use.

Sessions. A session on � is a sequence $h = \left( u _ { 0 } , a _ { 1 } , u _ { 1 } , \ldots , a _ { n } , u _ { n } \right)$ with $n \leq T$ . The user opens with a vague request $u _ { 0 }$ . In turn �, the agent may use its tools freely for up to Δ and then returns $a _ { t } = \left( m _ { t } , F _ { t } \right)$ : a message $m _ { t }$ and a set of delivered files $F _ { t }$ , such as data tables, code, backtest results, and reports. The user replies with $u _ { t }$ . The session ends when the user is satisfied or a limit is reached, and it records one end reason � (Appendix C). Agents connect through a generic interface with three calls (start, send, and finalize), so any agent design can be evaluated.

Deliverable. Only what the user receives is graded: the deliverable $D = \{ m _ { t } \} _ { t = 1 } ^ { n } \cup \bigcup _ { t = 1 } ^ { n } F _ { t }$ . The execution log � of tool calls and the agent’s workspace are recorded too, but they serve only to detect violations, such as data that postdates the cutof. They can make a check fail and never make it pass. The agent is thus judged as a real user would judge it, by what it hands over.

## 3.2 Strate<sub>gy</sub> Generator

Strategy space. A hidden strategy fixes one choice for each of six components of a quantitative strategy: stock selection, market timing, portfolio construction, risk control, execution, and transaction cost (Table 2).

Table 2: Options for the six strategy components.
<table><tr><td>Component</td><td>Options</td></tr><tr><td>1. Stock selection</td><td>Factors in four families. Fundamental: value, quality, growth. Technical: momentum, short-term reversal, low volatility. Information: sentiment, analyst revisions, insider buying Biotech event: catalyst proximity, FDA outcome, pipeline momentum. Combination: equal</td></tr><tr><td>2. Market timing</td><td>weight, IC-weighted, double sort, gradient boosting Buy-and-hold, moving-average crossover, time-series momentum, mean reversion, event- driven, sentiment timing. Exposure: all-in or all-out, continuous.</td></tr><tr><td>3. Portfolio construction</td><td>Equal weight, market-cap weight, risk parity, mean-variance. Covariance: sample, shrinkage, factor-based.</td></tr><tr><td>4. Risk control</td><td>None, stop-loss with position caps, drawdown gate, volatility targeting, VaR/CVaR limit, tail hedge.</td></tr><tr><td>5. Execution</td><td>Single fill, TWAP, VWAP, POV, implementation shortfall.</td></tr><tr><td>6. Transaction cost</td><td>None, fixed basis points.</td></tr></table>

Formally, $b = ( ( o _ { k } , \theta _ { k } ) ) _ { k = 1 } ^ { 6 } .$ , where $o _ { k } \in \mathcal { O } _ { k }$ is an option of component � and $\theta _ { k }$ holds its parameter values. Stock selection uses one to three factors from diferent families and, with several factors, a combination method. Coherence rules C exclude combinations that do not make sense; for example, risk parity and mean-variance weighting require a covariance estimator. The strategy space is

$$
\mathcal { B } = \{ ( o _ { 1 } , \dots , o _ { 6 } ) \in O _ { 1 } \times \dots \times O _ { 6 } : C ( o _ { 1 } , \dots , o _ { 6 } ) \} ,\tag{1}
$$

with $| \mathcal { B } | \approx 3 . 8 \times 1 0 ^ { 6 }$ before parameter values are drawn.

Dificulty. Each option � has a dificulty $\delta ( o ) \in \{ 0 , \ldots , 3 \}$ , and a strategy’s dificulty is $\begin{array} { r } { d ( b ) = \sum _ { k } \delta ( o _ { k } ) } \end{array}$

From strategy to task. The generator draws an option for each component and resamples whenever the combination violates C. Sampling is seeded, so every strategy can be reproduced and new ones can be drawn on demand. Each strategy is paired with a persona and test conditions to form a task, and the released benchmark freezes the first ten sampled tasks (seeds 0–9) for the leaderboard.

## 3.3 User Simulator

The User Simulator is a language model that plays the real user. It separates who the user is (the persona) from what the user wants (the belief), and a disclosure policy fixes how the belief is revealed.

Persona and belief. The persona pool has three personas: a retail investor, a generalist analyst, and a biotech specialist (Appendix D). Their prompts forbid evaluation vocabulary, such as “point-in-time” and “cutof”. The user states the date up to which data may be used in plain words, but never says that it will be checked. The belief � enters the prompt as plain-language convictions with exact numbers, such as “cut to cash if the drawdown reaches 20%”, which the user holds and does not read out.

Disclosure policy. Let � be the set of items that the belief voices, one for each option and parameter value it states, and let $H _ { k } \subseteq H$ be the items of component �. Parameter values that the belief does not state, such as the window of a volatility estimate, are not in �, and the agent is never held to them. Let $O \subseteq H$ be the items in the opening request, and $S _ { t } \subseteq H$ the items the user has stated by turn �, with $S _ { 0 } = O$ . The opening request is a single vague idea, so $| O | \ll | H |$ . After that, the user states only what the agent raises:

$$
S _ { t } \setminus S _ { t - 1 } \subseteq Q ( a _ { t } ) ,\tag{2}
$$

where $Q ( a _ { t } ) \subseteq H$ is the set of items that the agent asks about or proposes in turn �. Asked for a specific value, the user gives it directly. When a proposal conflicts with �, the user objects and says what it wants. When a deliverable deviates from �, the user names the wrong part, such as the position sizing, without restating the strategy. The agent therefore learns the strategy only by asking the right questions. We write $S = S _ { n }$ for the items stated by the end of the session. The simulator is prompted to follow this policy, and the Evaluator measures � from the user’s turns, with a phrase matcher and an LLM check for undecided items, rather than assuming it.

Turn decisions and termination. In each turn, the simulator reads the conversation and the latest deliverables and returns a structured decision: an action, an observation for every goal that cites the supporting messages or files, and its next message. A program validates the decision before the message is sent (Appendix D). The simulator may close the session only when every core goal is met with cited evidence: a backtest of this strategy on real data, whose return, Sharpe ratio, maximum drawdown, and number of trades trace back to trade dates and prices, and a report that covers the summary, method, results, and limitations. User satisfaction is thus a claim that can be checked against ℎ and �.

## 3.4 Evaluator

The Evaluator measures how well the agent served the user and tells it what to fix. Programs recompute numbers, an LLM judge grades only semantics, and every score carries its source and evidence. It grades a package built from � (the full conversation, the last two reports, and the tables and code blocks in the delivered files) against �, �, and facts computed by programs.

Dimensions. Each session is graded on 45 dimensions in seven categories (full rubric in Appendix B). Five categories � are scored: data (6 dimensions, 1 veto), strategy (12, 7), backtest (8, 4), assessment (5, 1), and user experience (6, 2). Their 37 dimensions each receive a score $s _ { j } \in \{ 0 , \ldots , 4 \}$ , with an anchored description for every level. Cost (3 dimensions) and process transparency (5) are reported but not scored. Program judges grade 17 dimensions, an LLM judge 19, and both together 6, while the 3 cost dimensions are measured directly. Where an LLM grade has a program counterpart, the program bounds it, $s _ { j } = \operatorname* { m i n } ( s _ { j } ^ { \mathrm { L L M } } , s _ { j } ^ { \mathrm { P } } )$ . So professionalism cannot exceed the share of metrics that state their basis.

Scoring and acceptance. Let $J _ { k }$ be the dimensions of category � that apply to the strategy and whose judge ran. A dimension for which the agent delivered nothing checkable stays in $J _ { k }$ and scores 0. The pre-veto quality of a session is

$$
Q = \frac { 2 5 } { | K | } \sum _ { k \in K } \frac { 1 } { | J _ { k } | } \sum _ { j \in J _ { k } } s _ { j } \ \in [ 0 , 1 0 0 ] ,\tag{3}
$$

which gives every category equal weight. Fifteen dimensions � are vetoes, each with a pass threshold $\theta _ { \nu }$ , such as point-in-time correctness, the fidelity of each strategy component, and the completeness of the backtest output (Appendix B). With pa $\begin{array} { r } { \mathrm { \ s s } = \prod _ { \nu \in V } \mathbf { 1 } \big [ s _ { \nu } \ge \theta _ { \nu } \big ] } \end{array}$ , the score and the acceptance of a session are

$$
\mathrm { S c o r e } = Q \cdot \mathrm { p a s s } ,\tag{4}
$$

$$
\mathrm { A C } = \mathrm { p a s s } \cdot \mathbf { 1 } [ e = \mathsf { s a t i s f i e d } ] \cdot \mathbf { 1 } [ \mathrm { g r a d i n g ~ c o m p l e t e } ] ,\tag{5}
$$

where grading is complete when every judge returned a verdict and no strategy item or audit check was left unverified. Sessions that hit the turn limit or a resource limit have no result and score 0. A polished report therefore cannot hide a leaky or irreproducible backtest, while � keeps the partial credit that a veto removes.

Key checks. Four groups of checks target the three silent failures, and the last three carry most of the vetoes. Data provenance. The agent receives no data, so every series it delivers must be traceable to a source, such as a URL or an API call with its query parameters, and untraceable data earns no credit. Programs also measure missing values, gaps, duplicates, stale prices, and outliers in the delivered data (Appendix B). Point-in-time correctness. FinCosmos does not hide future data; avoiding look-ahead is the agent’s responsibility. Every dated row in the delivered data is dated no later than $t _ { \mathrm { c u t } }$ , and a single late row fails the veto. Every fill comes on a later date than its signal, and fundamentals used at entry were announced no later than the entry date. Independent recomputation. The program does not trust the reported equity curve. It rebuilds the returns from the delivered trade log, or from the weight log and prices, and compares every reported metric with its own value, within fixed tolerances such as 0.02 for returns and drawdowns and 0.5 for the Sharpe ratio. A missing trade log or price series fails the veto, and mismatches lower the score. Strategyfidelity and elicitation. For each component �, the Evaluator checks every item of $H _ { k }$ against �. Each claim must be backed by a quote found in �, and a later retraction cancels an earlier claim. Let $M _ { k } \subseteq H _ { k }$ be the items that are missing. The fidelity score is capped by whether the user stated them:

$$
s _ { k } \leq \left\{ { 1 , \quad M _ { k } \cap S \not = \emptyset } , \right.\tag{6}
$$

With $\theta _ { k } = 2$ , missing a stated requirement fails the session. An agent that never asks cannot earn full credit, but it is not failed for requirements that never came up.

Feedback. After a session, the agent receives the score, source, and evidence of every dimension, such as the quotes matched, the files read, and the values recomputed. The feedback is leak-free: for the data and strategy categories, whose evidence would reveal �, the agent receives scores without reasons, except for point-in-time strictness, which depends only on $t _ { \mathrm { c u t } } .$ . Program judges also score every turn, which yields a score trajectory per session, and the agent may attempt a task again. Every score is stored with the evaluator version, a hash of the scoring code, so results remain comparable as the Evaluator is refined.

## 4 Evaluation Setu<sub>p</sub>

We use FinCosmos to evaluate four general-purpose agents and to test whether its user simulator and evaluator can be trusted. This section describes the agents and the environment (Section 4.1), the protocol and metrics (Section 4.2), and the human evaluation study (Section 4.3). Section 5 reports the results.

## 4.1 A<sub>g</sub>ents and Environment

Agents. We evaluate four commonly used agents, each with a recent model from its vendor: Claude Code with Claude Opus 5.5, Codex with GPT-6.1 Sol, Gemini CLI with Gemini 3.8 Flash, and Kimi Code with Kimi K3 (open-weight model). Each agent brings its own harness, tools, and context management. As a baseline, Claude Opus 5.5 answers the same requests without tools. Memory features are of, so nothing carries over between sessions.

Open-World Environment. Each session runs in a sandbox with a consistent runtime environment. Every agent receives the same environment and prompt settings. We provide the same prompt instructions, allowing agents to search for and retrieve web pages and run code in their working directory. We equipped the agents with the “Financial Modeling Prep” MCP server<sup>2</sup>, which contains 247 commonly used financial APIs to cover the data resources required for financial work. Search and web scraping tools are provided by the respective vendors; each session consists of up to six rounds, with each round lasting 300 seconds. All ten tasks involve U.S. stock trading, with portfolios ranging from a small basket of stocks to a specific sector or the entire market.

Simulator and evaluator. The user simulator runs on Claude Opus 4.8 with low reasoning efort. It is told that the assistant has web and shell access, also for the bare model, which has neither. The LLM judge runs on Claude Opus 5.5 through the Claude CLI with its default settings. We limit the judge model bias risk in two ways. Numbers and point-in-time properties are recomputed by programs rather than judged (Section 3.4), and Section 4.3 compares the judge and the simulator with human experts.

Table 3: Main results on the ten frozen tasks in the realistic condition, with three sessions per task (30 per agent). Bold marks the best of the four agents in each column; arrows give the better direction. Quality is the pre-veto quality with a 95% bootstrap interval over tasks, and category scores use the same 0–100 scale. Cut: agent turns stopped at the 300-second limit. Cost is API-equivalent; Codex and Kimi Code ran on subscription plans. The bare model is Claude Opus 5.5 without tools.
<table><tr><td rowspan="3">Agent</td><td colspan="3">Outcome</td><td colspan="6">Quality (0–100)</td><td colspan="3">Process</td><td colspan="2">Cost per session</td></tr><tr><td>Sat. (%) ↑</td><td>AC (%) ↑</td><td>pass^2 ↑</td><td>Quality [95% CI] ↑</td><td>↑</td><td>↑</td><td>Data Strategy Backtest Assess. ↑</td><td>↑</td><td>UX ↑</td><td>Vetoes (#) ↓</td><td>Cov. (%)↑</td><td>Cut (%)↓</td><td>Tokens (M) ↓</td><td>USD ↓</td></tr><tr><td>Claude Code</td><td>30</td><td>0</td><td>0</td><td>37.1 [27.8, 45.0]</td><td>44.7</td><td>38.2</td><td>26.9</td><td>23.2</td><td>52.2</td><td>7.4</td><td>54</td><td>65</td><td>1.13</td><td>2.24</td></tr><tr><td>Codex</td><td>33</td><td>0</td><td>0</td><td>37.2 [30.6, 43.3]</td><td>45.8</td><td>33.4</td><td>31.9</td><td>17.2</td><td>57.6</td><td>7.2</td><td>64</td><td>69</td><td>3.20</td><td>0.80</td></tr><tr><td>Gemini CLI</td><td>33</td><td>0</td><td>0</td><td>22.2 [16.9, 27.4]</td><td>25.3</td><td>30.8</td><td>15.6</td><td>7.0</td><td>32.4</td><td>9.3</td><td>57</td><td>75</td><td>3.51</td><td>0.80</td></tr><tr><td>Kimi Code</td><td>20</td><td>0</td><td>0</td><td>22.3 [14.3, 31.6]</td><td>26.8</td><td>23.4</td><td>16.5</td><td>6.5</td><td>38.2</td><td>10.1</td><td>57</td><td>74</td><td>0.89</td><td>0.83</td></tr><tr><td>Bare model</td><td>0</td><td>0</td><td>0</td><td>20.2 [18.8, 21.5]</td><td>0.0</td><td>49.8</td><td>4.6</td><td>1.8</td><td>44.6</td><td>9.1</td><td>41</td><td>0</td><td>0.01</td><td>0.24</td></tr></table>

## 4.2 Evaluation Protocol and Metrics

Tasks and runs. We use the ten tasks of the released benchmark. Each agent attempts every task three times in independent sessions.

Metrics. Our primary metric is the acceptance rate, the share of sessions with AC = 1 (Equation 5). To measure reliability, we also report pass^� [60], the probability that an agent is accepted in all of � independent sessions on a task. With $N = 3$ sessions per task, of which $N _ { \mathrm { A C } }$ are accepted, we estimate it as the mean of $\binom { N _ { \mathrm { A C } } } { k } / \binom { N } { k }$ over tasks, for $k \leq 3$ . For every session, we also report the pre-veto quality � (Equation 3) and the five category scores, which keep the partial credit that a veto removes. To show how agents fail, we report the share of sessions in which each veto fails and the distribution of end reasons (Appendix C). The silentfailure rate is the share of sessions that end with � = satisfied yet fail at least one veto, which we also break down by veto. As a counterfactual, we recompute acceptance from the LLM-judged dimensions alone, to measure how many of these sessions a judge would accept on its own.

Two metrics describe how an agent handles the user. They use the item sets �, �, and � of Section 3.3, with � recorded by the Evaluator. Because the user reveals details only when asked (Equation 2), the items in � \ � were elicited by the agent. The clarification coverage is $\vert S \setminus O \vert / \vert H \setminus O \vert$ . The assumption rate is the share of unstated items, � \ �, that the delivered strategy does not match. Because program judges score every turn, we also track how the score and the number of failed vetoes change over the turns of a session. Finally, we record the cost of each session: wall-clock time, tokens, tool calls, and turns. We test diferences between agents with paired permutation tests over tasks, with Holm–Bonferroni correction.

Learning from feedback. To test whether FinCosmos feedback supports self-improvement, we set up a learnable agent in a “Claude + Memory” configuration: Claude Code with a memory that persists across sessions. After each graded session, a review step reads the leak-free report (Section 3.4) and the per-turn program scores, and edits the memory before the next session. Through these edits, the agent learns from the evaluator’s diagnosis of its failures, acquires new skills, and reuses past experience. It practices each of the ten tasks until its first acceptance. We count the sessions and the memory-editing learning steps it needs, compare the failed vetoes early and late in practice, and grade every accepted session under the current rubric.

## 4.3 Human Evaluation Stud<sub>y</sub>

Two financial experts annotate the sessions of the main experiment to check whether the simulator, the LLM judge, and the evaluator can be trusted. Neither expert took part in building FinCosmos. They complete three tasks: (1) Simulator fidelity. The experts label every user message as error-free, task-benign, or task-critical against the hidden strategy. A task-benign error is a slip that does not afect whether a correct agent is accepted. A task-critical error can prevent a correct agent from being accepted, for example contradicting the hidden strategy, revealing details that were not asked for, or ending the session against the evidence. (2) Judge agreement. On a calibration set of 30 of these sessions, the experts grade the four LLM-judged vetoes on the judge’s anchored 0–4 scale, and we compare their grades with the judge’s. (3) Failure attribution. For every failed veto, the experts mark whether its reason holds, and for every session, whether the first critical error came from the agent, the simulator, or the evaluator.

Both experts label every item independently and resolve disagreements by discussion. We report Cohen’s � on the calibration grades. Appendix E summarizes the annotation guidelines.

## 5 Results and Discussions

In this section, 1) We report the main results (Section 5.1), 2) show that the sessions users accept still fail and that standard grading would count them as successes (Section 5.2), 3) locate where along the workflow agents break (Section 5.3), 4) test whether the benchmark can be trusted (Section 5.4), 5) and show that agent can learn from leaderboard’s feedback to solve all tasks (Section 5.5).

## 5.1 Main Results

Table 3 reports the main results. No agent is accepted in any of the 150 sessions, so pass^� is 0 for every agent and every �. The simulated user was satisfied in 20–33% of the sessions of the four agents, and every one of these sessions fails at least one veto (Section 5.2). Most sessions end the other way: in 77 of the 120 agent sessions, the user walks away without being satisfied, and 8 sessions reach the turn limit.

Pre-veto quality separates the agents, although ten tasks leave wide intervals. Codex (37.2) and Claude Code (37.1) are indistinguishable $( p = 0 . 9 9 $ , paired permutation test over tasks). Codex scores above Gemini CLI (22.2) and the bare model (20.2), and both diferences remain significant after Holm correction (adjusted $p = 0 . 0 2 )$ ; Claude Code’s gap to the bare model does not (adjusted $p = 0 . 1 0 )$ . Gemini CLI and Kimi Code (22.3) do not difer from the bare model, which has no tools at all $( p \ge 0 . 4 4 )$ . The bare model earns its quality from the strategy (49.8) and user-experience (44.6) categories, which reward a well-described plan, while it scores 0 on data and 4.6 on the backtest. A fluent conversation is not a correct deliverable.

Quality falls with task dificulty (Spearman $\rho = - 0 . 2 4 , p = 0 . 0 0 7$ , over the 120 agent sessions), from 41.6 on tasks of dificulty up to 10 to 26.6 on tasks of dificulty 15 and above. Time is a binding constraint: 70% of agent turns run into the 300-second limit. Cost varies more than quality. Codex reads 3.2 million input tokens per session, nearly three times as many as Claude Code for the same quality, while Claude Code is the most expensive at API prices (\$2.24 per session, against about \$0.80 for the other agents).

## 5.2 Silent Failures

The simulated user was satisfied in 35 of the 150 sessions (23%), and all 35 fail at least one veto, each caught by a program check. Most fail for substantive reasons: 29 delivered no trade log or no prices, so their returns cannot be rebuilt (3.2); 15 used data after the cutof or filled trades before their signal (1.1, 3.1); and 15 delivered a strategy other than the user’s (2.1–2.7). 33 of the 35 fail at least one of these three. The two strict completeness checks, process completeness (3.8) and the verifier audit (3.7), fail in nearly every session (120 and 119 of the 120 agent sessions) and do not separate the agents. Without them, 2 of the 150 sessions would be accepted, both by Codex: their returns can be partly rebuilt but do not match the report, and they fail no other veto. Appendix A shows the score of every dimension in every session.

Table 4: Sessions counted as successes under three graders, of 30 per agent. LLM-judged vetoes: 2.7, 4.1, 5.1, and 5.6. All vetoes: the acceptance rule of FinCosmos.
<table><tr><td>Agent</td><td>Signed off</td><td>+ LLM-judged vetoes</td><td>All vetoes</td></tr><tr><td>Claude Code</td><td>9</td><td>7</td><td>0</td></tr><tr><td>Codex</td><td>10</td><td>8</td><td>0</td></tr><tr><td>Gemini CLI</td><td>10</td><td>0</td><td>0</td></tr><tr><td>Kimi Code</td><td>6</td><td>2</td><td>0</td></tr><tr><td>All 150 sessions</td><td>35 (23%)</td><td>17 (11%)</td><td>0</td></tr></table>

Table 5: Sessions passing each step of the workflow, of 30 per agent. (1) The delivered strategy is the user’s (2.1–2.7). (2) Data stay within the cutof (1.1). (3) Fills follow their signals (3.1). (4) The delivered trades reproduce the reported returns $( 3 . 2 \ge 2 )$ . Agents: the four agents together, of 120. In sequence: agent sessions that pass this step and every earlier one.
<table><tr><td>Step</td><td>Claude Codex Gemini Kimi Bare Agents In seq.</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(1) User&#x27;s strategy</td><td>10</td><td>9</td><td>2</td><td>4</td><td>9</td><td>25</td><td>25</td></tr><tr><td>(2) Data in cutoff</td><td>11</td><td>24</td><td>11</td><td>9</td><td>0</td><td>55</td><td>15</td></tr><tr><td>(3) No look-ahead</td><td>21</td><td>23</td><td>9</td><td>10</td><td>0</td><td>63</td><td>14</td></tr><tr><td>(4) Reproducible</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

These sessions look convincing. Their pre-veto quality is 50.8, against 21.0 for the other sessions of the four agents $( p < 0 . 0 0 1$ , permutation test), and their user-experience score is 69.6, against 35.0. Had grading used only the vetoes that the LLM judge decides (2.7, 4.1, 5.1, 5.6), 17 of the 35 sessions (49%) would have been accepted, so the acceptance rate would have been 11% instead of 0. In 10 of these 17, every failed veto is recomputation, the audit, or the process check, all of which recompute results from the delivered files rather than read the report. The choice of grader also reorders the agents (Table 4). By user sign-of, Gemini CLI ties for first with 10 of 30 sessions, but none of them passes the LLM-judged vetoes (Fisher’s exact test, $p < 0 . 0 0 1 )$ , while Codex keeps $8 ~ ( p = 0 . 0 0 5$ against Gemini CLI).

The numbers in these reports also cannot be checked. In 29 of the 35 sessions, the agent delivered no trade log or no prices, so the returns it reports cannot be rebuilt, and in the 5 sessions where the program could rebuild them, none matches the report. Against the agent’s own equity curve, the reported metrics match in only 10 of the 35 sessions; 11 contradict the curve, and 12 report metrics without delivering one.

The Kimi Code session in Figure 1 shows the pattern. The user was satisfied, and the session scored a pre-veto quality of 47.9. Yet the model ranks stocks by 21-day momentum where the user asked for short-term reversal. Its report gives a maximum drawdown of −18.96% and says the metrics were recomputed from the delivered equity curve, but on that curve the drawdown is −21.91%, beyond the 20% stop the user asked for.

User satisfaction and correctness therefore diverge. Each of these 35 users received a deliverable that looks complete but is wrong in a way the report does not show, such as a strategy other than the one the user holds, data from after the cutof, or returns that no one can reproduce. A user who acts on it would commit capital on the strength of performance that does not exist. Benchmarks that grade a final answer, a target database state, or the user’s satisfaction would count these sessions as successes. In finance, where a deliverable is the basis of a trading decision, they are the costliest failures, and only checks on the delivery expose them.

## 5.3 Where A<sub>g</sub>ents Break

Along the workflow. Table 5 follows the sessions through the four steps of the workflow. Agents break most often at its two ends. Only 25 of the 120 agent sessions deliver the user’s strategy, and none delivers trades that reproduce its reported returns: the 5 sessions whose returns can be partly rebuilt, all by Codex, score 1 of 4. On their own, the middle steps pass about half the time. Taken in sequence, 25 sessions pass the first step, 15 the second, 14 the third, and none the fourth. Codex is strongest on data, while Gemini CLI rarely delivers the user’s strategy. The bare model delivers the user’s strategy in 9 sessions, as often as the agents, because it can describe a plan, but it passes no later step.

Elicitation. Agents learn little of the hidden strategy. On average, a task’s belief voices 6.7 items, and the opening request states 2.6–3.7 of them. By the end of a session, the user has stated 4.5–5.4. Agents therefore raise only 54–64% of the details they were not told, and the delivered strategy does not match 66–83% of the details that were never stated. More questions help: clarification coverage correlates with pre-veto quality (Spearman $\rho = 0 . 2 5$ , $p = 0 . 0 0 7 )$ . The fidelity vetoes fail in 38–52% of sessions for the components with real choices, most often stock selection (52%), and the ambiguity veto (2.7), which requires one executable definition for every rule, fails in 68%. Transaction cost, whose options are none or a fixed rate, fails in only 9%.

Progress over turns. Per-turn program scores show slow progress: failed program vetoes fall from 5.3 after the first turn to 3.8 after the sixth. Only 58 of the 120 agent sessions reach a fourth turn; of the 62 that end earlier, the user walks away in 51.

## 5.4 Can the Benchmark Be Trusted?

Validity and expert annotation. All 150 sessions are valid. The program validated every simulator decision before it reached the agent (Appendix D). Grading failed for 2 sittings, which were rerun under the same conditions. The 60 annotated sessions comprise 13 for each of the four agents, covering all ten tasks, and 8 for the bare model, stratified by end reason within each agent. The calibration set takes three sessions per task from the four agents and covers the vetoes 2.7, 4.1, 5.1, and 5.6. Agent names are masked, and the experts see the automatic grades only after tasks (1) and (2) of Section 4.3, when they check failed vetoes and attribute failures.

Simulator. Both experts flag a task-critical simulator error in the same 3 of the 60 sampled sessions, most often the user ending the session while the agent was still working. The user walks away in 41 of the sampled sessions, so at most 3 of these exits are premature: the early endings of Section 5.3 mostly reflect agents that stall, not an impatient user. The experts attribute 53 of the 60 failed sessions (88%) to the agent, 3 (5%) to the simulator, and 4 (7%) to the evaluator, so the lack of accepted sessions reflects the agents, not the benchmark.

Judge. On the calibration set, the LLM judge agrees with the experts at � = 0.87, close to the two experts’ agreement (� = 0.93). The judge is thus reliable on what it grades; the silent failures it lets through (Section 5.2) do not show in the conversation or report it reads, and only program checks on the delivery catch them.

## 5.5 Learnin<sub>g</sub> from Feedback

FinCosmos is built for agents that learn from feedback. A task can be attempted again, and each attempt returns a diagnostic, leak-free report together with per-turn program scores. Every score records the evaluator version that produced it, and the leaderboard records every attempt. Together, these provide the infrastructure for recursive self-improvement: an agent attempts a task, receives its grades, updates itself, and attempts again, while the hidden strategy stays sealed.

The Claude + Memory agent uses this loop to learn tasks that it could not pass at first (Table 6). No task is accepted at the first attempt, and the median task fails 8 vetoes in its first session. Through practice, the agent reaches acceptance on all ten tasks, after a median of 34.5 sessions per task (range 12–132, 506 in total), and every accepted session passes all 15 vetoes under the current rubric. Practice reduces failures (Figure 3).

Table 6: Learning from feedback on the ten tasks by dificulty. Sessions: up to the first acceptance. Failed vetoes: in the first session, and averaged over the first and last five sessions before acceptance. Satisfied: share of those sessions the user signed of. Last to clear: the veto that failed last.
<table><tr><td></td><td></td><td colspan="4">Failed vetoes per session</td></tr><tr><td>Diff.</td><td></td><td></td><td>Sess. 1st First 5</td><td>Last 5</td><td>Satisfied Last to clear</td></tr><tr><td>7</td><td>12</td><td>3</td><td>2.8</td><td>4.6</td><td>73% 3.2 Independent recomputation</td></tr><tr><td>10</td><td>132</td><td>14</td><td>9.4</td><td>1.6</td><td>31% 2.7 Strategy ambiguity</td></tr><tr><td>11</td><td>23</td><td>10</td><td>6.4</td><td>1.4</td><td>59% 2.7 Strategy ambiguity</td></tr><tr><td>11</td><td>26</td><td>4</td><td>6.4</td><td>2.2</td><td>28% 2.7 Strategy ambiguity</td></tr><tr><td>12</td><td>113</td><td>9</td><td>9.0</td><td>1.4</td><td>6% 2.7 Strategy ambiguity</td></tr><tr><td>14</td><td>43</td><td>8</td><td>6.4</td><td>1.2</td><td>10% 3.8 Process completeness</td></tr><tr><td>15</td><td>17</td><td>8</td><td>6.2</td><td>6.2</td><td>0% 2.1 Stock selection</td></tr><tr><td>15</td><td>61</td><td>6</td><td>3.6</td><td>3.0</td><td>50% 2.7 Strategy ambiguity</td></tr><tr><td>16</td><td>65</td><td>10</td><td>5.8</td><td>3.0</td><td>47% 2.7 Strategy ambiguity</td></tr><tr><td>17</td><td>14</td><td>3</td><td>2.4</td><td>4.8</td><td>38% 5.1 Conversation experience</td></tr><tr><td>Median 34.5</td><td></td><td>8</td><td>6.3</td><td>2.6</td><td>34% 2.7 in 6 of 10</td></tr></table>

![](images/d2247e819363d9246d0ef0f9c23f5a479ee53ecfe2dcf425ca4bd22d8a8e5400.jpg)  
Figure 3: Failed vetoes per session along practice for each of the ten tasks (gray) and their median (blue): in the first session, averaged over the first and the last five sessions before acceptance, and in the accepted session.

For the median task, the agent fails 6.3 vetoes per session over its first five sessions and 2.6 over its last five before acceptance. Progress is not always smooth: on three short runs, the last five sessions fail as many vetoes as the first five or more, and the task is then passed in a single step. Throughout practice, the simulated user is satisfied in a median of 34% of the sessions, long before the deliverable passes the vetoes, which is the gap that Section 5.2 describes. The memory changes as the agent learns: of the 155 learning steps in the latest round, 115 (74%) edited the memory, with 4.5 edits per step on average. Through these edits, the agent learns the deliverables that the evaluator demands, such as a trade log with fill prices, a specification table for the six components, and diagnostics with numbers.

The last veto to clear is most often strategy ambiguity (2.7), in 6 of 10 tasks. It fails in 77% of the 496 sessions before acceptance and in 28 of the 42 sessions that failed exactly one veto. Its reasons are withheld by the leak-free rule, so the agent must infer them from the user’s words. Learning from feedback is thus possible under the leak-free rule, and FinCosmos supplies both the loop and its measurement.

## 6 Conclusion

Agents doing financial work fail silently: an unrequested strategy, unverified data, or a backtest that uses the future produces a report as convincing as a correct one. FinCosmos measures these failures by keeping the user’s intent, the data, and time open, and by grading deliverables against the hidden specification and against invariants that programs can verify. Our experiments show that no agent completes the workflow. Agents break most often at its two ends, pinning down the user’s strategy and delivering verifiable results, grading by user sign-of or by an LLM judge would count their silent failures as successes, and an agent can learn from FinCosmos feedback.

Limitations. The benchmark freezes ten tasks with three sessions per agent, and its grammar covers long-only US equity strategies with a fixed-cost model. The simulator, the judge, and one evaluated agent come from the same model family; program checks and expert annotation limit, but do not remove, this risk. Most agent turns hit the 300-second limit, so longer turns may let agents get further. Program checks cannot prove that every figure is real, and feedback scores can leak information about the hidden strategy across attempts, so leaderboard results should come from frozen agents.

Ethics Statement. FinCosmos uses only public Web data, collects no personal data, and evaluates research assistance, not investment advice. The two annotators are paid team members doing this as regular work; judging agent outputs is not human-subjects research.

## Acknowled<sub>g</sub>ment

The authors acknowledge The Fin AI community for its research support, feedback, and collaborative environment that contributed to this work. This research was supported by the NVIDIA Academic Grant Program using 32K A100 GPU-hours on Brev.

## References

[1] David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, and Qiji Jim Zhu. The probability of backtest overfitting. Journal of Computational Finance, 20(4):39–69, 2017.

[2] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. �<sup>2</sup>-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv preprint arXiv:2506.07982, 2025.

[3] Antoine Bigeard, Langston Nashold, Rayan Krishnan, and Shirley Wu. Finance Agent Benchmark: Benchmarking LLMs on Real-world Financial Research Tasks. arXiv preprint arXiv:2508.00828, 2025.

[4] Yupeng Cao, Haohang Li, Weijin Liu, Wenbo Cao, Anke Xu, Lingfei Qian, Xueqing Peng, Minxue Tang, Zhiyuan Yao, Jimin Huang, et al. Fintrace: Holistic trajectory-level evaluation of llm tool calling for long-horizon financial tasks. arXiv preprint arXiv:2604.10015, 2026.

[5] Hao Chen, Xing Tang, Qirui Liu, Weijie Shi, Shiwei Li, Fuyuan Lyu, Weihong Luo, Xiku Du, and Xiuqiang He. Fighting numerical hallucinations via data-centric compilation for online financial qa. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 7082–7093, 2026.

[6] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al. FinQA: A dataset of numerical reasoning over financial data. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711, 2021.

[7] Yuyang Dai, Yan Lin, Zhuohan Xie, and Yuxia Wang. RealFin: How Well Do LLMs Reason About Finance When Users Leave Things Unsaid? In Findings of the Association for Computational Linguistics: ACL 2026, pages 25050–25080. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.findings-acl.1255. URL https://aclanthology.org/2026.findings-acl.1255/.

[8] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2Web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

[9] Felix Drinkall, Eghbal Rahimikia, Janet B. Pierrehumbert, and Stefan Zohren. Time machine GPT. In Findings ofthe Associationfor Computational Linguistics: NAACL 2024, 2024.

[10] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. DeepResearch Bench: A comprehensive benchmark for deep research agents. arXiv preprint arXiv:2506.11763, 2025.

[11] Paul Glasserman and Caden Lin. Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis. arXiv preprint arXiv:2309.17322, 2023.

[12] Jie Gong, Maowei Jiang, Zhiwei Liu, Yang Qiao, Wenxi Wu, Mengxi Xiao, Enze Zhang, Ziyan Kuang, Yankai Chen, Caishuang Huang, Meng Zhou, Xiku Du, Xue Liu, Guojun Xiong, Min Peng, Qianqian Xie, and Sophia Ananiadou. Shijianbench: From dialogue to decision for long-horizon evaluation of investment advisors. arXiv preprint arXiv:2608.01204, 2026.

[13] Xin Guo, Rongjunchen Zhang, Guilong Lu, Xuntao Guo, Shuai Jia, Zhi Yang, and Liwen Zhang. BizFinBench.v2: Towards Reliable LLMs in Finance via Real-User Data and Ofline/Online Bilingual Evaluation. arXiv preprint arXiv:2601.06401, 2026.

[14] Shijie Han, Haoqiang Kang, Bo Jin, Xiao-Yang Liu, and Steve Y Yang. XBRL agent: Leveraging large language models for financial report analysis. In Proceedings of the 5th ACM International Conference on AI in Finance, pages 856–864, 2024.

[15] Yi Han, Yan Wang, Lingfei Qian, Haohang Li, Yupeng Cao, Yueru He, Xueqing Peng, Nanhan Shen, Yitao Xu, Yankai Chen, Dongji Feng, Jimin Huang, Xue Liu, Jian-Yun Nie, and Sophia Ananiadou. Can LLM agents be CFOs? benchmarking long-horizon resource allocation in an uncertain enterprise environment. arXiv preprint arXiv:2603.23638, 2026.

[16] Jiajie He, Jiangyuan Hong, Xintong Chen, Dongling Ni, and Wenjin Liu. Are LLMs Good Financial User Simulators? Multi-view Investor Logic Alignment (MILA). arXiv preprint arXiv:2609.15727, 2026.

[17] Songrun He, Linying Lv, Asaf Manela, and Jimmy Wu. Chronologically consistent large language models. arXiv preprint arXiv:2502.21206, 2025.

[18] Fengrui Hua, Hengyi Yang, Xinlei Hao, Haohan Zhang, Bokai Cao, Yiyan Qi, Jia Li, and Jian Guo. Agentic Quantitative Trading: A Survey of Workflows, Systems, and Evaluation. arXiv preprint arXiv:2608.31041, 2026.

[19] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. FinanceBench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944, 2023.

[20] Haozhe Jia. HindsightBench: A Black-Box Behavioral Audit Protocol for Parametric Hindsight in Time-Indexed LLM Decision Tasks. arXiv preprint arXiv:2607.18867, 2026.

[21] Huaiyu Jia, Mingxuan Zhao, Jincheng Gao, Zifan Peng, Wentao Zhang, Siguang Li, and Shuo Sun. AlphaOpsBench: Benchmarking End-to-End Alpha Strategy Operationalization in Prediction Markets. arXiv preprint arXiv:2609.31390, 2026.

[22] Jiajie Jin, Yuyao Zhang, Yimeng Xu, Hongjin Qian, Yutao Zhu, and Zhicheng Dou. FinSight: Towards Real-World Financial Deep Research. arXiv preprint arXiv:2510.16844, 2025.

[23] Sunghwan Kim, Ryang Heo, Yongsik Seo, Jinyoung Yeo, and Dongha Lee. AgenticShop: Benchmarking agentic product curation for personalized web shopping. In Proceedings ofthe ACM Web Conference 2026, pages 2489–2500, 2026.

[24] Yaxuan Kong, Hoyoung Lee, Yoontae Hwang, Alejandro Lopez-Lira, Bradford Levy, Dhagash Mehta, Qingsong Wen, Chanyeol Choi, Yongjae Lee, and Stefan Zohren. Evaluating LLMs in Finance Requires Explicit Bias Consideration. arXiv preprint arXiv:2602.14233, 2026.

[25] Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs Get Lost In Multi-Turn Conversation. In International Conference on Learning Representations, 2026. URL https: //arxiv.org/abs/2505.06120.

[26] Changlun Li, Yao Shi, Chen Wang, Qiqi Duan, Runke Ruan, Weijie Huang, Haonan Long, Lijun Huang, Nan Tang, and Yuyu Luo. Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2505.11065.

[27] Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, Kp Subbalakshmi, Jimin Huang, et al. InvestorBench: A benchmark for financial decision-making tasks with LLM-based agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2509–2525, 2025.

[28] Weixian Waylon Li, Hyeonjun Kim, Mihai Cucuringu, and Tiejun Ma. Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery, 2026. doi: 10.1145/3770854.3785702. URL https://arxiv.org/abs/2505.07078.

[29] Weixian Waylon Li, Mengyu Wang, and Tiejun Ma. Summoning the Oracle to Slay It: Mitigating Look-Ahead Bias in Financial Backtesting with Large Language Models. In Proceedings ofthe 2026 Conference on Empirical Methods in Natural Language Processing, 2026. URL https://arxiv.org/abs/2605.24564.

[30] Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, and Xiangmin Xu. Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents. arXiv preprint arXiv:2510.07920, 2025.

[31] Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, and Jiang Bian. R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2505.15155.

[32] Zongqian Li, Yaoyiran Li, Yaohui Guo, Ming Zhang, Nigel Collier, and Eugene Ie. AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment. arXiv preprint arXiv:2608.28632, 2026.

[33] Shengyuan Colin Lin, Felix Tian, Keyi Wang, Xingjian Zhao, Jimin Huang, Qianqian Xie, Luca Borella, Matt White, Christina Dan Wang, Kairong Xiao, Xiao-Yang Liu, and Li Deng. Open FinLLM leaderboard: Towards financial AI readiness. arXiv preprint arXiv:2501.10963, 2025.

[34] Alejandro Lopez-Lira, Yuehua Tang, and Mingyin Zhu. The Memorization Problem: Can We Trust LLMs’ Economic Forecasts? arXiv preprint arXiv:2504.14765, 2025.

[35] Anas Mahmoud, MohammadHossein Rezaei, Zihao Wang, Anisha Gunjal, Bing Liu, and Yunzhong He. Reward Hacking in Rubric-Based Reinforcement Learning. arXiv preprint arXiv:2605.12474, 2026.

[36] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: A Benchmark for General AI Assistants. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2311.12983.

[37] Tarek Naous, Philippe Laban, Wei Xu, and Jennifer Neville. Flipping the Dialogue: Training and Evaluating User Language Models. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2510.06552.

[38] Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi Fan, Xiao-yong Wei, Shanru Lin, Hui Liu, Philip S Yu, et al. A survey of WebAgents: Towards next-generation AI agents for web automation with large foundation models. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 6140–6150, 2025.

[39] Xueqing Peng, Lingfei Qian, Yan Wang, Ruoyu Xiang, Yueru He, Yang Ren, Mingyang Jiang, Vincent Jim Zhang, Yuqing Guo, Jef Zhao, et al. MultiFinBen: Benchmarking large language models for multilingual and multimodal financial application. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 16934–16963, 2026.

[40] Xueqing Peng, Zhuohan Xie, Yupeng Cao, Haohang Li, Lingfei Qian, Yan Wang, Vincent Jim Zhang, Huan He, Xuguang Ai, Linhai Ma, Ruoyu Xiang, Yueru He, Yi Han, Shuyao Wang, Yuqing Guo, Mingyang Jiang, Yilun Zhao, Youzhong Dong, Xiaoyu Wang, Yankai Chen, Ye Yuan, Qiyuan Zhang, Fuyuan Lyu, Haolun Wu, Yonghan Yang, Zichen Zhao, Yuyang Dai, Fan Zhang, Rania Elbadry, Ayesha Gull, Muhammad Usman Safder, Nuo Chen, Fengbin Zhu, Tianshi Cai, Zimu Wang, Polydoros Giannouris, Yuechen Jiang, Zhiwei Liu, Mohsinul Kabir, Yuyan Wang, Yixiang Zheng, Yangyang Yu, Weijin Liu, Wenbo Cao, Anke Xu, Peng Lu, Jerry Huang, Mingquan Lin, Prayag Tiwari, Yijia Zhao, Víctor Gutiérrez-Basulto, Xiao-Yang Liu, Kaleb E Smith, Jiahuan Pei, Arman Cohan, Jimin Huang, Yuehua Tang, Alejandro Lopez-Lira, Xi Chen, Xue Liu, Junichi Tsujii, Jian-Yun Nie, and Sophia Ananiadou. Herculean: An Agentic Benchmark for Financial Intelligence. arXiv preprint arXiv:2605.14355, 2026.

[41] Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, and Huan Wang. UserBench: An Interactive Gym Environment for User-Centric Agents. arXiv preprint arXiv:2507.22034, 2025.

[42] Lingfei Qian, Xueqing Peng, Hanley Smith, Yi Han, Yueru He, Haohang Li, Yupeng Cao, Yangyang Yu, Guojun Xiong, Peng Lu, et al. When agents trade: Live multi-market trading arena for llm agents. In Proceedings ofthe ACM Web Conference 2026, pages 7833–7844, 2026.

[43] Bo Qu and Mingguang Chen. CLQT: A Closed-Loop, Cost-Aware, Strategy-Consistent Benchmark for Diagnostic Evaluation of LLM Portfolio-Management Agents. arXiv preprint arXiv:2606.29771, 2026.

[44] Soham Ray, Keshav Dhandhania, Victor Barres, and Karthik Narasimhan. �-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains. arXiv preprint arXiv:2603.13686, 2026.

[45] Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, and Seraphina Goldfarb-Tarrant. Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations. arXiv preprint arXiv:2601.17087, 2026.

[46] Quan Shi, Alexandra Zytek, Pedram Razavi, Karthik Narasimhan, and Victor Barres. �-Knowledge: Evaluating conversational agents over unstructured knowledge. arXiv preprint arXiv:2603.04370, 2026.

[47] Giulio Starace, Oliver Jafe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. PaperBench: Evaluating AI’s Ability to Replicate AI Research. arXiv preprint arXiv:2504.01848, 2025.

[48] Makar Ulesov, Vladislav Smirnov, Omar Ibrahim, and Arsenii Bobovnikov. Can LLMs Catch a Rigged Backtest? A Clean-Control Calibration Benchmark. arXiv preprint arXiv:2609.28090, 2026.

[49] Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, and Graham Neubig. Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2502.13069.

[50] Saizhuo Wang, Hang Yuan, Leon Zhou, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2025. URL https://arxiv.org/abs/2308.00016.

[51] Yan Wang, Yi Han, Lingfei Qian, Yueru He, Xueqing Peng, Dongji Feng, Zhuohan Xie, Vincent Jim Zhang, Rosie Guo, Fengran Mo, Jimin Huang, Yankai Chen, Xue Liu, and Jian-Yun Nie. Convfinre: A conversational and longitudinal benchmark for utility-grounded financial recommendation. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2026.

[52] Yan Wang, Keyi Wang, Shanshan Yang, Jaisal Patel, Jef Zhao, Fengran Mo, Xueqing Peng, Lingfei Qian, Yankai Chen, Víctor Gutiérrez-Basulto, et al. FinAuditing: A financial taxonomy-structured multi-document benchmark for evaluating LLMs. In Proceedings ofthe 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3456–3463, 2026.

[53] Zhensheng Wang, Wenmian Yang, Qingtai Wu, Lequan Ma, Yiquan Zhang, and Weijia Jia. BacktestBench: Benchmarking Large Language Models for Automated Quantitative Strategy Backtesting. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery, 2026. doi: 10.1145/3770855.3817460. URL https://arxiv.org/abs/2605.17937.

[54] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jefrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents. arXiv preprint arXiv:2504.12516, 2025.

[55] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. LiveBench: A challenging, contamination-limited LLM benchmark. In International Conference on Learning Representations, 2025.

[56] Yijia Xiao, Edward Sun, Di Luo, and Wei Wang. TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv preprint arXiv:2412.20138, 2024.

[57] Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2306.05443.

[58] Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, Yijing Xu, Haoqiang Kang, Ziyan Kuang, Chenhan Yuan, Kailai Yang, Zheheng Luo, Tianlin Zhang, Zhiwei Liu, Guojun Xiong, Zhiyang Deng, Yuechen Jiang, Zhiyuan Yao, Haohang Li, Yangyang Yu, Gang Hu, Jiajia Huang, Xiao-Yang Liu, Alejandro Lopez-Lira, Benyou Wang, Yanzhao Lai, Hao Wang, Min Peng, Sophia Ananiadou, and Jimin Huang. FinBen: A Holistic Financial Benchmark for Large Language Models. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2402.12659.

[59] Guojun Xiong, Zhiyang Deng, Keyi Wang, Yupeng Cao, Haohang Li, Yangyang Yu, Xueqing Peng, Mingquan Lin, Kaleb E Smith, Xiao-Yang Liu, Jimin Huang, Sophia Ananiadou, and Qianqian Xie. FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading. In Findings of the Association for Computational Linguistics: ACL 2025, pages 13921–13934. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-acl.716. URL https://aclanthology.org/2025.findings-acl.716/.

[60] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. �-bench: A benchmark for toolagent-user interaction in real-world domains. In International Conference on Learning Representations, 2025.

[61] Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Denghui Zhang, Rong Liu, Jordan W. Suchow, and Khaldoun Khashanah. FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design. arXiv preprint arXiv:2311.13743, 2023.

[62] Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yuechen Jiang, Yupeng Cao, Zhi Chen, Jordan W Suchow, Zhenyu Cui, Rong Liu, et al. FinCon: A synthesized LLM multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. Advances in Neural Information Processing Systems, 37:137010–137045, 2024.

[63] Fan Zhang, Zhen Li, Sijia Peng, and Yu Chen. When Alpha Disappears: A One-Switch Benchmark for Decision-Time Leakage in Financial Backtests. arXiv preprint arXiv:2605.23959, 2026.

[64] Wentao Zhang, Lingxuan Zhao, Haochong Xia, Shuo Sun, Jiaze Sun, Molei Qin, Xinyi Li, Yuqing Zhao, Yilei Zhao, Xinyu Cai, Longtao Zheng, Xinrun Wang, and Bo An. A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4314–4325. Association for Computing Machinery, 2024. doi: 10.1145/3637528.3671801. URL https://arxiv.org/abs/2402.18485.

[65] Wentao Zhang, Mingxuan Zhao, Jincheng Gao, Jieshun You, Huaiyu Jia, Yilei Zhao, Bo An, and Shuo Sun. AlphaForgeBench: Benchmarking End-to-End Trading Strategy Design with Large Language Models. arXiv preprint arXiv:2602.18481, 2026.

[66] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

[67] Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Jing Jiang, and Min Lin. Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2410.07137.

[68] Leijun Zhou, Zhihao Liu, Xiang Qu, Chenxu Liu, Yifei Liu, Yanke Yu, Jingzhe Xu, Xuejun Wu, Buyue Qian, Xi Chen, Yaowei Zheng, and Junhao Hu. GDPevo: Evaluating Agent Self-Evolution on Real Business Tasks. arXiv preprint arXiv:2608.03764, 2026.

[69] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. WebArena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024.

[70] Fengbin Zhu, Xiang Yao Ng, Ziyang Liu, Chang Liu, Xianwei Zeng, Chao Wang, Tianhui Tan, Xuan Yao, Pengyang Shao, Min Xu, Zixuan Wang, Jing Wang, Xin Lin, Junfeng Li, Jingxian Zhu, Yang Zhang, Wenjie Wang, Fuli Feng, Richang Hong, Huanbo Luan, Ke-Wei Huang, and Tat-Seng Chua. FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis. arXiv preprint arXiv:2510.13936, 2025.

[71] Yuxuan Zhu, Tengjun Jin, Yada Pruksachatkun, Andy Zhang, Shu Liu, Sasha Cui, Sayash Kapoor, Shayne Longpre, Kevin Meng, Rebecca Weiss, Fazl Barez, Rahul Gupta, Jwala Dhamala, Jacob Merizian, Mario Giulianelli, Harry Coppock, Cozmin Ududec, Jasjeet Sekhon, Jacob Steinhardt, Antony Kellermann, Sarah Schwettmann, Matei Zaharia, Ion Stoica, Percy Liang, and Daniel Kang. Establishing Best Practices for Building Rigorous Agentic Benchmarks. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2507.02825.

![](images/005a7a1d9e512b3190a367463a8cbfd1745fe91d65a410da4464b610bf22ccc5.jpg)  
Figure 4: Dimension scores of all 150 sessions in the main experiment. Each column is one session, grouped by agent; within each agent, the sessions that the simulated user signed of (top strip) come first, left of the dark line, and each group is ordered by task. Each row is one of the 37 scored dimensions; veto dimensions are in bold, and the shading behind each label marks its judge: programs, programs and an LLM, or an LLM alone (Table 7).

## A Per-Session Scores

Figure 4 shows the score of every scored dimension in every session.

## B Full Rubric

Table 7 lists all 45 dimensions. Every scored dimension uses a 0–4 scale with an anchored description for each level; these descriptions and the full definition of each check are released with the evaluator. Judge: P = program, L = LLM, H = program and LLM, L↓P = the lower of an LLM grade and a program score, L/P = LLM judgment scored by a program rule, R = reported only. Veto: the session fails if the score is below the given threshold. On the public leaderboard, option names, parameter names, and matching values in the evidence are replaced by a placeholder.

## C Session End Reasons

Every session records exactly one end reason (Table 8). The end reason determines whether the session is graded, whether it can be accepted, and whether it counts as a valid session.

Table 8: Session end reasons and their efect on grading.

Table 7: Rubric: the 45 dimensions, their judges, and veto thresholds.
<table><tr><td>ID</td><td>Dimension</td><td>Judge</td><td>Veto</td><td>ID</td><td>Dimension</td><td>Judge</td><td>Veto</td><td>ID</td><td>Dimension</td><td>Judge</td><td>Veto</td></tr><tr><td>Data</td><td></td><td></td><td></td><td>Backtest</td><td></td><td></td><td></td><td></td><td>User experience</td><td></td><td></td></tr><tr><td>1.1</td><td>Point-in-time strictness</td><td>P</td><td>&lt; 4</td><td></td><td>3.1 Point-in-time correctness</td><td>P</td><td>&lt; 2</td><td>5.1</td><td>Conversation experience</td><td></td><td>&lt; 2</td></tr><tr><td>1.2</td><td>Source traceability</td><td>L</td><td></td><td>3.2</td><td>Independent recomputation</td><td>P</td><td>b</td><td>5.2</td><td>Tone</td><td></td><td></td></tr><tr><td>1.3</td><td>Source diversity</td><td>L</td><td></td><td>3.3</td><td>Core metrics reported</td><td>P</td><td></td><td>5.3</td><td>Professionalism</td><td></td><td></td></tr><tr><td>1.4</td><td>Delivered data quality</td><td>P</td><td></td><td>3.4</td><td>Benchmark comparison</td><td>L/P</td><td></td><td>5.4</td><td>Narrative</td><td></td><td></td></tr><tr><td>1.5</td><td>Evidence grade match</td><td>L</td><td></td><td>3.5</td><td>Cost model</td><td>P</td><td></td><td>5.5</td><td>AI boilerplate</td><td></td><td></td></tr><tr><td>1.6</td><td>Quality self-check disclosure</td><td>L</td><td></td><td>3.6</td><td>Code reproducibility</td><td>P</td><td>C</td><td>5.6</td><td>Internal leak</td><td>L↓P L</td><td>&lt; 2</td></tr><tr><td>Strategy</td><td></td><td></td><td></td><td>3.7</td><td>Verifier audit</td><td>P</td><td>&lt; 4</td><td></td><td>Cost (reported only)</td><td></td><td></td></tr><tr><td>2.1</td><td>Stock selectiona</td><td>H</td><td>&lt; 2</td><td>3.8</td><td>Process completeness</td><td>P</td><td>&lt; 4</td><td></td><td>6.1 Time</td><td></td><td></td></tr><tr><td>2.2</td><td>Market timinga</td><td>H</td><td>&lt; 2</td><td></td><td>Assessment</td><td></td><td></td><td></td><td>6.2 Turns</td><td></td><td></td></tr><tr><td>2.3</td><td>Portfolio constructiona</td><td>H</td><td>&lt; 2</td><td>4.1</td><td>Grounded conclusions</td><td>L</td><td>&lt; 2</td><td>6.3</td><td>Tokens</td><td></td><td></td></tr><tr><td>2.4</td><td>Risk controla</td><td>H</td><td>&lt; 2</td><td>4.2</td><td>Evidence citation</td><td>P</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2.5</td><td>Executiona</td><td>H</td><td>&lt; 2</td><td>4.4</td><td>Risk disclosure</td><td>L</td><td></td><td></td><td>Process transparency (reported only)</td><td></td><td></td></tr><tr><td>2.6</td><td>Transaction costa</td><td>H</td><td>&lt; 2</td><td>4.5</td><td>Sensitivity analysis</td><td>L</td><td></td><td></td><td>7.1 Fetch failures</td><td>P P</td><td></td></tr><tr><td>2.7</td><td>Strategy ambiguity</td><td>L</td><td>&lt; 2</td><td>4.6</td><td>Out-of-sample validation</td><td>L</td><td></td><td>7.2</td><td>Intermediate reproducibility</td><td></td><td></td></tr><tr><td>2.8a</td><td>Signal rationale</td><td>L</td><td></td><td></td><td></td><td></td><td></td><td>7.3</td><td>Post-cutoff traces</td><td></td><td></td></tr><tr><td>2.8b</td><td>Signal validation</td><td>P</td><td></td><td></td><td></td><td></td><td></td><td>7.4</td><td>Delivery-process consistency</td><td>P P</td><td></td></tr><tr><td>2.9</td><td>Parameter justification</td><td>L</td><td></td><td></td><td></td><td></td><td></td><td>7.5</td><td>Process coverage</td><td></td><td></td></tr><tr><td>2.10</td><td>Boundary conditions</td><td>L/P</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2.12</td><td>Decision explainability</td><td>P</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>�</sup> A missing requirement that the user stated scores at most 1 and fails the session; a missing requirement that the user never stated caps the score at 2. For risk control, the deliverables must show positions changing because of the control. <sup>�</sup> Fails if the trade log or the prices are missing; mismatches with the report lower the score. <sup>�</sup> Bonus: counts only if it raises the category mean.

<table><tr><td>End reason</td><td>Meaning</td><td>Effect on grading</td></tr><tr><td>satisfied</td><td>The user closes the session after every core goal is met Graded normally; can be accepted. with evidence.</td><td></td></tr><tr><td>blocked</td><td>The user ends the session without being satisfied, for Graded normally, but the conversation example when the agent explicitly refuses to run the backtest.</td><td>experience veto (5.1) fails, so the score is 0.</td></tr><tr><td>stalled</td><td>FinCosmos stops the session after several turns without improvement in the per-turn scores (optional; can be accepted. off in our experiments).</td><td>Graded normally on what was delivered;</td></tr><tr><td>timeout</td><td>The session exceeds its time limit, or a turn times out Graded on what was delivered before the and the agent cannot be confirmed to have stopped.</td><td>cut; never accepted.</td></tr><tr><td>max_turns</td><td>The turn limit is reached before the user is satisfied.</td><td>No result: dimensions are graded as feedback, but the score is 0 and the</td></tr><tr><td>resource_killed</td><td>The agent exceeds its own resource limits.</td><td>session counts as a failure. Same as max_turns.</td></tr><tr><td>simulator_error</td><td>The user simulator fails, or the agent never completes Not a valid session; never accepted. a turn.</td><td></td></tr><tr><td>error</td><td>The agent adapter or the runner crashes.</td><td>Not a valid session; never accepted.</td></tr></table>

## D User Simulator Details

Personas. Each persona consists of a background prompt, a speaking style, focus areas, example utterances, and a list of forbidden terms. A retail investor uses finance terms confidently but often imprecisely. A generalist analyst speaks in a neutral, professional way and asks about metrics such as drawdown, turnover, and Sharpe ratio. A biotech specialist uses heavy industry jargon and thinks in terms of events such as catalysts and FDA decisions. The forbidden terms include evaluation vocabulary such as “point-in-time” and “cutof”.

Disclosure rules. The simulator’s prompt implements Equation 2 with five rules. (1) Open with a single vague idea. (2) Answer only what is directly asked. What is not asked, the user cannot recall and does not volunteer. (3) Asked for a specific value or method, such as a stop-loss threshold, give it directly rather than asking the agent to propose one. (4) When a proposal conflicts with the belief, object and say what you want.

(5) When a deliverable deviates from the belief, name the piece that is of, such as the position sizing or the factor source, but do not restate the strategy. The prompt also forbids assistant-like phrases such as “let me lock a few things”.

Turn decisions. In each turn, the simulator reads the conversation and the agent’s most recent deliverables: at most 20 files, 8,000 characters per file, and 12,000 characters in total, with the budget going to the newest files first. Long conversations are compressed to the last six messages and a summary of the earlier ones. The decision contains an action, one observation per goal of the task, and the next message. The actions are clarify (answer the agent’s question),follow up (push unfinished work forward), challenge (point out a visible inconsistency), close, and block, which ends the session without satisfaction and records a reason, such as the agent giving up or the user walking away. Each observation marks its goal as met, missing, uncertain, or not applicable and cites the IDs of the supporting messages or files. In the released tasks, the core goal is a backtest report on the user’s strategy.

Decision validation. A program checks every decision before its message is sent. It rejects a decision that leaves a goal unobserved, marks a goal as met without citing the agent’s messages or files, closes the session while a core goal is not met, raises a challenge or a block without evidence, uses internal vocabulary such as file names of the grading pipeline, asks for more work in a closing message, or repeats an earlier message verbatim. A rejected decision is regenerated with the reasons for rejection, up to six times. If no valid decision results, the simulator run is voided and the session does not count (Appendix C).

## E Ex<sub>p</sub>ert Annotation Guidelines

This appendix summarizes the guidelines of Section 4.3; the full guidelines and the reading pages are online<sup>3</sup>. Each session page shows the persona, the hidden strategy, the full conversation, and the delivered files. Parts of the strategy that the user was never given, and replies stopped at the 300-second limit, are marked.

Phase 1: without grades. (1) Every user message receives one label. Error-free: consistent with the hidden strategy; the message adds a detail only when the agent asked about or proposed it, or names the wrong part of a deliverable without restating the strategy. Task-benign: a slip that cannot change whether a correct agent is accepted, such as odd wording or an improvised value for a detail the strategy leaves open. Task-critical: an error that can stop a correct agent from being accepted, such as contradicting the hidden strategy, volunteering details nobody asked for, signing of on a deliverable that visibly misses a stated requirement, or giving up while the agent is still making progress. (2) On the calibration sessions, the experts grade the four LLM-judged vetoes (strategy ambiguity, grounded conclusions, conversation experience, and internal leak) on the judge’s anchored 0–4 scale, using only the conversation and the delivered files.

Phase 2: with the evaluator’s verdict. After submitting Phase 1, the experts see the vetoes that each session failed, with the evaluator’s reasons. (3) For every failed veto, they mark whether the reason holds for the delivery and is a real problem for an investor, or is wrong, for example because it flags something the user asked for. (4) They name the source of the first error that decides the outcome: the agent, the simulator, or the evaluator.