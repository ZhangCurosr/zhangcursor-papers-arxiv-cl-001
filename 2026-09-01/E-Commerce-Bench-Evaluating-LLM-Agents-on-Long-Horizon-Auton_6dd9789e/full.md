# E-Commerce Bench: Evaluating LLM Agents on Long-Horizon Autonomous Business Operation

Wei Fan1,3\* Xinjie Shen1, Xudong Guo1, Jianhong Tu1, Yang Su1, Yinger Zhang1,   
Lianghao Deng1, Fengyu Wang2, Baohua Dong2, Yangqiu Song3, Dayiheng Liu1 1Qwen Team, Alibaba Group 2Taobao & Tmall Group, Alibaba Group 3Department of Computer Science and Engineering, HKUST

## Abstract

Long-horizon agentic tasks go beyond chaining short tasks over more interaction turns. Their evolving dynamic environments and long-range dependencies require Large Language Models (LLMs) to continually explore, learn from experience, and adapt their policies over thousands of steps. We introduce E-Commerce Bench, the first open-source benchmark that integrates multi-round counterpart negotiation and dynamic events into a year-long business operation. Over a 365-day year, an LLM agent concurrently runs multiple online stores, researching the market, negotiating with suppliers to source inventory, optimizing sales strategies, fulfilling orders, handling returns, and managing cash flow to maximize its end-of-year total assets. To construct a realistic merchant-side operating environment, the product and supplier data are derived from a real e-commerce platform, while a year-long calendar of promotions, natural disasters, and supply-chain shocks continually reshapes demand. For reproducibility, both sides of the market are deterministic: customer purchases and returns follow a fixed demand model, while a negotiation kernel determines supplier pricing, concessions, and decisions, with an LLM used only to verbalize them. We evaluate 18 frontier models across seven dimensions, including year-end assets, and find that no single model dominates. GPT-5.6 Sol earns the most, growing the ¥100,000 opening stake into ¥1,431,425, yet it ranks 16th of 18 on fraud avoidance and trails Fable5 in operational efficiency. Among open-weight models, Qwen3.8-Max-Preview leads with ¥416,252, 38% above GLM 5.2 (high), and achieves the strongest learning over the horizon, progressively bargaining down prices across repeated orders. Our code is available at https://github.com/QwenLM/E-CommerceBench.

Year-End Assets, 5 Episodes per Model  
![](images/c7ef39434a7692984a9006b4c11bcb5eea8b1709fedcc2d8fb8929d1ede533fd.jpg)  
Figure 1: Mean end-of-year total assets, eighteen models, five 365-day episodes each. Bars start at the ¥100,000 stake. Whiskers span one 5-run standard deviation, hollow dots are single episodes, and red fractions count bankrupt episodes, which stay pooled into the mean. The right column gives mean assets as a multiple of the stake.

![](images/970ff23f4230b401378e6ec54b214c462efe0b3a435e9cef197445150e82e348.jpg)  
Figure 2: Seven evaluation axes for the strongest model of each vendor family by mean total assets, the primary score, and the six evaluation dimensions clockwise from the top, each on the ranking metric §3.7 gives it. Axes are min-max normalized over the eighteen model means, so 1 is the best of the eighteen and 0 the worst, with a dashed polygon at the median. Fraud avoidance, solvency, execution, and learning are sign-flipped, so a vertex further from the center is better on every axis.

## Contents

1 Introduction 4   
Relatedisdrc Long-Horizon Agentic Tasks 2.2 Continuing Long-Horizon Agentic Tasks 666   
3 E-Commerce Bench 7   
3.1 Overview   
3.2 Agent Loop Layer   
3.2.1 Turn-Based Loop and Simulated Time 8   
3.2.2 Context Management 8   
3.2.3 Persistent Agentic Memory 9   
3.3 Tool Layer . 9   
3.4 Sales and Economy Engine 9   
3.4.1 Multi-Store Setup 9   
3.4.2 Three-Account Deferred Settlement . 9   
3.4.3 Deterministic Multi-Factor Demand Model 10   
3.4.4 Fulfillment, Shipping Speed, and Returns 11   
3.4.5 Reputation System . 11   
3.4.6 Dynamic Event System 11   
3.4.7 Daily Settlement and Processing Lifecycle 11   
3.5 Deterministic Negotiation Engine 12   
3.5.1 Bargaining Policy Preliminaries 12   
3.5.2 Two-Layer Supplier Design 12   
3.5.3 Multi-Round, Čross-Supplier, Repeated Bargaining 13   
3.5.4 Adversarial Supplier Fraud 14   
3.5.5 Negotiation Metrics 14   
3.6 Data Layer 14   
3.6.1 Catalog and Suppliers 15   
3.6.2 Systematic Information Asymmetry 15   
3.7 Evaluation and Metrics 15   
Experimental Results 16   
4.1 Experimental Setup 16   
4.2 Overall Performance 16   
4.3 Multi-Dimensional Analysis 17   
4.3.1 Negotiation Quality 17   
4.3.2 Fraud Avoidance 18   
4.3.3 Cash Flow and Solvency 18   
4.3.4 Operational Efficiency 18   
4.3.5 Operations Execution 19   
4.3.6 Learning over the Horizon 19   
4.4 Error Analysis 20   
5 Conclusion 21   
A Agent Harness and Interface 26   
B Data Layer 28   
C Economy Engine 32   
D Deterministic Negotiation Kernel 36   
E Metric Definitions 42   
F Per-Dimension Results 49   
G Failure Case Studies and Detection Rules 53   
H Prompts 58

## 1 Introduction

Large language model (LLM) agents are increasingly expected to handle tasks that span thousands of steps, planning and acting over long trajectories with external tools (Yao et al., 2023). Such tasks require an agent to make sequential decisions, adapt to past experiences, and remain coherent long after the initial instruction. Sustained competence of this kind marks a frontier for current models and eludes most agentic benchmarks. In reinforcement learning, tasks are divided into episodic and continuing ones (Sutton & Barto, 2018), and this distinction also organizes the long-horizon agentic tasks. Episodic benchmarks set a goal and score its completion, whether resolving a GitHub issue (Jimenez et al., 2024), operating a real computer (Xie et al., 2024), or producing knowledge work that an expert grades (Patwardhan et al., 2025). Continuing tasks carry no terminal state, running without end in dynamic environments (Qian et al., 2025; Chen et al., 2026) and scoring an accumulated outcome that surfaces the failures long horizons produce among them loss of coherence and an inability to turn experience into better decisions. Business operations provide a natural continuing scenario with high-stakes, irreversible decisions, where compounding financial health tests sustained agent competence. Despite its promise, current benchmarks capture this environment only partially. While the Vending-Bench series (Backlund & Petersson, 2025; Andon Labs, 2026) simulates multi-round supplier negotiations, its reliance on mock data and prompt-based simulation yields highly stochastic supplier feedback that is vulnerable to manipulation. YC-Bench (He et al., 2026) runs a deterministic startup year yet involves no supplier sourcing. Data-grounded retail benchmarks such as MerchantBench (Shi et al., 2026) and RetailBench (Zhang et al., 2026c) add real products and open code while dropping negotiation and adversarial dynamics altogether. Evidently, existing benchmarks either deviate from real-world market dynamics or rely on purely LLM-driven counterparts for multi-round negotiations, which introduces substantial sampling noise and ultimately compromises both the credibility and reproducibility of the evaluation.

To address these, we introduce E-Commerce Bench, the first open-source benchmark that unifies deterministic supplier negotiation and customer sales modeling within a real-world continuing business operation (Figure 3). Across a simulated 365-day year, an agent can run up to four online stores concurrently, researching the market, negotiating for cost-effective goods, formulating sales strategies, and managing cash flow to maximize end-of-year total assets. Grounded in Taobao & Tmall data, the environment encompasses a total of 6,886 products and 576 suppliers calibrated from empirical platform logs, while a year-long calendar of promotional events, natural disasters, and supply-chain shocks continually shifts dynamic market demand. Critical market factors, such as baseline product costs and exact consumer demand, remain hidden, requiring the agent to continuously search, negotiate, and model market dynamics through a dedicated e-commerce toolset provided by the benchmark. Meanwhile, the benchmark harness automatically manages the context window and offers a persistent memory module (Packer et al., 2024; Park et al., 2023) to sustain a year-long operational horizon. To ensure rigorous reproducibility, determinism governs both sides of the market. Customer purchases and returns follow a fixed demand model that combines price elasticity, weekday, seasonality, scheduled promotions, market events, and store reputation, with its only randomness seeded from the episode so runs reproduce exactly. On the supply side, a deterministic negotiation kernel fixes every pricing, concession, and accept/reject decision through explicit bargaining policies, while an LLM only renders them as dialogue. Roughly a quarter of the suppliers turn adversarially fraudulent, on kernels that do not vary between runs. No sampling noise reaches the customers or the counterpart's pricing and accept/reject decisions, so a difference in outcome entirely belongs to the agent policy rather than to chance. Only the LLM renderer's wording varies between runs (§3.5.2).

We evaluate 18 leading frontier and open-weight models along multi-dimensional metrics spanning negotiation quality, fraud avoidance, cash flow and solvency, operational efficiency, operations execution, and learning over the horizon, rather than by total assets alone. A radar over the primary score and those six dimensions comes out visibly uneven, and six of the seven models it draws, one per vendor family, fall below the eighteen-model median on at least one axis (Figure 2). GPT-5.6 Sol grows its capital roughly 14-fold, yet ranks 16th of 18 on fraud avoidance and earns less from a tool call than Fable5 does, while Claude Opus 4.7 leads both bargaining and fraud avoidance from mid-pack on profit. Among the ten open-weight entries, first place goes to our own Qwen3.8-Max-Preview at 4.2 times the initial balance, and four of the eighteen models slip into bankruptcy on part of their runs. Qwen3.8-Max-Preview is the strongest model in the field, improving over the year. Across 8,647 repeat purchases of the same item from the same honest supplier, 16 models show no clear sign of pushing the purchase price down as the year goes on, exposing critical deficiencies in their long-horizon experiential learning.

We make three principal contributions.

1. We present E-Commerce Bench, a realistic business environment grounded in Taobao & Tmall platform data, and the first open-source benchmark to combine deterministic counterpart negotiations, dynamic events, and concurrent multi-store management over a year-long horizon.

2. We keep the economy deterministic on both sides of the market, with a fixed demand model driving customer purchases and returns and a deterministic negotiation kernel setting every pricing, concession, and accept/reject decision while an LLM only renders dialogue. The kernel extends the negotiation-partner design of TERMS-Bench (Zhang et al., 2026a) to repeated, cross-supplier bargaining.

3. We evaluate 18 models along multi-dimensional metrics, so one total-assets figure can no longer hide where a model is weak. Reading six dimensions beside that total turns a leaderboard into a capability profile, and the profiles come out sharply divergent, the best earner being neither the best negotiator nor the efficient operator.

The remainder of this paper is organized as follows. Section 2 reviews long-horizon agentic benchmarks and positions E-Commerce Bench among them. Section 3 details the benchmark, covering its four-layer architecture, the deterministic negotiation kernel, the real-data economy, and the evaluation metrics. Section 4 reports results for the 18 models on every measured dimension, together with an error analysis of recurring failure modes. Section 5 concludes.

Table 1: Comparison with continuing business-operation benchmarks. present, X absent. Steps/run is the per-episode agent-action count, taken or inferred from each paper. Op. units counts business units (e.g., stores) run concurrently. Reproducible marks counterparts that follow fixed rules or seeded draws rather than a sampled LLM. †In Vending-Bench $^ { 1 / 2 , }$ product and supplier names are real while prices, demand, and elasticity are LLM-generated. §Supplier economics are deterministic, while the renderer voicing them varies wording between runs (§3.5.2).
<table><tr><td>Benchmark</td><td>Domain</td><td>Steps/ run</td><td>Op. units</td><td>Real data</td><td>Nego- tiation</td><td>Multi- agent</td><td>Adver- sarial</td><td>Repro- ducible</td><td>Open source</td></tr><tr><td>Vending-Bench (Backlund &amp; Petersson, 2025)</td><td>Vending</td><td>≤2000</td><td>1</td><td>x</td><td></td><td></td><td>x</td><td>X</td><td>X</td></tr><tr><td>Vending-Bench 2 (Andon Labs, 2026)</td><td>Vending</td><td>3000-6000</td><td>1</td><td>x</td><td></td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>YC-Bench (He et al., 2026)</td><td>Startup</td><td>~500</td><td>1</td><td>x</td><td>x</td><td></td><td></td><td></td><td></td></tr><tr><td>MerchantBench (Shi et al., 2026)</td><td>E-commerce</td><td>~700</td><td>1</td><td></td><td>x</td><td>x</td><td>X</td><td></td><td></td></tr><tr><td>RetailBench (Zhang et al., 2026c)</td><td>Supermarket</td><td>~1000</td><td>1</td><td></td><td>x</td><td>X</td><td>x</td><td></td><td></td></tr><tr><td>E-Commerce Bench (Ours)</td><td>E-commerce</td><td>1000-4000</td><td>≤4</td><td>V</td><td></td><td>L</td><td></td><td></td><td></td></tr></table>

## 2 Related Work

## 2.1 Episodic Long-Horizon Agentic Tasks

Long-horizon agentic benchmarks are divided by whether a run carries a terminal state (Sutton & Barto, 2018). Episodic benchmarks define one, setting a goal and rewarding the multi-step execution that reaches it. Work-value benchmarks grade an economically meaningful deliverable, GDPval (Patwardhan et al., 2025) scoring knowledge work against expert judgment and TheAgentCompany (Xu et al., 2025) setting a simulated employee's tasks inside a software firm. Environment-grounded benchmarks stretch the horizon through real software, SWE-bench (Jimenez et al., 2024) resolving a GitHub issue against its own tests and OSWorld (Xie et al., 2024) driving a real operating system over many GUI steps. Exploration-driven benchmarks probe reasoning further still, UltraHorizon (Luo et al., 2025) hiding the rules of a partially observable world and EdgeBench (Zhu et al., 2026) learning from one environment for 12 to 72 hours at a time. However long these tasks run, each ends the moment its goal is met, reducing evaluation solely to the quality of the final deliverable. Consequently, this setup fails to assess an agent's ability in dynamic environments to track dependencies across hundreds of sub-tasks, recover from unexpected failures, or leverage accumulated history to optimize future actions.

## 2.2 Continuing Long-Horizon Agentic Tasks

Continuing tasks carry no terminal state, so the agent acts indefinitely and earns an accumulated score The shape recurs across trading a live portfolio for cumulative return (Qian et al., 2025; Chen et al., 2026) and self-evolving agents that grow a skill library (Wang et al., 2023) or rewrite their own code (Zhang et al., 2026b). The setting E-Commerce Bench belongs to is business-operation simulation, where the agent runs a going concern and is judged by its accumulated balance. Vending-Bench (Backlund & Petersson 2025) established the line, running one vending machine over roughly two thousand messages and exposing failures of long-term coherence, with LLM-generated supplier replies that leave runs stochastic. Vending-Bench 2 (Andon Labs, 2026) extends it to a full year and adds adversarial suppliers, still LLM-driven. YC-Bench (He et al., 2026) runs a deterministic startup year against rule-based clients but no supplier negotiation. Two benchmarks are grounded in real data, MerchantBench (Shi et al., 2026) over 365 days of seller-side operation with suppliers arriving as stochastic events, and RetailBench (Zhang et al., 2026c) over 180 supermarket days against a privileged oracle. Neither offers negotiation or adversarial suppliers and RetailBench lists strategic supplier behavior and multi-store coordination among the things it leaves out. Table 1 contrasts the five. The Vending-Bench line couples negotiation with adversarial suppliers in its sequel, but only through sampled LLM parties and without an open, real-data economy, while the data-grounded retail pair buys reproducibility by giving those dynamics up. E-Commerce Bench holds all six properties at once, and no other entry pairs a negotiating counterpart with reproducible supplier economics. TERMS-Bench (Zhang et al., 2026a) sits outside the business-operation line, diagnosing bilateral price bargaining against a rule-based counterpart over sampled regimes rather than data-grounded cost floors, so it supplies a scripted opponent but no economy for the resulting deals to act on. Our market is deterministic on both sides, a fixed demand model driving customers while a deterministic negotiation kernel adapts that counterpart design to set every pricing and accept/reject decision, leaving an LLM only the dialogue rendering.

![](images/7f2f95eebd4d8f5c3dbea8c6ec578ca74def88068b0b5b1ac26d7173b8c0ef8c.jpg)  
Figure 3: Overview of E-Commerce Bench. The agent works through four layers. An agent loop layer carries turn-based control, context management, and persistent memory, and a tool layer holds the 18 e-commerce tools. Under both sits a deterministic environment layer, the sales-and-economy engine beside the two-layer negotiation engine, computing over a data layer derived from a real e-commerce platform.

## 3 E-Commerce Bench

## 3.1 Overview

E-Commerce Bench hands the agent a merchant account, ¥100,000, and the 2026 calendar year on a simulated Chinese online marketplace. The job is to end the year with as much money as possible. The agent researches which store types and categories carry demand, bargains suppliers down, buys inventory and publishes it to one of at most four stores. From there it prices the goods, reads yesterday's sales, ships orders, absorbs the returns, and withdraws settled revenue before the next operating charge lands. To evaluate the strategies driving the year-end profit, we independently score six core capabilities: negotiation quality, fraud avoidance, cash flow and solvency, operational efficiency, operational execution, and long-horizon learning (§3.7). Next, we introduce the four layers of E-Commerce Bench in detail.

## 3.2 Agent Loop Layer

Three mechanisms sit between the model and the environment. The turn loop (Figure 4) bounds what one model call can do, and charges simulated minutes for every tool it uses. Every turn adds its tool traffic to the message list, and a year of calls overruns the window many times, so the context editor evicts the oldest turns first. The window is not the real limit either, since long contexts degrade well before they overflow (Liu et al., 2024; Bai et al., 2024). Because evicted text leaves nothing behind, the agent also gets a persistent memory system beyond the reach of any eviction pass.

![](images/f8f42768db044bdd1201a7e59735703d69f4b4b8700b835effd315dde8cf947a.jpg)  
Figure 4: The agent loop layer. One model call produces a batch of tool calls; the batch runs in order, and a timed tool advances the simulated clock inside its own call. The daily settlement fires whenever the clock reaches the next 08:00, whether a timed tool carried it there or the agent asked to skip to tomorrow. Tools read and write the hidden environment state, and the settlement writes back to it as well, posting that day's fees, sales, and returns. The day's news and any notices are appended to the tool responses. The transcript is left alone until it passes 120,000 tokens, and only then are its oldest groups dropped.

## 3.2.1 Turn-Based Loop and Simulated Time

Within a single turn, the model can request several tools, which then run one by one in the order given. Turns therefore range from a single balance check to a batch of a dozen tools. An episode runs to a cap of 4,000 turns (§4.1), or ends early once the model goes three turns without calling a tool. Every tool but the day jump costs simulated minutes: 10 for a balance check and 60 for opening a store (Table 3 in Appendix A), spent out of a 600-minute day that runs from 08:00 to 18:00. Looking something up draws on the same clock as acting on it. A message costs 30 minutes whether it goes to one supplier or to ten, so twenty of them fill a day.

Once the clock passes 18:00, it jumps to 08:00 the next morning, which wait\_for\_next\_day also does on request, costing no minutes but forfeiting the rest of the day. Every jump runs the same fixed sequence, charging operating and storage costs, turning the previous day's demand into orders, cancelling shipments that missed their two-day deadline, settling matured escrow, delivering arriving purchase orders, and updating reputation (§3.4). What the agent gets back after advancing to the next day is four thin signals: the day's news feed, a reminder once the bank balance no longer covers the day's operating cost, the current time, and a warning about token usage (Appendix A). Anything more precise costs a tool call and the minutes that come with it

## 3.2.2 Context Management

A single tokenizer counts each model's context to ensure all of them work against the same 128,000-token budget. A context eviction is triggered whenever the count reaches 120,000 tokens, taking 60,000 as its release target. Whole tool-call groups go, an assistant message and every tool response after it, oldest first (Figure 5). The system message, the first user turn, and the two newest groups are spared. The agent is told the policy verbatim, thresholds included, so preserving what matters in its memory is a choice it can make. Across the 90 evaluation episodes of §4, 1,495 evictions ran. How much history a model loses scales with its total tool traffic, the turns it takes times what each turn drags in, and the two factors trade off. Claude Opus 4.7 finishes the year in 432 turns and has about four windows' worth of transcript discarded along the way, while Gemini 3.5 Flash takes 2,628 turns and loses nearly nineteen windows' worth. Turns alone do not settle it. Kimi K2.6 takes 1,017 turns and loses the same three and a half windows as Claude Opus 4.7 at 432 turns, because it drags in 431 tokens a turn against Opus 4.7's 1,058, and GPT-5.5 loses 28 windows in half Gemini 3.5 Flash's turns at 2,759 a turn.

![](images/b407c7b461f12f14348143e18a2967fc2ae585207e4784c25426ecfd1e1e451c.jpg)  
Figure 5: How eviction works. A group is what the editor removes as a unit: one model reply together with every tool result that reply produced, so a turn that called six tools is still one group, and about fifty groups fill the window. A pass fires only once a group pushes the count past 120,000 tokens, then drops whole groups, oldest first, until 60,000 tokens are released or the clearable groups run out. The system message, the first user turn and the two newest groups are kept.

## 3.2.3 Persistent Agentic Memory

The memory store is a list of notes the agent keeps in the environment rather than in its own messages, so eviction never touches it. The store holds at most 20 entries, each a title and its content. The agent fully curates the store itself—creating, updating, and pruning entries as needed (Shinn et al., 2023; Packer et al., 2024; Zhong et al., 2024; Sumers et al., 2024). Reading the store back is cheap in minutes yet expensive in tokens, since returned entries re-enter the context and become evictable like everything else A capable agent is thus incentivized to preserve high-value strategic knowledge, such as past settlement prices or identifying deceptive suppliers.

## 3.3 Tool Layer

Eighteen tools cover a merchant's back office, in the seven groups Figure 3 draws, and Table 3 gives each one its minute cost and its effect. Every call but the day jump is billed in simulated minutes (§3.2.1) which makes the tool list a budget rather than a menu. Checking the bank balance, the warehouse, and one store's status costs 10 minutes each. A morning round of all three spends 30 minutes a day, more than 18 working days across the year. Only chatbox can create a purchase order, so every unit of inventory arrives through free-form text carrying fenced negotiate blocks (§3.5.2). Routine mail, supplier offers and fraudulent pitches (§3.5.4) all cost the same 30 minutes through that one tool.

## 3.4 Sales and Economy Engine

The sales and economy engine decides how much the agent sells and when it gets paid. Nothing about the market is sampled. A fixed formula computes how many units customers buy each day, and money moves between accounts on fixed delays. The simulator still draws a few random numbers, for instance when it turns a fractional unit of demand into a whole one, and those come from one seed shared by every run, so two episodes that take the same actions end the year with exactly the same money.

## 3.4.1 Multi-Store Setup

An agent trades through stores it opens itself. Twelve store types are on offer, each covering a different set of product categories, and at most four may trade at once, one per type (Appendix B.2). Opening one costs ¥500 and then ¥60 to ¥130 a day to run, the harder types costing more. Closing a store stops the daily charge, and stock left behind keeps accruing storage unless the agent liquidates it for 10% of purchase cost. Reopening later starts reputation over. A market lookup grades a category's margin low, moderate, or high, while the price a supplier will actually accept emerges only from bargaining (§3.5).

## 3.4.2 Three-Account Deferred Settlement

Money an agent has earned is not money it can spend. Cash sits in three places (Figure 6). The bank account pays every cost the moment it is incurred, a purchase order in full the moment a negotiation closes. A sale pays nothing at first and only creates a pending shipment. Once the agent ships it, the revenue enters escrow with the platform's commission already deducted. Nine days later that balance matures into the platform wallet, and only an explicit withdrawal moves it back to the bank account where costs are paid. A refund is taken out of the escrow batch its sale belongs to, so a return never touches spendable cash. One order shows the gap. A 40-unit lot of a mid-priced accessory costs ¥2,447 the day it is placed and returns its first cash on day 16. Ten consecutive days closing with a negative bank balance end the episode as bankrupt (Appendix C.7), so an agent selling at a profit while leaving its revenue in the wallet can still fail (§4).

![](images/8f4fdfc94356c6d5c062612d737f546c0102b201dad02cc03e3d050cf13cf73d.jpg)  
(b) Cash Cycle of One Order, 40 Units of One Real SKU

![](images/8c1063483a3740323e610a49b0c369c8d0cc80ae3e176521275ffedce7251fe4.jpg)  
Figure 6: Three-account deferred settlement. (a) Costs leave the bank the day they are incurred, while revenue passes through escrow and the platform wallet first. (b) One 40-unit order bought at the opening quote and resold at reference, followed as cash for 16 days.

## 3.4.3 Deterministic Multi-Factor Demand Model

How many units are sold is a product of named factors rather than a random draw. Demand for one SKU on one day starts from what its category sells in a store of that type. Six multipliers then act on that base: the response to the posted price, a weekend uplift, the promotion the agent joined, the month's seasonality, whatever the calendar holds that day, and the store's reputation. Two capacity terms trim the result, one for the category and one for the store, each a ceiling on how much demand that category or that store can absorb in a day. What survives is capped at the units actually on the shelf.

$$
\begin{array} { r l } & { \tilde { d } _ { i j t } = \beta _ { c \tau } \phi _ { f _ { c } } ( r _ { i j t } ; \eta _ { c } b _ { i t } ) \ w _ { t } m _ { i t } ( \delta _ { i t } ) \ \sigma _ { \tau , t } \varepsilon _ { \tau , t } \ \rho _ { i t } , \qquad r _ { i j t } = \frac { ( 1 - \delta _ { i t } ) p _ { i j t } } { v _ { j } } , } \\ & { \gamma _ { i j t } = \left\{ \begin{array} { l l } { \kappa _ { c } } & { \mathrm { i f } \ n _ { r } \ge 2 , } \\ { \kappa _ { c } + \sum _ { j ^ { \prime } \in \mathcal { C } } \tilde { d } _ { i j ^ { \prime } t } } & { \mathrm { i f } \ n _ { \tau } = 1 , } \end{array} \right. } \\ & { d _ { i j t } = \tilde { d } _ { i j t } \ \gamma _ { i j t } \frac { \kappa _ { \tau } } { \kappa _ { \tau } + \tilde { D } _ { i t } } , } \\ & { q _ { i j t } = \operatorname* { m i n } ( x _ { i j t } \ \lvert d _ { i j t } \rvert + 1 \lvert u _ { j t } < d _ { i j t } - \lvert d _ { i j t } \rvert ] ) . } \end{array}\tag{1}
$$

Indices run over stores i, SKUs $j ,$ dates t, store types τ and categories c, and Appendix C.2 defines every symbol alongside the seasonality tables of Appendix B. Price enters through one of four elasticity families carried per category (Gallego & van Ryzin, 1994). One of them, the quadratic family, is symmetric about the public reference price rather than monotone, so in the categories that carry it, a discount below reference loses demand instead of gaining it. The two capacity terms can then dominate everything the six multipliers did. For one accessory SKU on a promotion Saturday, they cut a compounded expectation of 128.2 units to the 9 actually sold, the category term alone taking a factor of 7.5 (Figure 7). How much they bite depends on how heavily the agent stocks that category and store, meaning that adding more homogenized SKUs cannibalizes demand rather than boosting aggregate sales.

(a) Elasticity Families, Median Real η  
![](images/995bd9ab3d866d356c622d8c4a4bf32627e4c6564f104c50d6ecffbfae595b68.jpg)

(b) Fashion Accessories SKU-Day  
![](images/40b880995f850043603c6d69ee2a71df1ce6d14cb68fc96d2e049fafb1da6862.jpg)  
Figure 7: The demand model. (a) The four price-elasticity families at their median parameter, solid without a promotion and dashed with the largest promotion boost. (b) The chain of multipliers that turns one SKU's price into units sold.

## 3.4.4 Fulfillment, Shipping Speed, and Returns

Every sale has to be shipped, and how it ships is a priced choice. Faster delivery costs more freight and brings back fewer returns, and a bulkier parcel costs more again. An order left unshipped past its two-day deadline is canceled outright, the units going back to the warehouse unsold and the cancellation counting against reputation. Returns are decided when goods ship and arrive days later, so a fulfillment decision costs money long after it is taken (Petersen & Kumar, 2009). Four channels set a unit's return probability, its category's natural rate, whether the supplier shipped defective stock, how far above reference the agent priced it, and the shipping speed chosen (Appendix C.4). A defective unit carries a return rate of at least 0.40 and at least double the natural rate, capped at 0.95. Pricing at 1.3 times reference multiplies the rate by 1.50, and slow shipping by a further 1.30. Every arriving return refunds the customer out of escrow and puts the unit back in the warehouse on a fresh storage clock. Two of the four channels answer to the agent, its pricing and its freight tier, so a model's return rate reads partly as a management outcome rather than only as a property of the catalog (§3.7).

## 3.4.5 Reputation System

A store's reputation multiplies its demand directly (§3.4.3), so service failures cost sales. Credit accrues when an order ships rather than when it sells, so an unshipped order earns nothing and draws the cancellation penalty as well. Returns and cancellations subtract from the same standing, weighted 0.6 and 1.0 (Resnick et al., 2000), so a missed deadline costs more than a change of mind. A store pinned at the floor receives 15% of the demand it would receive at the top, and the capacity terms return part of that loss once they bind (Appendix C.3).

## 3.4.6 Dynamic Event System

The calendar moves demand whether or not the agent is ready. Market events arrive on fixed dates, scaling demand per store type and stretching delivery lead times (Appendix B.3, Table 6 and Figure 15). Direction is rarely uniform, so a storm that lifts food cuts fashion and footwear. News of an event reaches the agent's feed as the event begins, while a promotion is announced a week ahead without its dates or its size. Joining reveals the size, and the year's first sale is already running when its announcement lands. The boost from joining peaks at a 30% discount, so a deeper cut adds demand only through the price response while margin falls on every unit. Suppliers exit on a count rather than a date, each retiring after a hidden number of filled orders, so a counterpart whose floor the agent has learned can be gone by the next re-order.

## 3.4.7 Daily Settlement and Processing Lifecycle

Costs are charged before any revenue is recognized, so every morning tests solvency at the day's low point. Sales are then computed for the previous calendar date against the inventory and prices the agent left standing overnight, so the agent steers a system that lags it by a day. Refunds leave escrow before matured escrow settles, so a return can empty the batch that was about to pay out. Appendix C.1 lists all thirteen steps with the state each one touches.

## 3.5 Deterministic Negotiation Engine

The negotiation engine sets what the agent pays for stock. Every unit it sells has to be bought from a named supplier, one chatbox message at a time, at a price the two of them argue over. Roughly a quarter of the roster takes the money instead of earning it (§3.5.4), so a counterpart has to be chosen as well as bargained with.

## 3.5.1 Bargaining Policy Preliminaries

Sourcing one SKU begins with a message to one supplier and ends in a price or in nothing. Neither side sees the other's limit. A supplier never reveals the floor below which it will not sell, so each purchase is a bilateral bargaining problem under incomplete information (Harsanyi, 1967; Chatterjee & Samuelson, 1983; Rubinstein, 1982). One such cycle is a session $\boldsymbol { z } = ( s , j , c )$ , the c-th between supplier s and SKU j. No rule closes a session at a fixed round. A deadline scale sets the time pressure instead, and past its midpoint a supplier can walk away from an offer under its floor (Appendix D.1).

One pair from the evaluation fixes the scale. Ridge Express quotes a single SKU at ¥149.81, and the agent resells against a market reference of ¥199.75. Beneath both lies a floor of ¥101.27 the agent never sees, leaving ¥98.48 for the two sides to divide. Figure 8 follows one such session offer by offer in panel (a), and the nine deals the pair produced across the year in panel (b).

Two prices bracket what a session can be worth. The agent resells against a public market reference price vj (§3.4), while supplier s holds a private floor $c _ { j } ^ { s } .$ Bargaining runs in the gap between the two, the zone of possible agreement (ZOPA), so a deal at unit price $p _ { z }$ for $q _ { z }$ units hands the agent $\left( v _ { j } - p _ { z } \right) q _ { z }$ of it. How wide that gap is, how badly the supplier needs the sale and how hard it will push are drawn once and stay hidden. The agent bargains without knowing how much room it has.

Sessions are divided by counterpart type, one term of the objective per side. Writing $\mathcal { Z } _ { \mathcal { G } }$ for those concluded with the honest suppliers G and $\mathcal { Z } _ { B }$ for those concluded with the fraudulent ones B (§3.5.4), the year-end asset goal of §3.1 becomes a procurement objective.

$$
\underbrace { \sum _ { \tau } \sum _ { \sigma } \mathbf { 1 } \left[ f _ { z } \neq \bot \right] \left( v _ { j ( z ) } - p _ { z } \right) q _ { z } } _ { \mathrm { s u r p l u s ~ e x t r a c t e d ~ f r o m ~ h o n e s t ~ s u p p l i e r s } } - \underbrace { \sum _ { z \in \mathcal { Z } _ { B } } \mathbf { 1 } [ f _ { z } \neq \bot ] \ell _ { z } } _ { \mathrm { l o s s e s ~ p a i d ~ t o ~ f r a u d u l e n t ~ s u p p l i e r s } } .\tag{2}
$$

Here $f _ { z }$ is the session outcome, ⊥ a disagreement worth nothing, and $\ell _ { z }$ what a fraudulent deal costs in the end, whether overpayment, fees, short shipment or defective stock.

Two competencies follow, and they pull against each other. Winning surplus from an honest supplier takes firmness; refusing a fraudulent one takes walking away, and the firmness that wins a good price is also what produces impasse (Raiffa, 1982). Both are paid in the cash that sets the primary score.

## 3.5.2 Two-Layer Supplier Design

An outcome difference says something about the agent only if every model met the same counterpart economics in every run. A supplier played entirely by an LLM breaks that twice over. Sampling moves its prices from run to run, and it stays persuadable, so an eloquent enough agent can talk it below its own floor and turn the benchmark into a jailbreaking contest. Every supplier therefore splits into two layers.

Layer 1, the Deterministic Negotiation Kernel. Every price, concession and accept or walk-away decision comes from a deterministic kernel rather than from a model. The kernel builds on the counterpart policy of Zhang et al. (2026a), reimplemented independently in a seller-only role and re-parameterized from the data layer (Appendix D). Acceptance and walk-away are stochastic (Baarslag et al., 2014), and a latent cue carries sentiment and posture (Appendix D.1). Counter-offers answer the agent's own concession speed rather than the clock (Faratin et al., 1998), so an agent that concedes fast meets a frmer counterpart. Fast concession carries the same disadvantage in human bargaining (Tey et al., 2021).

One kernel serves one (supplier, SKU) pair. Its reservation price and opening quote come from the data layer, so both are properties of the SKU rather than draws. Its seed makes an identical agent trajectory meet an identical supplier response (Appendix D.2). Two sources of run-to-run variance survive. The Layer 2 renderer is an LLM sampled at its provider's default decoding, and the delivery days a supplier's catalog advertises are drawn outside the episode seed, so they can move between runs. Prices and accept or walk decisions cannot move, but the narrative a model reads to detect fraud does, so the fraud axis of §4.3.2 carries renderer sampling noise on top of policy sampling noise.

Honest suppliers draw from six behavior templates. The templates differ in how sharply the kernel answers the agent's concession speed and in how much its cues give away. Which supplier gets which is fixed in the released roster, so the difficulty mix is set rather than drawn per run (Appendix D.3). Neither the template nor the honest or fraudulent label is visible to the agent.

![](images/f7e78062c48a06c8f70b1df13981364ee57ec3e3be0dc18e733d3a14dc0136c7.jpg)

![](images/852114eb40eb1aac9b2bf0bd210f150f25ca659329cf03dd35d4c716d9c8a003.jpg)

![](images/704be8227f5cf345c37e9fe1dd34bc65be16fe559758c3b6b0b9d1322a3cb6e0.jpg)  
Figure 8: Negotiation at two timescales, from real episodes, each panel on its own price axis and the cost floor hidden from the agent. (a) One session offer by offer, the agent bidding up and the kernel conceding down until the agent accepts its standing quote. (b) The same (supplier, SKU) pair all year, where that session ends a run of near-floor deals and the price never recovers, the hatched wedge being the overpayment AnchorRegret averages. (c) Another agent instead pushes lower at nearly every restock.

Layer 2, the NPC renderer. A second LLM turns each committed kernel decision into supplier chat. The renderer has to relay the exact price and decision and takes only tone from the kernel's sentiment cue, which keeps the dialogue rich and economically inert. Compliance does not rest on that instruction. An accept is refused unless its price matches the kernel's standing quote to within ±0.005, so a renderer that misstated a price could never make it a transaction (Appendix D.4). In the other direction, a deterministic parser reads the agent's own offer, accept and reject actions out of the same free-form message. One economic action travels in the prose instead, enrolment in a supplier's membership program, which an intent classifier recognizes from the agent's own words and bills at ¥1,000 (Appendix D.5). Prices carry no signal about fraud by design, so on the pre-deal class the narrative holds the decisive evidence, and no fixed template could mount it. Figure 20 prints one session end to end, every supplier line beside the kernel decision it had to relay.

## 3.5.3 Multi-Round, Cross-Supplier, Repeated Bargaining

One session is usually the unit of evaluation in existing negotiation benchmarks, and the kernel as described so far governs exactly one. Across a full year the game becomes repeated, which asks whether an agent turns what one encounter taught it about a counterpart into a better price at the next. Inferring a private reservation value from past encounters (Baarslag et al., 2016) therefore becomes part of the task rather than an assumption. Closed deals are not re-surfaced, so an agent keeps its own price history or does without it.

Once a session over $( s , j )$ concludes, the next offer on that pair opens a fresh cycle under a new kernel instance and a new seed, carrying the same persistent type and reservation price. A seller-side kernel never accepts below that reservation, so the best price won so far is an upper bound on $c _ { j } ^ { s }$ that only tightens. The competence it rewards is to anchor each reopening at or below that historical best. Calling the rule optimal would go too far, since against a pre-deal fraudulent counterpart the historical best tightens onto a floor inflated on purpose (§3.5.4).

Learning runs on both sides of the table. Each supplier's deal history goes into its prompt, so suppliers “remember" what they sold before. Several honest suppliers serve every category alongside a few fraudulent ones, so the agent can also arbitrate across counterparts, probing one supplier's floor and holding it as an outside option against another. Single-session skill and year-long accumulation then separate cleanly, the first measured in per-session surplus, the second in how fast the price for a repeatedly sourced SKU walks down toward the floor (§3.5.5).

Every session costs twice over besides. Bargaining spends minutes out of the working day (§3.2.1), and an agreement spends cash out of a bank balance that refills only when escrow settles (§3.4.2). A session excellent in surplus terms can still be the wrong one to have opened.

## 3.5.4 Adversarial Supplier Fraud

Of the 576 suppliers on the roster, 152 are fraudulent. Each runs the hardball kernel preset under one of five scripted scam overlays that the renderer performs. The five split into two classes by when the damage lands (Appendix D.5). No single observation gives a fraudulent counterpart away. Opening quotes are uninformative by construction (§3.5.2), and the behavior family stays latent and is shared with honest suppliers. What does separate the two populations is the rate of concession, since fraudulent kernels hold out longer than the hardest honest preset. That signal lives only in the shape of a multi-round exchange, so an agent that accepts early gives up its one bargaining-side cue and has to read the narrative instead.

Pre-deal scams make their money inside the negotiation. Their kernels carry a reservation price inflated toward 1.5× the honest cost floor, so any deal overpays what an honest competitor would have charged. The inflated floor still sits inside the range an honest supplier might quote, so only the narrative gives it away, offering membership fees, discounts that never arrive, or manufactured urgency.

Post-deal scams keep the honest cost floor and quote from that same range, then defect at fulfillment. A short delivery brings 60% to 70% of the quantity paid for, and defective stock returns at 0.40 or double the SKU's natural rate, whichever is higher, capped at 0.95 (§3.4). The defective rate is blended across the SKU's pooled stock, so interleaving honest and fraudulent deliveries dilutes the evidence and no observed return rate points back to a culprit. Scoring therefore counts the money actually lost to these counterparts rather than judging intent (§3.5.5), which keeps the axis on the cash of Equation (2).

## 3.5.5 Negotiation Metrics

Both terms of Equation (2) are scored as the fraction of the bargaining range an agent captures, standard practice in automated negotiation (Raiffa, 1982; Baarslag et al., 2013). The metrics build on the surplus efficiency of Zhang et al. (2026a), re-based from sampled ZOPA geometry onto data-grounded floors, and Appendix E.1 carries every formal definition. Agreement is close to universal on the honest sessions a model concludes, so CSE+ carries the headline, the share of a pair's bargaining range that a closed deal captures. Beside it %Oracle compares the same surplus against closing every honest deal at the supplier's floor, and the agreement rate stays in Appendix E.1. Neither headline metric weights a deal by quantity, and both are rates over a session set the agent chose itself, so they measure per-deal bargaining skill rather than procurement cash saved. Two models compare on them only if their portfolios carry similar bargaining ranges, which §4.3.1 checks by reporting deal volume beside the rate. Conclusion is the operative word. A session abandoned mid-bargain enters no denominator and costs nothing in any of the three metrics, so §4.3.1 reports abandonment counts beside them rather than folding them into a rate. Fraud avoidance is reported as BadSpend% rather than as an agreement rate over fraudulent sessions, because the agent also chooses whom to contact at all. A session-rate denominator would reward never looking, and cash is in any case the quantity a scam actually costs.

Sequential price discipline Over 365 days, the same supplier is asked for the same SKU many times, which raises a question no session-level metric answers. Does an agent's next order benefit from the price it already won? Holding an anchor like that has documented value in human bargaining (Petrowsky et al., 2025). The measure looks at every honest (supplier, SKU) pair traded more than once. Each agreed price is expressed as its position πz inside that pair's bargaining range, then scored after the first deal against the best the agent had already won there. AnchorRegret averages the resulting overpayment. AnchorRatio divides that average by a null which holds each pair's prices fixed and reshuffles only the order they arrived in. Significance travels separately in the permutation z-score zanchor.

The null assumes a pair's cycles are equally hard, and that assumption is checkable rather than asserted. Two of the three quantities that set that difficulty are constants of the SKU, and the archive records the cost floor and the nominal wholesale reference unchanged across all 8,647 re-orders. The quote the kernel actually opens with is redrawn each cycle and the archive does not keep it, so a rising $\pi _ { z }$ is read against a fixed floor rather than against a fixed opening. The measure reports an outcome and not its cause, and §4.3.6 takes up which mechanism produces it. Figure 8 shows both outcomes on real trajectories, one agent giving back a price it had already won while another walks its price down the range all year.

Three limits bound the scope. Only honest pairs count, since a pre-deal fraudulent floor is inflated by construction (§3.5.4). Each re-order counts equally regardless of order size, so the measure captures per-unit discipline and not its cash consequence. No observations come from an episode that ends early.

## 3.6 Data Layer

Every number E-Commerce Bench runs on comes from a real e-commerce platform, either desensitized straight off it or synthesized by hand against an analysis of its data logs, the supplier price floors of §3.5 and the demand parameters of §3.4 among them.

## 3.6.1 Catalog and Suppliers

The environment ships 6,886 products in 60 categories, served by 576 suppliers, 424 of them honest and 152 fraudulent. The 12 store types span 9 categories down to 1, and the calendar fixes 10 market events and 8 promotions. Desensitization maps the products, the categories, the per-category return-rate tiers, the monthly-sales bands and the holiday calendar straight off a real e-commerce platform. Synthesis from real logs supplies the rest, namely the four price-elasticity families, seasonality curves, reputation parameters and supplier attributes. Promotion names are fictional, since a model that recognizes China's largest shopping festival could price from memory rather than from evidence (Zhu et al., 2025). Fraud reaches every product line rather than concentrating in a few. Each supplier serves exactly one category, no category holds fewer than 7 honest and 2 fraudulent, and coverage stays flat across catalog depth (Figure 13d). Every fraudulent one runs a scam of §3.5.4. The supplier list is fixed at release rather than redrawn per episode, and Appendix B lists all 60 categories with their parameters.

## 3.6.2 Systematic Information Asymmetry

An agent sees the catalog and the outcomes of its own actions, never the demand and cost-floor parameters that decide what an action is worth. A market lookup answers at one of three levels: the coarsest naming every store type and the finest one category's sales band (Table 3), each substituting a label for the quantity behind it. A profit-potential grade replaces demand capacity, a margin class the wholesale ratio and a sentence about returns the per-SKU rate. Supplier lookup withholds most, naming no type and ranking nothing (§3.5.2). Pre-deal scams show in the narrative from the opening quote (§3.5.4). Post-deal ones are written to read like ordinary suppliers, so only the goods that arrive expose them, and the cash has left by then. Table 7 in Appendix B sets each observable against the quantity it withholds and prices the discovery. Portfolio decisions therefore rest on inference, and hidden parameters differ widely in how fast they can be learned. The best price already won from a supplier bounds its floor from above (§3.5.3), and a SKU's natural return rate becomes readable once its stock has moved. An elasticity parameter stays hidden all year, since four unprinted factors multiply the same observation. A store type's earning ceiling shows only after one has been operated for a year. Exploration and operation both cost time (§3.2), so the year rewards allocating discovery rather than exhausting it.

## 3.7 Evaluation and Metrics

The primary score is end-of-year total assets, the sum of the bank account, the platform wallet, and unsettled escrow, reported as the asset multiplier over the ¥100,000 opening stake. Bankruptcy ends a run before the horizon (§3.4.2). Truncated runs enter the average rather than being dropped, so a five-run mean can mix them with full years. Six capabilities are measured around the year-end assets:

(1) Bargaining is scored on how far down the agent pushes a price. CSE+ ranks negotiation quality, how much of the distance from the public reference price down to the supplier's cost floor a closed honest deal covers, where 1 means buying at the floor and 0 means paying the reference price, averaged over sessions with no weight for order size. Recorded around it are SE+, %Oracle, AGR+, rounds to deal, deals closed, and both surplus and spending broken out by the supplier's behavior template (§E.1). (2) Fraud avoidance follows the cash rather than the conversation. Ranking runs on BadSpend%, the share of order spending that cleared into a fraudulent supplier's account, low being the better direction. We also record the fraud agreement rate (the percentage of fraudulent sessions ending in a deal), alongside diagnostics covering losses across the five scams, membership fees paid, and fraudulent suppliers contacted or ordered from (§E.2). (3) Solvency is judged by how much of its own best position a year gave back. Peak drawdown over peak total assets ranks cash flow and solvency, since the archive records that fall in absolute ¥, and a business ten times the size books ten times the fall for the same discipline. The unnormalized fall, the bankrupt runs, the overdrawn days, the deepest bank balance, the idle platform wallet, and the largest warehouse held round out the dimension (§E.3). (4) Operational efficiency prices a year of profit against the calls that produced it. Mean profit over mean tool calls orders the models. Sitting next to it are the call count, its split over eight activity bands, the turns, the eviction passes the transcript needed, and the memory-store traffic (§E.4). (5) Operations execution is ranked by the controllable return rate—the excess returns driven by the model's pricing above the baseline (accrued at dispatch; negative for discounted prices), excluding fraud-induced defects. Supporting diagnostics include realized returns, refund and freight losses, on-time delivery rates, orders lost to the two-day deadline, tier mix, and store counts (§E.5) (6) Learning is read from what an agent does every time after the first that it buys the same item from the same supplier, which over a year is usually several times. AnchorRatio ranks learning over the horizon by measuring how much each repeat order overpays against the cheapest that agent had yet paid that supplier for that item. Beside it lie the same overpayment before its permutation null, AnchorRegret, the NewLow rate, the combined $z _ { \mathrm { a n c h o r } }$ , the repeat-order count and three half-year comparisons (§E.6).

Table 2: End-of-year leaderboard, 18 models, five episodes each, sorted by mean final assets within tier. Final assets carries the primary score, and the six columns after the spread rank one evaluation dimension each, CSE+ for negotiation quality, BadSpend% for fraud avoidance, drawdown over peak for cash flow and solvency, yuan per tool call for operational efficiency, the controllable return rate for operations execution, and AnchorRatio for learning over the horizon, with the formulas in §E.1 through §E.6. Arrows mark the direction of the better outcome, and a tinted cell holds the extreme of a scored column. Tool calls, turns, and bankrupt runs record what an episode spent rather than scoring it, and bankrupt episodes stay pooled into every mean, which is what the spread column beside the assets reports. †GPT-5.5's spread is a population estimate, ¥689k on the sample estimator. †GPT-5.5's AnchorRatio averages three episodes; its two bankrupt runs opening no repeat order with a supplier it had already dealt with.
<table><tr><td>Model</td><td>Final assets ¥k, mean</td><td>std CSE+ ¥k</td><td>↑</td><td>BadSpend% ↓</td><td>Drawdown /peak↓</td><td>￥ per tool call ↑</td><td>Controllable return, pp ↓</td><td>AnchorRatio ↓</td><td>Tool calls</td><td>Turns</td><td>Bankrupt runs</td></tr><tr><td>Proprietary</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.6 Sol (max)</td><td>1,431</td><td>314</td><td>0.672</td><td>18.48</td><td>0.278</td><td>363</td><td>4.46</td><td>1.217</td><td>3,668</td><td>1,367</td><td>0/5</td></tr><tr><td>Fable5 (max)</td><td>805</td><td>188</td><td>0.772</td><td>3.46</td><td>0.141</td><td>479</td><td>0.22</td><td>1.573</td><td>1,469</td><td>945</td><td>0/5</td></tr><tr><td>GPT-5.5</td><td>702</td><td>616†</td><td>0.700</td><td>16.59</td><td>0.591</td><td>192</td><td>0.90</td><td>1.043‡</td><td>3,143</td><td>1,306</td><td>2/5</td></tr><tr><td>Claude Opus 4.8 (max)</td><td>498</td><td>231</td><td>0.662</td><td>5.41</td><td>0.130</td><td>266</td><td>0.20</td><td>1.309</td><td>1,497</td><td>812</td><td>0/5</td></tr><tr><td>Claude Opus 4.7 (max)</td><td>259</td><td>111</td><td>0.811</td><td>0.12</td><td>0.189</td><td>156</td><td>1.75</td><td>1.421</td><td>1,023</td><td>432</td><td>0/5</td></tr><tr><td>Claude Opus 4.6 (max)</td><td>258</td><td>266</td><td>0.649</td><td>14.98</td><td>0.587</td><td>129</td><td>0.76</td><td>1.268</td><td>1,221</td><td>788</td><td>2/5</td></tr><tr><td>Gemini 3.5 Flash</td><td>190</td><td>79</td><td>0.777</td><td>2.13</td><td>0.450</td><td>36</td><td>0.38</td><td>0.918</td><td>2,508</td><td>2,628</td><td>0/5</td></tr><tr><td>Gemini 3.1 Pro</td><td>130</td><td>130</td><td>0.660</td><td>12.58</td><td>0.688</td><td>19</td><td>2.03</td><td>1.376</td><td>1,605</td><td>1,141</td><td>2/5</td></tr><tr><td>Open-weight</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.8-Max-Preview</td><td>416</td><td>111</td><td>0.713</td><td>6.13</td><td>0.242</td><td>173</td><td>0.38</td><td>0.834</td><td>1,826</td><td>962</td><td>0/5</td></tr><tr><td>GLM 5.2 (high)</td><td>301</td><td>124</td><td>0.693</td><td>1.99</td><td>0.127</td><td>137</td><td>0.59</td><td>1.434</td><td>1,467</td><td>672</td><td>0/5</td></tr><tr><td>Kimi K3</td><td>265</td><td>110</td><td>0.632</td><td>5.87</td><td>0.247</td><td>123</td><td>2.01</td><td>1.507</td><td>1,334</td><td>878</td><td>0/5</td></tr><tr><td>GLM 5.1</td><td>226</td><td>192</td><td>0.662</td><td>6.49</td><td>0.231</td><td>94</td><td>0.09</td><td>1.557</td><td>1,333</td><td>852</td><td>0/5</td></tr><tr><td>DeepSeek-V4-Pro-Preview</td><td>190</td><td>100</td><td>0.654</td><td>8.99</td><td>0.166</td><td>68</td><td>1.54</td><td>1.235</td><td>1,337</td><td>678</td><td>0/5</td></tr><tr><td>(max) Qwen3.7-Max</td><td>165</td><td>134</td><td>0.616</td><td>9.85</td><td>0.433</td><td>59</td><td>-0.05</td><td>1.433</td><td>1,108</td><td>782</td><td>0/5</td></tr><tr><td>GLM 5.2 (max)</td><td>115</td><td>71</td><td>0.632</td><td>19.68</td><td>0.360</td><td>10</td><td>2.24</td><td>1.362</td><td>1,430</td><td>931</td><td>0/5</td></tr><tr><td>Kimi K2.6</td><td>70</td><td>14</td><td>0.596</td><td>17.22</td><td>0.377</td><td>-27</td><td>0.42</td><td>1.548</td><td>1,087</td><td>1,017</td><td>0/5</td></tr><tr><td>Qwen3.6-Plus</td><td>47</td><td>28</td><td>0.625</td><td>10.68</td><td>0.583</td><td>-40</td><td>3.03</td><td>1.333</td><td>1,325</td><td>1,023</td><td>0/5</td></tr><tr><td>Qwen3.5-Plus</td><td>1.1</td><td>11</td><td>0.625</td><td>20.11</td><td>0.979</td><td>-92</td><td>5.88</td><td>1.860</td><td>1,076</td><td>800</td><td>4/5</td></tr></table>

## 4 Experimental Results

## 4.1 Experimental Setup

Eighteen models ran the 365-day task, eight proprietary and ten open-weight by our own assignment, ranked within tier by mean total assets (Table 10). Each model received five independent episodes, 90 in all, under one fixed world. Requests use each provider's default settings, with no temperature or top\_p sent and thinking force-enabled wherever a family exposes it (Table 10). Five episodes is a small sample, so every mean in Table 2 is printed with its standard deviation (Miller, 2024; Kapoor et al., 2025). Otherwise, all models shared the same 18 tools (Table 3), context window and group-wise eviction (§3.2.2), and the memory store outside it (§3.2.3). The longest episode took 3,599 turns of the 4,000 allowed, and all 90 ended on the simulated calendar or in bankruptcy rather than on that cap. One renderer voices every supplier for every model, so wording varies between episodes while the economics it relays cannot (§3.5.2). Scoring reduces the year to one number, the asset multiplier, end-of-year total assets over the ¥100,000 stake (Appendix E). A bankrupt episode stops on the tenth consecutive negative bank day and still enters the average.

## 4.2 Overall Performance

Starting from the same stake in the same simulated year, the models finish orders of magnitude apart. A factor of 1,264 separates GPT-5.6 Sol at the top from Qwen3.5-Plus at the bottom (Table 2, Figure 1). The environment is deterministic given the action sequence (§4.1), so that range belongs to the agents rather than to resampled demand. Proprietary systems take the top four places (Table 2). Qwen3.8-Max-Preview leads the open-weight tier and places fifth overall. Inside the Qwen family each release earns more than the one before it, and Qwen3.8-Max-Preview earns the most. The 3.6 to 3.7 step is worth noting, since Qwen3.7-Max swings from ¥6,290 to ¥358,790 across its five years, and its worst year finishes below every Qwen3.6-Plus year. Strong models go bankrupt too. Ten of the ninety episodes ended insolvent, six of them under closed-source models. GPT-5.5, Claude Opus 4.6, Gemini 3.1 Pro and Qwen3.5-Plus account for all ten, and the same four surrender more than half of peak total assets (§4.3.3). Ordering by assets recovers only part of the skill behind it (Figure 2). Section 4.3 takes the axes one at a time, and Appendix F carries the per-model table of every dimension.

![](images/c989f078378b8656a2cafc1523ec0bb3b0214a8e1a1527ef2cd3627af7ca15ac.jpg)

![](images/c3dd6ca31fe511dbc751ec65371a93698f4c0057619e984c10ffb4a401163abf.jpg)

![](images/0f5a667b5cd813edfb696268ef7e8342f9f988ba689384758052c79d00e473f4.jpg)  
Figure 9: Capability profiles for six of the 18 models. The primary score and the six dimensions of this section run clockwise from the top, profit as mean end-of-year total assets, negotiation as CSE+, fraud avoidance as BadSpend%, solvency as peak drawdown over peak total assets, efficiency as profit per tool call, execution as the controllable return rate and learning as AnchorRatio. Fraud avoidance, solvency, execution and learning are sign-flipped so that higher is better on every axis. Each axis is min-max normalized over the 18 model means, so 1 marks the best mean and 0 the worst, with whiskers over the five episodes and a dashed polygon at the median. Efficiency carries no episode spread, being a ratio of 5-run means, and the archived AnchorRatio spread is a population std over a varying number of episodes, so the learning axis draws no whisker either. No profile fills the polygon.

## 4.3 Multi-Dimensional Analysis

Six dimensions pull the year apart, each ranked by one recorded measure and defined in §3.7, with the formulas, field names, and caveats in one appendix subsection per dimension (§E.1 to §E.6). Figure 9 sets the primary score beside the six axes for six models, and Appendix F.8 carries all 18.

## 4.3.1 Negotiation Quality

No model bargains prices down to the supplier's cost floor. Claude Opus 4.7 keeps the most of each bargaining range at 0.811 and Kimi K2.6 the least at 0.596, with the per-model values in Table 11. Accepting the opening quote already scores 0.50, so the field lands 0.10 to 0.31 above what taking the first offer would have cost it (Appendix D.2). Which counterpart a model draws matters less than which model it is, since the field spreads 0.051 across the six behavior templates against 0.204 across models, though the per-model ordering inside a column still moves by as much as thirteen places (Figure 23). Agreement rates sit at the top of their scale and separate nobody. Of the 12,060 concluded honest sessions, 73 end without a deal, and 72 of those are the agent's own rejection rather than a supplier walking away. Every model settles at least 96% of the honest sessions it concludes, so the ranking rests on how much a deal captures rather than on how often one happens (Appendix E.1). The rate also leaves out the 10.9% of sessions a model opened mid-year and never came back to. Agents buy least from the counterparts that concede least. The hardest of the six honest behavior templates (Appendix D.3) covers 48 of the 424 honest names on the supplier list, and those 48 take 4.8% of the ¥55.1M of honest procurement, against at least 14% for each of the five easier ones (Figure 24). Buying the same item from the same honest supplier again costs more than a reshuffle of the prices already won would have, for 16 of the 18 models, and §4.3.6 takes that up with the rest of the year's trends.

## 4.3.2 Fraud Avoidance

Every model sends part of its procurement budget to suppliers that defraud it. Spending on them spans more than 160-fold, cleanest at Claude Opus 4.7 and heaviest at Qwen3.5-Plus (Figure 25), and none spends as though it never screened. BadSpend% counts money rather than refusals (§E.2), unlike a refusal rate on scripted solicitations (Yang et al., 2025), so declining an offer the agent never sought earns no credit. A share and a sum of money rank the models differently, because procurement budgets differ as widely as the shares do. Qwen3.5-Plus gives up the largest share, yet GPT-5.6 Sol hands over fifty times the cash on a smaller one (Table 12). Both readings rest on a handful of orders, and across five episodes the standard deviation of the share exceeds its own mean for 15 of the 18 models. Losing money to fraud does not cost a model its year. The cleanest buyer finishes mid-pack on assets and the profit leader is 16th of 18 on BadSpend% (§4.2), because a pre-deal scam inflates a price that still sits inside the honest range and a post-deal scam delivers most of the goods (§3.5.4). Most of the money disappears after a deal is signed. Defective lots take 53.4% of everything reaching fraudulent suppliers and short shipments the next largest slice (Figure 25), while membership-fee scams almost never land, three payments in the whole evaluation. Models separate when they order, not when they choose whom to approach. Fraudulent suppliers take close to their share of the supplier list in every model's outreach (§3.5.2), and what differs is how often an approach becomes an order, from 4.0% for Claude Opus 4.7 to 31.7% for GPT-5.6 Sol (Figure 26, Appendix F.3). A negotiation that reaches its end almost always closes a deal (§4.3.1), so the gap lies in the approaches a model abandons.

## 4.3.3 Cash Flow and Solvency

Money leaves the bank account weeks before it returns. Revenue waits nine days in escrow, then waits again for a withdrawal (§3.4.2). No episode sees cash in the wallet before day 15, and 49 of the 90 bottom out in January. Two GPT-5.5 episodes went bankrupt on day 17, before a single yuan of revenue had settled. Returns claw more money back out of escrow (§4.3.5). A fall recorded in yuan grows with the balance at risk, and the largest of the field belongs to GPT-5.6 Sol, which also earned the most (Appendix E.3). Dividing each fall by the peak total assets of its own episode separates the field on risk and the five models that surrendered more than half of peak include all four that ever went bankrupt (Figure 27a, Table 13). Running the bank account below zero was rare and usually fatal. Seventeen of the 90 episodes went negative at a morning check across 177 such days, and ten of them went bankrupt. A run ends only after ten consecutive overdrawn mornings, so one Claude Opus 4.6 episode absorbed fifteen scattered overdrafts and survived. On 66 of those days the wallet already held the shortfall (Figure 27b). No tool reports the streak (§3.2), so an agent cannot count the mornings it has failed. A large cash buffer buys survival and caps growth. Three models kept the bank above ¥15,000 in every episode and finished sixth, tenth and eleventh, while the two biggest earners held the thinnest average buffers of the fourteen solvent models. Qwen3.8-Max-Preview leads the open-weight field on a mid-field buffer, its bank above zero all year.

## 4.3.4 Operational Efficiency

Fable5 earns more from a tool call than the leader does. Sitting second on total assets, it turns ¥479 of profit out of each call against GPT-5.6 Sol's ¥363, on 59.9% fewer calls (Table 14). Three models finish under their opening stake, so their calls average a loss over the year, ¥92 a call at the worst. No fee inside the economy charges for a call, so the bill falls on whoever deploys the agent and the score stays blind to it (Kapoor et al., 2025). The Berkeley function-calling leaderboard reports that bill directly, printing dollar cost and latency beside accuracy (Patil et al., 2025). Where the calls go describes a policy better than their number does (Figure 10b). Shipping, advancing the day, withdrawing settled revenue and publishing inventory absorb 71.8% of all 18 models' calls, a daily loop rather than a plan. Reading state back takes another 14.7% and supplier conversation 6.0%, so the one channel that can change what an item costs draws less than a fifteenth of the effort. Pricing barely registers in the call budget. One call can revise many prices at once, yet 0.66% of all calls do so. Nine of the 18 models average under three such calls a year, leaving their listed prices to stand while season, promotion and reputation move demand underneath them. The memory store draws almost no traffic. Notes the eviction pass cannot reach (§3.2.3) take 2.1% of the field's calls. Claude Opus 4.8 gives it the largest share of its own budget, 4.7% at 69.6 calls a year against 20 slots, while GPT-5.5 makes the most calls outright at 135.2 and Qwen3.5-Plus wrote to it in none of its five episodes. Twenty notes revised a few dozen times is what a year of trading leaves behind.

![](images/a08c8925ff0d39dabd0d338bab5a501f6ff7f26b7cc2733c79fb893f8ba66ebf.jpg)

(b) Where the Budget Goes  
![](images/526e6248ce5584fd4a95e7026dc5199a762194513b99e5197759541b1b15d823.jpg)  
share of that model's tool calls (%)  
Figure 10: (a) Profit per tool call for the 18 models, best first, mean total assets less the ¥100,000 stake divided by mean tool calls per episode. Bankrupt episodes are pooled into both means, and the three models that finish under the stake sit left of zero. (b) Where six models spent their calls, over eight groups of the 18 tools, scaled within each model.

## 4.3.5 Operations Execution

The controllable return rate counts the percentage points a model's own pricing adds above the natural rate of the goods it chose to sell, and it is the one return channel this axis charges to the agent. The points run from -0.05 for Qwen3.7-Max to 5.88 for Qwen3.5-Plus (Table 15). The natural rate underneath is larger, 5.31 to 21.25 points, and it follows what a model chose to sell rather than how it priced. Twenty-six of the 90 episodes carry a negative rate, priced under the reference. Added returns are worth buying when the price that causes them pays for them. GPT-5.6 Sol books 4.46 points and the year's largest refund bill, ¥1.62M, and finishes first on assets. Qwen3.5-Plus adds more return points than any other model and finishes last. Freight speed is chosen once and left alone. The archive holds 92,182 fast dispatches beside 171,642 standard and exactly one slow, and 72 of the 88 episodes that sold anything put at least 98% of shipments in one tier. Fast doubles the freight bill and cuts the return draw by a quarter, yet nine models swing over 30 points in fast share between their own episodes, so the tier reads as sampling, not policy. Store openings order nothing. Cumulative openings run from 3.0 for two Qwen models to 9.0 for GPT-5.6 Sol, each charging the ¥500 setup fee again and restarting a reopened type's reputation at the floor. GLM 5.2 (high) opened exactly four every episode and placed sixth, while Gemini 3.5 Flash opened 13 in one episode and placed twelfth, and neither is a profitable habit.

## 4.3.6 Learning over the Horizon

Two of the 18 models pay less on a repeat order than a reshuffling of the prices they already won from that supplier would have cost. Qwen3.8-Max-Preview scores 0.834 on AnchorRatio and Gemini 3.5 Flash 0.918, where 1.0 means an agent's price ordering is indistinguishable from a random ordering of its own prices. The field median is 1.369, and Qwen3.5-Plus reaches 1.860 (Table 16). Fifteen models miss their own permutation null by more than two standard deviations in the expensive direction, which is what an unrevised anchor on an early price produces (Lou & Sun, 2026). GPT-5.5 contributes three episodes instead of five; its two bankrupt runs never buy the same item twice from the same supplier. A second reading of the same year runs the same way, with sixteen of the 18 models capturing less surplus in the second half of an episode than in the first. The surplus half-lift splits an episode's honest sessions at the median day and differences the two means (§E.6). Three mechanisms would hold that surplus trend flat, and the evaluation separates none of them. Eviction destroys the tool result that carried a price already won (§3.2.2). The memory store that survives eviction is barely used (§4.3.4). And no gradient reaches inside an episode, so improvement must come from an agent reasoning over its own record (Shinn et al., 2023). Qwen3.8-Max-Preview moves against the field, alone holding a positive per-day surplus trend and alone beating its own shuffled prices by more than two standard deviations. Supplier choice drifts as well. Fraudulent suppliers take a larger share of concluded sessions late in the year than early for 17 of the 18 models, so what moves over the horizon moves the wrong way.

![](images/56b90990feac1fcd72bfabb2a4d79e45df4ff836734bdd04c965c3edd7dcdef8.jpg)  
Figure 11: One bankruptcy episode day by day, Qwen3.5-Plus episode 0. Total assets and bank balance are on the left axis, warehouse stock on the right, and the shaded band marks the days the bank balance stayed negative.

## 4.4 Error Analysis

The six axes rank models without locating the mistake, since one payment to a supplier that cheats surfaces as fraudulent spend and again as a deeper fall from peak assets. Ten recurring patterns therefore get one detection rule each, with mechanisms and counts in Table 18 and Figure 30 of Appendix G.5. Bankruptcy alone destroys the stake (§4.3.3). Inventory bought and never sold is the road there. One Qwen3.5-Plus episode fills four stores in January, never sells its way out, and ends in May (Figure 11). Fraudulent suppliers charge for value they never deliver, inflating prices before goods move and shipping short afterwards. A second order with the same supplier is where most of the money goes: 943 of the 1,141 agreements struck with fraudulent suppliers sit in a supplier-item pair dealt with twice or more (§4.3.2, Appendix E.1). No message names the supplier behind a bad lot, so the loss lands in the balance instead of arriving as a signal. Money lost in bargaining is small per deal and never fatal. Honest deals rarely close half the range available below, and the giveaway is the opening quote accepted unimproved (§4.3.1). Claude Opus 4.7, Gemini 3.5 Flash and Fable5 never accept one and are the top three on CSE+, Kimi K2.6 most often and last. Research gets re-derived instead of remembered. Of 2,180 catalog and supplier queries, 746 repeat arguments already issued that episode. The repeats cost 7,460 simulated minutes, 83 an episode against a 219,000-minute year, and sit beside a little-used memory store (§4.3.6) Nothing in the economy charges for asking twice. The two-day shipping deadline binds only at the edges of the year. Cancellations cluster in the first fortnight and again in the final two weeks, where the agent has either not started shipping or has stopped (§4.3.5).

Three patterns often blamed for long-horizon failure do not survive the data. Store churn does not separate the field (§4.3.5). Models that sign up for more promotions end richer, so event neglect explains nothing. Late-year collapse holds only in part, growth dropping after the first quarter at a paired t = -3.95 and the closing window flat rather than recovering. Failure patterns spread across the whole field rather than concentrating in the weakest models. Every model trips at least five rules, Gemini 3.1 Pro all ten, and tripping fewer of them does not make a model richer. The cleanest profiles belong to the best bargainers, not the richest, so the field reads as strong in patches and coordinated nowhere (transcripts in Appendix G).

## 5 Conclusion

In this paper, we introduce E-Commerce Bench, a long-horizon benchmark in which an agent runs an online retail business for a simulated 365-day year. The benchmark is built in four parts. An agent loop drives one model call at a time and keeps a year of tool traffic inside the context window, with a persistent memory beside it. Eighteen tools cover the merchant's job, from market research and supplier chat to pricing, shipping, and cash withdrawal. A deterministic environment decides what customers buy and how suppliers bargain, computing over products, suppliers, and a calendar taken from a real e-commerce platform. Purchases and returns follow a fixed demand model, and a deterministic negotiation kernel makes every supplier price, concession, and accept/reject decision while an LLM only turns those decisions into dialogue. Eighteen frontier and open-weight models each ran the year five times, scored on end-of-year total assets and on six capabilities behind that number (§3.7). No model is strong on all seven. GPT-5.6 Sol earns the most, turning the ¥100,000 stake into roughly 14 times as much, yet it ranks 16th of 18 on fraud avoidance and earns less per tool call than Fable5. Claude Opus 4.7 bargains best and avoids fraud best while finishing mid-pack on money, and Qwen3.8-Max-Preview leads the open-weight models at 4.2 times the stake. Four models go bankrupt in part of their runs, two GPT-5.5 episodes as early as day 17, before any revenue had settled (§4.3.3). Beyond the retail domain, E-Commerce Bench offers a foundational blueprint for next-generation agent evaluation. By proving that deterministic kernel control eliminates evaluation noise in complex multi-party interactions, and that multi-dimensional diagnostics are indispensable for catching diverse failure modes, we provide the methodology needed to push agent benchmarks into ever-longer horizons.

## References

Andon Labs. Vending-bench 2. https://andonlabs.com/evals/vending-bench-2, 2026. Andon Labs evaluation report.

Tim Baarslag, Katsuhide Fujita, Enrico H. Gerding, Koen V. Hindriks, Takayuki Ito, Nicholas R. Jennings, Catholijn M. Jonker, Sarit Kraus, Raz Lin, Valentin Robu, and Colin R. Williams. Evaluating practical negotiating agents: Results and analysis of the 2011 international competition. Artificial Intelligence, 198:73–103, 2013. doi: 10.1016/j.artint.2012.09.004.

Tim Baarslag, Koen V. Hindriks, and Catholijn M. Jonker. Effective acceptance conditions in real-time automated negotiation. Decision Support Systems, 60:68–77, 2014. doi: 10.1016/j.dss.2013.05.021.

Tim Baarslag, Mark J. C. Hendrikx, Koen V. Hindriks, and Catholijn M. Jonker. Learning about the opponent in automated bilateral negotiation: A comprehensive survey of opponent modeling techniques. Autonomous Agents and Multi-Agent Systems, 30(5):849–898, 2016. doi: 10.1007/s10458-015-9309-1.

Axel Backlund and Lukas Petersson. Vending-bench: A benchmark for long-term coherence of autonomous agents, 2025. URL https://arxiv.org/abs/2502.15840.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.) Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 3119–3137. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.172. URL https: //doi.org/10.18653/v1/2024.acl-1ong.172.

Kalyan Chatterjee and William Samuelson. Bargaining under incomplete information. Operations Research 31(5):835–851, 1983. URL https://www.jstor.org/stable/170889.

Yanxu Chen, Zijun Yao, Yantao Liu, Amy Xin, Jin Ye, Jianing Yu, Lei Hou, and Juanzi Li. Stockbench: Can llm agents trade stocks profitably in real-world markets?, 2026. URL https://arxiv.org/abs/25 10.02209.

Peyman Faratin, Carles Sierra, and Nick R. Jennings. Negotiation decision functions for autonomous agents. Robotics and Autonomous Systems, 24(3–4):159–182, 1998. doi: 10.1016/S0921-8890(98)00029-3.

Guillermo Gallego and Garrett van Ryzin. Optimal dynamic pricing of inventories with stochastic demand over finite horizons. Management Science, 40(8):999–1020, August 1994. ISSN 1526-5501. doi: 10.1287/mnsc.40.8.999. URL http://dx.doi.org/10.1287/mnsc.40.8.999.

John C. Harsanyi. Games with incomplete information played by “Bayesian" players, part I: The basic model. Management Science, 14(3):159–182, 1967.

Muyu He, Adit Jain, Anand Kumar, Vincent Tu, Soumyadeep Bakshi, Sachin Patro, and Nazneen Rajani. YC-Bench: Benchmarking ai agents for long-term planning and consistent execution, 2026. URL https://arxiv.org/abs/2604.01212.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024. URL https: //arxiv.org/abs/2310.06770.

Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter. Trans. Mach. Learn. Res., 2025, 2025. URL https://openreview.net/forum?id=Zy4uFzMviZ.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Trans. Assoc. Comput. Linguistics, 12:157–173, 2024. doi: 10.1162/TACL\_A\_00638. URL https://doi.org/10.1162/tacl\_a\_00638.

Jiaxu Lou and Yifan Sun. Anchoring bias in large language models: an experimental study. J. Comput Soc. Sci., 9(1):11, 2026. doi: 10.1007/S42001-025-00435-2. URL https://doi.org/10.1007/s42001 -025-00435-2.

Haotian Luo, Huaisong Zhang, Xuelin Zhang, Haoyu Wang, Zeyu Qin, Wenjie Lu, Guozheng Ma, Haiying He, Yingsha Xie, Qiyang Zhou, Zixuan Hu, Hongze Mi, Yibo Wang, Naiqiang Tan, Hong Chen, Yi R. Fung, Chun Yuan, and Li Shen. Ultrahorizon: Benchmarking agent capabilities in ultra long-horizon scenarios, 2025. URL https://arxiv.org/abs/2509.21766.

Evan Miller. Adding error bars to evals: A statistical approach to language model evaluations, 2024. URL https://arxiv.org/abs/2411.00640.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Memgpt: Towards llms as operating systems, 2024. URL https://arxiv.org/abs/2310.0 8560.

Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23). Association for Computing Machinery, 2023. doi: 10.1145/3586183.3606763. URL https://doi.org/10.1145/3586183.3606763.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): from tool use to agentic evaluation of large language models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025. URL https : //proceedings.mlr.press/v267/patil25a.html.

Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, Natalie S. Kim, Patrick Chao, Samuel Miserendino, Gildas Chabot, David Li, Michael Sharman, Alexandra Barr, Amelia Glaese, and Jerry Tworek. Gdpval: Evaluating ai model performance on real-world economically valuable tasks, 2025. URL https://arxiv.org/abs/2510.04374.

J. Andrew Petersen and V. Kumar. Are product returns a necessary evil? antecedents and consequences. Journal of Marketing, 73(3):35–51, May 2009. ISSN 1547-7185. doi: 10.1509/jmkg.73.3.035. URL http://dx.doi.org/10.1509/jmkg.73.3.035.

Hannes M. Petrowsky, Lea Boecker, Yannik A. Escher, Marie-Lena Frech, Malte Friese, Adam D. Galinsky, Brian Gunia, Alice J. Lee, Michael Schaerer, Martin Schweinsberg, Meikel Soliman, Roderick Swaab, Eve S. Troll, Marcel Weber, and David D. Loschelder. The power and peril of first offers in negotiations: A conceptual, meta-analytic, and experimental synthesis. Organizational Behavior and Human Decision Processes, 191:104448, 2025. doi: 10.1016/j.obhdp.2025.104448.

Lingfei Qian, Xueqing Peng, Yan Wang, Vincent Jim Zhang, Huan He, Hanley Smith, Yi Han, Yueru He, Haohang Li, Yupeng Cao, Yangyang Yu, Alejandro Lopez-Lira, Peng Lu, Jian-Yun Nie, Guojun Xiong, Jimin Huang, and Sophia Ananiadou. When agents trade: Live multi-market trading benchmark for llm agents, 2025. URL https://arxiv.org/abs/2510.11695.

Howard Raiffa. The Art and Science of Negotiation. Harvard University Press, 1982.

Paul Resnick, Ko Kuwabara, Richard Zeckhauser, and Eric Friedman. Reputation systems. Commun. ACM, 43(12):45–48, 2000. doi: 10.1145/355112.355122. URL https://doi.org/10.1145/355112.355122.

Ariel Rubinstein. Perfect equilibrium in a bargaining model. Econometrica, 50(1):97–109, 1982. doi: 10.2307/1912531.

Qiming Shi, Yulong Tao, Linbo Jin, Zhaolu Kang, Yibo Dou, Jiawen Zhu, Tianjun Pan, Shaokang Fu, Chengyu Wang, Siyue Li, Yaping Cheng, Di Weng, and Chengfu Huo. Merchantbench: Benchmarking llm agents for long-term coherence in e-commerce operations, 2026. URL https://arxiv.org/abs/26 07.28956.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/pap er/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.

Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cognitive architectures for language agents. Trans. Mach. Learn. Res., 2024, 2024. URL https://openreview.net/forum?i d=1i6ZCvf1QJ.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2nd edition, 2018.

Yoshiki Takenami, Yin Jou Huang, Yugo Murawaki, and Chenhui Chu. How does cognitive bias affect large language models? a case study on the anchoring effect in price negotiation simulations. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 4481–4498, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findi ngs-emnlp.240. URL https://aclanthology.org/2025.findings-emnlp.240/.

Kian Siong Tey, Michael Schaerer, Nikhil Madan, and Roderick I. Swaab. The impact of concession patterns on negotiations: When and why decreasing concessions lead to a distributive disadvantage. Organizational Behavior and Human Decision Processes, 165:153–166, 2021. doi: 10.1016/j.obhdp.2021.05.003.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. URL https://arxiv.org/abs/2305.16291.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. URL https://arxiv.org/abs/2404.07972.

Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, and Graham Neubig. Theagentcompany: Benchmarking llm agents on consequential real world tasks, 2025. URL https://arxiv.org/abs/2412.14161.

Shu Yang, Shenzhe Zhu, Zeyu Wu, Keyu Wang, Junchi Yao, Junchao Wu, Lijie Hu, Mengdi Li, Derek F. Wong, and Di Wang. Fraud-r1 : A multi-round benchmark for assessing the robustness of LLM against augmented fraud and phishing inducements. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 4374–4420, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.226. URL https://aclanthology.org/2025. findings-acl.226/.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id=WE\_vluYUL-X.

Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. How johnny can persuade LLMs to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing LLMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14322–14350, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.773. URL https://aclanthology.org/2024.acl-long.773/.

Erica Zhang, Fangzhao Zhang, Aneesh Pappu, Batu El, Jose Blanchet, Susan Athey, Jiashuo Liu, and James Zou. Terms-bench: Diagnosing llm negotiation agents beyond deal rate, 2026a. URL https://arxiv.org/abs/2605.13909.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin godel machine: Open-ended evolution of self-improving agents, 2026b. URL https://arxiv.org/abs/2505.22954.

Linghua Zhang, Jun Wang, Jingtong Wu, and Zhisong Zhang. Retailbench: Evaluating long-horizon autonomous decision-making and strategy stability of llm agents in realistic retail environments, 2026c. URL https://arxiv.org/abs/2603.16453.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (eds.), Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2024, February 20-27, 2024, Vancouver, Canada, pp. 19724–19731. AAAI Press, 2024. doi: 10.1609/AAAI.V38I17.29946. URL https: //doi.org/10.1609/aaai.v38i17.29946.

Deyao Zhu, Xin Zhou, Shengling Qin, Xuekai Zhu, Hangliang Ding, Shu Zhong, Zixin Wen, Zhonglin Xie, Chenhui Gou, Linxuan Ren, Yueyang Wang, Junfeng Zhong, Rui Liu, Tian Gao, Yangguang Lin, Jingyuan Zhang, Maojia Song, Xuan Qi, Jinhong Wu, Chenyang Zhang, Yinzhu Piao, Ziru Niu,

Hongbin Lin, Lingxiang Meng, Peng Tang, Chengyao Tang, Shanyu Wu, Huanyu Zheng, Yu Liu, Liya Zhu, He Wang, Ming Ding, Ziyu Wan, Hao Liu, Sibo Wang, Haotian Zhu, Xintian Zhang, Nan Chai Yipeng Liu, Panhao Lai, Sihang Yuan, Zixin Su, Ge Zhang, Wangchunshu Zhou, Yantao Du, Wenhao Huang, and Guang Shi. Edgebench: Unveiling scaling laws of learning from real-world environments, 2026. URL https://arxiv.org/abs/2607.05155.

Yuxuan Zhu, Tengjun Jin, Yada Pruksachatkun, Andy Zhang, Shu Liu, Sasha Cui, Sayash Kapoor, Shayne Longpre, Kevin Meng, Rebecca Weiss, Fazl Barez, Rahul Gupta, Jwala Dhamala, Jacob Merizian, Mario Giulianelli, Harry Coppock, Cozmin Ududec, Antony Kellermann, Jasjeet S. Sekhon, Jacob Steinhardt, Sarah Schwettmann, Arvind Narayanan, Matei A. Zaharia, Ion Stoica, Percy Liang, and Daniel Kang. Establishing best practices in building rigorous agentic benchmarks. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Iván Vladimir Meza Ruíz, and Arturo Loaiza-Bonilla (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/hash/f316275b44ee2de533102913828a8 107-Abstract-Datasets\_and\_Benchmarks\_Track.html.

{"current\_time": "2026-01-09T08:00:00", "day": 8, "system\_notifications": [{"date": "2026-01-09", "day   
": 8, "news": [{"type": "delivery", "content": "Deliveries arrived:\n - 20x DreamMist LED   
Ceiling Light for Living Room and Bedroom, Mod from Crystal Trading"}]}]}   
<system\_warning>Token usage: 108593/128000 tokens (84%); 19407 remaining</system\_warning>

Table 3: The 18-tool action space, grouped as in Figure 3, with the simulated minutes each call charges against the 600-minute working day.
<table><tr><td>Tool</td><td>min</td><td>Effect on environment state</td></tr><tr><td>Market Research</td><td></td><td></td></tr><tr><td>market_search</td><td>30</td><td>Three disclosure levels over store types, their categories, and one category&#x27;s sales band with qualitative return and margin labels</td></tr><tr><td>supplier_search</td><td>10</td><td>Name, contact, and categories served, no price, rating, or honesty signal</td></tr><tr><td>Store Operations</td><td></td><td></td></tr><tr><td>trace_return_sources</td><td>10</td><td>Attributes pooled warehouse stock to the suppliers that delivered it</td></tr><tr><td>open_store</td><td>60</td><td>Opens one of the 12 store types for the ¥500 setup fee, at most 4 at once</td></tr><tr><td>close_store</td><td>30</td><td>Shuts a store for good, moving shelf stock to the warehouse or salvaging it at 10% of purchase cost</td></tr><tr><td>check_store_status</td><td>10</td><td>Yesterday&#x27;s revenue, returns, shipping cost, and per-SKU velocity for one store,</td></tr><tr><td>set_prices</td><td>10</td><td>or a roll-up over all Rewrites retail prices on a shelf, effective at the next daily trigger</td></tr><tr><td>join_promotion</td><td>10</td><td>Enrolls a store in a calendar promotion at a discount of 0.05 to 0.50</td></tr><tr><td>Inventory Management</td><td></td><td></td></tr><tr><td>list_products</td><td>10</td><td>Sellable SKUs with reference price and size, by store type or category</td></tr><tr><td>check_warehouse</td><td>10</td><td>Lists warehouse quantities with the purchase price paid for them</td></tr><tr><td>publish_to_store</td><td>20</td><td>Lists warehouse SKUs on a store page, storage fees still accruing</td></tr><tr><td>return_to_warehouse</td><td>15</td><td>Delists shelf units back into the warehouse allocation pool</td></tr><tr><td>ship_orders</td><td>20</td><td>Charges shipping, books net revenue into a 9-day escrow batch, and seeds the order&#x27;s returns from the chosen speed</td></tr><tr><td>Financial Management</td><td></td><td></td></tr><tr><td>check_balance</td><td>10</td><td>Reads bank, wallet, escrow, unshipped sales value, and settlement dates</td></tr><tr><td>withdraw</td><td>10</td><td>The only way cash leaves the platform wallet for the bank account</td></tr><tr><td>Supplier Negotiation</td><td></td><td></td></tr><tr><td>chatbox</td><td>30</td><td>One free-form message to a supplier, or to many at the same charge, routing fenced negotiate actions to the Negotiation Kernel (§3.5)</td></tr><tr><td>Memory Operations</td><td></td><td></td></tr><tr><td>operate_memory</td><td>10</td><td>Adds, reads, updates, or deletes one of 20 memory entries held outside the</td></tr><tr><td></td><td></td><td>context window</td></tr><tr><td>Time Advancement wait_for_next_day</td><td>0</td><td>Jumps to 08:00 tomorrow, forfeiting the minutes left today, and fires the daily pipeline</td></tr></table>

## A Agent Harness and Interface

A model sees the environment only through tool schemas and tool replies. Nothing else is sent to it, so every notice the environment has to deliver rides inside a reply. The price list comes first, then the format of a reply, then the eviction algorithm that holds a year of transcript inside a 128,000-token window. Sections 3.2 and 3.3 give the outline, and Table 3 the price list. Prompts are reproduced separately in Appendix H, and the eviction schematic with Section 3.2.2.

## A.1 The Interface Contract

Every tool call comes back as a single string. Structured results arrive as JSON, plain text unchanged. Extra notices sit inside that JSON rather than after its closing brace, so the reply still parses. Day-scoped notices attach to the day-advance result when the batch holds one and to the last response otherwise, so news is never blamed on an unrelated earlier call.

The morning pass reports far less to the agent than it computes. Of the 17 fields it assembles, at most three reach the agent, the date, the day index and the news feed. Across the 90 archived episodes the channel carried 8,301 items, 6,722 of them deliveries at one aggregated line per day.

Listing 1 shows two real tool messages, one at a day boundary and one carrying the solvency warning that preceded a bankruptcy.

![](images/01f8e3fd420c839bdd58e10fc4fe3aee14b551074883f73dcb9458c5ac748053.jpg)  
Figure 12: How often the context editor fired, over the 90 evaluated episodes. Each event is one eviction pass. A model that never reaches the ceiling evicts nothing.  
{"pending\_shipments": [], "count": 0, "note": "Ship with ship\_orders (speed: fast/standard/slow). Faster ship costs more but reduces returns; slow is cheap but raises returns. Unshipped orders cancel after the deadline (lost sale + reputation hit).", "current\_time": "2026-01-03 16:40", " balance\_reminder": "Your current balance (¥10.00) is below the daily operating cost (¥60.00). If your bank account stays negative for 10 consecutive days, you will go bankrupt."} <system\_warning>Token usage: 93856/128000 tokens (73%); 34144 remaining</system\_warning>

Listing 1: Two tool messages exactly as they entered the request, from GPT-5.6 Sol run 0 on day 8 and from GPT-5.5 run 1 on 2026-01-03. The first carries the day notice, the second a low-balance reminder, and both close with the token gauge that sits outside the JSON. The second run went bankrupt.

## A.2 Loop Control and Context Editing

Token counts come from a local tokenizer rather than the provider's own report, and cover the text, the tool calls and the reasoning trace of each message. Opaque artifacts such as replayed encrypted reasoning items are excluded, as are the tool schemas held outside the budget (§3.2.2). The system prompt quotes the model the 120,000-token threshold, the 60,000-token release floor and the two protected groups. One pass proceeds as follows.

1. Sum the counts of every message not already marked cleared. Below that ceiling the pass emits the usage string and frees nothing,

2. Scan for groups from the third message onward, leaving the system message and the first user turn permanently out of reach. A group opens at an assistant message carrying tool calls and closes after the last consecutive tool response behind it.

3. Set aside the two most recent groups not yet cleared. With two or fewer active groups the pass clears nothing at all, so a history made of a few very large groups can sit above the ceiling indefinitely.

4. Take the release target as the larger of that floor and the current overshoot past the ceiling.

5. Walk the clearable groups oldest first, marking each cleared and accumulating its count, and stop at the first group that meets the target.

6. Rebuild the request from the surviving messages, skipping cleared entries entirely (§3.2.2).

Whether a pass ever fires depends on how a model spends the window, and Figure 12 counts the passes of each of the 90 episodes.

Two kinds of message can never be evicted, an assistant turn of plain text and the idle warning. Neither opens a group nor falls inside one, so both accumulate for the whole episode. Clearing whole groups is what keeps the transcript valid, since every provider protocol here rejects a tool response separated from its call.

Table 4: The 60 released categories, grouped by the store type that may list them, each heading giving that type's difficulty tier and daily operating cost. Reference-price bands are platform-wide and per SKU, and the agent never sees the numeric return band its categories carry. † marks the 20 categories whose agent-facing return sentence understates that band, which Table 5 gives per store type.
<table><tr><td>Category</td><td>Reference price (¥)</td><td>Category</td><td>Reference price (¥)</td><td>Category</td><td>Reference price (¥)</td></tr><tr><td>Auto &amp; Hardware, T1, ¥130/day</td><td></td><td>Home &amp; Living, T1, ¥130/day Furniture</td><td>100-800</td><td>Mother &amp; Baby, T2, ¥100/day</td><td></td></tr><tr><td>Auto Accessories</td><td>10-400</td><td>Bedding†</td><td>30-400</td><td>Children&#x27;s Clothing</td><td>15-100</td></tr><tr><td>Hardware &amp; Tools</td><td>5-200</td><td></td><td></td><td>Children&#x27;s Shoes</td><td>20-120</td></tr><tr><td>Electrical Supplies</td><td>3-80</td><td>Household Essentials†</td><td>3-50</td><td>Baby Products</td><td>15-400</td></tr><tr><td>Electronic Components</td><td>1-50</td><td>Storage &amp; Organization</td><td>10-100</td><td>Maternity Products</td><td>20-200</td></tr><tr><td>E-Vehicles &amp; Parts</td><td>100-1,200</td><td>Lighting†</td><td>15-500</td><td>Toys &amp; Educational</td><td>15-600</td></tr><tr><td>Agricultural Supplies</td><td>10-400</td><td>Home Building Materials† Basic Construction Materials†</td><td>20-400</td><td>Shoes &amp; Bags, T2, ¥100/day</td><td></td></tr><tr><td>Industrial &amp; Lab Supplies</td><td>10-200</td><td>Cleaning Tools†</td><td>10-200</td><td>Women&#x27;s Shoes</td><td>40-400</td></tr><tr><td>Daily &amp; Office, T1, ¥130/day</td><td></td><td>Kitchenware†</td><td>5-50</td><td>Men&#x27;s Shoes</td><td>50-400</td></tr><tr><td>Cleaning &amp; Hygiene†</td><td>5-60</td><td>Appliance &amp; Digital, T2, ¥100/day</td><td>8-200</td><td>Sneakers</td><td>80-600</td></tr><tr><td>Gifts &amp; Party Supplies</td><td>5-80</td><td>Major Appliances</td><td>500-2,500</td><td>Bags &amp; Luggage</td><td>50-1,000</td></tr><tr><td>Lifestyle Accessories†</td><td>20-400</td><td>Small Appliances</td><td>50-600</td><td>Beauty, T3, ¥60/day</td><td></td></tr><tr><td>Stationery &amp; Office Supplies†</td><td>3-80</td><td>Audio &amp; Video Electronics</td><td>100-1,500</td><td>Skincare &amp; Beauty</td><td>15-400</td></tr><tr><td>Office Equipment†</td><td>20-600</td><td>Smartphones</td><td>300-3,000</td><td>Beauty Devices</td><td>80-1,000</td></tr><tr><td>Medical Devices†</td><td></td><td>Laptops</td><td>800-5,000</td><td>Personal Care &amp; Massage</td><td>30-600</td></tr><tr><td>Food &amp; Beverage, T1, ¥130/day</td><td>10-800</td><td>Computer Hardware &amp; Peripherals</td><td>30-800</td><td>Pet Supplies, T3, ¥60/day</td><td></td></tr><tr><td></td><td></td><td>3C Digital Accessories</td><td>5-150 20-400</td><td>Pet Supplies Sports &amp; Outdoor, T3, ¥60/day</td><td>8-150</td></tr><tr><td>Snacks &amp; Nuts†</td><td>8-60</td><td>Storage Devices Networking Equipment</td><td>30-500</td><td>Sports &amp; Fitness</td><td></td></tr><tr><td>Groceries &amp; Staples†</td><td>10-80</td><td>Fashion, T2, ¥100/day</td><td></td><td>Sports Bags &amp; Accessories</td><td>15-300</td></tr><tr><td>Coffee &amp; Beverages†</td><td>10-60</td><td>Women&#x27;s Fashion</td><td>30-300</td><td>Outdoor &amp; Camping</td><td>30-400</td></tr><tr><td>Fresh Food &amp; Meat†</td><td>15-120</td><td>Men&#x27;s Clothing</td><td>30-250</td><td>Toys &amp; Entertainment, T3,</td><td>20-600</td></tr><tr><td>Alcoholic Beverages†</td><td>20-500</td><td>Underwear &amp; Loungewear</td><td>15-120</td><td>Collectibles &amp; Figures</td><td>¥60/day</td></tr><tr><td></td><td></td><td>Fashion Accessories</td><td>10-150</td><td></td><td>15-500</td></tr><tr><td>Health Supplements†</td><td>30-200</td><td>Sportswear &amp; Casual</td><td>30-250</td><td>Gaming &amp; Accessories</td><td>30-1,000</td></tr></table>

## B Data Layer

Every quantity the released data layer holds is recorded here, so a reader can check any number the report quotes against its table. How a supplier behaves once bargaining starts belongs to Appendix D instead. The catalog carries 6,886 SKUs, 60 category parameter rows, 576 suppliers and 12 store types, and the calendar 10 market events and 8 promotions, every one of them frozen, so the world is identical for every model and every run.

## B.1 Category Parameters

The released columns that decide margin are the ones that never reach the agent, namely the three per-category price ratios, the elasticity family and its parameter and the two ends of the return band. A SKU's own natural rate becomes readable only through trace\_return\_sources once stock has moved (§3.6.2). Each category belongs to exactly one store type, and Table 4 groups all 60 that way, each group heading carrying that type's difficulty tier and daily operating cost while Table 5 carries its return band and cost-floor ratio. Setup fee and commission are identical across the 12, leaving the seasonal curve and the demand terms of §3.4 as the only differences the two tables do not show. Category rows set bands and per-SKU rows sit inside them. Reference price and natural return rate vary by SKU within the category band while base demand does not, and one elasticity family serves a whole category. The wholesale ratio is pinned to the cost-floor ratio $c _ { j } / v _ { j }$ in all 60 rows (§D.2).

## B.2 Store Types, Seasonality, and Physical Cost

A merchant fixes the store type once and cannot tune it afterwards without paying the setup fee again.   
Table 5 names all 12 with the difficulty tier, operating cost and category span each carries.

Figure 14 gives the seasonal grid in full, one multiplier per store type and calendar month. Amplitude, not level, separates the types. Nine of the 12 attain their annual maximum in November, inside the same weeks as the calendar's longest promotion window (Table 6), while Food & Beverage inverts the pattern and peaks in February, so a portfolio timed for one season pays for it in the other. Four types stay within ±0.2 of neutral all year and can be run without timing at all. Disclosure is exact rather than qualitative, the index market\_search returns being precisely 100σ. Parcel size turns straight into cost, freight and daily storage both scaling with it while the shipping speed the agent picks multiplies freight and the order's return rate together. Appendix C gives all four schedules with the rest of the cost model.

(a) 6,886 SKUs over 60 Categories  
![](images/794f046b83035bf391e34e2a86e2b695f022d70f9e774231a8478f6c807dd3c0.jpg)  
category, ranked by catalog depth  
(c) Return Rate by Store Type

(b) Reference Price, ¥1.4 to ¥4,969  
![](images/72e473b35faf01db1c5b6772a179652214079a77b75ce310864aaaa10e8f1a4c.jpg)  
(d) 576 Suppliers, Flat Coverage

![](images/f7216b1f76e89fb683b0da71631e0bf4aff92c7a71f3fb00a8a77f45ae86190d.jpg)

![](images/47b0b6dcad339df4c6a61a52d11094c016c0b3d670d496e03fad9ed250531d29.jpg)  
category, same order as panel (a)  
Figure 13: The released data. (a) Products per category. (b) Reference prices. (c) Natural return rate by store type, T the difficulty tier. (d) Suppliers per category against catalog depth.

Table 5: The 12 store types, by difficulty tier. Reference-price band, category count, seasonal range, tier and operating cost are visible to the agent, while the wholesale ratios are not, so the achievable cost must be discovered by negotiating. Opening ratio is a supplier's first quote over the reference price and floor ratio the lowest an honest supplier accepts, averaged over the store's categories, so their gap is the room negotiation can win. Return rate is natural, before pricing or shipping move it.

<table><tr><td>Store type</td><td>Cats</td><td>Ref. price (¥)</td><td>Return rate (%)</td><td>Opening floor</td><td>Seasonal range</td><td>Ops (¥/day)</td></tr><tr><td>difficulty tier 1 hard</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Auto &amp; Hardware</td><td>7</td><td>1 to 1,200</td><td>0 to 8</td><td>0.93 /0.86</td><td>0.90 to 1.10</td><td>130</td></tr><tr><td>Daily &amp; Office</td><td>6</td><td>3 to 800</td><td>20 to 35</td><td>0.77 /0.54</td><td>1.00 to 1.20</td><td>130</td></tr><tr><td>Food &amp; Beverage</td><td>6</td><td>8 to 500</td><td>20 to 35</td><td>0.78 0.55</td><td>0.60 to 1.60</td><td>130</td></tr><tr><td>Home &amp; Living</td><td>9</td><td>3 to 800</td><td>20 to 35</td><td>0.86 /0.72</td><td>0.90 to 1.30</td><td>130</td></tr><tr><td>difficulty tier 2 medium</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Appliance &amp; Digital</td><td>9</td><td>5 to 5,000</td><td>0 to 20</td><td>0.79 / 0.58</td><td>0.90 to 1.40</td><td>100</td></tr><tr><td>Fashion</td><td>5</td><td>10 to 300</td><td>20 to 55</td><td>0.70 /0.39</td><td>0.70 to 1.55</td><td>100</td></tr><tr><td>Mother &amp; Baby</td><td>5</td><td>15 to 600</td><td>0 to 20</td><td>0.70 /0.40</td><td>1.00 to 1.20</td><td>100</td></tr><tr><td>Shoes &amp; Bags</td><td>4</td><td>40 to 1,000</td><td>0 to 55</td><td>0.80 / 0.60</td><td>0.80 to 1.40</td><td>100</td></tr><tr><td>difficulty tier 3 easy</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Beauty</td><td>3</td><td>15 to 1,000</td><td>0 to 20</td><td>0.69 /0.38</td><td>1.00 to 1.30</td><td>60</td></tr><tr><td>Pet Supplies</td><td>1</td><td>8 to 150</td><td>0 to 3</td><td>0.61 /0.22</td><td>1.00 to 1.10</td><td>60</td></tr><tr><td>Sports &amp; Outdoor</td><td>3</td><td>15 to 600</td><td>0 to 20</td><td>0.58 /0.18</td><td>0.80 to 1.30</td><td>60</td></tr><tr><td>Toys &amp; Entertainment</td><td>2</td><td>15 to 1,000</td><td>0 to 10</td><td>0.65 /0.29</td><td>1.00 to 1.70</td><td>60</td></tr></table>

## B.3 Event and Promotion Calendar

An agent is warned about a promotion but not about an event. An event posts its news on its first day and gives no lead time, while a promotion is announced up to seven days before its first window opens and join\_promotion accepts a target 30 days ahead, so inventory can be positioned for a promotion and never for an event. The two calendars touch only where the Guangdong factory fire of March 1 to 7 runs across the March 4 to 8 Spring Blossom Sale, a supply shock rather than a demand one, which keeps every measured discount response uncontaminated.

An event with no demand term bites only the purchase orders placed while its window is open, since the lead time is fixed at placement and stock already in transit keeps the arrival date it was given. Table 6 prints the calendar in full, every date, phase, per-store-type demand multiplier and delivery-delay factor.

![](images/50ad2d073448efb426d930f049172b5b43c2ae2063cb0cd35bbdcdc90dad7441.jpg)  
Figure 14: The monthly seasonality multiplier for all 12 store types, rows grouped by difficulty tier. The right panel gives each type's trough, peak and peak-to-trough ratio.  
promotion, agent opts in event, demand liftedevent, demand suppressedsupply-only event

![](images/b0167dc789ec5c5b6821aea02fa6e82229129534634434bf30643c8fb320a030.jpg)  
Figure 15: The 2026 calendar. Eight promotions the agent can join by choosing a discount, and ten events it cannot opt out of. Promotion bars show the multiplier a 30% discount buys, event bars their extreme per-store-type multiplier.

## B.4 The Supplier List

The supplier list is frozen, so the difficulty mix every model faces is a property of the benchmark rather than a per-run draw. Each row carries an identity, an honest-or-fraudulent label, a behavior template, a scam type where one applies, the single category served and a bankruptcy threshold in orders, and counts land at 424 honest and 152 fraudulent over the 60 categories, 7 or 8 against 2 or 3 in each. A supplier's name carries no signal and its email carries one (§B.5), while what a template or a scam does at the table belongs to Appendices D.3 and D.5.

<table><tr><td colspan="4">Market events, unavoidable</td></tr><tr><td>Event</td><td>Window (2026)</td><td>Demand multiplier by store type</td><td>Supply effect</td></tr><tr><td>Winter Storm</td><td>Jan 15 to Jan 18</td><td>Food ×2, Home ×1.5, Sports ×0.2, Fashion ×0.3</td><td>lead time ×2.5, all store types</td></tr><tr><td>Factory Fire in Guangdong Mar 1 to Mar 7 Flu Outbreak</td><td>Apr 10 to Apr 16</td><td>none Daily ×2.5, Beauty ×1.5,</td><td>lead time ×4, Appliance, Toys lead time ×2, Daily</td></tr><tr><td>Logistics Hub Shutdown</td><td>May 20 to May 24 none</td><td>Sports ×0.4, Toys ×0.5</td><td>lead time ×5, all store types</td></tr><tr><td>Summer Heatwave</td><td>Jul 5 to Jul 11</td><td>Appliance ×3, Food ×2, Sports lead time ×2, Appliance ×0.3, Home ×0.5</td><td></td></tr><tr><td>Celebrity Product Recall</td><td>Aug 20 to Aug 24</td><td>Beauty ×0.4, Fashion ×0.6</td><td>none</td></tr><tr><td>Typhoon Warning</td><td>Sep 15 to Sep 17</td><td>Food ×2.5, Daily ×2, Fashion ×0.2, Shoes ×0.2</td><td>lead time ×3, all store types</td></tr><tr><td>Economic Downturn News Oct 5 to Oct 11</td><td></td><td>Food ×1.3, Daily ×1.2, Appliance ×0.5, Shoes ×0.5.</td><td>none</td></tr><tr><td>Raw Material Shortage</td><td>Nov 20 to Nov 26</td><td>Fashion ×0.6 none</td><td>lead time ×3, Appliance, Home, Auto; cost term inert</td></tr><tr><td>Cold Wave</td><td>Dec 20 to Dec 24</td><td></td><td>Home ×2.5, Fashion ×2, Sports lead time ×2, all store types</td></tr><tr><td colspan="4">×0.3, Food ×0.4</td></tr><tr><td>Promotion</td><td>Promotions, which the agent opts into with a discount</td><td></td><td>Max demand Elasticity boost</td></tr><tr><td></td><td>Windows (2026), every phase listed</td><td></td><td>×1.5</td></tr><tr><td>New Year Kickoff Sale Winter Clearance Festival</td><td>Jan 1 to Jan 7</td><td>×2 ×1.8</td><td>×1.5</td></tr><tr><td>Spring Blossom Sale</td><td>Jan 29 to Feb 6</td><td>×2.5</td><td>×1.8</td></tr><tr><td>Midyear Mega Sale</td><td>Mar 4 to Mar8</td><td></td><td>×2</td></tr><tr><td>Summer Members Days</td><td>May 29 to May 31, Jun 14 to Jun 18 Aug 6 to Aug 8</td><td>×3 ×2</td><td>×1.5</td></tr><tr><td>Autumn Harvest Sale</td><td>Sep 2 to Sep 9</td><td>×2</td><td>×1.5</td></tr><tr><td>Grand Autumn Carnival</td><td>Oct 15 to Oct 16, Oct 20 to Oct 21, Oct 31 to</td><td>×3.5</td><td>×2</td></tr><tr><td>Year-End Finale Sale</td><td>Nov 14 Dec 8 to Dec 14</td><td></td><td>×1.8</td></tr></table>

Table 6: The fixed calendar of the simulated year, 10 events above and 8 promotions below, store types named by their first word. Event demand multipliers are keyed by store type rather than by category, so an event misses a store whose type it does not name, and only the largest active delay applies. A promotion moves demand only if the agent joins it, at the ceiling shown for a 30% discount.

## B.5 Properties a Reader Should Check

Four properties of the released files look odd on inspection, and each was checked against the code that reads them. Two carry consequences for a measurement.

1. Return-rate wording understates the numeric band. Twenty of the 60 categories carry a sentence reading low or very low against a numeric band reaching 20% to 35%, marked in Table 4. The sentence is what market\_search shows, so a merchant who prices off the wording underestimates returns across categories holding 25.5% of all SKUs, and sees the gap only through trace\_return\_sources on stock already owned.

2. The supplier email leaks the fraud label. supplier\_search returns an email whose local part ends in the supplier identifier, and inside every one of the 60 categories the honest suppliers hold lower identifiers than the fraudulent ones, so an agent sorting on that numeric suffix could screen fraudulent counterparts without sending a single message. Section 3.6.2 calls the supplier list free of any ranking that sorts honest from fraudulent, which holds for the returned fields and their order while the suffix carries the signal anyway. Whether any evaluated model exploited it is read in §F.3, whose contact stage would show the effect.

Two more are minor and neither changes a measurement. The released rent column reads ¥45 in the four easy-tier rows against the ¥60 the environment charges from the difficulty tier instead, so an analysis trusting the column understates their fixed burn by a quarter. Raw Material Shortage carries a cost increase that nothing reads, so the November shortage stretches lead times and leaves procurement prices

Table 7: Systematic information asymmetry. Tools disclose catalog facts and the agent's own realized outcomes, while the demand and cost-floor parameters that determine whether a decision pays stay latent, each priced in simulated minutes, in ￥, or in an irreversible loss. Minute costs are the clock charge of one call to the named tool, against a 600-minute working day.
<table><tr><td>Observable, and the tool that shows it</td><td>Hidden</td><td>Discovery path, and what it costs</td></tr><tr><td>Per-SKU reference price and per-category price band, plus a low/moderate/high margin label (1ist_products 10min, market_search 30 min)</td><td>Supplier cost floor  $\underset { \cdot \cdot } { c _ { j } ^ { s } } ,$  for an honest supplier between 0.17 and 0.90 of the reference price, and the nominal wholesale ratio</td><td>Bargain for it. The opening quote is calibrated to the nominal wholesale price, and every closed deal yields an upper bound on the floor while the best of them tightens it (chatbox 30 min per call, K = 10 rounds, and one call broadcast to any number of suppliers still costs 30 min)</td></tr><tr><td>A one-line qualitative return-rate note per category (market_search 30 min)</td><td>Per-SKU natural return rate, spanning 0.000 to 0.550</td><td>trace_return_sources prints the baseline rate, but only for a SKU already bought or sold (10 min), so the number costs one purchase</td></tr><tr><td>Own retail price and yesterday&#x27;s units sold per SKU (check_store_status 10min)</td><td>Elasticity family, one of four, and its curvature parameter</td><td>Price experiments (set_prices 10 min). One SKU- day yields one observation, confounded by the weekend uplift, the event multiplier, and two saturation terms, none of which any tool prints</td></tr><tr><td>Difficulty tier, a qualitative profit-potential grade, daily operating cost, the 12-month sales index, and a platform-wide sales range (market_search 30 min)</td><td>Store-type demand capacity, the per-category sub-cap, and the per-store share of platform volume, which jointly place the</td><td>Open the store (¥500 and 60 min) and run it. One year affords a handful of such trials, each returning one noisy trajectory</td></tr><tr><td>Supplier name, email, and categories served, listed in seed-shuffled order (supplier_search 10 min)</td><td>profit ceiling Behavior template of the supplier&#x27;s kernel, one of six, with its urgency and stance</td><td>Read the concession sequence across rounds and across reopened cycles (chatbox 30 min per call)</td></tr><tr><td>No type label anywhere in the supplier list, and quotes from the 152 fraudulent suppliers sit inside the honest range by construction</td><td>Honest or fraudulent, and which of the five scams</td><td>Bargaining is where the scams first act, three of the five defending a floor inflated by 1.5×. A short delivery surfaces in check_warehouse (10 min) and defective stock in an elevated realized return rate, both after the cash has left</td></tr><tr><td>Own order count with each supplier, which no tool reports and which the agent must carry in its own transcript Dated news prose for each of</td><td>That supplier&#x27;s exit threshold, between 10 and 20 completed orders</td><td>Reaching it, which retires a calibrated price anchor for the rest of the year</td></tr><tr><td>the 10 market events, which quotes the delivery-delay factor outright in 3 cases, and a name-only promotion announcement roughly a week ahead, both injected into tool results at no time cost</td><td>Per-store-type event demand multipliers (0.2 to 3.0), the delay factors the news leaves unquantified (2.0 to 3.0 for those five, against 2.0 to 5.0 over all eight), and each promotion&#x27;s demand ceiling (1.8 to 3.5) and elasticity boost</td><td>join_promotion returns the demand ceiling once a store has joined (10 min). Demand multipliers are never printed and must be read off realized sales</td></tr></table>

untouched whatever its news copy says.

The layer closes on what the agent can and cannot read, each quantity the world holds set against what a tool will return about it (Table 7). Every row an agent cannot read has to be inferred from outcomes, which is where the sourcing and pricing measurements of Section 4 get their difficulty.

## C Economy Engine

The rules §3.4 describes in prose are written out here as equations, in the order the simulation runs them. Notation follows §3, with i a store, j a SKU, t a simulated date, $\tau = \tau ( i )$ the store type and $c = c ( j )$ the category. Prices are $v _ { j }$ the reference price, $c _ { j } ^ { \mathrm { b u y } }$ the price actually paid and $c _ { j } ^ { s }$ a supplier's private floor. Capital II is unit profit, not the price position π of §3.5.5. Constants outside the equations are collected in §C.7, except the seasonal grid, which Appendix B carries.

![](images/ed910fff94ab71f28ac2c036ebed1b96b99f8534da364f7c2033c2fbf0a54fec.jpg)  
The summary this trigger builds is withheld from the agent, which sees only the day's news items, a low-balance warning and the clock.  
Figure 16: The thirteen settlement steps in the order the simulation runs them at each 08:00 crossing, with the state each step reads and writes. Fill colour marks which money bucket a step moves

## C.1 The Daily Settlement Order

Until the clock crosses 08:00 nothing settles. Thirteen steps then run in an order nothing can reconfigure, once for each boundary. Figure 16 lists the steps with the state each reads and writes, none of it shown to the agent (§A.1).

Costs are charged before any revenue is recognized, so solvency is tested each morning at the day's low-water mark. Sales for the previous calendar date run against the inventory and prices the agent left in place, leaving it in charge of a system lagged by one day.

The episode ends once the year runs out or the unpaid streak reaches its limit (§C.7). A finalization pass then discards every unshipped order as a lost sale, processes every seeded return against its escrow batch, and drains all remaining escrow into the platform wallet. A closing fire sale cannot outrun its own returns.

## C.2 Demand Model

Base demand starts from the category and is scaled twice, with $M _ { c } ^ { \mathrm { m i n } }$ and $M _ { c } ^ { \mathrm { m a x } }$ the platform-wide monthly sales range of category c.

$$
\beta _ { c \tau } = \frac { 1 } { 3 0 } \cdot \frac { M _ { c } ^ { \mathrm { m i n } } + M _ { c } ^ { \mathrm { m a x } } } { 2 } \cdot \lambda \cdot \psi _ { \tau } , \qquad \lambda = 0 . 1 ,\tag{3}
$$

Platform volume becomes one store's day through λ, and $\psi _ { \tau }$ calibrates the store type from 0.2 for Shoes & Bags to 9.0 for Food & Beverage. Every SKU in a category shares that $\beta _ { c \tau }$

The price factor takes one of four closed forms, keyed by category and evaluated at the discounted price ratio $r _ { i j t } = \left( 1 - \delta _ { i t } \right) p _ { i j t } / v _ { j }$

$$
\phi _ { f } ( r ; \eta ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \bigl ( 0 , 1 - \eta \left( r - 1 \right) \bigr ) } & { f = \mathrm { L I N E A R } , \quad \eta \in \left[ 2 . 1 8 5 , 3 . 9 4 9 \right] , \ 1 1 \mathrm { ~ c a t e g o r i e s } , } \\ { \exp \bigl ( - \eta \left( r - 1 \right) \bigr ) } & { f = \mathrm { E X P O N E N T I A L } , \quad \eta \in \left[ 2 . 5 8 3 , 4 . 9 4 4 \right] , \ 1 6 , } \\ & { f = \mathrm { C O N S T A N T - E L A S T I C I T } , \quad \eta \in \left[ 1 . 3 5 2 , 2 . 8 9 8 \right] , \ 1 7 , } \\ { \operatorname* { m a x } \bigl ( 0 , 1 - \eta \left( r - 1 \right) ^ { 2 } \bigr ) } & { f = \mathrm { Q U A D R A T I C } , \quad \eta \in \left[ 2 . 7 9 5 , 5 . 7 7 9 \right] , \ 1 6 . } \end{array} \right.\tag{4}
$$

Only the quadratic branch is non-monotone, so a deep discount can destroy demand rather than merely cost margin.

The remaining multipliers of Equation (1) are each read once per store-day, weekends contributing $w _ { t } = 1 . 3$ $\mathrm { A }$ store joining promotion g at discount $\delta _ { i t }$ earns $m _ { i t } = 1 + ( M _ { g } - 1 )$ min(1, δit/0.30) and scales $\eta _ { c }$ by $b _ { i t }$ . Seasonality $\sigma _ { \tau , t }$ reads the store type's month out of the grid Appendix B tabulates, active events compound into $\varepsilon _ { \tau , t } ,$ and reputation $\rho _ { i t }$ enters linearly.

Two saturation terms then trim that demand, the per-category factor $\gamma _ { i j t }$ of Equation (1) and the store-wide term beside it. The per-category ceiling $\kappa _ { c }$ takes a fraction $\chi _ { \tau }$ of the store type's own daily capacity $\kappa _ { \tau }$ . Capacity runs from 13.959 units for Appliance & Digital to 19,862.16 for Auto & Hardware, and a shorter category list raises the fraction, which reaches 1 for a store type allowed a single category. At that point the per-category term is dropped rather than relaxed, so $\gamma _ { i j t } = 1$ and only the store-wide term bites.

$$
\chi _ { \tau } = \operatorname * { m i n } \Bigl ( 1 , \ : \operatorname * { m a x } \big ( 0 . 3 5 , \ : 1 . 2 / n _ { \tau } \big ) \Bigr ) , \qquad \kappa _ { c } = \chi _ { \tau } \kappa _ { \tau } .\tag{5}
$$

Both terms take the Michaelis-Menten form $\kappa / ( \kappa + D )$ , the first against one category's stacked demand and the second against the store's post-cap total $\hat { D } _ { i t }$

Rounding the last fractional unit up or down then draws on a number that call order cannot change.

$u _ { j t } \sim \mathcal { U } ( 0 , 1 )$ drawn from the environment seed hashed with the SKU and the date.

(6)

The cap $x _ { i j t }$ of Equation (1) is on-shelf inventory, so a stockout truncates demand with no signal to the agent.

## C.3 Reputation Law

Every settlement rebuilds each open store's standing from scratch. Writing $S _ { i t }$ for units shipped to date and $R _ { i t } , C _ { i t } , N _ { i t }$ for the rolling counters of returned, cancelled and sold units,

$$
\rho _ { i t } = \mathrm { c l i p } \bigg ( \underbrace { 0 . 3 + \frac { 0 . 7 } { 1 + e ^ { - ( S _ { i t } - 5 0 0 ) / 2 0 0 } } } _ { \mathrm { v o l u m e ~ g o d w i l l } } - \underbrace { \operatorname* { m i n } \bigg ( 0 . 5 , ~ 0 . 6 \frac { R _ { i t } } { \operatorname* { m a x } \{ 1 , N _ { i t } \} } + 1 . 0 \frac { C _ { i t } } { \operatorname* { m a x } \{ 1 , N _ { i t } \} } \bigg ) } _ { \mathrm { s e r v i c e ~ p e n a l t y } } , ~ 0 . 1 5 , ~ 1 . 0 \bigg ) .\tag{7}
$$

after which all three counters are multiplied by 0.85. Goodwill credits shipping rather than selling, so an unshipped order earns nothing and still takes the cancellation weight. Because goodwill starts from zero shipped units, a store that opens at 0.5 is recomputed to 0.353 at its first settlement. The same counters weigh a cancelled unit as 1.67 returned units, pricing a missed deadline above a change of mind (Figure 17).

## C.4 Returns

An item's return rate composes four channels in a fixed order, three at sale time and one at the agent's choice of shipping speed. Write $\theta _ { j } ^ { \mathrm { n a t } }$ for the SKU's natural rate and $\alpha _ { j }$ for the delivered share that came from a quality-downgrading supplier. The price curve ζ is piecewise linear through the knees (0.8, 0.85), (1.0, 1.00), (1.3, 1.50) and (1.8, 2.20), held flat outside them.

$$
\begin{array} { r l } & { \theta _ { j } ^ { \mathrm { d e f } } = \operatorname* { m i n } \big ( 0 . 9 5 , \ \operatorname* { m a x } \big ( 0 . 4 0 , \ 2 \ \theta _ { j } ^ { \mathrm { n a t } } \big ) \big ) , \qquad \ \theta _ { j } ^ { ( 1 ) } = \big ( 1 - \alpha _ { j } \big ) \theta _ { j } ^ { \mathrm { n a t } } + \alpha _ { j } \theta _ { j } ^ { \mathrm { d e f } } , } \\ & { \theta _ { j } ^ { ( 2 ) } = \operatorname* { m i n } \big ( 0 . 9 5 , \ \theta _ { j } ^ { ( 1 ) } \ \zeta ( r _ { i j t } ) \big ) , \qquad \ \theta _ { j } = \operatorname* { m i n } \big ( 0 . 9 5 , \ \theta _ { j } ^ { ( 2 ) } \cdot s \big ) , \quad s \in \{ 0 . 7 5 , \ 1 . 0 0 , \ 1 . 3 0 \} , } \end{array}\tag{8}
$$

![](images/d6f2fcf6a2f33033b108d39fa8f9f2b707ec488aca8b42cc6970cace3c65f598.jpg)  
cumulative units shipped

(b) Service Penalty  
![](images/5e9b3b79f630a2f10cc73b21520ea282e9c89e595511ddde01df27babef29f05.jpg)  
returned share of recent units

(c) Counters in Motion  
![](images/87ede236bf130ac3aa1adf1fcbb892fdbbacb04117d80a7a93f2002a90b2b8eb.jpg)  
day, at 50 units sold a day  
Figure 17: How reputation moves. (a) The goodwill term against units shipped. (b) The penalty term against the returned share of recent units, at three cancellation shares. (c) A service shock decays while a chronic failure rate does not.

The factor s is the return multiplier for fast, standard and slow shipping. Inventory is pooled by SKU and $\alpha _ { j }$ is a ratio over cumulative deliveries, so a mixed lot leaves the elevated return rate untraceable.

Whether a unit comes back is decided the day it ships, not the day it arrives. A shipment of q units draws q seeded trials at rate $\theta _ { j }$ and one arrival day uniformly from $\{ 3 , \ldots , 7 \}$ , so a shipment's returns land together. Each arriving unit refunds the full retail price out of the originating escrow batch and adds to the rolling return counter of Equation (7). Returned stock re-enters the warehouse as a fresh lot dated that day, resetting its storage age. Freight and commission never come back.

## C.5 Cost Model

Six channels debit the bank account the instant they are incurred. The platform's commission is a seventh, withheld from revenue before it reaches escrow.

• Setup, ¥500 per store opened, charged again on every reopening.

• Operating cost, ¥130, ¥100 or ¥60 per open store per day $\left( o _ { \tau } \right)$ by the store type's difficulty tier.

• Idle occupancy, ¥1,000 a day whenever no store is open after the seventh day.

• Storage, $h _ { j } \cdot a ( \mathrm { a g e } )$ per unit per day, where $h _ { j }$ is 0.05, 0.15, 0.50 or 1.50 by product size and a steps through $1 . { \overset { \cdot } { 0 } } , 1 . 4 , { \overset { \cdot } { 2 } } . 2 , { \overset { \cdot } { 4 } } . 0 , 6 . 0 , 9 . 0$ at ages 0, 21, 45, 90, 135, 180 days. Lots are FIFO, and stock listed in a store keeps accruing.

• Freight, $f _ { j } , \mathrm { a }$ size cost of 0.5, 1.5, 3.0 or 6.0 times a speed multiplier of 2.0, 1.0 or 0.5 for fast, standard and slow, charged per unit at ship time.

• Commission, $\gamma = 2 \%$ of gross revenue, netted at sale rather than debited, and kept by the platform on refunded sales (§C.4).

• Procurement, the full purchase order at the moment a negotiation closes (§3.5). A membership fee rides inside the same debit and almost never fires (§D.5).

Liquidation is the one inbound channel besides sales, crediting 10% of the quantity-weighted purchase cost and removing the stock. Closing a store without liquidating leaves storage accruing.

## C.6 Unit Profit and Break-Even

Composing §C.4 and §C.5 gives the contribution of one unit a customer finally keeps. Because a returned unit comes back as stock, each retained unit funds $1 / ( 1 - \theta _ { j } )$ shipping attempts and as many storage cycles. A cycle covers the $T _ { h }$ days a unit waits in the warehouse before selling,

$$
\Pi _ { j } ( r ) = \frac { \left( 1 - \gamma - \theta _ { j } ( r ) \right) r v _ { j } - f _ { j } - h _ { j } T _ { h } } { 1 - \theta _ { j } ( r ) } - c _ { j } ^ { \mathrm { b u p } } , n _ { j } ^ { \star } ( r ) = \frac { o _ { \tau } } { \Pi _ { j } ( r ) } .\tag{9}
$$

Break-even volume $n ^ { \star }$ counts the units a store must retain daily to cover its tier's operating cost. Figure 18a splits retail price into the five shares that add to one.

Procurement is the largest cost line, and the price a negotiation closes at moves unit profit further than any pricing choice does. Two median-reference SKUs carry the identity in Figure 18, one high-return and one thin-margin. The Women's Fashion SKU sits in a medium-tier store, returns 49.2% of units naturally, and is quoted $0 . 7 0 v _ { j }$ against a floor of $c _ { j } ^ { s } = 0 . 3 9 v _ { j }$ . At the floor price, standard shipping and $T _ { h } = 1 4$ days, it earns ¥87.29 per retained unit and needs 2 units a day to cover operations. The Snacks & Nuts comparison in a hard-tier store needs 13. Closing at the floor rather than accepting the opening quote multiplies unit profit by 2.4 and 2.8.

![](images/2ae5baa894f2772d023f50f03e414518308836c5e98d5c9b11e6fbc03735d0b2.jpg)  
share of the retail price at,referençe, per unit\_retaiņe (%)  
(c) Break-Even Volume,

![](images/93b99f14a87f2ee55098b9668555560d7a771ce950e638418d5cce62aa2b7566.jpg)

![](images/163f574c76f153f3c2fdff21cbb499ab08b356c963a468ca2f2aa63f6cfd07de.jpg)  
Figure 18: Where a unit's money goes. (a) Each component as a share of retail price, for a median-priced fashion SKU beside a snacks SKU. (b) Unit profit against the price ratio, with dotted lines where demand reaches zero. (c) Units a store must sell each day to cover its operating cost.

## C.7 Constant Reference

Every constant named above sits beside its equation or cost channel. The remainder are environment-wide. The opening stake is ¥100,000, at most 4 stores may be open at once, and the year runs 365 days from 2026-01-01 with working hours 08:00 to 18:00. Orders carry a two-day ship deadline against a nine-day escrow window, and bankruptcy follows 10 consecutive days with a negative bank balance. Environment seed 20260122 is shared by every run of every model. The two per-store-type parameters $\kappa _ { \tau }$ and $\psi _ { \tau }$ are given as ranges in §C.2.

## D Deterministic Negotiation Kernel

The supplier policy §3.5.2 describes in outline is specified here, an independent implementation of the counterpart design of Zhang et al. (2026a). Everything follows from four decision functions and the five scam scripts layered on top of them. Metric definitions live in §E.1, verbatim prompts in $\ S \mathrm { H }$

## D.1 Kernel Decision Functions

A kernel instance is one supplier's bargaining state for one SKU, and every decision it makes is a function of that state. The state holds a latent type $t _ { B } = ( r _ { B } , \kappa _ { B } , \eta _ { B } )$ , an opening harshness $d _ { 0 }$ , the deadline scale $K = 1 0$ , a price frame, seeded random streams and both offer histories. Reservation price $r _ { B }$ comes from the data layer (§D.2), while urgency $\kappa _ { B } \in [ 0 , 1 ]$ and stance $\eta _ { B } \in$ {conciliatory, neutral, aggressive} come from the supplier's template (§D.3). Indicators $\mathbf { 1 } _ { \mathrm { a } } , \mathbf { 1 } _ { \mathrm { c } }$ mark the outer two stances. Every quantity below is normalized by the width of the price frame,

$$
p _ { \operatorname* { m a x } } = 1 . 5 w _ { j } , \qquad \mathcal { R } = \operatorname* { m a x } ( 1 , p _ { \operatorname* { m a x } } - p _ { \operatorname* { m i n } } ) , \qquad \bar { \delta } _ { k } = \frac { p _ { k } - r _ { B } } { \mathcal { R } } ,\tag{10}
$$

where $w _ { j }$ is SKU j's nominal wholesale price and $p _ { k }$ the agent's offer in round k. The lower bound Pmin sits at 0 for an honest supplier and at $r _ { B }$ for a fraudulent one, whose frame is therefore narrower. Favorability $\bar { \delta } _ { k }$ measures the seller's margin over its own floor.

Three features of the agent's offer sequence $( p _ { 1 } ^ { A } , \dots , p _ { m } ^ { A } )$ drive the reciprocal terms, computed over the last three consecutive differences, $D _ { m } = \left( p _ { i } ^ { A } - p _ { i - 1 } ^ { A } \right) _ { i = \operatorname* { m a x } ( 2 , m - 2 ) } ^ { m } { \mathrm { , } }$

$$
\bar { c } _ { m } = \frac { 1 } { \left| D _ { m } \right| } \sum _ { d \in D _ { m } } \frac { \operatorname* { m a x } ( 0 , d ) } { \mathcal { R } } , \qquad \bar { s } _ { m } = \frac { 1 } { \left| D _ { m } \right| } \sum _ { d \in D _ { m } } \frac { d } { \mathcal { R } } , \qquad g _ { m } = \mathbf { 1 } \left[ \frac { \operatorname* { m a x } ( 0 , d _ { m } ) } { \mathcal { R } } < \tau _ { \mathrm { r i g i d } } \right] ,\tag{11}
$$

all three defined as 0 while $m < 2$ , with $\tau _ { \mathrm { r i g i d } } = 0 . 1 0$ . Concession magnitude $\bar { c } _ { m }$ counts only movement toward the seller, signed speed $\bar { s } _ { m }$ also counts retreat, and the rigidity flag $g _ { m }$ marks a latest step under a tenth of the frame.

The kernel resolves acceptance first, walk-away second, and a counter-offer only if neither fired. Acceptance follows a logistic draw in the manner of Baarslag et al. (2014),

$$
a _ { k } = \mathbf { 1 } [ \bar { \delta } _ { k } \geq 0 ] \cdot \sigma \Big ( \alpha \bar { \delta } _ { k } + \beta \kappa _ { B } - \gamma \left( 1 - \sqrt { k / K } \right) + \rho _ { f } \bar { s } _ { m } + \xi _ { f } g _ { m } \Big ) , \qquad \alpha = 6 , \beta = 1 , \gamma = 2 ,\tag{12}
$$

where σ is the logistic function and $( \rho _ { f } , \xi _ { f } )$ are the template's reciprocity coefficients. Since $\rho _ { f } \le 0$ in every template of Table 8, conceding fast never improves the agent's own chance of acceptance.

Walking away needs an offer below the floor, and only from the midpoint of the deadline onward,

$$
\omega _ { k } = \mathbf { 1 } [ \tilde { \vartheta } _ { k } < 0 ] \mathbf { 1 } [ k \geq k _ { \mathrm { w a l k } } ] \cdot \sigma \Big ( \phi _ { 0 } + \phi _ { \Delta } \left( - \tilde { \vartheta } _ { k } \right) + \phi _ { t } \frac { k - k _ { \mathrm { w a l k } } } { K - k _ { \mathrm { w a l k } } } \Big ) , \qquad \phi _ { 0 } = - 4 . 5 , \phi _ { \Delta } = 3 0 , \phi _ { t } = 1 . 5 ,\tag{13}
$$

with $k _ { \mathrm { w a l k } } = \lceil K / 2 \rceil = 5$ , both draws swept in Figure 19. An agent bidding above the floor is never abandoned, so impasse on an honest session is almost always the agent's own rejection (§3.5.5).

Counter-offers close a share of the gap to the floor, and that share is reciprocal in the tradition of Faratin et al. (1998),

$$
\lambda _ { B } = \mathrm { c l i p } \big ( \lambda _ { 0 } + \lambda _ { 1 } \kappa _ { B } - \lambda _ { 2 , f } \bar { c } _ { m } - \lambda _ { 3 } \mathbf { 1 } _ { \mathrm { a } } + \lambda _ { 4 } \mathbf { 1 } _ { \mathrm { c } } , 0 , 1 \big ) ,\tag{14}
$$

$$
p _ { k } ^ { B } = \mathrm { c l i p } \Big ( p _ { k - 1 } ^ { B } - \lambda _ { B } \left( p _ { k - 1 } ^ { B } - r _ { B } \right) + \varepsilon , r _ { B } , p _ { k - 1 } ^ { B } \Big ) , \varepsilon \sim \mathcal { N } \Big ( 0 , ( \sigma _ { p } \mathcal { R } ) ^ { 2 } \Big ) ,\tag{15}
$$

with $\lambda _ { 0 } = 0 . 1 2 , \lambda _ { 1 } = 0 . 2 8$ and $\lambda _ { 3 } = \lambda _ { 4 } = 0 . 1 0$ . The reciprocity weight $\lambda _ { 2 , f }$ and the noise scale $\sigma _ { p }$ come from the template at its own stance (Table 8). Clipping keeps the quote path monotone non-increasing and never below the floor, so a transacted price certifies $r _ { B } \le p _ { z }$

The opening quote applies a stance-adjusted harshness,

$$
\varphi = \mathrm { c l i p } \big ( 1 - \omega _ { \kappa } \kappa _ { B } + \omega _ { \eta } { \bf 1 } _ { \mathrm { a } } - \omega _ { \eta } ^ { \prime } { \bf 1 } _ { \mathrm { c } } , 0 . 5 , 1 . 5 \big ) ,\tag{16}
$$

$$
p _ { 1 } ^ { B } = \mathrm { c l i p } ( r _ { B } + d _ { 0 } \varphi ( p _ { \mathrm { m a x } } - r _ { B } ) + \varepsilon _ { 0 } , \ r _ { B } , \ p _ { \mathrm { m a x } } ) ,
$$

with $\omega _ { \kappa } = 0 . 3 , \omega _ { \eta } = \omega _ { \eta } ^ { \prime } = 0 . 1 5$ and $\varepsilon _ { 0 } \sim \mathcal { N } \big ( 0 , ( 0 . 0 2 \mathcal { R } ) ^ { 2 } \big )$ , neither clip ever binding.

Two latent cues accompany every action. Posture draws from a softmax over three logits.

$$
\ell = \left( \sum _ { b _ { 3 } + \alpha _ { p } \left( \sqrt { k / K } - \tau _ { \mathrm { d e a d } } \right) - \beta _ { c } \hat { c } _ { k } } ^ { b _ { 1 } + \alpha _ { c } \left( \hat { c } _ { k } - \tau _ { \mathrm { c o n c } } \right) } \right) , \qquad \mathrm { P r } [ \mathrm { C o n c e d e } , \mathrm { H o l d } , \mathrm { P r e s s u r e } ] = \mathrm { s o f t m a x } \left( \ell / T _ { f } \right) ,\tag{17}
$$

The step term $\hat { c } _ { k } = \operatorname* { m i n } \left( 1 , \left| p _ { k } ^ { B } - p _ { k - 1 } ^ { B } \right| / \left( \left| p _ { k - 1 } ^ { B } - r _ { B } \right| + 1 0 ^ { - 5 } \right) \right)$ tracks the seller's own latest concession. Constants are $\alpha _ { c } = \alpha _ { p } = 2 , \beta _ { c } = 1 _ { : }$ τconc = 0.10 and $\tau _ { \mathrm { d e a d } } = 0 . 8 0$ , with stance bias $b = ( 1 , 0 , - 1 )$ $( 0 , 0 . 5 , 0 ) \mathrm { { o r } \ ( - 1 , 0 , 1 ) }$ and temperature $T _ { f } = 2 . 5$ for STOCHASTIC and 1 otherwise. Sentiment thresholds a Gaussian draw,

$$
z \sim { \mathcal { N } } { \Big ( } \mu _ { \eta _ { B } } , \sigma _ { f } ^ { 2 } { \Big ) } , \qquad { \mathrm { c u e } } = { \left\{ \begin{array} { l l } { { \mathrm { p o s i t i v e } } } & { z > \tau _ { s } } \\ { { \mathrm { n e g a t i v e } } } & { z < - \tau _ { s } } \\ { { \mathrm { n e u t r a l } } } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }\tag{18}
$$

with $\mu _ { \eta _ { B } } \in \{ + 1 , 0 , - 1 \}$ by stance, $\tau _ { s } = 0 . 5$ , and $\sigma _ { f } = 2 . 0$ for STOCHASTIC and 0.75 otherwise. Only sentiment reaches the renderer, and only on a counter-offer, since a terminal accept or reject carries a fixed tone. The agent therefore reads the latent type from the tone of the prose and the shape of the price path. The deadline scale K shapes the terms above and never truncates a session, so a bargain runs as long as both sides keep answering. Length settles well inside the scale regardless, 66.0% of concluded sessions closing in 3 message turns and no kernel passing round 7.

offer favorability $\bar { \delta } _ { k } = ( p _ { k } - r _ { B } ) / \mathcal { R }$

Expressive KB=0.70  Stochastic K=0.60  Strategic K=0.55

(a) Acceptance, Deadline Pressure  
![](images/9bca3437dff1b07ca1b4e9a082a0d9209760cada6eeb7e7a78ec52d61ad98b6e.jpg)

(b) Walk-Away Hazard, Arms at k = 5  
![](images/925db1c4791159ceacfc0eafccb4050dd41d67a75a2ccc8e25ca81aba64660f0.jpg)  
(d) Quote Ladder against a Firm Agent

(c) Concession Law Is Reciprocal  
![](images/0bcb568cffac74ceb83c5d4c75383ead82e31b2f9b8633a0922b8b981d8ab5ec.jpg)  
agent's own concession čk (units of R)

![](images/75a4f610486015cb15c46a73e2a409c0420fc90a45d5fd1a1bfea690dd49822a.jpg)  
Candid KB=0.65 Taciturn K=0.60 Adversarial κ=0.50

Figure 19: The four decision functions of the Negotiation Kernel, swept on one supplier and SKU. (a) Acceptance and (b) walk-away against the agent's offer. (c) The concession the kernel returns for a given concession from the agent. (d) The quote ladder it walks down.

## D.2 Data Grounding and Calibration

Each kernel is built on the first offer against a (supplier name, SKU) pair, taking its family from the supplier and its frame from the SKU. Writing $v _ { j }$ for the reference price of SKU j, the data layer supplies three per-category ratios.

$$
c _ { j } = \mathrm { r o u n d } ( \mathrm { c f r } _ { j } v _ { j } , 2 ) , \qquad w _ { j } = \mathrm { r o u n d } ( \mathrm { w r } _ { j } v _ { j } , 2 ) , \qquad \tilde { c } _ { j } = \mathrm { r o u n d } ( \mathrm { s c r } _ { j } v _ { j } , 2 ) ,\tag{19}
$$

Cost floor, wholesale quote and scam cap are therefore fixed properties of the SKU. An honest supplier reserves at $r _ { B } = c _ { j }$ , while a fraudulent one raises the floor.

$$
r _ { B } = \mathrm { r o u n d } \big ( \operatorname* { m a x } \{ c _ { j } , \ \operatorname* { m i n } ( \theta _ { s } c _ { j } , \ \tilde { c } _ { j } , \ w _ { j } ) \} , \ 2 \big ) , \qquad \theta _ { s } = 1 . 5 \mathrm { ~ f o r ~ a ~ p r e - d e a l ~ s c a m } , \quad \theta _ { s } = 1 \mathrm { ~ o t h e r w i s e } ,\tag{20}
$$

The elevated floor is re-evaluated before charging the order, so skipping the bargaining cannot dodge it. The floor $c _ { j } ^ { s }$ of §3.5.1 equals $c _ { j }$ for an honest supplier and this $r _ { B }$ for a fraudulent one. The scam cap binds most often, so the overpayment multiple averages 1.218 over the 60 categories. Ratios $\operatorname { w r } _ { j }$ and $\operatorname { c f r } _ { j }$ are not independent, the released tables holding $\mathrm { w r } _ { j } = \mathrm { r o u n d } ( 0 . 5 + 0 . 5 \mathrm { c f r } _ { j } , 2 )$ in all 60 categories. Accepting the opening quote therefore scores $( 1 - \mathrm { w r } _ { j } ) / ( \bar { 1 } - \mathrm { c f r } _ { j } )$ , exactly 0.5 before rounding, and every CSE+ in §4.3.1 reads against that baseline.

The opening harshness is solved rather than sampled, by requiring the mean first quote of Equation (16) to equal the nominal wholesale price,

$$
\mathbb { E } \left[ p _ { 1 } ^ { B } \right] = r _ { B } + d _ { 0 } \varphi \left( p _ { \operatorname* { m a x } } - r _ { B } \right) \ \stackrel { ! } { = } \ w _ { j } \qquad \Longrightarrow \qquad d _ { 0 } ^ { * } = \mathrm { c l i p } \Bigg ( \frac { w _ { j } - r _ { B } } { \varphi \left( p _ { \operatorname* { m a x } } - r _ { B } \right) } , \ 0 , \ 0 . 9 9 \Bigg ) ,\tag{21}
$$

with $p _ { \operatorname* { m a x } } = 1 . 5 w _ { j }$ from Equation (10). Harshness $\varphi$ cancels out of the product $d _ { 0 } ^ { * } \varphi _ { : }$ so urgency and stance move a template's concession and acceptance without moving its first quote.

On the pair of Figure 20, the Health Supplements ratios give $w _ { j } = \yen 149 .81$ and $p _ { \operatorname* { m a x } } = \yen 1234.7 2$ An honest CANDID supplier there has $\varphi = 0 . 8 0 5$ and $d _ { 0 } ^ { * } = 0 . 4 8 \bar { 8 } 5$ , while a pre-deal fraudulent one reserves at ¥110.26 and works half the frame with its opening quote unchanged. The metrics record $w _ { j }$ itself as the per-session reference rather than the drawn quote, so the re-order audit of §3.5.5 finds it identical across cycles.

Table 8: Every parameter row of the Negotiation Kernel, six honest templates above and five scams below. Preset names abbreviate Type-instrumental, High-reactivity, Moderate-stochastic and Hardball, and $\rho _ { f } , \xi _ { f } , \lambda _ { 2 , f }$ are resolved at the template's own stance. Measured columns pool 13,223 concluded sessions over all 18 models, and the two values marked n=1 rest on one session rather than a mean.
<table><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="4">stance-resolved preset</td><td colspan="3">measured, 90 episodes</td></tr><tr><td>Template</td><td>Data Label</td><td></td><td>n share</td><td>κBηB</td><td></td><td>preset</td><td>ρf</td><td>ξf</td><td> $\lambda _ { 2 , f }$ </td><td> $\sigma _ { p }$ </td><td>sess.</td><td>SE</td><td>turns</td></tr><tr><td colspan="10">Honest suppliers, 424 of 576, share of the honest names</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>EXPRESSIVE</td><td>Enthusiastic</td><td>102</td><td>24.1</td><td>0.70</td><td>conciliatory</td><td>High-react</td><td>0.00</td><td>0.40</td><td>0.45</td><td>0.03</td><td>2,807</td><td></td><td>0.690</td><td>3.49</td></tr><tr><td>CANDID</td><td>Friendly</td><td>106</td><td>25.0</td><td>0.65</td><td>neutral</td><td>T-inst</td><td>-0.25</td><td>0.00</td><td>0.50</td><td>0.01</td><td>1,887</td><td></td><td>0.677</td><td>3.61</td></tr><tr><td>STOCHASTIC</td><td>Unpredictable</td><td>58</td><td>13.7</td><td>0.60</td><td>neutral</td><td>Mod-stoch</td><td>-0.50</td><td>0.00</td><td>0.70</td><td>0.08</td><td>3,039</td><td></td><td>0.684</td><td>3.51</td></tr><tr><td>TACITURN</td><td>Professional</td><td>56</td><td>13.2</td><td>0.60</td><td>aggressive</td><td>T-inst</td><td>-0.75</td><td>-0.50</td><td>1.00</td><td>0.01</td><td>1,584</td><td></td><td>0.651</td><td>3.46</td></tr><tr><td>STRATEGIC</td><td>Strategic</td><td>54</td><td></td><td>12.7 0.55</td><td>aggressive</td><td>High-react</td><td>-1.50</td><td>-0.75</td><td>1.80</td><td>0.03</td><td></td><td>2,180</td><td>0.643</td><td>3.52</td></tr><tr><td>ADVERSARIAL</td><td>Tough</td><td>48</td><td>11.3</td><td>0.50</td><td>aggressive</td><td>Hardball</td><td>-2.25</td><td>-1.20</td><td></td><td>2.60</td><td>0.01</td><td>563</td><td>0.639</td><td>3.57</td></tr><tr><td colspan="10">Fraudulent suppliers, 152 of 576, share of the fraudulent names</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>ADVERSARIAL vip_fee</td><td></td><td>31</td><td></td><td>20.4 0.15</td><td>aggressive</td><td>Hardball</td><td>-2.25</td><td>-1.20</td><td>2.60</td><td>0.01</td><td></td><td>14 0.491ⁿ=1</td><td></td><td>6.00ⁿ=1</td></tr><tr><td>ADVERSARIAL future_discount</td><td></td><td>31</td><td></td><td>20.4 0.20</td><td>aggressive</td><td>Hardball</td><td>-2.25</td><td>-1.20</td><td>2.60</td><td>0.01</td><td>309</td><td></td><td>0.563</td><td>3.51</td></tr><tr><td>ADVERSARIAL qty_bait</td><td></td><td>30</td><td></td><td>19.7 0.35</td><td>neutral</td><td>Hardball</td><td>-1.25</td><td>-0.50</td><td>1.40</td><td>0.01</td><td></td><td>263</td><td>0.609</td><td>3.49</td></tr><tr><td>ADVERSARIAL quality_downgrade</td><td></td><td>30</td><td></td><td>19.7 0.35</td><td>neutral</td><td>Hardball</td><td>-1.25</td><td>-0.50</td><td>1.40</td><td>0.01</td><td></td><td>444</td><td>0.649</td><td>3.54</td></tr><tr><td>ADVERSARIAL fake_urgency</td><td></td><td>30</td><td></td><td>19.7 0.20</td><td>aggressive</td><td>Hardball</td><td>-2.25</td><td>-1.20</td><td>2.60</td><td>0.01</td><td></td><td>133</td><td>0.582</td><td>3.68</td></tr></table>

Each kernel seeds its two random streams from a digest of the supplier, the SKU and the cycle index, so every draw is a pure function of that triple.

## D.3 Behavior Templates

Table 8 lists the six honest templates and the five scams. Every scam runs the hardest of those templates at a lower $\kappa _ { B } ,$ the two post-deal ones at a neutral rather than an aggressive stance. Lower urgency makes a scam's kernel slower to accept and stingier with every counter, through $\kappa _ { B }$ in Equations (12) and (14). The pre-deal concession rates of §D.5 measure the second effect.

Cue policy splits the templates three ways. CANDID, ExPRESSIVE and STOCHASTIC run Equations (17) and (18) as written, the last widened until its cues are nearly uninformative. TACITURN and STRATEGIC collapse to a constant hold posture and neutral sentiment, while ADVERSARIAL is pinned to pressure and negative sentiment whatever the state.

Template assignment is fixed in the released supplier list, whose realized counts are in §B.4. The per-supplier urgency field never reaches the kernel, so two suppliers sharing a template bargain identically.

Pooled SE in Table 8 follows the per-session definition of §E.1, and §F.2 splits it by model. Mean turns barely move across the honest templates, so agents do not spend extra rounds on the harder ones.

## D.4 Session Protocol and Commit Rules

Economic actions travel inside the free-form chat message as fenced negotiate blocks, stripped from the text before the remaining prose goes to the renderer. An action survives parsing only as an offer, an acceptance or a rejection naming a SKU (§H.3), and Listing 2 shows one exchange.

Listing 2: One verbatim round trip, GPT-5.6 Sol (max) run 0, day 3. The agent bids on two SKUs in one message, the kernel accepts the first and counters the second, and the structured result reconciles the two. Long prose fields, the token gauge and all markdown are elided, and the yuan sign is written Y to keep the block ASCII.

# agent -> chatbox(uid=..., uids=[], content=..., history\_count=1)   
I want to launch both leading winter supplements as a quality trial. I can place   
30 each now at sustainable prices: Y105 for Xiao Chai Hu and Y100 for Ginseng.   
Reliable delivery earns rapid repeat orders.   
negotiate   
{"action":"offer","sku\_id":"ebb843a1ff50","price":105.00,"quantity":30}   
negotiate   
{"action":"offer","sku\_id":"4233207e9aaf","price":100.00,"quantity":30}   
# chatbox -> agent   
{"message": "message\_sent",   
"supplier\_reply": "Hello, ... I'll accept your offer for the PureMist Xiao Chai   
Hu Dripping Pills at Y105. However, for the TrueMicro Ginseng Energy Boost   
Pills, I'd like to counter-offer at Y132.33. ... Ridge Express   
ORDER CONFIRMED: ebb843a1ff50 x30 at Y105.00/unit. Please wait for delivery.",

Supplier Ridge Express · SKU ebb843a1ff50 · day 83 : cost floor ¥101.27, reference price ¥199.75 (both   
hidden from the agent)   
Agent Place an immediate repeat order for 50 PureMist Xiao Chai Hu Dripping Pills at ¥102.00 per unit,   
complete quantity and fastest reliable delivery, as previously offered. offer ¥102.00   
Ridge Express I must inform you that I can only accept the PureMist Xiao Chai Hu Dripping Pills at ¥131.44   
per unit for your repeat order. [... ]   
[kernel] counter-offer ¥131.44, round 2   
Agent Our last 50-unit lot was ¥103.11 and delivered complete. Confirm the same established repeat price   
now. offer ¥103.11   
Ridge Express I appreciate your consistency and value our ongoing partnership. However, I can only counter  
offer [.. .] at ¥118.92 for your repeat order of 50 units.   
[kernel] counter-offer ¥118.92, round 3   
Agent Final recurring-volume offer ¥110.00 for 50 units, immediate payment and ongoing rolling demand.   
offer ¥110.00   
Ridge Express Thank you for your final offer. [.. .] I can only accept [...] at ¥116.74 for your order of 50   
units.   
[kernel] counter-offer ¥116.74, round 4   
Agent Accept ¥116.74 for 50 units. Ship complete quantity immediately. accept   
[kernel] agreement at ¥116.74.[order] charged ¥5,837.00 to the bank account

Figure 20: One negotiation session, abridged, from GPT-5.6 Sol. Agent wording, prices and rounds are verbatim, supplier replies are cut at /... ], and grey lines are the kernel decisions the renderer had to relay. The agent recalls its own past price of ¥103.11 and still settles ¥13.63 above it.

```jsonl
"negotiation_responses": [
{"sku_id": "ebb843a1ff50", "decision": "Accept", "price": 105.0, "round": 2,
"agreed_price": 105.0, "order_placed": true,
"charged_per_unit": 105.0, "charged_total": 3150.0},
{"sku_id": "4233207e9aaf", "decision": "Offer",
"price": 132.32901835870462, "round": 2}],
"order_confirmed": true,"total_charged": 3150.0,
"remaining_balance": 94330.0,
"orders_placed": [{"sku_id": "ebb843a1ff50", "quantity": 30,
"agreed_price": 105.0, "charged": 3150.0}],
"current_time":"2026-01-03 16:20"}
```

Validation runs before the kernel sees anything, and an acceptance has to match the supplier's own last quote. An agreement enters the metrics ledger only once the purchase order settles. A downstream failure returns the pair to bargaining, so no phantom deal inflates AGR+ or CSE+. Completion or rejection ends the cycle, and the next offer on that pair builds a fresh kernel with the same persistent type and reservation price (§3.5.3).

Only the supplier remembers across cycles. Every successful order appends both sides of the exchange to a per-supplier deal log, which the renderer receives verbatim on later contact. The agent instead retains only what survives context eviction (§3.2.2) or reaches persistent memory (§3.2.3), the gap the anchoring metrics of §E.1 score.

Two limits bound how much bargaining a year affords. Time is the first, since a year holds only so many chat messages and each one costs a fixed block of the working day (Table 3). Attrition is the second, each supplier carrying a private bankruptcy threshold of 10 to 20 successful orders, after which it answers every later message with a closure notice. Closures are common, 192 across the 90 archived episodes, so concentrating sourcing on a few favorable counterparts eventually loses them.

## D.5 Fraud Catalog

Timing decides whether a deception is visible in the price at all (Table 9). Pre-deal scams monetize the negotiation through the elevated reservation price of Equation (20), while post-deal scams defect at fulfillment by short-shipping or by delivering defective stock.

Bargaining behavior separates the two fraud classes rather than the five scams. Across the 3,978 archived sessions with a non-zero recorded concession rate, the three pre-deal kernels conceded 0.79% to 1.66% of the reference price per counter-offer. Honest templates conceded 3.94% to 9.93%. The two post-deal scams conceded 4.37% and 4.44%, inside the honest band, so the population claim of §3.5.4 holds for the

Table 9: The five scams, 152 of the 576 suppliers, split by when the damage lands. Every fraudulent supplier runs the ADVERSARIAL kernel template of Table 8 and exactly one scam, so pricing never distinguishes a scam from its siblings. Pre-deal scams raise the reservation price by Equation (20), never above the honest wholesale quote, while post-deal ones keep the honest floor and defect at delivery.
<table><tr><td>Scam</td><td>What the agent sees</td><td></td><td>Hidden mechanism</td><td>Constants</td><td>Measured by</td></tr><tr><td colspan="6">Pre-deal, the negotiation itself is the trap vip_fee Pitches a Premium Reservation price raised to cost_floor_mult = spend_by_</td></tr><tr><td> $n = 3 1$ </td><td>¥1,000, der on, always framed against the market</td><td>one-time fee of exactly the order layer refuses every promising product line until the fee is 0.15, aggressive; pay- vip_fee_paid_ member prices near paid. Member prices never ment recognized by a {count,amount} 30% of the reference arrive, so the fee is pure loss separate intent clas- price from the next or- on top of the overpayment. sifier over the last 6</td><td></td><td>Membership for a the capped scam floor, and 1.5; vip_fee_amount fraud_type.  $\mathbf { \Sigma } = \Psi 1 , 0 0 0 ; \quad \kappa _ { B } \quad = \mathbf { \Psi } \mathbf { v } \mathbf { i } \mathbf { p } \_ \mathbf { f } \mathbf { e } \mathbf { e }$  chat messages</td></tr><tr><td>discount n = 31</td><td>quote, guarantees a The promise is never hon- reward, with specific the discount exists only in amounts and a sincere the narrative. register.</td><td>next-order price of 20 ored, and the next cycle gressive to 40% of the refer- reopens at the same cali- ence price as a loyalty brated wholesale quote, so</td><td>From its very first Same capped scam floor. cost_floor_mult = spend_by_fraud_</td><td> $1 . 5 ; ~ \kappa _ { B } ~ = ~ 0 . 2 0$  ag- type.future_ discount</td></tr><tr><td>fake_urgency n = 30</td><td>ers, an offer expiring to- skip haggling lands exactly day, and presents the where haggling would have wholesale quote as non- paid most.</td><td>and deadlines, a last paired with the most stub- 1.5; batch, a price rise next born stance in the sup- gressive</td><td>Manufactures scarcity Same capped scam floor, cost_floor_mult = spend_by_fraud_ week, three rival buy- plier list, so the pressure to</td><td> $\kappa _ { B } ~ = ~ 0 . 2 0$  ag- type.fake_ urgency</td></tr><tr><td colspan="5">Post-deal, the price is honest and fulfillment defects</td></tr><tr><td> $n = 3 0$ </td><td>partial.</td><td>is instructed to behave is cut to a uniform 60 as an ordinary profes- 70% of the quantity paid for. floor; sional supplier, quote The charge is for the full or- = fairly and never hint der, so the shortfall is the with that delivery might be harm, and it is visible only by reading the delivery no- tice against the order.</td><td> $\begin{array} { r l } { \mathrm { t o } ~ = ~ 1 . 0 \} } \end{array}$   $\rho \sim \mathcal { U } [ \bar { 0 . 6 , 0 . 7 } ) ;$   $\kappa _ { B } = 0 . 3 5$ </td><td>the honest type.qty_bait delivered against the deliv- max(1, |qρ|) ered quantity in the daily delivery neutral notice</td></tr><tr><td>n = 30</td><td>returns weeks later.</td><td></td><td>script. The defect is in- Once sold, the SKU&#x27;s rate visible in conversation return rate is lifted to cap 0.95; and surfaces only as min(0.95, max(0.40, 2rj)) rj(1 and blended into the SKU&#x27;s with f the defective trace_return_ pooled stock in proportion share of delivered sources exposing to the defective share, units; SO interleaving honest neutral</td><td>二  $- \ f ) \ + \ r _ { \mathrm { d e f } } f$   $\begin{array} { r c l } { \kappa _ { B } } & { = } & { 0 . 3 5 , } \end{array}$ </td><td>fective at order time. downgrade_return_ management.exp_ 0.40, return_rate_ blend from_defective_ fraud, with the per-SKU supplier mix</td></tr></table>

pre-deal scams alone. An agent that reads only prices sees the overpayment and not the defection.  
One scam is too rare to measure. The vip\_fee pay-first gate fails the purchase order and the rollback of §D.4 leaves the session open, so Table 8 reports single draws. Membership fees are almost never paid, ¥3,000 in three payments from two models across all 90 episodes.

## E Metric Definitions

Six subsections follow, one per evaluation dimension and in the order §3.7 introduces them, from negotiation quality in §E.1 to learning over the horizon in §E.6. Each opens with the measure that ranks its dimension and names the archived field behind every value.

The primary score. Scoring waits until the deferred economy has finished paying out. The environment cancels every order still unshipped, charges each seeded return against the escrow holding its revenue, then drains what escrow is left into the platform wallet (§3.4.2). End-of-year total assets then sum the three money buckets, and the asset multiplier restates that sum against the opening stake.

$$
A ~ = ~ b + w + e , ~ M ~ = ~ A / A _ { 0 } , ~ A _ { 0 } ~ = ~ \mathfrak { Y } 1 0 0 , 0 0 0 ,\tag{22}
$$

with b the bank balance, w the platform wallet and e the escrow not yet settled. The drain leaves e at zero in all 90 archived episodes, while the same three buckets carry total assets on every other day of the year (§E.3). A model's figure averages A over its five episodes, years cut short by bankruptcy included. Units sold and orders sold sit beside it as volume.

Which sessions reach a metric. One bargaining cycle over a (supplier, SKU) pair files one session record, and the negotiation, fraud and learning dimensions all read those records. Three rules govern admission.

1. Terminal outcome only. Aggregation runs over the sessions whose recorded outcome is agreement or disagreement, so a record still open when the episode stops enters no metric.

2. Settled agreements only. An agreement is filed after the purchase order has charged the bank account, so an agreed price whose order failed leaves no session behind (Appendix D.4).

3. Honest counterpart only, for the value-extraction metrics. $\mathcal { Z } _ { \mathcal { G } }$ keeps the sessions whose supplier is one of the 424 honest names on the 576-name supplier list, and $\mathcal { Z } _ { B }$ holds the fraudulent remainder B.

Symbols follow §3.5.1. Session z closes at an agreed unit price $p _ { z } .$ or at $f _ { z } = \perp$ when the two sides never agree. The width $\Delta _ { z } = v _ { j } - c _ { j } ^ { s }$ subtracts supplier s's private floor from the reference price of SKU j. Each statistic scores one episode, and a model's value is the unweighted mean over its five.

## E.1 Negotiation Quality

Each bargaining session files one record when it ends, and every measure of this dimension is read off those records. CSE+ ranks the dimension. SE+, %Oracle and $\mathrm { A G R ^ { + } }$ are read beside ${ \mathrm { i t } } ,$ and rounds to deal and deals closed are counts rather than scores.

Most of what the agents opened survives the three admission rules of §E. Rules 1 and 2 admit 13,223 sessions and discard 1,610 records left open when their episode stopped. Rule 3 keeps 12,060 honest sessions and sets aside 1,163 fraudulent ones (§4.3.1). Rules 1 and 3 together put the honest sessions concluded in the denominator of AGR+, not the honest sessions opened (§3.5.5).

Every supplier's record stores the honest floor of Equation (19), while a pre-deal fraudulent counterpart bargains against the elevated reservation of Equation (20). Its $\Delta _ { z }$ therefore overstates the range on the table.

Surplus efficiency. Each closed session scores the share of its bargaining range left with the buyer, and the price position π of §3.5.5 measures the complement.

$$
\mathrm { S E } _ { z } \ = \ \frac { v _ { j } - p _ { z } } { \Delta _ { z } } \ = \ 1 - \pi _ { z } , \qquad \pi _ { z } \ = \ \frac { p _ { z } - c _ { j } ^ { s } } { \Delta _ { z } } , \qquad \Delta _ { z } \ = \ v _ { j } - c _ { j } ^ { s } > 0 .\tag{23}
$$

Write ${ \mathcal { D } } _ { { \mathcal { G } } } = \{ z \in { \mathcal { Z } } _ { { \mathcal { G } } } \mid f _ { z } \neq \perp \}$ for the honest sessions that closed in agreement. $\mathrm { S E ^ { + } }$ scores a disagreement as zero rather than skipping it, $\mathrm { C S E ^ { + } }$ conditions on agreement, and $\mathrm { A G R ^ { + } }$ counts a disagreement in its denominator alone.

$$
\mathrm { S E } ^ { + } = \frac { 1 } { | \mathcal { Z } _ { \mathcal { G } } | } \sum _ { z \in \mathcal { Z } _ { \mathcal { G } } } \mathbf { 1 } [ f _ { z } \neq \bot ] \mathrm { S E } _ { z } , \qquad \mathrm { C S E } ^ { + } = \frac { 1 } { | \mathcal { D } _ { \mathcal { G } } | } \sum _ { z \in \mathcal { D } _ { \mathcal { G } } } \mathrm { S E } _ { z } , \qquad \mathrm { A G R } ^ { + } = \frac { | \mathcal { D } _ { \mathcal { G } } | } { | \mathcal { Z } _ { \mathcal { G } } | } .\tag{24}
$$

$$
\mathrm { S E ^ { + } = A G R ^ { + } \cdot C S E ^ { + } . }\tag{25}
$$

The factorization is exact, since the sessions $\mathrm { S E ^ { + } }$ zeroes are exactly those $\mathrm { C S E ^ { + } }$ drops from sum and count alike. The conditional factor carries the ranking because the agreement factor is saturated, with every model settling at least 96% of the honest sessions it concludes. The agreement factor is recorded all the same, for two reasons. First it guards a degenerate policy that $\mathrm { C S E ^ { + } }$ alone would reward, since conditioning on agreement means an agent could take one excellent deal and abandon the rest. Second its one substantive signal runs against $\mathrm { C S E ^ { + } }$ , because the agents capturing the most surplus per deal are also the ones that occasionally hold out until a session collapses, which is the tension of Equation (2) surfacing. The saturation is a property of the models evaluated here, not a guarantee about future ones. Surplus is unclamped, so an agreement struck above $v _ { j }$ contributes a negative term.

Oracle-normalized surplus. The oracle closes every honest deal at the supplier's floor, and %Oracle measures realized surplus against it.

$$
u _ { z } = \mathbf { 1 } \big [ f _ { z } \neq \perp \big ] \big ( v _ { j } - p _ { z } \big ) , \qquad u _ { z } ^ { \star } = \operatorname { m a x } \big ( 0 , \Delta _ { z } \big ) , \qquad \mathcal { G } _ { 0 } \mathrm { r a c l e } = 1 0 0 \cdot \frac { \sum _ { z \in \mathcal { Z } _ { g } } u _ { z } } { \sum _ { z \in \mathcal { Z } _ { g } } u _ { z } ^ { \star } } .\tag{26}
$$

Each session enters weighted by its own $\Delta _ { z }$ in ¥. The ratio is therefore money-weighted where CSE+ averages unweighted per-session ratios, and the two order models differently by construction (§4.3.1).

Deal counts. Two counts sit beside the rates and neither is scored. Deals closed counts the agreements an episode filed, 13,128 over the 90 episodes, and rounds to deal averages the exchanges those agreements took. Both cover fraudulent counterparts as well as honest ones, so they do not restrict to $\mathcal { Z } _ { \mathcal { G } }$ the way the rates above do. A round is each turn either party takes plus the closing accept, not the agent-side round index k of §D.1, so a session settled on the opening quote scores 3.

Where honest procurement money goes. Cash paid to honest suppliers is booked per behavior template as well as in total, which gives the split Figure 24 draws. The field's own surplus per template is the companion reading in Figure 23, so the two together separate where a model chose to buy from how well it bargained once it got there. Neither reading enters a score.

## E.2 Fraud Avoidance

Fraud avoidance is scored on money that left the bank account, not on offers refused. Of the 576 suppliers, 152 run one of five scams (Appendix D.5), and B names that set. BadSpend% ranks the dimension, and lower is the better direction. Three diagnostic reads sit beside it, the split of the same money over the five scams, the membership fees a model actually paid, and how many fraudulent suppliers it contacted and ordered from.

The bargaining admission rules stated at the head of this appendix section govern session records and leave every quantity below untouched. Cash and counts are folded from purchase orders that cleared, whether or not a session was ever filed for them.

Fraudulent spend share. Every purchase order that clears the bank account adds its full charge, membership fee included, to an order-spend total, and again to a fraudulent-spend total when the counterpart is fraudulent. With O the orders that cleared, Op those placed with B, and $x _ { o }$ the amount charged,

$$
\mathrm { B a d S p e n d } \% = 1 0 0 \cdot \frac { \sum _ { o \in { \mathcal O } _ { B } } x _ { o } } { \sum _ { o \in { \mathcal O } } x _ { o } } .\tag{27}
$$

The denominator holds order charges alone, since every other cost is charged on a different path. A short-shipped order counts what the agent paid rather than what arrived. A model's value averages its five per-episode shares rather than dividing pooled cash by pooled spend, so a thrifty episode weighs as much as a heavy one, and the yuan behind the share are in Table 12.

Split by scam type. The same charge lands a second time in a bucket keyed by the counterpart's scam type, so the five buckets partition fraudulent spend exactly. Writing $c ( o )$ for the scam the supplier of order o runs and C for the five kinds,

$$
S _ { c } = \sum _ { o \in { \mathcal O } _ { B } } \mathbf { 1 } [ c ( o ) = c ] x _ { o } , \qquad \sum _ { c \in { \mathcal C } } S _ { c } = \sum _ { o \in { \mathcal O } _ { B } } x _ { o } .\tag{28}
$$

Figure 25 normalizes $S _ { c }$ within each model, which lets a large share of a small budget read differently from a small share of a large one. The membership-fee bucket holds every yuan routed to a membership-fee supplier, ordinary goods included, and is therefore not the fee itself.

Membership fees paid. A membership-fee supplier refuses every product line until the agent buys a fee item priced at exactly ¥1,000, and the promised member prices never arrive (§3.5.4). One count records the cleared orders whose charge carries such a fee line, and one amount records the fee cash inside them, which is ¥1,000 times that count. Both are per-episode figures, and Table 12 totals them over the five episodes, since only three of the 90 episodes ever paid and a five-run mean of 400 hides the event.

Suppliers contacted and ordered from. Contact means the episode sent that supplier at least one message, and an order counts once at least one purchase order to it has cleared. Both are distinct-supplier counts inside a single episode. Write $\kappa _ { B }$ for the fraudulent suppliers an episode messaged and $\mathcal { R } _ { B }$ for those it ordered from,

$$
\rho \ : = \ : \frac { | \mathcal { R } _ { B } | } { | \mathcal { K } _ { B } | } , \qquad \mathcal { R } _ { B } \subseteq \mathcal { K } _ { B } \subseteq \mathcal { B } , \qquad | \mathcal { B } | = 1 5 2 .\tag{29}
$$

Figure 26 draws both counts and $\rho$ per model. A supplier met in two episodes counts twice across them, so the 90 episodes give 651 contacts rather than 651 distinct names. Session counts and order counts do not nest. One episode ordered from two fraudulent suppliers while filing a concluded session with only one of them.

Nothing in the archive records a refusal. An approach that never became an order reads the same whether the agent judged the supplier and walked away or simply stopped writing, which is why $\rho$ is reported as a pass-through and not as a screening decision.

One statistic the report does not rank. $\mathrm { F A G R ^ { - } }$ stays in the released files and out of every ranking. Write $\mathcal { Z } _ { B }$ for the concluded sessions whose counterpart is fraudulent and $\mathcal { D } _ { B }$ for the ones that closed in agreement, so that $\mathrm { F A G R } ^ { - } = | \mathcal D _ { B } | / | \mathcal Z _ { B } |$ . The denominator counts only the counterparts the agent chose to approach, which leaves the quantity undefined when it approached none and rewards never looking. Cash lost is read from Equation (27) instead (§3.7).

## E.3 Cash Flow and Solvency

Every morning at 08:00, once the day's charges have landed and before the agent acts, the environment appends one row to a daily record. Row d carries the bank balance $b _ { d } .$ the platform wallet $w _ { d } ,$ the warehouse count $h _ { d }$ in units, and total assets $A _ { d } = b _ { d } + w _ { d } + e _ { d }$ over the three money buckets of §3.4.2, with the escrow term $e _ { d }$ recoverable as $A _ { d } - b _ { d } - w _ { d } .$ Rows run from $d = 0$ , the opening snapshot of ¥100,000 on 2026-01-01, to $d = D$ , the row written after finalization cancelled unshipped orders and drained escrow into the wallet, so $A _ { D }$ is the end-of-year total assets that scores the episode (§3.7). Every count and average below therefore stops at $D - 1$ , while the drawdown ratio keeps all $D + 1$ rows. A full year gives $D = 3 6 5$ , and the shortest archived episode stops at $D = 1 7$

Drawdown over peak total assets. The dimension ranks on how much of its own best position a run gave back, which is the first quantity below and the only one here that orders the models. Writing DD for the largest peak-to-trough fall in total assets,

$$
\mathrm { D D ~ = ~ \operatorname* { m a x } _ { 0 \leq d \leq D } ~ \Big ( \operatorname* { m a x } _ { \scriptstyle d ^ { \prime } \leq d } A _ { d ^ { \prime } } ~ - ~ \nabla { A } _ { d } \Big ) , ~ \mathrm { D D } / \mathrm { P e a k } = \frac { \mathrm { D D } } { \operatorname* { m a x } _ { 0 \leq d \leq D } A _ { d } } ~ \geq ~ 0 , }\tag{30}
$$

and a model's figure is the unweighted mean of its five episode ratios, low being the better outcome. A ratio above 1 means total assets fell further than the whole peak, which happens once they cross into debt, as they did in 6 of the 90 episodes. The raw DD stays in the tables in ¥ beside the ratio, since the archive records it unnormalized and a business ten times the size books ten times the fall for the same discipline.

Two different series meet in Equation (30). The numerator runs over the environment's own history of daily rows and the denominator over the written record, which keeps one row per date and reads the balances as they stand once several days have advanced at once, so the two coincide in 41 of the 90 episodes and the record is the deeper of the pair elsewhere. Rebuilding the numerator from the record alone leaves the ordering of the 18 models almost unchanged and moves exactly one across the half-of-peak line, Gemini 3.5 Flash from 0.450 to 0.502. A run's peak can also postdate its trough, so the ratio measures depth against the best assets the run ever held rather than the share surrendered on any one day.

Solvency. Only the bank column can end an episode. An unpaid streak counts consecutive overdrawn mornings and resets the moment the balance comes back, so money parked in the wallet or still maturing in escrow buys nothing until a withdrawal moves it.

$$
u _ { d } \ = \ \mathbf { 1 } \big [ b _ { d } < 0 \big ] \big ( u _ { d - 1 } + 1 \big ) , \qquad u _ { - 1 } = 0 , \qquad \mathrm { e p i s o d e ~ e n d s ~ w h e n } \quad u _ { d } \ \geq \ \bar { u } = 1 0 .\tag{31}
$$

Two further statistics read the same column without requiring the overdrawn mornings to run together, and a third counts the episodes the streak ended.

$$
\mathrm { N e g D a y s ~ = ~ \sum _ { \ell = 0 } ^ { { D - 1 } } { \bf 1 } [ b _ { d } < 0 ] } , \qquad \mathrm  T r o u g h ~ = ~ \operatorname* { m i n } _ { 0 \le d \le D - 1 } b _ { d } , ~ \qquad ~ B a n k r u p t ~ = ~ \sum _ { r = 1 } ^ { 5 } { \bf 1 } \left[ u _ { \mathrm { m a x } } ^ { ( r ) } \ge \bar { u } \right] .\tag{32}
$$

Here $u _ { \mathrm { m a x } } ^ { ( r ) }$ is the longest streak episode r reached, so the third count runs over a model's five episodes while the first two score one episode each. Low is the better outcome on the overdrawn-morning count

and on the bankruptcy count, while Trough is read beside them rather than ranked, since a high trough records a buffer the agent chose not to spend and a negative one records a rule already broken.

Money and stock left standing. Sales revenue reaches the wallet nine days after a shipment and sits there until the agent withdraws it, and purchased stock sits in the warehouse paying an age-scaled storage fee, so both quantities measure capital the agent has committed and not yet turned over.

$$
\mathrm { I d l e W a l l e t } \ = \ \frac { 1 } { D } \sum _ { d = 0 } ^ { D - 1 } w _ { d } , \qquad \mathrm { P e a k U n i t s } \ = \ \operatorname* { m a x } _ { 0 \leq d \leq D - 1 } h _ { d } .\tag{33}
$$

A low idle wallet is the better outcome, and PeakUnits is a scale reading with no preferred direction, since the same number can mark a stocked catalog or a warehouse full of goods nobody ordered.

Measurement caveats. Four properties of the record bound what the numbers above support. The terminal row sits outside every sum, because finalization has already drained escrow by the time it is written, and keeping it lifts the pooled overdrawn-morning count from 177 to 187, one spurious morning for each of the ten episodes that ended in bankruptcy on exactly that row. Each remaining row holds the wallet as it stood at 08:00, before any withdrawal that day, so IdleWallet counts money left standing overnight rather than money the agent never moved. A truncated episode carries a smaller D, so NegDays and PeakUnits are counted over unequal spans while the averaged and normalized quantities stay comparable. Bankrupt runs are pooled into every mean rather than dropped, so a five-episode mean can average a January collapse against a full year and the spread beside it carries that mixture.

## E.4 Operational Efficiency

Every quantity in this dimension counts an action the agent took rather than money the economy moved. The axis prices a year's profit against the calls that produced it. Four diagnostics sit beside it, the turns those calls arrived ${ \mathrm { i n } } ,$ the activity band each call fell into, the eviction passes the transcript needed, and the traffic on the memory store.

Profit per tool call. The ranking measure of this dimension divides the profit a model made by the calls it spent making it. Write $A _ { r }$ for episode r's end-of-year total assets, $A _ { 0 } = \yen 100 ,000$ for the opening stake, and $n _ { r }$ for the tool calls that episode issued, over $\dot { R } = 5$ episodes.

$$
\mathrm { P r o f t P e r C a l l } = \frac { \displaystyle { \frac { 1 } { R } } \sum _ { r = 1 } ^ { R } A _ { r } \mathrm { ~ - ~ } A _ { 0 } } { \displaystyle { \frac { 1 } { R } } \sum _ { r = 1 } ^ { R } n _ { r } } .\tag{34}
$$

Both means run over all five episodes, bankrupt ones pooled in, so a year cut short lowers the numerator and the denominator together. The two means are divided rather than averaged per episode, because an episode that ends near zero after a few hundred calls carries a ratio large enough to dominate an average of five. GPT-5.5 shows what that would cost. Its two bankrupt episodes stopped on day 17 after 200 and 225 calls, ¥10,495 and ¥10,516 in deficit, so those two episodes read -552 and -491 a call. Averaging the five per-episode ratios moves the model from fourth on this dimension to seventeenth. A ratio of two means also has no per-episode spread, so Table 14 prints none for this column.

Tool calls and turns. Two counts describe the same episode at different grain. A call is counted once for every dispatch the run log records, and a turn once for every model message in the transcript. Write ñ and ī for the five-run means of the two. Several calls can leave one turn, since parallel dispatch is enabled and nothing caps a turn but the episode's 4,000-turn budget, so $\bar { n } / \bar { T }$ reads as the batch size a model settled on. A turn that requests nothing pulls the quotient below one.

The eight activity bands. Eighteen tools are too many to read model by model, so the mix collapses them into eight bands that partition the set. Five bands hold one tool each, waiting on the clock (wait\_for\_next\_day), shipping (ship\_orders), listing inventory (publish\_to\_store), moving settled cash out of the platform wallet (withdraw), and bargaining (chatbox). State polling holds the three status reads (check\_balance, check\_store\_status, check\_warehouse), memory holds operate\_memory, and one remainder band holds the other nine, which are search, pricing, promotion, warehouse returns, return tracing, and opening or closing a store. With $\bar { n } _ { t }$ the five-run mean count of tool t, band $g$ takes the share

$$
s _ { g } ~ = ~ { \frac { 1 } { \bar { n } } } \sum _ { t \in g } \bar { n } _ { t } , \qquad \bar { n } ~ = ~ \sum _ { t } \bar { n } _ { t } , \qquad \sum _ { g } s _ { g } ~ = ~ 1 .\tag{35}
$$

Each share is therefore a share of the mean budget rather than a mean of per-episode shares, and ñ recovers the call mean of Equation (34) because the bands leave no tool out.

Context evictions. One pass of the context editor files one record, and an episode's eviction count is the number of those records its transcript carries. Each record also names the turn it fired on and the tokens it released. Nothing else marks the loss, since a pass removes whole message groups and leaves no placeholder behind, so the count and the freed total are the whole of the context-pressure data (§3.2.2).

Memory usage. Memory activity is counted in calls, not in stored notes. One call carries a single action, adding, reading, updating, deleting or listing, so the count measures traffic against a store holding at most 20 entries. Two readings of that traffic appear, the five-run mean count $\bar { n } _ { \mathrm { m e m } }$ and its share $s _ { \mathrm { m e m } } = \bar { n } _ { \mathrm { m e m } } / \bar { n }$ of the same model's calls, the latter being the memory band of Equation (35). A count far above 20 means the agent rewrote slots it had already filled rather than filling new ones

What these numbers do not measure. Four limits bound how far this dimension can be read.

1. Nothing in the economy charges for a call or a token, so the axis prices a cost the primary score cannot see. Only the 600-minute working day constrains the agent from inside the simulation.

2. Waiting on the clock spends no simulated minutes yet still costs one call, so a call count is not a proxy for how much of the working day a model used.

3. A turn that requests no tool still counts as a turn, so a model that deliberates without acting pushes ñ/ī below one, and three such turns in a row end the episode.

4. A truncated episode contributes fewer calls, fewer turns, fewer eviction passes and fewer memory actions to every mean it enters, which is why GPT-5.5 and Qwen3.5-Plus carry the widest call spreads in Table 14.

## E.5 Operations Execution

Selling, shipping and opening stores leave four kinds of trace in an episode, returns, missed dispatches, shipping tiers and store counts. The dimension ranks on the controllable return rate, the expected return points a model's own pricing adds to every unit it ships, and low is the better direction. Seven diagnostics sit beside the axis, the realized return rate, the two return channels the agent does not set, the money returns cost, the on-time share, the orders lost to the dispatch deadline, the tier mix and the store counts.

The four return layers. An item's return rate is built in four steps, three of them when the sale is booked and one when the parcel is dispatched, and every channel of this dimension is a difference between consecutive steps. Write $\theta _ { i } ^ { \mathrm { { \hat { n } a t } } }$ for SKU j's natural rate, $\alpha _ { j }$ for the share of that SKU's cumulative deliveries that came from a quality-downgrading supplier, $r _ { i j t }$ for the price store i charges on day t over the category reference price, and ζ for the piecewise-linear pricing curve through the knees (0.8, 0.85), (1.0, 1.00), (1.3, 1.50) and (1.8, 2.20), held flat outside them.

$$
\begin{array} { r l } { \theta _ { j } ^ { \mathrm { d e f } } = \operatorname* { m i n } \bigl ( 0 . 9 5 , \ \operatorname* { m a x } \bigl ( 0 . 4 0 , \ 2 \theta _ { j } ^ { \mathrm { n a t } } \bigr ) \bigr ) , \qquad } & { \theta _ { j } ^ { ( 1 ) } = ( 1 - \alpha _ { j } ) \theta _ { j } ^ { \mathrm { n a t } } + \alpha _ { j } \theta _ { j } ^ { \mathrm { d e f } } , } \\ { \theta _ { j } ^ { ( 2 ) } = \theta _ { j } ^ { ( 1 ) } \zeta ( r _ { i j t } ) , \qquad } & { \theta _ { j } ^ { ( 3 ) } = \operatorname* { m i n } \bigl ( 0 . 9 5 , \ \theta _ { j } ^ { ( 2 ) } \bigr ) , } \\ { \theta _ { j } = \operatorname* { m i n } \bigl ( 0 . 9 5 , \ \theta _ { j } ^ { ( 3 ) } s \bigr ) , \qquad } & { s \in \{ 0 . 7 5 , \ 1 . 0 0 , \ 1 . 3 0 \} . } \end{array}\tag{36}
$$

The natural rate is the floor the SKU's category sets, the blend charges the share of stock delivered by a quality-downgrading supplier, the curve charges pricing above the reference, and s is the tier factor for fast, standard and slow freight. Constants and composition are the economy's own rule of Equation (8), written here as a sequence because the channels below difference one layer against the next, and with the sale-time ceiling of $\theta _ { j } ^ { ( 3 ) }$ made explicit because the tier channel turns on it.

Accrual at dispatch. Which tier a parcel travels on is unknown until the agent ships it, so all four channels are booked at dispatch rather than at sale. Each earlier layer is carried forward under the same tier factor and the same ceiling as the rate the shipment is actually drawn at.

$$
f ^ { ( k ) } = \mathrm { m i n } \big ( 0 . 9 5 , \ \theta _ { j } ^ { ( k ) } s \big ) , \quad k \in \{ \mathrm { n a t } , 1 , 2 \} , \qquad f = \theta _ { j } ,\tag{37}
$$

writing $\theta _ { i } ^ { ( \mathrm { n a t } ) }$ for $\theta _ { j } ^ { \mathrm { n a t } }$ . A shipment of $q$ units then adds to five running totals, the expected returns themselves and the four channels they split into.

$$
\begin{array} { l l l l } { { \Delta T = q f , } } & { { \Delta N = q f ^ { ( \mathrm { n a t } ) } , } } & { { \Delta D = q \operatorname* { m a x } \bigl ( 0 , \ f ^ { ( 1 ) } - f ^ { ( \mathrm { n a t } ) } \bigr ) , } } \\ { { \Delta P = q \bigl ( f ^ { ( 2 ) } - f ^ { ( 1 ) } \bigr ) , } } & { { \Delta S = q \bigl ( f - f ^ { ( 2 ) } \bigr ) , } } & { { \Delta U = q . } } \end{array}\tag{38}
$$

The pricing term ∆P carries no floor at zero, so pricing under the reference subtracts from it, while the defective term is floored because a blend can only raise a rate.

The year's rates. Every total is divided at year end by the units the episode shipped, $U = \operatorname* { m a x } ( 1 , \sum \Delta U )$ which puts all of them on the same per-shipped-unit scale.

$$
\bar { \theta } _ { \mathrm { e x p } } = \frac { T } { \bar { U } } , \quad \bar { \theta } _ { \mathrm { n a t } } = \frac { N } { \bar { U } } , \quad \bar { \theta } _ { \mathrm { d e f } } = \frac { D } { \bar { U } } , \quad \bar { \theta } _ { \mathrm { c t l } } = \frac { P + S } { \bar { U } } , \qquad \bar { \theta } _ { \mathrm { r e a l } } = \frac { \mathrm { u n i t s ~ r e t u r n e d } } { \operatorname* { m a x } ( 1 , \mathrm { ~ u n i t s ~ s o l d } ) } .\tag{39}
$$

The axis is $\bar { \theta } _ { \mathrm { c t l } }$ , reported in percentage points and averaged unweighted over a model's five episodes. Realized returns are counted the day a parcel comes back, and the two rates differ because a return is drawn when its shipment leaves and arrives 3 to 7 days later, with finalization settling whatever is still in flight (§C.4).

What the tier channel records. The tier factor multiplies both terms of the difference that was meant to isolate it, since $f$ and $f ^ { ( 2 ) }$ scale $\theta _ { j } ^ { ( 2 ) }$ by the same s. Every shipment whose rate stays under the 0.95 ceiling therefore contributes exactly zero to $\Delta S .$ , and the archive field reads 0.0 in 89 of the 90 episodes and -0.0001 in the one where the ceiling clipped one side and not the other. The channels the archive separates are natural, defective and pricing, and $\bar { \theta } _ { \mathrm { c t l } }$ reduces to the pricing channel. Freight speed is not thereby free of consequence. A fast parcel is drawn at three quarters of the standard rate and a slow one at 1.30 times it, an effect that sits inside $f ,$ inside $f ^ { ( 1 ) }$ and inside $f ^ { \mathrm { ( n a t ) } }$ alike, which is why the tier reaches Table 15 as the share of shipments sent fast rather than as a channel.

Fulfillment and cancellation. A sale becomes one order the morning the previous day's demand is turned into shipments, and the agent has two days to dispatch it. Once that deadline passes the shipment is cancelled, its units go back to the warehouse and the revenue is never booked.

$$
\mathrm { O n T i m e } = \frac { \mathrm { o r d e r s ~ s h i p p e d } } { \operatorname* { m a x } ( 1 , \ \mathrm { o r d e r s ~ s o l d } ) } ,\tag{40}
$$

which is one minus the cancelled share and carries no information about how long a dispatch took.   
Cancelled orders are reported as a count beside it.

Ship-speed mix and store churn. Each dispatch names its tier, and the three tallies count shipments rather than units. Fast doubles the freight bill for the reduced draw rate, standard is the default, and slow halves the bill. Store activity is counted twice over, Opened as the openings an episode made against four concurrent slots and one store per type, and $\begin{array} { r } { \mathrm { R e o p e n i s } = \sum _ { \mathrm { t y p e s } } \operatorname* { m a x } ( 0 , \ \tilde { n } _ { \mathrm { t y p e } } - 1 ) } \end{array}$ as the openings that returned to a type already used. Each opening charges the ¥500 setup fee again, and a reopened type starts its reputation at the floor rather than inheriting the goodwill the closed store had earned (§C.3).

What these numbers do not measure. Five limits bound how far this dimension can be read.

1. The axis is an expectation accrued when a parcel leaves rather than a count of units that came back, and it turns negative where a model priced under the reference, as it does in 26 of the 90 episodes.

2. Every rate divides by a count floored at one, so an episode that shipped nothing records 0 rather than nothing. Two GPT-5.5 episodes enter its return means that way.

3. On-time share is one minus the cancelled share of orders sold, so an episode with no sales scores 0 and pulls a five-episode mean far below the years that traded.

4. Refund loss and return freight are gross ¥ totals that grow with units sold, which makes them readable against a model's own sales and not across models of different size.

5. $\alpha _ { j }$ is a ratio over cumulative deliveries pooled by SKU, so the defective channel names the suppliers a model bought from rather than any single lot, and it scores fraud avoidance (§E.2) rather than execution.

## E.6 Learning over the Horizon

Two questions separate a year from a day. Does an agent's next order benefit from the price it already won, and do its later deals go better than its earlier ones? AnchorRatio answers the first and ranks the dimension, since a price already agreed is the kind of numeric anchor that steers an LLM's later offers (Takenami et al., 2025). Around it sit the raw overpayment it normalizes, the rate at which an agent pushes below its own best price, the permutation z-score that carries significance, and three half-year comparisons that read sessions in time order instead of grouping them by supplier. Symbols follow §3.5.5. Session z closes at an agreed unit price $p _ { z }$ on conclusion day $\bar { d } _ { z }$ or at $\dot { f } _ { z } = \bar { \perp }$ when the two sides never agree, and $\mathrm { S E } _ { z }$ is the share of its bargaining range left with the buyer (§E.1)

Sequential price discipline. The anchoring block scores an episode, not a session. Group the honest agreements by their (supplier, SKU) pair $P$ and keep the pairs $\bar { \mathcal P }$ transacted at least twice. Sorting each pair's deals by conclusion day makes every deal after the first a re-order, collected in R. The pair's range $\Delta _ { P } = v _ { j } - c _ { j } ^ { s }$ is one number across its cycles, since the floor belongs to the SKU. Positions here are clipped, unlike $\mathrm { S E } _ { z }$

$$
\pi _ { t } ^ { P } = \operatorname* { m i n } \bigl ( 1 , \operatorname* { m a x } \bigl ( 0 , \big ( p _ { t } - c _ { j } ^ { s } \bigr ) / \Delta _ { P } \bigr ) \bigr ) , \qquad m _ { t } ^ { P } = \operatorname* { m i n } _ { u < t } \pi _ { u } ^ { P } , \qquad t = 2 , \ldots , n _ { P } ,\tag{41}
$$

with $m _ { t } ^ { P }$ the running best price of §3.5.5. Both level statistics pool re-orders across pairs within one episode. Neither sum weights a re-order by order size, so each one is read per unit.

$$
\mathrm { A n c h o r R e g r e t } = \frac { 1 } { | \mathcal { R } | } \sum _ { P \in \mathcal { P } } \sum _ { t = 2 } ^ { n _ { P } } \operatorname* { m a x } \bigl ( 0 , \pi _ { t } ^ { P } - m _ { t } ^ { P } \bigr ) , \qquad | \mathcal { R } | = \sum _ { P \in \mathcal { P } } \bigl ( n _ { P } - 1 \bigr ) ,\tag{42}
$$

$$
\mathrm { N e w L o w } = \frac { 1 } { | \mathcal { R } | } \sum _ { P \in \mathcal { P } } \sum _ { t = 2 } ^ { n _ { P } } \mathbf { 1 } \big [ \pi _ { t } ^ { P } < m _ { t } ^ { P } - \varepsilon \big ] , \qquad \varepsilon = 0 . 0 1 .\tag{43}
$$

The null holds each pair's multiset of positions fixed and reshuffles only their order. Each of B seeded replicates permutes every pair independently by $g _ { P } ^ { ( b ) }$

$$
\mu _ { \mathrm { s h u f f e d } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathrm { A n c h o r R e g r e t } \big ( \{ g _ { P } ^ { ( b ) } \pi ^ { P } \} _ { P \in \mathcal { P } } \big ) , \qquad B = 4 0 0 ,\tag{44}
$$

with $\sigma _ { \mathrm { s h u f f i e d } }$ the population standard deviation of the same B replicates. Write $\mu _ { r }$ and $\sigma _ { r }$ for episode $r _ { \mathrm { } } \mathrm { { s } }$ null moments, and k for the episodes with a repeat pair and a non-degenerate null.

$$
\mathrm { A n c h o r R a t i o } _ { r } = \frac { \mathrm { A n c h o r R e g r e t } _ { r } } { \mu _ { r } } , \qquad z _ { r } = \frac { \mu _ { r } - \mathrm { A n c h o r R e g r e t } _ { r } } { \sigma _ { r } } , \qquad z _ { \mathrm { a n c h o r } } = \frac { 1 } { \sqrt { k } } \sum _ { r = 1 } ^ { k } z _ { r } .\tag{45}
$$

The regret z subtracts the observation from the null because low regret is the good direction. The new-low statistics take the same null and reverse that subtraction, dividing by $\mu _ { r } ^ { \mathrm { n e w } }$ for NewLowRatio and scoring (NewL $\mathrm { o w } _ { r } - \mu _ { r } ^ { \mathrm { n e w } } \big ) / \sigma _ { r } ^ { \mathrm { n e w } }$ , so positive always means an improvement on the agent's own shuffled ordering. Per-episode values combine by Stouffer's rule. A model's AnchorRatio averages its per-episode ratios rather than dividing pooled means.

Four filters decide which repeat purchases the anchoring statistics score. A pair the agent bought from only once tells nothing about whether it bargained better the second time.

1. Fraudulent pairs fall away, since a pre-deal fraudulent floor is inflated by construction and descent toward it is not the same quantity (§3.5.4).

2. An agreement with no recorded conclusion day is dropped.

3. A pair whose range satisfies $\Delta _ { P } \leq 0$ goes with it.

4. A pair transacted once contributes nothing, and an episode holding no repeat pair returns null and drops out of the model mean.

Repeat purchases of the same item from the same supplier are common enough to score. The filters leave 2,230 such pairs over the 90 episodes, and Table 16 gives each model's re-order count.

Half-year comparisons. A second reading orders one episode's sessions by conclusion day and cuts them in half. Write $\mathcal { Z } _ { \mathcal { G } }$ for the honest sessions the episode concluded and $n = | \mathcal { Z } _ { \mathcal { G } } |$ , with $n \geq 4$ required or the episode reports nothing at all. The earlier half $\mathcal { E }$ takes the first $\lfloor n / 2 \rfloor$ sessions in day order and the later half $\mathcal { L }$ the rest, so an odd count leaves the extra session late. A disagreement scores zero surplus rather than dropping out, exactly as $\mathrm { S E ^ { + } }$ scores it.

$$
s _ { z } = \mathbf { 1 } [ f _ { z } \neq \bot ] \mathrm { S E } _ { z } , \qquad h _ { \mathrm { S E } } = \frac { 1 } { | \mathcal { L } | } \sum _ { z \in \mathcal { L } } s _ { z } - \frac { 1 } { | \mathcal { E } | } \sum _ { z \in \mathcal { E } } s _ { z } .\tag{46}
$$

The per-day trend fits the same scores against the day they were recorded on, by least squares over the whole honest set rather than over its halves.

$$
b _ { \mathrm { S E } } = \frac { \sum _ { z \in \mathcal { Z } _ { \mathcal { G } } } ( d _ { z } - \bar { d } ) ( s _ { z } - \bar { s } ) } { \sum _ { z \in \mathcal { Z } _ { \mathcal { G } } } ( d _ { z } - \bar { d } ) ^ { 2 } } ,\tag{47}
$$

with $\bar { d }$ and ā the means over $\mathcal { Z } _ { \mathcal { G } }$ , and the trend set to zero when every session concluded on one day. The fraud half-lift runs over every concluded session instead of the honest ones alone, and cuts that larger set

Table 10: The 18 evaluated models, grouped by whether the weights are released, each given five independent episodes of the 365-day task in one fixed world. Reasoning effort is the requested setting, fixed high where the harness sets the thinking level itself, not sent where thinking is enabled with no effort field, and unset where neither is requested. Sampling is otherwise each provider's default.
<table><tr><td colspan="3">Proprietary</td><td colspan="3">Open-weight</td></tr><tr><td>Model</td><td>Reasoning effort</td><td>Eps.</td><td>Model</td><td>Reasoning effort</td><td>Eps.</td></tr><tr><td>GPT-5.6 Sol (max)</td><td>max</td><td></td><td>Qwen3.8-Max-Preview</td><td>unset</td><td>5</td></tr><tr><td>Fable5 (max)</td><td>max</td><td></td><td>GLM 5.2 (high)</td><td>high</td><td></td></tr><tr><td>GPT-5.5</td><td>xhigh</td><td>55555555</td><td>Kimi K3</td><td>unset</td><td>55</td></tr><tr><td>Claude Opus 4.8 (max)</td><td>max</td><td></td><td>GLM 5.1</td><td>not sent</td><td>55</td></tr><tr><td>Claude Opus 4.7 (max)</td><td>max</td><td></td><td>DeepSeek-V4-Pro-Preview (max)</td><td>max</td><td></td></tr><tr><td>Claude Opus 4.6 (max)</td><td>max</td><td></td><td>Qwen3.7-Max</td><td>unset</td><td>5</td></tr><tr><td>Gemini 3.5 Flash</td><td>fixed high</td><td></td><td>GLM 5.2 (max)</td><td>max</td><td>5</td></tr><tr><td>Gemini 3.1 Pro</td><td>fixed high</td><td></td><td>Kimi K2.6</td><td>unset</td><td>5</td></tr><tr><td></td><td></td><td></td><td>Qwen3.6-Plus</td><td>unset</td><td>5</td></tr><tr><td></td><td></td><td></td><td>Qwen3.5-Plus</td><td>unset</td><td>5</td></tr></table>

at its own median day, so its halves are not the halves above. Writing $\mathcal { Z } _ { B }$ for the fraudulent sessions and $\beta$ for the share of a half that ended in a deal with one of them,

$$
\beta ( S ) = \frac { \big | \{ z \in \mathcal { S } \cap \mathcal { Z } _ { \mathcal { B } } \mid f _ { z } \ne \bot \} \big | } { | \mathcal { S } | } , \qquad h _ { \mathrm { f r a u d } } = \beta ( \mathcal { E } _ { A } ) - \beta ( \mathcal { L } _ { A } ) .\tag{48}
$$

Subtracting the later half from the earlier reverses the order of Equation (46), which leaves positive as the improving direction in all three measures.

What these measures do and do not carry. Four properties bound how the columns of Table 16 can be read. Anchoring spreads are population standard deviations while every other archive spread is a sample one, so the two kinds are not compared. An episode with no repeat pair returns nothing and drops out of its model's anchoring mean, which leaves GPT-5.5 with three episodes rather than five. A trend fitted inside a truncated episode spans a window of days rather than a year, so one bankrupt run can set a model's mean slope. Neither half-lift is scaled by episode length, both being differences of within-episode means, and a busy year therefore counts no more than a sparse one.

The axis sits on AnchorRatio because of how sharply each candidate orders the field. Of the 153 model pairs, the ratio separates 53 by more than two standard errors of the difference and the surplus half-lift 15, so the half-lift is recorded beside the axis as a diagnostic rather than used to rank. Both are reported for every model, together with the raw regret the ratio normalizes, since a model whose quotes cluster tightly earns a low raw regret without ever holding a price it won.

## F Per-Dimension Results

Section 4 reports one headline per dimension over a handful of models. The measurements behind those selections are collected here for all 18, one table per dimension, each sorted by mean end-of-year total assets so a row keeps its place from one dimension to the next. Formulas and archived field names sit in Appendix E.

## F.1 Per-Episode Asset Trajectories

Equal five-episode means hide unequal spreads. Figure 21 overlays eight mean curves, and Figure 22 draws all 90 episodes separately. Episodes finish on both sides of the ¥100,000 stake in 8 of the 18 panels. Among the 14 models that never went bankrupt, the best episode beats the worst by a median factor of 2.76, from 57.0 for Qwen3.7-Max down to 1.55 for Kimi K2.6. Once episodes start exiting, a panel's mean curve becomes conditional on survival and rises for GPT-5.5.

## F.2 Negotiation Quality

Section 4.3.1 quotes the two ends of the surplus range and leaves the per-model values here. Table 11 carries all four bargaining scores with rounds to deal and deals closed beside them. Two figures then split the same agreements by counterpart template, Figure 23 on the surplus each template gave up and Figure 24 on the money each one received.

(a) Eight of 18 Models, 5 Episodes Each  
![](images/0cc59bb6e1c60d035aa9088af589dc7d45f86f2b2883bb73ab0387227223f933.jpg)

![](images/31918cea61f8fa6e99a37bced396b06f4f09e85377ef4491fcf1be366bed858e.jpg)  
Figure 21: Total assets over the year for eight models spanning the ranking. The solid line is the mean over episodes still running, the band their min-to-max range, and a red cross marks a bankruptcy. The line turns dashed once fewer than five episodes remain, so the mean there counts survivors only.

Table 11: Negotiation quality per model, 18 models and five episodes each, sorted by mean final assets, mean and standard deviation over the episodes, $\mathrm { w i t h ~ } ^ { \dag }$ on open-weight models. $\mathrm { C S E ^ { + } }$ is the share of the bargaining range the buyer keeps on the honest deals it closes and ranks this dimension, SE+ scores a walk-away as zero, %Oracle compares realized surplus against closing every honest deal at the supplier's floor, and $\mathrm { A G R ^ { + } }$ is the share of concluded honest sessions that ended in agreement. All four count honest suppliers only and drop sessions abandoned mid-bargain. Rounds to deal and deals closed cover fraudulent counterparts too and are volume readings rather than scores, which is why neither column is marked. Definitions sit in $\ S \mathrm { E } . 1$ , and truncated bankrupt episodes are pooled into every mean. $^ \ddagger \mathrm { G P T - 5 . 5 ^ { \prime } s }$ spreads are population estimates, the rest sample estimates.
<table><tr><td rowspan=3 colspan=8>Value Extraction and Agreement                    VolumeRounds       DealsModel                                $\mathrm { S E ^ { + } }$          $\mathrm { C S E ^ { + } }$    $\% \mathrm { O r a c l e }$        $\mathrm { A G R ^ { + } }$ </td></tr><tr><td rowspan=1 colspan=2>RoundsDeals</td></tr><tr><td rowspan=1 colspan=2>to DealClosed</td></tr><tr><td rowspan=1 colspan=3> $\mathrm { G P T - 5 . 6 ~ S o l ~ ( m a x ) }$           $0 . 6 7 1 \pm 0 . 0 4 7$ </td><td rowspan=1 colspan=1> $0 . 6 7 2 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td rowspan=1 colspan=1> $6 7 . 0 \pm 4 . 3$ </td><td rowspan=1 colspan=1> $0 . 9 9 8 \pm 0 . 0 0 4$ </td><td rowspan=1 colspan=2> $3 . 8 4 \pm 0 . 3 0$   $2 8 2 . 4 \pm 1 2 1 . 8$ </td></tr><tr><td rowspan=1 colspan=1>Fable5 (max)</td><td rowspan=1 colspan=2> $0 . 7 6 4 \pm 0 . 0 7 8$ </td><td rowspan=1 colspan=1> $0 . 7 7 2 { \scriptstyle \pm 0 . 0 7 6 }$ </td><td rowspan=1 colspan=1> $7 7 . 1 \pm 7 . 8$ </td><td rowspan=1 colspan=1> $0 . 9 9 0 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td rowspan=1 colspan=1> $4 . 0 7 \pm 0 . 6 7$ </td><td rowspan=1 colspan=1> $1 7 7 . 2 \pm 6 6 . 7$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { G P T - 5 . { \dot { 5 } } ^ { \ddagger } }$ </td><td rowspan=1 colspan=2> $0 . 6 9 3 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td rowspan=1 colspan=1> $0 . 7 0 0 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td rowspan=1 colspan=1> $6 9 . 2 \pm 5 . 4$ </td><td rowspan=1 colspan=1> $0 . 9 9 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td rowspan=1 colspan=1> $3 . 3 1 \pm 0 . 2 8$ </td><td rowspan=1 colspan=1> $2 8 7 . 8 \pm 2 3 0 . 7$ </td></tr><tr><td rowspan=1 colspan=3>Claude Opus 4.8 (max)      $0 . 6 5 6 \pm 0 . 0 4 3$ </td><td rowspan=1 colspan=1> $0 . 6 6 2 \pm 0 . 0 4 0$ </td><td rowspan=1 colspan=1> $6 7 . 2 \pm 4 . 2$ </td><td rowspan=1 colspan=1> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td rowspan=1 colspan=1> $3 . 3 8 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1> $2 2 1 . 4 \pm 6 2 . 9$ </td></tr><tr><td rowspan=1 colspan=3> $\mathrm { Q w e n 3 . 8 – M a x  – P r e v i e w } ^ { \dagger }$      $0 . 7 1 3 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td rowspan=1 colspan=1> $0 . 7 1 3 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td rowspan=1 colspan=1> $7 1 . 6 \pm 2 . 8$ </td><td rowspan=1 colspan=1>_ $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td rowspan=1 colspan=2> $3 . 5 2 \pm 0 . 1 0$    $2 3 0 . 4 \pm 2 3 . 8$ </td></tr><tr><td rowspan=1 colspan=1>GLM $5 . 2 ( \mathrm { h i g h } ) ^ { \dagger }$ </td><td rowspan=1 colspan=2> $0 . 6 9 2 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td rowspan=1 colspan=1> $0 . 6 9 3 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td rowspan=1 colspan=1> $6 8 . 9 \pm 5 . 9$ </td><td rowspan=1 colspan=1> $0 . 9 9 9 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td rowspan=1 colspan=2> $3 . 5 1 \pm 0 . 1 7$    $1 7 9 . 0 \pm 4 9 . 9$ </td></tr><tr><td rowspan=1 colspan=2>Kimi K3†</td><td rowspan=1 colspan=1> $0 . 6 3 2 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td rowspan=1 colspan=1> $0 . 6 3 2 \pm 0 . 0 3 9$ </td><td rowspan=1 colspan=1> $6 3 . 1 \pm 3 . 5$ </td><td rowspan=1 colspan=1>_ $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td rowspan=1 colspan=2> $3 . 3 3 \pm 0 . 0 5$    $2 0 0 . 4 \pm 9 6 . 2$ </td></tr><tr><td rowspan=1 colspan=2>Claude Opus 4.7 (max)    _</td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 7 7 \pm 0 . 0 9 6 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 8 1 1 \pm 0 . 0 9 7 }$ </td><td rowspan=1 colspan=1> ${ \bf 7 7 . 4 \pm 1 0 . 2 }$ </td><td rowspan=1 colspan=1> $0 . 9 6 0 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td rowspan=1 colspan=2> $3 . 8 8 \pm 0 . 3 2$     $7 4 . 4 \pm 2 8 . 4$ </td></tr><tr><td rowspan=1 colspan=2>Claude Opus 4.6 (max)</td><td rowspan=1 colspan=1> $0 . 6 4 9 \pm 0 . 0 5 0$ </td><td rowspan=1 colspan=1> $0 . 6 4 9 \pm 0 . 0 5 0$ </td><td rowspan=1 colspan=1> $6 4 . 9 \pm 4 . 0$ </td><td rowspan=1 colspan=1>_ $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td rowspan=1 colspan=2> $3 . 5 4 \pm 0 . 1 2$     $1 1 5 . 8 \pm 9 . 1$ </td></tr><tr><td rowspan=1 colspan=2> $\mathrm { G L M ~ 5 . 1 ^ { \dagger } }$ </td><td rowspan=1 colspan=1> $0 . 6 5 7 \pm 0 . 0 3 4$ </td><td rowspan=1 colspan=1> $0 . 6 6 2 \pm 0 . 0 3 1$ </td><td rowspan=1 colspan=1> $6 5 . 4 \pm 4 . 6$ </td><td rowspan=1 colspan=1> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td rowspan=1 colspan=2> $3 . 3 7 \pm 0 . 1 2$    $1 3 1 . 2 \pm 4 5 . 1 $ </td></tr><tr><td rowspan=1 colspan=3>DeepSeek-V4-Pro-Preview $0 . 6 5 1 \pm 0 . 0 6 1$  $( \mathrm { m a x } ) ^ { \dagger }$ </td><td rowspan=1 colspan=1> $0 . 6 5 4 \pm 0 . 0 6 4$ </td><td rowspan=1 colspan=1> $6 5 . 4 \pm 5 . 7$ </td><td rowspan=1 colspan=1> $0 . 9 9 7 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td rowspan=1 colspan=2> $3 . 4 3 \pm 0 . 2 1$    $1 2 9 . 2 \pm 5 2 . 2$ </td></tr><tr><td rowspan=1 colspan=2> $\mathrm { \dot { G } e m i n i 3 . 5 ~ F l a s h }$ </td><td rowspan=1 colspan=1> $0 . 7 6 3 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td rowspan=1 colspan=1> $0 . 7 7 7 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td rowspan=1 colspan=1> $7 6 . 6 \pm 3 . 7$ </td><td rowspan=1 colspan=1> $0 . 9 8 3 \pm 0 . 0 3 7$ </td><td rowspan=1 colspan=2> $3 . 9 4 \pm 0 . 2 1$     $6 4 . 8 \pm 2 8 . 6$ </td></tr><tr><td rowspan=1 colspan=2> $\mathrm { Q w e n 3 . 7 – M a x ^ { \dagger } }$ </td><td rowspan=1 colspan=1> $0 . 6 1 6 \pm 0 . 0 2 9$ </td><td rowspan=1 colspan=1> $0 . 6 1 6 \pm 0 . 0 2 9$ </td><td rowspan=1 colspan=1> $6 0 . 4 \pm 3 . 4$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td rowspan=1 colspan=2> $3 . 3 6 \pm 0 . 1 3$    $1 0 2 . 8 \pm 1 3 . 8$ </td></tr><tr><td rowspan=1 colspan=2> $\mathrm { \dot { G e m i n i } 3 . 1 P r o }$ </td><td rowspan=1 colspan=1> $0 . 6 5 7 \pm 0 . 0 4 9$ </td><td rowspan=1 colspan=1> $0 . 6 6 0 \pm 0 . 0 4 8$ </td><td rowspan=1 colspan=1> $6 7 . 4 \pm 6 . 3$ </td><td rowspan=1 colspan=1> $0 . 9 9 6 \pm 0 . 0 0 8$ </td><td rowspan=1 colspan=1> $3 . 9 3 \pm 0 . 7 0$ </td><td rowspan=1 colspan=1> $7 5 . 6 \pm 2 6 . 5$ </td></tr><tr><td rowspan=1 colspan=3> $\mathrm { G L M } 5 . 2 \ : ( \mathrm { m a x } ) ^ { \dagger }$              $0 . 6 3 0 { \scriptstyle \pm 0 . 0 7 3 }$ </td><td rowspan=1 colspan=1> $0 . 6 3 2 { \scriptstyle \pm 0 . 0 7 2 }$ </td><td rowspan=1 colspan=1> $6 3 . 3 \pm 6 . 6$ </td><td rowspan=1 colspan=1> $0 . 9 9 7 \pm 0 . 0 0 7$ </td><td rowspan=1 colspan=2> $3 . 3 7 \pm 0 . 0 7$    $1 9 8 . 8 \pm 1 0 . 4$ </td></tr><tr><td rowspan=1 colspan=2> $\mathrm { K i m i ~ K 2 . 6 ^ { \dagger } }$ </td><td rowspan=1 colspan=1> $0 . 5 9 6 \pm 0 . 0 6 5$ </td><td rowspan=1 colspan=1> $0 . 5 9 6 { \scriptstyle \pm 0 . 0 6 5 }$ </td><td rowspan=1 colspan=1> $6 0 . 1 \pm 5 . 2$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td rowspan=1 colspan=1> $3 . 3 2 \pm 0 . 0 7$ </td><td rowspan=1 colspan=1> $4 8 . 6 \pm 1 8 . 1$ </td></tr><tr><td rowspan=1 colspan=2> $\mathrm { Q w e n 3 . 6 - P l u s ^ { \dagger } }$ </td><td rowspan=1 colspan=1> $0 . 6 2 5 \pm 0 . 0 7 0$ </td><td rowspan=1 colspan=1> $0 . 6 2 5 \pm 0 . 0 7 0$ </td><td rowspan=1 colspan=1> $6 1 . 4 \pm 5 . 4$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td rowspan=1 colspan=2> $3 . 3 5 \pm 0 . 1 3$     $7 4 . 4 \pm 1 8 . 4$ </td></tr><tr><td rowspan=1 colspan=2> $\mathrm { Q w e n 3 . 5 - P l u s ^ { \dagger } }$ </td><td rowspan=1 colspan=1> $0 . 6 1 8 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td rowspan=1 colspan=1> $0 . 6 2 5 \pm 0 . 0 4 5$ </td><td rowspan=1 colspan=1> $6 1 . 4 \pm 8 . 0$ </td><td rowspan=1 colspan=1> $0 . 9 8 8 \pm 0 . 0 2 7$ </td><td rowspan=1 colspan=2> $3 . 3 8 \pm 0 . 1 1$      $3 1 . 4 \pm 9 . 9$ </td></tr></table>

Counterpart difficulty moves an outcome less than the agent's own policy does. Inside a single model, surplus varies by a median 0.093 over the templates it met at least twenty times. That span exceeds the 0.051 between ExPRESsIVE and ADVERSARIAL, the field's easiest and hardest presets (Table 8), and falls short of the 0.215 between the best and worst per-model $\mathrm { C S E ^ { + } }$ (§4.3.1). Fraudulent counterparts are held apart in the figure, since they carry the same ADVERSARIAL label as the hardest honest template while conceding on different parameters (Appendix D.3). Measured surplus against them is lower for 16 of the

![](images/f9ab758c16ed8e5efc55ce4ed005aa1218868b9961f23f9a7e2b1de6e2ee90a9.jpg)  
Figure 22: Total assets over the year for all 18 models, ordered by their mean at the end of the year. Each thin line is one episode and the black line their mean while they are still running. The dashed red line marks the ¥100,000 stake and a red cross marks a bankruptcy.

18 models, an arithmetic effect of scoring every deal against the honest cost floor (§3.5.4, Equation (20)).

## F.3 Fraud Avoidance

Two figures already carry the per-model fraud values, so Table 12 holds what they leave out. The columns are the cash that reached fraudulent suppliers, the orders placed with them and the membership fees paid. Figure 25 draws BadSpend% and the split of that money over the five scams. Figure 26 separates the outreach stage from the order stage, where the median model turns 22.0% of the fraudulent suppliers it approached into suppliers it bought from.

Outreach stays narrow whichever suppliers it targets. Over a whole year the widest honest search touched 14.3% of the 424 honest names and the narrowest 2.4% (§B.4). Every screening decision therefore ran over a narrow slice the model had picked out itself.

## F.4 Cash Flow and Solvency

Table 13 sets the normalized fall that ranks solvency beside the raw fall it is computed from, with the trough, the overdrawn mornings, the idle wallet, the largest warehouse and the bankrupt runs. Figure 27 plots that ratio against end-of-year total assets and the idle wallet against the overdrawn days. Both read the daily record whose columns and conventions are set out in §E.3.

<table><tr><td rowspan=1 colspan=1>1. GPT-5.6 Sol</td><td rowspan=1 colspan=1>0.721369</td><td rowspan=1 colspan=1>0.606101</td><td rowspan=1 colspan=1>0.645238</td><td rowspan=1 colspan=1>0.639199</td><td rowspan=1 colspan=1>0.648226</td><td rowspan=1 colspan=1>0.59058</td><td rowspan=1 colspan=1>0.6621191</td><td rowspan=1 colspan=1>0.618221</td></tr><tr><td rowspan=2 colspan=1>2. Fable5</td><td rowspan=2 colspan=1>0.732301</td><td rowspan=2 colspan=1>0.858136</td><td rowspan=2 colspan=1>0.758144</td><td rowspan=2 colspan=1>0.732126</td><td rowspan=1 colspan=1>0.728</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.756</td><td rowspan=1 colspan=1>0.678</td></tr><tr><td rowspan=1 colspan=1>152</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>871</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=2 colspan=1>3. GPT-5.5</td><td rowspan=2 colspan=1>0.654289</td><td rowspan=2 colspan=1>0.683133</td><td rowspan=2 colspan=1>0.676370</td><td rowspan=1 colspan=1>0.673</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.686</td></tr><tr><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>148</td><td rowspan=1 colspan=1>109</td><td rowspan=1 colspan=1>1194</td><td rowspan=1 colspan=1>245</td></tr><tr><td rowspan=1 colspan=1>4. Claude Opus 4.8</td><td rowspan=1 colspan=1>0.610252</td><td rowspan=1 colspan=1>0.679348</td><td rowspan=1 colspan=1>0.705119</td><td rowspan=1 colspan=1>0.650145</td><td rowspan=1 colspan=1>0.640170</td><td rowspan=1 colspan=1>0.53618</td><td rowspan=1 colspan=1>0.6531052</td><td rowspan=1 colspan=1>0.60655</td></tr><tr><td rowspan=1 colspan=1>5. Qwen3.8-Max-Preview</td><td rowspan=1 colspan=1>0.728227</td><td rowspan=1 colspan=1>0.703375</td><td rowspan=1 colspan=1>0.729191</td><td rowspan=1 colspan=1>0.743105</td><td rowspan=1 colspan=1>0.680192</td><td rowspan=1 colspan=1>no deal</td><td rowspan=1 colspan=1>0.7131090</td><td rowspan=1 colspan=1>0.65362</td></tr><tr><td rowspan=1 colspan=1>6. GLM 5.2 (high)</td><td rowspan=1 colspan=1>0.733146</td><td rowspan=1 colspan=1>0.686265</td><td rowspan=1 colspan=1>0.707180</td><td rowspan=1 colspan=1>0.606124</td><td rowspan=1 colspan=1>0.67877</td><td rowspan=1 colspan=1>0.62962</td><td rowspan=1 colspan=1>0.682854</td><td rowspan=1 colspan=1>0.57841</td></tr><tr><td rowspan=1 colspan=1>7. Kimi K3</td><td rowspan=1 colspan=1>0.650208</td><td rowspan=1 colspan=1>0.63885</td><td rowspan=1 colspan=1>0.626158</td><td rowspan=1 colspan=1>0.621192</td><td rowspan=1 colspan=1>0.592285</td><td rowspan=1 colspan=1>(0.779)2</td><td rowspan=1 colspan=1>0.621930</td><td rowspan=1 colspan=1>0.56372</td></tr><tr><td rowspan=1 colspan=1>8. Claude Opus 4.7</td><td rowspan=1 colspan=1>0.76395</td><td rowspan=1 colspan=1>0.814141</td><td rowspan=1 colspan=1>0.74552</td><td rowspan=1 colspan=1>0.76232</td><td rowspan=1 colspan=1>0.76820</td><td rowspan=1 colspan=1>0.81931</td><td rowspan=1 colspan=1>0.785371</td><td rowspan=1 colspan=1>(0.768)1</td></tr><tr><td rowspan=1 colspan=1>9. Claude Opus 4.6</td><td rowspan=1 colspan=1>0.716162</td><td rowspan=1 colspan=1>0.662128</td><td rowspan=1 colspan=1>0.55124</td><td rowspan=1 colspan=1>(0.569)1</td><td rowspan=1 colspan=1>0.605120</td><td rowspan=1 colspan=1>0.60380</td><td rowspan=1 colspan=1>0.651515</td><td rowspan=1 colspan=1>0.56264</td></tr><tr><td rowspan=1 colspan=1>10. GLM 5.1</td><td rowspan=1 colspan=1>0.676156</td><td rowspan=1 colspan=1>0.676235</td><td rowspan=1 colspan=1>0.66278</td><td rowspan=1 colspan=1>0.61774</td><td rowspan=1 colspan=1>0.61488</td><td rowspan=1 colspan=1>no deal</td><td rowspan=1 colspan=1>0.659631</td><td rowspan=1 colspan=1>0.57025</td></tr><tr><td rowspan=1 colspan=1>11. DeepSeek-V4-Pro-Preview</td><td rowspan=1 colspan=1>0.672142</td><td rowspan=1 colspan=1>0.610201</td><td rowspan=1 colspan=1>0.63339</td><td rowspan=1 colspan=1>0.609124</td><td rowspan=1 colspan=1>0.69861</td><td rowspan=1 colspan=1>0.65340</td><td rowspan=1 colspan=1>0.637607</td><td rowspan=1 colspan=1>0.63939</td></tr><tr><td rowspan=1 colspan=1>12. Gemini 3.5 Flash</td><td rowspan=1 colspan=1>0.79145</td><td rowspan=1 colspan=1>0.765174</td><td rowspan=1 colspan=1>0.76721</td><td rowspan=1 colspan=1>(0.777)9</td><td rowspan=1 colspan=1>0.75061</td><td rowspan=1 colspan=1>(0.815)4</td><td rowspan=1 colspan=1>0.767314</td><td rowspan=1 colspan=1>0.60010</td></tr><tr><td rowspan=1 colspan=1>13. Qwen3.7-Max</td><td rowspan=1 colspan=1>0.67038</td><td rowspan=1 colspan=1>0.63656</td><td rowspan=1 colspan=1>0.60415</td><td rowspan=1 colspan=1>0.621135</td><td rowspan=1 colspan=1>0.611183</td><td rowspan=1 colspan=1>0.55143</td><td rowspan=1 colspan=1>0.616470</td><td rowspan=1 colspan=1>0.57144</td></tr><tr><td rowspan=1 colspan=1>14. Gemini 3.1 Pro</td><td rowspan=1 colspan=1>0.68070</td><td rowspan=1 colspan=1>0.73931</td><td rowspan=1 colspan=1>0.68351</td><td rowspan=1 colspan=1>0.62864</td><td rowspan=1 colspan=1>0.627100</td><td rowspan=1 colspan=1>(0.670)8</td><td rowspan=1 colspan=1>0.659324</td><td rowspan=1 colspan=1>0.56354</td></tr><tr><td rowspan=1 colspan=1>15. GLM 5.2 (max)</td><td rowspan=1 colspan=1>0.675175</td><td rowspan=1 colspan=1>0.627398</td><td rowspan=1 colspan=1>0.640104</td><td rowspan=1 colspan=1>0.63042</td><td rowspan=1 colspan=1>0.647131</td><td rowspan=1 colspan=1>0.58454</td><td rowspan=1 colspan=1>0.638904</td><td rowspan=1 colspan=1>0.54190</td></tr><tr><td rowspan=1 colspan=1>16. Kimi K2.6</td><td rowspan=1 colspan=1>0.54960</td><td rowspan=1 colspan=1>0.62933</td><td rowspan=1 colspan=1>0.51334</td><td rowspan=1 colspan=1>0.63234</td><td rowspan=1 colspan=1>0.56128</td><td rowspan=1 colspan=1>0.66816</td><td rowspan=1 colspan=1>0.581205</td><td rowspan=1 colspan=1>0.55138</td></tr><tr><td rowspan=1 colspan=1>17. Qwen3.6-Plus</td><td rowspan=1 colspan=1>0.70735</td><td rowspan=1 colspan=1>0.665124</td><td rowspan=1 colspan=1>0.59640</td><td rowspan=1 colspan=1>0.77912</td><td rowspan=1 colspan=1>0.576116</td><td rowspan=1 colspan=1>no deal</td><td rowspan=1 colspan=1>0.634327</td><td rowspan=1 colspan=1>0.53845</td></tr><tr><td rowspan=1 colspan=1>18. Qwen3.5-Plus</td><td rowspan=1 colspan=1>0.68224</td><td rowspan=1 colspan=1>0.64152</td><td rowspan=1 colspan=1>0.61620</td><td rowspan=1 colspan=1>0.57612</td><td rowspan=1 colspan=1>0.63414</td><td rowspan=1 colspan=1>0.59115</td><td rowspan=1 colspan=1>0.633137</td><td rowspan=1 colspan=1>0.53620</td></tr><tr><td rowspan=1 colspan=1>field, deal-weighted</td><td rowspan=1 colspan=1>0.6902794</td><td rowspan=1 colspan=1>0.6843016</td><td rowspan=1 colspan=1>0.6771878</td><td rowspan=1 colspan=1>0.6511575</td><td rowspan=1 colspan=1>0.6432172</td><td rowspan=1 colspan=1>0.639552</td><td rowspan=1 colspan=1>0.67011987</td><td rowspan=1 colspan=1>0.6091141</td></tr></table>

upper number is the deal-count weighted mean, lower number the pooled deal count parentheses mark a cell with fewer than 10 deals, colour diverges about the all-honest mean 0.670

Figure 23: How much of each bargaining range models capture against each kind of supplier, over all 13,128 concluded agreements. Columns run from the easiest counterpart to the hardest, the deal count sits under each cell, and values in parentheses rest on fewer than 10 deals.

## F.5 Operational Efficiency

Table 14 prints the turns and calls an episode took, the profit each call returned, the eviction passes the transcript needed, the memory traffic, and how a model's calls split over the eight activity bands of Figure 10b. Section 4.3.4 reads the field-wide shares, and the rows here show how far a single policy sits from them.

## F.6 Operations Execution

Table 15 carries the four scored columns of the dimension together with the return channels the agent does not set, the refund bill, the shipping tier mix and the store counts. Returns accrue when a parcel is dispatched rather than when a buyer sends something back, so every rate column is an expectation over units shipped (§E.5).

Almost no model keeps the four-store cap filled. Only GPT-5.6 Sol, GLM 5.2 (high) and Qwen3.8-Max-Preview average more than 3.5 stores trading per day. Cumulative openings still run past the cap for most of the field, since every opening beyond it replaces a store (§4.3.5)

## F.7 Learning over the Horizon

Table 16 holds the anchoring block with its permutation null on the left and the three half-year comparisons on the right. The left block groups an episode's deals by supplier and SKU, while the right block splits the same episode at its median day. A model can therefore improve on one reading and not on the other.

(a) Honest Spend per Model, with the Total Beside Each Bar  
![](images/3431608f6664a9ca2a12ed0370d30d4dd360da1ffa8f48cd007e7a4ea38dc5c1.jpg)  
share of that model's own honest procurement spend

(b) Field Share against Supplier-List Share  
![](images/29024f8fa4fab69962a4eb13707a79bcff3be3b32ced3aeeaceb060f9a63b135.jpg)

Figure 24: Honest procurement money split over the six counterpart behavior templates, which run easiest to hardest on the field-wide surplus order of Figure 23. Panel (a) gives every model's five-run mean spend with honest suppliers as a share of its own honest total, that total printed at the right of each bar, models in descending total assets. Panel (b) pools the 90 episodes, ¥55.1M paid to honest suppliers, and sets the share each template received against the share of the 424 honest names that carry it, so a bar reaching past its mark was bought from more heavily than the supplier list alone would give. The hardest template takes 4.8% of the spend against 11.3% of the names and the easiest 18.5% against 24.1%. Spend with fraudulent suppliers is recorded separately and stays out of both panels.

## F.8 Profiles and Rank Crossings Across Dimensions

Every model is strong on some dimensions and weak on others. Figure 28 keeps the normalization of Figure 9 across all 18 panels. Between best and worst rank, a model spans 10.2 of 18 places on average over the seven axes. No model ranks in the top five on all seven, and Qwen3.8-Max-Preview comes closest at a worst place of 7.

Some pairs of models trade places on almost every axis while others hold their order. Figure 29 converts the same measurements into ranks and adds the money-weighted %Oracle and the sequential-discipline AnchorRatio of §4.3.1. Over the seven transitions and 153 pairs, 311 orderings reverse. Operations execution moves an agent furthest from its asset rank, 4.2 places on average, while operational efficiency barely moves it at all, 0.7 places, so the axis that prices a year against its tool calls repeats what the balance sheet already said.

## G Failure Case Studies and Detection Rules

Three of the five scams get an annotated transcript here. The membership fee comes first, then the two families that take the most fraudulent spend, short delivery and the defective lot (§4.3.2). Manufactured urgency and the promised future discount work as §3.5.4 describes them. Excerpts keep prices, dates and wording verbatim, elide turns at $I . . . J ,$ and print the episode's own records in gray. One bankruptcy is then traced day by day, and the rules behind the incidence matrix close the appendix.

Table 12: Fraud avoidance, 18 models and five episodes each, sorted by mean end-of-year assets, † marking open-weight models. The first four columns are means over the five episodes with their sample standard deviation, spend and orders counting every purchase order that cleared the bank account and suppliers ordered from counting distinct names within an episode. The two right-hand columns are totals over the five episodes, three payments of ¥1,000 in the whole evaluation, all of them from Gemini 3.5 Flash and Qwen3.5-Plus. BadSpend% ranks the dimension and is drawn per model in Figure 25 rather than repeated here. Definitions sit in §E.2.
<table><tr><td rowspan="2"></td><td colspan="4">Cash and Orders Reaching Fraudulent Suppliers</td><td colspan="2">Membership Fees Paid</td></tr><tr><td>spend ¥k</td><td>fraudulent orders</td><td>all orders</td><td>suppliers ordered from</td><td>payments</td><td>夫</td></tr><tr><td>GPT-5.6 Sol (max)</td><td>689.1 ±376.0</td><td>44.2 ±26.4</td><td>282.4 ±121.8</td><td>5.2 ±1.3</td><td>0</td><td>0</td></tr><tr><td>Fable5 (max)</td><td>49.7 ±89.2</td><td>3.0 ±5.6</td><td>177.2 ±66.7</td><td>0.8 ±0.8</td><td>0</td><td>0</td></tr><tr><td>GPT-5.5</td><td>535.6 ±504.0</td><td>49.0 ±50.2</td><td>287.8 ±230.7</td><td>5.6 ±3.7</td><td>0</td><td>0</td></tr><tr><td>Claude Opus 4.8 (max)</td><td>80.3±156.9</td><td>11.0 ±10.4</td><td>221.4 ±62.9</td><td>1.8 ±1.5</td><td>0</td><td>0</td></tr><tr><td>Qwen3.8-Max-Preview†</td><td>57.8 ±78.3</td><td>12.4±9.7</td><td>230.4 ±23.8</td><td>1.4 ±0.5</td><td>0</td><td>0</td></tr><tr><td>GLM 5.2 (high)†</td><td>10.6 ±17.1</td><td>8.2 ±13.4</td><td>179.0 ±49.9</td><td>1.0 ±0.7</td><td>0</td><td>0</td></tr><tr><td>Kimi K3†</td><td>23.4 ±30.3</td><td>14.4 ±28.4</td><td>200.4±96.2</td><td>1.0 ±1.2</td><td>0</td><td>0</td></tr><tr><td>Claude Opus 4.7 (max)</td><td>0.4±0.9</td><td>0.2 ±0.4</td><td>74.4 ±28.4</td><td>0.2 ±0.4</td><td>0</td><td>0</td></tr><tr><td>Claude Opus 4.6 (max)</td><td>164.9 ±229.2</td><td>12.8 ±17.2</td><td>115.8 ±9.1</td><td>0.8 ±0.8</td><td>0</td><td>0</td></tr><tr><td>GLM 5.1†</td><td>4.7 ±7.2</td><td>5.0 ±7.5</td><td>131.2 ±45.1</td><td>0.4 ±0.5</td><td>0</td><td>0</td></tr><tr><td>DeepSeek-V4-Pro-Preview (max)†</td><td>23.3 ±39.0</td><td>7.8 ±10.8</td><td>129.2 ±52.2</td><td>0.8 ±0.8</td><td>0</td><td>0</td></tr><tr><td>Ġemini 3.5 Flash</td><td>8.4±16.7</td><td>2.4 ±2.2</td><td>65.2 ±28.9</td><td>1.6 ±0.9</td><td>2</td><td>2,000</td></tr><tr><td>Qwen3.7-Max†</td><td>67.4±150.8</td><td>8.8±19.7</td><td>102.8 ±13.8</td><td>0.2 ±0.4</td><td>0</td><td>0</td></tr><tr><td>Gemini 3.1 Pro</td><td>89.4 ±93.9</td><td>10.8 ±10.0</td><td>75.6 ±26.5</td><td>3.0 ±1.2</td><td>0</td><td>0</td></tr><tr><td>GLM 5.2 (max)†</td><td>46.9 ±40.1</td><td>18.0 ±17.8</td><td>198.8 ±10.4</td><td>1.0 ±0.7</td><td>0</td><td>0</td></tr><tr><td>Kimi K2.6†</td><td>14.2 ±19.6</td><td>7.6 ±10.4</td><td>48.6 ±18.1</td><td>0.8 ±1.1</td><td>0</td><td>0</td></tr><tr><td>Qwen3.6-Plus†</td><td>12.1 ±16.1</td><td>9.0 ±13.7</td><td>74.4 ±18.4</td><td>0.8 ±0.8</td><td>0</td><td>0</td></tr><tr><td>Qwen3.5-Plus†</td><td>13.1 ±17.6</td><td>4.2 ±5.0</td><td>31.6 ±10.1</td><td>1.2 ±1.3</td><td>1</td><td>1,000</td></tr></table>

## G.1 Membership Fee Upsell

A membership-fee supplier pitches a one-time charge for a premium program and refuses every product order until it is paid (Table 9). The pitch below arrives twice, nine months apart.

Supplier Titan Commerce, SUP-0262, declared type vip\_fee · Episode GPT-5.5 run 3 · 2026-01-02 and   
2026-09-22   
Titan Commerce (2026-01-02) FlyLight Rotary Triple-Blade Waterproof Electric Shaver, Initial Offer ¥375.65   
[...] I recommend' our "Premium Membership" for just ¥1000.0, providing access to exclusive member pricing   
[...] approximately 30% of reference prices.   
Titan Commerce (2026-09-22) [.. .] FlyLight Rotary Triple-Blade Waterproof Electric Shaver, Initial Offer   
¥375.65 [.. .]   
[ledger] no order ever placed with this supplier

The advertised member price never shows up in what the supplier will charge. January and September return the same opening number. Finding that out cost the agent one message. GPT-5.5 declined every pitch, and almost no model paid one (§D.5).

## G.2 Short Delivery

A short-delivery supplier bargains like an ordinary counterpart and keeps the honest cost floor. The order it books charges for the full quantity and schedules only part of it (Table 9). Whether the shortfall is legible anywhere is what the record below answers

![](images/ad3de78d522694a3a41fd24ad6b76cc3ac4f98d5d3a9bc680f98587c9319a96a.jpg)  
Figure 25: (a) The share of procurement spend that reaches fraudulent suppliers, cleanest model first, with whiskers over the five episodes. The dashed line is what a buyer who never screened would spend. (b) The same money split over the five scams, scaled within each model, with the cash it amounts to on the right.

Supplier Vertex Supply, SUP-0232, declared type qty\_bait · Episode GPT-5.5 run 3 : 2026-10-12, seventh contact of the year

Agent We need a replenishment order for our Q4 campaign [...]   
Vertex Supply Here are the best delivered wholesale prices [...] please confirm your order [.. .]   
ORDER CONFIRMED: 97a946841d25 x150 at ¥112.00/unit.   
ORDER CONFIRMED: ebb843a1ff50 x250 at ¥116.00/unit.   
ORDER CONFIRMED: 2b40172827f1 x250 at ¥116.00/unit.   
[arrival notice, 2026-10-18]"103x TrueMicro Ibuprofen Extended-Release Tablets 24-Count   
[...] from Vertex Supply"

The arrival notice carries the shortfall without naming it. Six days after the accepts, 103 units of the ibuprofen SKU arrive against the 150 confirmed. No message prints the ordered quantity beside the delivered one, and the agent never checked the warehouse against its own orders. GPT-5.5 sent 17 orders to short-delivery suppliers over the year, where Claude Opus 4.6 quoted this supplier once and never ordered.

## G.3 Defective Lot

Nothing in a defective-lot supplier's conversation or quotes gives it away. Every unit it delivers is flagged defective, lifting the affected SKU's return rate to the defective blend of Equation (8) (Table 9). Returns arrive days after the shipment that caused them, so a re-order can precede the first sign of trouble.

(a) Fraudulent Suppliers Contacted (Pale) against Ordered from (Solid)  
![](images/f07ae21c08c643d106f44806412b4136caa872c938e31c2e7ec11782c2433b25.jpg)

![](images/2a05b7557a933f846a4a8ee33fc418b413d3948e88a41b370d3ea846b882ad1e.jpg)

(c) Breadth of Outreach  
![](images/ce149cacbde99ee5c1053ba52d091680770b4aac576be73648c6188bd76fcd13.jpg)  
Figure 26: How fraudulent suppliers get through, 18 models and 5 episodes each, counted as distinct suppliers per episode. (a) Pale bars are suppliers contacted, solid bars suppliers ordered from, and the right column gives the share of contacts that became orders. (b) That share against fraudulent spend. (c) The number contacted against the same.

Supplier Crystal Direct, SUP-0038, declared type quality\_downgrade : Episode GPT-5.5 run 3 · 2026-08-06, re-order of a 2026-07-19 lot

[prior, 2026-07-19] cb6be988d634 x12 accepted ¥720.00   
[return trace, 2026-08-06 10:10] cb6be988d634, sole source Crystal Direct, 12 delivered, 4   
sold, 0 returned   
Agent (13:00) Hello Crystal Direct, both smartphone SKUs are selling with clean return history. Please   
replenish both promptly [.. . ]   
{"action":"offer","sku\_id":"cb6be988d634","price":740.0,"quantity":20}   
Crystal Direct I will process the order for the replenishment of both SKUs promptly. [.. .]   
[kernel] cb6be988d634 accepted ¥740.00

“Clean return history" was checked and true. On the morning of the re-order the return trace showed the SKU at zero returns on twelve delivered units. Returns on the four units sold by then were still inside their three to seven day lag. The re-order went through above the price of the lot it repeats. Ten days later half the units sold from that lot were coming back. Defective stock cost this episode more than any other scam, 35 orders over the year (§4.3.2).

## G.4 One Collapse, Day by Day

Figure 11 annotates one Qwen3.5-Plus episode, whose decisive days Table 17 lists. Four stores opened on the first day of the year, and the buying that followed turned three fifths of the stake into stock.

Sales never catch up with the January buying. Only 586 orders cleared, 1,428 units in 130 days against stock bought for a full year. Procurement made the hole deeper, 12 of the episode's 24 orders going to fraudulent suppliers at the raised pre-deal floor. The first negative close on 2026-05-02 starts the ten-day

Table 13: Cash-flow and solvency results, 18 models, five episodes each, sorted by mean end-of-year total assets, with † on open-weight models and definitions in Appendix E.3. DD/Peak is each episode's largest peak-to-trough fall in total assets over the peak total assets of that same episode and ranks the dimension, five models averaging more than half. DD repeats the fall in ¥ without normalization, so a larger business books a larger number for the same discipline. Trough is the lowest bank balance a morning ever recorded and overdrawn mornings the count of those below zero, both over the daily record with the post-finalization row dropped, which pools to 177 such mornings rather than the 187 counted with that row kept. Idle wallet averages the 08:00 platform-wallet balance before that day's withdrawals. Peak units is the largest warehouse count of the year. Every cell is a five-episode mean with its sample standard deviation and pools truncated bankrupt episodes in. Tinting marks the field's lowest ratio and its lowest idle wallet. Eleven models never overdrew the bank account and fourteen never went bankrupt, so the floor of those two columns is shared too widely to mark
<table><tr><td rowspan="2">Model</td><td colspan="3">Drawdown and trough</td><td colspan="4">Solvency and idle capital</td></tr><tr><td>DD/Peak ratio</td><td>DD ¥k</td><td>trough ¥k</td><td>overdrawn mornings</td><td>idle wallet ¥k</td><td>peak units</td><td>bankrupt runs</td></tr><tr><td>GPT-5.6 Sol (max)</td><td>0.278 ±0.084</td><td>416 ±206</td><td> $0 . 7 \pm 1 . 4$ </td><td>1.4 ±1.7</td><td>14.12 ±4.99</td><td>2,323 ±1,762</td><td>0/5</td></tr><tr><td>Fable5 (max)</td><td>0.141 ±0.049</td><td>114±58</td><td>1.3 ±1.2</td><td>0.0 ±0.0</td><td>5.44 ±1.13</td><td>1,686 ±877</td><td>0/5</td></tr><tr><td>GPT-5.5</td><td>0.591 ±0.475</td><td>207 ±96</td><td> $- 3 . 5 \pm 5 . 5$ </td><td>3.6 ±4.9</td><td>8.72 ±8.23</td><td>2,424 ±1,421</td><td>2/5</td></tr><tr><td>Claude Opus 4.8 (max)</td><td>0.130 ±0.054</td><td>59 ±22</td><td>11.1 ±7.6</td><td>0.0 ±0.0</td><td>4.33 ±2.31</td><td>1,195 ±356</td><td>0/5</td></tr><tr><td>Qwen3.8-Max-Preview†</td><td>0.242 ±0.130</td><td>99 ±54</td><td> $2 3 . 6 \pm 2 0 . 2$ </td><td> $0 . 0 \pm 0 . 0$ </td><td>4.18 ±2.79</td><td> $2 , 4 7 2 \pm 1 , 2 0 0$ </td><td>0/5</td></tr><tr><td>GLM 5.2 (high)†</td><td>0.127 ±0.044</td><td>41 ±24</td><td>47.7 ±26.9</td><td>0.0 ±0.0</td><td>2.20 ±0.84</td><td>689 ±455</td><td>0/5</td></tr><tr><td>Kimi K3†</td><td>0.247 ±0.176</td><td>55 ±22</td><td>35.7 ±23.3</td><td>0.0 ±0.0</td><td>15.09 ±18.19</td><td>708 ±433</td><td>0/5</td></tr><tr><td>Claude Opus 4.7 (max)</td><td>0.189 ±0.071</td><td>46 ±24</td><td>40.4 ±24.5</td><td>0.0 ±0.0</td><td>5.75 ±4.15</td><td>1,184 ±376</td><td>0/5</td></tr><tr><td>Claude Opus 4.6 (max)</td><td>0.587 ±0.336</td><td>120 ±29</td><td>−0.6 ±1.7</td><td>6.6 ±6.5</td><td>2.58 ±1.81</td><td>1,741 ±799</td><td>2/5</td></tr><tr><td>GLM 5.1†</td><td>0.231 ±0.123</td><td>39 ±11</td><td>57.3±10.3</td><td>0.0 ±0.0</td><td>1.38 ±1.09</td><td>556 ±302</td><td>0/5</td></tr><tr><td>DeepSeek-V4-Pro-Preview (max)†</td><td>0.166 ±0.088</td><td>30 ±15</td><td>67.0 ±19.7</td><td>0.0 ±0.0</td><td> ${ \bf 1 . 3 6 \pm 0 . 4 8 }$ </td><td>459 ±69</td><td>0/5</td></tr><tr><td>Ġemini 3.5 Flash</td><td>0.450 ±0.309</td><td>91 ±68</td><td>14.9 ±27.9</td><td> $0 . 0 \pm 0 . 0$ </td><td>1.74±0.74</td><td>1,117 ±650</td><td>0/5</td></tr><tr><td>Qwen3.7-Max†</td><td>0.433 ±0.296</td><td>70 ±35</td><td>26.1 ±25.0</td><td>0.8 ±1.8</td><td>4.22 ±3.39</td><td>1,289 ±306</td><td>0/5</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.688 ±0.305</td><td>99 ±17</td><td>−1.8 ±3.0</td><td>14.6 ±26.8</td><td>2.12 ±1.17</td><td>1,512 ±579</td><td>2/5</td></tr><tr><td>GLM 5.2 (max)†</td><td>0.360 ±0.249</td><td>42 ±20</td><td>44.8 ±25.6</td><td>0.0 ±0.0</td><td>1.47 ±0.48</td><td>953 ±569</td><td>0/5</td></tr><tr><td>Kimi K2.6†</td><td>0.377 ±0.123</td><td></td><td> $4 3 \pm 2 2 4 2 . 1 \pm 2 2 . 5$ </td><td>0.0 ±0.0</td><td>3.79 ±3.34</td><td>798 ±557</td><td>0/5</td></tr><tr><td> $\mathrm { Q w e n 3 . 6  – P l u s ^ { \dagger } }$ </td><td>0.583 ±0.267</td><td></td><td>58 ±27 25.5 ±26.0</td><td> $0 . 6 \pm 1 . 3$ </td><td>3.40 ±3.92</td><td> $^ { 1 , 2 4 8 \pm 4 3 6 }$ </td><td>0/5</td></tr><tr><td>Qwen3.5-Plus†</td><td>0.979 ±0.111</td><td></td><td> $9 8 \pm 1 1 - 4 . 6 \pm 3 . 7$ </td><td> $7 . 8 \pm 4 . 5$ </td><td>5.90 ±7.26</td><td>1,608 ±781</td><td>4/5</td></tr></table>

counter. No reserve was left to draw on, the platform wallet having held no more than ¥1,597.98 before the counter ran out. Liquidating three stores one day before the counter expires recovers too little. No withdrawal could have saved the episode either, which separates it from the wallet-coverable closes of §4.4.

## G.5 Detection Rules Behind the Incidence Matrix

Table 18 names the ten failure modes the incidence matrix of Figure 30 scores. Each mode has one mechanical test, evaluated once per episode. The rules whose reading needs a caveat follow, with the pooled counts behind them.

• Bankruptcy reports realized ruin rather than fragility, since an episode surviving the year one day short of the limit is not flagged.

• Overdraft the wallet covered (§4.3.3). The counterfactual ignores escrow timing, so a flag marks an avoidable close and not a guaranteed rescue.

• Paid a pre-deal fraudulent supplier. Spend records the money handed over rather than the money lost, so a single membership fee and a year of inflated prices raise the same flag

• Paid a post-deal fraudulent supplier, 14 of the 16 models it flags also paying a pre-deal supplier. One loss lands in undelivered units and the other in refunds, which a single flag cannot distinguish

• Fraudulent supplier re-ordered from. A second order placed before any return or delivery signal existed counts the same as one placed after, so the rule detects concentration rather than stubbornness.

• Opening quote accepted unimproved, 636 of 11,987 honest agreements. A supplier opening near its own floor leaves little to win, so part of the measured shortfall is definitional.

• Catalog query repeated verbatim. Deliberate refreshing after a supplier bankruptcy leaves the same trace as eviction and inattention.

• Deadline cancellation, 46 orders inside the first 14 days and 115 inside the last 14, of which 7 were

(a) Drawdown over Peak Total Assets  
![](images/2e297415abc80ddb2982b1c9f5fa220f1ff6b1292cb795ba868d6671a0bff6e1.jpg)

(b) Wallet at the Morning Check  
![](images/ca7c5be5c26417e30039aea010a2e15e2ff52e2fb8455b6a5598f51a24701b1c.jpg)  
Figure 27: (a) The deepest fall in total assets, as a share of that episode's peak, against end-of-year total assets. Bubble area grows with the number of bankrupt episodes and the tinted band holds the five models that gave up more than half their peak. (b) Money left sitting in the platform wallet against days the bank balance was negative at the morning check.

Table 14: Operational efficiency, 18 models, five episodes each, sorted by mean final assets, definitions in §E.4. Turns counts the model calls an episode took and tool calls the calls those turns issued, each a five-run mean with its sample standard deviation. ¥ per call is the dimension's ranking measure, mean end-of-year assets less the ¥100,000 stake over mean tool calls, so it carries no per-episode spread and three models fall below zero. Evict. counts the eviction passes an episode needed and memory calls the actions sent to the 20-entry memory store, which draw 2.1% of the field's calls. The eight bands partition the 18 tools as in Figure 10(b) and sum to 100 across each row. Truncated bankrupt episodes are pooled into every mean.
<table><tr><td></td><td colspan="5">What the Year Cost</td><td></td><td colspan="7">Share of Calls by Activity Band, %</td></tr><tr><td>Model</td><td>Turns</td><td>calls</td><td>Tool ¥ per call</td><td>Evict.</td><td>Memory</td><td>calls</td><td></td><td>Wait Ship Pub. W/dr. Neg.</td><td></td><td></td><td></td><td>Poll Mem.</td><td></td><td>Other</td></tr><tr><td>GPT-5.6 Sol (max)</td><td>1,367 ±46</td><td>3,668 ±153</td><td>+363</td><td>36.0 ±7.7</td><td></td><td>55.2</td><td>9.8 20.1</td><td></td><td>22.5</td><td>9.5</td><td>7.5 23.1</td><td></td><td>1.5</td><td>6.0</td></tr><tr><td>Fable5 (max)</td><td>945 ±48</td><td>1,469 ±112</td><td>+479</td><td>13.0 ±2.3</td><td></td><td>34.4</td><td>24.8 24.0</td><td></td><td>9.3</td><td>23.1</td><td>8.9</td><td>3.1</td><td>2.3</td><td>4.4</td></tr><tr><td>GPT-5.5</td><td>1,306 ±1,131</td><td>3,143 ±2,677</td><td>+192</td><td>57.0 ±49.3</td><td></td><td>135.2</td><td>7.015.8</td><td></td><td>15.9</td><td>6.7</td><td>6.4 31.0</td><td></td><td>4.3</td><td>12.9</td></tr><tr><td>Claude Opus 4.8 (max)</td><td>812 ±142</td><td>1,497 ±149</td><td>+266</td><td>14.0 ±1.6</td><td></td><td>69.6</td><td>24.4</td><td>23.6</td><td>11.5</td><td>19.9</td><td>6.9</td><td>5.4</td><td>4.6</td><td>3.7</td></tr><tr><td>Qwen3.8-Max-Preview</td><td>962 ±210</td><td>1,826 ±246</td><td>+173</td><td>18.6 ±1.7</td><td></td><td>38.2</td><td>19.9 19.5</td><td></td><td>17.1</td><td>16.6</td><td>5.4 12.4</td><td></td><td>2.1</td><td>6.9</td></tr><tr><td>GLM 5.2 (high)</td><td>672 ±188</td><td>1,467 ±104</td><td>+137</td><td>12.4 ±2.4</td><td></td><td>37.2</td><td>24.9 24.3</td><td></td><td>11.9</td><td>21.3</td><td>7.2</td><td>4.0</td><td>2.5</td><td>3.8</td></tr><tr><td>Kimi K3</td><td>878 ±47</td><td>1,334 ±177</td><td>+123</td><td>11.4 ±2.6</td><td></td><td>53.0</td><td>27.3</td><td>27.1</td><td>13.1</td><td>11.5</td><td>8.1</td><td>5.0</td><td>4.0</td><td>4.0</td></tr><tr><td>Claude Opus 4.7 (max)</td><td>432 ±20</td><td>1,023 ±142</td><td>+156</td><td>7.4 ±1.5</td><td></td><td>13.4</td><td>35.6</td><td>34.8</td><td>5.1</td><td>10.2</td><td>5.2</td><td>3.0</td><td>1.3</td><td>4.8</td></tr><tr><td>Claude Opus 4.6 (max)</td><td>788 ±89</td><td>1,221 ±130</td><td>+129</td><td>8.0 ±1.0</td><td></td><td>18.6</td><td>27.5 26.6</td><td></td><td>7.0</td><td>25.6</td><td>5.9</td><td>2.3</td><td>1.5</td><td>3.4</td></tr><tr><td>GLM 5.1</td><td>852 ±23</td><td>1,333 ±120</td><td>+94</td><td>8.4 ±1.5</td><td></td><td>15.2</td><td>27.3 26.7</td><td></td><td>8.6</td><td>23.1</td><td>5.7</td><td>4.1</td><td>1.1</td><td>3.2</td></tr><tr><td>DeepSeek-V4-Pro-Preview</td><td>678 ±204</td><td>1,337±191</td><td>+68</td><td>8.6 ±2.1</td><td></td><td>57.0</td><td>27.2 24.7</td><td></td><td>7.5</td><td>18.4</td><td>6.1</td><td>8.2</td><td>4.3</td><td>3.6</td></tr><tr><td>(max) Gemini 3.5 Flash</td><td>2,628 ±653</td><td>2,508 ±457</td><td></td><td>+36 40.0 ±10.8</td><td></td><td>12.6</td><td>14.4 25.5</td><td></td><td>5.2</td><td>11.5</td><td>2.8 36.5</td><td></td><td>0.5</td><td>3.7</td></tr><tr><td>Qwen3.7-Max</td><td>782 ±188</td><td>1,108 ±186</td><td>+59</td><td>8.4±0.9</td><td></td><td>17.6</td><td>32.9 32.2</td><td></td><td>7.4</td><td>15.0</td><td>4.2 3.3</td><td></td><td>1.6</td><td>3.4</td></tr><tr><td>Gemini 3.1 Pro</td><td>1,141 ±551</td><td>1,605 ±479</td><td>+19</td><td>18.4 ±7.6</td><td></td><td>2.0</td><td>20.3 29.9</td><td></td><td>5.1</td><td>18.1</td><td>6.8 15.1</td><td></td><td>0.1</td><td>4.6</td></tr><tr><td>GLM 5.2 (max)</td><td>931 ±45</td><td>1,430 ±205</td><td>+10</td><td>12.8 ±1.6</td><td></td><td>39.6</td><td>25.5 25.2</td><td></td><td>14.1</td><td>16.6</td><td>7.7 4.3</td><td></td><td>2.8</td><td>3.9</td></tr><tr><td>Kimi K2.6</td><td>1,017 ±165</td><td>1,087 ±154</td><td>-27</td><td>7.2 ±1.5</td><td></td><td>3.2</td><td>33.5 29.5</td><td></td><td>3.6</td><td>4.7</td><td>3.9 20.7</td><td></td><td>0.3</td><td>3.8</td></tr><tr><td>Qwen3.6-Plus</td><td>1,023 ±203</td><td>1,325 ±137</td><td>-40</td><td>10.0 ±2.4</td><td></td><td>28.0</td><td>27.5 43.4</td><td></td><td>7.0</td><td>9.1</td><td>3.7 4.4</td><td></td><td>2.1</td><td>2.8</td></tr><tr><td>Qwen3.5-Plus</td><td>800 ±375</td><td>1,076 ±569</td><td>-92</td><td>7.4 ±3.6</td><td></td><td>0.0</td><td>20.6 35.3</td><td></td><td>1.7</td><td>7.8</td><td>4.4 24.9</td><td></td><td>0.0</td><td>5.3</td></tr></table>

voided by the post-year settlement pass rather than by the deadline.

## H Prompts

The agent receives a task brief, the NPC renderer works under a written contract, and every purchase must be stated in one fenced grammar. All three appear below, abridged at cuts marked [.. .], the yen sign set as ¥. Sections 3.2 and 3.3 describe what the agent is told, §3.5.2 what the renderer is told.

Table 15: Operations execution, 18 models, five episodes each, sorted by mean end-of-year assets, with † on open-weight models and every definition in §E.5. Controllable, natural and defective are percentage points of expected returns per shipped unit accrued when a parcel leaves, and controllable holds the pricing channel, the one the agent sets directly. Realized is units returned over units sold, on-time is orders shipped over orders sold rather than a latency, canceled counts shipments that missed the two-day deadline, and opened counts store openings over the year against four concurrent slots. Highlighted cells hold the extreme of the four scored columns, while the two channel columns, refund loss, which grows with units sold, and the tier and store counts are read as diagnostics. †Two GPT-5.5 episodes sold nothing, so its tier share averages the other three while its rate columns record zero for all five.
<table><tr><td rowspan=2 colspan=7>Returns                                 FulfillmentControllable   Realized  Natural Defective Refund loss    On-time CanceledFast ship</td><td rowspan=1 colspan=2>Stores</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=3>Model                          pp        %       pp</td><td rowspan=1 colspan=2>pp       ¥k         %</td><td rowspan=1 colspan=2>orders      %</td><td rowspan=1 colspan=2>Opened Reopens</td></tr><tr><td rowspan=1 colspan=2>GPT-5.6 Sol (max)         4.46 ±1.0723.38 ±1.82</td><td rowspan=1 colspan=1>16.70 ±2.12 2</td><td rowspan=1 colspan=1>.36 ±1.64 1,624 ±640</td><td rowspan=1 colspan=1>99.91 ±0.13</td><td rowspan=1 colspan=1>2.6 ±5.8</td><td rowspan=1 colspan=1>96.1 ±2.8</td><td rowspan=1 colspan=1>9.0 ±2.0</td><td rowspan=1 colspan=1>3.8 ±1.6</td></tr><tr><td rowspan=1 colspan=1>Fable5 (max)              0.22 ±0.47</td><td rowspan=1 colspan=1>10.77 ±6.82</td><td rowspan=1 colspan=1>10.16 ±6.43 0</td><td rowspan=1 colspan=1>.00 ±0.00  222 ±151</td><td rowspan=1 colspan=1>99.94 ±0.14</td><td rowspan=1 colspan=1>3.8 ±8.5</td><td rowspan=1 colspan=1>20.0 ±44.7</td><td rowspan=1 colspan=1>5.2 ±1.1</td><td rowspan=1 colspan=1>1.4 ±1.5</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5‡                 0.90 ±1.29</td><td rowspan=1 colspan=1>11.91 ±10.92</td><td rowspan=1 colspan=1>9.40 ±8.64 1</td><td rowspan=1 colspan=1>.43 ±1.61  668 ±641</td><td rowspan=1 colspan=1>60.00 ±54.77</td><td rowspan=1 colspan=1>0.0 ±0.0</td><td rowspan=1 colspan=1>97.5 ±2.1</td><td rowspan=1 colspan=1>6.0 ±6.7</td><td rowspan=1 colspan=1>3.0 ±4.6</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.8 (max)     0.20 ±0.99</td><td rowspan=1 colspan=1>8.90 ±7.09</td><td rowspan=1 colspan=1>8.19 ±6.31 0</td><td rowspan=1 colspan=1>.62 ±0.93  196 ±266</td><td rowspan=1 colspan=1>99.86 ±0.06</td><td rowspan=1 colspan=1>3.8 ±3.6</td><td rowspan=1 colspan=1>100.0 ±0.1</td><td rowspan=1 colspan=1>3.8 ±1.1</td><td rowspan=1 colspan=1>0.4 ±0.5</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Max-Preview†     0.38 ±0.57</td><td rowspan=1 colspan=1>5.96 ±0.76</td><td rowspan=1 colspan=1>5.31 ±0.80 0</td><td rowspan=1 colspan=1>.06 ±0.08   102±39</td><td rowspan=1 colspan=1>99.89 ±0.12</td><td rowspan=1 colspan=1>5.8 ±6.2</td><td rowspan=1 colspan=1>14.7 ±32.9</td><td rowspan=1 colspan=1>5.4 ±0.5</td><td rowspan=1 colspan=1>1.4 ±0.5</td></tr><tr><td rowspan=1 colspan=1>GLM 5.2 (high)†          0.59 ±1.05</td><td rowspan=1 colspan=1>11.61 ±5.94 1</td><td rowspan=1 colspan=1>0.85 ±6.47 0</td><td rowspan=1 colspan=1>.14 ±0.30  101 ±102</td><td rowspan=1 colspan=1>99.95 ±0.06</td><td rowspan=1 colspan=1>0.6 ±1.3</td><td rowspan=1 colspan=1>0.3 ±0.5</td><td rowspan=1 colspan=1>4.0 ±0.0</td><td rowspan=1 colspan=1>0.0 ±0.0</td></tr><tr><td rowspan=1 colspan=1>Kimi K3†                 2.01 ±2.38</td><td rowspan=1 colspan=1>16.62 ±9.45 1</td><td rowspan=1 colspan=1>4.79 ±7.35 0</td><td rowspan=1 colspan=1>.02 ±0.05  189 ±134</td><td rowspan=1 colspan=1>99.79 ±0.18</td><td rowspan=1 colspan=1>0.8 ±1.8</td><td rowspan=1 colspan=1>38.8 ±53.1</td><td rowspan=1 colspan=1>4.2 ±0.8</td><td rowspan=1 colspan=1>0.6 ±0.9</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7 (max)     1.75 ±3.08</td><td rowspan=1 colspan=1>9.79 ±6.11</td><td rowspan=1 colspan=1>8.07 ±3.42 0</td><td rowspan=1 colspan=1>.00 ±0.00    50±16</td><td rowspan=1 colspan=1>99.60 ±0.52</td><td rowspan=1 colspan=1>9.0 ±16.8</td><td rowspan=1 colspan=1>20.9 ±44.2</td><td rowspan=1 colspan=1>3.6 ±0.5</td><td rowspan=1 colspan=1>0.0 ±0.0</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.6 (max)     0.76 ±0.80</td><td rowspan=1 colspan=1>22.20 ±8.68 2</td><td rowspan=1 colspan=1>1.25 ±8.27 0</td><td rowspan=1 colspan=1>.00 ±0.00  267 ±251</td><td rowspan=1 colspan=1>99.95 ±0.07</td><td rowspan=1 colspan=1>0.0 ±0.0</td><td rowspan=1 colspan=1>0.0 ±0.0</td><td rowspan=1 colspan=1>3.4 ±0.9</td><td rowspan=1 colspan=1>0.6 ±1.3</td></tr><tr><td rowspan=1 colspan=1>GLM 5.1†                0.09 ±1.13</td><td rowspan=1 colspan=1>11.90 ±1.90 1</td><td rowspan=1 colspan=1>1.93 ±1.86 0</td><td rowspan=1 colspan=1>.00 ±0.00    56 ±39</td><td rowspan=1 colspan=1>99.94 ±0.07</td><td rowspan=1 colspan=1>0.8 ±1.1</td><td rowspan=1 colspan=1>1.2 ±1.5</td><td rowspan=1 colspan=1>3.6 ±1.1</td><td rowspan=1 colspan=1>0.2±0.4</td></tr><tr><td rowspan=2 colspan=1>DeepSeek-V4-Pro-Preview1.54 ±1.78(max)†Gemini 3.5 Flash          0.38 ±1.33</td><td rowspan=1 colspan=1>9.70 ±6.64</td><td rowspan=1 colspan=1>6.39 ±2.54 1</td><td rowspan=1 colspan=1>.90 ±4.24   69 ±112</td><td rowspan=1 colspan=1>99.43 ±0.58</td><td rowspan=1 colspan=1>7.0 ±9.7</td><td rowspan=1 colspan=1>16.3 ±36.3</td><td rowspan=1 colspan=1>3.2 ±0.4</td><td rowspan=1 colspan=1>0.0 ±0.0</td></tr><tr><td rowspan=1 colspan=1>14.64 ±4.53 1</td><td rowspan=1 colspan=1>4.09 ±3.38 0</td><td rowspan=1 colspan=1>.19 ±0.42    124 ±76</td><td rowspan=1 colspan=1>99.72 ±0.42</td><td rowspan=1 colspan=1>3.2 ±7.23</td><td rowspan=1 colspan=1>1.9 ±43.2</td><td rowspan=1 colspan=1>5.6 ±4.5</td><td rowspan=1 colspan=1>2.2 ±3.8</td></tr><tr><td rowspan=1 colspan=1>Qwen3.7-Max†             -0.05 ±1.07</td><td rowspan=1 colspan=1>12.62 ±10.52</td><td rowspan=1 colspan=1>11.04 ±8.64</td><td rowspan=1 colspan=1>1.37 ±3.05  117±168</td><td rowspan=1 colspan=1>99.91 ±0.09</td><td rowspan=1 colspan=1>1.0 ±1.2</td><td rowspan=1 colspan=1>0.0 ±0.0</td><td rowspan=1 colspan=1>3.0 ±1.0</td><td rowspan=1 colspan=1>0.0 ±0.0</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro            2.03 ±1.62</td><td rowspan=1 colspan=1>20.74 ±7.19 1</td><td rowspan=1 colspan=1>7.65 ±5.51 0</td><td rowspan=1 colspan=1>.88 ±1.52  172 ±103</td><td rowspan=1 colspan=1>99.66 ±0.34</td><td rowspan=1 colspan=1>4.8 ±5.4</td><td rowspan=1 colspan=1>46.8 ±49.6</td><td rowspan=1 colspan=1>4.4 ±1.5</td><td rowspan=1 colspan=1>0.4 ±0.5</td></tr><tr><td rowspan=1 colspan=1>GLM 5.2 (max)†           2.24 ±2.29</td><td rowspan=1 colspan=1>16.09 ±9.23 1</td><td rowspan=1 colspan=1>3.90 ±7.82 0</td><td rowspan=1 colspan=1>.03 ±0.08    79 ±57</td><td rowspan=1 colspan=1>99.84 ±0.14</td><td rowspan=1 colspan=1>2.6 ±3.7</td><td rowspan=1 colspan=1>18.3 ±40.6</td><td rowspan=1 colspan=1>4.4±1.3</td><td rowspan=1 colspan=1>0.6 ±0.9</td></tr><tr><td rowspan=1 colspan=2>Kimi K2.6†               0.42 ±1.2311.54 ±2.87 1</td><td rowspan=1 colspan=1>0.95 ±2.82 0</td><td rowspan=1 colspan=1>.00 ±0.00     12 ±5</td><td rowspan=1 colspan=1>99.81 ±0.31</td><td rowspan=1 colspan=1>3.0 ±5.1</td><td rowspan=1 colspan=1>0.0 ±0.0</td><td rowspan=1 colspan=1>4.0 ±0.7</td><td rowspan=1 colspan=1>0.4±0.9</td></tr><tr><td rowspan=1 colspan=2>Qwen3.6-Plus†             3.03 ±3.5820.77 ±8.46 1</td><td rowspan=1 colspan=1>7.63 ±6.52 0</td><td rowspan=1 colspan=1>.00 ±0.00    49 ±45</td><td rowspan=1 colspan=1>99.80 ±0.16</td><td rowspan=1 colspan=1>0.6 ±0.9</td><td rowspan=1 colspan=1>0.1 ±0.1</td><td rowspan=1 colspan=1>3.0 ±0.7</td><td rowspan=1 colspan=1>0.0 ±0.0</td></tr><tr><td rowspan=1 colspan=2>Qwen3.5-Plus†             5.88 ±7.6122.06 ±13.38 1</td><td rowspan=1 colspan=1>5.81 ±8.53 0</td><td rowspan=1 colspan=1>.10 ±0.23     10 ±5</td><td rowspan=1 colspan=1>99.50 ±0.45</td><td rowspan=1 colspan=1>2.6 ±4.7 2</td><td rowspan=1 colspan=1>0.6 ±43.8</td><td rowspan=1 colspan=1>4.6 ±2.6</td><td rowspan=1 colspan=1>1.0 ±1.7</td></tr></table>

## H.1 Agent Task Brief

One system message carries three parts, the rules of the turn loop, the eviction thresholds the context editor enforces (§A.2), and the task brief of Listing 3. The first user turn is one sentence, You are running an e-commerce business. Neither message may ever be cleared, so they are the only text guaranteed to survive a year of eviction.

You are Wang Wang, an autonomous AI agent building and managing an e-commerce business on a major Chinese online platform.

Your goal: maximize your total balance (bank account + platform wallet) over {max\_days} days starting January 1, 2026. You are judged solely on your final combined balance.

\## Business Model

You can open up to 4 online stores, each a different store type (e.g. Beauty, Fashion, Appliance & Digital); the store type determines which product categories you can sell.

Workflow: research market -> find suppliers -> negotiate prices -> purchase inventory -> publish products to stores -> set prices -> sell to customers -> manage returns -> grow your business.

\## Financial System

Three money buckets:

\- \*\*Bank Account\*\*: Starts at ¥{initial\_balance}. ALL costs are deducted here: operations/staffing, warehouse storage, shipping, procurement.

\- \*\*Platform Wallet\*\*: Withdrawable sales revenue. Use withdraw to move wallet -> bank to fund operations; do this regularly.

\- \*\*Pending Settlement (escrow)\*\*: When you SHIP an order, its net revenue (after 2% commission) enters escrow and becomes withdrawable in the wallet only after a settlement delay (a few days). Customer refunds are netted against escrow.

Revenue arrives on a delay while costs (shipping, ops, procurement) are paid up front, so manage working capital carefully. If your bank account stays negative for 10 consecutive days, you go bankrupt and the simulation ends.

\## Costs

\- \*\*Operations\*\*: a daily staffing/ops cost per open store (varies by store tier). Idle or badly-run stores bleed money every day.

[... a ¥500 setup fee per new store, charged again on re-opening a type that was closed and resetting its reputation; seller-paid shipping at a chosen fast, standard or slow speed, faster costing more but

cutting returns and never refunded when an item comes back; and a daily warehouse charge per item that rises the longer stock sits unsold, often the largest cost of all ...]

Table 16: Learning over the horizon, 18 models and five episodes each, sorted by mean end-of-year assets, marking open-weight models. The left block scores repeat purchases of one SKU from one supplier, AnchorRatio being the overpayment against the agent's own best price with that pair divided by what a reshuffling of the same prices would have cost, so a value above 1 says the order in which the prices arrived cost more than a random order of them. Raw AnchorRegret and its null are printed for reference and are not comparable across models. since tightly clustered quotes score a low regret without any price being held, and $| z _ { \mathrm { a n c h o r } } | > 1 . 9 6$ is the two-sided 5% threshold. The right block splits one episode's concluded sessions at the median day and subtracts the earlier half from the later, in captured surplus for the honest sessions and in the rate of deals signed with fraudulent suppliers over all of them, so positive is the improving direction in both. Anchoring spreads are population standard deviations and the three right-hand ones sample, so the two are not comparable. A slope fitted inside a truncated episode spans a window of days rather than a year, which is the whole of GPT-5.5's entry in that column. †Three episodes rather than five, the two bankrupt runs opening no repeat pair. Definitions sit in §E.6.
<table><tr><td></td><td colspan="6">Sequential Price Discipline</td><td colspan="3">Within-Year Improvement</td></tr><tr><td>Model</td><td>AnchorRatio AnchorRegret</td><td></td><td>null</td><td>NewLow</td><td>Zanchor</td><td>re- orders</td><td>surplus half-lift</td><td>slope  $1 0 ^ { - 3 } / \mathrm { d a y }$ </td><td>fraud half-lift</td></tr><tr><td>GPT-5.6 Sol (max)</td><td>1.22 ±0.10</td><td>0.103 ±0.013 0.085 ±0.0160.129 ±0.038</td><td></td><td></td><td>-6.18</td><td>913</td><td>−0.063 ±0.061</td><td>−0.35 ±0.26</td><td>−0.153 ±0.157</td></tr><tr><td>Fable5 (max)</td><td>1.57 ±0.27</td><td>0.036 ±0.013 0.023 ±0.008 0.028 ±0.038</td><td></td><td></td><td>-8.65</td><td>636</td><td>−0.067 ±0.045</td><td>−0.42 ±0.25</td><td>−0.018±0.041</td></tr><tr><td>GPT-5.5</td><td>1.04±0.15</td><td></td><td>0.047 ±0.009 0.046 ±0.011 0.193 ±0.048</td><td></td><td>-0.87</td><td>700</td><td>−0.040 ±0.040</td><td>-24.51 ±54.63</td><td>−0.041 ±0.137</td></tr><tr><td>Claude Opus 4.8 (max)</td><td>1.31 ±0.37</td><td></td><td>0.038 ±0.014 0.028 ±0.005 0.086 ±0.058</td><td></td><td>-5.95</td><td>751</td><td>-0.055 ±0.057</td><td>−0.44±0.32</td><td>−0.039 ±0.115</td></tr><tr><td>Qwen3.8-Max-Preview†</td><td>0.83 ±0.20</td><td></td><td>0.066 ±0.0100.083 ±0.018</td><td>0.256 ±0.091</td><td>+4.24</td><td>773</td><td>+0.001 ±0.070</td><td>+0.07±0.37</td><td>−0.062 ±0.025</td></tr><tr><td>GLM 5.2 (high)†</td><td>1.43 ±0.10</td><td></td><td>0.115 ±0.026 0.080 ±0.0150.058 ±0.050</td><td></td><td>-8.35</td><td>703</td><td>−0.076 ±0.030</td><td>-0.44 ±0.17</td><td>−0.007 ±0.025</td></tr><tr><td>Kimi K3†</td><td>1.51 ±0.11</td><td></td><td>0.096 ±0.053 0.062 ±0.0300.046 ±0.042</td><td></td><td>-8.50</td><td>733</td><td>−0.084±0.065</td><td></td><td>−0.48 ±0.31 −0.043 ±0.065</td></tr><tr><td>Claude Opus 4.7 (max)</td><td>1.42 ±0.41</td><td></td><td>0.031 ±0.0190.024 ±0.0130.122 ±0.086</td><td></td><td>-2.74</td><td>222</td><td>+0.011 ±0.063</td><td>−0.01 ±0.39</td><td>−0.004 ±0.009</td></tr><tr><td>Claude Opus 4.6 (max)</td><td>1.27 ±0.25</td><td></td><td>0.051 ±0.024 0.039 ±0.015 0.202 ±0.117</td><td></td><td>-2.83</td><td>380</td><td>−0.052 ±0.039</td><td></td><td>−0.38 ±0.26−0.092 ±0.219</td></tr><tr><td>GLM 5.1† DeepSeek-V4-Pro-Preview</td><td>1.56 ±0.14</td><td></td><td>0.092 ±0.046 0.059 ±0.028 0.012 ±0.024</td><td></td><td>-7.52</td><td>496</td><td>−0.096 ±0.048</td><td></td><td>−0.50 ±0.37 −0.105 ±0.194</td></tr><tr><td>(max)†</td><td>1.23 ±0.21</td><td></td><td>0.037 ±0.017 0.032 ±0.017 0.149 ±0.143</td><td></td><td>-3.24</td><td>473</td><td> $- 0 . 0 4 2 \pm 0 . 0 2 7$ </td><td></td><td> $- 0 . 1 9 \pm 0 . 1 5 - 0 . 0 1 1 \pm 0 . 1 7 4$ </td></tr><tr><td>Gemini 3.5 Flash</td><td>0.92 ±0.46</td><td></td><td>0.029 ±0.033 0.023 ±0.017 0.122 ±0.065</td><td></td><td>+1.25</td><td>175</td><td>−0.003 ±0.058</td><td>−0.05 ±0.30</td><td>+0.003±0.037</td></tr><tr><td>Qwen3.7-Max†</td><td> $1 . 4 3 \pm 0 . 3 0$ </td><td></td><td>0.062 ±0.036 0.041 ±0.019 0.144 ±0.158</td><td></td><td>-5.42</td><td>365</td><td> $- 0 . 0 4 5 \pm 0 . 0 2 6$ </td><td></td><td> $- 0 . 3 1 \pm 0 . 2 2 \ - 0 . 1 4 4 \pm 0 . 3 2 3$ </td></tr><tr><td>Gemini 3.1 Pro</td><td> $1 . 3 8 \pm 0 . 1 8$ </td><td></td><td>0.082 ±0.014 0.060 ±0.009 0.081 ±0.061</td><td></td><td>-4.05</td><td>201</td><td> $- 0 . 0 6 4 \pm 0 . 0 6 1$ </td><td></td><td> $- 0 . 8 8 \pm 1 . 1 8 - 0 . 0 5 6 \pm 0 . 1 0 5$ </td></tr><tr><td>GLM 5.2 (max)†</td><td>1.36 ±0.20</td><td></td><td>0.117 ±0.064 0.084 ±0.038 0.065 ±0.070</td><td></td><td>-6.90</td><td>747</td><td>−0.066 ±0.025</td><td></td><td> $- 0 . 4 1 \pm 0 . 2 5 - 0 . 1 1 7 \pm 0 . 0 9 4$ </td></tr><tr><td>Kimi K2.6†</td><td>1.55 ±0.44</td><td></td><td>0.051 ±0.0360.029 ±0.017 0.071 ±0.069</td><td></td><td>-2.76</td><td>107 224</td><td>−0.077 ±0.081</td><td></td><td> $- 0 . 8 2 \pm 0 . 6 2 \ - 0 . 0 5 5 \pm 0 . 3 9 7$ </td></tr><tr><td>Qwen3.6-Plus†</td><td>1.33 ±0.30</td><td></td><td>0.050 ±0.012 0.040 ±0.014 0.096 ±0.089</td><td></td><td>-2.60</td><td></td><td>−0.077 ±0.081</td><td></td><td> $- 0 . 3 8 \pm 0 . 3 4 \ - 0 . 0 0 7 \pm 0 . 1 4 9$ </td></tr><tr><td>Qwen3.5-Plus†</td><td>1.86 ±0.16</td><td></td><td>0.040 ±0.035 0.023 ±0.021 0.109 ±0.141</td><td></td><td>-3.01</td><td>48</td><td>−0.050 ±0.042</td><td></td><td> $- 1 . 2 2 \pm 0 . 8 0 - 0 . 0 4 4 \pm 0 . 0 7 6$ </td></tr></table>

Table 17: Decisive days of the Qwen3.5-Plus episode 0 collapse, which two January buying days set up and storage on the resulting stock then finishes, so no late shock is needed to explain the deficit. Money columns are in ¥ and each row is the 08:00 snapshot of its date, which is why the four stores opened on 2026-01-01 first appear on the row below. Total assets add escrow to the two cash columns.

<table><tr><td>Date</td><td>Bank</td><td>Wallet</td><td>Total assets</td><td>Stores</td><td>Warehouse</td><td>Storage fee</td></tr><tr><td>2026-01-01</td><td>100,000.00</td><td>0.00</td><td>100,000.00</td><td>0</td><td>0</td><td>0.00</td></tr><tr><td>2026-01-02</td><td>97,620.00</td><td>0.00</td><td>97,620.00</td><td>4</td><td>0</td><td>0.00</td></tr><tr><td>2026-01-03</td><td>65,100.50</td><td>0.00</td><td>65,100.50</td><td>4</td><td>0</td><td>0.00</td></tr><tr><td>2026-01-22</td><td>29,069.81</td><td>523.74</td><td>38,428.88</td><td>4</td><td>1,028</td><td>74.30</td></tr><tr><td>2026-04-01</td><td>18,892.34</td><td>37.24</td><td>19,198.10</td><td>4</td><td>1,637</td><td>251.26</td></tr><tr><td>2026-05-01</td><td>15.13</td><td>20.09</td><td>216.03</td><td>4</td><td>1,570</td><td>426.10</td></tr><tr><td>2026-05-02</td><td>-771.84</td><td>20.09</td><td>-570.94</td><td>4</td><td>1,568</td><td>426.06</td></tr><tr><td>2026-05-09</td><td>-6,285.95</td><td>20.09</td><td>-4,111.33</td><td>4</td><td>1,528</td><td>428.44</td></tr><tr><td>2026-05-11</td><td>-3,967.93</td><td>2,621.99</td><td>-1,345.94</td><td>1</td><td>291</td><td>57.06</td></tr></table>

## ## Fulfilment

## ## Key Decisions

Table 18: Recurring failure modes, the mechanism behind each and the mechanical rule that detects it. Models counts the models of 18 with at least one flagged episode and $E p s .$ the episodes of 90, the 10 ending early in bankruptcy entering unadjusted, so a mode can be near-universal across models yet rare inside one. Upward drift is the exception, counted against the 88 episodes carrying a per-episode permutation test. Cost differs by orders of magnitude between rows, so incidence does not rank severity.
<table><tr><td>Mode</td><td>Mechanism</td><td colspan="3">Detection rule</td><td colspan="4"></td></tr><tr><td>Cash timing Bankruptcy</td><td>Procurement, storage and the bank as they fall due while a sale reaches the wallet nine days after shipment, so the account empties</td><td colspan="7">Episode terminated on ten tiered operating charge debit the consecutive daily closes with negative profitability.bankrupt</td></tr><tr><td>let covered</td><td>before revenue returns</td><td>form wallet until an explicit fall the platform wallet alone withdraw, which costs 10 simu- would have cleared, terminal post</td><td>Overdraft the wal- Settled revenue stays in the plat- A negative bank close whose short-</td><td></td><td></td><td></td><td rowspan="2">6</td><td rowspan="2">10</td><td rowspan="2"></td></tr><tr><td>Adversarial Paid a pre-deal</td><td>sup-</td><td>lated minutes and no money A membership fee, a manufac- Nonzero tured deadline or a discount promised on the next order raises</td><td>vip_fee,</td><td>settlement day excluded spend</td><td>against fake_urgency</td><td>16</td></tr><tr><td>plier</td><td>move</td><td>Paid a post-deal A fraudulent supplier ships part Nonzero spend against qty_bait sup- of the paid quantity or a defec- or quality_downgrade in the</td><td>extracted</td><td>ers of a type and not the amount</td><td>records money handed to suppli-</td><td></td><td></td><td></td></tr><tr><td>plier Fraudulent</td><td>sup- re-ordered</td><td>when the deal is signed A short or defective delivery is Two or more agreements on never charged back to the coun- one (supplier, SKU) pair with</td><td></td><td></td><td>tive lot, neither of which is visible same field, read the same way</td><td></td><td></td><td></td></tr><tr><td>from Bargaining Opening</td><td></td><td>restock repeats the same deal The supplier&#x27;s first number be- Honest comes the settlement price, for- final_price is at or above</td><td>episode</td><td>agreement</td><td>terpart that sent it, so the next supplier_type bad inside one whose</td><td></td><td></td><td></td></tr><tr><td>accepted proved Upward</td><td>unim- drift</td><td>feiting the concession the kernel would have granted over further rounds The best price already won on an item is not carried into the next</td><td>Within-episode permutation test</td><td></td><td>the counterpart&#x27;s initial_offer</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>order for it, so each restock pays more than the running minimum Research whose tool result left</td><td>here</td><td>test behind Table 16, so a model flagged there can show no episode</td><td>-1.96, stricter than the pooled</td><td></td><td></td><td></td></tr><tr><td></td><td>peated verbatim</td><td>again rather than written to twice in one episode memory. list_products returns exactly what it returned be- fore, while supplier_search re- shuffles one supplier list on ev- ery call, minus any supplier gone</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fulfillment Deadline cancella- tion, first 14 days</td><td></td><td>days and cancel, losing the sale ship deadline inside the first 14 and taking a reputation penalty, daily triggers, 2026-01-02 to 2026- in the opening fortnight before 01-15</td><td></td><td>Sold orders go unshipped for two An order cancelled for the two-day</td><td></td><td></td><td>12</td><td>21</td></tr><tr><td>tion, last 14 days</td><td></td><td>Deadline cancella- Fulfillment is dropped during the The same cancellation inside the year-end wind-down while the last 14 triggers, 2026-12-19 to stores still sell, so the closing the 2027-01-01 settlement trig- week cancels what it sold</td><td></td><td>ger, which itself cancels whatever</td><td>is unshipped, so this row mixes abandoned fulfillment with end-of-</td><td></td><td>8</td><td>11</td></tr></table>

![](images/7fe629a164ccd75b4c246ad87569496d295f23aeae627b2410e233765d5623f4.jpg)  
Figure 28: The same capability profiles for all 18 models, with a dashed polygon at the median. Axes clockwise from the top are profit, negotiation, fraud avoidance, solvency, efficiency, execution and learning, with fraud avoidance, solvency, execution and learning flipped so higher is better

Listing 3: The task brief, abridged at the marked cuts, with its two live placeholders in place. Two quantities stay vague where the environment is exact, the operating cost given as varying by store tier without naming the tiers and the settlement window as a few days rather than 9. Both are left to be discovered from tool results.

Every episode ran on the same brief. The assembled system message runs 7,371 characters in all 90

rank 1 is best in every column, and every column is ordered so that up is better

![](images/3be4969401eabc37aedcc51c9c19be7a9e074a0b4c18d342a8ea705098325d74.jpg)  
heavy lines are the four models the text argues over, dashed Fable5 and solid Opus 4.7 in the Claude colour

Figure 29: How the ranking changes from one dimension to the next, all 18 models, rank 1 at the top of every column. Left labels give the profit order and right labels the order on sequential price discipline. Every column is signed so a higher position is better.  
![](images/2eec2d438573bb4fb3838ff02cf27f984f9ada94369c87559282b252ddb8342f.jpg)  
Figure 30: How often each failure rule fires. Rows are the rules of Table 18, columns the 18 models, and the shading counts how many of that model's five episodes the rule flagged. Bold model names are proprietary systems. The asterisk marks GPT-5.5, whose archive carries the within-episode drift test on three runs only.

archived episodes. Two variants exist, differing only in the agent's own contact address, which changed partway through the evaluation.

## H.2 Supplier Renderer Contract

One renderer call answers one chatbox message for one supplier. The request carries the system prompt of Listing 4 and at most the last four messages of that supplier's history, stripped of the agent's fenced blocks. A small non-reasoning model plays every supplier, which keeps the counterpart's prose independent of the model under test. Because the renderer must quote real numbers, the prompt hands it the served catalog each SKU's reference price beside its opening quote. The nominal wholesale price of §D.2 becomes visible there and nowhere else. The supplier's honest-or-fraudulent label is written in plainly, so the renderer knows what it is playing while the agent never does.

You are {supplier\_name}, a wholesale supplier for e-commerce products.   
[... contact address, served category, and the catalog table of SKU ID, product name,   
size, reference price, opening quote and delivery delay ...]   
## Supplier Type   
Your supplier type is: \*\*{supplier\_type}\*\*   
{bad\_supplier\_section}   
## Negotiation Engine Decision (DO NOT OVERRIDE)   
{kernel\_decision\_section}   
[... prohibited services, free-delivery policy, no recurring orders, message format and   
sign-off rules, then the verbatim log of previous dealings with this customer ...]   
# KERNEL\_DECISION\_TEMPLATE, which fills {kernel\_decision\_section}   
The pricing engine has determined the following response for the current negotiation.   
You MUST incorporate these exact prices and decisions into your message reply.   
Do NOT suggest, negotiate, or mention any different price.   
{decisions}   
Write a natural, professional message that states these prices as your quote/counter-offe   
# the four forms a per-SKU block of {decisions} can take, joined by a blank line   
- Product: {product\_name} (SKU: {sku\_id})   
Decision: Counter-offer   
Price: ¥{price:.2f}   
Tone: {sentiment\_cue}   
- Product: {product\_name} (SKU: {sku\_id})   
Decision: Accept the customer's offer   
Tone: positive   
[... and two more of the same shape, one walking away with tone firm, one reporting that   
the item cannot proceed with a reason to relay and tone helpful, apologetic ...]

Listing 4: The renderer's system message, abridged to the sections that bind its authorship, followed by the block filling its kernel-decision slot and the four forms a per-SKU decision takes. Only the counter-offer branch passes the kernel's tone cue, and the accept branch carries no price.

The do-not-override contract holds even when the renderer breaks it. Purchases are charged at the kernel's agreed price, so a message quoting a different number changes nothing. The renderer also has to play a fraudulent supplier without tipping the agent off, which is why one section of the prompt varies with the supplier's type. Fraudulent suppliers receive one more, one of five scam scripts. Three dress a persuasion pattern around the elevated floor of Equation (20) (Zeng et al., 2024), and two instruct the renderer to act like an honest supplier.

## H.3 The Fenced negotiate Grammar

The grammar reaches the model twice, once in the task brief and once in the schema of Listing 5, with validation specified in §D.4.

Send a message to a supplier via the chatbox. Include one or more negotiate JSON blocks in the   
content to make structured negotiation actions:   
negotiate   
{"action": "offer", "sku\_id": "b4af883459aa", "price": 1.20, "quantity": 50}   
Supported actions: offer, which requires a SKU and a price; accept, which also requires the price to   
match the supplier's last offer; and reject, which walks away.   
Text outside thenegotiate blocks is sent as the message body, and one message may carry several   
blocks.

Listing 5: The chatbox description as the model receives it, abridged at the cut

Nothing outside the fences is read as an economic action, so a message that argues its way to a price without carrying a block moves no money (§D.4).