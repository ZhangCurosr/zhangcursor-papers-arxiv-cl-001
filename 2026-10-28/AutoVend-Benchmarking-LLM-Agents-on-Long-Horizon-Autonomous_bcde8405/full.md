# AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations

Qiming Shi<sup>1,2</sup>\*, Yikun Li<sup>1,3</sup>\*, Beidi Luan<sup>1</sup>, Rui Sun<sup>1</sup>, Yan Ma<sup>1</sup>, Jing Li<sup>1</sup>, Daxin Jiang<sup>1</sup>, Zuo Bai<sup>1,4†</sup>

<sup>1</sup>StepFun, <sup>2</sup>Zhejiang University, <sup>3</sup>Xi’an Jiaotong-Liverpool University, <sup>4</sup>FinStep {baizuo}@stepfun.com

## Abstract

Large language model (LLM) agents are extending agent-mediated commerce on the Web from shopping for consumers to operating businesses for sellers. In online-to-offline (O2O) commerce, decisions are issued through Web-based platforms but realized through costly physical execution. Evaluating such agents requires going beyond individual transactions to ask whether decisions made online turn into sustained economic returns. We introduce AutoVend, a benchmark in which one LLM agent manages 20 vending machines across five site types through a Web-based operating interface for 180 simulated days. Its data-grounded simulator combines a shared warehouse, procurement lead times, paid replenishment, inventory expiration, and recurring operating costs. We evaluate five LLM agents against a rule-based inventory-control baseline and diagnose their site coordination, execution, and performance over time. Four of five evaluated LLM agents fall short of this baseline, and returns per dollar of inference cost rank models differently from returns alone. Operational analyses show that gains in individual metrics do not consistently yield stronger economic growth, as inventory and execution costs erode these gains. Trajectory analysis further reveals difficulties in grounding feasible actions, managing inventory commitments, and converting feedback into effective policy revisions.

## 1 Introduction

Agent-mediated commerce is emerging as a Web application domain for large language model (LLM) agents, with buyer-side research spanning product discovery (Yao et al., 2022), personalized shopping (Kim et al., 2026), and customer support (Wang et al., 2025). Seller-side autonomy requires repeated decisions about assortment, pricing, procurement, and inventory whose consequences extend beyond an individual interaction (Shi et al., 2026; Fan et al., 2026). In online-to-offline (O2O) commerce, decisions are made on Web-based management platforms, while their economic returns depend on the costs and constraints of physical execution. For agents managing distributed retail through such platforms, this requires coordinating digital decisions with physical execution over time. Smart vending is a compact instance, as connected machines are increasingly run through cloud management platforms (Fortune Business Insights, 2026) while every restock requires a paid site visit. It also retains the dependencies among assortment, inventory, execution, and cash management. For Web economics, AutoVend thus provides a controlled testbed for seller-side agent-mediated commerce that measures both the returns an agent generates through a Web-based interface and the inference cost of generating them.

![](images/d99740117f378d6a5e316f823148ed9adf623350244df0b1b9caadece7ca709c.jpg)  
Figure 1: AutoVend connects Web-based agent decisions to distributed physical retail through shared inventory, procurement lead times, and paid on-site replenishment.

Sustaining performance in this setting requires three interdependent capabilities. (1) Crosssite Resource Coordination. The agent must match heterogeneous local demand while allocating shared resources and monitoring conditions across the network (Eppen, 1979). An allocation to one site changes the resources available to others, so local decisions must account for consequences across the network. (2) Physical Execution Planning. Stock in a shared warehouse cannot meet local demand until it is placed on machine shelves. Procurement requires advance ordering to account for delivery lead times, while on-site replenishment incurs visit costs. The agent must coordinate advance orders and site visits to maintain product availability while controlling execution expenses, storage costs, and expiration losses (Coelho et al., 2014; Nahmias, 1982). (3) Continual Strategy Improvement. Demand is initially unknown, and observed sales depend on the agent’s own assortment, prices, and stock availability. The agent must use accumulated feedback to refine its demand estimates and revise its decision rules so that subsequent decisions become more effective (Zhang et al., 2026; Shi et al., 2026).

Existing benchmarks provide important foundations for evaluating long-horizon business agents. Vending-Bench studies the operation of a single vending machine and simulates physical tasks through a separate execution agent (Backlund and Petersson, 2025). RetailBench evaluates supermarket operation with inventory aging, delayed deliveries, and cash-flow constraints (Zhang et al., 2026). MerchantBench examines seller-side e-commerce under mixed-latency feedback (Shi et al., 2026), while E-Commerce Bench extends business evaluation to multiple online stores with pooled inventory, fulfillment costs, and delayed settlements (Fan et al., 2026). These evaluations do not establish how effectively agents can coordinate shared inventory across heterogeneous physical retail sites when on-site replenishment itself incurs costs and delays. Assessing this capability requires an environment that brings these constraints together and exposes their cumulative economic consequences.

We introduce AutoVend, a benchmark for evaluating LLM agents on long-horizon autonomous vending operations. A single agent manages 20 vending machines spanning factory, campus, residential, commercial, and office locations, with heterogeneous demand and a shared warehouse. The agent selects products, sets prices, places procurement orders, and arranges replenishment through tool calls that mirror such a platform. The simulator models procurement lead times, paid on-site stocking, product expiration, and storage charges on warehouse stock. These mechanisms connect decisions made through the interface to inventory availability and subsequent economic outcomes.

AutoVend combines terminal net assets with operational metrics and trajectory diagnostics. We examine site coordination, the cost and availability of paid replenishment, and changes in operating performance over time. Our contributions are threefold.

• We introduce AutoVend, a benchmark for long-horizon seller-side agent-mediated commerce, with a unified task, operating interface, and evaluation protocol for managing a distributed retail network.

• We develop a simulator grounded in real vending records that models heterogeneous demand, procurement lead times, inventory aging, and paid site visits, and validate its demand behavior against observed sales patterns.

• We evaluate five LLM agents and find that four of five fall short of a rule-based inventorycontrol baseline. Operational diagnostics and trajectory analyses reveal difficulties in coordinating inventory commitments, execution costs, and policy revision.

## 2 Related Work

Agent Evaluation in Commerce. Commerce is a prominent domain for evaluating LLM agents on the Web, with benchmarks for online shopping (Yao et al., 2022; Peeters et al., 2026; Kim et al., 2026), customer service (Yao et al., 2025; Wang et al., 2025), and multi-agent markets (Bansal et al., 2025; Zheng et al., 2026). Their evaluations typically end with a completed task, dialogue, or transaction, whereas a seller’s decisions affect inventory and cash long after they are made. Seller-side benchmarks instead evaluate the sustained operation of a business mainly by its accumulated economic outcome. Merchant-Bench and StoreBench each cover a single online store (Shi et al., 2026; Raghuvanshi et al., 2026), Business Arena a single cross-border shop (Pan et al., 2026), and E-Commerce Bench up to four online stores that share a warehouse (Fan et al., 2026). In physical retail, RetailBench covers a single supermarket (Zhang et al., 2026) and Vending-Bench a single vending machine (Backlund and Petersson, 2025), a setting that Vending-Bench 2 extends to a simulated year and its Arena to several competing agents with one machine each (Andon Labs, 2025). AutoVend instead provides a controlled setting in which one agent operates 20 heterogeneous sites from a shared warehouse and every on-site action requires a paid visit (Table 1).

Long-Horizon Agent Evaluation. Longhorizon benchmarks commonly lengthen the work toward a defined goal, as in workflows on websites and computers (Zhou et al., 2024; Yuan et al., 2026), tasks in a simulated company (Xu et al., 2025), and exploration of partially observable environments (Luo et al., 2026). Sustained-operation benchmarks instead fix a span of simulated calendar time in which costs accrue and the environment evolves whether or not the agent acts, so performance is an accumulated economic result rather than goal completion (Backlund and Petersson, 2025; He et al., 2026). Because terminal outcomes alone reveal little about why agents succeed or fail, evaluations add progress rates (Ma et al., 2024), capability profiles across operating dimensions (Fan et al., 2026), and attribution of gains and losses to actions (Pan et al., 2026). Continuallearning benchmarks further ask whether agents turn accumulated experience into reusable capabilities (Zheng et al., 2025; Cheng et al., 2026). AutoVend ties its operational dimensions to individual sites and on-site execution and examines how performance changes across operating scales and over time.

## 3 The AutoVend Benchmark

AutoVend places an LLM agent in charge of a network of vending machines over a long operating horizon (Figure 2). The agent sets assortment and prices, orders stock, and arranges on-site replenishment, while customer demand is grounded in real operating data and hidden from the agent.

## 3.1 Task Formulation

An agent operates 20 vending machines that share one warehouse and draw on a catalog of 1,372 products over 180 simulated days, starting with 40,000 CNY in cash. We formulate the task as a partially observable Markov decision process (POMDP) (Kaelbling et al., 1998). The underlying state $s _ { t }$ comprises the simulation date, cash, inventory, outstanding orders, machine assortments and prices, and hidden demand conditions. An action $a _ { t } \in \mathcal A$ denotes a tool invocation. When the agent ends the day’s actions, the simulator settles sales and costs and advances to the next day. The observation kernel $Z ( o _ { t + 1 } \mid s _ { t + 1 } , a _ { t } )$ exposes daily operating summaries and tool results, while demand parameters and unmet demand remain hidden.

The objective is to maximize terminal net assets,

$$
\mathrm { N A } _ { T } = B _ { T } + \sum _ { k } c _ { k } h _ { k , T } ,\tag{1}
$$

where $T$ is the terminal interaction step at the end of the operating episode, $B _ { t }$ is cash, $c _ { k }$ is the unit cost of product $k ,$ and $h _ { k , t }$ counts the unexpired units of k on shelves, in the warehouse, or in transit. Buying stock converts cash into inventory at cost, so net assets change through sales, operating costs, fines, and inventory write-offs from expiration or disposal. With underlying demand and its response parameters hidden, the agent can refine its assortment, pricing, and replenishment strategy using feedback accumulated during operation.

## 3.2 World Simulator

Network and operations. The network comprises six worksite, two campus, four residential, six commercial, and two office machines, with differences in product preference, price sensitivity, and holiday response. Each machine holds at most 200 units and is replenished from one shared warehouse. Procurement is paid at placement. Orders arrive in the warehouse three days after placement. Staff are available for on-site visits on working days, and multiple visits can be made within a day. Replenishment and removal of shelf stock require physical execution through an on-site visit, which costs 35 CNY per machine per day regardless of the number of operations.

Inventory aging and costs. Batches expire 30 or 90 days after warehouse arrival. Selling expired products incurs a fine of 1,000 CNY per unit sold. Warehouse rent is 0.03 CNY per unit per day, and machine rent is 4 CNY per machine per day. The agent must balance stock availability against storage costs and expiration risk, since excess inventory incurs recurring charges and may expire before it is sold.

Customer demand. Demand is generated separately for each machine, with temporal dependence introduced by persistent daily shocks. The attraction of product k on machine m is

$$
\begin{array} { r } { w _ { m , k , t } = b _ { m , k } F _ { m , t } ^ { \eta } \operatorname* { m i n } \left( 1 , \frac { q _ { m , k , t } } { \phi } \right) } \\ { \times \left( \frac { p _ { m , k , t } } { \bar { p } _ { k } } \right) ^ { - \varepsilon _ { \ell ( m ) } } E _ { m , k , t } , } \end{array}\tag{2}
$$

where $b _ { m , k }$ is a base attraction, $F _ { m , t }$ the filled fraction of capacity, $q _ { m , k , t }$ the shelf stock, $p _ { m , k , t }$ the price, $\bar { p } _ { k }$ the reference price, and $\varepsilon _ { \ell \left( m \right) }$ the price elasticity of the site type. Fuller machines attract more customers (Balakrishnan et al., 2004), and an item displayed with fewer than $\phi = 5$ units loses attraction proportionally. The multiplier $E _ { m , k , t }$ captures temperature, rain, and holiday effects. Purchases follow

Table 1: Comparison with long-horizon business operation benchmarks. Op. Units counts the operating units managed by one agent. Diverse Types marks whether these units differ in type. Real Data marks whether the environment is grounded in real commercial datasets. Physical Ops. marks whether actions require physical execution. Open Source marks public release, upon publication for AutoVend.
<table><tr><td>Benchmark</td><td>Domain</td><td> $\mathbf { O p . }$  Units</td><td>Diverse Types</td><td>Real Data</td><td>Physical Ops.</td><td>Open Source</td></tr><tr><td>RetailBench (Zhang et al., 2026)</td><td>Supermarket</td><td>1</td><td>x</td><td></td><td>X</td><td></td></tr><tr><td>MerchantBench (Shi et al., 2026)</td><td>E-commerce</td><td>1</td><td>X</td><td></td><td>X</td><td></td></tr><tr><td>StoreBench (Raghuvanshi et al., 2026)</td><td>E-commerce</td><td>1</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>E-Commerce Bench (Fan et al., 2026)</td><td>E-commerce</td><td> ${ \leq } 4$ </td><td>√</td><td></td><td>X</td><td></td></tr><tr><td>Vending-Bench (Backlund and Petersson, 2025)</td><td>Vending</td><td>1</td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>AutoVend (ours)</td><td>Vending</td><td>20</td><td></td><td></td><td></td><td></td></tr></table>

$$
\begin{array} { c } { { N _ { m , t } \sim \mathrm { P o i s s o n } \Bigl ( \xi _ { m , t } \frac { U _ { m } W _ { m , t } } { U _ { m } + W _ { m , t } } \Bigr ) , } } \\ { { W _ { m , t } = \displaystyle \sum _ { k } w _ { m , k , t } , } } \end{array}\tag{3}
$$

Here, $N _ { m , t }$ counts purchase attempts, and $U _ { m }$ is the machine-specific weight assigned to the option of buying nothing. Each purchase attempt selects product k with probability $w _ { m , k , t } / W _ { m , t } .$ This implements a multinomial logit model with a no-purchase option (Talluri and van Ryzin, 2004). The random multiplier $\xi _ { m , t }$ averages one and is positively correlated across days, allowing demand fluctuations to persist and increasing variability beyond Poisson sampling. Sales equal the minimum of sampled demand and shelf stock, and unmet demand is lost.

## 3.3 Real-World Data Grounding

Data. We build the simulator from six months of order-level sales records covering 99 machines operated by a commercial smart vending company (Appendix C), totaling 220,128 units sold. We select 20 machines across five site types to construct the simulated network operated by the agent. The catalog keeps 1,372 products in 18 subcategories, covering 89.3% of units sold.

Parameter calibration. The base attraction in Eq. (2) factorizes as

$$
b _ { m , k } = \kappa _ { m } \pi _ { \ell ( m ) , g _ { 1 } ( k ) } \gamma _ { g _ { 2 } ( k ) } \omega _ { k } ,\tag{4}
$$

where $g _ { 1 } ( k )$ and $g _ { 2 } ( k )$ are the category and subcategory of product k. The product popularity $\omega _ { k }$ is the number of units sold per unit of exposure, computed from sales records and shrunk toward its subcategory mean by empirical Bayes. Exposure counts a machine’s days of offering the product weighted by its sales level. The site preference $\pi _ { \ell , g _ { 1 } }$ compares the drink and snack mix of labeled machines with the mix expected from the products they actually sold, so that stocking choices are not mistaken for site effects. The coefficients $\gamma _ { g _ { 2 } }$ align predicted subcategory shares on real shelves with realized shares. The scale $\kappa _ { m }$ sets the expected units of a typical real shelf at reference prices and neutral weather equal to the machine’s observed mean daily units.

The no-purchase weight is $U _ { m } = u R _ { m }$ with $u = x e / ( 1 - e )$ , where x is the geometric mean ratio between the attraction of real shelves and that of a fully stocked reference shelf, e is the elasticity of daily units with respect to shelf attraction estimated within machines, and $R _ { m }$ is the attraction of the reference shelf. The reference shelf sorts products by $b _ { m , k }$ and places ϕ units of each until capacity is full. With $x = 0 . 0 6 3 8$ and $e = 0 . 3 1 8$ , a typical real shelf sells the machine’s observed mean daily units, and a fully stocked reference shelf sells about 1.42 times that amount. The shock is $\xi _ { m , t } = \exp ( \sigma u _ { m , t } - \sigma ^ { 2 } / 2 )$ with $\begin{array} { r } { \begin{array} { l } { u _ { m , t } \ = \ \sum _ { j = 0 } ^ { 5 } \rho ^ { j } z _ { m , t - j } / \sqrt { \sum _ { j } \rho ^ { 2 j } } } \end{array} } \end{array}$ , where $z _ { m , t }$ are independent standard normal draws and $\sigma ^ { 2 } = \ln ( 1 + 1 / r )$ . The dispersion $r = 2 . 7 4$ is the median overdispersion $m ^ { 2 } / ( v - m )$ of daily units after removing weekday patterns, and $\rho = 0 . 2 1$ is the lag-one residual autocorrelation. The nested structure of the choice model keeps one nest per subcategory with similarity parameter λ = 1, at which the model reduces to the multinomial logit

![](images/3d0e17e92ad259f9c39c99058ae6f0dca19297690cd1459e9c5844c93029bf35.jpg)  
Figure 2: Overview of AutoVend. (a) Agent and interface, with MCP tools for remote operations, physical execution, and persistent memory. (b) World simulator, connecting a shared warehouse to 20 machines across five site types and generating sales through a data-grounded demand model with hidden parameters. (c) Daily interaction loop, from arrivals and recurring fees through agent queries and actions to sales and visit-cost settlement, followed by advancement to the next day.

## form used above.

Margins, shelf lives, price elasticities, environmental responses, and operating costs are benchmark settings. We set the fill exponent to $\eta = 0 . 5$ the facing threshold to $\phi ~ = ~ 5$ units, and sitespecific price elasticities between 2.4 and 3.5. Unit costs use assumed gross margins of 30–40% for drinks and 50–60% for snacks.

Validation. We re-estimate all data-derived parameters from records preceding an eight-week test window and evaluate the demand model on 67 machines. We reconstruct daily shelves from inventory records for 67 machines. We replay these shelves at reference prices and under neutral weather, comparing simulated with realized sales over 100 seeds (Figure 3).

Let $v _ { t }$ denote the realized sales for machineweek observation t. The weighted absolute percentage error is

$$
\mathrm { W A P E } = \frac { \sum _ { t } \left| v _ { t } - \hat { v } _ { t } \right| } { \sum _ { t } v _ { t } } ,\tag{5}
$$

where $\hat { v } _ { t }$ is the simulated counterpart. Subcategory fidelity is measured by the total variation distance between simulated and realized subcategory shares, and product ranking by the Spearman correlation between simulated and realized product sales within a machine. The persistence reference sets $\hat { v } _ { t }$ to each machine’s mean weekly sales over the preceding four weeks, and the naive reference keeps each machine’s earlier sales level and spreads it uniformly over the products on its shelf. Two ablations show which components matter, removing the overdispersed shock and replacing the estimated popularity with a uniform weight.

Total volume is reproduced within 0.7%, with a machine-week weighted absolute percentage error (WAPE) of 0.473 against 0.606 for a naive reference and 0.561 for a persistence reference that carries forward the preceding four weeks of real sales. The total variation distance (TVD) between simulated and realized subcategory shares is 0.153, compared with 0.206 for the naive reference. Within machines, the median Spearman correlation of product sales is 0.577. A uniform product popularity raises the TVD to 0.238, showing that estimated product popularity improves mix fidelity. Price and environmental responses remain unvalidated by design.

(a) Weekly sales per machine  
![](images/3cfa24f2d22b4df6d859709a0b7c43b255d0073b79980c553db550e9ff06a62b.jpg)

(b) Product mix  
![](images/a05ba5a06181594a0abd4e42189f283473254445a71896cfb1b35c136487d330.jpg)  
Figure 3: Held-out evaluation of the demand model on 67 machines over an eight-week test window, averaged over 100 seeds. (a) Simulated versus realized weekly sales per machine. (b) Subcategory shares of realized and simulated sales. Subcategories accounting for less than 0.5% of realized sales are pooled into Other, while TVD uses all original subcategories.

## 3.4 Agent Interface

Interaction loop. AutoVend exposes the remote operations available to a networked-vending operator as tools over the Model Context Protocol (MCP) (Anthropic, 2024), with physical tools standing in for dispatching staff on site. Each day begins with due order arrivals and charges for warehouse rent and fixed operating fees, followed by a daily observation for the agent. During the action window, the agent queries operating records and executes actions, receiving query results or execution feedback after each tool call. Valid actions update the operating state immediately. When the agent closes the action window, the simulator realizes demand using the resulting shelf inventory and prices, settles sales and on-site visit costs, and proceeds to the next day.

Prompts. A fixed system prompt specifies the objective, starting conditions, goods flow, operating costs, inventory and pricing constraints, and qualitative demand mechanisms without prescribing an operating strategy. A daily user prompt updates the date and supplies a compact report of cash and net assets, recent profit and loss, warehouse and in-transit stock, upcoming holidays, and each machine’s inventory, sales, and expiry status. Product-level details are retrieved through queries rather than included in the daily prompt. Appendices A and B provide the complete prompts and tool specifications.

Tools and memory. AutoVend provides 18 tools organized into three functional groups covering remote operations, physical execution, and persistent

memory.

• Remote operational tools. These support procurement, machine-specific pricing, warehouse disposal, and closing the daily action window. Queries provide product attributes and costs, warehouse and shelf inventory, product-level sales within and across machines, and financial records. Read-only SQL supports custom analysis of operating records, while calendar and weather queries provide published holidays and near-term weather information.

• Physical execution tools. These transfer warehouse stock onto machine shelves and return specified quantities or all remaining units of a product to the warehouse. Execution requires paid on-site visits, charged once per machine per day regardless of the number of operations.

• Memory tools. These allow the agent to read and overwrite a persistent notebook containing selected observations, plans, and findings. Conversation history continues across days, while the notebook survives context compaction.

## 4 Experiments

## 4.1 Experimental Setup

Baseline. We evaluate Claude Sonnet 5.5 (Anthropic, 2026), GPT-6.1 Sol (OpenAI, 2026), DeepSeek V4.1 Flash (DeepSeek-AI, 2026), Kimi K3 (Kimi Team, 2026), and GLM-5.3-Flash (Z.ai,

2026). We use the Pi harness (Zechner, 2026) and configure each model with its highest supported reasoning effort. Each agent operates the same 20-machine network for 180 simulated days, starting with 40,000 CNY and using identical prompts, tools, and daily interaction budgets. We report means and the standard deviation of terminal net assets over three runs per model, all in one frozen world with the same random seed. Variation across LLM runs thus reflects stochastic agent decisions and the demand draws they induce. The deterministic rule-based baseline attains 208.73 ± 2.26 thousand CNY (CV 1.1%) across seeds 1–3, and we report seed 1.

The rule-based baseline combines the newsvendor critical ratio (Arrow et al., 1951) with periodic order-up-to replenishment (Silver and Robb, 2008). It sets inventory targets from rolling 30-day demand and variability estimates, balancing lost sales against holding and spoilage costs, and replenishes sites weekly. Procurement accounts for delivery lead times, available stock, and outstanding orders. Assortments follow observed sales, while prices are adjusted through local gross-profit trials. All decisions use observable information.

Metrics. Our primary outcome, Net Assets, is terminal cash plus unexpired inventory valued at cost across shelves, the warehouse, and transit. Assortment compares expected gross profit with a heuristic reference mix under common prices, display volume, and environmental conditions. Waste Rate is inventory loss divided by valid sales cost plus loss, counting expiry and early disposal once. Pricing Regret is the expected gross-profit shortfall from a price-search reference at a fixed site, assortment, and inventory.

Shelf Utilization measures the mean fraction of capacity occupied by unexpired stock before sales across sites and days. We also report the share of site-days with empty shelves. Capital Efficiency is mean daily gross profit divided by mean inventory value at cost. Tool Calls counts tool invocations over the episode. API Efficiency averages per-run net asset growth per USD of estimated API expenditure, using standard reference rates for recorded input, output, cache, and compaction usage.

## 4.2 Main Results

Overall Performance. Long-horizon business management remains challenging, with four of five evaluated models underperforming the rule-based baseline in Table 2. GPT-6.1 Sol is the sole exception, while all five models obtain lower assortment scores than the baseline. All three GPT executions exceed the baseline, whereas none of the other 12 does, showing that most models’ weak averages cannot be explained by a few isolated failures.

Variability. Weak average performance is compounded by inconsistent execution. Kimi K3 exhibits a coefficient of variation of 45.6% across repeated runs. GPT-6.1 Sol, GLM-5.3-Flash, and Claude Sonnet 5.5 instead exhibit similar relative dispersion of around 11%, despite their substantially different mean returns. This contrast shows that consistency and economic effectiveness are distinct aspects of agent capability.

API Efficiency. Operating returns and inference efficiency reveal different model rankings. GPT-6.1 Sol achieves the highest terminal assets, whereas DeepSeek delivers the highest net-asset growth per dollar of estimated API expenditure. GPT’s mean estimated API expenditure is approximately 80 times DeepSeek’s, while its net-asset growth is about 2.9 times as large. GLM-5.3-Flash ranks second on both measures. DeepSeek also uses more tool calls than GLM while achieving higher API Efficiency, showing that interaction count alone does not capture inference cost.

## 5 Analysis

## 5.1 Operational Performance Profiles

To understand the operational strengths and weaknesses underlying the agents’ economic outcomes, we analyze their performance across five dimensions in Figure 4. No single model leads across all five dimensions, and stronger profitability does not consistently coincide with better inventory control, pricing, or capital efficiency. We examine these differences below to characterize how each agent combines product selection, inventory management, and resource use during long-horizon operations.

Profitability. GPT-6.1 Sol achieves the highest mean terminal net assets at 309.35 thousand CNY, followed by GLM-5.3-Flash at 148.43 thousand CNY. Kimi K3 achieves the lowest return at 91.52 thousand CNY.

Assortment. GPT-6.1 Sol leads the LLM agents in assortment quality, scoring 0.992. GLM-5.3- Flash follows at 0.817, while Claude Sonnet 5.5 has the lowest score at 0.500.

Inventory. Claude Sonnet 5.5 and GLM-5.3- Flash record the lowest waste rates at 0.16% and

Table 2: Main experimental results on AutoVend. LLM results are averaged over three repeated runs, and Rule-based reports seed 1. Net Assets and Std are in thousands of CNY. Waste Rate and Shelf Utilization are percentages, Pricing Regret is in CNY per machine per day, and API Efficiency is net-asset growth in CNY per USD of API cost.
<table><tr><td>Model</td><td>Net Assets ↑</td><td>Std</td><td>Assortment ↑</td><td>Waste Rate↓</td><td>Pricing Regret↓</td><td>Shelf Utilization</td><td>Tool Calls</td><td>API Efficiency ↑</td></tr><tr><td>GPT-6.1 Sol</td><td>309.35</td><td>34.09</td><td>0.992</td><td>0.60</td><td>30.18</td><td>88.91</td><td>3,950</td><td>1,117</td></tr><tr><td>GLM-5.3-Flash</td><td>148.43</td><td>16.70</td><td>0.817</td><td>0.27</td><td>2.45</td><td>60.65</td><td>670</td><td>7,023</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>133.92</td><td>27.20</td><td>0.622</td><td>1.85</td><td>4.22</td><td>82.40</td><td>2,118</td><td>31,010</td></tr><tr><td>Claude Sonnet 5.5</td><td>103.04</td><td>11.33</td><td>0.500</td><td>0.16</td><td>1.32</td><td>32.48</td><td>1,013</td><td>3,235</td></tr><tr><td>Kimi K3</td><td>91.52</td><td>41.75</td><td>0.661</td><td>0.60</td><td>3.29</td><td>42.29</td><td>517</td><td>1,920</td></tr><tr><td>Rule-based</td><td>210.97</td><td>0.00</td><td>1.034</td><td>0.17</td><td>4.32</td><td>64.65</td><td>1,624</td><td>N/A</td></tr></table>

![](images/b41333346543a2339c25b113fd418f87fccf7cdd3915996287a0c2f47343dfd1.jpg)  
Figure 4: Whole-episode operational profiles over 180 days. Each panel shows the mean operational profile of an LLM or the rule-based baseline. Profitability, Assortment, Inventory, Pricing, and Capital Efficiency correspond to terminal net assets, assortment quality, reversed waste rate, reversed pricing regret, and daily gross profit per unit of inventory capital, respectively. Each axis is min-max normalized across the five model means and the baseline, with larger values indicating better relative performance.

0.27%, respectively. DeepSeek V4.1 Flash has the highest rate at 1.85%. Inventory availability also differs substantially. Claude’s low waste accompanies empty shelves on 25.21% of site-days, whereas DeepSeek’s machines are empty on only 1.67%, combining broader inventory availability with greater losses.

Pricing. Claude Sonnet 5.5 achieves the lowest conditional pricing regret at 1.32 CNY per machine per day, followed by GLM-5.3-Flash at 2.45. GPT-6.1 Sol has a much larger gap at 30.18, while all other models fall between 1.32 and 4.22. Its operational profile thus combines the strongest realized profitability with the largest conditional pricing gap.

Capital Efficiency. Claude Sonnet 5.5 generates the highest daily gross profit per unit of inventory capital at 0.0589, followed by GLM-5.3-Flash at 0.0362.

## 5.2 RQ1: Can LLM agents effectively manage long-horizon business operations?

Most evaluated agents can establish profitable operations, but reliable long-horizon outperformance remains limited. All 15 LLM executions finish above the initial capital, whereas only GPT-6.1 Sol exceeds the rule-based baseline in all three executions. The trajectories in Figure 5(a) also distinguish sustained growth from early gains followed by stagnation, particularly for Kimi K3.

![](images/207b4ba8dd08f5ab94661047fa28603e236b411f8ebd129a4e83f74c4ae97da7.jpg)

![](images/8adca3c577c3acba52294fdd2f2546ffa1dc2e2c3ea9d9e158e597854913d747.jpg)

![](images/919ad1cee82fd9faaba0bbfcde5386699934ae53b9de95b059443420dbda004c.jpg)  
Figure 5: Long-horizon operations, site coordination, and physical execution. (a) Mean net-asset trajectories across three runs per LLM, compared with the rule-based baseline. (b) Mean reassignment gaps across six checkpoints, with whiskers showing the range across three executions and a fixed baseline. (c) Availability and visit-cycle returns, averaged across executions. Contributions exclude final incomplete cycles and costs other than visit fees.

## 5.3 RQ2: Can agents coordinate resources across sites?

Site coordination remains incomplete even for the best-performing agent. We evaluate this through the best reassignment of existing assortments across sites at six 30-day checkpoints, using the standardization of the assortment metric. GPT’s actual assignment falls 7.05% below this restricted optimum, compared with 2.61% for the baseline in Figure 5(b). Its strong aggregate return therefore coexists with opportunities to improve where its selected products are offered.

Because this diagnostic introduces no new products, the gap identifies unrealized value in matching existing offerings to heterogeneous sites. Conversely, a small reassignment gap does not establish strong coordination when the underlying assortments are weak. DeepSeek and Kimi illustrate this distinction through their limited reassignment gains and low assortment scores. Coordination must therefore be assessed jointly with product quality, since efficient placement cannot compensate for unsuitable offerings.

## 5.4 RQ3: Can agents plan physical execution?

Agents do not consistently balance local availability with economical execution. Over days 16–180, GPT and DeepSeek both avoid completely empty shelves, but GPT achieves approximately twice DeepSeek’s visit-cycle contribution per site-day in Figure 5(c). Sonnet and Kimi instead exhibit persistent empty-site exposure despite holding corresponding products in the warehouse.

These outcomes reflect two separate requirements of physical execution. Central inventory becomes available to customers only after a paid on-site transfer, so procurement alone cannot maintain local supply. Conversely, visits that preserve availability can absorb a substantial share of gross profit when they are too costly relative to the sales they support. Effective execution requires coordinating replenishment timing and visit frequency with local demand. Cycle returns, measured as sales gross profit minus visit fees divided by cycle duration, must therefore be interpreted alongside availability and the costs excluded from this measure.

## 5.5 RQ4: Can agents improve through feedback?

The trajectories provide limited evidence of consistent economic improvement as feedback accumulates. Comparing days 16–45 with days 136–165 in Figure 6, daily asset growth decreases for all five model means and in 13 of the 15 executions. Meanwhile, several agents improve individual operating metrics. Better waste control or pricing therefore does not consistently translate into a stronger overall operating policy.

An accounting decomposition helps explain this disconnect. DeepSeek generates more gross profit, but additional warehouse and visit expenses offset most of that gain. These patterns suggest that local corrections can coexist with increasingly costly inventory and execution commitments. However, changing environmental conditions and inventory states prevent this temporal comparison from causally identifying whether feedback itself improves the agents’ decisions.

![](images/4ea71b6e4051e1342b3982e808ce73a1a925dcf23809deab16e2c8c922c2651b.jpg)  
(c) Waste

![](images/e133a7d5dda0481cfe457bc86e16039619a0fb8407f6fcfd27c9d67ba528d371.jpg)

![](images/d621102b109725dcb8e3578234d1106aa610e6e2f802dbc65e2e363d90e3064e.jpg)

(d) Pricing gap  
![](images/c2ce664fe1ac3094c92ac6214428eff576b28eceacae6aa67e6dc73e04922070.jpg)  
Figure 6: Early- and late-period operating outcomes. Markers show mean LLM outcomes and rule-based baseline outcomes.

## 5.6 Failure Modes

Trajectory inspection identifies four observed failures in action feasibility, inventory cost planning, physical execution, and policy revision. These cases illustrate behavior without establishing prevalence or causal effects.

Incomplete Grounding of Action Plans. In one Kimi K3 run, all 49 allocations in a day 170 restocking batch were rejected because the requested products lacked sufficient warehouse stock. The agent had observed 4,701 warehouse units in aggregate and attempted replenishment across seven sites while an order remained in transit. After rejection, it ended the day and waited for delivery. The episode shows why allocation plans require product-level availability checks.

Insufficient Planning for Inventory Costs. In another Kimi run, six purchases between days 136 and 169 repeated an identical bundle of 21 products. Although core products required replenishment, the bundle also included products with no recorded sales during this interval. For one such product, 2,158 unexpired units were already in the warehouse before day 169, yet the agent ordered another 80 units. Procurement thus continued to expand inventory that was not being deployed or consumed.

Fragmented Physical Execution. In one

DeepSeek V4.1 Flash run, a campus machine generated only 5.14 CNY of gross profit between visits on days 125 and 126, against a 35 CNY visit fee. The agent returned on day 126 to add just four units, incurring another visit fee. This sequence illustrates how frequent responses to small stock changes can create interventions whose costs are poorly matched to site-level turnover.

Ineffective Use of Operating Feedback. GLM-5.3-Flash illustrates the difference between immediate error recovery and persistent policy revision. On day 163, four allocations requesting four units each were rejected because only three units of the product remained in the warehouse. The agent then successfully allocated those three units to one site. However, the same four infeasible allocations reappeared on day 168, followed by the same local repair. The feedback supported correction within each interaction but did not prevent the allocation pattern from recurring.

## 6 Conclusion

We introduced AutoVend, a benchmark for longhorizon autonomous business management across 20 heterogeneous vending sites with shared inventory, procurement lead times, and paid physical execution. Evaluation of five LLM agents over 180 simulated days shows that four of five underperform a rule-based inventory-control baseline. Operational diagnostics reveal that strong aggregate returns can coexist with coordination gaps, while improvements in individual metrics need not produce stronger economic growth. The trajectory cases further expose difficulties in verifying feasible allocations, adapting procurement, consolidating visits, and retaining corrections across decisions. Future work should test these findings, drawn from three runs per model in one environment, across varied operating conditions and controlled interventions such as explicit state verification and persistent policy updates.

## References

Andon Labs. 2025. Vending-Bench 2.

Anthropic. 2024. Introducing the Model Context Protocol.

Anthropic. 2026. Introducing Claude Sonnet 5.5. Technical blog.

Kenneth J. Arrow, Theodore Harris, and Jacob

Marschak. 1951. Optimal inventory policy. Econometrica, 19(3):250–272.

Axel Backlund and Lukas Petersson. 2025. Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents. Preprint, arXiv:2502.15840.

Anantaram Balakrishnan, Michael S. Pangburn, and Euthemia Stavrulaki. 2004. “Stack Them High, Let ’em Fly”: Lot-Sizing Policies When Inventories Stimulate Demand. Management Science, 50(5):630–644.

Gagan Bansal, Wenyue Hua, Zezhou Huang, Adam Fourney, Amanda Swearngin, Will Epperson, Tyler Payne, Jake M. Hofman, Brendan Lucier, Chinmay Singh, Markus Mobius, Akshay Nambi, Archana Yadav, Kevin Gao, David M. Rothschild, Aleksandrs Slivkins, Daniel G. Goldstein, Hussein Mozannar, Nicole Immorlica, and 5 others. 2025. Magentic Marketplace: An open-source environment for studying agentic markets. Preprint, arXiv:2510.25779.

Zerui Cheng, Jiawei Xu, Huacan Chai, Jiayang Sun, Pramod Viswanath, and Maxm Pan. 2026. AhaBench: Do agents turn experience into reusable insights? A long-horizon benchmark for continual learning. Preprint, arXiv:2609.05435.

Leandro C. Coelho, Jean-François Cordeau, and Gilbert Laporte. 2014. Thirty years of inventory routing. Transportation Science, 48(1):1–19.

DeepSeek-AI. 2026. DeepSeek-V4.1-Flash: Pushing the limits of KV cache compression. Preprint, arXiv:2609.19969.

Gary D. Eppen. 1979. Note—effects of centralization on expected costs in a multi-location newsboy problem. Management Science, 25(5):498–501.

Wei Fan, Xinjie Shen, Xudong Guo, Jianhong Tu, Yang Su, Yinger Zhang, Lianghao Deng, Fengyu Wang, Baohua Dong, Yangqiu Song, and Dayiheng Liu. 2026. E-Commerce Bench: Evaluating LLM Agents on Long-Horizon Autonomous Business Operation. Preprint, arXiv:2608.30730.

Fortune Business Insights. 2026. Intelligent vending machine market size, share & industry analysis, 2026– 2034. Report FBI107360.

Muyu He, Adit Jain, Anand Kumar, Vincent Tu, Soumyadeep Bakshi, Sachin Patro, and Nazneen Rajani. 2026. YC-Bench: Benchmarking AI agents for long-term planning and consistent execution. In Third Conference on Language Modeling.

Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. 1998. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1–2):99–134.

Sunghwan Kim, Ryang Heo, Yongsik Seo, Jinyoung Yeo, and Dongha Lee. 2026. AgenticShop: Bench marking agentic product curation for personalized

web shopping. In Proceedings of the ACM Web Conference 2026, WWW ’26, pages 2489–2500, New York, NY, USA. Association for Computing Machinery.

Kimi Team. 2026. Kimi K3: Open frontier intelligence. Preprint, arXiv:2607.24653.

Haotian Luo, Huaisong Zhang, Xuelin Zhang, Haoyu Wang, Zeyu Qin, Wenjie Lu, Guozheng Ma, Haiying He, Yingsha Xie, Qiyang Zhou, Zixuan Hu, Hongze Mi, Yibo Wang, Naiqiang Tan, Hong Chen, Yi R. Fung, Chun Yuan, and Li Shen. 2026. UltraHorizon: Benchmarking agent capabilities in ultra longhorizon scenarios. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 83084–83136. PMLR.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. AgentBoard: An analytical evaluation board of multi-turn LLM agents. In Advances in Neural Information Processing Systems, volume 37, pages 74325–74362, Red Hook, NY, USA. Curran Associates, Inc.

Steven Nahmias. 1982. Perishable inventory theory: A review. Operations Research, 30(4):680–708.

OpenAI. 2026. GPT-6.1 Sol. Model documentation.

Yijun Pan, Yukun Lian, Kunyu Shi, Junbo Li, Hongwei Xue, Sicong Xie, Guannan Zhang, and Xiaoying Xing. 2026. Business Arena: Benchmarking LLM agents in a realistic marketplace. Preprint, arXiv:2608.08621.

Ralph Peeters, Aaron Steiner, Luca Schwarz, Julian Yuya Caspary, and Christian Bizer. 2026. Web-Mall – a multi-shop benchmark for evaluating web agents. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’26, pages 3342–3348, New York, NY, USA. Association for Computing Machinery.

Daksh Raghuvanshi, Ved Vedere, and Yifan Wang. 2026. StoreBench: A live-commerce environment for evaluating and training autonomous operator agents. Preprint, arXiv:2610.10942.

Qiming Shi, Yulong Tao, Linbo Jin, Zhaolu Kang, Yibo Dou, Jiawen Zhu, Tianjun Pan, Shaokang Fu, Chengyu Wang, Siyue Li, Yaping Cheng, Di Weng, and Chengfu Huo. 2026. MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations. Preprint, arXiv:2607.28956.

Edward A. Silver and David J. Robb. 2008. Some insights regarding the optimal reorder period in periodic review inventory systems. International Journal ofProduction Economics, 112(1):354–366.

Kalyan Talluri and Garrett van Ryzin. 2004. Revenue management under a general discrete choice model of consumer behavior. Management Science, 50(1):15– 33.

Haoxin Wang, Xianhan Peng, Huang Cheng, Yizhe Huang, Ming Gong, Chenghan Yang, Yang Liu, and Jiang Lin. 2025. ECom-Bench: Can LLM agent resolve real-world E-commerce customer support issues? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 276–284, Suzhou (China). Association for Computational Linguistics.

Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Melroy Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, and 2 others. 2025. TheAgentCompany: Benchmarking LLM agents on consequential real world tasks. In Advances in Neural Information Processing Systems, volume 38, pages 10291–10321, Red Hook, NY, USA. Curran Associates, Inc.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. WebShop: Towards scalable realworld web interaction with grounded language agents. In Advances in Neural Information Processing Systems, volume 35, pages 20744–20757, Red Hook, NY, USA. Curran Associates, Inc.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik R Narasimhan. 2025. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. In The Thirteenth International Conference on Learning Representations.

Mengqi Yuan, Zilong Zhou, Xinzhuang Xiong, Weiming Wu, Jiayang Sun, Jiamin Song, Kaiqian Cui, Bowen Wang, Haoyuan Wu, Yitong Li, Dunjie Lu, Haikong Lu, Qi Zhen, Xinyuan Wang, Jiaqi Deng, Yuhao Yang, Cheng Chen, Boyuan Zheng, Alex Su, and 17 others. 2026. OSWorld 2.0: Benchmarking computer use agents on long-horizon real-world tasks. Preprint, arXiv:2606.29537.

Z.ai. 2026. GLM-5.3-Flash. Model card.

Mario Zechner. 2026. Pi. Software, version 1.1.0.

Linghua Zhang, Jun Wang, Jingtong Wu, and Zhisong Zhang. 2026. RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments. Preprint, arXiv:2603.16453.

Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, and Qianli Ma. 2025. LifelongAgentBench: Evaluating LLM agents as lifelong learners. Preprint, arXiv:2505.11942.

Yushuo Zheng, Huiyu Duan, Zicheng Zhang, Yucheng Zhu, Xiongkuo Min, and Guangtao Zhai. 2026.

Market-Bench: Benchmarking large language models on economic and trade competition. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 39893–39906, San Diego, California, United States. Association for Computational Linguistics.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. WebArena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations.

## A Prompts

volume per day.   
A.1 System Prompt Every unit of every SKU takes exactly   
one unit of volume, so rent is charged   
You operate a network of 20 smart per unit held and a machine’s   
vending machines for 180 days, capacity is a count of units.   
starting with 40000 in cash. On top of rent you pay a fixed fee of   
Each day is one turn: you receive an 80 per day whether or not you act.   
observation, may call tools, then call   
end\_day. On-site actions (restock, reduce,   
delist) cost 35 per device per day   
## Objective touched -- a device touched any number   
Maximize net assets at the end of the of times in one day is charged once.   
run. Ordering, repricing, discarding and   
all queries are remote and cost   
Net assets = cash + the cost value of   
every unexpired unit you hold, nothing.   
wherever it sits: on a shelf, in the Each tool description says which kind   
warehouse, or in transit. it is.   
## Shelf life   
- The only thing that increases it is   
selling above unit cost. Each SKU has a ‘shelf\_life\_days‘ value   
- The things that decrease it: in the catalog, counted from the day   
warehouse rent, labor, the daily fee, a unit arrives in the warehouse.   
expired-sale fines, and stock that Restocking, reducing and delisting do   
expires or is discarded unsold. not reset that clock.   
- Buying stock moves value from cash Expired units do not disappear, and   
into inventory; it is not by itself a they are worth nothing in net assets.   
gain or a loss.   
- Expired units on a shelf still sell,   
## Starting state and every expired unit sold costs an   
On day 1 every shelf and the warehouse extra 1000.   
Delisting sends them back to the   
are empty, no order is in transit,   
warehouse, which stops the fine   
and there is no sales history.   
exposure but does not make them   
The catalog is the only thing that   
exists: 1372 SKUs. sellable again.   
Nothing sells until you order stock, - Expired units in the warehouse keep   
wait for it to arrive, and restock a taking up volume and incurring rent.   
They cannot be restocked into   
device on site.   
anything but more expired stock;   
## Goods flow discarding is the only way to remove   
order -> in transit -> warehouse -> them.   
restock (on site) -> shelf -> sale   
## Funding   
Ordered units are debited from cash Cash can go negative.   
immediately. Rent, the daily fee, visits and   
- The supplier promises 3 days; actual purchases keep being charged, no   
interest is added, and the run does   
delivery may be earlier or later.   
Units in transit cannot be shelved not end.   
and cannot serve a customer. A negative balance counts against net   
Arrivals by day and any delayed assets one for one, so buying on   
orders are in the daily observation. credit does not change net assets by   
Units land in the warehouse. itself -- it moves cash into inventory   
The warehouse is the only pool a at cost.   
What it does cost is rent for every   
restock can draw from.   
- Reducing or delisting returns units day the stock sits, and 3 days before   
from a shelf to the warehouse; they it can be sold.   
are not destroyed.   
## Environment   
- A unit leaves your system only when   
a customer buys it or when you discard Shanghai, CN (Asia/Shanghai).   
it. All amounts are in CNY.   
Temperature, rain and public holidays   
## Prices affect demand, differently by product   
Prices are per (device, SKU), and each category and by site.   
of the 20 machines is configured   
independently. ## Demand   
- Fill level: a fuller device sells   
A price may be set anywhere from 0.50x   
to 2.00x the SKU’s suggested price; faster.   
Free volume depresses demand for   
out-of-band prices are rejected.   
every item on that device, not only   
## Costs for the missing ones.

- Depth: an item’s sales stop   
increasing beyond a few units on the   
shelf; further units only accrue rent.   
- Noise: daily sales are random around   
that item’s underlying rate, and that   
rate is what a longer window of   
observations estimates.   
- Demand rate, price response and   
product fit all vary with location   
type.   
A product that sells at one site can   
be dead stock at another, and the   
same price change can help at one site   
and hurt at another.   
## Memory   
Saved notes are never trimmed or   
shortened.   
They are limited to 6,000 characters.   
## Operating loop   
There is no human user: no one will   
prompt you, approve a plan, or flag a   
problem.   
When the day’s work is done, call   
end\_day.

## A.2 User Prompt

Daily user prompt template.

Day {day} of {horizon\_days} -- {   
weekday}, {date}.   
{observation}   
Keep running the business to maximize   
your net assets.   
When today’s work is done, call   
end\_day.

Initial daily prompt example. The initial daily prompt example is generated from the simulator’s starting state rather than taken from a historical agent run.

```csv
Day 1 of 180 -- Wednesday, 2025-01-01.
{
"balance": {
"cash":39920.0,
"shelf":0.0,
"warehouse":0,
"in_transit":0.0,
"inventory_at_cost":0.0,
"net_assets":39920.0
},
"pnl": {
"columns":[
"window","days","units_sold","
revenue","gross_profit",
"rent","labor","daily_fee","
expired_fine","net"
],
"rows":[
["yesterday
",0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
["last_7d
",0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
```

```csv
["last_30d
",0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
]
},
"devices": {
"columns":[
"device_id","location","capacity","
units_on_shelf",
"n_skus","empty_slots","
sold_yesterday","sold_7d_avg",
"expired_units","expiring_7d_units
","days_since_visit"
],
"rows":[
["CN_WORKSITE_01","worksite/canteen
",200,0,0,0,0,0.0,0,0,null],
["CN_WORKSITE_02","worksite/canteen
",200,0,0,0,0,0.0,0,0,null],
["CN_WORKSITE_03","worksite/canteen
",200,0,0,0,0,0.0,0,0,null],
["CN_WORKSITE_04","worksite/
plant_floor",200,0,0,0,0,0.0,0,0,null],
["CN_WORKSITE_05","worksite/
plant_floor",200,0,0,0,0,0.0,0,0,null],
["CN_WORKSITE_06","worksite/
plant_floor",200,0,0,0,0,0.0,0,0,null],
["CN_CAMPUS_01","campus/university
",200,0,0,0,0,0.0,0,0,null],
["CN_CAMPUS_02","campus/
school_dormitory",200,0,0,0,0,0.0,0,0,
null],
["CN_RESIDENTIAL_01","residential/
apartment",200,0,0,0,0,0.0,0,0,null],
["CN_RESIDENTIAL_02","residential/
apartment",200,0,0,0,0,0.0,0,0,null],
["CN_RESIDENTIAL_03","residential/
apartment",200,0,0,0,0,0.0,0,0,null],
["CN_RESIDENTIAL_04","residential/
dormitory",200,0,0,0,0,0.0,0,0,null],
["CN_COMMERCIAL_01","commercial/
sports_venue",200,0,0,0,0,0.0,0,0,null
],
["CN_COMMERCIAL_02","commercial/
hotel",200,0,0,0,0,0.0,0,0,null],
["CN_COMMERCIAL_03","commercial/
hotel",200,0,0,0,0,0.0,0,0,null],
["CN_COMMERCIAL_04","commercial/
sports_venue",200,0,0,0,0,0.0,0,0,null
],
["CN_COMMERCIAL_05","commercial/
leisure_venue",200,0,0,0,0,0.0,0,0,
null],
["CN_COMMERCIAL_06","commercial/
hotel",200,0,0,0,0,0.0,0,0,null],
["CN_OFFICE_01","office/
industrial_park",200,0,0,0,0,0.0,0,0,
null],
["CN_OFFICE_02","office/
office_building",200,0,0,0,0,0.0,0,0,
null]
]
},
"warehouse": {
"units":0,
"n_skus":0,
"value_at_cost":0,
"rent_per_day":0.0,
```

"expired\_units":0,   
"expiring\_7d\_units":0   
},   
"in\_transit": {   
"orders":0,   
"units":0,   
"value\_at\_cost":0.0,   
"arrivals":[],   
"delayed":[]   
},   
"calendar": {   
"today":{"is\_workday":false,"   
holiday\_name":"\u5143\u65e6"},   
"next\_14d":[]   
}   
}   
Keep running the business to maximize   
your net assets.   
When today’s work is done, call   
end\_day.

## B Tools

A question mark denotes an optional parameter. The six inventory and pricing mutation tools accept batches of 1–100 items, with quantities specified as positive integers. Structurally malformed batches are rejected before execution. For a structurally valid batch, operational constraints are checked separately for each item, so valid items execute even when other items are rejected. Results report accepted and rejected items together with rejection reasons. Rejected items do not change inventory or trigger a visit. The three physical tools incur one visit charge per machine per day, regardless of the number of successful operations. Query windows are restricted to 1, 3, 7, or 30 days, with a default of 30 days. Paginated queries default to 20 rows per page and allow at most 50. A notebook write exceeding the character limit is rejected without replacing the existing notes.

## C Ethical Use of Data

The order-level sales records used to build the simulator were provided by a commercial smart vending company with its authorization. The records contain no personal information about customers or staff. We release only the calibrated simulator parameters, not the raw records.

Table 3: Tools available to the AutoVend agent. A question mark denotes an optional parameter.
<table><tr><td>Tool</td><td>Parameters</td><td>Function and constraints</td></tr><tr><td colspan="3">Remote operational tools</td></tr><tr><td>place_order</td><td>items[{sku_id, qty, supplier_id?}]</td><td>Order warehouse stock and debit cash immediately The default supplier is default.</td></tr><tr><td>adjust_price</td><td>items[{device_id, sku_id, price}]</td><td>Set a machine-specific price within 0.5–2.0 times the reference price. The product need not already be stocked.</td></tr><tr><td>discard</td><td>items[{sku_id, qty}]</td><td>Destroy warehouse units, taking the earliest-expiring units first.</td></tr><tr><td>end_day</td><td>None query?,</td><td>Close the daily action window. category_11?, Search public product attributes and warehouse avail-</td></tr><tr><td>get_catalog</td><td>category_12?, in_warehouse_only?, sort_by?, page?,</td><td>ability. Sorting uses arbitrary catalog order, ascending price_min?, price_max?, or descending reference price, or reference margin.</td></tr><tr><td>get_categories</td><td>page_size? category_l1?</td><td>Return subcategory counts, reference price ranges, and median reference margins.</td></tr><tr><td>get_warehouse</td><td>page?,page_size?</td><td>Return warehouse quantities, costs, and expiry infor- mation by product.</td></tr><tr><td>get_device_detail</td><td>device_id, window_days?</td><td>Return product-level shelf quantities, prices, daily sales, and expiry information for one machine.</td></tr><tr><td>get_sku_across_devices</td><td>sku_id, window_days?</td><td>Return a product&#x27;s stock and realized sales across ma- chines, together with site types.</td></tr><tr><td>get_sales_history</td><td>window_days?</td><td>Return daily network sales, operating costs, visits, and the top 20 products by realized units sold in the</td></tr><tr><td>query_sq1</td><td>sq1</td><td>window. Execute a read-only SQLite query over operating records. Return at most 500 rows and stop queries exceeding two seconds.</td></tr><tr><td>get_calendar</td><td>days_ahead?</td><td>Return already published holidays and makeup work- days. The default horizon is 60 days, with a range of</td></tr><tr><td>get_weather</td><td>None</td><td>1–400. Return settlement weather for today and the following seven days.</td></tr><tr><td colspan="3">Physical execution tools</td></tr><tr><td>restock</td><td>items[{device_id, sku_id, qty}]</td><td>Transfer warehouse units to a machine, subject to stock availability and shelf capacity.</td></tr><tr><td>reduce</td><td>items[{device_id, sku_id, qty}]</td><td>Return a specified shelf quantity to the warehouse, taking the earliest-expiring units first.</td></tr><tr><td>delist</td><td>items[{device_id, sku_id}]</td><td>Return all remaining shelf units of a product to the warehouse.</td></tr><tr><td colspan="3">Memory tools</td></tr><tr><td>read_memory write_memory</td><td>None text</td><td>Read the persistent notebook in full. Replace the notebook with at most 6,000 characters.</td></tr><tr><td></td><td></td><td>An empty string clears it.</td></tr></table>