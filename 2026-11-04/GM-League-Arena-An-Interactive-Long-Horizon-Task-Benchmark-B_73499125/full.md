# GM League Arena: An Interactive Long-Horizon Task Benchmark Based on Professional Sports League Management

Junkai Zhou, Ziming Pan, Zhaoyi Zhang

Long-horizon decision-making under sustained competition requires large language models (LLMs) to balance immediate investment with future resource needs. However, local success in multi-step interactions can not adequately capture cumulative decision effects or demonstrate long-term goal achievement under resource constraints and sustained competition. To address this gap, we introduce GM Bench, a benchmark for managing professional sports teams across seasons under resource constraints and competition among teams. We build the environment simulators for four sports and add financial constraints, player demands, and market mechanisms to establish a unified task across sports. Diverse rosters of comparable strength are constructed from realworld player data, and each LLM is evaluated as a team general manager to assess its long-term management capabilities. We evaluate competitive performance, cash reserves, and cross-season improvement across four sports. Over five NBA seasons, Claude Opus 5.5 ranks first among 29 LLMs, while GPT-6 Astra and GPT-6.1 Sol combine strong performance with larger cash reserves.

## 1 Introduction

In complex environments, large language models need to understand rules, take actions, and revise decisions using feedback. Recent benchmarks span scientific tasks [3, 10], web interaction [4, 13], and long-horizon games [11, 15]. In long-horizon competition, participant interactions shape resource values and action opportunities, and strategies and demands evolve with the environment; models must balance current investment, future resources, and subsequent choices. Evaluating this ability requires evolving environments that link interactions and resource dynamics to cumulative decision outcomes.

Sports management provides a concrete setting for studying this process, making it possible to track how the model coordinates current inputs with future returns through a series of decisions: trades and costly signings can strengthen a team in the current season while depleting future assets and limiting roster flexibility. These effects unfold as seasons progress, while rivals continue to improve their rosters, prompting further resource reallocation. We therefore formulate team management across seasons as a long-horizon evaluation task, tracing how interactive decisions shape competitive performance and resources over time to test whether models can turn local interactions into sustainable long-term management outcomes.

We introduce GM Bench, which models team management across seasons with two components: interactive subjects and the simulation environment. The interactive subjects comprise LLM general managers (GMs), players and agents; only LLMs serving as GMs are evaluated. GMs draft, sign, and trade players using visible information while competing for market resources. Players and agents share a general LLM with a fixed configuration to express demands and negotiate. The environment uses the National Basketball Association (NBA) as its primary setting and comprises three modules: League Rules, Player & Team Analysis, and Competition Simulation. A built-in engine simulates games and enforces rules, while financial constraints and market signals shape management decisions.

We extend a unified task linking GMs, players, institutional rules, and information channels to hockey, baseball, and American football to assess long-term management under different rules and resources. Real-world player ratings are used to construct five roster types: balanced, single-star, dual-star, offensive, and defensive. Evaluation uses uniform information conditions and comparable initial strength while preserving differences in star configurations and offensive or defensive emphasis. The proposed metrics report competitive performance across seasons, terminal cash, and seasonal trajectories, while action logs are analyzed to assess execution reliability and resource costs.

Experiments across four sports reveal substantial differences in the long-term management capabilities of LLMs. The six leading NBA models remain among the top seven in each additional sport. In the NBA evaluation, for example, Claude Opus 5.5 ranks first and wins three championships over five seasons; GPT-6 Astra and GPT-6.1 Sol combine sustained gains with larger cash reserves. Grok 4.7 achieves stronger competitive returns with less remaining cash, whereas Kimi K3 combines steady improvement with a larger financial buffer. MiMo-V2.6-Pro retains substantial cash despite declining performance. Sonnet faces unaffordable renewals after an NBA championship run. These cases highlight the need to coordinate acquisition, retention, and financing for sustained long-term performance. Our contributions are threefold:

![](images/a34afeb61a6554e273be6de8a46ed7defbb3bd859cb79c4671ab72b6a7d5ea6b.jpg)  
Figure 1 Illustrative performance trajectories of four team management strategies over five seasons show that early competitive gains can diminish, whereas rebuilding and sustained investment can yield delayed benefits.

• We introduce GM Bench across four sports, combining multiple role competition and interactions to evaluate long-horizon resource allocation by LLMs.

• We develop a protocol jointly reporting competitive performance, terminal resources, and seasonal trajectories, with five roster types of comparable strength from real-world player data for strategy comparisons.

• We evaluate cross-season management and cross-sport generalization through action logs, and trading diagnostics, analyzing feedback use and long-term decision effects.

## 2 Task Definition

## 2.1 Roles and Management Objectives

Each evaluated LLM serves as a team GM, managing players and assets through selection, trades, and contract renewals. Players share a general LLM for interactions, while the environment handles match simulation and player rotation. The objective is to improve competitive performance across seasons by balancing current roster investment with future resource needs under resource constraints. We model the task as a partially observable stochastic game:

$$
\mathcal { G } = ( \mathcal { T } , S , \mathcal { A } , P , O , \mathbf { U } , H ) ,\tag{1}
$$

where I is the team set, $s$ comprises roster, contract, financial, and market states, and A is the joint GM action space. P governs state transitions, O determines information visibility, U represents management outcomes, and H is the decision horizon across seasons.

## 2.2 State Dynamics and Outcomes

Each GM observes team state, player demands, and public market information through $o _ { t } ^ { i } .$ . Model $m _ { i }$ for team i conditions its actions on this observation and available history $h _ { t } ^ { i } .$ while the environment advances under joint action $\mathbf { a } _ { t } \mathbf { : }$

$$
a _ { t } ^ { i } \sim \pi _ { m _ { i } } ( \cdot \mid o _ { t } ^ { i } , h _ { t } ^ { i } ) , \quad s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , \mathbf { a } _ { t } ) .\tag{2}
$$

Changes caused by actions persist and enter subsequent observations according to disclosure rules. The task combines repeated decisions, state dependencies across phases, and delayed feedback. Outcome vector $\mathbf { U } _ { i }$ reports competitive performance, resource status, and long-term improvement, capturing cumulative returns and their resource costs.

![](images/a7dd4af0a732f0489a7539c79a3e97a634be8e802e15412d9a22c2e5fc550245.jpg)  
Figure 2 Overview of GM Bench, illustrated with the NBA environment and extended to hockey, baseball, and American football. LLM general managers interact with players and their agents within a shared simulator that enforces league and financial rules, while evaluation tracks competitive performance, resources and finances, and cross-season improvement.

## 3 NBA Environment

The NBA instance draws inspiration from ZenGM<sup>1</sup>, and models a NBA environment with league rules, LLM-based management, financial constraints, and role interactions.

## 3.1 League Foundations

Player Data and State. Initial rosters are constructed from real NBA player data based on player ratings. Growth and aging change ratings, while injuries affect availability and on-court performance. Game outcomes depend on player condition, roster composition, and random events, allowing the effects of roster adjustments to emerge throughout the season.

Core Rules and Financial Settings. The environment implements the following rules and financial mechanisms: (1) Game and season rules, which define game procedures and regular-season and playoff schedules; (2) Salary cap, which restricts payroll allocation and player acquisition; (3) Luxury tax, which imposes additional costs when payroll exceeds the tax threshold; (4) Financial condition, which updates cash balances using revenue, operating expenses, and luxury tax, with balances carried forward throughout the seasons.

## 3.2 Market Interactions and GM Actions

Player Demands and Market Signals. All players share an LLM that expresses trade demands based on team performance, playing roles, and contract status. Player representatives generate trade requests and public rumors. GMs combine these signals with roster needs to assess player value, conduct negotiation, and make offers.

Management Actions and Execution. GMs receive information on players, finances, and league rules and can perform four types of management actions: (1) Drafting, selecting rookie players; (2) Signing, signing free players; (3) Trading, negotiating and revising offers involving players and draft picks with other GMs; (4) Contract renewal, negotiating salaries with players after each season.

## 4 General Sports Management

Differences in drafting, transfers, and registration across sports are implemented through the corresponding rules and game, while role interactions, intertemporal resource dependencies, and delayed feedback form a unified evaluation structure, as shown in Figure 2. GM Bench extends the NBA task to hockey, baseball, and American football to assess long-term management under different rules and roster needs.

## 4.1 Evaluation Target: General Manager

Each evaluated LLM serves as a team GM, allocating resources based on rosters, contracts, future assets, and market signals to improve competitive performance across seasons. Sports management follows a common feedback loop that links these decisions over time: GM actions change team states, shaping competitive and market outcomes that inform subsequent decisions. Within this loop, roster changes affect current strength, future payroll flexibility, and tradable assets, requiring GMs to coordinate immediate personnel decisions with long-term resource commitments

## 4.2 Environment Roles: Players and Agents

Players and agents are both non-player characters (NPCs) rather than evaluation targets. They provide interaction partners and market signals by expressing individual demands and communicating trade opportunities, respectively.

Players: Expressing Personal Demands. All players share a general LLM with a fixed configuration. Based on individual circumstances, they express preferences over competitive prospects, playing roles, and compensation and participate in renewal negotiations. These preferences inform GM negotiation and resource allocation, while the simulator handles player development, injuries, and game performance.

Agents: Trade Requests and Public Rumors. Agents generate trade requests and public rumors for rostered players according to disclosure rules, turning individual circumstances into market signals. Private requests may remain undisclosed, while public rumors indicate potential trading opportunities. GMs weigh these signals against player condition, contract obligations, and roster needs when deciding whether to act.

## 4.3 System Environment

The system validates actions, advances games, and scores outcomes. Stochastic simulation and built-in management policies update league states and provide feedback for subsequent GM decisions.

League and Teams. The league checks contract renewals and financial compliance. Teams negotiate through LLMs, and agreements are validated before execution. Under the predefined season format, regular-season records determine standings and playoff eligibility, while playoffs decide the champion. At season end, the system settles revenue and expenses, opens drafting, signing, and renewal windows, and carries roster, contract, and financial states into the next season.

Coaches and Game Engine. Coaches and the game engine handle in-game decisions, translating roster configurations into outcomes shaped by roster, player, and randomness. GMs use this feedback to assess roster adjustments.

Media and Market Information. The system controls the timing and visibility of player demands, public rumors, and media reports, while journalists and social media disseminate information. Draft windows provide prospect names, positions, ratings, and values, helping GMs identify market opportunities and assess roster fit.

## 5 Data Sources and Evaluation

We construct teams with comparable initial strength but distinct management needs from real-world player data to compare model strategies. Evaluation tracks competitive performance, resource retention, and long-term improvement,

![](images/e960ec84f1bb9b76ebcb6ac785a2bfc1a28080ce6c4671e8706f3f9a8774c50b.jpg)  
Figure 3 Schematic profiles of the five team types. Balanced emphasizes an evenly distributed roster; Star centers on one standout player; Twins features two leading players; Offense prioritizes offensive ability; and Defense emphasizes defensive ability. The axes represent roster balance (BL), single-star contrast (SC), dual-core advantage (DC), offensive index (OI), and defensive index (DI). Larger radial values indicate stronger emphasis on the corresponding characteristic. The final panel overlays all five profiles for comparison.

capturing cumulative decision returns and future investment capacity.

## 5.1 Player Data and Team Initialization

For each of the four sports, we construct five roster types from real-world player data based on ratings, as shown in Figure 3, with comparable initial strength but distinct needs for bench depth, support for core players, and offensive or defensive improvement: (1) Balanced, with relatively even strength across starters and reserves; (2) Star, centered on one standout star with weaker supporting players; (3) Twins, led by two core players; (4) Offense, emphasizing shooting; (5) Defense, emphasizing defensive ability.

## 5.2 Evaluation Metrics

We report competitive performance, resource and financial status, and long-term improvement across four sports to characterize management returns and resource costs.

Competitive Performance. A common Elo protocol [5] measures relative competitive performance within each league. All N teams start each of $T$ seasons at $R _ { i , s } ^ { ( 0 ) } = R _ { 0 } = 1 5 0 0$ , with no rating carryover. Games are processed chronologically, with $R _ { i , s } ^ { ( g ) }$ denoting the rating of team i after rated league game g in season s. For teams i and j, the pre-game expected score is

$$
p _ { i , s , g } = \frac { 1 } { 1 + 1 0 ^ { ( R _ { j , s } ^ { ( g - 1 ) } - R _ { i , s } ^ { ( g - 1 ) } ) / 4 0 0 } } ,\tag{3}
$$

with $p _ { j , s , g } = 1 - p _ { i , s , g } .$ Both ratings are updated from the same pre-game values:

$$
R _ { k , s } ^ { \left( g \right) } = R _ { k , s } ^ { \left( g - 1 \right) } + K \big ( y _ { k , s , g } - p _ { k , s , g } \big ) ,\tag{4}
$$

for $k \in \{ i , j \}$ . The observed score $y _ { k , s , g }$ is 1 for a win, 0 for a loss, and 0.5 for a draw where permitted. We fix $K = 2 0$ for all teams and games; nonparticipants retain their ratings. Updates reward unexpected wins more strongly and preserve the league-wide total $\begin{array} { r } { \sum _ { i = 1 } ^ { \bar { N } } R _ { i , s } ^ { ( \bar { g } ) } = N R _ { 0 } } \end{array}$

With $M _ { s }$ rated games in season s, we report mean season-end Elo:

$$
E _ { i , s } = R _ { i , s } ^ { ( M _ { s } ) } , \qquad \overline { { E } } _ { i } = \frac { 1 } { T } \sum _ { s = 1 } ^ { T } E _ { i , s } .\tag{5}
$$

Higher $\overline { { E } } _ { i }$ indicates stronger performance. We also report cumulative regular-season wins, losses, draws where applicable, and seasonal rank.

Resources and Finances. Terminal cash $C _ { i }$ is the available balance after revenue and expenses are settled at the end of the management horizon, reflecting capacity for subsequent investment. Cash and competitive records jointly reveal the trade-off between resource retention and competitive returns. Active contracts and future assets can provide additional context on team resources.

Long-Term Improvement. Seasonal win-rate and cash track management performance. We define improvement as the difference between mean win rates in the final and initial w seasons:

$$
G _ { i } = \frac { 1 } { w } \sum _ { s = T - w + 1 } ^ { T } v _ { i , s } - \frac { 1 } { w } \sum _ { s = 1 } ^ { w } v _ { i , s } ,\tag{6}
$$

where $v _ { i , s } = W _ { i , s } / ( W _ { i , s } + L _ { i , s } )$ is the regular-season win rate excluding draws, and w is prespecified $( 2 w \leq T )$ Positive and negative $G _ { i }$ indicate improvement and decline, respectively. Together with $\overline { { E } } _ { i }$ , records, and terminal resources, $G _ { i }$ characterizes long-term management outcomes.

## 6 Experiments

## 6.1 Experimental Settings

Models and Configuration. The evaluation comprises 31 LLMs from 12 families: OpenAI (7), Anthropic (4), Gemini, DeepSeek, and Qwen (3 each), Grok, GLM, MiMo, and Doubao (2 each), and Kimi, MiniMax, and HY (1 each); the details are shown in Appendix B. The experiments span five seasons (2022-2026). We use official APIs with high reasoning effort where supported; more details are shown in Appendix C.

Baseline. We design Native GM as a rule-based management baseline: (1) Player valuation: players are valued using ability, age, potential, contracts, and injuries; (2) Strategy selection: team results and roster composition guide strategy selection, contention prioritizes current strength, while rebuilding favors young players, draft picks, and inexpensive contracts; (3) Contract management: signings prioritize high-value free agents under salary constraints; renewals consider contract costs and player willingness.

## 6.2 Overall Results and Model Comparisons

Overall Competitive Performance. In NBA setting, initially comparable teams show substantial divergence across seasons, as shown in Table 1. Elo ranges from 1071.0 to 1977.3, with Claude Opus 5.5 ranking first and winning three championships, followed by GPT-6 Astra and GPT-6.1 Sol. Opus, Astra, and Sonnet collectively win all five championships. Regular-season wins span 35 to 47 in the first season and 8 to 76 in the final season, this widening spread captures the cumulative effects of management over time.

Native Policy as a Strategic Reference. The native policy ranks 14th in the NBA setting, with regular-season wins rising from 39 to 49. Thirteen LLMs rank above it and 16 below, including Gemini 3.8 Flash, GPT-5.6 Terra, and DeepSeek-V4.1-Flash. These results highlight room for improvement in the long-term management capabilities of many current LLMs.

Strategic Capacity Across Model Generations. Long-term management performance varies substantially across model generations, as shown in Table 1. Within families, GPT-6 Astra and GPT-6.1 Sol outperform GPT-4o, DeepSeek-V4-Pro exceeds DeepSeek-R1. Differences also emerge across tiers within the same generation, with Qwen3.8-Max exceeding Qwen3.8-Flash. However, older models can outperform newer releases both within and across families: GPT-4o exceeds Hy4 Preview and MiMo-V2.6-Flash, while Gemini 3.1 Pro Preview outperforms Gemini 3.8 Flash.

## 6.3 Resource Allocation Across Seasons

Star Acquisition and Competitive Returns. Trading observations and terminal cash balances reveal contrasting investment styles, as shown in Table 1: Claude 5 GMs actively trade for championship contention, whereas GPT-6 GMs favor cautious acquisitions. Opus acquires stars and wins three championships, while Astra and Sol preserve larger cash buffers for future roster adjustments. Grok 4.7 achieves stronger competitive returns with less remaining cash, whereas Kimi K3 combines sustained improvement with greater financial flexibility. These comparisons highlight competitive returns and future investment capacity as joint criteria for assessing star acquisitions.

<table><tr><td>Rank Model</td><td>Elo</td><td>Wins</td><td>Losses</td><td>Cash</td></tr><tr><td>1 Claude Opus 5.5 0</td><td>1977.3</td><td>446</td><td>66</td><td>312.48</td></tr><tr><td>2 GPT-6 Astra 8</td><td>1907.7</td><td>427</td><td>89</td><td>1006.62</td></tr><tr><td>3 GPT-6.1 Sol</td><td>1838.7</td><td>400</td><td>101</td><td>1152.33</td></tr><tr><td>4 Claude Sonnet 5.5 8</td><td>1813.3</td><td>385</td><td>98</td><td>308.05</td></tr><tr><td>5 Claude Fable 5.1</td><td>1794.4</td><td>377</td><td>106</td><td>419.07</td></tr><tr><td>6 Grok 4.7</td><td>1784.3</td><td>347</td><td>121</td><td>585.74</td></tr><tr><td>7 Gemini 3.1 Pro Preview</td><td>1753.1</td><td>334</td><td>131</td><td>-21.63</td></tr><tr><td>8 Grok 4.6</td><td>1713.6</td><td>316</td><td>150</td><td>443.52</td></tr><tr><td>9 Kimi K3</td><td>1698.8</td><td>308</td><td>130</td><td>868.23</td></tr><tr><td>10 DeepSeek-V4-Pro</td><td>1666.3</td><td>275</td><td>161</td><td>255.43</td></tr><tr><td>11 GLM-5.3</td><td>1652.6</td><td>279</td><td>161</td><td>188.42</td></tr><tr><td>12 Qwen3.8-Max</td><td>1602.5</td><td>250</td><td>189</td><td>292.10</td></tr><tr><td>13 GPT-6 Luna</td><td>1582.6</td><td>249</td><td>190</td><td>-41.43</td></tr><tr><td>14 Native GM strategy</td><td>1575.0</td><td>240</td><td>196</td><td>96.78</td></tr><tr><td>15 Gemini 3.8 Flash</td><td>1525.7</td><td>220</td><td>217</td><td>111.40</td></tr><tr><td>16 GPT-5.6 Terra</td><td>1499.0</td><td>196</td><td>232</td><td>527.42</td></tr><tr><td>17 MiniMax M3</td><td>1451.8</td><td>193</td><td>226</td><td>106.26</td></tr><tr><td>18 DeepSeek-V4.1-Flash</td><td>1418.4</td><td>178</td><td>232</td><td>294.00</td></tr><tr><td>19 Qwen3.8-Flash</td><td>1388.5</td><td>154</td><td>256</td><td>428.19</td></tr><tr><td>20 Qwen3.8-27B</td><td>1361.6</td><td>154</td><td>256</td><td>460.73</td></tr><tr><td>21 Doubao-Seed-2.1-Pro</td><td>1331.0</td><td>133</td><td>277</td><td>335.05</td></tr><tr><td>22 GLM-5.3-Flash</td><td>1305.8</td><td>117</td><td>293</td><td>581.30</td></tr><tr><td>23 MiMo-V2.6-Pro</td><td>1275.4</td><td>105</td><td>305</td><td>925.41</td></tr><tr><td>24 Gemini 3.5 Flash-Lite</td><td>1255.2</td><td>106</td><td>304</td><td>65.99</td></tr><tr><td>25 GPT-40</td><td>1206.4</td><td>91</td><td>319</td><td>360.08</td></tr><tr><td>26 DeepSeek-R1</td><td>1167.2</td><td>81</td><td>329</td><td>167.76</td></tr><tr><td>27 Claude Haiku 4.5</td><td>1150.6</td><td>61</td><td>349</td><td>244.02</td></tr><tr><td>28 Hy4 preview</td><td>1124.7</td><td>53</td><td>357</td><td>81.31</td></tr><tr><td>29 MiMo-V2.6-Flash</td><td>1107.2</td><td>51</td><td>359</td><td>-40.19</td></tr><tr><td>30 Doubao-Seed-2.1-Turbo</td><td>1071.0</td><td>42</td><td>368</td><td>354.07</td></tr></table>

<table><tr><td>Rank</td><td>Model</td></tr><tr><td>1 Claude Opus 5.5</td><td>22 23 24 25 26  $G _ { i }$  44 54 65 74 76+31.71</td></tr><tr><td>2 GPT-6 Astra</td><td>45 50 61 69 73 +28.66</td></tr><tr><td>3 GPT-6.1 Sol</td><td>45 49 60 67 71 +26.83</td></tr><tr><td>4 Claude Sonnet 5.5</td><td>43 54 75 6556 +14.63</td></tr><tr><td>5 Claude Fable 5.1</td><td>38 49 64 67 66 +28.05</td></tr><tr><td>6 Grok 4.7</td><td>36 46 50 59 63 +24.39</td></tr><tr><td>7 Gemini 3.1 Pro Preview</td><td>42 51 64 57 46 +6.10</td></tr><tr><td>8 Grok 4.6</td><td>38 46 49 55 57 +17.07</td></tr><tr><td>9 Kimi K3</td><td>42 46 55 58 62 +19.51</td></tr><tr><td>10 DeepSeek-V4-Pro</td><td>3644 46 53 56 +17.68</td></tr><tr><td>11 GLM-5.3</td><td>4752 53 54 57 +7.32</td></tr><tr><td>12 Qwen3.8-Max</td><td>41 42 48 48 52 +10.37</td></tr><tr><td>13 GPT-6 Luna</td><td>37 48 54 46 38 -0.61</td></tr><tr><td>14 Native GM strategy</td><td>39 44 46 48 49 +8.54</td></tr><tr><td>15 Gemini 3.8 Flash</td><td>40 40 41 42 43 +3.05</td></tr><tr><td>16 GPT-5.6 Terra</td><td>44 38 38 38 38 -3.66</td></tr><tr><td>17 MiniMax M3</td><td>39 38 38 38 41 +1.22</td></tr><tr><td>18 DeepSeek-V4.1-Flash</td><td>4738 37 36 38 -6.71</td></tr><tr><td>19 Qwen3.8-Flash</td><td>37 35 34 32 32 -4.88</td></tr><tr><td>20 Qwen3.8-27B</td><td>35 33 32 32 32 -2.44</td></tr><tr><td>21 Doubao-Seed-2.1-Pro</td><td>35 32 27 27 27 -7.93</td></tr><tr><td>22 GLM-5.3-Flash</td><td>4036 25 24 27 -15.24</td></tr><tr><td>23 MiMo-V2.6-Pro</td><td>40 37 23 22 22 -20.12</td></tr><tr><td>24 Gemini 3.5 Flash-Lite</td><td>42 34 22 24 23 -17.68</td></tr><tr><td>25 GPT-40</td><td>46 35 29 23 19 -23.78</td></tr><tr><td>26 DeepSeek-R1</td><td>39 32 18 19 17 -21.34</td></tr><tr><td>27 Claude Haiku 4.5</td><td>41 32 18 14 15 -26.83</td></tr><tr><td>28 HY4 Preview</td><td>46 32 18 13 12 -32.32</td></tr><tr><td>29 MiMo-V2.6-Flash</td><td>43 31 22 14 8 -31.71</td></tr><tr><td>30 Doubao-Seed-2.1-Turbo</td><td>43 32 18 12 14 -29.88</td></tr></table>

Table 1 NBA results on GM Bench. Each denotes one championship. Best values are bold and highlighted; second-best values are underlined. Negative cash is highlighted in red.  
Table 2 NBA regular-season wins and long-term improvement. Rank follows Table 1. G<sub>i</sub> is the change in average win rate from 2022–2023 to 2025–2026 (w = 2), expressed in percentage points.

Championship Peaks and Roster Continuity. Peak success and sustained improvement can diverge, as shown in Table 2. Sonnet wins a championship and peaks at 75 regular-season wins in 2024 before falling to 65 and 56. Sol improves steadily from 45 to 71, achieving a larger $G _ { i }$ . Fable records 67 and 66 wins in the final two seasons and achieving $G _ { i } = 2 8 . 0 5$ percentage points without a championship. Renewal observations show that Sonnet faces unaffordable salary demands and loses valuable players, highlighting the need to coordinate acquisition and retention.

Financial Feedback and Investment Capacity. Cash balances complement competitive results as a measure of investment capacity. Astra and Sol retain 1006.62 and 1152.33, respectively, compared with 312.48 for Opus. MiMo-V2.6-Pro retains 925.41 in cash but ranks 23rd, with regular-season wins falling from 40 to 22, showing high balances may coexist with weak competitive returns. Cash deficits occur at different competitive levels: Gemini 3.1 Pro Preview (rank 7, −21.63), GPT-6 Luna (rank 13, −41.43), and MiMo-V2.6-Flash (rank 29, −40.19). These management cases illustrate linked financial pressures: salary commitments tie up future revenue, while weak results reduce revenue and constrain reinvestment.

(a) American football
<table><tr><td>Rank</td><td>Model Elo Cash</td></tr><tr><td>1 Claude Opus 5.5</td><td>1870.1 228.19</td></tr><tr><td>2 Claude Sonnet 5.5</td><td>1761.6 78.00</td></tr><tr><td>3 GPT-6.1 Sol</td><td>1760.0205.34</td></tr><tr><td>4 Kimi K3</td><td>1750.0 86.48</td></tr><tr><td>5 GPT-6 Astra</td><td>1745.0163.94</td></tr><tr><td>6 Claude Fable 5.1</td><td>1736.3236.92</td></tr><tr><td>7 Grok 4.7</td><td>1733.0422.70</td></tr><tr><td>8 Gemini 3.1 Pro Preview</td><td>1714.4 210.74</td></tr><tr><td>9 DeepSeek-V4-Pro</td><td>1714.3117.67</td></tr><tr><td>10 GLM-5.3</td><td>1700.0233.54</td></tr><tr><td>11 Grok 4.6</td><td>1663.8 215.06</td></tr><tr><td>12 GPT-6 Luna</td><td>1644.4178.54</td></tr><tr><td>13 Gemini 3.8 Flash</td><td>1630.4 191.05</td></tr><tr><td>14 Qwen3.8-Max</td><td>1559.7 271.95</td></tr><tr><td>15 GPT-5.6 Terra</td><td>1547.8102.12</td></tr><tr><td>16 DeepSeek-V4.1-Flash</td><td>1518.7 93.57</td></tr><tr><td>17 MiniMax M3</td><td>1468.9 -67.53</td></tr><tr><td>18 Qwen3.8-Flash</td><td>1401.8161.60</td></tr><tr><td>19 Native GM strategy</td><td>1386.1 262.38</td></tr><tr><td>20 MiMo-V2.6-Pro</td><td>1385.0 74.50</td></tr><tr><td>21 Qwen3.8-27B</td><td>1365.1 391.39</td></tr><tr><td>22 GPT-5.4 Mini</td><td>1352.7196.24</td></tr><tr><td>23 Gemini 3.5 Flash-Lite</td><td>1320.5123.38</td></tr><tr><td>24 Doubao-Seed-2.1-Pro</td><td>1315.4 37.75</td></tr><tr><td>25 GLM-5.3-Flash</td><td>1310.2 248.22</td></tr><tr><td>26 DeepSeek-R1</td><td>1258.2173.47</td></tr><tr><td>27 MiMo-V2.6-Flash</td><td>1245.0 42.79</td></tr><tr><td>28 GPT-5.4 Nano</td><td>1220.8211.02</td></tr><tr><td>29 Hy4 preview</td><td>1210.6126.84</td></tr><tr><td>30 GPT-4o</td><td>1193.3 210.78</td></tr><tr><td>31 Doubao-Seed-2.1-Turbo</td><td>1163.5 169.85</td></tr><tr><td>32 Claude Haiku 4.5</td><td>1105.3221.24</td></tr></table>

(b) Ice hockey
<table><tr><td>Rank</td><td>Model</td><td>Elo Cash</td></tr><tr><td>1 Claude Opus 5.5</td><td></td><td>1855.4 63.10</td></tr><tr><td>2 GPT-6.1 Sol</td><td></td><td>1824.0 59.27</td></tr><tr><td>3 Claude Sonnet 5.5</td><td></td><td>1768.8 40.08</td></tr><tr><td>4 Gemini 3.1 Pro Preview</td><td></td><td>1754.7 138.45</td></tr><tr><td>5 GPT-6 Astra</td><td></td><td>1750.0 0.18</td></tr><tr><td>6 Claude Fable 5.1</td><td></td><td>1747.8116.69</td></tr><tr><td>7 Grok 4.7</td><td></td><td>1739.5 44.86</td></tr><tr><td>8Kimi K3</td><td></td><td>1665.4267.86</td></tr><tr><td>9 Grok 4.6</td><td></td><td>1617.6320.32</td></tr><tr><td>10 DeepSeek-V4-Pro</td><td></td><td>1591.9176.44</td></tr><tr><td>11 Native GM strategy</td><td></td><td>1572.7 -18.21</td></tr><tr><td>12 Qwen3.8-Max</td><td></td><td>1569.3 49.08</td></tr><tr><td>13 Qwen3.8-Flash</td><td></td><td>1557.6116.02</td></tr><tr><td>14 Gemini 3.8 Flash</td><td></td><td>1545.9 6.45</td></tr><tr><td>15 GPT-5.4 Mini</td><td></td><td>1544.4 10.93</td></tr><tr><td>16 GPT-5.6 Terra</td><td></td><td>1429.3 -43.07</td></tr><tr><td>17 DeepSeek-V4.1-Flash</td><td></td><td>1415.2 6.97</td></tr><tr><td>18 GLM-5.3</td><td></td><td>1391.0 75.21</td></tr><tr><td>19 Qwen3.8-27B</td><td></td><td>1390.2 217.59</td></tr><tr><td>20 MiniMax M3</td><td>1380.2</td><td>-58.10</td></tr><tr><td>21 GPT-5.4 Nano</td><td>1347.0</td><td>69.68</td></tr><tr><td>22 GPT-6 Luna</td><td></td><td>1333.0 51.33</td></tr><tr><td>23 GLM-5.3-Flash</td><td></td><td>1317.6 204.81</td></tr><tr><td>24 Doubao-Seed-2.1-Pro</td><td>1290.5</td><td>-39.41</td></tr><tr><td>25 GPT-40</td><td>1251.1</td><td>17.16</td></tr><tr><td>26 MiMo-V2.6-Pro</td><td>1237.3</td><td>-10.39</td></tr><tr><td>27 Claude Haiku 4.5</td><td></td><td>1235.6321.47</td></tr><tr><td>28 DeepSeek-R1</td><td>1191.0</td><td>80.67</td></tr><tr><td>29 Doubao-Seed-2.1-Turbo</td><td></td><td>1186.8119.83</td></tr><tr><td>30 Hy4 preview</td><td>1166.3</td><td>-11.50</td></tr><tr><td>31 MiMo-V2.6-Flash</td><td>1157.1</td><td>-5.79</td></tr><tr><td>32 Gemini 3.5 Flash-Lite</td><td>1118.8</td><td>-36.56</td></tr></table>

(c) Baseball
<table><tr><td>Rank</td><td>Model</td><td>Elo</td><td>Cash</td></tr><tr><td>1 GPT-6 Astra</td><td></td><td>1895.5</td><td>203.05</td></tr><tr><td>2 Claude Opus 5.5</td><td></td><td>1860.4</td><td>-33.64</td></tr><tr><td>3 GPT-6.1 Sol</td><td></td><td>1780.0</td><td>149.24</td></tr><tr><td>4 Qwen3.8-Max</td><td></td><td>1776.2</td><td>386.26</td></tr><tr><td></td><td>5 Claude Sonnet 5.5</td><td>1769.4</td><td>527.73</td></tr><tr><td>6 Grok 4.7</td><td></td><td>1747.2</td><td>447.90</td></tr><tr><td>7 Claude Fable 5.1</td><td></td><td>1737.1</td><td>32.27</td></tr><tr><td></td><td>8Native GM strategy</td><td>1727.0</td><td>406.52</td></tr><tr><td></td><td>9 Gemini 3.1 Pro Preview</td><td>1719.6 373.27</td><td></td></tr><tr><td>10 GPT-6 Luna</td><td></td><td>1707.3</td><td>323.11</td></tr><tr><td>11 Grok 4.6</td><td></td><td>1695.1</td><td>287.21</td></tr><tr><td>12 GLM-5.3</td><td></td><td>1666.9</td><td>264.28</td></tr><tr><td></td><td>13 DeepSeek-V4-Pro</td><td>1623.5</td><td>242.39</td></tr><tr><td>14 Kimi K3</td><td></td><td></td><td>1616.6 357.41</td></tr><tr><td></td><td>15 MiniMax M3</td><td>1540.1</td><td>-41.12</td></tr><tr><td></td><td>16 Doubao-Seed-2.1-Pro</td><td>1505.6</td><td>-686.43</td></tr><tr><td></td><td>17 GLM-5.3-Flash</td><td>1476.3</td><td>42.31</td></tr><tr><td>18 Qwen3.8-27B</td><td></td><td>1441.3</td><td>151.91</td></tr><tr><td>19 GPT-5.6 Terra</td><td></td><td>1441.1</td><td>43.03</td></tr><tr><td>20 MiMo-V2.6-Pro</td><td></td><td>1380.5</td><td>-609.77</td></tr><tr><td>21 Gemini 3.8 Flash</td><td></td><td>1378.0</td><td>-14.77</td></tr><tr><td>22 DeepSeek-V4.1-Flash</td><td></td><td>1350.9</td><td>21.55</td></tr><tr><td>23 Gemini 3.5 Flash-Lite</td><td></td><td>1346.8</td><td>-39.27</td></tr><tr><td>24 Qwen3.8-Flash</td><td></td><td>1302.4</td><td>106.73</td></tr><tr><td>25 Claude Haiku 4.5</td><td></td><td>1263.8</td><td>172.99</td></tr><tr><td>26 GPT-40</td><td></td><td>1256.5</td><td>60.17</td></tr><tr><td>27 DeepSeek-R1</td><td></td><td>1237.9</td><td>165.93</td></tr><tr><td>28 Hy4 preview</td><td></td><td>1216.5</td><td>206.73</td></tr><tr><td>29 Doubao-Seed-2.1-Turbo</td><td></td><td>1146.9</td><td>-22.45</td></tr><tr><td>30 MiMo-V2.6-Flash</td><td></td><td>1084.9</td><td>53.63</td></tr></table>

Table 3 Elo rankings and terminal cash in (a) American football, (b) ice hockey, and (c) baseball. The highest value in each column and sport is bold and highlighted; the second-highest is underlined. Negative cash is highlighted in red. The native GM strategy is boxed.

## 6.4 Results Across Additional Sports

American Football. Claude Opus 5.5 leads American football, matching its first-place NBA result, while Kimi K3 ranks fourth compared with ninth in the NBA, as shown in Table 2. The six highest-ranked NBA models remain in the top seven in all three additional sports, showing consistent competitive strength across settings. The native GM strategy ranks 19th, below its NBA placement of 14th. Grok 4.7 combines seventh place with the highest terminal cash, showing that competitive leadership and cash retention remain distinct across settings.

Ice Hockey. Opus again leads ice hockey, followed by GPT-6.1 Sol, preserving their strong NBA standings. Financial outcomes transfer less uniformly: Astra, one of the two NBA cash leaders, ranks fifth but retains only 0.18, whereas Haiku holds the largest balance despite ranking 27th. The native GM strategy ranks 11th, above its NBA and American football placements, but finishes with negative cash. Across evaluated teams, terminal deficits occur in 8 of 32 hockey teams and 7 of 30 baseball teams, compared with 1 of 32 in American football and 3 of 30 in the NBA. These results show that stronger competitive performance can coexist with financial strain.

Baseball. GPT-6 Astra leads baseball, compared with second in the NBA and fifth in both American football and ice hockey. Qwen3.8-Max and Kimi K3 reverse their relative positions: Qwen ranks 14th in American football and fourth in baseball, while Kimi ranks fourth and 14th, respectively. Opus, which leads the other three sports, ranks second but ends with negative cash. Its aggressive investment leads to bankruptcy late in the evaluation, followed by player departures and a series of losses. Sonnet combines fifth place with the highest cash balance, further illustrating how financial rankings shift across sports. The native GM strategy ranks eighth. Together, these experiments assess how LLMs adapt resource allocation across sports to sustain competitive performance and future investment capacity.

## 7 Related Work

Interactive Evaluation. Interactive benchmarks evaluate sequential actions and coordination in executable environments [9, 20]. WebArena, AppWorld, and TheAgentCompany cover web, application, and workplace workflows [14, 16, 19], while BrowserGym unifies web-agent evaluation [4]. τ<sup>2</sup>-bench tests agent–user collaboration in a shared environment [2], and BALROG evaluates planning and exploration across games [11].

Long-Term Management and Competition. Recent benchmarks examine the long-term effects of management decisions. Vending-Bench, EcoGym, and EnterpriseArena evaluate sustained business operation and resource allocation [1, 7, 8]. Market-Bench, Vending-Bench Arena, and CoffeeBench study procurement, pricing, and trading in multi-agent markets [12, 18]. E-Commerce Bench grounds year-long operations in real product and supplier data, linking repeated supplier negotiations, cash-flow management, and long-term returns [6].

Sports Management. Sports management links player development, contract constraints, and performance across seasons. FM-Bench combines long-term management, model competition, and equal-budget drafting in a synthetic football world [15]. FromPitch2Board uses paired seeds and controlled configurations to examine models, agent scaffolds, management roles, and evaluation horizons [17]. GM Bench evaluates cross-season resource allocation under competition across four sports, using real player data, roster types of comparable strength, and a shared LLM for player demands and negotiation.

## 8 Conclusion

We introduced GM Bench, an interactive benchmark for long-horizon sports management. A shared task across four sports connects competition among LLM general managers, player interactions, and persistent resource constraints to outcomes across seasons. Experiments reveal distinct balances between championship pursuit, financial flexibility, and roster continuity. Renewal trajectories link competitive success to subsequent retention demands, highlighting anticipation of future commitments as a key element of sustained improvement. By jointly tracking competitive performance, cash reserves, and seasonal trajectories, GM Bench makes cumulative decision effects observable and evaluates the ability to sustain progress under competition.

## Limitations

League Configuration. Each sport is instantiated through a specified simulator and league configuration. The unified task structure supports evaluation across four sports, while contract systems, roster constraints, financial rules, and competition formats remain specific to each implementation. Extending the benchmark to other leagues requires corresponding rule and interface adaptations.

Observation Horizon. Decisions are assessed over a finite number of seasons, capturing cumulative competitive and financial effects within the evaluation window. Full player careers and distant future assets may extend beyond this horizon. Longer evaluations would help characterize delayed player development, aging, and the outcomes of commitments that mature after the evaluation ends.

Evaluation Cost. Multi-season evaluation with interacting LLM roles requires repeated inference and simulation. Expanding coverage to additional models, random seeds, and league configurations increases computational cost and evaluation time. More efficient execution and parallelization of independent runs could facilitate broader experimental coverage.

## Ethical Considerations

GM Bench uses real-player data to initialize simulated states. Player demands, negotiations, and market rumors are generated as fictional scenario content and do not represent the actual views, intentions, or conduct of the individuals portrayed. We interpret these interactions within their simulated roles and restrict evaluation conclusions to model behavior. All management actions are executed within the simulator, and benchmark outputs are not intended to guide real-world recruitment, contract negotiations, or financial decisions.

We comply with applicable source licenses and attribution requirements when reusing data and software. We document the provenance and intended use of datasets, prompts, and trajectories, and exclude private personal information and confidential content from shared materials. Our paper distinguishes source player attributes from generated interactions, avoiding the presentation of synthetic statements as factual reports about real individuals.

## References

[1] Axel Backlund and Lukas Petersson. 2025. Vending-bench: A benchmark for long-term coherence of autonomous agents. arXiv preprint arXiv:2502.15840.

[2] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. 2025. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. Preprint, arXiv:2506.07982.

[3] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, et al. 2025. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. In International Conference on Learning Representations, volume 2025, pages 96934–96990.

[4] De Chezelles, Thibault Le Sellier, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lù, Ori Yoran, Dehan Kong, Frank F Xu, Siva Reddy, Quentin Cappart, et al. 2024. The browsergym ecosystem for web agent research. arXiv preprint arXiv:2412.05467.

[5] Arpad E Elo. 1978. The rating of chessplayers, past and present. (No Title).

[6] Wei Fan, Xinjie Shen, Xudong Guo, Jianhong Tu, Yang Su, Yinger Zhang, Lianghao Deng, Fengyu Wang, Baohua Dong, Yangqiu Song, et al. 2026. E-commerce bench: Evaluating llm agents on long-horizon autonomous business operation. arXiv preprint arXiv:2608.30730.

[7] Yi Han, Yan Wang, Lingfei Qian, Haohang Li, Yupeng Cao, Yueru He, Xueqing Peng, Nanhan Shen, Yitao Xu, Yankai Chen, et al. 2026. Can llm agents be cfos? benchmarking long-horizon resource allocation in an uncertain enterprise environment. arXiv preprint arXiv:2603.23638.

[8] Xavier Hu, Jinxiang Xia, Shengze Xu, Kangqi Song, Yishuo Yuan, Guibin Zhang, JinCheng Ren, Boyu Feng, Li Lu, Tieyong Zeng, et al. 2026. Ecogym: Evaluating llms for long-horizon plan-and-execute in interactive economies. arXiv preprint arXiv:2602.09514.

[9] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. Agentboard: An analytical evaluation board of multi-turn llm agents. Advances in neural information processing systems, 37:74325–74362.

[10] Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. 2025. Discoverybench: Towards data-driven discovery with large language models. In International Conference on Learning Representations, volume 2025, pages 4556–4579.

[11] Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wołczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kucinski, Lerrel Pinto, Rob Fergus, et al. 2025. Balrog: Benchmarking agentic llm and vlm reasoning on games. In´ International Conference on Learning Representations, volume 2025, pages 96666–96702.

[12] Issa Sugiura, Daichi Hattori, Kazuo Araragi, Keita Ogawa, Shota Onose, Taro Makino, Teppei Usuki, and Takashi Ishida. 2026. Coffeebench: Benchmarking long-horizon llm agents in heterogeneous multi-agent economies. arXiv preprint arXiv:2606.16613.

[13] Megh Thakkar, Nicolas Chapados, Christopher Pal, et al. 2025. Webarena verified: Reliable evaluation for web agents. In Workshop on Scaling Environmentsfor Agents.

[14] Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. 2024. Appworld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16022–16076.

[15] Tianyou Wang, Chongyang Gao, Kezhen Chen, Dong Chen, Yinghao He, Donghan Li, Wangcheng Xu, Hongjiu Zhang, and Chi Li. 2026. Fm-bench: A benchmark for long-horizon management with competing agents. arXiv preprint arXiv:2608.18423.

[16] Frank Fangzheng Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, et al. 2026. Theagentcompany: benchmarking llm agents on consequential real world tasks. Advances in Neural Information Processing Systems, 38.

[17] Peiyu Zang. 2026. Frompitch2board: Benchmarking llm agents in long-horizon football management. arXiv preprint arXiv:2609.34710.

[18] Yushuo Zheng, Huiyu Duan, Zicheng Zhang, Yucheng Zhu, Xiongkuo Min, and Guangtao Zhai. 2026. Market-bench: Benchmarking large language models on economic and trade competition. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 39893–39906.

[19] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. 2024. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606.

[20] Kunlun Zhu, Hongyi Du, Zhaochen Hong, Xiaocheng Yang, Shuyi Guo, Daisy Zhe Wang, Zhenhailong Wang, Cheng Qian, Xiangru Tang, Heng Ji, et al. 2025. Multiagentbench: Evaluating the collaboration and competition of llm agents. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8580–8622.

## A Benchmark Comparison

Table 4 compares representative benchmarks along four design dimensions: Demand negotiation incorporates demands from LLM-driven personnel into resource allocation; Needs adaptation tests strategy adjustment across teams with comparable initial strength but different configuration needs; Intertemporal diagnostics traces long-term action benefits and subsequent resource constraints through controlled comparisons or state replay; Institutional adaptation evaluates strategies across sports rules and resources within a unified management task. We also report the number of interacting agent roles in each environment.
<table><tr><td>Benchmark</td><td>Interacting agents per environment</td><td>Demand negotiation</td><td>Needs adaptation</td><td>Long-term diagnostics</td><td>Institutional adaptation</td></tr><tr><td>EcoGym [8]</td><td>1 management agent</td><td>X</td><td>X</td><td>△</td><td>X</td></tr><tr><td>EnterpriseArena [7]</td><td>1 CFO agent</td><td>X</td><td>X</td><td>△</td><td>X</td></tr><tr><td>FM-Bench Arena [15]</td><td>16 managers (15 LLM + 1 scripted)</td><td>X</td><td>X</td><td>V</td><td>X</td></tr><tr><td>Market-Bench [18]</td><td>20 LLM retailers</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>CoffeeBench [12]</td><td>6 LLM firms (1 evaluated + 5 reference)</td><td>X</td><td>X</td><td>△</td><td>X</td></tr><tr><td>E-Commerce Bench [6]</td><td>1 manager + up to 576 supplier NPCs</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>GM Bench</td><td>30 GMs + NNPC NPCs (NBA)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 4 Interaction scales and evaluation designs of representative benchmarks.  
Notes. ✓: explicit support; △: related comparisons of memory, frameworks, or incentives; ×: no documented support under the stated definitions. Counts include management agents and negotiation NPCs, excluding evaluators and simulated consumer demand. E-Commerce Bench provides a pool of 576 supplier NPCs with rule-based economic decisions and LLM-rendered dialogue; each episode may contact only a subset. N<sub>NPC</sub> counts player and agent NPCs in the NBA instance of GM Bench.

## B Model List

The model list comprises 31 distinct models, with 29 shared across all four sports. Basketball and baseball each use these 29 models plus one scripted baseline, yielding 30 participants to match the number of teams in the National Basketball Association (NBA) and Major League Baseball (MLB), respectively. American football and ice hockey additionally include GPT-5.4 Mini and GPT-5.4 Nano, yielding 31 models plus the scripted baseline, or 32 participants, to match the number of teams in the National Football League (NFL) and National Hockey League (NHL), respectively. The details are shown in Table 5.

<table><tr><td>Group</td><td>Model identifiers</td></tr><tr><td>OpenAI</td><td>GPT-6 Astra; GPT-6.1 Sol; GPT-6 Luna; GPT-5.6 Terra; GPT-4o; GPT-5.4 Mini; GPT-5.4 Nano.</td></tr><tr><td>Anthropic</td><td>Claude Opus 5.5; Claude Sonnet 5.5; Claude Fable 5.1; Claude Haiku 4.5.</td></tr><tr><td>Gemini</td><td>Gemini 3.1 Pro Preview; Gemini 3.8 Flash; Gemini 3.5 Flash-Lite.</td></tr><tr><td>DeepSeek</td><td>DeepSeek-V4-Pro; DeepSeek-V4.1-Flash; DeepSeek-R1.</td></tr><tr><td>Qwen</td><td>Qwen3.8-Max; Qwen3.8-Flash; Qwen3.8-27B.</td></tr><tr><td>Grok</td><td>Grok 4.7; Grok 4.6.</td></tr><tr><td>GLM</td><td>GLM-5.3; GLM-5.3-Flash.</td></tr><tr><td>MiMo</td><td>MiMo-V2.6-Pro; MiMo-V2.6-Flash.</td></tr><tr><td>Doubao</td><td>Doubao-Seed-2.1-Pro; Doubao-Seed-2.1-Turbo.</td></tr><tr><td>Kimi</td><td>Kimi K3.</td></tr><tr><td>MiniMax</td><td>MiniMax M3.</td></tr><tr><td>HY4</td><td>Hy4 preview.</td></tr><tr><td>Baseline</td><td>Native GM strategy.</td></tr></table>

Table 5 Models used across the four sports.

## C Reproducibility Details

## C.1 Evaluation Scope and Language

The benchmark covers basketball, American football, ice hockey, and baseball. Section 5.1 describes five roster types for each sport: Balanced, Star, Twins, Offense, and Defense. All model interactions use English text. Table 1 reports 30 NBA entries, comprising 29 LLM deployments and one scripted baseline. Table 3 reports 32 entries for American football, 32 for ice hockey, and 30 for baseball, each including the scripted baseline. These counts describe the reported model-policy entries. Random-seed repetitions are reported separately below.

## C.2 Inference and Experimental Selection

All evaluated models are accessed through their official APIs. Configurable reasoning effort is set to high for every model that supports this setting. The total API expenditure for the experiments is approximately USD 15,000, as reported in Section 6.1. The local Python interpreter is version 3.9.6.

## C.3 Runs and Result Aggregation

Each reported full-horizon model evaluation consists of one complete five-season run covering 2022–2026. Token expenditure constrained the number of full-horizon repetitions. Multiple short-horizon tests with different random seeds were used for cross-checking. The reported full-horizon results remain single-run measurements; short-horizon tests are not counted as independent five-season replications.

Section 5.2 defines the aggregation used in the result tables. Elo is the arithmetic mean of season-end ratings over the five seasons, with ratings initialized to 1500 at the start of each season and updated using K = 20 and a 400-point logistic scale. Terminal cash is measured at the end of the evaluation horizon. Table 2 reports regular-season wins by year and the difference in mean win rate between 2022-2023 and 2025-2026 (w = 2), expressed in percentage points. The five seasons form a temporally dependent trajectory. The tables do not report confidence intervals across independent full-horizon runs.