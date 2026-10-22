HomePage: https://lab.einsia.ai/quant-arena

![](images/731574cc41ffff1eebc2df59de27d4b615d6da0fffc4fe36cda73ebd6b7cd38c.jpg)  
Figure 1 Quant-Arena evaluates LLMs on complete quantitative research runs. Banners show total scores on the iterative research track.

## 1 Introduction

Quantitative research turns hypotheses about market behavior into executable trading strategies. Researchers construct features from market observations, combine them into predictive factors or signals, and use those signals to build and test portfolios. Large language models (LLMs) are a natural fit for this research loop: they can propose hypotheses, implement them in code, interpret feedback, and revise their approach. Recent systems already explore this potential through interactive alpha discovery and iterative factor search (Wang et al., 2023; Tang et al., 2025; Han et al., 2026).

Existing benchmarks measure diferent parts of this workflow. AlphaQT-Bench evaluates the correctness of quantitative trading code, while AlphaBench studies the generation, evaluation, and search of formulaic factors (AlphaQT-Bench; Luo et al., 2026). StockBench evaluates agents that make sequential trading decisions from market information (StockBench). AlphaForgeBench extends evaluation to generating factor and strategy code from natural-language objectives and backtesting it on historical daily market data (Zhang et al., 2026). To our knowledge, this is the first benchmark on real market data in which models construct features from raw order and trade records, revise their signals using bounded feedback, and submit a frozen strategy for out-of-sample assessment under common trading rules. This setting evaluates feature construction, signal combination, and strategy selection through their joint efect on the final portfolio.

We introduce Quant-Arena, a benchmark for end-to-end quantitative research on real A-share market data (Figure 2). At the input end, model-generated Python programs access raw Level-2 records, including individual orders and trades and order-book snapshots, through a controlled data interface. Models determine which features to construct and how to combine them into daily stock scores. During research, they can evaluate candidate strategies on 2022–2023, inspect the permitted feedback, and revise their code within a fixed budget. Each run then commits one frozen strategy. At the output end, the platform executes that strategy on the sealed year 2024 and measures the portfolio’s returns after trading costs.

Quant-Arena applies common portfolio and execution rules to make these outcomes comparable. Daily scores determine an equal-weight portfolio of the highest-ranked stocks, and a cash-account evaluator applies lotsize constraints and order-dependent costs, including market impact estimated from a simulated order book. Out-of-sample return and excess return over the market provide direct measures of profitability. Risk and signal statistics and the diversity of features used by the strategy provide complementary views of the result. Together, these measurements distinguish finding a predictive signal from turning it into a profitable portfolio.

A common research budget controls how much search each run can undertake. Repeated backtests favor candidates that fit historical noise (Bailey et al., 2017), so the selected strategy must be assessed on data that its research never observed. The platform also records candidate evaluations and final submissions. These records allow us to examine how models use feedback and select strategies alongside their eventual financial outcomes.

Our contributions are:

• An end-to-end quantitative research benchmark. Quant-Arena connects feature and signal construction from real market events to the out-of-sample performance of an executable strategy, making the complete research run the unit of evaluation.

• A common research and financial evaluation protocol. A bounded research process, frozen strategy submission, and sealed assessment under shared portfolio and trading rules support comparisons of profitability, risk, signal quality, and feature diversity.

• An empirical analysis of LLM quantitative research. We evaluate nine frontier models. No model beats the equal-weight market in 2024: the stocks a model ranks highest typically underperform the market average. The strongest model gains mainly by learning when to stay out of the market rather than by selecting better stocks.

• A public evaluation platform. Quant-Arena is released as a public arena that accepts submissions from any model or agent harness under the same protocol and sealed assessment.

The public leaderboard also lists Kepler-1, a proprietary strategy developed by Einsia AI.

![](images/f54b028cde740730c3fde19bdf9192d940370630245506904c44aa64250aa7b5.jpg)  
Figure 2 A model constructs features and stock-scoring code from raw market observations, revises candidates using research-period feedback, and freezes one strategy for assessment on the sealed period under common portfolio and execution rules.

## 2 The Quant-Arena Benchmark

Quant-Arena evaluates the out-of-sample profitability of stock-selection strategies developed by models from historical market data.

## 2.1 Task Formulation

A model receives a research objective, market-data access, and documentation for the data and evaluation interfaces. Within a fixed research budget, it writes a strategy f that constructs features, combines them into predictive signals, and scores stocks. For each trading day t, the strategy computes

$$
\mathbf { s } _ { t } = f ( \mathcal { H } _ { t } ) ,
$$

where $\mathcal { H } _ { t }$ contains the authorized historical observations available at that time and $\mathbf { s } _ { t }$ contains stock scores. The model determines the features and scoring function. A common financial evaluator converts these scores into portfolio positions and returns. The final submission is the frozen strategy program $f .$

## 2.2 Market Data and Benchmark Construction

We use real market observations from a fixed data release, stock universe, and trading calendar. Models construct features from the raw inputs without a predefined factor library. Missing observations and delisted stocks are retained without redefining daily universe membership.

The data interface restricts access by date, instrument, and field. At each decision date, the strategy can access authorized historical observations, including permitted warmup history. The same input definitions and access rules apply during research and assessment.

Table 1 Market data and temporal split used by the benchmark.
<table><tr><td>Property</td><td>Specification</td></tr><tr><td>Stock universe</td><td>Selected universe of 3,042 stocks</td></tr><tr><td>Inputs</td><td>Daily market data and raw Level-2 quotes, orders, and trades</td></tr><tr><td>Research period</td><td>2022-2023</td></tr><tr><td>Out-of-sample period</td><td>Sealed year 2024</td></tr></table>

## 2.3 Research and Assessment Protocol

Models may evaluate candidate strategies on the research period and revise their code using aggregate return, signal-quality and coverage statistics, and execution errors. Each run may evaluate at most five candidates on the research period. Participants may define training and validation splits within the research period; the benchmark prescribes no separate training window.

At the end of research, one strategy’s source code, execution environment, and declared features are frozen. The evaluator applies this strategy once to the sealed out-of-sample period. The strategy accesses the observations needed for execution through the authorized data interface. Neither these observations nor assessment feedback is returned to the model for further development.

## 2.4 Financial Evaluation and Metrics

The evaluator converts stock rankings into an equal-weight long portfolio targeting 75 stocks. The backtest maintains a cash account with 100-share lots, order-dependent trading costs, and market impact estimated from a simulated order book. Idle cash earns zero return, and daily net asset value includes cash and outstanding positions. Account returns are compounded over each evaluation period.

We report out-of-sample annualized return and excess return over a full-market equal-weight reference aligned to the same trading days. The reference is gross of costs. Long-only Sharpe $\mathrm { ( S R _ { L } ) }$ measures risk-adjusted performance, and maximum drawdown measures downside risk. Signal metrics include mean daily RankIC, its annualized information ratio (A), long–short Sharpe $\mathrm { ( S R _ { L S } ) }$ between the highest- and lowest-scored deciles, and turnover, the average daily replacement fraction of the portfolio. Daily RankIC is the rank correlation between stock scores and next-period returns; A is its mean divided by its standard deviation, multiplied by $\sqrt { N _ { \mathrm { a n n } } }$ , where $N _ { \mathrm { a n n } }$ denotes the number of trading days per year used for annualization.

The published in-sample score combines strategy quality and feature diversity. Let r be the strategy’s annualized return minus the reference’s annualized return, expressed as a decimal. Each metric is mapped to (0, 1) by a smooth, monotone transform,

$$
g _ { c } ( x ) = { \textstyle { \frac { 1 } { 2 } } } \left( 1 + { \frac { x } { c + | x | } } \right) ,
$$

which equals 0.5 at $x = 0$ , rewards gains with diminishing returns, and penalizes losses without truncating them to zero. The component scores are

$$
\begin{array} { l l l } { { s _ { \mathrm { R } } = g _ { 0 . 3 } ( r ) , } } & { { s _ { \mathrm { L } } = g _ { 1 } ( \mathrm { S R } _ { \mathrm { L } } ) , } } & { { s _ { \mathrm { L S } } = g _ { 3 } ( \mathrm { S R } _ { \mathrm { L S } } ) , } } \\ { { } } & { { } } & { { s _ { \mathrm { I C } } = \frac { 1 } { 2 } \left( 1 + \operatorname { t a n h } \displaystyle \frac { \mathrm { R a n k I C } } { 0 . 0 3 } \right) , } } & { { s _ { \mathrm { S T } } = g _ { 2 } ( A ) , } } & { { s _ { \mathrm { T O } } = \displaystyle \frac { 1 } { 1 + \mathrm { T u r n o v e r } } . } } \end{array}\tag{1}
$$

Quality is a weighted sum that places most of its weight on realized excess return:

$$
Q = 0 . 6 s _ { \mathrm { R } } + 0 . 2 s _ { \mathrm { L } } + 0 . 0 5 \left( s _ { \mathrm { L S } } + s _ { \mathrm { I C } } + s _ { \mathrm { S T } } + s _ { \mathrm { T O } } \right) .\tag{2}
$$

Excess return and long-only Sharpe measure what the portfolio earned; the four signal terms carry small weights because they vary less across runs and do not by themselves determine portfolio value. A strategy that tracks the market with no excess return and zero Sharpe has $s _ { \mathrm { { R } } } = s _ { \mathrm { { L } } } = 0 . 5$

A strategy may declare up to 300 named features. Correlated features are grouped, and K counts the qualifying clusters whose contribution to the strategy’s scores is detected. Diversity is

$$
D = Q { \frac { \operatorname* { m i n } ( K , 3 2 ) } { 3 2 } } .\tag{3}
$$

The fixed scale of 32 makes diversity grow linearly with K up to saturation, independently of the 300-feature declaration limit. Diversity scales with quality, so additional features add little to a weak strategy. The in-sample score is

$$
S _ { \mathrm { i n } } = 9 0 Q + 1 0 D .\tag{4}
$$

The out-of-sample score $S _ { \mathrm { o u t } }$ uses the same metric dimensions with an unpublished combination rule fixed for each benchmark version. The overall score is

$$
S = 0 . 3 S _ { \mathrm { i n } } + 0 . 7 S _ { \mathrm { o u t } } .\tag{5}
$$

We report individual metrics alongside the 0–100 scores.

## 3 Experiments

## 3.1 Experimental Setup

Our experiments address five questions. To evaluate how far current models are from a strong research agent, and how reasoning efort changes this, we compare 38 iterative configurations of nine models (Section 3.2). To evaluate why predictive skill does not become profit, we analyze next-day returns by signal decile (Section 3.3). To evaluate whether any strategy adapts to the market regime, we trace Astra’s account and exposure (Section 3.4). To evaluate what iteration contributes, we pair iterative and one-shot runs and follow candidates through the research period (Section 3.5). To evaluate whether models share a trading style, we measure their holding exposures (Section 3.6).

We run Claude Opus 5.5, Claude Sonnet 5.5, GPT-6 Astra, GPT-6 Sol, GPT-5.6 Terra, Gemini 3.8 Flash, DeepSeek V4.1 Flash, Kimi K3, and GLM-5.3 in the iterative and one-shot settings at every reasoning efort they support: six levels for Terra, five for Opus, Sonnet, Astra, and Sol, four for DeepSeek, three for Gemini and Kimi, and two for GLM. This gives 38 iterative and 38 one-shot configurations. The one-shot results are in Appendix B. We report the total score S with its research-period and sealed-period components, and for the sealed 2024 window the annualized return (ARR), excess ARR over the gross equal-weight market, mean RankIC, long-only Sharpe ratio, and maximum drawdown. The diagnostics in Sections 3.3–3.6 use, for each model, the iterative configuration with the highest total score (the shaded rows of Table 2).

## 3.2 Performance Across Models and Reasoning Efforts

Profitable strategies are rare, and only one beats the market. Thirteen of the 38 configurations earn positive ARR, and the only one above the 17.1% market is Sol at medium efort (18.4%, +1.4 pp). Most strategies keep under half of their capital in stocks (mean exposure 34–48% in 35 of the 38 runs), so they pay trading costs while capturing only part of the market’s gain.

Higher reasoning effort does not reliably raise the score. For six of the nine models the best efort is not the highest one supported (Table 2): Opus scores 44.28 at medium against 41.39 at max, Gemini 43.04 against 35.39, and Kimi 38.31 at low against 31.86 at max. Only Sonnet, Terra, and GLM peak at their highest efort. The range across a model’s eforts reaches 14.49 points (Terra), close to the 18.77-point range between the nine highest-efort scores. Efort acts through discrete design choices rather than gradual improvement. Astra’s xhigh and max runs add a market-state filter and hold cash on 70–75% of days, while its low, medium, and high runs stay about 47–48% invested, and its score jumps from 43.70 at high to 51.29

Table 2 Iterative results for every model and supported reasoning efort, Models are ordered by their best total score, and eforts increase within each model. Shaded rows mark each model’s highest-scoring efort, the configuration used by the diagnostics in Sections 3.3–3.6. Scores are on a 0–100 scale. Financial and signal metrics refer to the sealed 2024 window; the gross equal-weight market ARR is 17.1%. MDD is a negative return. Bold marks the best value in each column.
<table><tr><td>Model</td><td>Effort</td><td>S</td><td> $S _ { \mathrm { i n } }$ </td><td> $S _ { \mathrm { o u t } }$ </td><td>ARR (%)</td><td>Excess (pp)</td><td>RankIC</td><td>Sharpe</td><td>MDD (%)</td></tr><tr><td>GPT-6 Astra</td><td>low</td><td>47.22</td><td>58.10</td><td>42.56</td><td>2.3</td><td>-14.8</td><td>0.056</td><td>0.22</td><td>-25.7</td></tr><tr><td></td><td>medium</td><td>47.17</td><td>48.09</td><td>46.78</td><td>9.8</td><td>-7.2</td><td>0.044</td><td>0.53</td><td>-17.9</td></tr><tr><td></td><td>high</td><td>43.70</td><td>52.34</td><td>39.99</td><td>-0.9</td><td>-17.9</td><td>0.053</td><td>0.11</td><td>-26.3</td></tr><tr><td></td><td>xhigh</td><td>51.29</td><td>53.94</td><td>50.16</td><td>10.7</td><td>-6.3</td><td>0.101</td><td>0.86</td><td>-6.8</td></tr><tr><td></td><td>max</td><td>50.63</td><td>52.94</td><td>49.64</td><td>11.2</td><td>-5.9</td><td>0.082</td><td>0.81</td><td>-7.0</td></tr><tr><td> GPT-6 Sol</td><td>low</td><td>38.51</td><td>48.85</td><td>34.09</td><td>1.4</td><td>-15.7</td><td>-0.020</td><td>0.19</td><td>-30.1</td></tr><tr><td></td><td>medium</td><td>49.80</td><td>44.14</td><td>52.23</td><td>18.4</td><td>1.4</td><td>0.018</td><td>0.65</td><td>-35.9</td></tr><tr><td></td><td>high</td><td>42.89</td><td>52.16</td><td>38.92</td><td>-2.5</td><td>-19.5</td><td>0.042</td><td>0.10</td><td>-37.4</td></tr><tr><td></td><td>xhigh</td><td>51.14</td><td>55.50</td><td>49.26</td><td>8.6</td><td>-8.4</td><td>0.053</td><td>1.89</td><td>-2.8</td></tr><tr><td></td><td>max</td><td>45.31</td><td>46.49</td><td>44.81</td><td>8.1</td><td>-9.0</td><td>0.018</td><td>0.41</td><td>-30.0</td></tr><tr><td>GPT-5.6 Terra</td><td>none</td><td>33.09</td><td>26.67</td><td>35.85</td><td>2.4</td><td>-14.7</td><td>-0.042</td><td>0.22</td><td>-24.9</td></tr><tr><td></td><td>low</td><td>41.19</td><td>50.58</td><td>37.17</td><td>-4.6</td><td>-21.7</td><td>0.022</td><td>0.02</td><td>-31.4</td></tr><tr><td></td><td>medium</td><td>39.50</td><td>43.17</td><td>37.93</td><td>-4.3</td><td>-21.3</td><td>0.036</td><td>0.00</td><td>-29.0</td></tr><tr><td></td><td>high</td><td>41.54</td><td>44.72</td><td>40.17</td><td>0.0</td><td>-17.0</td><td>0.035</td><td>0.14</td><td>-29.9</td></tr><tr><td></td><td>xhigh</td><td>44.77</td><td>50.60</td><td>42.28 46.39</td><td>9.9</td><td>-7.1</td><td>-0.012</td><td>0.44</td><td>-32.6</td></tr><tr><td></td><td>max</td><td>47.58</td><td>50.37</td><td></td><td>8.4</td><td>-8.7</td><td>0.026</td><td>0.42</td><td>-28.8</td></tr><tr><td>Claude Opus 5.5</td><td>low</td><td>30.15</td><td>37.76</td><td>26.89</td><td>-29.4</td><td>-46.4</td><td>0.035</td><td>-1.21</td><td>-44.1</td></tr><tr><td></td><td>medium</td><td>44.28</td><td>56.78</td><td>38.92</td><td>4.4</td><td>-12.7</td><td>-0.002</td><td>0.29</td><td>-32.7</td></tr><tr><td></td><td>high</td><td>39.89</td><td>52.87</td><td>34.32</td><td>-11.2</td><td>-28.3</td><td>0.039</td><td>-0.25</td><td>-32.7</td></tr><tr><td></td><td>xhigh</td><td>41.92</td><td>54.48</td><td>36.53</td><td>-7.0</td><td>-24.0</td><td>0.043</td><td>-0.08</td><td>-33.2</td></tr><tr><td></td><td>max</td><td>41.39</td><td>50.41</td><td>37.52</td><td>-4.0</td><td>-21.1</td><td>0.030</td><td>-0.03</td><td>-25.1</td></tr><tr><td>Gemini 3.8 Flash</td><td>low</td><td>41.46</td><td>53.44</td><td>36.32</td><td>-6.2</td><td>-23.3</td><td>0.037</td><td>-0.04</td><td>-35.8</td></tr><tr><td></td><td>medium</td><td>43.04</td><td>53.58</td><td>38.52</td><td>-2.9</td><td>-19.9</td><td>0.045 0.022</td><td>0.06</td><td>-33.8</td></tr><tr><td></td><td>high</td><td>35.39</td><td>48.90</td><td>29.61</td><td>-19.1</td><td>-36.2</td><td></td><td>-0.49</td><td>-38.5</td></tr><tr><td> Claude Sonnet 5.5</td><td>low</td><td>34.97</td><td>47.91</td><td>29.43</td><td>-9.5</td><td>-26.6</td><td>-0.009</td><td>-0.13</td><td>-35.5</td></tr><tr><td></td><td>medium</td><td>39.42</td><td>43.39</td><td>37.73</td><td>-3.5</td><td>-20.6</td><td>0.041</td><td>-0.02</td><td>-24.3</td></tr><tr><td></td><td>high</td><td>40.50</td><td>54.11</td><td>34.67</td><td>-9.0</td><td>-26.1</td><td>0.044 0.015</td><td>-0.19</td><td>-34.7</td></tr><tr><td></td><td>xhigh</td><td>40.10 40.93</td><td>44.68</td><td>38.13 33.90</td><td>-0.5</td><td>-17.6</td><td>0.045</td><td>0.15</td><td>-35.0</td></tr><tr><td></td><td>max</td><td></td><td>57.33</td><td></td><td>-12.2</td><td>-29.2</td><td></td><td>-0.23</td><td>-36.1</td></tr><tr><td>Z GLM-5.3</td><td>high</td><td>38.67</td><td>49.31</td><td>34.11</td><td>-8.6</td><td>-25.7</td><td>0.018</td><td>-0.14</td><td>-31.6</td></tr><tr><td></td><td>max</td><td>38.91</td><td>47.96</td><td>35.03</td><td>-7.8</td><td>-24.9</td><td>0.046</td><td>-0.18</td><td>-27.8</td></tr><tr><td>KKimi K3</td><td>low</td><td>38.31</td><td>45.87</td><td>35.07</td><td>-8.5</td><td>-25.6</td><td>0.027</td><td>-0.10</td><td>-36.4</td></tr><tr><td></td><td>high</td><td>31.43</td><td>36.85</td><td>29.10</td><td>-20.0</td><td>-37.0</td><td>0.036</td><td>-0.84</td><td>-36.9</td></tr><tr><td></td><td>max</td><td>31.86</td><td>39.87</td><td>28.43</td><td>-25.6</td><td>-42.7</td><td>0.041</td><td>-1.14</td><td>-39.9</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>none</td><td>22.00</td><td>21.21</td><td>22.33</td><td>-24.1</td><td>-41.2</td><td>-0.022</td><td>-1.03</td><td>-29.5</td></tr><tr><td></td><td>low</td><td>36.45</td><td>48.46</td><td>31.30</td><td>-5.7</td><td>-22.7</td><td>-0.009</td><td>-0.06</td><td>-30.3</td></tr><tr><td></td><td>high</td><td>35.97</td><td>46.02</td><td>31.67</td><td>-11.3</td><td>-28.4</td><td>0.015</td><td>-0.25</td><td>-34.0</td></tr><tr><td></td><td>max</td><td>35.90</td><td>42.49</td><td>33.08</td><td>-11.0</td><td>-28.1</td><td>0.037</td><td>-0.28</td><td>-34.2</td></tr></table>

at xhigh. Sol follows the same pattern: its xhigh run is 5% invested on average against 34–46% at its other eforts.

## 3.3 Why Models Fall Short: Ranking Skill Does Not Reach the Portfolio

![](images/39392f794dad831e3d7c79eef5101eb6ee37f723a49e0ff378d5818255131720.jpg)  
Figure 3 Mean next-day return by daily score decile for the nine models’ best iterative configurations in 2024, in basis points (bp). H–L is the highest-minus-lowest decile spread, and the last entry in each header is the number of days with a signal.

Models rank stocks with some skill, but the stocks they buy are not the ones that earn the most. Opus, Sonnet, Gemini, DeepSeek, Kimi, and GLM show a hump (Figure 3): the second or third decile earns the most, 24–33 bp per day, and returns fall toward the top decile, so H–L lies between −10.5 and +4.7 bp. Terra and Astra rise across deciles (+28.6 and +38.7 bp), and Sol is flat. Average rank correlation does not predict profit: GLM has RankIC 0.046 and ARR −7.8%, while Opus at medium efort has RankIC −0.002 and ARR 4.4%, and across the 38 iterative runs the two are almost uncorrelated (Spearman 0.09). RankIC averages over the whole cross-section, but the portfolio holds only the 75 top-ranked stocks, and for most models that is where the decile profile declines.

## 3.4 Case Study: Astra Learns Market Timing

![](images/e1670f69283c20eaea5a5dce08639c1ca07eda59375d8b1c5b19d7fc05b4e5b7.jpg)  
Figure 4 Cash-account NAV of the iterative Astra run at xhigh efort in 2024, against the gross equal-weight market. Shaded bands mark days on which Astra holds stocks.

Astra’s lead comes from staying out of the market, not from choosing better stocks. Its program scores stocks only when the median five-session and twenty-session log returns of the universe are both positive, and holds cash otherwise. The first three candidates have no such rule. It appears in the fourth and fifth submissions, the fifth being the committed strategy, after research feedback showed net losses in both research years (−12.2% in 2022 and −13.6% in 2023). The rule keeps Astra out of the early-2024 decline (Figure 4): it first buys stocks on 28 February, its NAV is exactly 1.000 when the market bottoms at 0.704 on 6 February, and its maximum drawdown over January and February is −1.7% against −29.6% for the market. Over the year, Astra ends at 1.107 against 1.170 for the market, −6.3 pp of excess return, and 12.5% of its gain arrives in the five sessions from 24 to 30 September. Cash on 75% of trading days caps participation: Astra captures 16% of the market’s rebound from the February low, against 73% for Terra, so the lower drawdown comes with a lower upside. The behavior depends on efort: Astra’s low, medium, and high runs are invested on nearly every day and sufer drawdowns of −25.7%, −17.9%, and −26.3%.

Table 3 Paired iterative and one-shot results, each at the supported efort with the highest total score within its setting (eforts listed as iterative / one-shot). ∆ is iterative minus one-shot. $S _ { \mathrm { i n } }$ and $S _ { \mathrm { o u t } }$ are the research-period and sealed-period scores, and ARR diferences are in percentage points. The gross equal-weight market ARR is 17.1%.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Efforts</td><td colspan="3">S</td><td colspan="2">Window ∆</td><td colspan="3">ARR (%)</td></tr><tr><td>Iter.</td><td>One-shot</td><td> $\Delta$ </td><td> $\Delta { S _ { \mathrm { { i n } } } }$ </td><td> $\Delta S _ { \mathrm { o u t } }$ </td><td>Iter.</td><td>One-shot</td><td> $\Delta$ </td></tr><tr><td>GPT-6 Astra</td><td>xhigh / high</td><td>51.29</td><td>35.22</td><td>16.07</td><td>15.67</td><td>16.24</td><td>10.7</td><td>-4.8</td><td>15.5</td></tr><tr><td>GPT-6 Sol</td><td>xhigh / low</td><td>51.14</td><td>32.15</td><td>18.99</td><td>17.21</td><td>19.74</td><td>8.6</td><td>-7.6</td><td>16.2</td></tr><tr><td>GPT-5.6 Terra</td><td>max / none</td><td>47.58</td><td>42.14</td><td>5.44</td><td>3.81</td><td>6.14</td><td>8.4</td><td>9.5</td><td>-1.1</td></tr><tr><td>Claude Opus 5.5</td><td>medium / medium</td><td>44.28</td><td>42.61</td><td>1.67</td><td>9.59</td><td>-1.73</td><td>4.4</td><td>5.6</td><td>-1.2</td></tr><tr><td>Gemini 3.8 Flash</td><td>medium / medium</td><td>43.04</td><td>35.91</td><td>7.13</td><td>17.96</td><td>2.48</td><td>-2.9</td><td>1.6</td><td>-4.5</td></tr><tr><td>Claude Sonnet 5.5</td><td>max / high</td><td>40.93</td><td>40.24</td><td>0.69</td><td>13.05</td><td>-4.60</td><td>-12.2</td><td>1.4</td><td>-13.6</td></tr><tr><td>GLM-5.3</td><td>max /high</td><td>38.91</td><td>38.02</td><td>0.89</td><td>-4.58</td><td>3.24</td><td>-7.8</td><td>-4.2</td><td>-3.6</td></tr><tr><td>Kimi K3</td><td>low / high</td><td>38.31</td><td>32.01</td><td>6.30</td><td>7.98</td><td>5.59</td><td>-8.5</td><td>-20.5</td><td>12.0</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>low / max</td><td>36.45</td><td>29.92</td><td>6.53</td><td>12.51</td><td>3.96</td><td>-5.7</td><td>-11.9</td><td>6.2</td></tr><tr><td>Mean</td><td></td><td>43.55</td><td>36.47</td><td>7.08</td><td>10.36</td><td>5.67</td><td>-0.6</td><td>-3.4</td><td>2.9</td></tr></table>

## 3.5 Iteration Raises Scores More Reliably Than Returns

Iteration raises the score for every model but the return for only four. Pairing each model’s best iterative and one-shot configurations (Table 3), the mean score rises from 36.47 to 43.55, with gains from 0.69 (Sonnet) to 18.99 (Sol). ARR is higher in the iterative run for Sol, Astra, Kimi, and DeepSeek (by 6.2 to 16.2 pp) and lower for the other five, so the mean gain is only 2.9 pp. Feedback improves the research window more than the sealed year: the mean gain in $S _ { \mathrm { i n } }$ is 10.36 points against 5.67 in $S _ { \mathrm { o u t } }$ , and the iterative best configurations lose 12.4 points between the two windows against 7.7 for one-shot. Part of what iteration learns is specific to 2022–2023.

![](images/89bd3afc7bccfa1f65286d715101cd978e27ec646458111a9bf081e56977cb5f.jpg)

![](images/bcabab666bef8c2bfcb9031e0fd9e42efea0a56324a5cbab2b851c9d4e4a62ea.jpg)  
Figure 5 Research-period progress of the nine iterative runs. (a) Best-so-far excess ARR gain over the first valid candidate; means use the runs available at each index. (b) First, best, and committed candidates on 2022–2023.

Most research progress comes from repairing a weak first candidate. Mean best-so-far gains at validcandidate indices one to five are 0, 14.4, 16.7, 29.1, and 36.5 pp over 9, 9, 7, 4, and 3 runs (Figure 5), but the median gain is 6.5 pp at the second candidate and 7.0 pp at the end of the run. Two runs drive the mean: DeepSeek and Sol improve by 57.1 and 50.7 pp from first candidates near −44 and −45 pp excess ARR, and without them the mean final gain is 6.5 pp. The second candidate is the research-period best in four of nine runs, and three runs commit their best candidate. Large gains come from fixing a clear error in a poor first candidate, and once candidates are near break-even, further evaluations add only a few points.

## 3.6 Trading Styles

![](images/7eb11ed33309c6b4c15fd6d9f699645e157081a85eafe15627feb8253115f215.jpg)  
Figure 6 Holding-value-weighted style exposure of the nine models’ best iterative configurations in 2024, in crosssectional standard deviations relative to the market. Astra’s estimates use its 61 invested days.

Most models converge on small-cap, low-volatility, value-tilted holdings, and style does not separate winners from losers. Eight of nine models tilt toward smaller stocks, seven toward higher book-to-price, and most toward lower volatility and turnover (Figure 6). Opus and DeepSeek are the exceptions, leaning toward higher volatility (+0.54 and +0.46) and turnover (+0.56 and +0.57), and Terra stays within 0.11 of the market on every dimension. Earnings-to-price is negative for six models, so the value tilt comes from book value rather than earnings. Similar tilts end in diferent places: Astra (book-to-price +0.96, volatility −1.05) finishes at 1.11 and Sonnet (+0.54, −0.64) at 0.88. Account outcomes are set by market exposure and by the head of each ranking, as the previous sections show, and the style tilt adds little to them.

## 3.7 Kepler-1 Against the Models

![](images/1af3bfc0ba08f83ede0997952627c2681b9229752fc809ff705ab71b781cb88f.jpg)  
Figure 7 Cash-account NAV in 2024 for the nine models’ best iterative configurations and Kepler-1, against the gross equal-weight market.

Kepler-1 turns its edge into account value, while the best model barely finishes above its starting capital. Kepler-1 scores 77.56 and ends 2024 at 1.93 against 1.17 for the market (Figure 7), and four models finish above their starting capital: Astra (1.11), Sol (1.09), Terra (1.08), and Opus (1.04). The other five end between 0.88 (Sonnet) and 0.97 (Gemini). The lead does not come from a better average rank signal: Kepler-1’s RankIC of 0.085 is below Astra’s 0.101. It comes from converting signal into return, with a Sharpe ratio of 2.98 against 1.89 for the best model.

## 4 Related Work

## 4.1 Formulaic Alpha Mining with Language Models

Formulaic alpha mining searches for executable expressions that transform market observations into predictive cross-sectional signals. Published formula libraries demonstrate the interpretability and deployability of this representation (Kakushadze, 2016). Automated approaches then use evolutionary search or learned generators to explore expression spaces and combine the resulting factors (Zhang et al., 2020; Shi et al., 2024). These explicit search spaces support reproducible execution, but they are combinatorial, highly exposed to noisy backtest feedback, and often optimized around a narrow factor statistic rather than the behavior of the final portfolio.

LLMs extend this paradigm by connecting economic hypotheses, symbolic construction, and empirical diagnosis. LLMFactor uses sequential knowledge-guided prompting to extract interpretable factors from financial information (Wang et al., 2024); Automate Strategy Finding with LLM combines multimodal market inputs with a multi-agent portfolio workflow (Kou et al., 2024); Alpha-GPT introduces interactive LLM-based alpha generation (Wang et al., 2023); AlphaAgent regularizes generation for originality, hypothesis–factor alignment, and structural simplicity (Tang et al., 2025); and QuantaAlpha treats a complete mining trajectory as the unit of evolution and reuses efective sub-trajectories through mutation and crossover (Han et al., 2026). These systems establish the value of structured, multi-stage alpha discovery beyond one-shot formula generation. Quant-Arena provides a common protocol for evaluating the resulting model runs with executable strategy outputs and a frozen final-assessment boundary.

## 4.2 Quantitative Benchmarks and Evaluation Protocols

Existing financial infrastructure serves several diferent evaluation goals. Qlib standardizes AI-oriented quantitative workflows, while FinRL provides reproducible reinforcement-learning environments and trading evaluations (Yang et al., 2020; Liu et al., 2020). Financial language benchmarks such as FinBen and Finance Agent Benchmark evaluate domain understanding, generation, forecasting, and real-world research tasks (Xie et al., 2024; Bigeard et al., 2025); general agent benchmarks such as AgentBench and MLAgentBench evaluate multi-turn interaction, tool use, and iterative experimentation (Liu et al., 2023; Huang et al., 2023). AlphaBench focuses specifically on formulaic alpha generation, evaluation, and search tasks (Luo et al., 2026), while AlphaForgeBench targets reproducible end-to-end factor and strategy design with LLMs (Zhang et al., 2026). Recent trading-agent evaluation also emphasizes leakage control, attribution, and memory efects in historical markets (Zhu et al., 2026).

These benchmarks also difer in their unit of evaluation. AlphaQT-Bench (AlphaQT-Bench) treats executability, causal safety, functional correctness, and structural compliance as distinct code-level outcomes, with a small number of repair rounds. StockBench (StockBench) evaluates repeated sequential decisions on a small stock universe and reports passive-hold references, whereas QuantBench (QuantBench) studies conventional quantitative models, feature sets, and training seeds rather than LLM research systems. AlphaForgeBench expands the number of generated implementations through many short tasks, and KTD-Fin separates trading agent behavior from information access and attribution. These designs motivate reporting execution status, financial outcomes, and information boundaries.

Evaluating a quantitative benchmark also requires controlling the process that produced the final output. Repeated access to noisy backtests creates opportunities for multiple testing and selection bias (Bailey et al., 2017). Quant-Arena addresses this requirement through frozen data releases, date-scoped execution, bounded research budgets, bounded interaction feedback, explicit financial conventions, and sealed assessment of one committed strategy. The resulting protocol complements code, factor, and trading-decision benchmarks by connecting raw feature construction and bounded research to the sealed financial outcome of a committed strategy.

## 5 Conclusion

Quant-Arena evaluates a complete quantitative research run, from constructing features on raw market observations to revising candidates and committing one strategy for sealed financial assessment. A shared stock-scoring interface, portfolio rule, and cash-account evaluator connect model-authored research to comparable simulated outcomes.

Across 38 iterative configurations, the best model scores 51.3 against 77.6 for Kepler-1, a proprietary strategy developed by Einsia AI, and only one configuration, Sol at medium efort, beats the equal-weight market. Pairing each model’s best iterative and one-shot configurations, iteration raises the overall score for all nine models but sealed-period ARR for only four. Decile returns show why: most models rank stocks with some skill, yet the top-ranked names that the portfolio buys earn less than middle-ranked ones, so an average predictive correlation is insuficient to judge an investable strategy. The best-performing Astra configuration profits mainly by staying in cash during the early-2024 decline, which lowers drawdown without beating the market. Candidate trajectories show that research progress comes early and plateaus, and the efort sweep shows that a higher reasoning setting does not consistently improve the result. Together, these findings support evaluating quantitative research through both its development process and the financial outcome of its committed strategy.

## References

AlphaQT-Bench. AlphaQT-Bench: Benchmarking quantitative trading code generation and repair. Findings of ACL 2026, 2026. https://aclanthology.org/2026.findings-acl.138.pdf.

David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, and Qiji Jim Zhu. The probability of backtest overfitting. The Journal of Computational Finance, 20(4):39–69, 2017. doi: 10.21314/JCF.2016.322. https://doi. org/10.21314/JCF.2016.322.

Antoine Bigeard, Langston Nashold, Rayan Krishnan, and Shirley Wu. Finance agent benchmark: Benchmarking LLMs on real-world financial research tasks. arXiv preprint arXiv:2508.00828, 2025. https://arxiv.org/abs/2508. 00828.

Jun Han, Shuo Zhang, Wei Li, Yifan Dong, Tu Hu, Yumo Zhu, Xiaomin Yu, Xin Guo, Zhaowei Liu, Kunyi Wang, Jingping Liu, Tianyi Jiang, Ruichuan An, Sen Hu, Zhi Yang, Ronghao Che, and Huacan Wang. QuantaAlpha: An evolutionary framework for LLM-driven alpha mining. arXiv preprint arXiv:2602.07085, 2026. https://arxiv.org abs/2602.07085.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. MLAgentBench: Evaluating language agents on machine learning experimentation. arXiv preprint arXiv:2310.03302, 2023. https://arxiv.org/abs/2310.03302.

Zura Kakushadze. 101 formulaic alphas. Wilmott Magazine, 2016(84):72–80, 2016. https://arxiv.org/abs/1601.00991.

Zhizhuo Kou, Holam Yu, Jingshuo Peng, and Lei Chen. Automate strategy finding with LLM in quant investment. arXiv preprint arXiv:2409.06289, 2024. https://arxiv.org/abs/2409.06289.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as agents. arXiv preprint arXiv:2308.03688, 2023. https://arxiv.org/abs/2308.03688.

Xiao-Yang Liu, Hongyang Yang, Qian Chen, Runjia Zhang, Liuqing Yang, Bowen Xiao, and Christina Dan Wang. FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. arXiv preprint arXiv:2011.09607, 2020. https://arxiv.org/abs/2011.09607.

Haohen Luo, Ho Tin Ko, Jiandong Chen, David Sun, Yuan Zhang, and Chen Liu. AlphaBench: Benchmarking large language models in formulaic alpha factor mining. In The Fourteenth International Conference on Learning Representations, 2026. https://openreview.net/forum?id=d97Q8r7ZKZ.

QuantBench. QuantBench: Benchmarking quantitative models and features. arXiv preprint arXiv:2504.18600, 2025. https://arxiv.org/abs/2504.18600.

Hao Shi, Weili Song, Xinting Zhang, Jiahe Shi, Cuicui Luo, Xiang Ao, Hamid Arian, and Luis Seco. AlphaForge: A framework to mine and dynamically combine formulaic alpha factors. arXiv preprint arXiv:2406.18394, 2024. https://arxiv.org/abs/2406.18394.

StockBench. StockBench: Benchmarking large language models for sequential stock trading. arXiv preprint arXiv:2510.02209, 2025. https://arxiv.org/abs/2510.02209.

Ziyi Tang, Zechuan Chen, Jiarui Yang, Jiayao Mai, Yongsen Zheng, Keze Wang, Jinrui Chen, and Liang Lin. AlphaAgent: LLM-driven alpha mining with regularized exploration to counteract alpha decay. arXiv preprint arXiv:2502.16789, 2025. https://arxiv.org/abs/2502.16789.

Meiyun Wang, Kiyoshi Izumi, and Hiroki Sakaji. LLMFactor: Extracting profitable factors through prompts for explainable stock movement prediction. arXiv preprint arXiv:2406.10811, 2024. https://arxiv.org/abs/2406.10811.

Saizhuo Wang, Hang Yuan, Leon Zhou, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. Alpha-GPT: Human–AI interactive alpha mining for quantitative investment. arXiv preprint arXiv:2308.00016, 2023. https://arxiv.org/ abs/2308.00016.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, Yijing Xu, Haoqiang Kang, Ziyan Kuang, Chenhan Yuan, Kailai Yang, Zheheng Luo, Tianlin Zhang, Zhiwei Liu, Guojun Xiong, Zhiyang Deng, Yuechen Jiang, Zhiyuan Yao, Haohang Li, Yangyang Yu, Gang Hu, Jiajia Huang, Xiao-Yang Liu, Alejandro Lopez-Lira, Benyou Wang, Yanzhao Lai, Hao Wang, Min Peng, Sophia Ananiadou, and Jimin Huang. FinBen: A holistic financial benchmark for large language models. arXiv preprint arXiv:2402.12659, 2024. https://arxiv.org/abs/2402.12659.

Xiao Yang, Weiqing Liu, Dong Zhou, Jiang Bian, and Tie-Yan Liu. Qlib: An AI-oriented quantitative investment platform. arXiv preprint arXiv:2009.11189, 2020. https://arxiv.org/abs/2009.11189.

Tianping Zhang, Yuanqi Li, Yifei Jin, and Jian Li. AutoAlpha: An eficient hierarchical evolutionary algorithm for mining alpha factors in quantitative investment. arXiv preprint arXiv:2002.08245, 2020. https://arxiv.org/abs/ 2002.08245.

Wentao Zhang, Mingxuan Zhao, Jincheng Gao, Jieshun You, Huaiyu Jia, Yilei Zhao, Bo An, and Shuo Sun. AlphaForgeBench: Benchmarking end-to-end trading strategy design with large language models. arXiv preprint arXiv:2602.18481, 2026. https://arxiv.org/abs/2602.18481.

Taojie Zhu, Wentao Zhao, Rui Sun, Beidi Luan, Jiacheng Lu, Sinuo Wang, Jing Li, Daxin Jiang, Yonghong He, and Zuo Bai. From knowing to doing: A memory-controlled benchmark for LLM trading agents on stock markets. arXiv preprint arXiv:2605.28359, 2026. https://arxiv.org/abs/2605.28359.

## A Suite Specification

Table 4 lists the benchmark specification. A strategy may declare up to 300 named features.

Table 4 Public specification of the selected A-share suite.
<table><tr><td>Field</td><td>Specification</td></tr><tr><td>Market and universe</td><td>Selected A-shares; static universe of 3,042 stocks; missing and delisted stocks re- tained</td></tr><tr><td>Input profile</td><td>Raw daily and Level-2 market data through a date-scoped data interface; Python strategies</td></tr><tr><td>Windows</td><td>In-sample research 2022-2023 with authorized warmup history; sealed out-of- sample 2024; participants may define training and validation splits within the</td></tr><tr><td>Budget</td><td>research period At most 5 research evaluations</td></tr><tr><td>Feedback (iterative track)</td><td>Aggregate return, signal, and coverage statistics on the research window, plus execution errors</td></tr><tr><td>Portfolio</td><td>Top-75 equal-weight long portfolio, boundary ties retained; Top-200/400 diagnos- tics</td></tr><tr><td>Accounting</td><td>Cash account with 100-share lots; idle cash earns zero; liquidation at window end</td></tr><tr><td>Costs</td><td>Order-dependent trading costs and market impact estimated from a simulated order book</td></tr><tr><td>Features</td><td>Up to 300 declared features; usage tested by score-rank regression on correlation clusters; diversity normalized by a fixed scale of 32</td></tr><tr><td>Scoring</td><td>Sin = 90Q + 10D; total 0.3 Sin + 0.7 Sout</td></tr><tr><td>Runs</td><td>One run per submission; results published automatically and not withdrawable; daily submission limit per account</td></tr></table>

Boards. One-shot and Iterative use the standard model harness. System entries use participant-controlled agents; Custom entries support self-hosted models or unsupported endpoints with self-reported configurations.

Result records. For every run, the platform records the model, track, harness, all model calls, every research evaluation, the committed strategy, and both window assessments. Failed and cancelled runs remain in the record. Each oficial leaderboard entry links to its run record.

## B One-Shot Results and Matched-Effort Comparison

## B.1 One-Shot Results Across Reasoning Efforts

Table 5 reports the 38 one-shot runs in the layout of Table 2. The median configuration scores 31.61, and 4 of the 38 exceed 40 points, against 21 in the iterative setting. Six configurations have positive sealed-period ARR, and none exceeds the 17.1% market.

Without feedback, reasoning effort is just as inconsistent as with it. For eight of the nine models the best efort is not the highest one supported (Table 5). Terra scores 42.14 with no reasoning and 20.98 at max, a 21.16-point range that exceeds every iterative range, and Sol is best at low (32.15) and lowest at max (22.31). Only DeepSeek rises with efort, from 20.48 with no reasoning to 29.92 at max. With no feedback loop to correct a poor draft, each cell reflects a single draft, so the scores swing more with efort.

## B.2 Matched-Effort Comparison

Iteration helps at matched efforts as well. At each model’s highest supported efort (Gemini: high; others: max), the iterative run outscores the one-shot run for all nine models, with mean scores of 40.88 against 30.18, and has the higher ARR in six of the nine pairs. Across all 38 matched model–efort cells, the iterative run has the higher score in 34, the higher ARR in 26, and the higher RankIC in 28. Matching eforts gives a

Table 5 One-shot results for every model and supported reasoning efort, Layout and shading follow Table 2; shaded rows mark each model’s highest-scoring one-shot efort. The one-shot setting receives no performance feedback.
<table><tr><td>Model</td><td>Effort</td><td>S</td><td> $S _ { \mathrm { i n } }$ </td><td> $S _ { \mathrm { o u t } }$ </td><td>ARR (%)</td><td>Excess (pp)</td><td>RankIC</td><td>Sharpe</td><td>MDD (%)</td></tr><tr><td> Claude Opus 5.5</td><td>low</td><td>35.04</td><td>43.21</td><td>31.53</td><td>-4.5</td><td>-21.6</td><td>-0.011</td><td>0.00</td><td>-33.0</td></tr><tr><td></td><td>medium</td><td>42.61</td><td>47.19</td><td>40.65</td><td>5.6</td><td>-11.5</td><td>-0.001</td><td>0.34</td><td>-21.0</td></tr><tr><td></td><td>high</td><td>36.07</td><td>39.68</td><td>34.52</td><td>-7.5</td><td>-24.5</td><td>0.020</td><td>-0.14</td><td>-29.7</td></tr><tr><td></td><td>xhigh</td><td>39.13</td><td>49.05</td><td>34.88</td><td>-9.9</td><td>-26.9</td><td>0.040</td><td>-0.20</td><td>-35.0</td></tr><tr><td></td><td>max</td><td>39.70</td><td>42.07</td><td>38.69</td><td>-3.7</td><td>-20.8</td><td>0.046</td><td>0.03</td><td>-35.9</td></tr><tr><td> GPT-5.6 Terra</td><td>none</td><td>42.14</td><td>46.56</td><td>40.25</td><td>9.5</td><td>-7.6</td><td>-0.010</td><td>0.44</td><td>-27.8</td></tr><tr><td></td><td>low</td><td>27.82</td><td>26.59</td><td>28.35</td><td>-10.3</td><td>-27.4</td><td>-0.007</td><td>-0.24</td><td>-25.7</td></tr><tr><td></td><td>medium</td><td>21.01</td><td>20.44</td><td>21.25</td><td>-32.4</td><td>-49.4</td><td>-0.010</td><td>-1.51</td><td>-34.1</td></tr><tr><td></td><td>high</td><td>40.61</td><td>42.88</td><td>39.64</td><td>6.2</td><td>-10.9</td><td>-0.001</td><td>0.35</td><td>-27.5</td></tr><tr><td></td><td>xhigh</td><td>34.21</td><td>42.94</td><td>30.46</td><td>-9.8</td><td>-26.8</td><td>0.000</td><td>-0.17</td><td>-33.6</td></tr><tr><td></td><td>max</td><td>20.98</td><td>20.98</td><td>20.97</td><td>-32.3</td><td>-49.3</td><td>-0.009</td><td>-1.60</td><td>-32.6</td></tr><tr><td>Claude Sonnet 5.5</td><td>low</td><td>37.75</td><td>42.41</td><td>35.76</td><td>1.1</td><td>-16.0</td><td>-0.007</td><td>0.19</td><td>-30.3</td></tr><tr><td></td><td>medium</td><td>33.82</td><td>41.81</td><td>30.39</td><td>-11.5</td><td>-28.6</td><td>0.002</td><td>-0.21</td><td>-35.5</td></tr><tr><td></td><td>high</td><td>40.24</td><td>44.28</td><td>38.50</td><td>1.4</td><td>-15.6</td><td>0.010</td><td>0.20</td><td>-32.1</td></tr><tr><td></td><td>xhigh</td><td>37.05</td><td>47.90</td><td>32.40</td><td>-14.3</td><td>-31.3</td><td>0.029</td><td>-0.33</td><td>-36.0</td></tr><tr><td></td><td>max</td><td>33.89</td><td>41.19</td><td>30.77</td><td>-14.8</td><td>-31.9</td><td>0.018</td><td>-0.33</td><td>-36.9</td></tr><tr><td>Z GLM-5.3</td><td>high</td><td>38.02</td><td>52.54</td><td>31.79</td><td>-4.2</td><td>-21.3</td><td>-0.023</td><td>0.05</td><td>-37.3</td></tr><tr><td></td><td>max</td><td>31.46</td><td>34.70</td><td>30.07</td><td>-15.4</td><td>-32.4</td><td>0.022</td><td>-0.51</td><td>-32.1</td></tr><tr><td>1 Gemini 3.8 Flash</td><td>low</td><td>27.75</td><td>21.26</td><td>30.53</td><td>-12.1</td><td>-29.1</td><td>0.005</td><td>-0.41</td><td>-23.5</td></tr><tr><td></td><td>medium</td><td>35.91</td><td>35.62</td><td>36.04</td><td>1.6</td><td>-15.4</td><td>-0.006 -0.003</td><td>0.19</td><td>-23.8</td></tr><tr><td></td><td>high</td><td>30.26</td><td>25.93</td><td>32.11</td><td>-5.4</td><td>-22.5</td><td></td><td>-0.06</td><td>-24.6</td></tr><tr><td>GPT-6 Astra</td><td>low</td><td>30.08</td><td>35.05</td><td>27.95</td><td>-11.2</td><td>-28.3</td><td>-0.010</td><td>-0.28</td><td>-30.3</td></tr><tr><td></td><td>medium</td><td>31.02</td><td>36.73</td><td>28.57</td><td>-9.7</td><td>-26.7 -21.9</td><td>-0.012 0.007</td><td>-0.20</td><td>-30.3</td></tr><tr><td></td><td>high</td><td>35.22</td><td>38.27</td><td>33.92 26.80</td><td>-4.8</td><td>-33.8</td><td>-0.002</td><td>-0.03 -0.48</td><td>-30.9</td></tr><tr><td></td><td>xhigh max</td><td>30.30 31.47</td><td>38.46 40.13</td><td>27.75</td><td>-16.7 -18.3</td><td>-35.3</td><td>0.007</td><td>-0.54</td><td>-34.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-34.8</td></tr><tr><td> GPT-6 Sol</td><td>low</td><td>32.15</td><td>38.29</td><td>29.52</td><td>-7.6</td><td>-24.7</td><td>-0.014</td><td>-0.08</td><td>-32.2</td></tr><tr><td></td><td>medium</td><td>22.60</td><td>22.62</td><td>22.59</td><td>-31.4</td><td>-48.4</td><td>-0.006</td><td>-1.31</td><td>-34.1</td></tr><tr><td></td><td>high</td><td>23.15</td><td>21.07</td><td>24.04</td><td>-17.4</td><td>-34.4</td><td>-0.020 -0.022</td><td>-0.51</td><td>-29.6</td></tr><tr><td></td><td>xhigh max</td><td>23.29</td><td>23.84</td><td>23.05</td><td>-19.0</td><td>-36.1</td><td>-0.018</td><td>-0.60</td><td>-29.6</td></tr><tr><td></td><td></td><td>22.31</td><td>24.08</td><td>21.55</td><td>-23.7</td><td>-40.7</td><td></td><td>-0.98</td><td>-32.5</td></tr><tr><td>KKimi K3</td><td>low</td><td>31.58</td><td>38.58</td><td>28.58</td><td>-13.4</td><td>-30.5</td><td>0.000</td><td>-0.34</td><td>-33.8</td></tr><tr><td></td><td>high</td><td>32.01</td><td>37.89</td><td>29.48</td><td>-20.5</td><td>-37.6</td><td>0.038</td><td>-0.84</td><td>-38.6</td></tr><tr><td></td><td>max</td><td>31.65</td><td>40.87</td><td>27.70</td><td>-20.3</td><td>-37.3</td><td>0.016</td><td>-0.86</td><td>-35.1</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>none</td><td>20.48</td><td>22.01</td><td>19.82</td><td>-34.1</td><td>-51.1</td><td>-0.015</td><td>-1.79</td><td>-37.5</td></tr><tr><td></td><td>low</td><td>29.36</td><td>34.21</td><td>27.28</td><td>-17.4</td><td>-34.4</td><td>0.001</td><td>-0.49</td><td>-33.3</td></tr><tr><td></td><td>high</td><td>29.63</td><td>37.51</td><td>26.25</td><td>-25.4</td><td>-42.4</td><td>0.010</td><td>-0.92</td><td>-39.5</td></tr><tr><td></td><td>max</td><td>29.92</td><td>35.95</td><td>27.34</td><td>-11.9</td><td>-29.0</td><td>-0.008</td><td>-0.42</td><td>-26.7</td></tr></table>

larger gap than pairing best configurations (Table 3) because one-shot scores fall with efort more often, so a one-shot run at max efort is rarely its model’s best.

## C Additional Strategy Diagnostics

## C.1 Signal Similarity and Portfolio Overlap

a Spearman rank correlation

![](images/c7319da5430b230bbe85735f8e1a1cfccc182e5b98fce4c46588354576772ced.jpg)

b Top-75 Jaccard overlap (%)  
![](images/ff1de1ae8d4571b3207f9fdadf1c9ed5d44a56fc66242506e621ed7ec9da95ea.jpg)  
Figure 8 Pairwise mean daily Spearman signal correlations (a) and Top-75 Jaccard overlaps (b) among the nine models’ best iterative configurations in 2024 (36 pairs). Portfolio overlap measures the selected score sets, rather than identical realized trades under execution constraints.

The mean pairwise rank correlation is 0.47, while mean Top-75 Jaccard overlap is only 6.1% and its maximum is 35.6% (Figure 8). Broadly similar rankings therefore need not select the same extreme tail. Opus and DeepSeek are the most similar pair, with correlation 0.94 and overlap 35.6%, followed by Sol–GLM (0.87) and Sonnet–Astra (0.84). Terra is the most distinct strategy, with a mean correlation of 0.14 against the other eight.

Opus and DeepSeek are close to each other and correlate at about 0.5 or less with every other model. This agrees with the shared tilt toward higher volatility and turnover in Figure 6. The remaining models largely form one cluster of related signals.