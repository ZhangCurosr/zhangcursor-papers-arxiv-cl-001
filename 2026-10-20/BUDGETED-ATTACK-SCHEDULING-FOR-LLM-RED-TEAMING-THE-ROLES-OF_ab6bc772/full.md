# BUDGETED ATTACK SCHEDULING FOR LLM RED-TEAMING: THE ROLES OF STATIC PRIORS, SEMANTIC GUIDANCE, AND ONLINE FEEDBACK

Zhicheng Fang<sup>1,∗,†</sup> Han Sun<sup>1,2,∗</sup> Jiachen Lei<sup>1,3</sup> Xinyi Fang Zongmin Zhang<sup>1,4</sup> Chenxu Fu<sup>1</sup> Sizhe Fang<sup>1,5</sup> Wei Xu<sup>1,6,†</sup>

<sup>1</sup>Shanghai Qi Zhi Institute <sup>2</sup>East China Normal University

<sup>3</sup>The Chinese University of Hong Kong

<sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>5</sup>Georgia Institute of Technology <sup>6</sup>Tsinghua University

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors. fangzhicheng@sqz.ac.cn weixu@tsinghua.edu.cn

## ABSTRACT

As jailbreak attacks against large language models (LLMs) become increasingly diverse, red-teaming has come to rely on large libraries of jailbreak prompts and attack strategies to evaluate the robustness of victim models and their defense settings. However, attack effectiveness is highly condition-dependent: an attack that succeeds under one victim-defense setting may fail under another, while exhaustively evaluating the full attack library for each setting incurs substantial token and monetary costs. This motivates studying how jailbreak attacks should be prioritized under a limited attempt budget for efficient robustness evaluation. In this work, we formalize this problem as budgeted attack scheduling over a fixed attack library and investigate three complementary signals under a fixed victim–defense condition: (1) static historical attack success rate (ASR) priors; (2) semantic guidance from LLMs; and (3) online feedback from earlier queries. Building on these signals, we design four scheduling strategies and evaluate them across three victim models and three defense settings on GPT-5.2. Extensive experiments clearly characterize the conditions under which each signal is most effective. Historical ASR provides a strong initialization, but its reliability degrades under defenseinduced distribution shifts. Semantic guidance consistently improves or matches static prior-based scheduling in the cold-start regime, while online feedback becomes particularly valuable when transferred priors become stale. Semantic guidance and online feedback can also partially compensate for missing historical priors. Taken together, these findings characterize budgeted red-teaming as an attack scheduling problem in which historical priors, semantic guidance, and online feedback play complementary roles.

## 1 INTRODUCTION

Jailbreak attacks against large language models (LLMs) are rapidly evolving, becoming increasingly diverse in form and sophisticated in mechanism (Yi et al., 2024; Liu et al., 2024b). To cover a broader range of potential safety risks, jailbreak red-teaming is gradually shifting from evaluating individual attack methods in isolation toward systematic testing with large-scale attack libraries. Under this paradigm, researchers typically collect, reproduce, or construct diverse jailbreak strategies and evaluate them across a wide range of harmful queries, target models, and defense mechanisms to assess the robustness of LLMs and their associated safety defenses (Ganguli et al., 2022; Mazeika et al., 2024; Chao et al., 2024; Sheng et al., 2026; Fang et al., 2026).

As new attack strategies continue to emerge, combine, and evolve, jailbreak attack libraries are rapidly growing in scale. However, the expansion of these libraries also introduces practical challenges: each attack attempt incurs additional token, computational, and monetary costs, while its success is highly dependent on the specific harmful query, victim model, and defense setting (Sun et al., 2025; Fang et al., 2026). Consequently, exhaustively evaluating the entire attack library is often prohibitively expensive or even infeasible in practical red-teaming scenarios. Therefore, a key research question arises: how should jailbreak attacks be prioritized for efficient robustness evaluation under a limited attempt budget?

![](images/4cc408a80c2902857112d37a6ffeafb3ccf3bfd9faf137d107320b4cd5a85a6d.jpg)  
Figure 1: Workflow of budgeted attack scheduling. Given an attack library and an evaluation setting, the scheduler selects an ordered sequence of attacks under an attempt budget K. The execution module then applies these attacks sequentially until a jailbreak succeeds or the budget is exhausted, and reports budgeted attack performance, cost-related metrics, and scheduling-related analysis.

We formalize this problem as budgeted attack scheduling, as illustrated in Figure 1. Given an attack library and an evaluation setting, the scheduler uses the available signals to order attacks under an attempt budget K. The evaluator executes this order until the first successful jailbreak or budget exhaustion. This framework enables systematic evaluation of both attack effectiveness and scheduling efficiency, including attack success rate, attempt efficiency, token consumption, scheduling gain, and order sensitivity.

At the core of this framework is the Budgeted Scheduler. We consider three complementary sources of information for attack prioritization: (1) historical attack success rate (ASR) priors, (2) semantic guidance from an LLM, and (3) online feedback from earlier queries. Based on these signals, we design and compare four scheduling strategies. ASR Sort prioritizes attacks according to historical ASR priors. LLM-guided Sort augments these priors with semantic guidance. Thompson Sampling is initialized with historical priors and updates attack priorities using online feedback, while LLMguided Online combines all three signals.

Across three victim models and three representative defenses on GPT-5.2, our experiments clearly characterize the conditions under which different scheduling signals are most effective. Historical ASR priors provide an effective global initialization; semantic guidance improves or matches prioronly scheduling in the cold-start regime; and online feedback becomes particularly valuable when defense-induced distribution shifts make historical rankings stale. Semantic guidance and online feedback also partially compensate for missing historical priors. For LLM-based policies, cross query caching mitigates selector overhead. These results suggest that effective budgeted red-teaming is not about identifying a universally strongest attack, but about combining complementary signals to determine how attacks should be prioritized under the current evaluation condition.

Our main contributions are summarized as follows:

• We introduce budgeted attack scheduling, formalizing red-teaming under constrained attack budgets as a resource-aware scheduling problem, and provide its first systematic study.

• We establish a controlled decomposition of attack scheduling into three complementary information sources—historical priors, semantic guidance, and online feedback—to examine the contribution of each source to attack prioritization over a fixed attack library.

• We characterize the complementary roles of these signals: historical priors provide strong initialization, semantic guidance improves or matches prior-only scheduling at cold start and online feedback becomes particularly valuable when historical rankings become stale.

## 2 RELATED WORK

Jailbreak Attacks, Benchmarks, and Frameworks. LLM red-teaming spans red-team data collection (Ganguli et al., 2022), optimization-based suffixes (Zou et al., 2023), black-box refinement (Chao et al., 2023; Mehrotra et al., 2024), and automatic prompt construction (Liu et al., 2024a; Sun et al., 2025). Benchmarks such as HarmBench and JailbreakBench standardize harmful behaviors, judges, and reporting conventions (Mazeika et al., 2024; Chao et al., 2024), while Jailbreak Foundry translates jailbreak papers into 30 runnable attacks inside a unified evaluation framework (Fang et al., 2026). We use such a fixed reproduced attack library as input and study a budgeted attack scheduling question: when evaluators operate under a limited attack-attempt budget while accounting for overall cost, which attack should be prioritized next?

Defenses as Evaluation Conditions. Jailbreak defenses shape the evaluation condition under which a fixed attack library is tested. Existing approaches span several paradigms, including input rewriting and intent reconstruction (Wang et al., 2024; Luo et al., 2025; Zhang et al., 2025; Wang et al., 2025), perturbation- and consistency-based verification (Robey et al., 2025; Ji et al., 2025), detector-based safeguards (Inan et al., 2023; Zeng et al., 2024), and input sanitization (Geng et al., 2025). Our goal is not to introduce a new defense or directly compare defense effectiveness. Instead, we treat these defenses as part of the victim–defense evaluation condition, under which the same attack library may exhibit substantially different success patterns, attempt requirements, and cost profiles.

Budgeted Selection and Online Decision-Making. Budgeted testing and bandit-style decisionmaking (Tran-Thanh et al., 2010; Villar et al., 2015; Cavenaghi et al., 2021) study how to allocate limited trials under sequential feedback, with Thompson Sampling serving as a widely used Bayesian heuristic (Chapelle & Li, 2011; Russo et al., 2018). Related LLM systems also frame selection as a cost–quality tradeoff: routing and cascading methods choose among models or call based on query characteristics and expected utility (Chen et al., 2023; Hu et al., 2024; Ong et al., 2024), while LLM rankers use language models to order candidates conditioned on a query (Qin et al., 2024). In automated red-teaming, recent agentic systems select attack strategies using memory or feedback (Zhou et al., 2025). Our online protocol is conceptually closest to prequential evaluation (Dawid, 1984): a scheduler uses outcomes from earlier held-out queries in the same evaluation cell to prioritize later attacks. Unlike LLM routing, which selects among models or calls, and agentic red-teaming, which may generate or combine new attacks, we study how to schedule a fixed jailbreak attack library under an attempt budget while accounting for both attack-execution cost and scheduler overhead.

## 3 BUDGETED ATTACK SCHEDULING

We model red-teaming with a fixed attack library as an ordering problem under a limited attempt budget. This section defines the scheduling objective, the feedback available to cold-start and prequential online policies, and the four attack scheduling policies evaluated in the main experiments.

## 3.1 PROBLEM FORMULATION

Let $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { N } \}$ denote the fixed attack library. For a harmful query $q ,$ victim model $v ,$ defense condition $\mathcal { D } .$ , and historical ASR prior $\mathbf { p } = \left( p _ { 1 } , \ldots , p _ { N } \right)$ , where $p _ { i }$ is the historical success rate of attack $a _ { i }$ , a scheduler π outputs an ordered attack sequence

$$
\pi ( \boldsymbol { A } , \boldsymbol { q } , \mathbf { p } , \boldsymbol { v } , \mathcal { D } , \boldsymbol { H } _ { t - 1 } ) = ( a _ { i _ { 1 } } , a _ { i _ { 2 } } , \dots , a _ { i _ { N } } ) ,
$$

where $( i _ { 1 } , \dots , i _ { N } )$ is a permutation of the attack-library indices, and $H _ { t - 1 }$ denotes the historical feedback observable before the current scheduling decision in an online protocol. For each attempted attack, $s ( q , a , v , D ) \in \{ 0 , 1 \}$ indicates whether the attack succeeds.

Our primary optimization objective is

$$
\operatorname* { m a x } _ { \pi } \operatorname* { P r } [ \exists k \leq K : s ( q , a _ { i _ { k } } , v , \mathcal { D } ) = 1 ] ,
$$

Table 1: Attack scheduling policies considered in the main evaluation.
<table><tr><td>Policy</td><td>Historical ASR Prior</td><td>Semantic Guidance</td><td>Online Feedback</td><td>Scheduler Overhead</td></tr><tr><td>ASR Sort</td><td></td><td>X</td><td>X</td><td>None</td></tr><tr><td>LLM-guided Sort</td><td></td><td>√</td><td>X</td><td>LLM calls</td></tr><tr><td>Thompson Sampling</td><td></td><td>X</td><td>√</td><td>None</td></tr><tr><td>LLM-guided Online</td><td></td><td></td><td>1</td><td>LLM calls</td></tr></table>

where K denotes the attempt budget. We separately report attack-execution cost and scheduler overhead as evaluation outcomes. This allows us to assess whether gains in Success@K justify the additional scheduling cost.

For prequential online policies, performance additionally depends on the query order. Let $\sigma =$ $( q _ { 1 } , \dots , q _ { M } )$ denote a permutation of held-out queries under a fixed victim–defense condition. Online schedulers use historical priors and outcomes from earlier held-out queries in the same condition. For each current query, the attack order is fixed before execution; outcomes observed on that query are used only to update online state for subsequent queries. The scheduler does not condition on future outcomes, final aggregate rates, or outcomes from other cells. We write $S _ { K } ( \pi ; v , \mathcal { D } , \sigma )$ for Success@K under order σ. Online policy performance is summarized over fixed query-order permutations,

$$
\mathbb { E } _ { \sigma } [ S _ { K } ( \pi ; v , \mathcal { D } , \sigma ) ] ,
$$

and the same permutation set is used across online schedulers when assessing order sensitivity.

## 3.2 ATTACK SCHEDULING POLICIES

We consider three complementary scheduling signals: historical ASR priors, which provide a lowcost estimate of attack effectiveness from past evaluations; semantic guidance, which captures the compatibility between the current harmful query and different attack mechanisms; and online feedback, which updates attack priorities based on previously observed outcomes under the same victim–defense condition. To examine the contribution of these signals, we design four scheduling policies that incrementally incorporate semantic and online information on top of the historical prior, as summarized in Table 1. ASR Sort serves as the prior-only baseline; LLM-guided Sort augments the prior with semantic guidance; Thompson Sampling augments it with online feedback; and LLMguided Online combines all three signals.

Historical ASR Priors. Historical ASR priors p are computed on AdvBench50, a 50-sample subset of AdvBench used in prior work (Doumbouya et al., 2025; Zou et al., 2023). These priors are used only for scheduler initialization or static ranking, and are not used for tuning or evaluation on the final test set. Final scheduling evaluations are conducted on the full JBB-Behaviors benchmark, with no query overlap between evaluation and prior development.

ASR Sort. ASR Sort ranks attacks based on historical ASR priors p, prioritizing attacks with higher past success. It is static and query-agnostic, requiring no LLM selector call, making it a simple baseline for scenarios with limited resources.

LLM-guided Sort. LLM-guided Sort produces a query-aware ranking over the fixed attack library by incorporating historical attack priors together with semantic information about the query, attack mechanisms, and victim model. Each attack $a _ { i } ~ \in ~ { \mathcal { A } }$ has a concise mechanism description $d _ { i } ^ { a }$ and we collect these descriptions as $\mathcal { M } _ { A } = \{ d _ { 1 } ^ { a } , \ldots , d _ { N } ^ { a } \}$ . The victim model is represented by a description $d ^ { v }$ . We use GPT-4o (OpenAI et al., 2024) to summarize attack mechanisms from their implementations and victim-model characteristics from the corresponding model cards.

Given a query q, we construct a prompt $\mathcal { P } ( q , \mathbf { p } , d ^ { v } , \mathcal { M } _ { A } , \mathcal { Z } )$ , where $\mathcal { T }$ denotes a fixed ranking instruction. An LLM-based ranking operator then produces an ordering over the attack library:

$$
\pi _ { \mathrm { L L M } } = \mathrm { R a n k } _ { \mathrm { L L M } } \left( \mathcal { P } ( q , \mathbf { p } , d ^ { v } , \mathcal { M } _ { A } , \mathcal { T } ) \right) = ( i _ { 1 } , i _ { 2 } , \ldots , i _ { N } ) ,
$$

where $( i _ { 1 } , i _ { 2 } , \dots , i _ { N } )$ is a permutation of $\{ 1 , \ldots , N \}$ , yielding the attack order $( a _ { i _ { 1 } } , a _ { i _ { 2 } } , \ldots , a _ { i _ { N } } )$

Specifically, I instructs the scheduler to jointly consider three factors: historical attack effectiveness, the match between an attack mechanism and the current query, and its suitability for the victim model. As a prompt-level heuristic, $\mathcal { T }$ also encourages diversity among the highest-ranked attacks so that a limited budget covers different attack families. It also restricts the selector to the candidate set $\mathcal { A }$ and requires every attack to appear exactly once. The complete prompt is provided in Appendix A.

Unlike ASR Sort, LLM-guided Sort adapts the ranking to the current query $q$ and victim-model description $d ^ { v }$ , at the cost of an additional LLM selector call. The static context, including $\mathcal { M } _ { A }$ $d ^ { v }$ , and $\mathcal { T } ,$ can be cached across queries. We use GPT-4o as the LLM scheduler and report results averaged over five runs to reduce LLM-induced stochasticity.

Thompson Sampling. Thompson Sampling is a Beta–Bernoulli online policy initialized using historical priors. For each attack a, we maintain a Beta distribution parameterized by $( \alpha _ { a } , \beta _ { a } )$ Given the historical ASR prior $p _ { a } .$ , prior strength $\kappa ,$ and smoothing constant $\epsilon ,$ the parameters are initialized as

$$
\alpha _ { a } = \kappa p _ { a } + \epsilon , \qquad \beta _ { a } = \kappa ( 1 - p _ { a } ) + \epsilon .
$$

Before executing each query, the policy samples

$$
\theta _ { a } \sim \mathrm { B e t a } ( \alpha _ { a } , \beta _ { a } )
$$

once for every attack $a \in \mathcal A$ and constructs a complete attack ordering by sorting the sampled values in descending order. This ordering is fixed for the current query and is executed until the first successful jailbreak or the attempt budget K is exhausted.

After the current query is completed, the policy updates the posterior parameters of each attempted attack using its observed binary outcome $y _ { a } \in \{ 0 , 1 \}$ :

$$
\alpha _ { a }  \alpha _ { a } + y _ { a } , \qquad \beta _ { a }  \beta _ { a } + ( 1 - y _ { a } ) .
$$

Unattempted attacks remain unchanged. In our experiments, we set $\epsilon = 1$ and $\kappa = 5$

LLM-guided Online. LLM-guided Online extends the query-aware ranking of LLM-guided Sort with online feedback accumulated from previously processed queries. For each query $q ,$ we invoke the LLM selector once to obtain an initial ranking $\pi _ { \mathrm { L L M } } = ( i _ { 1 } , i _ { 2 } , . . . , i _ { N } )$ following the same procedure as in LLM-guided Sort. We convert this ordering into a rank score $r _ { q } ( a )$ , with larger scores for higher-ranked attacks. Specifically, for the attack ranked at position $j ,$ we define

$$
r _ { q } ( a _ { i _ { j } } ) = N - j + 1 .
$$

To incorporate online feedback, we maintain $N _ { \mathrm { s u c c } } ( a )$ and $N _ { \mathrm { f a i l } } ( a )$ , denoting the cumulative numbers of successful and failed applications of attack a on previously processed queries, respectively. We define the scheduling score as

$$
S ( a \mid q ) = r _ { q } ( a ) + \lambda \log \frac { 1 + N _ { \mathrm { s u c c } } ( a ) } { 1 + N _ { \mathrm { f a i l } } ( a ) } ,
$$

where λ controls the contribution of online feedback. This term increases the priority of attacks with accumulated successes while penalizing those with repeated failures. In our experiments, we set $\lambda = 1$

Before executing each query, we rank all attacks by $S ( a \mid q )$ and fix this order for the current query. After the query is completed, we update $N _ { \mathrm { s u c c } } ( a )$ and $N _ { \mathrm { f a i l } } ( a )$ for attempted attacks using their observed outcomes. The updated statistics are used only for subsequent queries.

## 3.3 DEFENSE CONDITIONS

We treat defenses as part of the evaluation conditions rather than as methods introduced in this work. Our taxonomy covers four settings: no defense, intent reconstruction, consistency-based defense, and detector-based defense. Section 4.1 specifies the representative defenses used in the main evaluation, while Appendix A.2 provides mechanism-level details.

## 4 EXPERIMENTAL SETUP

We instantiate the formulation in Section 3 as a controlled benchmark over a frozen attack library, held-out harmful queries, and a victim–defense matrix. This design allows us to compare scheduling effectiveness and cost within each evaluation condition.

## 4.1 BENCHMARK MATRIX

Attack Library. We use a frozen library of 30 reproduced attack instances from Jailbreak Foundry (Fang et al., 2026). We choose this library because Jailbreak Foundry validates the reproduced attacks under paper-matched settings and annotates them across complementary search and carrier mechanisms, giving the scheduler a heterogeneous but fixed attack set rather than a hand-picked collection of prompts. The library is fixed across schedulers, victim models, defenses, and queryorder permutations.

Victim–Defense Matrix. We stratify victim models according to their ASR on AdvBench50 and select DeepSeek-V4-Pro (preview) (DeepSeek-AI et al., 2026), Gemma-4-26B (Team et al., 2026), and GPT-5.2 (Singh et al., 2026) to represent low-, medium-, and high-ASR regimes, respectively. Budgeted vulnerability is then evaluated independently on held-out queries from JailbreakBench (Chao et al., 2024). We report results under the native setting for all three victim models and provide a detailed guardrail breakdown for GPT-5.2.

The defense dimension follows the categories defined in Section 3.3. We include a no-guardrail baseline together with three representative defense mechanisms: Backtranslation (Wang et al., 2024), SmoothLLM (Robey et al., 2025), and LlamaGuard (Inan et al., 2023).

## 4.2 METRICS AND EVALUATION PROTOCOL

Metrics. We evaluate each method in terms of attack success, attempt efficiency, and token cost. Success@K denotes the fraction of held-out episodes in which a successful jailbreak is achieved within K attempts; we use K = 5 in all main tables and compute Success@1/3 from the prefixes of the same trajectories. Average total attempts (ATA) measures the overall per-episode attempt cost under the fixed budget: a successful episode contributes the number of attempts until its first success, whereas a failed episode contributes the full budget K.

For token-related metrics, we report mean per-query consumption separately for attack execution and scheduler overhead. We also report the scheduler’s input-token cache hit rate.

Evaluation Protocol. Following the information boundary in Section 3, each held-out query q, victim v, and defense D is executed until first success or budget exhaustion. Thompson Sampling and LLM-guided Online are evaluated with five fixed matched query-order permutations (P1–P5) per completed victim–guardrail cell, resetting online state for each cell and permutation. Maintable online rows report means over these matched permutations, and Section 6 visualizes the perpermutation spread to assess robustness to query order.

## 5 MAIN RESULTS

Semantic guidance improves cold-start scheduling. Table 2 shows that historical ASR provides a strong and low-cost baseline: ASR Sort is already close to saturation on DeepSeek-V4-Pro and Gemma-4-26B, while leaving more room for improvement on the more challenging GPT-5.2. Semantic guidance is particularly useful when historical ASR provides an imperfect initial ranking. Before any online feedback is observed, LLM-guided Sort consistently matches or outperforms ASR Sort across victim models. On GPT-5.2, Success@1, Success@3, and Success@5 improve from 36.00%, 69.00%, and 77.00% to 39.00%, 74.00%, and 79.00%, respectively. A similar pattern emerges under defenses in Table 3: compared with ASR Sort, LLM-guided Sort improves Success@1 by 19.00 and 2.00 percentage points (pp), and Success@5 by 5.00 and 9.00 pp under Backtranslation and LlamaGuard, respectively. These results indicate that attack descriptions provide information complementary to aggregate historical success rates, enabling the scheduler to prioritize semantically plausible attack mechanisms even before receiving feedback from the current victim–defense condition.

Table 2: Attack performance and token usage across scheduling policies and victim models.
<table><tr><td rowspan="2">Victim Models</td><td rowspan="2">Scheduling Policies</td><td colspan="4">Success Metrics</td><td colspan="3">Token Usage</td></tr><tr><td>Success@1 Success@3 Success@5Avg. ATA</td><td></td><td></td><td></td><td>Attack Schedule Cache Hit Rate</td><td></td><td></td></tr><tr><td rowspan="4">DeepSeek-V4-Pro</td><td>ASR Sort</td><td>92.00</td><td>98.00</td><td>99.00</td><td>1.14</td><td>1366</td><td></td><td></td></tr><tr><td>LLM-guided Sort</td><td>92.00</td><td>99.00</td><td>99.00</td><td>1.12</td><td>1241</td><td>11242</td><td>89.13</td></tr><tr><td>Thompson Sampling</td><td>86.20</td><td>97.60</td><td>99.00</td><td>1.22</td><td>813</td><td></td><td></td></tr><tr><td>LLM-guided Online</td><td>92.00</td><td>99.00</td><td>99.00</td><td>1.14</td><td>1446</td><td>11242</td><td>89.13</td></tr><tr><td rowspan="4">Gemma-4-26B</td><td>ASR Sort</td><td>59.00</td><td>84.00</td><td>90.00</td><td>1.94</td><td>7979</td><td></td><td></td></tr><tr><td>LLM-guided Sort</td><td>59.00</td><td>85.00</td><td>90.00</td><td>1.93</td><td>8632</td><td>12229</td><td>90.93</td></tr><tr><td>Thompson Sampling</td><td>49.80</td><td>79.60</td><td>87.00</td><td>2.17</td><td>7210</td><td></td><td></td></tr><tr><td>LLM-guided Online</td><td>55.00</td><td>83.60</td><td>90.00</td><td>1.97</td><td>8175</td><td>12229</td><td>90.93</td></tr><tr><td rowspan="4">GPT-5.2</td><td>ASR Sort</td><td>36.00</td><td>69.00</td><td>77.00</td><td>2.60</td><td>3535</td><td></td><td></td></tr><tr><td>LLM-guided Sort</td><td>39.00</td><td>74.00</td><td>79.00</td><td>2.50</td><td>3357</td><td>11885</td><td>89.72</td></tr><tr><td>Thompson Sampling</td><td>43.20</td><td>69.20</td><td>77.60</td><td>2.53</td><td>2426</td><td></td><td></td></tr><tr><td>LLM-guided Online</td><td>44.00</td><td>73.80</td><td>79.80</td><td>2.40</td><td>2446</td><td>11885</td><td>89.72</td></tr></table>

Table 3: Attack performance and token usage across defense conditions on GPT-5.2.
<table><tr><td rowspan="2">Defense</td><td rowspan="2">Scheduling Policies</td><td colspan="4">Success Metrics</td><td colspan="3">Token Usage</td></tr><tr><td>Success@1 Success@3 Success@5 Avg. ATA</td><td></td><td></td><td></td><td></td><td>Attack Schedule Cache Hit Rate</td><td></td></tr><tr><td rowspan="4">Native</td><td>ASR Sort</td><td>36.00</td><td>69.00</td><td>77.00</td><td>2.60</td><td>3535</td><td></td><td></td></tr><tr><td>LLM-guided Sort</td><td>39.00</td><td>74.00</td><td>79.00</td><td>2.50</td><td>3357</td><td>11885</td><td>89.72</td></tr><tr><td>Thompson Sampling</td><td>43.20</td><td>69.20</td><td>77.60</td><td>2.53</td><td>2426</td><td></td><td></td></tr><tr><td>LLM-guided Online</td><td>44.00</td><td>73.80</td><td>79.80</td><td>2.40</td><td>2446</td><td>11885</td><td>89.72</td></tr><tr><td rowspan="4">Backtranslation</td><td>ASR Sort</td><td>12.00</td><td>50.00</td><td>62.00</td><td>3.39</td><td>4290</td><td></td><td></td></tr><tr><td>LLM-guided Sort</td><td>31.00</td><td>66.00</td><td>67.00</td><td>2.86</td><td>5603</td><td>9338</td><td>88.61</td></tr><tr><td>Thompson Sampling</td><td>25.40</td><td>51.60</td><td>62.20</td><td>3.21</td><td>4341</td><td></td><td></td></tr><tr><td>LLM-guided Online</td><td>39.00</td><td>66.00</td><td>67.00</td><td>2.71</td><td>2715</td><td>9338</td><td>88.61</td></tr><tr><td rowspan="4">SmoothLLM</td><td>ASR Sort</td><td>27.00</td><td>41.00</td><td>51.00</td><td>3.57</td><td>4563</td><td></td><td></td></tr><tr><td>LLM-guided Sort</td><td>27.00</td><td>45.00</td><td>54.00</td><td>3.41</td><td>5048</td><td>11121</td><td>89.40</td></tr><tr><td>Thompson Sampling</td><td>26.20</td><td>44.20</td><td>54.80</td><td>3.42</td><td>3987</td><td></td><td></td></tr><tr><td>LLM-guided Online</td><td>27.60</td><td>51.00</td><td>55.00</td><td>3.25</td><td>5001</td><td>11121</td><td>89.40</td></tr><tr><td rowspan="4">LlamaGuard</td><td>ASR Sort</td><td>3.00</td><td>19.00</td><td>19.00</td><td>4.55</td><td>13321</td><td></td><td></td></tr><tr><td>LLM-guided Sort</td><td>5.00</td><td>24.00</td><td>28.00</td><td>4.28</td><td>20233</td><td>11817</td><td>89.77</td></tr><tr><td>Thompson Sampling</td><td>5.40</td><td>17.60</td><td>25.40</td><td>4.43</td><td>11557</td><td></td><td></td></tr><tr><td>LLM-guided Online</td><td>14.00</td><td>33.20</td><td>34.60</td><td>3.89</td><td>10731</td><td>11817</td><td>89.77</td></tr></table>

Online feedback revises stale rankings under defense-induced distribution shifts. Table 3 and Figure 2 jointly reveal why static historical rankings become less reliable under defenses. We quantify the discrepancy between the Prior Pool and Test Pool using two complementary measures. The Kolmogorov–Smirnov D statistic (KS D) captures the maximum vertical distance between their empirical cumulative distribution function. The Wasserstein-1 distance (W ) measures the minimum average transport distance along the query-difficulty axis. From Native to LlamaGuard, KS D increases from 0.08 to 0.63 and W from 0.012 to 0.083, indicating increasingly severe distribution shift. This shift is also reflected in attack ordering: under LlamaGuard, jailcon cvt drops from 1st in the historical ranking to outside the top 10, while air rises from 7th to 1st. Consistently, the Success@5 of static ASR Sort falls from 77.00% to 19.00%.

Online feedback provides a mechanism for correcting this stale ranking using outcomes from the current condition. Under Backtranslation, Thompson Sampling improves Success@1 from 12.00% to 25.40% over ASR Sort, while LLM-guided Online raises Success@1 from 31.00% to 39.00% over LLM-guided Sort. Under the more severe LlamaGuard shift, LLM-guided Online improves Success@1 from 5.00% to 14.00% and Success@5 from 28.00% to 34.60% over LLM-guided Sort, while Thompson Sampling improves Success@5 to 25.40% over ASR Sort. Collectively, these results suggest that online feedback does not eliminate distribution shift, but provides an adaptive mechanism for revising attack priorities when historical effectiveness no longer transfers reliably.

Caching mitigates selector overhead, and better prioritization reduces attack-side cost. LLMbased selection introduces additional scheduling overhead, but our prompt design supports crossquery caching by placing static context before the current query: Tables 2 and 3 show that 88.61% to 90.93% of selector input tokens are served from cache. More importantly, better attack prioritization can reduce execution cost even as it improves success. Combining semantic guidance with online feedback raises early success rates, converting failed attempts into early terminations. For example, on GPT-5.2, Success@1 rises from 36.00% to 44.00% under the native setting and from 12.00% to 39.00% under Backtranslation, while attack-side token usage drops from 3,535 to 2,446 and from 4,290 to 2,715, respectively. Overall, caching mitigates selector overhead, while better prioritization delivers both higher success and lower attack-side cost.

![](images/fb375a527246055f78408fc38e79c980e80553e48bb3fa723a1ba2db61ffce10.jpg)  
Figure 2: Defense-induced shifts in query distribution and attack ordering. Top: Query diffi culty distributions, where difficulty is defined as one minus the attack success rate across all attacks; higher values indicate harder queries. Bottom: Attack-level changes in effectiveness, with attacks ordered from top to bottom by their ASR under the current test condition.

Table 4: Effect of historical priors on attack scheduling under the no-defense setting. Success@1/3/5 compares each scheduling policy with and without access to historical attack-success priors. ∆ denotes the corresponding improvement in pp, and Avg. ∆ reports the mean gain across the three attempt budgets.
<table><tr><td rowspan="2">Victim Models</td><td rowspan="2">Scheduling Policies</td><td colspan="3">With Historical Prior</td><td colspan="3">Without Historical Prior</td><td colspan="4">Prior Gain (pp)</td></tr><tr><td>Success@1</td><td>Success@3</td><td>Success@5</td><td>Success@1</td><td>Success@3</td><td>Success@5</td><td>∆@1</td><td>∆@3</td><td>∆@5</td><td>Avg. ∆</td></tr><tr><td rowspan="4">DeepSeek-V4-Pro</td><td>ASR Sort</td><td>92.00</td><td>98.00</td><td>99.00</td><td>15.00</td><td>72.00</td><td>83.00</td><td>+77.00</td><td>+26.00</td><td>+16.00</td><td>+39.67</td></tr><tr><td>LLM-guided Sort</td><td>92.00</td><td>99.00</td><td>99.00</td><td>47.00</td><td>93.00</td><td>96.00</td><td>+45.00</td><td>+6.00</td><td>+3.00</td><td>+18.00</td></tr><tr><td>Thompson Sampling</td><td>86.20</td><td>97.60</td><td>99.00</td><td>78.00</td><td>93.00</td><td>96.00</td><td>+8.20</td><td>+4.60</td><td>+3.00</td><td>+5.27</td></tr><tr><td>LLM-guided Online</td><td>92.00</td><td>99.00</td><td>99.00</td><td>47.00</td><td>93.00</td><td>97.00</td><td>+45.00</td><td>+6.00</td><td>+2.00</td><td>+17.67</td></tr><tr><td rowspan="4">Gemma-4-26B</td><td>ASR Sort</td><td>59.00</td><td>84.00</td><td>90.00</td><td>18.00</td><td>62.00</td><td>74.00</td><td>+41.00</td><td>+22.00</td><td>+16.00</td><td>+26.33</td></tr><tr><td>LLM-guided Sort</td><td>59.00</td><td>85.00</td><td>90.00</td><td>34.00</td><td>70.00</td><td>82.00</td><td>+25.00</td><td>+15.00</td><td>+8.00</td><td>+16.00</td></tr><tr><td>Thompson Sampling</td><td>49.80</td><td>79.60</td><td>87.00</td><td>47.00</td><td>76.00</td><td>80.00</td><td>+2.80</td><td>+3.60</td><td>+7.00</td><td>+4.47</td></tr><tr><td>LLM-guided Online</td><td>55.00</td><td>83.60</td><td>90.00</td><td>50.00</td><td>76.00</td><td>82.00</td><td>+5.00</td><td>+7.60</td><td>+8.00</td><td>+6.87</td></tr><tr><td rowspan="4">GPT-5.2</td><td>ASR Sort</td><td>36.00</td><td>69.00</td><td>77.00</td><td>5.00</td><td>10.00</td><td>19.00</td><td>+31.00</td><td>+59.00</td><td>+58.00</td><td>+49.33</td></tr><tr><td>LLM-guided Sort</td><td>39.00</td><td>74.00</td><td>79.00</td><td>20.00</td><td>33.00</td><td>55.00</td><td></td><td>+19.00+41.00+24.00</td><td></td><td>+28.00</td></tr><tr><td>Thompson Sampling</td><td>43.20</td><td>69.20</td><td>77.60</td><td>40.00</td><td>59.00</td><td>64.00</td><td>+3.20</td><td>+10.20+13.60</td><td></td><td>+9.00</td></tr><tr><td>LLM-guided Online</td><td>44.00</td><td>73.80</td><td>79.80</td><td>32.00</td><td>56.00</td><td>61.00</td><td>+12.00+17.80+18.80</td><td></td><td></td><td>+16.20</td></tr></table>

## 6 ANALYSIS: PRIOR-FREE SCHEDULING AND ONLINE ORDER SENSITIVITY

In practice, historical ASR priors are often unavailable for new attack libraries, victim models, or defense conditions. We therefore evaluate prior-free scheduling as a cold-start stress test of whether semantic guidance and online feedback can compensate for missing historical evidence. We also examine the order sensitivity of online adaptation across different query orders.

Under the prior-free setting, we retain each policy’s decision signals except historical ASR. ASR Sort degenerates to random ordering. Thompson Sampling is initialized with a uniform prior and learns only from online outcomes. For the two LLM-driven schedulers, we omit the historical ASR entries from the prompt while preserving attack descriptions and, where applicable, online feedback.

Order Sensitivity of Online Feedback

![](images/caf7551d54c40def2886c30e0fdc3aaf29f34d6bc5ed790df27063a6b41ec4b6.jpg)  
Figure 3: Order sensitivity of online feedback across native and defended settings. Success@K is evaluated over five query orders for Thompson Sampling and LLM-guided Online. Hollow markers show individual orders, solid markers show the mean, and vertical stems indicate the min–max range; the inset summarizes Success@5 variation across orders.

Prior-free scheduling exposes the limits of static ASR ranking. Table 4 shows that removing historical priors substantially degrades static ASR-based scheduling across all three victim models. ASR Sort exhibits average prior gains of 39.67, 26.33, and 49.33 pp on DeepSeek-V4-Pro, Gemma-4-26B, and GPT-5.2, respectively, indicating a strong dependence on historical ASR estimates. The degradation is concentrated in the budget-limited regime, particularly on models already close to saturation under historical priors: Success@1 drops from 92.00% to 15.00% on DeepSeek-V4-Pro, from 59.00% to 18.00% on Gemma-4-26B, and from 36.00% to only 5.00% on GPT-5.2. This demonstrates the value of historical ASR for providing an effective initial ordering; without such information, a query-agnostic static ranking retains no signal for prioritizing promising attacks.

Semantic and online signals mitigate prior-free cold starts. When historical priors are unavailable, semantic guidance and online feedback provide alternative signals for attack prioritization. On DeepSeek-V4-Pro, Thompson Sampling achieves 78.00% Success@1 and 93.00% Success@3 without priors, compared with only 15.00% and 72.00% for ASR Sort, while LLM-guided Online reaches 97.00% Success@5. Similar recovery is observed on Gemma-4-26B, where LLM-guided Online improves prior-free Success@1 from 18.00% to 50.00%, and on GPT-5.2, where it raises Success@1 from 5.00% to 32.00%. LLM-guided Sort also improves Success@5 from 19.00% to 55.00% on GPT-5.2. These results show that semantic guidance and interaction feedback can partially compensate for missing historical information, reducing reliance on a static historical ranking.

Order sensitivity of online feedback. Online schedulers are inherently path-dependent, as feedback from earlier queries updates the scheduler state and affects subsequent attack selection. Figure 3 evaluates this sensitivity across different query orders, with the variation at Success@5 highlighted in each condition. A clear pattern emerges: LLM-guided Online is consistently more stable than Thompson Sampling at larger attempt budgets. Across all six settings, its Success@5 variation remains within 0–2.0 pp, compared with 2.0–8.0 pp for Thompson Sampling. The difference is particularly pronounced under Gemma-4-26B (0.0 vs. 5.0 pp), Backtranslation (2.0 vs. 6.0 pp), SmoothLLM (0.0 vs. 8.0 pp), and LlamaGuard (1.0 vs. 7.0 pp). Although larger variations can still appear at Success@1 when only limited feedback is available, they generally diminish as the query budget increases. Overall, these results suggest that semantic guidance makes online scheduling substantially less sensitive to query order once sufficient feedback has accumulated.

## 7 CONCLUSION

We study budgeted attack scheduling for LLM red-teaming, with the goal of supporting efficient robustness evaluation using a fixed attack library under a limited attempt budget. Our results re veal distinct and complementary roles for the three scheduling signals: historical ASR priors provide strong initialization, semantic guidance captures query-specific attack compatibility, and online feedback revises stale rankings under victim–defense distribution shifts. Together, these findings recast red-teaming from exhaustive attack enumeration into an adaptive resource-allocation problem, where effective evaluation depends not only on which attacks are available, but also on how they are prioritized under different conditions.

## AI USE STATEMENT

Generative AI tools were consulted for writing suggestions during the preparation of this manuscript. The authors evaluated these suggestions and manually reviewed, revised, and proofread the text. The use of GPT-4o to generate descriptive inputs for the experiments is documented in Section 3.2 and Appendix A. The authors take responsibility for the final manuscript, including AI-assisted content.

## ETHICAL CONSIDERATIONS

This work studies how to prioritize existing jailbreak attacks for evaluating LLM safety under limited budgets. It uses a fixed attack library and held-out benchmark queries and does not introduce new attack-generation procedures. More efficient scheduling can support robustness assessment, but it may also reduce the cost of misuse. The methods should therefore be used for authorized safety evaluation, with appropriate handling of harmful model outputs. Any release of supporting artifacts should balance reproducibility with the risk of enabling misuse.

## REPRODUCIBILITY STATEMENT

Section 3 specifies the scheduling problem, the four policies, and the hyperparameters of the online policies. Section 4 describes the attack library, evaluation conditions, metrics, and matched query order protocol. Appendix A provides the prompt construction procedure, ranking instructions, and prompt template, while Appendix A.2 explains the defense conditions.

## REFERENCES

Emanuele Cavenaghi, Gabriele Sottocornola, Fabio Stella, and Markus Zanker. Non stationary multi-armed bandit: Empirical evaluation of a new concept drift-aware algorithm. Entropy, 23 (3), 2021. ISSN 1099-4300. URL https://www.mdpi.com/1099-4300/23/3/380.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. In NeurIPS Workshop on Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, and Eric Wong. JailbreakBench: An open robustness benchmark for jailbreaking large language models. In NeurIPS Datasets and Benchmarks Track, 2024.

Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. In Advances in Neural Information Processing Systems, volume 24, 2011.

Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while reducing cost and improving performance, 2023. URL https://arxiv.org/abs/2305. 05176.

A. Philip Dawid. Present position and potential developments: Some personal views: Statistical theory: The prequential approach. Journal of the Royal Statistical Society. Series A (General), 147(2):278–292, 1984.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, et al. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026. URL https://arxiv.org/abs/ 2606.19348.

Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, and Christopher D Manning. h4rm3l: A language for composable jailbreak attack synthesis. In The Thirteenth International Conference on Learning Representations, 2025.

Zhicheng Fang, Jingjie Zheng, Chenxu Fu, and Wei Xu. Jailbreak Foundry: From papers to runnable attacks for reproducible benchmarking. In International Conference on Machine Learning (ICML), 2026. URL https://openreview.net/forum?id=BSi2mfMDsx. Proceedings forthcoming.

Deep Ganguli et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022.

Runpeng Geng, Yanting Wang, Chenlong Yin, Minhao Cheng, Ying Chen, and Jinyuan Jia. Pisanitizer: Preventing prompt injection to long-context llms via prompt sanitization, 2025. URL https://arxiv.org/abs/2511.10720.

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-llm routing system, 2024. URL https://arxiv.org/abs/2403.12031.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. Llama guard: Llmbased input-output safeguard for human-ai conversations, 2023. URL https://arxiv.org/ abs/2312.06674.

Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, and Shiyu Chang. Defending large language models against jailbreak attacks via semantic smoothing. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 7–40, 2025.

Yukun Jiang, Mingjie Li, Michael Backes, and Yang Zhang. Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2025.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations, 2024a.

Xuannan Liu, Xing Cui, Peipei Li, Zekun Li, Huaibo Huang, Shuhan Xia, Miaoxuan Zhang, Yueying Zou, and Ran He. Jailbreak attacks and defenses against multimodal generative models: A survey, 2024b. URL https://arxiv.org/abs/2411.09259.

Weidi Luo, He Cao, Zijing Liu, Yu Wang, Aidan Wong, Bin Feng, Yuan Yao, and Yu Li. Dynamic guided and domain applicable safeguards for enhanced security in large language models. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pp. 6614–6635, 2025.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In International Conference on Machine Learning, 2024.

Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. Tree of attacks: Jailbreaking black-box llms automatically. In Ad vances in Neural Information Processing Systems, pp. 61065–61105, 2024.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms with preference data, 2024. URL https://arxiv.org/abs/2406.18665.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, et al. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.

Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, and Michael Bendersky. Large language models are effective text rankers with pairwise ranking prompting. In Proceedings ofthe 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics, 2024.

Alexander Robey, Eric Wong, Hamed Hassani, and George J. Pappas. SmoothLLM: Defending large language models against jailbreaking attacks. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=laPAh2hRFC.

Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. A tutorial on thompson sampling. Foundations and Trends in Machine Learning, 11(1):1–96, 2018.

Leyi Sheng, Han Sun, Zhen Sun, Yuntao Yue, Jinlin Wu, Xinlei He, and Jiaheng Wei. Pixjail: Self-evolving paper-to-pipeline reproduction for text-to-image jailbreak evaluation, 2026. URL https://arxiv.org/abs/2606.24081.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, et al. Openai gpt-5 system card, 2026. URL https://arxiv.org/abs/2601.03267.

Zhen Sun, Zongmin Zhang, Deqi Liang, Han Sun, Yule Liu, Yun Shen, Xiangshan Gao, Yilong Yang, Shuai Liu, Yutao Yue, and Xinlei He. “to survive, i must defect”: Jailbreaking llms via the game-theory scenarios, 2025. URL https://arxiv.org/abs/2511.16278.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, et al. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607.02770.

Long Tran-Thanh, Archie Chapman, Enrique Munoz de Cote, Alex Rogers, and Nicholas R. Jennings. Epsilon-first policies for budget-limited multi-armed bandits. Proceedings of the AAAI Conference on Artificial Intelligence, 24(1):1211–1216, 2010.

Sofia S. Villar, Jack Bowden, and James Wason. Multi-armed bandit models for the optimal design of clinical trials: Benefits and challenges. Statistical Science, 30(2), May 2015. ISSN 0883-4237. doi: 10.1214/14-sts504. URL http://dx.doi.org/10.1214/14-STS504.

Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and Juergen Rahmel. SelfDefend: LLMs can defend themselves against jailbreaking in a practical manner. In 34th USENIX Security Symposium (USENIX Security 25), pp. 2441–2460, 2025.

Yihan Wang, Zhouxing Shi, Andrew Bai, and Cho-Jui Hsieh. Defending LLMs against jailbreaking attacks via backtranslation. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 16031–16046, 2024.

Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, and Qi Li. Jailbreak attacks and defenses against large language models: A survey, 2024. URL https://arxiv. org/abs/2407.04295.

Wenjun Zeng, Yuchi Liu, Ryan Mullins, Ludovic Peran, Joe Fernandez, Hamza Harkous, Karthik Narasimhan, Drew Proud, Piyush Kumar, Bhaktipriya Radharapu, Olivia Sturman, and Oscar Wahltinez. Shieldgemma: Generative ai content moderation based on gemma, 2024. URL https://arxiv.org/abs/2407.21772.

Yuqi Zhang, Liang Ding, Lefei Zhang, and Dacheng Tao. Intention analysis makes LLMs a good jailbreak defender. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 2947–2968, 2025.

Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, and Bo Li. Autoredteamer: Autonomous red teaming with lifelong attack integration. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## A APPENDIX

## A.1 PROMPT COMPOSITION FOR LLM-GUIDED SORT

LLM-guided Sort constructs a query-aware ranking over the fixed attack library through a structured prompt that integrates both historical and semantic signals. Specifically, the prompt consists of four components: (1) the current query, (2) historical attack success rate (ASR) priors, (3) descriptions of the victim model and attack methods, and (4) ranking rules and instructions. Together, these components provide the scheduler with both empirical evidence about past attack effectiveness and semantic information required to assess attack–query compatibility under the current victim-model setting.

Let $A = \{ a _ { 1 } , \ldots , a _ { N } \}$ denote the fixed attack library and $\mathbf { p } = \left( p _ { 1 } , \ldots , p _ { N } \right)$ the corresponding historical ASR priors, where $p _ { i }$ denotes the historical success rate of attack $a _ { i }$ . Given a query q, we construct the scheduling prompt as

$$
\mathcal { P } ( q , \mathbf { p } , d ^ { v } , \mathcal { M } _ { A } , \mathcal { Z } ) ,
$$

where $d ^ { v }$ denotes the victim-model description, $\mathcal { M } _ { A }$ denotes the collection of attack-method de scriptions, and I denotes the ranking rules and instructions.

Descriptions of the Victim Model and Attack Methods. To provide informative semantic context for scheduling, we construct concise descriptions of both the victim model and individual attack methods from their original sources. The victim model is represented by a description $d ^ { v }$ , generated from its model card to summarize characteristics relevant to attack selection. For each attack $a _ { i } ,$ we provide its concrete code implementation to an LLM (GPT-4o) and generate a structured description $d _ { i } ^ { a }$ consisting of two components: (1) mechanism, which summarizes the key procedures and transformations performed by the attack, and (2) applicability hints, which infer from these procedures the types of queries or victim-model conditions under which the attack may be particularly effective. We denote the complete set of attack descriptions as

$$
\mathcal { M } _ { A } = \{ d _ { 1 } ^ { a } , . . . , d _ { N } ^ { a } \} .
$$

These descriptions provide the scheduler with explicit knowledge of both how an attack operates and when it is likely to be effective, enabling it to reason about the compatibility among the current query, the victim model, and different attack mechanisms rather than relying solely on aggregate historical ASR.

Description Example: JAIL-CON-CVT (Jiang et al., 2025). For JAIL-CON-CVT, the generated attack description is structured as follows:

Example Attack Description: JAIL-CON-CVT

[Attack Name] JAIL-CON-CVT

[Mechanism] The attack interleaves a harmful request and a benign request at the word level, placing harmful-request words outside delimiters and benign-request words inside them. The victim model is instructed to separate the interleaved sequence into two requests and answer both concurrently, while similarly interleaving the two response streams. If an attempt fails, a different benign request is randomly sampled and the combined prompt is reconstructed for another trial. (Summarizes how the attack operates)

[Applicability Hint] The attack is more effective on models that can reliably reconstruct interleaved semantic streams and follow multi-task instructions. It is particularly suitable for harmful requests with clear, self-contained intents that remain recoverable after word-level interleaving. Its effectiveness may decrease for requests requiring precise wording or long-range context, or under defenses that normalize or decompose the structured input. (Indicates when the attack is likely to be effective)

This example illustrates how the attack description captures both the concrete execution mechanism and the conditions under which the attack is expected to be more or less effective.

Ranking Rules and Instructions. We further provide a fixed instruction I that specifies how the scheduler should combine semantic and historical signals to produce a valid ranking over the fixed attack library. Rather than relying on a single criterion, I guides the scheduler to jointly consider three complementary factors: (1) query–attack compatibility, which captures how well an attack mechanism matches the semantic content, structure, and interaction pattern of the current query; (2) victim–attack compatibility, which reflects whether the attack mechanism is well suited to the behavioral and instruction-following characteristics of the victim model; and (3) historical attack effectiveness, represented by the historical ASR prior $p _ { i }$

Semantic compatibility serves as the primary ranking signal. The scheduler prioritizes attacks whose mechanisms and applicability conditions are well aligned with both the current query and the victim model, while historical ASR provides complementary empirical evidence about prior effectiveness. In particular, a high ASR can strengthen the ranking of a semantically plausible attack, but should not by itself dominate the ranking when the attack mechanism is poorly matched to the current setting. This design enables LLM-guided Sort to exploit historical experience while adapting attack selection to each individual query.

To improve attack coverage under a limited execution budget, I further introduces a diversity heuristic for the highest-ranked candidates. When multiple attacks exhibit similar estimated effectiveness, the scheduler is encouraged to prioritize mechanistically distinct attacks and avoid placing nearduplicate attack mechanisms at the top of the ranking. This increases the coverage of heterogeneous attack families within the first few trials and reduces redundant attempts.

Finally, I imposes explicit candidate-set and output constraints. The scheduler may only select from the predefined attack library A, must include every attack exactly once, and cannot introduce or modify attack methods. The resulting attacks are ordered from most to least promising and returned together with concise ranking rationales in a fixed JSON format, allowing the output to be directly parsed and executed by the downstream scheduler. The concrete ranking rules are summarized below.

## Ranking Rules for LLM-Guided Sort

[Rule 1: Query--Attack Compatibility] Treat the attack description as the primary selection signal. Prefer attacks whose mechanisms are well matched to the semantic content, structure, reasoning requirements, and interaction format of the current query.

[Rule 2: Victim--Attack Compatibility] Prefer attacks whose mechanisms and applicability conditions are well aligned with the behavioral characteristics of the victim model, including its instruction-following, reasoning, and structured-output capabilities.

[Rule 3: Historical Effectiveness] Use historical ASR as complementary empirical evidence. A high ASR may strengthen the ranking of a semantically plausible attack, but should not override poor compatibility with the current query or victim model.

[Rule 4: Top-Ranked Diversity] Prefer mechanistically diverse attacks among the highestranked candidates. When similarly promising alternatives are available, avoid placing near-duplicate attack mechanisms early in the ranking.

[Rule 5: Candidate-Set Constraint] Use only the provided attack IDs. Include every attack exactly once, without introducing, removing, duplicating, or modifying any candidate.

[Rule 6: Output Format] Return all attacks in descending order of estimated effectiveness, together with a concise rationale for each ranking decision, using the predefined JSON format.

Complete Prompt Example. A complete prompt combines the current query $q ,$ historical ASR priors p, the victim-model description $d ^ { \bar { v } }$ , attack-method descriptions $\mathcal { M } _ { A }$ , and the ranking instruction I into a unified input to the LLM scheduler. Since the construction of the descriptions and ranking rules has been detailed above, we omit their full contents here and illustrate only the overall prompt organization.

Complete Prompt for LLM-Guided Sort   
[Task]   
You are an attack scheduler for an authorized safety benchmark. Your task is to rank the provided at  
tack templates for the current query and victim model. Only rank the provided attack candidates; do not   
introduce or modify any attack method.   
[Victim Model]   
Name: {model}   
Description: {victim description}   
[Candidate Attacks]   
For each attack a , the following information is provided:   
Attack ID: {attack id}   
Historical ASR: {historic asr current}   
Attack Description: {attack description}   
Attack ID: {attack id}   
Historical ASR: {historic asr current}   
Attack Description: {attack description}   
[Ranking Rules and Instructions]   
{ranking rules and instructions}   
[Output Requirement]   
Return all candidate attacks from most to least promising. Use every provided attack ID exactly once and   
provide a concise rationale for each ranking decision. Return valid JSON only.   
[Output Format]   
{   
"ranking": [   
{   
"attack": "<attack\_id>",   
"reason": "<brief ranking rationale>"   
},   
]   
}   
[Current Query]   
{query}

The prompt therefore exposes the scheduler to the empirical prior and semantic information of every candidate attack in a unified format, while the fixed instruction I determines how these signals should be combined. The resulting JSON ranking can be directly parsed as the attack execution order.

## A.2 DEFENSE CONDITIONS AND REPRESENTATIVE METHODS

We treat defenses as part of the evaluation conditions rather than as methods introduced in this work. Our goal is therefore not to provide an exhaustive benchmark of jailbreak defenses, but to evaluate whether attack scheduling remains effective when the victim model is deployed under qualitatively different guardrail mechanisms. In particular, defenses can alter not only the overall attack success rate, but also the relative effectiveness of individual attacks. Such attack-specific changes are especially important for scheduling, since a ranking estimated under one condition may become stale once the deployment environment changes.

We organize the defense dimension according to the primary signal used to identify or suppress jailbreak behavior. This yields four evaluation settings: no defense, intent reconstruction, consistency-based defense, and detector-based defense. These categories capture three substantially different ways of introducing a guardrail around the victim model: recovering the latent malicious intent obscured by an adversarial prompt, testing whether the attack remains effective under input perturbations, or explicitly detecting unsafe inputs or outputs with a learned safety model. We instantiate these categories using Backtranslation (Wang et al., 2024), SmoothLLM (Robey et al., 2025), and LlamaGuard (Inan et al., 2023), respectively.

No defense. The no-defense setting serves as the reference condition and evaluates attacks directly against the safety alignment of the victim model without any additional guardrail. Formally, an adversarial query is passed directly to the victim model, and attack success is determined from the resulting response. This condition provides the historical attack effectiveness profile against which defense-induced changes can be compared. In our setting, it is particularly useful for determining whether an attack ordering learned from an undefended environment remains reliable after an additional defense is introduced.

Intent reconstruction: Backtranslation. Intent-reconstruction defenses target the mismatch between the surface form of a jailbreak prompt and its underlying harmful objective. Many jailbreak methods rely on role-playing, indirect instructions, adversarial formatting, or other forms of semantic disguise to conceal an otherwise recognizable malicious request. Rather than judging the adversarial prompt solely in its original form, an intent-reconstruction defense attempts to recover a more explicit representation of the latent request before making a safety decision.

We use Backtranslation (Wang et al., 2024) as the representative method for this category. Backtranslation exploits the model’s generated behavior to infer the underlying instruction that could have produced it, thereby exposing harmful intent that may be obscured in the original adversarial formulation. Conceptually, the defense transforms the evaluation from

$$
q _ { \mathrm { a d v } } \longrightarrow \mathrm { r e s p o n s e }
$$

into a process that additionally reasons about a reconstructed intent

$$
q _ { \mathrm { a d v } } \longrightarrow \mathrm { r e s p o n s e } \longrightarrow { \hat { q } } _ { \mathrm { i n t e n t } } ,
$$

where $\hat { q } _ { \mathrm { i n t e n t } }$ provides a less obfuscated representation of the request.

This mechanism is particularly relevant to attack scheduling because its effect is inherently attackdependent. Attacks whose success primarily derives from semantic obfuscation may be strongly weakened once their underlying intent is reconstructed, whereas attacks relying on different mechanisms may be affected to a lesser extent. Consequently, Backtranslation can substantially alter the relative ranking of attacks rather than merely reducing their success rates uniformly.

Consistency-based defense: SmoothLLM. Consistency-based defenses exploit the observation that many adversarial jailbreak prompts are brittle to small perturbations. Instead of attempting to explicitly infer malicious intent, these methods examine whether the behavior induced by the adversarial input remains stable under nearby variants of the same prompt. The underlying intuition is that an attack that depends on a precise adversarial pattern may lose its effectiveness when that pattern is slightly altered.

We adopt SmoothLLM (Robey et al., 2025) as the representative consistency-based defense. Given an input query q, SmoothLLM constructs multiple perturbed variants,

$$
\{ q ^ { ( 1 ) } , q ^ { ( 2 ) } , \dots , q ^ { ( M ) } \} ,
$$

evaluates the victim model on these variants, and aggregates the resulting behaviors to determine the final response. In this way, the defense explicitly tests the local robustness of the jailbreak mechanism rather than relying only on a single evaluation of the original input.

SmoothLLM provides a complementary evaluation condition to Backtranslation. Whereas Backtranslation challenges attacks by exposing their underlying semantics, SmoothLLM challenges the stability of the adversarial construction. Different attacks can vary substantially in perturbation robustness: attacks that rely on precise suffixes, token patterns, or fragile formatting may deteriorate sharply, while more semantically robust attacks may retain much of their effectiveness. The resulting non-uniform changes in attack success rates can therefore induce a substantial reordering of the attack library.

Detector-based defense: LlamaGuard. Detector-based defenses introduce an explicit safety model that classifies an input, an output, or both according to a predefined safety policy. Unlike intent reconstruction and consistency-based defenses, they do not necessarily modify the adversarial query itself. Instead, they add a learned decision layer around the victim model that determines whether the interaction should be allowed or rejected.

We use LlamaGuard (Inan et al., 2023) as the representative detector-based defense. LlamaGuard is a dedicated safeguard model trained to classify model inputs and outputs according to safety categories. Under such a deployment condition, a jailbreak is successful only when it both induces the desired unsafe behavior from the victim model and avoids being rejected by the external guard. Thus, the effective success probability depends jointly on victim vulnerability and detector evasion:

P(success) ≈ P(victim is successfully attacked) · P(guard accepts | attack).

This additional detection stage can change attack effectiveness in a manner that is qualitatively different from the previous two defenses. An attack may remain highly effective at eliciting an unsafe response from the victim model while becoming ineffective at the system level because its input or output is easily detected by the guard. Conversely, attacks that produce less readily detectable interaction patterns may become relatively more competitive. Detector-based defenses can therefore induce substantial attack-specific shifts even when the underlying victim model remains unchanged.

Rationale for representative selection. The three defenses are selected to maximize mechanistic diversity rather than to exhaustively enumerate existing jailbreak defenses. Backtranslation, SmoothLLM, and LlamaGuard intervene at different stages and rely on different signals: latent intent, perturbation consistency, and learned safety detection, respectively. As a result, they expose the scheduler to qualitatively distinct forms of defense-induced distribution shift.

This distinction is important for our evaluation. If every defense reduced all attack success rates by approximately the same factor, the original attack ordering would remain largely unchanged and a static historical ranking could still be effective. In practice, however, different guardrail mechanisms interact differently with different attack strategies. We can express this distinction as

$$
\mathrm { A S R } _ { \operatorname { d e f } } ( a ) \neq c \cdot \mathrm { A S R } _ { \operatorname { n a t i v e } } ( a ) ,
$$

for a common scaling factor c across attacks a. Instead, the defense induces an attack-dependent transformation

$$
\mathrm { A S R } _ { \operatorname { d e f } } ( a ) = f _ { \operatorname * { d e f } } ( a , \mathrm { A S R } _ { \operatorname { n a t i v e } } ( a ) ) ,
$$

which can change both the magnitude and the ordering of attack effectiveness.

Accordingly, the purpose of these defense conditions is not merely to make the victim model uniformly harder to attack. They provide controlled environments in which historical attack priors can become unreliable for different reasons. Backtranslation can suppress attacks that depend on semantic disguise, SmoothLLM can suppress attacks that are locally brittle, and LlamaGuard can suppress attacks that are easily recognizable by an explicit safety detector. Together with the no-defense baseline, these conditions allow us to evaluate whether a scheduling policy can remain effective when the attack-effectiveness landscape changes across deployment environments.

Therefore, the four settings can be summarized as

Defense Condition Shift → Attack-Specific ASR Shift → Ranking Shift → Stale Historical Prior

This perspective directly matches the central evaluation objective of our scheduling framework: determining whether semantic guidance and online feedback can compensate for changes in attack effectiveness that cannot be captured by a fixed historical ranking.