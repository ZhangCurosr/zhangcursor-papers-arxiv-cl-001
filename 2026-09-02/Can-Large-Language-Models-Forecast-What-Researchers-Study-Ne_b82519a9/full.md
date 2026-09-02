# Can Large Language Models Forecast What Researchers Study Next?

Fenghai Li Zihan Tang\* Haofei Yu Yining Zhao Jiaxuan You University of Illinois Urbana-Champaign

## Abstract

Large language models increasingly generate research ideas, yet judging their novelty or feasibility at generation time does not establish whether they anticipate subsequent work. We introduce IDEAFORECASTBENCH to evaluate research idea forecasting. Given a community’s literature up to a cutoff, a system produces up to five ranked ideas, which are evaluated against later papers. The benchmark comprises 624 rolling episodes across 52 topics, with a fixed retrieve-then-judge protocol and separately reported results from two judges. We compare five history-compression strategies across GPT-4.1, Qwen2.5-7B/14B, and Qwen3.5-9B, together with a learned Mode-Decomposition Forecaster (MDF). Under the primary GPT-4.1-mini judge, Summary improves on Direct in Hit@5 and Precision@5 across all four backbones. Qwen2.5 scores above GPT-4.1, whereas Qwen3.5 scores below it. An outcome-blind assessment finds that Qwen2.5 produces broader forecasts, but does not identify how much breadth contributes to its advantage. Threshold and judge diagnostics further clarify the limits of interpreting realization as precise anticipation. IDEAFORECAST-BENCH provides a common task for studying which research ideas a community subsequently pursues and how reliably this outcome can be measured.

## 1 Introduction

Large language models (LLMs) are increasingly used for scientific literature understanding (Li et al., 2025), literature review (Tang et al., 2025), research ideation (Baek et al., 2025; Si et al., 2025), and autonomous discovery (Lu et al., 2024; Zheng et al., 2025). Research topics and citation impact evolve over time (Blei and Lafferty, 2006; Wang et al., 2013), and a community’s emerging problems and methods may signal its future work. By compressing this literature, LLMs may capture these signals. We therefore ask: can LLMs predict research ideas that later emerge in a research community?

![](images/7762edbcc89ee5ee44b6372806e010d95773f19e13c76c952808db0654ce9ab6.jpg)  
Figure 1: Executing an idea versus forecasting the field. Idea execution evaluates a proposal through its own experiment. Idea forecasting asks whether related research is realized in the community’s subsequent publications, using only the allowed historical literature as input.

We study this question through research idea forecasting. Prior work on research ideation primarily evaluates whether generated ideas are novel, feasible, or promising (Si et al., 2025; Baek et al., 2025), or evaluates the outcome of executing selected proposals one at a time (Si et al., 2026). We ask whether a forecaster can anticipate the ideas that a research community subsequently pursues (Figure 1). Given literature available up to a cutoff t, a forecaster produces a ranked list of research ideas, which are evaluated against papers appearing afterward. Subsequent publications provide an observable realization outcome beyond immediate judgments of an idea’s persuasiveness.

Research idea forecasting has three potential uses in scientific discovery and the study of research communities. (1) Early discovery: a strong forecaster could identify promising directions before they appear in the literature, extending ideation beyond immediate assessments of novelty or plausibility (Si et al., 2025; Baek et al., 2025; Guo et al., 2025). (2) Research decision-making: forecasts could help prioritize problems, methods, and directions for further investigation, informing research planning and autonomous discovery (Lu et al., 2024; Zheng et al., 2025). (3) Community modeling: forecasting provides an operational way to study how research communities evolve, with natural-language ideas as prediction units rather than topics, citations, or aggregate impact (Blei and Lafferty, 2006; Wang et al., 2013; Yu et al., 2025).

This task presents three challenges. (1) Noisy history: forecasting requires selecting relevant evidence from a large and heterogeneous literature. Under a limited context budget, a forecaster must decide which evidence to retain and how to represent it. (2) Dynamic research directions: research topics evolve over time (Blei and Lafferty, 2006), requiring forecasters to reassess which historical evidence remains informative. (3) Open-ended benchmarking: research ideas and hypotheses are difficult to evaluate (Si et al., 2025; Guo et al., 2025; Liu et al., 2025). Comparing forecasts with future papers grounds evaluation in subsequent research, but there is no single correct future idea, and semantic similarity alone does not establish that a paper realizes a predicted contribution. A broad statement may match many papers while making few testable commitments.

Contributions. We make two contributions. (1) IdeaForecastBench and its evaluation protocol. We define community-level idea forecasting and implement it with a shared rolling-window manifest, inspectable idea–paper matching, and diagnostics for sensitivity to the judge and specificity threshold (Section 3). (2) Forecasting through history compression. We organize five prompting strategies by what they retain from history, compare them across four generation backbones, and include the trainable Mode-Decomposition Forecaster (MDF) as a reference implementation (Section 4).

Empirical findings. Under GPT-4.1-mini judging, Summary improves Hit@5 over Direct from 0.487 to 0.756 on GPT-4.1 and from 0.571 to 0.949 on Qwen2.5-7B. These realization scores do not establish that nearly every forecast precisely anticipates a new discovery. An outcome-blind assessment of 832 forecasts finds that Qwen2.5’s outputs are broader than GPT-4.1’s. Its score advantage persists under a stricter matching gate for several strategies, but that gate is not a control for intrinsic generality. We therefore report the matching advantage without attributing it entirely to either better anticipation or broader wording. The learned reference’s scores also vary substantially across judges, motivating diagnostics alongside the leaderboard.

Our aim is to support comparable forecasts and inspectable judgments while tracking changes to benchmark construction, evaluation, and forecasting methods separately.

## 2 Related Work

Automatic research agents. Large language models automate code generation from papers (Seo et al., 2026), code-based experimentation (Jansen et al., 2025), end-to-end discovery from experiment to paper (Lu et al., 2024), community-level paper and review generation (Yu et al., 2025), and research idea generation (Baek et al., 2025; Si et al., 2025; Zhao et al., 2025). IdeaBench evaluates generated ideas for novelty and feasibility (Guo et al., 2025); HypoBench evaluates hypotheses for predictive utility, generalizability, and recovery of groundtruth hypotheses (Liu et al., 2025). Neither directly measures whether related ideas later emerge in the literature.

Self-evolving agents. Agents that improve across episodes draw on long-term memory (Zhong et al., 2024; Packer et al., 2023), reflective feedback (Shinn et al., 2023), embodied skill accumulation (Wang et al., 2024), and procedural, reasoning, and evolving-skill memories (Cao et al., 2026; Ouyang et al., 2026; Zhang et al., 2026). These agents are evaluated on conversation, document analysis, question answering, interactive, and coding tasks, whereas our task uses subsequent publications as delayed, inspectable feedback. Our reference forecaster uses a typed memory updated by fixed rules (Appendix B).

Forecasting benchmarks. Forecasting evaluations cover general future questions (Karger et al., 2025; Halawi et al., 2024), while broader financial benchmarks include prediction and decisionmaking tasks (Xie et al., 2023, 2024). Closer to science, Wen et al. (2025) predict the outcomes of empirical AI experiments without running them; Pre-Science (Ajith et al., 2026) decomposes an advance into structured prediction tasks; and CUSP (Wu et al., 2026) forecasts milestone events. We compare the concurrent PreScience and CUSP benchmarks by task design rather than numerical scores. PreScience’s contribution-generation subtask is closest to ours: it evaluates a prediction against a single held-out abstract. IDEAFORECASTBENCH evaluates a ranked set of ideas against a community’s post-cutoff paper stream (Table 1). This shifts the target from reconstructing one abstract or predicting a pre-specified milestone to communitylevel realization over rolling windows. Because continuations of existing work can also be realized, we distinguish realization from novelty and examine its measurement sensitivity in Section 5.4.

Table 1: Only IDEAFORECASTBENCH scores natural-language ideas against the full future stream. All share a time cutoff. Idea: the forecast is a naturallanguage idea $( \sqrt { I / \times I } \sim \mathrm { = y e s / n o / p a r t i a l ) }$ ; Scale counts the questions, unique in-scope papers, or events a benchmark is built from. IdeaForecastBench’s 42.8K is the current deduplicated count assigned to at least one topic.
<table><tr><td>Benchmark</td><td>Idea</td><td>Target</td><td>Scale</td></tr><tr><td>ForecastBench</td><td>X</td><td>resolved Qs</td><td>1K</td></tr><tr><td>PreScience</td><td>2</td><td>1 abstract</td><td>98K</td></tr><tr><td>CUSP</td><td>2</td><td>milestones</td><td>4.7K</td></tr><tr><td>IDEAFORECASTBENCH</td><td> $\checkmark$ </td><td>future stream</td><td>42.8K</td></tr></table>

## 3 Benchmarking Idea Forecasting

IDEAFORECASTBENCH evaluates naturallanguage forecasts against subsequent publications. Its unit is a topic–cutoff episode, not an idea with a single pre-specified answer (Figure 2).

## 3.1 Problem Definition for Idea Forecasting

Let $\mathcal { P } _ { c }$ be the papers assigned to topic $c , d ( p )$ a paper’s first-submission date, and t a monthly cutoff date. A forecaster observes the allowed history

$$
X _ { c , t } = \{ p \in \mathcal { P } _ { c } : d _ { 0 } \leq d ( p ) \leq t \} ,\tag{1}
$$

where $d _ { 0 }$ is the start of the evaluated corpus. It produces an ordered list $\hat { Y } _ { c , t } = ( \hat { y } _ { 1 } , \dots , \hat { y } _ { K } )$ with budget $K = 5$ . Each idea names a problem and an approach; its structured fields include a title, rationale, method description, and optional key terms. The method may select, summarize, cluster, or otherwise compress the history, but may not access post-cutoff papers as input.

The target is the topic’s post-cutoff paper set

$$
Y _ { c , t } = \{ p \in \mathcal { P } _ { c } : t < d ( p ) \leq e ( t ) \} ,\tag{2}
$$

where e(t) is the last day of the month obtained by adding three to the cutoff month. A forecast is operationally realized when a retrieved paper satisfies the idea–paper matching rubric of Section 3.3. The task measures whether subsequent work is consistent with the forecast. It does not assess the forecaster’s ability to execute the idea, the idea’s scientific value, or verbatim agreement with an abstract.

Open-ended targets. Many ideas may be realized in one episode, and broad forecasts may resemble several later papers. Because valid future ideas cannot be exhaustively enumerated, we evaluate a fixed forecast budget rather than recall over all possibilities. Interpretation requires both realization frequency and forecast informativeness.

Realization is a proxy. Publication provides an inspectable but delayed and incomplete record of realization. Unmatched ideas may appear later or outside the corpus, while matched ideas may be incremental continuations. We measure realization within a specified pool and horizon, not scientific value or uniquely novel anticipation. Section 5 examines the resulting measurement limitations.

## 3.2 Data Collection and Benchmark Construction

Ingestion and deduplication. We collect arXiv machine-learning papers with cat:cs.ML, retaining identifiers, titles, abstracts or contribution summaries, submission dates, and category metadata. Repeated results and cross-listings are merged by identifier (Appendix A).

Temporal assignment. We split papers by firstsubmission date, so later revisions do not move them into the target pool. This rule determines membership; text available at each cutoff requires a separate provenance check. Forecasters select and summarize only historical papers.

Research communities. Fixed names, aliases, and keyword rules assign papers to 52 overlapping topics from title/key-point text, including retrievalaugmented generation, reinforcement learning, and medical imaging. A paper can belong to several topics but appears once within each. Topic definitions remain fixed across methods and cutoffs.

Rolling windows. The corpus spans April 2024– September 2025. Twelve monthly cutoffs from July 2024 through June 2025 yield $5 2 \times 1 2 = 6 2 4$ episodes. History expands from April 2024; earlier literature is outside the evaluated snapshot.

![](images/9f9e443a61d5ad4b8a14fdd89116d9d662e892a92a244f1734d2f31c731fd44b.jpg)  
Figure 2: Overview of IDEAFORECASTBENCH. (1) Monthly cutoffs divide historical inputs from post-cutoff target papers across 52 overlapping topics; the bars show topic assignments over the displayed cutoff intervals, not unique papers. (2) Given a topic’s history up to t, a forecaster produces K=5 ranked ideas. Targets are papers first submitted after t through the last day of month t+3. (3) For each idea, the evaluator retrieves R=10 candidate papers and applies the $P { \mathrm { + } } M \geq 5 , S \geq 2$ matching gate, with at most one credited idea per paper. Judges are reported separately. (4) Hit@5, Precision@5, and MRR are averaged over 624 episodes; historical embedding distance is reported separately as the Novelty diagnostic.

Exact calendar boundary. For a July 1, 2024 cutoff, history includes that day and targets span July 2–October 31. The three-month endpoint offset thus spans parts of four calendar months. All results share this boundary; overlapping monthly targets remain grouped within topics for uncertainty estimation.

Common episode manifest. All 21 configurations share 624 topic–cutoff keys, endpoints, and pool sizes. Release verification must also compare paper identifiers, since equal counts do not establish identical membership (Appendix A).

Availability and scale. The eligibility threshold is two historical papers; the actual minimum is 33. History pools contain 33–4,530 papers (mean 589.9); target pools contain 35–1,722 (mean 313.3). Appendix A reports all topic-level counts and distinguishes this evaluation slice from the earlier snapshot.

Temporal access versus pretraining. Restricting inputs to historical papers does not rule out a pretrained backbone’s exposure to target papers. For trained forecasters, training labels and reward papers must also be separated from evaluation targets; earlier training cutoffs alone are insufficient. Appendix I.2 discusses the available contamination probe and its limitations.

## 3.3 Evaluation Protocol for Idea Forecasting

The protocol separates candidate retrieval, rubricbased matching, and aggregation to determine which ideas receive credit. Section 5 reports scores and measurement diagnostics.

Candidate retrieval. We embed each forecast and target paper with VOYAGE-3-LARGE (1024 dimensions), retrieving the top R = 10 by cosine similarity. This evaluation-time funnel is distinct from method-side historical retrieval. Similar terminology does not itself establish realization; the judge makes that decision.

FORECAST · LLM safety · cutoff 2024-10 · Direct Prompting   
“Automated multi-turn jailbreak search with tactic planning and   
turn-level token optimization.”   
Automated Red Teaming with GOAT arXiv:2410.01606   
“. . . instantiated with 7 red teaming attacks . . . reasoning through the   
choices of methods available.”   
P 3 M2 S 2 ✓ match (a predefined tactic set)   
AutoDAN-Turbo arXiv:2410.05295   
“. . . discover strategies from scratch, without . . . predefined scopes   
(e.g., specified candidate strategies).”   
P 3 M 2 S 1 no match (strategies self-discovered)  
Figure 3: A worked matching example. Two papers on the same problem receive different specificity judgments under Qwen3.5-9B. This example comes from the earlier evaluation and illustrates the matching rule; it is not a paired-judge observation from the current experiment.

Idea–paper matching. The judge sees the forecast’s title, rationale, approach, and key terms, together with the candidate paper’s title and abstract. It scores Problem (P), Method (M), and Specificity (S), each from 0 to 3, and provides a short rationale. The default gate is

$$
g ( \hat { y } _ { i } , p ) = { \bf 1 } [ P + M \ge 5 ~ \land ~ S \ge 2 ] .\tag{3}
$$

The gate requires close problem/method agreement and at least partial realization of the core idea. Raising S from 2 to 3 requires closer technical agreement; we examine this sensitivity separately.

Separately reported judges. GPT-4.1-mini is primary; Qwen3.5-9B separately evaluates the same predictions. We do not ensemble their scores. Comparisons distinguish execution failures from judgment differences (Section 5.4).

Crediting a forecast. In rank order, each prediction receives credit from the first passing candidate not already credited in that episode. Thus each paper supports at most one idea. Figure 3 illustrates rejection; Appendix D details aggregation.

Metrics. Let $a _ { i } \in \{ 0 , 1 \}$ be the credited-match indicator after within-window paper deduplication. We report

$$
\mathrm { H i t @ 5 = 1 } \left[ \sum _ { i = 1 } ^ { 5 } a _ { i } > 0 \right] ,\tag{4}
$$

$$
\mathrm { P r e c i s i o n @ 5 } = \frac { 1 } { 5 } \sum _ { i = 1 } ^ { 5 } a _ { i } .\tag{5}
$$

Hit@5 asks whether any forecast is realized; Precision@5 measures the fraction of the five-idea budget receiving distinct-paper credit. Missing outputs receive no credit. Mean reciprocal rank (MRR) is 1/ min $\{ i : a _ { i } = 1 \}$ , or zero if no idea is credited. Table 3 reports all three, averaged over the common episode manifest, alongside judgeindependent Novelty. Precision@5 is distinct from the judge’s Problem score P.

Novelty is a diagnostic. For each forecast we also compute its embedding distance from the closest visible historical paper,

$$
N ( \hat { y } _ { i } ) = 1 - \operatorname* { m a x } _ { x \in X _ { c , t } } \cos ( E ( \hat { y } _ { i } ) , E ( x ) ) .\tag{6}
$$

This judge-independent distance does not measure scientific originality: vague forecasts can be distant from prior work, while useful extensions remain close. We therefore report it separately from realization, without a composite score.

Paired uncertainty. We compute 95% percentile intervals from 10,000 topic-clustered bootstrap samples, retaining all cutoffs of each sampled topic. Comparisons pair topic–cutoff episodes. Topic overlap can still induce cross-topic dependence; full intervals appear in Appendix E.

Changing opportunity sets. Larger target pools offer more realization opportunities even with a fixed retrieval depth. We therefore pair methods on identical windows and examine topic consistency and pool-size sensitivity. These checks do not make scores from different snapshots interchangeable.

## 4 Forecasting via History Compression

With the benchmark’s history and target pools fixed, we compare how forecasters select and organize evidence. Five history-compression strategies use selection, abstraction, trajectories, or memory; MDF introduces a learned structured representation. We compare complete pipelines without equalizing token or compute budgets.

## 4.1 Five Forecasting Strategies

Table 2 summarizes the five prompting baselines. All condition on the permitted historical side and request the same five-idea output schema. Their prompts and selection rules are retained in Appendix C and Appendix J.

Truncation and selection. DIRECT forecasts from recent abstracts in one call, preserving local details but discarding older context. RETRIEVAL instead uses recent titles and keywords to select historical evidence with hybrid semantic and lexical similarity. It changes which papers are retained without abstracting them into a field-level account.

Table 2: What each historical representation preserves. Limits describe the reference implementations; available history can be shorter.
<table><tr><td>Strategy</td><td>Forecasting context</td></tr><tr><td>Direct</td><td>Up to 20 recent abstracts, without abstrac- tion.</td></tr><tr><td>Retrieval</td><td>Up to 20 historical papers selected for rele- vance to recent work.</td></tr><tr><td>Summary</td><td>One paragraph distilled from up to 60 re-</td></tr><tr><td>Topic Trend</td><td>cent paper snippets. A small set of cluster-level research trajec- tories.</td></tr><tr><td>Memory</td><td>Eight bullets from older work plus up to 20 recent abstracts.</td></tr></table>

Abstraction and trajectories. SUMMARY condenses recent snippets into roughly eight sentences and forecasts from that paragraph alone, testing whether themes and open problems support forecasting without paper-level detail. TOPIC TREND forecasts from clusters ranked by recent activity. Its output-budget behavior is analyzed in Section 5.1.

Two-tier memory. MEMORY compresses papers older than six months into eight bullets and combines them with recent abstracts, preserving more short-term detail than Summary. Because the corpus begins in April 2024, early cutoffs lack a substantial older-paper pool. Its effective context therefore changes as history accumulates; the experiment does not provide uniformly long histories.

## 4.2 A Learned Reference: MDF

The Mode-Decomposition Forecaster (MDF) introduces a structured intermediate representation of historical innovations. It represents an idea as $z ~ = ~ ( b , o , g )$ : a base direction b, an operator o, and a target gap g. For example, a base direction could be preference optimization, an operator an extension, and a gap robustness under distribution shift. The operator inventory contains EXTEND, TRANSFER, COMPOSE, BENCHMARK, ANALYZE, SIMPLIFY, SCALE, and ADAPT.

From an innovation to an idea. A typed memory $\mathcal { M } _ { t }$ stores innovations and their frequency, recency, and utility. The prior $p _ { \theta } ( z \mid \mathcal { M } _ { t } )$ predicts an innovation triple, and the realization policy $p _ { \psi } ( y \mid z , X _ { c , t } )$ converts it into a grounded idea.

Here, policy realization refers to generating idea text; benchmark realization refers to a subsequent paper matching that idea. The decomposition is

$$
p ( \boldsymbol { y } \mid X _ { c , t } ) \approx \sum _ { z } p _ { \psi } ( \boldsymbol { y } \mid z , X _ { c , t } ) p _ { \theta } ( z \mid \mathcal { M } _ { t } ) .\tag{7}
$$

Inference approximates the sum through a candidate pool, blends prior and realization scores, removes near-duplicates, and returns the top five predicted ideas (Algorithm 1). The underlying hypothesis is that predicting structured research moves may be easier than directly generating fully specified ideas.

Learning from delayed realization. The prior learns from hindsight-extracted triples; the realization policy uses a gated foresight reward, distinct from evaluation-time retrieval and the P/M/S gate. The evaluated checkpoint uses Qwen2.5-7B. Appendix B separates reference training settings from its unverified run manifest. We evaluate MDF as a complete pipeline. Without matched ablations, its score cannot identify the separate contributions of the prior, memory, or reinforcement learning.

## 5 Experiments and Analysis

We first compare realization and ranking performance under the shared protocol. We then examine how forecast generality, judge and threshold choices, and MDF’s evaluated representation affect the interpretation of these scores.

## 5.1 Experimental Setup

We evaluate five strategies on GPT-4.1, Qwen2.5- 7B/14B, and Qwen3.5-9B, plus MDF: 21 configurations on 624 episodes. Each judge scores the same generated predictions. Qwen3.5’s overlapping generation shards are resolved by a fixed whole-episode rule (Appendix E.1). Context limits are strategy-specific; token and call budgets are not equalized across strategies.

Table 3 uses final match flags with withinwindow paper deduplication. Strict-gate analysis uses all candidate judgments; representative P/M/S triples alone cannot reconstruct candidate selection (Appendix D).

Output-budget validity. Topic Trend fills five slots in 60.1–60.7% of windows on GPT-4.1 and Qwen2.5, but only 12.7% on Qwen3.5. Other strategies fill at least 98.9%. All episodes, including empty outputs, remain in the denominator; unfilled slots receive zero credit (Appendix G).

Table 3: Forecasting results on 624 common episodes. Bold marks column maxima, not statistical significance. Both judges use $P + M \ge 5$ and $S \geq 2$ on the same predictions. Novelty is historical distance, not accuracy. Qwen3.5-9B appears as both a generator and a judge; its generator results use the fixed shard selection in Appendix E.1. Qwen-judge scores remain provisional because of execution failures (Section 5.4). Intervals and supplementary metrics appear in Appendix E.
<table><tr><td rowspan="2"></td><td rowspan="2">Strategy</td><td rowspan="2">Novelty</td><td colspan="3">GPT-4.1-mini judge</td><td colspan="3">Qwen3.5-9B judge</td></tr><tr><td>Hit@5</td><td>Precision@5</td><td>MRR</td><td>Hit@5</td><td>Precision@5</td><td>MRR</td></tr><tr><td rowspan="5">GPT-4.1</td><td>Summary</td><td>0.181</td><td>0.756</td><td>0.297</td><td>0.519</td><td>0.822</td><td>0.323</td><td>0.548</td></tr><tr><td>Memory</td><td>0.160</td><td>0.729</td><td>0.267</td><td>0.456</td><td>0.772</td><td>0.292</td><td>0.464</td></tr><tr><td>Retrieval</td><td>0.130</td><td>0.696</td><td>0.257</td><td>0.465</td><td>0.692</td><td>0.237</td><td>0.396</td></tr><tr><td>Topic Trend</td><td>0.238</td><td>0.625</td><td>0.135</td><td>0.598</td><td>0.655</td><td>0.148</td><td>0.626</td></tr><tr><td>Direct</td><td>0.127</td><td>0.487</td><td>0.149</td><td>0.283</td><td>0.490</td><td>0.137</td><td>0.266</td></tr><tr><td rowspan="5">Qwen2.5-7B</td><td>Summary</td><td>0.189</td><td>0.949</td><td>0.528</td><td>0.686</td><td>0.968</td><td>0.563</td><td>0.707</td></tr><tr><td>Memory</td><td>0.161</td><td>0.869</td><td>0.412</td><td>0.612</td><td>0.917</td><td>0.454</td><td>0.626</td></tr><tr><td>Retrieval</td><td>0.112</td><td>0.769</td><td>0.338</td><td>0.524</td><td>0.837</td><td>0.372</td><td>0.562</td></tr><tr><td>Topic Trend</td><td>0.297</td><td>0.745</td><td>0.153</td><td>0.742</td><td>0.779</td><td>0.163</td><td>0.773</td></tr><tr><td>Direct</td><td>0.105</td><td>0.571</td><td>0.173</td><td>0.301</td><td>0.628</td><td>0.189</td><td>0.323</td></tr><tr><td rowspan="5">Qwen2.5-14B</td><td>Summary</td><td>0.201</td><td>0.954</td><td>0.553</td><td>0.716</td><td>0.973</td><td>0.613</td><td>0.737</td></tr><tr><td>Memory</td><td>0.178</td><td>0.913</td><td>0.440</td><td>0.610</td><td>0.962</td><td>0.504</td><td>0.648</td></tr><tr><td>Retrieval</td><td>0.145</td><td>0.854</td><td>0.396</td><td>0.555</td><td>0.886</td><td>0.409</td><td>0.563</td></tr><tr><td>Topic Trend</td><td>0.288</td><td>0.784</td><td>0.167</td><td>0.768</td><td>0.825</td><td>0.189</td><td>0.815</td></tr><tr><td>Direct</td><td>0.134</td><td>0.615</td><td>0.201</td><td>0.321</td><td>0.654</td><td>0.217</td><td>0.330</td></tr><tr><td rowspan="5">Qwen3.5-9B</td><td>Summary</td><td>0.190</td><td>0.532</td><td>0.172</td><td>0.316</td><td>0.498</td><td>0.144</td><td>0.233</td></tr><tr><td>Memory</td><td>0.149</td><td>0.471</td><td>0.134</td><td>0.274</td><td>0.316</td><td>0.077</td><td>0.159</td></tr><tr><td>Retrieval</td><td>0.139</td><td>0.454</td><td>0.130</td><td>0.267</td><td>0.277</td><td>0.066</td><td>0.138</td></tr><tr><td>Topic Trend</td><td>0.229</td><td>0.340</td><td>0.072</td><td>0.328</td><td>0.292</td><td>0.061</td><td>0.277</td></tr><tr><td>Direct</td><td>0.138</td><td>0.226</td><td>0.057</td><td>0.133</td><td>0.131</td><td>0.028</td><td>0.062</td></tr><tr><td>Qwen2.5-7B</td><td>MDF</td><td>0.200</td><td>0.545</td><td>0.171</td><td>0.310</td><td>0.296</td><td>0.080</td><td>0.158</td></tr></table>

## 5.2 Main Results

Summary leads in realization frequency. Summary has the highest Hit@5 and Precision@5 point estimates for every backbone under both judges. Its primary-judge Hit@5 gains over Direct are 0.269, 0.378, 0.338, and 0.306 for GPT-4.1, Qwen2.5-7B, Qwen2.5-14B, and Qwen3.5-9B. Memory and Retrieval also improve on Direct. These comparisons measure gains from complete pipelines; they do not isolate abstraction from context selection or additional model calls.

Realization frequency, yield, and rank differ. Qwen2.5-14B Summary reaches Hit@5 0.954 under the primary judge, but Precision@5 is 0.553. Thus, almost every episode contains a credited idea, but only about half of the five-slot budget receives distinct-paper credit. MRR is 0.716, indicating how early the first credited idea appears. The metrics distinguish realization frequency, credited yield, and first-match rank. Precision@5 remains informative when Hit@5 approaches its ceiling; historical embedding distance measures none of

these outcomes.

Backbone effects are large but not selfexplanatory. Qwen2.5 exceeds GPT-4.1 on each strategy, but Qwen3.5 does not share that advantage. Under the primary judge, its Summary Hit@5 is 0.532, versus 0.756 for GPT-4.1 and 0.954 for Qwen2.5-14B; the paired difference from GPT-4.1 is −0.224 (95% CI [−0.274, −0.173]). Qwen3.5’s Hit@5 is lower on all five strategies. The Qwen2.5 advantage therefore does not extend uniformly across the Qwen family, and these scores do not rank general model capability. The outcome-blind analysis below examines Qwen2.5’s breadth but cannot explain Qwen3.5’s lower scores.

Compression is not uniformly beneficial in every form. Topic Trend exceeds Direct in Hit@5 on all four backbones, but has lower Precision@5 on GPT-4.1 and Qwen2.5. It also exceeds Summary in MRR (0.598 vs. 0.519 on GPT-4.1), yet underfills the budget and reuses titles across windows. Its low budget completion on Qwen3.5 further limits attributing score differences to idea quality alone. Interpreting its ranking performance therefore also requires accounting for unfilled slots.

## 5.3 Generality and Forecasting Performance

Qwen2.5’s high scores leave open whether its forecasts anticipate future work more precisely or state broader ideas that more papers can satisfy. We examine forecast breadth and matching opportunities to assess this ambiguity, while keeping their association separate from causal evidence.

Outcome-blind assessment. The blind study samples one forecast per topic from the 15 GPT-4.1/Qwen2.5 configurations and MDF (832 forecasts); Qwen3.5 is not included. GPT-4.1-mini sees only title, rationale, and approach, without future papers, outcomes, backbone labels, or strategy labels. It scores problem, method, and scope specificity, plus testability, from 0 to 3. Generality is 12 minus their sum. The assessment thus measures breadth without candidate papers, although it still relies on an LLM.

Qwen2.5 is consistently broader. Figure 4(a) shows higher generality for both Qwen2.5 backbones across all five strategies. On Summary, GPT-4.1 scores 3.58 (95% CI [3.15, 4.00]), compared with 6.58 ([6.23, 6.94]) for Qwen2.5-7B and 6.48 ([6.13, 6.85]) for Qwen2.5-14B. The pattern supports a systematic difference in how precisely the backbones state their ideas, beyond isolated generic examples.

Breadth and matching opportunities. When we link blind ratings to outcomes, forecast-level match rates increase from 0.205 in the lowest generality bin to 0.406 in the highest; the correlation is 0.17 (0.21 excluding MDF). Candidate-level evidence is consistent with this association: Summary forecasts have a mean of 0.565 passing candidates among the retrieved ten for GPT-4.1, versus 1.470 and 1.642 for Qwen2.5-7B and 14B. These counts are restricted to the retrieved set and do not cover all future papers an idea could match. The association may also reflect differences in backbone, strategy, and topic (Appendix F).

What remains unresolved. Intrinsic specificity concerns how narrowly an idea is stated before outcomes are shown; the matching judge’s S concerns how closely a particular paper realizes it. A broad forecast can therefore receive high S, and an advantage under S ≥ 3 does not rule out an effect of generality. Limited overlap between the backbones’ generality distributions also makes conditional comparisons unstable. We therefore cannot estimate the share of the gap attributable to breadth or interpret a residual as pure forecasting ability. Such attribution requires a controlled specificity intervention or better-matched samples.

## 5.4 Judge and Threshold Sensitivity

Execution failures versus judge behavior. The evaluated artifacts record no judge parse failures for GPT-4.1-mini, but contain failures on the Qwenjudge side. For MDF, all candidate judgments fail in 72 of 624 windows. We therefore base the main performance interpretation on GPT-4.1-mini and treat Qwen-judge results as provisional. Until failed judgments are recovered, differences from the primary scores cannot be attributed to judge behavior alone.

Agreement is not accuracy. The candidate-level audit reports Summary binary agreement of 0.950 on GPT-4.1 forecasts and 0.894 on Qwen2.5-7B forecasts, with Cohen’s κ of 0.538 and 0.588. Most candidate pairs are negatives, so class imbalance partly accounts for the high raw agreement. Agreement alone neither establishes correctness nor rules out shared semantic biases; execution failures further limit interpretation.

Absolute levels change. For GPT-4.1 Summary, Hit@5 rises from 0.756 under GPT-4.1-mini to 0.822 under the Qwen judge. Qwen3.5 Summary changes in the opposite direction (0.532 to 0.498), as does MDF (0.545 to 0.296). Execution failures limit interpretation, and Qwen3.5’s two exports also have minor candidate-set differences (Appendix E.1). These shifts provide no basis for assuming a uniform judging advantage within a model family or applying a universal judge correction.

A stricter gate is a sensitivity analysis. We reuse all ten candidates’ scores when changing $S \geq 2 \mathrm { ~ t o ~ } S \geq 3 .$ . On Summary, Qwen2.5-7B’s paired Hit@5 advantage over GPT-4.1 changes from +0.192 to +0.163 (95% CI [0.120, 0.207]) under GPT-4.1-mini; the 14B advantage remains +0.181 ([0.133, 0.229]). The advantage persists under the stricter matching rule, but this result does not validate S as a measure of intrinsic specificity or control for forecast breadth. Judges may also differ in how often they assign the highest specificity category.

Human calibration has limited transfer. An earlier eight-annotator study includes 340 pairs and

![](images/365c6943d3d8fd831280854592818cb67877ea1487486fa96ccff4f4b503934f.jpg)

![](images/faa8b31380a50ea302ee65f9a1feb3f80dfdbff0ff6e66b311fcb186fc3fbfa6.jpg)  
Figure 4: Broader forecasts and higher realization scores coexist. (a) Outcome-blind generality on 52 forecasts per configuration, one per topic. (b) Paired Qwen2.5-minus-GPT-4.1 Hit@5 differences over 624 windows, using GPT-4.1-mini; 7B and 14B denote Qwen2.5-7B and Qwen2.5-14B. Circles use $S \geq 2 ;$ squares use $S \geq 3 ;$ bars show 95% topic-clustered intervals. This study covers GPT-4.1 and Qwen2.5, not Qwen3.5. The matching gate does not control intrinsic generality.

400 labels, with core-pair Fleiss’ $\kappa = 0 . 1 3 5$ . It documents disagreement in idea–paper matching on a different experimental slice and therefore does not calibrate the current scores (Appendix H). Automated judgments should therefore be interpreted as rubric-based measurements, not human ground truth.

## 5.5 MDF Diagnostics

MDF achieves Hit@5/Precision@5 of 0.545/0.171 under GPT-4.1-mini, versus 0.296/0.080 under Qwen3.5-9B, whose results include execution failures (Section 5.4). Under the primary judge, MDF trails the stronger prompting strategies and serves as a trainable reference.

Representation loss is not necessarily generation failure. The field audit reports a median approach length of four words and empty key-term lists for all 3,120 MDF forecasts. The inspected prediction adapter constructs the approach as operator: base\_direction, leaves key terms at their default, and truncates the rationale to 500 characters. Conversion can produce sparse evaluation fields even if the raw output contains a method description. Attribution to training requires comparing raw outputs, the deployed adapter, and the final text judged.

What the reference contributes. MDF provides a structured, trainable reference and shows why evaluation must preserve information from generated text to scored idea. Component ablations and seed variation remain unmeasured for this checkpoint; older-backbone experiments do not resolve those questions (Appendix G).

## 6 Discussion and Conclusion

We introduced IDEAFORECASTBENCH to study whether LLMs can forecast research ideas realized in a community’s subsequent publications. The benchmark compares ranked forecasts on shared topic–cutoff episodes and separately measures realization frequency, credited yield, and first-match rank. Summary achieves the highest Hit@5 and Precision@5 point estimates across the evaluated backbones under both judges. These results show that the representation of historical literature affects how often forecasts align with later papers, but do not by themselves establish precise anticipation of novel scientific contributions.

This distinction remains central to the question posed by our title. Qwen2.5’s higher realization scores coexist with broader forecasts, and its advantage under a stricter matching gate does not disentangle breadth from anticipation. Differences across judges and the MDF representation audit further limit a simple capability ranking. We therefore interpret the results as evidence of realization under a specified matching protocol, while leaving precise, novel anticipation unresolved. Better-aligned human calibration, controlled specificity comparisons, and forecasts frozen before target papers appear would strengthen that assessment.

## Limitations

Automated matching requires further calibration. Agreement between GPT-4.1-mini and Qwen3.5-9B does not establish correctness. The auxiliary human study concerns an earlier Qwenjudged experiment and cannot calibrate the current scores (Appendix H.2). A new study should align human and model instructions, sample current forecasts, and assess missed matches outside the retrieved candidate set.

Judge comparisons remain sensitive to execution and measurement. Qwen-judge results include execution failures, including all-candidate failures in 72 of 624 MDF windows. Until recovered, score differences cannot be attributed solely to judge behavior (Section 5.4). Agreement in rankings does not calibrate absolute scores. The outcome-blind generality assessment relies on one LLM, excludes Qwen3.5 generation, and does not identify the causal effect of forecast breadth.

The gate, retrieval depth, and cluster count are operating conventions. The protocol retrieves ten candidates and applies $P + M \ge 5 , S \ge 2$ Raising the gate to $S \geq 3$ tests matching sensitivity without controlling intrinsic specificity. Retrieval bounds the available evidence, while Coverage depends on a five-cluster partition (Appendix D). These choices provide no absolute scale of scientific originality or completeness.

The fixed snapshot does not eliminate pretraining exposure. Historical input filtering cannot exclude prior exposure to target papers. The contamination probe is an observational temporal comparison, not a causal estimate of memorization. Historical-text provenance and MDF trainingtarget separation require further audits (Appendices A and I.2). A prospective evaluation would freeze forecasts before collecting their target literature.

The benchmark covers selected machinelearning communities. The corpus covers arXiv cs.ML and 52 overlapping topics. Publications provide an incomplete, delayed record of research; unmatched ideas may be realized later or elsewhere. Cross-domain evaluation must account for publication timing and matching criteria. Topic-clustered intervals retain within-topic dependence, but overlapping membership can induce cross-topic dependence.

MDF is a reference forecaster. Token and compute budgets are not equalized across strategies. MDF’s sparse evaluated fields may reflect prediction-adapter information loss, rather than generation or training failure (Section 5.5). Matched adapter comparisons, component ablations, and seed variation remain unavailable. We present MDF as a trainable reference, without attributing its scores to an untested mechanism.

## Acknowledgments

We gratefully thank Rui Pan and Heng Wang for helpful discussions. We are especially grateful to Paul Liang, whose detailed comments on earlier drafts substantially improved this paper. We also thank the anonymous reviewers for their detailed feedback.

## References

Anirudh Ajith, Amanpreet Singh, Jay DeYoung, Nadav Kunievsky, Austin C. Kozlowski, Oyvind Tafjord, James Evans, Daniel S. Weld, Tom Hope, and Doug Downey. 2026. PreScience: A dataset and benchmark for scientific forecasting. arXiv preprint arXiv:2602.20459.

Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. 2025. ResearchAgent: Iterative research idea generation over scientific literature with large language models. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6709–6738, Albuquerque, New Mexico. Association for Computational Linguistics.

David M. Blei and John D. Lafferty. 2006. Dynamic topic models. In Proceedings of the 23rd International Conference on Machine Learning, pages 113– 120. Association for Computing Machinery.

Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, and Hai Zhao. 2026. Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 16803–16822, San Diego, California, United States. Association for Computational Linguistics.

Sikun Guo, Amir Hassan Shariatmadari, Guangzhi Xiong, Albert Huang, Myles Kim, Corey M. Williams, Stefan Bekiranov, and Aidong Zhang. 2025. IdeaBench: Benchmarking large language models for research idea generation. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5888–5899. Association for Computing Machinery.

Danny Halawi, Fred Zhang, Yueh-Han Chen, and Jacob Steinhardt. 2024. Approaching human-level forecasting with language models. In Advances in Neural Information Processing Systems, volume 37, pages 50426–50468.

Peter Jansen, Oyvind Tafjord, Marissa Radensky, Pao Siangliulue, Tom Hope, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Daniel S. Weld, and Peter Clark. 2025. CodeScientist: End-to-end semiautomated scientific discovery with code-based experimentation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 13370– 13467, Vienna, Austria. Association for Computational Linguistics.

Ezra Karger, Houtan Bastani, Yueh-Han Chen, Zachary Jacobs, Danny Halawi, Fred Zhang, and Philip E. Tetlock. 2025. ForecastBench: A dynamic benchmark of AI forecasting capabilities. In The Thirteenth International Conference on Learning Representations.

Sihang Li, Jin Huang, Jiaxi Zhuang, Yaorui Shi, Xiaochen Cai, Mingjun Xu, Xiang Wang, Linfeng Zhang, Guolin Ke, and Hengxing Cai. 2025. SciLitLLM: How to adapt LLMs for scientific literature understanding. In The Thirteenth International Conference on Learning Representations.

Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, and Chenhao Tan. 2025. HypoBench: Towards systematic and principled benchmarking for hypothesis generation. arXiv preprint arXiv:2504.11524.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. 2024. The AI scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. 2026. ReasoningBank: Scaling agent self-evolving with reasoning memory. In The Fourteenth International Conference on Learning Representations.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Minju Seo, Jinheon Baek, Seongyun Lee, and Sung Ju Hwang. 2026. Paper2Code: Automating code generation from scientific papers in machine learning. In The Fourteenth International Conference on Learning Representations.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652.

Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. 2026. The ideation–execution gap: Execution outcomes of LLM-generated versus human research ideas. In The Fourteenth International Conference on Learning Representations.

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. 2025. Can LLMs generate novel research ideas? A largescale human study with 100+ NLP researchers. In The Thirteenth International Conference on Learning Representations.

Xuemei Tang, Xufeng Duan, and Zhenguang Cai. 2025. Large language models for automated literature review: An evaluation of reference generation, abstract writing, and review composition. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1602–1617, Suzhou, China. Association for Computational Linguistics.

Dashun Wang, Chaoming Song, and Albert-László Barabási. 2013. Quantifying long-term scientific impact. Science, 342(6154):127–132.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2024. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research.

Jiaxin Wen, Chenglei Si, Yueh-Han Chen, He He, and Shi Feng. 2025. Predicting empirical AI research outcomes with language models. In Advances in Neural Information Processing Systems, volume 38, pages 2988–3005.

Sean Wu, Pan Lu, Yupeng Chen, Jonathan Bragg, Yutaro Yamada, Peter Clark, David Clifton, Philip Torr, James Zou, and Junchi Yu. 2026. Scientific reasoning does not reliably translate into scientific forecasting in frontier AI. arXiv preprint arXiv:2605.22681.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, Yijing Xu, Haoqiang Kang, Ziyan Kuang, Chenhan Yuan, Kailai Yang, Zheheng Luo, Tianlin Zhang, Zhiwei Liu, Guojun Xiong, and 15 others. 2024. FinBen: A holistic financial benchmark for large language models. In Advances in Neural Information Processing Systems, volume 37, pages 95716–95743.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. PIXIU: A comprehensive benchmark, instruction dataset and large language model for finance. In Advances in Neural Information Processing Systems, volume 36, pages 33469–33484.

Haofei Yu, Zhaochen Hong, Zirui Cheng, Kunlun Zhu, Keyang Xuan, Jinwei Yao, Tao Feng, and Jiaxuan You. 2025. ResearchTown: Simulator of human research community. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 73051–73096. PMLR.

Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. 2026. MemSkill: Learning and evolving memory skills for self-evolving agents. arXiv preprint arXiv:2602.02474.

Xinran Zhao, Boyuan Zheng, Chenglei Si, Haofei Yu, Ken Liu, Runlong Zhou, Ruochen Li, Tong Chen, Xiang Li, Yiming Zhang, and Tongshuang Wu. 2025. The Ramon Llull’s thinking machine for automated ideation. arXiv preprint arXiv:2508.19200.

Tianshi Zheng, Zheye Deng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Zihao Wang, and Yangqiu Song. 2025. From automation to autonomy: A survey on large language models in scientific discovery. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17733–17750, Suzhou, China. Association for Computational Linguistics.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing large language models with long-term memory. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(17):19724–19731.

## A Dataset Construction and Episode Manifest

## A.1 Sources, Dates, and Representations

The ingestion pipeline queries the arXiv export API with cat:cs.ML, pages through returned entries, and records each paper in a month-bucketed collection. Records retain the canonical identifier, title, first-submission date, category tags, and abstract or contribution summary. Deduplication by arXiv identifier precedes topic assignment. Cross-listed papers are retained once as papers and may acquire more than one topic membership.

The first-submission timestamp is the canonical splitting date. The last-updated timestamp is metadata, not a replacement for the submission date. For cutoff t, the historical and future membership tests are mutually exclusive: $d ( p ) \leq t$ and $t < d ( p ) \leq e ( t )$ , respectively. These membership rules do not establish that the current abstract had the same wording at first submission. A strict prospective release should therefore preserve versioned text and the date of every generated key point, alongside paper membership.

After identifier deduplication, the current cs.ML corpus contains 63,855 papers. Each of the 52 topics is specified by a name, aliases, and keywords, and assignment is deterministic over the available title/keypoint representation. Of the deduplicated papers, 42,760 are assigned to at least one topic and enter benchmark episodes. Topic overlap is intentional; it permits work such as multimodal retrieval to appear in more than one scientifically meaningful context. Consequently, the per-topic counts sum to 71,081 memberships, or 1.66 memberships per assigned paper. The membership sum is not a count of unique papers.

## A.2 Current Evaluation Slice

The result headers specify start\_month=2024-04, end\_month=2025-09, min\_cutoff\_month=2024-07, top\_k=5, and horizon\_months=3. The last setting is a month-offset parameter: the implemented upper endpoint is the end of cutoff month plus three. The twelve cutoffs are July 2024 through June 2025, each on its first day. The final target therefore ends on September 30, 2025. The configured minimum history is two papers; the actual minimum is 33.

Across 624 episodes, historical pools contain 33–4,530 papers (mean 589.9, median 386.5), and future pools contain 35–1,722 (mean 313.3, median 245.5). Per-topic corpus sizes range from 209 to 6,158, with median 1,026. Table 4 supplies the complete topic-level audit. These values are reconstructed from the result manifests, without treating overlapping topic memberships as independent papers.

Relationship to the earlier snapshot. An earlier benchmark snapshot covered January 2023–June 2025, with 95,276 ingested papers, 61,816 topic-assigned unique papers, and 1,343 eligible windows. Its main experimental subset contained 208 windows. Those statistics describe a different corpus and evaluation slice and are not combined with the current main results. The earlier source and figures remain archived to preserve the provenance of the auxiliary studies.

Reproducibility checks. For each backbone–strategy–judge result, we verify unique topic–cutoff keys, all 624 expected episodes, equal historical and future counts across configurations, and agreement between stored match flags and episode-level Hit@5, Precision@5, and reciprocal rank. These checks are necessary but do not establish corpus identity: equal counts cannot replace comparing sorted paper identifiers. Full verification also requires text or content hashes, topic-rule versions, and generation and evaluation configurations.

Table 4: Current topic manifest. Every topic has twelve cutoffs. Counts are within-topic paper counts; history and target columns give the minimum–maximum across those cutoffs.
<table><tr><td>Topic</td><td>Corpus papers</td><td>History range</td><td>Target range</td></tr><tr><td>3d_embodied</td><td>2348</td><td>329-1788</td><td>468-632</td></tr><tr><td>3d_nerf</td><td>498</td><td>91-406</td><td>92-140</td></tr><tr><td>active_learning</td><td>676</td><td>124-521</td><td>126-169</td></tr><tr><td>anomaly_detection</td><td>1299</td><td>191-972</td><td>267-327</td></tr><tr><td>autonomous_driving</td><td>768</td><td>136-593</td><td>146-190</td></tr><tr><td>code_llm</td><td>518</td><td>72-386</td><td>95-152</td></tr><tr><td>continual_learning</td><td>1023</td><td>171-782</td><td>199-263</td></tr><tr><td>dialogue_conv</td><td>310</td><td>41-217</td><td>55-93</td></tr><tr><td>domain_adaptation</td><td>1796</td><td>316-1415</td><td>381-421</td></tr><tr><td>efficient_finetuning</td><td>1216</td><td>180-931</td><td>244-329</td></tr><tr><td>federated_learning</td><td>2949</td><td>514-2292</td><td>608-720</td></tr><tr><td>generalization_ood</td><td>5819</td><td>794-4295</td><td>1125-1568</td></tr><tr><td>graph_gnn</td><td>3295</td><td>569-2556</td><td>678-786</td></tr><tr><td>image_gen_diffusion</td><td>1350</td><td>229-1057</td><td>265-352</td></tr><tr><td>image_recognition</td><td>1392</td><td>238-1078</td><td>289-329</td></tr><tr><td>image_segmentation</td><td>741</td><td>150-594</td><td>142-185</td></tr><tr><td>in_context_learning</td><td>2307</td><td>375-1727</td><td>470-603</td></tr><tr><td>information_extraction</td><td>209</td><td>44-167</td><td>35-57</td></tr><tr><td>knowledge_distillation</td><td>1366</td><td>205-1013</td><td>257-375</td></tr><tr><td>knowledge_graph</td><td>450</td><td>75-337</td><td>82-121</td></tr><tr><td>llm_agents</td><td>1409</td><td>139-940</td><td>229-469</td></tr><tr><td>1lm_alignment_rlhf</td><td>3418</td><td>456-2447</td><td>596-1002</td></tr><tr><td>1lm_factuality</td><td>1452</td><td>211-1047</td><td>257-405</td></tr><tr><td>1lm_instruction</td><td>988</td><td>131-684</td><td>178-304</td></tr><tr><td>llm_long_context</td><td>597</td><td>73-450</td><td>103-198</td></tr><tr><td>1lm_pretraining</td><td>2085</td><td>304-1509</td><td>401-576</td></tr><tr><td>1lm_reasoning_cot</td><td>666</td><td>57-444</td><td>90-247</td></tr><tr><td>1lm_reasoning_math</td><td>690</td><td>65-464</td><td>107-247</td></tr><tr><td>1lm_safety</td><td>478</td><td>69-369</td><td>92-138</td></tr><tr><td>machine_translation</td><td>527</td><td>93-414</td><td>108-128</td></tr><tr><td>medical_imaging</td><td>1233</td><td>217-915</td><td>245-318</td></tr><tr><td>medical_nlp</td><td>264</td><td>49-189</td><td>43-78</td></tr><tr><td>meta_learning</td><td>991</td><td>148-728</td><td>190-280</td></tr><tr><td>moe</td><td>702</td><td>75-497</td><td>133-205</td></tr><tr><td>molecular_graph</td><td>724</td><td>121-550</td><td>144-179</td></tr><tr><td>nas</td><td>297</td><td>59-232</td><td>47-82</td></tr><tr><td>object_detection</td><td>433</td><td>72-329</td><td>79-110</td></tr><tr><td>protein_structure</td><td>267</td><td>33-198</td><td>49-78</td></tr><tr><td>pruning_sparsity</td><td>1342</td><td>195-1009</td><td>257-357</td></tr><tr><td>quantization</td><td>1211</td><td>175-911</td><td>232-339</td></tr><tr><td>question_answering</td><td>765</td><td>131-596</td><td>137-210</td></tr><tr><td>rag_retrieval</td><td>733</td><td>85-564</td><td>153-221</td></tr><tr><td>recommendation</td><td>1029</td><td>169-769</td><td>178-264</td></tr><tr><td>reinforcement_learning</td><td>6158</td><td>978-4530</td><td>1111-1722</td></tr><tr><td>remote_sensing</td><td>613</td><td>99-465</td><td>118-159</td></tr><tr><td>scientific_ml</td><td>1503</td><td>207-1077</td><td>285-426</td></tr><tr><td>self_supervised</td><td>2306</td><td>394-1787</td><td>481-526</td></tr><tr><td>sgd_training</td><td>1668</td><td>282-1255</td><td>328-448</td></tr><tr><td>speech_audio</td><td>585</td><td>99-425</td><td>96-160</td></tr><tr><td>tabular_ml</td><td>1567</td><td>228-1162</td><td>321-405</td></tr><tr><td>time_series</td><td>2812</td><td>424-2122</td><td>587-692</td></tr><tr><td>vision_language</td><td>1238</td><td>177-945</td><td>228-323</td></tr></table>

## B MDF Architecture and Reference Configuration

Scope of these details. This appendix describes MDF’s architecture and the earlier Qwen3.5-9B reference configuration. The checkpoint evaluated in the main table instead uses Qwen2.5-7B. The judged artifacts do not include its complete training manifest. The hyperparameters below therefore describe the reference configuration, not verified settings for the evaluated checkpoint. Verification requires checkpoint identifiers, training-target dates, run overrides, and adapter hashes. Earlier ablations are not used to infer component gains for the evaluated checkpoint.

Formulation. Let $X _ { c , t }$ be the permitted historical papers and $y _ { j }$ an idea-level description of a target paper. MDF introduces the latent innovation $z = ( b , o , g )$ of Section 4 to organize the mapping from history to these descriptions (e.g. b = “preference optimization for alignment”, $o = \mathtt { E X T E N D } , g =$ “instability of DPO under distribution shift”). A simplifying conditional-independence approximation gives

$$
p ( \boldsymbol { y } _ { 1 } , \dots , \boldsymbol { y } _ { M } \mid \boldsymbol { X } _ { c , t } ) \approx \prod _ { j = 1 } ^ { M } \sum _ { z _ { j } } p ( \boldsymbol { y } _ { j } \mid z _ { j } , \boldsymbol { X } _ { c , t } ) p ( z _ { j } \mid \boldsymbol { X } _ { c , t } ) ,\tag{8}
$$

Algorithm 1 MDF joint inference (Section 4)   
Require: History $X _ { c , t }$ , memory $\mathcal { M } _ { t }$ , pool size C, forecast budget K, blend λ=0.4, trained prior $p _ { \theta } .$   
realization policy $p _ { \psi }$   
Ensure: Ranked idea list $\hat { Y } _ { c , t }$ with at most K entries   
1: $Z \sim p _ { \theta } ( z \mid \mathcal { M } _ { t } )$ ▷ sample C candidate latent innovations from the prior   
2: $S \gets \emptyset$ ▷ scored ideas   
3: for each $z _ { i } \in Z$ do   
4: $s _ { \mathrm { p r i o r } }  \vert z _ { i } \vert ^ { - 1 } \log p _ { \theta } ( z _ { i } \mid \mathcal { M } _ { t } )$ ▷ mean conditional log-probability per token   
5: retrieve evidence from $X _ { c , t } ; \hat { y } _ { i } \sim p _ { \psi } ( y \mid z _ { i } , X _ { c , t } )$ ▷ generate an idea from $z _ { i }$   
6: $\begin{array} { r } { s _ { \mathrm { r e a l i z e } }  \vert \hat { y } _ { i } \vert ^ { - 1 } \log p _ { \psi } ( \hat { y } _ { i } \mid z _ { i } , X _ { c , t } ) } \end{array}$ ▷ mean conditional log-probability per token   
7: $S \gets S \cup \{ ( \hat { y } _ { i } , \lambda s _ { \mathrm { p r i o r } } + ( 1 - \lambda ) s _ { \mathrm { r e a l i z e } } ) \}$ ▷ blend the two scores   
8: end for   
9: return DEDUPLICATEANDTOPK $( S , K )$ ▷ drop near-duplicates, keep the top K

which separates a prior over innovations from a policy that expresses each innovation as an idea. In the implementation, the prior conditions on compact memory $\mathcal { M } _ { t }$ rather than the full historical stream. Papers do not provide labels in the required triple schema, so hindsight extraction supplies pseudo-labels $\tilde { z }$ for supervised training. The prior learns to emit these triples as JSON. At inference, sampling approximates the latent sum, and deduplication produces a ranked list of up to K ideas rather than an exhaustive reconstruction of the target papers. Algorithm 1 uses token-normalized scores, with $| z _ { i } |$ and $| \hat { y } _ { i } |$ denoting scored output-token counts. The training reward uses retrieval depth $R _ { \mathrm { r e w } }$ , distinct from evaluation depth R in Section 3.3.

Backbones and adapters. The reference configuration uses Qwen3.5-9B for both stages, with LoRA adapters $( r { = } 1 6 , \alpha { = } 3 2$ , dropout 0.05, target “all-linear”). The prior uses supervised fine-tuning and the realization policy uses GRPO. A valid held-out temporal evaluation requires training target and reward pools to precede evaluation targets; checking training cutoffs alone is insufficient when horizons overlap. For the evaluated Qwen2.5-7B checkpoint, the training manifest must establish this separation; its score cannot do so.

Hindsight extraction. For each paper in a training episode’s future pool, a frozen extractor LLM (GPT-5.4, temperature 0.2, up to 2 retries) reads its title and abstract, a summary of the selected historical papers, and reference grounding. It returns a triple $\tilde { z } = ( b , o , g )$ as JSON, with the operator checked against the eight allowed values. A second frozen GPT-5.4 call (temperature 0) assesses whether the triple is entailed by the target abstract and whether the gap can be motivated from the permitted history. Triples failing either check are discarded. This procedure supplies hindsight supervision; it does not imply that training targets were available at the historical cutoff. The extractor user template appears in Prompt 2.

Memory. $\mathcal { M } _ { t }$ stores typed entries $( b , o , g ,$ , frequency, recency, utility). New innovations are appended (incrementing frequency on a match); recency decays by 0.9 per elapsed month; utility is an $\mathrm { E M A } \left( \underline { { \alpha } } { = } 0 . 3 \right)$ of delayed feedback. The prior conditions on the top-10 entries ranked by 0.45 recency + 0.35 frequency  + 0.20 utility (frequency normalized by the maximum in the inventory, utility tanh-normalized).

Prior SFT. The reference prior is trained for 3 epochs with learning rate $2 \times 1 0 ^ { - 5 }$ , per-device batch size 4, gradient accumulation 2, maximum sequence length 4096, warmup ratio 0.1, and weight decay 0.01. The target is the innovation JSON, and the objective is token-level negative log-likelihood.

Realization GRPO. For each condition $( z , X _ { c , t } )$ , the policy samples G trajectories $\{ y _ { 1 } , \dotsc , y _ { G } \}$ , scores them with reward $r ( y _ { i } )$ , and forms group-centered advantages $\hat { A } _ { i }$ . In schematic notation, let $\rho _ { i } = p _ { \psi } ( y _ { i } \mid$ $z , X _ { c , t } ) / p _ { 0 \mathrm { l d } } ( y _ { i } \mid z , X _ { c , t } )$ be the importance ratio and s = min $\begin{array} { r l } {  { \big ( \rho _ { i } \hat { A } _ { i } , \mathrm { c l i p } ( \rho _ { i } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i } \big ) } \quad } & { { } } \end{array}$ the clipped surrogate. The objective is

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { G R P O } } ( \psi ) = \mathbb { E } \Big [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } s _ { i } \Big ] - \beta \mathbb { D } _ { \mathrm { K L } } ( p _ { \psi } \| p _ { \mathrm { r e f } } ) . } \end{array}\tag{9}
$$

The reference implementation uses TRL’s unmodified GRPOTrainer, with $G { = } 8$ generations, KL weight $\beta { = } 1 0 ^ { - 3 }$ , the trainer’s default clip range ϵ, learning rate 10−<sup>5</sup>, 3 epochs, per-device batch size 1, gradient accumulation 2, maximum prompt/completion lengths 4096/1024, and LoRA on Qwen3.5-9B (r=16, $\alpha { = } 3 2 )$ . The setting scale\_rewards='none' mean-centers rewards without dividing by the within-group standard deviation, avoiding amplification when reward variance is small. This is a trainer configuration option, not a code modification. The reference runs use seed 0. A multi-seed comparison is unavailable for the evaluated checkpoint, so episode bootstrap intervals do not quantify training variability.

Reward gates. The reference reward $r ( y )$ is zero if any of three checks fails. Well-formedness requires a valid structured output. Grounding requires every cited paper to retrieve a historical neighbor with cosine similarity above 0.3 in the episode’s history index. This check removes citations without a historical neighbor; it does not verify that the neighbor supports the cited claim. Operator consistency requires the declared operator, after the collapse map below, to equal the assigned o; operators in the OTHER bucket pass without this constraint.

Rubric reward and shaping. For a gate-passing rollout, the reward retrieves $R _ { \mathrm { r e w } } = 5$ papers from the training episode’s future pool using SPECTER (allenai-specter, 768 dimensions). This index differs from the 1024-dimensional Voyage index used in evaluation. A judge scores each rollout–paper pair against the topic rubric, and the maximum score in [0, 1] is retained. The judge prompt specifies a one-time 0.2 penalty for violating a must\_not condition, floored at zero; the code does not subtract it again.

The reward also adds 0.1 × max(0, cos -sim) against the nearest retrieved future paper. The rubric score is near-binary (0 or in [0.6, 1.0]), so a group with all-zero rubric scores otherwise has no group-centered learning signal. The shaping term therefore supplies a continuous training signal for gate-passing rollouts. It is not used in benchmark evaluation. The rollout prompt appears in Prompt 3.

Rubric generation and validation. A topic rubric R contains criteria (3–7 matching requirements) and must\_not conditions (0–4 disqualifiers). A frozen GPT-5.4 generates it from positive examples drawn from a training episode’s future papers and negative examples drawn from its history. A separate Qwen3.5-9B judge scores the per-topic positive and negative idea–paper pairs, typically a few dozen in each group. Acceptance requires ROC- $\mathrm { A U C } \geq 0 . 7 0$ and no negative scoring at or above the positive median. The implementation calls the latter failures “leakage hits”; this is a score-separation diagnostic, not evidence that the judge memorized a topic or that training contamination is absent.

An optional rubric-refresh procedure partitions recent rollouts into high- and low-reward pools, regenerates the rubric, and applies the same validation criteria. The procedure replaces the rubric only after validation and records its version history, so changes in training reward can be compared with held-out performance. Refresh is disabled in the reference configuration, which uses a static validated rubric. These training rubrics are distinct from the fixed P/M/S evaluation rubric.

Operators. The extractor emits one of the eight inventory operators; for the operator gate and rubric these collapse to a closed set of four (LIMITATION-EXTENSION, CROSS-DOMAIN-TRANSFER, BENCHMARK-PROPOSAL, METHOD-COMPOSITION) plus an OTHER bucket. The map is: EXTEND → LIMITATION-EXTENSION; TRANSFER, ADAPT → CROSS-DOMAIN-TRANSFER; COMPOSE, SIMPLIFY → METHOD-COMPOSITION; BENCHMARK → BENCHMARK-PROPOSAL; and ANALYZE, SCALE → OTHER. Operators that fall in OTHER pass the operator gate unconstrained.

Joint inference. At inference we sample C=16 innovations from the prior (temperature 0.8); for each we retrieve evidence and generate an idea, rank by $0 . 4 s _ { \mathrm { p r i o r } } + 0 . 6 s _ { \mathrm { r e a l i z e } }$ (per-token-normalized log-probabilities), deduplicate at Jaccard 0.8, and return the top $K { = } 5$ . The full procedure is Algorithm 1 above.

## B.1 Prediction Adapter and Information Preservation

The implementation calls the raw realization text a proposal; an adapter converts it into the benchmark’s structured idea prediction. The local function proposal\_to\_idea\_prediction takes the first nonthinking line as the title, appends the raw body text to the innovation gap to form a rationale truncated at 500 characters, and sets the approach to operator: base\_direction. It does not explicitly fill key terms. A short approach field therefore does not establish that the raw output lacks a method. To trace information loss, a release should retain the raw output, parsed fields, and text passed to the embedding model and judge. Attributing such loss to reinforcement learning also requires testing the same adapter on untrained and trained generators.

## C Forecasting Baselines and Context Budgets

The five baselines share a ranked natural-language output schema: title, rationale, approach, confidence, and optional key terms. Confidence is not a calibrated probability and does not replace the explicit output rank. Prompt templates are retained in Appendix J. Parameters below describe the reference implementation; configuration overrides must be preserved with each run rather than inferred from a strategy name.

Direct Prompting. A single forecasting call reads up to 20 recent historical abstracts. The prompt explicitly states the cutoff and asks for five concrete research ideas. The default forecasting temperature is 0.4. Raw recency selection is a useful baseline because it retains paper-level detail without spending a separate call on compression.

Summary-Augmented Prompting. The first call reads up to 60 recent paper summaries, each truncated to 300 characters, and produces a single paragraph of approximately eight sentences. It is asked to capture dominant themes, methodological trajectories, and recurring open problems rather than list individual papers. A second call receives only this paragraph and the cutoff. Default temperatures are 0.3 for compression and 0.4 for forecasting. No recent raw abstracts are reintroduced at the second stage, distinguishing this strategy from Memory.

Retrieval-Augmented Prompting. The query uses titles and keywords from the recent historical literature. Hybrid semantic and keyword similarity selects up to 20 papers from the allowed history. The forecasting call then conditions on the selected evidence. This is method-side retrieval from history and must not be confused with the evaluation-time retrieval of post-cutoff candidate papers.

Memory-Augmented Prompting. History is split at six months before the cutoff. Up to 60 older papers, represented by 300-character snippets, are condensed into eight memory bullets. Forecasting receives those bullets together with up to 20 recent abstracts. The default compression and forecasting temperatures are 0.3 and 0.4. With insufficient older history, the long-term component is correspondingly limited; no additional pre-April-2024 context is assumed in the current slice.

Topic Trend. Historical papers are grouped into clusters, which are ranked by recent activity. The generator is asked for ideas associated with the selected clusters. This is a generator-dependent baseline, not a deterministic keyword forecast. The implementation can produce fewer than five usable predictions. We report its observed budget completion without filling missing slots after seeing outcomes.

Fairness and interpretation. The compared pipelines vary both information selection and the number of model calls. Their scores therefore do not isolate the effect of compression at equal compute. A controlled comparison would hold input tokens, output tokens, and generation calls fixed while varying only the historical representation. The current benchmark instead evaluates the complete reference strategies under common windows and a common output budget.

## D Evaluation Protocol and Measurement Details

## D.1 Retrieval, Rubric, and Crediting

Each prediction is embedded with the same model used for target-paper embeddings (VOYAGE-3-LARGE, 1024 dimensions). The top 10 candidates by cosine similarity are evaluated. The prompt includes predicted title, rationale, approach, and key terms and the candidate title and abstract. Temperature is fixed at zero in the reference judge implementation. GPT-4.1-mini and Qwen3.5-9B are reported separately; the latter is identified by the serving alias qwen35-9b-judge in the exported header.

The rubric distinguishes problem agreement, method agreement, and the extent to which the paper realizes the particular proposed idea. Each dimension takes integer values 0–3. A topical similarity without method agreement should not pass $P + M \ge 5$ . A core idea with differing implementation details can pass $S \geq 2 ; S = 3$ asks for a closer realization. The archived prompt listings provide concrete anchors, but prompt hashes and server versions should accompany the final release because a nominal judge model does not uniquely identify its evaluation behavior.

Candidates are judged independently. For credit assignment, ideas are processed in rank order, selecting the first passing candidate not previously credited in the same episode. This produces one binary credit per prediction and at most one credit per paper. Hit@5 is one if any credit exists, Precision@5 divides credited predictions by five, and the reciprocal rank is the inverse rank of the first credit, or zero for no credit. MRR averages this value over the complete 624-episode manifest.

Representative scores are not the candidate set. The compact results retain a P/M/S triple from the credited candidate, or from a representative candidate when none is credited. Applying the gate to that triple alone does not reproduce candidate selection. Even a passing candidate may receive no final credit if its paper has already been used. We therefore reconstruct the main table from final match flags and verify it against the stored metrics. Changing gates, retrieval depth, or paper pools requires all candidate judgments or a new retrieval/judging pass, not only the compact per-prediction triples.

Failure handling. The compact exports show no judge parse failures for GPT-4.1-mini, but do record failures for Qwen3.5-9B (Section 5.4); the earlier candidate audit does not establish failure-free execution. Missing outputs receive zero credit under the fixed five-slot budget. This is distinct from a judge failure or a retrieval miss. A reproducible run should log missing outputs, judge failures, and retrieval misses separately, including retries and failures remaining after retries. Affected windows must remain in the denominator. The reference parser and its prompt must be versioned together.

## D.2 Supplementary Metrics

MRR averages the reciprocal rank of the first credited idea, with zero for episodes without credit. Historical novelty first averages nearest-neighbor embedding distance over emitted ideas within an episode, then averages those episode values. The evaluator assigns an empty episode zero novelty by convention; this is not a judgment about an absent idea’s originality. Neither metric replaces joint reporting of Hit@5 and Precision@5.

The reference implementation also records a SOFT score: the mean of $( P + M + S ) / 9$ over credited predictions, set to zero when there are none. It is conditional on credited matches and thus should not be interpreted as independent evidence of forecasting quality. COVERAGE partitions target-paper embeddings into five KMeans clusters and counts the fraction containing a credited paper. It depends on the chosen clustering and judge. Table 7 reports both metrics and topic-clustered intervals for every configuration and judge; these supplementary diagnostics are not interchangeable with realization or ranking performance.

## D.3 Uncertainty and Sensitivity

Table reconstruction uses NumPy’s seed-0 generator to draw 10,000 samples of the 52 topics with replacement. All twelve cutoffs for a sampled topic remain together. The 2.5th and 97.5th percentiles define the interval. With equal episode counts per topic, the mean of topic means equals the overall episode mean. Paired gate-sensitivity analyses also use 10,000 topic resamples and seed 0, but a different random-number implementation; their last-digit interval endpoints may therefore differ from independently recomputed intervals.

The strict-gate Hit@5 comparisons in Figure 4 use all candidates from the server export. No strict-gate precision from its aggregate table is substituted for deduplicated Precision@5: counting every prediction with any passing candidate ignores the one-paper-one-credit rule. Candidate multiplicity is explicitly a different diagnostic and is never labeled precision.

Recall and calibration. Retrieval recall cannot be measured by judging only retrieved candidates. Likewise, judge–judge agreement does not measure agreement with human labels or establish a direction of bias. An end-to-end recall study would need to sample beyond retrieved candidates, align human and judge instructions, and specify an estimator for the sampling design. The auxiliary historical human study in Appendix H motivates this requirement but is not a new calibration experiment.

## E Full Current Results

Table 5 reports 42 judge-specific results: 21 configurations on 624 episodes. GPT and Qwen denote GPT-4.1-mini and Qwen3.5-9B judging, distinct from the Backbone column. Qwen-judge values remain provisional (Section 5.4). Topic-clustered intervals quantify episode-sampling uncertainty; they do not capture variation from training seeds, generator reruns, or shard selection.

Novelty does not use judge decisions. Its independently exported row means differ by less than $1 0 ^ { - 5 }$ and agree at the displayed precision; the main table uses the primary export for its single Novelty column.

The table excludes earlier 208-window results, Qwen3.5-9B ablations, and GPT-5.4 generations. These remain archived, not evidence for the current Qwen2.5-7B MDF. A judge shift measured on one archived generator is not applied as a universal correction.

Table 5: Default-gate scores and 95% intervals. Novelty is historical embedding distance, not a human originality score. Topic Trend retains its incomplete five-slot outputs.
<table><tr><td rowspan=1 colspan=12>Backbone      Strategy      Judge      Hit@5 [CI]        Precision@5 [CI]   MRR  Novelty</td></tr><tr><td rowspan=1 colspan=6>GPT-4.1        Summary    GPT   0.756 [0.696, 0.814]</td><td rowspan=1 colspan=6>0.297 [0.258, 0.338]  0.519   0.181</td></tr><tr><td rowspan=1 colspan=5>GPT-4.1        Summary     Qwen  0.822 [</td><td rowspan=1 colspan=1>0.772, 0.865]</td><td rowspan=1 colspan=2>0.323 [</td><td rowspan=1 colspan=1>0.288, 0.358]</td><td rowspan=1 colspan=3>0.548   0.181</td></tr><tr><td rowspan=1 colspan=5>GPT-4.1        Memory     GPT   0.729 [</td><td rowspan=1 colspan=1>0.671, 0.785]</td><td rowspan=1 colspan=2>0.267 [</td><td rowspan=1 colspan=1>0.230, 0.306]</td><td rowspan=1 colspan=3>0.456   0.160</td></tr><tr><td rowspan=1 colspan=5>GPT-4.1        Memory     Qwen  0.772 [</td><td rowspan=1 colspan=1>0.729, 0.816]</td><td rowspan=1 colspan=2>0.292 [</td><td rowspan=1 colspan=1>0.263, 0.323]</td><td rowspan=1 colspan=3>0.464   0.160</td></tr><tr><td rowspan=1 colspan=5>GPT-4.1        Retrieval     GPT   0.696 [</td><td rowspan=1 colspan=1>0.633, 0.756]</td><td rowspan=1 colspan=2>0.257 [</td><td rowspan=1 colspan=1>0.219, 0.298]</td><td rowspan=1 colspan=3>0.465   0.130</td></tr><tr><td rowspan=1 colspan=4>GPT-4.1        Retrieval     Qwen</td><td rowspan=1 colspan=1>0.692 [</td><td rowspan=1 colspan=1>0.641, 0.742]</td><td rowspan=1 colspan=2>0.237 [</td><td rowspan=1 colspan=1>0.209, 0.269]</td><td rowspan=1 colspan=3>0.396   0.130</td></tr><tr><td rowspan=1 colspan=4>GPT-4.1        Topic Trend  GPT</td><td rowspan=1 colspan=1>0.625 [</td><td rowspan=1 colspan=1>0.551, 0.697]</td><td rowspan=1 colspan=2>0.135 [</td><td rowspan=1 colspan=1>0.118, 0.154]</td><td rowspan=1 colspan=3>0.598   0.238</td></tr><tr><td rowspan=1 colspan=4>GPT-4.1        Topic Trend  Qwen</td><td rowspan=1 colspan=1>0.655 [</td><td rowspan=1 colspan=1>0.596, 0.712]</td><td rowspan=1 colspan=2>0.148 [</td><td rowspan=1 colspan=1>0.133, 0.163]</td><td rowspan=1 colspan=3>0.626   0.238</td></tr><tr><td rowspan=1 colspan=2>GPT-4.1</td><td rowspan=1 colspan=2>Direct        GPT</td><td rowspan=1 colspan=1>0.487 [</td><td rowspan=1 colspan=1>0.420, 0.554]</td><td rowspan=1 colspan=2>0.149 [</td><td rowspan=1 colspan=1>0.123, 0.176]</td><td rowspan=1 colspan=3>0.283   0.127</td></tr><tr><td rowspan=1 colspan=2>GPT-4.1</td><td rowspan=1 colspan=2>Direct        Qwen</td><td rowspan=1 colspan=1>0.490 [</td><td rowspan=1 colspan=1>0.438, 0.543]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.137 [</td><td rowspan=1 colspan=1>0.120, 0.154]</td><td rowspan=1 colspan=3>0.266   0.127</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=2>Summary    GPT</td><td rowspan=1 colspan=1>0.949 [</td><td rowspan=1 colspan=1>0.921, 0.973]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.528 [</td><td rowspan=1 colspan=1>0.486, 0.569]</td><td rowspan=1 colspan=3>0.686   0.189</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=2>Summary    Qwen</td><td rowspan=1 colspan=1>0.968 [</td><td rowspan=1 colspan=1>0.949, 0.984]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.563 [</td><td rowspan=1 colspan=1>0.526, 0.599]</td><td rowspan=1 colspan=3>0.707   0.189</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=2>Memory     GPT</td><td rowspan=1 colspan=1>0.869 [</td><td rowspan=1 colspan=1>0.824, 0.907]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.412 [</td><td rowspan=1 colspan=1>0.371, 0.453]</td><td rowspan=1 colspan=3>0.612   0.161</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=2>Memory     Qwen</td><td rowspan=1 colspan=1>0.917 [</td><td rowspan=1 colspan=1>0.881, 0.947]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.454 [</td><td rowspan=1 colspan=1>0.417, 0.491]</td><td rowspan=1 colspan=3>0.626   0.161</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.769 [</td><td rowspan=1 colspan=1>0.708, 0.825]</td><td rowspan=1 colspan=2>0.338 [</td><td rowspan=1 colspan=1>0.293, 0.384]</td><td rowspan=1 colspan=3>0.524   0.112</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.837 [</td><td rowspan=1 colspan=1>0.792, 0.878]</td><td rowspan=1 colspan=2>0.372 [</td><td rowspan=1 colspan=1>0.335, 0.410]</td><td rowspan=1 colspan=3>0.562   0.112</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=1>Topic Trend</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.745 [</td><td rowspan=1 colspan=1>0.668, 0.817]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.153 [</td><td rowspan=1 colspan=1>0.137, 0.168]</td><td rowspan=1 colspan=3>0.742   0.297</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=1>Topic Trend</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.779 [</td><td rowspan=1 colspan=1>0.718, 0.835]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.163 [</td><td rowspan=1 colspan=1>0.149, 0.176]</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.297</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=1>Direct</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.571 [</td><td rowspan=1 colspan=1>0.511, 0.628]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.173 [</td><td rowspan=1 colspan=1>0.149, 0.197]</td><td rowspan=1 colspan=1>0.301</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.105</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=1>Direct</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.628 [</td><td rowspan=1 colspan=1>0.577, 0.678]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.189 [</td><td rowspan=1 colspan=1>0.170, 0.209]</td><td rowspan=1 colspan=1>0.323</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.105</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Summary</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.954 [</td><td rowspan=1 colspan=1>0.925, 0.978]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.553 [</td><td rowspan=1 colspan=1>0.511, 0.594]</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.201</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Summary</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.973 [</td><td rowspan=1 colspan=1>0.954, 0.987]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.613 [</td><td rowspan=1 colspan=1>0.575, 0.650]</td><td rowspan=1 colspan=1>0.737</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.201</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Memory</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.913 [</td><td rowspan=1 colspan=1>0.877, 0.944]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.440 [</td><td rowspan=1 colspan=1>0.400, 0.481]</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.178</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Memory</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.962 [</td><td rowspan=1 colspan=1>0.941, 0.979]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.504 [</td><td rowspan=1 colspan=1>0.467, 0.542]</td><td rowspan=1 colspan=1>0.648</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.178</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.854 [</td><td rowspan=1 colspan=1>0.806, 0.897]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.396 [</td><td rowspan=1 colspan=1>0.348, 0.445]</td><td rowspan=1 colspan=1>0.555</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.145</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.886 [</td><td rowspan=1 colspan=1>0.853, 0.918]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.409 [</td><td rowspan=1 colspan=1>0.371, 0.448]</td><td rowspan=1 colspan=1>0.563</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.145</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Topic Trend</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.784 [</td><td rowspan=1 colspan=1>0.720, 0.845]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.167 [</td><td rowspan=1 colspan=1>0.153, 0.182]</td><td rowspan=1 colspan=1>0.768</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.288</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-14B</td><td rowspan=1 colspan=1>Topic Trend</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.825 [</td><td rowspan=1 colspan=1>0.774, 0.872]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.189 [</td><td rowspan=1 colspan=1>0.174, 0.205]</td><td rowspan=1 colspan=1>0.815</td><td rowspan=1 colspan=2>0.288</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-14B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Direct</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.615 [</td><td rowspan=1 colspan=1>0.556, 0.675]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.201 [</td><td rowspan=1 colspan=1>0.172, 0.231]</td><td rowspan=1 colspan=1>0.321</td><td rowspan=1 colspan=2>0.134</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-14B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Direct</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.654 [</td><td rowspan=1 colspan=1>0.604, 0.704]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.217 [</td><td rowspan=1 colspan=1>0.193, 0.242]</td><td rowspan=1 colspan=1>0.330</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.134</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Summary</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.532 [</td><td rowspan=1 colspan=1>0.468, 0.598]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.172 [</td><td rowspan=1 colspan=1>0.143, 0.203]</td><td rowspan=1 colspan=2>0.316</td><td rowspan=1 colspan=1>0.190</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Summary</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.498 [</td><td rowspan=1 colspan=1>0.441, 0.558]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.144 [</td><td rowspan=1 colspan=1>0.122, 0.167]</td><td rowspan=1 colspan=2>0.233</td><td rowspan=1 colspan=1>0.190</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Memory</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.471 [</td><td rowspan=1 colspan=1>0.407, 0.537]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.134 [</td><td rowspan=1 colspan=1>0.111, 0.158]</td><td rowspan=1 colspan=2>0.274</td><td rowspan=1 colspan=1>0.149</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Memory</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.316 [</td><td rowspan=1 colspan=1>0.264, 0.370]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.077 [</td><td rowspan=1 colspan=1>0.063, 0.091]</td><td rowspan=1 colspan=2>0.159</td><td rowspan=1 colspan=1>0.149</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-9B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.454 [</td><td rowspan=1 colspan=1>0.378, 0.530]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.130 [</td><td rowspan=1 colspan=1>0.103, 0.158]</td><td rowspan=1 colspan=2>0.267</td><td rowspan=1 colspan=1>0.139</td></tr><tr><td rowspan=1 colspan=2>Qwen3.5-9B</td><td rowspan=1 colspan=1>Retrieval</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.277 [</td><td rowspan=1 colspan=1>0.228, 0.327]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.066 [</td><td rowspan=1 colspan=1>0.054, 0.078]</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.139</td></tr><tr><td rowspan=1 colspan=2>Qwen3.5-9B</td><td rowspan=1 colspan=1>Topic Trend</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.340 [</td><td rowspan=1 colspan=1>0.276, 0.407]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.072 [</td><td rowspan=1 colspan=1>0.058, 0.086]</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.229</td></tr><tr><td rowspan=1 colspan=2>Qwen3.5-9B</td><td rowspan=1 colspan=1>Topic Trend</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.292 [</td><td rowspan=1 colspan=1>0.245, 0.340]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.061 [</td><td rowspan=1 colspan=1>0.050, 0.072]</td><td rowspan=1 colspan=1>0.277</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.229</td></tr><tr><td rowspan=1 colspan=2>Qwen3.5-9B</td><td rowspan=1 colspan=1>Direct</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.226 [</td><td rowspan=1 colspan=1>0.175, 0.282]</td><td rowspan=1 colspan=2>0.057 [</td><td rowspan=1 colspan=1>0.043, 0.073]</td><td rowspan=1 colspan=2>0.133</td><td rowspan=1 colspan=1>0.138</td></tr><tr><td rowspan=1 colspan=2>Qwen3.5-9B</td><td rowspan=1 colspan=1>Direct</td><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>0.131 [</td><td rowspan=1 colspan=1>0.101, 0.165]</td><td rowspan=1 colspan=2>0.028 [</td><td rowspan=1 colspan=1>0.021, 0.035]</td><td rowspan=1 colspan=3>0.062   0.138</td></tr><tr><td rowspan=1 colspan=2>Qwen2.5-7B</td><td rowspan=1 colspan=1>MDF</td><td rowspan=1 colspan=1>GPT</td><td rowspan=1 colspan=1>0.545 [</td><td rowspan=1 colspan=1>0.470, 0.620]</td><td rowspan=1 colspan=2>0.171 [</td><td rowspan=1 colspan=1>0.138, 0.206]</td><td rowspan=1 colspan=3>0.310   0.200</td></tr><tr><td rowspan=1 colspan=4>Qwen2.5-7B   MDF         Qwen</td><td rowspan=1 colspan=1>0.296 [</td><td rowspan=1 colspan=1>0.234, 0.362]</td><td rowspan=1 colspan=2>0.080 [</td><td rowspan=1 colspan=1>0.060, 0.102]</td><td rowspan=1 colspan=3>0.158   0.200</td></tr></table>

## E.1 Qwen3.5 Backbone: Selection and Verification

The Qwen3.5-9B export contains generation outputs and separate GPT-4.1-mini and Qwen3.5-9B judgments for all five strategies. Its generation service uses the alias gpt-4.1-qwen35; this is a serving identifier, not the GPT-4.1 backbone. All 624 topic–cutoff keys, cutoff dates, target endpoints, and historical/target counts agree with the original cohort.

Whole-episode selection. Summary contains 768 episode records with 624 unique keys. The 144 overlapping keys have different generated ideas, not simply repeated judgments of the same text. We sort generation filenames lexicographically and retain the first whole episode record, including an empty output when present. This gives precedence to fix0–fix2, then m0–m5, then s0–s1. Both judges use the corresponding selected source shard. Ideas, ranks, and candidate sets are never combined across generation versions. This rule makes the analysis reproducible, but does not establish which run the server intended as canonical. The analysis manifest records per-episode filenames and source-file hashes.

Selection sensitivity. Using the last rather than first file for overlapping episodes changes Summary Hit@5 from 0.532 to 0.543 under GPT-4.1-mini and from 0.498 to 0.508 under Qwen3.5-9B. These values reflect alternative selections of generated outputs, not confidence bounds. Both choices leave Summary below GPT-4.1 and Qwen2.5. We report the first-file convention consistently and retain all 624 episodes, unlike aggregate summaries that omit empty outputs. The other four strategies have no duplicate episode records.

Execution and candidate alignment. The selected outputs contain 141,850 candidate judgments per judge. GPT-4.1-mini has no recorded parse failures; the Qwen judge has 20 null judgments across 18 strategy–episode records, with no entirely failed episode (Table 6). These nulls receive no credit and remain unresolved. The two judges score matching ranked prediction titles, but 57 of 14,185 predictions have one candidate replaced between exports. Pair-level agreement must therefore be computed on shared candidate identifiers; the result columns do not compare judges on exactly identical candidate sets. This export does not repair the Qwen-judge failures in the original configurations.

Threshold check. Reapplying the stricter S ≥ 3 gate to all stored candidates reduces Qwen3.5 Summary Hit@5 from 0.532 to 0.050 under GPT-4.1-mini, and from 0.498 to 0.191 under the Qwen judge. The relative ordering of judges therefore reverses with the specificity threshold in this example. These are descriptive, fixed-output sensitivity results, not a calibration of either judge; the Qwen-side null judgments remain unresolved.

Table 6: Qwen3.5-9B generation and evaluation audit. Every strategy retains 624 episodes. Failed judgments count candidate pairs, not missing ideas.
<table><tr><td>Strategy</td><td>Ideas</td><td>Empty episodes</td><td>Full five-slot episodes</td><td>GPT failures</td><td>Qwen failures</td></tr><tr><td>Summary</td><td>3,110</td><td>2</td><td>622</td><td>0</td><td>7</td></tr><tr><td>Memory</td><td>3,095</td><td>5</td><td>619</td><td>0</td><td>7</td></tr><tr><td>Retrieval</td><td>3,085</td><td>7</td><td>617</td><td>0</td><td>3</td></tr><tr><td>Topic Trend</td><td>1,787</td><td>12</td><td>79</td><td>0</td><td>2</td></tr><tr><td>Direct</td><td>3,108</td><td>0</td><td>621</td><td>0</td><td>1</td></tr></table>

## E.2 Supplementary Soft and Coverage Scores

Table 7 retains the two additional evaluator outputs for the same 624 episodes. Soft averages normalized rubric scores over credited matches within each window (zero for no match); Coverage measures the fraction of target-paper clusters reached by credited matches. The former conditions on passing the gate, while the latter depends on the five-cluster partition. They provide diagnostics, not independent measures of scientific quality.

Table 7: Supplementary scores and 95% topic-clustered intervals. Both judges evaluate the same generated predictions. Topic Trend uses the as-run outputs, without padding.
<table><tr><td></td><td></td><td colspan="2">GPT-4.1-mini judge</td><td colspan="2">Qwen3.5-9B judge</td></tr><tr><td>Backbone</td><td>Strategy</td><td>Soft [CI]</td><td>Coverage [CI]</td><td>Soft [CI]</td><td>Coverage [CI]</td></tr><tr><td>GPT-4.1</td><td>Summary</td><td>0.605 [0.554, 0.652]</td><td>0.247 [0.218, 0.277]</td><td>0.700 [0.657, 0.738]</td><td>0.269 [0.243, 0.293]</td></tr><tr><td>GPT-4.1</td><td>Memory</td><td>0.581 [0.536, 0.627]</td><td>0.221 [0.194, 0.248]</td><td>0.650 [0.612, 0.688]</td><td>0.246 [0.223, 0.271]</td></tr><tr><td>GPT-4.1</td><td>Retrieval</td><td>0.552 [0.501, 0.601]</td><td>0.202 [0.179, 0.226]</td><td>0.584 [0.541, 0.625]</td><td>0.192 [0.174, 0.211]</td></tr><tr><td>GPT-4.1</td><td>Topic Trend</td><td>0.497 [0.439, 0.555]</td><td>0.132 [0.116, 0.149]</td><td>0.557 [0.506, 0.607]</td><td>0.144 [0.130, 0.158]</td></tr><tr><td>GPT-4.1</td><td>Direct</td><td>0.388 [0.335, 0.443]</td><td>0.127 [0.107, 0.149]</td><td>0.417 [0.373, 0.462]</td><td>0.124 [0.108, 0.139]</td></tr><tr><td>Qwen2.5-7B</td><td>Summary</td><td>0.765 [0.741, 0.786]</td><td>0.394 [0.368, 0.421]</td><td>0.838 [0.820, 0.853]</td><td>0.424 [0.398, 0.449]</td></tr><tr><td>Qwen2.5-7B</td><td>Memory</td><td>0.698 [0.662, 0.730]</td><td>0.324 [0.295, 0.351]</td><td>0.786 [0.754, 0.814]</td><td>0.355 [0.328, 0.381]</td></tr><tr><td>Qwen2.5-7B</td><td>Retrieval</td><td>0.617 [0.568, 0.661]</td><td>0.251 [0.222, 0.279]</td><td>0.720 [0.679, 0.759]</td><td>0.279 [0.256, 0.303]</td></tr><tr><td>Qwen2.5-7B</td><td>Topic Trend</td><td>0.599 [0.536, 0.659]</td><td>0.152 [0.137, 0.167]</td><td>0.673 [0.618, 0.726]</td><td>0.162 [0.148, 0.174]</td></tr><tr><td>Qwen2.5-7B</td><td>Direct</td><td>0.456 [0.408, 0.503]</td><td>0.149 [0.131, 0.168]</td><td>0.548 [0.503, 0.592]</td><td>0.166 [0.150, 0.182]</td></tr><tr><td>Qwen2.5-14B</td><td>Summary</td><td>0.768 [0.744, 0.787]</td><td>0.415 [0.389, 0.440]</td><td>0.838 [0.821, 0.853]</td><td>0.454 [0.429, 0.478]</td></tr><tr><td>Qwen2.5-14B</td><td>Memory</td><td>0.735 [0.706, 0.760]</td><td>0.344 [0.317, 0.372]</td><td>0.827 [0.806, 0.845]</td><td>0.385 [0.361, 0.408]</td></tr><tr><td>Qwen2.5-14B</td><td>Retrieval</td><td>0.685 [0.646, 0.720]</td><td>0.294 [0.266, 0.324]</td><td>0.762 [0.732, 0.789]</td><td>0.313 [0.290, 0.337]</td></tr><tr><td>Qwen2.5-14B</td><td>Topic Trend</td><td>0.633 [0.581, 0.685]</td><td>0.164 [0.150, 0.178]</td><td>0.712 [0.666, 0.757]</td><td>0.184 [0.171, 0.196]</td></tr><tr><td>Qwen2.5-14B</td><td>Direct</td><td>0.493 [0.445, 0.541]</td><td>0.173 [0.151, 0.195]</td><td>0.566 [0.523, 0.609]</td><td>0.192 [0.172, 0.213]</td></tr><tr><td>Qwen3.5-9B</td><td>Summary</td><td>0.421 [0.370, 0.473]</td><td>0.153 [0.130, 0.178]</td><td>0.427 [0.377, 0.478]</td><td></td></tr><tr><td>Qwen3.5-9B</td><td>Memory</td><td>0.373 [0.321, 0.425]</td><td>0.121 [0.102, 0.141]</td><td>0.268 [0.224, 0.315]</td><td>0.130 [0.112, 0.148]</td></tr><tr><td>Qwen3.5-9B</td><td>Retrieval</td><td>0.358 [0.298, 0.418]</td><td>0.115 [0.093, 0.138]</td><td>0.238 [0.196, 0.281]</td><td>0.072 [0.060, 0.086] 0.063 [0.053, 0.075]</td></tr><tr><td>Qwen3.5-9B</td><td>Topic Trend</td><td>0.269 [0.218, 0.323]</td><td>0.071 [0.058, 0.085]</td><td>0.249 [0.209, 0.291]</td><td>0.061 [0.050, 0.071]</td></tr><tr><td>Qwen3.5-9B</td><td>Direct</td><td>0.179 [0.138, 0.223]</td><td>0.054 [0.041, 0.068]</td><td>0.113 [0.087, 0.141]</td><td>0.028 [0.021, 0.035]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-7B</td><td>MDF</td><td>0.427 [0.367, 0.485]</td><td>0.138 [0.115, 0.161]</td><td>0.253 [0.199, 0.309]</td><td>0.071 [0.054, 0.089]</td></tr></table>

## F Outcome-Blind Generality Analysis

## F.1 Sampling and Rubric

The outcome-blind study covers GPT-4.1 and the two Qwen2.5 backbones across five strategies, plus MDF: 16 configurations, excluding Qwen3.5 generation. Using seed 0, it samples one prediction per topic per configuration, giving 52 predictions per configuration and 832 overall. Sampling spans cutoffs and ranks rather than selecting only successful predictions. GPT-4.1-mini at temperature zero receives only title, rationale, and approach; future papers, match decisions, model identity, and strategy identity are withheld. Withholding these fields controls the information presented, but writing style may still reveal model identity. The addition of Qwen3.5 to the main table does not extend this blind sample.

Four dimensions are scored from 0 to 3: (i) problem specificity, from a broad area to a precise task and failure mode; (ii) method specificity, from no concrete mechanism to an architecture, algorithm, or procedure; (iii) scope specificity, from no setting to a stated dataset, domain, modality, or scale; and (iv) testability, from no stated check to a measurable outcome or comparison. Generality is 12 minus the sum. We retain the full dimension-wise results in Table 8 rather than treating word count as a sufficient measure of specificity.

## F.2 Associations and Common Support

The outcome-linked summary groups 832 assessed forecasts into generality bins [0, 3), [3, 5), [5, 7), and [7, 13), with sample sizes 190, 170, 329, and 143. Their match rates are 0.2053, 0.2294, 0.3465, and 0.4056. This is a forecast-level association, not Hit@5, and it must not be plotted as four episode-level benchmark scores. The overall correlation is 0.17, or 0.21 without MDF. Individual forecast identifiers linking the ratings to outcomes should be retained with the analysis; aggregate ratings alone do not reconstruct that join.

Table 8: Outcome-blind assessment. The first four scores measure specificity (0–3 each); the final column is generality (12 minus their sum). Each row has one sampled forecast from every topic.
<table><tr><td>Backbone</td><td>Strategy</td><td>n</td><td>Problem</td><td>Method</td><td>Scope</td><td>Testability</td><td>Generality</td></tr><tr><td>GPT-4.1</td><td>Summary</td><td>52</td><td>2.52</td><td>2.00</td><td>1.79</td><td>2.12</td><td>3.58</td></tr><tr><td>GPT-4.1</td><td>Memory</td><td>52</td><td>2.60</td><td>2.21</td><td>2.29</td><td>2.40</td><td>2.50</td></tr><tr><td>GPT-4.1</td><td>Retrieval</td><td>52</td><td>2.83</td><td>2.44</td><td>2.29</td><td>2.58</td><td>1.87</td></tr><tr><td>GPT-4.1</td><td>Topic Trend</td><td>52</td><td>2.42</td><td>2.04</td><td>2.12</td><td>2.37</td><td>3.06</td></tr><tr><td>GPT-4.1</td><td>Direct</td><td>52</td><td>2.94</td><td>2.42</td><td>2.58</td><td>2.98</td><td>1.08</td></tr><tr><td>Qwen2.5-7B</td><td>Summary</td><td>52</td><td>1.87</td><td>1.52</td><td>0.98</td><td>1.06</td><td>6.58</td></tr><tr><td>Qwen2.5-7B</td><td>Memory</td><td>52</td><td>2.00</td><td>1.75</td><td>1.27</td><td>1.38</td><td>5.60</td></tr><tr><td>Qwen2.5-7B</td><td>Retrieval</td><td>52</td><td>2.10</td><td>1.83</td><td>1.52</td><td>1.52</td><td>5.04</td></tr><tr><td>Qwen2.5-7B</td><td>Topic Trend</td><td>52</td><td>2.00</td><td>1.67</td><td>1.48</td><td>1.79</td><td>5.06</td></tr><tr><td>Qwen2.5-7B</td><td>Direct</td><td>52</td><td>2.23</td><td>1.88</td><td>1.75</td><td>2.27</td><td>3.87</td></tr><tr><td>Qwen2.5-14B</td><td>Summary</td><td>52</td><td>1.81</td><td>1.54</td><td>1.06</td><td>1.12</td><td>6.48</td></tr><tr><td>Qwen2.5-14B</td><td>Memory</td><td>52</td><td>2.02</td><td>1.67</td><td>1.27</td><td>1.13</td><td>5.90</td></tr><tr><td>Qwen2.5-14B</td><td>Retrieval</td><td>52</td><td>2.08</td><td>1.81</td><td>1.33</td><td>1.17</td><td>5.62</td></tr><tr><td>Qwen2.5-14B</td><td>Topic Trend</td><td>52</td><td>1.83</td><td>1.37</td><td>1.17</td><td>1.40</td><td>6.23</td></tr><tr><td>Qwen2.5-14B</td><td>Direct</td><td>52</td><td>2.38</td><td>2.00</td><td>1.81</td><td>2.33</td><td>3.48</td></tr><tr><td>MDF</td><td>MDF</td><td>52</td><td>1.50</td><td>1.06</td><td>0.90</td><td>0.56</td><td>7.98</td></tr></table>

GPT-4.1 and Qwen2.5 have markedly different generality distributions. Within a narrow generality band, the backbones share little overlap and few comparable forecasts. Reweighting Qwen2.5 forecasts to the GPT-4.1 distribution is therefore unstable. We do not report a causal decomposition or a percentage of the backbone gap explained by generality. A stronger test would produce paired variants of the same forecast with differing specificity, independently verify that their core content is preserved, and evaluate them on the same candidates without outcome-informed rewriting.

## F.3 Matching Breadth and Gate Sensitivity

For each prediction, retrieved match multiplicity is the count of its ten retrieved candidates passing the gate. This is a truncated lower bound on matches in the target pool. Mean Summary multiplicity under the primary judge is 0.565, 1.470, and 1.642 for GPT-4.1, Qwen2.5-7B, and Qwen2.5-14B. These descriptive means include zero-match predictions. The supplied paired-multiplicity implementation skips some zero-match comparisons; its paired differences and intervals are excluded from this paper.

The stricter-gate contrasts in Figure 4 instead concern the existence of any passing candidate per window and retain zero-hit windows. These contrasts assess sensitivity to the matching rule, but do not identify the effect of intrinsic specificity. In particular, the matching rubric evaluates realization relative to a candidate paper; it does not independently measure how many commitments the forecast made before the paper was shown.

Pool-size stratification is also descriptive. At high Hit@5, a larger pool may produce a ceiling effect, reducing between-model gaps even when both gain more matching opportunities. We therefore do not use a narrowing gap at larger pools to reject the generality explanation. Using shared windows holds the target pool fixed across models, but does not hold forecast breadth fixed.

## G Output Validity and Learned-Forecaster Diagnostics

## G.1 Budget Completion and Reuse

The original 16 configurations have no empty prediction windows; the five Qwen3.5 strategies add 2, 5, 7, 12, and 0 empty windows for Summary, Memory, Retrieval, Topic Trend, and Direct, respectively. None of the 21 configurations has within-window identical-title duplicates. Table 6 reports Qwen3.5’s output counts, and Table 9 compares Topic Trend across backbones. Repetition across adjacent windows is not inherently invalid because history and targets overlap. We report it as a diagnostic, not a reason to remove outputs after observing their scores.

Table 9: Topic Trend output validity. Reuse counts repeated titles beyond their first occurrence across the run; it is not within-window duplication
<table><tr><td>Backbone</td><td>Predictions</td><td>Full-budget windows</td><td>Repeated titles</td><td>Reuse rate</td></tr><tr><td>GPT-4.1</td><td>2,738</td><td>379/624</td><td>831</td><td>30.4%</td></tr><tr><td>Qwen2.5-7B</td><td>2,726</td><td>375/624</td><td>1,270</td><td>46.6%</td></tr><tr><td>Qwen2.5-14B</td><td>2,738</td><td>379/624</td><td>1,205</td><td>44.0%</td></tr><tr><td>Qwen3.5-9B</td><td>1,787</td><td>79/624</td><td>133</td><td>7.4%</td></tr></table>

The fixed-budget evaluation assigns zero credit to unfilled slots. Precision over emitted ideas would answer a different question and could favor returning a smaller, selectively chosen set. It therefore cannot replace Precision@5 without changing the task. We report budget completion alongside the fixed-budget results.

## G.2 MDF: Observations versus Causes

MDF emits 3,120 forecasts, with all 624 windows filling the five-slot budget. The server’s field audit reports median approach length 4 words, 90th percentile 5, and empty key-term lists for every forecast. The blind assessment gives generality 7.98 (95% CI [7.50, 8.46]), above the prompting configurations. These measurements describe the evaluated representation; they do not directly assess the raw generated text.

The judge disagreement is substantial: default-gate Hit@5 is 0.545 under GPT-4.1-mini and 0.296 under Qwen3.5-9B. Candidate-level specificity-zero rates reported in the server audit are 58.54% and 92.99%, respectively. Recorded Qwen-judge failures (Section 5.4) can contribute zero scores, so this distribution cannot yet isolate semantic judge disagreement. The conversion behavior in Appendix B also prevents attributing sparse fields directly to training failure.

Three comparisons would help separate causes without retraining: (i) raw realization output versus final prediction fields, (ii) the deployed adapter versus the inspected implementation, and (iii) judgments of the same generated idea before and after conversion. If raw output is already underspecified, the limitation precedes conversion; if information is lost only in the adapter, evaluation input construction is implicated. The compact judged export does not support these comparisons, so no counterfactual score is claimed here.

## G.3 Status of Component Ablations

Earlier ablations used a Qwen3.5-9B configuration and do not establish the effect of the prior, reward, or memory in the evaluated Qwen2.5-7B checkpoint. A matched ablation would hold training data, backbone, conversion, decoding, and evaluation fixed while removing one component. Such experiments are unavailable for this checkpoint; the earlier tables remain archived as results of their original configuration.

## H Auxiliary Human Calibration Study

Scope. This auxiliary study documents the difficulty of idea–paper matching and the effects of annotation design. It samples an earlier 208-window experiment evaluated with Qwen3.5-9B, not the current 624- window evaluation. Its precision and recall estimates therefore characterize that historical sample and do not calibrate the current main table.

## H.1 Superseded Single-Annotator Study

The original sampling frame pooled 51,770 candidate pairs from five model–strategy rows, with a judgematch prevalence of approximately 1.7%. To avoid an almost-all-negative uniform sample, the study selected 80 pairs: 40 judge-matches, 25 borderline cases, and 15 clear negatives. An expert first labeled the pairs without the judge’s output and then reconsidered disagreements after seeing its judgment and rationale. Showing the automated rationale during reconciliation may shift labels toward the judge’s decision.

Table 10: Historical single-annotator study, superseded by the blind study below. These counts document the original sample, not current judge accuracy.
<table><tr><td>Judge stratum</td><td>Pairs</td><td>Human match</td></tr><tr><td>Match</td><td>40</td><td>36</td></tr><tr><td>No-match, borderline</td><td>25</td><td>3</td></tr><tr><td>No-match, clear</td><td>15</td><td>3</td></tr><tr><td>Total</td><td>80</td><td>42</td></tr></table>

The resulting 36/40 = 0.90 agreement on judge-matches is an optimistic reconciled estimate and is not cited as the judge’s precision. Stratified sampling requires correct prevalence weights for population estimates. A single annotator also provides no inter-annotator agreement measure.

## H.2 Blind Multi-Annotator Study

The follow-up study used eight annotators, withheld judge outputs, and included negative strata to permit weighted recall estimation. Across 340 distinct labeled pairs and 400 annotator-labels, the historical analysis reported precision 0.456 and pooled recall 0.043 (95% CI [0.026, 0.067]). Recall used a Horvitz– Thompson estimator over sampling strata and a cluster bootstrap over pairs. Thirty core pairs had at least two raters; Fleiss’ κ was 0.135 and Krippendorff’s α was 0.144.

Interpretation. The large change from the reconciled study illustrates sensitivity to annotation procedure. Low inter-rater agreement also limits confidence in treating one set of labels as definitive. Annotators received brief instructions, whereas the automated judge had anchored rubric levels and examples. Disagreement may therefore reflect different instructions or interpretations of the task, as well as model errors. These results motivate better-matched annotation protocols; they do not imply a universal correction for automated realization scores or that any difference between two scores cancels judge bias.

Participants and materials. The eight annotators were authors and research-group members participating voluntarily; no external participants were recruited and no compensation was provided. They labeled prediction text and public paper titles/abstracts. The study collected no personal data about participants. Sampling strata, frozen assignments, blind identifiers, and instructions are retained with the study materials. A new calibration study should sample the current generator outputs and both current judges, preserve the same rubric for people and models, and assess missed matches beyond the retrieval funnel if end-to-end recall is the objective.

## I Reproducibility, Contamination, and Release Scope

## I.1 Artifact Separation

The main results, all-candidate analyses, and historical studies serve different purposes. Compact judged files contain the final match flags, matched identifiers, ranks, and episode metrics needed to reconstruct the main table. All-candidate judgments support retrieval-set multiplicity and gate sensitivity. Raw model outputs and training manifests are needed to diagnose MDF and audit training-target separation. Distinguishing these artifact types prevents historical results or incomplete exports from being used to support claims they cannot establish about the current evaluation.

Each release should identify the corpus snapshot, canonical paper identifiers and date rule, topic assignments, cutoff manifest, generator checkpoint and prompt, decoding configuration, conversion adapter, embedding model and dimension, judge prompt and serving model, retrieval depth, gate, and metric version. A cached judgment is reusable only when the prediction, candidate text, and judge configuration are unchanged. A completed-window flag is not a substitute for such a cache key.

## I.2 Contamination Probe

A within-GPT-4.1 Summary probe compares earlier windows with the first six cutoffs of the current run. It attempts to align historical duration and uses the reported model knowledge boundary to distinguish potentially exposed from later windows. Under the default gate, Hit@5 is 0.7436 versus 0.7276, a difference of +0.0160 with 95% interval [−0.053, +0.085]. This is an observational temporal comparison, not a randomized test of memorization.

The two groups also differ in historical and future paper counts and possibly topic difficulty. With these confounders unresolved, an interval for the observed difference cannot bound the causal effect of contamination. A model’s stated knowledge boundary does not enumerate all training documents, and conclusions for GPT-4.1 do not establish that Qwen2.5 is unexposed. We consequently do not use this probe to label the entire current benchmark contamination-free or to explain away the backbone gap.

Training-specific leakage. For MDF, hindsight triples and reward pools can contain post-cutoff papers by design during training. This is legitimate supervision only if those target papers do not overlap evaluation targets. The audit must inspect target dates and identifiers, not just the dates at which training episodes begin. The compact evaluation package does not establish that separation by itself.

## I.3 Prospective Evaluation

A stronger future release would timestamp model weights, prompts, historical inputs, and generated forecasts before the target papers are collected. After the horizon closes, the frozen forecasts would be evaluated under a predeclared protocol. Freezing forecasts in advance would reduce exposure through retrospective generation, while publication lag, corpus coverage, and judge calibration would remain limitations. The present paper reports a retrospective benchmark and does not claim to have performed this prospective experiment.

## I.4 Research Use and Assistance

The benchmark uses public scholarly metadata and text. Redistribution remains subject to the provenance and licensing of the underlying records; code licensing does not determine rights in third-party abstracts. The human study uses voluntary research-group annotations of scientific text, as described in Appendix H.

AI assistants supported code inspection, data consistency checks, figure preparation, and manuscript drafting. Authors remain responsible for the scientific claims, experimental provenance, and released materials. Main-table confidence intervals are computed offline from stored evaluation results, without additional model generation or judging.

## J Detailed Prompts

The reference prompt templates are organized into three groups: MDF training prompts (Prompts 1–3; Appendices A and B), the five baseline forecasting prompts (Prompts 4–8; Appendix C), and the retrievethen-judge evaluation prompts (Prompts 9–10; Appendix D). Exact prompt hashes and any server-side overrides for the new runs should accompany the final release; retaining a template is not a verification that every deployed prompt was identical.

In each box, the navy bar names the prompt; navy bold headers mark the role of each segment (SYSTEM, USER, or a call-stage label such as COMPRESS/USER); and curly-brace tokens shown in navy, such as {cutoff\_month}, are runtime placeholders filled per forecasting episode.

J.1 MDF Training Prompts  
![](images/d6f46b50f56a1591640ae7a1e578e0d5c69545d9482bdeacefe7e3589096ece2.jpg)

Prompt 1  
Hindsight extraction user template   
USER   
Historical context (papers available before the cutoff):   
{context\_summary}   
Future paper to analyze:   
Title: {future\_title}   
Abstract: {future\_abstract}   
Reference grounding:   
{future\_grounding}   
Extract the innovation triple for this future paper given the historical context.   
Return JSON: {"base\_direction": "...", "operator": "...", "gap": "..."}

Prompt 2  
![](images/18466165e68337474c4da0f52fc0622d541114be5abd0ef27e627ddf5594f722.jpg)

J.2 Baseline Forecasting Prompts  
Topic Trend baseline (one call per top-ranked cluster)   
SYSTEM   
You are a precise research forecasting assistant. Return only valid JSON matching the   
requested schema.   
USER   
You are a research trend forecaster. The following keywords define a growing research   
topic cluster (literature cutoff: {cutoff\_month}):   
Keywords: {keyword\_list}   
Based on the trajectory of this topic, generate {n\_ideas} concrete research idea(s)   
likely to appear in the months following {cutoff\_month}. Do NOT use knowledge from   
after {cutoff\_month}.   
Return JSON only:   
[{"title": "...", "rationale": "...", "approach": "...", "confidence": 0.0, "key\_terms": []}]

## Prompt 4

Direct Prompting baseline   
SYSTEM   
You are a research forecasting assistant.   
Generate concrete, next-stage ideas that can be tested in the near term.   
Keep each idea specific, technical, and non-generic.   
USER   
You are given recent abstracts from the domain: {domain}.   
Forecast {n\_ideas} plausible research ideas likely to appear in {horizon}.   
The literature cutoff is {cutoff\_month}; do not use knowledge from after that cutoff.   
Abstracts:   
{abstracts}   
Return JSON only in the following format:   
{   
"ideas": [   
{   
"title": "short title",   
"rationale": "why this follows from the abstracts",   
"approach": "how to implement or test it",   
"score": 0.0,   
"confidence": 0.0,   
"key\_terms": ["optional", "keywords"]   
}   
}   
score\` and \`confidence\` must be floats between 0.0 and 1.0.  
Prompt 5

```jsonl
Summary-Augmented Prompting baseline
COMPRESS/SYSTEM
You are a research summarizer. Output a single short paragraph; no preamble, no bullets.
COMPRESS/USER
The following papers were published before {cutoff_month}.
{paper_block}
Compress this historical literature into a SHORT single-paragraph summary of about 8
sentences. Capture dominant research themes, methodological trajectories, and recurring
open problems. Do not list individual papers; produce a coherent prose summary.
FORECAST/SYSTEM
You are a research forecasting assistant working from a compressed historical summary.
Return only valid JSON matching the requested schema.
FORECAST/USER
Historical literature summary (cutoff {cutoff_month}):
{summary}
Working ONLY from this summary (do not invoke knowledge from after {cutoff_month}),
forecast {top_k} concrete research ideas likely to appear in the months following
{cutoff_month}.
Return JSON only:
{"ideas": [{"title": "...", "rationale": "...", "approach": "...", "confidence": 0.0,
"key_terms": []}]}
```

## Prompt 6

```jsonl
Retrieval-Augmented Prompting baseline
SYSTEM
You are a research forecasting assistant grounded in retrieved representative literature.
Return only valid JSON matching the requested schema.
USER
You are forecasting research ideas with a literature cutoff of {cutoff_month}.
The following representative papers were retrieved from the historical literature as the
most relevant grounding for your forecast:
{retrieved_block}
Based on these retrieved papers, forecast {top_k} concrete research ideas likely to
appear in the months following {cutoff_month}. Do NOT use knowledge from after
{cutoff_month}.
Return JSON only:
{"ideas": [{"title": "...", "rationale": "...", "approach": "...", "confidence": 0.0,
"key_terms": []}]}
```

Memory-Augmented Prompting baseline   
COMPRESS/SYSTEM   
You are a research analyst. Output only a bullet-point list, no preamble.   
COMPRESS/USER   
The following papers were published before {cutoff\_month}.   
{paper\_block}   
Summarize the 8 most important recurring research themes from these papers as concise   
bullet points (one line each). Focus on methodological trends and open problems, not   
individual papers.   
FORECAST/SYSTEM   
You are a research forecasting assistant with memory of long-term trends.   
Return only valid JSON matching the schema.   
FORECAST/USER   
Historical Memory (recurring themes from earlier literature):   
{memory}   
Recent abstracts (up to {cutoff\_month}):   
{recent\_block}   
Based on the historical memory and recent literature, forecast {top\_k} concrete research   
ideas likely to appear in the months after {cutoff\_month}. Do NOT use knowledge from   
after {cutoff\_month}.   
Return JSON only:   
{"ideas": [{"title": "...", "rationale": "...", "approach": "...", "confidence": 0.0,   
"key\_terms": []}]}

## Prompt 8

## J.3 Evaluation (Retrieve-then-Judge) Prompts

Judge system prompt   
SYSTEM   
You are an expert scientific reviewer. You are given a PREDICTED research direction -- a   
forecast made before the paper was published -- and a PUBLISHED paper. Your task is to   
judge whether the published paper is a realization of that prediction.   
Score the prediction-paper pair on three dimensions using a 0-3 scale:   
PROBLEM\_MATCH -- Does the paper address the same core research problem and goal?   
3 = Same core problem and goal -- the prediction and paper tackle the exact same question   
2 = Very similar problem: same domain challenge, minor difference in scope or framing   
1 = Adjacent problem: overlapping concern but different primary objective   
0 = Unrelated or only superficially connected problem   
METHOD\_MATCH -- Does the paper employ a similar technical mechanism or approach?   
3 = Same or near-identical mechanism -- the prediction's approach is directly implemented   
2 = Closely related approach: shares key technical ideas, differs in implementation detail   
1 = Broadly similar paradigm: same general category of method, substantially different specifics   
0 = Different technical approach entirely   
SPECIFICITY -- Does the paper realize the specific novelty described in the prediction?   
3 = Prediction's specific novelty is precisely present in the paper   
2 = Core specific novelty partially realized; paper simplifies or focuses on a subset   
1 = Prediction is generic enough to loosely fit, or paper addresses adjacent specifics   
0 = Prediction is keyword-only or meta-analytic, or paper entirely ignores the predicted novelty   
A prediction MATCHES if (PROBLEM\_MATCH + METHOD\_MATCH >= 5) AND (SPECIFICITY >= 2).   
Do NOT score based on shared topic or keyword overlap alone.   
Here are four reference examples ordered from clear non-match to clear match:   
--- EXAMPLE 1 (does NOT match -- paradigm overlap only)   
Predicted Research Direction:   
Title: Multiscale Generative Models for Long-Form Audio and Music with Hybrid   
Token-Spectrogram Representations   
Rationale: Diffusion models show promise for long high-fidelity music, and multiscale   
autoregressive architectures have proven effective for very long sequences. A likely   
near-term direction is combining hierarchical temporal structure with efficient   
long-context generation specifically for raw or near-raw audio.

Judge system prompt (continued)   
Approach: Use a two-level model where a global transformer predicts coarse musical   
structure over long spans, and local modules generate fine-grained spectrogram patches   
or audio tokens. Explore diffusion or autoregressive local decoders with recurrent   
memory for minute-scale coherence.   
Key Terms: music generation, audio modeling, multiscale transformer, diffusion,   
hierarchical generation, long-range coherence   
Published Paper:   
Title: MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers   
Abstract: We propose MEGABYTE, a multiscale decoder architecture for end-to-end modeling   
of sequences over one million bytes. It segments sequences into patches, using a local   
submodel within patches and a global model between patches. Experiments show competitive   
performance on long-context language modeling, image density estimation, and audio from   
raw files.   
PROBLEM\_MATCH: 1   
METHOD\_MATCH: 2   
SPECIFICITY: 1   
REASONING: The prediction targets audio and music generation as its primary objective,   
while MEGABYTE is a general-purpose byte-sequence architecture whose audio experiments   
are incidental -- different primary objectives. Both share a global-local hierarchy, so   
the method is broadly similar, but the prediction's specific novelties (spectrogram   
tokens, diffusion decoders, music-specific design) are entirely absent from MEGABYTE's   
domain-agnostic patch approach.   
--- EXAMPLE 2 (does NOT match -- mechanism diverges) ---   
Prediction: "Hierarchical Segment-Level Memory Routing with Learned Boundary Detection   
for Infinite-Context LLMs"   
Paper: "SnapKV", which retains important KV entries observed over a fixed prefix window.   
PROBLEM\_MATCH: 2   
METHOD\_MATCH: 1   
SPECIFICITY: 0   
REASONING: Both address long-context efficiency via selective KV retention, but the   
prediction's learned boundary detector, hierarchical routing, and cross-segment   
attention are absent from SnapKV's fixed-window heuristic.   
EXAMPLE 3 (MATCHES -- core idea realized, details differ)   
Prediction: "Adaptive KV Cache Compression via Importance Prediction and Mixed   
Precision".   
Paper: "Scissorhands", which evicts non-pivotal tokens under a fixed memory budget. ..   
PROBLEM MATCH: 3   
METHOD\_MATCH: 2   
SPECIFICITY: 2   
REASONING: Both reduce KV cache memory via importance-based eviction, so the core idea   
is present; the prediction's trained per-head importance predictor and mixed-precision   
storage are not.   
EXAMPLE 4 (MATCHES -- exact realization)   
Prediction: "Distilling Explicit Chain-of-Thought into Implicit Latent Reasoning".   
Paper: "Implicit Chain of Thought Reasoning via Knowledge Distillation".   
PROBLEM\_MATCH: 3   
METHOD\_MATCH: 3   
SPECIFICITY: 3   
REASONING: Both internalize chain-of-thought into latent states via teacher-student   
distillation with no intermediate tokens at inference -- problem, mechanism, and goal   
are identical.   
Respond with exactly four lines:   
PROBLEM\_MATCH: <0, 1, 2, or 3>   
METHOD\_MATCH: <0, 1, 2, or 3>   
SPECIFICITY: <0, 1, 2, or 3>   
REASONING: <one to two sentences explaining your scores>