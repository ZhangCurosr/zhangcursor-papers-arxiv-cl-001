# DELISTBENCH: Evaluating Search-Enabled LLMs for Auditable Corporate-Event Database Completion

Xuan Yao Li Shuping Dai Yang Zhou Yi Ke-Wei Huang Asian Institute of Digital Finance, National University of Singapore {yaoxuan, aidf.li, koreydai, zhouyi02, dishkw}@nus.edu.sg

## Abstract

Financial institutions need an independent way to detect missing, stale, and misclassified corporate-event records in vendor databases. We introduce SEARCH-TO-RECORD, a database-assurance task in which search-enabled large language models reconstruct institution-defined event records from public sources for a known security universe and historical cutoff, and DELISTBENCH, a 1,200-record benchmark for security-level delisting announcements. We evaluate five models in paired closed-book and web-enabled conditions. Web access raises announcementdate accuracy within seven days by 34.0–48.0 percentage points and event-status accuracy by approximately 2.8–21.7 points; the best system achieves 81.5% overall joint accuracy within seven days. Economy web systems achieve 75.9–78.3% overall joint accuracy within seven days at 4.5–6.6% of the API cost of the most expensive web system. Risk-based triage identifies low-error subsets, although the highestcoverage operating point still sends 27.3% of the balanced test set to review. The evaluation identifies web retrieval as the main source of timing gains and shows that lowcost systems can approach the best system’s accuracy. Together, SEARCH-TO-RECORD, DELISTBENCH, and the evaluation provide concrete deployment guidance: calibrate triage to local event prevalence and market mix, pre serve positive-event recall, and route positive and ambiguous cases to targeted review.

## 1 Introduction

Financial institutions maintain a security master that identifies the issuers, securities, listings, exchanges, and identifiers in scope. Corporate-event histories attached to that universe are often purchased from global data vendors. The central operational uncertainty is therefore not usually which companies exist; it is whether the event database is complete, accurate, timely, and consistent with the institution’s definition. Coverage gaps may be concentrated in small markets, local-language disclosures, cross-listed securities, or event types whose procedural boundaries differ across exchanges.

An independent public-source audit is expensive. For every target listing, an analyst may need to locate exchange and issuer notices, distinguish a formal action from a warning, identify the announcement rather than effective date, map the stated reason, and retain an evidence trail. Search-enabled large language models (LLMs) can perform these steps at low API cost, but fluent output is not necessarily database-ready. A model can retrieve the correct company but wrong listing, project a later event backward across a historical cutoff, or return a plausible date unsupported by its citations.

We study this setting as closed-universe, openevent-status completion. The target identity and cutoff are known; the existence and contents of a qualifying event record are not. We instantiate the task with delisting announcements, whose securitylevel scope, heterogeneous reasons, and easily confused procedural dates create a demanding test of retrieval and temporal adjudication.

Our contributions are:

• We formulate SEARCH-TO-RECORD (S2R), an auditable database-assurance task that reconstructs institution-defined event records independently of a vendor feed.

• We construct and will publicly release DELISTBENCH, a 1,200-record benchmark with balanced positive, later-event, and stilllisted strata, public evidence, and issuerfamily-disjoint splits.

• We compare five model families in paired closed-book and web-enabled conditions across accuracy, evidence, cost, and latency, and evaluate risk-based acceptance under both the balanced benchmark and a low-prevalence deployment framing.

## 2 Related Work

Financial event extraction. Financial event extraction typically assumes a supplied document. DCFEE extracts document-level events (Yang et al., 2018); fine-grained resources annotate businessnews triggers and arguments (Jacobs and Hoste, 2020); and FinReason targets reasons in company announcements (Chen et al., 2021). S2R instead includes retrieval and listing resolution and returns a database record rather than an event mention.

Search-grounded language models. WebGPT and ReAct established browser-assisted answering and interleaved tool use (Nakano et al., 2022; Yao et al., 2023). FreshLLMs and RealTime QA study retrieval for changing facts (Vu et al., 2024; Kasai et al., 2023); financial RAG benchmarks instead provide a bounded document collection (Lithgow-Serrano et al., 2025). Recent financial deep-research benchmarks study corporate analysis and forecasting (Zhu et al., 2025; Li et al., 2026). We use the open web but fix the target identity, event definition, and historical cutoff.

Evidence and selective prediction. ALCE evaluates citation quality (Gao et al., 2023), while LongFact decomposes claims for search-based verification (Wei et al., 2024). Selective classification trades coverage for lower error (Geifman and El-Yaniv, 2017); risk-control procedures can provide finite-sample guarantees under explicit assumptions (Bates et al., 2021). Our calibrated-acceptance threshold enforces an empirical calibration-set criterion, not a formal test-set guarantee.

## 3 Methodology

## 3.1 Search-to-Record Task

An S2R input is

$$
x _ { i } = ( e _ { i } , s _ { i } , v _ { i } , c _ { i } , z _ { i } , T _ { i } ) ,\tag{1}
$$

where $e _ { i }$ is the issuer, $s _ { i }$ the security, $v _ { i }$ the target venue, $c _ { i }$ the country, $z _ { i }$ an optional stable identifier, and $T _ { i }$ the cutoff. The system returns

$$
\hat { y } _ { i } = ( q _ { i } , d _ { i } , r _ { i } , E _ { i } , u _ { i } ) ,\tag{2}
$$

where $q _ { i } \in \{ \mathsf { y e s } , \mathsf { n o } .$ , unknown} is status at the cutoff, $d _ { i }$ the formal announcement date, $r _ { i }$ a normalized reason, $E _ { i }$ claim-level evidence, and $u _ { i }$ uncertainty. Date and reason are null unless $q _ { i } = \mathsf { y e s }$

The same output supports several databaseassurance decisions: a missing vendor event, a stale update, an incorrect date or reason, an event attached to the wrong listing, or confirmation that no qualifying event existed by the cutoff. The vendor record is not shown to the evaluated model; public reconstruction therefore provides an independent comparison rather than a paraphrase of the incumbent feed.

Delisting event definition. A positive requires a publicly disclosed formal decision, approved application, exchange determination, filed withdrawal, or completed delisting action directed at the specified security and venue. A deficiency warning, cure period, possible delisting, suspension, lasttrading date, effective date without a qualifying disclosure, or action concerning another listing is insufficient. Later reversal or continued trading does not invalidate a qualifying historical action. Current pages may reconstruct history, but the canonical date is the first qualifying public disclosure identified during benchmark construction, not the page-publication or effective-removal date. This makes later-event projection a first-class error rather than treating all retrieved delisting content as relevant.

## 3.2 DelistBench

We instantiate SEARCH-TO-RECORD as cutoffaware reconstruction of security-level delisting announcements. Figure 2 summarizes the resulting DELISTBENCH benchmark and evaluation protocol.

Benchmark construction and diagnostic strata. DELISTBENCH contains 1,200 records divided evenly among three diagnostic strata, as illustrated in Figure 2. Stratum A contains 400 positivecompletion records for which a qualifying delisting announcement occurred on or before the cutoff. Stratum B contains 400 future-announcement negatives: no qualifying formal delisting action had been disclosed by the cutoff, but a qualifying announcement was issued only after the cutoff. Stratum C contains 400 active controls for which no qualifying formal delisting action was identified for the exact listing at any time through the observation date of 6 August 2026. Although B and C both have gold status no, they represent fundamentally different failure modes.

![](images/53cdb94d7130001e5a7ec66b21f91976b8c57b4b29349f80bf3b64520a93013f.jpg)  
Figure 1: The five-stage SEARCH-TO-RECORD workflow: retrieve public evidence for a known target, apply cutoff-aware adjudication, construct a record, and route it to acceptance or review.

The announcement candidates span 59 issuer countries. Of the full benchmark, 38.2% are from the United States and Canada, 26.3% from Asia– Pacific, 26.3% from Europe, 4.8% from Africa and the Middle East, and 4.4% from Latin America and the Caribbean. Country is descriptive; evaluation remains listing- and venue-specific. Candidate A/B events originate from a licensed corporate-event source used only for discovery and internal audit. Release labels are reconstructed from permissible public evidence, and licensed text is not sent to evaluated APIs. Category C records are matched on geography, industry, name shape, and approximate difficulty. No qualifying formal delisting action was identified for the exact listing through 6 August 2026, and the listing remained active on that date.

The cutoff is 2020-12-31, with a 180-day exclusion buffer on each side to reduce boundary instability. Records use issuer-family-disjoint, stratified splits: 600 for risk-model training, 300 for threshold calibration, and 300 for locked testing. Gold stratum, event date, reason, difficulty, and internal evidence never enter model prompts.

## 3.3 Evaluation Protocol and Experimental Setup

Evaluation metrics. As summarized in Figure 2, decision accuracy is evaluated over all 1,200 records. Date and broad-reason accuracy are evaluated over all 400 A records, including missed positives. Joint correctness requires the correct decision for every record and, for A, an announcement date within the stated tolerance together with the correct broad reason. A correct no decision is sufficient for B and C. Invalid outputs and abstentions count as errors.

<table><tr><td>Model</td><td>Reasoning</td><td>Max</td><td>Web mode</td></tr><tr><td>GPT-5.6 Luna</td><td>low</td><td>4,096</td><td>native</td></tr><tr><td>DeepSeek V4 Pro</td><td>high</td><td>8,192</td><td>server-side</td></tr><tr><td>Gemini 3.6 Flash</td><td>default 8,192</td><td></td><td>native</td></tr><tr><td>DeepSeek V4 Flash</td><td></td><td>off 4,096</td><td>server-side</td></tr><tr><td>Gemini 3.5 Flash-Lite</td><td>medium 4,096</td><td></td><td>native</td></tr></table>

Table 1: Model configurations. Every model is evaluated in closed and web conditions; reasoning policy is fixed within each pair. Max denotes output-token limit.

Because joint accuracy can be inflated by conservative no answers, we also report A-only date and reason metrics. Evidence quality, cost, and latency are evaluated separately and do not enter hierarchical correctness.

Model configurations. All ten systems used the standard service tier, the same record-level schema, and one retained response per record. Closed and web arms differed in retrieval/evidence instructions and web access; other within-model settings were fixed. The within-model contrast estimates the configured web effect, whereas cross-model results compare complete systems rather than equalcompute architectures.

![](images/82e6e3f92431eb2862732a3641967b5142f4824a590622a577caff9dd56603d7.jpg)  
Figure 2: Overview of the DELISTBENCH design and evaluation protocol. The three balanced strata distinguish positive completion (A) from future-event negatives (B) and still-listed negatives (C). Levels 0–2 define hierarchical record correctness, while evidence quality, retrieval cost, latency, and review requirements are evaluated separately. The 59-country count describes the announcement-candidate pool, and the human-review panel includes an independent reliability check. Evaluation remains listing- and venue-specific.

Structured-output enforcement differed: OpenAI and Gemini used provider-native schemas, while DeepSeek used prompt-constrained JSON with local validation.

Costs use observed API usage and public list prices accessed on 11 August 2026 and exclude engineering, infrastructure, and human review. Latency is request wall time.

Risk-based review routing. For each system and correctness target, an L2-regularized logistic regression predicts record error from model confidence, schema validity, abstention, citation count, latency, answer length, and provider-reported search requests. Gold fields, category, proprietary identifiers, and the independent evidence audit are excluded. On the 300-record calibration split, we choose the largest predicted-risk threshold whose empirical accepted-set error is at most 5%, then apply it unchanged to the test split. Invalid outputs and abstentions always route to review.

## 4 Results

## 4.1 Record Reconstruction Performance

Event-status accuracy. Table 2 shows that every web-enabled system has higher status accuracy and macro-F1 than its closed-book counterpart. The category-specific results are less uniform: A-recall decreases by 4.8 pp for DeepSeek V4 Pro, and B-FEPR increases by 2.0 pp for Gemini 3.5 Flash-Lite. Nevertheless, web access reduces C-FPR for every model and substantially reduces B-FEPR for the other four. Gemini 3.6 Flash web achieves the highest status accuracy, macro-F1, A-precision, and A-recall.

Date and reason completion. The positive-event metrics in Table 3 show larger gains in information recovery. A-date-±7 accuracy increases by 34.0– 48.0 pp, while A-joint-±7 completion increases by 24.8–35.3 pp. Gemini 3.6 Flash web performs best, but it still exactly dates only 46.0% of A records and jointly recovers the decision, date within seven days, and broad reason for only 45.2%. Web access therefore addresses much of the information-access problem without making positive records reliably database-ready.

Overall joint accuracy remains vulnerable to negative-class dominance and should therefore be interpreted alongside A-specific date, reason, and completion metrics.

<table><tr><td>Model</td><td>Cond.</td><td>Accuracy</td><td>Macro-F1</td><td>A-Prec.</td><td>A-Recall</td><td>B-FEPR</td><td>C-FPR</td><td>Abstain</td></tr><tr><td rowspan="2">GPT-5.6 Luna</td><td>Closed</td><td>81.6</td><td>80.8</td><td>71.7</td><td>79.0</td><td>18.5</td><td>12.8</td><td>2.2</td></tr><tr><td>Web</td><td>93.8</td><td>92.9</td><td>98.2</td><td>83.0</td><td>1.0</td><td>0.5</td><td>0.4</td></tr><tr><td rowspan="2">DeepSeek V4 Pro</td><td>Closed</td><td>82.9</td><td>82.2</td><td>69.2</td><td>89.8</td><td>25.0</td><td>15.0</td><td>0.7</td></tr><tr><td>Web</td><td>94.7</td><td>94.1</td><td>99.1</td><td>85.0</td><td>0.8</td><td>0.0</td><td>0.9</td></tr><tr><td rowspan="2">DeepSeek V4 Flash</td><td>Closed</td><td>73.2</td><td>64.9</td><td>77.0</td><td>33.5</td><td>9.2</td><td>0.8</td><td>2.6</td></tr><tr><td>Web</td><td>94.9</td><td>94.7</td><td>99.1</td><td>86.5</td><td>0.8</td><td>0.0</td><td>0.6</td></tr><tr><td rowspan="2">Gemini 3.6 Flash</td><td>Closed</td><td>94.6</td><td>93.9</td><td>91.4</td><td>92.5</td><td>6.8</td><td>2.0</td><td>0.0</td></tr><tr><td>Web</td><td>97.3</td><td>97.0</td><td>99.2</td><td>92.8</td><td>0.8</td><td>0.0</td><td>0.1</td></tr><tr><td rowspan="2">Gemini 3.5 Flash-Lite</td><td>Closed</td><td>85.4</td><td>83.5</td><td>88.2</td><td>67.5</td><td>5.8</td><td>3.2</td><td>2.2</td></tr><tr><td>Web</td><td>92.0</td><td>90.9</td><td>91.1</td><td>84.2</td><td>7.8</td><td>0.5</td><td>0.0</td></tr></table>

Table 2: Announcement-decision performance by benchmark category (%). Accuracy, macro-F1, and abstention use all 1,200 records per system. A-precision and A-recall treat Category A as the positive class. B-FEPR is the share of Category B records incorrectly predicted as having a qualifying announcement by the cutoff; C-FPR is the corresponding false-positive rate among Category C controls with no qualifying announcement. Invalid outputs and abstentions count as errors in accuracy and macro-F1. Best values across all ten systems are bold; ties after rounding are retained.
<table><tr><td>Model</td><td>Cond.</td><td>Date exact</td><td>Date ±7</td><td>Reason</td><td>A joint exact</td><td>A joint ±7</td><td>Overall joint ±7</td><td>$/1k</td><td>Med. sec.</td></tr><tr><td>GPT-5.6 Luna</td><td>Closed</td><td>3.0</td><td>7.5</td><td>49.0</td><td>2.5</td><td>5.8</td><td>57.2</td><td>1.03</td><td>5.12</td></tr><tr><td rowspan="2">DeepSeek V4 Pro</td><td>Web</td><td>38.2</td><td>48.2</td><td>66.0</td><td>32.0</td><td>39.8</td><td>79.4</td><td>27.33</td><td>11.58</td></tr><tr><td>Closed</td><td>5.0</td><td>12.2</td><td>52.8</td><td>5.0</td><td>11.0</td><td>56.7</td><td>0.96</td><td>14.26</td></tr><tr><td rowspan="2">DeepSeek V4 Flash</td><td>Web</td><td>39.5</td><td>49.0</td><td>66.2</td><td>31.5</td><td>38.0</td><td>79.0</td><td>7.30</td><td>21.22</td></tr><tr><td>Closed</td><td>0.2</td><td>1.8</td><td>23.2</td><td>0.2</td><td>1.5</td><td>62.6</td><td>0.06</td><td>1.47</td></tr><tr><td></td><td>Web</td><td>39.2</td><td>49.8</td><td>65.2</td><td>29.8</td><td>36.8</td><td>78.3</td><td>1.81</td><td>8.28</td></tr><tr><td rowspan="2">Gemini 3.6 Flash</td><td>Closed</td><td>14.5</td><td>24.2</td><td>63.7</td><td>13.8</td><td>20.5</td><td>70.6</td><td>8.45</td><td>4.20</td></tr><tr><td>Web</td><td>46.0</td><td>58.2</td><td>71.8</td><td>36.2</td><td>45.2</td><td>81.5</td><td>11.83</td><td>6.27</td></tr><tr><td rowspan="2">Gemini 3.5 Flash-Lite</td><td>Closed</td><td>5.2</td><td>12.5</td><td>50.7</td><td>4.5</td><td>10.8</td><td>66.5</td><td>1.38</td><td>1.06</td></tr><tr><td>Web</td><td>36.5</td><td>46.5</td><td>64.2</td><td>29.2</td><td>36.0</td><td>75.9</td><td>1.23</td><td>1.99</td></tr></table>

Table 3: Record-completion and operational results. Date, reason, and A-joint metrics use all 400 Category A records, including misses. A-joint requires the correct positive status, date tolerance, and broad reason; overall joint-±7 also includes B/C records, for which a correct negative status is sufficient. Cost is estimated per 1,000 records.

Cost and latency. Table 3 shows that the highestpriced web system is not the most accurate. Gemini 3.6 Flash achieves the highest A-joint-±7 completion while costing less than half as much per 1,000 records as GPT-5.6 Luna. The economy systems occupy different points on the cost–performance frontier: DeepSeek V4 Flash web is 3.0 pp below GPT-5.6 Luna in A-joint-±7 completion while costing only 6.6% as much, whereas Gemini 3.5 Flash-Lite is 3.8 pp below at 4.5% of the cost and has the lowest web latency. These figures exclude analyst review, so API cost should not be interpreted as total deployment cost; lower evidence quality or acceptance coverage may shift costs to manual verification.

## 4.2 In-Depth Analysis

<table><tr><td>Joint-exact failure</td><td>Count</td><td>Share</td></tr><tr><td>Date mismatch only</td><td>1,553</td><td>41.4%</td></tr><tr><td>Missed announcement</td><td>741</td><td>19.8%</td></tr><tr><td>Date + reason</td><td>712</td><td>19.0%</td></tr><tr><td>Future projection on B</td><td>305</td><td>8.1%</td></tr><tr><td>Reason mismatch only</td><td>171</td><td>4.6%</td></tr><tr><td>C false positive</td><td>139</td><td>3.7%</td></tr><tr><td>Abstention</td><td>115</td><td>3.1%</td></tr><tr><td>Analysis-invalid</td><td>14</td><td>0.4%</td></tr></table>

Table 4: Mutually exclusive inventory over 3,750 jointexact failures from all ten systems. Shares may not sum to 100% because of rounding.

Error analysis. Date-only and date-plus-reason errors constitute 60.4% of failures (Table 4), while only 14 outputs across the ten 1,200-record system runs are analysis-invalid. The main bottleneck is thus not JSON production. It is identifying the formal event stage, binding it to the target listing, and preserving the cutoff.

Evidence audit. Codex, using OpenAI GPT-5.6 Sol with high reasoning effort, audited the evidence submitted by the five web systems for the same 150 records (750 case–system packages). It assessed the submitted URLs jointly for accessibility, exact listing identity, cutoff relevance, and support for the system’s own prediction. The audit used neither benchmark labels nor replacement searches and did not modify predictions or annotations.

Table 5: Submitted-evidence protocol outcomes by web system (%).
<table><tr><td></td><td colspan="3">Predicted yes</td><td colspan="3">Predicted no</td></tr><tr><td>Model</td><td>Pass</td><td>Fail</td><td>Unver.</td><td>Pass</td><td>Fail</td><td>Unver.</td></tr><tr><td>Gemini 3.6 Flash</td><td>40.8</td><td>20.4</td><td>38.8</td><td>100.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek V4 Pro</td><td>55.6</td><td>0.0</td><td>44.4</td><td>93.3</td><td>1.9</td><td>4.8</td></tr><tr><td>GPT-5.6 Luna</td><td>60.0</td><td>0.0</td><td>40.0</td><td>74.3</td><td>1.9</td><td>23.8</td></tr><tr><td>Gemini 3.5 Flash-Lite</td><td>38.3</td><td>17.0</td><td>44.7</td><td>99.0</td><td>0.0</td><td>1.0</td></tr><tr><td>DeepSeek V4 Flash</td><td>72.1</td><td>0.0</td><td>27.9</td><td>75.7</td><td>1.9</td><td>22.4</td></tr><tr><td>Overall</td><td>52.8</td><td>7.9</td><td>39.3</td><td>88.3</td><td>1.1</td><td>10.6</td></tr></table>

Notes: For predicted yes, Pass includes Supported and Partially supported; Fail includes Unsupported and No evidence. For predicted no, Pass additionally includes No evidence, while Fail denotes Unsupported. Unver. denotes Cannot assess. Percentages use sum-preserving rounding within each prediction block; one unknown prediction is excluded.

As shown in Table 5, 52.8% of evidence packages for yes predictions passed the protocol, compared with 88.3% for no predictions. This difference partly reflects the asymmetric protocol: affirmative predictions require supporting evidence, whereas negative predictions are not failed solely because no evidence was submitted. Unverifiable indicates insufficient submitted evidence, not an incorrect prediction. The results therefore support manual review when evidentiary reliability is important.

For validation, two human analysts each rated a different blinded sample of 100 packages using the same codebook. Exact human–Codex agreement on Pass, Fail, or Unverifiable was 156/200 packages (78.0%). Much of the disagreement reflected a more conservative Codex threshold for assessability: Codex classified 37 packages as Unverifiable, compared with 16 by the human reviewers. Among the 160 packages for which both Codex and the human reviewer rendered a Pass/Fail judgment, agreement was 143/160 (89.4%). Thus, the lower three-way agreement primarily reflects differences in handling inaccessible or insufficient evidence rather than opposing substantive judgments.

## 4.3 Risk-Based Acceptance and Review Routing

The risk model identifies useful low-error subsets at the mixed-case level. As shown in Table 6, Gemini 3.6 Flash accepts 72.7% of test records with

3.7% error, while DeepSeek V4 Pro accepts 65.7% with 2.5% error. GPT-5.6 Luna reaches 6.8% test error despite the 5% calibration target, showing that empirical calibration is not a test-set guarantee.

The Category A decomposition shows that the low aggregate error is driven primarily by reliable negative screening. Only 2.1%–18.9% of positive records are accepted, and their conditional accepted-error rates remain 44.4%–70.6%. Even Gemini 3.6 Flash produces only 10 jointexact, auto-accepted Category A records. These results motivate a differentiated workflow rather than uniform automation: low-risk negatives can be cleared automatically, while predicted-positive and ambiguous records receive additional verification. The observed review shares describe the balanced benchmark and should not be interpreted directly as production review loads.

## 5 Deployment and Discussion

## 5.1 Deployment Calibration

The benchmark deliberately sets $\pi _ { A } ~ = ~ \pi _ { B } ~ =$ $\pi _ { C } = 1 / 3$ to expose rare but consequential failures. A production security universe may contain substantially fewer positive event-period tasks. If $r _ { s } ( t )$ is the review rate in stratum s under threshold t, the deployment review load is

$$
R _ { \mathrm { d e p } } ( t ) = \sum _ { s \in \{ A , B , C \} } \pi _ { s } ^ { \mathrm { d e p } } r _ { s } ( t ) .\tag{3}
$$

Consequently, the 27.3% minimum review rate observed on the balanced test set is not a production estimate. If readily resolved no-event records dominate the universe, the overall review load may be substantially lower even when predictedpositive candidates receive intensive verification. This prevalence effect does not resolve the high Category A error rates in Table $\begin{array} { r } { 6 ; } \end{array}$ instead, it can make targeted positive verification operationally affordable. It reduces human-review and conditionalescalation costs, although the initial inference cost over the full universe remains.

This prevalence effect must not be used to hide missed events. A production policy should be selected under simultaneous constraints such as

$$
\begin{array} { r } { R _ { \mathrm { d e p } } \leq 5 \% , ~ \mathrm { R e c a l l } _ { A } \geq \gamma , } \\ { \mathrm { E r r } _ { \mathrm { a c c e p t } } \leq \delta . ~ } \end{array}\tag{4}
$$

These constraints apply to the same unified risk score and threshold; they do not imply categoryspecific models or thresholds. The recall constraint prevents a low review rate from being achieved by overlooking positive events, while the acceptederror constraint controls the reliability of records cleared without review. The post-hoc Category A decomposition in Table 6 remains an additional safety diagnostic, revealing whether acceptable aggregate performance is driven primarily by easy negative records.

<table><tr><td></td><td colspan="4">All records</td><td colspan="4">Category A</td></tr><tr><td>Web model</td><td>Accept</td><td>Cov.</td><td>Err.</td><td>Review</td><td>Accept</td><td>Cov.</td><td>Err.</td><td>Ready yield</td></tr><tr><td>GPT-5.6 Luna</td><td>177/300</td><td>59.0%</td><td>6.8%</td><td>41.0%</td><td>17/95</td><td>17.9%</td><td>70.6%</td><td>5.3%</td></tr><tr><td>DeepSeek V4 Pro</td><td>197/300</td><td>65.7%</td><td>2.5%</td><td>34.3%</td><td>9/95</td><td>9.5%</td><td>55.6%</td><td>4.2%</td></tr><tr><td>DeepSeek V4 Flash</td><td>83/300</td><td>27.7%</td><td>1.2%</td><td>72.3%</td><td>2/95</td><td>2.1%</td><td>50.0%</td><td>1.1%</td></tr><tr><td>Gemini 3.6 Flash</td><td>218/300</td><td>72.7%</td><td>3.7%</td><td>27.3%</td><td>18/95</td><td>18.9%</td><td>44.4%</td><td>10.5%</td></tr><tr><td>Gemini 3.5 Flash-Lite</td><td>106/300</td><td>35.3%</td><td>2.8%</td><td>64.7%</td><td>3/95</td><td>3.2%</td><td>66.7%</td><td>1.1%</td></tr></table>

Table 6: Joint-exact calibrated acceptance at a 5% empirical calibration-error target. Category A results are a post-hoc decomposition of the same frozen acceptance decisions; thresholds are not refitted by category. Error is measured among accepted test records. Ready yield is the share of all 95 Category A test records that are both accepted and joint-exact correct. Best web-system values are bold.

Because the logistic target is output correctness rather than event occurrence, matching only the delist/no-delist ratio is insufficient. Calibration must also reflect market, language, source accessibility, identifier quality, and the mixture of ordinary and temporally difficult negatives. A practical approach is to train the error ranker on the balanced benchmark, then calibrate probabilities and choose thresholds on a representative institutional sample. When only the target mixture is known, one can use stratum weights $w _ { s } = \pi _ { s } ^ { \mathrm { d e p } } / \pi _ { s } ^ { \mathrm { b e n c h } }$ , followed by prospective validation.

This framing positions the system as a riskadaptive verification workflow. Low-risk records can be cleared after the initial pass; medium-risk cases can trigger additional official-source retrieval, deterministic timeline checks, an independent verifier, or a stronger model; and unresolved highrisk cases can be reserved for human review. The pipeline therefore complements vendor data by auditing a known security universe and prioritizing scarce analyst attention, without treating uncertain positive records as database-ready.

## 5.2 Operational Implications

Detection versus record completion. The best web system reaches 97.3% decision accuracy but only 46.0% exact date accuracy on A. LLM-plussearch is therefore closer to high-quality candidate detection than to unattended complete-row generation. Separating status risk from conditional date/reason risk is a promising deployment design.

Geographic subgroup analysis. Country performance is a subgroup analysis of the fixed 1,200- record benchmark rather than an additional model run. Table 7 reports representative status and positive-completion metrics. The announcementcandidate pool spans 59 issuer countries; the finalized input-country field used for this subgroup analysis contains 51 country values, with the United States, United Kingdom, and Canada accounting for 65.4% of the records. To reduce instability, Panel A includes countries with at least 20 records, while Panel B additionally requires at least 10 Category A records. Web-enabled systems improve status macro-F1 and Category A recall in every displayed country, but web joint-±7 completion still ranges from 27.3% to 55.8%. Because country is correlated with language, source accessibility, exchange coverage, identifier quality, and event composition, these results should be interpreted as descriptive heterogeneity diagnostics rather than causal country effects or stable country rankings.

Institutional use. The intended use is not vendor replacement. It is an independent quality-assurance layer that flags missing, stale, or definitionmisaligned records; reconstructs permissible public evidence; and concentrates analyst attention. The relevant economic comparison is total cost per corrected or newly recovered record, including human review and the incumbent vendor baseline.

## 6 Limitations and Ethical Considerations

The balanced benchmark does not reflect natural event prevalence, and the reported risk estimates are not reweighted to an institutional distribution. Historical events are reconstructed using the current web rather than a sealed historical index, and live detection latency is not evaluated. Gold dates represent the first qualifying disclosures identified during construction, not necessarily the earliestever disclosures. Category C indicates that no qualifying action was identified through 6 August 2026, rather than proving that none existed. Country denotes issuer geography rather than listing market, and the absence of Vietnam-listed records limits conclusions about the motivating low-resourcemarket setting.

<table><tr><td colspan="7">Panel A: Announcement-status performance</td></tr><tr><td>Country</td><td>n</td><td>A/B/C</td><td>Closed macro-F1</td><td>Web macro-F1</td><td>Closed A-recall</td><td>Web A-recall</td></tr><tr><td>United States</td><td>300</td><td>90/109/101</td><td>86.5</td><td>91.9</td><td>78.4</td><td>80.2</td></tr><tr><td>United Kingdom</td><td>255</td><td>85/88/82</td><td>82.7</td><td>95.1</td><td>74.1</td><td>88.9</td></tr><tr><td>Canada</td><td>230</td><td>79/74/77</td><td>76.6</td><td>92.3</td><td>65.3</td><td>81.8</td></tr><tr><td>Australia</td><td>35</td><td>11/13/11</td><td>81.3</td><td>95.3</td><td>74.5</td><td>90.9</td></tr><tr><td>Japan</td><td>33</td><td>5/18/10</td><td>73.9</td><td>95.2</td><td>68.0</td><td>88.0</td></tr><tr><td>Switzerland</td><td>31</td><td>11/9/11</td><td>83.5</td><td>98.6</td><td>81.8</td><td>98.2</td></tr><tr><td>Sweden</td><td>29</td><td>5/13/11</td><td>84.3</td><td>96.2</td><td>88.0</td><td>100.0</td></tr><tr><td>Finland</td><td>22</td><td>7/6/9</td><td>89.4</td><td>100.0</td><td>85.7</td><td>100.0</td></tr></table>

<table><tr><td colspan="3">Panel B: Positive-record joint completion Country</td><td rowspan="2">Web exact</td><td rowspan="2">Closed ±7</td><td rowspan="2">Web ±7</td></tr><tr><td></td><td>A n</td><td>Closed exact</td></tr><tr><td>United States</td><td>90</td><td>18.0</td><td>45.6</td><td>28.4</td><td>55.8</td></tr><tr><td>United Kingdom</td><td>85</td><td>2.8</td><td>27.3</td><td>6.6</td><td>30.6</td></tr><tr><td>Canada</td><td>79</td><td>0.5</td><td>31.9</td><td>2.0</td><td>40.3</td></tr><tr><td>Australia</td><td>11</td><td>0.0</td><td>16.4 29.1</td><td>3.6</td><td>27.3</td></tr><tr><td>Switzerland</td><td>11</td><td>0.0</td><td></td><td>1.8</td><td>43.6</td></tr></table>

Table 7: Country-level performance (%), macro-averaged across the five models within each condition. Panel A includes countries with at least 20 benchmark records and reports status macro-F1 and Category A recall. Panel B additionally requires at least 10 Category A records and evaluates joint recovery of status, date, and broad reason on positive records only. Country denotes the benchmark country field supplied to the systems and is distinct from exchange country.

The evidence audit is Codex-assisted. Two nonoverlapping 100-package human-validation samples yielded 78.0% exact agreement with Codex; the full set of 750 packages was not independently double-coded. Because web pages, search indexes, prices, and model aliases may change, reproducible evaluation requires archiving prompts, access dates, configurations, and raw outputs.

Upon publication, we will publicly release the benchmark inputs, reconstructed labels, publicevidence references, split assignments, prompts, and evaluation code. Licensed discovery materials and restricted identifiers will remain private; released labels are reconstructed solely from permissible public evidence.

Automation may propagate plausible errors into downstream research, surveillance, or client systems. Production use should therefore preserve provenance, prevent silent overwrites, monitor subgroup recall and calibration drift, and retain human review for positive, conflicting, or high-impact records.

## 7 Conclusion

SEARCH-TO-RECORD reframes search-enabled LLM evaluation around a common institutional problem: auditing corporate-event records over a known security universe. On DELISTBENCH, web access sharply improves event timing and enables economy systems to approach stronger-system accuracy at low API cost. The remaining errors are concentrated in positive-record dates, reasons, listing scope, and cutoff discipline rather than output syntax. A simple risk layer provides meaningful triage, but balanced-benchmark coverage is not a production review estimate and its empirical target is not a guarantee. The path to an economical automated pipeline is therefore a combination of stronger positive completion, market-aware deployment calibration, risk-triggered machine escalation, and a small, explicitly budgeted human-review tail.

## References

Stephen Bates, Anastasios N. Angelopoulos, Lihua Lei, Jitendra Malik, and Michael I. Jordan. 2021. Distribution-free, risk-controlling prediction sets. Journal ofthe ACM, 68(6):1–34.

Pei Chen, Kang Liu, Yubo Chen, Taifeng Wang, and Jun Zhao. 2021. Probing into the root: A dataset for reason extraction of structural events from financial documents. In Proceedings of the 16th Conference of the European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 2042–2048. Association for Computational Linguistics.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488. Association for Computational Linguistics.

Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems 30, pages 4878–4887.

Gilles Jacobs and Veronique Hoste. 2020. Extracting fine-grained economic events from business news. In Proceedings ofthe 1st Joint Workshop on Financial Narrative Processing and MultiLing Financial Summarisation, pages 235–245. COLING.

Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, and Kentaro Inui. 2023. Realtime QA: What’s the answer right now? In Advances in Neural Information Processing Systems 36: Datasets and Benchmarks Track.

Xiangyu Li, Xuan Yao, Guohao Qi, Fengbin Zhu, Kelvin J. L. Koa, Xiang Yao Ng, Ziyang Liu, Xingyu Ni, Chang Liu, Yonghui Yang, et al. 2026. FinDeepForecast: A live multi-agent system for benchmarking deep research agents in financial forecasting. arXiv preprint arXiv:2601.05039.

Oscar Lithgow-Serrano, David Kletz, Vani Kanjirangat, David Adametz, Marzio Lunghi, Claudio Bonesana, Matilde Tristany-Farinha, Yuntao Li, Detlef Repplinger, Marco Pierbattista, Stefania Stan, and Oleg Szehr. 2025. Assessing RAG system capabilities on financial documents. In Proceedings ofThe 10th Workshop on Financial Technology and Natural Language Processing, pages 124–147. Association for Computational Linguistics.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. 2022. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.

Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. 2024. Fresh-LLMs: Refreshing large language models with search engine augmentation. In Findings ofthe Association for Computational Linguistics: ACL 2024, pages 13697–13720. Association for Computational Linguistics.

Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, and Quoc V. Le. 2024. Long-form factuality in large language models. arXiv preprint arXiv:2403.18802.

Hang Yang, Yubo Chen, Kang Liu, Yang Xiao, and Jun Zhao. 2018. DCFEE: A document-level Chinese financial event extraction system based on automatically labeled training data. In Proceedings ofACL 2018, System Demonstrations, pages 50–55. Association for Computational Linguistics.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Fengbin Zhu, Xiang Yao Ng, Ziyang Liu, Chang Liu, Xianwei Zeng, Chao Wang, Tianhui Tan, Xuan Yao, Pengyang Shao, Min Xu, et al. 2025. FinDeepResearch: Evaluating deep research agents in rigorous financial analysis. arXiv preprint arXiv:2510.13936.