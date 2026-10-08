# The Long Road to the Same Answer: Cognitive Bias Under Escalating Reasoning Budgets in Large Language Models

Obada Kraishan

College of Media and Communication

Texas Tech University

Lubbock, TX, USA

omareikr@ttu.edu

Abstract—Reasoning models allocate extra computation at inference time and present their answers as the product of deliberate thought. If this deliberation works the way dualprocess accounts of human cognition suggest, longer thinking should weaken the classic decision biases that fast, intuitive judgment produces. We test that prediction. Using 30 vignettes covering six biases (anchoring, framing, loss aversion, escalation of commitment, availability, confirmation) drawn from an established benchmark, we run a dose-response study across four model families, pairing each reasoning model with a matched non-reasoning sibling and requesting thinking ceilings of 0, 1,024, 4,096, and 8,192 tokens, for 12,350 API calls. Because a requested ceiling is not the same as realized deliberation, we log the reasoning tokens each call consumed and use realized consumption as the dose. Three results follow. First, reasoning models are not less biased than their siblings; the point estimate leans the other way in every family, but the item-level pooled contrast is not reliable (Delta = +0.031, t(29) = 1.45, p = .157). Second, bias magnitude does not reliably fall as realized deliberation grows: across the ranges these models actually produced, no slope is significantly negative, and where anything moves it is the signed score drifting further from the human direction. Third, anchoring is the only bias in the human direction (d = 1.89). Four of the other five lean the opposite way in all seven models; at the item level, with five items per bias, that reversal is reliable for framing and directional for escalation of commitment, confirmation, and loss aversion, while availability is absent. A one-line instruction to restate the anchor before answering lowered anchoring on all five anchoring items, which no amount of additional thinking did, although with five items the effect does not reach significance (p = .057). The results argue against treating test-time reasoning as a rationality guarantee and for auditing deployed models bias by bias.

Index Terms—cognitive bias, large language models, testtime compute, reasoning models, dual-process theory, anchoring, psychometric evaluation

## I. INTRODUCTION

Reasoning models such as OpenAI’s o-series, DeepSeek-R1, and the thinking variants of Claude and Qwen spend a controllable budget of tokens on an internal deliberation phase before committing to an answer [1]–[3]. The appeal is intuitive: a system that thinks longer should decide better, and test-time compute has in fact improved performance on mathematics, coding, and other verifiable tasks [4], [5].

These models are now placed in decision support, where the questions are not math problems but judgment calls: estimates, trade-offs, and choices between framed options. Whether extra thinking helps there is a different question, and it is the one this paper asks.

The question has a natural home in dual-process theory. In the human literature, biases such as anchoring and framing are attributed to fast, associative processing, and deliberate reasoning is the mechanism that can override them [6], [7]. The analogy between a reasoning trace and this deliberative mode is common in discussions of test-time compute, and it makes a concrete, falsifiable prediction: if the analogy holds, bias should shrink as deliberation grows. To our reading, this dose-response prediction has not been tested under controlled conditions, although its ingredients exist. Benchmarks now document a wide range of cognitive biases in language model [8]–[11], and separate work studies how reasoning traces relate to model behavior [12], [13]. What is missing is a design that holds the stimuli fixed, pairs reasoning models with matched non-reasoning siblings, and varies deliberation as an experimental factor while checking how much deliberation actually occurred.

The answer matters for practice. If deliberation reduces bias, allocating compute becomes a documented mitigation with a measurable cost-benefit curve. If it does not, then the trust that a longer visible deliberation invites is not backed by better judgment, and bias mitigation has to come from somewhere else. Either outcome changes how a deployed system should be configured; neither can be assumed from capability benchmarks.

We therefore run a dose-response study. From the 30,000- test benchmark of Malberg et al. [8] we freeze a battery of 30 decision vignettes covering six biases with distinct theoretical roots: anchoring [6], the framing effect [14], loss aversion [15], escalation of commitment [16], the availability heuristic [17], and confirmation bias [18]. Each vignette exists in a control and a bias-manipulated version, and every response is a choice on a fixed ordinal scale, which yields a bounded, direction-aware bias score. We administer the battery to seven models from four families (Anthropic, OpenAI, DeepSeek,

Qwen), crossing each reasoning model with requested thinking ceilings of 0, 1,024, 4,096, and 8,192 tokens and sampling each cell ten times, for 12,350 calls. A third condition on the anchoring items forces the model to restate the anchor before answering. For every call we log the reasoning tokens actually consumed, which turns out to matter: the requested ceiling and the realized deliberation diverge in ways that differ by provider.

The paper makes four contributions:

• a dose-response design for bias in reasoning models, with matched reasoning/non-reasoning pairs in four model families, a requested thinking ceiling as the manipulated factor, and realized reasoning tokens as the measured dose;

• evidence that reasoning does not confer resistance: no family shows a reasoning model with lower bias magnitude than its comparator, and no model’s bias magnitude reliably declines as realized deliberation grows;

• a presence-versus-reversal taxonomy in which anchoring alone appears in the human direction while four of the other five biases lean the opposite way in every model tested, with the item-level reliability of each reversal stated;

• a practical observation: forcing the model to verbalize the anchor lowered anchoring on every anchoring item, which no increase in deliberation did, at the cost of one prompt line.

We release the frozen battery, collection harness, scored dataset, per-call token usage, and analysis code.<sup>1</sup>

## II. RELATED WORK

This section situates the study in three lines of work: measurements of cognitive bias in language models, the testtime reasoning paradigm, and analyses of what reasoning traces actually do.

## A. Cognitive biases in language models

Behavioral experiments borrowed from psychology have become a standard instrument for studying language models. Binz and Schulz probed GPT-3 with classic vignettes and found several human-like errors [10]. Hagendorff et al. reported that intuitive biases present in earlier GPT models weakened or vanished in ChatGPT, an early sign that alignment training reshapes bias profiles [19]. Itzhak et al. showed the complementary effect: instruction tuning can also introduce biases that the base model lacked [20]. Larger inventories followed. Echterhoff et al. measured biases in managerial decision scenarios [9], and Malberg et al. built the benchmark used here, 30 biases instantiated in 200 scenarios and validated across 20 models [8]. Knipper et al. assessed eight biases across a broad model panel and reported bias-consistent behavior in 18 to 57 percent of instances; among their findings, the reasoning-oriented QwQ-32B and DeepSeek-R1 showed more resistance to framing specifically, with no comparable gain on other biases [11]. Related strands document social forms of bias pressure, including sycophancy [21], [22] and biases that emerge at the population level when model agents interact [23]. This literature establishes that biases can be measured, but it treats each model as a fixed object; the amount of inference-time deliberation is not manipulated, and whether a model’s response moves against the human direction, rather than merely failing to move with it, is not always distinguished.

## B. Test-time reasoning

Chain-of-thought prompting first showed that eliciting intermediate steps improves accuracy [24]. Current reasoning models internalize the idea: they are trained to produce an extended thinking phase whose length can be capped by the caller [1]–[3], and allocating more of this test-time compute can beat scaling parameters on verifiable tasks [4], [5]. The controllable budget is what makes the present study possible: it turns deliberation into an independent variable rather than a model property. As Section III-E shows, the control is less direct than the API surface suggests, which is why we measure consumption rather than trusting the request.

## C. What reasoning traces do

A separate line of work cautions against reading the trace as the computation. Turpin et al. showed that chain-of-thought explanations can rationalize answers driven by cues the text never mentions [12], and Chen et al. found the same pattern inside reasoning models, which verbalize the hints that moved them only a minority of the time [13]. Shojaee et al. documented reasoning effort that collapses precisely where problems get hard [25]. These results question the deliberation analogy from the inside, by inspecting traces. We question it from the outside, by testing its behavioral prediction: whatever the trace contains, more of it should produce less bias if it functions as System-2 deliberation. The two approaches are complementary, and our results agree with theirs.

## III. STUDY DESIGN

This section describes the research questions, the stimulus battery, the model panel, the experimental conditions and what the gateway did with them, the bias metric and its interpretation, and the collection procedure with its dataquality checks.

## A. Research questions

The study is organized around four questions, fixed before data collection.

• RQ1. Do reasoning models show smaller bias magnitudes than matched non-reasoning models of the same family?

• RQ2. Does bias decrease as deliberation grows?

• RQ3. Which of the six biases are present in the human direction, absent, or reversed, and which respond to deliberation?

• RQ4. Does forcing the model to verbalize the anchor change anchoring susceptibility?

TABLE I  
MODEL PANEL AND DESIGN CELLS PER MODEL. BUDGETS ARE REQUESTED EXTENDED-THINKING TOKEN CEILINGS.
<table><tr><td>Model</td><td>Family</td><td>Type</td><td>Budgets</td><td>Cells</td></tr><tr><td>claude-haiku-4.5-thinking claude-haiku-4.5</td><td>claude claude</td><td>reasoning instruct</td><td>0, 1k, 4k, 8k 0</td><td>120 30</td></tr><tr><td>gpt-5.1-reasoning</td><td>openai</td><td>reasoning</td><td>0, 1k, 4k, 8k</td><td>120</td></tr><tr><td>deepseek-r1</td><td>deepseek</td><td>no-disable</td><td>1k, 4k, 8k</td><td>90</td></tr><tr><td>deepseek-v3</td><td>deepseek</td><td>instruct</td><td>0</td><td>30</td></tr><tr><td>qwen3-thinking</td><td>qwen</td><td>no-disable</td><td>1k, 4k, 8k</td><td>90</td></tr><tr><td>qwen3-instruct</td><td>qwen</td><td>instruct</td><td>0</td><td>30</td></tr></table>

“no-disable” models cannot switch thinking off; their instruct siblings provide the family’s budget-0 point.

## B. Benchmark and battery

Stimuli come from the cognitive-bias benchmark of Malberg et al. [8], which instantiates each bias as paired prompts: a control version of a managerial decision scenario and a treatment version containing the bias manipulation, both answered on a fixed ordinal scale presented as numbered options. We selected six biases on theoretical grounds, covering numeric estimation (anchoring), choice architecture (framing, loss aversion, escalation of commitment), and belief-driven judgment (availability, confirmation). For each bias we sampled five items from distinct scenarios with a fixed seed (42) and froze the resulting 30-item battery before any experimental call was made; the battery did not change afterwards. Ten items use a 7-point scale and twenty use an 11-point scale. Each benchmark item carries a direction parameter $k \in \{ - 1 , + 1 \}$ and, where applicable, a scale-reversal flag for the treatment prompt, which we apply as specified so that positive scores always mean movement in the direction the bias predicts for humans.

## C. Model panel

Table I lists the panel: seven models from four families, chosen to span capability tiers and open versus closed weights while keeping a matched reasoning/non-reasoning pair inside each family. All calls were routed through a single API gateway (OpenRouter) so that the reasoning budget could be requested with one uniform parameter. Two properties of the panel required design decisions. For the OpenAI family no separate non-reasoning sibling of GPT-5.1 was available, so the comparator is the same model at budget 0 with thinking disabled, which holds the weights constant and toggles only deliberation. DeepSeek-R1 and the Qwen3 thinking variant cannot disable thinking at all: requests to turn it off were rejected or ignored, and at a zero budget the models thought anyway. We therefore treat budget 0 as undefined for these two models and use their instruct siblings (DeepSeek-V3, Qwen3- Instruct) as the family zero points, which is what the paired design was built for. This means the reasoning contrast is held at constant weights for the Claude and OpenAI families and crosses a sibling boundary for DeepSeek and Qwen; we return to this in the limitations.

## D. Conditions and budget manipulation

Every battery item was administered in a control and a treatment condition. Anchoring items received a third condition for RQ4: the treatment prompt plus one instruction to restate every number mentioned in the prompt before answering. All prompts ended with a fixed answer-format instruction (“Answer: Option ⟨number⟩”). Reasoning models ran at requested ceilings of 0, 1,024, 4,096, and 8,192 thinking tokens (the two no-disable models at the three positive ceilings); instruct models ran at budget 0 only. Each model × budget × item × condition cell was sampled ten times at temperature 1.0, giving 12,350 calls in total. Requests used asynchronous collection with retry and checkpointing; the token ceiling for thinking calls reserved a fixed answer allowance on top of the budget, which a pilot showed was necessary to keep long deliberations from truncating the visible answer.

## E. Requested ceiling versus realized deliberation

A requested ceiling is an instruction to the provider, not a measurement of what the model did. The gateway’s documentation states that Anthropic models honour the ceiling directly, that OpenAI reasoning models accept only an effort level and have the ceiling converted to one, and says nothing about how open-weight providers treat it. We therefore logged the reasoning tokens reported in the usage field of every response and treat that count, rather than the request, as the dose. Table II summarises the result, and it differs by provider in three ways.

For Claude, the ceiling was honoured and consumption rose with it, from zero at budget 0 to a mean of 437, 575, and 620 tokens at the three positive ceilings; no call exceeded its ceiling, and the model never came close to the larger ones. For GPT-5.1, thinking was off at budget 0 and nearly constant above it, at 116 to 118 tokens regardless of the requested ceiling, which is the effort-tier conversion at work: the three requested levels produced one realized level. For DeepSeek-R1 and Qwen3, the ceiling was not enforced. Both models consumed between 1,074 and 1,617 tokens on average at every setting including budget 0, and at the 1,024 ceiling 79% of R1 calls and 52% of Qwen3 calls exceeded it. Consumption rose slightly between the 1,024 and 4,096 settings and was flat beyond.

Two consequences follow for the analysis. The budget-0 contrast for Claude and GPT-5.1 is a true zero-deliberation comparison at constant weights. And the dose-response question (RQ2) is answered on realized tokens: per model, we regress bias on $\log _ { 2 }$ (realized tokens + 1) within the range that model actually produced, rather than on the requested label.

## F. Bias metric

Let $\mu _ { C }$ and $\mu _ { T }$ be the mean chosen option index across the ten samples of a cell’s control and treatment condition, on a scale with n options, after applying the benchmark’s treatmentscale reversal where flagged. The cell’s bias score is

$$
s ~ = ~ k \cdot \frac { \mu _ { T } - \mu _ { C } } { n - 1 } ~ \in ~ [ - 1 , + 1 ] ,\tag{1}
$$

TABLE II  
REALIZED REASONING TOKENS PER CALL (CELL MEANS) AGAINST THE REQUESTED CEILING.
<table><tr><td>Model</td><td> ${ \overline { { b = 0 } } }$  一</td><td> $\overline { { b = 1 , 0 2 4 } }$  一</td><td> $\overline { { b = 4 , 0 9 6 } }$  一</td><td> $\overline { { b = 8 , 1 9 2 } }$ </td></tr><tr><td>claude-haiku-4.5-thinking</td><td>0</td><td>437</td><td>575</td><td>620</td></tr><tr><td>gpt-5.1-reasoning</td><td>0</td><td>116</td><td>118</td><td>118</td></tr><tr><td>deepseek-r1</td><td>1,188</td><td>1,555</td><td>1,605</td><td>1,617</td></tr><tr><td>qwen3-thinking</td><td>1,074</td><td>1,321</td><td>1,440</td><td>1,436</td></tr></table>

Honoured for Claude; converted to an effort tier for GPT-5.1 (flat consumption); not enforced for DeepSeek-R1 and Qwen3, where 79% and 52% of calls exceeded the 1,024 ceiling. The $b = 0$ cells for the two no-disable models are the excluded conditions described in the text.

where k is the benchmark’s direction parameter. A positive s means the treatment moved the model in the direction the bias predicts for humans; a negative s means it moved the model the opposite way. We analyze s (signed) and |s| (magnitude, i.e., susceptibility regardless of direction). The verbalized condition yields an analogous score $s _ { v }$ against the same control.

What the score measures deserves one paragraph. The control and treatment versions of an item carry the same decision-relevant information; by construction they differ only in the manipulation (a numeric anchor, a gain or loss frame, a prior commitment, and so on) that the bias literature identifies as normatively irrelevant to the choice. Any systematic shift between the two therefore indexes sensitivity to an irrelevant feature of the prompt. That is the sense in which we use “bias”: susceptibility to the manipulation, not a judgment about the quality of either answer in isolation. Magnitude |s| is the size of that susceptibility, and the sign says whether the shift runs with or against the human pattern. A model with $| s | = 0$ is one whose answer does not depend on the irrelevant feature; a model with large |s| in either direction is one whose answer does.

## G. Procedure and data quality

All 12,350 calls completed without API failures. Answers were extracted with a tiered, range-aware parser (strict format first, then boxed and free-text patterns, all restricted to the item’s option range so that echoed stimulus numbers cannot be mistaken for answers). On analyzable data the parse rate was 99.4%, at or near 100% for five of the seven models. Two exclusions are documented rather than repaired. First, the budget-0 responses of the two no-disable models are excluded by design as described above. Second, Qwen3- Instruct provided a raw quantity (for example, a percentage) instead of an option index in 7.2% of its responses, mostly on anchoring estimation items; these responses were excluded, leaving seven of its 65 condition cells below eight samples but every cell with at least two; five scored cells are affected, 1.0% of the 510-cell grid. The scored dataset contains one row per model × budget × item cell (510 rows).

The unit of independent observation is the item. Where a comparison pools across models or budgets, the same 30 items recur, so we aggregate to the item level before testing and report item-clustered bootstrap intervals. Where a test is confined to one bias, only its five items are independent, so per-bias tests use the t distribution with four degrees of freedom on the five item means; Wald tests from mixed models assume many clusters and overstate significance at this size. Mixed models with a random item intercept are used where all 30 items enter. Within-family contrasts use paired t-tests and Wilcoxon signed-rank tests over the 30 items with Cohen’s $d _ { z } ;$ dose-response uses per-item ordinary-least-squares slopes on $\log _ { 2 }$ (realized tokens + 1) together with mixed models; Holm correction is applied within each family of tests.

TABLE III  
RQ1: BIAS MAGNITUDE |s| FOR REASONING MODELS (BUDGETS $> 0 )$ VERSUS MATCHED NON-REASONING COMPARATORS, PAIRED BY ITEM (n = 30).
<table><tr><td>Family</td><td> $\overline { { M _ { \mathrm { r e a s } } } }$ </td><td> $\overline { { M _ { \mathrm { i n s t r } } } }$ </td><td> $\overline { { \Delta } }$ </td><td>95% CI</td><td>t(29)</td><td> ${ p / } \mathrm { H o l m }$ </td><td> $\overline { { d _ { z } } }$ </td></tr><tr><td>claude</td><td>0.271</td><td>0.244</td><td>+0.027</td><td>[-0.024,0.088]</td><td>0.95</td><td>.714</td><td>0.17</td></tr><tr><td>deepseek</td><td>0.275</td><td>0.249</td><td>+0.026</td><td>[-0.013, 0.070]</td><td>1.20</td><td>.714</td><td>0.22</td></tr><tr><td>openai</td><td>0.299</td><td>0.243</td><td>+0.056</td><td>[-0.014, 0.140]</td><td>1.40</td><td>.686</td><td>0.26</td></tr><tr><td>qwen</td><td>0.301</td><td>0.287</td><td>+0.015</td><td>[−0.046, 0.079]</td><td>0.45</td><td>.714</td><td>0.08</td></tr></table>

Positive ∆ = reasoning model more biased. Pooled at the item level: $\Delta = + 0 . 0 3 1 , t ( 2 9 ) = 1 . 4 5 , p = . 1 5 7 , d _ { z } = 0 . 2 7 .$

## IV. RESULTS

This section reports the four research questions in turn. Descriptively, mean signed scores are negative for every model at every budget (range −0.23 to −0.10), which already hints that the dominant behavior is movement against the human bias directions; the exception, anchoring, is isolated in Sec. IV-C.

## A. RQ1: Reasoning versus matched non-reasoning models

Table III compares bias magnitude |s| for each reasoning model (averaged over its positive budgets) against its family comparator at budget 0, paired by item. In no family is the reasoning model less biased. All four point estimates lean the same way, with the reasoning model slightly higher $( d _ { z } = 0 . 0 8$ to 0.26), and none is reliable after Holm correction. Pooled across families by averaging each item’s four differences, so that the 30 items remain the independent units, the difference is $\Delta = + 0 . 0 3 1$ , 95% bootstrap CI $\left[ - 0 . 0 1 0 , + 0 . 0 7 4 \right]$ $t ( 2 9 ) = 1 . 4 5 , p = . 1 5 7 , d _ { z } = 0 . 2 7 ;$ the Wilcoxon test agrees $\left( p \ = \ . 2 5 3 \right)$ , as does a mixed model with a random item intercept $( z = 1 . 4 5 , p = . 1 4 7 )$ . We read this as no evidence that reasoning models are the lower-bias choice on these tasks. The consistent direction across families is worth noting but not worth a claim; the pooled interval includes zero. On signed scores no family shows a reliable difference either.

## B. RQ2: Dose-response over realized deliberation

Fig. 1 plots bias against the requested ceiling for orientation; Table IV gives the statistics that matter, computed on realized tokens within each model’s observed range. No model shows a reliable decline in magnitude. For Claude, whose realized deliberation spans 0 to about 800 tokens per cell, the peritem slope of |s| is indistinguishable from zero (+0.0029 per doubling, $t ( 2 9 ) = 0 . 9 1 , p = . 3 7 2 ;$ ; mixed model $p = . 2 5 9 )$ For DeepSeek-R1 (roughly 700 to 2,800 tokens per cell) and

![](images/b15d601e651ca838f2304a9ad284e6be0b4e256e7cc2a72f11937f488f27907f.jpg)

![](images/2890fec0f10777545e70115dae4a0da3110be7a7e7fed8326056a5786b6c40d3.jpg)  
Fig. 1. Bias against the requested reasoning ceiling (error bars: 95% bootstrap CIs). Left: magnitude |s|. Right: signed score s. The x-axis is the requested ceiling; Table II gives the deliberation each setting actually produced, and Table IV tests the trend on that realized quantity.

TABLE IV  
RQ2: TREND OF BIAS ON log (REALIZED REASONING TOKENS + 1), PER MODEL, WITHIN THE REALIZED RANGE.
<table><tr><td>Model</td><td>Range (tokens)</td><td> $\overline { { { \mathrm { s l o p e } } \ t ( 2 9 ) , p } }$ </td><td> $\overline { { \beta _ { \vert s \vert } \ ( p ) } }$ </td><td>βs (p)</td></tr><tr><td>claude-th.</td><td>0-805</td><td>0.91, .372</td><td>+0.003 (.259)</td><td>-0.006 (.018)</td></tr><tr><td>gpt-5.1</td><td>0-225</td><td>1.36, .184</td><td>+0.008 (.018)</td><td>-0.010 (.010)</td></tr><tr><td>deepseek-r1</td><td>707-2807</td><td>-1.63, .114</td><td>+0.003 (.958)</td><td>-0.018 (.776)</td></tr><tr><td>qwen3-th.</td><td>534-3440</td><td>0.25, .805</td><td>-0.066 (.213)</td><td>+0.093 (.075)</td></tr></table>

Range is the span of cell-mean realized tokens. “slope” tests the per-item OLS slope of |s| against zero; $\beta$ coefficients are from mixed models with random item intercepts, per doubling of realized tokens. No |s| trend is significantly negative.

Qwen3 (roughly 500 to 3,400) the slopes are likewise null $( p = . 9 5 8$ and $p = . 2 1 3$ in the mixed models). GPT-5.1 is the one model whose magnitude rises with realized tokens $( \beta = + 0 . 0 0 8 4 , p = . 0 1 8 )$ , but its realized range is 0 to about 225 tokens, so this is in effect the budget-0 versus any-thinking contrast of RQ1 rather than a dose gradient.

Where deliberation does anything, it is to the signed score, and it moves away from the human direction rather than toward zero: Claude $( \beta = - 0 . 0 0 5 9$ per doubling, $p = . 0 1 8 )$ and GPT-5.1 (β = −0.0104, $p = . 0 1 0 )$ both drift further negative as realized tokens increase; DeepSeek-R1 and Qwen3 do not $( p = . 7 7 6$ and $p = . 0 7 5$ , the latter in the positive direction). Taken together, across the range of deliberation these models actually produced, which the requested ceilings expanded for Claude but not for the others, no model’s susceptibility to the manipulations reliably decreased.

## C. RQ3: Which biases exist, and in which direction

Table V classifies each bias from the signed score over all 85 model × budget cells per bias, testing the five item means against zero with t(4) so that inference respects the five items that instantiate each bias; Fig. 2 shows the full bias × model map. Two statements have different evidential status here, and we keep them apart.

The first is about models. In all seven models, anchoring is positive in mean and framing, escalation of commitment, confirmation bias, and loss aversion are negative (Fig. 2); availability is negative in six of the seven. The pattern replicates across four training pipelines and both open and closed weights. Anchoring is present in the human direction and strongly so (M = +0.163, 95% CI [+0.111, +0.215], $t ( 4 ) = 5 . 0 2 , p _ { \mathrm { H o l m } } = . 0 3 7 , d = 1 . 8 9 ) \colon$ a numeric anchor pulls estimates toward itself in every model, from +0.11 (Qwen3- thinking) to +0.20 (Claude instruct).

TABLE V  
RQ3: PRESENCE CLASSIFICATION PER BIAS. SIGNED SCORE VS. 0 OVER 85 CELLS; t(4) ON THE FIVE ITEM MEANS; ITEM-CLUSTERED BOOTSTRAP CIS; HOLM-CORRECTED.
<table><tr><td>Bias</td><td>M</td><td>95%CI</td><td>t(4)</td><td>PH</td><td>d</td><td>Class</td></tr><tr><td>Anchoring</td><td> $\overline { { + . 1 6 3 } }$ </td><td> $[ + . 1 1 1 , + . 2 1 5 ]$ </td><td>5.02</td><td>.037</td><td>+1.89</td><td>Present</td></tr><tr><td>Framing</td><td>-.336</td><td>[−.403, −.245]</td><td>-7.57</td><td>.010</td><td></td><td>-2.35 Reversed</td></tr><tr><td>Escalation</td><td>-.070</td><td>[−.119, −.031]</td><td>-2.79.197</td><td></td><td></td><td>—0.68 Directional</td></tr><tr><td>Confirmation</td><td>-.139</td><td>-.249, −.043]</td><td>-2.38.227</td><td></td><td></td><td>-0.65 Directional</td></tr><tr><td>Loss aversion</td><td>-.586</td><td>-.914,+.031]</td><td>-1.91.256</td><td></td><td></td><td>—0.88 Directional</td></tr><tr><td>Availability</td><td>-.035</td><td>[−.068, +.005]</td><td>-1.71 .256 -0.42 Absent</td><td></td><td></td><td></td></tr></table>

d is the cell-level standardized mean. $\mathrm { \ddot { \cdot } P r e s e n t ^ { \ast } = }$ human-like direction; “Reversed” = reliable shift against it at the item level; “Directional” = opposite to the human direction in all seven models, but the item-level test does not reach .05 after correction with five items per bias; “Absent” = neither reliable nor consistent in sign across models.

The second is about biases as constructs, which requires generalizing over items, and with five items per bias that test is weaker. The framing effect, which in people flips risk preferences between gain and loss frames [14], runs backwards here and reliably so $( M = - 0 . 3 3 6 , t ( 4 ) = - 7 . 5 7 ,$ , p<sub>Holm</sub> = .010, $d = - 2 . 3 5 )$ . Escalation of commitment is reversed in every model and its interval excludes zero $( M = - 0 . 0 7 0 $ $[ - 0 . 1 1 9 , - 0 . 0 3 1 ] )$ , but its item-level test does not survive correction $( t ( 4 ) = - 2 . 7 9 , p _ { \mathrm { H o l m } } = . 1 9 7 ) ;$ confirmation bias is similar $( M ~ = ~ - 0 . 1 3 9 , ~ p _ { \mathrm { H o l m } } ~ = ~ . 2 2 7 )$ . Loss aversion shows the largest mean reversal in the battery $( M = - 0 . 5 8 6$ reaching −0.87 for Qwen3-Instruct) but also the widest itemclustered interval $( [ - 0 . 9 1 4 , + 0 . 0 3 1 ] , p _ { \mathrm { H o l m } } = . 2 5 6 )$ : the five loss-aversion items disagree about how far the models lean, even though all seven models lean the same way. Availability is small, not reliable at the item level $( M ~ = ~ - 0 . 0 3 5$ $p _ { \mathrm { H o l m } } = . 2 5 6 )$ , and positive in one model (Qwen3-thinking, $+ 0 . 0 3 )$ , so we classify it as absent. We therefore describe framing as reversed; escalation, confirmation, and loss aversion as directional, meaning opposite to the human direction in every model with item-level reliability that five items cannot establish; and availability as absent. No bias qualifies as deliberation-responsive; all six per-bias slope tests on realized tokens are null after correction.

## D. RQ4: Forcing the model to verbalize the anchor

The verbalized condition adds one instruction to the anchoring treatment: restate every number in the prompt before answering. The change runs in the helpful direction on every one of the five anchoring items, with item means from −0.004 to −0.036 and a pooled mean of $\Delta ~ = ~ - 0 . 0 1 5 ~ ( 9 5 \%$ CI $\left[ - 0 . 0 2 6 , - 0 . 0 0 4 \right]$ over cells). Tested at the item level, with five independent items, the effect falls just short of significance $( t ( 4 ) = - 2 . 6 5 , p = . 0 5 7 ;$ a sign test on five of five items gives $\begin{array} { r } { p \ = \ . 0 6 3 ) } \end{array}$ . Five of the seven models also move in that direction (Fig. 3). We therefore report the verbalization effect as directional rather than established: consistent across items and most models, small, and resting on five items. Its comparison point is still informative. A single prompt line moved anchoring the helpful way on every item, while no level of realized deliberation in RQ2 moved it at all.

![](images/aca7b25737eaa2da688686866aeb871c2b909ad8de5eb374d22962b4d8b12aeb.jpg)  
Fig. 2. Mean signed bias score by bias type and model. Red indicates movement in the human bias direction, blue the reverse. Anchoring is the only red row. The other biases are blue in every model, reasoning and instruct alike, with one exception: availability in Qwen3-thinking.

![](images/a7a9d20e0ee9e5b7e14de58a7ec825e939be598f51e4bcaba98873fc8237fe60.jpg)  
Fig. 3. Anchoring with and without forced anchor verbalization, by model. Verbalization lowers anchoring in five of seven models and on all five anchoring items; at the item level the reduction falls short of significance $( t ( 4 ) = \breve { - } 2 . 6 5 , p = . 0 5 7 )$

## V. DISCUSSION

This section interprets the main findings and draws out what they mean for deployment and for the deliberation analogy.

## A. A bias landscape that leans the wrong way

The most consistent result is not about reasoning at all. Across seven models spanning four training pipelines and both open and closed weights, anchoring runs in the human direction and four of the other five biases lean against it in every model (Fig. 2), while availability is small and inconsistent. At the item level that reversal is firmly established for framing and directional for escalation, confirmation, and loss aversion, so the strong form of the claim rests on one bias and the weaker form on four.

This picture departs from the benchmark results of Malberg et al. [8], who report bias-consistent behavior across their model panel, and from the one-sided framing in Knipper et al. [11], where reasoning models show more resistance to framing. Three differences in method bear on the comparison. We sample ten responses per cell at temperature 1.0 and score the mean shift, rather than scoring single responses; we use five items per bias drawn from their pools; and our panel consists of models released in 2025 and later, after the alignment generations those studies evaluated. We cannot attribute the divergence to any one of these from our data. What we can say is that, for four biases, the sign of the reversal is uniform across our seven models, which is not what item sampling alone would produce, and that a onesided “resistance” finding and a signed “reversal” finding are compatible: both say that reasoning models do not frame like people, and the signed metric adds that they move the other way.

One reading of the reversal is that this is what heavy preference tuning looks like from the outside. Hagendorff et al. observed human-like biases weakening from GPT-3 to ChatGPT [19], and Itzhak et al. showed that instruction tuning reshapes bias profiles in both directions [20]; our data are consistent with that trajectory continuing past neutrality. Loss aversion is the suggestive case: its reversal is what a riskneutral expected-value maximizer would produce on these choice pairs, and models trained to give “rational” answers may have internalized that norm, though with five items and a wide interval we present this as a hypothesis rather than a finding. Whatever the mechanism, the practical upshot holds for the biases that are reliably reversed: the familiar human bias catalog is the wrong checklist for auditing these systems. A review that asks “is the model framing-sensitive like a person?” will answer no and miss that the model deviates from frame-invariance just as far the other way.

## B. No debiasing dividend from deliberation

Against this backdrop, the reasoning results point one way. Matched within families, reasoning models are not less biased than their comparators (RQ1); the point estimates lean toward slightly more bias in every family, but the pooled interval includes zero and we do not claim a difference. Across the deliberation these models actually produced (RQ2), no model’s bias magnitude reliably declined. The qualification that the realized-token analysis adds is important: for GPT-5.1 the ceiling did not change how much the model thought, and for DeepSeek-R1 and Qwen3 the ceiling was not enforced, so those three models provide a narrow dose range. Claude is the one model whose deliberation the ceiling controlled, from zero to several hundred tokens, and across that range its bias magnitude was flat while its signed score drifted slightly further from the human direction. The dual-process prediction, more deliberation, less bias, finds no support in the range tested. This does not mean the thinking is idle; on verifiable tasks it demonstrably helps [4]. It means the deliberation does not function as a bias corrector here, which fits the trace-level evidence that reasoning text and answer-driving computation can come apart [12], [13], [25]. Anchoring makes the point concretely: the anchor keeps its pull at the highest realized deliberation we observed, so whatever those tokens do, they do not include discounting the number that is biasing the answer.

A methodological lesson sits beside the substantive one. A requested thinking ceiling, sent through a multi-provider gateway, produced three different realities: a ceiling that was honoured but rarely approached, a ceiling converted into a single effort setting, and a ceiling that was ignored. Studies that manipulate reasoning budget and do not log consumption cannot tell these cases apart. We recommend reporting realized reasoning tokens per condition as a matter of course.

## C. Anchor restatement versus deliberation

The one intervention that moved anchoring was not more compute but a redirection of attention: asking the model to restate the anchor before answering lowered anchoring on all five items (RQ4). The size is modest, one model (DeepSeek-V3) shifted the other way, and with five items the effect does not reach significance, so we present this as a cheap mitigation worth testing rather than an established one. Still, the contrast with RQ2 is the useful part. For this bias, one line of prompt engineering did what the available deliberation did not, which suggests that targeted prompting and per-bias audits are currently a better use of effort than budget increases when the goal is judgment quality rather than task accuracy.

## D. Implications for practice

Three recommendations follow. First, do not assume the reasoning tier of a model family is the lower-bias option; in our panel it never was, and the evidence gives no reason to expect it. Second, audit deployed models per bias and per direction, since for four of the six biases the sign of the deviation was model-agnostic here but its size varied by family. Third, where numeric anchors are a known hazard, an anchor-restating instruction is a low-cost mitigation worth testing in context.

## VI. LIMITATIONS

The design has boundaries worth stating plainly. The stimuli are vignettes from a single benchmark family; vignette-based measurement is the standard paradigm in both the human and machine literatures [6], [8], but effects can be sensitive to phrasing, and five items per bias is a narrow base. That narrowness is why three of the four reversals and the RQ4 verbalization effect are reported as directional rather than reliable, and why availability is classified as absent: with five items per bias, an item-level test has four degrees of freedom, and effects that are consistent in sign across every item and model can still fall short of significance. The deliberation manipulation did not work uniformly: realized reasoning tokens rose with the requested ceiling for Claude, were constant for GPT-5.1, and were uncapped for DeepSeek-R1 and Qwen3, so the dose-response conclusion is bounded by the realized ranges in Table IV, and we cannot speak to deliberation beyond roughly 3,400 tokens per cell. For the two no-disable models the reasoning contrast crosses a sibling boundary and so confounds deliberation with weights; the Claude and OpenAI contrasts hold weights constant and are the cleaner evidence for RQ1. Our claims are behavioral: the study measures what the models do, not why, and the overcorrection reading of the reversals is an interpretation the data are consistent with, not a demonstrated mechanism. Finally, all calls went through one routing provider during one collection window; model updates behind fixed identifiers are a general reproducibility hazard, which the released harness, frozen battery, and per-call usage logs mitigate but cannot remove.

## VII. CONCLUSION

We asked whether test-time reasoning immunizes language models against human cognitive biases, and within the deliberation these models actually produced, the answer is no on every axis measured: reasoning models are not less biased than matched siblings, bias magnitude does not reliably fall as realized deliberation grows, and the one bias that behaves like its human counterpart, anchoring, is as strong at the highest deliberation observed as at none. What the data show instead is a bias landscape that leans against the human catalog, consistently across seven models for four biases and reliably so for framing, together with a small, cheap prompt intervention that moved anchoring on every item where compute did not. A requested thinking budget turned out to be a poor proxy for thinking done, which is itself a finding this literature should absorb. For a venue concerned with cognitive machine intelligence, the message cuts both ways. The dual-process analogy earns its keep as a hypothesis generator, but its central prediction fails in the range tested, and evaluation practice should follow the data: measure biases in both directions, per model, log the deliberation that actually occurred, and treat requested budget as a cost knob rather than a rationality guarantee.

## REFERENCES

[1] A. Jaech, A. Kalai, A. Lerer et al., “OpenAI o1 system card,” arXiv preprint arXiv:2412.16720, 2024.

[2] D. Guo, D. Yang, H. Zhang et al., “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.

[3] A. Yang, A. Li, B. Yang et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[4] C. Snell, J. Lee, K. Xu, and A. Kumar, “Scaling LLM test-time compute optimally can be more effective than scaling model parameters,” arXiv preprint arXiv:2408.03314, 2024.

[5] N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candes, and T. Hashimoto, “s1: Simple\` test-time scaling,” arXiv preprint arXiv:2501.19393, 2025.

[6] A. Tversky and D. Kahneman, “Judgment under uncertainty: Heuristics and biases,” Science, vol. 185, no. 4157, pp. 1124–1131, 1974.

[7] J. S. B. T. Evans and K. E. Stanovich, “Dual-process theories of higher cognition: Advancing the debate,” Perspectives on Psychological Science, vol. 8, no. 3, pp. 223–241, 2013.

[8] S. Malberg, R. Poletukhin, C. M. Schuster, and G. Groh, “A comprehensive evaluation of cognitive biases in LLMs,” in Proceedings of the 5th International Conference on Natural Language Processing for Digital Humanities (NLP4DH), 2025, arXiv:2410.15413.

[9] J. Echterhoff, Y. Liu, A. Alessa, J. McAuley, and Z. He, “Cognitive bias in decision-making with LLMs,” in Findings of the Association for Computational Linguistics: EMNLP 2024, 2024.

[10] M. Binz and E. Schulz, “Using cognitive psychology to understand GPT-3,” Proceedings of the National Academy of Sciences, vol. 120, no. 6, p. e2218523120, 2023.

[11] R. A. Knipper, C. S. Knipper, K. Zhang, V. Sims, C. Bowers, and S. Karmaker, “The bias is in the details: An assessment of cognitive bias in LLMs,” arXiv preprint arXiv:2509.22856, 2025.

[12] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, “Language models don’t always say what they think: Unfaithful explanations in chainof-thought prompting,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[13] Y. Chen, J. Benton, A. Radhakrishnan et al., “Reasoning models don’t always say what they think,” arXiv preprint arXiv:2505.05410, 2025.

[14] A. Tversky and D. Kahneman, “The framing of decisions and the psychology of choice,” Science, vol. 211, no. 4481, pp. 453–458, 1981.

[15] D. Kahneman and A. Tversky, “Prospect theory: An analysis of decision under risk,” Econometrica, vol. 47, no. 2, pp. 263–291, 1979.

[16] B. M. Staw, “Knee-deep in the big muddy: A study of escalating commitment to a chosen course of action,” Organizational Behavior and Human Performance, vol. 16, no. 1, pp. 27–44, 1976.

[17] A. Tversky and D. Kahneman, “Availability: A heuristic for judging frequency and probability,” Cognitive Psychology, vol. 5, no. 2, pp. 207– 232, 1973.

[18] R. S. Nickerson, “Confirmation bias: A ubiquitous phenomenon in many guises,” Review of General Psychology, vol. 2, no. 2, pp. 175–220, 1998.

[19] T. Hagendorff, S. Fabi, and M. Kosinski, “Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in ChatGPT,” Nature Computational Science, vol. 3, pp. 833–838, 2023.

[20] I. Itzhak, G. Stanovsky, N. Rosenfeld, and Y. Belinkov, “Instructed to bias: Instruction-tuned language models exhibit emergent cognitive bias,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 771–785, 2024.

[21] M. Sharma, M. Tong, T. Korbak et al., “Towards understanding sycophancy in language models,” in International Conference on Learning Representations (ICLR), 2024.

[22] A. Fanous, J. Goldberg, A. A. Agarwal et al., “SycEval: Evaluating LLM sycophancy,” in Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES), 2025.

[23] A. F. Ashery, L. M. Aiello, and A. Baronchelli, “Emergent social conventions and collective bias in LLM populations,” Science Advances, vol. 11, no. 20, p. eadu9368, 2025.

[24] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022.

[25] P. Shojaee, I. Mirzadeh, K. Alizadeh, M. Horton, S. Bengio, and M. Farajtabar, “The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity,” arXiv preprint arXiv:2506.06941, 2025.