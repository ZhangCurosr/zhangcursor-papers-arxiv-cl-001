# Before You Continue: Evaluating Time-Sensitive Safety Awareness in Conversational Assistants

Peidi Dong Yunze Xiao Yu-Lin Tsai Yichen Zheng

Heyuan Huang Weihao Xuan Mark Dredze Jen-tse Huang

Carnegie Mellon University University of California, Berkeley The Hong Kong Polytechnic University

Johns Hopkins University The University of Tokyo RIKEN AIP

## Abstract

Large language model (LLM) assistants increasingly help users with everyday activities that involve time-dependent constraints. However, users may not mention these constraints in later requests, requiring assistants to recognize and adjust their advice accordingly. To evaluate such capability, we introduce TASC<sup>1</sup> (Temporal Action Safety in Conversation), a benchmark that evaluates whether LLMs apply time-dependent constraints when responding to conversational requests. TASC consists of 3,418 inputs derived from 128 templates, along with a separate set of 33 scenarios designed to compare models’ ability to calculate elapsed time with their ability to apply that information. Across 13 models, assistants account for timedependent constraints in only 30.5% of cases where the proposed action violates a rule, while rarely warning users unnecessarily. More importantly, models fail to apply these constraints in 62.1% of cases despite correctly calculating the elapsed time. Explicitly providing the timing resolves only about half of these failures, revealing a gap between temporal reasoning and its application in conversational advice.

## 1 Introduction

As large language models (LLMs) have shifted from tools to assistants, users are increasingly seeking advice and knowledge from LLM-backed conversational chatbots (Chatterji et al., 2025). These chatbots accompany people through activities that last over hours or days, such as taking medication, preparing for a procedure, curing a sealant, or storing food. However, users often assume that LLMs are already like real assistants, ignoring the limitations of these chatbots, which often leads to harm to their own safety.

Consider the first example in Figure 1. A user has an 11:30 appointment for a tooth extraction under sedation, and the LLM has reminded them that clear fluids are allowed only until two hours before arrival. At 10:05, the user asks whether big gulps or steady sips are better when drinking a bottle of water. The cutoff passed 35 minutes ago, and fasting before sedation is an official medical advice (American Society of Anesthesiologists, 2017). In our evaluation, Kimi K3 unsafely replies to the question, suggesting steady sips.

As seen from the above example, the model answers the user’s latest question well but does not apply the stated fasting rule when giving advice to remind the user not to drink. Merely answering the latest question is not enough for a good assistant, as users frequently return with followup questions asking only how to proceed, without mentioning the rules that still apply. Therefore, whether their next action is safe may depend on when an earlier context took place. A safe assistant must notice when a later request violates such a rule and change the advice accordingly or ask for missing timing information when needed. We refer to the joint ability to reason about time and apply relevant safety constraints as temporal safety in conversation, which motivates a basic question:

How reliably do conversational assistants use temporal reasoning to recognize when a user’s action violates a safety constraint and adjust their advice, without being explicitly asked to check the timing?

To measure this aspect, we introduce a benchmark, TASC (Temporal Action Safety in Conversation). The main set (TASC-M) contains 3,418 scenarios derived from 128 templates across 23 everyday domains. Each scenario pairs an earlier event and a time-dependent rule to a later user request. When the user’s action indicates a violation with the rule, a successful response must identify the violation and respond accordingly. Scenarios vary on six factors, such as when the user acts, how the latest message indicates a violation, or how time information is presented. For further control testing in addition to the ordinary request shown in the above example (Condition B), we introduce Condition A, directly asking the time, and Condition C, explicitly stating the time based on the ordinary requests. Finally, we include requests that satisfy the timing rule, termed non-violation controls, and cases where the assistant must ask for missing timing information. These cases enable more detailed analyses on model behavior.

![](images/f99b601a283a6355d67822317e4381e2bc41fc0a7de82d2e1f90ac9d37f5c05f.jpg)  
Figure 1: Four examples from the TASC-P of TASC. TASC covers 23 everyday domains, and we evaluate 13 models. The “Rule | Time” row gives the time-dependent rule that the model should apply and the time elapsed (or remaining) at the final user message. (i): the conversation history shared by all conditions. (ii) (Condition A): asked directly how much time has passed, models answer correctly in 85.0% of cases. (iii) (Condition B): given an ordinary request in which the user is about to break, or has already broken, the rule, models flag the violation in only 33.2% of cases and usually just answer the request. (iv) (Condition C): even when the timing is stated explicitly in the system prompt, models still miss 37.4% of violations.

We evaluate 13 models and find that they catch only 30.5% of violations, with CATCH rate ranging from 0.1% to 84.9% across models. We further discovered that the temporal reasoning is not the issue as models answer 92.4% of duration queries correctly, and 62.1% of correct duration answers coincide with an uncaught violation on the same scenario. Stating the timing explicitly raises the CATCH rate but recovers fewer than half of the misses. Finally, CATCH rate even drops when merely replacing standard timestamps with verbal cues. Note that the safety risk implied in this paper represents only a lower-bound as most common LLM chatbots do not add timestamps to messages. Our contributions are as follows:

• We create TASC, a benchmark with controlled variations for measuring model behavior on temporally safe actions.

• We evaluate 13 models, showing that failure is common among different models, indicating a universal safety risk.

• We conduct analyses showing where correct duration answers fail to translate into safe advice and how much explicit timing helps.

## 2 Constructing TASC

## 2.1 Overview

Each TASC scenario is built from a template with three parts: an earlier event, a time-dependent rule attached to that event, and a later user request whose surface topic does not mention the rule. The rule takes one of three forms: a minimum wait (e.g., hold a tablet for at least 48 hours), a deadline (e.g., act within 30 minutes), or a target window (e.g., draw blood between 110 and 130 minutes). A template is expanded into many inputs by varying the six factors below. Every input carries a gold response type that follows by rule from the template and the values, so no gold label is written by hand. There are four gold types: a violation, where the requested action breaks the rule and the reply should warn; a non-violation control, where the action satisfies the rule; a possible violation, where no time evidence is given and the reply should ask for it; and an emergency, where the template requires urgent action.

Timing. The final user message falls before, near, or after the boundary of the rule. For a minimum wait, acting before the boundary is a violation; for a deadline, acting after it is a violation; for a target window, both sides are violations. Near-boundary inputs sit just on the permitted side of the boundary and serve as the hardest non-violation controls, since the elapsed time is close to the threshold but the action is allowed. A further 20 inputs from 10 templates place a conflicting earlier statement in the history; we report them with the violations.

Request. The final message relates to the constrained action in one of three ways. In an ongoing request, the user says the action is happening now (“Second coat’s going on now — should I thin it?”). In a done request, the user reports the action as completed and asks about the next step. In a presupposed request, the user never states the action and only asks a question that assumes it (“Does the iron sit better with something in my stomach?”). Each step moves the violation one inference further from the surface of the message.

Time format. Time evidence reaches the model in one of four formats. With ISO timestamps, every message carries a [YYYY-MM-DD HH:MM] prefix, and the system prompt states that these are the real sending times. With current time only, the opening message states the time of the event and the system prompt gives the current time. With a verbal cue, no clock time appears at the request; a natural phrase implies the elapsed time (“Slept the night through, had breakfast —”). With none, all timestamps are removed. Because the model cannot then determine whether the rule is violated, the expected response under none differs from the other formats: instead of a warning, a correct reply asks the user how much time has passed.

Rule source. The rule is stated by the assistant in an earlier turn, by the user, or by neither. In the last case, the rule is implied by common knowledge about the activity and must be supplied by the model itself.

Distractors. Most conversations place the request shortly after the rule. A subset inserts two or eight additional small-talk exchanges between them, with the elapsed time held fixed. The longest conversations contain 21 messages, and the rule sits up to 19 messages above the request.

Risk. Each template carries a risk tier from 0 to 3. Tier 0 covers harmless outcomes such as over-steeped cold brew; tier 1 covers material or financial loss such as a failed caulk seal or a late rental return; tier 2 covers possible injury or illness, such as infant feeding, pest treatments, or chemical exposure; tier 3 covers medication, medical procedures, and emergencies. Tiers are assigned by the authors. Tier-3 thresholds are taken from drug labels or published clinical guidance rather than chosen freely.

The retained collection samples combinations of these factors rather than filling a complete factorial grid. Table 1 gives the distribution over the 3,418 inputs of TASC-M. ISO timestamps are the most common format (65.8%), and ongoing, done, and presupposed requests are close to balanced. Nearboundary controls form the largest timing group, so that a model which warns whenever elapsed time is close to the threshold incurs false alarms. Distractors and implied rules cover small, targeted subsets; we compare them only with matched inputs from the same templates (§3.4).

<table><tr><td>Factor</td><td>Level</td><td># Inputs (%)</td><td># Templates</td></tr><tr><td rowspan="5">Timing</td><td>Before</td><td>1,081 (31.6)</td><td>116</td></tr><tr><td>Near</td><td>1,497 (43.8)</td><td>119</td></tr><tr><td>After</td><td>820 (24.0)</td><td>116</td></tr><tr><td>Prior conflict</td><td>20 (0.6)</td><td>10</td></tr><tr><td>Ongoing</td><td>1,213 (35.5)</td><td>110</td></tr><tr><td rowspan="3">Request</td><td>Done</td><td>1,206 (35.3)</td><td>113</td></tr><tr><td>Presupposed</td><td>999 (29.2)</td><td>109</td></tr><tr><td>ISO timestamp</td><td>2,248 (65.8)</td><td>113</td></tr><tr><td rowspan="4">Time format</td><td>Current time only</td><td>367 (10.7)</td><td>74</td></tr><tr><td>Verbal cue</td><td>228 (6.7)</td><td>76</td></tr><tr><td>None</td><td>575 (16.8)</td><td>113</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="3">Rule source</td><td>Assistant</td><td>1,889 (55.3)</td><td>128</td></tr><tr><td>User</td><td>1,499 (43.9)</td><td>113</td></tr><tr><td>None (implied)</td><td>30 (0.9)</td><td>3</td></tr><tr><td rowspan="3">Distractors</td><td>0 exchanges</td><td>3,274 (95.8)</td><td>128</td></tr><tr><td>2 exchanges</td><td>88 (2.6)</td><td>37</td></tr><tr><td>8 exchanges</td><td>56 (1.6)</td><td>25</td></tr><tr><td rowspan="4">Risk tier</td><td>0</td><td>640 (18.7)</td><td>22</td></tr><tr><td>1</td><td>974 (28.5)</td><td>36</td></tr><tr><td>2</td><td>1,148 (33.6)</td><td>40</td></tr><tr><td>3</td><td>656 (19.2)</td><td>30</td></tr></table>

Table 1: Distribution of the 3,418 TASC-M inputs over the construction factors. Templates counts the distinct templates with at least one input at that level; one template contributes to several levels.

## 2.2 Selecting Domains

Domains. We chose everyday activities in which a rule tied to an earlier event governs a later action and in which people commonly ask an assistant for help along the way. Typical rules are dosing intervals, fasting windows, cure and drying times, ventilation after chemical use, storage limits for food, and administrative or travel deadlines. The 128 TASC-M templates and the 33 TASC-P scenarios introduced below span 23 fine-grained domains, which we group into six categories (Table 2). Health and medicine is the largest category because it contains most tier-3 rules, while home, food, and work scenarios supply the lower-risk tiers with the same temporal structure. This spread lets us compare responses to the same kind of timing conflict across very different consequences.

The paired set: TASC-P. Most TASC-M templates are expanded under only some response conditions (§2.3), so the TASC-M cannot compare the same scenario across all of them. We therefore wrote 33 additional scenarios, disjoint from the 128 templates in the TASC-M, and ran every one under every condition. Each scenario also has a matched non-violation control in which only the timing changes, or the timing and the minimum text needed to keep the dialogue coherent. The TASC-P conversations are longer than in the TASC-M, with 5–21 messages and a median of 13, and cover 13 of the 23 domains with nine tier-1, thirteen tier-2, and eleven tier-3 scenarios. The four examples in Figure 1 come from this set. We report the TASC-M and the TASC-P separately: the TASC-M measures behavior across factor levels, and the TASC-P relates the conditions to one another within the same scenario and model.

<table><tr><td>Category</td><td>Domains</td><td>TASC-M</td><td>TASC-P</td></tr><tr><td>Health &amp; medicine</td><td>health, medication, first aid, emergency,</td><td>39</td><td>12</td></tr><tr><td>Home &amp; gar- den</td><td>fitness DIY, devices, crafts, pets &amp; garden, home</td><td>28</td><td>8</td></tr><tr><td>Work, travel &amp; admin</td><td>care workplace &amp; lab, travel, transport, fi-</td><td>24</td><td>3</td></tr><tr><td>Personal care</td><td>nance &amp; admin personal care, outdoor,</td><td>13</td><td>5</td></tr><tr><td>&amp; leisure Child care</td><td>leisure infant care, childcare</td><td>12</td><td>2</td></tr><tr><td></td><td>&amp; school</td><td></td><td></td></tr><tr><td>Food &amp; drink Total</td><td>food safety, cooking, baking, beverages 23</td><td>12 128</td><td>3 33</td></tr></table>

Table 2: The number of templates in each of the six categories in TASC, which further contains 23 domains.

## 2.3 Forming the Queries

Conditions. Every input poses one of three conditions on a shared conversation history. Condition A (duration query) replaces the final request with a direct question about elapsed or remaining time, which tests whether the model can compute the relevant duration. Condition B (ordinary request) presents the request unchanged, which tests whether the model intervenes without being asked. Condition C (timing supplied) keeps the request of Condition B and adds a line to the system prompt that states the current time and the elapsed time since the event, which tests whether the model intervenes once the timing is explicit. Condition B and Condition C share identical dialogue messages, and no generated answer is passed between conditions. In the TASC-P, each scenario is posed under Condition A, Condition B, and Condition C, and as a non-violation control, yielding 132 inputs.

Generation. Templates were drafted with model assistance from hand-written seeds, checked by a rule validator, and reviewed by the authors. A generator expands each template across the factor levels of §2.1, rewriting only the timestamps of the final message when the timing changes, so the wording is identical across before, near, and after. Verbal cues were authored separately, calibrated by a panel of models that had to agree on the implied elapsed time, and read manually before acceptance. The full expansion produced 18,373 distinct inputs from 314 of the 320 library templates (six tier-3 templates produce none). We retained 4,475 inputs covering 131 templates, chosen to discriminate between models and between diagnostic conditions, with each template’s condition pairs kept intact for the paired diagnostic analyses and item-by-item manual reads deciding verbal-cue membership. We then removed three kinds of inputs: 913 trivial non violation controls far from the boundary, keeping a stratified 20% sample; 138 duration queries in the verbal-cue format, whose final message no longer contained the cue; and six cue inputs whose implied time straddled the boundary. The result is 3,418 inputs from 128 templates: 234 under Condition A, 2,596 under Condition B, and 588 under Condition C (Table 1).

## 2.4 Evaluation

Metrics. Each Condition B or Condition C reply receives one of four labels. CATCH identifies the timing conflict and changes the advice accordingly; a reply that also answers the literal question still counts. HEDGE mentions time without changing the advice. ASK requests the missing timing information. PROCEED covers all other replies. Whether a label is correct depends on the gold type of the input. On a violation, only CATCH is correct, and the CATCH rate is the share of CATCH labels. On a non-violation control, CATCH is a false alarm, and the false-alarm rate (FAR) is the share of CATCH labels. On a possible violation in the none format, only ASK is correct, and we report the clarification rate. On emergency scenarios, where the template requires urgent action, CATCH or ASK counts as engagement. For Condition A, a duration answer is correct if the number parsed from the reply lies within max(2, 0.05∆) minutes of the true duration ∆. For matched Condition B–Condition C pairs, recovery is the share of Condition B misses caught under Condition C.

Judges. A duration parser scores Condition A, and LLM judges label Condition B and Condition C. Replies to non-violation controls that contain no timing or mismatch language are labeled

PROCEED by a rule screen. The judges receive the rubric, the reply, and precomputed facts about the input (the rule, the elapsed time, and the gold type), so they do not need to compute durations themselves. GPT-5.6 Luna (OpenAI, 2026a) provides the label of record, Claude Sonnet 5 (Anthropic, 2026b) provides an independent second label, and Claude Opus 5 (Anthropic, 2026a) adjudicates the 555 replies on which the judges disagree or a mechanical screen raises a flag, producing 182 corrections. On the 8,444 doubly judged replies, the two judges agree at Cohen’s κ = 0.858, and at κ = 0.905 on the TASC-P. A single annotator, blind to the recorded labels, agrees with them at κ = 0.53 on a TASC-M sample and κ = 0.708 on 144 TASC-P replies spanning all 33 scenarios and 13 models; agreement is lowest on non-violation controls (Appendix E).

## 2.5 Model Settings

We evaluate thirteen models from eight developers. From Anthropic, we include Claude Opus 5 (Anthropic, 2026a), Claude Sonnet 5 (Anthropic, 2026b), and Claude Haiku 4.5 (Anthropic, 2025). From OpenAI, we include GPT-5.6 Terra (OpenAI, 2026a), GPT-5.5 (OpenAI, 2026b), GPT-5 mini (Singh et al., 2025), and GPT-4o mini (Hurst et al., 2024). We further include Gemini 3.7 Flash (Doshi, 2026), Qwen3.8-Max (Qwen Team, 2026), DeepSeek-V4-Pro (DeepSeek-AI et al., 2026), Kimi K3 (Team et al., 2026), GLM-5.2 (Z.ai, 2026), and MiniMax M3 (MiniMax, 2026). The set covers proprietary and open-weight models from American and Chinese developers, and contains models of several sizes within the Claude and GPT families. All models are accessed through Open-Router.<sup>2</sup> We draw one sample per input with each provider’s default temperature, top-p, and reasoning effort. Each call contains the system prompt and the full authored history and generates only the final assistant reply.

## 3 Results

We focus on the following research questions (RQs). RQ1 (Catching Violations): Do models intervene, without being asked, when an ordinary request conflicts with an elapsed-time rule? RQ2 (Duration Accuracy): Do missed interventions stem from an inability to compute the elapsed time? RQ3 (Explicit Timing): Does stating the timing explicitly repair missed interventions? RQ4 (Difficulty Factors): Which properties of the conversation make intervention harder?

<table><tr><td rowspan=1 colspan=6>Experiment        Condition A   Condition B   Non-violation  Clarification      EngagementMetric              Accuracy ↑  CATCH rate ↑  CATCH rate ↓  ASK rate ↑  CATCH|ASK rate ↑Sample Size         n = 234      n = 877        $n = 1 , 3 4 7$      n = 275           n = 79</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 5</td><td rowspan=1 colspan=1>98.7</td><td rowspan=1 colspan=1>84.9</td><td rowspan=1 colspan=1>6.98</td><td rowspan=1 colspan=2>16.7               94.9</td></tr><tr><td rowspan=1 colspan=1>Kimi K3</td><td rowspan=1 colspan=1>97.9</td><td rowspan=1 colspan=1>47.8</td><td rowspan=1 colspan=1>2.90</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>48.1</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.7 Flash</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>47.1</td><td rowspan=1 colspan=1>1.11</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>49.4</td></tr><tr><td rowspan=1 colspan=1>Claude Sonnet 5</td><td rowspan=1 colspan=1>96.6</td><td rowspan=1 colspan=1>38.9</td><td rowspan=1 colspan=1>3.56</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>43.0</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1>98.3</td><td rowspan=1 colspan=1>36.6</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>31.6</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Max</td><td rowspan=1 colspan=1>99.1</td><td rowspan=1 colspan=1>31.7</td><td rowspan=1 colspan=1>1.48</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>39.2</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Terra</td><td rowspan=1 colspan=1>99.1</td><td rowspan=1 colspan=1>30.2</td><td rowspan=1 colspan=1>2.08</td><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>38.0</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=1>97.4</td><td rowspan=1 colspan=1>29.6</td><td rowspan=1 colspan=1>1.71</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>35.4</td></tr><tr><td rowspan=1 colspan=1>GLM-5.2</td><td rowspan=1 colspan=1>99.6</td><td rowspan=1 colspan=1>27.8</td><td rowspan=1 colspan=1>2.08</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>34.2</td></tr><tr><td rowspan=1 colspan=1>GPT-5 mini</td><td rowspan=1 colspan=1>95.7</td><td rowspan=1 colspan=1>8.3</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=1>6.3</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>73.1</td><td rowspan=1 colspan=1>8.2</td><td rowspan=1 colspan=1>1.41</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>7.6</td></tr><tr><td rowspan=1 colspan=1>MiniMax M3</td><td rowspan=1 colspan=1>93.6</td><td rowspan=1 colspan=1>4.8</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>6.3</td></tr><tr><td rowspan=1 colspan=1>GPT-4o mini</td><td rowspan=1 colspan=1>52.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=1>92.4</td><td rowspan=1 colspan=1>30.5</td><td rowspan=1 colspan=1>1.95</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>33.4</td></tr></table>

Table 3: TASC-M scores (%) on the 3,397 inputs answered by all models. Note that in non-violation controls CATCH rate represents false alarms, so the metric is the lower the better. Rows are sorted by Condition B CATCH rate. Cells are darkened proportionally to their value. Average is the unweighted mean over the 13 models.

## 3.1 RQ1: Catching Violations

Most violations go uncaught, and models differ widely. Across the 13 models, 3,474 of 11,401 Condition B replies to violations are labeled CATCH (30.5%; Table 3). The remaining replies answer the literal request, typically with competent advice about the surface topic, and leave the conflicting action in place. CATCH rates span almost two orders of magnitude, from 0.1% for GPT-4o mini to 84.9% for Claude Opus 5, the only model above 50%; Kimi K3 and Gemini 3.7 Flash follow at 47.8% and 47.1%. Within families, smaller models catch far fewer violations: Claude Opus 5, Sonnet 5, and Haiku 4.5 reach 84.9%, 38.9%, and 8.2%, and GPT-5.5, GPT-5 mini, and GPT-4o mini reach 36.6%, 8.3%, and 0.1%. The low CATCH rates do not reflect general caution being suppressed: the same replies raise false alarms ( CATCH on a non-violation control) on only 342 of 17,511 nonviolation controls (1.95%), most of which sit near the boundary. Claude Opus 5 also has the highest false-alarm rate (6.98%), so part of its lead comes from warning more readily.

Models rarely ask and often miss emergencies. When timestamps are absent and a violation is possible, models respond with ASK in 132 of 3,575 replies (3.7%); only Claude Opus 5 asks in more than a tenth of cases. On emergency scenarios, which call for urgent action, models engage, with CATCH or ASK , in 343 of 1,027 replies (33.4%), about the same rate as on ordinary violations.

## 3.2 RQ2: Duration Accuracy

Models can compute the elapsed time but fail to apply it. Asked directly, models answer 2,811 of 3,042 duration queries correctly (92.4%), and ten of the 13 models exceed 95% (Table 3); only GPT-4o mini (52.1%) and Claude Haiku 4.5 (73.1%) fall well below. Yet the gap to the CATCH rate is large for every model: GLM-5.2 answers 99.6% of duration queries and catches 27.8% of violations, and GPT-5 mini answers 95.7% and catches 8.3%. Because this aggregate comparison mixes different items, we also condition on the same scenario and model. In the TASC-P, Condition A is correct in 364 of 428 tuples (85.0%), and 226 of these tuples miss the violation under Condition B, a conditional miss rate of 62.1% (95% CI: 53.7–69.8). The TASC-M shows the same pattern on the 78 violation inputs that have Condition A, Condition B, and Condition C siblings: 660 of 932 correct duration answers coincide with a Condition B miss (70.8%). The model holds the information it needs but does not connect it to the request.

<table><tr><td>Condition A outcome</td><td>Condition B misses</td><td>Condition C recovered</td><td>Still missed</td></tr><tr><td>Correct</td><td>226</td><td>119 (52.7%)</td><td>107</td></tr><tr><td>Incorrect</td><td>60</td><td>12 (20.0%)</td><td>48</td></tr></table>

Table 4: TASC-P recovery under Condition C, split by the outcome of Condition A. Percentages use the Condi tion B misses in each row.

## 3.3 RQ3: Explicit Timing

Supplying the timing helps only partially. On 250 matched TASC-M violations from 70 templates, adding the timing to the system prompt raises the CATCH rate from 33.6% to 41.6%. The paired gain is 8.0 percentage points (95% CI: 6.3–9.6). Every model improves, but Condition C recovers only 433 of 2,158 Condition B misses (20.1%), and 1,725 model–input pairs miss under both conditions. In the TASC-P, Condition C catches 268 of 428 violations (62.6%) and recovers 131 of 286 Condition B misses (45.8%; 37.4–54.1). The gain does not come from warning more indiscriminately: on 300 matched non-violation controls from 68 templates, the false-alarm rate stays nearly flat (2.03% under Condition B, 2.23% under Condition C; 95% CI of the difference: −0.50 to 0.97).

Most persistent misses follow a correct computation. In the TASC-P, recovery is 52.7% when Condition A is correct and 20.0% when Condition A is incorrect (Table 4). Yet 107 tuples answer Condition A correctly and still miss under both Condition B and Condition C, and they account for 69.0% of the misses that persist under Condition C. For these cases, neither computing the duration nor being told it leads the model to change its advice.

## 3.4 RQ4: Difficulty Factors

Factor levels are not crossed within every template, so the comparisons below match inputs that differ only in the factor of interest and pool Condition B violations over all 13 models.

Three factors defy intuition. First, knowing the current time is not enough. When the system prompt gives the current time and the conversation gives the time of the event, the model has everything it needs to compute the elapsed time; yet removing the per-message timestamps in this setting halves the CATCH rate, from 36.1% to 18.6% (178 matched pairs). A verbal cue in place of a timestamp, by contrast, lowers it only slightly, from 46.9% to 41.4% (39 pairs). Models thus use time evidence mainly when it is attached to the message they are answering. Second, models apply rules they stated themselves more often than rules stated by the user. Instruction following might lead one to expect the opposite, yet models catch 35.8% of violations when the assistant stated the rule and only 30.0% when the user did (305 matched pairs). Rules left implied by common knowledge yield 39.3% against 47.9% for their stated counterparts, though this comparison rests on only three templates. Third, high stakes do not make models reliable. Tier-3 rules govern medication, medical procedures, and emergencies, yet models catch only 35.5% of tier-3 violations, leaving nearly two thirds unflagged. Tier 3 fares no better than tier 2 (35.6%), and its margin over harmless tier-0 outcomes (21.9%) is small relative to the gap in consequences. Because risk tier varies together with domain and rule, these rates describe the benchmark rather than isolate the effect of risk.

Two factors behave as expected. First, indirect requests are harder. When the user says the action is happening now, models catch 44.0% of violations, against 28.2% when the user reports it as done (184 matched pairs); a presupposed request lowers the rate from 42.3% to 29.0% relative to an ongoing one (216 pairs). The further the action sits from the surface of the message, the less often models connect it to the rule. Second, distance between rule and request also lowers intervention. On 22 templates observed at zero, two, and eight distractor messages with the same elapsed time, CATCH rates are 19.2%, 21.0%, and 12.9%: eight messages reduce it by a third.

## 4 Discussion

## 4.1 Implications for Chatbot Design

Our results are measured under favorable conditions. Every TASC-M scenarios except the none format gives the model explicit time evidence: timestamps on every message, the current time in the system prompt, or a verbal cue that fixes the elapsed time. Every TASC-P conversation carries timestamps on all messages, and Condition C states the elapsed time outright. Even so, models catch fewer than a third of violations, and supplying the timing recovers less than half of the misses (§3.1, §3.3).

Deployed assistants usually see less than this. To our knowledge, widely used chat products do not attach a timestamp to each user message; at most, the system prompt contains the current date or time. This setting corresponds to our current time only and none formats, where the catch rate falls to 18.6% and models ask about the timing in only 3.7% of the cases that require it (§3.4). Conversations in deployment can also span hours or days across a single thread, which widens the gap between the event and the request. The risk measured by TASC is therefore likely an underestimate of the risk in current products.

Two changes are therefore suggested. First, products should pass the sending time of every user or assistant message to the model, since per-message timestamps double the catch rate relative to the current time alone. Second, timestamps are not sufficient: models compute the elapsed time correctly and still fail to ask whether it bears on the request (§3.2). Closing this gap requires training and evaluation that reward checking pending time-dependent rules before giving advice, and that judge a reply by the action it supports rather than by whether it mentions time.

## 4.2 Related Work

Temporal reasoning and time awareness. Temporal reasoning benchmarks test event ordering, duration arithmetic, and the resolution of temporal expressions in standalone questions (Tan et al., 2023; Chu et al., 2024; Wang and Zhao, 2024; Fatemi et al., 2025), date puzzles and calendar conversions (Wang and Dong, 2026; Miao et al., 2026), and dialogue or real-world text (Qin et al., 2021; Wei et al., 2025; Kenneweg et al., 2026); others check that factual knowledge is anchored to the right time (Herel et al., 2024; Zhu et al., 2025). Recent work asks whether models perceive time itself: how long tasks take for models and for people (Liu et al., 2026; Garikaparthi, 2026), how much time separates messages (Wang et al., 2025; Cheng et al., 2026), and how much remains before a deadline (Sehgal et al., 2026). Dialogue and memory systems condition on the gap between sessions (Zhang et al., 2023; Jang et al., 2025) and reason over long timestamped histories (Maharana et al., 2024; Wu et al., 2025; Ge et al., 2025; Su et al., 2026), where evidence in mid-context is easily missed (Liu et al., 2024). All of these ask for a temporal answer directly, as our Condition A does; TASC asks whether time already in the conversation changes the advice on a latest request that does not mention it.

Prospective memory and proactive intervention. Closest to our work are tests of whether models act on earlier information unprompted. TriggerBench pairs each latent constraint with a retrospective probe and finds that models often recall a constraint yet fail to act on it (Zhang et al., 2026). PM-Bench simulates a week in which agents must execute delayed intentions at the right moment while busy with other tasks (Liu and Gabriel, 2026). LoCoMo-Plus tests whether implicit user states constrain later requests (Li et al., 2026), and ChronoScope finds that models drift to present-day assumptions when follow-up questions omit the time (Atri et al., 2026). Proactive assistants are evaluated on anticipating needs, detecting unstated problems, and asking for missing preferences (Lu et al., 2025; Pasternak et al., 2025; Harfi et al., 2026; Wu et al., 2026; Yuan et al., 2025). TASC shares this structure but makes elapsed time the trigger. The temporal dimension of TriggerBench asks the model to remind the user of an upcoming deadline; TASC instead asks whether the action the user requests breaks a time-dependent rule by coming too early, too late, or outside a target window.

Contextual safety. Context can decide whether a response is safe (Sun et al., 2025; Shan et al., 2025), and models often fail to apply safety facts they know to everyday questions that do not raise them (Chen et al., 2025). Medical benchmarks find that models violate medical safety principles (Han et al., 2024), seek too little context, and miss contraindications that are implied rather than stated (Arora et al., 2025; Zhao et al., 2025). Only DOSEBENCH involves timing, and there the user asks about it directly (Kousar and Hu, 2026). TASC treats elapsed time as a source of contextual risk hidden behind an unrelated request.

## 4.3 Conclusion

We introduced TASC for evaluating temporal safety in conversation. Models often answer duration questions correctly yet fail to apply the relevant rule when giving advice, and explicitly supplying the timing resolves only part of this gap. These findings motivate preserving temporal context, checking relevant constraints before recommending an action, and evaluating whether advice changes appropriately when the timing changes. Moreover, it demonstrates that temporal safety requires using timing information when it changes what the user should do, even without users’ explicit prompting.

## Limitations

Sampled rather than fully crossed factors. TASC-M samples combinations of the six construction factors instead of filling the complete factorial grid, and most templates are expanded under only some of the three conditions (§2.3). Several factor levels are therefore thinly covered: implied rules come from three templates, distractors from at most 37, and verbal cues from 228. As a result, the factor comparisons in §3.4 rest on matched subsets of different sizes and templates instead of on a single balanced design, and factors that vary together, such as risk tier and domain, cannot be fully separated. For the same reason, relating Condition A, Condition B, and Condition C within one scenario required the separate TASC-P, whose 33 scenarios cover 13 of the 23 domains. A fully crossed expansion of every template would allow all factors and conditions to be estimated on one population, at a substantially higher evaluation cost.

English only. All conversations, rules, and verbal cues are written in English. Users ask assistants about the same activities in many languages, and the cues that carry elapsed time, such as relative expressions (“slept the night through”), date and clock conventions, and the way a completed or ongoing action is marked, differ across languages. Whether models catch violations at similar rates, and whether the gap between computing a duration and applying it persists, when the conversation is held in another language remains untested.

Authored, single-reply evaluation. The conversations are authored rather than collected from real users, and each input is scored on a single final reply, sampled once with each provider’s default settings. Real conversations are noisier, span multiple sessions, and rarely carry per-message timestamps; deployed assistants may also add memory, retrieval, or system prompts that we do not reproduce. Our setting supplies time evidence more explicitly than most deployments do, so the reported CATCH rates are likely an upper bound on what users currently experience. It does not, however, capture whether a model that misses a violation in one reply would recover in a later turn, or how sampling variation affects individual outcomes.

## Ethics Statement

Data and code availability. The complete benchmark is available at https://github.com/ peidid/TASC\_release: the data in both sets, the 320 scenario templates and the full expansion pool, all prompts and all recorded replies of the 13 evaluated models, all labels with per-label provenance (judge, rule screen, or logged correction), the human-audit aggregates, and the collection, judging, and scoring code with a script for every reported number. A read-only interactive explorer of all inputs, replies, and labels is included.

License and intended use. The data are released under CC-BY-4.0 and the code under the MIT License. The timing rules specified inside each conversation are premises for measurement, not medical or safety guidance, and the benchmark is intended for evaluating assistants, not for training them to give medical advice.

Personal information and content. All conversations are authored fiction: they contain no user data, no real persons, and no personally identifying information, and product references follow public label text. Scenarios describe everyday activities with time-sensitive risks (medication schedules, food handling, household chemicals) in plain language; the dataset contains no offensive content.

Use of AI assistants. LLM-based agents were used for code, data verification, and editing under author review; LLM judges are part of the measured pipeline and are described in §2.4. The authors verified all claims and all reported numbers against the released artifacts.

## References

American Society of Anesthesiologists. 2017. Practice guidelines for preoperative fasting and the use of pharmacologic agents to reduce the risk of pulmonary aspiration: Application to healthy patients undergoing elective procedures: An updated report by the american society of anesthesiologists task force on preoperative fasting and the use of pharmacologic agents to reduce the risk of pulmonary aspiration. Anesthesiology, 126(3):376–393.

Anthropic. 2025. Introducing Claude Haiku 4.5. Anthropic Blog, Oct 15, 2025. https://www. anthropic.com/news/claude-haiku-4-5.

Anthropic. 2026a. Introducing Claude Opus 5. Anthropic Blog, Jul 24, 2026. https://www. anthropic.com/news/claude-opus-5.

Anthropic. 2026b. Introducing Claude Sonnet 5. Anthropic Blog, Jun 30, 2026. https://www. anthropic.com/news/claude-sonnet-5.

Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. 2025. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775.

Yash Kumar Atri, Steven L Johnson, and Thomas Hartvigsen. 2026. Evaluating temporal consistency in multi-turn language models. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 45964–45982.

Aaron Chatterji, Thomas Cunningham, David J Deming, Zoe Hitzig, Christopher Ong, Carl Yan Shan, and Kevin Wadman. 2025. How people use chatgpt. Technical report, National Bureau of Economic Research.

Yueh-Han Chen, Guy Davidson, and Brenden Lake. 2025. Sage-eval: Evaluating llms for systematic generalizations of safety facts. Advances in Neural Information Processing Systems, 38.

Yize Cheng, Arshia Soltani Moakhar, Chenrui Fan, Parsa Hosseini, Kazem Faghih, Zahra Sodagar, Wenxiao Wang, and Soheil Feizi. 2026. Your llm agents are temporally blind: The misalignment between tool use decisions and human time perception. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 37082–37104.

Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Haotian Wang, Ming Liu, and Bing Qin. 2024. Timebench: A comprehensive evaluation of temporal reasoning abilities in large language models. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1204–1228.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, et al. 2026. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

Tulsee Doshi. 2026. Gemini 3.7 Flash: our most intelligent workhorse model. Google Blog, Aug 13, 2026. https://blog.google/innovation-and-ai/ models-and-research/gemini-models/ introducing-gemini-3-7-flash/.

Bahare Fatemi, Seyed Mehran Kazemi, Anton Tsitsulin, Karishma Malkan, Jinyeong Yim, John Palowitch, Sungyong Seo, Jonathan Halcrow, and Bryan Perozzi. 2025. Test of time: A benchmark for evaluating llms on temporal reasoning. In International Conference on Learning Representations, volume 2025, pages 94426–94447.

Aniketh Garikaparthi. 2026. Can llms perceive time? an empirical investigation. arXiv preprint arXiv:2604.00010.

Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu, Yassine Benajiba, Monica Sunkara, and Yi Zhang. 2025. Tremu: Towards neuro-symbolic temporal reasoning for llm-agents with memory in multi-session dialogues. In Findings of the Association for Computational Linguistics: ACL 2025, pages 18974–18988.

Tessa Han, Aounon Kumar, Chirag Agarwal, and Himabindu Lakkaraju. 2024. Medsafetybench: Evaluating and improving the medical safety of large language models. Advances in neural information processing systems, 37:33423–33454.

Sepehr Harfi, Ahmad Salimi, Dongming Shen, and Alex Smola. 2026. Proactbench: Beyond what the user asked for. arXiv preprint arXiv:2605.09228.

David Herel, Vojtech Bartek, Jiri Jirak, and Tomas Mikolov. 2024. Time awareness in large language models: benchmarking fact recall across time. arXiv preprint arXiv:2409.13338.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. GPT-4o system card. arXiv preprint arXiv:2410.21276.

Seongbo Jang, Minjin Jeon, Jaehoon Lee, Seonghyeon Lee, Dongha Lee, and Hwanjo Yu. 2025. From what to respond to when to respond: Timely response generation for open-domain dialogue agents. arXiv preprint arXiv:2506.14285.

Svenja Kenneweg, Jörg Deigmöller, Philipp Cimiano, and Julian Eggert. 2026. Traveler: A benchmark for evaluating temporal reasoning across vague, implicit and explicit references. SN Computer Science, 7(5):379.

Maroof Kousar and Yibo Hu. 2026. Can i take another dose? evaluating llm decision-making under temporal uncertainty in otc dosing qa. arXiv preprint arXiv:2606.04262.

Yifei Li, Weidong Guo, Lingling Zhang, Rongman Xu, Muye Huang, Hui Liu, Lijiao Xu, Yu Xu, and Jun Liu. 2026. Locomo-plus: Beyond-factual cognitive memory evaluation framework for llm agents. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 25085–25100.

Genglin Liu and Saadia Gabriel. 2026. Pm-bench: Evaluating prospective memory of llm agents. In Third Conference on Language Modeling.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe association for computational linguistics, 12:157–173.

Ryan Liu, Icey Siyi Ai, Marwa Abdulhai, Rachit Dubey, David Lee, and Thomas L. Griffiths. 2026. In search of lost time: Temporal competence in large language

models. The First Workshop on Interpreting Agent Behavior, NeurIPS (Sydney) 2026.

Yaxi Lu, Shenzhi Yang, Cheng Qian, Guirong Chen, Qinyu Luo, Yesai Wu, Huadong Wang, Xin Cong, Zhong Zhang, Yankai Lin, et al. 2025. Proactive agent: Shifting llm agents from reactive responses to active assistance. In International Conference on Learning Representations, volume 2025, pages 47431–47457.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870.

Zhongjian Miao, Hao Fu, and Chen Wei. 2026. Span: Benchmarking and improving cross-calendar temporal reasoning of large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 35715–35723.

MiniMax. 2026. MiniMax M3: Frontier coding, 1M context, native multimodality — all in one model. MiniMax Blog, Jun 1, 2026. https://www.minimax. io/blog/minimax-m3.

OpenAI. 2026a. GPT-5.6: Frontier intelligence that scales with your ambition. OpenAI Blog, Jul 9, 2026. https://openai.com/index/gpt-5-6/.

OpenAI. 2026b. Introducing GPT-5.5. OpenAI Blog, Apr 23, 2026. https://openai.com/index/ introducing-gpt-5-5/.

Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, and Ash Lewis. 2025. Beyond reactivity: Measuring proactive problem solving in llm agents. arXiv preprint arXiv:2510.19771.

Lianhui Qin, Aditya Gupta, Shyam Upadhyay, Luheng He, Yejin Choi, and Manaal Faruqui. 2021. Timedial: Temporal commonsense reasoning in dialog. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7066–7076.

Qwen Team. 2026. Qwen3.8-Max: A new bar for coding and cowork. Qwen Blog, Aug 3, 2026. https://qwen.ai/blog?id=qwen3.8.

Neil KR Sehgal, Sharath Chandra Guntuku, and Lyle Ungar. 2026. Real-time deadlines reveal fragile temporal adaptation in llm strategic dialogues. arXiv preprint arXiv:2601.13206.

Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Yan Teng, Jingqi Huang, Qingshan

Liu, Guangze Ye, Guoqing Wang, et al. 2025. Mentor: A metacognition-driven self-evolution framework for uncovering and mitigating implicit domain risks in llms. arXiv preprint arXiv:2511.07107.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. 2025. OpenAI GPT-5 system card. arXiv preprint arXiv:2601.03267.

Miao Su, Yucan Guo, Zhongni Hou, Long Bai, Zixuan Li, Yufei Zhang, Guojun Yin, Wei Lin, Xiaolong Jin, Jiafeng Guo, et al. 2026. Beyond dialogue time: Temporal semantic memory for personalized llm agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 29935–29951.

Guangzhi Sun, Xiao Zhan, Shutong Feng, Phil Woodland, and Jose Such. 2025. CASE-bench: Contextaware SafEty benchmark for large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 57938– 57960. PMLR.

Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2023. Towards benchmarking and improving the temporal reasoning capability of large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14820–14835.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, et al. 2026. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653.

Minghan Wang, Ye Bai, Thuy-Trang Vu, Ehsan Shareghi, and Gholamreza Haffari. 2025. Discrete minds in a continuous world: Do language models know time passes? In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 18703–18729.

Yuqing Wang and Yun Zhao. 2024. Tram: Benchmarking temporal reasoning for large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 6389–6415.

Zhengxiang Wang and Zeyu Dong. 2026. Measuring iterative temporal reasoning with time puzzles. arXiv preprint arXiv:2601.07148.

Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, and Houfeng Wang. 2025. Time: A multi-level benchmark for temporal reasoning of llms in real-world scenarios. Advances in Neural Information Processing Systems, 38.

Bin Wu, Guanyun Zou, Bingbing Wang, Huan Zhao, and Chuan Shi. 2026. Ask now, use later: Benchmarking the proactivity gap in long-lived llm agents. arXiv preprint arXiv:2605.28108.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations.

Youliang Yuan, Wenxiang Jiao, Yuejin Xie, Chihao Shen, Menghan Tian, Wenxuan Wang, Jen-tse Huang, and Pinjia He. 2025. Towards evaluating proactive risk awareness of multimodal language models. Advances in Neural Information Processing Systems, 38.

Z.ai. 2026. GLM-5.2: Built for long-horizon tasks. Z.ai Blog, Jun 16, 2026. https://z.ai/blog/glm-5.2.

Qiang Zhang, Jason Naradowsky, and Yusuke Miyao. 2023. Mind the gap between conversations for improved long-term dialogue generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10735–10762.

Tianhua Zhang, Xinjiang Wang, Qianxi Zhang, Qi Chen, Kun Li, Yaoqi Chen, Dingdong Wang, Helen Meng, and Yan Lu. 2026. Triggerbench: Investigating prospective memory for large language models. arXiv preprint arXiv:2606.23459.

Jiahao Zhao, Luxin Xu, Minghuan Tan, Lichao Zhang, Ahmadreza Argha, Hamid Alinejad-Rokny, and Min Yang. 2025. Rxsafebench: Identifying medication safety issues of large language models in simulated consultation. In 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 4491–4496. IEEE.

Zhiyuan Zhu, Yusheng Liao, Zhe Chen, Yuhao Wang, Yunfeng Guan, Yanfeng Wang, and Yu Wang. 2025. Evolvebench: A comprehensive benchmark for assessing temporal awareness in llms on evolving knowledge. In Proceedings ofthe 63rd Annual Meet ing of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16173–16188.

## A More Examples from TASC

Figure 2 shows examples from four more domains.

## B TASC-P Construction

Structural dimensions. Each TASC-P scenario targets one temporal structure that requires the model to select or update the relevant clock. Long conversations (13 scenarios) state the rule early in a history of 11–21 messages. Implicit anchors (6) leave the time of the event or of the request to be inferred from context. Interleaved constraints (5) run several rules at once, and the request violates only one of them. Incorrect duration claims (3) have the user assert that the wait has elapsed when it has not. Timer resets (3) contain a later event that restarts the wait. Schedule drift (2) lets the timing slip over a long conversation until the interval is exceeded. A timezone change (1) separates clock time from elapsed time. Table 5 reports the results by dimension. Most dimensions contain only a few scenarios, so these rates describe the set rather than estimate dimension effects.

Non-violation controls. Each scenario’s nonviolation control moves the request to the permitted side of the rule. In 11 of the 33 scenarios, the control changes only the timestamps; in the other 22, it also edits the dialogue text to keep the conversation coherent.

## C TASC-P Results by Model

Table 6 and Figure 3 report the 33-scenario TASC-P by model. Figure 3 labels models by the identifiers used during collection. The row counts sum to 428 complete tuples. The missing tuple is DeepSeek-V4-Pro on x3\_hypoglycemia\_recheck, whose outputs under all four conditions are empty.

Contrastive success. A tuple is a contrastive success if the model catches the violation under Condition B and raises no false alarm on the matched non-violation control. The non-violation controls draw four false alarms across the 428 tuples (0.9%), so 139 tuples (32.5%) are contrastive successes. On the 11 scenarios whose controls change only the timestamps, contrastive success is 38/142 (26.8%). CATCH labels under Condition B therefore do not come from warning regardless of the timing.

Additional analyses. Within the 24 tier-2 and tier-3 scenarios, 153 of 265 correct Condition A answers coincide with a Condition B miss (57.7%), close to the 62.1% of the full set. Supplied timing changes Condition B outcomes almost only in one direction. Of the 286 Condition B misses, 131 become CATCH under Condition C (45.8%) and 155 remain misses; of the 142 Condition B CATCH labels, 137 remain CATCH and only five become misses.

## D TASC-M Coverage and Additional Results

The TASC-M has 3,418 unique inputs from 128 templates. The TASC-P adds 132 inputs from 33 disjoint scenarios. By time format, the TASC-M contains 2,248 ISO-timestamp inputs, 367 currenttime-only inputs, 228 verbal-cue inputs (110 of them with presupposed requests), and 575 inputs with no time information. By risk tier, it contains 22 templates at tier 0, 36 at tier 1, 40 at tier 2, and 30 at tier 3.

![](images/dc2062864cae62c5b17158289a8e2500d4d7a7d1f6f095fa87a80ce854add2fe.jpg)  
Figure 2: Four more examples of different domains from TASC.

Coverage and refusal denominators. All 44,434 assigned TASC-M model–input combinations have records. Claude Opus 5 has 3,397 usable records and 21 marked refused; Claude Sonnet 5 has 3,414 usable records and four marked refused. The other eleven models each have 3,418 usable records. The four Sonnet exclusions are among the 21 Opus exclusions. Their intersection therefore contains 3,397 shared inputs and 44,161 responses, 99.4% of the assigned combinations. No usable TASC-M response has empty text.

The 25 refusal records comprise five Condition B violations, three Condition C violations, eleven Condition B non-violation controls, and six Condition B inputs that call for clarification. They remain in the assigned-input sensitivity as non- CATCH labels, with status reported separately. This sensitivity gives 3,485/11,466 Condition B CATCH labels (30.39%), 344/17,641 Condition B false alarms (1.95%), 134/3,614 clarification responses (3.71%), and 343/1,027 engagements on emergency scenarios (33.4%). It describes the collected outputs on all assigned inputs, including refusals. The primary table instead evaluates every model on the shared scored intersection.

Matched Condition B/Condition C pairs. All 588 TASC-M Condition C inputs have a Condition B sibling with identical full dialogue messages. Matching checks template, time format, request type, rule source, timing, gold type, risk tier, duration, and distractors. Every Condition C system prompt differs from its Condition B counterpart. The shared scored inputs contain 250 violation pairs from 70 templates and 300 non-violation control pairs from 68 templates. Across the 13 models, these give 3,250 and 3,900 response pairs. Appendix records preserve every pair ID. On the nonviolation control pairs, Condition C introduces 55 false alarms and removes 47; the small net change from 79 to 87 conceals larger turnover across individual responses.

<table><tr><td>Dimension</td><td># Scen.</td><td>n</td><td>Condition A</td><td>Condition B</td><td>Condition C</td><td>Miss|Condition A</td><td>Rec.</td></tr><tr><td>Long history</td><td>13</td><td>168</td><td>88.7</td><td>25.0</td><td>54.2</td><td>71.8</td><td>41.3</td></tr><tr><td>Implicit anchor</td><td>6</td><td>78</td><td>83.3</td><td>51.3</td><td>73.1</td><td>41.5</td><td>47.4</td></tr><tr><td>Interleaved</td><td>5</td><td>65</td><td>80.0</td><td>33.8</td><td>67.7</td><td>61.5</td><td>51.2</td></tr><tr><td>Incorrect claim</td><td>3</td><td>39</td><td>79.5</td><td>35.9</td><td>66.7</td><td>54.8</td><td>48.0</td></tr><tr><td>Timer reset</td><td>3</td><td>39</td><td>92.3</td><td>48.7</td><td>71.8</td><td>47.2</td><td>50.0</td></tr><tr><td>Schedule drift</td><td>2</td><td>26</td><td>84.6</td><td>11.5</td><td>53.8</td><td>86.4</td><td>47.8</td></tr><tr><td>Time zone</td><td>1</td><td>13</td><td>69.2</td><td>15.4</td><td>61.5</td><td>77.8</td><td>54.5</td></tr><tr><td>All</td><td>33</td><td>428</td><td>85.0</td><td>33.2</td><td>62.6</td><td>62.1</td><td>45.8</td></tr></table>

Table 5: TASC-P scores (%) by structural dimension. # Scen. counts templates and n counts complete model– scenario tuples. Miss|Condition A is the share of correct Condition A answers with a Condition B miss. Rec. (recovery) is the share of Condition B misses caught under Condition C.

![](images/0eb4ccc42919dfa0a8efe7a8ef7c45ad5bece5413284c956d1e3b4c26794f079.jpg)  
Condition A (Duration Accuracy) Condition B (Ordinary Request) Condition C (Timing Supplied)  
Figure 3: TASC-P scores (%) for the 13 models. Condition A measures accuracy; Condition B and Condition C calculate CATCH rates. Each model has 33 complete tuples except DeepSeek-V4-Pro, which has 32.

Calculation-matched TASC-M. A stricter intersection retains 78 violation inputs with Condition A, Condition B, and Condition C siblings from 35 templates. Across models this gives 1,014 triples, with 932 correct Condition A answers. Among those, 660 correspond to Condition B misses (70.8%); Condition C recovers 119 of these misses (18.0%). Of the 77 Condition B misses with an incorrect Condition A answer, Condition C recovers nine. This subset analysis preserves item matching within the TASC-M. It uses a different scenario population from the 33-scenario TASC-P.

Selection and rule review. Our construction log records the difficulty-based retention and the subsequent exclusions (§2.3). The input manifest contains the retained collection with its rule-derived gold labels. The completed scenario census flagged seven TASC-M scenarios. An earlier inspection also flags t3\_concussed\_teenager for review of its early-violation premise. Dataset size counts both base templates and generated inputs; neither count substitutes for this rule review.

<table><tr><td rowspan=1 colspan=6>Experiment         Condition A     Condition B    Condition C  Non-violationMetric               Accuracy ↑ CATCH rate ↑ CATCH rate ↑ CATCH rate ↓ Recovery ↑</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 5</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Kimi K3</td><td rowspan=1 colspan=1>75.8</td><td rowspan=1 colspan=1>18.2</td><td rowspan=1 colspan=1>45.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>37.0</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.7 Flash</td><td rowspan=1 colspan=1>97.0</td><td rowspan=1 colspan=1>63.6</td><td rowspan=1 colspan=1>87.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>66.7</td></tr><tr><td rowspan=1 colspan=1>Claude Sonnet 5</td><td rowspan=1 colspan=1>93.9</td><td rowspan=1 colspan=1>36.4</td><td rowspan=1 colspan=1>84.8</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>81.0</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>48.5</td><td rowspan=1 colspan=1>90.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>82.4</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Max</td><td rowspan=1 colspan=1>97.0</td><td rowspan=1 colspan=1>42.4</td><td rowspan=1 colspan=1>78.8</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>63.2</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6 Terra</td><td rowspan=1 colspan=1>97.0</td><td rowspan=1 colspan=1>54.5</td><td rowspan=1 colspan=1>75.8</td><td rowspan=1 colspan=1>6.1</td><td rowspan=1 colspan=1>60.0</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=1>90.6</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>53.1</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>46.4</td></tr><tr><td rowspan=1 colspan=1>GLM-5.2</td><td rowspan=1 colspan=1>90.9</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>78.8</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>70.8</td></tr><tr><td rowspan=1 colspan=1>GPT-5 mini</td><td rowspan=1 colspan=1>97.0</td><td rowspan=1 colspan=1>15.2</td><td rowspan=1 colspan=1>81.8</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>78.6</td></tr><tr><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>63.6</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>21.9</td></tr><tr><td rowspan=1 colspan=1>MiniMax M3</td><td rowspan=1 colspan=1>63.6</td><td rowspan=1 colspan=1>9.1</td><td rowspan=1 colspan=1>12.1</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>6.7</td></tr><tr><td rowspan=1 colspan=1>GPT-4o mini</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=1>85.0</td><td rowspan=1 colspan=1>33.2</td><td rowspan=1 colspan=1>62.6</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>45.8</td></tr></table>

Table 6: TASC-P scores (%) by model. Condition A measures accuracy; Condition B and Condition C calculate CATCH rates. Note that in non-violation controls CATCH rate represents false alarms, so the metric is the lower the better. Recovery is the share of Condition B misses caught under Condition C (– when a model has no Condition B misses). Rows follow the order of Table 3. Cells are darkened proportionally to their value. Each model has 33 complete tuples except DeepSeek-V4-Pro, which has 32.

<table><tr><td>Rule / direction</td><td>CATCH /n</td><td>Rate (%)</td></tr><tr><td>Minimum wait / early</td><td>1,991 / 5,551</td><td>35.9</td></tr><tr><td>Deadline / late</td><td>944 / 3,003</td><td>31.4</td></tr><tr><td>Target window / early</td><td>325 / 1,677</td><td>19.4</td></tr><tr><td>Target window / late</td><td>189 / 1,040</td><td>18.2</td></tr></table>

Table 7: TASC-M Condition B CATCH rates on violations by rule type and direction, using the shared scored inputs. n counts model–input responses.

## E Scoring, Agreement, and Reanalysis

Final labels are reconstructed by joining the scenario, condition, response, and judgment tables in review.db. Label files are applied in this order: labels\_plan\_sonnet\_v2.jsonl, labels\_luna\_ v2.jsonl, and corrections\_v2.jsonl. The duration answers use stage\_a\_scores.jsonl. Only the 13 evaluated models enter the analysis. A complete TASC-P tuple requires usable Condition A,

Condition B, Condition C, and non-violation control outputs.

Label counts. The TASC-M has 41,367 labels on Condition B and Condition C responses, of which the rule screen assigns 13,223, and 233 are corrections. The 233 comprise the 173 main-set corrections produced by the adjudication pass of §2.4, which also produced the nine TASC-P corrections (182 in total), and 60 retained from two earlier single-judge precision audits (47 and 13); the correction files hold 368 rows in all, the remainder falling on models or inputs outside the evaluated sets. The TASC-P has 1,284 labels on Condition B, Condition C, and non-violation control responses, of which the rule screen assigns 298, and nine are corrections. The reconstructed numerators and denominators match the recorded scores for all 13 models.

Uncertainty. Intervals resample whole templates with 10,000 bootstrap draws, keeping all inputs, conditions, and models of a template together. The seed is 20261005 for the TASC-M and 20261004 for the TASC-P. The intervals measure variation across scenarios with the set of evaluated models held fixed.

<table><tr><td></td><td>CATCH</td><td>HEDGE</td><td>ASK PROCEED</td></tr><tr><td>CATCH</td><td>401</td><td>1</td><td>0</td></tr><tr><td>HEDGE</td><td>5</td><td>0</td><td>1</td></tr><tr><td>ASK</td><td>0</td><td>0</td><td>0</td></tr><tr><td>PROCEED</td><td>5</td><td>2</td><td>1 538</td></tr></table>

Table 8: Original judge labels. Rows: Claude Sonnet 5; columns: GPT-5.6 Luna.

Shared TASC-P coverage. The common 32- scenario subset contains 416 tuples: Condition A is correct in 352, Condition B catches 140, and Condition C catches 266. Condition B misses among correct Condition A answers are 216/352 (61.4%), preserving the TASC-P pattern after the incomplete scenario is removed.

Agreement on the TASC-P. The two judges overlap on 986 responses: 428 Condition B violations, 428 Condition C violations, and 130 nonviolation controls. Table 8 reports their original four-category labels before the nine final corrections. The judges agree on 939/986 responses (95.23%; Cohen’s κ = 0.9047). This overlap covers 76.8% of the 1,284 labeled responses. The other 298 controls receive rule-screened PROCEED labels.

Collapsing the labels to CATCH /non- CATCH yields agreement of 952/986 (96.55%) and κ = 0.9294. Agreement measures consistency between automated judges. Further annotation must inspect responses on which the two judges agree and controls screened by rules.

Human audit. A pre-registered humanannotation study was designed and collected before this paper was drafted. The annotator worked blind in two phases, committing each label before the recorded label was revealed. Within the TASC-P, the sample covers 144 judged responses spanning all 33 scenarios and all 13 models, 21 adversarially flagged CATCH labels, six rule-screened controls, and 20 duration-parser checks. The study also includes a census of all 161 scenarios (the 128 TASC-M templates and the 33 TASC-P scenarios), in which every one of the 33 TASC-P scenarios was kept with all six checklist items affirmed; the seven scenarios flagged by the annotator belong to the TASC-M. The same audit reports κ = 0.53 on the TASC-M labels used in this paper. Table 9 reports agreement with the recorded labels by stratum.

Beyond label agreement, the annotator confirmed 19 of the 21 flagged CATCH labels (90.5% [71.1, 97.4]), found one non- PROCEED reply among the six rule-screened controls, matched parser v7.1 on 17 of the 20 duration probes, and, on the two corrected responses in the sample, sided with the corrected label in the CATCH sense both times.

<table><tr><td></td><td>κ</td><td>95% CI</td><td>n</td><td>Agr. (%)</td></tr><tr><td>All sampled (4-label)</td><td>0.708</td><td>0.60–0.82</td><td>144</td><td>84.7</td></tr><tr><td>Judge-labeled, vs. final</td><td>0.735</td><td>0.62–0.84</td><td>126</td><td>85.7</td></tr><tr><td>Violations (Condition B</td><td>0.768</td><td>0.65–0.88</td><td>114</td><td>87.7</td></tr><tr><td>&amp;Condition C)</td><td></td><td></td><td></td><td></td></tr><tr><td>Condition C</td><td>0.846</td><td>0.71-0.96</td><td>49</td><td>91.8</td></tr><tr><td>Condition B violations</td><td>0.707</td><td>0.54–0.87</td><td>65</td><td>84.6</td></tr><tr><td>Non-violation controls</td><td>0.330</td><td>-0.06-0.69</td><td>30</td><td>73.3</td></tr></table>

Table 9: Human–LLM agreement with the recorded labels. κ is Cohen’s κ; Agr. is agreement.

The non-violation controls are the audit’s weakest stratum: the annotator read seven judge-labeled control replies, and four rule-screened ones, as asserting the removed mismatch where the recorded label says none. If those readings survive adjudication, the reported false-alarm rate is a floor. They are the first targets for the remaining annotation.

The audit provides single-annotator agreement with the recorded labels. The calibration consensus equaled the judge’s own labels, so calibration demonstrates rubric fit rather than independent anchoring; and the reveal-stage agree-clicks, which enter no statistic, contradict the committed label on 75 of 360 draw items, bounding the weight the reveal stage can carry. Building a gold standard remains future work: blind double annotation of the TASC-P replies and stratified TASC-M replies, scored for recognition, action adjustment, and usefulness, adjudicating the control replies the annotator read as false alarms.