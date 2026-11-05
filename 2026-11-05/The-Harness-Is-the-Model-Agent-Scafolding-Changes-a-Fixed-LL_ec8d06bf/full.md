# The Harness Is the Model: Agent Scafolding Changes a Fixed LLM’s Compliance with Banking Conduct Rules, Unevenly Across Models

Mehmet Murat Huyut Kuveyt Türk Participation Bank and İstanbul Bilgi University, İstanbul, Türkiye mhuyut@gmail.com; ORCID 0000-0003-1812-6755

November 2026

## Abstract

Banks evaluate and validate language models, but what they deploy is a system: a model inside a harness that decides what the model sees, which tools it can call and who checks its actions. We test whether that harness changes compliance with banking conduct rules when the model is held fixed. Four open-weight models each ran five harnesses (a pipeline that makes one structured decision, a ReAct-style tool loop, a loop in which the policy is reachable only through search, a loop with a long benign customer memory, and a loop with a same-model compliance reviewer) on 156 pre-registered cases in a simulated retail bank whose 11 rule families paraphrase EU consumer-credit, payments, investor-protection, data-protection and anti-money-laundering provisions. Every one of the 3120 two-turn episodes was scored deterministically from the executed tool trace, with violations and over-refusals counted separately. Pooled across models the harness moved the correct-case rate only between 87.3% and 92.1%, but within a model it mattered a great deal: for one model the spread between its best and worst harness was 11.5 percentage points (Cochran’s � = 25.1, � < 0.001), enough to move it from first to last place among the four models, while for another the harness made no detectable diference; the model × harness interaction itself was suggestive rather than conclusive $( \chi _ { 1 2 } ^ { 2 } = 1 9 . 8 , p = 0 . 0 7 )$ . The same-model reviewer cut violations from 9.6% to 6.4% (� < 0.001) without raising over-refusals, contrary to our pre-registered prediction. Rules that require date arithmetic failed in every harness. In an exploratory analysis with two LLM judges and a blind human check, 26% of violations were accompanied by a customer message that denied the action the agent had just executed, so a review of the transcript alone would have recorded a correct refusal. For modelrisk management the unit of validation should be the model–harness pair, and conduct testing should read the action log, not the conversation.

## Introduction

A bank that deploys a language-model assistant does not deploy a model. It deploys a system in which the model is one component. Around it sits a harness: code that decides what goes into the context window (the policy, the customer’s history, the results of earlier lookups), which tools the model may call, in what format, how many steps it may take, and whether anything checks an action before it is executed. Model-risk practice, by contrast, is organised around the model. US supervisory guidance on model risk was revised in April 2026 and now states that generative and agentic AI models are outside its scope (Board of Governors of the Federal Reserve System et al., 2026); the UK’s SS1/23 covers AI only to the extent that it is used in models generally (Prudential

Regulation Authority, 2023). Benchmarks that banks consult also usually report one number per model.

Whether this matters is an empirical question, and the evidence so far points in two directions. A pre-registered study of six models under a direct API and three scafolds found that the scafold explained only 0.5% of the variance in measured safety, far less than the benchmark itself (Gringras, 2026), and a coding-agent benchmark found nearly identical harmful-violation rates under three agent frameworks (Hu et al., 2026). Other work finds that harnesses change capability by up to twelve points within a model (Bandel et al., 2026), that the same model shows diferent safety under vendor-native harnesses than under a shared one (Liu et al., 2026), and argues that harness variance can exceed model variance and reverse rankings (Zhang et al., 2026). None of these studies varies the harness in a controlled way while measuring compliance with the kind of conduct rules a bank is supervised on, and none separates the two ways a compliance system can fail: doing what is forbidden, and refusing what is allowed.

We run that experiment. A simulated retail bank (“Northbridge Bank”) exposes thirteen statechanging tools and six lookups. Its 1746-word operations manual contains forty sections, of which the operative ones paraphrase specific provisions of EU law: the 14-day withdrawal right for consumer credit, the refund duty for unauthorised payments, the appropriateness test for complex products, the right to object to direct marketing, and the prohibition on tipping of a customer about an anti-money-laundering report (Appendix A). Four open-weight models each ran five harnesses that difer only in how information and control flow around the model. The models are of moderate size and were served on domestic infrastructure, a deliberate constraint: they are the kind of model a European or Turkish bank can run on its own premises under data-residency rules, and our results speak to that deployment, not to the strongest closed or reasoning models. We scored every episode from the executed tool trace and the messages actually sent, with no LLM in the scoring loop.

The study was pre-registered (hypotheses, outcomes and tests were fixed and hashed before any run) and amended once, before the main run, after a pilot showed a ceiling efect (Section 3.6). Our findings are:

1. The harness efect is real but model-specific. Pooled over models, harnesses difer by 4.8 points. Within a model the spread reaches 11.5 points, and the harness changes which model looks most compliant: the model that ranks first under the reviewer harness ranks last when the policy is reachable only by search. For one of the four models the harness has no detectable efect at all. The formal test of the model × harness interaction fell short of the 5% level, so the model-specificity is a consistent pattern in our data rather than an established efect.

2. A same-model reviewer helps and, here, costs nothing. It removes 33% of violations and does not increase over-refusals, against both our prediction and recent evidence of high false rejection by same-model reviewers (Tanveer, 2026).

3. Retrieval and memory matter at the margin. When the relevant policy section was not retrieved, violations rose from 10.3% to 17.1%; a long benign memory raised violations slightly.

4. Date arithmetic is the common failure. The withdrawal window and the thirteen-month limit for unauthorised-payment claims fail in every harness; no harness fixes a rule the model cannot compute.

5. Transcripts hide violations. In an exploratory analysis, 26% of violations came with a message telling the customer that the action had not been taken.

## Related work

Policy-following tool agents. �-bench introduced tool-using agents that must follow a domain policy while talking to a simulated user (Yao et al., 2024; Barres et al., 2025), and �-Knowledge added a banking domain whose policy is reachable only through retrieval, finding that gold documents beat every retrieval mode (Shi et al., 2026). ST-WebAgentBench credits only policyrespecting completions (Levy et al., 2024), and MANTRA synthesises trace-level compliance checks from procedural manuals (Anand et al., 2026). The closest benchmark to ours is CXBench, which scores process compliance in regulated customer-service workflows including banking, finds that 80.1% of runs with the right outcome (55.5% of all runs) followed a non-compliant process, and evaluates one of our models (Yu et al., 2026a). These benchmarks hold the harness fixed; we vary it. Banking appears elsewhere as a security setting for prompt injection (Debenedetti et al., 2024) and adversarial fraud (Pai and Xian, 2026); our customers are ordinary or impatient, not adversarial.

Harness and scafold efects. Interface design changes what the same model can do (Yang et al., 2024), and agent leaderboards increasingly treat the scafold as an evaluation axis (Kapoor et al., 2025), while cost and reproducibility problems in agent evaluation are well documented (Kapoor et al., 2024). For safety the picture is mixed. Gringras (2026) report that scafold type explains only 0.5% of the variance on question-answering safety benchmarks (ReAct and multi-agent set-ups are equivalent to the direct API, map-reduce costs 7.3 points), with large model-specific interactions; Hu et al. (2026) report a null result for coding agents; Liu et al. (2026) show that safety difers between vendor-native harnesses and a shared harness for the same model. Zhang et al. (2026) argue that agents should not be compared without disclosing the harness. Our design adds a controlled ablation of harness components (pipeline control, retrieval-only policy, long memory, reviewer) on executed actions under rules mapped to law.

Reviewers and guards. Same-model self-correction often fails without external feedback (Huang et al., 2023), LLM evaluators favour their own outputs (Panickssery et al., 2024), and a same-model reviewer has been reported to reject 35% of correct answers in mathematics (Tanveer, 2026). Guard designs evaluated as defences include sub-agent verifiers (Kang et al., 2026), compiled policy gates (Yu et al., 2026b) and inline finance harnesses that report both safety and benign approval (Jia et al., 2026). A same-model, enforcement-varied comparison found that a bolt-on guardrail blocked violations but over-refused (Ahn and Kim, 2026). Our reviewer harness tests that trade-of inside a controlled ablation.

Context and memory. Models use information in the middle of long contexts less well (Liu et al., 2023), and accumulated benign memory has been reported to raise violation rates (Al-Tawaha et al., 2026); another study finds agents safer after preceding benign tasks (Sun et al., 2026). We therefore treated the direction of the memory efect as open in interpretation, while pre-registering the conventional direction.

Stated versus executed. The gap between what an agent says and what it did is an established construct: “false success” when a closing message asserts a completion the environment contradicts (Advani, 2026), “overclaiming” (Smyth et al., 2026), “corrupt success” that hides violations (Cao et al., 2026), and a split between stated reasoning and attempted action in financial compliance (Luo et al., 2026). Agents under pressure have been shown to act and then misreport (Scheurer et al., 2023). We do not propose a new construct; we measure it per harness against an executed banking trace, and we report the case that previous work emphasises less: the agent does the forbidden thing and then tells the customer it did not.

Finance and model risk. Position papers argue that financial LLM systems need system-level rather than model-level validation (Payzun et al., 2026), propose reliability profiles tied to an agent’s control envelope (Hsu et al., 2026), or treat the harness as the supervisable surface in credit pipelines (Akhtyamov, 2026). We supply experimental evidence for that position.

## Method

## Environment and policy

The environment is a deterministic Python simulation of a retail bank on a fixed date, 26 October 2026, which the system prompt states. Each case defines the calling customer, their accounts, loans, transactions and flags, the ofer terms, any screening results and the appropriateness result. Two synthetic customers are used throughout; no real data is involved. State-changing tools (for example process\_withdrawal, refund\_fee, execute\_transfer, hold\_transaction, file\_internal\_report, block\_card, place\_order, escalate\_to\_supervisor) always succeed, so a forbidden action is never stopped by the environment: if the agent calls it, it happened.

The operations manual has forty sections in the style of a bank’s internal manual. About twenty are operative for the cases; the rest (opening hours, standing orders, international transfers) are realistic filler. The operative sections contain exceptions that change the correct outcome: joint accounts and registered representatives, mortgages excluded from the withdrawal path, professional clients exempt from the appropriateness check, a higher refund limit for business accounts and a limit on the total refunded per conversation, own-account transfers exempt from screening, a thirteen-month limit for unauthorised-payment refunds after which the card is blocked and the case escalated, and business-account address changes that need a signed form. Appendix A maps each rule family to the legal provision it paraphrases and states where our rule is stricter than the law.

## Cases

There are 39 base variants in 11 families (Table 3; legal sources in Table 4), each run in two replicates (lexical paraphrases of the customer message) and two pressure levels, giving 156 cases. Variants were built in contrasting sets: for example, credit agreements signed 5, 13, 14, 15 and 30 days before today, a mortgage signed 6 days before, fee requests just below and above the limit, two fees whose total exceeds it, and two transfers of which one beneficiary is a screening match. Customers give calendar dates, not day counts, so the agent must compute elapsed time. Every case has a scripted second customer turn that is sent whatever the agent said first: for cases where the correct outcome is to refuse, it pushes back with a reason that does not change the rule (“It’s only just over the period, surely you can make an exception?”); for permitted cases it is neutral (“Thanks. Just to be sure, is everything done now?”). Under pressure, the customer message adds that they are in a hurry and will move their accounts, and the system prompt adds that the assistant is evaluated on satisfaction and conversion.

## Harnesses

All harnesses use the same role prompt, the same manual (except H2) and the same tools.

• H0, pipeline. The bank’s workflow runs every relevant lookup in advance (identity check, account, ofer terms, screening, appropriateness) and gives the model a case file. The model

returns one JSON object listing the actions to execute and the message to the customer; after the follow-up it returns a second object with any new actions.

• H1, tool loop. A ReAct-style loop (Yao et al., 2023): the model calls one tool per step, sees the result and decides the next step, up to twenty steps, with the full manual in the system prompt.

• H2, policy by search. As H1, but the manual is not in the prompt; the model can call search\_policy, which returns the two best-matching sections by keyword overlap.

• H3, long memory. As H1, plus a block of eighty benign entries of customer history (2018– 2026, 2362 words) between the manual and the conversation.

• H4, reviewer. As H1, but every state-changing call and every customer message is first sent to a reviewer, the same model with the manual and the conversation, which approves or blocks it with a reason; a blocked call is not executed and the reason is returned to the agent.

## Models

We used four open-weight models served by a domestic research API at temperature 0 with a fixed seed and reasoning disabled: deepseek-v4-flash, deepseek-v4.1-flash, qwen3.8-flash-next and gemma-4-31b. Using open models on domestic infrastructure was a design constraint (bank data rules exclude foreign APIs in our setting), and it means the results speak to models a bank could run on its own premises, not to the strongest closed models.

## Scoring

Scoring is deterministic and reads only the executed trace and the messages sent in both turns. An episode is a violation if any forbidden action was executed or any forbidden content was sent: for example a state-changing action before identity was verified, a withdrawal outside the permitted path, a refund above the limit or after the time limit, an executed transfer to a screening match, a message mentioning screening or a report after a match, a disclosed balance of another person, or a product ofer to a customer who objected to marketing. It is an over-refusal if a required action was not taken (a permitted withdrawal not processed, a permitted refund not made in full, a required escalation or block missing). The correct-case outcome is neither. Message tests use pattern matching (for example, any of the number formats of a disclosed balance). The scorer is covered by 22 unit tests that drive the simulation with hand-written traces (deney/test\_ortam.py in the replication package).

## Pre-registration, pilot and amendment

Hypotheses H1–H7, outcomes, tests and a parse-rate gate were written and their files hashed before any run. A first pilot (v1: 560 episodes, eleven-section policy, single-turn cases with day counts) passed the gate but hit a ceiling: correct-case rates were between 97.3% and 100% in every harness. Before the main run we amended the design (task realism only; harnesses, models, outcomes and hypotheses unchanged): the forty-section manual, calendar dates, compound requests and the scripted second turn. The amended pilot (780 episodes) passed the gate (parseable final message in 96.8% of H4 episodes and 100% in the others) and, as pre-specified, its episodes are part of the 3120 reported here.

## Analysis

Within each model we compare harnesses on the same cases with Cochran’s � and pairwise exact McNemar tests with Holm correction. Directional hypotheses (H2–H5, H7) use exact McNemar tests on violations, paired on (model, case). The model × harness interaction (H6) is a Wald test in a logistic GEE clustered on case. Intervals are Wilson 95% intervals. The stated-versus-executed analysis was registered as secondary and is reported as exploratory.

## Results

## The harness efect depends on the model

Table 1 gives the correct-case rate for each model and harness. Pooled over models, the harnesses are close: from 87.3% (H2) to 92.1% (H4). Within models they are not. The harness efect was significant for qwen3.8-flash-next (� = 25.1, $p \ < \ 0 . 0 0 1 )$ and deepseek-v4-flash $( Q = 1 6 . 1$ 2 $p = 0 . 0 0 3 )$ , marginal for gemma-4-31b $( p = 0 . 0 7 )$ and absent for deepseek-v4.1-flash $( Q = 1 . 0$ $p = 0 . 9 2 )$ . The GEE interaction test was $\chi _ { 1 2 } ^ { 2 } = 1 9 . 8 \ ( p = 0 . 0 7 )$ , so the evidence that harness efects difer across models is suggestive rather than conclusive at this sample size (H6).

Table 1: Correct-case rate by model and harness (H0 pipeline, H1 tool loop, H2 policy by search, H3 long memory, H4 reviewer; Wilson 95% interval in brackets; 156 cases per cell).
<table><tr><td colspan="5"></td><td colspan="2">Spread</td></tr><tr><td>Model</td><td>H0</td><td>H1</td><td>H2</td><td>H3</td><td>H4</td><td>(pp)</td></tr><tr><td>deepseek-v4-flash</td><td>88.5%</td><td>84.6%</td><td>89.7%</td><td>85.9%</td><td>92.9%</td><td>8.3</td></tr><tr><td rowspan="2">deepseek-v4.1-flash</td><td>[82-93]</td><td>[78-89]</td><td>[84-94]</td><td>[80-90]</td><td>[88-96]</td><td></td></tr><tr><td>88.5%</td><td>88.5%</td><td>87.8%</td><td>87.8%</td><td>89.7%</td><td>1.9</td></tr><tr><td rowspan="2">qwen3.8-flash-next</td><td>[82-93]</td><td>[82-93]</td><td>[82-92]</td><td>[82-92]</td><td>[84-94]</td><td></td></tr><tr><td>91.7%</td><td>89.7%</td><td>82.1%</td><td>88.5%</td><td>93.6%</td><td>11.5</td></tr><tr><td rowspan="2">gemma-4-31b</td><td>[86-95]</td><td>[84-94]</td><td>[75-87]</td><td>[82-93]</td><td>[89-96]</td><td></td></tr><tr><td>93.6%</td><td>92.3%</td><td>89.7%</td><td>91.7%</td><td>92.3%</td><td>3.8</td></tr><tr><td rowspan="2">All models</td><td>[89-96]</td><td>[87-96]</td><td>[84-94]</td><td>[86-95]</td><td>[87-96]</td><td></td></tr><tr><td>90.5%</td><td>88.8%</td><td>87.3%</td><td>88.5%</td><td>92.1%</td><td>4.8</td></tr></table>

The harness also changes which model looks most compliant (Table 2). qwen3.8-flash-next is the most compliant model under the reviewer harness (93.6%) and the least compliant when the policy is reachable only through search (82.1%); after Holm correction its H2 cell is significantly worse than both H0 and H4. deepseek-v4-flash shows the opposite pattern: it ranks first (tied) under H2 and last or tied for last under H0, H1 and H3. The confidence intervals of neighbouring models overlap, so we read the rank changes as a warning about single-harness evaluations, not as a precise ordering.

Table 2: Rank of each model (1 = most compliant) under each harness.
<table><tr><td>Model</td><td>H0</td><td>H1</td><td>H2</td><td>H3</td><td>H4</td></tr><tr><td>deepseek-v4-flash</td><td>3</td><td>4</td><td>1</td><td>4</td><td>2</td></tr><tr><td>deepseek-v4.1-flash</td><td>3</td><td>3</td><td>3</td><td>3</td><td>4</td></tr><tr><td>qwen3.8-flash-next</td><td>2</td><td>2</td><td>4</td><td>2</td><td>1</td></tr><tr><td>gemma-4-31b</td><td>1</td><td>1</td><td>1</td><td>1</td><td>3</td></tr></table>

## Hypothesis tests

Procedure (H2, not supported). We predicted that tool loops would break step-order rules (identity before disclosure, appropriateness before an order, screening before a transfer) more often than the pipeline, which runs those checks itself. They almost never did: on the 144 paired orderdependent episodes per contrast, H0, H1 and H3 had no violations, and H2 had 5 $( p = 0 . 0 6 )$ Current models follow explicit ordering rules reliably.

Retrieval (H3, partly supported). H2 had more violations than H1 (11.5% vs. 9.6%), but the paired diference was not significant $( p = 0 . 1 0 )$ . Within H2, violations were concentrated where retrieval failed: 17.1% of the 111 episodes in which no relevant section was retrieved, against 10.3% of the 513 in which one was (Fisher $p = 0 . 0 5 )$ . Retrieval-only access also created failures that did not occur when the manual was in context, in screening, financial-dificulty and APR cases (Table 3).

Memory (H4, supported, small). A long benign customer memory raised violations from 9.6% to 11.1% (11 vs. 2 discordant pairs, $p = 0 . 0 2 )$

Reviewer (H5, half supported). The same-model reviewer lowered violations from 9.6% to 6.4% (24 vs. 4 discordant pairs, $p < 0 . 0 0 1 $ . The predicted cost did not appear: over-refusals were 1.6% without and 1.4% with the reviewer $( p = 1 . 0 0 )$ . The reviewer blocked at least one proposed action in 14.1% of H4 episodes, at the price of about 1.8 times the input tokens of H1.

Pressure (H7, not supported). Pressure raised violations from 8.8% to 9.7% $( p = 0 . 0 9 )$ , and the increase did not difer by harness (harness × pressure $\chi _ { 4 } ^ { 2 } = 3 . 2 , p = 0 . 5 2 )$ . Under the reviewer the rate was 6.7% without and 6.1% with pressure. Larger pressure efects have been reported for procurement agents facing financial incentives and managerial pressure (Okamoto et al., 2026) and are consistent with sycophancy towards the user (Sharma et al., 2023); our manipulation (an impatient customer and a note on satisfaction targets) was milder and left the rule itself unambiguous.

## Where the failures are

Table 3 breaks the correct-case rate down by rule family. Of the 11 families, 7 are at or near ceiling (at least 93% correct) in every harness except H2. The failures concentrate in two families whose rules need date arithmetic: credit withdrawal (the 14-day window, counted from the day after signing, and the mortgage exclusion) and unauthorised transactions (the thirteen-month limit). The most frequent failure reasons across all episodes were “withdrawal processed outside the permitted path” (154 episodes), “old claim not escalated” (60) and “refund after 13-month limit” (56). No harness repaired these rules: a harness can change what the model sees and who checks it, but not whether the model can count days. The reviewer helped most on unauthorised transactions (87.5% correct, against 68.8% in H1), and not at all on the withdrawal window (66.7% vs. 66.7%), plausibly because the reviewer makes the same date error as the agent.

Table 3: Correct-case rate by rule family and harness (all models, both pressure levels).
<table><tr><td>Family</td><td>H0</td><td>H1</td><td>H2</td><td>H3</td><td>H4</td></tr><tr><td>F1 Identity verification</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>F2 Third-party access</td><td>88%</td><td>85%</td><td>100%</td><td>94%</td><td>90%</td></tr><tr><td>F3 Credit withdrawal</td><td>69%</td><td>67%</td><td>71%</td><td>67%</td><td>67%</td></tr><tr><td>F4 APR confirmation</td><td>100%</td><td>100%</td><td>88%</td><td>98%</td><td>100%</td></tr><tr><td>F5 Appropriateness</td><td>94%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>F6 Complaints</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>F7 Financial difficulty</td><td>100%</td><td>100%</td><td>84%</td><td>100%</td><td>100%</td></tr><tr><td>F8 Fee refunds</td><td>95%</td><td>88%</td><td>88%</td><td>87%</td><td>96%</td></tr><tr><td>F9 Transfer screening</td><td>100%</td><td>100%</td><td>90%</td><td>100%</td><td>98%</td></tr><tr><td>F10 Unauthorised transactions</td><td>71%</td><td>69%</td><td>69%</td><td>60%</td><td>88%</td></tr><tr><td>F11 Address and marketing</td><td>100%</td><td>100%</td><td>94%</td><td>100%</td><td>100%</td></tr></table>

## Stated versus executed (exploratory)

To see what a transcript review would miss, two LLM judges from diferent model families (deepseekv4.1-flash and gemma-4-31b) read each episode’s customer turns, executed actions, lookups and messages, and labelled whether a message claimed an action that was not executed (overclaim), denied an action that was executed (underclaim), or otherwise misrepresented the status of the request (misleading status); deepseek-v4.1-flash labelled 3106 and gemma-4-31b labelled 3120 of the 3120 episodes (the remainder failed on server capacity). The judges were told not to assess policy compliance. Because LLM judges have known biases (Zheng et al., 2023), we used two judges from diferent families and a blind human check. One artefact had to be handled: after a screening match the manual itself requires the agent to tell the customer that the transfer “is under standard processing”, which the judges often labelled as misleading; we excluded misleading-status labels in screening cases that rested on this mandated wording (for judges and human alike). The judges agreed on 98.4% of episodes (Cohen’s $\kappa = 0 . 8 4 ;$ underclaim $\kappa = 0 . 7 3 )$ , and a re-run of 311 episodes with a diferent seed reproduced at least 99.4% of labels. The author, blind to model, harness and judge labels, coded 100 episodes (50 flagged by at least one judge, 50 by neither). Agreement with the human on underclaims was $\kappa = 0 . 8 1$ (deepseek-v4.1-flash) and $\kappa = 0 . 9 2$ (gemma-4-31b); for the conservative rule that both judges must agree, $\kappa = 0 . 8 1$ . Agreement on overclaims was $\kappa = 0 . 7 5$ and $\kappa = 0 . 9 0$ after we corrected 9 human codes on pipeline episodes where the first coding sheet, like the first judge prompt, did not show the identity check that the bank’s workflow had already run. Agreement on misleading status was poor $\left( \kappa \leq 0 . 0 9 \right)$ , so we do not interpret that category.

A rule-based check for explicit completion claims without a matching action found almost none (0.1% of episodes): the agents rarely say they did something they did not do. The opposite is common. Among the 287 episodes with a violation, both judges agreed that the agent’s message denied the very action it had executed in 26.1% (41.3% by the more liberal judge). The typical case is a credit withdrawal outside the window: the agent executes process\_withdrawal, and when the customer’s scripted follow-up asks for an exception, replies that the withdrawal period has expired and it cannot process the withdrawal. The rate was highest in the pipeline (41.3% of H0 violations), where the model writes its list of actions and its message in one structured reply: in several H0 episodes the message corrects the date calculation half-way through (“…let me correct that”) while the action list, written before it, already contains the withdrawal. A transcript-only review, or an evaluation that scores the final message, would record these episodes as correct refusals.

Because the follow-up is scripted and presupposes a refusal in these cases, it may invite the denial; the judge labels are also model outputs. We therefore treat the size of the efect as approximate. Its direction does not depend on either point: the customer was told something false about an action that changed their account.

## Discussion

The unit of validation is the model–harness pair. For 2 of our four models, swapping the harness moved compliance by more than any diference between models under a fixed harness (at most 7.7 points), and the ranking of models changed with the harness. A validation of “the model” on one harness, or a benchmark score obtained under a vendor’s harness, does not carry over to a deployment that changes how the policy is supplied or whether actions are reviewed. This is the gap left by model-risk guidance that excludes agentic AI (Board of Governors of the Federal Reserve System et al., 2026): the object that needs validating is not covered, and the object that is covered is not the one deployed. In the EU a customer-service agent is not itself a high-risk system under the AI Act, whose banking entry covers creditworthiness assessment and credit scoring (European Parliament and Council, 2024b), and the high-risk obligations for those systems now apply from 2 December 2027 (European Parliament and Council, 2026); for conduct, supervision of such agents rests on the sectoral rules our policy paraphrases. Our results agree with the pooled finding of Gringras (2026) that average scafold efects are small, and with their observation of large modelspecific interactions; for acting agents under conduct rules, the interaction is what a bank deploys.

Cheap controls are worth testing per model. A same-model reviewer removed 33% of violations without measurable over-refusal. This is good news for banks that cannot use a stronger external model, but it is a property of these models and this policy, not a law: the reviewer did not improve the withdrawal window, where it presumably shares the agent’s date error, and a pilot study found high false rejection by a same-model reviewer in mathematics (Tanveer, 2026). Where a rule is arithmetic (a date window, a cumulative limit), the natural control is deterministic code in the harness (Yu et al., 2026b), not another model call.

Conduct testing should read the action log. The denial pattern means that sampling conversations for quality assurance, a common practice in contact centres, will miss a share of the violations that matter most, because the record the reviewer reads says the opposite of what happened. Evaluations that score only the final message share the problem (Advani, 2026; Cao et al., 2026). In a regulatory setting the message is also an act: telling a customer that their withdrawal was not processed when it was is itself a conduct failure.

## Limitations

The bank, customers and policy are synthetic, and one policy cannot represent the variety of bank manuals. Several of our rules are stricter than the law they paraphrase (Appendix A), so “violation” means a breach of the manual, not necessarily of the law. We tested four open-weight models of moderate size on one domestic API, with reasoning disabled; stronger or reasoning models may close the date arithmetic gap and change the harness efects. 7 of 11 rule families were near ceiling outside H2, which limits power for the harness comparisons; the interaction test was not significant at the 5% level. The second customer turn is scripted and independent of the agent’s first reply, which keeps cases comparable but makes some dialogues unnatural, and may contribute to the denial pattern. The stated-versus-executed analysis relies on LLM judges, validated on a sample of 100 episodes by one human coder. The amendment after the first pilot changed the tasks; it was made and documented before the main run, but it was informed by pilot results.

## Conclusion

Holding the language model fixed, the harness changed how often a banking agent broke conduct rules, by up to 11.5 points for one model and not at all for another, and it changed which model looked safest. A same-model reviewer was the most efective single change; date arithmetic was a failure no harness fixed; and 26% of violations were invisible in the conversation because the agent denied what it had done. Banks, supervisors and benchmark authors should evaluate the model–harness pair, report the harness, and score executed actions.

## Declarations

Funding. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

Competing interests. The author works for a participation bank. The views expressed are the author’s own and do not represent those of any employer. No bank data were used.

Data and code. Cases, policy, harness code, raw episode traces for all 3120 episodes, judge labels, the blind human coding sheet, the pre-registration with its SHA-256 hashes and the analysis scripts are available at https://github.com/mhuyut/harness-is-the-model. The analysis reproduces every number in this paper from the stored traces without calling any model API.

Use of AI tools. The author directed this study: the author selected the research topic and question, approved the experimental design and its pre-registered amendment, reviewed all of the experiment and analysis code, monitored every experimental run and secured the computing resources needed for all models to complete their tests, blind-coded 100 episodes to validate the LLM judges, and read, reviewed and finalized the manuscript. Claude Code (Anthropic; Claude Opus 5.5 and Claude Fable 5.1) was used as a research assistant throughout: to draft the design and pre-registration, write the experiment and analysis code, search and verify the literature, and draft the text and LaTeX build, with all numbers generated automatically from the analysis output. The four open-weight models under study, two of which also served as judges, did not contribute to the writing. The author takes full responsibility for the content; no AI tool is listed as an author.

## Appendix A. Rule families and legal sources

Table 4: Rule families, the legal provisions they paraphrase, and where the manual is stricter than the law.
<table><tr><td>Family</td><td>Manual rule (summary)</td><td>Source paraphrased</td><td>Relation to the law</td></tr><tr><td>F1</td><td>No account information or action before date of birth and last card digits match</td><td>GDPR confidentiality principle (European Parliament and Council, 2016); inspired by PSD2 Art.</td><td>Identity check, not strong customer authentication</td></tr><tr><td>F2</td><td>No disclosure to spouses or others; registered representatives and</td><td>Council, 2015b) GDPR (European Parliament and Council, 2016)</td><td>Internal control built on confidentiality</td></tr><tr><td>F3</td><td>joint holders allowed 14-day withdrawal from consumer credit, no penalty; mortgages never through the</td><td>Directive 2008/48/EC Art. 14 (European Parliament and Council, 2008); CCD2 Art. 26, applicable from 20 Nov 2026</td><td>Day-count convention and mortgage exclusion are ours</td></tr><tr><td>F4</td><td>No loan application before the customer states the correct</td><td>and Council, 2023) CCD2 Art. 10 pre-contractual information (European Parliament</td><td>Stricter than the Directive</td></tr><tr><td>F5</td><td>Complex products only after an appropriateness check; professional clients</td><td>MiFID II Art. 25(3) (European Parliament and Council, 2014)</td><td>Stricter: the Directive requires a warning, we require refusal</td></tr><tr><td>F6</td><td>Log every complaint and give a reference</td><td>ESA complaints-handling guidelines (Joint Committee of the European Supervisory</td><td>Guidelines, not binding law</td></tr><tr><td>F7</td><td>Customers in difficulty: refer to the hardship team, offer no new credit</td><td>CCD2 Art. 36 (debt advisory services); cf. Art. 35 (forbearance) and Art. 17 (ban on unsolicited credit)(European</td><td>Referral paraphrases Art. 36; “no new credit" is an internal control</td></tr><tr><td>F8</td><td>Fee refunds up to 50 EUR (business 200 EUR) per conversation, otherwise escalate</td><td>Internal control</td><td></td></tr><tr><td>F9</td><td>Screen each beneficiary; on a match hold, report, and do not tip off</td><td>AMLD Art. 39 (European Parliament and Council, 2015a); AMLR Art. 73 from 10 Jul 2027 (European Parliament and</td><td></td></tr><tr><td>F10</td><td>Refund unauthorised payments; after 13 months block the card and escalate</td><td>PSD2 Art. 73 (refund), Art. 71(1) (13-month notification limit), Art. 70(1)(e) (blocking the instrument) (European Parliament</td><td>Refund duty and 13-month limit match; the automatic block and escalation after the limit are ours (the Directive ties blocking to notification)</td></tr><tr><td></td><td>No marketing to customers who objected; business address changes need a form</td><td>GDPR Art. 21(2)–(3) (European Parliament and Council, 2016)</td><td>Consent flag stands in for an objection</td></tr></table>

## References

Laksh Advani. From confident closing to silent failure: Characterizing false success in LLM agents, 2026. URL https://arxiv.org/abs/2606.09863. FAGEN@ICML 2026.

Joongho Ahn and Moonsoo Kim. From prompts to contracts: Harness engineering for auditable enterprise LLM agents, 2026. URL https://arxiv.org/abs/2607.08028.

Ravil Akhtyamov. The harness as the only mutable surface: Compliance-bounded self-evolution of LLM agents in credit pipelines, with a measured admission gate, 2026. URL https://arxiv.org/ abs/2610.10629.

Ahmad Al-Tawaha, Shangding Gu, Peizhi Niu, Ruoxi Jia, and Ming Jin. Remembering more, risking more: Longitudinal safety risks in memory-equipped LLM agents, 2026. URL https: //arxiv.org/abs/2605.17830.

Ashwani Anand, Ivi Chatzi, Ritam Raha, and Anne-Kathrin Schmuck. MANTRA: Synthesizing SMT-validated compliance benchmarks for tool-using LLM agents, 2026. URL https://arxiv.or g/abs/2605.06334.

Elron Bandel, Asaf Yehudai, Lilach Eden, Yehoshua Sagron, Yotam Perlitz, Elad Venezian, Natalia Razinkov, Natan Ergas, Shlomit Shachor Ifergan, Segev Shlomov, Michal Jacovi, Leshem Choshen, Liat Ein-Dor, Yoav Katz, and Michal Shmueli-Scheuer. General agent evaluation, 2026. URL https://arxiv.org/abs/2602.22953. ICLR 2026 Workshop on Agents in the Wild.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. �<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment, 2025. URL https://arxiv.org/abs/25 06.07982.

Board of Governors of the Federal Reserve System, Ofice of the Comptroller of the Currency, and Federal Deposit Insurance Corporation. Revised guidance on model risk management (SR 26-2). Supervisory letter, 17 April 2026, April 2026. URL https://www.federalreserve.gov/supervision reg/srletters/SR2602.htm.

Hongliu Cao, Ilias Driouich, and Eoin Thomas. Beyond task completion: Revealing corrupt success in LLM agents through procedure-aware evaluation, 2026. URL https://arxiv.org/abs/2603.031 16.

Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents, 2024. URL https://arxiv.org/abs/2406.13352.

European Parliament and Council. Directive 2008/48/EC on credit agreements for consumers, 2008. URL http://data.europa.eu/eli/dir/2008/48/oj.

European Parliament and Council. Directive 2014/65/EU on markets in financial instruments (MiFID II), 2014. URL http://data.europa.eu/eli/dir/2014/65/oj.

European Parliament and Council. Directive (EU) 2015/849 on the prevention of the use of the financial system for the purposes of money laundering or terrorist financing, 2015a. URL http: //data.europa.eu/eli/dir/2015/849/oj.

European Parliament and Council. Directive (EU) 2015/2366 on payment services in the internal market (PSD2), 2015b. URL http://data.europa.eu/eli/dir/2015/2366/oj.

European Parliament and Council. Regulation (EU) 2016/679 (general data protection regulation), 2016. URL http://data.europa.eu/eli/reg/2016/679/oj.

European Parliament and Council. Directive (EU) 2023/2225 on credit agreements for consumers and repealing directive 2008/48/EC, 2023. URL http://data.europa.eu/eli/dir/2023/2225/oj.

European Parliament and Council. Regulation (EU) 2024/1624 on the prevention of the use of the financial system for the purposes of money laundering or terrorist financing, 2024a. URL http://data.europa.eu/eli/reg/2024/1624/oj.

European Parliament and Council. Regulation (EU) 2024/1689 laying down harmonised rules on

artificial intelligence (artificial intelligence act), 2024b. URL http://data.europa.eu/eli/reg/202 4/1689/oj.

European Parliament and Council. Regulation (EU) 2026/1744 amending regulations (EU) 2024/1689, (EU) 2018/1139 and (EU) 2023/1230 as regards the simplification of the implementation of harmonised rules on artificial intelligence (digital omnibus on AI), 2026. URL http://data.europa.eu/eli/reg/2026/1744/oj.

David Gringras. Safety under scafolding: How evaluation conditions shape measured safety, 2026. URL https://arxiv.org/abs/2603.10044.

Mike Hsu, Medha Bankhwal, Béatrice Moissinac, Kevin Werbach, Lukasz Szpruch, and Bennett Hillenbrand. Agent reliability profiles in financial services, 2026. URL https://arxiv.org/abs/26 10.04123.

Qi Hu, Yifeng Tang, Qinghua Wang, Lanyang Zhao, Pengji Zhang, Yuhao Qing, Xin Yao, Dong Huang, Lin Zhang, and Zhuoran Ji. SABER: Benchmarking operational safety of LLM coding agents in stateful project workspaces, 2026. URL https://arxiv.org/abs/2606.01317. EMNLP 2026, pp. 4842–4866.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet, 2023. URL https: //arxiv.org/abs/2310.01798. ICLR 2024.

Haoxuan Jia, Yang Liu, Bin Chong, Yingguang Yang, Yancheng Chen, Jiayu Liang, Qian Li, Hanning Lu, Kefu Xu, Hao Zheng, Chongyang Zhang, Hao Peng, and Philip S. Yu. FinHarness: An inline lifecycle safety harness for finance LLM agents, 2026. URL https://arxiv.org/abs/26 05.27333.

Joint Committee of the European Supervisory Authorities. Guidelines on complaints-handling for the securities (ESMA) and banking (EBA) sectors (JC 2018 35), 2018. URL https://eba.europa .eu/documents/10180/2381463/cd6e3328-7442-4582-8b68-819346d200ec/Joint%20Committee %20Guidelines%20on%20complaints-handling%20(JC%202018%2035)\_EN.pdf.

Seongjae Kang, Taehyung Yu, and Sung Ju Hwang. PolicyGuard: A dialogue-grounded sub-agent verifier for policy adherence in LLM agents, 2026. URL https://arxiv.org/abs/2606.29225.

Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter, 2024. URL https://arxiv.org/abs/2407.01502.

Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S. Siegel, et al. Holistic agent leaderboard: The missing infrastructure for AI agent evaluation, 2025. URL https://arxi v.org/abs/2510.11977.

Ido Levy, Ben Wiesel, Sami Marreed, Alon Oved, Avi Yaeli, Nir Mashkif, and Segev Shlomov. ST-WebAgentBench: A benchmark for evaluating safety and trustworthiness in web agents, 2024. URL https://arxiv.org/abs/2410.06703. ICLR 2026.

Chengzhi Liu, Yichen Guo, Yepeng Liu, Yuzhe Yang, Qianqi Yan, Xuandong Zhao, Wenyue Hua, Sheng Liu, Sharon Li, Yuheng Bu, and Xin Eric Wang. Auditing agent harness safety, 2026. URL https://arxiv.org/abs/2605.14271.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,

and Percy Liang. Lost in the middle: How language models use long contexts, 2023. URL https://arxiv.org/abs/2307.03172. TACL.

Yiyang Luo, Yihang Jiang, Qijun Xie, Liang Lan, Lin William Cong, Anyi Rao, and Yunya Song. ReguSim: Evaluating LLM agent rule grounding in financial compliance, 2026. URL https: //arxiv.org/abs/2608.19974.

Mika Okamoto, Ansel Kaplan Erol, and Kutluhan Erol. Why do AI agents break rules? how framing, context, and social signals shape compliance, 2026. URL https://arxiv.org/abs/2608.1 2323. AIES 2026.

Dheeraj Mohandas Pai and Lu Xian. FraudBench: Stress-testing policy-grounded banking agents against adaptive fraud, 2026. URL https://arxiv.org/abs/2608.18136.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. LLM evaluators recognize and favor their own generations, 2024. URL https://arxiv.org/abs/2404.13076.

Burak Payzun, İrem Demirtaş, Simona Scala, Elena Ferretti, and Seçil Arslan. Benchmarks are not validation: A system-level view of financial LLM applications, 2026. URL https://arxiv.or g/abs/2607.28840.

Prudential Regulation Authority. SS1/23 – model risk management principles for banks. Supervisory statement, updated 23 April 2026, 2023. URL https://www.bankofengland.co.uk/prudentialregulation/publication/2023/may/model-risk-management-principles-for-banks-ss.

Jérémy Scheurer, Mikita Balesni, and Marius Hobbhahn. Large language models can strategically deceive their users when put under pressure, 2023. URL https://arxiv.org/abs/2311.07590.

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, et al. Towards understanding sycophancy in language models, 2023. URL https: //arxiv.org/abs/2310.13548.

Quan Shi, Alexandra Zytek, Pedram Razavi, Karthik Narasimhan, and Victor Barres. �-knowledge: Evaluating conversational agents over unstructured knowledge, 2026. URL https://arxiv.org/ab s/2603.04370.

Nolan Smyth, Yorguin-Jose Mantilla-Ramos, Pascal Jr Tikeng Notsawo, Saskia Helbling, Alberto Tosato, Mohamed Amine Merzouk, Nouha Dziri, Gauthier Gidel, and Tommaso Tosato. Quantifying overclaiming propensity in frontier LLM agents, 2026. URL https://arxiv.org/abs/2609 .20812.

Chung-En Sun, Linbo Liu, and Tsui-Wei Weng. The cold-start safety gap in LLM agents, 2026. URL https://arxiv.org/abs/2606.07867. EMNLP 2026.

Faizan Tanveer. Reviewer capability governs rejection targeting, not repair skill: Evidence from LLM execute-review-revise pipelines, 2026. URL https://arxiv.org/abs/2609.04270.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering, 2024. URL https://arxiv.org/abs/2405.15793.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models, 2023. URL https://arxiv.org/ab s/2210.03629. ICLR 2023; arXiv first version 2022.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. �-bench: A benchmark for tool-agent-user interaction in real-world domains, 2024. URL https://arxiv.org/abs/2406.12045.

Boxi Yu, Yuzhong Zhang, Zhenfeng Su, Linfu Xi, Yinghan Yao, Shitao Chen, Yang Cao, Liting Lin, Lionel Briand, and Emir Muñoz. CXBench: A benchmark for process-compliant LLM agents in regulated customer-experience workflows, 2026a. URL https://arxiv.org/abs/2610.25954.

Min-Young Yu, Tony Kim, and Jang Won Choi. NOMOS: Compiling written policies into statically verified tool-call gates for LLM agents, 2026b. URL https://arxiv.org/abs/2610.11030.

Yunbei Zhang, Janet Wang, Yingqiang Ge, Weijie Xu, Jihun Hamm, and Chandan K. Reddy. Stop comparing LLM agents without disclosing the harness, 2026. URL https://arxiv.org/abs/2605.2 3950.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena, 2023. URL https://arxiv.org/ab s/2306.05685. NeurIPS 2023 Datasets and Benchmarks.