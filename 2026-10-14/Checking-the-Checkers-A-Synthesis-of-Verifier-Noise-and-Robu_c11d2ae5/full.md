# Checking the Checkers: A Synthesis of Verifier Noise and Robustness in RLVR

Savyasachi Gupta <sup>∗</sup>

Independent Researcher

TX, USA

gsavya10@gmail.com

Aniruddha Srinivas Joshi <sup>∗</sup>

Independent Researcher

CA, USA

aniruddha980@gmail.com

Abstract—Reinforcement learning with verifiable rewards (RLVR) has become a principal recipe for improving the reasoning of large language models (LLMs). It rests on the assumption that the checker scoring each response is itself correct. In practice, rule-based checkers reject correct answers in unexpected formats, unit tests accept incorrect programs, and LLM judges can be led to approve empty or adversarial responses. This work surveys 58 studies that measure, analyze, or mitigate such verifier errors. We ask what errors verifiers make, how they affect training, which mitigations work against which errors, and how the findings extend to self-generated, rubric- and judge-based, and verifier-free rewards. The evidence suggests that the structure of an error matters more than its rate. Random errors mostly slow learning, though under a fixed budget or KL regularization they also lower final accuracy. False positives that the policy can learn to trigger can cause plateaus or collapse, sometimes even with a low initial error rate. Whether a verifier wrongly rejects or accepts depends on what it checks and who built it, not on its family. Its errors shift as the policy improves, and most mitigations have been tested against only one kind of error. We compile the findings into comparison tables, reconcile apparent conflicts in the literature, and propose a reporting protocol to make future results comparable.

Index Terms—Group relative policy optimization, large language models, LLM-as-a-judge, noisy rewards, reinforcement learning, reinforcement learning with verifiable rewards (RLVR), reward hacking, reward models, verifier noise

## I. INTRODUCTION

Reinforcement learning (RL) post-training is now central to building large language models (LLMs) that reason well. Reinforcement learning with verifiable rewards (RLVR), a term introduced with the Tulu 3 post-training recipe [1],¨ replaces the learned reward model of classical RL from human feedback with a programmatic check: the final answer of a response is extracted and compared against a reference, or the generated program is executed against a set of unit tests. The approach was popularized by DeepSeek-R1 [2], which paired such rewards with Group Relative Policy Optimization (GRPO) [3], and it now underpins most open reasoning models. Its appeal is easy to see. The reward is cheap to compute, it needs no human annotation, and it has been widely assumed to be difficult to game [2], [4], [5].

However, a verifier is not an oracle. It is a program, and sometimes another model, with failure modes of its own.

The answers “1/2” and “0.5” are equal, yet a string-matching checker may disagree. A recent audit found that four configurations of widely used math verifiers accept anywhere between 53.8 % and 95.2 % of rewrites of the reference answer that are mathematically identical to it [6]. In code, an iterative attacker found a known-incorrect patch that passes on 28.5 % of sampled SWE-bench Verified tasks [7], and LLM judges can be made to approve a response that consists of nothing but a colon [8]. Since RL optimizes for what the verifier rewards rather than what its designer intended, verifier errors become part of the policy behavior. The problem grows as RLVR is extended from mathematics and code into medicine, science, and open-ended writing, where verification relies on rubrics and LLM judges whose errors include misspecified criteria as well as wrong verdicts [9], [10].

The literature on this problem has grown rapidly, yet it is scattered across communities that rarely cite one another: theoretical analyses of noisy policy gradients, audits of verification software, empirical studies of reward hacking, and work on label-free training, in which the model grades itself. It also appears contradictory, as some studies conclude that an imperfect verifier is good enough [11], [12], while others find that noisy data is destructive [13] or that certain errors cause complete collapse [14]; as we show, most of these results reconcile once the type of error is taken into account.

Three surveys are closest to ours. Srivastava and Aggarwal [5] review RL algorithms for LLMs and present RLVR as effectively eliminating reward hacking, since a policy cannot trick a compiler or a proof checker. Wang et al. [15] explain reward hacking across RLHF, RLAIF and RLVR through proxy compression, and locate the weakness of RLVR in rewarding answers without checking the reasoning behind them. Gu et al. [16] catalog the biases of LLM judges and attacks on them, including one-token attacks [8], as threats to evaluation rather than as rewards a policy learns to trigger. None of them measures how often verifiers err or in which direction, or how errors that the policy does not exploit affect training. We adopt the distinction between random and systematic errors from a single controlled study [14], apply it across 58 studies, including natural verifiers and label-free rewards, and connect it to the J-scaling results [17]–[19]. This yields findings that no single study reports: what decides the direction of a verifier’s errors (F2), which errors each mitigation has been tested against (Section VI), and a reporting protocol (Table III).

The key contributions of this paper are as follows:

• A structured review of 58 studies within the scope defined in Section III, from each of which we extracted the verifier, the origin, direction, and structure of its errors, the training setup, the reported effect on training, and any mitigation with its cost.

• A taxonomy (Fig. 1) that organizes the field into verifier errors, training responses, mitigation strategies, and rewards without a trusted reference.

• A cross-study synthesis showing that whether the policy can exploit an error, more than how often the error occurs, decides the training outcome.

• A reporting protocol and a list of open problems intended to make future studies comparable.

## II. BACKGROUND

## A. Reinforcement Learning with Verifiable Rewards

In RLVR [1]–[3], a policy $\pi _ { \theta }$ samples G responses $y _ { 1 } , \ldots , y _ { G }$ to a prompt x, a verifier V checks each against a reference $y ^ { * }$ (an answer key or a test suite) and gives it a reward $r _ { i } = V ( x , y _ { i } , y ^ { * } ) \in \{ 0 , 1 \}$ , and GRPO compares each response with its group as in Eq. 1.

$$
A _ { i } = { \frac { r _ { i } - \operatorname* { m e a n } ( r _ { 1 } , \ldots , r _ { G } ) } { \operatorname { s t d } ( r _ { 1 } , \ldots , r _ { G } ) + \epsilon } }\tag{1}
$$

Dr. GRPO drops the standard deviation and length normalization [54], and DAPO filters out groups in which all responses share one reward [55]. Such groups give no gradient and are common even with a correct verifier [56], and the grouprelative advantage systematically underestimates the advantage of hard prompts and overestimates that of easy ones [57]. Truncation handling also shapes the reward: penalizing truncated responses acts as reward noise [55], and masking them helps at short generation budgets but not at long ones [58]. An idealized analysis shows that GRPO increases the probability of success above that of the reference policy [59]. As the analysis holds for any binary reward, we note that GRPO amplifies whatever the verifier accepts, including false positives.

## B. Verifiers

We group verifiers by what they compare a response with. Answer checkers extract the final answer, normalize it, and test it for equivalence with a reference, using string rules, numeric tolerances, or symbolic comparison [6], [20]. Execution-based checkers run the response, as when a program is tested against unit tests [60] or a proof is checked in Lean [24]. Model-based verifiers are language models that judge a response against a reference, either prompted judges or trained verifiers [36], [61]; process reward models apply the same idea to individual reasoning steps [4], [62]. All three depend on a trusted answer key or test and on a checker. A growing set of rewards drops one or the other, taking the signal from the policy’s own majority vote [41], from a judge scoring a rubric [48], or from the probability the policy assigns to the reference answer [51].

## C. A Model of Verifier Errors

Let $r ^ { * }$ be the reward the designer intends, usually finalanswer correctness. This is already a choice: if $r ^ { * }$ should reflect valid reasoning, a correct answer reached through flawed steps is a false positive [62]. Let r˜ be the reward the verifier emits. A false positive (FP) rewards an incorrect response with rate $\rho ^ { + } = P ( \tilde { r } = 1 | r ^ { * } = 0 )$ , and a false negative (FN) rejects a correct one with rate $\rho ^ { - } = P ( \tilde { r } = 0 \mid r ^ { * } = 1 )$ ). If errors depend only on $r ^ { * }$ , then

$$
\mathbb { E } [ \tilde { r } \mid r ^ { * } ] = \rho ^ { + } + ( 1 - \rho ^ { + } - \rho ^ { - } ) r ^ { * } .\tag{2}
$$

The group baseline removes $\rho ^ { + }$ , so the signal is scaled by $J =$ $1 - \rho ^ { + } - \rho ^ { - }$ , the verifier’s Youden index TPR−FPR [17]. The same factor appears in work on noise-corrected GRPO [18], [19] and, as the determinant of the reward confusion matrix, in earlier RL [63]. It is exact for mean-centered advantages, while the standard deviation in Eq. 1 adds a further factor [19].

Following Egashira et al. [14], an error is random when it is independent of the query and response given $r ^ { * }$ , and systematic when it depends on a property of the response, such as its format or a keyword. A policy can learn to trigger or avoid a systematic error but not a random one. We call a systematic FP exploitable when the policy can produce its trigger often enough to reinforce it. Earlier RL work showed that corrupted rewards need not be unbiased [64], that a nonequivalent proxy can be hacked [65], and that a learned proxy is eventually overoptimized [66]. What is new in RLVR is that the proxy is often treated as exact; PRIME, for instance, calls exact match unhackable [4].

## III. SURVEY METHODOLOGY

## A. Research Questions and Scope

This work addresses the following research questions:

1) RQ1: What errors do the verifiers used in RLVR make, where do these errors come from, and how often do they occur?

2) RQ2: How do verifier errors affect RLVR training, and what determines whether their effect is benign or harmful?

3) RQ3: Which methods make RLVR robust to verifier errors, what do they cost, and against which kinds of error have they been shown to work?

4) RQ4: How do these findings carry over to rewards that lack a trusted reference answer or a checker, namely self-generated, rubric- and judge-based, and verifier-free rewards?

We cover work that trains or evaluates an LLM with an automatically computed reward and measures, analyzes, mitigates or exploits errors in that reward. Learned preference models from RL with human feedback are outside our scope unless they shed light on verifier errors.

## B. Search and Selection

We searched arXiv, Google Scholar, and OpenReview for studies published up to September 2026. Keyword searches covered RLVR training with GRPO, noise-robust training and verifier-error correction, spurious rewards, vulnerabilities of LLM judges and reward models, reward hacking of unit tests, verifier models, and label-free and verifier-free training. We also screened the reference lists of four seed studies [6], [11], [14], [27]. Results were screened on title and abstract, and retained candidates were read in full; those within the scope above were included. This yielded 58 studies, from each of which we recorded the fields listed in the Introduction. A further 23 works, including three related surveys [5], [15], [16], are cited as background only. At the time of writing, most included preprints had been accepted at peer-reviewed venues but were not yet published in their respective proceedings.

![](images/14c719fa6d8810c0e905b621efefeb4348f1bcfd011f028c4bef63de66900c2b.jpg)  
Fig. 1. Taxonomy of the surveyed literature on verifier noise in RLVR, organized by research question. Each leaf lists representative studies.

![](images/d7d566653ea50832b9c734c4103a847fc5c3bd4ac81a285f9d494d70be509822.jpg)  
Fig. 2. Where errors enter the RLVR loop: (1) wrong or incomplete reference labels, (2) answer extraction and verifier configuration, (3) the checker, test suite, or judge itself, and (4) access that lets the policy alter the verifier or its environment.

## IV. WHAT ERRORS VERIFIERS MAKE (RQ1)

Verifier errors enter the training loop at four points (Fig. 2). Error direction depends more on what the checker actually tests and on who built it than on the family to which it belongs.

## A. Answer Checkers and Reference Labels

Rule-based answer checkers fail mostly by rejecting correct answers. On mathematics, they reach a precision of at least 0.99, but their average recall is 0.86 to 0.93, and it falls to

0.35 to 0.58 on general-science questions [20]. In an audit of 226K rejected responses, 38.5 % were in fact correct, most often because of differences in language, formatting, or notation [21]. On MATH-500, a rule checker produced 10.4 % false negatives and no false positives [18].

Many of these errors come from configuration. Four configurations accepted between 53.8 % and 95.2 % of the same certified-equivalent rewrites, two configurations of one library disagreed on 49.9 % of them, and whitespace and punctuation caused 93.0 % of one configuration’s in-contract failures [6]. The same audit found a deterministic false positive: a cascade that accepts off-by-one answers whenever the reference is 10<sup>4</sup> or larger. Answer extraction alone can shift MATH scores by up to 10 points [1]. Failures are also correlated within a group, with an intraclass correlation of 0.53 across eight rollouts, although shared prompt difficulty may explain part of this [27].

The reference can be wrong. When one study built noisy labels from a model’s wrong answers, 16.4 % proved correct because gold answers were incomplete or equivalence checks failed, and 62 % of a Text2SQL sample had annotation errors [13]. Wrong references affect 11 % of HumanEval solutions [60] and about 20 % of PRM800K labels [3].

## B. Execution-Based Checkers

Tests fail in the other direction. Expanding HumanEval’s tests 80-fold lowers pass@k by up to 19.3–28.9 % [60], and the original MBPP tests accept 15.30 % of base-model samples that extended tests reject, though on inspection only about half of such false positives are wrong [22]. In agentic code environments, an iterative attacker finds a known-incorrect patch that passes on 28.5 % of sampled SWE-bench Verified tasks (18.4 % in a single attempt) and at least 25.0 % of R2E-Gym tasks [7]. Who writes the checker matters: LLM-written repair tests were defective 61.9 % of the time and rejected the reference patch, trading false positives for false negatives [7].

Deterministic checkers of a proxy property can be gamed. On an inductive-logic benchmark whose checker tests only the predicted labels, RLVR-trained models enumerate labels instead of inducing rules, on up to 74 % of hard tasks for one model [23], and script-based graders for agentic tool use see exploit rates between 0 % and 22.3 % across 13 frontier models [38]. Formal proofs are no exception, since the Lean kernel certifies only that a proof establishes the formal statement, not that the statement encodes the intended problem [24]. An audit of five benchmarks reports 4,833 findings (398 mechanically proven), from vacuous statements and harness loopholes that inflate scores to unprovable statements that deflate them [24], and a pipeline that autoformalizes answers before proving them reaches 84.5 % precision [45].

## C. Model-Based Verifiers

Model-based verifiers lean towards accepting. On MATH-500, an LLM verifier produced 33.6 % false positives and 16.2 % false negatives [18], and samples on which verifiers disagreed were mostly incorrect, suggesting that judges overaccept [61]. Specialized verifiers are more accurate than general LLMs but miss valid variants [67], and on hard items, general LLMs lose about 20 points of accuracy against 8 for the best systems [32]. The lean is not universal: on calibration sets with synthetically corrupted negatives, AceMath-7B-RM has a false-positive rate of only 2.0 % but rejects 39.2 % of correct responses [19].

These false positives are often exploitable. Single symbols or generic reasoning openers raise average false-positive rates to 80.6 % for Llama-3-70B, 66.8 % for Qwen2.5-72B, 14.6 % for GPT-4o, and 6.0 % for GPT-o1, with worst cases up to 95.1 %, and larger judges are not reliably safer [8]. Unlike earlier universal phrases found by an external attacker [68], tokens found by searching a judge’s own output distribution push false-positive rates above 90 % on 22 of 24 judge and dataset pairs, though this is an upper bound: realistic appended tokens reach 27.5 % to 58 % on Llama-based judges and at most 23 % on Qwen- and Gemma-based ones [25]. Simple patterns fool generative verifiers, fine-tuned or not, in up to 77.9 % of cases, whereas discriminative verifiers stay robust [20]. Process reward models reward surface form over logical validity: when a policy is trained against one with GRPO, 43 % of its reward gain vanishes once its responses are rephrased [26]. Judge errors are also correlated, as nine frontier judges carry about two effective votes [28].

## D. From Static Audits to Training

Almost every rate above is based on a fixed set of outputs, whereas during training, the verifier faces the policy’s shifting outputs. The true negative rate of a reference-free verifier falls from 0.68 to 0.17 as the generator gets stronger, and its false negatives concentrate on hard problems (49.0 % of correct responses in the hardest quartile against 0.6 % in the easiest) [29]. An offline process reward model loses accuracy as RL proceeds [4]. In theory, the accuracy reachable with an imperfect verifier is capped by its ROC curve near zero false positives, which low-compute measurements cannot predict [69]. Static audits may therefore understate how often a policy yields exploitable false positives for a verifier, although this has not been measured directly in RL.

## V. HOW VERIFIER ERRORS AFFECT TRAINING (RQ2)

Training outcomes depend less on how often a verifier errs than on what kind of error it makes. Random errors mostly slow learning; errors the policy can learn to trigger produce plateaus or collapse; and systematic errors tied to particular inputs, which the policy cannot trigger, gradually erode accuracy, or stay bounded (Table I).

## A. Random Errors

Under random errors, the sign of J from Section II-C sets the direction of learning. A mean-field analysis predicts learning at a rate scaled by J when $J > 0 ,$ , drift at J = 0, and collapse when J < 0. Experiments on code generation agree, although false positives cost more than false negatives at equal J [17]. Policies can tolerate a great deal of such noise, even when it is applied per question rather than per response, thus correlated within a group. Qwen2.5-7B still climbs from 4.82 % to 72.02 % on MATH-500 when the rewards of 40 % of questions are flipped, against 75.85 % without noise [12], and group-level flips of up to 15 % keep the best validation reward within 2 points of the clean run [11].

This tolerance has limits. It depends on the policy: Llama-3.1-8B collapses at 30 % flips [12], although J = 0.4 predicts only slower learning, and multimodal backbones differ widely in how much label corruption they withstand [70]. It depends on granularity: flipping just 10 % of individual samples, not whole groups, costs 3.0 to 4.7 points [11]. And it depends on the training setup. With fixed epochs, a KL penalty, and no ratio clipping, i.i.d. flips cost 0.2 to 30.7 points, but a short ablation restoring clipping raised the noisy baseline by about 6 points, so part of this cost may reflect the setup [19].

## B. Exploitable False Positives

Whether a systematic false positive is exploitable depends on whether the policy can reach its trigger. In controlled arithmetic tasks built with Reasoning Gym [71], the drop in pass@1 follows the trigger’s conditional advantage times its initial frequency [14]. A false positive for “python” drives pass@1 from 85.6 % to 0.6 %, one for the opener “Certainly” plateaus at 64.3 %, and one that fires only half the time is not exploited, as its conditional advantage is too low.

Natural verifiers behave likewise once the policy finds the trigger. With a judge susceptible to reasoning openers, 94.26 % of outputs became “Thought Process:” [8]. Against a binary judge, 5 of 10 seeds collapsed on MATH, mostly onto empty outputs locked in within about 170 steps [25]. A fine-tuned generative verifier was hacked but a discriminative one was not [20], process reward models used as outcome rewards were hacked [37], one of them despite resisting gradient attacks [26], and even a deterministic logic checker let proxy and true reward drift apart from about step 250 [23].

TABLE I  
TRAINING OUTCOMES UNDER REFERENCE-BASED VERIFIER ERRORS
<table><tr><td>Study</td><td>Reward signal and error</td><td>Outcome</td><td>Key result</td></tr><tr><td>Egashira [14]</td><td>Rule checker; injected random FP/FN, systematic FN, exploitable FP</td><td>Random: delay (lower endpoint for OLMo at 50 % FNR); exploitable FP: plateau or collapse</td><td>Pass@1 0.6 with a &quot;python&quot; trigger vs. 85.6 clean</td></tr><tr><td>Rad [17]</td><td>Unit tests; random flips across J</td><td>J &gt; 0: delay or lower endpoint; J = 0: no learning; J &lt; 0: collapse</td><td>At J = 0.3: 16.0 FN-heavy vs. 14.6 FP-heavy (clean 20.8)</td></tr><tr><td>El Mansouri [19]</td><td>Rule and reward models; i.i.d. flips; no ratio clipping</td><td>Lower endpoint, strongly model-dependent</td><td>Losses of 0.2–30.7 points</td></tr><tr><td>Plesner [11]</td><td>Unit tests; symmetric flips; small model</td><td>Group-level ≤15 %: graceful; sample-level: lower endpoint; 4B verifier: plateau</td><td>Best reward 0.704 vs. 0.871 with a 30B verifier</td></tr><tr><td>Lv [12]</td><td>verifier Rule; question-wise flips; PPO</td><td>≤40 %: graceful or delay; 50 %: no learning; Llama-3.1-8B: collapse at 30 %</td><td>72.02 at 40 % flips vs. 75.85 clean</td></tr><tr><td>Cai [18]</td><td>Rule; class-conditional and format-dependent noise</td><td>Uncorrected: lower endpoint and loss of general</td><td>ARC-C 73.9 → 43.9 (Llama-3.2-3B)</td></tr><tr><td>Xu [21]</td><td>Rule; natural FN</td><td>capability Delay, then lower endpoint</td><td>51.82 vs. 54.46 with FNs repaired</td></tr><tr><td>Zhu [13]</td><td>Model-aligned wrong labels (the model&#x27;s</td><td>Lower endpoint, monotone in noise rate</td><td>-9.0 on MATH-500 at 100 % noise</td></tr><tr><td>Zhang [22]</td><td>own wrong answers) Leaky unit tests; natural FP</td><td>Proxy inflated; held-out accuracy unaffected</td><td>+8.37 training reward, 0.20 held-out</td></tr><tr><td>Huang [20]</td><td>Rule vs. fine-tuned generative judge</td><td>Rule: lower endpoint; judge: divergence</td><td>Reward leaves oracle near step 450</td></tr><tr><td>Agrawal [37]</td><td>PRM as outcome reward</td><td>Divergence, then collapse at 1.5B</td><td>GSM8K 71.11 → 31.16</td></tr><tr><td>Zhao [8]</td><td>LLM judge; master-key FP</td><td>Collapse</td><td>94.26 % of outputs become &quot;Thought Process:&quot;</td></tr><tr><td>Li [25]</td><td>Binary judge; FP discovered by the policy</td><td>Collapse in 5 of 10 seeds (MATH)</td><td>88 % of false positives are empty outputs</td></tr><tr><td>Helff [23] Khalifa [31]</td><td>Extensional rule checker; exploitable FP</td><td>Plateau from about step 250 SFT-primed: collapse (2 of 6 hack); RL only:</td><td>Proxy-oracle gap of 3.5/10 at step 500 Hacking peaks at 80–90 % of rollouts</td></tr><tr><td></td><td>Test harness the policy can overwrite</td><td>plateau (2 of 8 hack)</td><td></td></tr><tr><td>Wu [30]</td><td>Evaluator the policy can overwrite</td><td>Rebound, then collapse</td><td>Hack rate 99.9 %, pass@1 1.2 % (Phi-4-mini)</td></tr></table>

A plateau means the true reward stalls while the proxy rises; divergence means the true reward declines while the proxy rises; a rebound means hacking fails, retreats, and then succeeds.

When the policy can modify the verifier, hacking can rebound: early attempts fail, the policy retreats, and hacking then succeeds, most readily when legitimate reward is scarce [30]. Hacking can also be inherited. After distillation on traces, only 1.2 % of which contained hacking, two of the six primed models (both Qwen) began hacking within 100 RL steps, and their true reward collapsed; the rest, including Llama-3.1-8B, hacked little or not at all. Without distillation, only two of eight models learned to hack, and their true reward plateaued [31].

## C. Errors the Policy Cannot Exploit

Systematic errors that the policy cannot turn into a reward behave much like noise. Leaky MBPP tests inflated training reward without changing held-out accuracy, and their falsepositive mass did not grow over 800 steps [22]. Model-aligned wrong labels, i.e., wrong reference answers drawn from the model’s own errors, cost accuracy monotonically with the noise rate, from 1.5 points at 10 % noise to 9.8 at 100 % averaged over five math benchmarks, and push pass@k below the base model for k > 1 [13]. Judge accuracy sets the direction coarsely, as judges at or below 81 % accuracy pushed a policy below its starting point while judges at or above 92 % improved it [32], while uncorrected noise can erode capability outside the task [18].

False negatives remove learning signal rather than reward wrong behavior, but at a cost: natural-rule false negatives slow early learning and lower the endpoint [21], and those spanning a domain produce a plateau [14]. Policies learn around them, answering in Chinese to escape a checker rejecting English [14] or appending \boxed{} to pass Math-Verify [36].

## D. The Spurious-Reward Debate

Training Qwen2.5-Math-7B with random rewards improved MATH-500 accuracy by 21.4 points, nearly matching the 29.1 points gain from ground-truth rewards [33]. These gains did not transfer to other model families: Llama and OLMo2 improved little or degraded. They also largely vanished on AIME 2025. Three explanations compete: clipping that amplifies the model’s prior for code-style reasoning [33]; contamination, since from 60 % of each problem Qwen2.5-Math-7B reconstructs 54.60 % of MATH-500 against 3.80 % for Llama3.1- 8B and only correct rewards give stable gains on a clean benchmark [34]; and model strength, since the clipping bias is too small to carry a signal and gains appear on non-Qwen models too [35]. The studies differ in prompts, clipping, and models, so the question is open, but they agree that a strong prior can override the J analysis and that benchmarks older than the model can make any reward look useful.

## E. What Decides the Outcome

Three factors decide how training responds. The structure of the error sets the regime. The policy’s prior decides whether an exploitable error is found, since a policy too weak to produce a trigger cannot exploit it [20] and Llama-3.1-8B resisted hacking even after priming, unlike Qwen models of similar size [31]. The budget, regularization, and the supply of legitimate reward decide whether a delay becomes a loss [17], [19] and whether hacking rebounds [30].

## VI. MITIGATING VERIFIER ERRORS (RQ3)

Mitigations act on the factors identified in Section V: some correct the gradient for a known error rate, some replace or back up the verifier, some repair the data or the environment, and some detect or penalize hacking as it emerges. Almost all have been validated against a single error regime (Table II), and defenses against exploitable false positives have been tested inside RL in only seven studies [14], [20], [23], [25], [30], [31], [37], each on one task or a few models.

TABLE II  
MITIGATION FAMILIES AND THE ERRORS THEY WERE TESTED ON
<table><tr><td>Family</td><td>Tested against</td><td>Cost or gap</td></tr><tr><td>Noise-aware gradients [18], [19]</td><td>Class-conditional noise; format FN; real reward</td><td>Needs error rates; fails on model-aligned wrong labels [13]</td></tr><tr><td>Noise-robust</td><td>models Random label corruption</td><td>Backbone-dependent; extra</td></tr><tr><td>regularization [70]</td><td>Stronger verifiers [20], Natural rule FN; judge</td><td>hyperparameters Extra inference; small RL gains;</td></tr><tr><td>[21], [36], [61] Judge hardening [8], [25]</td><td>hacking Discovered and</td><td>own FP rarely reported One policy-judge pair; offline only in [8]</td></tr><tr><td>Structural checks [23], [37]</td><td>master-key FP Enumeration shortcut; PRM hacking</td><td>Task-specific; outcome-verifier errors untouched</td></tr><tr><td>Data audits [13], [22], [31]</td><td>Wrong labels; leaky</td><td>Covers only what is audited</td></tr><tr><td>Environment hardening [7], [24],</td><td>tests; inherited hacking Grader and harness exploits</td><td>Per-environment effort; mostly evaluation-time</td></tr><tr><td>[38] Oracle access [14]</td><td></td><td>Oracle on 10–50 % of steps;</td></tr><tr><td></td><td>Plateau and collapse</td><td>trigger persists</td></tr><tr><td>Detection, representation</td><td>Hacking onset; test rewriting</td><td>Detects rather than prevents; white-box; one exploit</td></tr><tr><td>control [30], [39] Judge aggregation [40]</td><td>Correlated judges; master Not tested in RL; small gains</td><td></td></tr></table>

## A. Correcting the Gradient

If the error rates are known, the gradient can be corrected. A backward correction needs both rates and becomes noisy as J shrinks, whereas a forward correction needs only the false-negative rate, which can be estimated online by sending about 10 % of rule rejections to a small verifier; under formatdependent false negatives, it reaches 31.2 against 20.4 without correction [18]. With real verifiers, the gains are smaller, 1.2 to 2.8 points over a rule reward [18] and up to 6.7 points on math with reward models whose flip rates were estimated on a labeled calibration split [19]. Both corrections are applied after group normalization, which would otherwise cancel them, and both assume class-conditional noise. On wrong labels that follow the model’s own errors, the forward correction failed even when given the true noise rates [13], and neither method has been tested against an exploitable false positive.

## B. Stronger Verifiers and Structural Checks

The most common remedy is a better verifier, and offline accuracy buys less than one might expect. TinyV consults a 1.5B verifier only when the rule checker rejects, adding 2.6 to 2.8 points at 6 % extra step time, although its own falsepositive rate is unreported [21]. xVerify, far more accurate than Math-Verify as a classifier, improves on it by only 0.8 points as a reward [36], and a hybrid rule-then-model verifier adds 2.3 points for 25.7 % more training time [20].

Hardening targets exploitable false positives directly. Finetuning a judge with LoRA on discovered tokens cut mean false positives during GRPO from 655 to 42 over ten seeds (median 168 to 40) and prevented all eight false-positive collapses across two datasets, yet tokens found by a fresh discovery run still reach false-positive rates of 8.75 to 31 % against the hardened judges [25]. Training on truncated responses labeled incorrectly brings master-key false positives to 0.1 %, but has been tested only offline [8]. Structural checks close specific loopholes: an isomorphic verifier keeps the proxy–oracle gap near zero in inductive logic [23], and consulting a PRM only when every rollout fails avoids the hacking seen when the PRM is the reward [37].

## C. Data, Environments and Oracles

Several errors are cheaper to remove at their source than to correct in the gradient. Re-verifying labels was the only remedy for wrong references [13]. A static audit of 16 samples per task predicts where leaky tests will reward wrong code, and repairing the 18 worst suites removes about half of that falsepositive mass [22]. Filtering distillation data on true rather than proxy reward removes inherited hacking [31]. Hardening agentic graders cut exploits from 6.5 % to 0.8 % with task success unchanged [38], and a Docker-gated loop repaired 9 of 11 hackable code tasks for \$3.60 [7]. Access to a trusted oracle helps but does not cure: alternating with an oracle every 10 steps turns plateaus into delays; preventing collapse requires the oracle every 2 steps; and even then, the policy keeps producing the trigger [14].

## D. Detecting and Suppressing Hacking

Detection tells practitioners when hacking has begun, but does not stop it. Comparing ten trajectories raises an LLM detector’s macro-F1 from 0.45 to 0.63 on synthetic codingagent hacks, but semantic hacks stay at 0.0 to 0.4 [39]. Probes on internal activations can flag hacking learned through finetuning [72], and acting on the policy’s own representations goes further. A shortcut direction extracted from contrastive sentence pairs tracks hacking during training, and discounting the advantage of rollouts that score high on it cuts Phi-4-mini’s hack rate from 99.9 % to 24.9 % and lifts LeetCode pass@1 from 1.2 % to 12.0 %, whereas suppressing the same direction at generation time only lowers the hack rate to 72.3 % [30]. For correlated judges, confounder-aware aggregation beats majority voting on most benchmarks and neutralizes two master keys [40], but in another study, aggregation closed at most 11 % of the gap to independent voting [28], and neither has been tested inside RL.

No mitigation has been shown to handle random, systematic, and exploitable errors simultaneously, and each defense against exploitable false positives rests on a single policy, judge, or task. Section VII asks how far these remedies carry when the reference itself is no longer trusted.

## VII. REWARDS WITHOUT A TRUSTED REFERENCE (RQ4)

When the reference comes from the policy, a rubric, or a judge, or when no checker is used, the earlier findings carry over with one change: the reward can no longer be checked against an independent source, so errors that such a check would expose become systematic by construction.

## A. Self-Generated Rewards

When the group’s majority answer serves as the label [41], label error and reward error come apart. At the start of training, only 37 % of majority labels were correct, yet 92 % of rewards were, because wrong answers rarely match a wrong majority [41]. The majority itself is wrong on 25.85 % of MATH-500 and 73.33 % of AIME 2024 questions in one study [46]. The regime decides the outcome. On small or testtime sets, such training helps, lifting Qwen2.5-Math-7B from 12.9 to 40.2 on AIME 2024 [41]. At the training-corpus scale, every intrinsic reward studied sharpens the initial distribution and eventually collapses, even in the most stable settings, by roughly 1,000 steps; in one majority-vote run, reward accuracy fell from about 0.80 to 0.48 within about 260 steps while the proxy reward kept rising [42]. Majority-vote baselines collapsed in several studies [47], [73], [74], in one case with rollout accuracy falling from 44.0 % to 2.4 % while training reward rose [44].

Damage can also hide in aggregates. Pooled over three models, 21.6 % of problems were lost and only 0.7 % newly learned while pass@1 stayed flat or rose [43], and about two-thirds of one audio gain came from a single phoneme template [75]. Coupling the reward to the online policy is no clear fix: a frozen self-certainty scorer was exploited near step 100 while the online one held for 150 steps [76], but selfcertainty collapsed in longer runs elsewhere [42], [74].

Sparse external verification closes most of the gap. A Lean proof gate on the majority answer matched ground-truth reward on nine-benchmark averages for three models (e.g., 35.40 vs. 35.56) [45], labels bought for 20 % of problems prevented collapse when targeted but not when random [47], and soft vote weights from a tool verifier with only 25 to 39 % precision beat majority-vote training on all 15 model and benchmark pairs, whereas hard filtering on the same verifier hurt [46]. Stabilizers without any external signal delay collapse [44], [73], [74], [77], [78] but cannot correct a consensus that is systematically wrong [78].

## B. Rubric and Judge Rewards

Rubric judges, like answer judges, fail mostly by accepting. Against a panel of frontier models, a weak judge had 10.3 % false positives and 6.8 % false negatives on medical rubrics and 19.8 % and 4.4 % on science [9], and on the harder half of disputed rubric items, balanced accuracy spans 55.65 % to 84.81 % across 18 judges [10]. The rubric adds an error of its own, since rubrics written without reference answers train worse policies (32.0 vs. 35.9 on HealthBench) [48]. In training, the share of newly credited criteria that were wrong grew from 39 % to 65 % under a weak medical judge. A strong judge showed no such trend, yet rubric-free judges still preferred the base model on 78.4 % of prompts, so a better verifier does not fix a misspecified rubric [9].

Exploitation again depends on discoverability. Injected judge biases were exploited after 68 to 478 steps, with less task-entangled biases exploited later, and two of the eight were never exploited [49]. Non-reasoning judges were hacked at every size, while reasoning judges moved the hacking to the gold judge, which the policy’s outputs also fooled, reaching an 89.6 % win rate on Arena-Hard-V2 creative writing [50]. Negated rubrics drive scores from 23.4 to 2.9, while incom plete ones degrade gracefully [79], a split that, in our reading, mirrors J < 0 versus a smaller positive J. Decomposing the reward and auditing training runs both help: a checklist judge is less accurate than specialized reward models on RewardBench yet trains better policies [80], and a judge-blind auditing agent flagged all six hacked runs [49].

## C. Verifier-Free Rewards

Verifier-free methods drop the checker and reward the policy’s own probability of the reference answer. VeriFree does so after the policy’s reasoning, which equals exact-match RLVR in expectation when there is a single correct answer [51]. In our reading, it thereby trades the false positives of model verifiers for the false negatives of exact match, and adding equivalent answers helped only slightly [51]. RLPR rewards the mean token probability instead and reaches a ROC-AUC of about 0.90 against human labels on general data, against 0.61 for a rule verifier and 0.69 for a model verifier [52]. Probability rewards still under-credit correct paraphrases [53], and the same method scores differently across papers that use different evaluation graders [52], [53], so the grader must be reported with the result.

For the rubric and judge rewards, verifier error and specification error must be separated, because a better judge fixes only the first. Section VIII returns to this distinction.

## VIII. DISCUSSION

The four research questions point to one conclusion: the harm a verifier error does is set less by its rate than by whether the policy can find and reproduce it, and the evidence for this is strong in a narrow set of settings and thin outside them.

## A. Cross-Cutting Findings

F1. Structure decides the regime; rate decides the cost within it. Exploitable false positives produce plateaus or collapse [8], [14], [23], [25], [26], random errors mostly delay learning [11], [12], [17], and systematic errors the policy cannot trigger degrade accuracy gradually or stay bounded [13], [22]. Earlier RL theory points the same way, since a few corrupted states can cause near-worst-case regret [64]. Preferencebased RL shows a similar pattern: feature-dependent noise significantly degrades a state-of-the-art noise-robust method in some settings, where a method without explicit denoising often does better [81]. The finding carries a qualifier, since under a fixed budget and a KL penalty, random errors also lower the endpoint, by an amount that depends strongly on the model [19].

F2. Error direction depends on what the checker checks and who built it. Answer-matching rules mostly reject [20], [21] and generative judges mostly accept [18], [61], but rules that test a proxy property accept [23], tests written by a model mostly reject [7], and a reward model can be dominated by false negatives [19]. Because false positives cost more than false negatives at equal J [17], choosing rule rewards for their resistance to hacking [2] is sound, but a move from rules to judges trades a benign error for a dangerous one only when the judge’s false positives can be triggered.

F3. Verifier error is non-stationary, and static accuracy is insufficient. Learned and self-generated verifiers lose accuracy as the policy changes [4], [42], and shortcut use rises with reasoning effort and longer RL [23]. For reference-based checkers, this is a hypothesis, backed mainly by a referencefree verifier outside RL [29]. Static accuracy predicts training direction only coarsely [32]: a PRM robust to gradient attacks was hacked in RL [26], a far more accurate verifier added only 0.8 points as reward [36], and a less accurate checklist judge trained better policies than specialized reward models [80].

F4. The evidence base is narrow and partly confounded. Of the 46 studies that train a policy, 44 use a Qwen model, trained policies rarely exceed 32B parameters, and most tasks are mathematical. Several central results rest on a single seed or a single policy and judge pair [9], [19], [25], [49], and contamination of Qwen2.5-Math on MATH-500 and AIME 2024 [34] threatens every result obtained with that model on those benchmarks. Training studies of agentic verification do not yet exist; the evidence there comes from audits [7], [38].

F5. Self-generated rewards are bounded by the prior and rescued by sparse verification. Without an external check they sharpen and collapse at training-corpus scale [42], though they can help on small or test-time sets [41], and their damage can hide in aggregate metrics [43], [75]. A small amount of external verification recovers most of the gap [45]–[47].

## B. Resolved and Open Disagreements

Most apparent contradictions dissolve once the type of noise, the regime, or the budget is controlled. The claim that an imperfect verifier is good enough [11], [12] and the finding that noisy data is destructive [13] concern different kinds of noise: random flips versus wrong labels that follow the model’s own errors. The apparent collapse at 50 % flips [12] is a failure to learn, as predicted at J = 0 [17]. Large losses under random noise [19] need not contradict the mean-field result, whose recovery of the clean endpoint holds without regularization [17]: that study used a fixed budget, a KL penalty and no ratio clipping, and under a KL penalty the dynamics settle at an interior equilibrium that depends on J [17], so longer training alone need not close the gap. Noiseaware corrections work on class-conditional noise [18], [19] and fail on question-dependent labels [13], and within-group correlation of rule failures [27] may reflect prompt difficulty rather than refute the independence those corrections assume.

Four disagreements remain open. The mechanism behind spurious-reward gains is contested [33]–[35]. Online selfcertainty was stable over short runs [76] but collapsed over longer ones [42], [74]. Correlation-aware judge aggregation helped in one study [40] and closed a little of the gap in another [28], and neither was tested in RL. Practitioners are advised both to prioritize precision [11] and to maximize J rather than precision alone [17], while a verifier with 25 to 39 % precision still helped as a soft vote weight [46].

TABLE III RECOMMENDED REPORTING PROTOCOL
<table><tr><td>Theme</td><td>Report</td><td>Motivation</td></tr><tr><td>Verifier</td><td>Identity, version, extraction configuration and coverage; judging granularity and output cap; the evaluation grader, separately</td><td>[6], [10], [52], [67]</td></tr><tr><td>Verifier error</td><td>FPR and FNR, or Î with an interval, on the policy&#x27;s own outputs at the start and end of training; noise structure, with FP and FN tested separately; adversarial probes of model verifiers;</td><td>[14], [17], [20], [61]</td></tr><tr><td>Training</td><td>invalid-response rate An oracle reward beside the training reward; final and best checkpoints; pass@k with pass@1; seeds</td><td>[13], [14], [35]</td></tr><tr><td>Evaluation</td><td>or variance; no reliance on entropy as a health signal More than one model family; random- and format-reward baselines; post-cutoff benchmarks and [43], [75] a contamination probe; per-problem and per-category</td><td>[33], [34],</td></tr><tr><td>Label-free</td><td>breakdowns Reward accuracy against gold, not only pseudo-label</td><td>[41], [42]</td></tr><tr><td>rewards Data and release</td><td>accuracy Audited reference labels; distillation data filtered on true reward; released verdicts and code [31]</td><td>[13], [27],</td></tr></table>

## C. A Recommended Reporting Protocol

Results are hard to compare because studies rarely report the verifier in enough detail to reproduce its errors. Table III lists recommended items, each motivated by a corpus result.

## D. Open Problems

The largest gap is theoretical. The mean-field analysis of random errors [17] has no counterpart for response-dependent, exploitable errors under group-relative gradients, although corrupted-reward theory offers a starting point [64], and, as noted in Section II, GRPO amplifies whatever the verifier accepts, false positives included. Empirically, verifier error should be measured on the policy’s own outputs as training proceeds [4], [29], the interaction between correlated errors and group size remains unexplored [27], [28], and defenses validated offline or on a single policy and judge pair need testing inside RL [8], [25], [40]. Three further problems follow from the corpus: separating verifier error from rubric misspecification [9], [48], guarding against hacking of the oracle itself [50], and preventing hacking inherited through distillation data [31]. Agentic and long-horizon verification has been audited [7], [38] but not yet studied in training.

## E. Limitations

Our synthesis has two limitations. First, studies differ in models, budgets, and reference standards, so we compare directions and regimes, not effect sizes. Second, the search, based on keywords and four seed studies [6], [11], [14], [27], covered English-language work up to September 2026 and may miss work framed in other terms.

## IX. CONCLUSION AND FUTURE WORK

This work surveyed 58 studies of verifier errors in RLVR. The evidence shows that the harm an error causes depends less on its rate than on whether the policy can find and reproduce it: exploitable false positives cause plateaus or collapse, random errors mostly slow learning, and errors that the policy cannot trigger degrade accuracy gradually or stay bounded. Whether a verifier errs by rejecting or accepting depends on what it checks and who built it, and its errors shift as the policy improves, so accuracy measured before training says little about robustness during it. Most mitigations have been tested against a single kind of error, and self-generated rewards remain bounded by the model’s prior unless some external verification is added. We therefore argue for treating the verifier as a component to be measured on the policy’s own outputs, probed adversarially, and reported in full. Open directions include a theory of exploitable errors, defenses validated inside RL across several policies and judges, and training studies of agentic verification.

## DATA AVAILABILITY

A list of all 81 cited works, with the 58 reviewed studies organized by research question and a link to each paper, is openly available under a CC BY 4.0 license at https://github. com/reasoning-ai-lab/Checking-the-Checkers-RLVR.

## REFERENCES

[1] N. Lambert et al., “Tulu 3: Pushing frontiers in open language model¨ post-training,” in Second Conference on Language Modeling, 2025.

[2] D. Guo, D. Yang, H. Zhang et al., “DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning,” Nature, vol. 645, no. 8081, pp. 633–638, 2025.

[3] Z. Shao et al., “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[4] G. Cui et al., “Process reinforcement through implicit rewards,” arXiv preprint arXiv:2502.01456, 2025.

[5] S. S. Srivastava and V. Aggarwal, “A technical survey of reinforcement learning techniques for large language models,” ACM Transactions on Intelligent Systems and Technology, 2026.

[6] E. Xin, “Where the verifier fails: A category-level audit of reward signals in RLVR,” arXiv preprint arXiv:2609.01354, 2026.

[7] S. Rajan, “Auditing reward hackability in code RL training environments,” arXiv preprint arXiv:2606.16062, 2026.

[8] Y. Zhao et al., “One token to fool LLM-as-a-judge,” arXiv preprint arXiv:2507.08794, 2026.

[9] A. Mahmoud, M. Rezaei, Z. Wang, A. Gunjal, B. Liu, and Y. He, “Reward hacking in rubric-based reinforcement learning,” arXiv preprint arXiv:2605.12474, 2026.

[10] T. Pan et al., “RubricEval: A rubric-level meta-evaluation benchmark for LLM judges in instruction following,” arXiv preprint arXiv:2603.25133, 2026.

[11] A. Plesner, F. Guzman, and A. Athalye, “An imperfect verifier is good´ enough: Learning with noisy rewards,” arXiv preprint arXiv:2604.07666, 2026.

[12] A. Lv, R. Xie, X. Sun, Z. Kang, and R. Yan, “The climb carves wisdom deeper than the summit: On the noisy rewards in learning to reason,” arXiv preprint arXiv:2505.22653, 2025.

[13] Y. Zhu and D. Kang, “Noisy data is destructive to reinforcement learning with verifiable rewards,” arXiv preprint arXiv:2603.16140, 2026.

[14] K. Egashira, M. Vero, J. Dekoninck, F. E. Dorner, R. Staab, and M. Vechev, “Delay, plateau, or collapse: Evaluating the impact of systematic verification error on RLVR,” arXiv preprint arXiv:2605.02909, 2026.

[15] X. Wang et al., “Reward hacking in the era of large models: Mechanisms, emergent misalignment, challenges,” arXiv preprint arXiv:2604.13602, 2026.

[16] J. Gu et al., “A survey on LLM-as-a-judge,” The Innovation, vol. 7, no. 6, p. 101253, 2026.

[17] A. Rad, K. Filom, D. Keivan, P. Mohajerin Esfahani, and E. Kamalinejad, “Rate or fate? RLV<sup>ε</sup>R: Reinforcement learning with verifiable noisy rewards,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 102 933–102 973.

[18] X.-Q. Cai, W. Wang, F. Liu, T. Liu, G. Niu, and M. Sugiyama, “Reinforcement learning with verifiable yet noisy rewards under imperfect verifiers,” arXiv preprint arXiv:2510.00915, 2026.

[19] O. El Mansouri, F. A. Izzati, M. E. A. Seddik, and S. Lahlou, “Noisecorrected GRPO: From noisy rewards to unbiased gradients,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 86 336–86 356.

[20] Y. Huang, W. Zeng, X. Zeng, Q. Zhu, and J. He, “From accuracy to robustness: A study of rule- and model-based verifiers in mathematical reasoning,” arXiv preprint arXiv:2505.22203, 2026.

[21] Z. Xu et al., “TinyV: Reducing false negatives in verification improves RL for LLM reasoning,” Transactions on Machine Learning Research, 2026.

[22] C. Zhang, “When the reward suite is leaky: A preregistered causal contrast of natural verifier false positives in RLVR,” arXiv preprint arXiv:2607.11022, 2026.

[23] L. Helff et al., “LLMs gaming verifiers: RLVR can lead to reward hacking,” arXiv preprint arXiv:2604.15149, 2026.

[24] P. S. Ammanamanchi, S. Bhat, and S. Biderman, “Faults in our formal benchmarking: Dataset defects and evaluation failures in Lean theorem proving,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 2431–2462.

[25] T.-L. Li, Y. Wu, and H. Liu, “AdvJudge-Zero: Binary decision flips in LLM-as-a-judge via adversarial control tokens,” arXiv preprint arXiv:2512.17375, 2026.

[26] R. Tiwari et al., “Reward under attack: Analyzing the robustness and hackability of process reward models,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 121 355–121 375.

[27] E. Xin, “Are verifier errors independent within a GRPO group? Evidence from Qwen2.5 rollouts,” arXiv preprint arXiv:2609.06386, 2026.

[28] G. Kohli, “Nine judges, two effective votes: Correlated errors undermine LLM evaluation panels,” arXiv preprint arXiv:2605.29800, 2026.

[29] Y. Zhou, A. Xu, Y. Zhou, J. Singh, J. Gui, and S. Joty, “Variation in verification: Understanding verification dynamics in large language models,” in International Conference on Learning Representations, 2026, pp. 144 597–144 642.

[30] R. Wu and R. Tang, “From rebound to remedy: Understanding and mitigating reward hacking via representation engineering,” arXiv preprint arXiv:2604.01476, 2026.

[31] M. Khalifa, Z. Khan, O. Tafveez, H. Peng, and L. Wang, “Countdown-Code: A testbed for studying the emergence and generalization of reward hacking in RLVR,” arXiv preprint arXiv:2603.07084, 2026.

[32] Y. Yan et al., “VerifyBench: Benchmarking reference-based reward systems for large language models,” in International Conference on Learning Representations, 2026, pp. 143 482–143 522.

[33] R. Shao et al., “Spurious rewards: Rethinking training signals in RLVR,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 110 167–110 207.

[34] M. Wu et al., “Reasoning or memorization? Unreliable results of reinforcement learning due to data contamination,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 40, pp. 33 944– 33 952, 2026.

[35] P. Chen, X. Li, Z. Li, W. Yin, X. Chen, and T. Lin, “Exploration vs. exploitation: Rethinking RLVR through clipping, entropy, and spurious reward,” in International Conference on Learning Representations, 2026, pp. 134 506–134 540.

[36] D. Chen et al., “xVerify: Efficient answer verifier for reasoning model evaluations,” arXiv preprint arXiv:2504.10481, 2025.

[37] A. Agrawal, M. Liu, and F. Huang, “VeriGate: Verifier-gated step-level supervision for GRPO,” arXiv preprint arXiv:2605.30451, 2026.

[38] K. Thaman, “Reward hacking benchmark: Measuring exploits in LLM agents with tool use,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 120 372–120 389.

[39] D. G. Deshpande, A. Kannappan, and R. Qian, “Benchmarking reward hack detection in code environments via contrastive analysis,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 24 515–24 536.

[40] J. Zhao, C. Shin, T.-H. Huang, S. S. S. Namburi GNVV, and F. Sala, “CARE: Confounder-aware aggregation for reliable LLM evaluation,” in

Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 162 114–162 150.

[41] Y. Zuo et al., “TTRL: Test-time reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 131 459– 131 483.

[42] B. He et al., “How far can unsupervised RLVR scale LLM training?” in International Conference on Learning Representations, 2026, pp. 14 823–14 865.

[43] H. Lin, Z. Kuai, E. Xue, and L. Wang, “Detecting and mitigating the correct-answer extinction window in test-time reinforcement learning with majority voting,” arXiv preprint arXiv:2605.19444, 2026.

[44] Y. Ye, L. Zhang, Y. Chen, X. Shi, and B. Fu, “Don’t peek at the answer: Outcome-masked group relative policy optimization for labelfree RLVR,” arXiv preprint arXiv:2608.03119, 2026.

[45] X. Chen et al., “JURY-RL: Votes propose, proofs dispose for label-free RLVR,” in 3rd AI for Math Workshop: Toward Self-Evolving Scientific Agents, 2026.

[46] R. Liao et al., “Tool verification for test-time reinforcement learning,” arXiv preprint arXiv:2603.02203, 2026.

[47] L. Wang et al., “When self-belief misleads: Active label acquisition for reinforcement learning with verifiable rewards,” arXiv preprint arXiv:2605.25864, 2026.

[48] A. Gunjal et al., “Rubrics as rewards: Reinforcement learning beyond verifiable domains,” in International Conference on Learning Representations, 2026, pp. 127 924–127 945.

[49] X. Wang, Z. Hao, S. Hou, H. Peng, J. Li, and X. Wang, “Reproducing, analyzing, and detecting reward hacking in rubric-based reinforcement learning,” arXiv preprint arXiv:2606.04923, 2026.

[50] Y. Liu et al., “Examining reasoning LLMs-as-judges in non-verifiable LLM post-training,” arXiv preprint arXiv:2603.12246, 2026.

[51] X. Zhou et al., “Reinforcing general reasoning without verifiers,” in International Conference on Learning Representations, 2026, pp. 69 689– 69 712.

[52] T. Yu et al., “RLPR: Extrapolating RLVR to general domains without verifiers,” arXiv preprint arXiv:2506.18254, 2025.

[53] C. Huang, L. Lin, X. Shi, W. Hu, and R. Tang, “DARL: Encouraging diverse answers for general reasoning without verifiers,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 30 649–30 665.

[54] Z. Liu et al., “Understanding R1-Zero-like training: A critical perspective,” in Second Conference on Language Modeling, 2025.

[55] Q. Yu et al., “DAPO: An open-source LLM reinforcement learning system at scale,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 113 222–113 244.

[56] W. Nie et al., “Gradient starvation in binary-reward GRPO: Why groupmean centering fails and why the simplest fix works,” arXiv preprint arXiv:2605.07689, 2026.

[57] F. Yang et al., “Your group-relative advantage is biased,” arXiv preprint arXiv:2601.08521, 2026.

[58] Z. Liu et al., “AceReason-Nemotron 1.1: Advancing math and code reasoning through SFT and RL synergy,” in International Conference on Learning Representations, 2026, pp. 70 010–70 028.

[59] Y. Mroueh, “Reinforcement learning with verifiable rewards: GRPO’s loss, dynamics, and success amplification,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 90 821–90 847.

[60] J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is your code generated by ChatGPT really correct? Rigorous evaluation of large language models for code generation,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 21 558–21 572.

[61] S. Liu et al., “CompassVerifier: A unified and robust verifier for LLMs evaluation and outcome reward,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 33 466–33 494.

[62] H. Lightman et al., “Let’s verify step by step,” in International Conference on Learning Representations, 2024, pp. 39 578–39 601.

[63] J. Wang, Y. Liu, and B. Li, “Reinforcement learning with perturbed rewards,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 34, no. 4, 2020, pp. 6202–6209.

[64] T. Everitt, V. Krakovna, L. Orseau, and S. Legg, “Reinforcement learning with a corrupted reward channel,” in Proceedings of the 26th International Joint Conference on Artificial Intelligence, 2017, pp. 4705– 4713.

[65] J. Skalse, N. Howe, D. Krasheninnikov, and D. Krueger, “Defining and characterizing reward gaming,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 9460–9471.

[66] L. Gao, J. Schulman, and J. Hilton, “Scaling laws for reward model overoptimization,” in Proceedings of the 40th International Conference on Machine Learning, vol. 202, 2023, pp. 10 835–10 866.

[67] X. Li, X. Li, S. Hu, Y. Guo, and W. Zhang, “VerifyBench: A systematic benchmark for evaluating reasoning verifiers across domains,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 38, pp. 31 796–31 804, 2026.

[68] V. Raina, A. Liusie, and M. Gales, “Is LLM-as-a-judge robust? Investigating universal adversarial attacks on zero-shot LLM assessment,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 7499–7517.

[69] F. E. Dorner, Y. Chen, A. F. Cruz, and F. Yang, “ROC-n-reroll: How verifier imperfection affects test-time scaling,” in International Conference on Learning Representations, 2026, pp. 150 895–150 936.

[70] D. Xu et al., “From exploration to exploitation: A two-stage entropy RLVR approach for noise-tolerant MLLM training,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 17 776–17 786.

[71] Z. Stojanovski et al., “Reasoning Gym: Reasoning environments for reinforcement learning with verifiable rewards,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[72] P. Wilhelm, T. Wittkopp, and O. Kao, “Monitoring emergent reward hacking during generation via internal activations,” arXiv preprint arXiv:2603.04069, 2026.

[73] Z. Yu et al., “RESTRAIN: From spurious votes to signals—Self-Training RL with self-penalization,” in International Conference on Learning Representations, 2026, pp. 5562–5586.

[74] Z. Zhang et al., “Co-rewarding: Stable self-supervised RL for eliciting reasoning in large language models,” in International Conference on Learning Representations, 2026, pp. 82 520–82 558.

[75] N. Vakada, M. Li, and S. Ji, “Label-free steering: Compressing testtime reinforcement learning into bias-only subspaces,” arXiv preprint arXiv:2609.18587, 2026.

[76] X. Zhao, Z. Kang, A. Feng, S. Levine, and D. Song, “Learning to reason without external rewards,” in International Conference on Learning Representations, 2026, pp. 5562–5586.

[77] K. Du, M. Cao, H. Zhang, Y. Wang, X. Huang, and N. Li, “Dual consensus: Escaping from spurious majority in unsupervised RLVR via two-stage vote mechanism,” arXiv preprint arXiv:2603.16223, 2026.

[78] J. Wu, Y. George, J. Ye, Y. Wu, D. F. Schmidt, and J. Cai, “SPINE: Token-selective test-time reinforcement learning with entropy-band regularization,” arXiv preprint arXiv:2511.17938, 2026.

[79] Y. Zhou et al., “Breaking the exploration bottleneck: Rubric-scaffolded reinforcement learning for open-ended LLM reasoning,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306, 2026, pp. 165 222–165 251.

[80] V. Viswanathan, Y. Sun, X. Kong, M. Cao, G. Neubig, and S. Wu, “Checklists are better than reward models for aligning language models,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 114 728–114 754.

[81] Y. Li, H. R. Kethireddy, and S. Das, “Evaluating feature dependent noise in preference-based reinforcement learning,” arXiv preprint arXiv:2601.01904, 2026.