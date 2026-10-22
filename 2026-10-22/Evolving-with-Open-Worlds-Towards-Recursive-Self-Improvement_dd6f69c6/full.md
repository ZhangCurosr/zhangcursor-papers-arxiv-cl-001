# Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting

Zhipeng Chen<sup>1,2\*§</sup>, Yunsen Liang<sup>1,2\*§</sup>, Yanxi Chen<sup>1</sup>, Hongkang Zhao<sup>1</sup>, Lingyun Feng<sup>1</sup>, Ruiyang Ren<sup>3</sup>, Guoxin Chen<sup>1</sup>, Jia Deng<sup>1,2§</sup>, Feng Chang<sup>2</sup>, Chuan Hao<sup>2</sup>, Ran Tao<sup>2</sup>, Jian Yang<sup>2</sup>, Ji-Rong Wen<sup>1†</sup>, Bryan Dai<sup>2†</sup>, Wayne Xin Zhao<sup>1†</sup> <sup>1</sup>Gaoling School of Artificial Intelligence, Renmin University of China <sup>2</sup>IQuest Research <sup>3</sup>Nanyang Technological University <sup>\*</sup>Equal contributors <sup>†</sup>Corresponding Authors <sup>§</sup>Work done during an internship at IQuest Research. Email: {zhipeng\_chen,yunsenliang04,jrwen}@ruc.edu.cn, batmanfly@gmail.com

## Abstract

Most contemporary artificial intelligence (AI) systems are released with frozen parameters and improve through external updates, whereas an ideal system would keep improving from its own experience, a capacity we call recursive self-improvement (RSI). Yet studying RSI requires an environment that is continuously changing, objectively verifiable, and naturally grounded, posing a fundamental challenge for agent evolution. Here we introduce DEEPEVOLVE, a sustainable evolution agent system grounded in future event prediction, in which agents continuously interact with a dynamically evolving environment and improve by internalizing and externalizing the experience gained through these interactions. A living environment synthesizes verifiable tasks, and a recursive-evolution engine turns the reasoning trajectory behind each prediction back into capability: decoupling search from prediction adapts parameter-level evolution to the contamination-prone live environment, while distilling reusable skills lets the agent evolve from the limited live trajectories such an environment affords. On the external FutureX leaderboard, which evaluates predictions against real-world outcomes and is updated weekly, our self-improved forecaster ranked first across all systems over three consecutive weeks, achieving up to a 195% improvement in rating. On our synthesized evaluation, it ranked among the top three across four base LLMs, outperforming the strongest baseline by up to 10.7 Bradley–Terry points. More broadly, these results suggest that a system built for self-improvement is best understood as an agent that co-evolves with a genuinely living, openended environment, pointing to a shift in AI from training static models on fixed data toward cultivating agents that grow stronger from their own experience, with such an open world as the indispensable basis of that growth. Our code and data are available at https://github.com/RUCAIBox/DeepEvolve.

## 1. Introduction

Intelligence is often measured by what a system can do at a single moment in time [2, 19]. Yet the more consequential question may not be how capable a model is today, but whether it can become more capable through its own experience [9, 14, 23]. Across the natural and social sciences, the systems we regard as most powerful (e.g., biological evolution, the scientific method, human expertise) are precisely those that improve themselves cumulatively, correcting past errors and compounding past gains [8, 21, 25]. This suggests that an ideal artificial system should be measured less by what it can do at any single moment than by how its capabilities form and grow over time. A system that improves itself from experience is a prerequisite for AI that acts as an autonomous agent and as a genuine instrument of discovery. In this work, we study this form of recursive self-improvement (RSI) in the concrete and demanding setting of real-world event forecasting.

Studying recursive self-improvement, however, is not merely a matter of method. It places unusually high demands on the research setting itself. An agent can only improve recursively if its environment keeps posing fresh challenges and can tell, without human intervention, whether the agent is doing better than before. What this calls for is an interactive environment, yet the fixed datasets that dominate today’s evaluation offer no such thing: inert and unchanging, they neither renew the challenge nor judge progress on their own. Real-world event forecasting, we argue, is a natural testbed for such study: it is grounded in the unfolding world rather than a hand-constructed dataset, verifiable through future events whose objective ground truth arrives with time, and non-stationary because the world keeps changing. Together these properties close the loop that recursion requires: the agent observes the world, makes predictions, receives future outcomes, learns from the resulting experience, and improves the predictions it makes next (Figure 1a). Whereas prior work on event and time-series forecasting [11, 30] has largely pursued single-shot predictive accuracy, we instead treat forecasting as an environment in which to study how agents improve recursively through continuous interaction with a changing world.

Succeeding in this environment first requires a few foundational capabilities for solving any single forecasting task. An agent must be able to gather task-relevant evidence from a vast information space, as exemplified by search-augmented systems [26, 27], and to reason, combining and weighing that evidence to draw multi-step conclusions [1, 5]. Both capabilities are already standard in existing systems, and we treat them as the foundation on which our self-improvement mechanisms build. Recursive selfimprovement goes further, requiring a capability different in kind: an agent that not only solves the task in front of it but also changes its own behavior in light of past successes and failures. Such an agent addresses a distinct question: how its capacity to solve tasks can itself keep improving. We take this ability to evolve as the central concern of this work, and reserve the name DEEPEVOLVE for the framework we propose to realize it (Section 4).

Realizing such an evolving agent runs against a basic limitation of today’s AI: it is largely static and dependent on outside intervention to advance, a set of parameters frozen at release rather than a system that keeps improving on its own [4, 25]. This shows up on two fronts. Evaluation is static, scoring a model on a fixed dataset [6, 16] that can be memorized or contaminated; and improvement is passive, with capability gains driven from outside by human-curated data and offline retraining [7, 12] rather than by the model’s own experience. This condition also differs from neighboring lines of work: self-training on self-generated data [13, 28] lacks an external, verifiable ground truth and can reinforce a model’s own biases; continual learning [15, 20] operates over static or artificially partitioned task streams rather than real, non-stationary feedback; and reinforcement learning for reasoning [22, 23] optimizes reward on a task but does not, by itself, target improvement that compounds across time. By recursive, we mean exactly this compounding: the parameters, skills, and environment produced by one round of improvement become the input to the next, forming a self-improvement loop rather than a one-off gain [8, 21].

Our central argument departs from most work on self-improvement, which pushes on a single lever, be it the model’s weights [13], its prompts [10], or its memory [24]. What makes self-improvement sustainable, we argue, is a system that keeps accumulating experience on its own rather than being fed hand-curated data from outside, and our answer is a living environment in which the agent continually acts, observes outcomes, and learns. The logic is directional. The environment is the source, posing progressively harder tasks that supply the experience without which no further improvement is possible; the agent’s parameters and skills are the two forms its evolution takes, both shaped from one substrate, the forecasting trajectories that are reinforced into parameters, distilled into skills, and mined for the weaknesses that build a harder environment. The three thus couple into a single self-sustaining loop, in which the environment evolves to let the agent keep evolving.

Building on this view, we introduce DEEPEVOLVE, an event-forecasting framework for studying recursive self-improvement in a continuously evolving real-world environment, in which the agent and its environment co-evolve. DEEPEVOLVE comprises three modules arranged as a closed loop rather than a pipeline. PROMETHEUS, a living data environment, continually ingests real-world events to form the non-stationary, verifiable world in which forecasting takes place. BRANCHUS, an agentic forecaster, produces predictions together with their full trajectories, generating the learning substrate described above. ATHENA, the recursive-evolution engine, learns from these trajectories through reinforcement learning and skill accumulation to drive self-improvement. The loop closes when ATHENA’s improvements strengthen BRANCHUS’s forecasts, and these stronger forecasts in turn place new demands on PROMETHEUS, pushing the environment to evolve and sustaining an open-ended cycle of agent-environment co-evolution.

This co-evolution produces a strong forecasting system. Across all four LLMs, BRANCHUS ranks among the top three overall and leads its block, and because every result is scored against real-world outcomes, this consistency reflects an advantage intrinsic to the design (BT of 62.2 on QWEN3.8-MAX, +10.7 over the strongest baseline in Table 1). More importantly, the gains compound rather than saturate: as the agent’s own live trajectories accumulate into a growing pool, the skills distilled from them lift forecasting quality round by round rather than jumping once and plateauing (Figure 6). Our experiments trace this improvement to how environment, parameters, and skills evolve together, each by its own means. Reinforcement learning sharpens the parameters, tightening probabilistic judgment beyond what accuracy registers (the numeric calibration score improves by 0.31–0.38 while choice accuracy moves by only 1.3 points in Figure 5), yet its reach is bounded by the experience available. The experience distilled into skills lifts capability without touching the weights, learning from only limited trajectories rather than repeated rollouts and so sidestepping the instability that gradient updates can introduce. The environment closes the loop: when it co-evolves with the agent, synthesizing verifiable tasks aimed at the agent’s demonstrated weaknesses, these otherwise isolated and bounded gains add up to a larger improvement (Figure 9). This is evidence that recursive self-improvement arises from the three levers evolving as one.

Taken together, these findings establish recursive self-improvement as the co-evolution of an agent and its living environment. The environment is the abstraction we commit to. Our first contribution is to construct a real, dynamic, and verifiable environment for forecasting: grounded in the unfolding world, nonstationary as that world changes, and scored against outcomes that resolve on their own. This environment lets all three levers improve in concert. Its verifiable outcomes become rewards that sharpen parameters; its live experience supplies the trajectories that distil into skills; and its dynamic form lets it be rewritten toward the agent’s exposed weaknesses, so each round feeds the next. Our second contribution turns this opening into method: DEEPEVOLVE formalizes the environment, the agent, and the recursive-evolution engine as three modules in a closed loop (Section 4). More broadly, the development of AI may shift from training static models on fixed datasets toward cultivating evolving agents within living environments, and with it the measure of progress: the rate, ceiling, and durability of a system’s improvement, which only an evolving environment can reveal.

Forecasting is the concrete setting we build and study, and the way we construct this environment, and the methodology that evolves an agent inside it, carry over to any setting that is grounded, verifiable, and non-stationary, so the same loop can be extended to far more general scenarios. It also recasts how recursive self-improvement might be pursued, suggesting that progress may come less from any single, ever-larger model than from co-evolving parameters, skills, and the environment that supplies their experience. Such progress is not without cost: it remains bounded by the quality and quantity of available experience, long-horizon evaluation is expensive, and stability under non-stationary drift is not guaranteed. Yet even bounded, it reframes the question with which we began. The deepest measure of intelligence may lie less in what a system can do at a single moment than in whether it can grow stronger from its own experience, and our study offers concrete evidence that such growth can be made to occur, and to be measured, in a real and verifiable world.

![](images/b2922fa1757dce52e64b85a48a649ef1582fd1b3fb77506b5a2cbb31ddb1e686.jpg)  
Figure 1. Overview of DEEPEVOLVE for recursive self-improvement in real-world event forecasting. a, The closed loop connects question generation, forecasting, and evolution, with resolved outcomes supplying verifiable rewards and updates feeding into subsequent rounds. b, PROMETHEUS generates questions through templates and agent-driven discovery, followed by temporal validation, deduplication, conflict detection, and diversity-aware selection. c, BRANCHUS separates evidence gathering, by a search team of a scheduler, searcher, and critic, from probabilistic prediction, by a predict team. The search team is time-sensitive, as its result depends on when it runs, while the predict team is time-invariant, reasoning only over the search team’s cached evidence and reasoning trajectories. Both teams retrieve relevant forecasting skills. d, ATHENA updates the predict team’s parameters using cached evidence and verifiable rewards, distills search trajectories into reusable forecasting skills, and converts observed weaknesses into task-synthesis skills for PROMETHEUS. Resolved outcomes inform parameter and environment updates but are withheld from the behavior-level judge.

## 2. Results

## 2.1. Experimental Design and DeepEvolve Framework

To forecast future events and to test whether an agent can improve itself from doing so, we develop DEEP-EVOLVE, a framework that turns experience into capability through a closed loop of three components. PROMETHEUS, the living data environment, gathers daily news and synthesizes it into forecasting tasks whose ground truth is revealed only once the events resolve. BRANCHUS, the agentic forecaster, gathers evidence with external tools and estimates the probability of each event, producing a full prediction trajectory. ATHENA, the recursive-evolution engine, consumes these trajectories and drives improvement along three axes. It applies reinforcement learning with verifiable rewards (RLVR) to update the parameters of BRANCHUS, distills trajectories into reusable forecasting skills, and evolves adversarial task-synthesis skills in PROMETHEUS to make the environment harder. Each forecasting trajectory is rich enough to feed all three axes at once: because a future-event prediction forces the agent to gather evidence and weigh it under uncertainty, the same trajectory can be reinforced against its resolved outcome, mined for reusable skills, and analysed for the weaknesses from which a harder environment is built. Experience is thus fed back to improve the agent’s parameters, its skills, and its environment jointly, realizing the recursive self-improvement loop at the heart of this work. The subsections that follow examine these three axes in turn, after first establishing the strength of the system as a whole.

We draw forecasting tasks from two complementary sources. The first is FutureX [29], an external benchmark refreshed every week that asks models to predict events in the coming week. Its tasks span four difficulty levels (L1 to L4), take multiple-choice, ranking, and numerical forms, and cover domains from finance and politics to sports, natural disasters, and technology, so strong performance reflects general forecasting ability. The second is a higher-throughput evaluation synthesized by PROMETHEUS, used purely as a task generator with its adversarial evolution disabled so the evaluation stays unbiased. Both sources are genuinely live rather than static or cached. Each day, PROMETHEUS ingests fresh news from 125 websites and writes forecasting questions about events that resolve the following day, so BRANCHUS must predict outcomes whose ground truth does not yet exist. This one-day horizon shortens the verification cycle and lets us evaluate models far more frequently than a weekly benchmark allows.

Every question concerns a future event, so predictions are verified against realized outcomes rather than a fixed answer key. For the PROMETH EUS evaluation, once an event resolves we task Codex [17] with retrieving the realized outcome from the open web and scoring the prediction against it. Beyond raw accuracy, we report a calibration metric: the model outputs a probability for each event, and we check whether the empirical frequency of those events matches the predicted probabilities. We also stabilize these per-question scores before aggregating. On FutureX we adopt its official sliding-window Bradley–Terry (BT) model [3], which infers a consistent latent strength from head-to-head weekly results. On PROMETHEUS, a single day of questions is too small and too variable to rank harnesses reliably, so unless stated otherwise every reported number averages over several consecutive days of live evaluation.

## 2.2. Recursive evolution lifts DEEPEVOLVE to the strongest forecasting system

As promised in the introduction, DEEPEVOLVE attains strong and generalizable forecasting performance. The numbers here report BRANCHUS, the module that performs inference, after the full recursive selfimprovement of DEEPEVOLVE, so they are a system-level result with parameters, skills, and environment already co-evolved rather than a starting point. We pair the evolved system with four LLMs that differ in size, provider, and reasoning behavior (QWEN3.8-MAX, GPT-5.6-SOL, DEEPSEEK-V4-PRO, and CLAUDE-OPUS-4.6) and compare against a diverse suite of baselines on the PROMETHEUSsynthesized evaluation (Table 1; Section 4.5 for details), with FutureX giving the externally refreshed view below (Figure 2). Regardless of the base model, DEEPEVOLVE is strongest: it attains the highest BT score and best rank in every block, and the four BRAN CHUS configurations occupy the top three positions overall, led by QWEN3.8-MAX at a BT of 62.2. The margin over the best baseline ranges from +10.7 to +0.5 BT but never changes sign. Because the evaluation resolves against real-world outcomes rather than a fixed answer key, this cross-model consistency indicates an advantage intrinsic to DEEPEVOLVE rather than a fortunate match between one design and one model. The subsections that follow isolate how each axis of self-improvement turns the agent’s own experience into this frontier.

![](images/49d5d70959eb31ed5a7e91eba2e28f1a23dfb1564cbb985dae2489178a686968.jpg)  
Figure 2. Overall FutureX ranking across four weekly evaluations. Each panel reports the sliding-window Bradley–Terry rating of every harness for one week, with BRANCHUS (ours) shown against the baseline harnesses. The complete official leaderboards are provided in Appendix A (Figure 11).

Model capability and agent design are complementary rather than substitutable. A stronger LLM lifts the ceiling for every system, so the strongest result in the table is reached by placing BRANC HUS on the most capable model (QWEN3.8-MAX); conversely, BRAN CHUS rescues a weaker base, reaching rank 3 on DEEPSEEK-V4-PRO and outscoring every baseline irrespective of the model those baselines are built on. The two effects compose and neither alone is sufficient to reach the frontier, which suggests that further gains require jointly improving the base model and the agent that scaffolds it, motivating the recursive-evolution mechanisms studied in the remainder of this paper.

Crucially, this advantage reflects well-calibrated judgement rather than over-confident guessing. Raw accuracy alone does not separate the systems: on choice questions the baselines occasionally edge out BRANCHUS by a few tenths of a point, for example MAS on DEEPSEEK-V4-PRO and OPENCLAW on CLAUDE-OPUS-4.6. The decisive gap instead appears on calibration and numeric sharpness: BRANCHUS attains the best Brier score in three of the four blocks and the best relative WIS on both QWEN3.8-MAX (0.737) and GPT-5.6-SOL (0.865), so its interval forecasts are tighter than the strongest baseline rather than merely competitive. Because the Brier score and WIS jointly penalize overconfident wrong answers and unjustified hedging, BRANCHUS does not trade calibration for accuracy;

Table 1. Forecasting performance of BRANCHUS against baseline harnesses across four LLMs on the PROMETHEUS-synthesized evaluation. We report accuracy, Brier, and AUROC for choice questions, relative WIS for numeric questions, and the Bradley–Terry score (BT) with its rank overall. The rank is a single ordering over all 23 harness–LLM systems in the table, not a within-block rank. Best values per column within each base-LLM block are in bold.
<table><tr><td rowspan="2">Harness</td><td colspan="3">Choice Questions</td><td>Numeric Questions</td><td colspan="2">Overall Performance</td></tr><tr><td>Accuracy ↑</td><td>Brier ↓</td><td>AUROC ↑</td><td>Relative WIS ↓</td><td>BT↑</td><td>Rank↓</td></tr><tr><td colspan="8">QWEN3.8-MAX</td></tr><tr><td>REACT</td><td>70.7</td><td>0.346</td><td>0.784</td><td>0.913</td><td>47.9</td><td>16</td></tr><tr><td>MAS</td><td>70.3</td><td>0.343</td><td>0.808</td><td>0.871</td><td>51.5</td><td>11</td></tr><tr><td>OPENCLAW</td><td>71.9</td><td>0.356</td><td>0.749</td><td>0.933</td><td>47.1</td><td>18</td></tr><tr><td>DEERFLOW</td><td>71.9</td><td>0.354</td><td>0.748</td><td>0.927</td><td>45.5</td><td>20</td></tr><tr><td>MIROFLOW</td><td>70.3</td><td>0.351</td><td>0.790</td><td>0.900</td><td>50.6</td><td>13</td></tr><tr><td>BRANCHUS (Ours)</td><td>72.3</td><td>0.336</td><td>0.787</td><td>0.737</td><td>62.2</td><td>1</td></tr><tr><td colspan="7"></td></tr><tr><td>REACT</td><td>70.3</td><td>0.340</td><td>GPT-5.6-SOL 0.812</td><td>0.915</td><td>53.4</td><td>7</td></tr><tr><td>MAS</td><td>72.3</td><td>0.342</td><td>0.776</td><td>0.913</td><td>55.0</td><td>5</td></tr><tr><td>OPENCLAW</td><td>71.9</td><td>0.346</td><td>0.777</td><td>0.909</td><td>53.4</td><td>8</td></tr><tr><td>MIROFLOW</td><td>69.5</td><td>0.370</td><td>0.765</td><td>4.152</td><td>40.7</td><td>23</td></tr><tr><td>BRANCHUS (Ours)</td><td>71.5</td><td>0.332</td><td>0.813</td><td>0.865</td><td>57.9</td><td>2</td></tr><tr><td colspan="7">DEEPSEEK-V4-PRO</td></tr><tr><td>REACT</td><td>71.5</td><td>0.350</td><td>0.769</td><td>0.848</td><td>53.1</td><td>10</td></tr><tr><td>MAS</td><td>72.7</td><td>0.340</td><td>0.773</td><td>0.823</td><td>55.9</td><td>4</td></tr><tr><td>OPENCLAW</td><td>71.5</td><td>0.363</td><td>0.733</td><td>0.920</td><td>48.2</td><td>14</td></tr><tr><td>DEERFLOW</td><td>70.3</td><td>0.371</td><td>0.732</td><td>0.882</td><td>47.4</td><td>17</td></tr><tr><td>MIROFLOW</td><td>69.9</td><td>0.351</td><td>0.787</td><td>0.851</td><td>53.9</td><td>6</td></tr><tr><td>BRANCHUS (Ours)</td><td>71.9</td><td>0.341</td><td>0.783</td><td>0.835</td><td>56.4</td><td>3</td></tr><tr><td colspan="7"></td></tr><tr><td>REACT</td><td>72.3</td><td>0.355</td><td>CLAUDE-OPUS-4.6 0.739</td><td>1.106</td><td>42.0</td><td>21</td></tr><tr><td>MAS</td><td>71.5</td><td>0.376</td><td>0.714</td><td>1.105</td><td>41.6</td><td>22</td></tr><tr><td>OPENCLAW</td><td>73.8</td><td>0.351</td><td>0.725</td><td>0.893</td><td>51.2</td><td>12</td></tr><tr><td>DEERFLOW</td><td>72.3</td><td>0.352</td><td>0.756</td><td>1.021</td><td>48.2</td><td>15</td></tr><tr><td>MIROFLOW</td><td>70.7</td><td>0.360</td><td>0.751</td><td>0.959</td><td>46.0</td><td>19</td></tr><tr><td>BRANCHUS (Ours)</td><td>73.0</td><td>0.331</td><td>0.793</td><td>0.900</td><td>53.1</td><td>9</td></tr></table>

its probabilities and intervals faithfully reflect the evidence it gathered, which is what lifts it to the top of the BT ranking even where its top-1 accuracy is only matched.

Finally, this advantage is stable over time rather than an artifact of a single evaluation window. Figure 2 tracks the sliding-window BT rating of every harness across four consecutive weekly evaluations, and BRANCHUS (dark blue) sits at or near the top of the field in each panel, holding its lead as the task distribution is refreshed, whereas standard workflows are both lower and less steady, swinging more sharply between weeks and gaining less from a stronger LLM. We attribute this gap to two related bottlenecks: these workflows execute fixed, largely one-pass evidence gathering, so additional model capability is spent re-reading the same limited context rather than expanding source coverage; and they lack an explicit mechanism for tracking hypotheses and calibrating confidence across steps, so long chains of reasoning accumulate uncorrected errors and drift toward over-confident predictions. Together, these limitations explain why standard workflows saturate well below the frontier defined by BRANCHUS, and why simply scaling up the LLM is insufficient to close the gap on long-horizon forecasting tasks. This frontier is where our agent begins rather than ends: the remaining subsections show that recursively evolving BR ANCH US pushes it beyond this already-leading base form, so that capability grows from the agent’s own experience over successive rounds instead of jumping once and plateauing.

![](images/d8f82e59bce1719d581e8c36d9c14341c273b5f723d2764e16e8b12aec0d59a8.jpg)  
(a) MAPE across evolution rounds.

![](images/2bc2182aac2a47484cf955750b74c00387557f5618c5e2cf8a93d05a5a607727.jpg)  
(b) Brier score across evolution rounds.  
Figure 3. Parameter evolution improves forecasting quality continuously. Across two evolution rounds, the training set grows from 600 to 1,000 examples, accompanied by consistent improvements in forecasting performance. Left: the performance of relative WIS. Right: calibration, measured by the Brier score. For both metrics, lower is better.

## 2.3. Search-prediction decoupling adapts RFT and RLVR to contamination-prone live environments

The first of the three axes is the model’s parameters. The obstacle to internalizing forecasting experience is that the environment is live and easily contaminated: once an event resolves, its answer becomes retrievable from the web, so the task can no longer train a model to predict rather than look up. The clean window is short, yet RFT and RLVR must roll out and refine over the same tasks repeatedly, which leaves too little uncontaminated, verifiable experience to support training at all. Thefunction-oriented design of BRANCHUS removes this obstacle by splitting the incompatible needs across teams organized by function: a time-sensitive search team gathers evidence while a question is still open, and a timeinsensitive predict team reasons over it into a forecast. We cache the evidence and both teams’ trajectories while the task is still fresh, so the many RFT/RLVR rounds that follow read from this frozen pool rather than from a web that has since revealed the answer. Because the predict team is trained only on fixed, pre-cached evidence, it cannot influence what it sees, which also closes the on-policy shortcut in which RLVR would manufacture answer-entailing evidence; any gain reflects better reasoning over the same information. Once enough cached trajectories resolve into verifiable labels, we train the predict team with two complementary paradigms: RLVR optimizes against each resolved outcome, and RFT fine-tunes on successful trajectories.

Under this function-oriented protocol, RLVR yields consistent gains across both LLMs (Figure 5): choice accuracy rises by 1.3 points on each model, while the numeric score (1 − bWIS) improves by 0.31 and 0.38. These clean, consistent gains from a reward computed against resolved outcomes are themselves evidence that the design has kept contamination out of the training signal, since a leaked answer would have collapsed the reward long before it could sharpen the model. The two kinds of metric respond at different rates, and we attribute this to the metrics themselves. Accuracy is binary and insensitive to how belief mass shifts, whereas calibration is continuous and reflects any move toward the true outcome probabilities. Calibration thus surfaces improvements too subtle for accuracy to register, acting as an early indicator that RLVR is tightening the model’s probabilistic judgment even when the top-1 answer is unchanged.

RFT, under the same protocol, also improves every axis across two LLMs, again led by calibration (Figure 4): accuracy rises by 0.7 to 2.9 points, while Brier and relative WIS improve consistently. These gains compound over rounds rather than being spent in one pass: training GLM-4.7-Flash with RFT on each round’s cached trajectories, the first round already sharpens forecasts (Brier 0.651 → 0.636, relative WIS 1.02→0.99), and the second round keeps pushing them down (Brier 0.630, relative WIS 0.98) as the task pool grows from 600 to 1000 (Figure 3). This is the multi-round self-improvement at work: because the environment is live, every round brings a fresh batch of newly resolved events, and the new experience distilled from them carries the forecasting quality a step further. The live stream of information is thus what sustains continual evolution, so the model never runs out of fresh, verifiable experience to learn from. Evolving the environment alongside the model amplifies this effect, since tasks that keep targeting the agent’s current weaknesses turn this incoming information into richer experience for the next round and drive still larger gains, which we discuss in Sec. 2.7.

![](images/a67c08c34e9f7a6e090df1818c766be7f9d7ba6fd97d1209384bf52506c28bc0.jpg)

![](images/2367ea58f4455ea295441952f19fab32fa48823423ac35027423934a0f367553.jpg)  
Figure 4. RFT delivers consistent, calibration-led gains over the base model. Each panel traces the Base → RFT transition for two LLMs on one axis of forecasting quality, with every axis oriented so that upward movement is an improvement. Fine-tuning on successful trajectories lifts every axis, and the improvement is led by calibration rather than raw accuracy. Left: choice accuracy. Right: numeric interval forecasts, measured by relative WIS.

## 2.4. Evolving skills unlock recursive self-improvement from fleeting live trajectories

The live trajectories that the function-oriented design cached for RLVR can instead be read for the lessons they contain, internalizing the same scarce experience in a lighter, more transparent form: reusable forecasting skills that sit outside the model weights and are invoked at inference time. A single live trajectory thus feeds more than one lever, as RLVR turns it into a parameter update while skill distillation turns the same record into an explicit, inspectable procedure learned without any gradient update. The stakes here are sharper than for parameters, because fresh trajectories are collected only within each task’s short open window, so the axis is useful only if a mere handful of them can already lift the system. We let ATHENA distill the lessons from BRANCHUS’s trajectories into skill entries and build libraries over three evolution rounds, from nested pools of 88, 186 and 262 live trajectories, each round extending the previous one as it accumulates more trajectories, giving BRANCHUS the current library at inference so that any change in performance is attributable to the skills alone.

A handful of live trajectories already unlocks clear improvements on both axes of forecasting quality (Figure 6). Relative to the skill-free BRANCHUS baseline, the Brier score drops from 0.135 to 0.122 (0.013 lower), and the same pattern reappears on the numeric side of the benchmark, where the MAPE falls from 2.80% to 2.28% (0.52 points lower). Two features of the result are worth emphasising. First, the skills are inserted without touching the model weights, so the gain is a direct measurement of the value of externalised live experience rather than an interaction with the underlying policy. Second, the joint improvement in the Brier score and the numeric MAPE mirrors the pattern observed under RLVR, indicating that distilled skills and reinforcement learning act along the same axis of forecasting competence and can, in principle, be composed.

![](images/32ffacbb67a94e2a46948332a2fc37bcda84deb1051180091127d8142395a377.jpg)  
(a) Qwen3-4B-Thinking.

![](images/ac3ccb64d346924dc311d9d86befa6eee0ea8851905ae1fcba9b000a1c812da8.jpg)  
(b) Qwen3-8B.  
Figure 5. RLVR sharpens both forecasting-quality metrics across LLMs. For each LLM, we compare the base model against its RLVR-tuned counterpart on choice accuracy (left blue axis) and the numeric score (right red axis, 1 − bWIS). Each metric shows two side-by-side bars, the base model and its RLVR-tuned counterpart, as absolute values. RLVR improves both metrics on both models.

As each round accumulates more live trajectories into the growing pool, forecasting quality improves round by round, though at a slowing pace. Each pool adds gain on top of the last, with the Brier score reaching 0.007, 0.012 and 0.013 below the baseline across the three rounds and the MAPE descending from 2.80% to 2.40%, 2.32% and 2.28% (Figure 6), so the skills distilled at a later round help BRANCHUS more than those from an earlier one. Later rounds move the aggregate metric less but are far from idle, for what they add is breadth: topical coverage broadens roughly threefold, from 168 domains and 173 tasks at round 1 (88 trajectories) to 479 domains and 505 tasks at round 3 (262 trajectories) (Figure 7b), as later skills reach domains that appeared only rarely before. The slowdown therefore reflects a shift in what each new live trajectory buys rather than saturation of the mechanism: the dominant forecasting patterns are captured early, so even a small pool internalizes the core modes of reasoning, after which added skills extend coverage into domains the evaluation set rarely samples and their benefit is diluted in the overall metric. Having now internalized the same trajectories into both parameters and skills, we next compare where each axis helps and at what cost, examining how the benefit of skills depends on task structure, before turning to the environment that supplies the experience.

## 2.5. Skill evolution guides evidence gathering while parameter evolution sharpens evidence use

The same forecasting experience can be turned into capability in two ways. Parameter evolution internalizes it, folding the lessons of past trajectories into the model weights, whereas skill evolution externalizes it, writing those lessons out as explicit, reusable procedures that sit beside the weights. The two forms act on different stages of forecasting: externalized skills shape how the agent explores for evidence, while internalized parameters shape how well it uses the evidence it has gathered, and a practical system must know when each is worth its cost. To expose where skills help, we partition the questions synthesized by PROMETHEUS by how they were generated: structure synthesis problems, scraped from fixed fields of a designated website and instantiated over a hand-designed template, so they share a stereotyped structure and a few recurring reasoning patterns; and open synthesis problems, freely proposed by CODEX from the day’s news, so they are heterogeneous in topic, format, and required evidence. Alongside accuracy and calibration, we track the agent’s tool-call budget, which reveals whether skills make the agent do more work per question or the same work more directly.

![](images/f2005e85cefddb612ec3ca6c22498065d8489b54bdf19f6811392ca7b4cffa27.jpg)  
(a) MAPE across evolution rounds.

![](images/bd32c1ec0921112607a88f4f1074f0f240c9065e62d23b2206e553a95c6beebf.jpg)  
(b) Brier score across evolution rounds.  
Figure 6. Evolving skills improve forecasting quality continuously. Distilled skills, inserted at inference without touching the weights, sharpen BRANCHUS’s forecasts on both axes of forecasting quality. Because each round adds its live trajectories to the pool (nested pools of 88, 186, and 262 trajectories), the skills ATHENA extracts over successive evolution rounds lift forecasting quality round by round rather than in a one-off gain. Left: numeric accuracy, measured by MAPE. Right: calibration, measured by the Brier score. For both metrics, lower is better.

Table 2. Token cost of distilling skills from one trajectory, by stage. Values are averages per trajectory; output tokens include reasoning tokens.
<table><tr><td>Stage</td><td>Step</td><td>Calls</td><td>Input</td><td>Output</td><td>Total</td></tr><tr><td rowspan="3">Evaluation</td><td>Step scoring (searcher, critic)</td><td>6.49</td><td>100,781</td><td>8,476</td><td>109,257</td></tr><tr><td>Scheduler scoring</td><td>1.45</td><td>34,487</td><td>2,259</td><td>36,745</td></tr><tr><td>Trajectory diagnosis</td><td>1.05</td><td>52,396</td><td>1,098</td><td>53,494</td></tr><tr><td rowspan="2">Extraction</td><td>Experience extraction</td><td>1.02</td><td>29,363</td><td>1,372</td><td>30,735</td></tr><tr><td>Experience filtering</td><td>0.91</td><td>29,520</td><td>508</td><td>30,029</td></tr><tr><td rowspan="2">Generalization</td><td>Four-level generalization</td><td>2.02</td><td>18,409</td><td>2,354</td><td>20,763</td></tr><tr><td>Generalization review</td><td>2.02</td><td>19,813</td><td>2,299</td><td>22,112</td></tr><tr><td colspan="2">Total</td><td>14.96</td><td>284,769</td><td>18,366</td><td>303,135</td></tr></table>

Where skills help turns out to depend sharply on how the problem was synthesized (Figure 8a). On structure synthesis problems, which reuse a few recurring patterns, recursively evolved skills cut the agent’s tool use: as the library grows, tool calls drop by 44% (from 32 to 18 per question) while MAPE stays at least as strong as the skill-free baseline. One encounter is often enough for ATHENA to distil a skill that captures the whole family, letting BRANCHUS skip exploratory search and act on a compiled shortcut. On open synthesis problems drawn from domains the skills were never designed for, the signature reverses: the agent issues 13% more tool calls (from 30 to 34 per question) and investigates more broadly, and here the quality gains are the largest in the plot, with MAPE falling by 32% (from 3.40% to 2.31%) and closing almost the entire gap to the accuracy on structure synthesis problems. On these novel problems a skill cannot supply the exact procedure, so it instead acts as a high-level plan that opens up broader lines of search than the agent would pursue unaided. Externalizing experience therefore improves the agent’s exploration: it makes evidence gathering either more direct, when a matching skill compiles the task away, or more far-reaching, when a loosely related skill widens the search over an unfamiliar domain.

![](images/8683c9b4039f8ceb81efd1a153496b6c8ea18e2acb4eaa2c70e8d91ba96b3d73.jpg)  
(a) Number of distilled skills.

![](images/d780543813fd7fb7cc60ece88fffe8939eec2163b56c016161ab38547819b1c1.jpg)  
(b) Topical coverage.  
Figure 7. Distilled skills scale across evolution rounds. As the collected trajectories grow round by round, ATHENA keeps distilling new skills from them, so the library expands rather than saturating, and later skills extend into domains seen only rarely before. Left: the number of distilled skills at each evolution round. Right: the breadth of topical coverage, measured by the number of the domains and concrete tasks that the skills cover.

Internalizing the same experience into parameters acts on a different stage and leaves this exploration behaviour untouched. Because ATHENA updates only the predict team, which reasons over cached evidence, and never the search team that issues tool calls, RLVR changes neither the number of searches nor what evidence is gathered; what it changes is how well that fixed evidence is used, as the predict team learns to weigh and integrate the same observations into a sharper, better-calibrated forecast. The two forms also differ sharply in cost. Externalization is pure inference, with no rollout and no backward pass, a gap that only widens with model size (Figure 8b and Table 2). This cheapness, however, comes with a ceiling: a few trajectories, on the order of 88, already yield most of the achievable gain, because externalization can only reorganise patterns some trajectory already exhibits and cannot discover strategies outside BRANCHUS’s current repertoire. Internalization spends its larger budget precisely on behaviours the model does not yet exhibit, and keeps improving past the point where externalization plateaus.

Externalization and internalization are therefore complementary rather than substitutes: externalizing experience into skills gives rapid, cheap gains in exploration up to a well-defined ceiling, and internalizing it into parameters gives slower, costlier gains in evidence use that push beyond it. This suggests a simple division of labour for the recursive loop, in which a compute-efficient system externalizes experience into skills every round and reserves parameter updates for occasional longer bursts, recovering most of the capability of pure RLVR at a fraction of the compute and letting the same budget drive many more turns of the agent and environment loop. The compute saved this way is best spent on the third axis, the environment itself, since, as we show next, an environment that keeps posing new weakness-targeted tasks is what decides whether recursive improvement continues to compound or eventually flattens.

2.6. An evolving environment continuously turns the agent’s own failures into adversarial challenges The two agent-side axes share a common ceiling: whether experience is internalized into parameters or externalized into skills, both can only reorganise what the agent has already encountered, so each is bounded by the distribution of tasks the environment poses, and neither can lift this ceiling because neither controls the experience it feeds on. The third axis does. Rather than draw tasks from a fixed distribution, we let the environment evolve in response to the agent, which is the engine of the living environment set out in the introduction: the agent’s own forecasts expose its weaknesses, and those weaknesses drive the environment to pose harder tasks. We operationalise it by coupling ATHENA to the task synthesis of PROMETHEUS: ATHENA analyses each batch of trajectories not for what worked but for what failed, distilling each recurring failure mode into a task-synthesis skill that PROMETHEUS uses to generate the next round, so that the environment continuously turns the agent’s own failures into the adversarial challenges it must next overcome. The forecasting trajectory thus feeds its third and final lever, the same record reinforced into parameters and distilled into skills now mined for the failures from which a harder environment is built, closing the recursive loop in which all three axes evolve from one shared substrate.

![](images/f66aaa5306456e679be0f9244e8dbea4ca67b239be3ec1e39780f205346864c2.jpg)  
(a) Where skills help, by task structure.

![](images/1d22c3b41468257dade103bb51ee896f2aeef80a9ec18773ffb0e389d8dcffaf.jpg)  
(b) What skills cost, by model size.  
Figure 8. Skills versus parameter updates: where they help and what they cost. Externalizing experience into skills and internalizing it into parameters act at different stages and carry different costs, so the two axes are complementary rather than interchangeable. Left: forecasting accuracy and the agent’s tool-call budget as the skill library grows, separated into structure synthesis problems that reuse recurring patterns and open synthesis problems from unfamiliar domains. Right: the compute cost of skill extraction against parameter updates as the base model scales up.

The questions synthesized under the evolved skill are measurably harder for the current agent, the signature a weakness-targeting environment should produce. Evaluating the same BRANCHUS checkpoint on questions from the original and the evolved skill, selection accuracy drops from 0.60 to 0.53 and the Brier score worsens from 0.52 to 0.58 (Figure 9a). This added difficulty is not noise: if the evolved questions were mere coin flips, the agent’s confidence would stop separating correct from incorrect answers, and AUROC would collapse toward 0.5. Instead AUROC is essentially unchanged, edging up from 0.678 to 0.68, so the agent’s uncertainty remains informative. By mining the agent’s own trajectories for its failure modes, the environment manufactures genuinely harder yet answerable questions, which is the raw material the next round of self-improvement needs.

A worked example makes concrete what the statistics summarise (Appendix E.1). Running the synthesis pipeline twice over an identical corpus, once with the baseline generator and once after injecting a task-synthesis skill from $S _ { \mathrm { e n v } }$ , isolates the skill’s effect while the source material is held fixed. For a Women’s 77 kg weightlifting final, the baseline keeps the nearly-decided predicate “who wins the gold”, whereas the skill-steered generator reframes the same event around a precise numeric quantity, the gold medallist’s total in kilograms, with the 268 kg standard as a salient threshold, giving a question that remains objective and resolvable from one official page yet is genuinely uncertain. The same sharpening recurs across domains, as a coarse air-quality category band becomes a request for the exact index value. These numerical-forecast and threshold-judgment categories are where BRANCHUS’s error is highest, and the skill-steered batch concentrates its questions there. The environment-level skill thus holds each question’s objectivity and reproducible resolvability fixed and moves only its difficulty into the dimension where the agent is weakest, the mechanism behind the aggregate drop in accuracy and Brier at preserved AUROC reported above.

![](images/fabc3a7f39bd10f982562443a2cd19e41ab17e2060883f8655444984718931a5.jpg)  
(a) Difficulty of synthesized questions.

![](images/ca4af9699dc40aad99e89f5b2f804de970b9d151f7955f4fc93731172ffa084e.jpg)  
(b) Breadth of distilled experience.  
Figure 9. Environment evolution targets genuine weaknesses and enriches the resulting experience. Coupling task synthesis to the agent’s own failure modes turns its weaknesses into harder challenges, and the richer trajectories these produce feed back into the system’s evolution. Left: the difficulty of the synthesized questions under the original versus the evolved skill, showing that evolution raises difficulty while keeping each question well-posed and resolvable. Right: the distilled experience obtained from the same trajectory pool, comparing a weakness-targeting environment against a fixed one in the number of skills and the breadth of their coverage.

## 2.7. A targeted and challenging environment drives sustained system evolution

The harder questions are valuable not in themselves but because the richer experience they generate flows back into both of BRANCHUS’s evolution axes, so a weakness-targeting environment improves the system more than a fixed one does. For the skill axis, the evolved questions probe parts of the task space the agent has not mastered, and the experience distilled from them covers substantially more ground. From the same pool of 262 trajectories, coupling task synthesis to the agent’s weaknesses raises the number of distilled skills from 1312 to 1403 and broadens coverage from 320 to 337 domains and from 331 to 357 concrete tasks (Figure 9b). The gain outlasts the current round, since a skill library spanning more domains is more likely to match the questions the agent meets next, making experience accumulated under an evolving environment more broadly reusable than experience accumulated under a static one.

The parameter axis benefits in the same direction. Relative to plain RFT, RFT on the adversarially synthesized hard problems improves every forecasting metric we track, lowering the Brier score by 1.6%, raising AUROC by 2.7%, and tightening numeric forecasts by 1.0% in relative WIS and 4.0% in Md-sAPE (Figure 10). Because both axes draw on the same enriched experience, the evolved environment lifts the system through the skills it distils and the parameters it updates alike. The agent and its environment thus enter a productive cycle: each round of environment evolution exposes a weakness, and each round of agent evolution repairs it, so capability compounds rather than saturates. This is our clearest evidence that an evolving environment is an essential part of recursive self-improvement rather than an optional add-on, since a fixed distribution of tasks eventually exhausts what it can teach whereas one that keeps distilling challenges from the agent’s own mistakes holds the loop open.

Taken together, the preceding subsections trace a single trajectory as it is turned, in turn, into updated parameters, distilled skills, and a harder environment. No one of these axes is sufficient on its own:

![](images/6ad1e678f5c88b8e0b86fa0c1c1b4764e3b7dd92cd2ddcbb81b3cd8b84406fe1.jpg)  
Figure 10. Fine-tuning on the harder, evolved questions pays off. Across every metric, RFT on the adversarially synthesized hard problems improves over plain RFT. Each row is drawn on its own scale.

parameters and skills internalize only the experience already available, and it is the evolving environment that keeps that experience expanding. It is their joint evolution, grounded in the information-rich trajectories that event forecasting naturally produces, that turns a strong forecaster into one whose capability compounds across rounds, exactly the form of recursive self-improvement set out in the introduction.

## 3. Discussion

This study presents DEEPEVOLVE, a recursive self-improvement framework that treats real-world event forecasting as a living environment the agent learns inside, rather than a static task it is scored against. Across a diverse suite of agent baselines spanning model families and domains, DEEPEVOLVE attains the strongest forecasting performance, matching or surpassing every alternative with clear gains in calibration (Section 2.2). Beyond this static advantage, the system keeps improving from its own experience. Skill evolution shows this most clearly: as each round adds its live trajectories to the pool, the skills distilled from the growing pool lift forecasting quality step by step rather than in a single jump. These results show that an evolving agent inside an evolving environment can sustain improvement where prior approaches saturate, reframing progress from training a static model on a fixed dataset to cultivating a system whose capability accrues from its own experience.

The central contribution of DEEPEVOLVE is a real, dynamic, and verifiable environment that turns forecasting into a usable training signal. This environment couples three modules: a living data source (PROM ETHEUS), an agentic forecaster (BRAN CHUS), and a recursive-evolution engine (ATH ENA). Such an environment is difficult to train on directly, because forecasting data expires once an event resolves, and a question answerable only before resolution loses its training signal afterward. DEEPEVOLVE addresses this by separating evidence gathering from prediction: a time-sensitive search team caches the evidence available at forecast time, and a time-invariant predict team learns over that frozen context, so resolved outcomes supply verifiable rewards while the agent sees only the information available to it at forecast time. Within this environment the agent retains experience through two complementary routes, compressing it into parameters by reinforcement learning on verifiable rewards and externalizing it into an explicit, reusable skill library distilled from forecasting trajectories. Combining the two yields the fast, inspectable gains of skill accumulation together with the deeper gains that only parameter updates reach, a combination that surpasses any single-lever training.

These two routes, one internal and one external, strengthen different capabilities of the agents. Parameter evolution writes experience into the weights: reinforcement learning trains the predict team on cached evidence, so the model draws fuller and better-calibrated conclusions from the information in hand. Skill evolution writes experience into an explicit, reusable library that guides the search team, so the agent gathers evidence more accurately and comprehensively in the first place. The two are complementary rather than redundant: one sharpens reasoning over fixed evidence, the other improves the evidence that reasoning receives. Learning along both routes at once is itself a key technical enabler of recursive self-improvement, because they differ not only in what they strengthen but in what they demand of the experience that drives them (Section 2). Internal learning needs a large volume of trajectories to evolve stably, since reinforcement learning must roll out and refine over the same tasks repeatedly, which makes it hard to apply in a live, contamination-prone environment whose clean, verifiable window is short. External learning is far more stable, as a mere handful of live trajectories already lifts the system within each task’s narrow open window, exactly the regime in which parameter updates run short of data. Running both in tandem thus lets the system keep improving from scarce fresh experience, so the strongest configurations pair a parameter-trained predict team with a mature skill library, and the effective capability of the system is defined jointly by its weights and its accumulated skills.

Both weights and skills draw on one source, the experience the environment supplies, which makes the environment the governing term of the system, the one on which the other two depend. A fixed environment fails this role in two ways. First, it saturates: both routes exhaust the same task distribution, as the predict team resamples evidence it has already mastered and the skill library finds only patterns it has already distilled, so added compute relearns the same regularity and further agent-side training struggles to drive the curve higher. Second, it is prone to contamination: once the outcomes of past events become publicly retrievable, the model can exploit a reasoning shortcut, recalling a resolved answer instead of reasoning toward a forecast. To sustain improvement, the environment itself must therefore evolve, and evolve purposefully: it must stay fresh rather than reuse resolved events, and it must grow harder in a targeted way, posing new tasks at the current edge of the agent’s competence. The dynamic environment we construct meets both demands by drawing new events and scoring them against outcomes that resolve only in the future, which provides a signal that cannot have been seen in advance, to force the agent to genuinely forecast rather than recall and to keep the experience that drives recursive self-improvement renewing round after round.

This evolution is driven by the cooperation between ATHENA and PROMETHEUS. ATHENA distills the failures it finds in the agent’s trajectories into task-synthesis skills, and PROMETHEUS uses them to generate the next round of tasks, so the environment acts as a teacher whose difficulty tracks the agent’s current weaknesses: as the agent improves and its weaknesses shift, the tasks shift with them, and both sides keep changing. Two boundaries constrain this teacher. First, it is only as discerning as its diagnoses: a wrong prediction is an explicit failure that can be turned into a harder task, whereas a silent weakness such as overconfidence stays hidden from this signal, so the environment improves the agent most where its errors are already visible. Second, its pace is set by the world, because a round resolves only after the events it concerns have occurred, so questions that take months to settle make each round of improvement slow and costly.

Several limitations of the present system point to future directions, each bearing on how far recursive self-improvement can run. The first concerns the experience the environment supplies, its fuel: where resolvable, informative events are rare or strongly culture-dependent, trajectories stay thin, so broadening the sources of experience in both quantity and quality is a natural way to raise the ceiling of the recursive loop. The second concerns the skill library as the world drifts, since a skill distilled under one regime may apply differently once the underlying regularity shifts; forgetting, time-weighting, and failure detection would keep the externalized experience reliable, so that improvement accumulates genuine capability. The third concerns the evolution operators themselves: richer forms of parameter evolution that let updates compound round by round as skill accumulation already does, and more expressive forms of skill evolution in how experience is distilled, organized, and retrieved, both offer room to turn the same trajectories into larger gains. Pursuing these directions together is a natural way to strengthen the recursive loop.

The lasting product of this work is a methodology, a technical route for exploring recursive selfimprovement, portable to any domain that is grounded, verifiable, and non-stationary. Carrying it elsewhere asks only for the three terms forecasting supplies here: a living source of tasks whose outcomes resolve over time, an agent that records its trajectories, and an evolution engine that turns those trajectories into updated weights, reusable skills, and harder tasks. Scientific discovery, software engineering, and strategic planning each meet this description. In every such domain the environment commands the process, as an agent climbs only as far as its environment raises the bar, so building environments that keep evolving is the central problem recursive self-improvement must solve. In place of ever larger static models on ever larger static datasets, the alternative is to cultivate evolving agents within evolving environments, setting the ceiling on progress by the openness of the environment rather than the size of a snapshot. DEEPEVO LVE is an early demonstration that this path can be walked, moving the measure of intelligence from what a system can do at a single moment to whether it can grow stronger through its own experience, observably and reproducibly in a real and verifiable world.

## 4. Approach

## 4.1. Overview

DEEPEVOLVE is a closed loop in which an agent and its environment improve one another over time, realized by three modules (Figure 1), rather than a feed-forward pipeline from input to prediction. PROMETHEUS (Section 4.2), the living data environment, turns real-world news into forecasting tasks whose ground truth is revealed only once the events resolve, supplying both the tasks and an objective, time-delivered reward. BRANCHUS (Section 4.3), the agentic forecaster, gathers evidence with external tools and reasons over it to issue a probabilistic prediction, producing both an answer and a full trajectory. ATHENA (Section 4.4), the recursive-evolution engine, converts these trajectories back into capability: it updates BRANCHUS’s parameters, distills its reasoning into reusable skills, and analyzes its failures into task-synthesis skills that make PROMETHEUS pose harder, weakness-targeted questions. The three thus form a single evolving system, in which a stronger BRANCHUS meets a steadily more demanding PROMETHEUS, while the world remains the external ground of every reward.

Notation and Problem Setup. We make this loop precise before instantiating its three modules. BRANCHUS is a policy $\pi _ { \theta }$ with parameters � that, given a forecasting question � and a repository of forecasting skills S, produces a probabilistic prediction together with the reasoning trajectory that supports it,

$$
( \hat { p } , \tau ) \sim \pi _ { \theta } ( \cdot \mid q , S ) ,\tag{1}
$$

where $\hat { p }$ is a distribution over the possible outcomes of � and � records the evidence gathered, the hypotheses entertained, and the intermediate decisions taken. PROMETHEUS is the environment E: at round � it emits a batch of questions ${ \bar { Q } } ^ { ( t ) }$ , and once the underlying events resolve it reveals, for each �, an objective outcome $y _ { q }$ that defines a verifiable reward

$$
r ( q , \hat { p } ) \ = \ \mathbf { 1 } \Big [ \mathop { \mathrm { a r g } } \mathop { \operatorname* { m a x } } _ { o } \hat { p } ( o ) = y _ { q } \Big ] \in \{ 0 , 1 \} ,\tag{2}
$$

which checks whether the outcome the model deems most likely matches the resolved outcome $y _ { q } .$ , so that time rather than human annotation supplies supervision. We summarize the state of the system at round � by the triple

$$
\Sigma ^ { ( t ) } = \big ( \theta ^ { ( t ) } , S ^ { ( t ) } , \mathcal { E } ^ { ( t ) } \big ) ,\tag{3}
$$

whose three components are the agent’s parameters, its accumulated skill library, and the environment that supplies its tasks. ATHENA advances this state through three evolution operators, each learning from the trajectory pool ${ \mathcal { T } } ^ { ( t ) } = \{ \tau \}$ collected over ${ \bar { Q } } ^ { ( t ) }$ : a parameter update $\Phi _ { \mathrm { r l } }$ , a skill update $\Phi _ { \mathrm { s k i l l } }$ , and an

Algorithm 1 DEEPEVOLVE: one evolution round   
Input: state $\Sigma ^ { ( t ) } = ( \theta ^ { ( t ) } , S ^ { ( t ) } , \mathcal { E } ^ { ( t ) } )$ , forecast horizon �   
Output: updated state $\Sigma ^ { ( t + 1 ) }$   
⊲ Task synthesis (PROMETHEUS)   
1: ${ \mathcal { D } } ^ { ( t ) } \gets \mathbf { C } \mathbf { R A W L } ( { \mathcal { E } } ^ { ( t ) } )$   
// crawl web pages: calendars,feeds, developing news   
2: $\tilde { Q } ^ { ( t ) } \gets \mathbf { G } \mathrm { E N } ( \mathcal { \hat { D } } ^ { ( t ) } , S _ { \mathrm { e n v } } ^ { ( t ) } )$   
// synthesize question drafts $q = ( e , o , f , t _ { \mathrm { c u t } } , t _ { \mathrm { d l } } )$   
3: ${ \cal Q } ^ { ( t ) } \gets s$ ELECT ◦ FILTER $( \tilde { Q } ^ { ( t ) } )$   
// temporal/dedup/conflictfiltering, diversity-aware selection   
⊲ Rollout (BRANCHUS)   
4: $\mathcal T ^ { ( t ) } \gets \{ ( \hat { p } , \tau ) \sim \pi _ { \theta ^ { ( t ) } } ( \cdot \mid q , S ^ { ( t ) } ) : q \in Q ^ { ( t ) } \}$   
5: observe outcomes $y _ { q }$ and rewards $r ( q , \hat { p } )$ as events resolve within �   
⊲ Evolution (ATHENA)   
6: $\theta ^ { ( t + 1 ) } \gets \Phi _ { \mathrm { r l } } ( \theta ^ { ( t ) } , \mathcal { T } ^ { ( t ) } )$   
//parameter level: RLVR   
7: $S ^ { ( t + 1 ) } \gets \Phi _ { \mathrm { s k i l l } } ( S ^ { ( t ) } , \mathcal { T } ^ { ( t ) } )$   
// behavior level: skill distillation   
8: $\mathcal { E } ^ { ( t + 1 ) } \gets \Phi _ { \mathrm { e n v } } ( \mathcal { E } ^ { ( t ) } , \mathcal { T } ^ { ( t ) } )$   
// environment level: adversarial synthesis   
9: return $\Sigma ^ { ( t + 1 ) } = ( \theta ^ { ( t + 1 ) } , S ^ { ( t + 1 ) } , \mathcal { E } ^ { ( t + 1 ) } )$

environment update $\Phi _ { \mathrm { e n v } }$ (Section 4.4). Writing the rollout of $\pi _ { \theta ^ { ( t ) } }$ on ${ \bar { Q } } ^ { ( t ) }$ as $\mathcal { T } ^ { ( t ) }$ , one round of recursive self-improvement is the update

$$
\begin{array} { r } { \boldsymbol { \theta } ^ { ( t + 1 ) } = \Phi _ { \mathrm { r l } } \big ( \boldsymbol { \theta } ^ { ( t ) } , \mathcal { T } ^ { ( t ) } \big ) , \quad \boldsymbol { S } ^ { ( t + 1 ) } = \Phi _ { \mathrm { s k i l l } } \big ( \boldsymbol { S } ^ { ( t ) } , \mathcal { T } ^ { ( t ) } \big ) , \quad \boldsymbol { \mathcal { E } } ^ { ( t + 1 ) } = \Phi _ { \mathrm { e n v } } \big ( \boldsymbol { \mathcal { E } } ^ { ( t ) } , \mathcal { T } ^ { ( t ) } \big ) , } \end{array}\tag{4}
$$

which maps $\Sigma ^ { ( t ) } \mapsto \Sigma ^ { ( t + 1 ) }$ and makes the products of one round the raw materials of the next (Algorithm 1). The word recursive refers to this composition: the triple co-evolves rather than converging to a fixed set of weights, because $\Phi _ { \mathrm { e n v } }$ keeps generating a next batch $\boldsymbol { Q } ^ { ( t + 1 ) }$ matched to the capability of the updated agent $\pi _ { \theta ^ { ( t + 1 ) } }$

## 4.2. Prometheus: A Living Data Environment

PROMETHEUS constructs batches of objective, short-horizon prediction questions through two complementary mechanisms. In template-based generation, source-specific adapters retrieve public calendars, schedules, feeds, and structured pages, identify upcoming events, and instantiate predefined question templates. In agent-driven open discovery, CODEX [17] gathers information from relevant websites, interprets scheduled events and developing stories, and constructs structured question drafts. Its sources span many domains, including sports, economics, monetary policy, energy, weather, public health, technology, government affairs, and cryptocurrency. Candidates from both routes enter a shared construction and publication pipeline, which we formalize next. Formally, each candidate question is a tuple

$$
q ~ = ~ \left( e , o , f , t _ { \mathrm { c u t } } , t _ { \mathrm { d l } } \right) ,\tag{5}
$$

pairing an identifier � for the underlying real-world event with an identifier � for the specific outcome being asked about (such as the winner of a match or its total score), an answer format $f ,$ , a generation time $t _ { \mathrm { c u t } }$ that marks when the question is created, and a forecast deadline $t _ { \mathrm { d l } }$ . The pipeline proceeds in three stages. It first crawls the web pages exposed by the current environment, ${ \mathcal { D } } ^ { ( t ) } = { \mathrm { C } } \mathrm { { R A W L } } ( { \mathcal { E } } ^ { ( t ) } )$ , yielding a corpus $\mathcal { D } ^ { ( t ) }$ of calendars, feeds, and developing news from which questions can be posed. A generator then turns this corpus into a set of candidate drafts, $\tilde { \mathbfcal { Q } } ^ { ( t ) } = \mathbf { G } \mathrm { E N } ( \mathcal { D } ^ { ( t ) } , S _ { \mathrm { e n v } } ^ { ( t ) } )$ , where each draft is a tuple of the form in Equation (5) and $S _ { \mathrm { e n v } } ^ { ( t ) }$ is the library of task-synthesis skills that ATHENA feeds back to steer generation toward BRA NCHU S’s weaknesses (Section 4.4). A filter-and-select map finally yields the published batch, so that the pipeline is the composition

$$
\begin{array} { r } { \boldsymbol { Q } ^ { ( t ) } = \mathrm { S E L E C T } \circ \mathrm { F I L T E R } \left( \mathrm { G E N } \left( \boldsymbol { \mathcal { D } } ^ { ( t ) } , \boldsymbol { S } _ { \mathrm { e n v } } ^ { ( t ) } \right) \right) , } \end{array}\tag{6}
$$

whose stages we now describe.

Validation and Temporal Filtering. The pipeline first identifies events whose outcomes are expected to become observable within the requested time horizon �, measured from the start of question generation. Once the generation time $t _ { \mathrm { c u t } }$ is recorded, each candidate is reassessed for temporal eligibility and is retained only if its deadline and expected resolution time $t _ { \mathrm { r e s } }$ satisfy

$$
t _ { \mathrm { c u t } } ~ < ~ t _ { \mathrm { d l } } ~ \mathrm { a n d } ~ t _ { \mathrm { c u t } } ~ < ~ t _ { \mathrm { r e s } } ( q ) ~ \leq ~ t _ { \mathrm { c u t } } + H ,\tag{7}
$$

so that the answer remains open at generation time and resolves within the requested horizon.

Deduplication and Conflict Detection. The remaining drafts undergo duplicate removal and conflict detection. Each draft receives a reproducible identifier id $( q )$ derived from the underlying event $e ,$ the result � being predicted, the answer format $f ,$ and the relevant time boundaries. Drafts whose id(�) already appears in the question bank are excluded. Within the current batch, drafts that share the same $( e , o )$ are compared: when their structured attributes agree, one representative is retained despite differences in wording; when they disagree, the group is discarded to keep the published versions of a question consistent.

Diversity-Aware Selection. The remaining candidates are selected to promote diversity across events, domains, sources, and question types. The selector first favors events not yet represented in the batch. It then considers a second question for an already represented event, up to a maximum of two questions per event. Among otherwise comparable candidates, it favors domains and sources with fewer questions in the historical bank and current batch, followed by question types and answer formats that occur less frequently in the current batch, and then earlier forecast deadlines. This procedure promotes a diverse set of questions, though the final distribution remains constrained by the number of eligible questions available from each domain and source.

## 4.3. Branchus: Agentic Forecasting System

BRANCHUS is the agentic forecaster that answers the questions posed by PROMETHEUS: given a question, it gathers evidence from the outside world, reasons over it, and issues a probabilistic prediction, recording the full trajectory that leads to each answer. Two mechanisms underpin its design. A multi-agent team organization separates collecting evidence from turning evidence into a prediction, so each part can be scaled and improved independently. A skill management mechanism equips the agents with an evolving repository of forecasting knowledge, from hand-authored seeds to knowledge distilled from past experience. Together these make BRANCHUS both a strong forecaster and a producer of the structured trajectories that ATHENA later learns from.

Multi-Agent Teams. BR ANCH US factors the policy of Equation (1) into two cooperating teams, a search team $\pi _ { \theta } ^ { \mathrm { s } }$ and a predict team $\pi _ { \theta } ^ { \mathrm { p } }$ , organized by whether their work depends on when it is done. The search team performs search: it collects and vets evidence relevant to a question, and consists of a searcher, a scheduler, and a critic. Its work is time-sensitive, because the available evidence changes as an event approaches and, once the event resolves, its answer becomes directly retrievable, so the same search yields a different result depending on when it is run. The scheduler decomposes the question and dispatches sub-agents, the searcher retrieves information from external sources, and the critic assesses the quality, relevance, and reliability of what has been gathered. The model invokes these sub-agents on its own as needed, iterating between retrieval and judgement until the evidence is sufficient, and every step is retained as the search trajectory $\tau _ { s } \sim \pi _ { \theta } ^ { s } ( \cdot \mid q , S )$ . The predict team performs prediction, turning this evidence into a forecast. Its work is time-invariant, because it reasons only over evidence already gathered: given the same evidence, it produces the same forecast regardless of when it runs. Rather than re-searching, it calls a local tool to access the search team’s cached reasoning trajectories, which lets it aggregate a set $\{ \tau _ { \mathrm { s } } ^ { ( j ) } \} _ { j = 1 } ^ { m }$ of such trajectories into a single view of the evidence and reconcile the partial conclusions they contain. From this aggregated evidence it produces the final probabilistic answer,

$$
\hat { p } \sim \pi _ { \theta } ^ { \mathrm { p } } \big ( \cdot \mid q , \{ \tau _ { s } ^ { ( j ) } \} _ { j = 1 } ^ { m } , S \big ) ,\tag{8}
$$

which recovers Equation (1) with $\tau = ( \{ \tau _ { \mathrm { s } } ^ { ( j ) } \} _ { j } , \tau _ { \mathrm { p } } )$ . This division of labour keeps evidence gathering and prediction loosely coupled: the search team explores broadly and in parallel, while the predict team concentrates the resulting trajectories into a calibrated judgement.

Skill Management. BRAN CHUS draws on a repository of forecasting skills S, compact pieces of procedural knowledge that guide how a question is approached. Skills are of two kinds. Task-specific skills are authored by hand and form a seed set $\textstyle S _ { 0 } !$ : each encodes knowledge about a particular class of question, including important source websites to consult and prompts that steer the reasoning, giving the agent a useful starting competence before any experience is accumulated. Experience-distilled skills, by contrast, are consolidated from BRANCHUS’s own trajectories by ATHENA and grow as the system evolves, so that $S ^ { ( t ) } = S _ { 0 } \cup \bigcup _ { t ^ { \prime } < t } \Phi _ { \mathrm { s k i l l } } ( S ^ { ( t ^ { \prime } ) } , \mathcal { T } ^ { ( t ^ { \prime } ) } )$ . To make skills easy to organize, retrieve, and reuse, we structure each skill as a four-level tuple $\boldsymbol { s } = \left( s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 } \right)$ ordered from the abstract to the concrete, so that a general forecasting principle $s _ { 1 }$ at the top specializes progressively down to a concrete procedure or source $s _ { 4 }$ at the bottom. At inference time, an agent assigned role � to question � retrieves a relevant subset and keeps, for each matched skill, the most specific level that still applies,

$$
\begin{array} { r l } { \mathrm { R } \mathrm { E T R I E V E } ( q , a , S ) } & { = \Bigl \{ s _ { \ell ^ { \star } ( s ) } : s \in S , \ \mathrm { M A T C H } ( s , q , a ) \Bigr \} , \qquad \ell ^ { \star } ( s ) = \operatorname* { m a x } \{ \ell : s _ { \ell } \mathrm { a p p l i e s } \ \mathrm { t o } \ q \} . } \end{array}\tag{9}
$$

This layered representation lets BR ANCH US match a new question to skills at the appropriate level of generality, so that a question without an exact precedent still benefits from a broader level, and lets the repository hold both broad heuristics and highly specific recipes within a single consistent schema.

## 4.4. Athena: Recursive Evolution System

ATH ENA is the recursive-evolution engine of DEEPEVOLVE: it extracts experience from the trajectory pool $\mathcal { T } ^ { ( t ) }$ that BRA NCHU S produces and turns it into improvement through the three operators of Equation (4). At the parameter level, $\Phi _ { \mathrm { r l } }$ applies reinforcement learning to BRANC HUS’s predictions to update its weights �. At the behaviour level, $\Phi _ { \mathrm { s k i l l } }$ distils reusable skills from its trajectories to strengthen the agent while leaving those weights untouched. At the environment level, $\Phi _ { \mathrm { e n v } }$ synthesizes adversarial skills from its failures to make PROMETHEUS generate harder, weakness-targeted tasks. The first two operators evolve the agent and the third evolves the environment, so that both improve from the same stream of experience.

Parameter-Level Evolution: Reinforcement Learning on Branchus’s Predictions. At the parameter level, $\Phi _ { \mathrm { r l } }$ improves BRANCHUS by reinforcement learning, using Group Relative Policy Optimization (GRPO) [23] on questions synthesized by PROMETHEUS. Training proceeds in two stages that separate evidence gathering from parameter updates. First, the search team runs over these questions and its reasoning trajectories $\{ \tau _ { \mathrm { s } } \}$ are cached; once enough have accumulated, we perform reinforcement learning with verifiable rewards (RLVR) on the predict team alone, using the cached evidence. Training only the predict team on fixed, pre-collected trajectories rules out answer leakage, since the model can no longer shortcut the task by re-collecting evidence that trivially entails the answer; yet because the evidence rather than the answer is fixed, the predict team still explores freely, sampling � candidate predictions $\{ \hat { p } _ { i } \} _ { i = 1 } ^ { G }$ from the same evidence and updating on their outcomes. Concretely, each sampled prediction receives the verifiable reward of Equation $( 2 ) , r _ { i } = r ( q , \hat { p } _ { i } ) \in \{ 0 , 1 \}$ , which is turned into a group-relative advantage by normalizing within the sampled group,

$$
A _ { i } \ = \ { \frac { r _ { i } - \mathrm { m e a n } \big ( \{ r _ { j } \} _ { j = 1 } ^ { G } \big ) } { \mathrm { s t d } \big ( \{ r _ { j } \} _ { j = 1 } ^ { G } \big ) } } ,\tag{10}
$$

and the predict team is updated by maximizing the clipped GRPO objective over the fixed evidence,

$$
\Phi _ { \mathrm { r l } } : ~ \theta ^ { ( t + 1 ) } = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { q \sim Q ^ { ( t ) } , \{ \hat { p } _ { i } \} \sim \pi _ { \theta ^ { ( t ) } } ^ { \mathrm { P } } } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } \Bigl ( w _ { i } ( \theta ) A _ { i } , ~ \mathrm { c l i p } \bigl ( w _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon \bigr ) A _ { i } \Bigr ) \right] ,\tag{11}
$$

where $w _ { i } ( \theta ) = \pi _ { \theta } ^ { \mathrm { p } } ( \hat { p } _ { i } \mid q , \{ \tau _ { \mathrm { s } } ^ { ( j ) } \} , S ) / \pi _ { \theta ^ { ( t ) } } ^ { \mathrm { p } } ( \hat { p } _ { i } \mid \cdot )$ is the importance ratio and � the clipping range. The resolved ground truth $y _ { q }$ from PRO METH EUS supplies the reward signal in Equation (2) without human annotation. This regime motivated the multi-agent design: by isolating prediction from search, the two-team organization makes it possible to reinforce the predict team on cached evidence, so that the measured improvement reflects better reasoning over the same information rather than opportunistic re-collection.

Behavior-Level Evolution: Distilling Reusable Skills from Branchus’s Trajectories. At the behavior level, $\Phi _ { \mathrm { s k i l l } }$ turns B R A N C H U S’s reasoning trajectories into reusable skills. These skills guide the agent at inference time and improve how it works while leaving its weights fixed. Formally, $\Phi _ { \mathrm { s k i l l } }$ maps the trajectory pool to a set of new skills, $\Phi _ { \mathrm { s k i l l } } ( S ^ { ( t ) } , { \mathcal { T } } ^ { ( t ) } ) = S ^ { ( t ) } \cup \Delta S ^ { ( t ) }$ , through the composition $\Delta S ^ { ( t ) } = ( \mathrm { O R G A N I Z E }$ ◦ FILTER ◦ EX TRACT ◦ EVAL) (T <sup>(�)</sup>). In the first stage, a judge model EVA L scores the scheduler, searcher, and critic of the search team against role-specific rubrics. The resolved outcome $y _ { q }$ is withheld from the judge, because forecast correctness is an imperfect proxy for reasoning quality. In the second stage, EXTRACT uses these scores to identify positive experiences from helpful but non-obvious actions, and negative experiences from recurring mistakes or failure modes. Separating evaluation from extraction lets each step focus on a single judgment. A FILTER then discards weak experiences. It removes those that merely restate a rubric, lack support in the trajectory, or only prescribe which evidence to use rather than how to forecast. In the third stage, ORGANIZE grades each surviving experience into a four-level skill $\boldsymbol { s } = \left( s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 } \right)$ : it keeps the concrete procedure applied in the trajectory verbatim as the most specific level $s _ { 4 } ,$ and the three levels above it generalize that procedure step by step, first to the question type, then within its domain, and finally to a cross-domain principle $s _ { 1 }$ . Each level stays within its role’s responsibility and remains independently applicable, so the skill can be matched at whichever level of abstraction fits a new question. At inference time an agent retrieves skills by Equation (9), so a question without an exact precedent still benefits from a broader level. Because distillation needs no gradient update, � stays fixed and only $s$ changes, in contrast to $\Phi _ { \mathrm { r l } }$ . This makes distillation cheap enough to repeat after every round.

Environment-Level Evolution: Adversarial Skill Synthesis Targeting Branchus’s Weaknesses. At the environment level, $\Phi _ { \mathrm { e n v } }$ turns feedback from past forecasts into task-synthesis skills $S _ { \mathrm { e n v } }$ . These skills guide how PROMETHEUS generates later questions, which reshapes the environment $\mathcal { E } ^ { ( t + 1 ) }$ and hence the next batch $\boldsymbol { Q } ^ { ( t + 1 ) }$ . They specify what types of questions to construct and how to construct them, rather than how BRANCHUS should answer them. The goal is adversarial: expose categories of questions on which BRANCHUS has struggled, while keeping every task clear, objectively verifiable, and reproducibly resolvable. The update starts from the resolved triples $\{ ( q , \hat { p } , y _ { q } ) \}$ in $\mathcal { T } ^ { ( t ) }$ and their rewards. We map each question to a category $c = \operatorname { C A T } ( q )$ by answer type and task feature, and aggregate the evaluable predictions into an error profile

$$
\rho ^ { ( t ) } ( c ) \ = \ \mathbb { E } _ { ( q , \hat { p } , y _ { q } ) : \operatorname { c A T } ( q ) = c } \left[ 1 - r ( q , \hat { p } ) \right] ,\tag{12}
$$

the mean error rate for category �. We read this profile as evidence of category-level difficulty rather than as a direct diagnosis of a reasoning failure, since a single error may simply reflect the intrinsic uncertainty of an event. The profiled categories include numerical forecasts, market prices, dynamic rankings, threshold judgments, and high-variance events. The categories with the largest error, arg max<sub>�</sub> $\rho ^ { ( t ) } ( c )$ , mark where BRANCHUS is least reliable, and this signal directs where to construct new questions. ATHENA then turns the signal into reusable construction guidance $S _ { \mathrm { e n v } } ^ { ( t ) } = \mathrm { S Y N T H E S I Z E } \bar { ( } \rho ^ { ( t ) } )$ . Each synthesized skill specifies a target category, the conditions under which it applies, how to discover relevant events, rules for building questions, and quality constraints. These skills enter the generator, so that GEN in $\mathcal { E } ^ { ( t + 1 ) }$ leans toward the identified weaknesses. After the new questions are forecast and their outcomes become available, we update the error profile of Equation (12) and repeat the synthesis. This loop lets PROMETHEUS adapt the task distribution ${ \bar { Q } } ^ { ( t ) }$ to BRANCHUS’s observed weaknesses across rounds, which keeps the recursion of Equation (4) from saturating.

## 4.5. Models and Implementation

Our evaluation pairs established agent harnesses with shared base LLMs as strong external baselines, and adds in-framework comparisons that isolate the core designs of DEEPEVOLVE. To make these comparisons fair, we enforce implementation parity. Every harness runs on the same PROMETHEUS questions, receives the same external tools under identical tool descriptions and a unified tool-calling format, and uses its own official prompt. The only factors that vary across conditions are the harness design and the base LLM. We avoid cross-paradigm comparisons that would break this parity, such as static-corpus question answering or dialogue, where evidence is already available rather than gathered under uncertainty from an unfolding world. This keeps the gains reported in Section 2.2 attributable to the design of DEEPEVOLVE rather than to mismatched task definitions. We now give the three concrete settings behind this design.

Base LLMs and Agent Harnesses. We evaluate five representative frontier LLMs that span different model families and capabilities: GPT-5.5, GPT-5.6-Sol, Claude-Opus-4.6, DeepSeek-V4-Pro, and Qwen3.8-Max. This range lets us test whether forecasting performance and evaluation robustness hold across model families rather than for a single base model. We pair each base LLM with six agent harnesses: ReAct, MAS, OpenClaw, DeerFlow, MiroFlow, and BRANCHUS, our purpose-built harness for real-world event forecasting. These harnesses cover a broad range of designs: a lightweight reasoning-and-tool-use loop (REACT [27]), a multi-agent system (MAS), and three deep-research pipelines (OPENCLAW [18], DEERFLOW, and MIROFLOW), spanning general-purpose reasoning frameworks and search-oriented systems. This model×harness grid separates the contribution of the base LLM from that of the harness, and shows how harness design affects open-world forecasting across base models.

Prompt Specification. For each harness, we use its official prompt as released by the original authors, rather than writing our own. This way, every framework runs in the configuration it was designed and tuned for, so performance differences reflect the harness design rather than our prompt engineering. The complete prompt templates for the three modules of DEEPEVOLVE are provided in Appendix D.

External Tool Settings. We give the models a web search tool that issues queries and retrieves content, which grounds their forecasts in up-to-date evidence. For PRO METH EUS, we adopt the CODEX [17] harness and use its built-in tools to crawl news from relevant web pages, which supplies the raw material for building questions. We also design three local tools that expose the answer, the reasoning trajectory, and the collected evidence of the search team to the predict team of BRANCHUS. With these tools, the predict team reuses the search team’s outputs without issuing new external searches, so it reasons over exactly the evidence already collected.

## References

1 Fei Bai, Yingqian Min, Beichen Zhang, Zhipeng Chen, Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan Wang, and Hongteng Xu. Towards effective code-integrated reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 30022–30030, 2026.

2 Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosiute, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemí Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional AI: harmlessness from AI feedback. CoRR, abs/2212.08073, 2022. doi: 10.48550/ARXIV.2212.08073. URL https://doi.org/10 .48550/arXiv.2212.08073.

3 Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

4 Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott M. Lundberg, Harsha Nori, Hamid Palangi, Marco Túlio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with GPT-4. CoRR, abs/2303.12712, 2023. doi: 10.48550/ARXIV.2303.12712. URL https://doi.org/10.485 50/arXiv.2303.12712.

5 Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, et al. An empirical study on eliciting and improving r1-like reasoning models. arXiv preprint arXiv:2503.04548, 2025.

6 Neil Chowdhury, James Aung, Chan Jun Shern, Oliver Jaffe, Dane Sherburn, Giulio Starace, Evan Mays, Rachel Dias, Marwan Aljubeh, Mia Glaese, Carlos E. Jimenez, John Yang, Leyton Ho, Tejal Patwardhan, Kevin Liu, and Aleksander Madry. Introducing swe-bench verified, August 2024. URL https://openai.com/index/introducing-swe-bench-verified/.

7 Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models. J. Mach. Learn. Res., 25:70:1–70:53, 2024. URL https://jmlr.org/papers/v25/23-0870.html.

8 Jeff Clune. Ai-gas: Ai-generating algorithms, an alternate paradigm for producing general artificial intelligence. CoRR, abs/1905.10985, 2019. URL http://arxiv.org/abs/1905.10985.

9 DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025. doi: 10.48550/ARXIV.2501.12948. URL https://doi.org/10 .48550/arXiv.2501.12948.

10 Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. Promptbreeder: Self-referential self-improvement via prompt evolution. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings ofMachine Learning Research, pages 13481–13544. PMLR / OpenReview.net, 2024. URL https://proceedings.mlr.press/v235/fernando24a.html.

11 Danny Halawi, Fred Zhang, Yueh-Han Chen, and Jacob Steinhardt. Approaching human-level forecasting with language models. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc /paper\_files/paper/2024/hash/5a5acfd0876c940d81619c1dc60e7748-Abs tract-Conference.html.

12 Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, 2015. URL http://arxiv.org/abs/1503.02531.

13 Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 1051–1068. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.67. URL https://doi.org/10.18653/v1/ 2023.emnlp-main.67.

14 Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tülu 3: Pushing frontiers in open language model post-training. CoRR, abs/2411.15124, 2024. doi: 10.48550/ARXIV.2411.15124. URL https://doi.org/10.48550/arXiv.2411.15124.

15 Brandon Shuen Yi Loke, Filippo Quadri, Gabriel Vivanco, Maximilian Casagrande, and Saúl Fenollosa. Overcoming catastrophic forgetting in neural networks. CoRR, abs/2507.10485, 2025. doi: 10.48550 /ARXIV.2507.10485. URL https://doi.org/10.48550/arXiv.2507.10485.

16 Mike A Merrill, Alexander Glenn Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Jenia Jitsev, Marianna Nezhurina, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jeffrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Kumar Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighoff, Robert Kwesi Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=a7Qa4CcHak.

17 OpenAI. Codex: Ai coding partner from openai. https://openai.com/codex/, 2026. Product page. Accessed: 2026-04-29.

18 OpenClaw. OpenClaw: Personal ai assistant. https://github.com/openclaw/openclaw, 2026. GitHub repository. Accessed: 2026-04-29.

19 Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/b1efde53be364a7 3914f58805a001731-Abstract-Conference.html.

20 German Ignacio Parisi, Ronald Kemker, Jose L. Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. Neural Networks, 113:54–71, 2019. doi: 10.1016/J.NEUNET.2019.01.012. URL https://doi.org/10.1016/j.neunet.2019.01 .012.

21 Jürgen Schmidhuber. Gödel machines: Fully self-referential optimal universal self-improvers. In Ben Goertzel and Cassio Pennachin, editors, Artificial General Intelligence, Cognitive Technologies, pages 199–226. Springer, 2007. doi: 10.1007/978-3-540-68677-4\_7. URL https://doi.org/10.1 007/978-3-540-68677-4\_7.

22 John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

23 Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

24 Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_f iles/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Con ference.html.

25 David Silver and Richard S Sutton. Welcome to the era of experience. Google DeepMind Blog, 2025.

26 Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning. arXiv preprint arXiv:2503.05592, 2025.

27 Shunyu Yao, Jeffrey Zhao, Dian Yu, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In NeurIPS 2022 Foundation Models for Decision Making Workshop, 2022.

28 Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. Star: Bootstrapping reasoning with reasoning. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022,

29 Zhiyuan Zeng, Jiashuo Liu, Siyuan Chen, Tianci He, Yali Liao, Yixiao Tian, Jinpeng Wang, Zaiyuan Wang, Yang Yang, Lingyue Yin, Mingren Yin, Zhenwei Zhu, Tianle Cai, Xinjie Chen, Zehui Chen, Jiecao Chen, Yantao Du, Xiang Gao, Jiacheng Guo, Liang Hu, Jianpeng Jiao, Xiangsheng Li, Jingkai Liu, Shuang Ni, Zhoufutu Wen, Ge Zhang, Kaiyuan Zhang, Xin Zhou, Wenhao Huang, Jose Blanchet, Xipeng Qiu, and Mengdi Wang. Futurex: An advanced live benchmark for LLM agents in future prediction. In Carl Vondrick, Bharath Hariharan, Colin Raffel, Lerrel Pinto, Diyi Yang, and Aleksandra Faust, editors, The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc, 2026. URL https://proceedings. iclr.cc/paper\_files/paper/2026/hash/21315abf210cef71317e887e9cda78 b3-Abstract-Conference.html.

30 Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, and Dan Hendrycks. Forecasting future world events with neural networks. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/aec870a6772336c 15dac992c16f2e7c9-Abstract-Datasets\_and\_Benchmarks.html.

## A. Full FutureX Overall Leaderboards

For completeness, we reproduce here the full official FutureX overall leaderboards corresponding to the four weekly evaluations summarized in Figure 2. Each leaderboard reports the sliding-window Bradley– Terry rating of every participating harness for the week in question, together with the accompanying per-model statistics released by the benchmark. Whereas Figure 2 retains only the harnesses relevant to our comparison, the tables below list the complete field of entrants, allowing the position of BRANCHUS to be read against the entire leaderboard rather than the reduced set. Across the first three weeks BRANCHUS attains the top overall rating; in the final week it remains among the strongest entrants, and we discuss the factors behind this week-to-week variation in Section 2.

<table><tr><td rowspan=1 colspan=1>Overall rankings use a sliding      Terrycalculation across the latest eight weeks. An entrymust have results in at least three of those weeks to be listed.Captured: 2026-08-17 02:47:35 CSTRows: 22Source: futurex.liveAll models shown after expanding &quot;View results for all models* on the Overall tab.</td></tr><tr><td rowspan=1 colspan=1>RANK MODEL        FRAMEWORK       SUBMISSION TYPE AVO SCORE PEAK SCORE WEEKS EVALUATED BT RATINO</td></tr><tr><td rowspan=1 colspan=1>¥ #1 GLM-5.1       Branchus-beta      Self-Submitted   53.59   59.90       5   1.15</td></tr><tr><td rowspan=1 colspan=1>è #2 Apodex-1.0-mini    Apodex-FutureFlow-alpha  Self-Submitted   51.71   59.17       7   0.39</td></tr><tr><td rowspan=1 colspan=1>è #3 Qwen-3-A3B     MurMur         Self-Submitted   51.67   55.80      8  0.31</td></tr><tr><td rowspan=1 colspan=1>#4  claude_sonnet_4.6   H20_Al_Super_Agent_v1.82 Self-Submitted   51.93  57.96      8   0.30</td></tr><tr><td rowspan=1 colspan=1>#5  DeepSeek-V4-Flash  MiroCorgi        Self-Submitted   50.59  58.49      7   0.19</td></tr><tr><td rowspan=1 colspan=1>#6  H-Base-3.0      HAgent-V4        Self-Submitted   50.58   56.35       4   0.09</td></tr><tr><td rowspan=1 colspan=1>#7  glm5.1-w4a8     DragonAgent-0525     Self-Submitted   46.54   51.05       4   0.05</td></tr><tr><td rowspan=1 colspan=1>#8  H-Base-3.0      HAgent-V1        Self-Submitted   50.09   57.91       4  0.05</td></tr><tr><td rowspan=1 colspan=1>#9  GPT-5.5-High     Search          FutureX Hosted   44.74  49.19       4   0.02</td></tr><tr><td rowspan=1 colspan=1>#10 GPT-5.4-high     Search         FutureX Hosted   44.30   50.19       4  0.01</td></tr></table>

(a) Week of 2026-08-17.

<table><tr><td colspan="7">Overall rankings use a sliding-window Bradley- -Terry calculation eight weeks. An entry must have results in at least three of those weeks to be listed. Captured: 2026-08-26 04:50:50 CST Rows: 11 Source: futurex.live</td></tr><tr><td colspan="7">All models shown after expanding &quot;View results for all models&quot; on the Overall tab.</td></tr><tr><td>RANK MODEL</td><td>FRAMEWORK</td><td></td><td>SUBMISSION TYPE</td><td>AVG SCORE</td><td>PEAK SCORE WEEKS EVALUATED</td><td>BT RATING</td></tr><tr><td>I #1</td><td>GLM-5.1</td><td>Branchus-beta</td><td>Self-Submitted</td><td>56.90</td><td>59.90</td><td>4 1.03</td></tr><tr><td>é #2</td><td>DeepSeek-V4-Pro</td><td>JSearch_V1_S2</td><td>Self-Submitted</td><td>55.73</td><td>59.69</td><td>4 0.40</td></tr><tr><td>à #3</td><td>Apodex-1.0-mini</td><td>Apodex-FutureFlow-alpha</td><td>Self-Submitted</td><td>53.73</td><td>62.05</td><td>7 0.39</td></tr><tr><td>#4</td><td>DeepSeek-V4-Pro</td><td>JSearch_V1_S1</td><td>Self-Submitted</td><td>55.53</td><td>59.24</td><td>4 0.31</td></tr><tr><td>#5</td><td>DeepSeek-V4-Pro</td><td>deerflow</td><td>Self-Submitted</td><td>53.51</td><td>57.52</td><td>4 0.26</td></tr><tr><td>#6</td><td>Qwen-3-A3B</td><td>MurMur</td><td>Self-Submitted</td><td>53.47</td><td>61.46</td><td>8 0.24</td></tr><tr><td>#7</td><td>DeepSeek-V4-Flash</td><td>MiroCorgi</td><td>Self-Submitted</td><td>53.25</td><td>61.35</td><td>7 0.18</td></tr><tr><td>#8</td><td>claude_sonnet_4.6</td><td>H20_AI_Super_Agent_v1.82</td><td>Self-Submitted</td><td>53.23</td><td>57.96</td><td>8 0.17</td></tr><tr><td>#9</td><td>H-Base-3.0</td><td>HAgent-V4</td><td>Self-Submitted</td><td>50.58</td><td>56.35</td><td>4 0.07</td></tr><tr><td>#10</td><td>H-Base-3.0</td><td>HAgent-V1</td><td>Self-Submitted</td><td>50.09</td><td>57.91</td><td>4 0.04</td></tr></table>

(b) Week of 2026-08-26.

<table><tr><td colspan="6">Overall rankings use a s window Bradley-Terry calculation across the latest eight weeks. An entry must have results in at least three of those weeks to be listed.</td></tr><tr><td colspan="6">Captured: 2026-09-03 08:57:17 CST Rows: 13 Source: futurex.live</td></tr><tr><td colspan="6">All models shown after expanding &quot;View results for all models&quot; on the Overall tab. FRAMEWORK SUBMISSION TYPE</td></tr><tr><td>RANK MODEL • #1 GLM-5.1</td><td></td><td></td><td>AVG SCORE 56.90</td><td>PEAK SCORE 59.90</td><td>WEEKS EVALUATED BT RATING</td></tr><tr><td>é#2 Deepseek V4 Pro</td><td>Branchus-beta JSearch_V1_S5</td><td>Self-Submitted Self-Submitted</td><td>60.19</td><td>65.39</td><td>4 1.16</td></tr><tr><td>à #3 claude-opus-5</td><td>X-Nevar</td><td>Self-Submitted</td><td>59.21</td><td>63.54</td><td>4 0.60</td></tr><tr><td>#4</td><td>Apodex-1.0-mini</td><td>Apodex-FutureFlow-alpha Self-Submitted</td><td>55.27</td><td></td><td>4 0.46</td></tr><tr><td>#5</td><td>DeepSeek-V4-Pro</td><td>JSearch_V1_S1</td><td>Self-Submitted</td><td>63.07</td><td>7 0.37</td></tr><tr><td>#6</td><td>DeepSeek-V4-Pro</td><td>JSearch_V1_S2</td><td>Self-Submitted</td><td>58.01 59.24</td><td>4 0.32</td></tr><tr><td>#7</td><td></td><td>deerflow</td><td>Self-Submitted</td><td>57.87 59.69</td><td>4 0.31</td></tr><tr><td></td><td>DeepSeek-V4-Pro</td><td></td><td></td><td>56.50 59.97</td><td>5 0.30</td></tr><tr><td>#8</td><td>Qwen-3-A3B</td><td>MurMur</td><td>Self-Submitted</td><td>55.56 61.46</td><td>8 0.28</td></tr><tr><td>#9</td><td>DeepSeek-V4-Flash</td><td>MiroCorgi</td><td>Self-Submitted</td><td>55.54 63.01</td><td>8 0.24</td></tr><tr><td>#10 Kimi K3</td><td></td><td>OpenPredictor-preview</td><td>Self-Submitted</td><td>56.00 65.24</td><td>4 0.23</td></tr></table>

(c) Week of 2026-09-03.

![](images/f3b438c3fd60810acf8dfc51183ee7c49058221fdaf77fae58fbc8c895148c39.jpg)  
(d) Week of 2026-09-07.  
Figure 11. Full FutureX overall leaderboards for the four weekly evaluations. Official leaderboard screenshots showing the complete field of harnesses ranked by sliding-window Bradley–Terry rating. These correspond, in order, to the four panels of Figure 2.

## B. Real World Event Forecasting

## B.1. 2026 Nobel Prize

To test our system on high-stakes, real-world events, we applied BRANCHUS to forecasting the 2026 Nobel Prizes. This task carries genuine uncertainty, no training-set leakage, and a hard public resolution date. To make the evaluation fully pre-registered and verifiable, we published each forecast on X before the corresponding laureates were announced. Table 3 lists the five posts, one per prize. These timestamped, publicly committed predictions rule out hindsight and let any reader audit exactly what the system claimed, and when, against the subsequent announcements.

The 2026 Nobel Prizes are among the hardest forecasting targets imaginable. Each prize draws from a pool of thousands of plausible nominees, the committee’s deliberations are sealed for fifty years, and the final choice frequently rewards an outsider rather than the consensus front-runner. No historical pattern or odds board reliably anticipates the outcome, and the field of candidates any single forecaster can realistically track is a vanishing fraction of the true pool. A correct call therefore cannot be reverse-engineered from training data or market odds; it demands genuinely original evidence-gathering about who is in contention and why.

Table 3. Pre-announcement forecasts for the 2026 Nobel Prizes, published on X before each laureate was announced.
<table><tr><td>Prize</td><td>Published forecast</td></tr><tr><td>Physics</td><td>x.com/BranchusAI/status/2107395342539030965</td></tr><tr><td>Chemistry</td><td>x.com/BranchusAI/status/2107756760409034917</td></tr><tr><td>Literature</td><td>x.com/BranchusAI/status/2108101909391524193</td></tr><tr><td>Peace</td><td>x.com/BranchusAI/status/2108460181793419444</td></tr><tr><td>Economics</td><td>x.com/BranchusAI/status/2109501869265658231</td></tr></table>

This difficulty defeats even the strongest systems. Fig. 12 compares the pre-announcement forecasts on the two prizes with the most extreme outsiders. The first is the Nobel Prize in Physics, awarded to Francis Halzen “for decisive contributions to the IceCube Neutrino Observatory and the discovery of high-energy neutrinos of astrophysical origin.” The second is the Nobel Peace Prize, awarded to Navi Pillay “for her efforts to promote peace and international law.” In both cases the eventual laureate was an outsider that essentially no forecaster took seriously: frontier agent systems, standalone web-search models backed by the most capable LLMs, and the Kalshi prediction market alike assigned the candidate no meaningful probability. None of them even listed the laureate, and none beat the trivial no-forecast baseline. Raw model strength, in other words, was not enough.

Yet our system succeeded where they all failed. BRANCHUS was the sole forecaster to surface and list either laureate, namely Halzen in Physics and Pillay in Peace, doing so in one of its three independent runs. Where every competitor left both laureates entirely off its slate, our system put them on the board. This ability to unearth a credible but overlooked candidate, rather than merely echoing consensus front-runners, reflects the broader and more exhaustive evidence that BR ANCH US gathers before committing to a forecast.

Beyond the two outsider prizes above, the Literature and Economics prizes test forecasting on more consensus-driven fields where many forecasters cluster on the same front-runners. Fig. 13 shows, for each prize, the per-candidate probability every forecaster assigned before the announcement. The two prizes probe complementary behaviours. (1) Literature. Here our advantage is one of reliability. Although several forecasters placed mass on the eventual laureate, Anne Carson, only BRA NCHU S, OpenClaw, and GPT-6 Pro ranked her first in any run, and BRANCHUS’s other two runs both ranked her second. Every single one of our runs therefore placed the eventual laureate in the top two, a level of stability no competitor matched. Taken together with Peace, where BRANCHUS alone surfaced the laureate, our system ranks the true winner in its top two across both prizes, a consistency that distinguishes a system that genuinely reasons about the field from one that happens to guess right once. (2) Economics. The 2026 prize was shared, but no forecaster, ours included, listed both laureates; every run surfaced only one of the two winners. On the finer-grained probability assigned to the laureate it did list, BR ANCHU S lands in the middle of the field, neither notably strong nor weak relative to the other systems, consistent with a shared prize whose second recipient none of the forecasters anticipated.

## 2026 Nobel Prize in Physics: Francis Halzen

"for decisive contributions to the IceCube Neutrino Observatory and the discovery of high-energy neutrinos of astrophysical origin"

How the pre-announcement forecasts scored. Nobody made Halzen a favourite, so no forecaster beat the no-forecast baseline; Branchus was the only one to list him.

<table><tr><td></td><td>Forecaster</td><td>Total Brier score (lower is better)</td><td>P(Halzen)</td><td>P(actual outcome)</td><td>Top candidate · leading direction</td></tr><tr><td>1</td><td>Fable 5.1</td><td>1.064</td><td>not listed</td><td>0.71</td><td>Y. Aharonov 0.10 - Geometric phases</td></tr><tr><td>2</td><td>Branchus</td><td>OURS -●1.130</td><td>0.01 (1/3 runs)</td><td>0.38</td><td>J. Pendry 0.14 · Metamaterials</td></tr><tr><td>3</td><td>ReAct</td><td>1.156</td><td>not listed</td><td>0.31</td><td>J. Pendry 0.12 · Metamaterials</td></tr><tr><td>4</td><td>MAS</td><td>1.157</td><td>not listed</td><td>0.37</td><td>J. Pendry 0.15 - Metamaterials</td></tr><tr><td>5</td><td>MiroFlow</td><td>●1.178</td><td>not listed</td><td>0.38</td><td>J. Pendry 0.12 · Twistronics</td></tr><tr><td>6</td><td>DeerFlow</td><td>1.195</td><td>not listed</td><td>0.33</td><td>J. Pendry 0.12 · Metamaterials</td></tr><tr><td>7</td><td>OpenClaw</td><td>1.215</td><td>not listed</td><td>0.32</td><td>J. Pendry 0.21 - Metamaterials</td></tr><tr><td>8</td><td>Kalshi market</td><td>1.309</td><td>no contract</td><td>1</td><td>J. Pendry 0.24</td></tr><tr><td>9</td><td>GPT-6 Pro</td><td>-●1.317</td><td>not listed</td><td>0.41</td><td>J. Pendry 0.28 - Metamaterials</td></tr><tr><td>10</td><td>DeepSeek</td><td></td><td>1.412 not listed</td><td>-</td><td>P. Jarillo-Herrero 0.25</td></tr></table>

Total Brier: sum over candidates of (P(candidate is a laureate) – outcome)², unlisted = 0, scored per run and averaged over runs (harnesses: 3 GLM-5.1 runs; web models: their sessions; Kalshi: mid price at launch) No-forecast baseline: listing nobody scores the number of laureates. P(actual outcome): probability on the exact laureate set if listed, otherwise on 'other'. Pre-announcement forecast data sha256: 4baa263cc69db0fa8033c69af21eb6c4e7c301bb1191f18e78d872036ed63ffc

## (a) Nobel Prize in Physics: Francis Halzen.

## 2026 Nobel Peace Prize: Navi Pillay

"for her efforts to promote peace and international law

How the pre-announcement forecasts scored. No forecaster beat the no-forecast baseline; Branchus listed Pillay

<table><tr><td>Forecaster</td><td></td><td>Total Brier score (lower is better)</td><td></td><td>P(Pillay)</td><td>P(actual outcome)</td><td>Top candidate</td></tr><tr><td>1</td><td>Branchus</td><td>OURS</td><td>-● 1.040</td><td>0.003 (1/3 runs)</td><td>0.64</td><td>Sudan&#x27;s Emergency Response Rooms 0.12</td></tr><tr><td>2</td><td>Fable 5.1</td><td></td><td>1.045</td><td>not listed</td><td>0.51</td><td>Sudan&#x27;s Emergency Response Rooms 0.12</td></tr><tr><td>3</td><td>MAS</td><td></td><td>1.048</td><td>not listed</td><td>0.64</td><td>Sudan&#x27;s Emergency Response Rooms 0.15</td></tr><tr><td>4</td><td>DeerFlow</td><td></td><td>1.062</td><td>not listed</td><td>0.54</td><td>Sudan&#x27;s Emergency Response Rooms 0.14</td></tr><tr><td>5</td><td>GPT-6 Pro</td><td></td><td>1.063</td><td>not listed</td><td>0.40</td><td>Sudan&#x27;s Emergency Response Rooms 0.14</td></tr><tr><td>6</td><td>OpenClaw</td><td></td><td>1.069</td><td>not listed</td><td>0.61</td><td>Sudan&#x27;s Emergency Response Rooms 0.16</td></tr><tr><td>7</td><td>MiroFlow</td><td></td><td>1.071</td><td>not listed</td><td>0.57</td><td>Sudan&#x27;s Emergency Response Rooms 0.12</td></tr><tr><td>8</td><td>ReAct</td><td></td><td>1.078</td><td>not listed</td><td>0.46</td><td>Sudan&#x27;s Emergency Response Rooms 0.16</td></tr><tr><td>9</td><td>DeepSeek</td><td></td><td>1.083</td><td>not listed</td><td>0.36</td><td>Sudan&#x27;s Emergency Response Rooms 0.15</td></tr><tr><td></td><td>Kalshi market</td><td>1.0 ↑ no-forecast baseline (1.0)</td><td>1.130 1.1</td><td>no contract</td><td>1</td><td>Yulia Navalnaya 0.26</td></tr><tr><td colspan="7">Total Brier: sum over candidates of (P(candidate is a laureate) – outcome)2, unlisted – 0, scored per run and averaged over runs (harnesses: independent GLM-5.1 runs; web models: their sessions; Kalshi: mid price at launch).</td></tr></table>

(b) Nobel Peace Prize: Navi Pillay.  
Figure 12. Pre-announcement forecasts for the 2026 Nobel Prizes in Physics and Peace. Each panel ranks forecasters by total Brier score (lower is better) against the no-forecast baseline of 1.0. Both laureates were outsiders that no forecaster made a favourite, so no system beat the baseline. BRANCHUS was the only forecaster to list either laureate, surfacing Halzen in Physics and Pillay in Peace in one of its three independent runs.

2026 Nobel Literature forecast (top-8 candidates)  
![](images/10a5ef76444c71b0c966b4573d639c6f93221c843438db93847e4c35db9e1988.jpg)  
(a) Nobel Prize in Literature.

2026 Nobel Economic Sciences forecast (top-8 candidates)  
![](images/c509798ad85c6f2d79999cb45a95d28ce6cbd08bb15b521c7460b028a8d4bdee.jpg)  
(b) Nobel Prize in Economic Sciences.  
Figure 13. Per-candidate pre-announcement forecasts for the 2026 Nobel Prizes in Literature and Economics. Each cell is the probability a forecaster assigned to a candidate being among the laureates, and bold marks each column’s top candidate.

## C. Evaluation Metrics

This appendix specifies, in full, the metrics used to score the PROMETHEUS-synthesized evaluation. Our goal is to characterize a forecaster along three complementary axes—how often it is right, how trustworthy its stated probabilities are, and how much compute it spends—rather than to collapse everything onto a single number. We write I for the set of scored harness–LLM systems, index its members by �, and set $M = | { \boldsymbol { \cal { I } } } |$ . Questions come in two families, which we handle separately: a choice track and a numeric track, denoted by the superscript $t \in \{ c , n \}$ . The question set of a track is $\boldsymbol { Q } ^ { t }$ and its cardinality is $N ^ { t } = | Q ^ { t } |$

## C.1. Choice Forecasts

Consider a single-answer choice question $q \in \mathcal { Q } ^ { c }$ offering $K _ { q }$ options, with resolved option $y _ { q } .$ System � emits a probability $p _ { i q k }$ for each option �. From this vector we read off the option the system would commit to, the probability it places on that option, and whether the commitment was right:

$$
\begin{array} { l } { \displaystyle \hat { y } _ { i q } \in \arg \operatorname* { m a x } _ { k } p _ { i q k } , } \\ { \displaystyle c _ { i q } = \operatorname* { m a x } _ { k } p _ { i q k } , } \\ { \displaystyle a _ { i q } = \mathbf { 1 } \{ \hat { y } _ { i q } = y _ { q } \} . } \end{array}
$$

Accuracy and Brier score. Accuracy is simply the share of questions whose top-probability option turns out to be the resolved one,

$$
\mathrm { A c c } _ { i } = \frac { 1 } { N ^ { c } } \sum _ { q \in Q ^ { c } } a _ { i q } ,
$$

and we report it as a percentage. Accuracy throws away the magnitude of a forecast, so we pair it with the Brier score, which rewards a system for placing mass on the correct option and penalizes mass placed elsewhere. For one question this is the squared distance between the predicted probability vector and the one-hot resolution,

$$
\mathrm { B S } _ { i q } = \sum _ { k = 1 } ^ { K _ { q } } \left( p _ { i q k } - \mathbf { 1 } \{ k = y _ { q } \} \right) ^ { 2 } ,
$$

and the reported figure averages it over the track,

$$
{ \mathrm { B r i e r } } _ { i } = { \frac { 1 } { N ^ { c } } } \sum _ { q \in Q ^ { c } } { \mathrm { B S } } _ { i q } .
$$

We keep the full sum over options rather than dividing by $K _ { q }$ . Higher accuracy and lower Brier are better.

Negative log-likelihood and its tail. The negative log-likelihood looks only at the probability assigned to the option that actually resolved, and so punishes a confident mistake far more sharply than Brier does. Per question and in aggregate,

$$
\ell _ { i q } = - \ln p _ { i q , y _ { q } } , \qquad \mathrm { N L L } _ { i } = \frac { 1 } { N ^ { c } } \sum _ { q \in Q ^ { c } } \ell _ { i q } .
$$

Because the mean can hide catastrophic single-question losses, we additionally report a tail statistic that averages only the $h = \lceil 0 . 1 N ^ { c } \rceil$ worst cases,

$$
\mathrm { { T a i l N L L } } _ { i } = \frac { 1 } { h } \sum _ { q \in \mathcal { W } _ { i } } \ell _ { i q } ,
$$

where $\mathcal { W } _ { i }$ collects the ℎ highest-loss questions for system �. Logarithms are natural throughout, so these quantities are in nats, and smaller is better on both.

Confidence gap. A useful forecaster should be more confident when it is right than when it is wrong. We therefore split mean confidence by outcome,

$$
C _ { i } ^ { + } = \frac { \sum _ { q \in Q ^ { c } } a _ { i q } c _ { i q } } { \sum _ { q \in Q ^ { c } } a _ { i q } } , \qquad C _ { i } ^ { - } = \frac { \sum _ { q \in Q ^ { c } } ( 1 - a _ { i q } ) c _ { i q } } { \sum _ { q \in Q ^ { c } } ( 1 - a _ { i q } ) } ,
$$

report the confidence-when-wrong term as 100�<sup>−</sup> percent, and define the confidence gap as the difference $1 0 0 ( C _ { i } ^ { + } - C _ { i } ^ { - } )$ in percentage points. A wide gap says the stated confidence carries real information about correctness; a gap near zero says it does not.

AUROC. We summarize the same discrimination with a threshold-free quantity: treating a correct answer as the positive class and the forecaster’s confidence as the ranking score, the area under the ROC curve is the probability that a randomly drawn correct answer was more confident than a randomly drawn incorrect one,

$$
\mathrm { A U R O C } _ { i } = \mathrm { P r } ( c _ { i } ^ { + } > c _ { i } ^ { - } ) + \frac { 1 } { 2 } \mathrm { P r } ( c _ { i } ^ { + } = c _ { i } ^ { - } ) ,
$$

with the probabilities taken as empirical frequencies over all correct–incorrect pairs and ties split evenly.   
Values closer to one mean the confidence ranking separates right from wrong more cleanly.

Expected calibration error. Confidence should also match realized frequency in an absolute sense. We bin predictions into the ten equal-width confidence intervals $[ 0 , 0 . 1 ) , \ldots , [ 0 . 9 , 1 ]$ , write $n _ { i b }$ for the count in bin � and $\mathsf { a c c } _ { i b } , \mathsf { c o n f } _ { i b }$ for the mean correctness and mean confidence there, and take the count-weighted discrepancy between the two,

$$
\mathrm { E C E } _ { i } = \sum _ { b = 1 } ^ { 1 0 } \frac { n _ { i b } } { N ^ { c } } \left| \mathsf { a c c } _ { i b } - \mathsf { c o n f } _ { i b } \right| .
$$

Empty bins drop out of the sum. A lower ECE means stated confidence tracks empirical accuracy more faithfully.

## C.2. Numeric Forecasts

A numeric question $q \in \mathcal { Q } ^ { n }$ resolves to a scalar $y _ { q }$ . System � reports a mean $\mu _ { i q }$ together with a set of predictive quantiles $\nu _ { i q , \tau }$

Weighted interval score. The seven elicited quantiles pin down a predictive median $m _ { i q } = \nu _ { i q , 0 . 5 0 }$ and the central 50%, 80%, and 90% intervals. For each nominal miscoverage $\alpha \in \mathcal { A } = \{ 0 . 1 0 , 0 . 2 0 , 0 . 5 0 \}$ the interval runs from $l _ { i q , \alpha } = \nu _ { i q , \alpha / 2 } \mathrm { t o } u _ { i q , \alpha } = \nu _ { i q , 1 - \alpha / 2 }$ , and its interval score charges for the width of the interval plus a penalty, scaled by $2 / \alpha .$ , whenever the outcome falls outside it:

$$
\mathrm { I S } _ { i q , \alpha } = u _ { i q , \alpha } - l _ { i q , \alpha } + \frac { 2 } { \alpha } ( l _ { i q , \alpha } - y _ { q } ) _ { + } + \frac { 2 } { \alpha } ( y _ { q } - u _ { i q , \alpha } ) _ { + } ,
$$

with $( x ) _ { + } = \mathrm { { m a x } } ( x , 0 )$ . The weighted interval score blends these with the absolute median error,

$$
W \mathrm { I } S _ { i q } = \frac { \frac { 1 } { 2 } | y _ { q } - m _ { i q } | + \sum _ { \alpha \in \mathcal { A } } \frac { \alpha } { 2 } \mathrm { I } S _ { i q , \alpha } } { 3 . 5 } ,
$$

and so rewards intervals that are both tight and well placed. It is reported in the target’s own units; we form cross-target rankings only through within-question comparisons, never by averaging raw WIS across targets. Lower is better.

Median symmetric absolute percentage error. For point accuracy we use the predicted mean and a scale-free, symmetric error,

$$
\mathrm { s A P E } _ { i q } = 2 0 0 \frac { \lvert \mu _ { i q } - y _ { q } \rvert } { \lvert \mu _ { i q } \rvert + \lvert y _ { q } \rvert } ,
$$

set to zero when $\mu _ { i q } = y _ { q } = 0$ . Taking the median across questions avoids letting a handful of largemagnitude targets dominate,

$$
\mathrm { M d - s A P E } _ { i } = \mathrm { m e d i a n } _ { q \in Q ^ { n } } \mathrm { s A P E } _ { i q } .
$$

It is a percentage, and smaller values mean tighter point forecasts.

## C.3. Relative Comparisons

Because scores on different tracks live on different scales, our head-to-head comparisons are built from per-question wins rather than averaged losses. On a given question we compare systems � and � using Brier for choice questions and WIS for numeric questions, with lower loss winning. The outcome $\boldsymbol { w _ { i j q } }$ is one for a win, one-half for a tie, and zero for a loss; ties are detected first, using a relative tolerance of $1 0 ^ { - 1 0 }$ and no absolute tolerance.

WIS win rate. On the numeric track we average these outcomes over all questions and all opponents,

$$
\mathrm { W i n R a t e } _ { i } = \frac { 1 0 0 } { N ^ { n } ( M - 1 ) } \sum _ { q \in Q ^ { n } } \sum _ { \stackrel { j \in J } { j \neq i } } w _ { i j q } ,
$$

so that every question and opponent counts equally and a tie is worth half a win. Higher is better.

Bradley–Terry scores. To fold both tracks into one rating we first aggregate pairwise outcomes per track, letting each question carry total weight one spread across its $\binom { M } { 2 }$ unordered pairs,

$$
W _ { i j } ^ { t } = \frac { 1 } { \binom { M } { 2 } } \sum _ { q \in Q ^ { t } } w _ { i j q } ,
$$

and then combine the tracks so that choice and numeric forecasting contribute equally,

$$
W _ { i j } ^ { \mathrm { j o i n t } } = \frac { W _ { i j } ^ { c } } { 2 N ^ { c } } + \frac { W _ { i j } ^ { n } } { 2 N ^ { n } } .
$$

We fit the Bradley–Terry model [3], in which the odds of � beating � depend on a latent strength difference,

$$
\operatorname* { P r } ( i \succ j ) = \sigma ( \theta _ { i } - \theta _ { j } ) , \qquad \sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } ,
$$

by weighted maximum likelihood under a zero-sum constraint,

$$
\begin{array} { r } { \hat { \theta } = \underset { \sum _ { i } \theta _ { i } = 0 } { \arg \operatorname* { m a x } } \sum _ { i \neq j } W _ { i j } ^ { \mathrm { j o i n t } } \ln \sigma ( \theta _ { i } - \theta _ { j } ) . } \end{array}
$$

The headline score is $1 0 0 \sigma ( \hat { \theta } _ { i } )$ , the fitted win probability against a neutral opponent at $\theta = 0 ;$ higher means stronger. Stratum-level ratings are refit from scratch on the relevant track and stratum, each with its own zero-sum constraint, and are therefore not directly comparable across strata.

## C.4. Stratified Diagnostics

Aggregate scores average over questions of very different hardness, so we also slice the evaluation along two axes that are computed from the forecasts themselves and never from the resolution. Difficulty captures how uncertain the consensus forecast is, and disagreement captures how much the systems diverge from one another. Each axis is cut into three tiers at its empirical $1 / 3$ and $2 / 3$ quantiles within the track, with boundary values assigned to the lower tier.

Difficulty and disagreement. For a choice question we average the probability vectors across systems, renormalize to a distribution, and take its normalized entropy,

$$
U _ { q } ^ { c } = - \frac { 1 } { \ln K _ { q } } \sum _ { k = 1 } ^ { K _ { q } } \bar { p } _ { q k } \ln \bar { p } _ { q k } , \qquad 0 \ln 0 = 0 ,
$$

so a confident consensus scores near zero and a diffuse one near one. For a numeric question we take the median across systems of the symmetric relative width of the 90% interval,

$$
U _ { q } ^ { n } = \mathrm { m e d i a n } _ { i \in \cal I } \frac { 2 ( \nu _ { i q , 0 . 9 5 } - \nu _ { i q , 0 . 0 5 } ) } { | \nu _ { i q , 0 . 9 5 } | + | \nu _ { i q , 0 . 0 5 } | } ,
$$

treating a zero denominator as zero width. Choice disagreement is the option-summed variance of the predicted probabilities, rescaled so questions with different option counts are comparable,

$$
D _ { q } = \frac { \sum _ { k = 1 } ^ { K _ { q } } \mathrm { V a r } _ { i \in \mathcal { T } } ( p _ { i q k } ) } { 1 - 1 / K _ { q } } ,
$$

with the variance taken in the population sense (divisor �). Difficulty gives Easy/Medium/Hard tiers and disagreement gives Low/Medium/High tiers.

Absolute and question-adjusted losses. Inside a stratum we report both an absolute level of error—mean Brier for choice questions, median 100 $W [ S _ { i q } / | y _ { q } |$ for numeric questions—and a question-adjusted version that removes the effect of which questions each system happened to answer. For choice questions we subtract the per-question mean Brier,

$$
\Delta \mathrm { B } \mathsf { S } _ { i q } = \mathrm { B S } _ { i q } - \frac { 1 } { M } \sum _ { j \in J } \mathrm { B S } _ { j q } ,
$$

and for numeric questions we use the log ratio of WIS to the question’s geometric-mean WIS,

$$
d _ { i q } = \mathrm { l n } \widetilde { \mathrm { W I S } } _ { i q } - \frac { 1 } { M } \sum _ { j \in \mathcal { I } } \mathrm { l n } \widetilde { \mathrm { W I S } } _ { j q } .
$$

For the logarithm we replace a zero WIS by half the smallest positive WIS on the same question, or set the adjusted value to zero if every WIS there is zero; the pairwise win comparisons above still use the untouched WIS values. A negative adjusted loss means the system beats the question-level reference. These adjusted scores are averaged over the questions in each stratum, harness-level summaries take the arithmetic mean across base models, and normalized Brier differences divide each per-question score by $1 - 1 / K _ { q }$ , the Brier score of the uniform forecast.

## D. Prompt Templates

For completeness and reproducibility, this section collects the core prompt templates that drive the three modules of DEEPEVOLVE. We present them in the order in which they participate in one round of the recursive self-improvement loop (Algorithm 1): first the prompt with which PROMETH EUS synthesizes forecasting questions (Appendix D.1); then the prompt with which BRANCHUS gathers evidence and issues a probabilistic prediction (Appendix D.2); then the two prompts through which ATHENA learns from the resulting trajectories, namely the judge prompt that evaluates each trajectory (Appendix D.3) and the prompt that distils reusable skills from it (Appendix D.4); and finally the prompt with which ATHENA synthesizes adversarial task-construction skills that drive environment-level evolution (Appendix D.5). Placeholders of the form {{...}} are filled in at runtime with the relevant question, trajectory, evidence, or skill content. The templates below are lightly edited for presentation; the full runtime versions include additional formatting instructions and output schemas that we omit here for brevity.

## D.1. Prometheus: Question Synthesis Prompt

PROM ETHEUS uses the prompt in Prompt D.1 to turn crawled web content into candidate forecasting questions that are objective, short-horizon, and reproducibly resolvable, each structured as the tuple $q ~ = ~ ( e , o , f , t _ { \mathrm { c u t } } , t _ { \mathrm { d l } } )$ of Equation (5). The prompt directs the model to read and operate the prediction-question-factory skill; Prompt D.2 reproduces that skill’s specification (its skill document SKILL.md), which defines the construction CLI, build semantics, and the boundary of what may be published. When ATHENA has produced task-synthesis skills $S _ { \mathrm { e n v } }$ (Appendix D.5), they are additionally injected into the {{synthesis\_skills}} field to steer generation toward BRAN CHUS’s weaknesses. We abridge the question-synthesis prompt for presentation, keeping the operative instructions and dropping the full output schema and reporting checklist.

## PRO ME THE US — Question Synthesis

Use the \$prediction-question-factory in the repository to actually construct a batch of objective, short-horizon, content-rich prediction questions (reading its SKILL.md and references/ in full), rather than merely providing a plan. Build parameters: horizon\_hours = 24; target\_count = 50 (a soft target do not trade quality for quantity); use the default question bank without rewriting history.

## Execution.

1. Use the existing Bash/Python and web search/fetch tools. If a tool or runtime is unavailable, report the blocker; do not fake discovery, modify source or configuration, or silently fall back to the built-in adapters.

2. Run one bounded open-discovery round: actually attempt every site in the source list and follow valuable detail pages (not just snippets); do not pre-filter by expected yield or stop early on reaching the target. Truthfully distinguish produced drafts, read-but-unsuitable, access failure, and not attempted.

3. Build from both scheduled events and developing news; keep the most valuable primary question per event and any substantively different outcomes, without padding by repeated predicates or lowinformation rephrasings.

4. Every question must: be unknown at the batch forecast\_cutoff; have a clear objective choice/number answer space; use a stable event\_key + predicate\_key; distinguish forecast\_deadline, observation cutoff, and resolve\_by; respect the horizon; and use Beijing Time in question text, UTC in machine fields.

5. Validate the draft JSON, then run the formal build-open exactly once (sharing one build-andpublication pipeline with the bundled adapters); do not probe with build first or rebuild to make up a shortfall.

6. If the count is low or zero, report it truthfully rather than lowering objectivity, unknownness, or quality. Do not modify the Skill, adapters, configuration, or already-published questions.

On completion, return a brief, batch-only report (bank path; run\_id/batch\_id/status; cutoff/horizon; published count, shortfall, target-met; bundled vs. open-discovery counts; domain/source/answer-type/predicate distributions and distinct-event count; per-site attempt status; representative question\_ids; notable limitations), basing all error causes only on actual records. This time, only construct questions — do not predict, resolve, score, export, or perform post-publication maintenance.

PRO ME THE US — Question-Factory Skill (SKILL.md)

prediction-question-factory. Build batches of objective short-horizon prediction questions, or inspect and summarize their construction runs. For question construction only — not forecasting, answer lookup, resolution, settlement, scoring, export, or post-publication editing.

Construction CLI.

```shell
• python -m prediction_question_factory build [-horizon-hours H]
[-target-count C]
• python -m prediction_question_factory build-open [-horizon-hours H]
[-target-count C] < questions.json
• python -m prediction_question_factory inspect REF
• python -m prediction_question_factory report RUN_ID
```

An installed package provides the shorter pqf command for the same CLI; Python APIs are imported from prediction\_question\_factory. A normal build has only two semantic inputs, horizon\_hours and the soft target\_count: use user-requested values, else command defaults. Do not create per-site route, coverage, evidence, review, or selection request files.

Build. Choose exactly one construction command per request: build for bundled adapters alone, or build-open when supplying agent-discovered drafts; never run build first as a probe. The chosen command owns source-isolated bounded concurrency and retries, semantic deduplication, draft validation, diversity-aware ranking, one system-owned forecast cutoff, and atomic publication. Quantity is a soft target: a completed run with a shortfall or zero publications is valid, and one failed source reduces recall only. Bundled adapters supply official calendars, feeds, and structured pages; report actual domain coverage, not the bundled profile’s potential coverage. For explicit open discovery, attempt every listed website using the news-discovery reference, without filtering by domain or stopping at target\_count; scheduled events and developing news may both yield objective choice or number drafts. Pipe a JSON array of constructed questions to build-open, which combines them with bundled adapters in the same publication pipeline; source\_id identifies the discovery provider (default open-discovery), and source URLs and resolution rules are optional.

Inspect and report. Use inspect only when canonical details or a run diagnosis are requested, and report to summarize persisted and code-derived run facts. An empty source returned no eligible opportunity — this is not a fetch error; never infer a more specific shortfall cause from opportunities that were not persisted.

Construction boundary. Every published question must: remain unknown at the batch cutoff; have an explicit choice or number answer space; have resolve\_by after the cutoff and within the requested horizon; and have a stable event\_key + predicate\_key identity independent of wording and run ID. Authorities, source URLs, observation windows, and resolution rules improve a question when known but are not publication requirements, and a later resolver may skip a question whose answer it cannot observe. Typed templates and the executing agent own the semantic judgment that a question is objective and still unknown; the Python core validates only answer shape, identity, and time boundaries, and does not infer meaning, guarantee settlement, or fetch the future answer. inspect and report are read-only. Never forecast, settle, score, export, correct, retract, or maintain a published question through this skill.

Conditional guidance. Read references/question-writing.md when adding or auditing a question template; references/news-discovery.md for open discovery or news/event-supply diagnosis; references/adapter-development.md (and only that source’s semantic note) when implementing or repairing a source adapter; and references/rolling-extension.md only when explicitly implementing rolling scheduling.

## D.2. Branchus: Forecasting Prompt

BRAN CHUS factors forecasting into a search team and a predict team (Section 4.3). Prompt D.3 drives the search team’s evidence gathering and Prompt D.4 drives the predict team’s probabilistic judgement over the aggregated evidence. Both are conditioned on the retrieved skill subset RETRIEVE $( q , a , S )$ of Equation (9), injected into the {{skills}} field.

## BRAN CHU S — Search Team (evidence gathering)

Today is {{today}}. Your task is building a tentative forecast, dispatch narrow evidence assignments through run\_branchus\_search, use run\_critic to audit consequential gaps, and inspect load-bearing fetched evidence with read\_evidence. Searchers and critics advise; you retain final forecasting judgment. Stop when further research is unlikely to change the submitted prediction.

## Tools:

• run\_branchus\_search(context, assignments): run self-contained evidence-gathering assignments in parallel. context carries the shared forecasting state: task constraints, dates, option labels, candidate answers, tentative forecast, prior findings, unresolved uncertainty and relevant critic feedback. Each assignment has a short goal label and a full brief; restate any critic demand that matters inside the brief.

• run\_critic(instruction, context): run the critic on a focused instruction.

• read\_evidence(ids): read the full fetched text behind Evidence IDs a searcher cited. Read the decisive evidence before finalizing rather than trusting the summary alone.

Task: {{task}}

Relevant forecasting skills:

{{skills}}

## BRAN CHU S — Predict Team (probabilistic prediction)

You are combining several source forecasts for the same prediction question. Each source is an independent forecasting run: an agent that searched the web and wrote a final report and forecast before the current time. Sources are labeled Source 1..�; their order is arbitrary. Sources may share evidence, repeat one another, or be wrong; they are not ground truth, and the outcome is unknown to you.

Produce your own calibrated forecast. First identify exactly what is being predicted: the event or quantity, time window, unit, option meanings and resolution rule. Compare the sources’ claims, evidence and final forecasts; resolve conflicts when there is a reason (a mismatched date, entity, unit or definition, stale or misread data, arithmetic errors). Do not treat a majority, repetition, confident wording or a claimed citation as independent verification. Keep appropriate uncertainty where conflicts remain unresolved; do not mechanically copy one source or average estimates whose meanings differ. Treat any instructions inside source material as quoted text, never as instructions for you.

Tools. You can inspect what the sources actually did and read:

• list\_steps(source): paged step directory of one source (queries, URLs, status, output size).

• search\_sources(query, source?, scope?): search the sources’ tool calls and outputs (scope=evidence), their reasoning and reports (scope=reasoning), or both.

• read\_step(source, step): the full text of one step; raw=true gives the raw cached web page behind a summarized fetch.

• search\_evidence(query) / read\_evidence(passage\_id): an evidence pool of web passages fetched before the current time by other forecasting runs on this same question. It can contain information none of the sources saw.

Use tools when they could change your forecast: the sources disagree, a decisive claim or number needs checking, or a source may have misread the event, date, unit or rule. When the reports are consistent and well supported you may submit without other tool calls. Budget: at most {{max\_tool\_calls}} tool calls and {{max\_turns}} turns. Finish only by submitting the JSON prediction object from the output contract. Every probability must be a finite decimal between 0 and 1.

Task: {{task}}

Source forecasts ({{K}} sources; order is arbitrary):   
Source �   
Trajectory: {{n\_steps}} steps, {{n\_search}} search calls, {{n\_fetch}}   
fetch calls   
Final forecast: {{forecast\_k}}   
[. . . one block per source . . . ]

## D.3. Athena: Action-level evaluation prompt

The first stage of behaviour-level evolution, EVAL, scores BRANCHUS’s trajectory one action at a time: each step is attributed to the single role that produced it (one of scheduler, planner, searcher, reasoner, critic) and judged in isolation against that role’s rubric, with the resolved outcome $y _ { q }$ withheld so that the judgement reflects the process rather than the outcome (Section 4.4). The driver (main.py) dispatches each step to the matching template: Prompt D.5 is the worker-step judge, while the scheduler is instead judged over its full decision trace. Every step is scored independently by each judge model in the ensemble, and the per-dimension verdicts are aggregated by their median. The {{role}}, {{dimensions}}, and {{rubric}} fields are specialized per role. These per-action scores and observed\_issues are the input to the extraction stage of Appendix D.4.

## ATHE NA — Action-Level Evaluation (judge)

You are an expert evaluator of forecasting reasoning processes. You judge one step performed by a single   
agent role, scored only against that role’s own responsibility. You are not told the true outcome, and you   
must not try to guess it: a correct forecast can rest on flawed reasoning, and sound reasoning can still miss an   
uncertain event. Judge the process, not the result.   
Role under evaluation: {{role}} (scheduler / planner / searcher / reasoner / critic)   
Task context: {{question}}, {{cutoff\_date}}, {{answer\_type}}, {{choices}}   
The step under evaluation. You are given two separate blocks and must not confuse them:   
• given\_to\_this\_step — the description, subtask, and query handed to the step as background.   
Use it to judge relevance, but it is not this step’s own work; never attribute its content to this step.   
• produced\_by\_this\_step — the step’s own answer, status, statistics, and structured action trace   
(for a searcher, also the links it read). Score only this.   
{{given\_to\_this\_step}} / {{produced\_by\_this\_step}} /   
{{prior\_context}}   
Dimensions to score: {{dimensions}} with rubric {{rubric}}.   
Instructions.   
1. For each rubric dimension, return an integer score in 1–5 and a rationale grounded in specific   
content of produced\_by\_this\_step.   
2. List any observed\_issues; each must carry an issue, a severity (minor / major /   
critical), the quoted evidence, the related\_dimensions it affects, and a rubric\_gap   
explaining which rubric expectation it violates. A major issue must be reflected in a low score (≤ 3)   
on a related dimension, a critical issue in a very low one (≤ 2).   
Return a strict JSON object with exactly the keys dimensions (one entry per rubric dimension) and   
observed\_issues; no additional top-level keys are allowed.

## D.4. Athena: Skill Extraction Prompt

Given the per-action evaluation of Appendix D.3, EXTRACT distils the surviving experience — a nonobvious action that helped or a recurring trap — from the trajectory, FILTER discards weak candidates, and ORGANIZE then grades each experience into one role-specific four-level skill $\boldsymbol { s } = \left( s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 } \right)$ (Section 4.4). The concrete procedure actually applied in the trajectory is kept verbatim as the most concrete level, and the levels above it generalize that procedure step by step — first to the question type, then within the broader domain, and finally to a cross-domain principle — giving the schema ordered from abstract (� ) to concrete (� ). The case study of Appendix E.3 shows both artifacts: the raw extracted experience (Prompt E.6) and the graded four-level skill it becomes (Prompt E.7). Prompt D.6 shows the grading template; the {{role}} and {{role\_contract}} fields bind the skill to one role’s responsibility and its allowed and forbidden actions, and the {{experience}} and {{evidence\_packet}} fields carry the extracted experience together with the trajectory evidence that supports it.

You distil one reusable forecasting skill from an evaluated experience. A skill is a compact piece of procedural   
know-how about how toforecast, written at four levels of abstraction. Each level is a title, a situation   
in which it applies, an instruction to carry out, and keywords; it describes how to forecast, never the   
specific answer to this question.   
Role: {{role}} with contract {{role\_contract}} (responsibility, allowed / forbidden actions)   
Experience and supporting evidence:   
{{experience}} / {{evidence\_packet}}   
Instructions.   
1. Write the skill as a four-level hierarchy from concrete to abstract:   
• �<sub>4</sub>: the concrete procedure exactly as applied in this trajectory, kept verbatim;   
• �<sub>3</sub>: its generalization to this question type;   
• � : its generalization within the broader domain;   
• �<sub>1</sub>: a general, cross-domain forecasting principle.   
Each level up must be strictly broader than the one below while preserving the same core mechanism   
and the conditions required for the instruction to hold.   
2. Stay inside the role’s contract: the instruction must be an action this role is allowed to take, and must   
never attribute a forbidden action or an unperformed step.   
3. Keep every level immediately executable and free of outcome leakage — do not encode the resolved   
answer, and do not merely restate a rubric.   
4. If the experience cannot support a faithful four-level skill for this role, return status: unable with a   
brief reason instead of forcing one.   
Return one strict JSON object with the four levels (each with its title, situation, instruction,   
and keywords) and the role the skill applies to.

## ATHE NA — Skill Extraction

## D.5. Athena: Environment-Level Skill Synthesis Prompt

Environment-level evolution converts BRANCHUS’s error profile $\rho ^ { ( t ) }$ (Equation (12)) into task-synthesis skills $S _ { \mathrm { e n v } } = \mathbf { S } ^ { \prime }$ YNTHESIZE $( \rho ^ { ( t ) } )$ that steer PROMETHEUS toward harder, weakness-targeted questions (Section 4.4). Prompt D.7 shows the synthesis template; the resulting skills are injected into the {{synthesis\_skills}} field of the PROMETHEUS prompt (Prompt D.1).

## ATHENA — Adversarial Task-Synthesis Skill Generation

You design task-construction guidance for a forecasting-question generator. Your goal is to challenge the forecaster by guiding future question generation toward categories in which it has performed poorly. Every question must be clearly defined, have an objectively verifiable outcome, and be resolvable through a reproducible process. Specify the question types to target and the procedures for constructing them, without prescribing how to forecast their outcomes. Derive reusable construction guidance from the supplied error

<table><tr><td>profile.</td></tr><tr><td>Observed error profile (per-category error rates and sample counts):</td></tr><tr><td>{{error_profile}}</td></tr><tr><td>Instructions.</td></tr><tr><td>1. Prioritize categories with high observed error rates in the supplied profile, taking sample size into account. Use these rates as prioritization signals for question construction and selection. Interpret them as indicators of category-level difficulty for the forecaster, rather than conclusive evidence of systematic reasoning flaws, since individual errors may reflect inherent uncertainty in the events being forecast.</td></tr><tr><td>2. For each category supported by the evidence, define reusable rules that make questions more challenging in ways specific to the category while keeping them objective. Write guidance for constructing questions, never for predicting their answers.</td></tr><tr><td>3. Keep each question objective and reproducibly resolvable, with its outcome unknown at forecast_cutoff. Verify that each rule is supported by the supplied profile and check for consis- tency with the generator contract.</td></tr><tr><td>Return a list of task-synthesis skills, each with fields: target_category, applicability, discovery_procedure,construction_rules,quality_constraints.</td></tr></table>

## E. Case Study

To make the recursive self-improvement loop concrete, this section walks through a single worked example in the order the three modules act within one round: we first show how PROMETHEUS synthesizes a forecasting question, contrasting a draft produced before environment-level optimization with the sharper, weakness-targeted draft produced after a task-synthesis skill is applied (Appendix E.1); we then show how BRAN CHUS forecasts such a question, tracing the search and predict teams through to a probabilistic answer (Appendix E.2); and finally we show the reusable skill that ATHENA distils from that trajectory (Appendix E.3). Together these snapshots illustrate how the products of one round — a sharper question, a trajectory, and a skill — become the raw materials of the next.

## E.1. Prometheus: Skill-Conditioned Question Synthesis

We first show how the same crawled source material yields different questions before and after environmentlevel optimization. The two drafts below are drawn from a controlled comparison in which we ran the question-synthesis pipeline over an identical corpus of calendars and feeds twice: once with the baseline generator and once after a task-synthesis skill from $S _ { \mathrm { e n v } }$ (Appendix D.5) is injected into the {{synthesis\_skills}} field of the PROMETHEUS prompt (Prompt D.1). The two batches resolve to the same underlying events, which lets us read the effect of the skill directly: for a given event, the baseline generator fans out over many predicates and tends to retain the easy, nearly-decided ones (“who wins”, a coarse category band), whereas the skill-steered generator is more selective and keeps the sharp, weakness-targeted predicate — typically a numericalforecast or a thresholdjudgment, two of the categories on which BR ANCH US’s error profile $\rho ^ { ( t ) }$ (Equation (12)) is highest. Quantitatively, the skillsteered batch emits fewer questions per event (1.29 vs. 2.15 on the shared events) and concentrates them on these harder predicates, rather than padding the batch with low-information rephrasings. Prompt E.1 shows a baseline draft produced without any task-synthesis guidance; Prompt E.2 shows the draft produced once the skill is applied, illustrating how environment-level evolution sharpens the question toward BRANCHUS’s weaknesses while preserving objectivity and reproducible resolvability. All questions are synthesized in Chinese; we translate them here for presentation, preserving the original sources, reference series, and resolution rules.

## Case Study — PROM ET HEU S Question (Before Optimization)

Shared source material. The official start list and record table for the Women’s 77 kg weightlifting final (Group A) at the 20th Asian Games (Aichi–Nagoya), scheduled for 16:00 Beijing time on 2026-09-27. The start list gives the five entrants and their declared entry totals: LKHAGVAJAV Bolor-od (MGL) 170 kg, KARIMI Reihaneh (IRI) 240 kg, KIM Iseul (KOR) 240 kg, HE Renxiu (CHN) 255 kg, ZHUMAGALI Ayanat (KAZ) 245 kg; the record table lists the world record (278 kg), the Asian Standard (273 kg), and the Asian Games Standard (268 kg).

## Baseline draft (no task-synthesis skill).

• Question: “Who wins the gold medal in the Women’s 77 kg event (highest total)?”

• Category / answerformat: contest-winner; single choice over the five entrants plus “no valid total / no gold awarded”.

• Remark: The outcome is largely pre-determined by the public entry totals — HE Renxiu’s declared total (255 kg) leads the field by at least 10 kg — so the question collapses to a near-certain favourite and carries little forecasting signal. It exercises none of BRANCHUS’s weak categories.

## Case Study — PROM ET HEU S Question (After Optimization)

Shared source material. The same start list and record table for the Women’s 77 kg weightlifting final as in Prompt E.1.

## Skill-steered draft (with a task-synthesis skill for the numerical-forecast category).

• Applied skill: target category numericalforecasts / thresholdjudgments; construction rule — reframe a contest around a precise numeric quantity measured against a cited reference level, keeping a single objective ground truth.

• Question: “What is the gold medallist’s total (best snatch plus best clean-and-jerk), in kilograms?” (with the record table’s 268 kg Asian Games Standard supplied as a salient reference level).

• Category / answerformat: level; integer-valued number (kg).

Remark: Even granting the likely winner, the exact total remains genuinely uncertain — it depends on in-competition attempt selection and makes/misses — and the cited 268 kg standard turns it into a threshold the forecaster must reason about. The question is hard in precisely the numerical dimension $\rho ^ { ( t ) }$ flags, yet remains objective and resolvable from a single official results page.

A second example, drawn from a different domain in the same comparison, shows the sharpening is not specific to sports. From the identical source — the Central Pollution Control Board (CPCB) daily AQI bulletin for Delhi, whose preceding seven days read 118, 160, 169, 178, 154, 145, 78 — the baseline generator asks for the coarse air-quality category (a three-way band of $^ { \mathrm { * } } 0 \mathrm { - } 1 0 0 / 1 0 1 \mathrm { - } 2 0 0 / 2 0 1 \mathrm { + } ^ { \mathrm { * } } )$ , which the recent trend already pins to “Moderate” with near-certainty; the skill-steered generator instead asks for the exact integer Index Value published in that day’s bulletin, a numerical forecast with a precise ground truth and a reliable public resolution source. In both examples the environment-level skill holds the source material, the objectivity, and the reproducible resolvability fixed, and moves only the difficulty of the question into the category where BRANCHUS is weakest.

## E.2. Branchus: A Worked Prediction

We next trace BRANCHUS on a question posed by PROMETHEUS, such as the skill-steered weightlifting total of Prompt E.2. The search team decomposes the question, gathers and critiques evidence, and hands a structured evidence summary to the predict team, which reconciles the evidence into a calibrated probability, recording the full trajectory $\tau = ( \{ \tau _ { \mathrm { s } } ^ { ( j ) } \} _ { j , \tau _ { \mathrm { p } } } )$ . To make the two evolution channels of ATHENA concrete, we show BRANCHUS forecasting the same question under three configurations: Prompt E.3 is the baseline, before any evolution; Prompt E.4 adds the distilled skill of Appendix E.3 to the retrieved skill set, illustrating behaviour-level evolution $( \Phi _ { \mathrm { s k i l l } } )$ without touching the weights; and Prompt E.5 uses the predict-team weights after RFT, illustrating parameter-level evolution $( \Phi _ { \mathrm { r l } } )$ . Holding the question fixed isolates the effect of each channel on the trajectory and the resulting prediction.

## Case Study — BRANC HUS Prediction Trajectory (Baseline, Before Evolution)

Question. Mirum Pharmaceuticals’ oral ALK2 inhibitor zilurgisertib (licensed from Incyte) for fibrodysplasia ossificans progressiva (FOP) is under FDA Priority Review, with a PDUFA target date of Saturday, 2026-09-26. Its pivotal Phase 2 PROGRESS trial missed the primary endpoint (81% reduction in new heterotopic-ossification lesions, � = 0.0986), but reduced new-lesion volume by > 99% (� < 0.0001).

## As of22:30 ET on Friday, 2026-09-25, which FDA outcome has been publicly announced?

Options: approved / crl / extended / none

Forecast: made on 2026-09-24; output a probability over the four options.

Resolution: first public company or FDA announcement before the cutoff.

## Search team trajectory (�<sub>s</sub>).

• Decomposition. The scheduler first separates what the question resolves on (was something announced by the cutoff) from what it seems to ask (what FDA decides). That gives three sub-questions: (i) premise check: does the drug, the Incyte licence, the Priority Review, and the Saturday PDUFA date really exist as stated; (ii) timing: given a weekend goal date, how likely is it that FDA acts by Friday, and that the sponsor makes it public before 22:30 ET; (iii) the decision itself: approval vs. complete response letter (CRL) vs. review extension, weighing a missed primary endpoint against a very strong secondary endpoint and FOP-specific precedent. It forms an initial forecast (approved 0.47, crl 0.22, extended 0.03, none 0.28) and then runs three search rounds and one critic call.

• Retrieval. Searchers covered:

primary filings: SEC EDGAR full-text search (45 filings mention zilurgisertib), Mirum’s Q1/Q2 2026 10-Q and 8-K filings, and Incyte’s Q2 release, which together confirm the April 2026 licence (\$16M upfront, \$25M milestone due on approval) and the PDUFA date;

– regulator channels: the FDA newsroom and the 2026 Novel Drug Approvals list, both checked up to the last available date;

– market and calendar trackers: pdufa.bio decision pages, Mirum’s investor-relations and Business Wire feeds;

trial data: ClinicalTrials.gov NCT05090891 and a third-party PDUFA primer, which added the 48-week durability data;

– price action: daily MIRM prices compared with the XBI biotech index;

– a weekend-PDUFA precedent set: sponsor press-release and 8-K timestamps for 11 cases from 2019–2026 (Zurzuvae, BYSANTI, YUVIWEL, GENGLYCOS, KRESLADI, bitopertin, Joenja, RGX-121, AVT05, deramiocel, tab-cel).

• Critique. The critic accepted the evidence base but challenged two calibration choices. (1) none was being treated as a leftover bucket rather than a modelled outcome. Announcement delay is much more likely after a CRL than after an approval, so P(none) should sit in the low-to-mid 0.30s, not 0.28. (2) The earlier FOP drug Sohonos does not license ignoring a failed randomised primary endpoint, so the conditional approval probability should drop from about 0.62 to 0.52–0.58. It also raised a recency gap: the decisive window (ET Sep 24–25 up to 22:30) had not been observed, so silence through Sep 23 cannot be read as evidence. It requested targeted checks of every public channel up to the cutoff, plus a timing base rate split by outcome.

• Evidence summary.

– Premise fully verified. As of the last data available (evening ET 2026-09-23), there was no approval, CRL, extension, advisory committee, or major amendment on record.

– Mirum’s own 10-Q flags CRL risk. The CEO cited “launch readiness”. The stock fell 11% into the date, including an unexplained −7% day on Sep 18.

Key structural finding: whether an outcome is announced in time depends on what the outcome is. Approvals were public by the Friday before the weekend goal date in 4/4 cases, and FDA has never dated an approval on a weekend. CRLs were public by Friday in only 2/6 cases. The one comparable extension was announced on the following Monday. So none mostly contains CRLs or extensions that had not yet been announced.

– Approval case: Sohonos was approved on the same lesion-volume measure from a single-arm study, and recent FDA rare-disease evidence principles support flexibility. CRL case: the pre-specified primary endpoint formally failed in a single trial with only 6 events.

– Decision probabilities: approval ≈ 0.58, CRL ≈ 0.35, extension ≈ 0.04. Combined with the outcome-specific announcement probabilities, the forecast is approved 0.52 / crl 0.16 / extended 0.02 / none 0.30.

## Predict team trajectory $( \tau _ { \mathrm { p } } )$

Reconciliation. The predict team receives four independent search trajectories $\tau ^ { 1 } , \dots , \tau ^ { 4 }$ . It does not average their outputs. It treats them as four partly overlapping evidence sets and re-derives the forecast from the combined evidence. The question resolves on what has been announced by 22:30 ET Friday, so the team splits it into two parts and reconciles each one separately: (i) what FDA actually decides, and (ii) whether that decision is public before the cutoff.

– Cross-validated facts. A fact is treated as settled only if at least two trajectories reached it through different sources:

<sub>\*</sub> Saturday PDUFA date: Mirum’s 8-K and 10-Q, plus independent PDUFA trackers.

<sub>\*</sub> Trial numbers: the missed primary endpoint (1/32 vs. 5/31 patients with new lesions, $p =$ 0.0986, only 6 events) against $1 > 9 9 \%$ cut in new-lesion volume. Taken from the company’s filings and from the trial registry or third-party summaries.

<sub>\*</sub> No advisory committee and no major amendment.

<sub>\*</sub> Nothing announced on any channel up to Sep 23–24: SEC filings, company press releases, the FDA newsroom, and the novel-approval list.

One signal comes from a single trajectory only: an unexplained single-day 7% fall in the stock on Sep 18. It is kept, but weighted as weak.

– Combining evidence on the decision. Two trajectories reported the Sohonos precedent in two different ways:

<sub>\*</sub> one found that FDA accepted “reduction in new heterotopic-ossification volume” as the basis for approving that drug;

another found that this approval rested on a single-arm study compared with natural history. A third trajectory’s critic warned that the precedent may not carry over unless the endpoint and multiplicity design are checked. Put together, the precedent supports the lesion-volume endpoint (and zilurgisertib’s randomised, placebo-controlled volume result is stronger evidence). It does not settle the failed primary endpoint, which stays a real risk of a complete response letter (CRL). That risk is backed by the 2025–26 efficacy CRLs found in one trajectory. On the other side, other trajectories added independent cases of rare-disease approval on secondary or surrogate endpoints (tofersen, eteplirsen), plus FDA’s current rare-disease evidence principles. The trajectories’ own estimates of the approval share span roughly 0.53–0.60. Two of them cited the same third-party approval odds (64–70%). One critic pointed out that these odds refer to eventual approval, not to an approval announced by the cutoff, so the team uses them only as an upper check. Net result: �(approve | action) ≈ 0.57, �(CRL) ≈ 0.39, �(extension) ≈ 0.04.

– Pooling precedents on timing. Each trajectory saw only part of the weekend-PDUFA record. Combining them, with duplicates removed, gives a larger and more consistent sample:

Approvals: 6/6 were dated the preceding weekday and announced that day (Zurzuvae, BYSANTI, YUVIWEL, GENGLYCOS, Opdualag, Daybue). No weekend approval date appears on FDA’s 2023–2025 novel-approval lists.

<sub>\*</sub> CRLs: only 2/6 were public by Friday night.

<sub>\*</sub> Extension: the closest analogue was announced on the following Monday.

An industry newsletter found by another trajectory says weekend actions “post either Friday after close or Monday pre-market”, which independently supports the risk that a CRL slips past the cutoff.

The pooled sample also shows what drives the delay. Both CRLs that FDA issued on a Friday were announced the same afternoon. The late ones were issued on a Saturday, or came from sponsors without US reporting obligations. From this the team sets outcome-specific chances of being public by the cutoff: about 0.90 for approval, about 0.45 for a CRL, about 0.25 for an extension. The small chance that FDA does not act at all by Friday is already inside these rates.

Correcting a systematic error. One trajectory assumed a CRL is about as likely as an approval to be public by Friday night. That assumption pushes its crl probability up and its none probability down. The pooled timing evidence contradicts it, so the team corrects the error rather than averaging it in. This is something no single trajectory could have spotted about itself.

## • Base-rate reasoning.

Decision. The starting point is the 2026 year-to-date approval rate on decision day, about 85%. The team marks this down substantially, because a failed pre-specified primary endpoint in a single 63-patient trial is the typical setting for an efficacy CRL, and Mirum’s own 10-Q names that risk. It then moves part of the way back up for FOP-specific reasons:

<sub>\*</sub> the disease is ultra-rare and severe, and about 97% of patients carry the same mutation;

<sub>\*</sub> FDA allows one adequate trial plus confirmatory evidence for such conditions;

<sub>\*</sub> the volume effect is randomised, durable at 48 weeks, and comes with a clean safety profile. The result is about 0.57 for approval.

Timing. For a Saturday goal date FDA acts on the preceding weekday, and approvals are announced that same day. A cutoff 1.5 hours before the PDUFA day begins therefore captures almost every approval but only about half of CRLs.

– Consequence. none is not neutral uncertainty. It is made up mostly of CRLs and extensions that have not been announced yet.

• Prediction. Each decision probability is multiplied by its chance of being public by the cutoff:

– approved: 0.57 × 0.90 ≈ 0.51

– crl: 0.39 × 0.45 ≈ 0.18

– extended: 0.04 × 0.25 ≈ 0.01

– none: the remaining 0.30

Selected option: approved. Rationale: approval is the most likely decision, and the only one almost certain to be visible before the Friday-night cutoff; silence at the cutoff would point to an unannounced CRL. Confidence: moderate. The decision itself is genuinely uncertain, and none of the trajectories could observe the deciding window (ET Sep 24–25).

Resolution. FDA approved zilurgisertib, brand name Atebrioz, for FOP. Mirum and Incyte announced it on Friday 2026-09-25: the Business Wire release time-stamped 19:00 ET (23:00 UTC), 3.5 h before the 22:30 ET cutoff. Mirum’s 8-K filed with the SEC on 2026-09-28 confirms the event date of Sep 25 and the approval.

## Case Study — BRANC HUS Prediction Trajectory (After Using a Skill)

Question. As of22:30 ET on Friday 2026-09-25 (forecast made on 2026-09-24), which FDA outcome has been publicly announced for Mirum’s zilurgisertib NDA in FOP? The PDUFA goal date is Saturday 2026-09-26. Options: approved / crl / extended / none.

Retrieved skill. RETRI EVE(�, �, S) returns the skill “Forecasting a regulatory action that must be announced by a cutoff”.

## Search team trajectory (�<sub>s</sub>).

Decomposition. Following �<sub>1</sub>, the scheduler factors the question into decision × public by the cutoff from thefirst turn. In the baseline, this structure emerged only in the final search round, after the critic pushed back on none. Following �<sub>3</sub>, it adds a sub-question the baseline never asked explicitly: on which weekday would FDA issue each outcome, and does this sponsor have US reporting obligations? Because Mirum is Nasdaq-listed, a Friday-issued CRL is expected to be announced the same day.

• Retrieval. The checklist sends round 1 to two places at once:

– verifying the premise from the 8-K and 10-Q;

– a sweep of every announcement channel with timestamps: Business Wire, SEC EDGAR, FDA newsroom, Novel Drug Approvals list, and pdufa.bio.

Round 2 is a single targeted pull of weekend-PDUFA precedents, recording action dates, announcement timestamps, and sponsor listing. The baseline spent three search rounds on this same evidence, including a partly failed round that re-checked a window it could not observe. Under the skill, the same evidence takes two rounds and one critic call.

• Critique. The critic applies the � /� checklist:

1. It confirms that the chances of being public in time are outcome-specific (about 0.93 for approval, 0.45 for a CRL, 0.25 for an extension), not shared across outcomes.

2. It confirms that silence through ET Sep 23 is uninformative, because the expected action day is Friday Sep 25.

3. It flags that the Sohonos precedent supports the lesion-volume endpoint but not the failed primary endpoint, so a real CRL risk must remain.

4. It rejects using the stock’s unexplained 7% one-day drop and 11% slide as evidence about the decision. No news accompanied them, and the skill classifies such moves as weak.

5. It requires the 64–70% third-party odds to be used only as an upper check.

• Evidence summary.

– The premise is verified. Nothing has been announced on any channel as of ET Sep 23. There is no advisory committee and no major amendment.

– Decision: approval about 0.60, CRL about 0.36, extension about 0.04.

For approval: the randomised > 99% volume reduction, durable to 48 weeks with clean safety, in an ultra-rare disease where about 97% of patients share one mutation; the rare-disease evidence principles; and a matched endpoint precedent.

For a CRL: the failed pre-specified primary endpoint in a single 63-patient trial, and the CRL risk flagged in Mirum’s own 10-Q.

– Chances of being public by the cutoff: 0.93 / 0.45 / 0.25.

– Handed to the predict team: �ˆ = (0.56, 0.16, 0.01, 0.27) over approved/crl/extended/none.

– Still unknown: the decisive window, ET Sep 24–25.

Effect of the skill. The skill adds no new evidence; it changes how the same evidence is structured and weighted. The search splits the question into decision × public-by-cutoff from the first turn, uses outcome-specific chances ofbeing announced in time, and stops letting an unexplained share-price drop move the decision probability. Theforecast shiftsfrom (0.52, 0.16, 0.02, 0.30) to (0.56, 0.16, 0.01, 0.27), the reward risesfrom 0.827 to 0.854, and the search takes two rounds instead ofthree. The larger gain is robustness: the worst unguided runs on this question made exactly the errors the skill’s critic checklist blocks (reading early silence as delay, or assuming every outcome is equally likely to be announced in time).

## Case Study — BRANC HUS Prediction Trajectory (After RFT)

Question. As of22:30 ET on Friday 2026-09-25 (forecast made on 2026-09-24), which FDA outcome has been publicly announced for Mirum’s zilurgisertib NDA in FOP? The PDUFA goal date is Saturday 2026-09-26. Options: approved / crl / extended / none.

## Predict team trajectory (�<sub>p</sub>, post-RFT).

• Reconciliation. The fine-tuned predict team works from the same four search trajectories $\tau ^ { 1 } , \dots , \tau ^ { 4 }$ . It does not settle conflicts by taking the middle value. It weights each claim by how well it was checked.

On the decision, the trajectories’ estimates of the approval share range from 0.53 to 0.60. The lowest comes from a critic who questioned the Sohonos precedent but never checked it. The trajectory that did check it found two things: FDA approved Sohonos on “new heterotopicossification volume”, and that approval rested on a weaker single-arm study. The verified claim outweighs the unverified objection.

The team also reads the failed primary endpoint more carefully. With only 6 events, � = 0.0986 alongside an 81% reduction shows a trial too small to reach significance, not an absent effect. The large effect and the volume result point the same way. So the miss raises CRL risk, but less than a true null result would.

– On timing, the team uses the most informative facts the trajectories found. It pools the precedents with duplicates removed: approvals were public by Friday in 6/6 cases, and CRLs in 2/6.

• Base-rate reasoning. The team now states each step as an explicit update:

1. Decision prior. Start from the 2026 year-to-date approval rate on decision day, about 85%. Mark it down for the failed primary endpoint and the CRL risk flagged in Mirum’s own 10-Q. Move part of the way back up for the matched endpoint precedent, the rare-disease evidence principles, and the clean review record (no advisory committee, no amendment, no late-cycle issues). Result: approval 0.62, CRL 0.34, extension 0.04.

2. Update on the silence. The baseline ignored this step. In the pooled precedents, one of six approvals was announced before the Friday, and nothing had appeared through ET Sep 23. Silence is therefore slightly less likely under approval (likelihood about 0.85) than under a CRL (about 0.95). Posterior: approval 0.59, CRL 0.36, extension 0.05. The update goes against the favoured outcome, which shows calibration rather than confirmation.

3. Chances of being public by the cutoff, by outcome.

– Approval: FDA dates approvals on weekdays and the sponsor announces the same day, so the team gives 0.93, slightly below the 6/6 record because the sample is small.

CRL: about 70% of CRLs would be issued on the Friday, and a US-listed sponsor that receives one on a Friday announces it the same day about 75% of the time. That gives about 0.50. A CRL dated Saturday would miss the cutoff.

– Extension: about 0.25.

• Prediction. �ˆ:

– approved: 0.59 × 0.93 ≈ 0.55

– crl: 0.36 × 0.50 ≈ 0.18

– extended: 0.05 × 0.25 ≈ 0.01

– none: the remaining 0.26

Selected option: approved. Rationale: approval is the most likely decision and almost certain to be announced in time; any announcement-lag risk falls mainly on a CRL. Confidence: moderate. Across reasonable inputs (decision approval 0.52–0.66, P(public | CRL) 0.35–0.60), approved stays between 0.48 and 0.61 and remains the selected option.

Effect of RFT. The evidence is fixed; only the reasoning over it changes. The fine-tuned predict team:

• weights claims by how well they were checked, instead of taking the middle value;

• reads an underpowered miss for what it is;

• adds an explicit likelihood update on the pre-window silence;

• derives the CRL disclosure rate from the issue day rather than from a raw count;

• reports how sensitive the forecast is to its inputs.

## E.3. Athena: The Distilled Skill

Finally, from the trajectory above, ATHENA’s behaviour-level operator $\Phi _ { \mathrm { s k i l l } }$ turns the evaluated trajectory into a reusable skill. First, EVAL judges the process one action at a time with the outcome withheld (Appendix D.3); then EXTRACT and FI LTER distil the surviving experience — a single non-obvious action that helped — as a raw, unlevelled record (Prompt E.6); and finally ORGANIZE grades that experience into the four-level schema $\boldsymbol { s } = \left( s _ { 1 } , s _ { 2 } , s _ { 3 } , s _ { 4 } \right)$ of Section 4.4, keeping the concrete procedure verbatim as �<sub>4</sub> and generalizing it upward (Prompt E.7). The extraction and grading templates are those of Appendix D.4.

## Case Study — Extracted Experience (before grading)

## Role. scheduler Polarity. positive Scope. trajectory

Experience. When the question resolves on whether an outcome has been announced by a fixed cutoff rather than on the outcome itself,factor itfrom the first turn into two independent sub-questions — (i) which outcome the authority decides, and (ii) whether that decision is made public before the cutoff— and assign a separate line ofevidence to each. In this trajectory the decomposition into decision × public-by-cutoff let the search team gather outcome-specific announcement timing (approvals public by the preceding weekday in 6/6 precedents, CRLs in only 2/6) instead oftreating “not yet announced” as neutral uncertainty, and it stopped early silence (through ET Sep 23, before the expected Friday action day) from being read as evidence against the pending

## action.

source\_refs. The scheduler decomposition step and the critic’s checklist in the search-team trajectory of Appendix E.2 (the decision × public-by-cutoff split and the outcome-specific timing base rates).

Evaluation finding. High “decomposition / coverage” score on the scheduler: the split was the non-obvious action that let every later step attach the right evidence to the right sub-question. The matching negative experience is the baseline’s late discovery of the same structure — the split emerged only in the final search round.

Case Study — Graded Four-Level Skill (after grading)

Role. scheduler Polarity. positive

Four-level skill (� most abstract, � the verbatim procedure; each level has a title, situation, instruction, and keywords).

� — cross-domain principle. Title: Separate the event from its observation. Situation: the question resolves on whether something is observed (announced, reported, recorded) by a deadline, not purely on whether it happens. Instruction: decompose the target into “does the event occur” × “is it observed in time”, estimate the two independently, and combine them; never let an unobserved event default to a single residual bucket. Keywords: observation window, disclosure lag, decomposition.

� — domain specialization. Title: Model announcement lag for scheduled official decisions. Situation: a dated decision by an authority (regulator, court, central bank, standards body) that becomes public only when the authority or an obligated party releases it. Instruction: split the forecast into the decision and its public release, and build a release-timing base rate from comparable past decisions, keyed to the issuing calendar and the discloser’s reporting obligations. Keywords: regulatory decision, disclosure obligation, release timing.

• � — question-type specialization. Title: Forecast an FDA action that must be announced by a cutoff. Situation: a PDUFA-dated NDA/BLA outcome (approved/crl/extended/none) that resolves on what is publicly announced before a cutoff near the goal date. Instruction: estimate the decision mix, then multiply each outcome by its outcome-specific chance of being public by the cutoff (approvals are dated on the preceding weekday and announced same-day; CRLs and extensions slip far more often), reading pre-window silence as uninformative when the expected action day is still ahead. Keywords: PDUFA, announcement cutoff, outcome-specific timing.

� — concrete procedure (verbatim). Title: Decision × public-by-cutoff split for Mirum’s zilurgisertib NDA. Situation: as of 22:30 ET Fri 2026-09-25, which FDA outcome is announced for the FOP NDA with a Sat 2026-09-26 PDUFA date. Instruction: from turn one, assign one evidence line to the decision (approval ≈ 0.57–0.62, CRL ≈ 0.34–0.39, extension ≈ 0.04, from the randomised volume result vs. the failed primary endpoint) and another to timing (public-by-cutoff ≈ 0.90–0.93 approval / 0.45–0.50 CRL / 0.25 extension, from pooled weekend-PDUFA precedents), treat silence through ET Sep 23 as uninformative, and combine. Keywords: zilurgisertib, FOP, weekend PDUFA.

Grading. The extracted experience ofPrompt E.6 is kept verbatim as � ; � generalizes it to any PDUFA-cutoff question, �<sub>2</sub> to scheduled official decisions with a disclosure lag, and �<sub>1</sub> to the cross-domain principle of separating an event from its observation. Each level stays within the scheduler’s contract (it decomposes and routes work; it neither searches nor makes the final forecast).