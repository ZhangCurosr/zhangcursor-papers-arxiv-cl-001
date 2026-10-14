# Auditing Failure Attribution: Label Predictability and Judge Behavior

Tianyang Xu Jing Gao Purdue University {xu1868, jinggao}@purdue.edu

## Abstract

Failure-attribution benchmarks ask an LLM judge which step or agent caused a failure. A high score should mean the judge checked the faulty step against the run (a cross-step check). However, spurious signals (surface cues) that predict the annotation can produce a high score. In this paper, we propose four tests to separate checking from cue use. Input restriction asks whether partial input predicts the annotation. Cue following checks whether moved cues change answers. Residual accuracy scores judges where cue-only predictors fail. Aggregation asks whether the failure mix or metric changes rankings. Real observations motivate these tests: on three benchmarks, simple predictors match or beat GPT-4.1. In two Who&When Pro systems, a classifier reading one step’s reasoning finds 95–100% of faulty steps, motivating input restriction. In one of them, error injection leaves no reasoning after the faulty step; judges fine-tuned on its labels beat GPT-4.1 by 41 points. Adding task-related reasoning after that step moves this cue: finetuned judges lose 31–99 points, but zero-shot judges lose at most 9, motivating cue following. On Aegis, blaming every agent matches GPT-4.1 on micro-F1 but trails by 19 points in exact-set accuracy, motivating aggregation. We recommend reporting these tests when developing failure-attribution benchmarks.

## 1 Introduction

Failure attribution helps developers debug agent trajectories (step-by-step logs of agent runs) by locating the step or agent responsible for a failure. A final pass/fail signal does not identify the faulty step, so failure-attribution benchmarks ask a model, called a judge, which step or agent caused the failure. Failure attribution is essential for improving LLM and agentic systems, and several benchmarks have therefore been developed (Zhang et al., 2025; Liu et al., 2026; Zhang et al., 2026; Kong et al., 2026; Chen et al., 2026; In et al., 2026).

In this paper, we evaluate these benchmarks themselves. A high score should indicate that the judge identified the faulty step by checking it against the rest of the trajectory (a cross-step check). If the score can instead be obtained from spurious signals (surface cues) that predict the annotation, the benchmark rewards cue use rather than debugging: leaderboards mislead, and judges trained on its labels learn the cues and then blame the wrong step in runs where the cues do not hold. Such cues have long misled NLP benchmarks (Gururangan et al., 2018; Geirhos et al., 2020), and LLM judges are swayed by surface features (Zheng et al., 2023; Chen et al., 2024); failure attribution faces both problems. Auditing a benchmark for such cues is therefore a prerequisite for trusting its results.

We audit three recent benchmarks that differ in how failures arise and how answers are scored. Who&When Pro injects an error at one step into runs of several agent systems, such as the textgame agent ALFAgent, and asks for the faulty step (Liu et al., 2026). AFTraj mixes such injected errors with natural failures that agents make on their own and asks for the first faulty turn (Zhang et al., 2026). Aegis asks for the set of faulty agents in a multi-agent conversation (Kong et al., 2026).

Each benchmark has a limitation that lets a predictor without a cross-step check match or beat GPT-4.1. On Who&When Pro, one step’s text gives the error away: a classifier reading one step’s reasoning at a time picks the faulty step in 95–100% of runs from the two agent systems that record reasoning, ALFAgent and smolagents (Table 2), whereas GPT-4.1 picks it in 58.8% of ALFAgent runs. In ALFAgent, error injection also leaves no reasoning after the faulty step, so picking the last step with reasoning finds the error 99.7% of the time. Judges fine-tuned on ALFAgent labels learn this cue: even at 1.5B parameters, they score 99.7– 100%, 41 points above GPT-4.1, yet lose 31 to 99 points when we add task-related reasoning after the error, while zero-shot judges lose at most 9. On AFTraj, a classifier that reads one turn at a time ties GPT-4.1 only because it wins on injected errors and loses on natural failures, so the benchmark’s mix of the two decides the ranking. On Aegis, labeling every agent faulty ties GPT-4.1 on micro-F1 but trails it by 19 points in exact-set accuracy (Table 4).

These limitations motivate a four-test audit framework that systematically checks whether a benchmark score reflects cross-step checks. Input restriction asks whether the annotation can be predicted from a restricted input alone: one step’s text, step positions, the presence of reasoning text, or agent names. An input that predicts the annotation is a cue. Cue following asks whether the judge uses a cue; input restriction cannot show this (Feng et al., 2019). Keeping the error fixed, we move or reword the cue, as when we add reasoning after ALFAgent’s faulty step, and check whether the answer changes. Residual accuracy is the judge’s accuracy on the trajectories that a cue-only predictor gets wrong; it matters when a cue alone finds nearly every error, because overall accuracy then cannot separate checking from cue use. Finally, aggregation asks whether a comparison between two methods survives a change in the failure mix (AF-Traj) or the scoring metric (Aegis). The tests need no new annotation and apply to any benchmark that labels a faulty step or agent. They complement one another: on each benchmark, one test or metric alone would have misled us (Section 7).

Our contributions are:

• A four-test audit framework that separates cross-step checks from cue use (Section 3).

• Identification of limitations in three failureattribution benchmarks: cues in Who&When Pro, a failure mix that decides rankings in AFTraj, and a lenient metric in Aegis (Sections 4–6).

• Evidence that fine-tuning on benchmark labels teaches judges to rely on where ALFAgent’s reasoning stops: they beat GPT-4.1 by 41 points, but adding task-related reasoning after that point costs them up to 99 points (Section 4).

• An analysis of what each test alone would miss, with a reporting template for benchmark developers (Section 7, Table 5).

Table 1: Evaluated records, and the inputs of each restricted-input predictor (Section 3.1); semicolons separate predictors.
<table><tr><td>Benchmark / subset (records)</td><td>Restricted inputs</td></tr><tr><td>Who&amp;When Pro</td><td></td></tr><tr><td>All text (6,257)</td><td>step position</td></tr><tr><td>ALFAgent text (311)</td><td>presence of reasoning; step text; reasoning length</td></tr><tr><td>smolagents text (3,276)</td><td>step text; step position</td></tr><tr><td>Video (DVD 168, EVA 748)</td><td>step position</td></tr><tr><td>AFTraj</td><td></td></tr><tr><td>Official test (163)</td><td>step position; step text</td></tr><tr><td>Aegis (600)</td><td>ordered speaker names</td></tr></table>

## 2 Benchmarks and Judges

Table 1 lists the records we evaluate. Who&When Pro provides text trajectories with injected errors and video-agent trajectories from DeepVideoDiscovery (DVD) and EVA, each annotated with a faulty step (Liu et al., 2026). Two of its eight text systems, ALFAgent and smolagents, record the agent’s reasoning. We run judges on the 311 trajectories of ALFAgent, the only text system in which error injection makes the reasoning stop at the error (Section 4; Appendix Table 6). AFTraj mixes injected and natural failures in coding, mathematics, and agentic tasks (Zhang et al., 2026). Aegis asks for the set of faulty agents, called speakers (Kong et al., 2026). On ALFAgent the judges are three zero-shot LLMs (GPT-4.1, Claude Sonnet 5, o3) and two open models fine-tuned on ALFAgent labels; on AFTraj, four zero-shot LLMs (GPT-4.1, DeepSeek V4 Pro, Qwen3-8B, Qwen3-32B) and two trained classifiers, one reading every step so far and one reading only the current step (Section 5.1); on video and Aegis, GPT-4.1.

## 3 Four-Test Audit Framework

The four tests ask whether a cue exists, whether the judge uses it, whether the judge succeeds where the cue fails, and whether a ranking survives a change of failure mix or metric (Figure 1). Throughout, f is a judge, $X = ( x _ { 1 } , \dots , x _ { T } )$ a trajectory with annotation $Y _ { \textrm { \textrm { \textrm { \textrm { \textrm { \textrm { \textrm { F } } } } } } } }$ , and $\scriptstyle A _ { P } ( f )$ the judge’s exact accuracy on a set of records P: the fraction on which f returns the annotated step or, for speakers, exactly the annotated set. All intervals are 95% bootstrap intervals over groups of records from the same task.

![](images/0c4ac6c8e0af77cd8a7bbd929983fcbbc4900cae71491be3ad66df5491bd6ab8.jpg)  
Figure 1: Four tests for attribution scores. Each box gives the question a test asks and how we run it (Section 3).

## 3.1 Input restriction

Locating an error such as the claim in Figure 2b that a book is ready to be placed requires a crossstep check: an earlier step says that the book was not found. The annotation may be predictable without that check. We therefore give a predictor h only part r(X) of the input (one step’s text, step positions, whether reasoning is present, or speaker names). We select or train h on tasks disjoint from evaluation (Appendix A) and report $\boldsymbol { \mathcal { A } } _ { P } ( h , r )$ , the exact accuracy of h when it sees only $r ( X )$ , plus micro-F1 for speaker sets. Most of our predictors score each step on its own and pick the highest:

$$
\hat { Y } _ { \mathrm { l o c a l } } = \underset { 1 \leq t \leq T } { \arg \operatorname* { m a x } } s ( x _ { t } , t , T ) .\tag{1}
$$

The scorer s sees one step’s text, its position, and the trajectory length, and nothing from other steps, so a predictor of this form cannot perform a crossstep check. If it recovers the annotation anyway, the annotation does not require that check, and a high score cannot show that a judge did it. A step’s position tells only how often errors fall there, and where reasoning stops can be an artifact of error injection. Neither is evidence of the mistake.

## 3.2 Cue following

Predictability shows that a cue exists, not that the judge uses it. To test that, we alter the cue while leaving the error where it is and ask whether the judge’s answer moves with it. Let c(X) be the step a cue points to (e.g., the last step with reasoning) and τ an intervention that moves the cue. We keep the records where the judge originally answered c(X) and τ moved the cue, $D _ { c } = \{ X : f ( X ) =$ $c ( X ) , \ c ( \tau ( X ) ) \neq \ c ( X ) \}$ , and report how often the judge follows:

$$
{ \mathcal { F } } ( f , c , \tau ) = \operatorname* { P r } [ f ( \tau ( X ) ) = c ( \tau ( X ) ) \mid X \in D _ { c } ] .\tag{2}
$$

A judge that tracks the cue has $\mathcal { F }$ near one. A judge can also depend on a cue without following it: when the cue moves, its accuracy falls although its answers avoid $c ( \tau ( X ) )$ . We therefore also report how many answers in $D _ { c }$ name the faulty step after the move, and the accuracy change of the judge and, as a baseline, of a predictor that cannot see the cue. A wording cue, such as the error step’s phrasing, has no position to follow, so we measure accuracy instead. We paraphrase the cue-bearing text with ρ and report the paired change $A _ { P } ( f \circ$ $\rho ) - \mathcal { A } _ { P } ( f )$ ; as a control, we also paraphrase a different step. A judge that relies on the wording should lose accuracy only when the cue-bearing text is paraphrased.

## 3.3 Residual accuracy

Residual accuracy is the judge’s accuracy on the records the predictor gets wrong; a high value shows that the judge finds errors the cue misses. Let G (gains) be the fraction of all records the judge gets right and the predictor gets wrong, and L (losses) the reverse. Shared successes and failures cancel, so

$$
\mathcal { A } _ { P } ( f ) - \mathcal { A } _ { P } ( h , r ) = G - L .\tag{3}
$$

A small gap can hide large gains and losses, so we report both G and L.

## 3.4 Aggregation

Even without a cue, a released score fixes one weighting of failure sources or domains and one metric. Holding every prediction fixed, we recompute the score under other weights. Two methods that lead on different groups (e.g., the injected and natural subsets of AFTraj) swap ranks at some crossing weight. If this weight lies near the released mixture, the ranking reflects how the benchmark was built rather than which method is better (Section 5.2). For speaker sets we compare micro-F1, which gives partial credit to a prediction that mixes faulty and non-faulty speakers, with exact-set accuracy, which gives none.

## 4 Who&When Pro Text Audits

One step’s text gives away the faulty step in both Who&When Pro systems that record reasoning, ALFAgent and smolagents. ALFAgent, whose runs have 30 steps and one injected error (Liu et al., 2026), has a second cue, where its reasoning stops, so we test there whether judges use these cues.

## 4.1 Input restriction: one step’s text

One step read alone finds every ALFAgent error. Our independent-step classifier, a logistic regression on TF–IDF features of each step with field names removed, picks the highest-scoring step. On five task-disjoint folds it selects the annotated step in all 311 trajectories when it sees the full step text or the reasoning field alone (Table 2); without the reasoning field, accuracy drops to 21.2%. Phrases that recur across error steps, such as “repetitive loop”, do not explain this: deleting them leaves accuracy at 98.4% (Appendix B.4). Its largest weights go to common words such as already, so, and now (Appendix Table 7B and Appendix B.6).

This cue does not depend on where the reasoning stops. In smolagents (3,276 trajectories), the reasoning runs to the final step in almost every trajectory, and the final step is the error in only 52.1%. The same classifier still finds the annotated step in 95.2% [94.3, 96.0] of trajectories from the reasoning alone, and in 90.8% of those whose error is not at the final step (Appendix B.5). In both systems, a top score therefore needs no cross-step check.

## 4.2 Input restriction: where reasoning stops

In 310 of 311 ALFAgent trajectories, reasoning runs through the annotated step and stops there (the reasoning boundary). Predicting the last step with reasoning (the boundary rule) is therefore 99.7% accurate without reading a word. This is because Who&When Pro injects an error by adding an instruction to the agent’s LLM prompt at one step and then resumes execution after that step (Liu et al., 2026, their Section 3.3), and in ALFAgent the resumed steps carry no reasoning. Position, a strong cue in most other systems, is not one here: a rule that always picks the final or the penultimate step, whichever cross-validation prefers, is right on only 2 of 311 (Appendix B.2).

Table 2: Exact accuracy (%) of the independent-step classifier by what it sees of each step (field names removed; five task-disjoint folds).
<table><tr><td>Available information</td><td>ALFAgent smolagents</td><td></td></tr><tr><td>All step text</td><td>100.0</td><td>91.9</td></tr><tr><td>Reasoning only</td><td>100.0</td><td>95.2</td></tr><tr><td>All text except reasoning</td><td>21.2</td><td>72.1</td></tr></table>

## 4.3 Cue following

We present each of the 311 trajectories in four input versions and ask GPT-4.1, Claude Sonnet 5, and o3 to identify the error step. Original preserves the released fields, Field-neutral removes field names, and No reasoning removes reasoning text. Template continued preserves the original text and adds “The trajectory continues with the information available at this step.” to each empty reasoning field after the error, which moves the reasoning boundary to step 30 but not the error.

The zero-shot judges use the reasoning but do not follow the boundary. Under Original they score 58.8% (GPT-4.1), 57.2% (Claude Sonnet 5), and 56.3% (o3) (Figure 3a; Appendix Table 10), and removing reasoning lowers them by 43.7, 42.1, and 45.7 points. Under Template continued, only 1 of the three judges’ 536 correct predictions at the old boundary moves to step 30 (Figure 3b), and accuracy falls by at most 2.6 points.

Task continued fills the empty fields after the error with task-related reasoning written by DeepSeek V4 Pro from the task and earlier steps, moving the boundary to step 30. This text does not pull GPT-4.1 or Claude Sonnet 5 to the new boundary: none of their 361 correct predictions at the old boundary moves to step 30, and accuracy changes by −4.2 points for GPT-4.1 [−8.4, 0.0] and −9.0 for Claude Sonnet 5 (Appendix B.7).

![](images/caacba463f2f792640770864dc18363a365dc4e7566741891cdd5ef3ceb0aa11.jpg)  
The independent-step classifier finds both labels without comparing steps.  
Figure 2: Two ALFAgent errors, shaded orange: the agent substitutes a CD for a cellphone (a) and declares a book ready to be placed before it has found one (b). The classifier scores each step alone, without the task or other steps such as step 7 in (b), which notes that the book is not there.

Paraphrasing the error step. To test wording, we paraphrase the reasoning at the annotated error step (Gold paraphrased) and, as a control, the reasoning at the step before it (Prior paraphrased). DeepSeek V4 Pro rewrites each string without seeing the label or the trajectory.

Rewording the error step leaves both tested judges’ accuracy nearly unchanged. Relative to Original, GPT-4.1 changes by +1.4 points [−2.2, +4.9] under Gold paraphrased and Claude Sonnet 5 by +0.4; the control lowers each judge by less than 2 points, and every interval includes zero. The rewrites keep the claims (56 of 60 humanreviewed pairs are equivalent, 4 differ slightly) and weaken the wording cue without removing it: the classifier, not retrained, still selects the annotated step in 273 of 285 records (Appendix B.8).

## 4.4 Residual accuracy and fine-tuned judges

The zero-shot judges thus neither follow the cues nor reach the scores the cues allow: moving the boundary costs them at most 9 points, and none scores above 63% on any input, far below the classifier’s 100%. Residual accuracy credits them with nothing beyond the cues: the boundary rule misses one trajectory, which no zero-shot judge gets right under Original, and the classifier misses none.

Fine-tuned judges. Judges fine-tuned on the labels learn the boundary. We fine-tune Qwen2.5- 1.5B and Qwen3-8B with two seeds each on ALFAgent’s Original input, using the classifier’s five taskdisjoint folds (Appendix B.9). All four fine-tuned judges reach 99.7–100% on Original, 41 points above GPT-4.1, yet Task continued costs them 31 to 99 points: 97.7 and 99.0 for the 1.5B model and

35.7 and 30.9 for the 8B model. The same text costs Qwen3-8B 4.5 points before fine-tuning and costs the classifier, which cannot see the boundary, 9.0 points, so the added text alone explains at most 9 of the lost points. The fine-tuned judges depend on the boundary without following it (Section 3.2): for each fine-tuned judge, at most 8.4% of the answers at the old boundary (26 of 310) move to step 30, and every answer that turns wrong lands past the error, mostly toward the end. A generic sentence (Template continued) separates zero-shot and fine-tuned judges less sharply: the ranges of their losses overlap (Appendix Tables 10 and 13). On the second-seed models, two controls cost nothing: adding the generic sentence to every step before the error, which keeps the boundary, and rewording the error step. Because the fine-tuned judges score near 100%, a leaderboard score cannot tell them from judges that check across steps.

## 5 AFTraj

On AFTraj, a classifier that sees each turn (step) alone ties GPT-4.1 only because it wins on injected failures and loses on natural ones, and the released mix is where the two cancel.

## 5.1 Input restriction and residual accuracy

AFTraj shows each method a growing prefix of the trajectory: after each new turn, the method says whether a failure has occurred so far and, if it has, at which turn. The turn named in its first alarm is its predicted error turn. The official test set, called the first-alarm test, has 163 failures: 108 injected errors and 55 natural failures.

We compare six methods (Appendix C.1). The four zero-shot judges of Section 2 and Probe, a linear classifier on frozen Qwen3-8B hidden states, see every turn so far. Current-turn TF–IDF, a restricted-input predictor, sees only the words of the current turn. The two classifiers are trained on 1,264 trajectories from other tasks.

![](images/3ad5dfc65e4595dc8da27b7d01b2cc8db59293b90b629ebb49c5b0115a6b6d8e.jpg)

![](images/81c8cfa4bacac5cc2335034722ef86d06d0791b1d3ed7fcbc85eb00ea8b94a8f.jpg)

![](images/0480276b3224d4e8e9d57ce84e6eb4f374c60c3fbbc6efedf5969446813dc953.jpg)  
Figure 3: ALFAgent reasoning interventions. (a) Exact step accuracy with 95% task-group bootstrap intervals. The first four rows, Original to Template continued, use all 311 records. The last three rows (Original, Gold paraphrased, and Prior paraphrased) use the 285 records of the paraphrase test (Appendix B.8), with GPT-4.1 and Claude Sonnet 5 only. (b) Template continued moves the boundary to step 30.

The tie hides two different sets of successes. Current-turn TF–IDF and GPT-4.1 are each right on 82 of the 163 failures (50.3%) but share only 36. GPT-4.1 is right on 46 of the 81 failures TF– IDF misses (residual accuracy 56.8% [45.8, 67.6], above its overall 50.3%) and wrong on 46 that TF– IDF gets right (Appendix C.2). Position alone does not explain TF–IDF’s score. A predictor that always alarms at the fourth turn, chosen by crossvalidation, is right on 42 of 163 (25.8%).

Table 3: AFTraj first-alarm accuracy (%) by failure source for four of the six methods (all in Appendix Table 15). Released: 108 injected, 55 natural.
<table><tr><td>Method</td><td>Injected</td><td>Natural</td><td>Released</td></tr><tr><td>Current-turn TF-IDF</td><td>65.7</td><td>20.0</td><td>50.3</td></tr><tr><td>GPT-4.1</td><td>48.1</td><td>54.5</td><td>50.3</td></tr><tr><td>Probe</td><td>77.8</td><td>30.9</td><td>62.0</td></tr><tr><td>DeepSeek V4 Pro</td><td>63.9</td><td>54.5</td><td>60.7</td></tr></table>

## 5.2 Aggregation: injected and natural failures

The released mix of failure sources decides the tie (Table 3). On injected failures TF–IDF is right on 65.7% and GPT-4.1 on 48.1%, a lead of 17.6 points [4.0, 31.5]. On natural failures GPT-4.1 is right on 54.5% and TF–IDF on 20.0%, a gap of 34.5 points [15.4, 52.0]. At the released share, 66.3% injected, these two opposite gaps cancel: TF–IDF is right on 19 more injected failures and 19 fewer natural ones. The mix also decides the top rank. Probe, the most accurate method, owes its lead to injected failures and ranks above the runner-up, DeepSeek V4 Pro, only when injected failures make up more than 63.0%, near the released 66.3% (Appendix C.3).

Whether failure source or domain drives these differences, the test set’s composition sets the ranking. The first-alarm test alone cannot separate source from domain: every coding failure is injected, and most agentic failures are natural. On its 68 agentic failures (22 injected, 46 natural), the injected-minus-natural accuracy gap shrinks from 45.7 points on the full test set to 12.5 for currentturn TF–IDF and from 46.9 to 8.5 for Probe, and every interval includes zero (Appendix Table 15B).

Within one domain, injected errors still carry recognizable text in the turn itself, as on ALFAgent. Because the test set has only 68 agentic failures, a separate experiment uses 468 agentic failures from the AFTraj corpus (148 injected, 320 natural; Appendix C.4). Two classifiers each pick the highestscoring turn: a word classifier reads the turn’s text, and a structure classifier sees only the turn’s role, position, length, and thought:/action: markers. The word classifier beats the structure classifier on injected failures (64.2% [56.8, 71.6] against 44.6% [36.5, 52.7]) but not on natural ones (58.1% against 63.1%), so the words of a turn carry a signal beyond its structure only for injected errors.

## 6 Aegis

Micro-F1 cannot tell GPT-4.1 from labeling every speaker faulty, but exact-set accuracy can.

On micro-F1, GPT-4.1 ties the all-speakers predictor (Appendix D), which sees only the speaker list: the predictor scores 49.2% and GPT-4.1 50.0%, and the paired interval of the 0.8-point gap is [−1.3, +2.8]. The benchmark’s macro-F1, averaged over speaker names, even ranks the predictor higher (Appendix Table 17). Exact-set accuracy separates them: 1.5% for the predictor and 20.7% for GPT-4.1, a gap of 19.2 points [15.6, 22.8]. GPT-4.1 is exactly right on 124 of the 591 conversations where the predictor is wrong (residual accuracy 21.0%, Table 4B), and it loses only the predictor’s 9 correct conversations.

Micro-F1 lets recall pay for false alarms. The predictor names everyone, so its precision is only 37.8%; its recall is 70.2%, not 100%, because some annotated names match no listed speaker. For a developer, each non-faulty speaker flagged is a cost.

Name formatting does not explain why micro-F1 and exact-set accuracy disagree. Of the 417 annotated name occurrences that match no listed speaker, 275 differ from a listed name only in case or surrounding whitespace; after fixing these, GPT-4.1 leads by only 4.3 points [2.3, 6.3] on micro-F1 but by 19.5 points on exact sets (Table 4).

## 7 What Each Test Would Have Missed

On each benchmark, one test or metric alone would have misled us; another test corrects it.

On ALFAgent, cue following on zero-shot judges alone would suggest that the cues are harmless: under Task continued, none of GPT-4.1’s and Claude Sonnet 5’s 361 answers at the old boundary moves to step 30. Input restriction corrects this by showing that the cues give the labels away. On fine-tuned judges, the follow rate alone would mislead in the same way, since at most 8.4% of their answers move to step 30; their accuracy drop of 31 to 99 points shows they rely on the boundary.

On AFTraj, input restriction alone shows currentturn TF–IDF tying GPT-4.1 at 82/163, as if equally good. Aggregation corrects this: GPT-4.1 is 34.5 points ahead on natural and 17.6 behind on injected failures, so they tie only at the released mix.

On Aegis, micro-F1 alone would equate GPT-4.1 with labeling every speaker faulty; exact-set accuracy shows a 19-point gap.

On the 168 DVD trajectories (Appendix E), residual accuracy alone would overrate GPT-4.1: it finds 27 (60.0%) of the 45 errors missed by always guessing the final step, but misses 39 that this guess finds, so it trails the guess by 7.1 points.

## 8 Related Work

Input restriction and cue following. Partialinput baselines, which expose shortcut learning (Geirhos et al., 2020), predict NLI labels from the hypothesis alone (Gururangan et al., 2018; Poliak et al., 2018) and reading-comprehension answers from the question, the passage, or one sentence (Kaushik and Lipton, 2018; Chen et al., 2016). A successful partial-input baseline shows that a cue exists, not that models use it (Feng et al., 2019). Whether a model uses a cue is usually tested on new examples that break it (Niven and Kao, 2019; Mc-Coy et al., 2019) or on examples from unseen annotators, whose writing lacks the training annotators’ cues (Geva et al., 2019). Cue following instead edits the same trajectory, moving one cue but not the error, so any change in the answer comes from the cue alone; measuring accuracy also catches judges that depend on a cue without following it.

Residual accuracy and aggregation. Hardsubset analyses score models on questions a heuristic cannot answer (Sugawara et al., 2018). Residual accuracy reflects only gains, but the judge’s accuracy minus the predictor’s equals gains minus losses (Equation 3), so we also report losses. Test populations and aggregation rules can decide rankings (Binette and Reiter, 2024; Zhang and Hardt, 2024), and benchmark validity requires rankings that support claims about the intended capability (Bowman and Dahl, 2021; Bean et al., 2025; Freiesleben and Zezulka, 2026). We add the failuremix weight at which two methods swap ranks and its distance from the released mixture.

Attribution benchmarks and judges. We audit Who&When Pro (Liu et al., 2026), which succeeds Who&When (Zhang et al., 2025), as well as AFTraj (Zhang et al., 2026) and Aegis (Kong et al., 2026). Other benchmarks allow several valid attributions per failure (In et al., 2026) or log each agent’s full inputs (Chen et al., 2026), and DoVer checks an attribution by editing the blamed step and rerunning (Ma et al., 2026). Closest to our method is the shortcut test in CatchBench (Zhao et al., 2026), a benchmark for predicting agent failures from the configuration declared before a run. A rule that ignores the configuration’s content reaches perfect F1 on one of its six data sources, and its authors argue that a score cannot be interpreted until the labeling process has been tested for such shortcuts. We go further by testing whether judges use such cues, and find that fine-tuning on benchmark labels teaches them to. LLM judges of agent trajectories accept failed runs that end in confident language (Advani, 2026) and raise many false alarms that recall alone hides (Mohammadi, 2026). Unlike these studies, which ask whether a run failed, we target cues that point to the faulty step.

Table 4: GPT-4.1 versus the all-speakers predictor (labels every speaker faulty) on 600 Aegis conversations. A: scores (%) and GPT-4.1 minus predictor (points, paired 95% bootstrap intervals); in normalized rows, names are lowercased and trimmed. B: conversations where both, only GPT-4.1, only the predictor, or neither is exactly right.  
A. Paired scores
<table><tr><td>Names</td><td>System</td><td>Precision</td><td>Recall</td><td>Micro-F1</td><td>Exact set</td></tr><tr><td>As released</td><td>GPT-4.1</td><td>56.0</td><td>45.1</td><td>50.0</td><td>20.7 (124/600)</td></tr><tr><td></td><td>All speakers</td><td>37.8</td><td>70.2</td><td>49.2</td><td>1.5 (9/600)</td></tr><tr><td></td><td>Difference</td><td></td><td></td><td>+0.8 [−1.3, +2.8]</td><td>+19.2 [+15.6, +22.8]</td></tr><tr><td>Normalized</td><td>GPT-4.1</td><td>75.4</td><td>60.7</td><td>67.3</td><td>23.0 (138/600)</td></tr><tr><td></td><td>All speakers</td><td>48.4</td><td>89.9</td><td>62.9</td><td>3.5 (21/600)</td></tr><tr><td></td><td>Difference</td><td></td><td></td><td>+4.3 [+2.3, +6.3]</td><td>+19.5 [+15.9, +23.2]</td></tr></table>

B. Exact-set agreement on 600 conversations
<table><tr><td>Names</td><td>Both correct</td><td>GPT-4.1 only</td><td>All speakers only</td><td>Both wrong</td><td>Residual accuracy (%, 95% interval)</td></tr><tr><td>As released</td><td>0</td><td>124</td><td></td><td>467</td><td>124/591 = 21.0 [17.7,24.4]</td></tr><tr><td>Normalized</td><td>9</td><td>129</td><td>12</td><td>450</td><td>129/579 = 22.3 [18.9,25.9]</td></tr></table>

Table 5: The four tests on each benchmark or subset (Sections 4–6, Appendix C.3; video: Appendix E). pp: percentage points; “—”: not run.
<table><tr><td></td><td>Input restriction</td><td>Cue following</td><td>Residual accuracy</td><td>Aggregation</td></tr><tr><td>ALFAgent Independent-step</td><td>classifier 311/311 (95.2% on smolagents); pp; fine-tuned judges at most</td><td>Task continued: zero-shot judges 0 of 361 follow, lose at most 9.0 boundary rule 310/311. 8.4% follow, yet lose 31–99 pp.</td><td>The classifier misses none; no zero-shot judge finds the boundary rule&#x27;s one miss.</td><td></td></tr><tr><td>AFTraj</td><td>Current-turn TF-IDF and GPT-4.1 both 82/163.</td><td></td><td>Against current-turn TF-IDF: GPT-4.1 gains 46 and loses 46 (residual accuracy 56.8%).</td><td>TF–IDF vs. GPT-4.1: +17.6 pp on injected, —34.5 pp on natural. Probe and DeepSeek V4 Pro swap ranks at 63.0% injected.</td></tr><tr><td>Aegis</td><td>All speakers 49.2% micro-F1, GPT-4.1 50.0%.</td><td></td><td>Against all speakers: GPT-4.1 gains 124 exact sets and loses 9. 19.2 pp on exact sets.</td><td>Gap to all speakers: 0.8 pp on micro-F1 (interval includes 0),</td></tr><tr><td>Video</td><td>Final-step guess 63.3%, — GPT-4.1 59.8% (mean of DVD and EVA).</td><td></td><td>Against the final-step guess (DVD): GPT-4.1 gains 27, loses 39.</td><td></td></tr></table>

## 9 Conclusion

On all three failure-attribution benchmarks we audit, a predictor without a cross-step check matches or beats GPT-4.1. On Who&When Pro, one step’s text gives away the labels in both systems that record reasoning, and judges fine-tuned on ALFAgent beat GPT-4.1 by 41 points but lose 31 to 99 when task-related reasoning is added after the error; on AFTraj, a classifier that sees one turn ties GPT-4.1 only at the released failure mix; on Aegis, micro-F1 hides GPT-4.1’s 19-point exact-set lead over blaming every agent. A high score alone therefore cannot show that a judge checks across steps, which motivates our four tests. The tests need no new annotation, so we recommend that benchmark developers report the applicable tests next to the main score, using Table 5 as a template.

## Limitations

We run cue following only on ALFAgent, where one step’s text recovers all 311 labels and all errors come from one injection pipeline. The judges we run on ALFAgent answer our own prompt, not the benchmark’s (Appendix B.3). Our fine-tuned judges cover one subset, two model sizes, and two seeds each; their Task continued losses hold across seeds, but their Template continued losses vary widely (Table 13). We use three zero-shot judges on ALFAgent but only GPT-4.1 on video and Aegis, and we evaluate Aegis on one release of its data (Appendix D.3).

Some results have other possible explanations. On ALFAgent, the error-step text that the classifier picks up may state the mistake plainly enough for a careful reader to spot it from that text alone, or it may carry traces of how the error was injected; our tests do not separate these two cases. The paraphrase test changes only the wording of the error step and keeps its claims, so it cannot show whether judges rely on those claims. An automatic check that the paraphrases keep their meaning (Appendix B.8) used GPT-4.1, which is also a judge. The 60 human-reviewed pairs do not depend on this check. On AFTraj, every coding failure is injected, so part of the injected–natural difference may reflect domain. Within the first-alarm test’s 68 agentic failures, the injected-minus-natural gap keeps its sign for five of six methods, but every interval for this gap includes zero. The separate experiment on 468 agentic failures shows that injected errors carry recognizable text within the agentic domain too.

Finally, we score only against the released annotations, so a judge that blames a different but also valid step or agent counts as wrong.

## Ethical Considerations

This work audits public benchmarks; it collects no data from people and releases no new model or dataset. Its main risk is that the cues it documents could be used to inflate a judge’s score. The four tests are designed to expose such inflation, and reporting them next to the main score makes it visible. We use the three benchmarks under their licenses (Appendix A). The authors themselves carried out the human checks of generated text (Appendices B.7 and B.8). All experiments together used about 20 GPU-hours (Appendix A).

## References

Laksh Advani. 2026. From confident closing to silent failure: Characterizing false success in LLM agents.

arXiv preprint arXiv:2606.09863.

Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, Franziska Sofia Hafner, Harry Mayne, Jan Batzner, Negar Foroutan, Chris Schmitz, Karolina Korgul, Hunar Batra, Oishi Deb, Emma Beharry, Cornelius Emde, Thomas Foster, Anna Gausen, María Grandury, Simeng Han, Valentin Hofmann, Lujain Ibrahim, and 23 others. 2025. Measuring what matters: Construct validity in large language model benchmarks. In Advances in Neural Information Processing Systems, volume 38.

Olivier Binette and Jerome P. Reiter. 2024. Improving the validity and practical usefulness of AI/ML evaluations using an estimands framework. arXiv preprint arXiv:2406.10366.

Samuel R. Bowman and George E. Dahl. 2021. What will it take to fix benchmarking in natural language understanding? In Proceedings ofthe 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4843–4855, Online. Association for Computational Linguistics.

Danqi Chen, Jason Bolton, and Christopher D. Manning. 2016. A thorough examination of the CNN/Daily Mail reading comprehension task. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2358–2367, Berlin, Germany. Association for Computational Linguistics.

Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. 2024. Humans or LLMs as the judge? A study on judgement bias. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 8301–8327, Miami, Florida, USA. Association for Computational Linguistics.

Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, and Qing Wang. 2026. Seeing the whole elephant: A benchmark for failure attribution in LLM-based multi-agent systems. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19888–19905, San Diego, California, United States. Association for Computational Linguistics.

Shi Feng, Eric Wallace, and Jordan Boyd-Graber. 2019. Misleading failures of partial-input baselines. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5533– 5538, Florence, Italy. Association for Computational Linguistics.

Timo Freiesleben and Sebastian Zezulka. 2026. The benchmarking epistemology: Validity theory for evaluating machine learning models. Philosophy ofScience, pages 1–30.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel,

Matthias Bethge, and Felix A. Wichmann. 2020. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673.

Mor Geva, Yoav Goldberg, and Jonathan Berant. 2019. Are we modeling the task or the annotator? An investigation of annotator bias in natural language understanding datasets. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1161–1166, Hong Kong, China. Association for Computational Linguistics.

Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R. Bowman, and Noah A. Smith. 2018. Annotation artifacts in natural language inference data. In Proceedings of the 2018 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 107–112, New Orleans, Louisiana. Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR).

Yeonjun In, Mehrab Tanjim, Jayakumar Subramanian, Sungchul Kim, Uttaran Bhattacharya, Wonjoong Kim, Sangwu Park, Somdeb Sarkhel, and Chanyoung Park. 2026. Rethinking failure attribution in multi-agent systems: A multi-perspective benchmark and evaluation. arXiv preprint arXiv:2603.25001.

Divyansh Kaushik and Zachary C. Lipton. 2018. How much reading does reading comprehension require? A critical investigation of popular benchmarks. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 5010–5015, Brussels, Belgium. Association for Computational Linguistics.

Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, and Xue Feng. 2026. Aegis: Automated error generation and attribution for multi-agent systems. In International Conference on Learning Representations.

Jiale Liu, Huajun Xi, Shaokun Zhang, Yifan Zeng, Tianwei Yue, Chi Wang, Jian Kang, Qingyun Wu, and Huazheng Wang. 2026. Who&When Pro: Can LLMs really attribute failures in AI agents? arXiv preprint arXiv:2607.09996.

Ming Ma, Jue Zhang, Fangkai Yang, Yu Kang, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. 2026. DoVer: Intervention-driven auto debugging for LLM multi-agent systems. In International Conference on Learning Representations.

R. Thomas McCoy, Ellie Pavlick, and Tal Linzen. 2019. Right for the wrong reasons: Diagnosing syntactic

heuristics in natural language inference. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3428–3448, Florence, Italy. Association for Computational Linguistics.

Hadi Mohammadi. 2026. trajectory-judge: What outcome-only LLM judges miss on agent trajectories. arXiv preprint arXiv:2609.00038.

Timothy Niven and Hung-Yu Kao. 2019. Probing neural network comprehension of natural language arguments. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 4658–4664, Florence, Italy. Association for Computational Linguistics.

Adam Poliak, Jason Naradowsky, Aparajita Haldar, Rachel Rudinger, and Benjamin Van Durme. 2018. Hypothesis only baselines in natural language inference. In Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pages 180–191, New Orleans, Louisiana. Association for Computational Linguistics.

Qwen Team. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Qwen Team. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Saku Sugawara, Kentaro Inui, Satoshi Sekine, and Akiko Aizawa. 2018. What makes reading comprehension questions easier? In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4208–4219, Brussels, Belgium. Association for Computational Linguistics.

Boxuan Zhang, Jianing Zhu, Zeru Shi, Dongfang Liu, and Ruixiang Tang. 2026. AgentForesight: Online auditing for early failure prediction in multi-agent systems. arXiv preprint arXiv:2605.08715.

Guanhua Zhang and Moritz Hardt. 2024. Inherent trade-offs between diversity and stability in multitask benchmarks. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 58984–59002. PMLR.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. 2025. Which agent causes task failures and when? On automated failure attribution of LLM multi-agent systems. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 76583–76599. PMLR.

Yue Zhao, Mengyuan Li, Ruolin Li, Prince Zizhuang Wang, Shuli Jiang, Linsey Pang, Xiongye Xiao, and Xiyang Hu. 2026. CatchBench: When can an agent failure be caught? arXiv preprint arXiv:2608.22808.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623.

## A Evaluation Details

Task groups. Records that come from the same task form a task group. For Aegis, conversations built from the same user query form a query group. Whenever we split or resample data, a group stays together.

Data. Who&When Pro text has 6,257 trajectories from eight agent systems. The boundary rule (Appendix B.1) and the position rules, which always predict the final or the penultimate step (Appendix B.2), are evaluated on all of them. The independent-step classifier on smolagents uses that system’s 3,276 trajectories, which form 1,411 task groups (Appendix B.5). Every other text experiment uses the 311 ALFAgent trajectories (119 task groups, 30 steps each).

Some ALFAgent tests use fewer judges. Among the three zero-shot judges, Task continued and the paraphrase test (Section 4.3) are run with GPT-4.1 and Claude Sonnet 5 only, without o3. A control version of Template continued, called Template before error, appends the same sentence to every step before the error, which keeps the reasoning boundary at the error. Template before error is run on three models only, the two second-seed finetuned models and Qwen3-8B before fine-tuning (Table 13), and not on GPT-4.1, Claude Sonnet 5, or o3. These three Qwen models are also the only Qwen models that run the paraphrase test.

The paraphrase test uses 285 of the 311 trajectories, because it needs a step with reasoning before the error and the 26 trajectories whose error is at step 1 have none. One of the 26 has no reasoning at any step, so it is also the one trajectory that the boundary rule misses.

Who&When Pro video has 168 DVD and 748 EVA trajectories. Video results that combine the two systems take the mean of their accuracies, so DVD and EVA count equally. AFTraj uses the 163 failures in the first-alarm test. Aegis uses 600 conversations in 441 query groups. Step and turn numbers follow each benchmark’s released numbering; AFTraj turns start at 0.

Choosing restricted-input predictors. A restricted-input predictor must not be chosen on the records it is scored on. The Aegis all-speakers predictor and the AFTraj fixed-turn predictor are chosen from candidates on one half of the groups and scored on the other half, and then the halves are swapped. We sort the group identifiers and cut the sorted list in the middle, so the split is reproducible. The choice maximizes micro-F1 for Aegis and exact accuracy for AFTraj. Current-turn TF–IDF is trained on AFTraj tasks outside the first-alarm test (Appendix C.1). The Who&When Pro position rules, the video position predictor, and the independent-step classifiers on ALFAgent and smolagents are chosen or trained by five-fold cross-validation over task groups. The fine-tuned judges are trained on the same five folds as the ALFAgent classifier. The boundary rule (predict the last step with nonempty reasoning) has no parameter, so nothing is chosen. Appendices B–E list the candidate predictors.

Intervals and invalid answers. Every interval comes from resampling task groups (query groups for Aegis) 10,000 times. For a difference between two methods, both methods are scored on the same resample. The bootstrap seed is 0, except for video and for the 468-failure experiment on AFTraj (Appendix C.4), which use seed 1. An answer that cannot be scored counts as a miss. Every model returns a scorable answer on at least 95% of records in every experiment.

Models. The model identifiers are openai/gpt -4.1, anthropic/claude-sonnet-5, openai/o3 (medium reasoning effort), Qwen/Qwen2.5-1.5 B-Instruct, Qwen/Qwen3-8B, Qwen/Qwen3-32B, and deepseek/deepseek-v4-pro. DeepSeek V4 Pro has two roles. It is one of the AFTraj judges, and it writes the reasoning added in Task continued and the paraphrases used in the paraphrase test. The writing calls were served by snapshot deepse ek-v4-pro-0813.

Licenses. Who&When Pro is released under CC BY 4.0, AFTraj under the MIT License, and Aegis under the MIT License. We use the three benchmarks only to evaluate failure-attribution methods, which these licenses permit, and we do not redistribute their data.

Compute and software. Qwen2.5-1.5B, Qwen3- 8B, and Qwen3-32B have about 1.5, 8, and 32 billion parameters. GPT-4.1, Claude Sonnet 5, o3, and DeepSeek V4 Pro are called through their providers’ APIs. Fine-tuning and evaluating the Qwen judges of Appendix B.9 took about 8 GPUhours, with each run on one NVIDIA RTX A6000 GPU (48 GB). Extracting the Probe features and running the two Qwen3 judges on AFTraj took

Table 6: The boundary rule and the position rule on each Who&When Pro text system. The boundary rule predicts the last step with nonempty reasoning; the position rule (Appendix B.2) predicts the final or penultimate step, chosen per system. Reasoning: trajectories with any nonempty reasoning. Boundary and Position: the number of trajectories on which each rule is right, with the percentage of the system’s trajectories in parentheses. “—”: the system has no per-step reasoning, so the boundary rule makes no prediction.

<table><tr><td>System</td><td></td><td>N Reasoning</td><td>Boundary</td><td>Position</td></tr><tr><td>ALFAgent</td><td>311</td><td>310</td><td>310 (99.7)</td><td>2 (0.6)</td></tr><tr><td>smolagents</td><td>3,276</td><td></td><td>3,276 1,707 (52.1)</td><td>1,707 (52.1)</td></tr><tr><td>MathChat</td><td>920</td><td>0</td><td></td><td>725 (78.8)</td></tr><tr><td>Magentic-One</td><td>811</td><td>0</td><td></td><td>454 (56.0)</td></tr><tr><td>Debate</td><td>445</td><td>0</td><td></td><td>204 (45.8)</td></tr><tr><td>MacNet</td><td>261</td><td>0</td><td></td><td>109 (41.8)</td></tr><tr><td>DyLAN</td><td>152</td><td>0</td><td></td><td>0 (0.0)</td></tr><tr><td>MetaGPT</td><td>81</td><td>0</td><td></td><td>14 (17.3)</td></tr></table>

about 12 more GPU-hours. The TF–IDF classifiers run on CPUs. The fine-tuned judges use PyTorch 2.11.0, Transformers 5.19.0, and PEFT 0.21.2, and the independent-step classifiers use scikit-learn 1.7.2.

AI assistance. The authors used generative AI for analysis code, figures, and editing, and take responsibility for the interpretation and the submission.

## B Who&When Pro Text: Protocols and Complete Results

This appendix gives the details behind Section 4. Appendices B.1 and B.2 compare ALFAgent with the other Who&When Pro text systems. Appendix B.3 describes how the judges are prompted and what changes under the benchmark’s own prompt. Appendix B.4 describes the independentstep classifier and checks that its accuracy does not come from recurring phrases or from reasoning length. Appendix B.6 shows, with three examples, what the agent’s reasoning says at the error step. Appendices B.7 and B.8 give the construction and full results of the two interventions the main text summarizes, Task continued and the paraphrase test. Appendix B.5 runs the classifier on smolagents, and Appendix B.9 fine-tunes two judges on the labels.

## B.1 Reasoning boundary across text systems

The boundary rule tracks the error only in ALFAgent (Table 6). There it is right on 310 of

311 trajectories, because the reasoning stops before the final step, exactly at the error. No other system shows this.

In smolagents the rule amounts to guessing the final step. The reasoning field is nonempty through the last step in all but nine of the 3,276 trajectories, so the rule predicts the final step there. Its 1,707 hits are exactly the trajectories whose error is at the final step. In the nine trajectories where reasoning stops earlier, the rule is wrong.

The other six systems have no per-step reasoning field, so the rule makes no prediction, which is scored as a miss. Magentic-One has 1,358 fields named “reasoning,” but they are arguments of tool calls rather than a per-step reasoning field, so we do not count them.

## B.2 Position rules across text systems

Position, a strong cue in most other Who&When Pro text systems, is not one in ALFAgent. A position rule ignores the text and always predicts the same position. We allow two positions, the final step and the penultimate step, and choose one for each system by five-fold cross-validation over task groups. Every fold chooses penultimate for ALFAgent, MacNet, and MetaGPT, and final for the other five systems. The resulting rules are right on 3,215 of all 6,257 text trajectories (51.4%) and on 78.8% of MathChat trajectories, but on only 2 of the 311 ALFAgent trajectories.

Three scoring details affect these numbers. For Debate and DyLAN the annotation is a whole round, a block of consecutive steps, instead of a single step. A prediction therefore counts as correct if its step falls anywhere in that round. Some MacNet labels also list alternative steps, and the benchmark’s scorer counts a prediction at any of them as correct. Two Magentic-One trajectories have a single step and therefore no penultimate step; when cross-validation compares the two positions, the penultimate rule is scored as wrong on both. Table 6 gives the result for each system; 95% intervals from resampling task groups are in the supplementary material.

## B.3 Judge prompts and settings

All our ALFAgent judges, zero-shot and fine-tuned, receive the same simple prompt, which we call the controlled prompt; only in one experiment do GPT-4.1 and Claude Sonnet 5 also answer the benchmark’s own prompt (Table 9B). The controlled prompt shows the 30 numbered steps of one trajectory and asks only for the step of the injected error, as {"step": <integer>}. The gold label is never in the prompt.

GPT-4.1 and Claude Sonnet 5 run at temperature 0, and o3 runs at medium reasoning effort. o3 produces 15 invalid outputs out of 311 under No reasoning. Every other combination of judge and input version produces at most three.

The controlled prompt does not understate the judges: under the benchmark’s own prompt, their step accuracy is lower, 50.8% against 58.8% for GPT-4.1 and 51.4% against 57.2% for Claude Sonnet 5. That prompt asks, in a single response, for the agent that made the error, the step, the error mode (one of 17 in the benchmark’s taxonomy), and a one-line reason. We score the responses with the parser released with the benchmark (Table 9B, where Joint means that agent, step, and mode are all right). Both judges almost always name the agent correctly: GPT-4.1 in 100.0% of trajectories and Claude Sonnet 5 in 97.4%. Getting agent, step, and mode all right is much rarer: 25.1% for GPT-4.1 and 22.8% for Claude Sonnet 5.

Template continued raises o3’s accuracy by a net 4.5 points (Table 10). Under Template continued, o3 gives a different answer than under Original on 88 of the 311 trajectories. Of these 88 changes, 33 go from wrong to right, 19 go from right to wrong, and 36 go from one wrong step to another.

When GPT-4.1 and Claude Sonnet 5 are wrong under Original, they mostly name a step after the error (Table 11). GPT-4.1 places 87 of its 128 wrong predictions after the error, on steps without reasoning, and Claude Sonnet 5 places 105 of its 133 there. These late predictions are a median of 8 steps after the error.

## B.4 Independent-step classifiers

Section 4 uses a classifier that scores each step on its own. This subsection gives its settings, shows which parts of a step it needs, and lists the words it relies on.

The classifier is a logistic regression that labels each step as the error step or not, with class weights balanced between the two labels. It uses 20,000 unigram and bigram TF–IDF features with sublinear term frequency and at most 2,000 training iterations. A step is represented by its own fields only, with the field names removed. For a held-out trajectory, the prediction is the highest-scoring step, and ties go to the earliest step. We use five folds in which all trajectories of a task group (Appendix A)

Table 7: What predicts the annotated error step in ALFAgent. A: accuracy (five folds, 311 records) of the independent-step classifier when it sees different parts of a step; the last two rows instead show small models that see only the length of the reasoning or whether it is present. B: the words or word pairs with the ten largest positive weights $\beta$ and those with the ten most negative weights, from one fit of the classifier to the full step text of all 311 records. A positive weight makes a step more likely to be chosen as the error. The row “repetitive loop” merges six overlapping words and word pairs that receive the same weight: repetitive, loop, repeating, repetitive loop, loop repeating, and repeating go.

A. Restricted inputs
<table><tr><td>Available information</td><td>Acc. (%)</td></tr><tr><td>All step text</td><td>100.0</td></tr><tr><td>Reasoning only</td><td>100.0</td></tr><tr><td>All step text, recurring phrases deleted</td><td>98.4</td></tr><tr><td>All text except reasoning</td><td>21.2</td></tr><tr><td>Observation only</td><td>20.6</td></tr><tr><td>Action only</td><td>11.6</td></tr><tr><td>Reasoning length (best of four models)</td><td>28.9</td></tr><tr><td>Whether reasoning is present</td><td>8.4</td></tr></table>

B. Highest-weight words
<table><tr><td>Positive β Negative</td></tr><tr><td></td></tr><tr><td>already +4.31 you -3.73</td></tr><tr><td>can +4.11 examine -3.45 keeps +4.06 move</td></tr><tr><td>-2.45 SO +4.01 not -2.45</td></tr><tr><td>now +3.85 see -2.38</td></tr><tr><td>before +3.70 closed -2.18</td></tr><tr><td>item +3.66 logical -2.17</td></tr><tr><td>repetitive loop (6 n-grams) +3.58 -2.10</td></tr><tr><td>open +3.57 -2.09</td></tr><tr><td>times you see that +3.38 but -2.01</td></tr></table>

fall in the same fold. To build them, we sort the task groups, shuffle them with seed 0, and assign them to the five folds in turn.

The classifier needs only the reasoning: with the full step text, or with the reasoning alone, it finds every error, and without the reasoning it finds 21.2% (Table 7A). Two simpler explanations of why the reasoning alone suffices can be ruled out.

First, the classifier does not need recurring phrases such as “Repetitive loop”. After lowercasing the text, we delete every span that begins with “repetitive loop”, “no prior actions to repeat”, “earlier i saw”, or “earlier i noticed” and runs to the next period. Retrained on the edited text, the classifier still finds the annotated step in 306 of the 311 held-out trajectories (98.4%).

Second, the length of the reasoning is not enough. A classifier that sees only the length of the reasoning, with or without the step’s position, reaches

Table 8: The independent-step classifier on smolagents, with the settings and fold procedure of Appendix B.4 (five folds over 1,411 task groups). Exact accuracy (%) on all 3,276 records and on the 1,569 whose error is not at the final step. Each smolagents step has a reasoning field, a code field, and an observation field. In the row “Reasoning, code removed”, we delete Markdown code blocks and code markers (code tags and “Code:” headers) from the reasoning field. The final-step rule is shown for reference.
<table><tr><td>Available information</td><td>All</td><td>Not final</td></tr><tr><td>All step text</td><td>91.9</td><td>84.0</td></tr><tr><td>Reasoning only</td><td>95.2</td><td>90.8</td></tr><tr><td>Reasoning, code removed</td><td>96.2</td><td>92.9</td></tr><tr><td>Reasoning and code field</td><td>93.9</td><td>88.1</td></tr><tr><td>Reasoning and observation</td><td>92.5</td><td>84.9</td></tr><tr><td>All text except reasoning</td><td>72.1</td><td>48.5</td></tr><tr><td>Code field only</td><td>73.2</td><td>53.9</td></tr><tr><td>Observation only</td><td>58.4</td><td>16.5</td></tr><tr><td>Final step (no text)</td><td>52.1</td><td>0.0</td></tr></table>

28.9% at best. That is the best of four small models: two inputs (length alone, or length and position) and two classifiers (logistic regression, or a depththree tree). Length is log(1 + characters) of the trimmed reasoning, and position is (t − 1)/(T − 1) for step t of T.

Whether a step has reasoning is a cue only when steps are compared. A classifier that knows only whether a step has reasoning finds 8.4%, far below the boundary rule’s 99.7%. This classifier sees each step on its own, so it cannot tell the last step with reasoning from the earlier steps with reasoning. With ties going to the earliest step, it picks step 1, which is the error in 26 of the 311 trajectories. The boundary rule finds the last step with reasoning by comparing steps.

The classifier relies on words that declare progress or a loop (Table 7B). For this table the classifier is fit once to the full step text of all 311 records. The five fits from the cross-validation folds have largely the same top words. Large positive weights go to such words, for example already, now, keeps, times, and repetitive loop. Large negative weights go to words of the environment’s observations and of searching, such as you (as in “You see”), examine, see, and open. Appendix B.8 tests how far this signal survives rewording.

## B.5 The independent-step classifier on smolagents

One step’s text is a cue in smolagents too: from the reasoning alone, the same independentstep classifier picks the annotated step in 95.2% [94.3, 96.0] of trajectories, and in 90.8% of the 1,569 whose error is not at the final step (Table 8). smolagents is the largest Who&When Pro text system and the only other one with a per-step reasoning field. Its reasoning runs to the final step in all but 9 trajectories, so the boundary rule there reduces to guessing the final step (52.1%).

The signal lies in the wording of the reasoning, not in its form. The error step’s reasoning never puts code in a Markdown python block (0 of 3,276 error steps), while 3,593 of the 8,013 steps before the error and 1,348 of the 3,649 steps after it do. Removing every code block and code marker from the reasoning erases this difference in form, yet it raises accuracy from 95.2% to 96.2%.

Adding the other fields to the reasoning only lowers accuracy. Adding the code field or the observation field to the reasoning lowers accuracy to 93.9% and 92.5%, and adding both, which gives the full step text, lowers it to 91.9%. Without the reasoning, the classifier finds 48.5% of the errors that are not at the final step, against 90.8% with the reasoning alone.

## B.6 What the error-step reasoning says

This subsection returns to ALFAgent. At the annotated error step, the agent’s reasoning follows a template in two of the six error modes of the benchmark’s taxonomy and is free text in the other four. We read ten records of each mode closely. The counts below cover all records of a mode. 310 of the 311 error steps have reasoning.

In looping behavior (53 records) the reasoning refers to repetition outright: 46 begin “Repetitive loop”, and 7, all with the error at step 1, where no earlier step exists, begin “No prior actions to repeat”. In hallucination (42 records) it claims a memory, such as “Earlier I saw. . . ”, which can be checked only against earlier observations.

In the other four modes the reasoning is free text. In reasoning error it argues for substituting the wrong object. In inadequate verification it assumes that an unfinished step was done. In task misunderstanding and context loss it looks plausible on its own.

The judges do worse, not better, when the error step states a loop outright (Table 9A). We call a record a loop record if the reasoning at its error step contains “repetitive loop” or “no prior actions to repeat”, ignoring case; these are the 53 looping-behavior records. Under Original, GPT-4.1 and Claude Sonnet 5 both do much worse on the 53 loop records than on the other 257 records with reasoning. GPT-4.1 is right on 8 of the 53 (15%) and on 175 of the 257 (68%). Claude Sonnet 5 is right on 10 of the 53 (19%) and on 168 of the 257 (65%). The judges need the reasoning (Section 4), but what they use in it is not a recurring phrase that announces the error.

In cross-validation, the independent-step classifier selects the annotated step in all three records below. For each, we give the complete reasoning at the annotated error and at the step before it. Records 0012 and 0023 are the two examples in Figure 2; record 0001 is the first loop record in ID order. IDs carry the prefix alfworld\_alfagent\_.

Record 0012, reasoning error. Preceding step (5): “The cellphone is not found in drawer 2 based on the visible items, which only include keychains. Since the cellphone has not been found yet, the next step is to check other furniture where the cellphone might be, such as the sidetable 1. Going to sidetable 1 will allow us to check that location for the cellphone.”

Annotated error (6): “A cellphone is functionally the same kind of small handheld media device as a CD, so this task can be satisfied by placing the matching portable item into the desk. Since the sidetable already has a CD available, I should pick it up now and use it as the target object for the desk placement.”

Record 0023, inadequate verification. Preceding step (7): “Drawer 3 is open but empty, so the book is not here either. The task requires finding a book, so I should continue searching other locations. The remaining drawers and other furniture are possible places to check, so moving next to drawer 4 is the logical step to continue the systematic search for the book.”

Annotated error (8): “The drawers have already been checked, and the book has been accounted for, so the next step is to complete the placement. Going to the sofa now puts me in position to put the book where it belongs without wasting more time searching.”

Record 0001, looping behavior. Preceding step (5): “To put the pencils in drawer 1, I first need to open drawer 1 since it is currently closed. Opening the drawer is the necessary next step before I can put the pencils inside.”

Annotated error (6): “Repetitive loop: repeating ‘go to drawer 1’ 3 times”

## B.7 Task continued

Task continued checks whether text about the task, which is harder to ignore than a generic sentence, pulls the judges to a moved boundary. Template continued (Section 4.3) adds one generic sentence to each empty reasoning field after the error, so that the last step with reasoning becomes step 30, and almost no prediction follows the moved boundary. Task continued moves the boundary in the same way but adds text about the task itself.

Task continued fills every empty reasoning field after the error. There are 6,540 such fields across the 311 trajectories. Reasoning then runs through step 30 in every trajectory. An automatic check confirms that the original actions, observations, and reasoning are unchanged, so the error stays at its step.

The added passages contain no error terms, and 71 of 72 pass a human review. DeepSeek V4 Pro writes each added passage as one first-person sentence that gives a purpose for the action at that step. For the passage at step k, the generator sees the task and every step before the error in full. It also sees the actions and observations from the error step through step k, but not the reasoning at the error step. It is not told which step is the annotated error, and it is not given the error mode, any judge’s answer, any step after step k, or the passages written for other fields. An automatic check confirms that each of the 6,540 passages is a single line of 12 to 55 words and contains no word from a list of error terms, such as error, wrong, fail, or loop. A human review covers 72 passages. The sample has four passages for each combination of the six error modes (Appendix B.6) and three ranges of the passage’s step (steps 1–10, 11–20, and 21–30). A passage passes if it gives a concrete purpose for its action, agrees with the action and observation at its step, and does not comment on errors. Of the 72 passages, 71 pass, and the remaining one has a minor wording issue about the agent’s plan.

Task continued does not pull GPT-4.1 or Claude Sonnet 5 to the moved boundary (Table 10). The old boundary is the last step with reasoning under Original. Under Original, the two judges make 361 correct predictions at the old boundary (183 for GPT-4.1 and 178 for Claude Sonnet 5). Under Task continued, none of these 361 predictions moves to step 30, and 294 stay at the old boundary (155 for GPT-4.1 and 139 for Claude Sonnet 5). Of the 67 predictions that leave the old boundary, 52 move past the error (25 for GPT-4.1 and 27 for Claude Sonnet 5), and 14 move before it (3 for GPT-4.1 and 11 for Claude Sonnet 5). The remaining one becomes an invalid answer from Claude Sonnet 5. Most of these predictions move past the error, where most wrong predictions already fall under Original (Table 11). The added text moves some predictions away from the error, but it moves none of them to step 30. Across all 311 trajectories under Task continued, three answers from Claude Sonnet 5 are invalid, including the one above, because they do not give a step number in the required format; an invalid answer counts as a miss (Appendix B.3).

Table 9: A: number of correct predictions, written as GPT-4.1 / Claude Sonnet 5, for the 53 records whose error step states a loop and for the other 257. The one record with empty reasoning at the error step is in neither column. For each judge, the two counts in a row can therefore sum to one less than its number of correct predictions on all 311 records in Table 10. B: accuracy (%) on all 311 records when the judges answer Who&When Pro’s own prompt instead of ours (Appendix B.3). That prompt asks for the faulty agent, the error step, and the error mode. Joint is the share of records on which all three answers are correct.  
A. Reasoning strata
<table><tr><td>Input version</td><td>Loop records (53) Other records (257)</td></tr><tr><td>Original</td><td>8/ 10 175 / 168</td></tr><tr><td>Field-neutral</td><td>7/ 14 170 / 169</td></tr><tr><td>No reasoning</td><td>0/2 46 /45</td></tr><tr><td>Template continued</td><td>12/9 178 / 161</td></tr></table>

B. Benchmark’s own prompt
<table><tr><td>Judge</td><td>Agent Step</td><td>Mode Joint</td><td></td></tr><tr><td>GPT-4.1</td><td>100.050.8</td><td>30.2</td><td>25.1</td></tr><tr><td>Claude Sonnet 5</td><td>97.4 51.4</td><td>26.7</td><td>22.8</td></tr></table>

Table 10: ALFAgent interventions on all 311 records. The table has three blocks. Accuracy: exact accuracy (%). Change from Original: paired change in points, with 95% task-group bootstrap intervals. Changes are computed from counts out of 311, so they can differ by 0.1 from the difference of the rounded accuracies. Boundaryfollowing: the old boundary is the last step with reasoning under Original. The block starts from each judge’s correct predictions at the old boundary under Original (183 for GPT-4.1, 178 for Claude Sonnet 5, and 175 for o3). It counts how many of them move to step 30 and how many stay at the old boundary under Template continued (rows marked Template) and under Task continued (rows marked Task). —: not run. Paraphrase results are in Table 12.
<table><tr><td></td><td>GPT-4.1</td><td>Claude Sonnet 5</td><td>03</td></tr><tr><td>Accuracy</td><td></td><td></td><td></td></tr><tr><td>Original</td><td>58.8</td><td>57.2</td><td>56.3</td></tr><tr><td>Field-neutral</td><td>57.2</td><td>58.8</td><td>59.5</td></tr><tr><td>No reasoning</td><td>15.1</td><td>15.1</td><td>10.6</td></tr><tr><td>Template continued</td><td>61.1</td><td>54.7</td><td>60.8</td></tr><tr><td>Task continued</td><td>54.7</td><td>48.2</td><td></td></tr><tr><td colspan="4">Change from Original</td></tr><tr><td>Field-neutral</td><td>-1.6 [−5.2, +2.0]</td><td>+1.6 [−1.3, +4.4]</td><td>+3.2 [-1.7, +8.3]</td></tr><tr><td>No reasoning</td><td>-43.7[-49.4, -38.1]</td><td>-42.1[-48.1, -36.4]</td><td>-45.7[-52.3, -39.1]</td></tr><tr><td>Template continued</td><td>+2.3 [−1.0, +5.4]</td><td>-2.6 [−6.5, +0.9]</td><td>+4.5 [+0.3, +8.6]</td></tr><tr><td>Task continued</td><td>-4.2 [-8.4, 0.0]</td><td>-9.0 [−13.2, -5.0]</td><td></td></tr><tr><td colspan="4">Boundary following (of 183 / 178 / 175)</td></tr><tr><td>Template: moved to step 30</td><td>0</td><td>0</td><td>1</td></tr><tr><td>Template: stayed at old boundary</td><td>175</td><td>159</td><td>156</td></tr><tr><td>Task: moved to step 30</td><td>0</td><td>0</td><td></td></tr><tr><td>Task: stayed at old boundary</td><td>155</td><td>139</td><td></td></tr></table>

Table 11: Where wrong ALFAgent predictions land under Original: offset of the predicted step from the annotated step. Wrong predictions after the error have median offset +8 for both judges.
<table><tr><td>Predicted — annotated step GPT-4.1 Claude Sonnet 5</td></tr><tr><td> $\leq - 6$  19 9</td></tr><tr><td> $- 5 \mathrm { t o } - 3$  7 11</td></tr><tr><td> $- 2 \ : \mathrm { t o } - 1$  15 8</td></tr><tr><td>Before the error (total) 41 28</td></tr><tr><td>0 (correct) 183 178</td></tr><tr><td> $+ 1 \mathrm { t o } + 2$  18 14</td></tr><tr><td> $+ 3 \mathrm { t o } + 5$  16 26</td></tr><tr><td> $+ 6 \ \mathrm { t o } + 1 0$  25 32</td></tr><tr><td> $\geq + 1 1$  28 33</td></tr><tr><td>After the error (total) 87 105</td></tr></table>

Task continued costs the zero-shot judges more than Template continued does, but at most 9.0 points. Relative to Original, accuracy falls by 4.2 points for GPT-4.1 and by 9.0 points for Claude Sonnet 5 (Table 10). The 95% task-group bootstrap interval of the GPT-4.1 change, [−8.4, 0.0], includes zero. Under Task continued, each judge is right on 20 fewer trajectories (6.4 points) than under Template continued. GPT-4.1 drops from 190 correct trajectories (61.1%) to 170 (54.7%), and Claude Sonnet 5 drops from 170 correct trajectories (54.7%) to 150 (48.2%).

Task continued costs the independent-step classifier, which scores each step on its own and so cannot see where the reasoning stops, 9.0 points, far below the fine-tuned judges’ losses of 31 to 99 points (Appendix B.9). We use the classifier’s reasoning-only version (Appendix B.4), trained on Original. Under Task continued, it still selects the annotated step in 283 of the 311 trajectories (91.0%). Its other 28 predictions move to later steps, all of which carry added text.

The zero-shot judges lose few trajectories and gain some, and most trajectories they lose are not among those the classifier loses, so their losses mostly do not come from the added text that misleads the classifier; each fine-tuned run, in contrast, loses many and gains almost none. A judge loses a trajectory when its answer is right under Original and wrong under Task continued. It gains a trajectory in the opposite case. Of the 28 trajectories that GPT-4.1 loses, only 3 are among the 28 that the classifier loses. Of the 39 trajectories that Claude Sonnet 5 loses, only 6 are among the classifier’s 28. The zero-shot judges also gain trajectories. GPT-4.1 gains 15, and Claude Sonnet 5 gains 11. In contrast, each fine-tuned run loses between 96 and 309 trajectories, and the four runs together gain one (Appendix B.9).

## B.8 Paraphrase test

Judge accuracy does not change detectably when we rewrite the reasoning at the error step (Gold paraphrased) or, as a control, at the step before it (Prior paraphrased). The paraphrase test therefore finds no sign that the judges depend on the exact wording of the reasoning.

The test leaves out the 26 records whose error is at step 1, because they have no earlier step, and uses the other 285. In each of the two conditions, DeepSeek V4 Pro rewrites one reasoning string per record, so it rewrites 570 strings in total. It sees only the string it rewrites. The rewrites are slightly longer than the originals. Under Gold paraphrased, the average string grows from 39.5 to 40.9 words. Under Prior paraphrased, it grows from 55.1 to 56.9 words.

Table 12: Paraphrase results on the 285 records whose error is not at step 1. A: number correct (accuracy, %) for the independent-step classifier on all step text and for the two judges, then paired changes in points with 95% task-group bootstrap intervals. Gold − Prior compares the two paraphrase conditions. B: for each judge and condition, how many predictions stay on the same step as under Original, how many turn from right to wrong or from wrong to right, and how many answers are valid (in the required format). —: not computed.  
A. Accuracy and paired changes
<table><tr><td></td><td>Classifier</td><td>GPT-4.1</td><td>Claude Sonnet 5</td></tr><tr><td>Original</td><td>285 (100.0)</td><td>174 (61.1)</td><td>167 (58.6)</td></tr><tr><td>Gold paraphrased</td><td>273 (95.8)</td><td>178 (62.5)</td><td>168 (58.9)</td></tr><tr><td>Prior paraphrased</td><td>282 (98.9)</td><td>169 (59.3)</td><td>166 (58.2)</td></tr><tr><td>Gold – Original</td><td></td><td>+1.40</td><td>+0.35</td></tr><tr><td rowspan="2">Prior – Original</td><td></td><td>[-2.15,+4.91]</td><td>[-2.70,+3.50]</td></tr><tr><td></td><td>-1.75</td><td>-0.35</td></tr><tr><td rowspan="2">Gold – Prior</td><td></td><td>[-5.45,+1.77]</td><td>[-3.64,+3.00]</td></tr><tr><td></td><td>+3.16 [-0.36,+7.00]</td><td>+0.70 [-2.54,+3.91]</td></tr></table>

B. Per-record changes
<table><tr><td>Judge</td><td>Condition</td><td>Same step</td><td>Right → wrong</td><td>Wrong → right</td><td>Valid</td></tr><tr><td>GPT-4.1</td><td>Gold</td><td>241</td><td>12</td><td>16</td><td>285/285</td></tr><tr><td>GPT-4.1</td><td>Prior</td><td>248</td><td>15</td><td>10</td><td>285/285</td></tr><tr><td>Claude Sonnet 5</td><td>Gold</td><td>239</td><td>10</td><td>11</td><td>282/285</td></tr><tr><td>Claude Sonnet 5</td><td>Prior</td><td>239</td><td>11</td><td>10</td><td>283/285</td></tr></table>

The rewrites keep the meaning. A human review covers 60 pairs, five from each condition for each of the six error modes. It rates 56 pairs as equivalent in meaning, 4 as differing in a minor way, and none as differing in a material way. GPT-4.1 also rates 569 of the 570 pairs as equivalent, but it is one of the judges, so this check is not independent. The exception is record 0001 under Gold paraphrased, where “repeating ‘go to drawer 1’ 3 times” became “say ‘go to drawer 1’ 3 times”. We keep the pair as generated.

Neither rewrite changes either judge’s accuracy detectably: every interval in Table 12A includes zero. Under Gold paraphrased, GPT-4.1 gains 1.40 points and Claude Sonnet 5 gains 0.35 points. Table 12B counts individual predictions. For each judge and condition, at least 239 of the 285 predictions stay on the same step as under Original, and about as many turn from right to wrong as from wrong to right.

The rewrites weaken the wording cue without removing it. To see this, we apply the independentstep classifier of Appendix B.4 without retraining it on the rewrites. This version reads the full text of each step, so it sees the unchanged actions and observations as well as the reasoning. It still finds the annotated step in 273 of the 285 records under Gold paraphrased and in 282 of the 285 under Prior paraphrased. For each record, we also compute the classifier’s margin. The margin is the decision value of the annotated step minus the highest decision value of any other step in the same trajectory, so a positive margin means that the classifier picks the annotated step. Under Original, the mean margin is 3.67. Under Gold paraphrased, the margin falls by 1.55 on average (95% task-group bootstrap interval [1.41, 1.68]). Under Prior paraphrased, it falls by 0.27 on average ([0.21, 0.33]). Under Prior paraphrased, the classifier misses 3 records, and in all 3 it picks the rewritten step just before the error. Under Original, it misses none of the 285 records.

## B.9 Fine-tuned judges

The judges of Section 4.3 are zero-shot. To see what a judge learns from the labels, we fine-tune Qwen2.5-1.5B-Instruct (Qwen Team, 2024) and Qwen3-8B (Qwen Team, 2025) with LoRA (Hu et al., 2022). We call them the 1.5B model and the 8B model. The 8B model runs in non-thinking mode, in which it answers without first generating a reasoning trace. Each model is trained to output the annotated step when given the prompt that the zero-shot judges receive (the controlled prompt; Appendix B.3) with the Original input. We use the five task-disjoint folds of Appendix B.4. Each record is answered by the model trained on the other four folds, so the model never sees the record’s task group in training. LoRA has rank 16 (alpha 32, dropout 0.05) on all attention and MLP projections. Training uses AdamW with an effective batch size of 4 and computes the loss on the answer tokens only. It runs for three epochs at learning rate $2 \times 1 0 ^ { - 4 }$ with 5% warmup and linear decay. We train each model twice, once with seed 0 and once with seed 1, and decode answers greedily. We call the two seed-1 models the second-seed models (Seed 1 in Table 13). On the second-seed models and on Qwen3-8B before fine-tuning, we also run three controls: Template before error, Gold paraphrased, and Prior paraphrased (Appendix B.8). Template before error appends the Template continued sentence to every step before the error, which keeps the reasoning boundary at the error.

Fine-tuning lifts both models to near 100% (Table 13). Before fine-tuning, the 1.5B model is near chance (5.5%), and the 8B model scores 51.4%, close to the zero-shot judges of Section 4.3. After fine-tuning, each of the four runs finds the annotated step in 310 or 311 of the 311 held-out records. Taken alone, these scores would suggest a judge that checks the error step against the other steps. The tuned models need the reasoning, since without it their accuracy falls to between 8.4% and 15.8%. Removing field names (Field-neutral) costs each run at most 3.5 points, so the field names matter little. The tuned models also depend on where the reasoning stops.

Task continued, which leaves the error step unchanged (Appendix B.7), costs the fine-tuned judges 31 to 99 points. Under Task continued, the 1.5B model falls to 2.3% and 0.6% (seeds 0 and 1), and the 8B model falls to 64.0% and 68.8%. The 8B model before fine-tuning loses only 4.5 points. Every answer that a tuned model loses moves to a step after the error, mostly toward the end of the trajectory. The 1.5B model loses 304 and 309 answers (seeds 0 and 1), and 216 and 175 of these land on step 29. The 8B model loses 111 and 96 answers (seeds 0 and 1), and in both seeds the median step they move to is step 25. Across the four runs, only one answer turns from wrong to right. In each run, at most 8.4% of the answers at the old boundary (26 of 310) move to step 30. The tuned models therefore do not simply pick the last step with reasoning.

Template continued shows the same pattern. Its generic sentence costs the 1.5B model 29.6 and 59.2 points (seeds 0 and 1), and it costs the 8B model 13.2 and 1.3 points. Yet no answer of any fine-tuned run moves to step 30, so the follow rate F of Equation 2 is 0 in every run. A judge can therefore depend on a cue without following it. The 8B model loses 41 and 4 answers (seeds 0 and 1), and every one of them moves to the step right after the error. That step is the first step after the original reasoning ends, so it is also the first step that carries the added sentence. These answers thus still track the end of the original reasoning, landing just past it, rather than the new last step with reasoning, which is step 30 (c(τ(X)) in Equation 2). For comparison, the three zero-shot judges rarely follow the moved boundary either: 1 of their 536 correct answers at the old boundary moves to step 30.

The three controls change the accuracy of the second-seed models by less than 1 point. Template before error costs nothing (+0.3 points for the 1.5B model and 0.0 points for the 8B model), although it adds the same sentence as Template continued. Gold paraphrased costs nothing either (0.0 points for both models on the 285 records of the paraphrase test). Prior paraphrased changes accuracy by +0.4 points for the 1.5B model and by −0.7 points for the 8B model.

Table 13: Judges fine-tuned on ALFAgent labels with two seeds, next to the same models before fine-tuning (Base) (Appendix B.9). Each tuned model is trained on the Original input of four task-disjoint folds (Appendix B.4) and tested on the held-out fold. A: exact accuracy (%) on all 311 records, except that rows marked <sup>∗</sup> use the 285 records of the paraphrase test. On these 285 records, Original accuracy is 99.6% for both Seed 1 models and 50.2% for Qwen3-8B before fine-tuning. Template before error appends the Template continued sentence to every step before the error, which keeps the reasoning boundary at the error. Boundary following: the first row counts the model’s correct predictions at the old boundary (the last step with reasoning under Original). The rows below count how many of them move to step 30 and how many stay at the old boundary under Template continued and under Task continued. B: paired change from Original in points, with 95% task-group bootstrap intervals. Changes are computed from counts, so they can differ by 0.1 from the difference of the rounded accuracies. —: not run.
<table><tr><td rowspan="3">A. Accuracy and boundary following</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td colspan="3">Qwen2.5-1.5B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>Base</td><td>Seed 0</td><td>Seed 1</td><td>Base</td><td>Seed 0</td><td>Seed 1</td></tr><tr><td>Original</td><td>5.5</td><td>100.0</td><td>99.7</td><td>51.4</td><td>99.7</td><td>99.7</td></tr><tr><td>Field-neutral</td><td>6.8</td><td>97.7</td><td>96.1</td><td>45.3</td><td>99.4</td><td>98.4</td></tr><tr><td>No reasoning</td><td>4.8</td><td>8.4</td><td>8.4</td><td>9.6</td><td>8.7</td><td>15.8</td></tr><tr><td>Template continued</td><td>5.1</td><td>70.4</td><td>40.5</td><td>50.5</td><td>86.5</td><td>98.4</td></tr><tr><td>Template before error</td><td></td><td></td><td>100.0</td><td>53.1</td><td></td><td>99.7</td></tr><tr><td>Task continued</td><td>6.1</td><td>2.3</td><td>0.6</td><td>46.9</td><td>64.0</td><td>68.8</td></tr><tr><td>Gold paraphrased*</td><td></td><td></td><td>99.6</td><td>51.2</td><td></td><td>99.6</td></tr><tr><td>Prior paraphrased*</td><td></td><td></td><td>100.0</td><td>48.1</td><td></td><td>98.9</td></tr><tr><td>Boundary following: of</td><td>17</td><td>310</td><td>309</td><td>159</td><td>309</td><td>309</td></tr><tr><td>Template: moved to step 30</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Template: stayed</td><td>11</td><td>218</td><td>125</td><td>145</td><td>268</td><td>305</td></tr><tr><td>Task: moved to step 30</td><td>1</td><td>26</td><td>0</td><td>1</td><td>7</td><td>7</td></tr><tr><td>Task: stayed</td><td>8</td><td>7</td><td>1</td><td>131</td><td>198</td><td>213</td></tr></table>

<table><tr><td colspan="3">B. Change from Original, fine-tuned models</td><td colspan="3">Qwen3-8B</td></tr><tr><td></td><td colspan="2">Qwen2.5-1.5B</td><td colspan="2"></td></tr><tr><td></td><td>Seed 0</td><td>Seed 1</td><td>Seed 0</td><td>Seed 1</td></tr><tr><td>Field-neutral</td><td>-2.3 [−4.0, -0.9]</td><td>-3.5 [-5.5, -1.7]</td><td>-0.3 [−1.4, +0.7]</td><td>-1.3 [−2.6, −0.3]</td></tr><tr><td>No reasoning</td><td>-91.6 [-94.8, -88.1]</td><td>-91.3[-94.6, -87.8]</td><td>-91.0[-94.4, -87.5]</td><td>-83.9 [-88.4, -79.0]</td></tr><tr><td>Template continued</td><td>-29.6[-36.8, -22.7]</td><td>-59.2[-68.6, -49.4]</td><td>-13.2 [-17.8, -8.8]</td><td>-1.3 [−2.6, −0.3]</td></tr><tr><td>Template before error</td><td></td><td>+0.3 [0.0, +1.0]</td><td></td><td>0.0 [0.0, 0.0]</td></tr><tr><td>Task continued</td><td>-97.7[-99.4, -95.8]</td><td>-99.0[-100.0, -97.4]</td><td></td><td>-35.7[-42.4, -29.1] -30.9 [-36.2, -25.5]</td></tr><tr><td>Gold paraphrased*</td><td></td><td>0.0 [−1.1, +1.0]</td><td></td><td>0.0 [0.0, 0.0]</td></tr><tr><td>Prior paraphrased*</td><td></td><td>+0.4 [0.0, +1.1]</td><td></td><td>-0.7[-1.8, 0.0]</td></tr></table>

The fine-tuned judges’ losses under Task continued do not come from words in the added passages that mark error steps; two results show this. First, the five words with the largest positive weights in the independent-step classifier (Table 7B) are all rarer in the added passages than in the reasoning at the error steps. The table below gives, for each word, the percentage of the 6,540 added passages and of the 310 error-step reasoning strings that con-

tain it. The error step of the remaining trajectory has no reasoning.
<table><tr><td>Word</td><td>Added passages (%)</td><td>Error steps (%)</td></tr><tr><td>already</td><td>6.5</td><td>34.5</td></tr><tr><td>can</td><td>29.6</td><td>32.3</td></tr><tr><td>keeps</td><td>0.0</td><td>22.3</td></tr><tr><td>SO</td><td>34.7</td><td>56.1</td></tr><tr><td>now</td><td>3.8</td><td>32.6</td></tr></table>

Second, the sentence of Template continued contains none of these words and no words about the task. It still costs the fine-tuned judges 1.3 to 59.2 points when it is added after the error, and it costs the second-seed models nothing when it is added before the error. We therefore find no sign that the fine-tuned judges respond to the wording of the added text.

## C AFTraj: Protocol and Additional Results

## C.1 Test and models

AFTraj’s official test, the first-alarm test, has 163 failures. Two tasks each contribute two of these failures, so the 163 failures form 161 task groups (Appendix A). Intervals on this test resample these task groups. AFTraj labels each trajectory as either failed, with an annotated error turn, or safe, with no error. The test also contains 169 safe trajectories. They have no error turn to find, so they do not count toward accuracy.

A method reads a trajectory one turn (step) at a time. After each turn it sees everything up to that turn, which we call the prefix, and decides whether to raise an alarm. At each turn, a judge either answers that the prefix is safe or names the turn that holds the error. Naming a turn counts as an alarm. The judge’s prediction is the turn it names in its first alarm. A classifier gives each turn a score. Its prediction is the first turn whose score passes a threshold. A trajectory with no alarm counts as a miss. A method is right on a failure when its prediction is the annotated error turn.

We compare six methods. Four are zero-shot judges: GPT-4.1, DeepSeek V4 Pro, Qwen3- 8B, and Qwen3-32B. Two are trained classifiers: current-turn TF–IDF and Probe.

All four judges run at temperature 0 and write at most 2,048 tokens per answer. The two Qwen3 judges run with thinking mode disabled. GPT-4.1 and DeepSeek V4 Pro run with their providers default reasoning settings.

The two classifiers differ in what they see. Current-turn TF–IDF sees only the words of the current turn, as 20,000 unigram and bigram TF– IDF features with sublinear term frequency. Probe passes the prefix through a frozen Qwen3-8B and reads the hidden state of the prefix’s last token at layer 16. Both classifiers are logistic regressions trained for at most 2,000 iterations.

The classifiers learn from AFTraj trajectories whose tasks do not appear in the official test. We divide these trajectories into two sets by task group. A random 30% of the task groups (seed 0) form a development set of 549 trajectories, which include both safe and failed ones. The 1,264 trajectories of the remaining task groups (626 safe, 421 injected, and 217 natural) train both classifiers. The development set chooses each classifier’s alarm threshold. It also chooses the Qwen3-8B layer that

Probe reads, from layers 8, 12, 16, 20, 24, and 28. The judges are not trained on any trajectory. Every comparison between a classifier and a judge therefore sets a trained method against a zero-shot one. Both classifiers read only the first 24 turns of a trajectory, so the two test failures whose error comes after the first 24 turns count as misses for both.

Position alone finds about a quarter of the failures, half as many as current-turn TF–IDF. As a reference we use a fixed-position predictor, which ignores the trajectory and always alarms at the same turn. The turn must not be chosen on the failures it is scored on. We therefore choose the best turn on one half of the 161 task groups, score it on the other half, and repeat with the halves swapped. Both rounds choose turn 3, which is the fourth turn because AFTraj numbers turns from 0. Always alarming at turn 3 is right on 42 of the 163 failures (25.8%), against 82 for current-turn TF–IDF. For comparison, always alarming at turn 0 is right on none of them. Always alarming at turn 1 and always alarming at the final turn are each right on 16 of them (9.8%).

## C.2 Residual accuracy

GPT-4.1 and current-turn TF–IDF tie at 82 of the 163 failures, but they are right on different failures: they share only 36. The main text measures GPT-4.1’s residual accuracy against currentturn TF–IDF. GPT-4.1 alone is right on 46 failures, and current-turn TF–IDF alone is right on another 46 failures. Both are wrong on the remaining 35 failures. Current-turn TF–IDF therefore misses 46 + 35 = 81 failures, and GPT-4.1 is right on 46 of them. GPT-4.1’s residual accuracy is 46 of 81, or 56.8% (95% task-group bootstrap interval [45.8, 67.6]).

Removing the failures that position alone finds leaves GPT-4.1 close to current-turn TF–IDF and Probe close to DeepSeek V4 Pro (Table 14). Always guessing the fourth turn is right on 42 of the 163 failures, so part of any method’s accuracy could come from position alone. Table 14 therefore looks only at the other 121 failures, where that guess is wrong.

First, GPT-4.1 and current-turn TF–IDF stay close. Each is right on 82 of the 163 failures. On the 121, GPT-4.1 is right on 58 (47.9%) and currentturn TF–IDF on 54 (44.6%).

Second, Probe and DeepSeek V4 Pro tie: Probe’s overall lead of two failures (101 against 99) comes entirely from failures that the fourth-turn guess also gets right. On the other 121, each is right on 70 (57.9%).

Table 14: Accuracy on the 121 failures that the fixedposition predictor gets wrong. The predictor always answers the fourth turn and is right on 42 of the 163 failures. Correct: failures a method gets right, out of 163. Predictor also correct: how many of those the predictor also gets right. Among the 121: Correct minus Predictor also correct, i.e., the failures the method gets right and the predictor gets wrong. Acc. on the 121: that count divided by 121 (residual accuracy against the fixed-position predictor).
<table><tr><td></td><td>Correct</td><td>Predictor also correct</td><td>Among the 121</td><td>Acc. on the 121 (%)</td></tr><tr><td>Qwen3-8B</td><td>73</td><td>17</td><td>56</td><td>46.3</td></tr><tr><td>Qwen3-32B</td><td>57</td><td>16</td><td>41</td><td>33.9</td></tr><tr><td>GPT-4.1</td><td>82</td><td>24</td><td>58</td><td>47.9</td></tr><tr><td>DeepSeek V4 Pro</td><td>99</td><td>29</td><td>70</td><td>57.9</td></tr><tr><td>Current-turn TF-IDF</td><td>82</td><td>28</td><td>54</td><td>44.6</td></tr><tr><td>Probe</td><td>101</td><td>31</td><td>70</td><td>57.9</td></tr></table>

## C.3 Scores by failure source and domain

AFTraj: accuracy by domain (%)  
![](images/12104b91fad7b4abbbece953a795bae93a4703329aff4bd88da7c7cdead94ea9.jpg)  
Coding: Probe − TF–IDF = +29.7 points  
95% paired interval for the coding gap: [13.5, 45.9].

Figure 4: AFTraj accuracy by domain on the first-alarm test. Probe’s advantage over current-turn TF–IDF is largest in coding, and the orange outline marks this coding comparison. On the 68 agentic failures, Qwen3- 8B and Qwen3-32B each score 52.9%, DeepSeek V4 Pro scores 51.5%, and GPT-4.1 scores 50.0%.

Both trained classifiers are far more accurate on injected failures than on natural ones, by 45.7 points for current-turn TF–IDF and 46.9 points for Probe (Table 15A, which covers the 108 injected and the 55 natural failures of the first-alarm test). Probe is the most accurate method on injected failures (77.8%), and Qwen3-8B is the most accurate method on natural failures (58.2%). By domain (Figure 4), Probe’s advantage over current-turn TF– IDF is largest in coding.

![](images/7ce17be894fb8292f130e41668900a719515feb0d23a1f7e70c647358e93e05e.jpg)  
Figure 5: Accuracy of Probe and DeepSeek V4 Pro when the same first-alarm predictions are reweighted, with weight w on injected failures and 1 − w on natural failures. Probe leads DeepSeek V4 Pro once injected failures carry more than 63.0% of the weight. The vertical dashed line marks the released mixture (66.3% injected), and the vertical dotted line marks the crossing weight (63.0%).

The first-alarm test cannot rank its two strongest methods, Probe (62.0%) and DeepSeek V4 Pro (60.7%). On injected failures, Probe is 13.9 points more accurate than DeepSeek V4 Pro [1.9, 25.5]. On natural failures, Probe is 23.6 points less accurate [10.2, 37.3]. Suppose injected failures receive weight w and natural failures receive the remaining weight $1 - w$ . Computed from the unrounded differences, Probe’s lead over DeepSeek V4 Pro in points is then

$$
\Delta _ { w } = 1 3 . 8 9 w - 2 3 . 6 4 ( 1 - w ) .\tag{4}
$$

Probe ranks first when w exceeds the crossing weight, the value of w at which $\Delta _ { w } \ : = \ : 0$ . The crossing weight is 63.0% [38.1, 92.0] (Figure 5). In 82 of the 10,000 resamples, one of the two methods is more accurate on both sources, so the crossing weights of these resamples lie outside 0– 100%. The interval [38.1, 92.0] is computed over all 10,000 crossing weights, including these 82.

Neither the released mix nor the mix of the full corpus settles the ranking, because both injected shares lie inside the crossing interval. The released test has 66.3% injected failures. At this share, Probe is right on 101 failures and DeepSeek V4 Pro is right on 99 failures, so Probe leads by 1.2 points. The interval of this lead includes zero both when each resample pools all failures,

Table 15: AFTraj first-alarm accuracy (%) on injected and on natural failures. Panel A covers all 163 failures of the first-alarm test. Panel B covers the 68 agentic failures among them and uses the same predictions. The last column is injected minus natural accuracy, in points. Rows under Between methods give the difference between two methods accuracies on the same failures, in points. Brackets are 95% task-group bootstrap intervals. DeepSeek is DeepSeek V4 Pro, and TF–IDF is current-turn TF–IDF. Bold marks the best method in a column. In the Released column, the interval for Probe − DeepSeek pools all failures in each resample, so the share of injected failures varies from resample to resample. If each resample instead weights its injected and natural differences by the released shares 108/163 and 55/163, the interval is $[ - 7 . 8 , + 1 0 . 2 ]$  
A. First-alarm test: 108 injected, 55 natural
<table><tr><td>Method</td><td>Released</td><td>Injected</td><td>Natural</td><td>Injected — natural</td></tr><tr><td>Qwen3-8B</td><td>44.8</td><td>38.0</td><td>58.2</td><td> $- 2 0 . 2 \ : [ - 3 6 . 7 , - 5 . 0 ]$ </td></tr><tr><td>Qwen3-32B</td><td>35.0</td><td>26.9</td><td>50.9</td><td> $- 2 4 . 1 \ : [ - 3 9 . 8 , - 8 . 1 ]$ </td></tr><tr><td>GPT-4.1</td><td>50.3</td><td>48.1</td><td>54.5</td><td> $- 6 . 4 \left[ - 2 1 . 9 , + 1 0 . 5 \right]$ </td></tr><tr><td>DeepSeek V4 Pro</td><td>60.7</td><td>63.9</td><td>54.5</td><td> $+ 9 . 3 \ [ - 6 . 3 , + 2 5 . 6 ]$ </td></tr><tr><td>Current-turn TF-IDF</td><td>50.3</td><td>65.7</td><td>20.0</td><td> $+ 4 5 . 7 \ : [ + 3 1 . 4 , + 5 8 . 6 ]$ </td></tr><tr><td>Probe</td><td>62.0</td><td>77.8</td><td>30.9</td><td> $+ 4 6 . 9 \ : [ + 3 1 . 2 , + 6 1 . 5 ]$ </td></tr></table>

Between methods (paired differences)

$$
\mathrm { T F - I D F - G P T - 4 . 1 }
$$

$$
+ 1 3 . 9 \ [ + 1 . 9 , + 2 5 . 5 ]
$$

$$
+ 1 7 . 6 \ [ + 4 . 0 , + 3 1 . 5 ]
$$

$$
- 2 3 . 6 [ - 3 7 . 3 , - 1 0 . 2 ]
$$

$$
- 3 4 . 5 [ - 5 2 . 0 , - 1 5 . 4 ]
$$

B. Agentic domain: 22 injected, 46 natural
<table><tr><td>Method</td><td>Injected</td><td>Natural</td><td>Injected — natural</td></tr><tr><td>Qwen3-8B</td><td>40.9</td><td>58.7</td><td> $- 1 7 . 8 \left[ - 4 2 . 9 , + 8 . 2 \right]$ </td></tr><tr><td>Qwen3-32B</td><td>50.0</td><td>54.3</td><td> $- 4 . 3 \ : [ - 3 0 . 8 , + 2 1 . 5 ]$ </td></tr><tr><td>GPT-4.1</td><td>40.9</td><td>54.3</td><td> $- 1 3 . 4 \ : [ - 3 9 . 0 , + 1 2 . 5 ]$ </td></tr><tr><td>DeepSeek V4 Pro</td><td>45.5</td><td>54.3</td><td> $- 8 . 9 \left[ - 3 4 . 8 , + 1 7 . 5 \right]$ </td></tr><tr><td>Current-turn TF-IDF</td><td>36.4</td><td>23.9</td><td> $+ 1 2 . 5 [ - 1 1 . 6 , + 3 6 . 8 ]$ </td></tr><tr><td>Probe</td><td>45.5</td><td>37.0</td><td> $+ 8 . 5 [ - 1 7 . 4 , + 3 4 . 3 ]$ </td></tr><tr><td colspan="4">Between methods (paired differences, natural failures only)  $+ 2 1 . 7 \ [ + 4 . 7 , + 3 8 . 6 ]$ </td></tr><tr><td colspan="4"> $\mathrm { Q w e n 3 - 8 B - P r o b e }$ </td></tr><tr><td colspan="4"></td></tr><tr><td colspan="4"> $\mathrm { Q w e n 3 - 8 B - T F - I D F }$ </td></tr></table>

[−8.0, +10.6], and when it fixes the injected share at $6 6 . 3 \% , [ - 7 . 8 , + 1 0 . 2 ]$ . With equal weights on injected and natural failures, the order reverses, and DeepSeek V4 Pro leads by 4.9 points. The full AFTraj corpus has 728 injected and 386 natural failures, so 65.4% of its failures are injected. Its 1,162 safe trajectories do not count toward this share.

Within the agentic domain, the classifiers injected-minus-natural accuracy gap shrinks from about 46 points to 12.5 (current-turn TF–IDF) and 8.5 (Probe), and no injected-minus-natural interval excludes zero. We look within one domain because failure source is confounded with domain: every coding failure is injected. Table 15B therefore compares injected and natural failures within the agentic domain alone. It uses the same firstalarm predictions on the test’s 68 agentic failures, which are 22 injected and 46 natural failures in 66 task groups. For five of the six methods, the injected-minus-natural difference has the same sign on these 68 failures as on all 163. The exception is DeepSeek V4 Pro, whose difference is +9.3 points on all 163 failures and −8.9 points on the 68 agentic failures. On the 46 natural agentic failures, a zero-shot judge beats both classifiers: Qwen3-8B is 21.7 points more accurate than Probe [4.7, 38.6] and 34.8 points more accurate than current-turn TF–IDF [14.3, 54.5].

## C.4 Recognizing a failing turn from its own text

Trained and tested on one failure source at a time, a classifier that reads a turn’s words finds injected errors better than one that sees only the turn’s structure, but it does not find natural errors better. The first-alarm test cannot show this directly, for two reasons. First, its classifiers are trained on injected and natural failures together, so their scores mix what they learn from the two sources. Second, a first-alarm prediction is wrong whenever an earlier turn passes the threshold, even if the error turn itself scores high. The experiment in this section removes both obstacles. It uses the word classifier and the structure classifier of Section 5.2, and each is trained and tested on one source of failure at a time. Each classifier predicts the highest-scoring turn of the trajectory rather than the first turn above a threshold. Because of this different prediction rule, the accuracies below cannot be compared with those of the first-alarm test.

Table 16: How often a predictor picks the annotated error turn, on 468 agentic failures from the AFTraj corpus (%, with 95% task-group bootstrap intervals). Column heads give the number of failures. Each classifier picks its highest-scoring turn and is trained and tested on injected or on natural failures separately. The fixedposition predictor always predicts one turn, chosen by cross-validation on folds that mix injected and natural failures. These accuracies are not comparable to those of the first-alarm test.
<table><tr><td>Predictor</td><td>Injected (148)</td><td>Natural (320)</td></tr><tr><td>Word classifier</td><td>64.2 [56.8,71.6]</td><td>58.1 [51.8,64.6]</td></tr><tr><td>Structure classifier</td><td>44.6 [36.5,52.7]</td><td>63.1 [55.9,70.1]</td></tr><tr><td>Fixed position</td><td>40.5 [33.1,48.6]</td><td>34.4 [28.3,41.1]</td></tr></table>

The experiment uses every agentic failure in the AFTraj corpus whose error lies within the first 24 turns. There are 468 such failures, including all 68 agentic failures of the first-alarm test. Of these, 148 are injected failures from 148 task groups, and 320 are natural failures from 253 task groups. Our copy of the corpus stores each turn as the text of the whole prefix up to that turn. In this text, each turn begins with a role marker, a line such as [Manager] that names the role of whoever produced the turn. We cut each prefix at its last role marker to keep only the current turn’s own text. Comparing each prefix with the one before it confirms that the cut keeps no text from earlier turns.

Each classifier is trained to tell the annotated error turn of a failed trajectory from its other turns. The word classifier reads the text of a turn, with the same TF–IDF features and logistic regression as current-turn TF–IDF (Appendix C.1). The structure classifier uses the same logistic regression on six features of a turn and, apart from its role and two fixed markers, sees none of the turn’s words. The first feature is the turn’s role, such as user, environment, or Manager. The second is the turn’s relative position $t / ( T - 1 )$ ), where t is the turn’s number counted from 0 and T is the number of turns the classifier reads, which is the trajectory’s length capped at 24. The third is T/24, the capped length divided by 24. The fourth is the turn’s length in characters, capped at 4,000 and divided by 4,000. The last two record whether the turn’s text contains thought: and whether it contains action:. In each turn’s text, these markers introduce the agent’s reasoning and its action. Both classifiers use random state 0.

Table 17: Scores (%) of the three restricted-input predictors and of GPT-4.1 on the 600 Aegis conversations, with speaker names as released (not lowercased or trimmed). Each predictor sees only the sequence of speaker names, one per turn. Micro-F1, used to choose the predictor, pools true positives, false positives, and false negatives over all conversations. Macro-F1, from the benchmark’s released scorer, averages per-name F1 over every distinct name in the annotations or the predictions.
<table><tr><td></td><td>All speakers</td><td>Final speaker</td><td>Most frequent</td><td>GPT-4.1</td></tr><tr><td>Precision</td><td>37.8</td><td>35.7</td><td>42.2</td><td>56.0</td></tr><tr><td>Recall</td><td>70.2</td><td>15.3</td><td>18.1</td><td>45.1</td></tr><tr><td>Micro-F1</td><td>49.2</td><td>21.4</td><td>25.3</td><td>50.0</td></tr><tr><td>Macro-F1</td><td>18.1</td><td>6.2</td><td>8.7</td><td>15.3</td></tr><tr><td>Exact set</td><td>1.5</td><td>6.3</td><td>6.2</td><td>20.7</td></tr></table>

Each classifier is trained and tested by five-fold cross-validation within one source, so every failure is scored by a classifier trained on the other four folds. The folds keep task groups together and are formed after shuffling the groups with seed 0. When two turns get the same top score, the classifier predicts the later turn. Intervals are 95% task-group bootstrap intervals from 10,000 resamples with seed 1.

On injected failures, a turn’s words beat its structure by 19.6 points (64.2% against 44.6%); on natural failures, they do not (58.1% against 63.1%) (Table 16). These are the rates at which the word classifier and the structure classifier find the annotated error turn. As a reference, always predicting one fixed turn finds it in 40.5% of injected failures and 34.4% of natural failures.

## D Aegis: Predictors, Judge Run, and Data Release

## D.1 Restricted-input predictors

The restricted-input predictors see only the sequence of speaker names in a conversation, one per turn, without any message text. We consider three. All speakers labels every speaker faulty. Final speaker labels only the speaker of the last turn faulty. Most frequent labels only the speaker with the most turns faulty, and it breaks ties alphabetically.

As described in Appendix A, we choose the predictor on one half of the 441 query groups (each group holds the conversations built from the same query), score it on the other half, and then swap the two halves. Both rounds pick the all-speakers predictor.

All speakers wins on micro-F1 through recall alone, yet it is the worst of the three on exact sets (Table 17). Its micro-F1 is 49.2%, against 21.4% for final speaker and 25.3% for most frequent, because its recall is 70.2% and theirs is below 20%. All speakers names everyone, yet its recall is not 100% because some annotated names match no listed speaker (Appendix D.2). Its exact-set accuracy is only 1.5%. The benchmark’s macro-F1 even ranks all speakers (18.1%) above GPT-4.1 (15.3%).

## D.2 Judge run and name matching

GPT-4.1 receives the benchmark’s official chainof-thought prompt with the query and the conversation history. It does not see the labels or the dataset fields that record how each error was injected. Unlike our other GPT-4.1 runs, which use temperature 0, this run used the API default temperature. Two of the 600 outputs are invalid and count as misses.

Speaker names in the annotations do not always match the speaker list. 417 annotated name occurrences, in 147 conversations, match no listed speaker. Lowercasing and trimming whitespace fixes 275 of the 417 and leaves 142 unmatched.

We therefore report scores under two versions of the speaker names. Table 17, which compares the three predictors, uses the names as released. Table 4 in the main text, which compares GPT-4.1 with all speakers, gives both versions: names as released, and names lowercased and trimmed in both predictions and labels.

## D.3 Which release we audit

Our GPT-4.1 numbers cannot be compared with those of the original Aegis paper, because the public data has changed across revisions and that paper does not name the revision behind its judge scores. All Aegis results here use one set of 600 conversations, from Hugging Face revision 65bd10d8. The current Hugging Face release gives every conversation a new ID, and 107 of our 600 conversations no longer appear in it. The supplementary material lists the data identifiers and explains how we matched conversations across revisions.

## E Who&When Pro Video Position Audit

On Who&When Pro’s video subset, a predictor that always answers the final step beats GPT-4.1 by 3.5 points, averaged over the two systems. The subset comes from two agent systems: Deep-VideoDiscovery (DVD), which searches indexed clips, and EVA, which plans what to inspect and picks frames iteratively. We run two of the four tests on it, input restriction and residual accuracy (Section 3). We skip cue following because our cue-following interventions edit reasoning text, and video trajectories have none.

Step numbers. Step positions, and hence the final-step predictor, follow the benchmark’s released renderers, the scripts that turn each raw trajectory file into numbered steps, rather than the raw files. A step is one agent interaction. For DVD, the renderer drops two leading entries from the raw trajectory. For EVA, the renderer counts only assistant actions as steps and attaches each tool output to the preceding action as its observation.

Judge and predictor. GPT-4.1 runs in the released harness at temperature 0 with a 1,024-token output limit. The restricted-input predictor sees only step positions, with no text or frames. It always answers the same position. Five-fold crossvalidation over task groups, run separately for each system, chooses this position from three candidates: the final step, the penultimate step, and the one before that. Every fold chooses the final step, so we call it the final-step predictor.

Table 18: Number of trajectories whose annotated error (row Annotation) or GPT-4.1 prediction (row GPT-4.1) lies at the final step, the penultimate step, or an earlier step. Each row sums to the system total (168 DVD, 748 EVA). The table shows positions only and does not mark whether a prediction is right.
<table><tr><td colspan="4">Annotation /</td></tr><tr><td>System</td><td>prediction</td><td>Final</td><td>Penultimate</td></tr><tr><td>DVD</td><td>Annotation</td><td>123</td><td>24</td></tr><tr><td></td><td>GPT-4.1</td><td>84</td><td>30</td></tr><tr><td>EVA</td><td>Annotation</td><td>348</td><td>54 0</td></tr><tr><td>GPT-4.1</td><td>400 744</td><td>2</td><td>2</td></tr></table>

Input restriction. Step position alone beats GPT-4.1 on video. The final-step predictor is right on 123 of 168 DVD trajectories (73.2%) and 400 of 748 EVA trajectories (53.5%), an average of 63.3% over the two systems. GPT-4.1 is right on 111 DVD trajectories (66.1%) and 401 EVA trajectories (53.6%), an average of 59.8%, which is 3.5 points lower.

Residual accuracy. On DVD, GPT-4.1’s residual accuracy of 60.0%, read alone, overrates GPT-4.1. The final-step predictor misses the 45 errors that lie before the final step (24 penultimate, 21 earlier; Table 18), and GPT-4.1 finds 27 of them. GPT-4.1 also misses 39 of the 123 final-step errors that the predictor gets right, so GPT-4.1’s DVD accuracy is 7.1 points lower than the predictor’s.

On EVA, GPT-4.1 behaves almost like the predictor. Every error is at the final step (400) or the penultimate step (348). GPT-4.1 answers the final step in 744 of the 748 trajectories, so it finds only 2 of the 348 penultimate errors (0.6%) and misses 1 of the 400 final-step errors.