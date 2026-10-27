# Not One Belief: Answer Payloads, Persistence, and Dissociated Confidence Signals in Conversational Question Answering

Duc-Vinh Tran<sup>1\*</sup>

<sup>1\*</sup>School of Information and Communication Technology, Hanoi University of Science and Technology, Hanoi, 100000, Vietnam.

Corresponding author(s). E-mail(s): vinh.td225538@sis.hust.edu.vn;

## Abstract

Conversational question-answering systems are expected to fix their errors when users push back, and to signal which answers to trust. In short-answer question answering, we split each user follow-up into its speech act (re-ask, agree, or say the answer is wrong) and its payload (an answer string, correct or not). On the same transcripts we measure four confidence signals: verbalized confidence, lengthnormalized sequence likelihood, a hidden-state linear probe fitted at the first turn and then frozen, and the self-evaluation probability P(True). Answers change when the user supplies an answer string. They almost never change under the speech act alone: in every retained full-size model–dataset cell, the correct answer converts at least four times as many wrong answers as re-asking or agreement. The four signals disagree on the same turns. On Qwen2.5-7B the frozen probe rises both on answers that stay wrong and on answers that switch to the correct one, so it goes with answer persistence rather than truth. This survives scoring on held-out questions, and Llama-3-8B with cleaned answers supports it; on four other models it does not hold up on held-out questions. For 81 Qwen2.5-7B errors that survive one re-ask, re-asking, a fresh chat, or an abstain option repairs almost none; the correct answer repairs 75–95%. In two preregistered replications on four models, every observed repair rate without an answer string stays below a preset 0.15, and a post hoc rerun with about 300 such items per model excludes 0.15 for every arm except the empty elicit. With one BM25-retrieved TriviaQA sentence, a sentence that contains the answer repairs 67–76% of persistent errors, and one that lacks it repairs at most 12% (16% under model-read labels). In a post hoc end-to-end test with BM25 passages, gating evidence on whether it contains the gold answer cut harm on TriviaQA, but a gate that does not know the answer repaired few errors, so the check remains a heuristic.

Keywords: Large language models, Confidence calibration, Multi-turn dialogue, Sycophancy, Probing, Retrieval-augmented generation

## 1 Introduction

Chat assistants built on large language models (LLMs) now answer questions in customer support, tutoring, search, and clinical or legal help lines. There, a wrong answer costs something. Two questions follow, both about the reliability of conversational QA and evidence retrieval under user feedback. When a user pushes back, does the system fix its errors or just change its mind? And can the system’s own confidence tell the operator which answers to trust? Recent work on the second question asks whether LLM confidence stays calibrated as a conversation goes on [1, 2]. These studies usually score one confidence measure (verbalized confidence, the self-evaluation probability P(True), or sequence likelihood) against correctness, as if each were an estimate of the same underlying belief.

For that to work, the measures must move together, and every follow-up must count as the same kind of evidence. In our data, neither holds. So we ask a simpler question: when a user sends a follow-up message, which quantity does it change? Figure 1 summarizes the answer. We split each follow-up into two parts. The speech act is what the user does: re-ask, agree, or say the model is wrong. The payload is an optional answer string in the message, either the correct (“gold”) answer or a plausible wrong one (a “foil”). Payloads change the model’s answer. Speech acts alone almost never repair an error. Changes to the answer, whether or not the new answer is true, move a frozen probe readout that, on Qwen2.5-7B, goes with answer persistence. We call it the persistence readout s<sub>t</sub> and make no psychological or causal claim about it. The other measures respond to other things: verbalized confidence to the number of turns, sequence likelihood to how fluently the model repeats itself, and P(True) to explicit contradiction. Two papers that calibrate diferent measures and call both “confidence” are not reporting comparable numbers.

Recent single-turn work already shows that confidence measures can come apart. Kumaran [3] finds that verbal confidence predicts a model’s later decision to commit to or withhold an answer better than it predicts correctness, while calibrated token log-probabilities track correctness. Zhang et al. [1] study multi-turn confidence as hints accumulate and propose InfoECE and P(Suficient), but they do not separate what the user does from the answer string the user gives. Kumaran [3] work in a single turn. Our contribution is to control the user’s turn in multi-turn dialogue, separate the speech act from the answer payload, and measure four signals on the same transcripts.

## What moves the answer.

We write I→C for an item that is incorrect at turn 1 and correct at turn 2, and C→I for the reverse. Re-asking (persist) and agreeing with the model (confirm) produce almost no I→C changes, so the error survives. Stating the gold string produces many, with or without “you are wrong” around it. Stating a plausible foil produces many

![](images/f8dde8ca5960c2f5924601eae9ac1b528cd279f10a0ec1782cb6080a28e48ea1.jpg)  
Fig. 1 Which part of the second user turn (T2) changes which quantity. Solid lines show what a controlled T2 message changes; dashed lines show which confidence measure attaches to it. Green boxes form the payload chain, orange boxes the speech-act chain, and blue boxes are the confidence measures. The persistence readout s is defined in Section 3.4

C→I changes. An empty challenge (“I think that’s wrong”) makes answers unstable without pointing them anywhere. The string moves the answer. The stance does not. Models difer only in how they take up the string. Qwen2.5 and Llama-3-8B usually copy it verbatim (the fraction of I→C answers that copy it is 0.70–0.91 on NQ and HotpotQA). Mistral-7B-v0.3 restates it inside a longer sentence.

## The measures disagree.

On Qwen2.5-7B-Instruct, verbalized confidence falls as the user keeps talking, even when accuracy is flat (persist) or rising (correct). Length-normalized sequence likelihood often rises on the same turns. A linear probe on hidden states, fitted once on first-turn answers and then frozen, rises both on items that stay wrong and on items that switch to gold. Kadavath-style P(True) [4] barely moves, except under foil persuasion, where it collapses on C→I changes (mean change ≈ −0.26 overall and −0.85 on C→I items). One conversation, four diferent stories. The same holds for calibrated scores: a single user turn sends the expected calibration error (ECE) down for verbalized confidence, up for likelihood, and nowhere for P(True) (Section 5).

## Persistent errors are hard to undo.

Asking again or starting a new chat does not fix a persistent error unless the user supplies a string. Among 81 Qwen2.5-7B TriviaQA items that are still wrong after one re-ask, re-asking through turn 5, a fresh chat, and an ABSTAIN option repair two, zero, and zero items. The model abstains on 94% of these stay-wrong items and on 62% of stay-correct items. A gold string at turn 3 repairs 0.75–0.95 of the same items, depending on phrasing, and an “I don’t know” turn repairs 5/81. Two preregistered replications (230 stay-wrong items from a second Qwen slice, 160 from Llama-3-8B, 139 from Mistral-7B-v0.3, and 61 from Qwen2.5-32B) keep every observed repair rate without an answer string below a 0.15 threshold fixed in advance, under the preregistered answer matcher. A post hoc rerun on about 300 new stay-wrong items per model excludes 0.15 for every such arm except the empty elicit (Section 6.1). With one BM25 sentence per TriviaQA question, retrieval repairs errors only to the extent that the sentence contains the answer string (Section 6.1).

## Contributions.

We ofer a map of interventions and their efects, not a new method:

1. The answer string, not the speech act, moves the answer. Answers follow a stated gold or foil string. Re-asking and agreement do not repair errors (Section 4).

2. The measures come apart across turns. Single-turn work has shown that verbalized confidence and correctness-tracking signals are not interchangeable [3, 5]. We show the same split in dialogue for four measures (verbalized confidence, likelihood, a frozen hidden-state probe, and P(True)), in raw scores and in calibration error, and we show which part of the user turn each one responds to (Section 5).

3. On Qwen2.5-7B the frozen probe goes with answer persistence, and persistent errors are hard to undo. The persistence readout $s _ { t } = \sigma ( v _ { \mathrm { a c c } } ^ { \top } \tilde { h } _ { t } )$ is a frozen probe on the current answer state (Section 3.4). On Qwen2.5-7B it rises on stay-wrong items and on I→C items, also on held-out questions and above a shufled-label control. Llama-3-8B with cut answers supports this; on the other four models it does not hold on held-out questions. Resetting the chat or ofering abstention does not recover these errors; a gold string at turn 3 does (Section 6.1).

4. A retrieval heuristic. Before relying on evidence to repair an answer, check that it contains the answer string. This is a hypothesis for system design, not a validated rule. Across four models, a retrieved sentence without it repairs at most 0.16 of stay-wrong items under any of our scorers, and it breaks up to 0.70 of correct answers (Section 6.1).

The first three points share one argument: if two measures respond to diferent controlled interventions and disagree on the same transcript, they cannot both estimate a single latent conversational confidence. We do not propose a new uncertainty estimator. Section 8 lists what we do not claim.

## 2 Related work

## Sycophancy.

Instruction-tuned LLMs tend to adopt a view the user states, even when it is wrong. Perez et al. [6] documented this with model-written evaluations and found that it grows with model size and with reinforcement learning from human feedback. Sharma et al. [7] showed that sycophancy is common across production assistants and traced it partly to human preference judgments that reward agreement. Wei et al. [8] found that scaling and instruction tuning increase sycophancy, that models agree with a user’s clearly false arithmetic claims, and that fine-tuning on simple synthetic data reduces it. These works show that a user’s stated answer is persuasive. They usually bundle the stance (“I think you are wrong”) with the stated string. Our factorial design separates the two. For short factual answers, the string does almost all of the work.

## Re-asking and answer flipping under pushback.

Other work studies what happens when a user challenges an answer, with or without ofering a new one. In the FlipFlop experiment, Laban et al. [9] showed that a simple “Are you sure?” makes models change their answers and lowers accuracy. Xie et al. [10] found that models waver under skeptical, negating, or misleading follow-ups even when the first answer was correct, and proposed a training method to reduce this. Wang et al. [11] showed that ChatGPT often cannot defend a correct answer against invalid arguments. Huang et al. [12] found that models rarely improve their reasoning by intrinsic self-correction without outside feedback, and Laban et al. [13] showed that performance drops across many tasks in multi-turn conversation. Our results agree on one side and add the other: empty challenges break correct answers (C→I), but neither challenges nor re-asks repair wrong ones (I→C). We also test whether a persistent error survives a chat reset.

## Verbalized confidence.

Asking a model to state its confidence is the cheapest measure in deployed systems. Lin et al. [14] showed that a fine-tuned GPT-3 can express calibrated uncertainty in words. Without such fine-tuning, the stated confidence of instruction-tuned models is often overconfident or poorly calibrated [15], though prompting can help, especially for models trained with human feedback [16]. Geng et al. [17] survey confidence estimation and calibration for LLMs. Calibration is usually measured with ECE [18]. In our dialogues, the movement of verbalized confidence is mostly a Qwen2.5-7B efect of turn depth, not a response to evidence.

## Self-evaluation with P(True).

Kadavath et al. [4] asked a model whether its own proposed answer is true and read the probability of “True”. They found this fairly well calibrated for large models on many single-turn tasks. In our dialogues P(True) ranks correct answers well at both turns but barely moves under re-asking or agreement. It responds mainly to contradiction by a foil.

## Probing hidden states for truthfulness.

Several papers train classifiers on hidden activations to read out whether a statement or answer is true. Azaria and Mitchell [19] trained such a classifier on true and false statements. Marks and Tegmark [20] reported linear structure that separates true from false statements. Burns et al. [21] found latent knowledge without labels. Orgad et al. [22] showed that truthfulness signals concentrate in the exact answer tokens, generalize poorly across datasets, and that a model may encode the correct answer while generating a wrong one. A probe can learn the task itself rather than reveal what the representation encodes, so probe results need care [23]. We fit a probe at turn 1, freeze it, and follow it after the user speaks again.

## Confidence as commitment.

Two recent single-turn studies are close to our result on the measures. Kumaran [3] used a two-stage task: a model answers and reports its confidence, then later decides whether to give the answer to a user or abstain. Across several models, verbal confidence predicted the later commit-or-abstain decision better than it predicted correctness, while calibrated token log-probabilities tracked correctness. In Gemma models, a post-answer hidden state encoded the later decision, and steering along a direction specific to verbal confidence shifted abstention. Miao and Ungar [5] found that a linear accuracy probe and the direction behind verbalized confidence are nearly orthogonal, and used activation steering to bring verbalized confidence closer to the probe. We do not claim these points as new. Our study difers in three ways. The user turn is controlled, so we can separate the speech act from the answer string. The probe is fitted at turn 1 and frozen, and we follow the same items across turns. And we study the frozen probe readout, not verbalized confidence: in our dialogues, on Qwen2.5-7B, it is the probe that moves with the answer the model keeps. We run no steering, so unlike these two studies our evidence is correlational. Sarkar et al. [24] show, in a collaborative dialogue task between two LLMs, that a partner’s assertion can shift a model’s linear representation of whether a proposition is true. Our probe result points the same way for short factual answers.

## Multi-turn calibration.

The closest work studies confidence across turns. Zhang et al. [1] evaluate verbalized confidence, P(True), and a new P(Suficient) score as hints accumulate, and propose an information-aware ECE (InfoECE) that requires confidence to rise steadily as gold information arrives. Zhang et al. [2] show that persuasion worsens a turn-level ECE (ECE@T) and that sequence likelihood can rise when the answer flips to a wrong one, and they train a calibrator, MTCal, to repair the score. Both ask how well a score tracks correctness. We use their setting (short-answer dialogue under persuasion) and ask an earlier question: which quantity a given user turn moves (Table 1).

## Knowledge conflicts and retrieval.

Models follow context that contradicts their stored knowledge to varying degrees [25, 26], and whether retrieved context is suficient to answer the question predicts errors in retrieval-augmented generation [27]. Lathkar [28] reports that, in multi-hop retrievalaugmented QA, one confirmed intermediate fact can raise the rate of confident wrong answers before complete evidence lowers it. That work is single-turn and measures confidence; we measure whether evidence repairs an error that has already survived a re-ask. In our data, evidence repairs such an error only when it contains the answer string.

## 3 Experimental setup

## 3.1 Dialogue protocol

Each dialogue starts at turn 1 (T1) with a short-answer question. At turn 2 (T2), and at later turns when used, the user sends one of four core messages, fixed across models:

Table 1 How this work relates to the closest prior work. “Left open” lists what each design does not separate
<table><tr><td></td><td>Question asked</td><td>Left open</td></tr><tr><td>Zhang et al. [1]</td><td>Estimation as evidence accumulates; InfoECE; P(Sufficient)</td><td>Frozen hidden-state readout; gold vs. foil vs. empty challenge; whether verbalized confidence, likelihood, and P(True) agree on</td></tr><tr><td>Zhang et al. [2]</td><td>ECE@T after persuasion; likeli- hood; a learned calibrator</td><td>one transcript Verbalized confidence vs. frozen probe; agreement and correction turns; reset; removing the answer</td></tr><tr><td>Kumaran [3]</td><td>Does verbal confidence track correctness or a later com- mit/abstain decision? (single</td><td>string User follow-up turns; behaviour across turns</td></tr><tr><td>Miao and Ungar [5]</td><td>turn) confidence direction; steering and P(True)</td><td>Accuracy probe vs. verbalized- User follow-up turns; likelihood</td></tr><tr><td>Sycophancy [6, 7]</td><td>(single turn)</td><td>User states X ⇒ model says X Separating “you are wrong&quot; from the stated string</td></tr><tr><td>This work</td><td>Which user changes which quantity across such as steering turns; st is not truth</td><td>intervention A new estimator; causal tests</td></tr></table>

confirm Yes, that’s correct. Please answer again, briefly, in one short phrase.   
persuade One of five challenge templates plus a foil.   
correct The same five templates plus a gold alias.

Ablations keep the same T1 and change only the T2 message, to separate the speech act from the payload. gold plain and foil plain state a string without saying “wrong”. challenge empty challenges the answer without ofering a replacement. confirm strong repeats the model’s T1 answer and calls it correct. persist short is Again, briefly. Exact wording is in Online Resource 1, Sect. A.

## 3.2 Notation and outcome categories

We compare each item’s correctness at turn t with its correctness at T1. An item is stay-correct (SC) if it is correct at both turns, stay-wrong (SW) if it is wrong at both, I→C if it changes from incorrect to correct, and C→I if it changes from correct to incorrect. We report I→C and C→I as counts out of the number of items (usually 200) or as rates. For a measure X, $\Delta X = X _ { \mathrm { T 2 } } - X _ { \mathrm { T 1 } }$ is its change from T1 to T2, averaged over the stated items. The copy rate is the fraction of I→C answers that repeat the payload string exactly. A unit is one model×dataset combination. Seed copies of the same unit are not counted as independent.

## 3.3 Confidence measures

We use the four quantities that current multi-turn papers treat as interchangeable measures of confidence. The list is not exhaustive. Token entropy, logit margin, and semantic entropy [29] read the same next-token distribution as likelihood, so we expect them to behave like it unless an intervention separates them.

• Verbalized confidence (∆verb for its change) is what products elicit [15, 16]. The model is asked to state a confidence, which is parsed from a separate CONF field into [0, 1].

• Sequence likelihood (“ll”) is the quantity scored by ECE@T and MTCal [2]: exp of the mean log-probability of the answer tokens, teacher-forced under the chat template.

• Frozen probe is the kind of hidden-state readout that the truthfulness-probing literature reads as a signal of truth [19, 20]. It is defined in Section 3.4.

• P(True) is the Kadavath self-evaluation used alongside P(Suficient) [1, 4]: a forced choice between A (true) and B (false), scored by the softmax probability of A. We computed it on the Qwen2.5-7B TriviaQA set of 800 rows (200 items × four core messages).

## 3.4 The persistence readout: an operational definition

Let $h _ { t } \in \mathbb { R } ^ { d }$ be the last-layer hidden state averaged (mean-pooled) over the answer tokens at turn $t ,$ and let $\tilde { h } _ { t }$ be $h _ { t }$ standardized with the training-set mean and variance. We fit a weight vector $v _ { \mathrm { a c c } } \in \mathbb { R } ^ { d }$ by logistic regression to predict T1 correctness from $h _ { 1 }$ , using only the unique persist T1 items of that model×dataset. GroupKFold cross-validation is used only as a sanity check. We then freeze $v _ { \mathrm { a c c } }$ and never refit it after the user speaks. The persistence readout at turn t is

$$
s _ { t } = \sigma \big ( \boldsymbol { v } _ { \mathrm { a c c } } ^ { \top } \tilde { \boldsymbol { h } } _ { t } \big ) ,\tag{1}
$$

where $\sigma ( x ) = 1 / ( 1 + e ^ { - x } )$ is the logistic function. We use “probe” and $s _ { t }$ interchangeably and write $\Delta s = s _ { 2 } - s _ { 1 }$ . Three points make the definition precise:

• What it is. $s _ { t }$ is a frozen readout of the current answer state. The linear map is one choice. Our claims about $s _ { t }$ apply to any frozen readout that is fitted only on unique T1 states, is not refitted after the user speaks, and rises on stay-wrong and on I→C items (Section 6).

• What the name means. “Persistence” names an observed association with the answer the model keeps. We make no claim about a psychological state, and no claim that $s _ { t }$ causes the answer. The main analyses fit the probe on unique T1 items and apply it to rows of the same questions. A post hoc rerun on held-out questions (Section 6) keeps the result on Qwen2.5-7B only.

• Why it is not truth. If $s _ { t }$ measured truth, it would rise only when the answer becomes gold. Instead it rises on stay-wrong and on I→C items and falls on C→I items.

• The behaviour it goes with. An item is stay-wrong if it is wrong at T1 and still wrong after persist at T2. On Qwen2.5-7B TriviaQA there are n=81 such items, and $\Delta s > 0$ on 79 of their 80 distinct questions. Reset and abstention fail on this set; a gold string at T3 does not (Section 6.1).

We do not read $v _ { \mathrm { a c c } }$ as a discovered truth direction [20, 21]. A multilayer perceptron (MLP) probe with one hidden layer matches the T1 discrimination of the linear probe and shows the same pattern, so the result does not depend on linearity.

## 3.5 Answer matching

Answers are normalized (lowercased, articles removed, whitespace collapsed) and counted correct on an exact match or on whole-word containment, in either direction, between the answer and a gold alias [30]. We call this the contain matcher. It is the primary, preregistered scorer. Where the distinction matters (Qwen3) we also report exact-normalized scores. Containment has a known weakness: it credits an answer that names the gold only to reject it (Norman Rockwell, not Edward Hopper). This matters for Mistral’s long answers, which an LLM therefore read in full. Other models were screened only with a negation regular expression (Online Resource 1, Sect. I). The second preregistration also scored every answer with an LLM judge. The judge missed its preregistered validation bar, so contain stays primary, and judge and modelread rescorings are reported beside it (Online Resource 1, Sect. H). Post hoc, two independent people labelled a blind sample of 300 V2 rows; on the 272 clear ENDORS-ES/DOES NOT pairs Cohen’s κ was 0.86 (agreement 253/272), with disagreements resolved by discussion. This is a post-hoc reliability check, not a change of the primary scorer (Online Resource 1, Sect. H).

Under our generation settings, Llama-3-8B-Instruct does not stop at its end-ofturn token <|eot id|> and repeats its answer after a leaked assistant role header. For Llama we score the text before the first leaked role header, because in the raw runon text the matcher misses answers glued to the header (9–19% of Llama T2 labels change).

## 3.6 Models and data

The main grid uses instruction-tuned models from 3B to 14B parameters: Qwen2.5- {3,7,14}B [31], Qwen2-7B, Qwen3-8B [32], Llama-3-8B-Instruct [33], Mistral-7B-Instruct-v0.3 [34], and Phi-4 [35]. Qwen2.5-32B-Instruct has one TriviaQA T1/T2 cell. We use it only to find stay-wrong items for the turn-3 battery and the retrieval study (Section 6.1), and it is not one of the 24 units below. We had no access to the gated Llama-3.1 and Gemma checkpoints.

Datasets are TriviaQA rc.nocontext [30], NQ-open [36], HotpotQA questions without passages [37], TruthfulQA [38], and WebQuestions [39]. For TriviaQA we use a main slice [0, 200), fixed after shufling with seed 42, plus a disjoint slice [200, 400) and an extra slice of n = 500. GSM8K [40] uses the same short-answer protocol on gradeschool mathematics (matching the last number); we make no claim about long-form reasoning. We exclude PopQA because several models fall to accuracy 0, a matcher and format failure. Unless stated otherwise, $n = 2 0 0$ , seed 42, and decoding is greedy.

## 3.7 Replication predicates

To check whether each pattern generalizes, we evaluate five yes/no predicates on every independent unit:

$\mathrm { P _ { e r r } } \colon$ on stay-wrong items, ∆probe > 0 under both persist and confirm.

$\mathrm { P _ { g o l d } } \colon$ I→C under correct is at least 0.15, and $\mathrm { I {  } C }$ is at most 0.03, 0.05, and 0.05 under persist, confirm, and persuade.

$\mathrm { P _ { v e r b } } \mathrm { . }$ : mean ∆verb $< ~ 0$ on all four core messages (models whose verbalized confidence is flagged VERB UNTRUSTED are skipped).

$\mathrm { P } _ { \mathrm { l l } } \colon$ under persist, the most common sign combination is verbalized confidence not up and likelihood up, or mean $\Delta \mathrm { l l } > 0$ with mean ∆verb $< 0$

$\mathrm { P _ { a r t i f a c t } } \mathrm { : }$ at least 80% of C→I changes under correct are items whose T1 answer was already gold under a loose alias check, that is, a matcher artifact rather than a real loss.

On 24 independent units, $\mathrm { P _ { e r r } = P _ { l l } = P _ { a r t i f a c t } = 2 4 / 2 4 . ~ P _ { g o l d } }$ holds on $1 9 / 2 4$ . The five failures are the three Qwen3 units, which we keep in the denominator although their cells are truncated “self-repair” cells that we withdrew after extracting answers past </think> (Online Resource 1, Sect. C), and Llama-3-8B on TriviaQA and NQ (Section 4). $\mathrm { P _ { v e r b } }$ holds on all six Qwen2.5-7B dataset units and on no other model family.

## 3.8 Preregistration and what it cost

Before any generation, a frozen pilot preregistration [41] fixed two hypotheses and a stopping rule. H1 stated that the frozen $v _ { \mathrm { a c c } }$ predicts T2 correctness after misleading persuasion better than verbalized confidence does. H2 stated that verbalized confidence rises from T1 to T2 under persuasion while the probe’s predicted probability of correctness does not. H2 failed its gate: verbalized confidence fell (−0.17). H1 met neither preregistered bar: the advantage was +0.06 with a bootstrap interval covering zero. The verdict fixed in advance was to continue with no surviving hypothesis. We report H2 as failed rather than dropping it. What replaced the hypotheses was an observation: the measures were responding to diferent parts of the user’s turn.

Apart from two later preregistered studies of persistent errors and retrieval [42, 43] (Section 6.1), nothing in this paper is confirmatory. Every other result is exploratory measurement, and we label it as such. All three preregistration documents are in the supplementary material (Online Resource 2).

## 4 The answer string, not the speech act

Table 2 and Figure 2 are the controlled core of the study: T1 is identical and only the T2 message changes. On the 200 main TriviaQA items, the accusation does almost none of the work. Removing it leaves gold uptake nearly intact (gold plain converts 57 items against 65 for correct), and a bare foil pulls more answers away from gold than the templated persuade message (77 vs. 30). Removing the string instead removes the direction. challenge empty scatters answers both ways (71 C→I, 12 I→C), which looks like added randomness rather than an update. Agreement, bare or emphatic, behaves like re-asking, and dropping “briefly” does not loosen uptake.

Table 2 Core T2 messages on Qwen2.5-7B-Instruct, TriviaQA $( n = 2 0 0 ,$ , seed 42). I→C and C→I are counts out of 200; “acc T2” is accuracy at T2; ∆verb is the mean change in verbalized confidence. correct long drops the word “briefly”; its I→C stays at 67 (copy rate 0.60)
<table><tr><td>T2</td><td>I→C</td><td>C→I</td><td>acc T2</td><td>∆verb</td></tr><tr><td>persist</td><td>0</td><td>5</td><td>.570</td><td>-0.10</td></tr><tr><td>confirm</td><td>2</td><td>8</td><td>.565</td><td>-0.10</td></tr><tr><td>confirm_strong</td><td>0</td><td>2</td><td>.585</td><td>-0.06</td></tr><tr><td>persist_short</td><td>4</td><td>12</td><td>.555</td><td>-0.01</td></tr><tr><td>challenge_empty</td><td>12</td><td>71</td><td>.300</td><td>-0.16</td></tr><tr><td>gold_plain</td><td>57</td><td>10</td><td>.830</td><td>-0.08</td></tr><tr><td>correct</td><td>65</td><td>15</td><td>.845</td><td>-0.15</td></tr><tr><td>foil_plain</td><td>1</td><td>77</td><td>.215</td><td>-0.14</td></tr><tr><td>persuade</td><td>0</td><td>30</td><td>.445</td><td>-0.17</td></tr><tr><td>correct_long</td><td>67</td><td>一</td><td>一</td><td>一</td></tr></table>

![](images/2b82ce2021591849881cd2b7515e54d0c65ddc6940cb0bce9848d73be159d222.jpg)  
Fig. 2 Speech act vs. payload on Qwen2.5-7B TriviaQA $( n = 2 0 0 )$ : I→C and C→I counts per T2 message. Gold and foil strings move answers; re-asking and agreement do not. A bare foil statement flips more items than the templated persuade message. An empty challenge makes answers unstable without directing them

The split does not depend on our exact T2 wording. Twenty-seven everyday paraphrases of the four T2 messages, applied to the same T1 histories, keep mean I→C at 0.026 when no string is named and 0.275 when gold is named (foil C→I 0.325; Online Resource 1, Table D1). Hedged, assistant-like phrasing does not replace the payload either. I think it’s {gold} still converts (0.295), while I think you might be of does not (0.045). The same three ablations meet the same gates on NQ, HotpotQA, and WebQuestions. The pattern also holds on GSM8K under the short-answer protocol: persist converts 14/188 T1-wrong items, and on the 174 items still wrong, gold converts 0.437 against 0.103 for Try again.

Table 3 The payload split holds across models and datasets: an excerpt of the model×dataset matrix (counts out of 200). The I→C columns compare persist with correct; the C→I columns compare persist with persuade. Stating gold converts 48–116 items on every row under the contain matcher (39 for Mistral TriviaQA once answers that a model reading them judged not to endorse the gold are removed; Online Resource 1, Sect. I); re-asking converts 0–11. Qwen3 is omitted (truncated thinking cells). Llama is scored before the leaked role header (Section 3). P<sub>err</sub> holds on all of these units
<table><tr><td></td><td></td><td colspan="2">I→C</td><td colspan="2">C→I</td></tr><tr><td>Model</td><td>Data</td><td>persist</td><td>correct</td><td>persist</td><td>persuade</td></tr><tr><td>Qwen2.5-7B</td><td>TriviaQA</td><td>0</td><td>65</td><td>5</td><td>30</td></tr><tr><td>Qwen2.5-7B</td><td>NQ</td><td>3</td><td>108</td><td>2</td><td>21</td></tr><tr><td>Qwen2.5-7B</td><td>HotpotQA</td><td>0</td><td>116</td><td>1</td><td>20</td></tr><tr><td>Qwen2-7B</td><td>TriviaQA</td><td>5</td><td>71</td><td>4</td><td>46</td></tr><tr><td>Mistral-7B</td><td>TriviaQA</td><td>2</td><td>49</td><td>0</td><td>23</td></tr><tr><td>Llama-3-8B</td><td>TriviaQA</td><td>8</td><td>48</td><td>1</td><td>56</td></tr><tr><td>Phi-4</td><td>TriviaQA</td><td>0</td><td>50</td><td>0</td><td>8</td></tr><tr><td>Qwen2.5-14B</td><td>TriviaQA</td><td>2</td><td>51</td><td>1</td><td>22</td></tr><tr><td>Qwen2.5-3B</td><td>TriviaQA</td><td>6</td><td>84</td><td>12</td><td>27</td></tr><tr><td>Llama-3-8B</td><td>NQ</td><td>11</td><td>107</td><td>6</td><td>34</td></tr><tr><td>Phi-4</td><td>NQ</td><td>1</td><td>109</td><td>2</td><td>10</td></tr><tr><td>Mistral-7B</td><td>HotpotQA</td><td>2</td><td>108</td><td>10</td><td>19</td></tr></table>

## Generalization across the grid.

In every retained full-size cell (all except PopQA, the 64-token Qwen3 cells, and the n≈20 turn-depth subsamples), I→C under correct is at least four times that under persist or confirm (Table 3). The closest cases are Llama-3-8B TriviaQA (48 vs. 11) and GSM8K (76 vs. 18 for Try again). Across cells, I→C under persist has a median of 2/200 (minimum 0). An earlier persist I→C of 40 came from the truncated Qwen3 cell that we withdraw in Online Resource 1, Sect. C. Llama-3-8B misses P<sub>gold</sub> on TriviaQA and NQ because re-asking and agreement convert slightly more than the gates allow (persist I→C 8 and 11/200; confirm 11 and 10/200). This is still far below gold’s 48 and 107. Phi-4 barely moves under persuade (accuracy 0.70 → 0.71 on TriviaQA) but still adopts gold (I→C = 50, C→I = 0). Qwen2.5 and Llama-3-8B mostly copy the payload; Mistral-7B restates it inside a longer sentence, sometimes only to reject it. Copy rates and the model-read audit are in Online Resource 1, Sect. I.

## 5 Four measures, one transcript

The four measures disagree in sign in every row of Table 4 and Figure 3. Under correct, verbalized confidence falls while accuracy climbs to 0.845. The disagreement remains when we split items by what the answer did. Under persist, P(True) barely moves on items that stayed wrong (+0.012), while the probe rises by +0.38 on the same items. Under persuade, P(True) falls most where the answer flipped (C→I −0.845, stay-wrong −0.171), yet the probe on stay-wrong items still rises (+0.27). Under correct, P(True) moves both ways at once (I→C +0.148, C→I −0.411) around a mean of ≈ 0, while the probe rises by +0.64 on I→C items and +0.27 on the 16 stay-wrong items. No single ECE curve can capture that.

Table 4 Mean change from T1 to T2 $( \Delta = \mathrm { T 2 - T 1 } )$ of each measure on Qwen2.5-7B TriviaQA (n = 200). Values split by what the answer did are given in the text
<table><tr><td></td><td>∆verb</td><td>∆ll</td><td>∆probe</td><td> $\Delta P ( \mathrm { T r u e } )$ </td></tr><tr><td>persist</td><td>-0.10</td><td>+0.14</td><td>+0.15</td><td>+0.007</td></tr><tr><td>confirm</td><td>-0.10</td><td>+0.14</td><td>+0.15</td><td>+0.002</td></tr><tr><td>persuade</td><td>-0.17</td><td>+0.02</td><td>-0.03</td><td>-0.258</td></tr><tr><td>correct</td><td>-0.15</td><td>+0.09</td><td>+0.17</td><td>-0.003</td></tr></table>

![](images/abbafc5db0a207718333f1869b83ea83f299825c4bd5cbeee53eccab0ef8d925.jpg)  
Fig. 3 Mean T2−T1 change of four measures on one Qwen2.5-7B TriviaQA set. Under persist, verbalized confidence falls, likelihood and the probe rise, and P(True) stays still. Under persuade, P(True) collapses while the probe on stay-wrong items still rises (Table 4)

## The drop in verbalized confidence is model-specific.

The drop survives two obvious explanations. Residualizing on answer length leaves a mean $\Delta ~ \mathrm { o f ~ - 0 . 1 3 }$ , and the word “briefly” is not responsible, since correct long still converts. It does not survive a change of model. $\mathrm { P _ { v e r b } }$ holds on 6/6 Qwen2.5-7B dataset units and on no other family once untrusted parses are dropped. Phi-4 moves the other way on all four core messages. Qwen2.5-3B and 14B fail on confirm (+0.013, +0.044). Llama-3’s −0.537 is flagged VERB UNTRUSTED because its values pile up at 0. Qwen3 fails to parse on 96% of truncated rows because the CONF field sits inside <think>.

Under persist on Qwen2.5-7B TriviaQA, the most common sign pattern is likelihood rising while verbalized confidence does not rise (172/200 in the pilot cell; 51 with a strict drop). In that cell, likelihood (the quantity MTCal calibrates) and verbalized confidence point in opposite directions on the same tokens.

## Other models (post hoc).

On Qwen2.5-14B, Mistral-7B-v0.3, Phi-4, and Qwen2-7B (not preregistered; Online Resource 1, Sect. J), likelihood still rises under persist except on Mistral, P(True) stays within 0.03, and the all-item ∆probe is negative on three of the four. The exact pattern in Table 4 is a Qwen2.5-7B result; what holds across models is the disagreement.

Table 5 ECE (10 bins) and AUROC against correctness on one transcript: Qwen2.5-7B TriviaQA, persuade, common n=195 (items whose verbalized confidence parses at both turns). Both probe columns are out-of-fold under GroupKFold over items: each fold’s T1-fitted map scores that fold’s own held-out T1 and T2 states, so the row is estimated the same way at both turns. The in-sample figures (ECE ≤0.002, AUROC 1.000) measure the regression on its own training set and are not calibration numbers. Verbalized ECE falls while mean confidence stays about 0.3 below accuracy, so it is not a repair
<table><tr><td rowspan="2">Measure</td><td colspan="2">ECE</td><td colspan="2">AUROC</td></tr><tr><td>T1</td><td>T2</td><td>T1</td><td>T2</td></tr><tr><td>Verbalized</td><td>0.497</td><td>0.375</td><td>0.564</td><td>0.729</td></tr><tr><td>Likelihood</td><td>0.230</td><td>0.411</td><td>0.852</td><td>0.630</td></tr><tr><td>Probe (held out)</td><td>0.118</td><td>0.260</td><td>0.848</td><td>0.649</td></tr><tr><td>P(True)</td><td>0.220</td><td>0.203</td><td>0.852</td><td>0.862</td></tr></table>

## The same split appears in ECE.

Table 5 scores all four measures against T1 and T2 correctness on one transcript. Under persuade, verbalized ECE falls (0.50 → 0.38), likelihood ECE nearly doubles (0.23 → 0.41), and P(True) ECE is flat (0.22 → 0.20). The split in Figure 1 therefore also appears in the field’s standard metric. A study has to say which measure it calibrates.

The drop in verbalized ECE is not better calibration. Stated confidence stays about 0.3 below accuracy at both turns (0.28 vs. 0.59, then 0.10 vs. 0.44). AUROC (how well a score ranks correct above wrong answers) shows the same. At T1, verbalized confidence is close to uninformative (0.564), while likelihood, the probe, and P(True) all sit near 0.85. At T2 the order changes: verbalized confidence rises to 0.729 and likelihood drops to 0.630. Which measure a study happens to elicit decides how calibrated the result looks, and also whether it ranks correct answers at all. A post hoc rerun reproduces every cell of Tables 4 and 5 within ±0.002. An older output file listed verbalized ECE as 0.582 → 0.435; it divided confidences that were already in [0, 1] by 100 again, and the values above are the corrected ones (Online Resource 1, Sect. J).

The probe’s own drop in AUROC (0.848 → 0.649) is not noise. It is the persistence efect of Section 6 seen in calibration terms. A readout that rises on items that stay wrong (Table 4) must lose discrimination against correctness once the user has spoken, because it now follows the answer the model keeps rather than what is true. In the same turn, the probe becomes a worse readout of truth and a better readout of answer persistence, and a single “confidence” number hides this.

## 6 Answer persistence, not truth

On Qwen2.5-7B, the frozen $v _ { \mathrm { a c c } }$ is a weak readout of truth and a consistent correlate of answer persistence. The post hoc controls below, including a rerun on held-out questions, limit the strong form of this result to Qwen2.5-7B, with Llama-3-8B on cut answers as support.

## Direction and size.

On all 24 independent units, stay-wrong ∆probe is positive under both persist and confirm $\left( \mathrm { P _ { e r r } = 2 4 / 2 4 } \right)$ . These are in-sample numbers: the probe scores questions it was fitted on. On held-out questions the sign survives on Qwen2.5-7B and Llama-3- 8B and fails on Qwen2.5-14B, Mistral, and Phi-4 (see Held-out questions below). A sign count is the weakest form of this result, so we also report magnitudes. Across the 32 unit×message cells with item-level states, the median of the mean ∆probe is 0.264 (interquartile range [0.127, 0.384], range [0.025, 0.758]), and each cell’s own bootstrap 95% interval excludes zero (32/32). These cells are TriviaQA for seven models (Qwen3 in its truncated cell) and Qwen2.5-7B on five further slices. Six of the 32 are seed or configuration copies of the main Qwen2.5-7B cell. Sign counts over all 113 cells are in Online Resource 1, Sect. J. An MLP probe, a probe refitted on persist T2, and layer and token ablations keep the sign (Online Resource 1, Sect. J).

So the probe rises when the model keeps a wrong answer and when it adopts gold, and it falls when a correct T1 answer is pulled away (persuade C→I ∆probe −0.35 in the unique-fit follow-up). It goes with the current answer, not with a latent true/false feature [20].

## Post hoc controls.

Four checks were defined in a note written before the numbers and were not preregistered (Online Resource 1, Sect. J). (a) A probe fitted on shufled T1 labels (100 seeds) stays near 0 on Qwen2.5-7B, against 0.383 for the real probe; on Qwen2.5-14B persist, Mistral, and Phi-4, 29–59% of shufles reach the real value. (b) On Qwen2.5- 7B the rise does not need a changed answer (identical answers 0.397, n = 67; changed 0.318, n = 14). (c) Within stay-wrong items, ∆probe and ∆likelihood are only weakly related (Spearman −0.09 to 0.27). (d) A planned re-read without dialogue history is not interpretable: re-scoring the T1 answer did not reproduce the stored states (cosine 0.81, below the required 0.95, even after reusing the generated token ids), so we draw no conclusion from it (Online Resource 1, Sect. J).

## Held-out questions (post hoc).

We refitted the probe in five folds grouped by question, so it never saw a test question at any turn, and scored stay-wrong items on the held-out folds only (Online Resource 1, Table J10). On Qwen2.5-7B, ∆s is 0.321 [0.255, 0.389] under persist and 0.324 [0.252, 0.395] under confirm (in sample 0.383, 0.384), and no shufled-label probe reaches it. Llama-3-8B, with states recomputed on cut answers, gives 0.201 [0.131, 0.275] and 0.119 [0.051, 0.192], also above every shufle. On the other models the result does not hold. Qwen2-7B drops to 0.067 [−0.003, 0.136] and 0.019 [−0.058, 0.090]. Mistral and

Phi-4 turn negative. Qwen2.5-14B is negative under persist (−0.100) and positive under confirm (0.117 [0.019, 0.212]). The held-out probe still ranks T1 correctness on every model (AUROC 0.69–0.85).

## 6.1 Stay-wrong errors are hard to fix

Take the 81 Qwen2.5-7B TriviaQA items that are wrong at T1 and still wrong after persist at T2 (80 distinct questions; one appears twice in the slice). Further persist turns convert one of them by T3 (accuracy on stay-wrong items = 0.012), two by T5 (0.025), and three by T10 (0.037). A reset at T3 (a fresh chat with only the original question) converts none. A T3 that continues the chat and ofers abstention (If you are not confident . . . reply ABSTAIN) draws abstention on 94% of stay-wrong and 62% of stay-correct items. Every stay-wrong item that still answers is wrong, so coverage on stay-wrong items is 0.06. Gating on the probe $\mathrm { ( p r o b e } _ { T 2 } > \mathrm { p r o b e } _ { T 1 }$ , true for 79 of the 80 distinct stay-wrong questions) does not change these accuracies. Re-asking, wiping the transcript, and ofering abstention do not repair these errors.

A gold string at T3 does (Table 6). Every gold phrasing recovers 0.75–0.95 of the 81 items. Hedging does not lower it (0.79). A diferent gold alias recovers 0.77 $( n _ { \mathrm { S W } } = 6 9 )$ , with a copy rate of only 0.49 for that surface form (main gold messages 0.61–0.71), so the repair is not only echoing. The gold retrieved-note phrasing leaves stay-correct items intact (accuracy 0.965, C→I 4/114).

## 6.2 Preregistered replications

Two preregistrations fixed models, items, messages, metrics, and a decision rule before any generation: a message without an answer string that reaches ≥ 0.15 repair on stay-wrong items breaks the null for that model [42, 43]. The first ran the turn-3 battery on Qwen’s n=500 slice (230 stay-wrong items) and on the original Llama and Mistral items (61 and 58; Online Resource 1, Sect. E). The second regenerated Llama-3-8B and Mistral-7B-v0.3 on the same 500 items with correct stop tokens (160 and 139 stay-wrong items) and added Qwen2.5-32B (61), all on GPU (Online Resource 1, Sect. F).

Under the preregistered matcher, every observed rate without an answer string is below 0.15 on every unit, so the decision rule never fires (Table 6). That is weaker than excluding 0.15, and several small cells did not (Wilson upper bounds up to 0.26). A post hoc rerun on 298 (Llama) and 269 (32B) new stay-wrong items (Online Resource 1, Table J11) now excludes it for re-asking, fresh chat, and ABSTAIN (at most 6/269, upper bounds $\leq 0 . 0 4 8 )$ . The empty elicit does not: 35/298 [0.086, 0.159] and 32/269 [0.086, 0.163]; pooled with earlier rows, Llama excludes 0.15 (50/458, upper 0.141) and 32B does not (38/330, upper 0.154). Llama’s highest arms in the first replication (up to 9/61 = 0.148) come mostly from the leaked role header. In a post hoc check on the 54 rows that are still stay-wrong after cutting answers at the header, fresh chat recovers 2/54 (0.04) and the other arms without an answer string recover none (Online Resource 1, Table E2). In the second replication the highest such arm is the empty elicit, at $6 / 6 1 = 0 . 0 9 8$ on 32B and $1 5 / 1 6 0 = 0 . 0 9 4$ on Llama (Wilson upper bounds 0.198 and 0.149; the 32B interval includes 0.15). Mistral’s gold rates (0.72/0.71) overstate uptake; read by a model, they are 0.55/0.25.

Table 6 T3 accuracy on persist stay-wrong items (contain matcher; rows 2–5 and the last column are preregistered). Re-ask: persist at T3 (main 81: .025 at T5); reset: fresh chat; abst.: ABSTAIN option; empty: elicit with no string; gold plain/retrieve are oracle arms; last column: best BM25 evidence sentence with/without a gold alias. <sup>∗</sup>Read by a model: .55/.25 and .63/.06. Intervals and other scorers are in Online Resource 1, Sects. E–H
<table><tr><td>Stay-wrong unit</td><td></td><td></td><td></td><td></td><td>n re-ask/reset/abst. empty gold plain/retr. answer/no answer</td></tr><tr><td>Qwen2.5-7B main</td><td>81</td><td> $. 0 1 / . 0 0 / . 0 0$ </td><td>.06</td><td>.78/.93</td><td>.67/.05</td></tr><tr><td>Qwen2.5-7B n=500</td><td>230</td><td> $. 0 3 / . 0 5 / . 0 0$ </td><td>.03</td><td> $. 9 2 \dot { / } . 9 4 $ </td><td></td></tr><tr><td>Llama-3-8B</td><td>160</td><td> $. 0 1 \dot { / } . 0 0 \dot { / } . 0 1$ </td><td>.09</td><td> $. 8 9 \dot { / } . 9 1$ </td><td> $. 6 8 / . 1 2$ </td></tr><tr><td>Mistral-7B</td><td>139</td><td> $. 0 1 \dot { / } . 0 0 \dot { / } . 0 0$ </td><td>.01</td><td> $. 7 2 \dot { / } . 7 1 ^ { * }$ </td><td> $. 7 6 \dot { / } . 0 2 \dot { ^ { \ast } }$ </td></tr><tr><td>Qwen2.5-32B</td><td>61</td><td>.00/.00/.00</td><td>.10</td><td> $. 9 \dot { 2 } / . 9 0 $ </td><td>.69/.11</td></tr></table>

One result goes against the null. With judge labels, and model-read labels for the rows where the matcher and judge disagree, Llama’s empty elicit is $2 4 / 1 6 0 = 0 . 1 5 0$ [0.103, 0.213], so the decision rule applied to those labels does not replicate for Llama. But 9 of the 24 repairs were already correct at T1 and T2 under those labels; with staywrong defined by them (post hoc), the arm is $1 5 / 1 4 9 = 0 . 1 0 1 \ [ 0 . 0 6 2 , 0 . 1 5 9 ]$ (Online Resource 1, Sect. H).

Frozen s<sub>2</sub> does not predict which stay-wrong items gold leaves unrepaired (AUROC 0.39, 0.49, 0.62 on Llama, Mistral, 32B; every interval contains 0.5; Online Resource 1, Tables E2 and F3).

## 6.3 Retrieval repairs only when it contains the answer

Every gold arm above is an oracle: the user hands the model the right answer. The second preregistration therefore tested real evidence on the same stay-wrong items and on up to 150 stay-correct items per model. BM25 [44] ranks the sentences of each question’s TriviaQA evidence documents, and T3 inserts one sentence into the retrieved-note message. The top-ranked sentence repairs 0.22–0.27 of stay-wrong items, and how often it contains the answer explains that rate. Pooled over four models, 90/111 items recover when it contains a gold alias and 20/330 when it does not. Holding the ranking fixed, the best sentence that contains the answer repairs 0.67–0.76, and the best sentence that lacks it repairs 0.02–0.12. The diference is 0.56–0.73, with every interval above 0, and 0.55–0.61 with model-read labels, under which Llama’s noanswer sentence reaches $2 6 / 1 6 0 = 0 . 1 6 .$ . Sentences without the answer are not neutral. They cut stay-correct accuracy from 1 to 0.65, 0.78, 0.30, and 0.78.

## 7 Implications for conversational systems

You have to split conversational confidence into its parts before you can calibrate it. ECE@T on likelihood [2] scores fluency, and P(Suficient) [1] scores evidence. Neither measures uptake of the user’s string or the persistence readout $s _ { t } .$ . On one transcript, I→C and C→I counts show whether a user string overwrote the answer, s<sub>t</sub> (on Qwen2.5-7B) follows the answer the model keeps, P(True) shows whether a contradiction landed, and likelihood shows fluency. Table 7 lists what this means in practice.

Table 7 Suggested changes to evaluation and system design if conversational confidence is not one latent belief
<table><tr><td>Current practice</td><td>Suggested change</td></tr><tr><td>Report ECE@T, verbalized- confidence AUROC, or P(True) as the multi-turn confidence number Bundle “you are wrong&quot;with a gold or foil string Reset the chat, or gate on a drop in verbalized confidence, to undo</td><td>State which measure in Figure 1 is scored. Scores on one transcript are not substitutes, and fixing one does not fix the others. Separate the speech act from the answer string; otherwise one cannot say what moved. These act on the history and the stated con- fidence, not on st: accuracy on the 81 stay-</td></tr><tr><td>errors Add retrieval and expect repair</td><td>wrong items stays 0. A gold string at T3 repairs them; a foil string overwrites stay-correct answers (Section 6.1). Heuristic, not validated end to end: check that the</td></tr><tr><td></td><td>evidence contains the answer string. A retrieved BM25 sentence that contains it repairs 0.67–0.76 of stay-wrong items, one without it 0.02–0.12 (four models).</td></tr></table>

## For evaluation.

Studies of multi-turn confidence should state which measure they use, separate the speech act from the answer string in their user turns, and use an answer matcher that rejects negated mentions of the gold answer. A calibration result for one measure is not evidence about another.

## For system design.

A re-ask, a chat reset, or an abstain option will not undo a persistent error, and a drop in verbalized confidence will not flag one. In our data, the only reliable repair was new content that contained the answer. For retrieval-augmented assistants this suggests a heuristic: before presenting retrieved evidence as a correction, check whether it contains a candidate answer string. In our data, evidence without one rarely repairs an error and can break up to 0.70 of answers that were correct. In a post hoc end-toend test (BM25 passages, TriviaQA and HotpotQA, three models; Online Resource 1, Table J12), the gold-based gate cut harm on TriviaQA while keeping most repairs (Llama, k = 1: harm 0.179 → 0.038, net per 100 items 15.9 → 27.4); on HotpotQA the gain was mixed. A gate that checks the model’s own answer instead of the gold repaired only 0.02–0.12 of errors, and dense retrieval and a judge check were not run. So the check is a heuristic for systems that can verify answers, not a validated rule.

## 8 Limitations

## What we do not claim.

We do not claim that verbalized confidence always falls over turns (we observe this for Qwen2.5-7B only), that hidden states contain a truth direction, or that we outperform MTCal [2] or P(Suficient) [1] on ECE. We do not claim to be first to show that verbalized confidence and correctness-tracking signals come apart, or that confidence can track commitment; single-turn studies showed both [3, 5]. We make no psychological or causal claim about the persistence readout. We propose no new uncertainty estimator.

## Task scope.

All experiments use English short-answer question answering, with GSM8K run under the same short-answer protocol. We did not study naturalistic or multi-party dialogue, long-form reasoning, coding, or agents, and we did not run closed APIs or logged user conversations. Our evidence is about controlled interventions, not open-ended human conversation. The turn-3 battery contains no reasoning prompt, so we do not claim that step-by-step reconsideration cannot repair persistent errors. The factorial design is not a formal causal-identification argument.

## Model scale and families.

The main grid covers 3B–14B open models. Above 14B we ran only Qwen2.5-32B, and only the stay-wrong battery, its probe, and retrieval. The drop in verbalized confidence is a Qwen2.5-7B result in our grid, and we do not generalize it to other families. Qwen3 needs the answer extracted after </think> (Online Resource 1, Sect. C).

## Sample size of the key cell.

The central stay-wrong analysis on Qwen2.5-7B rests on n = 81 items (80 distinct questions) from one dataset. The preregistered replications extend this to 230, 160, 139, and 61 items on four models, but the turn-3 battery and the retrieval study cover TriviaQA only.

## The persistence readout is defined by a probe.

s is an operational quantity: a frozen readout fitted on T1 states. We make claims about how it behaves when we intervene, not about what the model “believes”. We report no activation-steering or patching results, so we cannot say whether s<sub>t</sub> is causally involved in producing the answer. The single-turn studies that do steer [3, 5] use diferent readouts and tasks, so their causal results do not transfer to ours. Llama-3-8B probe states in the main analyses were pooled over the raw generated answer tokens, including any repetition after the leaked role header. In a post hoc rerun on cut answers, stay-wrong ∆probe stays positive on every Llama unit we could test but is much smaller (TriviaQA persist 0.865 → 0.333, NQ 0.399 → 0.034, HotpotQA 0.238 → 0.030; Online Resource 1, Sect. J). Llama magnitudes from the main analyses should be read as upper bounds. The post hoc shufled-label control (Section 6) and the held-out-question rerun (Section 6) leave the stay-wrong rise standing on Qwen2.5- 7B, with Llama-3-8B on cut answers as support. On Qwen2.5-14B, Mistral, Phi-4, and Qwen2-7B it does not hold on held-out questions, so the in-sample $\mathrm { P _ { e r r } = 2 4 / 2 4 }$ overstates how general it is.

## Answer matching is fragile.

The primary contain matcher credits answers that name the gold only to reject it, which inflates Mistral’s gold uptake in particular. PopQA is omitted because the matcher and output format failed. Answer labels come from the matcher, an LLM judge, and model reads; post hoc, two people labelled a blind 300-row V2 sample with Cohen’s κ=0.86 on 272 clear pairs (Online Resource 1, Sect. H). Replication and retrieval results are also reported under contain, judge, and model-read labels, and the Mistral results after a rejection audit (Online Resource 1, Sects. H and I). Under these alternative labels Llama’s empty-elicit arm reaches the 0.15 threshold (Section 6.1). Table 5 reports ECE for all four measures on one message type of one model×dataset cell. We do not report ECE as a grid-wide result, and we claim neither a win nor a loss against MTCal.

## 9 Conclusion

In short-answer question answering, an LLM’s confidence in a conversation is not one belief waiting for a better estimator. What the user supplies (an answer string) moves the answer. How the user phrases the turn (re-asking, agreeing, or accusing) does not. Four common confidence measures respond to diferent parts of the same turn, and on Qwen2.5-7B a frozen hidden-state readout follows the answer the model keeps rather than its truth. Errors that survive one re-ask are not undone by re-asking, resetting, or ofering abstention. A correct answer string repairs them, including one inside real retrieved evidence. For practitioners our data suggest a heuristic that still needs endto-end testing: before relying on evidence to correct a model, check that it contains the answer.

Supplementary information. Online Resource 1 (ESM 1.pdf) contains the supplementary appendices, Sects. A–I (prompt wording, additional and robustness controls, the Qwen3 analysis, both preregistered replications, retrieval, scoring, and the rejection audit). Online Resource 2 (ESM 2.zip) contains the three preregistration documents (PREREG.md, PREREG T3REP.md, PREREG TMLR.md) referred to in Section 3.8.

## Statements and Declarations

• Funding: No funding was received for conducting this study.

• Competing interests: The authors have no competing interests to declare that are relevant to the content of this article.

• Ethics approval and consent to participate: Not applicable. The study involves no human participants, human data, or animals; all questions and answers come from public question-answering datasets.

• Consent for publication: Not applicable.

• Data availability: The underlying QA datasets analysed in this study are publicly available: TriviaQA, Natural Questions (NQ-open), HotpotQA, TruthfulQA, WebQuestions, and GSM8K (see the references in Section 3). The generated dialogues, model answers, labels, and analysis outputs that support the findings of this study are available from the corresponding author on reasonable request (vinh.td225538@sis.hust.edu.vn).

• Materials availability: Not applicable.

• Code availability: The code used to run the experiments and produce the tables and figures, together with the preregistration documents and their timestamped freezing commits, is available from the corresponding author on reasonable request (vinh.td225538@sis.hust.edu.vn).

• Author contributions: Duc-Vinh Tran: conceptualization, methodology, software, investigation, formal analysis, writing (original draft and review and editing).

## References

[1] Zhang, C., Yang, R., Zhu, X., Li, C., Hu, T., Dong, Y.R., Yang, D., Collier, N.: Confidence estimation for LLMs in multi-turn interactions. In: Findings of the Association for Computational Linguistics: ACL 2026, pp. 25661–25676. Association for Computational Linguistics, San Diego, California, United States (2026). https://doi.org/10.18653/v1/2026.findings-acl.1280 . https://aclanthology.org/2026.findings-acl.1280/

[2] Zhang, Z., Li, C., Liu, X., Shen, C., Liu, Z., Patras, I.: Confidence should be calibrated more than one turn deep. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 38578–38594. Association for Computational Linguistics, San Diego, California, United States (2026). https://doi.org/10.18653/v1/2026.acl-long.1787 . https://aclanthology.org/2026.acl-long.1787/

[3] Kumaran, D.: Reported Confidence in LLMs Tracks Commitment More Than Correctness. arXiv preprint arXiv:2606.29490 (2026). https://arxiv.org/abs/ 2606.29490

[4] Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Mann, E., et al.: Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221 (2022)

[5] Miao, M.M., Ungar, L.: Closing the Confidence-Faithfulness Gap in Large Language Models. arXiv preprint arXiv:2603.25052 (2026). https://arxiv.org/abs/ 2603.25052

[6] Perez, E., Ringer, S., Lukoˇsi¯ut˙e, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., et al.: Discovering language model

behaviors with model-written evaluations. In: Findings of the Association for Computational Linguistics: ACL 2023 (2023)

[7] Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S.R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S.R., et al.: Towards understanding sycophancy in language models. In: International Conference on Learning Representations (ICLR) (2024)

[8] Wei, J., Huang, D., Lu, Y., Zhou, D., Le, Q.V.: Simple synthetic data reduces sycophancy in large language models. arXiv preprint arXiv:2308.03958 (2023)

[9] Laban, P., Murakhovs’ka, L., Xiong, C., Wu, C.-S.: Are you sure? challenging LLMs leads to performance drops in the FlipFlop experiment. arXiv preprint arXiv:2311.08596 (2023)

[10] Xie, Q., Wang, Z., Feng, Y., Xia, R.: Ask again, then fail: Large language models’ vacillations in judgment. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Bangkok, Thailand (2024). https: //doi.org/10.18653/v1/2024.acl-long.577

[11] Wang, B., Yue, X., Sun, H.: Can ChatGPT defend its belief in truth? evaluating LLM reasoning via debate. In: Findings of the Association for Computational Linguistics: EMNLP 2023 (2023)

[12] Huang, J., Chen, X., Mishra, S., Zheng, H.S., Yu, A.W., Song, X., Zhou, D.: Large language models cannot self-correct reasoning yet. In: International Conference on Learning Representations (ICLR) (2024)

[13] Laban, P., Hayashi, H., Zhou, Y., Neville, J.: LLMs get lost in multi-turn conversation. In: International Conference on Learning Representations (ICLR) (2026). https://arxiv.org/abs/2505.06120

[14] Lin, S., Hilton, J., Evans, O.: Teaching models to express their uncertainty in words. Transactions on Machine Learning Research (2022)

[15] Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., Hooi, B.: Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In: International Conference on Learning Representations (ICLR) (2024)

[16] Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., Manning, C.D.: Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2023)

[17] Geng, J., Cai, F., Wang, Y., Koeppl, H., Nakov, P., Gurevych, I.: A survey of

confidence estimation and calibration in large language models. In: Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) (2024)

[18] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 70, pp. 1321–1330. PMLR, Sydney, Australia (2017)

[19] Azaria, A., Mitchell, T.: The internal state of an LLM knows when it’s lying. In: Findings of the Association for Computational Linguistics: EMNLP 2023 (2023)

[20] Marks, S., Tegmark, M.: The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In: Conference on Language Modeling (COLM) (2024)

[21] Burns, C., Ye, H., Klein, D., Steinhardt, J.: Discovering latent knowledge in language models without supervision. In: International Conference on Learning Representations (ICLR) (2023)

[22] Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., Belinkov, Y.: LLMs know more than they show: On the intrinsic representation of LLM hallucinations. In: International Conference on Learning Representations (ICLR) (2025)

[23] Hewitt, J., Liang, P.: Designing and interpreting probes with control tasks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2019)

[24] Sarkar, R., Ramu, P., Rudinger, R.: Language Models Encode the Contextual Truth of Propositions. arXiv preprint arXiv:2608.03035 (2026). https://arxiv.org/ abs/2608.03035

[25] Longpre, S., Perisetla, K., Chen, A., Ramesh, N., DuBois, C., Singh, S.: Entitybased knowledge conflicts in question answering. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2021)

[26] Xie, J., Zhang, K., Chen, J., Lou, R., Su, Y.: Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In: International Conference on Learning Representations (ICLR) (2024)

[27] Joren, H., Zhang, J., Ferng, C.-S., Juan, D.-C., Taly, A., Rashtchian, C.: Suficient context: A new lens on retrieval augmented generation systems. In: International Conference on Learning Representations (ICLR) (2025)

[28] Lathkar, A.B.: Anchored Confabulation: Partial Evidence Non-Monotonically

Amplifies Confident Hallucination in LLMs. arXiv preprint arXiv:2604.25931 (2026). https://arxiv.org/abs/2604.25931

[29] Kuhn, L., Gal, Y., Farquhar, S.: Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In: International Conference on Learning Representations (ICLR) (2023)

[30] Joshi, M., Choi, E., Weld, D.S., Zettlemoyer, L.: TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In: Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL 2017) (2017)

[31] Qwen Team: Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 (2024)

[32] Qwen Team: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

[33] Grattafiori, A., et al.: The Llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)

[34] Jiang, A.Q., et al.: Mistral 7B. arXiv preprint arXiv:2310.06825 (2023)

[35] Abdin, M., et al.: Phi-4 technical report. arXiv preprint arXiv:2412.08905 (2024)

[36] Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al.: Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics 7, 452–466 (2019)

[37] Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., Manning, C.D.: HotpotQA: A dataset for diverse, explainable multi-hop question answering. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2018)

[38] Lin, S., Hilton, J., Evans, O.: TruthfulQA: Measuring how models mimic human falsehoods. In: Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL 2022) (2022)

[39] Berant, J., Chou, A., Frostig, R., Liang, P.: Semantic parsing on Freebase from question-answer pairs. In: Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP) (2013)

[40] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al.: Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 (2021)

[41] Tran, D.V.: Pilot Preregistration: Multi-turn Confidence Geometry. Frozen 2026- 09-19 09:43 UTC, before generation. H2 failed; H1 inconclusive; locked verdict CONTINUE. Supplementary file PREREG.md (Online Resource 2). (2026)

[42] Tran, D.V.: Preregistration: T3 Replication of Stay-Wrong Persistence. Frozen before generation, with two pre-generation amendments. Supplementary file PREREG T3REP.md (Online Resource 2). (2026)

[43] Tran, D.V.: Preregistration: Judge Rescoring, Larger T3 Samples, Retrieval with Real Evidence, and a 32B Model. Frozen before generation and judging, with five dated post-hoc notes. Supplementary file PREREG TMLR.md (Online Resource 2). (2026)

[44] Robertson, S., Zaragoza, H.: The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval 3(4), 333–389 (2009)