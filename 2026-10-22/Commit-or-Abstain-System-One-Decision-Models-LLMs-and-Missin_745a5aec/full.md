# Commit or Abstain? System One Decision Models, LLMs, and Missing Legal Authority

M. MIKAIL DEMIR, Department of Information Science, University at Albany, SUNY, USA

Given the enormous number of claims to be checked, verifiable legal AI requires scalable, cost-efective models that answer the practical needs of law. System One decision models are built on large language models but, instead of generating text, return a calibrated probability for each option. This suits the verification of legal authority, where the question is closed: a given holding, or none of them. We test whether such models abstain when the correct authority is missing. We extend CaseHOLD with an abstention condition in which the correct holding is replaced in 1,670 of 5,314 items, and compare four decision models with six generative LLMs. When no option is correct, generative LLMs still pick one in 38–86% of items; the most cautious decision model does so in 30%. On standard CaseHOLD, a decision model matches the best generative LLM (88.1% vs. 88.2%) at 1/55 of the cost. Accepting that LLM’s answer only when a decision model agrees halves its wrong citations, and distilling a decision model’s probabilities into Qwen3-8B, without labels, yields a well-calibrated local model. Legal AI should be judged by its wrong citations, not by accuracy alone.

Additional Key Words and Phrases: legal AI, CaseHOLD, calibration, abstention, decision models, knowledge distillation

“Resolve to be honest at all events; and if in your own judgment you cannot be an honest lawyer, resolve to be honest without being a lawyer.”

— Abraham Lincoln, Notes for a Law Lecture (c. 1850)

## 1 Introduction

Language models are now used across legal practice, and most visibly in legal research. Law has also become one of the best-documented domains of their failure. Since late 2022, courts have sanctioned lawyers and pro se litigants for filings that cite authorities that do not exist or do not support the propositions they are cited for [3, 33]. Each such filing adds work to an already overburdened court system. Someone has to check the citations, and courts must then hold hearings and decide on sanctions. The problem now reaches the bench, with judges’ own orders among the documented cases [5].

The problem is not limited to general-purpose chatbots. In a preregistered evaluation, commercial legal research tools built on retrieval-augmented generation, some marketed as hallucination-free, still hallucinated on 17% to 33% of queries [32]; general-purpose models did so at least 58% of the time [7]. Courts now see the same failure in practice. In October 2026, a federal court reviewed filings whose author admitted taking propositions from a Westlaw AI tool without confirming them [34]. Nor is the problem fading with newer models. Across eight generations of ChatGPT models, citation hallucination rates did not consistently decline, and GPT-5.1 produced hallucinated citations more often than the best mid-2024 GPT-4o release [31].

Institutions are responding. California requires its courts to adopt generative-AI use policies that address hallucinated output [22], and the State Bar of California has proposed amending its rule on candor to the tribunal so that lawyers must verify that cited authorities exist and are accurate [45]. Texas has proposed that signers of filings represent that their citations, quotations, and holdings are not fabricated [46]. Appellate courts now sanction unverified citations with fees, fines, and referrals to the bar [37, 52]. Yet the cases keep growing, and one public tracker lists 2,203 court matters involving AI hallucinations, 1,264 of them from 2026 alone [5]. Fabrication is only part of the problem. In 933 of these matters, a real authority was cited for a proposition it does not support [5].

![](images/905e199421e3cf7b2b0fe3371d8ee3f85d227519498642539643390e7be24026.jpg)

(b) How often models still pick a holding  
![](images/d2f7f4b2db0618f68423195d8464ff7efedaf3dbf20e85bde20a2e249101c7b4.jpg)  
Fig. 1. When the correct authority is missing, most models still cite one. (a) A CaseHOLD item whose true holding was replaced by a similar holding from another case (C); every generative LLM chose it. (b) Share of the 1,670 such items in which each model picked a holding instead of abstaining, against cost per 1,000 questions. System One models are the four cheapest and include the most cautious; among generative LLMs, more spending does not reliably buy caution.

Technical responses mostly add more language-model machinery: retrieval augmentation, better training data, model editing, decoding strategies [21], and agentic verification. These eforts work against a structural pressure. Kalai et al. [24] trace hallucinations to a simple classification problem, deciding whether a statement is valid. Models make errors on this problem, and the errors persist because training and evaluation reward guessing. Under accuracy-style scoring, a model that always guesses scores higher than one that admits it is unsure. In legal writing, the guess is the costly outcome, a confident citation to the wrong authority. Legal studies observe the same tendency. When told to stop if no factual basis existed, most models produced arguments anyway [56].

Most evaluations ask whether a model picks the right authority from a set of candidates, but in practice the right authority is often not among them. A useful system should then say so rather than choose the closest option. We therefore ask whether a model, when the correct authority is missing, commits to a wrong one or abstains. The two errors are not equally costly. A wrong commitment can reach a filing; an unnecessary abstention costs a lawyer an extra lookup. Accuracy treats them the same. What matters instead is whether a model’s confidence can tell a user which answers to check.

Legal NLP posed many of these problems as classification well before large language models. CaseHOLD asks which of five holdings a court gave for a cited case [57]; ContractNLI asks whether a contract entails, contradicts, or does not mention a hypothesis [26]; MAUD poses expert questions about merger agreements [51]. These tasks were addressed with fine-tuned encoders. Since large language models arrived, the same tasks are mostly posed to general-purpose generative models.

The underlying need has not changed. Lawyers and courts need to check, at scale, whether a specific authority supports a specific proposition, and to be told when it does not. This calls for precision and calibrated confidence more than open-ended reasoning. A verification model should decide only when the evidence is suficient [2], and small models trained for a narrow verification task can match GPT-4 at a fraction of its cost [48]. If more reasoning alone solved the problem, the strongest models would not continue to hallucinate [31]. In legal case-based reasoning, reasoning models even spend more computation on wrong answers than on right ones [55].

System One decision models revisit the classification framing with modern models. TypeSafe released the first of them, Jev, in September 2026 [49]. The name follows the fast, intuitive System 1 of Kahneman [23], and commentators soon began calling them decision models [53]. They are themselves built on large language models, but instead of generating text they return a typed decision and a probability for each option. By late September, an analysis of GitHub found 2,170 public projects using Jev [30], and a community index now tracks more than 170 decision models from many developers [35]. This design fits the task of checking an authority well. The question is closed (this holding, or none of them), so there is nothing to write, and the probability states how sure the model is, which is what a user needs in order to decide what to check. TypeSafe trains its Jev model with reinforcement learning for calibrated decisions (RLCD), which optimizes these probabilities against outcomes rather than human preferences [50]. Whether such models are more trustworthy than generative LLMs where it matters most in law, when the right authority is not available, has not been tested.

We examine this with CaseHOLD. We add a condition in which the correct holding is sometimes replaced by a similar holding from another case, and a sixth option states that none of the holdings applies. We compare four System One decision models with six generative LLMs on all 5,314 test items. We ask how often each model commits to a holding when none is correct, what its caution costs, and whether the two kinds of model can cover for each other.

Our contributions are:

• A test of abstention for legal authority. We extend CaseHOLD so that in 1,670 of its 5,314 test items the correct holding is replaced by a similar holding from another case, and measure how often a model cites a holding when none applies (Figure 1a).

• A comparison across model types and prices. We evaluate four System One decision models and six generative LLMs from five vendors, reporting accuracy, calibration, abstention, and cost (Figure 1b).

• A cost-aware view of combining them. We show how pairing a System One model with a generative LLM trades wrong citations against items sent for human review, and frame the choice of operating point by the unequal cost of the two errors.

• Local deployment and checks on what the results mean. Trained only on a System One model’s probabilities, with no labeled data, a local Qwen3-8B model learns to answer and abstain like its teacher and becomes well calibrated. Contamination tests extending Arvin [1] show that models know much of the case law but do not depend on it.

Code, data, and the outputs of every model are available at https://github.com/mmikaildemir commit-or-abstain.

## 2 Related Work

## 2.1 Legal Datasets with Verifiable Answers

Legal NLP has a long line of datasets whose answers can be checked against a source. Many of them come from contracts. ContractNLI asks whether a contract entails, contradicts, or says nothing about a hypothesis [26], CUAD marks the clauses lawyers look for in contract review [20], and MAUD asks expert questions about merger agreements [51]. COLIEE has run shared tasks on retrieving and entailing statutes and case law for years [17]. In CaseHOLD, the dataset we use, a model picks which of five holdings a court gave for a cited case [57]; Section 3 describes it in detail. LexGLUE [4] and LegalBench [19] later gathered tasks like these into suites for evaluating language models, and LawBench did the same for Chinese law [14]. In almost all of them, the correct answer is among the candidates and every error counts the same. Neither is true when a lawyer checks a citation.

## 2.2 Abstention and Unanswerable Questions

Outside law, SQuAD 2.0 added questions that look answerable but are not, so that a system had to learn when to stay silent [40]. Later work asked the same of language models with multiple-choice questions. Models lose much of their accuracy when none ofthe above is the right option [47], and they keep choosing an option even when every option is wrong [18]. Across twenty datasets, AbstentionBench even finds that reasoning fine-tuning can make abstention worse [25]. Kalai et al. [24] argue that this follows from how models are graded, since a guess is rewarded over an admission of uncertainty. Our abstain condition brings this test to legal authority, where a confident guess becomes a citation to the wrong holding.

## 2.3 System One Decision Models

System One models answer typed questions with a probability for each option instead of generated text [50]. Jev, the first of them, is trained with RLCD, and its architecture and training remain proprietary. Adoption has been fast. An analysis of 2,170 public projects finds Jev used mainly for judging attributes and scoring, as a component inside larger workflows [30]. A community index now tracks more than 170 decision models from many developers [35].

Open work has taken two routes to the same behavior. One trains for it directly, implementing Jev’s objective with strictly proper scoring rules [15]. The other reads it out of existing language models, whose next-token probabilities over the option letters already behave as a decision model, and light fine-tuning closes most of the remaining gap [16, 29]. A third recipe distills a model’s own post-reasoning answer into a single-pass readout [13]. Open-weight models such as Laya, a 421M-parameter encoder, now exist alongside Jev [36].

Independent evaluations have appeared only in recent weeks. Jev is competitive with frontier generative LLMs on knowledge and commonsense benchmarks but falls behind on multi-step arithmetic [11, 28]. Decision models do best when the answer can be resolved from the evidence given, but often overstate the probability of their most likely answer [13], and their calibration shifts with the wording of a question [44]. Two studies bear directly on abstention. In medical question answering, most evaluated models struggle to recognize when the correct answer is missing [12]. Jev’s confidence can also stay high when no answer-relevant information is given at all [43].

Decision models are also combined with other models. They serve as gates in front of generative LLMs [27], and escalating their uncertain cases to a generative LLM can match the larger model’s accuracy at close to the decision model’s cost [39], in the line of earlier cascades and routers [6, 38]. Jev’s judgments have also been distilled into small task-specific evaluators for chess and passage reranking [41], though not, to our knowledge, into a general-purpose generative LLM.

In law, Zhang et al. [54] compare Jev with nine generative LLMs on ContractNLI and find it the cheapest and fastest, though less accurate than hosted models. No prior work evaluates decision models on choosing legal authority, or combines them with generative LLMs to limit wrong citations.

## 3 Method

Figure 2 gives an overview of the study.

![](images/2fd927dc3d79a06497daec49338fec44169aa7a319b0eb53884e26a813c5cf9c.jpg)  
Fig. 2. Overview of the study. CaseHOLD test items are presented in two conditions. In the abstain condition, the correct holding is replaced in 1,670 of 5,314 items, and a sixth option, none of these, becomes correct. System One decision models return a probability for each option, and generative LLMs return one answer. Their outputs are scored, combined through three policies, and used to train a local model.

## 3.1 Task and Data

CaseHOLD is built from U.S. court opinions in the Harvard Law Library case law corpus [57]. Courts often follow a citation with a parenthetical stating the cited case’s holding, for example (holding that consideration exists where a customer must visit the store). CaseHOLD replaces this parenthetical with the token <HOLDING>. Each item consists of a short excerpt from the citing opinion and five candidate holdings. One is the holding the court wrote, and the other four are holdings from other citations with the most similar wording. We use the full test split of 5,314 items.

## 3.2 Two Conditions

The original condition is CaseHOLD as published, with five options and one correct answer.

The abstain condition adds a sixth option, None ofthe listed holdings is the one the cited case is being cited for in this passage. In a random 31% of items (1,670), the correct holding is replaced by another real holding, and the sixth option becomes the correct answer. The replacement is selected with CaseHOLD’s procedure for wrong options. It is the holding from another test item with the highest TF-IDF cosine similarity to the removed holding, subject to CaseHOLD’s similarity cap of 0.75. Replacements have a mean similarity of 0.28 to the removed holding, compared with 0.30 for CaseHOLD’s own wrong options. In the remaining 3,644 items the correct holding is kept, and the sixth option is wrong.

## 3.3 Models

We compare two kinds of model, both built on large language models, that difer in their output (Table 1). System One decision models receive the passage as state and a single choice question whose options are the holdings, and they return a probability for each option. We evaluate Jev [50], Perplexity’s pplx-decider, Liquid AI’s d1-ultra, and OpenAI’s GPT-6 Luna Decisions through Open-Router’s decisions interface. Generative LLMs receive the same instruction, passage, and options as a chat message and return the key of one option. The six generative LLMs come from five vendors and cover a range of prices. They include GPT-4o-mini, which Arvin [1] evaluated on CaseHOLD, and GLM 5.3 Flash, whose price per question is close to Jev’s.

Table 1. Models evaluated. Cost is the amount billed per 1,000 questions in the original condition; batchprocessed models were billed at batch rates.
<table><tr><td>Model</td><td>Type</td><td>Provider</td><td>Mode</td><td>$/ 1,000</td></tr><tr><td>Jev 1.14</td><td>System One</td><td>TypeSafe</td><td>sync</td><td>0.035</td></tr><tr><td>pplx-decider v1.2</td><td>System One</td><td>Perplexity</td><td>sync</td><td>0.012</td></tr><tr><td>d1-ultra</td><td>System One</td><td>Liquid AI</td><td>sync</td><td>0.021</td></tr><tr><td>GPT-6 Luna Decisions</td><td>System One</td><td>OpenAI</td><td>sync</td><td>0.058</td></tr><tr><td>Gemini 3.8 Flash</td><td>Generative LLM</td><td>Google</td><td>batch</td><td>0.659</td></tr><tr><td>Claude Haiku 5.5</td><td>Generative LLM</td><td>Anthropic</td><td>batch</td><td>0.066</td></tr><tr><td>GPT-6 Luna</td><td>Generative LLM</td><td>OpenAI</td><td>batch</td><td>0.077</td></tr><tr><td>Mistral Small 2603</td><td>Generative LLM</td><td>Mistral AI</td><td>batch</td><td>0.234</td></tr><tr><td>GLM 5.3 Flash</td><td>Generative LLM</td><td>Z.ai</td><td>sync</td><td>0.065</td></tr><tr><td>GPT-4o-mini</td><td>Generative LLM</td><td>OpenAI</td><td>sync</td><td>0.084</td></tr></table>

## 3.4 Prompts and Setings

All models receive the same one-sentence instruction, which states that a court opinion cites a case, that the parenthetical describing its holding has been replaced by <HOLDING>, and that the task is to identify the holding the citing court gave. Decision models receive the passage as state and the holdings as the options of a choice question. Generative LLMs receive the passage and lettered options in one message and reply with a JSON object containing only the option key. This format is enforced with structured output, since some models otherwise produced an explanation first. Generative LLMs are not asked for a confidence score; Appendix B reports a pilot with stated confidence.

Generative LLMs run at temperature 0 with the lowest reasoning setting each model ofers above none; GPT-4o-mini has no reasoning setting. We record reasoning tokens and the billed cost of every call.<sup>1</sup> The full prompts are in Appendix A.

## 3.5 Metrics

We report accuracy and macro F1 in both conditions. In the abstain condition, two error rates are reported separately. The wrong-pick rate is the share of items without a correct holding in which the model picked a holding. The false-abstention rate is the share of items with a correct holding in which the model chose the none option. For decision models, we also report expected calibration error over ten bins and the AUROC of the none probability for separating items with and without a correct holding. Confidence intervals are 95% percentile bootstrap intervals with 2,000 resamples. Paired comparisons between models use exact McNemar tests on the same items.

## 3.6 Combining Decision Models and Generative LLMs

We evaluate three policies that combine a decision model with a generative LLM, all on the abstain condition. In the veto policy, the generative LLM’s answer is accepted unless the decision model’s probability of none exceeds a threshold. In the agreement gate, an answer is accepted only when both models choose the same option, and other items are sent for review. This follows the observation that hard items tend to defeat some models but not others [9]. In System One first, the decision model answers when its top probability exceeds a threshold, and the generative LLM is called only for the remaining items, which reduces cost.

Each item ends as a correct answer, a wrong citation, or a review, meaning an item flagged for a person although the correct holding was present. We score a policy by � · wrong citations + reviews, where � is the cost of a wrong citation relative to an unnecessary review. We know of no empirical estimate of � for legal research. The values � = 1, 5, 10 are therefore illustrative, and Figure 4 shows the full trade-of. Thresholds are tuned on a random 30% of items, and results are reported on the remaining 70% (3,720 items).

## 3.7 Contamination Checks

Since CaseHOLD is built from public opinions, a model may recognize a cited case instead of reasoning from the passage. Following Arvin [1], we test this with three variants of each passage, one with case names and citations anonymized, one showing only the cited case’s citation, and one with that citation removed. Appendix C describes the procedure and results.

## 3.8 Distillation

We test whether a decision model’s probabilities can train a model that runs locally. Qwen3-8B is fine-tuned with LoRA on 10,000 items from CaseHOLD’s training split in the abstain condition. We compare three versions: the base model, a model trained only to match Jev’s probabilities, and a model trained on gold labels. The first two use no labeled data. Each model’s answer distribution is read from its next-token probabilities over the option letters [16, 29], so that its calibration can be measured in the same way as a decision model’s. Earlier work distilled Jev into task-specific evaluators [41]; here the student is a general-purpose generative LLM. Each version was trained once on a single NVIDIA RTX A6000 (48 GB). A training run took about 90 minutes, and evaluation on the full test split about 13 minutes. Training settings are in Appendix D.

## 4 Results

## 4.1 Accuracy and Cost

On the original task, the best decision model and the best generative LLM are tied. pplx-decider reaches 88.1% and Gemini 3.8 Flash 88.2% (diference −0.2 points, 95% CI $[ - 1 . 2 , 0 . 8 ] , p = 0 . 7 4 )$ , at about 1/55 of the cost (Table 2). Three of the four decision models are more accurate than every generative LLM except Gemini. Jev, the decision model whose price is closest to GLM 5.3 Flash, is 3.7 points more accurate $( [ 2 . 6 , 4 . 8 ] , p < 1 0 ^ { - 1 0 } )$ . Macro F1 tracks accuracy within 0.002 for every model. All models lose accuracy in the abstain condition, by 9.9 to 18.7 points.

## 4.2 Commiting When No Holding Is Correct

When the correct holding is missing, most models still pick one (Figure 1). Among generative LLMs, the wrong-pick rate ranges from 37.8% for Gemini 3.8 Flash to 85.6% for GPT-4o-mini, the model Arvin [1] evaluated on CaseHOLD. Among decision models, it ranges from 30.1% for Jev to 60.8% for GPT-6 Luna Decisions. Jev picks a wrong holding 7.7 points less often than Gemini $( [ - 1 0 . 0 , - 5 . 3 ] , p < 1 0 ^ { - 9 } )$ and 31.5 points less often than GLM 5.3 Flash, the generative LLM closest to its price $( [ - 3 4 . 0 , - 2 9 . 1 ] , p < 1 0 ^ { - 1 0 0 } )$

Caution has a price. Jev abstains on 26.2% of items that do have a correct holding, while every generative LLM does so on fewer than 8%. The probabilities of the decision models separate the two kinds of item well (AUROC 0.73 to 0.89), so a user can set a threshold on them.

Table 2. Results on all 5,314 CaseHOLD test items. Wrong pick: share of the 1,670 abstain-condition items without a correct holding in which the model picked a holding (95% bootstrap CI). False abstention: share of the 3,644 items with a correct holding in which the model abstained. AUROC of the probability of none is available only for models that return probabilities. 95% CIs for accuracy are within ±1.4 points.
<table><tr><td rowspan="2">Model</td><td colspan="2">Accuracy (%)</td><td rowspan="2"></td><td colspan="3"></td><td rowspan="2">$ / 1,000</td></tr><tr><td>Original</td><td>Abstain</td><td>Wrong pick (%)</td><td>False abst. (%)</td><td>AUROC</td></tr><tr><td>System One decision models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.14</td><td>78.3</td><td></td><td>63.9</td><td>30.1 [27.8, 32.3]</td><td>26.2</td><td></td><td>0.784</td><td>0.035</td></tr><tr><td>pplx-decider v1.2</td><td>88.1</td><td></td><td>73.1</td><td>56.5 [54.2, 58.9]</td><td>1.8</td><td></td><td>0.887</td><td>0.012</td></tr><tr><td>d1-ultra</td><td>84.9</td><td></td><td>73.6</td><td>41.0 [38.6, 43.3]</td><td>9.5</td><td></td><td>0.856</td><td>0.021</td></tr><tr><td>GPT-6 Luna Decisions</td><td>72.2</td><td></td><td>58.2</td><td>60.8 [58.4, 63.0]</td><td>9.5</td><td></td><td>0.726</td><td>0.058</td></tr><tr><td>Generative LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.8 Flash</td><td>88.2</td><td></td><td>78.4</td><td>37.8 [35.5, 40.1]</td><td>4.2</td><td></td><td></td><td>0.659</td></tr><tr><td>Claude Haiku 5.5</td><td>75.2</td><td></td><td>56.5</td><td>81.1 [79.1, 82.9]</td><td>2.6</td><td></td><td></td><td>0.066</td></tr><tr><td>GPT-6 Luna</td><td>77.6</td><td></td><td>62.1</td><td>67.1 [64.7, 69.5]</td><td>2.8</td><td></td><td></td><td>0.077</td></tr><tr><td>Mistral Small 2603</td><td>70.9</td><td></td><td>57.0</td><td>65.1 [62.8, 67.4]</td><td></td><td>7.7</td><td></td><td>0.234</td></tr><tr><td>GLM 5.3 Flash</td><td>74.6</td><td></td><td>60.8</td><td>61.6 [59.2, 64.0]</td><td></td><td>7.0</td><td></td><td>0.065</td></tr><tr><td>GPT-4o-mini</td><td>67.4</td><td></td><td>50.1</td><td>85.6 [83.9, 87.3]</td><td></td><td>2.1</td><td></td><td>0.084</td></tr></table>

![](images/a68ce453022d8f2a820b2f7e126913a24e77c20a0ccf1a537c82eb73fdf3dd22.jpg)  
Fig. 3. Abstention as a threshold on the probability of none, on all 5,314 abstain-condition items. Each curve traces one System One model as the threshold moves; AUROC in parentheses. Triangles mark each decision model’s own answer, and circles mark the generative LLMs, which ofer a single operating point. The x-axis is cut at 50%.

Figure 3 shows the choices such a threshold ofers. A generative LLM gives a single operating point, while a decision model can be moved along its curve. At the false-abstention rate of every generative LLM except Gemini 3.8 Flash, pplx-decider and d1-ultra both recognize more of the items without a correct holding. Gemini lies above all four curves. At its false-abstention rate of 4.2%, pplx-decider recognizes 57.7% of these items, against 62.2% for Gemini.

![](images/06fbd5a918002f2b7b42e689ce952c93f4db23d5f624116959fa26e17135010c.jpg)  
Fig. 4. Wrong citations against unnecessary reviews on the abstain condition (held-out 70% of items, � = 3,720). Each point is a model alone. The curves show the veto policy as its threshold varies; diamonds show the agreement gate. Moving down a curve buys fewer wrong citations with more review.

## 4.3 Combining Decision Models and Generative LLMs

Combining the two kinds of model lowers wrong citations, but sends more items to a person for review (Figure 4). The veto and System One first policies need a threshold, which we chose on a random 30% of items (Section 3). To keep that choice from flattering the results, every number in this section comes from the remaining 70% (3,720 items), which played no part in choosing it.

Take Gemini 3.8 Flash as an example. Alone, it cites a wrong holding on 18.4% of items. With the agreement gate, we accept Gemini’s answer only when pplx-decider gives the same answer. This cuts wrong citations by half, to 9.4%. The price is review: 12.7% of items now go to a person even though the correct holding was there, against 3.0% for Gemini alone. The share of correct answers barely changes (78.0% against 78.5%), and the extra cost is about 1%.

Cheaper pairs also help. Gating GLM 5.3 Flash with pplx-decider gives fewer wrong citations than Gemini alone (12.8%), at about one-ninth of Gemini’s cost.

Which policy is best depends on how much worse a wrong citation is than an unnecessary review. If the two count the same (� = 1), a decision model alone is nearly as good as anything else. d1-ultra makes about as many wrong citations as Gemini (19.5%) at 1/30 of the cost. If a wrong citation counts five times as much (� = 5), a combined policy is best for every pair we tested. At � = 10, the best policy for most pairs sends more than half of the items to review, and wrong citations fall to about 2% or less.

## 4.4 Contamination

Models know much of the case law but do not depend on it. The strongest models pick the right holding from the citation alone well above chance, yet anonymizing case names and citations costs only 0.5 to 5.7 points. Appendix C gives the full results.

Table 3. Qwen3-8B on the abstain condition (all 5,314 test items) before fine-tuning, after fine-tuning on Jev’s probabilities alone, and after fine-tuning on gold labels. Each version was trained once on 10,000 training items.
<table><tr><td>Model</td><td>Accuracy (%)</td><td>Wrong pick (%)</td><td>False abst. (%)</td><td>AUROC</td><td>ECE</td></tr><tr><td>Qwen3-8B, no fine-tuning</td><td>43.6</td><td>21.8</td><td>58.1</td><td>0.658</td><td>0.504</td></tr><tr><td>Qwen3-8B, Jev&#x27;s probabilities</td><td>63.8</td><td>31.2</td><td>24.6</td><td>0.777</td><td>0.025</td></tr><tr><td>Qwen3-8B, gold labels</td><td>74.9</td><td>38.1</td><td>7.2</td><td>0.889</td><td>0.023</td></tr><tr><td>Jev 1.14 (teacher)</td><td>63.9</td><td>30.1</td><td>26.2</td><td>0.784</td><td>0.056</td></tr></table>

## 4.5 Distillation

This experiment asks whether a decision model’s probabilities can be used to train a model that a firm or court could run on its own hardware. We fine-tuned Qwen3-8B in two ways, once to match Jev’s probabilities and once on the gold labels, and compare both with the model before fine-tuning (Table 3). The first version never sees a gold label. All three are evaluated on the full abstain condition.

Before fine-tuning, Qwen3-8B is not usable for this task. It answers none on most items, whether or not a correct holding is present, and its probabilities are far from calibrated (ECE 0.504).

Trained only on Jev’s probabilities, Qwen3-8B behaves much like Jev. Its accuracy is 63.8%, against Jev’s 63.9%. It picks a wrong holding and abstains wrongly about as often as Jev, and its AUROC is 0.777, against Jev’s 0.784. Its calibration error falls to 0.025, which is lower than Jev’s own.

Gold labels still give a more accurate model, at 74.9%. Most of that gain comes from abstaining less often when a correct holding is present (7.2% against 24.6%), and the gold-trained model picks a wrong holding more often (38.1%). Measured by accuracy, Jev’s probabilities close about two-thirds of the gap between the untrained model and the gold-trained one, without any labeled data.

## 5 Discussion

## 5.1 Two Ways to Fail

The two kinds of model fail in opposite directions. Most generative LLMs rarely abstain and often pick a wrong holding. Jev abstains often and picks a wrong holding least. We think this follows from how each is trained. Generative LLMs are tuned and graded on tasks where answering is rewarded and abstaining is not [24], and more reasoning does not reliably help them abstain [25]. Decision models are trained to return probabilities that match outcomes. That gives a user something to act on. Their probability of none separates items with and without a correct holding (AUROC 0.73 to 0.89).

We do not read this as a verdict for one kind of model. Jev’s caution means it abstains on about one in four items that do have a correct holding, and a decision model’s confidence is not always a reliable sign of missing knowledge [43].

## 5.2 Accuracy Is the Wrong Scoreboard for Legal Authority

A wrong citation and an unnecessary review are not equal errors. The first can reach a filing and end in sanctions [37, 52]. The second costs a lawyer the time it takes to look something up. Accuracy counts them the same, which is how a model that guesses can outscore one that abstains. Our earlier work made the same point for precedent treatment, where some misclassifications matter far more than others [8].

How much worse a wrong citation is depends on the user, the matter, and the court. A pro se litigant drafting a first motion and a firm filing an appellate brief do not face the same stakes. We therefore report the full trade-of rather than one score. Choosing an operating point is a decision about acceptable risk, and it belongs to the lawyer or firm using the tool, not to its vendor. That choice can only be made well if evaluations of legal AI tools report, next to accuracy, how often a tool cites an authority when none applies and how often it declines when one does.

## 5.3 A Cheap Check, Not a Substitute

A decision model costs a small fraction of a generative LLM call, so it can be asked for a second opinion on every answer. Accepting a generative LLM’s answer only when a decision model agrees halved wrong citations in our test while keeping nearly all correct answers. This agrees with the observation that hard verification items tend to defeat some models but not others [9], so disagreement is informative.

None of this removes the lawyer’s duty to read what they cite. Courts have been clear that every cited authority must be read and verified, whatever its source [37]. What a combined system can do is point that reading at the answers most likely to be wrong.

Knowing the case law is also not the same as respecting what the passage supports. Shown only a case’s citation, with none of the surrounding passage, the strongest models still picked its holding in about three of four items, well above the one-in-five chance rate (Appendix C). They have learned much of the case law itself. But legal authority depends on what a court actually held in the opinion at hand. In verification, prior knowledge can even override the provided text [9]. Our abstain condition rewards a model only for recognizing that the holding in front of it is not the one the court relied on.

## 5.4 Running the Check Locally

Many firms cannot send client material to external services, which rules out hosted models for much of their work [10]. The distillation result suggests one way forward. Trained on Jev’s probabilities alone, with no labeled data, a local Qwen3-8B model reproduced Jev’s accuracy and abstention behavior and became well calibrated, after 90 minutes of training on one workstation GPU. The teacher’s probabilities can be collected on public material, such as published opinions, and the student can then run on the firm’s own hardware. The student also inherits the teacher’s mistakes, so a firm should test it on its own work before relying on it.

## 6 Limitations

Contamination. CaseHOLD is built from public opinions decided before 2021, and the training data of the closed models we test is not disclosed. Our perturbation checks show that models know much of the case law but do not depend on recognizing the case; they cannot rule out that the models saw these passages in training.

A constructed absence. In our abstain condition, the correct holding is replaced by a real holding from another case. In practice, authority goes missing in other ways as well: a case may have been overruled, may come from the wrong jurisdiction, or may simply not exist. Our condition measures one controlled form of absence, not all of them.

The cost of errors. The values of � we report are illustrative. We know of no empirical estimate of how much worse a wrong citation is than an unnecessary review, and the right value will difer across practice settings.

Scope. We test one task, United States case law in English, and one prompt per kind of model. Generative LLMs may behave diferently under other prompts, reasoning settings, or retrieval, and the models we test will change; our results describe versions available in October 2026. Each distillation configuration was trained once, on 10,000 items, so its results carry more uncertainty than those of the hosted models.

## 7 Conclusion

We asked what generative LLMs and System One decision models do when the authority a court relied on is not among the candidates. Most generative LLMs still pick a holding, some in more than four of five such cases, while the most cautious decision model does so in fewer than one in three, at a fraction of the cost. Neither kind of model is reliable on its own, but pairing them halves the wrong citations of the strongest generative LLM we tested, and a decision model’s probabilities can teach a local model to behave the same way without labeled data.

What calibrated probabilities ofer is not correctness but a dial: the user can decide how cautious the system should be. Lincoln’s advice to young lawyers was to be honest before anything else. For a research tool, honesty starts with saying when it cannot find the authority.

## References

[1] Chuck Arvin. 2025. Identifying Legal Holdings with LLMs: A Systematic Study ofPerformance, Scale, and Memorization. In Proceedings ofthe Twentieth International Conference on Artificial Intelligence and Law (ICAIL ’25). doi:10.1145 3769126.3769128 arXiv:2505.02172.

[2] Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2022. Fact Checking with Insuficient Evidence. Transactions ofthe Association for Computational Linguistics 10 (2022), 746–763. doi:10.1162/tacl\_a\_00486

[3] Billups 2025. Billups v. Louisville Municipal School District. No. 24-CV-74, 2025 WL 3691871 (N.D. Miss. Dec. 19, 2025).

[4] Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, and Nikolaos Aletras. 2022. LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. In Proceedings of the 60th Annual Meeting ofthe Association for Computational Linguistics (ACL).

[5] Damien Charlotin. 2026. AI Hallucination Cases Database. https://www.damiencharlotin.com/hallucinations/. Accessed October 19, 2026; 2,203 cases

[6] Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research (2024). arXiv:2305.05176.

[7] Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E. Ho. 2024. Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models. Journal ofLegal Analysis 16, 1 (2024), 64–93.

[8] M. Mikail Demir and M. Abdullah Canbaz. 2025. Validate Your Authority: Benchmarking LLMs on Multi-Label Precedent Treatment Classification. In Proceedings ofthe Natural Legal Language Processing Workshop 2025. 172–183. doi:10.18653/v1/2025.nllp-1.13

[9] M. Mikail Demir and M. Abdullah Canbaz. 2026. When Citations Mislead? A Claim-Level Benchmark for Legal Hallucination Detection. In Proceedings ofthe Twenty-First International Conference on Artificial Intelligence and Law (ICAIL ’26). doi:10.1145/3836937.3836948

[10] M. Mikail Demir, Hakan T. Otal, and M. Abdullah Canbaz. 2025. LegalGuardian: A Privacy-Preserving Framework for Secure Integration of Large Language Models in Legal Practice. arXiv:2501.10915.

[11] Tobias Deußer, Lorenz Sparrenberg, and Rafet Sifa. 2026. Evaluating and Benchmarking the System One Model Jev. arXiv:2609.37647.

[12] Yanrui Du, Danyang Zhao, Sendong Zhao, Wenhao Li, et al. 2026. How Reliable Are System-One Decisions in Medicine? A Comparative Study of Jev, Jev-like Models, and LLMs. arXiv:2610.18529.

[13] Feiyu Duan, Jiayu Lin, Jia Wang, Jun Xiang, Jialiang Wu, et al. 2026. General Decision Models: Benchmarking and Insights Beyond Jev. arXiv:2610.03935.

[14] Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent Ng. 2024. LawBench: Benchmarking Legal Knowledge of Large Language Models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

[15] Zhimin Gao and Pichao Wang. 2026. OpenJev-RLCD: A Working RLCD Implementation. arXiv:2609.38850.

[16] Jonathan Geuter, Andrew Cheng, and David Alvarez-Melis. 2026. Releasing the Inner Jev: A Simple Recipe for Open Decision Models. arXiv:2610.17205.

[17] Randy Goebel, Yoshinobu Kano, Mi-Young Kim, Juliano Rabelo, Ken Satoh, and Masaharu Yoshioka. 2024. Overview and Discussion of the Competition on Legal Information, Extraction/Entailment (COLIEE) 2023. The Review of

Socionetwork Strategies 18, 1 (2024), 27–47. doi:10.1007/s12626-023-00152-0

[18] Gracjan Góral, Emilia Wiśnios, Piotr Sankowski, and Paweł Budzianowski. 2025. Wait, that’s not an option: LLMs Robustness with Incorrect Multiple-Choice Options. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (ACL). arXiv:2409.00113.

[19] Neel Guha et al. 2023. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track.

[20] Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. 2021. CUAD: An Expert-Annotated NLP Dataset for Lega Contract Review. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track.

[21] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Transactions on Information Systems 43, 2 (2025), 1–55. doi:10.1145 3703155

[22] Judicial Council of California. 2025. California Rules of Court, Rule 10.430: Generative Artificial Intelligence Use Policies. https://courts.ca.gov/cms/rules/index.cfm?title=ten&linkid=rule10\_430. Efective September 1, 2025.

[23] Daniel Kahneman. 2011. Thinking, Fast and Slow. Farrar, Straus and Giroux

[24] Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, and Edwin Zhang. 2025. Why Language Models Hallucinate. arXiv:2509.04664.

[25] Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. arXiv:2506.09038.

[26] Yuta Koreeda and Christopher D. Manning. 2021. ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts. In Findings ofthe Association for Computational Linguistics: EMNLP 2021. Association for Computational Linguistics, 1907–1919.

[27] Jiawei Li. 2026. Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses. arXiv:2610.02267.

[28] Xing Li, Qingcheng Chang, Jinzhong Ning, Changfeng Xu, Shenlong Zhang, et al. 2026. Specialized Decision Models vs. General-Purpose LLMs: Benchmarking Jev Across Knowledge, Reasoning, and Multilingual Tasks. arXiv:2610.11978.

[29] Yinheng Li and Justin Wagle. 2026. LLM-as-Jev: LLMs Are Already Jev-Style Decision Models – When and How to Fine-Tune Them. arXiv:2610.02076

[30] Guoming Ling, Muen Xue, and Zijian Ye. 2026. Jev in the Wild: A Data-Driven Analysis of the Jev Model’s Functionality, Applications and Ecosystem. arXiv:2609.30216.

[31] Patty Liu, Dominik Stammbach, and Peter Henderson. 2026. Who Checks the Citations? Benchmarking Legal Hallucination Detection. arXiv:2606.21155.

[32] Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E. Ho. 2025. Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools. Journal ofEmpirical Legal Studies 22, 2 (2025), 216–242. doi:10.1111/jels.12413

[33] Mata 2023. Mata v. Avianca, Inc. 678 F. Supp. 3d 443 (S.D.N.Y. 2023).

[34] Mercante 2026. Mercante v. Tarzia. No. 24-CV-8471 (MKB), Mem. & Order, ECF No. 114 (E.D.N.Y. Oct. 2, 2026).

[35] multimodalart and contributors. 2026. Decision Index 0.3.2: Community Leaderboard of System One Decision Models. https://huggingface.co/spaces/multimodalart/jev-decision-index.

[36] Gowthamkumar Nandakishore. 2026. Laya as a Typed Probabilistic Assessor: An Independent Reproduction and a Preregistered Study of Calibration and Selective Escalation. arXiv:2609.33843.

[37] Noland 2025. Noland v. Land of the Free, L.P. 336 Cal. Rptr. 3d 897 (Cal. Ct. App. 2025).

[38] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to Route LLMs from Preference Data. In International Conference on Learning Representations (ICLR).

[39] Amir Rafe and Subasish Das. 2026. Benchmarking System One Decision Models against Trained Classifiers and Language Models for Automated Decision Gates. arXiv:2610.00346.

[40] Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know What You Don’t Know: Unanswerable Questions for SQuAD. In Proceedings ofthe 56th Annual Meeting ofthe Association for Computational Linguistics (ACL).

[41] Mohamad Yazan Sadoun, Sarah Sharif, and Yaser Mike Banad. 2026. From Probabilities to Decisions: Search and Multi-Teacher Distillation with Jev. arXiv:2610.09188

[42] Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. 2023. NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark. In Findings ofthe Association for Computational Linguistics: EMNLP 2023. arXiv:2310.18018.

[43] Sharath M. Shankaranarayana, Davor Runje, and Jan Jannink. 2026. Beyond Answer Confidence: A Controlled Audit of Self-Knowledge in a Black-Box Decision Model. arXiv:2610.01006

"state": {"citing\_passage": "<citing\_prompt>"},   
"questions": {"holding": {   
"type": "choice",   
"instructions": "<instruction>",   
"criteria": {"A": "<holding\_0>", "B": "<holding\_1>",   
"C": "<holding\_2>", "D": "<holding\_3>",

[44] Rahul Sharma, Andrew B. Duncan, Gaétan Marceau Caron, and Sebastian J. Vollmer. 2026. TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models. arXiv:2610.11392.

[45] State Bar of California. 2026. Proposed Amendments to the Rules of Professional Conduct Related to Artificial Intelligence. https://www.calbar.ca.gov/public-comment/proposed-amendments-rules-professional-conduct-relatedartificial-intelligence. Public comment closed August 6, 2026; not yet adopted.

[46] Supreme Court of Texas. 2026. Preliminary Approval of Amendments to Texas Rule of Civil Procedure 13 and Texas Rules of Appellate Procedure 9 and 52. Misc. Docket No. 26-9054, https://www.txcourts.gov/media/1462936/269054.pdf. June 26, 2026.

[47] Zhi Rui Tam, Cheng-Kuang Wu, Chieh-Yen Lin, and Yun-Nung Chen. 2025. None of the Above, Less of the Right: Parallel Patterns between Humans and LLMs on Multi-Choice Questions Answering. In Findings ofthe Association for Computational Linguistics: ACL 2025. arXiv:2503.01550.

[48] Liyan Tang, Philippe Laban, and Greg Durrett. 2024. MiniCheck: Eficient Fact-Checking of LLMs on Grounding Documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 8818–8847.

[49] TypeSafe. 2026. Introducing System One Models & Jev. https://typesafe.ai/blog/introducing-system-one-models-andjev. Accessed October 21, 2026.

[50] TypeSafe. 2026. System One Models and Machine Learning Primer. https://docs.typesafe.ai/concepts/system-one. Accessed October 19, 2026.

[51] Steven H. Wang, Antoine Scardigli, Leonard Tang, Wei Chen, Dimitry Levkin, Anya Chen, Spencer Ball, Thomas Woodside, Oliver Zhang, and Dan Hendrycks. 2023. MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 16369–16382.

[52] Whiting 2026. Whiting v. City of Athens. Nos. 24-5918/5919, 25-5424 (6th Cir. Mar. 13, 2026).

[53] Simon Willison. 2026. Jev Introduces a New Shape of LLM—System One, aka Decision Models. https://simonwillison. net/2026/Sep/21/jev/. Accessed October 21, 2026.

[54] Fan Zhang, Yankai Chen, Zhuohan Xie, Yixi Zhou, et al. 2026. Same Scores, Diferent Decisions: Evaluating Jev and Language Models for Legal Document Understanding. arXiv:2609.27678.

[55] Li Zhang, Matthias Grabmair, Morgan Gray, and Kevin Ashley. 2026. Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning. In Proceedings of the Symposium on Computer Science and Law (CSLAW ’26). ACM, 71–85. doi:10.1145/3788646.3789522

[56] Li Zhang, Morgan Gray, Jaromir Savelka, and Kevin D. Ashley. 2025. Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments. arXiv:2506.00694.

[57] Lucia Zheng, Neel Guha, Brandon R. Anderson, Peter Henderson, and Daniel E. Ho. 2021. When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings. In Proceedings ofthe Eighteenth International Conference on Artificial Intelligence and Law (ICAIL ’21). ACM, 159–168. doi:10.1145 3462757.3466088

## A Prompts

All models receive the same instruction. Placeholders are in angle brackets; the option NONE appears only in the abstain condition.

Instruction.

A court opinion cites a case and the parenthetical describing that case’s holding has been replaced by <HOLDING>. Which holding statement is the one the citing court gave for that case?

System One decision models. One request per item to the decisions interface, with the passage as state and one choice question:

{"model": "<model id>",

"E": "<holding\_4>",   
"NONE": "None of the listed holdings is the one   
the cited case is being cited for in   
this passage."}}}}

Generative LLMs. One user message per item, no system prompt: <instruction>

Citing passage: <citing\_prompt>

Options:   
A. <holding\_0>   
B. <holding\_1>   
C. <holding\_2>   
D. <holding\_3>   
E. <holding\_4>   
NONE. None of the listed holdings is the one the cited case   
is being cited for in this passage.

Reply with only the option key (A, B, C, D, E, NONE).

The reply was constrained by a JSON schema with a single field, answer, whose value must be one of the option keys. Requests used temperature 0 and a maximum of 4,096 output tokens.

## B Pilot Study and Additional Results

Stated confidence. Before the main study, we ran 1,000 randomly chosen test items with GPT-6 Luna prompted to return an answer together with its confidence from 0 to 100. In the abstain condition, it picked a holding in 60% of the items without a correct one, and its stated confidence on those wrong picks averaged 0.91. Its expected calibration error was 0.29, against 0.07 and 0.08 for the two decision models run on the same items. Abstaining whenever its stated confidence fell below a high threshold (0.90) reduced its wrong picks to a level similar to a decision-model veto, but such a threshold has to be found on labeled data. Because generative LLMs do not state confidence in normal use, the main study asks them for an answer only.

Distillation into a smaller encoder. We also fine-tuned the legal BERT model released with Case-HOLD [57] on the full training split (42,509 items), three seeds per setting. Trained on gold labels, it reached 75.7% on the original task and 61.3% on the abstain condition. Adding Jev’s probabilities to the training loss raised original-task accuracy by 0.7 points and halved its calibration error (0.043 to 0.022); on the abstain condition, it made the model abstain more often without improving accuracy. Because this encoder is already nearly as accurate as Jev on CaseHOLD, Jev has little to add. The larger gain for Qwen3-8B (Section 4) comes from a student that starts far below its teacher.

## C Contamination Checks

CaseHOLD is built from public opinions that predate the models we test, so a model may recognize a cited case rather than reason from the passage. For closed models, whose training data is not disclosed, contamination cannot be measured directly [42]. We therefore use perturbation tests as a proxy. They cannot show whether a model saw the data, only whether its answers depend on recognizing the case.

Arvin [1] introduced such a test for CaseHOLD. A language model rewrote each passage so that every case name and citation was replaced by a plausible fictitious one; GPT-4o’s macro F1 fell only from 0.744 to 0.728, and 88% of its answers were unchanged. We repeat this anonymization with GPT-6 Luna as the rewriter on a random sample of 500 test items, and we check each rewrite automatically. The cited case’s citation must be found where the holding was removed, and the rewritten passage must stay close to the original. We keep the 406 items that pass. Of the rest, 76 rewrites changed too much of the passage, and in 18 the citation could not be located. We add two variants on the same items: citation only, in which the model sees only the cited case’s citation, and citation removed, in which that citation is replaced by [citation omitted]. High accuracy from the citation alone means a model knows the case; accuracy that survives anonymization and citation removal means it does not depend on that knowledge.

Table 4. Accuracy (%) on 406 test items in the original condition, with real citations, with case names and citations replaced by fictitious ones, with the cited case’s citation removed, and with only that citation shown. Chance is 20%.
<table><tr><td>Model</td><td>Original</td><td>Anonymized</td><td>Citation removed</td><td>Citation only</td></tr><tr><td>Jev 1.14</td><td>78.8</td><td>77.1</td><td>78.3</td><td>45.3</td></tr><tr><td>pplx-decider v1.2</td><td>90.1</td><td>89.4</td><td>87.7</td><td>71.2</td></tr><tr><td>d1-ultra</td><td>86.2</td><td>84.0</td><td>83.3</td><td>63.1</td></tr><tr><td>GPT-6 Luna Decisions</td><td>73.8</td><td>73.3</td><td>72.3</td><td>41.7</td></tr><tr><td>Gemini 3.8 Flash</td><td>88.7</td><td>85.5</td><td>83.5</td><td>74.4</td></tr><tr><td>Claude Haiku 5.5</td><td>78.1</td><td>76.1</td><td>75.6</td><td>50.0</td></tr><tr><td>GPT-6 Luna</td><td>78.6</td><td>76.6</td><td>77.6</td><td>53.0</td></tr><tr><td>Mistral Small 2603</td><td>73.4</td><td>67.7</td><td>71.9</td><td>41.1</td></tr><tr><td>GLM 5.3 Flash</td><td>77.6</td><td>72.7</td><td>74.4</td><td>46.6</td></tr><tr><td>GPT-4o-mini</td><td>70.7</td><td>67.5</td><td>68.2</td><td>34.7</td></tr></table>

Models know much of the case law, but they do not depend on it (Table 4). Shown only the citation of the cited case, the strongest models pick the right holding far above chance: 74.4% for Gemini 3.8 Flash and 71.2% for pplx-decider. Yet replacing every case name and citation with a fictitious one costs between 0.5 and 5.7 points, and removing the cited case’s citation costs between 0.5 and 5.2. The decision models lose the least under anonymization (0.5 to 2.2 points). The strong results of pplx-decider are therefore not an artifact of recognizing the case, and its knowledge of case law is shared by the strongest generative LLM.

## D Distillation Setings

We fine-tuned Qwen3-8B with LoRA (rank 16, � = 32, dropout 0.05) on the attention and feedforward projections, in bfloat16 with gradient checkpointing. Training used AdamW with a learning rate of 10<sup>−4</sup>, a cosine schedule with 5% warmup, batch size 4 with 4 accumulation steps (efective batch 16), and one pass over 10,000 randomly sampled training items. The model saw the same instruction as the API models, with options lettered A–E, F for none, and thinking disabled. Passages were truncated to their last 600 tokens and each option to 160 tokens; one CaseHOLD holding exceeds 20,000 tokens. The Jev-trained model minimized the KL divergence from Jev’s probabilities to the model’s distribution over the option letters; the gold-trained model minimized cross-entropy against the correct option. Each run used one NVIDIA RTX A6000 (48 GB).