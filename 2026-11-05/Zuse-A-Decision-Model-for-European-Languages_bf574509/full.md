# Zuse: A Decision Model for European Languages

Umair Mehmood Imam Founder and Builder, Bineric AI Assistant Professor, OsloMet, Norway support@bineric.com ORCID: 0009-0002-7892-9189

November 2026

## Abstract

Decision models read a document and return a probability for each of a fixed set of options instead of generating text. Several appeared in 2026, but none has published results outside English. We describe Zuse, three decision models (4B, 9B and 27B) that train a low-rank adapter and a pointer head on frozen open-weight bases, targeting 27 European languages, and use them to study how this design behaves across languages and tasks. On a broad European suite of 11 sources excluded from training, covering 9 task groups and 27 languages, a development set that also guided model selection, the 27B model reaches 87.5% accuracy with a pooled expected calibration error of 0.016, and 0.035 when weighted per source. Post-hoc analyses of these single-seed results give three findings. First, miscalibration is structured by task rather than language: on parallel reading and topic items in 26 languages the confidence gap varies about as much as sampling noise, while across sources it spans −0.07 to +0.09, with knowledge tasks overconfident and acceptability judgements underconfident at all three sizes, so one fitted temperature cannot correct both. Second, the share of decisions that can be automated at a 5% error budget follows chance-corrected skill (Spearman 0.97 over 33 source and model pairs) more closely than raw accuracy (0.91). Third, reach beyond the training languages is set by the base model: across 210 languages, changing the training run moved scores by at most about 2 points within a size, while languages outside the base’s documented list trail by 23 to 28 points at every size. On English public suites Zuse is behind Jev and Kev, including Kev models on the same bases. We also report two pitfalls in building multi-source training data: label-word collision between topic schemes, and an injection augmentation that taught a shortcut. Zuse can be tried at https://bineric.com/zuse.

## 1 Introduction

Many operational uses of language models require a choice among known options rather than generated text. Ticket routing, policy checks, moderation and form extraction all have this shape. A decision model suits such tasks: it reads a document and a set of questions, then returns a probability for each option. Because it cannot produce text, its output stays inside the options supplied, and its probabilities can be used to decide which cases to automate and which to send to a person.

This model class has developed quickly, with Jev [1], the open reproduction Kev [2] and Clef [3] all appearing in 2026. We found no per-language evaluation for any of them. Language coverage matters in Europe, where public bodies and companies work across many languages, including several with few speakers and limited training data. Whether the design transfers across such languages has not been measured in public.

We ask whether the decision-model design carries across European languages, and what governs calibration and automation there. We study this with Zuse, a family of three decision models targeting 27 European languages. We make six contributions:

1. A per-language evaluation of decision models. We report results for 27 European languages,

11 sources and 9 task groups, and reach across 210 languages. To our knowledge this is the first such evaluation for any decision model.

2. Calibration under source shift. Miscalibration depends on the task, in the same direction at all three sizes, with no detectable language efect on parallel items. Pooled calibration error understates per-source error by a factor of two.

3. Automation limits. The automatable share tracks chance-corrected skill and how well confidence ranks right answers above wrong ones. Thresholds chosen in-sample do not transfer between splits.

4. Base-model efects. On identical data, base choice outweighed parameter count, and reach outside the training languages came from the base.

5. Two pitfalls for multi-source decision data, each with a diagnostic others can run: label-word collision, and accuracy that rises under attack.

6. A model family with a full comparison, including suites where it trails Jev and Kev.

Section 2 reviews related work. Sections 3 and 4 describe the model and data. Section 5 sets out the experimental design, and Section 6 reports results. Section 7 analyses training choices, Section 8 discusses limitations and Section 9 concludes.

## 2 Related Work

Decision models. TypeSafe’s Jev [1] introduced the typed request format of a state and named questions, trained with a method the company calls Reinforcement Learning for Calibrated Decisions. Its architecture, data and parameter count are undisclosed. Kev [2] is an open-weight reproduction on Qwen bases, and its public evaluation suites and serving code underpin much of the work reported here. Kev pairs a pretrained language model with a pointer head. Its three smaller models train a LoRA adapter on a frozen base, while Kev-27B fine-tunes every weight of a posttrained Qwen release. Clef [3] ofers Jev-compatible open-weight models on the same Qwen bases. Zuse uses the adapter design at all three sizes and targets European languages.

Calibration and selective prediction. Temperature scaling [5] is the standard post-hoc method for calibrating neural classifiers, and the Brier score [6] and expected calibration error [7] are the usual measures. Reject-option classification goes back to Chow [8], and Geifman and El-Yaniv [9] applied risk-coverage analysis to deep networks. Calibration degrades under distribution shift [27], and multilingual models are known to be less calibrated outside high-resource languages [28]. Our automation measure is an in-sample risk-coverage point, and we report its limits in Section 5.2.

Multilingual evaluation. Belebele [11] and SIB-200 [12] test reading comprehension and topic classification across many languages. ScandEval [15] provides Scandinavian benchmarks, including the acceptability data we use. We use these sets to measure transfer, and we add a comparison of language-model bases, including European-developed ones.

Prompt injection. Text inside a document can steer a language model [10, 29]. Decision models are exposed to the same risk because documents are untrusted input, yet their fixed outputs make the efect simple to measure as a change in the selected option. We use that measure in Section 7.3. Language models choosing among options are also sensitive to option order [31], and models can reach high scores through shortcuts in how a task is built [30].

## 3 Model

## 3.1 Architecture

Zuse uses a frozen open-weight language model as a reader (Figure 1). Each question forms one token sequence that holds the state, the instruction and the options, following the layout used by Kev [2]. Caller text is escaped so that it cannot contain the structural tokens. A pointer head reads the hidden states at the decision position and at each option, and gives each of the K options a score $z _ { k }$ . Probabilities follow from a single temperature $T \ [ 5 ]$

$$
p _ { k } = \frac { \exp ( z _ { k } / T ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { j } / T ) }\tag{1}
$$

Yes/no questions are two-option choices. Score questions use ordered options, and the expected position gives the score.

In most evaluations, each row carries one question. In serving, the state is encoded once and cached so that several questions can share it. This reuses Kev’s serving stack [2], which we did not modify in design.

## 3.2 Model family

Only a low-rank LoRA adapter [4] and the head are trained, with the base weights frozen. These two parts turn a text-generating language model into a decision model: the adapter changes how the base reads a request, and the head replaces text output with a probability over the supplied options. Table 1 lists the family. All three bases are published under Apache-2.0 [17].

## 3.3 Training

The adapter and head are trained with a cross-entropy objective and a permutation-consistency term, which penalises disagreement between a row and a copy of it with shufled options. Augmentations include noneof-the-above and injection. In none-of-the-above augmentation, that option appears as both the correct and an incorrect answer, following Kev [2]. In injection augmentation, a planted instruction in the state names an option while the label stays unchanged, and that named option is sometimes correct (Section 7.3). The temperature is fitted on a development set by minimising negative log-likelihood. We do not disclose loss weights, schedules or other training settings.

## 4 Training Data

We do not disclose the sources, composition or size of the training data, which we treat as commercially sensitive. The models target 27 European languages. No customer data was used in training, and no output of a commercial API model was used. Section 8 sets out what this non-disclosure means for readers.

## 5 Experimental Setup

## 5.1 Evaluation suites

Table 2 lists the suites and how each was used. The status column matters for interpretation, because several suites influenced development.

![](images/55e8adc2c8264cdbea0b324101fb200ad9cd6e8a4554fbb7525816344f380a98.jpg)  
Figure 1: Zuse architecture: document and question in, one probability per option out. The low-rank adapter sits inside the frozen base’s layers.

Table 1: The Zuse model family. Memory is resident memory when served.
<table><tr><td>Model</td><td>Base</td><td>Memory, 16-bit</td><td>Weights, 4-bit</td></tr><tr><td>Zuse-27B</td><td>Qwen3.8-27B</td><td>65.5 GB</td><td>15.1 GB</td></tr><tr><td>Zuse-9B</td><td>Qwen3.5-9B-Base</td><td>21.9 GB</td><td>5.6 GB</td></tr><tr><td>Zuse-4B</td><td>Qwen3.5-4B-Base</td><td>14.3 GB</td><td>3.1 GB</td></tr></table>

The European suite contains Belebele reading [11], SIB-200 topics [12], multilingual MMLU [26, 13], two Norwegian knowledge sets [22], a sentiment set [25], Swiss court judgments [16], MInDS-14 banking intents [14], ScaLA acceptability in four Nordic languages [15], a German retrieval set [24] and French yes/no questions [23]. None of its sources appears in training, but some of its tasks resemble tasks seen in training. The suite therefore tests new sources rather than wholly new tasks. The German retrieval set is near ceiling for all three models (99.5 to 100%).

Two sources in Transfer v4 need caution, because data used in training or in earlier runs may share origin with them. We did not confirm row-level overlap. Without those two sources, Zuse-27B’s accuracy changes by about one point or less on both splits.

## 5.2 Metrics

For N decisions with confidence c<sub>i</sub>, the highest option probability, we report accuracy, the multi-class Brier score [6] and expected calibration error over 15 equalwidth bins [7].

The automatable share at error budget b is the largest fraction of decisions, taken in descending order of confidence, whose error rate stays within b:

$$
A ( b ) = \operatorname* { m a x } { \Big \{ } { \frac { n } { N } } : { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbf { 1 } [ { \widehat { y } } _ { ( i ) } \neq y _ { ( i ) } ] \leq b { \Big \} }\tag{2}
$$

This follows the risk-coverage view of selective prediction [9]. The threshold is chosen on the rows it is scored on, and per-group figures use a separate threshold for each group, so A(b) is an in-sample upper bound. Thresholds do not transfer cleanly: a threshold set on the Transfer v4 test split gives 8.9% error against a 5% budget on the development split, and in the reverse direction 2.4% error at 61% coverage. We write auto@5 for A(0.05).

Order flips are the share of choice rows whose answer changes when options are reordered. We report this against reversed order and, separately, against a third random order, with and without averaging two orders. Both are measured on 1,223 choice rows from a 1,500-row sample of the European suite.

Planted-instruction following is the share of the same 1,500 rows on which the model selects an option named by a planted instruction. Its baseline is how often the clean model picks that option unprompted. The planted text has one wording in each of ten languages, including English, uses the English wording for all other languages, and always names a wrong option.

## 5.3 Selection procedure and statistical treatment

Almost all configurations were trained once. We selected models using the European suite together with the Transfer v4 development split, and used both to diagnose errors and to plan later runs. The English test splits were scored on five 27B checkpoints. Their accuracies span 0.859 to 0.866, and test scores were recorded alongside flagship decisions, so independence from selection cannot be claimed.

On a 764-row split, an accuracy near 0.86 has a 95% margin of about 2.5 points. For the European suite the margin is about 0.5 points under an independence assumption, and wider in practice because rows cluster within sources. Retraining on identical data moved two small-model results by up to 0.5 points, which is our only direct measure of training noise. Seeds for those retrainings are unconfirmed, and source-level scores vary far more than 0.5 points.

Table 2: Evaluation suites and their use during development.
<table><tr><td>Suite</td><td>Content</td><td>Size</td><td>Use during development</td></tr><tr><td rowspan="2">European suite</td><td>11 sources excluded from training, 9 task groups, 27</td><td>20,044 rows</td><td>Model selection, error diagnosis, design of later runs</td></tr><tr><td>languages Transfer v4 and Public English suites</td><td>764 rows in Transfer v4 per split</td><td>Development split used for selection and data choices;</td></tr><tr><td>SemIf [2]</td><td>Authored English decisions</td><td>144 and 108 rows</td><td>checkpoints Monitored during development</td></tr><tr><td>Breadth report [2]</td><td>plus perturbations 14 public datasets in 5 areas</td><td>1,990 records, devel- Scored after freezing</td><td></td></tr><tr><td>When2Call [20]</td><td>Tool-use choice</td><td>opment split 3,423 tasks</td><td>Scored after freezing</td></tr><tr><td>World languages</td><td>Belebele [11] and SIB-200 [12]</td><td>210 languages, 32,700 rows</td><td>Scored after freezing; European rows overlap the</td></tr><tr><td>Long documents</td><td>Synthetic documents, 2k to 32k tokens</td><td>1,350 rows</td><td>European suite&#x27;s sources Scored on synthetic documents outside training</td></tr></table>

## 6 Results

Unless stated otherwise, scores come from the final models at 16-bit precision. The products available today are 4-bit builds, whose results we report separately. The breadth report and When2Call were run against deployed 4-bit endpoints.

## 6.1 English public suites

Table 3 gives results on the public suites, with Jev and Kev figures as reported by the Kev project [2]. Jev has only been run on development splits, so test and development results cannot be compared across systems. On the Transfer v4 development split, Zuse-27B is 2.5 points below Jev. An unpaired test on that accuracy gap gives p = 0.19, but paired predictions are not available, so this is only indicative. Jev is also ahead on Brier score and auto@5. Kev-27B is 2.9 points ahead of Zuse-27B on the test split. On the same bases, Kev-9B and Kev-4B lead Zuse-9B and Zuse-4B by 4.8 to 6.3 points on Transfer v4, with lower Brier scores. Zuse-27B scores 0.924 on SemIf against Kev-27B’s reported 0.903, but SemIf is close to saturated and the Kev project does not use it to rank models.

## 6.2 Breadth and tool choice

Zuse-27B is behind Jev on the breadth report, and not clearly separated from Kev-27B. Table 4 shows a chance-corrected index from 0 to 100, built from 14 public datasets. The diference from Jev is 5.8 points (95% paired bootstrap CI 2.2 to 9.0). The diference from Kev-27B is 2.7 points (95% CI −0.2 to 5.3).

The Jev and Kev rows were measured by the Kev project [2]. The Jev rows were collected in September 2026 through a gateway, and the model revision is not exposed. Zuse rows are our own, and were run against a deployed 4-bit endpoint rather than the 16- bit evaluation build. That run met server errors, and all 1,990 records were answered after retries. This is a report in the style of a community index, not a third-party score.

On When2Call [20], a four-way choice (chance 25%) between using a tool, answering directly, asking for more information and saying it cannot help, and using our own instruction wording, Zuse-27B scored 57.5% and Zuse-9B 39.7% over 3,423 tasks. Mean confidence was 72.9% and 67.8%, so both models were overconfident on this task.

## 6.3 European suite

Table 5 gives the results on the European suite. The pooled calibration error is low, but this masks diferences between groups. Within sources it ranges from 0.007 to 0.091, reaching 0.091 on the Norwegian quiz, 0.082 on grammar and 0.079 on multilingual MMLU. By language it reaches 0.067, largely through task mix. Across the 14 breadth datasets, on the deployed endpoint and with 10 bins, the mean is 0.100. Weighted by rows, the per-source error is 0.035, more than twice the pooled figure, because overconfidence on some sources ofsets underconfidence on others. Calibration should therefore be checked for any specific task.

This miscalibration is structured by task rather than language. Belebele and SIB-200 give parallel items in 26 of the 27 languages, 400 rows each. On those items the gap between mean confidence and accuracy has a standard deviation across languages of 0.015 for Zuse-27B, and 0.014 and 0.013 for the smaller models, close to the sampling noise of about 0.014 at this size. Across sources the gap ranges from −0.071 to +0.091 for Zuse-27B and up to +0.144 for Zuse-4B. The direction holds at all three sizes: multilingual MMLU and the Norwegian quiz are overconfident, and grammar is underconfident. The single temperature fitted on development data reduced error on 7 of the 11 sources for Zuse-27B, and temperatures fitted per source would range from about 0.25 to 2.5. A deployment that moves to a new task therefore needs its own calibration, while moving to a new language on a known task needs less.

Table 3: English public suites. Cells show accuracy / Brier / auto@5 where available; Kev cells show accuracy / Brier. Jev and Kev figures are as reported by Kev [2].
<table><tr><td>Suite and split</td><td>Zuse-27B</td><td>Zuse-9B</td><td>Zuse-4B</td><td>Jev</td><td>Kev-27B</td><td>Kev-9B</td><td>Kev-4B</td></tr><tr><td>Transfer v4, development</td><td>0.832 7 0.246 0.576</td><td>0.772 0.316 0.410</td><td>0.755 / 0.342 0.380</td><td>0.857 / 0.211 / 0.70</td><td>0.851 / 0.225</td><td>0.820 / 0.289</td><td>0.817 / 0.269</td></tr><tr><td>Transfer v4, test</td><td>0.860 0.204 0.776</td><td>0.797 0.279 0.598</td><td>0.775 0.313 0.512</td><td></td><td>0.889 / 0.156</td><td>0.852 / 0.217</td><td>0.838 / 0.242</td></tr><tr><td>Transfer v9, development</td><td>0.786 0.305 0.462</td><td>0.716 0.382 /0.302</td><td>0.695 0.410 0.305</td><td></td><td></td><td></td><td></td></tr><tr><td>Transfer v9, test</td><td>0.787 0.285 /0.609</td><td>0.723 0.369 / 0.485</td><td>0.694 0.415 / 0.392</td><td></td><td></td><td></td><td></td></tr><tr><td>SemIf, 144 authored rows</td><td>0.924</td><td>0.847</td><td>0.882</td><td>0.965</td><td>0.903</td><td></td><td></td></tr></table>

Table 4: Breadth report on the development split, by area.
<table><tr><td>Area</td><td>Jev</td><td>Kev-27B</td><td>Z-27B</td><td>Z-9B</td></tr><tr><td>Knowledge, reasoning</td><td>37.7</td><td>33.8</td><td>36.0</td><td>24.6</td></tr><tr><td>Language understanding</td><td>77.4</td><td>75.6</td><td>69.7</td><td>57.8</td></tr><tr><td>Retrieval, classification</td><td>75.4</td><td>62.9</td><td>63.0</td><td>57.7</td></tr><tr><td>Tools, automation</td><td>63.6</td><td>64.2</td><td>59.0</td><td>58.5</td></tr><tr><td>Arts, human taste</td><td>12.7</td><td>14.7</td><td>10.0</td><td>6.7</td></tr><tr><td>Overall</td><td>53.3</td><td>50.2</td><td>47.5</td><td>41.1</td></tr><tr><td>95% CI lower</td><td>50.7</td><td>47.4</td><td>44.8</td><td>38.4</td></tr><tr><td>95% CI upper</td><td>56.3</td><td>53.3</td><td>50.7</td><td>44.2</td></tr></table>

Figure 2 shows the automatable share by task group. It mostly follows accuracy (Spearman 0.93 across the nine groups), but accuracy is not corrected for chance. Nordic grammar is a clear case: 72.6% on balanced labels, with only 8% automatable at a 5% error budget. On Swiss court rulings Zuse-27B reaches 78.7%, below the 79.3% from always answering no, so it shows no gain over the majority answer, and only 1.5% can be automated. Banking intents and reading comprehension reach far higher shares. Across the 11 sources and three models, the automatable share correlates more strongly with chance-corrected skill, taking the larger of the majority-class and uniform baselines, than with raw accuracy (Spearman 0.97 against 0.91 over 33 pairs). It correlates most strongly with how well confidence ranks right answers above wrong ones (AUROC, Spearman 0.98 over the 31 pairs where it is defined). These are in-sample upper bounds with per-group thresholds (Section 5.2).

Figure 3 shows results by language. Icelandic (600 rows) and Bokmål (1,400 rows) sit within sampling noise of 80%. Language averages reflect task mix as well as language, since Nordic languages carry the grammar and quiz sources. On Belebele reading and SIB-200 topics in the final model, those languages score between 87 and 96.

## 6.4 World languages

Table 6 reports reach across languages, including the languages trained on. Reading covers 122 languages from Belebele, and topic classification 205 from SIB-200, for 210 languages and 32,700 rows in total. We grouped languages using an approximate list of 140 codes derived by hand from the language list Qwen published for Qwen3, applied here to later Qwen versions.

On reading and topic together, Zuse-27B reaches 85% or above in 115 languages and falls below 60% in 25. Four of the five weakest are written in N’Ko, Ol Chiki or Tifinagh, at 28 to 31%, and the weakest, Central Kanuri in Arabic script, scores 22%. These have topic rows only, where uniform chance is 14.3% and always choosing the largest class gives 22 to 26%.

## 6.5 Inference speed

Table 7 reports model time at 16-bit precision on one B300 GPU, using Kev’s serving stack. On 100 records, this path difered from the unoptimised path by at most 0.019 in probability, with at most one changed answer per model. In a 4-bit deployment on a 24 GB card, a two-question request took about 230 ms on a cached document. On 24 GB cards, documents are limited to 8,192 tokens.

## 7 Analysis of Training Choices

This section compares single-seed runs, so it is observational rather than a controlled study. Runs in a comparison may also difer in settings we do not report, so these comparisons are indicative. We treat gaps under 0.5 points on the European suite as noise (Section 5.3).

## 7.1 Base model and size

The largest efect we observed came from the base model. On an early corpus and identical data, moving from Qwen3.5-4B to Qwen3.8-27B raised accuracy on held-out sources from 75.4 to 85.4. On the same data and recipe, Qwen3.8-27B scored 85.6 against 75.8 for EuroLLM-22B [18], and led in all 27 languages. Borealis 2 31B [19], a Norwegian-trained model on a Gemma base, reached 82.9. On the same data, EuroLLM-22B also scored below Qwen3.5-4B (77.8) and Qwen3.5-9B (80.0), and on the early corpus Mistral-Nemo-12B scored 66.1 against 75.4 for Qwen3.5-4B, so base choice outweighed parameter count in these runs. Borealis 2 did not lead Qwen3.8- 27B even on Bokmål (76.0 against 77.1) and was at chance on grammar (49.5). These comparisons have limits: the recipe was developed on Qwen bases, and the bases difer in post-training.

Table 5: European suite. Order flips use 1,223 choice rows and planted instructions a 1,500-row sample.
<table><tr><td>Measure</td><td>Zuse-27B</td><td>Zuse-9B</td><td>Zuse-4B</td></tr><tr><td>Accuracy (%)</td><td>87.5</td><td>84.1</td><td>81.2</td></tr><tr><td>Brier score</td><td>0.185</td><td>0.229</td><td>0.264</td></tr><tr><td>Expected calibration error</td><td>0.016</td><td>0.023</td><td>0.032</td></tr><tr><td>Automatable at 1 / 2 / 5 / 10% error (%)</td><td>40 / 58 / 77 / 94</td><td>30 / 50 / 67 / 86</td><td>26 / 42 / 61 / 78</td></tr><tr><td>Order flips, reversed order (%)</td><td>5.6</td><td>6.2</td><td>8.9</td></tr><tr><td>Order flips against a third order, single / two averaged (%) Planted instruction followed, baseline in brackets (%)</td><td>3.7 / 3.3</td><td>4.4 / 3.5 12.2 (7.1)</td><td>6.9 / 5.2</td></tr><tr><td></td><td>8.1 (5.8) 87.4</td><td>83.6</td><td>14.2 (8.3)</td></tr><tr><td>Accuracy at 4-bit (%) Brier score at 4-bit</td><td>0.188</td><td></td><td>80.6 0.275</td></tr><tr><td>Expected calibration error at 4-bit</td><td>0.016</td><td>0.236 0.026</td><td>0.034</td></tr><tr><td>auto@5 at 4-bit (%)</td><td>76.0</td><td>66.3</td><td>58.6</td></tr><tr><td>Separate long-document set, 2k to 32k tokens (%)</td><td>98.1</td><td>98.1</td><td>97.6</td></tr></table>

![](images/5cdb3c694ac429fcde16ae0e77626716a640f4956f60190361a8ff54ef5699dc.jpg)  
Figure 2: Accuracy and in-sample automation share by task group, Zuse-27B.

Within each size, world-language scores varied little across the nine checkpoints we scored, staying within 1.2 points of that size’s final model for languages in the Qwen list and within 2.1 outside it. Between sizes they difer by 14 to 16 points on reading. This suggests that reach outside the training languages comes largely from the base. The evidence is limited, because all nine checkpoints trained on similar data and there is no control.

## 7.2 Label-word collision

Adding one topic-labelled data family lowered accuracy on the SIB-200 topic test by about 10 points. The closest isolated pair of runs difers only in that family: 87.6 without it and 77.3 with it. Overall accuracy on the European suite moved from 86.7 to 83.4.

The cause appears to be shared label words. The added family used two label words that also name SIB-200 categories, with narrower meanings. On the afected run, the model predicted one of those categories 2.4 times as often as it occurs. Datasets merged for training may therefore conflict through their label vocabulary even when their languages and domains look compatible.

## 7.3 Injection training and shortcut learning

Our first injection augmentation produced results that looked strong and proved misleading. On a Qwen3.5- 4B run, following of planted instructions fell from 37.6% to 0.2%, but accuracy under attack rose above clean accuracy, from 75.6 to 81.7. Every planted note had named a wrong option, so the model could learn that a named option is incorrect.

In a run that also dropped the reordered copies, we changed the augmentation so that planted notes sometimes name the correct option. The headline figure then looked worse, at 16.8% following against a 9.6% baseline, with accuracy under attack at 71.9 against 75.5 clean. This is the more honest result, because the model is no longer exploiting the construction of the attack. In the final Zuse-27B, following is 8.1% against a 5.8% baseline, and accuracy falls from 87.1 to 85.6 on the same sample. Robustness evaluations should therefore check for accuracy that

Accuracy (%) on the European suite. Nordic rows carry more grammar and quiz tasks.

Zuse-27B scores at least 80% in 26 of 27 languages  
![](images/0a114cfbb3bbecad456080d9198c7921aa06a5c7b81f0bac543a508890303724.jpg)  
Figure 3: Accuracy by language on the European suite, all three models.

Table 6: Accuracy (%) on world languages, by whether the language is in the Qwen list. Cells show reading / topic.
<table><tr><td>Group</td><td>Reading languages</td><td>Topic languages</td><td>Zuse-27B</td><td>Zuse-9B</td><td>Zuse-4B</td></tr><tr><td>In the list</td><td>97</td><td>139</td><td>85.1 / 87.8</td><td>76.5 / 86.1</td><td>71.0 / 84.6</td></tr><tr><td>Outside the list</td><td>25</td><td>66</td><td>60.5 / 65.2</td><td>48.7 / 59.6</td><td>44.8 / 56.2</td></tr></table>

improves under attack.

## 7.4 Order stability

Adding reordered copies of rows had almost no efect on order flips (13.6% to 13.1%, in a step that also added injection training), whereas a permutation-consistency loss helped. Across 27B runs, flips against a third order averaged 6.4% in twelve runs without the loss and 4.5% in eight runs with it. Those groups also difer in data, so the comparison is indicative. At a stronger setting on the 4B model, the loss cut flips from 10.5% to 6.8% and cost about 1.5 accuracy points. At a weaker setting on the 27B model we saw no accuracy cost.

## 8 Limitations and Threats to Validity

Model limitations. Knowledge-heavy questions remain weak in this training recipe: 75.9% on the quiz group, and 0.675 on MMLU-Pro [21] (200 development rows), equal to Kev-27B’s reported 0.675 and below Jev’s reported 0.84. Nynorsk scores 67.3%. Order sensitivity and instruction following are reduced but not removed. Accepted input runs to 64k tokens on larger GPUs, but accuracy above 32k is unmeasured because we lacked enough held-out documents. There is no image input.

Selection bias. The European suite shaped model selection, error diagnosis and the design of later runs, including removal of one data family and one targeted run. The Transfer v4 development split shaped data choices and flagship selection. Scores on both are likely to be optimistic. The English test splits were seen several times, and their scores were recorded alongside flagship decisions.

Missing comparisons. We have not run Jev, Kev or Clef on the European suite, so we make no claim about relative European performance. Nor have we compared frontier language models with structured output on the same rows.

Comparability. Jev and Kev figures come from the Kev project. Jev’s revision is unknown and its automation figure uses a development split. Zuse’s breadth scores came from a deployed endpoint rather than the evaluation build used elsewhere.

Table 7: Model time per request, new document / cached document, and throughput.
<table><tr><td>Model</td><td>2 questions (ms)</td><td>6 questions (ms)</td><td>5 questions, 2,200 tokens (ms)</td><td>Requests per second, 1 / 32 clients</td></tr><tr><td>Zuse-27B</td><td>32  / 18</td><td>45 / 31</td><td>182  / 51</td><td>20 /  47</td></tr><tr><td>Zuse-9B</td><td>13 / 8</td><td>17 / 12</td><td>86 / 20</td><td>49 / 118</td></tr><tr><td>Zuse-4B</td><td>10 / 6.5</td><td>14 / 10</td><td>84 / 18</td><td>62 / 141</td></tr></table>

Statistical power. Results come from single seeds, and our noise estimate rests on two small-model retraining pairs. English test splits are small. Rows within the European suite are clustered by source, so simple margins understate uncertainty.

Construct validity. Not all training labels were independently checked. Long-document rows are simple fact checks. The planted-instruction test uses few wordings and always names a wrong option. All results come from public benchmarks or synthetic documents, not from a live deployment.

Undisclosed training details. We do not disclose the training data or the full training recipe, so the training cannot be reproduced from this paper, and readers must rely on our statement that the evaluation sources in the European suite were excluded from training. We compared the source identifiers in the final training list with the suite’s 11 sources and found no match, but did not compare row text for that final list. Public suites can be run against the 4-bit Zuse-27B and Zuse-9B. Zuse-4B, the 16-bit builds and the sampled European suite rows are not available.

## 9 Conclusion

We presented Zuse, a family of adapter-based decision models targeting 27 European languages, and reported its results across English, European and worldlanguage evaluations. On European sources excluded from training the largest model reaches 87.5% accuracy, though that figure is a development result. On English public suites it is behind Jev and Kev on every aggregate measure where both were scored. Four observations are likely to carry beyond this system, though each rests on single-seed runs of one model family: miscalibration follows the task more than the language, so calibration should be checked per task; the automatable share follows chance-corrected skill; base-model choice outweighed parameter count and set language reach; and merged datasets can conflict through their label vocabulary, while robustness results should be checked for shortcuts.

The priority for future work is a clean evaluation: a new European test suite, scored once and alongside open decision models. Repeated seeds for the main comparisons would resolve several of the confounds identified here.

## Availability

Readers can test Zuse-27B through a live demo at https://bineric.com/zuse, which needs no API key, or through the Bineric API. Zuse-27B and Zuse-9B can also be run in a reader’s own cloud account as 4-bit models through AWS Marketplace, and an Azure Marketplace ofer is in preparation. Deployment guides and sample inputs are at https://github.com/ binericai/models/tree/main/zuse. Zuse weights are not openly released, and Zuse-4B is not ofered. The training data and training code are not released.

## Competing Interests

The author is the founder of Bineric AI, which sells Zuse, and an assistant professor at OsloMet. This work was carried out at Bineric AI and was not part of the author’s duties at OsloMet. AI tools assisted with drafting and auditing this paper.

## Acknowledgements

This work was completed in part at the Open Models Codefest, part of the NVIDIA, Deutsche Telekom, and Open Hackathons Program, using NVIDIA B300 GPUs. The author would like to acknowledge NVIDIA for their support. Zuse builds on Qwen models by the Qwen team at Alibaba Cloud under the Apache License 2.0, and on Kev by Jared Palmer under the Apache License 2.0.

## References

[1] TypeSafe. Introducing System One models and Jev. 2026. https://typesafe.ai/blog/ introducing-system-one-models-and-jev

[2] J. Palmer. Kev: open decision models. 2026. https: //github.com/jaredpalmer/kev

[3] Cloudflare. Introducing Clef: our open-source decision models. 2026. https://blog.cloudflare.com/ clef-decision-models/

[4] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021.

[5] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. arXiv:1706.04599, 2017.

[6] G. W. Brier. Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review 78(1), 1950.

[7] M. P. Naeini, G. F. Cooper, M. Hauskrecht. Obtaining Well Calibrated Probabilities Using Bayesian Binning. In AAAI, 2015.

[8] C. K. Chow. On Optimum Recognition Error and Reject Tradeof. IEEE Transactions on Information Theory 16(1), 1970.

[9] Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. arXiv:1705.08500, 2017.

[10] X. Liu et al. Automatic and Universal Prompt Injection Attacks against Large Language Models. arXiv:2403.04957, 2024.

[11] L. Bandarkar et al. The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. arXiv:2308.16884, 2023.

[12] D. I. Adelani et al. SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects. arXiv:2309.07445, 2023.

[13] D. Hendrycks et al. Measuring Massive Multitask Language Understanding. arXiv:2009.03300, 2020.

[14] D. Gerz et al. Multilingual and Cross-Lingual Intent Detection from Spoken Data. arXiv:2104.08524, 2021.

[15] D. S. Nielsen. ScandEval: A Benchmark for Scandinavian Natural Language Processing. arXiv:2304.00906, 2023.

[16] J. Niklaus et al. Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark. arXiv:2110.00806, 2021.

[17] Qwen Team, Alibaba Cloud. Qwen3.8-27B and Qwen3.5 model releases. Apache License 2.0, 2026. https://huggingface.co/Qwen

[18] EuroLLM Team. EuroLLM-22B: Technical Report. arXiv:2602.05879, 2026.

[19] National Library of Norway AI Lab. Borealis 2 31B, preview release. 2026. https://huggingface. co/NbAiLab/borealis2-31b-preview

[20] H. Ross, A. S. Mahabaleshwarkar, Y. Suhara. When2Call: When (not) to Call Tools. In NAACL, 2025. arXiv:2504.18851.

[21] Y. Wang et al. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. arXiv:2406.01574, 2024.

[22] V. Mikhailov et al. A Collection of Question Answering Datasets for Norwegian. arXiv:2501.11128, 2025.

[23] M. Faysse et al. CroissantLLM: A Truly Bilingual French-English Language Model. arXiv:2402.00786, 2024.

[24] Deutsche Telekom. Ger-RAG-eval. Hugging Face dataset. https://huggingface.co/datasets/ deutsche-telekom/Ger-RAG-eval

[25] K. Enevoldsen et al. MMTEB: Massive Multilingual Text Embedding Benchmark. arXiv:2502.13595, 2025.

[26] Alexandra Institute. m\_mmlu: machine-translated MMLU. Hugging Face dataset. https:// huggingface.co/datasets/alexandrainst/m\_mmlu

[27] Y. Ovadia et al. Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. In NeurIPS, 2019. arXiv:1906.02530.

[28] K. Ahuja et al. On the Calibration of Massively Multilingual Language Models. In EMNLP, 2022. arXiv:2210.12265.

[29] K. Greshake et al. Not What You’ve Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. arXiv:2302.12173, 2023.

[30] R. Geirhos et al. Shortcut Learning in Deep Neural Networks. Nature Machine Intelligence 2, 2020. arXiv:2004.07780.

[31] C. Zheng et al. Large Language Models Are Not Robust Multiple Choice Selectors. In ICLR, 2024. arXiv:2309.03882.