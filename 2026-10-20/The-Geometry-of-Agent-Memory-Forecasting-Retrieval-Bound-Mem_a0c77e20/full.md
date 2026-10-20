# The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures, and Characterizing Its Limits

Iliya Zhechev<sup>1</sup> Radoslav Ralev<sup>2</sup> Chris Coleman<sup>3</sup> Srijith Rajamohan<sup>4</sup> Jen Agarwal<sup>5</sup> Tyler Hutcherson<sup>6</sup> Simba Khadder<sup>7</sup>

<sup>1</sup>iliya.zhechev@redis.com <sup>2</sup>radoslav.ralev@redis.com <sup>3</sup>ccol@u.northwestern.edu <sup>4</sup>srijith.rajamohan@gmail.com <sup>5</sup>honey.agarwal@gmail.com <sup>6</sup>tyler.hutcherson@redis.com <sup>7</sup>simba.khadder@redis.com

Accepted as a poster at the PALM Workshop at NeurIPS 2026.

## Abstract

We study whether geometric relationships between conversation turns, memory texts, and queries can predict answer correctness before generation. We evaluate four memory implementations with dense or hybrid retrieval and one observationsummary comparator. We embed these texts in a shared space. A logistic classifier uses 25 features describing query proximity, memory–conversation correspondence, and conversation structure. On LongMemEval-small, pooled AUROC is 0.640 over 2,367 question–system pairs, below a question-difficulty baseline of 0.655. Performance varies by question type: geometry reaches 0.841 on single-session-assistant questions, but system identity alone reaches 0.826. After controlling for difficulty and identity, geometry adds no statistically established pooled improvement (∆ = +0.002, 95% CI [−0.010, 0.014]). A fixed-store context-selection experiment finds higher accuracy with proximity-ranked memories than with random subsets, consistent with ordinary retrieval relevance. On LoCoMo, nearest-turn similarity has modest predictive value for single-hop questions and near-chance estimates for several other categories. Training uses correctness labels; prediction uses pre-answer features and no additional LLM generation call. The results characterize associations between memory representations and correctness, rather than establish a reliable deployment diagnostic. In particular, embedding proximity does not verify fact coverage or distinguish memory-construction, retrieval, and generation errors.

## 1 Introduction

Long-term agent memory can use text retrieval, summaries, knowledge graphs, symbolic stores, or parametric updates. This study focuses on text memory representations associated with dense retrieval. We also include an observation-summary system as a comparator. Benchmark QA accuracy measures one aspect of these systems, alongside cost, latency, storage, and other concerns. A separate question is whether a signal available before generation can help identify queries that are likely to receive an incorrect answer.

![](images/2a29bc6d3332f9a599812311bc2bff7a09b12115d9d9bfc4c2550e7f42bd72b2.jpg)  
(a) Illustrative t-SNE projection. 2-D distances do not preserve original cosine distance

(b) Sorted original-space query-to-memory distances for these examples  
![](images/485ed1f384f4602158135dcb829a5c842743b5ba78d7f6b6bdd2c27c9e1b5821.jpg)  
Figure 1: Conversation turns, memories, and queries for two selected examples. The correctly answered query is closer to memory points than the incorrectly answered query. The t-SNE projection is illustrative; all measurements use original-space cosine distance. The examples do not establish fact coverage or the cause of an error.

We study three representations: source conversation turns, memory texts produced by a system, and an incoming query. Dense retrieval systems already embed memories and queries. We additionally embed conversation turns in the same space (Figure 1). We compute two feature blocks and conversation covariates:

• Query reachability: nearest-memory distance, mean distance to the five closest memories, local memory density, and nearest-turn distance. Here, reachability names an embedding-proximity measurement.

• Memory–conversation correspondence: turn-to-memory proximity, session alignment, memory count, and memory dispersion. These describe geometric correspondence rather than verified coverage of individual facts.

A logistic classifier predicts answer correctness from these features. We examine variation across question types, incremental prediction beyond question difficulty and system identity, and accuracy under different context-selection rules. The strongest observed AUROC is 0.841 for single-sessionassistant questions. However, system identity alone reaches 0.826 on that same type. The high within-type score therefore does not establish independent prediction of instance-level memory failures. Pooled over all types, geometry adds no significant gain beyond difficulty and system identity (§5.6).

Memory construction can omit evidence, retrieval can miss evidence already stored, and generation can fail despite sufficient retrieved context. We test the hypothesis that the first two cases leave geometric signatures. Our use of retrieval-bound refers to this motivating hypothesis, not a verified classification of the observed errors. Distances alone cannot show whether an answer-bearing fact was stored, included in the actual prompt, or used correctly.

A warning score could eventually support abstention, broader retrieval, or recovery from the source conversation. Our present results do not validate such a deployment policy. The scope analysis uses labeled outcomes, and prediction across unseen memory architectures remains unreliable.

Analysis code and numerical artifacts: github.com/ichko/agent-memory-geometry.

## 2 Related work

Agent-memory benchmarks and systems. LongMemEval [Wu et al., 2025] evaluates 500 questions in six categories over multi-session histories. LoCoMo [Maharana et al., 2024] uses a different conversational dataset and question taxonomy. Memory-system studies report end-to-end QA results alongside other measurements [Chhikara et al., 2025, Rasmussen et al., 2025]. We examine whether features of the memory representation are associated with per-question correctness.

Selective prediction and confidence. Selective prediction ranks examples by expected correctness and allows deferral [Hendrycks and Gimpel, 2017, Geifman and El-Yaniv, 2017]. Verbalized confidence and P(True) use the answer model’s own judgment [Kadavath et al., 2022]. Our geometric score is supervised during training but does not require another generation call at prediction time.

Retrieval diagnostics and query-performance prediction. Query–passage similarity is an established retrieval signal. Query-performance prediction estimates retrieval difficulty from query or ranked-list statistics [Cronen-Townsend et al., 2002, Carmel and Yom-Tov, 2010]. RAG evaluation also measures answer groundedness, often using LLM judges [Es et al., 2024, Saad-Falcon et al., 2024]. Our reachability features overlap with ordinary similarity and ranked-list signals. The setting here is a memory representation constructed from a conversation. We add conversation–memory correspondence, compare correctness across memory implementations, and examine differences by question type. The intervention tests the value of relevant context, without establishing a new geometric cause separate from relevance.

## 3 Geometric features

A shared model $E : \mathrm { t e x t } \to  { \mathbb { R } ^ { d } }$ embeds conversation turns T, stored memory texts M, and a query q. Distances are cosine distances between unit-normalized embeddings. The core reachability scalar is $\mathtt { q \_ n n \_ m e m } = \operatorname* { m i n } _ { m \in M } d ( q , m )$ . The query block has six features. The correspondence block has 14 features: seven coverage, two alignment, and five memory-dispersion features. The full model also includes five conversation-structure covariates, for 25 features in total (Appendix A).

Coverage here means proximity between embedded texts. A nearby memory may omit the required detail, express a stale fact, or contradict the source. A distant memory may still contain the answer. The classifier is logistic regression on standardized scalar features. Inspectable inputs do not give its coefficients a causal interpretation.

## 4 Experimental setup

Systems and benchmark. We evaluate five implementations on LongMemEval-small. Four retrieve text using dense similarity, with hybrid retrieval in one case. The fifth constructs an observation summary and passes it to the answer model. We use the following functional names:

• Sem-RAG: semantic-chunk RAG over source turns (65.8% accuracy).

• Obs-Extract: a local Mastra-style observation-memory implementation (67.9%). Its logged representation is one observation summary.

• Ext-Dense: instruction-based extraction and dense retrieval (72.6%).

• Svc-Mem: a local LangMem-based implementation (73.2%).

• Ext-Hybrid: instruction-based extraction and hybrid retrieval (86.0%).

These results concern the particular configurations in our harness. They do not represent all configurations of the named libraries or products. The answer model and judge use the gpt-4o API alias across these runs. Answer prompts remain specific to each implementation. The feature analysis uses deduplicated memory texts recorded in answer logs, rather than the precise context passed to the answer model. Joining these representations and judged answers gives 2,367 question–system pairs over 500 distinct questions. Appendix B gives counts and implementation details.

Protocol. We use StratifiedGroupKFold with five folds, grouping by question\_id and using seed 0. This keeps all system rows for a question in one fold. Scaling is fitted on each training fold. Reported AUROCs use one cross-validation pass. Each per-type result comes from a separate classifier trained within that type. We predict correctness and report AUROC for that ranking; reversing labels and scores gives the corresponding failure AUROC. 95% intervals resample question groups from fixed out-of-fold predictions, with 1,000 draws. They capture sampling variation conditional on that fit, not model-refitting or split-selection uncertainty. Paired comparisons use the same question-group resampling.

The original analysis imputes undefined features using column means before the fold split. The question-difficulty baseline likewise fits unsupervised PCA on the full question set. These operations do not use labels, but expose preprocessing to held-out covariates. Grouping controls the specific leakage caused by repeated question rows. The conversation-only baseline is 0.505 under this grouped protocol. Primary embeddings use text-embedding-3-small.

## 5 Results

## 5.1 Prediction varies by question type

Pooled geometry AUROC is 0.640 [0.615, 0.664] over 2,367 rows. Table 1 and Figure 2 show variation across types. Single-session-assistant has the strongest score, 0.841, whereas temporal reasoning reaches 0.628. Preference and knowledge-update are close to chance. The multi-session score is modestly above chance, with an interval excluding 0.5.

Table 1: Geometry AUROC by type, with separately fitted classifiers and question-group bootstrap intervals. n counts question–system pairs. These results do not classify the causes of individual errors.
<table><tr><td>Question type</td><td>n</td><td>AUROC</td><td>95% CI</td><td>% wrong</td></tr><tr><td>single-session-assistant</td><td>265</td><td>0.841</td><td>[0.802, 0.881]</td><td>27.5</td></tr><tr><td>temporal-reasoning</td><td>623</td><td>0.628</td><td>[0.567, 0.683]</td><td>27.8</td></tr><tr><td>single-session-user</td><td>333</td><td>0.607</td><td>[0.478, 0.762]</td><td>8.1</td></tr><tr><td>single-session-preference</td><td>137</td><td>0.567</td><td>[0.485, 0.659]</td><td>48.2</td></tr><tr><td>multi-session</td><td>639</td><td>0.566</td><td>[0.514, 0.618]</td><td>33.3</td></tr><tr><td>knowledge-update</td><td>370</td><td>0.542</td><td>[0.459, 0.617]</td><td>21.1</td></tr></table>

Within-type evaluation holds question type constant, but it does not hold system identity or question content constant. For single-session-assistant, identity alone reaches 0.826. Geometry’s paired improvement $\mathrm { i s + 0 . 0 1 5 \ [ \bar { - } 0 . 0 2 0 , 0 . 0 4 9 ] }$ and is not statistically established. Memory count alone reaches 0.830. The strongest score is therefore compatible with representation-size and systemaccuracy differences (§5.6). The single-session-user estimate is particularly uncertain because only 27 of its 333 rows are failures.

## 5.2 Feature-block ablation

Table 2 compares the correspondence and query blocks with the full 25-feature model. Their pooled AUROCs are 0.580, 0.598, and 0.640. The full model also includes conversation-structure covariates, so the comparison does not isolate a correspondence–query interaction.

Full geometry exceeds the query block by +0.042 [0.019, 0.065] and correspondence by +0.060 [0.036, 0.085] in paired question-group bootstrap comparisons. The query-minus-correspondence difference is +0.018 [−0.013, 0.049], so their numerical ordering is not statistically established. Within types, neither block consistently dominates. For single-session-assistant, correspondence alone reaches 0.843. For preference and single-session-user, the full model underperforms correspondence alone. The pooled ordering does not describe every type.

## 5.3 The reachability gap is an offline description

For each type, the reachability gap is the mean nearest-memory distance for incorrect answers minus that for correct answers. Single-session-assistant has a substantially larger positive gap than other types (Table 3). This describes an association between proximity and correctness. It does not identify whether the required fact was absent from the store, absent from the prompt, or incorrectly used. Similar mean distances for knowledge-update and preference do not imply identical distance distributions or rule out other predictive features.

Geometry AUROC varies across question types  
![](images/beaab6165e7bd431c4b27fdf5df88ee2f6fd7a6ab70397ffe30ea8f72681d1b7.jpg)  
Figure 2: AUROC within each type (five implementations). Bars show full geometry; diamonds show a system-identity-only classifier under the same protocol. Intervals are question-group bootstrap 95% intervals conditional on fixed out-of-fold predictions.

Table 2: Feature-block AUROCs under grouped CV. Full geometry includes both blocks and five conversation covariates. Paired pooled comparisons are reported in the text; per-type block comparisons are point estimates. <sup>∗</sup>Single-session-user has few errors and substantial uncertainty.
<table><tr><td>Question type</td><td>memory-conv. correspondence</td><td>query reachability</td><td>full geometry</td></tr><tr><td>single-session-assistant</td><td>0.843</td><td>0.825</td><td>0.841</td></tr><tr><td>single-session-preference</td><td>0.605</td><td>0.478</td><td>0.567</td></tr><tr><td>temporal-reasoning</td><td>0.590</td><td>0.612</td><td>0.628</td></tr><tr><td>multi-session</td><td>0.562</td><td>0.517</td><td>0.566</td></tr><tr><td>knowledge-update</td><td>0.547</td><td>0.512</td><td>0.542</td></tr><tr><td>single-session-user</td><td>0.645</td><td>0.475</td><td>0.607</td></tr><tr><td>pooled</td><td>0.580</td><td>0.598</td><td>0.640</td></tr></table>

Table 3: Mean nearest-memory cosine distances by judged correctness. AUROC is from the full per-type classifier in Table 1. The large-gap category has the highest AUROC, but the smaller-gap categories do not show a precise monotonic ordering.
<table><tr><td>Question type</td><td>correct</td><td>wrong</td><td>gap</td><td>AUROC</td></tr><tr><td>single-session-assistant</td><td>0.377</td><td>0.618</td><td>+0.241</td><td>0.841</td></tr><tr><td>single-session-user</td><td>0.533</td><td>0.579</td><td>+0.046</td><td>0.607</td></tr><tr><td>temporal-reasoning</td><td>0.526</td><td>0.570</td><td>+0.044</td><td>0.628</td></tr><tr><td>multi-session</td><td>0.534</td><td>0.567</td><td>+0.033</td><td>0.566</td></tr><tr><td>knowledge-update</td><td>0.456</td><td>0.454</td><td>-0.003</td><td>0.542</td></tr><tr><td>single-session-preference</td><td>0.570</td><td>0.541</td><td>-0.029</td><td>0.567</td></tr></table>

Gaps are computed before rounding the mean distances.

For knowledge-update, the full classifier reaches 0.542. Exploratory nonlinear and raw-embedding probes did not establish a reliable memory-specific improvement on this category. Selecting a stale fact is one possible explanation. We have not audited the actual evidence sufficiently to identify that cause.

The gap uses correctness labels and describes the evaluated data offline. It cannot tell a deployed agent whether an individual prediction is trustworthy. It concerns nearest-memory distance rather than the full feature set. Preference has a negative gap while correspondence-only reaches 0.605, illustrating that distinction.

## 5.4 Reference-answer proximity: a proxy analysis

We embed the reference answer as an offline anchor. Let $m ^ { * }$ be its nearest memory, $c = d ( a , m ^ { * } )$ its distance to reference answer $^ { a , }$ and $r = d ( q , m ^ { * } )$ its distance to the query. Thresholds $\tau _ { c }$ and $\tau _ { r }$ are the 75th percentiles of these quantities among correctly answered rows. An incorrect row is answer-distant if $\dot { \mathbf { \zeta } } c > \tau _ { c }$ , query-distant if $c \leq \tau _ { c }$ and $r > \tau _ { r }$ , and neither otherwise.

These bins do not identify extraction, retrieval, or reasoning errors. Proximity cannot establish answer entailment, and r does not show whether $m ^ { * }$ reached the actual prompt. Table 4 retains the measurements with operational names.

Table 4: Reference-answer proximity among 630 failed rows. Thresholds are the 75th percentiles among correct rows. These are proxy bins, rather than verified extraction, retrieval, or reasoning failure rates. Percentages are rounded independently.
<table><tr><td>Question type</td><td># fail</td><td>answer-distant</td><td>query-distant</td><td>neither</td></tr><tr><td>single-session-assistant</td><td>73</td><td>62%</td><td>5%</td><td>33%</td></tr><tr><td>multi-session</td><td>213</td><td>37%</td><td>21%</td><td>43%</td></tr><tr><td>knowledge-update</td><td>78</td><td>31%</td><td>8%</td><td>62%</td></tr><tr><td>temporal-reasoning</td><td>173</td><td>25%</td><td>13%</td><td>61%</td></tr><tr><td>single-session-user</td><td>27</td><td>19%</td><td>19%</td><td>63%</td></tr><tr><td>single-session-preference</td><td>66</td><td>0%</td><td>3%</td><td>97%</td></tr><tr><td>all</td><td>630</td><td>31%</td><td>13%</td><td>56%</td></tr></table>

For single-session-assistant, 62% of failed rows are answer-distant and 5% are query-distant. The 62% does not mean the fact was verified as absent. None of the 2,367 evaluated memory representations is empty. In Obs-Extract, however, the whole summary is represented as one text, changing the granularity of the comparison.

At the 90th-percentile thresholds, the pooled neither share increases from 56% to 77%. The reference anchor and correctness-conditioned thresholds couple this analysis to the labels. We have not measured separate AUROCs for verifiedfact absent andfact present but not retrieved cases. That requires checking answer entailment in the store and actual prompt, including current and stale facts for knowledge-update questions.

## 5.5 Proximity-ranked context selection affects accuracy

On Ext-Hybrid, we keep the memory pool fixed and vary the subset provided to the answer model: 15 cosine-closest memories, the native ranking, random subsets over five seeds, or 15 farthest memories. Table 5 reports accuracy under two answer-model aliases.

Table 5: Interventional accuracy when the answer model is given different memory subsets from the same store (Ext-Hybrid, K=15, two answer models).
<table><tr><td>Memories shown to the model</td><td>gpt-4o-mini</td><td>gpt-4o</td></tr><tr><td>Farthest-15 (cosine floor)</td><td>0.068</td><td>0.060</td></tr><tr><td>Random-15 (5 seeds)</td><td>0.114</td><td>0.091</td></tr><tr><td>Closest-15 (pure cosine)</td><td>0.752</td><td>0.790</td></tr><tr><td>Native-15 (system&#x27;s own retriever)</td><td>0.738</td><td>0.822</td></tr></table>

Closest-15 reaches 0.752 with gpt-4o-mini, compared with 0.114 for random selection. With gpt-4o, the values are 0.790 and 0.091. Native ranking reaches 0.738 and 0.822. The ordering between native and cosine selection depends on the model.

The intervention shows that selected context affects answer accuracy. Cosine ranking is an ordinary relevance criterion. The experiment does not isolate reachability from relevance, establish causal effects of all 25 features, or show that the classifier improves retrieval. Fixed memory counts can also produce different context token lengths.

## 5.6 Baselines and system confounding

Pooled baselines. Geometry (0.640) exceeds nearest-memory similarity alone (0.603), with paired ∆ = +0.037 [0.012, 0.063]. Question difficulty uses type, length, and PCA-64 of the question embedding. Its AUROC is 0.655. Adding geometry gives 0.681, an improvement of +0.026 [0.010, 0.042] over difficulty alone.

The pooled increment is not established after adding system identity. Identity alone reaches 0.578. Difficulty plus identity reaches 0.679, while adding geometry gives 0.681. The paired difference is +0.002 [−0.010, 0.014]. The evaluated geometry therefore does not establish additional pooled prediction beyond these controls.

The strongest type also has a system confound. Within single-session-assistant, identity alone reaches 0.826 [0.782, 0.871], compared with geometry at 0.841 [0.802, 0.881]. The paired difference is +0.015 [−0.020, 0.049]. Memory count alone reaches 0.830. Obs-Extract accounts for 41 of this type’s 73 errors, whereas Ext-Hybrid answers all 56 evaluated questions of this type correctly. Type stratification alone does not remove the architecture confound.

Within-system prediction and held-out systems. Separately trained within-system geometry models have modest AUROCs: Sem-RAG 0.657, Ext-Dense 0.597, Ext-Hybrid 0.635, Obs-Extract 0.552, and Svc-Mem 0.580. Training on four systems and testing on the fifth gives a mean AUROC of 0.603, with 0.467 on one held-out system. Those fits reuse question identities across training and test systems. They test an unseen system on overlapping questions, rather than joint transfer to new systems and questions.

Other signals and embedding models. For three systems with confidence elicitation, verbalized confidence has AUROC about 0.43 and P(True) 0.43 [0.393, 0.466] (n = 1,368). This indicates reversed ranking in this setting. AUROC alone does not establish probability miscalibration.

Recomputing geometry gives pooled AUROCs 0.640 (text-embedding-3-small), 0.645 (text-embedding-3-large), 0.626 (BAAI/bge-small-en-v1.5), and 0.621 (thenlper/gte-base). Their mean is 0.633, with sample standard deviation 0.012 across models. This checks pooled performance. It does not establish stability of every per-type or per-system result across embeddings.

## 5.7 Selective answering: an exploratory illustration

Ranking original out-of-fold predictions by P(correct) and retaining the top-scoring fraction raises accuracy among retained rows from 0.734 at full coverage to 0.786 at 70% coverage and 0.841 at 30%. These are retrospective calculations on the evaluation predictions, rather than a separately tuned and tested deferral policy. They do not establish a benefit over difficulty-plus-identity deferral, and no intervals for these coverage points are reported. The score requires no additional LLM generation call, but embeddings and feature computation still incur cost.

## 5.8 Nearest-turn similarity on LoCoMo

The LoCoMo analysis uses conversation turns as candidate memories, without a separate extraction step. With gpt-4o-mini generation and judging, Closest-15 accuracy is 0.501, compared with Random-15 at 0.051 and Farthest-15 at 0.003 (n = 1,986 per condition). This reproduces the proximity-ranked versus random-context ordering.

Table 6 reports AUROC for the fixed score − min<sub>t∈T</sub> d(q, t) against full-context answer labels. No logistic classifier is fitted in this category analysis, and no trained LongMemEval predictor is transferred. Single-hop reaches 0.603, while multi-hop, temporal, and open-domain have near-chance point estimates. Adversarial questions reach 0.612. Answerability can affect query relevance in that category, so we treat it as descriptive rather than separate mechanistic evidence.

The single-hop score is consistent with a weaker version of the LongMemEval association. We have only 10 conversations, and do not report conversation-level confidence intervals for the category scores. The comparison does not establish a common cause for the errors.

Table 6: LoCoMo category results for a fixed nearest-turn similarity score. n counts questions. Gaps use full-context answer correctness. These descriptive point estimates have no category confidence intervals.
<table><tr><td>LoCoMo category</td><td>n</td><td>% wrong</td><td>gap</td><td>AUROC</td></tr><tr><td>single-hop</td><td>841</td><td>17.7</td><td>+0.026</td><td>0.603</td></tr><tr><td>adversarial (no answer)</td><td>446</td><td>62.6</td><td>+0.032</td><td>0.612</td></tr><tr><td>multi-hop</td><td>282</td><td>43.3</td><td>-0.001</td><td>0.520</td></tr><tr><td>open-domain</td><td>96</td><td>65.6</td><td>-0.003</td><td>0.487</td></tr><tr><td>temporal</td><td>321</td><td>81.3</td><td>-0.007</td><td>0.468</td></tr></table>

## 6 Discussion

Interpretation. Proximity and representation features are associated with correctness, and the association varies by question type. However, system identity alone approaches the high withintype score, and geometry does not add significant pooled value beyond identity and difficulty. The intervention confirms the importance of selected context, consistent with established retrieval relevance.

For memory-system evaluation, this distinction matters: a high pooled or within-type prediction score can reflect differences between implementations, rather than detect unsupported queries within an implementation. Comparisons should therefore include system-identity and question-difficulty controls, alongside results within comparable memory representations.

The reachability gap summarizes outcomes offline. It does not certify evidence sufficiency or a live prediction’s reliability. An answer-distant reference anchor suggests a possible memory omission, but cannot verify one. A useful next study would audit whether an answer-entailing memory exists in the store and reaches the actual prompt. That could separate memory construction, retrieval, and generation errors more directly than embedding distances.

Limitations. The study covers two English benchmarks and five harness implementations. Obs-Extract exposes a whole summary as one memory text, so representation granularity differs substantially across systems. Question type does not control architecture or question-content effects. The analysis uses one grouped-CV split and fixed-prediction bootstrap intervals. Full-set imputation and question PCA expose preprocessing to held-out covariates. Feature choices were explored on this benchmark without a final independent test set or nested model selection. The intervention uses one extracted-memory system and fixed memory counts rather than equal token budgets. The referenceanchor bins use labels and thresholds, with no entailment audit. LoCoMo category estimates lack conversation-level uncertainty. API aliases were not recorded as dated model snapshots. These limits preclude claims of general deployment reliability.

Privacy and broader impacts. A validated diagnostic might help reduce unsupported answers through deferral or additional evidence gathering. An unreliable score could instead encourage false confidence or unnecessarily withhold useful answers. We have not evaluated fairness across user groups or languages. Correspondence features require access to source-turn embeddings. Deployments should enforce consent, retention limits, joint deletion of turns and derived embeddings, and access controls between users. Memory poisoning and disclosure to external embedding providers also require consideration. We evaluate existing benchmarks, without introducing a study with human participants.

Conclusion. Geometry describes memory representations and their association with answer correctness. The observed association is strongest for single-session-assistant, but system differences account for much of that result. The incremental pooled signal beyond difficulty and identity is not established. The contribution is a characterization of the evaluated signals and their limits. Verified failure attribution and deployment decisions require further study.

## References

David Carmel and Elad Yom-Tov. Estimating the Query Difficulty for Information Retrieval. Morgan & Claypool, 2010.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Steve Cronen-Townsend, Yun Zhou, and W. Bruce Croft. Predicting query performance. In Proceedings of the 25th ACM SIGIR Conference on Research and Development in Information Retrieval, 2002.

Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. RAGAS: Automated evaluation of retrieval augmented generation. In Conference of the European Chapter of the Association for Computational Linguistics (EACL): System Demonstrations, 2024.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations (ICLR), 2017.

Saurav Kadavath, Tom Conerly, Amanda Askell, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Annual Meeting of the Associationfor Computational Linguistics (ACL), 2024.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.

Peter J. Rousseeuw. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal ofComputational and Applied Mathematics, 20:53–65, 1987.

Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In European Signal Processing Conference (EUSIPCO), 2007.

Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. ARES: An automated evaluation framework for retrieval-augmented generation systems. In Conference of the North American Chapter ofthe Associationfor Computational Linguistics (NAACL), 2024.

Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9:2579–2605, 2008.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations (ICLR), 2025.

## A The 25 geometric features

All features use the embedding space over the turn set T (with session labels), the memory set $M ,$ and the query q; distances are cosine. $\varepsilon _ { p }$ is the p-th percentile of pairwise turn distances (a per-example, adaptive scale).

Table 7: The 25 scalar features, grouped by the two relationships of §1.
<table><tr><td>Family / feature</td><td>Definition</td></tr><tr><td>Correspondence: coverage (7) cov_mean_nn, cov_max_nn cov_hausdorff</td><td>mean / worst nearest-memory distance over turns</td></tr><tr><td>cov_mean_session, cov_worst_session</td><td>Hausdorff(T, M) mean / worst per-session coverage</td></tr><tr><td>cov_at_eps_p10, cov_at_eps_p5 Correspondence: alignment (2)</td><td>fraction of turns with a memory within  $\varepsilon _ { p }$ </td></tr><tr><td>mem_silhouette bridging_fraction</td><td>silhouette of M under nearest-turn session labels fraction of memories whose two nearest turns are in different sessions</td></tr><tr><td>Memory dispersion (5) mem_count mem_eff_rank</td><td>number of memories |M| effective rank of M</td></tr><tr><td>mem_mean_pairwise mem_redundancy_at_eps</td><td>mean pairwise memory distance fraction of memories with another memory within εp (p∈{5, 10})</td></tr><tr><td></td><td></td></tr><tr><td>Query reachability (6)</td><td></td></tr><tr><td>q_nn_mem q_topk_mean_mem q_memory_density, -p10, -p5</td><td>distance from q to nearest memory (core reachability) mean distance to the min(5, |M|) closest memories</td></tr></table>

## B Reproducibility details

Implementations and row coverage. Sem-RAG is the harness store rag-mem, Ext-Dense is redis-instruct, Ext-Hybrid is remis-instruct-full, Obs-Extract is mastra-om, and Svc-Mem is langmem. The latter is a local library integration, rather than a managed service. Rows are Sem-RAG 412, Ext-Dense 456, Ext-Hybrid 500, Obs-Extract 499, and Svc-Mem 500. The 133 missing rows relative to 5 × 500 reflect incomplete logged answer/judge coverage. Missingness need not be random, so run accuracies are not a fully paired system comparison. The predictor table includes 500 distinct question IDs. Per-type row/distinct-question counts are: single-session-assistant 265/56, temporal-reasoning 623/133, single-session-user 333/70, single-session-preference 137/30, multi-session 639/133, and knowledge-update 370/78.

For each example, the feature code deduplicates the memory strings in its answer-log records and treats their union as M. It does not reconstruct the exact answer-model prompt. Claims about the complete backend store or actual prompt inclusion would require a separate audit.

Classifier and preprocessing. We use scikit-learn LogisticRegression, L2 regularization, C = 1, solver=lbfgs, max\_iter=2000, and no class weighting. StratifiedGroupKFold uses five folds, shuffling, and seed 0. Each training fold fits its own standard scaler. Undefined entries are mean-imputed column-wise before splitting. There are 2,495 such entries: five features undefined for each of 499 single-summary Obs-Extract rows. Columns with no finite values are filled with zero. Per-type results fit separate models on the corresponding rows. The difficulty baseline uses type one-hot encoding, character count, and PCA-64 of question embeddings. PCA uses all unique question texts without labels, including held-out covariates. Future evaluation should fit imputation and PCA within each fold and reserve an independent final test set.

Feature choices and computation. Query top-k uses $k = 5$ . Density thresholds are the 25th, 10th, and 5th percentiles of pairwise turn distances. Coverage and redundancy use the 10th and 5th percentiles because the 25th-percentile versions saturated in exploratory analysis. The conversation GMM selects $K \in \{ 1 , . . . , \operatorname* { m i n } ( 1 0 , | T | ) \}$ by BIC, with diagonal covariance, seed 0, and reg\_ $\mathsf { c o v a r } \mathsf { = } \ 1 0 ^ { - 4 }$ . Memory effective rank is exponential entropy of normalized singular values [Roy and Vetterli, 2007]. Silhouette measurements use cosine distances [Rousseeuw, 1987]. Figure 1 uses t-SNE only for display [van der Maaten and Hinton, 2008]. Embedding inputs are capped at 24,000 characters; local models also apply their tokenizer’s context limits. We use their default encoding configuration. Embedding computation is additional work unless those representations are already retained. Pairwise-distance features have quadratic time or memory cost in the relevant set size. We do not provide complete runtime, hardware, or API-cost accounting.

Uncertainty and additional checks. Most intervals resample question groups 1,000 times from fixed out-of-fold predictions, using seed 0. The new single-session-assistant paired comparison uses the same five folds and 1,000 question-group bootstrap draws. Some original pooled baseline differences use 2,000 draws. Intervals exclude classifier refitting and exploratory feature-selection uncertainty. The embedding comparison reports sample standard deviation across four models. The pooled feature-block differences use 1,000 paired question-group draws from the same outof-fold predictions as Table 2. Leave-one-system-out fits use L2 logistic regression, $C = 1$ , and max\_iter=3000, training on four systems and testing the fifth. Question identities overlap between training and test systems.

Generation, judging, and prompts. LongMemEval runs use the gpt-4o alias for generation and judging. They were logged between March 31 and April 9, 2026. The context-selection intervention uses gpt-4o-mini and gpt-4o answer aliases, with gpt-4o-mini judging. Answer and judge calls use temperature 0. Answer prompts vary by implementation, supplying memories as context and the question as a user message. LongMemEval judging uses question-type-specific prompts. Dated model snapshots were not retained, so exact replay is not guaranteed. The linked analysis release supports offline reproduction from derived numerical artifacts. It excludes the original multi-backend harness, raw textual logs, and embedding caches. Memory construction, generation, and judging are not rerun.

LoCoMo procedure. We use 10 conversations and 1,986 questions: single-hop 841, multi-hop 282, temporal 321, open-domain 96, adversarial 446. Generation and judging use gpt-4o-mini at temperature 0. The judge uses the harness’s generic answer-correctness prompt. Category-table labels come from the full-conversation condition (rich). The fixed score is nearest-turn similarity in original space. No classifier is trained on LoCoMo labels for that table. Labels serve evaluation and the offline mean-distance gap. The intervention selects 15 source turns per query. Appropriate uncertainty would resample conversations, but category intervals were not calculated.

Data sources and LLM assistance. We use released LongMemEval and LoCoMo data and credit their creators [Wu et al., 2025, Maharana et al., 2024]. Source repositories are https:// github.com/xiaowu0162/LongMemEval and https://github.com/snap-research/locomo. LoCoMo is distributed under CC BY-NC 4.0; we use it for noncommercial research. A complete asset-license inventory is not supplied in this manuscript. LLMs serve as memory extractors, answer generators, and judges. LLM-based tools also assisted with analysis code and manuscript revision. Reported results refer to logged experiments or the stated reanalysis of their feature tables.