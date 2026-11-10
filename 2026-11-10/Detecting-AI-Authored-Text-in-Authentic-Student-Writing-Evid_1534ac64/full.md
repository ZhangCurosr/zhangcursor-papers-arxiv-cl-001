# Detecting AI-Authored Text in Authentic Student Writing: Evidence from PISA 2025

Nils-Jonathan Schaller<sup>1</sup>,

Thorben Jansen<sup>1</sup>, Olaf Köller<sup>1</sup>, Andrea Horbach<sup>1,2</sup>

<sup>1</sup>Leibniz Institute for Science and Mathematics Education, Kiel, Germany

<sup>2</sup>Kiel University, Kiel, Germany

schaller@leibniz-ipn.de

## Abstract

AI-text detectors reach near-perfect scores in evaluations that contrast fully human-written texts with texts generated from researcher designed prompts. In authentic student writing, however, AI use is voluntary and often partial. We study this setting with the PISA 2025 FLA dataset, in which 15-year-olds in Germany wrote English argumentative essays with and without access to a gpt-4o-mini-based chatbot. Keystroke, clipboard and chat logs let us trace every character of the final essay to its origin. From this provenance, we derive textand sentence-level labels for human-written, AI-generated and mixed text, and for text writ ten after chatbot use without traceable AI material. We evaluate a feature-based detector and a classifier on frozen Llama-3.1-8B embeddings, both adapted from top-ranked sharedtask systems. Both separate AI from human text reliably (macro-F1 up to 0.96 for texts and 0.95 for sentences), independently of spelling errors. The feature-based detector, however, relies on lexical complexity, which a simple student-style rewriting prompt removes (AI recall 0.92 → 0.60 with gpt-4o-mini, 0.41 with Qwen3). The embedding-based detector stays robust for whole texts (AI recall ≥ 0.94) but not for single sentences (AI recall as low as 0.59). Mixed text falls between the two classes, and text without traceable AI material is classified as human-written in about 95% of texts.

## 1 Introduction

Generative AI has become part of how students write: surveys report widespread use of chatbots for homework and writing assignments. In Germany, 57% of 12- to 19-year-olds had used ChatGPT by 2024, and two thirds of those using AI tools do so for school and homework (Feierabend et al., 2024); surveys in the US show a similar trend (Sidoti et al., 2025).

For writing assessment, this raises a validity problem: if a text may have been produced partly by a chatbot, it no longer shows reliably what the student can do, and teachers find it hard to tell students’ own writing from generated text (Cotton et al., 2024; Fleckenstein et al., 2024). Automated detectors are therefore increasingly used in education (Turnitin, 2023), although their errors are costly: a false positive amounts to an accusation of misconduct, and widely used detectors misclassify texts by non-native writers as AI-generated at high rates (Liang et al., 2023).

In published evaluations, such detectors perform very well: in the 2025 AI vs. Human shared task on academic essays, the best systems reached macro-F1 above 0.98 (Chowdhury et al., 2025). These evaluations, however, compare fully human-written texts with texts that researchers generated from carefully designed prompts. Students’ AI use in the classroom might look different, as they decide themselves whether to use a chatbot, prompt it loosely, combine its output with their own writing, or use it only for ideas. Partial use also means that AI material may be confined to individual sentences, so a useful detector should not only decide whether a text contains AI material but also where. It is unclear how detectors behave on such texts, how they treat partial AI use, and whether students could evade them with simple means.

We study these questions on the PISA 2025 FLA dataset, in which students in Germany aged 15 wrote English argumentative essays with and without access to a chatbot based on gpt-4o-mini. Keystrokes, clipboard operations and chat interactions were logged, so we can trace every character of the final essay to its origin: typed by the student, pasted from the chatbot, or retyped from it. From this provenance, we derive labels at text and sentence level that distinguish human-written (HUMAN), AI-generated (AI) and MIXED text, as well as text written after chatbot use that contains no traceable AI material (POTENTIAL\_INFLUENCE, PI). Unlike labels derived from the experimental condition, e.g. if students had access to a chatbot or not, these reflect what actually ended up in the essay.

On this data, we evaluate two complementary detectors adapted from top-ranked shared-task systems: a feature-based classifier over interpretable linguistic features (Indurthi and Varma, 2025) and a classifier over frozen Llama-3.1-8B embeddings (Jiao et al., 2025). We ask:

RQ1 How well do state-of-the-art detectors identify AI-authored text in ecologically valid student writing, at text and sentence level? How do they treat text that combines AI output with the student’s own writing, and text written after chatbot use that contains no copied or retyped AI output?

RQ2 What do the detectors rely on, and is it robust to changes a student could plausibly make, such as fixing or adding spelling errors, or asking the chatbot to make its text sound like a student’s?

Our contributions are:

• A benchmark of authentic student essays with provenance-based AI-authorship labels at text and sentence level, including partial and untraceable AI use. We will release the dataset under CC BY 4.0 and code upon publication.<sup>1</sup>

• An evaluation of two complementary top shared-task detectors on this authentic data.

• An analysis of the detectors’ robustness to realistic manipulations.

## 2 Related Work

Research on AI-generated text in education has asked two questions: how well humans, in particular teachers, can tell AI-generated from student writing, and how well automated detectors can.

Human detection. Teachers distinguish AIgenerated from student writing only moderately well. Pre-service and experienced teachers recognised fewer than half of GPT-4-generated essays (recall = 45.1% and 37.8%), and their confidence was unrelated to their accuracy (Fleckenstein et al., 2024). College instructors picked ChatGPT essays from a pair in 70% of cases (chance: 50%; Waltzer et al., 2024). Both instructors and novice EFL teachers report relying mainly on the absence of errors (Waltzer et al., 2024; De Wilde, 2024), so that well-written learner texts are sometimes taken for AI. Experiment 3 tests whether automated detectors share this reliance. Humans thus detect AI-generated text only moderately above chance, if at all, and their general confidence says little about their accuracy, which motivates automated detection.

Automated detection. On Graduate Record Examination essays, Jiang et al. (2024) detected gpt-3.5-turbo essays with accuracy above 0.98 and suggested keystroke logs for the harder case of partial AI use, which our dataset captures. The 2025 AI vs. Human shared task (Chowdhury et al., 2025) contrasted IELTS and TOEFL essays with essays generated by seven LLMs, among them gpt-4o-mini, prompted to write like non-native students; the top systems reached macro-F1 above 0.98. We adopt two complementary systems from it. Jiao et al. (2025) (rank 1 for English, macro-F1 = 0.999) classify mean-pooled hidden states of a frozen Llama-3.1-8B with a small convolutional network; their ablations attribute most of the signal to the LLM representation. Indurthi and Varma (2025) (rank 6, macro-F1 = 0.986) train classical classifiers on interpretable style, complexity, bias, affect and moral features, of which complexity features were the most discriminative.

Robustness. Such results could suggest that the task is solved, but detectors are vulnerable to changes in surface form. None of 14 tools tested by Weber-Wulff et al. (2023) exceeded an accuracy of 80%, and accuracy dropped to 42% for manually edited and 26% for paraphrased ChatGPT output. On student essays, simple perturbations, like synonym replacement, lowered the accuracy of finetuned RoBERTa detectors to 0.50–0.60 (Peng et al., 2023). Experiments 3 and 4 test related manipulations that a student could plausibly apply.

Our setting. In all these studies, the AI texts were prompted by researchers, and texts were either fully human-written or fully AI-generated; Weber-Wulff et al. (2023) explicitly left hybrid writing open. In our data, students used the chatbot voluntarily and combined its output with their own writing. Keystroke logs make every character attributable to its origin and allow labels for whole texts and single sentences, including partial (MIXED) and untraceable (PI) AI use.

## 3 Data

We use data from a writing study conducted with 2,429 15-year-old students in German secondary schools alongside the PISA 2025 assessment, as an addition to the PISA 2025 Foreign Language Assessment (OECD, 2021). Following Schaller et al. (2026), we refer to it as PISA 2025 FLA. The data comprise 4,091 writing sessions on two argumentative tasks (advertisement and teacher), taken from the MEWS study (Keller et al., 2020).

Each student wrote one essay per task — one with (Chat) and one without (NoChat) unrestricted access to a gpt-4o-mini-based chatbot for writing support (OpenAI, 2024). The assignment of conditions to tasks and the order in which the tasks were completed were randomized. Not all students completed both tasks. Writing sessions were limited to 30 minutes and logged as an event stream of keystrokes, cursor positions, clipboard operations, and the full text of all chat requests and responses.

In a naive approach, one might assign all essays from the Chat condition and all their contained sentences an AI gold standard label and label all NoChat material as human. However, as users from the Chat group could have chosen not to use the chatbot and not every instance of chatbot use is reflected in the final essay, we assign the final labels in a more informed way by first determining every character’s origin and then aggregating this information into sentence and text labels. 1,889 of the 1,981 Chat essays involved chatbot use, but only 1,134 contain AI material in the final essay (Appendix D).

## 3.1 Determining character provenance

Characters entered by keystroke are labelled as typed. Pasted material inherits the origin of the most recent copy or cut event: chat for chatbot responses, task for the task description, and typed for text from the essay itself or the student’s own chat prompts.

Since the logger occasionally misses copy events, a paste that does not match the clipboard content is compared to all chatbot responses received so far and labelled chat if its similarity is at least 0.85; otherwise it is marked as external. We measure similarity with difflib.SequenceMatcher (Python Software Foundation, 2026) after lowercasing and normalising whitespace; if one text is fully contained in the other, similarity is set to 1.

We record two additional kinds of information necessary for the later gold standard label. First, we record whether each character was inserted before or after the student’s first chatbot interaction, which separates HUMAN from POTENTIAL\_INFLUENCE text (Section 3.2).

Second, as individual events are occasionally dropped by the logger, we use a rule-based heuristic to reconstruct the missing edit from the surrounding text states for such cases; characters originating from these reconstructions are marked as inferred, everything else as observed.

## 3.2 Gold Labels

Sentence labels. We segment the final essays and the chatbot responses with pySBD (Sadvilkar and Neumann, 2020), adding a split rule for the frequent omission of whitespace after sentencefinal punctuation in learner texts. Each sentence is then labelled based on the character provenance described in Section 3.1.

All sentences written before the first chatbot interaction, and all NoChat sentences, are labelled HUMAN, unless at least 10% of their characters come from unresolved pastes (UNKNOWN\_PASTE) or at least 50% originate from the task description (TASK). Thresholds were set manually. Varying them (AI share 80–95%, unresolved-paste share 5–20%, task share 30–70%, similarity 0.80–0.95) changes the labels of at most 1% of sentences, though it affects the size of the small MIXED class.

Post-AI sentences are labelled AI if at least 90% of their characters are AI-sourced, MIXED if this share is above 0%, and POTENTIAL\_INFLUENCE otherwise (see retyping below).

Manual retyping of chatbot output is not visible at character level, as retyped characters are logged as keystrokes. We therefore compare each post-AI sentence without AI characters to all sentences of the session’s chatbot responses and label it AI if the maximum similarity (SequenceMatcher ratio) is $\geq 0 . 9 0$ and MIXED if it is $\ge ~ 0 . 8 5$ . This range captures retyped sentences with small edits. Most MIXED sentences instead arise from partial pasting. The remaining sentences are labelled UNKNOWN\_PASTE if their unresolved paste share reaches 10%, and POTENTIAL\_INFLUENCE otherwise. POTENTIAL\_INFLUENCE thus marks material written after the student started using the chatbot, but without obvious copy-pasting or retyping

of chatbot responses.

Text labels. Essays with at least 50% inferred characters are labelled UNKNOWN\_LOGGER, as a high share across a whole essay indicates unreliable logging of the session. Otherwise, the following rules apply in order: An essay is labelled AI if at least half of its sentences are AI; otherwise MIXED if it contains at least one AI or MIXED sentence; otherwise UNKNOWN\_PASTE if it contains at least one UNKNOWN\_PASTE sentence; otherwise PO-TENTIAL\_INFLUENCE if it contains at least one POTENTIAL\_INFLUENCE sentence. All remaining essays, i.e. those written entirely before chatbot use or in the NoChat condition, are labelled HUMAN. TASK sentences do not affect the text label.

## 3.3 Experimental Subset and Data Splits

We release the full labelled dataset; for our experiments we use the following subset:

Exclusions. We only use material that is not removed by any of the following exclusion criteria and that carries one of the four labels HUMAN, AI, MIXED or POTENTIAL\_INFLUENCE. At text level, we exclude essays with fewer than 10 or more than 10,000 characters, essays consisting only of non-lexical material, essays labelled UN-KNOWN\_LOGGER or UNKNOWN\_PASTE, and essays not written predominantly in English, identified semi-automatically with lingua (Stahl, 2023) and word-frequency heuristics based on wordfreq (Speer et al., 2018). At sentence level, we additionally exclude sentences labelled TASK or UN-KNOWN\_PASTE and sentences with at least 50% inferred characters. Table 7 in Appendix A reports the material remaining after each step.

Data splits. We split the texts into 80% train, 10% development and 10% test data, assigning all essays of a student to the same partition and balancing labels and writing tasks across partitions. Sentences inherit their data split assignment from the text they belong to. As texts in the train split were slightly longer, the train split on the sentence level contains more than 80% of all sentences.

Final dataset. The experimental subset comprises 3,303 essays by 2,204 students and 44,133 sentences. It contains 1,728 essays from the Chat and 1,575 from the NoChat condition. 1,662 of the essays are based on the advertisement and 1,641 on the teacher writing task. The splits comprise

<table><tr><td rowspan="2">Label</td><td colspan="2">Sentences</td><td colspan="2">Texts</td></tr><tr><td>complete</td><td></td><td>subset complete subset</td><td></td></tr><tr><td>HUMAN</td><td>23,560</td><td>18,756</td><td>1,802</td><td>1,664</td></tr><tr><td>AI</td><td></td><td>21,26617,358</td><td>1,009</td><td>952</td></tr><tr><td>POT.-INFL.</td><td>8,133</td><td>7,290</td><td>403</td><td>381</td></tr><tr><td>MIXED</td><td>872</td><td>729</td><td>321</td><td>306</td></tr><tr><td>UNKNOWN_PASTE</td><td>1,963</td><td></td><td>152</td><td></td></tr><tr><td>UNKNOWN_LOGGER TASK</td><td></td><td></td><td>226</td><td></td></tr><tr><td>Total</td><td>151 55,94544,133</td><td></td><td>3,913</td><td>3,303</td></tr></table>

Table 1: Label distribution in the complete release and in the experimental subset.

2,634 / 335 / 334 texts and 36,683 / 3,751 / 3,699 sentences for train, development, and test.

## 4 Experiments and Results

We frame AI-authorship detection as supervised classification at text and sentence level, using the same classifiers and splits (Section 3.3) throughout. Experiment 1 establishes baselines and examines the ambiguous classes MIXED and PI; Experiments 2–4 examine what the detectors rely on, through feature analysis (Exp. 2) and surface manipulations that could evade detection (Exp. 3 and 4).

## 4.1 Models

We explore two complementary systems that ranked among the top in the 2025 Shared Task: a feature-based classifier over interpretable linguistic features and a classifier over LLM embeddings. Hardware, compute, model versions and licenses, and software versions are given in Appendix E.

Feature-based. Following Indurthi and Varma (2025), we represent each text with the NELA feature set (Gruppi et al., 2020), comprising the groups Style (50 features), Complexity (7), Bias (8), Affect (9) and Moral (11), for a total of 85 features. On these we train Logistic Regression, Random Forest, Extra Trees and XGBoost (Chen and Guestrin, 2016) with scikit-learn default hyperparameters (Pedregosa et al., 2011) and a fixed random seed, deliberately reproducing the original setup.

LLM-embedding-based. Following Jiao et al. (2025), we use a frozen Llama-3.1-8B (Grattafiori et al., 2024) as a proxy model to obtain a single representation per text: input tokens are truncated to a maximum length of 1024, the last-layer hidden states are averaged across all tokens, and the resulting 4096-dimensional vector is classified by a Refined Feature Classification Network (RFCN), a compact convolutional network with three 1D convolutional layers (32, 64, 96 filters; kernel sizes 24, 16, 8) followed by three fully connected layers. The LLM is not fine-tuned; only the RFCN is trained, with AdamW, a linear warmup-decay schedule, batch size 128, learning rate 3e-4, dropout 0.1, and 20 epochs, selecting the checkpoint with the highest macro-F1 on the development set. For hyperparameters not specified in the original paper (pooling size, fully-connected layer widths) see Appendix E. Due to compute constraints we load the model in 8-bit precision rather than fp16. RFCN results are from single training runs; repeated runs differed by up to 0.02 macro-F1, so we do not interpret differences of this size between models. All non-retrained RFCN evaluations in Experiments 1, 3 and 4 use the same model.

Evaluation. Our main metric is macro-F1. We additionally report accuracy in Experiment 1 and the recall of the AI class in Experiment 4, where only AI texts are manipulated.

## 4.2 Experiment 1: Baselines

Experiment 1 establishes how well the two detectors separate AI from HUMAN text. This binary setup is the basis for all following experiments. As a secondary analysis, we examine how the detectors handle the ambiguous classes MIXED and POTEN-TIAL\_INFLUENCE (PI), the grey zone of real-world cases where AI use is partial or unattributable: Are these cases flagged as AI by a detector trained on AI and HUMAN only, and can they be separated as classes of their own when the detector is trained on them?

For the baseline, we train the classifiers on AI and HUMAN only. To test recognition, we evaluate the same models on three further test sets in which the grey-zone items are added and relabelled as AI (MIXED, PI, or both). We also apply the binary models to the PI and MIXED items alone. Since the models only know AI and HUMAN, this forces each item into one of the two, and the resulting split shows whether a model perceives these cases as more AI- or HUMAN-like.

To test learnability, we train classifiers with the grey-zone classes as their own targets, giving a three-way (AI/HUMAN/MIXED or AI/HUMAN/PI) or four-way classifier.

<table><tr><td colspan="3">Text</td><td colspan="2">Sentence</td></tr><tr><td>Test set</td><td>XGB</td><td>RFCN</td><td>XGB</td><td>RFCN</td></tr><tr><td>AI/HUMAN</td><td>0.95</td><td>0.96</td><td>0.88</td><td>0.95</td></tr><tr><td colspan="3">grey-zone items added as AI</td><td></td><td></td></tr><tr><td>+ MIXED</td><td>0.90</td><td>0.89</td><td>0.86</td><td>0.94</td></tr><tr><td>+ PI</td><td>0.81</td><td>0.84</td><td>0.76</td><td>0.81</td></tr><tr><td>+ MIXED, PI</td><td>0.79</td><td>0.80</td><td>0.76</td><td>0.81</td></tr></table>

Table 2: Macro-F1 of the binary detectors, trained on AI/HUMAN only. Top: AI-vs-HUMAN test set; bottom: grey-zone items added to the test set as AI. All models in Table 8.

AI vs. Human Baseline. On the core AI-vs-HUMAN task, both the feature-based XGBoost model and the RFCN on frozen Llama-3.1-8B embeddings are effective (Table 2). At text level they are near-identical: macro-F1 = 0.95 (XGBoost) and 0.96 (RFCN), accuracy = 0.95 and 0.97. At sentence level they diverge: XGBoost drops to macro-F1 = 0.88, consistent with less evidence per sentence, whereas the RFCN holds at macro-F1 = 0.95. Both remain below the scores of the original systems on the artificially generated shared-task essays (macro-F1 = 0.986, Indurthi and Varma, 2025; macro-F1 = 0.999, Jiao et al., 2025), as expected for authentic, student-initiated AI use. In the following, we report XGBoost, the best feature-based model, and the RFCN. See Appendix A Table 8 for results on the other models.

The grey zone: MIXED and PI. Adding the greyzone items to the test set as AI lowers performance, more so for PI than for MIXED (Table 2). XGBoost text-level macro-F1 drops from 0.95 to 0.90 with MIXED, to 0.81 with PI, and to 0.79 with both. The RFCN behaves similarly (text-level macro-F1 = 0.80 with both); at the sentence level, it remains slightly ahead of XGBoost (macro-F1 = 0.81 vs. 0.76). Applying the binary models to the grey-zone items (Table 3) shows why the detectors rarely flag these items as AI.

Both models classify almost all PI texts as HU-MAN (94.6% XGBoost, 94.7% RFCN), nearly as often as actual HUMAN texts (97.0%, 97.6%). At sentence level, XGBoost classifies more PI as AI (20.6%), the RFCN hardly more (8.8%). MIXED splits between the two classes (58.1% AI for XG-Boost, 32.3% for the RFCN at text level; only 31 test texts).

Trained as classes of their own (Table 3), the grey-zone classes are hardly separable for XG-

<table><tr><td></td><td colspan="2">Text</td><td colspan="2">Sentence</td></tr><tr><td></td><td>XGB</td><td>RFCN</td><td>XGB</td><td>RFCN</td></tr><tr><td>(a) Binary models: % predicted as AI, per true class</td></tr><tr><td>AI 91.7</td><td>94.8</td><td>85.6</td><td>94.2 5.0</td></tr><tr><td>HUMAN</td><td>3.0</td><td>2.4</td><td>11.3</td></tr><tr><td>PI</td><td>5.4</td><td>5.3</td><td>20.6 8.8</td></tr><tr><td>MIXED</td><td>58.1</td><td>32.3 52.1</td><td>45.5</td></tr><tr><td colspan="4">(b) Grey-zone classes as own targets: macro-F1</td></tr><tr><td>AI/HUMAN/MIXED</td><td>0.69</td><td>0.81</td><td>0.58 0.66</td></tr><tr><td>AI/HUMAN/PI</td><td>0.61</td><td>0.68 0.57</td><td>0.69</td></tr><tr><td>all four</td><td>0.49</td><td>0.62 0.43</td><td>0.52</td></tr></table>

Table 3: Grey-zone classes. (a) Predictions of the binary AI-vs-HUMAN models; (b) multiclass detectors trained with MIXED and/or PI as separate classes.

Boost. Its text macro-F1 falls to 0.49–0.69, while accuracy stays at 0.76–0.87, because the majority classes AI and HUMAN carry the accuracy and the added class is rarely predicted (text level: recall = 0.19 for MIXED, 0.08 for PI; 89% of PI texts are still classified as HUMAN). Part of this may be due to the small size of the grey-zone classes (306 MIXED and 381 PI vs. 1,664 HUMAN texts). The RFCN separates them better, leading XGBoost by 0.12 macro-F1 on AI/HUMAN/MIXED (0.81 vs. 0.69) and by 0.13 on the four-way task (0.62 vs. 0.49). Experiment 1 shows that both detectors separate AI from HUMAN text well, but not what this separation rests on.

## 4.3 Experiment 2: Which features carry the signal?

Experiment 2 asks which features the feature-based detector relies on, and whether they capture general differences between AI and student writing or only differences specific to one writing task. We restrict this analysis to XGBoost, as the dimensions of the RFCN’s embeddings have no interpretable meaning. We group the NELA features intoform (Style, Complexity), which describes how a text is written, and content (Affect, Bias, Moral), which is lexicon-based and tied to what a text says. Content features are the likelier source of task-specific signal: On a given task, the chatbot and the students may write about different aspects of the topic, and hence use different vocabulary, without this difference carrying over to another task.

Feature importance. We explain the text-level XGBoost detector from Experiment 1, trained on both tasks, with SHAP values (Lundberg and Lee, 2017). For every test essay, SHAP assigns each feature a value indicating how strongly and in which direction it pushed the prediction towards AI or HU-MAN. We report the mean absolute SHAP value per feature as a share of the total importance, both per feature and summed per group.

Form features account for 74.4% of the detector’s SHAP importance (Style 41.7%, Complexity 32.7%), content features for 25.6% (Affect 12.1%, Bias 9.9%, Moral 3.6%; Table 4). The importance is concentrated on a few features: average word length alone accounts for 19.5%, followed by punctuation (6.1%) and function words (4.7%); the ten most important features together account for 51.6%. The detector predicts AI for texts with longer words, more punctuation, higher readability indices (SMOG, LIX), more plural nouns and more positive words, and HUMAN for texts with more function words, adverbs and verbs. In short, AI text is lexically more complex, while student text is shorter and simpler.

Cross-task transfer. To test whether these features generalise across tasks, we train XGBoost on one writing task and evaluate it on the same task (within) and on the other task (cross). If a feature captures a task-independent difference between AI and HUMAN writing, what is learned on one task should carry over to the other; if the difference arises from the task, for example because chatbot and students address different aspects of it, performance drops. Both tasks contain a similar share of AI texts (35% and 38% in the training set), and training sets are downsampled to equal size (1,030 texts) with label proportions preserved. We train on each NELA group, on form and content combined, and on the full NELA set, and report macro-F1 within and cross, each averaged over both tasks, as well as the drop from within to cross for each transfer direction.

Within a task, all feature sets perform similarly (macro-F1 = 0.91–0.95), but they transfer differently (Table 4). Content features drop the most (by 0.17 macro-F1 on average and by 0.29 when trained on teacher), so they partly learn the task rather than the authorship. Form features transfer better (Complexity: 0.03, Style: 0.08, Style and Complexity combined: 0.10), with losses mainly when training on advertisement and testing on teacher, while content fails mainly in the opposite direction. Only the full NELA set is stable in both directions (drop ≤ 0.02), as the groups compensate for each other’s losses. With about 130 test texts per task, single per-direction differences should be read with caution. Overall, the difference between AI and HUMAN writing lies more in form than in content and largely holds across the two tasks.

<table><tr><td>Feature set</td><td>SHAP % Within</td><td>Cross</td><td></td><td>Drop</td></tr><tr><td>Form</td><td></td><td></td><td></td><td>a→t t→a</td></tr><tr><td>Style (50)</td><td>41.7</td><td>0.92</td><td>0.85</td><td>0.10 0.05</td></tr><tr><td>Complexity (7)</td><td>32.7</td><td>0.91</td><td>0.88</td><td>0.04 0.02</td></tr><tr><td>Style + Complex. (57)</td><td>74.4</td><td>0.95</td><td>0.85</td><td>0.20 0.01</td></tr><tr><td>Content (28)</td><td>25.6</td><td>0.91</td><td>0.75</td><td>0.05 0.29</td></tr><tr><td>Full NELA (85)</td><td>100</td><td>0.93</td><td>0.92</td><td>0.02 0.01</td></tr></table>

Table 4: XGBoost: share of SHAP importance and cross-task transfer (macro-F1). Drop: within minus cross per direction (a = advertisement, t = teacher).

## 4.4 Experiment 3: Spelling Manipulation

Experiment 3 tests whether the detectors use spelling errors, the most visible surface difference between student and chatbot text, as a shortcut. No NELA feature measures spelling directly, so misspellings can affect XGBoost only indirectly (e.g. through word length or part-of-speech tags), whereas the RFCN processes the raw tokens.

Manipulation. We identify misspellings automatically following Horbach et al. (2017), accepting words found in a dictionary, frequent in English, or manually verified as topic words or names, and correct them with SymSpell (details in Appendix C). Student texts contain far more misspellings than chatbot output (4.36 vs. 0.72 per 100 words in the text-level test set). In CORRECT, we correct all misspellings in HUMAN texts (4.36 → 0.01). In MISTAKE, we insert misspellings attested in HUMAN texts into AI texts, sampled by frequency, until the error rate matches human writing (0.72 → 5.06). We evaluate the baseline detectors on each manipulated test set and additionally retrain them on each manipulated variant.

Results. As expected, neither manipulation affects XGBoost: macro-F1 stays within 0.01 of the baseline throughout (Table 5). The RFCN loses 0.02 macro-F1 at text level and up to 0.03 at sentence level (MISTAKE: 0.95 → 0.92). These drops are small but significant in a paired bootstrap over students (Appendix C). They are also one-directional: inserted misspellings turn AI predictions into HUMAN (86 sentences, 0 reverse), and corrections turn HUMAN into AI (19 sentences, 1 reverse). Retrained on manipulated data, the RFCN performs as well as before. Spelling, a cue teachers rely on, is thus no shortcut for XGBoost and only a weak cue for the RFCN.

<table><tr><td rowspan="2">Train / test</td><td colspan="2">Text</td><td colspan="2">Sentence</td></tr><tr><td>XGB</td><td>RFCN</td><td>XGB</td><td>RFCN</td></tr><tr><td>orig / orig</td><td>0.95</td><td>0.96</td><td>0.88</td><td>0.95</td></tr><tr><td>orig / CORRECT</td><td>0.95</td><td>0.95</td><td>0.88</td><td>0.94</td></tr><tr><td>orig / MISTAKE</td><td>0.94</td><td>0.94</td><td>0.88</td><td>0.92</td></tr><tr><td>CORRECT / CORRECT</td><td>0.94</td><td>0.95</td><td>0.88</td><td>0.94</td></tr><tr><td>MISTAKE / MISTAKE</td><td>0.95</td><td>0.96</td><td>0.88</td><td>0.94</td></tr></table>

Table 5: Macro-F1 under the spelling manipulation. Train / test data: original (orig) or manipulated (CORRECT, MISTAKE).

## 4.5 Experiment 4: Rewriting AI text to sound like a student

Experiment 4 asks whether students could have hidden their AI use by asking the chatbot to make its text sound like their own writing. Whereas Experiment 3 manipulated spelling, we target the form features identified in Experiment 2. We rewrite all AI texts of the text-level test sets with gpt-4o-mini, the model available to the students, using two prompts. The STUDENT prompt is a simple request of the kind a student might write. The INFORMED prompt additionally lists the properties that distinguish HUMAN from AI text in Experiment 2. As it uses knowledge learned by the detector, it serves as a best case for evasion rather than a realistic scenario. To test whether the results depend on the rewriting model, we repeat the rewriting with Qwen3-14B (Yang et al., 2025) (4-bit, via Ollama<sup>2</sup>), which belongs to a different model family than both gpt-4o-mini and the Llama embeddings of the RFCN. All generations use temperature 0; HUMAN texts are not changed. Both prompts are given in Appendix B.

Manipulation check. Both gpt-4o-mini prompts moved the form features towards human writing: median average word length fell from 5.69 to 4.45 (STUDENT) and 4.39 (INFORMED; human: 4.35), and both rewrites use more pronouns and function words. Only the INFORMED prompt shortened the texts (median word count: 325 → 183; human: 163), and neither reduced punctuation nor positive sentiment. The rewrites remain semantically close to the originals (mean cosine similarity based on all-MiniLM-L6-v2 text embeddings (Reimers and Gurevych, 2019) = 0.84 for STUDENT, 0.79 for INFORMED).

Evaluation. We evaluate the Experiment 1 models on the rewritten test sets without retraining and report macro-F1 and AI recall, i.e. the share of rewritten AI items still detected. At sentence level, rewritten texts are re-segmented with pySBD; since they cannot be paired with the original sentences, all are labelled AI (including any the student had written), and we report AI recall on them.

Results. Rewriting substantially lowers XGBoost detection (Table 6): with gpt-4o-mini, its AI recall falls from 0.92 to 0.60 (STUDENT) and 0.39 (INFORMED). Even the STUDENT prompt, written without knowledge of the detector, hides about a third of detected AI texts. Qwen3-14B rewrites lower recall as much or more. The detector rests largely on form differences that a single prompt can remove.

The RFCN, in contrast, is unaffected at text level: its AI recall stays at 0.94 or above for both prompts and both rewriting models (original: 0.95). This is unlikely to be merely recognition of gpt-4o-mini’s style, which produced all AI training data: texts rewritten by Qwen3, from a different model family, are detected just as reliably. The RFCN thus captures properties of LLM-generated text that survive style-level rewriting; we leave it to future work to determine which properties these are.

Sentence level. At sentence level, rewriting affects both detectors more strongly. The AI recall of XGBoost falls from 0.86 to 0.25–0.49 across rewrites. Unlike at text level, the RFCN is also clearly affected, though less than XGBoost: its AI recall falls from 0.94 to 0.59–0.90. Here, the rewriting model matters about as much as the prompt: Qwen3 lowers AI recall more than gpt-4o-mini for both prompts (STUDENT: 0.74 vs. 0.90; IN-FORMED: 0.59 vs. 0.70). As AI recall on rewritten sentences varied by up to 0.05 between training runs, only larger differences should be interpreted. The stronger effect of Qwen3 may indicate that part of this signal is specific to gpt-4o-mini, or that Qwen3 rewrites more strongly.

## 5 Conclusion

We evaluated two state-of-the-art AI-text detectors on authentic student writing, with labels derived from keystroke, clipboard and chat logs. Regarding RQ1, AI text from loosely prompted, studentinitiated chatbot use is not substantially harder to detect than researcher-generated text. Both detectors separate AI from HUMAN reliably at text level. At sentence level, which is needed to locate AI material in partially AI-written essays, only the embedding-based detector keeps this performance; the feature-based detector drops noticeably. Partial AI use, however, does not fit the binary framing. MIXED text falls between the two classes, and the detectors disagree on where it belongs. PI text is read as human-written and stays hard to separate even when trained as a class of its own. Regarding RQ2, what the detectors rely on matters more than how well they score. The feature-based detector has mainly learned that AI text is lexically more complex. This difference holds across tasks, but a single rewriting prompt removes it. The embedding-based detector still recognises authorship of rewritten whole texts, regardless of the rewriting model, but loses a substantial share of rewritten single sentences, especially when another model does the rewriting. Spelling errors, a cue teachers report relying on, play almost no role for either detector. Text-based detection can thus reliably tell whether a whole essay was produced with a chatbot, but locating AI material at sentence level is fragile, as a single rewriting prompt is enough to hide much of it. AI support that leaves no copied or retyped text in the essay, such as using the chatbot for ideas or language help, cannot be identified from the text at all. Future work should identify which properties the embedding-based detector relies on, improve its robustness at sentence level, and examine what kinds of AI use underlie PI text.

<table><tr><td rowspan="2">Test set</td><td colspan="2">XGBoost</td><td colspan="2">RFCN</td></tr><tr><td>Rec. AI</td><td>F1</td><td>Rec. AI</td><td>F1</td></tr><tr><td colspan="5">Text level</td></tr><tr><td>original</td><td>0.92</td><td>0.95</td><td>0.95</td><td>0.96</td></tr><tr><td>4o-mini STUDENT</td><td>0.60</td><td>0.81</td><td>0.99</td><td>0.98</td></tr><tr><td>40-mini INFORMED</td><td>0.39</td><td>0.69</td><td>0.97</td><td>0.98</td></tr><tr><td>Qwen3 STUDENT</td><td>0.41</td><td>0.70</td><td>0.94</td><td>0.96</td></tr><tr><td>Qwen3 INFORMED</td><td>0.33</td><td>0.65</td><td>0.97</td><td>0.97</td></tr><tr><td colspan="5">Sentence level</td></tr><tr><td>original</td><td>0.86</td><td>0.88</td><td>0.94</td><td>0.95</td></tr><tr><td>4o-mini STUDENT</td><td>0.49</td><td>0.69</td><td>0.90</td><td>0.92</td></tr><tr><td>40-mini INFORMED</td><td>0.29</td><td>0.57</td><td>0.70</td><td>0.82</td></tr><tr><td>Qwen3 STUDENT</td><td>0.35</td><td>0.61</td><td>0.74</td><td>0.84</td></tr><tr><td>Qwen3 INFORMED</td><td>0.25</td><td>0.54</td><td>0.59</td><td>0.77</td></tr></table>

Table 6: Detection on rewritten AI test items. Rec. AI: recall of the AI class; F1: macro-F1.

## Limitations

Our gold labels rely on manually set thresholds (e.g. the 90% AI share and the 0.85/0.90 similarity thresholds for retyped text), and provenance cannot capture AI material that students paraphrased heavily or used only for ideas. All AI text was produced by gpt-4o-mini, and all writers are 15-year-old EFL students from Germany on two argumentative tasks, so results may not transfer to other generators, populations or genres. The text-level test set is small (96 AI, 169 HUMAN texts). Finally, our manipulations are simulated; real students may evade detection in other ways, such as editing the output by hand.

## Ethical considerations

AI-authorship detectors should not be used as standalone evidence of academic misconduct or for consequential decisions about individual students. False positives may lead to unjustified accusations, and our results show substantial uncertainty for mixed, rewritten, and post-AI writing. The PISA-FLA data were accessed under the data-protection and research-use procedures of the original study; analyses use pseudonymized data and do not attempt to identify individual students. Data collection followed the standard PISA consent procedures and was approved by an ethics committee and by the ministries of education of all participating federal states. Artificial Intelligence has been used for creating this work in tasks as supporting code writing, translation and editing of text. All generated code was tested before usage.

## References

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pages 785–794, San Francisco, CA, USA. ACM.

Shammur Absar Chowdhury, Hind Almerekhi, Mucahid Kutlu, Kaan Efe Kele¸s, Fatema Ahmad, Tasnim Mohiuddin, George Mikros, and Firoj Alam. 2025. GenAI content detection task 2: AI vs. human – academic essay authenticity challenge. In Proceedings ofthe 1stWorkshop on GenAI Content Detection (GenAIDetect), pages 323–333, Abu Dhabi, UAE. International Conference on Computational Linguistics.

Debby R. E. Cotton, Peter A. Cotton, and J. Reuben Shipway. 2024. Chatting and cheating: Ensuring academic integrity in the era of ChatGPT. Innovations in Education and Teaching International, 61(2):228– 239.

Vanessa De Wilde. 2024. Can novice teachers detect ai-generated texts in efl writing? ELT Journal, 78(4):414–422.

Sabine Feierabend, Thomas Rathgeb, Yvonne Gerigk, and Stephan Glöckler. 2024. JIM-Studie 2024: Jugend, Information, Medien. Basisuntersuchung zum Medienumgang 12- bis 19-Jähriger in Deutschland. Technical report, Medienpädagogischer Forschungsverbund Südwest (mpfs), Stuttgart. Accessed 8 October 2026.

Johanna Fleckenstein, Jennifer Meyer, Thorben Jansen, Stefan D Keller, Olaf Köller, and Jens Möller. 2024. Do teachers spot ai? evaluating the detectability of ai-generated texts among student essays. Computers and Education: Artificial Intelligence, 6:100209.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Maurício Gruppi, Benjamin D. Horne, and Sibel Adalı. 2020. Nela-gt-2019: A large multi-labelled news dataset for the study of misinformation in news articles. Preprint, arXiv:2003.08444.

Andrea Horbach, Yuning Ding, and Torsten Zesch. 2017. The influence of spelling errors on content scoring performance. In Proceedings ofthe 4th Workshop on Natural Language Processing Techniquesfor Educational Applications (NLPTEA 2017), pages 45–53, Taipei, Taiwan. Asian Federation of Natural Language Processing.

Vijayasaradhi Indurthi and Vasudeva Varma. 2025. Tesla at GenAI detection task 2: Fast and scalable method for detection of academic essay authenticity. In Proceedings of the 1stWorkshop on GenAI Content Detection (GenAIDetect), pages 317–322, Abu Dhabi, UAE. International Conference on Computational Linguistics.

Yang Jiang, Jiangang Hao, Michael Fauss, and Chen Li. 2024. Detecting chatgpt-generated essays in a largescale writing assessment: Is there a bias against nonnative english speakers? Computers & Education, 217:105070.

Kaijie Jiao, Xingyu Yao, Shixuan Ma, Sifan Fang, Zikang Guo, Benfeng Xu, Licheng Zhang, Quan Wang, Yongdong Zhang, and Zhendong Mao. 2025. CMI-AIGCX at GenAI detection task 2: Leveraging multilingual proxy LLMs for machine-generated text detection in academic essays. In Proceedings of the

1stWorkshop on GenAI Content Detection (GenAIDetect), pages 290–298, Abu Dhabi, UAE. International Conference on Computational Linguistics.

Stefan D. Keller, Johanna Fleckenstein, Maleika Krüger, Olaf Köller, and André A. Rupp. 2020. English writing skills of students in upper secondary education: Results from an empirical study in switzerland and germany. Journal of Second Language Writing, 48:100700.

Weixin Liang, Mert Yuksekgonul, Yining Mao, Eric Wu, and James Zou. 2023. GPT detectors are biased against non-native English writers. Patterns, 4:100779.

Scott Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. Preprint, arXiv:1705.07874.

OECD. 2021. Pisa 2025 foreign language assessment framework. Technical report, OECD Publishing, Paris.

OpenAI. 2024. GPT-4o mini: Advancing cost-efficient intelligence. https://openai.com/index/gpt-4 o-mini-advancing-cost-efficient-intellige nce/. Blog post, published July 18, 2024. Accessed: 2026-10-09.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Xinlin Peng, Ying Zhou, Ben He, Le Sun, and Yingfei Sun. 2023. Hidding the ghostwriters: An adversarial evaluation of ai-generated student essay detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10406–10419.

Python Software Foundation. 2026. difflib — helpers for computing deltas. https://docs.python.org/ 3/library/difflib.html. Python 3 documentation.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Nipun Sadvilkar and Mark Neumann. 2020. PySBD: Pragmatic sentence boundary disambiguation. In Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS), pages 110–114, Online. Association for Computational Linguistics.

Nils-Jonathan Schaller, Daniel Mora Melanchthon, Thorben Jansen, Olaf Köller, and Andrea Horbach. 2026. KEYSCORE — keystroke-enhanced automated essay scoring. In Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026), pages 347– 357, San Diego, California, USA. Association for Computational Linguistics.

Olivia Sidoti, Eugenie Park, and Jeffrey Gottfried. 2025. About a quarter of U.S. teens have used ChatGPT for schoolwork – double the share in 2023. Pew Research Center. Accessed 8 October 2026.

Robyn Speer, Joshua Chin, Andrew Lin, Sara Jewett, and Lance Nathan. 2018. Luminosoinsight/wordfreq: v2.2.

Peter M. Stahl. 2023. Lingua: The most accurate natural language detection library for java and other jvm languages, ported to python. https://github.com /pemistahl/lingua-py.

Turnitin. 2023. Turnitin turns on AI writing detection capabilities for educators and institutions. Press release. Accessed 8 October 2026.

Tal Waltzer, Celeste Pilegard, and Gail D Heyman. 2024. Can you spot the bot? identifying ai-generated writing in college essays. International Journal for Educational Integrity, 20(1):11.

Debora Weber-Wulff, Alla Anohina-Naumeca, Sonja Bjelobaba, Tomáš Foltynek, Jean Guerrero-Dib, Olu-\` mide Popoola, Petr Šigut, and Lorna Waddington. 2023. Testing of detection tools for ai-generated text. International Journal for Educational Integrity, 19(1):26.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

## A Additional Tables

<table><tr><td>Step</td><td>Texts</td><td>Sentences</td></tr><tr><td>Logged sessions</td><td>4,091</td><td></td></tr><tr><td>Text level — empty / length outliers</td><td>3,913</td><td>55,945</td></tr><tr><td>— UNK. textsº a</td><td>3,535</td><td>50,138</td></tr><tr><td>– non-English</td><td>3,303</td><td>46,076</td></tr><tr><td>Sentence level</td><td></td><td></td></tr><tr><td>- ≥ 50% inferred</td><td>3,303</td><td>44,358</td></tr><tr><td></td><td></td><td></td></tr><tr><td>— TASK / UNK._PASTE</td><td>3,303</td><td>44,133</td></tr></table>

Table 7: From logged sessions to the experimental subset; each row reports the material remaining after the step. <sup>a</sup>Texts labelled UNKNOWN\_LOGGER or UN-KNOWN\_PASTE.

<table><tr><td rowspan="2">Model</td><td colspan="2">Text</td><td colspan="2">Sentence</td></tr><tr><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td></tr><tr><td>Logistic Regression</td><td>0.94</td><td>0.95</td><td>0.85</td><td>0.85</td></tr><tr><td>Random Forest</td><td>0.93</td><td>0.94</td><td>0.87</td><td>0.87</td></tr><tr><td>Extra Trees</td><td>0.93</td><td>0.94</td><td>0.87</td><td>0.87</td></tr><tr><td>XGBoost</td><td>0.95</td><td>0.95</td><td>0.88</td><td>0.88</td></tr><tr><td>RFCN</td><td>0.96</td><td>0.97</td><td>0.95</td><td>0.95</td></tr></table>

Table 8: Core AI-vs-HUMAN detection, macro-F1 and accuracy. All five models are shown; the grey-zone and multiclass analyses below use the two headline models (XGBoost, RFCN).

## B Rewriting Prompts

Table 9 shows the two prompts used in Experiment 4 (Section 4.5). Both were used unchanged with gpt-4o-mini and Qwen3-14B.

## C Spelling Manipulation (Experiment 3)

A word counts as correctly spelled if it is listed in the Hunspell en\_US dictionary<sup>3</sup> or is frequent in English according to wordfreq (Speer et al., 2018) (Zipf frequency ≥ 3.5, i.e. about three occurrences per million words); the latter accepts common British spellings, which en\_US rejects. Topic words and names are often missing from both resources and would be flagged as errors. We therefore also accept words from our corpus, under conditions that keep out recurring learner misspellings: a word must occur in at least five texts and under at least two labels, so that a single student repeating a misspelling does not count as evidence; it must not be German; and it must not be one edit away from a frequent English word (Zipf ≥ 4), unless it is mostly capitalised, i.e. a name. This yields 290 candidate types, of which we manually accepted 32 as correct spellings (mainly topic words and names). Frequent German words and contractions without an apostrophe (dont) are not counted as errors. Frequent misspellings were corrected by hand at type level, all others with the nearest dictionary word suggested by SymSpell<sup>4</sup> (Damerau– Levenshtein distance ≤ 3). In the text-level test set, CORRECT alters 151 of 169 HUMAN texts and MIS-TAKE alters 95 of 96 AI texts; the MISTAKE error rate overshoots human writing because of errors already present in the AI texts.

<table><tr><td>STUDENT</td><td>Rewrite the following text so that it sounds like it was written by a 15-year-old German student in English class. Only output the rewritten text.</td></tr><tr><td>INFORMED</td><td>Rewrite the following essay so that it reads like the work of a 15-year-old German student writing in English as a foreign language. Keep the same position and main arguments, but change the style: • Make it considerably shorter, about 150–200 words. • Use short, common words instead of long or formal ones. • Use simple sentences and everyday vocabulary; repeat words rather than using synonyms. • Write in a personal way, with many pronouns (I, we, you, they) and verbs rather than nouns. • Use little punctuation: mostly full stops and a few commas, no semicolons, colons, dashes or quotation marks. • Keep a neutral tone; avoid enthusiastic or strongly positive wording. • Do not add spelling or grammar mistakes. Only</td></tr></table>

Table 9: Rewriting prompts; the essay is appended to each prompt.

Table 10 compares the RFCN under each spelling scenario of Experiment 3 (Section 4.4) with the unmanipulated baseline on the same test items.

## D Chatbot Use in the Chat Condition

Table 11 shows how often students in the Chat condition used the chatbot and how often its output ended up in the final essay (Section 3).

## E Implementation Details

Models. We use the base model Llama-3.1-8B (Llama 3.1 Community License), loaded in 8-bit precision with bitsandbytes due to GPU memory constraints. As no code was released, we reimplemented the RFCN following Jiao et al. (2025); for unspecified hyperparameters we use max-pooling (size 2) after each convolutional layer and fully connected layers of width 128 and 64 before the output layer, with ReLU activations and dropout before and after the first fully connected layer. We use a 10% linear warmup, AdamW with default weight decay (0.01), unweighted cross-entropy, and seed 42. Embeddings are mean-pooled over non-padding tokens and stored in float16. For Experiment 4 we use gpt-4o-mini (gpt-4o-mini-2024-07-18, OpenAI Terms of Use) and Qwen3-14B (14.8B parameters, Q4\_K\_M quantization, Ollama 0.40.2, thinking disabled, Apache 2.0). No hyperparameter search was performed.

<table><tr><td>Train / test</td><td>∆F1</td><td>95% CI</td><td>Flips</td></tr><tr><td>Text level</td><td></td><td></td><td></td></tr><tr><td>orig / CORRECT</td><td></td><td>-0.016[-0.032, -0.004]*</td><td>0/4</td></tr><tr><td>orig / MISTAKE</td><td>-0.025</td><td>[-0.048, -0.008]*</td><td>6/0</td></tr><tr><td>CORRECT / CORRECT</td><td>-0.012</td><td>[-0.034, 0.008]</td><td>2/5</td></tr><tr><td>MISTAKE / MISTAKE</td><td>-0.004</td><td>[-0.021, 0.013]</td><td>3/2</td></tr><tr><td>Sentence level</td><td></td><td></td><td></td></tr><tr><td>orig / CORRECT</td><td></td><td>-0.006[-0.012, -0.002]*</td><td>1/ 19</td></tr><tr><td>orig / MISTAKE</td><td>-0.028</td><td>[-0.037,-0.020]*</td><td>86 / 0</td></tr><tr><td>CORRECT / CORRECT</td><td>-0.003</td><td>[-0.010, 0.003]</td><td>20 / 37</td></tr><tr><td>MISTAKE / MISTAKE</td><td>-0.005</td><td>[-0.012, 0.003]</td><td>70 / 32</td></tr></table>

Table 10: RFCN under the spelling manipulation, compared with orig / orig on the same test items (265 texts, 3,067 sentences). ∆F1: difference in macro-F1. 95% CI: paired bootstrap over students (2,000 resamples); \* = CI excludes zero. Flips: items whose prediction changed, AI→HUMAN / HUMAN→AI.
<table><tr><td></td><td>Texts</td><td>Sentences</td></tr><tr><td>Chat condition</td><td>1,981</td><td>31,936</td></tr><tr><td>chatbot used</td><td>1,889</td><td></td></tr><tr><td>AI material in final essay</td><td>1,134</td><td></td></tr><tr><td>Retyping suspects</td><td></td><td>572</td></tr></table>

Table 11: Chatbot use in the Chat condition of the complete release (3,913 texts, before exclusions). Retyping suspects: post-AI sentences without AI-sourced characters that match a chatbot sentence with similarity ≥ 0.85.

Compute. All local experiments ran on a single NVIDIA RTX 3060 (12 GB). Extracting Llama-3.1-8B embeddings, training all RFCN variants and rewriting with Qwen3-14B took approximately 6 GPU hours in total, most of it for embedding extraction. Rewriting with gpt-4o-mini used the OpenAI API.

Software. Table 12 lists software versions.

<table><tr><td>Package</td><td>Version</td><td>Used for</td></tr><tr><td>Python</td><td>3.11.16</td><td></td></tr><tr><td>nela-features</td><td>3.0.1</td><td>NELA features</td></tr><tr><td>scikit-learn</td><td>1.7.1</td><td>LR, RF, ET</td></tr><tr><td>xgboost</td><td>3.2.0</td><td>XGBoost</td></tr><tr><td>shap</td><td>0.51.0</td><td>Exp. 2</td></tr><tr><td>pysbd</td><td>0.3.4</td><td>Segmentation</td></tr><tr><td>lingua</td><td>2.1.1</td><td>Language ID</td></tr><tr><td>wordfreq</td><td>3.1.1</td><td>Language ID, spelling</td></tr><tr><td>symspellpy</td><td>6.10.0</td><td>Exp. 3</td></tr><tr><td>sentence-transformers</td><td>6.1.0</td><td>Exp. 4</td></tr><tr><td>torch</td><td>2.5.1</td><td>RFCN</td></tr><tr><td>transformers</td><td>5.17.0</td><td>Llama embeddings</td></tr><tr><td>accelerate</td><td>1.15.0</td><td>Model loading</td></tr><tr><td>bitsandbytes</td><td>0.50.2</td><td>8-bit loading</td></tr><tr><td>Ollama</td><td>0.40.2</td><td>Qwen3 (Exp. 4)</td></tr></table>

Table 12: Software versions.