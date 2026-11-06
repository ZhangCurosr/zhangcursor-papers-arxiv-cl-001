# Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes

Vladimir Beskorovainyi Besk Tech https://besk.tech

November 2026

## Abstract

Typed-decision models score options that the caller supplies at request time, so their accuracy depends on label wording that the model’s developer does not control. We compare two releases of a 307M-parameter multilingual cross-encoder, Ru-Jev, trained on thirteen Russian-language tasks; the later release adds an augmentation that presents labels in several surface forms. Holding questions fixed and varying only how eighteen routing labels are worded, we evaluate ten label vocabularies, produced by three separate processes and checked for overlap with the training data, with five candidates per question and, for eight of them, also with eighteen. On held-out vocabularies made of ordinary or borrowed words (English-derived jargon, ofice slang, department names), the augmented release is more accurate in every setting, by up to 0.24. Eleven of these fourteen gains survive a Holm correction (slang with eighteen candidates and descriptive sentences with five do not), and for jargon on all labels the label-level interval excludes zero in both settings. On abbreviations and opaque codes it gives no usable gain: on abbreviations it is not better, and worse on the clean slice, and on opaque codes both releases stay at or below chance once labels related to training options are removed. In most cells where accuracy rises, the risk–coverage area falls, but with eighteen candidates confident errors rise on jargon and slang. On two datasets neither release saw, accuracy moves by −8.4 and +5.6 points. We also report four evaluation hazards from comparing the two releases.

## 1 Introduction

Text classifiers have historically fixed their label set at training time. A growing class of interfaces does not: the caller supplies the candidate options at request time, in the caller’s own words. This covers zero-shot classification by entailment (Yin et al., 2019), prompt-based classification with verbalizers (Schick and Schütze, 2021), structured output from instruction-following models, and compact typed-decision models that return a decision of a declared type with a probability and generate no free text (Deußer et al., 2026; Wang et al., 2026; Convai Innovations, 2026).

When the label set arrives at request time, accuracy becomes a function of the strings the caller chose. Benchmarks usually score models on the label names the dataset authors wrote, while deployments often use names the organisation already has: department names, internal codes, abbreviations and borrowings from English. That label wording afects behaviour is established. Webson and Pavlick (2022) find prompt-based models far more sensitive to the choice of target words than to the instruction; Gao et al. (2021) search over label words because the choice matters; Holtzman et al. (2021) identify surface form competition in multiple-choice scoring; Gao et al.

(2023) show that training on label descriptions improves robustness to verbalizer choice. In vision, robustness gained from one family of corruptions often fails to transfer to others (Geirhos et al., 2018).

We ask a practical question. If training presents each label in several surface forms, which kinds of unfamiliar label wording does the model then handle better? We answer it with probes that hold the questions fixed and vary only the wording, at two candidate-set sizes, with every probe label checked against the training data.

Our contributions are:

1. A controlled measurement over ten label vocabularies, two candidate-set sizes and three probe sources, with exact paired tests, label-level resampling and a multiple-comparison correction (Sections 5 and 6).

2. The finding that label-wording augmentation improves accuracy on held-out vocabularies of ordinary and borrowed words, but not on abbreviations or opaque codes, and that a probe at a single candidate-set size can misstate the efect (Section 6).

3. Out-of-distribution and cross-model results against an open model on the same backbone (Section 7).

4. Four hazards in release-to-release comparison, each with its measured efect on numbers we had prepared for publication (Section 9).

## 2 Typed decisions

A typed decision is a triple $( c , O , t )$ : a context $c ,$ a set of candidate options $O = \{ o _ { 1 } , \ldots , o _ { k } \}$ , and a declared type $t \in$ {boolean, choice, scale}. The model returns a distribution over �.

Ru-Jev<sup>1</sup> is a pointwise cross-encoder, the same family as entailment-based zero-shot classifiers and passage rerankers. Each pair is scored independently and the scores are normalised across options. With an encoder $f _ { \theta }$ , a linear head � and $h ( c , o ) = f _ { \boldsymbol { \theta } } ( c , o ) _ { \mathrm { [ C L S ] } }$ 2

$$
p ( o _ { i } \mid c , O ) = \frac { \exp \Big ( w ^ { \top } h ( c , o _ { i } ) / T \Big ) } { \sum _ { j = 1 } ^ { k } \exp \big ( w ^ { \top } h ( c , o _ { j } ) / T \big ) } ,\tag{1}
$$

where $T$ is a single scalar temperature fitted after training (Guo et al., 2017), a one-parameter variant of Platt scaling (Platt, 2000). Training uses cross-entropy over the options of each example.

Two properties follow. The output does not depend on the order of the options, since no option sees another. And because each option’s score is computed independently, the distribution over any subset of a candidate set is obtained exactly by renormalising that subset’s scores; we use this to evaluate the same probe at five and at eighteen candidates from one scoring pass. The price is cost linear in �. Models that read all options in one pass, such as Laya (Convai Innovations, 2026), trade order invariance for lower cost at large � (Section 8).

## 3 Model and data

The encoder is mmBERT-base (Marone et al., 2026), 307M parameters of which 110M are outside the embedding matrix, released under the MIT licence. Inputs are truncated to 256 tokens.

Training uses thirteen tasks built from open Russian-language datasets: six tasks from RussianSuperGLUE (Shavrina et al., 2020), RuCoLA (Mikhailov et al., 2022), one reasoning task pooling the ruOpenBookQA and ruWorldTree sets of MERA (Fenogenova et al., 2024), four classification tasks from ruMTEB (Snegirev et al., 2025) (review sentiment, review rating, and the intent and scenario tasks of MASSIVE (FitzGerald et al., 2023)), and a toxicity set (Dementieva et al., 2021). The mix contains 143,991 training and 26,746 validation decisions. Each source keeps its own licence; the repository lists them per source, including one whose README asks for research use while its licence file is an unmodified MIT licence.

Training runs four epochs with a checkpoint after each, and the third is released (stored zeroindexed as epoch 2). The fourth overfits: training loss falls to 0.085 while validation loss rises to 0.820 (Table 1), and a temperature fitted on its outputs reaches 3.92, against 1.864 for the released third epoch. Epoch selection and temperature fitting both used the validation split, which includes the MASSIVE validation rows; the datasets of Section 7 played no role in model selection.

<table><tr><td>Epoch</td><td>Training loss</td><td>Validation loss</td><td>Validation accuracy</td></tr><tr><td>1</td><td>0.6029</td><td>0.5054</td><td>0.7536</td></tr><tr><td>2</td><td>0.4336</td><td>0.4576</td><td>0.7920</td></tr><tr><td>3</td><td>0.2714</td><td>0.4831</td><td>0.8181</td></tr><tr><td>4</td><td>0.0849</td><td>0.8204</td><td>0.8139</td></tr></table>

Table 1: Training trajectory of release v1.3. The third epoch is released.

## 4 The two releases

Release v1.1 presents each MASSIVE scenario label as its English key and as an ordinary Russian word; in the rebuilt v1.1 training mix both forms occur for all 18 scenarios, and Russian-phrase strings occur for two. Release v1.3 stores several surface forms per label and draws one uniformly per training example. As specified in the data builder:

• MASSIVE intents (60 labels): the English key, an ordinary Russian label, the key transliterated into Cyrillic, and a clipped form of that transliteration produced by rule;

• MASSIVE scenarios (18 labels): the English key, an ordinary Russian word, and a Russian phrase;

• toxicity: five wordings of the two labels, of which v1.1 used one;

• review sentiment: five wordings of the label set, and review rating: four, where v1.1 used one each.

Transliterated and clipped forms are never applied to scenario labels.

The releases difer in more than the augmentation. Between them the data builder was changed to seed validation carves by task (Section 9), which repartitioned several tasks, and an intermediate release, v1.2, changed other details. Architecture and objective are the same, but v1.1 was trained for three epochs and v1.3 for four (with the third released), so their learning-rate schedules difer. We therefore attribute diferences to the release, and to the augmentation only where the probe design isolates it.

## 5 Evaluation design

Questions. All wording probes use the same 1,000 questions from the source validation split of the MASSIVE scenario task (7 are duplicates). Neither release trained on these rows, but both used them for epoch selection and temperature fitting. Each question has one gold scenario among 18.

Two candidate-set sizes. With five candidates, each question is shown with the gold label and four distractors drawn deterministically from the other 17; the distractor labels are identical across all vocabularies and models (chance 0.20). With eighteen, every question is shown with all labels (chance 0.056). Absolute accuracies and efect sizes are reported for both settings and never compared across them.

Vocabularies. Each vocabulary renames all 18 scenarios.

• Card bench, six vocabularies written during development of the model, before these experiments: English key, ordinary Russian word, Russian phrase, English-derived jargon (вэзер, смолток, фуд-доставка, логистика), department names (метеослужба) and descriptive sentences. The jargon vocabulary was committed 6.5 hours before the transliteration rules of Section 4: it was not fitted to them, but it existed when they were written.

• Rule-derived probe: code identifiers such as ALARM\_SCN and three-letter Cyrillic abbreviations such as БУД, generated by rule from existing labels by the AI coding agent that trained the model, and scored with five candidates only.

• Independent probe: ofice slang (синоптик, побудка) and opaque codes (ПГД, БДК), written by a separate AI assistant session that had not trained the model and had seen no scores for these vocabularies, and passed on verbatim by the author.

Overlap with training. Every probe label was compared with every option string in the rebuilt training mixes of both releases, by exact equality and by prefix in both directions, for gold labels and distractors. Two card-bench vocabularies are training styles: the English key and the Russian word occur in both releases’ training data for all 18 labels, so they serve as controls, not probes. The Russian phrase occurs in v1.3’s training data for all 18 labels and in v1.1’s for two (one of which is identical to the Russian word), so its gain is expected by construction. The departmentname and descriptive vocabularies have no exact match in either release, but all descriptive labels and two department labels have a prefix relation to some training option, so neither has a strict slice; the jargon vocabulary has three exact matches in v1.3’s training data (audio, news, weather), which its clean slice removes. Slices: for jargon, clean drops the three labels that coincide with the output of the transliteration rule (audio, news, weather; 15 labels, 882 questions) and strict drops every label with an exact or prefix relation to a training option (11 labels, 625 questions). For the independent probe, strict drops labels with an exact or prefix relation (slang 14 labels, codes 9); for abbreviations, clean drops the one exact match. Filtering applies to the gold label; distractors are not filtered.

Reading rule. For the independent probe, a rule for reading the eighteen-candidate slang comparison on the strict slice was recorded at 08:53 local time on 5 November 2026, after scoring had started and after the v1.1 score file had been written, but before any score file was opened. The rule said: v1.3 higher with $p < 0 . 0 5$ means the gain transfers to slang. File hashes are in the repository. We do not call this pre-registration, and the five-candidate analysis of this probe was added afterwards.

<table><tr><td></td><td></td><td></td><td colspan="4">Five candidates (chance 0.20)</td><td colspan="4">Eighteen candidates (chance 0.056)</td></tr><tr><td>Vocabulary, slice</td><td>Labels</td><td>n</td><td>v1.1</td><td>v1.3</td><td>∆</td><td>95% CI, labels</td><td>v1.1</td><td>v1.3</td><td>∆</td><td>95% CI, labels</td></tr><tr><td>English key†</td><td>18</td><td>1000</td><td>0.957</td><td>0.962</td><td>+0.005</td><td>[−0.01, +0.02]</td><td>0.894</td><td>0.900</td><td>+0.006</td><td>[−0.02, +0.03]</td></tr><tr><td>Russian word†</td><td>18</td><td>1000</td><td>0.956</td><td>0.960</td><td>+0.004</td><td>[−0.01, +0.02]</td><td>0.883</td><td>0.892</td><td>+0.009</td><td>[−0.02, +0.04]</td></tr><tr><td>Russian phrase</td><td>18</td><td>1000</td><td>0.881</td><td>0.957</td><td>+0.076</td><td>[+0.02, +0.14]</td><td>0.703</td><td>0.885</td><td>+0.182</td><td>[+0.04, +0.34]</td></tr><tr><td>English-derived jargon, all</td><td>18</td><td>1000</td><td>0.633</td><td>0.855</td><td> $+ 0 . 2 2 2 ^ { \ast \ast \ast }$ </td><td>[+0.09, +0.37]</td><td>0.431</td><td>0.640</td><td>+0.209***</td><td>[+0.05, +0.40]</td></tr><tr><td>English-derived jargon, clean</td><td>15</td><td>882</td><td>0.624</td><td>0.838</td><td> $+ 0 . 2 1 4 ^ { * * * }$ </td><td>[+0.07, +0.40]</td><td>0.427</td><td>0.595</td><td>+0.168***</td><td>[+0.02, +0.34]</td></tr><tr><td>English-derived jargon, strict</td><td>11</td><td>625</td><td>0.550</td><td>0.786</td><td>+0.235***</td><td>[+0.05, +0.45]</td><td>0.328</td><td>0.470</td><td>+0.142***</td><td>[−0.04, +0.38]</td></tr><tr><td>Department names</td><td>18</td><td>1000</td><td>0.835</td><td>0.869</td><td>+0.034*</td><td>[−0.03, +0.12]</td><td>0.632</td><td>0.733</td><td>+0.101***</td><td>[−0.00, +0.24]</td></tr><tr><td>Descriptive sentences</td><td>18</td><td>1000</td><td>0.939</td><td>0.958</td><td>+0.019</td><td>[−0.00, +0.04]</td><td>0.848</td><td>0.878</td><td>+0.030*</td><td>[+0.00, +0.06]</td></tr><tr><td>Office slang, all</td><td>18</td><td>1000</td><td>0.648</td><td>0.780</td><td> $+ 0 . 1 3 2 ^ { * * * }$ </td><td>[+0.01, +0.26]</td><td>0.428</td><td>0.476</td><td>+0.048</td><td>[−0.11, +0.21]</td></tr><tr><td>Office slang, strict</td><td>14</td><td>773</td><td>0.595</td><td>0.761</td><td> $+ 0 . 1 6 6 ^ { * * * }$ </td><td>[+0.02, +0.32]</td><td>0.369</td><td>0.419</td><td>+0.050</td><td>[−0.15, +0.24]</td></tr><tr><td>Code identifiers</td><td>18</td><td>1000</td><td>0.856</td><td>0.889</td><td>+0.033*</td><td>[−0.01, +0.09]</td><td></td><td></td><td>not scored</td><td></td></tr><tr><td>Abbreviations, all</td><td>18</td><td>1000</td><td>0.403</td><td>0.379</td><td>-0.024</td><td>[−0.18, +0.17]</td><td></td><td></td><td>not scored</td><td></td></tr><tr><td>Abbreviations, clean</td><td>17</td><td>959</td><td>0.420</td><td>0.360</td><td>-0.060*</td><td>[−0.20, +0.12]</td><td></td><td></td><td>not scored</td><td></td></tr><tr><td>Opaque codes, all</td><td>18</td><td>1000</td><td>0.265</td><td>0.270</td><td>+0.005</td><td>[−0.08, +0.09]</td><td>0.104</td><td>0.146</td><td>+0.042**</td><td>[−0.01, +0.10]</td></tr><tr><td>Opaque codes, strict</td><td>9</td><td>512</td><td>0.135</td><td>0.168</td><td>+0.033</td><td>[−0.04, +0.13]</td><td>0.018</td><td>0.061</td><td>+0.043**</td><td>[+0.01, +0.09]</td></tr></table>

Table 2: Accuracy of the two releases on the same 1,000 questions, varying only label wording. Δ is v1.3 minus v1.1. Stars give Holm-adjusted exact McNemar � over the 23 non-control cells: $^ \ast < 0 . 0 5$ $^ { * * } < 0 . 0 1 , ^ { * * * } < 0 . 0 0 1$ . The label-level interval resamples labels, then questions. <sup>†</sup>Training style for both releases (control). <sup>‡</sup>Training style for v1.3 only. The rule-derived probe (code identifiers, abbreviations) was scored with five candidates only. Raw McNemar counts and per-label results are in the repository.

Statistics. For each cell we report the exact McNemar test on paired answers and a 95% interval from a two-stage bootstrap that resamples labels and then questions within labels, since a label, not a question, is the unit that generalises. With 9 to 18 labels per cell these intervals are wide and approximate. �-values are Holm-adjusted over 23 cells: all cells of Table 2 except the six training-style controls (English key, Russian word and Russian phrase, in both settings), plus the two unseen-dataset comparisons of Section 7. Confidence is evaluated with the area under the risk–coverage curve (AURC, lower is better), the gap between mean confidence and accuracy, and the number of wrong answers above 0.8 confidence per 1,000 questions. Ru-Jev probabilities use the released temperatures $( T = 2 . 2 7 5 4$ for v1.1, 1.8640 for v1.3).

## 6 Which wordings the augmented release gains on

Ordinary and borrowed words. On every held-out vocabulary made of words, the augmented release is more accurate in both settings (Table 2). The largest gains are on English-derived jargon: +0.235 with five candidates and +0.142 with eighteen on the strict slice, which excludes every label with an exact or prefix relation to a training option. Ofice slang gains +0.166 with five candidates (label-level interval above zero) and +0.050 with eighteen; the recorded reading rule for the latter returns “transfer” (raw $p = 0 . 0 1 6 )$ , but the gain does not survive the Holm correction and its labellevel interval spans zero. Department names gain +0.034 and +0.101, descriptive sentences +0.019 and +0.030. The direction is the same in all fourteen word-vocabulary cells; the size depends on the candidate-set size in diferent ways for diferent vocabularies, so a probe at one size alone can misstate the efect. On the strict slice with eighteen candidates the slang result is a reshufle as much as a lift: 5 labels improve and 6 get worse, some by half the scale (справочное бюро 0.000 to 0.585, перечни 0.534 to 0.086).

Abbreviations and codes. On three-letter abbreviations v1.3 is not better: −0.024 on all labels (not significant) and −0.060 on the clean slice, a diference created by one excluded label on which v1.3 gains 0.829. Opaque codes are essentially unreadable for both releases once overlapping labels are removed: on the strict slice with eighteen candidates v1.1 is below chance and v1.3 at chance, and with five candidates neither difers from the other. Code identifiers such as ALARM\_SCN contain the trained key and are easy for both, so we do not treat them as a test of robustness.

Confidence. In most cells where accuracy improves, the risk–coverage area falls, so a caller who thresholds on confidence is better served (among non-control cells, the only exception is slang on all labels with eighteen candidates, 0.354 to 0.395): on strict jargon AURC goes from 0.235 to 0.078 with five candidates and from 0.452 to 0.418 with eighteen; on strict slang from 0.223 to 0.121 with five candidates and is unchanged (0.476 against 0.470) with eighteen. The count of confident errors tells a less favourable story. With five candidates it falls on jargon (110 to 62 per 1,000 on the strict slice). With eighteen it rises on jargon (79 to 109 on all labels) and on slang (151 to 191 on the strict slice), consistent with the augmented release being more confident overall, while it falls on department names (123 to 87). On abbreviations it rises (77 to 118 on the clean slice). The augmentation therefore does not, by itself, make confidence safer on unfamiliar wording; a deployment should refit the temperature on its own labels.

Reading. The release trained on several surface forms per label is more accurate on unfamiliar labels that are words, ordinary or borrowed, including words of kinds it never saw in training (slang, department names). It gives no usable gain on abbreviations or opaque codes, where the label string carries little of its meaning. Because the releases difer in more than the augmentation (Section 4) and each was trained once, these are diferences between two trained models, consistent with the augmentation as their cause but not a controlled ablation of it.

## 7 Unseen datasets and an open comparison model

Two Russian datasets were used by neither release and are not named in Laya’s model card, which does not list its training data: news headline topics from ruMTEB (six classes; licence declared as MIT in the repository metadata, no licence file) and single-label rows of CEDR emotions (Sboev et al., 2021) (five classes; Apache-2.0 in the metadata). Laya (Convai Innovations, 2026) is a 322M model on the same mmBERT backbone that reads all options in one pass. It was given a one-line instruction, which its interface requires and Ru-Jev does not take; option keys were neutral, so the option text carried the meaning for both.

<table><tr><td rowspan="2">Model</td><td colspan="3">News headlines (k = 6)</td><td colspan="3">Emotions (k = 5)</td></tr><tr><td>Acc.</td><td>Conf.-acc.</td><td>AURC</td><td>Acc.</td><td>Conf.—acc.</td><td>AURC</td></tr><tr><td>Ru-Jev v1.1</td><td>0.652</td><td>+0.028</td><td>0.209</td><td>0.455</td><td>+0.085</td><td>0.444</td></tr><tr><td>Ru-Jev v1.3</td><td>0.568</td><td>+0.095</td><td>0.260</td><td>0.511</td><td>-0.064</td><td>0.302</td></tr><tr><td>Laya</td><td>0.610</td><td>+0.242</td><td>0.221</td><td>0.605</td><td>+0.056</td><td>0.256</td></tr></table>

Table 3: Datasets no model trained on, 1,000 questions each. Ru-Jev uses its released temperatures; Laya returns its own probabilities.

No model is more accurate on both datasets (Table 3). v1.3 is 8.4 points below v1.1 on headlines (Holm $p < 1 0 ^ { - 6 }$ ; five of six labels worse, none better) and 5.6 points above it on emotions. Laya is markedly overconfident on headlines and not on emotions. When each model refits its own temperature on half the rows and is scored on the other half, residual calibration error is 0.056 to 0.067 for Ru-Jev (25 seeded splits) and 0.055 and 0.081 for Laya (an earlier split procedure without a recorded seed). We make no claim of superiority over Laya in accuracy or calibration.

On the card bench Laya scores well below Ru-Jev (with eighteen candidates, 0.150 against 0.640 on jargon). The MASSIVE training split is in Ru-Jev’s mix, and Laya’s own card reports near-chance zero-shot performance on typed decisions, so that gap measures home advantage and is not reported as a comparison.

## 8 In-distribution accuracy, calibration and cost

In-distribution accuracy. On the six RussianSuperGLUE tasks, whose rows and option sets are identical in both builds, the releases answer exactly 8,120 of 10,157 decisions correctly and disagree on 1,680, 840 in each direction (McNemar � = 1.0; paired standard error of the diference 0.0040). Per-row predictions and the SHA-256 of both weight files are in the repository. The word-sense task supplies 84% of these decisions.

<table><tr><td>Task</td><td>v1.1</td><td>v1.3</td><td>Only v1.1 right</td><td>Only v1.3 right</td><td>n</td></tr><tr><td>russe (word sense)</td><td>0.822</td><td>0.824</td><td>650</td><td>667</td><td>8505</td></tr><tr><td>danetqa (yes/no)</td><td>0.742</td><td>0.753</td><td>58</td><td>67</td><td>821</td></tr><tr><td>terra (entailment)</td><td>0.775</td><td>0.733</td><td>39</td><td>26</td><td>307</td></tr><tr><td>parus (causal choice)</td><td>0.680</td><td>0.590</td><td>22</td><td>13</td><td>100</td></tr><tr><td>rcb (three-way NLI)</td><td>0.450</td><td>0.509</td><td>23</td><td>36</td><td>220</td></tr><tr><td>rwsd (coreference)</td><td>0.564</td><td>0.480</td><td>48</td><td>31</td><td>204</td></tr><tr><td>All six</td><td>0.7994</td><td>0.7994</td><td>840</td><td>840</td><td>10157</td></tr></table>

Table 4: Identical rows and option wording for both releases.

Calibration. Over 25 random half-splits of these rows (seed recorded in the repository), fitting a temperature on one half and scoring the other, v1.3 takes a lower temperature in every split (1.50–1.60 against 1.94–2.10): its raw scores are less overconfident. Residual expected calibration error (Naeini et al., 2015) after each release receives its own temperature is the same (0.0190 against 0.0191; the sign of the diference changes across splits).

Selective prediction. With a calibrated probability, a caller can answer above a threshold and route the rest to a person (El-Yaniv and Wiener, 2010). In the system for citizen appeals that motivated this work, every prediction was reviewed by an operator and that review, not the model, dominated handling time (Beskorovainyi, 2026); a threshold is the natural next step. Table 5 gives the trade-of for v1.3 on its own validation split, with the temperature fitted on the disjoint half. Because the same split informed epoch selection, these figures are optimistic, and Sections 6 and 7 show that the released temperature does not carry over to unfamiliar labels or new domains. A script in the repository refits it on a few hundred labelled rows.

<table><tr><td>Share answered</td><td>Threshold</td><td>Accuracy on answered</td><td>Errors per 1000</td></tr><tr><td>50%</td><td>0.85</td><td>0.967</td><td>33</td></tr><tr><td>70%</td><td>0.69</td><td>0.921</td><td>79</td></tr><tr><td>90%</td><td>0.54</td><td>0.859</td><td>141</td></tr><tr><td>100%</td><td>0.23</td><td>0.823</td><td>177</td></tr></table>

Table 5: Coverage against accuracy, v1.3, on the 13,373 validation decisions held out from temperature fitting. Accuracy on this half is 0.823; on the full validation split of 26,746 decisions it is 0.818 (Table 1).

Cost. Table 6 gives the median latency of one decision on one Apple M4 CPU using all ten cores, over 25 runs. Ru-Jev is 1.2 times slower than Laya with two options and 2.4 times slower with eighteen.
<table><tr><td>Options k</td><td>2</td><td>5</td><td>18</td></tr><tr><td>Ru-Jev v1.3 (ms)</td><td>18.5</td><td>32.8</td><td>73.8</td></tr><tr><td>Laya (ms)</td><td>15.5</td><td>17.8</td><td>30.5</td></tr></table>

Table 6: Median latency per decision, Apple M4 CPU, all ten cores, 25 runs after 5 warm-up runs.

## 9 Four hazards in comparing releases

Our first comparison of the releases reported a clean accuracy gain and two large regressions. None of the three was real. We list the hazards because none is exotic and each was invisible in the output of a comparison that did not test for it.

1. Splits coupled through a shared random stream. The data builder carved each task’s validation slice with the module-level random generator, seeded once and consumed by every other random call in the file. Adding label styles to one task changed the number of draws consumed before the next task was carved, and every later task received a diferent partition. Intersecting the two builds shows the result is leakage: 90.3% of the new acceptability validation rows and 89.4% of the reasoning rows had been in the v1.1 training data, along with 47.2% of review rating and 21.6% of toxicity. The apparent regression on acceptability (0.909 to 0.647) was v1.1 scored on its own training rows; on held-out rows it scores 0.650. The fix seeds each carve from the task’s identity, random.Random(f"split:{task}"). We verified it by rebuilding the previous release’s data from its commit and confirming that the validation rows of the six tasks whose option sets never changed were identical to the earlier build.

2. Matching rows on input text alone. Matching validation rows by input text gave nine comparable tasks. Under the augmented recipe a row with the same text can carry an option wording the baseline never saw, so scoring the baseline there measures robustness, not accuracy. Requiring identical text and option wording leaves six tasks. A toxicity gain of 0.722 to 0.954 that we had presented as a capability gain was of this kind: v1.3 trained on all five toxicity wordings, v1.1 on one, and v1.1 scores 0.961 with the wording it trained on.

3. Mixed denominators. A summary row computed on one random half of the comparable rows (0.7948 against 0.7920 on 5,078 decisions) was printed beside a per-task table counting all 10,157, and a calibration advantage (0.0172 against 0.0160) was read of that one split. On all rows accuracy is identical, and the calibration diference changes sign across splits. Both figures had reached the published model card and were withdrawn. A related slip appeared in our drafts: a worst-case gap computed as the best vocabulary on all labels minus the jargon vocabulary on the clean slice.

4. Integrity checks against a stale reference. To estimate seed variance we retrained both recipes on a remote machine and checked its training mix against a local copy by SHA-256, row count and per-task counts. All matched. The local copy predated the commit that added the transliteration styles, so the seed runs meant to replicate v1.3 were trained without them; their jargon accuracy (0.588 and 0.611) fell near v1.1’s. Row counts could not catch this, since the new styles change option strings, not the number of rows. Counting distinct option strings per task did (120 for intents in the stale copy, 198 after rebuilding; some generated forms coincide). These runs were never published and are excluded from every result here.

Protocol. Seed every partition from the identity of what is partitioned; define comparability over everything the model consumes and print the overlap; measure leakage directly; choose a control that a change must not move; keep one denominator per table; verify artefacts by content that the change is expected to alter, not by hashes against a copy of unknown age; and evaluate wording robustness at more than one candidate-set size.

## 10 Limitations

One training run per release. The released v1.3 has no valid second seed. Two correctly built replicates of the v1.1 recipe scored 0.658 and 0.622 on jargon with five candidates (0.645 and 0.599 on the clean slice), against 0.633 (0.624) for the released v1.1, so the released baseline lies inside its own seed range. That range, 0.036 on all labels and 0.046 on the clean slice, is roughly four to six times smaller than the jargon gains of v1.3 (+0.222 and +0.214), but comparable to several smaller diferences in Table 2, which should not be treated as reproducible. Seed variance was measured only for the v1.1 recipe, on jargon, with five candidates; we have no seed estimate for eighteen candidates or for the slang and code probes.

Not a single-factor comparison. The releases difer in data partitions and in toxicity wording as well as in label augmentation (Section 4). The out-of-distribution changes in particular cannot be attributed to the augmentation alone.

Vocabularies. Each style is one vocabulary of 18 labels, so style and vocabulary are confounded, and label-level intervals rest on 9 to 18 clusters. All vocabularies were written by the author’s project or by AI agents working in it; none was collected from deployed systems. Abbreviations and code identifiers were scored with five candidates only.

Scope. Wording probes cover one routing task in Russian. The model answers only within the options it is given and cannot report that none fits; a none-of-these option must be supplied explicitly. One temperature serves all tasks and does not carry over to new labels or domains. Reasoning tasks with fewer than a thousand training rows remain weak; a 560M backbone (xlmroberta-large) tried once did not improve them.

## 11 Conclusion

When the caller supplies the label set, an evaluation that varies the model but not the label wording measures the easy case. Varying the wording at two candidate-set sizes showed that the release trained on several surface forms per label is more accurate on unfamiliar labels that are words, ordinary or borrowed, and gives no usable gain on abbreviations or opaque codes; and that better accuracy on unfamiliar wording does not by itself bring fewer confident errors when the candidate set is large. A practical reading is to choose augmentation styles that resemble the wording a deployment will use, to refit the temperature on that wording, and to evaluate on styles absent from the augmentation at more than one candidate-set size. We also recommend the release comparison protocol of Section 9.

## Data and code availability

Weights of both releases are on Hugging Face (https://huggingface.co/besktech/ru-jev; v1.1 on a branch). Training code, the data builder, all label vocabularies, the overlap audit, the latency benchmark, per-row scores for every experiment in this paper, the analysis script that produces Table 2 including AURC and confident errors, the SHA-256 of both weight files and of the files referred to in the reading rule, and the recorded split seed are at https://github.com/beskvladimir-create/ru-jev; the release used for this paper (v1.3.0) is archived at Zenodo, doi:10.5281/zenodo.23986847. Code and weights are under the MIT licence; training data keep their own licences, listed per source in the repository.

## Acknowledgements

Generative AI tools from Anthropic (Claude Code and Claude) were used to write and run experiment code and to draft and revise the manuscript. The author directed the work and takes full responsibility for its contents.

## References

V. Beskorovainyi. AI appeals processor: a deep learning approach to automated classification of citizen appeals in government services. arXiv preprint arXiv:2604.03672, 2026.

Convai Innovations. laya-multilingual: a non-autoregressive System 1 decision model. Model card, https://huggingface.co/convaiinnovations/laya-multilingual, accessed November 2026.

D. Dementieva, D. Moskovskiy, V. Logacheva, D. Dale, O. Kozlova, N. Semenov, and A. Panchenko. Methods for detoxification of texts for the Russian language. Multimodal Technologies and Interaction, 5(9):54, 2021.

T. Deußer, L. Sparrenberg, and R. Sifa. Evaluating and benchmarking the System One model Jev. arXiv preprint arXiv:2609.37647, 2026.

R. El-Yaniv and Y. Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11:1605–1641, 2010.

A. Fenogenova et al. MERA: A comprehensive LLM evaluation in Russian. In Proceedings of ACL, pages 9920–9948, 2024.

J. FitzGerald et al. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. In Proceedings of ACL, pages 4277–4302, 2023.

L. Gao, D. Ghosh, and K. Gimpel. The benefits of label-description training for zero-shot text classification. In Proceedings of EMNLP, 2023.

T. Gao, A. Fisch, and D. Chen. Making pre-trained language models better few-shot learners. In Proceedings of ACL-IJCNLP, pages 3816–3830, 2021.

R. Geirhos, C. R. Medina Temme, J. Rauber, H. H. Schütt, M. Bethge, and F. A. Wichmann. Generalisation in humans and deep neural networks. In Advances in Neural Information Processing Systems 31, 2018.

C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In Proceedings of ICML, pages 1321–1330, 2017.

A. Holtzman, P. West, V. Shwartz, Y. Choi, and L. Zettlemoyer. Surface form competition: Why the highest probability answer isn’t always right. In Proceedings of EMNLP, pages 7038–7051, 2021.

M. Marone, O. Weller, W. Fleshman, E. Yang, D. Lawrie, and B. Van Durme. mmBERT: A modern multilingual encoder with annealed language learning. In Proceedings of ICML, PMLR 306:86768–86783, 2026.

V. Mikhailov, T. Shamardina, M. Ryabinin, A. Pestova, I. Smurov, and E. Artemova. RuCoLA: Russian corpus of linguistic acceptability. In Proceedings of EMNLP, pages 5207–5227, 2022.

M. P. Naeini, G. F. Cooper, and M. Hauskrecht. Obtaining well calibrated probabilities using Bayesian binning. In Proceedings of AAAI, 2015.

J. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Advances in Large Margin Classifiers. MIT Press, 2000.

A. Sboev, A. Naumov, and R. Rybka. Data-driven model for emotion detection in Russian texts. Procedia Computer Science, 190:637–642, 2021.

T. Schick and H. Schütze. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of EACL, pages 255–269, 2021.

T. Shavrina et al. RussianSuperGLUE: A Russian language understanding evaluation benchmark. In Proceedings of EMNLP, pages 4717–4726, 2020.

A. Snegirev, M. Tikhonova, A. Maksimova, A. Fenogenova, and A. Abramov. The Russian-focused embedders’ exploration: ruMTEB benchmark and Russian embedding model design. In Proceedings of NAACL, pages 236–254, 2025.

Z. Wang, Z. Zhang, X. Wang, P. Wang, Z. Ye, H. Zhao, Z. Wu, and S. Yan. Chinese-Jev: Bringing System One model to Chinese-language tasks. arXiv preprint arXiv:2609.36965, 2026.

A. Webson and E. Pavlick. Do prompt-based models really understand the meaning of their prompts? In Proceedings of NAACL, pages 2300–2344, 2022.

W. Yin, J. Hay, and D. Roth. Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. In Proceedings of EMNLP-IJCNLP, pages 3914–3923, 2019.