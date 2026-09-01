# Language Proficiency Assessment from Eye Movements in Naturalistic Passage Reading

Shachar Frenkel<sup>∗</sup>, Ido Falah\*, Omer Shubi, Yevgeni Berzak

Faculty of Data and Decision Sciences,

Technion - Israel Institute of Technology

{fshachar,ido.falah,shubi}@campus.technion.ac.il, berzak@technion.ac.il

## Abstract

Standard language proficiency tests rely on linguistic tasks such as vocabulary, grammar and reading comprehension quizzes. An alternative, cognitively motivated approach, introduced in Berzak et al. (2018), proposed instead to predict language proficiency from behavioral traces of eye movements in reading. In this work, we validate and extend this approach from single sentences to more naturalistic reading of contextualized passages in English as a second language, new proficiency measures, prediction models, and reading in an information seeking regime. We find that the approach is effective in all these evaluations. We further address two key open questions on eye movement based proficiency testing: (1) potential scoring biases that reflect the proximity of the reader’s native language to English, which may undermine validity, and (2) its reliability. We find that eye movement based proficiency scores are indeed biased towards L1s that are linguistically closer to English. We propose a score debiasing method which effectively remedies this issue. The reliability analyses suggest that eye movement proficiency scores are more reliable than standard language proficiency scores. Overall, our results strengthen and broaden the empirical foundations for future eye movement based language assessment technologies.<sup>1</sup>

## 1 Introduction

English has been the de-facto global lingua franca for many decades, and the majority of English speakers worldwide learn, speak and read in English as a foreign language. As a result, educational institutions, English teachers and English learners are in an ever-growing need for effective and accurate tools for assessing English proficiency. Thus far, the highest quality language assessments for

English L2 speakers have been developed by public organizations and commercial companies such as the Educational Testing Service (ETS) and Cambridge Assessment English.

Such assessments are extremely labor-intensive, costly and time-consuming to create. Consequently, they are not deployed frequently, and their commercial variants involve substantial fees for test takers. Furthermore, materials from standardized tests are typically not made publicly available. The scarcity of high-quality materials for language assessments in the public domain leads many institutions and educators to create their own in-house assessments. Those assessments are not only difficult to develop, but are also often of lower quality, as they rarely receive the same attention to item construction, scaling and validation as in standardized tests. While in recent years new possibilities have emerged for faster development of test materials using generative AI, such tests are similarly subject to various quality assurance issues (Chuang and Yan, 2025).

Importantly, in addition to practical test construction and validation bottlenecks, the testing methodology of standard language proficiency tests itself is fundamentally limited. First, it relies on small samples of ad-hoc test items. Even more importantly, it is based on an offline signal: the end responses to these items. It has no ability to trace online cognitive processes during language processing as they unfold in real time. This limits the informativeness of the test and the quality of the feedback that can be provided to test takers.

An alternative approach, introduced in Berzak et al. (2018), proposed assessing L2 proficiency as a byproduct of ordinary reading, using behavioral traces of eye movements. This approach measures proficiency via online cognitive signals, and has the potential of obviating many of the practical drawbacks of traditional proficiency tests. Berzak et al. (2018) introduced two variants of this framework. The first is EyeScore, a standalone measure which estimates L2 proficiency using the similarity of the reader’s eye movements to L1 readers. The second uses eye movement features to predict performance on external standardized proficiency tests.

While Berzak et al. (2018) provided conceptual and methodological frameworks for eye movement based language proficiency testing, the generality and robustness of their approach has not been studied beyond their data, in which participants read single sentences. Furthermore, two key open questions that are pertinent to the real-world adoption potential of eye movement based proficiency tests have not been examined. The first is whether the validity of eye movement based scores may be undermined by biases stemming from the linguistic proximity of the reader’s L1 to English. The second question concerns the reliability of such scores, which is yet to be characterized.

Our primary contributions with respect to these questions are the following.

• Replication we validate eye movement based English proficiency scoring on 1,265 L2 participants from two passage reading corpora, MECO L2 (Kuperman et al., 2023, 2025) and OneStopL2, using different language proficiency measures, several prediction models, and in two reading regimes: reading for comprehension and information seeking.

• L1 Bias we find that EyeScore favors L2 speakers whose native languages are more similar to English. We propose a debiasing method which alleviates this issue.

• Reliability we analyze the internal consistency and split-half reliability of EyeScore. We find that EyeScore is more reliable than scores of standard language proficiency tests.

## 2 Background and Related Work

The prediction of language proficiency from eye movements is rooted in extensive literature in psycholinguistics which suggests that eye movements are influenced by linguistic properties of the text, and reflect real time language comprehension processes of the reader (e.g., Just and Carpenter, 1980; Reichle et al., 1998; Engbert et al., 2005; Demberg and Keller, 2008; Rayner et al., 2012). Additionally, ample evidence indicates that different levels of language proficiency yield differences in real time processing of the linguistic input (e.g. Clahsen and Felser, 2006; Whitford and Titone, 2012;

Cop et al., 2015a; Berzak and Levy, 2023). Combining these two factors, one can posit a causal chain where differences in language proficiency lead to differences in online comprehension processes, which in turn have behavioral traces in eye movements in reading. Decoding linguistic proficiency from eye movements can thus be viewed as an attempt to reverse-engineer this process, starting from the eye movement signal, and inferring back the linguistic proficiency of the reader.

Our work primarily builds on Berzak et al. (2018) who demonstrated the feasibility of predicting English proficiency from eye movements in reading using single sentence data from CELER (Berzak et al., 2022), with the Michigan Placement Test and TOEFL as external references. Our investigation of validity with respect to the reader’s L1 is motivated by work on systematic influences of the L1 on L2 learning and processing (Odlin, 1989; Jarvis and Pavlenko, 2008; Jeon and Yamashita, 2014). Such influences may impact various aspects of L2 production and comprehension, including reading, where prior work has demonstrated that the speaker’s L1 can be decoded from eye movement patterns in reading English L2 (Berzak et al., 2017; Reich et al., 2022; Skerath et al., 2023).

The current work is further related to studies that used eye movements for the prediction of reading comprehension (e.g., Copeland et al., 2014; Ahn et al., 2020; Reich et al., 2022; Mézière et al., 2023; Shubi et al., 2024). In these studies, reading comprehension performance is evaluated on the same textual materials for which the eye tracking data is collected. While the two tasks are related, our focus here is on external proficiency tests as measures of a more general linguistic ability.

More broadly, our study is situated within a nascent line of research on using eye movements for prediction of reader characteristics, such as dyslexia (e.g., Rello and Ballesteros, 2015; Haller et al., 2022; Laurinavichyute et al., 2025), and reader-text interactions such as different reading regimes (e.g., Shubi et al., 2025a; Meiri et al., 2025). EyeBench (Shubi et al., 2025b) is a recent benchmark which assembles several such tasks.

## 3 Eye Movement Based Estimation of Language Proficiency

We study two approaches to L2 proficiency testing using eye movements in reading, EyeScore and prediction of external proficiency scores.

## 3.1 EyeScore

The first type of tests is derived entirely using eye movements in reading, without external references. Specifically, we evaluate EyeScore, proposed by Berzak et al. (2018), which determines L2 proficiency via the similarity of eye movement patterns to L1 speakers. This approach is supported by empirical evidence on differences between L1 and L2 readers with respect to standard eye movement measures (Cop et al., 2015a; Berzak et al., 2022; Kuperman et al., 2023) and the extent to which they depend on the text (Cop et al., 2015b; Whitford and Titone, 2012, 2017; Berzak and Levy, 2023).

The procedure for calculating EyeScore is as follows. Given a set of participants D consisting of L1 readers $D _ { L 1 }$ and L2 readers $D _ { L 2 }$ , each participant $p \in D$ is represented by an eye movement feature vector $v _ { p } \in \mathbb { R } ^ { d }$ , where the dimensionality d depends on the feature set used (see Section 4). The vectors are then z-scored feature-wise using a z-scaler fitted on $D _ { L 2 }$ , yielding a normalized vector $\tilde { v } _ { p }$ for each participant. The L1 prototype vector $\bar { v } _ { L 1 }$ is the average of the normalized L1 vectors: $\begin{array} { r } { \bar { v } _ { L 1 } = \frac { 1 } { | D _ { L 1 } | } \sum _ { p \in D _ { L 1 } } \tilde { v } _ { p } } \end{array}$ . The EyeScore of an L2 participant is the cosine similarity between their normalized feature vector and the L1 prototype:

$$
{ \mathrm { E y e S c o r e } } _ { p } = { \frac { { \tilde { v } } _ { p } \cdot { \bar { v } } _ { L 1 } } { \| { \tilde { v } } _ { p } \| \| { \bar { v } } _ { L 1 } \| } }\tag{1}
$$

## 3.2 Predicting Scores on External Tests

The second approach to eye movement based proficiency testing uses eye movements to predict outcomes on standard language proficiency tests. As such, it enables obtaining a performance estimate on an external test of interest, without the need for administering it. Here, we examine using eye movement feature vectors with Ridge regression, as in Berzak et al. (2018). We additionally evaluate the decision tree-based model LightGBM (Ke et al., 2017), and a recent neural model for tabular data, TabPFN-3 (Grinsztajn et al., 2026). While more complex neural models for eye movements and text were introduced in recent years, we focus on the above models, and specifically on Ridge regression, for comparability with Berzak et al. (2018), and as standard machine learning models were found to be better suited for eye movement settings with a limited number of samples (Shubi et al., 2025b).

## 3.3 Baselines

Reading Speed (WPM) Our primary baseline is reading speed, the number of words read per minute (WPM). It captures the tendency of more proficient individuals to read faster. Importantly, it can be obtained without eye tracking equipment, and thus allows assessing the added value of eye movements for proficiency evaluation. In comparisons to EyeScore, we z-normalize WPM as Eye-Score features. In comparisons to prediction of external proficiency tests, we use WPM as input to the prediction model.

Average Train Score (Avg. Train) For the task of predicting scores on external tests, we further provide a baseline where we assign the score of a test participant to be the mean score among the training set participants.

## 4 Eye Movement Representations

The eye movement trajectory in reading is divided into fixations, where the gaze position is stable, and saccades, which are rapid transitions between fixations (Schotter and Dillon, 2025). Our eye movement representations are derived from this trajectory, and largely follow Berzak et al. (2018). We describe the feature sets in brief below, and provide additional details in Section A.

Average Fixation Metrics (Avg. Fix.) Fixation and saccade measures: First Fixation Duration (FF), Gaze Duration (GD), Total Fixation Duration (TF) and Go-Past Duration (GP), averaged across all the words, as well as first pass Skip Rate (SR) and first pass Regression Probability (RP). This feature set was not used in Berzak et al. (2018).

Syntactic Clusters (S-Clusters) The above fixation measures averaged per part-of-speech category, using Universal part-of-speech tags (Petrov et al., 2012) and Penn Treebank (PTB) part-of-speech tags (Santorini, 1990).

Word Property Coefficients (WP-Coefs) The linguistic word properties length, frequency and surprisal are known to affect reading times (e.g., Kliegl et al., 2004; Rayner et al., 2004). We encode the strength of this influence using the coefficients of linear models fitted to predict the fixation measures of each reader from length, frequency and surprisal. Surprisals are obtained from GPT-2 (Radford et al., 2019), and frequencies from Wordfreq (Speer et al., 2018).

Transitions The number of saccades in each direction between each pair of words in the text.

Word Fixation Metrics (Word Fix.) GD and TF for each individual word token in the text.

We note that the last two feature sets require the same texts for all participants (Fixed Text regime in Section 5.3), while the other three are also applicable when different participants read different texts (Any Text regime in Section 5.3).

## 5 Experimental Setup

## 5.1 Eye Movement Data

Our study is enabled by three recent large scale eye tracking datasets with passage reading in English (see Section B.1 for additional details).

MECO L2 (Kuperman et al., 2023, 2025) was collected across multiple sites using EyeLink 1000 (1000Hz) and similar eye trackers. It includes 1,106 L2 participants from 18 L1 backgrounds, and 95 L1 participants. All the participants read the same 12 informational texts (one per screen), and answered two comprehension questions after each text. For any given participant, eye movement data is available for 5–12 of the texts.

OneStopL2 was collected with an EyeLink 1000 Plus (1000Hz), and has 278 L2 participants from 10 L1 backgrounds. The texts are 30 newswire articles. Each participant reads one of three 10- article batches with 54 passages. Each passage is presented either in its original or simplified form, and is followed by a comprehension question. 154 participants read in an ordinary reading for comprehension regime, and 124 in an information seeking regime, where they receive the question prior to reading the passage. Our main analysis focuses on the reading for comprehension part of the data.

OneStop (Berzak et al., 2025) was collected with an EyeLink 1000 Plus, and includes 360 English L1 participants (180 reading for comprehension, 180 information seeking). It has the same experimental design and the same textual materials as OneStopL2. We use this dataset as an L1 reference for analyses with OneStopL2.

## 5.2 Standard English Proficiency Tests

In addition to eye movements data, the eye tracking datasets above include results for the following proficiency tests taken by the participants in-lab.

LexTALE (Lemhöfer and Broersma, 2012) An English vocabulary test which is often used as a proxy for English proficiency. The score range for

LexTALE is 0 to 100. Test results are available for all the OneStopL2 participants, 100 OneStop participants, 1,007 L2 participants of MECO L2 and 93 L1 participants of MECO L2. L1 participants with missing LexTALE scores are assigned the average L1 score in each dataset.

Michigan Placement Test (Form B) A standardized English proficiency test with four sections: listening comprehension, grammar, vocabulary and reading comprehension. It comprises 100 questions, with a score range of 0 to 100. Test results are available for all the OneStopL2 participants. Berzak et al. (2018) used the listening and grammar sections of this test. We follow them in assigning L1 participants with the maximal test score.

Composite Proficiency While MECO L2 does not include a standardized English proficiency test, it has the following four tests which are related to language proficiency: spelling recognition (Andrews and Hersch, 2010), vocabulary (Beglar, 2010) (groups 2–5), and two variants (word and pseudoword) of the Test of Word Reading Efficiency (TOWRE) (Torgesen et al., 1999). We use them to derive a Composite Proficiency score by applying PCA on the participant × test score matrix, and extracting the first principal component.

## 5.3 Prior Eye Movement Data Availability

When designing an eye movement based proficiency test, one needs to consider the evaluation setting with respect to which eye movement data is available, from which participants, and on which textual materials. Here, we assume that no prior eye movement data is available for the test participant. For the textual materials of the test, we follow Berzak et al. (2018) in distinguishing between the Fixed Text and Any Text regimes.

Fixed Text Eye movement data for the textual materials of the test is available from other participants. We evaluate a stringent variant of this setting where no eye movement data is available for texts other than those presented to the test participant. This setting is akin to traditional proficiency tests that rely on specific test forms.

Any Text No eye movement data is available for the test materials. In other words, all existing eye movement data is on different texts than those presented to the test participant. This is a much more general setting that supports eye tracking based testing using arbitrary texts.

## 5.4 Evaluation Measures

EyeScore The evaluation of standalone eye tracking based proficiency scores is a challenging methodological problem. Following Berzak et al. (2018), we use scores from external proficiency tests (LexTALE, Michigan and Composite scores described in Section 5.2) as heuristic measures for evaluating the validity of EyeScore. In Sections 6.1 and 7.3 we report the Pearson r coefficient between EyeScore and these tests. In Section 7 we examine another aspect of EyeScore’s validity by developing a measure of L1 bias relative to standard tests. Cronbach’s α and split-half Pearson r are used for estimating the reliability of EyeScore in Section 8.

Direct Prediction of External Scores In Section 6.2, we evaluate the regression models trained to predict LexTALE, Michigan and Composite scores using Pearson r and Mean Absolute Error (MAE) between predicted and true scores.

## 5.5 Data Splits

Both EyeScore and the prediction of external test scores are evaluated using cross validation. As we focus on proficiency assessment for L2, only L2 participants appear in the test set. For predicting external test scores, we follow Berzak et al. (2018) in including L1 participants in the training set.

MECO L2 We divide the 12-passage dataset into two parts with 6 passages each and use leaveone-participant-out cross validation. We perform the evaluation for each participant twice: once for each half of the data, and average the results. In the Fixed Text regime, the training passages are from the same half of the data read by the test participant, and in the Any Text regime they are from the other half. As not all participants have eye movement data for all 12 passages, in the Fixed Text regime, we impute the missing passage features for the Transitions and the Word Fixation Metrics feature sets with the mean values in the training set. On average, there are 1,148 participants (1,053 L2, 95 L1) in the training set in both Text regimes.

OneStopL2 Each participant reads one of three 10-article batches. As there are two variants of each batch with respect to the difficulty version of each passage (original or simplified), there are 6 subgroups of participants reading exactly the same texts. In the Fixed Text regime, we perform leaveone-participant-out cross validation for each group, with an average of 24 L2 and 24 L1 training participants. In the Any Text regime we perform 3-fold cross validation where test participants read one batch, and training participants are from the other two batches, with an average of 100 L2 and 99 L1 training participants. Additional details on the data splits are described in Section B.2.

## 5.6 Hyperparameter Tuning

For predicting external proficiency scores with Ridge regression, we tune the Ridge regularization parameter on the training set with cross validation, over 10 log-spaced values between $1 0 ^ { - 3 }$ and $1 0 ^ { 4 }$ . Section B.3 provides additional details on this procedure, and on the hyperparameter tuning for LightGBM and TabPFN-3.

## 6 Replication Results

## 6.1 EyeScore

Table 1 presents the Pearson r of EyeScore with external English proficiency tests for L2 participants. The results are consistent across the two datasets and three external reference tests, and the general trends are in line with Berzak et al. (2018), with mostly similar or higher correlations. Among the three feature sets that apply to both the Fixed and Any Text regimes, WP-Coefs tends to perform best. Importantly, all three feature sets yield comparable results in the Fixed and Any Text regimes, suggesting that they are robust to the specifics of the textual input. The Word Fixation Metrics feature set is the strongest in the Fixed Text regime.

<table><tr><td></td><td colspan="4">MECO L2</td><td colspan="4">OneStopL2</td></tr><tr><td></td><td colspan="2">LexTALE</td><td colspan="2">Composite</td><td colspan="2">LexTALE</td><td colspan="2">Michigan</td></tr><tr><td>Features</td><td>Fixed</td><td>Any</td><td>Fixed</td><td>Any</td><td>Fixed</td><td>Any</td><td>Fixed</td><td>Any</td></tr><tr><td>WPM</td><td>0.48</td><td>0.48</td><td>0.45</td><td>0.45</td><td>0.47</td><td>0.48</td><td>0.47</td><td>0.47</td></tr><tr><td>Avg. Fix.</td><td>0.49</td><td>0.50</td><td>0.45</td><td>0.46</td><td>0.48</td><td>0.49</td><td>0.50</td><td>0.49</td></tr><tr><td>S-Clusters</td><td>0.50</td><td>0.50</td><td>0.47</td><td>0.46</td><td>0.50</td><td>0.51</td><td>0.50</td><td>0.51</td></tr><tr><td>WP-Coefs</td><td>0.54***</td><td>0.54***</td><td>0.51***</td><td>0.51***</td><td>0.49</td><td>0.51</td><td>0.51</td><td>0.52</td></tr><tr><td>Transitions</td><td>0.48</td><td></td><td>0.40*</td><td>-</td><td>0.51</td><td></td><td>0.49</td><td>-</td></tr><tr><td>Word Fix.</td><td>0.54***</td><td></td><td>0.51***</td><td></td><td>0.54**</td><td>1</td><td>0.55**</td><td>-</td></tr></table>

Table 1: EyeScore. Pearson r correlations between Eye-Score and standard English proficiency tests. Presented are mean r correlations using bootstrap. Statistically significant differences from the WPM baseline using a bootstrap test are marked with ∗ $p \ < \ 0 . 0 5$ ∗∗ $p < 0 . 0 1 , { } ^ { * * * } p < 0 . 0 0 1$ . “Fixed” is the Fixed Text regime in which all the eye movement data is for the same texts presented to the test participant. “Any” is the Any Text regime in which no eye movement data is available for the texts of the test participant. The best result for each eye tracking dataset, benchmark proficiency test, and Text regime is marked in bold.

The main difference compared to Berzak et al. (2018) is a much stronger performance of the WPM reading speed baseline $( r \le 0 . 2 8 $ in their data). We hypothesize that this difference could stem from strategic sentence rereading behavior for performing better on a question presented after each sentence in CELER, which is less feasible in passage reading in MECO L2 and OneStopL2. Despite this difference, eye movement feature sets systematically outperform WPM, confirming their value above and beyond reading speed. Similar results are obtained in the OneStopL2 information seeking regime (Table 7 in Section C), suggesting that EyeScore is robust to the reading manner.

<table><tr><td></td><td colspan="8">MECO L2 LexTALE</td><td colspan="8">OneStopL2</td></tr><tr><td></td><td colspan="3">Fixed</td><td colspan="2"></td><td colspan="3">Composite</td><td colspan="2">LexTALE Fixed</td><td colspan="2"></td><td colspan="2">Michigan Fixed</td><td colspan="2"></td></tr><tr><td>Features</td><td>r</td><td>MAE</td><td>r</td><td>Any MAE</td><td>Fixed r</td><td>MAE</td><td>r</td><td>Any MAE</td><td>r</td><td>MAE</td><td>Any r</td><td>MAE</td><td>r</td><td>MAE</td><td>Any</td><td>MAE</td></tr><tr><td>Avg. Train</td><td>0.00</td><td>9.92</td><td>0.00</td><td>9.92</td><td>0.00</td><td>11.37</td><td>0.00</td><td>11.37</td><td>-0.06</td><td>10.32</td><td>-0.02</td><td>10.25</td><td>-0.07</td><td>7.52</td><td>r -0.01</td><td>7.40</td></tr><tr><td>WPM</td><td>0.48</td><td>8.53</td><td>0.48</td><td>8.52</td><td>0.44</td><td>9.96</td><td>0.44</td><td>9.95</td><td>0.42</td><td>10.21</td><td>0.47</td><td>9.62</td><td>0.37</td><td>6.42</td><td>0.47</td><td>6.14</td></tr><tr><td></td><td></td><td> $0 . 5 5 ^ { * * * } 8 . 1 6 ^ { * * * }$ </td><td></td><td> $0 . 5 4 ^ { * * * } 8 . 1 7 ^ { * * * }$ </td><td> $0 . 5 4 ^ { * * * } \quad 9 . 3 9 ^ { * * * }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.87</td></tr><tr><td>Avg. Fix. S-Clusters</td><td></td><td> $0 . 5 6 ^ { * * * } 8 . 1 0 ^ { * * * }$ </td><td></td><td> $0 . 5 5 ^ { * * * } \ 8 . 1 4 ^ { * * * }$ </td><td> $0 . 5 5 ^ { * * * } \quad 9 . 2 6 ^ { * * * }$ </td><td></td><td></td><td> $\mathbf { 0 . 5 4 ^ { * * * } } \ \mathbf { 9 . 4 1 ^ { * * * } }$   $\mathbf { 0 . 5 4 ^ { * * * } 9 . 3 9 ^ { * * * } }$ </td><td>0.47 0.49</td><td>10.00 9.75</td><td>0.55* 0.57*</td><td>9.04 8.67*</td><td>0.45* 0.47*</td><td>6.34 6.26</td><td>0.56* 0.57*</td><td>5.82</td></tr><tr><td>WP-Coefs</td><td></td><td> $0 . 5 6 ^ { * * * } 8 . 1 2 ^ { * * * }$ </td><td></td><td> $0 . 5 6 ^ { * * * } \ 8 . 1 4 ^ { * * * }$ </td><td> $0 . 5 5 ^ { * * * } \quad 9 . 3 0 ^ { * * * }$ </td><td></td><td></td><td> $\mathbf { 0 . 5 4 ^ { * * * } } \ \mathbf { 9 . 4 1 ^ { * * * } }$ </td><td>0.48</td><td>9.65</td><td>0.58*</td><td>8.57*</td><td>0.46*</td><td>6.36</td><td>0.56*</td><td>5.91</td></tr><tr><td>Transitions</td><td></td><td></td><td></td><td></td><td> $0 . 5 6 ^ { * * * } \quad 9 . 3 7 ^ { * * * }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 4 7 ^ { * * }$ </td><td>6.07</td><td></td><td></td></tr><tr><td>Word Fix.</td><td> $0 . 5 5 ^ { * * * } 8 . 0 8 ^ { * * * }$ </td><td> $\mathbf { 0 . 6 2 ^ { * * * } 7 . 5 9 ^ { * * * } }$ </td><td></td><td>=</td><td> $\mathbf { 0 . 5 9 ^ { * * * } 8 . 8 6 ^ { * * * } }$ </td><td></td><td>一</td><td></td><td>0.50*  $\mathbf { 0 . 5 6 ^ { * * * } } \ \mathbf { 8 . 8 6 ^ { * * } }$ </td><td>9.78</td><td>1</td><td></td><td> $\mathbf { 0 . 5 2 ^ { * * * } }$ </td><td>5.86</td><td></td><td>一</td></tr></table>

Table 2: Prediction of standard English proficiency test scores. Pearson r and Mean Absolute Error (MAE) for prediction of external proficiency scores using eye movements with Ridge regression. Presented are mean values using bootstrap. Statistically significant differences from the WPM baseline using a bootstrap test are marked with $^ { * * , } p < 0 . 0 5 , ^ { * * * , } p < 0 . 0 1 , ^ { * * * , } p < 0 . 0 0 1$ . “Fixed” is the Fixed Text regime in which all the eye movement data is for the same texts presented to the test participant. “Any” is the Any Text regime in which no eye movement data is available for the texts of the test participant. The best result for each evaluation measure in each eye tracking dataset, reference proficiency test and Text regime is marked in bold.

## 6.2 Predicting External Proficiency Scores

The results for predicting external proficiency test scores from eye movements of L2 participants are presented in Table 2. As can be expected from a model explicitly trained to predict external test outcomes, for any given Pearson r evaluation, the correlations tend to be higher than those for Eye-Score in Table 1, and the improvements over the WPM baseline are more substantial. Here too, the results are in line with Berzak et al. (2018), with similar or higher correlations in most cases. The primary exception is OneStopL2 Fixed Text, where the results for both Michigan and LexTALE are low, even compared to the corresponding Eye-Score results. We attribute this to the small training data in OneStopL2 for the Fixed Text regime (see Section 5.5), suggesting that differently from Eye-Score, a sizable number of L2 training participants is needed for robust prediction of external scores.

Table 8 in Section C presents similar results in information seeking, which suggest that the prediction of external test results largely generalizes to this reading regime. Section D includes the results for LightGBM (Tables 9 and 10) and TabPFN-3 (Tables 11 and 12). LightGBM yields similar performance to Ridge regression on MECO L2, and tends to perform worse on OneStopL2. TabPFN-3 largely outperforms Ridge regression on MECO L2, and performs similarly on OneStopL2.

## 6.3 Comparison to Correlations between Standard Proficiency Tests

Especially informative references for the presented results are the correlations between different standard proficiency tests. These can be thought of as an upper bound for EyeScore and as a target benchmark for the prediction task. In our datasets, the Pearson r between LexTALE and the Composite score in MECO L2 is 0.55, and between LexTALE and Michigan in OneStopL2 is 0.61. Reported correlations among widely used standardized English proficiency tests are typically in the range of 0.61–0.85.<sup>2</sup> The EyeScore correlations with standard tests in Table 1 are lower, likely reflecting the radical difference in the testing methodology. Direct prediction of external test scores from eye movements is at the lower end of this range for the best performing feature sets. Overall, these results strengthen the evidence for the viability of eye movement based language proficiency testing.

## 7 Measuring and Mitigating L1 Bias in EyeScore

Since EyeScore relies on similarity of reading patterns to the average English L1 reader, this measure might favor readers whose L1 is more similar to English. Below, we investigate this question in the Any Text regime.

## 7.1 Measuring L1 Bias

We use a two-step procedure for measuring Eye-Score L1 bias. The procedure relies on a standard English proficiency test which serves as a control for possible differences in English proficiency across the L1 groups in our sample of participants. For a given external proficiency Test $\in \ \{ \mathrm { L e x T A L E }$ , Composite, Michigan}, we fit an ${ \mathrm { E y e S c o r e } } _ { p } \sim { \mathrm { T e s t } } _ { p }$ regression on all L2 participants and obtain for each participant p the residual by subtracting the model prediction from Eye-Score:

$$
\mathrm { r e s } _ { p , \mathrm { T e s t } } : = \mathrm { E y e S c o r e } _ { p } - ( \hat { \beta } _ { 0 } + \hat { \beta } _ { \mathrm { T e s t } } \cdot \mathrm { T e s t } _ { p } )\tag{2}
$$

where $\hat { \beta } _ { 0 }$ and $\hat { \beta } _ { \mathrm { T e s t } }$ are the fitted regression coefficients and $\mathrm { T e s t } _ { p }$ is the participant’s test score.

In the second step, we examine whether the residuals systematically depend on the linguistic proximity of the participant’s L1s to English. To this end, we fit a regression model that predicts the residuals from the L1 distance from English ${ \mathrm { r e s } } _ { p , \mathrm { T e s t } } \sim d ( \mathrm { L } 1 _ { p } ,$ , English). The fitted model predicts a participant’s residual based on their L1:

$$
\widehat { \mathrm { r e s } } _ { p , \mathrm { T e s t } } = \hat { \beta } _ { 0 } + \hat { \beta } _ { d i s t } \cdot d ( \mathrm { L } 1 _ { p } , \mathrm { E n g l i s h } )\tag{3}
$$

A slope coefficient $\hat { \beta } _ { d i s t }$ which is significantly smaller than 0 would indicate that EyeScore is biased towards L1s that are closer to English, above and beyond the reference proficiency test.

We compute d(L1, English) as the angular distance between binarized language vectors with linguistic features from URIEL+ (Khan et al., 2025, v1.2). We use 103 syntactic features from WALS (Dryer and Haspelmath, 2022), Ethnologue (Lewis et al., 2015), and SSWL (Collins and Kayne, 2009), 28 phonological features from WALS and Ethnologue, and 42 writing system features from Script-Source (Holloway, 2013).

The $\mathrm { r e s } _ { p , \mathrm { L e x T A L E } } \sim d ( \mathrm { L } 1 _ { p } , \mathrm { E n g l i s h } )$ regression fit for EyeScore using MECO L2 in the Any Text regime and WP-Coefs is presented in Figure 1. We observe that participants whose L1 is less similar to English, indeed tend to score lower on EyeScore relative to the LexTALE reference $( \hat { \beta } _ { d i s t } = - 0 . 5 2 , r = - 0 . 1 8 , p < 1 0 ^ { - 7 } )$ . Table 3 suggests that this negative correlation holds for all the feature sets relative to LexTALE and the Composite score in MECO L2, and relative to Michigan for WP-Coefs in OneStopL2. In the other cases the coefficients are not significant, but consistently negative. Figures 2 to 5 in Section E.1 present the corresponding regression fits.

![](images/3dacf0066b32ccc3507a570bb03e1231a29b39e7018b17978ebed5b686682358.jpg)  
Figure 1: L1-English Proximity Bias in EyeScore. Residuals of an ${ \mathrm { E y e S c o r e } } _ { p } \sim { \mathrm { L e x T A L E } } _ { p }$ regression as a function of the linguistic distance of the participant’s L1 to English: res<sub>p,LexTALE</sub> $\sim d ( \mathrm { L } 1 _ { p } ,$ English). The figure depicts MECO L2 participants in the Any Text regime using the WP-Coefs feature set. Each circle is a participant. Diamond shapes denote L1 means. Jitter is added for legibility. The dashed line is the model fit, which suggests that EyeScore favors participants whose L1 is more similar to English.

## 7.2 Debiasing EyeScore

To mitigate the L1 bias in EyeScore, we propose a simple correction procedure that subtracts from EyeScore the bias predicted by the participant’s L1 distance to English. The debiased EyeScore is

$$
{ \mathrm { E y e S c o r e } } _ { p } ^ { \mathrm { d b } } = { \mathrm { E y e S c o r e } } _ { p } - { \widehat { \mathrm { r e s } } } _ { p , { \mathrm { T e s t } } }\tag{4}
$$

where $\widehat { \mathrm { r e s } } _ { p , \mathrm { T e s t } }$ is the predicted mean bias for the participant’s L1 relative to an external test.

EyeScore<sup>db</sup> is computed for test set participants by adjusting their EyeScore using a $\widehat { \mathrm { r e s } } _ { p , \mathrm { T e s t } }$ model fitted on a training set using k-fold cross validation, where k is the number of L1s. We consider two data split variants, Seen L1: stratified k-fold cross validation with all the L1s in the training set and test set of each split, and a more stringent New L1 split: k-1 L1s in the training set and a held out L1 in the test set.

<table><tr><td></td><td colspan="6">MECO L2</td><td colspan="6">OneStopL2</td></tr><tr><td></td><td></td><td colspan="2">LexTALE</td><td colspan="3">Composite</td><td colspan="2">LexTALE</td><td colspan="3"></td></tr><tr><td></td><td>EyeScore</td><td>EyeScoredb</td><td></td><td>EyeScore</td><td>EyeScoredb</td><td></td><td>EyeScore</td><td>EyeScoredb</td><td>EyeScore</td><td></td><td>EyeScoredb</td></tr><tr><td>Features</td><td></td><td>Seen L1</td><td>New L1</td><td></td><td>Seen L1</td><td>New L1</td><td></td><td>Seen L1 New L1</td><td></td><td>Seen L1</td><td>New L1</td></tr><tr><td>WPM</td><td> $- 1 . 3 7 ^ { * * * }$ </td><td>-0.05</td><td>-0.43</td><td>-1.97***</td><td>-0.60</td><td>-0.98**</td><td>-1.28</td><td>0.00 -0.01</td><td> $- 1 . 8 7 ^ { * }$ </td><td>-0.62</td><td>-0.57</td></tr><tr><td>Avg. Fix.</td><td> $\cdot 0 . 7 8 ^ { \ast \ast \ast }$ </td><td>-0.03</td><td>-0.14</td><td> $- 1 . 1 8 ^ { * * * }$  一</td><td>-0.40</td><td>-0.51*</td><td>-0.75</td><td>0.00 0.03</td><td>-1.15</td><td>-0.39</td><td>-0.35</td></tr><tr><td>S-Clusters</td><td> $- 0 . 4 3 ^ { * * }$ </td><td>-0.02</td><td>-0.11</td><td> $- 0 . 7 1 ^ { \ast \ast \ast }$ </td><td>-0.28</td><td>-0.37*</td><td>-0.53 -0.01</td><td>0.03</td><td>-0.88</td><td>-0.36</td><td>-0.30</td></tr><tr><td>WP-Coefs</td><td> $- 0 . 5 2 ^ { * * * }$ </td><td>-0.02</td><td>-0.10</td><td>-0.71***</td><td>-0.19*</td><td> $- 0 . 2 7 ^ { * * }$ </td><td>-0.63</td><td>-0.01 -0.01</td><td>-0.88*</td><td>-0.25</td><td>-0.24</td></tr></table>

Table 3: L1-English Proximity Bias in EyeScore and EyeScore<sup>db</sup> in the Any Text regime. Presented are $\hat { \beta } _ { d i s t }$ coefficients from $\widehat { \mathrm { r e s } } _ { p , \mathrm { T e s t } } = \hat { \beta } _ { 0 } + \hat { \beta } _ { d i s t } \cdot d ( L 1 _ { p } ,$ English) regressions, which capture EyeScore and $\mathrm { E y e S c o r e ^ { d b } }$ bias towards L1s that are linguistically more similar to English. In all cases, EyeScore<sup>db</sup> is obtained by debiasing EyeScore relative to LexTALE. Seen L1 / New L1 refers to whether the training data for deriving the debiasing transformation included participants from the same L1 as the test participant. Significance of a t-test for the $\hat { \beta } _ { d i s t }$ coefficients is marked with $^ { \ast \ast , } p < 0 . 0 5 ,$ ‘<sup>∗∗</sup>’ p < 0.01, $^ { 6 * * * } p < 0 . 0 0 1$ . In MECO L2, the $\hat { \beta } _ { d i s t }$ coefficients of EyeScore<sup>db</sup> are significantly smaller than their EyeScore counterparts in all cases $( p < 0 . 0 5$ , bootstrap test).

To evaluate the debiasing procedure, we refit the models from Equations (2) and (3) using EyeScore<sup>db</sup>, and compare the resulting $\hat { \beta } _ { d i s t }$ coefficients to those fitted with EyeScore. In our primary evaluation, EyeScore<sup>db</sup> is obtained by debiasing EyeScore relative to LexTALE (i.e. EyeScore − $\widehat { \mathrm { r e s } } _ { p , \mathrm { L e x T A L E } } )$ . Thus, for LexTALE, the same test is used for debiasing and for evaluation, while the Composite and Michigan evaluations capture cross-test robustness.

Table 3 presents the resulting $\hat { \beta } _ { d i s t }$ coefficients in the Any Text regime. Section E.2 Table 13 presents the corresponding Pearson r of the models. For LexTALE, we find that the bias is eliminated nearly entirely on both MECO L2 and OneStopL2, especially in the Seen L1 evaluations. Most of the bias is also removed when the LexTALE-debiased scores are evaluated against the other two external tests, Michigan and Composite. This outcome suggests cross-test robustness in debiasing and evaluation. Here too, the Seen L1 evaluations are more effective compared to New L1. A similar pattern of results is obtained when EyeScore<sup>db</sup> is obtained by debiasing EyeScore relative to Composite (Section E.3 Table 14) and relative to Michigan (Section E.3 Table 15).

Section E.4 presents largely similar results for the Fixed Text regime (Table 16) and for information seeking reading in OneStopL2 (Table 17). Overall, the results indicate that the debiasing procedure is effective in mitigating L1 bias. Finally, we note that a similar procedure for L1 bias diagnosis and removal can be implemented with a single multiple regression model of the form ${ \mathrm { E y e S c o r e } } _ { p } \sim { \mathrm { T e s t } } _ { p } + d ( { \mathrm { L } } 1 _ { p } , { \mathrm { E n g l i s h } } )$

## 7.3 Does Debiasing EyeScore Hurt Correlations with Standard Tests?

This is something you might be wondering about. To examine this question, in Section E.5 Table 18 we present $\Delta { r } _ { \mathrm { d b } } .$ , the difference for L2 participants between (1) the Pearson r of an EyeScore<sup>db</sup> debiased relative to LexTALE with each of the standard proficiency tests, and (2) the corresponding Pearson r for EyeScore (Table 1). Although significant in some cases, the absolute differences in Pearson r between the two measures are small. In the Any Text regime, the average $\Delta r _ { \mathrm { d b } }$ across feature sets for the correlation with LexTALE is −0.008 for MECO L2 and −0.016 for OneStopL2. The average $\Delta r _ { \mathrm { d b } }$ with the Composite score is +0.006, and −0.007 with Michigan. Similar results are obtained in the Fixed Text regime. We further note that the average correlation between EyeScore and EyeScore<sup>db</sup> across all feature sets in both datasets is 0.99, indicating the overall stability of the debiasing procedure.

## 8 EyeScore Reliability

An important psychometric indicator for the quality of a test is its reliability, namely the extent to which participant scores are consistent across different variants of the test. Despite the significance of this test characteristic, the reliability of eye-tracking-based proficiency tests has yet to be measured. Here, we examine two standard measures of reliability: internal consistency and splithalf reliability, with respect to EyeScore.

## 8.1 Internal Consistency

Internal consistency measures the degree to which items in a test capture the same underlying construct. We use Cronbach’s α (Cronbach, 1951), a standard measure of internal consistency, defined as $\begin{array} { r } { \alpha = \frac { n } { n - 1 } ( 1 - \frac { \sum _ { i = 1 } ^ { n } V _ { i } } { V _ { t } } ) } \end{array}$ , where n is the number of test items, $V _ { i }$ is the variance of test item i scores across participants and $V _ { t }$ is the variance of the test scores across participants. The range of possible values for this measure is 0 to 1. We also report the standard error of measurement (SEM) (Harvill, 1991) for Cronbach’s α, which is used to estimate measure uncertainty, and defined as SD · ${ \sqrt { 1 - \alpha } } ,$ where SD is the standard deviation estimate of test scores across participants.

Cronbach’s α requires the test to consist of multiple items, a score for each participant on each item, and an overall test score for each participant based on an aggregation of the item-level scores. A possible unit for an item that applies to both MECO L2 and OneStopL2 is eye movements over a passage. Accordingly, we construct EyeScore<sup>items</sup>, an aggregated-per-item variant of EyeScore, where we first compute for each participant an EyeScore for each passage, and then define the overall EyeScore of the participant as the mean of the item-level scores. EyeScore<sup>items</sup> correlates very strongly with EyeScore (average Pearson r 0.983, Section F.1 Table 19) and has the same or slightly higher correlations with external proficiency tests (average ∆r +0.011, Section F.1 Table 20). Since Cronbach’s α assumes that all participants have scores for all the items, in MECO L2 we use only the 144 participants who have eye movement data for all 12 passages. Additional details on the evaluation of internal consistency are provided in Section F.2.

## 8.2 Split-Half Reliability

This measure is a proxy for test-retest reliability, where instead of taking the test twice, a single test is divided into two halves. Then, the Pearson r correlation between the test scores on each half is computed across participants. As the results of a split-half analysis can vary across splits (Cronbach, 1951), we report the mean over the Pearson r coefficients of 20 random splits of the passages. Additional details on the split-half reliability measurement procedure are provided in Section F.3.

## 8.3 Results

The Cronbach’s α and split-half reliability of Eye-Score in the Any Text regime are presented in Table 4. We find that in nearly all cases, the reliability of EyeScore is extremely high according to both measures, with an average Cronbach’s α of 0.98 and an average Pearson r of split-half reliability of 0.93. Reliability analyses in the Fixed Text regime (Section F.4 Table 21) and for the information seeking reading regime in OneStopL2 (Section F.4 Table 22) yield similar results.

<table><tr><td></td><td colspan="3">MECO L2</td><td colspan="3">OneStopL2</td></tr><tr><td>Features</td><td>Cronbach&#x27;s α</td><td>SEM</td><td>Split-Half r</td><td>Cronbach&#x27;s α</td><td>SEM</td><td>Split-Half r</td></tr><tr><td>WPM</td><td>0.98</td><td>0.13</td><td>0.93</td><td>0.99</td><td>0.07</td><td>0.98</td></tr><tr><td>Avg. Fix.</td><td>0.98</td><td>0.08</td><td>0.92</td><td>0.99</td><td>0.04</td><td>0.98</td></tr><tr><td>S-Clusters</td><td>0.98</td><td>0.04</td><td>0.93</td><td>1.00</td><td>0.02</td><td>0.98</td></tr><tr><td>WP-Coefs</td><td>0.91</td><td>0.05</td><td>0.79</td><td>0.99</td><td>0.02</td><td>0.95</td></tr></table>

Table 4: Reliability of EyeScore for L2 participants in the Any Text regime.

Importantly, the reliability of EyeScore tends to be higher than the reliability of standardized English proficiency tests. In OneStopL2, the Michigan test has a Cronbach’s α of 0.91 (stratified), and a split-half Pearson r of 0.92. The average Cronbach’s α across the IELTS reading and listening sections is 0.91 (IELTS, 2026). The overall reliability of TOEFL-iBT is 0.90 (Manna et al., 2025). Additional details on the reliability of these tests are provided in Section F.5.

## 9 Conclusion

We present the first replication study of a language proficiency testing paradigm in which instead of standard test items, proficiency is derived from eye movements in reading. We find that this methodology is effective in passage reading across datasets, standard proficiency references and reading regimes. We further investigate validity with respect to L1 bias and the reliability of the method.

Overall, our experiments provide substantial support for the viability of eye movement based proficiency testing. In the future, we envision that it can be deployed either as a standalone technology or as an add-on to traditional proficiency tests. The limitations section below details some of the work that is needed in order to make this paradigm viable for real-world settings.

## 10 Limitations

Our study has a number of limitations. Perhaps the cardinal limitation of eye tracking based approaches to language proficiency assessment is that they can be used for assessing only comprehension related linguistic abilities. Production abilities for spoken and written language currently remain out of scope for such methods.

Our L1 bias correction procedure has several limitations. First, it relies on the estimation of L1 distances from English, which can vary according to the chosen set of language features, how missing feature values are handled, and which metric is used to measure language distances. Another limitation is the dependence on an external reference test, whose choice can also influence the procedure outcomes. Even more importantly, we rely on the assumption that the external tests are not themselves biased towards specific L1s, and in particular towards L1s that are closer to English. If this assumption does not hold, our procedure will underestimate the amount of L1 bias that needs to be removed. Finally, our debiasing pipeline assumes that participants have a single L1, and thus cannot be straightforwardly applied to participants with several L1s.

Additional limitations concern the available data. While the recent introduction of MECO L2, OneStopL2 and OneStop has considerably expanded the possibilities for developing eye movement based language proficiency testing, these datasets are restricted to English, cover only two textual domains, and include participants from specific language backgrounds and a restricted range of ages. A more general investigation would require the collection and analysis of data for additional L2 target languages and textual domains, additional participant populations such as children, and participants from additional native language backgrounds.

It is further important to note that the used datasets were collected with state-of-the-art eye tracking equipment in laboratory conditions. To establish performance in real-world scenarios, additional work is needed with lower-quality eyetracking equipment and in settings outside of the lab. Finally, further data collection and experimentation are needed to characterize additional properties of eye movement based language proficiency tests, such as the extent to which participants can cheat in such tests, for example, by using top-down reading strategies such as skimming the text, as well as test reliability across different test sessions.

## 11 Ethical Considerations

All three eyetracking datasets used in this study, MECO L2, OneStopL2 and OneStop, were collected under institutional IRB protocols. All the participants have provided written consent for participation. All the data is anonymized. Studying the relation between eye movements and linguistic proficiency is among the intended use cases for which the datasets were collected. We further note that while it has been demonstrated that eyetracking data can be used for reader identification (e.g. Jäger et al., 2019), the anonymization of the used datasets precludes such uses.

## Acknowledgments

This work was supported by ISF grant 1499/22.

## References

Seoyoung Ahn, Conor Kelton, Aruna Balasubramanian, and Greg Zelinsky. 2020. Towards predicting reading comprehension from gaze behavior. In Symposium on Eye Tracking Research and Applications, pages 1–5. Association for Computing Machinery.

Sally Andrews and Jolyn Hersch. 2010. Lexical precision in skilled readers: Individual differences in masked neighbor priming. Journal of Experimental Psychology: General, 139(2):299.

David Beglar. 2010. A Rasch-based validation of the Vocabulary Size Test. Language testing, 27(1):101– 118.

Yevgeni Berzak, Boris Katz, and Roger Levy. 2018. Assessing language proficiency from eye movements in reading. In Proceedings ofthe 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1986–1996, New Orleans, Louisiana. Association for Computational Linguistics.

Yevgeni Berzak and Roger Levy. 2023. Eye movement traces of linguistic knowledge in native and non-native reading. Open Mind, 7:179–196.

Yevgeni Berzak, Jonathan Malmaud, Omer Shubi, Yoav Meiri, Ella Lion, and Roger Levy. 2025. OneStop: A 360-participant English eye tracking dataset with different reading regimes. Scientific Data.

Yevgeni Berzak, Chie Nakamura, Suzanne Flynn, and Boris Katz. 2017. Predicting native language from gaze. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 541–551.

Yevgeni Berzak, Chie Nakamura, Amelia Smith, Emily Weng, Boris Katz, Suzanne Flynn, and Roger Levy. 2022. CELER: A 365-participant corpus of eye movements in L1 and L2 English reading. Open Mind, 6:41–50.

Eugene Charniak, Don Blaheta, Niyu Ge, Keith Hall, John Hale, and Mark Johnson. 2000. BLLIP 1987-89 WSJ corpus release 1. Linguistic Data Consortium, Philadelphia, 36.

Ping-Lin Chuang and Xun Yan. 2025. Language assessment in the era of generative artificial intelligence: Opportunities, challenges, and future directions. System, 134:103846.

Harald Clahsen and Claudia Felser. 2006. Grammatical processing in language learners. Applied Psycholinguistics, 27(1):3–42.

Chris Collins and Richard Kayne. 2009. Syntactic structures of the world’s languages (SSWL).

Uschi Cop, Denis Drieghe, and Wouter Duyck. 2015a. Eye movement patterns in natural reading: A comparison of monolingual and bilingual reading of a novel. PloS one, 10(8):e0134008.

Uschi Cop, Emmanuel Keuleers, Denis Drieghe, and Wouter Duyck. 2015b. Frequency effects in monolingual and bilingual natural reading. Psychonomic bulletin & review, 22(5):1216–1234.

Leana Copeland, Tom Gedeon, and Sumudu Mendis. 2014. Predicting reading comprehension scores from eye movements using artificial neural networks and fuzzy output error. Artificial Intelligence Research, 3(3):35–48.

Lee J Cronbach. 1951. Coefficient alpha and the internal structure of tests. Psychometrika, 16(3):297–334.

Lee J Cronbach, Peter Schönemann, and Douglas McKie. 1965. Alpha coefficients for stratifiedparallel tests. Educational and Psychological Measurement, 25(2):291–312.

Vera Demberg and Frank Keller. 2008. Data from eyetracking corpora as evidence for theories of syntactic processing complexity. Cognition, 109(2):193–210.

Matthew Dryer and Martin Haspelmath. 2022. The world atlas of language structures online.

Education First. 2025. EF EPI: EF English proficiency index: A ranking of 123 countries and regions by English skills. Accessed 29 August 2026.

Ralf Engbert, Antje Nuthmann, Eike M Richter, and Reinhold Kliegl. 2005. SWIFT: a dynamical model of saccade generation during reading. Psychological review, 112(4):777.

Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Mihir Manium, Shi Bin Hoo, Magnus Bühler, Anurag Garg, Dominik Safaric, Jake Robertson, Benjamin Jäger, Simone Alessi, Adrian Hayler, Vladyslav Moroshan, Lennart Purucker, Philipp Singer, Alan Arazi, and 22 others. 2026. TabPFN-3: Technical report. Preprint, arXiv:2605.13986.

Patrick Haller, Andreas Säuberli, Sarah Kiener, Jinger Pan, Ming Yan, and Lena Jäger. 2022. Eye-tracking based classification of Mandarin Chinese readers with and without dyslexia using neural sequence models. In Proceedings of the Workshop on Text Simplification, Accessibility, and Readability (TSAR-2022), pages 111–118.

Leo M Harvill. 1991. Standard error of measurement: an NCME instructional module on. Educational Measurement: issues and practice, 10(2):33–41.

Steph Holloway. 2013. ScriptSource – Writing systems, computers and people. SIL International.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spaCy: Industrialstrength Natural Language Processing in Python.

IELTS. 2026. Test statistics. Accessed August 18th, 2026.

Naoki Ikeda, Tony Clark, Spiros Papageorgiou, Lixiong Gu, Renka Ohta, Andrew Blackhurst, and Emma Bruce. 2025. Aligning scores of language proficiency tests: A score concordance study between IELTS Academic and TOEFL iBT®. ETS Research Report Series, 2025(1):1–28.

Lena A Jäger, Silvia Makowski, Paul Prasse, Sascha Liehr, Maximilian Seidler, and Tobias Scheffer. 2019. Deep eyedentification: Biometric identification using micro-movements of the eye. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 299–314. Springer.

Scott Jarvis and Aneta Pavlenko. 2008. Crosslinguistic influence in language and cognition. Routledge.

Eun Hee Jeon and Junko Yamashita. 2014. L2 reading comprehension and its correlates: A meta-analysis. Language learning, 64(1):160–212.

Marcel A Just and Patricia A Carpenter. 1980. A theory of reading: from eye fixations to comprehension. Psychological review, 87(4):329.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Aditya Khan, Mason Shipton, David Anugraha, Kaiyao Duan, Phuong H. Hoang, Eric Khiu, A. Seza Dogruöz, and En-Shiun Annie Lee. 2025.˘ URIEL+: Enhancing linguistic inclusion and usability in a typological and multilingual knowledge base. In Proceedings of the 31st International Conference on Computational Linguistics, pages 6937–6952. Association for Computational Linguistics.

Reinhold Kliegl, Ellen Grabner, Martin Rolfs, and Ralf Engbert. 2004. Length, frequency, and predictability effects of words on eye movements in reading. Europeanjournal ofcognitive psychology, 16(1-2):262– 284.

Michael J Kolen, Lingjia Zeng, and Bradley A Hanson. 1996. Conditional standard errors of measurement for scale scores using IRT. Journal of Educational Measurement, 33(2):129–140.

Victor Kuperman, Sascha Schroeder, Cengiz Acartürk, Niket Agrawal, Dominick M Alexandre, Lena S Bolliger, Jan Brasser, César Campos-Rojas, Denis Drieghe, Dušica Filipovic Ður´ devi¯ c, and 1 others.´ 2025. New data on text reading in English as a second language: The wave 2 expansion of the multilingual eye-movement corpus (MECO). Studies in Second Language Acquisition, 47(2):677–695.

Victor Kuperman, Noam Siegelman, Sascha Schroeder, Cengiz Acartürk, Svetlana Alexeeva, Simona Amenta, Raymond Bertram, Rolando Bonandrini, Marc Brysbaert, Daria Chernova, and 1 others. 2023. Text reading in English as a second language: Evidence from the multilingual eye-movements corpus. Studies in second language acquisition, 45(1):3–37.

Anna Laurinavichyute, Anastasiya Lopukhina, and David Robert Reich. 2025. Automatic detection of dyslexia based on eye movements during reading in Russian. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 59–66.

Kristin Lemhöfer and Mirjam Broersma. 2012. Introducing LexTALE: A quick and valid lexical test for advanced learners of English. Behavior research methods, 44(2):325–343.

M. Paul Lewis, Gary F. Simons, and Charles D. Fennig. 2015. Ethnologue: Languages of the world.

Venessa Manna, Shuhong Li, Spiros Papageorgiou, and Lixiong Gu. 2025. TOEFL iBT® technical manual. ETS Research Report Series, 2025(1).

Yoav Meiri, Omer Shubi, Cfir Avraham Hadar, Ariel Kreisberg Nitzav, and Yevgeni Berzak. 2025. Déjà vu? Decoding repeated reading from eye movements. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 19460–19482.

Yoav Meiri, Omer Shubi, Keren Gruteke Klein, David R. Reich, and Yevgeni Berzak. 2026. lacclab/psycholing-metrics: v1.1.9. Zenodo. https: //doi.org/10.5281/zenodo.20376826.

Diane C. Mézière, Lili Yu, Erik D. Reichle, Titus von der Malsburg, and Genevieve McArthur. 2023. Using eye-tracking measures to predict reading comprehension. Reading Research Quarterly, 58(3):425–449.

Terence Odlin. 1989. Language transfer. Cambridge, UK: Cambridge.

Slav Petrov, Dipanjan Das, and Ryan McDonald. 2012. A universal part-of-speech tagset. In Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC’12), pages 2089– 2096, Istanbul, Turkey. European Language Resources Association (ELRA).

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Keith Rayner, Jane Ashby, Alexander Pollatsek, and Erik D Reichle. 2004. The effects of frequency and predictability on eye fixations in reading: implications for the E-Z Reader model. Journal of Experimental Psychology: Human Perception and Performance, 30(4):720.

Keith Rayner, Alexander Pollatsek, Jane Ashby, and Charles Clifton Jr. 2012. Psychology of reading. Psychology Press.

David Robert Reich, Paul Prasse, Chiara Tschirner, Patrick Haller, Frank Goldhammer, and Lena A Jäger. 2022. Inferring native and non-native human reading comprehension and subjective text difficulty from scanpaths in reading. In 2022 Symposium on eye tracking research and applications, pages 1–8.

Erik D Reichle, Alexander Pollatsek, Donald L Fisher, and Keith Rayner. 1998. Toward a model of eye movement control in reading. Psychological review, 105(1):125.

Luz Rello and Miguel Ballesteros. 2015. Detecting readers with dyslexia using machine learning with eye tracking measures. In Proceedings of the 12th international web for all conference, pages 1–8.

Beatrice Santorini. 1990. Part-of-speech tagging guidelines for the Penn Treebank project (3rd revision, 2nd printing). Ms., Department of Linguistics, UPenn. Philadelphia, PA.

Elizabeth R Schotter and Brian Dillon. 2025. A beginner’s guide to eye tracking for psycholinguistic studies of reading. Behavior research methods, 57(2):68.

Omer Shubi, Cfir Avraham Hadar, and Yevgeni Berzak. 2025a. Decoding reading goals from eye movements. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5616–5637.

Omer Shubi, Yoav Meiri, Cfir Avraham Hadar, and Yevgeni Berzak. 2024. Fine-grained prediction of reading comprehension from eye movements. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 3372–3391.

Omer Shubi, David Robert Reich, Keren Gruteke Klein, Yuval Angel, Paul Prasse, Lena Ann Jäger, and Yevgeni Berzak. 2025b. EyeBench: Predictive modeling from eye movements in reading. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Lina Skerath, Paulina Toborek, Anita Zielinska, Maria´ Barrett, and Rob Van Der Goot. 2023. Native language prediction from gaze: a reproducibility study. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 4: Student Research Workshop), pages 152–159.

Robyn Speer, Joshua Chin, Andrew Lin, Sara Jewett, and Lance Nathan. 2018. LuminosoInsight/wordfreq: v2.2.

Joseph K Torgesen, Carol Alexander Rashotte, and Richard K Wagner. 1999. TOWRE: Test of word reading efficiency. Pro-ed Austin, TX.

Veronica Whitford and Debra Titone. 2012. Secondlanguage experience modulates first-and secondlanguage word frequency effects: Evidence from eye movement measures of natural paragraph reading. Psychonomic bulletin & review, 19(1):73–80.

Veronica Whitford and Debra Titone. 2017. The effects of word frequency and word predictability during first-and second-language paragraph reading in bilingual older and younger adults. Psychology and aging, 32(2):158.

## Appendix

## A Eye Movement Representations

## A.1 Average Fixation Metrics (Avg. Fix.)

This feature set includes the following 6 features.

1. First Fixation Duration (FF) the duration of the first fixation on a word.

2. Gaze Duration (GD) the time spent from first entering a word to first leaving it.

3. Total Fixation Duration (TF) the sum of all the fixation durations on a word.

4. Go-Past Duration (GP) the time from first entering a word until proceeding to its right.

5. Skip Rate (SR) The percentage of words skipped during first pass reading.

6. Regression Probability (RP) the percentage of words with backward saccades during firstpass reading.

FF, GD, TF and GP are averaged across all words, with skipped words assigned a value of 0. SR and RP are computed over all the textual materials presented to the reader.

## A.2 Syntactic Clusters (S-Clusters)

The S-Clusters feature set consists of 258 features. Each feature is one of the 6 reading measures above, averaged for a given part-of-speech tag. We use 43 part-of-speech tags, 14 Universal tags (Petrov et al., 2012) and 29 PTB tags (Santorini, 1990), extracted with spaCy (Honnibal et al., 2020).

## A.3 Word Property Coefficients (WP-Coefs)

The WP-Coefs feature set consists of 24 features. The features are the coefficients from a linear model RT ∼ 1 + length + surprisal + frequency, fitted for each participant on each of the measures $\mathrm { R T } ~ \in ~ \{ F F , G D , T F , G P \}$ . We use the same features in logistic regression models to predict word skips and regressions. . Surprisal and word frequency values were extracted using psycholing-metrics version 1.1.9 (Meiri et al., 2026).

## A.4 Transitions

Each feature represents the number of saccades in each direction between each pair of words in the passage. We include only pairs that had at least one transition for at least one participant. As described in Section 5.5, in MECO L2, we divide the passages into two parts and train on each part separately. This results in 39,442 features, on average, across the two parts. In OneStopL2 and OneStop, there are 6 unique sets of reading materials (one of the three article batches, in one of two text difficulty level sequences), with an average of 99,372 features.

As TabPFN-3 has an upper limit of 2,000 features, we keep a fixed number of Transition features from the middle of each paragraph, such that this limit it met. On average, there are 37 transition features per passage for OneStopL2 and OneStop, and 333 features per passage for MECO L2.

## A.5 Word Fixation Metrics (Word Fix.)

The Word Fixation Metrics feature set consists of an average of 1,658 features in MECO L2 across the two subsets of textual materials, and an average of 11,719 features in OneStopL2 and OneStop across the 6 textual material subsets. For TabPFN-3, we take into account only the first 2,000 words presented to the reader, which cover all 12 passages in MECO L2 and the first 9–11 passages in OneStop and OneStopL2.

## A.6 Differences from the Features of Berzak et al. (2018)

The Syntactic Clusters, Word Property Coefficients, Transitions and Word Fixation Metrics feature sets are similar, but not identical to those used in Berzak et al. (2018). Differently from Berzak et al. (2018), we do not speed-normalize the underlying eye movement measures, as preliminary experiments suggested that raw measures yielded better performance. We add GP, SR and RP to Syntactic Clusters, and regressions to Word Property Coefficients. When computing Word Property Coefficients, Surprisal is obtained using GPT-2 (Radford et al., 2019) rather than an n-gram model, and frequency counts are from Wordfreq (Speer et al., 2018) rather than from BLLIP-WSJ (Charniak et al., 2000), which matched the domain of their textual materials.

## B Experimental Setup

## B.1 Eye Tracking Datasets

We use the following versions of the eyetracking datasets:

• MECO L2 - v2.1.

• OneStop - v1.0.3.

• OneStopL2 - v0.1.

MECO L2 The MECO L2 dataset contains a total of 1,437,848 word tokens for which eye movement data were collected. The dataset comprises 2,628,941 fixations, corresponding to 2,189 fixations per participant. The L2 readers account for 1,311,713 word tokens and 2,465,736 fixations, corresponding to 2,229 fixations per participant, while the L1 readers account for 126,135 word tokens and 163,205 fixations, or 1,718 fixations per participant. Table 5 presents the number of participants from each L1 background.

<table><tr><td>L1</td><td>Participants</td></tr><tr><td>Basque</td><td>35</td></tr><tr><td>Brazilian Portuguese</td><td>54</td></tr><tr><td>Danish</td><td>25</td></tr><tr><td>Dutch</td><td>46</td></tr><tr><td>Estonian</td><td>58</td></tr><tr><td>Finnish</td><td>51</td></tr><tr><td>German</td><td>135</td></tr><tr><td>Greek</td><td>48</td></tr><tr><td>Hebrew</td><td>45</td></tr><tr><td>Hindi</td><td>98</td></tr><tr><td>Icelandic</td><td>45</td></tr><tr><td>Italian</td><td>51</td></tr><tr><td>Mandarin</td><td>89</td></tr><tr><td>Norwegian</td><td>62</td></tr><tr><td>Russian</td><td>96</td></tr><tr><td>Serbian</td><td>43</td></tr><tr><td></td><td></td></tr><tr><td>Spanish Turkish</td><td>86 39</td></tr><tr><td></td><td></td></tr><tr><td>English (L1)</td><td>95</td></tr></table>

Table 5: MECO L2 Participants. Number of participants from each L1 background.

OneStopL2 We exclude the repeated reading portion of OneStopL2 and OneStop. The first reading portion of OneStopL2 contains 1,600,086 word tokens for which eye movement data were collected. The dataset comprises 2,677,486 fixations, corresponding to 9,772 fixations per participant. Table 6 presents the number of participants from each L1 background in the two conditions and overall.

OneStop The first reading portion of OneStop contains 2,097,963 word tokens for which eye movement data were collected. The dataset comprises 2,027,969 fixations, corresponding to 5,665 fixations per participant.

<table><tr><td>L1</td><td>Ordinary Reading</td><td>Information Seeking</td><td>All</td></tr><tr><td>Arabic</td><td>14</td><td>11</td><td>25</td></tr><tr><td>Chinese</td><td>18</td><td>15</td><td>33</td></tr><tr><td>French</td><td>16</td><td>6</td><td>22</td></tr><tr><td>Hebrew</td><td>15</td><td>17</td><td>32</td></tr><tr><td>Japanese</td><td>13</td><td>6</td><td>19</td></tr><tr><td>Korean</td><td>18</td><td>13</td><td>31</td></tr><tr><td>Portuguese</td><td>18</td><td>17</td><td>35</td></tr><tr><td>Russian</td><td>17</td><td>17</td><td>34</td></tr><tr><td>Spanish</td><td>17</td><td>18</td><td>35</td></tr><tr><td>Vietnamese</td><td>5</td><td>3</td><td>8</td></tr></table>

Table 6: OneStopL2 Participants. Number of participants from each L1 background in the reading for comprehension and information seeking regimes.

## B.2 Data Splits

As noted in Section 5.2, external proficiency test scores are not available for all participants. When evaluating EyeScore against these scores, we retain all participants in the training portion of the splits of Section 5.5, regardless of score availability, and evaluate on the held-out participants who have a score for the test in question. For predicting scores on external tests, in contrast, we train and evaluate only on participants with a reported score for that test. Below, we describe dataset-specific evaluation constraints and the corresponding preprocessing and sampling procedures.

OneStop and OneStopL2 We keep only participants who had no missing passages in their eye movement recordings. This results in the exclusion of one participant from the information seeking portion of OneStopL2, three from the ordinary reading for comprehension portion of OneStopL2, and two from the ordinary reading for comprehension portion of OneStop. In OneStopL2 v0.1, for a given text, there is a difference in the number of participants reading it in each of the two difficulty versions (original vs simplified), unlike in OneStop. Thus, for the prediction of each external proficiency score in OneStopL2, instead of including all L1 participants in the training set of each split, we include only L1 participants matched to L2 participants in that training set. Specifically, each L2 participant is assigned an L1 participant who read the same texts at the same difficulty versions.

OneStopL2 Information Seeking Section 5.5 reports the training-split statistics for the ordinary reading for comprehension portion of OneStopL2. For the information seeking part, the corresponding averages are 20 L2 and 20 L1 training participants in the Fixed Text regime, and 81 L2 and 81 L1 in the Any Text regime.

## B.3 Hyperparameter Tuning

Ridge regression In the Any Text regime, for MECO L2, we tune the Ridge regression regularization parameter α twice for each participant, once for each half of their texts, using all other participants’ other texts via leave-one-participantout cross validation. For OneStop, we tune α via leave-one-participant-out cross validation.

In the Fixed Text regime, for MECO L2, we again tune α twice for each participant - once for each half of their texts, this time using all other participants’ same texts via leave-one-participantout cross validation. For OneStop, we tune α for each participant using all other participants in their textual material group via leave-one-participant-out cross validation.

LightGBM In both regimes, we use 5-fold cross validation. The hyperparameter grid search comprised maximum leaf counts of 15, 31, and 63 per tree, learning rates of 0.03 and 0.1, and featuresampling fractions of 0.3 and 1.0, where 1.0 indicates that all features were considered.

TabPFN-3 we use the pretrained model, without any hyperparameter tuning.

## C The Information Seeking Reading Regime

<table><tr><td colspan="3">LexTALE</td><td colspan="2">Michigan</td></tr><tr><td>Features</td><td>Fixed</td><td>Any</td><td>Fixed</td><td>Any</td></tr><tr><td>WPM</td><td>0.56</td><td>0.59</td><td>0.45</td><td>0.44</td></tr><tr><td>Avg. Fix.</td><td>0.49</td><td>0.49*</td><td>0.47</td><td>0.44</td></tr><tr><td>S-Clusters</td><td>0.50</td><td>0.50*</td><td>0.47</td><td>0.45</td></tr><tr><td>WP-Coefs</td><td>0.48</td><td>0.51</td><td>0.43</td><td>0.43</td></tr><tr><td>Transitions</td><td>0.58</td><td>一</td><td>0.49</td><td>-</td></tr><tr><td>Word Fix.</td><td>0.60</td><td>-</td><td>0.52*</td><td>-</td></tr></table>

Table 7: EyeScore. Pearson r correlations of Eye-Score with standard English proficiency tests in the OneStopL2 information seeking regime.

<table><tr><td rowspan="3"></td><td colspan="4">LexTALE</td><td colspan="4">Michigan</td></tr><tr><td colspan="2">Fixed</td><td colspan="2">Any</td><td colspan="2">Fixed</td><td colspan="2">Any</td></tr><tr><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td></tr><tr><td>Avg. Train</td><td>-0.09</td><td>11.27</td><td>0.00</td><td>10.93</td><td>-0.09</td><td>10.81</td><td>0.00</td><td>10.40</td></tr><tr><td>WPM</td><td>0.53</td><td>9.95</td><td>0.52</td><td>10.21</td><td>0.29</td><td>8.90</td><td>0.38</td><td>8.46</td></tr><tr><td>Avg. Fix.</td><td>0.53</td><td>10.04</td><td>0.60</td><td>9.05*</td><td>0.36</td><td>8.81</td><td>0.55**</td><td>7.95</td></tr><tr><td>S-Clusters</td><td>0.50</td><td>9.94</td><td>0.59</td><td>8.96*</td><td>0.40</td><td>8.39</td><td>0.53**</td><td>8.05</td></tr><tr><td>WP-Coefs</td><td>0.43*</td><td>11.01</td><td>0.57</td><td>9.58</td><td>0.33</td><td>8.90</td><td>0.48</td><td>8.70</td></tr><tr><td>Transitions</td><td>0.53</td><td>9.72</td><td></td><td></td><td>0.42*</td><td>7.98**</td><td></td><td></td></tr><tr><td>Word Fix.</td><td>0.51</td><td>9.77</td><td></td><td></td><td>0.53***</td><td>7.76**</td><td>1</td><td></td></tr></table>

Table 8: Prediction of standard English proficiency test scores with Ridge regression in the OneStopL2 information seeking regime.

## D Prediction of External Proficiency Tests with LightGBM and TabPFN-3

<table><tr><td></td><td colspan="4">LexTALE</td><td colspan="4">Composite</td></tr><tr><td></td><td colspan="2">Fixed</td><td colspan="2">Any</td><td colspan="2">Fixed</td><td colspan="2">Any</td></tr><tr><td>Features</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td></tr><tr><td>Avg. Train</td><td>0.00</td><td>9.92</td><td>0.00</td><td>9.92</td><td>0.00</td><td>11.37</td><td>0.00</td><td>11.37</td></tr><tr><td>WPM</td><td> $0 . 4 8 _ { ( - 0 . 0 0 ) }$ </td><td> $8 . 4 3 _ { ( - 0 . 1 0 ) }$ </td><td> $0 . 4 6 _ { ( - 0 . 0 2 ) }$ </td><td> $8 . 6 2 _ { ( + 0 . 1 0 ) }$ </td><td> $0 . 4 5 _ { ( + 0 . 0 1 ) }$ </td><td> $9 . 9 0 _ { ( - 0 . 0 6 ) }$ </td><td> $0 . 4 6 _ { ( + 0 . 0 2 ) }$ </td><td> $9 . 8 0 _ { ( - 0 . 1 5 ) }$ </td></tr><tr><td>Avg. Fix.</td><td> $0 . 5 4 _ { ( - 0 . 0 0 ) } ^ { * * * }$ </td><td> $8 . 1 4 _ { ( - 0 . 0 2 ) } ^ { \ast }$ </td><td> $0 . 5 4 _ { ( - 0 . 0 0 ) } ^ { * * * }$ </td><td> $8 . 1 1 _ { ( - 0 . 0 6 ) } ^ { * * * }$ </td><td> $0 . 5 2 _ { ( - 0 . 0 2 ) } ^ { \ast \ast \ast }$ </td><td> $9 . 5 0 _ { ( + 0 . 1 1 ) } ^ { * * }$ </td><td> $\mathbf { 0 . 5 3 } _ { ( - 0 . 0 1 ) } ^ { * * * }$ </td><td> $9 . 5 1 _ { ( + 0 . 1 0 ) } ^ { * }$ </td></tr><tr><td>S-Clusters</td><td> $0 . 5 8 _ { ( + 0 . 0 3 ^ { * } ) } ^ { * * * }$ </td><td> $7 . 8 2 _ { ( - 0 . 2 8 ^ { * * * } ) } ^ { * * * }$ </td><td> $\mathbf { 0 . 5 5 } _ { ( + 0 . 0 0 ) } ^ { * * * }$ </td><td> $8 . 0 6 _ { ( - 0 . 0 9 ) } ^ { * * * }$ </td><td> $0 . 5 7 _ { ( + 0 . 0 1 ) } ^ { * * * }$ </td><td> $9 . 2 1 _ { ( - 0 . 0 5 ) } ^ { * * * }$ </td><td> $\mathbf { 0 . 5 3 } _ { ( - 0 . 0 1 ) } ^ { * * * }$ </td><td> $\mathbf { 9 . 4 6 } _ { ( + 0 . 0 7 ) } ^ { * * }$ </td></tr><tr><td>WP-Coefs</td><td> $0 . 5 4 _ { ( - 0 . 0 2 ) } ^ { * * * }$ </td><td> $8 . 1 1 _ { ( - 0 . 0 1 ) } ^ { * * }$ </td><td> $\mathbf { 0 . 5 5 } _ { ( - 0 . 0 0 ) } ^ { * * * }$ </td><td> $\mathbf { 8 . 0 5 } _ { ( - 0 . 0 9 ) } ^ { * * * }$ </td><td> $0 . 5 3 _ { ( - 0 . 0 2 ) } ^ { * * * }$ </td><td> $9 . 4 4 _ { ( + 0 . 1 4 ) } ^ { * * * }$ </td><td> $\mathbf { 0 . 5 3 } _ { ( - 0 . 0 1 ) } ^ { * * * }$ </td><td> $9 . 5 7 _ { ( + 0 . 1 7 ) }$ </td></tr><tr><td>Transitions</td><td> $0 . 5 9 _ { ( + 0 . 0 4 ^ { * * * } ) } ^ { * * * }$ </td><td> $7 . 7 4 _ { ( - 0 . 3 4 ^ { * * * } ) } ^ { * * * }$ </td><td></td><td></td><td> $0 . 5 3 _ { ( - 0 . 0 2 ^ { * } ) } ^ { * * * }$ </td><td> $9 . 3 4 _ { ( - 0 . 0 3 ) } ^ { * * * }$ </td><td>一</td><td></td></tr><tr><td>Word Fix.</td><td> $\mathbf { 0 . 6 3 } _ { ( + 0 . 0 1 ) } ^ { * * * }$ </td><td> $\mathbf { 7 . 4 5 } _ { ( - 0 . 1 4 ) } ^ { * * * }$ </td><td></td><td></td><td> $\mathbf { 0 . 6 1 } _ { ( + 0 . 0 2 ) } ^ { * * * }$ </td><td> $\mathbf { 8 . 7 6 } _ { ( - 0 . 1 0 ) } ^ { * * * }$ </td><td></td><td></td></tr></table>

Table 9: Prediction of standard English proficiency test scores with LightGBM on MECO L2. Statistically significant differences from the WPM baseline are marked in the superscript. Differences from the Ridge regression results (Table 2) are marked in the subscript in blue when better than Ridge regression and red when worse. Statistically significant differences (bootstrap test) are marked with $^ { \ast \ast , } p < 0 . 0 5 ,$ ‘<sup>∗∗</sup>’ p < 0.01, ‘<sup>∗∗∗</sup>’ $p < 0 . 0 0 1$

<table><tr><td></td><td colspan="4">LexTALE</td><td colspan="4">Michigan</td></tr><tr><td></td><td colspan="2">Fixed</td><td colspan="2">Any</td><td colspan="2">Fixed</td><td colspan="2">Any</td></tr><tr><td>Features</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td></tr><tr><td>Avg. Train</td><td>-0.06</td><td>10.32</td><td>-0.02</td><td>10.25</td><td>-0.07</td><td>7.52</td><td> $- 0 . 0 1$ </td><td>7.40</td></tr><tr><td>WPM</td><td> $0 . 1 9 _ { ( - 0 . 2 3 ^ { * * } ) }$ </td><td> $1 1 . 2 6 _ { ( + 1 . 0 6 ^ { * } ) }$ </td><td> $0 . 4 4 _ { ( - 0 . 0 4 ) }$ </td><td> $9 . 5 6 _ { ( - 0 . 0 6 ) }$ </td><td> $0 . 1 3 _ { ( - 0 . 2 4 ^ { * * * } ) }$ </td><td> $7 . 5 4 _ { ( + 1 . 1 2 ^ { * * * } ) }$ </td><td> $0 . 4 5 _ { ( - 0 . 0 3 ) }$ </td><td> $6 . 3 7 _ { ( + 0 . 2 3 ) }$ </td></tr><tr><td>Avg. Fix.</td><td> $0 . 2 6 _ { ( - 0 . 2 1 ^ { * * } ) }$ </td><td> $1 0 . 9 4 _ { ( + 0 . 9 4 ) }$ </td><td> $0 . 4 5 _ { ( - 0 . 0 9 ^ { \ast } ) }$ </td><td> $9 . 8 5 _ { ( + 0 . 8 1 ^ { \ast } ) }$ </td><td> $0 . 1 6 _ { ( - 0 . 3 0 ^ { * * * } ) }$ </td><td> $7 . 4 4 _ { ( + 1 . 1 0 ^ { * * } ) }$ </td><td> $0 . 4 9 _ { ( - 0 . 0 7 ^ { * * } ) }$ </td><td> $6 . 4 0 _ { ( + 0 . 5 3 ^ { \ast } ) }$ </td></tr><tr><td>S-Clusters</td><td> $0 . 3 6 _ { ( - 0 . 1 3 ^ { * * } ) } ^ { * * }$ </td><td> $1 0 . 1 9 _ { ( + 0 . 4 4 ) } ^ { \ast }$ </td><td> $\mathbf { 0 . 5 3 } _ { ( - 0 . 0 4 ) }$ </td><td> $9 . 1 3 _ { ( + 0 . 4 6 ) }$ </td><td> $0 . 2 4 _ { ( - 0 . 2 3 ^ { * * * } ) }$ </td><td> $7 . 2 8 _ { ( + 1 . 0 3 ^ { * * * } ) }$ </td><td> $\mathbf { 0 . 5 0 } _ { ( - 0 . 0 7 ^ { * } ) }$ </td><td> ${ \bf 6 . 1 3 } _ { ( + 0 . 3 1 ) }$ </td></tr><tr><td>WP-Coefs</td><td> $0 . 3 4 _ { ( - 0 . 1 4 ^ { * } ) } ^ { * }$ </td><td> $1 0 . 3 0 _ { ( + 0 . 6 6 ) }$ </td><td>0.51(−0.07*)</td><td> ${ \bf 8 . 9 1 } _ { ( + 0 . 3 3 ) }$ </td><td> $0 . 2 8 _ { ( - 0 . 1 8 ^ { * } ) } ^ { * }$ </td><td> $7 . 1 4 _ { ( + 0 . 7 7 ^ { * } ) }$ </td><td> $0 . 4 8 _ { ( - 0 . 0 8 ) }$ </td><td> $6 . 1 3 _ { ( + 0 . 2 2 ) }$ </td></tr><tr><td>Transitions</td><td> $0 . 4 5 _ { ( - 0 . 0 6 ) } ^ { * * * }$ </td><td> $1 0 . 1 9 _ { ( + 0 . 4 1 ) }$ </td><td></td><td></td><td> $0 . 2 9 _ { ( - 0 . 1 7 * * ) }$ </td><td> $6 . 9 6 _ { ( + 0 . 8 9 * * * ) }$ </td><td></td><td></td></tr><tr><td>Word Fix.</td><td> $\mathbf { 0 . 4 6 } _ { ( - 0 . 0 9 * ) } ^ { * * * }$ </td><td> $\mathbf { 9 . 6 8 } _ { ( + 0 . 8 2 ) } ^ { * * }$ </td><td>-</td><td></td><td> $\mathbf { 0 . 4 9 } _ { ( - 0 . 0 3 ) } ^ { * * * }$ </td><td> $\mathbf { 5 . 9 5 } _ { ( + 0 . 1 0 ) } ^ { * * * }$ </td><td></td><td></td></tr></table>

Table 10: Prediction of standard English proficiency test scores with LightGBM on OneStopL2.

<table><tr><td rowspan="2"></td><td colspan="4">LexTALE</td><td colspan="3">Composite</td></tr><tr><td colspan="2">Fixed</td><td colspan="2"> $\operatorname { A n y }$ </td><td colspan="2">Fixed</td></tr><tr><td>Features</td><td></td><td>MAE</td><td>MAE</td><td>r</td><td>MAE</td><td></td><td>MAE</td></tr><tr><td>Avg. Train</td><td>0.00</td><td>9.92</td><td>0.00</td><td>9.92</td><td>11.37</td><td>0.00</td><td>11.37</td></tr><tr><td>WPM</td><td> $0 . 4 9 _ { ( + 0 . 0 1 ^ { * } ) }$ </td><td> $8 . 4 1 _ { ( - 0 . 1 2 ^ { * * } ) }$ </td><td> $0 . 4 9 _ { ( + 0 . 0 1 ^ { * } ) }$ </td><td> $8 . 4 2 _ { ( - 0 . 1 0 ^ { * } ) }$ </td><td> $0 . 4 8 _ { ( + 0 . 0 3 ^ { * * * } ) }$   $9 . 7 1 _ { ( - 0 . 2 5 ^ { * * * } ) }$ </td><td> $0 . 4 8 _ { ( + 0 . 0 3 ^ { * * * } ) }$ </td><td> $9 . 7 2 _ { ( - 0 . 2 4 ^ { * * * } ) }$ </td></tr><tr><td>Avg. Fix.</td><td> $0 . 5 7 _ { ( + 0 . 0 3 ^ { * * * } ) } ^ { * * * }$ </td><td> $7 . 9 5 _ { ( - 0 . 2 2 ^ { * * } ) } ^ { * * * }$ </td><td> $0 . 5 7 _ { ( + 0 . 0 3 ^ { * * * } ) } ^ { * * * }$ </td><td> $7 . 9 3 _ { ( - 0 . 2 4 ^ { * * * } ) } ^ { * * * }$ </td><td> $0 . 5 6 _ { ( + 0 . 0 2 ^ { * } ) } ^ { * * * }$   $9 . 1 6 _ { ( - 0 . 2 3 ^ { * * * } ) } ^ { * * * }$ </td><td> $\mathbf { 0 . 5 6 } _ { ( + 0 . 0 2 ^ { * } ) } ^ { * * * }$ </td><td> $\mathbf { 9 . 1 7 } _ { ( - 0 . 2 4 ^ { * * * } ) } ^ { * * * }$ </td></tr><tr><td>S-Clusters</td><td> $0 . 5 9 _ { ( + 0 . 0 4 ^ { * * * } ) } ^ { * * * }$ </td><td> $7 . 7 7 _ { ( - 0 . 3 3 ^ { * * * } ) } ^ { * * * }$ </td><td> $\mathbf { 0 . 5 8 } _ { ( + 0 . 0 3 ^ { * * * } ) } ^ { * * * }$ </td><td> $\boldsymbol { 7 . 8 7 } _ { ( - 0 . 2 7 * * * ) } ^ { * * * }$ </td><td> $0 . 5 7 _ { ( + 0 . 0 2 ^ { * } ) } ^ { * * * }$   $9 . 1 1 _ { ( - 0 . 1 5 ^ { * } ) } ^ { * * * }$ </td><td> $0 . 5 5 _ { ( + 0 . 0 1 ^ { * } ) } ^ { * * * }$ </td><td> $9 . 2 6 _ { ( - 0 . 1 2 ^ { * } ) } ^ { * * * }$ </td></tr><tr><td>WP-Coefs</td><td> $0 . 5 8 _ { ( + 0 . 0 3 ^ { * * * } ) } ^ { * * * }$ </td><td> $7 . 8 7 _ { ( - 0 . 2 5 ^ { * * * } ) } ^ { * * * }$ </td><td> $\mathbf { 0 . 5 8 } _ { ( + 0 . 0 2 ^ { * * } ) } ^ { * * * }$ </td><td> $7 . 9 1 _ { ( - 0 . 2 3 ^ { * * * } ) } ^ { * * * }$ </td><td> $0 . 5 5 _ { ( + 0 . 0 0 ) } ^ { * * * }$   $9 . 2 7 _ { ( - 0 . 0 3 ) } ^ { * * * }$ </td><td> $0 . 5 5 _ { ( + 0 . 0 1 ) } ^ { * * * }$ </td><td> $9 . 3 5 _ { ( - 0 . 0 6 ) } ^ { * * * }$ </td></tr><tr><td>Transitions</td><td> $0 . 5 4 _ { ( - 0 . 0 1 ) } ^ { * }$ </td><td> $8 . 2 3 _ { ( + 0 . 1 5 ^ { * } ) }$ </td><td>=</td><td></td><td> $0 . 5 1 _ { ( - 0 . 0 5 ^ { * * * } ) }$   $9 . 6 3 _ { ( + 0 . 2 6 ^ { * * * } ) }$ </td><td>1</td><td></td></tr><tr><td>Word Fix.</td><td> $\mathbf { 0 . 6 3 } _ { ( + 0 . 0 2 ) } ^ { * * * }$ </td><td> $\mathbf { 7 . 4 0 } _ { ( - 0 . 1 9 ^ { * } ) } ^ { * * * }$ </td><td>=</td><td></td><td> $\mathbf { 0 . 6 0 } _ { ( + 0 . 0 1 ) } ^ { * * * }$   $\mathbf { 8 . 7 7 } _ { ( - 0 . 0 9 ) } ^ { \ast \ast \ast }$ </td><td>=</td><td>=</td></tr></table>

Table 11: Prediction of standard English proficiency test scores with TabPFN-3 on MECO L2.

<table><tr><td rowspan="3"></td><td colspan="4">LexTALE</td><td colspan="4">Michigan</td></tr><tr><td colspan="2">Fixed</td><td colspan="2">Any</td><td colspan="2">Fixed</td><td colspan="2">Any</td></tr><tr><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td><td>r</td><td>MAE</td></tr><tr><td>Avg. Train</td><td> $- 0 . 0 6$ </td><td>10.32</td><td> $- 0 . 0 2$ </td><td>10.25</td><td> $- 0 . 0 7$ </td><td>7.52</td><td> $- 0 . 0 1$ </td><td>7.40</td></tr><tr><td>WPM</td><td> $0 . 4 2 _ { ( + 0 . 0 0 ) }$ </td><td> $9 . 8 0 _ { ( - 0 . 4 0 ) }$ </td><td> $0 . 4 9 _ { ( + 0 . 0 1 ) }$ </td><td> $9 . 4 3 _ { ( - 0 . 1 9 ) }$ </td><td> $0 . 4 1 _ { ( + 0 . 0 4 ) }$ </td><td> $6 . 6 6 _ { ( + 0 . 2 4 ) }$ </td><td> $0 . 4 9 _ { ( + 0 . 0 2 ) }$ </td><td> $6 . 2 2 _ { ( + 0 . 0 8 ) }$ </td></tr><tr><td>Avg. Fix.</td><td>0.46(−0.01)</td><td> $9 . 7 9 _ { ( - 0 . 2 2 ) }$ </td><td> $0 . 5 3 _ { ( - 0 . 0 1 ) }$ </td><td> $9 . 1 0 _ { ( + 0 . 0 6 ) }$ </td><td> $0 . 4 7 _ { ( + 0 . 0 2 ) }$ </td><td> $6 . 7 9 _ { ( + 0 . 4 5 ^ { * } ) }$ </td><td>0.56(−0.00)</td><td> $6 . 0 4 _ { ( + 0 . 1 8 ) }$ </td></tr><tr><td>S-Clusters</td><td> $0 . 5 0 _ { ( + 0 . 0 1 ) }$ </td><td> $8 . 9 9 _ { ( - 0 . 7 5 ^ { \ast } ) }$ </td><td> $0 . 5 4 _ { ( - 0 . 0 3 ) }$ </td><td> $9 . 0 5 _ { ( + 0 . 3 8 ) }$ </td><td> $0 . 3 7 _ { ( - 0 . 1 0 ^ { * * } ) }$ </td><td> $7 . 1 4 _ { ( + 0 . 8 9 * * * ) }$ </td><td> $\mathbf { 0 . 5 7 } _ { ( + 0 . 0 0 ) }$ </td><td> $\mathbf { 5 . 9 1 } _ { ( + 0 . 1 0 ) }$ </td></tr><tr><td>WP-Coefs</td><td> $0 . 5 0 _ { ( + 0 . 0 2 ) }$ </td><td> $9 . 0 6 _ { ( - 0 . 5 9 ) }$ </td><td> $\mathbf { 0 . 5 8 } _ { ( - 0 . 0 0 ) } ^ { * }$ </td><td> $\mathbf { 8 . 3 6 } _ { ( - 0 . 2 2 ) } ^ { * * }$ </td><td> $0 . 4 2 _ { ( - 0 . 0 4 ) }$ </td><td> $7 . 0 0 _ { ( + 0 . 6 4 ^ { * } ) }$ </td><td> $0 . 5 5 _ { ( - 0 . 0 1 ) }$ </td><td> $6 . 0 1 _ { ( + 0 . 1 1 ) }$ </td></tr><tr><td>Transitions Word Fix.</td><td> $0 . 5 1 _ { ( + 0 . 0 1 ) }$   $\mathbf { 0 . 5 6 } _ { ( - 0 . 0 0 ) } ^ { * * * }$ </td><td> $9 . 4 9 _ { ( - 0 . 2 9 ) }$   $\mathbf { 8 . 9 6 } _ { ( + 0 . 1 0 ) } ^ { * }$ </td><td></td><td></td><td> $0 . 4 9 _ { ( + 0 . 0 2 ) } ^ { \ast }$   $\mathbf { 0 . 5 5 } _ { ( + 0 . 0 3 ) } ^ { * * * }$ </td><td> $6 . 3 1 _ { ( + 0 . 2 3 ) }$   $\mathbf { 5 . 8 7 } _ { ( + 0 . 0 1 ) } ^ { * * }$ </td><td></td><td></td></tr></table>

Table 12: Prediction of standard English proficiency test scores with TabPFN-3 on OneStopL2.

## E Measuring and Mitigating L1 Bias

## E.1 L1 Bias Across Feature Sets and Datasets

![](images/51f26f4e36c2e8699fc264f635878aa637341672802dfd5db3db76b078f76417.jpg)  
Figure 2: L1-English Proximity Bias in EyeScore, MECO L2. Residuals of an EyeScore ∼ LexTALE regression as a function of the linguistic distance of the participant’s L1 to English, in the Any Text regime.

![](images/221afe3597f2d0accdfb05e8843575f5eeaff9b323f9e4f02d3734d0c4bffa2b.jpg)  
Figure 3: L1-English Proximity Bias in EyeScore, OneStopL2. Residuals of an EyeScore ∼ LexTALE regression as a function of the linguistic distance of the participant’s L1 to English, in the Any Text regime.

![](images/5fba494bcfb79b365e39e67544ea50a7b0fe46e82153f2f426fc32db56bdcc2c.jpg)  
Figure 4: L1-English Proximity Bias in EyeScore, MECO L2. Residuals of an EyeScore ∼ Composite regression as a function of the linguistic distance of the participant’s L1 to English, in the Any Text regime.

![](images/956492250d95b4ab6fa35b428ee5181edb09f8ada924bb845c24e8e255fffbfd.jpg)  
Figure 5: L1-English Proximity Bias in EyeScore, OneStopL2. Residuals of an EyeScore ∼ Michigan regression as a function of the linguistic distance of the participant’s L1 to English, in the Any Text regime.

## E.2 L1 Bias: Pearson r

<table><tr><td></td><td colspan="6">MECO L2</td><td colspan="6">OneStopL2</td></tr><tr><td></td><td colspan="3">LexTALE</td><td colspan="3">Composite</td><td colspan="3">LexTALE</td><td colspan="3">Michigan</td></tr><tr><td></td><td>EyeScore</td><td> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td></td><td>EyeScore</td><td> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td></td><td>EyeScore</td><td> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td></td><td>EyeScore</td><td colspan="2"> $\mathrm { E y e S c o r e ^ { d b } }$ </td></tr><tr><td>Features</td><td></td><td>Seen Ll</td><td>New Ll</td><td></td><td>Seen Ll</td><td>New Ll</td><td></td><td>Seen Ll</td><td>New Ll</td><td></td><td>Seen Ll</td><td>New Ll</td></tr><tr><td>WPM</td><td> $- 0 . 1 3 ^ { * * * }$ </td><td>0.00</td><td>-0.04</td><td>-0.18***</td><td>-0.06</td><td>-0.09**</td><td>-0.12</td><td>0.00</td><td>0.00</td><td>-0.17*</td><td>-0.06</td><td>-0.05</td></tr><tr><td>Avg. Fix.</td><td> $- 0 . 1 2 ^ { * * * }$ </td><td>0.00</td><td>-0.02</td><td>-0.17***</td><td>-0.06</td><td>-0.07*</td><td>-0.10</td><td>0.00</td><td>0.00</td><td>-0.16</td><td>-0.06</td><td>-0.05</td></tr><tr><td>S-Clusters</td><td> $- 0 . 0 9 ^ { * * }$ </td><td>0.00</td><td>-0.02</td><td> $- 0 . 1 5 ^ { * * * }$ </td><td>-0.06</td><td>-0.08*</td><td>-0.09</td><td>0.00</td><td>0.01</td><td>-0.15</td><td>-0.06</td><td>-0.05</td></tr><tr><td>WP-Coefs</td><td> $- 0 . 1 8 ^ { * * * }$ </td><td>-0.01</td><td>-0.03</td><td> $- 0 . 2 3 ^ { \ast \ast \ast }$ </td><td>-0.06*</td><td>-0.09**</td><td>-0.14</td><td>0.00</td><td>0.00</td><td>-0.20*</td><td>-0.06</td><td>-0.05</td></tr></table>

Table 13: L1-English Proximity Bias in EyeScore and EyeScore<sup>db</sup> in the Any Text regime. Pearson r of the model ${ \mathrm { r e s } } _ { p , \mathrm { T e s t } } \sim d ( \mathrm { L } 1 _ { p } ,$ English).

E.3 Debiasing Obtained by Composite and Michigan
<table><tr><td rowspan="3"></td><td colspan="3">LexTALE</td><td colspan="3">Composite</td></tr><tr><td>EyeScore</td><td colspan="2"> $\mathrm { E y e S c o r e } ^ { \mathrm { d b } }$ </td><td>EyeScore</td><td colspan="2"> $\mathrm { E y e S c o r e } ^ { \mathrm { d b } }$ </td></tr><tr><td>Features</td><td>Seen Ll</td><td>New Ll</td><td></td><td>Seen Ll</td><td>New Ll</td></tr><tr><td>WPM</td><td> $- 1 . 3 7 ^ { * * * }$ </td><td>0.52</td><td>0.30</td><td> $- 1 . 9 7 ^ { * * * }$ </td><td>-0.01</td><td>-0.24</td></tr><tr><td>Avg. Fix.</td><td> $- 0 . 7 8 ^ { * * * }$ </td><td>0.35</td><td>0.35</td><td> $- 1 . 1 8 ^ { * * * }$  一</td><td>-0.01</td><td>-0.02</td></tr><tr><td>S-Clusters</td><td> $- 0 . 4 3 ^ { * * }$ </td><td>0.25</td><td>0.23</td><td> $- 0 . 7 1 ^ { * * * }$ </td><td>0.00</td><td>-0.02</td></tr><tr><td>WP-Coefs</td><td> $- 0 . 5 2 ^ { * * * }$ </td><td>0.16</td><td>0.13</td><td> $- 0 . 7 1 ^ { * * * }$ </td><td>0.00</td><td>-0.04</td></tr></table>

Table 14: L1-English Proximity Bias in EyeScore and EyeScore<sup>db</sup> in the Any Text regime, MECO L2. EyeScore<sup>db</sup> obtained by debiasing EyeScore relative to the Composite test.
<table><tr><td rowspan="3"></td><td colspan="3">LexTALE</td><td colspan="3">Michigan</td></tr><tr><td>EyeScore</td><td colspan="2"> $\overline { { \mathrm { E y e S c o r e } ^ { \mathrm { d b } } } }$ </td><td>EyeScore</td><td colspan="2"> $\mathrm { E y e S c o r e } ^ { \mathrm { d b } }$ </td></tr><tr><td></td><td>Seen Ll</td><td>New Ll</td><td></td><td>Seen Ll</td><td>New Ll</td></tr><tr><td>WPM</td><td>-1.28</td><td>0.64</td><td>0.61</td><td>-1.87*</td><td>0.03</td><td>0.06</td></tr><tr><td>Avg. Fix.</td><td>-0.75</td><td>0.41</td><td>0.47</td><td>-1.15</td><td>0.02</td><td>0.09</td></tr><tr><td>S-Clusters</td><td>-0.53</td><td>0.34</td><td>0.41</td><td>-0.88</td><td>0.00</td><td>0.08</td></tr><tr><td>WP-Coefs</td><td>-0.63</td><td>0.23</td><td>0.26</td><td>-0.88*</td><td>0.00</td><td>0.03</td></tr></table>

Table 15: L1-English Proximity Bias in EyeScore and EyeScore<sup>db</sup> in the Any Text regime, OneStopL2. EyeScore<sup>db</sup> obtained by debiasing EyeScore relative to the Michigan test.

E.4 L1 Bias in the Fixed Text and Information Seeking Regimes
<table><tr><td rowspan="3"></td><td colspan="6">MECO L2</td><td colspan="4">OneStopL2</td></tr><tr><td colspan="3">LexTALE</td><td colspan="3">Composite</td><td colspan="2">LexTALE</td><td colspan="3">Michigan</td></tr><tr><td>EyeScore</td><td> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td></td><td>EyeScore</td><td> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td>EyeScore</td><td></td><td> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td>EyeScore</td><td> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td></td></tr><tr><td>Features</td><td></td><td>Seen L1</td><td>New Ll</td><td></td><td>Seen L1</td><td>New Ll</td><td>Seen L1</td><td>New Ll</td><td></td><td>Seen Ll</td><td>New Ll</td></tr><tr><td>WPM</td><td> $- 1 . 3 7 ^ { * * }$ </td><td>-0.05</td><td>-0.43</td><td>-1.96***</td><td>-0.60</td><td>-0.97**</td><td>-1.57 -0.11</td><td>-0.01</td><td>-2.08*</td><td>-0.70</td><td>-0.52</td></tr><tr><td>Avg. Fix.</td><td> $- 0 . 7 8 ^ { * * * }$ </td><td>-0.03</td><td>-0.15</td><td>-1.18***</td><td>-0.41</td><td>-0.52*</td><td>-0.69</td><td>0.07 0.10</td><td>-1.07</td><td>-0.32</td><td>-0.26</td></tr><tr><td>S-Clusters</td><td> $- 0 . 4 8 ^ { * * }$ </td><td>-0.02</td><td>-0.11</td><td> $- 0 . 7 7 ^ { * * * }$ </td><td>-0.29</td><td> $- 0 . 3 9 ^ { * * }$ </td><td>-0.53 0.07</td><td>0.08</td><td>-0.87</td><td>-0.27</td><td>-0.23</td></tr><tr><td>WP-Coefs</td><td>-0.49***</td><td>-0.02</td><td>-0.11</td><td>-0.70***</td><td>-0.21*</td><td>-0.29**</td><td>-0.73* 0.02</td><td>0.05</td><td>-0.97**</td><td>-0.20</td><td>-0.17</td></tr><tr><td>Transitions</td><td> $- 0 . 0 5 ^ { * * * }$ </td><td>-0.07***</td><td>-0.09***</td><td>-0.09***</td><td>-0.11***</td><td>-0.12***</td><td>-0.14 -0.13</td><td>-0.12</td><td>-0.22*</td><td>-0.23</td><td>-0.22</td></tr><tr><td>Word Fix.</td><td> $- 0 . 5 0 ^ { * * * }$ </td><td>-0.02</td><td>-0.11</td><td> $- 0 . 7 0 ^ { * * * }$ </td><td>-0.20*</td><td> $- 0 . 2 9 ^ { * * }$ </td><td>-0.44 -0.05</td><td>-0.07</td><td>-0.64*</td><td>-0.25</td><td>-0.25</td></tr></table>

Table 16: L1-English Proximity Bias in EyeScore and EyeScore<sup>db</sup> in the Fixed Text regime.
<table><tr><td rowspan="2"></td><td colspan="3">LexTALE</td><td colspan="3">Michigan</td></tr><tr><td>EyeScore</td><td colspan="2"> $\mathrm { E y e S c o r e ^ { d b } }$ </td><td>EyeScore</td><td colspan="2"> $\mathrm { E y e S c o r e } ^ { \mathrm { d b } }$ </td></tr><tr><td>Features</td><td></td><td>Seen Ll</td><td>New Ll</td><td></td><td>Seen Ll</td><td>New Ll</td></tr><tr><td>WPM</td><td>0.01</td><td>-0.01</td><td>0.31</td><td>-1.51</td><td>-1.46</td><td>-1.22</td></tr><tr><td>Avg. Fix.</td><td>-0.30</td><td>-0.08</td><td>0.26</td><td>-1.09</td><td>-0.83</td><td>-0.51</td></tr><tr><td>S-Clusters</td><td>-0.20</td><td>-0.06</td><td>0.20</td><td>-0.91</td><td>-0.73</td><td>-0.50</td></tr><tr><td>WP-Coefs</td><td>-0.39</td><td>-0.02</td><td>0.08</td><td>-0.85*</td><td>-0.48</td><td>-0.37</td></tr></table>

Table 17: L1-English Proximity Bias in EyeScore and EyeScore<sup>db</sup> in the Any Text regime, for the OneStopL2 information seeking reading regime.

E.5 EyeScore<sup>db</sup> Correlations with Standard Proficiency Tests
<table><tr><td></td><td colspan="8">MECO L2</td><td colspan="8">OneStopL2</td></tr><tr><td></td><td colspan="3">LexTALE</td><td colspan="4"></td><td colspan="4">LexTALE</td><td colspan="4"></td></tr><tr><td></td><td colspan="2">Fixed</td><td colspan="2">Any</td><td colspan="2">Fixed</td><td colspan="2">Any</td><td colspan="2">Fixed</td><td colspan="2">Any I</td><td colspan="2">Fixed</td><td colspan="2">Any</td></tr><tr><td>Features</td><td>Seen L1</td><td>New L1</td><td>Seen L1</td><td>New L1</td><td>Seen L1</td><td>New L1</td><td>Seen L1</td><td>New L1</td><td>Seen L1</td><td>New L1</td><td>Seen L1</td><td>New L1</td><td>Seen L1</td><td>New L1</td><td>Seen L1</td><td>New L1</td></tr><tr><td>WPM</td><td>-0.015***</td><td>-0.002</td><td>-0.015***</td><td>-0.002</td><td>-0.002</td><td>+0.015**</td><td>-0.002</td><td>+0.015***</td><td>-0.030**</td><td>-0.030*</td><td>-0.017</td><td>-0.011</td><td>-0.021</td><td>-0.024</td><td>-0.005</td><td>-0.005</td></tr><tr><td>Avg. Fix.</td><td>-0.013***</td><td>-0.002</td><td>-0.013***</td><td>-0.003</td><td>-0.002</td><td>+0.011**</td><td>-0.001</td><td>+0.011**</td><td>-0.011</td><td>-0.013</td><td>-0.014</td><td>-0.016</td><td>-0.007</td><td>-0.007</td><td>-0.002</td><td>-0.011</td></tr><tr><td>S-Clusters</td><td>-0.012***</td><td>-0.002</td><td>-0.011**</td><td>-0.001</td><td>-0.002</td><td>+0.011**</td><td>-0.001</td><td>+0.011**</td><td>-0.012</td><td>-0.016</td><td>-0.014</td><td>-0.015</td><td>-0.008</td><td>-0.008</td><td>-0.002</td><td>-0.009</td></tr><tr><td>WP-Coefs</td><td>-0.015***</td><td>-0.006</td><td>-0.016***</td><td>-0.007</td><td>+0.001</td><td>+0.015***</td><td>+0.002</td><td>+0.015**</td><td>-0.010</td><td>-0.020</td><td>-0.020</td><td>-0.019</td><td>+0.008</td><td>-0.010</td><td>-0.002</td><td>-0.017</td></tr><tr><td>Transitions</td><td>+0.014***</td><td>+0.017**</td><td></td><td>-</td><td>+0.016***</td><td>+0.016**</td><td></td><td></td><td>+0.016</td><td>+0.014</td><td>一</td><td>-</td><td>+0.005</td><td>-0.003</td><td>一</td><td></td></tr><tr><td>Word Fix.</td><td>-0.016***</td><td>-0.006</td><td></td><td></td><td>+0.000</td><td>+0.015**</td><td></td><td></td><td>-0.017</td><td>-0.018</td><td></td><td></td><td>-0.012</td><td>-0.013</td><td></td><td></td></tr></table>

Table 18: The effect of EyeScore L1 debiasing on correlations with standard proficiency tests. Presented is $\Delta r _ { \mathrm { d b } } .$ , the Pearson r of EyeScore<sup>db</sup> with each standard English proficiency test minus the Pearson r of EyeScore with the same test. Seen L1 / New L1: did the data for computing the debiased score include participants from the same L1 as the test participant. Statistically significant differences using a bootstrap test are marked with ‘<sup>∗</sup>’ $p < 0 . 0 5 , ^ { * * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$

## F Test Reliability

## F.1 Comparison of EyeScore<sup>items</sup> and EyeScore

<table><tr><td>Features</td><td>MECO L2 Fixed Any</td><td>OneStopL2 Fixed</td><td>Any</td></tr><tr><td>WPM</td><td>1.00 1.00</td><td>0.99</td><td>1.00</td></tr><tr><td>Avg. Fix. S-Clusters</td><td>0.99 0.99 0.99 0.99</td><td>0.98 0.98</td><td>0.98 0.98</td></tr><tr><td>WP-Coefs</td><td>0.95 0.93</td><td>0.96</td><td>0.97</td></tr><tr><td>Transitions</td><td>0.99 -</td><td>0.98</td><td>一</td></tr><tr><td>Word Fix.</td><td>1.00 -</td><td>1.00</td><td>-</td></tr></table>

Table 19: Pearson r correlation between EyeScore<sup>items</sup> and EyeScore.

<table><tr><td></td><td colspan="4">MECO L2</td><td colspan="4">OneStopL2</td></tr><tr><td></td><td colspan="2">LexTALE</td><td colspan="2">Composite</td><td colspan="2">LexTALE</td><td colspan="2">Michigan</td></tr><tr><td>Features</td><td>Fixed</td><td>Any</td><td>Fixed</td><td>Any</td><td>Fixed</td><td>Any</td><td>Fixed</td><td>Any</td></tr><tr><td>WPM</td><td>+0.01</td><td>+0.01</td><td>+0.01</td><td>+0.01</td><td>0.00</td><td>0.00</td><td>-0.01</td><td>0.00</td></tr><tr><td>Avg. Fix.</td><td>+0.01</td><td>+0.01</td><td>+0.01</td><td>+0.01</td><td>+0.02</td><td>+0.03</td><td>+0.02</td><td>+0.03</td></tr><tr><td>S-Clusters</td><td>+0.02</td><td>+0.02</td><td>+0.02</td><td>+0.02</td><td>+0.03</td><td>+0.03</td><td>+0.03</td><td>+0.03</td></tr><tr><td>WP-Coefs</td><td>-0.01</td><td>-0.01</td><td>-0.02</td><td>-0.02</td><td>+0.04</td><td>+0.04</td><td>0.00</td><td>+0.01</td></tr><tr><td>Transitions</td><td>0.00</td><td>=</td><td>+0.02</td><td>-</td><td>+0.02</td><td>=</td><td>+0.01</td><td>=</td></tr><tr><td>Word Fix.</td><td>0.00</td><td></td><td>0.00</td><td>=</td><td>0.00</td><td></td><td>0.00</td><td></td></tr></table>

Table 20: EyeScore<sup>items</sup> Pearson r correlation with standard proficiency tests. Presented is $\Delta r _ { \mathrm { i t e m s } } ,$ the Pearson r of EyeScore<sup>items</sup> with each standard English proficiency test minus the Pearson r of EyeScore with the same test.

## F.2 Internal Consistency Evaluation Procedure

As described in Section 8.1, the item-level analysis requires a separate EyeScore for each passage read by each participant. We therefore apply the EyeScore procedure to passage-level feature vectors, such that each participant is represented by one vector for each recorded passage. The way these vectors are used to compute EyeScore differs between the Fixed Text and Any Text regimes.

In the Fixed Text regime, when calculating each item’s EyeScore, we are only interested in feature vectors extracted for the same item across participants. This resembles the original setup, where each participant has only one vector representing them, and we repeat this process for each item.

In the Any Text regime, when calculating each item’s EyeScore, each training participant contributes multiple passage-level feature vectors. We perform the normalization over all such vectors, effectively normalizing them relative to all passagelevel vectors of L2 participants in train, and construct the L1 prototype analogously as the average L1 participant over passages.

## F.3 Split-Half Reliability Evaluation Procedure

In MECO L2, as mentioned in Section 5.5, we divide the 12 passages into two parts with 6 passages each. To preserve this balance in the split-half reliability analysis, we randomly split each of the two parts separately. Since not all participants have eye movement data for all passages, a split may result in a participant appearing in only one of the halves. Thus, for each split, we include only participants with eye movement data for at least one passage in each half. On average, this filtering removes 5 of the 1,106 participants per split.

As mentioned in Section 5.5, in OneStopL2 and OneStop, there are 6 participant subgroups, each group reading its own set of passages. Thus, for each split of the split-half reliability analysis, we randomly split each subgroup’s set of passages separately and combine the corresponding halves across subgroups.

## F.4 Reliability in the Fixed Text Regime and in Information Seeking

<table><tr><td></td><td colspan="3">MECO L2</td><td colspan="3">OneStopL2</td></tr><tr><td>Features</td><td>Cronbach&#x27;s α</td><td>SEM</td><td>Split-Half r</td><td>Cronbach&#x27;s α</td><td>SEM</td><td>Split-Half r</td></tr><tr><td>WPM</td><td>0.98</td><td>0.13</td><td>0.93</td><td>0.99</td><td>0.09</td><td>0.98</td></tr><tr><td>Avg. Fix.</td><td>0.98</td><td>0.08</td><td>0.92</td><td>0.99</td><td>0.04</td><td>0.99</td></tr><tr><td>S-Clusters</td><td>0.98</td><td>0.04</td><td>0.94</td><td>1.00</td><td>0.02</td><td>0.99</td></tr><tr><td>WP-Coefs</td><td>0.92</td><td>0.06</td><td>0.79</td><td>0.98</td><td>0.03</td><td>0.94</td></tr><tr><td>Transitions</td><td>0.97</td><td>0.01</td><td>0.91</td><td>0.99</td><td>0.01</td><td>0.99</td></tr><tr><td>Word Fix.</td><td>0.98</td><td>0.04</td><td>0.95</td><td>1.00</td><td>0.02</td><td>0.99</td></tr></table>

Table 21: Reliability of EyeScore in the Fixed Text regime.

<table><tr><td></td><td colspan="3">Fixed Text</td><td colspan="3">Any Text</td></tr><tr><td>Features</td><td>Cronbach&#x27;s α</td><td>SEM</td><td>Split-Half r</td><td>Cronbach&#x27;s α</td><td>SEM</td><td>Split-Half r</td></tr><tr><td>WPM</td><td>0.98</td><td>0.13</td><td>0.97</td><td>0.98</td><td>0.09</td><td>0.97</td></tr><tr><td>Avg. Fix.</td><td>0.99</td><td>0.05</td><td>0.98</td><td>0.99</td><td>0.05</td><td>0.98</td></tr><tr><td>S-Clusters</td><td>0.99</td><td>0.03</td><td>0.98</td><td>0.99</td><td>0.03</td><td>0.98</td></tr><tr><td>WP-Coefs</td><td>0.98</td><td>0.04</td><td>0.91</td><td>0.99</td><td>0.03</td><td>0.94</td></tr><tr><td>Transitions</td><td>0.99</td><td>0.01</td><td>0.98</td><td></td><td></td><td>=</td></tr><tr><td>Word Fix.</td><td>0.99</td><td>0.03</td><td>0.99</td><td></td><td></td><td></td></tr></table>

Table 22: Reliability of EyeScore in the OneStopL2 information seeking reading regime.

## F.5 Reliability of Standard English Proficiency Tests

Michigan Placement Test (OneStopL2) Split-half Pearson r was calculated over 20 splits, with a balanced number of items from each test section across the two halves. The resulting Cronbach’s α of the listening part is 0.74, for the grammar section it is 0.84, for the vocabulary section it is 0.82, and for the reading section it is 0.78. The stratified Cronbach’s α (Cronbach et al., 1965) overall is 0.91, and the average Pearson r for split-half reliability is 0.92. In the OneStopL2 information seeking regime, the reliability scores for the section are 0.75 for listening, 0.85 for grammar, 0.83 for vocabulary, and 0.78 for reading. The overall stratified Cronbach’s α of 0.95 and the split-half Pearson r is 0.96.

IELTS The IELTS test comprises four sections: listening, reading (in one of two variants, ‘academic’ or ‘general training’), speaking, and writing. IELTS (2026) report average Cronbach’s α values of 0.90 for the listening section, 0.92 for academic reading, 0.91 for general training reading, as well as inter-rater reliability estimates of 0.90 for speaking and 0.92 for writing.

TOEFL-iBT Manna et al. (2025) report Cronbach’s α of 0.94 for the speaking section and 0.87 for the writing section (stratified). The reading and listening parts of the test are evaluated using a reliability estimate based on item response theory (IRT) (Kolen et al., 1996), with a reliability score of 0.86 for the reading section and 0.88 for the listening section. The reported overall reliability score is 0.90.