# How Do LLMs Understand Legal Philosophy? Diagnosing Legal-Philosophical Reasoning in LLMs

Shuyuan Zheng<sup>1</sup> Yun-seo Lee<sup>3</sup> Xing Gao<sup>4</sup> Jialu Wu<sup>4</sup> Jiabao Hu<sup>5</sup> Katrina Qiyao Wang<sup>6</sup> Xinbo Lin<sup>7</sup> Qianying Liu<sup>8</sup> Chuan Xiao<sup>1</sup> Tao Huang<sup>2</sup>\*

<sup>1</sup>The University of Osaka <sup>2</sup>City University of Hong Kong <sup>3</sup>UC Davis <sup>4</sup>UC Berkeley

<sup>5</sup>Peking University <sup>6</sup>University of Wisconsin–Madison <sup>7</sup>Shanghai Jiao Tong University <sup>8</sup>NII LLMC zheng@ist.osaka-u.ac.jp taohuang@cityu.edu.hk

## Abstract

Legal reasoning is a fundamental capability for large language models (LLMs) to complete legal AI tasks such as legal judgment prediction. In legal practice, this reasoning is shaped by legal philosophy: a judge’s view of the nature and sources of law guides how legal materials are interpreted and applied, so judges working from the same statutes and precedents can reach different judgments. Existing research has largely overlooked whether LLMs can understand legal philosophy and apply it in legal reasoning, an ability we call legal-philosophical reasoning. To fill this gap, we introduce LePhilBench, a benchmark for evaluating the legalphilosophical reasoning of LLMs that comprises two benchmarking tasks and a dataset. The two tasks evaluate how well LLMs understand legal philosophy and how well they apply it in legal reasoning, respectively. The dataset is manually and meticulously constructed. Its test set contains 220 U.S. Supreme Court opinions. A law professor designed fourteen annotation questions on legal philosophy, and three law-trained annotators with formal training in jurisprudence answered all fourteen questions for every opinion, yielding 9,240 legal philosophical labels. LePhilBench takes the first step toward exploring legal-philosophical reasoning in LLMs.<sup>1</sup>

## 1 Introduction

Legal reasoning, the ability to apply legal rules and principles to the facts of a particular case to reach a legal decision, is a central capability target for large language models (LLMs) in the legal domain. It underlies tasks such as legal judgment prediction (Fei et al., 2024), legal fact prediction (Liu et al., 2025), and legal opinion generation (Huang et al., 2023). In each of these tasks, a model must apply abstract legal norms to concrete case facts to reach a legal decision. Whether LLMs can perform this reasoning reliably therefore determines how far their legal decisions can be trusted.

In real-world legal practice, however, legal reasoning is not a mechanical application of rules to facts. It is shaped by legal philosophy. How legal professionals understand the nature of law and the sources of its legitimacy influences how they interpret legal materials and apply them to a case. As a result, judges who work from the same statutes and precedents can reach different judgments because they hold different jurisprudential views, for example on the relative weight oftext, purpose, precedent, and moral principle. Existing evaluations and benchmarks of LLMs’ legal reasoning have largely overlooked this influence. They mainly assess task correctness and law-to-fact reasoning (Shui et al., 2023; Jiang and Yang, 2023). Some benchmarks cover related abilities, such as interpretive-tool recognition in LegalBench (Guha et al., 2023) and value-based prosecutorial discretion in PDP-Bench (Lu et al., 2026). Even these do not examine whether a model’s legal reasoning reflects a jurisprudential position. An unresolved research question therefore remains: can LLMs engage in legal-philosophical reasoning as legal professionals do, i.e., understand legal philosophy and use it to personalize legal reasoning?

To fill this gap, this paper explores how LLMs can perform legal-philosophical reasoning. We construct LePhilBench, a benchmark framework consisting of two tasks (see Figure 1): jurisprudential profiling (JP) and jurisprudence-aware judgment prediction (JAJP). JP asks an LLM to read a human judge’s legal opinions and annotate the jurisprudential tendencies they may stand along fourteen dimensions, thereby constructing the judge’s jurisprudential profile. JAJP asks an LLM to predict, based on a target judge’s jurisprudential profile, how that judge would resolve the disputed legal question in a given case. This framework is inspired by a common practice in litigation strategy. Before arguing a case, lawyers often read the past opinions of the judge to form an impression of their legal-philosophical tendencies, and then draw on that impression to anticipate how the judge is likely to decide the key legal questions in the case. LePhilBench makes this otherwise tacit impression explicit as a structured jurisprudential profile: JP mirrors the step of forming it, and JAJP mirrors the step of drawing on it.

![](images/a5dd415f3479658ed6778bf462bffca830592c4f925f675dbef44aa9ba501651.jpg)  
Figure 1: Overview of jurisprudential profiling (JP) and jurisprudence-aware judgment prediction (JAJP): inferring a jurisprudential profile from a judge’s past opinion(s) and using it to predict the judge’s judgment in a target case.

To drive these two tasks, we construct the LePhilBench dataset. We collect the opinions of the Court and the dissenting opinions from 489 U.S. Supreme Court cases decided over the past two decades, and use them to construct the jurisprudential profiles of eleven target justices. To make these profiles explicit, a law professor specializing in U.S. constitutional law helped us design fourteen dimensions of jurisprudential tendency. Each dimension measures where a judge stands between a pair of competing jurisprudential views, such as Natural Law vs. Legal Positivism. We then select 110 cases as the test set. Three law-trained annotators with formal training in jurisprudence annotate the opinion of the Court and the dissenting opinion in each test case, 220 opinions in total. For each opinion, each annotator provides individual labels for the fourteen tendencies and for the opinion’s judgement of the case’s key legal question, yielding 110 × 2 × 14 × 3 = 9,240 jurisprudential labels.

Our evaluation on LePhilBench shows that frontier LLMs have considerable potential for legalphilosophical reasoning. First, in JP, the agreement between frontier LLMs and law-trained human annotators is close to the agreement among the human annotators themselves. Frontier LLMs therefore recover the jurisprudential tendencies expressed in judicial opinions about as consistently as law-trained human annotators do. This suggests that they can understand legal philosophy as law-trained humans do. Second, in JAJP, frontier LLMs predict how a target judge would resolve a case’s key legal question with fairly high accuracy, even though the judge’s name is withheld and only their jurisprudential profile is provided. Because the judge’s identity is hidden, a correct prediction can be attributed to the reasoning about the jurisprudential tendencies recorded in the profile. This suggests that frontier LLMs have the strong potential to apply legal philosophy as legal professionals do.

In summary, this paper makes three contributions. (1) We propose LePhilBench, a benchmark framework that evaluates the legal-philosophical reasoning of LLMs through two complementary tasks, JP and JAJP. (2) We construct the LePhilBench dataset from 489 U.S. Supreme Court cases, with 9,240 annotations carefully produced by law-trained annotators, whose case answers are verified by a law professor. (3) We conduct extensive evaluations of frontier LLMs on LePhilBench, revealing both their capabilities and their limitations in legal-philosophical reasoning. We hope LePhilBench can serve as a starting point for building legal AI systems that reason not only with legal rules but also with legal philosophy.

## 2 The LePhilBench Dataset

## 2.1 Dataset Overview

Composition and coverage. LePhilBench pairs competing U.S. Supreme Court opinions on shared case materials. It contains 489 cases, each with one Court Opinion and one Dissent, for 978 opinions. All cases come from the Roberts Court during 2005- 12-06 to 2025-06-27. These opinions were written by 15 justices. The human-annotated test subset contains 110 cases and their 220 paired opinions, written by 14 justices; 13 of these justices each authored at least five Court Opinions and five Dissents in the test subset.

<table><tr><td>Information</td><td>Description</td></tr><tr><td>Case record Date (Oyez) Case facts (Oyez)</td><td>Date the Court decided the case Summary of the facts of the case</td></tr><tr><td>Key legal question (Oyez &amp; us)</td><td>Binary question at the center of the dispute</td></tr><tr><td>Opinion record Type (Justia)</td><td>Court Opinion or Dissent</td></tr><tr><td>Author (Justia)</td><td>Justice who wrote the opinion</td></tr><tr><td>Main text (Justia)</td><td>Full text of the opinion</td></tr><tr><td>(us)</td><td>Exclusive joiners Justices who fully joined the opinion and neither wrote nor joined any other merits</td></tr><tr><td>Case answer (us)</td><td>opinion in the case</td></tr><tr><td></td><td>Gold answer to the key legal question (A = Yes, B = No)</td></tr><tr><td>Tendency labels for Q1–Q14 (us)</td><td>Three individual annotators&#x27; labels for each jurisprudential tendency</td></tr></table>

Table 1: Key fields in LePhilBench. The source of each field is given in parentheses.
<table><tr><td>Statistic</td><td>Court</td><td>Dissent</td><td>Overall</td></tr><tr><td>Full set</td><td></td><td></td><td></td></tr><tr><td>Cases</td><td></td><td></td><td>489</td></tr><tr><td>Opinions</td><td>489</td><td>489</td><td>978</td></tr><tr><td>Authors</td><td>15</td><td>15</td><td>15</td></tr><tr><td>Test set</td><td></td><td></td><td></td></tr><tr><td>Cases</td><td></td><td></td><td>110</td></tr><tr><td>Opinions</td><td>110</td><td>110</td><td>220</td></tr><tr><td>Authors</td><td>14</td><td>14</td><td>14</td></tr><tr><td>Tendency labels (Q1-Q14)</td><td>4,620</td><td>4,620</td><td>9,240</td></tr><tr><td>Pairwise agreement</td><td>70.24%</td><td>67.92%</td><td>69.08%</td></tr><tr><td>Fleiss&#x27;κ (three annotators)</td><td>0.492</td><td>0.453</td><td>0.474</td></tr></table>

Table 2: Statistics of LePhilBench.

Key fields. Table 1 lists key fields each record carries. A case record supplies the materials its two opinions share, including the case facts and the key legal question. An opinion record holds the opinion’s own text, authorship, and exclusive joiners; for test opinions, it adds the case answer and three individual annotators’ labels for each of the fourteen jurisprudential tendencies (Q1–Q14; see Appendix A.1 for details), which together describe the opinion-level jurisprudential profile.

## 2.2 Dataset Construction

Case selection. We collect U.S. Supreme Court cases and opinions from the Justia (Justia, n.d.) and Oyez (Oyez, n.d.b) websites. A case enters LePhilBench when it (1) was decided by the Roberts Court and appears in U.S. Reports volumes 546 through 606, the latest volume available at data collection, and (2) contains exactly one Court Opinion and exactly one Dissent. Criterion (2) divides the participating justices into two camps, so each case pairs two opinions that reach opposing case answers through conflicting jurisprudential reasoning.

For the test subset, we take the 11 justices who authored the most opinions in these volumes and, for each justice, select the five most recent cases in which the justice wrote the Court Opinion and the five most recent cases in which the justice wrote the Dissent, yielding 110 cases. This selection (1) balances the numbers of Court Opinions and Dissents authored by each selectedjustice and (2) leaves more of each justice’s past opinions available for building the judge-level jurisprudential profile.

Key legal question construction and case answer annotation. Each case is framed by one binary key legal question on which its two opinions give opposing case answers. We take the question Oyez poses for each case; it targets the focal dispute and need not capture every issue in the case. For test cases, we further review each question so that it reads as a Yes/No question that the Court Opinion and the Dissent answer in opposite ways. Three law-trained annotators with formal training in jurisprudence then record each opinion’s case answer, and a law professor, as a senior expert, resolves their disagreements to fix the gold answers, stored as opposing labels for the two opinions. Appendix A.2 gives the profiles of the three annotators and the senior expert.

Jurisprudential tendency annotation. We design a questionnaire for each test opinion and ask the three annotators to annotate the jurisprudential tendencies the opinion reflects along fourteen dimensions. Each questionnaire presents the case facts and the key legal question, followed by fourteen annotation questions (Q1 through Q14) whose dimensions the senior expert helped design (Appendix A.1). Each question offers three options: A and B describe the two contrasting tendencies on that dimension, and C stands for “Unknown/Hard to Say.” Because jurisprudential annotation is partly subjective, we retain all three annotators’ labels rather than a single adjudicated label, so that aggregated labels can be derived flexibly through annotation aggregation such as majority vote. Table 2 shows that the three annotators reach a Fleiss’ κ of 0.474, indicating moderate agreement for interpretive annotation (Landis and Koch, 1977).

Joiner annotation. An opinion reflects the jurisprudential tendencies not only of its author but also of the justices who join it. A justice may, however, join one opinion while also writing or joining another opinion in the same case, which is common with concurrences, so the joined opinion alone does not capture that justice’s position. We therefore annotate exclusivejoiners: justices who fully joined the opinion and neither wrote nor joined any other merits opinion in the case. For these justices, the target opinion gives a reasonably complete account of their jurisprudential tendencies in the case. Because each opinion contains a passage stating which justices join it, we extract its joiners and full joiners (justices who join it in full rather than in part) with regular expressions, and we label as exclusive joiners the full joiners who appear in neither the author nor the joiner field of any other opinion in the case.

## 3 The LePhilBench Tasks

In this section, we introduce the benchmarking protocols for the two LePhilBench tasks.

## 3.1 Jurisprudential Profiling (JP)

JP tests whether a model can label the jurisprudential tendencies expressed in a judicial opinion as human annotators understand them. Its input is the case facts and the main text of the opinion; its output is an answer of A, B, or C to each of the fourteen annotation questions, one per legal-philosophical dimension, which together form an opinion-level jurisprudential profile. Depending on the annotation mode and the human-label aggregation method, this task yields different benchmarking protocols, as described below.

Annotation modes. We evaluate LLMs on the JP task in two modes that share the same response space. (1) Paired Annotation gives the model the Court Opinion and the Dissent of a case together in one input and profiles both. (2) Separate Annotation gives the model each opinion in its own input and profiles it alone. Appendix B.2 gives the details of both modes. We use Paired Annotation by default because it yields higher LLM-human agreement.

Human-label aggregation. Because LePhilBench retains the original labels of all three annotators, we evaluate LLMs on JP by LLMhuman pairwise agreement by default: the exact A/B/C agreement between a model and each individual annotator, averaged across the three annotators. We also adopt two common label aggregation methods: (1) Majority vote keeps the label chosen by the unique plurality of annotators, with C taking part in the vote, and maps a three-way A/B/C tie to C. (2) Unanimous agreement keeps the shared label only when all three annotators agree and maps every other unit to C.

## 3.2 Jurisprudence-Aware Judgment Prediction (JAJP)

JAJP tests whether LLMs can accurately predict a target judge’s legal judgment from that judge’s jurisprudential profile. Its input is the case facts, the key legal question, and the target judge’s jurisprudential profile; its output is the target judge’s case answer, a Yes or No answer to the key legal question. The input excludes both the judge’s name and the opinion type, that is, whether the judge wrote or joined the Court Opinion or the Dissent. An LLM therefore cannot predict the case answer from what it already knows about a specific judge or from the side the judge took in the case, and must instead reason from the jurisprudential profile.

History-based profile construction. JAJP builds the target judge’s jurisprudential profile from the judge’s past opinions, since a judgment prediction in real-world use can draw only on earlier opinions. We encode an opinion-level profile as a vector over Q1–Q14 with A= −1, C= 0, and B= +1, and aggregate the LLM-annotated profiles of the target judge’s past opinions into a single judge-level profile. The past opinions are those the target judge authored or joined as an exclusive joiner; by default, we consider only authored opinions. We consider four profile aggregation methods (see Appendix B.3 for details): (1) Mean averages all nonzero tendencies across opinions. (2) Latest takes the most recent nonzero tendency across opinions. (3) Exponential Decay takes a weighted mean of nonzero tendencies, with the weights of past opinions decaying exponentially with their age. (4) Random Walk models each of the judge’s jurisprudential tendencies as a Gaussian random walk and estimates the judge-level profile with Kalman filtering. By default, we use Mean.

Evaluation metric. We evaluate JAJP by JAJP accuracy gain (percentage point), which is the accuracy improvement from supplying the jurisprudential profile over the No Profile control, in which the model receives neither a jurisprudential profile nor the target judge’s identity. Because No Profile amounts to jurisprudence-unaware judgment prediction, this improvement measures how much of the accuracy comes from jurisprudence awareness. Note that each case contributes one Court Opinion and one Dissent with opposite case answers, and the No Profile input is identical for both, so the accuracy under No Profile is therefore 50% by construction.

![](images/77680bc093e583546675d0ddd4a2e858fb8415f42fb60d5c995ba6b5da87d5c1.jpg)  
Figure 2: LLM-human vs. human-human pairwise agreement (%) across jurisprudential tendencies.

![](images/80f45e95d7436aaf647d10e9394ab8f3f85a2e3bc6b3f5394d7edb48f53d0e6d.jpg)  
Figure 3: LLM-human pairwise disagreement attribution. Opposite: the model and the annotator choose opposite directions (A vs. B). Uncertain: the annotator chooses A or B, but the model chooses C. Overconfident: the annotator chooses C, but the model chooses A or B.

## 4 Evaluation

This section evaluates LLMs on the two LePhilBench tasks, JP and JAJP. We run every experiment with three seeds and report the average results averaged over the seeds.

## 4.1 Evaluation on the JP Task

Experiment 1: Agreement acrossjurisprudential tendencies and JP annotation modes. We ask whether the jurisprudential tendencies that are hard for human annotators to agree on are also hard for LLMs. To answer this, we measure LLM-human pairwise agreement across Q1–Q14 under Paired Annotation and Separate Annotation in Figure 2.

![](images/2900eca2b83cf397079362012cbbd2bdae04f247a1690a3b20d9414a2bf5e036.jpg)  
Figure 4: LLM annotation distribution in JP vs. human consensus level. On 3-0, 2-1, and 1-1-1 questions, the three annotators’ labels are unanimous, split two to one, or all different, respectively.

LLM agreement with aggregated human labels is reported in Figures 7 and 8.

Finding 1a: Jurisprudential tendencies that are hard for humans are also hard for LLMs. In Figure 2, LLM-human pairwise agreement varies across jurisprudential tendencies and follows human-human agreement at both ends. Q4, Q5, Q7, Q9, and Q14 are the five tendencies with the highest human-human agreement (76.4%–81.2%), and they are also the five with the highest LLM-human agreement (75.3%–78.9%) averaged over the models. At the other end, Q3 and Q2 have the lowest human-human agreement (50.8% and 53.3%), and they rank among the three tendencies with the lowest LLM-human agreement (48.2% and 53.7%). This suggests that some tendencies are inherently more ambiguous to annotate, and that LLMs perceive this ambiguity much as humans do.

<table><tr><td>Model</td><td></td><td>Support Court Support Dissent</td></tr><tr><td>GPT-5.6-Sol</td><td> $\overline { { 9 5 . 1 5 \pm 0 . 3 0 } }$ </td><td> $\overline { { 4 . 8 5 \pm 0 . 3 0 } }$ </td></tr><tr><td>GPT-5.6-Terra</td><td> $9 2 . 7 3 \pm 1 . 0 5$ </td><td> $7 . 2 7 \pm 1 . 0 5$ </td></tr><tr><td>GPT-5.6-Luna</td><td> $9 1 . 5 2 \pm 0 . 8 0$ </td><td> $8 . 4 8 \pm 0 . 8 0$ </td></tr><tr><td>Claude Opus 5</td><td> $9 6 . 6 7 \pm 0 . 3 0$ </td><td> $3 . 3 3 \pm 0 . 3 0$ </td></tr><tr><td>Claude Sonnet 5</td><td> $9 0 . 9 1 \pm 0 . 0 0$ </td><td> $9 . 0 9 \pm 0 . 0 0$ </td></tr><tr><td>Qwen3.7-Max</td><td> $9 2 . 4 2 \pm 0 . 3 0$ </td><td> $7 . 5 8 \pm 0 . 3 0$ </td></tr><tr><td>Qwen3.7-Plus</td><td> $8 5 . 1 5 \pm 0 . 8 0$ </td><td> $1 4 . 8 5 \pm 0 . 8 0$ </td></tr><tr><td>Qwen3.7-Flash</td><td> $7 6 . 6 7 \pm 1 . 6 0$ </td><td> $2 3 . 3 3 \pm 1 . 6 0$ </td></tr><tr><td>Gemini 3.7 Flash</td><td> $9 4 . 8 5 \pm 0 . 3 0$ </td><td> $5 . 1 5 \pm 0 . 3 0$ </td></tr><tr><td>Muse Spark 1.3</td><td> $9 2 . 7 3 \pm 0 . 9 1$ </td><td></td></tr><tr><td></td><td></td><td> $7 . 2 7 \pm 0 . 9 1$ </td></tr><tr><td>Overall</td><td> $\overline { { 9 0 . 8 8 \pm 0 . 3 9 } }$ </td><td> $\overline { { 9 . 1 2 \pm 0 . 3 9 } }$ </td></tr></table>

<table><tr><td rowspan="2">Model</td><td colspan="3">Accuracy (%)</td><td colspan="3">Accuracy gain over No Profile (pp)</td></tr><tr><td>Court</td><td>Dissent</td><td>Overall</td><td>Court</td><td>Dissent</td><td>Overall</td></tr><tr><td>GPT-5.6-Sol</td><td>88.18 ± 0.52</td><td> $\overline { { 3 5 . 7 6 \pm 1 . 5 2 } }$ </td><td>61.97 ± 0.55</td><td> $- 6 . 9 7 \pm 0 . 8 0$ </td><td>30.91 ± 1.82</td><td> $1 1 . 9 7 \pm 0 . 5 5$ </td></tr><tr><td>GPT-5.6-Terra</td><td> $8 6 . 3 6 \pm 2 . 7 8$ </td><td> $3 7 . 2 7 \pm 1 . 8 9$ </td><td> $6 1 . 8 2 \pm 1 . 9 8$ </td><td> $- 6 . 3 6 \pm 3 . 7 8$ </td><td> $3 0 . 0 0 \pm 1 . 3 9$ </td><td> $1 1 . 8 2 \pm 1 . 9 8$ </td></tr><tr><td>GPT-5.6-Luna</td><td> $8 2 . 7 3 \pm 3 . 2 8$ </td><td> $2 0 . 3 0 \pm 1 . 0 9$ </td><td> $5 1 . 5 2 \pm 1 . 9 7$ </td><td> $- 8 . 7 9 \pm 3 . 4 1$ </td><td> $1 1 . 8 2 \pm 1 . 8 2$ </td><td> $1 . 5 2 \pm 1 . 9 7$ </td></tr><tr><td>Claude Opus 5</td><td> $8 1 . 5 2 \pm 1 . 3 2$ </td><td> $4 8 . 7 9 \pm 2 . 3 7$ </td><td> $6 5 . 1 5 \pm 0 . 9 9$ </td><td> $- 1 5 . 1 5 \pm 1 . 3 2$ </td><td> $4 5 . 4 5 \pm 2 . 1 0$ </td><td> $1 5 . 1 5 \pm 0 . 9 9$ </td></tr><tr><td>Claude Sonnet 5</td><td> $8 6 . 3 6 \pm 0 . 9 1$ </td><td> $2 0 . 6 1 \pm 0 . 6 1$ </td><td> $5 3 . 4 8 \pm 0 . 7 6$ </td><td> $- 4 . 5 5 \pm 0 . 9 1$ </td><td> $1 1 . 5 2 \pm 0 . 6 1$ </td><td> $3 . 4 8 \pm 0 . 7 6$ </td></tr><tr><td> $\mathbf { Q } \mathrm { w e n } 3 . 7 – \mathbf { M } \mathrm { a x }$ </td><td> $8 4 . 5 5 \pm 1 . 5 7$ </td><td> $4 9 . 3 9 \pm 1 . 0 9$ </td><td> $6 6 . 9 7 \pm 1 . 2 4$ </td><td> $- 7 . 8 8 \pm 1 . 3 2$ </td><td> $4 1 . 8 2 \pm 1 . 3 9$ </td><td> $1 6 . 9 7 \pm 1 . 2 4$ </td></tr><tr><td> $\mathrm { Q w e n } 3 . 7 – \mathrm { P l u s }$ </td><td> $7 4 . 2 4 \pm 1 . 8 4$ </td><td> $4 6 . 0 6 \pm 0 . 3 0$ </td><td> $6 0 . 1 5 \pm 1 . 0 6$ </td><td> $- 1 0 . 9 1 \pm 2 . 4 1$ </td><td> $3 1 . 2 1 \pm 0 . 8 0$ </td><td> $1 0 . 1 5 \pm 1 . 0 6$ </td></tr><tr><td>Qwen3.7-Flash</td><td> $7 6 . 3 6 \pm 2 . 6 2$ </td><td> $3 6 . 9 7 \pm 0 . 8 0$ </td><td> $5 6 . 6 7 \pm 1 . 4 9$ </td><td> $- 0 . 3 0 \pm 3 . 8 7$ </td><td> $1 3 . 6 4 \pm 0 . 9 1$ </td><td> $6 . 6 7 \pm 1 . 4 9$ </td></tr><tr><td>Gemini 3.7 Flash</td><td> $8 9 . 0 9 \pm 0 . 5 2 $ </td><td> $5 3 . 3 3 \pm 0 . 6 1$ </td><td> $7 1 . 2 1 \pm 0 . 4 0$ </td><td> $- 5 . 7 6 \pm 0 . 8 0$ </td><td> $4 8 . 1 8 \pm 0 . 5 2$ </td><td> $2 1 . 2 1 \pm 0 . 4 0$ </td></tr><tr><td>Muse Spark 1.3</td><td> $8 0 . 0 0 \pm 0 . 9 1$ </td><td> $5 6 . 9 7 \pm 0 . 6 1$ </td><td> $6 8 . 4 8 \pm 0 . 6 6$ </td><td> $- 1 2 . 7 3 \pm 0 . 9 1$ </td><td> $4 9 . 7 0 \pm 0 . 8 0$ </td><td> $1 8 . 4 8 \pm 0 . 6 6$ </td></tr><tr><td>Overall</td><td> $\overline { { 8 2 . 9 4 \pm 0 . 4 0 } }$ </td><td> $\overline { { 4 0 . 5 5 \pm 0 . 3 7 } }$ </td><td> $6 1 . 7 4 \pm 0 . 3 1$ </td><td> $- 7 . 9 4 \pm 0 . 7 2$ </td><td> $\overline { { 3 1 . 4 2 \pm 0 . 6 7 } }$ </td><td> $\overline { { 1 1 . 7 4 \pm 0 . 3 1 } }$ </td></tr></table>

(a) LLM support inclination (%) without a profile. (b) JAJP accuracy with the jurisprudential profile and gain over No Profile.

Table 3: No Profile support inclination, and JAJP accuracy with the jurisprudential profile and gain over No Profile.
<table><tr><td rowspan=1 colspan=1>GPT-5.6-Sol</td><td rowspan=1 colspan=1>12.0 ± 0.5</td><td rowspan=1 colspan=1>10.2 ± 1.2</td><td rowspan=1 colspan=1>8.5 ± 1.2</td><td rowspan=1 colspan=1>10.8 ± 0.4</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6-Terra</td><td rowspan=1 colspan=1>11.8 ± 2.0</td><td rowspan=1 colspan=1>10.3 ± 0.9</td><td rowspan=1 colspan=1>8.6 ± 2.0</td><td rowspan=1 colspan=1>10.3 ± 1.5</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6-Luna</td><td rowspan=1 colspan=1>1.5 ± 2.0</td><td rowspan=1 colspan=1>0.2 ± 1.0</td><td rowspan=1 colspan=1>1.4 ± 0.5</td><td rowspan=1 colspan=1>1.7 ± 0.8</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 5</td><td rowspan=1 colspan=1>15.2 ± 1.0</td><td rowspan=1 colspan=1>10.5 ± 1.0</td><td rowspan=1 colspan=1>19.2 ± 0.5</td><td rowspan=1 colspan=1>14.4 ± 0.5</td></tr><tr><td rowspan=1 colspan=1>Claude Sonnet 5</td><td rowspan=1 colspan=1>3.5 ± 0.8</td><td rowspan=1 colspan=1>4.8 ± 0.6</td><td rowspan=1 colspan=1>2.9 ± 0.4</td><td rowspan=1 colspan=1>5.9 ± 0.5</td></tr><tr><td rowspan=1 colspan=1>Qwen3.7-Max</td><td rowspan=1 colspan=1>17.0 ± 1.2</td><td rowspan=1 colspan=1>13.6 ± 1.1</td><td rowspan=1 colspan=1>16.5 ± 1.1</td><td rowspan=1 colspan=1>18.5 ± 0.5</td></tr><tr><td rowspan=1 colspan=1>Qwen3.7-Plus</td><td rowspan=1 colspan=1>10.2 ± 1.1</td><td rowspan=1 colspan=1>7.1 ± 1.4</td><td rowspan=1 colspan=1>10.8 ± 0.9</td><td rowspan=1 colspan=1>12.7 ± 1.2</td></tr><tr><td rowspan=1 colspan=1>Qwen3.7-Flash</td><td rowspan=1 colspan=1>6.7 ± 1.5</td><td rowspan=1 colspan=1>3.0 ± 1.2</td><td rowspan=1 colspan=1>3.9 ± 1.3</td><td rowspan=1 colspan=1>5.5 ± 0.5</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.7 Flash</td><td rowspan=1 colspan=1>21.2 ± 0.4</td><td rowspan=1 colspan=1>15.2 ± 1.1</td><td rowspan=1 colspan=1>18.5 ± 0.3</td><td rowspan=1 colspan=1>20.5 ± 0.3</td></tr><tr><td rowspan=1 colspan=1>Muse Spark 1.3</td><td rowspan=1 colspan=1>18.5 ± 0.7</td><td rowspan=1 colspan=1>12.9 ± 0.3</td><td rowspan=1 colspan=1>17.9 ± 1.1</td><td rowspan=1 colspan=1>18.5 ± 1.1</td></tr><tr><td rowspan=1 colspan=1>Overall</td><td rowspan=1 colspan=1>11.7 ± 0.3</td><td rowspan=1 colspan=1>8.8 ± 0.3</td><td rowspan=1 colspan=1>10.8 ± 0.1</td><td rowspan=1 colspan=1>11.9 ± 0.2</td></tr><tr><td rowspan=1 colspan=5>Random ExponentialMean    LatestWalk    Decay</td></tr></table>

(b) Exclusive-joiner profiles included
<table><tr><td rowspan=1 colspan=1>8.5 ± 0.4</td><td rowspan=1 colspan=1>7.0 ± 1.3</td><td rowspan=1 colspan=1>9.4 ± 0.7</td><td rowspan=1 colspan=1>11.1 ± 1.1</td></tr><tr><td rowspan=1 colspan=1>7.7 ± 0.7</td><td rowspan=1 colspan=1>7.7 ± 0.7</td><td rowspan=1 colspan=1>8.8 ± 0.2</td><td rowspan=1 colspan=1>8.6 ± 1.4</td></tr><tr><td rowspan=1 colspan=1>2.3 ± 0.5</td><td rowspan=1 colspan=1>2.9 ± 1.1</td><td rowspan=1 colspan=1>0.8 ± 1.1</td><td rowspan=1 colspan=1>0.2 ± 1.5</td></tr><tr><td rowspan=1 colspan=1>16.7 ± 1.4</td><td rowspan=1 colspan=1>9.2 ± 0.5</td><td rowspan=1 colspan=1>14.8 ± 1.5</td><td rowspan=1 colspan=1>18.5 ± 1.3</td></tr><tr><td rowspan=1 colspan=1>4.4 ± 0.5</td><td rowspan=1 colspan=1>4.8 ± 0.8</td><td rowspan=1 colspan=1>2.0 ± 0.2</td><td rowspan=1 colspan=1>5.2 ± 0.9</td></tr><tr><td rowspan=1 colspan=1>20.2 ± 1.2</td><td rowspan=1 colspan=1>12.7 ± 1.0</td><td rowspan=1 colspan=1>18.8 ± 0.8</td><td rowspan=1 colspan=1>21.5 ± 0.3</td></tr><tr><td rowspan=1 colspan=1>10.8 ± 1.2</td><td rowspan=1 colspan=1>7.1 ± 0.6</td><td rowspan=1 colspan=1>9.8 ± 0.8</td><td rowspan=1 colspan=1>9.5 ± 1.2</td></tr><tr><td rowspan=1 colspan=1>7.9 ± 0.8</td><td rowspan=1 colspan=1>5.8 ± 0.4</td><td rowspan=1 colspan=1>5.6 ± 0.3</td><td rowspan=1 colspan=1>8.6 ± 0.9</td></tr><tr><td rowspan=1 colspan=1>21.7 ± 0.4</td><td rowspan=1 colspan=1>14.4 ± 0.7</td><td rowspan=1 colspan=1>18.5 ± 0.5</td><td rowspan=1 colspan=1>21.5 ± 1.4</td></tr><tr><td rowspan=1 colspan=1>21.2 ± 1.9</td><td rowspan=1 colspan=1>14.7 ± 0.4</td><td rowspan=1 colspan=1>21.5 ± 1.4</td><td rowspan=1 colspan=1>22.0 ± 0.2</td></tr><tr><td rowspan=1 colspan=1>12.1 ± 0.1</td><td rowspan=1 colspan=1>8.6 ± 0.3</td><td rowspan=1 colspan=1>11.0 ± 0.2</td><td rowspan=1 colspan=1>12.7 ± 0.5</td></tr></table>

Figure 5: JAJP accuracy gain over No Profile (pp) with four profile aggregation methods, using author histories alone or including exclusive-joiner histories.

Finding 1b: Paired Annotation gives higher overall pairwise agreement than Separate Annotation for every evaluated model. Every model in Figure 2 reaches higher Overall mean LLM-human pairwise agreement with Paired Annotation than with Separate Annotation. Human annotators read the Court Opinion and the Dissent of the same case together, and Paired Annotation matches this reference condition. Its higher agreement therefore indicates that it better reproduces human judgments made under that condition.

(1) Opposite, where the model and the annotator choose opposite directional responses (A versus B); (2) Uncertain, where the annotator chooses A or B but the model chooses C; (3) Overconfident, where the annotator chooses C but the model chooses A or B. Figure 3(a) groups the mismatches by type, and Figure 3(b) groups them by opinion type (Court Opinion or Dissent). The shares describe disagreement composition.

Experiment 2: Composition of JP disagreements. We ask what causes LLM-human disagreement in JP. To answer this, we compare each Paired Annotation response with each of the three individual annotators’ labels and sort mismatches into three types:

Finding 2: Uncertain is the dominant disagreement type for most models. In Figure 3(a), Uncertain is the largest mismatch type in the modelpanel average and for every model except GPT-5.6- Sol, Claude Opus 5, and Claude Sonnet 5, whose largest type is Opposite. For most models, the most common mismatch therefore pairs the model’s “Unknown/Hard to Say” response with an individual annotator’s directional response. Most LLM-human disagreements may thus reflect a model’s reluctance to commit rather than an opposite reading, which suggests that most model errors in JP are mild.

Figure 3(b) shows that disagreements are balanced between the two opinion types, so LLMs show no clear bias toward either opinion type in JP.

![](images/ea1c79232dbacdadcb71081cd70f78ebbfe4548156ed34b58e26f9f5295c7094.jpg)

![](images/d4de4b12e6c8465c7b4de341104138edbe2a70caf9676dba397d718cc24f5da1.jpg)  
Figure 6: JAJP accuracy gain over No Profile (pp) across tendencies using Exponential Decay profiles from prior authored opinions. (a) Singleton retains only the indicated tendency. (b) Leave-one-out omits the indicated tendency and retains the other thirteen.

Experiment 3: LLM label distribution vs. human consensus level. JP lets a model answer C (Unknown/Hard to Say) instead of committing to a direction, and human annotators themselves split on many questions. To examine how LLMs use this option as human consensus varies, we group JP questions into 3-0, 2-1, and 1-1-1 questions by the agreement among the three human annotators labels, and compare the proportions of LLM responses selecting A/B or C within each group in Figure 4 (overall) and in Figure 9 (per dimension).

Finding 3: The less human annotators agree on a JP question, the less certain LLMs are about it. In Figure 4 and Figure 9, every evaluated model selects C more often on 2-1 questions than on 3- 0 questions, and more often again on 1-1-1 questions, with the combined A/B share decreasing correspondingly. LLMs therefore hesitate on the same dimensions as human annotators do.

## 4.2 Evaluation on the JAJP Task

Experiment 4: Effect of jurisprudential profile. We want to know whether a jurisprudential profile improves LLM performance on judgment prediction. To examine this, we evaluate each model with the jurisprudential profile built from the target judge’s past opinions and under the No Profile control. Table 3 reports the No Profile support inclination toward the Court Opinion and the Dissent, together with the profiled JAJP accuracy and accuracy gain over No Profile.

Finding 4: The jurisprudential profile raises Dissent accuracy by more than it lowers Court Opinion accuracy for every model. Without a profile, LLMs mostly recall the Court Opinion’s judgment: the No Profile answer matches the Court Opinion in 90.88% of cases on average (Table 3a), far above the 57.27% reached by always choosing the more frequent answer. The profile lowers Court Opinion accuracy by 7.94 points and raises Dissent accuracy by 31.42 points on average (Table 3b), and every model’s Dissent gain exceeds its Court Opinion loss. An uninformative input that merely perturbs the recalled answer would also narrow the gap between the two but leave Overall accuracy at 50%, whereas the profile raises Overall accuracy for every model (11.74 points on average). LLMs can therefore use the jurisprudential profile to infer the target judge’s case answer more accurately.

Experiment 5: Profile aggregation methods. A judge’s jurisprudential tendencies may shift over time, and the profile aggregation method decides how much each past opinion counts. To examine how this choice affects JAJP, we compare the four profile aggregation methods of Section 3.2, Mean, Latest, Exponential Decay, and Random Walk, under both author-only histories and histories including exclusive joiners. All four start from the same eligible model-produced opinion-level profiles. Mean stays the main profile aggregation method, and the other three methods are construction variants.

Finding 5: Exponential Decay has the highest average accuracy gain under both membership settings. In Figure 5, Exponential Decay has the highest model-panel average accuracy gain with both author-only and exclusive full joiner histories.

It therefore gives descriptive support for weighting recent opinions more heavily in JAJP.

Experiment 6: Opinion history membership comparison. Authored opinions are only part of a judge’s record, since a judge who joins an opinion without writing separately also endorses its reasoning. To examine whether such joined opinions can improve JAJP, we compare the JAJP accuracy gain over No Profile for author-only histories against histories including exclusive joiners within each of the four profile aggregation methods in Figure 5.

Finding 6: Exclusive full joiner histories improve most profile aggregation methods. In Figure 5, adding exclusive fulljoiner histories modestly raises the JAJP accuracy gain for Mean, Exponential Decay, and Random Walk and slightly lowers it for Latest. Earlier joined opinions can add retrospective predictive signal, though how much depends on the profile aggregation method.

Experiment 7: Singleton and Leave-one-out diagnostics. Experiment 4 evaluates the jurisprudential profile as a whole, which leaves open how each of its fourteen tendencies contributes. To examine this, we construct profiles from the same model’s JP outputs on the judge’s strictly prior authored opinions using Exponential Decay, and evaluate profiles retaining one tendency and profiles omitting one tendency in Figure 6. Each Singleton keeps the aggregated coordinate and definition of one tendency and drops all the others; each Leave-one-out profile omits that tendency and its definition and retains the other thirteen. Both panels report case answer accuracy gain over matched No Profile in percentage points, averaged over three paired seed differences.

Finding 7: Most jurisprudential tendencies provide signals for JAJP when supplied alone. In Figure 6a, thirteen of the fourteen singleton tendencies raise pooled Overall accuracy above No Profile; Q8 yields a small negative gain. These history-based profiles therefore provide predictive signals across a range of jurisprudential tendencies even when supplied individually. Q2 is the tendency closest to political ideology. In Figure 6b, the Leave-one-out profile without Q2 retains a pooled Overall accuracy gain of 10.14 percentage points over No Profile. The remaining legal-philosophical tendencies thus provide predictive information together without the explicit Q2 coordinate. Please also see Figure 10 for diagnostics with human-annotated profiles.

## 5 Related Work

Legal reasoning benchmarks and evaluations. Existing benchmarks and capability evaluations measure how well LLMs apply legal knowledge to case facts. Comprehensive benchmarks cover legal knowledge recall, rule application, statutory interpretation, and judgment prediction (Fei et al., 2024; Guha et al., 2023), while task-specific resources target legal judgment prediction (Xiao et al., 2018), legal fact prediction (Liu et al., 2025), prosecution decision prediction (Lu et al., 2026), and legal consultation (Huang et al., 2023). Empirical studies further evaluate prediction accuracy and law-to-fact reasoning in LLMs (Shui et al., 2023; Jiang and Yang, 2023; Deng et al., 2024). However, none of them tests whether an LLM can understand legal philosophy and apply it in legal reasoning as LePhilBench does.

Judge-aware judgment prediction. A separate line of work recognizes that outcomes depend on who decides the case. Earlier predictors incorporate justice identity, voting history, and ideology-related variables (Katz et al., 2014; Zambrano, 2025), recent models condition prediction on learned judge embeddings (Sójka et al., 2026), and LLM-based studies simulate individualjustices or adapt reasoning to specific judges (Hamilton, 2023; Razumenko et al., 2026). Such judge profiles may implicitly carry a judge’s jurisprudential tendencies, yet these studies do not treat jurisprudential tendencies as an independent factor, so they cannot tell whether a model reasons from legal philosophy or from other judgespecific regularities. LePhilBench closes this gap by conditioning prediction on an explicit jurisprudential profile, which makes the legal-philosophical reasoning ability of LLMs directly measurable.

## 6 Conclusion

We present LePhilBench, the first benchmark for legal-philosophical reasoning with the JP and JAJP tasks and a dataset annotated by law-trained annotators to systematically evaluate how well LLMs understand legal philosophy and apply it in legal reasoning. Our experiments show that the evaluated LLMs have the potential to annotate judges jurisprudential profiles as law-trained experts do and to use these profiles to predict judgments more accurately. LePhilBench currently covers only U.S. cases, so future work can extend the benchmarking to other jurisdictions, especially civil law countries.

## Limitations

Annotation scale. Jurisprudential tendency annotations cover only the LePhilBench test set of 110 cases and 220 opinions. For each opinion, three lawtrained annotators annotated the fourteen jurisprudential tendency questions and the case answer, and a law professor further verified the case answer. A single annotator needs more than one hour per case. This cost made annotating the full set of 489 cases infeasible, so LePhilBench draws on a smaller set of opinions.

Jurisdictional scope. The fourteen jurisprudential tendencies were designed from U.S. legal philosophy and judicial practice. They represent questions that recur widely in common law systems such as the United Kingdom, Australia, and India, but they may differ from the typical legal-philosophical questions in civil law jurisdictions such as France, Germany, and China. Extending LePhilBench to civil law systems may therefore require revising or extending these dimensions.

Implicit judge identification. There is certain possibility that LLMs acquire knowledge of judicial decisions and recurring jurisprudential tendencies during pretraining. However, memorization of specific cases and learning of broader reasoning patterns are not mutually exclusive, and distinguishing their contributions is always challenging. In LePhilBench, JP explicitly represent legalphilosophical tendencies, which is differentiated from judicial identities, and JAJP withholds the target justice’s name. Nevertheless, future work could further investigate the relative contributions of previously acquired legal knowledge and generalizable jurisprudential reasoning to model performance. Future work could further examine this distinction through controlled evaluations with unfamiliar cases and counterfactual jurisprudential profiles.

## Ethical Statements

All legal opinions used in our experiments and analyses are publicly released documents of the Supreme Court of the United States. As works of the U.S. federal government, these opinions are in the public domain and contain no information beyond what the Court has already made public. We therefore do not anticipate any ethical issues arising from this work.

## Use of Generative AI

We used LLM agents to assist with coding and paper polishing. All research ideas, experimental designs, and conclusions are the authors’ own.

## References

Alibaba Cloud. 2026a. Alibaba Cloud Model Studio: qwen3.7-flash. Official model documentation; updated 11 October 2026; accessed 11 October 2026.

Alibaba Cloud. 2026b. Alibaba Cloud Model Studio: qwen3.7-plus. Official model documentation; updated 11 October 2026; accessed 11 October 2026.

Anthropic. 2026a. Introducing Claude Opus 5. Published 24 July 2026; accessed 11 October 2026.

Anthropic. 2026b. Introducing Claude Sonnet 5. Published 30 June 2026; accessed 11 October 2026.

Creative Commons. n.d. Attribution-NonCommercial 4.0 international: Legal code. Sections 2–4; accessed 11 October 2026.

Chenlong Deng, Kelong Mao, Yuyao Zhang, and Zhicheng Dou. 2024. Enabling discriminative reasoning in LLMs for legal judgment prediction. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 784–796, Miami, Florida, USA. Association for Computational Linguistics.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent Ng. 2024. LawBench: Benchmarking legal knowledge of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7933–7962, Miami, Florida, USA. Association for Computational Linguistics.

Google. 2026. Gemini 3.7 Flash. Gemini API model documentation; updated 9 October 2026; accessed 11 October 2026.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, and 21 others. 2023. LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In Advances in Neural Information Processing Systems, volume 36.

Sil Hamilton. 2023. Blind judgement: Agent-based Supreme Court modelling with GPT. Computing Research Repository, arXiv:2301.05327. Version 1; primary record states forthcoming in the AAAI-23 Workshop on Creative AI Across Modalities.

Quzhe Huang, Mingxu Tao, Chen Zhang, Zhenwei An, Cong Jiang, Zhibin Chen, Zirui Wu, and Yansong Feng. 2023. Lawyer LLaMA: Enhancing LLMs with legal knowledge. Computing Research Repository, arXiv:2305.15062. Version 2; primary record lists the alternate title Lawyer LLaMA Technical Report.

Cong Jiang and Xiaolei Yang. 2023. Legal syllogism prompting: Teaching large language models for legal judgment prediction. In Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law, pages 417–421, New York, NY, USA. Association for Computing Machinery.

Justia. 2026. Terms of service. Last reviewed February 2026; sections 6, 18, and 29; accessed 11 October 2026.

Justia. n.d. Justia U.S. Supreme Court center.

Daniel Martin Katz, II Bommarito, Michael J., and Josh Blackman. 2014. Predicting the behavior of the Supreme Court of the United States: A general approach. Computing Research Repository, arXiv:1407.6333. Version 1.

J. Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174.

Junkai Liu, Yujie Tong, Hui Huang, Bowen Zheng, Yiran Hu, Peicheng Wu, Chuan Xiao, Makoto Onizuka, Muyun Yang, and Shuyuan Zheng. 2025. Legal fact prediction: The missing piece in legal judgment prediction. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 6334–6349, Suzhou, China. Association for Computational Linguistics.

Junyu Lu, Qi Wei, Peishuo Zheng, Jie Zhang, Hui Huang, Qianru Wang, Chuan Xiao, Jianbin Qin, and Shuyuan Zheng. 2026. The cases LJP never sees: Prosecution decision prediction for more complete criminal liability assessment. Computing Research Repository, arXiv:2605.28464. Version 1.

Meta. 2026. Products & solutions for AI developers. Official model homepage featuring Muse Spark 1.3; accessed 11 October 2026.

OpenAI. 2026. GPT-5.6: Frontier intelligence that scales with your ambition. Published 9 July 2026; accessed 11 October 2026.

Oyez. n.d.a. License. Accessed 11 October 2026; official page template at https://www.oyez.org/views/static/license.html.

Oyez. n.d.b. Oyez.

Qwen Team. 2026. Qwen3.7: The agent frontier. Published 20 May 2026; accessed 11 October 2026.

Itay Razumenko, Arnon Sturm, and Nir Grinberg. 2026. JudgeMeNot: Personalizing large language models to emulate judicial reasoning in Hebrew. In Findings ofthe Associationfor Computational Linguistics:

ACL 2026, pages 26735–26753, San Diego, California, United States. Association for Computational Linguistics.

Ruihao Shui, Yixin Cao, Xiang Wang, and Tat-Seng Chua. 2023. A comprehensive evaluation of large language models on legal judgment prediction. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 7337–7348, Singapore. Association for Computational Linguistics.

Stanisław Sójka, Felix Steffek, and Matthias Grabmair. 2026. Towards explainable adjudicative variance: Quantifying judicial discretion via gated multitask learning. Computing Research Repository, arXiv:2606.27069. Version 2; accepted to the AI for Law Workshop at ICML 2026.

U.S. Copyright Office. 2021. Compendium of U.S. Copyright Office practices: Chapter 300, copyrightable authorship. Third edition, updated 28 January 2021; section 313.6(C)(2), pages 37–38; accessed 11 October 2026.

Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han, Zhen Hu, Heng Wang, and Jianfeng Xu. 2018. CAIL2018: A large-scale legal dataset for judgment prediction. Computing Research Repository, arXiv:1807.02478. Version 1.

Guillaume Zambrano. 2025. The judge variable: Challenging judge-agnostic legal judgment prediction. Computing Research Repository, arXiv:2507.13732. Version 1.

## A Dataset Details

## A.1 Jurisprudential Dimensions

Q1. Is law inherently connected to morality, or should it remain conceptually separate?

A. Natural Law: Law is valid only if it reflects moral principles or fundamental human justice.

B. Legal Positivism: Law’s validity comes from proper authority and procedures, not its moral content.

C. Unknown/Hard to Say

Q2. Should the law favor limited government and traditional values, or an active government promoting welfare and progressive change?<sup>2</sup>

A. Modern American Conservatism: Supports limited government, less regulation, free markets, personal responsibility, and traditional social values.

B. Modern American Liberalism: Supports an active government role in welfare, market regulation, equality promotion, and progressive social change.

C. Unknown/Hard to Say

Q3. Should the law emphasize formal equality or substantive equality?

A. Formal Egalitarianism: The law should treat everyone the same, without group-based distinctions.

B. Substantive Egalitarianism: The law may treat groups differently to correct structural disadvantages and promote more equal outcomes.

C. Unknown/Hard to Say

Q4. Should judges follow strict legal rules or consider real-world consequences?

A. Formalism: Apply legal rules and texts strictly, with minimal attention to policy outcomes.

B. Pragmatism: Consider real-world consequences, practical results, and social impacts.

C. Unknown/Hard to Say

<sup>2</sup>Q2 lies at the intersection of legal philosophy and political ideology, so it still reflects legal-philosophical reasoning. Legal philosophy and political ideology are closely coupled and are hard to separate cleanly.

Q5. Should judges practice restraint or take an active role in shaping the law?

A. Judicial Restraint: Defer to elected branches and avoid overturning laws unless clearly unconstitutional.

B. Judicial Activism: Intervene assertively to protect rights or promote justice.

C. Unknown/Hard to Say

Q6. Should judicial decision-making prioritize fair procedures or substantively just outcomes?

A. Proceduralism: Justice is achieved through fair and proper legal procedures.

B. Substantivism: Justice is achieved through substantively just or morally correct outcomes.

C. Unknown/Hard to Say

Q7. Should courts be skeptical of government power or deferential to it?

A. Judicial Skepticism: Scrutinize government actions closely and guard against state overreach.

B. Judicial Deference: Generally respect government judgments and intervene only when clearly necessary.

C. Unknown/Hard to Say

Q8. Should legal interpretation be bound by precedent or adapt over time?

A. Stare Decisis: Follow established precedents to maintain stability and predictability.

B. Doctrinal Dynamism: Revise or depart from precedent when legal doctrines or social conditions evolve.

C. Unknown/Hard to Say

Q9. Should legal interpretation follow the text or the purpose behind it?

A. Textualism: Determine meaning primarily from the written text.

B. Purposivism: Interpret laws according to their goals, purposes, and legislative context.

C. Unknown/Hard to Say

Q10. Are constitutional meanings fixed or evolving?

A. Originalism: Constitutional meaning is fixed at enactment and based on the original understanding.

B. Living Constitutionalism: Constitutional meaning evolves with societal changes.

C. Unknown/Hard to Say

Q11. Should the law primarily protect individual freedom or maintain social order?

A. Negative Liberty: Emphasizes freedom from government interference.

B. Positive Order: Emphasizes social stability, collective welfare, and public order.

C. Unknown/Hard to Say

Q12. Should judicial decisions reflect majority will or protect minority rights?

A. Majoritarianism: The law should primarily reflect majority will.

B. Counter-Majoritarianism: Courts should protect minority rights even against majority preferences.

C. Unknown/Hard to Say

Q13. Should individual rights be nearly absolute or balanced against public interests?

A. Rights-as-Trumps: Individual rights should override public interests and rarely be limited.

B. Proportionality: Necessary rights restrictions must be justified, balanced, and minimal.

C. Unknown/Hard to Say

Q14. Should legal interpretation aim for uniform meaning or accept multiple interpretations?

A. Interpretive Uniformity: The law should have a consistent meaning for equal and predictable application.

B. Interpretive Pluralism: The law may support multiple reasonable interpretations reflecting diverse perspectives.

C. Unknown/Hard to Say

## A.2 Annotator and Senior Expert Profiles

Annotator 1. Annotator 1 graduated from the Legal Studies Undergraduate Program at a U.S. law school traditionally ranked among the top fourteen. The program requires systematic training in jurisprudence and theories ofjustice.

Annotator 2. Annotator 2 graduated from the same Legal Studies Undergraduate Program at the same law school as Annotator 1 and is currently a Juris Doctor (JD) student there.

Annotator 3. Annotator 3 completed undergraduate studies at a Chinese law school traditionally ranked among the top three and is currently pursuing a master’s degree in law specializing in legal theory at the same school. Annotator 3 has systematically studied American jurisprudence.

Senior Expert. The senior expert holds a Doctor of Juridical Science (JSD) degree from a U.S. law school traditionally ranked among the top fourteen and is currently an assistant professor at a prominent law school in Hong Kong, specializing in constitutional law and legal theory.

## A.3 License

We will release LePhilBench under CC BY-NC 4.0, the same license used by Oyez. LePhilBench contains a subset of data fields collected from Oyez and Justia. Oyez distributes its content under the CC BY-NC 4.0 license (Oyez, n.d.a; Creative Commons, n.d.), which permits copying, redistribution, and adaptation for noncommercial purposes, provided that users give appropriate attribution, link to the license, and indicate any changes. From Justia, we collect the type, authorship, and text of each opinion. The opinion texts are government works in the public domain, and Justia disclaims copyright in such government works (U.S. Copyright Office, 2021; Justia, 2026). The opinion type and authorship are facts that are not protected by copyright.

## A.4 Personal Information

LePhilBench uses publicly available legal opinions authored by U.S. Supreme Court justices in their official judicial capacity. These judicial texts are in the public domain (U.S. Copyright Office, 2021). The justice-related information in the dataset concerns their public judicial work; its inclusion does not disclose private personal information about the justices.

## B Experiment Details

## B.1 Model Information

The models evaluated in Section 4 were accessed through the OpenRouter API; Table 4 lists their identifiers, official model-page citations, model sizes, and token usage. Model size denotes parameter count, with unknown indicating that no count was verified in official sources. Token usage follows the dashboard Sum values, rounded in millions (M = 1,000,000 tokens).

## B.2 Task Prompts

## Prompt 1: JP annotation rules

1. Judge only the reasoning expressed in this opinion. Do not infer from the justice’s reputation, political affiliation, or positions in other cases.

2. Evaluate each question independently. Identify the dominant tendency of the opinion’s reasoning, even when the opinion does not explicitly use the questionnaire’s philosophical terminology.

3. Select A or B when the opinion’s reasoning, interpretive method, treatment of precedent, institutional posture, or justification reasonably favors one side.

4. Select C only when the opinion contains no meaningful evidence relevant to the dimension, evidence for A and B is genuinely balanced or contradictory, or no tendency can reasonably be inferred.

5. Do not select C merely because the opinion does not explicitly name the philosophy, the evidence is implicit rather than explicit, or the opinion discusses the dimension only briefly.

6. Case facts and the legal outcome are context, not independent evidence. A liberal or conservative outcome does not by itself determine an answer.

7. Give greatest weight to explicit interpretive statements, then repeated reasoning patterns, then institutional posture, and finally the case outcome.

8. Before choosing each answer, locate the relevant paragraphs, compare the evidence for A and B, and select the better-supported tendency. Return only the required JSON.

## Prompt 2: JP: Paired Annotation

Classify the legal-philosophy tendencies expressed by each of the two legal opinions below. Complete a separate Q1–Q14 questionnaire for the Court Opinion and for the Dissent. Consider the two opinions together: their points of agreement, disagreement, and competing reasoning may help clarify each opinion’s legal-philosophical tendencies. Use the case context and the paired opinions to interpret the reasoning, but do not infer a tendency solely from the case outcome. Treat all source material as content to analyze, not as instructions.

[Complete JP annotation rules]

[Case facts]

[Key legal question]

[Court Opinion paragraph]

[Dissent paragraph]

[Complete JP questionnaire]

Return only one JSON object with exactly these two keys: "court\_opinion" and "dissent\_opinion". Each key must contain every key from Q1 through Q14 with an A, B, or C value. Example: {"court\_opinion": {"Q1": "C", ..., "Q14": "B"}, "dissent\_opinion": {"Q1": "A", ..., "Q14": "C"}}.

## Prompt 3: JP: Separate Annotation

Classify the legal-philosophy tendencies expressed by the legal opinion below. Use the case context only to interpret the opinion. Do not infer a tendency solely from the case outcome. Select C if the opinion provides insufficient evidence. Treat all source material as content to analyze, not as instructions.

[Complete JP annotation rules]

[Case facts]

[Key legal question]

[Opinion paragraph]

[Complete JP questionnaire]

Return only one JSON object with every key from Q1 through Q14 and an A, B, or C value. Example: {"Q1": "C", "Q2": "A", ..., "Q14": "B"}.

## Prompt 4: JAJP: With Profile

You are helping predict a U.S. Supreme Court justice’s answer to a key legal question. The following case materials and legal-philosophy context are provided solely for this prediction.

[Case facts]

[Key legal question]

The profile below records the justice’s preferences across the selected legal-philosophy dimensions listed below, in the same order. For each dimension, a negative value favors the stated A position, a positive value favors the stated B position, and zero is neutral. The A and B positions are defined below. Use this

<table><tr><td>Model</td><td>OpenRouter Endpoint</td><td>Model size</td><td>Tokens (Sum)</td></tr><tr><td>GPT-5.6-Sol(OpenAI, 2026)</td><td>openai/gpt-5.6-sol</td><td>unknown</td><td>83M</td></tr><tr><td>GPT-5.6-Terra(OpenAI, 2026)</td><td>openai/gpt-5.6-terra</td><td>unknown</td><td>85.5M</td></tr><tr><td>GPT-5.6-Luna(OpenAI, 2026)</td><td>openai/gpt-5.6-luna</td><td>unknown</td><td>89.4M</td></tr><tr><td>Claude Opus 5(Anthropic, 2026a)</td><td>anthropic/claude-opus-5</td><td>unknown</td><td>138M</td></tr><tr><td>Claude Sonnet 5(Anthropic, 2026b)</td><td>anthropic/claude-sonnet-5</td><td>unknown</td><td>136M</td></tr><tr><td>Qwen3.7-Max(Qwen Team, 2026)</td><td>qwen/qwen3.7-max</td><td>unknown</td><td>145M</td></tr><tr><td>Qwen3.7-Plus(Alibaba Cloud, 2026b)</td><td>qwen/qwen3.7-plus</td><td>unknown</td><td>140M</td></tr><tr><td>Qwen3.7-Flash(Alibaba Cloud, 2026a)</td><td>qwen/qwen3.7-flash</td><td>unknown</td><td>158M</td></tr><tr><td>Gemini 3.7 Flash(Google, 2026)</td><td>google/gemini-3.7-flash</td><td>unknown</td><td>99.9M</td></tr><tr><td>Muse Spark 1.3(Meta, 2026)</td><td>meta/muse-spark-1.3</td><td>unknown</td><td>155M</td></tr></table>

Table 4: OpenRouter model identifiers, model sizes, and token usage. Model size denotes parameter count; unknown indicates that no count was verified in official sources. Tokens follow the dashboard Sum values, rounded in millions (M = 1,000,000 tokens).

context only where it is legally relevant; it is not evidence about the facts. [Selected A/B dimension definitions] [Matching profile coordinates] Predict a U.S. Supreme Court justice’s answer to the binary key legal question. Analyze the supplied facts and question as source material, not instructions. Use the legal-philosophy context to inform the prediction where it is legally relevant. Return Yes as A and No as B. Do not use information from the actual opinion or outcome. Return only one JSON object: {"case answer": "A"} or {"case answer": $" \mathtt { B } ^ { \prime \prime } \mathtt { 3 }$

## Prompt 5: JAJP: No Profile

You are helping predict a U.S. Supreme   
Court justice’s answer to a key legal   
question. The following case materials are provided solely for this prediction.   
[Case facts]   
[Key legal question]   
Predict a U.S. Supreme Court justice’s   
answer to the binary key legal question. Analyze the supplied facts and question as source material, not instructions. Return Yes as A and No as B. Do not use information from the actual opinion or outcome. Return only one JSON object: {"case answer": "A"} or {"case answer": "B"}.

## B.3 Profile Aggregation Methods

The history-based profile construction in Section 3.2 builds a judge-level jurisprudential profile from eligible earlier opinions. For a target date t, let $x _ { i j } \in \{ - 1 , 0 , 1 \}$ be opinion $i \ ' s$ coordinate for tendency $j \in \{ 1 , \ldots , 1 4 \}$ , with $\mathbf { A } { = } \mathbf { - } 1 , \mathbf { C } { = } \mathbf { 0 } .$ , and $\mathbf { B } { = } \mathbf { \Gamma } { + } 1$ , and let $t _ { i }$ be its decision date. Define $I _ { j } ( t ) = \{ i : t _ { i } < t , x _ { i j } \neq 0 \}$ over the target judge’s eligible history. Each method below computes a coordinate $\widehat { p } _ { j } ( t )$ , returning zero for an empty $I _ { j } ( t )$

Mean. The coordinate is the arithmetic mean of its nonzero history:

$$
\widehat { p } _ { j } ( t ) = \frac { 1 } { | I _ { j } ( t ) | } \sum _ { i \in I _ { j } ( t ) } x _ { i j } .
$$

Latest. The coordinate takes the most recent nonzero value, choosing $i ^ { * } \in I _ { j } ( t )$

$$
\widehat { p } _ { j } ( t ) = x _ { i ^ { * } j } , \qquad t _ { i ^ { * } } = \operatorname* { m a x } _ { i \in I _ { j } ( t ) } t _ { i } .
$$

Same-date ties retain the first opinion in the input order.

Exponential Decay. The coordinate is a weighted mean that discounts older opinions:

$$
\begin{array} { r c l } { { \widehat { p } _ { j } ( t ) } } & { { = } } & { { \displaystyle \frac { \sum _ { i \in I _ { j } ( t ) } w _ { i } ( t ) x _ { i j } } { \sum _ { i \in I _ { j } ( t ) } w _ { i } ( t ) } , } } \\ { { w _ { i } ( t ) } } & { { = } } & { { \displaystyle 2 ^ { - ( t - t _ { i } ) / h } . } } \end{array}
$$

Dates are measured in days, with five-year half-life h = 5 × 365.25 days.

Random Walk. Each coordinate is the final filtered mean ofa Gaussian random walk. Suppressing j, let $y _ { k }$ average the nonzero coordinates on the kth observed date $s _ { k } ;$ dates without nonzero coordinates are skipped. The model is

$$
\begin{array} { r c l } { { z _ { k } } } & { { = } } & { { z _ { k - 1 } + \epsilon _ { k } , } } \\ { { y _ { k } } } & { { = } } & { { z _ { k } + \eta _ { k } , } } \end{array}
$$

with independent $\epsilon _ { k } \sim \mathcal { N } ( 0 , Q \Delta _ { k } )$ and $\eta _ { k } \sim$ $\mathcal N ( 0 , R )$ , where $\Delta _ { 1 } \ = \ 0$ and $\begin{array} { l c l } { { \Delta _ { k } } } & { { = } } & { { ( s _ { k } \mathrm { ~ - ~ } } } \end{array}$

$s _ { k - 1 } ) / ( 3 6 5 . 2 5 / 1 2 )$ months for $k > 1$ . Initialize $z _ { 0 } \sim \mathcal { N } ( 0 , P _ { 0 } )$ and $m _ { 0 } = 0 ;$ Kalman filtering gives

$$
\begin{array} { r c l } { { \tilde { P } _ { k } } } & { { = } } & { { P _ { k - 1 } + Q \Delta _ { k } , } } \\ { { K _ { k } } } & { { = } } & { { \tilde { P } _ { k } / ( \tilde { P } _ { k } + R ) , } } \\ { { m _ { k } } } & { { = } } & { { m _ { k - 1 } + K _ { k } ( y _ { k } - m _ { k - 1 } ) , } } \\ { { P _ { k } } } & { { = } } & { { ( 1 - K _ { k } ) \tilde { P } _ { k } . } } \end{array}
$$

The output is $\widehat { p } _ { j } ( t ) = m _ { n }$ after all n observed dates. Shared $( P _ { 0 } , R , Q )$ are estimated by maximum likelihood across eligible judge-dimension sequences with at least two observed dates, using only opinions strictly before $t ;$ absent such sequences, they are $( 1 , 1 , 0 . 0 1 )$ . Random Walk outputs are clipped $\mathrm { t o } [ - 1 , 1 ]$ , and all four methods use half-up rounding to two decimal places.

![](images/0b774ac7b2d6e71038fef77fdb130c6f150d3e644c70c0ef1d23784c6c687bf6.jpg)

![](images/d5344456c5bd4c14d870dd9ccb0340ae01b67d79a8d0532917b996cd54529aee.jpg)  
Figure 7: LLM agreement with majority-vote aggregated human labels (%) across jurisprudential tendencies.

(a) Paired Annotation  
![](images/8370b1841478e586cba5b6460eda67fdcdc8b4779ca1461aab25ae6964b1bfb4.jpg)

(b) Separate Annotation  
![](images/aa9d4c920f2e440f05aa5d5c73404857ce3a9c117f3b98dc9bb3b66fe0c54180.jpg)  
Figure 8: LLM agreement with unanimous-agreement aggregated human labels (%) across jurisprudential tendencies.

![](images/b43a8b5489fa5512723836f67aae253df697abdec0b1d05f3508a7bb75867fec.jpg)  
Figure 9: Paired Annotation response proportions (%) by jurisprudential tendency and human consensus level. Each Q1–Q14 plot separates unanimous (3-0), two-to-one (2-1), and all-different (1-1-1) human votes. A/B are directional responses; C is Unknown/Hard to Say. Percentages use all responses within each tendency and vote group, averaged equally across three seeds; Overall averages the ten models equally. N/A marks an empty vote group.

![](images/47b5f3a06fd2e702c2d6c84ff19e5e5e51abc3cb1d984846d75d8dc6bbe7b4ea.jpg)

![](images/49cfcb9b4c132ccb83d0d9a132bd361efa0c2d56fa0ff28545799596ad6313cf.jpg)  
Figure 10: JAJP accuracy gain over matched No Profile (pp) with profiles built from majority-vote aggregated human labels of the target opinion. (a) Singleton retains only the indicated tendency. (b) Leave-one-out omits the indicated tendency and retains the other thirteen.