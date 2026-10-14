# Attacking Only Half the Data: Two-Sided Evaluation of AI-Text Detector Robustness

Apu Chandraw Shill Verbex.ai apu3660@gmail.com

Rabindra Nath Nandi Independent Researcher rabindro.rath@gmail.com

## Abstract

We investigate the robustness of widely used AI text detectors under attacks applied to both human and machine text. Prior evaluations typically attack only machine text, overlooking that human text is routinely modified through grammar correction, formatting, or invisible character insertion. We introduce a two-sided protocol that applies attacks to both classes. Across two corpora and nine detectors, reported AUROC gains vanish under two-sided evaluation, with attacked human text misclassified at rates up to 100%. While simple preprocessing reverses character-level attacks, word-level synonym substitution remains effective, meaningpreserving, and irreversible.

## 1 Introduction

Machine-generated text detectors are increasingly deployed in high-stakes contexts such as academic integrity checks, content moderation, and hiring. Yet their reliability has been questioned from the very beginning (Krishna et al., 2023). A central problem is that detectors often fail when text is altered, even slightly, by paraphrasing, grammar correction, or invisible character insertion. These failures matter because text is rarely submitted in a perfectly “clean” form.

Prior work has largely focused on attacks against machine-generated text, comparing it to untouched human writing. For example, paraphrasing has been shown to collapse detector performance (Krishna et al., 2023), and subsequent studies argued that this weakness is fundamental (Sadasivan et al., 2025). Benchmarks such as RAID (Dugan et al., 2024) have made paraphrase the standard stress test. More recent work has begun to explore the human side: polishing human text with an LLM can cause detectors to misclassify it (Saha and Feizi, 2025), while DetectRL (Wu et al., 2024) attacks human text but evaluates against clean machine text.

However, these designs remain one-sided, either attacking only machine or only human text, and often re-tuning thresholds so that false positives never surface. As a result, they miss the full picture.

In this work, we address the missing half. We introduce a two-sided evaluation protocol that applies attacks to both human and machine text and scores them against each other. This design reveals failures that are invisible under one-sided evaluation. For example, under zero-width insertion, detectors flag up to 98% of human documents while appearing to improve AUROC when only machine text is considered. The apparent “gain” is actually a collapse once human text is included. Across two corpora and nine detectors, we find that reported gains under attack vanish under two-sided evaluation, and the cost of ignoring the human side can be devastating.

Our findings also show that some damaging attacks are surprisingly easy to undo. Simple preprocessing steps, such as stripping zero-width characters, collapsing repeated spaces, or folding homoglyphs restore detection performance to near-clean levels. By contrast, word-level synonym substitution remains effective, meaning-preserving, and irreversible.

Our main contributions are as follows:

• We introduce a two-sided protocol that applies each attack to human as well as machine text and scores the two against each other.

• We show that reported gains under attack are not real. Across two corpora and nine detectors, 39 of 162 cells show an AUROC gain, 31 significantly, and none survives two-sided as a significant gain.

• We measure the cost of the missing half. At a threshold set to 1% FPR of clean human documents, attacked human text is flagged at up to 100%.

• We show that DIPPER paraphrase no longer evades most detectors. It raises AUROC for five of nine while lowering detection for all nine, and a defense built on it fails for the same reason.

• We test sanitization as a defense. Zero-width and whitespace attacks are fully recoverable, and homoglyphs only with a matching table. Word substitutions are not recoverable.

• We show that generator choice and text length move detectors further than any attack, and that short text is where false accusations concentrate.

The source code and data will be made publicly available upon acceptance.

## 2 Related Work

Zero-shot and supervised detection. Likelihood-based statistics date to GLTR (Gehrmann et al., 2019). DetectGPT scores how far a document’s likelihood drops when a few words are changed (Mitchell et al., 2023), and Fast-DetectGPT computes the same thing in one forward pass (Bao et al., 2024). Binoculars compares a pair of models (Hans et al., 2024), and Lastde reads token likelihoods as a time series (Xu et al., 2025). Supervised detectors instead fine-tune encoders on labeled data (Guo et al., 2023), and RADAR trains one against a paraphraser (Hu et al., 2023).

Attacks and robustness benchmarks. Krishna et al. (2023) established paraphrase evasion and proposed retrieval as a defense. RAID (Dugan et al., 2024) provides the largest attack-annotated corpus, including the character-level attacks we study. DetectRL (Wu et al., 2024) is the closest prior work. It does attack human text. But it scores that text against clean machine text, which is the usual one-sided design in mirror image. It also reports only AUROC and F1, with the threshold chosen by Youden’s J on the evaluation set itself. Because that threshold is refitted for every condition, it cannot show a change in the false-positive rate. Creo and Pudasaini (2025) show that homoglyph substitution drives seven detectors to nearrandom behavior and name unicode normalization as a possible defense without evaluating it. RAFT (Wang et al., 2024) and GREATER (Li et al., 2025) establish word-level substitution as an attack family where Sai Teja et al. (2025) organise attacks into a taxonomy, report TPR at strict FPR, and build a detector that normalizes its input and feeds the discrepancy between the raw and canonical text to a classifier. That treats normalization as a feature source inside a trained model. We instead measure it as a preprocessing defense in isolation, asking how much of a detector’s clean operating point plain sanitization restores (Section 4.7), which their design does not report.

Paraphrase does still defeat detectors when the paraphraser is purpose-built for evasion: Shportko and Verbitsky (2025) drive Binoculars from 0.750 to 0.553 F1 with a commercial humanizer, Ranganath and Ramesh (2026) reach mean TPR@1%FPR 0.001 with a reinforcement-trained policy, and Zha et al. (2025) report failure under iterated paraphrase. Our result concerns DIPPER, the paraphraser of Krishna et al. (2023), applied once. What has expired is not paraphrasing but incidental paraphrasing.

Attacks that help the detector. Rivera Soto et al. (2026) show that style transfer can raise detectability. The rewriting leaves its own signature, and Binoculars goes from 79 to 92 AUROC. We see the same effect for DIPPER paraphrase and for unicode attacks on likelihood detectors, and we show the gain comes from how it is measured. La Cava and Tagarelli (2026) independently report that Binoculars and Fast-DetectGPT break under homoglyph and zero-width attacks. McGovern et al. (2025) find that generator fingerprints survive paraphrase, which our results support.

False positives on human text. Saha and Feizi (2025) are the closest prior work. They run human text through an LLM and show detectors flag it, with GLTR rising from 6.8% to 40.9% on lightly edited text. Three things separate our result. Their edits are rewrites a reader can see, where ours look identical. Their threshold is tuned for accuracy, where ours is fixed to flag 1% of held-out clean human text. And they never attack the machine text, so the two sides are not treated alike. Liang et al. (2023) show detectors misclassify non-native English writing, with false-positive rates above 60% on TOEFL essays.

Paraphrase defenses. Tufts et al. (2025) also use strict FPR thresholds, but only on unattacked human text. Rivera Soto et al. (2025) train a model to undo paraphrasing and gain +22% AUROC. But our experiment shows an opposite outcome (Section 4.9).

## 3 Experimental Setup

Data. We use two datasets: RAID<sup>1</sup>(Dugan et al., 2024) and DetectRL<sup>2</sup> (Wu et al., 2024), and Figure 8 shows how both are sampled and scored. From RAID, we sample evenly across eight domains, selecting 90 machine and 180 human documents per domain of at least 30 words, each with eleven attack variants. This yields 27,029 texts from 2,160 sources, spanning 11 generators with greedy and sampling decoding. For length analysis, 20% of sources are truncated to 50, 100, and 200 words. Human texts receive the same attacks as machine texts, enabling false-positive analysis.

DetectRL is sampled similarly: 1,440 human and 720 machine sources across four domains and four generators, each with six attack variants. Since DetectRL lacks zero-width attacks, we add RAID’s implementation (Section 4.4), producing 17,280 texts. We use its raw corpus rather than prepared splits, as two splits mislabel 33% of human text as machine (Appendix C).

Detectors. We evaluate nine detectors, all normalized so that higher scores indicate more machine-like text. They fall into three families: Modern zero-shot: Binoculars (Falcon-7B pair), Fast-DetectGPT (analytic, on gpt-neo-2.7B, gpt2- xl, gpt2), and Lastde (gpt-neo-2.7B). Legacy likelihood: mean log-likelihood and log-rank (gpt-neo-2.7B). Trained classifiers: RADAR (Hu et al., 2023) and HC3-trained RoBERTa (Guo et al., 2023). The detector list is available at Table 10.

Metrics and Protocol. For each detector and attack we report two key metrics: AUROC (Area Under the Receiver Operating Characteristic), which measures the probability that a detector ranks a machine text higher than a human text, and TPR (True Positive Rate) at 1% FPR (False Positive Rate), which measures the fraction of machine texts correctly flagged when only 1% of clean human texts are falsely flagged, following Krishna et al. (2023).

Each threshold is the 99th percentile of the scores on half the clean human documents, and everything is measured on the other half (Appendix E). We also report two numbers the onesided protocol never produces: the share of attacked human documents that the same thresholds flag as AI and the AUROC between the two attacked classes. Let S for a detector’s score, H and

M for clean human and machine text, and $H _ { a } , M _ { a }$ for the same documents under attack a, the two AUROCs differ in one argument:

$$
\operatorname { A U R O C 1 } ( a ) = \operatorname { A U R O C } ( S ( H ) , \ S ( M _ { a } ) )\tag{1}
$$

$$
\operatorname { A U R O C 2 } ( a ) = \operatorname { A U R O C } ( S ( H _ { a } ) , S ( M _ { a } ) )\tag{2}
$$

Every comparison carries a 95% bootstrap confidence interval (CI) from 1,000 seeded replicates, each refitting its own threshold. On the held-out half, the thresholds flag 0.011 of clean human documents on RAID and 0.009 on DetectRL against a target of 0.01, pooled over the nine detectors (Table 3), so the false accusations of Section 4.3 come from the attacks and not from a badly fitted threshold.

Length control and excluded data. For the length analysis, we cut human text to the same lengths as the truncated AI text so that AUROC compares machine against human rather than short against long. We dropped one document, a paraphrased text containing nothing but an ellipsis that Lastde could not score, leaving 243,254 scores.

## 4 Results

We evaluate nine detectors across eleven RAID attacks and repeat the study on DetectRL (Table 3, Figure 8). Importantly, both human and machine texts are attacked. When scored against clean human text, 22 detector–attack pairs appear to improve. However, when scored against attacked human text, 39 pairs become significantly worse (Table 3, Table 7). These apparent improvements occur because attacks also push human text toward “AI-like” (Figure 4), causing detectors to misclassify up to 98% of innocent writers at thresholds calibrated to flag only 1% (Figure 1, Table 9).

We further find that paraphrase no longer reliably hides AI text. AUROC even rises for five of nine detectors, yet all nine detect fewer AI documents at a fixed cutoff (Table 1, Table 8). By contrast, simple character edits cause far greater damage (Figure 2). Preprocessing can undo two of the three worst attacks, but word substitution remains resistant (Table 6, Table 4). Finally, for eight of the nine detectors, the choice of generator influences performance more strongly than any attack.

![](images/f34380f3681cdf19762d9de2a9f50054fd3ed3538f53902c28fa66ea2b021123.jpg)

Figure 1: False-positive rate on attacked human text at thresholds calibrated for 1% FPR on clean human text (red line).
<table><tr><td></td><td colspan="2">AUROC</td><td colspan="2">TPR@1%FPR</td></tr><tr><td>Detector</td><td>clean</td><td>para.</td><td>clean</td><td>para.</td></tr><tr><td>Binoculars</td><td>.854</td><td>.935</td><td>.74</td><td>.68</td></tr><tr><td>FDG (neo)</td><td>.828</td><td>.913</td><td>.69</td><td>.59</td></tr><tr><td>FDG (gpt2-xl)</td><td>.763</td><td>.907</td><td>.58</td><td>.53</td></tr><tr><td>FDG (gpt2)</td><td>.725</td><td>.849</td><td>.38</td><td>.26</td></tr><tr><td>Lastde</td><td>.840</td><td>.814</td><td>.41</td><td>.18</td></tr><tr><td>LogLik</td><td>.752</td><td>.693</td><td>.21</td><td>.06</td></tr><tr><td>LogRank</td><td>.771</td><td>.708</td><td>.23</td><td>.07</td></tr><tr><td>RADAR</td><td>.874</td><td>.857</td><td>.24</td><td>.21</td></tr><tr><td>RoBERTa-cls</td><td>.724</td><td>.814</td><td>.23</td><td>.19</td></tr></table>

Table 1: Clean vs. paraphrase-attacked performance. Bold marks where AUROC rises under attack while detection at 1% FPR falls.

## 4.1 Paraphrasing no longer evades most detectors

DIPPER paraphrase raises AUROC for five of the nine detectors, yet all of them catch fewer AI texts at a fixed cutoff (Table 1). The attack shifts human text toward “AI-like” while making AI scores more uniform (Figure 4). In effect, the paraphraser leaves its own signature on the text rather than removing the generator’s. Binoculars illustrates this most clearly: AUROC increases by 0.081, but detection falls from 0.74 to 0.68 (CI [−0.129, −0.008]), with both changes statistically significant (Figure 6).

## 4.2 The simplest attacks do the most damage

Figure 2 maps all 99 detector–attack combinations. The most damaging attacks are simple character edits. For example, zero-width insertion reduces RoBERTa by 0.705 AUROC and drives its detection rate to zero. Homoglyph substitutions show mixed effects: they raise Fast-DetectGPT at GPT-2 by +0.230, but lower it by 0.304 at GPT-Neo.

<table><tr><td rowspan="2">backbone</td><td rowspan="2">clean AUROC</td><td colspan="3">zero-width</td><td colspan="3">homoglyph</td></tr><tr><td>1-s</td><td>2-s</td><td>FPR</td><td>1-s</td><td>2-s</td><td>FPR</td></tr><tr><td>RAID</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>gpt-neo</td><td>0.828</td><td>0.69</td><td>0.64† 0.02 0.52 0.60† 0.01</td><td></td><td></td><td></td><td></td></tr><tr><td>falcon-7b</td><td>0.854</td><td>1.00*</td><td></td><td>0.58† 1.00 0.84 0.64† 0.23</td><td></td><td></td><td></td></tr><tr><td>DetectRL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>gpt-neo</td><td>0.902</td><td>0.67</td><td></td><td>0.66† 0.01</td><td>一</td><td>一</td><td>一</td></tr><tr><td>falcon-7b</td><td>0.942</td><td>1.00*</td><td></td><td>0.47† 0.99</td><td>一</td><td>一</td><td>一</td></tr></table>

Table 2: Fast-DetectGPT at gpt-neo-2.7B and at the falcon-7b pair authors recommend. 1-s and 2-s are oneand two-sided AUROC, and FPR is the fraction of attacked human documents flagged. <sup>∗</sup> marks a significant gain over clean, <sup>†</sup> a significant loss. DetectRL’s homoglyph attack also substitutes words, so they are left blank.

No single attack harms all nine detectors, and the same attack can benefit one detector while severely weakening another, depending on the backbone used (Figure 3).

## 4.3 Attacks move both classes

We score attacked human text against attacked AI text and measure the score shift.

For a class C under attack a, the shift is defined as the paired score change divided by the clean gap between the classes:

$$
\Delta _ { a } ( C ) = { \frac { 1 } { | C | } } \sum _ { x \in C } { \frac { S ( x _ { a } ) - S ( x ) } { \bar { S } ( M ) - \bar { S } ( H ) } }\tag{3}
$$

For example, zero-width insertion moves human text +2.3 separation units toward “AI,” while machine text shifts only +1.4 (Figure 4). This causes innocent human text to be flagged. Under log-likelihood, zero-width insertion raises AUROC from 0.752 (clean) to 0.996 (one-sided), but only 0.576 (two-sided). At a threshold set to flag 1% of clean human text, attacked human text is flagged 98% of the time (95% CI [0.87, 1.00]). The same attack produces 81% false positives for log-rank and 62% for Binoculars (Figure 1).

A stronger detector is a worse accuser. Fast-DetectGPT’s reference implementation defaults to a falcon-7b pair, so we re-ran it there too (Table 2). It performed better on clean text by increasing AU-ROC for both RAID and DetectRL. But it flags 99.9% of attacked human documents, whereas the weaker backbone flagged only 2%. So, improving the detector made it confidently wrong about more innocent writers.

![](images/f8f2475430d68e8bf7ce156184395ee070082d82fcfd647c5eb7b793f9bee742.jpg)  
Figure 2: Effect of each attack on each detector (∆AUROC vs. no attack). Blue helps the detector; red breaks it; ∗ marks deltas whose 95% bootstrap CI excludes zero. Effects are family-antagonistic: the zero-width column simultaneously contains the largest positive and largest negative entries.

![](images/b51f77c0fcd2136b83f75e4794a7cc5688543a938560879e3f55c2776455e6b3.jpg)  
Figure 3: Change in AUROC under attack, one mark per detector by family. Synonym substitution breaks probability-based detectors, Unicode attacks break the classifiers.

## 4.4 Replication of results on DetectRL corpus

We repeat the study on DetectRL (Wu et al., 2024), which differs from RAID in domain (arXiv, news, stories, reviews), generator (GPT-3.5-turbo, PaLM-2, Claude-instant, Llama-2-70b), and attack implementation. Six of its seven attacks are unique to DetectRL. Since DetectRL does not include a zerowidth attack, we generate one from its documents using RAID’s insertion rule.

The false accusations are even worse here than on RAID. Under zero-width insertion, loglikelihood and log-rank flag all 1,440 attacked human documents, whereas Lastde flags 68% and Binoculars 31%. The overall pattern mirrors RAID: out of 63 cells, none shows a significant two-sided gain (Table 3). Under back-translation, RADAR’s AUROC rises from 0.921 to 0.956, but when measured two-sided it falls to 0.554, and the detector wrongly flags 37% of human documents (Table 13).

![](images/bea2697f14c9d227dc790bd0b44a3d4c1412c7a04b452e3b38bd372c084280f8.jpg)  
Figure 4: Paired score shift under attack for human (blue) and AI (orange) text, in units of each detector’s clean human to AI gap. Positive means toward “AI”, negative toward “human”.

<table><tr><td></td><td>RAID</td><td>DetectRL</td></tr><tr><td>domains</td><td>8</td><td>4</td></tr><tr><td>generator configurations</td><td>22</td><td>4</td></tr><tr><td>(detector, attack) cells</td><td>99</td><td>63</td></tr><tr><td>realized FPR, clean human</td><td>0.011</td><td>0.009</td></tr><tr><td>worst FPR, attacked human</td><td>0.98</td><td>1.00</td></tr><tr><td>one-sided AUROC gains</td><td>22</td><td>17</td></tr><tr><td>significant</td><td>18</td><td>13</td></tr><tr><td>two-sided gains</td><td>0</td><td>0</td></tr><tr><td>significant two-sided losses</td><td>39</td><td>54</td></tr></table>

Table 3: The two-sided result on both corpora

## 4.5 Short text draws false accusations

Fast-DetectGPT struggles more on short documents. Its true positive rate at a 1% threshold drops from 0.72 to 0.32 when the text length is reduced to 50 words, while its false-accusation rate remains at 0.01. Binoculars remains stable across all lengths. RADAR appears to improve, rising from 0.24 to 0.74, but in reality it flags 61% of 50-word human texts at the same threshold (Figure 7).

Short texts such as chat messages, exam answers, and social posts are where false accusations concentrate. In these cases, the choice of detector largely determines the outcome. Thus, short texts amplify detector bias and increase the risk of misclassification.

## 4.6 Only synonym substitution survives sanitization

Synonym substitution replaces about a fifth of the words in a document (median 21%). It keeps the meaning better than paraphrase, at 0.92 sentence similarity against 0.87 (Table 15).

It is most effective against probability-based detectors. It drops Binoculars by 0.181 AUROC and from 0.74 to 0.30 TPR@1%FPR, whereas Lastde and legacy likelihood pairs fall by 0.290 and 0.24, respectively. The attack pushes both classes toward “human”, but machine text more than twice as far (Table 4). Machine text is built from highprobability words, and replacing those with less likely synonyms costs it the most.

It barely has any effect on the trained classifiers. Because they read word choice and style rather than token probabilities, and a synonym word leaves those intact. For zero-shot detectors, its Falsepositive rate on attacked human text is ≤0.003 and acts like pure evasion.

![](images/993ca47f6cd56d03d91a55a82a929b3ad9e28fb34e17c99a00d3681c11975f13.jpg)  
Figure 5: Attacked documents re-scored after sanitization, which strips zero-width characters, folds confusables, collapses whitespace, and applies NFKC. Thresholds are unchanged from the clean calibration.

## 4.7 Sanitization removes the Unicode attacks entirely

Homoglyphs and zero-width characters change the tokens a detector sees while leaving the page identical to a reader, so they can be reversed. We remove the zero-width characters and use a curated table of 47 pairs (Table 18) to replace each look-alike character with the letter it imitates, so Cyrillic U+0430 becomes Latin a. We then apply NFKC normalization to turn text back into its plain form. Each step repairs only its own kind of change, which is why all three steps are needed.

The text and the detection both come back. Zero-width insertion leaves 0.000 character trigram similarity to the source, and sanitization brings it to 0.999, with 97% of documents restored byte for byte (Table 5). The remaining 3% differ only because NFKC also normalizes characters the source already contained. Re-scored at the unchanged thresholds, every detector returns to its clean operating point (Figure 5). Log-likelihood’s falseaccusation rate under zero-width falls from 0.985 back to 0.010, and homoglyph detection recovers to 86 to 104% of clean. DetectRL’s homoglyph attack also swaps words, so recovery there stops at 16 to 91%.

Extra spaces are easy to undo. The whitespace attack adds extra spaces between words. So replacing every group of spaces with a single space brings detection back to 90 to 100% of its clean value for all nine detectors (Table 6).

A larger confusable table is worse. Unicode’s full UTS #39 list removes more of the substituted characters than our curated table (98.0% against 61.8%, Table 16), yet it does not recover detection and raises false accusations on clean human text for four of nine detectors, where the curated table raises none (Table 6). The large table maps many different characters to the same letter. For example, a look-alike capital I comes back as a lowercase l. On RAID, it rewrites 15% of clean documents against the curated table’s 3% (Table 16). Unicode attacks look strong in the literature when the text is not cleaned, so the reported number measures whether the text was processed, not how good the detector is. One line of preprocessing decides whether these attacks look devastating or harmless.

<table><tr><td></td><td></td><td></td><td colspan="2">TPR@1%</td><td colspan="2">score shift</td></tr><tr><td>detector</td><td>family</td><td>∆AUROC</td><td>clean</td><td>syn.</td><td>AI</td><td>human</td></tr><tr><td>Binoculars</td><td>modern zero-shot</td><td>-0.181</td><td>0.74</td><td>0.30</td><td>-0.74</td><td>-0.30</td></tr><tr><td>FDG-neo</td><td>modern zero-shot</td><td>-0.245</td><td>0.69</td><td>0.22</td><td>-1.01</td><td>-0.68</td></tr><tr><td>FDG-gpt2xl</td><td>modern zero-shot</td><td>-0.214</td><td>0.58</td><td>0.23</td><td>-1.14</td><td>-0.74</td></tr><tr><td>FDG-gpt2</td><td>modern zero-shot</td><td>-0.167</td><td>0.38</td><td>0.11</td><td>-1.10</td><td>-0.74</td></tr><tr><td>Lastde</td><td>modern zero-shot</td><td>-0.290</td><td>0.41</td><td>0.09</td><td>-0.78</td><td>-0.38</td></tr><tr><td>LogLik</td><td>legacy stat.</td><td>-0.238</td><td>0.21</td><td>0.09</td><td>-0.93</td><td>-0.77</td></tr><tr><td>LogRank</td><td>legacy stat.</td><td>-0.236</td><td>0.23</td><td>0.09</td><td>-0.90</td><td>-0.73</td></tr><tr><td>RADAR</td><td>trained clf.</td><td>-0.008</td><td>0.24</td><td>0.16</td><td>-0.09</td><td>+0.01</td></tr><tr><td>RoBERTa</td><td>trained clf.</td><td>-0.028</td><td>0.23</td><td>0.17</td><td>-0.14</td><td>-0.05</td></tr></table>

Table 4: Synonym substitution by detector family. Probability-based detectors collapse; trained classifiers are nearly unaffected. Score shifts are paired (attacked minus source) in units of each detector’s clean human–AI separation: the attack pushes both classes toward “human”, but AI text roughly twice as far.

<table><tr><td rowspan="2">attack</td><td colspan="2">trigram sim. to source</td><td rowspan="2">exact restore</td></tr><tr><td>attacked</td><td>sanitized</td></tr><tr><td>zero-w.</td><td>0.000</td><td>0.999</td><td>97%</td></tr><tr><td>homogl.</td><td>0.118</td><td>0.988</td><td>31%</td></tr><tr><td>synonym</td><td>0.689</td><td>0.688</td><td>0%</td></tr><tr><td>paraphr.</td><td>0.504</td><td>0.504</td><td>0%</td></tr><tr><td>whitesp.</td><td>0.930</td><td>0.929</td><td>0%</td></tr><tr><td>up/low</td><td>0.948</td><td>0.947</td><td>0%</td></tr><tr><td>art-del</td><td>0.987</td><td>0.987</td><td>3%</td></tr><tr><td>ins-par</td><td>0.974</td><td>0.974</td><td>4%</td></tr><tr><td>number</td><td>0.979</td><td>0.979</td><td>47%</td></tr><tr><td>alt-sp.</td><td>0.993</td><td>0.993</td><td>34%</td></tr><tr><td>misspell</td><td>0.991</td><td>0.991</td><td>22%</td></tr></table>

Table 5: Text similarity to the source before and after sanitization, and the share of documents restored exactly. For word-level attacks, a non-zero exact restore means the attack found nothing to change, not that sanitization repaired it.

## 4.8 Generator choice matters more than any attack

We score each detector on clean text from eleven generators, each under greedy decoding and under sampling, 22 setups in all (Table 11). Binoculars separates ChatGPT text from human text perfectly and scores 1.000 AUROC, but on MPT with sampling, it scores only 0.430. That gap of 0.570 is three times the damage its worst attack does (0.181). The gap is 0.374 for RADAR and 0.644 for log-likelihood, and only RoBERTa’s zero-width loss 0.705 exceeds its generator gap. The weak setups are older base models with sampling, and every perfect score uses greedy decoding. A study that evaluates only chat models is not likely to have encountered the ones that defeat the detector.

<table><tr><td rowspan="2">attack</td><td colspan="3">detection after cleaning, % of clean</td></tr><tr><td>curated</td><td>full UTS #39</td><td>curated + ws</td></tr><tr><td>zero-w.</td><td>100-100</td><td>100-102</td><td>99-100</td></tr><tr><td>homogl.</td><td>86-104</td><td>80-105</td><td>86-103</td></tr><tr><td>whitesp.</td><td>0-92</td><td>92-101</td><td>90-100</td></tr><tr><td>cost</td><td>0/9</td><td>4/9</td><td>1/9</td></tr></table>

Table 6: Detection after each sanitizer configuration, as a percentage of clean detection. Cells show the range over the nine detectors. Cost counts detectors that wrongly flag more clean human text.

## 4.9 Paraphrasing as a defense

Section 4.1 showed that paraphrasing makes text appear more machine-like. This suggests a possible defense: rather than scoring the document directly, paraphrase it first and then score the output. An attacker can craft a document to slip past a detector, but cannot control what a paraphraser will produce. To test this, we sampled 100 human and 100 machine documents across RAID’s eight domains, along with those same machine documents under paraphrase attack, and generated K=5 paraphrases of each.

Under Fast-DetectGPT, machine text fell from a mean score of 4.56 to −0.50, while human text fell only from −0.24 to −1.68. The gap between the classes shrank from 4.80 to 1.18. As a result, the five paraphrases separate the classes worse than the single document they replaced, with AUROC dropping from 0.95 to 0.81 on clean text and from 0.96 to 0.77 on attacked text. Log-likelihood shows the same trend, falling from 0.86 to 0.63.

A paraphraser’s output reflects the paraphraser itself, not the input. Paraphrases overwrite the authorship signal of the text, which determines what the detector reads. The paraphrases still separate the classes at 0.81, consistent with McGovern et al. (2025), but this is worse than the 0.95 separation the defense was meant to improve.

## 5 Recommendations

(1) Attack the human text too. Report attackedhuman FPR alongside attacked-AI TPR. Attacking only AI text overestimates robustness. (2) Report TPR at a fixed FPR, with CIs. AUROC can rise while detection at a 1% threshold falls (Figure 6). (3) Report the generator mix. It moves performance further than any attack does (Section 4.8), so a number quoted without it is not comparable across papers. (4) Report results by text length. A detector can look strongest on short text while accusing most of its short human inputs (Section 4.5). (5) Report per-family results. Robustness does not transfer between families, so a claim about one detector says nothing about the other. (6) Strip zero-width characters and collapse whitespace. Both are free and both fully reverse their attacks (Section 4.7). Fold confusables with a table matched to the attacks you expect. And say whether you sanitized, because otherwise results for these attacks measure preprocessing rather than detectors. (7) Study the attacks that survive sanitization. Word-level substitution keeps the meaning, cannot be undone, and costs Binoculars more than any other attack, yet the main benchmarks omit it.

## 6 Conclusion

Our results show that current AI text detectors are systematically fragile. Simple character-level edits, paraphrasing, and short text lengths distort both human and machine classes, producing misleading AUROC gains while inflating false-positive rates. Replication across RAID and DetectRL confirms that these weaknesses are structural rather than dataset-specific. Stronger backbones can make a detector more confident but more wrong, and defenses such as paraphrasing overwrite authorship signals rather than preserving them.

These findings highlight the need for transparent evaluation protocols. Robustness must be measured two-sided, with attacked-human false positives reported alongside attacked-AI true positives, and results disaggregated by generator, text length, and detector family. While sanitization can reverse the Unicode attacks, meaning-preserving substitutions remain a critical open challenge. In sum, AI text detection is not yet reliable for high-stakes use, and progress will depend on careful reporting and a focus on attacks that survive preprocessing.

## 7 Limitations

Our corpora are English-only and were collected in 2024, so newer generators may shift the absolute numbers, though the two-sided result reproduces across both. We test post-hoc text attacks, not adversarial prompting (Tufts et al., 2025) or finetuned evasion, and rewriting defenses, not retrieval (Krishna et al., 2023) or watermarking. Some DetectRL machine documents open with a chat-style introduction, mostly from one generator. They are easy to detect. But human text never opens that way, so the false-accusation results hold.

## 8 Ethical Considerations

This study relies exclusively on datasets that are publicly available, ensuring compliance with accepted research standards. No personally identifiable information (PII) was collected, stored, or analyzed at any stage of the work. All experiments were conducted on text corpora drawn from open sources, and no private or sensitive data were involved.

The findings highlight limitations in current AI text detectors, but do not introduce new risks. We emphasize that any models, code, or evaluation protocols released should be applied responsibly. In particular, detectors should not be used for punitive or surveillance purposes against individuals, and their deployment must align with responsible AI guidelines. Researchers and practitioners are encouraged to avoid harmful or misleading applications and to ensure that these tools are used only in socially beneficial contexts.

## References

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2024. Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In Proceedings of ICLR.

Aldan Creo and Shushanta Pudasaini. 2025. Silver-Speak: Evading AI-generated text detectors using homoglyphs. In Proceedings of the 1st Workshop on GenAI Content Detection (GenAIDetect), pages 1–46, Abu Dhabi, UAE.

Liam Dugan, Alyssa Hwang, Filip Trhlík, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. 2024. RAID: A shared benchmark for robust evaluation of machinegenerated text detectors. In Proceedings ofACL.

Sebastian Gehrmann, Hendrik Strobelt, and Alexander M Rush. 2019. GLTR: Statistical detection and visualization of generated text. In Proceedings of ACL: System Demonstrations.

Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu. 2023. How close is ChatGPT to human experts? comparison corpus, evaluation, and detection. arXiv preprint arXiv:2301.07597.

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Spotting LLMs with Binoculars: Zero-shot detection of machine-generated text. In Proceedings of ICML.

Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2023. RADAR: Robust AI-text detection via adversarial learning. In Advances in Neural Information Processing Systems.

Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. 2023. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. In Advances in Neural Information Processing Systems.

Lucio La Cava and Andrea Tagarelli. 2026. Luminol-AIDetect: Fast zero-shot machine-generated text detection based on perplexity under text shuffling. arXiv preprint arXiv:2604.25860.

Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, and Xiaoming Liu. 2025. Iron sharpens iron: Defending against attacks in machine-generated text detection with adversarial training. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics.

Weixin Liang, Mert Yuksekgonul, Yining Mao, Eric Wu, and James Zou. 2023. GPT detectors are biased against non-native English writers. Patterns.

Hope McGovern, Rickard Stureborg, Yoshi Suhara, and Dimitris Alikaniotis. 2025. Your large language models are leaving fingerprints. In Proceedings of the 1st Workshop on GenAI Content Detection (GenAIDetect), pages 85–95.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. 2023.

DetectGPT: Zero-shot machine-generated text detection using probability curvature. In Proceedings of ICML.

Suraj Ranganath and Atharv Ramesh. 2026. StealthRL: Reinforcement learning paraphrase attacks for multidetector evasion of AI-text detectors. arXiv preprint arXiv:2602.08934.

Rafael Rivera Soto, Barry Chen, and Nicholas Andrews. 2026. Attacks on machine-text detectors retain stylistic fingerprints. In Proceedings ofICML.

Rafael Alberto Rivera Soto, Barry Y. Chen, and Nicholas Andrews. 2025. Mitigating paraphrase attacks on machine-text detection via paraphrase inversion. In Findings of the Association for Computational Linguistics: ACL 2025, pages 4421–4433.

Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. 2025. Can AI-generated text be reliably detected? Stress testing AI text detectors under various attacks. Transactions on Machine Learning Research.

Shoumik Saha and Soheil Feizi. 2025. Almost AI, almost human: The challenge of detecting AI-polished writing. In Findings of the Association for Computational Linguistics: ACL 2025.

Lekkala Sai Teja, Annepaka Yadagiri, Sangam Sai Anish, Siva Gopala Krishna Nuthakki, and Partha Pakray. 2025. Modeling the attack: Detecting AIgenerated text by quantifying adversarial perturbations. arXiv preprint arXiv:2510.02319.

Andrii Shportko and Inessa Verbitsky. 2025. Paraphrasing attack resilience of various AI-generated text detection methods. In Proceedings of NAACL Student Research Workshop.

Brian Tufts, Xuandong Zhao, and Lei Li. 2025. A practical examination of AI-generated text detectors for large language models. In Findings ofNAACL.

James Wang, Ran Li, Junfeng Yang, and Chengzhi Mao. 2024. RAFT: Realistic attacks to fool text detectors. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing.

Junchao Wu, Runzhe Zhan, Derek F Wong, Shu Yang, Xinyi Yang, Yulin Yuan, and Lidia S Chao. 2024. DetectRL: Benchmarking LLM-generated text detection in real-world scenarios. In Advances in Neural Information Processing Systems, Datasets and Benchmarks.

Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, and Fei Wu. 2025. Trainingfree LLM-generated text detection by mining token probability sequences. In Proceedings ofICLR.

Yiwei Zha, Rui Min, and Shanu Sushmita. 2025. PAD-Ben: A comprehensive benchmark for evaluating AI text detectors against paraphrase attacks. arXiv preprint arXiv:2511.00416.

![](images/a5426d0b1e79f1e642e69db171d3acf0905b1b8e210a18b948206af6d324842c.jpg)  
Figure 6: ∆AUROC vs. ∆TPR@1%FPR for all 99 cells. Points in the shaded quadrant are cells where AUROC reports improvement while deployment-relevant TPR degrades; for Binoculars×paraphrase both deltas are individually significant, with opposite signs.

## A RAID full result grid

Table 7 reports the two-sided AUROC from Section 4.3 where attacked human text scored against attacked AI text. Figure 2 plots the one-sided values for the same grid, with significance markers. Reading them together shows what the one-sided protocol hides. The pairs that look like the largest gains one-sided are among the largest losses twosided.

Table 8 reports detection at the 1% threshold, and Table 9 the false-accusation rate on attacked human text for every detector and attack. Both numbers are needed. Because one of them can look fine while the other is catastrophic. Table 11 breaks the clean AUROC out by generator, which Section 4.8 summarises. Every detector’s best setup is a greedy decode and every worst is a sampled one.

## B DetectRL result grids

Tables 14, 12 and 13 show the repeated experiment results with the three grids on DetectRL. Six of its seven attacks are its own, so the columns do not match the RAID grids. But it shows us the same pattern we found on RAID.

Table 14 has the same layout as Table 7. Every significant one-sided gain sits above a significant two-sided loss, as on RAID. No pair gains significantly two-sided. Table 13 shows that under zerowidth insertion, log-likelihood and log-rank flag every one of the 1,440 attacked human documents.

RADAR flags 37% of them under back-translation, which the standard protocol scores as a significant improvement.

## C DetectRL sampling and a labeling defect

We sample DetectRL’s source corpus rather than its prepared splits. Each source record contains a human document, a machine document, and six attacked variants of each, so every attack is paired with its clean original. We draw 360 human and 180 machine sources per domain, seeded, retaining only complete records with texts of at least 30 words. This matches RAID sample sizes.

Prepared splits show a labeling defect. In one evaluation set, 336 of 1,008 records labelled as human actually contain machine text. The corresponding training set shows the same issue, with 10,847 of 32,574 records mislabeled.

We confirmed this by matching affected records back to the source corpus and by measuring character trigram overlap. Fields designated as human align strongly with human text, while those designated as machine align strongly with machine text, indicating that mislabelling occurred during split construction. The problem is limited to those sets. Our sampler repeats the trigram check each run and rejects samples if it fails.

## D The curated confusables table

The curated table holds 47 mappings onto 34 ASCII targets, drawn from Cyrillic (26 sources), Greek (19) and two others. It covers the characters RAID’s homoglyph attacker substitutes, which are almost entirely Cyrillic and Greek lookalikes of Latin letters. Unicode’s full UTS #39 list has 1,861 ASCII-target mappings by comparison. Table 16 gives both tables’ coverage on each corpus, and Table 17 the per-detector recovery that Table 6 reports as ranges.

One gap is worth recording. RAID replaces capital I with either Greek U+0399 or Cyrillic U+0406, and our table covers only the first. Everything else folds back correctly. So a document with a capital I that drew the Cyrillic variant comes back readable but not byte-for-byte identical. That is why homoglyph documents restore exactly 31% of the time in Table 5 while their trigram similarity reaches 0.988.

<table><tr><td colspan="11">AUROC</td></tr><tr><td>detector</td><td>clean</td><td>protocol</td><td></td><td>paraphr. synonym homogl.</td><td></td><td></td><td>. zero-w. whitesp. up/low</td><td></td><td>ins-par art-del</td><td></td><td></td><td>alt-sp. misspell number</td><td></td></tr><tr><td>Binoculars</td><td>0.85</td><td>1-sided</td><td>0.93*</td><td>0.67</td><td>0.80</td><td>0.99*</td><td>0.82</td><td>0.84</td><td>0.85</td><td>0.84</td><td>0.85</td><td>0.85</td><td>0.85</td></tr><tr><td rowspan="2">FDG-neo</td><td></td><td>2-sided</td><td>0.81†</td><td>0.80†</td><td>0.68†</td><td>0.65†</td><td>0.84</td><td>0.86</td><td>0.85</td><td>0.86</td><td>0.85</td><td>0.85</td><td>0.86</td></tr><tr><td>0.83</td><td>1-sided</td><td>0.91*</td><td>0.58</td><td>0.52</td><td>0.69</td><td>0.77</td><td>0.78</td><td>0.82</td><td>0.80</td><td>0.82</td><td>0.82</td><td>0.80</td></tr><tr><td rowspan="2">FDG-gpt2xl 0.76</td><td></td><td>2-sided</td><td>0.73†</td><td>0.77†</td><td>0.60†</td><td>0.64†</td><td>0.81</td><td>0.82</td><td>0.81</td><td>0.83</td><td>0.83</td><td>0.82</td><td>0.82</td></tr><tr><td></td><td>1-sided</td><td>0.91*</td><td>0.55</td><td>0.81*</td><td>0.44</td><td>0.70</td><td>0.70</td><td>0.78</td><td>0.72</td><td>0.75</td><td>0.74</td><td>0.73</td></tr><tr><td rowspan="2">FDG-gpt2</td><td></td><td>2-sided</td><td>0.74</td><td>0.69†</td><td>0.52†</td><td>0.58†</td><td>0.71†</td><td>0.74</td><td>0.80</td><td>0.76</td><td>0.76</td><td>0.76</td><td>0.75</td></tr><tr><td>0.72</td><td>1-sided</td><td>0.85*</td><td>0.56</td><td>0.96*</td><td>0.48</td><td>0.52</td><td>0.61</td><td>0.72</td><td>0.64</td><td>0.71</td><td>0.69</td><td>0.69</td></tr><tr><td rowspan="2">Lastde</td><td></td><td>2-sided</td><td>0.66†</td><td>0.68†</td><td>0.45†</td><td>0.58†</td><td>0.57†</td><td>0.68†</td><td>0.75</td><td>0.72</td><td>0.72</td><td>0.71</td><td>0.71</td></tr><tr><td>0.84</td><td>1-sided</td><td>0.81</td><td>0.55</td><td>0.97*</td><td>0.90*</td><td>0.86*</td><td>0.69</td><td>0.83</td><td>0.78</td><td>0.82</td><td>0.81</td><td>0.81</td></tr><tr><td rowspan="2">LogLik</td><td></td><td>2-sided</td><td>0.73†</td><td>0.75†</td><td>0.55†</td><td>0.55†</td><td>0.82</td><td>0.81</td><td>0.82</td><td>0.82</td><td>0.83</td><td>0.83</td><td>0.84</td></tr><tr><td>0.75</td><td>1-sided</td><td>0.69</td><td>0.51</td><td>0.86*</td><td>1.00*</td><td>0.68</td><td>0.66</td><td>0.74</td><td>0.70</td><td>0.74</td><td>0.74</td><td>0.74</td></tr><tr><td rowspan="2">LogRank</td><td></td><td>2-sided</td><td>0.67†</td><td>0.73</td><td>0.62†</td><td>0.58†</td><td>0.74</td><td>0.75</td><td>0.74</td><td>0.75</td><td>0.75</td><td>0.75</td><td>0.77</td></tr><tr><td>0.77</td><td>1-sided</td><td>0.71</td><td>0.54</td><td>0.86*</td><td>0.99*</td><td>0.72</td><td>0.68</td><td>0.76</td><td>0.72</td><td>0.76</td><td>0.76</td><td>0.76</td></tr><tr><td rowspan="2">RADAR</td><td></td><td>2-sided</td><td>0.68†</td><td>0.73†</td><td>0.62†</td><td>0.58†</td><td>0.76</td><td>0.76</td><td>0.76</td><td>0.77</td><td>0.77</td><td>0.77</td><td>0.79</td></tr><tr><td>0.87</td><td>1-sided</td><td>0.86</td><td>0.87</td><td>0.91*</td><td>0.96*</td><td>0.86</td><td>0.88</td><td>0.90*</td><td>0.86</td><td>0.87</td><td>0.87</td><td>0.88</td></tr><tr><td rowspan="2">RoBERTa</td><td></td><td>2-sided</td><td>0.79†</td><td>0.85†</td><td>0.57†</td><td>0.57†</td><td>0.87</td><td>0.87</td><td>0.84†</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td></tr><tr><td>0.72</td><td>1-sided</td><td>0.81*</td><td>0.70</td><td>0.08</td><td>0.02</td><td>0.07</td><td>0.67</td><td>0.39</td><td>0.69</td><td>0.72</td><td>0.71</td><td>0.72</td></tr><tr><td rowspan="2"></td><td></td><td>2-sided</td><td>0.65†</td><td>0.71</td><td>0.55†</td><td>0.47†</td><td>0.64†</td><td>0.74</td><td>0.68†</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: AUROC on RAID, every detector against every attack. 1-sided scores attacked AI text against clean human text, which is what the standard protocol reports. 2-sided scores it against attacked human text. <sup>∗</sup> marks a significant increase over clean, <sup>†</sup> a significant decrease. Every starred one-sided gain sits above a daggered two-sided loss, and no pair increases significantly two-sided.
<table><tr><td></td><td colspan="10">TPR@1%FPR – fraction of AI documents detected</td></tr><tr><td>detector</td><td>none</td><td>paraphr.</td><td>synonym</td><td>homogl.</td><td>zero-w.</td><td>whitesp.</td><td>up/low</td><td>ins-par art-del</td><td></td><td>alt-sp.</td><td>misspell</td><td>number</td></tr><tr><td>Binoculars</td><td>0.74</td><td>0.68</td><td>0.30</td><td>0.16</td><td>0.80</td><td>0.62</td><td>0.67</td><td>0.72</td><td>0.67</td><td>0.72</td><td>0.72</td><td>0.71</td></tr><tr><td>FDG-neo</td><td>0.69</td><td>0.59</td><td>0.22</td><td>0.08</td><td>0.06</td><td>0.49</td><td>0.52</td><td>0.67</td><td>0.57</td><td>0.67</td><td>0.65</td><td>0.61</td></tr><tr><td>FDG-gpt2xl</td><td>0.58</td><td>0.53</td><td>0.23</td><td>0.23</td><td>0.03</td><td>0.37</td><td>0.40</td><td>0.57</td><td>0.47</td><td>0.56</td><td>0.53</td><td>0.53</td></tr><tr><td>FDG-gpt2</td><td>0.38</td><td>0.26</td><td>0.11</td><td>0.61</td><td>0.00</td><td>0.09</td><td>0.15</td><td>0.33</td><td>0.20</td><td>0.36</td><td>0.33</td><td>0.34</td></tr><tr><td>Lastde</td><td>0.41</td><td>0.18</td><td>0.09</td><td>0.69</td><td>0.22</td><td>0.37</td><td>0.16</td><td>0.36</td><td>0.27</td><td>0.36</td><td>0.34</td><td>0.31</td></tr><tr><td>LogLik</td><td>0.21</td><td>0.06</td><td>0.09</td><td>0.11</td><td>0.95</td><td>0.14</td><td>0.16</td><td>0.19</td><td>0.17</td><td>0.20</td><td>0.19</td><td>0.16</td></tr><tr><td>LogRank</td><td>0.23</td><td>0.07</td><td>0.09</td><td>0.10</td><td>0.79</td><td>0.16</td><td>0.16</td><td>0.21</td><td>0.18</td><td>0.22</td><td>0.22</td><td>0.19</td></tr><tr><td>RADAR</td><td>0.24</td><td>0.21</td><td>0.16</td><td>0.00</td><td>0.00</td><td>0.13</td><td>0.19</td><td>0.21</td><td>0.19</td><td>0.23</td><td>0.22</td><td>0.23</td></tr><tr><td>RoBERTa</td><td>0.23</td><td>0.19</td><td>0.17</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.11</td><td>0.01</td><td>0.16</td><td>0.21</td><td>0.21</td><td>0.22</td></tr></table>

Table 8: TPR at a 1% false-positive threshold, every detector × every attack. Cells are the fraction of attacked AI documents scoring above the detector’s threshold. Truncation cells are reported separately in Section 4.5.
<table><tr><td></td><td colspan="10">FPR – fraction of human documents wrongly flagged</td></tr><tr><td>detector</td><td>none</td><td>paraphr.</td><td>synonym</td><td>homogl.</td><td>zero-w.</td><td>whitesp. up/low</td><td></td><td>ins-par</td><td>art-del</td><td>alt-sp.</td><td>misspell</td><td>number</td></tr><tr><td>Binoculars</td><td>0.01</td><td>0.14</td><td>0.00</td><td>0.00</td><td>0.62</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.01</td></tr><tr><td>FDG-neo</td><td>0.01</td><td>0.20</td><td>0.00</td><td>0.01</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.01</td></tr><tr><td>FDG-gpt2xl</td><td>0.01</td><td>0.15</td><td>0.00</td><td>0.13</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.01</td></tr><tr><td>FDG-gpt2</td><td>0.01</td><td>0.07</td><td>0.00</td><td>0.70</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td></tr><tr><td>Lastde</td><td>0.01</td><td>0.01</td><td>0.00</td><td>0.71</td><td>0.10</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.01</td><td>0.00</td></tr><tr><td>LogLik</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.98</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.00</td></tr><tr><td>LogRank</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.81</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.00</td></tr><tr><td>RADAR</td><td>0.01</td><td>0.03</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>RoBERTa</td><td>0.01</td><td>0.07</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.01</td><td>0.01</td></tr></table>

Table 9: False-positive rate on attacked human text at the same 1%-FPR thresholds. Cells are the fraction of attacked human documents scoring above the same threshold. The none column is the nominal rate by construction.

## E How the 1% threshold is set

The clean human documents are split in half, a calibration half $H _ { \mathrm { c a l } }$ and a held-out half $H _ { \mathrm { e v a l } }$ , each

<table><tr><td>name</td><td>scoring model(s)</td><td>size</td><td>family</td><td>source</td></tr><tr><td>Binoculars</td><td>Falcon-7B + Falcon-7B-Instruct</td><td>7B×2</td><td>modern zero-shot</td><td>Hans et al. (2024)</td></tr><tr><td>FDG-neo</td><td>Fast-DetectGPT / GPT-Neo</td><td>2.7B</td><td>modern zero-shot</td><td>Bao et al. (2024)</td></tr><tr><td>FDG-gpt2xl</td><td>Fast-DetectGPT / GPT-2 XL</td><td>1.5B</td><td>modern zero-shot</td><td>Bao et al. (2024)</td></tr><tr><td>FDG-gpt2</td><td>Fast-DetectGPT / GPT-2</td><td>124M</td><td>modern zero-shot</td><td>Bao et al. (2024)</td></tr><tr><td>Lastde</td><td>Lastde / GPT-Neo</td><td>2.7B</td><td>modern zero-shot</td><td>Xu et al. (2025)</td></tr><tr><td>LogLik</td><td>mean log-likelihood / GPT-Neo</td><td>2.7B</td><td>legacy statistical</td><td>Gehrmann et al. (2019)</td></tr><tr><td>LogRank</td><td>mean log-rank / GPT-Neo</td><td>2.7B</td><td>legacy statistical</td><td>Gehrmann et al. (2019)</td></tr><tr><td>RADAR</td><td>RADAR-Vicuna-7B</td><td>355M</td><td>adversarially trained</td><td>Hu et al. (2023)</td></tr><tr><td>RoBERTa</td><td>HC3 ChatGPT-detector</td><td>125M</td><td>supervised classifier</td><td>Guo et al. (2023)</td></tr></table>

Table 10: The nine detectors. All run from public checkpoints without modification; scores are normalized so that higher means more AI-like.
<table><tr><td>detector</td><td>best generator</td><td>worst generator</td><td>gap</td><td>worst attack</td><td>loss</td></tr><tr><td>Binoculars</td><td>ChatGPT (1.000)</td><td>MPT (0.430)</td><td>0.570</td><td>synonym</td><td>0.181</td></tr><tr><td>FDG-neo</td><td>GPT-4 (1.000)</td><td>MPT (0.425)</td><td>0.575</td><td>homoglyph</td><td>0.304</td></tr><tr><td>FDG-gpt2xl</td><td>GPT-2 (1.000)</td><td>Mistral (0.373)</td><td>0.627</td><td>zero-width</td><td>0.321</td></tr><tr><td>FDG-gpt2</td><td>GPT-3 (0.973)</td><td>Mistral (0.368)</td><td>0.605</td><td>zero-width</td><td>0.248</td></tr><tr><td>Lastde</td><td>GPT-2 (0.999)</td><td>GPT-4 (0.400)</td><td>0.599</td><td>synonym</td><td>0.290</td></tr><tr><td>LogLik</td><td>GPT-2 (0.978)</td><td>MPT (0.334)</td><td>0.644</td><td>synonym</td><td>0.238</td></tr><tr><td>LogRank</td><td>GPT-2 (0.988)</td><td>MPT (0.348)</td><td>0.640</td><td>synonym</td><td>0.236</td></tr><tr><td>RADAR</td><td>GPT-4 (0.971)</td><td>Mistral (0.597)</td><td>0.374</td><td>whitespace</td><td>0.017</td></tr><tr><td>RoBERTa</td><td>Mistral-Chat (0.943)</td><td>MPT (0.321)</td><td>0.622</td><td>zero-width</td><td>0.705</td></tr></table>

Table 11: Generator choice against the worst attack, clean AUROC on RAID. Best and worst are each detector’s highest and lowest AUROC over the 22 generator and decoding setups, and gap is their difference. Every best uses greedy decoding and every worst uses sampling. Loss is the largest drop any attack causes. Bold marks the larger of gap and loss.
<table><tr><td colspan="11">TPR@ 1%FPR – fraction of AI documents detected</td></tr><tr><td>detector</td><td>none</td><td>DIPPER</td><td>back-tr.</td><td>polish</td><td>word</td><td>char</td><td></td><td>homogl.</td><td>zero-w.</td></tr><tr><td>Binoculars</td><td>0.73</td><td>0.78</td><td>0.09</td><td>0.60</td><td>0.21</td><td>0.25</td><td></td><td>0.26</td><td>0.36</td></tr><tr><td>FDG-neo</td><td>0.65</td><td>0.72</td><td>0.07</td><td>0.48</td><td>0.08</td><td>0.15</td><td></td><td>0.11</td><td>0.06</td></tr><tr><td>FDG-gpt2xl</td><td>0.57</td><td>0.58</td><td>0.07</td><td>0.39</td><td>0.05</td><td>0.06</td><td></td><td>0.03</td><td>0.00</td></tr><tr><td>FDG-gpt2</td><td>0.40</td><td>0.37</td><td>0.04</td><td>0.22</td><td>0.02</td><td>0.04</td><td></td><td>0.05</td><td>0.00</td></tr><tr><td>Lastde</td><td>0.56</td><td>0.63</td><td>0.04</td><td>0.31</td><td>0.00</td><td></td><td>0.00</td><td>0.01</td><td>0.70</td></tr><tr><td>LogLik</td><td>0.42</td><td>0.37</td><td>0.01</td><td>0.22</td><td>0.00</td><td></td><td>0.00</td><td>0.00</td><td>1.00</td></tr><tr><td>LogRank</td><td>0.46</td><td>0.45</td><td>0.01</td><td>0.25</td><td>0.00</td><td></td><td>0.00</td><td>0.00</td><td>1.00</td></tr><tr><td>RADAR</td><td>0.41</td><td>0.55</td><td>0.43</td><td>0.33</td><td>0.28</td><td></td><td>0.29</td><td>0.28</td><td>0.02</td></tr><tr><td>RoBERTa</td><td>0.17</td><td>0.17</td><td>0.07</td><td>0.11</td><td>0.02</td><td></td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

Table 12: TPR at a 1% false-positive threshold on DetectRL. Six of the seven attacks are DetectRL’s own; zero-width is ${ \mathrm { R A I D } } ^ { \prime } { \mathrm { s } } ,$ applied by us because DetectRL has no equivalent.

of 720. The threshold is the 99th percentile of the calibration scores,

$$
\tau = Q _ { 0 . 9 9 } ( S ( H _ { \mathrm { c a l } } ) ) ,\tag{4}
$$

and every rate in the paper is the share of a set that scores above it,

$$
\begin{array} { r } { \mathrm { F P R } = \frac { 1 } { | H _ { \mathrm { e v a l } } | } \sum _ { x \in H _ { \mathrm { e v a l } } } { \mathbf 1 } [ S ( x ) > \tau ] , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { \mathrm { T P R } = \frac { 1 } { | M | } \sum _ { x \in M } \mathbf { 1 } [ S ( x ) > \tau ] , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { \mathrm { F P R } _ { a } = \frac { 1 } { | H _ { a } | } \sum _ { x \in H _ { a } } { \mathbf { 1 } } [ S ( x ) > \tau ] . } \end{array}\tag{7}
$$

For Binoculars on RAID, τ = −0.869. It flags 6 of the 720 held-out clean human documents, $\mathrm { F P R } ~ = ~ 0 . 0 0 8$ against the 0.010 asked for. It flags 531 of the 720 clean machine documents, $\mathrm { T P R } = 0 . 7 4$ , the entry in Table 1. Under zerowidth insertion it flags 886 of the 1,440 attacked human documents, $\mathrm { F P R } _ { a } = 0 . 6 2$ , the Binoculars entry in Figure 1. The same τ produced all three. Only the text changed. The bootstrap recomputes τ inside each of its 1,000 replicates.

## F Reproducibility

We will release all scores, sampling manifests, seeds and analysis code. The scores total 724,096, made up of 270,289 RAID, 172,800 on DetectRL and 281,007 re-scores under three sanitizer settings. Bootstrap intervals use 1,000 replicates at seed 42. Compute was about 3 GPU-hours on one H100 for the RAID sweep, 2 for DetectRL, 3 for the sanitized re-scores, and 1.5 for the Falcon-7b backbone runs.

<table><tr><td>detector</td><td>none</td><td>DIPPER</td><td>back-tr.</td><td>polish</td><td>word</td><td>char</td><td>homogl.</td><td>zero-w.</td></tr><tr><td>Binoculars</td><td>0.01</td><td>0.08</td><td>0.00</td><td>0.14</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.31</td></tr><tr><td>FDG-neo</td><td>0.01</td><td>0.14</td><td>0.00</td><td>0.13</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td></tr><tr><td>FDG-gpt2xl</td><td>0.01</td><td>0.07</td><td>0.00</td><td>0.12</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>FDG-gpt2</td><td>0.01</td><td>0.04</td><td>0.00</td><td>0.07</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td></tr><tr><td>Lastde</td><td>0.01</td><td>0.12</td><td>0.00</td><td>0.09</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.68</td></tr><tr><td>LogLik</td><td>0.01</td><td>0.02</td><td>0.00</td><td>0.06</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td></tr><tr><td>LogRank</td><td>0.01</td><td>0.02</td><td>0.00</td><td>0.05</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td></tr><tr><td>RADAR</td><td>0.01</td><td>0.12</td><td>0.37</td><td>0.09</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.02</td></tr><tr><td>RoBERTa</td><td>0.01</td><td>0.01</td><td>0.00</td><td>0.05</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

Table 13: False-positive rate on attacked human text in DetectRL, at the same 1%-FPR thresholds. Log-likelihood and log-rank flag every one of the 1,440 human documents under zero-width insertion.
<table><tr><td rowspan="2"></td><td colspan="9">AUROC</td></tr><tr><td>clean</td><td>protocol DIPPER back-tr. polish</td><td></td><td></td><td></td><td>word</td><td>char</td><td>homogl. zero-w.</td><td></td></tr><tr><td>detector Binoculars</td><td>0.96</td><td>1-sided</td><td>0.97*</td><td>0.60</td><td>0.86</td><td>0.73</td><td>0.77</td><td>0.81</td><td>0.99*</td></tr><tr><td rowspan="3">FDG-neo</td><td></td><td>2-sided</td><td>0.91†</td><td>0.76†</td><td>0.75†</td><td>0.86†</td><td>0.88†</td><td>0.89†</td><td>0.55†</td></tr><tr><td>0.90</td><td>1-sided</td><td> $0 . 9 6 ^ { * }$ </td><td>0.46</td><td>0.79</td><td>0.51</td><td>0.55</td><td>0.54</td><td>0.67</td></tr><tr><td></td><td>2-sided</td><td> $0 . 8 7 ^ { \dagger }$ </td><td>0.76†</td><td>0.70†</td><td>0.83† 0.80†</td><td></td><td>0.83†</td><td>0.66†</td></tr><tr><td rowspan="2">FDG-gpt2x1 0.88</td><td></td><td>1-sided</td><td> $0 . 9 3 ^ { * }$ </td><td>0.54</td><td>0.77</td><td>0.50</td><td>0.47</td><td>0.43</td><td>0.45</td></tr><tr><td></td><td>2-sided</td><td> $0 . 8 4 ^ { \dagger }$ </td><td>0.81†</td><td>0.68†</td><td></td><td>0.84† 0.77†</td><td>0.80†</td><td>0.63†</td></tr><tr><td rowspan="2">FDG-gpt2</td><td>0.83</td><td>1-sided</td><td>0.86*</td><td>0.45</td><td>0.69</td><td>0.42</td><td>0.44</td><td>0.47</td><td>0.41</td></tr><tr><td></td><td>2-sided</td><td>0.76†</td><td>0.76†</td><td>0.63†</td><td>0.78†</td><td>0.64†</td><td>0.70†</td><td>0.39†</td></tr><tr><td rowspan="2">Lastde</td><td>0.90</td><td>1-sided</td><td>0.94*</td><td>0.41</td><td>0.77</td><td>0.20</td><td>0.29</td><td>0.33</td><td>0.97*</td></tr><tr><td></td><td>2-sided</td><td>0.84†</td><td>0.73†</td><td>0.67†</td><td>0.78†</td><td>0.77†</td><td>0.80†</td><td>0.57†</td></tr><tr><td rowspan="2">LogLik</td><td>0.91</td><td>1-sided</td><td>0.92</td><td>0.25</td><td>0.81</td><td>0.42</td><td>0.32</td><td>0.42</td><td>1.00*</td></tr><tr><td></td><td>2-sided</td><td> $0 . 8 6 ^ { \dagger }$ </td><td>0.85†</td><td>0.68†</td><td>0.90</td><td>0.88†</td><td>0.89</td><td>0.71†</td></tr><tr><td rowspan="2">LogRank</td><td>0.91</td><td>1-sided</td><td>0.92</td><td>0.27</td><td>0.80</td><td>0.46</td><td>0.33</td><td>0.45</td><td>1.00*</td></tr><tr><td></td><td>2-sided</td><td> $0 . 8 6 ^ { \dagger }$ </td><td>0.84†</td><td>0.69†</td><td>0.90</td><td>0.88†</td><td>0.89</td><td>0.70†</td></tr><tr><td rowspan="2">RADAR</td><td>0.92</td><td>1-sided</td><td> $0 . 9 5 ^ { * }$ </td><td>0.96*</td><td>0.90</td><td>0.89</td><td>0.93</td><td>0.92</td><td>0.97*</td></tr><tr><td></td><td>2-sided</td><td> $0 . 8 3 ^ { \dagger }$ </td><td>0.55†</td><td>0.77†</td><td>0.88†</td><td>0.91</td><td>0.91</td><td>0.48†</td></tr><tr><td rowspan="2">RoBERTa</td><td>0.84</td><td>1-sided</td><td> $0 . 8 6 ^ { * }$ </td><td>0.79</td><td>0.77</td><td>0.69</td><td>0.64</td><td>0.63</td><td>0.08</td></tr><tr><td></td><td>2-sided</td><td> $0 . 7 7 ^ { \dagger }$ </td><td>0.75†</td><td>0.58†</td><td>0.83</td><td>0.80</td><td>0.80</td><td>0.50†</td></tr></table>

Table 14: AUROC on DetectRL, every detector against every attack. 1-sided scores attacked AI text against clean human text. 2-sided scores it against attacked human text. <sup>∗</sup> marks a significant increase over clean, <sup>†</sup> a significant decrease. No cell increases significantly two-sided.

<table><tr><td>attack</td><td>semantic sim.</td><td>positions changed</td></tr><tr><td>synonym</td><td>0.917</td><td>20.9%</td></tr><tr><td>paraphr.</td><td>0.874</td><td>96.1%</td></tr><tr><td>art-del</td><td>0.991</td><td>81.6%</td></tr><tr><td>up/low</td><td>1.000</td><td>4.7%</td></tr><tr><td>ins-par</td><td>1.000</td><td>1.5%</td></tr><tr><td>number</td><td>0.997</td><td>1.1%</td></tr><tr><td>misspell</td><td>0.994</td><td>1.0%</td></tr><tr><td>alt-sp.</td><td>0.994</td><td>0.7%</td></tr></table>

Table 15: Semantic meaning preserved by word-level attacks determined by cosine similarity of the attacked text and its source, over 2,160 documents per attack. Positions changed is the share of word positions that differ from the source. Unicode and whitespace attacks are omitted since their scores depend on the tokenizer, not on meaning (Table 5).

<table><tr><td>corpus</td><td>table</td><td>mappings</td><td>reversed</td><td>clean rewritten</td></tr><tr><td>RAID</td><td>curated</td><td>47</td><td>99.3%</td><td>60 (2.8%)</td></tr><tr><td>RAID</td><td>full</td><td>1,861</td><td>99.9%</td><td>330 (15.3%)</td></tr><tr><td>DetectRL</td><td>curated</td><td>47</td><td>61.8%</td><td>17 (0.8%)</td></tr><tr><td>DetectRL</td><td>full</td><td>1,861</td><td>98.0%</td><td>44 (2.0%)</td></tr></table>

Table 16: The curated table against the full UTS #39 list. Reversed is the percentage of the attack’s substituted characters turned back into the original letter. Clean rewritten is how many of the 2,160 unattacked source documents the table alters.

![](images/c120155453e1917c21df6c16dbdc4327ec1b615a69acd80d62edb1e58163221f.jpg)

![](images/2dd7b37fd1f922b09f9abe050b52281ddbc8088be7c353e1a1d947424467e898.jpg)  
Figure 7: Controlled truncation of identical sources. Left: detection at 1%-FPR thresholds. Right: false-positive rate on human text at the same thresholds. RADAR’s apparent short-text gain is length bias: it flags 61% of 50-word human texts.

![](images/514e1421382dad3089313dbe5f089afdf00b759f679d55798029dad9f156a318.jpg)  
Figure 8: The measurement pipeline. (a) We sample RAID by domain and score every text with all 9 detectors (Similar approach for DetectRL). (b) Clean human documents are split in half. One half fits each detector’s threshold τ at a nominal 1% false-positive rate, the other verifies it works. Attacked text never enters calibration, so the attacked boxes carry the full n. Standard evaluations score only the two AI branches. The highlighted branch is what this study adds, applying the same 11 attacks to human documents and counting who gets accused.

<table><tr><td>detector</td><td colspan="3">zero-width</td><td colspan="3">homoglyph</td><td colspan="3">whitespace</td></tr><tr><td></td><td>curated</td><td>full</td><td>cur.+ws</td><td>curated</td><td>full</td><td>cur.+ws</td><td>curated</td><td>full</td><td>cur.+ws</td></tr><tr><td>Binoculars</td><td>100</td><td>100†</td><td>100</td><td>97</td><td>98†</td><td>97</td><td>84</td><td>100†</td><td>99</td></tr><tr><td>FDG-neo</td><td>100</td><td>100</td><td>100</td><td>95</td><td>95</td><td>95</td><td>70</td><td>99</td><td>99</td></tr><tr><td>FDG-gpt2xl</td><td>100</td><td>100†</td><td>99</td><td>92</td><td>92†</td><td>91</td><td>64</td><td>99†</td><td>99</td></tr><tr><td>FDG-gpt2</td><td>100</td><td>100†</td><td>100</td><td>90</td><td>93†</td><td>90</td><td>23</td><td>99†</td><td>99</td></tr><tr><td>Lastde</td><td>100</td><td>100</td><td>100</td><td>87</td><td>80</td><td>88</td><td>92</td><td>99</td><td>99</td></tr><tr><td>LogLik</td><td>100</td><td>100</td><td>100</td><td>93</td><td>86</td><td>92</td><td>67</td><td>100</td><td>100</td></tr><tr><td>LogRank</td><td>100</td><td>100</td><td>100</td><td>86</td><td>82</td><td>86</td><td>67</td><td>100</td><td>100</td></tr><tr><td>RADAR</td><td>100</td><td>100</td><td>99</td><td>104</td><td>105</td><td>103</td><td>56</td><td>101</td><td>100</td></tr><tr><td>RoBERTa</td><td>100</td><td>102†</td><td>100†</td><td></td><td>94 81†</td><td>94†</td><td>0</td><td>92†</td><td>90†</td></tr></table>

Table 17: Detection after each sanitizer configuration, per detector, as a percentage of clean detection at the 1% threshold. Table 6 gives the range of each column. <sup>†</sup> marks a configuration that raises the detector’s FPR on clean human text.

<table><tr><td colspan="5">target ← source code points</td></tr><tr><td></td><td>A U+0391, U+0410</td><td>S U+0405</td><td></td><td>k U+03BA</td></tr><tr><td></td><td>B U+0392, U+0412</td><td></td><td>T U+03A4, U+0422 1 U+217C</td><td></td></tr><tr><td></td><td>C U+0421</td><td></td><td></td><td>X U+03A7, U+0425o U+03BF, U+043E</td></tr><tr><td></td><td>E U+0395, U+0415</td><td>Y U+03A5</td><td></td><td>p U+03C1, U+0440</td></tr><tr><td></td><td>HU+0397, U+041D</td><td>Z U+0396</td><td></td><td>q U+051B</td></tr><tr><td></td><td>I U+0399</td><td></td><td>a U+03B1, U+0430 s U+0455</td><td></td></tr><tr><td></td><td>J U+0408</td><td>c U+0441</td><td></td><td>ν U+03BD</td></tr><tr><td></td><td>K U+039A, U+041A d U+0501</td><td></td><td></td><td>w U+0461</td></tr><tr><td></td><td>M U+039C, U+041C e U+0435</td><td></td><td></td><td>x U+0445</td></tr><tr><td></td><td>N U+039D</td><td>g U+0261</td><td></td><td>y U+0443</td></tr><tr><td></td><td>0 U+039F, U+041E</td><td>i U+0456</td><td></td><td></td></tr><tr><td></td><td>PU+03A1, U+0420</td><td> j U+0458</td><td></td><td></td></tr></table>

Table 18: The 47 curated confusable mappings. Each row gives an ASCII target and the code points folded onto it. Applied after zero-width stripping and before NFKC.