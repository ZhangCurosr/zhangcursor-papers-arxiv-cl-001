# Does a Norwegian further-trained model give more outdated legal values than its base model? A preregistered paired comparison of Borealis 2 and Gemma 4

Eirik Botten Nicolaysen EcoDeco AS ORCID 0009-0001-9188-6788 eirik@ecodeco.no

## Abstract

We compare how often two released language models give outdated numeric values from Norwegian statutes and regulations: NbAiLab/borealis2-26b-a4b-preview (Borealis 2), which the National Library of Norway further trained from google/gemma-4-26B-A4B-it (Gemma 4), and Gemma 4 itself. The study was preregistered (doi:10.5281/zenodo.23858965) and uses the closed norregel benchmark, scored deterministically against a value history. On the 411 of 503 primary items where both models gave an interpretable answer, Borealis 2 alone was outdated on 10 items and Gemma 4 alone on 11 (d = −0.2 percentage points, 95% cluster bootstrap interval [−2.7, +2.5], exact one-sided McNemar p = 0.668). By the preregistered rule this is no efect of interest: the whole interval lies within the smallest efect of interest of ±10 percentage points. Among the secondary measures, only the share of answers giving a value that took efect after the question date difers after Holm correction; Gemma 4 gave such answers more often. An exploratory comparison of two smaller Borealis models had no discordant pairs and is not informative. The design compares released models and does not isolate the efect of any part of their training.

## 1 Introduction

The Simula CAISS audit of Borealis 1, which is built on Gemma 3, points to outdated legal values without measuring them [9]. On printed p. 11 (PDF page 12) of the report dated 25 June 2026 it states: “Part of the advantage is parametric knowledge of Norwegian rules and thresholds, which drift as the rules change; the rubric scores confident correct recall as a pass, yet the same behaviour becomes the confident-hallucination failure mode 3 (§4.1) once a recalled fact is stale, so the comparative ranking is a snapshot.” The present study measures one part of this for Borealis 2: on the same dated questions, does the further-trained model give outdated values more often than the model it is built on?

The study is descriptive. Borealis 2 and Gemma 4 difer in the National Library’s continued pre-training and fine-tuning, of which newspaper text is one of many changes, and each model comes from a single training run. A diference between them could therefore not be attributed to newspaper text or to any other part of the training (preregistration §2, §14).

## 2 Related work

The preregistration reports a scoping search that included 90 works; its limits (a criterion added after the search had started, a stopping rule that was not met, and candidates that were not screened) are stated there [4]. The five closest works are FiscalQA Pro, French tax law with deterministic scoring and numeric tolerance [6]; Prior, Schultz and Grabmair, German law, with the failure modes post-cutof staleness and recency bias, scored by an LLM judge [17]; Feng, numeric state-policy values with a same-state stale error class [8]; TIDE, the value in force on a date, for tarif rates [18]; and evolveQA, evolving knowledge [13]. This study difers from these by using Norwegian statutes and regulations from Lovdata, deterministic scoring against a value history [1], and a paired comparison of a nationally further-trained model with its base model. It makes no claim to be the first such comparison.

Table 1: Models and pinned revisions.
<table><tr><td>Role</td><td>Hugging Face ID</td><td>Revision</td></tr><tr><td>(a)</td><td>NbAiLab/borealis2-26b-a4b-preview</td><td>6f14c8f8913a9d751b80eee1c4a72d2ca4bfaa98</td></tr><tr><td>(a)</td><td>google/gemma-4-26B-A4B-it</td><td>4d7ae4984b7db7de8f8457170b3f1a419ee76d52</td></tr><tr><td>(b)</td><td>NbAiLab/borealis-4b</td><td>939cf2ce2a5036151e719979f9b10fdc3d4b5ece</td></tr><tr><td>(b)</td><td>NbAiLab/borealis-open-4b</td><td>9445eb2acee3dd7adf7124781f06962ccd47c870</td></tr></table>

## 3 Method

Preregistration. The study was registered on Zenodo (doi:10.5281/zenodo.23858965, published 2026-11-01T13:52:48Z). No norregel question was run on the models before the registration; the embargo was lifted at 2026-11-01T13:53:05Z after the publication was confirmed. The analysis code, its lock file, the run script and the item manifest were frozen in the preregistration.

Hypotheses. H-a (confirmatory): among interpretable answers, the share of outdated answers is higher for Borealis 2 than for Gemma 4 on the same items. H-b (exploratory): the share is higher for NbAiLab/borealis-4b than for NbAiLab/borealis-open-4b, two Borealis 1 models whose fine-tuning data difer, according to the model card, by tasks derived from newspaper material [14]. H-b enters the confirmatory analysis only as secondary measure S-5.

Models and run. Table 1 lists the four models with their pinned revisions [10, 14–16]. Each model ran on Modal on one A100-80GB GPU in bf16 with vLLM 0.21.0 and transformers 5.8.1, greedy decoding (temperature 0.0, seed 20261117, which has no efect at temperature 0.0), a maximum of 1,024 new tokens, reasoning of, and the question as the only user message, without a system prompt. Each container checked the revision, the vLLM version and the chat template before answering. Answers were written as a hash chain in chunks of at most 60 questions; the chain heads were stamped with OpenTimestamps. Borealis 2 ran from 2026-11-01T14:01:39Z to 2026-11-01T15:51:50Z, Gemma 4 from 2026-11-01T16:00:01Z to 2026-11-01T18:32:10Z, borealis 4b from 2026-11-01T18:48:32Z to 2026-11-01T19:56:14Z and borealis-open-4b from 2026-11- 01T19:59:21Z to 2026-11-01T20:55:13Z.

Items and reference date. The items, the answer key and the scorer come from norregel [1–3]: 916 dated numeric values from Lovdata’s open data, with reference date 2026-10-16. The items are closed; the manifest holds item IDs only. The primary items are the H1 set of norregel v2 whose answer-key value took efect on or before 2025-01-01, the earlier of the two models stated training cutofs (Borealis 2 2025-01-01, Gemma 4 2025-01-31): 503 items in 89 series (Table 2).

Scoring and analysis. Answers are classed by norregel’s locked scorer v2 as correct, outdated (the answer equals an earlier value in the item’s series but not the answer key), other error, or uninterpreted; uninterpreted is an outcome, never counted as an error. The primary measure is the share of outdated answers among interpretable answers, paired on the items where both models’ answers are interpretable. H-a is tested with an exact one-sided McNemar test [12], $\alpha = 0 . 0 5$ , with $d = ( b _ { 1 0 } - b _ { 0 1 } ) / n$ and a 95% cluster bootstrap interval over series $( B = 1 0 , 0 0 0$ seed 20261117). The decision rule is set in the preregistration (lines 168–170): supported if $p < 0 . 0 5$ and the lower bound is above zero; no efect of interest if the whole interval lies within ±10 percentage points, the smallest efect of interest taken over from norregel $\mathrm { v 2 } ;$ otherwise inconclusive. The planned minimum detectable diference was 4.42 percentage points with cluster adjustment (3.32 without), from a proxy pair in the v2 data. Secondary measures with at least one discordant pair are corrected with Holm’s procedure [11].

Table 2: Item flow from all items to the analysed pairs.
<table><tr><td>Step</td><td>Items</td><td></td><td>Dropped Reason for dropping</td></tr><tr><td>All items, run on every model</td><td>916</td><td></td><td></td></tr><tr><td>Primary set</td><td>880</td><td></td><td>36 in norregel&#x27;s excluded_items, as in v2</td></tr><tr><td>H1 set within the primary set</td><td>567</td><td></td><td>313 not in norregel v2&#x27;s h1_eligible_items</td></tr><tr><td>Primary items (a) and (b)</td><td>503</td><td>64</td><td>key took effect after 2025-01-01</td></tr><tr><td>Pairs, both interpretable, (a)</td><td>411</td><td></td><td>92 uninterpreted: Borealis 2 only 57, Gemma 4 only 33, both 2</td></tr><tr><td>Pairs, both interpretable, (b)</td><td>411</td><td>92</td><td>uninterpreted: borealis-4b only 92, borealis-open-4b only 0, both 0</td></tr></table>

Table 3: Classes on the 503 primary items per model (unpaired). The share of outdated answers uses each model’s interpretable answers as the denominator.
<table><tr><td>Model</td><td>Correct</td><td>Outdated</td><td>Other error</td><td>Uninterpreted</td><td>Outdated / interpretable</td><td></td></tr><tr><td>Borealis 2</td><td>14</td><td>25</td><td>405</td><td>59</td><td></td><td>25 / 444 (5.6%)</td></tr><tr><td>Gemma 4</td><td>23</td><td>26</td><td>419</td><td>35</td><td></td><td>26 / 468 (5.6%)</td></tr><tr><td>borealis-4b</td><td>2</td><td>6</td><td>403</td><td>92</td><td></td><td>6 / 411 (1.5%)</td></tr><tr><td>borealis-open-4b</td><td>2</td><td>6</td><td>495</td><td>0</td><td></td><td>6 / 503 (1.2%)</td></tr></table>

## 4 Results

Completeness. All four models answered all 916 items; every chain is intact, with one revision and one vLLM version per model. Table 2 shows how the 916 items narrow to the analysed pairs. In both pairs 411 items remain, but the items that drop out difer: 17 of them are the same in both pairs.

H-a (confirmatory). Of the 411 pairs, Borealis 2 alone was outdated on $b _ { 1 0 } = 1 0$ items and Gemma 4 alone on $b _ { 0 1 } = 1 1$ . The diference is $d = - 0 . 2$ percentage points, 95% cluster bootstrap interval [−2.7, +2.5], and the exact one-sided McNemar test gives $p = 0 . 6 6 8$ . The p-value is not below 0.05, the lower bound is not above zero, and the whole interval lies within ±10 percentage points. By the preregistered rule the result is no $e f f e c t$ of interest. No other reading of the interval against the smallest efect of interest is made.

Per model. Table 3 gives the classes on the primary items for each model, unpaired.

Secondary measures. Table 4 gives the secondary measures. After Holm correction over the 5 testable measures, the null hypothesis is rejected only for S-2 (later value: a value that took efect after the question date): Gemma 4 alone gave such a value on 31 items and Borealis 2 alone on 4 $( d = - 5 . 4$ percentage points, two-sided $p < \mathrm { 0 . 0 0 0 1 } )$ . The study does not identify a cause for this diference.

Table 4: Secondary measures, paired on the primary items. Two-sided exact McNemar; Holm over the 5 measures with at least one discordant pair. $b _ { 1 0 } \colon$ : first model only; b<sub>01</sub>: second model only. S-5 compares borealis-4b with borealis-open-4b; the others Borealis 2 with Gemma 4.
<table><tr><td>Measure</td><td>n</td><td> $b _ { 1 0 }$ </td><td> $b _ { 0 1 }$ </td><td>d (pp)</td><td>p</td><td>Holm</td></tr><tr><td>S-1 outdated, all answers</td><td>503</td><td>10</td><td>11</td><td>-0.2</td><td>1.0000</td><td>not rejected</td></tr><tr><td>S-2 later value</td><td>503</td><td>4</td><td>31</td><td>-5.4</td><td>&lt;0.0001</td><td>rejected</td></tr><tr><td>S-3 correct</td><td>503</td><td>4</td><td>13</td><td>-1.8</td><td>0.0490</td><td>not rejected</td></tr><tr><td>S-4 uninterpreted</td><td>503</td><td>57</td><td>33</td><td>+4.8</td><td>0.0149</td><td>not rejected</td></tr><tr><td>S-6 outdated = most recent differing value</td><td>411</td><td>2</td><td>10</td><td>-1.9</td><td>0.0386</td><td>not rejected</td></tr><tr><td>S-5 (b) outdated, both interpretable</td><td>411</td><td>0</td><td>0</td><td>+0.0</td><td></td><td>not testable</td></tr></table>

Table 5: Uninterpreted answers per model, on the primary items and on all items, with subclasses.
<table><tr><td>Model</td><td>Primary</td><td>Subclasses</td><td>All 916</td><td>Subclasses</td></tr><tr><td>Borealis 2</td><td>59</td><td>refusal 49, no answer line 9, several numbers 1</td><td>157</td><td>refusal 119, no answer line 34, several numbers 4</td></tr><tr><td>Gemma 4</td><td>35</td><td>no answer line 26, refusal 9</td><td>131</td><td>refusal 73, no answer line 58</td></tr><tr><td>borealis-4b</td><td>92</td><td>no answer line 92</td><td>159</td><td>no answer line 159</td></tr><tr><td>borealis-open-4b</td><td>0</td><td></td><td>5</td><td>no answer line 5</td></tr></table>

Uninterpreted answers. Table 5 gives the uninterpreted answers. “No answer line” means that the locked scorer found no answer line of the required form, which requires the label at the start of a line. Over all 916 items, Borealis 2 has 34 such answers: 21 end with the template placeholder SVAR: <verdi> or an empty SVAR:, 0 contain SVAR: in another position, and 13 contain no SVAR:. Gemma 4 has 58 (17, 0 and 41), borealis-4b 159 (1, 148 and 10) and borealis-open-4b 5 (0, 5 and 0). The placeholder answers are missing answers; the answers with SVAR: in another position are a format outcome of the locked scorer. Neither group is re-scored.

(b), exploratory. Within the registered analysis, H-b enters as S-5, which has no discordant pairs and cannot be tested (Table 4). An exploratory copy of the analysis, with only the test label changed, gives the same picture for the primary measure: n = 411 and 0 discordant pairs. Both models repeated their output heavily: borealis-4b 264 distinct completions over 916 items, the two most frequent occurring 217 and 138 times, and borealis-open-4b 41 distinct completions, the two most frequent occurring 222 and 184 times. With no discordant pairs and such output, (b) is not informative about H-b. Whether the training of the two models was otherwise identical is not known.

## 5 Limitations

The design does not isolate newspaper text or any other part of the training: (a) compares two released models that difer in continued pre-training and fine-tuning, each from one training run. Borealis 2 is a preview model. Greedy decoding on a GPU does not guarantee identical answers across runs. The maximum of 1,024 tokens is lower than the 16,384 used for API models in v2; 1 Borealis 2 answer stopped at the maximum. Some answers in the “no answer line” subclass are template placeholders, that is, missing answers, and they reduce the number of analysed pairs. An outdated answer cannot be told apart from a correct value given for the wrong date, and the answer key has residual errors (norregel v2). The training cutofs are those stated by the providers; for Borealis 2 no separate cutof is stated for its fine-tuning data or its newspaper data. The Borealis 2 licence text calls the model a derivative of Gemma 3, while its base is Gemma 4.

The planned minimum detectable diference rests on a proxy pair from the v2 data and was computed for a two-sided test with a normal approximation, whereas the primary test is exact and one-sided; the power of the exact one-sided test was not computed.

## 6 Relation to norregel v2

Gemma 4 was measured in norregel v2 through Google’s API, with thinking on and a maximum of 16,384 tokens; 349 of its 916 answers there were empty because they stopped at the maximum. Here it ran locally in bf16 with thinking of and a maximum of 1,024 tokens. The numbers here are therefore not a replication of v2, and diferences between v2 and this study are not interpreted.

## 7 Deviations

The deviations during the run are listed in the result report [5]; the full deviation log with timestamps is kept in the author’s register and is not deposited. In short: D-1, the operator script that drove the chunks labels a stop at the cost cap “rule 7”, whereas the preregistration places the cost cap under rule 6; no stop occurred. D-2, the (b) models were run with a second copy of the run script, which checks the Gemma 3 chat format taken from chat\_template.jinja at the pinned revisions instead of the model card, after the first copy failed the template check before any norregel question was run. D-3, the exploratory (b) analysis uses a copy of the analysis code with only the test label changed. D-4, the cost cap was set before the run and enforced before every chunk, but was entered in the deviation log only after the run. D-5, during the run the hash chains were held only in a non-persistent location until the author’s instruction to copy them after each model; the copies are byte-identical with the analysed chains. D-6 to D-8 concern cost bookkeeping, the descriptive scripts and Modal’s run history. None changes a locked file. Of the deviations recorded before the run, those that bear on it (A-20 to A-24 in the author’s register) are described in the preregistration (§3, §8, §15).

## 8 Data and code availability

The preregistration with the frozen files is at doi:10.5281/zenodo.23858965 [4]. The results record [5], reserved DOI 10.5281/zenodo.23870450, holds the result report, the analysis outputs, the item flow, the descriptive supplement, the four chain heads with their OpenTimestamps stamp (confirmed in Bitcoin block 974179) and the code that is not frozen: the operator script, the (b) run-script copies, the (b) analysis copy and the descriptive and report scripts. The frozen run script and analysis code are in the preregistration. Raw model answers are not published. The NB-License of Borealis 2 and borealis-4b ties every use of model output to its use restrictions, and the Gemma Terms of Use of borealis-open-4b make the user responsible for outputs and their use; the Gemma 4 answers (Apache 2.0) are withheld as well, because the pairs are analysed together. The items are closed. The norregel code, the tests that run without the items and the published documentation are also at https://github.com/avalyset/norregel.

Source attribution. Inneholder data under Norsk lisens for ofentlige data (NLOD) tilgjengeliggjort av Stiftelsen Lovdata. (Contains data under the Norwegian Licence for Open Government Data (NLOD) made available by Stiftelsen Lovdata.) Licence: NLOD 2.0, https: //data.norge.no/nlod/no/2.0. Source: https://api.lovdata.no/. The items were derived from these data by norregel (extraction, chaining, question templates); these changes were made by the author and are not Lovdata’s responsibility. [7]

## AI disclosure

Claude (Anthropic), including Claude Code, was used as a tool under the author’s direction for study-design support, the literature search, drafting the preregistration, the report and this paper, operating the model runs and running the locked analysis; independent Claude subagents checked the documents. The author is responsible for the content.

## Competing interests

The author is the founder and CEO of EcoDeco AS, which ofers independent evaluation of language models. EcoDeco AS received no payment for this study.

## Funding

This work received no external funding.

## Acknowledgements

GPU compute was paid with Modal’s free monthly credits (Starter plan).

## References

[1] Eirik Botten Nicolaysen. norregel: a preregistered benchmark of dated numeric values in Norwegian law. arXiv:2610.29207, 2026. URL https://arxiv.org/abs/2610.29207.

[2] Eirik Botten Nicolaysen. norregel: Preregistration and results of a closed benchmark for numeric values in Norwegian statutes, regulations and parliamentary decisions (v2). Zenodo, concept DOI, doi:10.5281/zenodo.23422302, 2026. URL https://doi.org/10.5281/zenodo. 23422302.

[3] Eirik Botten Nicolaysen. norregel, Zenodo version 2.0. Zenodo, doi:10.5281/zenodo.23437444, 2026. URL https://doi.org/10.5281/zenodo.23437444.

[4] Eirik Botten Nicolaysen. norregel × Borealis: Preregistration of a paired comparison of outdated numeric values from Norwegian statutes and regulations in Borealis 2 and Gemma 4. Zenodo, doi:10.5281/zenodo.23858965, 2026. URL https://doi.org/10.5281/zenodo.23858965.

[5] Eirik Botten Nicolaysen. norregel × Borealis: Results of a preregistered paired comparison of outdated numeric values from Norwegian statutes and regulations in Borealis 2 and Gemma 4. Zenodo, reserved doi:10.5281/zenodo.23870450, 2026. URL https://doi.org/10. 5281/zenodo.23870450.

[6] Rose Cymbler, Daniel Guez, and Laurent Fabre. Temporal misgrounding in legal RAG: A versioned-corpus benchmark for French tax law. arXiv:2608.09393, 2026. URL https: //arxiv.org/abs/2608.09393.

[7] Digitaliseringsdirektoratet. Norsk lisens for ofentlige data (NLOD) 2.0. data.norge.no, 2023. URL https://data.norge.no/nlod/no/2.0. Last updated 2023-03-07.

[8] Jiayu Feng. Right number, wrong state? Measuring cross-jurisdiction substitution in LLM recall of state policy. arXiv:2610.09458, 2026. URL https://arxiv.org/abs/2610.09458.

[9] Sushant Gautam, Leon Moonen, Michael A. Riegler, Jostein Jensen, Annika Willoch Olstad, Lillian Røstad, Olav Lysne, Klas Pettersen, and Finn Schwall. Auditing the Borealis Norwegian language models. Living report, version v2026-06-25, CAISS, Simula Research Laboratory, 2026. URL https://doi.org/10.5281/zenodo.20840166. doi:10.5281/zenodo.20840166.

[10] Google. Model card: google/gemma-4-26B-A4B-it. Hugging Face, 2026. URL https: //huggingface.co/google/gemma-4-26B-A4B-it.

[11] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979. URL https://www.jstor.org/stable/4615733.

[12] Quinn McNemar. Note on the sampling error of the diference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947. doi:10.1007/BF02295996.

[13] Nishanth Sridhar Nakshatri, Shamik Roy, Manoj Ghuhan Arivazhagan, Hanhan Zhou, Vinayshekhar Bannihatti Kumar, and Rashmi Gangadharaiah. When facts change: Probing LLMs on evolving knowledge with evolveQA. arXiv:2510.19172, doi:10.48550/arXiv.2510.19172, 2025. URL https://arxiv.org/abs/2510.19172.

[14] NbAiLab. Model card: NbAiLab/borealis-4b. Hugging Face, 2026. URL https://huggingface. co/NbAiLab/borealis-4b.

[15] NbAiLab. Model card: NbAiLab/borealis-open-4b. Hugging Face, 2026. URL https: //huggingface.co/NbAiLab/borealis-open-4b.

[16] NbAiLab. Model card: NbAiLab/borealis2-26b-a4b-preview. Hugging Face, 2026. URL https://huggingface.co/NbAiLab/borealis2-26b-a4b-preview.

[17] Max Prior, Andreas Schultz, and Matthias Grabmair. Asking for an old friend: Diagnosing and mitigating temporal failure modes in LLM-based statutory question answering. arXiv:2605.23497, 2026. URL https://arxiv.org/abs/2605.23497.

[18] Mahbub E Sobhani, Md. Faiyaz Abdullah Sayeedi, Fahmid Hasan Chowdhury, Md Adnan Arefeen, Farig Sadeque, Md. Faizul Bari, and Swakkhar Shatabda. Time present and time past: Benchmarking large language models on temporally evolving document understanding. arXiv:2608.08512, 2026. URL https://arxiv.org/abs/2608.08512.