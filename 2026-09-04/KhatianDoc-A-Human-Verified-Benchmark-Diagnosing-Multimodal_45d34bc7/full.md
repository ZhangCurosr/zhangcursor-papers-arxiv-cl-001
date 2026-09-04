# KhatianDoc: A Human-Verified Benchmark Diagnosing Multimodal LLM Failure on Bengali Legal Land Records

Tasmiad Hasan Arafat Zaman Ratul Sarker Sadman Saalim

S.M. Shah Nawaz Hossain Khan Raiyan Ibne Reza Sumaiya Tabassum Nimi

North South University, Dhaka, Bangladesh

{tasmiad.hasan, arafat.ratul, sarker.saalim, nawaz.hossain, raiyan.reza, sumaiya.nimi}@northsouth.edu

## Abstract

Land ownership in Bangladesh is recorded in Ana-Ganda-Kora-Kranti-Til, a base-16 positional fraction system with dedicated Unicode glyphs, no mainstream font, and no coverage in any OCR pipeline or tokenizer. The handwritten records that carry these fractions, RS Khatians, are the authoritative title record for millions of parcels and a frequent subject of civil litigation, yet no benchmark has asked whether a machine can read one. We introduce KhatianDoc, a four-task benchmark built from 107 real RS Khatian records from the Vumi (land) Ofice of Munshiganj, Bangladesh: symbol recognition, base-16-to-decimal conversion, structured field extraction, and legal document question answering over 1,634 QA pairs. Ground truth was transcribed by hand, verified by a land-law practitioner to full agreement, and anonymized through positional tokens that keep the referential distinctions multihop questions depend on. We evaluate six mul timodal LLMs (8B to 72B+, open and closed) under a fixed zero-shot protocol. Five QA categories, 39.3% of our stratified set, return zero correct answers from every model; on the arithmetic task, every model that emits a number does worse than a constant-mean baseline, with exact- and near-match scores coinciding: decorrelation, not approximation. Auditing our own metrics surfaced two artifacts in opposite directions: we correct a refusalscoring bug and report the fixed scores beside the originals, and flag an inflated metadata metric as an upper bound. KhatianDoc documents not a performance gap but the absence of a capability, with verified ground truth for future systems. Code and data, with a redacted image release, are publicly available.

Dataset: https://huggingface.co/datasets/ RaiyanKhaan/KhatianDoc

Keywords: Bengali NLP; legal document understanding; low-resource benchmark; vision-

language models; handwritten document OCR;   
land records; base-16 numerals.

## Highlights:

• First benchmark for Bengali RS Khatian records and Ana-Ganda base-16 fractions.

• Six multimodal LLMs score zero on 39.3% of legal QA: a hard reasoning floor.

• Every model underperforms a constant-mean baseline on base-16 arithmetic.

• A metric self-audit fixes a refusal-scoring bug and flags an inflated metric.

• Positional-token anonymization with a redacted public image release.

## 1 Introduction

Land ownership in Bangladesh is defined, in law, by Ana-Ganda-Kora-Kranti-Til, a base-16 positional fraction system written by hand into government survey records called RS Khatians. These records carry direct legal force in property disputes (World Bank, 2024). They are the authoritative record of title for millions of parcels, and yet they sit almost entirely outside modern digital infrastructure. The Ana-Ganda glyphs have no standard font support, no OCR coverage, and no place in any NLP tokenizer vocabulary we are aware of. The legal-NLP community has built strong systems for court judgments and contracts (Chalkidis et al., 2022; Malik et al., 2021), but nothing that has had to cope with all of what a Khatian throws at a reader at once: a degraded handwritten table, a non-decimal positional numeral system, and privacy-sensitive questions about who owns what.

We built KhatianDoc to measure that gap concretely. It draws on 107 real RS Khatian records obtained through the Vumi Ofice of Munshiganj district and organizes them into four tasks of increasing scope: symbol recognition, base-16 arithmetic, structured field extraction, and legal document QA over 1,634 question–answer pairs. Every field of ground truth was written out by hand and then checked, line by line, by a lawyer with land-law experience; the second pass reached complete agreement on both glyph identity and decimal value. When we ran six multimodal LLMs against it under one fixed zero-shot protocol, we did not find a smooth curve of partial competence. We found a floor. Five QA categories, together 39.3% of the evaluation set, return zero correct answers from every model tested, and on the arithmetic task every model that produces a number lands above the error of a baseline that ignores its input entirely and always guesses the dataset mean. Along the way, checking our scoring code against its own outputs surfaced two artifacts (one that inflates a metadata-matching metric through constant and null fields, one that penalizes models for refusing correctly), which we surface and report openly.

![](images/0a8e770f153b426f3be4103695bd0e66234a73781b8ba004f34989a77afe5e0b.jpg)  
Figure 1: Two records from the public KhatianDoc release. Every document is a full-page scan of the same standardized RS form completed by hand. Identity-bearing regions are masked with opaque boxes before release: the owner name and address (column 1), form and khatian numbers, district / upazila / mouza headers, and the oficer’s seal and signature at the bottom. The fields the core tasks score (share column অংশ, revenue রাজē, plot number দাগ নং, land class জিমর েèণী, and the Ana-Ganda area entries) sit outside every masked region and are left intact. The Task 3 metadata header fields that fall under a mask (khatian and form number, district, upazila, mouza) are handled by a separate public-release scoring protocol, defined in §7.

Our contributions are threefold.

• The benchmark. KhatianDoc is, to our knowledge, the first benchmark for Bengali RS Khatian records, and it contains the first machinecheckable, legally verified encoding of the Ana-Ganda base-16 fraction system (§3).

• The diagnosis. A zero-shot evaluation of six frontier multimodal LLMs documents universal failure on complex legal reasoning and sub-baseline performance on non-decimal arithmetic (§5, §6).

• The methodology. We report transferable lessons: a metric self-audit that corrects two evaluation artifacts, a privacy-preserving positional-token pipeline that keeps multi-hop QA answerable, and a redacted-image release policy for records that name real people (§3, §6, §7).

Section 2 places KhatianDoc against prior work;   
§3 describes how it was built; §4 fixes the protocol;   
and §5–6 report results and failure analysis.

## 2 Related Work

Document IE and legal DocVQA. Documentunderstanding models (Xu et al., 2020; Huang et al., 2022; Kim et al., 2022; Mathew et al., 2021) and the corpora they are trained on (Jaume et al., 2019; Huang et al., 2019; Park et al., 2019; Pfitzmann et al., 2022) overwhelmingly assume typed, Latin-script forms. Legal-domain benchmarks (Chalkidis et al., 2022; Hendrycks et al., 2021a) work over clean digital prose. None has met a document whose single most important numeric field is written in a base-16 positional script with no font and no OCR support, exactly the setting KhatianDoc’s Tasks 1 and 3 probe.

Multilingual and low-resource legal NLP. ILDC (Malik et al., 2021), MultiEURLEX (Chalkidis et al., 2021), IndicBERT (Doddapaneni et al., 2023), and BanglaBERT (Bhattacharjee et al., 2022) all target typed, modern prose in base-10. RS Khatians are handwritten, heavily abbreviated, tabular, and carry their legally binding values in a numeral system absent from every tokenizer vocabulary we checked. The gap here is not one of language coverage but of script- and numeral-system-level invisibility.

Quantitative and symbolic reasoning. GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), MathVista (Lu et al., 2024), NumGLUE (Mishra et al., 2022), FinQA (Chen et al., 2021), and LegalBench (Guha et al., 2023) test arithmetic exclusively in base-10. Khatian-Doc’s Task 2 asks for something with a diferent algebraic shape (a positional base-16 fraction with four nested subunits) that no reasoning benchmark we know of instantiates. The sub-baseline results in §5 suggest that it is this structural novelty, not mere dificulty, that drives the failure.

Privacy-preserving legal AI. Standard deidentification collapses every name onto a single generic tag (Lison et al., 2021; Sweeney, 2002; Li et al., 2022). For multi-hop legal QA this quietly breaks the task: two parties replaced by the same [NAME] token make a share-comparison question unanswerable by construction. KhatianDoc’s positional-token pipeline (§3.3) preserves the referential distinctions multi-hop reasoning needs, a design choice we have not seen in prior legal-NLP benchmarks.

## 3 The KhatianDoc Benchmark

KhatianDoc is built from real government land records and verified end to end without any automated transcription. This section covers how the documents were sourced, transcribed, anonymized, and cut into tasks. §4 then describes how we test models against them.

## 3.1 Source documents

The benchmark consists of 107 RS (Revisional Survey) Khatian records obtained through the Vumi (ভূিম) Ofice of Munshiganj district, Bangladesh. Each record is a full-page scan of a printed government form filled in by hand, recording ownership, plot boundaries, and fractional inheritance shares for a single Mouza, Kumariya (কু মািরয়া). Six documents run onto a second physical page; we keep those as paired scans $\left( \mathtt { \Pi } _ { - } \mathtt { p } 1 / \mathtt { \Pi } _ { - } \mathtt { p } 2 \right)$ rather than stitching them, so that document boundaries match what the ofice actually issued. Because every record shares one issuing ofice, one Mouza, and one standardized form, the corpus isolates a single, welldefined administrative genre: a legal instrument in current use, not a synthetic imitation of one. Figure 1 shows two examples from the release.

## 3.2 Ground-truth construction

Every field in KhatianDoc (metadata, table rows, and Ana-Ganda symbol strings alike) was transcribed by hand directly from the scan. No OCR system was used at any point, and this was deliberate. The Ana-Ganda Unicode range (§3.4) has no reliable font or OCR support, so leaning on automated transcription for the ground truth would have folded in precisely the errors the benchmark is meant to measure in the systems under test. Annotators produced a first transcription of each document; a Bangladeshi lawyer with land-law expertise then reviewed every transcription independently in a second pass, paying particular attention to the fraction strings, whose reading carries direct legal weight for the recorded share. That second pass reached complete agreement with the corrected transcriptions on both symbol identity and decimal value, and its output is the ground truth used throughout the paper. The corpus is not large, but it is small in the way a gold set is small: every value in it has been read twice by a human, once by a lawyer, and none of it was guessed by a machine.

## 3.3 Privacy-preserving anonymization

RS Khatians are live legal records that name real individuals: owners, their fathers or husbands, and residence prefixes. After transcription, we replaced every such field with a positional token ([PERSON\_1], [PERSON\_2], …), applied consistently across the Task 3 structured records and carried through the Task 4 question, answer, and evidence-span pipeline. We use positional rather than generic tokens on purpose: several Task 4 questions are multi-hop and turn on telling two named parties in the same document apart, and a single flat [NAME] token would erase that distinction and make part of the QA unanswerable. Structural and legal fields (Mouza, khatian number, land class, share fractions) are left untouched in the text ground truth, since anonymizing them would delete exactly what the tasks are built to test. The released page images are handled separately: they additionally mask the header regions carrying the khatian and form numbers and the district/upazila/mouza labels, so the image release and the text release do not expose the same fields. We make this split explicit as a two-tier scoring protocol in §7, since it is the one place where the public artifacts and the secure originals diverge.

## 3.4 The Ana-Ganda-Kora-Kranti-Til system

Bangladeshi ownership shares are recorded in a base-16 positional fraction system, Ana-Ganda-Kora-Kranti-Til (আনা-গŌা-কড়া-Ëািť-িতল), not in base-10 decimals. The primary unit is the Ana: 16 Ana make one full plot, and Ana values are written with dedicated Unicode glyphs in the range U+09F4–U+09F9. Four further subunits (Ganda, Kora, Kranti, Til) refine a share below a single Ana, to a precision no ordinary base-10 fraction expresses directly (full conversion rules and a worked example are in Appendix B). This is the current legal standard for property division in the country, and it lives almost entirely outside mainstream digital typography: the glyphs have no standard font support, and no OCR pipeline we tested recognizes them at all. To our knowledge, KhatianDoc is the first dataset to encode Ana-Ganda values as machine-checkable ground truth.

## 3.5 Task definitions and statistics

KhatianDoc has four tasks of widening scope, from an isolated glyph to a full document, summarized in Table 1.

Task 1 (Symbol recognition) pairs 95 grayscale crops of Ana-Ganda symbols with ground-truth Unicode transcriptions spanning 52 distinct strings. The crops are 256 256 PNGs drawn from the documents (80 in-the-wild crops plus 15 canonical unit symbols), and the label set is dominated by three glyphs: the 4-Ana mark ৷ (105 occurrences), the 1-Ana mark ৴ (82), and the Bengali digit ১ (54).

Task 2 (Fraction-to-decimal) reuses those same strings as text input and asks for each one’s decimal value under the base-16 rule, over 53 distinct targets. Labels are constructed to line up with Task 1 exactly, so recognition and arithmetic can be scored as two independent stages over the same underlying values.

Task 3 (Field extraction) asks for a documentlevel metadata tier and a row-level tier (261 rows across 107 documents) from each full-page scan.

Task 4 (Document QA) derives 1,634 question– answer pairs from the Task 3 ground truth: 1,206 across six simple categories and 428 across four complex categories that require multi-step reasoning. A stratified 300-question subset (73.7% simple, 26.3% complex) is the standardized evaluation set used throughout the paper. Full taxonomies and example questions are in Appendix C.

<table><tr><td>Task</td><td>N</td><td>Key stat</td><td>Metric</td></tr><tr><td>T1 Symbol Rec.</td><td>95</td><td>52 strings</td><td>CER, EM</td></tr><tr><td>T2 Fraction→Dec.</td><td>95</td><td>53 values</td><td>Exact, MAE</td></tr><tr><td>T3 Field Extr.</td><td>107</td><td>261 rows</td><td>F1, EM</td></tr><tr><td>T4 Document QA</td><td>1,634</td><td>300 eval</td><td>EM, ANLS</td></tr></table>

Table 1: KhatianDoc task statistics. Task 4 results use the 300-item stratified evaluation subset (73.7% simple, 26.3% complex).

With a human-verified, privacy-preserving ground truth in place across all four tasks, we now ask whether current multimodal LLMs can use it.

## 4 Experimental Setup

We evaluate six multimodal LLMs (Table 6, Appendix A) spanning closed commercial APIs and open weights, from 8B to over 70B parameters where disclosed. All six are reached through a single OpenRouter key with identical retry and rate-limit logic, which removes provider-specific client code as a confound. Every task runs under fixed, zero-shot, deterministic instructions: no incontext examples, no chain-of-thought elicitation. Blocked or dropped API responses are logged as empty strings rather than discarded or silently retried, so an empty prediction in our results is a recorded outcome, not a missing data point.

Generation length is capped per task: 30 tokens for Tasks 1 and 2, 500 for Task 3, and 200 for Task 4, matched to the expected output length of each. Scoring uses CER and EM for Task 1; Exact Decimal Match and MAE for Task 2; metadata EM and row-level F1 for Task 3; and EM with ANLS for Task 4. Prompts, formal metric definitions, thresholds, and infrastructure details are in Appendices A–F.

## 5 Results

Table 2 reports primary metrics for all six models across the four tasks. No model clears 26% on any single metric except Task 3 metadata EM, where several score higher for a reason §6.2 traces to constant and null fields rather than extraction, and individual metrics repeatedly bottom out at exactly zero.

## 5.1 A reasoning floor: five categories at zero

Figure 2 breaks Task 4 down by category and model over the 300-item subset, and the shape of it is hard to miss. Four categories (fraction share, legal fraction math, counterfactual check, and conditional filtering) plus multi-hop reasoning return zero correct answers from every one of the six models, with no single exception across 118 questions. This holds across the full range of scale and access we tested, from an 8B open model to 72B+ closed frontier systems. A fifth category, total area, comes close: four of the six models score zero on it and the remaining two recover only a handful of answers each, so we report it alongside rather than folding it into the zero-exception count. Together these five categories make up 39.3% of the stratified evaluation set (Table 3). The simple lookup categories on the left of Figure 2 show the only real signal in the task, and even there it is patchy and model-specific.

<table><tr><td>Model</td><td>T1 CER↓</td><td>T1 EM↑</td><td>T2 Exact↑</td><td>T3 RowF1↑</td><td>T3 Meta-EM↑</td><td>T4 ANLS↑</td><td>T4 EM↑</td></tr><tr><td>Gemini 2.5 FL</td><td>93.29</td><td>0.00</td><td>2.11</td><td>5.16</td><td>45.79</td><td>21.12</td><td>13.33</td></tr><tr><td>Qwen2.5-VL-72B</td><td>80.91</td><td>1.05</td><td>6.32</td><td>11.97</td><td>9.81</td><td>5.88</td><td>0.34</td></tr><tr><td>Qwen3-VL-8B</td><td>76.89</td><td>0.00</td><td>2.11</td><td>17.48</td><td>11.99</td><td>19.54</td><td>13.00</td></tr><tr><td>Llama 4 Scout</td><td>90.91</td><td>1.05</td><td>0.00</td><td>6.43</td><td>35.83</td><td>17.84</td><td>11.33</td></tr><tr><td>Gemma 4 26B</td><td>84.70</td><td>0.00</td><td>5.26</td><td>25.84</td><td>30.84</td><td>4.68</td><td>0.33</td></tr><tr><td>GPT-4o Mini</td><td>96.56</td><td>0.00</td><td>1.05</td><td>0.10</td><td>2.02</td><td>3.48</td><td>0.00</td></tr></table>

Table 2: Primary metrics per model, all values %. Best per column in bold. No model clears 26% on any metric except Task 3 metadata EM, which §6.2 shows is inflated by constant and null fields rather than extraction.

<table><tr><td>Category</td><td>n</td><td>Models at 0%</td></tr><tr><td>fraction_share</td><td>39</td><td>6/6</td></tr><tr><td>legal_fraction_math</td><td>20</td><td>6/6</td></tr><tr><td>counterfactual_check</td><td>20</td><td>6/6</td></tr><tr><td>conditional_filtering</td><td>20</td><td>6/6</td></tr><tr><td>multi_hop_reasoning</td><td>19</td><td>6/6</td></tr><tr><td>Total</td><td>118</td><td>39.3% of 300</td></tr><tr><td>total_area (cf.)</td><td>7</td><td>4/6</td></tr></table>

Table 3: Task 4 categories with zero success across all six models (top). total\_area is shown for comparison (bottom): four of six score zero, the other two answer a few correctly.

## 5.2 Arithmetic below a context-free baseline

Task 2 strips out visual recognition: models get the symbol string as text, together with the base-16 conversion rule, and must return a decimal. A trivial baseline that always predicts the dataset’s mean decimal value (0.3935) achieves a mean absolute error of 0.237. Figure 3 shows that every model with a defined numeric output finishes above that error, between 0.40 and 0.42, despite being handed the conversion rule verbatim in the prompt. And for every model in Table 2, Exact Decimal Match and the more lenient Near Match (within 0.005) return the same value, meaning no model ever lands in a close-but-not-exact band. Read together, these two facts say that model outputs are decorrelated from the input string rather than approximating it. This is not partial competence at base-16 arithmetic; it is the absence of it.

## 5.3 Divergent patterns within Tasks 1 and 3

Task 1 CER sits in a narrow, uniformly high band (76.89 to 96.56) regardless of scale or access (Figure 4). A failure that flat across such diferent systems looks more like a shared blind spot than a capability that scales with parameters. Task 3 behaves diferently: row-level F1 stays low across the board (0.10 to 25.84), while metadata EM varies far more widely and, for two models, runs well above what row F1 alone would predict. Gemini 2.5 Flash Lite adds a further wrinkle from another direction: its output fails to parse against the required JSON schema on 82 of 107 documents (the worst parse rate of any model), and yet it records the best overall Task 4 performance in Table 2. We return to both the metadata-EM pattern and this parsing tension in §6.2.

## 6 Failure Analysis and Discussion

Section 5 established that the failure is widespread. This section shows that it is also patterned, points to where our own scoring code shaped the picture, and pulls the patterns into a single taxonomy.

## 6.1 Symbol collapse and numeral-script substitution

Task 1 asks models to transcribe glyphs they have never seen in training, and none of them respond with random noise. Each family collapses instead toward a specific, characteristic substitute. Gemini 2.5 Flash Lite reaches for the Bengali Rupee Mark (U+09F2) on 11 of 95 items; Qwen2.5- VL-72B, Qwen3-VL-8B, and Gemma 4 26B fall back to digit-slash fraction notation; GPT-4o Mini emits characters from the Brahmi Unicode range (U+11156, U+11158). Representative predictions per pattern are in Appendix D.

![](images/5e53f5e02897460beac9624bb54e1e50d1e5aefabc71b3004ff2ce627d3d9999.jpg)  
Figure 2: Task 4 Exact Match (%) by category and model on the 300-item stratified subset. The red line separates simple lookup categories (left) from complex reasoning categories (right). Every complex category, plus fraction share, is zero for all six models; the only non-zero cells are scattered across the simple lookups. A dot marks a true zero.

![](images/9b53c7eb621cd7911f2550a6992e67ec3ce341ad9f905911678e7e409486da41.jpg)  
Figure 3: Task 2 mean absolute error against a constantmean baseline (dashed, MAE = 0.237). Every model that emits a number does worse than a predictor that ignores its input entirely.

A second, script-level failure shows up in Task 3. Both the ground truth and the prompt use Bengaliscript digits, yet GPT-4o Mini emits Western Arabic digits on 97 of 107 documents, against the five other scored models that stay in Bengali or mixed script on the large majority of documents (Figure 5; full breakdown in Appendix E). Because Task 3 scoring is exact-string match, a Western-digit prediction against a Bengali-script reference scores wrong regardless of whether the underlying value is right. This conflates a wrong value with a right value in a form the metric refuses to credit, a distinction we flag here because it matters for anyone reusing exact-match scoring on mixed-script data.

![](images/36a3029b3a27a015b442ce696f775686798d5acc6f30d6866f85b5ec7ce83ab3.jpg)  
Figure 4: Task 1 Character Error Rate (%). The band is uniformly high and roughly scale-independent, from an 8B open model to 72B+ closed systems.

## 6.2 Two places our own metrics needed correcting

Checking KhatianDoc’s scoring code against its results turned up one artifact in each direction.

Marking a correct answer wrong. Task 4’s prompt tells models to answer with a fixed refusal sentence whenever the requested information is absent. Several ground-truth answers for genuinely absent information instead use a short-form negation, েনই, on its own. Gemini 2.5 Flash Lite, which follows the prompt to the letter, is scored wrong on these items despite answering exactly as instructed. A milder version of the same problem affects Llama 4 Scout, which answers খিতয়ান নং-১৬ against a reference of ১৬ alone. We report a normalized rescoring that maps the prompt’s canonical refusal sentence and any ground-truth shorthand onto a single scored class; this raises Gemini 2.5 Flash Lite’s Task 4 EM from 13.33% to 21.67% and Llama 4 Scout’s from 11.33% to 18.33% (Table 4). No other model’s Task 4 EM changes, since only these two produced the refusal-form answers the bug mishandled.

![](images/c0de4e5362d8993ec64748ee2d92944b7da55fbbd8b813c639c83cc900101e7c.jpg)  
Figure 5: Numeral-script substitution on Task 3: share of predicted digits in Bengali versus Western Arabic script, for the six models scored on the task. GPT-4o Mini outputs Western digits almost exclusively, despite Bengali-script prompts and references.

<table><tr><td>Model</td><td>T4 EM (raw)</td><td>T4 EM (corrected)</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>13.33</td><td>21.67</td></tr><tr><td>Llama 4 Scout</td><td>11.33</td><td>18.33</td></tr></table>

Table 4: Refusal-normalized rescoring of Task 4 EM (%), raw values from Table 2 beside their corrected counterparts. Only these two models are afected; all other Task 4 EM values are unchanged.

Marking an absent answer correct. Task 3 metadata EM has the opposite problem: it rewards fields that need no reading of the image. Two of its six fields (district and upazila) take a single value across the entire corpus, so a model recovers them from prior knowledge alone; and groundtruth total\_area is null for 65 of 107 documents, so any model that emits null earns a free match. A model can therefore post a sizable metadata EM by reproducing two constants and matching null, without extracting anything from the page, which is why the metadata-EM column in Table 2 runs well above every other metric. Rather than publish a single corrected number that would depend on per-field bookkeeping we do not release, we take the cleaner route: we do not treat metadata EM as a scored quantity at all. We report it raw in Table 2 as an upper bound and make row-level F1 (which credits only fields actually recovered from the image) the metric of record for Task 3.

<table><tr><td>Model</td><td>T1 pattern</td><td>T3 note</td></tr><tr><td>Gemini 2.5 FL</td><td>Rupee-mark sub.</td><td>parse fail 82/107</td></tr><tr><td>Qwen2.5-VL-72B</td><td>Digit-slash</td><td>Bengali 83/104</td></tr><tr><td>Qwen3-VL-8B</td><td>Digit-slash</td><td>Bengali 105/107</td></tr><tr><td>Llama 4 Scout</td><td></td><td>verbose-EM gap</td></tr><tr><td>Gemma 4 26B</td><td>Digit-slash</td><td>Bengali 104/104</td></tr><tr><td>GPT-4o Mini</td><td>Brahmi hallucin.</td><td>Western 97/107</td></tr></table>

Table 5: Dominant failure pattern per model, from sampled qualitative review. A dash means no single pattern dominated. Full per-instance examples in Appendix D.

We surface both openly because the results our claims actually rest on (the row-level extraction numbers and the core Task 4 reasoning categories) are untouched by either one.

## 6.3 A failure taxonomy across models

Table 5 collects the dominant failure pattern per model, and the groupings are distinct. The Qwen and Gemma models share a single Task 1 pattern with no Task 3 anomalies. Gemini 2.5 Flash Lite and Llama 4 Scout each show one issue of their own: a high parse-failure rate and a verbositydriven EM gap, respectively. GPT-4o Mini stands alone in substituting Western digits for Bengali script. Taken with the metric audit, these patterns shape how Table 2 should be read: the failures are systematic, model-family-specific, and in two places partly attributable to scoring artifacts we have now surfaced.

## 7 Conclusion

KhatianDoc makes two contributions to legal NLP. The first is a resource: 107 real Vumi-ofice Khatian records structured into four tasks, with ground truth built entirely by hand, checked by a legal professional, and anonymized through a positionaltoken pipeline that preserves the referential structure multi-hop questions need. Inside that resource, the first machine-checkable encoding of the Ana-Ganda-Kora-Kranti-Til base-16 fraction system gives the benchmark its hardest material and its clearest motivation. The second is a diagnosis: across six multimodal LLMs from 8B to

70B+ parameters, 39.3% of the stratified evaluation set returns zero correct answers from every model, every model that emits a number underperforms a context-free baseline on the arithmetic task, and failure on both fronts is systematic and model-family-specific rather than random. Along the way we found two places where our own evaluation would have produced misleading numbers without a close audit (one inflating a metadata metric through constant and null fields, one penalizing correct refusals), and we surface both openly. A benchmark that anticipates its own critique is more useful to build on than one that does not.

## Limitations

Geographic and administrative scope. All 107 documents come from a single Mouza, Kumariya, in Munshiganj district. RS Khatian forms are standardized nationally, so the document structure generalizes, but layout variation introduced by other districts, other Vumi ofices, or older survey rounds is not represented in this release. Models that beat the current baseline on KhatianDoc may not carry those gains to more varied Khatian corpora without further evaluation.

Task 1 sample size. The 95 symbol crops in Task 1 are drawn from that same single-Mouza corpus. Fifty-two unique strings cover a meaningful slice of the Ana-Ganda inventory, but rarer compound fractions are likely underrepresented; broader coverage would need source documents from a wider geographic base or a larger annotated corpus.

Zero-shot evaluation only. We report zeroshot performance exclusively. No fine-tuning or retrieval-augmented condition is evaluated, so we cannot say how much of the observed gap is addressable with task-specific adaptation or incontext examples. We treat zero-shot as a diagnostic baseline, not a ceiling.

Human legal evaluation of complex QA. The four complex Task 4 categories are scored automatically against exact or near-exact reference strings. Legal correctness and string similarity are not the same thing: a model that reasons to a legally valid answer through a valid paraphrase may be marked wrong, and a model that produces the reference string by coincidence may be marked right. Human legal-expert evaluation of complex-category predictions is left for future work.

OCR-augmented evaluation. Every evaluation here runs in image-only mode, with no preextracted text. An OCR-augmented pipeline that passes recognized text alongside the image might shift the Task 3 and 4 failure distribution substantially. We treat image-only as the baseline condition and leave OCR-augmented results to future work, flagging their absence because the gap between the two conditions is likely large for this document type.

## Ethical Considerations

Data provenance. The 107 source documents were obtained through oficial channels from the Vumi Ofice of Munshiganj district, under the standard government record-access mechanism. No document was scraped, purchased from a third party, or sourced from a private individual. These are oficial government survey records rather than private correspondence, though they do contain personally identifying land-ownership information, which is what motivates the handling described next.

Anonymization and image release. Text anonymization (§3.3) removes every personal identifier from the released ground-truth files: owner names, father and husband names, and residence prefixes are all replaced by positional tokens before any file is used for annotation, training, or evaluation, and this is carried through Task 4 questions, answers, and evidence spans. No original name or residence string appears in any released text file.

The scanned images need their own answer, because a redacted transcript sitting beside an unredacted scan protects no one. We therefore define two scoring protocols and state which one every number in this paper comes from.

Secure-original protocol. Every result reported here is computed on the original, unredacted scans, handled under access control, with all fields of all four tasks scored. These scans are not part of the open release; they are retained and made available only to vetted researchers through the same Vumiofice channel, under terms that match the sensitivity of live legal records.

Public-redacted protocol. The open release ships redacted page images (Figure 1): the owner name-and-address column, father and husband names, residence prefixes, form and khatian numbers, the district/upazila/mouza header labels, and the oficer’s seal and signature are masked with opaque boxes. Because masking removes those regions from view, the public protocol excludes the masked fields from image-based scoring rather than pretending they can still be read. Concretely, Task 1, Task 2, the row-level content of Task 3 (plot number, land class, share fractions, and the Ana-Ganda area entries), and every Task 4 question that rests on those unmasked fields are scored exactly as on the originals, since their regions are never touched. The Task 3 metadata fields that fall under a mask (khatian and form number, district, upazila, and mouza) are not scored on the public images; two of them are corpus-constants and the rest are identity-adjacent header fields, so dropping them costs the benchmark little, and §6.2 already argues that metadata EM is not a load-bearing metric. Owner references stay scorable throughout, because the ground truth encodes them as positional tokens keyed to row order rather than to the masked names themselves.

The two protocols therefore measure the same four tasks, difering only in that the public one omits the masked metadata headers, a gap we state outright so that results computed on the redacted release and on the secure originals are never silently compared. In short: anonymized text and redacted images are public, original scans are access-controlled, and the paper’s numbers are the secure-original ones.

Potential for misuse. KhatianDoc is meant as a diagnostic benchmark for documentunderstanding research. The failure patterns it documents could in principle inform adversarial evaluation of legal-AI systems in South Asian jurisdictions, but those same patterns are already visible to anyone holding a standard Khatian form and an API key. The more pressing risk runs the other way: downstream land-administration tools built on models shown here to fail badly on fraction math and multi-hop legal reasoning could emit incorrect ownership outputs with real legal consequences for afected citizens. We recommend against deploying any model evaluated here on real Khatian records in an unaudited capacity.

Benefit to afected communities. Landownership disputes are among the most common sources of civil litigation in Bangladesh, and the inaccessibility of Khatian records to digital tools is one practical obstacle to resolving them. KhatianDoc is a first step toward automated, privacy-respecting tools that could assist lawyers, surveyors, and citizens who work with these records. It is released to enable that research, not to replace the legal professionals who currently do this work.

## References

Abhik Bhattacharjee, Tahmid Hasan, Wasi Ahmad, Kazi Samin Mubasshir, Md Saiful Islam, Anindya Iqbal, M. Sohel Rahman, and Rifat Shahriyar. 2022. BanglaBERT: Language model pretraining and benchmarks for low-resource language understanding evaluation in Bangla. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1318–1327.

Ilias Chalkidis, Manos Fergadiotis, and Ion Androutsopoulos. 2021. MultiEURLEX – a multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 6974–6996.

Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, and Nikolaos Aletras. 2022. LexGLUE: A benchmark dataset for legal language understanding in English. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4310–4330.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. FinQA: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Sumanth Doddapaneni, Rahul Aralikatte, Gowtham Ramesh, Shreya Goyal, Mitesh M. Khapra, Anoop Kunchukuttan, and Pratyush Kumar. 2023. Towards leaving no Indic language behind: Building monolingual corpora, benchmark and models for Indic languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12402–12426.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, et al. 2023. LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. InAdvances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track).

Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. 2021a. CUAD: An expert-annotated NLP dataset for legal contract review. In Advances in Neural Information Processing Systems 34 (Datasets and Benchmarks Track).

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems 34 (Datasets and Benchmarks Track).

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. LayoutLMv3: Pre-training for document AI with unified text and image masking. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4083–4092.

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. 2019. ICDAR2019 competition on scanned receipt OCR and information extraction (SROIE). In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520.

Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. 2019. FUNSD: A dataset for form understanding in noisy scanned documents. In 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW), volume 2, pages 1–6.

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. OCR-free document understanding transformer. In European Conference on Computer Vision (ECCV), pages 498–517.

Xuechen Li, Florian Tramèr, Percy Liang, and Tatsunori Hashimoto. 2022. Large language models can be strong diferentially private learners. In International Conference on Learning Representations (ICLR).

Pierre Lison, Ildikó Pilán, David Sánchez, Montserrat Batet, and Lilja Øvrelid. 2021. Anonymisation models for text data: State of the art, challenges and future directions. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4188–4203.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2024. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR).

Vijit Malik, Rishabh Sanjay, Shubham Kumar Nigam, Kripabandhu Ghosh, Shouvik Kumar Guha, Arnab Bhattacharya, and Ashutosh Modi. 2021. ILDC

for CJPE: Indian legal documents corpus for court judgment prediction and explanation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4046–4062.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. DocVQA: A dataset for VQA on document images. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2200–2209.

Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. 2022. NumGLUE: A suite of fundamental yet challenging mathematical reasoning tasks. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 3505–3523.

Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. 2019. CORD: A consolidated receipt dataset for post-OCR parsing. In Workshop on Document Intelligence at NeurIPS 2019.

Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter Staar. 2022. DocLayNet: A large human-annotated dataset for documentlayout analysis. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3743–3751.

Latanya Sweeney. 2002. k-anonymity: A model for protecting privacy. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(5):557–570.

World Bank. 2024. Access to land in South Asia: The World Bank guidance note. Technical report, World Bank, Washington, DC.

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. LayoutLM: Pre-training of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1192–1200.

## A Evaluated Models

Table 6 lists the six multimodal LLMs, their Open-Router identifiers, and access type.

<table><tr><td>Model</td><td>OpenRouter ID</td><td>Access</td></tr><tr><td>Gemini 2.5 Flash Lite</td><td>google/gemini-2.5-flash-lite</td><td>Closed</td></tr><tr><td>Qwen2.5-VL-72B</td><td>qwen/qwen2.5-vl-72b-instruct</td><td>Open</td></tr><tr><td>Qwen3-VL-8B</td><td>qwen/qwen3-vl-8b-instruct</td><td>Open</td></tr><tr><td>Llama 4 Scout</td><td>meta-1lama/1lama-4-scout</td><td>Open</td></tr><tr><td>Gemma 4 26B</td><td>google/gemma-4-26b-a4b-it</td><td>Open</td></tr><tr><td>GPT-4o Mini</td><td>openai/gpt-4o-mini</td><td>Closed</td></tr></table>

Table 6: Evaluated models, accessed uniformly through OpenRouter.

## B The Ana-Ganda-Kora-Kranti-Til Numeral System

Unlike plain base-10 decimals or pure base-16 hexadecimals, the Ana-Ganda system is a mixed-radix positional fraction. Table 7 gives the subunit hierarchy and Table 8 the dedicated Unicode glyphs.

<table><tr><td>Subunit</td><td>Relation</td><td>Decimal (of 1 plot)</td></tr><tr><td>Ana</td><td>16 Ana = 1 Plot</td><td>0.0625</td></tr><tr><td>Ganda</td><td>20 Ganda = 1 Ana</td><td>0.003125</td></tr><tr><td>Kora</td><td>4 Kora = 1 Ganda</td><td>0.00078125</td></tr><tr><td>Kranti</td><td>4 Kranti = 1 Kora</td><td>0.0001953125</td></tr><tr><td>Til</td><td>4 Til = 1 Kranti</td><td>0.000048828125</td></tr></table>

Table 7: Mixed-radix subunit hierarchy of the Ana-Ganda system.

<table><tr><td>Codepoint</td><td>Glyph / description</td></tr><tr><td>U+09F4</td><td> Currency Numerator One</td></tr><tr><td>U+09F5</td><td> Currency Numerator Two</td></tr><tr><td>U+09F6</td><td>ə Currency Numerator Three</td></tr><tr><td>U+09F7</td><td>I Currency Denominator Sixteen (1 Ana)</td></tr><tr><td>U+09F8</td><td>n Currency Numerator Four</td></tr><tr><td>U+09F9</td><td>• Currency Denominator Twenty (1 Ganda)</td></tr></table>

Table 8: Unicode codepoints used in Task 1 and Task 2. Standard Bengali numerals (০–৯) carry higher-order counts of these subunits.

Worked example. Consider a share recorded as 3 Ana, 5 Ganda, and 2 Kora. The decimal conversion sums the mixed-radix parts:

3 Ana = 3 0.0625 = 0.1875   
5 Ganda = 5  0.003125 = 0.015625   
2 Kora = 2 0.00078125 = 0.0015625   
Total = 0.2046875

This non-standard structure is why Task 2 injects the rule explicitly into the prompt, and why models still fail to approximate it (§5.2). Across the 95 crops the decimal values span 0.0625 to 1.19, with a mean of 0.3935 and 53 distinct targets.

## C Dataset Statistics and Task Examples

Corpus composition. The 95 Task 1 crops split into compound Ana-Ganda fractions (55; 57.9%), simple Ana symbols (35; 36.8%), hasanta-numeric full-share marks (3; 3.2%), a lone decimal number (1), and a lone fraction slash (1). The 107 documents yield 261 rows (mean 2.44 rows/doc, range 1–9) and 573 owner entries over 459 unique anonymized names. Sixty-three rows retain bracketed or UNCLEAR transcription markers directly from the ground truth; we keep these verbatim rather than manufacture a clean value. The Task 4 categories and their full counts are in Table 9.

<table><tr><td>Category</td><td>Count</td></tr><tr><td>plot_lookup</td><td>261</td></tr><tr><td>land_classification</td><td>255</td></tr><tr><td>area_query</td><td>225</td></tr><tr><td>metadata_lookup</td><td>214</td></tr><tr><td>fraction_share</td><td>212</td></tr><tr><td>multi_hop_reasoning</td><td>107</td></tr><tr><td>legal_fraction_math</td><td>107</td></tr><tr><td>conditional_filtering</td><td></td></tr><tr><td></td><td>107</td></tr><tr><td>counterfactual_check</td><td>107</td></tr><tr><td>total_area</td><td>39</td></tr><tr><td>Total</td><td>1,634</td></tr></table>

Table 9: Full Task 4 QA distribution across all ten categories (six simple, four complex).

Task 3 JSON schema. Models are instructed to output the following exact structure.

{   
"metadata": {   
"khatian\_no": "string", "district":   
"string",   
"upazila": "string", "mouza": "string",   
"rs\_no": "string", "total\_area":   
"string|null"   
},   
"rows": [{   
"dag\_no": "string", "land\_class":   
"string",   
"share": "string", "area\_by\_share":   
"string",   
"owners": ["[PERSON\_1]", "[PERSON\_2]"]   
}]   
}

## Task 4 complex examples.

• Legalfraction math: দাগ নং ১২৩ এ [PERSON\_1] এবং [PERSON\_2] এর েমাট জিমর পিরমাণ কত? (What is the total land area held by Person 1 and Person 2 in Dag 123?) Requires: row filtering, extracting two Ana-Ganda strings, converting to decimal, summing, and multiplying by total area.

• Counterfactual check: যিদ [PERSON\_1] তার অংেশর অেধর্ক [PERSON\_3] েক দান কের, তেব [PERSON\_3] এর নতুন অংশ কত আনা হেব? (If Person 1 donates half their share to Person 3, what will Person 3’s new share be, in Ana?) Requires: multi-hop extraction, base-16 division, and base-16 addition.

## D Symbol Collapse Examples

Table 10 gives representative raw outputs for the family-specific out-of-vocabulary collapse pat-

terns of §6.1.

<table><tr><td>Model</td><td>Target</td><td>Raw output</td></tr><tr><td>Gemini FL</td><td>心</td><td>(Rupee Mark)</td></tr><tr><td>Qwen 72B</td><td>心</td><td>S/S (Digit-Slash)</td></tr><tr><td>GPT-40</td><td>心</td><td>U+11156 (Brahmi)</td></tr></table>

Table 10: Raw outputs illustrating family-specific outof-vocabulary collapse on Task 1.

## E Numeral-Script Substitution

Table 11 reports the percentage of predicted digits in Bengali versus Western Arabic script for Task 3, the numeric form of Figure 5, over the six models scored on the task. GPT-4o Mini outputs Western digits almost exclusively despite Bengali-script prompts and documents.

<table><tr><td>Model</td><td>Bengali</td><td>Western</td></tr><tr><td>Gemini FL</td><td>95%</td><td>5%</td></tr><tr><td>Qwen 72B</td><td>100%</td><td>0%</td></tr><tr><td>Qwen 8B</td><td>98%</td><td>2%</td></tr><tr><td>Llama 4</td><td>90%</td><td>10%</td></tr><tr><td>Gemma 4</td><td>99%</td><td>1%</td></tr><tr><td>GPT-4o Mini</td><td>0%</td><td>100%</td></tr></table>

Table 11: Percentage of predicted digits in Bengali script versus Western Arabic script (Task 3).

The metadata exploit. As noted in §6.2, Task 3 metadata EM can be inflated without any visual extraction. Two of its fields (district and upazila) are constant across the entire 107-document corpus, and total\_area is null in the ground truth for 65 documents. A model that outputs the corpusconstant regional values and leaves the rest null therefore scores near 50% on metadata EM without reading a single page: it collects two constants it could have memorized and a run of free null– null matches. This is why we treat the metadata-EM column as an upper bound and base extraction claims on row-level F1 instead, which credits only fields actually recovered from the image.

The refusal-string mismatch. Task 4 prompts instruct: “If absent, output exactly àদত্ খিতয়ােন এই তথঁ েনই।”. However, 14 ground-truth answers for genuinely absent fields use the shorthand েনই alone. A model that follows the instruction perfectly (àদত্ খিতয়ােন এই তথঁ েনই।) is scored 0 against the reference েনই under standard exact match. Normalizing both to one semantic class corrects the penalty, as detailed in §6.2.

## F Prompt Templates

## Task 2 (fraction-to-decimal).

You are an expert in Bangladeshi land measurement. Convert the following Ana-Ganda fraction string to a decimal value. Rules: 16 Ana = 1.0. 20 Ganda = 1 Ana. 4 Kora = 1 Ganda. 4 Kranti = 1 Kora. 4 Til = 1 Kranti. Output ONLY the decimal number to 10 places.

Input: [SYMBOL\_STRING] Output:

## Task 4 (legal QA).

Answer the following question based ONLY on the provided Khatian image. Answer concisely in Bengali. If the information is not present in the document, you must output EXACTLY: àদত্ খিতয়ােন এই তথঁ েনই।

Question: [QUESTION\_STRING] Answer: