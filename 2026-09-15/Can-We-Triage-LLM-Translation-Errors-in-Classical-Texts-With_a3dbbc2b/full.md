# Can We Triage LLM Translation Errors in Classical Texts Without Human References?

Source Novelty, GEMBA Scoring, and Budgeted Review through Pali-to-English Translation

Máté Metzger Independent Researcher, Hungary

## Abstract

As large language models become capable translators of classical texts, the binding constraint shifts from producing fluent output to deciding which outputs need expert review, especially where no human reference translation exists. Whether such translations can be triaged for error risk using no reference at inference time is tested through Pali-to-English translation, a demanding testbed: a canonical classical language, low-resource for modern NLP yet backed by a large, segment-aligned source-translation corpus suitable for calibration. An existing human translation serves only to calibrate and validate the detector, never to compute its signals. Three LLMs translated 15,493 passages, and five reference-free signals were compared: source novelty, source-candidate embedding distance, peer-translation disagreement, English-to-Pali backtranslation, and no-reference GEMBA scoring. These signals were calibrated on a 3,000-item reference-informed LLMadjudicated sample and checked against a 500-item author-adjudicated anchor. Source novelty proved a useful source-side risk prior but not a per-candidate error detector: rare, less formulaic passages failed more often, yet novelty alone missed many candidatespecific errors. Peer-translation disagreement and backtranslation added only secondary signal. The strongest method was no-reference GEMBA scoring by a separate panel of models generally regarded as stronger than the translators: reviewing the top 10% by GEMBA risk caught 81.6% of panel-major errors in the LLM-adjudicated calibration set, and these scores remained the best reference-free signal when validated against a humanadjudicated anchor. A same-tier panel, judges from the translators’ own class, none scoring its own output, stayed useful but clearly worse, indicating the signal depends on the judge being stronger than the audited system, not on the GEMBA prompt alone. A budgeted triage workflow is proposed: prioritize by source novelty, exploit peer disagreement among candidate translations, and reserve a stronger no-reference judge for candidateaware review. The workflow is meant to transfer to other classical-to-modern settings, such as Latin, Ancient Greek, and Sanskrit, where references exist for calibration but not for newly translated texts. Confirming that transfer is left to future work.

Keywords: LLM translation; translation quality estimation; classical-language NLP; GEMBA scoring; classical languages

## 1 Introduction

Large language models (LLMs) now produce fluent translations for texts far outside the high-resource modern-language pairs on which machine translation research has historically concentrated. This creates a new practical problem for classical-language scholarship. Many classical languages preserve vast corpora, but they are low-resource languages for modern NLP, and the number of people able to translate them with philological competence is small. As a result, large amounts of classical material remain untranslated or only partially translated. Much of this material is also philosophically or religiously dense: translation often requires domain expertise, while still allowing multiple defensible interpretations, registers, and English renderings. A model translation may read well, and may even agree with a plausible English rendering in broad outline, while still omitting a doctrinally important phrase, reversing agency, mistranslating a technical term, or silently smoothing away a difficult construction. The question is not only whether LLMs can translate classical texts. It is how scholars can triage their outputs when the most valuable use case is precisely the one in which expert translators are scarce and no human reference translation exists for the passage being newly translated.

This paper studies that problem through Pali-to-English translation. Pali is a Middle Indo-Aryan language and the classical language of the Theravada Buddhist canon, or Tipitaka. The Pali Canon preserves one of the most extensive bodies of early Buddhist literature and remains central to Buddhist studies, monastic education, meditation practices, and access to Buddhist thought (von Hinüber, 1996; Gethin, 1998; Gombrich, 2006; Bodhi, 2005). It is also a challenging translation domain. Many passages are highly formulaic, while others are elliptical, syntactically dense, or dependent on technical terms whose English renderings are interpretive rather than mechanical.

Pali is therefore a useful stress case for reference-free translation error triage. It is lowerresource than Latin, Ancient Greek, or Sanskrit in the sense relevant to current natural language processing: there are fewer dedicated computational models, benchmarks, and widely used NLP tools, despite the religious and scholarly importance of the corpus. At the same time, Pali has an unusually useful digital infrastructure for this kind of experiment. SuttaCentral and its Bilara data model provide segmented JSON source texts and corresponding English translations at large scale (SuttaCentral, 2026a; SuttaCentral and Bilara contributors, 2026). This combination is rare: a classical language with substantial cultural importance, limited dedicated NLP tooling, and a large segment-aligned source-translation corpus suitable for calibration.

The central task in this study is no-reference at inference time, not no-reference in the entire research pipeline. We use Bhikkhu Sujato’s English translation as an adjudication aid when calibrating and validating the detector. We do not use that reference to compute the risk signals; the detector uses only information that would be available for a newly translated passage without a human reference. The proposed inference-time input is only the Pali source and one or more AI candidate translations. The research-time references are used to ask whether the reference-free signals actually correlate with translation errors.

This framing matters because conventional automatic translation evaluation is poorly matched to the humanistic problem. Reference-based metrics are useful when the goal is to compare systems against a known target, but classical translation is not exhausted by agreement with one reference. Translation studies has long emphasized that translation is interpretive, situated, and shaped by audience, purpose, and tradition (Berman, 1992; Venuti, 2008; Tymoczko, 2007; Bassnett, 2013; Munday, 2016). In Buddhist translation, this is not an abstract point: choices around doctrinal terms such as dukkha, dhamma, sa˙nkh¯ara, or nibb¯ana affect how readers understand doctrine. A good audit method must therefore distinguish error from legitimate variation.

The contribution of this paper is an empirical test of reference-free triage signals for Palito-English LLM translation. It builds on multi-reference Pali benchmarking work that uses existing human translations to evaluate model outputs (Metzger and Phophichit, 2026), but asks a different operational question: what can be done when such references are unavailable at inference time? We translate 15,493 Pali passages with three LLMs, compute source-side and candidate-side risk metrics, calibrate those metrics against a 3,000-item reference-informed LLM-adjudicated sample, and validate the main conclusions against a 500-item authoradjudicated anchor set. The goal is not automatic certification of correctness. It is selective review: to estimate which passages most deserve human attention when expert review time is limited.

## 2 Literature Review

## 2.1 Translation Variation, Classical Texts, and Buddhist Philology

The translation of classical and religious texts is not a simple transfer of stable propositions into another language. Translation theory has repeatedly challenged the idea that a translation can be evaluated only as proximity to a single target string. Berman foregrounds the ethical and interpretive difficulty of receiving the foreign text without domestication (Berman, 1992). Venuti argues that fluent translation can conceal the translator’s interpretive labor (Venuti, 2008), while Tymoczko and Bassnett emphasize that translation choices are embedded in cultural, political, and institutional settings (Tymoczko, 2007; Bassnett, 2013). Munday’s survey of translation studies similarly treats equivalence as only one problem among many, rather than a neutral criterion (Munday, 2016).

These concerns are especially relevant to Pali Buddhist translation. The canon has been transmitted through religious communities and scholarly institutions over centuries, and its modern English translations differ in register, audience, and interpretive style (von Hinüber, 1996; Gethin, 1998; Bodhi, 2005; Gombrich, 2006). A plain modern rendering, a doctrinally conservative rendering, and a philologically literal rendering may all be defensible while differing substantially on the surface. This is one reason that multiple references can improve machine translation training in domains where meaningful variation is expected (Wu et al., 2024). For the present study, the lesson is methodological: the target should not be matching any single human translation, but a translation supported by the source text.

## 2.2 Machine Translation Evaluation and Quality Estimation

Machine translation evaluation began from reference overlap metrics such as BLEU, which enabled large-scale system comparison but also made target-string similarity a proxy for quality (Papineni et al., 2002). Subsequent metrics attempted to soften this dependence through synonym matching, edit distance, and character-level matching, including METEOR, Translation Edit Rate, and chrF (Banerjee and Lavie, 2005; Snover et al., 2006; Popovi´c, 2015). Neural and learned metrics later moved beyond surface overlap by using contextual embeddings or learned human-judgment prediction, including BERTScore, BLEURT, and COMET (Zhang et al., 2020; Sellam et al., 2020; Rei et al., 2020). Work on metric reliability has also shown that automatic scores can mislead when used outside the conditions under which they correlate with human judgments (Mathur et al., 2020; Marie et al., 2021).

Quality estimation addresses a different but closely related task: estimating translation quality without a human reference. This is the tradition most directly connected to the present study. OpenKiwi, TransQuest, COMETKiwi, and xCOMET all represent attempts to predict quality or identify errors from source and candidate output, sometimes with fine-grained error information (Kepler et al., 2019; Ranasinghe et al., 2020; Rei et al., 2022; Guerreiro et al., 2024). Human evaluation practice has also shifted toward explicit error annotation, especially through Multidimensional Quality Metrics (MQM), because aggregate fluency scores alone do not reveal whether a translation contains consequential meaning errors (Lommel et al., 2014; Freitag et al., 2021; Lommel et al., 2024).

The present study shares the quality-estimation goal but differs in domain and deployment assumptions. Most quality-estimation work is trained and evaluated on modern language pairs with established benchmark data. Pali-to-English translation lacks that infrastructure.

Recent domain-specific low-resource QE work shows that closed LLMs can perform strongly with prompting, while more robust open or local QE systems may require adaptation with labeled data (Gurav et al., 2026). The method tested here therefore uses transparent corpusderived features, peer-translation disagreement, backtranslation, and LLM-based scoring as reference-free signals that can be calibrated with a sampled reference-aided adjudication set.

## 2.3 LLMs as Translation Judges

Recent work has shown that LLMs can serve as scalable evaluators of generated text, but also that their judgments are not neutral instruments. MT-Bench and related work made LLM-asa-judge evaluation prominent (Zheng et al., 2023). G-Eval and Prometheus further showed that LLM evaluators can follow rubrics and produce scores aligned with human judgments in some settings (Liu et al., 2023; Kim et al., 2023). At the same time, LLM judges can show position bias, self-preference, model-family bias, and instability across prompts or rubrics (Wang et al., 2024). For translation, Kocmi and Federmann introduced GPT Estimation Metric Based Assessment (GEMBA), showing that prompted LLMs can be strong translation-quality evaluators, including in no-reference modes (Kocmi and Federmann, 2023a). GEMBA-MQM extended this idea toward error-span and MQM-style evaluation without human references (Kocmi and Federmann, 2023b).

This literature motivates the present design but also constrains its claims. We use LLM judges because they make a 3,000-item calibration feasible, but we do not treat them as a gold standard. Their labels are checked against a 500-item author anchor, and the paper separately evaluates whether GEMBA merely predicts judgments made by models or remains useful when compared to author labels.

## 2.4 Classical-Language NLP and the Pali Test Case

Classical languages occupy an awkward place in NLP. They may have large surviving corpora and high cultural importance, yet they are often low-resource for modern computational purposes: fewer annotated datasets, fewer domain-specific models, uneven tokenization, and a shortage of expert-labeled evaluation data. Broader work on linguistic diversity in NLP has shown how sharply language resources are concentrated in a small number of high-resource languages (Joshi et al., 2020), and participatory low-resource MT work emphasizes that data creation and evaluation must be tied to communities and domain experts rather than treated as purely technical extraction (Nekoto et al., 2020).

Recent classical-language NLP work shows both the promise and the limits of LLMassisted philology. Latin BERT demonstrates the value of language-specific contextual models for classical philological tasks (Bamman and Burns, 2020). LITERA frames Latin-to-English LLM translation as a research-assistance workflow rather than a replacement for expert translators (Rosu, 2025). Work on Ancient Greek medical and philosophical prose shows that LLMs can produce useful translations but fail sharply on rare technical language (Zainaldin et al., 2026). Sanskrit-English work such as Itihasa demonstrates that even large parallel corpora for premodern Indic texts can remain difficult for standard translation architectures (Aralikatte et al., 2021).

Pali has received less dedicated computational attention than Latin, Ancient Greek, or Sanskrit, though digital Buddhist studies projects have made important infrastructure available. Zigmond’s computational analysis of the Pali Canon demonstrates the feasibility of corpus-level quantitative work on Pali texts (Zigmond, 2021). Prior work on AI-driven translation of ancient Buddhist scriptures used GEMBA and other metrics to evaluate Pali-to-English LLM translations in a narrower scripture-translation setting (Phophichit and Metzger, 2026). PaliBench shows how independently translated Pali passages can be turned into a multi-reference benchmark for classical-language translation (Metzger and Phophichit, 2026). SuttaCentral provides early Buddhist texts, translations, parallels, and the Bilara translation infrastructure, including segment-level JSON data used in this study (SuttaCentral, 2026b; SuttaCentral and Bilara contributors, 2026). This makes Pali a particularly informative test case: under-resourced enough to expose the difficulty of classical-language translation, but structured enough to support a large empirical audit.

The gap addressed here is therefore narrow but important. Existing MT evaluation research gives tools for reference-based scoring, quality estimation, error annotation, and LLM judging. Existing classical-language NLP shows that LLM translation can be useful but fragile. What is missing is a practical, empirically calibrated way to triage AI translations of classical texts when no human reference translation is available at the point of use. This paper tests that missing layer.

## 3 Methodology

## 3.1 Study Design

This study evaluates whether Pali-to-English large language model (LLM) translations can be triaged for error risk without using human reference translations at inference time. The practical target is not automatic certification of correctness. The target is selective review: given a large set of AI translations, can a reference-free system rank or flag those most likely to contain correction-worthy errors, so that a limited human review budget is spent where it matters most? This places the study within the broader quality-estimation tradition in machine translation, where the goal is to estimate translation quality from source and candidate output rather than from candidate-reference overlap (Rei et al., 2022; Guerreiro et al., 2024).

The design separates three tasks that are often conflated in automatic translation evaluation. First, candidate translations are produced from Pali source passages. Second, noreference risk signals are computed from the source passage, the candidate translation, and, for some signals, other AI translations of the same passage. Third, a sampled subset is adjudicated using a reference-aided procedure so that the no-reference signals can be calibrated against error labels. In other words, human reference translations are used to evaluate the detector, not to compute the detector.

The study is deliberately framed around budgeted triage, but not as a simple cheap-versusexpensive binary. The signals tested here differ in what they require. Source novelty is a source-only prior and can be computed before translation. Source-candidate embedding distance requires the candidate translation and embeddings. Peer-centroid distance is noreference in the human-translation sense, but it is not intrinsically cheap because it requires additional AI translations of the same passage. Backtranslation and GEMBA-style scoring require further LLM calls and are therefore higher-effort candidate-aware checks. This triage framing reflects the intended use case for classical-language translation: many passages may be translated, but only a small proportion can receive intensive expert attention.

Although implemented here for Pali-to-English translation, the methodological unit is more general: source passage, candidate translation, optional peer translations, and a calibration set with adjudicated error labels. The Pali-specific parts of the workflow are the tokenizer, source-corpus rarity estimates, and philological adjudication criteria. The same design can therefore be adapted to other classical-to-modern translation settings where enough source text is available to estimate source difficulty and at least a sampled set can be adjudicated.

Table 1: Final corpus filtering rules.
<table><tr><td>Rule</td><td>Reason</td></tr><tr><td>Exclude passage ids ending in : 0</td><td>Removes title and front-matter units rather than substantive translated passages.</td></tr><tr><td></td><td>Require non-empty Pali and English passage text Removes unusable source or reference records.</td></tr><tr><td>tokens</td><td>Require at least 10 normalized Pali content-word Removes very short passages that provide little information for translation-error evaluation.</td></tr><tr><td>Require every Pali segment in the passage to have a corresponding English segment</td><td>Avoids incomplete source-reference alignment.</td></tr><tr><td>Require English/Pali character ratio between 0.5 Removes likely extraction, segmentation, or and 2.0</td><td>alignment anomalies.</td></tr></table>

## 3.2 Corpus Construction

The main corpus was constructed from Bilara JSON files containing Pali source text and Bhikkhu Sujato’s corresponding English translations, available through SuttaCentral’s public GitHub repository under permissive licensing. Bilara stores texts at segment level. A segment is one keyed unit such as mn1:1.1 or mn1:1.2. A passage is the top-level key obtained by removing the final segment component after the last dot: for example, mn1:1.1 and mn1:1.2 both belong to passage mn1:1. In this study, the passage, not the segment, is the translation and evaluation unit, because passages are usually more semantically complete and interpretable while remaining short enough for controlled model calls and adjudication.

All Pali segments belonging to a passage were concatenated in Bilara segment order. The corresponding English segments (Bhikkhu Sujato’s translations) were concatenated in the same way for reference-aided adjudication. Title and front-matter passages ending in :0 were excluded. The final corpus was then filtered to remove very short, incomplete, or anomalously aligned units before translation. The filtering rules are shown in Table 1.

The source profile contained 26,165 eligible non-title passages before final filtering. The final test corpus contains 15,493 passages and 91,474 source segments, retaining 59.2% of the profiled non-title passage pool. Non-exclusive rejection counts were 6,416 passages under the 10-content-token threshold, 3,755 passages with incomplete English segment coverage, and 1,775 passages outside the English/Pali character-ratio bounds. On the Pali source side, the final corpus contains 6.48 million characters, 725,623 whitespace-delimited words, and 3.73 million cl100k\_base tokens, roughly equivalent to 1,450-1,600 printed A4 pages depending on words-per-page assumptions; Sujato’s corresponding English passages were retained as reference aids for adjudication.

## 3.3 Source Novelty and Corpus Stratification

A central hypothesis of the study is that error risk is partly conditioned by source difficulty. For this reason, every Pali passage was assigned a continuous source novelty index before sampling. The index is source-only: it can be computed before any AI translation is generated. This choice is also motivated by recent ancient-language translation work in which terminology rarity was a strong predictor of LLM translation failure (Zainaldin et al., 2026).

Normalized content tokens were produced by Unicode normalization, lowercasing, normalizing ˙m to m<sub>.</sub>, extracting word-like Pali tokens, dropping tokens shorter than four characters, and excluding a small set of common particles such as atha, ca, eva, evam<sub>.</sub>, iti, kho, pana, pi, and v¯a. Source novelty combines four z-scored source-side components:

1. content-word rarity, measured as the density of normalized content tokens with document frequency ≤ 10 in the Pali corpus;

2. character 5-gram rarity, measured as the density of normalized character 5-grams with document frequency ≤ 3;

3. formulaicity, measured by selected repeated Pali 5-10 word n-grams;

4. source-neighborhood density, measured as the top-5 mean similarity to other Pali passages using character 3-5 gram TF-IDF cosine similarity.

The index is computed as:

source\_novelty\_index =   
(   
z(content\_word\_df10\_rarity)   
+ z(char5gram\_df3\_rarity)   
- z(repeated\_formula\_count)   
- z(top5\_source\_neighborhood\_density)   
) / 4

Higher values indicate passages that are rarer, less formulaic, and more isolated from near-neighbor source passages. The final corpus was divided into low, middle, and high novelty bands for descriptive analysis and stratified sampling. These bands are not treated as natural categories; the continuous novelty score is the primary source-side measure. Corpus counts by novelty band are reported in Appendix Table A1.

## 3.4 Candidate Translation Generation

Three OpenRouter-hosted translator models were selected for the main experiment: deepseek/ deepseek-v4-pro, qwen/qwen3.6-plus, and x-ai/grok-4.3. The aim was to use capable models that could plausibly produce useful translations, but not the strongest and most expensive frontier systems. This was a deliberate calibration choice: the study requires enough errors to evaluate triage behavior, while still keeping full-corpus translation practically affordable. Prior PaliBench results also informed the choice of model families by identifying systems that were relatively strong but still produced measurable high-drift outliers relative to human-reference consensus, some of which are expected to correspond to real errors (Metzger and Phophichit, 2026). The design is therefore about evaluating error-risk triage, not constructing a leaderboard of the best possible Pali translators.

Each model translated all 15,493 Pali passages using the same passage-level translation prompt. The prompt instructed the model to translate into English only, preserve named persons, places, numbers, lists, negation, agents, and doctrinal terms, and return JSON output. The exact translator, adjudication, backtranslation, and GEMBA prompts are included in the reproducibility package as part of the runnable scripts. Passages were batched under an approximate 3,000-token input budget per API call. The translation script supported interruption and resumption. It also detected source-copy-like failures, where the model output substantially reproduced the Pali source instead of translating it. These occurred frequently during Grok generation; the script retried them with smaller batches and retained persistent failures as candidate outputs rather than silently deleting them.

DeepSeek and Qwen produced complete translation files for all 15,493 passages. Grok produced 15,491 ordinary translations and two persistent source-copy failures. These two failures were retained because copying the Pali source instead of translating it is a real translation failure and should be visible to a triage system. The full translation table therefore contains 46,479 translation instances, one for each translator model - passage pair.

Table 2: Translator models and candidate output counts.
<table><tr><td>Translator model</td><td>Output instances</td><td>Notes</td></tr><tr><td>deepseek/deepseek-v4-pro</td><td>15,493</td><td>Complete translation file.</td></tr><tr><td>qwen/qwen3.6-plus</td><td>15,493</td><td>Complete translation file.</td></tr><tr><td>x-ai/grok-4.3</td><td>15,493</td><td>Includes two persistent source-copy failure candidates.</td></tr><tr><td>Total</td><td>46,479</td><td>Three candidate translations for each Pali passage.</td></tr></table>

## 3.5 Embeddings and Full-Corpus Feature Table

All Pali source passages, English references, and candidate translations were embedded using google/gemini-embedding-2-preview through OpenRouter. This model family was selected because Gemini embeddings report strong performance on large embedding benchmarks such as MTEB/MMTEB, including multilingual and retrieval-oriented tasks, and because preliminary retrieval tests indicated better behavior than the other candidate embedding models tested (Muennighoff et al., 2023; Lee et al., 2025). The resulting SQLite cache contains 77,465 embeddings: Pali source, English reference, DeepSeek candidate, Qwen candidate, and Grok candidate for each of the 15,493 passages.

The full-corpus feature table contains one row per translation instance. The features fall into five groups:

1. Source-only features: source novelty index and related rarity/formulaicity components.

2. Candidate-source features: embedding distance between Pali source and English candidate, and source/candidate length ratios.

3. Peer-consensus features: distance from the candidate translation to the centroid of the other two AI translations of the same passage.

4. Reference-aided diagnostic features: distance between candidate translation and English (Sujato) reference, used only for analysis and not as a no-reference detector.

5. Hard anomaly flags: persistent source-copy or structurally invalid translation failures.

Peer-centroid distance is a no-human-reference signal. For each passage and candidate model, the other two AI translations are embedded and averaged to form a peer centroid. The candidate’s cosine distance from that centroid measures how much it diverges from the other AI renderings. This use of peer translations is related to multi-hypothesis MT evaluation, where model-output variability can substitute for or complement human-reference variation (Fomicheva et al., 2020). It is a candidate-specific adequacy proxy, but it is not a low-effort signal in a single-translation deployment because it requires multiple translations of the same source passage. A normalized peer-drift variant was also computed, but the raw candidateto-peer-centroid distance is preferred because only two peer translations are available and the denominator of a normalized two-peer envelope can be unstable.

## 3.6 Calibration Sample

The $^ { 4 6 , 4 7 9 }$ translation instances were ranked by an initial risk score:

Table 3: Calibration sample design.
<table><tr><td>Calibration stratum</td><td>Definition</td><td>Pool size</td><td>Sample size</td></tr><tr><td>Very high</td><td>Literal top 500 by initial risk rank, with hard anomalies sorted first</td><td>500</td><td>500</td></tr><tr><td>High</td><td>Remaining ranks after top 500 through top 20%</td><td>8,796</td><td>500</td></tr><tr><td>Upper mid</td><td>20-40% risk percentile range</td><td>9,296</td><td>500</td></tr><tr><td>Medium</td><td>40-60% risk percentile range</td><td>9,296</td><td>500</td></tr><tr><td>Lower mid</td><td>60-80% risk percentile range</td><td>9,296</td><td>500</td></tr><tr><td>Low</td><td>Bottom 20%</td><td>9,295</td><td>500</td></tr><tr><td colspan="2">Total</td><td>46,479</td><td>3,000</td></tr></table>

z(source\_novelty\_index)  
+ z(candidate\_distance\_to\_peer\_centroid)  
+ z(source\_candidate\_distance)

Hard anomaly flags (e.g. source-copy failures) were forced into the top-ranked tail. This preliminary score was used only to construct a calibration sample broad enough to estimate both high-risk and low-risk behavior. It was not treated as the final detector.

The calibration set contains 3,000 translation instances and was sampled with fixed seed 20260519. The top 500 highest-risk instances were included deterministically as a very-highrisk stratum. The remaining 2,500 items were sampled as 500-item strata from the rest of the top 20%, the 20-40% range, the 40-60% range, the 60-80% range, and the bottom 20% of the full risk ranking. Sampling was stratified by translator model and novelty band where possible. This design deliberately overrepresents high-risk material while still covering the full score distribution.

The calibration sample contains 987 DeepSeek translations, 1,030 Qwen translations, and 983 Grok translations. By source novelty band, it contains 863 low-novelty, 852 middle-novelty, and 1,285 high-novelty instances. The high-novelty overrepresentation is expected because high novelty was part of the initial risk score and the top-ranked tail was intentionally enriched. Counted item-wise, the 3,000 sampled translation instances contain 17,104 Pali source segments, 1.16 million source characters, 129,902 whitespace-delimited source words, and 670,300 cl100k\_base source tokens, while representing 2,587 unique Pali passages because some passages are sampled with more than one candidate model.

## 3.7 Error Adjudication

Each calibration item was adjudicated by three LLM judges: openai/gpt-5.5, google/ gemini-3.1-pro-preview, and anthropic/claude-sonnet-4.6. Judges saw the Pali source, the English reference as an adjudication aid, and the candidate English translation. They did not see the translator model identity, risk score, novelty band, GEMBA score, or any other metrics. This follows the growing use of strong LLMs as scalable evaluators while retaining the need to check them against human judgment because LLM evaluators can exhibit systematic biases (Zheng et al., 2023; Wang et al., 2024).

The judge task was binary at the primary level: decide whether the candidate is a valid translation variation or contains a translation error. If an error was present, the judge also assigned severity:

• Minor error: a local or limited issue that would be worth correcting but does not materially mislead the reader about the passage’s main meaning.

• Major error: an error that materially changes the meaning, omits essential content, reverses polarity, assigns agency or roles incorrectly, adds unsupported content, or mishandles an important doctrinal or contextual term.

Judges returned structured JSON. One API call was made per judge per passage. The full calibration adjudication therefore used 9,000 judge calls. All calls completed with valid schema output.

Panel labels were aggregated by majority vote. A translation was labeled ERROR if at least two judges labeled it as an error. Among panel-error items, it was labeled MAJOR if at least two judges assigned major severity; otherwise it was labeled MINOR. A translation was labeled VALID if fewer than two judges labeled it as an error.

## 3.8 Source-Prior and Embedding-Based Triage Signals

The first group of triage signals does not invoke additional LLM judges after translations and embeddings are available. Three signals were evaluated singly and in combination: source novelty as a source-only difficulty prior, source-candidate embedding distance as a candidateaware source-to-translation distance, and peer-centroid distance as a measure of divergence from the other AI translations of the same passage. These signals differ in deployment cost: source novelty can be computed before translation, source-candidate distance requires one candidate translation and embeddings, and peer-centroid distance requires multiple candidate translations.

The initial equal-weight score used all three features. Subsequent exploration tested each feature alone and transparent z-scored linear combinations. A simple cross-validation-selected source-novelty-plus-peer score was retained as the main refined routing rule:

refined\_source\_peer\_score =

0.7 \* z(source\_novelty\_index)

\+ 0.3 \* z(candidate\_distance\_to\_peer\_centroid)

The weight search evaluated simple z-scored linear combinations of source novelty, peercentroid distance, and source-candidate distance, selecting weights by held-out major-error recall at fixed review budgets. Grouped cross-validation by Pali passage id was used so that translations of the same Pali passage could not appear in both training and held-out folds. Small learned models, including logistic regression and shallow tree-based classifiers, were also tested as exploratory baselines but were not chosen as the primary method because they did not clearly outperform the transparent novelty-heavy score. A small exploratory run with a public pruned COMETKiwi no-reference QE checkpoint produced near-random ranking performance on the calibration and author-anchor labels, so it was not retained as a main baseline.

## 3.9 Higher-Effort Candidate-Aware Checks

The second group contains higher-effort reference-free checks intended for cases where a passage has already been routed for deeper scrutiny, or where budget allows a more intensive audit. These checks are candidate-aware and require additional LLM calls.

## 3.9.1 Backtranslation

Each calibration candidate was translated back from English into Pali using openai/gpt-5.5. The backtranslation prompt instructed the model to translate the meaning of the candidate translation, avoid reconstructing unseen source text, preserve named persons, places, numbers, lists, negation, agents, and doctrinal terms, and return only a JSON object containing the Pali backtranslation. This produced complete backtranslations for all 3,000 calibration items.

Backtranslation risk was measured by comparing the original Pali source to the backtranslated Pali. Several lexical metrics were tested: chrF risk, character 5-gram weighted Jaccard risk, content-token weighted Jaccard risk, and recall-like variants focused on rare tokens or n-grams. Higher risk means the backtranslation preserves less source-specific Pali material.

## 3.9.2 GEMBA No-Reference Scoring

GEMBA-style no-reference scoring asks an LLM to assign a direct quality score to a translation given only the source and candidate translation. The prompt was based on the no-reference GEMBA-DA prompt (Kocmi and Federmann, 2023a), with an added numeric-only output constraint for reliable parsing. Scores were interpreted on a 0-100 scale where higher scores indicate better translation quality. Risk features were then computed as 100 - score or corresponding summaries across multiple scorers.

We used GEMBA-DA rather than GEMBA-MQM because the present task is budgeted triage: translations must be ranked by review priority under fixed review fractions. A scalar direct-assessment score is therefore easier to calibrate as a risk signal than MQM annotation. GEMBA-MQM is closely related prior work for diagnostic error-span detection without references (Kocmi and Federmann, 2023b), but in this study diagnostic labeling is handled separately by the LLM adjudication panel and the author anchor.

Two GEMBA panels were tested. The "strong" GEMBA panel used openai/gpt-5.5, google/gemini-3.1-pro-preview, and anthropic/claude-sonnet-4.6. The "peer" GEMBA panel used the translator models themselves where possible, excluded self-scoring, and added moonshotai/kimi-k2.6 so that each candidate still received three "peer" panel scores. Thus, a DeepSeek translation was scored by Qwen, Grok, and Kimi; a Qwen translation by DeepSeek, Grok, and Kimi; and a Grok translation by DeepSeek, Qwen, and Kimi.

For each panel, the primary GEMBA feature is mean risk across the three available scores. Several derived features were also computed as exploratory checks, including median risk, lower-two mean risk, minimum-score risk, maximum-score risk, and score range. Mean risk is the main reported ranking signal; the other summaries test whether alternative aggregation or scorer disagreement adds information beyond the average score.

## 3.10 Author Anchor Set

Because the 3,000-item calibration labels are produced by LLM judges, a 500-item authoradjudicated anchor set was created with fixed seed 20260523 to test whether the LLM panel and GEMBA risk signals align with human judgment. The anchor set was diagnostically enriched rather than prevalence-balanced. It was sampled to stress the most important boundary cases: "strong" GEMBA high-risk items, "strong" GEMBA low-risk items, LLM panel errors outside the "strong" GEMBA top 20%, "strong" versus "peer" GEMBA disagreements, high-source-novelty items with low GEMBA risk, and random stratified controls.

The author adjudicator has formal training in Pali and saw the same core evidence as the LLM judges: Pali source, Sujato reference aid, and candidate translation. The author anchor set was labeled using the same three outcome labels: VALID, MINOR, and MAJOR.

Table 4: Author anchor sampling design.
<table><tr><td>Anchor group</td><td>Items</td></tr><tr><td>&quot;strong&quot; GEMBA high risk and LLM panel major</td><td>80</td></tr><tr><td>&quot;strong&quot; GEMBA high risk and LLM panel valid 70 or minor</td><td></td></tr><tr><td>Outside &quot;strong&quot; GEMBA top 20% but LLM panel 100 error</td><td></td></tr><tr><td>&quot;strong&quot; GEMBA low risk and LLM panel valid</td><td>70</td></tr><tr><td>&quot;strong&quot; versus &quot;peer&quot; GEMBA disagreement</td><td>80</td></tr><tr><td>High source novelty but low &quot;strong&quot; GEMBA risk</td><td>50</td></tr><tr><td>Random stratified controls</td><td>50</td></tr><tr><td>Total</td><td>500</td></tr></table>

The anchor set is not used to estimate corpus prevalence. Its purpose is to test whether the LLM panel and the strongest risk metrics remain informative when checked against a human adjudicator.

## 3.11 Statistical Analysis

The main labels are three-class severity labels (VALID, MINOR, MAJOR) and binary reductions (ERROR versus VALID, MAJOR versus non-major). Error-rate estimates are reported as proportions with Wilson score intervals where appropriate. Stratified full-corpus prevalence estimates weight each calibration stratum by its actual size in the 46,479-instance corpus.

Ranking metrics are evaluated by area under the receiver operating characteristic curve (AUC), review-budget recall, and review precision. AUC is used as a threshold-independent ranking measure: it estimates how often a randomly chosen error receives a higher risk score than a randomly chosen non-error. Review-budget curves ask: if only the top 1%, 5%, 10%, or 20% of items under a given risk score are reviewed, what share of known major errors are captured, and how many reviewed items are actually errors? Each metric is evaluated under its own ranking: for example, a 10% GEMBA budget means selecting the highest-risk 10% by GEMBA score, while a 20% source-novelty budget means selecting the highest-risk 20% by source novelty.

Because the calibration design oversamples high-risk items, raw percentages in the 3,000- item calibration set should not be interpreted as corpus prevalence. Stratified estimates are used when making full-corpus claims. Conversely, metric comparisons inside the 3,000-item set are treated as calibrated ranking comparisons, not direct prevalence estimates.

## 4 Results

## 4.1 Calibration Labels and Error Enrichment

The LLM panel labeled 207 of 3,000 calibration items as major errors (6.9%, 95% Wilson interval 6.0-7.9%), 337 as minor errors, and 2,456 as valid variations. Total panel-labeled errors were 544 of 3,000 (18.1%). Because the calibration sample intentionally oversampled the high-risk tail, these raw rates are not corpus prevalence estimates.

Error rates vary strongly by risk stratum. The top 500 very-high-risk items have a panelmajor rate of 28.0% and any-error rate of 49.6%. All other strata have much lower major-error rates, ranging from 1.4% to 4.6%. The result confirms that the initial no-reference risk ranking concentrated major errors in the extreme tail, but also shows that non-tail strata still contain errors.

Table 5: LLM-panel error rates by calibration stratum.
<table><tr><td>Stratum</td><td>n</td><td>Major error % (95% CI)</td><td>Minor error % (95% CI)</td><td>Any error % (95% CI)</td></tr><tr><td>Very high</td><td>500</td><td>28.0% (24.2-32.1)</td><td>21.6% (18.2-25.4)</td><td>49.6% (45.2-54.0)</td></tr><tr><td>High</td><td>500</td><td>4.6% (3.1-6.8)</td><td>12.2% (9.6-15.4)</td><td>16.8% (13.8-20.3)</td></tr><tr><td>Upper mid 20-40</td><td>500</td><td>3.4% (2.1-5.4)</td><td>9.0% (6.8-11.8)</td><td>12.4% (9.8-15.6)</td></tr><tr><td>Medium 40-60</td><td>500</td><td>1.4% (0.7-2.9)</td><td>9.2% (7.0-12.1)</td><td>10.6% (8.2-13.6)</td></tr><tr><td>Lower mid 60-80</td><td>500</td><td>2.2% (1.2-3.9)</td><td>7.2% (5.2-9.8)</td><td>9.4% (7.1-12.3)</td></tr><tr><td>Low bottom 20</td><td>500</td><td>1.8% (0.9-3.4)</td><td>8.2% (6.1-10.9)</td><td>10.0% (7.7-12.9)</td></tr></table>

Table 6: Stratified full-corpus prevalence estimates from LLM-panel labels.
<table><tr><td>Label</td><td>Estimated count in 46,479 instances</td><td>Estimated prevalence</td></tr><tr><td>Major error</td><td>1,362.6</td><td>2.9% (2.3-3.5)</td></tr><tr><td>Minor error</td><td>4,304.5</td><td>9.3% (8.2-10.3)</td></tr><tr><td>Any error</td><td>5,667.1</td><td>12.2% (11.0-13.4)</td></tr></table>

After weighting each stratum by its full-corpus size, the estimated panel-labeled fullcorpus major-error prevalence is 2.9% (approximate 95% CI 2.3-3.5%). The estimated minorerror prevalence is 9.3% (8.2-10.3%), and the estimated any-error prevalence is 12.2% (11.0- 13.4%).

Broad risk-boundary estimates first evaluate the original preliminary risk rank used to construct the calibration sample. This rank combined three z-scored components with equal weight: source novelty, candidate distance from the peer-translation centroid, and sourcecandidate embedding distance, with hard anomaly flags forced into the top-ranked tail. Its purpose was to create a calibration sample enriched for likely errors while still covering the full risk distribution; it was not assumed to be the final detector. Under this original rank, reviewing the top 20% of the full corpus would capture an estimated 40.0% of panelmajor errors and 30.5% of all panel errors. Reviewing the top 40% would capture 63.2% of panel-major errors. These estimates demonstrate useful enrichment, but also show that the preliminary rank is not sufficient as a final detector. Detailed broad-boundary estimates are reported in Appendix Table A2.

## 4.2 Source-Prior and Peer-Translation Routing

The first group of signals asks a simple question: can we rank translations so that a limited review budget sees more errors than random review would? These signals are "reference-free" in the deployment sense: they do not use a human translation of the passage being audited. They differ, however, in cost. Source novelty is source-only and can be computed before translation. Peer-centroid distance requires several independent AI translations of the same Pali passage and therefore measures candidate disagreement, not source difficulty alone. Source-candidate embedding distance and length anomaly are candidate-level checks, but in this experiment they were weaker.

Table 7: Leading source-prior and embedding-based scores at a 20% review budget.
<table><tr><td>Score</td><td>Major recall</td><td>Minor recall</td><td>Any-error recall</td><td>Major precision</td><td>Any-error precision</td><td>Reviewed non-error share</td></tr><tr><td>Refined source- novelty-plus- peer score: 0.7 novelty + 0.3</td><td>60.6%</td><td>31.3%</td><td>38.4%</td><td>8.9%</td><td>23.4%</td><td>76.6%</td></tr><tr><td>peer Source novelty only</td><td>48.7%</td><td>35.9%</td><td>39.0%</td><td>7.1%</td><td>23.8%</td><td>76.2%</td></tr><tr><td>Peer-centroid distance only</td><td>40.9%</td><td>28.4%</td><td>31.4%</td><td>6.0%</td><td>19.2%</td><td>80.8%</td></tr><tr><td>Original equal</td><td>39.8%</td><td>27.4%</td><td>30.4%</td><td>5.8%</td><td>18.5%</td><td>81.5%</td></tr><tr><td>composite Source- candidate</td><td>36.5%</td><td>20.4%</td><td>24.3%</td><td>5.4%</td><td>14.8%</td><td>85.2%</td></tr><tr><td>embedding distance only Length</td><td>25.9%</td><td>18.4%</td><td>20.2%</td><td>3.8%</td><td>12.3%</td><td>87.7%</td></tr></table>

Table 7 reports stratified full-corpus estimates of recall, precision, and the reviewed nonerror share at a 20% review budget. Source novelty alone captured an estimated 48.7% of panel-major errors. Peer-centroid distance alone captured 40.9%. The original equalweight composite (source novelty, peer-centroid distance, and source-candidate embedding distance) captured about the same amount, 39.8%, because it gave too much weight to weaker components.

The best transparent routing score was a novelty-heavy combination: 0.7 source novelty plus 0.3 peer-centroid distance. This score captured an estimated 60.6% of panel-major errors, 31.3% of minor errors, and 38.4% of all errors at a 20% review budget. Its major-error precision was only 8.9%, so it is not a reliable stand-alone error detector. Its value is as a routing rule: source novelty identifies passages that are intrinsically harder or less formulaic, while peercentroid distance adds evidence that one candidate translation diverges from other model renderings of the same source.

Cross-validation supported this weighting direction but also showed that the result should not be overinterpreted. When passages were grouped so that translations of the same Pali passage could not appear in both training and held-out folds, the selected weight was usually novelty-heavy, and the source-candidate embedding component was usually dropped. Heldout major-error recall at a 20% review budget averaged 54.9%, with substantial variation across folds. Small learned models, including logistic regression and shallow tree-based models, reached similar but not clearly better performance. Given the limited number of major-error labels, the transparent 0.7 novelty plus 0.3 peer-distance score is preferable to a learned detector at this stage.

## 4.3 Backtranslation

Backtranslation was tested as a separate candidate-aware check. Each English candidate in the 3,000-item calibration set was translated back into Pali, and the backtranslated Pali was compared with the original source. The intuition is that if the English translation omits or distorts source-specific material, a round-trip backtranslation may preserve less of the original Pali wording.

Backtranslation was only performed on the 3,000-item calibration set. This set is not a normal random slice of the corpus. It was deliberately enriched with high-risk items, especially the top-risk tail. That makes the sample easier for many risk scores to rank, because it contains many obvious or semi-obvious high-risk cases. For this reason, the backtranslation results are reported only as within-sample comparisons. They should not be read as full-corpus deployment estimates.

As a standalone metric, chrF backtranslation risk was the strongest backtranslation variant tested. It achieved major-vs-nonmajor AUC 0.824. At a 20% review budget within the enriched calibration set, it captured 67.1% of panel-major errors, with 23.2% major-error precision. This indicates that round-trip lexical loss contains real error signal.

However, backtranslation did not clearly outperform the simpler source-novelty-plus-peer score on the same 3,000 items. In this within-sample comparison, the 0.7 source novelty plus 0.3 peer-centroid score achieved AUC 0.820 and captured 71.0% of panel-major errors at a 20% review budget, with 24.5% major-error precision. Adding chrF backtranslation risk increased AUC only to 0.832 and major-error recall only to 72.5%, with 25.0% major-error precision. The gain is therefore small relative to the extra LLM calls required.

Overlap analysis explains the small marginal gain. At the same 20% budget, chrF backtranslation risk caught 139 major errors, while the source-novelty-plus-peer score caught 147. Of these, 128 were the same errors. Backtranslation therefore appears to be useful supporting evidence, but not a central detector in this experiment. Detailed curves are reported in Appendix Table A3.

## 4.4 "Strong" GEMBA Scoring

No-reference GEMBA scoring, also computed only on the 3,000-item calibration set, was the strongest signal tested. Mean GEMBA risk from the "strong" panel achieved AUC 0.970 for panel-major versus non-major, 0.985 for major versus valid, and 0.910 for any error versus valid. The score distribution separated labels clearly: panel-major errors had mean GEMBA score 69.3, panel-minor errors 87.0, and panel-valid translations 95.2.

At a 5% review budget, GEMBA mean risk captured 59.4% of panel-major errors with 82.0% major-error precision and 98.7% any-error precision. At 10%, it captured 81.6% of panel-major errors with 56.3% major-error precision. At 20%, it captured 94.7% of panel-major errors and 71.9% of all panel errors. These results are much stronger than the source-prior, embedding-based, and backtranslation results.

A second way to use GEMBA is as a red-flag rule: mark a translation for review if even one "strong" GEMBA scorer gives it a low score. This is different from ranking by the average GEMBA score. With a cutoff of 60 or lower from any scorer, only 4.5% of the calibration sample was flagged. This small flagged set was very concentrated: 80.1% of flagged items were panel-major errors and 96.3% were some kind of panel-labeled error. However, because the rule is strict, it found only 52.7% of all panel-major errors. A looser cutoff of 70 flagged 6.3% of the sample and found 65.2% of panel-major errors, but precision fell to 71.1%. Thus, a single very low GEMBA score is a strong warning sign, but mean GEMBA risk is better when the goal is to rank all translations under a fixed review budget.

Disagreement alone was less informative. GEMBA score range had AUC 0.810 for major versus non-major, far below mean risk. A range of at least 20 selected 13.9% of the sample and captured 57.5% of panel-major errors, but with only 28.5% major-error precision. High disagreement without an actually low score was especially weak. The main GEMBA signal is therefore low perceived quality, not merely scorer disagreement.

Combining GEMBA with source-novelty-plus-peer routing or backtranslation did not improve over GEMBA mean risk. For example, GEMBA mean risk alone had major AUC 0.970, while GEMBA mean risk plus the source-novelty-plus-peer score had AUC 0.935 and GEMBA mean risk plus backtranslation plus the source-novelty-plus-peer score had AUC 0.920. Once "strong" GEMBA scores are available, the other signals add little to ranking performance. Their main role is upstream: deciding which items should receive higher-effort GEMBA scoring.

Table 8: "Strong" GEMBA ranking performance on the 3,000-item calibration set. AUC is a property of the full ranking and therefore repeats across budget cutoffs for the same score; recall and precision are calculated at the listed review budget.
<table><tr><td>Metric</td><td>Major AUC</td><td>Error-vs- valid AUC</td><td>Budget</td><td>Major recall</td><td>Major precision</td><td>Any-error recall</td><td>Any-error precision</td></tr><tr><td>GEMBA mean risk, top 5%</td><td>0.970</td><td>0.910</td><td>5%</td><td>59.4%</td><td>82.0%</td><td>27.2%</td><td>98.7%</td></tr><tr><td>GEMBA mean risk, top 10%</td><td>0.970</td><td>0.910</td><td>10%</td><td>81.6%</td><td>56.3%</td><td>48.2%</td><td>87.3%</td></tr><tr><td>GEMBA mean risk, top 20%</td><td>0.970</td><td>0.910</td><td>20%</td><td>94.7%</td><td>32.7%</td><td>71.9%</td><td>65.2%</td></tr><tr><td>GEMBA minimum- score risk,</td><td>0.948</td><td>0.875</td><td>10%</td><td>77.3%</td><td>53.3%</td><td>43.8%</td><td>79.3%</td></tr><tr><td>top 10% GEMBA score range, top 10%</td><td>0.810</td><td>0.774</td><td>10%</td><td>45.9%</td><td>31.7%</td><td>30.7%</td><td>55.7%</td></tr></table>

## 4.5 GEMBA, Source Novelty, and Translator Effects

One concern was that GEMBA might fail on high-novelty Pali because those same passages are difficult for both translators and judges. The data did not support this concern. Source novelty correlated moderately with GEMBA risk (r = 0.408) and GEMBA score range (r = 0.374), indicating that rarer and less formulaic passages do tend to receive lower and more variable scores. However, GEMBA remained highly discriminative within each novelty band. Mean-risk AUC for panel-major versus non-major was 0.978 in the low-novelty band, 0.949 in the middle band, and 0.960 in the high band.

Adding source novelty to GEMBA worsened performance. Major-error AUC fell from 0.970 for GEMBA mean risk alone to 0.968 with a 0.9/0.1 GEMBA/novelty blend, 0.954 with a 0.7/0.3 blend, and 0.926 with an equal blend. Source novelty also caught few major errors missed by GEMBA at equal budget: at 10%, source novelty found only seven major errors not already caught by GEMBA; at 20%, it found only two. Thus, source novelty is useful as a source-side prior, but it should not be used as a heavy correction once GEMBA scores are available. It identifies passages on which machine translation is more likely to fail, not whether a particular translation has failed.

GEMBA also remained strong by translator. "strong" GEMBA mean risk achieved majorvs-nonmajor AUC 0.981 on DeepSeek candidates, 0.968 on Grok candidates, and 0.963 on Qwen candidates. The calibration set contained fewer DeepSeek major errors (31) than Grok (57) or Qwen (119), so per-translator estimates should be read with different uncertainty. Still, the signal is not confined to one candidate model. Full per-translator GEMBA results are reported in Appendix Table A4.

## 4.6 Is GEMBA Just Strong-Model Auditing?

The "strong" GEMBA panel used models that are plausibly stronger than the translator models. To test whether the result was simply a strong-model-judges-weak-models artifact, the same 3,000 calibration items were scored with a "peer" GEMBA panel built from the translator-model families themselves: DeepSeek, Qwen, and Grok, with Kimi 2.6 added so that self-scoring could be excluded while still giving each item three scores. "peer" GEMBA was substantially weaker than "strong" GEMBA, but it still remained clearly useful.

Table 9: "Strong" versus "peer" GEMBA.
<table><tr><td>Panel</td><td>Budget</td><td>Major AUC</td><td>Error-vs- valid AUC</td><td>Major recall</td><td>Major precision</td><td>Any-error recall</td><td>Any-error precision</td></tr><tr><td>&quot;strong&quot; GEMBA</td><td>5%</td><td>0.970</td><td>0.910</td><td>59.4%</td><td>82.0%</td><td>27.2%</td><td>98.7%</td></tr><tr><td>&quot;peer&quot; GEMBA</td><td>5%</td><td>0.885</td><td>0.799</td><td>43.5%</td><td>60.0%</td><td>23.2%</td><td>84.0%</td></tr><tr><td>&quot;strong&quot; GEMBA</td><td>10%</td><td>0.970</td><td>0.910</td><td>81.6%</td><td>56.3%</td><td>48.2%</td><td>87.3%</td></tr><tr><td>&quot;peer&quot; GEMBA</td><td>10%</td><td>0.885</td><td>0.799</td><td>61.4%</td><td>42.3%</td><td>36.4%</td><td>66.0%</td></tr><tr><td>&quot;strong&quot; GEMBA</td><td>20%</td><td>0.970</td><td>0.910</td><td>94.7%</td><td>32.7%</td><td>71.9%</td><td>65.2%</td></tr><tr><td>&quot;peer&quot; ĠEMBA</td><td>20%</td><td>0.885</td><td>0.799</td><td>79.2%</td><td>27.3%</td><td>56.6%</td><td>51.3%</td></tr></table>

Mean-risk AUC for panel-major versus non-major fell from 0.970 with the "strong" panel to 0.885 with the "peer" panel. Error-versus-valid AUC fell from 0.910 to 0.799. At a 10% review budget, "strong" GEMBA captured 81.6% of panel-major errors, while "peer" GEMBA captured 61.4%. At 20%, the corresponding values were 94.7% and 79.2%.

Overlap analysis shows that "peer" GEMBA mostly catches the same major errors as "strong" GEMBA, rather than adding a large independent set. Under the scoring-specific budget rule described above, "strong" GEMBA caught 169 panel-major errors and "peer" GEMBA caught 127 at a 10% budget; 122 were shared, five were peer-unique, and 47 were strong-unique. At a 20% budget, "peer" GEMBA added only two major errors not caught by "strong" GEMBA.

Inter-evaluator rank consistency was moderate across both GEMBA panels. The "strong" panel showed its highest rank agreement between GPT-5.5 and Claude Sonnet 4.6 (ρ = 0.731), while "peer" panel agreement was weaker and more uneven, especially for Qwen/Grok (ρ = 0.437). This supports using averaged GEMBA scores rather than relying on a single scorer, and it reinforces the conclusion that scorer choice matters.

These results support two claims. First, GEMBA is judge-dependent: stronger scorer models produce a substantially better triage signal. Second, the prompt and scoring method still carry real no-reference information, because the weaker "peer" panel remained meaningfully above the source-prior, embedding-based, and backtranslation signals.

## 4.7 Human Author-Anchor Validation

The 500-item author anchor provides an independent check on the LLM-panel labels and GEMBA risk scores. Because the anchor set is diagnostically enriched, its raw label distribution is not a prevalence estimate. The author labeled 310 items valid, 99 minor errors, and 91 major errors.

The LLM panel showed high sensitivity but conservative overcalling relative to the author labels. Three-class exact agreement was 72.4%, with Cohen’s kappa 0.559. Binary error recall was 97.4%, meaning that the panel missed very few author-labeled errors. Binary error precision was lower, 64.5%, because many author-valid items were labeled minor errors by the panel. Major-error recall was 93.4% and major-error precision 72.6%.

Table 10: Pairwise Spearman rank correlations between no-reference GEMBA evaluator scores. Each cell reports Spearman’s ρ, with the number of co-rated calibration items in parentheses. The "strong" GEMBA panel scored all 3,000 calibration items. The "peer" GEMBA panel excluded self-scoring, so co-rated counts vary. Blank cells indicate evaluator pairs that did not belong to the same GEMBA scoring panel.
<table><tr><td>Evaluator</td><td>GPT-5.5</td><td>Gemini 3.1 Pro</td><td>Claude Sonnet 4.6</td><td>DeepSeek V4 Pro</td><td>Qwen3.6 Plus</td><td>Grok 4.3</td><td>Kimi K2.6</td></tr><tr><td>GPT-5.5</td><td>1.000 (3,000)</td><td>0.612 (3,000)</td><td>0.731 (3,000)</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>0.612 (3,000)</td><td>1.000 (3,000)</td><td>0.569 (3,000)</td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Sonnet 4.6</td><td>0.731 (3,000)</td><td>0.569 (3,000)</td><td>1.000 (3,000)</td><td></td><td></td><td></td><td></td></tr><tr><td>DeepSeek</td><td></td><td></td><td></td><td>1.000</td><td>0.595</td><td>0.544</td><td>0.581</td></tr><tr><td>V4 Pro</td><td></td><td></td><td></td><td>(2,013)</td><td>(983)</td><td>(1,030)</td><td>(2,013)</td></tr><tr><td>Qwen3.6 Plus</td><td></td><td></td><td></td><td>0.595</td><td>1.000</td><td>0.437</td><td>0.510</td></tr><tr><td>Grok 4.3</td><td></td><td></td><td></td><td>(983)</td><td>(1,970)</td><td>(987)</td><td>(1,970)</td></tr><tr><td></td><td></td><td></td><td></td><td>0.544</td><td>0.437</td><td>1.000</td><td>0.542</td></tr><tr><td>Kimi K2.6</td><td></td><td></td><td></td><td>(1,030)</td><td>(987)</td><td>(2,017)</td><td>(2,017)</td></tr><tr><td></td><td></td><td></td><td></td><td>0.581</td><td>0.510</td><td>0.542</td><td>1.000</td></tr><tr><td></td><td></td><td></td><td></td><td>(2,013)</td><td>(1,970)</td><td>(2,017)</td><td>(3,000)</td></tr></table>

Table 11: Author and LLM-panel labels on the 500-item anchor set. Overall exact author-panel agreement was 362/500 items (72.4%).
<table><tr><td>Label</td><td>Author count</td><td>LLM-panel count</td><td>Author-panel agreement</td></tr><tr><td>Valid</td><td>310</td><td>213</td><td>208 (67.1%)</td></tr><tr><td>Minor error</td><td>99</td><td>170</td><td>69 (69.7%)</td></tr><tr><td>Major error</td><td>91</td><td>117</td><td>85 (93.4%)</td></tr></table>

Table 12: GEMBA AUC against author labels.
<table><tr><td>Metric</td><td>Major vs non-major</td><td>Major vs valid</td><td>Error vs valid</td></tr><tr><td>&quot;strong&quot; GEMBA mean risk</td><td>0.924</td><td>0.966</td><td>0.931</td></tr><tr><td>&quot;strong&quot; GEMBA minimum-score risk</td><td>0.904</td><td>0.951</td><td>0.907</td></tr><tr><td>&quot;strong&quot; GEMBA score range</td><td>0.755</td><td>0.819</td><td>0.805</td></tr><tr><td>&quot;peer&quot; GEMBA mean risk</td><td>0.789</td><td>0.833</td><td>0.772</td></tr><tr><td>&quot;peer&quot; GEMBA minimum-score risk</td><td>0.747</td><td>0.774</td><td>0.709</td></tr><tr><td>&quot;peer&quot; GEMBA score range</td><td>0.674</td><td>0.702</td><td>0.649</td></tr></table>

Panel-major labels were substantially more reliable than panel-minor labels. Of 117 panelmajor items in the anchor set, 85 were author-major, 26 were author-minor, and six were author-valid. In contrast, panel-minor labels included many items the author considered valid. This pattern is important for interpreting the 3,000-item calibration results: the LLM panel is a high-sensitivity triage labeler, not an expert oracle, and it tends to overcall minor errors. At the same time, the boundaries between valid variation and minor error, and between minor and major error, are partly judgment-dependent, especially in interpretively open passages.

Individual judges showed similar performance profiles. GPT-5.5 had exact agreement 73.0% and kappa 0.569; Gemini 3.1 Pro had exact agreement 71.0% and kappa 0.525; Claude Sonnet 4.6 had exact agreement 72.6% and kappa 0.530. GPT-5.5 and Gemini had higher majorerror recall, while Claude had somewhat higher binary error precision. Full individual-judge metrics are reported in Appendix Table A5.

## 4.8 GEMBA Against the Author Anchor

"strong" GEMBA remained highly predictive when evaluated against author labels, although less strongly than against LLM-panel labels. Mean "strong" GEMBA risk achieved authoranchor AUC 0.924 for major versus non-major, 0.966 for major versus valid, and 0.931 for error versus valid. "peer" GEMBA was again weaker, with AUC 0.789 for major versus non-major, 0.833 for major versus valid, and 0.772 for error versus valid.

Within the author-anchor set, the top 10% of items by "strong" GEMBA risk captured 47.3% of author-major errors with 86.0% major-error precision and 96.0% any-error precision. Within the same set, the top 20% captured 73.6% of author-major errors with 67.0% majorerror precision and 97.0% any-error precision. These are diagnostic anchor-set results, not deployment-calibrated full-corpus budget estimates. "peer" GEMBA captured fewer authormajor errors and had lower precision at the same within-anchor budgets.

Single-low-score thresholds also remained useful on the author anchor as secondary warning rules. Mean GEMBA remains the main ranker, but a single very low score is operationally easy to interpret. If any "strong" GEMBA scorer assigned a score ≤ 60, the selected set contained 111 items (22.2% of the anchor), captured 75.8% of author-major errors, and had 62.2% major-error precision and 90.1% any-error precision. A stricter threshold of ≤ 50 selected 81 items, captured 62.6% of author-major errors, and had 70.4% major-error precision. Additional threshold results are reported in Appendix Table A6.

Table 13: GEMBA mean-risk budget curves against author labels.
<table><tr><td>Panel</td><td>Budget</td><td>Major recall</td><td>Major precision</td><td>Any-error recall</td><td>Any-error precision</td></tr><tr><td>&quot;strong&quot; GEMBA</td><td>5%</td><td>24.2%</td><td>88.0%</td><td>13.2%</td><td>100.0%</td></tr><tr><td>&quot;strong&quot; GEMBA</td><td>10%</td><td>47.3%</td><td>86.0%</td><td>25.3%</td><td>96.0%</td></tr><tr><td>&quot;strong&quot; &quot;GEMBA</td><td>20%</td><td>73.6%</td><td>67.0%</td><td>51.1%</td><td>97.0%</td></tr><tr><td>&quot;peer&quot; GEMBA</td><td>5%</td><td>18.7%</td><td>68.0%</td><td>13.2%</td><td>100.0%</td></tr><tr><td>&quot;peer&quot; GEMBA</td><td>10%</td><td>36.3%</td><td>66.0%</td><td>25.3%</td><td>96.0%</td></tr><tr><td>&quot;peer&quot; GEMBA</td><td>20%</td><td>56.0%</td><td>51.0%</td><td>44.7%</td><td>85.0%</td></tr></table>

The author-anchor results change the strength but not the direction of the main finding. Against LLM-panel labels, "strong" GEMBA appeared extremely strong. Against author labels, it remains the best reference-free signal tested, but with lower recall at the same within-anchor review fractions. This is a more realistic diagnostic check for human-facing use: "strong" GEMBA is powerful as a triage tool, but its performance depends on scorer strength and it does not eliminate the need for human review.

## 4.9 Cross-Method Summary

Across the tested reference-free signals, the strongest separation came from no-reference GEMBA scoring by the "strong" evaluator panel. Source novelty, peer-centroid distance, and source-novelty-plus-peer composites enriched for major errors but did not approach GEMBAlevel precision or recall. Backtranslation added measurable but modest signal over the source-novelty-plus-peer score. "Peer" GEMBA remained useful but was consistently weaker than "strong" GEMBA, showing that evaluator choice substantially affects performance.

The result pattern is therefore ordered rather than binary: source-only and embeddingbased signals are useful for triage; backtranslation may be used as supporting evidence; "strong" GEMBA is the best candidate-aware signal tested, while "peer" GEMBA (in case stronger models are not available) is still clearly useful. The discussion sketches how these signals could inform a practical review workflow, while noting that the complete workflow is not evaluated as a single end-to-end cascade.

## 5 Discussion

## 5.1 What "No-Reference" Means Here

The central methodological distinction is that the system is no-reference at inference time, not no-reference in the entire research pipeline. The English translation is used to help produce calibration labels and author-anchor labels, but it is not used to compute any of the risk signals tested for a newly translated passage. This is the condition that makes the results relevant to untranslated or under-translated classical material: the reference functions as a measuring instrument for the experiment, not as an input to the proposed triage system.

This distinction also keeps the study from treating one English translation as the definition of correctness. Reference-based metrics such as BLEU, BERTScore, and COMET have been central to machine translation evaluation (Papineni et al., 2002; Zhang et al., 2020; Rei et al., 2020), but they presuppose at least one reference translation. For Pali and other classical languages, the motivating use case is precisely where that reference may not exist. Moreover, legitimate translation variation is not noise: translation studies has long emphasized that translation is interpretive, stylistically situated, and shaped by audience and purpose (Venuti, 2008; Tymoczko, 2007).

## 5.2 Source Difficulty Is a Risk Prior, Not an Error Detector

One of the clearest findings is that source novelty is informative. Passages with rarer vocabulary, fewer repeated formulas, and lower neighborhood similarity were more likely to contain panel-major errors. The refined source-novelty-plus-peer score captured an estimated 60.6% of panel-major errors at a 20% full-corpus review budget, while source novelty alone captured an estimated 48.7%. This supports the intuition that some passages are intrinsically riskier for LLM translation before any candidate translation is inspected.

At the same time, source novelty must be interpreted narrowly. It does not detect whether a specific candidate translation is wrong. It estimates that a passage is difficult. This makes it useful for routing: a high-novelty passage may deserve extra scrutiny, multiple translations, or a stronger evaluation model. It does not justify marking the output as erroneous.

This result aligns with recent work on LLM translation of Ancient Greek technical prose. Zainaldin and colleagues found that terminology rarity was a strong predictor of catastrophic translation failure in Galenic texts, especially in passages with dense technical vocabulary (Zainaldin et al., 2026). Our Pali results are not identical in domain or measurement, but they point in the same direction: in classical-language translation, rare or less formulaic source material can be a substantial risk factor. This is encouraging for generalization, because source rarity and formulaicity are computable in many classical corpora even when no target-language reference exists.

Once "strong" GEMBA scores are available, adding source novelty makes the ranking worse rather than better. This means source novelty belongs upstream as a difficulty prior, not downstream as a correction to a strong candidate-aware evaluator. In practical terms, source novelty helps decide which passages deserve deeper checks; GEMBA helps judge the particular candidate translation.

## 5.3 Peer Agreement and Backtranslation Are Useful but Secondary

Peer-centroid distance is conceptually attractive because it uses other AI translations as a substitute for human references. This resembles multi-hypothesis MT evaluation, where variation among machine outputs can help model translation variability and partially replace reference variation (Fomicheva et al., 2020). In our results, peer-centroid distance did add signal: as a single feature it captured an estimated 40.9% of panel-major errors at a 20% fullcorpus review budget. Combined with source novelty, it contributed to the best transparent source-novelty-plus-peer routing score.

However, peer-centroid distance is not a cheap signal in the ordinary sense. It requires multiple translations of the same passage, plus embeddings. It is reference-free, but not cost-free. This matters for deployment. If a translation workflow already produces multiple model outputs, peer distance is a natural by-product. If it produces only one candidate, peer distance becomes an additional translation expense.

Backtranslation also carried signal, but only modestly. Round-trip lexical loss is intuitively relevant: if an English translation drops source-specific Pali material, a backtranslation may fail to recover it. In this study, chrF backtranslation risk plus the source-novelty-pluspeer score improved major-vs-nonmajor AUC from 0.820 to 0.832. At a 20% calibrationsample review budget, major recall increased only from 71.0% to 72.5%. This is evidence that backtranslation sees something real, but not enough to make it a central detector.

This finding is useful because it prevents overbuilding the system. Backtranslation is expensive, and its marginal gain was small. It may still be valuable in targeted settings, especially when a human reviewer wants another view of what semantic material survived the candidate translation. But as a scoring layer, it should remain supporting evidence rather than a coequal partner to GEMBA.

## 5.4 GEMBA Works, but It Is Judge-Dependent

The strongest candidate-aware signal was no-reference GEMBA. This is consistent with Kocmi and Federmann’s finding that prompted LLMs can be strong evaluators of translation quality in both reference-based and reference-free modes (Kocmi and Federmann, 2023a). In our calibration set, "strong" GEMBA mean risk reached AUC 0.970 for panel-major versus nonmajor and captured 81.6% of panel-major errors at a 10% review budget. These numbers are much stronger than source novelty, peer distance, or backtranslation.

The author anchor makes this result more credible but also more realistic. Against author labels, "strong" GEMBA mean risk remained the best signal tested, with AUC 0.924 for major versus non-major and 0.966 for major versus valid. However, within the author-anchor set, the top 10% by "strong" GEMBA risk captured only 47.3% of author-major errors. This gap is expected: the 3,000-item calibration labels are produced by an LLM panel, while the anchor checks those labels and scores against human judgment. The anchor therefore lowers the apparent strength of GEMBA, but it does not reverse the conclusion.

The "strong" versus "peer" GEMBA comparison also matters. When the scorer panel was changed from GPT-5.5, Gemini 3.1 Pro, and Claude Sonnet 4.6 to a "peer" panel closer to the translator set, performance dropped substantially. Major AUC fell from 0.970 to 0.885 against panel labels, and from 0.924 to 0.789 against author labels. This confirms that GEMBA is not model-independent. The prompt matters, but scorer strength matters too.

The appropriate conclusion is therefore not that "GEMBA is gold." It is narrower and more targeted: "strong" GEMBA is the best candidate-aware no-reference triage signal tested, but its performance depends on scorer strength and it does not remove the need for human review. This conclusion also fits the broader LLM-as-judge literature. Strong LLM judges can approximate human preferences surprisingly well in some settings, but documented biases and instability require calibration and human anchoring (Zheng et al., 2023; Wang et al., 2024).

## 5.5 The LLM Panel Is a Triage Labeler, Not an Oracle

The author-anchor analysis is important because it prevents circularity. If the only labels were LLM-panel labels, "strong" GEMBA could be criticized as merely predicting judgments made by related LLMs. The anchor shows that the panel and GEMBA scores are meaningfully aligned with author judgment, but also that the panel overcalls errors, especially minor ones.

This should be framed as a feature of the study design rather than a failure. The panel is used as a high-sensitivity triage labeler. It missed very few author-labeled errors in the anchor set: binary error recall was 97.4%, and major-error recall was 93.4%. The cost was lower precision. Many author-valid translations were labeled minor errors by the panel, and panel-minor labels were much less reliable than panel-major labels.

For a quality-oriented translation audit, this asymmetry is acceptable and even desirable. Missing a major error in a religious or philosophical text is more serious than over-referring a defensible variation for review. But it also means that panel-derived prevalence estimates should be interpreted as triage-prevalence estimates, not as final expert error rates.

## 5.6 Limitations

The study has several limitations. First, the system is no-reference at inference time, but not no-reference in its research design. Sujato’s English translation is used to support LLM and author adjudication. This is appropriate for calibration, but it means that the reported performance depends partly on the suitability of that reference-aided adjudication process.

Second, the 3,000-item calibration labels are produced by an LLM panel. The 500-item author anchor reduces the risk of circularity, but it is still a single-author check rather than an independent expert consensus panel. The author has formal Pali training, but is not an independent external expert, and the study does not measure inter-rater reliability among multiple Pali-trained readers. For that reason, panel-derived prevalence estimates should be read as triage-prevalence estimates, not final expert error rates.

Third, the best-performing signal, "strong" GEMBA, is model-dependent and computationally expensive. Its performance fell when "strong" scorer models were replaced with "peer" scorer models closer to the translator set. The conclusion is therefore not that any LLM can reliably judge any classical-language translation, but that "strong" no-reference LLM scoring can be highly informative when calibrated and human-anchored. In real-world deployments, however, the strongest available models may already be used for translation, leaving no clearly stronger evaluator for GEMBA scoring. In that setting, "peer" GEMBA may still be useful, but its weaker performance should be expected.

Fourth, the experiment is limited to Pali-to-English, three candidate translator models, one large segmented corpus, and passage-level translation. Some Pali passages depend on broader discourse context, formulaic ellipsis, or traditional interpretive background that a passage-level prompt may not provide. The findings should therefore be tested on other classical languages and on other text types before being treated as a general property of LLM translation.

Finally, AI translation of religious and philosophical texts has ethical stakes beyond technical accuracy. The proposed system is a tool for allocating review attention, not for assigning interpretive authority. Deployment on living traditions should include human translators, scholars, and relevant communities in deciding what counts as acceptable translation, what errors matter most, and how AI-generated text should be presented to readers. Any AI-assisted workflow in religious or philosophical domains should serve accessibility by broadening access to texts for which no human translation exists, not by replacing human expertise.

## 5.7 Implications for Classical-Language Translation Workflows

The results suggest a practical, but not yet end-to-end validated, workflow for AI-assisted classical-language translation. First, compute source novelty over the source corpus to identify passages that are likely to be difficult. Second, if the workflow permits multiple candidate translations, compute peer-centroid distance to identify outputs that diverge from other renderings. Third, use "strong" no-reference GEMBA scoring for the subset where review resources are available. Fourth, route high-risk items to human review, retranslation, or both.

This workflow is not a replacement for philological expertise. It is a workload allocation system. Its value is highest when the corpus is large, expert time is scarce, and the cost of missing a major error is high. That description fits many classical-language settings: Pali, Latin, Ancient Greek, Sanskrit, Coptic and other traditions where large bodies of untranslated texts exist but expert translators are few.

Current LLM translation work on classical languages is moving quickly. LITERA, for example, frames Latin-to-English LLM translation as a research-assistance system rather than a fully autonomous replacement for translators (Rosu, 2025). The Ancient Greek Galen study similarly shows that LLMs can produce high-quality translations for some passages while failing sharply on rare technical material (Zainaldin et al., 2026). Our Pali results fit this emerging pattern. LLMs are useful enough to justify serious evaluation infrastructure, but not reliable enough to be left unaudited.

The most general contribution of the present study is therefore methodological. The proposed design does not require human references for the texts being newly translated. It requires source text, candidate translations, and a calibration set where labels can be obtained with reference aid or human expertise. Once calibrated, the same pattern can be transferred: source-side risk prior, candidate-aware no-reference signals, "strong" GEMBA scoring, and human review of the flagged tail. For classical-to-modern translation, that may be a more realistic goal than fully automatic quality certification.

A natural next step is to turn the calibrated labels into a dedicated quality-estimation model for Pali or related classical languages. Domain-specific low-resource QE research suggests that lightweight adaptation can improve robustness when prompt-only evaluation is insufficient (Gurav et al., 2026). The present study does not train such a model; it provides the kind of labeled triage data and error-risk analysis that would make that future work possible.

## Acknowledgments

The author gratefully acknowledges SuttaCentral and Bhikkhu Sujato for their translation work and for making Pali texts and English translations accessible in a structured digital format. Their work made this study possible.

## Funding

This study received no external funding and was self-funded by the author.

## Data Availability

The Pali source texts and Bhikkhu Sujato’s corresponding English reference translations are available through SuttaCentral’s public GitHub repository at https://github.com/ suttacentral/bilara-data; Sujato’s translations are released under a Creative Commons Zero (CC0) licence. A reproducibility package for this study, including the full underlying data generated for the analysis, LLM translations, scores, and scripts, is available at https://github.com/MateMetzger/pali-translation-error-triage.

## Generative AI Use Statement

Generative AI was used in two distinct ways in this study. First, LLM outputs are the object of study: the research evaluates and triages AI-generated Pali-to-English translations. Second, generative AI, specifically GPT-5.5, was used as an assistive aid for phrasing, language editing, polishing, and code generation during the research and manuscript preparation process. Generative AI was not used to autonomously generate manuscript sections or to independently interpret the research data or derive conclusions. All AI-assisted outputs were reviewed by the author, who takes responsibility for the content of the manuscript.

## References

Rahul Aralikatte, Miryam de Lhoneux, Anoop Kunchukuttan, and Anders Søgaard. Itihasa: A large-scale corpus for sanskrit to english translation. In Proceedings of the 8th Workshop on Asian Translation (WAT2021), pages 191–197, 2021. DOI: 10.18653/v1/2021.wat-1.22.

David Bamman and Patrick J. Burns. Latin bert: A contextual language model for classical philology, 2020. arXiv:2009.10053.

Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, 2005.

Susan Bassnett. Translation Studies. Routledge, London, 4 edition, 2013. ISBN: 9780415506731.

Antoine Berman. The Experience of the Foreign: Culture and Translation in Romantic Germany. State University of New York Press, Albany, 1992. Translated by S. Heyvaert. ISBN: 9780791408766.

Bhikkhu Bodhi. In the Buddha’s Words: An Anthology of Discourses from the Pali Canon. Wisdom Publications, Boston, 2005. ISBN: 9780861714919.

Marina Fomicheva, Lucia Specia, and Francisco Guzman. Multi-hypothesis machine translation evaluation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1218–1232, 2020. DOI: 10.18653/v1/2020.acl-main.113.

Markus Freitag, George Foster, David Grangier, Viresh Ratnakar, Qijun Tan, and Wolfgang Macherey. Experts, errors, and context: A large-scale study of human evaluation for machine translation. Transactions of the Associationfor Computational Linguistics, 9:1460–1474, 2021. DOI: 10.1162/tacl\_a\_00437.

Rupert Gethin. The Foundations of Buddhism. Oxford University Press, Oxford, 1998. ISBN: 9780192892232.

Richard F. Gombrich. Theravada Buddhism: A Social History from Ancient Benares to Modern Colombo. Routledge, London, 2 edition, 2006. ISBN: 9780415365086.

Nuno M. Guerreiro, Ricardo Rei, Daan van Stigt, Luísa Coheur, Pierre Colombo, and André F. T. Martins. xcomet: Transparent machine translation evaluation through fine-grained error detection. Transactions of the Associationfor Computational Linguistics, 12:979–995, 2024. DOI: 10.1162/tacl\_a\_00683.

Namrata Bhalchandra Patil Gurav, Akashdeep Ranu, Archchana Sindhujan, and Diptesh Kanojia. Domain-specific quality estimation for machine translation in low-resource scenarios. In Proceedings of the Second Workshop on Language Modelsfor Low-Resource Languages (LoResLM 2026), pages 630–650, 2026. DOI: 10.18653/v1/2026.loreslm-1.55.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. The state and fate of linguistic diversity and inclusion in the nlp world. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6282–6293, 2020. DOI: 10.18653/v1/2020.acl-main.560.

Fabio Kepler, Jonay Trénous, Marcos Treviso, Miguel Vera, and André F. T. Martins. Openkiwi: An open source framework for quality estimation. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 117–122, 2019. DOI: 10.18653/v1/P19-3020.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing fine-grained evaluation capability in language models, 2023. arXiv:2310.08491.

Tom Kocmi and Christian Federmann. Large language models are state-of-the-art evaluators of translation quality, 2023a. arXiv:2302.14520.

Tom Kocmi and Christian Federmann. Gemba-mqm: Detecting translation quality error spans with gpt-4, 2023b. arXiv:2310.13988.

Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, Xiaoqi Ren, Shanfeng Zhang, Daniel Salz, Michael Boratko, Jay Han, Blair Chen, Shuo Huang, Vikram Rao, Paul Suganthan, Feng Han, Andreas Doumanoglou, Nithi Gupta, Fedor Moiseev, Cathy Yip, Aashi Jain, Simon Baumgartner, Shahrokh Shahi, Frank Palma Gomez, Sandeep Mariserla, Min Choi, Parashar Shah, Sonam Goenka, Ke Chen, Ye Xia, Sai Meher Karthik Duddu, Yichang Chen, Trevor Walker, Wenlei Zhou, Rakesh Ghiya, Zach Gleicher, Karan Gill, Zhe Dong, Mojtaba Seyedhosseini, Yunhsuan Sung, Raphael Hoffmann, and Tom Duerig. Gemini embedding: Generalizable embeddings from gemini, 2025. arXiv:2503.07891; DOI: 10.48550/arXiv.2503.07891.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. Geval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, 2023. DOI: 10.18653/v1/2023.emnlp-main.153.

Arle Lommel, Serge Gladkoff, Alan Melby, Sue Ellen Wright, Ingemar Strandvik, Katerina Gasova, Angelika Vaasa, Andy Benzo, Romina Marazzato Sparano, Monica Foresi, Johani Innis, Lifeng Han, and Goran Nenadic. The multi-range theory of translation quality measurement: Mqm scoring models and statistical quality control, 2024. arXiv:2405.16969.

Arle Richard Lommel, Aljoscha Burchardt, and Hans Uszkoreit. Multidimensional quality metrics (mqm): A framework for declaring and describing translation quality metrics. Tradumàtica, 12:455–463, 2014.

Benjamin Marie, Atsushi Fujita, and Raphael Rubino. Scientific credibility of machine translation research: A meta-evaluation of 769 papers. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 7297–7306, 2021. DOI: 10.18653/v1/2021.acl-long.566.

Nitika Mathur, Timothy Baldwin, and Trevor Cohn. Tangled up in bleu: Reevaluating the evaluation of automatic machine translation evaluation metrics. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 4984–4997, 2020. DOI: 10.18653/v1/2020.acl-main.448.

Máté Metzger and Nadnapang Phophichit. Palibench: A multi-reference blueprint for classical language translation benchmarks, 2026. arXiv:2605.16881; DOI: 10.48550/arXiv.2605.16881.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2014–2037, 2023. DOI: 10.18653/v1/2023.eaclmain.148.

Jeremy Munday. Introducing Translation Studies: Theories and Applications. Routledge, London, 4 edition, 2016. ISBN: 9781138912557.

Wilhelmina Nekoto, Vukosi Marivate, Tshinondiwa Matsila, Timi Fasubaa, Tajudeen Kolawole, Taiwo Fagbohungbe, Solomon Oluwole Akinola, Shamsuddeen Hassan Muhammad, Salomon Kabongo, Salomey Osei, Sackey Freshia, Rubungo Andre Niyongabo, Ricky Macharm, Perez Ogayo, Orevaoghene Ahia, Musie Meressa, Mofe Adeyemi, Masabata Mokgesi-Selinga, Lawrence Okegbemi, Laura Jane Martinus, Kolawole Tajudeen, Kevin Degila, Kelechi Ogueji, Kathleen Siminyu, Julia Kreutzer, Jason Webster, Jamiil Toure Ali, Jade Abbott, Iroro Orife, Ignatius Ezeani, Idris Abdulkabir Dangana, Herman Kamper, Hady Elsahar, Goodness Duru, Ghollah Kioko, Espoir Murhabazi, Elan van Biljon, Daniel Whitenack, Christopher Onyefuluchi, Chris Emezue, Bonaventure Dossou, Blessing Sibanda, Blessing Itoro Bassey, Ayodele Olabiyi, Arshath Ramkilowan, Alp Öktem, Adewale Akinfaderin, and Abdallah Bashir. Participatory research for low-resourced machine translation: A case study in african languages, 2020. arXiv:2010.02353.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, 2002. DOI: 10.3115/1073083.1073135.

Nadnapang Phophichit and Máté Metzger. Quality estimation of ai-driven translations of ancient buddhist scriptures: A multi-model automated translation and evaluation framework. Digital Scholarship in the Humanities, 2026. DOI: 10.1093/llc/fqag075.

Maja Popovi´c. chrf: Character n-gram f-score for automatic mt evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392–395, 2015. DOI: 10.18653/v1/W15-3049.

Tharindu Ranasinghe, Constantin Or˘asan, and Ruslan Mitkov. Transquest: Translation quality estimation with cross-lingual transformers. In Proceedings of the 28th International Conference on Computational Linguistics, pages 5070–5081, 2020. DOI: 10.18653/v1/2020.colingmain.445.

Ricardo Rei, Craig Stewart, Ana C. Farinha, and Alon Lavie. Comet: A neural framework for mt evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2685–2702, 2020. DOI: 10.18653/v1/2020.emnlp-main.213.

Ricardo Rei, Marcos Treviso, Nuno M. Guerreiro, Chrysoula Zerva, Ana C. Farinha, Christine Maroti, José G. C. de Souza, Taisiya Glushkova, Duarte M. Alves, Alon Lavie, Luísa Coheur, and André F. T. Martins. Cometkiwi: Ist-unbabel 2022 submission for the quality estimation shared task. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 634–645, 2022. DOI: 10.18653/v1/2022.wmt-1.60.

Paul Rosu. Litera: An llm based approach to latin-to-english translation. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 7796–7809, 2025. DOI: 10.18653/v1/2025.findings-naacl.434.

Thibault Sellam, Dipanjan Das, and Ankur P. Parikh. Bleurt: Learning robust metrics for text generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7881–7892, 2020. DOI: 10.18653/v1/2020.acl-main.704.

Matthew Snover, Bonnie Dorr, Rich Schwartz, Linnea Micciulla, and John Makhoul. A study of translation edit rate with targeted human annotation. In Proceedings of the 7th Conference of the Association for Machine Translation in the Americas, pages 223–231, 2006.

SuttaCentral. About suttacentral, 2026a. Accessed May 23, 2026. https://suttacentral. net/about.

SuttaCentral. Introduction to suttacentral, 2026b. Accessed May 23, 2026. https:// suttacentral.net/introduction.

SuttaCentral and Bilara contributors. bilara-data: Content for the bilara translation app, 2026. Accessed May 23, 2026. https://github.com/suttacentral/bilara-data.

Maria Tymoczko. Enlarging Translation, Empowering Translators. St. Jerome Publishing, Manchester, 2007. ISBN: 9781900650663.

Lawrence Venuti. The Translator’s Invisibility: A History of Translation. Routledge, London, 2 edition, 2008. ISBN: 9780415394550.

Oskar von Hinüber. A Handbook of Pali Literature. Walter de Gruyter, Berlin, 1996. DOI: 10.1515/9783110814989.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, 2024. DOI: 10.18653/v1/2024.acl-long.511.

Si Wu, John Wieting, and David A. Smith. Multiple references with meaningful variations improve literary machine translation, 2024. arXiv:2412.18707.

James L. Zainaldin, Cameron Pattison, Manuela Marai, Jacob Wu, and Mark J. Schiefsky. Evaluating llm-based translation of a low-resource technical language: The medical and philosophical greek of galen, 2026. arXiv:2602.24119.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations, 2020. https://openreview.net/forum?id=SkeHuCVFDr.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. arXiv:2306.05685.

Dan Zigmond. Toward a computational analysis of the pali canon. Journal of the Oxford Centre for Buddhist Studies, 20, 2021.

## A Supplementary Tables

Table A1: Final corpus by source novelty band.
<table><tr><td>Source novelty band</td><td>Passages</td><td>Segments</td><td>Pali characters</td><td>Sujato characters</td><td>Novelty range</td></tr><tr><td>Low</td><td>5,526</td><td>34,838</td><td>2,812,173</td><td>2,846,041</td><td>-7.812 to -0.385</td></tr><tr><td>Middle</td><td>5,697</td><td>32,084</td><td>2,363,188</td><td>2,286,894</td><td>-0.385 to 0.229</td></tr><tr><td>High</td><td>4,270</td><td>24,552</td><td>1,302,811</td><td>1,360,729</td><td>0.230 to 3.295</td></tr></table>

Table A2: Full-corpus review estimates for broad initial-risk boundaries.
<table><tr><td>Reviewed region</td><td>Reviewed instances</td><td>Major recall</td><td>Major precision</td><td>Any-error recall</td><td>Any-error precision</td></tr><tr><td>Top 20%</td><td>9,296</td><td>40.0%</td><td>5.9%</td><td>30.5%</td><td>18.6%</td></tr><tr><td>Top 40%</td><td>18,592</td><td>63.2%</td><td>4.6%</td><td>50.8%</td><td>15.5%</td></tr><tr><td>Top 60%</td><td>27,888</td><td>72.7%</td><td>3.6%</td><td>68.2%</td><td>13.9%</td></tr><tr><td>Top 80%</td><td>37,184</td><td>87.7%</td><td>3.2%</td><td>83.6%</td><td>12.7%</td></tr></table>

Table A3: Backtranslation metrics compared with the refined source-novelty-plus-peer score.
<table><tr><td>Metric</td><td>Major AUC</td><td>Error-vs- valid AUC</td><td>Budget</td><td>Major recall</td><td>Major precision</td><td>Any-error recall</td><td>Any-error precision</td></tr><tr><td>Refined source- novelty- plus-peer score</td><td>0.820</td><td>0.739</td><td>10%</td><td>47.3%</td><td>32.7%</td><td>30.5%</td><td>55.3%</td></tr><tr><td>chrF back- translation risk</td><td>0.824</td><td>0.745</td><td>10%</td><td>46.4%</td><td>32.0%</td><td>30.1%</td><td>54.7%</td></tr><tr><td>chrF back- translation + source- novelty- plus-peer</td><td>0.832</td><td>0.753</td><td>10%</td><td>50.7%</td><td>35.0%</td><td>31.6%</td><td>57.3%</td></tr><tr><td>score Refined source- novelty- plus-peer</td><td>0.820</td><td>0.739</td><td>20%</td><td>71.0%</td><td>24.5%</td><td>51.7%</td><td>46.8%</td></tr><tr><td>score chrF back- translation risk chrF back-</td><td>0.824</td><td>0.745</td><td>20%</td><td>67.1%</td><td>23.2%</td><td>46.7%</td><td>42.3%</td></tr><tr><td>translation + source- novelty- plus-peer score</td><td>0.832</td><td>0.753</td><td>20%</td><td>72.5%</td><td>25.0%</td><td>52.2%</td><td>47.3%</td></tr></table>

Table A4: "strong" GEMBA by candidate translator.
<table><tr><td>Candidate translator</td><td>Calibration n</td><td>Panel-major n</td><td>Major AUC</td><td>Major-vs-valid AÚC</td><td>Error-vs-valid AUC</td></tr><tr><td>DeepSeek</td><td>987</td><td>31</td><td>0.981</td><td>0.988</td><td>0.925</td></tr><tr><td>Grok</td><td>983</td><td>57</td><td>0.968</td><td>0.985</td><td>0.893</td></tr><tr><td>Qwen</td><td>1,030</td><td>119</td><td>0.963</td><td>0.984</td><td>0.913</td></tr></table>

Table A5: Individual LLM judges versus author labels.
<table><tr><td>Judge</td><td>Exact agreement</td><td>Kappa</td><td>Error recall</td><td>Error precision</td><td>Major recall</td><td>Major precision</td></tr><tr><td>GPT-5.5</td><td>73.0%</td><td>0.569</td><td>98.9%</td><td>65.3%</td><td>94.5%</td><td>69.9%</td></tr><tr><td>Gemini 3.1 Pro</td><td>71.0%</td><td>0.525</td><td>93.2%</td><td>66.0%</td><td>94.5%</td><td>65.2%</td></tr><tr><td>Claude Sonnet 4.6</td><td>72.6%</td><td>0.530</td><td>87.4%</td><td>70.3%</td><td>81.3%</td><td>65.5%</td></tr></table>

Table A6: Single-low-score warning thresholds against author labels.
<table><tr><td>Panel</td><td>Rule</td><td>Selected</td><td>Major recall</td><td>Major precision</td><td>Any-error recall</td><td>Any-error precision</td></tr><tr><td>&quot;strong&quot;</td><td>Any score ≤ 50</td><td>81 (16.2%)</td><td>62.6%</td><td>70.4%</td><td>38.9%</td><td>91.4%</td></tr><tr><td>&quot;strong&quot;</td><td>Any score ≤ 60</td><td>111 (22.2%)</td><td>75.8%</td><td>62.2%</td><td>52.6%</td><td>90.1%</td></tr><tr><td>&quot;strong&quot;</td><td>Any score ≤ 70</td><td>141 (28.2%)</td><td>80.2%</td><td>51.8%</td><td>65.3%</td><td>87.9%</td></tr><tr><td>&quot;strong&quot;</td><td>Any score ≤ 80</td><td>192 (38.4%)</td><td>91.2%</td><td>43.2%</td><td>78.9%</td><td>78.1%</td></tr><tr><td>&quot;peer&quot;</td><td>Any score ≤ 50</td><td>140 (28.0%)</td><td>58.2%</td><td>37.9%</td><td>48.4%</td><td>65.7%</td></tr><tr><td>&quot;peer&quot;</td><td>Any score ≤ 60</td><td>160 (32.0%)</td><td>64.8%</td><td>36.9%</td><td>53.2%</td><td>63.1%</td></tr><tr><td>&quot;peer&quot;</td><td>Any score ≤ 70</td><td>211 (42.2%)</td><td>78.0%</td><td>33.6%</td><td>63.2%</td><td>56.9%</td></tr></table>