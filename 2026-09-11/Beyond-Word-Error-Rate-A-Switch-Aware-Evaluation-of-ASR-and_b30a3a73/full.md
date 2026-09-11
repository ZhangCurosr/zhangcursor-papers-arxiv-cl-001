# Beyond Word Error Rate: A Switch-Aware Evaluation of ASR and Audio Language Models on English–Yoruba Code-Switched Speech

Chibuzor Okocha and Christian Grant

Department of Computer & Information Science & Engineering

University of Florida, Gainesville, FL, USA

{c.okocha, christan}@ufl.edu

Abstract—Automatic speech recognition (ASR) systems and audio language models (audio LMs) now report low error rates on monolingual benchmarks, but their behavior on codeswitched speech in low-resource, diacritic-rich languages remains poorly characterized. We present a switch-aware evaluation of eleven modern systems (six ASR models and five audio LMs) on English–Yoruba code-switched speech, using a deterministic 2000-utterance evaluation set and a shared scoring pipeline. Beyond word error rate (WER), we report switch-localized diagnostics: a switch-entry token error rate (SETER), windowed switch-point error rates (SPER@k), language-specific error rates, and a diacritic-insensitive WER. Our central finding is that aggregate WER hides code-switching behavior. The best system by WER (an ASR model) is statistically indistinguishable from a leading audio LM on WER, yet the audio LM is significantly better on every switch-localized metric. Across faithful systems, Yoruba token recognition collapses (error ≥ 0.97 for almost all systems) while English tokens are recognized far better, and errors concentrate sharply at switches into Yoruba. Several generative audio LMs fail as exact transcribers, producing translation, verbosity, and prompt leakage that are strongly prompt-dependent. We release manifests, metric implementations, and evaluation scripts to support reproducible, switchaware benchmarking for African code-switched speech.

Index Terms—code-switching, automatic speech recognition, audio language models, low-resource languages, Yoruba, evaluation metrics

## I. INTRODUCTION

Recent ASR systems and audio language models report strong results on widely used monolingual benchmarks. Yet a large fraction of the world’s speakers are bilingual and routinely code-switch, alternating between two languages within a single utterance. Code-switching breaks the monolingual assumptions baked into most systems: tokenizers, language priors, and decoding all degrade at language boundaries [1], [2]. This degradation is especially acute when one of the languages is low-resource and orthographically rich, as is the case for many African languages.

Most code-switching ASR research targets Mandarin– English (SEAME, ASCEND) or Arabic–English (ArzEn) [4]– [6]. English–Yoruba is comparatively underexplored, despite Yoruba being spoken by tens of millions and presenting distinctive challenges: dense lexical mixing and a tone orthography that uses sub-dot characters (e<sub>.</sub>, o<sub>.</sub> , s<sub>.</sub>) and tone diacritics. At the same time, evaluation practice has not kept pace with the models. Aggregate WER, the default metric, sums errors over an utterance that is usually dominated by a matrix language; errors at the rarer switch points and embedded-language tokens are diluted [2]. A model can therefore appear strong by WER while failing precisely where code-switching happens.

We take an evaluation-first stance. Rather than proposing a new model, we ask how well current systems actually transcribe English–Yoruba code-switched speech, and whether standard WER explains their behavior. We benchmark eleven systems under a shared, deterministic protocol and complement WER with switch-aware diagnostics that localize errors to switch boundaries, embedded-language tokens, and diacritics.

Our contributions are:

• A switch-aware benchmark of six ASR systems and five audio LMs on English–Yoruba code-switched speech, with a shared, deterministic 2000-utterance evaluation set and paired statistical testing.

• Evidence that WER and switch-localized metrics rank systems differently: the best ASR model by WER is statistically tied with a leading audio LM that is in turn significantly better at switch boundaries. Across systems, WER rank is essentially uncorrelated with Yoruba token error and only weakly correlated with switch-entry error, so WER is a poor proxy for code-switching fidelity.

• A characterization of Yoruba recognition collapse and a strong, system-consistent switch-direction asymmetry, together with the effects of switch density, language dominance, and domain.

• An analysis of generative audio LM failure modes (translation, verbosity, prompt leakage), their prompt sensitivity, and an automatic error taxonomy that separates recognition from generation errors without human labels.

• Released manifests, metric implementations, and evaluation scripts for reproducible, switch-aware benchmarking of African code-switched speech.

## II. RELATED WORK

Code-switching ASR and metrics. Code-switching ASR has been studied mainly on Mandarin–English (SEAME, AS-CEND) and Arabic–English (ArzEn) corpora [4]–[6], typically scored with WER, CER, and mixed error rate (MER), which counts errors per token in each language but still aggregates over the whole utterance. Because the matrix language dominates most utterances, such aggregate metrics underweight the embedded-language tokens and switch points that distinguish code-switching from monolingual ASR. Ugan et al. [2] make this explicit: fine-tuning on monolingual data from both languages improves classical WER on code-switched test sets even as accuracy on the actually code-switched words degrades. They propose the Point-of-Interest Error Rate (PIER), which restricts scoring to embedded-language tokens, and the DECM benchmark for bilingual ASR [3]. Related diagnostics include switch-point WER and language-pair-specific error rates. Our diagnostics are complementary to PIER: instead of scoring by language membership alone, SETER and SPER@k localize scoring to the switch boundary (the first embedded token and a window around it), and we additionally report per-language error rates and a diacritic-insensitive WER. To our knowledge, these switch-localized diagnostics have not previously been applied to English–Yoruba or, more broadly, to a side-by-side comparison of ASR systems and audio LMs.

Audio language models. Instruction-following audio LMs couple a speech encoder to a large language model and are prompted in natural language. Recent systems include Qwen2- Audio and the omni-modal Qwen2.5-Omni [13], [14], Kimi-Audio [15], and the Audio Flamingo family [16], [17]; LLMdecoder ASR models such as SALM [10] (the basis of Canary-Qwen) blur the line further. These models are usually benchmarked on audio understanding and question-answering tasks, where free-form generation is desirable; their reliability as faithful, verbatim transcribers of code-switched, low-resource speech is much less examined, and the generation-oriented decoding that helps understanding can hurt transcription. Concurrent work has begun to probe audio LMs’ semantic reasoning over African-accented speech [21]; we focus instead on verbatim transcription of code-switched speech. We therefore evaluate them head-to-head with dedicated ASR systems spanning transducer (Parakeet-TDT [8], [9]), attention-encoderdecoder (Whisper [7]), and LLM-decoder (Canary-Qwen [10], Qwen3-ASR [11], Granite Speech [12]) designs.

Low-resource and African speech. African languages remain under-served by speech technology despite large speaker populations. A growing body of resources and benchmarks targets African-accented English and African languages, including AfriSpeech-200 [18] for clinical and general-domain ASR, AfriSpeech-Dialog [20] for spontaneous conversation, AfriSpeech-MultiBench [23] and AfriVox [22] for multidomain and multilingual evaluation of ASR systems and speech LLMs, and AfriNames [24] for entity-rich recognition. Yoruba in particular has motivated work on diacritic (tone and sub-dot) restoration [19], since stripped diacritics are common in web text and change word identity. Yet most of these resources target monolingual ASR, spoken-language understanding, or text-to-speech; code-switched, diacritic-aware evaluation of modern ASR systems and audio LMs is largely absent. We address this gap for English–Yoruba and treat diacritics explicitly through a diacritic-insensitive WER variant.

TABLE I  
AUDITED STATISTICS OF THE AFRICODESWITCH VALIDATION CORPUS.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Metadata rows</td><td>9,966</td></tr><tr><td>Rows with available audio</td><td>8,181</td></tr><tr><td>Rows missing audio</td><td>1,785</td></tr><tr><td>Available audio (hours)</td><td>9.54</td></tr><tr><td>Speakers / prompts / domains Mean duration (s)</td><td>100 / 2,937 /  13 4.20</td></tr><tr><td>Mean words / utterance</td><td>8.52</td></tr><tr><td>Mean / max switches per utterance</td><td>2.60 / 9</td></tr><tr><td>Dominance (EN / bal. / YO)</td><td>2,799 /  3,336 / 2,046</td></tr></table>

## III. CORPUS

We evaluate on AFRICODESWITCH, an English–Yoruba code-switched speech corpus. We use the validation partition and stage a deterministic set of evaluation manifests so that every system is scored on identical utterances. Table I summarizes the audited statistics. Of 9,966 metadata rows, 8,181 have available audio (9.54 hours); the remaining 1,785 rows lack audio and are excluded. Utterances are short (mean 4.20 s, 8.52 words) and densely mixed (mean 2.60 switches, up to 9), spanning 100 speakers, 2,937 prompts, and 13 domains.

Word-level language tags are parsed from the corpus metadata and used to derive, for each utterance, switch points, switch direction, language dominance (English-dominant, balanced, Yoruba-dominant), and switch density. Text is Unicode NFC-normalized. The corpus is balanced across dominance (2,799 English-dominant, 3,336 balanced, 2,046 Yorubadominant) but speaker gender is skewed (6,969 female vs. 1,212 male rows), which we note as a limitation. Unless stated otherwise, results use the 2000-utterance cap2000 manifest; prompt-robustness studies use cap1000.

## IV. SWITCH-AWARE METRICS

All metrics derive from a minimum-edit-distance alignment between the reference token sequence $R = ( r _ { 1 } , \ldots , r _ { N } )$ and the hypothesis. The alignment yields substitution, deletion, and insertion counts S,D,I and labels each reference token as correctly recognized or not. Let $e _ { i } = 1 { \mathrm { ~ i f ~ } } r _ { i }$ is substituted or deleted (not correctly recognized) and $e _ { i } = 0$ otherwise. For any set of reference positions $T \subseteq \{ 1 , \ldots , N \}$ we define the masked error rate

$$
\mathrm { E R } ( T ) = \frac { 1 } { | T | } \sum _ { i \in T } e _ { i } .\tag{1}
$$

Every language-specific and switch-aware metric below is an instance of (1) over a different position set; corpus-level values are the mean over utterances.

WER and CER. Word error rate is the standard

$$
\mathrm { W E R } = \frac { S + D + I } { N } ,\tag{2}
$$

computed after NFC normalization, case-folding, and punctuation stripping; CER is the analogous ratio over characters. Normalization is diacritic-sensitive by default.

Diacritic-insensitive WER. Let $\phi ( \cdot )$ strip combining tone and sub-dot marks from a token. $\mathrm { W E R _ { d i } }$ is WER recomputed on $\phi ( R )$ and $\phi ( H ) ;$ comparing it to WER isolates the error attributable to Yoruba diacritics.

Language-specific error rates. With lang $\mathbf { \eta } _ { [ } r _ { i } ) \in \{ \mathrm { E N } , \mathrm { Y O } \}$ taken from the reference tags,

$$
\begin{array} { r } { \mathrm { E N - E R } = \mathrm { E R } ( L _ { \mathrm { E N } } ) , \quad \mathrm { Y O - E R } = \mathrm { E R } ( L _ { \mathrm { Y O } } ) , } \end{array}\tag{3}
$$

where $L _ { \ell } = \{ i : \mathrm { l a n g } ( r _ { i } ) = \ell \}$ . These quantify which language a system fails on.

SETER (Switch-Entry Token Error Rate). A switch-entry token is the first token of a new language run. Defining the switch-entry set

$$
P = \{ i \geq 2 : \log ( r _ { i } ) \neq \log ( r _ { i - 1 } ) \} ,\tag{4}
$$

$\mathbf { S E T E R } { = } \mathbf { \overrightarrow { E R } } ( P )$ measures whether a system “lands” each switch.

SPER@k (Switch-Point Error Rate). Let $d _ { i } = \operatorname* { m i n } _ { j \in P } | i -$ j| be the distance from position i to the nearest switch entry, and let $W _ { k } = \{ i : d _ { i } \leq k \}$ be the window of radius k around switch points. Then

$$
\mathrm { S P E R } @ k = \mathrm { E R } ( W _ { k } ) , \qquad k \in \{ 0 , 1 , 2 , 3 \} .\tag{5}
$$

By construction SPER@0 = SETER (the entry token alone), and increasing k widens the neighbourhood toward the whole utterance. We foreground SPER@1 and SPER@3.

Output-quality flags. Per-utterance heuristics flag blank output, truncation, degenerate repetition, length expansion (hypothesis far longer than reference), and prompt leakage (instruction text echoed in the output); they characterize generative failure modes rather than scoring transcription accuracy.

Relative to PIER [2], which scores embedded-language “points of interest,” SETER and SPER@k are positional diagnostics centred on switch boundaries, while the language-specific rates capture the complementary languagemembership view.<sup>1</sup>

## V. EXPERIMENTAL SETUP

We evaluate six ASR systems: Parakeet-TDT-0.6B-v2 [8], [9], Qwen3-ASR-1.7B [11], Canary-Qwen-2.5B [8], [10], Whisper-large-v3 [7] (forced English and auto language detection), and Granite-Speech-4.1-2B [12]; and five audio LMs: Kimi-Audio-7B-Instruct [15], Audio Flamingo 3 [17], Audio Flamingo 2 [16], Qwen2-Audio-7B-Instruct [13], and Qwen2.5-Omni-7B [14]. All systems run zero-shot with greedy decoding. The main benchmark uses a single transcription instruction (the primary prompt) for all audio LMs. For two prompt-sensitive audio LMs we additionally evaluate direct and anti-translation prompt variants. Inference and scoring are driven by manifest-based runners on a Slurm cluster; the same manifests, normalization, and alignment are used for every system. Two intended systems (Qwen3-Omni and a gated Cohere transcription model) could not be run due to environment and access constraints and are left to future work.

## VI. RESULTS

## A. WER does not explain switch behavior

Table II reports the main benchmark. Parakeet attains the best WER (66.1%) and CER (35.7%). Strikingly, Kimi-Audio, a generative audio LM, ties it on WER (66.2%) while winning five of the remaining seven columns, including the diacriticinsensitive WER, the English error rate, and all three switchlocalized metrics (SETER, SPER@1, SPER@3). In other words, the two systems that look identical by aggregate WER behave differently exactly where code-switching occurs.

A paired utterance-level bootstrap (1,000 resamples over the shared manifest) confirms this dissociation (Table III). The WER difference between Parakeet and Kimi is not significant $( \Delta = + 0 . 1 3$ points, $p = 0 . 7 6 .$ , CI crossing zero), whereas Kimi’s advantage on SETER, SPER@1, and SPER@3 is significant $( p < 0 . 0 0 1$ in all three cases). The effect is not unique to this pair: Canary-Qwen has significantly worse WER than Parakeet $( p < 0 . 0 0 1 )$ but statistically indistinguishable SETER and SPER $( p > 0 . 1 2 )$ . Switch-localized fidelity and aggregate WER are therefore separable axes of performance.

This separation holds across the whole pool. Ranking the systems by WER and by each diagnostic, the Spearman correlation between WER and English token error is high $( \rho = 0 . 7 4 )$ , but the correlation between WER and Yoruba token error is essentially zero $( \rho = - 0 . 1 0 )$ , and the correlation between WER and switch-entry error (SETER) is only moderate $( \rho = 0 . 6 7 $ , not significant at $\alpha = 0 . 0 5 )$ . A system’s aggregate WER is thus largely determined by its matrix-language (English) accuracy and tells us little about the code-switching-specific behavior the task cares about. The boundary window itself is the hardest region: SPER@0 (the switch token) exceeds SPER@3 for every system (e.g., $6 8 . 9 \%  6 5 . 2 \%$ for Parakeet), so error eases only gradually as the window widens away from the switch.

## B. Yoruba recognition collapses

Errors are overwhelmingly driven by Yoruba, not by ordinary English ASR error. Among the eight faithful systems, English token error ranges from 33% to 46%, while Yoruba token error $\mathrm { i s } \geq 9 6 . 6 \%$ and $\mathrm { i s } \geq 9 7 . 5 \%$ for every system other than auto-language Whisper. In effect, English tokens are recognized roughly two-thirds of the time, while Yoruba tokens are almost never correct. We recomputed the languagespecific rates directly from manifest tags and alignments as a sanity check; the recomputed values match the reported rates exactly for the top systems (e.g., Parakeet YO-ER 0.9903, EN-ER 0.3444), confirming the collapse is real and not a scoring artifact.

## C. Errors localize at switches into Yoruba

Figure 1 decomposes errors structurally. Panel (a) plots token error against position relative to a switch: error spikes at the boundary when the switch is into Yoruba, but stays far lower when the switch is into English. Panel (b) makes the asymmetry explicit: switch-token error for EN→YO is 0.95– 1.00 across systems, versus 0.40–0.53 for YO→EN. Panel (c)

TABLE II  
MAIN CAP2000 BENCHMARK (2,000 SHARED UTTERANCES, PRIMARY PROMPT). ALL VALUES ARE PERCENTAGES; LOWER IS BETTER. $\mathrm { W E R } _ { \mathrm { d i } } \mathrm { : }$ DIACRITIC-INSENSITIVE WER; EN/YO-ER: ENGLISH/YORUBA TOKEN ERROR RATE. BEST VALUE PER COLUMN AMONG FAITHFUL SYSTEMS IS IN BOLD. THE LOWER BLOCK CONTAINS GENERATIVE MODELS WHOSE WER>100% INDICATES SUBSTANTIAL OVER-GENERATION RATHER THAN ORDINARY SUBSTITUTION.
<table><tr><td>Model</td><td>Family</td><td>WER</td><td>CER</td><td> $\mathrm { W E R _ { d i } }$ </td><td>EN-ER</td><td>YO-ER</td><td>SETER</td><td>SPER@1</td><td>SPER@3</td></tr><tr><td>Parakeet-TDT-0.6B-v2</td><td>ASR</td><td>66.1</td><td>35.7</td><td>66.0</td><td>34.4</td><td>99.0</td><td>68.9</td><td>67.2</td><td>65.2</td></tr><tr><td>Kimi-Audio-7B-Instruct</td><td>LALM</td><td>66.2</td><td>37.4</td><td>64.3</td><td>32.9</td><td>97.5</td><td>66.8</td><td>65.0</td><td>63.2</td></tr><tr><td>Qwen3-ASR-1.7B</td><td>ASR</td><td>67.6</td><td>39.6</td><td>67.9</td><td>37.9</td><td>99.2</td><td>70.4</td><td>68.2</td><td>66.0</td></tr><tr><td>Canary-Qwen-2.5B</td><td>ASR</td><td>70.0</td><td>40.1</td><td>70.0</td><td>33.5</td><td>99.4</td><td>68.7</td><td>66.8</td><td>64.8</td></tr><tr><td>Audio Flamingo 3</td><td>LALM</td><td>70.1</td><td>38.9</td><td>69.5</td><td>40.2</td><td>98.4</td><td>73.1</td><td>69.7</td><td>67.3</td></tr><tr><td>Whisper-large-v3</td><td>ASR</td><td>71.2</td><td>41.7</td><td>70.9</td><td>36.8</td><td>98.9</td><td>70.2</td><td>68.0</td><td>66.0</td></tr><tr><td>Granite-Speech-4.1-2B</td><td>ASR</td><td>73.5</td><td>42.8</td><td>74.0</td><td>38.9</td><td>99.4</td><td>72.9</td><td>69.6</td><td>67.2</td></tr><tr><td>Whisper-large-v3 (auto)</td><td>ASR</td><td>74.8</td><td>39.2</td><td>71.7</td><td>46.0</td><td>96.6</td><td>73.0</td><td>71.3</td><td>69.2</td></tr><tr><td>Audio Flamingo 2</td><td>LALM</td><td>119.8</td><td>112.3</td><td>128.8</td><td>98.0</td><td>99.7</td><td>99.2</td><td>99.3</td><td>98.9</td></tr><tr><td>Qwen2-Audio-7B-Instruct</td><td>LALM</td><td>137.1</td><td>123.4</td><td>146.3</td><td>57.7</td><td>98.8</td><td>77.2</td><td>76.4</td><td>75.3</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>LALM</td><td>202.9</td><td>123.0</td><td>143.4</td><td>57.3</td><td>91.3</td><td>78.0</td><td>75.2</td><td>72.8</td></tr></table>

TABLE III

PAIRED BOOTSTRAP COMPARISON, KIMI-AUDIO VS. PARAKEET (C A P2000). ∆ IS KIMI MINUS PARAKEET IN POINTS; NEGATIVE FAVORS KIMI. 95% CIS AND TWO-SIDED BOOTSTRAP p.
<table><tr><td>Metric</td><td>∆ (pts)</td><td>95% CI</td><td>p</td></tr><tr><td>WER</td><td>+0.13</td><td>[−0.67,+1.12]</td><td>0.76</td></tr><tr><td>SETER</td><td>-2.09</td><td>-3.33,-0.83</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>SPER@1</td><td>-2.20</td><td>-3.01,-1.38</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>SPER@3</td><td>-2.02</td><td>[−2.64, −1.37]</td><td> $< 0 . 0 0 1$ </td></tr></table>

shows a monotonic effect of dominance: error climbs from ∼0.42 on English-dominant utterances to ∼0.65 (balanced) to ∼0.83 (Yoruba-dominant). Kimi-Audio is consistently lowest within each stratum. These patterns indicate that switchboundary degradation is a distinct failure mode tied to producing Yoruba content, not a uniform smearing of WER.

## D. Switching rate, domain, and speaker

Three further factors modulate difficulty. Switching rate: grouping utterances by switch density, mean WER over the strong systems rises monotonically from 66.9% (low) to 67.9% (moderate) to 71.2% (high), and SETER rises more steeply $( 6 6 . 7 \%  7 0 . 1 \%  7 2 . 1 \% )$ : denser switching hurts boundary fidelity more than it hurts aggregate WER. Domain: difficulty is heterogeneous across the 13 domains (Fig. 2), spanning a 10-point WER range from conversational/expository domains (General, Education, Religion; ∼63–66%) to terminology-heavy ones (Science, Sports, Fashion; ∼71–74%), reflecting how much rare, often borrowed vocabulary each domain carries. Speaker gender: despite the corpus skew toward female speakers, error rates are comparable across gender (mean WER 68.9% male vs. 68.4% female; Yoruba error 0.99 vs. 0.99). The dominant disparity in this benchmark is along the language axis, not speaker gender, a point we return to in Section IX.

## E. Diacritics are not yet the bottleneck

Stripping diacritics barely changes WER for the strong systems (Parakeet +0.06, Canary −0.05, Qwen3-ASR −0.29, Whisper-v3 +0.30 points). The exception among faithful systems is Kimi-Audio (+1.85 points), consistent with it producing more Yoruba content, and therefore more diacritics to get wrong. Because most systems rarely produce correct Yoruba at all, diacritic accuracy is currently a second-order concern; it will become central once base Yoruba recognition improves. We report $\mathrm { W E R _ { d i } }$ so that future, stronger systems can be assessed on diacritics directly.

## F. Generative audio LMs often fail as transcribers

The lower block of Table II shows three audio LMs with WER above 100%, indicating systematic over-generation rather than substitution. Output-quality flags localize the cause: Qwen2-Audio leaks the prompt into the transcript in 32% of cap2000 outputs, and Qwen2.5-Omni shows elevated repetition (3.1%) and length expansion (3.7%), including degenerate loops. By contrast, all ASR systems and Kimi-Audio are clean on these flags. These behaviors are highly prompt-dependent (Table V): switching Qwen2-Audio from the primary prompt to a direct instruction cuts prompt leakage from 33.7% to 0.2% and WER from 135.6% to 103.3%; Qwen2.5-Omni improves similarly. Prompting is thus a major confound when benchmarking generative audio LMs for transcription, and a single prompt can under-state or over-state a model’s true transcription ability.

## G. An automatic error taxonomy

Many failure modes are detectable directly from the alignment and the output, without human labeling. Table IV summarizes two such views. The first decomposes wordlevel errors into substitutions, deletions, and insertions. Every faithful system is substitution-dominated (69–73% of errors), the signature of recognition failure: the model emits the wrong word rather than inventing or dropping content. The two worst audio LMs invert this profile: insertions account for 46% (Qwen2-Audio) and 65% (Qwen2.5-Omni) of their errors, the signature of generation failure. The second view counts generative pathologies per utterance. These are negligible for ASR systems and Kimi-Audio but isolate the specific breakdown of each weak model: prompt leakage dominates Qwen2- Audio (31.9%), while repetition loops and length expansion dominate Qwen2.5-Omni. This automatic taxonomy cleanly separates recognition from generation errors, but it cannot by itself distinguish a faithful mis-transcription from a fluent translation or paraphrase, which requires output-side language identification or human judgment. The curated 100-utterance sample (below) targets exactly those semantic categories.

Token position relative to switch  
(a) Error around switch point  
![](images/bdd38d9e1525acaad044f0eff0236f9dd560eff14c37976e8dcebd88b4a4d75c.jpg)

![](images/dbf9b2b86fd0d9f7e2b9774980baae1fa2a68a05aaa9bf7186d64b8a469f897d.jpg)

(c) Effect of language dominance  
![](images/0c9c4aa0c92f073ca4e00449e0a4630bc7134d4b57f2ca00c509001d119c0a95.jpg)  
TABLE IV

Fig. 1. Structural error analysis on cap2000. (a) Token error vs. position relative to the switch point for Parakeet and Kimi-Audio, by direction; error peak at switches into Yoruba. (b) Switch-token error by direction across strong systems: switching into Yoruba is near-uniformly hard. (c) Token error increases monotonically with Yoruba dominance. EN→YO: switch into Yoruba; YO→EN: switch into English.  
![](images/1209645e6d261d28560b3bc08af8f79422139249d2bb3c89568e4ebdbbf4a2b8.jpg)  
Fig. 2. Mean WER per domain on cap2000, averaged over the six strong systems. Difficulty spans a 10-point range; terminology-dense domains (Fashion, Sports, Science) are hardest.

## VII. QUALITATIVE ANALYSIS

The metrics above are illustrated by recurring patterns in the outputs. We highlight three.

Landing the switch. For Ó bá wa nu chalkboard (one switch), Kimi-Audio recovers the embedded word (“. . . nu

AUTOMATIC ERROR TAXONOMY ON C A P2000 (PRIMARY). LEFT: COMPOSITION OF WORD-LEVEL ERRORS INTO SUBSTITUTIONS/DELETIONS/INSERTIONS (% OF ERRORS). RIGHT:   
PER-UTTERANCE INCIDENCE (%) OF GENERATIVE PATHOLOGIES: PROMPT LEAKAGE, REPETITION, LENGTH EXPANSION. RECOGNITION-ERROR   
SYSTEMS ARE SUBSTITUTION-DOMINATED; OVER-GENERATING SYSTEMS ARE INSERTION-DOMINATED.

<table><tr><td></td><td colspan="3">Error composition</td><td colspan="3">Pathology rate</td></tr><tr><td>Model</td><td>Sub</td><td>Del</td><td>Ins</td><td>Leak</td><td>Rep</td><td>Exp</td></tr><tr><td>Parakeet</td><td>69.2</td><td>26.9</td><td>3.8</td><td>0.2</td><td>0.1</td><td>0.0</td></tr><tr><td>Kimi-Audio</td><td>70.1</td><td>22.8</td><td>7.1</td><td>0.3</td><td>0.2</td><td>0.1</td></tr><tr><td>Qwen3-ASR</td><td>71.0</td><td>23.7</td><td>5.3</td><td>0.2</td><td>0.1</td><td>0.1</td></tr><tr><td>Canary-Qwen</td><td>73.2</td><td>17.0</td><td>9.8</td><td>0.2</td><td>0.2</td><td>0.1</td></tr><tr><td>Audio Flamingo 3</td><td>72.9</td><td>20.4</td><td>6.7</td><td>0.2</td><td>0.5</td><td>0.2</td></tr><tr><td>Whisper-v3</td><td>69.5</td><td>20.6</td><td>9.9</td><td>0.2</td><td>0.3</td><td>0.1</td></tr><tr><td>Granite</td><td>71.5</td><td>17.4</td><td>11.2</td><td>0.3</td><td>0.2</td><td>0.1</td></tr><tr><td>Whisper-auto</td><td>72.2</td><td>17.6</td><td>10.2</td><td>0.2</td><td>0.7</td><td>0.4</td></tr><tr><td>Audio Flamingo 2</td><td>73.3</td><td>8.8</td><td>17.8</td><td>0.2</td><td>0.8</td><td>1.3</td></tr><tr><td>Qwen2-Audio</td><td>52.1</td><td>1.6</td><td>46.2</td><td>31.9</td><td>0.9</td><td>3.5</td></tr><tr><td>Qwen2.5-Omni</td><td>30.8</td><td>4.3</td><td>64.9</td><td>0.7</td><td>3.1</td><td>3.7</td></tr></table>

TABLE V

PROMPT ROBUSTNESS ON CAP1000. LOWER WER/SETER IS BETTER;“LEAK” IS THE PROMPT-LEAKAGE RATE.
<table><tr><td>Model</td><td>Prompt</td><td>WER</td><td>SETER</td><td>Leak</td></tr><tr><td rowspan="3">Qwen2-Audio-7B</td><td>primary</td><td>135.6</td><td>76.5</td><td>33.7</td></tr><tr><td>direct</td><td>103.3</td><td>77.6</td><td>0.2</td></tr><tr><td>anti-translation</td><td>139.6</td><td>81.0</td><td>0.3</td></tr><tr><td rowspan="3">Qwen2.5-Omni-7B</td><td>primary</td><td>183.0</td><td>77.6</td><td>0.7</td></tr><tr><td>direct</td><td>125.9</td><td>75.8</td><td>1.6</td></tr><tr><td>anti-translation</td><td>194.1</td><td>77.3</td><td>0.4</td></tr></table>

chalkboard”, SETER = 0) while Parakeet, Qwen3-ASR, Canary, Audio Flamingo 3, and Whisper all miss it (“chopboard”, “chobod”, “chopper”, . . . ; SETER = 1). For mo love airports, but mo hate flying, Kimi-Audio and Whisper transcribe it verbatim (WER = 0), whereas Parakeet hallucinates English (“Molov Airport Bomo eight flying”).

WER wins, switch lost. The advantage is not universal. On a Yoruba-heavy utterance, Kimi-Audio enters a degenerate loop (repeating a short phrase, WER = 4.39) while Parakeet returns a short, wrong-but-bounded hypothesis $( \mathbf { W E R } = 1 . 0 0 ) $ , a case where Parakeet’s lower WER masks that neither system recovers the switch.

Translation and language confusion. Generative models frequently translate or summarize instead of transcribing (e.g., rendering a Yoruba clause as an English paraphrase), and occasionally emit text in unrelated scripts. These are exactly the behaviors penalized by switch-localized metrics but partially hidden by utterance-level WER on matrix-heavy inputs.

Complementing the automatic taxonomy of Table IV, we curated a 100-utterance sample across the top five systems for fine-grained human annotation of the categories that automation cannot resolve: in particular translation-insteadof-transcription, along with Yoruba deletion/substitution, English hallucination, boundary collapse, and diacritic loss. The human-labeled taxonomy is in progress and released with the benchmark.

## VIII. DISCUSSION

Why WER hides switching here. The dissociation we observe is structural, not incidental. In an asymmetric codeswitching setting the matrix language (English) supplies most reference tokens and is recognized two to three times more accurately than the embedded language, so aggregate WER is dominated by matrix-language accuracy, exactly what the cross-system correlations show (WER tracks EN-ER at $\rho =$ 0.74 but is uncorrelated with YO-ER at $\rho = - 0 . 1 0 )$ . A single corpus-level WER therefore cannot, even in principle, reflect how well a system handles the embedded language or the switch points that define the task. For low-resource code-switching we accordingly argue that language-specific and switch-localized metrics should be reported as standard alongside WER, not treated as optional diagnostics.

The bottleneck is Yoruba coverage, not diacritics. Two findings together locate the failure: Yoruba token error sits near ceiling for every faithful system, yet stripping diacritics barely moves WER. Models are thus not getting Yoruba words almost-right modulo tone marks; they largely fail to produce Yoruba content at all. This implies an ordering of priorities for practitioners: expanding Yoruba lexical and acoustic coverage (data, tokenization, pretraining exposure) must precede orthographic refinements such as diacritic restoration, which only become measurable once base recognition improves. We report a diacritic-insensitive WER precisely so this secondorder effect can be tracked as systems get better.

Switch direction points to a matrix-language prior. The strong asymmetry (near-total error when switching into Yoruba but roughly half that when switching back into English) suggests that decoding is governed by a dominant highresource prior: models readily snap back to English after an embedded span but resist entering Yoruba. This is consistent with boundary errors arising from language balance within the model rather than from acoustic difficulty alone, and it motivates interventions that act at the boundary specifically: language-balanced or switch-aware decoding, and supervision targeted at embedded-language entry tokens rather than at uniform WER.

Audio LMs cut both ways. Kimi-Audio shows that an instruction-following audio LM can match the best dedicated ASR system on WER while significantly outperforming it at switch boundaries, plausibly because an LLM decoder carries broader multilingual text knowledge that helps it realize Yoruba spans. Yet the same model family also contains the worst transcribers in our pool, failing through translation, verbosity, and prompt leakage rather than mis-recognition; these pathologies are largely artifacts of prompting and decoding, since a direct prompt cut Qwen2-Audio’s leakage from 34% to near zero. Audio LMs are thus a promising route to better embedded-language recognition, but only when constrained toward verbatim transcription, and any fair benchmark of them must control the prompt. More broadly, our results caution against ranking code-switching systems by aggregate WER, a practice with direct fairness consequences: WERonly leaderboards render the exclusion of embedded-language speakers invisible. They also chart concrete next steps: Yorubaand boundary-focused fine-tuning, language-balanced decoding, and transcription-constrained prompting for audio LMs.

## IX. LIMITATIONS AND ETHICS

This is an evaluation-only study on a single English–Yoruba corpus; no training or fine-tuning is performed, and results may not transfer to other dialects, accents, or recording conditions. The corpus is gender-skewed toward female speakers; reassuringly, we find no large error gap between male and female speakers in this sample, which suggests the dominant disparity is along the language axis (Yoruba vs. English) rather than speaker demographics, though the skew still limits the strength of any per-gender claim. Switch-aware metrics are computed from reference-side language tags and an automatic alignment; tag and alignment quality bound their precision, and we mitigate this with a recomputation sanity check. The fine-grained error taxonomy relies on human annotation that is still underway. We view low resource code-switched evaluation as an accessibility and fairness concern: systems that cannot transcribe Yoruba effectively exclude its speakers, and reporting only WER can make this exclusion invisible.

## X. CONCLUSION

We presented a switch-aware evaluation of modern ASR systems and audio language models on English–Yoruba codeswitched speech, showing that aggregate WER masks a distinct, boundary-localized failure mode that switch-localized and language-specific metrics bring into view. We release the manifests, metric implementations, and evaluation scripts to make these failures visible and to provide a reproducible baseline for African code-switched speech; switch-aware finetuning and adaptation are natural next steps.

## ACKNOWLEDGMENT

The authors thank the contributors and annotators who made the AFRICODESWITCH corpus possible, and gratefully acknowledge the computational resources that supported these experiments. We also thank colleagues for their feedback on earlier versions of this work.

## REFERENCES

[1] S. Sitaram, K. R. Chandu, S. K. Rallabandi, and A. W. Black, “A survey of code-switched speech and language processing,” arXiv:1904.00784, 2019.

[2] E. Y. Ugan, N.-Q. Pham, L. Bärmann, and A. Waibel, “PIER: a novel metric for evaluating what matters in code-switching,” in Proc. IEEE ICASSP, 2025, pp. 1–5.

[3] E. Y. Ugan, N.-Q. Pham, and A. Waibel, “DECM: evaluating bilingual ASR performance on a code-switching/mixing benchmark,” in Proc. LREC-COLING, 2024.

[4] D.-C. Lyu, T.-P. Tan, E. S. Chng, and H. Li, “SEAME: a Mandarin– English code-switching speech corpus in South-East Asia,” in Proc. INTERSPEECH, 2010, pp. 1986–1989.

[5] H. Lovenia et al., “ASCEND: a spontaneous Chinese–English dataset for code-switching in multi-turn conversation,” in Proc. LREC, 2022.

[6] I. Hamed, N. T. Vu, and S. Abdennadher, “ArzEn: a speech corpus for code-switched Egyptian Arabic–English,” in Proc. LREC, 2020, pp. 4237–4246.

[7] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023, pp. 28492–28518.

[8] M. Sekoyan, N. R. Koluguri, N. Tadevosyan, P. Zelasko, T. Bartley, N. Karpov, J. Balam, and B. Ginsburg, “Canary-1B-v2 & Parakeet-TDT-0.6B-v3: efficient and high-performance models for multilingual ASR and AST,” arXiv:2509.14128, 2025.

[9] H. Xu, F. Jia, S. Majumdar, H. Huang, S. Watanabe, and B. Ginsburg, “Efficient sequence transduction by jointly predicting tokens and durations,” in Proc. ICML, 2023.

[10] Z. Chen, H. Huang, A. Andrusenko, O. Hrinchuk, K. C. Puvvada, J. Li, S. Ghosh, J. Balam, and B. Ginsburg, “SALM: speech-augmented language model with in-context learning for speech recognition and translation,” in Proc. IEEE ICASSP, 2024, pp. 13521–13525.

[13] Y. Chu et al., “Qwen2-Audio technical report,” arXiv:2407.10759, 2024.

[14] J. Xu et al., “Qwen2.5-Omni technical report,” arXiv:2503.20215, 2025.

[15] KimiTeam, “Kimi-Audio technical report,” arXiv:2504.18425, 2025.

[12] G. Saon et al., “Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities,” arXiv:2505.08699, 2025.

[11] X. Shi et al., “Qwen3-ASR technical report,” arXiv:2601.21337, 2026.

[16] S. Ghosh, Z. Kong, S. Kumar, S. Sakshi, J. Kim, W. Ping, R. Valle, D. Manocha, and B. Catanzaro, “Audio Flamingo 2: an audio-language model with long-audio understanding and expert reasoning abilities,” in Proc. ICML, 2025.

[17] A. Goel, S. Ghosh, J. Kim, S. Kumar, Z. Kong, S.-g. Lee, C.-H. H. Yang, R. Duraiswami, D. Manocha, R. Valle, and B. Catanzaro, “Audio Flamingo 3: advancing audio intelligence with fully open large audiolanguage models,” arXiv:2507.08128, 2025.

[18] T. Olatunji, T. Afonja, A. Yadavalli, C. C. Emezue, S. Singh, B. F. P. Dossou, J. Osuchukwu, S. Osei, A. L. Tonja, N. Etori, and C. Mbataku, “AfriSpeech-200: pan-African accented speech dataset for clinical and general domain ASR,” Trans. Assoc. Comput. Linguist., vol. 11, pp. 1669–1685, 2023.

[19] I. Orife, “Attentive sequence-to-sequence learning for diacritic restoration of Yorùbá language text,” in Proc. INTERSPEECH, 2018, pp. 2848– 2852.

[20] M. Sanni, T. Abdullahi, D. D. Kayande, E. Ayodele, N. A. Etori, M. S. Mollel, M. Yekini, C. Okocha, L. E. Ismaila, F. Omofoye, B. A. Adewale, and T. Olatunji, “AfriSpeech-Dialog: a benchmark dataset for spontaneous English conversations in healthcare and beyond,” in Proc. NAACL-HLT, 2025.

[21] C. Okocha and C. Grant, “Afrispeech semantics: evaluating audio semantic reasoning in spoken language models across domains and accents,” in Proc. ACL, 2026.

[22] B. Awobade, M. Sanni, T. Abdullahi, C. Okocha, K. Ezema, D. D. Kayande, L. E. Ismaila, T. Olatunji, and G. A. Katuka, “AfriVox: probing multilingual and accent robustness of speech LLMs,” in Proc. EACL, 2026, pp. 2672–2690.

[23] G. Z. Ashungafac, M. Sanni, B. Awobade, A. Gichamba, and T. Olatunji, “AfriSpeech-MultiBench: a verticalized multidomain multicountry benchmark suite for African accented English ASR,” in Proc. IJCNLP-AACL, 2025, pp. 3642–3653.

[24] T. Olatunji, T. Afonja, B. F. P. Dossou, A. L. Tonja, C. C. Emezue, A. M. Rufai, and S. Singh, “AfriNames: most ASR models ‘butcher African names,” in Proc. INTERSPEECH, 2023, pp. 5077–5081.