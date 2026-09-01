# VOCAL MUSIC UNDER PHONEME-CONDITIONAL ANALYSIS

Hayoon Kim<sup>♭♮</sup>

Kyogu Lee<sup>♭♮♯†</sup>

<sup>♭</sup>Music and Audio Research Group, Seoul National University <sup>♮</sup>Department of Intelligence and Information, Seoul National University <sup>♯</sup>AIIS, Seoul National University <sup>†</sup>IPAI, Seoul National University

{hyway, kglee}@snu.ac.kr

## ABSTRACT

The vocal music of each language carries a distinctive sonic identity, even without instrumental accompaniment. We ask whether these differences are measurable and traceable to specific phonemes. To tackle this question, we introduce phoneme-conditional analysis, which isolates the acoustic effect of typologically distinctive phonemes by comparing marker syllables against matched non-marker controls within the same song, holding singer, melody, and genre constant. Across nine typologically diverse languages and thousands of songs, we measure effects along five acoustic dimensions. Song-level profiles built from these effects identify the language of an unaccompanied vocal at 85.5% balanced accuracy in a nineway classification with folds grouped by artist; whether the separability arises by accumulation of the phoneme-local effects themselves is left open. Our findings suggest that phonological structure leaves systematic and measurable traces in how each language is sung.

## 1. INTRODUCTION

Arabic popular music and Japanese pop sound fundamentally different, not merely in language but in how pitch, timing, and vocal quality themselves behave. Arabic singing favors dense micro-ornamentation around a narrow melodic core and flexes syllable durations around the beat [1]; Japanese pop distributes syllables with nearmetronomic uniformity and avoids melisma [2]. Neither pattern is genre-specific, since Arabic ornamentation persists from maqam to contemporary pop [3] and Japanese syllabic evenness from enka to J-pop [4]—suggesting these patterns reflect the languages themselves, not musical convention alone.

The hypothesis that linguistic structures shape musical patterns is supported by a robust body of empirical evidence. Rhythmic variability in instrumental music mirrors the prosodic timing of the composer’s native language [5]; lexical tone constrains melodic contour, near-obligatorily in Cantonese pop and more loosely in Mandarin [6]; and text-setting maps linguistic stress to musical meter along language-specific phonological lines [7].

Despite these insights, critical gaps limit the explanatory power of the existing literature. First, rhythm studies have relied on symbolic notation [8], where speech–music correlations often weaken or reverse; sub-segmental durations that survive quantization to the metrical grid go unmeasured (Resolution gap). Second, tone-melody research has established how pitch-bearing features shape melody, yet consonant inventories’ influence on singing remains virtually unexamined (Phonological gap). Third, rhythm, tone, and timbre are typically studied in isolation on small samples [9], leaving no unified synthesis of a language’s vocal identity (Integration gap).

To bridge these gaps, we propose three hypotheses (H1– H3) spanning sub-segmental mechanics to systemic organization as elaborated in Figure 1. We rigorously test H1, provide preliminary evidence for H2, and defer H3 to future work.

H1 (Phoneme-local constraints) posits that the articulatory and acoustic properties of individual speech sounds physically affect the musical realization of their host syllables: a pharyngeal constriction alters the formant structure of adjacent vowels, a complex onset cluster delays the syllable’s perceptual center, a moraic nasal demands its own rhythmic slot. Being consequences of vocal tract physics during singing, these effects are isolable through within-song matched controls. H2 (Emergent conventions) holds that sufficiently frequent H1 effects come to characterize a language’s vocal music at large—its default rhythmic profile, ornamentation strategies, and characteristic vocal quality. H3 (Tradition autonomy) proposes that such conventions persist beyond the phonological contexts that produced them, as when a singer’s native-language habits transfer to second-language performance.

Guided by this framework, our contributions are threefold: (1) a three-level framework separating phonemelocal mechanics from language-wide convention, which attributes the weak notation-level rhythm correlations to measurement resolution; (2) a phoneme-conditional analysis method using within-song matched controls; and (3) application to 9 languages demonstrating language-specific acoustic signatures, enabling 85.5% nine-way classification accuracy with artist-grouped folds. <sup>1</sup>

## Three-Level Hypothesis

![](images/644c976efe58e97a203e8f0e1f3aafa524598bd76862a910df3c964115617977.jpg)

H1 Phoneme-level articulatory and acoustic properties directly shape the musical realization of their host syllables H2 Recurrent H1 effects accumulate to form language-specific patterns in vocal music H3 These patterns become conventionalized and persist as part of a musical tradition’s identity

Figure 1. Three-level hypothesis linking phoneme-level constraints to language-specific musical patterns and their conventionalization in musical traditions.

## 2. RELATED WORKS

## 2.1 Rhythmic Variations and the Vocal Paradox

The normalized Pairwise Variability Index (nPVI) [10] is the standard metric for comparing linguistic and musical rhythm. Following the English–French comparison of instrumental themes [5], the stress-timed/syllable-timed mirroring has been replicated across languages [11], genres [12], and periods [13], but in vocal music these differences diminish or reverse [8]—weakest where the language is actually sung. Notated durations take few distinct values, so nPVI over them largely counts adjacent notes written equal [14]: symbolic notation, rather than the speech–music link itself, is the limiting factor (Resolution gap). We therefore compute nPVI over forced-aligned phoneme durations.

## 2.2 Tone–Melody Correspondence and its Boundaries

Lexical tone constrains melodic contour to a traditiondependent degree, from over 90% parallel tone–melody motion in Cantonese pop [6] to markedly looser alignment elsewhere [15]. Methodologically this literature converges on the bigram level: what matters is whether successive tone and pitch movements are parallel or contrary [16, 17]. Text-setting work extends this pitch-centric paradigm to prosodic weight and to syllable perceptual centers, the instant a syllable is heard to begin, which its onset consonants displace [18, 19]; but the acoustic consequences of segmental inventories—pharyngeals, ejectives, retroflexes—remain unaddressed (Phonological gap).

## 2.3 Phonetic Baselines and Multilingual Corpora

Marker selection rests on phonemes with wellcharacterized acoustic signatures [20]: F3 lowering for English /ô/ [21], F2 lowering for Arabic emphatics [22], VOT and f0-onset separation in the Korean laryngeal contrast [23,24], lengthening of French nasal vowels [25], and closure-duration ratios for Japanese geminates [26, 27]. These are the denominators of our preservation ratios (Table 4). On the corpus side, multilingual singing datasets [28] and anthropological surveys [9, 29] establish cross-cultural regularities at the level of whole utterances or pieces; songs are globally slower, higher, and more pitch-stable than speech [30]. What remains open is the mechanism—which phonological properties produce these differences, and how phoneme-local effects aggregate into a tradition’s characteristic sound (Integration gap), which our within-song design targets.

## 3. METHOD

## 3.1 Language Selection

We select nine languages (Table 1) spanning six families and three prosodic types (stress-, syllable-, and moratimed), maximizing typological diversity subject to chartdata availability. Each contributes one or more markers: phonemes typologically rare (<20% of the world’s languages per WALS/PHOIBLE [31, 32]) and acoustically measurable from audio, the one exception being French nasal vowels at ∼25%, admitted for the sake of a Romance language. Seven languages carry pure Type A markers, whose articulatory properties constrain the sung syllable; Mandarin and Swedish add Type T (tonal or pitchaccent) markers, testing whether F0-based contrasts survive melodic constraint. For Arabic we use the Egyptian dialect [33].

## 3.2 Corpus Construction

For each language we sample 1,000–2,100 songs from Spotify national daily charts [34] via a public aggregator [35], matching metadata to YouTube audio-only streams at 48 kHz / 24-bit FLAC. Language identification accepts a track when at least two of four signals agree— GlotLID on title/description, a channel allowlist, MMS-LID-4017 on audio, and GlotLID on Whisper transcripts— and duplicates are removed by Chromaprint fingerprinting. After separation, transcription, and alignment, 9,331 tracks carry a phone-level alignment, of which 5,024 pass the alignment-agreement gate of Section 3.3 (370–751 per language). All analyses use this gated set; we use only metadata and feature vectors.

## 3.3 Processing Pipeline

Source separation — We isolate vocals using Mel-Band RoFormer [36], retaining only tracks with an estimated SDR greater than 3 dB.

Lyrics and G2P — Lyrics come from pre-collected databases [37–39]; grapheme-to-phoneme conversion uses language-specific tools [40–44] (Table 1).

Forced alignment — Each track is aligned in a single CTC pass over the whole song, avoiding dependence on ASR word timestamps. Two acoustic models run in parallel and are averaged: MMS Forced Alignment [45] over uroman transliteration [46], and a wav2vec2 CTC model emitting espeak IPA symbols directly, fine-tuned on 5,062 sung utterances from GTSinger [28]. Their errors are nearly independent (� = 0.09 on NUS-48E) and averaging helps:

<table><tr><td>Language</td><td>Family</td><td>Type</td><td>Primary marker</td><td>Rarity</td><td>Acoustic cue</td><td>G2P</td><td>Charts</td></tr><tr><td>Arabic (Eg.)</td><td>Afroasiatic</td><td>A</td><td>Pharyngeal /9 ħ/, emphatic  $/ \mathrm { t } ^ { \mathrm { S } } \mathrm { \ s } ^ { \mathrm { S } } /$ </td><td>&lt;5%</td><td>F1↑, F2↓</td><td>epitran</td><td>EG</td></tr><tr><td>English</td><td>IE-Germanic</td><td>A</td><td>Schwa /ə/, approximant /1/</td><td>&lt;5%</td><td>F3 lowering</td><td>espeak</td><td>US, UK</td></tr><tr><td>French</td><td>IE-Romance</td><td>A</td><td>Nasal V / 5 ă/, uvular /ı/</td><td>~25%</td><td>A1–P0, F2</td><td>epitran</td><td>FR</td></tr><tr><td>Hindi</td><td>IE-Indo-Aryan</td><td>A</td><td>Retroflex /t q/, breathy  $/ \mathrm { b } ^ { \mathrm { H } } \mathrm { d } ^ { \mathrm { H } } /$ </td><td>&lt;5%</td><td>F3 trans., H1–H2</td><td>epitran</td><td>IN</td></tr><tr><td>Japanese</td><td>Japonic</td><td>A</td><td>Moraic geminates /k: t:/, nasal /N/, bilabial /Φ/</td><td>&lt;5%</td><td>Closure dur.</td><td>pykakasi</td><td>JP</td></tr><tr><td>Korean</td><td>Koreanic</td><td>A</td><td>3-way laryngeal (asp/tense/lax)</td><td>&lt;1%</td><td>VOT, H1–H2</td><td>g2pK2</td><td>KR</td></tr><tr><td>Turkish</td><td>Turkic</td><td>A</td><td>High back unrounded /u/</td><td>~6%</td><td>F1↓, F2↓</td><td>epitran</td><td>TR</td></tr><tr><td>Mandarin</td><td>Sino-Tibetan</td><td>A+T</td><td>Retroflex /t §/ + 4 tones</td><td>~6%</td><td>Spec. CÓG, F0</td><td>pypinyin</td><td>TW</td></tr><tr><td>Swedish</td><td>IE-Germanic</td><td>A+T</td><td>Sje-sound /f/ + long V, pitch accent</td><td>&lt;1%</td><td>Spec. COG, F0</td><td>eSpeak</td><td>SE</td></tr></table>

Table 1. Language selection and phonological markers. Type A: articulatory markers measured via within-song matched comparison. A+T: articulatory plus tonal or pitch-accent contrasts that may interact with melodic pitch. G2P: graphemeto-phoneme tool; all output is IPA, Unicode NFC-normalized. Charts: Spotify daily charts by country. Rarity: approximate share of the world’s languages with the feature; where a marker bundles several segments, the rarest component is given.

the blend places 74.6% of onsets within 50 ms against 72.3% and 69.5% for the two models. Blank frames between CTC token spans are redistributed to neighboring phones, and each onset is snapped to the strongest spectral change within ±30 ms when locally prominent. Because both models see the same phone sequence, their median per-track disagreement serves as a ground-truth-free confidence signal; tracks disagreeing by more than 200 ms are excluded (alignment-agreement gate). The gate selects on transcript completeness rather than on performer: excluded tracks carry a median of 402 lyric characters against 1,678 for retained ones, and 723 of the 1,914 retained artists also appear among the excluded.

F0 extraction — Fundamental frequency comes from an ensemble of FCPE [47] and RMVPE [48]. Frames below 0.3 RMVPE voicing confidence are masked; frames where the estimators disagree by >2 semitones receive halved confidence and are re-thresholded, filtering instrumental leakage from imperfect separation.

Feature extraction — For each aligned phoneme we extract features from the central 70% of the segment, in five modules: articulatory impact (F1–F3, spectral tilt, H1– H2, HNR, and F1/F2 deltas to neighbouring phonemes), melodic deviation (F0 mean, range, slope, sd), rhythmic adaptation (duration), melisma (notes per phoneme), and timbral signature (spectral centroid, MFCC). Text-setting alignment applies only to Mandarin and Swedish. Songlevel aggregates are nPVI and melisma rate.

## 3.4 Alignment Validation

On NUS-48E [49] (47 sung recordings with manual phone boundaries) our aligner places onsets with a median error of 23 ms, 74.6% within 50 ms, and recovers durations at a median 0.85 of true length. The underestimate is compressive rather than constant, inflating phones under 81 ms (pred/true 1.38) and cutting those over 210 ms (0.53). Two consequences follow for Section 4.3. Random boundary error inflates variance without biasing a paired difference: with the per-phone error centred at −18 ms and sd = 148 ms—a conservative figure, set by a tail of gross misplacements—it adds 0.9–1.7 ms of standard error over the pairs behind the three duration effects. Compression attenuates: a long (≥150 ms) versus short (≤100 ms) pairing of the same class recovers 34% of the true difference (+78 against +228 ms), so our millisecond magnitudes are lower bounds; being monotone in true duration it shrinks a contrast without reversing its sign, though this does not cover class-specific bias. As a floor on the aligner’s own contribution, phones paired by class and true-duration bin—true difference +0.06 ms—measure −1.6 ms, 95% CI [−4.0, +0.6]; this is English sung material, the only corpus here with manual boundaries, assumed to transfer. Two alternatives do markedly worse. Distributing lyric words across ASR word intervals by index gives 443 ms median onset error, 14.6% within 50 ms, and a 0.18 duration ratio. Given the same phone sequences, the Montreal Forced Aligner [50] aligned only 8 of 47 sung recordings usably; on spoken recordings by the same singers—a different subset—it reaches 20.0 ms against our 22.9 ms.

## 3.5 Marker Classification and Matched Pairing

Each phoneme is labeled marker or non-marker by language-specific rules from the phonological typology literature (Table 1). For each marker we select a matched control in the same song by additive scoring: phone-type match (weight 10; vowel markers additionally matched on height×backness) with temporal proximity $( 1 / ( 1 + | i - j | ) )$ as tiebreaker, each control reused at most three times. Korean is further decomposed into aspirated–lax, tense–lax, and aspirated–tense comparisons; Type T markers, where every syllable bears a tone, instead pair contour tones against level tones matched on rime; Swedish pitch accent yields too few such pairs to fit.

## 3.6 Statistical Analysis

Within-song effect sizes — For each language × dimension cell we compute paired Cohen’s � between marker and matched control, taking max |�| across the features in that dimension, so the reported magnitude is an upper bound on the dimension’s effect. Significance uses permutation tests (5,000 iterations) with the null built on the same maximum statistic, Bonferroni-corrected across the 50 cells $( \alpha = 0 . 0 5 / 5 0 )$ . Because each cell rests on 3,855– 153,849 pairs, significance is near-automatic; we therefore fix a practical floor of $| d | \geq 0 . 1$ in advance and read cells below it as significant but negligible.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Examples</td><td colspan="6">Transition</td><td colspan="3"> $\Delta \mathsf { F }$ </td><td colspan="3">F0</td><td rowspan="2">|a|</td></tr><tr><td> $\overline { { { \mathrm { \Pi } } _ { \mathrm { F 1 } } } }$ </td><td>F2</td><td>F3</td><td>tilt</td><td>HNR</td><td>H1-H2</td><td> $\overline { { \mathrm { ~ \ F 1 ~ } } }$ </td><td> $\mathrm { F } 2$ </td><td> $\overline { { \mathrm { F } 3 } }$ </td><td> $\operatorname { d e v }$ </td><td>range</td><td>slope</td></tr><tr><td>Retroflex</td><td> $\mathrm { / t s / }$ </td><td>+0.16</td><td>+0.12</td><td>+0.23</td><td>+0.23</td><td>–0.25</td><td>+0.04</td><td>+0.12</td><td>+0.18</td><td>+0.12</td><td>+0.04</td><td>+0.06</td><td>-0.08</td><td>0.14</td></tr><tr><td>High back unrounded</td><td> $/ \mathrm { m } I$ </td><td>–0.08</td><td>+0.10</td><td>-0.06</td><td>+0.24</td><td>-0.07</td><td>+0.15</td><td></td><td></td><td></td><td>+0.01</td><td>-0.03</td><td>+0.00</td><td>0.08</td></tr><tr><td>Tense</td><td> $ { / \mathrm { p } }  { \mathrm { u } }$ </td><td>-0.05</td><td>+0.07</td><td>+0.04</td><td>+0.10</td><td>–0.13</td><td>+0.04</td><td>-0.02</td><td>+0.05</td><td>+0.03</td><td>+0.14</td><td>+0.01</td><td>+0.04</td><td>0.06</td></tr><tr><td>Bilabial fricative</td><td> $/ \Phi \mathrm { \{ \} / \mathrm { \Omega } }$ </td><td>+0.01</td><td>-0.23</td><td>+0.02</td><td>-0.09</td><td>-0.00</td><td>+0.05</td><td>-0.00</td><td>–0.07</td><td>+0.01</td><td>–0.15</td><td>+0.07</td><td>+0.01</td><td>0.06</td></tr><tr><td>Aspirated</td><td> $/ \mathrm { p } ^ { \mathrm { h } } \mathrm { k } ^ { \mathrm { h } } /$ </td><td>+0.04</td><td>+0.06</td><td>+0.02</td><td>+0.04</td><td>-0.20</td><td>+0.13</td><td>+0.02</td><td>+0.02</td><td>+0.02</td><td>+0.01</td><td>+0.06</td><td>-0.00</td><td>0.05</td></tr><tr><td>Nasal vowel</td><td> $/ \tilde { \varepsilon } \tilde { \textbf { 0 } } \tilde { \mathbf { 0 } }$ </td><td>+0.03</td><td>-0.03</td><td>+0.07</td><td>-0.02</td><td>+0.14</td><td>+0.08</td><td></td><td></td><td></td><td>+0.02</td><td>+0.02</td><td>+0.02</td><td>0.05</td></tr><tr><td>Schwa/rhotic</td><td> $/ \mathrm { { } _ { 9 \perp } } \partial ^ { \parallel }$ </td><td>-0.02</td><td>+0.05</td><td>+0.05</td><td>+0.15</td><td>–0.10</td><td>+0.05</td><td>+0.00</td><td>-0.00</td><td>-0.02</td><td>-0.07</td><td>-0.03</td><td>-0.03</td><td>0.05</td></tr><tr><td>Pharyngeal/emphatic</td><td> $/ \mathrm { { C } \hbar t ^ { \mathrm { { \mathrm { { f } } } } } / }$ </td><td>+0.09</td><td>-0.07</td><td>+0.01</td><td>-0.04</td><td>-0.06</td><td>-0.02</td><td>+0.02</td><td>-0.02</td><td>+0.01</td><td>-0.06</td><td>+0.06</td><td>-0.00</td><td>0.04</td></tr><tr><td>Uvular</td><td> $/ \mathrm { q } \chi \mathrm { B } /$ </td><td>+0.02</td><td>-0.01</td><td>+0.01</td><td>-0.00</td><td>-0.03</td><td>–0.07</td><td>-0.00</td><td>-0.02</td><td>-0.01</td><td>+0.02</td><td>+0.01</td><td>-0.02</td><td>0.02</td></tr></table>

Table 2. Paired Cohen’s � for vowels adjacent to marked versus control consonants. — marks cells with <30 pairs.

Linear mixed models — Per-language LMMs of the form feature ∼ is\_marker + beat\_phase + (1|song), fitted by REML, control for metrical position; where they fail to converge we fall back to Wilcoxon signed-rank tests. Language classification — Song-level feature vectors (35-dimensional: 20 marker–control difference statistics + 15 raw phoneme distribution statistics) are classified by a Random Forest (200 trees) with 5-fold cross-validation grouped by artist, balanced by undersampling to the smallest class. A song enters only if all 35 features are defined, costing 0–153 tracks per language (Hindi 442 → 289) and leaving 289 songs per language over 1,323 artists. Artists recur in chart data—2.0 songs per artist across those 2,601 songs—so a song-level split would let the model reach a language through its performers; grouping removes that route at a cost of 1.2 accuracy points. Splitting the vector into its two blocks separates what the marker contrasts contribute from what the raw distributions do: the 20 difference statistics alone reach 79.4% and the 15 distribution statistics alone 55.7%, so the distributions are worth 6.1 points of the 85.5%.

## 4. RESULTS

All results are computed on the 5,024 gated tracks (Section 3.3). Significant effects appear in at least three acoustic dimensions for 9 of the 10 marker sets, the exception being Hindi. Which channel dominates depends on the metric: timbre and voice quality carry the largest effect sizes, while formant transitions are the best preserved of the channels, and then only for retroflexion (Section 4.3).

## 4.1 Phoneme Properties Shape Singing Acoustics

Table 2 presents Cohen’s � for vowels adjacent to each consonant category, pooled across languages. It reports individual features of the adjacent vowel, whereas Table 3 reports the largest |�| over a dimension, so the two are not on the same scale. The dominant channels are voice quality and spectral tilt rather than formant transitions. Retroflexes show the largest single cell (� = −0.25, trans HNR; the same row carries +0.23 on trans tilt and F3, so the ranking is by |�|, not direction), followed by the Turkish high back unrounded vowel (+0.24, trans tilt) and bilabial /F (−0.23, trans F2). Korean aspirated and tense stops separate by channel rather than by magnitude (−0.20 HNR vs.

+0.14 F0 deviation), consistent with their phonation-type distinction. Of the 102 filled cells, 21 clear the $| d | \geq 0 . 1$ floor, 11 reach 0.15 and 6 reach 0.20. These are small effects measured precisely rather than large ones.

## 4.2 Language-Specific Fingerprints

Table 3 shows paired Cohen’s � across five modules per language, ranked by |�|. Mandarin’s dual markers separate along melisma and timbral magnitude rather than along a temporal/spectral axis. Both sets act most strongly on timbre—retroflex markers (Type A, 0.25) carry the largest single effect in the corpus (� = −0.80, MFCC-1 over 25,899 pairs), the tonal set (Type T, 0.20) less than half that (−0.36). What is the tonal set’s alone is melisma (+0.28 vs. −0.07); rhythmic lengthening is shared (+0.14 vs. +0.11, � = +24.7 ms; millisecond effects in this section are LMM coefficients, against the raw paired differences of Table 4), and melodic deviation distinguishes neither $\left( + 0 . 0 7 \mathrm { v s . } + 0 . 0 6 \right)$ . The timbral column is an MFCC effect in most rows and a spectral centroid for Korean and Mandarin (T), which is why the −0.80 cell has no counterpart in Table 2, whose columns carry no cepstral features.

Korean is the strongest consonant-marker language, driven by timbre and melisma, followed by Arabic, Turkish and English, each with a timbral component of the same order. The ranking is largely a timbre ranking: dropping that column reorders every row above Japanese. Six marker sets reach significance in all five modules and Japanese three, with melisma avoidance (−0.11) and shortening (−16.4 ms). Swedish’s tonal markers are too sparse to fit separately, so only Swedish (A) appears. Hindi reaches the fewest modules and is least powered (4,405 pairs), so its nulls are uninformative, not negative. Pair counts differ by an order of magnitude, so asterisks are not comparable between rows and the ranking is by |�|.

Splitting Korean pairs by laryngeal category, aspirated– lax and tense–lax yield nearly identical effects while aspirated–tense differences are minimal: aspirated and tense stops pattern together against lax. This follows the Seoul Korean tonogenetic merger [51], in which F0 onset has largely replaced VOT as the primary laryngeal cue. Melodic pitch does not fully override that cue—the contrast retains ∼30% of its speech magnitude—so the grouping in singing follows the same F0-based partition as contemporary speech.

<table><tr><td colspan="7"> $\begin{array}{c} \begin{array} { r l } & { \mathsf { \Gamma } _ { \mathsf { N } } ^ { \mathrm { o v } ^ { \mathrm { { a c v } ^ { \mathrm { { a } } } } } } \mathsf { \Gamma } _ { \mathrm { N } } ^ { \mathrm { o v } ^ { \mathrm { { a } } \mathrm { { i } } \mathrm { { c } } } } } \end{array} \overset { \mathrm { ~ } } { \underset { \mathrm { P } } { \mathrm { e } } ^ { \mathrm { { i } } \mathrm { o } ^ { \mathrm { { i } } \mathrm { o } ^ { \mathrm { { i } } \mathrm { c } } } } } \mathsf { \Gamma } _ { \mathrm { N } } ^ { \mathrm { o u s } ^ { \mathrm { { c } } } } \mathsf { \Gamma } _ { \mathrm { N } } ^ { \mathrm { o u s } ^ { \mathrm { { i } } \mathrm { c } } } \mathsf { \Gamma } _ { \mathrm { N } } ^ { \mathrm { ~ } } \mathrm { { \Gamma } } _ { \mathrm { i } } ^ { \mathrm { { i } } \mathrm { { s } } ^ { \mathrm { { i } } \mathrm { { s } } ^ { \mathrm { { a } } } } } \mathsf { \Gamma } _ { \mathrm { N } } ^ { \mathrm { { i } } \mathrm { { c } } ^ { \mathrm { { a } } \mathrm { { i } } \mathrm { { c } } ^ { \mathrm { { a } } \mathrm { { i } } \mathrm { { c } } } } }  \end{array}$ </td></tr><tr><td>Marker set</td><td></td><td></td><td></td><td></td><td>|a|</td><td>Pairs</td></tr><tr><td>Mandarin (A)</td><td></td><td> $- 0 . 2 0 ^ { * } + 0 . 0 7 ^ { * } + 0 . 1 1 ^ { * } - 0 . 0 7 ^ { * }$ </td><td></td><td>–0.80*</td><td>0.25</td><td>25,899</td></tr><tr><td>Korean</td><td></td><td> $- 0 . 2 1 ^ { * } + 0 . 0 9 ^ { * } - 0 . 0 7 ^ { * } - 0 . 2 6 ^ { * } \overline { { + 0 . 4 7 ^ { * } } }$ </td><td></td><td></td><td>0.22</td><td>48,492</td></tr><tr><td>Mandarin (T)</td><td></td><td> $- 0 . 1 6 ^ { * } + 0 . 0 6 ^ { * } + 0 . 1 4 ^ { * } + 0 . 2 8 ^ { * } - 0 . 3 6 ^ { * }$ </td><td></td><td></td><td>0.20</td><td>153,849</td></tr><tr><td>Arabic</td><td></td><td> $+ 0 . 1 7 ^ { * } - 0 . 0 6 ^ { * } + 0 . 0 3 ^ { * } - 0 . 0 5 ^ { * } - 0 . 4 2 ^ { * }$ </td><td></td><td></td><td>0.15</td><td>25,498</td></tr><tr><td>Turkish</td><td></td><td> $+ 0 . 2 3 ^ { * } + 0 . 0 5 ^ { * } - 0 . 0 6 ^ { * } + 0 . 0 0 ~ - 0 . 3 5 ^ { * }$ </td><td></td><td></td><td>0.14</td><td>32,277</td></tr><tr><td>English</td><td></td><td> $+ 0 . 1 1 ^ { * } - 0 . 0 6 ^ { * } - 0 . 2 2 ^ { * } - 0 . 0 8 ^ { * } + 0 . 2 1 ^ { * }$ </td><td></td><td></td><td>0.14</td><td>78,968</td></tr><tr><td>French</td><td></td><td> $+ 0 . 1 5 ^ { * } + 0 . 0 2 ^ { * } + 0 . 1 5 ^ { * } + 0 . 0 8 ^ { * } + 0 . 2 0 ^ { * }$ </td><td></td><td></td><td>0.12</td><td>34,020</td></tr><tr><td>Japanese</td><td></td><td> $- 0 . 0 6 \ - 0 . 0 8 ^ { \ast } - 0 . 1 0 ^ { \ast } - 0 . 1 1 ^ { \ast } - 0 . 0 6$ </td><td></td><td></td><td>0.08</td><td>15,213</td></tr><tr><td>Hindi</td><td></td><td> $+ 0 . 0 8 + 0 . 0 7 ^ { * } - 0 . 0 3 + 0 . 0 1 - 0 . 2 0 ^ { * }$ </td><td></td><td></td><td>0.08</td><td>4,405</td></tr><tr><td>Swedish (A)</td><td></td><td> $+ 0 . 0 6 ^ { \ast } + 0 . 0 2 ^ { \ast } + 0 . 0 1$ </td><td>+0.01</td><td>+0.12*</td><td>0.05</td><td>65,536</td></tr></table>

Table 3. Paired Cohen’s � between markers and matched controls over five dimensions. <sup>\*</sup>: Bonferroni-corrected $\alpha = 0 . 0 5 / 5 0$ . pairs: matched pairs per set.

## 4.3 Speech-to-Singing Preservation Hierarchy

Table 4 compares singing effects against published speech baselines. Every row rests on enough pairs to be significant at $p ~ < ~ . 0 0 1$ , so the argument is about the ratio to the baseline rather than about whether the effect is nonzero. Preservation is channel-dependent and does not follow the effect-size ranking: formant coarticulation is best preserved where the constriction is a tongue-body gesture, while duration is uniformly attenuated and in two cases reverses sign.

Duration attenuated — No durational contrast reaches half its speech baseline. Mandarin tonal lengthening retains ∼30% (+26 ms vs. ∼50–120 ms [52]) and French nasal vowels ∼19% (+7 vs. +40 ms). Korean aspirated/tense stops reverse: 10 ms shorter than lax controls in singing, against ∼25–40 ms longer in speech. Japanese geminates remain shortened (−15 ms), consistent with absorption into the beat grid. These three are 4–9× the aligner’s null floor (1.6 ms, Section 3.4) against 16× for Mandarin, and both reversals fall on long consonants, where a boundary inside the closure does most damage; we therefore read them as directions rather than magnitudes.

Timbral shape attenuated — The Mandarin retroflex centroid shift measures +1,342 Hz, ∼45% of its speech baseline: attenuated, but the largest spectral effect in the corpus.

Formants reduced outside retroflexion — The Mandarin retroflex F2 shift is fully preserved (+543 vs. ∼500 Hz) and Hindi retroflex F3 lowering reaches ∼33% (−82 vs. −250 Hz), but English /ô/ F3 retains only ∼5% (−37 Hz) and Arabic emphatic F2 ∼4% (−10 Hz), the one cell we grade absent. The channel survives where the constriction is a tongue-body gesture the singer must make anyway, not where it is a secondary articulation.

Pitch and voice quality attenuated — The Korean laryngeal F0 contrast retains ∼30% (+7.7 Hz vs. +20–30 Hz) rather than vanishing, and the lax–tense H1–H2 contrast ∼19% (−1.5 vs. ∼8 dB), while Mandarin tone contours stay near the floor (+1.6 Hz). The pattern is competition for the same articulators: the tongue body is not the melody’s to use, while duration and F0 are, and give way where it makes a demand.

![](images/bd195a0dd9ef99b7075d39a78dca87c7254eeb7360dcffba9de9d32b9df49285.jpg)

![](images/ec75ac291a61f287dc932269749917044860dc0a338d6fd0154e79f6b4c288d7.jpg)  
Figure 2. Song-level nPVI (left) and melisma rate (right), 5,024 gated tracks, 370–751 per language. Boxes span the IQR, median in red, whiskers 1.5 IQR, outliers omitted. Kruskal–Wallis $H = 7 3 7$ and 533 on 8 df; language accounts for $\varepsilon ^ { 2 } = 0 . 1 5$ and 0.11 of the rank variance.

## 4.4 Song-Level Distributions

Figure 2 shows nPVI and melisma rate over forced-aligned phoneme durations rather than notated values. Both separate the nine languages far beyond chance (Kruskal–Wallis $H  { \mathrm { ~ = ~ } } 7 3 7$ and 533 on 8 df, $\varepsilon ^ { 2 } = 0 . 1 5$ and 0.11). Subsegmental timing therefore carries language information that notation-based measures do not access. It is not the rhythm-class dimension: the ordering does not track the stress/syllable/mora split, so this is separation rather than a recovery of the speech–music nPVI correlation.

## 4.5 Language Classification

Figure 3 shows the confusion matrix for a Random Forest on 35-dimensional song-level feature vectors, achieving 85.5% balanced accuracy (chance = 11.1%) with artistgrouped folds. Arabic (95%) and Turkish (94%) are the most identifiable and Japanese the least at 73%; confusions are structured, not diffuse, pairing Swedish with English and Mandarin with Korean. Grouping costs 1.2 points against a song-level split (86.7%), bounding what comes from performer identity. The block ablation locates the separability in the marker–control contrasts.

Neither marker rate nor effect magnitude predicts perlanguage recall (Spearman $\rho = + 0 . 2 7$ and +0.50, � = 0.49 and 0.17), so H2’s aggregation prediction is unsupported at $n \ = \ 9 .$ This does not contradict the block ablation. The classifier reads a song’s whole profile of contrasts—which dimensions move, in which direction, and how consistently within the song—while the rank test asks only whether languages with larger average effects are easier to identify. A language can separate on a distinctive combination of small contrasts with no large one, and a large effect buys nothing if a competing language carries one of the same size in the same dimension: recall depends on how far a language sits from the others relative to its own spread, not on magnitude against a within-song control. H2 predicts the second drives the first, and nine languages have little power to see it—the effect-magnitude correlation carries the sign H2 requires, but at � = 9 that is not separable from noise. We therefore read the prediction as unsupported rather than refuted, as with the Hindi nulls of Section 4.2. Vocals are source-separated, so residual accompaniment remains an uncontrolled cue.

<table><tr><td>Category</td><td>Language</td><td>Feature</td><td>Pairs</td><td>Speech</td><td>Singing</td><td>Ratio</td><td>Verdict</td></tr><tr><td rowspan="6">Spectral /</td><td>Mandarin (A)</td><td>Retroflex F2 shift [53]</td><td>25,688</td><td>~500 Hz</td><td>+543 Hz</td><td>≥109%</td><td>preserved</td></tr><tr><td>Mandarin (A)</td><td>Retroflex centroid [54]</td><td>25,899</td><td>~3,000 Hz</td><td>+1,342 Hz</td><td>~45%</td><td>attenuated</td></tr><tr><td>Hindi</td><td>Retroflex F3 lowering [55]</td><td>4,347</td><td>-250 Hz</td><td>−82 Hz</td><td>~33%</td><td>attenuated</td></tr><tr><td>English</td><td>// adj. vowel F3 [21]</td><td>78,094</td><td>-700 Hz</td><td>-37 Hz</td><td>~5%</td><td>attenuated</td></tr><tr><td>Arabic</td><td>Emphatic F2 lowering [22]</td><td>25,215</td><td>-250 Hz</td><td>-10 Hz</td><td>~4%</td><td>absent</td></tr><tr><td>Mandarin (T)</td><td>Tone adj. vowel F2</td><td>152,622</td><td></td><td>-204 Hz</td><td></td><td>novel</td></tr><tr><td rowspan="3">Voice quality</td><td>Korean</td><td>Lax-tense H1–H2 [24]</td><td>36,985</td><td>~8 dB</td><td>-1.5 dB</td><td>~19%</td><td>attenuated</td></tr><tr><td>Mandarin (A)</td><td>Retroflex adj. H1–H2</td><td>22,161</td><td></td><td>-0.3 dB</td><td></td><td>novel</td></tr><tr><td>Mandarin (T)</td><td>Tone adj. H1–H2</td><td>142,166</td><td></td><td>+0.4 dB</td><td></td><td>novel</td></tr><tr><td rowspan="5">Temporal / rhythmic</td><td>Mandarin (T)</td><td>Tone syllable dur. [52]</td><td>153,849</td><td>~50–120 ms</td><td>+26 ms</td><td>~30%†</td><td>attenuated</td></tr><tr><td>French</td><td>Nasal vowel duration [25]</td><td>34,020</td><td>+40 ms</td><td>+7.4 ms</td><td>~19%†</td><td>attenuated</td></tr><tr><td>Korean</td><td>Asp/tense vs lax dur. [24]</td><td>48,492</td><td>~25–40 ms</td><td>-10 ms</td><td>†</td><td>reversed</td></tr><tr><td>Japanese</td><td>Geminate/singleton dur. [26]</td><td>15,213</td><td>+102–204 ms</td><td>-15 ms</td><td>†</td><td>reversed</td></tr><tr><td>Arabic</td><td>Pharyngeal duration</td><td>25,498</td><td></td><td>+4 ms</td><td></td><td>novel</td></tr><tr><td rowspan="2">Pitch (F0)</td><td>Korean</td><td>Asp/tense vs lax F0 [51]</td><td>25,985</td><td>+20-30 Hz</td><td>+7.7 Hz</td><td>~30%</td><td>attenuated</td></tr><tr><td>Mandarin (T)</td><td>Tone contour [56]</td><td>131,759</td><td></td><td>+1.6 Hz</td><td></td><td>novel</td></tr></table>

Table 4. Speech-to-singing preservation ratios: effect sizes documented in related works against our singing measurement. preserved ≥50% of the speech effect, attenuated 5–50%, absent <5%, reversed opposite in sign to speech, so no ratio is defined, novel no baseline and so off the scale. Speech and singing come from different corpora and speakers, so thresholds are conventions. †: the aligner compresses duration contrasts, recovering 34% of a long-versus-short difference (Section 3.4), so temporal ratios are lower bounds; the Japanese baseline converts a 2.0–3.0× closure ratio at our control mean of 102 ms; the tone-contour row has no hertz baseline, and +1.6 Hz is 0.6% of the mean F0 of those syllables.

## 5. CONCLUSION

Main findings — Three results. First, singing keeps the articulatory gestures the tongue body must make regardless of the melody and attenuates the rest: the Mandarin retroflex F2 shift is at least as large in singing as in speech and retroflex timbre reaches ∼45%, while no durational contrast reaches half its speech baseline (the largest, Mandarin tonal lengthening, is ∼30%, and the aligner compresses these), and the Korean and Japanese ones reverse sign. Voice quality is attenuated, not eliminated. Second, the dominant channel is voice quality and spectral tilt rather than formant transitions: phoneme identity in singing is carried by source-spectrum properties more than by vocal tract resonance. Third, song-level vectors built from these contrasts support 85.5% nine-way classification with artist-grouped folds, so the result reflects the language rather than its performers. Since neither marker frequency nor effect magnitude predicts per-language accuracy at � = 9, this establishes separability, not the accumulation H2 predicts.

Discussion — Within-song matched pairs hold singer, genre, and melody constant, so H1 effects are consequences of vocal tract physics rather than musical tradition. At the language level (H2) no such separation is possible— a dark Arabic timbre may reflect cumulative pharyngeal articulation, a tradition that evolved around it, or both. Distinguishing them needs variation that phonology and tradition do not share, which chart data does not supply. We therefore treat H1 as established and H2 as empirically demonstrated but causally open. Our vocals are separated, not natively unaccompanied. For MIR this makes language an audible property of a vocal track, and one that sits in phonation and spectral shape rather than the pitch and timing singing-voice models are usually evaluated on.

![](images/ba90d72262ff24bd5dc5a890f662116ea3aff38584b46abc4fbb273b00aae876.jpg)  
Figure 3. Nine-way language classification from musical prosodic features. Cells are row-normalized recall, omitted below 3%. Folds are grouped by artist.

Limitations and future work — Nine languages exclude click languages, ejective-rich inventories, and richer tone systems. Our 2018–24 chart corpus ties each language to one or two countries. Phoneme boundary errors propagate to every feature and set a floor on what the design can measure: single-phone effects are recoverable only when onset error is well below phone duration. Preservation ratios use published speech baselines from different corpora, speakers, and protocols, so they are order-of-magnitude comparisons rather than matched contrasts. A second constraint is lyric quality—46% of tracks fail the agreement gate, concentrated in Arabic and Swedish, needing better lyrics, not a better aligner. Residual accompaniment survives separation in the timbral and spectral channels, where our largest effects sit. And we measure acoustics, not audibility: a perceptual test is the probe. H3 remains untested; the natural probe is cover songs and multilingual singers.

## 6. ACKNOWLEDGMENTS

This work was partly supported by the Youlchon Foundation (Nongshim Corporation and affiliated companies), Korea [50%]; by the National Research Foundation of Korea (NRF) under grant [No. RS-2025-24683892, 45%]; and by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant [No. RS-2021-II211343, 5%], funded by the Korea government (MSIT).

## 7. AI USAGE STATEMENT

Generative AI tools were used for code development assistance and language polishing. All research design, experiments, analysis, and conclusions were produced by the authors.

## 8. REFERENCES

[1] A. J. Racy, Making music in the Arab world: The culture and artistry of Tarab. Cambridge University Press, 2004, no. 17.

[2] A. Tokita and D. W. Hughes, The Ashgate research companion to Japanese music. Ashgate Publishing, Ltd., 2008.

[3] M. A. Frishkopf, Music and media in the Arab world. American Univ in Cairo Press, 2010, no. 4108-4109.

[4] B. C. Wade, Music in Japan: Experiencing Music, Expressing Culture. New York: Oxford University Press, 2005.

[5] A. D. Patel and J. R. Daniele, “An empirical comparison of rhythm in language and music,” Cognition, vol. 87, no. 1, pp. B35–B45, 2003.

[6] P. C. Wong and R. L. Diehl, “How can the lyrics of a song in a tone language be understood?” Psychology ofMusic, vol. 30, no. 2, pp. 202–209, 2002.

[7] B. Hayes, “Textsetting as constraint conflict,” Towards a typology of poetic forms, pp. 43–61, 2009.

[8] D. Temperley, “Rhythmic variability in european vocal music,” Music Perception: An Interdisciplinary Journal, vol. 35, no. 2, pp. 193–199, 2017.

[9] P. E. Savage, S. Brown, E. Sakai, and T. E. Currie, “Statistical universals reveal the structures and functions of human music,” Proceedings of the National Academy ofSciences, vol. 112, no. 29, pp. 8987–8992, 2015.

[10] E. Grabe and E. L. Low, “Durational variability in speech and the rhythm class hypothesis,” in Laboratory Phonology 7, ser. Phonology and Phonetics, C. Gussenhoven and N. Warner, Eds. Berlin: Mouton de Gruyter, 2002, vol. 4-1, pp. 515–546.

[11] M. Sadakata, P. Desain, H. Honing, A. D. Patel, J. R. Iversen et al., “A cross-cultural study of the rhythm in English and Japanese popular music,” in Proceedings of the international symposium on musical acoustics. ISMA Nara, Japan, 2004, pp. 41–44.

[12] L. VanHandel and T. Song, “The role of meter in compositional style in 19th century French and german art song,” Journal ofNew Music Research, vol. 39, no. 1, pp. 1–11, 2010.

[13] N. Temperley and D. Temperley, “Music-language correlations and the “scotch snap”,” Music Perception, vol. 29, no. 1, pp. 51–63, 2011.

[14] N. Condit-Schultz, “Deconstructing the nPVI: a methodological critique of the normalized pairwise variability index as applied to music,” Music Perception: An Interdisciplinary Journal, vol. 36, no. 3, pp. 300–313, 2019.

[15] J. Kirby and P. Ðô, “Tone-melody correspondence in<sup>˜</sup> vietnamese popular song,” in Proceedings of the International Conference on Tone and Intonation in Europe (TIE), 2016.

[16] L. McPherson and K. M. Ryan, “Tone-tune association in tommo so (dogon) folk songs,” Language, vol. 94, no. 1, pp. 119–156, 2018.

[17] D. R. Ladd and J. Kirby, “Tone–melody matching in tone-language singing,” 2020.

[18] K. M. Ryan, “Syllable weight and natural duration in textsetting popular music in english,” English Language & Linguistics, vol. 26, no. 3, pp. 559–582, 2022.

[19] B. Hayes and A. Kaun, “The role of phonological phrasing in sung and chanted verse,” The linguistic review, vol. 13, no. 3-4, pp. 243–304, 1996.

[20] P. Ladefoged and I. Maddieson, The Sounds of the World’s Languages. Oxford: Blackwell, 1996.

[21] R. E. Hagiwara, Acoustic realizations of American/r/as produced by women and men. University of California, Los Angeles, 1995.

[22] A. Jongman, W. Herd, M. Al-Masri, J. Sereno, and S. Combest, “Acoustics and perception of emphasis in Urban Jordanian Arabic,” Journal of Phonetics, vol. 39, no. 1, pp. 85–95, 2011.

[23] T. Cho and P. Ladefoged, “Variation and universals in VOT: evidence from 18 languages,” Journal of phonetics, vol. 27, no. 2, pp. 207–229, 1999.

[24] T. Cho, S.-A. Jun, and P. Ladefoged, “Acoustic and aerodynamic correlates of Korean stops and fricatives,” Journal ofphonetics, vol. 30, no. 2, pp. 193–228, 2002.

[25] P. Delattre and M. Monnot, “The role of duration in the identification of French nasal vowels,” IRAL: International Review of Applied Linguistics in Language Teaching, vol. 6, no. 3, p. 267, 1968.

[26] M. S. Han, “Acoustic manifestations of mora timing in Japanese,” The Journal of the Acoustical Society of America, vol. 96, no. 1, pp. 73–82, 1994.

[27] S. Kawahara, “The phonetics of sokuon, or geminate obstruents,” Handbook ofJapanese phonetics and phonology, pp. 43–78, 2015.

[28] Y. Zhang, C. Pan, W. Guo, R. Li, Z. Zhu, J. Wang, W. Xu, J. Lu, Z. Hong, C. Wang et al., “GTSinger: A global multi-technique singing corpus with realistic music scores for all singing tasks,” Advances in Neural Information Processing Systems, vol. 37, pp. 1117– 1140, 2024.

[29] S. A. Mehr, M. Singh, D. Knox, D. M. Ketter, D. Pickens-Jones, S. Atwood, C. Lucas, N. Jacoby, A. A. Egner, E. J. Hopkins et al., “Universality and diversity in human song,” Science, vol. 366, no. 6468, p. eaax0868, 2019.

[30] Y. Ozaki, A. Tierney, P. Q. Pfordresher, J. M. McBride, E. Benetos, P. Proutskova, G. Chiba, F. Liu, N. Jacoby, S. C. Purdy et al., “Globally, songs and instrumental melodies are slower and higher and use more stable pitches than speech: A registered report,” Science advances, vol. 10, no. 20, p. eadm9797, 2024.

[31] M. S. Dryer and M. Haspelmath, Eds., WALS Online (v2020.4). Leipzig: Max Planck Institute for Evolutionary Anthropology, 2013. [Online]. Available: https://wals.info

[32] S. Moran and D. McCloy, Eds., PHOIBLE 2.0. Jena: Max Planck Institute for the Science of Human History, 2019. [Online]. Available: https://phoible.org

[33] V. Danielson, The Voice ofEgypt: Umm Kulthum, Arabic Song, and Egyptian Society in the Twentieth Century. University of Chicago Press, 2008.

[34] Spotify, “Spotify charts,” https://charts.spotify.com/, 2026, accessed: 2026-04-28.

[35] Kworb, “Spotify charts aggregated statistics,” https:// kworb.net/spotify/, 2026, accessed: 2026-04-28.

[36] J.-C. Wang, W.-T. Lu, and J. Chen, “Mel-roformer for vocal separation and vocal melody transcription,” arXiv preprint arXiv:2409.04702, 2024.

[37] LRCLIB, “LRCLIB: A community-driven synchronized lyrics database,” 2024, accessed: 2025. [Online]. Available: https://lrclib.net

[38] Musixmatch S.p.A., “Musixmatch: The world’s largest lyrics platform,” 2010, accessed: 2025. [Online]. Available: https://www.musixmatch.com

[39] Genius Media Group, “Genius: Song lyrics and music knowledge,” 2009, accessed: 2025. [Online]. Available: https://genius.com

[40] R. H. Dunn and J. Duddington, “eSpeak NG: Text-tospeech synthesizer,” 2015, accessed: 2025. [Online]. Available: https://github.com/espeak-ng/espeak-ng

[41] K. Park and J. Cho, “g2pK2: Korean graphemeto-phoneme converter,” 2020, forked as g2pK2. Accessed: 2025. [Online]. Available: https://github. com/Kyubyong/g2pK

[42] D. R. Mortensen, S. Dalmia, and P. Littell, “Epitran: Precision G2P for many languages,” in Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), N. Calzolari, K. Choukri, C. Cieri, T. Declerck, S. Goggi, K. Hasida, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, S. Piperidis, and T. Tokunaga, Eds. Miyazaki, Japan: European Language Resources Association (ELRA), May 2018. [Online]. Available: https://aclanthology.org/L18-1429/

[43] H. Miura, “pykakasi: Japanese kana-kanji to romaji¯ transliterator,” 2014, based on KAKASI. Accessed: 2025. [Online]. Available: https://github.com/miurahr/ pykakasi

[44] mozillazg, “pypinyin: Chinese characters to pinyin converter,” 2014, accessed: 2025. [Online]. Available: https://github.com/mozillazg/python-pinyin

[45] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi et al., “Scaling speech technology to 1,000+ languages,” Journal of Machine Learning Research, vol. 25, no. 97, pp. 1–52, 2024.

[46] U. Hermjakob, J. May, and K. Knight, “Out-of-the-box universal romanization tool uroman,” in Proceedings of ACL 2018, system demonstrations, 2018, pp. 13–18.

[47] Y. Luo, R. Zhang, L.-C. Liu, T. Li, and H. Liu, “Fcpe: A fast context-based pitch estimation model,” arXiv preprint arXiv:2509.15140, 2025.

[48] H. Wei, X. Cao, T. Dan, and Y. Chen, “Rmvpe: A robust model for vocal pitch estimation in polyphonic music,” arXiv preprint arXiv:2306.15412, 2023.

[49] Z. Duan, H. Fang, B. Li, K. C. Sim, and Y. Wang, “The nus sung and spoken lyrics corpus: A quantitative comparison of singing and speech,” in 2013 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference. IEEE, 2013, pp. 1–9.

[50] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, “Montreal forced aligner: Trainable text-speech alignment using kaldi,” in Proc. Interspeech 2017, 2017, pp. 498–502.

[51] Y. Kang, “Voice Onset Time merger and development of tonal contrast in Seoul Korean stops: A corpus study,” Journal ofPhonetics, vol. 45, pp. 76–90, 2014.

[52] S. Duanmu, The phonology of standard Chinese. Oxford University Press, 2007.

[53] F. Li, “The phonetic development of voiceless sibilant fricatives in English, Japanese and Mandarin chinese,” Ph.D. dissertation, The Ohio State University, 2008.

[54] S.-I. Lee-Kim, “Revisiting Mandarin ‘apical vowels’: An articulatory and acoustic study,” Journal of the International Phonetic Association, vol. 44, no. 3, pp. 261–282, 2014.

[55] P. Ladefoged and P. Bhaskararao, “Non-quantal aspects of consonant production: A study of retroflex consonants,” Journal of phonetics, vol. 11, no. 3, pp. 291–302, 1983.

[56] M. Schellenberg, “Does language determine music in tone languages?” Ethnomusicology, vol. 56, no. 2, pp. 266–278, 2012.