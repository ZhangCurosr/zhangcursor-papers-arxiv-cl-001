# A Corpus-Aligned Uthmani-to-Standard Quranic Word Mapping and a Deterministic Recitation Validator

Yahya Mohamed Elnawasany

Independent Researcher, Egypt

yahyaalnwsany39@gmail.com

## Abstract

Quranic text is distributed in two orthographic forms that are byte-level distinct: the Uthmani script used in every printed mushaf, and the Standard (Imla’i) Arabic form that every mainstream Arabic NLP tool is built for. The gap is concentrated in one Unicode character, U+0670 (superscript alef), which appears in some of the most frequently recited words in the Quran and is silently mishandled by general-purpose Arabic normalizers. We release a 2,290-pair, corpus-aligned Uthmani-to-Standard word mapping constructed by aligning the complete 6,236-verse Quran across both orthographic forms, together with a seven-step text normalization pipeline built on it. Normalizing both forms of all 6,236 verses through that pipeline yields identical strings for 90.9% of verses, and we characterize the residual divergence rather than assert that it is closed. On top of the normalized text, we build a deterministic, LLM-free Quranic recitation validator using a four-layer verse-matching search (exact, morphological, relaxed, fuzzy) and word-error-rate-graded feedback across five severity tiers. The validator scores 98.4% (122/124) on a 124-case suite emitted by the released test harness, and both failures share one mechanism: a single substitution error can make a diferent verse an exact match. A full-corpus census additionally quantifies an inherent text-only ambiguity afecting 16.5% of verses, and on 34 recitation transcripts drawn from a deployed Arabic ASR system the validator identifies the correct verse in every case. We release the mapping, the script that builds it, the validator, and the evaluation harness under open licenses; every number in this paper except the deployment measurement, whose transcripts are not ours to publish, is reproduced by running them. Keywords: Quranic Arabic, Arabic NLP resources, text normalization, recitation validation, Uthmani script, language resources

## 1. Introduction

Quranic recitation validation – determining whether spoken text matches the canonical Quran and, if not, where – requires comparing automatic-speechrecognition (ASR) output against an indexed reference corpus. This is harder than it appears for a specific, underreported reason: the Quran exists in two orthographic forms that are byte-level distinct even when phonetically identical. The Uthmani script, standardized in the era of the third Caliph and used in every printed mushaf and in authoritative digital corpora such as Tanzil (Tanzil Project, 2024), includes characters absent from Standard (Imla’i) Arabic – the form every mainstream Arabic NLP tool, ASR model, and text corpus is built around. The single most consequential of these characters is U+0670, the Arabic Superscript Alef (alef khanjariyya), which Uthmani writes as a diacritic above the preceding letter. What Standard writes in its place is not fixed, and that is the dificulty: for most of these words it writes a full alef, so that Uthmani al-kafirin gains one; but for a sizeable minority it writes nothing at all, and alrahman – among the most frequently recited words in the Quran – is spelled without an alef in both forms. A single global rule is therefore wrong in one direction or the other whichever way it is written, which is what motivates a word-level resource rather than another character-level heuristic. To our knowledge, no published word-level resource maps between the two forms, and the normalization routines shipped with mainstream Arabic NLP toolkits – Farasa (Abdelali et al., 2016), CAMeL Tools (Obeid et al., 2020) and PyArabic (Zerrouki, 2023) – treat U+0670 as an ordinary diacritic: they either strip it silently, losing the long vowel it represents, or leave it untreated, in both cases producing byte-level mismatches between Uthmani-sourced text and Standard-form indexes or ASR output.<sup>1</sup>

This paper makes two contributions. First, a 2,290-pair, corpus-aligned Uthmani-to-Standard word mapping (§3), constructed by aligning the complete Quran across both orthographic forms – to our knowledge the first published resource of its kind. Second, a deterministic recitation validator built on top of it (§4), which normalizes ASR output through a seven-step pipeline anchored by the mapping and identifies the recited verse via a four-layer search, scoring 98.4% on a 124-case evaluation suite (§5) with no language model involved anywhere in the pipeline. Both the dataset and the validator’s code are released publicly (§6).

## 2. Related Work

Digital Quranic text resources are well established: the Tanzil project (Tanzil Project, 2024) maintains the canonical Uthmani and Simple-Arabic encod ings used as ground truth throughout this work, and the Quranic Arabic Corpus (Dukes, 2011) pro vides morphological annotation (root, lemma, part of speech) for every word, which our validator’s linguistic-matching layer (§4) draws on. Neither resource, however, provides an explicit word-leve alignment between the Uthmani and Standard or thographic forms – the specific gap this paper ad dresses. The problem of mapping a variant written form onto a single conventional one is itself wel established in Arabic NLP: CODA (Habash et al., 2012) defines a conventional orthography for dialec tal Arabic precisely so that dialectal text becomes tractable for tools built for Modern Standard Arabic. Our task has the same shape but a diferent char acter. CODA is prescriptive – it stipulates a conven tion for varieties that have none, and is defined over undiacritized text – whereas the Uthmani and Stan dard forms are both already standardized and fully diacritized, so the correspondence between them is descriptive and recoverable by alignment over parallel corpora rather than stipulated by guideline. General Arabic NLP surveys (Habash, 2010; Dar wish and Magdy, 2014) document root-based mor phology and diacritization as central Arabic NLP challenges, and diacritization-restoration systems (Zerrouki and Balla, 2019) address the general ab sence of diacritics in Modern Standard Arabic text; the Uthmani/Standard divide is a distinct problem, since Quranic text is already fully diacritized in both forms and the mismatch is orthographic rather than a missing-information problem. Prior work on com putational recitation assessment (Alqahtani et al. 2019) surveys two dominant approaches – forced phoneme alignment and end-to-end acoustic clas sification – both of which require a Tajweed-aware acoustic model or a large labeled-error audio cor pus, neither of which exists at production quality for Uthmani-script Quranic recitation. The closest adjacent efort is Tarteel (Khan et al., 2021), which releases a large crowd-sourced corpus of Quranic recitation audio paired with its text and underpins an open recitation-ASR system; it addresses the complementary half of the problem, producing the transcript that a text-side validator must then recon cile against Uthmani-encoded reference text. We take a third approach: operating on ASR text out put after normalization, which trades acoustic-leve

Tajweed detection for a fully deterministic, trainingdata-free, immediately deployable pipeline.

## 3. The Uthmani-to-Standard Mapping

## 3.1. Construction

The mapping is built by tools/build\_map.py, released with the dataset, in four passes over the complete 6,236-verse Quran in its Uthmani and Simple-Arabic forms (Tanzil Project, 2024). Every count below is printed by that script.

(1) Direct alignment: for the 5,813 verses whose two forms have identical word counts, each Uthmani word is aligned to its Standard counterpart positionally; only words containing U+0670 are retained as keys, since words without it need no entry. (2) Ornamental-mark handling: the Uthmani ornamental mark U+06DE (rub el-hizb) occurs in 199 verses, has no Standard counterpart, and is stripped before alignment. (3) Structuralmismatch alignment: in the remaining 423 verses difflib.SequenceMatcher (Ratclif and Metzener, 1988) finds the longest common word subsequence; runs of equal length are aligned positionally, and runs of unequal length are resolved by a dynamic program that aligns each Uthmani word to a span of one to three consecutive Standard words by character similarity. This last step is what recovers the one-to-many splits, which are the structurally interesting case: Uthmani writes the vocative particle joined to its noun as one word (yaqawmi), where Standard separates them (ya qawmi). 58 entries are one-to-many, and they are derived by the same alignment as every other entry rather than written by hand. (4) Majority-vote conflict resolution: exactly two keys occur with more than one Standard form across the corpus, and in both the majority is unambiguous, so the most frequent form is kept; no key ends in a tie, and no entry required manual curation.

## 3.2. Statistics

The resulting dataset contains 2,290 unique Uthmani-Standard word pairs, harvested from 8,881 aligned occurrences of U+0670. The corpus contains 2,291 distinct Uthmani word forms carrying U+0670, so coverage is 2,290 of 2,291: one form, appearing in a single verse whose two readings admit no consistent span alignment, is deliberately left out rather than guessed. Words without U+0670 require no mapping and are handled by the character-level rules in §4.1. To our knowledge, this is the first published word-level Uthmani-Standard mapping; the closest prior resources either operate exclusively on Standard text or on raw Uthmani text without cross-form validation.

<table><tr><td>Step</td><td>Effect on bismillah ir-rahman ir- rahim</td></tr><tr><td>1. NFC 2. Word-</td><td>No change (already composed) al-rahman: U+0670 resolved via</td></tr><tr><td>map lookup 3. Tashkeel removal</td><td>the 2,290-pair map Harakat stripped; mapped U+0670 untouched</td></tr><tr><td>4. Contex- tual U+0670 5.</td><td>No unmapped U+0670 remains No change needed</td></tr><tr><td>Alef/hamza unify</td><td></td></tr><tr><td>6. Word- initial rules 7. Non-</td><td>No change needed Clean Standard-form output,</td></tr></table>

Table 1: Worked example: seven-step normalization of a verse opening containing U+0670.

Table 1 traces a representative case – the opening of Surah Al-Fatiha, whose word al-rahman carries U+0670 – through the full seven-step pipeline (§4.1), showing where the mapping (step 2) resolves the character that a generic normalizer would otherwise strip or mishandle.

## 4. The Recitation Validator

## 4.1. Seven-Step Normalization Pipeline

Both the ASR-transcribed recitation and every reference verse are normalized identically before comparison, so that a phonetically correct recitation matches regardless of which orthographic form it was written in. How far that holds is measurable, and we measure it rather than assert it: normalizing the Uthmani and the Standard form of all 6,236 verses and comparing the results, 5,669 verses (90.9%) reduce to identical strings. The residual 567 verses are analysed in §8; they are not U+0670 failures but other Uthmani-Standard divergences the character rules do not yet cover. The steps are: (1) Unicode NFC canonicalization; (2) Uthmani-to-Standard word-map lookup (§3) for every token containing U+0670; (3) tashkeel and Quranic-mark removal (U+064B–U+065F, U+06D6–U+06FC, and the Arabic Extended-A marks U+08D3–U+08FF, which occur between letters in Uthmani text and so must be cleared before any rule that inspects adjacent letters), leaving any remaining unmapped U+0670 untouched; (4) contextual resolution of remaining U+0670 – replaced with alef (U+0627) except after ya-maqsura, dhal, ha, or lam, where it is a modifier diacritic and is removed; (5) alef/hamza unification, mapping all hamza-bearing alef variants (U+0623, U+0625, U+0622, U+0671) to bare alef and connected hamza forms to bare hamza; (6) alef-madda resolution, deleting a hamza written immediately before an alef, since Uthmani spells alef madda as hamza plus alef wherever it occurs and not only word-initially, together with yamaqsura folding; (7) non-Arabic character removal and whitespace collapse. The released implementation performs these as eleven operations; the grouping into seven above is expository, and the code names each step against this list.

## 4.2. Four-Layer Search

Given normalized query text, the validator searches all 6,236 verses via four progressively permissive layers, run only as far as needed: (1) exact token AND-matching over a precomputed per-verse word set; (2) morphological expansion of unmatched layer-1 tokens to their lemma and Arabic root via the Quranic Arabic Corpus’s (Dukes, 2011) annotation, robust to single-word ASR substitution errors; (3) relaxed coverage matching, accepting a verse when ≥60% of normalized query tokens are present, for inputs of at least three tokens; (4) fuzzy characterlevel matching for severely degraded input, scoring each verse by the best Ratclif/Obershelp similarity (Ratclif and Metzener, 1988) between the query and any same-length window of the verse, accepted at 0.45. Layer 4 uses the standard library’s difflib.SequenceMatcher rather than a faster third-party matcher deliberately: an optional dependency makes the validator’s output depend on which version of that library happens to be installed, which would defeat the determinism the rest of the design is built for. Candidates are scored by a coverage-weighted formula that favors short verses where the query fills a large fraction of the verse text, and inputs exceeding eight normalized words trigger a separate multi-verse alignment mode that segments and scores each constituent verse independently.

The three constants – 60% coverage, 0.45 similarity, eight words – were set by inspection during development and never tuned against the evaluation suite. Because that is an assertion a reader cannot check, we instead report how much they matter (Table 3). Suite accuracy is flat for any multi-verse trigger between four and eight words and degrades above it; it is unchanged for every relaxed-coverage threshold from 0.40 to 0.80; and it is unchanged for fuzzy thresholds from 0.25 to 0.55, because layer 4 is reached only when the three layers above it return nothing at all.

One entry in that table deserves saying out loud. A fuzzy threshold of 0.65 scores 123/124 rather than 122/124, because it happens to starve the wrong candidate in case ME03. We have not adopted it. The only evidence for 0.65 over 0.45 is the suite it would then be scored on, and a threshold chosen that way measures nothing except its own selection. A disabled layer 4 produces the same 99.2% by accident, which is the same mistake reached without even the excuse of a search.

<table><tr><td>Category</td><td>N</td><td>Pass</td><td>Acc.</td></tr><tr><td>Normalization</td><td>11</td><td>11</td><td>100%</td></tr><tr><td>Corpus access</td><td>7</td><td>7</td><td>100%</td></tr><tr><td>Single verse, perfect</td><td>10</td><td>10</td><td>100%</td></tr><tr><td>Single verse, substitution</td><td>4</td><td>3</td><td>75%</td></tr><tr><td>Single verse, deletion</td><td>3</td><td>3</td><td>100%</td></tr><tr><td>Single verse, tashkeel</td><td>6</td><td>6</td><td>100%</td></tr><tr><td>Multi-verse, full surah</td><td>7</td><td>7</td><td>100%</td></tr><tr><td>Multi-verse, with errors</td><td>3</td><td>2</td><td>67%</td></tr><tr><td>Multi-verse, consecutive</td><td>3</td><td>3</td><td>100%</td></tr><tr><td>Multi-verse, full page</td><td>3</td><td>3</td><td>100%</td></tr><tr><td>Edge cases</td><td>5</td><td>5</td><td>100%</td></tr><tr><td>Generated from corpus</td><td>62</td><td>62</td><td>100%</td></tr><tr><td>Total</td><td>124</td><td>122</td><td>98.4%</td></tr></table>

Table 2: Validator accuracy by test category, as emitted by the released harness.

## 4.3. WER-Graded Feedback

Once a verse is identified, a word-level dif (via the same sequence-matching approach) yields a Word Error Rate, mapped to five feedback tiers: WER = 0 (perfect), ≤ 0.10 (minor errors, encouraging), ≤ 0.30 (specific corrections listed), ≤ 0.60 (verse flagged for review), > 0.60 (verse likely misidentified; no answer given). No language model participates in matching, scoring, or feedback generation – every step is deterministic and auditable, which matters for a tool whose output is communicated to users as authoritative correction on a religious text.

## 5. Evaluation

## 5.1. Accuracy

The validator was evaluated on the 124-case suite in the released repository, which scores 98.4% (122/124). Table 2 is written by the harness itself on every run, so the grouping reported here is the grouping the code actually uses rather than a taxonomy maintained alongside it.

Half the suite (62 cases) is generated by a released script from the corpus, which injects substitutions, deletions, insertions and tashkeel variation into verses, whole short surahs, and page-length spans under a fixed random seed. The other 62 are hand-written assertions over the normalizer, the corpus accessor, and the multi-verse aligner. Both failures fall in the hand-written half.

## 5.2. Failure Analysis

Both failures are the same failure, and it is a property of the search design rather than of the data it runs on.

Case SS03 presents the isolated word wahid (“one”) and expects 112:1 (Qul huwa llahu ahad); the validator returns 6:19. The reason is not morphological expansion: 6:19 literally contains the word wahid, so it is an exact match at layer 1, while 112:1 – which has ahad, not wahid – is not. A single substitution error turned a diferent verse into the exact match.

Case ME03 shows why this is not a curiosity of one-word inputs. It recites all four verses of Surah Al-Ikhlas, fifteen words, with the same substitution in the opening verse. Multi-verse alignment selects its starting verse by searching the opening tokens; those tokens – qul huwa llahu wahid – are all four present in 6:19 and so it anchors there, fails to align the rest of the span, falls back to single-verse mode, and reports the whole recitation as verse 112:4 with ten insertions. The output is confidently and comprehensively wrong.

The general statement is that layer 1 has no notion of how surprising a match is: a verse containing every query token is accepted regardless of how much of that verse the query covers or whether a near-miss elsewhere would be likelier. Fixing it means scoring candidates against the alternatives rather than accepting the first exact hit, which we leave to future work rather than patching against the two cases that expose it here.

## 5.3. Verse-Opening Ambiguity Census

A full census of the Quran’s 6,236 verses finds that 1,030 verses (16.5%) share an identical normalized four-word opening with at least one other verse, forming 369 ambiguous groups; the most frequent shared opening, ya ayyuha alladhina amanu (“O you who believe”), opens 88 distinct verses across 20 surahs. This quantifies an inherent limitation of text-only validation – resolving it requires either continuation past the ambiguous opening or conversational context establishing which surah is being recited – and does not afect the remaining 83.5% of verses, which are unambiguous from a four-word opening alone.

## 5.4. Real ASR Output

The suite above is synthetic: its “noisy” inputs are perturbations we injected. Since the motivating application is a deployed Arabic voice system, we also ran the validator over transcripts it produced in ordinary use – 1,407 user utterances transcribed by a diacritizing Arabic ASR model, of which 751 are four words or longer.

<table><tr><td>Parameter</td><td>Value</td><td>Suite</td></tr><tr><td>Multi-verse trigger</td><td>4 words 6 words</td><td>98.4% 98.4%</td></tr><tr><td></td><td>8 words 12 words</td><td>98.4% 96.8%</td></tr><tr><td></td><td>20 words</td><td>88.7%</td></tr><tr><td>Relaxed coverage</td><td>0.40-0.80</td><td>98.4%</td></tr><tr><td>Fuzzy similarity</td><td>0.25-0.55 0.65</td><td>98.4%</td></tr></table>

Table 3: Threshold sensitivity, one parameter varied at a time. <sup>†</sup>This setting scores higher on this suite and is deliberately not adopted (§4).
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Uthmani-Standard pairs</td><td>2,290</td></tr><tr><td>U+0670 form coverage Cross-form round-trip</td><td>2,290 / 2,291 5,669 / 6,236 (90.9%)</td></tr><tr><td>Validator test cases</td><td>124</td></tr><tr><td>Validator accuracy</td><td>98.4% (122/124)</td></tr><tr><td>Real ASR transcripts</td><td>34 / 34</td></tr><tr><td>Ambiguous verse groups</td><td>369</td></tr><tr><td>Ambiguous verses</td><td>1,030 / 6,236 (16.5%)</td></tr></table>

Table 4: Summary statistics for both released resources.

Ground truth has to come from somewhere other than the system under test, so we label only those transcripts whose normalized form is an exact contiguous span of exactly one verse – a criterion both stricter than and independent of the validator’s fourlayer search. 34 transcripts qualify, and the validator returns the correct verse for 34 of 34.

That number should be read for what it is. Labelling by exact containment selects transcripts the ASR happened to get right, so this measures the clean end of real input and not the degraded end the fuzzy layer exists for. The honest summary is that real recitation transcripts, when they are transcribed correctly, are identified correctly, and that we do not yet have labelled ground truth for the ones that are not. A further observation is simply how rare recitation is in this trafic: of the 957 utterances of three words or more, 93 (10%) overlap a verse by 80% or more of their tokens, the rest being spoken questions.

## 6. Data and Code Availability

The 2,290-pair Uthmani-to-Standard mapping, the script that builds it from the corpus, the validator, and the evaluation harness are released publicly at https://github.com/NightPrinceY/ muslim-quran-validator (data: CC-BY 4.0; code: MIT). The repository depends on nothing outside the Python standard library, and running it reproduces the construction counts of §3, the coverage and round-trip figures of §4.1, the ambiguity census, Table 2, Table 3 and the 98.4% total. Two figures it cannot reproduce: the toolkit comparison in §1, which additionally requires PyArabic and CAMeL Tools at the versions named there, and the deployment measurement in §5.4, whose transcripts are not ours to publish. The evaluation suite (§5) and the reference Quran text derive from the Tanzil corpus (Tanzil Project, 2024) and the Quranic Arabic Corpus (Dukes, 2011), both of which are themselves openly licensed for research use.

## 7. Ethical Considerations

The released mapping and validator operate exclusively on canonical, publicly available Quranic text, and nothing derived from a user is released with them. The deployment measurement in §5.4 is the one part of this work that touches user data: the transcripts were captured by the deployed system under its published privacy policy, which ofers a recording opt-out that is honoured per participant at the point the participant joins. We report aggregate counts only, release no transcripts, and the recordings are not part of either published artifact. The validator’s design choice to decline an answer at high WER (§4) rather than guess is deliberate: an incorrect authoritative-sounding correction on a religious text is a more serious failure mode than a declined answer, and the five-tier design exists specifically to avoid presenting low-confidence output with unwarranted certainty.

## 8. Limitations

The validator operates on ASR text output, not raw audio, so acoustic Tajweed properties that do not change the transcribed phoneme sequence – incorrect elongation (madd) duration, nasalization (ghunna) quality, or assimilation (idgham) – are undetectable by construction; a student who recites every word correctly but violates these rules receives a perfect score. The verse-opening ambiguity documented in §5 is an inherent property of the text, not a resolvable engineering gap, without adding audio-level or conversational context the current text-only design does not use. The evaluation suite is our own: we wrote the 124 cases, chose their grouping, and set the search thresholds, and there is no held-out split separating development from evaluation. It was built to cover the normalization edge cases we knew of, so it measures coverage of anticipated failure modes rather than performance on an independent sample, and the figure should be read with that construction in mind – the two failures it does expose were both found by cases written to test something else. The crossform round trip is 90.9%, not 100%: for 567 verses the two forms still normalize to diferent strings. The residue is itemized rather than waved at – 628 divergent word pairs across 263 distinct types, of which the ten most frequent account for 30%. They are systematic classes the character rules do not yet cover: the definite article’s assimilated lam written once in Uthmani and twice in Standard (al-layl), words Uthmani spells with a letter fewer (Dawud, alnabiyyin), and medial hamza that Standard writes as alef (yas’alunaka). Each is tractable and none is addressed here. The deployment figure in §5.4 carries the selection bias described there, and 34 cases is a small sample. The Uthmani-Standard mapping is specific to the Hafs an Asim narration used in the reference corpora; other canonical narrations (e.g. Warsh an Nafi) would require a separate alignment pass, which we identify as future work.

## 9. Conclusion

We release a corpus-aligned Uthmani-to-Standard Quranic word mapping – to our knowledge the first published resource addressing this Arabic-NLP encoding gap – together with a deterministic, LLM-free recitation validator built on it that scores 98.4% on the released evaluation suite. Both resources are released openly, together with the scripts that build and evaluate them, and we hope the mapping in particular is useful to any downstream Arabic NLP task that must interoperate between Uthmani-encoded Quranic sources and Standard-Arabic-oriented tooling.

## 10. Acknowledgements

The author thanks Prof. Marwa Seddiq for supervision and guidance throughout this project.

## 11. Bibliographical References

Ahmed Abdelali, Kareem Darwish, Nadir Durrani, and Hamdy Mubarak. 2016. Farasa: A fast and furious segmenter for Arabic. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations, pages 11–16, San Diego, California. Association for Computational Linguistics.

Muteb Alqahtani, Laiali Almazaydeh, and Mahmoud Al-Rousan. 2019. A survey on quran recitation recognition and its applications. International

Journal of Advanced Computer Science and Applications, 10(5).

Kareem Darwish and Walid Magdy. 2014. Arabic information retrieval. Foundations and Trends in Information Retrieval, 7(4):239–342.

Nizar Habash. 2010. Introduction to arabic natural language processing. Synthesis Lectures on Human Language Technologies, 3(1):1–187.

Nizar Habash, Mona Diab, and Owen Rambow. 2012. Conventional orthography for dialectal Arabic. In Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC’12), pages 711–718, Istanbul, Turkey. European Language Resources Association.

Hamzah I. Khan, Anas Abou Allaban, and Mohamed Moussa. 2021. The Tarteel dataset: Crowd-sourced and labeled Quranic recitation. In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS).

Ossama Obeid, Nasser Zalmout, Salam Khalifa, Dima Taji, Mai Oudah, Bashar Alhafni, Go Inoue, Fadhl Eryani, Alexander Erdmann, and Nizar Habash. 2020. CAMeL tools: An open source python toolkit for Arabic natural language processing. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 7022–7032, Marseille, France. European Language Resources Association.

John W Ratclif and David E Metzener. 1988. Pattern-matching: The Gestalt approach. Dr. Dobb’s Journal, 13(7):46–51.

Taha Zerrouki. 2023. PyArabic: A python package for Arabic text. Journal of Open Source Software, 8(84):4886.

Taha Zerrouki and Amar Balla. 2019. Tashkeel: Novel open source arabic text diacritization. In Proceedings of the International Arab Conference on Information Technology.

## 12. Language Resource References

Kais Dukes. 2011. The quranic arabic corpus. https://corpus.quran.com.

Tanzil Project. 2024. Tanzil: Quran text project. https://tanzil.net.