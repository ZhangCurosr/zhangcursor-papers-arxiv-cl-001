# OPINIONATED, HESITANT AND STRESSED: THREE STUDIES OF HOW POLITICIANS SPEAK IN FOUR SLAVIC PARLIAMENTS

Ivan Porupski,<sup>1</sup> Nikola Ljubešić<sup>1,</sup> <sup>2,</sup> <sup>3</sup>

<sup>1</sup>Department of Knowledge Technologies, Jožef Stefan Institute, Ljubljana, Slovenia

<sup>2</sup>Institute of Contemporary History, Ljubljana, Slovenia

<sup>3</sup>Faculty of Computer and Information Science, University of Ljubljana, Slovenia

Keywords: parliamentary speech, sentiment and acoustics, filled pauses, primary stress, cross-Slavic, spoken corpora, ParlaSpeech

## 1 I N T R O D U C T I O N

Opinionated about what they say, hesitant about how they say it, stressed on where they put the accent. Parliamentary speech sits at an unusually rich intersection: it is planned but not scripted, public but personal, politically charged but procedurally constrained. Speakers deliver it under time pressure and adversarial attention, and what they say is inseparable from how they say it — the pitch of a rebuttal, the hesitation before a contested figure, the stress placed on a contested word. Until recently, studying these phenomena at scale across languages has been impractical: spoken corpora have been small, mostly monolingual, and designed for automatic speech recognition rather than for linguistic or social-scientific inquiry (Porupski, 2026). The situation is now changing, and with it the kinds of questions one can realistically ask.

This paper pursues three such questions, each addressed to a diferent facet of how politicians actually speak. First, does the emotional tone of what a politician says leave a measurable trace in how they say it — and does this trace look the same across languages? Second, what predicts when and how often speakers hesitate, and do the predictors pattern consistently across parliaments or fragment by language? Third, when a language admits multiple legitimate stress positions for the same word, is a speaker’s preference for early or late stress coherent across the lexicon, or does it fragment by part of speech?

Each question is approachable today because of ParlaSpeech (Ljubešić et al., 2025c), a multilingual collection of spoken parliamentary corpora covering Croatian (HR), Czech (CZ), Polish (PL), and Serbian (RS), totalling over 6,000 hours of speech aligned with oficial ParlaMint transcripts (Erjavec et al., 2025). Beyond transcript alignment and rich speaker metadata, ParlaSpeech 3.0 adds Universal Dependencies annotations, utterance-level sentiment predictions, transformer-based filled pause detection, and — for Croatian and Serbian — word- and grapheme-level alignments with primary stress markers. These layers make the three questions above not merely askable but systematically answerable: sentiment can be cross-referenced with acoustics, filled pauses with syntactic structure, stress patterns with speaker demographics and part of speech.

We present the current state of the three investigations in Sections 2–4 and then devote the remainder of the paper, Section 5, to the research directions these studies open up.

## 2 SENTIMENT AND ACOUSTICS

Does the emotional tone ofwhat a politician says leave a measurable trace in how they say it?

The intuition is old: angry speech sounds diferent from conciliatory speech, and listeners pick up on this long before they parse the words. Whether and how this intuition survives large-scale measurement across languages is a diferent question. Working with over one million utterances across all four parliaments, we examined how pitch (F0), intensity, and speech rate covary with utterance-level sentiment (Porupski & Ljubešić, 2026). Sentiment scores came from the XLM-R-ParlaSent model (Mochtak et al., 2023); acoustic features were extracted with Praat. We tested three competing hypotheses: a linear valence pattern (negative is louder/higher/faster, positive is quieter/lower/slower), a U-shaped arousal pattern (both extremes activated, neutral flattest), or a mixture of the two. The analysis combined Wilcoxon signed-rank tests on sentiment extremes, Kendall monotonic trend analysis, and an inflection-based split to detect arousal-related curvature.

Valence dominates. In 96% of feature–language combinations we found significant diferences between sentiment extremes and monotonic trends across the full range (11/12 and 12/12 tests, respectively). As sentiment moves from negative to positive, pitch and intensity decrease systematically; speech rate shows a similar tendency, with some language-specific variation. The picture is not purely linear, however. Figure 1 shows a characteristic J-shape: a sustained decrease from negative through neutral sentiment, followed by a clear upturn at the most positive end of the scale. The inflection point sits at sentiment value 3.5 (the neutral-positive boundary) and is fairly consistent across features and languages. Arousal-like curvature contributes secondary modulation in 42% of tests, most prominently for speech rate (63% support).

Figure 1: Global feature trend averaged across pitch, intensity, and speech rate, normalised and pooled across all four languages. The minimum at sentiment value 3.5 (neutral-positive) marks the inflection point where a valence-driven decrease transitions into an arousal-related upturn.  
![](images/0103d2b15df6942e29099ca609269ede1214f988d9f45c1671b5f39e6a8923a9.jpg)

Cross-linguistic diferences add nuance rather than override the pattern. Croatian relies almost exclusively on linear valence tracking; Polish exhibits the strongest arousal modulation, with U-shaped patterns in pitch and intensity; Czech and Serbian sit between, with their arousal signatures concentrated in speech rate. The overall finding is that negative speech is produced with higher pitch, greater intensity, and faster rate across all four parliaments, and that the partial upturn at the positive extreme is a genuine cross-linguistic regularity rather than an artefact of any single language.

## 3 FILLED PAUSE PRODUCTION

## What predicts when and how often speakers hesitate?

Filled pauses (“um”, “uh”, “eee”) sit awkwardly between disfluency and device. They can signal planning load, buy floor-holding time, or function rhetorically; they also appear to vary with age, gender, and communicative context in ways that the conversational-corpora literature has not fully reconciled. Parliamentary speech offers an unusually clean setting to probe their predictors: the domain is held constant, speaker metadata is rich, and the adversarial-yet-formal register concentrates the kind of under-pressure planning that elicits hesitation. We fitted a negative binomial

Figure 2: Incidence rate ratios from negative binomial GEE models with independent working correlation (baseline specification), for the pooled global model and each parliament. Error bars show 95% confidence intervals; red markers indicate non-significant efects (CI crosses 1). Reference: female coalition speaker of average age and centrist orientation.  
![](images/c7f86becb1b7a95698b527ac4dc8e75ec90be56391b94ccabbe7e698ee216182.jpg)

GEE (Generalised Estimating Equations) model with independent working correlation to predict FP frequency per utterance (with log-duration ofset), using speakerlevel clusters and robust sandwich standard errors (Porupski et al., 2026). Filled pauses were detected via a transformer-based wav2vec2-BERT classifier (Ljubešić et al., 2025b).

Several patterns emerge across the 1,001,787 utterances from 1,561 speakers in our analysis, summarised in Figure 2. Speech rate is the dominant predictor: faster speech yields substantially fewer FPs (−36% per SD globally), consistent across all four parliaments. Sentiment shows a small but stable positive association: more positive speech corresponds to more FPs (+6% per unit globally), stable across the full corpus. Age reduces FP frequency (−16% per decade globally), clearly significant in CZ and RS (∼−23% and −22% per decade) with the same direction but nonsignificant in HR and PL. Power status shows an opposition discount globally (−21%), significant in CZ (−22%) and HR (−33%) but non-significant in PL and borderline reversed in RS. Political orientation shows no robust global efect.

The most striking finding is cross-linguistic: gender efects reverse direction between language groups. In the South Slavic parliaments, male speakers produce substantially fewer FPs than female speakers (HR −47%, RS −60%); in the West Slavic parliaments (CZ, PL), there is no significant gender diference at all. This contradicts the typical “men hesitate more” pattern reported in conversational corpora and points to something language-, culture-, or institution-specific. A cross-linguistic design is essential for seeing it: any single-language study would either confirm or contradict the received wisdom, but only the comparison exposes the reversal as such.

Women remain a numerical minority among speakers in all four legislatures; the clustered, robust-SE specification reflects the resulting sample-size asymmetry in wider confidence intervals for the less-represented group, but the underlying imbalance is worth bearing in mind when interpreting the size ofthe gender efect rather than only its significance.

Several non-exclusive explanations could underlie this split: South Slavic parliamentary cultures may carry stronger gendered expectations around confident, disfluency-free speech, suppressing or penalising hesitation more for one gender than the other; floor-time allocation and seniority structures difer across these four legislatures in ways that could shape who speaks with more rehearsed fluency; and broader public-speaking norms outside parliament may compound the efect if they diverge between South and West Slavic public life. We cannot adjudicate between these accounts with the current data, and see this as a natural target for the politicalrhetoric programme sketched in Section 5.5.

## 4 PRIMARY STRESS IN CROATIAN

Is a speaker’s preference for early vs. late stress consistent across the lexicon, or does it fragment by part of speech?

Figure 3: MDS mapping into 2D of pairwise correlation coeficients between speaker-specific preferences for early vs. late stress across parts of speech.

![](images/14c7060e74716c379724c400acf8bf6bff4ce62cb3de8b10db7b361d90e19add.jpg)

Croatian has a well-documented accentual variation: the same multisyllabic word often admits two legitimate stress positions (e.g. náglašavam vs. naglašávam), and the distribution of early and late patterns varies across the lexicon. Descriptive work by Ivić (1994) treats Neo-Štokavian accentuation as structurally constrained but allows for paradigm-internal alternations, while Kapović (2015) foregrounds diachronic stratification and analogical levelling. More recent experimental work (Pletikos Olof & Bradfield, 2019) shows that variation is gradient, shaped by frequency, morphology, and register. What none of this prior work has been able to address is whether speaker-level preferences cohere across the lexicon: when a given speaker favours early stress in verbs, do they also favour it in nouns, adjectives, adverbs, and proper names?

Every multisyllabic word in the Croatian portion of ParlaSpeech was automatically annotated for primary stress using a fine-tuned speech transformer achieving 99% word-level accuracy (Ljubešić et al., 2025a). We focus on Croatian here because its stress-annotation pipeline was the first to be validated at scale; the same alignments exist for Serbian (Section 1), and extending the present analysis to that language is discussed in Section 5.4. From 11.3 million multisyllabic tokens (496 speakers), we retained words exhibiting stress variation — those where the secondary stress position accounted for at least 10% of occurrences — leaving 935k tokens from 491 speakers. Each speaker was represented by their proportion of early stress usage per part of speech.

Pairwise Pearson correlations across parts of speech reveal a tight central cluster of verbs, adjectives $( \boldsymbol { \mathrm { r } } = \boldsymbol { 0 } . 8 \boldsymbol { 8 } ^ { \star \star \star } )$ , and nouns $( \boldsymbol { \mathsf { r } } = 0 . 8 4 ^ { \star \star \star } )$ ). Proper nouns are only partially aligned with this cluster $( \mathsf { r } = 0 . 6 3 - 0 . 7 0 ^ { \star \star \star } )$ , and adverbs lie largely outside it $\ l ( \ r { r } = 0 . 2 3 -$ $0 . 2 7 ^ { \star \star \star } )$ . Adverbs and proper nouns are mutually near-independent $( \mathsf { r } = 0 . 0 4 , \mathsf { n } . 5 . )$ The MDS projection in Figure 3 summarises the picture: speaker-level stress preferences are coherent within an open-class core (verbs–adjectives–common nouns) but decouple for adverbs and proper nouns. This suggests that “a speaker’s stress system” is not a single setting but a structured bundle, and that adverbs in particular behave more like lexical idiosyncrasies than like instantiations of a general preference.

## 5 FUTURE WORK

The three studies above each open onto larger programmes, which we sketch along five axes: corpus expansion, corpus phonetics, disfluencies, primary stress variation, and political rhetoric.

## 5 . 1 C o r p u s Ex p a n s i o n a n d A d d i t i o n a l A n n o t a t i o n L a y e r s

Slovenian is the immediate next language addition: the pipeline extends naturally and the linguistic proximity to the existing languages makes it methodologically tractable. Bosnian, additional Serbian varieties, Bulgarian, and Ukrainian are strong further candidates, each bringing distinct typological or sociolinguistic leverage. On the annotation side, silent pauses are the obvious complement to filled pauses — inter-word silences can be extracted directly from the existing word-level alignments at essentially no additional cost — and the filled pause inventory should be further subclassified by phonetic type (schwa-like, eee-like, and so on). Topic labels from the ParlaCAP project would add a policy-domain axis, enabling everything from withinspeaker register comparisons to topic-controlled acoustic analysis.

## 5.2 Corpus Phonetics

The phonetic description of the four languages represented in ParlaSpeech rests heavily on classical lab-based studies with small speaker samples and read speech. Corpus linguistics transformed syntax and lexicography by making large naturalistic samples the primary empirical basis; a corpus-phonetics turn is now possible for these languages, and the precise grapheme- and phoneme-level alignments available for Croatian and Serbian make it immediately tractable. Vowel spaces can be remeasured at population scale by extracting F1/F2 from every vowel token and comparing the empirical space to the textbook five-vowel system, with conditioning on gender, age, and speech rate. The degree of vowel reduction in connected speech, often claimed to be limited in South Slavic relative to (say) Russian, becomes a measurable rather than an asserted property. Phoneme duration inventories are recoverable not just as means but as full distributions, allowing the phonological short/long vowel contrast to be assessed for how cleanly it separates in spontaneous speech.

More targeted comparisons follow. Voice onset time in plosives, classically described as prevoiced for voiced stops and short-lag for voiceless in Croatian and Serbian, can be verified at scale and checked for stability across speech rates and speaker demographics. Consonant cluster simplification in fast speech can be systematically measured, with the set of simplifying elements compared to predictions from the phonological literature. The Croatian–Serbian comparison is especially attractive: two closely related varieties processed through the same annotation pipeline mean that any measured diferences are genuinely linguistic rather than methodological. And across all of these measurements, conditioning on speech rate yields a hyper-/hypo-articulation profile — a characterisation of what gives first as speakers speed up: vowel quality, consonant duration, the voicing contrast, or some interaction.

## 5.3 Disfluencies

Filled pauses are only one dimension of the hesitation story, and the frequency analysis reported in Section 3 is only one angle on filled pauses themselves. A natural question the existing annotation layers put within reach is whether filled pauses cluster at particular positions in the syntactic tree — for instance between rather than within phrases — and whether any such pattern is consistent across parliaments. An information-theoretic complement asks whether surprisal rises following a filled pause: related work suggests it does, but the generality of the efect remains open. A further dimension, foreshadowed in the frequency analysis, is FP duration: frequency and duration are separable aspects of disfluency whose predictor profiles need not coincide — a cleavage we are pursuing in follow-up work. Extending the inventory further to repetitions and repairs alongside filled and silent pauses would complete the spoken-fluency picture.

Once silent pauses are extracted, the full pausing profile becomes tractable: speakerlevel trends in the ratio of filled to silent pauses, co-occurrence statistics, and whether the predictors identified for FP frequency transfer to silent pauses or pattern diferently. The timing information in the alignment layer also opens questions beyond disfluency proper — redefining multi-word expressions as sequences whose inter-word pauses are systematically shorter than expected given length and frequency, and reassessing contested syntactic boundaries via pause distribution across candidate parses. In a complementary direction, ongoing collaboration with Croatian psycholinguists examines word production duration as a function of concreteness and imageability ratings, with scope for extension to valence and arousal norms and to the other languages as such norms become available.

## 5 . 4 P r i m a r y S t r e s s Va r i a t i o n

The most immediate extension is to Serbian, for which the same word- and grapheme-level stress alignments already exist (Section 1); replicating the coherence analysis there would show whether the verbs–adjectives–nouns cluster and the adverb/proper-noun periphery reported for Croatian reflect a general Neo-Štokavian pattern or something specific to Croatian speakers.

The analysis in Section 4 treats parts of speech at the broadest grain; subcategory analysis is the obvious next step, asking whether the verbs–adjectives–nouns coherence holds uniformly across aspect, tense, voice, or declension class. A more ambitious direction is to classify speakers into Croatian’s two main systems — pitchaccent and stress — and use the classification to probe several hypotheses: that stress-system speakers deviate from their own system more often than pitch-system speakers deviate from theirs, that deviations are more lexically specific than contextdriven, and that temporal trends within sessions reveal accommodation to previous speakers. Overcorrection is a related target: stress-system speakers overgeneralising early stress onto words where early stress is non-standard, and — symmetrically — pitch-system speakers applying late stress where stress-system speakers do not.

## 5 . 5 P o l i t i c a l R h e t o r i c

Parliamentary speech is a political act, and the acoustic and disfluency signals examined above are also potentially rhetorical resources. Do speakers shift pitch, rate, and intensity when moving between policy domains — defence versus healthcare, for instance — independently of sentiment? Do the same politicians sound measurably diferent in government versus opposition, when the same individual can be tracked across sessions on either side of a transition? Does acoustic accommodation to the previous speaker show up within debate sessions, generalising the accommodation hypothesis from primary stress to the full acoustic profile? And can filled pauses be decomposed into planning-related and strategic-rhetorical subtypes using syntactic tree position and the sentiment trajectory oftheir surrounding context? Each ofthese is a research question in its own right, and together they point toward a treatment of parliamentary speech as a site for studying the rhetorical exploitation of paralinguistic resources.

Pursuing these questions well will benefit from expertise beyond corpus phonetics. Sociolinguists and historians are better placed than we are to characterise how the four parliaments’ distinct historical trajectories and institutional cultures shape norms of public speech, and procedural heterogeneity we have not yet modelled — most obviously the balance between scripted, pre-written delivery and extemporaneous response, which plausibly varies by parliament, party, and debate type — is a natural candidate for future conditioning variables once reliably annotated.

## 6 CONCLUSION

The three studies presented here use the same corpus and broadly the same methodological toolkit, but they address very diferent kinds of question: a cross-modal one about how sentiment is acoustically realised, a behavioural one about who hesitates and when, and a structural one about whether a speaker’s phonological preferences hold together across the lexicon. What unites the findings is that parliamentary speech, despite its formal register and genre constraints, is a rich site for observing how language, emotion, and politics interact, and that the patterns one sees are neither uniform across languages nor random with respect to them. Negative sentiment recruits a consistent acoustic profile — higher, louder, faster — across all four parliaments, with a cross-linguistically stable upturn at the most positive end of the scale. The predictors of hesitation rate pattern broadly similarly across languages except along the gender axis, where South and West Slavic parliaments diverge sharply. Croatian speakers’ stress preferences cohere within a verbal core but fragment at the periphery, complicating any account that treats “speaker system” as unitary.

Taken together, these results argue for parliamentary speech as a productive domain for comparative research on the intersection of language, afect, and political behaviour, and they argue more specifically for cross-linguistic designs in which language-level variation is part of the signal rather than something to be controlled away. The forward programme sketched in Section 5 suggests that the findings reported here are preliminary in a useful sense: each answers its question while opening several new ones that the same corpus, suitably extended, can be used to pursue. Building further large, open, comparable, richly annotated spoken datasets — across additional languages, annotation layers, and parliamentary contexts — is the natural way to turn these openings into a sustained research programme.

## ACKNOWLEDGEMENTS

This work was supported in part by the project “Large Language Models for Digital Humanities” (Grant GC-0002), the project “EPIC-SI - Early Parent-Child Communication in Slovenian: Corpus-based Insights” (J6-70222), the research programme “Language Resources and Technologies for Slovene” (Grant P6-0411), and the Research Infrastructure DARIAH-SI (I0-E007), all funded by the ARIS Slovenian Research and Innovation Agency.

## REFERENCES

Erjavec, T., Kopp, M., Ljubešić, N., Kuzman, T., Rayson, P., Osenova, P., Ogrodniczuk, M., Çöltekin, Ç., Koržinek, D., Meden, K., Skubic, J., Rupnik, P., Agnoloni, T., Aires, J., Barkarson, S., Bartolini, R., Bel, N., Calzada Pérez, M., Dar <sup>‘</sup>gis, R., . . . Fišer, D. (2025). ParlaMint II: Advancing comparable parliamentary corpora across Europe. Language Resources and Evaluation, 59(3), 2071–2102. https://doi.org/10.1007/s10579-024-09798-w

Ivić, P. (1994). Srpskohrvatski dijalekti i njihova struktura. Izdavačka knjižarnica Zorana Stojanović.

Kapović, M. (2015). Povijest hrvatske akcentuacije: Fonetika. Matica hrvatska.

Ljubešić, N., Porupski, I., & Rupnik, P. (2025a). Identifying Primary Stress Across Related Languages and Dialects with Transformer-based Speech Encoder Models. Interspeech 2025, 5768–5772. https://doi.org/10.21437/Interspeech.2025-205

Ljubešić, N., Porupski, I., & Rupnik, P. (2025b, July). Identifying filled pauses in speech across South and West Slavic languages. In J. Piskorski, P. Přibáň, P. Nakov, R. Yangarber, & M. Marcinczuk (Eds.), Proceedings of the 10th workshop on Slavic natural language processing (Slavic NLP 2025) (pp. 1–8). Association for Computational Linguistics. https: //doi.org/10.18653/v1/2025.bsnlp-1.1

Ljubešić, N., Rupnik, P., Porupski, I., & Pungeršek, T. K. (2025c). The ParlaSpeech v3 Collection of Spoken Parliamentary Corpora from the Croatian, Czech, Polish and Serbian Parliament Enriched with Linguistic and Paralinguistic Annotation Layers [Dataset]. CLARIN Annual Conference Proceedings, 137. http://hdl.handle.net/11356/1833

Mochtak, M., Rupnik, P., & Ljubešić, N. (2023). The ParlaSent multilingual training dataset for sentiment identification in parliamentary proceedings. arXiv preprint. http://arxiv. org/abs/2309.09783

Pletikos Olof, E., & Bradfield, J. (2019). Croatian pitch-accents: Fact and fiction. Proceedings of the 19th International Congress ofPhonetic Sciences, Melbourne, Australia 2019, 855– 858.

Porupski, I. (2026). Open-source spoken and speech corpora: A survey with European focus and Slavic emphasis [Seminar paper, Jožef Stefan International Postgraduate School, Ljubljana].

Porupski, I., Dropuljić, B., & Ljubešić, N. (2026). Umm... With Transformers? Insights from Filled Pause Use across Four Slavic Parliaments [Accepted to Interspeech 2026].

Porupski, I., & Ljubešić, N. (2026). From Vocal Cues to Political Views: Acoustic Patterns across Sentiment in South and West Slavic Parliamentary Speech [Manuscript under review].