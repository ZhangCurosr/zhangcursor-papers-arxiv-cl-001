# Typological Feature Induction as a Benchmark for Inductive Metalinguistic Reasoning

Chen Zhang<sup>1</sup>\* Seth Aycock<sup>2</sup>\* Yusuke Miyao<sup>1,3</sup> <sup>1</sup>NII LLMC <sup>2</sup>University of Amsterdam <sup>3</sup>The University of Tokyo zhangchen@nii.ac.jp s.aycock@uva.nl yusuke@is.s.u-tokyo.ac.jp

## Abstract

Existing metalinguistic reasoning benchmarks require both inference of rules and their application, making failures hard to diagnose. We introduce typological feature induction (TFI) to target the former step. Given sentences in an unnamed language paired with translations in a known pivot language, models must predict a typological feature of the unnamed language. From Grambank and multi-parallel corpora, we automatically build evaluation questions over 195 languages. We find that LLMs identify 59– 80% of these languages from only 16 sentence pairs; and misaligning the translations barely affects their accuracy, suggesting reliance on prior knowledge of the language. Enciphering text to prevent recognition increases the corpus utility, but mainly for features where target and pivot languages agree. On languages which models cannot identify, neither open model outperforms a majority-value baseline. Since induction from raw text aligns with field linguists use-case, this work indicates the models tested are not yet ready for this role. TFI offers a scalable path to track progress in this direction<sup>1</sup>.

## 1 Introduction

Metalinguistics is the study of language phenomena using a metalanguage (Schneider and Anastasopoulos, 2026), either as natural language about language, or symbolic notation encoding language information. While central to field linguistics, language acquisition and language teaching, questions regarding metalinguistic abilities of NLP models remain underexplored. Of particular interest is metalinguistic reasoning: the task of applying reasoning methods to solve problems about language.

Recent metalinguistic reasoning work evaluates LLMs on tasks such as rosetta stone-style linguistic Olympiad problems (Sánchez et al., 2026; Bean et al., 2024), translation from a grammar book (Aycock et al., 2025; Tanzer et al., 2024), or interlinear gloss prediction from grammar snippets (Yang et al., 2025). Such tasks remain difficult for frontier models (Choudhary et al., 2025; Marmonier et al., 2025). We posit that these tasks’ difficulty arises partly from conflating different aspects of metalinguistic reasoning: inductive reasoning (hypothesising rules of the linguistic system), and analogical reasoning (applying the system to new data).

![](images/5e1bc0de78b90be10908b65e4399b665c86d1184f0b740ffcd0f84363c06786d.jpg)  
Figure 1: A TFI item for Hausa, with an example sentence pair, and reasoning excerpts from Qwen3.6. In PARALLEL, the model names the language and answers correctly by recall. With the target side enciphered, it aligns three months word by word with watanni uku (‘months three’) and answers with the English order.

In light of this division, we propose typological feature induction (TFI), a task that isolates LLMs inductive metalinguistic reasoning based on raw data, while also probing understanding of typology, a fundamental facet of linguistic analysis (Ploeger et al., 2026). A model is shown n sentence pairs between a known pivot language and an unnamed target language, and is asked for the value of one typological feature of the target, such as the order of numeral and noun (Figure 1). Questions and gold values come from Grambank (Skirgård et al., 2023) and the sentences from multi-parallel corpora, so items are built automatically for any language covered by both, and each answer is a single property that can be checked against expert coding.

TFI also exposes the central difficulty of evaluating induction in LLMs. These models have seen many of the world’s languages, and withholding the name does not hide the language: from 16 sentence pairs, the three models we test name 59–80% of our 195 target languages. A correct answer may therefore be recalled rather than induced. We pursue three research questions:

RQ1 To what extent can LLMs infer typological features from a parallel corpus? (§4.1)

RQ2 How much of this reflects language recall rather than induction from the corpus? (§4.2)

RQ3 When recall is removed, what do the models take from the corpus? (§4.3–§4.5)

To answer them, we vary what the corpus contains: we remove the translations or break the sentence alignment, encipher the target side so that the language cannot be recognised, and replace English with 14 other pivot languages.

In plaintext, a corpus raises accuracy, but the gain saturates within 16 pairs, and removing the translations, breaking the alignment or adding the language name barely changes it (§4.1). The models name the language in their reasoning and answer from knowledge of it (§4.2). Once the cipher makes the language unrecognisable, more pairs, the translations and their alignment all help (§4.3), but this help comes from the pivot: it raises accuracy only where the target shares the pivot’s value, and pulls answers towards that value (§4.4). Finally, on languages none of the models can identify, neither open model beats a majority baseline across Grambank’s features (§4.5).

Our contributions are as follows:

• We introduce TFI, which isolates inductive metalinguistic reasoning from rule application, with a benchmark generated from Grambank covering 195 languages.

• Through controlled corpus ablations, pivot swaps, and ciphering, we show that corpus gains are largely attributable to recall.

• We identify two shortcuts that look like induction: recognising the language and recalling its grammar, and projecting the pivot language’s grammar onto the target.

• With both shortcuts removed, the models do not beat a majority baseline across Grambank’s features, indicating substantial headroom for inductive metalinguistic reasoning.

## 2 Methodology

## 2.1 Isolating Induction from Analogy

Theoretical Framework We introduce the task of typological feature induction (TFI): given n unlabelled sentence-level data points D in target language $L _ { t }$ and paired translations in pivot language $L _ { p }$ (English unless specified), the task is for model θ to predict values $v _ { 1 \cdot . i }$ for typological features $F _ { 1 . . i }$ . Figure 1 exemplifies the task, defined as:

$$
\hat { v } _ { 1 \ldots i } = f _ { \theta } ( D _ { n } ^ { L _ { t } , L _ { p } } , F _ { 1 \ldots i } ) , \qquad \hat { v } _ { j } \in \mathcal { V } _ { j }\tag{1}
$$

where $\nu _ { j }$ is the value set for feature $F _ { j } \left( { \bf e . g . \{ 0 , 1 \} } \right.$ or {0, 1, 2} in Grambank). Evaluation is exact match accuracy, comparing $\hat { v } _ { i }$ against the gold value $v _ { i } ^ { \mathrm { g o l d } }$ for $L _ { t }$ . Under Schneider and Anastasopoulos’s (2026) taxonomy, TFI instantiates a system-level, multilingual, processing-focused, natural metalinguistic task. Our characterisation of TFI focuses on LLMs’ inductive understanding rather than analogical (and generative) capabilities.

Data Sources We source features F and values v from Grambank (Skirgård et al., 2023), a database of 195 typological feature specifications in 2,467 languages. Grambank defines features in terms of questions, including a short linguistic description and possible values across languages.

Unlabelled language data D is derived from FLORES+ (Costa-jussà et al., 2022) and BOU-QuET (Andrews et al., 2025), both multi-parallel human-translated sentence-level corpora. We select languages in Grambank matched through Glottolog (Hammarström et al., 2026), resulting in 195 target languages (See Appendix A for full lists).

## 2.2 Isolating Induction from Recall

A model that has seen a language in pretraining can answer from memory rather than from the corpus; that is, the model relies on parametric recall rather than induction. We first probe how often the text gives the language away. We then vary the input in three ways (Table 1): ablations remove part of the corpus, a cipher hides the language’s identity, and a pivot swap changes the known side.

Language Identification To measure how often the text gives the language away, we show each model $n \in \{ 1 6 , 6 4 , 2 5 6 \}$ unlabelled pairs and ask it to name the target language. A guess counts as correct if it gives the language’s name or a recognised variant of it.

Plaintext Ablations MONO removes the pivot side and MISALIGNED shuffles the target-side lines so that pairs no longer match; the model is not told that the pairs are misaligned. PARALLEL+NAME adds the language name to PARALLEL.

Cipher To remove the shortcut through recognition, we encipher the target side, following prior work that uses substitution ciphers to test reasoning without memorisation (Khouja et al., 2026). Permuting characters within a language’s own alphabet is not enough: 87% of the Latin-script languages have a unique set of non-ASCII characters ( ˛a ´c ˛e ł is Polish), and under such a cipher Qwen3.6 still named 3–9 languages per run, citing the diacritics in its reasoning. Our cipher therefore decomposes each character into a base letter and a modifier and maps the two separately, drawing the targets from a pool shared across all languages: if a is mapped to e, then á is mapped to è. A language’s diacritic inventory is thereby replaced rather than permuted. Vowels map to vowels and consonants to consonants, so the text stays plausible as a natural language. Its cost is that enciphered text tokenizes approximately 1.3× longer, which is affordable.

Pivot Swap To test what the models take from the known side, we replace English in CIPHER-PARALLEL with 14 other pivots, chosen to cross transitive verb position (verb-medial, -final, -initial) with script (Latin, non-Latin): French, Indonesian, Swahili, Russian, Hebrew, Mandarin, Turkish, Hungarian, Korean, Hindi, Persian, Japanese, Welsh and Egyptian Arabic. A pivot that is also a target is removed from the targets in its own runs.

## 3 Experimental Setup

Baselines NAME-ONLY gives the language name and no corpus to measure pure recall. MAJOR-ITY answers every question with the most frequent value across all 2,467 Grambank languages, and RANDOM is the expected accuracy of a uniform guess over the options.

Models We evaluate two open-weight models, Gemma 4 31B-it (Team, 2026) and Qwen3.6-

<table><tr><td></td><td>Description</td></tr><tr><td>PARALLEL</td><td>n aligned pairs</td></tr><tr><td>MONO</td><td>n target sentences only</td></tr><tr><td>MISALIGNED</td><td>n pairs, target side shuffled</td></tr><tr><td>PARALLEL+NAME</td><td>n aligned pairs + language name</td></tr><tr><td>CIPHER-PARALLEL</td><td>n pairs, target side enciphered</td></tr><tr><td>CIPHER-MONO</td><td>MONO on enciphered text</td></tr><tr><td>CIPHER-MISALIGNED</td><td>MISALIGNED on enciphered text</td></tr><tr><td>NAME-ONLY</td><td>language name, no corpus</td></tr><tr><td>MAJORITY</td><td>most frequent value in Grambank</td></tr><tr><td>RANDOM</td><td>random guess over options</td></tr></table>

Table 1: Experimental settings and baselines.

27B (Qwen Team, 2026), and one closed model, GPT-5.6 Luna (OpenAI, 2026). For each language we sample corpora of $n \_ { \mathrm { ~ \in ~ } }$ {16, 32, 64, 128, 256, 512, 1024} sentence pairs. Corpora are nested: the 16 pairs at n=16 are the first 16 at n=32 etc. Each open-model setting is run with three corpus seeds; GPT-5.6 Luna is run once per setting for cost reasons. See implementation details and prompt templates in Appendix C.

Evaluation Sets We evaluate on three sets:

• Core: We manually select 22 features deemed to be inferrable from parallel sentences, covering word order, word classes and morphology. Every Grambank-coded pair of these features with the 195 languages gives 4,084 items. See the list in Appendix B.

• Latin-Script: We only apply the cipher (§2.2) to Latin-script orthographies because non-Latin scripts are more likely to reveal the language and might require completely different ciphering methods. We obtain a 2,657-item subset for 127 languages.

• Extended: This set tests features beyond the 22 core ones. Asking about every coded feature for all 195 languages would give 32,040 items, which is too costly to run. We therefore restrict this set to the languages that none of the models can identify. Here a language counts as identified if a model names it, or the broader language it belongs to, in more than half of its languageidentification runs. For these 24 languages we ask about every feature Grambank codes, giving 3,775 items over all 195 features.

## 4 Results

We first measure how much a plaintext corpus helps (RQ1, §4.1) and how much of that help comes from recognising the language (RQ2, §4.2). We then block recall in two ways and ask what the models take from the corpus instead (RQ3): by enciphering the target side (§4.3), which also lets us swap the pivot language (§4.4), and by choosing languages the models cannot identify (§4.5).

![](images/efed7c01ce3845f3fd13e932790f6ecfa6544e004369f559511dc2d80979ac3f.jpg)  
Figure 2: Accuracy of TFI on the core evaluation set as the number of sentence pairs n grows. Error bars show the standard deviation over three corpus seeds for the open models; GPT-5.6 Luna was run on one.

## 4.1 Corpus Effects

To answer RQ1, we start from plaintext corpora. Figure 2 gives accuracy on the core evaluation set. Both open models answer far above MAJORITY (0.605) even without a corpus, and a corpus raises their accuracy further: with 128 pairs, Gemma 4 gains 5.9 points over NAME-ONLY and Qwen3.6 5.6 points. GPT-5.6 Luna starts higher (0.826) and gains nothing: no corpus setting is more than about one point above NAME-ONLY, and with 16 pairs it is two points below it.

Few pairs give strong performance. For both open models, 16 pairs already deliver 70–85% of the gain observed at 128 pairs. Beyond that the curves flatten. From 16 to 128 pairs Gemma 4 gains 1.8 points and Qwen3.6 0.9 points. From 128 to 1,024 pairs Gemma 4 gains 0.3 points, and Qwen3.6 loses 1.3 points. Qwen’s decline is almost entirely a failure to answer rather than wrong answers: the number of unparsed outputs rises from about 5 to about 51 per run between 128 and 1,024 pairs (Appendix D). GPT-5.6 Luna shows the same shape at a higher level, rising from 0.806 at 16 pairs to 0.829 at 1,024, no better than NAME-ONLY.

Parallelism and alignment barely matter. If the models induced features by aligning target sentences with their translations, removing the English side (MONO) or breaking the alignment (MISALIGNED) should hurt. Up to 128 pairs neither does: at all four corpus sizes from 16 to 128 pairs, PARALLEL differs from MONO and from MIS-

![](images/6c535d270e9f28cb05427985f66c92ec992bfe71362a2855d7ac33be97941691.jpg)  
Table 2: Two reasoning patterns: (a) Naming the language and answering from recall. (b) Citing corpus pairs as evidence, including pairs that do not match.

ALIGNED by at most 0.6 points for either open model, with no consistent sign (Figure 2). The same holds for GPT-5.6 Luna, whose MONO score (0.835) is at least as high as its PARALLEL score.

Differences appear only with long corpora and remain small. For Gemma 4, alignment is worth 0.9, 1.1 and 1.8 points at 256, 512 and 1,024 pairs, and parallelism 1.0 point at 1,024. Qwen’s larger gaps at these sizes (up to 4.1 points at 1,024) come mostly from unanswered items: its MONO and MIS-ALIGNED runs fail to produce an answer for 60–206 items per run, against 17–51 for PARALLEL. Most of these outputs do not loop verbatim but keep going through corpus sentences without reaching a conclusion (Appendix D).

Models cite the corpus, regardless of alignment. Most outputs quote the corpus. On 1,000 random items at n=128, 79% of Gemma 4’s outputs, 86% of Qwen3.6’s and 92% of GPT-5.6 Luna’s quote at least one corpus sentence, on average 3.4, 5.6 and 3.5 of the 128. They quote as often in MISALIGNED (80%, 89% and 94%). Gemma 4 and GPT-5.6 Luna rarely say that the pairs do not match. We label the same 1,000 outputs using an LLM judge. 3.5% of Gemma 4’s MISALIGNED outputs and 0.2% of GPT-5.6 Luna’s say so. Qwen3.6 says so in 33%, and about half of these say that the whole corpus is shuffled, yet its accuracy is the same as in PAR-ALLEL. In Table 2b, Gemma 4 even glosses an English word with target words from an unrelated sentence. Therefore a quoted sentence does not reliably indicate that the answer came from it.

<table><tr><td></td><td></td><td>n=16</td><td>n=64</td><td>n=256</td></tr><tr><td rowspan="3">before cipher (all, 195)</td><td>Gemma 4</td><td>0.660</td><td>0.662</td><td>0.658</td></tr><tr><td>Qwen3.6</td><td>0.593</td><td>0.581</td><td>0.576</td></tr><tr><td>GPT-5.6 Luna</td><td>0.800</td><td>0.809</td><td>0.805</td></tr><tr><td rowspan="3">before cipher (Latin, 127)</td><td>Gemma 4</td><td>0.643</td><td>0.646</td><td>0.640</td></tr><tr><td>Qwen3.6</td><td>0.583</td><td>0.562</td><td>0.559</td></tr><tr><td>GPT-5.6 Luna</td><td>0.795</td><td>0.811</td><td>0.806</td></tr><tr><td rowspan="3">after cipher (Latin, 127)</td><td>Gemma 4</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Qwen3.6</td><td>0.008</td><td>0.000</td><td>0.000</td></tr><tr><td>GPT-5.6 Luna</td><td>0.039</td><td>0.039</td><td>0.031</td></tr></table>

Table 3: Language identification accuracy, i.e. the proportion of languages each model names correctly from n unlabelled sentence pairs.

## Language name adds nothing beyond corpora.

Giving the language name on top of the corpus (PARALLEL+NAME) leaves Gemma 4 unchanged at 128 pairs (+0.06 points) and lowers Qwen3.6 by 1.0 point. Whatever information the name carries, the target-language text already supplies it.

Interpretation Taken together, the corpus behaves less like evidence to be analysed than like a cue: a handful of target-language sentences is enough to reach the plateau, the translations and their alignment add little, naming the language adds nothing further, and the pairs the models cite need not match. This is the pattern expected if the text mainly activates what the model already knows about the language, which we test next.

## 4.2 Language Identification

Models can identify the language. For RQ2, we first ask whether the models recognise the language at all. We give each model n sentence pairs and ask it to name the target language (Table 3). From only 16 pairs, Gemma 4 names 66% of the 195 languages correctly, Qwen3.6 59% and GPT-5.6

![](images/51d880dd2dc5a04dca18f61f3c512acbb01e4db09818348a44ea365570ef06de.jpg)  
Figure 3: Proportions of outputs that name a specific language, in PARALLEL and CIPHER-PARALLEL, on the Latin-script evaluation set.

Luna 80%; more pairs do not help. On the 127 Latin-script languages to which the cipher applies, the rates are almost the same (64%, 58% and 80%); these languages are no less recognisable than the rest. This indicates that the LLMs have substantial exposure to the target languages during pretraining.

Models name languages while answering. To see whether the models also name the language while answering the typological questions, we use gpt-oss-20b (OpenAI, 2025) as a judge on the outputs of the two open models in PARALLEL and CIPHER-PARALLEL, using the Latin-script evaluation set. The judge extracts the language that the output identifies as the target (Appendix E).

In PARALLEL, Gemma 4 names a specific language in 36–40% of its outputs and Qwen3.6 in 57–63%, mostly the correct one (Figure 3). The name usually comes in the first sentence, and the output then draws on knowledge of that language (Table 2a). Both models name the language more often with more corpus: from n=16 to n=128 the share rises by 3 points for Gemma 4 and 6 points for Qwen3.6, presumably because more sentences give more cues for recognising the language.

Identification is not the whole story. The gain is not confined to languages the model can name: for the languages Gemma 4 cannot identify from 16 pairs, the corpus adds 7.1 points over NAME-ONLY at 128 pairs, more than the 5.3 points it adds for the languages it names. The text may therefore also reveal partial knowledge, such as the language’s family or area. Either way, plaintext accuracy mixes recall with whatever the models induce, so we next remove the language’s identity from the text.

![](images/61e9dc7ee8a12a4b73034af9863997a9479a8a484df959984756e83dcfdfe0b4.jpg)  
Figure 4: Accuracy of TFI on the 127 Latin-script languages with the target side enciphered. Error bars show the standard deviation over three corpus seeds.

## 4.3 Induction and Ciphering

For RQ3, we first block recall by enciphering the target side and ask whether the corpus matters.

Ciphers provide a clean testbed. The cipher all but removes identification on the 127 languages (Table 3, bottom): from 64% to 0% for Gemma 4 and from 58% to at most 0.8% for Qwen3.6. GPT-5.6 Luna still names 3–4%, all from a handful of well-known European languages (Italian at every n). Nor do the models recognise the language while answering TFI items: under the cipher almost no output names a language (Figure 3). The cipher thus largely blocks the route through recall.

After ciphering, corpus size, parallelism and alignment all matter. Figure 4 shows accuracy on the 127 enciphered languages. Unlike in plaintext, what the input contains now makes a clear difference. Going from 16 to 128 pairs raises accuracy by 2.8 points for Gemma 4 and 2.3 for Qwen3.6. Without the pivot side (CIPHER-MONO), Gemma 4 loses 2.5 points at 16 pairs and 4.3 at 128, and Qwen3.6 loses 2.6 and 3.5. Shuffling the pairs (CIPHER-MISALIGNED) costs more still: 2.9 and 5.3 points for Gemma 4, 2.8 and 6.1 for Qwen3.6. Once the models can no longer recognise the language, they do make use of the English side and of how it lines up with the target.

## 4.4 Pivot Language Effects

What exactly do the models take from the pivot side (RQ3)? If they used it to locate constructions in the target, the benefit should not depend on what the pivot’s own grammar looks like. We run the pivot swap (§2.2) on the ciphered data at n=128 with 15 pivots including English.

Pivots help only where their grammar is the answer. For each item we ask whether the target’s Grambank value equals the pivot language’s own value for that feature. Figure 5 shows the gain over CIPHER-MONO for the two kinds of item. The pattern is the same for almost every pivot: adding the pivot side raises accuracy where the target agrees with the pivot and lowers it where it does not. The models appear to read the target through the pivot’s grammar rather than through the target’s own: with Japanese (verb-final) as the pivot, Gemma 4 calls 74% of the non-verb-final targets verb-final, against 15% without a pivot. Figure 1 shows the same projection on a single item: Qwen3.6 aligns an enciphered noun phrase with its English translation word by word and reads off the English order.

Pivots pull answers to their own values. A pivot that agrees with the target raises accuracy, but this has two explanations: the models may use the pivot to analyse the target, or they may simply copy the pivot’s value, which is then also correct. Items where target and pivot disagree tell the two apart. If the models analysed the target, the pivot side should make them give the target’s value more often. Instead, it makes them give the pivot’s value more often (Figure 6). Over all items, adding the pivot side raises the chance of answering with the pivot’s value by 20 points for Gemma 4 and 14 for Qwen3.6, and does not raise the chance of answering with the target’s value (Appendix F). The pull is strongest on word order (Figure 8 in Appendix F).

This qualifies the result of §4.3. Once the target is unrecognisable, the parallel side does change the answers, but mostly by moving them towards the pivot’s values. With English as the pivot, this projection looks like induction, because English shares its value with the target on 62% of items.

## 4.5 Extending the Feature Set

Another way to block recall for RQ3 is to choose languages the models do not know. The extended set does this, and by asking about every Grambank feature coded for these languages it also tests whether a model can find a less common value.

Models do not beat the majority baseline. On all items, neither model reaches MAJORITY (0.748) at any corpus size (Figure 7, top). NAME-ONLY, which shows what the models recall about these languages, is already below MAJORITY (0.729 for Gemma 4, 0.683 for Qwen3.6), and the corpus brings both close to MAJORITY but not above it.

![](images/828dc32ecb9cec79cd4fc635fb5b46d60d6311338fdbb51d250010f572301e5d.jpg)  
Figure 5: Change in accuracy from adding different pivot sides to an enciphered corpus (n=128; CIPHER-PARALLEL − CIPHER-MONO).

![](images/781f4ea1925b132e31adb2b1f58352b78c94ac2166cf9b705857de175f1a494b.jpg)  
Figure 6: Distribution of answers on Grambank features, for items where the target’s value differs from the pivot’s, without the pivot side (CIPHER-MONO) and with it (CIPHER-PARALLEL).

The non-majority items, where the most frequent value is wrong, show whether a model can find a less common value from the text (Figure 7, bottom). A model that leans towards frequent values scores low here regardless, so we compare each model with itself without a corpus (NAME-ONLY). Adding a corpus lowers accuracy rather than raising it: at 16 pairs, Gemma 4 falls from 0.410 to 0.367 and Qwen3.6 from 0.466 to 0.355, and more pairs do not bring either model clearly above NAME-ONLY.

## 5 Discussion

We now discuss implications for language documentation and research on metalinguistic abilities.

![](images/8ea44ae39c6660c5238b57f111c9aeb21751bd0a5f1efe308a5e12ffad3c57af.jpg)  
Figure 7: Accuracy on the extended set as the number of sentence pairs n grows. The top panel shows all 3,775 items, the bottom panel the 952 items whose answer is not the most frequent value in Grambank. Error bars show the standard deviation over three corpus seeds.

## 5.1 Language Documentation

Despite more than two decades of NLP work aimed at language documentation, documentary linguists rarely use it, partly because NLP projects assume an amount or type of data that linguists do not have (Gessler, 2022; Gessler et al., 2025). Typological prediction is a case in point. It has mostly meant predicting features from other features of the same language (Bjerva et al., 2020), and in a survey by Bjerva (2024), 34 linguists rate such predictions as not useful; what they ask for instead is prediction from text, with attributions to the examples in the data. TFI is set up for this case: its input is the kind of data documentation produces, and its answers can be checked against expert coding.

Gessler et al. (2025) urge NLP researchers to state when a method would actually help linguists;

our results suggest when an LLM would not. On languages it does not already know, the ones that need documenting, a model falls back on typological majority and on the pivot’s grammar. LLM output on an unfamiliar language is thus best treated as a hypothesis for the linguist to check, especially when it matches the pivot’s value.

This division of labour reflects a basic difference between the two. A field linguist starts from no knowledge of the language and checks hypotheses against the data; a model starts from broad knowledge of many languages and, as we find, uses it in place of the data whenever it can. Accuracy alone therefore cannot put the two on one scale.

## 5.2 Metalinguistic Abilities of LLMs

We posited that metalinguistic tasks are hard partly because they conflate inducing a rule with applying it. Our results show that induction alone is already hard: once recall is blocked, the models fall back on typological majority, much as models on constructed languages handle common patterns far better than rare ones (Taguchi and Sproat, 2026). Part of the headroom on olympiad problems and grammar book translation (Aycock et al., 2025; Choudhary et al., 2025) may thus lie in finding the rule. The converse of TFI, which states the rule and asks the model to apply it (Marmonier et al., 2025; Yang et al., 2025), would measure the other step, and paired with TFI it would locate a failure on either side of the rule.

Benchmarks built on natural text must also block recall, since models fall back on what they know of a language whenever they recognise it. Recent work therefore enciphers the data (Bafna et al., 2026; Khouja et al., 2026), but a cipher within the language’s own alphabet leaves its diacritics as a fingerprint. Future benchmarks could remove recall more finely, for instance with models whose training data is open. The pivot also deserves closer study, since our results show that its grammar can pass for induction.

Tools may help with induction, much of which is mechanical work that a program does more reliably than a model in context: aligning words, retrieving sentences with a candidate marker and counting word orders. Typological signal can even be extracted from representations learned over parallel text (Östling and Kurfalı, 2023). Equipping models with such tools and asking them to cite the pairs behind each answer is a natural next step.

## 6 Related Work

Typology in NLP Typological databases such as WALS (Dryer and Haspelmath, 2013) and Grambank (Skirgård et al., 2023) serve NLP as a source of features for cross-lingual transfer (Ammar et al., 2016) and as a yardstick for the diversity of multilingual evaluation (Ploeger et al., 2024). Completing typological databases is a longstanding task (Ponti et al., 2019), whether from other features (Bjerva et al., 2020), from parallel text through word alignment or learned language representations (Östling, 2015; Malaviya et al., 2017), from a language’s name and its genealogical and areal neighbours (Wang et al., 2026), or from grammatical descriptions and glossed examples (Hus and Anastasopoulos, 2026; Yang et al., 2026). These approaches take the language’s identity or a description of it as given, or process a whole corpus with a dedicated pipeline; TFI instead tests whether a model can induce a feature in context from raw text, as linguists have asked for (Bjerva, 2024).

Metalinguistic Reasoning LLMs’ ability to reason about language has been tested with linguistics olympiad puzzles (¸Sahin et al., 2020; Bean et al., 2024; Sánchez et al., 2026; Majmudar and Filatova, 2025; Choudhary et al., 2025), gloss prediction from grammar snippets (Yang et al., 2025), and translation from a grammar book and dictionary (Tanzer et al., 2024; Zhang et al., 2024a,b; Hus and Anastasopoulos, 2024; Zhang et al., 2025; Aycock et al., 2025). Related work asks whether models can learn from explicit grammatical explanations (Marmonier et al., 2025) or benefit from stepby-step linguistic analyses (Pei et al., 2026). Since test languages may have been seen in pretraining, recent work constructs novel languages (Taguchi and Sproat, 2026), or obfuscates natural languages with ciphers and permutations (Bafna et al., 2026; Khouja et al., 2026). These all conflate inducing a rule with applying it; TFI isolates induction.

## 7 Conclusion

We introduce the task of typological feature induction to isolate inductive metalinguistic reasoning. We find that models heavily rely on recall; on languages they cannot identify, they fail to beat a majority baseline across Grambank’s features. This substantial headroom indicates models’ inductive reasoning skills remain weak. We therefore emphasise to both linguists and NLP practitioners that such models are immature for direct application to real-world field linguistics tasks. We hope our findings stimulate further research into understanding, and improving LLMs’ metalinguistic abilities.

## Limitations

Our corpora are FLORES+ and BOUQuET, chosen because they cover many languages with the same content. They consist of written text, most of it translated rather than originally composed in the target language, whereas documentation corpora are often transcribed speech or elicited sentences. Translation may bring the target text closer to the source language in places, but this cannot account for the pivot effect we report: the target text is identical across the 15 pivots, and answers follow whichever pivot is shown (§4.4). Testing TFI on documentation corpora is a natural extension.

We treat a language as unknown when none of the models names it (§4.5). A model may still recognise the family or region of such a language, so this set reduces rather than removes recall. Any remaining recall would raise accuracy, so it does not affect our finding that the models fail to beat the majority baseline.

Finally, some Grambank features may not be inducible from a given sample of sentences, as the corpora were not built to exhibit them. Our accuracies are therefore a lower bound on what a model could achieve with sufficient evidence. In future work, we plan to have linguists mark which features a sample of the data supports, which would allow precision and recall to be measured over inducible features only, closer to the task of a field linguist.

## References

Waleed Ammar, George Mulcaire, Miguel Ballesteros, Chris Dyer, and Noah A. Smith. 2016. Many languages, one parser. Transactions ofthe Association for Computational Linguistics, 4:431–444.

Pierre Andrews, Mikel Artetxe, Mariano Coria Meglioli, Marta R. Costa-jussà, Joe Chuang, David Dale, Mark Duppenthaler, Nathanial Paul Ekberg, Cynthia Gao, Daniel Edward Licht, Jean Maillard, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Eduardo Sánchez, Ioannis Tsiamas, Arina Turkatenko, Albert Ventayol-Boada, and Shireen Yates. 2025. BOUQuET : dataset, benchmark and open initiative for universal quality evaluation in translation. In Proceedings ofthe 2025 Conference

on Empirical Methods in Natural Language Processing, pages 27515–27535, Suzhou, China. Association for Computational Linguistics.

Seth Aycock, David Stap, Di Wu, Christof Monz, and Khalil Sima’an. 2025. Can LLMs Really Learn to Translate a Low-Resource Language from One Grammar Book? In Proceedings ofThe Thirteenth International Conference on Learning Representations, pages 12334–12357, Singapore.

Niyati Bafna, Ryan Soh-Eun Shim, Barbara Plank, David Yarowsky, and Hale Sirin. 2026. Rashid: A cipher-based framework for exploring in-context language learning. arXiv preprint arXiv:2603.22497.

Andrew Bean, Simi Hellsten, Harry Mayne, Jabez Magomere, Ryan Chi, Scott A Hale, Hannah R Kirk, et al. 2024. Lingoly: A benchmark of olympiadlevel linguistic reasoning puzzles in low resource and extinct languages. Advances in Neural Information Processing Systems, 37:26224–26237.

Johannes Bjerva. 2024. The role of typological feature prediction in NLP and linguistics. Computational Linguistics, 50(2):781–794.

Johannes Bjerva, Elizabeth Salesky, Sabrina J. Mielke, Aditi Chaudhary, Giuseppe G. A. Celano, Edoardo Maria Ponti, Ekaterina Vylomova, Ryan Cotterell, and Isabelle Augenstein. 2020. SIGTYP 2020 Shared Task: Prediction of Typological Features. In Proceedings of the Second Workshop on Computational Research in Linguistic Typology, pages 1–11, Online. Association for Computational Linguistics.

Mukund Choudhary, KV Aditya Srivatsa, Gaurja Aeron, Antara Raaghavi Bhattacharya, Dang Khoa Dang Dinh, Ikhlasul Akmal Hanif, Daria Kotova, Ekaterina Kochmar, and Monojit Choudhury. 2025. UNVEIL-ING: What makes linguistics olympiad puzzles tricky for LLMs? In Second Conference on Language Modeling.

Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, and 19 others. 2022. No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv preprint. ArXiv:2207.04672 [cs].

Matthew S. Dryer and Martin Haspelmath, editors. 2013. WALS Online (v2020.6). Zenodo.

Luke Gessler. 2022. Closing the NLP Gap: Documentary Linguistics and NLP Need a Shared Software Infrastructure. In Proceedings of the Fifth Workshop on the Use of Computational Methods in the Study of Endangered Languages, pages 119–126, Dublin, Ireland. Association for Computational Linguistics.

Luke Gessler, Alexis Palmer, and Katharina Von Der Wense. 2025. Understanding the Gap: an Analysis of Research Collaborations in NLP and Language Documentation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 867– 877, Vienna, Austria. Association for Computational Linguistics.

Harald Hammarström, Robert Forkel, Martin Haspelmath, and Sebastian Bank. 2026. Glottolog 5.3.

Jonathan Hus and Antonios Anastasopoulos. 2024. Back to school: Translation using grammar books. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 20207–20219, Miami, Florida, USA. Association for Computational Linguistics.

Jonathan Hus and Antonios Anastasopoulos. 2026. A RAG approach for typological database completion. In Proceedings of the 8th Workshop on Research in Computational Linguistic Typology and Multilingual NLP, pages 39–49, Rabat, Morocco. Association for Computational Linguistics.

Jude Khouja, Lingyi Yang, Karolina Korgul, Simeon Hellsten, Vlad A. Neacsu, Harry Mayne, Ryan Othniel Kearns, Andrew M. Bean, and Adam Mahdi. 2026. LINGOLY-TOO: Disentangling reasoning from knowledge with templatised orthographic obfuscation. In The Fourteenth International Conference on Learning Representations.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Patrick Littell, David R. Mortensen, Ke Lin, Katherine Kairis, Carlisle Turner, and Lori Levin. 2017. URIEL and lang2vec: Representing languages as typological, geographical, and phylogenetic vectors. In Proceedings of the 15th Conference of the European Chapter ofthe Associationfor Computational Linguistics: Volume 2, Short Papers, pages 8–14, Valencia, Spain. Association for Computational Linguistics.

Neh Majmudar and Elena Filatova. 2025. Can LLMs generate and solve linguistic olympiad puzzles? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 19163–19200, Suzhou, China. Association for Computational Linguistics.

Chaitanya Malaviya, Graham Neubig, and Patrick Littell. 2017. Learning language representations for typology prediction. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pages 2529–2535, Copenhagen, Denmark. Association for Computational Linguistics.

Malik Marmonier, Rachel Bawden, and Benoît Sagot. 2025. Explicit learning and the LLM in machine

translation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 31372–31422, Suzhou, China. Association for Computational Linguistics.

OpenAI. 2025. gpt-oss-20b model card. Preprint, arXiv:2508.10925.

OpenAI. 2026. GPT-5.6: Frontier intelligence that scales with your ambition.

Robert Östling. 2015. Word order typology through multilingual word alignment. In Proceedings ofthe 53rd Annual Meeting ofthe Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 205–211, Beijing, China. Association for Computational Linguistics.

Robert Östling and Murathan Kurfalı. 2023. Language embeddings sometimes contain typological generalizations. Computational Linguistics, 49(4):1003– 1051.

Renhao Pei, Yihong Liu, Sampo Pyysalo, Hinrich Schütze, and Shaoxiong Ji. 2026. Reasoning over grammar: Can synthetic linguistic reasoning traces enhance low-resource machine translation? arXiv preprint arXiv:2606.03782.

Esther Ploeger, Wessel Poelman, Miryam de Lhoneux, and Johannes Bjerva. 2024. What is ”Typological Diversity” in NLP? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5681–5700, Miami, Florida, USA. Association for Computational Linguistics.

Esther Ploeger, Wessel Poelman, Andreas Holck Høeg-Petersen, Anders Schlichtkrull, Miryam de Lhoneux, and Johannes Bjerva. 2026. A principled framework for evaluating on typologically diverse languages. Computational Linguistics, 52(1):237–269.

Edoardo Maria Ponti, Helen O’Horan, Yevgeni Berzak, Ivan Vulic, Roi Reichart, Thierry Poibeau, Ekate-´ rina Shutova, and Anna Korhonen. 2019. Modeling language variation and universals: A survey on typological linguistics for natural language processing. Computational Linguistics, 45(3):559–601.

Qwen Team. 2026. Qwen3.6-27B: Flagship-level coding in a 27B dense model.

Gözde Gül ¸Sahin, Yova Kementchedjhieva, Phillip Rust, and Iryna Gurevych. 2020. PuzzLing Machines: A Challenge on Learning From Small Data. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1241–1254, Online. Association for Computational Linguistics.

Eduardo Sánchez, Belen Alastruey, Christophe Ropers, Arina Turkatenko, Pontus Stenetorp, Mikel Artetxe, and Marta R. Costa-jussà. 2026. Linguini: A benchmark for language-agnostic linguistic reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Nathan Schneider and Antonios Anastasopoulos. 2026. Speaking of language: Reflections on metalanguage research in NLP. In Proceedings ofThe Big Picture v2: Crafting a Research Narrative, pages 107–118, San Diego, CA, USA. Association for Computational Linguistics.

Hedvig Skirgård, Hannah J. Haynie, Harald Hammarström, Damián E. Blasi, Jeremy Collins, Jay Latarche, Jakob Lesage, Tobias Weber, Alena Witzlack-Makarevich, Michael Dunn, Ger Reesink, Ruth Singer, Claire Bowern, Patience Epps, Jane Hill, Outi Vesakoski, Noor Karolin Abbas, Sunny Ananth, Daniel Auer, and 100 others. 2023. Grambank v1.0.

Chihiro Taguchi and Richard Sproat. 2026. Creating ConLangs to probe the metalinguistic grammatical knowledge of LLMs. In Findings of the Association for Computational Linguistics: ACL 2026, pages 29096–29148, San Diego, California, United States. Association for Computational Linguistics.

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. 2024. A Benchmark for Learning to Translate a New Language from One Grammar Book. In Proceedings ofthe Twelfth International Conference on Learning Representations.

Gemma Team. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Qianwen Wang, York Hay Ng, Aditya Khan, and En-Shiun Annie Lee. 2026. Typological feature prediction with large language models: An in-context learning approach. arXiv preprint arXiv:2609.03775.

Changbing Yang, Christopher Hammerly, Freda Shi, and Jian Zhu. 2026. Llm agents as computational typologists. arXiv preprint arXiv:2609.07791.

Changbing Yang, Franklin Ma, Freda Shi, and Jian Zhu. 2025. LingGym: How far are LLMs from thinking like field linguists? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1314–1340, Suzhou, China. Association for Computational Linguistics.

Chen Zhang, Jiuheng Lin, Xiao Liu, Zekai Zhang, and Yansong Feng. 2025. Read it in two steps: Translating extremely low-resource languages with codeaugmented grammar books. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3977–3997, Vienna, Austria. Association for Computational Linguistics.

Chen Zhang, Xiao Liu, Jiuheng Lin, and Yansong Feng. 2024a. Teaching large language models an unseen language on the fly. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 8783–8800, Bangkok, Thailand. Association for Computational Linguistics.

Kexun Zhang, Yee Man Choi, Zhenqiao Song, Taiqi He, William Yang Wang, and Lei Li. 2024b. Hire a linguist!: Learning endangered languages in LLMs with in-context linguistic descriptions. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 15654–15669, Bangkok, Thailand. Association for Computational Linguistics.

## A Full Language List

Table 8 lists the 195 target languages with their ISO 639-3 code, Glottocode, language family, script (ISO 15924 codes) and source corpus, and marks which evaluation sets each belongs to. All 195 languages are in the core set; 127 are in the Latinscript set and 24 in the extended set. The languages come from 40 families, led by Indo-European (49), Atlantic-Congo (26), Austronesian (23) and Afro-Asiatic (14), and are written in 26 scripts.

Latin-script and extended sets. The Latin-script set contains the Latin-script languages that the cipher (§2.2) can encipher. The cipher maps each letter that decomposes into a base letter and a diacritic onto a pool of precomposed Latin letters, and permutes the letters without a base letter (seven letters such as æ, N and œ) among themselves. Seven of the 134 Latin-script languages need more letters than this pool holds and are excluded. Ewe, Fon, Kxoe, Hassaniyya and Dar Daju Daju use many letters without a base letter, such as E, O and click letters, and Vietnamese and Duala add tone marks to vowels that already carry other diacritics. The extended set contains the 24 languages that none of the three models identifies (§3): Arhuaco, Bafia, Brahui, Burushaski, Central Kanuri, Dandami Maria, Dar Daju Daju, Duala, Dyula, Eton (Cameroon), Fur, Hassaniyya, Hehe, Koyra Chiini Songhay, Kxoe, Lendu, Lugbara, Mai Brat, Ngambay, Nuer, Siamou, Western Cham, Xârâcùù and Yemsa. Fifteen of them are in the Latin-script set.

Matching corpus varieties to Grambank. Grambank does not list ISO 639-3 codes, so corpus languages are matched to Grambank entries through Glottolog. Grambank codes a few languages at the level of a macrolanguage while the corpora use the codes of its members; we assign the macrolanguage’s values to the member that is its standard variety, giving Bosnian, Croatian and Serbian the values of Serbo-Croatian (sout1528) and Standard Latvian those of Latvian (latv1249). Where FLORES+ has several dialect versions of one language, we keep only the variety that Grambank codes (North Levantine Arabic, standard Catalan). Twi, for which Grambank codes only the Akan macrolanguage, is excluded. Each target uses

a single script.

<table><tr><td></td><td></td><td></td><td>Lang. Feat. Items Used for</td></tr><tr><td>Core</td><td>195</td><td>224,084</td><td>§4.1</td></tr><tr><td>Latin-Script</td><td>127</td><td></td><td>22 2,657 §4.2–§4.4</td></tr><tr><td>Extended</td><td>24</td><td></td><td>195 3,775 §4.5</td></tr></table>

Table 4: Statistics of the three evaluation sets. Feat. is the number of distinct Grambank features asked about. The extended set asks about every feature coded for each of its languages.

## B Features

Statistics The statistics of three evaluation sets are shown in Table 4.

List of Core Features Table 11 lists the 22 Grambank features in the core set, grouped into word order, word classes and morphology following Grambank’s own feature groups. We chose features whose value shows up in word order or overt marking within a sentence, so that a sentence-level corpus can in principle decide them. Not every feature is coded for every language, so the number of items per feature ranges from 162 to 194. Some features are strongly skewed: 180 of 186 languages lack demonstrative classifiers (GB038), and 157 of 194 have SV order in intransitive clauses (GB130).

## C Implementation Details

## C.1 Prompt Templates

Each item is sent as a single user turn through the model’s chat template, with no system prompt. The corpus comes first and the question second, so the 22 items of a language share the same prompt prefix. Answer options use Grambank’s numeric value codes and are listed in a fixed random order per item; the answer is read from the final Answer: line.

Question Every setting ends with the same question block. {question} is Grambank’s question and {summary} the Summary section of its feature description. We do not use the Examples section, which names languages together with their values.

You are a linguist answering a question about the grammar   
of a specific language.   
[Language: {language} (ISO 639-3: {iso}) only in NAME-  
ONLY and +NAME]   
Question: {question}   
What this question means:   
{summary}   
Options:   
{code} = {label}   
{instruction} You must choose exactly one option.   
End your reply with a final line in the form: Answer:   
<{codes}>

Here is an example of Polish for the feature GB024, with the corpus shortened to one pair:

PARALLEL, Polish, GB024   
Below is a parallel corpus between English and an unnamed   
target language. Each pair is one sentence: EN is English,   
TL is the target language.   
EN: 28-year-old Vidal had joined Barça three seasons ago,   
from Sevilla.   
TL: 28-letni Vidal przyszedł do Barcelony z Sewilli trzy   
sezony temu.   
[. . . n pairs in total]   
---   
You are a linguist answering a question about the grammar   
of a specific language.   
Question: What is the order of numeral and noun in the   
NP?   
What this question means:   
This feature focuses on the order of cardinal numerals (not   
ordinal numerals) and nouns in quantified noun phrases. In   
some languages only one order (Numeral-Noun or Noun-  
Numeral) is possible. In other languages both orders may   
be possible. This can take the form of free variation (where   
Numeral-Noun and Noun-Numeral order can occur inter  
changeably). The order may also depend on some other   
grammatical category which is neutral with respect to prag  
maticality (e.g. definiteness). If the order depends on the   
noun, consider only the orders that apply to an open class of   
nouns and disregard orders that are restricted to a specific   
closed class of nouns.   
Options:   
2 = N-Num   
1 = Num-N   
3 = both   
Answer based only on the corpus above. You must choose   
exactly one option.   
End your reply with a final line in the form: Answer:   
<2/1/3>

GPT-5.6 Luna reasons before answering, but the API does not return this reasoning, so its outputs would otherwise contain little more than the answer line. To see what evidence it uses, we ask it to write out its analysis. In the plaintext settings and NAME-ONLY, the sentence End your reply with a final line in the form: is replaced by First write your step-by-step analysis of the evidence, quoting the specific sentences you rely on. Then end with a final line in theform: (in NAME-ONLY: First write your step-by-step reasoning. Then end with afinal line in theform:).

Language Identification The probe uses the header and corpus of PARALLEL (or CIPHER-PARALLEL), followed by:

Language identification   
What is the target language (TL)? Reply with your single   
best guess.   
If you are unsure, still give your best guess.   
End your reply with a final line in the form: Language:   
<name>

## C.2 Hyperparameters

Open models run on vLLM (Kwon et al., 2023) in bfloat16 with each model’s recommended sampling parameters (temperature 1.0, top-p 0.95; top-k 64 for Gemma, 20 for Qwen), thinking mode off, and a 12,288-token output budget. Open models ran over four NVIDIA A100 40GB GPUs per job. All experiments, including pilot runs, took about 2,000 GPU hours. GPT-5.6 Luna uses server defaults with a 4,096-token reasoning budget; in the plaintext settings its prompt additionally asks it to explain its answer.

Our open models run with thinking mode off, although they still reason step by step in their outputs. To check that this does not understate their ability, we ran both models with thinking on in CIPHER-PARALLEL at n=128 on the first corpus seed, with a 32,768-token output budget and otherwise the same settings. Thinking makes outputs 5–14 times longer but does not raise overall accuracy for either model: 0.622 with thinking vs. 0.636 without for Qwen3.6, and 0.644 in both cases for Gemma 4, and the two runs give the same answer on 83% and 79% of items.

## C.3 Artifacts and Licenses

Grambank and Glottolog are released under CC BY 4.0, FLORES+ and BOUQuET under CC BY-SA 4.0. FLORES+ and BOUQuET require accepting their terms of use before download, so we release only item definitions, corpus sentence IDs and code, not the text. Gemma 4, Qwen3.6 and gpt-oss-20b are used under their published model licenses, and GPT-5.6 Luna through its API. All are used for research evaluation, in line with their intended use.

## D Unanswered items

All accuracies in the main text count an unanswered item as wrong. Table 5 separates the two components. For Gemma 4 the distinction is immaterial: at most 16 of the 4,084 items per run go unanswered, and accuracy on answered items is within 0.3 points of accuracy on all items. For Qwen3.6 it is not. Between 128 and 1,024 pairs, the number of unanswered items per run rises from 5 to 51 under PARALLEL and from 11–29 to 191–198 under MONO and MISALIGNED. Accuracy on answered items changes much less: from 0.789 to 0.784 under PARALLEL, from 0.792 to 0.784 under MISALIGNED, and from 0.792 to 0.770 under MONO. The decline of Qwen3.6 with corpus size is therefore mostly a decline in the rate at which it answers.

Runaway Outputs Almost all unanswered items are outputs that reach the 12,288-token limit. They are rarely verbatim loops; typically the model works through corpus sentences one by one (“Let’s look at: EN: . . . TL: . . . ”) without reaching a conclusion. They concentrate on features that require locating the verb or the negator in every sentence (GB130–GB133, GB137, GB138), and more so without aligned translations: at 512 pairs, 3.5% of these items under PARALLEL, 9.7% under MONO and 12.7% under MISALIGNED. Since the outputs show no sign of converging, a larger budget is unlikely to recover them, and we count them as wrong.

Context Overflow At 1,024 pairs the Santali corpus (Ol Chiki script) exceeds Qwen3.6’s 262,144- token context window, so its 21 items return no output in every run (0.5 points). This is the only case of context overflow in the experiments.

## E Language Naming in Reasoning

We use gpt-oss-20b with low reasoning effort and greedy decoding, so that the judge is neither of the evaluated models. For each output it extracts the most specific language, family or region that the output says the target is, together with a supporting quote. Languages mentioned for comparison (“unlike English”), the pivot language, and names that occur inside quoted corpus text do not count. Whether an extracted name is the target is decided by the same matcher as Table 3. We judged the outputs of the first corpus seed at $n \in \{ 1 6 , 1 2 8 \}$ 21,256 outputs in total.

In a pilot on 100 outputs, the judge sometimes recognised the language itself from quoted corpus words although the output named none, so we discard any extracted name that does not occur in the output. A random sample of 40 remaining extracted names were all correct.

<table><tr><td></td><td></td><td colspan="3">n=128</td><td colspan="3">n=512</td><td colspan="3">n=1024</td></tr><tr><td></td><td></td><td>all</td><td>answered</td><td>unans.</td><td>all</td><td>answered</td><td>unans.</td><td>all</td><td>answered</td><td>unans.</td></tr><tr><td></td><td>PARALLEL</td><td>0.788</td><td>0.789</td><td>5</td><td>0.777</td><td>0.786</td><td>47</td><td>0.775</td><td>0.784</td><td>51</td></tr><tr><td>Qwen3.6-27B</td><td>MONO</td><td>0.790</td><td>0.792</td><td>11</td><td>0.749</td><td>0.777</td><td>146</td><td>0.734</td><td>0.770</td><td>191</td></tr><tr><td></td><td>MISALIGNED</td><td>0.787</td><td>0.792</td><td>29</td><td>0.749</td><td>0.786</td><td>193</td><td>0.746</td><td>0.784</td><td>198</td></tr><tr><td></td><td>PARALLEL</td><td>0.807</td><td>0.807</td><td>0</td><td>0.810</td><td>0.810</td><td>1</td><td>0.810</td><td>0.812</td><td>7</td></tr><tr><td>Gemma 4 31B</td><td>MONO</td><td>0.803</td><td>0.803</td><td>1</td><td>0.804</td><td>0.805</td><td>3</td><td>0.800</td><td>0.801</td><td>8</td></tr><tr><td></td><td>MISALIGNED</td><td>0.801</td><td>0.801</td><td>0</td><td>0.799</td><td>0.799</td><td>2</td><td>0.792</td><td>0.795</td><td>16</td></tr></table>

Table 5: Accuracy on all 4,084 items (unanswered counted as wrong, as in the main text), accuracy on answered items only, and the number of unanswered items per run. Mean over three corpus seeds.

Table 6 splits accuracy by whether the plaintext output names the target correctly.

<table><tr><td colspan="3"></td><td>Named correctly Not named</td></tr><tr><td rowspan="2">Gemma 4</td><td>n=16</td><td> $. 7 8 4  . 6 2 0$ </td><td> $. 7 6 0 \to . 6 1 2$ </td></tr><tr><td>n=128</td><td> $. 8 2 1  . 6 7 6$ </td><td> $. 7 6 1  . 6 2 3$ </td></tr><tr><td rowspan="2">Qwen3.6</td><td>n=16</td><td> $. 7 9 3  . 5 8 8$ </td><td> $. 7 1 5 \to . 6 4 2$ </td></tr><tr><td> $n { = } 1 2 8$ </td><td> $. 8 1 0 \to . 6 4 6$ </td><td> $. 7 2 4 \to . 6 3 8$ </td></tr></table>

Table 6: Accuracy in PARALLEL → CIPHER-PARALLEL on the Latin-script items, split by whether the plaintext output names the target correctly (first corpus seed).

## F Further Analyses of Pivot Swap

Exceptions in Figure 5. English and French with Gemma 4 go the same way as the other pivots but miss significance on the disagreeing items; only Gemma 4 with Mandarin and Qwen3.6 with Hungarian show no effect on either side.

Choice model. Each option of each answer is one row (three answers per item, one per corpus seed). A linear probability model with a fixed effect per answer predicts whether the option was chosen from three indicators: it is the target’s value, the pivot’s own value, or the majority value across Grambank. We fit it on CIPHER-MONO and CIPHER-PARALLEL together, letting each coefficient change in CIPHER-PARALLEL, with standard errors clustered by target and pivot. CIPHER-MONO never shows the pivot, so its pivot coefficient (10 points for Gemma 4, 11 for Qwen3.6) only measures how often the pivot’s value is what the models say anyway, and the effect of the pivot side is the change from CIPHER-MONO to CIPHER-PARALLEL. The pivot side raises the coefficient of the pivot’s value by 20.0 points for

![](images/13ef6288e0ad3289bce11a71099f0daeeb1c2e19628de50e5e3ab434ef8cbe6b.jpg)  
Figure 8: Change in accuracy when the pivot shares the target’s value for the feature, with a fixed effect for each target and feature, so that pivots are compared on the same question (n=128, 15 pivots). Bars are 95% confidence intervals with standard errors clustered by target and pivot.

Gemma 4 (t=7.9) and 14.4 for Qwen3.6 (t=8.0), and changes that of the target’s value by −0.4 (t=−0.3) and −2.3 (t=−1.8). A conditional logit gives the same result (change in the pivot coefficient +0.49 and +0.37 log-odds, in the target coefficient +0.04 and −0.01). With a fixed effect for each target and feature, a matching pivot raises accuracy as much when its value is not the majority one.

Word order. Comparing pivots on the same target and feature, a pivot that shares the target’s value raises accuracy by 46 points on word-order features for Gemma 4 and 36 for Qwen3.6, against 6–14 on word classes and 7–10 on morphology (Figure 8). The effect is largest for adjective–noun order and for verb-final and verb-medial order, up to 67 points, and close to zero for singular marking and the existential verb. Where the target differs from the pivot, the pivot’s value is given in 44% of Gemma 4’s word-order answers without a pivot side and 65% with it, 38% of the way from no pivot effect to always copying; on morphology the share barely moves (43% to 47%).

Overall effect of each pivot. Because each pivot helps where the target shares its value and hurts where it does not, its overall effect depends on how often the targets happen to share its values. Verbmedial pivots, whose word order matches most targets, give overall gains of up to 4.8 points (English, Gemma 4), while Turkish, Korean, Hindi and Japanese give losses of 1.8–4.6 points with both models. English owes its advantage to agreement: 62% of items share English’s value, against 50% for Japanese.

What explains the pivot’s effect. Fixed effects for each target and feature explain 63% of the variance in item accuracy for Gemma 4 and 73% for Qwen3.6, against a ceiling of 83% and 86% set by sampling noise over the three corpus seeds. Of the remaining variance that depends on the pivot and is not noise, the identity of the pivot explains 2%, the syntactic distance between pivot and target (lang2vec (Littell et al., 2017)) 8%, whether the pivot shares the target’s value 27–29%, and that agreement with a separate effect per feature 54–57%. Once syntactic distance is controlled, the verb-final penalty above shrinks to 1.1–1.5 points and is no longer significant. General similarity adds nothing beyond agreement on the tested features: regressed on the 1,899 (target, pivot) pairs, agreement on the remaining Grambank features predicts the gain on its own (Gemma 4: β=0.29, t=8.3; Qwen3.6: β=0.18, t=5.5), but not once agreement on the tested features is in the model (β=−0.01, t=−0.3; β=−0.07, t=−2.2), which remains strong (t=19.9 and 17.4).

Choosing the pivot. Choosing the syntactically closest pivot for each target would raise accuracy from 64.2% to 68.7% for Gemma 4 and from 62.1% to 67.2% for Qwen3.6 over English, as much as the best pivot chosen on held-out seeds. Closeness is measured on the target’s documented grammar, which is what the task asks for, so this is not a usable strategy but another sign that the gain comes from agreement.

<table><tr><td></td><td colspan="2">Gemma 4</td><td colspan="2">Qwen3.6</td></tr><tr><td></td><td>Unid.</td><td>Others</td><td>Unid.</td><td>Others</td></tr><tr><td>MAJORITY</td><td>.676</td><td>.603</td><td>.676</td><td>.603</td></tr><tr><td>PARALLEL</td><td>.695</td><td>.798</td><td>.701</td><td>.775</td></tr><tr><td>CIPHER-PARALLEL</td><td>.614</td><td>.646</td><td>.626</td><td>.630</td></tr></table>

Table 7: Accuracy on the Latin-script core items at n=128, mean over three corpus seeds, split by whether the language is unidentified, i.e. belongs to the extended set (15 languages, 318 items), or not (112 languages, 2,339 items).

## G Cost of the Cipher

The cipher is meant to remove recall, but it also changes how the text is tokenized and removes any cue carried by word forms. To separate the two, Table 7 splits the Latin-script core items by whether the language is one that none of the models can identify (the 15 Latin-script languages of the extended set). On these languages there is little to recall, yet enciphering lowers accuracy by 8.1 points for Gemma 4 and 7.5 for Qwen3.6, against 15.2 and 14.5 on the other languages. Roughly half of the drop under the cipher therefore does not come from lost recall. Plaintext word forms may still reveal a family or region, or enciphered text may simply be harder to process; our data do not separate the two. Either way, the cost is shared by all enciphered settings, so the comparisons in §4.3 and §4.4, which are all between enciphered settings, are not affected. On these 15 languages, plaintext accuracy is only 2 points above MAJORITY, in line with the extended set (§4.5).

<table><tr><td>Language</td><td>ISO</td><td>Glottocode</td><td>Family</td><td>Script</td><td>Src.</td><td>Core</td><td>L E</td></tr><tr><td>Achinese</td><td>ace</td><td>achi1257</td><td>Austronesian</td><td>Latn</td><td>F</td><td>22</td><td>√</td></tr><tr><td>Afar</td><td>aar</td><td>afar1241</td><td>Afro-Asiatic</td><td>Latn</td><td>B</td><td>21</td><td>√</td></tr><tr><td>Aguaruna</td><td>agr</td><td>agua1253</td><td>Chicham</td><td>Latn</td><td>B</td><td>20</td><td>√</td></tr><tr><td>Amharic</td><td>amh</td><td>amha1245</td><td>Afro-Asiatic</td><td>Ethi</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Amis</td><td>ami</td><td>amis1246</td><td>Austronesian</td><td>Latn</td><td>B</td><td>22</td><td>√</td></tr><tr><td>Arhuaco</td><td>arh</td><td>arhu1242</td><td>Chibchan</td><td>Latn</td><td>B</td><td>22</td><td>V V</td></tr><tr><td>Armenian</td><td>hye</td><td>nucl1235</td><td>Indo-European</td><td>Armn</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Awadhi</td><td>awa</td><td>awad1243</td><td>Indo-European</td><td>Deva</td><td>F</td><td>20</td><td></td></tr><tr><td>Ayacucho Quechua</td><td>quy</td><td>ayac1239</td><td>Quechuan</td><td>Latn</td><td>F</td><td>21</td><td></td></tr><tr><td>Bafia</td><td>ksf</td><td>bafi1243</td><td>Atlantic-Congo</td><td>Latn</td><td>B</td><td>20</td><td>√</td></tr><tr><td>Balinese</td><td>ban</td><td>bali1278</td><td>Austronesian</td><td>Latn</td><td>F</td><td>22</td><td></td></tr><tr><td>Bambara</td><td>bam</td><td>bamb1269</td><td>Mande</td><td>Latn</td><td>F+B</td><td>18</td><td>√</td></tr><tr><td>Bashkir</td><td>bak</td><td>bash1264</td><td>Turkic</td><td>Cyrl</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Basque</td><td>eus</td><td>basq1248</td><td>Isolate</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Belarusian</td><td>bel</td><td>bela1254</td><td>Indo-European</td><td>Cyrl</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Bemba (Zambia)</td><td>bem</td><td>bemb1257</td><td>Atlantic-Congo</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Bhojpuri</td><td>bho</td><td>bhoj1244</td><td>Indo-European</td><td>Deva</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Bodo (India)</td><td>brx</td><td>bodo1269</td><td>Sino-Tibetan</td><td>Deva</td><td>F+B</td><td>20</td><td></td></tr><tr><td>Borana-Arsi-Guji Oromo</td><td>gax</td><td>bora1271</td><td>Afro-Asiatic</td><td>Latn</td><td></td><td>18</td><td>√</td></tr><tr><td>Bosnian</td><td>bos</td><td>sout1528</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Brahui</td><td>brh</td><td>brah1256</td><td>Dravidian</td><td>Arab</td><td></td><td>22</td><td></td></tr><tr><td>Breton</td><td>bre</td><td>bret1244</td><td>Indo-European</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Buginese</td><td>bug</td><td>bugi1244</td><td>Austronesian</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Burmese</td><td>mya</td><td>nucl1310</td><td>Sino-Tibetan</td><td>Mymr</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Burushaski</td><td>bsk</td><td>buru1296</td><td>Isolate</td><td>Arab</td><td></td><td>22</td><td></td></tr><tr><td>Campidanese Sardinian</td><td>sro</td><td>camp1261</td><td>Indo-European</td><td>Latn</td><td></td><td>21</td><td>√</td></tr><tr><td>Catalan</td><td>cat</td><td>stan1289</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td>L</td></tr><tr><td>Cebuano</td><td>ceb</td><td>cebu1242</td><td>Austronesian</td><td>Latn</td><td>F+B</td><td>18</td><td>√</td></tr><tr><td>Central Aymara</td><td>ayr</td><td>cent2142</td><td>Aymaran</td><td>Latn</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Central Kanuri</td><td>knc</td><td>cent2050</td><td>Saharan</td><td>Arab</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Central Kurdish</td><td>ckb</td><td>cent1972</td><td>Indo-European</td><td>Arab</td><td>F+B</td><td>1</td><td></td></tr><tr><td>Chechen</td><td>che</td><td>chec1245</td><td>Nakh-Daghestanian</td><td>Cyrl</td><td></td><td>22</td><td></td></tr><tr><td>Cherokee</td><td>chr</td><td>cher1273</td><td>Iroquoian</td><td>Cher</td><td></td><td>22</td><td></td></tr><tr><td>Chuvash</td><td>chv</td><td>chuv1255</td><td>Turkic</td><td>Cyrl</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Crimean Tatar</td><td>crh</td><td>crim1257</td><td>Turkic</td><td>Latn</td><td></td><td>22</td><td></td></tr><tr><td>Croatian</td><td>hrv</td><td>sout1528</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td>L</td></tr><tr><td>Cusco Quechua</td><td>quz</td><td>cusc1236</td><td>Quechuan</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Czech</td><td>ces</td><td>czec1258</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Dandami Maria</td><td>daq</td><td>dand1238</td><td>Dravidian</td><td>Deva</td><td></td><td>22</td><td></td></tr><tr><td>Danish</td><td>dan</td><td>dani1285</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Dar Daju Daju</td><td>djc</td><td>dard1243</td><td>Dajuic</td><td>Latn</td><td></td><td>12</td><td>√</td></tr><tr><td>Dargwa</td><td>dar</td><td>darg1241</td><td>Nakh-Daghestanian</td><td>Cyrl</td><td></td><td>19</td><td></td></tr><tr><td>Duala</td><td>dua</td><td>dual1243</td><td>Atlantic-Congo</td><td>Latn</td><td></td><td>20</td><td>√</td></tr><tr><td>Dutch</td><td>nld</td><td>dutc1256</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td>V</td></tr><tr><td>Dyula</td><td>dyu</td><td>dyul1238</td><td>Mande</td><td>Latn</td><td></td><td>20</td><td>√</td></tr><tr><td>Dzongkha</td><td>dzo</td><td>dzon1239</td><td>Sino-Tibetan</td><td>Tibt</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Eastern Mari</td><td>mhr</td><td>gras1239</td><td>Uralic</td><td>Cyrl</td><td></td><td>22</td><td></td></tr><tr><td>Egyptian Arabic</td><td>arz</td><td>egyp1253</td><td>Afro-Asiatic</td><td>Arab</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Erzya Eton (Cameroon)</td><td>myv eto</td><td>erzy1239 eton1253</td><td>Uralic Atlantic-Congo</td><td>Cyrl Latn</td><td></td><td>22 21</td></tr><tr><td>Language</td><td>ISO</td><td>Glottocode</td><td>Family</td><td>Script Src.</td><td>Core</td><td></td><td>L E</td></tr><tr><td>Hassaniyya</td><td>mey</td><td>hass1238</td><td>Afro-Asiatic</td><td>Latn B</td><td></td><td>22</td><td></td></tr><tr><td>Hausa</td><td>hau</td><td>haus1257</td><td>Afro-Asiatic</td><td>Latn</td><td>F+B</td><td>19</td><td></td></tr><tr><td>Hebrew</td><td>heb</td><td>hebr1245</td><td>Afro-Asiatic</td><td>Hebr</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Hehe</td><td>heh</td><td>hehe1240</td><td>Atlantic-Congo</td><td>Latn</td><td></td><td>21</td><td>√</td></tr><tr><td>Highland Puebla Nahuatl</td><td>azz</td><td>high1278</td><td>Uto-Aztecan</td><td>Latn</td><td></td><td>14</td><td></td></tr><tr><td>Hindi</td><td>hin</td><td>hind1269</td><td>Indo-European</td><td>Deva</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Hungarian</td><td>hun</td><td>hung1274</td><td>Uralic</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Icelandic</td><td>isl</td><td>icel1247</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Igbo</td><td>ibo</td><td>nucl1417</td><td>Atlantic-Congo</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Iloko</td><td>ilo</td><td>ilok1237</td><td>Austronesian</td><td>Latn</td><td>F+B</td><td>20</td><td></td></tr><tr><td>Indonesian</td><td>ind</td><td>indo1316</td><td>Austronesian</td><td>Latn</td><td>F+B</td><td>19</td><td></td></tr><tr><td>Iranian Persian</td><td>pes</td><td>west2369</td><td>Indo-European</td><td>Arab</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Iraqw</td><td>irk</td><td>iraq1241</td><td>Afro-Asiatic</td><td>Latn</td><td></td><td>21</td><td></td></tr><tr><td>Irish</td><td>gle</td><td>iris1253</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td></td><td>√</td></tr><tr><td>Italian</td><td>ita</td><td>ital1282</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Japanese</td><td>jpn</td><td>nucl1643</td><td>Japonic</td><td>Jpan</td><td></td><td>22</td><td>√</td></tr><tr><td>Javanese</td><td></td><td>java1254</td><td>Austronesian</td><td>Latn</td><td>F+B F+B</td><td>22</td><td>√</td></tr><tr><td></td><td>jav quc</td><td>kich1262</td><td>Mayan</td><td></td><td></td><td>20</td><td></td></tr><tr><td>K'iche'</td><td>kab</td><td>kaby1243</td><td>Afro-Asiatic</td><td>Latn Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Kabyle</td><td>kac</td><td>kach1280</td><td>Sino-Tibetan</td><td>Latn</td><td></td><td>10</td><td>√</td></tr><tr><td>Kachin</td><td>kal</td><td>kala1399</td><td>Eskimo-Aleut</td><td>Latn</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Kalaallisut</td><td>kam</td><td>kamb1297</td><td>Atlantic-Congo</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Kamba (Kenya)</td><td>kan</td><td>nucl1305</td><td>Dravidian</td><td>Knda</td><td>F+B</td><td>21</td><td>√</td></tr><tr><td>Kannada</td><td>cak</td><td>kaqc1270</td><td>Mayan</td><td>Latn</td><td>F+B</td><td>22 22</td><td></td></tr><tr><td>Kaqchikel Kashmiri</td><td>kas</td><td>kash1277</td><td>Indo-European</td><td>Arab</td><td></td><td></td><td>√</td></tr><tr><td></td><td>kjh</td><td>khak1248</td><td>Turkic</td><td>Cyrl</td><td></td><td>22</td><td></td></tr><tr><td>Khakas Khmer</td><td>khm</td><td>cent1989</td><td>Austroasiatic</td><td>Khmr</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Khoekhoe</td><td>naq</td><td>nama1264</td><td>Khoe-Kwadi</td><td>Latn</td><td></td><td>22 18</td><td></td></tr><tr><td>Kikuyu</td><td>kik</td><td>kiku1240</td><td>Atlantic-Congo</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Kipsigis</td><td>sgc</td><td>kips1239</td><td>Nilotic</td><td>Latn</td><td></td><td>22</td><td>r</td></tr><tr><td>Korean</td><td>kor</td><td>kore1280</td><td>Koreanic</td><td>Kore</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Koyra Chiini Songhay</td><td>khq</td><td>koyr1240</td><td>Songhay</td><td>Latn</td><td></td><td>22</td><td></td></tr><tr><td>Kurukh</td><td>kru</td><td>kuru1302</td><td>Dravidian</td><td>Deva</td><td></td><td>18</td><td></td></tr><tr><td>Kxoe</td><td>xuu</td><td>kxoe1243</td><td>Khoe-Kwadi</td><td>Latn</td><td></td><td>22</td><td></td></tr><tr><td>Lao</td><td>lao</td><td>laoo1244</td><td>Tai-Kadai</td><td>Laoo</td><td>F+B</td><td>19</td><td></td></tr><tr><td>Lendu</td><td>led</td><td>lend1245</td><td>Central Sudanic</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Levantine Arabic</td><td>apc</td><td>nort3139</td><td>Afro-Asiatic</td><td>Arab</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Lingala</td><td>lin</td><td>ling1263</td><td>Atlantic-Congo</td><td>Latn</td><td>F+B</td><td>17</td><td>L</td></tr><tr><td>Lithuanian</td><td>lit</td><td>lith1251</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Lombard</td><td>lmo</td><td>lomb1257</td><td>Indo-European</td><td>Latn</td><td></td><td>22</td><td>√ √</td></tr><tr><td>Lugbara</td><td>lgg</td><td>lugb1240</td><td>Central Sudanic</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Luo (Kenya and Tanzania)</td><td>luo</td><td>luok1236</td><td>Nilotic</td><td>Latn</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Lushai</td><td>lus</td><td>lush1249</td><td>Sino-Tibetan</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Macedonian</td><td>mkd</td><td>mace1250</td><td>Indo-European</td><td>Cyrl</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Magahi</td><td>mag</td><td>maga1260</td><td>Indo-European</td><td>Deva</td><td></td><td>20</td><td></td></tr><tr><td>Mai Brat</td><td>ayz</td><td>maib1239</td><td>Isolate</td><td>Latn</td><td></td><td>22</td><td>√</td></tr><tr><td>Maithili</td><td>mai</td><td>mait1250</td><td>Indo-European</td><td>Deva</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Makhuwa</td><td>vmw</td><td>makh1264</td><td>Atlantic-Congo</td><td>Latn</td><td>F+B</td><td>21</td><td>√</td></tr><tr><td>Malayalam Maltese</td><td>mal mlt</td><td>mala1464 malt1254</td><td>Dravidian Afro-Asiatic</td><td>Mlym</td><td>F+B</td><td>22 22</td></tr><tr><td>Language</td><td>ISO</td><td>Glottocode</td><td>Family</td><td>Script Src.</td><td></td><td>Core</td><td>L E</td></tr><tr><td>Nuer</td><td>nus</td><td>nuer1246</td><td>Nilotic</td><td>Latn F+B</td><td></td><td>20 √</td><td>V</td></tr><tr><td>Occitan (post 1500)</td><td>oci</td><td>occi1239</td><td>Indo-European</td><td>Latn F</td><td></td><td>22 √</td><td></td></tr><tr><td>Odia</td><td>ory</td><td>oriy1255</td><td>Indo-European</td><td>Orya</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Pangasinan</td><td>pag</td><td>pang1290</td><td>Austronesian</td><td>Latn</td><td></td><td>22</td><td></td></tr><tr><td>Panjabi</td><td>pan</td><td>panj1256</td><td>Indo-European</td><td>Guru</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Paraguayan Guaraní</td><td>gug</td><td>para1311</td><td>Tupian</td><td>Latn</td><td>F+B</td><td>10</td><td></td></tr><tr><td>Plains Cree</td><td>crk</td><td>plai1258</td><td>Algic</td><td>Latn</td><td></td><td>19</td><td></td></tr><tr><td>Plateau Malagasy</td><td></td><td>plat1254</td><td>Austronesian</td><td>Latn</td><td></td><td>19</td><td></td></tr><tr><td>Polish</td><td>plt pol</td><td>poli1260</td><td>Indo-European</td><td>Latn</td><td>F+B F+B</td><td>22</td><td></td></tr><tr><td>Portuguese</td><td>por</td><td>port1283</td><td>Indo-European</td><td>Latn</td><td></td><td>√ √</td><td></td></tr><tr><td>Pulaar</td><td></td><td>pula1263</td><td>Atlantic-Congo</td><td></td><td></td><td>22 √</td><td></td></tr><tr><td></td><td>fuc</td><td>pure1242</td><td></td><td>Latn</td><td></td><td>19 √</td><td></td></tr><tr><td>Purepecha Romansh</td><td>tsz</td><td></td><td>Tarascan</td><td>Latn Latn</td><td></td><td>22</td><td></td></tr><tr><td>Russian</td><td>roh</td><td>roma1326</td><td>Indo-European</td><td></td><td></td><td>21</td><td></td></tr><tr><td></td><td>rus smo</td><td>russ1263</td><td>Indo-European</td><td>Cyrl</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Samoan</td><td></td><td>samo1305</td><td>Austronesian Huavean</td><td>Latn</td><td></td><td>22</td><td></td></tr><tr><td>San Dionisio Del Mar Huave</td><td>hve</td><td>sand1278 sant1410</td><td>Austroasiatic</td><td>Latn</td><td></td><td>22</td><td></td></tr><tr><td>Santali</td><td>sat</td><td></td><td></td><td>Olck</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Serbian</td><td>srp</td><td>sout1528</td><td>Indo-European</td><td>Cyrl</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Shona</td><td>sna</td><td>shon1251</td><td>Atlantic-Congo</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Siamou</td><td>sif</td><td>siam1242</td><td>Isolate</td><td>Latn</td><td></td><td>19</td><td></td></tr><tr><td>Sicilian</td><td>scn</td><td>sici1248</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Slovenian</td><td>slv</td><td>slov1268</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Somali</td><td>som</td><td>soma1255</td><td>Afro-Asiatic Indo-European</td><td>Latn Arab</td><td>F+B</td><td>20</td><td></td></tr><tr><td>Southern Pashto</td><td>pbt sot</td><td>sout2649 sout2807</td><td></td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Southern Sotho</td><td></td><td></td><td>Atlantic-Congo Afro-Asiatic</td><td></td><td>F+B</td><td>22</td><td></td></tr><tr><td>Standard Arabic</td><td>arb ekk</td><td>stan1318</td><td>Uralic</td><td>Arab</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Standard Estonian</td><td>lvs</td><td>esto1258 latv1249</td><td>Indo-European</td><td>Latn Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Standard Latvian</td><td>zsm</td><td>stan1306</td><td>Austronesian</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Standard Malay</td><td>sun</td><td>sund1252</td><td>Austronesian</td><td>Latn</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Sundanese Swahili</td><td>swh</td><td>swah1253</td><td>Atlantic-Congo</td><td>Latn</td><td>F+B</td><td>19</td><td></td></tr><tr><td>Swedish</td><td>swe</td><td>swed1254</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>19</td><td></td></tr><tr><td>Tagalog</td><td>tgl</td><td>taga1270</td><td>Austronesian</td><td>Latn</td><td></td><td>20</td><td></td></tr><tr><td>Tamasheq</td><td>taq</td><td>tama1365</td><td>Afro-Asiatic</td><td>Latn</td><td>F+B</td><td>20</td><td></td></tr><tr><td>Tamil</td><td>tam</td><td>tami1289</td><td>Dravidian</td><td>Taml</td><td>F+B</td><td>22 22</td><td></td></tr><tr><td>Tatar</td><td>tat</td><td>tata1255</td><td>Turkic</td><td>Cyrl</td><td>F+B</td><td>21</td><td></td></tr><tr><td>Tausug</td><td>tsg</td><td>taus1251</td><td>Austronesian</td><td>Latn</td><td></td><td>20</td><td></td></tr><tr><td>Telugu</td><td>tel</td><td>telu1262</td><td>Dravidian</td><td>Telu</td><td>F+B</td><td>20</td><td></td></tr><tr><td>Thai</td><td>tha</td><td>thai1261</td><td>Tai-Kadai</td><td>Thai</td><td>F+B</td><td>20</td><td></td></tr><tr><td>Tibetan</td><td>bod</td><td>tibe1272</td><td>Sino-Tibetan</td><td>Tibt</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Tok Pisin</td><td>tpi</td><td>tokp1240</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>20</td><td></td></tr><tr><td>Tosk Albanian</td><td>als</td><td>tosk1239</td><td>Indo-European</td><td>Latn</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Tsonga</td><td>tso</td><td>tson1249</td><td>Atlantic-Congo</td><td>Latn</td><td>F+B</td><td>20</td><td>√ √</td></tr><tr><td>Tswana</td><td>tsn</td><td>tswa1253</td><td>Atlantic-Congo</td><td>Latn</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Turkish</td><td>tur</td><td>nucl1301</td><td>Turkic</td><td>Latn</td><td>F+B</td><td>22</td><td>√</td></tr><tr><td>Turkmen</td><td>tuk</td><td>turk1304</td><td>Turkic</td><td>Latn</td><td></td><td>22</td><td></td></tr><tr><td>Tuvinian</td><td>tyv</td><td>tuvi1240</td><td>Turkic</td><td>Cyrl</td><td></td><td>19</td><td>√</td></tr><tr><td>Ukrainian</td><td>ukr</td><td>ukra1253</td><td>Indo-European</td><td>Cyrl</td><td>F+B</td><td>22</td><td></td></tr><tr><td>Umbundu</td><td>umb vie</td><td>umbu1257 viet1252</td><td>Atlantic-Congo Austroasiatic</td><td>Latn</td><td>F+B</td><td>22</td></tr></table>

Table 8: All 195 target languages, with family from Grambank. Src. is the source corpus (F: FLORES+, B: BOUQuET); Core is the number of core-set items; L and E mark the Latin-script and extended sets.

<table><tr><td>Group</td><td>ID</td><td>Grambank question</td><td>Values</td><td>Items</td><td>Gold</td></tr><tr><td>Word order</td><td>GB024</td><td>What is the order of numeral and noun in the NP?</td><td>[1] Num–N; [2] N–Num; [3] both</td><td>186</td><td>117/49/20</td></tr><tr><td></td><td>GB025</td><td>What is the order of adnominal demonstrative and noun?</td><td>[1] Dem–N; [2] N-Dem; [3] both</td><td>186</td><td>102/58/26</td></tr><tr><td></td><td>GB065</td><td>What is the pragmatically unmarked order of adnominal possessor noun and possessed noun?</td><td>[1] Possessor-Possessed; [2] Possessed–Possessor; [3] both</td><td>190</td><td>75/75/40</td></tr><tr><td></td><td>GB130</td><td>What is the pragmatically unmarked order of S and V in intransitive clauses? Is a pragmatically unmarked constituent</td><td>[1] SV; [2] VS; [3] both</td><td>194</td><td>157/26/11</td></tr><tr><td></td><td>GB131</td><td>order verb-initial for transitive clauses? Is a pragmatically unmarked constituent</td><td>[0] absent; [1] present</td><td>191</td><td>159/32</td></tr><tr><td></td><td>GB132</td><td>order verb-medial for transitive clauses? Is a pragmatically unmarked constituent</td><td>[0] absent; [1] present</td><td>191</td><td>90/101</td></tr><tr><td></td><td>GB133</td><td>order verb-final for transitive clauses? What is the order of adnominal property</td><td>[0] absent; [1] present</td><td>191</td><td>119/72</td></tr><tr><td></td><td>GB193</td><td>word and noun?</td><td>[0] they cannot be used attributively; [1] ANM–N; [2] N–ANM; [3] both</td><td>181</td><td>0/81/75/25</td></tr><tr><td>Word classes</td><td>GB020</td><td>Are there definite or specific articles?</td><td>[0] absent; [1] present</td><td>186</td><td>103/83</td></tr><tr><td></td><td>GB038</td><td>Are there demonstrative classifiers?</td><td>[0] absent; [1] present</td><td>186</td><td>180/6</td></tr><tr><td></td><td>GB074</td><td>Are there prepositions?</td><td>[0] absent; [1] present</td><td>184</td><td>61/123</td></tr><tr><td></td><td>GB075</td><td>Are there postpositions?</td><td>[0] absent; [1] present</td><td>181</td><td>95/86</td></tr><tr><td></td><td>GB126</td><td>Is there an existential verb?</td><td>[0] absent; [1] present</td><td>162</td><td>56/106</td></tr><tr><td>Morphology</td><td>GB030</td><td>Is there a gender distinction in independent</td><td>[0] absent; [1] present</td><td>193</td><td>117/76</td></tr><tr><td></td><td>GB042</td><td>3rd person pronouns? Is there productive overt morphological</td><td>[0] absent; [1] present</td><td>191</td><td>130/61</td></tr><tr><td></td><td>GB044</td><td>singular marking on nouns? Is there productive morphological plural</td><td>[0] absent; [1] present</td><td>190</td><td>44/146</td></tr><tr><td></td><td>GB082</td><td>marking on nouns? Is there overt morphological marking of</td><td>[0] absent; [1] present</td><td>186</td><td>88/98</td></tr><tr><td></td><td>GB083</td><td>present tense on verbs? Is there overt morphological marking on the</td><td>[0] absent; [1] present</td><td>190</td><td>49/141</td></tr><tr><td></td><td>GB084</td><td>verb dedicated to past tense? Is there overt morphological marking on the</td><td>[0] absent; [1] present</td><td>187</td><td>82/105</td></tr><tr><td></td><td>GB137</td><td>verb dedicated to future tense? Can standard negation be marked</td><td>[0] absent; [1] present</td><td>180</td><td>124/56</td></tr><tr><td></td><td>GB138</td><td>clause-finally? Can standard negation be marked</td><td>[0] absent; [1] present</td><td>179</td><td>106/73</td></tr><tr><td></td><td>GB172</td><td>clause-initially? Can an article agree with the noun in</td><td>[0] absent; [1] present</td><td>179</td><td>147/32</td></tr></table>

Table 11: The 22 core features. Items is the number of the 195 languages Grambank codes for the feature; Gold is how many of them have each value, in the order listed.