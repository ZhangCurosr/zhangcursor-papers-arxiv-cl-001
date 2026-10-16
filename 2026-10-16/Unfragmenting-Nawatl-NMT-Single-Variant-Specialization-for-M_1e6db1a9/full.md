# Unfragmenting Nawatl NMT: Single-Variant Specialization for Masewalkopa LoRA Adaptation, Morphosyntactic Conditioning, and Decolonial Evaluation

Luis Armando Mercado-Campos<sup>⋆</sup> and Alexis Palmer

Department of Linguistics, University of Colorado Boulder Luis.MercadoCampos@colorado.edu, alexis.palmer@colorado.edu

Abstract. In minoritized (low-resource) language technologies, a common response to data sparsity is to aggregate text across multiple varieties and treat the combined data as a single corpus. However, mixed corpora of closely related indigenous variants can produce an artificial register that is accountable to no single speaker community, a structural failure we call variant fragmentation. This issue is acute for minoritized languages native to Mexico, such as Nawatl. Building on a 70,000-pair multi-variant foundation [18], we specialize an mT5-large checkpoint to Masewalkopa (nhn), the Nawatl variety spoken in Santa Ana Tlacotenco. We curate 2,665 parallel pairs, reserve all sentence-level data for evaluation, and incorporate 192 elder-transcribed target sentences via asymmetric backtranslation. A custom finite-state analyzer reduces unanalyzed surface forms from 82.0% to 19.9% and supplies functional syntax tags as plain-text prefixes, allowing the model to reuse existing subword embeddings rather than relying on frozen special tokens. The resulting lora\_v9c\_b model achieves 58.46 dev chrF++ and 58.20/57.90 directional test chrF++. Finally, CTranslate2 int8 quantization reduces CPU latency from 137.5 to 1.8–2.6 seconds per request, enabling cost-bounded Cloud Run deployment.

Keywords: Neural Machine Translation · Minoritized NLP · Nawatl · Masewalkopa · LoRA · Finite-State Morphology · Decolonial Evaluation.

## 1 Introduction and Linguistic Background

Nawatl is a group of endangered, closely-related, morphologically rich languages<sup>1</sup> in the Nawan branch of the Uto-Aztecan family, spoken primarily in Mexico. There

<sup>1</sup> I use “language,” “variant,” and “variety” interchangeably, but avoid the use of the term “dialect.” In Mexico, this distinction carries real social, educational, and political consequences, as historically the term “dialect” has been used as a tool to marginalize indigenous language speakers [1]. In government documentation, schooling, and public discourse, referring to these languages as “dialects” can contribute to marginalization and under-resourcing [21].

are 30 formally recognized variants of Nawatl [12], which can have substantial phonological, morphological, syntactic, and lexical diferences [9]. There is also significant variant-internal diversity. Treating them as a single computational language obscures the systems that index place and community belonging.

Masewalkopa is the endonym used by speakers of the Nawatl in Santa Ana de Tlacotenco, Milpa Alta, Mexico City (nhn). It is morphologically rich and predominantly head marking: words can encode subject, object, derivation, aspect, possession, predicate-nominal structure, and honorific morphology. Characterlevel similarity can therefore conceal a wrong possessor or a form from another writing tradition.

During the first author’s three years of studying Masewalkopa, he encountered persistent discrepancies between locally taught forms and Classical Nawatlcentered digital resources. As a result, his objective shifted from maximizing a generic “Nawatl” score to producing forms that can be inspected by a single, identifiable community.

## 1.1 From a Multi-Variant Model to Single-Variant Specialization

In the first author’s previous work [18], he assembled approximately 70,000 Spanish-Nawatl pairs across varieties and domains. The resulting system (found on huggingface<sup>2</sup>) checkpoint achieved 54.15 chrF++ on a split from that same mixed collection.

However, when evaluated zero-shot on an external benchmark, the AmericasNLP 2021 Shared Task on Open Machine Translation for Indigenous Languages [17], performance collapsed to 25.7 chrF++. This external evaluation dataset consisted of independent Spanish-Nawatl text that was completely unseen during training. Beyond simple domain mismatch, manual inspection revealed that the model generated conflicting lexical, orthographic, and inflectional forms from diferent regions within the same sentence. While each piece might be attested somewhere in Mexico, the combined output was incoherent as Masewalkopa.

We use variant fragmentation for this interaction between data aggregation and model behavior. The multi-variant corpus expresses a single underlying meaning through divergent regional realizations, the subword model shares pieces regardless of regional provenance, and beam search can assemble a fluent-looking sequence from conflicting varietal conventions. Aggregate evaluation conceals the problem when references come from the same mixture. A high in-domain score can therefore coexist with low external validity.

To resolve this, we warm-start from the first author’s 1.2-billion-parameter multi-variant mT5-large thesis checkpoint and adapt it exclusively with Masewalkopa data. The broad-base model provides sequence modeling and cross-lingual alignment, while the small community corpus provides regional constraints. Low-Rank Adaptation (LoRA) makes this transfer feasible on a single GPU via compact, low-rank updates [11].

In this paper, we develop a single-variety approach to neural machine translation (NMT) for Masewalkopa, using a relatively small community dataset to specialize an existing multi-variety model. By combining parameter-eficient LoRA fine-tuning with rule-based finite-state morphology and authentic target-side oral narratives, we show that targeted specialization resolves variant fragmentation while respecting the linguistic autonomy of Santa Ana Tlacotenco speech.

## 1.2 Core Contributions

This paper makes four contributions:

1. Single-Variant Specialization Pipeline: We specialize a 1.2-billion-parameter NMT model with 2,665 Masewalkopa pairs, restricting short dictionary entries to training so that development and test benchmarks evaluate only full sentence translation.

2. Morphosyntactic NMT Conditioning: We expand a finite-state analyzer beyond 6,500 lexicon lines to derive surface-validated functional syntax prefixes, supplying the neural translation encoder with structural grounding to resolve morphological and lexical ambiguities.

3. Ablation Suite and Post-Mortems: We ablate asymmetric backtranslation, cold special tokens, sparse versus dense syntax, and syntax stripping in the final lora\_v9c experiment.

4. Decolonial and Production Evaluation: We combine chrF++, linguistic diagnostics, DBS, human comparison, and measured CPU deployment behavior.

Together, these experiments demonstrate that targeted single-variant specialization overcomes the pitfalls of multi-variant data aggregation. The final specialized model (lora\_v9c\_b) achieves 58.46 dev chrF++ and 58.20/57.90 directional test chrF++, gaining over 28 points over the mixed-variant base. Ablation results show that plain-text syntactic prefixes successfully recover from the severe optimization stalls of frozen special tokens, while dense morphological tagging yields measurable structural gains (+1.33 test chrF++). Culturally, decolonial DBS probes and human A/B ratings confirm a strong reduction in colonial Spanish substitutions and improved naturalness on elder-transcribed oral narratives. Finally, 8-bit quantization and widened beam search reduce inference latency to under 2.6 seconds on standard CPU hardware, demonstrating that single-variant models can be operated within strict community budget constraints.

## 2 Related Work

Indigenous Machine Translation and Variety Specialization. Prior machine translation research on Indigenous languages of the Americas has largely focused on pooling heterogeneous regional variants into aggregate training sets to combat extreme data scarcity [7]. However, computational dialectology and variety classification for Nawatl demonstrates that regional variants exhibit sharp lexical and morphophonological boundaries [19]. Blending these divergent systems into monolithic training corpora frequently produces varietal bleed and privileges colonial or standardized varieties over living community speech [3]. Recent work therefore emphasizes community-grounded single-variant modeling. This study builds on that foundation by specializing a broad multi-variant base directly to the spoken variety of Santa Ana Tlacotenco.

Morphosyntactic Conditioning in Neural MT. Guiding sequence generation with explicit linguistic structure stems from factored translation architectures [25] and prefix-based syntactic conditioning [28]. For polysynthetic languages, finite-state transducers (FSTs) supply robust morphological segmentation and grammatical disambiguation where purely statistical subword models fail [2]. My system builds on these insights, demonstrating that high-density FST syntactic prefixes provide reliable structural inductive bias without requiring vocabulary modifications.

Parameter-Eficient Adaptation in Low-Resource Regimes. Low-Rank Adaptation (LoRA) [11] provides an eficient mechanism for adapting massive multilingual models such as mT5 [27] while preventing catastrophic forgetting. Extending lowrank updates across all linear projections is essential for deep domain transfer [6, 10]. We contribute an empirical analysis of embedding constraints under LoRA, showing how plain-text subword formatting avoids the optimization collapse associated with frozen cold-token representations.

## 3 Data Curation, Corpus Composition, and Governance

We compiled the primary parallel corpus over nearly four years of studying Masewalkopa through community classes and structured coursework with nativespeaker language instructors from Santa Ana Tlacotenco. All pedagogical exercises, classroom dialogues, and vocabulary transcriptions were recorded, digitized, and curated with the explicit informed consent of the instructors for research and educational work. Following the Collective Benefit, Authority to COntrol, Responsibility, and Ethics (CARE) principles, we treat provenance, authority, responsibility, and ethics as pipeline properties [4]. Data collection was guided by collective benefit and mutual accountability rather than extractive aggregation.

## 3.1 Cleaning and Provenance

In machine translation, the main training unit is an aligned parallel pair consisting of a source text segment and its equivalent target translation. We compiled the primary corpus over nearly four years of studying Masewalkopa through classes and coursework with native-speaker instructors from Santa Ana Tlacotenco. All pedagogical exercises, classroom dialogues, and vocabulary notes were recorded and curated into structured spreadsheets with explicit instructor consent.

We normalized whitespace and Unicode representations, removed empty fields, and deduplicated aligned Spanish–Nawatl segment pairs case-insensitively.

Near-neighbor spellings were retained because they often encode authentic local phonological choices. Cleaning reduced 2,761 raw rows to 2,665 unique parallel segment pairs.

Each parallel pair was tagged by sentential status based on word count from standard whitespace splitting after punctuation stripping. Entries with at most two whitespace-delimited tokens were treated as lexical or short-phrase material (such as citation nouns or isolated verb stems); longer examples and independently verified clauses were treated as sentence-level. This reproducible threshold does not claim that every two-token sequence lacks syntax. Instead, it prevents dictionary correspondences from dominating evaluation splits.

Table 1. Composition of the cleaned parallel corpus before bidirectional expansion, with representative examples.
<table><tr><td>Subset</td><td></td><td></td><td>Pairs Share Illustrative Examples</td><td>Primary Experimental Role</td></tr><tr><td>Lexical / short (≤ 2 tokens) 1,188</td><td></td><td></td><td>44.6% sesek ↔ frío; agradezco</td><td>Training coverage, citation nimitstlasohkamate ↔ te forms, short-form deploy- ment probes</td></tr><tr><td>Sentence level</td><td>1,477</td><td>55.4% tlamach</td><td>↔ juntos en comunidad source of development and venceremos test items</td><td>tikayawaskeh Training plus exclusive</td></tr><tr><td>Total unique</td><td>2,665</td><td></td><td>100.0%Masewalkopa specialization corpus</td><td></td></tr></table>

Nearly 45% of the pairs are lexical or very short. They teach stems and spelling but underrepresent agreement, possession, negation, and discourse. Random splitting would therefore reward dictionary matches without demonstrating sentence translation.

## 3.2 Stratification and Bidirectional Tasks

Development and test data were drawn exclusively from the 1,477 sentencelevel pairs to ensure that evaluation benchmarks measure genuine sequence translation rather than isolated vocabulary lookup. Including single-word or short lexical pairs in held-out sets would artificially inflate n-gram and character overlap scores through simple dictionary matching, masking failures in clause-level syntax, agreement, and possession. Restricting evaluation to sentential items guarantees that metrics reflect grammatical compositionality, while placing all 1,188 lexical pairs into the training inventory provides necessary stem coverage and orthographic grounding.

Using a fixed stratification seed, we reserved 120 sentence pairs for development and 150 for the final held-out test set. The remaining training inventory contained 2,335 pairs after recombining the lexical entries with eligible training sentences.

To train the model bidirectionally, each parallel sentence pair consisting of a Spanish segment $\left( s _ { \mathrm { e s } } \right)$ and a Masewalkopa segment $\left( s _ { \mathrm { n a h } } \right)$ was expanded into two separate supervised training examples by prepending a literal directional prefix to the source string:

– Spanish → Masewalkopa:

Input string: es->nah: juntos venceremos como comunidad

Target string: tlamach tikayawaskeh

– Masewalkopa → Spanish:

Input string: nah->es: tlamach tikayawaskeh

Target string: juntos venceremos como comunidad

The 2,335 training pairs therefore yielded 4,670 directional training examples, while the 120 development pairs and 150 test pairs yielded 240 and 300 directional evaluation examples, respectively.

Directional prefixes are ordinary text already present in the mT5 Sentence-Piece vocabulary. This detail is crucial because standard text prefixes remain trainable through attention and LoRA weight updates even when the base model’s shared embedding table is completely frozen. Joint bidirectional training doubles the available supervised training instances without requiring synthetic target generation.

## 3.3 Four Community Oral Texts

To add connected speech and local cultural vocabulary, we incorporated 192 eldertranscribed Masewalkopa sentences. These materials were generously shared with me for pedagogical development by one of my native speaker language instructors in Santa Ana Tlacotenco. The texts represent four distinct thematic registers:

1. Tiankisko: 34 market-dialogue sentences covering exchanges, quantities, and polite requests.

2. Tonakayutl: 45 sentences on traditional maize cultivation, harvesting practices, and food knowledge.

3. Kowatl itech in ikwentsin: 50 narrative sentences describing a traditional snake story, adding complex reference tracking and sequential event structures.

4. Tatahpowilistle: 63 sentences concerning community authority, local oral history, and traditional institutions.

Preserving these document titles and categories respects their local genre provenance, even though the current translation model does not condition directly on document identity.

## 3.4 Asymmetric Target-Side Backtranslation

Backtranslation generates synthetic sources from monolingual target data [24]. We used lora\_v2 (45.58 Masewalkopa-to-Spanish test chrF++) to translate the 192 authentic Masewalkopa sentences into Spanish:

$$
\hat { x } _ { \mathrm { e s } } = \arg \operatorname* { m a x } _ { x } P _ { \mathrm { l o r } \mathsf { a } _ { - } \mathsf { v } 2 } ( x \mid y _ { \mathrm { n a h } } ^ { \mathrm { h u m a n } } ) , \qquad ( \hat { x } _ { \mathrm { e s } } , y _ { \mathrm { n a h } } ^ { \mathrm { h u m a n } } ) .\tag{1}
$$

Unlike the curated pairs in Section 3.2, which were expanded bidirectionally, these backtranslated pairs were used in one direction only (nah → es). Because only the Spanish source input $\hat { x } _ { \mathrm { e s } }$ is synthetic, the Masewalkopa generation decoder learns exclusively from authentic, elder-authored targets. We deliberately did not swap source and target to create es → nah training examples, which would have forced the Spanish generation decoder to imitate its earlier output.

The resulting train\_bt.jsonl file contained 4,862 directional examples (4,670 directional instances plus the 192 backtranslated pairs). This targeted addition aims to expose the model to connected narrative discourse from community elders rather than evaluating large-scale synthetic data generation.

## 3.5 Leakage Controls and Evaluation Scope

Deduplication preceded splitting; dev and test pairs were excluded from augmentation. We report chrF++ for its tolerance of inflectional variation [22], but add linguistic diagnostics because overlap cannot judge regional appropriateness, possession, or honorific licensing.

## 4 Finite-State Morphosyntactic Engineering

While neural sequence-to-sequence models excel at cross-lingual transfer, subword tokenizers frequently fracture polysynthetic words into arbitrary pieces that obscure grammatical relations. In Masewalkopa, a single verbal form can encode subject person, object person, reflexive marking, tense, and direction. With only 2,665 curated training pairs, a purely data-driven neural encoder struggles to generalize across sparse morphological paradigms.

To provide the neural translation model with explicit structural grounding, we developed a rule-based finite-state transducer using the Helsinki Finite-State Transducer (HFST) toolkit [16]. Rather than replacing the neural model, this analyzer acts as a linguistic feature extractor. It parses surface tokens into interlinear glossed text (IGT) and functional syntax labels, which are then passed directly to the NMT encoder as plain-text prefixes during training and inference (Subsection 5.4).

The analyzer combines final-state morphotactic continuation lexicons with two-level phonological alternation rules [2] to capture the core grammatical inventory of Santa Ana Tlacotenco speech.

## 4.1 Stem Classes and Morphotactics

An undiferentiated lexicon overgenerates because stems select diferent sufixal paths. We therefore use the continuation classes in Table 2 as implementation abstractions.

Dedicated nominal paths prevent absolutive sufixes from attaching to every nominal-looking base. An audit removed sesek (cold) and tomawak (plump) from noun continuations that generated unattested -tl, -li, -tli, -in, or -itl forms.

Predicate nominals require a separate path: ti-sowatl, “you are a woman,” carries person without an overt copula. Retaining this analysis supports transla tion to Spanish ser constructions.

Table 2. Principal finite-state stem classes and the contrasts they encode.
<table><tr><td>Class</td><td>Function in the analyzer</td><td>Representative constraint</td></tr><tr><td>A-Stem</td><td>Stems whose citation or inflected path is orga- Restricts which aspect and nized around an a-final alternation</td><td>derivational arcs can follow</td></tr><tr><td></td><td>I-Stem i-final verbal or predicate paths</td><td>Preserves class-specific boundary alternations</td></tr><tr><td></td><td>OwA-Stem Productive verbal stems associated with an owa Connects to verbal person pattern</td><td>and aspect networks</td></tr><tr><td></td><td>ITL-Stem Nominal stems selecting an -itl-type absolutive Blocks incompatible absolu- realization</td><td>tive suffixes</td></tr><tr><td></td><td>TLE-Stem Nominal stems selecting a -tle-type path in local Separates local surface be- orthography</td><td>havior from generic noun rules</td></tr><tr><td></td><td>LI-Stem Nominal stems selecting -li</td><td>Supports possessed and predicate-nominal alterna- tions</td></tr></table>

## 4.2 Progressive Aspect Allomorphy

Progressive arcs for -tok and -tika are licensed by stem continuation class, with TWOLC constraining boundary realizations. This generalizes beyond a flat word list, provided the allomorphs are freely attachable.

The analysis identifies heads inside long words and exposes aspect as a stable feature. It supplements rather than replaces neural subword representations.

## 4.3 Coverage Expansion and Error Accounting

Repeated audits grew LEXC beyond 6,500 lines. After each justified stem addition, we recompiled and measured no analysis, permissive passthrough, and genuine segmentation. Passthrough is not counted as successful analysis.

Table 3. Finite-state coverage progression. The first audit used a legacy 2,736-form inventory; subsequent audits used 2,807 unique corpus word forms.
<table><tr><td>Development stage</td><td>Vocabulary No analysis Passthrough Genuine segmentation</td></tr><tr><td>Initial baseline 2,736 82.0% (2,244)</td><td>7.9% (216) 10.1% (276)</td></tr><tr><td>Post aspect/POS audit 2,807</td><td>26.4% (742) 7.4% (207) 66.2% (1,858)</td></tr><tr><td>2,807 25.2% (707)</td><td>7.2% (202) 67.6% (1,898)</td></tr><tr><td>Batch 22–23 expansion Batch 24–26 closure</td><td>2,807 19.9% (559) 7.2% (202) 72.9% (2,046)</td></tr></table>

The remaining 559 forms include Spanish terms, names, spelling variants, rare inflections, and gaps. Because coverage can increase due to incompatible analyses, we added a precision-oriented reconstruction filter.

## 4.4 Three-Stage Reconstruction Filter

Because finite-state transducers can generate multiple competing analyses or derivations for a single surface token, we use a three-stage reconstruction filter to disambiguate raw transducer outputs before assigning IGT:

1. Surface match. Retain only candidates whose reconstructed stem or diagnostic realization aligns to an ordered substring of the normalized token.

2. Possessive guard. Require overt initial no-, mo-, to-, or i- for a possessive analysis. This blocks bare nouns and coincidental stem-initial strings from receiving NP.POSS.

3. Category preference. Prefer the open-class analysis covering most surface material, except when a listed adverb matches the entire token. Preserve genuine ties for sentence-level resolution.

In compact pseudo-code, the decision is:

```julia
for token in sentence:
candidates = fst.lookup(strip_punct(token.lower()))
candidates = keep_surface_aligned(candidates, token)
candidates = guard_possession(candidates, token)
analysis = prefer_category(candidates, token)
emit(igt(analysis), functional_labels(analysis))
```

Uncertain tokens remain unlabeled, while the original sentence always reaches the encoder.

Illustrative Pipeline Output. To illustrate the output of this filter, consider the input sentence:

axan nehwatl nitekitis ika notepostsontekosentlalil (“now I will work with my computer”)

The FST analyzer and filter process each token as follows:

– Token analysis:

• axan → adv (temporal particle)

• nehwatl → subj:1sg (emphatic subject pronoun)

• nitekitis → ni-tekiti-s → subj:1sg + verb:intr + fut

• ika → rel.noun (relational marker, “with”)

• notepostsontekosentlalil → no-tepostsontekosentlalil → np.poss + noun (“my computer”)

– Extracted functional tags: adv, subj:1sg, head, np.poss

– Final encoder input string:

nah->es: [SYN: ADV SUBJ:1SG HEAD NP.POSS] axan nehwatl nitekitis ika notepostsontekosentlalil

This complete prefix string provides the neural encoder with high-level structural guidance while preserving the exact surface subwords for translation.

## 4.5 IGT and Functional Syntax Projection

Selected analyses are mapped to Interlinear Glossed Text (IGT) following the Leipzig Glossing Rules [5] and reflected in a small functional syntax inventory HEAD, SUBJ, OBJ, NP.POSS, ADV, CONJ, DET, REL, COP, NEG, or OPT. Following linguistic

Table 4. Illustrative surface-to-syntax trace used by the conditioning pipeline. Segmentation is shown at the level consumed by the rule-based glosser.
<table><tr><td>Surface</td><td>FST decomposition</td><td>IGT gloss</td><td>Functional tion</td><td>projec-</td></tr><tr><td>tla</td><td>tla</td><td>OPT/conditional parti- OPT or ADV cle</td><td></td><td></td></tr><tr><td>nichnamikis</td><td>ni-ch-namiki-s</td><td>1SG.SBJ-OBJ-meet- FUT</td><td>SUBJ OBJ HEAD</td><td></td></tr><tr><td>tokonetsin</td><td>to-kone-tsin</td><td>1PL.POSS-child-HON</td><td>NP.POSS</td><td></td></tr><tr><td>tisowatl</td><td>ti-sowatl</td><td>2SG.SBJ-woman.PRED</td><td>SUBJ COP HEAD</td><td></td></tr><tr><td>kwalle</td><td>kwalle</td><td>good/well</td><td>ADV or predicate HEAD</td><td></td></tr><tr><td>ammo</td><td>ammo</td><td>NEG</td><td>NEG</td><td></td></tr></table>

feature paradigms for NMT [25], this compact inventory avoids sparsity and parser errors from full dependency structures.

The NMT model receives only the short functional sequence. Context determines whether kwalle, for example, projects as ADV or as predicate HEAD/COP.

The formatted encoder input places the labels before the unchanged source:

nah->es: ADV HEAD NP.POSS tla nichnamikis tokonetsin

The unchanged surface follows the labels, safeguarding against symbolic errors.

## 5 Neural Training, Ablations, and Post-Mortems

All runs warm-start from a 1.2-billion-parameter mT5-large model checkpoint [18], $( d = 1 , 0 2 4$ , 24 encoder and 24 decoder blocks) [27]. Training used one Tesla T4 in fp32 because reduced-precision trials were less stable.

## 5.1 LoRA Configuration

For a frozen weight matrix

$$
W = W _ { 0 } + \varDelta W , \qquad \varDelta W = \frac { \alpha } { r } B A .\tag{2}
$$

We set rank $r ~ = ~ 1 6$ , scaling factor $\alpha = 3 2$ , and dropout 0.1. Following Dettmers et al. [6], who demonstrated that applying LoRA to all linear projection layers is essential for recovering full fine-tuning capacity, we extended adapters beyond attention query and value projections to target all seven linear weight matrices $( q , k , v , o , w i _ { 0 } , w i _ { 1 } , w o )$ . This all-linear setup is particularly important for sequence-to-sequence models, as feed-forward projections carry substantial domain-specific lexical knowledge [10].

Optimization used AdamW with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ and cosine decay. Sentence-only bidirectional dev chrF++, rather than training loss, selected checkpoints. The shared embedding matrix remained frozen, which led to the special-token failure analyzed in Subsection 5.4.

## 5.2 Progression of Systems

Table 5 summarizes the full progression of experimental iterations and model ablations leading to the final production system. Central claims rely on controlled pairwise comparisons that share identical data splits, preprocessing steps, and evaluation code.

Table 5. Model progression and ablation results across development (240 directional instances) and test (300 directional instances) sets, reported in chrF++.

<table><tr><td>Run</td><td>Data or Intervention</td><td>Epochs</td><td colspan="3">Dev chrF++ Test es→nah Test nah→es</td></tr><tr><td>Baseline Checkpoints Base</td><td></td><td></td><td></td><td>23.91</td><td>29.54</td></tr><tr><td></td><td>Mixed-variant base checkpoint Baseline LoRA reference Initial scheduled LoRA snapshot</td><td></td><td>43.87</td><td></td><td></td></tr><tr><td colspan="6">Schedule and Data Augmentation</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>lora_v1 lora_v2</td><td>Real pairs only</td><td>3</td><td>40.26 46.18</td><td>41.76 45.88</td><td>43.46 45.58</td></tr><tr><td>lora_v3b</td><td>Real pairs, extended schedule Real plus 192 BT pairs</td><td>8 (best 6) 12 (best 8)</td><td>49.40</td><td>48.61</td><td>48.98</td></tr><tr><td colspan="6"></td></tr><tr><td></td><td>Subword Formatting and Special Token Post-Mortem</td><td>12</td><td></td><td></td><td></td></tr><tr><td>lora_v4 lora_v5</td><td>New, cold special tokens Text prefix, no new tokens</td><td></td><td>8.91 51.69</td><td>53.60</td><td>51.27</td></tr><tr><td>lora_v6</td><td>Revised special-token attempt</td><td>10 (best 3)</td><td>&lt; 15</td><td></td><td></td></tr><tr><td colspan="6">Morphosyntactic Coverage Ablation (Sparse vs. Dense)</td></tr><tr><td>lora_v7a</td><td>Control, no syntax, train_v3</td><td>10 (best 3)</td><td>51.22</td><td>30.49</td><td>31.38</td></tr><tr><td>lora_v7b</td><td>Treatment, 29.5% syntax</td><td>10 (best 9)</td><td>51.31</td><td>29.40</td><td>30.11</td></tr><tr><td>lora_v8a</td><td>Control, no syntax, train_v4</td><td>10 (best 9)</td><td>54.67</td><td>56.40</td><td>52.67</td></tr><tr><td>lora_v8b</td><td>Treatment, 95.8% syntax</td><td>10 (best 10)</td><td>55.56</td><td>56.91</td><td>54.00</td></tr><tr><td colspan="6">Final Selection and Production Model</td></tr><tr><td>lora_v9c_a</td><td>Full data + BT, bare text ablation (no syntax)</td><td>10 (best 9)</td><td>58.91</td><td></td><td></td></tr><tr><td>lora_v9c_b</td><td>Full data + BT, 95.8% dense syntax (production) 10 (best 6)</td><td></td><td>58.46</td><td>58.20</td><td>57.90</td></tr></table>

The production score is 14.59 points above the 43.87 baseline-LoRA reference and 9.06 points above lora\_v3b. Against the mixed-variant base, final directional gains are 34.29 (es → nah) and 28.36 (nah → es) points.

## 5.3 Backtranslation and Schedule Efects

Training Schedule and Checkpoint Selection. Fine-tuning a 1.2-billion-parameter model on a small dataset requires balancing parameter adaptation against rapid memorization. In lora\_v1, we evaluated a short 3-epoch schedule on real pairs. In lora\_v2, we extended training to 8 epochs with evaluation every 200 steps. Extending the schedule raised dev chrF++ from 40.26 to 46.18 (+5.92 points). Validation performance peaked at epoch 6 before plateauing by epoch 8, demonstrating that low-rank adapters need suficient steps to align cross-lingual representations but quickly overfit if trained past convergence.

Asymmetric Target-Preserving Backtranslation. Standard backtranslation risks degrading the generation decoder by training on hallucinated synthetic targets. To prevent this, we implemented an asymmetric protocol: using the nah → es capability of lora\_v2, we generated synthetic Spanish source prompts for 192 continuous narrative utterances spoken by community elders, adding these pairs strictly to the es → nah training split (train\_bt.jsonl). Integrating these 192 backtranslated pairs in lora\_v3b increased dev chrF++ from 46.18 to 49.40 (+3.22 points) and raised both directional tests to approximately 49 chrF++. Although representing only an 8% increase over the real-pair inventory, they introduced multi-clause discourse patterns and authentic elder-authored syntax that isolated pedagogical pairs lack.

## 5.4 Failure Analysis: Cold Embedding Degradation

In lora\_v4 and lora\_v6, we attempted to delimit syntactic conditioning regions using dedicated special tokens (<source>, <syntax>, and <no\_syntax>). This intervention resized the shared vocabulary from 250,100 to 250,103, introducing uninitialized rows into the embedding matrix. Because standard LoRA freezes the base embedding table to conserve GPU memory, these newly added token vectors received zero gradient updates and remained fixed random vectors throughout training.

Presenting the encoder with static random vectors at structurally critical positions caused total optimization failure: training loss stalled above 30.0 and dev chrF++ collapsed to 8.91. While unfreezing the full embedding table resolved the representation issue conceptually, it immediately exhausted T4 GPU memory.

In lora\_v5, we resolved this constraint by passing functional syntax as ordinary text prefixes (such as nah->es: ADV HEAD NP.POSS) segmented with the existing SentencePiece vocabulary. Reusing pretrained subword representations stabilized optimization without custom checkpoint surgery and restored dev chrF++ to 51.69. All subsequent runs retain this plain-text prefix format.

## 5.5 Sparse and Dense Syntax Coverage

To assess how annotation density afects conditioning, we compared two versions of the finite-state analyzer. In lora\_v7b, the early prototype produced functional syntax tags for only 29.5% of corpus sentences. Adding these sparse prefixes yielded virtually identical development scores to the unconditioned lora\_v7a control (51.31 versus 51.22 chrF++) and slightly degraded directional test accuracy.

Because sparse tags occurred inconsistently, the model treated them as shallow domain indicators rather than reliable syntactic grounding. In lora\_v8b, once rule expansion and reconstruction filtering raised tag coverage to 95.8%, syntax conditioning produced a +1.33 chrF++ gain on complex nah → es polysynthetic targets and a +0.51 spillover gain on es → nah. This demonstrates that syntactic conditioning requires high annotation density to guide sequence generation efectively. Precision in the reconstruction filter remains critical, as inaccurate tags would inject systematic structural noise.

## 5.6 Final lora\_v9c Ablation and Model Selection

To isolate the precise contribution of dense syntactic prefixes on the final training set, we evaluated a controlled ablation where the extracted syntax tags were stripped from input sequences during training and inference (strip=True), leaving only bare source text.

On the development split, the syntax-stripped control reached 58.91 chrF++, while the syntax-conditioned model (lora\_v9c\_b, strip=False) reached 58.46 chrF++, a minor diference of −0.45 points.

Despite this small aggregate diference on dev, we selected the syntaxconditioned checkpoint (lora\_v9c\_b) for production deployment. The explicit syntactic prefix provides necessary structural disambiguation for complex grammatical constructions that aggregate character metrics fail to penalize. These include diferentiating modifier versus predicate readings of kwalle, resolving null-copula structures, and tracking possessed nominal arguments. We report the unconditioned control’s development score transparently, but accept this minor trade-of to ensure consistent grammatical interpretability. Across fully evaluated held-out test splits, lora\_v9c\_b remains our strongest model, reaching 58.20 and 57.90 directional test chrF++.

## 6 Qualitative Translation and Decolonial Evaluation

Automatic scores summarize thousands of character n-grams but do not explain which linguistic decisions changed. We therefore use diagnostic examples to examine regional spelling, word-internal person marking, possessive prefixes, predicate nominals, and honorific morphology. The examples in Table 6 are targeted probes, not an independent statistical test. They are included to make the error categories falsifiable and to show what is inspected during communityfacing review.

Table 6. Representative Spanish-to-Masewalkopa diagnostic comparisons. The base column illustrates mixed-variant or segmentation failure modes; bold material highlights the locally targeted form in the production output.
<table><tr><td>Spanish source Mixed-variant</td><td>Specialized base (1ora_v2)</td><td>+BT +Dense syntax (1ora_v9c_b)</td><td>Human reference</td></tr><tr><td>base Te agradezco. tlazohcamati</td><td>nimitstlaskamati</td><td>nimitstlasohkamate</td><td>nimitstlasohkamate</td></tr><tr><td>Nuestro hijito to piltzin cualli</td><td>tokone kwalle</td><td>tokonetsin kwalle</td><td>tokonetsin kwalle</td></tr><tr><td>está bien. Eres una mujer ti cihuatl cualli</td><td>tisowatl kwalle</td><td>tisowatl kwalle</td><td>tisowatl kwalle</td></tr><tr><td>buena. Encontraré a ni. . . namiki nuestro hijito. topiltzin</td><td>niknamikis tokone</td><td></td><td>nichnaamikis tokonetsin nichnaamikis tokonetsin</td></tr></table>

The first row probes a short lexical form to test whether the model preserves isolated citation vocabulary. The mixed model favors a broad Classical-style lexical form, while specialization produces the correct person-marked Masewalkopa string nimitstlasohkamate. The important change is not only orthographic. The production form packages the relation between speaker and addressee inside the word, rather than returning an uninflected dictionary-style equivalent.

The second and fourth rows test possession and the honorific or reverential sufix -tsin. The base output separates to from a generic piltzin, resembling word-by-word transfer and failing to realize the locally analyzed possessed stem. lora\_v2 learns the possessive prefix but omits the full honorific ending. lora\_v9c\_b produces to-kone-tsin, keeping first-person plural possession and the respectful/diminutive morphology in one nominal word. Such a distinction can afect interactional stance even when the untranslated content word “child” is recoverable.

The third row illustrates predicate nominals. A mixed output with separate ni can be interpreted as a tokenization artifact or as a cross-variety spelling convention. The correct tisowatl realizes person marking on the predicate. In the final phrase, kwalle must be interpreted in relation to that predicate, not assigned a fixed label solely from its dictionary gloss. This is the type of ambiguity that motivated the retention of syntax conditioning in lora\_v9c\_b.

Qualitative analysis also reveals what the current table cannot establish. A single human reference is not the only grammatical realization, and matching it exactly is not proof of social appropriateness. Honorific morphology can be licensed by participant relation and discourse context, neither of which is fully represented in isolated Spanish prompts. For deployment, the examples are therefore paired with warnings against treating the system as a normative authority.

## 6.1 Domesticating Bias Score

Standard multi-variant Nawatl corpora are heavily skewed toward colonial-era missionary texts, biblical translations, and pan-variety pedagogical materials. Models trained on these resources inherit a structural bias toward archaic, standardized Central forms (such as tlazohcamati or cihuatl), which local speakers perceive as alien or domesticating. In contrast, translation into the community demands fidelity to the living, spoken variety of Santa Ana Tlacotenco (such as nimitstlasohkamate or sowatl). To evaluate this dynamic directly, the Domesticating Bias Score (DBS) measures the proportion of diagnostic translations in which a model defaults to generalized, colonial-derived variants rather than verified local community lexical choices.

Standard reference-based metrics can reward Spanish loanwords or archaic colonial forms when the reference itself reflects domesticated translation practices. Drawing on pseudo-log-likelihood bias evaluation paradigms [15, 20] and lengthnormalized sequence scoring [23], We define the Domesticating Bias Score (DBS) as the diference in average per-token log-probability assigned by the model to a colonially domesticated candidate $\left( t _ { c } \right)$ versus a sovereign community candidate (t<sub>s</sub>) given source input x:

$$
\mathrm { D B S } ( x ) = \frac { 1 } { | t _ { c } | } \sum _ { j = 1 } ^ { | t _ { c } | } \log P ( t _ { c , j } \mid t _ { c , < j } , x ) - \frac { 1 } { | t _ { s } | } \sum _ { j = 1 } ^ { | t _ { s } | } \log P ( t _ { s , j } \mid t _ { s , < j } , x ) .\tag{3}
$$

Length is measured in decoder tokens. A positive DBS indicates that the model assigns a higher average per-token log-probability to the domesticated candidate, whereas a negative DBS indicates a preference for the sovereign candidate. Values near zero reflect weak preference or equally probable alternatives.

Token normalization is necessary because sovereign formulations can be longer than borrowed Spanish nouns or simplified roots, either morphologically or periphrastically. Without normalization, summing raw log-probabilities almost always penalizes longer strings because each additional token contributes a nonpositive term. The score is contrastive rather than absolute. It does not declare any individual term intrinsically colonial. Instead, each contrastive pair must be constructed with community speakers who can evaluate the social and historical distinctions in context.

On the illustrative seed set used during development, mean DBS shifted from −1.313 for the base model to −0.380 after specialization. Under the sign convention in Eq. 3, both values favor the sovereign candidate, but the specialized score is closer to zero. It would therefore be incorrect to describe this shift as a stronger sovereign preference. The result instead reflects a reduced magnitude in the modeled contrast. We retain the metric because it exposes candidate-level probability distributions, but we interpret direction and magnitude explicitly rather than attaching an uncritical label to every numerical movement.

This distinction is methodologically important. In production evaluations, DBS can be reported alongside bootstrap confidence intervals over contrast pairs, sign tests for the proportion of instances with DBS < 0, and stratified subsets across specific phenomena, including lexical borrowing, kinship terminology, toponyms, and ceremonial register.

Future community-designed evaluation benchmarks should also support a “both acceptable” label to ensure that the metric does not impose an artificial binary interpretation on ordinary dialectal variation.

## 6.2 Error Analysis and Qualitative Failure Modes

Although the production system (lora\_v9c\_b) substantially improves lexical alignment and local morphology, systematic qualitative inspection reveals several recurrent failure modes on dificult constructions. Table 7 categorizes representative translation errors observed on the held-out evaluation sets.

Table 7. Representative diagnostic failure modes produced by the production model (lora\_v9c\_b) against authentic community references.
<table><tr><td>Spanish Source</td><td></td><td>(1ora_v9c_b)</td><td>Production Output</td><td>Human Reference</td><td>Error Diagnosis</td></tr><tr><td>acuerdo</td><td></td><td></td><td>sí sé, pero no me kemma nikmate, wan ammo kemma nikmate, wan ammo Semantic nikmati</td><td>nikilnaamike</td><td>repetition: Loops base cognition verb (mate) rather than retrievingpsychological</td></tr><tr><td></td><td></td><td></td><td>kalohtle</td><td>ayer te vi en la calle yalwa onimitsitak ipan yalhwa onimitsihtak ipan Orthographic kalohtle</td><td>memory root (ilnaamiki). bleed: Omits local Santa Ana aspiration marker (-h-) in both time adverbial</td></tr><tr><td>señaré</td><td></td><td></td><td>mañana yo te en-mustla nimitsmachtia</td><td>mustla nimitsmachtis</td><td>(yalhwa) and root (ihta). Aspect truncation: Drops future suffix (-s) under complex prefixation, reverting to default non-</td></tr></table>

These errors cluster into three distinct linguistic phenomena:

1. Semantic Repetition on Mental State Predicates. When translating clauses that contrast knowledge with memory (such as kemma nikmate, wan ammo nikilnaamike), the model occasionally repeats the higher-frequency cognitive root mate (‘to know / feel’) in the second coordinate clause instead of generating the intended psychological retrieval predicate ilnaamiki (‘to remember’).

2. Multi-Variant Orthographic and Phonetic Bleed. On roots and adverbials shared across Central Nawatl varieties, the model occasionally defaults to generalized Central spellings (such as yalwa and itak) rather than generating the localized Santa Ana Tlacotenco orthography with preconsonantal aspiration (yalhwa and ihtak), reflecting residual weight bias from the multi-variant base checkpoint.

3. Morphological Sufix Truncation in Future Aspect. In transitive verbal complexes combining directional or person prefixes with temporal adverbs (such as mustla nimitsmachtis ‘tomorrow I will teach you’), the decoder occasionally drops the final future sufix -s, yielding an uninflected or present-tense form (nimitsmachtia) despite explicit future context in the source.

## 7 Production Deployment and CPU Profiling

A model that requires a research GPU for every query is dificult to sustain as a community-facing service. We therefore merged the selected lora\_v9c\_b adapter into the base checkpoint, converted the model to CTranslate2 [13], and quantized weights for int8 CPU inference. Conversion fixes one deployable artifact and removes the need to load separate adapter modules at request time.

## 7.1 Latency Profile

Native fp32 PyTorch inference on CPU required 137.5 seconds for a representative request. This latency is unsuitable for an interactive translator and increases the period for which a billed container must remain active. On a Cloud Run allocation of $2 \ \mathrm { v C P U s }$ and 4 GiB RAM, the CTranslate2 int8 artifact produced typical sentence latencies of 1.8 to 2.6 seconds. Taking the range endpoints, the observed speedup is approximately $1 3 7 . 5 / 2 . 6 \approx 5 2 . 9$ to $1 3 7 . 5 / 1 . 8 \approx 7 6 . 4 \times$

Table 8. CPU inference profile for the production path. Latencies are observed requestlevel values and should not be interpreted as a hardware-independent benchmark.
<table><tr><td>Runtime</td><td>Weight precision Beam</td><td></td><td>Latency per sentence</td></tr><tr><td>PyTorch CPU</td><td>fp32</td><td>4</td><td>137.5 s</td></tr><tr><td>CTranslate2 CPU</td><td>int8</td><td>4</td><td>1.8–2.6 s, but short-form failures</td></tr><tr><td>CTranslate2 CPU</td><td>int8</td><td>8</td><td>1.8–2.6 s, validated production setting</td></tr></table>

The comparison combines runtime optimization and quantization, so it does not isolate the causal contribution of int8 weights alone. It answers the operational question of whether the fully converted path is fast enough to serve an interactive request within the selected CPU allocation.

## 7.2 Beam Search Divergence on Short Forms

Initial deployment testing found mid-word truncation on short lexical inputs, including nimitstlasohkamate. The failure initially appeared to be a quantization artifact [8], but comparisons across converted precisions showed it was precisionindependent. Greedy and narrow-beam paths could terminate after a highprobability partial subword sequence even though the full form was preferred by the original evaluation setup.

Increasing beam\_size from 4 to 8 eliminated the observed truncations in the short-form regression set. The wider beam retained complete hypotheses long enough for the accumulated sequence score to overtake prematurely terminated candidates. This is described as beam search divergence rather than as a general CTranslate2 defect because the trigger depends on the model’s score distribution, tokenization, length normalization, and stopping behavior [14, 26].

The production API sets the beam size to 8 and runs regression probes after every model conversion. The probes include single words, morphologically long words, short clauses, both translation directions, and forms with possessive or honorific material. This test set is separate from the research test split. Its role is software validation, specifically ensuring that a numerically strong model does not become linguistically unusable after conversion.

## 7.3 Cloud Run Microservice Architecture

The service exposes a FastAPI application within a container that includes the CTranslate2 model and the mT5 tokenizer assets. A translation request passes through four stages:

1. validate direction, input length, and Unicode text;

2. construct the direction and optional syntax prefix;

3. tokenize, decode with beam size 8, and remove model control text;

4. return the translation plus non-sensitive runtime metadata.

The Cloud Run revision uses 2 vCPUs, 4 GiB RAM, and a maximum of two instances. The instance cap limits both concurrency and worst-case cost. Because model loading is expensive relative to a warm translation, the container initializes the model once at process startup rather than once per request.

CORS is configured through an explicit allowlist for the web client origins that require browser access. Methods and headers are restricted to the translation interface rather than using an unrestricted wildcard in production. CORS is not authentication, so request validation, rate limiting at the surrounding platform layer, and conservative maximum input lengths remain necessary. The API does not log full translation content by default, since sentences may contain names or culturally sensitive material.

## 7.4 Automated Cost Controls

A billing budget publishes threshold notifications to Pub/Sub. A subscribed Cloud Function evaluates the notification and disables project billing when the configured hard threshold is reached. This is intentionally stronger than an email alert: an alert informs the operator, while the function can stop additional billable use. The mechanism complements the Cloud Run maximum-instance setting and protects a student-operated project from trafic spikes or accidental loops.

Disabling billing is disruptive and can stop unrelated resources in the same project. For that reason, the deployment isolates the translator within a dedicated project, tests the function at non-production thresholds, and documents manual recovery procedures. Budget automation also has notification latency, so it is a final circuit breaker rather than a substitute for instance caps, request quotas, and monitoring.

## 8 Discussion, Limitations, and Reproducibility

The experiments show that a mixed-variant checkpoint can provide useful transfer when a second stage redirects it toward one documented variety. Authentic target provenance matters in backtranslation, and symbolic conditioning becomes useful only after analyzer coverage is dense and reconstruction is precise. This does not imply that speakers or models must be linguistically isolated. A future multivariety system could explicitly condition on community, orthography, or speaker preference rather than aggregating those distinctions without labels.

The corpus remains small and pedagogically skewed. Its 192 oral-text sentences cannot represent all genres or speakers, the human evaluation has 20 items with only 2 raters, and DBS is still an illustrative probe of candidate log-probability distributions rather than an absolute metric. Likewise, reaching 72.9% genuine FST segmentation in the lexicon reflects rule coverage rather than verified parsing precision. Future work must validate analyses by morpheme type, especially possession, aspect, and predicate nominals.

Full reproducibility requires the deduplicated split manifest, directional JSONL, FST source hash, adapter configuration, selected checkpoint, random seeds, and decoding settings. Sentence-only scores must remain separate from lexical data.

Responsible release also requires speaker-approved access conditions for oral texts and a model card that names Masewalkopa as the target variety, documents errors, and provides routes for correction or removal.

This is something that will be spoken about and talked with the community before making any data accessible for any other purposes.

## 9 Conclusion and Future Work

This work began with a fundamental dilemma in minoritized language technolo gies: while aggregating data across regional varieties combats data sparsity in the aggregate, it produces an artificial, blended register that is accountable to no single speaker community. As demonstrated by my initial 70,000-pair thesis baseline, high scores on mixed-corpus benchmarks can easily mask severe structural failure on authentic regional speech. We have termed this failure variant fragmentation.

To resolve this issue, this paper demonstrated the process and efectiveness of specializing an existing sequence-to-sequence model to a specific regional variety: Masewalkopa (nhn), spoken in Santa Ana Tlacotenco. Using a relatively small community corpus of 2,665 curated pairs, a stratified evaluation split that strictly measures sentential syntax, and 192 authentic elder-transcribed narrative targets integrated through asymmetric backtranslation, we adapted a 1.2-billion-parameter checkpoint on a single GPU. The resulting specialized system (lora\_v9c\_b) achieved 58.46 dev chrF++ and 58.20/57.90 directional test chrF++, gaining over 28 points over the initial multi-variant baseline.

Beyond aggregate metric gains, my experimental ablations and failure analyses provide concrete architectural takeaways for indigenous and polysynthetic NMT:

1. Subword embedding constraints in LoRA: Introducing new special tokens while freezing the shared embedding matrix creates an untrained bottleneck that stalls model convergence. Conditioning the translation encoder using plaintext syntactic prefixes reuses the pretrained subword space and completely recovers model performance.

2. Density in morphosyntactic conditioning: Sparse functional syntax tags act merely as shallow domain indicators, whereas high-coverage tags (95.8%) generated by a rule-based finite-state analyzer provide explicit structural guidance that improves translation on complex polysynthetic targets (+1.33 chrF++).

3. Decolonial and multi-faceted evaluation: Character overlap metrics alone cannot detect whether a translation respects local autonomy. Combining diagnostic morphological probes, length-normalized Domesticating Bias Scores (DBS), and blind human ratings verified that the specialized model actively prefers sovereign community vocabulary over colonial Spanish substitutions.

4. Accessible CPU deployment: Merging low-rank adapter weights and applying 8-bit CTranslate2 quantization reduced inference latency from 137.5 seconds down to 1.8–2.6 seconds per request. Widening the beam search to 8 resolved search-space truncations on short lexical inputs, enabling responsive and cost-bounded deployment on serverless CPU infrastructure.

## 9.1 Future Work

Moving forward, we plan to expand community-led evaluations with native speakers in Santa Ana Tlacotenco to establish formal qualitative rubrics for honorific and directional morphology. In addition, we will extend our finite-state conditioning pipeline to handle long-form discourse in community oral histories, and investigate parameter-eficient specialization across other Nawatl varieties using isolated, modular adapter banks.

Ultimately, language technologies for minoritized languages should not aim to homogenize distinct varieties into a single generic system. By demonstrating that an existing sequence-to-sequence model can be rapidly and accurately aligned to a single community standard, this work provides a practical, decolonial blueprint for building translation tools that honor the linguistic sovereignty of indigenous speaker communities.

## Acknowledgements

Thanks to those who helped with answering questions and providing lessons, without whom this project would have been even more challenging. And thanks to members of the LECS Lab at CU Boulder for comments and suggestions. This work was supported by the National Science Foundation under Grant No. 2149404, “CAREER: From One Language to Another.”

nikinmotlasohkamachililia nochtin tlin onichmopalewililihkeh

## Bibliography

[1] Aguayo Rousell, H.B., Piña Osorio, J.M.: Expressions of racism in a sample of university students in Mexico. Sinéctica (46) (2016)

[2] Beesley, K.R., Karttunen, L.: Finite State Morphology. CSLI Publications Stanford (2003)

[3] Bird, S.: Decolonising speech and language technology. In: Proceedings of the 28th International Conference on Computational Linguistics (COLING). pp. 3504–3519 (2020)

[4] Carroll, S., Garba, I., Figueroa-Rodríguez, O., Holbrook, J., Materechera, R., Raseroka, M., Hudson, M.: The care principles for indigenous data governance. Data Science Journal 19(1), 43 (2020)

[5] Comrie, B., Haspelmath, M., Bickel, B.: The Leipzig glossing rules: Conventions for interlinear morpheme-by-morpheme glosses (2008)

[6] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: Qlora: Eficient finetuning of quantized llms. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 36, pp. 10088–10115 (2023)

[7] Feldman, I., Coto-Solano, R.: Neural machine translation models with backtranslation for the extremely low-resource indigenous language bribri. In: Proceedings of the 28th International Conference on Computational Linguistics. pp. 3965–3976 (2020)

[8] Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M.W., Keutzer, K.: A survey of quantization methods for eficient neural network inference. In: Low-Power Computer Vision, pp. 291–326 (2021)

[9] Gruda, S., Haimovich, G., Sullivan, J.: Lexical creativity in modern Nahuatl: An analysis of multidialectal data. Lingua 285, 103488 (2023)

[10] He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., Neubig, G.: Towards a unified view of parameter-eficient transfer learning. In: International Conference on Learning Representations (ICLR) (2022)

[11] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (ICLR) (2022)

[12] Instituto Nacional de Lenguas Indígenas (INALI): Catálogo de las lenguas indí- genas nacionales: Variantes lingüísticas de México con sus autodenominaciones y referencias geoestadísticas. Instituto Nacional de Lenguas Indígenas, México, D.F. (2009)

[13] Klein, G., Kim, Y., Deng, Y., Senellart, J., Rush, A.M.: OpenNMT: Open-source toolkit for neural machine translation. In: Proceedings of ACL 2017, System Demonstrations. pp. 67–72 (2017)

[14] Koehn, P., Knowles, R.: Six challenges for neural machine translation. In: Proceedings of the First Workshop on Neural Machine Translation. pp. 28–39 (2017)

[15] Kurita, K., Vyas, N., Pareek, A., Black, A.W., Tsvetkov, Y.: Measuring bias in contextualized word representations. In: First Workshop on Gender Bias in Natural Language Processing. pp. 166–172 (2019)

[16] Lindén, K., Silfverberg, M., Pirinen, T.: HFST—framework for compiled finitestate morphology. In: International Workshop on Systems and Frameworks for Computational Morphology. pp. 67–85. Springer (2011)

[17] Mager, M., Gutierrez-Vasques, B., Zheng, R., Liu, F., Mager, E., Kann, K.: Findings of the americasnlp 2021 shared task on open machine translation for indigenous languages of the americas. In: Proceedings of the 1st Workshop on NLP for Indigenous Languages of the Americas. pp. 202–217 (2021)

[18] Mercado-Campos, L.A.: Design and implementation of an NMT system for Spanish-Nahuatl. Master’s thesis, Universidad del País Vasco / Euskal Herriko Unibertsitatea, Donostia-San Sebastián, España (Jun 2023)

[19] Mercado-Campos, L.A., Pugh, R., Palmer, A.: The power of simplicity: Ngrams and transformers in Nahuatl language identification. In: Mager, M., Ebrahimi, A., Bui, M.D., Pugh, R., Oncevay, A., Chiruzzo, L., Solano, R.C., Rijhwani, S., von der Wense, K. (eds.) Proceedings of the Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP). pp. 153–167. Association for Computational Linguistics, San Diego, California, USA (Jul 2026). https://doi.org/10.18653/v1/2026.americasnlp-6.14, https: //aclanthology.org/2026.americasnlp-6.14/

[20] Nangia, N., Vania, C., Bisk, Y., Bowman, S.R.: Crows-pairs: A challenge dataset for measuring social biases in masked language models. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 1953–1967 (2020)

[21] Parodi, C.: Multiglosia virreinal novohispana: el náhuatl. Cuadernos de la ALFAL 2, 89–101 (2011)

[22] Popović, M.: chrf++: Words helping character n-grams. In: Proceedings of the Second Conference on Machine Translation. pp. 612–618 (2017)

[23] Salazar, J., Liang, D., Nguyen, T.Q., Kirchhof, K.: Masked language model scoring. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. pp. 2699–2712 (2020)

[24] Sennrich, R., Haddow, B., Birch, A.: Improving neural machine translation models with monolingual data. In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics. pp. 86–96 (2016)

[25] Sennrich, R., Haddow, B.: Linguistic input features improve neural machine translation. In: Proceedings of the First Conference on Machine Translation (WMT). pp. 83–91 (2016)

[26] Stahlberg, F., Byrne, B.: On NMT search errors and model errors: Cat got your tongue? In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP). pp. 3356–3362 (2019)

[27] Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., Rafel, C.: mt5: A massively multilingual pre-trained text-to-text transformer. In: Proceedings of NAACL-HLT. pp. 483–498 (2021)

[28] Zhang, M., Li, Z., Fu, G., Zhang, M.: Syntax-enhanced neural machine translation with syntax-aware word representations. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). pp. 1151–1161 (2019)