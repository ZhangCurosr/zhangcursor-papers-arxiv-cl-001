# SNO-PI: Label-Scarce SNOMED CT Entity Linking with Pre-trained Encoders and Curated Terminology Knowledge

Piyawoot Songsiritat, MBBS, FRACGP

The University of Queensland, Greater Brisbane Clinical School, Brisbane, Australia p.songsiritat@uq.edu.au

## Abstract

We present SNO-PI, the top-performing submission on SNOMED International’s SNOMED CT Entity Linking Benchmark as of 18 October 2026. SNO-PI achieves a macro character IoU of 0.5387, outperforming the organisers’ reference system (0.4582; based on the 2024 challenge winner’s approach) by 18% relative. Linking clinical mentions to SNOMED CT is a zero- and few-shot problem over a very large label space: the 272 annotated MIMIC-IV discharge summaries cover only 6,595 of the roughly 219,000 eligible concepts, and a third of those occur only once. This motivates a design built on transfer from pre-trained models and human-curated knowledge. SNO-PI combines highrecall span proposal, SapBERT candidate retrieval and listwise ensemble reranking with three BERT-family encoders, each further pre-trained with masked language modelling on unlabelled MIMIC-IV notes and then on SNOMED CT synonyms and hierarchy before task fine-tuning. We find that curated knowledge helps most on concepts never seen in training, yet these concepts remain the main source of error.

## 1 Introduction

Much of what clinicians record about a patient lives only in free text. A discharge summary carries diagnoses, examination findings, procedures and allergies in the shorthand of clinical practice: “hct”, “SOBOE”, “lap chole”. Until these are linked to a reference terminology such as SNOMED CT, they are invisible to allergy alerts, problem lists, registries and research queries. Unlike general-domain entity linking, clinical linking cannot rely on large labelled corpora: shareable clinical notes are scarce because they contain protected health information, and labelling them requires clinical expertise, which is costly and slow to obtain. The 2024 SNOMED CT Entity Linking Challenge [1] released a public corpus [2] of discharge notes densely annotated with SNOMED CT [3], drawn from MIMIC-IV-Note [4, 5]. Its organisers concluded that data, rather than model architecture, limited performance, with rare and unseen concepts the hardest [1]. SNOMED International continues it as an open benchmark [6], hosted by DrivenData and scored on a hidden set of about 20 notes.

The scale of the gap is stark. The 272 annotated notes contain about 75,000 labels, but they touch only 6,595 of the roughly 219,000 concepts eligible for linking [2], or about 3%. Any test note will therefore mention concepts for which no labelled example exists. Yet the domain is unusually rich in curated knowledge: SNOMED CT itself supplies synonyms, fully specified names and an is-a hierarchy for every concept, the UMLS [7] links these to millions of further names, and hundreds of thousands of unlabelled clinical notes are available for self-supervision.

SNO-PI is built around this asymmetry. Its contributions are: (i) a recall-first linking pipeline in which curated terminology enters at every stage; (ii) a staged training recipe (clinical MLM → SNOMED pre-training → task fine-tuning) that lets cross-encoders learn concepts they never see labelled; and (iii) an analysis of where scarcity still bites, including note-level variation in annotation conventions.

## 2 Task and the scarcity problem

Task. Systems output (note, start, end, concept) tuples for mentions of concepts under three SNOMED CT hierarchies: Clinical finding, Procedure and Body structure. Negated findings and family history link to the plain concept, laterality is pre-coordinated into body structures, and lab abbreviations map to measurement procedures (hgb → Measurement oftotal hemoglobin concentration). The primary metric is macro character IoU. Let $G _ { c }$ and $P _ { c }$ be the sets of (note, character) positions covered by gold and predicted spans of concept c, and let $\mathcal { C } = \mathcal { C } _ { \mathrm { g o l d } } \cup \mathcal { C } _ { \mathrm { p r e d } }$ . Then

$$
\mathrm { I o U } _ { c } = \frac { \left| { G _ { c } \cap P _ { c } } \right| } { \left| { G _ { c } \cup P _ { c } } \right| } , \qquad \mathrm { M a c r o } = \frac { 1 } { \left| \mathscr { C } \right| } \sum _ { c \in \mathscr { C } } \mathrm { I o U } _ { c } ,\tag{1}
$$

so a rare concept counts as much as a common one and every spurious concept adds a zero. The secondary supportweighted score weights $\mathrm { I o U } _ { c }$ by the number of gold spans of c. Submissions are code packages that the organisers’ harness executes on hidden test notes, offline on one A100 within one hour, and scores with the official metric; participants never see the hidden notes or their annotations.

How scarce are the labels? The 272 notes hold 75,463 spans over 6,595 distinct concepts. The distribution is longtailed: 2,504 concepts occur once and 4,788 five times or fewer. Figure 1 shows how slowly coverage grows with annotation effort. With 174 training notes, only 77% of the concepts in a random 20-note test sample have been seen at least once, and the curve extrapolates to only about 84% at all 272 notes. Because the metric is macro-averaged, these unseen concepts (about 20% of the concepts in a sample, but only 9% of spans) weigh far more than their frequency.

![](images/c678b2436a0e3fcfcb62fec632ced589721bf4d0ee3260161349f78fcc0eb016.jpg)

Figure 1: Label coverage grows slowly with annotation effort. Test gold coverage over random 20-note samples, for random subsets of training notes (3 seeds). Dashed: log-linear extrapolation to all 272 notes.  
![](images/cba14b0e24fd93b18bba9ad81ca37744f7eddeb6664db35042dcd1612230527b.jpg)  
Figure 2: Synthetic illustration of note-level variation: one list item is linked, its neighbours are not.

Scarce evidence of convention. Six clinicians annotated the corpus, mostly one per note, with mean pairwise Jaccard agreement of 0.85, which the organisers treat as an upper bound on achievable IoU [1]. Within the remaining disagreement, peripheral choices vary between notes. In Figure 2, only one of several listed allergies is annotated; clinically, this does not mean the patient tolerates the other drugs, but reflects selective annotation, which matters when judging model errors against this reference. Such choices are consistent within a note, plausibly reflecting individual annotators’ habits: pcp is annotated at every occurrence in 32 notes and at none in 38, and the same phrase elsewhere in the note predicts annotation better than the phrase alone (83.5% vs 70.9%). Hand-written normalisation rules all lowered our score, whereas learned rerankers absorbed these conventions implicitly through the NONE option and alias statistics.

Evaluation protocol. We held out the 68 notes of the original 2024 test split plus 30 training notes (dev) and trained on the remaining 174. Every score is the mean over 1,000 random 20-note subsets (sd about 0.014), and a change was accepted only if it gained more than 0.003 on both dev and test on paired subsets. Final submissions were retrained on all 272 public notes. Models were trained on NVIDIA GeForce RTX 50-series GPUs.

<table><tr><td>Knowledge source</td><td>Where it enters</td><td>Measured effect</td></tr><tr><td>SNOMED terms and synonyms</td><td>span proposal, retrieval index, option text</td><td>only source of candidates for unseen concepts (R@10 .77)</td></tr><tr><td>UMLS synonym sets (via SapBERT)</td><td>retrieval</td><td>R@10 .969 overall</td></tr><tr><td>Pre-trained encoders (general and biomedical)</td><td>reranker initialisation, ensemble</td><td>prior that SNOMED pre-training builds on (SNOMED pre-training row); BioLinkBERT best on unseen in a controlled base-size</td></tr><tr><td>50k unlabelled notes (MLM)</td><td>tagger and encoders</td><td>comparison (.784) tagger F1 +.009; small but consistent reranker gains</td></tr><tr><td>SNOMED pre-training</td><td>reranker initialisation</td><td>concept acc. .26 → .85 / .71 (seen / unseen) with no labels (Table 2); +.015 macro after</td></tr><tr><td>SNOMED hierarchy</td><td>option text, pre-training hard negatives</td><td>fine-tuning (ablation) scope filter, parent in scope via is-a keeps the ~22% of gold annotations a semantic-tag filter would drop; part of SNOMED pre-training gain</td></tr></table>

Table 1: Where non-label knowledge enters SNO-PI and what it bought (local test, 174 training notes).

## 3 Compensating for scarce labels

We describe the pipeline (Figure 3) by the knowledge source each component draws on. Table 1 summarises the measured effect of each source.

Curated terminology for recall. Exact matches of inscope SNOMED CT terms are united with the spans of a BERT-CRF [8, 9] tagger and with all span texts seen in training. The union reaches 94–95% exact and 99% overlap recall of gold spans. Candidates are retrieved with SapBERT [10], a PubMedBERT [11] encoder trained on UMLS synonym sets, over an index of all in-scope SNOMED descriptions plus 12.7k training aliases. Because the index is built from curated synonyms rather than from labels, it reaches concepts no annotator has used: recall@10 is 0.97 overall and 0.77 on concepts unseen in training.

Pre-trained encoders as a knowledge base. The rerankers start from encoders that already carry broad language knowledge and, for two of the three, biomedical knowledge: Bio\_ClinicalBERT [12] was trained on PubMed abstracts and MIMIC-III notes, BioLinkBERTlarge [15] on PubMed with citation links, and ModernBERT-large [16] on a large general corpus including scientific text. The staged recipe builds on this prior rather than replacing it. The clearest evidence is Table 2: after SNOMED pre-training alone, with no annotated notes, the encoder already picks the right concept for 85% of seen and 71% of unseen gold spans. Biomedical pre-training matters most where labels are absent: in a controlled comparison of base-size encoders under one early recipe, BioLinkBERT had the highest unseen-concept accuracy (.784, against .763 to .778 for the others), and a general-domain encoder with a weaker adaptation (DeBERTa-v3 [17]) was clearly worse (.451 vs .487 macro). The three encoders were also chosen for diverse pre-training: any two different encoders combined gain +.010 to .015 macro, while two Bio\_ClinicalBERT variants gain nothing.

![](images/6b1a0f6f1ab2655def9028acc7970785f416506b3164e7ff1fd5ae429d6afbeb.jpg)  
Figure 3: SNO-PI architecture. Solid arrows: inference. Dashed boxes: the training recipe for the encoders. The illustrative input is synthetic.

Unlabelled clinical text. The tagger and all three reranker encoders were further pre-trained with masked language modelling [8] on 50,000 unlabelled MIMIC-IV discharge summaries (166M tokens; held-out perplexity 13.94 → 2.86 for Bio\_ClinicalBERT [12]), a form of domainadaptive pre-training [13] that adapts them to the abbreviations, list structure and line-wrapped layout of discharge notes. In our experiments it improved both downstream tasks, span tagging and reranking, consistently on dev and test.

SNOMED pre-training: learning concepts without labels. Before seeing any annotated note, each reranker is trained on SNOMED CT itself: a synonym of a concept acts as the query, and the model must pick the right concept from fully specified names, synonyms and parents among SapBERT-retrieved options, with the concept’s parent and most similar siblings added as hard negatives. This stage teaches the model the terminology’s structure across concepts that the 272 notes never mention, following the idea of enhancing cross-encoders with knowledge-graph hierarchy for zero- and few-shot medical entity linking [14]. Table 2 shows the C-DAPT reranker after each training stage. A separate ablation put the stage’s gain after fine-tuning at +0.015 macro, almost all on unseen concepts.

<table><tr><td rowspan="2"></td><td colspan="2">Concept acc.</td><td>Linked</td></tr><tr><td>seen</td><td>unseen</td><td>(unseen)</td></tr><tr><td>Stage SapBERT retriever (top-1)</td><td>.285</td><td>.550</td><td>.216</td></tr><tr><td>C-DAPT, raw encoder</td><td>.262</td><td>.266</td><td>.000</td></tr><tr><td>+ SNOMED stage</td><td>.849</td><td>.705</td><td>.106</td></tr><tr><td>+ note fine-tuning</td><td>.948</td><td>.829</td><td>.725</td></tr></table>

Table 2: The C-DAPT reranker (Bio\_ClinicalBERT after MIMIC-IV MLM) after each training stage, local test. Concept acc.: the gold concept is ranked first among the ten concept options, before the threshold and the NONE check. Linked: it also passes the p ≥ 0.5 threshold and the NONE check (unseen concepts only).

Listwise reranking with curated option text. Each span is scored against its 10 options plus an explicit NONE, with a softmax over the 11 choices so that linking and abstention are learned jointly. Fine-tuning minimises

$$
\mathcal { L } = \left\{ \begin{array} { l l } { - \log p _ { y } - \lambda \log ( p _ { y } + p _ { \mathcal { O } } ) , } & { y \neq \mathcal { O } , } \\ { - \log p _ { \mathcal { O } } , } & { y = \mathcal { O } , } \end{array} \right.\tag{2}
$$

where p are the softmax probabilities, y is the gold concept or ∅ (NONE) for non-gold spans, and λ > 0 is a weight. The second term penalises only probability on wrong concepts, so the model learns to abstain conservatively rather than link a wrong concept. Each option is shown as its fully specified name, two shortest synonyms and its parent in the hierarchy, so the model can reason about concepts by their curated description rather than by memorised labels. Each of the three encoders is fine-tuned twice with different input context, giving six rerankers: a short context (the span’s line under its nearest section header) and a wide context (±1 sentence under its section-header path, plus up to three competing overlapping spans, also candidates: ...). The competing spans in the wide context gained +0.018 dev / +0.013 test macro. The final submission adds a seventh member: Qwen3-8B [18], LoRA-tuned [19] to score all options in a single forward pass via a linear head on its hidden states, without text generation. Tested in isolation, it is the strongest single member, and in the ensemble it gave a small improvement on the hidden test set (0.5352 → 0.5387 macro).

Simulating scarcity in training. Training candidates are built with 5-fold cross-fitted aliases and dictionaries, so that about 20% of training mentions look unseen, as they will at test time.

Decision. Member probabilities are averaged; a span is linked if its top concept beats NONE with $p \geq 0 . 5$ , and overlaps are resolved greedily by confidence.

## 4 Results

SNO-PI ranks first on the benchmark on both metrics (Table 3).

<table><tr><td>System</td><td>Macro</td><td>Weighted</td></tr><tr><td>SNO-PI (rank 1)</td><td>0.5387</td><td>0.6629</td></tr><tr><td>Rank 2 entry</td><td>0.4985</td><td>0.6423</td></tr><tr><td>Rank 3 entry</td><td>0.4695</td><td>0.6273</td></tr><tr><td>Rank 4 entry</td><td>0.4657</td><td>0.6165</td></tr><tr><td>Organisers&#x27; reference (Super Dictionary)</td><td>0.4582</td><td>0.6234</td></tr><tr><td>Rank 6 entry</td><td>0.4214</td><td>0.5831</td></tr><tr><td>Rank 7 entry</td><td>0.4085</td><td>0.5621</td></tr><tr><td>Organisers&#x27; reference (FAISS + Qwen3)</td><td>0.2321</td><td>0.2160</td></tr></table>

Table 3: The public benchmark leaderboard on 18 October 2026.

Its macro score is 0.040 above the next-best entry and 0.081 (18% relative) above the organisers’ reference Super Dictionary solution, which follows the approach of the 2024 winner, KIRIs [20]. Table 4 traces the system’s development; local estimates tracked the official scores.

<table><tr><td>Development stage</td><td>Local dev / test</td><td>Hidden test macro / wtd</td></tr><tr><td>Three-encoder ensemble</td><td>.504 / .511</td><td>.500 / .628</td></tr><tr><td>+ six members, wider context</td><td>.532 / .530</td><td></td></tr><tr><td>+ retrained on all 272 notes</td><td></td><td>.535 / .658</td></tr><tr><td>+ Qwen3-8B member (final)</td><td></td><td>.539 / .663</td></tr></table>

Table 4: Development history; each row adds to the one above. The first row uses MLM and SNOMED pre-training. Local scores are the mean over 1,000 random 20-note subsets (trained on 174 notes). Hidden-test scores are the official results of our submissions (October 2026).

Labels still matter most. Curated knowledge narrows the gap but does not close it. Table 5 replaces each stage with an oracle: a perfect reranker over our existing candidates would add about 0.30 macro, against 0.06 for perfect retrieval and 0.12 for perfect spans. Concepts with 0–2 training examples make up 36% of concepts but 58% of the remaining gap, and the share of gold concepts we predict at all rises steeply with their first few labelled examples (Figure 4). Unseen concepts mostly fail before the reranker (no span or not retrieved), while rare-but-seen concepts fail inside it. Even a small increase in labels pays: aliases from 204 instead of 174 notes, with rerankers unchanged, added +0.004 test macro.

![](images/f35fb1cf927c5a553e78cf8dd74f0afb4b6ffab082b21d8d7863afb2ad661eea.jpg)  
Figure 4: Concept prediction coverage increases with training frequency: share of test gold concepts predicted at least once, by the concept’s number of training examples (earlier three-encoder system, local test, 300 random 20-note samples).

<table><tr><td></td><td>Dev</td><td>Test</td></tr><tr><td>Gold spans in span set (exact)</td><td>95.2%</td><td>94.2%</td></tr><tr><td>Retrieval recall@ 10 (seen / unseen)</td><td>.974 / .768</td><td>.969 / .766</td></tr><tr><td>Linked to gold, seen concepts</td><td>.888</td><td>.858</td></tr><tr><td>Linked to gold, unseen concepts</td><td>.678</td><td>.685</td></tr><tr><td>Non-gold spans correctly rejected</td><td>91.0%</td><td>91.6%</td></tr><tr><td>Macro IoU, as is</td><td>.532</td><td>.530</td></tr><tr><td>+ perfect reranker</td><td>.837</td><td>.824</td></tr><tr><td>+ perfect retrieval</td><td>.891</td><td>.882</td></tr><tr><td>+ perfect spans</td><td>1.000</td><td>1.000</td></tr></table>

Table 5: Stage-wise analysis of the six-member system (174 training notes), with oracles applied cumulatively.

## 5 When more knowledge did not help

Several attempts to inject further curated or pre-trained knowledge improved single models or intermediate metrics but not the final ensemble (Table 6). A recurring pattern is that gains outside SNOMED, or in recall alone, do not transfer to the linking decision. The hierarchy near misses in Table 6 also show a limit of exact-match scoring: predicting Renal calculus for gold Staghorn calculus scores zero, the same as an unrelated diagnosis, although the two are clinically close. A hierarchy-aware metric would reflect clinical usefulness better.

## 6 Limitations

The system used the AU 2025-08-31 release of SNOMED CT, as the prescribed November 2025 International release could not be obtained; 100 training concepts are inactive in it. With about 20 hidden notes, differences of a few thousandths of macro IoU are within noise. Some error stems from note-level annotation conventions unresolvable from the text.

## 7 Conclusion

With labels for approximately 3% of eligible concepts, SNOMED CT entity linking is a low-resource problem inside a knowledge-rich domain. SNO-PI leads the benchmark by drawing on unlabelled clinical text, the curated synonyms and structure of SNOMED CT, and large pretrained models at every stage, and this knowledge is most valuable precisely where labels are absent. The steep gains from each additional labelled example, however, suggest that targeted annotation of rare concepts remains the most effective next step.

<table><tr><td>Idea</td><td>Outcome</td></tr><tr><td>sap4: our own SapBERT retrained on UMLS 2025AB with harder negatives, then mention-tuned on gold spans harder negatives lowered</td><td>UMLS held-out acc@169.2% → 80.8%, but almost all of the gain is outside SNOMED, and unseen-concept recall. Mention tuning gave the best retriever measured (R@10 .955 → .962; unseen .738 → .785), yet only +.001 / +.002 in the ensemble.</td></tr><tr><td>SNOMED is-a hierarchy encoder (HiT [21]) as retriever</td><td>Worse than SapBERT: R@ 10 .950 vs .955, unseen .716 vs .738.</td></tr><tr><td>Hierarchy-aware SNOMED pre-training loss (soft targets, accuracy unchanged. relation margins); pre-training on every in-scope concept</td><td>Within seed noise; unseen</td></tr><tr><td>KRISSBERT [22]: contextual retrieval as a second retriever, with source-agreement features</td><td>Recovers 366 of 1,058 SapBERT misses (only 85 unseen); +.005 per three-model ensemble, ≤0 in the six-model ensemble.</td></tr><tr><td>Hierarchy near-miss resolution (parent/child pooling, specificity preference)</td><td>Oracle +.085; every rule ≤ +.0005, as annotators split a phrase between neighbouring concepts.</td></tr></table>

Table 6: A selection of negative results (paired ∆ macro, dev / test).

Code availability. The author plans to release the code on GitHub after clean-up at https://github.com/ pisong314/sno-pi.

## Acknowledgements and disclosure

The author holds the academic title of Senior Lecturer at the Greater Brisbane Clinical School, The University of Queensland. This research was conceived, developed and conducted independently and was not funded or supervised by the University; the views expressed are the author’s own. This work used the MIMIC-IV-Note data under the PhysioNet Credentialed Health Data Use Agreement; no note text is reproduced here beyond short, non-identifying span strings. The author designed the system and directed all experiments. An AI assistant (Anthropic’s Claude, via Claude Code) was used for coding and for drafting this manuscript. The author reviewed all results and text and takes full responsibility for them.

## References

[1] R. Davidson, W. Hardman, G. Amit, et al. SNOMED CT entity linking challenge. J. Am. Med. Inform. Assoc., 32(9):1397–1406, September 2025. doi:10.1093/jamia/ocaf104.

[2] W. Hardman, M. Banks, R. Davidson, D. Truran, N. W. Ayuningtyas, H. Ngo, A. Johnson, T. Pollard. SNOMED CT entity linking challenge (version 1.2.1). PhysioNet, 2026. doi:10.13026/az93-vm52.

[3] SNOMED International. SNOMED CT. https://www. snomed.org.

[4] A. E. W. Johnson, L. Bulgarelli, L. Shen, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10:1, 2023.

[5] A. Johnson, T. Pollard, S. Horng, L. A. Celi, R. Mark. MIMIC-IV-Note: deidentified free-text clinical notes (version 2.2). PhysioNet, 2023.

[6] SNOMED International and DrivenData. SNOMED CT Entity Linking Benchmark. https://www.drivendata.org/ benchmarks/310/benchmark-snomed-ct/, 2026. Accessed 18 October 2026.

[7] O. Bodenreider. The Unified Medical Language System (UMLS): integrating biomedical terminology. Nucleic Acids Res., 32:D267– D270, 2004.

[8] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proc. NAACL, 2019.

[9] J. Lafferty, A. McCallum, F. Pereira. Conditional random fields: probabilistic models for segmenting and labeling sequence data. In Proc. ICML, 2001.

[10] F. Liu, E. Shareghi, Z. Meng, M. Basaldella, N. Collier. Selfalignment pretraining for biomedical entity representations. In Proc. NAACL, 2021.

[11] Y. Gu, R. Tinn, H. Cheng, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Trans. Comput. Healthcare, 3(1), 2021.

[12] E. Alsentzer, J. Murphy, W. Boag, et al. Publicly available clinical BERT embeddings. In Proc. Clinical NLP Workshop, 2019.

[13] S. Gururangan, A. Marasovic, S. Swayamdipta, et al. Don’t stop´ pretraining: adapt language models to domains and tasks. In Proc. ACL, 2020.

[14] F. Gallego, P. Ruas, F. M. Couto, F. J. Veredas. Enhancing crossencoders using knowledge graph hierarchy for medical entity linking in zero- and few-shot scenarios. Knowledge-Based Systems, 314:113211, 2025. doi:10.1016/j.knosys.2025.113211.

[15] M. Yasunaga, J. Leskovec, P. Liang. LinkBERT: pretraining language models with document links. In Proc. ACL, 2022.

[16] B. Warner, A. Chaffin, B. Clavié, et al. Smarter, better, faster, longer: a modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. arXiv:2412.13663, 2024.

[17] P. He, J. Gao, W. Chen. DeBERTaV3: improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In Proc. ICLR, 2023.

[18] A. Yang, et al. Qwen3 technical report. arXiv:2505.09388, 2025.

[19] E. J. Hu, Y. Shen, P. Wallis, et al. LoRA: low-rank adaptation of large language models. In Proc. ICLR, 2022.

[20] G. Amit, Y. Bilu, I. Girshovitz, C. Yanover. SNOMED CT entity linking challenge: first place solution (KIRIs). Winner documentation and code. DrivenData, 2024. https://github.com/ drivendataorg/snomed-ct-entity-linking.

[21] Y. He, Z. Yuan, J. Chen, I. Horrocks. Language models as hierarchy encoders. In Proc. NeurIPS, 2024.

[22] S. Zhang, H. Cheng, S. Vashishth, et al. Knowledge-rich selfsupervision for biomedical entity linking. In Findings of EMNLP, 2022.