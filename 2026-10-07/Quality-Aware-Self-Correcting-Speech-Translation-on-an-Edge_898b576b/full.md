# Quality-Aware Self-Correcting Speech Translation on an Edge Device

Zubair Ajmal Farooq   
School of CS and EE   
University of Surrey   
United Kingdom   
zubair5ajmal@gmail.com   
Diptesh Kanojia   
Institute for People-Centred AI   
University of Surrey   
United Kingdom   
d.kanojia@surrey.ac.uk

## Abstract

We present a fully offline speech-to-speech translation pipeline that runs on a Jetson Nano (4 GB) and corrects its own weak translations without retraining. A Whisper-tiny ASR feeds an Opus-MT translator; multilingual BERT cosine similarity acts as a Quality Estimation (QE) gate, triggering a secondary-pass correction when confidence falls below a pre-defined threshold τ. We compare three correction methods: QE reranking (M1), Minimum Bayes-Risk decoding (M2), and constrained beam search (M3). On 1,012 FLORES-200 sentences (English-Spanish), M2 at τ=0.90 produces statistically significant improvements over greedy decoding on BLEU (+0.67, p<0.001), ChrF (+0.51, p<0.001), and COMET (+0.0020 at N=3, p=0.002); M1 yields no significant gains, and M3 is significantly worse than baseline (p>0.99). Our central finding is that QE functions effectively as a gate but poorly as a ranker: removing the QE model from candidate selection (M1→M2) does not hurt quality and frees 680 MB from the critical path. Using a gain-to-edit ratio adapted from the post-editingeffort literature, we further show that smaller candidate pools (N=3) yield more surgical corrections with better semantic adequacy, while larger pools (N=10) maximise lexical reward. We release the system and demonstrate live translation across six language pairs.

## 1 Introduction

Speech translation increasingly relies on cloud services that expose audio, transcripts, and intent. For healthcare, legal, and low-bandwidth deployments this is unacceptable. Edge devices offer a privacypreserving alternative, but cascaded speech translation pipelines (ASR → MT → TTS) compound errors at each stage, and small models compatible with edge hardware are particularly susceptible. We ask: can a sub-\$150 edge device translate speech, detect its own weak translations, and correct them, fully offline, without retraining?

We answer affirmatively. Our system runs end-to-end on a Jetson Nano (4 GB) and uses a lightweight Quality Estimation (QE) model to trigger a second-pass correction over an N-best candidate pool. We evaluate three correction strategies under a 4 × 3 grid of QE thresholds and beam sizes, with paired bootstrap significance testing on 1,012 FLORES-200 sentences.

Our contributions are:

• A complete offline speech translation system with QE-triggered self-correction running on 4 GB of RAM (§3).

• Statistically significant quality improvements via Minimum Bayes-Risk reranking (M2) on BLEU, ChrF, and COMET, with paired bootstrap evidence (§6).

• The finding that QE serves as a reliable trigger but an unreliable candidate ranker, with practical implications for memory footprint on edge hardware (§6.4).

• An adapted gain-to-edit ratio (§6.3) that connects per-sentence COMET improvements to edit distance, providing an analytical lens on the lexical-vs-semantic trade-off observed across beam pool sizes.

• A latency analysis (§7) demonstrating that correction overhead on this hardware is dominated by model reload, not inference, and that a single-load architecture brings per-utterance latency to within a tolerable range for conversational translation aids (<30s steady state).

## 2 Related Work

Edge MT and on-device translation. End-toend on-device speech translation has typically required server-class hardware or compressed cascades (Radford et al., 2023; Tiedemann and Thottingal, 2020). Edge deployments of single-stage MT components are common (mobile translators,

USB sticks), but composed pipelines with ondevice quality control are rare. Our work targets the smallest publicly available NVIDIA edge SKU with 4 GB unified memory.

Quality Estimation for MT. Reference-free QE (Specia et al., 2018) estimates translation quality without parallel references. We use multilingual BERT cosine similarity as a lightweight QE signal; stronger neural QE models such as COMET-Kiwi (Rei et al., 2022b) offer better correlation with human judgement but were too large for our memory budget. We adopt the position that even a noisy QE signal can be useful as a gate, even where it fails as a ranker.

Self-correction and reranking. Minimum Bayes-Risk decoding (Kumar and Byrne, 2004) selects the candidate with highest peer-consensus utility from an N-best list, and has been shown to outperform maximum-likelihood decoding when paired with strong utility functions (Freitag et al., 2022). QE-based reranking selects the candidate with the highest reference-free quality score (Fernandes et al., 2022). Constrained beam search forces decoding to include specified terms (Post and Vilar, 2018), and has been used in domain adaptation and terminology enforcement. We apply all three to the self-correction setting and quantify their relative performance under bootstrap testing.

Edit-distance-based effort metrics. Editdistance measures of MT quality and effort date to the early 1990s (Frederking and Nirenburg, 1994; King, 1996). The Translation Edit Rate (TER) (Snover et al., 2006) and its human-targeted variant HTER (Snover et al., 2009; Specia and Farzindar, 2010) are the most widely used. Post-editing effort metrics including post-edit distance and time-to-edit (Daems et al., 2017) normalise quality or effort over edit counts to characterise human correction work. Our gain-to-edit ratio (§6.3) adapts this lineage to the automatic self-correction setting: rather than measuring human effort against a reference, we measure neural quality gain (COMET) per unit of edit distance between two MT versions of the same source.

## 3 System

The pipeline is a cascade of four components, each chosen for its memory footprint under a 4 GB unified-memory ceiling.

ASR. OpenAI Whisper-tiny (\~39 M parameters) transcribes English audio. We use the public Py-Torch implementation with greedy decoding.

MT. Helsinki-NLP Opus-MT en-es (\~78 M parameters) translates the transcript. First-pass decoding uses greedy search (N=1); second-pass correction uses beam search with $N \in \{ 3 , 5 , 1 0 \}$

QE. Multilingual BERT base (Devlin et al., 2019) produces sentence embeddings for source and target; we use cosine similarity as the QE score s ∈ [−1, 1]. mBERT was selected over COMET-Kiwi for memory reasons (\~680 MB vs. \~2 GB peak). We do not claim cosine similarity is the best QE signal; we claim it is adequate as a binary trigger.

TTS. eSpeak-ng synthesises Spanish output. Output latency is dominated by audio synthesis rather than text processing.

Memory architecture. Under sequential loading (one model resident at a time), each stage incurs a load/unload cost dwarfing inference. Under a single-load architecture all four models reside concurrently in \~3.6 GB, well within the 4 GB ceiling; per-utterance latency drops accordingly (§7).

Multilingual extension. Although our evaluation is English-Spanish, the system extends to six language pairs (en ↔ es, fr, de) by swapping the Opus-MT model and the TTS voice; live demonstration confirms operation across all pairs at comparable steady-state latency. We restrict quantitative evaluation to en-es for tractability and depth.

## 4 Self-Correction Methods

When the QE score s falls below threshold τ, the system generates N alternative translations and selects one via one of three methods.

M1: QE reranking. The N candidates are scored against the source using the same mBERT cosine measure used as the gate. The highestscoring candidate is selected.

M2: Minimum Bayes-Risk (MBR). Each candidate is scored by its mean ChrF similarity to the other N−1 candidates (peer consensus). The candidate with highest mean similarity is selected. MBR does not consult the source or the QE model after candidate generation.

M3: Constrained beam search (CBS). Content words longer than three characters in the greedy first-pass output are extracted as soft anchors. Beam search is repeated under a constrained decoding objective that boosts hypotheses containing these anchors; the resulting candidates are then reranked by mBERT QE.

## 5 Experimental Setup

Data. We use the FLORES-200 eng\_Latn → spa\_Latn devtest split: 1,012 sentences drawn from English Wikinews, balanced across topical domains. For the ASR component, we use FLEURS (Conneau et al., 2023) English audio with paired transcripts.

Grid. We sweep $\tau \in \{ 0 . 7 5 , 0 . 8 0 , 0 . 8 5 , 0 . 9 0 \}$ and $N ~ \in ~ \{ 3 , 5 , 1 0 \}$ across all three correction methods, producing 36 configurations. The greedy baseline (N=1, no correction) is computed once and used as the comparison point for every configuration.

Metrics. We report sacreBLEU (Post, 2018), ChrF (Popovic´, 2015), and COMET-22 (Rei et al., 2022a). Significance is assessed by paired bootstrap resampling (Koehn, 2004) with n=1000 resamples; we report 95% confidence intervals and one-tailed p-values for the hypothesis that the correction method improves over the baseline.

Hardware. All ASR/MT/QE runs were performed on an NVIDIA Jetson Nano 4 GB Developer Kit (Maxwell GPU, 128 CUDA cores, 4 GB LPDDR4) with a 40 mm PWM cooling fan. COMET evaluation was performed on Google Colab CPU (the model exceeds available memory on the Nano). Identical model weights and hypothesis files were used in both environments.

## 6 Results

## 6.1 Aggregate Quality

Table 1 reports the strongest configuration for each correction method at τ=0.90 together with bootstrap significance. M2 produces significant improvements on all three metrics; M1 produces no significant change; M3 is significantly worse than baseline.The full 4 × 3 grid across all(τ, N) combinations appears in Appendix A.

## 6.2 N-Trade-off: Lexical vs Semantic

A direct corollary of running the full grid is that different beam pool sizes win different metrics (Ta-

<table><tr><td>Method</td><td>∆BLEU</td><td>∆ChrF</td><td>∆COMET</td></tr><tr><td>M1 (QE-rerank)</td><td>-0.14</td><td>-0.11</td><td>-0.0004</td></tr><tr><td>M2 (MBR)</td><td> $\mathbf { + 0 . 6 7 ^ { * * * } }$ </td><td> ${ \bf + 0 . 5 1 ^ { * * * } }$ </td><td>+0.0011</td></tr><tr><td>M3 (CBS)</td><td> $- 0 . 9 6 ^ { \dagger }$ </td><td> $- 0 . 8 6 ^ { \dagger }$ </td><td> $- 0 . 0 0 6 1 ^ { \dagger }$ </td></tr></table>

Table 1: Aggregate deltas over greedy baseline at $\tau { = } 0 . 9 0$ , N=10. Paired bootstrap, n=1000. ∗∗∗ $p { < } 0 . 0 0 1 . \ ^ { \dagger } ;$ significantly worse than baseline $\mathrm { ( } p { > } 0 . 9 9 $ one-tailed). ∆COMET for M2 at N=10 is not individually significant (p=0.107); see §6.2.

<table><tr><td>N</td><td>∆BLEU</td><td>∆ChrF</td><td>△COMET</td></tr><tr><td>3</td><td> $+ 0 . 5 9 ^ { * * * }$ </td><td> $+ 0 . 3 7 ^ { * * * }$ </td><td>+0.0020* **</td></tr><tr><td>5</td><td> $+ 0 . 4 7 ^ { * * }$ </td><td> $+ 0 . 3 2 ^ { * * }$ </td><td>+0.0012</td></tr><tr><td>10</td><td> $\mathbf { + 0 . 6 7 ^ { * * * } }$ </td><td> $\mathbf { + 0 . 5 1 ^ { * * * } }$ </td><td>+0.0011</td></tr></table>

Table 2: M2 (MBR) corrections at τ=0.90 across beam pool sizes. Paired bootstrap, n=1000. <sup>∗∗</sup>: p<0.01. ∗∗∗ $p { < } 0 . 0 0 1$ . ∆COMET at N=5 and $N { = } 1 0$ are not significant at α=0.05 (one-tailed $\scriptstyle p = 0 . 0 8 2$ and 0.107).

ble 2). $\operatorname { A t } { \tau } { = } 0 . 9 0 \colon$

• BLEU and ChrF peak at N=10 (+0.67 and +0.51, both p<0.001).

• COMET peaks at N=3 (+0.0020, p=0.002); N=5 and N=10 produce smaller, nonsignificant gains.

The full grid (Appendix A) shows that this pattern holds robustly: M2 outperforms M1 and M3 at every combination where correction actually fires $( \tau \geq 0 . 8 5 )$

This pattern is consistent with the structure of MBR: the consensus candidate is chosen for maximal peer agreement on lexical surface, not semantic adequacy against the source. With N=10, the consensus has more room to drift toward a safe, lexically common phrasing that rewards n-gram overlap against the reference but may dilute semantic precision. With N=3, the candidate pool is small and the consensus stays closer to the original greedy output, preserving semantic adequacy at the cost of lexical novelty.

## 6.3 Gain-to-Edit Ratio

To make the lexical-versus-semantic trade-off observable per sentence rather than only at corpus level, we adapt the edit-distance lineage from postediting-effort metrics (Snover et al., 2009; Specia and Farzindar, 2010; Daems et al., 2017) to the automatic self-correction setting. We define the gain-to-edit ratio (GER) per triggered sentence as

<table><tr><td>N</td><td>GER (median)</td><td>Triggered</td></tr><tr><td>3</td><td>+0.0053</td><td>485</td></tr><tr><td>5</td><td>+0.0023</td><td>542</td></tr><tr><td>10</td><td>+0.0048</td><td>580</td></tr></table>

Table 3: Median per-sentence GER for M2 at $\tau { = } 0 . 9 0$ Triggered = number of sentences where QE fired and correction was applied (out of 1,012).

$$
\mathrm { G E R } = { \frac { \Delta \mathrm { C O M E T } } { \mathrm { T E R } ( \mathbf { b a s e l i n e } , \mathrm { c o r r e c t e d } ) } }\tag{1}
$$

where ∆COMET is the per-sentence COMET improvement of the corrected output over the baseline and TER is the Translation Edit Rate (Snover et al., 2006) between the two MT versions. GER is not a new metric; it is an adapted ratio whose components and form derive from prior work. We use it as an analytical lens. A higher GER indicates surgical correction (small edit, real semantic gain); a lower GER indicates blunt rewrites that perturb the output without improving adequacy.

Table 3 reports median GER for M2 across N. The ranking aligns with the COMET column of Table 2: N=3 has the highest median GER, N=5 the lowest, N=10 intermediate. The mean GER for N=10 is negative (−0.0019) while its median is positive (+0.0048); this asymmetry indicates a long tail of large, low-gain edits at larger N. Median is the more robust summary.

GER thus surfaces a property that aggregate BLEU/ChrF/COMET cannot. The lexical and semantic metrics disagree about which N is best; GER explains why both are right: at smaller N, MBR makes targeted edits that preserve semantic adequacy; at larger N, MBR is more willing to substantially rewrite the output, which can win on lexical surface but introduces noise.

Figure 1 integrates the four metrics (BLEU, ChrF, COMET, GER median) for M2 at τ=0.90 across the three beam pool sizes, making the permetric winners explicit.

## 6.4 QE as Gate, Not Ranker

A second observation from Table 1 is that removing the QE model from the candidate-selection step while retaining it as the trigger which is the difference between M1 and M2 does not degrade quality. M2 in fact outperforms M1 on every metric. This is consistent with prior findings that reference-free

Different N values win different metrics  
![](images/d88514af29bf912d9a39ca0ff756ed5417037642cc1b0f44b13c3299b6f19227.jpg)  
Figure 1: Improvement over greedy baseline across four metrics for M2 (MBR) at τ=0.90, with paired bootstrap significance (p<0.05). ∆COMET and GER median are scaled ×100 for joint visibility. Lexical metrics (BLEU, ChrF) peak at N=10; semantic metric (COMET) and per-sentence efficiency (GER median) peak at N=3.

QE signals at the segment level are noisier discriminators among similar candidates than between clearly good and clearly bad translations (Fernandes et al., 2022). The practical implication for edge deployment is direct: once the gate has fired, the QE model can be unloaded for the duration of MBR selection, freeing \~680 MB of resident memory.

This finding generalises beyond our specific system. Wherever a cascaded MT pipeline uses a lightweight QE model both for triggering and for ranking, our results suggest splitting these roles: keep the QE model as a trigger gate, but let the candidates themselves vote on the selection.

## 6.5 M3 Failure

CBS underperforms baseline on every metric and at every N, with $p { > } 0 . 9 9$ in the one-tailed bootstrap test. We attribute this to a structural problem rather than implementation error: the anchors are extracted from the greedy first-pass output, the very output that the QE gate has just flagged as low-confidence. Constraining the second pass to preserve those anchors forces the system to retain whatever errors the trigger fired on. CBS would likely be useful in a different role (terminology enforcement, named entity preservation) but is not appropriate as a quality-driven correction strategy on noisy first-pass output.

## 7 Latency Analysis

Per-utterance latency on the Jetson Nano under sequential loading averages approximately 82 s, with model reload accounting for the majority of stage cost. Correction adds approximately 23.9 s, dominated by the load/unload cycle around MT Pass 2 rather than by the beam search itself.

Under a single-load architecture, all four models remain resident for the duration of the session. We measured this configuration on the Nano: peak resident memory was 3.6 GB (within the 4 GB ceiling), and per-utterance latency for one ∼10 s en-es utterance was approximately 41 s (including first-utterance load amortisation). The MBR selfcorrection overhead reduces from 23.9 s to 13.2 s once model reload is removed from the critical path.

![](images/8351c4ab23fa7a90191a8277217b2d26fa0661c5f558b5336724fda459d76a88.jpg)  
Figure 2: Per-stage latency comparison on the Jetson Nano. Sequential-load (top, ∼82 s) is dominated by perstage model reload; single-load resident (bottom, ∼41 s) eliminates reload at the cost of higher peak memory.

Critically, the bottleneck under the original architecture is RAM, not algorithmic. The Nano’s 4 GB ceilingforced the sequential architecture: under those constraints, the user-perceived latency reflects memory bandwidth rather than the computational cost of self-correction. Hardware with a modestly larger memory budget (e.g., Jetson Orin Nano 8 GB, Raspberry Pi 5 with swap, mobileclass SoCs) would run the same pipeline at nearreal-time latency without any change to the algorithms reported here.

## 8 Discussion and Limitations

Our findings can be summarised as follows. (1) QE-triggered MBR self-correction produces statistically significant improvements on lexical metrics and on COMET at N=3. (2) The QE model is valuable as a trigger but not as a ranker, which has direct memory-footprint consequences for edge deployments. (3) The lexical-versus-semantic trade-off across beam pool sizes is explainable in terms of MBR’s optimisation target and is observable at sentence level through GER. (4) CBS is structurally unsuited to correction on flagged output. (5) Edge latency is RAM-bound, not compute-bound.

Limitations. First, evaluation is restricted to English-Spanish; we demonstrate but do not evaluate the system across the additional five language pairs supported by the same architecture. Second, the QE signal is multilingual BERT cosine similarity, which is a weak proxy for semantic adequacy. Stronger reference-free QE models (COMET-Kiwi) would likely produce a more discriminative gate but exceed available memory; quantising or distilling such models for edge use is an open direction. Third, FLORES-200 devtest is a writtentext benchmark with controlled domains; performance under spontaneous speech with disfluencies, code-switching, or noisy acoustic conditions has not been measured. Fourth, COMET-22 is a learned metric whose own biases propagate into GER; using multiple independent quality estimators would strengthen claims about semantic adequacy.

Conclusion. An edge device can translate speech, detect its own weak translations, and correct them, fully offline, without retraining. The required machinery (MBR self-correction over a small candidate pool, QE used as a gate rather than as a ranker, models held resident under a modest memory budget) is straightforward to implement and produces measurable quality improvements. The principal obstacle to deploying conversational ondevice speech translation today is not the cost of self-correction algorithms but the cost of RAM.

## References

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. 2023. FLEURS: Fewshot learning evaluation of universal representations of speech. In IEEE Spoken Language Technology Workshop.

Joke Daems, Sonia Vandepitte, Robert J. Hartsuiker, and Lieve Macken. 2017. Identifying the machine translation error types with the greatest impact on post-editing effort. Frontiers in Psychology, 8:1282.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186.

Patrick Fernandes, António Farinhas, Ricardo Rei, José G.C. de Souza, Perez Ogayo, Graham Neubig,

and André F.T. Martins. 2022. Quality-aware decoding for neural machine translation. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 1396–1412.

Robert Frederking and Sergei Nirenburg. 1994. Three heads are better than one. In Proceedings of the Fourth Conference on Applied Natural Language Processing, pages 95–100.

Markus Freitag, David Grangier, Qijun Tan, and Bowen Liang. 2022. High quality rather than high model probability: Minimum Bayes-Risk decoding with neural metrics. Transactions of the Association for Computational Linguistics, 10:811–825.

Margaret King. 1996. Evaluating natural language processing systems. Communications of the ACM, 39(1):73–79.

Philipp Koehn. 2004. Statistical significance tests for machine translation evaluation. In Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, pages 388–395.

Shankar Kumar and William Byrne. 2004. Minimum Bayes-Risk decoding for statistical machine translation. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, pages 169–176.

Maja Popovic. 2015. chrF: character n-gram F-score´ for automatic MT evaluation. In Proceedings ofthe Tenth Workshop on Statistical Machine Translation, pages 392–395.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186– 191.

Matt Post and David Vilar. 2018. Fast lexically constrained decoding with dynamic beam allocation for neural machine translation. In Proceedings of the 2018 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1314–1324.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning.

Ricardo Rei, José G.C. de Souza, Duarte Alves, Chrysoula Zerva, Ana C. Farinha, Taisiya Glushkova, Alon Lavie, Luisa Coheur, and André F.T. Martins. 2022a. COMET-22: Unbabel-IST 2022 submission for the metrics shared task. In Proceedings of the Seventh Conference on Machine Translation, pages 578–585.

Ricardo Rei, Marcos Treviso, Nuno M. Guerreiro, Chrysoula Zerva, Ana C. Farinha, Christine Maroti, José G.C. de Souza, Taisiya Glushkova, Duarte M. Alves, Alon Lavie, Luisa Coheur, and André F.T. Martins. 2022b. COMETKiwi: IST-unbabel 2022 submission for the quality estimation shared task. In Proceedings ofthe Seventh Conference on Machine Translation.

Matthew Snover, Bonnie Dorr, Richard Schwartz, Linnea Micciulla, and John Makhoul. 2006. A study of translation edit rate with targeted human annotation. In Proceedings of the 7th Conference of the Associationfor Machine Translation in the Americas: Technical Papers, pages 223–231.

Matthew Snover, Nitin Madnani, Bonnie Dorr, and Richard Schwartz. 2009. Fluency, adequacy, or HTER? exploring different human judgments with a tunable MT metric. In Proceedings of the Fourth Workshop on Statistical Machine Translation, pages 259–268.

Lucia Specia and Atefeh Farzindar. 2010. Estimating machine translation post-editing effort with HTER. In Proceedings of the Second Joint EM+/CNGL Workshop “Bringing MT to the User: Research on Integrating MT in the Translation Industry”, pages 33–41.

Lucia Specia, Carolina Scarton, and Gustavo Henrique Paetzold. 2018. Quality Estimation for Machine Translation. Morgan & Claypool.

Jörg Tiedemann and Santhosh Thottingal. 2020. OPUS-MT – building open translation services for the world. In Proceedings of the 22nd Annual Conference of the European Associationfor Machine Translation, pages 479–480.

## A Full Grid Search results

Table 4 reports the complete 4 × 3 grid search over QE thresholds $\tau \in \{ 0 . 7 5 , 0 . 8 0 , 0 . 8 5 , 0 . 9 0 \}$ and beam sizes $N \in \{ 3 , 5 , 1 0 \}$ for all three correction methods on 1,012 FLORES-200 sentences. The internal baseline values are BLEU 26.09 and ChrF 54.85; parenthesised values are deltas from these baselines. Trigger rates indicate the proportion of sentences for which QE fired correction at each threshold.

<table><tr><td>T</td><td>N</td><td> $\mathrm { T r i g } \%$ </td><td>BLEU-M1</td><td>BLEU-M2</td><td>BLEU-M3</td><td>ChrF-M1</td><td>ChrF-M2</td><td>ChrF-M3</td></tr><tr><td>0.75</td><td>3</td><td>0.1%</td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td><td> $5 4 . 8 7 \ : ( + 0 . 0 2 ) $ </td></tr><tr><td>0.75</td><td>5</td><td>0.1%</td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $5 4 . 8 7 \ : ( + 0 . 0 2 ) $ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td><td> $5 4 . 8 7 \ : ( + 0 . 0 2 ) $ </td></tr><tr><td>0.75</td><td>10</td><td>0.1%</td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $5 4 . 8 6 ( + 0 . 0 1 ) $ </td><td> $5 4 . 8 7 \ : ( + 0 . 0 2 ) $ </td><td> $5 4 . 8 6 ( + 0 . 0 1 ) $ </td></tr><tr><td>0.80</td><td>3</td><td>1.3%</td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 1 \ : ( + 0 . 0 2 )$ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td><td> $5 4 . 8 7 \ : ( + 0 . 0 2 ) $ </td></tr><tr><td>0.80</td><td>5</td><td>1.3%</td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 0 9 \left( + 0 . 0 0 \right)$ </td><td> $2 6 . 1 1 \ : ( + 0 . 0 2 )$ </td><td> $5 4 . 8 6 ( + 0 . 0 1 ) $ </td><td> $5 4 . 8 4 \ : ( - 0 . 0 1 )$ </td><td> $5 4 . 8 7 \ : ( + 0 . 0 2 ) $ </td></tr><tr><td>0.80</td><td>10</td><td>1.3%</td><td> $2 6 . 0 9 \left( + 0 . 0 0 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $2 6 . 1 0 \left( + 0 . 0 1 \right)$ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td><td> $5 4 . 8 7 \ : ( + 0 . 0 2 ) $ </td><td> $5 4 . 8 6 ( + 0 . 0 1 ) $ </td></tr><tr><td>0.85</td><td>3</td><td>22.7%</td><td>26.23 (+0.14)</td><td>26.34 (+0.25)</td><td> $2 6 . 2 2 \ : ( + 0 . 1 3 )$ </td><td> $5 4 . 9 3 \ : ( + 0 . 0 8 )$ </td><td> $5 4 . 9 9 \ : ( + 0 . 1 4 ) $ </td><td> $5 4 . 9 5 \ : ( + 0 . 1 0 )$ </td></tr><tr><td>0.85</td><td>5</td><td>22.7%</td><td>26.14 (+0.05)</td><td> $2 6 . 2 6 ( + 0 . 1 7 )$ </td><td> $2 6 . 2 2 \ : ( + 0 . 1 3 )$ </td><td> $5 4 . 8 4 \ : \mathrm { ( - 0 . 0 1 ) }$ </td><td> $5 4 . 9 4 \ : ( + 0 . 0 9 )$ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td></tr><tr><td>0.85</td><td>10</td><td>22.7%</td><td> $2 6 . 1 1 \ : ( + 0 . 0 2 )$ </td><td> $2 6 . 4 0 ( + 0 . 3 1 )$ </td><td> $2 6 . 0 2 \ : ( - 0 . 0 7 )$ </td><td> $5 4 . 8 4 \ : ( - 0 . 0 1 )$ </td><td> $5 5 . 0 4 ( + 0 . 1 9 )$ </td><td> $5 4 . 7 8 \ : ( - 0 . 0 7 )$ </td></tr><tr><td>0.90</td><td>3</td><td>86.2%</td><td> $2 6 . 4 5 \ : ( + 0 . 3 6 )$ </td><td> $2 6 . 6 8 \ : ( + 0 . 5 9 )$ </td><td> $2 5 . 9 4 \ : ( - 0 . 1 5 )$ </td><td> $5 5 . 0 4 ( + 0 . 1 9 )$ </td><td> $5 5 . 2 2 \ : ( + 0 . 3 7 )$ </td><td> $5 4 . 6 5 \ : ( - 0 . 2 0 )$ </td></tr><tr><td>0.90</td><td>5</td><td>86.2%</td><td> $2 6 . 1 6 \ : ( + 0 . 0 7 )$ </td><td> $2 6 . 5 6 ( + 0 . 4 7 )$ </td><td> $2 5 . 6 8 ( - 0 . 4 1 )$ </td><td> $5 4 . 8 5 \ : ( + 0 . 0 0 ) $ </td><td> $5 5 . 1 7 \ ( + 0 . 3 2 )$ </td><td> $5 4 . 3 9 \ : ( - 0 . 4 6 )$ </td></tr><tr><td>0.90</td><td>10</td><td>86.2%</td><td> $2 5 . 9 6 \left( - 0 . 1 3 \right)$ </td><td> ${ \bf 2 6 . 7 6 } \left( + 0 . 6 7 \right)$ </td><td> $2 5 . 1 4 \ : ( - 0 . 9 5 )$ </td><td> $5 4 . 7 4 \ : ( - 0 . 1 1 )$ </td><td> ${ \pmb 5 } { \pmb 5 } . 3 { \pmb 5 } \left( + 0 . 5 0 \right)$ </td><td> $5 3 . 9 9 \left( - 0 . 8 6 \right)$ </td></tr></table>

Table 4: Full $4 \times 3$ grid search results over 1,012 FLORES-200 sentences. Internal baselines: BLEU 26.09, ChrF 54.85. M1 = QE-Rerank, M2 = MBR, M3 = CBS. Parenthesised values are deltas from baseline. Bold marks the strongest M2 configuration reported in the main body.