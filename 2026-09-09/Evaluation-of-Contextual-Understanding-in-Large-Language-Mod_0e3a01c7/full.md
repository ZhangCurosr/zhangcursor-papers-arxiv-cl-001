# Evaluation of Contextual Understanding in Large Language Models

Subavarshana Arumugam <sup>\*</sup> <sup>1</sup> Mamta Nallaretnam <sup>\*</sup> <sup>1</sup> Kithuni Wickramasinghe <sup>\*</sup> <sup>1</sup> Chamath Gunapala <sup>\*</sup> <sup>1</sup> Pragatheeswaran Vipulanandan <sup>2</sup> Uthayasanker Thayasivam <sup>1</sup> Kamal Premaratne <sup>2</sup>

## Abstract

Large Language Models (LLMs) demonstrate impressive performance across diverse NLP tasks, yet their ability to exhibit genuine contextual understanding remains uncertain. Traditional evaluation metrics such as perplexity, BiLingual Evaluation Understudy (BLEU), or surface-level accuracy fail to reveal how well LLMs extract, integrate, and reason over contextual information–a gap particularly critical in question answering, where models must align responses with contextually grounded knowledge rather than memorized associations. We propose a novel knowledge graph-based evaluation framework introducing Semantic Structural Similarity for KGs (S3KG), a hybrid similarity measure integrating structural and semantic similarity into a continuous evaluation score, alongside a diagnostic framework for categorizing reasoning errors. To validate this pipeline, we evaluate S3KG against established metrics on a curated question-answer (QA) benchmark, demonstrating its effectiveness in measuring correctness, faithfulness, and interpretability in LLM-generated responses.

## 1. Introduction

Large language models (LLMs) have transformed natural language processing, demonstrating strong performance on tasks ranging from open-domain question answering (QAing) to complex reasoning (Izacard & Grave, 2021; Wei et al., 2022). Yet a fundamental question persists: do these models genuinely understand the context they process, or do they exploit statistical correlations to produce plausible outputs without true comprehension? This is especially consequential in high-stakes domains such as medical and legal QAing, where responses must be grounded in the provided context rather than memorized priors.

## 2. Related Work

## 2.1. LLM Evaluation and Contextual Understanding

Existing evaluation metrics–perplexity, BLEU (Papineni et al., 2002), and token-level accuracy–measure surfacelevel fluency and overlap but fail to capture relational depth or factual faithfulness. Zhu et al. (2024) benchmark LLMs across several structured tasks, including co-reference resolution and dialogue state tracking, finding that models capture general context patterns but fail in fine-grained interpretation. Yan et al. (2024) probe reasoning fidelity by manipulating in-context examples through logical substitutions, revealing systematic failures in formal reasoning that surface-level metrics do not expose.

On the construction side, LLM-driven KG frameworks such as GraphRAG (Edge et al., 2024) and KEA (Haskins & Adams, 2025) have demonstrated that high-quality relational triplets can be extracted with minimal supervision, while hallucination-oriented evaluators like GraphEval (Sansford et al., 2024) adopt few-shot prompting with instruction tuning to encourage label consistency across graphs. Despite these advances, no existing metric jointly captures relational structure and semantic faithfulness in a continuous interpretable score.

## 2.2. Graph Similarity Methods

Embedding-based approaches such as TransE (Bordes et al., 2013) and RotatE (Sun et al., 2019) learn entity representations over fixed vocabularies, making them illsuited for cross-graph comparison where entity sets are disjoint. The WL graph kernel (Shervashidze et al., 2011) offers training-free structural comparison but treats labels as opaque symbols, penalizing semantically equivalent but lexically distinct terms. The WWL kernel (Togninalli et al., 2019) partially addresses this with Wasserstein node-distribution comparison, yet still lacks semantic label grounding. KEA (Haskins & Adams, 2025) combines WL kernels with SBERT (Reimers & Gurevych, 2019) clustering to bridge the lexical gap, but its cluster-merge is lossy because merging semantically close labels discards fine-grained relational distinctions. Taken together, these methods either treat labels as opaque symbols or rely on lossy clustering, leaving a clear need for a similarity measure that preserves fine-grained relational distinctions while handling semantic variation.

## 2.3. Our Contributions

Our work makes three main contributions.

• Semantic structural similarity for KGs (S3KG) is a hybrid structural–semantic similarity metric that converts LLM responses and reference answers into knowledge graph (KG) triplets and produces a continuous interpretable evaluation score.

• The Contextual Understanding Score (CUS) is a model-level aggregate of two complementary dimensions: factual accuracy and contextual faithfulness, enabling cross-model comparison across benchmarks.

• The Triplet Analyzing Unit (TAU) is a diagnostic component that categorizes reasoning errors at the triplet level for fine-grained behavioral analysis.

## 3. Methodology

![](images/20d3bcd7c48d653580a13ddd141e80e13603acd8866d77629c7d6e1906678d26.jpg)  
Figure 1. KG-based evaluation pipeline. Given a QA pair, three KGs are constructed from the LLM response, gold answer, and supporting context, then compared via S3KG to produce GoldSim, CtxSim, and CUS.

As illustrated in Fig.1, initially the candidate LLM is prompted with the question alongside its relevant context. The generated response is collected as the LLM answer, which together with the ground truth answer and the supporting context is used for the knowledge graph construction.

## 3.1. Knowledge Graph Construction

Three KGs are constructed per QA instance from the gold answer, model-generated response, and supporting context, using the few-shot prompting strategy with instruction tuning from Sansford et al. (2024) and Haskins & Adams (2025) applied uniformly across all three sources to ensure a consistent label space. Following extraction, all entity and relation labels are normalized via lowercasing and lemmatization to eliminate residual surface-level variation, ensuring the three graphs are structurally compatible for meaningful comparison.

## 3.2. S3KG: Semantic Structural Similarity for KGs

S3KG computes similarity between two graphs $G _ { 1 }$ and $G _ { 2 }$ at two levels: at the triplet level, individual facts are matched for fine-grained correspondence; at the graph level, overall topology is compared. Both structural and semantic similarity are considered at each level. The full pipeline is illustrated in Appendix A (Figure 2).

Triplet-Level Matching. Each triplet $( h , r , t )$ is serialized into a natural language (NL) string and encoded with SBERT (Reimers & Gurevych, 2019) (paraphrase-MPNet-base-v2). For each triplet in $\mathcal { T } _ { 1 }$ , the most semantically similar triplet in $\mathcal { T } _ { 2 }$ is identified by cosine similarity, forming a filtered set $\widehat { T } _ { 2 } \subseteq \mathcal { T } _ { 2 }$ that anchors the structural comparison to semantically relevant content.

Soft Label Alignment. Standard Weisfeiler–Lehman (WL) kernels treat lexically distinct but semantically equivalent labels as entirely disjoint. S3KG resolves this by independently aligning entity and relation labels: each label ℓ in $G _ { 1 }$ is replaced by its closest counterpart in $G _ { 2 }$ (by SBERT cosine similarity) whenever the similarity exceeds a threshold (we use 0.65); otherwise it is left unchanged. Entity and relation labels are aligned separately to prevent cross-type collisions, yielding aligned graphs $\widetilde { G } _ { 1 }$ and $ { \widetilde { G } } _ { 2 }$

Structural Similarity via WL Kernel. The normalized WL kernel (Shervashidze et al., 2011) score, with multiple iterations (we use 5) to capture multi-hop neighbourhood patterns over the aligned graphs) yields structural similarity as

$$
\mathrm { S } _ { \mathrm { W L } } ( \widetilde { G } _ { 1 } , \widetilde { G } _ { 2 } ) = \frac { K ( \widetilde { G } _ { 1 } , \widetilde { G } _ { 2 } ) } { \sqrt { K ( \widetilde { G } _ { 1 } , \widetilde { G } _ { 1 } ) \cdot K ( \widetilde { G } _ { 2 } , \widetilde { G } _ { 2 } ) } } .\tag{1}
$$

Semantic Similarity via SBERT Mean-Pool. Each graph is represented by the mean SBERT embedding of its triples. The semantic similarity $\mathbf { S } _ { \mathrm { S B E R T } } ( T _ { 1 } , \widehat { T } _ { 2 } )$ is the cosine between these pooled representations, clipped to [0, 1] to discard negative correlations.

Final Score. The structural and semantic scores are blended via mixing coefficient $\alpha \in [ 0 , 1 ]$ (we use $\alpha = 0 . 5 )$ as

$$
\mathrm { S } _ { \mathrm { S 3 K G } } = \left( 1 - \alpha \right) \mathrm { S } _ { \mathrm { W L } } + \alpha \mathrm { S } _ { \mathrm { S B E R T } } .\tag{2}
$$

## 3.3. Triplet Analyzing Unit (TAU)

For the 5% of lowest-similarity QA pairs, a triplet analysis unit is applied to examine how the LLM-generated KG diverges from the ground truth. Each triplet is converted into an NL sentence and encoded using a Sentence Transformer, and cosine similarity is computed between ground truth and LLM triplets, with scores above a threshold (we use 0.76) treated as aligned. After removing aligned triplets, the remaining pairs are categorized into four error types: relation wrong (entities match but relation differs), entity wrong (relation aligns but entities differ), extra triplets (hallucinated by the LLM), and missing triplets (not captured by the LLM).

## 3.4. Datasets and Models

Two datasets are employed in this study for their longform answer coverage: PubMedQA (Jin et al., 2019), comprising biomedical QA pairs drawn from research articles, and MesaQA (Wang et al., 2025), comprising consumer healthcare QA pairs requiring multi-span evidence integration. Three instruction-tuned 7B-parameter models are evaluated: Llama-2-7b-chat-hf, Gemma-7b-it, and Mistral-7B-Instruct-v0.2, with Falcon-7B included as a baseline. Responses are collected across a temperature sweep of {0.0, 0.3, 0.7, 1.0} to analyze the effect of generation stochasticity on contextual faithfulness. Full temperature sensitivity results appear in Appendix C.

## 4. Experiments and Results

Our evaluation targets the full KG-based LLM evaluation pipeline (Fig. 1), which comprises two core components: (1) KG construction, which extracts structured representations from the LLM response, gold answer, and supporting context; and (2) S3KG similarity module, which compares these graphs to produce the Comparative LLM Understanding Score (CUS). To rigorously assess S3KG’s similarity scoring in isolation, we first benchmark it independently across nine semantic equivalence datasets in §4.1, before evaluating the complete pipeline on QA benchmarks in §4.3.

## 4.1. S3KG Benchmark Evaluation

Datasets and Task Formulation. We evaluate on nine datasets spanning three structural categories, each cast as a binary semantic equivalence task (N=400, balanced), with performance measured by maximum F1 via threshold sweep. The short-text category comprises MRPC (Dolan & Brockett, 2005), PAWS-Wiki (Zhang et al., 2019), and STS12 (Agirre et al., 2012) (10–22 words). The KGperturbed paragraph category comprises five datasets (SK-Codex 400, SK-Combined, SK-FindKG, SK-GloBI, SK-Oregano) built by perturbing entity relationships in

KG-derived paragraphs (69–126 words) across encyclopedic (Safavi & Koutra, 2021), financial (Li & Sanna Passino, 2024), biological (Poelen et al., 2014), and food ontology (Boudin et al., 2023) domains. The Wikipedia Entity-Swap dataset (399 pairs) serves as an anti-circularity control, constructed via NLP-based perturbations with no KG involvement (Sennrich et al., 2016; Wei & Zou, 2019). Baselines include ROUGE-1/2/L (Lin, 2004), BLEU (Papineni et al., 2002), BERTScore (Zhang et al., 2020), MiniLM (Wang et al., 2020), and Sentence-T5- base (Ni et al., 2022). S3KG is evaluated with $\alpha \in$ $\{ 0 . 0 , 0 . 1 , \ldots , 1 . 0 \}$ ; we report the best-performing α per dataset alongside AUROC, with the full sweep in Appendix B.

Results. Table 1 reports F1 and AUROC across all benchmarks. S3KG achieves top-1 or top-2 F1 on 7 of 9 datasets. On KG-perturbed paragraphs, structural signals are most valuable: S3KG gains up to +7.6 F1 points (SK-GloBI) and correctly captures relational role distinctions where ROUGE-1 collapses to near-random (PAWS-Wiki AUC = 0.490, S3KG F1 = 0.766). The exception is SK-FindKG, where financial vocabulary introduces KG extraction noise and Sentence-T5-base leads (F1 = 0.848), highlighting extraction quality as a bottleneck. On short texts, dense models dominate due to sparse relational structure (Sentence-T5- base: $\mathrm { M R P C } = 0 . 7 6 6 , \mathrm { S T S 1 2 } = 0 . 8 5 3 )$ . On the Wikipedia Entity-Swap control, S3KG achieves the strongest result (F1 = 0.872 vs MiniLM 0.821), confirming gains are not an artifact of the KG pipeline. Optimal α varies by dataset and is further discussed in Appendix B.

## 4.2. TAU Evaluation

Table 2 illustrates the TAU performace, on a manually annotated subset of the MesaQA and PubMed datasets, where aligned triplet pairs between ground-truth and LLMgenerated KGs were labeled by three medical students, achieving strong agreement (pairwise F1: 0.97 − −0.99). Our method uses sentence-level semantic similarity for alignment, while the KEA baseline relies on componentwise matching. Our approach improves Micro F1 (0.895 vs 0.836) and Macro F1 (0.782 vs 0.622), driven by a +34.8% recall gain with a modest drop in precision. This enables robust alignment of semantically equivalent relations despite lexical variation, which are often missed by KEA. Overall, these results demonstrate that our method outperforms KEA in TAU performance.

Table 1. F1 / AUROC scores across all benchmark datasets. Bold indicates best value per column. S3KG uses the best-performing α per dataset selected by grid search (full sweep in Appendix B). <sup>†</sup>ROUGE-1 excluded from Wiki Swap due to surface-form artefact. Best α: MRPC 0.3, PAWS 0.5, STS12 0.1, C400 0.5, Comb. 0.5, Find 0.0, GloBI 0.6, Oreg. 0.4, W-Swap 0.1.
<table><tr><td rowspan="2"></td><td colspan="3">Short Text</td><td colspan="5">KG-Perturbed Paragraphs</td><td>Anti-Circ.</td></tr><tr><td>MRPC</td><td>PAWS</td><td>STS12</td><td>C400</td><td>Comb.</td><td>Find</td><td>GloBI</td><td>Oreg.</td><td>W-Swap</td></tr><tr><td>S3KG (Ours)</td><td>0.692/0.673</td><td>0.766/0.795</td><td>0.786/0.834</td><td>0.932/0.973</td><td>0.834/0.829</td><td>0.767/0.796</td><td>0.892/0.935</td><td>0.812/0.892</td><td>0.872/0.890</td></tr><tr><td>ROUGE-1</td><td>0.745/0.784</td><td>0.678/0.490</td><td>0.725/0.754</td><td>0.835/0.917</td><td>0.732/0.728</td><td>0.745/0.745</td><td>0.784/0.833</td><td>0.745/0.782</td><td>†</td></tr><tr><td>ROUGE-2</td><td>0.720/0.721</td><td>0.715/0.721</td><td>0.681/0.656</td><td>0.822/0.894</td><td>0.707/0.711</td><td>0.717/0.706</td><td>0.776/0.833</td><td>0.752/0.791</td><td>0.860/0.772</td></tr><tr><td>ROUGE-L</td><td>0.729/0.760</td><td>0.735/0.807</td><td>0.703/0.710</td><td>0.792/0.855</td><td>0.722/0.717</td><td>0.719/0.721</td><td>0.763/0.800</td><td>0.792/0.835</td><td>0.729/0.311</td></tr><tr><td>BLEU</td><td>0.687/0.677</td><td>0.716/0.747</td><td>0.671/0.644</td><td>0.806/0.884</td><td>0.715/0.708</td><td>0.711/0.710</td><td>0.775/0.819</td><td>0.745/0.794</td><td>0.868/0.745</td></tr><tr><td>BERTScore</td><td>0.758/0.816</td><td>0.691/0.702</td><td>0.682/0.636</td><td>0.823/0.916</td><td>0.757/0.792</td><td>0.739/0.761</td><td>0.816/0.871</td><td>0.743/0.798</td><td>0.747/0.645</td></tr><tr><td>MiniLM</td><td>0.723/0.748</td><td>0.687/0.638</td><td>0.833/0.894</td><td>0.875/0.943</td><td>0.770/0.789</td><td>0.802/0.844</td><td>0.780/0.817</td><td>0.773/0.814</td><td>0.821/0.811</td></tr><tr><td>sent-T5</td><td>0.766/0.816</td><td>0.674/0.668</td><td>0.853/0.928</td><td>0.876/0.944</td><td>0.770/0.827</td><td>0.848/0.902</td><td>0.728/0.760</td><td>0.797/0.871</td><td>0.762/0.806</td></tr></table>

C400: SK-Codex 400; Comb.: SK-Combined; Find: SK-FindKG; Oreg.: SK-Oregano; W-Swap: Wikipedia Entity-Swap.

Table 2. TAU performance vs. KEA baseline.
<table><tr><td>Metric</td><td>KEA</td><td>Ours</td></tr><tr><td>Micro Precision</td><td>0.853</td><td>0.847</td></tr><tr><td>Micro Recall</td><td>0.738</td><td>0.949</td></tr><tr><td>Macro Precision</td><td>0.953</td><td>0.803</td></tr><tr><td>Macro Recall</td><td>0.641</td><td>0.946</td></tr><tr><td>Micro F1</td><td>0.836</td><td>0.895</td></tr><tr><td>Macro F1</td><td>0.622</td><td>0.782</td></tr></table>

## 4.3. LLM Contextual Understanding Evaluation

To evaluate LLMs, S3KG is applied across three KGs per QA instance i: $K G _ { \mathrm { L L M } } ^ { ( i ) } , K G _ { \mathrm { g o l d } } ^ { ( i ) }$ , and $K G _ { \mathrm { c t x } } ^ { ( i ) }$ . Gold-$\mathbf { S i m } ( i ) = \mathbf { S } _ { \mathbf { S } \mathbf { 3 } \mathrm { K G } } ( K G _ { \mathrm { L L M } } ^ { ( i ) } , K G _ { \mathbf { g o l d } } ^ { ( i ) } )$ measures factual accuracy; CtxSim $( i ) = \mathrm { S } _ { \mathrm { S 3 K G } } ( K G _ { \mathrm { L L M } } ^ { ( i ) } , K G _ { \mathrm { c t x } } ^ { ( i ) } )$ measures contextual faithfulness. Since neither alone reflects true understanding, CUS penalises imbalance between the two as

$$
\mathbf { C U S } ( i ) = \frac { 2 \cdot \mathbf { G o l d S i m } ( i ) \cdot \mathbf { C t x S i m } ( i ) } { \mathbf { G o l d S i m } ( i ) + \mathbf { C t x S i m } ( i ) } .\tag{3}
$$

The dataset-level score is the mean of CUS(i) over all N samples. Table 3 reports mean GoldSim, CtxSim, and CUS over N = 400 samples per model–dataset combination at $\alpha = 0 . 5$ . Mistral-7B leads on CUS across both datasets (MesaQA: 0.678; PubMedQA: 0.592), driven by the highest CtxSim in each case (0.736 and 0.733), reflecting strong contextual faithfulness. CtxSim exceeds Gold-Sim in all eight model–dataset combinations, indicating that instruction-tuned models systematically elaborate on context rather than producing concise reference-style responses. A consistent ≈10 percentage-point CUS gap between MesaQA and PubMedQA across all models reveals domain complexity, biomedical vocabulary, and reasoning as key bottlenecks rather than model size or architecture. Falcon-7B shows the weakest performance on both datasets

Table 3. LLM evaluation results (mean over N=400 samples, α=0.5).
<table><tr><td>Dataset</td><td>Model</td><td>GoldSim</td><td>CtxSim</td><td>CUS</td></tr><tr><td rowspan="2">MesaQA</td><td>Gemma-7B Llama-2-7B</td><td>0.6889 0.6570</td><td>0.7031 0.7236</td><td>0.6774 0.6752</td></tr><tr><td>Mistral-7B Falcon-7B</td><td>0.6491 0.6036</td><td>0.7356</td><td>0.6780</td></tr><tr><td rowspan="4">PubMedQA</td><td>Gemma-7B</td><td>0.5235</td><td>0.6458 0.6401</td><td>0.6063 0.5587</td></tr><tr><td>Llama-2-7B</td><td>0.5220</td><td>0.6567</td><td>0.5651</td></tr><tr><td>Mistral-7B</td><td>0.5138</td><td>0.7331</td><td>0.5923</td></tr><tr><td>Falcon-7B</td><td>0.4541</td><td>0.5560</td><td>0.4800</td></tr></table>

and across all metrics.

On the 5% lowest similarity pairs, Mistral-7B remains the most reliable, recovering more reference-aligned triplets than other models, while Gemma-7B is the weakest, particularly on MesaQA. PubMedQA hard cases show sharply lower aligned-triplet recovery across all models, confirming that biomedical terminology and entity variability dominate failure modes.

## 5. Discussion

S3KG moves beyond surface-level metrics via typeseparated, one-to-one label alignment and WL kernel multihop sensitivity. The GoldSim/CtxSim decomposition exposes model-specific trade-offs invisible to aggregate metrics; the consistent PubMedQA–MesaQA domain gap confirms Mistral-7B handles domain-specific health knowledge more reliably than general health QA. KG extraction quality remains the primary bottleneck, and future work will incorporate GPT-4 and attention-based directional alignment to better distinguish semantically inverse relations.

## 6. Conclusion

We presented a KG-based evaluation pipeline combining S3KG and the TAU for reproducible, interpretable measurement of LLM contextual understanding in QA. S3KG achieves best-per-dataset F1 of 0.766–0.932 and AUROC up to 0.973, consistently outperforming lexical and neural baselines on KG-rich datasets while remaining competitive on short-text settings. The framework is dataset-agnostic and readily extensible, supporting broader efforts toward trustworthy and verifiable AI evaluation.

## Acknowledgments

The work of Kamal Premaratne (KP) was supported by the U.S. National Science Foundation (NSF) under Award No. 2530256. The authors also acknowledge the developers and open-source communities behind the Falcon, Mistral, Llama, and Gemma large language models, as well as the Hugging Face platform for providing access to open-source models and tools that supported this research.

## References

Agirre, E. et al. SemEval-2012 task 6: A pilot on semantic textual similarity. In Proceedings ofthe First Joint Conference on Lexical and Computational Semantics (\*SEM), 2012.

Bordes, A., Usunier, N., Garc´ıa-Duran, A., Weston, J., and´ Yakhnenko, O. Translating embeddings for modeling multi-relational data. In Advances in Neural Information Processing Systems, volume 26, pp. 2787–2795. Curran Associates, Inc., 2013.

Boudin, M., Diallo, G., Drance, M., and Mougin, F. The´ OREGANO knowledge graph for computational drug repurposing. Scientific Data, 10:871, 2023. doi: 10.1038/ s41597-023-02757-0. URL https://www.nature. com/articles/s41597-023-02757-0. Food ontology and natural compound knowledge graph for drug repurposing.

Dolan, W. B. and Brockett, C. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., and Larson, J. From local to global: A graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

Haskins, R. and Adams, B. Kea explain: Explanations of hallucinations using graph kernel analysis. arXiv preprint arXiv:2507.03847, 2025.

Izacard, G. and Grave, E. Leveraging passage retrieval with generative models for open domain question answering. In EMNLP, 2021.

Jin, Q., Dhingra, B., Liu, Z., Cohen, W., and Lu, X. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2567–2577, 2019.

Li, X. V. and Sanna Passino, F. FinDKG: Dynamic knowledge graphs with large language models for detecting global trends in financial markets. In Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF 2024), pp. 573–581, 2024. doi: 10.1145/3677052.3698603. URL https://arxiv. org/abs/2407.10909. Financial knowledge graph extracted from news articles using LLMs.

Lin, C.-Y. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out: Proceedings ofthe ACL-04 Workshop, pp. 74–81, 2004.

Ni, J., Abrego, G. H., Constant, N., Ma, J., Hall, K. B.,<sup>´</sup> Chang, M., and Yang, Y. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 272–279, 2022.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311–318, Philadelphia, PA, USA, 2002.

Poelen, J. H., Simons, J. D., and Mungall, C. J. GloBI: Global biotic interactions. [Online]. Available: https: //www.globalbioticinteractions.org, 2014. Accessed: Jan. 15, 2025.

Reimers, N. and Gurevych, I. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992, Hong Kong, China, 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410.

Safavi, T. and Koutra, D. Codex: A comprehensive knowledge graph completion benchmark. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021.

Sansford, H., Richardson, N., Maretic, H. P., and Saada, J. N. Grapheval: A knowledge-graph based llm hallucination

evaluation framework. arXiv preprint arXiv:2407.10793, 2024.

Sennrich, R., Haddow, B., and Birch, A. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 1715–1725. Association for Computational Linguistics, 2016. doi: 10.18653/v1/P16-1162. URL https: //aclanthology.org/P16-1162/. Subword segmentation method used for node replacement perturbations.

Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., and Borgwardt, K. M. Weisfeiler-Lehman graph kernels. Journal ofMachine Learning Research, 12:2539–2561, 2011.

Sun, Z., Deng, Z.-H., Nie, J.-Y., and Tang, J. RotatE: Knowledge graph embedding by relational rotation in complex space. In Proceedings ofthe 7th International Conference on Learning Representations (ICLR), 2019.

Togninalli, M., Ghisu, E., Llinares-Lopez, F., Rieck, B., and´ Borgwardt, K. Wasserstein Weisfeiler–Lehman graph kernels. In Advances in Neural Information Processing Systems, volume 32, pp. 6439–6449. Curran Associates, Inc., 2019.

Wang, J.-I., Huang, H.-H., and Chen, H.-H. MESAQA: A dataset for multi-span contextual and evidencegrounded question answering. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 10891–10901, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025. coling-main.724/.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. MiniLM: Deep self-attention distillation for taskagnostic compression of pre-trained transformers. In Advances in Neural Information Processing Systems, volume 33, pp. 5776–5788, 2020.

Wei, J. and Zou, K. EDA: Easy data augmentation techniques for boosting performance on text classification tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6382–6388. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1670. URL https: //aclanthology.org/D19-1670/. NLP-based perturbation techniques including deletion operations.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Yan, J., Wang, C., Huang, J., and Zhang, W. Do large language models understand logic or just mimick context? arXiv preprint arXiv:2402.12091, 2024.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations (ICLR), 2020.

Zhang, Y., Baldridge, J., and He, L. PAWS: Paraphrase adversaries from word scrambling. In Proceedings of the 2019 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pp. 1702–1711, 2019.

Zhu, Y., Moniz, J. R. A., Bhargava, S., Lu, J., Piraviperumal, D., Li, S., Zhang, Y., Yu, H., and Tseng, B.-H. Can large language models understand context? In Findings of the Association for Computational Linguistics: EACL 2024, pp. 2004–2018, mar 2024.

## A. S3KG Similarity Pipeline

![](images/6bd7fd12ea1794823d72b557577ebb0dea3408a926346554fec64e475c4fc040.jpg)  
Figure 2. S3KG similarity pipeline. Triple-level SBERT matching identifies semantically relevant triples in $G _ { 2 } ;$ soft label alignment resolves lexical mismatches between entity and relation labels; the normalised WL kernel and SBERT mean-pool scores are blended via mixing coefficient α to produce the final S3KG score.

## B. Hyperparameter α selection

This appendix reports the complete S3KG α sweep results across all evaluation datasets. Each table shows F1 and AUROC for $\alpha \in \{ 0 . 0 , 0 . 1 , \ldots , 1 . 0 \}$ , where $\alpha = 0 . 0$ recovers a pure KG structural embedding and $\alpha = 1 . 0$ recovers a pure dense sentence-transformer representation, as defined in Equation 2. The best-performing α per dataset (by F1) is shown in bold.

## B.1 Short-Text Datasets

Table 4 presents the α sweep for MRPC, PAWS-Wiki, and STS12. The optimal α differs notably across datasets: MRPC peaks at $\alpha = 0 . 3$ , PAWS-Wiki at $\alpha = 0 . 5$ , and STS12 at $\alpha = 0 . 1$ . The consistently low optimal values indicate that retaining a meaningful KG structural component is beneficial for short-text paraphrase detection, and that a pure dense-embedding representation (α = 1.0) is suboptimal for all three tasks.

Table 4. S3KG α sweep on short-text datasets (F1 / AUROC). The best α per dataset by F1 is shown in bold.
<table><tr><td colspan="3">MRPC</td><td colspan="2">PAWS-Wiki</td><td colspan="2">STS12</td></tr><tr><td>α</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td></tr><tr><td>0.0</td><td>0.676</td><td>0.654</td><td>0.694</td><td>0.739</td><td>0.783</td><td>0.828</td></tr><tr><td>0.1</td><td>0.681</td><td>0.664</td><td>0.745</td><td>0.781</td><td>0.786</td><td>0.834</td></tr><tr><td>0.2</td><td>0.683</td><td>0.669</td><td>0.760</td><td>0.790</td><td>0.780</td><td>0.834</td></tr><tr><td>0.3</td><td>0.692</td><td>0.673</td><td>0.764</td><td>0.793</td><td>0.780</td><td>0.831</td></tr><tr><td>0.4</td><td>0.680</td><td>0.674</td><td>0.764</td><td>0.795</td><td>0.780</td><td>0.827</td></tr><tr><td>0.5</td><td>0.680</td><td>0.675</td><td>0.766</td><td>0.795</td><td>0.775</td><td>0.824</td></tr><tr><td>0.6</td><td>0.681</td><td>0.674</td><td>0.764</td><td>0.795</td><td>0.768</td><td>0.820</td></tr><tr><td>0.7</td><td>0.688</td><td>0.674</td><td>0.764</td><td>0.795</td><td>0.764</td><td>0.815</td></tr><tr><td>0.8</td><td>0.684</td><td>0.672</td><td>0.764</td><td>0.795</td><td>0.762</td><td>0.810</td></tr><tr><td>0.9</td><td>0.683</td><td>0.671</td><td>0.764</td><td>0.795</td><td>0.756</td><td>0.805</td></tr><tr><td>1.0</td><td>0.681</td><td>0.671</td><td>0.764</td><td>0.731</td><td>0.756</td><td>0.790</td></tr></table>

## B.2 KG-Perturbed Paragraph Datasets

Table 5 presents results for the five KG-perturbed paragraph datasets. Four of the five datasets favour a balanced blend of structural and dense signal: SK-Codex 400 and SK-Combined peak at α = 0.5, SK-GloBI at $\alpha = 0 . 6 ,$ , and SK-Oregano at $\alpha = 0 . 4$ . SK-FindKG is the single exception, where the pure KG embedding $( \alpha = 0 . 0 )$ yields the highest F1 of 0.767, suggesting that the structural signal in FindKG is particularly discriminative and is diluted rather than enhanced by dense representations.

Table 5. S3KG α sweep on KG-perturbed paragraph datasets (F1 / AUROC). The best α per dataset by F1 is shown in bold.
<table><tr><td rowspan="2">α</td><td colspan="2">SK-Codex 400</td><td colspan="2">SK-Combined</td><td colspan="2">SK-FindKG</td><td colspan="2">SK-GloBI</td><td colspan="2">SK-Oregano</td></tr><tr><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td></tr><tr><td>0.0</td><td>0.871</td><td>0.969</td><td>0.776</td><td>0.801</td><td>0.767</td><td>0.796</td><td>0.811</td><td>0.898</td><td>0.777</td><td>0.884</td></tr><tr><td>0.1</td><td>0.922</td><td>0.973</td><td>0.791</td><td>0.825</td><td>0.760</td><td>0.798</td><td>0.883</td><td>0.934</td><td>0.803</td><td>0.892</td></tr><tr><td>0.2</td><td>0.927</td><td>0.973</td><td>0.819</td><td>0.829</td><td>0.762</td><td>0.793</td><td>0.888</td><td>0.937</td><td>0.811</td><td>0.892</td></tr><tr><td>0.3</td><td>0.927</td><td>0.973</td><td>0.828</td><td>0.830</td><td>0.748</td><td>0.788</td><td>0.889</td><td>0.936</td><td>0.812</td><td>0.892</td></tr><tr><td>0.4</td><td>0.929</td><td>0.973</td><td>0.833</td><td>0.829</td><td>0.744</td><td>0.783</td><td>0.891</td><td>0.936</td><td>0.812</td><td>0.892</td></tr><tr><td>0.5</td><td>0.932</td><td>0.973</td><td>0.834</td><td>0.829</td><td>0.741</td><td>0.779</td><td>0.890</td><td>0.935</td><td>0.810</td><td>0.892</td></tr><tr><td>0.6</td><td>0.932</td><td>0.973</td><td>0.832</td><td>0.828</td><td>0.736</td><td>0.776</td><td>0.892</td><td>0.935</td><td>0.812</td><td>0.892</td></tr><tr><td>0.7</td><td>0.932</td><td>0.973</td><td>0.833</td><td>0.828</td><td>0.734</td><td>0.772</td><td>0.892</td><td>0.935</td><td>0.812</td><td>0.891</td></tr><tr><td>0.8</td><td>0.932</td><td>0.973</td><td>0.832</td><td>0.828</td><td>0.731</td><td>0.769</td><td>0.892</td><td>0.935</td><td>0.810</td><td>0.891</td></tr><tr><td>0.9</td><td>0.932</td><td>0.973</td><td>0.828</td><td>0.827</td><td>0.729</td><td>0.765</td><td>0.892</td><td>0.934</td><td>0.810</td><td>0.891</td></tr><tr><td>1.0</td><td>0.932</td><td>0.932</td><td>0.828</td><td>0.799</td><td>0.728</td><td>0.756</td><td>0.892</td><td>0.917</td><td>0.810</td><td>0.770</td></tr></table>

## B.3 Wikipedia Entity-Swap (Anti-Circularity Control)

Table 6 presents the α sweep for the Wikipedia Entity-Swap dataset, which serves as an anti-circularity control. The best performance is achieved at α = 0.1 (F1 = 0.872, $\mathrm { A U R O C } = 0 . 8 9 0 .$ , Precision = 0.780), confirming that even a small contribution from the KG structural embedding improves over the pure dense baseline, while heavier structural weighting $( \alpha \ge 0 . 2 )$ offers no further benefit.

Table 6. S3KG α sweep on Wikipedia Entity-Swap (F1 / AUROC / Precision / Recall). The best α by F1 is shown in bold.
<table><tr><td>α</td><td>F1</td><td>AUROC</td><td>Precision</td><td>Recall</td></tr><tr><td>0.0</td><td>0.821</td><td>0.892</td><td>0.698</td><td>0.995</td></tr><tr><td>0.1</td><td>0.872</td><td>0.890</td><td>0.780</td><td>0.990</td></tr><tr><td>0.2</td><td>0.869</td><td>0.890</td><td>0.783</td><td>0.975</td></tr><tr><td>0.3</td><td>0.868</td><td>0.890</td><td>0.773</td><td>0.990</td></tr><tr><td>0.4</td><td>0.865</td><td>0.890</td><td>0.777</td><td>0.975</td></tr><tr><td>0.5</td><td>0.868</td><td>0.890</td><td>0.773</td><td>0.990</td></tr><tr><td>0.6</td><td>0.865</td><td>0.889</td><td>0.777</td><td>0.975</td></tr><tr><td>0.7</td><td>0.865</td><td>0.889</td><td>0.777</td><td>0.975</td></tr><tr><td>0.8</td><td>0.865</td><td>0.889</td><td>0.777</td><td>0.975</td></tr><tr><td>0.9</td><td>0.865</td><td>0.889</td><td>0.777</td><td>0.975</td></tr><tr><td>1.0</td><td>0.865</td><td>0.852</td><td>0.777</td><td>0.975</td></tr></table>

## C. Temperature Sensitivity Results

The sampling temperature is a key hyperparameter governing how deterministically a language model generates text. A temperature of zero corresponds to greedy decoding, where the model always selects the most probable next token, yielding highly consistent and context-adherent outputs. As temperature increases, the sampling distribution becomes broader, allowing the model to explore a wider range of responses. Low temperatures such as 0.3 preserve factual grounding while introducing modest lexical variation, whereas a mid-range value of 0.7 is commonly adopted in practice to balance fluency and diversity. At temperature 1.0, the model samples directly from its raw output distribution, producing the most varied responses but with a greater risk of factual drift away from the provided context.

We selected $T \in \{ 0 . 0 , 0 . 3 , 0 . 7 , 1 . 0 \}$ to span the full practical operating range of instruction-tuned models, from fully deterministic inference to high-entropy generation. This allows us to examine whether contextual faithfulness, as measured by GoldSim and CtxSim, is robust to generation stochasticity or degrades meaningfully as randomness increases. Tables 7 and 8 report these results. Most models remain stable within ±0.01–0.02 across all settings; the notable exception is Falcon-7B on MesaQA, where GoldSim falls from 0.6036 at $T { = } 0 . 0 \ \mathrm { t o } \ 0 . 4 6 6 2$ at T=1.0, indicating that higher sampling randomness substantially degrades factual alignment for this model.

Table 7. Mean GoldSim per model across temperatures.
<table><tr><td>Dataset</td><td>Model</td><td>T=0.0</td><td>T=0.3</td><td>T=0.7</td><td>T=1.0</td></tr><tr><td rowspan="4">MesaQA</td><td>Llama-2-7B</td><td>0.6570</td><td>0.6588</td><td>0.6580</td><td>0.6406</td></tr><tr><td>Gemma-7B</td><td>0.6889</td><td>0.7060</td><td>0.7015</td><td>0.6931</td></tr><tr><td>Mistral-7B</td><td>0.6491</td><td>0.6567</td><td>0.6523</td><td>0.6444</td></tr><tr><td>Falcon-7B</td><td>0.6036</td><td>0.6085</td><td>0.5558</td><td>0.4662</td></tr><tr><td rowspan="4">PubMedQA</td><td>Llama-2-7B</td><td>0.5220</td><td>0.5185</td><td>0.5143</td><td>0.5087</td></tr><tr><td>Gemma-7B</td><td>0.5235</td><td>0.5149</td><td>0.5153</td><td>0.5204</td></tr><tr><td>Mistral-7B</td><td>0.5138</td><td>0.5121</td><td>0.5085</td><td>0.5023</td></tr><tr><td>Falcon-7B</td><td>0.4541</td><td>0.4330</td><td>0.4184</td><td>0.3852</td></tr></table>

This appendix provides complete benchmark results referenced in the main paper, including short-text datasets (Section 4.1), KG-perturbed paragraph datasets, the Wikipedia Entity-Swap anti-circularity control, and a summary heatmap visualization.

## D.1 Short-Text Datasets

Table 9 presents the full F1 and AUROC results for MRPC, PAWS-Wiki, and STS12. S3KG achieves competitive performance, with sentence-T5-base leading on STS12 due to the short-text nature of the dataset.

Table 8. Mean CtxSim per model across temperatures.
<table><tr><td>Dataset</td><td>Model</td><td>T=0.0</td><td>T=0.3</td><td>T=0.7</td><td>T=1.0</td></tr><tr><td rowspan="4">MesaQA</td><td>Llama-2-7B</td><td>0.7236</td><td>0.7212</td><td>0.7248</td><td>0.7026</td></tr><tr><td>Gemma-7B</td><td>0.7031</td><td>0.7217</td><td>0.7045</td><td>0.7013</td></tr><tr><td>Mistral-7B</td><td>0.7356</td><td>0.7585</td><td>0.7287</td><td>0.7223</td></tr><tr><td>Falcon-7B</td><td>0.6458</td><td>0.6571</td><td>0.6122</td><td>0.5195</td></tr><tr><td rowspan="4">PubMedQA</td><td>Llama-2-7B</td><td>0.6567</td><td>0.6563</td><td>0.6594</td><td>0.6468</td></tr><tr><td>Gemma-7B</td><td>0.6401</td><td>0.6540</td><td>0.6434</td><td>0.6319</td></tr><tr><td>Mistral-7B</td><td>0.7331</td><td>0.7352</td><td>0.7314</td><td>0.7003</td></tr><tr><td>Falcon-7B</td><td>0.5560</td><td>0.5165</td><td>0.5120</td><td>0.4571</td></tr></table>

Table 9. Performance on short-text datasets (F1 / AUROC).
<table><tr><td>Method</td><td>MRPC</td><td>PAWS-Wiki</td><td>STS12</td></tr><tr><td>S3KG</td><td>0.692 / 0.673</td><td>0.766 / 0.795</td><td>0.786 / 0.834</td></tr><tr><td>ROUGE-1</td><td>0.745 / 0.784</td><td>0.678 / 0.490</td><td>0.725 / 0.754</td></tr><tr><td>ROUGE-2</td><td>0.720 / 0.721</td><td>0.715 / 0.721</td><td>0.681 / 0.656</td></tr><tr><td>ROUGE-L</td><td>0.729 / 0.760</td><td>0.735 /0.807</td><td>0.703 / 0.710</td></tr><tr><td>BLEU</td><td>0.687 / 0.677</td><td>0.716 / 0.747</td><td>0.671 / 0.644</td></tr><tr><td>BERTScore</td><td>0.758 / 0.816</td><td>0.691 / 0.702</td><td>0.682 / 0.636</td></tr><tr><td>MiniLM sentence-T5-base</td><td>0.723 / 0.748 0.766 / 0.816</td><td>0.687 / 0.638 0.674 /0.668</td><td>0.833 / 0.894 0.853 / 0.928</td></tr></table>

## D.2 KG-Perturbed Paragraph Datasets

Table 10 reports results for the five KG-perturbed paragraph datasets. S3KG achieves the highest F1 on four of five datasets, with gains up to +7.6 F1 points on SK-GloBI.

Table 10. Performance on KG-perturbed paragraph datasets (F1 / AUROC).
<table><tr><td>Method</td><td>C400</td><td>Comb.</td><td>Find</td><td>GloBI</td><td>Oreg.</td></tr><tr><td>S3KG</td><td>0.932/0.973</td><td>0.834/0.829</td><td>0.767/0.796</td><td>0.892/0.935</td><td>0.812/0.892</td></tr><tr><td>ROUGE-1</td><td>0.835/0.917</td><td>0.732/0.728</td><td>0.745/0.745</td><td>0.784/0.833</td><td>0.745/0.782</td></tr><tr><td>ROUGE-2</td><td></td><td>0.822/0.8940.707/0.7110.717/0.7060.776/0.8330.752/0.791</td><td></td><td></td><td></td></tr><tr><td>ROUGE-L</td><td></td><td></td><td></td><td>0.792/0.8550.722/0.7170.719/0.7210.763/0.8000.792/0.835</td><td></td></tr><tr><td>BLEU</td><td></td><td></td><td></td><td>0.806/0.8840.715/0.7080.711/0.7100.775/0.8190.745/0.794</td><td></td></tr><tr><td>BERTScore</td><td></td><td></td><td></td><td>0.823/0.9160.757/0.7920.739/0.7610.816/0.8710.743/0.798</td><td></td></tr><tr><td>MiniLM</td><td></td><td></td><td></td><td>0.875/0.9430.770/0.7890.802/0.8440.780/0.8170.773/0.814</td><td></td></tr><tr><td>sent-T5</td><td></td><td></td><td></td><td>0.876/0.9440.770/0.8270.848/0.9020.728/0.7600.797/0.871</td><td></td></tr></table>

C400: SK-Codex 400; Comb.: SK-Combined; Find: SK-FindKG; Oreg.: SK-Oregano.

## D.3 Wikipedia Entity-Swap Results

Table 11 presents results on the Wikipedia Entity-Swap anti-circularity control. S3KG achieves the highest F1 (0.872) and AUC (0.890), confirming that gains are not an artefact of the KG construction pipeline.<sup>1</sup>

Table 11. Results on Wikipedia Entity-Swap $( N = 4 0 0 )$
<table><tr><td>Method</td><td>F1</td><td>AUC</td><td>Prec.</td><td>Rec.</td></tr><tr><td>S3KG</td><td>0.872</td><td>0.890</td><td>0.780</td><td>0.990</td></tr><tr><td>ROUGE-2</td><td>0.860</td><td>0.772</td><td>0.773</td><td>0.970</td></tr><tr><td>ROUGE-L</td><td>0.729</td><td>0.311</td><td>0.573</td><td>1.000</td></tr><tr><td>BLEU</td><td>0.868</td><td>0.745</td><td>0.776</td><td>0.985</td></tr><tr><td>BERTScore</td><td>0.747</td><td>0.645</td><td>0.615</td><td>0.950</td></tr><tr><td>MiniLM</td><td>0.821</td><td>0.811</td><td>0.729</td><td>0.940</td></tr><tr><td>sentence-T5-base</td><td>0.762</td><td>0.806</td><td>0.621</td><td>0.985</td></tr></table>

## D.4 Best S3KG Variant Summary

Table 12 summarizes the best-performing α variant per dataset, selected by maximum F1 via grid search over α ∈ {0.0, 0.1, . . . , 1.0}.

Table 12. Best S3KG variant per dataset.
<table><tr><td>Dataset</td><td>F1/AUROC</td><td>Verdict</td></tr><tr><td>MRPC  $( \alpha = 0 . 3 )$ </td><td>0.692/0.673</td><td>Behind (sparse KG)</td></tr><tr><td>PAWS-Wiki (α = 0.5)</td><td>0.766/0.795</td><td>Strong (+3.1 F1)</td></tr><tr><td>STS12 (α = 0.1)</td><td>0.786/0.834</td><td>Behind (short text)</td></tr><tr><td>SK-Codex 400 (α = 0.5)</td><td>0.932/0.973</td><td>Strong (+5.7 F1)</td></tr><tr><td>SK-Combined (α = 0.5)</td><td>0.834/0.829</td><td>Strong (+6.0 F1)</td></tr><tr><td>SK-FindKG (α = 0.0)</td><td>0.767/0.796</td><td>Behind (domain noise)</td></tr><tr><td>SK-GloBI (α = 0.6)</td><td>0.892/0.935</td><td>Strong (+7.6 F1)</td></tr><tr><td>SK-Oregano (α = 0.4)</td><td>0.812/0.892</td><td>Comparable (+1.2)</td></tr><tr><td>Wiki Swap (α = 0.1)</td><td>0.872/0.890</td><td>Best meaningful score</td></tr></table>

## D.5 Performance Heatmap

Figure 3 visualizes the F1 and AUROC scores of S3KG and all baseline methods across all benchmark datasets. Gold borders indicate the best-performing method per dataset.

This appendix provides a detailed worked example from the PAWS-Wiki dataset, illustrating how S3KG correctly identifies semantic opposition where ROUGE-1 fails. Table 13 compares a positive (similar) pair and a negative (not similar) pair.

In the positive example, word-order variation yields identical KG triplets and si $1 _ { \alpha } = 1 . 0 ,$ , correctly predicting Similar. In the negative example, reversed subject–object roles expose semantic opposition despite identical surface tokens: ROUGE-1 incorrectly predicts Similar, while S3KG correctly predicts Not Similar.

S3KG vs. Baselines — F1 Score and ROC-AUC Across All Datasets
<table><tr><td colspan="10">F1 Score</td></tr><tr><td>MRPC</td><td>0.692</td><td>0.745</td><td>0.720</td><td>0.729</td><td>0.687</td><td>0.758</td><td>0.723</td><td>0.765</td><td rowspan="3"></td></tr><tr><td>PAWS-Wiki</td><td>0.766</td><td>0.678</td><td>0.715</td><td>0.735</td><td>0.716</td><td>0.691</td><td>0.687</td><td>0.674</td></tr><tr><td>Semantic-KG Combined</td><td>0.834</td><td>0.732</td><td>0.707</td><td>0.722</td><td>0.715</td><td>0.757</td><td>0.770</td><td>0.774</td></tr><tr><td>Wiki Swap Dataset</td><td>0.872</td><td>1.000</td><td>0.860</td><td>0.729</td><td>0.868</td><td>0.747</td><td>0.821</td><td>0.767</td><td rowspan="6">-0.8 FSo ore -0.7</td></tr><tr><td>Semantic-KG Codex 400</td><td>0.932</td><td>0.835</td><td>0.822</td><td>0.792</td><td>0.806</td><td>0.823</td><td>0.875</td><td>0.872</td></tr><tr><td>Semantic-KG FindKG</td><td>0.767</td><td>0.745</td><td>0.717</td><td>0.719</td><td>0.711</td><td>0.739</td><td>0.802</td><td>0.848</td></tr><tr><td>Semantic-KG GloBI</td><td>0.892</td><td>0.784</td><td>0.776</td><td>0.763</td><td>0.775</td><td>0.816</td><td>0.780</td><td>0.730</td></tr><tr><td>Semantic-KG Oregano</td><td>0.812</td><td>0.745</td><td>0.752</td><td>0.792</td><td>0.745</td><td>0.743</td><td>0.773</td><td>0.800</td></tr><tr><td>STS12</td><td>0.786</td><td>0.725</td><td>0.681</td><td>0.703</td><td>0.671</td><td>0.682</td><td>0.833</td><td>0.856</td></tr><tr><td></td><td>S3KG (Best α) (Ours)</td><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L Method</td><td>BLEU</td><td>BERTScore</td><td>MiniLM</td><td>sentence- T5-base</td><td></td></tr><tr><td colspan="10">ROC-AUC</td></tr><tr><td>MRPC</td><td>0.673</td><td>0.784</td><td>0.721</td><td>0.760</td><td>0.677</td><td>0.816</td><td>0.748</td><td>0.816</td><td>1.0</td></tr><tr><td>PAWS-Wiki</td><td>0.795</td><td>0.490</td><td>0.721</td><td>0.807</td><td>0.747</td><td>0.702</td><td>0.638</td><td>0.668</td><td>-0.9</td></tr><tr><td>Semantic-KG Combined</td><td>0.829</td><td>0.728</td><td>0.711</td><td>0.717</td><td>0.708</td><td>0.792</td><td>0.789</td><td>0.828</td><td></td></tr><tr><td>Wiki Swap</td><td>0.890</td><td>1.000</td><td>0.772</td><td>0.311</td><td>0.745</td><td>0.645</td><td>0.811</td><td>0.790</td><td></td></tr><tr><td>Dataset Semantic-KG Codex 400</td><td>0.973</td><td>0.917</td><td>0.894</td><td>0.855</td><td>0.884</td><td>0.916</td><td>0.943</td><td>0.945</td><td></td></tr><tr><td>Semantic-KG FindKG</td><td>0.796</td><td>0.745</td><td>0.706</td><td>0.721</td><td>0.710</td><td>0.761</td><td>0.844</td><td>0.902</td><td></td></tr><tr><td>Semantic-KG GloBI</td><td>0.935</td><td>0.833</td><td>0.833</td><td>0.800</td><td>0.819</td><td>0.871</td><td>0.817</td><td>0.761</td><td></td></tr><tr><td>Semantic-KG Oregano</td><td>0.892</td><td>0.782</td><td>0.791</td><td>0.835</td><td>0.794</td><td>0.798</td><td>0.814</td><td>0.871</td><td></td></tr><tr><td>STS12</td><td>0.834</td><td>0.754</td><td>0.656</td><td>0.710</td><td>0.644</td><td>0.636</td><td>0.894</td><td></td><td>0.928</td></tr><tr><td></td><td>S3KG (Best α) (Ours)</td><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L 口</td><td>BLEU Method Best per dataset</td><td>BERTScore</td><td>MiniLM</td><td></td><td>sentence- T5-base</td></tr></table>

Figure 3. F1 score (left) and AUROC (right) of S3KG and baseline methods across all benchmark datasets. Gold borders indicate the best-performing method per dataset. S3KG (navy border, leftmost column) represents the best-performing α variant selected per dataset from the full α sweep (Appendix B).

Table 13. S3KG pipeline scores for two PAWS-Wiki sentence pairs.
<table><tr><td>Positive Example</td><td></td><td>Negative Example</td></tr><tr><td>Text  $s _ { 1 }$ </td><td>His father returned as a inished violinist of the Russian School to Bombay.</td><td>Renzo Furlan won 6–3, 6–4 against Thomas Johansson in the finals.</td></tr><tr><td>Text  $s _ { 2 }$ </td><td>His father returned to Bombay as a finished violinist of the Russian school.</td><td>Thomas Johansson won 6–3, 6–4 against Renzo Furlan in the finals.</td></tr><tr><td>Triplet (s1) Triplet (s2)</td><td>(father, returned_to, Bombay)</td><td>(Renzo Furlan, won_against, Thomas Johansson)</td></tr><tr><td></td><td>(father, returned_to, Bombay)</td><td>(Thomas Johansson, won_against, Renzo Furlan)</td></tr><tr><td>simTP (α=0) simsT (α=1)</td><td>1.0000</td><td>0.8310</td></tr><tr><td></td><td>1.0000</td><td>0.1361</td></tr><tr><td>simα (α=0.5) Threshold τ</td><td>1.0000</td><td>0.4835</td></tr><tr><td></td><td>0.92</td><td>0.92</td></tr><tr><td>S3KG predic- tion</td><td>Similar √</td><td>Not Similar √</td></tr><tr><td>ROUGE-1 score</td><td>1.0000</td><td>1.0000</td></tr><tr><td>ROUGE-1 pre- Similar √ diction</td><td></td><td>Similar ×</td></tr><tr><td>True label</td><td>Positive (1)</td><td>Negative (0)</td></tr></table>