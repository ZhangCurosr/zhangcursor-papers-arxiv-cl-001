# When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination

Karan Parekh, Sanjana Pendyala Ravinder, Sana Mhapsekar, Medina Maloku

University of North Texas

August 2026

## Abstract

Large language models are increasingly proposed as automated auditors of document quality, yet their reliability as detectors of planted errors is poorly characterised. We construct a contaminated corpus of 150 academic papers spanning supply chain management and medical research, injecting 450 known contaminants of three types: typographical corruption, semantic reversal, and absurd out-of-context insertion. We then evaluate Google Gemini 3.0 Pro’s ability to recover a 180-contaminant answer-key subset across 60 documents under three prompting regimes of increasing scale: single document, small batch, and large batch. Detection holds at small scale and then collapses: 50% recovery on single documents, 60% on small batches, and 2.8% on large batches. The failure mode at scale is not abstention but fabrication. Rather than reporting incomplete processing, the model produced confident findings including invented contaminants of its own, absurdities such as “telepathic squirrel” and “quantum-powered toaster” that mimic the style of the planted material but do not appear in any document. Detection also varies by contamination type: absurd insertions were recovered at 75% in completed evaluations, while semantic reversals and typographical corruptions were each recovered at only 50%. The corruptions most likely to occur in the wild, plausible ones, are the ones most often missed. We conclude that LLM document auditing degrades not gracefully but deceptively, and outline the harness such systems require: bounded batch sizes, direct content injection, and mechanical verification of every reported finding against source text.

## 1. Introduction

If an LLM is asked to audit more documents than it can process, what does it do? The safe behaviour is abstention, and models can in principle recognise the limits of their own knowledge and be tuned to decline (Kadavath et al., 2022; Zhang et al., 2024). The behaviour we observe is fabrication, and fabrication in the expected genre: asked to find planted absurdities, the model invented plausible-sounding absurdities of its own.

This matters because LLMs are being deployed as quality gates: reviewing documents, checking claims, flagging errors (Zheng et al., 2023; Liang et al., 2024; Lovering et al., 2025; Son et al., 2025). Such deployments implicitly assume the auditor is at worst incomplete, not actively misleading. Hallucination is by now well documented as a general property of language generation (Ji et al., 2023; Huang et al., 2025), but the auditing setting inverts the usual concern: here the model’s output is itself the quality signal, so a fabricated finding does not merely mislead a reader, it corrupts the control that was supposed to catch the error. We test that assumption by planting known contaminants in real academic documents and measuring what a model reports back as workload increases.

A note on terminology. In the LLM literature, contamination usually refers to benchmark or training data leakage, the presence of evaluation material in a model’s pretraining corpus (Magar and Schwartz, 2022; Sainz et al., 2023; Golchin and Surdeanu, 2024). We use the word in a narrower and unrelated sense: deliberately planted defects inside the documents a model is asked to audit.

## Contributions:

1. A contaminated-corpus construction method with a three-type taxonomy spanning surface corruption (typos), semantic corruption (meaning reversal), and contextual corruption (absurd insertion), applied to 150 real academic PDFs (450 contaminants).

2. A scaled evaluation protocol measuring recovery of a 180-contaminant answer key across 60 documents at three batch sizes.

3. Documentation of a deceptive failure mode: under load the model does not report failure, it generates fluent, well-structured, wholly fabricated audit findings, including invented contaminants styled after the expected ones.

## 2. Corpus and Contamination Method

Source documents. 150 published academic papers in two domains, supply chain management and medical and pharmaceutical sciences, chosen to test generalisation across technical vocabularies.

Contamination taxonomy. Each contaminated document received three planted contaminants, one per type:

<table><tr><td>Type</td><td>Mechanism</td><td>Examples (actual)</td><td>Detectability hypothesis</td></tr><tr><td>Typo</td><td>Surface corruption of a real word</td><td>“efficency”, “buisness”, “knowlege”</td><td>Detectable by spelling alone</td></tr><tr><td>Conflictin g</td><td>Semantic reversal of a directional claim</td><td>“improvements&quot; becomes “deteriorations”; “robust” becomes “fragile”</td><td>Requires understanding the claim</td></tr><tr><td>Nonsense</td><td>Absurd out-of- context insertion</td><td>&quot;supply chain&quot; becomes “dolphin choir&quot;; “parameter&quot; becomes “parachute&quot;</td><td>Contextually obvious</td></tr></table>

Every contaminant was logged with contaminant ID, document ID, page, and original and replacement text, forming a complete answer key.

Deployment. The corpus was published as a reference-accessible knowledge base, enabling evaluation via document references rather than inline content.

## 3. Evaluation Protocol

Google Gemini 3.0 Pro (free tier) was prompted to identify contaminants under three regimes:

Part I, single document. One supply chain and one medical document. 6 contaminants.

Part II, small batch. One batch of 7 supply chain documents and one of 3 medical documents. 30 contaminants.

• Part III, large batch. 48 documents in six batches of 8. 144 contaminants.

Responses were scored programmatically against the answer key (exact and fuzzy matching with manual adjudication of page and type mismatches). Each contaminant-level response was additionally rated on five Likert criteria: usefulness, accuracy, clarity, completeness, and overall satisfaction.

## 4. Results

4.1 Detection collapses with batch size

<table><tr><td>Regime</td><td>Docs</td><td>Contaminants</td><td>Recovere d</td><td>Rate</td></tr><tr><td>Part I, single document</td><td>2</td><td>6</td><td>3</td><td>50%</td></tr><tr><td>Part II, small batch</td><td>10</td><td>30</td><td>18</td><td>60%</td></tr><tr><td>Part II, large batch</td><td>48</td><td>144</td><td>4</td><td>2.8%</td></tr><tr><td>Overall</td><td>60</td><td>180</td><td>25</td><td>13.9%</td></tr></table>

Within Part II the supply chain batch of 7 recovered 15/21 (71.4%) while the medical batch of 3 recovered only 3/9 (33.3%), indicating domain and document effects alongside the batch-size effect. The single supply chain document in Part I was an outlier at 0/3 while its medical counterpart scored 3/3. In Part III, four of six batches recovered zero contaminants; no batch recovered more than two of twenty-four. Position and length effects in long-context models are well established. Accuracy falls when the relevant material sits in the middle of a long context rather than at either end (Liu et al., 2024), and models that score near perfectly on simple needle-in-a-haystack probes frequently fail more demanding tasks well below their advertised context length (Hsieh et al., 2024). What our result adds is the shape of the failure at the point of overload: the model does not return a lower score on the planted items, it returns a different set of items.

## 4.2 The failure mode at scale is confident fabrication

In the large-batch regime the model returned findings for the requested documents in fluent, well-structured form. The findings were fabricated. The scoring log records the model reporting invented contaminants including “quantum-powered toaster”, “telepathic squirrel”, “disco-dancing warehouse”, “flying pancake”, “magic carpet”, “pizza cutter”, and “pogo stick equation”. None of these strings appear in any corpus document. The actual planted contaminants in those same documents, items such as “dolphin choir”, “efficency” and “deteriorations”, went almost entirely unreported (4 of 144 recovered).

Two properties of this failure mode deserve emphasis. First, it is genre-consistent: the invented contaminants imitate the absurdist style of the real planted ones, suggesting the model inferred the kind of answer expected and generated to that expectation rather than to the documents. A related pattern is documented for sycophancy, where models produce what a reader is expected to prefer rather than what the evidence supports, in part because preference data rewards it (Sharma et al., 2024). Second, it is formally indistinguishable from success: evaluator ratings show clarity and coherence as the highest-scoring criterion across all 180 observations while accuracy and trustworthiness scored lowest. The fabricated audits read exactly like real ones. This is what makes the failure dangerous rather than merely inconvenient, because fluency and confidence are poor proxies for correctness. LLM judges can be swayed by ordering and presentation rather than substance (Wang et al., 2024), verbalised confidence is systematically overstated (Kadavath et al., 2022; Xiong et al., 2024), and human raters and preference models reward convincingly written but incorrect answers (Sharma et al., 2024). A downstream consumer of these audits has no available signal that separates the fabricated run from the successful one. Low recall on error finding is not unique to our setting either: across 83 published papers containing errors severe enough to have prompted errata or retraction, no frontier model exceeded 21.1% recall or 6.1% precision, with poorly calibrated confidence and low run-to-run consistency (Son et al., 2025).

One partial recovery illustrates the boundary: for a document contaminated with “volcano dance”, the model reported “disco dance”, the right location and genre with the wrong content, consistent with reconstruction from partial context rather than reading.

4.3 Detection asymmetry by contamination type

Within completed evaluations (Parts I and II, 36 contaminants):

<table><tr><td>Type</td><td>Recovere d</td><td>Rate</td></tr><tr><td>Nonsense (absurd insertion)</td><td>9/12</td><td>75%</td></tr><tr><td>Typo (surface corruption)</td><td>6/12</td><td>50%</td></tr><tr><td>Conflicting (semantic reversal)</td><td>6/12</td><td>50%</td></tr></table>

Contextually absurd insertions were caught most reliably. Semantic reversals, which preserve grammatical plausibility while inverting a claim’s direction, and simple typographical corruptions were each missed half the time. The contaminations most representative of real-world corruption, plausible ones, are precisely the ones the model misses most. Across all 180 contaminants including the fabrication-dominated large batches, recovery was 18.3% for nonsense, 13.3% for conflicting, and 10% for typos.

Independent work reports a similar order of magnitude: on expert-inserted inconsistencies in long technical documents, the strongest model tested recovered 64% of them and every model tested missed roughly half (Lovering et al., 2025).

## 4.4 False positives

In the single-document supply chain trial the model reported six contaminants, none planted. Whether these are hallucinations or genuine pre-existing defects in the source paper is unresolved; either way, an auditor whose findings require their own audit loses much of its value. The ambiguity is not ours alone. Lovering et al. (2025) report that of the unplanted items their models flagged, a large majority were judged on review to be genuine pre-existing errors in the source papers, which means unplanted flags cannot be treated as noise without adjudication. Settling this requires expert review of the source document rather than answer-key matching, and we did not perform it.

## 5. Discussion and Deployment Implications

1. Bound the batch. Detection was serviceable at 1 to 10 documents and collapsed at 48. Auditing pipelines should shard aggressively regardless of nominal context capacity.

2. Inject content, do not reference it. Reference-based access was the trigger for fabrication. Placing document text directly in context removes the conditions under which the model generated from expectation rather than evidence.

3. Verify mechanically. Every reported finding should be string-checked against source text before acceptance. Our scoring pipeline did this by construction; the fabricated findings would have been rejected automatically. Production systems typically lack this loop. Decomposing a generation into checkable units and verifying each against a source is an established mitigation, whether by atomic fact scoring against a knowledge source (Min et al., 2023) or by planning verification questions and answering them independently of the draft (Dhuliawala et al., 2024). In the auditing setting the check is cheaper still, because a reported finding is a claim that a specific string appears in a specific document, which is decidable by exact match.

4. Design for loud failure. The model never reported inability to process the large batches. Wrapping systems must detect and surface incompleteness themselves, because the model will not volunteer it, and its silence is dressed as an answer. Refusal-aware tuning shows that declining to answer can be taught as a general skill (Zhang et al., 2024), but an integrator cannot assume it is present in a model they do not control.

## 6. Limitations

Single model (Gemini 3.0 Pro, free tier), single scored run per regime, no prompt-variation or temperature sweep. The answer key covers 180 of the 450 planted contaminants; the remainder were not evaluated. Contaminant density was fixed at three per document, one per type. Likert ratings were author-assigned rather than independently rated. Results reflect model access at evaluation time and may not transfer to current models; the protocol is model-agnostic and inexpensive to repeat.

## 7. Conclusion

Planted-contamination evaluation is a cheap, repeatable way to measure whether an LLM auditor can be trusted. Ours cannot be, unsupervised: detection collapses with batch size, the collapse presents as confident output rather than surfaced failure, the model invents findings in the genre it expects, and the most plausible corruptions are the least detected. None of this argues against LLM document auditing. It is the specification for the harness such auditing must run inside.

## Data and code availability

Evaluation pipeline, prompts, scoring scripts and full detection logs: github.com/karanparekh14/llm-contamination-detection-eval. Contaminated corpus and answer key: github.com/karanparekh14/genai-rag-qa-portfolio.

## Acknowledgements

This work originated in coursework for ADTA-DAST 5770 at the University of North Texas.

## References

Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A., and Weston, J. (2024). Chain-of-Verification Reduces Hallucination in Large Language Models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3563 to 3578. doi:10.18653/v1/2024.findings-acl.212

Golchin, S., and Surdeanu, M. (2024). Time Travel in LLMs: Tracing Data Contamination in Large Language Models. In The Twelfth International Conference on Learning Representations (ICLR 2024). arXiv:2308.08493

Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., and Ginsburg, B. (2024). RULER: What’s the Real Context Size of Your Long-Context Language Models? In First Conference on Language Modeling (COLM 2024). arXiv:2404.06654

Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., et al. (2025). A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Transactions on Information Systems, 43(2), Article 42. doi:10.1145/3703155

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., et al. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys, 55(12), Article 248. doi:10.1145/3571730

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., et al. (2022). Language Models (Mostly) Know What They Know. arXiv preprint arXiv:2207.05221

Liang, W., Zhang, Y., Cao, H., Wang, B., Ding, D. Y., Yang, X., Vodrahalli, K., He, S., et al. (2024). Can Large Language Models Provide Useful Feedback on Research Papers? A Large-Scale Empirical Analysis. NEJM AI, 1(8), AIoa2400196. doi:10.1056/AIoa2400196

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12, pages 157 to 173. doi:10.1162/tacl\_a\_00638

Lovering, C. J., Ebner, S., Smock, B., Krumdick, M., Rabbani, S., Muhammad, A., Reddy, V., and Tanner, C. (2025). On Finding Inconsistencies in Documents. arXiv preprint arXiv:2512.18601

Magar, I., and Schwartz, R. (2022). Data Contamination: From Memorization to Exploitation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 157 to 165. doi:10.18653/v1/2022.acl-short.18

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076 to 12100. doi:10.18653/v1/2023.emnlp-main.741

Sainz, O., Campos, J., García-Ferrero, I., Etxaniz, J., Lopez de Lacalle, O., and Agirre, E. (2023). NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10776 to 10787. doi:10.18653/v1/2023.findings-emnlp.722

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S., Durmus, E., Hatfield-Dodds, Z., et al. (2024). Towards Understanding Sycophancy in Language Models. In The Twelfth International Conference on Learning Representations (ICLR 2024). arXiv:2310.13548

Son, G., Hong, J., Fan, H., Nam, H., Ko, H., Lim, S., Song, J., Choi, J., et al. (2025). When AI Co-Scientists Fail: SPOT, a Benchmark for Automated Verification of Scientific Research. arXiv preprint arXiv:2505.11855

Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Kong, L., et al. (2024). Large Language Models are not Fair Evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440 to 9450. doi:10.18653/v1/2024.acl-long.511

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., and Hooi, B. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. In The Twelfth International Conference on Learning Representations (ICLR 2024). arXiv:2306.13063

Zhang, H., Diao, S., Lin, Y., Fung, Y., Lian, Q., Wang, X., Chen, Y., Ji, H., et al. (2024). R-Tuning: Instructing Large Language Models to Say ‘I Don’t Know’. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational

Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7113 to 7139. doi:10.18653/v1/2024.naacl-long.394

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track. arXiv:2306.05685