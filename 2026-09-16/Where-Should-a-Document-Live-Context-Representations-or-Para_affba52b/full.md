# Where Should a Document Live: Context, Representations, or Parameters?

Nathanaël Carraz Rakotonirina Momchil Hardalov

Gonzalo Iglesias Adrià de Gispert

Amazon AGI

{ncarraz, momchilh, gjii, agispert}@amazon.com

## Abstract

To answer questions outside of their pretraining data, large language models (LLMs) need access to new information, which can be presented in the context window as documents, encoded into the model’s parameters, or injected as latent representations. However, each of these methods comes with different efficiency, cost, and performance trade-offs, with no single winner. We present a controlled comparison of representation-based (KV-cache based) and parametric (fine-tuning-based) adaptation methods on five knowledge-intensive benchmarks. We show that in the oracle setting, Cartridges (KV) are the most accurate injection method at nearly every storage budget, outperforming parametric methods by 10 points. Compaction (KV) matches Cartridges only at low compression rates, lagging behind the parametric methods by 10 points at rates higher than 50×. In the more realistic multidocument retrieval scenario, Cartridges are the only method that matches in-context learning (ICL), leading the parametric methods by 29 points and Compaction by 15 points. Nonetheless, Cartridges are also the only method, besides full fine-tuning and large MLP adapters, that suffers from catastrophic forgetting, i.e., a 6% performance degradation on control benchmarks, with 13% in coding.

## 1 Introduction

Large language models (LLMs) often need additional information, such as enterprise knowledge bases, patient records, or legal corpora to answer questions that were not part of their training data. Techniques such as in-context learning (Brown et al., 2020, ICL) and retrieval-augmented generation (Lewis et al., 2020, RAG) offer a straightforward solution by placing the document in the context window. However, they are bounded by the context length and require re-processing the entire document for every query.

Knowledge injection methods, on the other hand, encode the documents once into compact adapters, replacing the textual context during inference (Eyuboglu et al., 2025; Su et al., 2025; Caccia et al., 2025). These methods fall into two families: Representation-based methods, which incorporate the document into a compressed key–value (KV) cache prefix (e.g. Cartridges (Eyuboglu et al., 2025) and Compaction (Zweiger et al., 2026)); and Parametric methods, which encode the knowledge into the model parameters (e.g. full fine-tuning, LoRA (Hu et al., 2022), or MLP adapters (Houlsby et al., 2019)). Throughout the paper we use adapters as an umbrella term for both KV caches and model weights.

Even though these methods are widely used in practice, prior work mainly evaluates questionanswering (QA) tasks over Wikipedia documents that the model has likely already seen in pretraining, therefore testing recall rather than genuinely encoding new knowledge (Ovadia et al., 2024; Su et al., 2025). Eyuboglu et al. (2025) takes a step towards a fairer comparison between parameter-based and representation-based methods, but it covers only a limited set of methods and in a single-document setting. In contrast, in our work we compare head-to-head five state-ofthe-art adaptation approaches on five knowledgeintensive tasks, both in the single-document and in the more realistic multi-document retrieval setting. Moreover, we quantify catastrophic forgetting on representative control benchmarks (math, instruction following, general knowledge, coding) and perform a storage-matched and runtime analysis of each method. Our analysis reveals that there is no single best method, and which method to use will depend on the use case. Our main contributions are the following:

• We present a holistic, controlled comparison of different knowledge injection methods:

(i) representation-based (Cartridges, Compaction); and (ii) parametric (LoRA, MLP adapters, full fine-tuning) on knowledgeintensive tasks in both single- and multipledocument settings.

• We study how adapter (KV caches or weights) size influences model performance and catastrophic forgetting, quantifying the trade-off for each knowledge injection method.

• We show that adapter composition in a multidocument retrieval setting is possible under certain conditions, but it remains challenging.

## 2 Related work

Parameter-Efficient Fine-Tuning. PEFT methods adapt a model by training only a small subset of its parameters or representations. They can be broadly categorized into two groups. Representation-based methods prepend a sequence of trainable vectors to the input and optimize only those vectors, and sometimes their activations across layers (Li and Liang, 2021; Lester et al., 2021; Liu et al., 2022; Eyuboglu et al., 2025). Parametric methods train a small number of parameters. Representative examples include adapters (Houlsby et al., 2019), small trainable modules inserted between layers, and LoRA (Hu et al., 2022), which learns a low-rank update to the weight matrices. Compared with full fine-tuning, PEFT offers several advantages: it requires less memory and compute to train, it can be stored and served efficiently (Chen et al., 2023), and multiple modules or adapters can be composed (Yadav et al., 2023; Zhong et al., 2024; Prabhakar et al., 2025; Hardalov et al., 2026). Finally, it better preserves the model’s general capabilities. In particular, LoRA matches the accuracy and sample efficiency of full finetuning while being less prone to catastrophic forgetting (Schulman and Lab, 2025).

Knowledge injection. Once Large Language Models (LLMs) are trained, they encode only the knowledge present in their training data. To answer questions about specific documents, a knowledge base, or any other source of knowledge, they must go through an adaptation phase. The most straightforward way is to add the relevant documents to the model’s context, possibly preceded by a retrieval stage that selects and filters the documents to be inserted (Lewis et al., 2020; Guu et al., 2020; Gao et al., 2023). An alternative to in-context injection is to encode the documents directly in the representations (Kujanpää et al., 2024; Rakotonirina and Baroni, 2024; Eyuboglu et al., 2025) or parameters (Xiao et al., 2023; Su et al., 2025) of the model. Representation-based knowledge injection is akin to prompt compression methods (Mu et al., 2023; Chevalier et al., 2023; Qin et al., 2024; Łajewska et al., 2025) that encode the context into shorter sequences of vectors or soft tokens. Recent work (Eyuboglu et al., 2025; Su et al., 2025; Caccia et al., 2025) has demonstrated that the best way to inject new knowledge is to train on synthetic data derived from the corpus rather than on the corpus itself, and to use a distillation objective rather than next-token prediction.

Comparing context, representations, and parameters. Several lines of work compare contextbased and parametric knowledge injection, but reach mixed and narrowly scoped conclusions. Ovadia et al. (2024) report that retrieval-augmented generation (RAG), which relies on in-context learning, consistently beats unsupervised fine-tuning for fact injection. However, they do not cover supervised fine-tuning or parameter-efficient adaptation. Parametric RAG (Su et al., 2025) encodes each document into a separate LoRA adapter and merges the adapters of the retrieved documents at inference time. It reports higher accuracy than in-context RAG in addition to lower latency. Subsequent work (Tang et al., 2025) shows that parametric methods do not consistently outperform text-based RAG, and that combining the two performs best. Both these works train and evaluate on QA datasets whose documents are Wikipedia paragraphs that were presumably already seen during pretraining, so the model is tested on recalling facts already encoded in its weights rather than on encoding new ones. In contrast, we evaluate on knowledgeintensive tasks with uncontaminated documents, as confirmed by our No Context baseline, which queries the LLM without access to external documents. Closest to our setting, Eyuboglu et al. (2025) compare Cartridges against in-context learning and LoRA, but report only limited results for the multi-document case where adapters are composed. We instead run an extensive, controlled comparison that spans all adaptation families and knowledge-intensive datasets varying widely in document length and task type, covering both the single-document setting (only the gold document is encoded) and the more realistic multi-document setting (several documents are retrieved and composed).

## 3 Methodology

Adaptation methods. We conduct a holistic evaluation on a variety of state-of-the-art adaptation methods from different families: context-based, representation-based, and parametric. We evaluate the following methods:

• No context: only the question is given to the model. This baseline quantifies how much can be answered from pretraining knowledge alone, with no access to external documents.

• ICL: the document is placed in the model’s context. This is the standard in-context baseline. In the multiple document setting, this is standard text-based RAG.

• Cartridges (Eyuboglu et al., 2025): a representation-based method that trains a small KV-cache prefix that is prepended to the question during inference. For the multidocument setting, Cartridges are trained following Hardalov et al. (2026).

• Compaction (Zweiger et al., 2026): a representation-based method that compresses the document into a compact set of key–value pairs by matching the attention the model would place on the full document.

• LoRA (Hu et al., 2022): a parametric method that adds trainable low-rank updates to the feed-forward projections of every layer, keeping the base model frozen.

• MLP adapters (Houlsby et al., 2019): a parametric method that inserts a small MLP after the attention block of every layer, training only these modules while keeping the base model frozen.

• Full fine-tuning: a parametric method that updates all model parameters, serving as an upper bound on capacity.

Knowledge injection settings. The task we are focusing on is what is the best approach for injecting new knowledge into a model. Importantly, we do not limit the QA task to a single-document scenario, where only the gold document is passed along with the questions in the model context, but also consider the more realistic scenario where the model answers based on documents retrieved from the corpus. In our experiments, we learn one adapter per document and pass only the question and the system instruction into the model prompt. These two settings can be formalized as follows:

• Single (gold) document: This is the oracle setting, i.e., we encode only the gold document in the model prompt; the goal is to isolate how knowledge in a single document is injected, independent of retrieval.

• Multiple documents: This follows the RAG setting, i.e., for each question we retrieve the top-k chunks and compose their corresponding adapters. For the representation-based methods we concatenate the k retrieved KV caches, and for the parametric methods we merge the k retrieved weights into a single set of weights. We have compared different merging approaches in Appendix B, based on which we selected the simple model merge by averaging. We also jointly train on all documents for parametric methods.

## 4 Experimental Settings

Models. We use Qwen3-8B (Yang et al., 2025) as our main model. We also validate our results on Gemma-3-12B (Team, 2025) in Appendix C.

Dataset. We evaluate on the following datasets (see Table 1 for dataset statistics) using their corresponding metric:<sup>1</sup>

• LongHealth (Adams et al., 2024): A clinical QA benchmark consisting of detailed fictional patient records with multiple-choice questions that test information extraction, negation understanding, and temporal sorting over long medical notes. We extract the model’s answer by fuzzy option matching, then compute exact match against the gold option.

• QASPER (Dasigi et al., 2021): An infoseeking QA dataset over full NLP research papers, where questions are written by NLP practitioners who saw only the title and abstract. Answers include extractive spans, freeform text, and yes/no responses. The metric is token-level F1 with the reference answers.

• QuALITY (Pang et al., 2022): A multiplechoice reading comprehension benchmark over long-form fiction and non-fiction narratives, where questions require careful reading and reasoning rather than surface-level pattern matching. The metric is exact match on the extracted answer letter (A, B, C, or D).

<table><tr><td>Dataset</td><td>Docs</td><td>Questions</td><td>Qs/Doc</td><td>Avg. Tok.</td><td>Total Tok. Task Type</td><td></td><td>Domain</td></tr><tr><td>LongHealth</td><td>20</td><td>400</td><td>20.0</td><td>11,700</td><td></td><td>236K Multiple-choice (5-way)</td><td>Clinical patient records</td></tr><tr><td>QASPER</td><td>407</td><td>1,451</td><td>3.5</td><td>4,751</td><td></td><td>665K Extract./free-form/yes-no</td><td>Full research papers</td></tr><tr><td>QuALITY</td><td>115</td><td>2,086</td><td>18.1</td><td>5,713</td><td></td><td>1.9M Multiple-choice (4-way)</td><td>Fiction &amp; non-fiction narratives</td></tr><tr><td>T2-RB/FinQA</td><td>380</td><td>1,147</td><td>3.0</td><td>1,026</td><td></td><td>392K Math. Calculation</td><td>Corporate earnings reports w/ tables</td></tr><tr><td>TechQA</td><td>496</td><td>610</td><td>1.8</td><td>1,509</td><td></td><td>748K Extractive</td><td>IBM IT support technotes</td></tr></table>

Table 1: Dataset statistics where Docs is the number of unique documents (one cartridge per document), Qs/Doc is the average number of questions per document, and Avg. Tok. is the average document length in tokens (Qwen3-8B tokenizer).

• T<sup>2</sup>-RAGBench/FinQA (Strich et al., 2026; Chen et al., 2021b): A decontextualized variant of FinQA designed for RAG evaluation, where questions about corporate earnings reports have been rewritten to be contextindependent. The task requires numerical reasoning by constructing mathematical formulas over tables and text extracted from financial filings. The model is expected to produce a formula, which we evaluate to a numerical value and compare against the ground-truth answer with a relative tolerance of 1%.

• TechQA (Castelli et al., 2020): A technical support QA dataset drawn from IBM’s Technote corpus, where questions require extracting precise solutions from IT documentation covering enterprise software and infrastructure issues. We use DeepSeek-Distilled-Qwen-32B (DeepSeek-AI, 2025) as a judge to evaluate the model’s prediction against the reference answer (see Appendix F for more details).

To assess whether knowledge injection degrades the model’s general abilities, we additionally evaluate on four control benchmarks: GSM8K (Cobbe et al., 2021) for grade-school math, HumanEval (Chen et al., 2021a) for code generation, IFEval (Zhou et al., 2023) for instruction following, and MMLU (Hendrycks et al., 2021) for broad knowledge. We evaluate adapted models and compare against the original model to measure catastrophic forgetting (Section 6).

Training data. Following Eyuboglu et al. (2025) and Hardalov et al. (2026), we train on LLMgenerated synthetic data called Self-Study data. More concretely, a question-generator LLM receives a part of the original document and a seed prompt, and produces a set of questions. An answer-generator LLM, given the same document, but not the seed prompt, answers each question providing tokens and their log probability distributions. Same as Hardalov et al. (2026) we use a GPT-OSS 120B (Agarwal et al., 2025) as the question generator and our target model as the answer generator. We generate n = 20 questions per chunk (more details in Appendix G). We train all methods using the same dataset.

Objective function. We use the distillation objective that minimizes the KL divergence between a teacher, which is a model with the document in context, and a student, which is the same model to be adapted. We select this distillation objective since it achieves a better performance compared to next-token-prediction for knowledge injection (Eyuboglu et al., 2025). We corroborate these findings for LoRA in Appendix D.

Retrieval. For the multi-document setting, we index the documents into chunks and retrieve the relevant chunks for each question, without query reformulation. For ICL and representation-based methods, the documents or KV caches are concatenated following their retrieval order. With 1,024- token chunks, LongHealth and QuALITY average ∼2 chunks per unique adapter, while for TechQA and FinQA the ratio is closer to 1:1 (see Table 9). We provide more details about retrieval and the RAG baseline in Appendix H.

Hyperparameters. For the fixed adapter size experiments, we use a compression rate of 2× for Cartridge and Compaction, a rank of 64 (with α = 128) for LoRA, and a bottleneck dimension of 512 for MLP adapters. LoRA is applied to the feed-forward projections of every layer, and the MLP adapters are added as a residual bottleneck on the output of every transformer layer. When comparing the methods across a range of adapter sizes, we vary the compression rate over 2×, 10×, 20×, 50×, and 100×. We then choose the LoRA rank and the MLP bottleneck dimension so that each adapter matches the per-document memory footprint of the KV-cache methods at that rate. Because storage is tied to document length, these matched sizes are dataset-dependent. We do not report a storage-matched adapter at 100× because the target footprint falls below the smallest trainable adapter (a rank of at least one). During inference, we use the decoding hyperparameters recommended for Qwen3 (Yang et al., 2025): temperature 0.6, top-p 0.95, and top-k 20, with a maximum of 4,096 generated tokens. We list full training hyperparameters in Appendix A.

<table><tr><td>Method</td><td>LongHealth (Acc.)</td><td>QuALITY (Acc.)</td><td>QASPER (F1)</td><td>FinQA (EM)</td><td>TechQA (Judge)</td><td>Avg.</td></tr><tr><td>No context</td><td> $3 7 . 5 \pm 1 . 1$ </td><td> $4 3 . 6 \pm 0 . 4$ </td><td> $1 9 . 2 \pm 0 . 6$ </td><td> $2 . 9 \pm 0 . 0$ </td><td> $2 1 . 1 \pm 1 . 1$ </td><td>24.9</td></tr><tr><td>ICL</td><td> $8 7 . 4 \pm 0 . 8$ </td><td> ${ \bf 8 2 . 5 \pm 0 . 3 }$ </td><td> ${ \pm 6 . 7 \pm 0 . 4 }$ </td><td> ${ \bf 6 6 . 8 \pm 2 . 7 }$ </td><td> $7 4 . 7 \pm 0 . 9$ </td><td>73.6</td></tr><tr><td>Cartridge</td><td> $8 1 . 1 \pm 1 . 1$ </td><td> $7 8 . 6 \pm 0 . 9$ </td><td> $\underline { { 5 4 . 9 \pm 0 . 3 } }$ </td><td> $6 2 . 7 \pm 0 . 2$ </td><td> $7 5 . 8 \pm 0 . 7$ </td><td>70.6</td></tr><tr><td>Compaction</td><td> ${ \bf 8 7 . 7 \pm 0 . 9 }$ </td><td> $\underline { { 8 2 . 1 } } \pm 0 . 3$ </td><td> $5 4 . 8 \pm 0 . 1$ </td><td> $6 6 . 4 \pm 0 . 2$ </td><td> $7 6 . 0 \pm 2 . 4$ </td><td>73.4</td></tr><tr><td>LoRA</td><td> $7 5 . 3 \pm 0 . 9$ </td><td> $7 3 . 7 \pm 1 . 2$ </td><td> $5 0 . 3 \pm 0 . 6$ </td><td> $4 9 . 0 \pm 0 . 9$ </td><td> $7 4 . 3 \pm 2 . 6$ </td><td>64.5</td></tr><tr><td>MLP adapters</td><td> $7 4 . 8 \pm 0 . 9$ </td><td> $7 2 . 3 \pm 1 . 2$ </td><td> $4 7 . 5 \pm 0 . 9$ </td><td> $4 4 . 1 \pm 0 . 3$ </td><td> $7 2 . 5 \pm 1 . 8$ </td><td>62.2</td></tr><tr><td>Full fine-tuning</td><td> $6 9 . 0 \pm 2 . 4$ </td><td> $7 2 . 8 \pm 0 . 3$ </td><td> $3 9 . 8 \pm 0 . 3$ </td><td> $4 1 . 6 \pm 2 . 8$ </td><td> $7 6 . 0 \pm 3 . 4$ </td><td>59.8</td></tr></table>

Table 2: Single-document scores at a fixed adapter size (Qwen3-8B). Cartridge and Compaction use 2× compression, LoRA rank 64, and MLP a bottleneck of 512. Scores are averaged across 3 runs (± standard deviation); bold marks the best method per dataset and underline the second best.

## 5 Experimental Results

## 5.1 Single Oracle document

Fixed-size adapter. Table 2 shows the model performance in oracle setting, i.e., providing only the encoded gold document. No Context (24.9) performs far below the other methods, confirming that the knowledge is genuinely injected rather than recalled. This gap is largest on FinQA, where answering a question requires extracting specific numbers from the text and tables of the document. In this setting, representation-based methods have a clear edge and outperform the parametric ones on nearly all datasets. Compaction is strongest overall with an average score of 73.4, matching the ICL upper bound of 73.6, followed by Cartridges (70.6). On the parametric side, LoRA is the best method (64.5 on average), ahead of MLP adapters (62.2) and full fine-tuning (59.8). Full fine-tuning is only competitive on TechQA and weakest on datasets whose metric penalizes malformed output (QASPER F1, FinQA formulas), which we relate to catastrophic forgetting (studied in § Catastrophic forgetting).

Comparing adapter sizes. We also compare the methods across a range of adapter sizes in the single-document setting in Figure 1. The two representation-based methods behave very differently as the compression rate increases. Compaction performs best at low compression but degrades steeply, overtaken by the parametric methods at high compression: on FinQA it falls from 66.4 at 2× to 19.3 at 20×, and on LongHealth from 87.7 to 46.5 at 100×. Cartridges instead stay nearly flat (e.g. LongHealth 81.1 → 77.3, QuAL-ITY 78.6 → 76.4, TechQA 75.8 → 76.9 from 2× to 100×) and dominate the storage-matched parametric methods on every dataset. LoRA and MLP adapter improve with size up to a point and then plateau. We confirm this by training wider adapters in Appendix A.

## 5.2 Multiple retrieved documents

Fixed-size adapter. We retrieve the top-k chunks for each question and compose their adapters (mapping each chunk to its source document and removing duplicates): KV caches are concatenated for the representation-based methods, and weights are averaged for the parametric methods. Figure 2 reports performance against the number of retrieved documents (k = 1, 3, 5, 10). In the comparison, we include a single adapter, trained jointly on all documents only for the parametric methods, as Hardalov et al. (2026) shows that Cartridges perform worse when trained jointly, while Compaction shifts the rope embedding to the original size of the compacted caches, which goes beyond the context window of the model. ICL and Cartridges maintain or improve their performance as k increases: from k=1 to k=10, Cartridges rise on LongHealth (70.8 → 83.2) and hold on TechQA $( 7 0 . 7  7 0 . 9 )$ and QuALITY (72.5 → 74.6). In contrast, both Compaction and the merged parametric adapters degrade monotonically. Compaction drops from k=1 to k=10 on every dataset (LongHealth 72.9 → 56.3, TechQA 65.4 → 26.4), and the merged adapters fall even faster: LoRA on TechQA collapses from 57.4 at k=1 to 23.5 at k=3, and on QuALITY from 70.8 to 44.2 at k=10. The degradation is most severe on FinQA, where merging collapses accuracy from 34.5 at k=1 to 4.9 at k=3. Joint training outperforms merging as soon as a single distractor is added (e.g. FinQA 13.0 vs. 4.9 and TechQA 40.0 vs. 23.5 at k=3), except on QuALITY, where it takes the lead only at k=5. Compaction and parametric injection are thus effective in the single-document setting but do not support multi-document composition.

![](images/9b6c33fa936c0852edd6916663b26a3f5c22d27616620dfa0b0342fbd9e5b4d3.jpg)  
Figure 1: Single-document score versus adapter size (MiB, log scale). Cartridges and Compaction are swept over compression rates 2×, 10×, 20×, 50×, and 100×. LoRA and the MLP adapter use the rank and bottleneck dimension that match the KV-cache memory footprint at each rate.

![](images/41ca9732a09f2577a8c049bffc15da5de79f805a150955505f711269fd27e1f4.jpg)  
Figure 2: Multi-document results (Qwen3-8B). Score versus the number of retrieved documents k, composing the top-k retrieved per-document artifacts. (Top) Fixed adapter size. ICL refers to RAG in this setting, ICL (Oracle) is the in-context upper bound with the gold document in context, and Joint is a single adapter trained on al documents. (Bottom) Varying adapter size. Each method is swept over the storage-matched compression ratios (where available), with method encoded by color and compression rate by shade (dark = larger adapter / lower compression).

Varying adapter size. We repeat the multidocument analysis across adapter sizes in Figure 2. Cartridges are mostly stable across compression rates; on LongHealth, for example, the k=10 score is 83.2 at 2× and still 76.5 at 100×. The only exception is FinQA, where heavier compression degrades performance (50.7 down to 22.1 from 2× to 20×), as in the single-document case. For LoRA and MLP, adapter size has essentially no effect on the composed result: by k=10 their scores converge regardless of the rate, to around 5 points on FinQA and 40–49 on LongHealth at both 2× and 50×. Compaction behaves similarly under composition but is further hurt by compression, dropping on LongHealth at k=10 from 56.3 at 2× to 37.2 at 50×. In short, using a larger rank, a wider bottleneck, or a lower compression rate does not fix the composition problem.

![](images/1f614eb3a41a87877adbc34966475cf75ee166819307ea1b85b00403bf82f961.jpg)  
Adapter size (LoRA rank r / MLP bottleneck d)  
Figure 3: Forgetting on control benchmarks (Qwen3-8B). The score is averaged across the five source datasets, and error bars are the standard error across those datasets. (Top) The representation-based methods are swept over compression rate; the lowest-compression column merges the 2× and 3× points (the two longest datasets use 3×). (Bottom) The parametric methods are swept over adapter size.

## 6 Analysis

Catastrophic forgetting. We assess how knowledge injection affects the general capabilities of the model by evaluating the adapted models on four control benchmarks, sampling 3 adapters (an adapter encodes a single document) per training dataset. We plot performance as a function of compression rate for the representation-based methods (Figure 3, top) and as a function of adapter size for the parametric methods (Figure 3, bottom). Among the representation-based methods, Compaction stays at the base model at every compression rate, while Cartridges degrade by 6% on average, mainly on code generation, where HumanEval drops by 16 points at high compression. In the parametric methods, LoRA retains the base model’s capabilities at every rank, whereas the MLP adapter is stable up to a moderate bottleneck (d≤384) and then degrades as it widens: on GSM8K it falls from ∼92 at the smallest bottleneck to ∼60 at the largest, with parallel drops on every benchmark.

The large MLP adapters are trained at a lower learning rate $( 2 \times 1 0 ^ { - 5 }$ instead of $1 0 ^ { - 4 } )$ , which we found necessary to avoid a much steeper collapse (e.g., GSM8K down to 24 at $1 0 ^ { - 4 } )$ . LoRA is more robust at matched parameter count, pointing to the low-rank constraint, rather than the number of parameters alone, as what accounts for preserving general capability.

Cost analysis. At inference time, ICL processes all the retrieved documents for each query during the prefill phase<sup>2</sup> and then decodes while attending over all of them. This can be a large recurring cost, from about 1k tokens on FinQA to 11k on LongHealth for the document alone, paid on every query. Representation-based and parametric knowledge injection methods remove the document from the prompt, prefilling only the question. The representation-based methods still attend over a stored prefix, but a compressed one, so their per-query cost shrinks with the compression rate. With representation-based methods a 10× token reduction yields ∼ 100× fewer prefill FLOPs due to quadratic attention. At inference, adapters are often pre-computed BF16 tensors, so loading requires only a GPU memory transfer, not a forward pass. On Qwen3-8B with H200 GPUs, loading 10 cartridges at 20× compression (6K KV tokens, 850 MiB) takes 30–50 ms, compared with 400–800 ms to prefill the equivalent 120K raw tokens. For the parametric methods the model attends to only the question, as the document lives in the weights. Overall, every injection method is cheaper per query than ICL, with the parametric methods the cheapest at inference and independent of document length.

## 7 Discussion

Which method to use. We find that no single method dominates across all considered settings. Cartridges are the most accurate knowledge injection method, including in the more realistic multidocument setting, but they degrade the model’s general capabilities at high compression rates. Compaction does not suffer from any forgetting but only matches Cartridges in the single-document setting and at low compression. The parametric methods lag behind Cartridges irrespective of the adapter size. Their advantage is instead that the adapter size does not depend on the document and that they are less prone to forgetting, except for large MLP adapters, which are never chosen in practice as they do not bring any performance boost. Jointly finetuning on all documents outperforms merging the parametric adapters when the number of retrieved documents is high. Full fine-tuning is dominated throughout: it is the most expensive to train and store yet the least accurate, and it forgets more than any low-rank adapter.

Composition is not straightforward. Combining per-document artifacts does not always work out of the box. This holds across families except for Cartridges: merging low-rank or full-rank adapters collapses with even a single distractor, and concatenating several independently compressed caches degrades in the same way. Although Su et al. (2025) showed promising results with merging LoRA adapters as an alternative to RAG, their findings were limited to small Wikipedia snippets, and the approach does not hold for the more complex tasks we consider. The composition of Cartridges in (Eyuboglu et al., 2025) was also limited, but the mixed training proposed in Hardalov et al. (2026) allows cartridges to be concatenated and perform well in the multi-document setting. A class of methods called KV-cache reuse addresses the problem of concatenating independently compressed caches, and could be applied to Compaction. KV Packet (Chen et al., 2026) and KVLink (Yang et al., 2026) add trainable soft tokens to bridge the discontinuities between the caches. C<sup>2</sup>KV (Du et al.,

2026) instead compresses and composes jointly, learning a compression objective under which the caches remain mergeable rather than optimizing each cache alone. For the parametric adapters, we are not aware of any competitive approach to combine per-document weights: more advanced weightmerging techniques do not lead to any significant improvement (Appendix B). Joint fine-tuning is the best available option but still lags composed cartridges, especially on information-dense datasets like FinQA.

On catastrophic forgetting. Among the representation-based methods, only Cartridges suffer from catastrophic forgetting on HumanEval. This suggests that the forgetting does not inherently come from the compressed KV caches. A possible fix is to initialize the KV caches using Compaction during Cartridges training. Regarding the parametric methods, only the full-rank MLP adapter forgets despite using the same number of parameters for all methods. Full fine-tuning also severely degrades the model’s general capabilities, suggesting that it is the full-rank nature of the update, rather than the parameter budget, that drives catastrophic forgetting. The low-rank constraint thus acts as a regularizer, suggesting that any parametric injection method intended to preserve general ability should be built around a subspace constraint.

## 8 Conclusion

We present a controlled comparison of representation-based and parametric knowledge injection methods across five knowledge-intensive benchmarks, in both single- and multi-document settings. We find that no method wins on every axis, but the trade-offs are clear. Cartridges are the most accurate injection method at nearly every storage budget and the only one that extends to multiple retrieved documents, at the cost of mild forgetting on the control benchmarks. Compaction matches the in-context oracle at low compression but degrades quickly as compression increases. The parametric adapters trail Cartridges at matched storage and cannot be composed across documents, but they are fixed-size and cheap to serve. Composing independently trained per-document artifacts remains the main open problem; for the representation-based family, compressed-cache reuse is a promising route.

## Limitations

Our study focuses on two mid-sized instructiontuned models, Qwen3-8B and Gemma-3-12B, and we do not test whether the same trade-offs hold for larger models. We also only investigate knowledgeintensive tasks and do not cover acquiring or improving skills, such as reasoning or coding, which may interact differently with each injection method. Our multi-document analysis relies on a single retrieve-then-compose pipeline: mean-merging for the parametric adapters, and prompt concatenation for the representation-based methods. The composition findings should therefore be read as the behavior of these standard merging operators, which more sophisticated reranking, filtering (Asai et al., 2024), routing, retrieval, or merging schemes may improve. All methods are trained on the same selfstudy synthetic data, so their relative performance is tied to the quality of that data and of the generators that produce it.

## Ethical Considerations

Our work is a controlled comparison rather than a deployed system, but the knowledge injection methods we study carry the usual risks of languagemodel applications. Because the injected knowledge is stored in the weights or KV caches, it is harder to inspect or update than an explicitly retrieved passage, and an adapted model can still hallucinate. All datasets we use are publicly available and, to the best of our knowledge, free of personal data, and we use each of them in accordance with its license (see Appendix I). Our training data is generated by an LLM (self-study) and is not additionally filtered. Finally, the catastrophic forgetting we measure is itself a safety-relevant failure mode, as injecting new knowledge can silently erode the general capabilities of the model.

## References

Lisa Adams, Felix Busch, Daniel Truhn, and Keno K. Bressem. 2024. LongHealth: A question answering benchmark with long clinical documents. arXiv preprint arXiv:2401.14490.

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-

trieve, generate, and critique through self-reflection. In International conference on learning representations, volume 2024, pages 9112–9141.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901.

Lucas Caccia, Alan Ansell, Edoardo Ponti, Ivan Vulic,´ and Alessandro Sordoni. 2025. Training plug-n-play knowledge modules with deep context distillation. arXiv preprint arXiv:2503.08727.

Vittorio Castelli, Rishav Chakravarti, Saswati Dana, Anthony Ferritto, Radu Florian, Martin Franz, Dinesh Garg, Dinesh Khandelwal, Scott McCarley, Michael McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus, John Pitrelli, Saurabh Pujar, Salim Roukos, Andrzej Sakrajda, Avi Sil, Rosario Uceda-Sosa, and 2 others. 2020. The TechQA dataset. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1269–1278, Online.

Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Bing Li, and Ulf Schlichtmann. 2026. Kv packet: Recomputation-free context-independent kv caching for llms. arXiv preprint arXiv:2604.13226.

L Chen, Z Ye, Y Wu, D Zhuo, L Ceze, and A Krishnamurthy. 2023. Punica: multi-tenant lora serving. arxiv. arXiv preprint arXiv:2310.18547.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021a. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021b. FinQA: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711, Online and Punta Cana, Dominican Republic.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics, pages 4599–4610.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.

Chuheng Du, Junyi Chen, Hanlin Tang, Kan Liu, Tao Lan, Lin Qu, Chaoyue Niu, Shengzhong Liu, Guihai Chen, and Fan Wu. 2026. C<sup>2</sup>KV: Compressed and composable kv cache reuse for efficient llm inference. arXiv preprint arXiv:2607.17715.

Sabri Eyuboglu, Ryan Saul Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Ruoyu Liu, Atri Rudra, James Y. Zou, Azalia Mirhoseini, and Christopher Re. 2025. Cartridges: Lightweight and general-purpose long context representations via self-study. In ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. The language model evaluation harness.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, Haofen Wang, et al. 2023. Retrievalaugmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1):32.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938.

Momchil Hardalov, Gonzalo Iglesias, and Adrià de Gispert. 2026. Cartridges at scale: Training modular kv caches over large document collections. arXiv preprint arXiv:2606.04557.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR).

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang,

Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3.

Kalle Kujanpää, Harri Valpola, and Alexander Ilin. 2024. Knowledge injection via prompt distillation. arXiv e-prints, pages arXiv–2412.

Weronika Łajewska, Momchil Hardalov, Laura Aina, Neha Anna John, Hang Su, and Lluis Marquez. 2025. Understanding and improving information preservation in prompt compression for LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 17520–17541, Suzhou, China.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pages 4582– 4597.

Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2022. P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 61–68.

Jesse Mu, Xiang Li, and Noah Goodman. 2023. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327– 19352.

Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. 2024. Fine-tuning or retrieval? comparing knowledge injection in LLMs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel R. Bowman. 2022. QuALITY: Question answering with long input texts, yes! Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics, pages 5336–5358.

Akshara Prabhakar, Yuanzhi Li, Karthik Narasimhan, Sham Kakade, Eran Malach, and Samy Jelassi. 2025. Lora soups: Merging loras for practical skill composition tasks. In Proceedings ofthe 31st International

Conference on Computational Linguistics: Industry Track, pages 644–655.

Guanghui Qin, Corby Rosset, Ethan Chau, Nikhil Rao, and Benjamin Van Durme. 2024. Dodo: Dynamic contextual compression for decoder-only lms. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9961–9975.

Nathanaël Carraz Rakotonirina and Marco Baroni. 2024. Memoryprompt: A light wrapper to improve context tracking in pre-trained language models. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 11187– 11195.

John Schulman and Thinking Machines Lab. 2025. Lora without regret. Thinking Machines Lab: Connectionism. Https://thinkingmachines.ai/blog/lora/.

Jan Strich, Enes Kutay Isgorur, Maximilian Trescher, Chris Biemann, and Martin Semmann. 2026. T2- RAGBench: Text-and-table benchmark for evaluating retrieval-augmented generation. In Proceedings ofthe 19th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 165–191.

Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented generation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1240–1250.

Yichen Tang, Weihang Su, Qingyao Ai, and Yiqun Liu. 2025. Understanding parametric knowledge injection in retrieval-augmented generation. arXiv preprint arXiv:2510.12668.

Gemma Team. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Chaojun Xiao, Zhengyan Zhang, Xu Han, Chi-Min Chan, Yankai Lin, Zhiyuan Liu, Xiangyang Li, Zhonghua Li, Zhao Cao, and Maosong Sun. 2023. Plug-and-play document modules for pre-trained models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15713–15729.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. 2023. TIES-merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2026. Kvlink: Accelerating large language models via efficient kv cache reuse. Advances in Neural Information Processing Systems, 38:133797– 133824.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In International Conference on Machine Learning.

Ming Zhong, Yelong Shen, Shuohang Wang, Yadong Lu, Yizhu Jiao, Siru Ouyang, Donghan Yu, Jiawei Han, and Weizhu Chen. 2024. Multi-lora composition for image generation. arXiv preprint arXiv:2402.16843.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Adam Zweiger, Xinghong Fu, Han Guo, and Yoon Kim. 2026. Fast KV compaction via attention matching. arXiv preprint arXiv:2602.16284.

## A Hyperparameters

We follow the hyperparameters in (Hardalov et al., 2026) for Cartridges. We train for 20 epochs on QASPER, FinQA, and TechQA, and 5 epochs on LongHealth and QuALITY. The longer-document datasets require more epochs to converge. All parametric methods use the AdamW optimizer with gradient clipping at 1.0, and an effective batch size of 64 (via gradient accumulation). We found that using learning-rate schedule or warmup did not result in any performance boost. LoRA uses rank $r { = } 6 4$ and $\alpha { = } 1 2 8 .$ , applied to the feed-forward projections (gate, up, down) of every layer, at learning rate $1 \times 1 0 ^ { - 4 }$ . The MLP adapter has a bottleneck dimension of d=512 inserted as a residual on every layer’s output, also at $1 \times 1 0 ^ { - 4 }$ . The two largest variants (d ∈ {1536, 3072}) instead use $2 \times 1 0 ^ { - 5 }$ as $1 \times 1 0 ^ { - 4 }$ was training-unstable at that size. We set the learning rate to $2 \times 1 0 ^ { - 5 }$ for full fine-tuning.

## B LoRA Merging Approaches

In the multi-document setting we compose perdocument LoRA adapters by merging the top-k retrieved adapters per question. The main results use a linear average (mean). Here we compare it against three alternatives from the model-merging literature: cat, which concatenates the low-rank factors; TIES (Yadav et al., 2023), which trims each adapter to its highest-magnitude entries (density 0.5), resolves sign conflicts, and averages the agreeing weights; and DARE (Yu et al., 2024), which randomly drops and rescales weights before a linear merge. We compare the merging methods with top-5 retrieved documents in Table 3. The four merge strategies are within roughly one point of one another on every dataset. The failure is therefore in composing independently trained adapters at all, not in the choice of merge operator.

## C Gemma-3-12B Results

To check that our findings are not specific to Qwen3-8B, we replicate the single-document setting on Gemma-3-12B (Team, 2025) (Table 4). Similar to Qwen3-8B, No context is far below all knowledge injection methods, and Compaction at 2× nearly matches the in-context oracle (65.2 vs. 68.9) while the parametric LoRA trails behind (50.8). Unlike on Qwen3-8B, the trained Cartridges underperform here (37.3 on average): on Gemma-3-12B they stay below the training-free

<table><tr><td>Merge</td><td>LongH.</td><td>QuAL.</td><td>FinQA</td><td>TechQA</td></tr><tr><td>Mean</td><td> $5 1 . 2 \pm 1 . 9$ </td><td> $5 6 . 0 \pm 0 . 7$ </td><td> $5 . 1 \pm 0 . 1$ </td><td> $2 1 . 7 \pm 0 . 4$ </td></tr><tr><td>Cat</td><td> $5 1 . 0 \pm 2 . 2$ </td><td> $5 6 . 0 \pm 0 . 7$ </td><td> $5 . 0 \pm 0 . 1$ </td><td> $2 2 . 1 \pm 1 . 0$ </td></tr><tr><td>TIES</td><td> $5 1 . 2 \pm 2 . 3$ </td><td> $5 6 . 0 \pm 0 . 6$ </td><td> $5 . 1 \pm 0 . 1$ </td><td> $2 1 . 9 \pm 0 . 9$ </td></tr><tr><td>DARE</td><td> $5 0 . 9 \pm 2 . 0$ </td><td> $5 6 . 0 \pm 0 . 7$ </td><td> $5 . 0 \pm 0 . 1$ </td><td> $2 2 . 5 \pm 0 . 8$ </td></tr></table>

Table 3: LoRA merging approaches (Qwen3-8B, rank 64). We merge the adapters of the five retrieved documents per question. Scores are the mean ± standard deviation over 3 seeds.
<table><tr><td>Method</td><td>LongHealth QuALITY (Acc.)</td><td>(Acc.)</td><td>QASPER (F1)</td><td>FinQA (EM)</td><td>TechQA (Judge)</td><td>Avg.</td></tr><tr><td>No context</td><td> $4 0 . 6 \pm 1 . 4$ </td><td> $2 8 . 0 \pm 0 . 7$ </td><td> $1 9 . 3 \pm 0 . 3$ </td><td> $5 . 2 \pm 0 . 0$ </td><td> $1 4 . 8 \pm 1 . 2 2 1 . 6 $ </td><td></td></tr><tr><td>ICL (oracle)</td><td> $7 8 . 5 \pm 0 . 4$ </td><td> $7 7 . 4 \pm 0 . 2$ </td><td> ${ \bf 5 1 . 3 \pm 0 . 3 }$ </td><td> ${ \bf 6 1 . 9 \pm 0 . 0 }$ </td><td> ${ \bf 7 5 . 4 \pm 1 . 9 6 8 . 9 }$ </td><td></td></tr><tr><td>Cartridge</td><td> $4 9 . 8 \pm 2 . 1$ </td><td> $4 3 . 0 \pm 1 . 4$ </td><td> $2 3 . 4 \pm 1 . 2$ </td><td> $9 . 5 \pm 1 . 7$ </td><td> $6 0 . 6 \pm 0 . 3 3 7 . 3$ </td><td></td></tr><tr><td>Compaction</td><td> ${ \bf 8 1 . 4 \pm 0 . 3 }$ </td><td> $7 4 . 4 \pm 0 . 5$ </td><td> $4 8 . 0 \pm 0 . 2$ </td><td> $6 1 . 2 \pm 0 . 9$ </td><td> $6 1 . 0 \pm 2 . 6 6 5 . 2$ </td><td></td></tr><tr><td>LoRA</td><td> $5 6 . 5 \pm 1 . 2$ </td><td> $6 2 . 4 \pm 0 . 5$ </td><td> $3 4 . 8 \pm 1 . 0$ </td><td> $3 4 . 7 \pm 0 . 2$ </td><td> $6 5 . 6 \pm 1 . 0 5 0 . 8$ </td><td></td></tr></table>

Table 4: Single-document results on Gemma-3-12B. Same protocol as Table 2 (Compaction 2×, LoRA rank 64. Scores averaged over 3 runs ± standard deviation where available).
<table><tr><td>Method</td><td>GSM8K</td><td>HumanEval</td><td>IFEval</td><td>MMLU</td></tr><tr><td>Base model</td><td>88.7</td><td>83.5</td><td>78.4</td><td>72.6</td></tr><tr><td>Cartridge</td><td>36.1 ± 23.9</td><td>54.4 ± 14.6</td><td> $3 9 . 2 \pm 1 5 . 7$ </td><td> $5 6 . 9 \pm 8 . 2$ </td></tr></table>

Table 5: Forgetting on control benchmarks (Gemma-3-12B). Base-model accuracy followed by the 2× trained Cartridges.

Compaction on every dataset, which we attribute to their catastrophic forgetting as presented in Table 5.

## D Distillation vs. Next-token Prediction

Our training objective distills the teacher’s nexttoken distribution rather than training on the teacher’s sampled tokens with the standard nexttoken-prediction cross-entropy. Table 6 compares the two objectives for LoRA in the single-document setting, holding everything else fixed. Distillation outperforms next-token-prediction on every dataset and is close to 4 points higher on average, with the largest gains on the reasoning-heavy datasets. This corroborates the objective choice of Eyuboglu et al. (2025) and motivates its use throughout our experiments.

## E Evaluation Prompts

We use task-specific system prompts during evaluation. All prompts instruct the model to wrap its final answer in <answer> tags. In the Cartridge setting, the document context is encoded in the trained KV cache prefix and is not included in the text prompt, so the model receives only the system prompt and the user question. In the Oracle (Text) baseline, the full document text is prepended to the user message. Table 7 lists the exact prompts used.

Generate {n} diverse questions that test   
knowledge of the information in the   
corpus above. Each question should cover   
a different fact, detail, or aspect   
of the corpus. Vary the style: mix   
factual recall, comparison, reasoning,   
and detail-oriented questions. Include   
specific details (ids, names, titles,   
dates, numerical values, etc.) in each   
question so it is clear what you are   
asking about.   
Output ONLY a JSON array of strings,   
e.g., ["question 1", "question 2", ...].   
No other text, no markdown fences, no   
explanation.

<table><tr><td>Dataset</td><td>Distillation</td><td>Next-token prediction</td></tr><tr><td>LongHealth</td><td> $7 5 . 2 \pm 0 . 9$ </td><td> $7 2 . 6 \pm 1 . 6$ </td></tr><tr><td>QuALITY</td><td> $7 3 . 7 \pm 1 . 2$ </td><td> $7 0 . 7 \pm 0 . 7$ </td></tr><tr><td>QASPER</td><td> ${ \bf 5 0 . 3 \pm 0 . 6 }$ </td><td> $4 6 . 9 \pm 1 . 0$ </td></tr><tr><td>FinQA</td><td> ${ \bf 4 9 . 0 \pm 0 . 9 }$ </td><td> $4 1 . 3 \pm 0 . 6$ </td></tr><tr><td>TechQA</td><td> $7 4 . 8 \pm 3 . 3$ </td><td> $7 3 . 2 \pm 1 . 2$ </td></tr><tr><td> $\operatorname { A v g } .$ </td><td>64.6</td><td>60.9</td></tr></table>

Table 6: Distillation vs. next-token prediction for per-document LoRA (Qwen3-8B, rank 64), singledocument setting. Scores are the mean ± standard deviation over 3 seeds.

For FinQA, we use the evaluation prompt recommended in the T<sup>2</sup>-RAGBench paper (Strich et al., 2026).<sup>3</sup> For $\mathrm { Q A S P E R ^ { 4 } }$ and QuALITY, we adopt the prompts from previous work (Gao et al., 2024; Eyuboglu et al., 2025; Zweiger et al., 2026). For LongHealth, we use the prompt format from the original benchmark<sup>5</sup>. For TechQA, since no standard evaluation prompt exists, we conducted a prompt search over several variants, including a minimal instruction (“Answer concisely”), a domain-specific expert framing, a chain-of-thought variant, and a structured extraction format, and selected the prompt that yielded the highest Oracle (Text) accuracy on the development set.

## F LLM-as-a-Judge evaluation

A separate judge model (DeepSeek-Distilled-Qwen-32B) evaluates whether the TechQA generated answer is factually equivalent to the reference answer. The judge outputs a JSON object with a justification and a binary grade. A score of 1 is assigned when the judge returns "grade": "correct", and 0 otherwise. Table 8 shows the exact judge prompt.

## G Self-Study Data Synthesis Details

We generate the self-study data following Eyuboglu et al. (2025); Hardalov et al. (2026).

Question generation model. We use GPT-OSS 120B (Agarwal et al., 2025) as the question generator $( M _ { Q } )$ . This model receives the full document context in its system prompt and generates batches of 20 diverse questions per call. The answer generator $( M _ { A } )$ is the target model itself (Qwen3-8B), ensuring that the distillation signal reflects the student’s own output distribution.

Multi-question prompt format. In batched mode, the question generator receives the following instruction (shown for the question type):

Seed prompt types. Each synthesis call randomly selects a seed prompt type from five categories: structuring (requests to organize information into JSON, YAML, or other formats), summarization (requests to summarize specific sections), question (factual recall and reasoning questions), use\_case (practical downstream application tasks), and creative (open-ended discussion prompts). This diversity ensures the cartridge is trained on varied interaction patterns rather than only factoid QA.

The structured JSON output format enables reliable parsing; responses that fail to parse are discarded (typically <5% of calls).

Proportional-to-length sampling. We replace uniform random chunk sampling with a proportional-to-length strategy ensuring balanced document coverage. Each document $d _ { i }$ in the collection is assigned a sampling weight $\begin{array} { r } { w _ { i } = \frac { | d _ { i } | } { \operatorname* { m i n } _ { j } | d _ { j } | } } \end{array}$ , so that longer documents, which are assumed to contain more facts, are sampled proportionally more often during synthesis.

Sampling rounds and temperatures. For each dataset, we run 4 independent sampling rounds, one per seed prompt type (question, structuring, summarization, use\_case), each generating 10,000 samples for a total of 40,000 training examples per dataset.<sup>6</sup> $M _ { Q }$ (GPT-OSS 120B, the question generator) uses temperature 0.6 with top-p=0.95, top-k=20, and a maximum of 4,096 completion tokens per call. $M _ { A }$ (Qwen3-8B, the teacher/answer generator) uses temperature 0.0 (greedy decoding) with a maximum of 2,048 completion tokens, ensuring deterministic distillation targets. Both models operate with thinking mode enabled.

<table><tr><td>Benchmark</td><td>System Prompt + User Message Format</td></tr><tr><td>T2-RAGBench / FinQA</td><td>System: You are an expert in answering financial questions by constructing mathematical formulas based on a simple syntax. - Task: Provide a FORMULA ANSWER to the question based on the given context. Guidelines: 1. Answer Type: A formula is either a number or one of: add(f1, f2), subtract(f1, f2), multiply(f1, f2), divide(f1, f2), exp(f1, f2), greater(f1, f2) 2. Reasoning: Carefully analyze the context. Pay special attention to the table. 3. Final Answer: &quot;&lt;answer&gt; FORMULA &lt;/answer&gt;&quot; User: {context} Question: {question}</td></tr><tr><td>LongHealth</td><td>System: Please reference the patient medical records to answer the user&#x27;s questions. Choose the single best option and provide your answer exactly as it appears in the options. Wrap your answer in: &lt;answer&gt; The correct option text here &lt;/answer&gt; User: {question} A. {option_a} B. {option_b} C. {option_c} D. {option_d} E. {option_e}</td></tr><tr><td>QASPER</td><td>System: You are a research assistant answering questions about a scientific paper. Answer as briefly as possible. Give only the answer, no explanation. If the question cannot be answered from the paper, say &quot;Unanswerable&quot;. For yes/no questions, answer &quot;Yes&quot; or &quot;No&quot;. Wrap your answer in: &lt;answer&gt; ... &lt;/answer&gt; User: {question}</td></tr><tr><td></td><td>System: You are a careful reader answering multiple-choice questions about a long article. Read the article and choose the single best answer option. Provide your answer as the letter (A, B, C, or D) wrapped in answer tags. Example: &lt;answer&gt; B &lt;/answer&gt; User: {question}</td></tr><tr><td></td><td>System: You are an expert technical support assistant specializing in IT infrastructure, software products, and enterprise systems. Answer the technical question to the best of your ability. Be concise and factual. Wrap your answer in: &lt;answer&gt; YOUR_ANSWER &lt;/answer&gt;</td></tr></table>

Table 7: Evaluation prompts used for each benchmark. Variables in braces are replaced with benchmark data at evaluation time.

Sampling configuration. Within each round, we generate 10,000 synthesis samples with a batch size of 4 contexts and up to 128 parallel API calls. Documents are sampled with a fixed random seed for reproducibility. For multi-note documents (e.g.,

LongHealth patient records), we sample one note per prompt to ensure fine-grained coverage of individual clinical notes.

## H RAG Baseline: Indexing and Retrieval Details

We describe the full RAG pipeline below.

Document indexing. Each dataset’s documents are indexed with chunk size C = 512 tokens. Documents are chunked using a fixed-size strategy with a 10% token overlap between adjacent chunks. Chunks are embedded using Amazon Titan Embed Text v2 (amazon.titan-embed-text-v2:0) via Amazon Bedrock<sup>7</sup> with 1024-dimensional embeddings and retrieved via cosine similarity.

Role Prompt   
System You are to act as an impartial judge, evaluating whether an answer to a question matches a   
provided reference answer. Your task is to determine if the given answer is correct based   
on its factual equivalence to the reference answer, ignoring differences in punctuation   
and phrasing.   
When evaluating the answer, consider the following criteria:   
1. Factual equivalence: Does the answer convey the same information as the reference   
answer?   
2. Completeness: Does the answer address all parts of the question that the reference   
answer addresses?   
3. Accuracy: Is the information in the answer consistent with the reference answer?   
4. Additional information: If the answer contains more information than the reference   
answer, does it remain consistent and not contradict itself?   
Your response should be structured as follows:   
1. A justification for your decision, explaining your reasoning based on the evaluation   
criteria.   
2. A grade of either "correct" or "incorrect".   
Important note on deflections and invalid questions:   
- If the answer is a deflection or does not attempt to answer the question, grade it as   
"incorrect" unless the reference answer is also a deflection.   
- If both the answer and the reference answer indicate that the question is invalid or   
cannot be answered, grade it as "correct".   
- If the answer is a placeholder like {{YOUR\_ANSWER}}, any generic phrase that does NOT   
address the question, or an empty answer, then count it as "incorrect".   
Your response should be in json format as follows:   
{"justification": "...", "grade": "correct" or "incorrect"}   
User Here is the question:   
{query}   
Here is the answer to be judged:   
{answer}   
Here is the reference answer:   
{expected\_answer}  
Table 8: LLM-as-a-Judge prompt used for scoring free-form answers on TechQA. The judge model (DeepSeek-Distilled-Qwen-32B) receives the system prompt defining evaluation criteria and a user message containing the question, generated answer, and reference answer.

Retrieval. At evaluation time, each question is issued as a retrieval query against the Knowledge Base. We retrieve the top-K chunks for $K \in \{ 1 , 3 , 5 , 1 0 \}$ . To avoid redundant API calls, we retrieve K = 10 once per chunk size and slice for smaller K values. Retrieved chunks are concatenated in score order and prepended to the question as the context for the reader model. Table 9 reports how many unique source documents the top-k retrieved chunks cover.

Effective token counts. Table 10 reports the average number of context tokens consumed per query for Text RAG at chunk size 1024 across different k values. Due to the 10% overlap between adjacent chunks and shorter final chunks in some documents, the effective token count is slightly below the nominal k × 1024.

<table><tr><td>Dataset</td><td>k=1</td><td>k=3</td><td> $k { = } 5$ </td><td> $k { = } 1 0$ </td></tr><tr><td>LongHealth</td><td>1.0</td><td>1.8</td><td>2.5</td><td>4.4</td></tr><tr><td>QuALITY</td><td>1.0</td><td>1.4</td><td>2.2</td><td>4.8</td></tr><tr><td>FinQA</td><td>1.0</td><td>3.0</td><td>5.0</td><td>8.7</td></tr><tr><td>TechQA</td><td>1.0</td><td>3.0</td><td>4.9</td><td>8.1</td></tr></table>

Table 9: Average number of unique source documents covered by the top-k retrieved chunks (chunk size 1024). Datasets with shorter documents (FinQA, TechQA) have higher chunk diversity, while longerdocument datasets (LongHealth, QuALITY) exhibit within-document clustering.

Retrieval Quality: Recall and MRR. We use the same document-level retrieval as Hardalov et al. (2026). Across all four datasets (LongHealth,

<table><tr><td>Dataset</td><td>k=1</td><td>k=3</td><td>k=5</td><td>k=10</td></tr><tr><td>LongHealth</td><td>986</td><td>2,960</td><td>4,944</td><td>9,860</td></tr><tr><td>QuALITY</td><td>941</td><td>2,856</td><td>4,733</td><td>9,412</td></tr><tr><td>FinQA</td><td>~960</td><td>~2,880</td><td>~4,800</td><td>~9,600</td></tr><tr><td>TechQA</td><td>~960</td><td>~2,880</td><td>~4,800</td><td>~9,600</td></tr></table>

Table 10: Average context tokens per query for Text RAG (chunk size 1024).

QuALITY, FinQA, and TechQA), the dense retriever achieves consistently high retrieval quality, with Recall@10 reaching approximately 95% or higher across chunk sizes, indicating that the relevant documents are almost always retrieved. This suggests that remaining performance differences are primarily due to information loss from chunking rather than retrieval failures. For the complete retrieval methodology, detailed Recall@K and MRR@K results, and further analysis, can be found in Appendix G (Hardalov et al., 2026).

## I Artifact Licenses and Terms of Use

Licenses and terms for use/distribution of artifacts. The five knowledge-intensive benchmarks used in this work are publicly available under the following licenses:

• LongHealth (Adams et al., 2024): Released under the Apache-2.0 License.<sup>8</sup> The dataset consists of entirely fictional patient records created by the authors; no real patient data is included.

• QASPER (Dasigi et al., 2021): Released under the CC BY 4.0 License.<sup>9</sup> The dataset consists of questions and answers over NLP research papers; the dataset itself is distributed under CC BY 4.0.

• QuALITY (Pang et al., 2022): Released under the CC BY 4.0 License.<sup>10</sup> Source texts are drawn from Project Gutenberg (public domain) and other permissively licensed collections, including nonfiction and fiction sources such as Slate articles from the Open American National Corpus, The Long+Short, Freesouls, and Open Access books. These texts are published works and do not contain personal data, though some may include mature themes typical of literary and journalistic content.

• FinQA (Chen et al., 2021b): Released under the MIT License.<sup>11</sup> The underlying financial reports are sourced from corporate filings made available via the SEC EDGAR system, including earnings reports and annual/quarterly reports (e.g., 10-K and 10-Q documents). These are publicly accessible financial disclosures prepared by reporting companies. T<sup>2</sup>-RAGBench (Strich et al., 2026) is also released under the MIT License.

• TechQA (Castelli et al., 2020): The NVIDIA TechQA-RAG-Eval variant used in this work is released under the Apache-2.0 License and is explicitly cleared for commercial and noncommercial use.<sup>12</sup> It is derived from the original IBM TechQA dataset, whose code repository is also Apache-2.0.<sup>13</sup>

We additionally use four control benchmarks to measure catastrophic forgetting, all publicly available:

• GSM8K (Cobbe et al., 2021): grade-school math word problems, released under the MIT License.

• HumanEval (Chen et al., 2021a): Python code-generation problems, released under the MIT License.

• IFEval (Zhou et al., 2023): instructionfollowing prompts, released under the Apache-2.0 License.

• MMLU (Hendrycks et al., 2021): multiplechoice questions across 57 subjects, released under the MIT License.

The models used in our experiments, Qwen3- 8B (Yang et al., 2025) and Gemma-3-12B (Team, 2025), are released under the Apache 2.0 License and the Gemma Terms of Use respectively, both of which permit research use.

We do not release new datasets in this work. The trained cartridge artifacts (KV cache parameters) are derivatives of the above datasets and the model weights; any release of such artifacts would be subject to the intersection of the applicable licenses above.

## Offensive content and personal data.

• LongHealth: The dataset consists of entirely fictional patient records with no real individuals. No anonymization was required. We verified that no real names, addresses, or identifying information appear in the data.

• QASPER: Source texts are NLP research papers. No personal data or offensive content is expected; the domain is scientific writing.

• QuALITY: Source texts are fiction and nonfiction narratives, these are published literary works; no personal data is present.

• FinQA / T<sup>2</sup>-RAGBench: Source texts are corporate earnings reports from SEC EDGAR. These are formal financial documents; no personal data or offensive content is present.

• TechQA: Source texts are IBM Technote IT support documents. These are technical documentation; no personal data or offensive content is expected.

• Control benchmarks (GSM8K, HumanEval, IFEval, MMLU): These are widely used, publicly released evaluation suites of math word problems, programming problems, instruction-following prompts, and academic multiple-choice questions. They contain no personal data, and no offensive content is expected.

No additional anonymization steps were taken beyond those applied by the original dataset creators, as none of the datasets contain personal data about private individuals.

## J AI use disclosure

We used AI to assist with code writing and manuscript typesetting.