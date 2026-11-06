# Dataset Collection and Fine-Tuning of NLLB for German–English Translation

Deeraj S. K.<sup>1</sup>, Yash Ingle<sup>1</sup>, Pruthwik Mishra<sup>1</sup>

<sup>1</sup>Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology, India

u23ai050@coed.svnit.ac.in, u23ai062@coed.svnit.ac.in, pruthwikmishra@aid.svnit.ac.in

## Abstract

This paper describes the curation of a high-quality parallel corpus from OPUS Books and the fine-tuning of Meta’s NLLB-200 models for German–English translation. We detail a dataset extraction pipeline that filters 45k raw OPUS sentence pairs to 25k high-quality examples through encoding validation, length filtering, language identification, deduplication, and quality scoring. We present a reproducible fine-tuning methodology optimized for consumer GPUs using mixed precision, gradient accumulation, and gradient checkpointing. Our fine-tuned NLLB-600M model achieves statistically significant improvements (+3.4 BLEU, +2.6 chrF++, -3.2 TER) over the base model on literary and technical German-English translation. We demonstrate practical deployment by integrating the finetuned model into LinguaHaru, an existing open-source document translation system, achieving format-preserving translation across PDF and Office documents. All dataset preparation scripts, training code, and evaluation tools are provided for reproducibility at GitHub <sup>1</sup>. The fine-tuned model is available at HuggingFace <sup>2</sup>.

Index Terms: neural machine translation, NLLB, dataset preparation, fine-tuning, OPUS

## 1. Introduction

Neural machine translation (NMT) has achieved remarkable progress through large-scale multilingual models like NLLB-200 [1], M2M-100 [2], and mBART [3]. However, these general-purpose models often underperform on specific domains or language pairs due to limited exposure to domainspecific vocabulary, style, and discourse patterns during pretraining. Fine-tuning on targeted parallel corpora has emerged as an effective approach to domain adaptation [4, 5].

This paper addresses the practical challenge of fine-tuning NLLB models for German–English literary and technical translation using freely available OPUS parallel corpora [6]. We focus on three key aspects: (1) systematic dataset curation from noisy web-scale corpora, (2) memory-efficient fine-tuning on consumer hardware, and (3) practical deployment in a document translation pipeline.

## 1.1. Key Contributions

Our work makes the following contributions:

• A systematic dataset extraction methodology for OPUS Books, applying multi-stage filtering to produce a curated

25k German–English parallel corpus from 45k raw sentence pairs

• A reproducible fine-tuning pipeline for NLLB-600M optimized for 4GB consumer GPUs through mixed-precision training, gradient accumulation, and checkpointing

• Quantitative evaluation with statistical significance testing, demonstrating +3.4 BLEU improvement over the base model

• Practical integration into LinguaHaru [25], an open-source document translation system, enabling format-preserving translation across multiple document types

• Complete open-source release of dataset preparation, training, and evaluation scripts with pinned dependencies

## 1.2. Motivation

Domain adaptation through fine-tuning addresses several limitations of general-purpose NMT models: (1) limited exposure to domain-specific terminology and stylistic conventions, (2) suboptimal performance on specialized content due to distribution mismatch between pre-training and target domains, and (3) need for offline deployment in privacy-sensitive environments [7, 8]. Our work demonstrates that carefully curated small-scale datasets (25k pairs) combined with memory-efficient training techniques can produce substantial quality improvements on consumer hardware, making domain adaptation accessible beyond large research institutions.

## 2. Related Work

## 2.1. Multilingual Neural Machine Translation

The transformer architecture [21] enabled training of largescale multilingual NMT systems. M2M-100 [2] pioneered direct many-to-many translation without English pivoting across 100 languages. mBART [3] demonstrated the effectiveness of denoising pre-training for multilingual sequence-to-sequence models. The NLLB project [1] extended this to 200+ languages through specialized data mining, iterative back-translation [9], and careful balancing of high- and low-resource languages. NLLB models are available in multiple sizes (600M, 1.3B, 3.3B, 54B parameters), with distilled variants offering practical deployment tradeoffs [10].

## 2.2. Domain Adaptation for NMT

Domain adaptation techniques for NMT include continued training on in-domain data [11], mixed fine-tuning with regularization [4], and instance weighting [12]. Recent work on parameter-efficient fine-tuning through adapters [13] and LoRA [14] enables domain adaptation with minimal memory overhead for decoder-only models, though application to encoderdecoder architectures remains less mature. Our work follows the continued training approach with conservative hyperparameters to balance adaptation and catastrophic forgetting [15].

## 2.3. Parallel Corpus Construction

OPUS [6] aggregates parallel corpora from diverse sources including legislative documents (Europarl [16]), subtitles (OpenSubtitles), literature (Books), and web-crawled content (ParaCrawl [17]). Data quality significantly impacts fine-tuning effectiveness [18]. Prior work has shown that careful filtering (length ratios, language identification [19], deduplication) improves downstream performance more than simply increasing corpus size [20]. We apply aggressive multi-stage filtering to extract high-quality subsets optimized for literary translation.

## 3. Deployment Context

Our fine-tuned NLLB model is deployed within LinguaHaru [25], an existing open-source document translation system. LinguaHaru implements a four-stage pipeline: (1) extraction of text and layout from PDF/Office documents using PyMuPDF and python-docx, (2) segmentation into translation units with provenance tracking, (3) translation via pluggable backends (LLM APIs, local models, or NLLB), and (4) reconstruction preserving original formatting. The modular architecture allows transparent replacement of translation backends without modifying the document processing pipeline. Our contribution is the finetuned NLLB backend, while the format-preserving infrastructure is provided by the existing LinguaHaru codebase.

## 4. Dataset Collection from OPUS

We extract and curate a German–English parallel corpus from OPUS Books [6] for fine-tuning NLLB models. OPUS Books contains professionally translated literary works with high alignment quality, making it suitable for literary and technical domain adaptation.

## 4.1. Extraction Pipeline

We targeted the OPUS Books subcorpus for its curated literary translations with sentence-level alignments. Our extraction script (scripts/prepare opus dataset.py) implements an eight-stage filtering pipeline:

1. Corpus Download and Initialization: Download or locate OPUS Books parallel files for German–English from the OPUS API. The corpus is distributed in TMX (Translation Memory eXchange) format with pre-aligned sentence pairs.

2. Encoding Validation: Parse aligned sentence pairs and remove pairs with non-UTF8 or malformed encodings. This step filters approximately 2% of raw pairs that contain encoding artifacts or corrupted characters.

3. Length Filtering: Apply preliminary length filters to retain pairs where both source and target are between 10 and 200 characters. This removes fragments (headlines, author names) and excessively long segments that may be misaligned paragraphs rather than sentences. The 10-200 character range captures typical literary sentence lengths while excluding outliers.

4. Length Ratio Filtering: Remove pairs with extreme length ratios (target/source outside [0.5, 2.0]). German and English have relatively similar sentence structures; extreme ratios typically indicate alignment errors or one-to-many sen-

tence mappings.

5. Language Identification: Apply language detection using the langdetect library to confirm language assignment. Pairs where either source or target are detected as a different language (with confidence > 0.9) are removed. This filters code-switched segments and remaining alignment errors.

6. Deduplication: Remove exact duplicate pairs using hashbased deduplication. Then apply fuzzy deduplication using normalized Levenshtein distance (threshold 0.95) to remove near-duplicates that differ only in punctuation or whitespace. Literary corpora often contain repeated boilerplate (copyright notices, chapter headings) that should not dominate the training distribution.

7. Quality Scoring: Compute a quality score for each pair based on: (a) character overlap ratio between source and target (penalizing untranslated content), (b) presence of unusual characters or markup artifacts, and (c) sentence-level perplexity from a pre-trained language model. Pairs below the 10th percentile quality score are discarded.

8. Stratified Splitting: Sort remaining pairs by length and perform stratified sampling to ensure balanced length distributions across splits. Randomly shuffle within strata and split into train (80%), validation (10%), and test (10%). This ensures validation and test sets cover the full range of sentence lengths seen in training.

The resulting dataset contains 25,000 high-quality sentence pairs after filtering from an initial pool of approximately 45,000 raw pairs. The scripts save JSONL files (one pair per line) and a single combined JSON used by the fine-tuning pipeline. Dataset splits and metadata are included in the repository’s data/ directory with MD5 checksums for reproducibility.

## 4.2. Filtering rationale and statistics

The chosen filters prioritize alignment quality over raw size, because noisy sentence pairs harm fine-tuning on relatively small datasets. After filtering, the corpus statistics were approximately: We report the key dataset statistics in Table 1. These numbers come directly from our extraction and filtering pipeline and are included with the data artifacts in the repository.

The table summarizes the exact split counts and average character lengths for source (German) and target (English). These aggregated statistics are also written as JSON metadata alongside the processed splits so they can be re-used by downstream training and analysis scripts.

## 5. Fine-tuning NLLB

This section describes the training recipe, tokenizer handling, and practical engineering decisions used to fine-tune an NLLB variant on the OPUS-derived 25k corpus.

## 5.1. Base model and tokenizer

We used facebook/nllb-200-distilled-600M as the base for fine-tuning because it offers a reasonable quality / memory tradeoff for consumer GPUs. Important tokenizer considerations included setting tokenizer.src lang and using tokenizer.as target tokenizer() when encoding targets to ensure correct language tokens and target-side tokenization behavior for NLLB’s sentencepiece tokenizer.

Practical fixes implemented in the training script (scripts/finetune nllb.py) addressed common issues encountered during development:

Table 1: OPUS-derived dataset statistics afterfiltering
<table><tr><td>Split</td><td>#Pairs</td><td>Avg. DE chars</td><td>Avg. EN chars</td></tr><tr><td>Train</td><td>20,000</td><td>175</td><td>150</td></tr><tr><td>Validation</td><td>2,500</td><td>172</td><td>149</td></tr><tr><td>Test</td><td>2,500</td><td>168</td><td>154</td></tr><tr><td>Total</td><td>25,000</td><td>171</td><td>151</td></tr></table>

Table 2: Training hyperparameters and practical settings
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Base model</td><td>facebook/nllb-200-distilled-600M</td></tr><tr><td>Per-device batch size</td><td>1–4 (GPU dependent)</td></tr><tr><td>Gradient accumulation</td><td>32</td></tr><tr><td>Effective batch size</td><td>32</td></tr><tr><td>Learning rate</td><td>2 ×  $1 0 ^ { - 5 }$  with linear warmup</td></tr><tr><td>Warmup</td><td>10% of steps</td></tr><tr><td>Max sequence length</td><td>512 tokens</td></tr><tr><td>Epochs</td><td>3 (early stopping)</td></tr><tr><td>Optimizer</td><td>AdamW, weight decay 0.01</td></tr><tr><td>Mixed precision</td><td>FP16</td></tr><tr><td>Tokenizer</td><td>SentencePiece with language codes</td></tr><tr><td>Checkpointing</td><td>Best by validation BLEU</td></tr><tr><td>Logging</td><td>Console + file</td></tr></table>

• Explicitly set source/target language codes (e.g., deu Latn → eng Latn) to avoid tokenizer warnings and incorrect language prefixes.

• Use small per-device batch sizes (1–4) and compensate with gradient accumulation when GPU memory is constrained.

• Disable external TensorBoard reporting by default to avoid runtime dependency problems; logging to the console and local files were used instead.

## 5.2. Training hyperparameters

The main training hyperparameters used in the reported runs were: We summarize the principal hyperparameters and practical settings used during fine-tuning in Table 2. These values are chosen to balance stability, training time, and memory constraints on consumer GPUs.

These settings are also parameterized in the provided training script (scripts/finetune nllb.py) so that experiments can be re-run with minimal modification.

## 5.3. Preprocessing and data pipeline

Preprocessing uses the HuggingFace datasets API for efficient batched operations and memory-mapped dataset handling. The complete pipeline consists of:

1. Data Loading: Load JSONL splits produced by the OPUS preparation script using the datasets.load dataset function. Each example contains source, target, and optional metadata fields (quality score, length, original corpus ID).

2. Tokenization: Tokenize source sentences with tokenizer(src lang=’deu Latn’) and target sentences using tokenizer.as target tokenizer() context manager to ensure correct language-specific token prefixes and tokenization behavior. NLLB’s sentencepiece tokenizer requires explicit language codes to generate proper

language-identification tokens.

3. Truncation and Padding: Apply truncation to max length=512 tokens, which accommodates 95% of sentence pairs in our corpus. Longer sentences are truncated from the right, potentially losing tail context but preserving sentence beginnings which typically contain main clauses in German.

4. Label Preparation: Prepare decoder input IDs and labels by shifting target sequences appropriately. Labels for padding tokens are set to -100 to exclude them from loss computation following standard seq2seq training conventions.

5. Batching: Assemble DataCollatorForSeq2Seq to produce dynamically-padded batches. The collator pads sequences to the longest example in each batch rather than the global maximum, significantly reducing computational overhead for batches containing mostly short sentences.

6. Caching: Enable dataset caching (.map() with cache file name) to avoid re-tokenizing on subsequent training runs or when resuming from checkpoints.

Checkpointing strategy uses validation BLEU and chrF++ scores computed every epoch. We save the best model by validation BLEU and the last checkpoint for potential training resumption. Early stopping triggers if validation BLEU does not improve for 2 consecutive evaluations, preventing overfitting on the relatively small training set.

Evaluation Metrics: We compute BLEU [23] using sacre-BLEU with default tokenization (13a), chrF++ (character ngram F-score with word bigrams), and TER (Translation Edit Rate). All metrics are computed on detokenized outputs to reflect real-world translation quality rather than tokenization artifacts.

## 5.4. GPU and runtime considerations

Fine-tuning transformer models on consumer GPUs requires careful memory management and optimization. Our implementation addresses several practical challenges:

Memory Constraints: The NLLB-600M model requires approximately 2.4GB for model weights in FP32. With optimizer states (AdamW maintains two momentum buffers per parameter), gradients, and activation memory, a naive implementation would exceed 10GB. Consumer GPUs like the RTX 3050 (4GB) or GTX 1660 (6GB) cannot accommodate this without optimization.

Mixed-Precision Training: We use automatic mixed precision (AMP) with FP16 activations and gradients, reducing memory footprint by nearly 50%. Loss scaling prevents gradient underflow for small gradient values. This is enabled through HuggingFace Trainer’s fp16=True flag, which internally uses PyTorch’s native AMP implementation.

Gradient Accumulation: With per-device batch size limited to 1-2 on 4GB GPUs, we use gradient accumulation over 32 steps to simulate an effective batch size of 32. This trades training time for memory efficiency: gradients are accumulated across multiple forward-backward passes before a single optimizer step.

Gradient Checkpointing: Activation checkpointing (also called gradient checkpointing) trades computation for memory by recomputing activations during the backward pass rather than storing them. This reduces peak memory by 30-40% at the cost of approximately 20% slower training. Enable via gradient checkpointing=True in the model configuration.

Training Time: On an RTX 3050 (4GB), training for 3 epochs over 20k pairs with batch size 1 and gradient accumulation 32 requires approximately 6-8 hours. This includes validation evaluation every epoch. On higher-end consumer GPUs (RTX 3080, RTX 4070) with 10-12GB memory, batch size can increase to 4-8, reducing training time to 2-3 hours.

Disk I/O Optimization: We use memory-mapped datasets (HuggingFace datasets library default) to avoid loading the entire dataset into RAM. Checkpoint saving is configured to keep only the best model and last checkpoint, limiting disk usage to approximately 5GB.

These optimizations make fine-tuning feasible on consumer hardware while maintaining training stability and convergence. Full training runs on larger models (NLLB-1.3B, 3.3B) or substantially larger datasets will require cloud instances with 16GB+ GPUs or multi-GPU training.

## 6. Experimental Results

## 6.1. Training Setup

We fine-tuned facebook/nllb-200-distilled-600M on our curated 25k German–English corpus using the hyperparameters in Table 2. Training was performed on an NVIDIA RTX 3050 (4GB) with FP16 mixed precision, batch size 1, and gradient accumulation over 32 steps. The model converged after 3 epochs (1,875 total steps, approximately 6-8 hours), with best validation BLEU achieved at epoch 2. Peak GPU memory usage was 3.8GB (95% utilization).

## 6.2. Translation Quality Evaluation

We evaluated translation quality on the held-out test set (2,500 pairs) using BLEU [23], chrF++, and TER metrics computed with sacreBLEU. Table 3 shows results with bootstrap confidence intervals (1,000 iterations). The fine-tuned model achieves statistically significant improvements (p < 0.01, paired bootstrap test) across all metrics.

Table 3: Translation quality on held-out test set (2,500 pairs)
<table><tr><td>Model</td><td>BLEU</td><td>chrF++</td><td>TER</td></tr><tr><td>NLLB-200 Base</td><td>32.4</td><td>58.7</td><td>48.2</td></tr><tr><td>Fine-tuned (25k)</td><td>35.8</td><td>61.3</td><td>45.0</td></tr><tr><td>Improvement</td><td>+3.4</td><td>+2.6</td><td>-3.2</td></tr><tr><td>95% CI (Bootstrap)</td><td>±0.6</td><td>±0.4</td><td>±0.5</td></tr></table>

TER (Translation Edit Rate) is included as an additional complementary metric to BLEU and chrF++; lower TER indicates fewer edits required to match the reference. All reported numbers are computed with sacreBLEU/chrf/TER implementations from the sacrebleu package (version 2.3.1) used in the repository’s evaluation script.

Statistical Significance: We performed bootstrap resampling (1,000 iterations) to compute 95% confidence intervals. The improvements are statistically significant (p < 0.01, paired bootstrap test), indicating that fine-tuning provides consistent quality gains beyond random variation.

Metric Interpretation: The +3.4 BLEU improvement is substantial for literary domain adaptation. For context, professional human translation typically scores 55-60 BLEU against a single reference, while base NLLB’s 32.4 places it in the ”usable but requires post-editing” range. Our fine-tuned 35.8 approaches ”good first draft” quality for literary content.

Length-Stratified Analysis: We analyzed performance by sentence length:

Table 4: BLEU scores by sentence length category
<table><tr><td>Length (tokens)</td><td>Count</td><td>Base</td><td>Fine-tuned</td></tr><tr><td>Short (5-15)</td><td>612</td><td>38.2</td><td>41.5 (+3.3)</td></tr><tr><td>Medium (16-40)</td><td>1,456</td><td>33.1</td><td>36.9 (+3.8)</td></tr><tr><td>Long (41-100)</td><td>432</td><td>26.7</td><td>30.2 (+3.5)</td></tr></table>

Fine-tuning provides consistent improvements across all length ranges, with slightly larger gains on medium-length sentences (16-40 tokens) that represent typical literary sentence complexity.

## 6.3. Deployment in LinguaHaru

We integrated the fine-tuned model into LinguaHaru by configuring the model path in system config.json. The modular architecture required no code changes; switching between base and fine-tuned NLLB models involves only configuration updates. Translation speed on representative documents (RTX 3050, batch size 8) ranges from 13.9 to 17.1 pages/minute, enabling practical offline document translation.

## 6.4. Error Analysis

Manual evaluation of 100 randomly-sampled test translations revealed error distributions shown in Table 5. Fine-tuning reduces total errors from 69 to 46, with largest improvements in awkward phrasing (-33%) and mistranslations (-50%). Common remaining issues include rare technical terminology not in the literary corpus, German compound nouns requiring cultural knowledge, and pronoun resolution across sentence boundaries. These limitations are inherent to sentence-level translation and could be addressed through document-level context modeling or terminology glossaries [24].

Table 5: Error type frequency (N=100 test samples)
<table><tr><td>Error Type</td><td>Base</td><td>Fine-tuned</td></tr><tr><td>Rare terminology</td><td>18</td><td>12</td></tr><tr><td>Awkward phrasing</td><td>24</td><td>16</td></tr><tr><td>Gender/pronoun errors</td><td>9</td><td>6</td></tr><tr><td>Omissions</td><td>7</td><td>4</td></tr><tr><td>Mistranslations</td><td>6</td><td>3</td></tr><tr><td>Total errors</td><td>64</td><td>41</td></tr></table>

• Academic Publishing: Translation of research papers maintaining LaTeX formatting and citations

• Technical Documentation: Multilingual software manuals preserving screenshots and code blocks

• Legal Documents: Contract translation with strict formatting requirements

• Corporate Communications: Internal reports and presentations requiring offline processing

Each domain benefits from the combination of format preservation, customizable translation quality through finetuning, and privacy-preserving offline operation.

## 6.5. Future Directions

Potential improvements include:

• Multi-language fine-tuning using adapter-based approaches for efficient parameter sharing

• Integration of terminology glossaries directly into NLLB tokenization for consistent term translation

• Quantization for reduced memory footprint (4-bit/8-bit) enabling deployment on mobile devices

• Automatic domain detection for intelligent model selection based on document content

• Support for document-level context in longer translations using memory-efficient attention mechanisms

• Integration with translation memory systems for leveraging previous translations

• Active learning pipelines for continuous model improvement from user corrections

## 7. Conclusion

This paper demonstrates practical fine-tuning of NLLB-200 for German–English translation using curated parallel corpora from OPUS Books. Our contributions include: (1) a systematic eightstage dataset extraction methodology reducing 45k raw OPUS pairs to 25k high-quality examples through encoding validation, length filtering, language identification, deduplication, and quality scoring; (2) memory-efficient fine-tuning on 4GB consumer GPUs using FP16, gradient accumulation, and gradient checkpointing, achieving statistically significant improvements (+3.4 BLEU, p < 0.01); and (3) practical deployment in LinguaHaru, demonstrating format-preserving document translation for PDF and Office formats.

The complete workflow, dataset preparation, training scripts, and evaluation code is released open-source with pinned dependencies for reproducibility. Our methodology is applicable to other language pairs and domains within the OPUS ecosystem. Future work includes expanding to additional language pairs, integrating terminology glossaries [24], and exploring document-level context modeling to address remaining limitations in pronoun resolution and long-range dependencies.

## 8. References

[1] NLLB Team et al., “No Language Left Behind: Scaling Human-Centered Machine Translation,” arXiv preprint arXiv:2207.04672, 2022.

[2] A. Fan et al., “Beyond English-Centric Multilingual Machine Translation,” Journal of Machine Learning Research, vol. 22, pp. 1–48, 2021.

[3] Y. Liu et al., “Multilingual Denoising Pre-training for Neural Machine Translation,” Transactions of the Association for Computational Linguistics, vol. 8, pp. 726–742, 2020.

[4] C. Chu, R. Dabre, and S. Kurohashi, “An Empirical Comparison of Domain Adaptation Methods for Neural Machine Translation,” in Proc. 55th Annual Meeting of the Association for Computational Linguistics, 2017, pp. 385–391.

[5] M. A. Farajian et al., “Multi-Domain Neural Machine Translation through Unsupervised Adaptation,” in Proc. 2nd Conference on Machine Translation (WMT), 2017, pp. 127–137.

[6] J. Tiedemann, “OPUS – Parallel Corpora for Everyone,” in Proc. 19th Annual Conf. of the European Association for Machine Translation (EAMT), 2016.

[7] P. Koehn and R. Knowles, “Six Challenges for Neural Machine Translation,” in Proc. 1st Workshop on Neural Machine Translation, 2017, pp. 28–39.

[8] R. Sennrich, B. Haddow, and A. Birch, “Neural Machine Translation of Rare Words with Subword Units,” in Proc. 54th Annual Meeting of the Association for Computational Linguistics, 2016, pp. 1715–1725.

[9] R. Sennrich, B. Haddow, and A. Birch, “Improving Neural Machine Translation Models with Monolingual Data,” in Proc. 54th Annual Meeting ofthe Associationfor Computational Linguistics, 2016, pp. 86–96.

[10] G. Hinton, O. Vinyals, and J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv preprint arXiv:1503.02531, 2015.

[11] M.-T. Luong and C. D. Manning, “Stanford Neural Machine Translation Systems for Spoken Language Domains,” in Proc. International Workshop on Spoken Language Translation (IWSLT), 2015.

[12] R. Wang et al., “Instance Weighting for Neural Machine Translation Domain Adaptation,” in Proc. 2017 Conference on Empirical Methods in Natural Language Processing, 2017, pp. 1482–1488.

[13] A. Bapna and O. Firat, “Simple, Scalable Adaptation for Neural Machine Translation,” in Proc. 2019 Conference on Empirical Methods in Natural Language Processing, 2019, pp. 1538–1548.

[14] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” in Proc. International Conference on Learning Representations (ICLR), 2022.

[15] R. M. French, “Catastrophic Forgetting in Connectionist Networks,” Trends in Cognitive Sciences, vol. 3, no. 4, pp. 128–135, 1999.

[16] P. Koehn, “Europarl: A Parallel Corpus for Statistical Machine Translation,” in Proc. 10th Machine Translation Summit, 2005, pp. 79–86.

[17] M. Banon et al., “ParaCrawl: Web-Scale Acquisition of Parallel Corpora,” in Proc. 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 4555–4567.

[18] H. Khayrallah and P. Koehn, “On the Impact of Various Types of Noise on Neural Machine Translation,” in Proc. 2nd Workshop on Neural Machine Translation and Generation, 2018, pp. 74–83.

[19] A. Joulin et al., “Bag of Tricks for Efficient Text Classification,” in Proc. 15th Conference of the European Chapter of the Association for Computational Linguistics, 2017, pp. 427–431.

[20] H. Schwenk et al., “WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia,” in Proc. 16th Conference of the European Chapter of the Association for Computational Linguistics, 2021, pp. 1351–1361.

[21] A. Vaswani et al., “Attention is All You Need,” in Advances in Neural Information Processing Systems 30 (NIPS 2017), 2017, pp. 5998–6008.

[22] T. Wolf et al., “Transformers: State-of-the-Art Natural Language Processing,” in Proc. 2020 Conf. on Empirical Methods in Natural Language Processing: System Demonstrations, 2020, pp. 38–45.

[23] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: A Method for Automatic Evaluation of Machine Translation,” in Proc. 40th Annual Meeting of the Association for Computational Linguistics, 2002, pp. 311–318.

[24] G. Dinu et al., “Training Neural Machine Translation to Apply Terminology Constraints,” in Proc. 57th Annual Meeting of the Associationfor Computational Linguistics, 2019, pp. 3063–3068.

[25] YANG-Haruka, “LinguaHaru: Format-preserving document translation pipeline,” GitHub repository, https://github.com/YANG-Haruka/LinguaHaru, 2025.