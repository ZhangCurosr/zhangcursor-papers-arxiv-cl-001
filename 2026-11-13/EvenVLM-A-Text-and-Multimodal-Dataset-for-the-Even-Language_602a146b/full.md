# EvenVLM: A Text and Multimodal Dataset for the Even Language with Vision-Language Fine-Tuning

Daniil Makovetskii<sup>1</sup> and Ivan Bondarenko<sup>1</sup>

<sup>1</sup>Novosibirsk State University, Novosibirsk, Russia d.makovetskii@g.nsu.ru i.bondarenko@g.nsu.ru

## Abstract

We present EvenVLM, the first text and multimodal (VQA) dataset for Even (Lamut), an endangered Northern Tungusic language spoken by approximately 6,800 people in Siberia. Our dataset comprises 50,521 structured Even-Russian entries extracted from dictionary and phrasebook sources via GPT-4o Mini-based parsing, from which we synthetically generate 107,543 training examples spanning translation, grammar, reading comprehension, and semantic classification. To address tokenization inefficiency on Even’s agglutinative morphology, we propose a three-step tokenizer expansion algorithm that adds Even-specific graphemes (ӈ ө ӫ), whitelisted vocabulary, and BPE merges, reducing average tokens per word by 27% (4.99 to 3.64). Using LoRA fine-tuning of Qwen3-VL-4B, we adapt the model for text tasks (0.53% trainable parameters, rank 32) and separately for visual question answering (rank 64, frozen vision encoder). Our best configuration achieves 11.4% exact match and a 13.6 weighted score on text benchmarks, 0.1283 chrF for sentence translation, and 65% accuracy on VQA (up from 0% for the base model and text-only fine-tuning). We release our datasets, evaluation benchmarks, and training pipeline to support further research on Even and other low-resource languages.

## 1 Introduction

Even (Lamut) is a Northern Tungusic language spoken by approximately 6,800 people (out of 19,900 who identify as Even) across the Magadan, Kamchatka, and Sakha regions of Siberia, according to the 2020–2021 census (Interactive Atlas of Indigenous Peoples of the North, Siberia, and Far East<sub>,</sub> 2025). The language is classified as endangered: the number of native speakers has declined sharply over recent decades, and intergenerational transmission is limited. Despite the growing interest in multilingual AI, very few researchers work on Even in the context of machine learning, natural language processing, or data science. Consequently, any first steps in this direction—whether dataset creation, model adaptation, or evaluation— are valuable for documenting, preserving, and supporting the Even language. While large visionlanguage models (VLMs) have shown remarkable capabilities in high-resource languages (Bai et al. <sub>,</sub> 2025), their application to languages with fewer than 10,000 speakers remains virtually unexplored (Lupaşcu et al.<sub>,</sub> 2026).

This work directly addresses the ORACLE workshop’s mission of open reasoning across lan-<sub>guages</sub> and cultures. We contribute <sub>an</sub> <sub>open</sub> data and evaluation resource for a critically underrepresented language, enabling culturally grounded multilingual assessment of vision-language models (Lupaşcu et al. <sub>,</sub> 2026). By providing evaluation tools in Even itself—rather than through translation to Russian—we <sub>support</sub> trust in multilingual GenAI for indigenous communities, who can assess model outputs in their own language. Our pipeline is designed to be reproducible and adaptable to other low-resource agglutinative languages.

In this work, we address this gap by fine-tuning Qwen3-VL-4B, a 4.4B-parameter VLM, for Even language tasks spanning both textual and multimodal domains. Our contributions <sub>are</sub> threefold:

1. Dataset. We introduce the first text and VQA dataset for Even, comprising 50,521 dictionary entries synthetically expanded to 107,543 training examples, and a multimodal set of 189 images with 477 bilingual annotations, producing 1,908 VQA examples.

2. Tokenizer. We propose a three-step algorithm adapting the Qwen tokenizer to Even’s agglutinative morphology, reducing token fragmentation by 27%.

3. Pipeline. We release a complete, reproducible pipeline for adapting a vision-language model to a low-resource agglutinative language, covering tokenizer expansion, curriculum-based data synthesis, and two-stage LoRA fine-tuning.

All datasets, evaluation benchmarks, and the complete training pipeline are released in open access. Model weights will be made available upon acceptance.

## 2 Related Work

Lupascu et al. (Lupaşcu et al. 2026) survey multimodal models for low-resource languages and find fewer than ten papers applying VLMs to languages with under 10,000 speakers. Prior work is concentrated on high-resource languages (Bai et al. 2025; Liu et al. 2023; Chen et al. 2023), and standard tokenizers fragment low-resource words excessively (Sennrich et al. 2016; Rust et al. 2021). Even is a “left-behind” language with virtually no NLP resources (Joshi et al. 2020). For the purposes of this study, we rely on the primary available machine-readable sources: an Even-Russian dictionary (Kanyukova and Levin 2025) and a phrasebook (Sharina and Kuzmina 2014). Recent advances in low-resource synthetic supervision, such as R2-MultiOmnia and self-training approaches for multilingual reasoning (Ranaldi et al. 2025; Ranaldi and Pucci 2025), provide promising directions for data generation. These methods leverage large teacher models to autonomously generate reasoning trajectories. While this scales well and produces rich reasoning chains, it fundamentally relies on a teacher model with acceptable performance in the target language. For severely under-resourced languages like Even, where base VLMs yield 0% such selftraining pipelines face an insurmountable coldstart problem. In contrast, our template-based dictionary synthesis is deterministic and highly controllable. It allows us to enforce strict linguistic constraints and morphological accuracy without requiring a pre-trained teacher model, making it a more viable approach for languages at the zeroshot extreme of the <sub>resource spectrum,</sub> albeit at the cost of lower scalability compared to autonomous reasoning generation.

## 3 The Even Language and Data Collection

## 3.1 Language Profile

Even is historically also known by the exonym Lamut used especially in older ethnographic literature. The language (эвенский язык) belongs to the Northern Tungusic branch of the Tungusic language family. The dataset primarily represents the Eastern dialect <sub>group</sub> which is the most widely spoken dialect cluster (Malchukov 1995). A smaller portion of the data covers the Western dialect <sub>group</sub> which differs from Eastern varieties in phonology and lexicon. It is agglutinative, with words formed by concatenating a root with multiple suffixes marking case, number, possession, and verbal categories. The Cyrillic-based alphabet includes three graphemes not present in Russian: ӈ (eng), ө (oe), and ӫ (o-umlaut). These characters <sub>are</sub> absent from standard tokenizer <sub>vo-</sub> cabularies, motivating our adaptation in Section 5 <sub>.</sub>

## 3.2 Source Data

We collected Even language data from two sources:

• Kanyukova & Levin (2025) Even-Russian Dictionary. 256 pages containing headwords, translations, example sentences, and grammatical notes.

• Sharina & Kuzmina (2014) Russian-Even Phrasebook. 87 pages of thematic dialogues, phrases, and vocabulary.

## 3.3 Data Extraction Pipeline

PDF documents were processed with GPT-4o Mini (OpenAI<sub>,</sub> 2024) using specialized parsing prompts designed to handle complex, multi-level dictionary entries (headword translation example grammatical metadata). The structured output was manually verified and cleaned, producing a final <sub>corpus</sub> of 50,521 entries. To ensure the linguistic accuracy of our morphological generation templates and to resolve specific translation ambiguities, we consulted a linguistics specialist at a university in Yakutia who is a native Even speaker. Each entry contains four fields: even <sup>russian</sup><sub>,</sub> category<sub>,</sub> <sup>and</sup> source <sub>.</sub>

## 4 Data Synthesis

## 4.1 Template-Based Generation

From the 50,521-entry cleaned corpus, we generate training examples via templated transformations. Each dictionary entry is expanded into multiple examples of different task types. The full dataset comprises 107,543 training and 5,658 validation examples, organized in a three-stage curriculum:

• Stage 1 – Vocabulary (47,366 examples). Word-level translation in both directions (Even Russian, Russian Even) and phrase translation.

• Stage 2 – Grammar (18,977 examples). Declension, negation, plural formation, and verb conjugation exercises derived from morphological templates.

• Stage 3 – Comprehension (41,200 examples). Sentence translation, fill-in-the-blank, reading comprehension, and semantic classification, including few-shot demonstrations.

The task distribution is provided in Table 4 (Appendix A). Figure 1 shows the domain distribution of the source corpus, spanning 31 categories with the largest shares in verbs (23.1%) and general vocabulary (13.7%).

![](images/03f010fc3783aebb4e1830a3c7abbb44014035375c3dec1c0dabe04ed615a24c.jpg)  
Figure 1: Domain distribution of the Even corpus (50,521 entries, 31 categories).

## 4.2 Multimodal VQA Dataset

We collected 189 images related to Even culture and daily life (nature, reindeer herding, hunting, clothing, tools, food, household, folklore). Each image received bilingual (Even/Russian) questionanswer annotations, yielding 477 annotation pairs. By generating both short and long answer variants, we expanded these to 1,908 training examples. A held-out test set of 20 examples balanced across 10 categories provides the VQA evaluation benchmark.

## 5 Tokenizer Adaptation

The original Qwen3-VL-4B tokenizer (151,669 tokens) fragments Even words into an average of 4.99 tokens (median 5), with only 7.6% occupying two or fewer tokens. The three special Cyrillic characters (ӈ<sub>,</sub> ө<sub>,</sub> ӫ) are absent from the vocabulary, causing further fragmentation.

## 5.1 Expansion Algorithm

We propose the following algorithm:

Step 1: Special graphemes. Add ӈ<sub>,</sub> ө<sub>,</sub> and ӫ as individual tokens (3 tokens).

Step 2: Whitelist + BPE merging.

• Whitelist: 209 high-frequency Even words from the dictionary; 197 required merging (the remaining 12 were already single tokens).

• VQA answers: 101 recurrent answer words added <sub>as</sub> whole tokens.

• BPE : Iterative merging of frequent adjacent token pairs on the full Even text corpus (51,477 texts). Maximum 300 tokens total across this step, minimum pair frequency 10, <sub>over</sub> 5 rounds.

Step 3: Special character combinations. Extract bigrams and trigrams containing ӈ<sub>,</sub> ө<sub>,</sub> or ӫ from dictionary entries and add those meeting fre-<sub>quency</sub> thresholds. In the current <sub>corpus,</sub> no combination exceeded the threshold, so this step contributed 0 tokens at this <sub>stage</sub> but remains in the pipeline for future dataset expansion.

## 5.2 Results

The expansion added 303 tokens, growing the vocabulary to 151,972. Table 5 (Appendix A) summarizes the improvements.

The most frequent BPE tokens reflect Even’s agglutinative morphology: suffix “-н” (5,010 occurrences), roots “дэ” (4,882), “эй” (4,140), and “эл” (3,338). After expansion, resize\_token\_embeddings(151972) enlarges the input embeddings and language model head, initializing new rows via the default Transformers approach.

## 6 Model Architecture and Training

## 6.1 Base Model

Qwen3-VL-4B (Bai et al. 2025) is a 4.4Bparameter vision-language model with a Transformer decoder (text) and a ViT image encoder (336  336 input, 14  14 patches). It supports multilingual inputs and multimodal generation natively.

## 6.2 Stage 1: Text Fine-Tuning with LoRA

LoRA configuration. We apply LoRA (Hu et al. <sub>,</sub> 2021) to the <sub>query,</sub> key, value, and output projection matrices of all attention layers with rank $r \quad = \quad 3 2 , \ \alpha \quad = \quad 6 4 .$ and dropout 0.1. This adds 23,592,960 trainable parameters (0.53% of 4,461,408,768 total). Other parameters remain frozen. The rank is chosen <sub>as a</sub> trade-off between adaptation capacity and GPU memory (V100 32GB): $r = 3 2$ fits within the 32 GB budget while providing sufficient capacity for textdomain adaptation.

Training setup. Training uses LoRA rank $r =$ 32 $\alpha = 6 4 .$ dropout 0.1, effective batch size 32 (gradient accumulation $2 \times )$ , learning rate $2 \times 1 0 ^ { - 5 }$ with cosine decay, and 6 epochs on a single V100- SXM2-32GB (full details in Table 6<sub>,</sub> Appendix A).

Training dynamics. The initial loss of 3.9 decreases rapidly during the first epoch. Final training loss reaches approximately 1.019; validation loss improves from 1.5 to a best of 1.179 at step 20,000 (Figure 2). Total training time is 6.8 hours. The best checkpoint is saved, then LoRA weights are merged into the base model.

![](images/c3f2519b0df5630e985c45a7fa8deaa0a6d2be2e902ab167cbc77a70a102eae0.jpg)  
Figure 2: Training and validation loss curves for Stage 1 (text LoRA fine-tuning). Validation loss reaches minimum of 1.179 at epoch 6.

## 6.3 Stage 2: VQA Fine-Tuning

LoRA configuration. The two-stage design allows different LoRA configurations per stage: Stage 2 applies a higher rank $( r ~ = ~ 6 4 , ~ \alpha ~ =$ 128<sub>,</sub> dropout 0.05) across seven projection matrices (q, k, v, o, gate, up, down), compared to Stage 1’s $r \ = \ 3 2$ across four matrices. This increase reflects three factors: (i) multimodal learning requires cross-modal alignment capacity beyond text-only adaptation, (ii) the frozen vision encoder places additional burden on language-side parameters to learn visual grounding, and (iii) technical constraints (GPU memory, training time) set a lower ceiling for Stage 1, where the larger text dataset (50,521 entries vs. 1,908 VQA pairs) would have made r = 64 prohibitively expensive. The vision encoder is frozen (zero trainable parameters), and we wrap its forward pass with torch.no\_grad() to avoid unnecessary autograd graph construction and activation storage. Additionally, we set vision\_model.eval() to disable dropout in ViT. These optimizations accelerate the forward pass by an estimated $2 { - } 3 \times .$ <sub>,</sub> as ViT forward was the primary bottleneck (97% of step time prior to optimization).

Training setup. VQA training uses batch size 8 (effective 8, no gradient accumulation) for 3 epochs, learning rate $1 \times 1 0 ^ { - 4 }$ with cosine decay, maximum sequence length 2,048 tokens, and image resolution capped at $5 1 2 \times 5 1 2$ pixels (262,144 pixels maximum).

Training dynamics. Loss decreases from 10.2 to 6.1, indicating incomplete convergence due to limited VQA data volume. Nevertheless, the model acquires basic multimodal capability, as shown in Section 7<sub>.</sub>

## 7 Evaluation and Results

## 7.1 Benchmark Suite and Metrics

Our text evaluation set comprises 100 examples spanning reading comprehension (RC, 30), classification (CL, 20), grammar (GR, 20), and sentence translation (TR, 30). Standard metrics for each task type are unsuitable for an agglutinative low-resource language like Even. BLEU (Papineni et al. 2002) requires multiple reference translations and performs poorly on short sequences. ROUGE (Lin 2004) is designed for summarization, not morphological accuracy. CER and WER <sub>operate</sub> at the character and word level but conflate root and affix errors equally. For Even, a correct root with a wrong suffix should receive partial credit—this is essential for an agglutinative language where a single word encodes multiple morphological categories and full vocabulary coverage is infeasible.

For short-answer tasks (RC, CL, GR), we therefore report Exact Match (binary correctness, with multi-reference <sub>support</sub> via $^ { 6 6 , 9 9 } )$ and Edit Similarity (normalized Levenshtein ratio, 1 edit\_dist $/ ( | s _ { 1 } | + | s _ { 2 } | ) )$ , classified into three categories: exact $( \geq 1 . 0 )$ , partial $( \geq 0 . 7 5 )$ , and wrong $( < ~ 0 . 7 5 )$ , with weights 1.0, 0.5, and 0.0 respectively. The 0.75 threshold is chosen empirically: for agglutinative Even, a correct root with one incorrect suffix yields a Levenshtein ratio of approximately 0.75–0.85, which this boundary captures as partial credit. The Weighted Score is the sum of these weights across all examples. For TR, we use chrF (β = 2 n-grams 1–6) (Popović 2015), which <sub>operates on</sub> character <sub>n-grams</sub> and is <sub>ro-</sub> bust to Even’s relatively flexible word order while capturing morphological overlap that word-level metrics would miss. VQA accuracy is measured via Exact Match on 20 test examples, appropriate given the short-answer format (1–3 words).

We evaluate three configurations:

1. Base: Untrained Qwen3-VL-4B.

2. Base + Tokenizer: Tokenizer expanded but no fine-tuning.

3. Base + Tokenizer + LoRA: Full pipeline (our method).

The base model cannot produce Even output; tokenizer expansion alone slightly degrades performance (the 303 new tokens receive randomly initialized embeddings that carry no linguistic information, adding noise during decoding); our full method substantially improves all metrics by adapting these embeddings through fine-tuning.

Table 1: Main comparative results. RC = Reading Comprehension, CL = Classification, GR = Grammar, TR = Sentence Translation.
<table><tr><td>Metric</td><td>Base</td><td>+ Tokenizer</td><td>+ LoRA</td></tr><tr><td>Exact Match</td><td>0.0</td><td>0.0</td><td>11.4</td></tr><tr><td>Weighted Score</td><td>2.1</td><td>0.7</td><td>13.6</td></tr><tr><td>chrF</td><td>0.0723</td><td>0.0694</td><td>0.1283</td></tr></table>

## 7.2 VQA Results

Table 2 reports VQA accuracy across configurations. Text-only training (Stage 1) yields zero VQA capability—multimodal performance requires multimodal training. Stage 2 (VQA LoRA) achieves 65% accuracy (13/20), confirming that 1,908 examples suffice for basic visuallexical mapping despite incomplete convergence. Note that due to the small test set size (n = 20), these <sub>accuracy</sub> estimates have wide confidence intervals (approx. 43.2%–82.0% at the 95% confidence level). Expanding these evaluation sets is a primary goal for future work.

## 7.3 Qualitative Observations

Table 3 shows sample model outputs across task types, illustrating successful predictions as well as typical errors.

Table 2: VQA evaluation results (Exact Match on 20 tests).
<table><tr><td>Configuration</td><td>Correct</td><td>Accuracy</td></tr><tr><td>Base</td><td>0/20</td><td>0%</td></tr><tr><td>Base + Stage 1</td><td>0/20</td><td>0%</td></tr><tr><td>Full pipeline (Stage 1 + Stage 2)</td><td>13/20</td><td>65%</td></tr></table>

Table 3: Qualitative examples of model outputs.
<table><tr><td colspan="2">Task Input</td><td>Target</td><td>Output</td></tr><tr><td></td><td>L  -? </td><td></td><td>(EM)</td></tr><tr><td rowspan="2"></td><td>   ?</td><td></td><td> ()</td></tr><tr><td>RC  TTT?</td><td></td><td>TTTH (wrong)</td></tr><tr><td></td><td>   </td><td>TM</td><td></td></tr><tr><td colspan="2"> ?</td><td> (4)</td><td>(partial) (3) (wrong)</td></tr></table>

Translation. The model translates frequent Even words (reindeer herding, household items, nature) confidently; rare words are weaker. Notably, the model sometimes translates words not present in the training set<sub>,</sub> constructing plausible Even forms from known morphological elements and analogies—a valuable property for an agglutinative language where full vocabulary <sub>coverage</sub> is infeasible.

Classification. The model performs four semantic classification types correctly: noun type (animal / plant / object), kinship (relative / nonrelative), part of speech (pronoun / noun), and domain (natural / household).

VQA by category. Per-category VQA accuracy is expected to be highest for simple “what” questions on concrete categories (food, instruments, folklore), and lower for context-dependent questions (“what season?”), counting (“how many?”), and verbose descriptions. The partial convergence (loss 10.2 6.1) suggests substantial room for improvement with more data.

## 8 Error Analysis

We observe three error types: (1) Semantic substitution—the model <sub>outputs a</sub> related but incorrect word (e.g., a synonym from another dialect), indicating partial lexical acquisition; (2) Grammatical inaccuracies—correct roots with incorrect inflections, partially captured by Weighted Levenshtein; and (3) Translation incompleteness for long sentences with unfamiliar words. In VQA, the model produces semantically plausible but inexact answers (e.g., “илан” (three) instead of “дыгэн” (four)), suggesting basic visual-lexical associations that lack precision due to limited data.

## 9 Conclusion and Future Work

We presented EvenVLM, the first text and multimodal dataset for Even, with a full pipeline for fine-tuning a VLM using tokenizer adaptation and LoRA. Using 50K dictionary entries expanded to 107K synthetic examples and 1,908 VQA examples, we adapted a 4.4B model for translation, grammar, comprehension, classification, and VQA tasks. All datasets, benchmarks, and code are released openly. Limitations include small VQA data volume, limited reading comprehension data, lack of parallel text corpora, Eastern dialect skew (with limited coverage of Western variants), and the need for large-scale native speaker validation of the full synthetic <sub>corpus.</sub> Furthermore, we acknowledge that the current evaluation scale (n=20 for VQA and 100 for text benchmarks) introduces statistical noise and limits the <sub>power</sub> of our empirical claims. Expanding these evaluation sets is a primary goal for future work. Future work includes expanding both the text corpus and the multimodal VQA dataset in size and quality, with <sub>a</sub> focus <sub>on open-access resources</sub> for model training and adaptation; involving a larger number of native Even speakers in data collection, validation, and evaluation; expanding cross-dialect <sub>rep-</sub> resentation; and extending the pipeline to Evenki— a closely related Northern Tungusic language— which shares typological features with Even and could benefit from similar adaptations.

## Ethics Statement

All dataset sources are publicly available dictionaries and phrasebooks cited in the references. VQA images are sourced from open internet resources and cultural heritage collections. No personal data, recordings of speakers, or sensitive information is included. The data is released under CC-BY-NC 4.0 to protect the linguistic heritage of the Even community from commercial exploitation. We acknowledge the Even community’s rights over their linguistic data and commit to engaging community representatives in future dataset iterations and val-

idation.

## Acknowledgments

We thank the Even language community for preserving their linguistic heritage, and the creators of the Kanyukova and Levin dictionary and the Sharina and Kuzmina phrasebook.

## References

S. Bai, Y. Cai, R. Chen, et al. 2025. Qwen3-vl technical report arXiv preprint

Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, et al. 2023. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks arXiv preprint<sub>.</sub>

E. J. Hu, Y. Shen, P. Wallis, et al. 2021. LoRA: Lowrank adaptation of large language models<sub>.</sub> In Proceedings of ICLR<sub>.</sub>

Interactive Atlas of Indigenous Peoples of the North, Siberia, and Far East. 2025. Evens — 2020–2021 <sub>census</sub> data

P. Joshi, S. Santy, A. Budhiraja, K. Bali, and M. Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world<sub>.</sub> In Proceedings of ACL<sub>.</sub>

U. V. Kanyukova and V. I. Levin. 2025. Even-Russian Dictionary (Эвенско-русский словарь)<sub>.</sub> Almaz-Graf, St. Petersburg.

C.-Y. Lin. 2004. ROUGE: A package for automatic evaluation of summaries<sub>.</sub> In Proc. of ACL Workshop<sub>.</sub>

H. Liu, C. Li, Q. Wu, and Y. J. Lee. 2023. Improved baselines with visual instruction tuning Advances in Neural Information Processing Systems (NeurIPS)

M. Lupaşcu, A.-C. Rogoz, M.-S. Stupariu, et al. 2026. Large multimodal models for low-resource languages: A survey<sub>.</sub> Information Fusion 131:104189.

A. L. Malchukov. 1995. Even<sub>.</sub> Languages of the World/Materials 12. LINCOM Europa, München.

OpenAI. 2024. GPT-4o Mini: Advancing cost-efficient intelligence <sub>.</sub>

K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. 2002. BLEU: <sub>a</sub> method for automatic evaluation of <sub>ma-</sub> chine translation In Proc. ofACL

M. Popović. 2015. chrF: character n-gram F-score for automatic MT evaluation In Proceedings of the Tenth Workshop on Statistical Machine Translation <sub>pages</sub> 392–395.

L. Ranaldi and G. Pucci. 2025. Multilingual reasoning via self-training<sub>.</sub> In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)<sub>.</sub>

L. Ranaldi, F. Ranaldi, and G. Pucci. 2025. R2- MultiOmnia: Leading multilingual multimodal reasoning via self-training<sub>.</sub> In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)<sub>.</sub>

P. Rust, J. Pfeiffer, I. Vulić, S. Ruder, and I. Gurevych. 2021. How good is your tokenizer? on the monolingual performance of multilingual language models<sub>.</sub> In Proceedings ofACL

R. Sennrich, B. Haddow, and A. Birch. 2016. Neural machine translation of <sub>rare</sub> words with subword units<sub>.</sub> In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) pages 1715–1725.

S. I. Sharina and R. P. Kuzmina. 2014. Russian-Even Phrasebook (Русско-эвенский разговорник)<sub>.</sub> IGIiPMNS SO RAN, Yakutsk.

## A Additional Details

Tables 4 5 6 and 7 provide details on dataset composition, tokenizer expansion, and training configurations.

Table 4: Training dataset composition. Total: 107,543 examples.
<table><tr><td>Task</td><td>Count</td><td>Stage</td></tr><tr><td>Translation (word)</td><td>44,391</td><td>1</td></tr><tr><td>Translation (sentence)</td><td>27,365</td><td>3</td></tr><tr><td>Translation (phrase)</td><td>13,363</td><td>1</td></tr><tr><td>Grammar (declension)</td><td>8,022</td><td>2</td></tr><tr><td>Grammar (negation)</td><td>4,526</td><td>2</td></tr><tr><td>Grammar (plural/conjugation)</td><td>3,876</td><td>2</td></tr><tr><td>Reading comprehension</td><td>3,000</td><td>3</td></tr><tr><td>Classification</td><td>3,000</td><td>3</td></tr><tr><td>Other</td><td>2,198</td><td>1-3</td></tr></table>

Table 5: Tokenizer expansion results. Fragmentation reduced by 27%.
<table><tr><td>Metric</td><td>Before</td><td>After</td></tr><tr><td>Vocabulary size</td><td>151,669</td><td>151,972</td></tr><tr><td>Avg. tokens/word</td><td>4.99</td><td>3.64</td></tr><tr><td>Median tokens/word</td><td>5</td><td>4</td></tr><tr><td>Words ≤ 2 tokens</td><td>7.6%</td><td>15.4%</td></tr></table>

Table 6: Stage 1 (text) training hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Effective batch size</td><td> $3 2 ( 1 6 \times 2$  grad. accum.)</td></tr><tr><td>Epochs</td><td>6  $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Learning rate LR scheduler</td><td>Cosine (3% warmup)</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Max sequence length</td><td>512</td></tr><tr><td>Mixed precision</td><td></td></tr><tr><td></td><td>fp16</td></tr><tr><td>Gradient checkpointing</td><td>Yes</td></tr><tr><td>Save/eval frequency</td><td>Every 2,000 steps</td></tr></table>

Table 7: Stage 2 (VQA) training hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>LoRA rank r</td><td>64</td></tr><tr><td>LoRA α</td><td>128</td></tr><tr><td>LoRA dropout</td><td>0.05</td></tr><tr><td>Target modules up, down projections</td><td>q, k, v, 0, gate,</td></tr><tr><td>Effective batch size</td><td>8</td></tr><tr><td>Epochs</td><td>3</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Vision encoder</td><td>Frozen + torch.no_grad()</td></tr><tr><td>Max pixels</td><td> $2 6 2 , 1 4 4 \left( \sim 5 1 2 \times 5 1 2 \right)$ </td></tr><tr><td>Max sequence length</td><td>2,048</td></tr><tr><td>VQA train examples</td><td>1,908</td></tr></table>