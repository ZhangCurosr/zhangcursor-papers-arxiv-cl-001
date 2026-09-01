# Quantifying and Mitigating Korean Jamo-Level Typographical Vulnerabilities in Large Language Models

Seojin Lee Hwanhee Lee Chung-Ang University, Seoul, Korea {seojin0311,hwanheelee}@cau.ac.kr https://github.com/SJLee0311/korean-jamo-typo

## Abstract

Korean introduces an additional typographical perturbation level not captured by ordinary character-level edit models: because syllable blocks are internally composed of subcharacter units called jamo, keyboard-level errors can occur within a syllable, either producing a valid but semantically altered character or exposing raw jamo on the surface. Both outcomes disrupt sub-word tokenization and are not reliably corrected by existing grammatical error correction pipelines, leaving LLMs directly exposed to corrupted inputs. To quantify this vulnerability, we apply five jamo-level perturbation types to the KMMLU benchmark and evaluate four language models, finding that accuracy declines monotonically with perturbation intensity and that parameter scaling does not confer robustness against intra-syllabic noise. We further show that typo-corrupted inputs induce a distinct shift in internal representations that is not reducible to ordinary answer incorrectness, and that a simple linear probe trained on these representations detects unseen perturbation types with high AUROC. Motivated by this signal, we propose Typo-Aware Chain-of-Thought (TACoT), which routes inputs to chain-of-thought inference only when the probe detects a likely typo, recovering a substantial portion of the CoT accuracy gain at a fraction of the inference cost.

## 1 Introduction

Large Language Models (LLMs) have achieved remarkable success across diverse natural language processing tasks, leading to their widespread deployment. However, their robustness against input noise remains a critical bottleneck. Among various user-generated noises, typographical errors are ubiquitous. Characterizing and mitigating LLM brittleness under typographical perturbations has thus emerged as an essential line of trustworthy AI research (Gao et al., 2018; Zhu et al., 2024a,b). Nevertheless, existing robustness benchmarks are predominantly English-centric, modeling typos merely as surface-character edits (Gao et al., 2018; Zhu et al., 2024a,b; Gan et al., 2024), a paradigm that fails to capture intra-syllabic perturbations unique to scripts like Korean.

<table><tr><td>米</td><td>Clean Typo form → from</td><td>Visible character-level edit</td></tr><tr><td></td><td>Clean Typo 在 →再</td><td>Character confusion</td></tr><tr><td rowspan="2">IⅢI B H 11</td><td>Clean Typo 召 → </td><td>Valid-form ambiguity</td></tr><tr><td>Clean Typo 号 → 号ス</td><td>Exposed standalone jamo</td></tr></table>

Figure 1: Typographical perturbations across writing systems. In English and Chinese, errors occur at the visible character level, whereas Korean errors occur within syllable blocks and surface as one of two failure modes: valid-form ambiguity or exposed standalone jamo that break the syllabic structure.

While classical spelling correction targets linear edit distances (Kukich, 1992; Brill and Moore, 2000) and Chinese systems focus on phonologically or visually similar character substitutions (Cheng et al., 2020; Sun et al., 2021), the Korean writing system introduces an additional perturbation level beyond visible character edits. Korean is a structured alphabetic syllabary where individual letters called jamo are assembled into syllable blocks. Because keyboard input is typically produced through individual jamo keystrokes, realistic Korean typos occur at the intra-syllabic level rather than the surface syllable level (Kwon et al., 2021).

These intra-syllabic perturbations pose a critical challenge: the lack of an effective correction prior. As shown in Figure 1, a jamo-level perturbation may convert a syllable into a contextually incorrect valid character (causing semantic ambiguity) or expose standalone jamo on the surface, breaking the syllabic structure. Both outcomes disrupt sub-word tokenizers optimized for well-formed syllabic blocks. Furthermore, grammatical error correction pipelines are not designed for intra-syllabic perturbations and may inadvertently alter sentence semantics (Kim et al., 2024). Consequently, LLMs must directly process these corrupted representations, rendering them vulnerable.

To systematically understand and combat this vulnerability, we investigate the impact of Korean typographical errors on LLMs by addressing two research questions: 1) How vulnerable are state-ofthe-art LLMs to realistic jamo-level perturbations? 2) Do LLMs implicitly recognize such structural noise within their internal representations, and can we leverage these signals for efficient defense?

To answer these questions, we construct a Korean typo benchmark applying five keyboardderived perturbation types to the KMMLU evaluation suite (Son et al., 2025). Across four language models, accuracy declines monotonically with perturbation intensity, confirming that parameter scaling fails to alleviate intra-syllabic brittleness. Moreover, our probing analysis reveals that typo-corrupted inputs induce a distinct representational shift that is not reducible to ordinary answer incorrectness, and that a simple linear probe detects this signal across held-out error types with high AUROC. Motivated by this signal, we propose Typo-Aware Chain-of-Thought (TACoT), which employs the lightweight internal probe to monitor representations at inference time, invoking CoT only when a typographical anomaly is detected. Our results show that this typo-aware routing substantially recovers CoT accuracy while reducing inference cost on clean inputs.

In summary, our contributions are as follows:

• We formalize a five-type jamo-level perturbation framework based on Korean keyboard mechanics, capturing structural anomalies that lack standard correction priors.

• We establish a controlled benchmark to quantify accuracy drops across diverse LLMs, showing that parameter scaling alone fails to resolve intrasyllabic brittleness.

• We discover a distinct typo signal in LLM internal representations and introduce a Typo-Aware CoT framework that leverages this signal for efficient, low-cost routing defense.

## 2 Related Work

## 2.1 Typographical Robustness in LLMs

Character-level perturbations have long been used to study the brittleness of neural NLP models. DeepWordBug showed that small black-box character edits can substantially degrade text classifiers (Gao et al., 2018). With the rise of LLMs, PromptRobust and PromptBench extended robustness evaluation to prompt perturbations, including typo-like changes (Zhu et al., 2024a,b). More recent work studies typographical errors in LLMspecific settings: Gan et al. (2024) show that adversarial typos can disrupt CoT reasoning trajectories, while Zhao et al. (2026) evaluate LLM robustness against multilingual keyboard-based typographical errors. These studies establish typo robustness as an important failure mode for LLMs, but they primarily model typos as visible character-level perturbations. Our work focuses on Korean, where typographical errors can also arise within the internal composition of syllable blocks.

## 2.2 Korean Grammatical Error Correction

Korean grammatical error correction (GEC) aims to detect and correct erroneous Korean text, including grammatical, spacing, punctuation, and spelling errors. Prior work has developed Korean GEC datasets, annotation schemes, and error taxonomies (Yoon et al., 2023), as well as neural correction models based on Transformer architectures, copying mechanisms, and synthetic grammatical noise (Lee et al., 2021). KoGEC further formulates Korean GEC as a translation-style correction problem by fine-tuning pre-trained multilingual translation models (Kim et al., 2024). These studies provide important resources and models for Korean text correction. However, their primary goal is to restore well-formed text, whereas our work studies how controlled Korean typos affect downstream LLM behavior. In particular, we focus on jamo-level perturbations that alter the internal composition of Korean syllable blocks, quantify their impact on LLM accuracy, and analyze whether they induce detectable hidden-state signals.

![](images/2a3f053288d1e5ebf6497ca8cb15eebaf9fbd180b51d46c4d5b8c80131ba2d6e.jpg)  
Figure 2: Representative outputs from the KoGEC model on typo-perturbed KMMLU questions. The model can repair simple exposed-jamo errors (Repaired), but may leave valid-form typos unchanged (Left unchanged) or rewrite heavily corrupted inputs into fluent yet semantically altered text (Lossy rewrite).

## 3 Korean Typographical Errors

## 3.1 Intra-Syllabic vs. Character-Level Errors

Conventional spelling correction models typos as visible character edits under edit distance assumptions (Kukich, 1992; Brill and Moore, 2000), and Chinese spelling correction targets phonologically or visually similar character substitutions using pinyin, glyph, or radical information (Cheng et al., 2020; Sun et al., 2021). In both cases, the error unit aligns with a visible character, giving correction systems a stable target for recovery.

Korean shares this character-level typo space, but also introduces an intra-syllabic perturbation level. Since Korean syllable blocks are composed ofjamo (onset, nucleus, and optional coda) and keyboard input is produced through individual jamo keystrokes, typographical errors can occur within a syllable block. A jamo-level perturbation may yield another well-formed Korean syllable (e.g., <sup>가</sup>ᆷ → <sup>ᄀ</sup>ᅡ), producing valid-form ambiguity, or may break the syllable block and expose standalone jamo on the surface (e.g., 문<sup>ᄌ</sup>ᅦ → 문ㅈ<sup>ᄌ</sup>ᅦ), producing a structurally degraded sequence with no stable character-level recovery unit (Kwon et al., 2021). Figure 1 illustrates this distinction; the resulting orthographic disruption is invisible to standard character-level correction pipelines and to tokenizers optimized for well-formed syllabic blocks.

## 3.2 Limitations of Korean GEC

Whether typo-corrupted inputs can be automatically corrected prior to LLM inference remains an open question. However, our typographical setting fundamentally differs from standard GEC. While conventional GEC systems are designed to rectify grammaticality, spacing, punctuation, and surface spelling, our perturbations specifically target the internal jamo composition of Hangul syllables. Consequently, valid-form perturbations yield another well-formed Korean surface, making the error difficult to flag as ungrammatical. Conversely, exposedjamo sequences break the syllable structure altogether, producing degraded strings that do not correspond cleanly to the word- or syllable-level edits assumed by standard correction systems.

To analyze this mismatch, we evaluate Ko-GEC (Kim et al., 2024) on typo-perturbed KMMLU questions in Figure 2. While KoGEC repairs simple exposed-jamo errors, it leaves validform perturbations unchanged or rewrites heavily corrupted inputs into semantically altered text. For instance, severe corruptions lead to the loss of the semantic subject, shifting the question’s core meaning. Consequently, Korean jamo typos are not reliably reducible to existing GEC pipelines, demonstrating that pre-correction does not provide a stable solution to this robustness bottleneck.

## 4 Korean Typographical Perturbation Framework

## 4.1 Evaluation Benchmark

We use KMMLU (Son et al., 2025) as the main evaluation benchmark. KMMLU consists of 35,030 expert-level four-choice questions collected from original Korean examinations. We apply typo perturbations only to the question; answer choices are kept unchanged to ensure that accuracy loss reflects input comprehension failure rather than output matching artifacts. Further dataset details are provided in Appendix A.

## 4.2 Typo Taxonomy

We define five typo types grounded in realistic keyboard input errors on a standard Korean layout, each targeting a different aspect of the jamo– syllable structure. Table 1 summarizes the types with examples derived from the sentence “<sup>가</sup>ᆷ<sup>ᄀ</sup>ᅵ <sup>ᄀ</sup>ᅡ <sup>ᄃ</sup>ᅡ <sup>ᄂ</sup>ᅡᆺ<sup>ᄃ</sup>ᅡ<sup>ᄂ</sup>ᅵ <sup>ᄃ</sup>ᅡᆼ<sup>해ᄋ</sup>ᅵ<sup>ᄋ</sup>ᅣ” (“It’s great that you’ve recovered from your cold.”).

<table><tr><td>Typo Type</td><td>Perturbed</td></tr><tr><td>Jamo Substitution</td><td></td></tr><tr><td>Jongseong Deletion</td><td></td></tr><tr><td>Jamo Repetition</td><td></td></tr><tr><td>Space Deletion Jamo Transposition</td><td></td></tr><tr><td></td><td></td></tr><tr><td colspan="2">Clean:</td></tr></table>

Table 1: Korean typographical error taxonomy. Perturbed characters are shown in red. Original sentence shown at the bottom.

Jamo Substitution. A jamo is replaced by one from an adjacent key, frequently yielding another valid Korean syllable and making error detection ambiguous (e.g., <sup>가</sup>ᆷ → ᄀ<sup>ᅥ</sup>ᆷ).

Jongseong Deletion. The jongseong of a syllable is dropped (e.g., <sup>가</sup>ᆹ → <sup>ᄀ</sup>ᅡ), always yielding a wellformed Korean syllable and thus being particularly difficult to detect.

Jamo Repetition. A jamo component is duplicated, surfacing as a raw jamo adjacent to the host syllable. Such sequences fall outside standard syllable blocks and are invisible to word-level spell checkers.

Space Deletion. Inter-eojeol spaces are removed, concatenating adjacent spacing units (eojeol are Korean orthographic spacing units, roughly analogous to words). Unlike the other types, this does not alter the internal jamo composition of syllables.

Jamo Transposition. The order of jamo within a syllable is swapped (e.g., <sup>가</sup>ᆷ → ㄱㅁㅏ), dissolving the onset–nucleus–coda structure and exposing raw jamo poorly aligned with standard vocabularies.

This taxonomy is not an arbitrary construction: it covers the large majority of real Korean input errors observed in practice, as we verify in Appendix C.

## 4.3 Perturbation Intensity

For each typo type, we define five intensity levels corresponding to perturbation rates of 5% through

25% of the Korean syllables in the question field. Each level is applied independently from the original clean input, so types are never mixed within a single instance.<sup>1</sup> Within each level, the affected syllables are chosen at random. We confirm in Appendix E that this randomness does not affect our conclusions. This yields 35,030×5×5 = 875,750 perturbed instances in total.

## 5 Impact of Korean Typographical Errors

## 5.1 Experimental Setup

Models. We evaluate four models spanning a range of sizes and Korean language orientations. EXAONE-3.5-2.4B-Instruct and EXAONE-3.5-7.8B-Instruct are bilingual (Korean–English) instruction-tuned models developed by LG AI Research, designed for real-world instruction following with strong Korean proficiency (An et al., 2026). A.X-3.1-Light is a Korean sovereign AI model developed entirely from scratch by SK Telecom, trained on a Korean-focused multilingual corpus.<sup>2</sup> Qwen3-4B-Instruct-2507 is a general-purpose multilingual model by Alibaba; we use the 2507 checkpoint, which features significant improvements in instruction following and multilingual comprehension (Yang et al., 2025). This model serves as a non-Korean-specialized baseline to probe whether typo robustness is tied to Korean-specific training.

Implementation Details. All models are evaluated on KMMLU in a zero-shot setting. We use greedy decoding with temperature 0 and max\_new\_tokens set to 8; unless otherwise noted, all experiments follow this configuration. Full prompt and inference details are provided in Appendix D.

## 5.2 Results

Figure 3 shows accuracy across five typo intensity levels for all four open-source models. Across all four models, accuracy declines monotonically as intensity increases, confirming that Korean typographical errors consistently degrade performance regardless of model size or Korean specialization. Jamo-level perturbations induce the largest drops: at the highest intensity, accuracy falls by up to 10.0%p for EXAONE-7.8B and 7.0%p for Qwen3- 4B relative to their clean baselines. Notably, model size does not correlate with robustness: EXAONE-7.8B exhibits a larger drop than its 2.4B counterpart under Jamo Transposition, suggesting that scale alone does not confer immunity to intra-syllabic perturbations.

![](images/a6ace171bd816e88c4220989788bfd70c36b26f1b50424797102ca469a2129d7.jpg)

![](images/00147b0edb06d7fe037d877bc1a388b744bd9cfce56426ccc6b85d199c6e87fc.jpg)

![](images/bb9ce93340411f17ad7f8b1e0156b16f051edd920c7893ec5fdbe82b51f72409.jpg)

![](images/7ea06499f13873e2a11e0be226560830c940d8fec8dd53168bf55d5b3f34d18f.jpg)  
Figure 3: KMMLU accuracy across five typo intensity levels for all four models. Dashed lines indicate clean baseline accuracy.

Space Deletion stands out as a consistent exception: across all models, accuracy under space deletion remains close to the clean baseline even at the highest intensity. While perceptually disruptive to human readers, Korean subword tokenizers appear to partially recover eojeol boundaries during segmentation, effectively absorbing the perturbation before it reaches the model’s internal representations. This divergence between human perception and model behavior highlights that the impact of a typo type cannot be inferred from surface-level intuition alone.

These results establish that even well-performing LLMs are meaningfully vulnerable to naturally occurring Korean typographical errors. Critically, this vulnerability persists regardless of model scale or Korean specialization, suggesting that the problem is structural rather than a matter of insufficient training. We therefore turn to the models’ internal representations to ask whether this structural disruption leaves a detectable signal. We further evaluate larger-scale models in Appendix E.

## 6 Probing Internal Representations of Korean Typos

The previous section shows that Korean typographical errors consistently degrade LLM accuracy. We now examine whether this degradation is reflected in the models’ internal representations. If typos merely push the model toward an ordinary incorrect-answer state, then hidden states under typo inputs should largely align with clean inputs that the model already answers incorrectly. In contrast, if typo inputs induce a distinct internal state, we should observe a separable hidden-state shift that is not reducible to answer correctness. If such a signal exists and is separable from ordinary answer failure, it could serve as a lightweight, training-free basis for targeted mitigation.

## 6.1 Locating Typo-Sensitive Layers

We first identify where typo-induced changes are most visible inside the model. For each model, we extract the last-token hidden state from every transformer layer on the General Knowledge subset of HAERAE Bench (HAERAE-GK), using clean inputs and their typo-perturbed counterparts (see Appendix A for dataset details). To select a typosensitive layer without training a classifier, we compute a Fisher-style separation score (Fisher, 1936) for each layer:

$$
J ( l ) = \frac { \| \mu _ { \mathrm { t y p o } } ^ { ( l ) } - \mu _ { \mathrm { c l e a n } } ^ { ( l ) } \| _ { 2 } ^ { 2 } } { \mathrm { t r } ( \Sigma _ { \mathrm { c l e a n } } ^ { ( l ) } ) + \mathrm { t r } ( \Sigma _ { \mathrm { t y p o } } ^ { ( l ) } ) } .\tag{1}
$$

where $\mu _ { \mathrm { c l e a n } } ^ { ( l ) }$ and $\mu _ { \mathrm { t y p o } } ^ { ( l ) }$ are the mean hidden states of clean and typo inputs at layer l, and the denominator measures within-class dispersion. Intuitively, this criterion favors layers where clean and typo representations are far apart while remaining internally compact. We use the Fisher-selected layer for each model in the remaining hidden-state analyses; layer-wise score curves and selected layer indices are provided in Appendix F.

## 6.2 Typo Shift Is Distinct from Answer Incorrectness

A typo input is typically both corrupted and answered incorrectly, so a possible concern is that the representational shift we observe may simply reflect answer failure: typo inputs may look different only because they cause the model to move toward the same representation as clean questions it answers incorrectly. To test this, we compare two directions in the representation space at the Fisher-selected layer. Let $\mu _ { \mathrm { c o r r e c t } }$ and $\mu _ { \mathrm { w r o n g } }$ denote the mean hidden states of clean inputs the model answers correctly and incorrectly; we define the clean incorrectness direction as $d _ { \mathrm { w r o n g } } =$ $\mu _ { \mathrm { w r o n g } } - \mu _ { \mathrm { c o r r e c t } }$ and the typo direction as $d _ { \mathrm { t y p o } } =$ $\mu _ { \mathrm { t y p o } } - \mu _ { \mathrm { c o r r e c t } }$ . For visualization, we project hidden states onto a two-dimensional basis whose first axis is $d _ { \mathrm { w r o n g } }$ and whose second axis is the component of $d _ { \mathrm { t y p o } }$ orthogonal to $d _ { \mathrm { w r o n g } }$

Figure 4 shows the resulting projections for clean-correct, clean-wrong, typo-correct, and typowrong examples. Across models, typo examples shift along the typo direction regardless of whether the final answer is correct or wrong. This indicates that typo inputs do not merely reproduce the hidden-state pattern of ordinary incorrect answers. Instead, they induce a distinct representational displacement that coexists with, but is not reducible to, answer correctness.

## 6.3 Hidden-State Typo Detection

The directional analysis suggests that Korean typo inputs leave a detectable representational signal.

![](images/961e2f6a05aed00c07f15a182a5f523ab1ca6741616ebc38c87073b4588b2b8a.jpg)

![](images/f21c640915e673667db5eee80d166c19ee833f036ecf0ca837b54b7534564cd4.jpg)  
Figure 4: Direction projection of hidden states at the Fisher-selected layer. Full projections for all models are provided in Appendix F.

We next test whether this signal can be used to identify typo inputs with a simple linear probe. For each model, we train a logistic regression classifier on the last-token hidden state at the Fisher-selected layer, using clean inputs and typo-perturbed inputs as binary labels, focusing on the four jamo-level typo types.<sup>3</sup>

To test whether the probe captures a general typo signal rather than memorizing specific perturbation patterns, we use a held-out typo type evaluation. For each held-out jamo-level type, the classifier is trained on the remaining three types from HAERAE-GK and evaluated on the excluded type in KMMLU. Within each fold, we balance the training set by randomly sampling typo examples to match the number of clean inputs, drawing only from the non-held-out typo types. Thus, the test set differs both in questions and in typo type, and the held-out type is never used for probe training.

<table><tr><td>Model</td><td>Sub.</td><td>Jong.</td><td>Rep.</td><td>Trans.</td><td>Avg.</td></tr><tr><td>EXAONE-2.4B</td><td>0.912</td><td>0.885</td><td>0.927</td><td>0.971</td><td>0.924</td></tr><tr><td>EXAONE-7.8B</td><td>0.929</td><td>0.910</td><td>0.933</td><td>0.974</td><td>0.937</td></tr><tr><td>A.X-Light</td><td>0.928</td><td>0.922</td><td>0.936</td><td>0.987</td><td>0.943</td></tr><tr><td>Qwen3-4B</td><td>0.849</td><td>0.843</td><td>0.939</td><td>0.990</td><td>0.905</td></tr></table>

Table 2: Held-out typo type detection AUROC using a logistic regression probe on Fisher-selected hidden states, pooled over all five perturbation intensities. Perintensity results are reported in Table 9.

Table 2 reports AUROC on the held-out type. The probe achieves strong detection performance across all models, with mean AUROC ranging from 0.905 to 0.943. Jamo Transposition is consistently easiest to detect, while Jongseong Deletion is the hardest, mirroring their relative Fisher separation scores. These results show that typo-induced hidden-state shifts are not merely type-specific artifacts, but generalize to unseen jamo-level perturbation patterns.

## 7 Mitigating Korean Typos via Internal Representations

The previous section shows that Korean typos induce a detectable representational signal, especially for jamo-level perturbations that alter the internal composition of Korean syllables. We now use this signal for mitigation. A simple way to improve robustness is to ask the model to reason more carefully under typo-corrupted inputs. However, applying chain-of-thought (CoT) prompting (Wei et al., 2022) to every input substantially increases output length and inference cost, including for clean or weakly affected inputs. We therefore introduce TACoT (Typo-Aware Chain-of-Thought), a probeguided inference strategy that applies standard decoding by default and invokes CoT only when the hidden-state probe signals a likely typo.

## 7.1 TACoT: Typo-Aware CoT

Our typo-aware method uses the Fisher-selected layer identified in Section 6. For each model, we train a logistic regression probe on the last-token hidden state from HAERAE-GK, using clean inputs and their typo-perturbed counterparts. For the typo class, we draw an equal number of examples from the pool of jamo-level perturbations, covering four typo types and five perturbation intensities, to prevent the probe from being dominated by the larger typo pool. The routing threshold is selected by maximizing Youden’s J statistic (Youden, 1950) on a held-out validation split; full training details are provided in Appendix G.

At inference time, each KMMLU input is first passed through the model to extract the last-token hidden state at the Fisher-selected layer. The probe then estimates P(typo). If this probability exceeds the learned threshold, we route the instance to CoT inference; otherwise, we use standard inference. Formally, the routing rule is

$$
\mathrm { r o u t e } ( x ) = \left\{ \begin{array} { l l } { \mathrm { C o T } , } & { P ( \mathrm { t y p o } \mid h _ { l } ( x ) ) \geq \theta , } \\ { \mathrm { S t a n d a r d } , } & { P ( \mathrm { t y p o } \mid h _ { l } ( x ) ) < \theta , } \end{array} \right.
$$

where $h _ { l } ( x )$ is the hidden state at the Fisherselected layer and θ is the selected threshold.

## 7.2 Baselines

We compare TACoT against four inference strategies. Standard uses the default multiple-choice prompt and generates at most 8 tokens. Meta-Cognition uses a system prompt that explicitly warns the model that the question may contain Korean typos, but still requires a direct single-letter answer. GEC first applies KoGEC (Kim et al., 2024) correction to the input and then uses the standard prompt. Although KoGEC can repair simple exposed-jamo errors, it is not designed for intrasyllabic jamo perturbations and may leave validform typos unchanged or introduce semantic alterations (Section 3.2); we nonetheless include it as a relevant correction-oriented baseline. CoT applies the CoT prompt to every input with a maximum generation length of 1024 tokens. The exact system prompts for Standard, Meta-Cognition, and CoT inference are shown in Appendix D.

## 7.3 Results

Table 3 compares the mitigation methods in terms of accuracy, and Figure 5 summarizes the average output token cost of CoT and TACoT across models. We report accuracy on clean inputs (Base) and on each typo type, where typo accuracies are averaged over five perturbation intensity levels.

Overall, GEC-based pre-correction and Meta-Cognition prompting provide little to no reliable improvement over Standard inference. In several cases, they even reduce accuracy. For example,

<table><tr><td>Model</td><td>Method</td><td>Base</td><td>Sub.</td><td>Jong.</td><td>Rep.</td><td>Space.</td><td>Trans.</td></tr><tr><td rowspan="5">EXAONE-2.4B</td><td>Standard</td><td>41.9</td><td>39.1</td><td>38.9</td><td>40.6</td><td>41.7</td><td>38.4</td></tr><tr><td>Meta-Cognition</td><td> $4 2 . 1 + 0 . 2 $ </td><td> $3 9 . 2 \substack { + 0 . 1 }$ </td><td>38.9</td><td> $4 0 . 7 + 0 . 1 $ </td><td> $4 1 . 9 \substack { + 0 . 2 }$ </td><td> $3 8 . 6 \substack { + 0 . 2 }$ </td></tr><tr><td>GEC</td><td>41.9</td><td> $3 8 . 5 \substack { - 0 . 6 }$ </td><td> $3 8 . 7 _ { - 0 . 2 }$ </td><td> $4 0 . 7 + 0 . 1 $ </td><td> $4 1 . 5 _ { - 0 . 2 }$ </td><td> $3 7 . 7 _ { - 0 . 7 }$ </td></tr><tr><td>CoT</td><td> $4 5 . 2 \substack { + 3 . 3 }$ </td><td> $4 1 . 9 + 2 . 8 $ </td><td> $4 1 . 5 \substack { + 2 . 6 }$ </td><td> $4 3 . 9 + 3 . 3 $ </td><td> $4 4 . 9 \substack { + 3 . 2 }$ </td><td> $4 1 . 1 + 2 . 7$ </td></tr><tr><td>TACoT</td><td> $4 2 . 6 \substack { + 0 . 7 }$ </td><td> $4 1 . 3 \substack { + 2 . 2 }$ </td><td> $4 0 . 6 \substack { + 1 . 7 }$ </td><td> $4 3 . 5 \substack { + 2 . 9 }$ </td><td> $4 3 . 1 + 1 . 4 $ </td><td> $4 1 . 0 \substack { + 2 . 6 }$ </td></tr><tr><td rowspan="5">EXAONE-7.8B</td><td>Standard</td><td>47.2</td><td>43.7</td><td>43.3</td><td>45.3</td><td>46.9</td><td>41.4</td></tr><tr><td>Meta-Cognition</td><td> $4 7 . 4 \substack { + 0 . 2 }$ </td><td> $4 3 . 5 _ { - 0 . 2 }$ </td><td> $4 3 . 1 \AA _ { - 0 . 2 }$ </td><td> $4 4 . 9 \substack { - 0 . 4 }$ </td><td> $4 7 . 1 + 0 . 2 $ </td><td> $4 0 . 9 _ { - 0 . 5 }$ </td></tr><tr><td>GEC</td><td>47.2</td><td> $4 2 . 7 \substack { - 1 . 0 }$ </td><td> $4 2 . 5 \AA _ { - 0 . 8 }$ </td><td> $4 5 . 6 \substack { + 0 . 3 }$ </td><td> $4 6 . 9$ </td><td> $4 1 . 9 \substack { + 0 . 5 }$ </td></tr><tr><td>CoT</td><td> $4 9 . 7 + 2 . 5 $ </td><td> $4 6 . 9 + 3 . 2 $ </td><td> $4 6 . 2 + 2 . 9$ </td><td> $4 8 . 6 \substack { + 3 . 3 }$ </td><td> $4 9 . 4 + 2 . 5 $ </td><td> $4 4 . 8 \substack { + 3 . 4 }$ </td></tr><tr><td>TACoT</td><td>47.2</td><td> $4 6 . 3 \substack { + 2 . 6 }$ </td><td> $4 5 . 5 \substack { + 2 . 2 }$ </td><td> $4 8 . 3 \substack { + 3 . 0 }$ </td><td> $4 7 . 5 \substack { + 0 . 6 }$ </td><td> $4 4 . 7 + 3 . 3 $ </td></tr><tr><td rowspan="5">A.X-Light</td><td>Standard</td><td>49.5</td><td>46.2</td><td>45.6</td><td>48.5</td><td>49.4</td><td>46.2</td></tr><tr><td>Meta-Cognition</td><td> $5 0 . 8 \substack { + 1 . 3 }$ </td><td> $4 6 . 9 \substack { + 0 . 7 }$ </td><td> $4 6 . 0 \substack { + 0 . 4 }$ </td><td> $4 9 . 3 \substack { + 0 . 8 }$ </td><td> $5 0 . 6 \substack { + 1 . 2 }$ </td><td> $4 7 . 0 \substack { + 0 . 8 }$ </td></tr><tr><td>GEC</td><td>49.5</td><td> $4 5 . 0 { - } 1 . 2 $ </td><td> $4 5 . 0 _ { - 0 . 6 }$ </td><td> $4 7 . 9 _ { - 0 . 6 }$ </td><td> $4 9 . 1 _ { - 0 . 3 }$ </td><td> $4 4 . 0 \AA 2 . 2$ </td></tr><tr><td>CoT</td><td> $5 7 . 1 + 7 . 6 $ </td><td> $5 2 . 5 + 6 . 3 $ </td><td> $5 1 . 7 \substack { + 6 . 1 }$ </td><td> $5 4 . 9 + 6 . 4 $ </td><td> $5 6 . 7 + 7 . 3 $ </td><td> $5 2 . 1 + 5 . 9$ </td></tr><tr><td>TACoT</td><td> $4 9 . 6 \substack { + 0 . 1 }$ </td><td> $4 9 . 8 + 3 . 6 $ </td><td> $4 8 . 9 \substack { + 3 . 3 }$ </td><td> $5 2 . 8 \substack { + 4 . 3 }$ </td><td> $4 9 . 7 \substack { + 0 . 3 }$ </td><td> $5 1 . 0 { + } 4 . 8 $ </td></tr><tr><td rowspan="5">Qwen3-4B</td><td>Standard</td><td>50.5</td><td>46.8</td><td>46.3</td><td>49.2</td><td>50.4</td><td>46.4</td></tr><tr><td>Meta-Cognition</td><td> $5 0 . 0 { \cdot } 0 . 5$ </td><td> $4 5 . 8 \AA \mathrm { - } 1 . 0$ </td><td> $4 5 . 5 \substack { - 0 . 8 }$ </td><td> $4 8 . 4 \AA { - } 0 . 8$ </td><td> $4 9 . 9 _ { - 0 . 5 }$ </td><td> $4 5 . 3 \AA _ { - 1 . 1 }$ </td></tr><tr><td>GEC</td><td>50.4-0.1</td><td> $4 6 . 3 \AA { - 0 . 5 }$ </td><td> $4 6 . 5 \substack { + 0 . 2 }$ </td><td> $4 8 . 9 _ { - 0 . 3 }$ </td><td> $5 0 . 0 _ { - 0 . 4 }$ </td><td> $4 5 . 7 \AA _ { - 0 . 7 }$ </td></tr><tr><td>CoT</td><td> $5 3 . 7 + 3 . 2 $ </td><td> $4 7 . 9 + 1 . 1$ </td><td> $4 7 . 5 \substack { + 1 . 2 }$ </td><td> $5 1 . 2 \substack { + 2 . 0 }$ </td><td> $5 3 . 3 \substack { + 2 . 9 }$ </td><td> $4 7 . 5 \substack { + 1 . 1 }$ </td></tr><tr><td>TACoT</td><td> $5 0 . 6 \substack { + 0 . 1 }$ </td><td> $4 7 . 1 + 0 . 3 $ </td><td> $4 6 . 5 \substack { + 0 . 2 }$ </td><td> $5 0 . 9 { + } 1 . 7 $ </td><td> $5 0 . 5 \substack { + 0 . 1 }$ </td><td> $4 7 . 3 + 0 . 9 $ </td></tr></table>

Table 3: Comparison of mitigation strategies on KMMLU (accuracy in %). Each accuracy column reports the mean over five perturbation intensity levels; colored annotations indicate gain/loss relative to Standard (blue: improvement, red: degradation). Sub.: Jamo Substitution; Jong.: Jongseong Deletion; Rep.: Jamo Repetition; Space.: Space Deletion; Trans.: Jamo Transposition.

![](images/4d26ec0c6d2422469e5e4ae7adae3e9357b1ff1464bc0caca89fd45924ee7b6b.jpg)  
Figure 5: Average output tokens for CoT and TACoT across models. TACoT consistently reduces token cost while preserving most of the accuracy gains of CoT.

By invoking CoT only when the hidden-state probe detects a likely typo, TACoT preserves most of the accuracy gains of CoT across all models and typo types. As shown in Figure 5, this selective routing reduces average output length by approximately 37% compared to CoT, ranging from 26% for EXAONE-2.4B to 49% for A.X-Light.

GEC lowers performance on Jamo Substitution for all four models, and Meta-Cognition prompting degrades Qwen3-4B across all typo types. This suggests that Korean jamo typos are not reliably handled either by applying a general-purpose correction model before inference or by simply warning the model that typos may be present.

The remaining gap between TACoT and CoT is informative. TACoT is strongest on jamo-level perturbations such as Jamo Repetition and Jamo Transposition, which also produce clear representational separation in Section 6.3. In contrast, Space Deletion remains difficult for the routing mechanism: although CoT can still improve Space Deletion accuracy, this perturbation induces weak representational separation and is therefore less likely to trigger CoT routing. These results show that the representational typo signal is not only diagnostic but also practically useful for routing, enabling cost-aware mitigation without sacrificing most of the accuracy gains that full CoT provides.

CoT is the most consistently effective mitigation strategy. Across models, it improves accuracy not only on typo-corrupted inputs but also on clean inputs. The gains are especially large for A.X-Light, where CoT improves the clean baseline from 49.5% to 57.1% and raises all typo-type accuracies by more than 5.9 percentage points.

TACoT provides a more efficient compromise.

## 7.4 Robustness of TACoT

We examine the robustness of TACoT along two axes: whether it generalizes to a different domain and task format, and whether its gain reflects the value of the probe’s routing choices rather than simply invoking CoT more often.

To test whether our findings hold beyond closed-form multiple-choice QA, we evaluate on HRM8K (Ko et al., 2025), a Korean mathematical reasoning benchmark requiring free-form solutions rather than a single-letter answer. See Appendix A for dataset details. We apply the same five jamolevel perturbations to the question text.

<table><tr><td>Model</td><td>Method</td><td>Base</td><td>Sub.</td><td>Jong.</td><td>Rep.</td><td>Space.</td><td>Trans.</td></tr><tr><td rowspan="4">EXAONE-2.4B</td><td>Standard</td><td>31.5</td><td>25.6</td><td>24.9</td><td>29.0</td><td>30.7</td><td>24.2</td></tr><tr><td>Meta-Cognition</td><td> $3 0 . 2 \AA { - 1 . 3 }$ </td><td> $2 5 . 2 \AA . 0 . 4 $ </td><td> $2 5 . 0 { \scriptstyle + 0 . 1 }$ </td><td> $2 8 . 4 \AA { - } 0 . 6$ </td><td> $3 0 . 4 \ – 0 . 3$ </td><td> $2 3 . 4 . 0 . 8$ </td></tr><tr><td>CoT</td><td> $3 6 . 5 + 5 . 0$ </td><td> $3 0 . 2 \substack { + 4 . 6 }$ </td><td> $2 9 . 3 + 4 . 4$ </td><td> $3 3 . 7 + 4 . 7 $ </td><td> $3 6 . 0 + 5 . 3$ </td><td> $2 8 . 5 + 4 . 3 $ </td></tr><tr><td>TACoT</td><td> $3 1 . 9 \substack { + 0 . 4 }$ </td><td> $2 9 . 4 + 3 . 8 $ </td><td> $2 8 . 2 \substack { + 3 . 3 }$ </td><td> $3 3 . 2 \substack { + 4 . 2 }$ </td><td> $3 3 . 0 \substack { + 2 . 3 }$ </td><td> $2 8 . 3 + 4 . 1$ </td></tr><tr><td rowspan="4">EXAONE-7.8B</td><td>Standard</td><td>37.7</td><td>32.5</td><td>31.8</td><td>35.8</td><td>37.2</td><td>30.6</td></tr><tr><td>Meta-Cognition</td><td> $3 6 . 9 _ { - 0 . 8 }$ </td><td> $3 2 . 3 _ { - 0 . 2 }$ </td><td>31.8</td><td> $3 5 . 1 _ { - 0 . 7 }$ </td><td> $3 7 . 0 _ { - 0 . 2 }$ </td><td> $2 9 . 6 - 1 . 0$ </td></tr><tr><td>CoT</td><td> $4 5 . 4 + 7 . 7$ </td><td> $3 9 . 1 + 6 . 6 $ </td><td> $3 8 . 3 \substack { + 6 . 5 }$ </td><td> $4 3 . 0 \substack { + 7 . 2 }$ </td><td> $4 4 . 6 \substack { + 7 . 4 }$ </td><td> $3 6 . 8 + 6 . 2 $ </td></tr><tr><td>TACoT</td><td>37.7</td><td> $3 7 . 9 + 5 . 4 $ </td><td> $3 6 . 7 + 4 . 9$ </td><td> $4 2 . 0 \substack { + 6 . 2 }$ </td><td> $3 9 . 3 \substack { + 2 . 1 }$ </td><td> $3 6 . 6 + 6 . 0$ </td></tr><tr><td rowspan="4">A.X-Light</td><td>Standard</td><td>40.4</td><td>36.4</td><td>35.7</td><td>39.9</td><td>40.7</td><td>36.6</td></tr><tr><td>Meta-Cognition</td><td> $4 0 . 5 \substack { + 0 . 1 }$ </td><td>36.4</td><td> $3 5 . 9 \substack { + 0 . 2 }$ </td><td> $3 9 . 7 _ { - 0 . 2 }$ </td><td> $4 0 . 4 \ – 0 . 3$ </td><td> $3 6 . 9 + 0 . 3 $ </td></tr><tr><td>CoT</td><td> $4 8 . 0 \substack { + 7 . 6 }$ </td><td> $4 3 . 1 + 6 . 7 $ </td><td> $4 2 . 1 + 6 . 4$ </td><td> $4 7 . 0 \substack { + 7 . 1 }$ </td><td> $4 7 . 8 + 7 . 1$ </td><td> $4 3 . 3 + 6 . 7 $ </td></tr><tr><td>TACoT</td><td> $4 1 . 7 \substack { + 1 . 3 }$ </td><td> $4 2 . 3 + 5 . 9$ </td><td> $4 1 . 1 + 5 . 4 $ </td><td> $4 6 . 1 + 6 . 2 $ </td><td> $4 3 . 9 \substack { + 3 . 2 }$ </td><td> $4 3 . 1 + 6 . 5 $ </td></tr><tr><td rowspan="4">Qwen3-4B</td><td>Standard</td><td>34.0</td><td>26.6</td><td>26.5</td><td>31.7</td><td>34.0</td><td>27.3</td></tr><tr><td>Meta-Cognition</td><td> $1 8 . 0 \substack { - 1 6 . 0 }$ </td><td> $1 1 . 8 \substack { - 1 4 . 8 }$ </td><td> $1 2 . 6 _ { - 1 3 . 9 }$ </td><td> $1 4 . 2 \ – 1 7 . 5$ </td><td> $1 6 . 1 \ – 1 7 . 9$ </td><td> $1 2 . 5 \AA _ { - 1 4 . 8 }$ </td></tr><tr><td>CoT</td><td> $4 2 . 2 \substack { + 8 . 2 }$ </td><td> $3 5 . 4 + 8 . 8 $ </td><td> $3 5 . 1 + 8 . 6 $ </td><td> $4 0 . 8 \substack { + 9 . 1 }$ </td><td> $4 2 . 7 + 8 . 7 $ </td><td> $3 6 . 1 + 8 . 8 $ </td></tr><tr><td>TACoT</td><td> $3 7 . 7 + 3 . 7 $ </td><td> $3 5 . 1 + 8 . 5 $ </td><td> $3 4 . 6 \substack { + 8 . 1 }$ </td><td> $4 0 . 5 \substack { + 8 . 8 }$ </td><td> $3 9 . 8 + 5 . 8 $ </td><td> $3 6 . 0 \substack { + 8 . 7 }$ </td></tr></table>

Table 4: Accuracy (%) on HRM8K, averaged over five intensity levels per typo type. Colored annotations indicate gain/loss relative to Standard (blue: improvement, red: degradation). Sub.: Jamo Substitution; Jong.: Jongseong Deletion; Rep.: Jamo Repetition; Space.: Space Deletion; Trans.: Jamo Transposition. GEC is excluded, as KoGEC degenerates on mathematical inputs, producing empty or repetition-looped outputs even for clean questions.

Table 4 shows that the same trends observed on KMMLU carry over to this different domain and task: typos degrade accuracy relative to Base, CoT is the strongest mitigation, and TACoT recovers most of its gain.

misses.
<table><tr><td>Model</td><td> $l _ { 1 }$ </td><td> $l _ { 2 }$ </td><td> $l _ { 3 }$ </td><td> $l _ { 4 }$ </td><td> $l _ { 5 }$ </td></tr><tr><td>EXAONE-2.4B</td><td>+0.27</td><td>+0.76</td><td>+1.35</td><td>+1.71</td><td>+2.28</td></tr><tr><td>EXAONE-7.8B</td><td>-0.09</td><td>+0.08</td><td> $+ 0 . 6 0$ </td><td> $+ 1 . 1 2$ </td><td>+1.67</td></tr><tr><td>A.X-Light</td><td>0.00</td><td>+0.20</td><td> $+ 0 . 7 0$ </td><td> $+ 1 . 0 6$ </td><td>+1.63</td></tr><tr><td> $\mathrm { Q w e n } 3 – 4 \mathrm { B }$ </td><td>-0.03</td><td>+0.19</td><td> $+ 0 . 5 1$ </td><td>+0.60</td><td>+0.77</td></tr></table>

Table 5: Accuracy gap between TACoT and the random router, in points, on the base-correct pool, averaged over the four jamo-level typo types and five random seeds.

We also examine whether TACoT’s gain comes from the probe’s routing choices, or simply from invoking CoT more often. To isolate this, we compare TACoT against a rate-matched random router: the same number of inputs are routed to CoT per typo condition, chosen at random rather than by the probe. We restrict this comparison to the 7,311 questions all four models answer correctly when clean, so that CoT offers no benefit on clean inputs and any gain must come from recovering typocorrupted ones.

As shown in Table 5, the gap between TACoT and the random router is negligible at the mildest intensity level, where the perturbation is too weak for CoT to recover anything; from there it grows monotonically, reaching up to 2.3 points at the highest level. This is consistent with a valid routing signal: where direct inference fails most and the probe detects corruption most reliably, TACoT sends CoT precisely to the damaged inputs a random router

## 8 Conclusion

This work systemically analyzes the impact of Korean typographical errors on LLMs. Unlike ordinary character-level edits, we show that Korean typos occur within the intra-syllabic jamo composition, consistently degrading model performance regardless of scale or language specialization. We demonstrate that these typos induce a distinct, identifiable shift in internal representations that is separate from ordinary answer incorrectness, allowing a simple linear probe to generalize across unseen error types. Leveraging this signal, we propose TACoT, a routing framework that achieves costaware mitigation by selectively invoking chain-ofthought inference. Overall, our findings establish Korean typo robustness as a crucial linguistic bottleneck and a valuable testbed for examining input corruption within LLM hidden states.

## Limitations

Our study has several limitations. Our perturbations are automatically generated from predefined typo types and intensity levels. Although they are grounded in Korean keyboard input and Korean jamo structure, they may not fully match the distribution of real user typos. Similarly, evaluation resources for Korean LLM robustness remain relatively limited compared with English, and broader coverage across domains, task formats, and naturally occurring typo distributions would strengthen future analyses. On the modeling side, our probing and TACoT experiments rely on access to internal hidden states, which restricts their applicability to locally-served open-weight models; this is fundamentally incompatible with API-based LLM deployment, the most common real-world usage scenario. For this reason, our main probing and mitigation experiments are conducted on four smallto-mid-scale models (2.4B–7.8B), while the larger models in Appendix E are evaluated only for accuracy degradation, without corresponding probe or TACoT validation. Beyond model scale, TACoT is also a routing-based mitigation method: it detects typo-corrupted inputs and improves downstream accuracy by invoking more careful reasoning, but it does not directly correct or realign the typo-induced representational shift itself. Developing methods that explicitly repair such representational distortions remains an important direction for future work.

## Ethics Statement

This work uses publicly available benchmark data and does not involve personal or sensitive information. The typo perturbations are synthetic and are designed to evaluate model robustness under realistic Korean input noise. While such perturbations could potentially be used to stress or evade deployed NLP systems, our goal is to characterize and mitigate this vulnerability. We report aggregate model behavior and propose a defensive routing method intended to improve robustness against typo-corrupted inputs.

## Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2021-II211341; Artificial Intelligence Graduate School Program at Chung-Ang University) and (No. RS-2026- 25546026; Leading Generative AI Human Resources Development). This work was also supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (RS-2026-25494299). This work used datasets from ‘The Open AI Dataset Project (AI-Hub, S. Korea)’. All data information can be accessed through ‘AI-Hub (www.aihub.or.kr)’.

## References

AI-Hub. High-frequency error correction data by interface (keyboard/voice). https://www.aihub.or. kr/aihubdata/data/view.do?dataSetSn=71560. National Information Society Agency, South Korea.

Soyoung An, Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Yountae Jung, Hyosang Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, and 13 others. 2026. Exaone 3.5: Series of large language models for realworld use cases. Preprint, arXiv:2412.04862.

Eric Brill and Robert C. Moore. 2000. An improved error model for noisy channel spelling correction. In Proceedings ofthe 38th Annual Meeting ofthe Associationfor Computational Linguistics, pages 286–293, Hong Kong. Association for Computational Linguistics.

Xingyi Cheng, Weidi Xu, Kunlong Chen, Shaohua Jiang, Feng Wang, Taifeng Wang, Wei Chu, and Yuan Qi. 2020. SpellGCN: Incorporating phonological and visual similarities into language models for Chinese spelling check. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 871–881, Online. Association for Computational Linguistics.

Ronald A Fisher. 1936. The use of multiple measurements in taxonomic problems. Annals of eugenics, 7(2):179–188.

Esther Gan, Yiran Zhao, Liying Cheng, Mao Yancan, Anirudh Goyal, Kenji Kawaguchi, Min-Yen Kan, and Michael Shieh. 2024. Reasoning robustness of LLMs to adversarial typographical errors. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10449–10459, Miami, Florida, USA. Association for Computational Linguistics.

Ji Gao, Jack Lanchantin, Mary Lou Soffa, and Yanjun Qi. 2018. Black-box generation of adversarial text sequences to evade deep learning classifiers. In 2018 IEEE Security and Privacy Workshops (SPW), pages 50–56.

Google DeepMind. 2026. Gemini 3.1 flash-lite model card. https://deepmind.google/models/ model-cards/gemini-3-1-flash-lite/.

Taeeun Kim, Youngsook Song, and Semin Jeong. 2024. KoGEC : Korean grammatical error correction with pre-trained translation models. In Proceedings ofthe 38th Pacific Asia Conference on Language, Information and Computation, pages 170–180, Tokyo, Japan. Tokyo University of Foreign Studies.

Hyunwoo Ko, Guijin Son, and Dasol Choi. 2025. Understand, solve and translate: Bridging the multilingual mathematical reasoning gap. In Proceedings ofthe 5th Workshop on Multilingual Representation Learning (MRL 2025), pages 78–95, Suzhuo, China. Association for Computational Linguistics.

Karen Kukich. 1992. Techniques for automatically correcting words in text. ACM Comput. Surv., 24(4):377–439.

Ohjoon Kwon, Dohyun Kim, Soo-Ryeon Lee, Junyoung Choi, and SangKeun Lee. 2021. Handling out-ofvocabulary problem in hangeul word embeddings. In Proceedings ofthe 16th Conference ofthe European Chapter of the Association for Computational Linguistics: Main Volume, pages 3213–3221, Online. Association for Computational Linguistics.

Myunghoon Lee, Hyeonho Shin, Dabin Lee, and Sung-Pil Choi. 2021. Korean grammatical error correction based on transformer with copying mechanisms and grammatical noise implantation methods. Sensors, 21(8).

Guijin Son, Hanwool Lee, Sungdong Kim, Seungone Kim, Niklas Muennighoff, Taekyoon Choi, Cheonbok Park, Kang Min Yoo, and Stella Biderman. 2025. KMMLU: Measuring massive multitask language understanding in Korean. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4076–4104, Albuquerque, New Mexico. Association for Computational Linguistics.

Guijin Son, Hanwool Lee, Suwan Kim, Huiseo Kim, Jae cheol Lee, Je Won Yeom, Jihyu Jung, Jung woo Kim, and Songseong Kim. 2024. HAE-RAE bench: Evaluation of Korean knowledge in language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 7993–8007, Torino, Italia. ELRA and ICCL.

Zijun Sun, Xiaoya Li, Xiaofei Sun, Yuxian Meng, Xiang Ao, Qing He, Fei Wu, and Jiwei Li. 2021. Chinese-BERT: Chinese pretraining enhanced by glyph and Pinyin information. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2065–2075, Online. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Soyoung Yoon, Sungjoon Park, Gyuwan Kim, Junhee Cho, Kihyo Park, Gyu Tae Kim, Minjoon Seo, and Alice Oh. 2023. Towards standardizing Korean grammatical error correction: Datasets and annotation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6713–6742, Toronto, Canada. Association for Computational Linguistics.

William J Youden. 1950. Index for rating diagnostic tests. Cancer, 3(1):32–35.

Raoyuan Zhao, Yihong Liu, Lena Altinger, Hinrich Schuetze, and Michael A. Hedderich. 2026. Evaluating robustness of large language models against multilingual typographical errors. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16059–16078, San Diego, California, United States. Association for Computational Linguistics.

Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Gong, and Xing Xie. 2024a. Promptrobust: Towards evaluating the robustness of large language models on adversarial prompts. In Proceedings ofthe 1st ACM Workshop on Large AI Systems and Models with Privacy and Safety Analysis, LAMPS ’24, page 57–68, New York, NY, USA. Association for Computing Machinery.

Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, and Xing Xie. 2024b. Promptbench: a unified library for evaluation of large language models. J. Mach. Learn. Res., 25(1).

## A Dataset Details

KMMLU. KMMLU (Son et al., 2025) is a Korean expert-level multiple-choice benchmark containing 35,030 four-choice questions across 45 subjects, including humanities, social sciences, and STEM. Unlike Korean benchmarks translated from English, KMMLU is collected from original Korean examinations and therefore provides a natural evaluation setting for Korean language understanding. We use KMMLU as the main evaluation benchmark throughout the paper: measuring accuracy under controlled typo perturbations (Section 5), evaluating held-out typo detection on unseen questions (Section 6.3), and comparing mitigation methods (Section 7). KMMLU is never used for layer selection, probe training, or threshold tuning.

HAERAE-GK. For calibration, we use the General Knowledge subset of HAERAE Bench (HAERAE-GK) (Son et al., 2024), a Korean knowledge benchmark whose source is disjoint from KMMLU. HAERAE-GK contains 176 four-choice questions. We apply the same five typo taxonomies and five perturbation intensity levels to this split, and use it for all fitting or selection steps: Fisher layer selection (Section 6), probe training for heldout typo detection (Section 6.3), and probe training and threshold selection for TACoT (Section 7.1). This design keeps the KMMLU test questions unseen during calibration and prevents leakage from evaluation data.

HRM8K. To test generalization beyond KMMLU, we additionally evaluate on HRM8K (Ko et al., 2025), a Korean mathematical reasoning benchmark of 8,011 problems spanning multiple sources, including translated and native Korean math competitions. Unlike KMMLU, HRM8K requires a free-form generated solution rather than a single-letter answer, and its subject matter (mathematical word problems) is disjoint from the encyclopedic and professional knowledge covered by KMMLU. We apply the same five typo taxonomies and five perturbation intensity levels to the question text, and use HRM8K exclusively for evaluation (Section 7.4): the probe and TACoT routing threshold are still trained on HAERAE-GK, with no HRM8K questions used for fitting or selection.

## B Perturbation Generation Details

We detail how each of the five perturbation types (Section 3) is generated, and release the full implementation so the benchmark can be regenerated exactly.

For a Hangul syllable s, DECOMPOSE $: ( s ) \ = \quad$ $( c , v , f )$ returns its onset, nucleus, and (possibly empty) coda, with inverse COMPOSE. The number of edits for type t at level $\ell \in \{ 5 , 1 0 , 1 5 , 2 0 , 2 5 \} \%$ is $k = \operatorname* { m a x } ( 1 , \lfloor n _ { t } ( x ) { \cdot } \ell { \cdot } 0 . 0 5 \rfloor )$ , where $n _ { t } ( x )$ is the number of spaces (T4) or complete Hangul syllables (otherwise). Levels are applied independently to the clean input.

T1, T3, and T5 share one structure: select k eligible sites without replacement, decompose each, apply a type-specific edit, and recompose (Algorithm 1).

Algorithm 1 Shared structure for T1, T3, T5   
1: procedure $\mathrm { P E R T U R B } ( x , k , e l i g i b l e )$   
2: $I \gets \mathrm { S E L E C T S I T E S } ( x , k , e l i g i b l e )$   
3: for all $i \in I$ do   
4: $( c , v , f ) \gets \mathrm { D E C O M P O S E } ( x _ { i } )$   
5: $x _ { i } \gets \mathrm { E D I T } ( c , v , f )$   
6: end for   
7: return x   
8: end procedure

The type-specific edit differs as follows. T1 (Substitution) shuffles the component order and replaces the first component that has an adjacencymap entry with a uniformly sampled candidate. T3 (Repetition) picks one component at random and emits it twice: once inside the recomposed syllable, once as a standalone jamo appended after it. T5 (Transposition) swaps the onset and coda if a coda exists, or the onset and nucleus otherwise, producing a raw jamo sequence rather than a valid syllable; since some syllables are invariant under transposition, 2k sites are oversampled and the procedure stops once k edits have actually changed the input.

T2 (Jongseong Deletion) and T4 (Space Deletion) do not follow this structure. T2 drops the coda of syllables that have one: EDIT $( c , v , f )$ = $\mathrm { C o M P O S E } ( c , v , \varepsilon ) ( \breve { \vec { \circ } } \vec { \mathnormal }  \breve { \vec { \circ } } \vec { \mathnormal } )$ . T4 selects space characters instead of syllables and removes them, merging adjacent eojeol $( \breve { \breve { \circ } } \downarrow \breve { \breve { \jmath } } \wedge \jmath \breve { \breve { \circ } } \rbrack \to \breve { \breve { \circ } } \rbrack \breve { \breve { \jmath } } \wedge \jmath \breve { \jmath }$ <sup>ᄒ</sup>ᅬ).

Substitution candidates (T1) come from the standard Dubeolsik Korean keyboard layout: each key’s horizontal and diagonal neighbors, keeping only the two nearest by key-center distance when more remain. Full adjacency maps are released with our implementation.

## C Naturalness of the Perturbation Taxonomy

To verify that our five perturbation types reflect real-world typing behavior rather than an artificial construction, we analyze the AI-Hub "High-Frequency Error Correction Data by Interface (Keyboard/Voice)" (AI-Hub), a corpus of real Korean keyboard and voice input errors containing 729,755 annotated error spans. We map it to our five perturbation types.

Table 6 shows that our taxonomy accounts for 96.2% of real errors, with only 3.8% unclassified. Jamo-fragmenting errors (Repetition + Transposition) constitute 16.8% of keyboard typos, confirming that structural jamo corruption is a genuine, frequent phenomenon rather than a synthetic artifact. The most frequent real jamo confusions (e.g.,

ㅣ↔ㅏ adjacent-key, ㅔ↔ㅐ near-homophone) also match the keyboard-layout assumptions underlying our substitution model.

## D Prompt and Inference Details

Figures 6, 7, and 8 show the system prompts used in our experiments. The user message appends the question, answer choices formatted as (A)–(E), and a Korean answer cue. GEC-based inference uses the standard prompt with the corrected question as input, applying the publicly released Ko-GEC model (Kim et al., 2024) to typo-perturbed questions before standard inference.

All models are served via vLLM with greedy decoding at temperature 0; non-CoT generations use a maximum of 8 new tokens and CoT generations use a maximum of 1024 new tokens. Hidden states are extracted using HuggingFace Library. The logistic regression probe is implemented in scikit-learn with L2 regularization (C=1.0) on standardized features.

## E Additional Results for Typo Robustness

One might further expect larger or more capable models to be more robust to typographical noise. Figure 9 evaluates two API-scale models, Gemini-3.1-Flash-Lite (Google DeepMind, 2026) and Qwen3-235B-A22B-2507 (Yang et al., 2025), on the same perturbations. The degradation pattern persists: jamo-level perturbations consistently cause larger accuracy drops than Space Deletion across both models, suggesting that Korean typo vulnerability is not a limitation of model scale or training regime.

Standard Prompt   
당신은 한국어 시험 ᄆ<sub>ᅮ</sub>ᆫ제ᄅ<sub>ᅳ</sub>ᆯ 푸ᄂ<sub>ᅳ</sub>ᆫ AI입니다.   
(You are an AI that solves Korean exam questions.)   
주어진 선택지 ᄌ<sub>ᅮ</sub>ᆼ 정답의 알파벳(대ᄆ<sub>ᅮ</sub>ᆫ자 한 글자)만   
ᄎ<sub>ᅮ</sub>ᆯ력하세요.   
(Output only the letter of the correct answer in uppercase.)

Figure 6: Standard system prompt used for baseline inference and GEC-based inference.  
![](images/0e4426636abfcb9879b523e7e92e2654e9611483463863593452f4559dbabea3.jpg)  
Figure 7: Meta-Cognition system prompt used for metacognition inference.

Beyond model scale, another possible source of variation is the perturbation procedure itself, since affected syllables are selected at random within each intensity level (Section 3.1). We verify that this randomness does not drive our findings by regenerating the entire benchmark under two additional seeds and re-evaluating all four models; the corruption rate is fixed by the intensity, so only which syllables are affected varies across seeds.

Table 7 reports the mean accuracy over all 25 typo conditions (5 types × 5 levels) under three seeds. The mean moves by at most 0.07 points across seeds, and even the largest single-condition deviation is negligible: in the worst case across all models and conditions, Jongseong Deletion at level 3 on EXAONE-7.8B, accuracy ranges only from 43.26 to 43.83, a 0.57-point spread. No condition shows a meaningful seed effect.

## F Additional Representational Analyses

## F.1 Layer and Type Selection via Fisher Score

Figure 10 plots the Fisher separation score across transformer layers. The curves show that typo sensitivity is not uniformly distributed across the model: each model has a localized region where clean and typo representations are more clearly separated. We select the typo-sensitive layer from these peaks and use the selected layer for the subsequent direction and probe analyses. The selected layers are layer 10 for EXAONE-2.4B, layer 9 for EXAONE-7.8B, layer 9 for A.X-3.1-Light, and layer 18 for Qwen3-4B.

<table><tr><td>Error Source</td><td>Sub.</td><td>Jong.</td><td>Rep.</td><td>Space.</td><td>Trans.</td><td>Unclass.</td></tr><tr><td>All</td><td>34.4%</td><td>18.9%</td><td>9.5%</td><td>50.1%</td><td>1.8%</td><td>3.8%</td></tr><tr><td>Keyboard</td><td>39.5%</td><td>19.6%</td><td>12.9%</td><td>43.5%</td><td>3.9%</td><td>0.9%</td></tr></table>

Table 6: Coverage of real Korean input errors by our taxonomy, on 729,755 error spans from AI-Hub. All includes both keyboard and voice input errors; Keyboard restricts to the keyboard-only subset. Each error may carry multiple labels, so rows can exceed 100%.

CoT Prompt   
당신은한국어 시험 ᄆ<sub>ᅮ</sub>ᆫ제ᄅ<sub>ᅳ</sub>ᆯ 푸ᄂ<sub>ᅳ</sub>ᆫ AI입니다.   
(You are an AI that solves Korean exam questions.)   
ᄆ<sub>ᅮ</sub>ᆫ제에오타가 포함되어 있ᄋ<sub>ᅳ</sub>ᆯ 수 있습니다.   
(The question may contain typographical errors.)   
ᄆᆫ제의 의도ᄅᆯ 파악하기 위해 단계적으로 ᄇᆫ석한 뒤,   
마지막 ᄌᆯ에 ’정답: X’ 형식으로 알파벳 한 글자만 ᄎᆯ   
력하세요.   
(Analyze the question step by step to infer its intended   
meaning, then output only the answer letter in theformat   
‘Answer: X’ on the last line.)  
Figure 8: Chain-of-thought system prompt used for full-CoT and typo-aware mitigation inference.

<table><tr><td>Model</td><td>Seed 1</td><td>Seed 2</td><td>Seed 3</td><td>Max  $\Delta$ </td></tr><tr><td>EXAONE-2.4B</td><td>39.76</td><td>39.76</td><td>39.72</td><td>0.04</td></tr><tr><td>EXAONE-7.8B</td><td>44.11</td><td>44.05</td><td>44.08</td><td>0.06</td></tr><tr><td>A.X-Light</td><td>47.17</td><td>47.24</td><td>47.22</td><td>0.07</td></tr><tr><td>Qwen3-4B</td><td>47.79</td><td>47.79</td><td>47.77</td><td>0.02</td></tr></table>

Table 7: Mean accuracy over all 25 typo conditions, under three random seeds for perturbation sampling.

Table 8 reports the Fisher separation score at the selected layer for each typo type, averaged over perturbation intensities. Space Deletion consistently yields the lowest scores across all models, with an average of 0.009 compared to 0.052–0.164 for jamo-level types, confirming that it induces qualitatively weaker representational shifts. Among jamo-level types, Jamo Transposition produces the largest separation, while Jongseong Deletion produces the smallest — a pattern that mirrors their relative probe detection difficulty in Table 2.

![](images/735c84d8bbfe222a08522be9541f4f2ae53dd934a097db0c1408644cc5dde1a5.jpg)

![](images/6571266a52add7e1a2d9b9db0e9e5a80dd0a2111032d1df481649f3007dda82d.jpg)  
Figure 9: KMMLU accuracy under Korean typo perturbations for API-scale models. Dashed lines indicate clean baseline accuracy.

## F.2 Detection Robustness Across Intensity

Section 6.3 reports held-out detection AUROC pooled over all five perturbation intensities (5%– 25%), showing strong detection across all four models. Table 9 breaks this same held-out probe down by intensity level, averaged over the four held-out jamo-level typo types, and the same conclusion holds at a finer granularity: even at the weakest level $l _ { 1 }$ , AUROC is 0.79–0.85, far above chance, and rises monotonically with intensity. If the probe were merely detecting semantic incoherence rather than the typo itself, its performance should drop sharply at low intensities where sentence coherence is largely preserved; instead, de-

![](images/eb5f3c38dc302a6651ba9539a6e812a898ac07253b78a5293c14ce3e77338676.jpg)

![](images/137bad09de90e2bb9d7102e95931a5606216e48d820e80ae61170ccb4d2f6f9b.jpg)

![](images/20f74f5fdc4dcf599f485af99d4627b4e07ef5dc703002fa294564581936256b.jpg)

![](images/4200b8c96c1741d505275ebcca279a30f1b5c03d8d0732ec890a2e9e57f6d871.jpg)  
Figure 10: Layer-wise Fisher separation score $J ( l )$ between clean and typo hidden states. Dashed vertical lines indicate the selected typo-sensitive layer for each model.

<table><tr><td>Model</td><td>Sub.</td><td>Jong.</td><td>Rep.</td><td>Space.</td><td>Trans.</td></tr><tr><td>EXAONE-2.4B</td><td>0.062</td><td>0.061</td><td>0.103</td><td>0.010</td><td>0.220</td></tr><tr><td>EXAONE-7.8B</td><td>0.112</td><td>0.093</td><td>0.115</td><td>0.021</td><td>0.230</td></tr><tr><td>A.X-Light</td><td>0.034</td><td>0.035</td><td>0.039</td><td>0.004</td><td>0.093</td></tr><tr><td>Qwen3-4B</td><td>0.023</td><td>0.020</td><td>0.055</td><td>0.001</td><td>0.112</td></tr><tr><td>Avg.</td><td>0.058</td><td>0.052</td><td>0.078</td><td>0.009</td><td>0.164</td></tr></table>

Table 8: Fisher separation score J at the Fisher-selected layer, averaged over perturbation intensities. Abbreviations denote Jamo Substitution (Sub.), Jongseong Deletion (Jong.), Jamo Repetition (Rep.), Space Deletion (Space.), and Jamo Transposition (Trans.).

tection is already strong at $l _ { 1 }$ , indicating that the probe responds to the perturbation itself.
<table><tr><td>Model</td><td> $l _ { 1 }$ </td><td> $l _ { 2 }$ </td><td> $l _ { 3 }$ </td><td> $l _ { 4 }$ </td><td> $l _ { 5 }$ </td></tr><tr><td>EXAONE-2.4B</td><td>0.789</td><td>0.883</td><td>0.942</td><td>0.972</td><td>0.984</td></tr><tr><td>EXAONE-7.8B</td><td>0.846</td><td>0.921</td><td>0.966</td><td>0.984</td><td>0.991</td></tr><tr><td>A.X-Light</td><td>0.826</td><td>0.924</td><td>0.975</td><td>0.991</td><td>0.997</td></tr><tr><td>Qwen3-4B</td><td>0.798</td><td>0.880</td><td>0.940</td><td>0.972</td><td>0.985</td></tr></table>

Table 9: Held-out detection AUROC by perturbation intensity, averaged over the four held-out jamo-level typo types.

## F.3 Isolating the Typo Signal from Answer Correctness

Figure 11 visualizes hidden states after projecting them onto two directions: the clean incorrectness direction and the typo direction orthogonal to it. If typo-induced errors were merely ordinary answer failures, typo examples would collapse mainly along the clean incorrectness axis. Instead, both typo-correct and typo-wrong examples consistently shift along the orthogonal typo direction. This supports the main-text claim that Korean typos induce a distinct representational displacement rather than only increasing the likelihood of an incorrect answer.

## G Probe Training Details

For held-out typo detection (Section 6.3), we train a logistic regression probe on the last-token hidden state at the Fisher-selected layer, using clean inputs and typo-perturbed inputs from HAERAE-GK as binary labels. The training set is balanced by randomly sampling typo examples to match the number of clean inputs, drawing only from the nonheld-out typo types.

For TACoT (Section 7.1), we split the balanced HAERAE-GK calibration data into training and validation subsets with an 80/20 split. The probe is trained with standardized hidden states using logistic regression. The routing threshold θ is selected by maximizing Youden’s J statistic (Youden, 1950), $J = \mathrm { T P R } - \mathrm { F P R }$ , on the validation split. No KMMLU examples are used for either probe training or threshold selection.

## H Case Study

We provide representative recovery cases where Standard inference predicts an incorrect answer, while CoT inference recovers the correct answer. The examples cover four jamo-level typo types: Jamo Substitution, Jongseong Deletion, Jamo Repetition, and Jamo Transposition. For readers unfamiliar with Korean, we briefly describe the role of the CoT trajectory before each figure.

Figure 12 shows a Jamo Substitution case. The corrupted question asks for the output voltage of an amplifier, but several Korean words are misspelled. Standard inference selects the wrong option, whereas CoT recovers the intended voltagegain calculation and obtains the correct answer.

Figure 13 shows a Jongseong Deletion case. The input contains valid-looking but corrupted Korean forms, so the typo is not visually obvious as an invalid string. CoT nevertheless extracts the numerical quantities and performs the unit conversion needed to answer the question.

![](images/527bf878360e6f6f2c533fe42b66b237665b4f93731860b06ef39212659d922b.jpg)  
Base-Correct Base-Wrong

Figure 11: Full direction projections of hidden states at the Fisher-selected typo-sensitive layer for all evaluated models. The horizontal axis corresponds to the clean incorrectness direction, while the vertical axis captures the typo direction orthogonal to it.

Figure 14 shows a Jamo Repetition case. The repeated jamo exposes an abnormal Korean character sequence inside the word for “screw.” CoT explicitly identifies the corrupted form and uses the surrounding context to infer that the question refers to a micrometer.

Figure 15 shows a Jamo Transposition case. The typo breaks multiple syllable blocks and exposes raw jamo throughout the question. CoT first reconstructs the likely clean question, then reasons over the answer choices to identify the only output device.

## I Use of AI Assistants

ity, and grammar, and not for ideation or content generation.

We authored this manuscript ourselves; AI assistance was used solely for refinement of style, clar-

A.X-Light: Jamo Substitution   
Typo input   
40[db]의잔압이득ᄅ<sub>ᅳ</sub>ᆯ 가 증폭기에 10[mV]의전압ᄋ<sub>ᅳ</sub>ᆯ 입력에 가하면 ᄎ<sub>ᅮ</sub>ᆯ력전압은 몇 [V] 인가?   
[An amplifier with a voltage gain of 40[dB] has an input of 10[mV]. What is the output voltage in [V]?]   
Choices   
(A) 0.1[V] (B) 1[V] (C) 10[V] (D) 100[V]   
Standard output   
C   
CoT trajectory   
이ᄆ<sub>ᅮ</sub>ᆫ제ᄅ<sub>ᅳ</sub>ᆯᄑ<sub>ᅮ</sub>ᆯ기 위해 증폭기의 전압 이득(전압 이득 = ᄎ<sub>ᅮ</sub>ᆯ력 전압/입력 전압)ᄋ<sub>ᅳ</sub>ᆯ 이용하겠습니다.   
주어진정보:   
• 잔압(전압 이득) =40 dB   
• 입력전압 =10 mV   
먼저,dB(데시벨)로 표현되 전압 이득ᄋᆯ 선형 값으로 변화해야 합니다. 전압 이득(선형) = 10<sup>(dB/20)</sup>   
따라서: 전압 이득(선형) = 10<sup>(40/20)</sup> = 10<sup>2</sup> = 100   
이제ᄎ<sub>ᅮ</sub>ᆯ력전압ᄋ<sub>ᅳ</sub>ᆯ계산할 수 있습니다: ᄎ<sub>ᅮ</sub>ᆯ력 전압 = 입력 전압 × 전압 이득(선형)   
출력 전<sup>아</sup>ᆸ = 10 mV × 100 = 1000 mV = 1 V   
[To solve this, I use the voltage gain formula (gain = output / input). Given: voltage gain = 40 dB, input = 10 mV.   
Converting to linear: 10<sup>(40/20)</sup> = 100. Output = 10 mV × 100 = 1000 mV = 1 V.]   
CoT output   
B  
Figure 12: Case study for A.X-Light under Jamo Substitution. Standard inference predicts an incorrect answer, while CoT inference recovers the correct answer.

EXAONE-7.8B: Jongseong Deletion   
Typo input   
노선의전체의 기이가 2km 인 다각측량ᄋ<sub>ᅳ</sub>ᆯ 실시하였더니, 폐합비가 1/5000 이어다. 폐합오차ᄂ<sub>ᅳ</sub>ᆫ 며 cm 인가?   
[A traverse survey on a route oftotal length 2km yielded a closure ratio of1/5000. What is the closure error in cm?]   
Choices   
(A) 0.04 cm (B) 0.4 cm (C) 4 cm (D) 40 cm   
Standard output   
B   
CoT trajectory   
ᄆ<sub>ᅮ</sub>ᆫ제ᄅ<sub>ᅳ</sub>ᆯᄇ<sub>ᅮ</sub>ᆫ석해보면, 주어진 정보ᄂ<sub>ᅳ</sub>ᆫ 다음과 같습니다:   
• 노선의전체길이: 2km   
• <sup>ᄑ</sup>ᅨ<sup>하</sup>ᆸ<sup>ᄇ</sup>ᅵ: 1/5000   
폐합비ᄂ<sub>ᅳ</sub>ᆫ 측량에서 전체 길이에 대한 허용 오차 비ᄋ<sub>ᅲ</sub>ᆯᄋ<sub>ᅳ</sub>ᆯ 나타냅니다. 따라서 폐합오차ᄅ<sub>ᅳ</sub>ᆯ 계산하ᄂ<sub>ᅳ</sub>ᆫ 공식은 다음   
과같습니다:   
폐합오차 = 전체 길이×폐합비   
단위ᄅᆯ 맞추기 위해 전체 길이ᄅᆯ 미터로 변화합니다: 2km = 2000m   
이제 계산ᄋ<sub>ᅳ</sub>ᆯ 진행합니다: 폐합오차 = 2000m × <sup>1</sup><sub>5000</sub> = 0.4m   
미터ᄅᆯ센티미터로 변화하면: 0.4m = 40cm   
따라서정답은:   
[Analyzing the problem, the given information is as follows: total route length = 2km, closure ratio = 1/5000. The closure   
ratio represents the allowable error relative to the total length in surveying. Therefore, closure error = total length ×   
closure ratio. Converting to meters: 2km = 2000m. Closure error = 2000m × 1/5000 = 0.4m. Converting to centimeters:   
0.4m = 40cm. Therefore, the answer is:]   
CoT output   
D  
Figure 13: Case study for EXAONE-7.8B under Jongseong Deletion. Standard inference predicts an incorrect answer, while CoT inference recovers the correct answer.

![](images/cfae0893f74fb9648e1d3194c0d33952ebe777e053ea4a9ec55a9115e67d900c.jpg)  
Figure 14: Case study for EXAONE-2.4B under Jamo Repetition. Standard inference predicts an incorrect answer, while CoT inference recovers the correct answer.

Qwen3-4B: Jamo Transposition   
Typo input   
컴퓨터ㅏㄱ이해할 수 ㅇㅆㅣㄴㄴㅡ 형태로 데ㅣㅇ터ㄹㄹㅡ 변화하여 입력시키ᄂᆫ ㅈㅇㅏ치가 아닌 것은?   
[Which ofthefollowing is NOT a device that converts data into aform understandable by a computer?]   
Choices   
(A) <sup>ᄏ</sup>ᅵ<sup>보드</sup> (B)ᄀ<sup>ᅪᄆ</sup>ᅡ<sup>ᄋ</sup>ᅮ<sup>스</sup> (C) <sup>ᄃ</sup>ᅵ<sup>ᄌ</sup>ᅵ<sup>ᄐ</sup>ᅡ<sup>ᄋ</sup>ᅵ<sup>ᄌ</sup>ᅥ (D)XYᄑᆯ<sup>로ᄐ</sup>ᅥ   
[(A) Keyboard (B) Optical mouse (C) Digitizer (D) XYplotter]   
Standard output   
C   
CoT trajectory   
ᄆᆫ제ᄅᆯ단계적으로 ᄇᆫ석해 보겠습니다.   
ᄆᆫ제 내용 요약:   
"컴퓨터ㅏㄱ이해할 수 ㅇㅆㅣㄴㄴㅡ 형태로 데ㅣㅇ터ㄹㄹㅡ 변화<sub>ᆫ</sub>하여 입력시키ᄂ<sub>ᅳ</sub>ᆫ ㅈㅇㅏ치가 아닌 것은?"   
이 ᄆᆫ장은 명백히 오타가 많고, 자연스러ᄋᆫ 한국어가 아닙니다. 이ᄅᆯ 정리해보면, 다음과 같은 의미로 해석할   
수 있습니다:   
"컴퓨터가이해할 수 있ᄂᆫ 형태로 데이터ᄅᆯ 변화하여 입력시키ᄂᆫ장치가 아닌 것은?"   
즉, 컴퓨터에 데이터ᄅ<sub>ᅳ</sub>ᆯ 입력하ᄂ<sub>ᅳ</sub>ᆫ 장치 ᄌ<sub>ᅮ</sub>ᆼ에서 가장 적절하지 않은 것ᄋ<sub>ᅳ</sub>ᆯ 고르ᄂ<sub>ᅳ</sub>ᆫ ᄆ<sub>ᅮ</sub>ᆫ제입니다.   
각 선택지ᄅ<sub>ᅳ</sub>ᆯ 살펴보겠습니다:   
(A) 키보드: 키ᄅ<sub>ᅳ</sub>ᆯ 누르면 컴퓨터가 이해할 수 있ᄂᆫ 코드로 변화하여 입력합니다. 적절한 입력 장치입니다. (B)   
과마우스: 움직임ᄋ<sub>ᅳ</sub>ᆯX,Y좌표로변화하여입력합니다. 적절한입력장치입니다. (C)디지타이저: 인간의입력   
ᄋ<sub>ᅳ</sub>ᆯ 디지털 신호로 변화합니다. 적절한 입력 장치입니다. (D) XYᄑ<sub>ᅳ</sub>ᆯ로터: 데이터ᄅ<sub>ᅳ</sub>ᆯ 기반으로 물리적 형태로 ᄎ<sub>ᅮ</sub>ᆯ   
력합니다. 입력장치가 아니라 ᄎ<sub>ᅮ</sub>ᆯ력 장치입니다.   
따라서,"컴퓨터가이해할 수 있ᄂ<sub>ᅳ</sub>ᆫ 형태로 데이터ᄅ<sub>ᅳ</sub>ᆯ변화하여 입력시키ᄂ<sub>ᅳ</sub>ᆫ 장치가 아닌 것"은: (D) XYᄑ<sub>ᅳ</sub>ᆯ로터   
[The question contains many typos. Reconstructed: “Which is NOT a device that converts data into a form a computer   
can understand?” (A) Keyboard: input device. (B) Optical mouse: input device. (C) Digitizer: input device. (D) XY   
plotter: outputs data physically — an output device, not an input device.]   
CoT output   
D  
Figure 15: Case study for Qwen3-4B under Jamo Transposition. Standard inference predicts an incorrect answer, while CoT inference recovers the correct answer.