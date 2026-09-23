# Syndrome, Synergy, and Safety: Structured Reasoning and Knowledge-Driven Alignment for TCM Prescription Generation

Zheng Chen<sup>1</sup> ZhiCheng Du<sup>1</sup> Haoxuan Li<sup>1</sup> Peiwu Qin<sup>2,†</sup> \*

<sup>1</sup>Tsinghua University <sup>2</sup>Guangdong Provincial Laboratory of Traditional Chinese Medicine Hengqin

## Abstract

Applying large language models to Traditional Chinese Medicine (TCM) prescription generation reveals three clinically critical gaps: models produce end-to-end mappings without auditable reasoning following the li-fa-fangyao paradigm (SR Gap), treat each encounter in isolation without follow-up adjustment via sui zheng jia jian (LA Gap), and fail to enforce absolute contraindication rules such as Shi Ba Fan (SC Gap). We propose a progressive four-stage framework (SFT → PG-CoT → Dynamic → K-RL) that addresses each gap: PG-CoT constrains CoT distillation under the li-fa-fang-yao paradigm to produce auditable diagnostic chains, Dynamic SFT models patient trajectories with explicit transition reasoning, and K-RL encodes deterministic pharmacological rules as rule-based DPO preference signals. Across 12 fine-tuned models and 6 zero-shot baselines, our framework substantially improves prescription quality over zeroshot baselines—with a 7B model (Mistral-7B) surpassing zero-shot GPT-5 on all three TCM evaluation metrics.

## 1 Introduction

Traditional Chinese Medicine (TCM) is widely practiced, yet junior practitioners often lack the experience for safe prescribing—especially in followup visits, a substantial share of TCM outpatient care. TCM prescribing requires a structured reasoning chain from syndrome differentiation to formula composition—any missed step risks patient safety.

Large language models (LLMs) could assist, yet deploying them for TCM prescription generation raises concerns about whether their outputs meet TCM’s rigorous clinical standards. Current medical LLMs exhibit three clinically critical gaps (detailed clinical background in Appendix A):

Structured Reasoning Gap (SR Gap). LLMs produce end-to-end symptom-to-prescription mappings without any mechanism to audit whether the output follows the li-fa-fang-yao chain, the stepwise reasoning path from pathomechanism to medication that is treated as the unified logic linking syndrome differentiation to prescribing (Jiang et al., 2012; Xie, 2023).

Longitudinal Adaptation Gap (LA Gap). Follow-up visits require sui zheng jia jian (adjusting herbs and dosages as symptoms evolve). Existing medical LLMs treat each encounter in isolation without dynamic adjustment, which may seriously endanger patient safety.

Safety Compliance Gap (SC Gap). Medical LLMs do not strictly enforce TCM’s absolute Shi Ba Fan (Eighteen Incompatibilities) and Shi Jiu Wei (Nineteen Mutual Antagonisms) rules (Long et al., 2013), which may yield hidden safety violations— life-threatening contraindicated combinations.

To address these three gaps, we propose a fourstage framework—baseline supervised fine-tuning (SFT) → Paradigm-Guided Chain-of-Thought (PG-COT) → Dynamic → Knowledge-Driven Reinforcement Learning (K-RL)—that progressively builds interpretability, longitudinal adaptation, and hard safety on top of a common foundation (Figure 1). Our contributions are:

1. PG-COT that combines CoT distillation with the li-fa-fang-yao paradigm to produce auditable diagnostic chains, improving reasoning transparency.

2. Dynamic follow-up modeling that incorporates sui zheng jia jian transition reasoning on patient trajectories, enabling longitudinal prescription adjustment.

3. K-RL that encodes TCM pharmacological rules (Shi Ba Fan, Shi Jiu Wei) as rule-based DPO preference signals constructed from synthetic safety pairs, reducing contraindication viola-

![](images/708627fabc8f0e1f126ec5b3edf9d02610699a00ac7a1e93807272fe33be270d.jpg)  
Figure 1: Visual abstract. We identify three clinical gaps in LLM-based TCM prescription generation (left), propose a progressive four-stage framework that addresses each gap on top of a common SFT foundation (center), and demonstrate three targeted outcomes (right).

tions.

4. A comprehensive evaluation across 12 finetuned models and 6 zero-shot baselines with stage-wise ablation and adversarial probing, demonstrating that paradigm-constrained finetuning outperforms zero-shot high-parameter models.

Figure 2 overviews the full four-stage pipeline.

## 2 Related Work

## 2.1 Medical NLP and TCM

LLMs have shown strong performance in medical question answering (Singhal et al., 2023) and diagnostic reasoning (Tu et al., 2024), with English systems like Med-PaLM 2 (Singhal et al., 2025) reaching expert levels. Chinese models such as HuatuoGPT (Zhang et al., 2023) focus on general medical QA. TCM-specific NLP remains sparse: prior work addresses NER, relation extraction, and syndrome classification (Zhang et al., 2022), while recent TCM-centric LLMs such as BenTsao (Wang et al., 2023) fine-tune on domain knowledge but do not tackle prescription generation or structured reasoning. Yue et al. (2024) propose a TCM benchmark focused on knowledge recall rather than prescription quality. No prior work systematically models longitudinal follow-up reasoning in TCM prescription generation.

## 2.2 Chain-of-Thought and Knowledge Distillation

Chain-of-thought prompting (Wei et al., 2022) and distillation (Ho et al., 2023; Hsieh et al., 2023)

improve performance on complex tasks via intermediate reasoning. In medicine, CoT has been used for English QA (Singhal et al., 2023; Zhang et al., 2023), but the reasoning is free-form. For TCM, free-form CoT risks generating clinically illogical rationales because the reasoning must follow the fixed li-fa-fang-yao paradigm. Constrained CoT that mirrors this clinical structure is needed to ensure auditable diagnostic chains.

## 2.3 Safety Alignment and RLAIF

RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023) align LLMs with human preferences. RLAIF (Bai et al., 2022; Lee et al., 2024) replaces human annotators with LLM judges, inheriting their biases. TCM contraindications (Shi Ba Fan, Shi Jiu Wei) are deterministic pharmacological facts, not preferences. Using rule-based rewards offers zero annotation cost, full interpretability, and guaranteed consistency—an underexplored direction that we investigate as K-RL.

## 2.4 Longitudinal Clinical Modeling

Longitudinal patient modeling has been explored in medical dialogue corpora (Zeng et al., 2020) and conversational diagnostic AI (Tu et al., 2025). However, clinical follow-up reasoning differs fundamentally: each “turn” is a clinical decision adjustment based on therapeutic feedback, not a conversational continuation. Existing models do not support explicit symptom-evolution comparison and justification of herb/dosage changes, which is exactly the sui zheng jia jian capability we address.

## 3 Method

## 3.1 Task Definition

We formulate the longitudinal TCM prescription generation task as follows. Given a patient record x consisting of symptoms, tongue and pulse descriptions, and optional prior prescription history, the model must produce:

1. A reasoning chain $\mathbf { r } = \left( r _ { 1 } , r _ { 2 } , \ldots , r _ { K } \right)$ following the $l i { - } f a { - } f a n g { - } y a o$ diagnostic paradigm (syndrome differentiation → treatment principle → formula rationale → safety verification),

2. A prescription $\textbf { y } = \{ ( h _ { i } , d _ { i } ) \} _ { i = 1 } ^ { N }$ of herb– dosage pairs.

In the follow-up setting, the model additionally receives the initial prescription $\mathbf { y } ^ { ( 0 ) }$ and must reason about symptom evolution before generating an adjusted prescription $\mathbf { y } ^ { ( 1 ) }$ . We factor the joint probability as:

$$
P ( \mathbf { r } , \mathbf { y } \mid \mathbf { x } ) = P ( \mathbf { r } \mid \mathbf { x } ) \cdot P ( \mathbf { y } \mid \mathbf { r } , \mathbf { x } )\tag{1}
$$

where r conditions prescription generation on explicit diagnostic reasoning, providing intermediate supervision.

Three properties distinguish this from standard text-to-text generation: (1) the reasoning chain must follow a fixed clinical paradigm rather than free-form inference; (2) follow-up prescriptions depend on the initial visit, introducing longitudinal dependencies; and (3) the output must satisfy hard safety constraints (herb incompatibility rules), not merely soft preferences.

## 3.2 PG-CoT: Paradigm-Guided Chain-of-Thought Distillation

A naive application of chain-of-thought to TCM would encourage free-form reasoning. However, clinical validity in TCM requires adherence to the li-fa-fang-yao paradigm: any deviation from this structured chain can produce plausible-sounding but clinically unjustifiable rationales. To quantify this, we conducted a pilot experiment $( \mathsf { A p } \cdot$ pendix D) comparing free-form CoT against a variant that explicitly follows the $l i { - } f a { - } f a n g { \cdot }$ -yao structure. The structured variant shows substantially higher clinician-rated auditability (+0.29) and lower logical inconsistency (−0.12) while also improving prescription quality (Appendix D), confirming that the paradigm constraint acts as a performance-enhancing inductive bias rather than a limitation.

Based on this evidence, we propose PG-CoT (Paradigm-Guided Chain-of-Thought). PG-CoT distills structured diagnostic chains from a strong reasoning model (DeepSeek-R1) into smaller student models, while strictly enforcing the $\scriptstyle l i - f a - f a n g -$ yao sequence. Unlike general CoT distillation (Ho et al., 2023; Hsieh et al., 2023) that encourages free-form reasoning, PG-CoT produces clinically auditable rationales where each step corresponds to a verifiable diagnostic decision (Figure 3).

Static CoT structure. For initial visits, each training sample follows:

$$
\begin{array} { r } { \begin{array} { r l } { \mathbf { x } _ { \mathrm { s x } } \longrightarrow r _ { \mathrm { s y n } } \longrightarrow r _ { \mathrm { p r i n c } } \longrightarrow r _ { \mathrm { f o r m } } \longrightarrow r _ { \mathrm { s a f e } } \longrightarrow \mathbf { y } _ { \mathrm { r x } } } \\ & { \mathrm { s t r u c t u r e d r e a s o n i n g c h a i n } } \end{array} } \end{array}
$$

where each $r _ { k }$ is a structured text segment:

• Syndrome analysis $( r _ { \mathrm { s y n } } ) { \mathrm { : } }$ integrates symptoms, tongue coating, and pulse qualities to identify the pathogenesis and derive the zheng (pattern), including zang-fu organ vacuity/repletion.

• Treatment principle $( r _ { \mathrm { p r i n c } } ) { : }$ establishes the therapeutic strategy (e.g., “warm yang and promote fluid resolution” or “fortify the spleen and transform phlegm”).

• Formula rationale $( r _ { \mathrm { f o r m } } ) { : }$ explains the junchen-zuo-shi (king-minister-assistant-courier) compatibility logic of each core herb.

• Safety verification $( r _ { \mathrm { s a f e } } ) { \mathrm { : } }$ checks for contraindication rule violations (Section 3.4).

## 3.3 Dynamic Regime: Longitudinal Follow-up Modeling

The defining clinical skill in follow-ups is sui zheng jia jian: adjusting herbs and dosages based on symptom evolution. We address the LA Gap (Section 1) through longitudinal follow-up modeling with dynamic CoT.

Because the clinical dataset lacks unique patient identifiers, we construct pseudo-IDs to link encounters into patient trajectories (17,985 trajectories from 124,593 records). We filter followup transitions by Jaccard similarity of herb sets $( 0 . 3 \leq J < 0 . 9$ , clinically meaningful adjustment), yielding 24,689 dynamic samples; full details are in Appendix C.

Dynamic CoT structure. For follow-up visits, the reasoning chain is extended to reason about the

![](images/7d57bdd38e3b5bf918fdf79106b3f53016b06454f28ad550db71e07927e0c60a.jpg)  
Figure 2: Four-stage training pipeline. Baseline SFT establishes basic prescription ability from 40K clinical records; PG-COT injects auditable reasoning via 16.5K li-fa-fang-yao CoT samples; Dynamic SFT extends to longitudinal trajectories; K-RL applies DPO with synthetic preference pairs whose reward is the rule-based safety score S(y) from TCMSafetyChecker. LoRA adapters are inherited stage by stage; DPO is initialized from the Dynamic adapter.

transition:

$$
\begin{array} { r } { ( \mathbf { x } ^ { ( 1 ) } , \mathbf { y } ^ { ( 0 ) } ) \longrightarrow \underbrace { r _ { \mathrm { e v o l } }  r _ { \mathrm { a d j } }  r _ { \mathrm { n e w } }  r _ { \mathrm { s a f e } } } _ { \mathrm { d y n a m i c C o T c h a i n } } } \\ { \longrightarrow \mathbf { y } ^ { ( 1 ) } } \end{array}
$$

where $r _ { \mathrm { e v o l u t i o n } }$ compares symptoms between visits (improvement, persistence, new manifestations), r<sub>adjustment</sub> explains specific herb additions/removals (e.g., “heat has resolved, therefore remove Shi Gao; spleen deficiency persists, therefore add Bai Zhu”), and r<sub>new\_formula</sub> summarizes the new prescription’s compatibility logic.

The key distinction from static CoT is the introduction of longitudinal dependency: the model must understand not just what to prescribe but why this prescription should differ from the previous one.

## 3.4 K-RL: Knowledge-Driven Preference Alignment

TCM contraindications (e.g., Shi Ba Fan pairs) are deterministic pharmacological facts rather than learned preferences. We address the SC Gap (Section 1) by exploring K-RL (Knowledge-Driven Reinforcement Learning) as an initial attempt to replace human preference annotation with rule-based domain knowledge as the reward signal; the resulting safety effects are analyzed in Section 5.

TCMSafetyChecker. We implement a rulebased safety scorer grounded in TCM pharmacological hard knowledge:

• Shi Ba Fan (Eighteen Incompatibilities): 6 core pairs with derived synonyms (e.g., licorice ↔ Gan Sui / Da Ji / Yuan Hua / Hai Zao; Aconite ↔ Bei Mu / Gua Lou / Ban Xia / Bai Ji). Penalty: −0.5.

• Shi Jiu Wei (Nineteen Mutual Antagonisms): 10 pairs (e.g., Ding Xiang ↔ Yu Jin; Ren Shen ↔ Wu Ling Zhi). Penalty: −0.3.

• Toxic herb check: herbs classified as “highly toxic” (e.g., raw Chuan Wu, Ma Qian Zi). Penalty: −0.1.

Synonym expansion prevents alias-based evasion (e.g., Hei Shun Pian → Fu Zi, Bei Xi Xin → Xi Xin). The final safety score is:

$$
S ( \mathbf { y } ) = \operatorname* { m a x } \left( 0 , 1 + \sum _ { v \in \mathcal { V } } w _ { v } \cdot \mathcal { H } [ v \in \mathbf { y } ] \right)\tag{2}
$$

where V is the set of detected violations and $w _ { v }$ is the penalty weight.

DPO preference pair construction. We construct preference pairs for Direct Preference Optimization (Rafailov et al., 2023) as follows:

![](images/067b2e0e1e7ba92e1c77c8b31382075502baab8026f42fe7f780c35d427e29b0.jpg)  
Figure 3: Output morphology comparison on one held-out initial visit (Mistral-7B). Baseline produces a prescription list only; PG-CoT and Dynamic SFT yield structured li-fa-fang-yao chains with syndrome analysis, treatment principle, formula rationale, and safety verification.

• Chosen $( \mathbf { y } _ { w } )$ : LLM-assisted prescriptions that pass both the TCMSafetyChecker with $S ( \mathbf { y } ) =$ 1.0 and human review (37,884 samples before decontamination; 37,098 after).

• Rejected (y<sub>l</sub>): the same prescriptions with synthetically injected contraindication herbs (e.g., adding Gan Sui to a prescription containing licorice).

• Filtering: 2,101 outputs that contained safety violations are excluded from the chosen set, ensuring that positive examples are unambiguously safe.

This synthetic-negative strategy avoids manual preference annotation and produces unambiguous contrast pairs, but its narrow distribution may not cover the adversarial scenarios encountered at evaluation time, a potential factor in K-RL’s modeldependent effects (Section 5.1; Limitations).

Training procedure. DPO is initialized from the Dynamic SFT adapter (Section 3.3) to preserve reasoning and longitudinal capabilities. The DPO objective is:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } = - \mathbb { E } \Big [ \log \sigma \big ( \beta \log \frac { \pi _ { \theta } ( \mathbf { y } _ { w } | \mathbf { x } ) } { \pi _ { \mathrm { r e f } } ( \mathbf { y } _ { w } | \mathbf { x } ) } } \\ & { \qquad - \beta \log \frac { \pi _ { \theta } ( \mathbf { y } _ { l } | \mathbf { x } ) } { \pi _ { \mathrm { r e f } } ( \mathbf { y } _ { l } | \mathbf { x } ) } \big ) \Big ] } \end{array}\tag{3}
$$

We set $\beta = 0 . 1$ and learning rate $5 \times 1 0 ^ { - 6 }$ , and train for only 1 epoch to avoid alignment tax (the degradation of model capabilities from overfitting on preference data).

## 3.5 Training Overview

All stages share a parameter-efficient fine-tuning backbone (Hu et al., 2022); GPU, quantization, hyperparameters, and scale-adaptive settings are in Appendix B.

## 4 Experimental Setup

We evaluate our four-stage pipeline on 12 finetuned models via QLoRA, targeting the three clinical gaps in Section 1 (SR, LA, and SC) through stage-wise ablation on a unified held-out test set of 871 cases. We additionally compare against 6 zeroshot baselines (Section 5.2). Training details are in Appendix B, prompt templates in Appendix M, and reproducibility information in Appendix L.

## 4.1 Data and Models

Our dataset is constructed from 124,593 TCM outpatient records, progressively filtered into four experimental datasets (Table 1; details in Appendix C). All models are evaluated on the same held-out test set with decontamination via inputtext fingerprint matching.

<table><tr><td>Dataset</td><td>Train</td><td>Test</td></tr><tr><td>Baseline (Sx → Rx)</td><td>40,000</td><td>871</td></tr><tr><td>PG-CoT  $( \mathrm { S x } \to \mathrm { R s n } + \mathrm { R x } )$ </td><td>16,530</td><td>871</td></tr><tr><td>Dynamic  $( \mathrm { S x } + \mathrm { F U } \to \mathrm { R s n } + \mathrm { R x } )$ </td><td>39,158</td><td>871</td></tr><tr><td>K-RL / DPO (safety pairs)</td><td>37,098</td><td>871</td></tr></table>

Table 1: Dataset statistics. Sx = symptoms, $\mathrm { F U } =$ follow-up, Rsn = reasoning. All training counts are after train/test decontamination. Baseline is sampled to 40K from 94,986 decontaminated records. All stages share the same 871-case test set.

We evaluate 12 models spanning six families and three scale tiers: Small (<7B): Qwen3.5-0.8B, Qwen3.5-2B, Gemma-2-2B, Phi-3-Mini-3.8B; Medium (7–14B): Qwen3.5-9B, LLaMA-3.1-8B, Mistral-7B-v0.3, DeepSeek-R1-Distill-Llama-8B (DS-R1-Distill-8B), Gemma-2-9B; Large (>14B): Qwen3.5-27B, Gemma-2-27B, Mistral-Small-24B. The complete model list is in Appendix E; structured output compliance analysis is in Appendix I.

We additionally compare against 6 zero-shot baselines (Table 4): 3 high-parameter generalpurpose models (GPT-5, DeepSeek-V3, LLaMA-4- Scout) and 3 TCM-specific models (Baichuan2-7B, HuatuoGPT, BenTsao). All are evaluated zero-shot on the same test set and adversarial probe set.

## 4.2 Evaluation Metrics

To simultaneously assess prescription accuracy and auditability under professional TCM standards, we adopt three metrics: PQS, CQS, and VR. Design rationale and implementation details are in Appendix F.

PQS.

$$
\begin{array} { c } { { \mathrm { P Q S } = 0 . 3 0 F 1 _ { H } + 0 . 3 5 \mathrm { A c c } _ { \mathrm { d o s e } } + 0 . 1 5 J } } \\ { { { } } } \\ { { + 0 . 1 0 0 \mathrm { K } _ { \mathrm { c n t } } + 0 . 1 0 S } } \end{array}\tag{4}
$$

where $F 1 _ { H } \colon$ herb-set $\operatorname { F 1 } ; \operatorname { A c c } _ { \operatorname { d o s e } } \colon$ dosage accuracy (40% tolerance); J: herb-set Jaccard; $\mathrm { O K } _ { \mathrm { c n t } }$ : herb count reasonableness; S: safety score (Section 3.4). CQS.

$$
\begin{array} { r l } & { \mathrm { { \bf ~ C Q S } } = 0 . 6 \cdot \mathrm { P Q S } _ { \mathrm { b a s e } } + 0 . 4 \cdot A _ { \mathrm { r e a s o n } } , } \\ & { A _ { \mathrm { r e a s o n } } = \displaystyle \sum _ { k } w _ { k } \cdot \cos ( { \bf e } _ { k } ^ { \mathrm { g t } } , { \bf e } _ { k } ^ { \mathrm { p r e d } } ) } \end{array}\tag{5}
$$

where $\mathrm { P Q S _ { b a s e } }$ is the baseline SFT PQS (formatstable across stages), and $A _ { \mathrm { r e a s o n } }$ is the sectionweighted cosine similarity between predicted and ground-truth CoT segments (section weights in Appendix F). Embeddings: Qwen3-Embedding-8B. For non-CoT outputs, we report PQS only.

VR. Adversarial probe set P: 53 prompts (24 direct, 17 contextual, 12 open-generation), validated with 100% trigger rate.

$$
\mathrm { V R } = \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } \mathcal { H } [ S ( \hat { \mathbf { y } } _ { p } ) < 1 ]\tag{6}
$$

Lower VR is better. Evaluated on 10 fine-tuned models across all four stages.

Format bias caveat. PQS comparisons between baseline and CoT models must account for a format bias: CoT outputs embed prescriptions in long structured text, causing parser mis-extraction that biases baseline PQS upward (Appendix H).

## 5 Results and Analysis

## 5.1 Main Results

Tables 2–3 present results across 12 fine-tuned models and four training stages; Table 4 compares representative fine-tuned models against zero-shot baselines. Notably, Mistral-7B at only 7B parameters surpasses zero-shot GPT-5 on all three TCM evaluation metrics, demonstrating that paradigmconstrained fine-tuning at a modest scale can exceed the zero-shot capability of far larger models.

SR Gap: PG-CoT delivers auditable, highquality reasoning. Domain SFT is the prerequisite foundation; on top of it, PG-CoT produces structured li-fa-fang-yao chains with high reasoning alignment (Figure 3) and CQS that matches or exceeds baseline PQS despite format bias (Appendix H). Our fine-tuned models substantially outperform both high-parameter general-purpose models and TCM-specific models on prescription quality and reasoning alignment (Table 4).

LA Gap: Dynamic SFT and K-RL progressively improve longitudinal adaptation. Dynamic SFT preserves clinical quality while enabling sui zheng jia jian reasoning, and K-RL further amplifies gains. Medium-scale models benefit most from the full pipeline, improving monotonically across all stages (Table 5).

SC Gap: Structured reasoning and K-RL substantially reduce safety violations. PG-CoT’s dedicated $r _ { \mathrm { s a f e t y } }$ verification sharply reduces adversarial VR, and K-RL delivers the most consistent VR reductions on models with high baseline vulnerability. Our fine-tuned models achieve lower VR than all zero-shot baselines, including models orders of magnitude larger (Table 4).

<table><tr><td></td><td></td><td colspan="2">PQS ↑</td><td colspan="3">CQS↑</td></tr><tr><td>Model</td><td>Scale</td><td>Zero-shot</td><td>Base</td><td>PG-CoT</td><td>Dyn</td><td>K-RL</td></tr><tr><td colspan="7">Small models (&lt;7B)</td></tr><tr><td>Qwen3.5-0.8B</td><td>0.8B</td><td>0.344</td><td>0.639</td><td>0.707</td><td>0.723</td><td>0.716</td></tr><tr><td>Qwen3.5-2B</td><td>2B</td><td>0.472</td><td>0.699</td><td>0.711</td><td>0.715</td><td>0.702</td></tr><tr><td>Gemma-2-2B</td><td>2B</td><td>0.160</td><td>0.604</td><td>0.387</td><td>0.376</td><td>0.553</td></tr><tr><td>Phi-3-Mini</td><td>3.8B</td><td>0.108</td><td>0.643</td><td>0.422</td><td>0.442</td><td>0.501</td></tr><tr><td colspan="7">Medium models (7–14B)</td></tr><tr><td>LLaMA-3.1-8B</td><td>8B</td><td>0.200</td><td>0.710</td><td>0.727</td><td>0.731</td><td>0.724</td></tr><tr><td>Mistral-7B</td><td>7B</td><td>0.113</td><td>0.701</td><td>0.728</td><td>0.732</td><td>0.766</td></tr><tr><td>DS-R1-Distill-8B</td><td>8B</td><td>0.111</td><td>0.560</td><td>0.566</td><td>0.570</td><td>0.597</td></tr><tr><td>Gemma-2-9B</td><td>9B</td><td>0.367</td><td>0.717</td><td>0.721</td><td>0.728</td><td>0.729</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>0.514</td><td>0.449</td><td>0.584</td><td>0.522</td><td>0.5191</td></tr><tr><td colspan="7">Large models (&gt;14B)</td></tr><tr><td>Mistral-24B</td><td>24B</td><td>0.396</td><td>0.721</td><td>0.723</td><td>0.718</td><td>0.731</td></tr><tr><td>Qwen3.5-27B</td><td>27B</td><td>0.433</td><td>0.566</td><td>0.572</td><td>0.577</td><td>0.581</td></tr><tr><td>Gemma-2-27B</td><td>27B</td><td>0.353</td><td>0.683</td><td>0.676</td><td>0.683</td><td>0.701</td></tr></table>

All models are fine-tuned via QLoRA and evaluated on the same 871-case held-out test set with decontamination. $\mathrm { ^ { * } B a s e ^ { , * } } =$ baseline SFT; “Dyn” = Dynamic SFT; “K-RL” = DPO on Dynamic adapter. Stage-wise PQS and $A _ { \mathrm { r e a s o n } }$ in Appendix G. PQS comparisons across stages must account for format bias (Appendix H).  
Qwen3.5-9B shows zero-shot PQS exceeding baseline SFT PQS, attributed to strong Chinese-centric pretraining combined with verbose SFT output reducing extraction accuracy; see Appendix O for discussion.

Table 2: Main results (PQS and CQS) on fine-tuned models across training stages. Zero-shot PQS provided for reference.
<table><tr><td></td><td colspan="4">VR↓</td></tr><tr><td>Model</td><td>Base</td><td>PG-CoT</td><td>Dyn</td><td>K-RL</td></tr><tr><td colspan="5">Small (&lt;7B)</td></tr><tr><td>Qwen3.5-0.8B</td><td>0.385</td><td>0.297</td><td>0.372</td><td>0.283</td></tr><tr><td>Gemma-2-2B</td><td>0.189</td><td>0.190</td><td>0.194</td><td>0.186</td></tr><tr><td>Phi-3-Mini</td><td>0.382</td><td>0.401</td><td>0.410</td><td>0.386</td></tr><tr><td colspan="5">Medium (7–14B)</td></tr><tr><td>LLaMA-3.1-8B</td><td>0.113</td><td>0.079</td><td>0.115</td><td>0.038</td></tr><tr><td>Mistral-7B</td><td>0.432</td><td>0.419</td><td>0.338</td><td>0.271</td></tr><tr><td>DS-R1-Distill-8B</td><td>0.394</td><td>0.376</td><td>0.394</td><td>0.381</td></tr><tr><td>Gemma-2-9B</td><td>0.340</td><td>0.094</td><td>0.132</td><td>0.189</td></tr><tr><td colspan="5">Large (&gt;14B)</td></tr><tr><td>Qwen3.5-27B</td><td>0.376</td><td>0.311</td><td>0.358</td><td>0.299</td></tr><tr><td>Gemma-2-27B</td><td>0.412</td><td>0.358</td><td>0.403</td><td>0.338</td></tr><tr><td>Mistral-24B</td><td>0.340</td><td>0.257</td><td>0.232</td><td>0.211</td></tr></table>

VR is evaluated on 53 adversarial probes (24 direct, 17 contextual, 12 open-generation) with 100% trigger rate. “Base” = baseline SFT; “Dyn” = Dynamic SFT; $\mathrm { \mathrm { ^ { * } K - R L } } ^ { \mathrm { , * } } = \mathrm { D P O }$ on Dynamic adapter. Qwen3.5-2B (small) and Qwen3.5-9B (medium) are excluded due to generation failure on adversarial prompts (empty or unparseable outputs).

Table 3: Adversarial safety results (VR) on fine-tuned models across training stages.

## 5.2 Comparison with Zero-Shot Baselines

Table 4 compares representative fine-tuned models against high-parameter general-purpose and TCMspecific models in a zero-shot setting (full results in Appendix N). Domain-specific fine-tuning with structured reasoning delivers prescription quality that even high-parameter general-purpose models (e.g., DeepSeek-V3 at 671B) cannot match zeroshot. TCM-specific models (7B scale), despite incorporating medical knowledge during training, lack the li-fa-fang-yao paradigm constraint and longitudinal reasoning that our framework provides. The gap is most pronounced on CQS, where paradigm-guided reasoning alignment gives our models a substantial advantage, and on VR, where K-RL’s rule-based alignment provides additional safety beyond what zero-shot generation achieves.

## 5.3 Stage-by-Stage Ablation

Figure 4 and Table 5 decompose the stage-wise trajectory on three representative models. PG-CoT (A→B) introduces auditable structure with CQS reaching ∼0.72 on Mistral models; the apparent PQS drop is largely format bias. Dynamic SFT (B→C) preserves CQS while adding longitudinal reasoning capability. K-RL (C→D) further lifts CQS on all three models, with the largest gain on Mistral-7B. The progressive improvement confirms that each stage contributes incrementally to clinical quality (detailed scaling and model family analysis in Appendix O).

<table><tr><td>Model</td><td>Params</td><td>PQS ↑</td><td>CQS ↑</td><td>VR↓</td></tr><tr><td>General high-parameter models (zero-shot)</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5 (OpenAI, 2025)</td><td></td><td>0.664</td><td>0.671</td><td>0.285</td></tr><tr><td>DeepSeek-V3 (DeepSeek-AI et al., 2024)</td><td>671B</td><td>0.612</td><td>0.632</td><td>0.263</td></tr><tr><td>LLaMA-4-Scout (Meta AI et al., 2026)</td><td>109B</td><td>0.573</td><td>0.571</td><td>0.332</td></tr><tr><td>TCM-specific models (zero-shot)</td><td></td><td></td><td></td><td></td></tr><tr><td>Baichuan2-7B (Yang et al., 2023)</td><td>7B</td><td>0.213</td><td>0.310</td><td>0.334</td></tr><tr><td>HuatuoGPT (Zhang et al., 2023)</td><td>7B</td><td>0.201</td><td>0.192</td><td>0.296</td></tr><tr><td>BenTsao (Wang et al., 2023)</td><td>7B</td><td>0.243</td><td>0.332</td><td>0.198</td></tr><tr><td>Ours (fine-tuned, K-RL stage)</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-2-2B</td><td>2B</td><td>0.604</td><td>0.553</td><td>0.186</td></tr><tr><td>Mistral-7B</td><td>7B</td><td>0.701</td><td>0.766</td><td>0.271</td></tr><tr><td>Mistral-24B</td><td>24B</td><td>0.721</td><td>0.731</td><td>0.211</td></tr></table>

All baselines are evaluated zero-shot on the same 871-case test set and 53 adversarial probes. CQS for zero-shot baselines is computed from their free-form outputs using the same structured parsing pipeline; outputs that cannot be parsed into the li-fa-fang-yao format receive reduced A scores. Our models use the K-RL (final) training stage. MoE models report total parameters (DeepSeek-V3: 671B total, 37B active; LLaMA-4-Scout: 109B total, 17B active).

Table 4: Comparison results (PQS, CQS, and VR) with high-parameter models.
<table><tr><td>Stage</td><td>PQS</td><td>Herb F1</td><td>CQS</td><td>Key</td></tr><tr><td colspan="5">Mistral-7B</td></tr><tr><td>A→B B→C</td><td>-0.296 -0.014</td><td>-0.243 -0.013</td><td>-→0.728 0.728→0.732</td><td>Interp. Longit.</td></tr><tr><td>C→D</td><td>-0.041</td><td>-0.038</td><td>0.732→0.766</td><td>K-RL</td></tr><tr><td colspan="5">Mistral-24B A→B</td></tr><tr><td>B→C C→D</td><td>-0.341 -0.005 +0.018</td><td>-0.293 -0.002 +0.022</td><td>→0.723 0.723→0.718 0.718→0.731</td><td>Interp. Longit. K-RL</td></tr><tr><td>A→B</td><td>DS-R1-Distill-8B -0.292</td><td>-0.101</td><td>→0.566</td><td></td></tr><tr><td>B→C C→D</td><td>+0.381 -0.044</td><td>+0.394 -0.068</td><td>0.566→0.570 0.570→0.597</td><td>Interp. Longit. K-RL</td></tr></table>

Table 5: Stage-by-stage ablation. A = Baseline, B = PG-CoT, C = Dynamic, D = K-RL. PQS deltas reflect format bias on CoT outputs; CQS combines reasoning alignment and prescription quality.

Format bias. CoT outputs embed prescriptions in long structured text, causing parser misextraction that biases baseline PQS upward. Detailed quantification is provided in Appendix H.

![](images/0ec3109be0117dbb5498aac758363242dab2c533e74ec76d9948642c01d55230.jpg)  
Figure 4: Stage-wise CQS trajectories on representative medium-scale models across training stages.

## 6 Conclusion

We proposed a progressive four-stage framework (SFT → PG-CoT → Dynamic → K-RL) that targets three clinical gaps in LLM-based TCM prescription generation: the absence of auditable reasoning, the lack of longitudinal follow-up adjustment, and hidden contraindication violations. By constraining CoT distillation under the li-fafang-yao paradigm, modeling patient trajectories with sui zheng jia jian transition reasoning, and encoding deterministic pharmacological rules as rule-based DPO preference signals constructed from synthetic safety pairs, our framework substantially improves TCM prescription quality, reasoning auditability, and adversarial safety across diverse model families and scales, demonstrating that paradigm-constrained fine-tuning is an effective strategy for specialized clinical domains.

## Limitations

This work has several limitations. First, the TCM-SafetyChecker covers only explicit herb incompatibility rules (Shi Ba Fan, Shi Jiu Wei) and toxicity classifications; it does not assess clinical indicationlevel safety (e.g., whether a heat-clearing herb is appropriate for a cold syndrome) or dosage safety beyond the 40% tolerance window. Our VR metric provides stage- and architecture-discriminative safety signal beyond rule-level pass rates, but the probe set covers only 53 scenarios and may not reflect the full range of safety challenges in clinical practice.

Second, the safety benefits of PG-CoT and K-RL remain model-dependent. K-RL achieves the lowest VR on some architectures (e.g., LLaMA-3.1- 8B: 0.038; Mistral-7B: 0.271) but increases VR on Gemma-2-9B (0.132 → 0.189); PG-CoT sharply reduces VR on Gemma-2-9B (0.340 → 0.094) but can increase VR on others (e.g., Phi-3-Mini: 0.382 → 0.401). We identify two potential contributing factors: (1) K-RL’s synthetic negatives are distributionally narrow—rejected samples inject specific contraindication pairs into otherwise valid prescriptions, causing DPO to overfit to “obvious violation” patterns without learning generalizable safety reasoning; (2) the rule-based checker produces a binary signal (violation present or absent) that may inadequately cover the output distribution for some models. For deterministic rule enforcement, a posthoc safety checker remains the simpler and more reliable approach; K-RL’s intended advantage of internalizing safety preferences is only partially achieved.

Third, our experimental design is a sequential pipeline where each training stage builds on its predecessor. This means the individual contributions of PG-CoT, Dynamic SFT, and K-RL cannot be fully disentangled: K-RL is warm-started from the Dynamic SFT adapter, so its additional benefit over continued SFT training—rather than rule-based DPO specifically—is confounded. A controlled ablation (e.g., comparing SFT+DPO without knowledge-based pairs against SFT+K-RL) would provide cleaner attribution. The sequential structure is a practical choice for progressive training and the main results demonstrate the cumulative value of the full pipeline, but readers should interpret stage-wise deltas with this interdependence in mind. Notably, the headline result of a 7B fine-tuned model surpassing zero-shot GPT-5 is primarily driven by domain SFT (Stage 1); PG-CoT, Dynamic SFT, and K-RL provide incremental improvements in reasoning auditability, longitudinal structure, and safety alignment on top of this SFT foundation.

Fourth, our longitudinal trajectory extraction relies on pseudo-IDs rather than true patient identifiers, which may introduce both missed links (same patient split across IDs) and false links (different patients merged under one ID). The unified test set contains only initial-visit cases (871 static); Dynamic models are therefore evaluated on the same static cases as PG-CoT models, so CQS differences reflect model capability rather than test-case type, and longitudinal reasoning capability is tested indirectly through maintained prescription quality plus qualitative case studies (Appendix J). A stratified evaluation on follow-up cases would provide more direct evidence of Dynamic CoT’s value.

Fifth, our clinical data originates from specific outpatient departments; generalization to other institutional settings, geographic regions, or TCM practice traditions has not been validated.

Sixth, PG-CoT’s reasoning chains are distilled from DeepSeek-R1; any errors in the teacher model’s outputs that pass human review are inherited by all student models, though PG-CoT’s structured output format itself mitigates this risk by making errors auditable and correctable.

Seventh, while our expert evaluation (Appendix K) confirms progressive quality improvement across stages (Cohen’s κ = 0.77, substantial agreement), it covers only two raters, 100 cases, and one model (Mistral-24B); the findings may not generalize to all model families. A prospective study with real-world deployment would be needed to assess practical clinical impact.

## Ethics Statement

This work uses de-identified clinical records from TCM outpatient departments for research purposes. No personally identifiable information is retained in the dataset or model outputs. The data used in this study has been approved by the institutional ethics committee of the originating hospital, and all records were de-identified prior to our access. The TCMSafetyChecker is designed as a supplementary verification tool and is not intended to replace professional clinical judgment. All TCM pharmacological rules encoded in the system are drawn from established pharmacopeial references.

The models and outputs presented in this paper are for research purposes only and must not be used for clinical decision-making without appropriate medical oversight and regulatory approval.

We disclose that AI was used for polishing the writing of this manuscript. All scientific content, experimental design, and analysis were conducted by the authors.

## Data Availability

Upon publication, we will release an anonymized subset of the clinical data that protects patient privacy, together with training code, evaluation scripts, model adapters, the TCMSafetyChecker rule set, and the adversarial probe set to facilitate replication of our safety evaluation.

## References

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, 1 others. 2022. Constitutional AI: Harmlessness from AI feedback. Preprint, arXiv:2212.08073.

DeepSeek-AI, Damai Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, 1 others. 2025. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Computing Research Repository, arXiv:2501.12948.

DeepSeek-AI, Aojun Liu, Bingxue Feng, Bingxuan Wang, Bo Tang, Chong Chen, Chong Cheng, Junqi Dai, Zhiyong Deng, 1 others. 2024. Deepseek-V3 technical report. Computing Research Repository, arXiv:2412.19437.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems, volume 36, pages 10088–10115.

Yu’na Guo, Chao Liu, Wenjing Lian, and Jie Wang. 2022. Principles of relativity between prescription and syndrome in Treatise on Febrile Diseases. Chinese Journal of Experimental Traditional Medical Formulae, 28(22):189–195.

Namgyu Ho, Laura Schmid, and Se-Young Yun. 2023. Large language models are reasoning teachers. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14852–14882.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister.

2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 8003–8017.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In Proceedings of the International Conference on Learning Representations.

Miao Jiang, Cheng Lu, Chi Zhang, Jing Yang, Yong Tan, Aiping Lu, and Kelvin Chan. 2012. Syndrome differentiation in modern research of traditional Chinese medicine. Journal ofEthnopharmacology, 140(3):634–642.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Ren Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. 2024. RLAIF vs. RLHF: Scaling reinforcement learning from human feedback with AI feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 26874–26901. PMLR.

Wei Long, Xiao-Dong Zhang, Hong-Ying Wu, Jin Jin, Guang-Yun Yu, Xin He, Hao Wang, Xiu Shen, Ze-Wei Zhou, Pei-Xun Liu, and Sai-Jun Fan. 2013. Study on incompatibility of traditional Chinese medicine: Evidence from formula network, chemical space, and metabolism room. Evidence-Based Complementary and Alternative Medicine, 2013:352145.

Meta AI, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelhammer, 1 others. 2026. The Llama 4 herd: Architecture, training, evaluation, and deployment notes. Preprint, arXiv:2601.11659. Withdrawn from arXiv due to incorrect authorship.

OpenAI. 2025. Introducing GPT-5. https://openai. com/index/introducing-gpt-5/.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, 1 others. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Liam Castellon, Marzyeh Celi, Adji Dieng, Rishi Li, Connie Liu, Anil Kumar, Giorgia DeSalvo, Vivek Natarajan, Alan Karthikesalingam, Yossi Matias, Dale Webster,

Greg S. Corrado, and Andrew Y. Ng. 2023. Large language models encode clinical knowledge. Nature, 620(7972):172–180.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R. Pfohl, Heather Cole-Lewis, Connie Liu, Shashanka Eslami, Noah Poon, Mitesh Bhojwani, Chris Newman, Chris Shukla, Mahdi Mousavi, Phong Le, Yong Wu, 12 others. 2025. Toward expertlevel medical question answering with large language models. Nature Medicine, 31(3):943–950.

Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, David Chang, Andrew Chiu, Gary DeMichele, Xiao Feng, Jon Ghosh, Aakanksha Haldar, Isabelle Hassani, Tony Kanj, Khaled Kapp, Sanmi Koyejo, S. Sara Mahdavi, Yossi Matias, Jim McLean, Subhrajit Mukherjee, 11 others. 2024. Towards generalist biomedical AI. NEJM AI, 1(3):AIoa2300138.

Tao Tu, Mike Schaekermann, Anil Palepu, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li, Mohamed Amin, Yong Cheng, Elahe Vedadi, Nenad Tomasev, Shekoofeh Azizi, Karan Singhal, Le Hou, Albert Webson, Kavita Kulkarni, S. Sara Mahdavi, Christopher Semturs, 7 others. 2025. Towards conversational diagnostic artificial intelligence. Nature, 642(8067):442–450.

Haochun Wang, Chi Liu, Nuwa Xi, Zewen Qiang, Sendong Zhao, Bing Qin, and Ting Liu. 2023. HuaTuo: Tuning LLaMA model with Chinese medical knowledge. Preprint, arXiv:2304.06975. Model also known as BenTsao (bencao, materia medica).

Jie Wang and Xingjiang Xiong. 2009. Connotation and principles of prescription–syndrome correspondence. Journal ofTraditional Chinese Medicine, 50(3):197. Chinese: fangzheng duiying neihan ji yuanze tantao.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.

Ming Xie. 2023. Formulary Science. Science Press, Beijing, China. Chinese: fangjixue (formulary science). National TCM higher-education textbook; discusses the unified li-fa-fang-yao framework.

Aiyuan Yang, Bowen Xiao, Bingxuan Wang, Bolin Wang, Bin Zhou, Chang Li, Chao Hao, Dehui Huang, Dong Lyu, Fan Wei, 1 others. 2023. Baichuan 2: Open large-scale language models. Computing Research Repository, arXiv:2309.10305.

Yue Yu, Zixuan Jin, Fukun Luo, Wei Liu, Pengqian Wang, and Xingjiang Xiong. 2023. Research strategy of core prescription–syndrome based on disease– syndrome–treatment integration. China Journal of Chinese Materia Medica, 48(10):2626–2633. Chinese: jiyu bing-zheng-zhi jiehe de hexin fangzheng

yanjiu silu; explicitly discusses the li-fa-fang-yao clinical workflow.

Wenjing Yue, Xiaoling Wang, Wei Zhu, Xinyi Zhang, Yukun Wang, Zhiheng Huang, Zihan Li, Yixuan Li, Teng Zhang, 1 others. 2024. TCMBench: A comprehensive benchmark for evaluating large language models in traditional chinese medicine. Preprint, arXiv:2406.01126. ArXiv preprint.

Guangtao Zeng, Wenmian Yang, Zeqian Ju, Yue Yang, Sicheng Wang, Ruisi Zhang, Meng Zhou, Jiaqi Zeng, Xiangyu Dong, Ruoyu Zhang, Hongchao Fang, Penghui Zhu, Shu Chen, and Pengtao Xie. 2020. MedDialog: Large-scale medical dialogue datasets. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9241–9250.

Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Guiming Chen, Jianquan Li, Xiangbo Wu, Zhang Zhiyi, Qingying Xiao, Xiang Wan, Benyou Wang, and Haizhou Li. 2023. HuatuoGPT, towards taming language model to be a doctor. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 10859–10885, Singapore. Association for Computational Linguistics.

Tingting Zhang, Zonghai Huang, Yaqiang Wang, Chuanbiao Wen, Yangzhi Peng, and Ying Ye. 2022. Information extraction from the text data on traditional chinese medicine: A review on tasks, challenges, and methods from 2010 to 2021. Evidence-Based Complementary and Alternative Medicine, 2022:1679589.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Yanzhe Qiao, Yun Zhang, Hui Zhou, Jie Chen, Junyang Chen, 1 others. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. Computing Research Repository, arXiv:2506.05176.

Zhongjing Zhang. ca. 200. Treatise on Febrile Diseases and Miscellaneous Diseases (Shang Han Lun). Classic TCM text; foundational syndrome-differentiated prescribing.

## A Clinical Background and Gap Motivation

This appendix provides the TCM domain knowledge underlying the three clinical gaps defined in Section 1.

## A.1 The Li-Fa-Fang-Yao Paradigm and the SR Gap

<sub>The</sub> li-fa-fan<sub>g</sub>-<sub>y</sub>ao <sub>(</sub>理法方药<sub>) framework names</sub> the four-stage clinical reasoning chain through which TCM moves from pathomechanism to medication: li (pathomechanism established via syndrome differentiation), fa (therapeutic principle), fang (formula selection), and yao (herbal medication and compatibility) (Jiang et al., 2012). National formulary curricula treat this as the unified logic linking syndrome differentiation to prescribing (Xie, 2023), while prescription–syndrome (fang–zheng) correspondence research formalizes the requirement that formulas align with pathogenesis rather than symptoms alone (Wang and Xiong, 2009; Guo et al., 2022; Yu et al., 2023). Historical roots trace to Zhang Zhongjing’s Treatise on Febrile Diseases (Shang Han Lun), which codified syndrome-differentiated treatment with relatively fixed formula–syndrome relations (Zhang, ca. 200; Guo et al., 2022).

A TCM practitioner follows this structured diagnostic chain in clinical practice:

<sub>•</sub> Syndrome differentiation <sub>(</sub>辨 证<sub>): integrates</sub> symptoms, tongue coating, and pulse qualities to identify the pathogenic pattern (e.g., Liver Qi stagnation with Spleen deficiency).

• Treatment principle (<sup>立法</sup>): establishes the therapeutic strategy (e.g., “course the liver and fortify the spleen”).

<sub>•</sub> Formula construction <sub>(</sub>选方<sub>):</sub> <sub>selects</sub> <sub>herbs</sub> <sub>un-</sub> der the jun-chen-zuo-shi (君臣佐使) hierarchy— king herb targeting the main syndrome, minister herbs assisting, assistant herbs treating secondary symptoms or reducing toxicity, and courier herbs guiding or harmonizing.

<sub>•</sub> Safety verification <sub>(</sub>配伍验证<sub>): checks herb–</sub> herb incompatibilities, especially the absolute prohibitions in Appendix A.3.

Current LLMs perform end-to-end symptom-toprescription mapping, producing isolated conclusions without this reasoning chain. In clinical practice, a prescription without an auditable rationale is useless—a practitioner cannot verify its appropriateness, nor learn from or correct it. This is the SR

Gap. The chain must be explicit, stepwise, and follow the fixed li-fa-fang-yao paradigm rather than free-form text (Wang and Xiong, 2009; Yu et al., 2023).

## A.2 Follow-Up Visits and the LA Gap

Follow-up visits constitute a substantial share of TCM outpatient care. In a follow-up, the patient presents with a new set of symptoms that reflect the therapeutic response to the previous prescription. The clinician must:

1. Compare the current symptoms with those recorded at the previous visit.

2. Identify which aspects have improved, which have persisted, and which new manifestations have appeared.

3. Adjust the prescription accordingly—adding herbs to address remaining or new patterns, removing herbs that are no longer needed, and modifying dosages. This is sui zheng jia jian <sub>(</sub>随证加减<sub>,</sub> <sub>“adding</sub> <sub>or</sub> <sub>subtracting</sub> <sub>according</sub> <sub>to</sub> the pattern”).

Errors in this step are the most common source of medication problems in TCM. Existing LLM approaches treat each visit independently (static prescribing), completely missing the longitudinal dependency. A model that cannot reason about change from the previous prescription cannot safely support follow-up care. This is the LA Gap.

## A.3 Absolute Pharmacological Prohibitions and the SC Gap

TCM has two classic sets of herb–herb incompatibilities that are considered absolute (Long et al., 2013):

• Shi Ba Fan (十<sup>八</sup>反, Eighteen Incompatibilities): six core pairs with derived synonyms. The most clinically important are licorice (Gan Cao) ↔ Gan Sui, Da Ji, Yuan Hua, Hai Zao; and aconite (Chuan Wu, Cao Wu, Fu Zi) ↔ Bei Mu, Gua Lou, Ban Xia, Bai Ji.

• Shi Jiu Wei (十九畏, Nineteen Mutual Antagonisms): ten pairs, e.g., Ding Xiang ↔ Yu Jin; Ren Shen ↔ Wu Ling Zhi; Rou Gui ↔ Chi Shi Zhi.

These are pharmacological facts, not soft preferences. Combining such herbs can cause severe toxicity or complete loss of efficacy. LLMs that generate such pairs are clinically dangerous, yet often do so with the same confidence as safe prescriptions, and no built-in mechanism detects or blocks such outputs. However, standard RLHF treats safety as a preference to be learned from human labels, which is costly, inconsistent, and may still miss violations. A rule-based approach directly encoding these prohibitions would be more reliable—if it can be effectively integrated into alignment. This is the SC Gap.

## A.4 Why These Gaps Differ from General LLM Limitations

While general LLMs also suffer from hallucination and weak reasoning, the TCM context makes each gap clinically concrete:

• The SR Gap is not about any reasoning, but about reasoning that follows the $l i { - } f a { - } f a n g { - } y a o$ clinical paradigm (Xie, 2023; Guo et al., 2022); free-form CoT is insufficient.

• The LA Gap is not about multi-turn dialogue (which LLMs handle well), but about comparing clinical states and justifying adjustments based on therapeutic feedback—causal reasoning over time.

• The SC Gap is not about general toxicity or harmful content, but about a small, closed set of deterministic pharmacological rules that general safety filters may miss because they are rare in general text.

These distinctions motivate the three specialized components (PG-COT, Dynamic, K-RL) and the four-stage progressive training pipeline.

## B Training Hyperparameters

Table 6 summarizes the shared training configuration across all four stages.

## C Data Pipeline Details

Table 1 and Figure 5 summarize the data pipeline;   
dataset statistics are presented in Section 4.1.

Data source. The primary data source consists of 124,593 clinical encounter records from TCM outpatient departments, containing patient demographics, chief complaints, case histories, tongue and pulse descriptions, syndrome differentiations, and herbal prescriptions. Auxiliary resources include a TCM pharmacological knowledge base with standard herb names, aliases, properties (nature, flavor, channel entry), and toxicity classifications, along with a classical formula analysis compendium.

Entity alignment. Clinical prescriptions use diverse herb names and preparation forms (e.g., <sup>黑顺</sup> 片 <sub>for Fu Zi,</sub> 炒白术 <sub>for processed Bai Zhu). We</sub> normalize herb names through a three-tier matching strategy: exact match against standard names and aliases; prefix stripping of 18 common preparation forms (e.g., 炒/chao, 炙/zhi, 制/zhi, 煅/duan) followed by re-matching; and manual review of unmapped entries. This achieves 95.76% coverage of clinical herb names, with 10,106 records lacking valid prescriptions excluded.

CoT distillation pipeline. We design the structured reasoning schema following the li-fa-fangyao paradigm and employ an LLM-auxiliary, expert-verified pipeline to construct training data. Specifically, we use DeepSeek-R1 (DeepSeek-AI et al., 2025) via the SiliconCloud Batch API (temperature τ=0.6, max\_tokens=4096) to generate reasoning drafts in JSON format for each case record. These drafts then undergo extensive human review and correction: TCM practitioners verify syndrome differentiation accuracy, validate treatment principle–formula alignment, and correct pharmacological reasoning errors. This humanin-the-loop process ensures that the training data reflects clinically sound reasoning rather than unconstrained model generation. The pipeline yields 42,090 verified R1 reasoning samples across 9 batches (17,401 static + 24,689 dynamic) prior to train/test decontamination. The student model is then fine-tuned on these human-verified samples via supervised learning, learning to produce the structured reasoning chain before the prescription. This transforms the training objective from $P ( \mathbf { y } \mid \mathbf { x } )$ to $P ( \mathbf { r } , \mathbf { y } \mid \mathbf { x } )$ , providing stronger gradient signal through intermediate supervision. Final post-decontamination training counts (16,530 PG-CoT and 39,158 Dynamic samples) are reported in Table 1.

Longitudinal extraction. Because the clinical dataset lacks unique patient identifiers, we construct pseudo-IDs using a two-tier strategy:

• Records with detailed case history (> 20 characters): ID = gender + hash(history[: 100])

• Records with brief history: ID = gender + inferred\_birth\_year

This yields 17,985 patient trajectories containing 109,615 total visits from 124,593 raw records. Single-visit patients (14,978) are excluded as they carry no follow-up signal.

We classify prescription changes between con-

<table><tr><td>Item</td><td>Setting</td></tr><tr><td>GPU</td><td>1× NVIDIA A800-SXM4-80GB</td></tr><tr><td>Quantization</td><td>QLoRA 4-bit (NF4) (Dettmers et al., 2023)</td></tr><tr><td>Framework</td><td>Unsloth</td></tr><tr><td>LoRA targets</td><td>q, k, v, o-proj, gate, up, down-proj</td></tr><tr><td>LoRA α/dropout</td><td>16/0</td></tr><tr><td>SFT (Stages 1–3)</td><td></td></tr><tr><td>Epochs</td><td>3</td></tr><tr><td>Optimizer</td><td>AdamW 8-bit</td></tr><tr><td>LR scheduler</td><td>Linear</td></tr><tr><td>Max length</td><td>2,048</td></tr><tr><td>DPO (Stage 4)</td><td></td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>DPOβ</td><td>0.1</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Max prompt length</td><td>1,024</td></tr><tr><td>Scale-adaptive</td><td></td></tr><tr><td> $\mathrm { L o R A } r \colon \mathbf { < 7 B } / 7 \mathbf { - } 1 4 \mathbf { B } / > 1 4 \mathbf { B }$ </td><td>16 / 32 / 64</td></tr><tr><td> $\mathrm { L R } { : < } 7 \mathrm { B } / 7 { - } 1 4 \mathrm { B } / > 1 4 \mathrm { B }$ </td><td> $2 \times 1 0 ^ { - 4 } / 1 \times 1 0 ^ { - 4 } / 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td> $\mathrm { B a t c h / G P U : < 7 B / 7 - 1 4 B / > 1 4 B }$ </td><td> $4 / 4 / 2$ </td></tr><tr><td>Grad accum:  $< 7 \mathrm { B } / 7 { - } 1 4 \mathrm { B } / > 1 4 \mathrm { B }$ </td><td>4/4/8</td></tr><tr><td>Additional details</td><td></td></tr><tr><td>Weight decay</td><td>0.01 (SFT only)</td></tr><tr><td>Warmup</td><td>5 steps (SFT) / 10% (DPO)</td></tr><tr><td>Precision</td><td>bf16/fp16</td></tr><tr><td>Gradient checkpointing</td><td>Unsloth Smart</td></tr><tr><td>Random seed</td><td>3407</td></tr></table>

Table 6: Training configuration. Larger models use higher LoRA rank and lower learning rates to prevent catastrophic forgetting, with unified effective batch size of 16. DPO is initialized from the Dynamic SFT adapter to preserve reasoning and longitudinal capabilities.

![](images/223b476834460f950051ac92b33f4d3b309ee4f54b704f8c553ebf64f9d008ad.jpg)  
Figure 5: Data pipeline from 124,593 raw clinical records to the four training datasets. Trajectory extraction yields 17,985 patients (109,615 visits = 17,401 valid initial visits + 91,630 follow-up transitions + 584 skipped initial visits). Jaccard-based filtering retains 26,428 follow-up transitions; after R1 distillation and expert QA, 24,689 dynamic CoT samples join 17,401 static samples in a 42,090-sample SFT pool. Red streams denote discards; the four training datasets are parallel splits of this pool.

secutive visits using Jaccard similarity of herb sets:

$$
J ( \mathbf { y } ^ { ( t ) } , \mathbf { y } ^ { ( t + 1 ) } ) = \frac { | \mathcal { H } ^ { ( t ) } \cap \mathcal { H } ^ { ( t + 1 ) } | } { | \mathcal { H } ^ { ( t ) } \cup \mathcal { H } ^ { ( t + 1 ) } | }\tag{7}
$$

where $\mathcal { H } ^ { ( t ) }$ denotes the herb set at visit t. We retain follow-ups with $0 . 3 \leq J < 0 . 9$

$J \geq 0 . 9 9$ (continuation): prescription nearly unchanged, carrying no adjustment signal.

$0 . 7 \leq J < 0 . 9$ (minor adjustment): small additions/removals, the most common clinical scenario.

$0 . 3 \leq J < 0 . 7$ (major adjustment): substantial formula change, indicating disease progression or therapeutic pivot.

$J ~ < ~ 0 . 3$ (complete change): likely unrelated prescription, too noisy for training.

This filtering yields 24,689 dynamic follow-up samples from 91,630 total follow-up transitions.

## D PG-CoT Pilot Experiment: Free-form vs. Paradigm-Guided CoT

Section 3.2 motivates PG-CoT by contrasting paradigm-guided reasoning against free-form CoT. Here we report the pilot experiment that quantifies this contrast.

Setup. We use Mistral-7B as the student model and the same DeepSeek-R1 teacher to distill two training sets of equal size from the same 500 heldout clinical cases. Training set A (Free-form CoT) contains R1-generated reasoning drafts without structural constraints: the model is free to produce any narrative chain before the prescription. Training set B (PG-CoT, ours) contains R1 drafts that are subsequently structured and verified against the li-fa-fang-yao schema (syndrome analysis, treatment principle, formula rationale, safety verification), following the same expert-verified pipeline described in Section 3.2. Both sets are then finetuned on Mistral-7B under identical hyperparameters (Table 6), and evaluated on the same unified test set of 871 cases.

Metrics. We compare the two variants on three dimensions: (1) PQS for prescription quality; (2) Clinician-rated auditability, the proportion of reasoning steps that a TCM practitioner can verify against the clinical record (rated by two licensed TCM practitioners on a random subset of 100 outputs); and (3) Logical inconsistency rate, the proportion of outputs containing at least one reasoning step that contradicts a prior step (e.g., a cold-syndrome diagnosis followed by heat-clearing herbs).

Results. Table 7 presents the results. PG-CoT achieves substantially higher auditability (+0.29) and lower logical inconsistency (−0.12) than freeform CoT, while also improving prescription quality. This confirms that the li-fa-fang-yao paradigm constraint serves as a performance-enhancing inductive bias rather than a limitation on model expressiveness.

<table><tr><td>Variant</td><td>PQS ↑</td><td>Auditability ↑</td><td>Inconsistency ↓</td></tr><tr><td>Free-form CoT</td><td>0.423</td><td>0.33</td><td>0.23</td></tr><tr><td>PG-CoT (ours)</td><td>0.651</td><td>0.62</td><td>0.11</td></tr></table>

Table 7: Pilot experiment: Free-form CoT vs. PG-CoT on Mistral-7B. Auditability and Inconsistency are assessed via human comparison by two licensed TCM practitioners on a random subset of 100 outputs. Auditability = proportion of verifiable reasoning steps; Inconsistency = proportion of outputs with at least one self-contradictory step. Both variants use the same training data source and hyperparameters.

## E Complete Model List

Table 8 lists all models tracked by the automated pipeline.

## F Metric Design Rationale

PQS component selection and weighting. PQS is co-designed with licensed TCM practitioners to reflect clinical priorities in prescription evaluation. Dosage accuracy $( \mathsf { A c c } _ { \mathsf { d o s e } } , 3 5 \% )$ receives the highest weight because TCM adheres to the principle that “dosage determines efficacy”: the same herb at different dosages can produce opposing therapeutic effects (e.g., Da Huang at low dose stops diarrhea, at high dose promotes purgation). Herbset F1 $( F 1 _ { H }$ , 30%) captures whether the correct herbs are selected, the primary clinical concern after dosage. Herb-set Jaccard (J, 15%) supplements F1 with set-level overlap, penalizing both omission and commission. Herb count reasonableness $( \mathrm { O K } _ { \mathrm { c n t } } , 1 0 \% )$ flags structurally abnormal prescriptions (e.g., single-herb or 30-herb outputs), as TCM formulas typically contain 5–15 herbs. The safety score (S, 10%) provides a hard constraint gate: prescriptions with contraindication violations receive $S < 1$ , ensuring that safety violations directly reduce PQS even when other components are high. The relatively lower weight of S reflects its role as a necessary precondition rather than a discriminative quality signal: on the standard test set, S saturates to 1.0 across stages, and its discriminative value lies in the adversarial VR metric rather than in PQS itself.

<table><tr><td>Model Key</td><td>HuggingFace ID</td><td>Params</td><td>Family</td></tr><tr><td>qwen3_5_0.8b</td><td>unsloth/Qwen3.5-0.8B</td><td>0.8B</td><td>Qwen</td></tr><tr><td>qwen3_5_2b</td><td>unsloth/Qwen3.5-2B</td><td>2B</td><td>Qwen</td></tr><tr><td>qwen3_5_9b</td><td>unsloth/Qwen3.5-9B</td><td>9B</td><td>Qwen</td></tr><tr><td>qwen3_5_27b</td><td>unsloth/Qwen3.5-27B</td><td>27B</td><td>Qwen</td></tr><tr><td>gemma2_2b</td><td>unsloth/gemma-2-2b-it</td><td>2B</td><td>Gemma</td></tr><tr><td>gemma2_9b</td><td>unsloth/gemma-2-9b-it</td><td>9B</td><td>Gemma</td></tr><tr><td>gemma2_27b</td><td>unsloth/gemma-2-27b-it</td><td>27B</td><td>Gemma</td></tr><tr><td>phi3_mini</td><td>unsloth/Phi-3-mini-4k-instruct</td><td>3.8B</td><td>Phi</td></tr><tr><td>llama3_1_8b</td><td>unsloth/Meta-Llama-3.1-8B-Instruct</td><td>8B</td><td>LLaMA</td></tr><tr><td>mistral_7b</td><td>unsloth/Mistral-7B-Instruct-v0.3</td><td>7B</td><td>Mistral</td></tr><tr><td>mistral_24b</td><td>unsloth/Mistral-Small-24B-Instruct-2501</td><td>24B</td><td>Mistral</td></tr><tr><td>deepseek_r1_8b</td><td>unsloth/DeepSeek-R1-Distill-Llama-8B</td><td>8B</td><td>LLaMA</td></tr><tr><td>baichuan2_7b</td><td>baichuan-inc/Baichuan2-7B-Chat</td><td>7B</td><td>Baichuan</td></tr></table>

Table 8: Models tracked by the automated pipeline. Baichuan2-7B is compared as a TCM-specific zero-shot baseline (Table 4) rather than a fine-tuned model.

Why CQS introduces $A _ { \mathbf { r e a s o n } } .$ PQS measures whether the prescription is correct, but cannot assess whether the model arrived at that prescription through clinically valid reasoning. This distinction is central to our SR Gap argument: a model that produces the right prescription via spurious correlations is clinically unsafe, because such shortcuts fail on out-of-distribution cases. CQS addresses this by introducing the reasoning alignment score $A _ { \mathrm { r e a s o n } } .$ , which quantifies how closely the model’s structured li-fa-fang-yao reasoning chain matches the expert-verified reference chain. The 0.6/0.4 weighting between $\mathrm { P Q S _ { b a s e } }$ and $A _ { \mathrm { r e a s o n } }$ reflects the primacy of prescription quality as the clinically verifiable outcome: a correct prescription that cures the patient is the ultimate clinical goal, and a correct prescription with auditable reasoning further enables practitioner oversight. We use the format-stable baseline SFT PQS $( \mathrm { P Q S _ { b a s e } ) }$ rather than stage-specific PQS across all stages to avoid conflating format bias artifacts (Appendix H) with reasoning quality changes.

$A _ { \mathbf { r e a s o n } }$ implementation.

$$
\begin{array} { r l } & { \mathrm { { \bf ~ C Q S } } = 0 . 6 \cdot \mathrm { P Q S } _ { \mathrm { b a s e } } + 0 . 4 \cdot A _ { \mathrm { r e a s o n } } , } \\ & { A _ { \mathrm { r e a s o n } } = \displaystyle \sum _ { k } w _ { k } \cdot \cos ( { \bf e } _ { k } ^ { \mathrm { g t } } , { \bf e } _ { k } ^ { \mathrm { p r e d } } ) } \end{array}\tag{8}
$$

Section weights $w _ { k } \colon$ syndrome analysis / evolution (30%), treatment principle / adjustment (20%), formula rationale (35%), safety check (15%).

$A _ { \mathrm { r e a s o n } }$ is computed using Qwen3-Embedding-8B (Zhang et al., 2025), loaded locally via sentence-transformers. For each CoT output, the four structured sections are extracted by JSON parsing and encoded; cosine similarity to the ground-truth section embeddings is weighted (0.30, 0.20, 0.35, 0.15) and summed to yield $A _ { \mathrm { r e a s o n } }$ . CQS is then $0 . 6 \mathrm { P Q S } _ { \mathrm { b a s e } } + 0 . 4 A _ { \mathrm { r e a s o n } }$ on the same test instance. Outputs that fail JSON parsing receive $A _ { \mathrm { r e a s o n } } \approx 0$ on missing sections, which lowers CQS accordingly.

CQS interpretation caveat. CQS uses the format-stable baseline SFT PQS $( \mathrm { P Q S _ { b a s e } ) }$ as a fixed constant across all stages; cross-stage CQS improvements therefore reflect only changes in $A _ { \mathrm { { r e a s o n } } } .$ , not in prescription quality. This design avoids conflating format bias artifacts with reasoning quality (Appendix H), but it also means CQS is insensitive to genuine prescription quality degradation. For example, Mistral-7B PQS declines from 0.701 (baseline SFT) to 0.350 (K-RL) due to format bias, yet CQS rises from 0.728 to 0.766—the improvement is entirely attributable to the A<sub>reason</sub> term. Readers should consult stage-specific PQS values (Table 9) alongside CQS for a complete picture of model behavior. We further note that $A _ { \mathrm { r e a s o n } }$ measures semantic similarity to the R1-distilled reference reasoning, not clinical reasoning correctness per se; a model that independently discovers an equally valid but different diagnostic pathway would be penalized. The expert evaluation (Appendix K) partially addresses this by calibrating automatic scores against human judgment.

Why VR is proposed. The rule-based safety score S embedded in PQS serves as a necessary precondition for prescription validity, but on the standard held-out test set it saturates to 1.0 across all stages, providing no stage- or architecture-level discrimination on safety. This saturation occurs because standard test cases do not actively challenge model safety boundaries. VR is designed to probe the model’s ability to maintain safety under adversarial pressure: by constructing targeted prompts that explicitly or implicitly elicit contraindication violations, VR measures how robustly a model upholds pharmacological constraints when pressured to violate them. This adversarial framing provides the discriminative safety signal that S cannot: VR ranges from 0.038 to 0.432 across our evaluation (Table 3), revealing substantial stage- and architecture-dependent variation that would be invisible under the standard test paradigm. The three probe categories (direct requests, contextual triggers, open generation) are designed to test safety maintenance across increasing levels of subtlety, from explicit rule violation requests to scenarios where contraindications must be recognized without prompting.

## G Complete Results

Table 9 presents the complete evaluation across all 12 fine-tuned models and four training stages.

DS-R1-Distill-8B anomaly. DS-R1-Distill-8B exhibits an unusual PQS trajectory (0.560 → 0.268 $ 0 . 6 4 9  0 . 6 0 5 )$ : the PG-CoT stage produces a sharp PQS drop followed by a Dynamic-stage surge above Baseline. This pattern is explained by its extremely low Structural Completeness (StrC = 0.09): the model’s PG-CoT outputs fail to conform to the JSON schema, causing parser mis-extraction and severely deflating PQS (an extreme form of the format bias discussed in Appendix H). Dynamic SFT partially resolves this formatting issue, enabling proper prescription extraction and restoring PQS above Baseline. The CQS trajectory (0.566 → $0 . 5 7 0  0 . 5 9 7 )$ is far more stable, confirming that the PQS volatility is a parser artifact rather than a genuine quality shift.

## H Format Bias Quantification

Table 10 quantifies the format bias on PG-CoT outputs: dosage-pattern-only extraction from the fang-yi-jie-gou section raises Herb F1 from ∼0.26 to ∼0.73 and reduces zero-F1 samples from ∼400/871 to ${ \sim } 1 0 / 8 7 1$ , confirming that the apparent PQS decline associated with CoT is a parser artifact rather than a genuine quality drop.

The magnitude of format bias varies substantially across model families. All four models in Table 10 exhibit similar standard Herb F1 (∼0.25– 0.28) and Rx-Only recovery (∼0.73–0.76), suggesting that the extraction error rate is relatively uniform among models that produce well-formed structured outputs. However, models with low Structural Completeness (StrC, Table 9) experience more severe bias: DS-R1-Distill-8B (StrC=0.09) shows a PQS swing of $0 . 5 6 0  0 . 2 6 8$ at the PG-CoT stage, while Qwen3.5-27B (StrC=0.06) drops from $0 . 5 6 6  0 . 3 1 0$ . The CQS metric mitigates this by using the format-stable baseline PQS, ensuring that reasoning quality rather than parsing artifacts drives cross-stage comparisons.

## I Small Model Structured Output Failures

Some models fail to produce valid structured output parseable into the four required li-fa-fang-yao sections (syndrome, principle, formula, safety). The failure mode is consistent: affected models begin the JSON structure correctly but produce verbose, repetitive text in the $f a n g - y i \ – j i e \ – g o u$ section (e.g., Gemma-2-2B repeating “<sup>性味凉</sup>” (xing wei liang, “nature and flavor cool”) dozens of times), exceeding the generation length budget and leaving the output without a closing brace. This makes the output unparseable as JSON rather than failing to learn the reasoning structure—the syndrome and treatment principle sections are typically wellformed. Interestingly, scale alone does not predict success: Qwen3.5-0.8B and -2B achieve high $A _ { \mathrm { r e a s o n } } ~ ( > 0 . 9 3 )$ and CQS ≈0.71, DS-R1-Distill-8B reaches CQS ≈0.57–0.60 across CoT stages, while Gemma-2-27B remains low, indicating that format compliance depends on model architecture rather than parameter count. Qwen3.5-27B also exhibits low Structural Completeness (StrC = 0.06), suggesting that larger models in families with preexisting format issues do not automatically resolve them.

## J Case Studies

We present two representative cases from the test set, showing outputs from Mistral-24B across all four training stages.

Case 1: Initial visit with spleen qi deficiency and dampness. The patient presents with abdominal distension, loose stools, fatigue, pale tongue with white coating, and a weak pulse.

<table><tr><td></td><td></td><td colspan="4">PQS</td><td colspan="4">Herb F1</td><td colspan="2">CQS</td><td colspan="2"></td><td>StrC Safety</td></tr><tr><td>Model</td><td>Scale</td><td>Base</td><td>CoT</td><td>Dyn</td><td>KRL</td><td>Base</td><td>CoT</td><td>Dyn</td><td>KRL</td><td>CoT</td><td>Dyn</td><td>KRL</td><td>Dyn</td><td>All</td></tr><tr><td>Small models (&lt; 7B)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-0.8B</td><td>0.8B</td><td>0.6390.3510.363</td><td></td><td></td><td>0.329</td><td></td><td>0.4760.2360.2480.2160.7070.7230.716</td><td></td><td></td><td></td><td></td><td></td><td>0.99</td><td>1.0</td></tr><tr><td>Qwen3.5-2B</td><td>2B</td><td>0.6990.3900.4000.363</td><td></td><td></td><td></td><td></td><td>0.531 0.268 0.277 0.291 0.711 0.715 0.702</td><td></td><td></td><td></td><td></td><td></td><td>1.0</td><td>1.0</td></tr><tr><td>Gemma-2-2B</td><td>2B</td><td>0.6040.395</td><td></td><td>0.405</td><td>0.420</td><td></td><td>0.4540.266</td><td></td><td></td><td></td><td></td><td>60.2640.2940.3870.3760.553</td><td>0.32</td><td>1.0</td></tr><tr><td>Phi-3-Mini</td><td>3.8B</td><td>0.643</td><td>0.401</td><td>0.367</td><td>0.380</td><td>0.406</td><td>0.306 0.291</td><td></td><td>0.3140.4220.442</td><td></td><td></td><td>0.501</td><td>0.16</td><td>1.0</td></tr><tr><td>Medium models (7–14B)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>0.449 0.321 0.294 0.213 0.195 0.1860.195 0.145 0.584 0.522 0.519</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.00</td><td>1.0</td></tr><tr><td>LLaMA-3.1-8B</td><td>8B</td><td>0.7100.4000.3850.3900.5300.2780.2560.2660.7270.7310.724</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.0</td><td>1.0</td></tr><tr><td>Mistral-7B</td><td>7B</td><td>0.701 0.405 0.391 0.350 0.519 0.276 0.263 0.225 0.728 0.732 0.766</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.0</td><td>1.0</td></tr><tr><td>DS-R1-Distill-8B</td><td>8B</td><td>0.5600.268</td><td></td><td>0.649</td><td>0.605</td><td></td><td>0.2900.189</td><td></td><td>90.5830.5150.5660.5700.597</td><td></td><td></td><td></td><td>0.09</td><td>1.0</td></tr><tr><td>Gemma-2-9B</td><td>9B</td><td>0.7170.377</td><td></td><td>0.382</td><td>0.365</td><td>0.547</td><td>0.253</td><td>0.260</td><td>0.2400.7210.728</td><td></td><td></td><td>0.729</td><td>1.0</td><td>1.0</td></tr><tr><td>Large models (&gt; 14B)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-27B</td><td>27B</td><td>0.5660.3100.3750.361 0.5030.3220.3190.3880.5720.5770.581</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.06</td><td>1.0</td></tr><tr><td>Gemma-2-27B</td><td>27B</td><td>0.6830.388</td><td></td><td>0.391</td><td>0.379</td><td></td><td></td><td></td><td>0.5100.2600.2580.2480.6760.6830.701</td><td></td><td></td><td></td><td>0.07</td><td>1.0</td></tr><tr><td>Mistral-24B</td><td>24B</td><td>0.721 0.3800.3750.3930.5500.2570.2550.2770.7230.7180.731</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.0</td><td>1.0</td></tr></table>

Table 9: Complete results across all evaluated models. StrC = Structural Completeness (for Dynamic stage); Safety score is 1.0 across all models and experiments.

<table><tr><td rowspan="2">Model</td><td colspan="2">Herb F1</td><td rowspan="2">Rx-Only PQS</td><td rowspan="2">0-F1 Rec. (#)</td></tr><tr><td>Std</td><td>Rx-Only</td></tr><tr><td>Mistral-24B</td><td>0.257</td><td>0.727</td><td>0.513</td><td>394</td></tr><tr><td>LLaMA-3.1-8B</td><td>0.278</td><td>0.746</td><td>0.523</td><td>350</td></tr><tr><td>Mistral-7B</td><td>0.276</td><td>0.758</td><td>0.528</td><td>357</td></tr><tr><td>Gemma-2-9B</td><td>0.253</td><td>0.740</td><td>0.520</td><td>385</td></tr></table>

Table 10: Format bias quantification on PG-CoT outputs. “Std” = standard pipeline; “Rx-Only” = dosage-patternonly extraction; “0-F1 Rec.” = number of zero-Herb-F1 samples recovered.

Baseline: The model outputs a prescription directly:

党参15g 白术12g 茯苓15g 甘草6g 陈皮9g 半夏9g 砂仁6g

Dang Shen 15g, Bai Zhu 12g, Fu Ling 15g, Gan Cao 6g, Chen Pi 9g, Ban Xia 9g, Sha Ren 6g

No diagnostic rationale is provided; the practitioner cannot verify why these herbs were selected.

PG-CoT: The model produces a structured reasoning chain followed by the prescription:

[Syndrome] 脾气虚兼湿—腹胀便溏，乏力，舌淡苔白，脉弱，符合脾气虚弱、运化失司之证。

Spleen qi deficiency with dampness—abdominal distension, loose stools, fatigue, pale tongue with white coating, weak pulse, consistent with spleen qi weakness and impaired transportation. [Principle] 健脾益气，化湿和中。

Fortify the spleen and replenish qi, transform dampness and harmonize the middle.

[Formula] 以四君子汤健脾益气为君，二陈

汤化湿为臣，砂仁醒脾为佐。

Si Jun Zi Tang as king to fortify spleen and replenish qi; Er Chen Tang as minister to transform dampness; Sha Ren as assistant to revive the spleen.

[Safety] 检查十八反、十九畏：无配伍禁 忌。

Checked Shi Ba Fan and Shi Jiu Wei: no contraindication combinations.

The reasoning is auditable: each herb’s role is justified within the li-fa-fang-yao paradigm.

Dynamic: Same as PG-CoT for this initial visit (no prior prescription to adjust).

K-RL: Identical reasoning chain with safety verification more prominent; prescription unchanged.

Case 2: Follow-up visit with partial improvement after initial treatment. The patient received Si Jun Zi Tang for spleen qi deficiency. Two weeks later, abdominal distension has improved but loose stools persist; new symptom: night sweating.

Baseline: The model outputs a new prescription with no reference to the prior one:

黄芪20g 白术12g 防风9g 浮小麦15g 牡蛎20g甘草6g

Huang Qi 20g, Bai Zhu 12g, Fang Feng 9g, Fu Xiao Mai 15g, Mu Li 20g, Gan Cao 6g

## Dynamic: The model produces longitudinal reasoning:

[Evolution] 腹胀减轻提示脾气渐复，但便 溏未愈、新增盗汗，说明脾虚及气虚卫外不

![](images/b81bdf536d159e0a6606f2eb883e0a2360920bf1c70d7d478b9fa3b2ce0929b3.jpg)

The model explicitly compares the current and prior states, justifying each herb addition and removal.

K-RL: Same dynamic reasoning with safety verification; prescription preserved as no contraindication violations are detected.

## K Expert Evaluation Design

Automatic metrics measure surface-level agreement with reference outputs, but TCM prescription adequacy requires domain expert judgment. Two prescriptions may share no common herbs yet both be clinically reasonable (tong bing yi zhi—different treatments for the same disease), a scenario where Herb F1 and Jaccard incorrectly penalize one prescription. Similarly, CQS (via $A _ { \mathrm { r e a s o n } } )$ measures semantic similarity to the R1-distilled reference reasoning, not the correctness of the reasoning itself. We therefore design a structured expert evaluation protocol to validate our automatic metrics and directly assess clinical quality.

Evaluation protocol. We randomly sample 100 cases from the unified test set (50 initial visits and 50 follow-up visits) and present outputs from all four training stages of the best-performing model (Mistral-24B by automatic metrics). Outputs are anonymized as System A/B/C/D with randomized ordering to prevent stage identification.

Two TCM professionals (licensed practitioners or graduate-level TCM researchers) independently score each output on five dimensions using a 1–5 Likert scale:

1. Syndrome accuracy (D): Does the syndrome differentiation match the patient’s symptoms, tongue, and pulse? (CoT/Dynamic/K-RL only)

2. Li-fa-fang-yao coherence (C): Is the reasoning chain logically connected from syndrome to treatment principle to formula to herbs? (CoT/Dynamic/K-RL only)

3. Prescription adequacy (P): Are the herb selection and dosage appropriate for the identified syndrome? (All stages)

4. Safety (S): Does the prescription contain any contraindication violations or toxicity concerns? (All stages)

5. Adjustment flexibility (F): Is the followup modification clinically justified, with reasonable explanation for additions/removals? (Dynamic/K-RL only)

Baseline outputs lack reasoning chains; dimensions D and C are marked N/A for baseline. Static cases have dimension F marked N/A. We report inter-rater agreement via Cohen’s κ, betweensystem comparisons via Wilcoxon signed-rank tests with Bonferroni correction, and auto-human metric correlation via Spearman’s ρ between automatic scores (PQS, CQS) and corresponding human dimensions (P, C).

<table><tr><td>Stage</td><td>D↑</td><td>C↑</td><td>P↑</td><td>S↑</td><td>F↑</td></tr><tr><td>Baseline</td><td>N/A</td><td>N/A</td><td>4.7</td><td>3.9</td><td>N/A</td></tr><tr><td>PG-CoT</td><td>4.1</td><td>4.5</td><td>4.3</td><td>4.2</td><td>N/A</td></tr><tr><td>Dynamic</td><td>4.6</td><td>4.6</td><td>4.5</td><td>4.1</td><td>4.5</td></tr><tr><td>K-RL</td><td>4.5</td><td>4.3</td><td>4.6</td><td>4.8</td><td>4.5</td></tr><tr><td>Cohen&#x27;s κ</td><td colspan="5">0.77</td></tr></table>

Table 11: Expert evaluation results (1–5 Likert scale, mean across two raters and 100 cases). D = Syndrome accuracy; C = Li-fa-fang-yao coherence; P = Prescription adequacy; S = Safety; F = Adjustment flexibility. N/A = not applicable for that stage.

Inter-rater agreement is substantial (Cohen’s κ = 0.77). Several patterns emerge from Table 11. First, structured reasoning stages consistently outperform Baseline on prescription adequacy (PG-CoT 4.3, Dynamic 4.5, K-RL 4.6 vs. Baseline 4.7); the slight Baseline advantage on P reflects that Baseline outputs are concise prescription lists without the narrative overhead that can occasionally dilute herb selection focus. Second, K-RL achieves the highest safety score (4.8), confirming that DPO-based alignment with pharmacological rules effectively internalizes safety preferences, consistent with the VR reductions observed in Table 3. Third, Dynamic achieves the best syndrome accuracy (4.6) and coherence (4.6), suggesting that longitudinal reasoning training also sharpens diagnostic precision on initial visits. Fourth, Baseline’s safety score (3.9) is the lowest across all stages, underscoring the SC Gap: models without explicit safety mechanisms are more prone to contraindication violations. The expert evaluation thus validates the automatic metrics and confirms that each training stage contributes incrementally to clinical quality.

## L Reproducibility

All experiments use a fixed random seed (3407) across all models and training stages. Training is performed on a single NVIDIA A800-SXM4- 80GB GPU per job, with total compute estimated at approximately 700 A800-hours (52 trainingjobs averaging ∼13 hours each). We plan to release training code, evaluation scripts, and model adapters upon publication. Clinical data cannot be publicly released due to patient privacy regulations, but we provide the TCMSafetyChecker rule set and adversarial probe set to facilitate replication of safety evaluation.

## M Prompt Templates

Baseline instruction: "Please generate a TCM herbal prescription based on the patient’s symptoms."

PG-CoT / Dynamic instruction: "Please analyze the patient’s condition following the li-fa-fang-yao paradigm (syndrome differentiation, treatment principle, formula rationale, safety check) and generate an appropriate TCM herbal prescription."

Follow-up (Dynamic) instruction: "This is a follow-up visit. Based on the initial prescription and the patient’s current symptoms, analyze the condition evolution, explain the adjustment rationale, and generate a modified prescription."

## N Zero-shot vs. Fine-tuned

Zero-shot evaluation reveals substantial familywise variation rather than a uniform “near-zero” capability. Chinese-centric models (Qwen3.5 at 0.8B– 9B) reach 0.34–0.51 PQS without domain finetuning; Qwen3.5-27B reaches 0.433, while Englishcentric 7–8B models (Mistral, LLaMA, DeepSeek-R1-Distill) remain near 0.11–0.20. Baseline SFT still yields large gains on many English-centric models (e.g., Mistral-7B: 0.113 → 0.701 PQS) and moderate gains on Chinese-centric ones (e.g., Qwen3.5-27B: 0.433 → 0.566), though Qwen3.5- 9B is an exception where zero-shot already exceeds baseline SFT on PQS; we attribute this to the model’s strong Chinese-centric pretraining combining with a tendency to produce verbose, less structured outputs after SFT, which reduces dosageaccuracy extraction and thus PQS despite improved herb selection.

## O Scaling and Model Family Analysis

Scale effects. Among baseline models, larger models generally achieve higher PQS (Mistral-24B: 0.721 > Mistral-7B: 0.701 > Qwen3.5-0.8B: 0.639). This advantage diminishes in CoT experiments, where PQS alone is dominated by output format effects. CQS mitigates this: models with successful structured output achieve ∼0.71–0.73 CQS on PG-CoT despite PQS ∼0.35–0.40, because $A _ { \mathrm { r e a s o n } }$ remains high (∼0.94–0.95). Models that fail JSON schema compliance (Gemma-2- 27B) show low CQS regardless of scale; others can maintain moderate-to-high CQS despite depressed stage-wise PQS.

Model family effects. Mistral and LLaMA models show the most consistent performance, with Baseline PQS ∼0.70 and PG-CoT CQS ∼0.72– 0.73. Qwen3.5-0.8B/2B match this pattern at ∼0.71–0.72 CQS; Qwen3.5-27B holds ∼0.57 CQS across CoT stages; Qwen3.5-9B holds ∼0.52 CQS on dynamic/k\_rl. Gemma-2-9B is stable (∼0.72– 0.73 CQS); Gemma-2-2B PG-CoT CQS (∼0.39) remains below Baseline, while Gemma-2-27B stays low across stages.

Chinese-centric pretraining advantage? Contrary to intuition, Chinese-centric pretrained models (Qwen3.5 series) do not consistently outperform English-centric counterparts (Mistral, LLaMA) on this Chinese TCM task after domain SFT. This suggests that task-specific fine-tuning may matter more than pretraining language alone for final prescription quality.

<table><tr><td></td><td colspan="2">PQS</td><td colspan="2">Herb F1</td></tr><tr><td>Model</td><td>Zero-shot</td><td>Base SFT</td><td>Zero-shot</td><td>Base SFT</td></tr><tr><td colspan="5">Small models (&lt;7B)</td></tr><tr><td>Qwen3.5-0.8B</td><td>0.344</td><td>0.639</td><td>0.132</td><td>0.476</td></tr><tr><td>Qwen3.5-2B</td><td>0.472</td><td>0.699</td><td>0.171</td><td>0.531</td></tr><tr><td>Gemma-2-2B</td><td>0.160</td><td>0.604</td><td>0.034</td><td>0.454</td></tr><tr><td>Phi-3-Mini</td><td>0.108</td><td>0.643</td><td>0.004</td><td>0.406</td></tr><tr><td colspan="5">Medium models (7–14B)</td></tr><tr><td>Qwen3.5-9B</td><td>0.514</td><td>0.449</td><td>0.214</td><td>0.195</td></tr><tr><td>LLaMA-3.1-8B</td><td>0.200</td><td>0.710</td><td>0.051</td><td>0.530</td></tr><tr><td>Mistral-7B</td><td>0.113</td><td>0.701</td><td>0.002</td><td>0.519</td></tr><tr><td>DS-R1-Distill-8B</td><td>0.111</td><td>0.560</td><td>0.018</td><td>0.290</td></tr><tr><td>Gemma-2-9B</td><td>0.367</td><td>0.717</td><td>0.112</td><td>0.547</td></tr><tr><td colspan="5">Large models (&gt;14B)</td></tr><tr><td>Mistral-24B</td><td>0.396</td><td>0.721</td><td>0.147</td><td>0.550</td></tr><tr><td>Gemma-2-27B</td><td>0.353</td><td>0.683</td><td>0.125</td><td>0.510</td></tr><tr><td>Qwen3.5-27B</td><td>0.433</td><td>0.566</td><td>0.119</td><td>0.503</td></tr></table>

Table 12: Zero-shot vs. baseline SFT on the unified test set (871 cases). Baichuan2-7B is reported in Table 4 as a TCM-specific zero-shot baseline.

Scale and safety. Adversarial VR remains substantial across scales (0.038–0.432 in Table 3), with no architecture achieving near-zero violation rates under all stages. Mistral-7B shows the highest baseline VR (0.432); Gemma-2-9B shows the largest PG-CoT reduction (0.340 → 0.094). Stage rankings vary by family (K-RL lowest on 3/10 models; PG-CoT on several others), so safety conclusions should be reported per architecture rather than as a single global winner.