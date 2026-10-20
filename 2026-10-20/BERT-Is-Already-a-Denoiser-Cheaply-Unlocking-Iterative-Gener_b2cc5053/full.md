# BERT Is Already a Denoiser: Cheaply Unlocking Iterative Generation in Masked Language Models

Julio C. Amador Díaz López<sup>\*</sup>

October 20, 2026

## Abstract

Masked language models (MLMs) such as BERT are conventionally treated as representation models rather than generators, despite being pretrained to reconstruct corrupted text. We ask whether this pretrained denoising capability can be converted directly into iterative conditional generation without explicitly training a diffusion model. We construct a simple decoder that repeatedly applies a pretrained MLM to masked target positions and progressively commits its highest-confidence predictions according to an external cosine schedule. The model receives no diffusion timestep, learned noise schedule, or reverse-diffusion objective.

Frozen BERT already exhibits substantial denoising capability, reconstructing 61.0%, 37.5%, and 24.4% of masked tokens at 15%, 45%, and 60% corruption, respectively, although generation deteriorates sharply as visible context disappears. This capability alone does not yield task-directed generation: frozen BERT reaches only 11.0% on ARC-Easy and 4.2% on MMLU. Ordinary MLM adaptation, however, rapidly converts the same architecture into a usable generator. Full adaptation reaches 28.2% and 24.6%, respectively, while parameter-efficient adaptation substantially reduces unmatched outputs. On open-ended conditional generation, full MLM adaptation reaches BLEU/ROUGE-L of 0.163/0.583 on QQP and 0.107/0.327 on Quasar-T, placing it in the same broad performance range as published BERT models trained through diffusion without explicit diffusion training. Finally, applying the same generative interface to stronger masked representations substantially improves performance: RoBERTa-large reaches 44.2% on ARC-Easy and 33.8% on MMLU.

These results separate three components of masked iterative generation: pretraining supplies the representation and denoising operator, task adaptation determines what the model generates, and an iterative commitment rule supplies the decoding process. Explicit diffusion training is therefore not necessary to obtain effective iterative conditional generation from pretrained MLMs in the settings studied here.

## 1 Introduction

Masked language models (MLMs) such as BERT [4] are typically treated as representation models rather than generators. Their pretraining objective reconstructs tokens removed from otherwise observed text, producing bidirectional representations that have proved effective across classification, retrieval, and structured prediction tasks[6]. In contrast, modern text generation is dominated by autoregressive models and, more recently, by diffusion-based approaches that formulate generation as iterative denoising.

The distinction is less clear than this division of labour suggests. A MLM is already trained to map corrupted language back toward clean language. It is then reasonable to assume that its pretrained denoising operator might already contain enough of the machinery that generation requires. If the model is repeatedly applied while its most confident predictions are progressively inserted into the context, maskedtoken prediction itself defines an iterative generation procedure.

This connection has motivated several approaches to generative masked language modelling. BERT has been interpreted as a Markov random field from which text can be sampled iteratively [17], while conditional masked language models have used iterative refinement for non-autoregressive generation [7]. More recently, discrete diffusion models have formalized iterative text denoising as a probabilistic generative process. Particularly relevant to our question, DiffusionBERT initializes such a process from pretrained BERT, explicitly exploiting the shared denoising structure of masked-language-model pretraining and discrete diffusion [8]. This raises a more basic question: how much additional generative machinery is required once a pretrained masked language model already exists?

We investigate this question by deliberately removing the diffusion machinery. Starting from ordinary pretrained BERTfamily MLMs, we introduce no forward diffusion process, reverse diffusion parameterization, timestep embedding, or learned noise schedule. Instead, generation consists of repeated MLM evaluation followed by a simple external commitment schedule. At each iteration, the model predicts all unresolved target positions, the highest-confidence predictions are committed, and the remaining positions stay masked for the next pass. Once committed, tokens are never remasked. The result is a monotonic iterative decoder built directly around the pretrained MLM.

This construction lets us separate three components that are often bundled together in generative masked models: the representation and denoising operator learned during MLM pretraining, the task adaptation that aligns them with the desired output, and the decoding procedure that progressively turns masked predictions into a completed sequence. We study these components through a sequence of experiments spanning frozen text reconstruction, multiple-choice generation, paraphrase generation, question generation, and different pretrained BERT-family representations.

We first ask whether useful iterative denoising is already present before any generative adaptation. On corrupted natural text, frozen bert-base-uncased reconstructs 61.0% of masked tokens at 15% masking, 37.5% at 45%, and 24.4% even when 60% of the evaluated sequence is masked. Distributional quality remains high at low corruption but deteriorates sharply as visible context disappears, with repetition emerging at very high masking ratios. These results establish both the capability and its boundary: pretrained BERT is already an effective conditional denoiser, but it does not become a useful unconditional generator simply by masking the entire sequence.

We then test whether this denoising capability directly yields task-conditioned generation. We find that it does not; when the question and answer choices are visible but the answer itself must be generated through a masked target region, frozen BERT reaches only 11.0% accuracy on ARC-Easy and 4.2% on MMLU with most outputs failing to resolve to any candidate answer. However, ordinary answer-only MLM adaptation changes this behavior rapidly. A parameterefficient BitFit condition raises ARC-Easy accuracy to 25.2% while reducing unmatched generations from 71.4% to 12.8%; full fine-tuning reaches 28.2% on ARC-Easy and 24.6% on MMLU. In contrast, requiring the model to generate an intermediate rationale increases the generative burden without improving final-answer performance.

The same mechanism extends to open-ended conditional generation. On QQP, full MLM adaptation reaches BLEU = 0.163 and ROUGE-L = 0.583; on Quasar-T, it reaches BLEU = 0.107 and ROUGE-L = 0.327. These results place an ordinarily adapted MLM in the same broad performance range as a published explicitly trained masked-diffusion baseline on these tasks, despite requiring neither a diffusion objective nor diffusion-time conditioning [8]. The comparison is not a controlled head-to-head evaluation because the training and inference procedures differ. Rather, it provides evidence for the narrower claim tested here: explicit diffusion training is not required to obtain competitive iterative conditional generation from a pretrained MLM in these settings.

Finally, we ask what determines the competence of the resulting generator. Frozen pseudo-log-likelihood controls reveal no large hidden multiple-choice capability in BERT-base that is merely inaccessible to the iterative decoder. Instead, replacing the underlying masked representation changes generative performance substantially. BERT-large provides only a modest gain over BERT-base, whereas RoBERTa-large reaches 44.2% on ARC-Easy and 33.8% on MMLU under the same generative framework. Because BERT-large and RoBERTa-large are similar in parameter scale, this comparison indicates that generative competence depends strongly on the quality of the underlying masked representation rather than model size alone.

Taken together, the results support a simple account of how pretrained MLMs become generators. MLM pretraining supplies both the underlying representation and a conditional denoising operator. Task-specific MLM adaptation aligns these with the desired output, while an iterative commitment rule converts repeated applications of the denoising operator into a decoding process. The quality of the pretrained representation, in turn, constrains the competence that the resulting generator can express. Our contributions are therefore fourfold:

• We show that substantial iterative denoising is already present in frozen BERT and characterize how it degrades as visible context is progressively removed.

• We demonstrate that denoising capability and taskdirected generation are empirically separable: frozen BERT is a useful denoiser but a poor task-conditioned generator, while ordinary MLM adaptation cheaply converts the same architecture into a usable generative interface.

• We show that standard MLM adaptation and simple iterative decoding can reach the performance range of an explicitly trained masked-diffusion model on conditional generation, without a diffusion objective, timestep conditioning, or diffusion sampling process.

• We show that the competence of the resulting generator tracks the underlying masked representation: stronger pretrained representations produce substantially stronger generators under essentially the same decoding procedure.

These findings narrow the role that must be assigned to explicit diffusion training in masked text generation. They do not imply that diffusion provides no additional benefits: probabilistic modelling, diversity, unconditional generation, and alternative sampling behaviour remain distinct questions. Rather, they show that the basic ingredients required for effective iterative conditional generation can already be latent in a pretrained masked language model. In this regime, generation can be obtained by exposing and adapting an existing denoising capability rather than by learning the denoising process anew.

## 2 Related Work

Our work sits at the intersection of three lines of research: masked language models as generative models, discrete diffusion for language, and recent attempts to simplify masked diffusion toward ordinary masked modelling.

## 2.1 BERT and Masked Language Models as Generative Models

BERT [4] was introduced primarily as a bidirectional representation model trained with a masked language modelling objective. Although it is not autoregressive, its conditional token distributions can still define a generative procedure. Wang and Cho [17] formalized BERT as a Markov random field and showed that sentences can be sampled by repeatedly masking and resampling individual tokens. Their result established that masked language models are not intrinsically restricted to discriminative use.

Subsequent work has developed iterative generation methods based on masked prediction. Mask-Predict [7] introduced confidence-guided iterative refinement for conditional sequence generation, while SUNDAE [16] investigated text generation through repeated denoising. These approaches demonstrate that masked prediction can support non-autoregressive generation, although their training and decoding procedures differ from those of pretrained BERT.

Our work instead examines how much of this capability is already present in an existing MLM and how much additional adaptation is required to exploit it.

More recently, Clavié et al. [3] introduced ModernBERT-Large-Instruct, a 0.4B-parameter encoder model that uses its MLM head for instruction-tuned generative classification. The model demonstrates strong zero-shot performance on classification and knowledge-based tasks, illustrating that pretrained masked representations can support task-directed outputs without a conventional classification head. Their approach focuses on single-token answers, whereas we investigate iterative multi-token generation.

## 2.2 Discrete Diffusion Models for Language

Diffusion models provide a probabilistic framework in which generation is defined through reversal of a progressively destructive forward process. While early diffusion models were developed primarily for continuous data, discrete variants extended the framework to categorical state spaces [1]. For language, discrete diffusion is attractive because corruption can be represented directly through token replacement or absorbing mask states.

DiffusionBERT [8] is the most direct precursor to our work. It combines pretrained BERT with an absorbing-state discrete diffusion process, motivated by the shared denoising structure of MLM pretraining and diffusion. The model learns a reverse diffusion process and investigates explicit timestep designs together with a token-frequency-dependent spindle noise schedule. For conditional generation, DiffusionBERT performs partial denoising conditioned on visible source text. Its results demonstrate that pretrained BERT provides an effective initialization for discrete diffusion and substantially accelerates convergence.

Our work starts from the same empirical observation; i.e., that BERT is already a pretrained denoiser, but tests a different hypothesis. Instead of asking how best to adapt BERT into a diffusion model, we ask whether a diffusion model is necessary for the conditional-generation setting at all. We therefore remove the explicit forward process, reverse diffusion parameterization, timestep conditioning, and noise schedule, retaining only repeated MLM prediction and an external commitment schedule.

Other discrete diffusion approaches pursue more general generative modelling objectives. Score-Entropy Discrete Diffusion (SEDD), for example, learns discrete score ratios using a score-entropy objective and achieves competitive language modelling performance relative to autoregressive baselines [12]. These approaches demonstrate that discrete diffusion can be a competitive generative paradigm in its own right. Our aim is narrower: we test whether pretrained masked language models already contain enough of the relevant denoising structure that a substantially simpler adaptation procedure suffices for conditional generation.

## 2.3 Masked Diffusion and Classical MLM Objectives

Recent work has further narrowed the conceptual gap between masked diffusion and classical masked language modelling. Sahoo et al.[14] derive a simplified masked diffusion objective whose training loss can be expressed as a mixture of classical MLM losses. Their Masked Diffusion Language Model (MDLM) achieves strong language-modelling performance and shows that encoder-only masked models can support efficient generative samplers.

Related theoretical work argues that masked diffusion models can be time-agnostic: under suitable formulations, both training and sampling can be expressed without an explicit diffusion-time variable, making masked diffusion closely related to ordinary masked models [18].

These results are highly complementary to ours. They simplify the training objective of newly trained masked diffusion models and show that diffusion-specific structure can collapse toward masked modelling. We instead begin from models that have already undergone large-scale MLM pretraining and ask whether that existing denoising representation can be reused directly. The distinction is therefore between training a generative masked model using an MLM-like objective and retrofitting an already pretrained MLM into a conditional generator. Our experiments target the latter.

## 2.4 Positioning of the Present Work

The literature therefore establishes several pieces of the picture independently. BERT can be sampled iteratively [17]; confidence-guided masked refinement can support conditional generation [7]; dedicated denoising autoencoders can generate text [16]; and discrete diffusion can combine effectively with pretrained masked models [8]. More recent work further shows that masked diffusion objectives can simplify toward mixtures of MLM losses and may not require an explicit time variable [14, 18].

Our contribution is to connect these observations through a direct empirical minimality test. We characterize the denoising capability already present in frozen BERT, show that this capability is distinct from task-directed generation, demonstrate that standard MLM adaptation plus iterative decoding is sufficient to acquire useful conditional generation, and show that the quality of the resulting generator tracks the competence of the underlying masked representation. In this sense, the question addressed here is not whether masked or diffusion models can generate language, but how much additional generative machinery is actually required once a pretrained MLM already exists.

## 3 Methods

We investigate whether pretrained masked language models (MLMs) can perform iterative conditional generation through ordinary MLM adaptation and a simple decoding procedure. Our approach combines a pretrained MLM, which supplies conditional token predictions, with an iterative decoder that progressively fills masked positions. Task-specific MLM fine-tuning is used where adaptation is required.

We first characterize the denoising behavior of frozen BERT on corrupted natural text, then examine its adaptation to multiple-choice and open-ended conditional generation. Finally, we investigate how generation performance depends on the underlying pretrained representation.

## 3.1 Iterative Masked Decoding

Given a sequence $x ^ { ( 0 ) }$ containing a set of masked positions $\mathcal { M } ^ { ( 0 ) }$ , the MLM predicts a vocabulary distribution at each masked position. At iteration t, the predicted token $\hat { x } _ { i } ^ { ( t ) }$ and its confidence $c _ { i } ^ { ( t ) }$ are:

$$
\hat { x } _ { i } ^ { ( t ) } = \underset { v \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \theta } \big ( v \mid x ^ { ( t - 1 ) } \big ) , \qquad c _ { i } ^ { ( t ) } = \underset { v \in \mathcal { V } } { \operatorname* { m a x } } p _ { \theta } \big ( v \mid x ^ { ( t - 1 ) } \big ) .\tag{1}
$$

for every unresolved position $i \in \mathcal { M } ^ { ( t - 1 ) }$

Rather than accepting all predictions simultaneously, the decoder commits the $k _ { t }$ positions with the highest confidence. Their predicted tokens are inserted into the sequence, and the remaining positions stay masked:

$$
\mathcal { M } ^ { ( t ) } = \mathcal { M } ^ { ( t - 1 ) } \setminus \mathcal { C } ^ { ( t ) } ,\tag{2}
$$

where $\mathcal { C } ^ { ( t ) }$ is the set of newly committed positions.

The MLM is then evaluated again on the partially completed sequence. Committed tokens remain fixed, while unresolved positions are reconsidered using the additional context. Generation terminates when all masked positions have been resolved.

Unlike autoregressive decoding, the order of generation is not constrained by sequence position. Confidence-based commitment can resolve positions anywhere in the sequence, allowing subsequent predictions to use context introduced on either side.

Commitment schedule. Let $N = | \mathcal { M } ^ { ( 0 ) } |$ be the initial number of masked positions and T the requested number of decoding passes, with $T ^ { \prime } = \operatorname* { m a x } ( 1$ , min $( T , N ) )$ . The number of positions remaining masked after iteration t follows a cosine schedule:

$$
r _ { t } = \mathrm { r o u n d } \left[ N \cos \left( { \frac { \pi t } { 2 T ^ { \prime } } } \right) \right] , \qquad t = 0 , \dots , T ^ { \prime } ,\tag{3}
$$

with $r _ { 0 } ~ = ~ N , ~ r _ { T ^ { \prime } } ~ = ~ 0$ , and monotonicity enforced by $r _ { t } \gets \operatorname* { m i n } ( r _ { t } , r _ { t - 1 } )$ . The number of positions committed at iteration t is therefore $k _ { t } = r _ { t - 1 } - r _ { t }$

The schedule commits relatively few predictions initially and progressively more as context accumulates. Iterations with $k _ { t } = 0$ are skipped.

Unless otherwise stated, token selection is greedy (argmax), corresponding to the zero-temperature limit. The first experiment additionally evaluates alternative commitment orders and stochastic token sampling as decoding controls.

## 3.2 Reconstruction and Conditional Generation

The same decoder is used in two settings:

Full-sequence reconstruction. For denoising experiments, selected positions in a natural-language sequence are replaced by the model’s mask token:

$$
x _ { i } ^ { ( 0 ) } = \left\{ \begin{array} { l l } { [ \mathrm { M A S K } ] , } & { i \in \mathcal { M } ^ { ( 0 ) } , } \\ { x _ { i } , } & { i \notin \mathcal { M } ^ { ( 0 ) } . } \end{array} \right.\tag{4}
$$

The decoder reconstructs these positions while leaving unmasked tokens unchanged. The primary configuration uses 16 requested passes, confidence-based commitment, and greedy token selection, without vocabulary restrictions.

Conditional target-region generation. For taskconditioned generation, the input consists of visible conditioning context followed by a masked target region:

$$
x ^ { ( 0 ) } = [ \mathrm { c o n t e x t } ; \underbrace { [ \mathrm { M A S K } ] , \dots , [ \mathrm { M A S K } ] } _ { N \mathrm { t a r g e t s l o t s } } ] .\tag{5}
$$

Only target positions are progressively resolved. The conditioning context remains fixed, while committed target tokens become visible on subsequent passes. The commitment schedule is calculated independently for each example, allowing different target lengths within a batch.

Multiple-choice generation uses four requested passes in the primary configuration; open-ended generation uses eight.

## 3.3 Task-Specific Decoding

Multiple-choice generation. For ARC-Easy and MMLU, the question and candidate answers are provided as visible context. The primary prompt format is:

Question: <question>   
Choices:   
A. <choice0>   
B. <choice1>   
C. <choice2>   
D. <choice3>   
Answer: [MASK] ... [MASK].

Inputs use the native special and mask tokens of each backbone, with a maximum length of 512 tokens. When necessary, question tokens are truncated before the choices and answer cue.

The primary configuration uses whole-word decoding and an adaptive answer-length budget. If the reference answer contains w whitespace-delimited words, the target region contains $N = w + 2$ mask slots. The additional slots provide a small generation buffer. Because this construction uses reference-answer length, we also evaluate fixed budgets of 4, 8, and 16 slots that do not depend on the reference answer.

Before token selection, model-specific special tokens and BERT’s [unused<sub>\*</sub>] tokens are excluded. Under wholeword decoding, BERT WordPiece continuation tokens beginning with ## are also excluded. This restriction is not applied to RoBERTa’s BPE vocabulary. Excluded logits are set to −∞ before softmax.

Answer matching. Generated tokens are decoded to text and mapped to the visible candidate answers. After lowercasing and removing non-alphanumeric characters, exact matches are accepted directly. Otherwise, candidates are scored using the larger of token-level F1 and character-level sequence similarity. A candidate is accepted only when its score is at least 0.5 and uniquely exceeds the alternatives.

Empty, ambiguous, and unmatched outputs are counted as incorrect. We report both answer accuracy and the unmatched rate,

$$
U = \frac { 1 } { N _ { \mathrm { e v a l } } } \sum _ { j = 1 } ^ { N _ { \mathrm { e v a l } } } { \bf 1 } \left[ \hat { y } _ { j } \nleftrightarrow A _ { j } \right] ,\tag{6}
$$

where $A _ { j }$ is the candidate set for example j. No additional language model or learned matching component is used.

Open-ended generation. QQP paraphrase generation and Quasar-T question generation use task-specific conditioning prompts followed by a fully masked target region. Target length is fixed using the training target-length distribution. Unlike multiple-choice generation, subword tokens are permitted. Outputs are decoded using the native tokenizer, with special tokens removed, and evaluated directly against reference sequences.

## 3.4 Models, Datasets, and Experimental Design

Our primary model is bert-base-uncased [4]. We subsequently evaluate BERT-large, RoBERTa-large [11], and a RoBERTa-large backbone adapted to RACE [10].

We use four task families: WikiText [13] for text reconstruction; ARC-Easy [2] and MMLU [9] for multiple-choice generation; Quora Question Pairs (QQP) for paraphrase generation; and Quasar-T [5] for question generation.

Unless otherwise specified, ARC-Easy and MMLU evaluations use fixed seed-42 subsets of 500 examples, reused across the main model and adaptation comparisons.

## 3.5 Experimental Conditions and Evaluation

Frozen denoising (A). We evaluate frozen BERT-base on 256 WikiText blocks of 128 tokens, using nested masking ratios of 15, 30, 45, 60, 75, 90, and 100%. The primary condition uses subword masking and 16 requested decoding passes.

We additionally compare confidence-based commitment with random and left-to-right commitment, holding the cosine schedule fixed. A one-shot condition resolves all masked positions in a single forward pass. Supplementary ablations vary decoding passes, sampling temperature, and subword versus whole-word masking.

For stochastic decoding, tokens are sampled from temperature-scaled vocabulary distributions,

$$
\hat { x } _ { i } ^ { ( t ) } \sim \mathrm { C a t e g o r i c a l } \left[ \mathrm { s o f t m a x } \left( \frac { z _ { i } ^ { ( t ) } } { \tau } \right) \right] , \qquad \tau > 0 ,\tag{7}
$$

where $z _ { i } ^ { ( t ) }$ denotes the MLM logits. The probability of the sampled token is used for confidence ranking.

We report masked-token reconstruction accuracy, GPT-2- large negative log-likelihood (NLL), MAUVE, and repeated four-gram rate (rep-4). These measure reconstruction, external language-model predictability, distributional similarity, and local repetition, respectively.

Frozen conditional generation (B). We apply the targetregion decoder to frozen BERT-base on ARC-Easy and MMLU. The primary condition uses four-pass confidence decoding, adaptive whole-word answer budgets, and the prompt format described above. Fixed-budget conditions test sensitivity to reference-answer length.

MLM adaptation (C). We adapt BERT-base using maskedtoken prediction over the answer region while keeping questions and candidate choices visible. The objective is ordinary masked-token cross-entropy,

$$
\mathcal { L } _ { \mathrm { M L M } } = - \sum _ { i \in \mathcal { M } } \log p _ { \theta } \left( x _ { i } \mid x _ { \backslash \mathcal { M } } \right) ,\tag{8}
$$

where M denotes masked target positions. Training includes partially observed answer configurations, including high target corruption.

We compare BitFit, which adapts bias terms and the MLM output components used by the implementation; Top-2 and Top-4, which unfreeze the highest two or four transformer layers together with the output components; and Full, which updates the complete model. The principal comparison uses approximately 100k training examples.

Conditional text generation (D–E). We adapt BERT-base to QQP paraphrase generation and Quasar-T question generation using masked-target prediction. We evaluate frozen BERT, BitFit adaptation, and full fine-tuning, reporting eightpass decoding as the principal configuration. Generation quality is measured using BLEU, ROUGE-L, and BERTScore.

We also report published conditional-generation results from DiffusionBERT [8] as a literature reference. These results were obtained under different training and inference configurations and do not constitute a controlled head-tohead comparison.

Discriminative pseudo-likelihood (F). We evaluate frozen BERT-base using normalized pseudo-log-likelihood (PLL) scoring [15]. Pairwise and full-context formulations score candidate answers without free generation, providing a discriminative control on the same ARC-Easy and MMLU evaluation subsets.

Representation-quality experiments (G–H). We repeat the generative interface with BERT-large, RoBERTa-large, and a RACE-adapted RoBERTa-large backbone. On ARC-Easy (G), we evaluate frozen generation, BitFit, and Full adaptation at 10k, 50k, and 100k examples, together with discriminative scores. For the RACE model, we additionally report its native multiple-choice performance.

Because the RACE-adapted backbone does not retain the standard pretrained MLM readout, we supply the original RoBERTa-large MLM head for iterative generation. Poor frozen generation in this configuration may therefore reflect a mismatch between the adapted backbone and the supplied head.

On MMLU (H), we evaluate the Full 100k checkpoints using the same 500-example, four-pass protocol. The BERTbase checkpoint is reused from experiment C. Table 1 the experiments.

Table 1: Summary of the experimental design.
<table><tr><td>Exp.</td><td>Model(s)</td><td>Data</td><td>Intervention</td><td>Primary purpose</td></tr><tr><td>A</td><td>BERT-base</td><td>WikiText</td><td>Frozen; 15–100% masking</td><td>Pretrained denoising</td></tr><tr><td>B</td><td>BERT-base</td><td>ARC-Easy, MMLU</td><td>Frozen iterative generation</td><td>Generation without adaptation</td></tr><tr><td>C</td><td>BERT-base</td><td>ARC-Easy, MMLU</td><td>BitFit / Top-2 / Top-4 / Full</td><td>Generative adaptation</td></tr><tr><td>D</td><td>BERT-base</td><td>QQP</td><td>MLM FT + iterative decoding</td><td>Paraphrase generation</td></tr><tr><td>E</td><td>BERT-base</td><td>Quasar-T</td><td>MLM FT + iterative decoding</td><td>Question generation</td></tr><tr><td>F</td><td>BERT-base</td><td>ARC-Easy, MMLU</td><td>Frozen PLL scoring</td><td>Discriminative competence</td></tr><tr><td>G</td><td>BERT/RoBERTa variants</td><td>ARC-Easy</td><td>Frozen / BitFit / Full</td><td>Representation quality</td></tr><tr><td>H</td><td>BERT/RoBERTa variants</td><td>MMLU</td><td>Full @100k</td><td>Representation transfer to MMLU</td></tr></table>

## 3.6 Implementation

At inference, the transformer processes the complete partially observed sequence, but vocabulary logits are computed only for masked positions relevant to the current decoding step. This reduces memory consumption without changing the conditional token predictions. All decoding is performed in evaluation mode without parameter updates.

The framework uses ordinary MLM training and an external commitment schedule. No diffusion timestep, noise-level embedding, learned reverse transition, or additional generative model is introduced. Once committed, tokens are never remasked.

## 4 Results

## 4.1 Pretrained BERT Supports Iterative Denoising

Frozen BERT-base reconstructs substantial portions of corrupted natural text using iterative masked decoding. At 15% masking, 61.0% of masked tokens were recovered, with MAUVE = 0.899 (Table 2). Reconstruction accuracy declined progressively as masking increased, reaching 37.5% at 45% masking and 24.4% at 60%.

Distributional quality deteriorated more sharply than token accuracy. MAUVE fell to 0.415 at 45% masking and 0.265 at 60%, while repetition increased rapidly above 60%. At 90– 100% masking, outputs increasingly collapsed into repetitive sequences. Their lower GPT-2 NLL therefore reflects greater predictability rather than improved generation quality.

At 45% masking, confidence-based commitment achieved the highest reconstruction accuracy (37.5%), compared with 34.4% for left-to-right and 33.6% for random commitment (Table 3). Random commitment, however, obtained the highest MAUVE, indicating that reconstruction fidelity and distributional similarity do not necessarily favour the same decoding order.

These results establish that pretrained BERT already provides a useful conditional denoising operator, although its ability to reconstruct text deteriorates substantially when most of the conditioning context is removed.

## 4.2 Task-Conditioned Generation Requires Adaptation

The reconstruction capability of frozen BERT did not translate directly into multiple-choice generation. Accuracy reached only 11.0% on ARC-Easy and 4.2% on MMLU, with unmatched-output rates of 71.4% and 80.8%, respectively.

Ordinary answer-only MLM adaptation substantially improved both measures (Table 4). After approximately 100k training examples, BitFit increased ARC-Easy accuracy to 25.2% and MMLU accuracy to 17.4%, while reducing unmatched outputs to 12.8% and 33.4%. Full adaptation achieved the strongest results, reaching 28.2% and 24.6% accuracy, respectively.

The reduction in unmatched outputs under BitFit is particularly informative: limited parameter adaptation was sufficient to make the pretrained model substantially more effective at expressing task-directed answers through the generative interface.

## 4.3 Conditional Text Generation

The same adaptation procedure extended to open-ended conditional generation. On QQP, Full adaptation reached BLEU = 0.163, ROUGE-L = 0.583, and BERTScore = 0.903. Bit-Fit already achieved ROUGE-L = 0.470 after 5k training examples and 0.507 after 10k (Table 5).

On Quasar-T, Full adaptation reached BLEU = 0.107 and ROUGE-L = 0.327. BitFit performed best at 10k examples and declined with further training (Table 6).

For context, the published DiffusionBERT results report ROUGE-L = 0.585 on QQP and 0.342 on Quasar-T. Our Full models reached 0.583 and 0.327, respectively, although the BLEU comparisons differ across tasks. Because training and decoding protocols are not matched, these values establish similarity in the range of reported task performance, rather than comparative superiority.

The central observation is that standard MLM adaptation and iterative commitment can produce substantial conditional-generation quality without explicit diffusionmodel training.

## 4.4 Generation Performance Depends on the Underlying Representation

We next examined whether the relatively modest multiplechoice accuracy of BERT-base reflected limitations of the generative interface or the underlying representation.

Frozen BERT-base achieved 31.4% ARC-Easy accuracy under pairwise pseudo-log-likelihood (PLL) scoring and 25.6% under full-context PLL. On MMLU, the corresponding scores were 22.6% and 23.8% (Table 7). These results do not reveal a large zero-shot discriminative advantage that the generative procedure consistently fails to recover, although frozen PLL and adapted generation are not directly equivalent evaluations.

Table 2: Frozen BERT-base reconstruction across masking ratios. Results use 16-pass greedy confidence decoding. Lower NLL and rep-4, and higher masked accuracy and MAUVE, are preferable.
<table><tr><td>Mask ratio</td><td>Masked accuracy</td><td>GPT-2 NLL</td><td>MAUVE</td><td>rep-4</td></tr><tr><td>15%</td><td>0.610</td><td>3.75</td><td>0.899</td><td>0.011</td></tr><tr><td>30%</td><td>0.501</td><td>3.83</td><td>0.779</td><td>0.011</td></tr><tr><td>45%</td><td>0.375</td><td>3.92</td><td>0.415</td><td>0.010</td></tr><tr><td>60%</td><td>0.244</td><td>3.98</td><td>0.265</td><td>0.012</td></tr><tr><td>75%</td><td>0.126</td><td>3.80</td><td>0.168</td><td>0.067</td></tr><tr><td>90%</td><td>0.050</td><td>2.67</td><td>0.015</td><td>0.367</td></tr><tr><td>100%</td><td>0.019</td><td>1.51</td><td>0.004</td><td>0.624</td></tr></table>

Table 3: Commitment-order ablation at 45% masking.
<table><tr><td>Strategy</td><td>Accuracy</td><td>MAUVE</td><td>rep-4</td></tr><tr><td>Confidence</td><td>0.375</td><td>0.415</td><td>0.010</td></tr><tr><td>Left-to-right</td><td>0.344</td><td>0.422</td><td>0.008</td></tr><tr><td>Random</td><td>0.336</td><td>0.449</td><td>0.008</td></tr></table>

Stronger pretrained representations yielded stronger generators under the same iterative decoding procedure. On ARC-Easy, Full adaptation reached 30.4% for BERT-large and 44.2% for RoBERTa-large, compared with 28.2% for BERT-base (Table 8).

The contrast between BERT-large and RoBERTa-large is particularly informative. Despite similar parameter counts, RoBERTa-large substantially outperformed BERT-large, suggesting that pretraining and representation quality contribute beyond model scale alone.

The RACE-adapted backbone achieved 49.0% with its native multiple-choice readout but only 0.4% in frozen iterative generation using a supplied MLM head. After Full adaptation, its generative accuracy increased to 41.8%. The frozen result therefore reflects, at least in part, incompatibility between the adapted representation and the supplied generative readout.

The same ordering broadly persisted on MMLU. RoBERTa-large reached 33.8% accuracy and RoBERTa-RACE 32.6%, compared with 24.6% for BERT-base (Table 9).

These findings indicate that the iterative generation procedure does not impose a fixed performance ceiling near the BERT-base results. The competence of the resulting generator depends substantially on the pretrained representation and its compatibility with the MLM readout.

## 4.5 Summary

Across the experiments, frozen BERT demonstrates substantial iterative denoising capability, but task-conditioned generation requires adaptation. Ordinary MLM fine-tuning makes this capability usable for multiple-choice and open-ended generation, while stronger pretrained representations produce stronger generators under the same decoding procedure.

Together, the results support the view that pretrained MLMs already supply the conditional denoising operator needed for iterative generation. Task adaptation aligns this operator with the desired outputs, and confidence-guided commitment provides a simple mechanism for constructing complete sequences.

## 5 Discussion

Our results suggest that iterative conditional generation can be decomposed into three components: a pretrained denoising operator, task-specific adaptation, and a procedure for progressively constructing the output. BERT already provides the first component through masked-language-model pretraining. Ordinary MLM adaptation can supply the taskspecific behaviour, while confidence-guided commitment turns repeated masked predictions into a completed sequence.

The distinction is important because denoising and taskdirected generation are different capabilities. Frozen BERT reconstructs partially corrupted natural text effectively, yet performs poorly when asked to generate multiple-choice answers. The pretrained model has learned to predict missing linguistic content, but additional adaptation is needed to produce the particular outputs required by a task.

## 5.1 From Pretrained Denoising to Generative Behaviour

The reconstruction experiments establish that useful iterative denoising is already present in pretrained BERT. This capability depends strongly on the available context: reconstruction remains effective under moderate corruption but deteriorates as most of the sequence becomes masked. Conditional generation provides a different setting, in which a visible prompt or source sequence can supply information about the masked target.

The adaptation experiments show that this distinction can be exploited using the original MLM objective. Even restricted parameter updates substantially improve multiplechoice generation and reduce unmatched outputs. Full finetuning provides further gains, but useful task-directed behaviour emerges without retraining the complete model.

One interpretation is that adaptation serves two related functions. Interface adaptation makes task solutions expressible through a masked output region, while competence adaptation modifies the representation to improve the solutions available to that interface. The BitFit results are consistent with the first mechanism, whereas the additional gains from full fine-tuning suggest a role for the second. These mechanisms are not separately identified by the present experiments, but they provide a useful framework for interpreting the different adaptation regimes.

Table 4: Multiple-choice generation before and after MLM adaptation. All adapted models use approximately 100k training examples and the same four-pass decoder. Unmatched outputs are counted as incorrect.
<table><tr><td rowspan="2">Adaptation</td><td colspan="2">ARC-Easy</td><td colspan="2">MMLU</td></tr><tr><td>Accuracy ↑</td><td>Unmatched ↓</td><td>Accuracy ↑</td><td>Unmatched ↓</td></tr><tr><td>Frozen</td><td>0.110</td><td>0.714</td><td>0.042</td><td>0.808</td></tr><tr><td>BitFit</td><td>0.252</td><td>0.128</td><td>0.174</td><td>0.334</td></tr><tr><td>Top-2</td><td>0.242</td><td>0.226</td><td>0.176</td><td>0.454</td></tr><tr><td>Top-4</td><td>0.258</td><td>0.122</td><td>0.174</td><td>0.328</td></tr><tr><td>Full</td><td>0.282</td><td>0.088</td><td>0.246</td><td>0.158</td></tr></table>

Table 5: QQP paraphrase generation. DiffusionBERT results are published reference values from He et al. [8], not a controlled reproduction. Training budgets count examples for our models.
<table><tr><td>Model</td><td>Examples</td><td>BLEU↑</td><td>ROUGE-L ↑</td><td>BERTScore ↑</td></tr><tr><td>Frozen</td><td>0</td><td>0.004</td><td>0.101</td><td>0.666</td></tr><tr><td>BitFit</td><td>5k</td><td>0.085</td><td>0.470</td><td>0.870</td></tr><tr><td>BitFit</td><td>10k</td><td>0.107</td><td>0.507</td><td>0.883</td></tr><tr><td>BitFit</td><td>100k</td><td>0.121</td><td>0.533</td><td>0.889</td></tr><tr><td>Full</td><td>100k</td><td>0.163</td><td>0.583</td><td>0.903</td></tr><tr><td>DiffusionBERT (reported)</td><td>一</td><td>0.242</td><td>0.585</td><td>一</td></tr></table>

Under this interpretation, learning a conditional generator does not necessarily require learning a new denoising operator. A pretrained MLM may already supply that operator; adaptation can instead establish the mapping between conditioning information and the desired output.

## 5.2 The Role of the Pretrained Representation

The backbone comparisons indicate that the generative interface does not impose a fixed performance ceiling. BERTlarge provides only modest improvements over BERT-base, whereas RoBERTa-large produces substantially stronger results under the same iterative decoding procedure. Because BERT-large and RoBERTa-large have similar parameter counts, model scale alone does not explain the observed differences.

The discriminative controls provide complementary evidence. Frozen BERT-base exhibits relatively limited multiplechoice competence under pseudo-log-likelihood scoring, while stronger pretrained representations generally support stronger adapted generators. Nevertheless, discriminative scoring and adapted generation measure different capabilities, and their results should not be interpreted as interchangeable.

The RACE-adapted RoBERTa experiment further illustrates the importance of the output interface. Its native multiple-choice readout performs substantially better than frozen iterative generation using a supplied MLM head. Generative adaptation subsequently recovers considerable performance. Because the supplied head was not jointly trained with the adapted backbone, the initial failure cannot be attributed solely to the representation.

Taken together, these findings suggest that generative performance depends on both the information available in the pretrained representation and the ability of the MLM readout to express that information in the required target format. Iterative decoding provides the generation mechanism, but does not independently determine the competence of the resulting model.

## 5.3 Implications for Iterative Generation

The conditional-generation results demonstrate that ordinary MLM adaptation can support paraphrase and question generation at a level broadly comparable to published BERT adapted, masked-diffusion results. The comparison with DiffusionBERT is not controlled, but it establishes that an explicitly trained diffusion process is not a prerequisite for obtaining this range of performance on the evaluated tasks.

Our decoder provides a particularly simple construction. At each iteration, the MLM predicts unresolved positions, and the most confident predictions are committed according to an external schedule. The model receives no diffusion timestep or noise-level conditioning, and committed tokens are never remasked. Generation therefore proceeds through repeated conditional inference with an increasingly complete context.

This construction does not establish that explicit diffusion modelling is redundant. A learned diffusion process may offer advantages in sampling diversity, revision of earlier predictions, output-length modelling, or generation from weakly conditioned inputs. These possibilities are outside the present evaluation. The contribution here is to isolate a simpler route to iterative conditional generation using an existing pretrained denoiser.

This has practical implications for the broader family of masked language models. Many domains already possess pretrained encoders developed from specialized corpora. Where these models have learned useful conditional representations, task-specific MLM adaptation and iterative decoding may provide a route to generative applications without training a new generative backbone from scratch.

The broader architectural implication is that representation learning, task adaptation, and generation need not be treated as a single inseparable training problem. Their contributions can be studied and optimized independently, even though the resulting capabilities remain interdependent.

Table 6: Quasar-T question generation. DiffusionBERT values are published reference results rather than a controlled reproduction.
<table><tr><td>Model</td><td>Examples</td><td>BLEU↑</td><td>ROUGE-L ↑</td><td>BERTScore ↑</td></tr><tr><td>Frozen</td><td>0</td><td>0.002</td><td>0.072</td><td>0.666</td></tr><tr><td>BitFit</td><td>10k</td><td>0.050</td><td>0.233</td><td>0.744</td></tr><tr><td>BitFit</td><td>100k</td><td>0.026</td><td>0.190</td><td>0.741</td></tr><tr><td>Full</td><td>100k</td><td>0.107</td><td>0.327</td><td>0.781</td></tr><tr><td>DiffusionBERT (reported)</td><td>一</td><td>0.097</td><td>0.342</td><td>一</td></tr></table>

Table 7: BERT-base discriminative PLL controls and adapted generative accuracy (%).
<table><tr><td>Method</td><td>ARC-Easy</td><td>MMLU</td></tr><tr><td>Chance (~)</td><td>25.0</td><td>25.0</td></tr><tr><td>Full generation</td><td>28.2</td><td>24.6</td></tr><tr><td>PLL pairwise</td><td>31.4</td><td>22.6</td></tr><tr><td>PLL full-context</td><td>25.6</td><td>23.8</td></tr></table>

## 5.4 Limitations

Several limitations frame these findings. First, the experiments focus on conditional generation. Frozen BERT deteriorates sharply at high masking ratios, and the present results do not establish competitive unconditional generation from a fully masked sequence.

Second, DiffusionBERT comparisons use published values obtained under different training and inference protocols. They indicate the range of achievable conditional-generation quality but cannot establish comparative efficiency or superiority. A controlled comparison would require matched training data, optimization budgets, decoding procedures, and evaluation settings.

Third, the principal multiple-choice evaluations use fixed subsets of 500 examples. Larger evaluations and additional tasks are needed to establish more precise estimates of absolute performance and generalization.

Fourth, the primary multiple-choice decoder uses reference-answer length to determine its output budget. Fixed-length controls remove this dependency, but a generalpurpose generator should determine output length independently. Open-ended generation also relies on a predetermined target-length budget.

Finally, the experiments do not isolate which features of MLM pretraining account for the observed denoising capability, nor do they establish that confidence-based commitment or the cosine schedule is optimal. The present decoding procedure was chosen for simplicity and to expose the generative capabilities of the pretrained models.

## 6 Conclusion

Pretrained masked language models already contain a useful conditional denoising operator. By combining this capability with ordinary MLM adaptation and confidence-guided iterative decoding, we obtain task-conditioned generation without explicitly training a diffusion model.

The experiments also establish an important boundary: pretrained denoising alone does not guarantee useful generation. Task adaptation is needed to direct the model toward the required outputs, and the resulting performance depends substantially on the quality and accessibility of the underlying representation.

These findings identify a lightweight route from pretrained masked representations to iterative conditional generation, separating the acquisition of denoising capability from the adaptation and decoding mechanisms used to express it.

## References

[1] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces, 2023. URL https://arxiv.org/abs/2107.03006.

[2] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge, 2018. URL https://arxiv.org/abs/1803.05457. arXiv preprint arXiv:1803.05457.

[3] Benjamin Clavié, Nathan Cooper, and Benjamin Warner. It is all in the [mask]: Simple instructiontuning enables bert-like masked language models as generative classifiers. Natural Language Processing Journal, 11:100150, 2025. ISSN 2949-7191. doi: https: //doi.org/10.1016/j.nlp.2025.100150. URL https: //www.sciencedirect.com/science/ article/pii/S2949719125000263.

[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423/.

[5] Bhuwan Dhingra, Kathryn Mazaitis, and William W. Cohen. Quasar: Datasets for question answering by search and reading. CoRR, abs/1707.03904, 2017. doi: 10.48550/ARXIV.1707.03904. URL https://doi. org/10.48550/arXiv.1707.03904.

[6] Jose A. Diaz-Garcia and Julio Amador Diaz Lopez. A survey on cutting-edge relation extraction tech-

Table 8: ARC-Easy performance across pretrained representations (%). Discriminative scores use PLL unless otherwise indicated; generative results use four-pass confidence decoding.
<table><tr><td>Model</td><td>Discriminative</td><td>Frozen gen.</td><td>BitFit 100k</td><td>Full 100k</td><td>Full unmatched</td></tr><tr><td>BERT-base</td><td>31.4</td><td>11.0</td><td>25.2</td><td>28.2</td><td>8.8</td></tr><tr><td>BERT-large</td><td>29.6</td><td>11.8</td><td>28.2</td><td>30.4</td><td>9.8</td></tr><tr><td>RoBERTa-large</td><td>41.2</td><td>17.0</td><td>34.2</td><td>44.2</td><td>5.2</td></tr><tr><td>RoBERTa-RACE</td><td>33.8</td><td>0.4</td><td>35.4</td><td>41.8</td><td>3.6</td></tr></table>

Note: The RACE-adapted model also achieved 49.0% using its native multiple-choice readout. Its frozen generative condition uses a supplied RoBERTa-large MLM head.

Table 9: MMLU results after Full adaptation at 100k training examples (%).
<table><tr><td>Model</td><td>Accuracy</td><td>Unmatched</td></tr><tr><td>BERT-base</td><td>24.6</td><td>15.8</td></tr><tr><td>RoBERTa-large</td><td>33.8</td><td>8.6</td></tr><tr><td>RoBERTa-RACE</td><td>32.6</td><td>9.8</td></tr></table>

niques based on language models. Artificial Intelligence Review, 58(287), 2025. doi: 10.1007/ s10462-025-11280-0. URL https://doi.org/ 10.1007/s10462-025-11280-0.

[7] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6112–6121, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1633. URL https://aclanthology.org/D19-1633/.

[8] Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. DiffusionBERT: Improving generative masked language models with diffusion models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4521–4534, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.248. URL https:// aclanthology.org/2023.acl-long.248/.

[9] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In Proceedings ofthe 9th International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2009.03300.

[10] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel, editors, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785–794, Copenhagen, Denmark, September 2017. Association

for Computational Linguistics. doi: 10.18653/v1/ D17-1082. URL https://aclanthology.org/ D17-1082/.

[11] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach, 2019. URL https://arxiv.org/abs/1907.11692.

[12] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 32819–32848. PMLR, 21– 27 Jul 2024. URL https://proceedings.mlr. press/v235/lou24a.html.

[13] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. CoRR, abs/1609.07843, 2016. doi: 10.48550/ ARXIV.1609.07843. URL https://doi.org/10. 48550/arXiv.1609.07843.

[14] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models, 2024. URL https://arxiv.org/abs/2406.07524.

[15] Julian Salazar, Davis Liang, Toan Q. Nguyen, and Katrin Kirchhoff. Masked language model scoring. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 2699–2712, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020. acl-main.240. URL https://aclanthology. org/2020.acl-main.240/.

[16] Nikolay Savinov, Junyoung Chung, Mikolaj Binkowski, Erich Elsen, and Aaron van den Oord. Step-unrolled denoising autoencoders for text generation, 2022. URL https://arxiv.org/abs/2112.06749.

[17] Alex Wang and Kyunghyun Cho. BERT has a mouth, and it must speak: BERT as a Markov random field language model. In Antoine Bosselut, Asli Celikyilmaz, Marjan Ghazvininejad, Srinivasan Iyer, Urvashi Khandelwal, Hannah Rashkin, and Thomas Wolf, editors,

Proceedings ofthe Workshop on Methodsfor Optimizing and Evaluating Neural Language Generation, pages 30–36, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/ W19-2304. URL https://aclanthology.org/ W19-2304/.

[18] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling, 2025. URL https://arxiv.org/abs/2409.02908.