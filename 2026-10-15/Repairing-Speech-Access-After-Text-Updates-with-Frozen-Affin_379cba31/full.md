# Repairing Speech Access After Text Updates with Frozen Affine Fields

Ming-Hao Hsu Yuxiang Wang Yingda Shen Zhizheng Wu

School of Data Science

The Chinese University of Hong Kong, Shenzhen, China wuzhizheng@cuhk.edu.cn

## Abstract

Speech-capable language models inherit most of their knowledge from text-rich language backbones, and continued model maintenance is likewise dominated by text updates. Knowledge acquired through a text update can nevertheless remain difficult to access when the same query is spoken. We formulate this mismatch as post-update speech access repair, where the knowledge-bearing checkpoint stays frozen and a detachable operator repairs speech-only access in one model pass. We introduce Affine Fields for Interface Repair across Modalities (AFFIRM), which first learns and freezes a base correction for speech hidden states, then adds query-specific corrections with zero mean over the operator-training queries. Paired speech and text states provide the coordinates that condition these corrections on each query. On balanced synthetic-speech PopQA, paired comparisons do not establish a clear advantage for AFFIRM over global SALSA steering or low-rank ReFT interventions under the same frozen single-state constraint. On recorded SLUE-SQA-5 questions with text passages, a full-scale PopQA-fitted AFFIRM operator transfers without refitting and improves ROUGE-L by 0.0941 over its paired No Repair baseline. Low-rank ReFT interventions also transfer positively, whereas the matched SALSA variant falls 0.0707 below its paired baseline. These results show that text acquisition and speech access can diverge after an update and should be verified and repaired separately.

## 1 Introduction

Large language models increasingly serve as the reasoning core of interactive systems that use tools and act on external environments (Yao et al., 2023; Schick et al., 2023). Recent speech-language models extend this interaction from typed text to spoken input and output (Tang et al., 2024; Hu et al., 2024; Xu et al., 2025; Shih et al., 2024). Most general-purpose training data and factual maintenance material for these systems still come from text: web pages, documents, code, and manuals are available at far larger scale than paired speech supervision (Raffel et al., 2020; Gao et al., 2021; Wang et al., 2023). Speech-capable models therefore commonly attach audio encoders and alignment modules to a text-pretrained language backbone, while text remains the direct route for factual adaptation and targeted editing (Tang et al., 2024; Hu et al., 2024; Xu et al., 2025; Meng et al., 2022, 2023).

Native multimodality does not by itself make every text update equally usable from speech. Studies of speech-language models report differences between matched spoken and typed inputs in semantic understanding, reasoning, and factual-recall pathways (Xiang et al., 2025; Cuervo et al., 2026; Modica et al., 2026; Hsu et al., 2026b,c). We study a post-update instance of this mismatch: after a successful text-only update, a model may answer an updated fact when the question is typed yet fail when the same question is spoken. The correct text response shows that the acquired checkpoint can express the updated fact, while the paired speech failure defines the post-update speech-access gap that we repair.

This problem arises whenever knowledge is maintained through text while users continue to query the resulting system by voice. We consider a contract in which the acquired checkpoint is immutable and the repair remains detachable. At deployment, the question is supplied only as speech, and the repair edits one decoder state in one model pass without a transcript. The contract separates the knowledge-bearing checkpoint from the interface that makes its updated knowledge accessible (Figure 1).

The central design question is how to combine a base repair learned across speech queries with a correction tailored to each input. We combine these roles in AFFIRM, or Affine Fieldsfor Interface Repair across Modalities. We first learn and freeze a base correction, called the shared access parent. We then add a query-specific correction, called the centered route field, whose mean over the operatortraining queries is zero. Paired spoken and typed queries provide the hidden-state coordinates used to condition this correction.

![](images/115bcb7c24e37fbb93b3e349db79a2230c27d5765091412c801bf6ddd58b4c85.jpg)  
Figure 1: Decoupled knowledge acquisition and speech access repair. (a) Text-only acquisition can leave a speech-access gap. (b) AFFIRM combines a base correction (shared access parent) with a query-specific correction (centered route field) and edits the final prompt state with the acquired checkpoint frozen. The example shows recorded outputs for one PopQA test question held out from operator fitting, with AFFIRM-DIRECT as the repair.

The experiments first establish a large paired text–speech gap after successful text acquisition. On balanced PopQA, paired bootstrap comparisons do not establish a clear advantage for AFFIRM-DIRECT over DiReFT, LoReFT, or SALSA. When operators fitted on full-scale PopQA transfer without refitting to recorded SLUE questions, AFFIRM-DIRECT and ReFT improve over No Repair, whereas the matched SALSA variant degrades. Parent comparisons and a two-seed depth sweep examine which design choices affect performance, and the compact variant also improves held-out access on Kimi-Audio.

Our contributions are:

• We formulate post-update speech access repair, separating text knowledge acquisition from detachable speech access through an immutable acquired checkpoint.

• We introduce AFFIRM, a staged affine field that combines a shared access parent with a centered query-specific route field learned from paired speech–text routes.

• We show that methods with comparable synthetic-speech performance can differ under recorded-speech transfer, where AFFIRM and

ReFT improve over No Repair while the tested SALSA variant regresses.

## 2 Related Work

Text-centered speech language models and modality gaps. Modern speech language models commonly connect an audio encoder to a pretrained text decoder through projectors, adapters, and staged multimodal training (Tang et al., 2024; Hu et al., 2024; Xu et al., 2025; Hsu et al., 2026a). Behavior alignment and cross-modal distillation train the speech route to reproduce capabilities available from text (Wang et al., 2023, 2024, 2026b; Hu et al., 2026; Cao et al., 2026). Speech inputs nevertheless remain weaker on some semantic and reasoning tasks, and representation similarity is associated with the remaining modality gap (Xiang et al., 2025; Cuervo et al., 2026; Wang et al., 2026a). These studies improve speech during multimodal adaptation, whereas we address the later maintenance stage after a text update has established the target knowledge.

Cross-modal knowledge acquisition and access. Multimodal models can store or acquire information through one route without making it equally available from another (Wen et al., 2025; Basu et al., 2024). Speech and text can also rely on distinct factual-recall pathways inside a shared multimodal model (Modica et al., 2026). Knowledgeediting methods change stored associations in language and audio-language models (Meng et al., 2022, 2023; Chung et al., 2026). Our task instead begins after acquisition succeeds and repairs the route by which a frozen checkpoint accesses the acquired fact.

Frozen representation interventions. A complementary line of work leaves pretrained model weights fixed and modifies behavior through narrow interventions on hidden representations. Inference-Time Intervention learns directions at selected attention heads, while activation addition modifies residual-stream states with behaviorspecific directions (Li et al., 2023; Turner et al., 2023). ReFT generalizes learnable representation interventions at chosen layers and positions, including the DiReFT and LoReFT parameterizations (Wu et al., 2024). These methods adapt behavior through a fixed text-model backbone, but they do not separate a text-acquisition stage from speech access to the acquired knowledge. AFFIRM brings this intervention family to post-update speech access by learning from paired speech–text routes and retaining a detachable repair around a frozen shared parent.

Global and conditional steering. Learned speech steering can optimize shared layer-wise additive directions with a frozen backbone (Yegorova et al., 2026). Input-dependent steering predicts an intervention from each multimodal context (Parekh et al., 2025), while Steer-MoE dynamically mixes expert directions inside audio-encoder layers (Feng et al., 2025). Angle and length interventions provide another representation-level route for reducing the speech–text modality gap (Xiang et al., 2025). Together, these approaches establish global and input-conditioned steering as complementary mechanisms for reducing speech–text discrepancies. AFFIRM addresses post-update factual access through a detachable operator that edits one decoder state of a frozen acquired checkpoint.

## 3 Preliminaries and Problem Formulation

## 3.1 Post-update access

Let $F _ { \theta ^ { + } }$ be a speech-capable autoregressive checkpoint obtained by text-only answer-likelihood training from $F _ { \theta _ { 0 } }$ . Each fact i has a reference answer $y _ { i } \ = \ ( y _ { i , 1 } , \ldots , y _ { i , T _ { i } } )$ and semantically matched text and speech queries $x _ { i } ^ { T }$ and $x _ { i } ^ { S }$ . For modality $m \in \{ T , S \}$ at decoder block $\ell ,$ let $h _ { i } ^ { m } \in \mathbb { R } ^ { d }$ denote the hidden state at the final attended prompt position before the model predicts $y _ { i , 1 }$ . The two states are directly comparable because they come from the same checkpoint and correspond to matched queries. We verify $F _ { \theta ^ { + } }$ through the text route and then freeze it. For acquired-fact evaluation on PopQA and TOFU, the disjoint operator-training, development, and evaluation sets $\mathcal { D } _ { \mathrm { o p } } , \mathcal { D } _ { \mathrm { d e v } }$ , and $\mathcal { D } _ { \mathrm { e v a l } }$ are subsets of the text-acquisition set. Evaluation facts provide no operator gradients or selection signals.

For any scored set E and scoring function $q ,$ route-specific access is

$$
\begin{array} { c } { { \displaystyle \mathrm { A c c e s s } _ { m } ( F _ { \theta ^ { + } } ; { \mathcal E } ) = \frac { 1 } { | { \mathcal E } | } \sum _ { i \in { \mathcal E } } q ( F _ { \theta ^ { + } } ( x _ { i } ^ { m } ) , y _ { i } ) \ : , } } \\ { { \Delta _ { T  S } = \mathrm { A c c e s s } _ { T } - \mathrm { A c c e s s } _ { S } \ : . } } \end{array}\tag{1}
$$

A positive $\Delta _ { T  S }$ is an operational post-update access gap: the acquired checkpoint can answer through text more reliably than through the matched speech interface.

## 3.2 Frozen repair contract

After text acquisition, we fit a detachable repair operator $R _ { \phi } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ $\mathcal { D } _ { \mathrm { o p } }$ and deploy it as

$$
\begin{array} { r l } & { \widetilde { h } _ { i } ^ { S } = h _ { i } ^ { S } + \alpha R _ { \phi } ( h _ { i } ^ { S } ) , } \\ & { \widetilde { y } _ { i } = F _ { \theta ^ { + } } ( x _ { i } ^ { S } ; h _ { i } ^ { S }  \widetilde { h } _ { i } ^ { S } ) , } \end{array}\tag{2}
$$

where $\theta ^ { + }$ remains frozen during operator training and inference. The scalar α controls intervention strength and is fixed before test evaluation. The question is supplied only as speech, without a transcript, matched text-question state, retrieved example, or second model pass. Text-query success establishes acquisition, the paired route gap diagnoses inaccessible behavior, and held-out speech improvement measures repair. We separately test transfer to SLUE, where recorded questions accompany text passages and the task, prompt, speakers, and acoustics change together without refitting.

## 4 AFFIRM: Shared Access Parent with Centered Route Field

## 4.1 Method overview

AFFIRM decomposes post-update repair into a shared access parent and a query-conditioned route field around that parent. We first learn a lowdimensional code that predicts the difference between matched text and speech prompt states, without answer labels. Stage A trains the parent using answer supervision through the frozen decoder. Stage B freezes the parent and the coordinate map, then trains only the readout that converts these codes into query-specific corrections. The acquired checkpoint remains frozen throughout, and deployment stores only the compact operator.

## 4.2 Affine repair field

For a speech state $h ^ { S } \in \mathbb { R } ^ { d }$ at decoder block $\ell ,$ every AFFIRM variant adds the parent and centered correction:

$$
\begin{array} { c } { { R _ { \phi } ( h ^ { S } ) = G _ { \psi } ( h ^ { S } ) + \beta U \big ( z ( h ^ { S } ) - \bar { z } \big ) , } } \\ { { \widetilde { h } ^ { S } = h ^ { S } + \alpha R _ { \phi } ( h ^ { S } ) . } } \end{array}\tag{3}
$$

Here $G _ { \psi }$ is the shared access parent, $z ( h ^ { S } ) \in \mathbb { R } ^ { r }$ is a paired-route coordinate, and $U \in \mathbb { R } ^ { d \times r }$ maps route variation back to the decoder state. The coefficients $\beta$ and α control the route-field strength and overall intervention strength, respectively. With $\begin{array} { r } { \bar { z } = | \mathcal { D } _ { \mathrm { o p } } | ^ { - 1 } \sum _ { i } z ( h _ { i } ^ { S } ) } \end{array}$ , the route field has zero mean on the operator-training population and therefore represents variation around the fitted parent. The route field is affine in the route coordinate z, while z itself depends nonlinearly on $h ^ { S }$ through Equation 4.

## 4.3 Answer-free paired-route coordinates

The frozen checkpoint produces paired states $\{ ( h _ { i } ^ { S } , h _ { i } ^ { T } ) \} _ { i = 1 } ^ { N }$ at block $\ell ,$ where $N = | D _ { \mathrm { o p } } |$ and $\Delta _ { i } = h _ { i } ^ { T } - h _ { i } ^ { S }$ . Let $\mu _ { S }$ and $\bar { \Delta }$ be the corresponding population means, and let $\sigma _ { S }$ and $\sigma _ { \Delta }$ be scalar rootmean-square scales over samples and dimensions. We fit a low-rank coordinate map D and temporary readout V using

$$
\begin{array} { r l r } {  { z _ { i } = \mathrm { S i L U } ( D \frac { h _ { i } ^ { S } - \mu _ { S } } { \sigma _ { S } } ) , } } \\ & { } & { \delta _ { i } = \frac { \Delta _ { i } - \bar { \Delta } } { \sigma _ { \Delta } } , } \\ & { } & { D ^ { * } , V ^ { * } = \arg \operatorname* { m i n } _ { D , V } \frac { 1 } { N d } \sum _ { i } \| V z _ { i } - \delta _ { i } \| _ { 2 } ^ { 2 } . } \end{array}\tag{4}
$$

The fit uses paired prompt states but no answer labels. We freeze the coordinate map $D ^ { * }$ and discard the temporary readout $V ^ { * }$ , whose role was to predict state differences. Stage B learns a separate readout $U$ to turn these fixed coordinates into corrections that improve answer generation.

## 4.4 Stage A: decoder-aware access parents

Stage A trains the parent through the frozen decoder using teacher-forced answer cross-entropy.

If $p _ { i , t } ^ { G }$ denotes the answer distribution after applying the parent to $h _ { i } ^ { S }$ , the objective is

$$
\mathcal { L } _ { A } ( \psi ) = - \frac { 1 } { N } \sum _ { i } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } \log p _ { i , t } ^ { G } ( y _ { i , t } ) ,\tag{5}
$$

where prompt tokens are masked and only $\psi$ receives gradients. The direct and expert parents are

$$
\begin{array} { l } { G _ { \mathrm { d i r e c t } } ( h ) = g , } \\ { G _ { \mathrm { e x p e r t } } ( h ) = { s } \displaystyle \sum _ { k = 1 } ^ { K } \mathrm { s o f t m a x } ( W h ) _ { k } e _ { k } , } \end{array}\tag{6}
$$

where $g$ is one shared vector, $W$ is a router, $e _ { k }$ are learned directions, and s is a learned scale. All parent parameters are optimized with Equation 5 and frozen before Stage B. In the balanced PopQA diagnostic, the $K = 6 4$ softmax concentrates almost all weight on the same expert across operator-training states, yielding a nearly constant correction. In this diagnostic, the expert bank acts as an alternative parameterization of a shared correction. We also use the compact $K = 1 \mathrm { f o r m } G ( h ) = s e$ to study intervention depth and cross-backbone portability, with AFFIRM-DIRECT as the primary variant.

## 4.5 Stage B: selective correction around a frozen parent

Stage B initializes $U = 0$ and trains only U while the checkpoint, $D ^ { * }$ , and the selected parent remain frozen. For each fitting item, teacher-forced decoding supplies text-route and parent-only speech logits. We choose the text route when its mean goldanswer log probability exceeds the parent route by more than threshold $\tau _ { \ast }$ , and otherwise use the parent route as a preservation teacher. Let $\pi _ { i , t } ^ { * }$ be the selected teacher’s top-k distribution after temperature scaling and renormalization, and let $p _ { i , t } ^ { U }$ be the fullfield speech distribution. The readout minimizes

$$
U ^ { * } = \arg \operatorname* { m i n } _ { U } \sum _ { i } \frac { 1 } { N T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } D _ { \mathrm { K L } } \big ( \mathrm { s g } ( \pi _ { i , t } ^ { * } ) \ : \| p _ { i , t } ^ { U } \big ) ,\tag{7}
$$

where sg denotes stop-gradient. The text teacher supplies correction when text retains a clear answerlikelihood advantage, while the parent teacher preserves an already useful answer distribution. Zero initialization makes the first forward pass in Stage B reproduce the parent-only predictions, and backpropagation through the frozen decoder updates only $U$

## 4.6 Fitting and deployment

For the primary Qwen experiments, we select the block, overall strength α, and route-field coefficient β on development data, then fix the operator before testing. At inference, AFFIRM edits the final prompt state during prefill and then continues ordinary greedy generation without editing answertoken states. It requires no question transcript, matched text-question state, pre-update checkpoint $F _ { \theta _ { 0 } }$ , or external ASR system. Appendix A gives the fitting schedules and inference settings for each variant.

## 5 Experimental Setup

Model and factual acquisition. We use Qwen2.5-Omni-7B (Xu et al., 2025). For full-scale experiments, we fine-tune one checkpoint on text question–answer pairs from all 14,267 PopQA facts over 16 relations (Mallen et al., 2023) and another on all 4,000 TOFU questions over 200 fictional authors (Maini et al., 2024). We then freeze each checkpoint and fit its repair operators. We also evaluate compact operators on separately acquired Kimi-Audio-7B (KimiTeam et al., 2025) and MiMo-Audio-7B (Xiaomi LLM-Core Team, 2025) checkpoints (Appendix D.5).

Full-scale operator splits. For PopQA, we group questions linked by a shared subject or normalized question string before splitting, so no group crosses the 8,572/2,848/2,847 operator train/development/test split. PopQA-20 and PopQA-60 use nested subsets of 2,853 and 8,560 training examples, approximately 20% and 60% of the full acquisition corpus, with the same development and test sets. TOFU uses 120/40/40 train/development/test authors, yielding 2,400/800/800 examples without author overlap.

Balanced comparison and baseline fitting. For the balanced PopQA comparison, we use a separate checkpoint acquired on 1,000 facts and a 600/199/201 operator train/development/test split that keeps duplicate questions together. We separately fit each method on PopQA-20 for recordedspeech transfer and on TOFU training authors for author-disjoint evaluation. Appendix B gives the implementation and development-selection details for every reproduced baseline.

Speech-only baselines. The frozen-checkpoint comparison evaluates the following speech-only representation-level repair methods:

• No Repair applies the acquired checkpoint to speech without an added intervention.

• DiReFT and LoReFT (Wu et al., 2024) learn low-rank hidden-state interventions at a selected decoder state.

• L2S (Parekh et al., 2025) predicts an inputdependent steering intervention from the multimodal context.

• SALSA (Yegorova et al., 2026) learns one shared speech-aware steering activation vector with a frozen backbone.

All speech-only methods receive the question as audio, without its transcript or a paired text-question state, and use one pass through the frozen acquired checkpoint. Speech LoRA (Hu et al., 2022), speech-to-text knowledge distillation (S2T-KD, Wang et al., 2026b), a Whisper ASR cascade (Radford et al., 2023), and full speech supervision are reported separately as operational alternatives because they relax the frozen single-state deployment contract.

Donor-assisted controls. Angle and Length (Xiang et al., 2025) modify speech representations through direction and norm adjustments using a paired text-question state as a per-example inference-time donor. We report them separately because the additional typed question is unavailable to speech-only methods.

Speech and transfer. CosyVoice renders the PopQA and TOFU questions used for fitting and evaluation (Du et al., 2024). Our transfer evaluation uses all 256 recorded SLUE-SQA-5 questions from 50 human speakers and five source datasets (Shon et al., 2023). Each SLUE input contains a text passage and a spoken question, so speech-only refers to the question modality. We apply each frozen PopQA-20 operator with no SLUE training examples, refitting, or target-domain strength retuning. The separate Angle and Length controls access the paired text question at inference. This tests transfer to passage-based question answering under changes in task, prompt, speakers, and acoustics, rather than recall of the acquired PopQA facts.

AFFIRM variants. Each variant combines a parent with the centered route field:

• AFFIRM-DIRECT combines a direct shared parent with the centered route field and is the primary AFFIRM variant.

• AFFIRM-EXPERT combines the expert parent with the centered route field after Stage A.

• AFFIRM-K1 uses a compact scale–direction parent for depth and cross-backbone characterization.

Optimization, selection, and metrics. The hidden width is $d = 3$ , 584 and low-rank operators use $r \ = \ 6 4$ . Stage B freezes its parent and D, trains zero-initialized U, and deploys it with fixed $\beta = 0 . 2 5$ . We sweep $\ell \in \{ 4 , 8 , 1 2 , 1 6 , 2 0 , 2 4 , 2 8 \}$ only on development data, calibrate matched edit norms before test generation, and report ROUGE-L recall (RL), normalized exact match (EM), and SLUE answer-span accuracy. Appendix A gives complete schedules, doses, prompts, counts, seeds, and uncertainty procedures.

## 6 Results

## 6.1 The text update leaves a measurable speech-access gap

Text-only acquisition succeeds without making the acquired knowledge equally usable from speech. On CosyVoice-rendered paired questions over all 14,267 PopQA facts, the acquired checkpoint reaches 0.9213 RL and 0.9187 EM through text, compared with 0.1840 RL and 0.1519 EM through speech. The held-out PopQA and TOFU questions test access to facts included in text acquisition. To test whether speech states retain question-specific information, we retrieve each speech state’s matching text state among 600 paired questions. After subtracting each modality’s mean state, top-1 retrieval accuracy is 50.3%, compared with 0.17% chance (Appendix D).

## 6.2 Methods comparable on synthetic speech diverge on recorded speech

Table 1 compares methods on balanced PopQA, author-disjoint TOFU, and recorded SLUE questions. Learned operators are fitted separately on balanced PopQA, TOFU, and PopQA-20, with only the PopQA-20 operator transferred to SLUE.

On balanced synthetic-speech PopQA, SALSA and AFFIRM-DIRECT recover substantial access, and the paired-bootstrap comparisons with DiReFT, LoReFT, or SALSA do not resolve an advantage for AFFIRM-DIRECT (Appendix Table 8). Among the speech-only rows, AFFIRM-DIRECT provides the strongest TOFU RL and EM point estimates.

Donor-assisted control. Supplying the paired text-question state gives the model access to the typed question at inference. Under this relaxed condition, Angle improves TOFU RL and EM over No Repair, whereas Length remains near the no-repair point (Table 2).

On SLUE, AFFIRM-DIRECT improves RL by 0.0941 and LoReFT improves by 0.0774, while SALSA falls 0.0707 below its paired No Repair baseline. These operators use no SLUE training examples, question transcripts, or target-task strength selection. The SLUE SALSA operator is fitted on PopQA-20, with its strength calibrated to match AFFIRM’s mean edit norm on source speech states. In a separate control, we recalibrate SALSA on unlabeled SLUE inputs to match AFFIRM’s mean edit norm, and SALSA still falls below No Repair (Appendix B). The positive AFFIRM result concerns the complete operator under this joint shift in task, prompt, speakers, and acoustics.

## 6.3 Parent parameterization affects source and transfer performance

The operators in Tables 1 and 4 edit block 20, the same site as DiReFT and LoReFT (Appendix Table 7). For the matched parent analysis in Table 3, we select block 4 using development data. TOFU operators are fitted on TOFU training authors, whereas SLUE operators are fitted on PopQA-20 and transferred unchanged.

With the centered route field included in both methods, AFFIRM-EXPERT scores higher on TOFU, while AFFIRM-DIRECT has higher point estimates on SLUE. On SLUE, AFFIRM-DIRECT exceeds AFFIRM-EXPERT by $+ 0 . 0 4 3 0 \pm 0 . 0 1 8 5$ EM (bootstrap SE). The point-estimate ranking of the two complete operators therefore differs between TOFU and SLUE.

## 6.4 Scale, stability, and cross-backbone scope

The full-scale PopQA results in Appendix Table 12 preserve the gains at two operator-training budgets, and three Stage-B seeds reproduce both the PopQA-20 AFFIRM-DIRECT and TOFU AFFIRM-EXPERT development results (Appendix Table 13). The compact AFFIRM-K1 variant also improves heldout access on a separately acquired Kimi-Audio checkpoint, from 0.6321/0.0850 to 0.6451/0.1188 RL/EM. This result supports portability of the compact variant to a second backbone under the same frozen speech-only constraint. The primary Qwen AFFIRM-DIRECT operator stores 0.46M learned values, edits one state, and uses one speech pass without a question transcript. Appendix Table 17 compares it with speech LoRA, distillation, ASR, and full speech supervision under relaxed deployment constraints.

<table><tr><td></td><td colspan="2">Balanced PopQA</td><td colspan="2">TOFU</td><td colspan="3">SLUE no-refit</td></tr><tr><td>Method</td><td>RL</td><td>EM</td><td>RL</td><td>EM</td><td>Span</td><td>RL</td><td>EM</td></tr><tr><td>No Repair</td><td>0.2736</td><td>0.2189</td><td>0.5003</td><td>0.1000</td><td>0.4063</td><td>0.4435</td><td>0.2344</td></tr><tr><td>DiReFT (Wu et al., 2024)</td><td>0.5945</td><td>0.5821</td><td>0.5117</td><td>0.1075</td><td>0.4609</td><td>0.4981</td><td>0.2734</td></tr><tr><td>LoReFT (Wu et al., 2024)</td><td>0.5995</td><td>0.5920</td><td>0.5179</td><td>0.1138</td><td>0.4766</td><td>0.5209</td><td>0.3047</td></tr><tr><td>L2S (Parekh et al., 2025)</td><td>0.5361</td><td>0.5274</td><td>0.5107</td><td>0.1000</td><td>0.3906</td><td>0.4261</td><td>0.2656</td></tr><tr><td>SALSA (Yegorova et al., 2026)</td><td>0.6119</td><td>0.6020</td><td>0.5237</td><td>0.1150</td><td>0.3203</td><td>0.3730</td><td>0.2656</td></tr><tr><td>AFFIRM-DIRECT</td><td>0.6318</td><td>0.6219</td><td>0.5310</td><td>0.1200</td><td>0.4922</td><td>0.5376</td><td>0.3008</td></tr></table>

Table 1: Post-update factual access across source repair, author-disjoint evaluation, and no-refit recordedspeech transfer. PopQA reports the balanced test, TOFU reports 800 author-disjoint questions after TOFU fitting, and SLUE reports 256 recorded questions with text passages after PopQA-20 fitting without target-task training examples or strength retuning. RL denotes ROUGE-L recall, EM denotes normalized exact match, and Span denotes SLUE answer-span accuracy. All rows use one speech-only pass without a text-question donor, and AFFIRM-DIRECT edits decoder block 20. The displayed No Repair row is paired with AFFIRM and ReFT. On SLUE, L2S and SALSA use their separately generated No Repair outputs for the paired differences reported in the text and Appendix Table 9.
<table><tr><td rowspan="2">Method</td><td colspan="2">Balanced PopQA</td><td colspan="2">TOFU</td><td colspan="3">SLUE no-refit</td></tr><tr><td>RL</td><td>EM</td><td>RL</td><td>EM</td><td>Span</td><td>RL</td><td>EM</td></tr><tr><td>No Repair</td><td>0.2769</td><td>0.2239</td><td>0.4996</td><td>0.0988</td><td>0.4102</td><td>0.4437</td><td>0.2422</td></tr><tr><td>Angle (Xiang et al., 2025)</td><td>0.5398</td><td>0.4925</td><td>0.6547</td><td>0.2550</td><td>0.4219</td><td>0.4619</td><td>0.2227</td></tr><tr><td>Length (Xiang et al., 2025)</td><td>0.2794</td><td>0.2239</td><td>0.5050</td><td>0.1025</td><td>0.4141</td><td>0.4476</td><td>0.2344</td></tr></table>

Table 2: Donor-assisted controls under a relaxed inference condition. Angle and Length modify the speech state using the paired text-question state at inference. No Repair reports the outputs generated alongside these controls on each dataset. RL, EM, and Span follow the definitions in Table 1.

## 7 Component and Depth Analysis

Shared parent and route field. We test each component’s contribution under the same frozen one-state contract (Table 4). Either component recovers substantial access, and their AFFIRM-DIRECT composition gives the strongest RL across PopQA-20, PopQA-60, and TOFU. Adding the route field to the shared parent improves RL and EM on both PopQA budgets, and adding the shared parent to the route field improves RL on both PopQA budgets (Appendix Table 10). On TOFU, the paired-bootstrap analysis does not resolve any of the four corresponding RL and EM differences.

For the balanced PopQA AFFIRM-DIRECT operator, before inference scaling, the shared parent vector has norm 31.48, the per-query route-field corrections have mean norm 8.69, and their mean cosine with the parent is 0.0104. The mean centered route code over the operator-training set has norm below $2 \times 1 0 ^ { - 6 }$ . This geometry is consistent with a dominant shared translation and a smaller, nearly orthogonal query-specific refinement.

Compact-parent depth. In the two-seed TOFU development sweep (Appendix D.4), we hold the split, K = 1 parent form, route-field coefficient, and target edit norm fixed across blocks 4 through 28. Both seeds select block 4, and mean full-field RL declines from 0.5550 at block 4 to 0.4788 at block 28. Within this compact variant, early blocks perform best, a pattern consistent with greater remaining decoder depth for integrating the edited prompt state. The route field changes RL by at most 0.0068 at any block and seed, and the sign of its contribution varies across blocks and seeds. The parent alone also peaks at block 4 in both seeds, so the depth trend comes from the parent.

<table><tr><td></td><td colspan="2">TOFU</td><td colspan="3">SLUE no-refit</td></tr><tr><td>Method</td><td>RL</td><td>EM</td><td>Span</td><td>RL</td><td>EM</td></tr><tr><td>AFFIRM-DIRECT (block 4)</td><td>0.5588</td><td>0.1425</td><td>0.5000</td><td>0.5335</td><td>0.3750</td></tr><tr><td>AFFIRM-EXPERT (block 4)</td><td>0.5798</td><td>0.1888</td><td>0.4805</td><td>0.5275</td><td>0.3320</td></tr></table>

Table 3: Parent parameterization affects the complete operator’s source and transfer performance. AFFIRM-DIRECT and AFFIRM-EXPERT combine the centered route field with the direct and expert parents, respectively. Both rows edit decoder block 4. TOFU reports RL and EM on 800 author-disjoint questions after TOFU fitting. SLUE reports answer-span accuracy (Span), RL, and EM on 256 recorded questions after the corresponding PopQA-20-fitted operators transfer without refitting.
<table><tr><td></td><td colspan="2">PopQA-20</td><td colspan="2">PopQA-60</td><td colspan="2">TOFU</td></tr><tr><td>Method</td><td>RL</td><td>EM</td><td>RL</td><td>EM</td><td>RL</td><td>EM</td></tr><tr><td>No Repair</td><td>0.1787</td><td>0.1500</td><td>0.1787</td><td>0.1500</td><td>0.5003</td><td>0.1000</td></tr><tr><td>+ Direct Parent</td><td>0.3156</td><td>0.2986</td><td>0.3269</td><td>0.3087</td><td>0.5252</td><td>0.1138</td></tr><tr><td>+ Route Field</td><td>0.3212</td><td>0.3038</td><td>0.3379</td><td>0.3203</td><td>0.5265</td><td>0.1213</td></tr><tr><td>AFFIRM-DIRECT</td><td>0.3353</td><td>0.3137</td><td>0.3493</td><td>0.3270</td><td>0.5310</td><td>0.1200</td></tr></table>

Table 4: Component ablation of the shared access parent and centered route field. Rows beginning with “+” add one component to the frozen acquired checkpoint, and AFFIRM-DIRECT combines both components. PopQA-20, PopQA-60, and TOFU report RL and EM on their locked test splits. All learned operators edit decoder block 20 and use development-matched edit norms.

## 8 Discussion and Conclusion

The experiments show that text acquisition and speech access can diverge after an update. AFFIRM implements speech-access repair as a detachable operator, combining a shared access parent with a smaller centered route field for query-specific variation. This structure complements speech-pathway retraining when checkpoint immutability and onepass speech inference matter.

On balanced synthetic speech, paired comparisons do not establish a clear AFFIRM advantage over SALSA or ReFT. Operators fitted separately on full-scale PopQA differ in transfer to recorded SLUE questions with text passages, where ReFT and AFFIRM improve over No Repair while the tested SALSA variant regresses.

With the centered route field included, the expert parent gives higher block-4 TOFU scores and the direct parent gives higher SLUE point estimates. The compact TOFU sweep favors early decoder blocks.

Knowledge can be acquired through the richest supervision route, and speech access can then be verified and repaired without modifying the acquired checkpoint.

## Limitations

We study short-answer factual access after textonly updates, primarily with Qwen2.5-Omni-7B. Our PopQA and TOFU evaluations measure access to facts included in text acquisition, with disjoint operator-training, development, and test groups. They therefore characterize transfer of the repair across acquired facts rather than acquisition of unseen knowledge. The paired text–speech generation gap operationalizes access but does not localize every acoustic or internal cause.

Our PopQA and TOFU experiments use synthetic English speech from one fixed voice. Recorded SLUE extends the evaluation to human speech, but jointly changes the task, prompt, speakers, and acoustics. It also supplies a text passage, so this evaluation measures no-refit behavior transfer rather than closed-book recall of newly acquired facts from human recordings. The transfer result concerns the complete fitted operator and does not isolate the centered route field’s contribution.

Kimi-Audio provides a locked-test evaluation of the compact realization, while MiMo-Audio gives mixed development results. The single-state intervention restricts adaptation capacity, and broader speech fine-tuning or an ASR cascade can attain higher accuracy when the deployment contract is relaxed. Long-form generation, multilingual access, repeated checkpoint updates, and architecturespecific scaling remain future directions.

## Ethical Considerations

Our experiments concern access to factual information through spoken queries. Improved access can make voice interfaces more useful, but an intervention can also change a previously correct answer or expose incorrect information stored by the model. Aggregate benchmark gains do not establish reliability for consequential applications. The English synthetic-speech and recorded-speech evaluations leave performance across languages, accents, and user groups incompletely characterized. Use of the models and speech datasets remains subject to their respective licenses and usage conditions.

AI assistance. Generative-AI tools assisted with planning, experiment orchestration, code review, literature synthesis, analysis, and editing. The authors verified the retained measurements against experiment artifacts and take responsibility for the paper’s text, code, results, and claims.

## References

Samyadeep Basu, Martin Grayson, Cecily Morrison, Besmira Nushi, Soheil Feizi, and Daniela Massiceti. 2024. Understanding information storage and transfer in multi-modal large language models. In NeurIPS.

Di Cao, Dongjie Fu, Hai Yu, Siqi Zheng, Xu Tan, and Tao Jin. 2026. X-OPD: Cross-modal on-policy distillation for capability alignment in speech LLMs. CoRR, abs/2603.24596.

Sung Kyun Chung, Jiaheng Dong, Qiuchi Hu, Jinuo Sun, Gongping Huang, Hong Jia, and Ting Dang. 2026. Localizing and editing knowledge in large audio-language models. In Interspeech, pages 6416– 6420. ISCA.

Santiago Cuervo, Skyler Seto, Maureen de Seyssel, Richard He Bai, Zijin Gu, Tatiana Likhomanenko, Navdeep Jaitly, and Zakaria Aldeneh. 2026. Closing the gap between text and speech understanding in LLMs. In ICLR.

Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, Zhifu Gao, and Zhijie Yan. 2024. CosyVoice: A scalable multilingual zero-shot textto-speech synthesizer based on supervised semantic tokens. CoRR, abs/2407.05407.

Ruitao Feng, Bixi Zhang, Sheng Liang, and Zheng Yuan. 2025. Steer-MoE: Efficient audio-language

alignment with a mixture-of-experts steering module. CoRR, abs/2510.13558.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2021. The Pile: An 800GB dataset of diverse text for language modeling. CoRR, abs/2101.00027.

Ming-Hao Hsu, Yuxuan Hu, Shujie Liu, Jinyu Li, and Zhizheng Wu. 2026a. Is text all you need? text as a universal information bottleneck for speech LLMs. In NeurIPS.

Ming-Hao Hsu, Xiaohai Tian, Jun Zhang, Lu Lu, Yuxuan Wang, and Zhizheng Wu. 2026b. Geometric alignment without functional equivalence: A layerwise analysis of the speech-text modality gap. In NeurIPS.

Ming-Hao Hsu, Xiaohai Tian, Jun Zhang, and Zhizheng Wu. 2026c. Entity binding failures in speech LLM reasoning: Diagnosis and chain-of-thought intervention. In Interspeech, pages 904–909.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In ICLR. OpenReview.net.

Jing Hu, Danxiang Zhu, Xianlong Luo, Dan Zhang, Shuwei He, Yishu Lei, Haitao Zheng, Shikun Feng, Jingzhou He, Yu Sun, Hua Wu, and Haifeng Wang. 2026. CORD: bridging the audio-text reasoning gap via weighted on-policy cross-modal distillation. CoRR, abs/2601.16547.

Shujie Hu, Long Zhou, Shujie Liu, Sanyuan Chen, Lingwei Meng, Hongkun Hao, Jing Pan, Xunying Liu, Jinyu Li, Sunit Sivasankaran, Linquan Liu, and Furu Wei. 2024. WavLLM: Towards robust and adaptive speech large language model. In EMNLP (Findings), Findings of ACL, pages 4552–4572. Association for Computational Linguistics.

KimiTeam, Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, Zhengtao Wang, Chu Wei, Yifei Xin, Xinran Xu, Jianwei Yu, Yutao Zhang, Xinyu Zhou, Y. Charles, and 21 others. 2025. Kimi-Audio technical report. CoRR, abs/2504.18425.

Kenneth Li, Oam Patel, Fernanda B. Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. In NeurIPS.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. 2024. TOFU: A task of fictitious unlearning for LLMs. In COLM.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating

effectiveness of parametric and non-parametric memories. In ACL (1), pages 9802–9822. Association for Computational Linguistics.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In NeurIPS.

Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. 2023. Massediting memory in a transformer. In ICLR. Open-Review.net.

Luca Modica, Filip Landin, Mehrdad Farahani, Livia Qian, Gabriel Skantze, and Richard Johansson. 2026. Do factual recall mechanisms carry over from text to speech in multimodal language models? In \*SEM, pages 401–409. Association for Computational Linguistics.

Jayneel Parekh, Pegah Khayatan, Mustafa Shukor, Arnaud Dapogny, Alasdair Newson, and Matthieu Cord. 2025. Learning to steer: Input-dependent steering for multimodal LLMs. In NeurIPS.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In ICML, Proceedings of Machine Learning Research, pages 28492–28518. PMLR.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67.

Craig Saunders, Alexander Gammerman, and Volodya Vovk. 1998. Ridge regression learning algorithm in dual variables. In ICML, pages 515–521. Morgan Kaufmann.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. In NeurIPS.

Min-Han Shih, Ho-Lam Chung, Yu-Chi Pai, Ming-Hao Hsu, Guan-Ting Lin, Shang-Wen Li, and Hung-yi Lee. 2024. GSQA: An end-to-end model for generative spoken question answering. In Interspeech, pages 2970–2974.

Suwon Shon, Siddhant Arora, Chyi-Jiunn Lin, Ankita Pasad, Felix Wu, Roshan S. Sharma, Wei-Lun Wu, Hung-yi Lee, Karen Livescu, and Shinji Watanabe. 2023. SLUE Phase-2: A benchmark suite of diverse spoken language understanding tasks. In ACL (1), pages 8906–8937. Association for Computational Linguistics.

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. 2024. SALMONN: towards generic hearing abilities for large language models. In ICLR. Open-Review.net.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. CoRR, abs/2308.10248.

Chaoren Wang, Heng Lu, Xueyao Zhang, Shujie Liu, Yan Lu, Jinyu Li, and Zhizheng Wu. 2026a. Closing the modality reasoning gap for speech large language models. In ACL (1), pages 18821–18835. Association for Computational Linguistics.

Chen Wang, Minpeng Liao, Zhongqiang Huang, Jinliang Lu, Junhong Wu, Yuchen Liu, Chengqing Zong, and Jiajun Zhang. 2023. BLSP: bootstrapping language-speech pre-training via behavior alignment of continuation writing. CoRR, abs/2309.00916.

Chen Wang, Minpeng Liao, Zhongqiang Huang, and Jiajun Zhang. 2024. BLSP-KD: bootstrapping languagespeech pre-training via knowledge distillation. CoRR, abs/2405.19041.

Enzhi Wang, Qicheng Li, Zhiyuan Tang, and Yuhang Jia. 2026b. Cross-modal knowledge distillation for speech large language models. In ICASSP, pages 16112–16116. IEEE.

Yuxin Wen, Yangsibo Huang, Tom Goldstein, Ravi Kumar, Badih Ghazi, and Chiyuan Zhang. 2025. Quantifying cross-modality memorization in visionlanguage models. In NeurIPS.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. 2024. ReFT: Representation finetuning for language models. In NeurIPS.

Bajian Xiang, Shuaijiang Zhao, Tingwei Guo, and Wei Zou. 2025. Understanding the modality gap: An empirical study on the speech-text alignment mechanism of large speech language models. In EMNLP, pages 5187–5202. Association for Computational Linguistics.

Xiaomi LLM-Core Team. 2025. MiMo-Audio: Audio language models are few-shot learners. CoRR, abs/2512.23808.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025. Qwen2.5-Omni technical report. CoRR, abs/2503.20215.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In ICLR. OpenReview.net.

Yekaterina Yegorova, Argyrios Gerogiannis, Haolong Zheng, Julia Hockenmaier, Chang D. Yoo, and Mark A. Hasegawa-Johnson. 2026. SALSA: Speech aware LLM adaptation via learned steering activation vectors. CoRR, abs/2606.00460.

## A Training and Evaluation Details

We describe the data splits, model checkpoints, training procedures, and evaluation settings.

## A.1 Datasets and splits

Table 5 records the exact populations used by the reported experiments. Before splitting PopQA, we group questions into connected components linked by shared subjects or normalized duplicate questions. For TOFU, we group all questions about the same fictional author. Each group belongs to a single operator-training, development, or test split.

The PopQA-20 and PopQA-60 subsets are nested and are drawn only from the 8,572 operatortraining rows. The full acquired checkpoint saw every PopQA fact through text, including operator development and test facts. This design measures generalization of the access operator across already acquired facts. TOFU uses 120, 40, and 40 train, development, and test authors. CosyVoice-300M-SFT renders every PopQA and TOFU speech question at 16 kHz with one fixed English female voice. The same cached waveform is reused across methods for an item. We evaluate on all 256 SLUE recordings from 50 speakers without fitting on SLUE.

## A.2 Acquired checkpoints and prompts

Text acquisition updates all Qwen2.5-Omni-7B parameters for ten epochs on the corresponding text QA corpus. It uses batch size one with four-step gradient accumulation, maximum sequence length 500, learning rate $2 \times 1 0 ^ { - 5 }$ , zero weight decay, a 0.03 warm-up ratio, and gradient-norm clipping at 1. The balanced and full PopQA acquisitions are separate checkpoints, and TOFU uses its own acquired checkpoint. Every access operator is fitted only after acquisition, and the acquired checkpoint remains frozen during operator fitting and evaluation.

Text questions use a plain question-answer prompt, while speech questions use the model’s native audio wrapper with the same answer target. The SLUE prompt contains a text passage followed by a recorded speech question under Qwen’s standard multimodal system prompt. No automatic transcript or text version of the spoken question is inserted for speech-only methods. The five SLUE source datasets change both the linguistic task mixture and the prompt structure relative to short PopQA questions.

## A.3 State extraction and operator optimization

The primary operator state is the final attended prompt position after decoder block 20. The matched parent comparison and compact depth characterization use the blocks specified in their respective sections. Qwen has hidden width $d =$ 3,584, and every low-rank operator uses rank $r =$ 64. Table 6 lists the trainable parameters and schedules for the AFFIRM variants.

Stage B uses batch size one, weight decay $1 0 ^ { - 4 }$ gradient-norm clipping at 1, teacher temperature 1, and the top 32 teacher logits. The item-level switch uses a 0.1-nat mean gold-answer log-probability advantage. The sparse text distribution supervises the route field when text is better than the frozen selected parent by more than this threshold. Otherwise, the frozen-parent distribution supplies the preservation target. The route field is trained at unit output scale and deployed with $\beta = 0 . 2 5$

The stored learned-value counts are 3,584 for DIRECT PARENT, 458,752 for ROUTE FIELD, 458,753 for EXPERT PARENT, 462,336 for AFFIRM-DIRECT, 462,337 for AFFIRM-K1, and 917,505 for AFFIRM-EXPERT. Every realization edits one prompt-final state during the first generation step and requires no extra input under the speech-only contract.

## A.4 Frozen inference calibration

We fix all intervention strengths before test evaluation. Equal alpha values do not imply equal interventions across a norm-preserving vector and an affine field. We therefore compare learned operators through realized mean edit norms rather than raw alphas. The fixed AFFIRM-DIRECT alphas are 0.447348 on PopQA-20, 0.429581 on PopQA-60, and 0.289593 on TOFU. The corresponding EXPERT PARENT alphas are 0.869909, 0.760362, and 0.458232. The PopQA-20 AFFIRM-DIRECT alpha is reused unchanged for SLUE. The block-4 AFFIRM-EXPERT alphas are 1.2469 on PopQA-20, 1.0672 on PopQA-60, and 0.826551 on TOFU. Its PopQA-20 alpha is reused unchanged for SLUE. The DIRECT PARENT alphas are 0.464829, 0.453870, and 0.294240. For the matched Qwen operators, we calibrate to the same mean edit norm on development states.

For the compact cross-backbone characterization, the fixed Kimi-Audio and MiMo-Audio alphas are 0.601419 and 1.146747. We calibrate these strengths on 2,280 cached Kimi operatortraining states and 2,400 cached MiMo operatortraining states without using answer scores. The fixed target mean edit norms are 48.542453 for Kimi and 36.078838 for MiMo, taken from previously fitted affine operators on the respective backbones.

<table><tr><td>Setting</td><td>Acquisition</td><td>Operator train</td><td>Development</td><td>Test</td></tr><tr><td>Balanced PopQA</td><td>1,000</td><td>600</td><td>199</td><td>201</td></tr><tr><td>PopQA-20</td><td>14,267</td><td>2,853</td><td>2,848</td><td>2,847</td></tr><tr><td>PopQA-60</td><td>14,267</td><td>8,560</td><td>2,848</td><td>2,847</td></tr><tr><td>TOFU</td><td>4,000</td><td>2,400</td><td>800</td><td>800</td></tr><tr><td>SLUE-SQA-5 transfer</td><td>一</td><td>0</td><td>0</td><td>256</td></tr></table>

Table 5: Dataset and split accounting. Acquisition is the text-update population seen by the acquired checkpoint.
<table><tr><td>Stage or method</td><td>Trainable object</td><td>Epochs</td><td>Learning rate</td><td>Seed</td></tr><tr><td>Paired geometry</td><td>D and temporary readout</td><td>1,000 steps</td><td>0.001</td><td>5907</td></tr><tr><td>AFFIRM-DIRECT Stage A</td><td>Shared vector g</td><td>3</td><td>0.01</td><td>8307</td></tr><tr><td>AFFIRM-DIRECT Stage B</td><td>Local readout U</td><td>1</td><td>0.0001</td><td>6007</td></tr><tr><td>AFFIRM-EXPERT Stage A</td><td>Router, 64 experts, and scale</td><td>3</td><td>0.001/0.01/0.0001</td><td>8307</td></tr><tr><td>AFFIRM-EXPERT Stage B</td><td>Local readout U</td><td>1</td><td>0.0001</td><td>6007</td></tr><tr><td>AFFIRM-K1 Stage A</td><td>Direction e and scalar s</td><td>3</td><td>0.01/0.0001</td><td>8307</td></tr><tr><td>AFFIRM-K1 Stage B</td><td>Local readout U</td><td>1</td><td>0.0001</td><td>6007</td></tr></table>

Table 6: Main operator optimization. Only the listed objects are trainable, and the acquired checkpoint remains frozen.

## A.5 Decoding, metrics, uncertainty, and failures

Evaluation uses greedy decoding with at most 160 new tokens on PopQA and TOFU and at most 64 new tokens on SLUE. ROUGE-L is the recall form of the longest-common-subsequence score against the reference answer. Exact match lowercases text, strips punctuation and articles, normalizes whitespace, and then tests equality. SLUE answer-span accuracy credits a generation when its normalized text contains the normalized short reference answer. Retention divides the number of no-repair-correct items that remain exact under intervention by the total number of no-repair-correct items.

We report paired test differences with bootstrap standard errors (SEs), computed as the sample standard deviation of 10,000 group-bootstrap estimates with seed 3407. Balanced PopQA resamples duplicate-question groups. Full-scale PopQA resamples bound subject–question components for direct-parent and route-field contrasts, and retains the original relation-group resampling for expertparent contrasts. TOFU resamples authors, and SLUE resamples speakers. Inferential statements and bolding use paired percentile-bootstrap bounds, computed separately from the SEs. A difference is resolved when its 2.5th and 97.5th percentiles have the same nonzero sign. These SEs quantify testsample uncertainty, while Table 13 reports sample standard deviations across training seeds. All reported evaluations cover their stated sample counts with no failed generations.

## A.6 Software and hardware

Qwen experiments use PyTorch 2.3.1, a Transformers 4.52.0 development snapshot, qwen-omni-utils 0.0.9, NumPy 2.0.2, and Python 3.9.23. Model execution uses BF16 on NVIDIA A800-SXM4- 80GB accelerators, while learned operator tensors and calibration statistics are retained in FP32. Each training or generation run uses one accelerator. The operational timing study in Section E reports standalone end-to-end wall time and peak allocated CUDA memory for its matched runs.

## B Baseline Implementations and Inference Inputs

We reproduce the central operation of each evaluated baseline on the factual-access datasets and select task-specific settings using development data. Table 7 summarizes the training data, intervention mechanisms, and additional inference inputs.

DiReFT and LoReFT use one answer crossentropy epoch with learning rate $1 0 ^ { - 3 }$ , batch size one, and gradient-norm clipping at 1. Their full PopQA-20 source-development doses are 0.073351 and 0.212747. These doses are fixed before the operators are applied unchanged to all 256 SLUE items.

L2S fits paired text and speech states for 100 epochs with batch size 64, learning rate $5 \times 1 0 ^ { - 5 }$ a maximum cosine weight of 0.1, and an internal 8.33% validation fraction. Its balanced and full PopQA-20 score-blind alphas are 0.415363 and 0.463409. The full artifact is applied unchanged to SLUE.

The balanced-PopQA SALSA artifact is trained for three epochs at learning rate $5 \times 1 0 ^ { - 4 }$ and uses development-selected alpha 3.600604. That artifact and alpha produce only the 201-item balanced-PopQA result in Table 1. The SLUE row instead uses a full PopQA-20 artifact fitted on 2,853 operator-training examples. Score-blind calibration on those source speech states matches AFFIRM’s source mean edit norm of 25.448338 and selects alpha 2.012277. The frozen SALSA artifact then realizes mean edit norm 25.1334 on SLUE.

In a separate control, we recalibrate SALSA using unlabeled SLUE inputs to match AFFIRM’s mean edit norm on the target inputs. It selects SALSA alpha 1.886724 and matches AFFIRM’s realized SLUE mean edit norm to within 0.031%. SALSA remains below its paired No Repair replay, with RL 0.3759 versus 0.4437 and a difference of −0.0678 ± 0.0272 (bootstrap SE). Relative to source-calibrated SALSA, the dose reduction changes RL by only $+ 0 . 0 0 2 9 \pm 0 . 0 0 8 8$ (bootstrap SE). SALSA’s negative RL transfer persists after matching the mean edit norm on SLUE inputs.

Angle and Length use the paired text-question state at inference. On balanced PopQA development data, we select Angle over all coordinates and Length over the bottom three coordinates by RL with EM as the tie-breaker. The selected rule is then reused for balanced PopQA test and recorded SLUE. We apply the same development-only rule selection to TOFU using its author-disjoint split.

## C Statistical Evidence for Main Comparisons

This section reports paired differences with bootstrap SEs for the comparisons that carry the main empirical argument. Bold estimates satisfy the paired-bootstrap criterion in Appendix A.5. Each method-minus-baseline difference uses the No Repair outputs generated alongside that method unless a method-to-method contrast is stated explicitly.

## C.1 Balanced source-domain comparisons

Table 8 reports paired group-bootstrap comparisons on the balanced PopQA test set. We compute SEs from paired per-item score differences between AFFIRM-DIRECT and each comparator.

The paired-bootstrap analysis does not resolve the RL differences.

## C.2 No-refit recorded-speech transfer

Table 9 reports every speech-only method relative to the No Repair outputs generated alongside it. The AFFIRM/ReFT and L2S/SALSA evaluations generated these outputs separately. The archived AFFIRM-DIRECT and SALSA configurations specify the same acquired checkpoint, question manifest, system prompt, and generation limit, but their No Repair outputs differ on some items. The source of these differences has not been isolated, so each method’s paired improvement uses its own generated baseline rather than substituting the baseline from another run.

The paired-bootstrap analysis supports positive differences for the complete AFFIRM-DIRECT operator on all three recorded-speech metrics. DiReFT and LoReFT also improve over their paired baselines, while SALSA has a resolved negative RL difference. The bootstrap intervals for L2S’s differences from No Repair include zero on all three metrics.

## C.3 Full-scale component contrasts

Table 10 reports the paired differences and SEs behind the component analysis. Positive values favor AFFIRM-DIRECT.

For SLUE answer-span accuracy, AFFIRM-DIRECT minus ROUTE FIELD is $+ 0 . 0 6 2 5 0 \pm 0 . 0 2 1 6 3$ (bootstrap SE). The corresponding difference from EXPERT PARENT is +0.03516 ± 0.01835 (bootstrap SE).

<table><tr><td>Method</td><td>Training data</td><td>Mechanism and inference inputs</td></tr><tr><td>DiReFT (Wu et al., 2024)</td><td>600 balanced, 2,853 PopQA-20, or 2,400 TOFU rows</td><td>Rank-64 low-rank block-20 edit with a development-selected source dose</td></tr><tr><td>LoReFT (Wu et al., 2024)</td><td>600 balanced, 2,853 PopQA-20, or 2,400 TOFU rows</td><td>Rank-64 orthogonal low-rank block-20 edit with a development-selected source dose</td></tr><tr><td>L2S (Parekh et al., 2025)</td><td>600 balanced, 2,853 PopQA-20, or 2,400 TOFU paired states</td><td>Context-conditioned predictor with a 100-dimensional bottleneck, with intervention strength calibrated by matching edit norms on</td></tr><tr><td>SALSA (Yegorova et al., 2026)</td><td>600 balanced, 2,853 PopQA-20, or 2,400 TOFU rows</td><td>source states without answer scores One norm-preserving steering activation vector with a development-selected or source-state norm-matched dose</td></tr><tr><td>Angle (Xiang et al., 2025)</td><td>Paired speech and text development states only</td><td>No fitted operator, a paired text-question donor, and the development-selected all-coordinate angle rule</td></tr><tr><td>Length (Xiang et al., 2025)</td><td>Paired speech and text development states only</td><td>No fitted operator, a paired text-question donor, and the development-selected bottom-three length rule</td></tr></table>

Table 7: Baseline training data and inference inputs. Balanced PopQA, full PopQA-20, and TOFU use separately fitted artifacts. Angle and Length are donor-assisted controls because they receive the paired text question at inference.
<table><tr><td>Comparator</td><td> $\Delta \mathrm { R L } \pm \mathrm { S E }$ </td><td> $\Delta \mathrm { E M } \pm \mathrm { S E }$ </td></tr><tr><td>DiReFT</td><td> $+ 0 . 0 3 7 3 1 \pm 0 . 0 2 0 7 4$ </td><td> $+ 0 . 0 3 9 8 0 \pm 0 . 0 2 1 2 2$ </td></tr><tr><td>LoReFT</td><td> $+ 0 . 0 3 2 3 4 \pm 0 . 0 2 0 4 8$ </td><td> $+ 0 . 0 2 9 8 5 \pm 0 . 0 2 1 2 0$ </td></tr><tr><td>SALSA</td><td> $+ 0 . 0 1 9 9 0 \pm 0 . 0 1 6 0 4$ </td><td> $+ 0 . 0 1 9 9 0 \pm 0 . 0 1 7 1 9$ </td></tr><tr><td>EXPERT PARENT</td><td> $- 0 . 0 0 2 4 9 \pm 0 . 0 1 2 2 9$ </td><td> $- 0 . 0 0 4 9 8 \pm 0 . 0 1 3 2 5$ </td></tr></table>

Table 8: AFFIRM-DIRECT minus matched single-state comparators on balanced PopQA. We estimate SEs from 10,000 bootstrap resamples of duplicate-question groups.

## C.4 Block-4 repair comparisons

Table 11 compares block-4 AFFIRM-DIRECT with the block-4 AFFIRM-EXPERT and the block-20 expert parent on the same SLUE questions. Positive values favor AFFIRM-DIRECT.

The paired-bootstrap analysis supports a resolved exact-match advantage for AFFIRM-DIRECT on this joint shift, with positive point estimates in answer span and RL.

## D Full Results and Characterization

## D.1 Synthesized-question content retention

To test whether synthetic-speech states preserve question-specific content, we analyze the 600 paired text and CosyVoice speech prompt-final states captured for AFFIRM fitting at block 20. After subtracting the route-specific population mean, speech-to-text exact-pair retrieval reaches 0.5033 top-1, 0.6967 top-5, and 0.5979 mean reciprocal rank over all 600 candidates. Global top-1 chance is 0.0017. When candidates are restricted to the same PopQA relation, top-1 remains 0.5033 while item-averaged chance is 0.0267 with a median of 52 candidates. The reverse text-to-speech direction gives 0.5183 global and 0.5200 within-relation top-1. These results show that the internal speech state preserves substantial item-specific question identity rather than only the relation template. They rule out total loss of question identity as a sufficient explanation but do not isolate the cause of the remaining generation gap.

## D.2 Full-scale PopQA matrix

We evaluate block-4 AFFIRM-EXPERT on 2,847 test questions. A block-4 parent-only PopQA test result is unavailable, so this row does not provide a same-layer component ablation.

<table><tr><td>Method</td><td> $\Delta \mathrm { \ S p a n } \pm \mathrm { \ S E }$ </td><td> $\Delta \mathrm { R L } \pm \mathrm { S E }$ </td><td> $\Delta \mathrm { E M } \pm \mathrm { S E }$ </td></tr><tr><td>DiReFT</td><td> $\mathbf { + 0 . 0 5 4 6 9 \pm 0 . 0 1 7 2 0 }$ </td><td> $\mathbf { + 0 . 0 5 4 5 6 \pm 0 . 0 1 7 3 7 }$ </td><td> $\mathbf { + 0 . 0 3 9 0 6 \pm 0 . 0 1 3 8 8 }$ </td></tr><tr><td>LoReFT</td><td> $\mathbf { + 0 . 0 7 0 3 1 \pm 0 . 0 2 3 8 6 }$ </td><td> $\mathbf { + 0 . 0 7 7 3 4 \pm 0 . 0 2 2 2 5 }$ </td><td> $\mathbf { + 0 . 0 7 0 3 1 \pm 0 . 0 1 6 1 0 }$ </td></tr><tr><td>L2S</td><td> $- 0 . 0 1 9 5 3 \pm 0 . 0 2 1 9 7$ </td><td> $- 0 . 0 1 7 5 8 \pm 0 . 0 2 0 8 3$ </td><td> $+ 0 . 0 2 3 4 4 \pm 0 . 0 1 8 2 8$ </td></tr><tr><td>SALSA</td><td> $\mathbf { - 0 . 0 8 9 8 4 } \pm \mathbf { 0 . 0 3 0 8 1 }$ </td><td> $\mathbf { - 0 . 0 7 0 7 0 \pm 0 . 0 2 6 4 3 }$ </td><td> $+ 0 . 0 2 3 4 4 \pm 0 . 0 2 4 0 0$ </td></tr><tr><td>AFFIRM-DIRECT</td><td> $\mathbf { + 0 . 0 8 5 9 4 } \pm \mathbf { 0 . 0 2 5 0 7 }$ </td><td> $\mathbf { + 0 . 0 9 4 0 8 \pm 0 . 0 2 4 5 0 }$ </td><td> $\mathbf { + 0 . 0 6 6 4 1 \pm 0 . 0 1 9 4 4 }$ </td></tr></table>

Table 9: No-refit SLUE method minus its paired No Repair replay. We estimate SEs from 10,000 bootstrap resamples of the 50 speakers.
<table><tr><td>Setting</td><td>Comparator</td><td> $\Delta \mathrm { R L } \pm \mathrm { S E }$ </td><td> $\Delta \mathrm { E M } \pm \mathrm { S E }$ </td></tr><tr><td>PopQA-20</td><td>DIRECT PARENT</td><td> $\mathbf { + 0 . 0 1 9 7 4 } \pm \mathbf { 0 . 0 0 4 } 2 5$ </td><td> $\mathbf { + 0 . 0 1 5 1 0 \pm 0 . 0 0 4 4 2 }$ </td></tr><tr><td></td><td>ROUTE FIELD</td><td> $\mathbf { + 0 . 0 1 4 1 2 \pm 0 . 0 0 4 8 4 }$ </td><td> $\mathbf { + 0 . 0 0 9 8 3 \pm 0 . 0 0 4 9 5 }$ </td></tr><tr><td></td><td>EXPERT PARENT</td><td> $+ 0 . 0 0 0 8 3 \pm 0 . 0 0 3 6 3$ </td><td> $- 0 . 0 0 1 7 6 \pm 0 . 0 0 3 8 1$ </td></tr><tr><td>PopQA-60</td><td>DIRECT PARENT</td><td> $\mathbf { + 0 . 0 2 2 3 2 \pm 0 . 0 0 4 9 4 }$ </td><td> $\mathbf { + 0 . 0 1 8 2 6 \pm 0 . 0 0 5 1 1 }$ </td></tr><tr><td></td><td>ROUTE FIELD</td><td> $\mathbf { + 0 . 0 1 1 3 5 \pm 0 . 0 0 5 0 7 }$ </td><td> $+ 0 . 0 0 6 6 7 \pm 0 . 0 0 5 2 1$ </td></tr><tr><td></td><td>EXPERT PARENT</td><td> $+ 0 . 0 0 2 2 6 \pm 0 . 0 0 4 9 4$ </td><td> $\mathbf { + 0 . 0 1 0 5 4 } \pm \mathbf { 0 . 0 0 6 0 0 }$ </td></tr><tr><td>TOFU</td><td>DIRECT PARENT</td><td> $+ 0 . 0 0 5 7 7 \pm 0 . 0 0 3 6 4$ </td><td> $+ 0 . 0 0 6 2 5 \pm 0 . 0 0 4 0 4$ </td></tr><tr><td></td><td>ROUTE FIELD</td><td> $+ 0 . 0 0 4 5 4 \pm 0 . 0 0 5 3 0$ </td><td> $- 0 . 0 0 1 2 5 \pm 0 . 0 0 4 4 8$ </td></tr><tr><td></td><td>EXPERT PARENT</td><td> $+ 0 . 0 0 3 6 9 \pm 0 . 0 0 5 6 1$ </td><td> $- 0 . 0 0 1 2 5 \pm 0 . 0 0 4 1 2$ </td></tr><tr><td>SLUE</td><td>ROUTE FIELD</td><td> $\mathbf { + 0 . 0 6 2 8 3 \pm 0 . 0 2 1 4 3 }$ </td><td> $+ 0 . 0 3 5 1 6 \pm 0 . 0 1 8 0 5$ </td></tr><tr><td></td><td>EXPERT PARENT</td><td> $+ 0 . 0 2 5 9 1 \pm 0 . 0 1 5 8 5$ </td><td> $+ 0 . 0 3 5 1 6 \pm 0 . 0 1 9 1 5$ </td></tr></table>

Table 10: AFFIRM-DIRECT minus full-scale component and parent comparators. SEs use the group-bootstrap procedures in Appendix A.5.

## D.3 Stage-B seed stability

Table 13 summarizes development performance across three Stage-B training runs for PopQA-20 and TOFU.

PopQA-20 development scores are stable across the three Stage-B optimizer seeds at a fixed intervention strength. For TOFU, each seed uses an intervention strength calibrated to the same target mean edit norm.

## D.4 Compact-parent depth characterization

We evaluate AFFIRM-K1 across decoder blocks using two training seeds and 800 TOFU development items, as summarized in Section 7. The split, K = 1 parent parameterization, local coefficient, and target edit norm are fixed across blocks 4 through 28.

Mean full-field RL declines from 0.5550 at block 4 to 0.4788 at block 28. The route field changes RL by at most 0.0068 at any block and seed, its twoseed average never exceeds 0.0025, and its sign varies across blocks and seeds. The parent alone also peaks at block 4 in both seeds, so the route field does not determine the selected site.

## D.5 Cross-backbone characterization

Table 14 compares the compact parent-plus-routefield variant across speech-language backbones. Each backbone uses its own frozen text-acquired checkpoint, with intervention strengths calibrated on the states specified in Appendix A. The local coefficient $\beta = 0 . 2 5$ is transferred without backbonespecific retuning.

We evaluate the selected Kimi operator on a separate test set. Its full field improves test RL and EM from 0.6321/0.0850 to 0.6451/0.1188 and raises 233 item-level ROUGE-L scores. A second training seed on Kimi-Audio reaches development RL/EM of 0.5992/0.1325 against the same 0.5655/0.0875 baseline. MiMo is mixed, with a 0.0125 EM increase and a 0.0044 RL decrease relative to No Repair.

## D.6 Geometric and capacity controls

To assess query-dependent geometry and model capacity, we compare a constant route mean, an answer-free low-rank route map, and a highcapacity kernel regressor on balanced PopQA (Table 15). All nonzero edits operate at block 20 and use the same speech-only deployment interface.

<table><tr><td>Comparator</td><td> $\Delta \mathrm { S p a n } \pm \mathrm { S E }$ </td><td> $\Delta \mathrm { R L } \pm \mathrm { S E }$ </td><td> $\Delta \mathrm { E M } \pm \mathrm { S E }$ </td></tr><tr><td>EXPERT PARENT (block 20)</td><td> $+ 0 . 0 4 3 0 \pm 0 . 0 2 5 5$ </td><td> $+ 0 . 0 2 1 8 \pm 0 . 0 2 1 6$ </td><td> $\mathbf { + 0 . 1 0 9 4 \pm 0 . 0 2 5 1 }$ </td></tr><tr><td>AFFIRM-EXPERT (block 4)</td><td> $+ 0 . 0 1 9 5 \pm 0 . 0 2 8 2$ </td><td> $+ 0 . 0 0 6 0 \pm 0 . 0 2 6 6$ </td><td> $\mathbf { + 0 . 0 4 3 0 \pm 0 . 0 1 8 5 }$ </td></tr></table>

Table 11: Block-4 AFFIRM-DIRECT minus its comparators on no-refit SLUE. We estimate SEs from 10,000 bootstrap resamples of the 50 speakers. The expert-parent row uses a different intervention block.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Block</td><td colspan="2">PopQA-20</td><td colspan="2">PopQA-60</td></tr><tr><td>RL</td><td>EM</td><td>RL</td><td>EM</td></tr><tr><td>No Repair</td><td>一</td><td>0.1787</td><td>0.1500</td><td>0.1787</td><td>0.1500</td></tr><tr><td>DIRECT PARENT</td><td>20</td><td>0.3156</td><td>0.2986</td><td>0.3269</td><td>0.3087</td></tr><tr><td>ROUTE FIELD</td><td>20</td><td>0.3212</td><td>0.3038</td><td>0.3379</td><td>0.3203</td></tr><tr><td>EXPERT PARENT</td><td>20</td><td>0.3345</td><td>0.3154</td><td>0.3470</td><td>0.3165</td></tr><tr><td>AFFIRM-DIRECT</td><td>20</td><td>0.3353</td><td>0.3137</td><td>0.3493</td><td>0.3270</td></tr><tr><td>AFFIRM-EXPERT</td><td>4</td><td>0.3727</td><td>0.3562</td><td>0.3871</td><td>0.3709</td></tr></table>

Table 12: Full-scale PopQA factual access. PopQA-20 and PopQA-60 use operator-training budgets of approximately 20% and 60% of the full acquisition population and share the same locked test set. Block is shown because only same-block rows are controlled architectural comparisons.

The route map achieves higher RL and EM than the mean shift on balanced PopQA. Residual KRR stores 9.4 times more learned values than the route map but does not improve its test scores. The remaining AFFIRM-DIRECT gap is consistent with decoder-facing training contributing beyond representation-only mapping.

## E Comparison with Speech Fine-Tuning and ASR

We compare external repair operators with alternatives that update model parameters or add an ASR pass.

## E.1 Matched-data direct speech fine-tuning

Table 16 summarizes a development comparison between a decoder-trained route map and direct Speech LoRA. The four data budgets use nested training subsets and the same 199-item development set. We evaluate this sweep only on development data. The route map first fits rank-64 pairedstate geometry for 1,000 steps, then updates its readout for one epoch at learning rate 0.001 while retaining the empirical mean route difference. Its selective distillation uses the text route as the correction teacher and the unmodified speech route as the preservation teacher. This variant has no separately trained access parent, so its resource measurements describe the route map rather than the staged AFFIRM procedure in Section 4. We calibrate its intervention strength without answer scores to the same mean edit norm at every budget. Speech LoRA uses rank-16 adapters in all 28 decoder blocks and a common five-epoch cosine schedule. For each Speech LoRA budget, the table reports the development checkpoint selected by RL with EM as the tie-breaker.

After selecting among five development checkpoints, Speech LoRA is at least as accurate at every unique-pair budget. At 150 pairs and approximately matched wall time, the route map reaches 0.5720/0.5628 RL/EM in 202.6 seconds, while the first Speech LoRA epoch reaches 0.5695/0.5578 in 197.8 seconds. The route map stores fewer learned values, while Speech LoRA uses less peak training memory at every budget and trains faster at the largest budget.

## E.2 Test accuracy and deployment requirements

Table 17 compares test accuracy and deployment requirements on balanced PopQA, PopQA-60, and TOFU.

Broader adaptation and extra inference resources can reach higher accuracy than the detachable onestate operator.

<table><tr><td>Method and setting</td><td>Development RL</td><td>Development EM</td><td>Mean edit norm</td></tr><tr><td>AFFIRM-DIRECT, PopQA-20</td><td>0.3462 ± 0.0018</td><td> $0 . 3 2 4 6 \pm 0 . 0 0 1 7$ </td><td>25.469±0.020</td></tr><tr><td>AFFIRM-EXPERT, TOFU</td><td> $0 . 5 6 1 9 \pm 0 . 0 0 1 9$ </td><td> $0 . 1 5 7 1 \pm 0 . 0 0 3 1$ </td><td> $\approx 1 9 . 2 3 2$ </td></tr></table>

Table 13: Stage-B stability across three training seeds. RL and EM are reported as mean ± sample standard deviation.

<table><tr><td></td><td></td><td></td><td colspan="2">No Repair</td><td colspan="2">Parent</td><td colspan="2">Full field</td></tr><tr><td>Backbone</td><td>Split</td><td>Block</td><td>RL</td><td>EM</td><td>RL</td><td>EM</td><td>RL</td><td>EM</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>Development</td><td>4</td><td>0.4695</td><td>0.0713</td><td>0.5575</td><td>0.1613</td><td>0.5585</td><td>0.1563</td></tr><tr><td>Kimi-Audio</td><td>Development</td><td>20</td><td>0.5655</td><td>0.0875</td><td>0.5973</td><td>0.1338</td><td>0.5960</td><td>0.1350</td></tr><tr><td>Kimi-Audio</td><td>Test</td><td>20</td><td>0.6321</td><td>0.0850</td><td></td><td></td><td>0.6451</td><td>0.1188</td></tr><tr><td>MiMo-Audio</td><td>Development</td><td>25→36</td><td>0.7175</td><td>0.2450</td><td>0.7113</td><td>0.2563</td><td>0.7130</td><td>0.2575</td></tr></table>

Table 14: AFFIRM-K1 characterization across speech-language backbones on TOFU. Each reported score is computed over 800 questions. Development rows show No Repair, the parent, and the full field for each backbone. For Kimi, we also report test results after model selection on development data.

<table><tr><td>Method</td><td>Learned values</td><td>RL</td><td>EM</td></tr><tr><td>No Repair</td><td>0</td><td>0.2736</td><td>0.2189</td></tr><tr><td>Mean route shift</td><td>3,584</td><td>0.5323</td><td>0.5224</td></tr><tr><td>Representation-only route map</td><td>458,752</td><td>0.5808</td><td>0.5672</td></tr><tr><td>Residual KRR (Saunders et al., 1998)</td><td>4,300,800</td><td>0.5808</td><td>0.5672</td></tr><tr><td>AFFIRM-DIRECT</td><td>462,336</td><td>0.6318</td><td>0.6219</td></tr></table>

Table 15: Representation controls on the 201-item balanced PopQA test. The mean edit norms are 22.51 for the controls and 22.62 for AFFIRM-DIRECT.

<table><tr><td>Method</td><td>Unique pairs</td><td>Selected epoch</td><td>RL</td><td>EM</td><td>Time (s)</td><td>Peak GiB</td><td>Learned values</td></tr><tr><td>Route map</td><td>75</td><td>1</td><td>0.5343</td><td>0.5226</td><td>123.0</td><td>21.31</td><td>0.46M</td></tr><tr><td>Speech LoRA</td><td>75</td><td>3</td><td>0.5419</td><td>0.5276</td><td>267.4</td><td>17.61</td><td>40.37M</td></tr><tr><td>Route map</td><td>150</td><td>一</td><td>0.5720</td><td>0.5628</td><td>202.6</td><td>21.42</td><td>0.46M</td></tr><tr><td>Speech LoRA</td><td>150</td><td>2</td><td>0.5972</td><td>0.5829</td><td>339.9</td><td>17.69</td><td>40.37M</td></tr><tr><td>Route map</td><td>300</td><td>一</td><td>0.5494</td><td>0.5327</td><td>468.7</td><td>21.42</td><td>0.46M</td></tr><tr><td>Speech LoRA</td><td>300</td><td>3</td><td>0.6323</td><td>0.6231</td><td>575.5</td><td>17.69</td><td>40.37M</td></tr><tr><td>Route map</td><td>600</td><td>一</td><td>0.5771</td><td>0.5628</td><td>951.4</td><td>21.42</td><td>0.46M</td></tr><tr><td>Speech LoRA</td><td>600</td><td>2</td><td>0.6323</td><td>0.6281</td><td>913.8</td><td>17.68</td><td>40.37M</td></tr></table>

Table 16: Matched-data development comparison of a decoder-trained route map and Speech LoRA. The route map uses the empirical mean route difference in place of AFFIRM’s trained access parent and the unmodified speech output as its preservation target. Its learned-value count includes both projection matrices and the empirical mean route difference. Time is standalone end-to-end training time in seconds and excludes development generation. Speech LoRA checkpoint times are reconstructed from checkpoint timestamps and stored full-run duration, so they are approximate.

<table><tr><td>Setting</td><td>Method</td><td>Pairs</td><td>Trainable</td><td>Deployed change</td><td>RL</td><td>EM</td></tr><tr><td rowspan="6">Balanced PopQA</td><td>No Repair</td><td>0</td><td>0</td><td>None</td><td>0.2736</td><td>0.2189</td></tr><tr><td>AFFIRM-DIRECT</td><td>600</td><td>0.46M</td><td>One-state external operator</td><td>0.6318</td><td>0.6219</td></tr><tr><td>Speech LoRA</td><td>600</td><td>40.37M</td><td>Internal adapters in 28 blocks</td><td>0.6443</td><td>0.6368</td></tr><tr><td>ASR cascade</td><td>0</td><td>0</td><td>External ASR and a second text pass</td><td>0.6990</td><td>0.6915</td></tr><tr><td>Full speech supervision</td><td>1,000</td><td>Full model</td><td>Full-model update</td><td>0.9801</td><td>0.9801</td></tr><tr><td>No Repair</td><td>0</td><td>0</td><td>None</td><td>0.1787</td><td>0.1500</td></tr><tr><td rowspan="4"></td><td>AFFIRM-DIRECT</td><td>8,560</td><td>0.46M</td><td>One-state external operator</td><td>0.3493</td><td>0.3270</td></tr><tr><td>Speech LoRA</td><td>8,560</td><td>40.37M</td><td>Internal adapters in 28</td><td>0.4962</td><td>0.4830</td></tr><tr><td>S2T-KD</td><td>8,560</td><td>40.37M</td><td>blocks Internal adapters in 28 blocks</td><td>0.4943</td><td>0.4805</td></tr><tr><td>No Repair</td><td>0</td><td>0</td><td>None</td><td>0.5003</td><td>0.1000</td></tr><tr><td rowspan="4">TOFU</td><td>AFFIRM-DIRECT</td><td>2,400</td><td>0.46M</td><td>One-state external</td><td>0.5310</td><td>0.1200</td></tr><tr><td>Speech LoRA</td><td>2,400</td><td>40.37M</td><td>operator Internal adapters in 28</td><td>0.6509</td><td>0.2513</td></tr><tr><td>S2T-KD</td><td>2,400</td><td>40.37M</td><td>blocks Internal adapters in 28</td><td>0.6504</td><td>0.2275</td></tr><tr><td></td><td></td><td></td><td>blocks</td><td></td><td></td></tr></table>

Table 17: Accuracy and mutability across frozen and relaxed deployment contracts. All scores are computed on the full test splits. Speech LoRA and S2T-KD insert rank-16 adapters throughout all 28 decoder blocks, and S2T-KD adds a temperature-2 forward-KL term with weight 0.5 toward the text-route output distribution. The ASR cascade transcribes each spoken question with Whisper-large-v3 and answers the transcript with the acquired checkpoint.