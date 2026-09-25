# ADAPTIVE FISHER-WHITENED CROSS-COVARIANCE FOR LOW-RESOURCE SPEECHRECOGNITION

Asmee Mishra<sup>1</sup>, Mengjie Qian<sup>1</sup>, Brechtje Post<sup>2</sup>, Kate Knill<sup>1</sup>

<sup>1</sup>Department of Engineering, University of Cambridge, United Kingdom <sup>2</sup>Phonetics Laboratory, University of Cambridge, United Kingdom

## ABSTRACT

Adapting multilingual speech foundation models to low-resource languages remains difficult, especially for languages that are poorly represented during pre-training. While parameter-efficient finetuning (PEFT) reduces the cost of adapting large models, conventional approaches such as LoRA rely on generic low-rank parameterizations and do not explicitly use downstream task information to define the adaptation subspace. To investigate whether taskinformed PEFT can better support low-resource ASR, we apply Fisher-Whitened Cross-Covariance Analysis (FCCA) to Whisper and Qwen3-ASR, and introduce two complementary extensions: Asymmetric-Coupled FCCA (AC-FCCA), which exploits structured cross-layer sharing, and Adaptive-Rank FCCA (AR-FCCA), which reallocates adaptation capacity across projection matrices under a fixed parameter budget. Under controlled multilingual experiments, we evaluate these approaches on languages that are poorly represented or unsupported during pre-training alongside wellrepresented languages. Standard FCCA is competitive with, and usually outperforms, trainable-parameter-budget-matched LoRA. AR-FCCA provides the most consistent improvement over standard FCCA across both model architectures, with statistically significant gains in several evaluation settings, while retaining the same number of trainable parameters. These results show that task-informed subspace construction can be effective for low-resource speech adaptation, and that adaptive rank allocation provides a robust way to improve parameter efficiency without increasing model capacity.

Index Terms— Low-resource ASR, parameter-efficient finetuning, Fisher-Whitened Cross-Covariance, Whisper, Qwen3-ASR

## 1. INTRODUCTION

Speech foundation models, such as Whisper [1] and Qwen3- ASR [2], have substantially improved multilingual automatic speech recognition (ASR), allowing a single model to recognise dozens of languages. However, many languages remain underrepresented during pre-training, including low-resource, endangered and regional languages, for which only a few hours of labelled speech are available. Developing speech technologies for these languages benefits practical applications, linguistic research and language documentation [3, 4]. In particular, speech technologies can accelerate the documentation, annotation, and analysis of endangered languages, providing scalable tools for phonetic, phonological, and sociolinguistic studies. Consequently, efficiently adapting speech foundation models with limited labelled data has become an important research problem for multilingual low-resource ASR.

Full fine-tuning remains an effective adaptation strategy but requires updating all model parameters, resulting in substantial computational and memory costs. Parameter-efficient fine-tuning (PEFT) addresses this limitation by adapting only a small subset of parameters while maintaining competitive performance. Representative approaches include adapter tuning [5, 6], prefix tuning [7], prompt tuning [8], and Low-Rank Adaptation (LoRA) [9], with LoRA becoming one of the dominant approaches for adapting speech foundation models. Recent studies have further investigated continual language learning [10] and source language adaptation for low-resource ASR [11]. Recent PEFT methods have increasingly exploited downstream data to identify more informative adaptation directions [12, 13, 14, 15], for example through gradient-informed initialisation in LoRA-GA [16] and activation-based subspace selection and rank allocation in EVA [17]. Fisher-Whitened Cross-Covariance Analysis (FCCA) takes this further by deriving paired low-rank adaptation subspaces from Fisher-whitened input–error statistics [18]. This task-informed PEFT has shown promising results for large language models (LLMs), but its effectiveness for speech foundation models, particularly under low-resource adaptation, remains unexplored.

This work investigates whether task-informed PEFT transfers effectively to multilingual low-resource ASR by adapting FCCA to speech foundation models. The original FCCA study focuses on decoder-only autoregressive LLMs (i.e. Qwen2.5-3B-Instruct [19]), whereas speech models such as Whisper use an encoder–decoder architecture that must first learn representations from continuous acoustic input. These differences make it unclear whether the same task-informed subspace construction will behave similarly for speech recognition. Building upon FCCA, we propose two complementary approaches for improving task-informed adaptation. Asymmetric-Coupled FCCA (AC-FCCA) investigates whether structured cross-layer parameter sharing can improve adaptation efficiency, while Adaptive-Rank FCCA (AR-FCCA) redistributes adaptation capacity across projection matrices under a fixed parameter budget. These two approaches investigate different aspects of task-informed PEFT: how adaptation subspaces can be shared across layers, and how adaptation capacity can be distributed across matrices.

In this work, we evaluate the proposed methods on low-resource ASR across multiple languages using Whisper and Qwen3-ASR, chosen to cover different model architectures and their strong performance on multilingual ASR. Experimental results demonstrate that AR-FCCA consistently improves standard FCCA across diverse multilingual benchmarks while remaining competitive with strong PEFT baselines. The main contributions of this work are: (1) the first systematic investigation of task-informed PEFT (i.e. FCCA) for multilingual low-resource ASR; (2) proposing two complementary approaches, AC-FCCA and AR-FCCA, extending FCCA through complementary cross-layer sharing and adaptive-rank allocation strategies; and (3) extensive experiments across multiple languages and two speech foundation models.

## 2. PROPOSED METHODS

Ultra-low-parameter PEFT is attractive because it reduces trainable and optimiser-state memory, but at such small budgets the quality of the chosen update subspace becomes critical. Several LoRA variants use task-dependent statistics to improve the initial adaptation subspace. LoRA-GA [16] uses downstream gradients, EVA [17] uses activation statistics, and CorDA [12] uses task-conditioned activation covariance to orient a decomposition of the pretrained weights, among other approaches [13, 14, 15]. However, these retain LoRAsized trainable factors. FCCA [18] instead fixes task-informed left and right bases and trains only a small core matrix, yielding a substantially smaller adaptation budget. We first review FCCA and then introduce two complementary extensions: AC-FCCA for structured cross-layer sharing and AR-FCCA for adaptive rank allocation.

## 2.1. Fisher-Whitened Cross-Covariance Adaptation (FCCA)

For an adapted weight matrix $\boldsymbol { W } \in \mathbb { R } ^ { m \times n }$ , FCCA parameterises the update as

$$
\boldsymbol { W } ^ { \prime } = \boldsymbol { W } + s \boldsymbol { P } \boldsymbol { R } \boldsymbol { Q } ^ { \intercal } ,\tag{1}
$$

where $\boldsymbol { P } \in \mathbb { R } ^ { m \times r }$ and $Q \in \mathbb { R } ^ { n \times r }$ are fixed task-informed bases and only $R \in \mathbb { R } ^ { r \times r }$ is trainable.

Given layer input x and back-propagated output error δ, FCCA estimates

$$
G = \mathbb { E } [ \delta x ^ { \top } ] , A = \mathrm { d i a g } ( \mathbb { E } [ x ^ { \odot 2 } ] ) + \epsilon I , D = \mathrm { d i a g } ( \mathbb { E } [ \delta ^ { \odot 2 } ] ) + \epsilon I ,
$$

and forms the Fisher-whitened cross-covariance $\tilde { G } = D ^ { - 1 / 2 } G A ^ { - 1 / 2 }$ With the rank-r decomposition $\tilde { G } \approx U _ { r } \Sigma _ { r } V _ { r } ^ { \top }$ , the bases are obtained by inverse whitening, $P _ { 0 } ~ = ~ D ^ { - 1 / 2 } U _ { r } , Q _ { 0 } ~ = ~ A ^ { - 1 / 2 } V _ { r } ,$ followed by thin QR orthogonalisation.

Unlike LoRA, whose trainable parameter count scales with $r ( m + n )$ , FCCA trains only the $\cdot ^ { 2 }$ parameters in R, where $r \ll m , n .$ . This frozen-core formulation is similar to LoRA-XS [20], but differs in how the fixed bases are constructed: LoRA-XS uses singular directions from the pretrained weight spectrum, whereas FCCA derives task-informed directions from downstream statistics. On LLM benchmarks, FCCA has shown stronger matched-budget performance than LoRA-XS-based alternatives [18].

## 2.2. Asymmetric-Coupled FCCA (AC-FCCA)

Standard FCCA constructs independent left and right adaptation subspaces for each matrix, although neighbouring Transformer layers may share task-relevant structure. We quantify cross-layer alignment between orthonormal rank-r bases X and Y by

$$
S ( X , Y ) = r ^ { - 1 } \| X ^ { \top } Y \| _ { F } ^ { 2 } ,
$$

with larger values indicating greater shared subspace structure.

Our preliminary analysis shows projection-dependent asymmetry: $q / k / v$ projections align more strongly across layers on the input side, whereas o aligns more strongly on the output side, with alignment strongest between adjacent layers. AC-FCCA therefore chooses to couple only this better-aligned side across adjacent layers while keeping the opposite side layer-specific.

Asymmetric one-sided FCCA (AC-FCCA-H). For each adjacent-layer pair $( \ell , \ell + 1 )$ , we share a rank-r basis on the betteraligned side. For $q / k / v$

$$
V = \mathrm { T o p E i g } _ { r } \left( \sum _ { j \in \{ \ell , \ell + 1 \} } \tilde { G } _ { j } ^ { \top } \tilde { G } _ { j } \right) , \qquad U _ { j } = \mathrm { o r t h } ( \tilde { G } _ { j } V ) ,
$$

whereas for o,

$$
U = \mathrm { T o p E i g } _ { r } \left( \sum _ { j \in \{ \ell , \ell + 1 \} } \tilde { G } _ { j } \tilde { G } _ { j } ^ { \intercal } \right) , \qquad V _ { j } = \mathrm { o r t h } ( \tilde { G } _ { j } ^ { \intercal } U ) .
$$

The shared basis is defined in Fisher-whitened coordinates. After layer-specific FCCA unwhitening and QR orthogonalisation, each matrix obtains native bases $P _ { j } , Q _ { j }$ and $\Delta W _ { j } = \bar { P _ { j } } R _ { j } Q _ { i } ^ { \top }$

Shared–private asymmetric FCCA (AC-FCCA-SP). To relax full one-sided sharing, the coupled rank-128 subspace is split into $r _ { s } = 6 4$ shared and $r _ { p } = 6 4$ private directions.

Define

$$
C _ { j } = \left\{ \begin{array} { l l } { \widetilde { G } _ { j } ^ { \top } \widetilde { G } _ { j } , } & { q / k / v , } \\ { \widetilde { G } _ { j } \widetilde { G } _ { j } ^ { \top } , } & { o . } \end{array} \right.
$$

For each adjacent-layer pair, shared basis S and private bases $P _ { j }$ maximise

$$
J = \sum _ { j } \left[ \mathrm { t r } ( \boldsymbol { S } ^ { \top } C _ { j } \boldsymbol { S } ) + \mathrm { t r } ( P _ { j } ^ { \top } C _ { j } P _ { j } ) \right] ,\tag{2}
$$

subject to $S ^ { \top } S = P _ { j } ^ { \top } P _ { j } = I$ and $S ^ { \top } P _ { j } = 0$ . We optimise this objective by block-coordinate ascent, alternating

$$
P _ { j } = \mathrm { T o p E i g } _ { r _ { p } } \left( \Pi _ { S } ^ { \perp } C _ { j } \Pi _ { S } ^ { \perp } \right) , \qquad \Pi _ { S } ^ { \perp } = I - S S ^ { \top } ,
$$

with

$$
S = \mathrm { T o p E i g } _ { r _ { s } } \left( \Pi _ { P _ { \bigcup } } ^ { \perp } \left( \sum _ { j } C _ { j } \right) \Pi _ { P _ { \bigcup } } ^ { \perp } \right) ,
$$

where $P _ { \cup }$ is the numerical union span of the private bases. Updates alternate to convergence; the shared and private bases are then concatenated, and the opposite-side basis is recovered as in AC-FCCA-H.

## 2.3. Adaptive-Rank FCCA (AR-FCCA)

Recent adaptive-rank LoRA methods relax the assumption of a uniform rank across weight matrices, instead allocating the rank budget according to matrix importance [21, 22]. Motivated by this idea, we extend FCCA with matrix-specific ranks while preserving the same total trainable-core budget. FCCA naturally provides a task-informed criterion for rank allocation. For matrix ℓ, let $\tilde { G } _ { \ell } = U _ { \ell } \Sigma _ { \ell } V _ { \ell } ^ { \top }$ . Under the FCCA local quadratic approximation, the utility of the optimal rank-r update is proportional to the retained Fisher-whitened spectral energy $\begin{array} { r } { \dot { E } _ { \ell } ( r ) \stackrel { \cdot } { = } \dot { \sum _ { k = 1 } ^ { r } } \sigma _ { \ell , k } ^ { 2 } } \end{array}$ , where $\sigma _ { \ell , k }$ is the k-th singular value of ${ \cal \tilde { G } } _ { \ell } .$ Since the trainable FCCA core for matrix ℓ contains $r _ { \ell } ^ { 2 }$ parameters, AR-FCCA selects matrix-specific ranks under a fixed global parameter budget:

$$
\operatorname* { m a x } _ { \{ r _ { \ell } \} } \sum _ { \ell = 1 } ^ { L } E _ { \ell } ( r _ { \ell } ) \quad \mathrm { s . t . } \quad \sum _ { \ell = 1 } ^ { L } r _ { \ell } ^ { 2 } = B , \qquad B = L r _ { 0 } ^ { 2 } ,\tag{3}
$$

where $r _ { 0 }$ is the uniform FCCA baseline rank. This reallocates adaptation capacity toward matrices that retain more task-relevant spectral energy per additional trainable parameter, while keeping the overall core budget unchanged. At the selected rank, the standard FCCA construction is applied: $P _ { \ell } = \mathrm { q r } ( D _ { \ell } ^ { - 1 / 2 } U _ { \ell , : r _ { \ell } } ) , \qquad Q _ { \ell } =$ $\mathrm { q r } ( A _ { \ell } ^ { - 1 / 2 } V _ { \ell , : r _ { \ell } } )$ . The bases remain frozen and only $R _ { \ell } \in \mathbb { R } ^ { r _ { \ell } \times r _ { \ell } }$ is trained, giving $\Delta W _ { \ell } = P _ { \ell } R _ { \ell } Q _ { \ell } ^ { \top }$ . In all experiments, $r _ { 0 } = 1 2 8 .$ and AR-FCCA exactly matches the total trainable-core budget of uniform FCCA.

## 3. EXPERIMENTAL SETUP

Datasets. Experiments use five FLEURS languages [23], spanning low-resource targets and better-represented multilingual controls. Asturian, Sorani Kurdish, and Kyrgyz form the primary low-resource evaluation, while Mandarin and Persian test whether the observed behaviour extends to better-represented languages. For Whisper, unseen languages use tokens from closely related languages. All adaptation uses only the target-language FLEURS training split (7.5–10.5 h), with hyperparameters selected on validation and final results reported on the held-out test set.

Table 1: FLEURS dataset statistics (hours) for target languages.
<table><tr><td>Language</td><td>Lang. Token</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>Asturian</td><td>&lt;|es|&gt;</td><td>7.5</td><td>0.9</td><td>2.4</td></tr><tr><td>Sorani Kurdish</td><td>&lt;|fa|&gt;</td><td>10.5</td><td>1.2</td><td>3.0</td></tr><tr><td>Kyrgyz</td><td>&lt;|kk|&gt;</td><td>9.3</td><td>1.3</td><td>3.2</td></tr><tr><td>Mandarin</td><td>&lt;|zh|&gt;</td><td>9.7</td><td>1.3</td><td>3.1</td></tr><tr><td>Persian</td><td>《|fa|&gt;</td><td>10.0</td><td>1.5</td><td>3.7</td></tr></table>

Models. Experiments use Whisper medium [1] and Qwen3- ASR-1.7B [2]. Whisper medium is a 769M-parameter encoder– decoder model with cross-attention, whereas Qwen3-ASR combines an audio encoder with an autoregressive LLM-style decoder. Whisper is evaluated on all five languages; Qwen3-ASR provides crossmodel validation on Asturian and Sorani Kurdish, neither of which is among its supported languages.

Baselines and compared approaches. The following methods are used as baselines in the experiments: Vanilla, using the pretrained model directly without any fine-tuning; full fine-tuning (FFT), updating all model parameters; LoRA [9], a widely-used PEFT baseline. These are compared with standard FCCA [18] and the proposed AC-FCCA and AR-FCCA variants.

Adaptation Configurations. For Whisper, LoRA and FCCAbased methods adapt the $q / k / v / o$ projections in encoder selfattention, decoder self-attention, and decoder cross-attention (288 matrices); for Qwen3-ASR, the corresponding audio-encoder and text-decoder projections are adapted (208 matrices). FCCA and its variants use 256 calibration examples sampled from the training set with the same seed, and base rank $r \ = \ 1 2 8$ , training 0.61% and 0.20% of Whisper-medium and Qwen3-ASR parameters, respectively. Calibration is a one-off cost, requiring approximately 20, 40, 90, and 28 s for FCCA, AC-FCCA-H, AC-FCCA-SP, and AR-FCCA, respectively, on an RTX 6000 GPU.

Training Configuration. Hyperparameters are selected by a small sequential validation-set search. For Whisper-medium, FFT searches learning rates in $\left[ 4 , 7 \right] \times 1 0 ^ { - 5 }$ with effective batch size (EBS) 32, LoRA in $\left[ 1 , 4 \right] \times \bar { 1 } 0 ^ { - 4 }$ with EBS 32, and FCCA in $[ 1 , 1 5 ] \times 1 0 ^ { - 4 }$ with EBS 8. FCCA uses r “ 128, while LoRA uses $r = 1 2 8$ $r = 8 ,$ , and $r = 6 .$ . All systems use AdamW, weight decay 0.01, linear warm-up ratio 0.05, gradient clipping at 1.0, bf16 precision, and at most 12 epochs with early stopping. After selecting the best standard FCCA configuration, its optimisation hyperparameters are reused unchanged for AC-FCCA-H, AC-FCCA-SP, and AR-FCCA, avoiding variant-specific tuning.

Decoding Configuration. All systems use deterministic greedy decoding (beam size 1) without an external language model. The target language and transcription task are specified at inference, and the repetition guard described below is applied during generation.

Post-processing and Evaluation. Before scoring, references and hypotheses undergo NFKC normalization [24], case-folding, punctuation/symbol and control-character removal, Unicode-digit canonicalization, and whitespace collapse. Persian additionally canonicalizes Arabic-script variants for evaluation. Mandarin hypotheses are converted from Traditional to Simplified Chinese with OpenCC [25]; whitespace is then removed from references and hypotheses before CER computation. We report CER for Mandarin and WER for all other languages. To suppress decoding hallucinations, generation terminates when any token cycle of length 1–8 repeats six consecutive times.

## 4. RESULTS AND DISCUSSION

## 4.1. Whisper Baseline Performance

We first present the baseline experiments on Whisper under different adaptation strategies. As shown in Table 2, Vanilla Whisper shows low CER on Mandarin and reasonable WER on Persian, both of which are supported languages in Whisper, but degrades substantially on unsupported languages. This contrast highlights the difficulty of recognising languages with limited pre-training support. Full fine-tuning provides the strongest conventional adaptation baseline across most languages. On Mandarin, however, Vanilla Whisper already achieves 8.88% CER, leaving limited room for improvement with fewer than 10 hours of training data. LoRA with rank 128 (LoRA-128) approaches FFT while updating substantially fewer parameters. To separate the effect of adaptation strategy from trainable-parameter budget, we additionally include rank-8 LoRA (LoRA-8), which has a parameter budget comparable to the FCCAbased methods. Compared to LoRA-128, LoRA-8 performs 2–8% WER worse, which is expected given its smaller update space.

## 4.2. Task-informed PEFT - FCCA variants

Here, we investigate the performance of standard FCCA (Section 2.1) and proposed FCCA variants (Section 2.2 and 2.3) on Whisper. We use rank 128 for FCCA to match the rank of LoRA-128, which provides performance close to FFT while remaining parameter efficient. Owing to the frozen-core formulation, rank-128 FCCA trains only 0.61% of Whisper parameters, giving a trainable-parameter budget comparable to LoRA-8. This allows two complementary comparisons: LoRA-128 provides a rank-matched baseline, while LoRA-8 provides a parameter-matched baseline.

FCCA trains 16 times fewer parameters than LoRA-128 and yields WER 0.36–7.14 percentage points higher than LoRA-128. Moreover, under the parameter-matched comparison, FCCA outperforms LoRA-8 on 3 languages in Whisper and almost matches it in others, suggesting that task-informed subspace construction is more effective than the budget-matched standard LoRA method. FCCA and all FCCA variants on Asturian and Kyrgyz, as well as AC-FCCA-H and AR-FCCA on Mandarin, give significant results at $p < 0 . 0 5$ under MAPSSWE against LoRA-8.

The AC-FCCA variants introduce structured cross-layer sharing on top of standard FCCA. They outperform FCCA across four of the five Whisper languages and AC-FCCA-H yields significant results at $p < 0 .$ .05 under MAPSSWE on two languages, improving WER from 44.97% to 44.35% on Sorani Kurdish and CER from 10.46% to 10.19% on Mandarin. Our preliminary analysis had shown that both AC-FCCA constructions achieved higher Fisher-weighted gradient capture than standard FCCA on a disjoint probe set, suggesting improved preservation of task-relevant update structure.

AR-FCCA gives the most consistent improvement over standard FCCA; it reduces WER by 0.32–1.13 percentage points across all languages and achieves statistically significant gains at $p \ < \ 0 . 0 5$ under MAPSSWE over standard FCCA on Sorani Kurdish and Mandarin. The improvements are observed across both unsupported and supported languages, showing that adaptive rank allocation provides a robust extension to standard FCCA without increasing the overall parameter budget.

Table 2: WER (%) on FLEURS test sets using Whisper medium; CER (%) for Mandarin.
<table><tr><td rowspan="2">Method</td><td rowspan="2">%Para</td><td colspan="3">Unsupported</td><td rowspan="2">Supported Per.</td></tr><tr><td>Ast.</td><td>Sor.</td><td>Kyr. Man.</td></tr><tr><td>Vanilla FFT</td><td>0</td><td>51.11</td><td>114.03</td><td>90.46</td><td>8.88</td><td>47.30</td></tr><tr><td rowspan="3">LoRA-128 LoRA-8</td><td>100</td><td>15.60</td><td>35.53</td><td>18.39</td><td>9.97</td><td>14.20</td></tr><tr><td>9.82</td><td>15.96</td><td>37.83</td><td>19.75</td><td>10.10</td><td>14.92</td></tr><tr><td>0.61</td><td>19.17</td><td>44.29</td><td>25.63</td><td>10.66</td><td>16.16</td></tr><tr><td>FCCA AC-FCCA-H AC-FCCA-SP AR-FCCA</td><td>0.61</td><td> $1 7 . 0 7 ^ { * }$   $1 6 . 9 7 ^ { * }$   ${ \bf 1 6 . 6 4 ^ { * } }$   $1 6 . 6 7 ^ { * }$ </td><td>44.97  $4 4 . 3 5 ^ { \dagger }$  44.83 43.84†</td><td>22.25*  $2 2 . 1 2 ^ { * }$   $2 1 . 8 5 ^ { * }$ </td><td>10.46  ${ \bf 1 0 . 1 9 ^ { \dag * } }$  10.41</td><td>16.20 16.65 16.33</td></tr></table>

<sup>˚</sup> denotes p ă 0.05 versus LoRA-8;<sup>:</sup> denotes p ă 0.05 versus FCCA.

Table 3: WER (%) on FLEURS test sets using Qwen3-ASR-1.7B.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=3>%Para</td><td rowspan=1 colspan=1>Ast.      Sor.</td></tr><tr><td rowspan=4 colspan=1>VanillaFFTLoRA-128LoRA-6</td><td rowspan=1 colspan=3>0</td><td rowspan=1 colspan=1>48.94    105.30</td></tr><tr><td rowspan=1 colspan=3>100</td><td rowspan=1 colspan=1>16.06    37.93</td></tr><tr><td rowspan=2 colspan=3>4.500.21</td><td rowspan=1 colspan=1>16.05     40.78</td></tr><tr><td rowspan=1 colspan=2>21</td><td rowspan=1 colspan=1>19.78     44.99</td></tr><tr><td rowspan=2 colspan=1>FCCAAC-FCCA-HAC-FCCA-SPAR-FCCA</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>0.20</td><td rowspan=1 colspan=1>19.63     44.3619.94     44.64 $\mathbf { 1 8 . 5 6 } ^ { \dagger * }$    $\pmb { 4 2 . 9 8 } ^ { \dagger * }$ </td></tr></table>

<sup>˚</sup> denotes p ă 0.05 versus LoRA-6; : denotes p ă 0.05 versus FCCA.

## 4.3. Generalization to Qwen3-ASR

This subsection investigates whether the observations on Whisper generalize to a structurally different speech foundation model. Qwen3-ASR is evaluated on Asturian and Sorani Kurdish, both of which are unsupported by the pretrained model. As shown in Table 3, Vanilla Qwen3-ASR performs poorly on both languages, with WERs of 48.94% and 105.30%, respectively. As with Whisper, FFT and LoRA-128 give largest improvement on WER; for Asturian, FFT and LoRA are almost matched at 16.06% and 16.05%.

Standard FCCA substantially improves the pretrained model, reducing WER to 19.61% on Asturian and 44.22% on Sorani, but as with Whisper, it still trails the rank-matched LoRA-128. FCCA also beats its trainable-parameter-matched LoRA version (LoRA-6) by 0.17% WER on Asturian and 0.77% on Sorani Kuridish. However, the AC-FCCA variants which offered improvements on FCCA in Whisper, increase WER in Qwen3-ASR. In contrast, AR-FCCA again gives the strongest FCCA-based performance, reducing WER further to 18.56% and 42.98%, respectively. These improvements over standard FCCA are statistically significant for both Asturian $( p = 0 . 0 0 1 )$ and Sorani $( p = 0 . 0 5 )$ under MAPSSWE.

## 4.4. Analysis of Adaptive Rank Allocation

The consistent gains from AR-FCCA across Whisper and Qwen3- ASR motivate a closer examination of this approach. Figure 1 shows the rank allocation for Whisper on Asturian. Rank allocation patterns are highly consistent across languages within each model. AR-FCCA jointly chooses per-matrix ranks to maximize retained Fisherwhitened spectral energy under a fixed global budget, so larger ranks are assigned where adding further directions continues to capture substantial task-relevant spectral energy. In Whisper, this concentrates rank in decoder output projections, while Qwen3-ASR assigns high ranks to q{o in the text tower but to v{k in the audio tower, with audio rank also tending to increase with depth.

![](images/c83d1fd8f5365272784cf6d59fe0c43738f861903f0275afc631d9514a9033ae.jpg)  
Fig. 1: Adaptive FCCA rank allocation across Whisper model layers and attention projections for Asturian. Darker cells indicate larger allocated rank.

To examine why AR-FCCA consistently improves over uniformrank FCCA, we further test whether the rank allocation selected before training remains advantageous as optimisation progresses. For each intermediate checkpoint, the FCCA statistics and Fisherwhitened singular spectra are recomputed, while keeping the AR-FCCA ranks $\overline { { r _ { \ell } ^ { \mathrm { A R } } } }$ fixed to their initial values. We then compare this fixed allocation with uniform rank 128 under the same total trainable-core budget. Following the FCCA local quadratic formulation, the utility of a rank allocation is measured by the retained spectral energy $\begin{array} { r } { \sum _ { \ell } \sum _ { k \leqslant r _ { \ell } } \sigma _ { \ell , k } ^ { 2 } } \end{array}$ , which is proportional to the best local rank-constrained loss reduction under the FCCA surrogate. At checkpoint e, we define $\begin{array} { r } { \Delta _ { \mathrm { f r e s h } } ^ { ( e ) } = 1 0 0 \left( \frac { \sum _ { \ell } \sum _ { k = 1 } ^ { r _ { \ell } ^ { \mathrm { A R } } } ( \sigma _ { \ell , k } ^ { ( e ) } ) ^ { 2 } } { \sum _ { \ell } \sum _ { k = 1 } ^ { 1 2 8 } ( \sigma _ { \ell , k } ^ { ( e ) } ) ^ { 2 } } - 1 \right) } \end{array}$ %. Across both Whisper and Qwen3-ASR and all evaluated languages, $\Delta _ { \mathrm { f r e s h } } ^ { ( e ) } > 0$ at every checkpoint. This indicates that the rank allocation selected from the initial FCCA geometry continues to retain more task-relevant spectral utility than uniform rank allocation throughout training.

## 5. CONCLUSION

This work investigated task-informed PEFT for multilingual lowresource ASR through FCCA and two complementary extensions. Across Whisper and Qwen3-ASR, standard FCCA provided a highly parameter-efficient alternative to conventional PEFT with matched parameter budget, while the AC-FCCA variants gave only limited additional gains. AR-FCCA provided the most consistent improvements over standard FCCA by reallocating the same trainable budget across projection matrices, with gains observed across both speech foundation models and multiple languages and statistically significant improvements in several settings. Further analysis showed that AR-FCCA assigns markedly different ranks across layers and projections, and that these allocations retain higher FCCA spectral utility than uniform rank allocation throughout training. A limitation is the lack of systematic parameter-matched LoRA comparisons such as LoRA-XS, which we leave for future work.

## Acknowledgements

This paper reports on research supported by Cambridge Language Sciences Incubator Fund.

## 6. REFERENCES

[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International Conference on Machine Learning (ICML), 2022.

[2] X. Shi et al., “Qwen3-ASR Technical Report,” arXiv preprint arXiv:2601.21337, 2026.

[3] L. Lonergan, M. Qian, H. Berthelsen, A. Murphy, C. Wendler, N. N. Chiarain, C. Gobl, and A. N. Chasaide, “Automatic´ speech recognition for irish: the abair-eist system,” in´ Proceedings of the 4th Celtic Language Technology Workshop within LREC2022, 2022, pp. 47–51.

[4] L. Lonergan, I. Saratxaga, J. Sloan, O. M. Bravo, M. Qian, N. N. Chiarain, C. Gobl, and A. N. Chasaide, “Fotheidil:´ An automatic transcription system for the Irish language,” in Proceedings of the 5th Celtic Language Technology Workshop, 2025, pp. 35–45.

[5] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for NLP,” in International Conference on Machine Learning (ICML), 2019.

[6] J. Pfeiffer, A. Ruckl ¨ e, C. Poth, A. Kamath, I. Vuli ´ c, S. Ruder,´ K. Cho, and I. Gurevych, “AdapterHub: A framework for adapting transformers,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020, pp. 46–54.

[7] X. L. Li and P. Liang, “Prefix-Tuning: Optimizing Continuous Prompts for Generation,” in Proc. ACL-IJCNLP (Volume 1: Long Papers), 2021, pp. 4582–4597.

[8] B. Lester et al., “The Power of Scale for Parameter-Efficient Prompt Tuning,” in Proc. EMNLP, 2021, pp. 3045–3059.

[9] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations (ICLR), 2022.

[10] M. Qian, S. Tang, R. Ma, K. M. Knill, and M. J. Gales, “Learn and Don’t Forget: Adding a New Language to ASR Foundation Models,” in Proc. Interspeech, 2024, pp. 2544–2548.

[11] T. T. T. T. Dang, M. Qian, and K. Knill, “Sequential Adapter Stacking for Cross-Lingual Low-Resource ASR,” arXiv preprint arXiv:2609.15758, 2026.

[12] Y. Yang, X. Li, Z. Zhou, S. L. Song, J. Wu, L. Nie, and B. Ghanem, “CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning,” in Advances in Neural Information Processing Systems, 2024, vol. 37.

[13] L. Li, D. Li, C. Lin, W. Li, W. Xue, S. Han, and Y. Guo, “AIRA: Activation-informed low-rank adaptation for large models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 1729–1739.

[14] D. Das, H. Park, M. Hayat, S. Choi, S. Yun, and F. Porikli, “ConsNoTrainLoRA: Data-driven weight initialization of lowrank adapters using constraints,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 498–507.

[15] L. Li, C. Lin, D. Li, Y.-L. Huang, W. Li, T. Wu, J. Zou, W. Xue, S. Han, and Y. Guo, “Efficient fine-tuning of large models via nested low-rank adaptation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 22252–22262.

[16] S. Wang, L. Yu, and J. Li, “LoRA-GA: Low-rank Adaptation With Gradient Approximation,” Advances in Neural Information Processing Systems, vol. 37, pp. 54905–54931, 2024.

[17] F. Paischer, L. Hauzenberger, T. Schmied, B. Alkin, M. P. Deisenroth, and S. Hochreiter, “One initialization to rule them all: Fine-tuning via explained variance adaptation,” in Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning, 2024.

[18] W. Ye et al., “Frozen Cores Need Task Signal: Fisher-Whitened Cross-Covariance for Low-Resource LLM Adaptation,” arXiv preprint arXiv:2609.00762, 2026.

[19] Q. Team, “Qwen2.5: A Party of Foundation Models,” September 2024.

[20] K. Bałazy, M. Banaei, K. Aberer, and J. Tabor, “Lora-xs: Lowrank adaptation with extremely small number of parameters,” in ECAI 2025: 28th European Conference on Artificial Intelligence 25-30 October 2025, Bologna, Italy-Including 14th Conference on Prestigious Applications of Intelligent Systems (PAIS 2025) Proceedings. SAGE Publications 1 Oliver’s Yard, 55 City Road, London, EC1Y 1SP, 2024, pp. 3194–3201.

[21] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and T. Zhao, “Adaptive budget allocation for parameterefficient fine-tuning,” in International Conference on Learning Representations (ICLR), 2023.

[22] “I-LoRA: An adaptive rank allocation approach using integrated gradients,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.

[23] A. Conneau et al., “FLEURS: Few-shot learning evaluation of universal representations of speech,” in Proc. SLT. IEEE, 2023, pp. 798–805.

[24] The Unicode Consortium, “Unicode Standard Annex #15: Unicode Normalization Forms,” https://www.unicode. org/reports/tr15/, 2025.

[25] OpenCC Contributors, “OpenCC: Open Chinese Convert,” https://github.com/BYVoid/OpenCC, Traditional and Simplified Chinese conversion software.