# SyRA-CS: Synthetic-to-Real Adaptation for Code-Switched ASR with Meta-Learning

Fadel Thior, Ahmer Tabassum, Haania Siddiqui, Hanan Aldarmaki, Thamar Solorio Mohamed bin Zayed University of Artificially Intelligence fadel.thior@mbzuai.ac.ae

## Abstract

Code-switched (CS) automatic speech recognition (ASR) remains difficult because real CS speech is scarce and synthetic speech does not fully match it. We investigate whether synthetic CS speech is more useful for learning an adaptation-friendly initialization than for ordinary continual fine-tuning. We propose SyRA-CS, a pair-specific synthetic-toreal adaptation framework based on Reptile meta-learning. For each language pair and pretrained model, we meta-train on the corresponding synthetic CS-FLEURS subset and then adapt using one hour of labeled real target speech. We evaluate four architectural settings: Encoder–CTC, FastConformer–RNN-T, Attention Encoder–Decoder (AED) and Speech-LLM. Experiments cover on Arabic-, French-, German-, and Hindi-English, against syntheticonly transfer, real-only fine-tuning, and continual fine-tuning. SyRA-CS improves over continual fine-tuning on WER in all 16 model– pair comparisons and on CER in 15 of 16, with relative macro WER reductions of 17.7% for XLS-R and 5.3–5.8% for the other three models. The gains hold across the tested models but vary in size across architectures, language pairs and metrics.

## 1 Introduction

Code-switching (CS)—the alternation of two or more languages within a conversation or utterance— is a common phenomenon in multilingual communities, but presents additional challenges for Automatic Speech Recognition (ASR) systems. The challenge is both linguistic and data-centric: real CS corpora are expensive to collect and transcribe, and switching patterns vary with speaker community, accent, discourse function, topic, and orthographic convention. As a result, ASR models trained for one CS condition often degrade under another (Sitaram et al., 2019; Lyu et al., 2010; Li and Fung, 2012).

Synthetic CS speech offers an attractive way to cheaply expand supervision data (Yan et al., 2025; Dhawan et al., 2023), but synthetic speech does not perfectly match real speech. Synthetic utterances may have cleaner acoustics, less speaker and accent variability. Therefore, simply fine-tuning on synthetic CS speech may overfit to synthetic regularities instead of generalizing to real CS recognition. An alternative approach is meta-learning: a method that trains a model to become adaptable, by repeatedly simulating small adaptation episodes and updating the parameters so that a few gradient steps on a new, data-scarce task lead to good performance (Finn et al., 2017). The output is not a final recognizer but an initialization from which fine-tuning on a small labeled set is fast and stable. This paper asks a focused question: under a limited real data budget, can a meta-learned initialization from synthetic CS speech adapt better to real CS speech compared to standard training strategies? We use Reptile (Nichol et al., 2018) because it is a first-order meta-learning algorithm that repeatedly simulates adaptation episodes and moves the initialization toward parameters that are easy to reach after a few gradient steps: our contribution is not a new meta-learning optimizer or a benchmark of all meta-learning algorithms (Sec tion 4.1). Instead, we test whether a lightweight meta-initializer is useful for synthetic-to-real CS ASR under severe target scarcity. We train one Reptile initialization for each combination of language pair and pretrained model. Our experiments cover four architectural settings: XLS-R 300M with a CTC prediction head (Babu et al., 2021), Nemotron 3.5 with a FastConformer–RNN-T architecture (NVIDIA, 2026a), Whisper-small with an attention encoder–decoder architecture (OpenAI, 2026), and Nemotron-Labs-Audex-2B as a Speech-LLM (NVIDIA, 2026b). Within each model–pair combi nation, the training regimes start from the same pretrained checkpoint and use the corresponding synthetic CS-FLEURS subset. Methods requiring realdomain adaptation use the same one-hour labeled target subset. Hindi-English is evaluated on MUCS 2021 (Diwan et al., 2021), while Arabic-English, French-English, and German-English are evaluated on SwitchLingua (Xie et al., 2025). This design evaluates within-pair synthetic-to-real adaptation across different pretrained ASR systems; it does not learn a shared multilingual meta-initialization. Our contributions are:

• We formulate pair-specific synthetic-to-real codeswitched ASR as a data-efficient meta-learning problem, using synthetic speech to learn an initialization for adaptation with one hour of labeled real target speech

• We investigate Reptile across four architectural settings: Encoder–CTC, FastConformer–RNN-T, attention encoder–decoder, and Speech-LLM, using comparisons against synthetic-only transfer, real-only fine-tuning, and continual fine-tuning within each pretrained model.

• We characterize the code-switching profile of each real target set (switching ratio) and assess the robustness of the gains to the two smallest evaluation sets.

## 2 Related Work

## 2.1 Multilingual self-supervised

Self-supervised speech learning has become the dominant paradigm for low-resource ASR. Baevski et al. (2020) introduced wav2vec 2.0, which learns contextualized acoustic representations from raw unlabeled audio and can then be fine-tuned with limited transcribed speech. HuBERT improved masked prediction with cluster-based targets (Hsu et al., 2021), and XLS-R scaled the wav2vec 2.0 recipe to nearly half a million hours of multilingual audio across 128 languages (Babu et al., 2021). Large multilingual speech resources such as FLEURS (Conneau et al., 2022), Common Voice (Ardila et al., 2020), VoxPopuli (Wang et al., 2021), and MMS (Pratap et al., 2023) have further strengthened multilingual transfer.

## 2.2 Code-switched ASR and synthetic data

A recent systematic review of end-to-end CS-ASR studies notes that, although data augmentation is widely used, it remains less explored than many model-centric approaches (Agro et al., 2025). In response, recent studies have used synthetic text and speech to improve low-resource code-switched ASR. Several works explore how synthetic codeswitched data can be created or leveraged when real code-switched speech is scarce. Nguyen and Tran (2025b) show that ASR systems can be trained with synthetic phrase-level code-switching derived from monolingual resources, while Nguyen and Tran (2025a) propose AsyncSwitch, which adapts models on large amounts of code-switched text before aligning this knowledge with limited paired speech. More broadly, Hamed et al. (2025) find that the value of synthetic code-switched data depends strongly on its quality and the downstream task. At the same time, recent work has introduced stronger benchmarks and architectures for code-switched ASR. Yan et al. (2025) present CS-FLEURS, a multilingual benchmark covering 113 unique codeswitched language pairs, and show that synthetic training data improves performance on both seen and unseen pairs. Ugan et al. (2024) show that bilingual ASR systems still struggle on realistic code-switching benchmarks, while Ye et al. (2024) propose a mixture-of-experts architecture to better handle language switching in streaming and non-streaming settings. Earlier work also explored semi-supervised and unified approaches (Tazakka et al., 2024; Dhawan et al., 2023). Overall, these works show that synthetic data is becoming an important tool for code-switched ASR, especially in low-resource settings where real code-switched speech is scarce. They also suggest that the main challenge is not only generating more data, but generating data that is linguistically and acoustically useful for adaptation.

## 2.3 Meta-learning for ASR

Meta-learning is well suited to low-resource ASR because it explicitly learns parameters that can adapt quickly to new conditions with limited supervision. Early work applied MAML-style metalearning to end-to-end ASR, learning initializations from high-resource languages that adapt quickly to low-resource targets (Hsu et al., 2020), and Winata et al. (2020) extended this idea to code-switched speech with meta-transfer learning, which draws episodes from monolingual corpora so that the initialization transfers to limited real CS data. Our work differs in that episodes are sampled from synthetic CS speech of the same language pair and the meta-learner is first-order Reptile rather than MAML. Recent ASR work has strengthened this direction: Meta-Adapter shows that meta-initialized adapters improve self-supervised speech adaptation for low-resource languages (Chen et al., 2024), SMILE extends meta-learning to few-shot multilingual ASR through speech meta in-context learning (Hsu and Lee, 2024), and MetaMixSpeech shows that task augmentation can improve generalization in multilingual low-resource ASR (Chen et al., 2025). Together, these studies motivate metalearning as an effective framework for fast speech adaptation under severe data scarcity. Related ideas also appear in multilingual NLP, where targeted adaptation has been shown to improve transfer to low-resource language families (Downey et al., 2024). In code-switched ASR, recent work such as AsyncSwitch emphasizes adaptation to limited paired code-switched speech after prior exposure to code-switched resources (Nguyen and Tran, 2025a).

![](images/fc47e124ea033c8b4dca8f871dc71f1f026985127553411a5e7ef23dd14de5f5.jpg)  
Figure 1: Overview of the Pair-Specific Synthetic-to-Real Reptile Adaptation Framework for Code-Switched ASR.

## 3 Data, Model, and Evaluation

## 3.1 Backbones and architecture scope

We evaluate SyRA-CS with four pretrained models spanning different ASR architectures:

• XLS-R 300M (Encoder-CTC): wav2vec2-xls-r-300m, with a linear CTC prediction head (Babu et al., 2021).

• Nemotron 3.5 (FastConformer–RNN-T): a FastConformer encoder with an RNN-T decoder (NVIDIA, 2026a).

• Whisper-small (AED): an attention-based Transformer encoder–decoder (OpenAI, 2026).

• Nemotron-Labs-Audex-2B (Speech-LLM): an audio-text language model used for speech transcription (NVIDIA, 2026b).

The synthetic CS-FLEURS data and real target data provide supervised training and adaptation examples. They are not used for additional selfsupervised pretraining.

Let a index the pretrained model and let θ denote its trainable parameters. Reptile operates on these parameters using the model-specific transcription objective:

$$
\mathcal { L } _ { a } = \left\{ \begin{array} { l l } { \mathcal { L } _ { \mathrm { C T C } } , } & { a = \mathrm { X L S - R } , } \\ { \mathcal { L } _ { \mathrm { R N N T } } , } & { a = \mathrm { N e m o t r o n } , } \\ { \mathcal { L } _ { \mathrm { A R } } , } & { a \in \{ \mathrm { W h i s p e r } , \mathrm { A u d e x } \} , } \end{array} \right.\tag{1}
$$

CTC marginalizes over frame-level alignments compatible with the transcript (Graves et al., 2006), whereas RNN-T uses a transducer alignment objective. For the autoregressive models, the transcription objective is

$$
\mathcal { L } _ { \mathrm { A R } } ( \theta ; x , y ) = - \sum _ { u = 1 } ^ { | y | } \log p _ { \theta } ( y _ { u } \mid y _ { < u } , x , c _ { a } ) ,\tag{2}
$$

where $c _ { a }$ denotes any model-specific task tokens or transcription prompt. For the Speech-LLM, this expression describes loss on the target transcription tokens. Across models, parameter counts, pretraining histories, tokenization, and decoding mechanisms differ. The cross-model experiments therefore examine whether the benefit of Reptile extends across architectural settings; absolute differences between models cannot be attributed to architecture alone.

## 3.2 Source tasks: CS-FLEURS data

Our synthetic source data are drawn from CS-FLEURS (Yan et al., 2025), a corpus of synthetic code-switched speech spanning multiple language pairs. We use four code-switched subsets from XTTS version as it contains training data for our meta learning experiments: Hindi-English (6 hours), French-English (10.28h), Arabic-English (4.51 hours) and German-English (9.37 hours), where each subset corresponds to one language pair and the reported durations denote the total amount of synthetic training audio available for that pair. For a model a and target pair p, synthetic episodes are sampled only from the corresponding source subset $S _ { p } .$ Thus, each meta-training run learns a model-specific, pair-specific initialization rather than an initialization trained jointly across language pairs.

## 3.3 Target tasks: MUCS and Switchlingua data

We evaluate adaptation under real-world codeswitched conditions using all four pairs as target tasks. Real-world Hindi-English speech data is from MUCS (Diwan et al., 2021), French-English (2.64 hours), Arabic-English (1.43 hours) and German-English (1.40 hours) are from the Switch-Lingua dataset (Xie et al., 2025). SwitchLingua provides real audio recordings from bilingual speakers, but the spoken prompts are generated code-switched text samples. Thus, the target audio is human speech, while the linguistic content follows a synthetic-script collection protocol. Unlike SwitchLingua, MUCS Hindi-English is extracted from human spoken tutorials. The speech is therefore real instructional audio, and the codeswitching primarily arises from technical lecture content rather than from generated code-switched text. For each target pair, we randomly select a 1-hour labeled subset for adaptation and evaluate final performance on the corresponding real codeswitched test set. As CS speech data is scarce, this setup is intentionally low-resource and is designed to measure how effectively a model trained on synthetic code-switched speech can transfer to real speech with only a small amount of target-domain supervision.

## 3.4 Metrics and Text normalization

We report Word Error Rate (WER) and Character Error Rate (CER) using Levenshtein edit distance

(Levenshtein, 1966):

$$
\begin{array} { l } { \displaystyle \mathrm { W E R } = \frac { S + D + I } { N _ { w } } , } \\ { \displaystyle \mathrm { C E R } = \frac { S _ { c } + D _ { c } + I _ { c } } { N _ { c } } . } \end{array}\tag{3}
$$

The same normalization is applied to references and hypotheses for all systems before scoring: unicode normalization, removal of markup artifacts, punctuation stripping, whitespace normalization, and case normalization for Latin-script text. We do not introduce system-specific post-processing. For Arabic-English, the evaluation uses the same common normalization pipeline and does not apply additional Arabic-specific orthographic collapsing unless explicitly configured in the released scoring script. WER and CER are standard and comparable across all four pairs; we complement them with a description of the code-switching profile of each test set. Multi-reference WER (Ali et al., 2015) and PolyWER (Kadaoui et al., 2024) are not used here because MUCS and SwitchLingua provide a single reference per utterance; constructing alternative references manually would introduce subjective normalization choices.

## 4 Meta Learning Adaptation

Let $\mathcal { P } =$ {ara-eng, fra-eng, deu-eng, hin-eng} and A = {XLSR, Nemotron, Whisper, Audex}. For each pair $p \in \mathcal P$ , let $S _ { p }$ denote its synthetic CS-FLEURS source set, $A _ { p }$ its one-hour real adaptation set, and $E _ { p }$ its held-out real evaluation set. For each model $a \in { \mathcal { A } }$ , we learn a separate initialization $\theta _ { a , p } ^ { \mathrm { m e t a } }$ starting from that model’s pretrained initialization $\theta _ { a , 0 }$ . The symbol θ denotes the parameters updated during training. At metaiteration t, each synthetic episode i starts from $\theta _ { a , p , i } ^ { ( 0 ) } = \theta _ { a , p , t }$ . It performs $k _ { a }$ inner updates using the model-specific loss from Section 3.1:

$$
\begin{array} { r } { \boldsymbol { \theta } _ { a , p , i } ^ { ( j + 1 ) } = \boldsymbol { \theta } _ { a , p , i } ^ { ( j ) } - \alpha _ { a } \nabla _ { \theta } \mathcal { L } _ { a } \big ( \boldsymbol { \theta } _ { a , p , i } ^ { ( j ) } ; B _ { p , i } ^ { ( j ) } \big ) , } \end{array}\tag{4}
$$

where $B _ { p , i } ^ { ( j ) } \subset S _ { p }$ is a synthetic minibatch and $j =$ $0 , \ldots , k _ { a } ^ { \dagger } - 1$ . This notation expresses a gradient step; the optimizer used in each implementation is reported with the experimental configuration.

Let $\widetilde { \theta } _ { a , p , i } \ = \ \theta _ { a , p , i } ^ { ( k _ { a } ) }$ . For a meta-batch of $m _ { a }$ episodes, the Reptile update is

$$
\begin{array} { l } { \displaystyle \theta _ { a , p , t + 1 } = \theta _ { a , p , t } } \\ { \displaystyle \quad + \frac { \beta _ { a , t } } { m _ { a } } \sum _ { i = 1 } ^ { m _ { a } } ( \widetilde { \theta } _ { a , p , i } - \theta _ { a , p , t } ) , } \end{array}\tag{5}
$$

where $\beta _ { a , t }$ is the outer step size. The update uses parameter displacements without differentiating through the inner optimization trajectory.

After meta-training, the initialization is adapted to the one-hour real subset:

$$
\begin{array} { r l } & { \phi _ { a , p } ^ { ( 0 ) } = \theta _ { a , p } ^ { \mathrm { m e t a } } , } \\ & { \phi _ { a , p } ^ { ( u + 1 ) } = \phi _ { a , p } ^ { ( u ) } - \eta _ { a } \nabla _ { \phi } \mathcal { L } _ { a } \bigl ( \phi _ { a , p } ^ { ( u ) } ; B _ { p } ^ { ( u ) } \bigr ) , } \end{array}\tag{6}
$$

where $B _ { p } ^ { ( u ) } \subset A _ { p }$ . The resulting model $\phi _ { a , p } ^ { \star }$ is evaluated on $E _ { p }$ . Neither meta-training nor target adaptation uses evaluation examples.

## 4.1 Why First-Order Meta-Learning?

Reptile is particularly suitable for our setting because it avoids differentiating through the inner optimization trajectory (Nichol et al., 2018). In MAML, after k inner updates $\theta _ { \tau } ^ { \prime } = U _ { \tau } ^ { k } ( \theta )$ , the meta-update requires

$$
\theta  \theta - \beta \nabla _ { \theta } \mathcal { L } _ { \tau } ^ { \mathrm { o u t } } ( U _ { \tau } ^ { k } ( \theta ) ) ,
$$

and therefore backpropagates through the sequence of inner-loop optimization steps. In its standard form, this introduces second-order derivatives and substantial memory and computational overhead. This overhead is particularly relevant when repeatedly adapting large pretrained speech models, including CTC, transducer, encoder–decoder, and Speech-LLM systems. Reptile instead uses the parameter displacement

$$
\theta  \theta + \beta ( \theta _ { \tau } ^ { \prime } - \theta ) ,
$$

without differentiating through the trajectory that produced $\theta _ { \tau } ^ { \prime }$ . The outer update can therefore be applied across our four architectural settings while the inner loop uses each model’s transcription objective. Our hypothesis is that repeated adaptation on synthetic CS episodes produces an initialization that transfers more effectively to limited real target speech than ordinary synthetic-domain finetuning. This hypothesis is tested through withinmodel comparisons against CFT. Reptile’s firstorder formulation reduces the derivative overhead associated with standard second-order MAML, but does not remove the cost of repeated inner-loop training.

## 5 Experiments and Results

## 5.1 Baselines and Reproducibility

For each model a and language pair p, we compare four training regimes:

<table><tr><td>Setting</td><td>XLS-R 300M</td><td>Nemotron 3.5 / Whisper-small / Audex-2B</td></tr><tr><td>Architecture</td><td>Encoder-CTC</td><td>FastConformer-RNN-T / AED / Speech-LLM</td></tr><tr><td>Trainable parameters</td><td>Full model</td><td>Full model</td></tr><tr><td>Meta-iterations</td><td>500</td><td>500</td></tr><tr><td>Inner steps ka</td><td>50</td><td>50</td></tr><tr><td>Inner learning rate αa</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Outer step size βa</td><td>0.1</td><td>0.1</td></tr><tr><td>Batch size</td><td>2</td><td>2</td></tr><tr><td>Target fine-tuning epochs 50</td><td></td><td>50</td></tr><tr><td>GPU</td><td>1× RTX 4090</td><td>RTX A4500</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Setting: model-specific, pair-specific Reptile</td></tr><tr><td colspan="3">Pairs: ara-eng, fra-eng, deu-eng, hin-eng</td></tr><tr><td colspan="3">Source: corresponding CS-FLEURS subset</td></tr><tr><td colspan="3">Target: 1h per pair; MUCS and SwitchLingua</td></tr><tr><td colspan="3"></td></tr></table>

Table 1: Reptile configurations. The grouped column reports shared settings; architectures follow the model order.

• Synthetic-only transfer: train from the model’s pretrained checkpoint on $S _ { p }$ and evaluate directly on $E _ { p }$ , without real adaptation.

• Real-only fine-tuning: fine-tunes the pretrained model on $A _ { p }$ only.

• Continual fine-tuning (CFT): train first on $S _ { p }$ and then adapt on $A _ { p } .$

• SyRA-CS: learn a Reptile initialization from episodes sampled from $S _ { p } ,$ then adapt on $A _ { p }$

Within each model–pair combination, all regimes start from the same pretrained checkpoint. Methods using real adaptation share the same one-hour subset, and all methods are evaluated on the same heldout set. Trainable parameter scope, tokenization, transcript normalization, and inference settings are fixed across regimes within each model. CFT is the closest baseline because it uses the same synthetic source set and real target subset as SyRA-CS. Their comparison tests the value of episodic Reptile training relative to ordinary sequential training. Table 1 reports the Reptile configuration for each model.

## 5.2 Main results

Table 2 compares the four training regimes within each pretrained model, and SyRA-CS improves over CFT in all 16 model–pair WER comparisons and in 15 of the 16 CER comparisons, and attains the lowest error among the four regimes in 31 of the 32 model–pair–metric cells. Macro averages over the four pairs are computed from the rounded scores in Table 2; relative macro reductions are computed from the macro error rates rather than by averaging the pair-level reductions in Figure 2.

<table><tr><td rowspan="2">Training regime</td><td rowspan="2">Real data</td><td colspan="2">Hindi-English</td><td colspan="2">French-English</td><td colspan="2">Arabic-English</td><td colspan="2">German-English</td></tr><tr><td>WER↓</td><td>CER↓</td><td>WER↓</td><td>CER↓</td><td>WER↓</td><td>CER↓</td><td>WER↓</td><td>CER↓</td></tr><tr><td>XLS-R 300M (Encoder-CTC)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Synthetic-only transfer</td><td>Oh</td><td>0.770</td><td>0.590</td><td>0.814</td><td>0.364</td><td>0.976</td><td>0.527</td><td>0.830</td><td>0.333</td></tr><tr><td>Real-only fine-tuning</td><td>1h</td><td>0.980</td><td>0.960</td><td>0.531</td><td>0.182</td><td>0.827</td><td>0.367</td><td>0.784</td><td>0.219</td></tr><tr><td>CFT from synthetic</td><td>1h</td><td>0.479</td><td>0.288</td><td>0.370</td><td>0.117</td><td>0.615</td><td>0.212</td><td>0.432</td><td>0.137</td></tr><tr><td>SyRA-CS (ours)</td><td>1h</td><td>0.416</td><td>0.257</td><td>0.350</td><td>0.112</td><td>0.461</td><td>0.163</td><td>0.333</td><td>0.104</td></tr><tr><td colspan="2">Nemotron 3.5 (FastConformer-RNN-T)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Synthetic-only transfer</td><td>Oh</td><td>0.554</td><td>0.490</td><td>0.277</td><td>0.160</td><td>0.538</td><td>0.319</td><td>0.337</td><td>0.145</td></tr><tr><td>Real-only fine-tuning</td><td>1h</td><td>0.531</td><td>0.472</td><td>0.233</td><td>0.130</td><td>0.504</td><td>0.320</td><td>0.309</td><td>0.147</td></tr><tr><td>CFT from synthetic</td><td>1h</td><td>0.488</td><td>0.426</td><td>0.219</td><td>0.118</td><td>0.454</td><td>0.269</td><td>0.267</td><td>0.113</td></tr><tr><td>SyRA-CS (ours)</td><td>1h</td><td>0.480</td><td>0.417</td><td>0.202</td><td>0.106</td><td>0.429</td><td>0.258</td><td>0.241</td><td>0.104</td></tr><tr><td>Whisper-small (AED)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Synthetic-only transfer</td><td>Oh</td><td>0.831</td><td>0.675</td><td>0.278</td><td>0.133</td><td>0.613</td><td>0.321</td><td>0.305</td><td>0.124</td></tr><tr><td>Real-only fine-tuning</td><td>1h</td><td>0.614</td><td>0.494</td><td>0.172</td><td>0.082</td><td>0.374</td><td>0.188</td><td>0.189</td><td>0.080</td></tr><tr><td>CFT from synthetic</td><td>1h</td><td>0.533</td><td>0.418</td><td>0.163</td><td>0.073</td><td>0.352</td><td>0.162</td><td>0.191</td><td>0.080</td></tr><tr><td>SyRA-CS (ours)</td><td>1h</td><td>0.499</td><td>0.349</td><td>0.153</td><td>0.071</td><td>0.348</td><td>0.181</td><td>0.171</td><td>0.071</td></tr><tr><td colspan="2">Nemotron-Labs-Audex-2B (Speech-LLM)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Synthetic-only transfer</td><td>Oh</td><td>0.766</td><td>0.640</td><td>0.194</td><td>0.110</td><td>0.721</td><td>0.496</td><td>0.324</td><td>0.147</td></tr><tr><td>Real-only fine-tuning</td><td>1h</td><td>0.525</td><td>0.460</td><td>0.136</td><td>0.074</td><td>0.607</td><td>0.427</td><td>0.199</td><td>0.107</td></tr><tr><td>CFT from synthetic</td><td>1h</td><td>0.520</td><td>0.431</td><td>0.121</td><td>0.064</td><td>0.577</td><td>0.443</td><td>0.193</td><td>0.102</td></tr><tr><td>SyRA-CS (ours)</td><td>1h</td><td>0.490</td><td>0.391</td><td>0.117</td><td>0.061</td><td>0.539</td><td>0.409</td><td>0.183</td><td>0.101</td></tr></table>

Table 2: Synthetic-to-real CS ASR performance across four pretrained models. WER and CER are reported as fractions. Real data denotes labeled target speech used for adaptation.

Largest gains for XLS-R. The encoder–CTC model benefits most: macro WER falls from 0.4740 to 0.3900 (17.7%) and macro CER from 0.1885 to 0.1590 (15.6%). The largest pair-level gains are on Arabic-English (WER 0.615 → 0.461) and German-English (0.432 → 0.333). XLS-R also has the weakest CFT baseline of the four models, leaving the most room for improvement.

Smaller but consistent gains for the other models. Nemotron 3.5 and Audex-2B improve both metrics on every pair, with macro WER/CER reductions of 5.3%/4.4% and 5.8%/7.5%, respectively. Audex gains most on Arabic-English (WER 0.577 → 0.539) and Hindi-English (0.520 → 0.490). Whisper-small improves WER on every pair (5.5% macro) and macro CER by 8.3%, driven mainly by Hindi-English (CER 0.418 → 0.349). Its Arabic-English result is the only regression: WER decreases slightly (0.352 → 0.348) while CER increases (0.162 → 0.181), so pair-level conclusions can depend on the metric.

Variation across models and language pairs. Figure 2 shows pair-level relative reductions against CFT. WER reductions range from 5.4% to 25.0% for XLS-R, 1.6% to 9.7% for Nemotron, 1.1% to 10.5% for Whisper, and 3.3% to 6.6% for Audex. All four models improve both WER and CER on Hindi-English, French-English, and German-English. On Hindi-English, WER improves by 13.2%, 1.6%, 6.4%, and 5.8% for XLS-R, Nemotron, Whisper, and Audex, respectively.

Because Hindi-English differs from the SwitchLingua pairs in corpus, domain, and switching density (Section 5.3) simultaneously, these results cannot isolate switching density as a driver of the gains.

Other baselines. Synthetic-only transfer yields the highest error in 28 of 32 cells, confirming a substantial synthetic-to-real gap. CFT outperforms real-only fine-tuning in 29 of 32 cells (exceptions: Whisper German-English WER, a tie on Whisper German-English CER, and Audex Arabic-English CER), making it a strong baseline. Realonly XLS-R nearly fails on Hindi-English (WER 0.980), which inflates its macro score, so we treat CFT as the primary comparison. Because CFT and SyRA-CS use the same synthetic subset and the same one-hour real subset, their gap reflects the training strategy rather than additional data.

## 5.3 Code-switching profile and Robustness checks

Figure 5 shows that code-switched utterances account for 88.5% of Arabic-English, 88.4% of French-English, 76.9% of German-English, and 80.8% of Hindi-English test utterances. Hindi-English has the highest reported switching density, at 2.996 switches per utterance. These statistics establish substantial mixed-language coverage, but do not identify which linguistic phenomena account for a model’s improvements.

Robustness to small evaluation sets. German-English and Arabic-English contain only 1.40 and

<table><tr><td rowspan="2"></td><td colspan="4">WER relative reduction (%)</td></tr><tr><td>+13.2</td><td>+5.4</td><td>+25.0</td><td>+22.9</td></tr><tr><td>Nemotron 3.5</td><td>+1.6</td><td>+7.8</td><td>+5.5</td><td>+9.7</td></tr><tr><td>Whisper-small</td><td>+6.4</td><td>+6.1</td><td>+1.1</td><td>+10.5</td></tr><tr><td>Audex-2B</td><td>+5.8</td><td>+3.3</td><td>+6.6</td><td>+5.2</td></tr><tr><td></td><td>Hi-En</td><td>Fr-En</td><td>Ar-En</td><td>De-En</td></tr></table>

<table><tr><td colspan="5">CER relative reduction (%)</td></tr><tr><td>+10.8</td><td>+4.3</td><td>+23.1</td><td>+24.1</td><td>Poss s ss S-cv = vctrT 20</td></tr><tr><td>+2.1</td><td>+10.2</td><td>+4.1</td><td>+8.0</td><td>10</td></tr><tr><td>+16.5</td><td>+2.7</td><td>-11.7</td><td>+11.3</td><td>0 -10</td></tr><tr><td>+9.3</td><td>+4.7</td><td>+7.7</td><td>+1.0</td><td>-20</td></tr><tr><td colspan="2">Hi-En Fr-En</td><td>Ar-En</td><td>De-En</td><td></td></tr></table>

Figure 2: Relative WER and CER reductions of SyRA-CS against CFT, computed as $1 0 0 ( e _ { \mathrm { C F T } } - e _ { \mathrm { S y R A } } ) / e _ { \mathrm { C F T } } .$ Positive values indicate improvement; negative values indicate higher error.

![](images/71419277338bc7c7a7bc4e0700088ac709adb5a195b3e34a5ba6237c4b3fafbc.jpg)  
Figure 3: Pair-level WER for all four models and training regimes. Values are percentages; lower is better.

![](images/d5b760d9a1191ed1b04f0225243bc0b03824781f8d560f147618fb1066766960.jpg)

1.43 hours of real speech before the one-hour adaptation split, leaving 104 and 218 test utterances. To assess dependence on these sets, we recompute the CFT–SyRA-CS macro comparison after excluding German-English and then after excluding both pairs (Figure 6). Without German-English, relative macro WER reductions are 16.2% for XLS-R, 4.3% for Nemotron, 4.6% for Whisper, and 5.9% for Audex, with CER reductions of 13.8%, 3.9%, 8.0%, and 8.2%. Retaining only Hindi-English and French-English, WER reductions are 9.8%, 3.5%, 6.3%, and 5.3%, and CER reductions are 8.9%, 3.9%, 14.5%, and 8.7%. All four models therefore keep positive macro gains on both metrics in every subset. XLS-R’s gain shrinks as its two strongest pairs are removed, whereas Whisper’s CER gain grows once its Arabic-English regression is excluded. These exclusions are descriptive sensitivity checks, not significance tests. All averages weight pairs equally; pooled error rates and paired uncertainty estimates require utterance-level edit statistics, and variability across training seeds and adaptation subsets is not assessed here.

Figure 4: Pair-level CER for all four models and training regimes. Values are percentages; lower is better.  
![](images/3db1299c568c5a87cc46c60f83fa3c066274067074d3d6c672f6233109398844.jpg)  
Figure 5: Code-switching ratio in each target test set.

## 6 Discussion

The results support the central hypothesis: synthetic CS speech is more effective when used to learn an adaptation-friendly initialization than when used only through ordinary sequential finetuning. This is shown by the consistent SyRA-CS improvements over CFT, the closest matched baseline. The largest Reptile benefit appears where the pretrained model is furthest from a usable CS recognizer. XLS-R is pretrained only with self-supervision and its CTC head is learned from scratch, so the initialization carries the most weight: real-only fine-tuning nearly fails on Hindi-English (WER 0.980), and SyRA-CS yields the largest gain (17.7% macro WER). Nemotron, Whisper, and Audex already have supervised ASR pretraining, start from stronger CFT baselines, and gain less (5.3– 5.8%). This suggests that a meta-learned initialization matters most when the backbone lacks taskspecific supervision, although a single model per architecture prevents a causal claim. Whisper’s Arabic-English result does not contradict this picture. Its WER improves while its CER worsens, a pattern that indicates character-level edits within otherwise correctly recognized words rather than additional word errors; this is the signature of Arabic orthographic variation that our shared, scriptagnostic normalization does not collapse. We therefore read the regression as a normalization artifact rather than a failure of the initialization. Practically, SyRA-CS uses no data beyond what CFT uses, making it a low-cost alternative under a onehour real-data budget, provided the additional metatraining compute is acceptable.

![](images/3f2e5067c5978c922cd65364e8136aa6cafba996ec24c10f3533d9eccc7474d0.jpg)  
Figure 6: Sensitivity of relative macro WER/CER reduction to the retained language pairs. Positive values favor SyRA-CS. Lines connect subset summaries; they do not represent a time trend or uncertainty intervals.Each retained pair receives equal weight.

## 7 Conclusion

We introduced SyRA-CS, a pair-specific Reptile framework for synthetic-to-real CS ASR adaptation. Across four pretrained models spanning encoder–CTC, FastConformer–RNN-T, AED, and Speech-LLM architectures, and with the same synthetic source data and one-hour real adaptation budget within each model, SyRA-CS lowers WER relative to continual fine-tuning in all 16 model–pair comparisons and CER in 15, with relative macro WER reductions of 5.3% to 17.7%. Gains are largest for XLS-R, smaller but consistent for the other three models, and remain positive when the two smallest evaluation sets are excluded. These results are consistent with the intended role of Reptile: learning a lightweight initialization that specializes readily to real CS speech. Future work will add compute-matched controls, multi-seed significance testing, CS diagnostics for all models, additional language pairs and more spontaneous speech corpora.

## Limitations

This study has four main limitations.

Pair-specific adaptation. Reptile is applied in a pair-specific setting rather than as a shared multilingual meta-learner. Thus, the experiments evaluate within-pair synthetic-to-real adaptation $p  p ,$ not a shared initialization $\theta _ { a , p } ^ { \mathrm { m e t a } }$ trained jointly over all $\{ S _ { p } \} _ { p \in \mathcal { P } }$ for cross-pair meta-transfer. The results therefore support pair-specific Reptile adaptation, not universal multilingual meta-initialization.

Unexplored synthetic and real data requirements. We evaluate SyRA-CS using fixed synthetic-source datasets and a one-hour labeled real adaptation budget, without systematically varying either data volume. The experiments therefore do not establish the minimum amount of synthetic or real speech needed to achieve a given level of recognition accuracy, nor how these two data sources trade off during adaptation. Controlled data-scaling experiments are needed to characterize these requirements and identify diminishing returns from additional data.

Model coverage and attribution of gains. Although the study includes four architectural settings, each is represented by a single pretrained model. Differences in model capacity, pretraining, tokenization, trainable parameter scope, and optimization prevent attributing differences in Reptile’s effectiveness to architecture alone. Gains are also not uniform (Whisper’s Arabic-English CER increases under SyRA-CS). Finally, we do not compare Reptile with alternative meta-learning algorithms such as MAML or its first-order variant, so we cannot determine whether the benefit is specific to Reptile.

Statistical reliability. All results come from a single training seed and a single random one-hour adaptation subset per pair. German-English and Arabic-English retain only 104 and 218 test utterances after the adaptation split, so several pair-level differences (e.g., Nemotron Hindi-English WER, +1.6%; Whisper Arabic-English WER, +1.1%)

are likely within seed and subset variance. The subset analysis in Section 5.3 is a descriptive sensitivity check, not a significance test; paired utterancelevel statistics and multi-seed runs are needed before the smaller gains can be considered reliable.

## References

Maha Tufail Agro, Atharva Kulkarni, Karima Kadaoui, Zeerak Talat, and Hanan Aldarmaki. 2025. Codeswitching in end-to-end automatic speech recognition: A systematic literature review. Preprint, arXiv:2507.07741.

Ahmed Ali, Walid Magdy, Peter Bell, and Steve Renais. 2015. Multi-reference wer for evaluating asr for languages with no orthographic rules. In 2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), pages 576–580.

Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Moreno, Frank Saunders, Francis M. Tyers, and Gregor Weber. 2020. Common voice: A massivelymultilingual speech corpus. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 4218–4222.

Arun Babu, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexis Conneau, and 1 others. 2021. XLS-R: Self-supervised crosslingual speech representation learning at scale. arXiv preprint arXiv:2111.09296.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Advances in Neural Information Processing Systems, volume 33, pages 12449–12460.

Yaqi Chen, Hao Zhang, Xukui Yang, Wenlin Zhang, and Dan Qu. 2024. Meta-adapter for self-supervised speech models: A solution to low-resource speech recognition challenges. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 11215–11221, Torino, Italia. ELRA and ICCL.

Yaqi Chen, Hao Zhang, Wenlin Zhang, XuKui Yang, Dan Qu, and Yunpeng Liu. 2025. Metamixspeech: Meta task augmentation for low-resource speech recognition. In Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China. Association for Computational Linguistics.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. 2022. FLEURS: Few-shot learning evaluation of universal representations of speech. arXiv preprint arXiv:2205.12446.

Vinit Dhawan, Disha Rekesh, and Boris Ginsburg. 2023. Unified model for code-switching speech recognition and language identification based on concatenated tokenizer. In Proceedings of the Workshop on Computational Approaches to Code Switching, pages 51–62.

Anuj Diwan, Rakesh Vaideeswaran, Sanket Shah, Ankita Singh, Srinivasa Raghavan, Shreya Khare, Vinit Unni, Saurabh Vyas, Akash Rajpuria, Chiranjeevi Yarra, Ashish Mittal, Prasanta Kumar Ghosh, Preethi Jyothi, Kalika Bali, Vivek Seshadri, Sunayana Sitaram, Samarth Bharadwaj, Jai Nanavati, Raoul Nanavati, and Karthik Sankaranarayanan. 2021. Mucs 2021: Multilingual and code-switching asr challenges for low resource indian languages. In Proceedings ofInterspeech 2021, pages 2686–2690.

C. M. Downey, Terra Blevins, Dhwani Serai, Dwija Parikh, and Shane Steinert-Threlkeld. 2024. Targeted multilingual adaptation for low-resource language families. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15647– 15663, Miami, Florida, USA. Association for Computational Linguistics.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning, pages 1126–1135.

Alex Graves, Santiago Fernendez, Faustino Gomez, and´ Jurgen Schmidhuber. 2006. Connectionist temporal¨ classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings ofthe 23rd International Conference on Machine Learning (ICML ’06), pages 369–376. ACM.

Injy Hamed, Thang Vu, and Nizar Habash. 2025. The impact of code-switched synthetic data quality is task dependent: Insights from mt and asr. In Proceedings of the 7th Workshop on Computational Approaches to Linguistic Code-Switching, pages 6–17.

Jui-Yang Hsu, Yuan-Jui Chen, and Hung-yi Lee. 2020. Meta learning for end-to-end low-resource speech recognition. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7844–7848.

Ming-Hao Hsu and Hung-yi Lee. 2024. Smile: Speech meta in-context learning for low-resource language automatic speech recognition. arXiv preprint arXiv:2409.10429.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460.

Karima Kadaoui, Maryam Al Ali, Hawau Olamide Toyin, Ibrahim Mohammed, and Hanan Aldarmaki. 2024. PolyWER: A holistic evaluation framework for

code-switched speech recognition. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pages 6144–6153, Miami, Florida, USA. Association for Computational Linguistics.

Vladimir I Levenshtein. 1966. Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8):707–710.

Ying Li and Pascale Fung. 2012. Mixed error rate for code-switching speech recognition. 2012 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 5535–5538.

Dau-Cheng Lyu, Tien-Ping Tan, Eng Siong Chng, and Haizhou Li. 2010. SEAME: A mandarin-english code-switching speech corpus in south-east asia. In Proceedings ofINTERSPEECH, pages 1986–1989.

Tuan Nguyen and Huy-Dat Tran. 2025a. Asyncswitch: Asynchronous text-speech adaptation for code-switched asr. arXiv preprint arXiv:2506.14190.

Tuan Nguyen and Huy-Dat Tran. 2025b. Can we train asr systems on code-switch without real code-switch data? case study for singapore’s languages. arXiv preprint arXiv:2506.14177.

Alex Nichol, Joshua Achiam, and John Schulman. 2018. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999.

NVIDIA. 2026a. Nemotron 3.5 asr model card. https://huggingface.co/nvidia/nemotron-3. 5-asr-streaming-0.6b. Accessed October 2026.

NVIDIA. 2026b. Nemotron-labs-audex-2b model card. https://huggingface.co/nvidia/ Nemotron-Labs-Audex-2B. Accessed October 2026.

OpenAI. 2026. Whisper-small model card. https: //huggingface.co/openai/whisper-small. Accessed October 2026.

Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, and Michael Auli. 2023. Scaling speech technology to 1,000+ languages. arXiv preprint arXiv:2305.13516.

Sunayana Sitaram, Khyathi Chandu, Sai Krishna Rallabandi, and Alan W. Black. 2019. A survey of codeswitched speech and language processing. arXiv preprint arXiv:1904.00784.

Rais Vaza Man Tazakka, Dessi Puji Lestari, Ayu Purwarianti, Dipta Tanaya, Kurniawati Azizah, and Sakriani Sakti. 2024. Indonesian-english codeswitching speech recognition using the machine speech chain based semi-supervised learning. In Proceedings of the 3rd Annual Meeting of the Special Interest Group on Under-resourced Languages@ LREC-COLING 2024, pages 143–148.

Enes Yavuz Ugan, Ngoc-Quan Pham, and Alex Waibel. 2024. Decm: Evaluating bilingual asr performance on a code-switching/mixing benchmark. In Proceedings Of The 2024 Joint International Conference On Computational Linguistics, Language Resources And Evaluation (LREC-COLING 2024), pages 4468– 4475.

Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. 2021. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 993–1003.

Genta Indra Winata, Samuel Cahyawijaya, Zhaojiang Lin, Zihan Liu, Peng Xu, and Pascale Fung. 2020. Meta-transfer learning for code-switched speech recognition. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3770–3776.

Peng Xie, Xingyuan Liu, Yequan Bie, Tsz Wai Chan, Yangqiu Song, Yang Wang, Hao Chen, and Kani Chen. 2025. Switchlingua: The first large-scale multilingual and multi-ethnic code-switching dataset. In Advances in Neural Information Processing Systems 38: Datasets and Benchmarks Track. Neural Information Processing Systems. Poster.

Brian Yan, Injy Hamed, Shuichiro Shimizu, Vasista Lodagala, William Chen, Olga Iakovenko, Bashar Talafha, Amir Hussein, Alexander Polok, Kalvin Chang, and 1 others. 2025. CS-FLEURS: A massively multilingual and code-switched speech dataset. arXiv preprint arXiv:2509.14161.

Shuaishuai Ye, Shunfei Chen, Xinhui Hu, and Xinkang Xu. 2024. Sc-moe: Switch conformer mixture of experts for unified streaming and non-streaming codeswitching asr. arXiv preprint arXiv:2406.18021.