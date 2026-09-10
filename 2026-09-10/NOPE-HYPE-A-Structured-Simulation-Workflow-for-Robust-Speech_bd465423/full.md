# NOPE-HYPE: A Structured Simulation Workflow for Robust Speech-to-Text Across Diverse Acoustic Environments

Niramay M. Patel IISER Bhopal, India niramay23@iiserb.ac.in

Bibek Behera IIT Bombay, India bibek.iitkgp@gmail.com

Raksha Sharma IIT Roorkee, India raksha.sharma@cs.iitr.ac.in

Abstract—Robust speech-to-text translation systems should perform reliably across diverse acoustic conditions, yet practical pipelines lack controllable tools for systematic environment exploration. Large speech models remain sensitive to unseen acoustic conditions, as training data rarely cover the full range of real environments. We present NOPE-HYPE, a structured training workflow that combines a controllable environment simulator, coverage-optimal environment reduction on Power Spectral Density (PSD) templates, and a small, interpretable hyperparameter search over simulator knobs. We show that simulator-generated noise achieves performance comparable to balanced realnoise training across Whisper and SeamlessM4T models, provide principled environment prototype sets, and identify practical default simulator configurations from a structured 27-run hyperparameter sweep.

Index Terms—speech recognition, noise augmentation, acoustic robustness, environment simulation, hyperparameter search

## I. INTRODUCTION

Speech-to-text (S2TT) translation systems often operate outside curated studio audio settings. The audio of the deployment often includes transportation noise, indoor hums, transient events, and speech-like interference from nearby people. Even strong speech foundation models can degrade under such shifts because training data and augmentation choices rarely cover the full space of real acoustic variability in a controlled way. As shown in Table I, models trained only on clean speech perform worse than those trained with noise injection.

A common mitigation is noise augmentation. In practice, this collapses into a single axis, such as Gaussian noise, a single colored-noise type, or a narrow SNR range [1], [2]. These choices are convenient, but they do not isolate factors that matter in real environments — the spectral shape differs by scene, energy changes over time, transient activity varies, and human activity adds interference that overlaps with speech. Learned audio generators can increase realism, but their latent factors are often hard to interpret and sweep independently, limiting their usefulness for a targeted robustness search [7]–[9].

In this paper, we shift the focus from adding arbitrary noise to systematically modeling environment-specific noise to build an S2TT system that performs well under both clean and noisy conditions. We propose NOPE-HYPE, a practical workflow designed to improve systematic robustness through interpretable control and structured search. We test our framework on two language pairs: English to Hindi and English to German. The workflow has three parts:

• SIMULATOR (SIM): A parameterized simulator for environmental noise and noise-in-speech mixtures. Rather than aiming for exactly realistic noise, it enables structured modeling of acoustic factors relevant to speech-to-text performance.

• Environment Reduction: When many environmental categories exist, uniform sampling becomes computationally expensive because each environment requires a full hyperparameter sweep of the simulator. We summarize each environment using a PSD template and select a small subset of prototypes that covers the full set under explicit distances.

• Hyperparameter Search: We perform a sequential and interpretable search over simulator parameters that yield a robust default configuration and clarify which acoustic factors strongly influence model performance.

The paper is organized as follows. Section II reviews robust speech recognition and augmentation. Section III summarizes the NOPE-HYPE workflow, including the controllable noise simulator (SIM) and environment reduction. Section VI describes experiments evaluating Whisper and SeamlessM4T, including ablations and a 27-run hyperparameter sweep. Sections X and XI discuss the results and summarize the main findings.

## II. RELATED WORK

Robust automatic speech recognition (ASR) has often used acoustic augmentation to improve performance in noisy conditions, including additive noise [1], [2], reverberation [3], speed and pitch perturbation [1], and time-frequency masking [4]. SpecAugment [4] established structured masking as a strong baseline, with later extensions to mixed-sample and hidden-space augmentation [5], [6]. However, these methods typically rely on coarse heuristics and do not allow systematic exploration of environment structure.

Synthetic data pipelines generate diverse acoustic conditions via simulated noise and reverberation, using either simple RIR/noise addition [1], [3] or generative models for realistic noise synthesis [10], [11]. While effective for robustness, they often trade simplicity for controllability, making structured ablation difficult [12].

Large-scale S2TT systems such as Whisper [13] and SeamlessM4T [14] achieve strong baseline robustness, but remain sensitive to training distributions and unseen noise. Evaluations typically focus on fixed benchmarks rather than systematic augmentation design or environment reduction, leaving limited understanding of how environment structure impacts robustness.

## III. OVERVIEW OF NOPE-HYPE

NOPE-HYPE is a structured workflow for improving and analyzing acoustic robustness in speech-totext systems, where robustness refers to maintaining strong performance under changing noise conditions and environment types. Instead of treating noise augmentation as a fixed pre-processing step, it decomposes robustness into three controllable stages.

We use real environmental recordings from the DE-MAND dataset [15] with the environment categories listed in supplementary material. First, each environment is summarized by a Power Spectral Density (PSD) template on a shared frequency grid (SFG). A SFG is used so that all environments are represented on the same frequency bins, allowing their PSD curves to be compared directly. Second, when many environment categories exist, we reduce the set by selecting k prototype environments using a coverage objective over PSD templates. Third, we train with SIM using fitted targets from selected environments, generating controlled environment noise mixed with clean speech at sampled SNR levels, and run structured sweeps over simulator parameters to select robustness-oriented configurations. As summarized in Algorithm 1, this workflow enables direct comparison across environments under a unified training protocol.

## IV. SIM: A CONTROLLABLE ENVIRONMENT AUDIO SIMULATOR

The simulator supports robust training and structured parameter search while providing direct control over key acoustic factors. It produces deterministic outputs under a fixed seed, enabling reliable ablations and controlled comparisons. It targets realistic environmental noise by matching stable statistical properties of real audio rather than trying to sound indistinguishable from real recordings.

<table><tr><td>Algorithm 1 Robustness Workflow 1: Compute PSD template for each environment e</td></tr><tr><td> $x _ { e }$  2: Optionally select few noise environments by coverage optimization on  $\{ x _ { e } \}$ </td></tr><tr><td>3: for each environment e in P (or all environments) do</td></tr><tr><td>4: Fit simulator targets: PSD, envelope stats, transient rate, optional tonal peaks</td></tr><tr><td>5: end for 6: for each simulator configuration θ (default or sweep)</td></tr><tr><td>do 7: Generate noise with simulator using θ 8: Mix with clean speech at sampled SNR (optional</td></tr><tr><td>RIR) 9: Train S2TT model with fixed compute budget</td></tr><tr><td>10: Evaluate on clean, real-noise, and sim-noise test</td></tr><tr><td>sets 11: end for 12: Select θ by robustness objective and report ablations</td></tr></table>

## A. Interface and Modes

SIM provides three commands: gen (ambience-only), mix (speech plus ambience at target SNR), and eval (noise similarity statistics). Synthesis uses $f _ { s } = 1 6$ kHz by default. It supports two modes:

• Preset mode: A fully procedural synthesis pipeline built from layered primitives. It generates a colored-noise bed (white, pink, or brown), applies multi-band spectral shaping with per-band gain control, and introduces slow stochastic modulation to create nonstationarity. Event primitives such as band-limited transients or ramps can also be added. This mode requires no real recordings and provides full parameter control, making it suitable for bootstrap experiments and factorized sweeps.

• Generated mode: Anchors synthesis to real recordings through targets estimated from real ambience, including a mean PSD curve, envelope log-RMS statistics with a slow-dynamics rate hint, transient-rate estimates from onset peaks, and optional tonal peaks. This mode aims to match the stable spectral and temporal properties of real environments while retaining explicit control over each component.

## B. Generated Mode Core Building Blocks

The generated mode synthesizes noise from realenvironment recordings by matching PSD structure, envelope dynamics, transient activity, chatter, and RIR. Fig. 1 summarizes the pipeline.

PSD estimation and PSD-matched bed. We estimate a target PSD using Welch averaging [17] and synthesize a stationary background bed by matching this target spectrum in the Fourier domain, using target magnitudes with randomized phases and a small number of calibration passes to reduce residual PSD mismatch. This design is motivated by the fact that, for stationary or slowly varying background noise, the PSD captures the dominant spectral structure of the environment, while phase randomization provides a simple way to generate noise in the PSD. During calibration, the PSD correction ratios are clipped to bounded ranges to maintain numerical stability and reproducibility.

![](images/750847559d24f2cd0214b8dc2b8395120c4e2611e146d9749353303bcc1f33a5.jpg)  
Fig. 1. SIM model pipeline.

Envelope shaping. We apply a smooth multiplicative envelope a(t) to the stationary PSD-matched bed to model nonstationarity. We generate a low-frequency control signal

$$
u ( t ) = \mathrm { L P F } _ { r _ { e } } ( \epsilon ( t ) ) ,
$$

where $\epsilon ( t )$ is white noise and $\mathrm { L P F } _ { r _ { e } } ( \cdot )$ denotes low-pass filtering with rate parameter $r _ { e }$ . The resulting signal is then exponentiated to produce a positive envelope,

$$
a ( t ) = \exp ( \sigma _ { e } u ( t ) ) ,
$$

where $r _ { e }$ controls how quickly the envelope varies and $\sigma _ { e }$ controls the magnitude of energy fluctuations. This stage allows the simulator to improve beyond a purely stationary background bed and capture the slow amplitude variation observed in real environments.

Transient injection. Real environments contain short impulsive events such as clicks or impacts. We model these by injecting band-limited transient bursts at a target rate λ events per second, shaped by a smooth envelope and filtered within an environment-specific frequency band.

Chatter injection. We model human background activity by mixing short speech segments sampled from a separate speech pool, band-limited and scheduled in burst-like intervals to mimic background murmur or conversation.

RIR. For indoor environments, we optionally apply a room impulse response during mixing to simulate reverberant propagation. This introduces reflections characteristic of enclosed spaces and provides explicit control over reverberation.

## C. Mixing

Given clean speech $s ( t )$ and simulated ambience $x _ { e } ( t )$ we form a noisy utterance at a target SNR:

$$
y ( t ) = s ( t ) + g x _ { e } ( t ) ,
$$

where the scaling factor is

$$
g = \frac { \mathrm { R M S } ( s ) } { \mathrm { R M S } ( x _ { e } ) } 1 0 ^ { - \mathrm { S N R } / 2 0 } .
$$

so that the resulting mixture matches the target SNR. The derivation of this expression is provided in supplementary material. After mixing, we apply mild soft clipping and peak normalization to keep the waveform within a stable numeric range and avoid occasional amplitude spikes.

## V. REDUCING ENVIRONMENT SETS

When many environments are available, sweeping simulator hyperparameters across all environments becomes computationally expensive and often redundant due to similar spectral structure. We reduce the environment set by treating it as a coverage problem in PSD-template space and selecting representative prototypes.

PSD templates and distances. Each environment $e _ { i }$ is represented by a centered PSD template $x _ { i } \in \mathbb { R } ^ { F }$ on a SFG. We compute distances between PSD templates using two complementary measures:

• Correlation distance: This distance captures whether two environments have similar spectral patterns.

$$
d _ { \mathrm { c o r r } } ( x , y ) = 1 - \mathrm { c o r r } ( x , y ) .
$$

• RMSE: This distance measures how close two PSD templates are in absolute magnitude across frequencies.

$$
d _ { \mathrm { r m s e } } ( x , y ) = \sqrt { \frac { 1 } { F } \sum _ { f } ( x _ { f } - y _ { f } ) ^ { 2 } } .
$$

Objectives of prototype selection. Let $\begin{array} { r l } { E } & { { } = } \end{array}$ $\{ e _ { 1 } , \dots , e _ { N } \}$ denote the set of environments represented by PSD templates. Our goal is to select a small subset of environments $P \subseteq E$ of size $k ,$ which will serve as prototypes. For any environment $e _ { i } ,$ we measure its distance from the selected prototype set as

$$
d ( e _ { i } , P ) = \operatorname* { m i n } _ { p \in P } d ( e _ { i } , p ) ,
$$

which assigns each environment to its closest prototype.

Since the number of environments is small $( N = 1 2 )$ we evaluate candidate prototype sets using exhaustive search. For each value of k, we enumerate all possible subsets $P \subseteq E$ with $| P | = k$ and evaluate each subset under both distance metrics. We consider two coverage objectives.

Worst-case coverage, or minimax coverage: As illustrated in the left panel of Fig. 2, this objective is determined by the farthest environment from its nearest prototype. It encourages every environment to remain close to at least one selected prototype:

$$
\operatorname* { m i n } _ { P \subseteq E , \ | P | = k \ e _ { i } \in E } { \operatorname* { m a x } } d ( e _ { i } , P ) .
$$

Average-case coverage: As illustrated in the right panel of Fig. 2, this objective depends on the mean distance from all environments to their nearest prototypes. It favors prototype sets that provide good overall coverage on average:

$$
\operatorname* { m i n } _ { P \subseteq E , \ | P | = k } \frac { 1 } { N } \sum _ { e _ { i } \in E } d ( e _ { i } , P ) .\tag{1}
$$

This process is conceptually related to clustering methods such as k-means, where cluster centers represent groups of data points. In our setting, the selected environments act as prototype centers, and each environment is assigned to its nearest prototype according to the chosen PSD-template distance.

![](images/c3c3dafbd67c46a0693fd1dcb850770c9ee7aa495fc32958523e839b15c75113.jpg)  
Fig. 2. Prototype-selection objectives: minimax (left) and average-case (right).

Selection of prototypes and choice of k. Using the coverage objectives defined above, we evaluate candidate prototype sets as a function of the number of selected environments k. As shown in Fig. 3, the largest reduction in representation error occurs between k = 1 and $k = 3 .$ after which improvements diminish. Based on worstcase coverage, we treat $k ~ = ~ 3$ and $k \ = \ 4$ as the main operating points. The set $k = 3$ provides compact coverage suitable for efficient experimentation, while k = 4 slightly improves worst-case coverage. The exact prototype sets and coverage metrics obtained through exhaustive evaluation are reported in supplementary material.

![](images/b0b04f2e761f6a7cf7c7c31557f66e685fd946806315e1020a9f258ffa961900.jpg)  
Fig. 3. Prototype representation quality vs. number of selected environments k. Both RMSE and correlation-distance exhibit diminishing returns after $k \in \{ 3 , 4 \}$

## VI. EXPERIMENTAL PROTOCOL

We used the Indic-ST corpus [16], focusing on 50 hours of English-Hindi speech for training and a 5-hour heldout evaluation set, with all audio processed at 16 kHz. We also use the same setting for English-German speech from the CoVoST dataset [18]. We evaluate robustness on Whisper [13] and SeamlessM4T [14], fine-tuning mediumsized pretrained checkpoints and evaluating Hindi and German generation on clean, real-noise, and simulatornoise conditions. We report BLEU, chrF, and WER as evaluation metrics. Full training and decoding details are provided in supplementary material.

## VII. PRIMARY ROBUSTNESS COMPARISON

We benchmark the primary robustness comparison on the English-Hindi dataset under five training noise conditions using the same 50-hour training budget. The regimes include clean-only training, Gaussian noise, pink noise, balanced real environment noise (uniformly sampled from 12 environments), and SIM-generated noise.

The key outcome is the close similarity between realnoise and simulator-noise training (Table I). In SeamlessM4T, the simulator slightly improves WER relative to real-noise training while maintaining comparable BLEU and chrF. On Whisper, the two are essentially tied. These results show that SIM-generated noise achieves performance comparable to balanced real-noise training under the evaluated conditions.

TABLE I  
MAIN ROBUSTNESS COMPARISON (50H TRAIN, 5H TEST).
<table><tr><td></td><td colspan="3">Seamless</td><td colspan="3">Whisper</td></tr><tr><td>Training condition</td><td>BLEU↑</td><td>WER↓</td><td>chrF↑</td><td>BLEU↑</td><td>WER↓</td><td>chrF↑</td></tr><tr><td>Clean only</td><td>47.628</td><td>0.432</td><td>66.400</td><td>34.168</td><td>0.572</td><td>53.034</td></tr><tr><td>Gaussian</td><td>46.523</td><td>0.447</td><td>65.609</td><td>29.994</td><td>0.627</td><td>46.340</td></tr><tr><td>Pink (10 dB)</td><td>47.606</td><td>0.431</td><td>66.848</td><td>34.019</td><td>0.570</td><td>53.117</td></tr><tr><td>Real env (12)</td><td>48.959</td><td>0.429</td><td>67.893</td><td>35.554</td><td>0.556</td><td>54.143</td></tr><tr><td>SIM noise</td><td>48.996</td><td>0.422</td><td>67.896</td><td>35.494</td><td>0.555</td><td>54.160</td></tr></table>

As shown in Table II, Gaussian noise is a weak proxy for real environments, with especially large degradation on Whisper. Pink noise is stronger than Gaussian, but it still remains consistently worse than real-noise and simulator-noise training on both models. This suggests that matching only coarse spectral color is insufficient, and that the consistent gains of SIM noise over Gaussian and pink noise provide a proof of concept that more struc tured and realistic environment simulation is beneficial for robust training. It motivates simulator components that go beyond stationary PSD matching to envelope dynamics, transient activity, and chatter.

TABLE II  
PERFORMANCE GAP RELATIVE TO SIM NOISE. NEGATIVE BLEU/CHRF AND POSITIVE WER INDICATE WORSE PERFORMANCE THAN THE SIMULATOR.
<table><tr><td>Model</td><td>Baseline</td><td>∆BLEU</td><td>∆WER</td><td>∆chrF</td></tr><tr><td>SeamlessM4T</td><td>Gaussian</td><td>-2.473</td><td>0.025</td><td>-2.287</td></tr><tr><td>SeamlessM4T</td><td>Pink</td><td>-1.390</td><td>0.009</td><td>-1.048</td></tr><tr><td>Whisper</td><td>Gaussian</td><td>-5.500</td><td>0.072</td><td>-7.820</td></tr><tr><td>Whisper</td><td>Pink</td><td>-1.475</td><td>0.015</td><td>-1.043</td></tr></table>

TABLE III  
EFFECT OF CLEAN:NOISE MIXTURE RATIO. ENTRIES ARE MEAN ± STD OVER SEEDS.
<table><tr><td>Setting</td><td>Noisy BLEU↑</td><td>Noisy WER↓</td><td>Noisy chrF↑</td></tr><tr><td>clean20_noise80</td><td>46.140±0.430</td><td>0.465±0.008</td><td>66.630±0.230</td></tr><tr><td>clean40_noise60</td><td>46.450±1.200</td><td>0.469±0.019</td><td>66.430±0.260</td></tr><tr><td>clean50_noise50</td><td>46.990±1.580</td><td>0.460±0.026</td><td>66.570±0.310</td></tr><tr><td>clean70_noise30</td><td>47.120±0.480</td><td>0.457±0.008</td><td>66.420±0.150</td></tr><tr><td>Setting</td><td>Clean BLEU↑</td><td>Clean WER↓</td><td>Clean chrF↑</td></tr><tr><td>clean20_noise80</td><td>48.760±0.190</td><td>0.436±0.003</td><td>67.880±0.110</td></tr><tr><td>clean40_noise60</td><td>48.470±0.630</td><td> $0 . 4 4 3 { \pm } 0 . 0 1 2$ </td><td>67.700±0.200</td></tr><tr><td>clean50_noise50</td><td>48.720±0.860</td><td> $0 . 4 4 0 { \pm } 0 . 0 1 4$ </td><td>67.800±0.030</td></tr><tr><td>clean70_noise30</td><td>49.910±0.190</td><td> $\mathbf { 0 . 4 2 5 { \scriptstyle \pm 0 . 0 0 3 } }$ </td><td>67.950±0.130</td></tr></table>

## VIII. IMPACT OF SIMULATOR COMPONENTS

We evaluate whether the simulator controls correspond to meaningful factors for robust training on SeamlessM4T. In each ablation, we keep the training pipeline fixed and vary only one control group.

Chatter ON/OFF. Enabling chatter improves both noisy and clean performance (Table IV), suggesting that speech-like interference is an important factor not well captured by stationary background noise alone.

TABLE IV  
CHATTER ABLATION. ENTRIES ARE MEAN ± STD OVER SEEDS
<table><tr><td>Setting</td><td>Noisy BLEU↑</td><td>Noisy WER↓</td><td>Noisy chrF↑</td></tr><tr><td>CHATTER_OFF</td><td>46.450±0.100</td><td>0.464±0.006</td><td>66.610±0.320</td></tr><tr><td>CHATTER_ON</td><td>47.870±0.140</td><td>0.450±0.001</td><td>67.440±0.860</td></tr><tr><td>Setting</td><td>Clean BLEU↑</td><td>Clean WER↓</td><td>Clean chrF↑</td></tr><tr><td>CHATTER_OFF</td><td>48.240±0.240</td><td>0.445±0.002</td><td> $6 7 . 6 3 0 { \scriptstyle \pm 0 . 4 0 0 }$ </td></tr><tr><td>CHATTER_ON</td><td>49.360±0.030</td><td>0.431±0.000</td><td> $\mathbf { 6 8 . 5 2 0 { \scriptstyle \pm 0 . 1 7 0 } }$ </td></tr></table>

RIR ON/OFF. Enabling RIR reduces performance on both noisy and clean evaluation (Table V), consistent with reverberation acting as a domain shift when training and test reverberation are mismatched [3]. RIR should be treated as an explicit control rather than a default.

ADDITIONAL ABLATIONS ACROSS SIMULATOR COMPONENTS.  
TABLE V
<table><tr><td>Setting</td><td>Noisy BLEU↑</td><td>Noisy WER↓</td><td>Noisy chrF↑</td></tr><tr><td>RIR off RIR on</td><td>45.86 44.90</td><td>0.4593 0.4798</td><td>65.62 64.36</td></tr><tr><td>Bursty Continuous</td><td>45.40 45.48</td><td>0.4676 0.4658</td><td>65.10 65.49</td></tr><tr><td>Full sim PSD only</td><td>46.97 45.07</td><td>0.4464 0.4645</td><td>66.90 65.78</td></tr><tr><td>Setting</td><td>Clean BLEU↑</td><td>Clean WER↓</td><td>Clean chrF↑</td></tr><tr><td>RIR off RIR on</td><td>47.73 47.21</td><td>0.4367 0.4542</td><td>67.10 65.96</td></tr><tr><td>Bursty Continuous</td><td>47.04 47.53</td><td>0.4482 0.4405</td><td>66.80 67.13</td></tr><tr><td>Full sim PSD only</td><td>48.94 47.58</td><td>0.4316 0.4399</td><td>67.90 67.10</td></tr></table>

Continuous vs. bursty chatter. The difference between continuous and bursty chatter is small, as shown in Table V. This suggests that the presence of chatter and its overall level matter more than its fine-grained temporal scheduling in this setting.

PSD-only vs. full simulator. The PSD-only variant performs worse than the full simulator, shown in Table V. This indicates that stationary spectral matching alone is insufficient; temporal dynamics and structured interference contribute meaningfully to robustness.

## IX. HYPERPARAMETER SEARCH OVER SIMULATOR KNOBS

After establishing that the simulator does not degrade performance and testing the importance of the knobs, we conduct a structured sweep over the simulator hyperparameters to systematically assess how different knob configurations affect performance. We fine-tune only the SeamlessM4T model, as it performs better than Whisper based on previous results.

TABLE VI  
SWEEP CONFIGURATIONS (SHARED ACROSS PSD TEMPLATES).
<table><tr><td>Cfg</td><td> $\mu _ { \mathrm { l o g } \mathrm { R M S } }$ </td><td> $\sigma _ { \mathrm { l o g R M S } }$ </td><td> $r _ { e }$  (Hz)</td><td>λ</td><td> $\sigma _ { \lambda }$ </td><td>µSNR</td><td> $\sigma _ { \mathrm { S N R } }$ </td></tr><tr><td>cfg01</td><td>-6.600</td><td>0.300</td><td>0.015</td><td></td><td>2.2000.300</td><td>0.000</td><td>3.000</td></tr><tr><td>cfg02</td><td>-6.600</td><td>0.300</td><td>0.046</td><td></td><td>3.6000.600</td><td>10.000</td><td>4.000</td></tr><tr><td>cfg03</td><td>-6.600</td><td>0.300</td><td>0.107</td><td></td><td>4.8000.250</td><td>20.000</td><td>5.000</td></tr><tr><td>cfg04</td><td>-5.200</td><td>0.500</td><td>0.015</td><td></td><td>3.6000.600</td><td>20.000</td><td>5.000</td></tr><tr><td>cfg05</td><td>-5.200</td><td>0.500</td><td>0.046</td><td></td><td>4.8000.250</td><td>0.000</td><td>3.000</td></tr><tr><td>cfg06</td><td>-5.200</td><td>0.500</td><td>0.107</td><td></td><td>2.2000.300</td><td>10.000</td><td>4.000</td></tr><tr><td>cfg07</td><td>-3.300</td><td>0.700</td><td>0.015</td><td></td><td>4.8000.250</td><td>10.000</td><td>4.000</td></tr><tr><td>cfg08</td><td>-3.300</td><td>0.700</td><td>0.046</td><td></td><td>2.2000.300</td><td>20.000</td><td>5.000</td></tr><tr><td>cfg09</td><td>-3.300</td><td>0.700</td><td>0.107</td><td></td><td>3.6000.600</td><td>0.000</td><td>3.000</td></tr></table>

We conduct a structured sweep consisting of 27 experiments on both the language pair datasets: 9 simulator configurations (Table VI) evaluated across 3 fitted PSD templates—cafe, washing (laundry), and car (Section V). Each configuration specifies a set of simulator controls governing background energy statistics, temporal dynamics, and speech-to-noise mixing.

The simulator configurations are defined by a set of parameters; the full list and definitions are provided in supplementary material. We select the best configurations by minimizing the robustness objective

$$
\mathrm { A v g W E R } _ { \mathrm { r o b } } = \frac { 1 } { 2 } \left( \mathrm { W E R } _ { \mathrm { r e a l } } + \mathrm { W E R } _ { \mathrm { s i m } } \right) ,\tag{2}
$$

while monitoring clean WER to avoid sacrificing clean accuracy.

## A. Sweep results and selected default

We summarize the sweep outcomes by aggregating performance across the three PSD templates. The goal is to identify configurations that perform consistently well under the robustness objective in Equation 2.

As shown in Table VII and Table VIII, configuration 3 (cfg03) and configuration 6 (cfg06) achieve the strongest robustness outcomes in this sweep, with consistently low variance across environments:

• cfg03: $\mu _ { \mathrm { l o g \tiny ~ R M S } } ~ = ~ - 6 . 6 , ~ \sigma _ { \mathrm { l o g \tiny ~ R M S } } ~ = ~ 0 . 3 , ~ r _ { e } ~ = ~$ 0.106812 Hz, λ = 4.8, σ<sub>λ</sub> = 0.25, SNR ∼ N(20, 5) dB.

• cfg06: $\mu _ { \mathrm { l o g \tiny ~ R M S } } \ = \ - 5 . 2 , \ \sigma _ { \mathrm { l o g \tiny ~ R M S } } \ = \ 0 . 5 , \ r _ { e } \ =$ 0.106812 Hz, λ = 2.2, σ<sub>λ</sub> = 0.30, SNR ∼ N (10, 4) dB.

TABLE VII  
SWEEP RESULTS AGGREGATED OVER PSD TEMPLATES (CAFE, WASHING, CAR). ENTRIES ARE MEAN ± STD OVER ENVIRONMENTS.
<table><tr><td>Cfg</td><td>Clean WER↓</td><td>Real WER↓</td><td>Sim WER↓</td><td> $\mathrm { A v g W E R } _ { \mathrm { r o b } } \downarrow$ </td></tr><tr><td>cfg01</td><td>0.427±0.003</td><td>0.450±0.005</td><td>0.442±0.002</td><td>0.446±0.003</td></tr><tr><td>cfg02</td><td>0.425±0.002</td><td>0.444±0.005</td><td>0.443±0.011</td><td>0.443±0.007</td></tr><tr><td>cfg03</td><td>0.422±0.001</td><td>0.437±0.005</td><td>0.439±0.010</td><td>0.438±0.007</td></tr><tr><td>cfg04</td><td>0.434±0.012</td><td>0.452±0.016</td><td>0.446±0.016</td><td>0.449±0.016</td></tr><tr><td>cfg05</td><td>0.432±0.009</td><td>0.449±0.009</td><td>0.447±0.007</td><td>0.448±0.008</td></tr><tr><td>cfg06</td><td>0.428±0.001</td><td>0.437±0.001</td><td>0.433±0.003</td><td>0.435±0.001</td></tr><tr><td>cfg07</td><td>0.438±0.008</td><td>0.456±0.005</td><td>0.456±0.013</td><td>0.456±0.009</td></tr><tr><td>cfg08</td><td>0.426±0.013</td><td>0.450±0.009</td><td>0.443±0.010</td><td>0.446±0.008</td></tr><tr><td>cfg09</td><td>0.438±0.010</td><td>0.454±0.009</td><td>0.449±0.008</td><td>0.452±0.009</td></tr></table>

TABLE IX  
BEST RUN PER PSD TEMPLATE (MIN $\mathrm { A v g W E R } _ { \mathrm { r o b } } )$ FOR EN-HI.
<table><tr><td>Env</td><td>Best exp</td><td>Clean WER</td><td>Real WER</td><td>Sim WER</td><td> $\mathrm { A v g W E R } _ { \mathrm { r o b } }$ </td></tr><tr><td>cafe</td><td>exp02</td><td>0.424</td><td>0.438</td><td>0.433</td><td>0.436</td></tr><tr><td>washing</td><td>exp12</td><td>0.420</td><td>0.432</td><td>0.429</td><td>0.430</td></tr><tr><td>car</td><td>exp24</td><td>0.426</td><td>0.438</td><td>0.431</td><td>0.434</td></tr></table>

TABLE VIII

SWEEP RESULTS FOR ENGLISH TO GERMAN AGGREGATED OVER PSD TEMPLATES. ENTRIES ARE MEAN ± STD OVER ENVIRONMENTS.
<table><tr><td>Cfg</td><td>Clean WER↓</td><td>Real WER↓</td><td>Sim WER↓</td><td> $\mathrm { A v g W E R } _ { \mathrm { r o b } } \downarrow$ </td></tr><tr><td>cfg01</td><td> $0 . 5 5 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 5 8 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 5 8 2 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 5 8 1 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>cfg02</td><td> $0 . 5 5 3 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 5 8 2 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 5 8 3 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 5 8 3 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>cfg03</td><td> $\mathbf { 0 . 5 4 9 } \pm \mathbf { 0 . 0 0 } 2$ </td><td> $\mathbf { 0 . 5 7 8 { \overset { . } { \bot } } 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 5 8 0 } \pm \mathbf { 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 5 7 9 } \pm \mathbf { 0 . 0 0 } 2$ </td></tr><tr><td>cfg04</td><td> $0 . 5 5 3 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 5 8 1 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 5 8 3 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 5 8 2 { \pm } 0 . 0 0 5$ </td></tr><tr><td>cfg05</td><td> $0 . 5 5 2 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 5 8 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>0.582±0.004</td><td> $0 . 5 8 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>cfg06</td><td> $\mathbf { 0 . 5 5 0 { \overset { . } { \bot } } 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 5 7 9 } \pm \mathbf { 0 . 0 0 } 2$ </td><td>0.581±0.000</td><td> $\mathbf { 0 . 5 8 0 { \overset { . } { \bot } } 0 . 0 0 1 }$ </td></tr><tr><td>cfg07</td><td> $0 . 5 5 2 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 5 7 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td>0.582±0.005</td><td> $0 . 5 8 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>cfg08</td><td> $0 . 5 5 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.581±0.005</td><td>0.585±0.008</td><td> $0 . 5 8 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td>cfg09</td><td> $0 . 5 5 1 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td>0.580±0.005</td><td>0.581±0.001</td><td> $0 . 5 8 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr></table>

![](images/b6eb1e57557cd9f07ac17d5ffb1d342985c80933857dad64a52074d370f13724.jpg)  
Fig. 4. Sweep sensitivity across language pairs. Values show the WER increase relative to the best configuration for each metric and language pair, using the clean WER and $\mathrm { A v g W E R } _ { \mathrm { r o b } }$ columns from Tables VII and VIII. Lower is better.

Fig. 4 shows that cfg03 and cfg06 are the most stable choices across both language pairs and across clean and robust evaluation. We do not claim that cfg03 or cfg06 is globally optimal. Rather, they are strong defaults within this small structured search space and across the three PSD templates studied. The comparison suggests that faster envelope dynamics $( r _ { e } )$ is consistently helpful, while robustness can be obtained either through a higher-SNR level with higher event rates (cfg03) or through a moderate-SNR level with lower event rates (cfg06). These trends are empirical for our setting and may change across deployment conditions.

To examine whether the selected default varies by environment, we also report the best run for each PSD template separately.

The best configuration varies slightly by PSD template, as shown in Table IX and Table X, which is expected because different spectral shapes and temporal properties imply different masking regimes. In practice, cfg03 and cfg06 provide good general defaults, while perenvironment tuning can be treated as a second-stage refinement when deployment is dominated by a single environment type.

TABLE X  
BEST RUN PER PSD TEMPLATE (MIN $\mathrm { A v g W E R } _ { \mathrm { r o b } } )$ FOR EN-DE.
<table><tr><td>Env</td><td>Best exp</td><td>Clean WER</td><td>Real WER</td><td>Sim WER</td><td> $\mathrm { A v g W E R } _ { \mathrm { r o b } }$ </td></tr><tr><td>cafe</td><td>exp04</td><td>0.554</td><td>0.578</td><td>0.581</td><td>0.580</td></tr><tr><td>washing</td><td>exp12</td><td>0.548</td><td>0.576</td><td>0.580</td><td>0.578</td></tr><tr><td>car</td><td>exp19</td><td>0.547</td><td>0.577</td><td>0.577</td><td>0.577</td></tr></table>

## X. DISCUSSION

The central design choice in NOPE-HYPE is controllability, so that changes in robustness can be attributed to explicit simulator controls or selection objectives. Training with simulator-generated noise matches balanced realnoise training under the same data budget and evaluation conditions, indicating that the simulator captures the statistics that matter for robustness in the studied settings. This framework highlights that prototype selection is most useful when the environment catalog is large and training budgets are limited. Compute PSD templates, select k environments via the coverage objective, and focus fitting and tuning on those prototypes. A conservative workflow is to start with $k = 3$ prototypes for rapid iteration, fit simulator targets from real recordings for those prototypes, train with cfg03 or cfg06 as robust default configurations, and then use ablations to decide whether chatter and RIR are appropriate for the intended deployment domain. When deployment is dominated by a single environment type, the same procedure can be refined by tuning knobs per PSD template using the per-environment selection approach.

## XI. CONCLUSION

We introduced NOPE-HYPE, a structured robustness workflow for speech-to-text training that combines a controllable environment simulator, coverage-optimal environment reduction on PSD templates, and a structured hyperparameter search over simulator knobs. We demonstrated performance comparable to balanced realnoise training across Whisper and SeamlessM4T. Through ablations, we showed that dynamics matter beyond stationary spectral matching and provided principled environment prototype sets. Finally, we selected practical default simulator configurations from a 27-run sweep. Together, these components form a reproducible and interpretable path toward robust S2TT training.

## REFERENCES

[1] T. Ko, V. Peddinti, D. Povey, and S. Khudanpur, “Audio augmentation for speech recognition,” in Proc. Interspeech, 2015, pp. 3586–3589, doi: 10.21437/Interspeech.2015-711.

[2] A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta, A. Coates, and A. Y. Ng, “Deep Speech: Scaling up end-to-end speech recognition,” arXiv:1412.5567, 2014.

[3] T. Ko, V. Peddinti, D. Povey, M. L. Seltzer, and S. Khudanpur, “A study on data augmentation of reverberant speech for robust speech recognition,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2017, pp. 5220–5224, doi: 10.1109/ICASSP.2017.7953152.

[4] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, “SpecAugment: A simple data augmentation method for automatic speech recognition,” in Proc. Interspeech, 2019, pp. 2613–2617, doi: 10.21437/Interspeech.2019-2680.

[5] G. Kim, D. K. Han, and H. Ko, “SpecMix: A mixed sample data augmentation method for training with time-frequency domain features,” arXiv:2108.03020, 2021.

[6] H. Wang, Y. Zou, and W. Wang, “SpecAugment++: A hidden space data augmentation method for acoustic scene classification,” arXiv:2103.16858, 2021.

[7] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, “WaveNet: A generative model for raw audio,” arXiv:1609.03499, 2016.

[8] K. Kumar, R. Kumar, T. de Boissiere, L. Gestin, W. Z. Teoh, J. Sotelo, A. de Brebisson, Y. Bengio, and A. Courville, “Mel-GAN: Generative adversarial networks for conditional waveform synthesis,” arXiv:1910.06711, 2019.

[9] H. Liu, Z. Chen, Y. Yuan, X. Mei, X. Liu, D. Mandic, W. Wang, and M. D. Plumbley, “AudioLDM: Text-to-audio generation with latent diffusion models,” arXiv:2301.12503, 2023.

[10] A. Fazel, W. Yang, Y. Liu, R. Barra-Chicote, Y. Meng, R. Maas, and J. Droppo, “SynthASR: Unlocking synthetic data for speech recognition,” arXiv:2106.07803, 2021.

[11] E. Casanova, C. Shulby, A. Korolev, A. Candido Junior, A. da Silva Soares, S. Alu´ısio, and M. A. Ponti, “ASR data augmentation in low-resource settings using cross-lingual multi-speaker TTS and cross-lingual voice conversion,” arXiv:2204.00618, 2023.

[12] D. Liu, A. Nassereldine, C. Xu, and J. Xiong, “Towards pretraining robust ASR foundation model with acoustic-aware data augmentation,” arXiv:2505.20606, 2025.

[13] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” arXiv:2212.04356, 2022.

[14] Seamless Communication et al., “SeamlessM4T: Massively multilingual & multimodal machine translation,” arXiv:2308.11596, 2023.

[15] J. Thiemann, N. Ito, and E. Vincent, “DEMAND: A collection of multi-channel recordings of acoustic noise in diverse environments,” in Proc. Int. Congr. Acoust. (ICA), 2013.

[16] S. Shah, K. R. Saxena, K. M. Bharadwaj, S. Adavanne, and N. Adiga, “IndicST: Indian multilingual translation corpus for evaluating speech large language models,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. Workshops (ICASSPW), 2025, pp. 1–5, doi: 10.1109/ICASSPW65056.2025.11011192.

[17] P. Welch, “The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms,” IEEE Trans. Audio Electroacoust., vol. 15, no. 2, pp. 70–73, 1967, doi: 10.1109/TAU.1967.1161901.

[18] C. Wang, J. Pino, A. Wu, and J. Gu, “CoVoST: A diverse multilingual speech-to-text translation corpus,” arXiv:2002.01320, 2020.

## ACKNOWLEDGMENT

AI-Generated Content Disclosure: The authors used AI-based writing and research-assistance tools, including OpenAI ChatGPT, Anthropic Claude Sonnet, Google

Gemini 3.1 Pro, and xAI Grok 4.3, to assist with manuscript review, language polishing, grammar correction, organization suggestions, LaTeX formatting suggestions, and drafting of explanatory text. AI assistance was applied to portions of the Abstract, Introduction, Related Work, Method description, Experimental Protocol, Discussion, Conclusion, and Acknowledgment sections.

The AI systems were not used to fabricate experimental results, datasets, evaluation metrics, figures, tables, or citations. All experimental design choices, simulator implementation, training runs, evaluation results, analyses, and final scientific claims were reviewed, edited, and verified by the authors, who take full responsibility for the final content of the paper.

# SUPPLEMENTARY MATERIAL

## APPENDIX A METHOD DETAILS

## A. Derivation of Noise Scaling Factor

Let s(t) denote the clean speech signal and $x _ { e } ( t )$ the simulated ambience. The mixed signal is

$$
y ( t ) = s ( t ) + g x _ { e } ( t ) ,\tag{3}
$$

where $g$ is chosen to achieve a target signal-to-noise ratio (SNR).

We define the SNR in decibels using RMS amplitudes:

$$
\mathrm { S N R } = 2 0 \log _ { 1 0 } \left( \frac { \mathrm { R M S } ( s ) } { \mathrm { R M S } ( g x _ { e } ) } \right) .\tag{4}
$$

Using the scaling property of RMS,

$$
\mathrm { R M S } ( g x _ { e } ) = g \mathrm { R M S } ( x _ { e } ) ,\tag{5}
$$

we obtain

$$
\mathrm { S N R } = 2 0 \log _ { 1 0 } \left( \frac { \mathrm { R M S } ( s ) } { g \mathrm { R M S } ( x _ { e } ) } \right) ,\tag{6}
$$

$$
\frac { \mathrm { S N R } } { 2 0 } = \log _ { 1 0 } \left( \frac { \mathrm { R M S } ( s ) } { g \mathrm { R M S } ( x _ { e } ) } \right) ,\tag{7}
$$

$$
1 0 ^ { \mathrm { S N R / 2 0 } } = \frac { \mathrm { R M S } ( s ) } { g \mathrm { R M S } ( x _ { e } ) } .\tag{8}
$$

Solving for g gives

$$
g = \frac { \mathrm { R M S } ( s ) } { \mathrm { R M S } ( x _ { e } ) } 1 0 ^ { - \mathrm { S N R } / 2 0 } .\tag{9}
$$

This is the scaling factor used during speech-noise mixing in SIM.

## APPENDIX B EXPERIMENTAL DETAILS

## A. Training and Decoding Details

Table XI summarizes the principal training and decoding settings used for the evaluated models.

TABLE XI  
TRAINING AND DECODING SETTINGS FOR SEAMLESSM4T AND WHISPER.
<table><tr><td>Setting</td><td>SeamlessM4T</td><td>Whisper</td></tr><tr><td>Checkpoint</td><td>hf-seamless-m4t-medium</td><td>whisper-medium</td></tr><tr><td>Trainer</td><td>HuggingFace Trainer</td><td>HuggingFace Seq2SeqTrainer</td></tr><tr><td>Audio sampling rate</td><td>16 kHz</td><td>16 kHz</td></tr><tr><td>Batch size</td><td>4 4</td><td>8</td></tr><tr><td>Gradient accumulation</td><td></td><td>4</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>3 × 10−5 Cosine</td></tr><tr><td>Scheduler</td><td>Cosine</td><td></td></tr><tr><td>Warmup ratio</td><td>0.1 0.03</td><td>0.1</td></tr><tr><td>Weight decay</td><td></td><td>0.03</td></tr><tr><td>Maximum gradient norm</td><td>0.5</td><td>0.5</td></tr><tr><td>Epoch budget</td><td>Up to 10</td><td>Up to 10</td></tr><tr><td>Early stopping</td><td>Patience 5, threshold 0.005</td><td>Patience 5, threshold 0.005</td></tr><tr><td>Precision</td><td>fp16 Yes</td><td>bf16 if available, otherwise fp16</td></tr><tr><td>Gradient checkpointing</td><td></td><td>Yes</td></tr><tr><td>Target decoding language</td><td>Hindi (en-hi); German (en-de)</td><td>Hindi (en-hi); German (en-de)</td></tr><tr><td>Beam size</td><td>4</td><td>4</td></tr><tr><td>Maximum new tokens</td><td>160</td><td>160</td></tr></table>

## B. Coverage-Based Environment Reduction

Table XII reports the environment prototypes obtained under the PSD-based coverage analysis described in the main paper. The prototypes are actual environments rather than synthetic cluster centroids.

TABLE XII  
PROTOTYPE ENVIRONMENTS SELECTED USING PSD COVERAGE.
<table><tr><td>k</td><td>Selected prototypes</td><td>max  $d _ { \mathrm { c o r r } }$ </td><td>mean  $d _ { \mathrm { c o r r } }$ </td><td> ${ \mathrm { R M S E } } _ { \mu }$  (dB)</td><td> $\mathrm { R M S E } _ { \operatorname* { m a x } }$  (dB)</td><td> $\operatorname { C o r r } _ { \mu }$ </td><td> ${ \mathrm { C o r r } } _ { \operatorname* { m i n } }$ </td></tr><tr><td>1</td><td>bus</td><td>0.077</td><td>0.034</td><td>1.986</td><td>3.073</td><td>0.966</td><td>0.923</td></tr><tr><td>2</td><td>cafe, office</td><td>0.040</td><td>0.021</td><td>1.316</td><td>2.302</td><td>0.979</td><td>0.966</td></tr><tr><td>3</td><td>cafe, washing, car</td><td>0.032</td><td>0.015</td><td>1.074</td><td>2.089</td><td>0.987</td><td>0.969</td></tr><tr><td>4</td><td>washing, station, car, meeting</td><td>0.028</td><td>0.012</td><td>0.748</td><td>1.897</td><td>0.992</td><td>0.977</td></tr><tr><td>5</td><td>cafe, living, washing, bus, meeting</td><td>0.027</td><td>0.010</td><td>0.670</td><td>2.077</td><td>0.994</td><td>0.973</td></tr><tr><td>6</td><td>living, washing, field, station, car, meeting</td><td>0.020</td><td>0.008</td><td>0.489</td><td>1.327</td><td>0.996</td><td>0.984</td></tr></table>

Increasing k improves spectral coverage of the environment set. The reduction is largest over the first few prototypes and becomes smaller after approximately $k = 3 – 4$ , consistent with the knee observed in the coverage curve in the main paper. We therefore use the k = 3 set, consisting of cafe, washing, and car, as the compact set for the subsequent simulator sweep.

## C. Full Simulator Parameter Sweep

For each language pair, the structured sweep contains nine simulator configurations evaluated over three PSD templates: cafe, washing, and car. This gives 27 experiments per language pair.

The full English-to-Hindi run-level results are reported below. To maintain readable text, the results are divided into clean/real-noise and simulator-noise tables rather than compressed into a single extremely wide table.

TABLE XIII  
FULL ENGLISH-TO-HINDI SWEEP: CLEAN AND REAL-NOISE EVALUATION.
<table><tr><td>Env</td><td>Exp</td><td>Cfg</td><td>Clean BLEU</td><td>Clean chrF</td><td>Clean WER</td><td>Real BLEU</td><td>Real chrF</td><td>Real WER</td></tr><tr><td>cafe</td><td>exp01</td><td>cfg01</td><td>48.668</td><td>67.761</td><td>0.425</td><td>47.421</td><td>66.771</td><td>0.444</td></tr><tr><td>cafe</td><td>exp02</td><td>cfg02</td><td>48.762</td><td>68.015</td><td>0.424</td><td>47.736</td><td>66.782</td><td>0.438</td></tr><tr><td>cafe</td><td>exp03</td><td>cfg03</td><td>48.959</td><td>67.903</td><td>0.423</td><td>47.422</td><td>66.582</td><td>0.438</td></tr><tr><td>cafe</td><td>exp04</td><td>cfg04</td><td>47.807</td><td>68.204</td><td>0.447</td><td>46.102</td><td>66.876</td><td>0.470</td></tr><tr><td>cafe</td><td>exp05</td><td>cfg05</td><td>47.869</td><td>67.975</td><td>0.443</td><td>46.771</td><td>66.912</td><td>0.457</td></tr><tr><td>cafe</td><td>exp06</td><td>cfg06</td><td>48.696</td><td>67.701</td><td>0.429</td><td>47.923</td><td>66.819</td><td>0.436</td></tr><tr><td>cafe</td><td>exp07</td><td>cfg07</td><td>48.491</td><td>67.617</td><td>0.435</td><td>47.037</td><td>66.399</td><td>0.456</td></tr><tr><td>cafe</td><td>exp08</td><td>cfg08</td><td>49.695</td><td>68.499</td><td>0.412</td><td>47.612</td><td>67.057</td><td>0.440</td></tr><tr><td>cafe</td><td>exp09</td><td>cfg09</td><td>47.261</td><td>67.436</td><td>0.449</td><td>46.143</td><td>66.484</td><td>0.463</td></tr><tr><td>washing</td><td>exp10</td><td>cfg01</td><td>48.315</td><td>68.126</td><td>0.430</td><td>46.949</td><td>67.160</td><td>0.452</td></tr><tr><td>washing</td><td>exp11</td><td>cfg02</td><td>49.264</td><td>68.164</td><td>0.424</td><td>47.855</td><td>66.812</td><td>0.446</td></tr><tr><td>washing</td><td>exp12</td><td>cfg03</td><td>49.125</td><td>68.263</td><td>0.420</td><td>47.745</td><td>66.681</td><td>0.432</td></tr><tr><td>washing</td><td>exp13</td><td>cfg04</td><td>48.520</td><td>67.527</td><td>0.433</td><td>47.059</td><td>66.057</td><td>0.443</td></tr><tr><td>washing</td><td>exp14</td><td>cfg05</td><td>48.445</td><td>68.079</td><td>0.428</td><td>47.361</td><td>67.011</td><td>0.440</td></tr><tr><td>washing</td><td>exp15</td><td>cfg06</td><td>48.642</td><td>68.168</td><td>0.429</td><td>47.983</td><td>67.127</td><td>0.436</td></tr><tr><td>washing</td><td>exp16</td><td>cfg07</td><td>47.692</td><td>67.853</td><td>0.446</td><td>46.554</td><td>66.637</td><td>0.461</td></tr><tr><td>washing</td><td>exp17</td><td>cfg08</td><td>48.764</td><td>67.964</td><td>0.428</td><td>46.976</td><td>66.352</td><td>0.454</td></tr><tr><td>washing</td><td>exp18</td><td>cfg09</td><td>48.325</td><td>67.860</td><td>0.429</td><td>47.144</td><td>66.718</td><td>0.445</td></tr><tr><td>car</td><td>exp19</td><td>cfg01</td><td>48.480</td><td>68.323</td><td>0.426</td><td>46.801</td><td>66.980</td><td>0.454</td></tr><tr><td>car</td><td>exp20</td><td>cfg02</td><td>48.777</td><td>67.943</td><td>0.427</td><td>47.328</td><td>66.603</td><td>0.447</td></tr><tr><td>car</td><td>exp21</td><td>cfg03</td><td>49.161</td><td>67.941</td><td>0.422</td><td>47.349</td><td>66.567</td><td>0.441</td></tr><tr><td>car</td><td>exp22</td><td>cfg04</td><td>48.911</td><td>68.028</td><td>0.424</td><td>47.445</td><td>66.541</td><td>0.442</td></tr><tr><td>car</td><td>exp23</td><td>cfg05</td><td>48.638</td><td>67.574</td><td>0.427</td><td>47.131</td><td>66.293</td><td>0.450</td></tr><tr><td>car</td><td>exp24</td><td>cfg06</td><td>48.852</td><td>68.243</td><td>0.426</td><td>47.752</td><td>66.862</td><td>0.438</td></tr><tr><td>car</td><td>exp25</td><td>cfg07</td><td>48.664</td><td>68.078</td><td>0.432</td><td>47.051</td><td>66.564</td><td>0.450</td></tr><tr><td>car</td><td>exp26</td><td>cfg08</td><td>48.169</td><td>67.857</td><td>0.438</td><td>46.718</td><td>66.479</td><td>0.456</td></tr><tr><td>car</td><td>exp27</td><td>cfg09</td><td>48.223</td><td>68.011</td><td>0.436</td><td>46.984</td><td>66.863</td><td>0.455</td></tr></table>

FULL ENGLISH-TO-HINDI SWEEP: SIMULATOR-NOISE EVALUATION.  
TABLE XIV
<table><tr><td>Env</td><td>Exp</td><td>Cfg</td><td>Sim BLEU</td><td>Sim chrF</td><td>Sim WER</td></tr><tr><td>cafe</td><td>exp01</td><td>cfg01</td><td>47.707</td><td>66.971</td><td>0.441</td></tr><tr><td>cafe</td><td>exp02</td><td>cfg02</td><td>48.112</td><td>67.316</td><td>0.433</td></tr><tr><td>cafe</td><td>exp03</td><td>cfg03</td><td>47.487</td><td>66.710</td><td>0.440</td></tr><tr><td>cafe</td><td>exp04</td><td>cfg04</td><td>46.550</td><td>67.108</td><td>0.464</td></tr><tr><td>cafe</td><td>exp05</td><td>cfg05</td><td>46.913</td><td>67.113</td><td>0.455</td></tr><tr><td>cafe</td><td>exp06</td><td>cfg06</td><td>47.955</td><td>66.902</td><td>0.435</td></tr><tr><td>cafe</td><td>exp07</td><td>cfg07</td><td>46.512</td><td>66.485</td><td>0.466</td></tr><tr><td>cafe</td><td>exp08</td><td>cfg08</td><td>47.726</td><td>67.340</td><td>0.437</td></tr><tr><td>cafe</td><td>exp09</td><td>cfg09</td><td>46.450</td><td>66.725</td><td>0.458</td></tr><tr><td>washing</td><td>exp10</td><td>cfg01</td><td>47.294</td><td>67.329</td><td>0.445</td></tr><tr><td>washing</td><td>exp11</td><td>cfg02</td><td>47.914</td><td>66.940</td><td>0.440</td></tr><tr><td>washing</td><td>exp12</td><td>cfg03</td><td>48.077</td><td>67.086</td><td>0.429</td></tr><tr><td>washing</td><td>exp13</td><td>cfg04</td><td>47.414</td><td>66.524</td><td>0.438</td></tr><tr><td>washing</td><td>exp14</td><td>cfg05</td><td>47.256</td><td>67.191</td><td>0.442</td></tr><tr><td>washing</td><td>exp15</td><td>cfg06</td><td>48.097</td><td>67.359</td><td>0.434</td></tr><tr><td>washing</td><td>exp16</td><td>cfg07</td><td>46.540</td><td>66.982</td><td>0.461</td></tr><tr><td>washing</td><td>exp17</td><td>cfg08</td><td>47.751</td><td>66.959</td><td>0.437</td></tr><tr><td>washing</td><td>exp18</td><td>cfg09</td><td>47.475</td><td>67.104</td><td>0.443</td></tr><tr><td>car</td><td>exp19</td><td>cfg01</td><td>47.416</td><td>67.355</td><td>0.441</td></tr><tr><td>car</td><td>exp20</td><td>cfg02</td><td>46.788</td><td>66.647</td><td>0.454</td></tr><tr><td>car</td><td>exp21</td><td>cfg03</td><td>47.350</td><td>66.523</td><td>0.447</td></tr><tr><td>car</td><td>exp22</td><td>cfg04</td><td>47.774</td><td>66.792</td><td>0.435</td></tr><tr><td>car</td><td>exp23</td><td>cfg05</td><td>47.520</td><td>66.540</td><td>0.444</td></tr><tr><td>car</td><td>exp24</td><td>cfg06</td><td>48.148</td><td>67.303</td><td>0.430</td></tr><tr><td>car</td><td>exp25</td><td>cfg07</td><td>47.563</td><td>66.882</td><td>0.441</td></tr><tr><td>car</td><td>exp26</td><td>cfg08</td><td>46.970</td><td>66.703</td><td>0.454</td></tr><tr><td>car</td><td>exp27</td><td>cfg09</td><td>47.647</td><td>67.278</td><td>0.445</td></tr></table>

Each experiment combines one of the nine configurations with one of the three PSD templates. These runlevel results are aggregated by configuration in the main paper to compute clean, real-noise, simulator-noise, and robustness-oriented WER statistics.

## APPENDIX C DATA AND ENVIRONMENT DETAILS

## A. DEMAND Environment Categories

The DEMAND dataset contains recordings from 18 real-world acoustic environments spanning indoor, outdoor, public, and transportation conditions. The environment categories considered in our data preparation are:

• Indoor: kitchen, living room, washing area, hallway, meeting room, office, cafeteria, and restaurant.

• Outdoor and public spaces: open field, park, river, station, street cafe, and public square.

• Transportation and traffic: traffic, bus, car, and metro.

The main experiments use a fixed subset of 12 environments from this catalog for the real-noise comparison and PSD-based environment reduction. The prototype-selection procedure is therefore performed with N = 12, as stated in the main paper. The subsequent structured hyperparameter sweep uses the three selected PSD templates: cafe, washing, and car.

## B. Spectral Examples from DEMAND

Figure 5 provides qualitative examples of acoustic structure in two DEMAND environments using mean Welch PSD curves and mean STFT spectrograms. The PSD summarizes the long-term frequency-wise energy distribution, while the spectrogram shows how spectral energy evolves over time. The examples illustrate that different environments exhibit distinct spectral profiles and temporal structures, motivating the use of environment-specific PSD template in NOPE-HYPE.

![](images/b6b9fb2c83ed0abfa29321b55bfdea3311b63753e1aaf378216a70763a1fa57c.jpg)  
Fig. 5. Qualitative examples of acoustic structure for two DEMAND environments, shown using mean Welch PSD curves and mean STFT spectrograms.

## C. Sweep Parameter Definitions

The simulator configurations are defined using the following parameters:

$\mu _ { \mathrm { l o g } }$ <sub>RMS</sub>: mean log-RMS level controlling the characteristic background noise energy.

$\sigma _ { \mathrm { l o g } }$ <sub>RMS</sub>: variability of the log-RMS background energy.

$r _ { e } :$ rate parameter controlling slow temporal variation of the ambience envelope.

• λ: mean rate of transient acoustic events.

$\sigma _ { \lambda } \colon$ variability of the transient-event rate.

• µ<sub>SNR</sub>: mean SNR used when mixing clean speech with environmental noise.

$\sigma _ { \mathrm { S N R } } \colon$ variability of the sampled SNR across utterances.