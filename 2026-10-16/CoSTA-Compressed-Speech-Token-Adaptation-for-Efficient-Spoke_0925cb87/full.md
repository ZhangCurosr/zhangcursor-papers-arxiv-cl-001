# CoSTA: Compressed Speech Token Adaptation for Efficient Spoken Dialogue Models

Yingzhuo Deng<sup>1,2</sup>, Chong Li<sup>1,2</sup>, Bingxin Liu<sup>1,2</sup>, Yi Shu<sup>2</sup>, Tianyu Peng<sup>1,2</sup>, Can Xie<sup>1,2</sup>, Siyao Song<sup>1,2</sup>, Shuo Ren<sup>1</sup>, Jiajun Zhang<sup>1,2,3</sup>\*

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences

<sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences <sup>3</sup>Wuhan AI Research

dengyingzhuo2024@ia.ac.cn; jjzhang@nlpr.ia.ac.cn

## Abstract

Spoken Dialogue Models (SDMs) typically share a speech-token space for generation and synthesis, leading to long autoregressive sequences and costly inference. Lower-rate speech tokenizers shorten sequences but may introduce compression–quality trade-offs and costly replacement without breaking this coupling. To address this limitation, we propose CoSTA, a compatibility-preserving post-hoc framework that decouples generation and synthesis token spaces while retaining the original speech tokenizer and decoder. CoSTA extends the generation token space with compressed representations via mean-initialized vocabulary extension, pronunciation adaptation, and dialogue adaptation, followed by deterministic expansion back to the original speech-token space before synthesis. On bilingual speech-dialogue benchmarks, CoSTA achieves approximately 1.7× expanded speech-token generation speed, lower full-response latency, and higher throughput while preserving response quality. Analyses show more sustained interaction under limited budgets and a practical trade-off among compression, decoding efficiency, and quality. Further experiments demonstrate crossarchitecture transfer and complementarity to low-rate tokenization.<sup>1</sup>

## 1 Introduction

Spoken dialogue models (SDMs) enable end-toend speech interaction by modeling speech input and generating text tokens together with speech tokens, which are then synthesized into waveforms by a speech decoder (Défossez et al., 2024; Zeng et al., 2024; Xu et al., 2025a; KimiTeam et al., 2025; Wu et al., 2025). Existing SDMs typically use the same token space for autoregressive generation and speech synthesis, tightly coupling generation with the representation required by the speech decoder. These speech-token sequences are often overly long, increasing autoregressive generation cost, complicating text-speech alignment, and making long-context speech modeling more difficult (Shen et al., 2024; Li et al., 2024). In practical interaction, this inefficiency slows responses and consumes generation budgets faster, limiting sustained interaction and spoken-response duration.

![](images/810429db350aa184414eb38104d4012deeba611f6fbaaa93fff71657b1f5f948.jpg)  
Figure 1: Overview of CoSTA. Existing SDMs couple generation and synthesis token spaces. CoSTA extends the generation token space with compressed representations and deterministically expands them back to the original space before synthesis.

One solution is to design speech tokenizers with lower token rates (Défossez et al., 2024; Zeng et al., 2025; Guo et al., 2024; Wang et al., 2025c; Guo et al., 2026). However, aggressive rate reduction may introduce a compression–quality trade-off, and replacing the tokenizer of an existing SDM may require retraining or adapting the tokenizer, paired speech decoder, and SDM together. More fundamentally, this route still couples generation and synthesis to the same representation. We therefore ask a different question: Can we decouple the token space used for autoregressive generation from that required for speech synthesis, while preserving the original speech tokenizer and decoder?

Speech-token sequences produced by existing tokenizers exhibit substantial redundancy and can be losslessly compressed using BPE (Gage, 1994; Sennrich et al., 2016; Radford et al., 2019). Across diverse tokenizer, data, and language settings, we uncover an empirical scaling regularity: BPE compression shortens speech-token sequences with increasing vocabulary size, following an approximate log-linear trend on held-out speech. This persistent compressibility suggests a route to more efficient generation without altering synthesis: using a compact autoregressive representation while deterministically recovering the original speech-token sequence required by the decoder.

This insight motivates generation–synthesis token-space decoupling and CoSTA, short for Compressed Speech Token Adaptation, a compatibility-preserving post-hoc token-space adaptation framework for existing large-scale SDMs. As illustrated in Figure 1, CoSTA extends the autoregressive generation token space with compressed representations while preserving the original synthesis interface. CoSTA adapts the SDM to generate these compressed representations through vocabulary extension with mean initialization, pronunciation adaptation, and dialogue adaptation. At inference time, generated compressed units are deterministically expanded back to the original speech-token representations before synthesis, reducing autoregressive speech-token steps while leaving the original speech tokenizer and decoder unchanged.

We validate CoSTA on two representative SDM architectures. In experiments with Step-Audio 2 mini (Wu et al., 2025), CoSTA achieves 1.736× and 1.659× relative expanded speech-token generation speed in Chinese and English, respectively, while preserving response quality according to both automatic and human evaluations. These gains further translate into lower full-response latency, higher throughput, and more sustained interaction under limited budgets. Vocabulary scaling further reveals a practical trade-off among compression, decoding efficiency, and response quality. Experiments on a different SDM architecture further demonstrate cross-architecture transfer and complementarity to low-rate speech tokenization.

Our contributions are summarized as follows:

• We uncover an approximate log-linear scaling regularity of speech-token compression over 10K–320K added units across eight tokenizer configurations, data sources, and languages.

• We formulate generation–synthesis tokenspace decoupling and introduce CoSTA, a compatibility-preserving post-hoc adaptation framework that enables existing SDMs to generate compact, losslessly expandable speechtoken sequences while retaining their original speech tokenizer and decoder.

![](images/a7194bda175546e6f53cf794e8d6d21bf4762208eab48c096b004bf4993c3dfc.jpg)  
Figure 2: Speech-token BPE scaling across eight tokenizer configurations. Compression rate decreases approximately linearly with $\log _ { 2 }$ vocabulary size.

• We demonstrate that CoSTA improves modelside and system efficiency while preserving response quality, with additional evidence of cross-architecture transfer and complementarity to low-rate speech tokenization.

• We further characterize compressed-token generation: CoSTA enables more sustained interaction under limited budgets, while vocabulary scaling reveals a practical trade-off among compression, efficiency, and quality.

## 2 From Speech-Token BPE Scaling to Token-Space Decoupling

Lossless token compression. Discrete speechtoken sequences contain repeated local patterns, making them naturally amenable for lossless compression. Here, lossless refers to reversible token segmentation: the compressed sequence can be deterministically expanded back to the original sequence. We therefore apply BPE to the original speech-token sequences. Let S denote the original speech-token vocabulary and s a speech-token sequence. BPE constructs an extended vocabulary ${ \mathcal { Z } } \supseteq S$ by merging frequent adjacent token pairs. Its encoder $\boldsymbol { B }$ produces the compressed sequence $\mathbf { z } = B ( \mathbf { s } )$ , and the deterministic expansion function π recovers the original sequence, i.e., $\pi ( \mathbf { z } ) = \mathbf { s }$ We define the compression rate as

$$
\rho ( \mathbf { z } ) = \frac { | \mathbf { z } | } { | \pi ( \mathbf { z } ) | } .\tag{1}
$$

Lower $\rho$ indicates fewer autoregressive steps for generating the same sequence consumed by the decoder. The number of added BPE units is denoted as $M = | \mathcal { Z } \backslash \mathcal { S } |$

![](images/b63d0f7a5bf9bcea12c3d0c5cdebfbf17e089fb0eca694c72fb139a740a7689f.jpg)  
Figure 3: CoSTA combines vocabulary extension with mean-based initialization, progressive pronunciation adaptation, and dialogue adaptation.

Empirical scaling regularity. We scale the added BPE vocabulary M from 10K to 320K in powers of two across eight representative speech tokenizer configurations on Chinese dialogue speech. For each setting, the BPE vocabulary is learned on the training split and fixed for evaluation on heldout speech. Figure 2 reveals a consistent scaling regularity: $\rho$ decreases approximately linearly with log M. We describe this empirical trend as

$$
\rho _ { c } ( M ) \approx a _ { c } { - b _ { c } } \log _ { 2 } ( M / 1 0 \mathrm { K } ) , \qquad b _ { c } > 0 ,\tag{2}
$$

where c denotes an evaluation setting. The loglinear model achieves $R ^ { 2 } ~ \ge ~ 0 . 9 8$ across eight tokenizer configurations and remains consistent across data sources and languages $( R ^ { 2 } \geq 0 . 9 9 4 ;$ Appendix C.4). Detailed settings and analyses are provided in Appendix D.

From scaling regularity to generation–synthesis token-space decoupling. The original speechtoken space produces long sequences for autoregressive generation. The empirical scaling regularity reveals persistent compressibility: larger BPE vocabularies yield shorter speech-token sequences, motivating their use for autoregressive generation. However, the speech decoder still requires the original token space. This motivates generation–synthesis token-space decoupling: the SDM generates compact BPE sequences in an extended token space, which are deterministically expanded back to the original space for synthesis. We instantiate this paradigm through the post-hoc adaptation framework CoSTA in Section 3.

## 3 Method

Section 2 motivates generation–synthesis tokenspace decoupling. Realizing this decoupling in an existing SDM, however, requires adapting the model while preserving compatibility with its original tokenizer–decoder pipeline. We therefore propose CoSTA, a compatibility-preserving post-hoc token-space adaptation framework. CoSTA adapts the SDM to generate compressed sequences in the extended token space Z, while deterministic expansion π(z) = s restores the original decoder-facing representation in S before waveform synthesis.

As shown in Figure 3, CoSTA consists of three steps: (1) vocabulary extension with mean-based initialization, (2) pronunciation adaptation on compressed TTS data, and (3) dialogue adaptation on compressed speech-to-speech dialogue data. At inference time, only a lightweight deterministic BPE expansion is inserted before the unchanged speech decoder. We detail each step below.

## 3.1 Vocabulary Extension with Mean-based Initialization

To instantiate the extended generation token space Z in an existing SDM, let V denote its original model vocabulary, with S ⊆ V denoting the original speech-token vocabulary. We extend V with

the added BPE units:

$$
\mathcal { V } ^ { \prime } = \mathcal { V } \cup ( \mathcal { Z } \setminus \mathcal { S } ) .\tag{3}
$$

For each added BPE unit $z \in { \mathcal { Z } } \backslash S _ { \mathrm { : } }$ , its deterministic expansion $\pi ( z ) = ( s _ { 1 } , \ldots , s _ { k } )$ is known. We therefore initialize its embedding $\mathbf { E } _ { z } ^ { \prime }$ and LM-head row $\mathbf { W } _ { z } ^ { \prime }$ by averaging the corresponding component rows:

$$
\begin{array} { r } { \mathbf { E } _ { z } ^ { \prime } = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \mathbf { E } _ { s _ { i } } , ~ \mathbf { W } _ { z } ^ { \prime } = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \mathbf { W } _ { s _ { i } } . } \end{array}\tag{4}
$$

Original vocabulary parameters remain unchanged. This mean-based initialization provides a more stable starting point than random initialization.

## 3.2 Pronunciation Adaptation

After vocabulary extension, the SDM has an extended generation token space but has not yet learned to use the added BPE units reliably. We therefore perform pronunciation adaptation on compressed TTS data. Following interleavedgeneration SDMs such as Step-Audio 2 mini (Wu et al., 2025), each target sequence contains both text and speech tokens. We apply BPE compression only to speech tokens and leave text tokens unchanged.

Each example consists of a text prompt p and a target response sequence y, where y contains interleaved text tokens and original speech tokens. Let ${ \widetilde { \mathbf { y } } } = C ( \mathbf { y } )$ denote the sequence obtained by applying BPE encoder $\boldsymbol { B }$ only to the speech tokens of y. We train the model to predict the compressed target sequence autoregressively:

$$
\mathcal { L } _ { \mathrm { t t s } } ( \theta ) = - \sum _ { ( p , \widetilde { \mathbf { y } } ) \in \mathcal { D } _ { \mathrm { t t s } } } \sum _ { t = 1 } ^ { | \widetilde { \mathbf { y } } | } \log P \left( \widetilde { y } _ { t } \mid p , \widetilde { \mathbf { y } } _ { < t } \right) .\tag{5}
$$

To stabilize adaptation to the extended generation token space, we adopt a progressive strategy inspired by prior work (Li et al., 2025a, 2026). In the first stage, we use 50% of the data to fine-tune only the embedding and LM-head layers while freezing the remaining model parameters. This calibrates the newly added vocabulary parameters before broader model adaptation. In the second stage, we use the remaining 50% of the data and additionally unfreeze the intermediate model layers to complete pronunciation adaptation.

## 3.3 Dialogue Adaptation

Pronunciation adaptation teaches the model to use the extended generation token space under text prompts, but does not yet adapt this ability to speech-to-speech dialogue. We therefore perform dialogue adaptation on compressed speechto-speech dialogue data, updating only the LLM parameters while keeping the speech encoder, adaptor, and speech decoder frozen. Each dialogue example contains a user’s speech query q and a target assistant response r. As in pronunciation adaptation, only the speech tokens of the interleaved target response are compressed. Let $\widetilde { \mathbf { r } } = C ( \mathbf { r } )$ The objective of dialogue adaptation is:

![](images/96c1dfe48cd8e9efac8034c1a6418f1cc3e6d8253095582d5ff2576635a1ef2a.jpg)  
Figure 4: Inference with deterministic expansion from the extended generation token space to the original space before unchanged decoding.

$$
\mathcal { L } _ { \mathrm { d l g } } ( \boldsymbol { \theta } ) = - \sum _ { ( \boldsymbol { q } , \widetilde { \mathbf { r } } ) \in \mathcal { D } _ { \mathrm { d l g } } } \sum _ { t = 1 } ^ { | \widetilde { \mathbf { r } } | } \log P \left( \widetilde { r } _ { t } \mid \boldsymbol { q } , \widetilde { \mathbf { r } } _ { < t } \right) .\tag{6}
$$

After this stage, the SDM can use the extended generation token space for speech-to-speech dialogue, with the generated speech-token stream mapped back to $\boldsymbol { \mathcal { S } }$ by π before waveform synthesis.

## 3.4 Inference with Deterministic Expansion

As shown in Figure 4, the adapted SDM generates interleaved text and speech tokens from the extended model vocabulary $\mathcal { V } ^ { \prime }$ . The speech-token stream lies in $\mathcal { Z }$ and may contain both original tokens in S and added BPE units in ${ \mathcal { Z } } \backslash { \mathcal { S } }$ . We apply the deterministic expansion π only to this speechtoken stream: original tokens are left unchanged, while each added BPE unit is expanded into its corresponding sequence in S. The recovered original speech-token sequence is then fed directly into the unchanged speech decoder for waveform synthesis. This compatibility-preserving mapping allows the SDM to take fewer autoregressive steps while keeping the synthesis interface unchanged.

<table><tr><td rowspan="2">Model</td><td colspan="2">AlpacaEval-zh</td><td colspan="2">APE-zh</td><td colspan="2">Claude-zh</td><td colspan="2">LCSTS-zh</td><td colspan="2">MLC-zh</td><td colspan="2">Wildchat-zh</td><td colspan="3"> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑</td><td> $\rho _ { g } \downarrow$ </td><td></td><td>Speed↑ Relative↑</td></tr><tr><td>Step-Audio 2 mini 100.00%</td><td></td><td>55.51</td><td>100.00%</td><td>60.88</td><td>100.00%</td><td>56.39</td><td>100.00%</td><td>57.05</td><td>100.00%</td><td>65.22</td><td>100.00%</td><td>57.40</td><td>100.00%</td><td>58.74</td><td>100.00%</td></tr><tr><td>Vanilla-FT-ZH</td><td>100.00%</td><td>57.84</td><td>100.00%</td><td>60.40</td><td>100.00%</td><td>54.99</td><td>100.00%</td><td>55.54</td><td>100.00%</td><td>63.47</td><td>100.00%</td><td>59.10</td><td>100.00%</td><td>58.56</td><td>99.69%</td></tr><tr><td>Vanilla-FT-MIX</td><td>100.00%</td><td>59.01</td><td>100.00%</td><td>59.65</td><td>100.00%</td><td>56.93</td><td>100.00%</td><td>56.03</td><td>100.00%</td><td>61.57</td><td>100.00%</td><td>59.08</td><td>100.00%</td><td>58.71</td><td>99.95%</td></tr><tr><td>CoSTA-ZH</td><td>67.50%</td><td>103.03</td><td>64.63%</td><td>101.68</td><td>68.52%</td><td>101.20</td><td>69.40%</td><td>94.46</td><td>62.70%</td><td>110.31</td><td>68.13%</td><td>101.29</td><td>66.81%</td><td>101.99</td><td>173.63%</td></tr><tr><td>CoSTA-MIX</td><td>69.12%</td><td>98.44</td><td>64.68%</td><td>102.34</td><td>70.58%</td><td>92.54</td><td>71.06%</td><td>91.37</td><td>63.03%</td><td>106.32</td><td>70.21%</td><td>93.22</td><td>68.11%</td><td>97.37</td><td>165.76%</td></tr><tr><td rowspan="2">Model</td><td colspan="2">AlpacaEval</td><td colspan="2">Gsm8kEval</td><td colspan="2">CommonEval</td><td colspan="2">Summary</td><td colspan="2">MLC</td><td colspan="2">WildchatEval</td><td colspan="3">Avg</td></tr><tr><td> $\rho _ { g } \downarrow$ </td><td>Speed↑</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑|</td><td> $\rho _ { g } \downarrow$ </td><td></td><td>Speed↑ Relative↑</td></tr><tr><td>Step-Audio 2 mini 100.00%</td><td></td><td>57.93</td><td>100.00%</td><td>60.03</td><td>100.00%</td><td>60.62</td><td>100.00%</td><td>58.74</td><td>100.00%</td><td>61.80</td><td>100.00%</td><td>59.00</td><td>100.00%</td><td>59.69</td><td>100.00%</td></tr><tr><td>Vanilla-FT-EN</td><td>100.00%</td><td>56.73</td><td>100.00%</td><td>58.14</td><td>100.00%</td><td>56.96</td><td>100.00%</td><td>56.16</td><td>100.00%</td><td>61.36</td><td>100.00%</td><td>56.67</td><td>100.00%</td><td>57.67</td><td>96.61%</td></tr><tr><td>Vanilla-FT-MIX</td><td>100.00%</td><td>57.43</td><td>100.00%</td><td>59.27</td><td>100.00%</td><td>57.29</td><td>100.00%</td><td>57.66</td><td>100.00%</td><td>61.90</td><td>100.00%</td><td>56.43</td><td>100.00%</td><td>58.33</td><td>97.72%</td></tr><tr><td>CoSTA-EN</td><td>73.97%</td><td>87.60</td><td>65.60%</td><td>102.04</td><td>71.60%</td><td>98.41</td><td>72.09%</td><td>95.21</td><td>64.71%</td><td>115.58</td><td>73.06%</td><td>95.46</td><td>70.17%</td><td>99.05</td><td>165.94%</td></tr><tr><td>CoSTA-MIX</td><td>75.45%</td><td>81.17</td><td>65.55%</td><td>97.92</td><td>72.34%</td><td>88.91</td><td>73.94%</td><td>83.29</td><td>64.30%</td><td>123.95</td><td>74.56%</td><td>87.80</td><td>71.02%</td><td>93.84</td><td>157.21%</td></tr></table>

Table 1: Model-side efficiency on Chinese and English speech-to-speech benchmarks. $\rho _ { g }$ denotes generated-token compression, Speed denotes expanded speech-token generation speed, and Relative normalizes Speed to Step-Audio 2 mini within each language setting.
<table><tr><td>Concurrency</td><td>Model</td><td>Full-Response Latency (s)↓</td><td>Throughput (req/s)↑</td><td>Peak Memory (GiB)</td></tr><tr><td>1</td><td>Step-Audio 2 mini / Vanilla-FT-ZH / CoSTA-ZH</td><td>10.73 / 9.72 / 7.14</td><td>0.103 / 0.114 / 0.160</td><td>78.12 / 77.64 / 78.56</td></tr><tr><td>4</td><td>Step-Audio 2 mini / Vanilla-FT-ZH / CoSTA-ZH</td><td>13.63 / 11.38 / 9.16</td><td>0.316 / 0.369 / 0.479</td><td>78.34 / 78.13 / 79.02</td></tr><tr><td>8</td><td>Step-Audio 2 mini / Vanilla-FT-ZH / CoSTA-ZH</td><td>15.56 / 13.71 / 9.49</td><td>0.515 / 0.601 / 0.820</td><td>78.90 / 79.17 / 79.14</td></tr><tr><td>1</td><td>Step-Audio 2 mini / Vanilla-FT-EN / CoSTA-EN</td><td>12.50 / 16.16 / 9.25</td><td>0.088 / 0.068 / 0.122</td><td>78.98 / 77.16 / 78.69</td></tr><tr><td>4</td><td>Step-Audio 2 mini / Vanilla-FT-EN / CoSTA-EN</td><td>15.52 / 20.17 / 11.59</td><td>0.276 / 0.212 / 0.372</td><td>79.24 / 78.33 / 79.12</td></tr><tr><td>8</td><td>Step-Audio 2 mini / Vanilla-FT-EN / CoSTA-EN</td><td>18.62 / 25.04 / 14.44</td><td>0.442 / 0.330 / 0.556</td><td>78.52 / 77.73 / 79.04</td></tr></table>

Table 2: System-level efficiency on AlpacaEval-zh (upper) and AlpacaEval (lower) at concurrency 1, 4, and 8 on a single A100 80GB GPU. Values are ordered as Step-Audio 2 mini, Vanilla-FT, and CoSTA.

## 4 Experiments

## 4.1 Experimental Setup

Implementation. We use the 8B Step-Audio 2 mini (Wu et al., 2025) as the SDM for its strong open-source support, simple architecture, and competitive dialogue performance; it uses S3Tokenizer V2 at 25 Hz. For pronunciation adaptation, we sample 300 hours each of Chinese and English speech from Emilia (He et al., 2024) and convert them to a unified timbre with Seed-VC (Liu, 2024). For dialogue adaptation, we follow OpenS2S (Wang et al., 2025a) to construct approximately 50K speechdialogue examples per language, with assistant speech synthesized by CosyVoice 2 (Du et al., 2024b) using a consistent voice. We learn the BPE vocabulary from dialogue speech-token sequences and use 80K added units, supported by the vocabulary-size trade-off analysis in Section 4.4. Vocabulary extension adds only embedding and LM-head rows, about 6% of the original model parameters. We train CoSTA-ZH, CoSTA-EN, and CoSTA-MIX for Chinese, English, and bilingual adaptation; further training details and data statistics are provided in Appendices A and B.1.

Baselines and benchmarks. We compare against the original Step-Audio 2 mini and Vanilla-

FT-ZH/EN/MIX, which use the same adaptation data and training configuration while retaining the original speech tokens. This controls for training-data scale and isolates the effect of BPEcompressed token adaptation. Qwen2.5-Omni (Xu et al., 2025a), Kimi-Audio (KimiTeam et al., 2025), and GLM-4-Voice (Zeng et al., 2024) serve as external quality references. We evaluate on 12 datasets from URO-Bench (Yan et al., 2025), covering aligned Chinese and English settings and spanning summarization, mathematical and common-sense reasoning, open-ended dialogue, and real-world conversation, providing broad coverage of realistic spoken-dialogue scenarios.

Efficiency metrics. We report three model-side efficiency metrics. First, the generated-token compression rate $\rho _ { g } = \rho ( \widehat { \mathbf { z } } )$ is computed using Eq. 1, where $\widehat { \mathbf { z } }$ denotes the model-generated speech-token sequence; a lower $\rho _ { g }$ indicates fewer autoregressive speech-token steps. Second, the expanded speech-token generation speed is

$$
\mathrm { S p e e d } = \frac { | \pi ( \widehat { \mathbf { z } } ) | } { T _ { \mathrm { d e c o d e } } } ,\tag{7}
$$

where $T _ { \mathrm { d e c o d e } }$ is LLM decoding time, measuring the effective number of original speech tokens generated per second after expansion. Third, relative speed, denoted as Relative in the tables, normalizes the average Speed to that of Step-Audio 2 mini under the same language setting.

<table><tr><td rowspan="2">Model</td><td rowspan="2">AlpacaEval-zh</td><td rowspan="2">APE-zh</td><td rowspan="2">Claude-zh</td><td rowspan="2">LCSTS-zh MLC-zh Wildchat-zh</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Avg</td></tr><tr><td>|G-Eval↑</td><td>ASR-CER↓ UTMOS↑</td><td></td></tr><tr><td>Qwen2.5-Omni</td><td>74.10</td><td>62.11</td><td>78.86</td><td>69.75</td><td>82.53</td><td>78.35</td><td>74.28</td><td>2.93%</td><td>2.95</td></tr><tr><td>Kimi-Audio</td><td>77.05</td><td>39.12</td><td>80.24</td><td>76.13</td><td>80.92</td><td>78.08</td><td>71.93</td><td>4.88%</td><td>2.52</td></tr><tr><td>GLM-4-Voice</td><td>84.76</td><td>23.16</td><td>83.54</td><td>76.36</td><td>78.85</td><td>84.41</td><td>71.85</td><td>3.77%</td><td>2.94</td></tr><tr><td>Step-Audio 2 mini</td><td>80.18</td><td>21.93</td><td>72.40</td><td>79.16</td><td>78.16</td><td>83.59</td><td>69.24</td><td>7.49%</td><td>3.62</td></tr><tr><td>Vanilla-FT-ZH</td><td>79.77</td><td>21.40</td><td>80.35</td><td>78.82</td><td>77.47</td><td>81.47</td><td>69.88</td><td>4.96%</td><td>3.77</td></tr><tr><td>Vanilla-FT-MIX</td><td>81.27</td><td>20.53</td><td>79.28</td><td>79.10</td><td>79.31</td><td>82.10</td><td>70.26</td><td>7.02%</td><td>3.77</td></tr><tr><td>CoSTA-ZH</td><td>81.36</td><td>24.56</td><td>86.49</td><td>80.56</td><td>79.54</td><td>81.09</td><td>72.27</td><td></td><td></td></tr><tr><td>CoSTA-MIX</td><td>80.95</td><td>24.56</td><td>85.05</td><td>78.99</td><td>77.24</td><td>82.39</td><td>71.53</td><td>1.51% 1.65%</td><td>3.75 3.75</td></tr></table>

<table><tr><td rowspan="2">Model</td><td rowspan="2">AlpacaEval</td><td rowspan="2" colspan="3">Gsm8kEval CommonEval Summary</td><td rowspan="2">MLC</td><td rowspan="2">WildchatEval</td><td colspan="3">Avg</td></tr><tr><td>G-Eval↑ ASR-WER↓ UTMOS</td><td></td><td></td></tr><tr><td>Qwen2.5-Omni</td><td>76.78</td><td>54.30</td><td>72.00</td><td>80.06</td><td>77.40</td><td>76.71</td><td>72.88</td><td>1.73%</td><td>4.28</td></tr><tr><td>Kimi-Audio</td><td>62.01</td><td>0.57</td><td>50.77</td><td>82.15</td><td>40.87</td><td>66.40</td><td>50.46</td><td>34.92%</td><td>2.86</td></tr><tr><td>GLM-4-Voice</td><td>80.87</td><td>29.32</td><td>66.77</td><td>92.15</td><td>69.30</td><td>81.26</td><td>69.95</td><td>11.55%</td><td>4.16</td></tr><tr><td>Step-Audio 2 mini</td><td>76.31</td><td>15.92</td><td>52.53</td><td>88.14</td><td>72.69</td><td>76.75</td><td>63.73</td><td>6.49%</td><td>4.53</td></tr><tr><td>Vanilla-FT-EN</td><td>73.74</td><td>16.67</td><td>72.03</td><td>86.10</td><td>72.50</td><td>78.47</td><td>66.59</td><td>10.16%</td><td>4.46</td></tr><tr><td>Vanilla-FT-MIX</td><td>76.18</td><td>11.05</td><td>72.63</td><td>85.37</td><td>68.55</td><td>78.93</td><td>65.45</td><td>12.00%</td><td>4.49</td></tr><tr><td>CoSTA-EN</td><td>81.34</td><td>15.52</td><td>75.87</td><td>89.21</td><td>74.01</td><td>83.84</td><td>69.97</td><td>2.50%</td><td>4.54</td></tr><tr><td>CoSTA-MIX</td><td>83.32</td><td>17.53</td><td>73.47</td><td>90.00</td><td>71.19</td><td>82.52</td><td>69.67</td><td>2.64%</td><td>4.54</td></tr></table>

Table 3: Automatic quality evaluation on Chinese and English speech-to-speech benchmarks. Benchmark columns report G-Eval; Avg reports average G-Eval, ASR-CER/WER, and UTMOS. Bold and underline mark the best and second-best results among Step-Audio 2 mini variants; external models serve as quality references.

At the system level, we additionally report mean full-response latency, request throughput, and peak GPU memory on AlpacaEval-zh and AlpacaEval at concurrency 1, 4, and 8 using a single A100 80GB with vLLM. First-packet latency is reported separately in Appendix C.5.

Quality metrics. Following the URO-Bench evaluation protocol, we report G-Eval, UT-MOS (Saeki et al., 2022), and WER/CER for response-content quality, speech quality, and textspeech alignment, respectively. Evaluation details are provided in Appendix B.2. We further conduct a MOS-style listening study with six evaluators, yielding 1,080 response-level evaluations of speech fluency and semantic quality; details are provided in Appendix C.6.

## 4.2 Main Results

Model-side efficiency. Table 1 shows substantial efficiency gains in both languages under the vLLM streaming inference setting. CoSTA-ZH reaches $\rho _ { g } \ : = \ : 6 6 . 8 1$ % and 1.736× relative speed, while CoSTA-EN reaches 70.17% and 1.659×. CoSTA-MIX shows similar improvements, whereas vanilla fine-tuning retains $\rho _ { g } \ = \ 1 0 0 \%$ with nearly unchanged generation speed. These results show that compressed-token modeling reduces autoregressive speech-token steps and translates this reduction into higher generation efficiency.

![](images/879d53c7c6ff5c231919e3c1b01f81907c9983651c71510ad881f0c9a4d2d5ac.jpg)  
Figure 5: Paired human evaluation of CoSTA against the base model and Vanilla-FT. Bars show wins, ties, and losses from 180 paired MOS ratings per language and metric.

System efficiency. Table 2 shows that decoding gains translate into lower full-response latency and higher throughput. Relative to the base model, CoSTA reduces latency by 32.8–39.0% in Chinese and 22.4–26.0% in English across the tested concurrency levels; throughput increases by 51.6–59.2% and 25.8–38.6%, respectively. It also outperforms matched Vanilla-FT at every tested setting, with at most 0.68 GiB additional peak memory. These workload-level results show that CoSTA’s modelside efficiency gains translate into system-level improvements in latency and throughput with minimal memory overhead.

Automatic quality evaluation. Table 3 shows that CoSTA preserves or improves average output quality while substantially improving efficiency. Compared with matched Vanilla-FT, CoSTA improves G-Eval and substantially reduces ASR-CER/WER in both Chinese and English, while UTMOS remains comparable; CoSTA-MIX shows the same overall trend. These results indicate that compressed-token generation does not trade response quality for efficiency and can improve textspeech alignment, while preserving speech quality after deterministic expansion. Overall, CoSTA achieves a favorable efficiency–quality trade-off within the adapted SDM.

<table><tr><td>Variant</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td></tr><tr><td>Step-Audio 2 mini</td><td>100.00%</td><td>7.49%</td><td>3.62</td><td>69.24</td></tr><tr><td>Vanilla-FT-ZH</td><td>100.00%</td><td>4.96%</td><td>3.77</td><td>69.88</td></tr><tr><td>CoSTA-ZH</td><td>66.73%</td><td>1.51%</td><td>3.75</td><td>72.27</td></tr><tr><td>w/o Timbre Conversion</td><td>67.01%</td><td>1.52%</td><td>3.75</td><td>70.93</td></tr><tr><td>w/o Progressive Adaptation</td><td>67.75%</td><td>1.46%</td><td>3.75</td><td>72.19</td></tr><tr><td>w/o Pronunciation Adaptation</td><td>69.56%</td><td>1.74%</td><td>3.74</td><td>72.12</td></tr><tr><td>w/o Dialogue Adaptation</td><td>81.05%</td><td>2.68%</td><td>3.74</td><td>24.35</td></tr></table>

Table 4: Component ablation on Chinese benchmarks. $\rho _ { g } ,$ ASR-CER, UTMOS, and G-Eval measure generatedtoken compression, text-speech alignment, speech quality, and response-content quality, respectively.

Human evaluation. Figure 5 shows that CoSTA wins more often than it loses against both the base model and Vanilla-FT in Chinese and English. Winor-tie rates reach 86.1–96.1% for speech fluency and 83.3–87.7% for semantic quality, indicating that efficiency gains preserve perceived quality. MOS scores show the same trend; detailed results are provided in Appendix C.6.

## 4.3 Ablation Study

Table 4 shows that the adaptation components play complementary roles. Timbre conversion mainly benefits response quality, while progressive adaptation modestly improves compressed-token generation. Pronunciation adaptation further strengthens compression and text-speech alignment, supporting reliable generation in the extended token space. Dialogue adaptation has the largest effect: removing it causes G-Eval to drop sharply from 72.27 to 24.35, indicating that learning compressed pronunciation alone is insufficient to recover spoken-dialogue capability. Appendix C.8 further analyzes the associated compressed-token behavior and response failures. Figures 6 and 8 further show that mean initialization provides faster convergence and more stable gradients than the alternatives; detailed initialization strategies are described in Appendix C.2.

## 4.4 Analyses

Interaction under limited budgets. Table 5 groups every 10 queries from AlpacaEval-zh and AlpacaEval into a stress-test session, yielding 14 Chinese and 19 English sessions, respectively. CoSTA sustains 7.21/5.79 effective turns in Chinese/English, compared with 3.71/1.84 for Vanilla-FT, while also producing longer speech within these interactions. This suggests that compressedtoken generation uses the available autoregressive generation budget more efficiently, sustaining more turns and longer speech. This setting evaluates generation-format sustainability rather than conversational coherence.

![](images/65f1bdd4c8651777901dc3fc3df48040aae84250d21eb7078ee3a8f5c3c7bb37.jpg)

Figure 6: Training loss during the first stage of pronunciation adaptation. Mean initialization achieves faster early convergence than the alternatives.
<table><tr><td>Model</td><td colspan="4"># Turns↑ Gen Tokens Final Tokens Duration (s)</td><td> $\rho _ { g } \downarrow$ </td></tr><tr><td>Step-Audio 2 mini</td><td>2.14</td><td>2208.21</td><td>2208.21</td><td>88.33</td><td>100.00%</td></tr><tr><td>Vanilla-FT-ZH</td><td>3.71</td><td>2440.50</td><td>2440.50</td><td>97.62</td><td>100.00%</td></tr><tr><td>CoSTA-ZH</td><td>7.21</td><td>2500.79</td><td>3766.86</td><td>150.67</td><td>66.39%</td></tr><tr><td>Step-Audio 2 mini</td><td>2.58</td><td>2218.74</td><td>2218.74</td><td>88.75</td><td>100.00%</td></tr><tr><td>Vanilla-FT-EN</td><td>1.84</td><td>1894.79</td><td>1894.79</td><td>75.79</td><td>100.00%</td></tr><tr><td>CoSTA-EN</td><td>5.79</td><td>2608.53</td><td>3650.79</td><td>146.03</td><td>71.45%</td></tr></table>

Table 5: Interaction sustainability under a limited generation budget. Gen/Final Tokens denote speech tokens before/after deterministic expansion; Duration denotes generated speech duration.

Vocabulary-size trade-off. Figure 7 shows that increasing the added vocabulary from 10K to 80K substantially improves compression and decoding speed while maintaining response quality. Extending further to 160K reduces $\rho _ { g }$ from 67.50% to 63.62%, but yields only a 1.1% speed gain, while G-Eval decreases from 81.36 to 78.87. These diminishing full-model gains motivate 80K as a practical balance between compression, decoding efficiency, response-content quality, and speech quality in our setting. Complete vocabulary-size results are provided in Appendix C.9, with additional analysis of output-layer overhead in Appendix C.10.

Transfer across SDM architectures. Table 6 evaluates CoSTA on OpenS2S (Wang et al., 2025a), whose Think–Talk architecture, 12.5-Hz GLM-4- Voice tokenizer, and speech decoder differ from Step-Audio 2 mini. CoSTA reduces speech-token steps by 21.17% and 18.64% in Chinese and English, respectively, while improving responsecontent quality and text-speech alignment over matched Vanilla-FT and maintaining comparable UTMOS. These results show that post-hoc tokenspace adaptation transfers to a different SDM ar-

![](images/57bd0db5c35b9601ed1f63a4e4453665f4832ba4cd044e94a871e96a0b9fcb93.jpg)

![](images/cc06516984e5cd531f4a724172ce68a425c5504cff2db0f8431d824174b160c6.jpg)  
Figure 7: Vocabulary-size trade-off on AlpacaEval-zh: relative expanded speech-token generation speed with (a) UTMOS and (b) G-Eval. The 0K baseline denotes the original Step-Audio 2 mini; other points denote CoSTA-ZH.

<table><tr><td>Model</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER/WER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td>OpenS2S</td><td>100.00%</td><td>11.08%</td><td>2.41</td><td>65.26</td></tr><tr><td>Vanilla-FT-ZH</td><td>100.00%</td><td>11.36%</td><td>2.38</td><td>67.39</td></tr><tr><td>CoSTA-ZH</td><td>78.83%</td><td>8.50%</td><td>2.36</td><td>69.18</td></tr><tr><td>OpenS2S</td><td>100.00%</td><td>4.65%</td><td>3.81</td><td>73.97</td></tr><tr><td>Vanilla-FT-EN</td><td>100.00%</td><td>5.16%</td><td>3.79</td><td>75.48</td></tr><tr><td>CoSTA-EN</td><td>81.36%</td><td>4.66%</td><td>3.78</td><td>76.23</td></tr></table>

Table 6: Transfer of CoSTA to OpenS2S. Upper/lower blocks report Chinese/English results using ASR-CER/WER, respectively. Matched Vanilla-FT uses the same architecture and language-specific adaptation data. chitecture in both language settings. Moreover, the additional compression on top of a native 12.5-Hz tokenizer suggests that CoSTA is complementary to low-rate speech tokenization and can be combined with it for further sequence reduction. Detailed results are provided in Table 21.

Additional data-scaling analyses and examples are provided in Appendices C.3 and E.

## 5 Related Work

Spoken dialogue models. Existing SDMs differ in how text and speech generation are coupled. Think–Talk architectures use separate Transformer backbones (Fang et al., 2025a; Wang et al., 2025b; Fang et al., 2025b; Wang et al., 2025a; Xu et al., 2025a,b; Team, 2026; Cui et al., 2026; Défossez et al., 2024; Team et al., 2025a); parallel-generation architectures use multiple output heads (KimiTeam et al., 2025; Chen et al., 2025; Xie and Wu, 2024a,b; Team et al., 2025b); and interleaved-generation architectures model interleaved text–speech sequences (Zhang et al., 2023; Li et al., 2025b; Zeng et al., 2024; Wu et al., 2025). Despite these differences, the generation token space is typically coupled to the space expected by the speech decoder. CoSTA instead decouples generation and synthesis through compatibility-preserving posthoc adaptation, enabling compact, losslessly expandable speech-token generation while retaining the original speech tokenizer and decoder.

Discrete speech tokens. Discrete speech tokens include acoustic tokens from codec models (Zeghidour et al., 2022; Défossez et al., 2023; Kumar et al., 2023; Ji et al., 2025; Zhang et al., 2024) and semantic tokens from discriminative (Du et al., 2024a,b, 2025) or self-supervised speech models (Baevski et al., 2020a,b; Hsu et al., 2021; Chen et al., 2022; Chung et al., 2021). Lower-rate tokenizers shorten speech-token sequences (Défossez et al., 2024; Zeng et al., 2025; Guo et al., 2024; Wang et al., 2025c; Guo et al., 2026), but aggressive rate reduction may introduce compression–quality trade-offs, while tokenizer replacement may require retraining or adapting the tokenizer, speech decoder, and SDM together. Moreover, this approach still couples generation and synthesis to the same token space. Acoustic-piece and BPE methods have been explored in ASR (Ren et al., 2022; Chang et al., 2023, 2024), speech continuation (Shen et al., 2024), and TTS (Zhu et al., 2023; Li et al., 2024), primarily through native training of small, task-specific models. In contrast, CoSTA enables post-hoc adaptation of existing large-scale SDMs for efficient compressed-token generation. We further uncover a consistent scaling regularity of speech-token compression and characterize the vocabulary-size trade-off.

## 6 Conclusion

We uncover an approximate log-linear scaling regularity in speech-token BPE and introduce CoSTA, which decouples generation and synthesis token spaces while retaining the original tokenizer and decoder. CoSTA achieves approximately 1.7× expanded speech-token generation speed and improves system efficiency while preserving response quality. It also supports more sustained interaction, reveals vocabulary-size trade-offs, and transfers across SDM architectures, complementing low-rate tokenization.

## Limitations

This work has several limitations and possible extensions. First, our dialogue-adaptation data is constructed from synthesized speech-to-speech examples. While such data provides controllable and consistent speech responses for studying speechtoken compression, it is also a practical and widely adopted choice for training spoken dialogue models, as large-scale, high-quality real speech-tospeech dialogue data is difficult to collect. Future work may further explore how synthetic and real conversational speech data complement each other. Second, we adopt BPE as a deterministic, lossless, and easily integrated compression method. Future work may investigate other lossless speech-token segmentation strategies and compare their behavior in spoken dialogue modeling. Finally, our evaluation focuses on general speech-to-speech dialogue and efficiency. Future work may extend the analysis to finer-grained scenarios, such as emotional dialogue, personalized speaking styles, and diverse acoustic environments.

## References

Alexei Baevski, Steffen Schneider, and Michael Auli. 2020a. vq-wav2vec: Self-supervised learning of discrete speech representations. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020b. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Xuankai Chang, Brian Yan, Kwanghee Choi, Jee-Weon Jung, Yichen Lu, Soumi Maiti, Roshan S. Sharma, Jiatong Shi, Jinchuan Tian, Shinji Watanabe, Yuya Fujita, Takashi Maekaku, Pengcheng Guo, Yao-Fei Cheng, Pavel Denisov, Kohei Saijo, and Hsiu-Hsuan Wang. 2024. Exploring speech recognition, translation, and understanding with discrete speech units: A comparative study. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic of Korea, April 14-19, 2024, pages 11481–11485. IEEE.

Xuankai Chang, Brian Yan, Yuya Fujita, Takashi Maekaku, and Shinji Watanabe. 2023. Exploration of efficient end-to-end ASR using discretized input from self-supervised learning. In 24th Annual Conference ofthe International Speech Communication

Association, Interspeech 2023, Dublin, Ireland, August 20-24, 2023, pages 1399–1403. ISCA.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, and Furu Wei. 2022. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE J. Sel. Top. Signal Process., 16(6):1505–1518.

Wenxi Chen, Ziyang Ma, Ruiqi Yan, Yuzhe Liang, Xiquan Li, Ruiyang Xu, Zhikang Niu, Yanqiao Zhu, Yifan Yang, Zhanxun Liu, Kai Yu, Yuxuan Hu, Jinyu Li, Yan Lu, Shujie Liu, and Xie Chen. 2025. Slamomni: Timbre-controllable voice interaction system with single-stage training. In Findings of the Associationfor Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Findings of ACL, pages 2262–2282. Association for Computational Linguistics.

Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, and Yonghui Wu. 2021. w2v-bert: Combining contrastive learning and masked language modeling for self-supervised speech pre-training. In IEEE Automatic Speech Recognition and Understanding Workshop, ASRU 2021, Cartagena, Colombia, December 13-17, 2021, pages 244–250. IEEE.

Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, Jiancheng Gui, Luoyuan Zhang, Xian Sun, Fuwei Huang, Moye Chen, Zhuo Lin, Hanyu Liu, Qingxin Gui, Qingzhe Han, and 17 others. 2026. Minicpm-o 4.5: Towards realtime full-duplex omni-modal interaction. Preprint, arXiv:2604.27393.

Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. 2023. High fidelity neural audio compression. Trans. Mach. Learn. Res., 2023.

Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, Zhifu Gao, and Zhijie Yan. 2024a. Cosyvoice: A scalable multilingual zero-shot textto-speech synthesizer based on supervised semantic tokens. Preprint, arXiv:2407.05407.

Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, and 3 others. 2025. Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training. Preprint, arXiv:2505.17589.

Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, Fan Yu, Huadai Liu, Zhengyan Sheng, Yue Gu, Chong Deng, Wen Wang,

Shiliang Zhang, Zhijie Yan, and Jingren Zhou. 2024b. Cosyvoice 2: Scalable streaming speech synthesis with large language models. Preprint, arXiv:2412.10117.

Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. 2024. Moshi: a speech-text foundation model for real-time dialogue. Preprint, arXiv:2410.00037.

Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. 2025a. Llama-omni: Seamless speech interaction with large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, and Yang Feng. 2025b. Llama-omni 2: Llm-based real-time spoken chatbot with autoregressive streaming speech synthesis. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 18617– 18629. Association for Computational Linguistics.

Philip Gage. 1994. A new algorithm for data compression. The C Users Journal, 12(2):23–38.

Zhifu Gao, Shiliang Zhang, Ian McLoughlin, and Zhijie Yan. 2022. Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition. In 23rd Annual Conference of the International Speech Communication Association, Interspeech 2022, Incheon, Korea, September 18-22, 2022, pages 2063–2067. ISCA.

Haohan Guo, Fenglong Xie, Kun Xie, Dongchao Yang, Dake Guo, Xixin Wu, and Helen Meng. 2024. Socodec: A semantic-ordered multi-stream speech codec for efficient language model based text-tospeech synthesis. In IEEE Spoken Language Technology Workshop, SLT 2024, Macao, December 2-5, 2024, pages 645–651. IEEE.

Yiwei Guo, Zhihan Li, Hankun Wang, Bohan Li, Chongtian Shao, Hanglei Zhang, Chenpeng Du, Xie Chen, Shujie Liu, and Kai Yu. 2026. Recent advances in discrete speech tokens: A review. IEEE Trans. Pattern Anal. Mach. Intell., 48(4):4184–4204.

Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang, Kai Chen, Pengyuan Zhang, and Zhizheng Wu. 2024. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation. In IEEE Spoken Language Technology Workshop, SLT 2024, Macao, December 2-5, 2024, pages 885–890. IEEE.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE ACM Trans. Audio Speech Lang. Process., 29:3451–3460.

Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, Xinyu Zhang, Pei Zhang, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. 2026. Qwen3-TTS technical report. Preprint, arXiv:2601.15621.

Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, Zehan Wang, Ruiqi Li, Ziang Zhang, Xiaoda Yang, Rongjie Huang, Yidi Jiang, Qian Chen, Siqi Zheng, and Zhou Zhao. 2025. Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. 2019. A study of bfloat16 for deep learning training. Preprint, arXiv:1905.12322.

KimiTeam, Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, Zhengtao Wang, Chu Wei, Yifei Xin, Xinran Xu, Jianwei Yu, Yutao Zhang, Xinyu Zhou, Y. Charles, and 21 others. 2025. Kimiaudio technical report. Preprint, arXiv:2504.18425.

Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, and Kundan Kumar. 2023. Highfidelity audio compression with improved RVQGAN. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Ann Lee, Hongyu Gong, Paul-Ambroise Duquenne, Holger Schwenk, Peng-Jen Chen, Changhan Wang, Sravya Popuri, Yossi Adi, Juan Pino, Jiatao Gu, and Wei-Ning Hsu. 2022. Textless speech-to-speech translation on real data. In Proceedings of the 2022 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 860–872, Seattle, United States. Association for Computational Linguistics.

Bohan Li, Feiyu Shen, Yiwei Guo, Shuai Wang, Xie Chen, and Kai Yu. 2024. On the effectiveness of acoustic BPE in decoder-only TTS. In 25th Annual Conference of the International Speech Communication Association, Interspeech 2024, Kos, Greece, September 1-5, 2024. ISCA.

Chong Li, Yingzhuo Deng, Wen Yang, Jiajun Zhang, and Chengqing Zong. 2026. Tokalign++: Advancing vocabulary adaptation via better token alignment. Preprint, arXiv:2605.13429.

Chong Li, Jiajun Zhang, and Chengqing Zong. 2025a. Tokalign: Efficient vocabulary adaptation via token alignment. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 4109–4126. Association for Computational Linguistics.

Tianpeng Li, Jun Liu, Tao Zhang, Yuanbo Fang, Da Pan, Mingrui Wang, Zheng Liang, Zehuan Li, Mingan Lin, Guosheng Dong, Jianhua Xu, Haoze Sun, Zenan Zhou, and Weipeng Chen. 2025b. Baichuan-audio: A unified framework for end-to-end speech interaction. Preprint, arXiv:2502.17239.

Songting Liu. 2024. Zero-shot voice conversion with diffusion transformers. Preprint, arXiv:2411.09943.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory F. Diamos, Erich Elsen, David García, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed precision training. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net.

OpenAI. 2024. GPT-4o mini: Advancing Cost-Efficient Intelligence. https://openai.com/index/gpt-4 o-mini-advancing-cost-efficient-intellige nce/. Accessed: 2026-05-25.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, Proceedings of Machine Learning Research, pages 28492–28518. PMLR.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, and 1 others. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2020, Virtual Event / Atlanta, Georgia, USA, November 9-19, 2020, page 20. IEEE/ACM.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In KDD ’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020, pages 3505–3506. ACM.

Shuo Ren, Shujie Liu, Yu Wu, Long Zhou, and Furu Wei. 2022. Speech pre-training with acoustic piece. In 23rd Annual Conference ofthe International Speech Communication Association, Interspeech 2022, Incheon, Korea, September 18-22, 2022, pages 2648– 2652. ISCA.

Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari. 2022. UTMOS: utokyo-sarulab system for voicemos challenge 2022. In 23rd Annual Conference ofthe International Speech Communication Association, Interspeech 2022, Incheon, Korea, September 18-22, 2022, pages 4521–4525. ISCA.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers. The Association for Computer Linguistics.

Feiyu Shen, Yiwei Guo, Chenpeng Du, Xie Chen, and Kai Yu. 2024. Acoustic BPE for speech generation with discrete tokens. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic ofKorea, April 14-19, 2024, pages 11746–11750. IEEE.

Core Team, Dong Zhang, Gang Wang, Jinlong Xue, Kai Fang, Liang Zhao, Rui Ma, Shuhuai Ren, Shuo Liu, Tao Guo, Weiji Zhuang, Xin Zhang, Xingchen Song, Yihan Yan, Yongzhe He, Cici, Bowen Shen, Chengxuan Zhu, Chong Ma, and 81 others. 2025a. Mimo-audio: Audio language models are few-shot learners. Preprint, arXiv:2512.23808.

Meituan LongCat Team, Bairui Wang, Bayan, Bin Xiao, Bo Zhang, Bolin Rong, Borun Chen, Chang Wan, Chao Zhang, Chen Huang, Chen Chen, Chen Chen, Chengxu Yang, Chengzuo Yang, Cong Han, Dandan Peng, Delian Ruan, Detai Xin, Disong Wang, and 114 others. 2025b. Longcat-flash-omni technical report. Preprint, arXiv:2511.00279.

Qwen Team. 2026. Qwen3.5-omni technical report. Preprint, arXiv:2604.15804.

Chen Wang, Tianyu Peng, Wen Yang, Yinan Bai, Guangfu Wang, Jun Lin, Lanpeng Jia, Lingxiang Wu, Jinqiao Wang, Chengqing Zong, and Jiajun Zhang. 2025a. Opens2s: Advancing fully opensource end-to-end empathetic large speech language model. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025 - System Demonstrations, Suzhou, China, November 4-9, 2025, pages 906–917. Association for Computational Linguistics.

Xiong Wang, Yangze Li, Chaoyou Fu, Yike Zhang, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, and Long Ma. 2025b. Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen LLM. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada,

July 13-19, 2025, Proceedings of Machine Learning Research. PMLR / OpenReview.net.

Yuancheng Wang, Dekun Chen, Xueyao Zhang, Junan Zhang, Jiaqi Li, and Zhizheng Wu. 2025c. Tadicodec: Text-aware diffusion speech tokenizer for speech language modeling. Preprint, arXiv:2508.16790.

Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, Mingrui Chen, Peng Liu, Wang You, Xiangyu Tony Zhang, Xingyuan Li, Xuerui Yang, Yayue Deng, Yechang Huang, Yuxin Li, and 90 others. 2025. Step-audio 2 technical report. Preprint, arXiv:2507.16632.

Zhifei Xie and Changqiao Wu. 2024a. Mini-omni: Language models can hear, talk while thinking in streaming. Preprint, arXiv:2408.16725.

Zhifei Xie and Changqiao Wu. 2024b. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. Preprint, arXiv:2410.11190.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025a. Qwen2.5-omni technical report. Preprint, arXiv:2503.20215.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, and 19 others. 2025b. Qwen3-omni technical report. Preprint, arXiv:2509.17765.

Ruiqi Yan, Xiquan Li, Wenxi Chen, Zhikang Niu, Chen Yang, Ziyang Ma, Kai Yu, and Xie Chen. 2025. Urobench: Towards comprehensive evaluation for endto-end spoken dialogue models. In Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 17211–17242. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2022. Soundstream: An end-to-end neural audio codec. IEEE ACM Trans. Audio Speech Lang. Process., 30:495– 507.

Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. 2024. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot. Preprint, arXiv:2412.02612.

Aohan Zeng, Zhengxiao Du, Mingdao Liu, Lei Zhang, Shengmin Jiang, Yuxiao Dong, and Jie Tang. 2025. Scaling speech-text pre-training with synthetic interleaved data. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. 2023. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, Findings of ACL, pages 15757–15773. Association for Computational Linguistics.

Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, and Xipeng Qiu. 2024. Speechtokenizer: Unified speech tokenizer for speech language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Xinfa Zhu, Yuanjun Lv, Yi Lei, Tao Li, Wendi He, Hongbin Zhou, Heng Lu, and Lei Xie. 2023. Vec-tok speech: speech vectorization and tokenization for neural speech generation. Preprint, arXiv:2310.07246.

## A Implementation Details and Hyperparameters

This section reports the implementation details and hyperparameters used in CoSTA. Unless otherwise specified, all models are initialized from Step-Audio 2 mini (Wu et al., 2025). The original speech tokenizer and speech decoder are kept unchanged throughout all experiments. The original speech-token vocabulary contains 6,656 tokens. We train the BPE tokenizer using the HuggingFace tokenizers $\mathrm { l i b r a r y } ^ { 2 }$ on speech-token sequences extracted from the general speech-to-speech dialogue corpus, and extend the original vocabulary with 80K additional BPE speech tokens. Since the BPE trainer does not directly operate on integervalued speech-token sequences, we map each original speech-token ID to a unique Unicode character before BPE training. During BPE expansion, generated BPE-token IDs are first converted into their corresponding Unicode strings, and BPE expansion is performed in the Unicode space; the resulting BPE-expanded Unicode sequence is then mapped back to the corresponding original speechtoken IDs. This conversion is deterministic and does not change the underlying speech-token sequence. All adaptation and inference experiments are conducted on NVIDIA A100 80GB GPUs. For all adaptation stages, we use the AdamW optimizer (Loshchilov and Hutter, 2019), train with bfloat16 mixed precision (Micikevicius et al., 2018; Kalamkar et al., 2019), and use DeepSpeed (Rasley et al., 2020) with ZeRO-1 optimizer-state partitioning (Rajbhandari et al., 2020) for distributed training.

## A.1 Pronunciation Adaptation

Pronunciation adaptation is conducted on compressed general TTS data. Following the progressive adaptation strategy, we split the pronunciationadaptation data into two halves. In the first stage, only the embedding and LM head layers of the vocabulary-extended model are updated. We use a relatively large learning rate of $1 . 6 \times 1 0 ^ { - 4 }$ in this stage, which leads to faster loss reduction and more stable optimization as shown in Figures 6 and 8. In the second stage, intermediate model layers are further unfrozen and trained with a smaller learning rate of $2 . 0 \times 1 0 ^ { - 5 }$ to complete pronunciation adaptation. The hyperparameters are shown in Table 7.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Data split</td><td>50% / 50%</td></tr><tr><td>Stage-1 trainable params</td><td>Embedding, LM head</td></tr><tr><td>Stage-2 trainable params Optimizer</td><td>+ Intermediate layers AdamW</td></tr><tr><td>Stage-1 learning rate</td><td> $1 . 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Stage-2 learning rate</td><td> $2 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Scheduler</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>0.03</td></tr><tr><td>Global batch size</td><td>256</td></tr><tr><td>Precision</td><td></td></tr><tr><td>Gradient clipping norm</td><td>bfloat16</td></tr><tr><td>Epochs</td><td>1.0</td></tr><tr><td>GPUs</td><td>3 8</td></tr></table>

Table 7: Hyperparameters used during pronunciation adaptation.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Trainable params</td><td>All LLM parameters</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Scheduler</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>0.03</td></tr><tr><td>Global batch size</td><td>256</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td>Gradient clipping norm</td><td>1.0</td></tr><tr><td>Epochs</td><td>3</td></tr><tr><td>GPUs</td><td>8</td></tr></table>

Table 8: Hyperparameters used during dialogue adaptation.

## A.2 Dialogue Adaptation

After pronunciation adaptation, we perform dialogue adaptation on compressed general speech-tospeech dialogue data. The target responses contain interleaved text and speech tokens, where only speech tokens are BPE-compressed and text tokens remain unchanged. The compressed speech tokens are arranged in the response sequence according to the original interleaving ratio between text tokens and speech tokens used by the base SDM. The hyperparameters are shown in Table 8.

## A.3 Inference

During inference, the adapted SDM generates an interleaved sequence of text tokens and speech tokens from the extended vocabulary. The generated speech tokens may include both original speech tokens and newly added BPE speech tokens. We apply BPE expansion only to the generated speechtoken stream: original speech tokens are kept unchanged, while each merged BPE token is deterministically expanded into its original speech-token sequence. The expanded speech tokens are then fed into the unchanged speech decoder for waveform synthesis. The inference hyperparameters are shown in Table 9.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Decoding mode</td><td>Greedy</td></tr><tr><td>Repetition penalty</td><td>1.2</td></tr><tr><td>Max new tokens</td><td>2048</td></tr><tr><td>BPE expansion</td><td>Enabled</td></tr><tr><td>Expansion target</td><td>Speech-token stream only</td></tr><tr><td>Text-token stream</td><td>Unchanged</td></tr><tr><td>Speech decoder</td><td>Original decoder</td></tr><tr><td>Hardware</td><td>1 NVIDIA A100 GPU</td></tr></table>

Table 9: Hyperparameters used during inference.

## A.4 Adaptation Cost

For CoSTA-ZH on eight A100 80GB GPUs, the two pronunciation-adaptation stages take 1:29:41 and 4:34:00, and dialogue adaptation takes 3:52:54. The total is 9:56:35 of training wall-clock time, approximately 79.5 A100 GPU-hours. Data preparation is additional: timbre conversion and dialogue synthesis each take approximately 1–2 days using eight A100 GPUs. These one-time costs should be considered alongside the inference savings; they are not included in the training-only total.

## B Training Data and Benchmarks

This section provides additional details about the training data and evaluation benchmarks used in our experiments.

## B.1 Training Data

Pronunciation-adaptation data. For pronunciation adaptation, we use Emilia (He et al., 2024), a large-scale multilingual speech generation dataset constructed from in-the-wild speech. Emilia contains diverse speech data across multiple languages and speaking styles, making it suitable for training speech generation models. In this work, we sample 300 hours of Chinese speech and 300 hours of English speech from Emilia. To reduce speaker variation and improve speech-token pattern consistency, we convert the sampled utterances into a unified timbre using Seed-VC (Liu, 2024). The converted speech is then tokenized by the original speech tokenizer of Step-Audio 2 mini (Wu et al., 2025), and the resulting speech tokens are compressed with the learned BPE tokenizer. Text tokens remain unchanged. The compressed speech tokens are arranged according to the original textspeech interleaving pattern used by the base SDM.

<table><tr><td>Data Type</td><td>Language</td><td>Scale</td></tr><tr><td>Text-to-speech</td><td>Chinese English</td><td>300h 300h</td></tr><tr><td>Speech dialogue</td><td>Chinese English</td><td>49,664 examples 49,408 examples</td></tr></table>

Table 10: Statistics of training data.

<table><tr><td>Dataset</td><td>#Samples</td><td>Task / Aspect</td><td>Eval. Mode</td></tr><tr><td>AlpacaEval-zh</td><td>147</td><td>Authentic, open-ended dialogue</td><td>Open</td></tr><tr><td>AlpacaEval</td><td>199</td><td>Authentic, open-ended dialogue</td><td>Open</td></tr><tr><td>APE-zh</td><td>190</td><td>Practical mathematical problems</td><td>QA</td></tr><tr><td>Gsm8kEval</td><td>582</td><td>Practical mathematical problems</td><td>QA</td></tr><tr><td>Claude-zh</td><td>222</td><td>Authentic, open-ended dialogue</td><td>Open</td></tr><tr><td>CommonEval</td><td>200</td><td>Authentic, open-ended dialogue</td><td>Open</td></tr><tr><td>LCSTS-zh</td><td>119</td><td>Summarization</td><td>Semi-open</td></tr><tr><td>Summary</td><td>118</td><td>Summarization</td><td>Semi-open</td></tr><tr><td>MLC-zh</td><td>145</td><td>Math, logic, and common sense</td><td>QA</td></tr><tr><td>MLC</td><td>177</td><td>Math, logic, and common sense</td><td>QA</td></tr><tr><td>Wildchat-zh</td><td>299</td><td>Real-world conversation</td><td>Open</td></tr><tr><td>WildchatEval</td><td>349</td><td>Real-world conversation</td><td>Open</td></tr></table>

Table 11: URO-Bench datasets used in our evaluation. The selected datasets cover understanding, reasoning, and oral conversation in Chinese and English speech-tospeech settings.

Dialogue-adaptation data. For dialogue adaptation, we follow the general-purpose bilingual speech-to-speech instruction-pair construction pipeline of OpenS2S (Wang et al., 2025a). Specifically, we first collect general text instructions from InstructS2S-200K (Fang et al., 2025a) and translate the English instructions into Chinese using Qwen3- 32B (Yang et al., 2025), yielding bilingual text instructions. We then select seed audio with neutral emotion labels and convert these text instructions into user speech queries. Following the speech response generation process in OpenS2S, Qwen3- 32B is used to generate dialogue-appropriate text responses, and CosyVoice 2 (Du et al., 2024b) synthesizes the corresponding assistant speech with a consistent reference voice. Following this pipeline, we initially construct 50K Chinese and 50K English general speech-to-speech instruction examples, each containing a user’s speech query and an assistant response. We further filter out examples whose query or response speech exceeds one minute, resulting in 49,664 Chinese and 49,408 English dialogue examples. The statistics of the training data are summarized in Table 10. During training, only speech tokens in the target response are BPE-compressed, while text tokens remain unchanged. The compressed speech tokens are arranged according to the original text-speech interleaving pattern used by the base SDM.

## B.2 Evaluation Benchmarks

We evaluate models on URO-Bench (Yan et al., 2025), a comprehensive benchmark for end-to-end spoken dialogue models in speech-to-speech scenarios. URO-Bench covers three major evaluation dimensions: Understanding, Reasoning, and Oral Conversation, and assesses not only responsecontent quality but also speech quality and speechtext alignment. In our experiments, we use 12 aligned Chinese and English datasets from URO-Bench, as summarized in Table 11. These benchmarks better reflect realistic spoken dialogue scenarios, making them well aligned with our goal of improving the efficiency and usability of speechto-speech SDMs.

Following the URO-Bench evaluation protocol, we evaluate output quality using its prompts and scoring procedures. Specifically, we report the following metrics: (1) G-Eval measures response-content quality by transcribing the generated speech with an ASR model and prompting GPT-4o mini (OpenAI, 2024) to score the transcription. We use Whisper-large-v3 (Radford et al., 2023) for English transcription and Paraformer-zh (Gao et al., 2022) for Chinese transcription. (2) UT-MOS (Saeki et al., 2022) estimates speech quality with a predicted mean opinion score that reflects clarity, naturalness, and fluency. (3) WER/CER evaluates the alignment between the speech and text outputs of SDMs by computing the Word Error Rate (WER) or Character Error Rate (CER) between the generated text and the transcription of the generated speech.

URO-Bench uses different evaluation prompts according to the task format. For open-ended tasks, including AlpacaEval-zh, AlpacaEval, Claude-zh, CommonEval, Wildchat-zh, and WildchatEval, we use prompt\_open. For semi-open tasks with reference answers, including LCSTS-zh and Summary, we use prompt\_semi\_open. For questionanswering tasks with deterministic references, including APE-zh, Gsm8kEval, MLC-zh, and MLC, we use prompt\_qa. These prompts are fed into GPT-4o mini together with the user instruction, the model response, and reference answer when available, to obtain the final G-Eval score. The complete prompts are provided below.

## prompt\_open

I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.

Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:

1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.

2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.

3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that do not contribute to the main point.

4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.

5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and model’s response:

\### [Instruction]

{question}

\### [Response]

{answer}

After evaluating, please output the score only without anything else. You do not need to provide any explanations.

## prompt\_semi\_open

I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.

Your task is to rate the model’s responses based on the provided user input transcription [Instruction], the model’s output transcription [Response] and some suggested answers [Reference]. The model’s response does not necessarily have to be identical to the suggested answers, as long as it aligns with the question and is reasonable.

Please evaluate the response on a scale of 1 to 5:

1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information. The response does not align with the question in any meaningful way.

2 points: The response is somewhat relevant but lacks accuracy, completeness, or coherence. It may partially address the query but introduces unnecessary information or deviates from the core issue. The response may not align well with the suggested answer but still provides some value.

3 points: The response is relevant and mostly accurate, but may lack conciseness or clarity. It addresses the question reasonably, but there might be slight deviations in approach or content. While it may not strictly align with the suggested answer, it still effectively addresses the core of the query.

4 points: The response is relevant, accurate, and concise. It provides a clear answer to the user’s question and avoids unnecessary details. While it may not exactly mirror the suggested answer, it effectively addresses the user’s query in a logical and well-reasoned manner.

5 points: The response is exceptionally relevant, accurate, and concise. It directly addresses the user’s query in the most efficient manner, providing exactly the information needed. The response may differ from the suggested answer in phrasing or approach but still aligns perfectly with the intent of the query, demonstrating a high level of reasoning and clarity.

Below are the transcription of user’s instruction, model’s response and the reference answer:

\### [Instruction]

{question}

\### [Response]

{answer}

\### [Reference]

{reference}

After evaluating, please output the score only without anything else. You do not need to provide any explanations.

## prompt\_qa

I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.

Your task is to rate the model’s responses based on the provided user input transcription [Question], the model’s output transcription [Response] and the correct answer [Reference]. Below are the transcription of user’s instruction, model’s response and the reference answer:

\### [Question]

{question}

\### [Response]

{answer}

\### [Reference]

{reference}

Is the model’s response correct based on the question and reference answer?

Please only output a single “Yes” or “No”. Do not output anything else.

## C Additional Results

This section provides detailed quality results, initialization and data-scaling analyses, compression across corpus settings, first-packet latency, human evaluation, OpenS2S transfer, and vocabulary-size and adaptation analyses.

## C.1 Detailed Main Results

In the main paper, we report per-dataset G-Eval scores and average ASR-CER/WER and UT-MOS results for Chinese and English URO-Bench datasets. To provide a more complete comparison, Tables 14 and 15 further report the detailed values of all metrics on each dataset, including G-Eval, ASR-CER/WER, and UTMOS. These results detail the average quality trends discussed in the main paper and the variation across tasks; improvements are not uniform for every metric and dataset.

## C.2 Initialization Strategy Analysis

Initialization strategies. We compare three strategies for initializing the embedding and LMhead rows of newly added BPE units. Random initialization initializes the new rows randomly, without using their constituent original speech tokens. Select initialization copies the embedding and LM-head row of one randomly selected token from the deterministic original-token expansion of each BPE unit. Mean initialization, used by CoSTA, averages the corresponding rows of all constituent tokens as defined in Eq. 4.

![](images/c6dc984b3bd9c390befac326a009b24634c1681ad5a61d528af59800b66d1193.jpg)  
Figure 8: Gradient norm during the first stage of pronunciation adaptation. Mean initialization yields more stable optimization than the alternatives.

We evaluate these strategies with learning rates of $2 \times 1 0 ^ { - 5 }$ and $1 . 6 \times 1 0 ^ { - 4 }$ during the first phase of pronunciation adaptation. Figures 6 and 8 show that mean initialization converges faster and maintains more stable gradients than the alternatives, particularly at the larger learning rate. This suggests that aggregating the constituent originaltoken representations provides a better starting point for adapting the newly added BPE units.

## C.3 Data Scaling Analysis

We further analyze how the amount of adaptation data affects CoSTA. These scaling experiments were conducted in an early setting without progressive adaptation, and were mainly evaluated on four Chinese URO-Bench datasets: AlpacaEvalzh, APE-zh, Claude-zh, and MLC-zh. We study two scaling settings: varying the amount of pronunciation-adaptation data while keeping the dialogue-adaptation data fixed, and varying the amount of dialogue-adaptation data while keeping the pronunciation-adaptation data fixed.

Scaling pronunciation-adaptation data. Table 16 reports the results when varying the amount of TTS data used for pronunciation adaptation, while fixing the dialogue-adaptation data to 50K examples. Increasing the amount of TTS data generally improves compressed-token learning, especially in terms of compression rate and speech-text alignment. However, the improvement becomes less pronounced as the TTS data size continues to increase. This suggests that pronunciation adaptation benefits from sufficient TTS data, while the marginal gain becomes smaller beyond a certain scale. Based on this trade-off, we use 300h TTS data as the default setting, which provides reliable compressed speech-token generation while keeping the adaptation cost moderate.

![](images/1c79e60738e7af7711f698ab9df362fadca8421d71821a61b6a10adb75775913.jpg)  
Figure 9: Held-out compression across data sources and languages with S3Tokenizer V2. Points are test measurements; lines are fits against log added vocabulary size. Emilia-TC denotes timbre-converted Emilia.

Scaling dialogue-adaptation data. Table 17 reports the results when varying the amount of speech-to-speech dialogue data, while fixing the pronunciation-adaptation data to 300h. The results show that using 50K dialogue examples provides a strong balance between compression behavior and response quality. Therefore, we adopt 300h TTS data and 50K dialogue examples as the experimental configuration.

## C.4 Compression Across Data Sources and Languages

Figure 9 evaluates S3Tokenizer V2 on test splits of Emilia, timbre-converted Emilia (Emilia-TC), and synthesized dialogue speech, in Chinese, English, and mixed-language settings. Each BPE vocabulary is learned on the corresponding training split and fixed at test time. All nine curves retain the approximate log-linear trend over 10K–320K, with $R ^ { 2 } = 0 . 9 9 4 2 – 0 . 9 9 9 7$ and RMSE 0.10–0.47 pp. Emilia-TC and dialogue speech remain more compressible than original Emilia in these measurements. This demonstrates the trend across settings, rather than transfer of one vocabulary between them. Appendix D gives protocols, complete fits, and training comparisons.

## C.5 First-Packet Latency

Table 18 separates LLM prefill, decoding to the first audio packet, and speech-decoder latency.

CoSTA reduces the LLM decode component from 164.42 to 122.98 ms in Chinese and from 165.08 to 115.96 ms in English, corresponding to reductions of 25.2% and 29.8%. Prefill and speech-decoder latency remain largely unchanged. The resulting first-packet latency reductions are 9.4% and 9.6%, reflecting that CoSTA primarily accelerates autoregressive speech-token decoding while retaining the original speech decoder; over longer responses, these generation-stage savings accumulate, leading to larger full-response latency benefits.

## C.6 Human Evaluation Protocol and Detailed Results

We randomly sample five queries from each of the 12 URO-Bench datasets, for 60 queries in total. Six evaluators rate the responses of Step-Audio 2 mini, matched Vanilla-FT, and CoSTA on a 1–5 scale for speech fluency and semantic quality. This yields $6 0 \times 3 \times 6 = 1 . 0 8 0$ response-level evaluations, each with two scores. Tables 19 and 20 report the means by dataset and language.

For Figure 5, we compare the MOS scores given by the same evaluator to the same query for CoSTA and each baseline. A higher, equal, or lower score counts as a win, tie, or loss based on the MOS ratings. Each language/metric/baseline comparison therefore contains $3 0 \times 6 = 1 8 0$ paired scores. CoSTA has more wins than losses in every aggregate comparison, with ties forming the majority for fluency. The paired comparisons provide qualitative evidence of perceived quality preservation, while detailed task-level variations are reported in Appendix C.6.

## C.7 Detailed OpenS2S Transfer Results

We adapt OpenS2S using vocabulary extension followed by both pronunciation and dialogue adaptation in each language. It retains its native GLM-4-Voice tokenizer and speech decoder. Chinese evaluation uses AlpacaEval-zh, Claude-zh, and LCSTS-zh; English evaluation uses AlpacaEval, CommonEval, and Summary. The matched Vanilla-FT baselines use the corresponding adaptation data without BPE compression. Table 21 reports generated-token compression, ASR-CER for Chinese or ASR-WER for English, UTMOS, and G-Eval. CoSTA improves average content scores and alignment over matched fine-tuning while slightly lowering average UTMOS. These results demonstrate that CoSTA’s compressed-token adaptation transfers across SDM architectures while maintaining favorable compression–quality trade-offs.

## C.8 Why Dialogue Adaptation Matters

Removing dialogue adaptation increases $\rho _ { g }$ from 66.73% to 81.05% and reduces G-Eval from 72.27 to 24.35 (Table 4). To examine this degradation, we analyze token usage on all six Chinese benchmarks, with 1,122 responses per model. We pool generated speech tokens across responses, so the usage statistics are token-weighted rather than responseweighted.

Reduced use of merged tokens. Table 22 distinguishes original speech tokens from added BPE units. Original and BPE shares are measured before expansion, and Original/BPE is the ratio of their counts. BPE coverage is the fraction of the expanded speech-token sequence contributed by BPE units; average BPE expansion is the mean number of original tokens represented by a generated BPE unit. For example, if $N _ { \mathrm { o r i g } }$ is the number of generated original tokens and B is the multiset of generated BPE units, coverage is

$$
\frac { \sum _ { z \in B } | \pi ( z ) | } { N _ { \mathrm { o r i g } } + \sum _ { z \in B } | \pi ( z ) | } .\tag{8}
$$

Without dialogue adaptation, the original-token share rises from 70.95% to 78.95%, an 8.00- percentage-point increase, while the Original/BPE ratio rises from 2.44 to 3.75. BPE coverage falls from 51.64% to 36.71%, a 14.93-percentage-point decrease, and average BPE expansion drops from 2.608 to 2.175. Table 22(b) further groups generated tokens by expansion length $L = \left| \pi ( z ) \right|$ where $L \ = \ 1$ denotes an original token. Units with $L \geq 5$ account for 2.29% of generated speech tokens in CoSTA-ZH but only 0.45% without dialogue adaptation. The ablated model therefore both uses merged units less often and selects shorter units when it does use them. These behaviors reduce the number of original tokens represented by each decoding step and help explain the weaker compression.

Failure to answer the user’s query. The content degradation also appears in representative responses. For an MLC-zh query, translated from Chinese as “If all flowers are plants, is a rose a plant?”, CoSTA-ZH answers “Yes, a rose is a plant.” The model without dialogue adaptation instead repeats the question. This example illustrates a failure to perform the dialogue task, consistent with the sharp decrease in G-Eval, rather than merely a change in token segmentation.

Taken together, these observations suggest that dialogue adaptation connects compressed-token generation learned from TTS data to appropriate spoken responses: it restores question-answering behavior while promoting more effective use of the extended BPE vocabulary. These complementary effects help explain why omitting the stage degrades both compression and response quality.

## C.9 Full-Model Results Across Vocabulary Sizes

Table 23 provides the complete AlpacaEval-zh results underlying Figure 7, including generatedtoken compression and absolute decoding speed. The 0K row is the original Step-Audio 2 mini model; the 10K–160K rows are adapted CoSTA-ZH models. Thus, the vocabulary-size comparison among adapted models uses the nonzero settings.

Increasing the added vocabulary from 10K to 80K reduces $\rho _ { g }$ from 76.66% to 67.50% and raises expanded speech-token speed from 78.67 to 103.03 tokens/s. At 160K, $\rho _ { g }$ further decreases to 63.62%, but speed reaches only 104.18 tokens/s, a 1.1% increase over 80K. Meanwhile, G-Eval decreases from 81.36 to 78.87, and UTMOS decreases by 0.02. These results motivate 80K as a balance of speed, compression, and quality under the tested conditions, rather than the setting that maximizes every metric.

## C.10 Computational Cost of Vocabulary Extension

Measurement setting. Vocabulary extension affects inference asymmetrically. The enlarged embedding table mainly increases parameter storage, while embedding lookup does not incur vocabularywide computation at each decoding step. In contrast, the LM head projects each hidden state over the full vocabulary, and the subsequent token selection also grows with vocabulary size. We therefore measure LM-head projection plus greedy argmax as the primary per-step computational overhead introduced by vocabulary extension. Transformer computation is excluded because its cost depends on the evolving prefix length, which itself changes with token compression and is therefore difficult to attribute to vocabulary extension alone.

Table 24 reports the complete 0K–320K measurements for S3Tokenizer V2 using identical BF16 hidden states on a single A100 GPU at batch size 1, with median latency reported per step. Compression uses exact pooled token counts from the held-out Dialogue-ZH corpus in Figure 9; it is distinct from generated-token compression in Table 23. At 80K, the added output-projection rows contain 286.72M parameters and occupy 546.88 MiB in BF16. These storage figures cover the LM head alone.

Compression versus per-step overhead. Let $t _ { M }$ be the median projection-and-selection latency with M added units and $\rho _ { M }$ the compression rate. We estimate the normalized output-layer cost for generating the same number of expanded speech tokens as

$$
C _ { \mathrm { h e a d } } ( M ) = \rho _ { M } \frac { t _ { M } } { t _ { 0 } } ,\tag{9}
$$

where $\rho _ { M }$ is represented as a fraction. This estimate captures the isolated output-layer cost and excludes Transformer computation. At 80K, test compression is 62.29%, while per-step latency rises from 0.6902 to 1.0251 ms (+48.52%), yielding a normalized cost of 0.925. At 160K, compression improves to 58.48%, but latency reaches 1.3598 ms (+97.01%), raising the normalized cost to 1.152. At 320K, compression reaches 54.94%, while latency increases to 2.0297 ms (+194.08%) and the normalized cost to 1.616. Thus, in this isolated measurement, continued compression gains eventually cease to offset the increasing per-step output-layer cost.

Relation to full-model decoding. The isolated measurements characterize the additional outputlayer cost of vocabulary extension, but do not determine full-model efficiency. Token compression also reduces the number of Transformer decoding steps, an effect captured by the full-model experiments rather than by $C _ { \mathrm { h e a d } }$ . Indeed, AlpacaEval-zh decoding speed still increases slightly from 80K to 160K despite the higher normalized output-layer cost. Together, these results show why stronger compression does not necessarily translate into proportional full-model speedup and motivate considering compression, runtime, and response quality jointly when selecting vocabulary size.

## D Held-Out Compression Analysis

## D.1 Evaluation and Fitting Protocol

Tokenizer configurations and sources. Table 12 documents the eight configurations and links their public reference releases. GLM-4-Voice, Mimi, and Qwen3-TTS follow their respective model families (Zeng et al., 2024; Défossez et al., 2024; Hu et al., 2026). The S3Tokenizer implementation provides the CosyVoice V1–V3 tokenizers (Du et al., 2024a,b, 2025). mHuBERT Base L11 KM1000 uses layer-11 features with 1,000 K-means clusters (Lee et al., 2022). Codebook sizes exclude added BPE units.

![](images/8d101401182487d59d119ddaea424140173f6f576e49f9bb86314b2e2bf70da7.jpg)  
Figure 10: Training-corpus compression across eight tokenizer configurations on Chinese dialogue speech, corresponding to the test results in Figure 2. Points are measurements and lines are log-linear fits. Mimi and Qwen3-TTS use their first semantic codebook.

Training and test evaluation. For each tokenizer/corpus setting and added vocabulary size $M \in \{ 1 0 , 2 0 , 4 0 , 8 0 , 1 6 0 , 3 2 0 \} .$ K, BPE is learned on the training corpus. The resulting vocabulary and merge rules are fixed when encoding the corresponding test corpus. Figure 2 reports Chinese dialogue test speech for eight tokenizer configurations; Figure 9 uses S3Tokenizer V2 at 25 Hz across three data sources and three language settings. Each setting has its own trained BPE vocabulary. The evaluation concerns held-out speech within each setting, not transfer of one vocabulary across settings. For Mimi and Qwen3-TTS, only the first semantic codebook is included.

Corpus-level compression. We pool token counts across utterances:

$$
\rho _ { \mathcal { D } } ( M ) = \frac { \sum _ { \mathbf { s } \in \mathcal { D } } \left| B _ { M } ( \mathbf { s } ) \right| } { \sum _ { \mathbf { s } \in \mathcal { D } } \left| \mathbf { s } \right| } .\tag{10}
$$

Thus, rates are token-weighted, not averages of per-utterance ratios. Figures 10 and 11 preserve the training measurements for comparison. These rates characterize encoding of existing speech and are distinct from compression of model-generated responses.

![](images/3982c75e2fbd551cd52ad52ee29d3e551029a606281639b00cd874e399ffa128.jpg)  
Figure 11: Training-corpus compression across data sources and languages with S3Tokenizer V2, corresponding to the test results in Figure 9. Points are measurements and lines are log-linear fits. Each vocabulary is learned on the training corpus represented by its curve.

Fitting and numerical precision. We fit $y = \alpha +$ βx by unweighted ordinary least squares, where $x = \log _ { 2 } ( M / 1 0 \mathrm { K } )$ and $y = 1 0 0 \rho$ . Here α is the fitted percentage at 10K and $\beta < 0$ is the change in percentage points per vocabulary doubling. This is the percentage-scale form of Equation 2. All six vocabulary settings have equal weight; 0K is excluded because its logarithm is undefined. With residuals $e _ { j } = y _ { j } - { \hat { y } } _ { j }$ , we report

$$
\begin{array} { c } { { \displaystyle \mathrm { R M S E } = \sqrt { \frac { 1 } { 6 } \sum _ { j = 1 } ^ { 6 } e _ { j } ^ { 2 } } , } } \\ { { { } } } \\ { { { \displaystyle R ^ { 2 } = 1 - \frac { \sum _ { j } e _ { j } ^ { 2 } } { \sum _ { j } ( y _ { j } - \bar { y } ) ^ { 2 } } } . } } \end{array}\tag{11}
$$

Training fits use the original plotted percentages rounded to 0.01 pp. Tokenizer test fits use the reported four-decimal ratios, also giving 0.01-pp precision; dataset/language test fits use exact pooled token counts. Consequently, the S3 V2 Chinesedialogue curves can differ slightly between panels despite representing the same setting. Tables 25 and 26 report all 34 fits.

## D.2 Fit Quality and Training–Test Consistency

Persistence on test speech. Across the eight tokenizer test curves, $R ^ { 2 }$ ranges from 0.9807 to 0.9997 and RMSE from 0.103 to 0.883 pp. Fitted compression falls by 2.70–4.19 pp per vocabulary doubling. Across the nine dataset/language test curves, $R ^ { 2 }$ ranges from 0.9942 to 0.9997 and RMSE from 0.104 to 0.474 pp, with decreases of 3.62–4.01 pp per doubling. Thus, the approximate trend persists across the evaluated test settings, with settingdependent levels and slopes.

<table><tr><td>Tokenizer</td><td>Public reference source</td><td>Codes</td><td></td><td>Hz Token stream used for BPE</td></tr><tr><td>GLM-4-Voice Tokenizer 12.5Hz</td><td>GLM-4-Voice tokenizer</td><td>16,384</td><td>12.5</td><td>Single quantized speech stream</td></tr><tr><td>S3Tokenizer V2 25Hz</td><td>CosyVoice2-0.5B</td><td>6,561</td><td>25</td><td>Single semantic-token stream</td></tr><tr><td>S3Tokenizer V3 25Hz</td><td>Fun-CosyVoice3-0.5B-2512</td><td>6,561</td><td>25</td><td>Single semantic-token stream</td></tr><tr><td>Mimi</td><td>kyutai/mimi</td><td>2,048</td><td>12.5</td><td>First semantic codebook only</td></tr><tr><td>S3Tokenizer V1 25Hz</td><td>CosyVoice-300M-25Hz</td><td>4,096</td><td>25</td><td>Single semantic-token stream</td></tr><tr><td>Qwen3-TTS-Tokenizer-12Hz</td><td>Qwen3-TTS tokenizer</td><td>2,048</td><td></td><td>12.5 First semantic codebook only</td></tr><tr><td>S3Tokenizer V1 50Hz</td><td>CosyVoice-300M</td><td>4,096</td><td></td><td>50 Single semantic-token stream</td></tr><tr><td>mHuBERT Base L11 KM1000</td><td>fairseq mHuBERT Base (L11, KM1000)</td><td>1,000</td><td></td><td>Layer-11 features quantized by K-means (K = 1,000)</td></tr></table>

Table 12: Tokenizer configurations for Chinese dialogue compression, using the figure names and linked public sources. Codes excludes added BPE units. Hz is the native rate; mHuBERT is omitted. Qwen3-TTS operates at 12.5 Hz despite its model name (official configuration).

Training–test consistency. The log-linear scaling trend remains consistent between training and held-out speech. Across all eight tokenizer configurations, test slopes are close to their training counterparts, with differences of 0.25–1.46 pp at 80K and 0.84–3.52 pp at 320K. For example, at 320K, GLM-4-Voice changes from 60.25% on training speech to 63.77% on held-out speech, while Mimi changes from 40.69% to 43.60%. Across data sources, the differences remain small: 320K gaps are at most 0.24 pp for Emilia and Emilia-TC, and 0.85–1.71 pp for dialogue speech. These results show that the observed scaling regularity consistently transfers from training data to heldout speech across tokenizer configurations, data sources, and languages.

Approximation quality and scope. Tables 25 and 26 quantify the approximation quality. The largest absolute test residual is 1.21 pp for Qwen3- TTS and 0.67 pp across dataset/language curves, while the fitted log-linear models achieve high R<sup>2</sup> across settings. These results show that the loglinear scaling trend provides an accurate empirical description over the tested 10K–320K vocabulary range across tokenizer configurations, data sources, and languages.

## E Case Study

We present representative case studies selected from the English AlpacaEval and WildchatEval benchmarks. Case Studies 1–2 cover factual and commonsense questions, Case Study 3 illustrates sentence rewriting, Case Study 4 presents an openended dialogue, and Case Study 5 shows an empathetic customer-service reply. Across these cases, CoSTA generates compact speech-token sequences while preserving response quality; the generated BPE units are deterministically expanded back to the original speech-token space before waveform synthesis. These examples qualitatively demonstrate efficient compressed-token generation through token-space decoupling with the unchanged speech decoder.

## F Licenses of Scientific Artifacts

We follow and report the licenses or usage terms of the scientific artifacts used in this work in Table 13.

<table><tr><td>Name Step-Audio 2 mini</td><td>License / Terms</td></tr><tr><td>Qwen2.5-Omni Kimi-Audio GLM-4-Voice / Tokenizer OpenS2S Emilia InstructS2S-200K Seed-VC Qwen3-32B CosyVoice 1/2/3 (S3Tokenizer V1–V3) Mimi Qwen3-TTS Tokenizer mHuBERT / fairseq URO-Bench UTMOS HuggingFace tokenizers Transformers vLLM DeepSpeed</td><td>Apache 2.0 license Apache 2.0 license Apache 2.0 / MIT licenses glm-4-voice Model License Apache 2.0 license CC BY-NC 4.0 license CC BY-NC 4.0 license GPL-3.0 license Apache 2.0 license Apache 2.0 license CC BY 4.0 license Apache 2.0 license MIT license MIT license MIT license Apache 2.0 license Apache 2.0 license Apache 2.0 license Apache 2.0 license</td></tr></table>

Table 13: Licenses and usage terms of scientific artifacts used in this work.

<table><tr><td rowspan="2">Model</td><td colspan="3">AlpacaEval-zh</td><td colspan="3">APE-zh</td><td colspan="3">Claude-zh</td><td colspan="3">Avg</td></tr><tr><td>CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td>CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td>CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td>CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td>Qwen2.5-Omni</td><td>2.09%</td><td>2.93</td><td>74.10</td><td>8.50%</td><td>3.02</td><td>62.11</td><td>2.47%</td><td>2.73</td><td>78.86</td><td>2.93%</td><td>2.95</td><td>74.28</td></tr><tr><td>Kimi-Audio</td><td>4.71%</td><td>2.52</td><td>77.05</td><td>8.29%</td><td>2.59</td><td>39.12</td><td>11.12%</td><td>2.08</td><td>80.24</td><td>4.88%</td><td>2.52</td><td>71.93</td></tr><tr><td>GLM-4-Voice</td><td>3.65%</td><td>2.86</td><td>84.76</td><td>5.51%</td><td>3.14</td><td>23.16</td><td>7.13%</td><td>2.59</td><td>83.54</td><td>3.77%</td><td>2.94</td><td>71.85</td></tr><tr><td>Step-Audio 2 mini</td><td>i 8.48%</td><td>3.58</td><td>80.18</td><td>4.35%</td><td>3.87</td><td>21.93</td><td>20.28%</td><td>3.33</td><td>72.40</td><td>7.49%</td><td>3.62</td><td>69.24</td></tr><tr><td>Vanilla-FT-ZH</td><td>5.11%</td><td>3.76</td><td>79.77</td><td>0.95%</td><td>4.06</td><td>21.40</td><td>19.51%</td><td>3.41</td><td>80.35</td><td>4.96%</td><td>3.77</td><td>69.88</td></tr><tr><td>Vanilla-FT-MIX</td><td>6.32%</td><td>3.74</td><td>81.27</td><td>1.00%</td><td>4.12</td><td>20.53</td><td>27.73%</td><td>3.39</td><td>79.28</td><td>7.02%</td><td>3.77</td><td>70.26</td></tr><tr><td>CoSTA-ZH</td><td>1.92%</td><td>3.70</td><td>81.36</td><td>1.37%</td><td>4.04</td><td>24.56</td><td>2.62%</td><td>3.43</td><td>86.49</td><td>1.51%</td><td>3.75</td><td>72.27</td></tr><tr><td>CoSTA-MIX</td><td>1.96%</td><td>3.70</td><td>80.95</td><td>2.33%</td><td>4.07</td><td>24.56</td><td>2.55%</td><td>3.39</td><td>85.05</td><td>1.65%</td><td>3.75</td><td>71.53</td></tr><tr><td rowspan="2">Model</td><td colspan="3">LCSTS-zh</td><td colspan="3">MLC-zh</td><td colspan="3">Wildchat-zh</td><td colspan="3">Avg</td></tr><tr><td>CER↓</td><td>UTMOS↑ G-Eval↑</td><td></td><td>CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td>CER↓</td><td></td><td>UTMOS↑ G-Eval↑</td><td>CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td>Qwen2.5-Omni</td><td>1.86%</td><td>3.04</td><td>69.75</td><td>1.31%</td><td>3.07</td><td>82.53</td><td>1.37%</td><td>2.88</td><td>78.35</td><td>2.93%</td><td>2.95</td><td>74.28</td></tr><tr><td>Kimi-Audio</td><td>2.42%</td><td>2.55</td><td>76.13</td><td>0.75%</td><td>2.83</td><td>80.92</td><td>1.96%</td><td>2.55</td><td>78.08</td><td>4.88%</td><td>2.52</td><td>71.93</td></tr><tr><td>GLM-4-Voice</td><td>1.80%</td><td>2.92</td><td>76.36</td><td>1.74%</td><td>3.24</td><td>78.85</td><td>2.78%</td><td>2.86</td><td>84.41</td><td>3.77%</td><td>2.94</td><td>71.85</td></tr><tr><td>Step-Audio 2 mini</td><td>1.28%</td><td>3.64</td><td>79.16</td><td>1.18%</td><td>3.85</td><td>78.16</td><td>9.39%</td><td>3.48</td><td>83.59</td><td>7.49%</td><td>3.62</td><td>69.24</td></tr><tr><td>Vanilla-FT-ZH</td><td>0.91%</td><td>3.71</td><td>78.82</td><td>1.23%</td><td>3.99</td><td>77.47</td><td>2.03%</td><td>3.70</td><td>81.47</td><td>4.96%</td><td>3.77</td><td>69.88</td></tr><tr><td>Vanilla-FT-MIX</td><td>3.83%</td><td>3.67</td><td>79.10</td><td>0.80%</td><td>3.98</td><td>79.31</td><td>2.44%</td><td>3.73</td><td>82.10</td><td>7.02%</td><td>3.77</td><td>70.26</td></tr><tr><td>CoSTA-ZH</td><td>1.18%</td><td>3.70</td><td>80.56</td><td>1.06%</td><td>3.96</td><td>79.54</td><td>0.90%</td><td>3.71</td><td>81.09</td><td>1.51%</td><td>3.75</td><td>72.27</td></tr><tr><td>CoSTA-MIX</td><td>1.16%</td><td>3.70</td><td>78.99</td><td>0.77%</td><td>3.96</td><td>77.24</td><td>1.15%</td><td>3.65</td><td>82.39</td><td>1.65%</td><td>3.75</td><td>71.53</td></tr></table>

Table 14: Detailed quality results on Chinese speech-to-speech benchmarks. Each benchmark reports CER, UTMOS, and G-Eval for text-speech alignment, speech quality, and response-content quality, respectively.

<table><tr><td rowspan="2">Model</td><td colspan="3">AlpacaEval</td><td colspan="3">Gsm8kEval</td><td colspan="3">CommonEval</td><td colspan="3">Avg</td></tr><tr><td>WER↓</td><td>UTMOS↑ G-Eval↑</td><td></td><td>WER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td>WER↓</td><td>UTMOS↑ G-Eval↑</td><td></td><td>WER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td>Qwen2.5-Omni</td><td>2.05%</td><td>4.22</td><td>76.78</td><td>4.25%</td><td>4.32</td><td>54.30</td><td>1.48%</td><td>4.27</td><td>72.00</td><td>1.73%</td><td>4.28</td><td>72.88</td></tr><tr><td>Kimi-Audio</td><td>20.77%</td><td>2.53</td><td>62.01</td><td>99.08%</td><td>2.77</td><td>0.57</td><td>37.62%</td><td>2.95</td><td>50.77</td><td>34.92%</td><td>2.86</td><td>50.46</td></tr><tr><td>GLM-4-Voice</td><td>13.92%</td><td>4.00</td><td>80.87</td><td>8.29%</td><td>4.25</td><td>29.32</td><td>26.28%</td><td>4.10</td><td>66.77</td><td>11.55%</td><td>4.16</td><td>69.95</td></tr><tr><td>Step-Audio 2 mini</td><td>9.82%</td><td>4.51</td><td>76.31</td><td>17.07%</td><td>4.53</td><td>15.92</td><td>1.64%</td><td>4.53</td><td>52.53</td><td>6.49%</td><td>4.53</td><td>63.73</td></tr><tr><td>Vanilla-FT-EN</td><td>23.82%</td><td>4.35</td><td>73.74</td><td>4.15%</td><td>4.53</td><td>16.67</td><td>9.73%</td><td>4.47</td><td>72.03</td><td>10.16%</td><td>4.46</td><td>66.59</td></tr><tr><td>Vanilla-FT-MIX</td><td>27.61%</td><td>4.43</td><td>76.18</td><td>3.63%</td><td>4.54</td><td>11.05</td><td>9.89%</td><td>4.51</td><td>72.63</td><td>12.00%</td><td>4.49</td><td>65.45</td></tr><tr><td>CoSTA-EN</td><td>3.08%</td><td>4.54</td><td>81.34</td><td>3.01%</td><td>4.54</td><td>15.52</td><td>2.80%</td><td>4.55</td><td>75.87</td><td>2.50%</td><td>4.54</td><td>69.97</td></tr><tr><td>CoSTA-MIX</td><td>3.19%</td><td>4.54</td><td>83.32</td><td>3.26%</td><td>4.55</td><td>17.53</td><td>2.61%</td><td>4.55</td><td>73.47</td><td>2.64%</td><td>4.54</td><td>69.67</td></tr><tr><td rowspan="2">Model</td><td colspan="3">Summary</td><td colspan="3">MLC</td><td colspan="3">WildchatEval</td><td colspan="3">Avg</td></tr><tr><td>WER↓</td><td>UTMOS↑ G-Eval↑</td><td></td><td>WER↓</td><td>UTMOS↑ G-Eval↑</td><td></td><td>WER↓</td><td></td><td>UTMOS↑ G-Eval↑</td><td>WER↓</td><td></td><td>UTMOS↑ G-Eval↑</td></tr><tr><td>Qwen2.5-Omni</td><td>0.38%</td><td>4.34</td><td>80.06</td><td>1.02%</td><td>4.30</td><td>77.40</td><td>1.20%</td><td>4.24</td><td>76.71</td><td>1.73%</td><td>4.28</td><td>72.88</td></tr><tr><td>Kimi-Audio</td><td>3.43%</td><td>3.14</td><td>82.15</td><td>27.72%</td><td>3.17</td><td>40.87</td><td>20.91%</td><td>2.58</td><td>66.40</td><td>34.92%</td><td>2.86</td><td>50.46</td></tr><tr><td>GLM-4-Voice</td><td>1.39%</td><td>4.26</td><td>92.15</td><td>10.79%</td><td>4.23</td><td>69.30</td><td>8.61%</td><td>4.11</td><td>81.26</td><td>11.55%</td><td>4.16</td><td>69.95</td></tr><tr><td>Step-Audio 2 mini</td><td>0.28%</td><td>4.55</td><td>88.14</td><td>3.23%</td><td>4.53</td><td>72.69</td><td>6.86%</td><td>4.52</td><td>76.75</td><td>6.49%</td><td>4.53</td><td>63.73</td></tr><tr><td>Vanilla-FT-EN</td><td>4.18%</td><td>4.51</td><td>86.10</td><td>3.63%</td><td>4.53</td><td>72.50</td><td>15.45%</td><td>4.41</td><td>78.47</td><td>10.16%</td><td>4.46</td><td>66.59</td></tr><tr><td>Vanilla-FT-MIX</td><td>10.09%</td><td>4.49</td><td>85.37</td><td>3.96%</td><td>4.52</td><td>68.55</td><td>16.85%</td><td>4.46</td><td>78.93</td><td>12.00%</td><td>4.49</td><td>65.45</td></tr><tr><td>CoSTA-EN</td><td>1.34%</td><td>4.55</td><td>89.21</td><td>2.00%</td><td>4.53</td><td>74.01</td><td>2.76%</td><td>4.54</td><td>83.84</td><td>2.50%</td><td>4.54</td><td>69.97</td></tr><tr><td>CoSTA-MIX</td><td>1.25%</td><td>4.55</td><td>90.00</td><td>2.07%</td><td>4.53</td><td>71.19</td><td>3.45%</td><td>4.54</td><td>82.52</td><td>2.64%</td><td>4.54</td><td>69.67</td></tr></table>

Table 15: Detailed quality results on English speech-to-speech benchmarks. Each benchmark reports WER, UTMOS, and G-Eval for text-speech alignment, speech quality, and response-content quality, respectively.

<table><tr><td rowspan="2">Model</td><td rowspan="2">TTS Data</td><td colspan="4">AlpacaEval-zh</td><td colspan="4">APE-zh</td><td colspan="4">Avg</td></tr><tr><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td colspan="2">Step-Audio 2 mini</td><td>100.00%</td><td>8.48%</td><td>3.54</td><td>79.95</td><td>100.00%</td><td>4.28%</td><td>3.87</td><td>22.11</td><td>100.00%</td><td>8.57%</td><td>3.66</td><td>63.34</td></tr><tr><td rowspan="5">CoSTA-ZH</td><td>500h</td><td>68.17%</td><td>1.62%</td><td>3.70</td><td>80.86</td><td>65.32%</td><td>1.11%</td><td>4.06</td><td>21.05</td><td>66.48%</td><td>1.52%</td><td>3.79</td><td>66.35</td></tr><tr><td>400h</td><td>68.73%</td><td>1.69%</td><td>3.70</td><td>82.40</td><td>65.88%</td><td>1.48%</td><td>4.05</td><td>25.26</td><td>67.35%</td><td>1.62%</td><td>3.79</td><td>67.71</td></tr><tr><td>300h</td><td>68.54%</td><td>1.91%</td><td>3.72</td><td>81.59</td><td>65.09%</td><td>1.02%</td><td>4.06</td><td>25.09</td><td>66.83%</td><td>1.66%</td><td>3.78</td><td>67.75</td></tr><tr><td>200h</td><td>69.29%</td><td>2.15%</td><td>3.68</td><td>82.36</td><td>65.69%</td><td>0.83%</td><td>4.07</td><td>26.84</td><td>67.24%</td><td>1.71%</td><td>3.78</td><td>67.99</td></tr><tr><td>100h</td><td>68.78%</td><td>1.75%</td><td>3.73</td><td>82.86</td><td>65.50%</td><td>1.55%</td><td>4.03</td><td>22.11</td><td>67.39%</td><td>1.80%</td><td>3.78</td><td>66.88</td></tr><tr><td rowspan="2">Model</td><td>TTS Data</td><td colspan="4">Claude-zh</td><td colspan="4">MLC-zh</td><td colspan="4">Avg</td></tr><tr><td></td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑ G-Eval↑</td><td></td><td></td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td></tr><tr><td rowspan="2">Step-Audio 2 mini</td><td></td><td>100.00%</td><td>20.21%</td><td>3.36</td><td>73.15</td><td>100.00%</td><td>1.30%</td><td>3.85</td><td>78.16</td><td>100.00%</td><td>8.57%</td><td>3.66</td><td>63.34</td></tr><tr><td>500h</td><td>69.35%</td><td>2.39%</td><td>3.41</td><td>84.86</td><td>63.09%</td><td>0.95%</td><td>3.99</td><td>78.62</td><td>66.48%</td><td>1.52%</td><td>3.79</td><td>66.35</td></tr><tr><td rowspan="4">CoSTA-ZH</td><td>400h</td><td>70.07%</td><td>2.42%</td><td>3.42</td><td>85.47</td><td>64.70%</td><td>0.88%</td><td>3.99</td><td>77.70</td><td>67.35%</td><td>1.62%</td><td>3.79</td><td>67.71</td></tr><tr><td>300h</td><td>69.77%</td><td>2.98%</td><td>3.39</td><td>84.77</td><td>63.93%</td><td>0.71%</td><td>3.93</td><td>79.54</td><td>66.83%</td><td>1.66%</td><td>3.78</td><td>67.75</td></tr><tr><td>200h</td><td>70.12%</td><td>2.84%</td><td>3.40</td><td>85.53</td><td>63.85%</td><td>1.02%</td><td>3.97</td><td>77.24</td><td>67.24%</td><td>1.71%</td><td>3.78</td><td>67.99</td></tr><tr><td>100h</td><td>70.38%</td><td>2.69%</td><td>3.39</td><td>85.77</td><td>64.90%</td><td>1.22%</td><td>3.95</td><td>76.78</td><td>67.39%</td><td>1.80%</td><td>3.78</td><td>66.88</td></tr></table>

Table 16: Effect of pronunciation-adaptation data scale on Chinese benchmarks. Dialogue-adaptation data is fixed to 50K examples. $\rho _ { g }$ denotes the generated-token compression rate, while ASR-CER, UTMOS, and G-Eval evaluate text-speech alignment, speech quality, and response-content quality, respectively.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dialogue Data</td><td colspan="4">AlpacaEval-zh</td><td colspan="4">APE-zh</td><td colspan="4"> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td></tr><tr><td rowspan="2">Step-Audio 2 mini</td><td></td><td>100.00%</td><td>8.48%</td><td>3.54</td><td>79.95</td><td>100.00%</td><td>4.28%</td><td>3.87</td><td>22.11</td><td>100.00%</td><td>8.57%</td><td>3.66</td><td>63.34</td></tr><tr><td>50K</td><td>68.54%</td><td>1.91%</td><td>3.72</td><td>81.59</td><td>65.09%</td><td>1.02%</td><td>4.06</td><td>25.09</td><td>66.83%</td><td>1.66%</td><td>3.78</td><td>67.75</td></tr><tr><td rowspan="4">CoSTA-ZH</td><td>40K</td><td>69.24%</td><td>1.79%</td><td>3.71</td><td>78.59</td><td>65.61%</td><td>1.36%</td><td>4.04</td><td>22.81</td><td>67.28%</td><td>1.72%</td><td>3.78</td><td>66.02</td></tr><tr><td>30K</td><td>68.82%</td><td>1.90%</td><td>3.73</td><td>79.23</td><td>66.04%</td><td>1.00%</td><td>4.04</td><td>18.07</td><td>67.50%</td><td>1.80%</td><td>3.78</td><td>64.95</td></tr><tr><td>20K</td><td>68.32%</td><td>2.02%</td><td>3.70</td><td>80.45</td><td>66.20%</td><td>0.95%</td><td>4.06</td><td>25.44</td><td>67.49%</td><td>1.97%</td><td>3.79</td><td>66.84</td></tr><tr><td>10K</td><td>68.82%</td><td>1.31%</td><td>3.75</td><td>82.59</td><td>68.10%</td><td>0.67%</td><td>4.02</td><td>24.56</td><td>68.19%</td><td>2.09%</td><td>3.80</td><td>67.25</td></tr><tr><td rowspan="2">Model</td><td rowspan="2">Dialogue Data</td><td colspan="4">Claude-zh</td><td colspan="4">MLC-zh</td><td colspan="4">Avg</td></tr><tr><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td rowspan="2">Step-Audio 2 mini</td><td></td><td>100.00%</td><td>20.21%</td><td>3.36</td><td>73.15</td><td>100.00%</td><td>1.30%</td><td>3.85</td><td>78.16</td><td>100.00%</td><td>8.57%</td><td>3.66</td><td>63.34</td></tr><tr><td>50K</td><td>69.77%</td><td>2.98%</td><td>3.39</td><td>84.77</td><td>63.93%</td><td>0.71%</td><td>3.93</td><td>79.54</td><td>66.83%</td><td>1.66%</td><td>3.78</td><td>67.75</td></tr><tr><td rowspan="4">CoSTA-ZH</td><td>40K</td><td>69.99%</td><td>2.68%</td><td>3.41</td><td>86.13</td><td>64.28%</td><td>1.07%</td><td>3.97</td><td>76.55</td><td>67.28%</td><td>1.72%</td><td>3.78</td><td>66.02</td></tr><tr><td>30K</td><td>70.25%</td><td>2.90%</td><td>3.38</td><td>84.80</td><td>64.88%</td><td>1.39%</td><td>3.98</td><td>77.70</td><td>67.50%</td><td>1.80%</td><td>3.78</td><td>64.95</td></tr><tr><td>20K</td><td>70.19%</td><td>2.93%</td><td>3.45</td><td>86.05</td><td>65.25%</td><td>1.97%</td><td>3.95</td><td>75.40</td><td>67.49%</td><td>1.97%</td><td>3.79</td><td>66.84</td></tr><tr><td>10K</td><td>70.30%</td><td>2.50%</td><td>3.50</td><td>84.59</td><td>65.55%</td><td>3.86%</td><td>3.95</td><td>77.24</td><td>68.19%</td><td>2.09%</td><td>3.80</td><td>67.25</td></tr></table>

Table 17: Effect of dialogue-adaptation data scale on Chinese benchmarks. Pronunciation-adaptation data is fixed to 300h. $\rho _ { g }$ denotes the generated-token compression rate, while ASR-CER, UTMOS, and G-Eval evaluate text-speech alignment, speech quality, and response-content quality, respectively.

<table><tr><td>Model</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑</td><td>Prefill (ms)↓</td><td>Decode (ms)↓</td><td>LLM Total (ms)↓</td><td>Speech Decoder (ms)↓</td><td>Total (ms)↓</td></tr><tr><td>Step-Audio 2 mini</td><td>100.00%</td><td>58.74</td><td>56.02</td><td>164.42</td><td>220.44</td><td>337.46</td><td>557.90</td></tr><tr><td>Vanilla-FT-ZH</td><td>100.00%</td><td>58.56</td><td>55.37</td><td>164.22</td><td>219.59</td><td>340.51</td><td>560.10</td></tr><tr><td>CoSTA-ZH</td><td>66.81%</td><td>101.99</td><td>54.55</td><td>122.98</td><td>177.53</td><td>328.06</td><td>505.59</td></tr><tr><td>Step-Audio 2 mini</td><td>100.00%</td><td>59.69</td><td>53.07</td><td>165.08</td><td>218.15</td><td>344.44</td><td>562.60</td></tr><tr><td>Vanilla-FT-EN</td><td>100.00%</td><td>57.67</td><td>55.72</td><td>165.10</td><td>220.82</td><td>339.52</td><td>560.34</td></tr><tr><td>CoSTA-EN</td><td>70.17%</td><td>99.05</td><td>53.24</td><td>115.96</td><td>169.20</td><td>339.17</td><td>508.36</td></tr></table>

Table 18: First-packet latency breakdown for Chinese and English settings. The upper and lower blocks report Chinese and English results, respectively.

<table><tr><td>Metric</td><td>Model</td><td>AlpacaEval-zh APE-zh</td><td></td><td></td><td>Claude-zh LCSTS-zh MLC-zh</td><td></td><td>Wildchat-zh</td><td>Avg.</td></tr><tr><td rowspan="3">Speech Fluency</td><td>Step-Audio 2 mini</td><td>4.70</td><td>4.77</td><td>3.77</td><td>4.47</td><td>4.83</td><td>4.37</td><td>4.48</td></tr><tr><td>Vanilla-FT-ZH</td><td>3.80</td><td>4.50</td><td>3.60</td><td>4.83</td><td>4.60</td><td>4.77</td><td>4.35</td></tr><tr><td>CoSTA-ZH</td><td>4.73</td><td>4.33</td><td>4.60</td><td>4.67</td><td>4.70</td><td>4.73</td><td>4.63</td></tr><tr><td rowspan="3">Semantic Quality</td><td>Step-Audio 2 mini</td><td>4.17</td><td>2.23</td><td>3.17</td><td>3.60</td><td>3.83</td><td>4.10</td><td>3.52</td></tr><tr><td>Vanilla-FT-ZH</td><td>4.00</td><td>1.83</td><td>3.43</td><td>3.80</td><td>3.70</td><td>4.23</td><td>3.50</td></tr><tr><td>CoSTA-ZH</td><td>4.40</td><td>2.17</td><td>3.60</td><td>4.10</td><td>4.00</td><td>4.23</td><td>3.75</td></tr></table>

Table 19: MOS-style human evaluation on Chinese URO-Bench datasets. Scores are averaged over six evaluators and five randomly sampled queries per dataset. Higher scores indicate better performance.

<table><tr><td>Metric</td><td>Model</td><td></td><td>AlpacaEval Gsm8kEval CommonEval Summary</td><td></td><td></td><td>MLC</td><td>WildchatEval</td><td>|Avg.</td></tr><tr><td rowspan="4">Speech Fluency</td><td>Step-Audio 2 mini</td><td>4.23</td><td>4.03</td><td>4.73</td><td>4.70</td><td>4.50</td><td>4.80</td><td>4.50</td></tr><tr><td>Vanilla-FT-EN</td><td>3.53</td><td>4.63</td><td>4.23</td><td>4.77</td><td>4.30</td><td>4.37</td><td>4.31</td></tr><tr><td>CoSTA-EN</td><td>4.80</td><td>4.70</td><td>4.77</td><td>4.87</td><td>4.83</td><td>4.73</td><td>4.78</td></tr><tr><td>Step-Audio 2 mini</td><td>3.50</td><td>2.33</td><td>1.23</td><td>4.57</td><td>4.20</td><td>4.27</td><td>3.35</td></tr><tr><td rowspan="3">Semantic Quality</td><td>Vanilla-FT-EN</td><td>3.13</td><td>1.77</td><td>4.00</td><td>4.40</td><td>4.80</td><td>4.27</td><td>3.73</td></tr><tr><td>CoSTA-EN</td><td>4.37</td><td>1.20</td><td>4.17</td><td>4.90</td><td>4.83</td><td>4.73</td><td>4.03</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 20: MOS-style human evaluation on English URO-Bench datasets. Scores are averaged over six evaluators and five randomly sampled queries per dataset. Higher scores indicate better performance.

<table><tr><td rowspan="2">Model</td><td colspan="4">AlpacaEval-zh</td><td colspan="4">Claude-zh</td><td colspan="4">LCSTS-zh</td><td colspan="4">Avg</td></tr><tr><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓ UTMOS↑</td><td></td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$  ASR-CER↓</td><td></td><td>UTMOS↑</td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-CER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td>OpenS2S</td><td>100.00%</td><td>9.02%</td><td>2.54</td><td>69.71</td><td>100.00%</td><td>14.40%</td><td>2.18</td><td>66.97</td><td>100.00%</td><td>9.82%</td><td>2.51</td><td>59.10</td><td>100.00%</td><td>11.08%</td><td>2.41</td><td>65.26</td></tr><tr><td>Vanilla-FT-ZH</td><td>100.00%</td><td>6.26%</td><td>2.49</td><td>72.29</td><td>100.00%</td><td>20.00%</td><td>2.14</td><td>66.64</td><td>100.00%</td><td>7.81%</td><td>2.50</td><td>63.25</td><td>100.00%</td><td>11.36%</td><td>2.38</td><td>67.39</td></tr><tr><td>CoSTA-ZH</td><td>78.34%</td><td>5.98%</td><td>2.52</td><td>70.61</td><td>79.92%</td><td>11.77%</td><td>2.08</td><td>71.65</td><td>78.23%</td><td>7.75%</td><td>2.49</td><td>65.27</td><td>78.83%</td><td>8.50%</td><td>2.36</td><td>69.18</td></tr><tr><td rowspan="2">Model</td><td colspan="4">AlpacaEval</td><td colspan="4">CommonEval</td><td colspan="4">Summary</td><td colspan="4"> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td> $\rho _ { g } \downarrow$ </td><td>ASR-WER↓ UTMOS↑</td><td></td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-WER↓ UTMOS↑</td><td></td><td>G-Eval↑</td><td> $\rho _ { g } \downarrow$ </td><td>ASR-WER↓ UTMOS↑ G-Eval↑</td><td></td><td></td><td> $\rho _ { g } \downarrow$ </td><td>ASR-WER↓</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td>OpenS2S</td><td>100.00%</td><td>6.96%</td><td>3.72</td><td>70.65</td><td>100.00%</td><td>4.92%</td><td>3.85</td><td>63.47</td><td>100.00%</td><td>2.07%</td><td>3.87</td><td>87.80</td><td>100.00%</td><td>4.65%</td><td>3.81</td><td>73.97</td></tr><tr><td>Vanilla-FT-EN</td><td>100.00%</td><td>8.44%</td><td>3.67</td><td>72.16</td><td>100.00%</td><td>4.67%</td><td>3.84</td><td>66.27</td><td>100.00%</td><td>2.36%</td><td></td><td>88.02</td><td>100.00%</td><td>5.16%</td><td>3.79</td><td>75.48</td></tr><tr><td>CoSTA-EN</td><td>82.37%</td><td>6.35%</td><td>3.65</td><td>74.47</td><td>80.87%</td><td>5.18%</td><td>3.84</td><td>66.53</td><td>80.83%</td><td>2.47%</td><td>3.86 3.85</td><td>87.68</td><td>81.36%</td><td>4.66%</td><td>3.78</td><td>76.23</td></tr></table>

Table 21: Detailed SDM-level generalization results on OpenS2S. CoSTA is compared with the original OpenS2S model and matched vanilla fine-tuning on Chinese and English benchmarks. $\rho _ { g }$ denotes the generated-token compression rate, while ASR-CER (Chinese) / ASR-WER (English), UTMOS, and G-Eval evaluate text-speech alignment, speech quality, and response-content quality, respectively.

(a) Token usage and BPE coverage
<table><tr><td>Model</td><td>Original</td><td>BPE</td><td>Original/BPE</td><td>BPE coverage</td><td>Avg. expansion</td></tr><tr><td>CoSTA-ZH</td><td>70.95%</td><td>29.05%</td><td>2.44</td><td>51.64%</td><td>2.608</td></tr><tr><td>w/o Dialogue Adaptation</td><td>78.95%</td><td>21.05%</td><td>3.75</td><td>36.71%</td><td>2.175</td></tr></table>

(b) Distribution of token expansion lengths
<table><tr><td>Model</td><td> $L = 1$ </td><td> $L = 2$ </td><td> $L = 3$ </td><td> $L = 4$ </td><td> $5 \leq L \leq 8$ </td><td> $L \geq 9$ </td></tr><tr><td>CoSTA-ZH</td><td>70.95%</td><td>24.28%</td><td>1.69%</td><td>0.79%</td><td>1.36%</td><td>0.93%</td></tr><tr><td>w/o Dialogue Adaptation</td><td>78.95%</td><td>19.22%</td><td>1.12%</td><td>0.27%</td><td>0.40%</td><td>0.05%</td></tr></table>

Table 22: Speech-token usage with and without dialogue adaptation on all six Chinese benchmarks (1,122 responses per model). Statistics are pooled over generated speech tokens. (a) Original/BPE shares and their ratio use counts before expansion; BPE coverage uses counts after expansion. Avg. expansion is the number of original tokens per generated BPE unit. (b) L is the expansion length, with L = 1 denoting an original token. Each percentage uses all generated speech tokens before expansion as its denominator.

<table><tr><td>Model</td><td>Size</td><td> $\rho _ { g } \downarrow$ </td><td>Speed↑</td><td>Relative↑</td><td>UTMOS↑</td><td>G-Eval↑</td></tr><tr><td>Step-Audio 2 mini</td><td>OK</td><td>100.00%</td><td>55.51</td><td>100.00%</td><td>3.58</td><td>80.18</td></tr><tr><td rowspan="5">CoSTA-ZH</td><td>10K</td><td>76.66%</td><td>78.67</td><td>141.71%</td><td>3.73</td><td>80.86</td></tr><tr><td>20K</td><td>74.33%</td><td>84.51</td><td>152.24%</td><td>3.71</td><td>80.73</td></tr><tr><td>40K</td><td>71.25%</td><td>93.20</td><td>167.89%</td><td>3.69</td><td>83.35</td></tr><tr><td>80K</td><td>67.50%</td><td>103.03</td><td>185.59%</td><td>3.70</td><td>81.36</td></tr><tr><td>160K</td><td>63.62%</td><td>104.18</td><td>187.65%</td><td>3.68</td><td>78.87</td></tr></table>

Table 23: Complete vocabulary-size results on AlpacaEval-zh underlying Figure 7. Size is the number of added BPE units; Speed is expanded speech tokens/s, normalized to the original Step-Audio 2 mini in Relative. The 0K row is the unadapted baseline.

<table><tr><td>Added units</td><td>Compression↓</td><td>Median ms/step↓</td><td>Overhead↓</td><td>Norm. Cost↓</td></tr><tr><td>0K</td><td>100.00%</td><td>0.6902</td><td>+0.00%</td><td>1.000</td></tr><tr><td>10K</td><td>73.27%</td><td>0.7320</td><td>+6.06%</td><td>0.777</td></tr><tr><td>20K</td><td>69.79%</td><td>0.7739</td><td>+12.13%</td><td>0.783</td></tr><tr><td>40K</td><td>66.15%</td><td>0.8576</td><td>+24.25%</td><td>0.822</td></tr><tr><td>80K</td><td>62.29%</td><td>1.0251</td><td>+48.52%</td><td>0.925</td></tr><tr><td>160K</td><td>58.48%</td><td>1.3598</td><td>+97.01%</td><td>1.152</td></tr><tr><td>320K</td><td>54.94%</td><td>2.0297</td><td>+194.08%</td><td>1.616</td></tr></table>

Table 24: Complete corpus-compression and isolated LM-head projection plus greedy-argmax costs for 0K–320K added BPE units, measured on a single A100 GPU with identical BF16 hidden states at batch size 1. Compression uses held-out Dialogue-ZH token counts. Derived costs use unrounded compression rates. Overhead is per-step latency relative to 0K; Norm. Cost is $\rho _ { M } t _ { M } / t _ { 0 }$ for a fixed expanded token count (Equation 9).

<table><tr><td>Tokenizer</td><td>Split</td><td>α (%)β (pp/doubling)</td><td></td><td> $R ^ { 2 }$ </td><td>RMSE (pp)</td><td>Max |e| (pp)</td></tr><tr><td rowspan="2">GLM-4-Voice Tokenizer 12.5Hz</td><td>Train</td><td>84.884</td><td>-4.808</td><td>0.9959</td><td>0.528</td><td>0.784</td></tr><tr><td>Test</td><td>84.866</td><td>-4.193</td><td>0.9989</td><td>0.242</td><td>0.386</td></tr><tr><td rowspan="2">S3Tokenizer V2 25Hz</td><td>Train</td><td>73.550</td><td>-4.013</td><td>0.9988</td><td>0.241</td><td>0.370</td></tr><tr><td>Test</td><td>73.397</td><td>-3.698</td><td>0.9997</td><td>0.103</td><td>0.149</td></tr><tr><td rowspan="2">S3Tokenizer V3 25Hz</td><td>Train</td><td>71.678</td><td>-4.107</td><td>0.9999</td><td>0.085</td><td>0.138</td></tr><tr><td>Test</td><td>71.569</td><td>-3.809</td><td>0.9991</td><td>0.191</td><td>0.309</td></tr><tr><td rowspan="2">Mimi</td><td>Train</td><td>60.365</td><td>-4.023</td><td>0.9968</td><td>0.389</td><td>0.565</td></tr><tr><td>Test</td><td>60.217</td><td>-3.539</td><td>0.9852</td><td>0.740</td><td>1.077</td></tr><tr><td rowspan="2">S3Tokenizer V1 25Hz</td><td>Train</td><td>57.479</td><td>-3.922</td><td>0.9939</td><td>0.526</td><td>0.741</td></tr><tr><td>Test</td><td>57.488</td><td>-3.664</td><td>0.9864</td><td>0.734</td><td>1.001</td></tr><tr><td rowspan="2">Qwen3-TTS-Tokenizer-12Hz</td><td>Train</td><td>57.064</td><td>-4.147</td><td>0.9947</td><td>0.516</td><td>0.766</td></tr><tr><td>Test</td><td>57.018</td><td>-3.684</td><td>0.9807</td><td>0.883</td><td>1.212</td></tr><tr><td rowspan="2">S3Tokenizer V1 50Hz</td><td>Train</td><td>52.896</td><td>-3.652</td><td>0.9920</td><td>0.559</td><td>0.774</td></tr><tr><td>Test</td><td>52.812</td><td>-3.501</td><td>0.9874</td><td>0.675</td><td>0.928</td></tr><tr><td rowspan="2">mHuBERT Base L11 KM1000</td><td>Train</td><td>38.288</td><td>-2.878</td><td>0.9902</td><td>0.490</td><td>0.712</td></tr><tr><td>Test</td><td>38.425</td><td>-2.699</td><td>0.9812</td><td>0.637</td><td>0.875</td></tr></table>

Table 25: Complete training and test fits for the eight tokenizer configurations on Chinese dialogue speech. Full names, configurations, and public reference sources are given in Table 12. Fits use $y = \alpha + \beta \log _ { 2 } ( M / 1 0 \mathrm { K } )$ , where $y = 1 0 0 \rho$ . Each row fits six vocabulary settings; errors are in percentage points.

<table><tr><td>Setting</td><td>Split</td><td>α (%)</td><td>β(pp/doubling)</td><td> $R ^ { 2 }$ </td><td>RMSE (pp)</td><td>Max |e| (pp)</td></tr><tr><td rowspan="2">Emilia-MIX</td><td>Train</td><td>85.902</td><td>-3.653</td><td>0.9929</td><td>0.529</td><td>0.752</td></tr><tr><td>Test</td><td>85.664</td><td>-3.621</td><td>0.9942</td><td>0.474</td><td>0.670</td></tr><tr><td rowspan="2">Emilia-EN</td><td>Train</td><td>85.513</td><td>-3.706</td><td>0.9942</td><td>0.485</td><td>0.693</td></tr><tr><td>Test</td><td>85.127</td><td>-3.680</td><td>0.9957</td><td>0.414</td><td>0.605</td></tr><tr><td rowspan="2">Emilia-ZH</td><td>Train</td><td>84.656</td><td>-3.751</td><td>0.9947</td><td>0.468</td><td>0.676</td></tr><tr><td>Test</td><td>84.464</td><td>-3.681</td><td>0.9954</td><td>0.426</td><td>0.618</td></tr><tr><td rowspan="2">Emilia-TC-MIX</td><td>Train</td><td>83.130</td><td>-3.768</td><td>0.9966</td><td>0.377</td><td>0.560</td></tr><tr><td>Test</td><td>82.889</td><td>-3.694</td><td>0.9965</td><td>0.375</td><td>0.551</td></tr><tr><td rowspan="2">Emilia-TC-EN</td><td>Train</td><td>82.556</td><td>-3.751</td><td>0.9973</td><td>0.334</td><td>0.496</td></tr><tr><td>Test</td><td>82.207</td><td>-3.669</td><td>0.9972</td><td>0.334</td><td>0.481</td></tr><tr><td rowspan="2">Emilia-TC-ZH</td><td>Train</td><td>81.824</td><td>-3.897</td><td>0.9977</td><td>0.322</td><td>0.494</td></tr><tr><td>Test</td><td>81.605</td><td>-3.815</td><td>0.9977</td><td>0.316</td><td>0.460</td></tr><tr><td rowspan="2">Dialogue-MIX</td><td>Train</td><td>77.096</td><td>-4.036</td><td>0.9991</td><td>0.211</td><td>0.336</td></tr><tr><td>Test</td><td>77.138</td><td>-3.908</td><td>0.9996</td><td>0.136</td><td>0.212</td></tr><tr><td rowspan="2">Dialogue-EN</td><td>Train</td><td>76.790</td><td>-4.242</td><td>0.9998</td><td>0.103</td><td>0.170</td></tr><tr><td>Test</td><td>76.621</td><td>-4.012</td><td>0.9996</td><td>0.142</td><td>0.222</td></tr><tr><td rowspan="2">Dialogue-ZH</td><td>Train</td><td>73.550</td><td>-4.013</td><td>0.9988</td><td>0.241</td><td>0.370</td></tr><tr><td>Test</td><td>73.400</td><td>-3.699</td><td>0.9997</td><td>0.104</td><td>0.148</td></tr></table>

Table 26: Complete training and test fits across data sources and languages with S3Tokenizer V2. Fits use $y = \alpha + \beta \log _ { 2 } ( M / 1 0 \mathrm { K } )$ , where $y = 1 0 0 \rho$ . Each row fits six vocabulary settings; errors are in percentage points.

## Case Study 1: Short Factual QA

## Input: What breed ofdog is the smallest?

## Step-Audio 2 mini

Text Content: The Chihuahua is considered one of the smallest dog breeds, though there are other tiny dogs like the Yorkshire Terrier and Pomeranian that can be even smaller than some Chihuahuas.

## Speech Tokens:

[1475, 1978, 4218, 4137, 2031, 2004, 72, 876, 2490, 5322, 3111, 3887, 2183, 1456, 710, 3404, 5591, 5026, 5830, 5092,   
5023, 4699, 1490, 1763, 4769, 4450, 5644, 1623, 168, 12, 3908, 3910, 5645, 4673, 2217, 4526, 138, 3151, 3872, 1686,   
1854, 2185, 2915, 5024, 3744, 443, 4275, 2085, 315, 65, 4672, 4916, 4915, 4914, 2124, 6501, 2849, 5597, 4958, 5074,   
4909, 4492, 4411, 4753, 3459, 1950, 2112, 60, 2333, 2759, 2681, 4949, 5078, 1128, 2076, 2166, 2160, 4367, 1685, 962,   
395, 557, 79, 1833, 6379, 6374, 5401, 27, 1458, 1707, 3648, 3645, 1701, 1704, 1947, 3891, 3894, 1950, 1959, 1914,   
306, 2243, 2810, 2912, 5828, 1044, 4194, 4442, 2384, 3107, 4106, 1892, 1865, 4079, 1919, 2003, 1976, 6320, 4779, 543,   
4528, 6059, 1687, 303, 2214, 6075, 5599, 5189, 2285, 3666, 2726, 2753, 1599, 66, 461, 494, 1223, 1220, 704, 1209, 81,   
5482, 6372, 6475, 2702, 3002, 475, 2139, 63, 80, 1294, 3480, 3884, 5090, 5806, 4114, 732, 6048, 5319, 3141, 3875,   
1680, 2247, 2274, 2922, 2366, 5294, 6293, 4115, 3158, 3401, 959, 1928, 1676, 1520, 737, 22, 3667, 2175, 1431, 5589.   
5598, 5671, 2845, 4311, 2135, 2171, 2180, 1652, 4553, 5047, 2373, 2938, 2699, 539, 125, 5, 1476, 1480, 1723, 1716,   
2040, 1272, 2216, 197, 1059, 2355, 25, 6178, 2172, 969, 4939, 2005, 2014, 5019, 4858, 4347, 2315, 3664, 1726, 6377,   
5402, 4918, 1170, 4314, 662, 5678, 5025, 5062, 2750, 380, 1688, 884, 1281, 74, 20, 3913, 6376, 5648, 4671, 4619, 658,   
4072, 1041, 4593, 2382, 1700, 2186, 1457, 575, 4133, 6077, 2894, 4372, 719, 515, 29, 794, 4428, 5726, 5645, 2484,   
1702, 6405, 4218, 4218, 4218]

Length: 267 Speech Tokens

## Vanilla-FT-EN

Text Content: The smallest breed of dog is the Chihuahua.

## Speech Tokens:

[3704, 4299, 6486, 6405, 4218, 1950, 1560, 801, 2246, 4428, 6377, 1035, 2130, 2125, 4949, 4945, 5029, 5059, 5985,   
4409, 4733, 4444, 6454, 4191, 2232, 1188, 2166, 2169, 4367, 3872, 2420, 5102, 231, 70, 1760, 3404, 4868, 2894, 6534,   
2112, 60, 2423, 5051, 4946, 494, 491, 515, 1125, 84, 168, 1707, 5838, 5109, 2922, 979, 1951, 386, 4841, 4454, 5409,   
2085, 72, 2327, 150, 2733, 5326, 3111, 3806, 2102, 2185, 710, 6077, 6157, 5050, 3643, 707, 3161, 2918, 5832, 1701,   
4299, 4299, 4299, 4299, 6486, 6486, 6486, 4299, 4299, 4299, 6486, 6486, 6405, 4218, 4218, 4218, 4299, 6159]

Length: 98 Speech Tokens

## CoSTA-EN

Text Content: The smallest dog breed is the chihuahua.

## Speech Tokens:

[3704, 7192, 6725, 6676, 1959, 3831, 45, 4442, 6792, 1035, 14382, 3410, 9992, 5068, 6148, 4459, 4498, 4429, 6979, 2209, 795, 7411, 353, 4946, 13225, 4770, 975, 6705, 2169, 9941, 5300, 2627, 11198, 484, 786, 7553, 17346, 4454, 8462, 3225, 140, 2336, 1356, 17640, 1072, 626, 1456, 3643, 3488, 25896, 5062, 5830, 2786, 17267, 974, 2927, 5832, 6669, 13230, 22981, 9416]

## Expanded Speech Tokens:

[3704, 4299, 6486, 6405, 4218, 4218, 4218, 6405, 6405, 1959, 3831, 45, 4442, 6373, 6377, 1035, 2130, 2125, 3410,   
4865, 4942, 5068, 6148, 4459, 4498, 4429, 2139, 4431, 2209, 795, 2139, 66, 353, 4946, 5675, 4942, 4770, 975, 4299,   
2166, 2169, 4358, 6050, 5300, 2627, 638, 566, 484, 786, 1763, 1951, 2573, 4760, 4454, 4429, 4272, 3225, 140, 2336,   
1356, 5326, 3840, 1072, 626, 1456, 3643, 3488, 6157, 5508, 5062, 5830, 2786, 3161, 974, 974, 2927, 5832, 4299, 6486,   
6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 6405, 6405, 6405, 6405, 6405, 6405, 4218, 4218, 4218, 4218,   
4218, 4299, 6078]

Length: 61 BPE Speech Tokens / 101 Expanded Speech Tokens

Compression Rate: 60.40%

## Case Study 2: Simple Commonsense QA

## Input: What color is the sky?

## Step-Audio 2 mini

Text Content: The sky is blue.

## Speech Tokens:

[1499, 4218, 6486, 4299, 2112, 1029, 74, 2261, 6455, 6374, 1890, 1788, 183, 2270, 5591, 5348, 5186, 4543, 4821, 5011, 1978, 1034, 4517, 5179, 6382, 1044, 2175, 2166, 721, 6475, 4690, 4613, 3155, 884, 2099, 2102, 2186, 2149, 4299, 6486, 6405, 4218]

## Length: 42 Speech Tokens

## Vanilla-FT-EN

Text Content: The sky is blue.

## Speech Tokens:

[1517, 4299, 6486, 6405, 4218, 4218, 4137, 1950, 3667, 1563, 73, 4514, 4441, 6455, 6373, 2106, 2031, 174, 2459, 4865, 4871, 4951, 4536, 4813, 1250, 1951, 557, 5093, 4444, 6454, 3456, 2112, 2160, 1208, 5988, 2263, 4606, 4531, 5420, 1775, 2018, 2099, 4286, 2021, 2140, 4299, 4299, 4299, 4299, 4299, 4299, 6486, 6486, 6486, 6486, 6486, 6405, 4218, 6162]

Length: 59 Speech Tokens

## CoSTA-EN

Text Content: The sky is blue.

## Speech Tokens:

[3704, 9189, 6657, 1764, 2990, 4447, 7478, 3456, 7246, 4514, 5348, 10357, 5269, 4813, 7115, 4858, 4449, 6390, 1776, 8886, 1450, 6476, 5635, 4771, 4772, 2585, 398, 11408, 2096, 6734, 6672, 43408]

## Expanded Speech Tokens:

[3704, 4299, 4299, 4218, 4218, 4218, 4299, 4299, 1764, 2990, 4447, 6455, 6380, 3456, 2112, 84, 4514, 5348, 6320, 6158, 5269, 4813, 1277, 2006, 4858, 4449, 6390, 1776, 2148, 2166, 1450, 6476, 5635, 4771, 4772, 2585, 398, 1370, 2102, 2096, 4299, 4299, 4299, 4299, 4299, 4299, 6486, 6486, 6486, 4299, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4299, 6078]

Length: 32 BPE Speech Tokens / 60 Expanded Speech Tokens

Compression Rate: 53.33%

## Case Study 3: Sentence Rewriting

Input: The order in which you add vectors does not matter. Please rewrite in simple and same meaning.

## Step-Audio 2 mini

Text Content: The order of adding vectors doesn’t matter.

## Speech Tokens:

[1499, 4218, 6486, 4299, 2112, 57, 2510, 566, 1763, 2032, 1358, 2906, 5102, 5815, 6058, 1518, 6067, 6059, 5329, 6050,   
1760, 1001, 4970, 4275, 6453, 1517, 1951, 3650, 5834, 3647, 5837, 138, 4445, 5047, 6181, 4235, 4284, 6534, 110, 5918,   
4570, 1788, 1869, 2247, 3105, 3098, 1685, 1688, 1697, 5912, 4671, 6374, 6373, 4245, 4299, 2085, 2253, 2, 2702, 5653,   
5652, 4438, 5853, 1744, 1803, 4068, 5105, 5194, 5508, 5185, 867, 3146, 1676, 1679, 3872, 3847, 3729, 6486, 6486,   
6405, 3975]

Length: 81 Speech Tokens

## Vanilla-FT-EN

Text Content: When you add vectors, it doesn’t matter which one comes first.

## Speech Tokens:

[3704, 4299, 6486, 6405, 4218, 4218, 4218, 2121, 4316, 4346, 3644, 2567, 4379, 4471, 4966, 4885, 3482, 4130, 6071,   
1517, 1978, 764, 4463, 4382, 5111, 4459, 57, 2417, 2152, 2074, 2097, 2219, 5191, 4536, 2112, 162, 1113, 4431, 3012,   
3116, 1676, 1679, 1688, 1616, 5911, 4672, 6134, 762, 4056, 3975, 4299, 4299, 1124, 5102, 4697, 1536, 2049, 1770, 59.   
3650, 4456, 5412, 5418, 4494, 5937, 2733, 132, 4074, 1639, 2195, 5108, 4540, 138, 3824, 1658, 1685, 1928, 1454, 2186,   
2185, 2912, 2408, 4849, 438, 4758, 3864, 1380, 4344, 5101, 5029, 4862, 11, 50, 1726, 1805, 1833, 165, 5346, 5356,   
5591, 666, 6498, 5409, 6374, 2916, 6537, 6534, 5780, 5320, 3863, 1685, 1619, 4509, 6376, 2139, 2247, 2193, 736, 4299,   
4299, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 4299, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4299, 6159]

Length: 138 Speech Tokens

## CoSTA-EN

Text Content: The order doesn’t matter when you add vectors.

## Speech Tokens:

[3704, 9189, 6676, 1725, 14093, 137, 8715, 2789, 2708, 4976, 4842, 297, 4601, 6864, 13239, 5189, 5348, 14410, 4439, 13344, 27059, 11544, 189, 2332, 7868, 2171, 18623, 110, 2276, 4643, 3454, 4129, 4367, 9321, 5435, 5920, 5108, 57, 1855, 2073, 1854, 73052, 11434, 7732, 3078, 3116, 11361, 3725, 4683, 4673, 7404, 1731, 6678, 20923, 6078]

## Expanded Speech Tokens:

[3704, 4299, 4299, 4218, 4218, 4218, 6405, 6405, 1725, 1941, 63, 137, 2003, 2006, 2789, 2708, 4976, 4842, 297, 4601,   
6059, 6058, 1689, 57, 5189, 5348, 5409, 5652, 4439, 5853, 5857, 4068, 173, 5192, 5917, 189, 2332, 3863, 1685, 2171,   
1456, 1456, 110, 2276, 4643, 3454, 4129, 4367, 1763, 1979, 5435, 5920, 5108, 57, 1855, 2073, 1854, 2195, 5998, 2383,   
6486, 2112, 4431, 3078, 3116, 1685, 1697, 3725, 4683, 4673, 5644, 5643, 1731, 4299, 4299, 4299, 4299, 4299, 4299,   
4299, 4299, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 4218, 6078]

Length: 55 BPE Speech Tokens / 94 Expanded Speech Tokens

Compression Rate: 58.51%

## Case Study 4: Open-ended Dialogue

Input: What is the effect ofsaying hello to the person?

## Step-Audio 2 mini

Text Content: Saying "Hello" is an act of greeting, which can have several effects on both parties involved.

## Speech Tokens:

[1499, 4218, 6486, 4299, 3894, 5158, 6455, 6458, 3222, 2246, 2195, 2573, 2825, 2834, 2753, 2807, 3995, 6182, 6181,   
3984, 2031, 5835, 5838, 3651, 2112, 5109, 5832, 2918, 1046, 3962, 6148, 2486, 4780, 4806, 5077, 2813, 635, 728, 1457,   
3644, 5827, 2028, 3645, 2922, 3165, 2032, 1951, 557, 4526, 6382, 3231, 2243, 3666, 1480, 1735, 1708, 5834, 5915,   
5914, 4543, 1869, 165, 1896, 4431, 1520, 1978, 515, 5078, 4284, 6534, 2109, 1221, 657, 4367, 6071, 5587, 5100, 231,   
75, 2816, 620, 1808, 4076, 3994, 4227, 4299, 3975, 5919, 5916, 3648, 3888, 4131, 4215, 4299, 2113, 3643, 5582, 240,   
5244, 3135, 1788, 2355, 4390, 19, 1479, 5106, 5841, 3647, 326, 2169, 4509, 6374, 6377, 5652, 4406, 4537, 4356, 6543,   
6310, 5086, 5097, 2003, 1028, 4519, 6453, 6537, 6540, 2656, 4540, 2924, 737, 197, 1203, 813, 4191, 6131, 5402, 5644,   
1759, 1979, 5591, 4874, 587, 4314, 2130, 2139, 712, 626, 5087, 5094, 4770, 6375, 1893, 2166, 5670, 5589, 5753, 6266,   
6050, 957, 2267, 566, 1295, 2996, 6373, 795, 2006, 4769, 1638, 4073, 2074, 6546, 713, 4946, 5022, 5053, 2789, 629,   
719, 2915, 4368, 6507, 1032, 2241, 732, 4299]

Length: 186 Speech Tokens

## Vanilla-FT-EN

Text Content: Saying hello greets them warmly, fostering connection and mutual respect. A simple gesture that often softens interactions, especially with those we’ve not met before.

## Speech Tokens:

[3704, 6486, 6486, 4299, 4218, 5133, 5401, 6377, 5648, 4451, 2222, 4841, 5102, 4994, 5047, 6180, 4642, 5106, 5832,   
5842, 3647, 3719, 5986, 47, 4457, 4807, 4997, 623, 632, 1370, 2184, 489, 1450, 4367, 3872, 4598, 4857, 5100, 1194,   
4679, 4435, 1734, 1833, 1035, 59, 2216, 2124, 4316, 4319, 2130, 1430, 1457, 2870, 6482, 6238, 4311, 6174, 3206, 4043,   
29, 386, 647, 1376, 2807, 2517, 6486, 6486, 3975, 5838, 5835, 3645, 2112, 6426, 6534, 3537, 491, 4942, 4411, 5644,   
2139, 4431, 2290, 6059, 6554, 6302, 5336, 5075, 6181, 6502, 1815, 168, 2203, 1479, 3669, 2228, 5920, 4539, 1869,   
3108, 5319, 4593, 159, 143, 32, 1482, 1484, 1483, 1735, 1951, 2924, 4382, 750, 1725, 161, 2127, 2128, 1646, 3563,   
5342, 4851, 951, 6055, 6051, 3099, 4772, 5009, 5000, 5086, 4284, 4366, 3863, 4499, 4428, 6376, 1896, 2166, 4511,   
5111, 1466, 815, 1923, 921, 3054, 1275, 4432, 2193, 735, 6486, 6486, 6486, 6486, 6486, 6486, 6405, 6162, 3894, 3648,   
1950, 1703, 2918, 2254, 6455, 6380, 807, 4445, 6012, 2175, 3591, 5065, 5005, 885, 2490, 3147, 2330, 4459, 4414, 4672,   
2814, 5487, 6054, 1673, 1679, 1676, 1685, 3875, 3813, 3651, 5919, 2922, 1545, 2085, 63, 2195, 220, 1516, 2003, 3407,   
4949, 5026, 3429, 6507, 1032, 2217, 4495, 5124, 6099, 5645, 6374, 5643, 4727, 4945, 5053, 6345, 6537, 6453, 146, 68,   
1480, 1481, 3671, 6385, 5404, 1763, 1285, 5021, 3751, 1887, 4515, 1632, 4115, 1598, 1493, 5921, 5998, 2379, 1569,   
6021, 5322, 3138, 890, 65, 1477, 3668, 5858, 5402, 5405, 33, 2031, 5109, 3732, 5838, 5838, 1464, 1978, 1490, 4379,   
4441, 6373, 2112, 2079, 2189, 5188, 4498, 4590, 4509, 4654, 4910, 4934, 4442, 5020, 4285, 1455, 1208, 2411, 4444,   
6130, 1959, 2100, 73, 2300, 4835, 2810, 641, 2099, 1856, 5909, 5653, 2970, 1788, 5916, 2919, 1701, 3570, 3643, 3314,   
4859, 3067, 6264, 1722, 3670, 1476, 5105, 5671, 5022, 111, 2733, 2220, 1641, 4071, 911, 3653, 5108, 138, 4920, 924,   
2163, 3157, 5750, 2986, 6507, 6534, 2884, 5084, 2906, 719, 680, 1865, 1649, 3836, 3844, 3726, 6486, 6486, 6486, 6486,   
6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 4299, 4299, 4299, 6159]

Length: 350 Speech Tokens

## CoSTA-EN

Text Content: Saying hello fosters connection and breaks ice, especially meaningful from experience.

## Speech Tokens:

[2975, 9039, 1950, 5136, 5404, 7084, 2982, 2267, 2573, 4742, 9727, 4885, 7184, 5839, 5, 3722, 5989, 47, 4540, 4726,   
2729, 16524, 728, 725, 2028, 6768, 6991, 8720, 685, 10458, 13823, 7472, 2196, 3107, 11971, 4531, 46605, 7046, 93,   
747, 5856, 2233, 9501, 1896, 141, 9701, 10551, 6955, 3668, 1501, 6776, 6949, 11175, 3671, 2155, 2175, 10911, 7151,   
9950, 2904, 7161, 6912, 801, 1735, 1948, 7536, 4871, 731, 12873, 3563, 4510, 5405, 10063, 762, 6710, 8056, 7153,   
2582, 4535, 7282, 6921, 2297, 13982, 4404, 8154, 4682, 4910, 5744, 2252, 5020, 913, 15772, 944, 5021, 4821, 6015,   
4480, 4966, 4238, 2076, 6674, 716, 8929, 2915, 4278, 8502, 3872, 3242, 26855, 3586, 4312, 7604, 4849, 1140, 24474,   
1839, 11667, 5101, 5335, 6293, 11772, 152, 14798, 8080, 5421, 8485, 5407, 771, 6669, 28091]

## Expanded Speech Tokens:

[2975, 4299, 6486, 6486, 6486, 6486, 4299, 4218, 1950, 5136, 5404, 6376, 6377, 2982, 2267, 2573, 4742, 5048, 5047,   
4885, 5352, 5109, 5839, 5, 3722, 5989, 47, 4540, 4726, 2729, 551, 635, 728, 725, 2028, 3888, 3888, 1701, 2112, 6426,   
6534, 6534, 685, 4949, 5022, 4410, 6454, 2112, 2244, 2196, 3107, 3863, 6062, 4531, 5725, 3213, 2112, 87, 93, 747,   
5856, 2233, 5921, 4539, 1896, 141, 5319, 4593, 2328, 890, 59, 1479, 1484, 3668, 1501, 2112, 1736, 4463, 4382, 49,   
3669, 3671, 2155, 2175, 2179, 4367, 6302, 6050, 4598, 4849, 2904, 4299, 168, 5158, 6374, 801, 1735, 1948, 6320, 5671,   
4871, 731, 359, 647, 3563, 4510, 5405, 6133, 6376, 762, 4299, 4299, 4299, 4218, 3651, 2922, 2112, 1951, 2582, 4535,   
5158, 6376, 2139, 2166, 2297, 5191, 4543, 4404, 5319, 4509, 4682, 4910, 5744, 2252, 5020, 913, 2131, 2127, 944, 5021,   
4821, 6015, 4480, 4966, 4238, 2076, 6534, 6534, 716, 626, 626, 2915, 4278, 6534, 6534, 4356, 3872, 3242, 5591, 4955,   
3586, 4312, 1763, 1277, 4849, 1140, 3024, 6373, 1839, 2166, 954, 5101, 5335, 6293, 4358, 1685, 152, 386, 35, 1480,   
3671, 5421, 5402, 5405, 5407, 771, 4299, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 6486, 4299, 4299, 4299, 4299,   
4299, 4299, 4299, 4299, 4299, 4299, 4299, 6159]

Length: 130 BPE Speech Tokens / 207 Expanded Speech Tokens Compression Rate: 62.80%

## Case Study 5: Empathetic Customer Dialogue

Input: What to say a customer that mentioned theirfather is in the hospital?

## Step-Audio 2 mini

Text Content: I’m really sorry to hear that your father is in the hospital. Is there anything I can do for you?

## Speech Tokens:

[1502, 1978, 4218, 4137, 1950, 1951, 3404, 3647, 203, 2124, 2128, 2152, 2179, 3881, 5587, 4603, 6066, 2267, 566,   
4431, 4672, 5645, 2484, 2702, 4949, 5779, 6293, 1685, 404, 303, 2217, 323, 4380, 4704, 2760, 5749, 6061, 3872, 63,   
315, 2, 1763, 2023, 4129, 1928, 6534, 6537, 3618, 5675, 6319, 4699, 543, 2252, 4115, 1676, 152, 4680, 2493, 20, 3912,   
46, 1217, 5589, 6318, 5591, 5671, 4384, 4915, 1191, 2160, 227, 873, 1352, 629, 2825, 2689, 2031, 1944, 3645, 3648,   
5835, 3651, 1464, 1707, 3894, 1707, 1950, 1950, 1735, 2258, 5179, 6381, 2013, 2255, 3863, 1919, 2005, 764, 8, 3666,   
4966, 4993, 6426, 4185, 4415, 4805, 5452, 1760, 1946, 4544, 447, 168, 25, 3667, 1563, 60, 1700, 4127, 5824, 6507,   
6534, 3600, 6059, 5668, 4939, 5345, 5342, 5068, 5043, 3495, 4299, 4218, 4137, 4137, 4137]

Length: 135 Speech Tokens

## Vanilla-FT-EN

Text Content: I’m sorry to hear about your father. Is there anything I can do to help?

## Speech Tokens:

[3704, 4299, 6486, 6405, 4218, 4218, 1951, 4133, 6320, 5105, 4382, 4985, 214, 2127, 1481, 5401, 6377, 6373, 74, 3161,   
5753, 6545, 6302, 5297, 4849, 1167, 2490, 2382, 6044, 3725, 5190, 4947, 2778, 5100, 5101, 4859, 962, 1679, 1688,   
1601, 1490, 5348, 387, 2166, 380, 6077, 4862, 5006, 1110, 4431, 2760, 4291, 6071, 3620, 4079, 6534, 6537, 6453, 2459,   
5185, 4455, 153, 2016, 32, 1652, 3875, 3884, 1610, 1843, 4299, 4299, 4299, 4299, 3894, 5838, 5109, 1788, 2031, 395,   
4850, 4444, 6130, 2112, 1029, 2267, 197, 3836, 3866, 1763, 764, 2924, 4462, 3018, 4399, 5073, 4407, 6457, 6129, 4535,   
5021, 5047, 6097, 1705, 1948, 3890, 5837, 4814, 681, 168, 2283, 2195, 753, 1727, 1914, 2247, 1862, 5993, 5500, 5580,   
951, 4434, 897, 4367, 68, 4380, 5118, 8, 734, 2675, 4972, 5088, 2139, 2160, 729, 1869, 6486, 6486, 6486, 6486, 6486,   
6486, 6486, 4299, 4299, 4299, 4299, 4218, 4218, 4218, 4218, 4299, 6078]

Length: 152 Speech Tokens

## CoSTA-EN

Text Content: I’m sorry to hear that. Is there anything I can do to help?

## Speech Tokens:

[3713, 7315, 7077, 7491, 4462, 5065, 4984, 2474, 8, 909, 1805, 3721, 6792, 6373, 1046, 245, 4943, 5779, 6544, 7358,   
4490, 10646, 240, 7328, 903, 4286, 2946, 4785, 9778, 4939, 4606, 6061, 6062, 805, 21388, 9488, 737, 14223, 1761,   
8607, 6933, 6720, 6750, 11002, 17816, 8103, 2267, 224, 9508, 7131, 2195, 17344, 5064, 4740, 6507, 6348, 2943, 4526,   
4985, 8428, 1474, 1948, 7288, 4814, 681, 9299, 9677, 1727, 17688, 1862, 3887, 5258, 63397, 876, 7309, 912, 1370,   
2919, 7800, 9462, 2675, 21749, 1437, 6935, 999, 1809, 6678, 21513]

## Expanded Speech Tokens:

[3713, 4299, 4299, 4299, 4299, 4299, 4299, 4218, 1951, 1946, 4133, 6077, 4462, 5065, 4984, 2474, 8, 909, 1805, 3721,   
6373, 6377, 6373, 1046, 245, 4943, 5779, 6544, 6553, 6050, 4490, 2825, 2834, 240, 2166, 4435, 903, 4286, 2946, 4785,   
4947, 2697, 4939, 4606, 6061, 6062, 805, 2094, 65, 2195, 2921, 737, 59, 1923, 1761, 4435, 2274, 735, 4299, 4299,   
6486, 4299, 4299, 4299, 4299, 4218, 1951, 1295, 5021, 5182, 6373, 2112, 1029, 2267, 224, 3863, 1679, 1763, 737, 2195,   
2370, 2938, 5064, 4740, 6507, 6348, 2943, 4526, 4985, 5047, 6180, 1474, 1948, 4133, 5834, 4814, 681, 168, 2283, 2228,   
3669, 1727, 2184, 2490, 1862, 3887, 5258, 5743, 5823, 876, 546, 4596, 912, 1370, 2919, 5109, 2932, 737, 1463, 2675,   
5080, 5088, 1437, 2166, 1431, 999, 1809, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299,   
4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 6078]

Length: 88 BPE Speech Tokens / 148 Expanded Speech Tokens

Compression Rate: 59.46%