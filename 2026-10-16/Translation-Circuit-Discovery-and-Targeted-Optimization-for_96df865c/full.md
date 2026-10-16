# Translation Circuit Discovery and Targeted Optimization for Low-Resource Multilingual Translation

Atai Waris Khan, Abdullah Mazhar, Raghu Kumar, Md Shad Akhtar Indraprastha Institute of Information Technology Delhi; {ataik,abdullahm,raghu22386,shad.akhtar}@iiitd.ac.in

## Abstract

Recent multilingual large language models achieve strong translation performance, yet the internal mechanisms responsible for lowresource multilingual translation remain insufficiently understood. In this paper, we study translation circuits in multilingual decoder-only LLMs for Hindi→Urdu and Hindi→Sindhi (Devanagari). We propose TITAN, a mechanistic framework that combines causal interven tion, sparse feature decomposition, and targeted preference optimization to identify and opti mize translation-crucial transformer components. Unlike conventional full-parameter finetuning, TITAN restricts optimization to sparse causally important circuits, enabling scalable and parameter-efficient adaptation while updating fewer than 5% of model parameters. Experimental results show that TITAN achieves BLEU improvements of +22.67% on Hindi→Urdu and +45.86% on Hindi→Sindhi over the mul tilingual baselines. Human expert evaluation further confirms substantial gains in adequacy and fluency across both language pairs. Additional analyses reveal that the identified translation circuits for the studied Indic language pairs operate over a shared semantic subspace rather than through an English-centric pivot representation. Overall, our findings demonstrate that mechanistic circuit discovery can directly improve low-resource multilingual translation while providing interpretable insights into multilingual transformer behavior.

## 1 Introduction

Recent multilingual large language models have demonstrated strong translation capabilities across many high-resource language pairs (Zhu et al., 2024; Xu et al., 2024; Alves et al., 2024). At the same time, mechanistic interpretability studies have begun analyzing how translation behavior emerges inside transformer models through circuit analysis and causal intervention (Zhang et al., 2026). However, existing work remains largely focused on English-centric and high-resource settings, leaving the mechanisms underlying lowresource multilingual translation insufficiently explored. This limitation is particularly important for South Asian languages, which often exhibit strong lexical overlap, shared grammatical structure, script variation, and phonological divergence (Subbarao, 2012). Consequently, current fine-tuning approaches optimize model parameters without identifying which components are causally responsible for translation behavior.

In this work, we study multilingual translation circuits in decoder-only LLMs for two lowresource South Asian directions: Hindi→Urdu and Hindi→Sindhi (Devanagari). These language pairs provide complementary linguistic settings: Hindi and Urdu share substantial lexical and grammatical overlap but differ primarily in script, whereas Hindi and Sindhi additionally differ in phonology and lexical composition due to Sindhi-specific implosive consonants and Persian-Arabic lexical influence (Raza et al., 2004). This setting enables controlled analysis of whether related low-resource translations rely on shared or specialized computational mechanisms.

To address this problem, we propose TITAN (Translation-aware Interpretability through Targeted Activation Networks), a mechanistic framework for parameter-efficient multilingual translation. TITAN combines CIRCE (Causal Intervention for Representation Circuit Engineering), sparse feature decomposition, and targeted preference optimization to identify and optimize translation-crucial transformer components. Unlike conventional full-parameter fine-tuning, TITAN restricts optimization to sparse causally important circuits, improving both interpretability and parameter efficiency.

Translation Circuit. We define a translation circuit as a sparse set of transformer components that causally contribute to a specific translation behavior. These components are identified through causal intervention by measuring their effect on translation-relevant outputs, yielding a circuit comprising causally important attention heads and MLP blocks for the studied translation task.

Our analysis reveals that the identified translation circuits for the studied low-resource Indic language pairs are distributed across attention and MLP components. We further show that the identified translation circuits for the studied Indic translation directions operate through a shared semantic subspace rather than through the English-centric pivot mechanisms reported in prior work (Zhang et al., 2026). Experimentally, TITAN achieves BLEU improvements of +33.75% on Hindi→Urdu and +45.86% on Hindi→Sindhi while updating fewer than 5% of model parameters. Human expert evaluation further confirms substantial improvements in adequacy and fluency across both language pairs. We additionally validate the generalisability of the discovered sparse-circuit behavior on English↔French and English↔Hindi translation directions.

Contributions. Our contributions are summarized as follows:

• We propose TITAN, a mechanistic framework that combines causal intervention, sparse feature decomposition, and targeted preference optimization for parameter-efficient multilingual translation.

• We show that the identified translation circuits for the studied low-resource Indic language pairs operate through a shared semantic subspace rather than an English-centric pivot mechanism.

• We demonstrate that restricting optimization to causally identified translation circuits substantially improves translation quality while updating fewer than 5% of model parameters.

Reproducibility. Code and dataset is opensourced on § GitHub 2 .

## 2 Related Work

Low-resource neural machine translation, mechanistic interpretability, and multilingual large language models have emerged as closely connected research directions. Prior work has improved multilingual transfer and translation quality for underrepresented languages, while recent mechanistic interpretability research enables causal analysis of transformer computations through activation intervention and sparse feature decomposition.

Low-Resource Neural Machine Translation. Low-resource neural machine translation (NMT) remains challenging due to limited parallel corpora and linguistic diversity across Indic languages. Recent work has explored multilingual transfer learning, back-translation, transliteration, and pivotbased modeling to improve translation quality for low-resource Indian languages (Wei et al., 2024; P M et al., 2024; Bala Das et al., 2024). Recent parameter-efficient approaches further investigate language-specific adaptation, including languagespecific LoRA, intrinsic language-specific subspaces, and language-aware parameter selection for multilingual knowledge transfer. These methods improve language-specific adaptation but primarily focus on parameter specialization rather than identifying components that are causally responsible for translation behavior.

## Mechanistic Interpretability and Multilingual

LLMs. Mechanistic interpretability seeks to reverse-engineer transformer computations through causal interventions and circuit analysis. Prior work has used activation and path patching to identify circuits underlying reasoning and attention. In multilingual neural machine translation, studies have identified language-specific subspaces emerging during multilingual fine-tuning (Cao et al., 2024), as well as language-aware parameters for targeted cross-lingual knowledge transfer (Dong et al., 2025). Language-specific fine-tuning has further been explored for adapting LLMs to low-resource translation settings (Liang et al., 2025). Recent circuit-discovery work, such as Emotion Circuits (Wang et al., 2025), combines contrastive examples with causal analysis to identify functionally relevant representations. Unlike these approaches, TITAN focuses on discovering translation-specific circuits and leveraging them for targeted multilingual translation optimization.

Preference Optimization for Translation Alignment. Preference optimization methods such as Direct Preference Optimization have recently been used to improve multilingual generation and translation quality through pairwise preference learning. Existing approaches generally optimize model parameters without identifying which components are causally responsible for translation behavior.

<table><tr><td rowspan="2">Pair</td><td rowspan="2">Split</td><td rowspan="2">#Sent</td><td colspan="2">Hindi</td><td colspan="2">Target Language</td><td rowspan="2">Avg. Len.</td></tr><tr><td>#Tokens</td><td>#Types</td><td>#Tokens</td><td>#Types</td></tr><tr><td rowspan="3">Hi-Sd</td><td>SFT Train</td><td>20K</td><td>404.6k</td><td>36.1K</td><td>398.2K</td><td>49.7K</td><td>20.1</td></tr><tr><td>DPO Train</td><td>15K</td><td>302.8K</td><td>30.5K</td><td>297.8K</td><td>41.6K</td><td>20.0</td></tr><tr><td>Contrastive-From SFT</td><td>10K</td><td>220.6K</td><td>15.8K</td><td>217.6K</td><td>20.5K</td><td>21.9</td></tr><tr><td rowspan="3">Hi-Ur</td><td>SFT Train</td><td>30K</td><td>562.3K</td><td>47.3K</td><td>599.2K</td><td>43.7K</td><td>19.4</td></tr><tr><td>DPO Train</td><td>30K</td><td>567.1K</td><td>47.4K</td><td>603.1K</td><td>43.8K</td><td>19.5</td></tr><tr><td>Contrastive-From SFT</td><td>10K</td><td>186.9K</td><td>15.6K</td><td>198.9K</td><td>15.1K</td><td>19.3</td></tr></table>

Table 1: Corpus statistics for the Hindi–Sindhi (Devanagari) and Hindi–Urdu parallel corpora used in TITAN. Each corpus is partitioned into SFT, DPO, and contrastive analysis splits. We sample a 10K contrastive subset from the SFT split of each language pair to construct contrastive-based intervention prompts for causal translation circuit identification. Hi–Sd: Hindi–Sindhi (Devanagari); Hi–Ur: Hindi–Urdu. Avg. Len. denotes the mean number of whitespace-tokenized tokens per sentence.

In contrast, TITAN combines mechanistic circuit discovery with targeted optimization to selectively optimize translation-relevant components.

## 3 Dataset

We evaluate TITAN on two low-resource South Asian translation pairs: Hindi–Urdu and Hindi– Sindhi (Devanagari). These language pairs provide complementary multilingual settings involving script divergence, lexical overlap, and phonological variation. Corpus statistics are summarized in Table 1.

## 3.1 Translation Corpora

We use the Hindi–Urdu and Hindi–Sindhi subsets of the Bharat Parallel Corpus Collection (BPCC) (Gala et al., 2023). Hindi and Urdu share substantial grammatical and lexical overlap but differ primarily in script, with Hindi written in Devanagari and Urdu in Arabic-Nastaliq. In contrast, although Hindi and Sindhi both use the Devanagari script, Sindhi additionally contains implosive consonants and stronger Persian-Arabic lexical influence.

After preprocessing and deduplication, the Hindi–Urdu corpus contains 60K parallel sentence pairs, while the Hindi–Sindhi corpus contains 35K sentence pairs. Both corpora are partitioned into supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and contrastive analysis splits.

## 3.2 Contrastive Translation Data

To support mechanistic analysis, we construct contrastive translation pairs consisting of valid translation prompts and counterfactual prompts where translation intent is removed while preserving grammatical structure. We sample 10K sentence pairs from the SFT split of each language pair and generate counterfactual variants using semantic perturbation strategies. For Hindi–Sindhi, sampling is additionally stratified to ensure sufficient coverage of Sindhi-specific implosive phoneme markers. Detailed prompt templates, perturbation taxonomies, and validation protocols are provided in Appendix Section A.

## 3.3 Preprocessing

We apply uniform preprocessing across both corpora, including Unicode normalization, duplicate removal, and sentence-length filtering. Hindi and Sindhi samples are validated using Devanagari Unicode ranges, whereas Urdu samples are verified using Arabic-script code points. For Hindi–Sindhi, we additionally preserve Sindhi-specific implosive phoneme markers to support phonological circuit analysis.

## 4 Methodology

We propose TITAN (Translation-aware Interpretability through Targeted Activation Networks), a mechanistic interpretability-guided framework for parameter-efficient multilingual translation. Rather than updating all transformer parameters, TITAN identifies the sparse set of components causally responsible for translation behavior and restricts optimization to the discovered translation circuits. The framework is built upon CIRCE (Causal Intervention for Representation Circuit Engineering), which combines subspace intervention with sparse feature decomposition to isolate translation-relevant computation. All CIRCE and TITAN experiments use Gemma-2 9B as the underlying base model.

Hindi–Urdu and Hindi–Sindhi provide complementary multilingual settings for circuit analysis. Hindi–Urdu exhibits strong grammatical overlap with script divergence, whereas Hindi–Sindhi shares script similarity but differs substantially in phonology and lexical composition due to Sindhispecific implosive phonemes and Persian-Arabic lexical influence. This setting enables TITAN to analyze how translation circuits specialize across distinct forms of linguistic variation.

## 4.1 Contrastive Translation Construction

To isolate translation-specific computation from general language modeling behavior, we construct contrastive translation pairs consisting of a valid translation prompt $X ^ { + }$ and a counterfactual prompt $X ^ { - }$

$$
X ^ { + } = { } ^ { 4 } [ \mathsf { S R C } ] : [ \mathsf { t e x t } ] \ - \ [ \mathsf { T G } \mathsf { T } ] : { } ^ { , }\tag{1}
$$

Counterfactual prompts preserve grammatical structure while removing translation intent through target nullification, semantic distortion, and contradictory task descriptors. This design allows TITAN to separate activations associated with translation generation from those associated with general multilingual processing.

Each sample additionally contains translation direction labels, vocabulary overlap annotations, and phoneme-level markers. For Hindi–Sindhi, we explicitly annotate samples containing Sindhispecific implosive phonemes to support phonological circuit analysis.

## 4.2 CIRCE: Translation Circuit Identification

TITAN identifies translation-relevant transformer components using CIRCE, a subspace-constrained extension of activation patching. For a transformer component c at layer $\ell ,$ we compute activation differences between positive and counterfactual prompts:

$$
\Delta \mathbf { a } _ { c } ^ { ( i ) } = \mathbf { a } _ { c } ( X _ { i } ^ { + } ) - \mathbf { a } _ { c } ( X _ { i } ^ { - } )\tag{2}
$$

The resulting activation differences are aggregated into a component-specific activation matrix M and decomposed into translation-relevant structure and residual variation:

$$
\begin{array} { r l } { \underset { \mathbf { S } _ { c } , \mathbf { E } _ { c } , \mathbf { T } } { \mathrm { m i n } } } & { { } \left\| \mathbf { M } _ { c } - \mathbf { S } _ { c } \mathbf { 1 } ^ { \top } - \mathbf { E } _ { c } \mathbf { \Gamma } ^ { \top } \right\| _ { F } } \\ { \mathrm { s . t . } } & { { } \mathrm { S p a n } ( \mathbf { S } _ { c } ) \perp \mathrm { S p a n } ( \mathbf { E } _ { c } ) } \end{array}\tag{3}
$$

Here, $\mathbf { S } _ { c }$ denotes a single translation-relevant direction, while $\mathbf { E } _ { c }$ captures residual variation; this direction defines the intervention basis $\mathbf { U } _ { c }$ . Instead of replacing the full activation, CIRCE intervenes only along the translation-relevant direction:

$$
\tilde { \mathbf { a } } _ { c } = \mathbf { U } _ { c } \mathbf { U } _ { c } ^ { \top } \mathbf { a } _ { c } ( X ^ { - } ) + ( \mathbf { I } - \mathbf { U } _ { c } \mathbf { U } _ { c } ^ { \top } ) \mathbf { a } _ { c } ( X ^ { + } )\tag{4}
$$

Component importance is measured using intervention-induced translation logit changes. Rather than using fixed thresholds, TITAN calibrates component selection from the empirical intervention distribution using a $\mathbf { P 9 9 }$ noise-floor criterion, yielding an operating threshold near 0.25%. This preserves sparse but reproducible lowresource translation signals while filtering negligible intervention effects. Components producing substantial translation degradation under intervention are retained as translation-crucial, forming the translation circuit used for optimization; CIRCE applies targeted SFT to these components while freezing all others.

For interpretability, discovered attention heads are further categorized into source, indicator, and positional heads based on their dominant routing behavior.

## 4.3 Sparse Feature Decomposition

Transformer activations frequently exhibit feature superposition, where multiple linguistic properties are encoded within the same representation. To disentangle these representations, TITAN trains Sparse Autoencoders (SAEs) exclusively on residual activations extracted from CIRCE-identified translation-crucial layers. Restricting SAE training to CIRCE-identified layers ensures that the discovered sparse features remain grounded in causally relevant translation circuitry rather than generic multilingual representations.

For an activation vector $\mathbf { h } ^ { ( \ell ) }$ , the encoder computes a sparse latent representation:

$$
\mathbf { z } ^ { ( \ell ) } = \operatorname { R e L U } \left( ( \mathbf { W } _ { \ell } ) ^ { \top } \mathbf { h } ^ { ( \ell ) } + \mathbf { b } _ { \ell } \right)\tag{5}
$$

The original activation is reconstructed through:

$$
\hat { \mathbf { h } } ^ { ( \ell ) } = \mathbf { W } _ { \ell } \mathbf { z } ^ { ( \ell ) }\tag{6}
$$

Training minimizes a joint reconstruction and sparsity objective:

$$
\mathcal { L } _ { \mathrm { S A E } } = \| \mathbf { h } ^ { ( \ell ) } - \hat { \mathbf { h } } ^ { ( \ell ) } \| _ { 2 } ^ { 2 } + \lambda \| \mathbf { z } ^ { ( \ell ) } \| _ { 1 }\tag{7}
$$

The resulting sparse representations expose disentangled translation features that are substantially easier to interpret than raw transformer activations.

## 4.4 Feature-Level Causal Scoring

We evaluate each SAE feature $z _ { j }$ using language selectivity and causal lift. To isolate language identity from semantic variation, selectivity is computed using semantically matched bilingual prompts expressing the same meaning in different languages:

$$
\widetilde { \mathrm { S e l } } _ { j } = \frac { \mathbb { E } \left[ z _ { j } ( x _ { \mathrm { t g t } } ) \right] - \mathbb { E } \left[ z _ { j } ( x _ { \mathrm { s r c } } ) \right] } { \mathrm { S t d } \left[ z _ { j } ( x _ { \mathrm { t g t } } ) \right] + \mathrm { S t d } \left[ z _ { j } ( x _ { \mathrm { s r c } } ) \right] + \epsilon }\tag{8}
$$

To estimate causal relevance beyond correlation, we intervene on feature j during decoding and measure the induced translation logit change:

$$
\mathrm { L i f t } _ { j } ^ { ( \ell ) } = \mathbb { E } _ { x \sim D } \left[ \frac { 1 } { | T | } \sum _ { t \in T } \Delta M _ { \alpha } ^ { ( \ell , j ) } ( x , t ) \right]\tag{9}
$$

where T denotes decoding steps and $\Delta M _ { \alpha } ^ { ( \ell , j ) }$ represents the translation logit change induced by intervention magnitude α.

We compute a joint feature score:

$$
\mathbf { S c o r e } _ { j } = \operatorname* { m a x } ( \widetilde { \mathrm { S e l } } _ { j } , 0 ) \cdot \mathrm { m a x } ( \mathrm { L i f t } _ { j } , 0 )\tag{10}
$$

Features exhibiting both high selectivity and positive causal lift are retained as translation-relevant features, while low-scoring features are discarded. This filtering stage provides a finer-grained characterization of the CIRCE-identified translation circuit by distinguishing causally relevant features from language-selective but causally weak features.

## 4.5 Targeted Translation Optimization

Using the discovered translation circuit, CIRCE performs targeted SFT on the identified components while freezing all others, whereas TITAN further applies DPO to the same components. Training proceeds hierarchically from word-level translation to phrase-level and finally sentence-level translation, allowing the model to gradually acquire increasingly complex translation structure.

To preserve the general language capabilities of the base model, optimization is regularized using a KL divergence constraint:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { M T } } + \lambda _ { \mathrm { K L } } \mathrm { K L } ( p _ { \theta } \| p _ { \mathrm { b a s e } } ) } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { M T } }$ denotes the standard translation objective.

After supervised fine-tuning, TITAN further refines translation quality using Direct Preference Optimization. Preference pairs are constructed by treating the gold reference translations from the parallel corpus as preferred outputs, while translations generated by the supervised fine-tuned model are treated as rejected outputs.

## 5 Experiments and Results

Here, we compare TITAN against multilingual machine translation systems and decoder-only large language models across Hindi-Urdu and Hindi-Sindhi (Devanagari) translation tasks. We further analyze the effectiveness of targeted circuit optimization under both SFT and DPO settings. In addition to quantitative evaluation, we provide mechanistic analysis to study the role of sparse translation circuits in low-resource multilingual translation.

Baselines. We compare TITAN against both multilingual neural machine translation systems and instruction-tuned decoder-only LLMs under a unified evaluation setup. The evaluated baselines include NLLB-200 and IndicTrans2, which represent strong multilingual translation models specifically designed for low-resource settings, as well as Gemma-2, Qwen-3.5, LLaMA-3.1, and Mistral3, which represent recent decoder-only multilingual LLMs. All models are evaluated under zero-shot, supervised fine-tuning (SFT), and DPO settings. CIRCE denotes targeted SFT on the causally identified translation-crucial components, while TITAN further applies DPO to the same components.

Performance Comparison. Table 2 reports translation performance across Hindi–Urdu and Hindi– Sindhi translation directions. Zero-shot performance remains consistently weak across both multilingual MT systems and decoder-only LLMs, particularly for Hindi–Sindhi due to additional phonological and lexical divergence. Supervised fine-tuning and DPO improve translation quality, with Gemma-2 and NLLB-200 emerging as the strongest multilingual baselines across most directions.

TITAN consistently outperforms all baselines across both language pairs. The gains are most pronounced for Hindi–Sindhi, where TITAN achieves relative BLEU improvements of +45.86% and +48.98% for H→S and S→H respectively over the strongest baselines. Similar improvements are observed for Hindi–Urdu, demonstrating that restricting optimization to causally identified translation circuits substantially improves low-resource multilingual translation. Notably, the CIRCE-only variant already surpasses several fully optimized baselines despite updating only sparse translationrelevant components. Overall, the results show that TITAN provides more scalable and parameterefficient alternative to conventional full-model optimization.

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="7">Hindi-Urdu</td><td colspan="7">Hindi–Sindhi (Devanagari)</td></tr><tr><td colspan="3">H→U</td><td colspan="3">U→H</td><td colspan="3">H→S</td><td colspan="3"></td><td colspan="3">S→H</td></tr><tr><td colspan="2">Models</td><td>BL</td><td colspan="2">CF2 TER</td><td>COM BL</td><td>CF2</td><td>TER</td><td>COM</td><td>BL</td><td>CF2</td><td>TER</td><td>COM</td><td>BL</td><td>CF2</td><td>TER COM</td></tr><tr><td rowspan="6">ZS</td><td>NLLB-200</td><td>0.01</td><td>0.09</td><td>105.65 57.08</td><td></td><td>0.31</td><td>0.61</td><td>108.63 55.39</td><td>0.44</td><td>1.05</td><td></td><td>101.23 41.34</td><td>2.31</td><td>10.58</td><td></td><td>43.08</td></tr><tr><td>IndicTrans2</td><td>4.33</td><td>25.19</td><td>80.69</td><td>62.85</td><td>2.12</td><td>23.63</td><td>87.16 60.57</td><td>0.59</td><td>3.08</td><td></td><td>119.88 41.76</td><td>1.15</td><td>7.38</td><td>91.36</td><td>109.90 55.16</td></tr><tr><td>Mistral-3</td><td>0.32</td><td>2.40</td><td>103.46 33.70</td><td></td><td>0.19</td><td>2.56</td><td>102.95 30.12</td><td></td><td>0.17</td><td>1.92</td><td>296.833.43</td><td>0.68</td><td>5.21</td><td></td><td>238.11 27.63</td></tr><tr><td>LLaMA-3.1</td><td>1.46</td><td>3.78</td><td>117.57 43.83</td><td></td><td>0.11</td><td>4.20</td><td>100.79 44.42</td><td></td><td>0.51</td><td>3.63</td><td>273.22 37.52</td><td>1.21</td><td>6.93</td><td></td><td>250.01 33.43</td></tr><tr><td>Qwen-3.5</td><td>0.21</td><td>3.02</td><td>99.23</td><td>40.30</td><td>0.53</td><td>3.30</td><td>99.02</td><td>41.22</td><td>0.21</td><td>3.02</td><td>99.23 34.80</td><td>0.53</td><td>3.30</td><td></td><td>109.02 31.20</td></tr><tr><td>Gemma-2</td><td>5.93</td><td>28.23</td><td>69.81</td><td>59.05</td><td>6.40</td><td>25.58</td><td>79.30</td><td>53.29</td><td>0.62</td><td>5.93</td><td>187.74 39.87</td><td>2.26</td><td>8.42</td><td>79.30</td><td>36.41</td></tr><tr><td rowspan="6">SF</td><td>NLLB-200</td><td>23.39</td><td>56.61</td><td>41.70</td><td>72.76</td><td>22.03</td><td>51.20</td><td>44.77</td><td>73.73</td><td>15.89</td><td>33.42</td><td>88.22 56.83</td><td>17.33</td><td>31.29</td><td>78.05</td><td>65.89</td></tr><tr><td>IndicTrans2</td><td>23.63</td><td>41.43</td><td>76.24</td><td>70.87</td><td>25.36</td><td>50.33</td><td>49.36</td><td>74.38</td><td>14.83</td><td>34.57</td><td>78.65 60.2</td><td>18.66</td><td>48.96</td><td>81.23</td><td>67.14</td></tr><tr><td>LLaMA-3.1</td><td>10.59</td><td>36.58</td><td>59.87</td><td>66.8</td><td>9.59</td><td>31.32</td><td>70.21</td><td>65.35</td><td>9.81</td><td>23.22 95.99</td><td>58.98</td><td>10.25</td><td>18.59</td><td>96.76</td><td>57.38</td></tr><tr><td>Qwen-3.5</td><td>8.89</td><td>33.90</td><td>73.27</td><td>64.05</td><td>7.25</td><td>30.86</td><td>73.15</td><td>51.63</td><td>5.63</td><td>17.10</td><td>110.99 38.52</td><td>7.21</td><td>21.85</td><td>99.02</td><td>45.31</td></tr><tr><td>Gemma-2</td><td>23.79</td><td>56.66</td><td>40.98</td><td>74.16</td><td>25.89</td><td>54.69</td><td>43.75</td><td>75.33</td><td>15.94</td><td>34.10</td><td>86.5059.86</td><td>17.38</td><td>47.56</td><td>75.28</td><td>72.77</td></tr><tr><td>Mistral-3</td><td>9.47</td><td>31.94</td><td>72.57</td><td>69.16</td><td>12.00</td><td>36.78</td><td>69.28</td><td>67.22</td><td>5.44</td><td>20.38</td><td>102.90 55.77</td><td>5.58</td><td>23.63</td><td>95.25</td><td>51.93</td></tr><tr><td rowspan="6">DPO CIRCE</td><td>NLLB-200</td><td>25.86</td><td>59.36</td><td>40.14</td><td>74.98</td><td>23.51</td><td>55.52</td><td>43.00</td><td>75.15</td><td>17.00</td><td>40.92</td><td>58.22 58.39</td><td>18.34</td><td>50.15</td><td>51.13</td><td>70.64</td></tr><tr><td>IndicTrans2</td><td>24.68</td><td>58.25</td><td>50.22</td><td>71.52</td><td>24.96</td><td>56.30</td><td>40.58</td><td>74.84</td><td>17.32</td><td>40.30</td><td>70.60</td><td>61.9</td><td>20.63</td><td>50.91</td><td>72.55 68.56</td></tr><tr><td>LLaMA-3.1</td><td>11.29</td><td>39.18</td><td>53.63</td><td>68.37</td><td>9.93</td><td>34.50</td><td>57.70</td><td>65.83</td><td>10.21</td><td>27.66 72.36</td><td>59.89</td><td>12.11</td><td>23.32</td><td>70.39</td><td>61.85</td></tr><tr><td>Qwen-3.5</td><td>8.90</td><td>33.70</td><td>71.64</td><td>64.98</td><td>7.32</td><td>30.90</td><td>71.21</td><td>52.74 8.98</td><td></td><td>29.31 96.44</td><td>44.96</td><td>8.36</td><td>30.02</td><td>87.37</td><td>50.04</td></tr><tr><td>Gemma-2</td><td>25.94</td><td>60.10</td><td>42.37</td><td>75.88</td><td>25.78</td><td>56.10</td><td>44.39 77.37</td><td>18.01</td><td>43.68</td><td>84.71</td><td>62.81</td><td>19.89</td><td>51.20</td><td>69.25</td><td>74.55</td></tr><tr><td>Mistral-3</td><td>9.29</td><td>30.61</td><td>73.60</td><td>68.84</td><td>11.22</td><td>36.10</td><td>68.28 70.01</td><td>6.22</td><td></td><td>25.58 97.50</td><td>56.88</td><td>7.76</td><td>32.84</td><td>88.25</td><td>54.69</td></tr><tr><td colspan="2">TITAN</td><td>28.03 62.47</td><td></td><td>58.56</td><td>79.33</td><td>26.23</td><td>53.88</td><td>71.36</td><td>80.52</td><td>22.19</td><td>52.90</td><td>45.21 65.76</td><td>25.47</td><td>61.48</td><td>28.36</td><td>76.5</td></tr><tr><td colspan="2">∆SFT(%)</td><td>31.82</td><td>65.93</td><td>52.30</td><td>80.52</td><td>27.01</td><td>55.44</td><td>72.10 81.28</td><td>23.25</td><td>55.92</td><td>42.52</td><td>70.17</td><td>27.80</td><td>65.33</td><td>25.54</td><td>78.95</td></tr><tr><td colspan="2"></td><td>个 33.75</td><td>↑ 5.77</td><td>→ 27.62</td><td>↑ 6.11</td><td>↑ 4.33</td><td>↑ 1.37</td><td>↓ R 60.23 5.05</td><td>个 45.86</td><td>↑</td><td>↑ 61.76 45.94 11.72</td><td>个</td><td>个 48.9833.44</td><td>个</td><td>个</td><td>↑ 66.07 5.90</td></tr></table>

Table 2: Translation performance comparison across Hindi-Urdu and Hindi-Sindhi (Devanagari) under zero-shot (ZS), supervised fine-tuning (SFT), and Direct Preference Optimization (DPO) settings. TITAN consistently outperforms multilingual translation baselines across both language pairs, demonstrating the effectiveness of sparse causal translation circuits for low-resource multilingual translation. All CIRCE and TITAN results use Gemma-2 9B as the underlying base model. BL: BLEU, CF2: ChrF2, TER: Translation Edit Rate, COM: COMET.

For a parameter-controlled comparison, we match LoRA SFT+DPO to TITAN at the same 3.14% trainable-parameter budget. Despite the identical budget, LoRA SFT+DPO achieves 26.37 BLEU, compared with 31.82 BLEU for TITAN (+5.45 BLEU), indicating that the gain comes from circuit-based component selection rather than parameter count.

Generalizability. We evaluate TITAN beyond the low-resource Hindi–Urdu and Hindi–Sindhi settings in two complementary ways. For each evaluation direction, the translation circuit is independently identified using CIRCE on that language pair before targeted optimization. We evaluate TITAN beyond the low-resource Hindi–Urdu and Hindi– Sindhi settings in two complementary ways.

First, we assess high-resource cross-lingual generalization on Hindi–English and English–French. Table 3 reports the results. For Hindi→English, TITAN improves BLEU from 29.33 with DPO to 34.21, while English→Hindi improves from 28.71 to 31.94. On English→French, BLEU increases from 30.58 to 33.57, and French→English from 33.09 to 36.40. Similar improvements are observed in ChrF++ and TER, indicating consistent gains across evaluation metrics.

Second, to test whether the observed behavior extends beyond closely related Indic languages, we evaluate TITAN on two typologically distinct low-resource directions: Hindi→Tamil and English→Irish. Table 4 reports the results. TITAN improves BLEU from 23.01 to 26.71 on Hindi→Tamil and from 29.52 to 33.19 on English→Irish over the DPO baseline, corresponding to relative improvements of 16.08% and 12.43%, respectively. The consistent performance ordering of SFT < DPO < CIRCE < TITAN across language pairs further supports the generality of circuit-based optimization beyond the closely related Indic language pairs used for circuit discovery.

<table><tr><td>Setting Direction BLEU chrF++ TER</td></tr><tr><td>COMET Hindi-English</td></tr><tr><td>SFT Hi→En 27.96 52.36 70.24 73.71</td></tr><tr><td>DPO Hi→En 29.33 54.37 66.35 75.43 CIRCE Hi→En 33.48 58.11 66.34 76.39</td></tr><tr><td>TITAN Hi→En 34.21 60.04 60.22 76.72</td></tr><tr><td>SFT En→Hi 26.08 50.88 70.49 78.15</td></tr><tr><td>DPO En→Hi 28.71 52.34 65.91 79.6</td></tr><tr><td>CIRCE En→Hi 31.03 56.99 65.16 82.36</td></tr><tr><td>TITAN En→Hi 31.94 57.51 62.21 83.95</td></tr><tr><td>English-French</td></tr><tr><td>SFT En→Fr 29.64 52.88 60.44 80.01</td></tr><tr><td>DPO En→Fr 30.58 55.24 58.63 80.88</td></tr><tr><td>CIRCE En→Fr 32.43 55.40 59.55 84.20 TITAN En→Fr 33.57 57.90 54.35 85.51</td></tr><tr><td>SFT Fr→En 31.02 54.89 54.17 82.5</td></tr><tr><td></td></tr><tr><td>DPO Fr→En 33.09 56.51 49.63 83.41</td></tr><tr><td>CIRCE Fr→En 34.87 55.72 50.13 85.98 TITAN Fr→En 36.40 59.25 46.08 86.77</td></tr></table>

Table 3: Cross-lingual generalization on unseen highresource Hindi–English and English–French translation tasks. TITAN consistently outperforms SFT and DPO baselines, using Gemma-2 9B as the base model; CIRCE applies targeted SFT, while TITAN further applies DPO to the same identified components.

## 6 Behavioral Pattern Analysis

We now investigate the internal behavioral patterns learned by multilingual decoder-only LLMs during low-resource translation. Specifically, we analyze whether TITAN identifies genuinely causal translation circuits, how translation responsibilities are distributed across transformer components, and how multilingual representations evolve across layers. Unless otherwise stated, analyses are performed on the strongest TITAN checkpoint for both Hindi→Urdu and Hindi→Sindhi (Devanagari).

<table><tr><td colspan="5">Setting Direction BLEU chrF++ TER COMET</td></tr><tr><td colspan="5">Hindi-Tamil</td></tr><tr><td>SFT DPO</td><td>Hi→Ta</td><td>22.61 40.66</td><td>81.66 77.20</td><td>77.20 79.00</td></tr><tr><td>CIRCE</td><td>Hi→Ta Hi→Ta</td><td>23.01 42.87 25.04 49.46</td><td>75.80</td><td>81.20</td></tr><tr><td>TITAN</td><td>Hi→Ta</td><td>26.71 50.60</td><td>72.20</td><td>82.70</td></tr><tr><td colspan="5">English-Irish</td></tr><tr><td>SFT</td><td>En→Ga</td><td>26.39</td><td>51.71 84.35</td><td>70.60</td></tr><tr><td>DPO</td><td>En→Ga</td><td>29.52 53.71</td><td>80.26</td><td>72.80</td></tr><tr><td>CIRCE TITAN</td><td>En→Ga En→Ga</td><td>30.84 54.42 33.19 57.82</td><td>76.90 66.70</td><td>74.60 75.70</td></tr></table>

Table 4: Performance on additional typologically diverse low-resource translation tasks. TITAN consistently outperforms SFT, DPO, and CIRCE on Hindi– Tamil and English–Irish, demonstrating that the observed gains extend beyond the closely related Indic language pairs used for primary evaluation.

## 6.1 Causal Validation via Progressive Knockout

We evaluate whether CIRCE-identified components are causally necessary for translation behavior through progressive knockout experiments, where identified attention heads and MLP blocks are sequentially ablated using mean-activation replacement. As shown in Figure 1, ablating CIRCEidentified components produces substantially larger degradation than ablating randomly selected components. To rule out circularity from targeted optimization, we additionally evaluate the same CIRCE-identified components in a general LoRA SFT+DPO model trained without restricting optimization to these components. Ablating eight CIRCE-identified components reduces BLEU from 25.53 to 16.54 (35.2%), whereas ablating eight randomly selected components reduces BLEU only from 25.53 to 25.41 (0.45%). For Hindi→Sindhi, removing a single key MLP reduces translation accuracy from 100% to 54.4%, whereas random MLP ablation causes only minor degradation. We further observe that attention-head ablation leads to gradual degradation, indicating distributed routing behavior, while MLP ablation causes sharper collapse, suggesting concentrated semantic transformation functions. Similar trends for Hindi→Urdu are provided in Appendix Section C.

## 6.2 Attention Routing Behavior

We next analyze the most causally important attention heads identified by CIRCE to understand how multilingual translation routing is implemented. As shown in Appendix Figures 3 and 4, both translation directions exhibit a consistent routing pipeline consisting of structural anchoring, source-language identification, translation-direction routing, and final target-language commitment.

![](images/ef8ce8858db3de1b2e3133eaba6d77311a21ff1c3bbe3f3b3532084e5b513807.jpg)  
Figure 1: Progressive knockout analysis for Hindi→Sindhi translation. CIRCE-identified components produce substantially larger degradation than randomly selected components.

Early-layer heads primarily attend to formatting boundaries and language identifiers, whereas later layers shift toward translation-direction routing and target-language commitment. Unlike prior Englishcentric analyses where attention heads predominantly align to source content tokens, the discovered Indic translation circuits rely heavily on explicit language-label anchoring. We attribute this behavior to the strong lexical and subword overlap between closely related Indic languages, where language identifiers provide more discriminative routing signals than raw lexical content. Additional head visualizations and layer-wise routing analyses are provided in Appendix Section D.

## 6.3 MLP Representation Dynamics

To analyze semantic processing within the translation circuit, we probe MLP hidden states using cosine similarity with source-language, targetlanguage, and translation-indicator token embeddings. As shown in Appendix Figures 5 and 6, source-language alignment gradually increases through middle layers, whereas target-language alignment emerges primarily in the final layers, indicating delayed target-language realization during multilingual translation. ; We further observe consistently negative English alignment across all layers for both translation directions, contradicting previously reported English-centric pivot mechanisms. Instead, dominant alignment emerges along shared Indic semantic directions, suggesting that closely related Indic translation operates through a shared semantic subspace rather than an explicit pivot language. Additional layer-wise analyses are provided in Appendix Section E.

## 6.4 Sparse Feature Behavior

Finally, we analyze Sparse Autoencoder (SAE) features extracted from the residual stream to study how translation-relevant representations are distributed across layers. As shown in Appendix Figures 7 and 8, both translation directions exhibit strongly layer-localized feature distributions, where target-language-specific features dominate early layers while Hindi-specific features emerge more prominently in later layers.

Importantly, highly selective features do not necessarily exhibit strong causal influence. Instead, translation behavior is governed by a sparse subset of high-lift features distributed across both source and target language populations. Together with the attention and MLP analyses, these findings suggest that multilingual translation emerges through sparse routing circuits combined with shared semantic computation rather than through explicit pivot-language translation. Additional feature-level analyses are provided in Appendix Section F.

## 7 Discussion

Expert Assessment. To complement automatic metrics, we conduct human evaluation using native bilingual experts proficient in Hindi–Urdu and Hindi–Sindhi (Devanagari). Experts independently evaluate randomly sampled translations from the SFT and DPO stages of both Gemma-2 baselines and the TITAN pipeline using adequacy and fluency scores on a 1–5 scale. Table 5 reports the percentage of ratings ≥ 4. Across both language pairs, expert judgements consistently confirm the effectiveness of TITAN over full-parameter baselines. For Hi–Ur, the final TITAN configuration improves adequacy from 36.7% to 76.7% and fluency from 53.3% to 86.7% relative to the Gemma-2 DPO baseline. Similar gains are already visible at the CIRCE-guided SFT stage, indicating that improvements arise from sparse-circuit optimization rather than DPO alone.

Hi–Sd exhibits the largest qualitative improvements, with TITAN achieving 93.5% adequacy and 61.3% fluency compared to 51.3% and 25.8% for the Gemma-2 DPO baseline. Experts further observed that TITAN better preserves semantic meaning and produces more morphologically appropriate translations for technical terminology.

<table><tr><td>Direction</td><td>System</td><td>Adequacy</td><td>Fluency</td></tr><tr><td colspan="4">Hindi → Urdu</td></tr><tr><td>Gemma-2 (SFT)</td><td>Baseline</td><td>33.3</td><td>20.0</td></tr><tr><td>Gemma-2 (DPO)</td><td>Baseline</td><td>36.7</td><td>53.3</td></tr><tr><td>CIRCE</td><td>Ours</td><td>56.7</td><td>56.7</td></tr><tr><td>TITAN</td><td>Ours</td><td>76.7</td><td>86.7</td></tr><tr><td colspan="4">Hindi → Sindhi</td></tr><tr><td>Gemma-2 (SFT)</td><td>Baseline</td><td>45.2</td><td>12.9</td></tr><tr><td>Gemma-2 (DPO)</td><td>Baseline</td><td>51.3</td><td>25.8</td></tr><tr><td>CIRCE</td><td>Ours</td><td>64.8</td><td>39.4</td></tr><tr><td>TITAN</td><td>Ours</td><td>93.5</td><td>61.3</td></tr></table>

Table 5: Human expert evaluation (% of ratings $\geq 4$ on a 1–5 scale) for adequacy and fluency across Hindi– Urdu and Hindi–Sindhi translation tasks. Best results per direction are highlighted in bold.

Several limitations remain. Sindhi outputs occasionally retain untranslated Hindi loanwords and exhibit minor Unicode inconsistencies around implosive phoneme characters. Performance also degrades on long compound sentences with multiple embedded clauses, where fluency decreases despite largely preserved adequacy.

Error Analysis. Table 2 reveals a consistent asymmetry in TITAN’s residual errors. Across all evaluation settings, TITAN underperforms the strongest baseline in only two cases, both involving TER for Hindi–Urdu translation directions. Despite higher TER values, TITAN simultaneously achieves stronger BLEU, ChrF2, and COMET scores on the same directions, indicating that the discovered translation circuits prioritize semantic fidelity over minimum-edit lexical reconstruction. For example, on U→H, TITAN improves BLEU from 25.78 to 27.01 and COMET from 77.37 to 81.28, while producing higher TER relative to the reference translation. Error patterns are strongest for Urduas-source directions, whereas the largest gains emerge for shared-script Hindi–Sindhi translation (+45.86% H→S and +48.98% S→H). This suggests that sparse circuit transfer is more effective under shared orthographic structure and becomes more challenging under Devanagari–Nastaliq script conversion.

## 8 Conclusion

We introduced TITAN, a mechanistic interpretability-guided framework for parameterefficient multilingual translation that identifies and optimizes sparse translation-crucial circuits instead of full-parameter adaptation. Experiments on Hindi–Urdu and Hindi–Sindhi show substantial improvements over multilingual baselines while updating fewer than 5% of parameters. Analyses further suggest that low-resource multilingual translation relies on sparse structured circuits within a shared semantic subspace rather than explicit English-centric pivot representations.

## 9 Ethics and Limitations

While TITAN performs strongly across lowresource and typologically diverse settings, broader validation across additional languages, scripts, and domains remains necessary. Although sparse circuit optimization reduces trainable parameters, causal intervention and sparse feature decomposition remain computationally expensive during circuit discovery. Occasional errors include untranslated loanwords, Unicode inconsistencies, and reduced fluency for long compound sentences. The current analysis also focuses on the identified translation circuits rather than the model’s complete multilingual representation space, limiting the scope of conclusions about global multilingual organization. TITAN does not guarantee factually or culturally perfect translations, particularly for rare expressions or specialized domains, and may inherit biases from its underlying pretraining and translation data. Our causal findings should therefore be interpreted within the evaluated models and language settings rather than as universal translation circuits. Further evaluation with broader domains, independent annotators, and additional language families would strengthen these conclusions. All datasets are publicly available research corpora, no personally identifiable information was collected, and human evaluation was conducted by bilingual experts. Real-world deployment should include appropriate human oversight and linguistic validation.

## Acknowledgments

The work is partially supported by a research project COIL-D at IIIT Delhi, funded by MeitY, Govt of India. The authors also acknowledge the support of the Infosys Foundation through Center of AI (CAI) at IIIT-Delhi.

## References

Duarte M. Alves, José Pombal, Nuno M. Guerreiro, Pedro H. Martins, João Alves, Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, Pierre Colombo, José G. C. de Souza, and André F. T. Martins. 2024. Tower: An open multilingual large language model for translation-related tasks. Preprint, arXiv:2402.17733.

Sudhansu Bala Das, Divyajyoti Panda, Tapas Kumar Mishra, Bidyut Kr. Patra, and Asif Ekbal. 2024. Multilingual neural machine translation for indic to indic languages. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 23(5).

Zhe Cao, Zhi Qu, Hidetaka Kamigaito, and Taro Watanabe. 2024. Exploring intrinsic language-specific subspaces in fine-tuning multilingual neural machine translation. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 21142–21157.

Tianyu Dong, Bo Li, Jinsong Liu, Shaolin Zhu, and Deyi Xiong. 2025. Mlas-lora: language-aware parameters detection and lora-based knowledge transfer for multilingual machine translation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15645–15660.

Jay Gala, Pranjal A Chitale, A K Raghavan, Varun Gumma, Sumanth Doddapaneni, Aswanth Kumar M, Janki Atul Nawale, Anupama Sujatha, Ratish Puduppully, Vivek Raghavan, Pratyush Kumar, Mitesh M Khapra, Raj Dabre, and Anoop Kunchukuttan. 2023. Indictrans2: Towards high-quality and accessible machine translation models for all 22 scheduled indian languages. Transactions on Machine Learning Research.

Xiao Liang, Yen-Min Jasmina Khaw, Soung-Yue Liew, Tien-Ping Tan, and Donghong Qin. 2025. Toward low-resource languages machine translation: a language-specific fine-tuning with lora for specialized large language models. IEEE Access, 13:46616– 46626.

Abhinav P M, Ketaki Shetye, and Parameswari Krishnamurthy. 2024. MTNLP-IIITH: Machine translation for low-resource Indic languages. In Proceedings of the Ninth Conference on Machine Translation, pages 751–755, Miami, Florida, USA. Association for Computational Linguistics.

Sarfraz Raza, FZ Agha, and R Usman. 2004. Phonemic inventory of sindhi and acoustic analysis of voiced implosives. Centerfor Research in Urdu language Processing (CRULP).

Karumuri Venkata Subbarao. 2012. South Asian languages: A syntactic typology. Cambridge University Press. A modern structural overview focusing on the high degree of syntactic convergence, grammar structures, and parametric variation across South Asian language families.

Chenxi Wang, Yixuan Zhang, Ruiji Yu, Yufei Zheng, Lang Gao, Zirui Song, Zixiang Xu, Gus Xia, Huishuai Zhang, Dongyan Zhao, and 1 others. 2025. Do llms" feel"? emotion circuits discovery and control. arXiv preprint arXiv:2510.11328.

Bin Wei, Zheng Jiawei, Zongyao Li, Zhanglin Wu, Jiaxin Guo, Daimeng Wei, Zhiqiang Rao, Shaojun Li, Yuanchang Luo, Hengchao Shang, Jinlong Yang, Yuhao Xie, and Hao Yang. 2024. Machine translation advancements of low-resource Indian languages by transfer learning. In Proceedings of the Ninth Conference on Machine Translation, pages 775–780, Miami, Florida, USA. Association for Computational Linguistics.

Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. 2024. A paradigm shift in machine translation: Boosting translation performance of large language models. Preprint, arXiv:2309.11674.

Hongbin Zhang, Kehai Chen, Xuefeng Bai, Xiucheng Li, Yang Xiang, and Min Zhang. 2026. Exploring the translation mechanism of large language models. Preprint, arXiv:2502.11806.

Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2024. Multilingual machine translation with large language models: Empirical results and analysis. Preprint, arXiv:2304.04675.

## Appendix

## A Translation Task Templates and Examples

## A.1 Dataset Construction

We construct contrastive translation datasets for two directions: Hindi→Sindhi (Hi→Sd, Devanagari) and Hindi→Urdu (Hi→Ur). The two directions dissociate factors that are typically conflated in multilingual circuit analysis. Hi→Sd involves no script boundary, thereby isolating cross-lingual semantic mapping from orthographic transduction while preserving a non-trivial transfer challenge through Sindhi-specific implosives and Persian– Arabic lexical borrowing. In contrast, Hi→Ur crosses a script boundary (Devanagari → Perso-Arabic Nastaliq) and, given the strong grammatical overlap between Hindi and Urdu, tests whether the identified circuit encodes semantic content independently of surface orthography.

Contrastive prompts are sampled from the SFT splits. We sample 10K pairs per direction.

## A.2 Prompt Format

The general template, introduced in Section 4.1 (Eq. 1), is:

$$
X ^ { + } = [ \mathsf { S R C } ] \colon [ \mathsf { t e x t } ] \ \to \ [ \mathsf { T G T } ] \colon\tag{12}
$$

where [SRC] and [TGT] are language-name tokens, and the trailing colon serves as the generation trigger and query position for attention analyses. We use the arrow glyph → because it tokenizes as a single token under the Gemma-2 SentencePiece tokenizer. By contrast, the hyphen (-) was empirically verified to fragment at the directive position, destabilizing query-position alignment.

## A.3 Contrastive Pair Construction

Each source sentence s yields one positive prompt $X ^ { + }$ and four counterfactual prompts $\{ X _ { k } ^ { - } \} _ { k = 1 } ^ { 4 }$ differing from $X ^ { + }$ only in the directive region. Construction follows two principles:

1. Structural isomorphism: identical tokenposition layout, punctuation, and sourcesentence span.

2. Targeted lexical substitution: a single semantic component of the directive is replaced.

Each perturbation type is instantiated with three lexical variants (e.g., Target Nullification: Nothing / Null / Void); reported results are averaged across variants.

Table 6 illustrates the four perturbation types with representative examples from both translation directions.

## A.4 Validation Protocol

Templates are verified along two axes: surface fluency and task-level disruption. Because the four perturbations alter the directive at different structural depths, a uniform fluency threshold is inappropriate; accordingly, we apply differentiated validation criteria.

Fluency-preserving perturbations (Semantic Obfuscation and Target Nullification) substitute a single target-slot token and are required to preserve surface fluency. Measured perplexity deviations relative to positive prompts confirm that these perturbations alter directive semantics without disrupting grammatical structure.

Structurally invasive perturbations (Paradox Insertion and Action Distortion) modify the directive at deeper structural levels and are therefore expected to induce higher perplexity by design. For these perturbations, validity is evaluated through task disruption rather than a fluency threshold.

For task-level disruption, all four perturbation types are required to yield zero translation accuracy on a held-out evaluation set, together with strong inter-annotator agreement between two annotators regarding whether each counterfactual successfully removes the translation directive. Templates failing the task-disruption criterion are discarded.

## A.5 Token Taxonomy

Each token position is assigned one of four functional roles for attention and SAE analyses (Section 4.2).

IND tokens form the circuit’s structural scaffold and draw spike attention from Indicator Heads. SRC tokens are the primary attention targets for Source Heads, while the TGT position serves as the readout site for logit-difference and unembedding analyses. For Hi→Sd, samples containing Sindhi-specific implosive phonemes are additionally tagged at the SRC level to support phonological circuit analysis.

<table><tr><td>Perturbation Type</td><td>Direction</td><td>X+ / X − Example</td><td>Ablated Component</td></tr><tr><td rowspan="4">Target Nullification</td><td>Hi→Sd</td><td>Hindi: ... → Sindhi:</td><td rowspan="2">Target language identity</td></tr><tr><td></td><td>Hindi: ... → Nothing:</td></tr><tr><td rowspan="2">Hi→Ur</td><td>Hindi: ... → Urdu:</td><td rowspan="2">Target language identity</td></tr><tr><td>Hindi: ... → Nothing:</td></tr><tr><td rowspan="4">Action Distortion</td><td rowspan="2">Hi→Sd</td><td>Hindi: ... → Sindhi:</td><td rowspan="2">Source-side instruction frame</td></tr><tr><td></td></tr><tr><td rowspan="2">Hi→Ur</td><td>Eat the following: ... → Sindhi:</td><td rowspan="2">Source-side instruction frame</td></tr><tr><td>Hindi: ... → Urdu: Eat the following: ... → Urdu:</td></tr><tr><td rowspan="4">Semantic Obfuscation</td><td rowspan="2">Hi→Sd</td><td></td><td rowspan="2">Translation-task semantics</td></tr><tr><td>Hindi: ... → Sindhi:</td></tr><tr><td rowspan="2">Hi→Ur</td><td>Hindi: ... → Color:</td><td rowspan="2">Translation-task semantics</td></tr><tr><td>Hindi: ... → Urdu: Hindi: ... → Color:</td></tr><tr><td rowspan="4">Paradox Insertion</td><td rowspan="2">Hi→Sd</td><td>Hindi: ... → Sindhi:</td><td rowspan="2">Coherent target slot</td></tr><tr><td>Hindi: ... → Silent Rock:</td></tr><tr><td rowspan="2">Hi→Ur</td><td>Hindi: ... → Urdu:</td><td rowspan="2">Coherent target slot</td></tr><tr><td>Hindi: ... → Silent Rock:</td></tr></table>

Table 6: Contrastive perturbation taxonomy with representative examples. Each $X ^ { - }$ alters a single semantic component of the directive while preserving structural isomorphism with $X ^ { + }$
<table><tr><td>Role</td><td>Tokens</td><td>Function</td></tr><tr><td>IND (Indicator)</td><td>Hindi, Sindhi, Urdu, →, :</td><td>Translation direction and structural scaf- fold</td></tr><tr><td>SRC (Source)</td><td>Content tokens of the source sentence Semantic payload</td><td></td></tr><tr><td>TGT (Target)</td><td>First generated token(s) at query posi- Translation output signal tion</td><td></td></tr><tr><td>RAND (Random)</td><td>Position-shuffled baseline</td><td>Control for positional effects</td></tr></table>

Table 7: Token role definitions used in attention and SAE analyses.

## B Experimental Setup

All Baseline fine-tuned models use identical training splits with LoRA rank 16 and batch size 8, with Gemma-2 9B serving as the underlying base model for CIRCE and TITAN. SFT models are trained for 3 epochs, while DPO optimization is performed for 2 epochs on NVIDIA A100 40GB GPUs. TITAN further incorporates Sparse Autoencoder decomposition with latent dimension 14336 for translationspecific feature discovery prior to targeted optimization.

## C Causal Validation via Progressive Knockout

In addition to the Hindi→Sindhi results presented in the main paper, Figure 2 presents the corresponding progressive knockout analysis for Hindi→Urdu translation. Across both translation directions, ablating CIRCE-identified components consistently produces substantially larger degradation than ablating randomly selected attention heads and MLP blocks, confirming that TITAN identifies causally meaningful translation circuits rather than incidentally activated neurons.

For Hindi→Urdu, removing the first key MLP reduces translation accuracy from 100% to approximately 87%, whereas random MLP ablation causes only marginal degradation. Similar to Hindi→Sindhi, attention-head ablation results in gradual performance decline, indicating distributed routing behavior across multiple heads, while MLP ablation causes sharper early-stage collapse, suggesting concentrated semantic transformation functions.

![](images/8e7618081c557b537c46127b822c9ee1685fb2843e652b3b644d26c00078fedf.jpg)  
Figure 2: Progressive knockout analysis for Hindi→Urdu translation. CIRCE-identified components produce substantially larger degradation than randomly selected components.

Across both language pairs, the gap between CIRCE-identified and random components widens monotonically as additional components are removed. Even after multiple interventions, randomly selected components preserve most translation capability, whereas CIRCE-identified circuits rapidly collapse toward near-chance performance. Together, these findings further support that multilingual translation behavior is highly localized within sparse transformer subcircuits.

## D Attention Routing Behavior

Figures 3 and 4 present the dominant CIRCEidentified attention heads for Hindi→Sindhi and Hindi→Urdu translation respectively. Across both language pairs, we observe a highly structured routing pipeline in which early-layer heads primarily attend to prompt separators and formatting boundaries, followed by mid-layer heads specializing in source-language identification through explicit language labels. Later layers progressively shift toward translation-direction routing and final targetlanguage commitment.

In both translation directions, the strongest causal heads emerge near the final generation layers, particularly around the terminal translation commitment stage. Notably, several heads consistently attend to language-indicator tokens rather than source content tokens, suggesting that routing decisions are strongly conditioned on explicit language identifiers. This behavior differs substantially from previously reported English-centric translation circuits, where attention heads predominantly align to source semantic content.

Although both translation directions exhibit similar routing structure, the Hindi→Urdu circuit displays broader and more distributed midlayer routing behavior compared to Hindi→Sindhi. We attribute this difference to the additional script-conversion complexity introduced by Urdu Nastaliq generation. Overall, these findings suggest that low-resource Indic translation relies heavily on label-aware routing mechanisms operating at prompt boundaries before semantic transformation occurs within downstream MLP representations.

## E MLP Representation Dynamics

Figures 5 and 6 present the layer-wise MLP representation dynamics for Hindi→Sindhi and Hindi→Urdu translation respectively. To analyze semantic processing within the translation circuit, we project MLP hidden states onto source-language, target-language, and translationindicator token embeddings using cosine similarity.

Across both translation directions, sourcelanguage alignment gradually increases through middle layers and peaks near the later network stages, indicating progressive encoding of sourcelanguage semantics within MLP representations. In contrast, target-language alignment remains weak for most layers before sharply increasing near the final generation layers, suggesting that multilingual translation postpones target-language realization until late-stage decoding.

We further analyze whether translation proceeds through an implicit English pivot, as suggested in prior multilingual interpretability studies. Projecting MLP transformation vectors onto English, Hindi, Urdu, and Sindhi representation directions reveals consistently negative English alignment across nearly all layers for both language pairs. This strongly contradicts previously reported English-centric multilingual pivot mechanisms.

Instead, the dominant latent alignment emerges along shared Indic semantic directions, particularly Hindi-centered representations, suggesting that closely related Indic translation directions operate through a shared Hindustani semantic subspace rather than through any explicit pivot language. In this shared latent space, semantic computation becomes approximately orthogonal to explicit language identifiers, enabling language-neutral processing prior to final target-language commitment.

![](images/7b9d8780ee14135ffb7e70603298531f59bc8885b17da74f99ba6c419eeb73b5.jpg)

![](images/66d9d7c105c08e5358a247e1f59ffe29a23a5633a30e4422431c8167d7dc7aa1.jpg)

![](images/d6df56a912d67c8776a1d3f873f06be6eaea52b41d7cc54a74395db61d887fb9.jpg)

![](images/79279ff7909e8929fc73f61cd1925623800fc5ce924146f733985093476965cd.jpg)  
Figure 3: Dominant CIRCE-identified attention heads for Hindi→Sindhi translation. Different heads specialize in structural anchoring, source-language identification, translation-direction routing, and target-language commitment.

![](images/e2591663cd23f7dc4c8f059c01a6abfa07beb05c5f2f7d17980585b260b160c2.jpg)

![](images/b8f368c11e3e66c3c5641ec2a6932b17a91d3b25cc7a0fac20b42d18ed0e3ddd.jpg)

![](images/ee722f4f0b9886aa780ca30e0983f9b9bc476af153edbd1ca8ccc8133c22fc3a.jpg)

![](images/3b4d591d5b3e24b90af8ecee676ee36e1fde452d44ae19327a6d545c64e963d7.jpg)  
Figure 4: Dominant CIRCE-identified attention heads for Hindi→Urdu translation. Compared to Hindi→Sindhi, the Urdu circuit exhibits broader mid-layer routing behavior associated with script-conversion complexity.

## F Sparse Feature Behavior

Figures 7 and 8 present the layer-wise Sparse Autoencoder (SAE) feature distributions for Hindi→Sindhi and Hindi→Urdu translation respectively. To analyze translation-relevant representations, we classify SAE features jointly using selectivity and causal-lift statistics in order to distinguish routing-oriented features from causally important translation features.

Across both language pairs, we observe strongly layer-localized feature distributions in which targetlanguage-specific features dominate earlier layers while Hindi-specific features emerge more prominently in later layers. Between these regions lies a middle-layer transition zone where neither language exhibits strong dominance, consistent with the shared latent-space behavior observed in the MLP analysis.

and MLP analyses yields a unified mechanistic interpretation of low-resource Indic translation. Attention heads primarily implement languageaware routing and prompt-boundary coordination, whereas MLP layers perform semantic transformation within a shared Hindustani latent subspace. Translation behavior therefore emerges through sparse routing circuits coupled with languageneutral semantic computation rather than through explicit pivot-language translation.

Importantly, highly selective features do not necessarily exhibit strong causal influence. Most language-specific features produce minimal causal lift despite strong selectivity scores, indicating that activation specialization alone is insufficient to explain translation behavior. Instead, translation is governed by a sparse subset of high-lift features distributed across both source and target language populations.

Combining the SAE findings with the attention

## Hi→Sd: MLP Behavioral Analysis across 42 Layers

§5.2a Reception Trend — MLP inputs vs. translation tokens  
![](images/162de9c42b5f19176ec3e3274872a3b9fcc6d430e3b5f7a10322ddd95ee2d584.jpg)

§5.2b Generation Trend — hidden state alignment with target token  
![](images/51184cc8e0ba428a3b75cb10c263257b611e068a45fb13e6c4ab1327f552bca2.jpg)

§5.2c Pivot Analysis — Latent language representation (Hi→Sd)  
![](images/220de1d7fccde398964070608f6cf8fbd81083febdde7ec08099be6029459df8.jpg)  
Figure 5: MLP representation dynamics for Hindi→Sindhi translation. Top: source-language reception behavior. Middle: target-language generation behavior. Bottom: latent pivot analysis across Hindi, Sindhi, and English representation directions.

Hi→Ur: MLP Behavioral Analysis across 42 Layers §5.2a Reception Trend — MLP inputs vs. translation tokens  
![](images/718ab2112b96a9f0dd14a2fc8053bc796b0f5440815fc55a62a4c179f3f704f3.jpg)

§5.2b Generation Trend — hidden state alignment with target token  
![](images/e4bb8edd4cc44ba29d19a6494711227d2bdcb73ef28e2e6a5106ecaaaae1cfe8.jpg)

§5.2c Pivot Analysis — Latent language representation (Hi→Ur)  
![](images/cdca7c092087b63b6368377d786598274825614dedeb6a6d8a8347d102f896eb.jpg)  
Figure 6: MLP representation dynamics for Hindi→Urdu translation. Similar to Hindi→Sindhi, translation behavior evolves through shared Indic semantic directions rather than explicit English pivots.

(A) Hindi vs Sindhi Feature Count per Layer  
![](images/b00efa4244a7e4fbe4781fcd33717aae22426b35ff64e8d0728c6f8c974dea2d.jpg)

(B) Mean Selectivity per Layer (↑ Hindi-biased · ↓ Sindhi-biased)  
![](images/c0027388740ff0dab9832ddd2768d699e9713b84a6e0f2827704092ef8d4634d.jpg)

(C) Selectivity × Lift Feature Space (both dimensions used for classification)  
![](images/59b5d42186fb8127dfa3a565bd805a3ce27a24674ceee9bff409579b3b6656a1.jpg)  
Figure 7: Sparse feature analysis for Hindi→Sindhi translation. Top: layer-wise distribution of Hindi-specific and Sindhi-specific SAE features. Middle: mean feature selectivity across layers. Bottom: joint selectivity–lift feature space highlighting sparse causally relevant translation features.

## Hi→Ur: Hindi vs Urdu SAE Feature Comparison Across Layers

(A) Hindi vs Urdu Feature Count per Layer  
![](images/01880bbd817af56b7a8dbd724baf75112ff565cc3537c8b4444ccce3000187ff.jpg)

(B) Mean Selectivity per Layer (↑ Hindi-like, ↓ Urdu-like)  
![](images/83bc9c7c699f1da19ef75e06a1543909f52b36f2a12e9f2ddd14d3ca31d0e8ad.jpg)

(C) Selectivity × Lift Feature Space (subsampled 500 features/layer; p99 threshold lines shown)  
![](images/934c02b86446a7a511e8479502e91cf78bc6acca875ca7f171adb967434f8b67.jpg)  
Figure 8: Sparse feature analysis for Hindi→Urdu translation. Similar to Hindi→Sindhi, causally important translation behavior is concentrated within a sparse subset of high-lift features.