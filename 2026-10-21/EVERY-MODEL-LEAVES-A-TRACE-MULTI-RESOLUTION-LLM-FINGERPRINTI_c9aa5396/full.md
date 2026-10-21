# EVERY MODEL LEAVES A TRACE: MULTI-RESOLUTION LLM FINGERPRINTING

Mohammad Reza Modarres<sup>\*1,2</sup> Armin Tourajmehr<sup>\*2</sup> Artin Tavasoli<sup>1</sup> Yadollah Yaghoobzadeh<sup>†1,2</sup> Mohammad Taher Pilehvar<sup>†3</sup>

<sup>1</sup>School of Electrical and Computer Engineering, College of Engineering, University of Tehran, Tehran, Iran <sup>2</sup>Tehran Institute for Advanced Studies, Khatam University, Iran <sup>3</sup>Cardiff University, United Kingdom

mr.modarres@ut.ac.ir, a.tourajmehr@teias.institute artin.tavassoli@ut.ac.ir, y.yaghoobzadeh@ut.ac.ir pilehvarmt@cardiff.ac.uk

<sup>∗</sup>Equal contribution. †Corresponding authors.

## ABSTRACT

Large language models leave model-specific traces in their generated text, but it remains unclear how finely these fingerprints can distinguish related models and which signals remain informative as obvious cues are reduced. We study this through fingerprint resolution, considering both model resolution, from different families to distilled and adapted variants, and signal resolution across behavioral, stylometric, lexical, and semantic evidence. We progressively suppress surface cues, paraphrase outputs, vary the amount of available text, and test generalization across prompt distributions and languages. Our results show that fingerprints persist beyond broad family differences, remaining detectable in distilled models and even LoRA adaptations of the same base model. Although attribution weakens as surface and lexical cues are altered, substantial model-specific signal remains. These signals transfer across prompt sources and languages, with semantic representations proving most robust across languages. These findings show that model identity is recoverable at multiple resolutions, but that recoverability depends on model similarity and the evidence available in the text.

## 1 Introduction

LLMs often leave recognizable traces in their generated text, making it possible to distinguish among different source models (Uchendu et al., 2020; Rao et al., 2025; Antoun et al., 2024). Some traces are easy to observe when a model generates text, such as producing longer responses, using more headings, following particular formatting conventions, or beginning answers in recurring ways. Such signals can make attribution relatively easy, but they are also fragile, and post-generation editing can weaken textbased detectors (Krishna et al., 2023; Dugan et al., 2024). What remains unclear is which model-specific signals survive when these visible cues are reduced, especially when the candidate models are closely related. We therefore ask: how finely can the identity of an LLM be resolved from its generated text, and which properties ofthe text still reveal that identity?

This question is increasingly important as modern LLMs become more diverse and increasingly appear as closely related model families rather than isolated systems. Different sizes, distilled descendants, and adapted variants may share architectures, training histories, or development processes while differing only in subtle ways, making source attribution more difficult. At the same time, generated text may be reformatted, edited, shortened, or paraphrased before attribution, weakening or removing model-identifying signals. Attribution must therefore address two related challenges by distinguishing increasingly similar models and determining which identifying signals remain observable after the text is transformed.

![](images/e2d274c9bb6a7ac432799fbc9acc9606db83cf59141bb62246b68249344b50c8.jpg)  
Figure 1: Overview of the fingerprint-resolution framework, linking model similarity, observable evidence, and generalization across prompts, languages, and text budgets.

We study this problem through fingerprint resolution, illustrated in Figure 1. Model resolution describes how attribution changes as candidate generators become more similar, from different families to within-family models, distilled descendants, and controlled LoRA (Hu et al., 2022) adaptations. Signal resolution describes which behavioral, stylometric, lexical, and semantic signals carry model identity and whether they survive as observable cues are reduced. Together, these complementary axes distinguish the existence of a fingerprint from the conditions under which it remains detectable.

Prior work has shown that source models can be identified using lexical and grammatical patterns, learned representations, and supervised attribution models (McGovern et al., 2025; Sun et al., 2025; Lu et al., 2026). Our goal is to use model attribution as a diagnostic tool rather than treat accuracy as the endpoint. We progressively remove observable cues, paraphrase model outputs, and control the amount of available text to determine which behavioral, stylometric, lexical, and semantic signals continue to reveal model identity. We further test whether fingerprints generalize beyond the conditions under which they are learned. Source-held-out experiments measure transfer across prompt distributions, while within- and cross-language experiments evaluate English, Chinese, and Persian. These settings address evidence that source attribution may weaken across languages (La Cava et al., 2026) and that fingerprint visibility can depend on language (Gloaguen et al., 2026).

Our results show that model fingerprints extend beyond broad family differences and remain detectable even between distilled models and their teachers, as well as between LoRA-adapted versions of the same base model. Attribution becomes harder as behavioral and stylometric cues are removed and lexical choices are altered through paraphrasing. Cross-lingual experiments show that semantic representations provide the most reliable transfer across languages, while cross-dataset results show substantial transfer across prompt sources but sharp degradation for highly constrained generation tasks. Model identity is therefore recoverable at different resolutions depending on model similarity and the evidence available in generated text.

Our key contributions are threefold: (1) A fingerprint-resolution framework spanning different levels of model similarity, revealing distinguishable fingerprints even in distilled models and LoRAadapted variants of the same base model; (2) A decomposition of model-identifying evidence using multiple signal families, progressive cue suppression, paraphrasing, and varying the amount of available text; and (3) A broad evaluation of fingerprint generalization across unseen prompt distributions, different generation tasks, and within- and cross-language settings in English, Chinese, and Persian.

## 2 Related Work

Prior work has shown that LLM-generated text contains model-specific signals that can be used to identify its source. These signals appear at different levels, from broad differences associated with model family and size (Antoun et al., 2024) to finer patterns in vocabulary, character use, and sentence structure that can persist across domains (McGovern et al., 2025). Other approaches use learned representations, contrastive learning, or specialized attribution models to improve source classification (Sun et al., 2025; Bitton et al., 2025; Lu et al., 2026; Guo et al., 2024; La Cava and Tagarelli, 2025). Distinguishing signals can even emerge among closely related models that differ only in training details such as initialization or data order (Suzuki et al., 2025; Tong et al., 2026). Together, these studies establish that model identity is often recoverable from generated text. Less clear is how that recoverability changes as candidate models become more similar and as the available identifying cues are progressively removed.

Several studies have also examined the robustness of model attribution. Some control for response length and formatting or apply transformations such as translation and summarization (Sun et al., 2025; McGovern et al., 2025). Related detection benchmarks show that robustness can deteriorate when outputs come from previously unseen generators (Li et al., 2024; Wang et al., 2024). Ownership-oriented methods take a different approach by implanting private trigger-response behaviors or probing models with selected prompts and repeated observations (Xu et al., 2024; Pasquini et al., 2025; Gao et al., 2025). Our setting is complementary. Rather than optimizing prompts or attribution models, we study passive attribution from ordinary model outputs and ask how much identifying information remains when text is shortened, reformatted, partly rewritten, or paraphrased. This motivates our view of fingerprinting as a resolution problem, determined jointly by the similarity of the candidate models and the amount and type of evidence that remains observable.

A smaller body of work considers multilingual attribution. Recent results suggest that source-model attribution does not always transfer reliably across languages (La Cava et al., 2026; Greco et al., 2026), while semantically conditioned watermarking shows that language can be used as a channel for embedding model-specific signals (Gloaguen et al., 2026). In contrast, we focus on naturally occurring model fingerprints rather than deliberately embedded signals, and examine which behavioral, lexical, and semantic cues remain informative within and across English, Chinese, and Persian.

## 3 Experimental Setup

As discussed in Section 2, most prior work frames attribution as a closed-set classification task in which a generated response is assigned to one of several predefined candidate models and performance is summarized by a single score. While this formulation establishes whether model-specific signals are recoverable, it provides limited insight into how attribution changes as candidate models become increasingly similar or which properties of the generated text support the prediction. These questions matter because modern LLMs often appear as closely related variants and because their outputs may be altered before attribution. We study these factors throughfingerprint resolution.

In this section, we first formalize attribution as a measure of fingerprint recoverability and then define model resolution (Sec. 3.1), signal representations (Sec. 3.2), and text transformations (Sec. 3.3) used to study both axes of fingerprint resolution. We then study these two axes jointly in Sec. 4 and examine whether the resulting fingerprints generalize across prompt distributions and languages in Sec. 5. Detailed generation, feature, and classifier settings are provided in Appendix A.

Problem Formulation. Let $\mathcal { M } = \{ M _ { 1 } , . . . , M _ { K } \}$ denote the set of candidate generators. Given a response $y ,$ the attribution task predicts its source model as $\hat { M } \ : = \ : f ( y )$ . Rather than treating the performance of $f$ as the endpoint, we use it as a measure of fingerprint recoverability. We examine how attribution changes along two complementary axes: model resolution, which varies how closely related the candidate generators are, and signal resolution, which varies the type and amount of modelidentifying evidence available in their outputs.

## 3.1 Model Resolution

Model resolution is examined across four levels of similarity between candidate generators. Our primary model set contains 19 models from eight families and supports the cross-family and within-family experiments. Additional models are used for the distillation and controlled-adaptation settings. Complete model lists and exact identifiers are provided in Appendix A.1. Across settings, we keep the attribution protocol as consistent as possible so that differences in performance primarily reflect the similarity between candidate models.

• Cross-Family. We select one representative from each of the eight families: Llama-4 Maverick, Qwen-3 32B, Mistral-Small-3.2 24B, Gemma-4 31B, DeepSeek-V4 Flash-0731, Claude Haiku 4.5, Gemini-2.5 Flash, and GPT-4.1.

• Within-Family. We compare all model pairs within families containing multiple models. This gives six Llama pairs, three Qwen pairs, and one pair each for other model families.

• Distillation Relationships. We examine three teacher–student pairs: Llama-3.1 8B vs. Llama-3.1-Minitron 4B, Mistral-NeMo 12B vs. Mistral-NeMo-Minitron 8B, and Gemma-2 27B vs. Gemma-2 2B.

• Controlled Adaptations. We use Llama-3.1-8B-Instruct and Qwen3-8B as base models and finetune each on GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) using LoRA. We then vary the strength of the learned LoRA update to create a controlled sequence of adaptations and test when their outputs become distinguishable from those of the corresponding base model. The fine-tuning procedure and LoRA settings are described in Appendix F.

## 3.2 Signal Resolution: From Surface to Semantics

Signal resolution covers four complementary levels of model-identifying signals, ranging from surfacelevel behavioral patterns to semantic representations.

• Behavioral and Structural. These features capture how a model organizes and presents its response. They cover response length and structure, formatting and layout, punctuation, and repetition tendency. Representative examples include sentence and paragraph counts, headings and lists, punctuation usage, and bigram and trigram repetition ratios.

• Stylometric. These features capture recurring patterns in vocabulary use and grammatical structure. They cover vocabulary diversity, word-form statistics, vocabulary distribution, function and grammatical words, repetition, and the ratio of numeric tokens. We concatenate 17 dense lexical features with POS (part-of-speech) unigram, bigram, and trigram TF–IDF representations, inspired by prior attribution work (McGovern et al., 2025).

• Lexical. To capture lexical form directly, we use TF–IDF representations at both the word and character levels. The word-level representation uses unigrams and bigrams, while the characterlevel representation uses 3–5-grams. We concatenate these two representations into a single lexical feature set.

• Semantic. We use openai/text-embedding-3-small to obtain a fixed embedding for each response. This provides a semantic-level view that can capture model-specific information not explicitly represented by the manually defined or surface-form features.

Two additional derived representations are considered. First, for the prompt-neutralized embedding, we compute the mean embedding across all candidate-model responses to the same prompt and subtract this mean from each response embedding. This removes variation shared across models for that prompt and emphasizes differences between their responses. Because this setting requires peer responses from all candidate models for the same test prompt, it differs from standard attribution and is used as a diagnostic representation. If attribution improves after neutralization, it suggests that removing prompt- and task-shared variation exposes model-specific signals that are more discriminative and less dependent on the prompt itself. Second, the combined representation concatenates the four main signal groups described above to capture their complementary information. Complete feature definitions, vectorizer settings, and dimensionality limits are provided in Appendix A.4.

<table><tr><td></td><td>View Transformation</td></tr><tr><td> $T _ { 0 }$ </td><td>Raw response</td></tr><tr><td> $T _ { 1 }$ </td><td>Remove words from the beginning and end</td></tr><tr><td> $T _ { 2 }$ </td><td> $T _ { 1 } + \mathrm { r e m o v e }$  formatting and layout cues</td></tr><tr><td> $T _ { 3 }$ </td><td> $T _ { 2 } +$  lowercase and remove punctuation, symbols, and emojis</td></tr><tr><td> $P _ { 1 }$ </td><td>Paraphrase approximately one third of eligible sentences in  $T _ { 0 }$ </td></tr><tr><td> $P _ { 2 }$ </td><td>Paraphrase the full  $T _ { 0 }$  response</td></tr><tr><td> $P _ { 1 } ^ { * }$ </td><td>Apply  $T _ { 3 }$  transformation to  $P _ { 1 }$ </td></tr><tr><td> $P _ { 2 } ^ { * }$ </td><td>Apply  $T _ { 3 }$  transformation to  $P _ { 2 }$ </td></tr></table>

Table 1: Response views used to progressively remove or rewrite model-identifying cues.

<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>52.42</td><td rowspan=1 colspan=1>60.72</td><td rowspan=1 colspan=1>80.07</td><td rowspan=1 colspan=1>66.04</td><td rowspan=1 colspan=1>81.41</td><td rowspan=1 colspan=1>83.75</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>48.78</td><td rowspan=1 colspan=1>58.93</td><td rowspan=1 colspan=1>76.27</td><td rowspan=1 colspan=1>60.81</td><td rowspan=1 colspan=1>76.14</td><td rowspan=1 colspan=1>76.13</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>41.88</td><td rowspan=1 colspan=1>56.34</td><td rowspan=1 colspan=1>72.48</td><td rowspan=1 colspan=1>56.23</td><td rowspan=1 colspan=1>71.86</td><td rowspan=1 colspan=1>74.42</td></tr><tr><td rowspan=2 colspan=1>T3P1</td><td rowspan=1 colspan=1>25.59</td><td rowspan=1 colspan=1>49.62</td><td rowspan=1 colspan=1>67.76</td><td rowspan=1 colspan=1>50.94</td><td rowspan=1 colspan=1>67.04</td><td rowspan=1 colspan=1>69.35</td></tr><tr><td rowspan=1 colspan=1>50.86</td><td rowspan=1 colspan=1>58.20</td><td rowspan=1 colspan=1>74.42</td><td rowspan=1 colspan=1>63.69</td><td rowspan=1 colspan=1>77.85</td><td rowspan=1 colspan=1>80.90</td></tr><tr><td rowspan=2 colspan=1>P2P1*</td><td rowspan=1 colspan=1>48.12</td><td rowspan=1 colspan=1>53.32</td><td rowspan=1 colspan=1>68.18</td><td rowspan=1 colspan=1>57.81</td><td rowspan=1 colspan=1>71.86</td><td rowspan=1 colspan=1>75.23</td></tr><tr><td rowspan=1 colspan=1>24.48</td><td rowspan=1 colspan=1>46.51</td><td rowspan=1 colspan=1>61.01</td><td rowspan=1 colspan=1>47.51</td><td rowspan=1 colspan=1>61.47</td><td rowspan=1 colspan=1>64.99</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>23.08</td><td rowspan=1 colspan=1>41.79</td><td rowspan=1 colspan=1>53.94</td><td rowspan=1 colspan=1>41.90</td><td rowspan=1 colspan=1>55.87</td><td rowspan=1 colspan=1>59.39</td></tr></table>

Table 2: Cross-family attribution accuracy (%) using logistic regression. Chance accuracy is 12.5%. <sup>†</sup>PN Embedding denotes the Prompt-Neutralized Embedding accuracy. This representation requires access to peer responses from all candidate models for the same prompt. We use “PN Embedding” and this symbol consistently throughout the paper.

## 3.3 Transformations

Attribution is evaluated as candidate models become more similar and as identifying signals are progressively weakened. Table 1 summarizes the response views used throughout the experiments. $T _ { 1 } \mathrm { - } T _ { 3 }$ form a cumulative signal-removal sequence, while $P _ { 1 }$ and $P _ { 2 }$ are generated directly from the raw response through LLM-based partial and full paraphrasing, respectively. We then apply $T _ { 3 }$ to $P _ { 1 }$ and $P _ { 2 }$ to obtain $P _ { 1 } ^ { * }$ and $P _ { 2 } ^ { * }$ , allowing us to test paraphrased text after the strongest surface normalization. Full transformation and paraphrasing details are provided in Appendix A.5.

## 3.4 Datasets and Evaluation

To make the attribution setting more realistic, we use open-ended prompts and sample responses at temperature 1.0, producing substantial generations; approximately 95% of responses exceed 50 words. The main English experiments use four prompt sets, $D _ { 1 } { - } D _ { 4 }$ , with 200 prompts from each source before filtering. $D _ { 1 }$ and $D _ { 2 }$ are independent samples from INFINITY-CHAT (Jiang et al., 2026), $D _ { 3 }$ contains researcher-designed prompts spanning five task categories (construction described in Appendix A.2), and $D _ { 4 }$ comes from a separate general-prompt source (Rajani et al., 2023). After removing one nearduplicate, the final set contains 799 prompts. All models follow a common decoding protocol (Appendix A.3). The multilingual experiments in Sec. 5 extend this setup to English, Mandarin Chinese, and Persian by translating the English prompts with GPT-4o-mini.

For attribution, we use Logistic Regression (LR) as the primary diagnostic classifier and a onehidden-layer MLP as a nonlinear robustness check. The main English experiments use five-fold, promptgrouped, dataset-stratified cross-validation, keeping all outputs and response views associated with the same prompt within a single fold. We reuse the same fold assignments across experiments and report accuracy. Classifier settings and evaluation details are provided in Appendix A.6.

## 4 Resolving Model Fingerprints

This section presents the results of the model- and signal-resolution experiments. Attribution is evaluated across different signal types while observable signals are progressively reduced to test whether model fingerprints persist. The analysis also spans increasing levels of model similarity, from distinct model families to closely related adaptations of the same base model. Neither LR nor the MLP consistently outperforms the other across settings. We therefore report LR results throughout the main text and provide the corresponding MLP results in the Appendices B and D.

Across Model Families. Table 2 presents the eight-way cross-family attribution results, with a chance level of 12.5%. The results show that attribution relies on more than formatting signals and response templates. Behavioral features show the largest relative decline under edge cropping (T1) and formatting removal (T2), consistent with their direct encoding of response structure, while stylometric and lexical representations remain more stable. Even after T3 removes formatting, capitalization, punctuation, symbols, and emojis, the lexical representation remains the strongest individual signal group. This suggests that model-specific word choices and local lexical patterns persist after visible surface signals are suppressed. Embedding accuracy also declines under T1–T3, indicating that frozen representations capture some of this affected lexical and stylistic information rather than semantic content alone.

Paraphrasing produces a complementary pattern. Accuracy decreases from P1 to P2 as more of the original text is rewritten, and falls further in P1<sup>∗</sup> and P2<sup>∗</sup> after applying T3. Combined performs best on the raw and paraphrased views, whereas the Lexical representation slightly exceeds it under T1–T3. This shift is consistent with paraphrasing weakening lexical realization while T3 removes additional structural and surface signals. Even under P2<sup>∗</sup>, all representations remain above chance, with Combined reaching 55.87%. Substantial cross-family information therefore remains recoverable after both lexical realization and visible surface form are heavily altered.

The Prompt-Neutralized Embedding (PN Embedding) achieves the highest accuracy across all views. We report it separately because it is a diagnostic, prompt-relative representation rather than a standard standalone classifier input. Unlike the other representations, PN Embedding requires peer responses from all candidate models to the same prompt so that their shared embedding mean can be removed. Its strong performance suggests that removing prompt- and task-shared variation exposes model-specific signals that are more discriminative and less dependent on the prompt itself. This finding also motivates the analysis of generalization across prompt distributions in Sec. 5. Corresponding results for the MLP classifier are provided in Appendix B. Across the reported settings, the largest standard deviation is 3.2%, indicating relatively low variability. To avoid overloading the tables, we omit standard deviations from the main presentation.

Within Model Families. Table 3 shows that model identity remains detectable among closely related Llama models, although accuracy is substantially lower than in the cross-family setting. The Combined representation performs best for most pairs, suggesting that distinctions between related models draw on multiple complementary signal groups. The Lexical representation is generally the strongest individual signal group, while behavioral and embedding representations provide weaker separation. Pairwise difficulty is not determined by parameter count alone: differences across model generations are often easier to detect than size differences within the same generation. Overall, within-family attribution appears to rely on several weak but complementary signals rather than a single dominant fingerprint. Full transformation results for the Llama within-family pairs are provided in Appendix C.

<table><tr><td rowspan=1 colspan=7>Model Pair              Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>3.1-70B vs. 3.3-70B</td><td rowspan=1 colspan=1>60.33</td><td rowspan=1 colspan=2>59.45      59.76</td><td rowspan=1 colspan=1>59.70</td><td rowspan=1 colspan=1>66.40</td><td rowspan=1 colspan=1>73.34</td></tr><tr><td rowspan=1 colspan=1>3.1-70B vs. 4-Maverick</td><td rowspan=1 colspan=1>60.45</td><td rowspan=1 colspan=1>63.33</td><td rowspan=1 colspan=1>69.02</td><td rowspan=1 colspan=1>60.51</td><td rowspan=1 colspan=1>72.09</td><td rowspan=1 colspan=1>76.35</td></tr><tr><td rowspan=1 colspan=1>3.1-8B vs. 3.1-70B</td><td rowspan=1 colspan=1>55.38</td><td rowspan=1 colspan=1>58.26</td><td rowspan=1 colspan=1>59.01</td><td rowspan=1 colspan=1>55.38</td><td rowspan=1 colspan=1>61.39</td><td rowspan=1 colspan=1>63.33</td></tr><tr><td rowspan=1 colspan=1>3.1-8B vs. 3.3-70B</td><td rowspan=1 colspan=1>57.51</td><td rowspan=1 colspan=1>61.51</td><td rowspan=1 colspan=1>60.76</td><td rowspan=1 colspan=1>58.64</td><td rowspan=1 colspan=1>64.27</td><td rowspan=1 colspan=1>68.21</td></tr><tr><td rowspan=1 colspan=1>3.1-8B vs. 4-Maverick</td><td rowspan=1 colspan=1>56.32</td><td rowspan=1 colspan=1>62.83</td><td rowspan=1 colspan=1>68.59</td><td rowspan=1 colspan=1>58.57</td><td rowspan=1 colspan=1>70.96</td><td rowspan=1 colspan=1>69.71</td></tr><tr><td rowspan=1 colspan=1>3.3-70B vs. 4-Maverick</td><td rowspan=1 colspan=1>58.64</td><td rowspan=1 colspan=1>61.64</td><td rowspan=1 colspan=1>68.02</td><td rowspan=1 colspan=1>57.76</td><td rowspan=1 colspan=1>67.58</td><td rowspan=1 colspan=1>73.22</td></tr></table>

Table 3: Within-family attribution accuracy (%) for Llama model pairs using logistic regression. Each row is a binary attribution task, with chance accuracy of 50%.
<table><tr><td>Teacher → Student</td><td>Behavioral</td><td>Stylometric</td><td>Lexical</td><td>Semantic</td><td>Combined</td><td>PN Embedding†</td></tr><tr><td>Llama 8B → Minitron 4B</td><td>66.67</td><td>69.49</td><td>71.24</td><td>70.68</td><td>74.56</td><td>80.58</td></tr><tr><td>Mistral 12B → Minitron 8B</td><td>99.31</td><td>99.44</td><td>99.12</td><td>99.62</td><td>99.62</td><td>99.87</td></tr><tr><td>Gemma 27B → Gemma 2B</td><td>69.92</td><td>67.98</td><td>69.80</td><td>72.74</td><td>77.94</td><td>88.47</td></tr></table>

Table 4: Teacher–student attribution accuracy (%) using logistic regression. Each row is a binary classification task with a chance level of 50%.

The same broad pattern extends to the other model families, although fingerprint strength varies considerably across them. Lexical or Combined representations are typically strongest, while behavioral and embedding representations alone are less reliable. Full results are reported in Appendix D.

Teacher–Student Attribution. Table 4 shows that distilled students remain readily distinguishable from their teachers. Compared with the standard within-family comparisons in Tables 3 and 15, the teacher–student pairs are similarly or more separable. The Llama teacher–student pair is more separable than any of the Llama pairs, while the Mistral pair is nearly perfectly separable. The Gemma pair shows performance comparable to the corresponding within-family comparison. The Combined representation performs best for Llama and Gemma, whereas nearly all representations strongly separate the Mistral pair. PN Embedding provides additional gains, particularly for Llama and Gemma. Overall, shared model lineage does not imply similar fingerprints. The changes introduced through distillation can leave a strong and detectable model-specific signature. Full transformation results for each teacher–student pair are provided in Appendix E.

Controlled LoRA Resolution. To study the finest model-resolution setting under controlled proximity, we train one LoRA adapter for each base model and adaptation dataset and vary only its contribution at inference time. For a fixed learned update $\Delta W _ { \mathrm { L o R A } }$ , each variant has effective weights $W _ { \alpha } = W _ { 0 } + \alpha \Delta W _ { \mathrm { L o R A } }$ , with $\alpha \in \{ 0 , . 0 5 , . 1 0 , . 2 0 , . 4 0 , . 6 0 , . 8 0 , 1 . 0 \}$ . We use Llama-3.1-8B-Instruct and Qwen3-8B adapted separately on GSM8K and MATH. The vs base setting compares each nonzero strength with $\alpha = 0$ and measures separation from the original model, while vs previous compares each strength with the immediately preceding one and measures how finely neighboring variants can be distinguished. Full details are provided in Appendix F.

Figure 2 shows a clear gap between these two settings. As LoRA strength increases, adapted models become progressively easier to distinguish from their bases, while neighboring variants remain much harder to separate. For Llama, vs base accuracy rises steadily, whereas vs previous remains close to chance across most strengths. Qwen shows stronger separation overall and clearer differences between some neighboring variants, especially at intermediate values of α; these local differences decrease again at higher strengths even as separation from the base continues to increase. Thus, increasing α does not produce a uniform increase in observable fingerprint change across models or adaptation tasks.

![](images/c88cfd3ca7eba3a89973f730edb919a1960ce3f5004ed6bbe8ed582da2d5ce32.jpg)

![](images/d10fb0d8035c8329821e36ca8a3a3b6b33c6e27b125f1f4886a77cff0aac62fc.jpg)  
(a) Llama-3.1-8B-Instruct

![](images/fdaabdb819950d64e0c135c1bb150defdf1878155c4612fc3b7df27fda04c83d.jpg)  
(b) Qwen3-8B  
Figure 2: Controlled LoRA fingerprint resolution using Combined logistic-regression accuracy (%). vs base compares each strength with the unadapted model, and vs previous with the preceding strength. Chance is 50%.

Table 20 shows the same overall pattern across Behavioral, Stylometric, Lexical, and Semantic representations. Neighboring Llama variants remain near chance across signal groups, while Qwen retains a modest local signal. Overall, adaptation can make a model increasingly distinguishable from its base even when successive variants remain difficult to resolve.

How Much Evidence Is Needed? Figure 3 examines how attribution changes with the amount and location of available text under P2<sup>∗</sup>, our strongest transformation. Even 50 words retain useful identifying information, with accuracy of approximately 22% for the beginning and center and 27% for the ending, compared with a 12.5% chance level. Accuracy increases steadily with the amount of available text and reaches about 50% for the full response.

The ending is the most informative region at every limited word budget, suggesting that later parts of a response retain more model-specific information after paraphrasing and surface normal-

![](images/e9f9c86b91e3d831266914e3d7865c44cf9c31c736f820c0703afaf638dd84ae.jpg)  
Figure 3: Cross-family attribution under P2<sup>∗</sup> across word budgets and text positions.

ization. The corresponding T0 and T3 analyses in Appendix G show the same accumulation pattern at higher accuracy levels. Overall, attribution depends on both the amount of available text and the extent to which its identifying signals have been altered.

## 5 Do Fingerprints Generalize?

We next test whether model fingerprints generalize beyond the conditions under which they are learned. We consider two complementary shifts, changes in prompt distribution and changes in language, to assess whether attribution relies on signals that persist across prompts and languages.

(a) Cross-dataset generalization
<table><tr><td>Train</td><td>Test</td><td>Accuracy (%)</td></tr><tr><td> $\mathrm { D } 2 + \mathrm { D } 3 + \mathrm { D } 4$ </td><td>D1</td><td>84.19</td></tr><tr><td> $\mathrm { D 1 } + \mathrm { D 3 } + \mathrm { D 4 }$ </td><td>D2</td><td>81.41</td></tr><tr><td> $\mathrm { D } 1 + \mathrm { D } 2 + \mathrm { D } 4$ </td><td>D3</td><td>68.63</td></tr><tr><td> $\mathsf { D 1 } + \mathsf { D 2 } + \mathsf { D 3 }$ </td><td>D4</td><td>67.88</td></tr><tr><td>Mean</td><td></td><td>75.52</td></tr></table>

(b) D3 attribution by task
<table><tr><td>Task</td><td>Acc. (%) Macro-F1 (%) # Prompts</td><td></td></tr><tr><td>Explanation</td><td>84.69</td><td>84.39 40</td></tr><tr><td>Argument / Advice</td><td>95.31</td><td>95.29 40</td></tr><tr><td>Creative</td><td>87.81</td><td>88.27 40</td></tr><tr><td>Rewrite</td><td>35.94</td><td>34.03 40</td></tr><tr><td>Source synthesis</td><td>39.38</td><td>36.00 40</td></tr><tr><td>Overall</td><td>68.63</td><td>68.41 200</td></tr></table>

Table 5: Cross-dataset attribution performance in the 8-way setting. (a) Accuracy when training on three prompt datasets and evaluating on the held-out fourth dataset. (b) Attribution performance on D3 by task type when training on D1, D2, and D4. Chance accuracy is 12.5%.

## 5.1 Cross-Dataset Generalization

We train the attribution classifier on three prompt datasets and evaluate it on the held-out fourth, repeating this for each dataset. As shown in Table 5(a), accuracy remains well above the 12.5% chance level in every split, with a mean of 75.52%. Performance is higher on D1 and D2, which share a data source, than on the more distinct D3 and D4 distributions. Prompt distribution therefore affects fingerprint recoverability without erasing model identity.

The same pattern holds under stronger text transformations. Across T0, T3, and $\mathrm { P } 2 ^ { * }$ , holding out the target dataset reduces Combined accuracy by only 4.8–5.9 points relative to mixed-source evaluation, whereas transforming the responses causes substantially larger losses. Lexical nearly matches Combined on raw text and is strongest among the standard representations under T3, while Combined performs better under $\mathrm { P } 2 ^ { * }$ after paraphrasing weakens lexical information. Distribution shift therefore has a measurable but smaller effect than directly removing or rewriting model-identifying signals.

Table 5(b) shows stronger variation across D3 task types. Attribution remains reliable for explanation, argument/advice, and creative generation, but drops sharply for rewriting and source synthesis. These tasks constrain more of the response content and structure, leaving less room for model-specific variation. This pattern is also consistent with the strong PN Embedding results in Appendix H.1, where removing prompt-shared variation makes model differences more visible. Detailed transformed and per-model results are provided in Appendix H.2.

## 5.2 Within-Language

To assess fingerprint recoverability within each language, we use an LLM to translate the English prompts into Persian and Mandarin Chinese and then perform the same eight-way cross-family attribution task on responses generated in each language. The translation procedure and prompt are provided in Appendix H.3.

The Persian and Mandarin Chinese results in Appendix I follow the same broad pattern as English. Attribution decreases as surface signals are removed, drops further under paraphrasing, and is lowest under P2<sup>∗</sup>. Even then, all representations remain above the 12.5% chance level. Lexical is strongest across most signal-suppressed views, while Combined becomes more competitive after paraphrasing. PN Embedding improves over the standard Semantic representation in every view, but usually remains below the strongest Lexical or Combined results in Persian and Chinese. Overall, fingerprint persistence extends beyond English, although its strength varies across languages.

## 5.3 Cross-Lingual

Because word- and character-level lexical features are strongly tied to language and script, we exclude the Lexical representation from the cross-lingual transfer experiments. Table 6 evaluates transfer by training the classifier in one language and testing it in another. Accuracy remains above chance in all six directions. Transfer is weaker than within-language attribution and varies across language pairs, showing that only part of the model-specific signal is shared across languages.

<table><tr><td rowspan=1 colspan=6>Train → TestBehavioral Stylometric Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>EN → FA</td><td rowspan=1 colspan=1>34.54</td><td rowspan=1 colspan=1>15.38</td><td rowspan=1 colspan=1>32.63</td><td rowspan=1 colspan=1>34.82</td><td rowspan=1 colspan=1>44.75</td></tr><tr><td rowspan=1 colspan=1>EN → ZH</td><td rowspan=1 colspan=1>35.01</td><td rowspan=1 colspan=1>16.76</td><td rowspan=1 colspan=1>40.23</td><td rowspan=1 colspan=1>39.00</td><td rowspan=1 colspan=1>59.10</td></tr><tr><td rowspan=1 colspan=1>FA → EN</td><td rowspan=1 colspan=1>21.77</td><td rowspan=1 colspan=1>21.04</td><td rowspan=1 colspan=1>16.81</td><td rowspan=1 colspan=1>19.67</td><td rowspan=1 colspan=1>47.99</td></tr><tr><td rowspan=1 colspan=1>FA → ZH</td><td rowspan=1 colspan=1>26.39</td><td rowspan=1 colspan=1>21.98</td><td rowspan=1 colspan=1>24.40</td><td rowspan=1 colspan=1>34.76</td><td rowspan=1 colspan=1>55.78</td></tr><tr><td rowspan=1 colspan=1>ZH → EN</td><td rowspan=1 colspan=1>27.30</td><td rowspan=1 colspan=1>20.68</td><td rowspan=1 colspan=1>30.03</td><td rowspan=1 colspan=1>36.39</td><td rowspan=1 colspan=1>59.46</td></tr><tr><td rowspan=1 colspan=1>ZH → FA</td><td rowspan=1 colspan=1>32.86</td><td rowspan=1 colspan=1>22.62</td><td rowspan=1 colspan=1>39.36</td><td rowspan=1 colspan=1>35.53</td><td rowspan=1 colspan=1>51.07</td></tr></table>

Table 6: Cross-lingual 8-way model attribution accuracy (%) on raw outputs (T0) using logistic regression. EN denotes English, FA Persian, and ZH Mandarin Chinese. Chance accuracy is 12.5%.

The feature results show where this transferable signal is concentrated. Stylometric features transfer poorly, consistent with their dependence on language-specific grammatical and stylistic patterns. Behavioral features remain useful in several directions, while Semantic and Combined representations are generally stronger. PN Embedding improves transfer in every direction, often substantially, indicating that removing prompt-shared variation reveals a more stable model-relative signal.

Appendix J reports average cross-lingual attribution performance across all six language-transfer settings under each response transformation. Accuracy declines from T0 through T3 as surface signals are removed, but remains above chance after severe Transformation. Paraphrasing alone has much less effect, with P1 and P2 retaining transfer close to the raw setting. The largest degradation appears under P1<sup>∗</sup> and P2<sup>∗</sup>, where paraphrasing is combined with T3. Overall, cross-lingual attribution appears to rely on cross-linguistically shared behavioral and structural patterns, together with more persistent modelspecific signals that survive substantial changes in wording and form.

## 6 Conclusion

We presented a multi-resolution study of LLM fingerprinting across model families, closely related models, distilled and LoRA-adapted variants, text transformations, prompt distributions, and languages. Model identity remains detectable at increasingly fine levels of similarity, although neighboring LoRA variants can be difficult to resolve. The identifying signal is distributed across multiple forms of evidence. Surface and behavioral signals are relatively fragile, lexical information remains strong after surface normalization, and combining signal types becomes more useful after paraphrasing. Fingerprints also transfer substantially across prompt distributions and languages, but recoverability depends strongly on the task. Attribution drops sharply for highly constrained tasks such as rewriting and source synthesis, where the prompt leaves less freedom for model-specific variation.

These findings support a view of LLM fingerprints as layered rather than tied to a single observable feature. Our fingerprint-resolution framework makes this dependence explicit by studying both how similar the candidate models are and how much identifying evidence remains available. The strong performance of PN Embedding further shows that removing variation shared across responses to the same prompt can substantially improve model separation, especially when task constraints are strong. Together, our results show that fingerprint recoverability depends jointly on model similarity, signal type, prompt and task constraints, and the transformations applied to the text, providing a broader diagnostic view of where model-specific behavior resides and when it remains observable.

## Limitations

Although our prompt sets are designed to cover diverse and largely general-purpose requests, they cannot represent the full range of prompts, interaction styles, and usage patterns encountered in everyday use. The multilingual experiments are limited to English, Chinese, and Persian, with non-English prompts translated from the English set rather than independently authored in each language. The controlled LoRA analysis uses two base models and two reasoning adaptations and may not reflect other adaptation methods or domains. Our evaluation also assumes a closed set of candidate models, whereas real-world provenance may involve unknown or unseen generators. PN Embedding is used as a diagnostic representation because it requires responses from all candidate models to the same prompt and is therefore not applicable to standard single-response attribution. Finally, our transformation suite cannot capture the full range of human editing, rewriting, translation, and post-processing that model outputs may undergo in practice.

## AI Use Statement

In this work, we use generative AI tools to assist with implementing and scaling experimental code. The experimental logic, configurations, and parameters are specified and reviewed by the authors. AIassisted code is inspected and tested, including through unit tests, before use.

Additionally, we use generative AI tools to identify potentially relevant literature and to improve the clarity and readability of the manuscript. All cited works are reviewed directly by the authors before inclusion. AI-assisted writing is used for editing and polishing author-written content rather than as a substitute for author review. All resulting text is manually checked, revised, and approved by the authors.

We also use generative AI assistance in the construction of a prompt dataset, following the procedure described in Appendix A.2. The resulting data and construction process are reviewed by the authors.

We take responsibility for the final content of this work, including text, code, data, claims, and other artifacts produced with the aid of generative AI.

## Ethics Statement

Our method is intended as a diagnostic tool for studying model-specific generation patterns, not as a definitive detector of AI-generated text or as a standalone basis for decisions about individuals. Attribution errors remain possible, particularly under distribution shifts and text transformations, and results should therefore be interpreted with appropriate caution in high-stakes settings. We release the generated model outputs to support reproducibility and further research. Because the dataset contains more than 15,000 generations, the outputs were not exhaustively screened for potentially harmful, offensive, or otherwise undesirable content. They reflect the behavior of the source models at the time of generation and should not be interpreted as endorsed content. Downstream users are responsible for applying appropriate safeguards and complying with the relevant model and data-use terms.

## Reproducibility Statement

The main text and appendices describe the experimental design, model and feature configurations, transformations, evaluation protocols, and additional implementation details needed to reproduce our results. We also make available the prompt datasets used in the study, all generated outputs comprising more than 15,000 generations from over 19 models, and the code used for model generation, LoRA finetuning, feature extraction, and classifier training and evaluation. These materials will be made publicly available on GitHub<sup>1</sup> to support independent verification and extension of the experiments.

## References

Wissam Antoun, Benoˆıt Sagot, and Djame Seddah. From text to source: Results in detecting large´ language model-generated content, 2024. URL https://arxiv.org/abs/2309.13322.

Yehonatan Bitton, Elad Bitton, and Shai Nisan. Detecting stylistic fingerprints of Large Language Models, 2025. URL https://arxiv.org/abs/2503.01659.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110. 14168.

Liam Dugan, Alyssa Hwang, Filip Trhl´ık, Andrew Zhu, Josh Magnus Ludan, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. RAID: A shared benchmark for robust evaluation of machinegenerated text detectors. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12463–12492, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.674. URL https://aclanthology.org/2024. acl-long.674/.

Irena Gao, Percy Liang, and Carlos Guestrin. Model equality testing: Which model is this api serving? In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 86369–86382, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ d73234a13815fc1f9779dd17d89be9b4-Paper-Conference.pdf.

Thibaud Gloaguen, Robin Staab, Nikola Jovanovic, and Martin Vechev. LLM fingerprinting via seman-´ tically conditioned watermarks. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=t38nZqqi3Z.

Matteo Greco, Anudeex Shetty, Andrea Tagarelli, and Jey Han Lau. Multighostbench: A multilingual benchmark for long-form llm-generated text attribution under distribution shifts, 2026. URL https: //arxiv.org/abs/2609.02379.

Xun Guo, Yongxin He, Shan Zhang, Ting Zhang, Wanquan Feng, Haibin Huang, and Chongyang Ma. Detective: Detecting AI-generated text via multi-level contrastive learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=cdTTTJfJe3.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= nZeVKeeFYf9.

Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, and Yejin Choi. Artificial hivemind: The open-ended homogeneity of language models (and beyond). In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https://openreview.net/forum?id= saDOrrnNTz.

Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Frederick Wieting, and Mohit Iyyer. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=WbFhFvjjKj.

Lucio La Cava and Andrea Tagarelli. OpenTuringBench: An open-model-based benchmark and framework for machine-generated text detection and attribution. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26655–26671, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025. emnlp-main.1354. URL https://aclanthology.org/2025.emnlp-main.1354/.

Lucio La Cava, Dominik Macko, Robert Moro, Ivan Srba, and Andrea Tagarelli. Authorship attribution in multilingual machine-generated texts. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 45136–45152, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026. acl-long.2091. URL https://aclanthology.org/2026.acl-long.2091/.

Yafu Li, Qintong Li, Leyang Cui, Wei Bi, Zhilin Wang, Longyue Wang, Linyi Yang, Shuming Shi, and Yue Zhang. MAGE: Machine-generated text detection in the wild. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 36–53, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.3. URL https://aclanthology.org/2024.acl-long.3/.

Xingyu Lu, Yumeng Ma, Xiang Zhou, Shengli Gan, Guiying Deng, Yang Wen, and Yanbing Liu. Synergizing stylometrics with semantics: Dual-path framework for LLM detection and attribution. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, Findings of the Association for Computational Linguistics: ACL 2026, pages 37252–37265, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176- 395-1. doi: 10.18653/v1/2026.findings-acl.1855. URL https://aclanthology.org/2026. findings-acl.1855/.

Hope McGovern, Rickard Stureborg, Yoshi Suhara, and Dimitris Alikaniotis. Your large language models are leaving fingerprints. In Firoj Alam, Preslav Nakov, Nizar Habash, Iryna Gurevych, Shammur Chowdhury, Artem Shelmanov, Yuxia Wang, Ekaterina Artemova, Mucahid Kutlu, and George Mikros, editors, Proceedings of the 1stWorkshop on GenAI Content Detection (GenAIDetect), pages 85–95, Abu Dhabi, UAE, January 2025. International Conference on Computational Linguistics. URL https://aclanthology.org/2025.genaidetect-1.6/.

Dario Pasquini, Evgenios M. Kornaropoulos, and Giuseppe Ateniese. LLMmap: Fingerprinting for large language models. In 34th USENIX Security Symposium (USENIX Security 25), pages 299– 318, Seattle, WA, August 2025. USENIX Association. ISBN 978-1-939133-52-6. URL https: //www.usenix.org/conference/usenixsecurity25/presentation/pasquini.

Nazneen Rajani, Lewis Tunstall, Edward Beeching, Nathan Lambert, Alexander M. Rush, and Thomas Wolf. No robots. https://huggingface.co/datasets/HuggingFaceH4/no\_robots, 2023.

Zixin Rao, Youssef Mohamed, Shang Liu, and Zeyan Liu. Two birds with one stone: Multi-task detection and attribution of llm-generated text, 2025. URL https://arxiv.org/abs/2508. 14190.

Mingjie Sun, Yida Yin, Zhiqiu Xu, J. Zico Kolter, and Zhuang Liu. Idiosyncrasies in Large Language Models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 57854–57885. PMLR, 2025. URL https: //proceedings.mlr.press/v267/sun25z.html.

Teppei Suzuki, Ryokan Ri, and Sho Takase. Natural fingerprints of Large Language Models, 2025. URL https://arxiv.org/abs/2504.14871.

Yao Tong, Haonan Wang, Siquan Li, Kenji Kawaguchi, and Tianyang Hu. Seedprints: Fingerprints can even tell which seed your large language model was trained from. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=Kan6Z0zzZi.

Adaku Uchendu, Thai Le, Kai Shu, and Dongwon Lee. Authorship attribution for neural text generation. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8384–8395, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main. 673. URL https://aclanthology.org/2020.emnlp-main.673/.

Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Chenxi Whitehouse, Osama Mohammed Afzal, Tarek Mahmoud, Toru Sasaki, Thomas Arnold, Alham Fikri Aji, Nizar Habash, Iryna Gurevych, and Preslav Nakov. M4: Multi-generator, multi-domain, and multi-lingual black-box machine-generated text detection. In Yvette Graham and Matthew Purver, editors, Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1369–1407, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.eacl-long.83. URL https: //aclanthology.org/2024.eacl-long.83/.

Jiashu Xu, Fei Wang, Mingyu Ma, Pang Wei Koh, Chaowei Xiao, and Muhao Chen. Instructional fingerprinting of large language models. In Kevin Duh, Helena Gomez, and Steven Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3277–3306, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. naacl-long.180. URL https://aclanthology.org/2024.naacl-long.180/.

## A Detailed Experimental Configuration

## A.1 Models and Generation

The full 19-model panel is grouped by family below.

• Llama

$$
{ \begin{array} { r l } & { - \mathrm { \ m e t a } { - 1 1 } \mathtt { a m a } / 1 \mathtt { l a m a } { - 3 } . 1 { - } 8 { \mathtt { b } } { - } \mathtt { i n s t r u c t } } \\ & { - \mathrm { \ m e t a } { - } 1 1 \mathtt { a m a } / 1 \mathtt { l a m a } { - } 3 . 1 { - } 7 0 \mathtt { b } { - } \mathtt { i n s t r u c t } } \\ & { - \mathrm { \ m e t a } { - } 1 1 \mathtt { a m a } / 1 \mathtt { l a m a } { - } 3 . 3 { - } 7 0 \mathtt { b } { - } \mathtt { i n s t r u c t } } \\ & { - \mathrm { \ m e t a } { - } 1 1 \mathtt { a m a } / 1 \mathtt { l a m a } { - } 4 { - } \mathrm { \ m a v e r i } \mathtt { c k } } \end{array} }
$$

## • Qwen

$$
\begin{array} { c } { { - \mathrm { \ c u e n / q w e n 3 - 8 b } } } \\ { { - \mathrm { \ c u e n / q w e n 3 - 1 4 b } } } \end{array}
$$

Table 7: Task categories in the D3 control set.
<table><tr><td>Task</td><td>Description</td></tr><tr><td>Explanation</td><td>Explain a concept clearly and accurately to a general reader.</td></tr><tr><td>Argument and advice</td><td>Evaluate a situation, weigh alternatives, and provide a justified recommendation</td></tr><tr><td>Creative writing</td><td>Produce an original story or scene under specified narrative conditions</td></tr><tr><td>Rewriting</td><td>Transform a supplied passage while preserving its central meaning.</td></tr><tr><td>Source synthesis</td><td>Combine four supplied source notes into a coherent response.</td></tr></table>

– qwen/qwen3-32b

## • Mistral

– mistralai/mistral-small-24b-instruct-2501

– mistralai/mistral-small-3.2-24b-instruct

## • Gemma

– google/gemma-4-26b-a4b-it

– google/gemma-4-31b-it

## • DeepSeek

– deepseek/deepseek-v4-flash

– deepseek/deepseek-v4-flash-0731

## • Anthropic

– anthropic/claude-haiku-4.5

– anthropic/claude-sonnet-4

## • Gemini

– google/gemini-2.5-flash

– google/gemini-2.5-flash-lite

## • OpenAI

– openai/gpt-4.1

– openai/gpt-4.1-mini

Cross-family representatives are selected using a fixed rule. We use the newest included release in each family, or the largest general-purpose instruction model when multiple included models belong to the same release.

## A.2 Design and Construction of the D3 Control Set

D3 is designed as a controlled complement to the naturally occurring prompt collections used elsewhere in the study. The authors first specify five task categories that differ in their objectives, response constraints, and expected difficulty: explanation, argument and advice, creative writing, rewriting, and source synthesis. Table 7 describes each task.

After defining this structure, we use GPT-5.6 Sol to generate 40 prompts for each category, producing 200 prompts in total. The model is instructed to maintain the predefined category boundaries while varying topics, scenarios, and requested outputs. The authors determine the task taxonomy, category balance, and construction requirements, while GPT-5.6 Sol is used only to draft the individual prompt texts.

All generated prompts subsequently undergo human review. Each prompt is evaluated for clarity, category fit, duplication, and whether it provides a meaningful task for every candidate model. The final set retains an equal number of prompts from each category. Prompt selection is completed before collecting the model responses and does not use attribution results or downstream classifier performance.

This construction provides several experimental benefits. First, exact category balance prevents the distribution from being dominated by a common task type. Second, the categories span different levels of response freedom, from open-ended creative generation to tightly constrained rewriting and sourcegrounded synthesis. This variation makes it possible to examine whether model fingerprints become weaker when the task restricts the linguistic choices available to the generator. Finally, constructing a novel prompt set reduces the risk that evaluated models reproduce memorized responses to established benchmark questions. Therefore, D3 is an AI-assisted, researcher-designed, and human-reviewed control set.

## A.3 Generation Setting

Raw responses are generated with temperature 1.0, top-p = 0.9, and a maximum of 2,048 output tokens. Raw generations are stored before any transformation. Response embeddings use text-embedding-3-small with 1,536 dimensions. We also evaluated several open-source embedding models. Among the models tested, text-embedding-3-small achieved the strongest attribution performance and is therefore used throughout the reported experiments.

Throughout the paper, we use model fingerprint to denote source-specific traces in the output of a generator under a fixed generation setup. In black-box API settings, such traces may reflect both the underlying model and components of the serving stack. Our controlled LoRA experiments complement this setting by holding the base model and generation protocol fixed while varying the contribution of a learned weight update, showing that weight-level changes themselves can also produce detectable traces.

## A.4 Complete Signal Feature Definitions

This section provides the complete definitions of the signal representations introduced in Sec. 3.2. Features are recomputed for every response view. All learned preprocessing, including feature standardization and TF–IDF vocabulary construction, is fitted only on the training portion of each fold.

Behavioral and structural features. We use 25 features describing response length, organization, formatting, punctuation, and repetition:

• Length and structure: word count, character count, sentence count, paragraph count, mean sentence-span length, and standard deviation of sentence-span length.

• Formatting and layout: Markdown heading count, bullet/list count, code-fence count, newline count, newline-to-character ratio, and counts of #, <sub>\*</sub>, hyphens, underscores, and backticks.

• Punctuation: counts of commas, periods, exclamation marks, question marks, colons, semicolons, and quotation marks.

• Repetition: repeated-bigram and repeated-trigram ratios.

For an n-gram sequence, the repetition ratio is the proportion of repeated n-grams relative to all observed n-grams. It is set to zero when the response is too short to form the corresponding n-gram.

Stylometric features. We use 17 dense features:

• Vocabulary diversity: token count, unique-token count, type–token ratio, MATTR-10, and MATTR-50.

• Word form: mean and standard deviation of word length.

• Vocabulary distribution: hapax ratio and most-common-word frequency ratio.

• Function and grammatical words: function-word ratio, pronoun ratio, modal ratio, and connective ratio.

• Repetition and numeric use: unigram, bigram, and trigram repetition ratios, and numeric-token ratio.

MATTR is computed as the mean type–token ratio over sliding windows of 10 or 50 tokens. For responses shorter than the specified window, the type–token ratio of the available response is used. The hapax ratio measures the proportion of vocabulary items that occur exactly once.

Part-of-speech features. The dense stylometric features are concatenated with POS unigram, bigram, and trigram TF–IDF features. POS sequences are extracted using spaCy en core web sm. The POS vectorizer uses min df=2 and retains at most 2,500 features. The resulting representation is the Stylometric representation used throughout the paper.

Lexical features. We concatenate two TF–IDF representations:

• lowercase word unigrams and bigrams, with at most 5,000 features;

• lowercase character 3–5-grams, with at most 5,000 features.

Both use min df=2 and sublinear term-frequency scaling. The combined representation therefore contains at most 10,000 dimensions.

Semantic features. Each response is embedded using text-embedding-3-small, producing a 1,536-dimensional vector. Every transformed or paraphrased response is embedded after the transformation is applied; embeddings from the raw response are not reused. Embedding dimensions are standardized using statistics computed from the training fold.

Although these representations are primarily designed to capture semantic information, they can also retain lexical, stylistic, and structural signals. This is reflected in our results, where attribution performance using the Semantic representation declines as progressively stronger text transformations are applied. We therefore use the term Semantic to denote the representation’s primary purpose, rather than to imply that it encodes semantic information exclusively.

Prompt-neutralized embeddings. For each prompt and response view, we compute the mean embedding across all candidate-model responses and subtract it from each individual response embedding:

$$
\bar { e } _ { p , t } = \frac { 1 } { K } \sum _ { m = 1 } ^ { K } e _ { p , m , t } , \qquad e _ { p , m , t } ^ { \prime } = e _ { p , m , t } - \bar { e } _ { p , t } .
$$

This removes components shared across responses to the same prompt and emphasizes relative differences between candidate models. Because this representation requires access to the other candidate responses for the same test prompt, we use it only as a diagnostic representation rather than as a standard single-response attribution setting.

Combined representation. The combined representation concatenates the four main signal groups: behavioral and structural features, stylometric and POS features, word and character TF–IDF, and frozen embeddings. Same-prompt centered embeddings are excluded. Dense behavioral, lexical, and embedding features are standardized using training-fold statistics, while POS and word/character TF–IDF features retain their sparse representations.

## A.5 Transformation and Paraphrasing Details

$\begin{array} { l l } { { T _ { 0 } { - } T _ { 3 } . } } & { { T _ { 0 } } } \end{array}$ is the unmodified response. For $T _ { 1 }$ , responses with at most 10 words are unchanged; responses with 11–20 words lose one word from each side; responses with 21–50 words lose five words from each side; and responses longer than 50 words lose 15 words from each side. $T _ { 2 }$ is applied to $T _ { 1 }$ and removes Markdown heading, list, blockquote, and code-fence markers; emphasis and inline-code delimiters; Markdown link/image syntax while retaining visible text; HTML tags; indentation; and repeated layout whitespace. Ordinary punctuation, capitalization, lexical content, and word order are preserved. $T _ { 3 }$ is applied to $T _ { 2 }$ and performs Unicode NFKC normalization, lowercasing, punctuation/symbol/emoji removal, and whitespace collapse while preserving lexical words, function words, numbers, and word order.

$P _ { 1 }$ and $P _ { 2 }$ . Both paraphrase conditions branch from $T _ { 0 }$ and use gpt-4o-mini-2024-07-18 with temperature 0, top-p = 1, a maximum of 2,048 output tokens, and reasoning disabled. The paraphraser never receives the source-model identity. For $P _ { 1 }$ , approximately one third of eligible natural-language sentences are selected deterministically using prompt identity rather than source label; headings, structural fragments, and executable code are not paraphrase targets. All selected targets from one response are rewritten together, while untouched text is restored unchanged. If a response contains no eligible prose, $P _ { 1 }$ leaves it unchanged. $P _ { 2 }$ rewrites the complete response while preserving its meaning, factual claims, entities, numbers, qualifications, language, and approximate amount of information. We require successful, non-empty outputs but do not filter paraphrases according to downstream attribution performance.

## A.5.1 P<sub>1</sub>: Partial Paraphrasing Prompt

For $P _ { 1 } ,$ , the target indices, full response, and selected sentences are inserted automatically for each example. JSON and output-schema instructions are omitted below for readability.

You are performing controlled partial editing for a scientific   
experiment.   
The full passage is provided for context. Rewrite ONLY the target   
sentences listed below.   
For EACH supplied target, substantially change its wording and   
syntax while preserving its meaning, facts, entities, numbers,   
qualifications, references, tone, and approximately the same sentence   
length.   
Do not rewrite or return any non-target sentence.   
Requested indices: [target indices]   
FULL PASSAGE:   
[full response]   
TARGETS:   
[selected target sentences]

## A.5.2 P<sub>2</sub>: Full-Response Paraphrasing Prompt

For P<sub>2</sub>, the complete response and response-specific word-count bounds are inserted automatically.

You are a controlled paraphrasing system used for a scientific   
text-transformation experiment.   
Rewrite the supplied text thoroughly while preserving its meaning   
exactly.   
Requirements:   
- Rewrite wording and sentence structure throughout the passage. Do   
not merely replace a few words with synonyms.   
- Preserve every factual claim, named entity, number, example,   
qualification, uncertainty, instruction, and conclusion.   
- Do not add new information, explanations, opinions, or facts.   
- Do not remove information or summarize.   
- Preserve the original language, tone, level of formality, and broad   
paragraph/list organization.   
- The original has approximately [word count] whitespace-delimited   
words. Target [lower bound]--[upper bound] words; do not compress or   
expand the content.   
- Do not add an introduction such as ‘Here is the paraphrased   
version.’   
- Return only the rewritten text.   
TEXT TO PARAPHRASE:   
[full response]

The $P _ { 2 }$ length bounds are computed separately for each response as approximately 85%–115% of its original whitespace-delimited word count. They are used to guide the paraphraser toward preserving the amount of information rather than as a post-generation acceptance criterion.

## A.6 Classifier and Evaluation Details

Logistic Regression uses an $L _ { 2 }$ penalty with the SAGA solver. Its regularization parameter is selected from $C \in \{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ using macro-F1 on an inner validation split. The MLP has one hidden layer of 256 units with ReLU activation and is trained with Adam using learning rate 10<sup>−3</sup>, weight decay 10<sup>−4</sup>, and batch size 128. Training is capped at 80 epochs, with early stopping after eight epochs without improvement. The selected number of epochs is then used when refitting on the full outer-training fold.

The main evaluation uses five-fold prompt-grouped, dataset-stratified cross-validation. Each outer fold therefore uses approximately 80% of the prompts for training and 20% for testing. Within each outer-training fold, 20% of prompts from each dataset are used for hyperparameter selection. All responses from the same prompt, across models and transformed views, remain grouped together, and the same outer and inner partitions are used across experimental conditions. Predictions from the five outer folds are concatenated before computing accuracy and macro-F1. For the balanced eight-way crossfamily task, chance accuracy is 12.5%, while within-family binary tasks have a 50% chance level and are interpreted separately.

## B Cross-Family MLP Results

Table 8 reports performance on the eight-way cross-family attribution task using a one-hidden-layer MLP classifier. The MLP generally performs slightly better than Logistic Regression across most repre-

<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>59.92</td><td rowspan=1 colspan=1>66.82</td><td rowspan=1 colspan=1>79.74</td><td rowspan=1 colspan=1>68.10</td><td rowspan=1 colspan=1>82.01</td><td rowspan=1 colspan=1>84.46</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>55.02</td><td rowspan=1 colspan=1>63.39</td><td rowspan=1 colspan=1>75.91</td><td rowspan=1 colspan=1>61.53</td><td rowspan=1 colspan=1>77.53</td><td rowspan=1 colspan=1>77.08</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>47.47</td><td rowspan=1 colspan=1>60.36</td><td rowspan=1 colspan=1>72.36</td><td rowspan=1 colspan=1>56.23</td><td rowspan=1 colspan=1>73.12</td><td rowspan=1 colspan=1>74.56</td></tr><tr><td rowspan=2 colspan=1>T3P1</td><td rowspan=1 colspan=1>29.41</td><td rowspan=1 colspan=1>53.13</td><td rowspan=1 colspan=1>67.98</td><td rowspan=1 colspan=1>51.71</td><td rowspan=1 colspan=1>68.21</td><td rowspan=1 colspan=1>70.06</td></tr><tr><td rowspan=1 colspan=1>59.79</td><td rowspan=1 colspan=1>65.08</td><td rowspan=1 colspan=1>74.12</td><td rowspan=1 colspan=1>65.05</td><td rowspan=1 colspan=1>78.63</td><td rowspan=1 colspan=1>81.62</td></tr><tr><td rowspan=2 colspan=1>P2P1*</td><td rowspan=1 colspan=1>56.65</td><td rowspan=1 colspan=1>60.09</td><td rowspan=1 colspan=1>68.29</td><td rowspan=1 colspan=1>58.62</td><td rowspan=1 colspan=1>72.68</td><td rowspan=1 colspan=1>75.99</td></tr><tr><td rowspan=1 colspan=1>28.44</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>61.19</td><td rowspan=1 colspan=1>48.64</td><td rowspan=1 colspan=1>62.69</td><td rowspan=1 colspan=1>66.38</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>26.69</td><td rowspan=1 colspan=1>45.67</td><td rowspan=1 colspan=1>53.57</td><td rowspan=1 colspan=1>43.30</td><td rowspan=1 colspan=1>56.54</td><td rowspan=1 colspan=1>61.05</td></tr></table>

Table 8: Cross-family attribution accuracy (%) using MLP. Chance accuracy is 12.5%.
<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>55.76</td><td rowspan=1 colspan=1>58.32</td><td rowspan=1 colspan=1>60.26</td><td rowspan=1 colspan=1>55.51</td><td rowspan=1 colspan=1>62.14</td><td rowspan=1 colspan=1>62.70</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>55.88</td><td rowspan=1 colspan=1>57.13</td><td rowspan=1 colspan=1>56.63</td><td rowspan=1 colspan=1>54.88</td><td rowspan=1 colspan=1>58.07</td><td rowspan=1 colspan=1>59.95</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>53.69</td><td rowspan=1 colspan=1>56.20</td><td rowspan=1 colspan=1>56.51</td><td rowspan=1 colspan=1>53.07</td><td rowspan=1 colspan=1>56.01</td><td rowspan=1 colspan=1>56.57</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>52.63</td><td rowspan=1 colspan=1>57.95</td><td rowspan=1 colspan=1>55.82</td><td rowspan=1 colspan=1>51.63</td><td rowspan=1 colspan=1>57.07</td><td rowspan=1 colspan=1>56.70</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>56.20</td><td rowspan=1 colspan=1>57.01</td><td rowspan=1 colspan=1>55.76</td><td rowspan=1 colspan=1>54.26</td><td rowspan=1 colspan=1>60.20</td><td rowspan=1 colspan=1>60.95</td></tr><tr><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>54.63</td><td rowspan=1 colspan=1>54.69</td><td rowspan=1 colspan=1>53.07</td><td rowspan=1 colspan=1>54.07</td><td rowspan=1 colspan=1>55.82</td><td rowspan=1 colspan=1>58.57</td></tr><tr><td rowspan=1 colspan=1>P1*</td><td rowspan=1 colspan=1>52.32</td><td rowspan=1 colspan=1>54.69</td><td rowspan=1 colspan=1>54.44</td><td rowspan=1 colspan=1>52.32</td><td rowspan=1 colspan=1>55.82</td><td rowspan=1 colspan=1>54.82</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>53.44</td><td rowspan=1 colspan=1>52.69</td><td rowspan=1 colspan=1>51.31</td><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>54.69</td><td rowspan=1 colspan=1>54.19</td></tr></table>

Table 9: Within-family attribution accuracy (%) for Llama-3.1-8B vs. Llama-3.1-70B using logistic regression. Chance accuracy is 50%.

sentations, while Lexical performance remains nearly unchanged. Importantly, the relative ordering of representations and the degradation across transformations remain similar across classifiers.

## C Detailed Within-Family Llama Results

To examine how fingerprint recoverability changes under increasingly strong transformations for closely related models, we report the complete within-family Llama pairwise results in Tables 9–14. Each comparison is a binary attribution task using logistic regression, with a chance accuracy of 50%. The tables report the same behavioral, stylometric, lexical, semantic, combined, and prompt-neutralized embedding representations used in the main experiments across all response views.

## D Within-Family Results for Additional Model Families

Table 15 shows substantial variation in fingerprint strength across model families. Some closely related models remain highly separable, while others achieve only modest gains over chance. Despite this variation, the overall signal pattern is consistent with the Llama results. The Lexical and Combined representations generally perform best, indicating that lexical evidence is central but benefits from additional stylistic and structural signals. Behavioral and embedding representations are less consistent on their own, suggesting that within-family identity is distributed across multiple forms of evidence. Table 16 reports the corresponding results for a one-hidden-layer MLP classifier. Neither classifier consistently outperforms the other by a large margin across settings.

<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>58.82</td><td rowspan=1 colspan=1>61.20</td><td rowspan=1 colspan=1>60.01</td><td rowspan=1 colspan=1>57.07</td><td rowspan=1 colspan=1>65.52</td><td rowspan=1 colspan=1>69.21</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>58.20</td><td rowspan=1 colspan=1>58.45</td><td rowspan=1 colspan=1>57.57</td><td rowspan=1 colspan=1>55.51</td><td rowspan=1 colspan=1>61.33</td><td rowspan=1 colspan=1>60.70</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>56.26</td><td rowspan=1 colspan=1>57.38</td><td rowspan=1 colspan=1>56.63</td><td rowspan=1 colspan=1>55.07</td><td rowspan=1 colspan=1>56.57</td><td rowspan=1 colspan=1>60.08</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>50.06</td><td rowspan=1 colspan=1>56.45</td><td rowspan=1 colspan=1>56.51</td><td rowspan=1 colspan=1>54.63</td><td rowspan=1 colspan=1>58.14</td><td rowspan=1 colspan=1>57.70</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>58.76</td><td rowspan=1 colspan=1>60.14</td><td rowspan=1 colspan=1>58.45</td><td rowspan=1 colspan=1>58.32</td><td rowspan=1 colspan=1>64.27</td><td rowspan=1 colspan=1>65.96</td></tr><tr><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>57.07</td><td rowspan=1 colspan=1>58.76</td><td rowspan=1 colspan=1>55.88</td><td rowspan=1 colspan=1>57.20</td><td rowspan=1 colspan=1>60.33</td><td rowspan=1 colspan=1>63.33</td></tr><tr><td rowspan=1 colspan=1>P1*</td><td rowspan=1 colspan=1>51.13</td><td rowspan=1 colspan=1>55.51</td><td rowspan=1 colspan=1>57.07</td><td rowspan=1 colspan=1>52.82</td><td rowspan=1 colspan=1>57.70</td><td rowspan=1 colspan=1>55.19</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>49.62</td><td rowspan=1 colspan=1>54.57</td><td rowspan=1 colspan=1>53.63</td><td rowspan=1 colspan=1>52.63</td><td rowspan=1 colspan=1>55.51</td><td rowspan=1 colspan=1>54.07</td></tr></table>

Table 10: Within-family attribution accuracy (%) for Llama-3.1-8B vs. Llama-3.3-70B using logistic regression. Chance accuracy is 50%.

<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>57.76</td><td rowspan=1 colspan=1>59.89</td><td rowspan=1 colspan=1>69.02</td><td rowspan=1 colspan=1>57.88</td><td rowspan=1 colspan=1>68.21</td><td rowspan=1 colspan=1>70.59</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>57.20</td><td rowspan=1 colspan=1>61.64</td><td rowspan=1 colspan=1>66.96</td><td rowspan=1 colspan=1>56.57</td><td rowspan=1 colspan=1>65.77</td><td rowspan=1 colspan=1>67.21</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>53.69</td><td rowspan=1 colspan=1>59.57</td><td rowspan=1 colspan=1>63.58</td><td rowspan=1 colspan=1>54.32</td><td rowspan=1 colspan=1>64.02</td><td rowspan=1 colspan=1>62.20</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>53.19</td><td rowspan=1 colspan=1>59.26</td><td rowspan=1 colspan=1>62.58</td><td rowspan=1 colspan=1>54.57</td><td rowspan=1 colspan=1>62.27</td><td rowspan=1 colspan=1>65.08</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>56.38</td><td rowspan=1 colspan=1>60.39</td><td rowspan=1 colspan=1>64.14</td><td rowspan=1 colspan=1>57.01</td><td rowspan=1 colspan=1>65.33</td><td rowspan=1 colspan=1>68.34</td></tr><tr><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>56.26</td><td rowspan=1 colspan=1>56.95</td><td rowspan=1 colspan=1>63.14</td><td rowspan=1 colspan=1>56.38</td><td rowspan=1 colspan=1>63.39</td><td rowspan=1 colspan=1>65.21</td></tr><tr><td rowspan=1 colspan=1>P1*</td><td rowspan=1 colspan=1>52.75</td><td rowspan=1 colspan=1>57.76</td><td rowspan=1 colspan=1>61.01</td><td rowspan=1 colspan=1>54.13</td><td rowspan=1 colspan=1>60.45</td><td rowspan=1 colspan=1>60.45</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>52.50</td><td rowspan=1 colspan=1>55.94</td><td rowspan=1 colspan=1>57.57</td><td rowspan=1 colspan=1>54.63</td><td rowspan=1 colspan=1>59.32</td><td rowspan=1 colspan=1>62.70</td></tr></table>

Table 11: Within-family attribution accuracy (%) for Llama-3.1-8B vs. Llama-4-Maverick using logistic regression. Chance accuracy is 50%.

<table><tr><td colspan="7" rowspan="1">View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td colspan="1" rowspan="1">TO</td><td colspan="1" rowspan="1">61.83</td><td colspan="1" rowspan="1">58.14</td><td colspan="1" rowspan="1">61.20</td><td colspan="1" rowspan="1">60.58</td><td colspan="1" rowspan="1">63.77</td><td colspan="1" rowspan="1">72.09</td></tr><tr><td colspan="1" rowspan="1">T1</td><td colspan="1" rowspan="1">61.33</td><td colspan="1" rowspan="1">58.39</td><td colspan="1" rowspan="1">56.82</td><td colspan="1" rowspan="1">55.19</td><td colspan="1" rowspan="1">62.52</td><td colspan="1" rowspan="1">62.58</td></tr><tr><td colspan="1" rowspan="1">T2</td><td colspan="1" rowspan="1">57.95</td><td colspan="1" rowspan="1">59.07</td><td colspan="1" rowspan="1">55.69</td><td colspan="1" rowspan="1">54.26</td><td colspan="1" rowspan="1">60.20</td><td colspan="1" rowspan="1">61.83</td></tr><tr><td colspan="1" rowspan="1">T3</td><td colspan="1" rowspan="1">55.19</td><td colspan="1" rowspan="1">57.13</td><td colspan="1" rowspan="1">56.26</td><td colspan="1" rowspan="1">53.38</td><td colspan="1" rowspan="1">58.89</td><td colspan="1" rowspan="1">59.57</td></tr><tr><td colspan="1" rowspan="1">P1</td><td colspan="1" rowspan="1">60.89</td><td colspan="1" rowspan="1">57.76</td><td colspan="1" rowspan="1">57.95</td><td colspan="1" rowspan="1">60.08</td><td colspan="1" rowspan="1">62.89</td><td colspan="1" rowspan="1">69.34</td></tr><tr><td colspan="1" rowspan="1">P2</td><td colspan="1" rowspan="1">60.58</td><td colspan="1" rowspan="1">55.57</td><td colspan="1" rowspan="1">56.51</td><td colspan="1" rowspan="1">55.88</td><td colspan="1" rowspan="1">60.01</td><td colspan="1" rowspan="1">63.45</td></tr><tr><td colspan="1" rowspan="1">P1*</td><td colspan="1" rowspan="1">55.19</td><td colspan="1" rowspan="1">55.76</td><td colspan="1" rowspan="1">54.51</td><td colspan="1" rowspan="1">53.19</td><td colspan="1" rowspan="1">58.32</td><td colspan="1" rowspan="1">58.95</td></tr><tr><td colspan="1" rowspan="1">P2*</td><td colspan="1" rowspan="1">55.07</td><td colspan="1" rowspan="1">54.19</td><td colspan="1" rowspan="1">54.88</td><td colspan="1" rowspan="1">51.69</td><td colspan="1" rowspan="1">55.44</td><td colspan="1" rowspan="1">55.32</td></tr><tr><td colspan="1" rowspan="1">TO</td><td colspan="1" rowspan="1">59.89</td><td colspan="1" rowspan="1">63.27</td><td colspan="1" rowspan="1">69.65</td><td colspan="1" rowspan="1">60.33</td><td colspan="1" rowspan="1">70.71</td><td colspan="1" rowspan="1">72.22</td></tr><tr><td colspan="1" rowspan="1">T1</td><td colspan="1" rowspan="1">59.26</td><td colspan="1" rowspan="1">62.08</td><td colspan="1" rowspan="1">67.83</td><td colspan="1" rowspan="1">56.51</td><td colspan="1" rowspan="1">69.09</td><td colspan="1" rowspan="1">67.33</td></tr><tr><td colspan="1" rowspan="1">T2</td><td colspan="1" rowspan="1">57.07</td><td colspan="1" rowspan="1">60.95</td><td colspan="1" rowspan="1">66.52</td><td colspan="1" rowspan="1">57.32</td><td colspan="1" rowspan="1">66.71</td><td colspan="1" rowspan="1">64.33</td></tr><tr><td colspan="1" rowspan="1">T3</td><td colspan="1" rowspan="1">55.82</td><td colspan="1" rowspan="1">61.08</td><td colspan="1" rowspan="1">65.83</td><td colspan="1" rowspan="1">58.07</td><td colspan="1" rowspan="1">66.77</td><td colspan="1" rowspan="1">64.58</td></tr><tr><td colspan="1" rowspan="1">P1</td><td colspan="1" rowspan="1">59.64</td><td colspan="1" rowspan="1">60.08</td><td colspan="1" rowspan="1">63.39</td><td colspan="1" rowspan="1">57.32</td><td colspan="1" rowspan="1">67.52</td><td colspan="1" rowspan="1">66.08</td></tr><tr><td colspan="1" rowspan="1">P2</td><td colspan="1" rowspan="1">58.57</td><td colspan="1" rowspan="1">59.45</td><td colspan="1" rowspan="1">62.89</td><td colspan="1" rowspan="1">56.01</td><td colspan="1" rowspan="1">62.64</td><td colspan="1" rowspan="1">62.83</td></tr><tr><td colspan="1" rowspan="1">P1*</td><td colspan="1" rowspan="1">55.94</td><td colspan="1" rowspan="1">58.26</td><td colspan="1" rowspan="1">61.70</td><td colspan="1" rowspan="1">55.26</td><td colspan="1" rowspan="1">64.27</td><td colspan="1" rowspan="1">61.58</td></tr><tr><td colspan="1" rowspan="1">P2*</td><td colspan="1" rowspan="1">56.01</td><td colspan="1" rowspan="1">57.32</td><td colspan="1" rowspan="1">59.89</td><td colspan="1" rowspan="1">53.19</td><td colspan="1" rowspan="1">59.89</td><td colspan="1" rowspan="1">60.95</td></tr></table>

Table 12: Within-family attribution accuracy (%) for Llama-3.1-70B vs. Llama-3.3-70B using logistic regression. Chance accuracy is 50%.

Table 13: Within-family attribution accuracy (%) for Llama-3.1-70B vs. Llama-4-Maverick using logistic regression. Chance accuracy is 50%.

<table><tr><td>View</td><td>Behavioral</td><td>Stylometric</td><td>Lexical</td><td>Semantic</td><td>Combined</td><td></td></tr><tr><td>TO</td><td>57.63</td><td>61.08</td><td>66.52</td><td>58.89</td><td>66.02</td><td>73.47</td></tr><tr><td>T1</td><td>59.89</td><td>60.95</td><td>64.46</td><td>56.32</td><td>64.96</td><td>68.84</td></tr><tr><td>T2</td><td>57.76</td><td>61.51</td><td>64.96</td><td>56.07</td><td>63.95</td><td>65.46</td></tr><tr><td>T3</td><td>52.82</td><td>59.39</td><td>64.77</td><td>57.20</td><td>63.83</td><td>68.46</td></tr><tr><td>P1</td><td>59.01</td><td>60.08</td><td>64.77</td><td>56.95</td><td>65.96</td><td>70.84</td></tr><tr><td>P2</td><td>57.07</td><td>55.38</td><td>59.82</td><td>55.19</td><td>61.33</td><td>67.58</td></tr><tr><td>P1*</td><td>52.88</td><td>56.13</td><td>62.45</td><td>56.70</td><td>61.33</td><td>66.83</td></tr><tr><td>P2*</td><td>51.31</td><td>53.57</td><td>57.20</td><td>53.63</td><td>56.45</td><td>61.83</td></tr></table>

Table 14: Within-family attribution accuracy (%) for Llama-3.3-70B vs. Llama-4-Maverick using logistic regression. Chance accuracy is 50%.

<table><tr><td>Family</td><td>Model Pair</td><td>Behavioral</td><td>Stylometric</td><td>Lexical</td><td>Semantic</td><td>Combined</td><td>PN Embedding†</td></tr><tr><td>DeepSeek</td><td>V4 vs. V4-0731</td><td>58.14</td><td>62.77</td><td>68.34</td><td>60.45</td><td>66.65</td><td>72.72</td></tr><tr><td>Gemma</td><td>4-26B vs. 4-31B</td><td>57.70</td><td>61.64</td><td>79.10</td><td>60.95</td><td>78.35</td><td>78.97</td></tr><tr><td>Mistral</td><td>Small-24B vs. Small-3.2-24B</td><td>74.28</td><td>86.05</td><td>94.24</td><td>90.86</td><td>94.37</td><td>98.75</td></tr><tr><td>Qwen</td><td>8B vs. 14B</td><td>56.45</td><td>58.51</td><td>57.76</td><td>54.44</td><td>62.33</td><td>62.33</td></tr><tr><td>Qwen</td><td>8B vs. 32B</td><td>63.77</td><td>71.15</td><td>69.46</td><td>69.09</td><td>73.78</td><td>85.98</td></tr><tr><td>Qwen</td><td>14B vs. 32B</td><td>64.83</td><td>70.84</td><td>71.09</td><td>63.58</td><td>74.97</td><td>84.48</td></tr><tr><td>Anthropic</td><td>Haiku-4.5 vs. Sonnet-4</td><td>69.65</td><td>69.90</td><td>77.28</td><td>79.47</td><td>82.85</td><td>92.87</td></tr><tr><td>Gemini</td><td>2.5-Flash vs. 2.5-Flash-Lite</td><td>59.01</td><td>64.27</td><td>71.15</td><td>59.89</td><td>70.53</td><td>74.72</td></tr><tr><td>OpenAI</td><td>GPT-4.1 vs. GPT-4.1-mini</td><td>63.52</td><td>72.09</td><td>78.66</td><td>70.65</td><td>79.41</td><td>87.61</td></tr><tr><td>Llama</td><td>3.1-70B vs. 3.3-70B</td><td>62.95</td><td>60.20</td><td>59.51</td><td>61.45</td><td>66.40</td><td>72.22</td></tr><tr><td>Llama</td><td>3.1-70B vs. 4-Maverick</td><td>60.89</td><td>64.89</td><td>68.90</td><td>61.45</td><td>72.09</td><td>73.59</td></tr><tr><td>Llama</td><td>3.1-8B vs. 3.1-70B</td><td>55.63</td><td>58.89</td><td>58.70</td><td>57.07</td><td>61.39</td><td>65.33</td></tr><tr><td>Llama</td><td>3.1-8B vs. 3.3-70B</td><td>60.76</td><td>59.51</td><td>59.01</td><td>57.45</td><td>62.39</td><td>68.52</td></tr><tr><td>Llama</td><td>3.1-8B vs. 4-Maverick</td><td>58.82</td><td>64.14</td><td>68.27</td><td>59.39</td><td>70.96</td><td>70.34</td></tr><tr><td>Llama</td><td>3.3-70B vs. 4-Maverick</td><td>59.45</td><td>61.45</td><td>65.71</td><td>59.45</td><td>67.52</td><td>74.16</td></tr><tr><td>Qwen</td><td>8B vs. 14B</td><td>54.57</td><td>59.76</td><td>58.70</td><td>54.88</td><td>61.14</td><td>62.20</td></tr><tr><td>Qwen</td><td>8B vs. 32B</td><td>67.02</td><td>73.03</td><td>69.15</td><td>68.15</td><td>73.78</td><td>85.61</td></tr><tr><td>Qwen</td><td>14B vs. 32B</td><td>66.65</td><td>72.28</td><td>70.71</td><td>64.71</td><td>74.72</td><td>80.41</td></tr><tr><td>Anthropic</td><td>Haiku-4.5 vs. Sonnet-4</td><td>77.60</td><td>70.53</td><td>75.16</td><td>79.29</td><td>82.54</td><td>92.30</td></tr><tr><td>Gemini</td><td>2.5-Flash vs. 2.5-Flash-Lite</td><td>60.14</td><td>63.08</td><td>70.03</td><td>60.89</td><td>72.03</td><td>74.59</td></tr><tr><td>OpenAI</td><td>GPT-4.1 vs. GPT-4.1-mini</td><td>66.33</td><td>72.47</td><td>78.72</td><td>73.65</td><td>80.23</td><td>88.17</td></tr><tr><td>DeepSeek</td><td>V4 vs. V4-0731</td><td>61.39</td><td>64.27</td><td>67.90</td><td>60.89</td><td>65.89</td><td>72.47</td></tr><tr><td>Gemma</td><td>4-26B vs. 4-31B</td><td>54.44</td><td>62.14</td><td>78.79</td><td>62.58</td><td>78.35</td><td>78.91</td></tr><tr><td>Mistral</td><td>Small-24B vs. Small-3.2-24B</td><td>83.23</td><td>87.30</td><td>93.05</td><td>90.93</td><td>94.37</td><td>98.62</td></tr></table>

Table 15: Within-family attribution accuracy (%) for non-Llama model pairs using logistic regression. Each row is a binary attribution task, with chance accuracy of 50%.

Table 16: Within-family attribution accuracy (%) for all model pairs using a one-hidden-layer MLP. Each row is a binary attribution task, with chance accuracy of 50%.

<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>66.77</td><td rowspan=1 colspan=1>69.99</td><td rowspan=1 colspan=1>71.37</td><td rowspan=1 colspan=1>69.42</td><td rowspan=1 colspan=1>74.27</td><td rowspan=1 colspan=1>81.34</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>66.96</td><td rowspan=1 colspan=1>68.54</td><td rowspan=1 colspan=1>70.62</td><td rowspan=1 colspan=1>70.74</td><td rowspan=1 colspan=1>71.31</td><td rowspan=1 colspan=1>79.45</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>65.76</td><td rowspan=1 colspan=1>68.10</td><td rowspan=1 colspan=1>69.74</td><td rowspan=1 colspan=1>70.24</td><td rowspan=1 colspan=1>71.69</td><td rowspan=1 colspan=1>80.33</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>66.08</td><td rowspan=1 colspan=1>66.90</td><td rowspan=1 colspan=1>70.30</td><td rowspan=1 colspan=1>68.28</td><td rowspan=1 colspan=1>71.25</td><td rowspan=1 colspan=1>78.81</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>66.58</td><td rowspan=1 colspan=1>67.78</td><td rowspan=1 colspan=1>69.36</td><td rowspan=1 colspan=1>70.18</td><td rowspan=1 colspan=1>73.01</td><td rowspan=1 colspan=1>80.83</td></tr><tr><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>59.46</td><td rowspan=1 colspan=1>61.60</td><td rowspan=1 colspan=1>65.07</td><td rowspan=1 colspan=1>65.13</td><td rowspan=1 colspan=1>67.40</td><td rowspan=1 colspan=1>71.50</td></tr><tr><td rowspan=1 colspan=1>P1*</td><td rowspan=1 colspan=1>65.51</td><td rowspan=1 colspan=1>67.34</td><td rowspan=1 colspan=1>70.37</td><td rowspan=1 colspan=1>66.08</td><td rowspan=1 colspan=1>71.31</td><td rowspan=1 colspan=1>77.93</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>56.68</td><td rowspan=1 colspan=1>59.58</td><td rowspan=1 colspan=1>62.55</td><td rowspan=1 colspan=1>61.92</td><td rowspan=1 colspan=1>65.57</td><td rowspan=1 colspan=1>71.88</td></tr></table>

Table 17: Teacher–student attribution accuracy (%) for Llama-3.1-8B vs. Llama-3.1-Minitron-4B using logistic regression. Chance accuracy is 50%.
<table><tr><td>View</td><td>Behavioral</td><td>Stylometric</td><td>Lexical</td><td>Semantic</td><td>Combined</td><td>PN Embedding†</td></tr><tr><td>TO</td><td>99.44</td><td>99.50</td><td>99.12</td><td>99.44</td><td>99.69</td><td>99.87</td></tr><tr><td>T1</td><td>99.06</td><td>99.19</td><td>98.87</td><td>98.81</td><td>99.12</td><td>99.37</td></tr><tr><td>T2</td><td>98.25</td><td>98.81</td><td>97.87</td><td>98.31</td><td>98.68</td><td>99.25</td></tr><tr><td>T3</td><td>98.43</td><td>98.62</td><td>97.74</td><td>98.56</td><td>98.56</td><td>100.00</td></tr><tr><td>P1</td><td>99.56</td><td>99.50</td><td>99.06</td><td>99.44</td><td>99.69</td><td>99.87</td></tr><tr><td>P2</td><td>73.06</td><td>82.14</td><td>85.84</td><td>84.40</td><td>88.22</td><td>93.86</td></tr><tr><td>P1*</td><td>98.18</td><td>98.75</td><td>97.99</td><td>98.62</td><td>98.50</td><td>99.62</td></tr><tr><td>P2*</td><td>63.78</td><td>76.63</td><td>81.83</td><td>77.26</td><td>83.08</td><td>88.85</td></tr></table>

Table 18: Teacher–student attribution accuracy (%) for Mistral-NeMo-Instruct vs. NeMo-Minitron-8B using logistic regression. Chance accuracy is 50%.

## E Detailed Teacher–Student Attribution Results

Tables 17–19 report the complete results for the three teacher–student relationships. As in the withinfamily analysis, each comparison is binary with 50% chance accuracy and uses logistic regression. Reporting all response views allows us to examine whether the distinction between a teacher and its distilled descendant persists after progressively removing surface cues and rewriting the generated text.

## F Controlled LoRA Adaptations

For each model–dataset pair defined in the main text, we use 1,200 training examples after removing prompts that overlap with the D1–D4 evaluation sets. MATH examples are sampled across all seven subject categories. Training uses supervised next-token prediction on the assistant response, with prompt tokens masked from the loss.

Adapters are trained for one epoch with maximum sequence length 512, learning rate $2 \times 1 0 ^ { - 4 }$ , perdevice batch size 16, warmup ratio 0.03, and no weight decay. We use AdamW with cosine scheduling, FP16, and no quantization. LoRA uses rank $r \ = \ 1 6 .$ , scaling parameter 32, dropout 0.05, no bias adaptation, and all linear layers as target modules.

Each LoRA variant generates responses to the same D1–D4 prompts used in the main experiments. Generation uses temperature 1.0, $\mathrm { t o p } { - } p = 0 . 9$ , a maximum of 2,048 new tokens, one generation per prompt, and no additional system prompt. Qwen3 thinking is disabled. Generation is stochastic, and outputs are not treated as paired across LoRA strengths.

Attribution uses raw T0 responses, Logistic Regression, and the same five-fold prompt-grouped split as the main experiments. The 799 prompt blocks yield 1,598 responses per binary comparison.

<table><tr><td>View</td><td>Behavioral</td><td>Stylometric</td><td>Lexical</td><td>Semantic</td><td>Combined</td><td>PN Embedding†</td></tr><tr><td>TO</td><td>71.74</td><td>68.23</td><td>68.36</td><td>70.86</td><td>76.94</td><td>89.22</td></tr><tr><td>T1</td><td>71.18</td><td>68.30</td><td>67.04</td><td>71.30</td><td>77.07</td><td>87.59</td></tr><tr><td>T2</td><td>60.34</td><td>68.61</td><td>67.67</td><td>64.10</td><td>68.05</td><td>81.20</td></tr><tr><td>T3</td><td>58.65</td><td>66.73</td><td>66.85</td><td>60.28</td><td>68.42</td><td>82.58</td></tr><tr><td>P1</td><td>70.99</td><td>67.54</td><td>65.79</td><td>69.24</td><td>76.32</td><td>88.22</td></tr><tr><td>P2</td><td>70.18</td><td>63.91</td><td>63.35</td><td>63.16</td><td>71.93</td><td>81.08</td></tr><tr><td>P1*</td><td>57.58</td><td>64.54</td><td>64.72</td><td>60.09</td><td>66.17</td><td>81.95</td></tr><tr><td>P2*</td><td>56.39</td><td>62.16</td><td>59.71</td><td>60.84</td><td>62.84</td><td>76.82</td></tr></table>

Table 19: Teacher–student attribution accuracy (%) for Gemma-2-27B vs. Gemma-2-2B using logistic regression. Chance accuracy is 50%.
<table><tr><td rowspan="2">Feature</td><td colspan="2">Llama-3.1-8B</td><td colspan="2">Qwen3-8B</td></tr><tr><td>vs Base</td><td>vs Previous</td><td>vs Base</td><td>vs Previous</td></tr><tr><td>Behavioral</td><td>59.26</td><td>52.52</td><td>68.44</td><td>56.56</td></tr><tr><td>Stylometric</td><td>58.37</td><td>51.48</td><td>68.89</td><td>55.99</td></tr><tr><td>Lexical</td><td>60.20</td><td>51.54</td><td>70.46</td><td>56.51</td></tr><tr><td>Semantic</td><td>58.48</td><td>51.07</td><td>69.47</td><td>55.35</td></tr></table>

Table 20: LoRA-variant attribution accuracy (%) using logistic regression. vs Base distinguishes each adapted model from its unadapted base model, whereas vs Previous distinguishes each LoRA strength from the immediately preceding strength. Results are averaged across GSM8K and MATH adaptations and all LoRA-strength comparisons. Binary chance accuracy is 50%.

The shared 0-vs-0.05 comparison is evaluated once, resulting in 52 unique binary tasks across the four model–dataset settings. Behavioral, Stylometric, Lexical, Semantic, PN Embedding, and Combined representations use the same definitions as in the main experiments.

Table 20 reports results for the four standard signal groups, averaged across GSM8K and MATH and across all strength comparisons. The feature-level results follow the pattern observed in Figure 2, with substantially stronger separation from the base model than between neighboring variants across signal groups.

## G Evidence-Amount

Within-response evidence experiments use exact windows of $N \in \{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 \}$ words. The center window is used as the primary location, with additional prefix, center, and suffix comparisons at selected budgets. Figure 4 shows Logistic Regression accuracy for T0 (raw text) and T3 in the eightway attribution setting as the number of available words increases. For this experiment, we include only responses containing at least 400 words so that the same set of responses is evaluated across all evidence budgets.

Because this analysis is restricted to responses containing at least 400 words so that the evaluation set remains fixed across word budgets, its Full-response value is not expected to exactly match the allresponse result in Table 2.

![](images/e750f6439ef7f49d8418a6152cc218646a893885fa1098d866448a87ade2279b.jpg)  
(a) Original outputs

![](images/a6c8e741a70c2357092599f96291edf04b3a8279d5c05f3ae9c13c21f64f1f3f.jpg)  
(b) T3

Figure 4: Attribution accuracy as a function of available text in the 8-way cross-family setting. (a) Original model outputs. (b) Outputs after the T3 surface transformation. Prefix, center, and suffix indicate the location from which the available words are selected.
<table><tr><td rowspan=1 colspan=7>Model              Explanation Argument Creative Rewrite Synthesis</td></tr><tr><td></td><td rowspan=1 colspan=1>Llama-4 Maverick</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>55.00</td><td rowspan=1 colspan=1>90.00</td></tr><tr><td></td><td rowspan=1 colspan=1>Qwen3-32B</td><td rowspan=1 colspan=1>80.00</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>82.50</td><td rowspan=1 colspan=1>12.50</td><td rowspan=1 colspan=1>40.00</td></tr><tr><td rowspan=1 colspan=2>Mistral Small 3.2</td><td rowspan=1 colspan=1>70.00</td><td rowspan=1 colspan=1>92.50</td><td rowspan=1 colspan=1>75.00</td><td rowspan=1 colspan=1>47.50</td><td rowspan=1 colspan=1>20.00</td></tr><tr><td rowspan=1 colspan=2>Gemma-4-31B</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>85.00</td><td rowspan=1 colspan=1>30.00</td><td rowspan=1 colspan=1>12.50</td></tr><tr><td></td><td rowspan=1 colspan=1>DeepSeek-V4</td><td rowspan=1 colspan=1>87.50</td><td rowspan=1 colspan=1>92.50</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>15.00</td><td rowspan=1 colspan=1>15.00</td></tr><tr><td></td><td rowspan=1 colspan=1>Claude Haiku 4.5</td><td rowspan=1 colspan=1>97.50</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>95.00</td><td rowspan=1 colspan=1>75.00</td><td rowspan=1 colspan=1>100.00</td></tr><tr><td rowspan=2 colspan=2>GPT-4.1</td><td rowspan=1 colspan=1>Gemini-2.5 Flash</td><td rowspan=1 colspan=1>77.50</td><td rowspan=1 colspan=1>100.00</td><td rowspan=1 colspan=1>90.00</td><td rowspan=1 colspan=1>40.00</td></tr><tr><td rowspan=1 colspan=1>67.50</td><td rowspan=1 colspan=1>82.50</td><td rowspan=1 colspan=1>75.00</td><td rowspan=1 colspan=1>12.50</td><td rowspan=1 colspan=1>15.00</td></tr></table>

Table 21: Per-model recall (%) across the five D3 task categories under cross-dataset generalization. The classifier is trained on D1, D2, and D4 and evaluated on D3. Chance recall is 12.5%.

## H Task-Level Variation and Prompt Effects in Model Attribution

## H.1 Does Removing Prompt Effects from Embeddings Improve Source Attribution?

We hypothesize that subtracting the prompt-wise mean response embedding reduces variation shared across candidate models, much of which reflects the prompt and task, and makes model-specific differences easier to detect. Table 23 supports this hypothesis. Centering increases the overall accuracy of embedding-based attribution from 50.50% to 74.88%.

The comparison with the other representations is particularly informative. PN Embedding is comparable to Lexical on explanation and argument/advice tasks and slightly stronger on creative writing. Its clearest advantage appears on the more constrained tasks. For rewriting and source synthesis, PN Embedding exceeds the strongest standard representation by 12.81 and 13.75 points, respectively. These tasks restrict the content and structure of the response, causing outputs from different models to share more task-related information. Removing this shared component exposes model-specific variation that is less visible in the original embedding space. Because centering requires responses from all candidate models to the same prompt, it is best understood as a prompt-relative analysis of model separation rather than a standard attribution setting.

## H.2 D3 Task-Level Attribution by Model

Table 21 reports per-model recall across the five D3 task categories under cross-dataset generalization. The results show substantial variation across task types and models. Recall is generally higher for explanation, argument/advice, and creative generation, whereas rewrite and source-synthesis tasks show lower recall for several models. Thus, the aggregate task-level differences in the main text are not uniform across all models. These differences also have practical implications for provenance analysis. In rewrite and source-synthesis workflows, the lower attribution accuracy may make it more difficult to identify the model responsible for a given output after it has been transformed or conditioned on source material. Conversely, when model provenance is important, task-specific attribution performance can be considered when evaluating or auditing model outputs rather than relying on results from more open-ended generation alone.

(a) Mixed-source evaluation
<table><tr><td rowspan=1 colspan=7>View BehavioralStylometricLexicalSemanticCombined PN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>52.42</td><td rowspan=1 colspan=1>60.72</td><td rowspan=1 colspan=1>80.07</td><td rowspan=1 colspan=1>66.04</td><td rowspan=1 colspan=1>81.41</td><td rowspan=1 colspan=1>83.75</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>25.59</td><td rowspan=1 colspan=1>49.62</td><td rowspan=1 colspan=1>67.76</td><td rowspan=1 colspan=1>50.94</td><td rowspan=1 colspan=1>67.04</td><td rowspan=1 colspan=1>69.35</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>23.08</td><td rowspan=1 colspan=1>41.79</td><td rowspan=1 colspan=1>53.94</td><td rowspan=1 colspan=1>41.90</td><td rowspan=1 colspan=1>55.87</td><td rowspan=1 colspan=1>59.39</td></tr></table>

(b) Held-out-dataset evaluation
<table><tr><td rowspan=1 colspan=7>View BehavioralStylometricLexicalSemanticCombined PN Embedding†</td></tr><tr><td rowspan=1 colspan=1>T0</td><td rowspan=1 colspan=1>50.70</td><td rowspan=1 colspan=1>56.83</td><td rowspan=1 colspan=1>75.45</td><td rowspan=1 colspan=1>61.13</td><td rowspan=1 colspan=1>75.52</td><td rowspan=1 colspan=1>80.75</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>25.14</td><td rowspan=1 colspan=1>47.13</td><td rowspan=1 colspan=1>64.32</td><td rowspan=1 colspan=1>46.05</td><td rowspan=1 colspan=1>62.25</td><td rowspan=1 colspan=1>66.45</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>22.86</td><td rowspan=1 colspan=1>39.05</td><td rowspan=1 colspan=1>49.61</td><td rowspan=1 colspan=1>38.51</td><td rowspan=1 colspan=1>51.10</td><td rowspan=1 colspan=1>56.29</td></tr></table>

Table 22: Attribution accuracy (%) under mixed-source and held-out-dataset evaluation. T0 uses the original responses, T3 removes surface cues, and $\mathrm { P } 2 ^ { * }$ combines full paraphrasing with surface normalization. Chance accuracy is 12.5%.

Table 22 compares mixed-source and held-out-dataset accuracy under the three response views.

## H.3 Multilingual Prompt Translation

For the multilingual experiments, we translate the D1–D4 English prompts into Persian and Mandarin Chinese before generating model responses. Translation uses gpt-4o-mini-2024-07-18 with temperature 0, top-p = 1, and a maximum of 4,096 output tokens. Persian translations use standard Persian script, while Mandarin translations use Simplified Chinese characters.

The translation procedure is designed to preserve the original task rather than adapt or execute it. The model is instructed to return only the translated prompt, preserve names, dates, numbers, constraints, requested tone, and meaningful line breaks, and retain literal fragments such as code, URLs, identifiers, and placeholders when appropriate. Prompts requesting code, diagrams, images, or other artifacts are translated as instructions rather than executed. We use the following system prompt, where [TARGET LANGUAGE] is replaced by the corresponding language specification:

You are a translation engine. Translate the user’s entire prompt into   
[TARGET LANGUAGE].   
STRICT OUTPUT CONTRACT:   
- Return ONLY the translated prompt itself and absolutely nothing   
else.   
- Do not add any introduction, explanation, note, label, quotation   
marks, or markdown fence.   
- NEVER write phrases such as "Here is the translation", "Translated   
prompt", "Translation:", or anything similar.   
- The first character of your response must be the first character   
of the translated prompt, and the last character must be the last   
character of the translated prompt.   
TRANSLATION RULES:   
- Translate the instruction; do NOT answer it, execute it, solve it,   
or follow it.   
- Some prompts request code, ASCII art, diagrams, images, scripts,   
or other non-prose outputs. Do NOT generate any requested code,   
ASCII art, diagram, image description output, or executable content.   
Translate the request itself as ordinary text for a downstream model.   
- If literal code, ASCII, a URL, HTML/XML, an identifier, or a   
placeholder already appears inside the source prompt, preserve that   
literal fragment when necessary and translate only the surrounding   
natural language. Do not invent new code or ASCII.   
- Preserve names, dates, numbers, constraints, requested tone, and   
meaningful line breaks.   
- Do not summarize, censor, improve, embellish, or rewrite the task   
beyond faithful translation.

After translation, the eight representative models generate responses to the translated prompts using the same decoding configuration as the main English experiment, with temperature 1.0, top-p = 0.9, and a maximum of 2,048 output tokens. Reasoning is disabled so that the generation protocol remains consistent across languages.

## I Within-Language Attribution in Persian and Chinese

Tables 24 and 25 extend the English eight-way classification results in Table 2 to Persian and Chinese.

## J Cross-Lingual Fingerprint Transfer

To examine whether model-specific fingerprints persist across languages, we train the attribution classifier on outputs in one language and evaluate it on outputs from another language. We consider all six directed transfers among English (EN), Persian (FA), and Mandarin Chinese (ZH). Evaluation uses the common set of 798 prompts and five prompt-disjoint folds, such that the translated counterpart of a target-language test prompt is never present in the source-language training split. Logistic-regression hyperparameters are selected using only the source-language training data. TF–IDF is excluded because its vocabulary is intrinsically tied to language and script. Table 26 averages accuracy over all six transfer directions.

For cross-lingual experiments, we retain the same feature definition across languages to avoid changing the classifier representation between source and target languages. Consequently, some Stylometric

<table><tr><td>Task</td><td>Behavioral</td><td>Stylometric</td><td>Lexical</td><td>Semantic</td><td>Combined</td><td>PN Embedding†</td></tr><tr><td>Explanation</td><td>62.19</td><td>72.81</td><td>87.50</td><td>55.00</td><td>84.69</td><td>86.56</td></tr><tr><td>Argument / Advice</td><td>82.19</td><td>87.81</td><td>96.25</td><td>72.19</td><td>95.31</td><td>93.75</td></tr><tr><td>Creative</td><td>53.13</td><td>64.06</td><td>90.63</td><td>68.75</td><td>87.81</td><td>92.19</td></tr><tr><td>Rewrite</td><td>29.06</td><td>25.94</td><td>27.81</td><td>30.31</td><td>35.94</td><td>48.75</td></tr><tr><td>Source synthesis</td><td>28.44</td><td>26.56</td><td>34.69</td><td>26.25</td><td>39.38</td><td>53.13</td></tr><tr><td>Overall</td><td>51.00</td><td>55.44</td><td>67.38</td><td>50.50</td><td>68.63</td><td>74.88</td></tr></table>

Table 23: Attribution accuracy (%) by D3 task category using the original responses. Logistic-regression classifiers are trained on D1, D2, and D4 and evaluated on the held-out D3 dataset. Chance accuracy is 12.5%.
<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>51.60</td><td rowspan=1 colspan=1>57.69</td><td rowspan=1 colspan=1>86.98</td><td rowspan=1 colspan=1>69.60</td><td rowspan=1 colspan=1>86.40</td><td rowspan=1 colspan=1>79.87</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>48.40</td><td rowspan=1 colspan=1>60.42</td><td rowspan=1 colspan=1>84.27</td><td rowspan=1 colspan=1>64.47</td><td rowspan=1 colspan=1>82.97</td><td rowspan=1 colspan=1>74.04</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>41.90</td><td rowspan=1 colspan=1>59.99</td><td rowspan=1 colspan=1>80.61</td><td rowspan=1 colspan=1>60.65</td><td rowspan=1 colspan=1>79.20</td><td rowspan=1 colspan=1>70.74</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>30.59</td><td rowspan=1 colspan=1>55.28</td><td rowspan=1 colspan=1>78.85</td><td rowspan=1 colspan=1>56.20</td><td rowspan=1 colspan=1>77.05</td><td rowspan=1 colspan=1>66.65</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>50.31</td><td rowspan=1 colspan=1>57.47</td><td rowspan=1 colspan=1>83.27</td><td rowspan=1 colspan=1>67.50</td><td rowspan=1 colspan=1>83.24</td><td rowspan=1 colspan=1>77.85</td></tr><tr><td rowspan=2 colspan=1>P2P1*</td><td rowspan=1 colspan=1>47.51</td><td rowspan=1 colspan=1>53.98</td><td rowspan=1 colspan=1>79.03</td><td rowspan=1 colspan=1>61.83</td><td rowspan=1 colspan=1>79.26</td><td rowspan=1 colspan=1>72.37</td></tr><tr><td rowspan=1 colspan=1>29.28</td><td rowspan=1 colspan=1>52.38</td><td rowspan=1 colspan=1>73.34</td><td rowspan=1 colspan=1>53.65</td><td rowspan=1 colspan=1>72.49</td><td rowspan=1 colspan=1>63.28</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>27.22</td><td rowspan=1 colspan=1>46.22</td><td rowspan=1 colspan=1>67.87</td><td rowspan=1 colspan=1>46.57</td><td rowspan=1 colspan=1>66.53</td><td rowspan=1 colspan=1>56.86</td></tr></table>

Table 24: Within-language 8-way cross-family attribution accuracy (%) on Persian outputs using logistic regression. Chance accuracy is 12.5%.  
components, including the English-trained POS representation, are not linguistically calibrated for Persian or Chinese. We therefore treat cross-lingual Stylometric accuracy as a fixed-feature diagnostic and do not interpret it as evidence of language-general POS structure.

<table><tr><td rowspan=1 colspan=7>View Behavioral Stylometric Lexical Semantic CombinedPN Embedding†</td></tr><tr><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>54.57</td><td rowspan=1 colspan=1>59.01</td><td rowspan=1 colspan=1>83.79</td><td rowspan=1 colspan=1>68.48</td><td rowspan=1 colspan=1>83.62</td><td rowspan=1 colspan=1>83.76</td></tr><tr><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>51.92</td><td rowspan=1 colspan=1>57.90</td><td rowspan=1 colspan=1>80.32</td><td rowspan=1 colspan=1>63.80</td><td rowspan=1 colspan=1>79.93</td><td rowspan=1 colspan=1>77.32</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>43.96</td><td rowspan=1 colspan=1>56.82</td><td rowspan=1 colspan=1>77.61</td><td rowspan=1 colspan=1>59.82</td><td rowspan=1 colspan=1>76.41</td><td rowspan=1 colspan=1>75.42</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>29.08</td><td rowspan=1 colspan=1>50.78</td><td rowspan=1 colspan=1>73.98</td><td rowspan=1 colspan=1>55.55</td><td rowspan=1 colspan=1>72.50</td><td rowspan=1 colspan=1>71.28</td></tr><tr><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>53.52</td><td rowspan=1 colspan=1>57.45</td><td rowspan=1 colspan=1>80.74</td><td rowspan=1 colspan=1>66.96</td><td rowspan=1 colspan=2>81.12          81.26</td></tr><tr><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>51.99</td><td rowspan=1 colspan=1>56.81</td><td rowspan=1 colspan=1>78.80</td><td rowspan=1 colspan=1>64.60</td><td rowspan=1 colspan=2>79.46         79.13</td></tr><tr><td rowspan=1 colspan=1>P1*</td><td rowspan=1 colspan=1>28.60</td><td rowspan=1 colspan=1>49.09</td><td rowspan=1 colspan=1>69.07</td><td rowspan=1 colspan=1>52.39</td><td rowspan=1 colspan=2>69.16         68.85</td></tr><tr><td rowspan=1 colspan=1>P2*</td><td rowspan=1 colspan=1>27.86</td><td rowspan=1 colspan=1>48.25</td><td rowspan=1 colspan=1>65.85</td><td rowspan=1 colspan=1>51.66</td><td rowspan=1 colspan=2>67.27         66.79</td></tr></table>

Table 25: Within-language 8-way cross-family attribution accuracy (%) on Chinese outputs using logistic regression. Chance accuracy is 12.5%.

<table><tr><td>View</td><td>Behavioral</td><td>Stylometric</td><td>Semantic</td><td>Combined</td><td>Best Deployable</td><td>PN Embedding†</td></tr><tr><td>TO</td><td>29.65</td><td>19.74</td><td>30.58</td><td>33.36</td><td>34.56</td><td>53.03</td></tr><tr><td>T1</td><td>27.34</td><td>19.88</td><td>29.80</td><td>31.94</td><td>33.23</td><td>48.57</td></tr><tr><td>T2</td><td>21.69</td><td>20.51</td><td>25.86</td><td>24.99</td><td>28.49</td><td>44.20</td></tr><tr><td>T3</td><td>19.42</td><td>18.56</td><td>24.21</td><td>24.10</td><td>26.66</td><td>39.00</td></tr><tr><td>P1</td><td>28.78</td><td>20.58</td><td>31.11</td><td>33.62</td><td>34.22</td><td>53.34</td></tr><tr><td>P2</td><td>29.95</td><td>19.36</td><td>30.60</td><td>33.88</td><td>35.31</td><td>50.21</td></tr><tr><td>P1*</td><td>18.59</td><td>18.65</td><td>23.30</td><td>23.40</td><td>25.54</td><td>37.85</td></tr><tr><td>P2*</td><td>17.54</td><td>18.09</td><td>23.26</td><td>23.08</td><td>24.48</td><td>36.64</td></tr></table>

Table 26: Average cross-lingual 8-way attribution accuracy (%) across the six directed language-transfer settings using logistic regression. Chance accuracy is 12.5%.