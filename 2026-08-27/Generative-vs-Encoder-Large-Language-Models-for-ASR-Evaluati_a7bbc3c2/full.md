# Generative vs. Encoder Large Language Models for ASR Evaluation: A Comparative Study

Thibault Baneras-Roux˜ <sup>1</sup>, Shashi Kumar<sup>1</sup>, Driss Khalil<sup>1</sup>, Sergio Burdisso<sup>1</sup>, Petr Motlicek<sup>1</sup>, Shiran Liu<sup>1</sup>, Mickael Rouvier<sup>2</sup>, Jane Wottawa<sup>3</sup>, Richard Dufour<sup>4</sup>

<sup>1</sup>Idiap Research Institute, Martigny, Switzerland <sup>2</sup>LIA, Avignon Universite, Avignon, France ´ <sup>3</sup>LIUM, Le Mans Universite, Le Mans, France´ <sup>4</sup>LS2N, Nantes Universite, Nantes, France´

Abstract—Automatic Speech Recognition (ASR) is typically evaluated using Word Error Rate (WER), which poorly reflects semantic similarity. While embedding-based metrics correlate better with human judgments, the respective roles of encoder and decoder-based Large Language Models (LLMs) remain underexplored. This paper presents a comparative study of both families for ASR evaluation. We analyze BERTScore and SemDist across different LLMs, layers, and pooling strategies, showing that both metrics can achieve strong correlation with human judgments when properly configured. For decoder models, we investigate generative LLMs in two settings: pairwise hypothesis selection via prompting and direct qualitative error classification. Our results show that encoder-based metrics remain highly competitive, while generative LLMs perform strongly in hypothesis comparison and improve the interpretability of ASR evaluation.

Index Terms—Automatic Speech Recognition, Large Language Models, Semantic Evaluation, BERTScore, SemDist, Human Perception

## I. INTRODUCTION

Automatic Speech Recognition (ASR) plays a central role in applications such as voice interfaces, transcription services, and accessibility tools. Because of its broad impact, robust evaluation methodologies are crucial for tracking progress and guiding system development. Although Word Error Rate (WER) has long been the standard metric, it remains limited by its strict matching rules and sensitivity to surface-form variations such as casing or minor lexical differences, which can obscure meaningful improvements in perceived transcription quality.

To address these shortcomings, a variety of evaluation approaches have been introduced. In particular, semantic metrics [1], [2] leveraging contextualized word representations have demonstrated stronger alignment with human judgments. Methods based on encoder architectures such as BERT [3] and related models [1], [4]–[6] have been especially influential in capturing contextual meaning beyond surface-level text overlap. However, comparatively little attention has been given to decoder-only large language models, including GPT [7], Llama [8], and Gemma [9], despite their strong generalpurpose reasoning and generation capabilities [10]–[12]. This gap raises the question of how encoder and decoder-based representations compare in structured ASR evaluation settings.

In this work, we systematically investigate the role of both encoder and decoder representations for ASR evaluation along complementary dimensions. First, we compare semantic representations derived from both encoder models and decoder LLMs, studying how different pooling strategies affect their ability to encode meaningful sentence-level similarity. Second, we examine the ”LLM-as-judge” paradigm, assessing whether decoder-based models can directly discriminate between competing transcription hypotheses by leveraging their contextual understanding.

Our evaluation is conducted on the HATS dataset [13], which provides detailed human judgments of ASR outputs and has shown that semantic metrics correlate more strongly with perception than WER. We analyze (1) semantic similarity metrics derived from encoder versus decoder embeddings under different pooling strategies, (2) pairwise hypothesis selection, and (3) the ability of LLMs to produce qualitative error assessments. Our findings indicate that decoder-based models can rival or surpass strong encoder-based metrics, while both families yields improved robustness and interpretability in comparison with standard ASR evaluation.

## II. RELATED WORK

## A. From Lexical to Semantic ASR Evaluation

Automatic Speech Recognition systems are commonly evaluated using Word Error Rate, which measures the minimum number of insertions, deletions, and substitutions required to transform a hypothesis into its reference. Owing to its simplicity and reproducibility, WER has become the de facto standard for ASR evaluation. However, numerous studies have highlighted its limitations [14]–[16]. Since every token error contributes equally to the final score, WER ignores the semantic importance of individual mistakes, penalizing harmless wording changes as severely as meaning-altering errors.

These shortcomings become particularly apparent when considering downstream applications. Improvements in WER do not necessarily translate into better performance in tasks that consume ASR transcripts [17], nor do they consistently reflect users’ perception of transcription quality [18], [19]. As a consequence, different evaluation metrics may rank ASR systems differently [20], suggesting that the choice of metric should depend on the intended application, especially when transcripts are designed for humans [21], [22].

## B. Embedding-based Semantic Metrics

To better account for meaning preservation, several semantic evaluation metrics have been proposed. Early approaches relied on static word embeddings [5], while more recent methods exploit contextual representations produced by Transformer encoders such as BERT [3]. These contextual embeddings capture sentence meaning beyond lexical overlap, making them particularly suitable for evaluating ASR hypotheses.

Among the most widely used semantic metrics are BERTScore [2], which compares contextual token embeddings, and SemDist [1], which measures the cosine distance between sentence embeddings. Both have been shown to correlate substantially better with human judgments than WER [1]. Their performance, however, depends on the underlying language model and on the representation extracted from it, motivating a deeper analysis of embedding choices.

Despite these improvements, embedding-based metrics remain difficult to interpret. Unlike WER or CER, which are expressed as intuitive error rates, semantic metrics generally produce cosine similarities without an obvious semantic interpretation. Several works have therefore proposed incorporating semantic severity into lexical metrics [6], [23] or analysing ASR errors according to their semantic impact [24], providing more informative evaluations than a single similarity score.

## C. Human-centered ASR Evaluation

A complementary research direction aims to design evaluation metrics that better reflect human perception [25], [26]. This has motivated the collection of datasets containing explicit human judgments of transcription quality, enabling direct comparison between automatic metrics and perceived quality.

The HATS dataset [13] is one such benchmark, providing pairwise human preferences over French ASR hypotheses. It demonstrated that semantic metrics achieve substantially higher agreement with human annotators than WER, confirming observations previously reported on proprietary datasets [18]. More recently, [19] showed that Character Error Rate (CER) can itself provide a stronger approximation of human judgments than WER in multilingual settings.

While previous work has primarily focused on encoderbased semantic representations, the emergence of decoderbased Large Language Models raises the question of whether generative models can also serve as effective ASR evaluation tools. This paper investigates both encoder and decoder LLMs, comparing their representations for semantic metrics and evaluating the ability of generative models to directly assess transcription quality.

## III. EMBEDDING-BASED ASR EVALUATION

As said previously, WER poorly reflects human perception. While embedding-based metrics correlate better with human judgments, their behavior depends strongly on the underlying representations and extraction strategy. In this work, we study encoder and decoder-based LLM embeddings for semantics metrics according to human expectations.

## A. Evaluation Setup

We rely on the HATS dataset as a reference for comparing automatic metrics against human judgments. Following prior work, we consider three subsets based on inter-annotator agreement: (i) 100% agreement, (ii) ≥70% agreement, and (iii) the full dataset. This allows us to analyze metric robustness under varying levels of annotation ambiguity. Agreement is computed as:

$$
{ \mathrm { A g r e e m e n t } } = { \frac { \operatorname* { m a x } ( A , B ) } { A + B } } .
$$

As a baseline, WER and CER achieve agreements of 63% and 77% respectively on the 100% agreement subset. Since embedding-based metrics produce continuous similarity scores, ties do not occur in our experiments, unlike discrete lexical metrics such as WER or CER.

## B. BERTScore with Encoder and Decoder LLMs

![](images/8de0b07eb4729da6a11d5137637134018f47ae06a42372864c134bdb6e779626.jpg)  
Fig. 1: Agreement with human judgments of BERTScore metric across models and representation layers. Human annotations are obtained from the HATS dataset subset with annotator consensus.

To investigate the influence of contextual representations on semantic evaluation, we compute BERTScore using the embeddings extracted from every transformer layer of each evaluated encoder and decoder LLM. The correlation between the resulting BERTScore values and human judgments on the HATS dataset is then measured, allowing us to identify which models and representation layers best capture the semantic similarity perceived by human annotators.

Figure 1 reports the agreement obtained for each model and layer. A first observation is that the optimal representation layer strongly depends on the model. Although intermediate layers often provide the highest agreement, the final layer remains optimal for several models, making it difficult to predict a priori which layer should be used. Consequently, layer selection should be regarded as an important hyperparameter when applying BERTScore for ASR evaluation.

Comparing the encoder models reveals that sentence-level fine-tuning primarily affects the last layers of the network, while earlier layers remain remarkably similar to those of the corresponding CamemBERT models. This specialization can lead to substantial improvements, as illustrated by Sentence-CamemBERT-large, but it is not systematic: Sentence-CamemBERT-base has decreasing performances in comparison with CamemBERT-base, suggesting that sentencelevel fine-tuning is beneficial only when the underlying representation is sufficiently expressive.

A similar behavior is observed for decoder models. The embedding-specialized Qwen3 models generally improve over the original model, with the largest differences appearing in the deepest layers. However, the last layers do not seem to be the most suitable representations for BERTScore.

Overall, both encoder and decoder-based LLMs achieve strong performance when an appropriate representation layer is selected. The best results are obtained with Sentence-CamemBERT-large and Qwen3-Embedding-8B, demonstrating that dedicated sentence or embedding-oriented fine-tuning can be benefic to both architectures. Interestingly, encoder models achieve performance comparable to the best decoder models despite having substantially fewer parameters, highlighting their efficiency for embedding-based ASR evaluation.

<table><tr><td>Model</td><td></td><td>Last Mean Mean*</td><td></td><td>Wgt. Wgt.*</td><td></td></tr><tr><td>Gemma-2-2b</td><td>73</td><td>79</td><td>79</td><td>79</td><td>78</td></tr><tr><td>Gemma-2b</td><td>76</td><td>80</td><td>80</td><td>79</td><td>80</td></tr><tr><td>Gemma-3-1b-pt</td><td>75</td><td>76</td><td>74</td><td>77</td><td>76</td></tr><tr><td>Gemma-3-27b-it</td><td>77</td><td>83</td><td>83</td><td>81</td><td>82</td></tr><tr><td>Gemma-4-31B</td><td>72</td><td>75</td><td>73</td><td>77</td><td>76</td></tr><tr><td>Gemma-7b</td><td>65</td><td>69</td><td>71</td><td>71</td><td>72</td></tr><tr><td>Qwen3-0.6B</td><td>70</td><td>80</td><td>80</td><td>78</td><td>78</td></tr><tr><td>Qwen3-0.6B-Base</td><td>74</td><td>79</td><td>79</td><td>81</td><td>80</td></tr><tr><td>Qwen3-1.7B</td><td>72</td><td>83</td><td>83</td><td>80</td><td>81</td></tr><tr><td>Qwen3-1.7B-Base</td><td>74</td><td>82</td><td>81</td><td>81</td><td>82</td></tr><tr><td>Qwen3-30B</td><td>72</td><td>84</td><td>84</td><td>83</td><td>82</td></tr><tr><td>Qwen3-4B</td><td>77</td><td>85</td><td>83</td><td>82</td><td>81</td></tr><tr><td>Qwen3-4B-Base</td><td>74</td><td>82</td><td>83</td><td>82</td><td>82</td></tr><tr><td>Qwen3-8B</td><td>75</td><td>85</td><td>83</td><td>83</td><td>83</td></tr><tr><td>Qwen3-8B-Base</td><td>71</td><td>78</td><td>78</td><td>80</td><td>77</td></tr><tr><td>Qwen3-Embedding-0.6B</td><td>88</td><td>84</td><td>85</td><td>85</td><td>84</td></tr><tr><td>Qwen3-Embedding-4B</td><td>88</td><td>86</td><td>85</td><td>86</td><td>84</td></tr><tr><td>Qwen3-Embedding-8B</td><td>88</td><td>89</td><td>87</td><td>89</td><td>87</td></tr><tr><td>Qwen3.5-27B</td><td>59</td><td>77</td><td>76</td><td>74</td><td>76</td></tr><tr><td>Qwen3.5-35B</td><td>68</td><td>68</td><td>68</td><td>67</td><td>68</td></tr><tr><td>CamemBERT-Base</td><td>83</td><td>84</td><td>84</td><td>83</td><td>83</td></tr><tr><td>CamemBERT-Large</td><td>74</td><td>82</td><td>81</td><td>80</td><td>80</td></tr><tr><td>Sentence-CamemBERT-Base</td><td>84</td><td>86</td><td>86</td><td>85</td><td>85</td></tr><tr><td>Sentence-CamemBERT-Large</td><td>89</td><td>90</td><td>89</td><td>88</td><td>88</td></tr><tr><td>Mean</td><td>74</td><td>80</td><td>80</td><td>80</td><td>80</td></tr></table>

TABLE I: Agreement with human judgments of SemDist with different models and pooling strategies, using last-layer representations. Human annotation are obtained from the HATS dataset subset with annotator consensus.

## C. SemDist with Encoder and Decoder LLMs

We evaluate semantic similarity using the SemDist metric, defined as a distance derived from cosine similarity between sentence-level embeddings obtained by aggregating token representations from LLMs.

Given a sequence of token embeddings produced by an LLM, a fixed-size sentence representation is obtained via pooling. We evaluate multiple strategies, where $t _ { i }$ denotes the embedding of token i in a sequence of length n:

• Last token: $t _ { n }$

• Mean: $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } t _ { i }$

• Mean without last: $\begin{array} { r } { \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } t _ { i } } \end{array}$

• Weighted mean: $\frac { \sum _ { i = 1 } ^ { n ^ { * } } i \dot { t } _ { i } } { \sum _ { i = 1 } ^ { n } i }$

• Weighted mean without last: $\frac { \sum _ { i = 1 } ^ { n - 1 } i t _ { i } } { \sum _ { i = 1 } ^ { n - 1 } i }$

We evaluate all combinations of models, layers, and pooling strategies in terms of correlation with human annotations on the HATS dataset. Table I summarizes performance across models and pooling methods.

Overall, the results indicate that model size alone is not a reliable predictor of performance. While larger models can be beneficial in some cases, smaller or mid-sized models can achieve comparable or even better correlations depending on the layer and pooling strategy. This suggests that the quality of semantic representations is driven more by architectural and training choices than by scale alone.

A second observation concerns pooling strategies. Excluding embedding-specialized models, last-token representations are generally suboptimal. This is consistent with the fact that autoregressive models are trained to predict the next token rather than to encode global sentence meaning in the final hidden state. In contrast, mean and weighted pooling are consistently strong, suggesting that aggregating token-level information provides a more stable estimate of sentence-level semantics, especially when hypothesis and reference lengths are comparable.

As expected, embedding-specialized models behave differently: particularly for Qwen models, the last-token representation becomes significantly more competitive, often matching or surpassing alternative pooling strategies. This confirms that embedding-oriented fine-tuning can reshape the role of the final hidden state, making it more suitable as a global semantic representation.

The results in Table I focus on the impact of model choice and pooling strategies when using representations from the last transformer layer. This allows us to isolate how different aggregation methods influence SemDist performance under a fixed representation depth. In contrast, Figure 2 extends this analysis by varying the representation layer, enabling a more comprehensive view of how both pooling strategy and layer depth jointly affect performance across encoder and decoder LLMs.

We observe that the optimal layer is highly modeldependent: neither early nor final layers are universally optimal, and intermediate layers are not systematically superior. This makes layer selection a critical design choice when using LLM embeddings for semantic evaluation.

When comparing architectures, both encoder and decoderbased models can achieve strong performance under appropriate layer and pooling configurations. Encoder models such as Sentence-CamemBERT-large achieve consistently high performance with relatively compact architectures, while decoderbased embedding models such as Qwen3-Embedding-8B reach similar or slightly higher scores. However, this comes at a significantly higher parameter cost in many cases, suggesting that encoders remain highly competitive in terms of efficiency.

![](images/93d2af95dce3c8eb9433fab604c3b989bcdcdc9e5b81bf1b2a8d20dc483bb389.jpg)  
Fig. 2: Agreement of human judgments of SemDist across encoder and decoder LLMs as a function of layer depth (from the 10th-to-last layer to the last layer). Each cell reports the best pooling strategy. Human annotation are obtained from the HATS dataset subset with annotator consensus.

The best overall results are obtained with Sentence-CamemBERT-large and Qwen3-Embedding-8B. Interestingly, while these models are consistently strong, the ranking of intermediate models is not stable across pooling strategies or layers. For instance, the relative ordering of Qwen variants can differ between SemDist and BERTScore, indicating that different metrics emphasize different aspects of the learned representations.

As suggested in Table I, Sentence-CamemBERT-Large appears to achieve its best performance in the deeper layers, where its representations are more aligned with semantic similarity. In contrast, Qwen3-Embedding-8B exhibits a more stable behavior across layers, with consistently strong results that are less sensitive to the specific depth of representation. This suggests that Sentence-CamemBERT-Large benefits more from layer selection, while Qwen3-Embedding-8B provides more robust embeddings throughout the network.

## IV. LLM AS A JUDGE FOR ASR EVALUATION

Unlike the previous section, which relied on embeddings extracted from encoder and decoder LLMs to build semantic similarity metrics, we now investigate the native generative capabilities of decoder LLMs. Rather than comparing vector representations, the models are prompted directly to reason about transcription quality. We consider two complementary settings: selecting the best hypothesis between two candidates and assigning an interpretable quality label to a single hypothesis.

LLM inferences are performed using the SDialog toolkit [27].

## A. Pairwise Hypothesis Selection

The first experiment evaluates whether a generative LLM can act as a judge by selecting, given a reference transcription and two ASR hypotheses, the hypothesis that best matches human preference. Unlike embedding-based metrics, this formulation allows the model to jointly consider lexical accuracy, semantic fidelity, grammatical coherence and contextual information before making a decision.

The model is prompted in a one-shot setting consisting of a single annotated example followed by the pair to evaluate. It is asked to briefly justify its decision before ending its answer with either A or B, allowing automatic extraction of the final prediction while preserving the model’s reasoning process. The complete prompt is provided in Appendix A.

Table II reports the agreement between LLM decisions and human preferences. Overall, performance tends to improve with newer and larger models, although model size alone does not explain the results. For example, Qwen3-8B outperforms Gemma3-27B despite having substantially fewer parameters, while Qwen3.5-27B surpasses Qwen3-30B. The best results are obtained by GPT-4.1 (94%) and the open-weight Qwen3.5- 35B (92%) on the subset with total consensus, showing that open models can approach proprietary systems on this task.

Previously, as shown in Table I, we observed that Qwen3- 1.7B produces good representations for a semantic metric such as SemDist, while Qwen3.5-35B does not appear to achieve comparable performance in this setting. Interestingly, we observe the opposite trend in the pairwise hypothesis selection task, where Qwen3.5-35B significantly outperforms Qwen3-1.7B. This contrast suggests that the ability to produce useful continuous embeddings is not necessarily aligned with the ability to perform discrete comparative reasoning. One possible explanation is that intrinsic evaluation of a single hypothesis relies on different capabilities than step-by-step comparative reasoning between two hypotheses.

For this specific task, these results also suggest that generative LLM outperform traditional lexical metrics such as WER and CER, as well as the best embedding-based semantic metrics evaluated in the previous section. Unlike these metrics, which compare hypotheses through fixed similarity functions, generative LLMs directly reason over complete transcriptions before making a decision. This suggests that they capture aspects of transcription quality that are not necessarily reflected in semantic representation, such as grammatical coherence, semantic consistency or tolerance to minor disfluencies.

Furthermore, its high agreement with human judgments suggests that generative LLMs could reduce the amount of manual annotation required when constructing perceptual evaluation datasets, thereby lowering annotation costs.

<table><tr><td>LLM</td><td>= 100 %</td><td>≥ 70 %</td><td>Full</td></tr><tr><td>GPT-4.1</td><td>94</td><td>85</td><td>79</td></tr><tr><td>Gemma3-27B</td><td>72</td><td>63</td><td>61</td></tr><tr><td>Gemma4-31B</td><td>87</td><td>78</td><td>73</td></tr><tr><td>Qwen3.5-35B</td><td>92</td><td>83</td><td>78</td></tr><tr><td>Qwen3.5-27B</td><td>91</td><td>83</td><td>77</td></tr><tr><td>Qwen3-30B</td><td>84</td><td>75</td><td>71</td></tr><tr><td>Qwen3-8B</td><td>80</td><td>74</td><td>72</td></tr><tr><td>Qwen3-1.7B</td><td>59</td><td>58</td><td>56</td></tr><tr><td>Qwen3-0.6B</td><td>50</td><td>47</td><td>47</td></tr></table>

TABLE II: Agreement of human judgments (HATS) with LLM choices (selecting the best hypothesis), with respect to annotator agreement level.

## B. Direct Classification of Hypotheses

In this section, we evaluate the capacity of language models to classify a hypothesis given a reference. Unlike the previous experiment, each (reference, hypothesis) pair is evaluated independently. The objective is to determine whether an LLM can provide an absolute and interpretable assessment of transcription quality without requiring another hypothesis for comparison.

Given a reference and a hypothesis, the model must assign one of the following labels (see prompt in Appendix B):

• identical: the hypothesis is identical to the reference (or differ only in case or hyphenation);

• useful: meaning is preserved despite minor errors (normalization, punctuation, spelling, capitals, abbreviations, slight syntactic variations without loss of comprehension);

• bad: meaning is partially altered (significant errors on keywords, important substitutions or omissions, but some content remains understandable);

• incomprehensible: meaning is completely lost (the sentence cannot be understood or correctly interpreted relative to the reference).

This formulation also enables a new way of reporting ASR performance. Instead of relying solely on a scalar metric such as WER or BERTScore, an LLM can assign a qualitative label to every transcription, yielding a distribution of qualitative outputs. Such a distribution is considerably more interpretable, as it characterizes the nature and severity of transcription errors rather than reducing system performance to a single score.

Since the HATS dataset does not contain such qualitative labels, direct evaluation against human annotations is not possible. Instead, we compare the predicted categories with SemDist scores computed using one the best-performing embedding model identified previously, Sentence-CamemBERTlarge. This provides an independent semantic similarity signal that is known to correlate well with human perception.

Table III reports both Spearman and Pearson correlations between the predicted categories and SemDist. Negative correlations are expected because lower SemDist values correspond to better transcriptions, whereas the predicted categories are ordered from identical (3) to incomprehensible (0). Larger LLMs generally exhibit stronger correlations, with GPT-4.1 achieving the best overall results.

<table><tr><td>LLM</td><td>Spearman</td><td>Pearson</td></tr><tr><td>GPT-4.1</td><td>-0.66</td><td>-0.63</td></tr><tr><td>Gemma4-31B</td><td>-0.63</td><td>-0.63</td></tr><tr><td>Qwen3-30B</td><td>-0.63</td><td>-0.66</td></tr><tr><td>Qwen3.5-27B</td><td>-0.60</td><td>-0.61</td></tr><tr><td>Qwen3.5-35B</td><td>-0.58</td><td>-0.55</td></tr><tr><td>Qwen3-1.7B</td><td>-0.46</td><td>-0.50</td></tr><tr><td>Qwen3-0.6B</td><td>-0.18</td><td>-0.22</td></tr></table>

TABLE III: Correlation between generated categories and SemDist with embeddings from the last layer of Sentence-CamemBERT-Large using mean pooling.

![](images/49f40e7e89644b4c776fa4a1b8d77a45181f30405590f706ec112a302874c955.jpg)  
Fig. 3: Distribution of SemDist scores across LLM-predicted quality classes.

Figure 3 further illustrates the distribution of SemDist values for each predicted category. The four classes follow the expected ordering, with identical hypotheses associated with the highest semantic similarity and incomprehensible hypotheses with the lowest. However, substantial overlap remains between adjacent categories, particularly between useful and bad, explaining the moderate correlation coefficients observed in Table III.

Although these results are not yet strong enough to replace continuous semantic metrics, they demonstrate that generative LLMs can provide meaningful qualitative descriptions of ASR errors. Such descriptions offer an important interpretability advantage over scalar metrics such as WER or SemDist and constitute a promising direction for future ASR evaluation. Improving the consistency of these qualitative judgments, potentially through better prompting, dedicated fine-tuning or hybrid evaluation strategies, remains an interesting avenue for future work.

## V. CONCLUSION

This paper systematically evaluates the use of large language models for ASR evaluation through three complementary perspectives.

First, we study embedding-based semantic metrics derived from both encoder and decoder LLMs. We show that

BERTScore and SemDist can achieve strong agreement with human judgments when appropriate combinations of model, layer, and pooling strategy are selected. Performance does not depend solely on model scale and no single layer consistently dominates across architectures. While decoder models have slightly superior performance, encoder-based LLMs remain highly competitive, despite typically having lower computational cost.

Second, we investigate generative LLMs as direct evaluators through a pairwise hypothesis selection task. We find that state-of-the-art models achieve very high agreement with human preferences, outperforming both classical metrics such as WER and CER and embedding-based semantic metrics. These results indicate that generative models can serve as strong relative judges of ASR system outputs, capturing aspects of transcription quality that go beyond lexical similarity.

Third, we explore the use of LLMs for qualitative classification of transcription errors. While correlations with continuous semantic scores remain moderate, the results show consistent ordering of error severity and suggest that LLMs are promising solutions to provide interpretable structured feedback on ASR outputs. In particular, they enable the construction of distributions over qualitative error categories, offering a more detailed view of system behavior than single-score metrics.

Overall, our results highlight that LLMs provide a flexible framework for ASR evaluation. Encoder-based representations yield strong and efficient semantic metrics, while generative models enable both accurate comparative judgments and interpretable qualitative analysis. Taken together, these approaches open the way toward more human-aligned and interpretable evaluation methodologies for speech recognition systems.

## APPENDIX

All experiments rely on fixed instruction-style prompts. This appendix reports the exact prompts used in the experiments described in Section IV-A and Section IV-B. The prompts are provided exactly as used during inference (translated from French for readability), without any modification or prompt engineering beyond the in-context example.

A. Prompt for Pairwise Hypothesis Selection

The LLM input follows the template:

<s>USER: A reference is an exact transcription of audio.   
Two incorrect hypotheses are proposed.   
Explain your reasoning and end your sentence by writing ’A’,   
’B’, ’a’, or ’b’ if undecided.   
Reference: it’s to him self   
Hypothesis A: are to him self   
Hypothesis B: it’s uh to him-self   
ASSISTANT: Although hypothesis B contains a disfluency   
(’uh’), it corresponds much better to the reference in terms   
of words and meaning. The disfluency can be tolerated if it   
is part of the original, while hypothesis A’s grammatical   
error is more problematic. Therefore, the most acceptable   
transcription is hypothesis B.   
USER: Reference: {reference}   
Hypothesis A: {hypothesis A}   
Hypothesis B: {hypothesis B}   
ASSISTANT:

where {reference}, {hypothesis A}, and   
{hypothesis B} correspond to the input reference   
and ASR hypotheses.

## B. Prompt for Direct Hypothesis Classification

The LLM input follows the template:

<s>USER: You are an expert evaluator of Automatic Speech   
Recognition (ASR). Given a reference transcription and an   
ASR hypothesis, classify the hypothesis into exactly one of   
the following categories:   
IDENTICAL: the hypothesis is identical to the reference (or   
differs only in case or hyphenation).   
USEFUL: the meaning is preserved despite minor errors such   
as normalization, punctuation, spelling, capitalization,   
abbreviations, or slight syntactic variations without loss   
of comprehension.   
BAD: the meaning is partially altered due to significant   
substitutions, omissions, or errors affecting important   
words, but the sentence remains partially understandable.   
INCOMPREHENSIBLE: the meaning is completely lost and the   
hypothesis cannot be correctly understood with respect to   
the reference.   
Respond only with a valid JSON object following this schema:   
{"category": "<IDENTICAL|USEFUL|BAD|INCOMPREHENSIBLE>"}   
Reference: it’s to him self   
Hypothesis: it’s uh to him-self   
ASSISTANT: {"category": "USEFUL"}   
USER: Reference: {reference}   
Hypothesis: {hypothesis}   
ASSISTANT:

where {reference} and {hypothesis} denote the input transcription pair used for evaluation.

## REFERENCES

[1] S. Kim, A. Arora, D. Le, C.-F. Yeh, C. Fuegen, O. Kalinli, and M. L. Seltzer, “Semantic distance: A new metric for asr performance analysis towards spoken language understanding,” in Proc. Interspeech 2021, 2021, pp. 1977–1981.

[2] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “Bertscore: Evaluating text generation with bert,” in International Conference on Learning Representations, 2019.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[4] T. Baneras-Roux, M. Rouvier, J. Wottawa, and R. Dufour, “Qualitative˜ evaluation of language model rescoring in automatic speech recognition,” in Interspeech, 2022.

[5] N.-T. Le, C. Servan, B. Lecouteux, and L. Besacier, “Better evaluation of asr in speech translation context using word embeddings,” in Interspeech 2016, 2016.

[6] L. Gordeeva, V. Ershov, O. Gulyaev, and I. Kuralenok, “Meaning error rate: Asr domain-specific metric framework,” in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, 2021, pp. 458–466.

[7] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al., “Improving language understanding by generative pre-training,” OpenAI Blog, 2018.

[8] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar\` et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

[9] G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Riviere, M. S. Kale, J. Love\` et al., “Gemma: Open models based on gemini research and technology,” arXiv preprint arXiv:2403.08295, 2024.

[10] K. Ahuja, H. Diddee, R. Hada, M. Ochieng, K. Ramesh, P. Jain, A. Nambi, T. Ganu, S. Segal, M. Ahmed et al., “Mega: Multilingual evaluation of generative ai,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 4232– 4267.

[11] Y. Labrak, A. Bazoge, E. Morin, P.-A. Gourraud, M. Rouvier, and R. Dufour, “Biomistral: A collection of open-source pretrained large language models for medical domains,” in Findings of the association for computational linguistics: acl 2024, 2024, pp. 5848–5864.

[12] H. Naveed, A. U. Khan, S. Qiu, M. Saqib, S. Anwar, M. Usman, N. Akhtar, N. Barnes, and A. Mian, “A comprehensive overview of large language models,” ACM Transactions on Intelligent Systems and Technology, vol. 16, no. 5, pp. 1–72, 2025.

[13] T. Baneras-Roux, J. Wottawa, M. Rouvier, T. Merlin, and R. Dufour,˜ “Hats: An open data set integrating human perception applied to the evaluation of automatic speech recognition metrics,” in International Conference on Text, Speech, and Dialogue, 2023, pp. 164–175.

[14] Y.-Y. Wang, A. Acero, and C. Chelba, “Is word error rate a good indicator for spoken language understanding accuracy,” in 2003 IEEE workshop on automatic speech recognition and understanding (IEEE Cat. No. 03EX721). IEEE, 2003, pp. 577–582.

[15] A. C. Morris, V. Maier, and P. D. Green, “From wer and ril to mer and wil: improved evaluation measures for connected speech recognition.” in Interspeech, no. 4-8, 2004, p. 2004.

[16] X. He, L. Deng, and A. Acero, “Why word error rate is not a good metric for speech recognizer training for the speech translation task?” in 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2011, pp. 5632–5635.

[17] B. Favre, K. Cheung, S. Kazemian, A. Lee, Y. Liu, C. Munteanu, A. Nenkova, D. Ochei, G. Penn, S. Tratz et al., “Automatic human utility evaluation of asr systems: Does wer really predict performance?” in Proc. Interspeech 2013, 2013, pp. 3463–3467.

[18] S. Kim, D. Le, W. Zheng, T. Singh, A. Arora, X. Zhai, C. Fuegen, O. Kalinli, and M. Seltzer, “Evaluating user perception of speech recognition system quality with semantic distance metric,” in Proc. Interspeech 2022, 2022, pp. 3978–3982.

[19] D. Thennal, J. James, D. P. Gopinath et al., “Advocating character error rate for multilingual asr evaluation,” in Findings of the Association for Computational Linguistics: NAACL 2025, 2025, pp. 4926–4935.

[20] T. Baneras-Roux, M. Rouvier, J. Wottawa, and R. Dufour, “A compre-˜ hensive analysis of tokenization and self-supervised learning in end-toend automatic speech recognition applied on french language,” in 2024 32nd European Signal Processing Conference (EUSIPCO). IEEE, 2024, pp. 141–145.

[21] S. Kafle and M. Huenerfauth, “Evaluating the usability of automatically generated captions for people who are deaf or hard of hearing,” in Proceedings of the 19th International ACM SIGACCESS Conference on Computers and Accessibility, 2017, pp. 165–174.

[22] S. Nam and D. Fels, “Simulation of subjective closed captioning quality assessment using prediction models,” International Journal of Semantic Computing, vol. 13, no. 01, pp. 45–65, 2019.

[23] S. Roy, “Semantic-wer: A unified metric for the evaluation of asr transcript for end usability,” arXiv preprint arXiv:2106.02016, 2021.

[24] T. Baneras-Roux, M. Rouvier, J. Wottawa, and R. Dufour, “A paradigm˜ for interpreting metrics and measuring error severity in automatic speech recognition,” in International Conference on Text, Speech, and Dialogue. Springer, 2024, pp. 174–183.

[25] N. Itoh, G. Kurata, R. Tachibana, and M. Nishimura, “A metric for evaluating speech recognizer output based on human-perception model,” in Proc. Interspeech 2015, 2015, pp. 1285–1288.

[26] Z. Sasindran, H. Yelchuri, and T. Prabhakar, “Semascore: A new evaluation metric for automatic speech recognition tasks,” in Proc. Interspeech 2024, 2024, pp. 4558–4562.

[27] S. Burdisso, S. Baroudi, Y. Labrak, D. Grunert, P. Cyrta, Y. Chen,¨ S. Madikeri, E. Villatoro-Tello, R. Marxer, and P. Motlicek, “Sdialog: A python toolkit for end-to-end agent building, user simulation, dialog generation, and evaluation,” in Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 3: System Demonstrations), 2026, pp. 320–340.