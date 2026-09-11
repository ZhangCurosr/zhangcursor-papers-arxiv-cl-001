# ON THE IMPACT OF ANONYMIZATION ON THE PERFORMANCE OF LARGE LANGUAGE MODELS

Tobias Deußer\*<sup>†,1,2,3</sup>, Max Hahnbück <sup>†,1,2</sup>, Lorenz Sparrenberg <sup>1,3</sup>, Tobias Uelwer <sup>4</sup>, Christian Bauckhage <sup>1,2,3</sup>, and Rafet Sifa <sup>1,2,3</sup>

<sup>†</sup>These authors contributed equally to this work.

<sup>1</sup>University of Bonn, Bonn, Germany

<sup>2</sup>Fraunhofer IAIS, Sankt Augustin, Germany

<sup>3</sup>Lamarr Institute for Machine Learning and Artificial Intelligence, Bonn, Germany <sup>4</sup>Microsoft Germany GmbH, Cologne, Germany

## ABSTRACT

As large language models are increasingly deployed in sensitive domains, anonymizing input data to protect personally identifiable information has become a critical practice. However, the impact of this anonymization on model utility is not well understood. This paper presents a systematic empirical study of the trade-off between privacy and performance. We evaluate five prominent language models across eleven diverse benchmarks, comparing their performance on original versus pseudonymized inputs. Our results reveal that while anonymization generally degrades performance, the effect is highly nuanced. We find that more capable models, such as Qwen2.5-72B and GPT-4o mini, suffer the largest performance drops, suggesting a stronger reliance on specific entity information. The impact is also task-dependent: performance on TruthfulQA improves with anonymization, while retrieval focused tasks like RGB experience a catastrophic decline. Further experiments show that reversible anonymization techniques that preserve entity uniqueness significantly outperform irreversible ones like redaction, and that explicitly prompting models about anonymization offers no discernible benefit. We conclude that anonymization is not a one-size-fits-all solution and must be co-designed with the model and task in mind to balance privacy and utility effectively. Our findings provide a crucial baseline for developing more robust, privacy-aware AI systems.

Keywords anonymization · large language models · natural language processing · machine learning

## 1 Introduction

Large Language Models (LLMs) have become increasingly central to modern artificial intelligence applications, from analyzing medical reports [1], writing legal documents [2], finding contradictions in financial documents [3], connecting patients with rare diseases with each other [4], to enabling automated auditing for financial reports [5]. Yet, as such LLM approaches are deployed in real-world settings, privacy concerns are becoming more and more prevalent [6, 7]. One common mitigation strategy is anonymization, i.e., removing or masking personally identifiable information (PII) before feeding inputs into a model [8, 9, 10].

But what does this actually cost us in terms of performance? In this paper, we take a systematic look at the tradeoff between privacy and capability. We evaluate a set of five models (see Table 3) spanning a wide range of sizes and capabilities across eleven diverse benchmarks (see Table 1). These benchmarks cover a spectrum of natural language processing (NLP) tasks, including general scientific reasoning (ARC, 11), medical question answering (MedQA, 12), emotional intelligence (EQ-Bench, 13), and translation (WMT 2014, 14), allowing us to assess the trade-off across different cognitive and linguistic domains.

We observe that, while anonymization generally degrades performance as expected, the magnitude of this effect is highly dependent on the model’s capability, the nature of the task, and the specific anonymization technique employed. Our key findings are threefold:

1. Model capability is inversely correlated with robustness to anonymization. More powerful models like Qwen2.5-72B [15] and GPT-4o mini [16] experience the largest performance drops, suggesting they rely more heavily on specific entity information to achieve their state-of-the-art performance.

2. Task sensitivity varies dramatically. Performance on retrieval-intensive benchmarks like RGB collapses catastrophically post-anonymization. Conversely, on TruthfulQA, a benchmark designed to test factual accuracy, performance consistently improves, suggesting that removing named entities can mitigate a model’s tendency to hallucinate incorrect associations.

3. Technique and prompting matter, but not always as expected. Reversible anonymization techniques like pseudonymization, which preserve entity uniqueness, significantly outperform irreversible methods like redaction. However, explicitly prompting a model that its input has been anonymized yields no discernible performance benefit.

Our results challenge the notion of anonymization as a one-size-fits-all solution, demonstrating that it must be codesigned with the model and task in mind. These findings provide actionable insights for practitioners and a foundation for the discussion on and development of more robust, privacy-aware, and ultimately more trustworthy NLP systems.

## 2 Related Work

The need to anonymize text data to protect personally identifiable information (PII) is a well-established practice driven by regulatory frameworks and the goal of developing trustworthy AI systems [8, 9]. However, this introduces a fundamental tension between preserving privacy and maintaining data utility for downstream applications [17]. Our work contributes to the growing body of research addressing this privacy-utility trade-off, with a focus on the capabilities and shortcomings of modern large-scale language models (LLMs).

Early anonymization systems predominantly combined named entity recognition (NER), rule-based filters, and external gazetteers to mask explicit identifiers [18, 19]. The advent of LLMs has initiated a paradigm shift in this domain. LLMs are now employed as potent, context-aware anonymization tools capable of zero-shot PII removal and replacement [20, 21]. This development is complemented by methodologies that distill the knowledge from large, proprietary models into smaller, open-source versions for practical deployment [10]. Conversely, the inferential capabilities of LLMs also position them as formidable adversaries in re-identifying individuals from inadequately anonymized text, thereby elevating the requirements for robust privacy measures [22]. This dual capacity of LLMs demands a more profound comprehension of their interplay with anonymized data.

Empirical evaluations of anonymization effects have mostly been domain-specific. In medicine, Ben Cheikh Larbi et al. [23] found variable performance losses across five clinical NLP tasks and persistent re-identification risk. In contrast, Vakili et al. [24] reported that pseudonymization before training Swedish Clinical BERT resulted in nearidentical downstream performance. Financial-domain work combining Differential Privacy with Federated Learning explicitly highlights the privacy-utility trade-offs in transformer-based text classification [25]. Meanwhile, analyses of pseudonymization strategies reveal that seemingly minor design differences can have a significant impact on classification and summarization outcomes [26].

The rapid evolution of privacy-preserving techniques has created a pressing need for robust evaluation frameworks. Shahriar and Dara [27] introduced Priv-IQ, a comprehensive multimodal benchmark measuring LLM privacy intelligence across eight competencies including visual privacy, multilingual capabilities, and knowledge of privacy law. Sun et al. [28] presented a systematic evaluation framework for privacy-preserving algorithms across different LLM architectures (Mistral-7B, Llama2-7b, Falcon-7b), examining three scenarios: protecting training data only, user queries only, or both.

Domain-specific evaluation tools have also emerged. Zhang et al. [29] introduced SafetyBench with over 11,000 questions across seven safety categories including privacy concerns, while Andriushchenko et al. [30] presented AgentHarm for evaluating harmful LLM agent behaviors including privacy violations. These specialized benchmarks complement general privacy evaluations by capturing nuanced risks in specific deployment contexts.

For a more complete overview on the topic of textual anonymization, we refer the interested reader to our survey paper on various anonymization techniques and recent advances [31].

While these studies offer vital insights, they are often confined to specific domains (e.g., clinical or financial texts), employ a restricted set of NLP tasks (most commonly classification), or are predicated on earlier model architectures such as BERT. Our research broadens this line of inquiry by conducting a large-scale, systematic empirical evaluation with a more extensive scope. We evaluate a diverse set of five modern and highly capable LLMs, with parameter counts ranging from 7 to 72 billion, across eleven distinct benchmarks that encompass a wide array of reasoning and linguistic capabilities. This allows us to transcend domain-specific findings and identify more generalizable patterns concerning the privacy-utility trade-off. We aim to establish a cross-domain baseline for understanding how input anonymization affects the performance of contemporary LLMs.

## 3 Methodology

This section details the methodology employed to investigate the impact of input anonymization on the performance of LLMs. We describe our experimental pipeline, the selection criteria for benchmarks and models, and the specific experimental setup used to compare performance on original versus anonymized inputs.

## 3.1 Pipeline

To systematically evaluate the effect of anonymization, we developed a modular pipeline for our experiments. This pipeline facilitates the processing of diverse benchmarks with various LLMs, both with and without anonymization. The core stages are:

1. Data preparation: Benchmarks are downloaded, preprocessed, and sampled as required. Each benchmark is encapsulated, storing prompt templates, questions, and reference answers.

2. Anonymization: Questions (and associated prompt components, if applicable) are processed by an anonymizer, which replaces personally identifiable information (PII) with placeholders. A mapping between original and anonymized entities is retained.

3. Model inference: Selected LLMs generate responses to both the original and anonymized versions of the benchmark questions.

4. De-anonymization (for evaluation): For anonymized outputs, placeholders in the LLM-generated responses are mapped back to their original PII using the stored mappings. This step is crucial for ensuring fair comparison with reference answers.

5. Evaluation: LLM responses (with original PII restored for anonymized versions) are assessed against reference answers using the predefined evaluation metrics specific to each benchmark.

6. Results aggregation: Performance scores are collated, and metrics such as the relative performance change between anonymized and non-anonymized runs are computed.

Figure 1 provides a conceptual overview of this pipeline. The design emphasises extensibility, allowing for the straightforward integration of new benchmarks, anonymization strategies, and LLMs through abstract base classes.

## 3.2 Dataset and Benchmark Selection

To evaluate the impact of anonymization across diverse linguistic and cognitive tasks, we selected eleven established benchmarks. Our selection was guided by several criteria:

1. Popularity and recognition: Widespread use within the research community.

2. Openness: Publicly available test sets and evaluation methodologies to ensure transparency and reproducibility.

3. Language coverage: Predominantly English-language benchmarks, with the inclusion of one German-English translation task (WMT 2014) to assess cross-lingual effects.

4. Task Diversity: A broad range of domains (e.g., science, medicine, commonsense reasoning) and prompt styles (e.g., zero-shot, few-shot, chain-of-thought) to avoid a narrow focus.

Domains such as coding and mathematics were excluded, as they typically do not involve PII, meaning anonymization would have minimal impact. We identified candidate benchmarks by consulting resources such as the Open LLM Leaderboard [32] and the Stanford HELM project [33]. The selected benchmarks are detailed in Table 1.

Given computational constraints associated with evaluating multiple LLMs across numerous benchmarks, we standard ised the number of samples. For most benchmarks, a maximum of 500 samples were randomly selected from their respective test sets. For benchmarks comprising multiple tasks (specifically, BIG-Bench Hard and MMLU-PRO), this cap was increased to 1,000 samples to ensure adequate representation per subtask. This resulted in a total of 6,210 unique samples being evaluated. An overview of original and adjusted sample sizes can be found in Table 2.

![](images/8d58149518181106bd81b49eccbc054166fb6d819b03f6fb00e18b9494b7d843.jpg)  
Figure 1: Architecture overview of our encoder-only contradiction detection approach.

Table 1: Overview of selected benchmarks: Domains, prompt type, and references. CoT denotes Chain-of-Thought; RAG denotes Retrieval-Augmented Generation.
<table><tr><td>Benchmark</td><td>Domain</td><td>Prompt Type</td><td>Reference</td></tr><tr><td>ARC</td><td>Grade-school science reasoning</td><td>Zero-Shot</td><td>[11]</td></tr><tr><td>BIG-Bench Hard</td><td>Challenging reasoning tasks</td><td>CoT, Few-Shot</td><td>[34]</td></tr><tr><td>EQ-Bench</td><td>Emotional intelligence</td><td>Zero-Shot, Self-Critique</td><td>[13]</td></tr><tr><td>HellaSwag</td><td>Commonsense reasoning</td><td>Zero-Shot</td><td>[35]</td></tr><tr><td>IFEval</td><td>Complex instruction following</td><td>Zero-Shot</td><td>[36]</td></tr><tr><td>MedQA</td><td>Medical knowledge &amp; reasoning</td><td>Zero-Shot</td><td>[12]</td></tr><tr><td>MMLU-PRO</td><td>Multitask academic &amp; professional knowledge</td><td>CoT, Few-Shot</td><td>[37]</td></tr><tr><td>MUSR</td><td>Multi-step reasoning</td><td>CoT, Zero-Shot</td><td>[38]</td></tr><tr><td>RGB</td><td>Retrieval-Augmented Generation</td><td>Zero-Shot</td><td>[39]</td></tr><tr><td>TruthfulQA</td><td>Truthfulness &amp; debunking miscon- ceptions</td><td>Zero-Shot</td><td>[40]</td></tr><tr><td>WMT 2014 (En-De)</td><td>Machine translation (English to Ger- Few-Shot man)</td><td></td><td>[14]</td></tr></table>

Table 2: Benchmark summary: Number of tasks, original sample counts, adjusted sample counts for this study, and average input length (average number of tokens) of adjusted samples.
<table><tr><td>Benchmark</td><td>Subtasks</td><td>Sample Counts</td><td>Adj. Sample Counts</td><td>Avg. Length</td></tr><tr><td>ARC</td><td>2</td><td>3548</td><td>500</td><td>95.79</td></tr><tr><td>BIG-Bench Hard</td><td>27</td><td>6421</td><td>999</td><td>812.11</td></tr><tr><td>EQ-Bench</td><td>2</td><td>342</td><td>342</td><td>404.76</td></tr><tr><td>HellaSwag</td><td>1</td><td>10042</td><td>500</td><td>133.98</td></tr><tr><td>IFEval</td><td>1</td><td>541</td><td>500</td><td>44.02</td></tr><tr><td>MedQA</td><td>1</td><td>1273</td><td>500</td><td>170.37</td></tr><tr><td>MMLU-PRO</td><td>13</td><td>11032</td><td>994</td><td>382.70</td></tr><tr><td>MUSR</td><td>3</td><td>375</td><td>375</td><td>1295.43</td></tr><tr><td>RGB</td><td>3</td><td>500</td><td>500</td><td>921.27</td></tr><tr><td>TruthfulQA</td><td>1</td><td>790</td><td>500</td><td>144.87</td></tr><tr><td>WMT 2014 (En-De)</td><td>1</td><td>3003</td><td>500</td><td>44.74</td></tr><tr><td>Total</td><td>55</td><td>38967</td><td>6210</td><td>417.62</td></tr></table>

Table 3: Overview of large language models (LLMs) evaluated in this study. Shortened names are used throughout the paper for brevity.
<table><tr><td>Short Name</td><td>Full Model Name</td><td>Developer</td><td>Reference</td><td>Parameter Count</td></tr><tr><td>GPT-4o mini</td><td>GPT-4o mini</td><td>OpenAI</td><td>OpenAI [16]</td><td>UNKNOWN</td></tr><tr><td>Teuken-7B</td><td>Teuken-7B-instruct-research-v0.4</td><td>OpenGPT-X</td><td>Ali et al. [41]</td><td>7B</td></tr><tr><td>Llama-3.1-8B</td><td>Llama-3.1-8B-Instruct</td><td>Meta</td><td>Dubey et al. [42]</td><td>8B</td></tr><tr><td>Gemma-2-27B</td><td>Gemma-2-27b</td><td>Google</td><td>Gemma Team et al. [43]</td><td>27B</td></tr><tr><td>Qwen2.5-72B</td><td>Qwen2.5-72B-Instruct</td><td>Alibaba</td><td>Qwen et al. [15]</td><td>72B</td></tr></table>

When official prompts were available with the benchmarks, they were utilised. If multiple official prompts existed (e.g., zero-shot vs. few-shot), we selected the one anticipated to yield stronger baseline performance, typically favouring few-shot prompts. For benchmarks providing only datasets without predefined prompts, we first sought established prompt implementations (e.g., from the HELM repository). If no suitable reference was found, we created minimal zeroshot prompts. Importantly, for consistency, any prompt components, including few-shot examples, were anonymized alongside the primary question input to ensure comprehensive PII removal from the text presented to the LLM. This approach is crucial as our focus is on the relative impact of anonymization rather than achieving absolute state-of-the-art scores.

## 3.3 Model Selection

Our selection of LLMs, as seen in Table 3, aims for diversity in terms of model providers, parameter sizes, and architectures, whilst remaining within feasible computational limits for extensive experimentation. The primary objective is to analyse the relative change in performance due to anonymization, rather than to benchmark absolute model capabilities against the state-of-the-art.

## 3.4 Experimental Setup

The core experiment compares the performance of each selected LLM on each benchmark under two conditions: (i) original, non-anonymized inputs, and (ii) anonymized inputs.

## 3.4.1 Anonymization Configuration

For the primary experiments, we employ a pseudonymization strategy implemented by our anonymization tool [10]. This tool is a sophisticated LLM-based system that performs named entity recognition (NER) to identify PII and subsequently replaces them. Specifically, entities are substituted with a category label and a unique, consistent identifier; for example, every occurrence of "London" within a document is replaced by "<LOC>-1", while "Berlin" would become "<LOC>-2". This ensures that the uniqueness of entities is preserved, which is crucial for downstream tasks that rely on entity distinctions. The underlying model is a fine-tuned version of a smaller, efficient language model, optimized for a high recall score. In our evaluations, this tool achieves an F1-score of over 88% and a recall of over 91%.

## 3.4.2 Comparison of Anonymization Techniques

To investigate the impact of different anonymization approaches, we evaluate a subset of models and benchmarks using four distinct techniques in addition to pseudonymization:

• Generalization: Replacing specific PII with broader category labels (e.g., “London” becomes “[CITY]”).

• Masking: Parts of entities are obscured in a consistent pattern, preserving some surface information (e.g., “London” becomes “Lo\*\*\*\*”).

• Randomization: Entities are substituted with fixed but randomly generated strings that remain consistent across the dataset (e.g., “London” becomes “Xj92Lp”)

• Redaction: Replacing PII with a fixed, generic marker (e.g., “[REDACTED]” or ).

This comparative analysis is performed on GPT-4o mini using the MedQA, MUSR, and RGB benchmarks.

## 3.4.3 Impact of Explicit Anonymization Prompts

We also explore whether explicitly informing the LLM that an input has been anonymized affects performance. For this, two prefixes are prepended to anonymized prompts for GPT-4o mini on the MedQA, MUSR, and RGB benchmarks:

• Short Prefix: “This prompt is anonymized. Some context may be missing; respond as accurately as possible.”

• Long Prefix: “This prompt has been anonymized by removing all sensitive entities. Some contextual details may be missing, but please generate the most accurate and effective response possible based on the available information.”

Performance is compared against anonymized inputs without any prefix and non-anonymized inputs.

## 4 Experiments

This section presents the empirical evaluation of LLM performance under anonymization. We first detail the main experimental setup and results concerning our primary pseudonymization strategy. Subsequently, we investigate the effects of varying anonymization techniques and the utility of explicitly informing models about input anonymization.

All experiments are conducted on a system equipped with four NVIDIA Tesla V100 GPUs, each with 32GB of memory. To ensure reproducibility of results, a global random seed is used for data sampling (where applicable for benchmarks) and any stochastic processes within the models if not entirely avoidable. Crucially, the sampling temperature for LLM generation is set to 0 for all models and tasks, aiming for as deterministic as possible outputs.

## 4.1 Impact of Pseudonymization on LLM Performance

We evaluated the five LLMs detailed in Table 3 (GPT-4o mini, Teuken-7B, Llama-3.1-8B-Instruct, Gemma-2-27B, and Qwen2.5-72B) across the eleven benchmarks described in Table 1. For each benchmark, models processed both original and anonymized inputs. The anonymization employed the pseudonymization strategy outlined in Section 3.4.1 (from Deußer et al. [10]), replacing PII with category-unique placeholders (e.g., “London” to “<LOC>-1”). Performance was measured using the standard metrics for each benchmark.

Table 5 presents the performance scores for all models on all benchmarks, under both original and anonymized conditions. Several patterns emerge from this comprehensive evaluation. First, we observe that model performance is not uniformly affected across benchmarks. While some tasks show minimal degradation (e.g., ARC, HellaSwag, IFEval with drops typically under 5 percentage points), others experience substantial losses (e.g., RGB with drops of 30-60 percentage points for most models).

Second, the data reveals interesting model-specific behaviors. Notably, Teuken-7B achieves a score of 0.00 on both ARC and HellaSwag regardless of anonymization. This suggests fundamental limitations in this model’s ability to handle these specific task formats, independent of entity information. In contrast, Qwen2.5-72B and GPT-4o mini demonstrate strong baseline performance across most benchmarks, achieving scores above 0.60 on 8 out of 11 tasks in the non-anonymized condition. However, this superior baseline performance comes with a trade-off: these same models experience the largest absolute performance drops when anonymization is applied.

Table 4: Overview of the benchmarks: Domain and Promptstyle
<table><tr><td>Benchmark</td><td>Domain</td><td>Prompt Type</td><td>Reference</td></tr><tr><td>ARC</td><td>Grade-school science reasoning</td><td>Zero-Shot</td><td>[11]</td></tr><tr><td>BIG-Bench Hard</td><td>Challenging reasoning tasks</td><td>CoT, Few-Shot</td><td>[34]</td></tr><tr><td>EQ-Bench</td><td>Emotional intelligence</td><td>Zero-Shot, Self-Critique</td><td>[13]</td></tr><tr><td>HellaSwag</td><td>Commonsense reasoning</td><td>Zero-Shot</td><td>[35]</td></tr><tr><td>IFEval</td><td>Complex instruction following</td><td>Zero-Shot</td><td>[36]</td></tr><tr><td>MedQA</td><td>Medical knowledge &amp; reasoning</td><td>Zero-Shot</td><td>[12]</td></tr><tr><td>MMLU-PRO</td><td>Multitask academic &amp; professional knowledge</td><td>CoT, Few-Shot</td><td>[37]</td></tr><tr><td>MUSR</td><td>Multi-step reasoning</td><td>CoT, Zero-Shot</td><td>[38]</td></tr><tr><td>RGB</td><td>Retrieval-Augmented Generation</td><td>Zero-Shot</td><td>[39]</td></tr><tr><td>TruthfulQA</td><td>Truthfulness &amp; debunking miscon- ceptions</td><td>Zero-Shot</td><td>[40]</td></tr><tr><td>WMT 2014 (En-De)</td><td>Machine translation (English to Ger- man)</td><td>Few-Shot</td><td>[14]</td></tr></table>

Third, TruthfulQA stands out as the only benchmark where all five models either maintain or improve performance under anonymization. This counterintuitive result warrants closer examination, which we provide in our subsequent analysis. The consistency of this improvement across models of varying architectures and capabilities suggests a systematic phenomenon rather than a random fluctuation.

Figure 2 illustrates the performance difference (anonymized minus non-anonymized scores) as a heatmap, providing a visual representation of anonymization impact across the model-benchmark matrix. The color intensity corresponds to the magnitude of performance change, with darker blue indicating smaller drops (or improvements) and yellow/orange indicating larger performance losses.

Several insights emerge from this visualization. First, RGB exhibits the most severe and consistent performance degradation across all models, with drops ranging from -0.22 (Teuken-7B) to -0.47 (GPT-4o mini). This catastrophic decline suggests that RGB’s retrieval-augmented generation tasks are fundamentally dependent on specific entity information that pseudonymization clearly obscures. The task requires models to integrate information from multiple sources while filtering out noise, and replacing concrete entities with abstract placeholders appears to disrupt this integration process severely.

Interestingly, the positive values for TruthfulQA represent a striking anomaly. This consistent improvement across multiple models suggests that anonymization may actually help models avoid factual errors in this specific context. We hypothesize that TruthfulQA questions often contain named entities that trigger incorrect memorized associations. By replacing “Which country in Europe...” with “Which country in <LOC>-1...”, we may prevent models from confidently generating plausible-sounding but incorrect facts tied to specific names they encountered during training.

Figure 3 displays the average performance across all benchmarks for each model. While Qwen2.5-72B and GPT-4o mini achieve the highest average scores, they also exhibit the largest average performance degradation when inputs are anonymized. This suggests that more capable models might rely more heavily on specific entity information, making them more susceptible to the information loss caused by anonymization. For instance, Qwen2.5-72B’s average performance drops by 6.9 percentage points, while Teuken-7B, the lowest-performing model, sees a smaller average drop of 2.3 percentage points.

The impact of anonymization varies considerably across benchmarks. Notably, TruthfulQA is the only benchmark where performance consistently improves with anonymization for most models (e.g., an increase of 8 percentage points for Llama-3.1). This suggests that removing named entities might reduce the model’s tendency to hallucinate or retrieve incorrect factual associations for this specific task.

Conversely, the RGB benchmark, which tests retrieval-augmented generation capabilities like noise robustness and information integration, suffers the most significant performance degradation. For example, GPT-4o mini’s score on RGB drops by approximately 48 percentage points (from 0.80 to 0.32). This substantial decrease highlights RGB’s reliance on precise entity information for its complex reasoning and information synthesis requirements.

Table 6 provides statistics on the average number of entities (total and unique) and input lengths across benchmarks, which help contextualize the varying impact of anonymization. While benchmarks like RGB and MUSR exhibit high entity counts and long inputs, only RGB experiences a substantial performance drop after anonymization. MUSR, despite similarly high entity statistics, shows minimal performance degradation. Conversely, benchmarks with few entities and short inputs, such as HellaSwag and IFEval, display only small performance drops, aligning with expectations. However, benchmarks like MMLU-PRO and MedQA show noticeable degradation despite moderate entity counts, suggesting that entity quantity alone does not determine impact. To investigate this further, we conducted correlation analyses between performance drops and each of the reported metrics (total entities, unique entities, and input length), but found no statistically significant correlations. This indicates that anonymization effects are likely mediated by task-specific dependencies on entity semantics and context, rather than raw input characteristics.

Table 5: Performance overview of the 5 tested LLMs across all benchmarks (anonymized and non-anonymized). Higher scores indicate better performance. The best scores are bold.
<table><tr><td>Benchmark</td><td>Anonymized</td><td>GPT-4o mini</td><td>Teuken-7B</td><td>Llama-3.1-8B</td><td>Gemma-2-27B</td><td>Qwen2.5-72B</td></tr><tr><td rowspan="2">ARC</td><td>no</td><td>0.62</td><td>0.00</td><td>0.84</td><td>0.29</td><td>0.95</td></tr><tr><td>yes</td><td>0.64</td><td>0.00</td><td>0.79</td><td>0.30</td><td>0.91</td></tr><tr><td rowspan="2">BIG-Bench Hard</td><td>no</td><td>0.24</td><td>0.18</td><td>0.23</td><td>0.25</td><td>0.23</td></tr><tr><td>yes</td><td>0.21</td><td>0.18</td><td>0.20</td><td>0.22</td><td>0.19</td></tr><tr><td rowspan="2">EQ-Bench</td><td>no</td><td>0.74</td><td>0.30</td><td>0.66</td><td>0.76</td><td>0.72</td></tr><tr><td>yes</td><td>0.72</td><td>0.29</td><td>0.64</td><td>0.76</td><td>0.70</td></tr><tr><td rowspan="2">HellaSwag</td><td>no</td><td>0.77</td><td>0.01</td><td>0.50</td><td>0.00</td><td>0.82</td></tr><tr><td>yes</td><td>0.74</td><td>0.00</td><td>0.49</td><td>0.00</td><td>0.78</td></tr><tr><td rowspan="2">IFEval</td><td>no</td><td>0.81</td><td>0.28</td><td>0.75</td><td>0.62</td><td>0.83</td></tr><tr><td>yes</td><td>0.77</td><td>0.27</td><td>0.70</td><td>0.57</td><td>0.79</td></tr><tr><td rowspan="2">MedQA</td><td>no</td><td>0.88</td><td>0.37</td><td>0.91</td><td>0.94</td><td>0.87</td></tr><tr><td>yes</td><td>0.83</td><td>0.34</td><td>0.88</td><td>0.87</td><td>0.83</td></tr><tr><td rowspan="2">MMLU-PRO</td><td>no</td><td>0.63</td><td>0.21</td><td>0.47</td><td>0.55</td><td>0.71</td></tr><tr><td>yes</td><td>0.55</td><td>0.18</td><td>0.42</td><td>0.49</td><td>0.61</td></tr><tr><td rowspan="2">MUSR</td><td>no</td><td>0.70</td><td>0.52</td><td>0.63</td><td>0.62</td><td>0.75</td></tr><tr><td>yes</td><td>0.70</td><td>0.54</td><td>0.62</td><td>0.59</td><td>0.73</td></tr><tr><td rowspan="2">RGB</td><td>no</td><td>0.80</td><td>0.53</td><td>0.69</td><td>0.63</td><td>0.82</td></tr><tr><td>yes</td><td>0.32</td><td>0.31</td><td>0.34</td><td>0.39</td><td>0.38</td></tr><tr><td rowspan="2">TruthfulQA</td><td>no</td><td>0.61</td><td>0.45</td><td>0.70</td><td>0.67</td><td>0.69</td></tr><tr><td>yes</td><td>0.60</td><td>0.51</td><td>0.78</td><td>0.68</td><td>0.74</td></tr><tr><td rowspan="2">WMT 2014 (En-De)</td><td>no</td><td>0.30</td><td>0.26</td><td>0.24</td><td>0.17</td><td>0.30</td></tr><tr><td>yes</td><td>0.27</td><td>0.24</td><td>0.19</td><td>0.16</td><td>0.27</td></tr></table>

Table 6: Average entity statistics across all benchmarks. For each benchmark, we report: (1) the average number of total entities (including repetitions), (2) the average number of unique entities, and (3) the average length of each input (in tokens).
<table><tr><td>Benchmark</td><td>Avg. Total Entities</td><td>Avg. Unique Entities</td><td>Avg. Length</td></tr><tr><td>ARC</td><td>0.83</td><td>0.61</td><td>97.65</td></tr><tr><td>BIG-Bench Hard</td><td>52.43</td><td>10.15</td><td>840.92</td></tr><tr><td>EQ-Bench</td><td>17.30</td><td>2.75</td><td>405.27</td></tr><tr><td>HellaSwag</td><td>0.95</td><td>0.61</td><td>135.32</td></tr><tr><td>IFEval</td><td>0.67</td><td>0.61</td><td>44.94</td></tr><tr><td>MedQA</td><td>2.53</td><td>2.27</td><td>174.54</td></tr><tr><td>MMLU-PRO</td><td>7.43</td><td>4.80</td><td>395.25</td></tr><tr><td>MUSR</td><td>67.47</td><td>8.69</td><td>1317.20</td></tr><tr><td>RGB</td><td>79.49</td><td>35.26</td><td>1000.23</td></tr><tr><td>TruthfulQA</td><td>12.68</td><td>7.67</td><td>170.99</td></tr><tr><td>WMT 2014</td><td>4.20</td><td>2.10</td><td>55.83</td></tr></table>

## 4.2 Comparing Different Anonymization Techniques

We compare five anonymization techniques (redaction, generalization, pseudonymization, randomization, and masking), as described in Section 3.4.2. This experiment is conducted using GPT-4o mini on a subset of three benchmarks: MedQA, MUSR, and RGB, chosen for their diverse characteristics.

Figure 4 shows the performance of GPT-4o mini under different anonymization methods. Reversible techniques (pseudonymization, randomization, and masking), which allow for a potential one-to-one mapping back to original entities, generally preserve performance better than irreversible ones (redaction, generalization).

![](images/c20ad4980dde45e3a00e47fa10755567830f05eb8736d0b2764c61faa5dd2973.jpg)  
Figure 2: Difference in benchmark performance scores (anonymized minus non-anonymized) across multiple language models and tasks. Negative values indicate performance drops due to anonymization.

Generalization (e.g., “London” → “[CITY]”) and Redaction (e.g., “London” → “ ”) consistently resulted in the largest performance drops. For instance, on MuSR, Generalization led to a score of 0.58 and Redaction 0.52, compared to 0.70 for pseudonymization and 0.73 for the original input. This is likely because these methods remove too much information, making it difficult for the model to establish necessary entity distinctions and for evaluation scripts, which expect original entities after de-anonymization, to score correctly.

Masking (e.g., “London” → “Lo\*\*\*\*”) and Randomization (e.g., “London” → “Xj92Lp”) performed comparably to or slightly worse than Pseudonymization. Masking performed well on MedQA and MUSR but saw a significant drop on RGB, possibly due to the nature of information required by RGB that simple partial string matching cannot satisfy.

These results underscore that techniques preserving unique identifiers for distinct entities, even if abstract, are preferable for maintaining utility.

## 4.3 Impact of Explicit Anonymization Prompts

Here, we investigate whether explicitly informing the LLM about input anonymization has an impact on performance. Using GPT-4o mini once again on MedQA, MUSR, and RGB, we prepend two types of instructional prefixes (short and long, as detailed in Section 3.4.3) to anonymized inputs and compared results against anonymized inputs without a prefix and original inputs.

Figure 5 summarises the findings. Adding an explicit textual cue that the input has been anonymized did not yield any significant or consistent improvement in LLM performance across the tested benchmarks and prefixes. The performance scores were very similar regardless of whether a short prefix, a long prefix, or no prefix was used with the anonymized input. This suggests that current LLMs do not substantially alter their reasoning strategy based on such meta-information, and the performance impact is primarily driven by the modification of content itself.

![](images/591d34493281a03930127c32849bce3480a9b85d83739602a441b341c8562ea1.jpg)

Figure 3: Average performance of the five different models under anonymized and non-anonymized conditions. For all models tested, the non-anonymized version shows equal or slightly higher average performance.  
Table 7: Example 1 (from the ARC Benchmark)
<table><tr><td>Original Case</td><td>Anonymized Case</td><td></td></tr><tr><td>Question: A solar eclipse occurs when</td><td>Question: A solar eclipse occurs when</td><td></td></tr><tr><td>A. the Moon blocks the Earth from the Sun.</td><td></td><td>A. the Moon blocks the &lt;LOC&gt;-1 from the &lt;LOC&gt;-2. B. the first four planets in the solar system are aligned.</td></tr><tr><td>B. the first four planets in the solar system are aligned.</td><td></td><td></td></tr><tr><td>C. Earth blocks the Moon from the Sun.</td><td></td><td>C. &lt;LOC&gt;-1 blocks the Moon from the &lt;LOC&gt;-2.</td></tr><tr><td>D. Earth&#x27;s shadow falls on the Sun.</td><td></td><td>D. &lt;LOC&gt;-1&#x27;s shadow falls on the &lt;LOC&gt;-2.</td></tr><tr><td>Predicted Answer: A</td><td>Score: 1.0</td><td>Predicted Answer: D</td></tr><tr><td>Reference Answer: A</td><td></td><td>Score: 0.0</td></tr></table>

## 4.4 Benchmark Examples

In this subsection, we explore why we observe the performance differences shown in Table 5, Figure 2 to 4, and described in the previous subsections by investigating various examples from the test sets of our studied datasets. Note that some questions and answers may have been shortened for readability in Tables 7 to 11.

In the question about solar eclipses depicted in Table 7, our anonymization tool replaced “Earth” and “Sun” with placeholders like <LOC>-1 and <LOC>-2. Therefore, the model was unable to ground its reasoning and selected an incorrect answer. Both the correct and incorrect options appeared partially plausible when anonymized, demonstrating that removing key entities can make it impossible to identify the reference answer.

In Table 8, the language model was asked which European country has the most Nobel Laureates in science. The model initially answered “United Kingdom” correctly. After anonymization, “Europe” was replaced with <LOC>-4, leading the model to generate a non-existent placeholder-style answer, <COUNTRY NAME>. This highlights a failure mode: the model does not adapt its reasoning to the anonymization scheme and may fabricate answers outside the tagging system.

![](images/a43774bff0b47cf95c1df3c37337587b5bfc294b1900dececb6eb1e694384dcc.jpg)  
Figure 4: Performance of GPT-4o mini on MedQA, MUSR, and RGB, comparing redaction, generalization, pseudonymization, randomization, and masking.

Table 8: Example 2 (from the TruthfulQA Benchmark)
<table><tr><td>Original Case</td><td>Anonymized Case</td></tr><tr><td>Question: Which country in Europe has the most Nobel Laureates in science?</td><td>Question: Which country in &lt;LOC&gt;-4 has the most Nobel Laureates in science?</td></tr><tr><td>Predicted Answer: As of my last knowledge update in Oct 2023, the country in Europe with the most Nobel Laureates in science is the United Kingdom. Score: 1.0</td><td>Predicted Answer: The country in &lt;LOC&gt;-4 with the most Nobel Laureates in science is &lt;COUNTRY NAME&gt;. Score: 0.0</td></tr><tr><td colspan="2">Reference Answer: The country with the most Nobel Laureates in science is the United Kingdom.</td></tr></table>

![](images/04dce00ab7e8bd03909667fdae9afff2985db058883e5e2672c251cc7ff29271.jpg)  
Figure 5: Performance of GPT-4o mini on MedQA, MUSR, and RGB with original inputs, anonymized inputs (no prefix), and anonymized inputs with short or long explicit prefixes.

Carefully designed prompts might mitigate this, but such correction would probably amount to informed guessing rather than robust understanding.

In the question about the euro appreciating against the dollar, shown in Table 9, concrete monetary values were replaced with <MONETARYVALUE>. The model incorrectly concluded that the euro depreciated. Although the initial reasoning was correct, the absence of concrete numbers led to a reversed conclusion. This example demonstrates that anonymization can introduce ambiguity that causes reasoning errors in quantitative tasks.

In Table 10, a medical question about protein classes affected in a patient with infertility and Kartagener syndrome-like symptoms was given to the language model. Anonymization changed the available answer options by removing all but one of the options. The model initially chose the wrong answer but selected the correct answer after anonymization due to the altered answer space. This improvement is coincidental and reflects interaction effects rather than genuine reasoning improvement.

Finally, even if an answer is correct, evaluation scripts may expect a specific format. In such cases, anonymization can change the score between the original and anonymized scenario, even if no entities were modified, as illustrated in Table 11.

Taken together, these examples show that anonymization does not have a uniform effect on model performance. It can obscure critical information, induce confusion and fabricated outputs, cause reasoning errors in quantitative tasks, or randomly improve results depending on how entities and options are replaced. Therefore, systematic errors are difficult to identify, as the impact of anonymization is highly question-, and thus dataset-, dependent.

Table 9: Example 3 (from the MMLU-Pro Benchmark)
<table><tr><td>Original Case</td><td>Anonymized Case</td></tr><tr><td>Question: Suppose yesterday the euro was trading on the foreign exchange market at $1.36 U.S. and today it is trading at $1.40 U.S. Which of the following is true?</td><td>Question: Suppose yesterday the euro was trading on the foreign exchange market at &lt;MONETARYVALUE&gt;-1 U.S. and today it is trading at &lt;MONETARYVALUE&gt;-2 U.S. Which of the following is true?</td></tr><tr><td>Options: A. Interest rates in the United States have decreased. B. The U.S. dollar has appreciated. C. The U.S. dollar has appreciated and the euro has</td><td>Options: A. Interest rates in the &lt;LOC&gt;-1 have decreased.</td></tr><tr><td>depreciated. D. The exchange rate has remained steady. E. The U.S. dollar has stayed the same and the euro has</td><td>B. The U.S. dollar has appreciated. C. The U.S. dollar has appreciated and the euro has depreciated.</td></tr><tr><td>depreciated. F. The euro has appreciated. G. Interest rates in Europe have increased</td><td>D. The exchange rate has remained steady. E. The U.S. dollar has stayed the same and the euro has depreciated.</td></tr><tr><td>H. The U.S. dollar has depreciated. I. The euro has depreciated. J. Interest rates in the United States have increased.</td><td>F. The euro has appreciated. G. Interest rates in &lt;LOC&gt;-2 have increased. H. The U.S. dollar has depreciated.</td></tr><tr><td>Predicted Answer: To determine the correct answer, we analyze the change in the exchange rate of the euro against the</td><td>I. The euro has depreciated. J. Interest rates in the &lt;LOC&gt;-1 have increased.</td></tr><tr><td>U.S. dollar. Yesterday, the euro was trading at $1.36 and today</td><td>Predicted Answer: To analyze the change in the exchange rate, we need to look at the values given. Yesterday, the euro</td></tr><tr><td>it is trading at $1.40. This means that it now takes more U.S. dollars to purchase one euro, which indicates that the euro has become more expensive, or has appreciated in value relative to the U.S. dollar. As a result, we can also infer that the U.S.</td><td>was trading at $&lt;MONETARYVALUE&gt;-1 and today it is trading at $&lt;MONETARYVALUE&gt;-2 U.S. This implies that it now takes more U.S. dollars to buy one euro than it did yesterday. Thus, the value of the euro has decreased relative to</td></tr><tr><td>dollar has depreciated since it now has less purchasing power compared to the euro.</td><td>the U.S. dollar, meaning the euro has depreciated. Since the euro has depreciated, it follows that the U.S. dollar</td></tr><tr><td>Both F and H contain correct information based on our analysis, but the most straightforward answer related to the change in exchange rate is F. Thus, the answer is (F). Score: 1.0</td><td>has appreciated in value relative to the euro because you can now get more euros for each dollar. Both B (the U.S. dollar has appreciated) and C (the U.S. dollar has appreciated and the</td></tr></table>

## 5 Conclusion

In this work, we systematically investigated the trade-off between privacy and utility by evaluating the impact of input anonymization on LLM performance. Our findings reveal a complex and nuanced relationship, confirming that while anonymization generally degrades performance, its effects are far from uniform and depend heavily on the model’s capability, the task’s nature, and the anonymization technique employed.

A central observation is that more capable models, such as Qwen2.5-72B and GPT-4o mini, while achieving higher absolute scores, also suffer the most significant performance drops. This suggests a greater reliance on the specific entity information that anonymization removes, making them more vulnerable to this form of information loss. The impact is also highly task-dependent: performance on TruthfulQA unexpectedly improved for most models, likely due to a reduction in factual hallucination, whereas the retrieval-heavy RGB benchmark saw a catastrophic decline, highlighting its sensitivity to precise entity context.

Our comparative analysis of anonymization techniques demonstrates the clear superiority of reversible methods (e.g., pseudonymization, randomization) over irreversible ones (e.g., redaction, generalization), underscoring the importance of preserving entity uniqueness for model reasoning. Furthermore, we found that explicitly prompting models about the anonymized nature of the input provided no discernible benefit, indicating that current models do not adapt their strategy based on such meta-information alone.

Ultimately, this research challenges the notion of anonymization as a simple, one-size-fits-all solution for privacy in LLMs. The results suggest that achieving privacy cannot be treated as a mere preprocessing step without considering

## Original Case

Question: A 30-year-old male presents to his primary care physician complaining of infertility. He and his wife have been trying to get pregnant for the past two years. They have used fertility monitors and other aids without success. A hysterosalpingogram in his wife was normal. The patient has a history of cleft lip and recurrent upper respiratory infections as a child. He was briefly hospitalized for severe pneumonia when he was 9-years-old. His temperature is 98.6<sup>◦</sup>F (37<sup>◦</sup>C), blood pressure is 120/85 mmHg, pulse is 90/min, and respirations are 18/min. On examination, he is a healthy-appearing male in no acute distress with fully developed reproductive organs. Notably, cardiac auscultation is silent in the left 5th intercostal space at the midclavicular line. This patient most likely has a mutation in which of the following classes of proteins?

## Options:

A: Microtubule monomeric protein B: Transmembrane ion channel protein C: Retrograde cytoskeletal motor protein D: Anterograde cytoskeletal motor protein

Answer: This clinical scenario and the details provided suggest a likely underlying genetic condition, specifically related to the patient’s history of cleft lip and recurrent respiratory infections, along with the cardiac finding of a silent auscultation area.

Considering the context, the patient probably has a mutation affecting the function of cilia or flagella, which can lead to both respiratory issues (due to impaired mucociliary clearance) and infertility (due to issues with sperm motility in conditions like immotile cilia syndrome). Conditions that affect cilia function often involve defects in axonemal proteins.

Among the answer choices, the most relevant class of proteins related to ciliary function, and particularly sperm motility, would be:

## D: Anterograde cytoskeletal motor protein.

This option is appropriate because anterograde transport is essential for the proper function of cilia, and defective motor proteins (like dynein) can result in the ciliary dysfunction seen in conditions that cause infertility and respiratory issues. Score: 0.0

## Reference Answer: C

## Anonymized Case

Question: A 30-year-old male presents to his primary care physician complaining of infertility. He and Azalea Kahler have been trying to get pregnant for the past two years. They have used fertility monitors and other aids without success. A hysterosalpingogram in Azalea Kahler was normal. The patient has a history of cleft lip and recurrent upper respiratory infections as a child. He was briefly hospitalized for severe pneumonia when he was 9-years-old. His temperature is 98.6<sup>◦</sup>F (37<sup>◦</sup>C), blood pressure is 120/85 mmHg, pulse is 90/min, and respirations are 18/min. On examination, he is a healthy-appearing male in no acute distress with fully developed reproductive organs. Notably, cardiac auscultation is silent in the left 5th intercostal space at the midclavicular line. This patient most likely has a mutation in which of the following classes of proteins?

## Options:

A: Microtubule monomeric protein

B: <PROD>-1

C: <PROD>-2

D: <PROD>-3

Answer: [...] While the options "A", "B", "C", and "D" likely refer to specific genetic mutations, without the specific knowledge of what "<PROD>-1", "<PROD>-2", and "<PROD>-3" represent, it’s difficult to pinpoint the exact answer among the provided options. However, proteins involved in ciliary structure and function are most relevant regarding the clinical scenario presented. [...] Given this information, I lean towards the answer choice more closely aligning with cytoskeletal proteins or structural proteins necessary for development/function of reproductive systems or cilia.

Nonetheless, the answer cannot be definitively verified without additional context for "<PROD>-1", "<PROD>-2", and "<PROD>-3". Further details about these options would be needed for a precise answer.

For now, if I had to choose based on the general implications of the question regarding infertility and developmental anomalies, option A: Microtubule monomeric protein could be a strong contender. However, additional context is vital for accuracy. Score: 1.0

Table 11: Example 5 (from the BIG-Bench Hard Benchmark)
<table><tr><td>Original Case</td><td>Anonymized Case</td></tr><tr><td>Question: The rover Spirit is a robotic probe that NASA has placed on Mars. The gravitational attraction of Mars is approximately 62% less than that of Earth. Compared to its measurements on Earth, on Mars the probe has</td><td>Question: The rover &lt;PROD&gt;-1 is a robotic probe that TerraSource has placed on &lt;LOC&gt;-1. The gravitational attraction of &lt;LOC&gt;-1 is approximately 62% less than that of &lt;LOC&gt;-2. Compared to its measurements on &lt;LOC&gt;-2, on &lt;LOC&gt;-1 the probe has</td></tr><tr><td>Choices: - A: the same weight and the same mass.</td><td>Choices:</td></tr><tr><td>- B: a larger weight, but smaller mass.</td><td>- A: the same weight and the same mass.</td></tr><tr><td>- C: a smaller mass and larger weight.</td><td>- B: a larger weight, but smaller mass.</td></tr><tr><td>- D: a smaller weight and the same mass.</td><td>- C: a smaller mass and larger weight.</td></tr><tr><td></td><td>- D: a smaller weight and the same mass.</td></tr><tr><td>Predicted Answer: D: a smaller weight and the same mass. Score: 0.0</td><td>Predicted Answer: D</td></tr></table>

the downstream task. It calls for a shift towards developing privacy-aware systems where the choice of anonymization strategy is co-designed with the model and application in mind. Future work should explore more granular anonymiza tion controls, selectively targeting entity types based on task requirements. Developing models that are inherently more robust to the removal of PII, or can be fine-tuned to understand anonymized representations, presents another promising research direction. As LLMs continue to evolve, ongoing evaluation will be crucial to track whether this privacy-utility trade-off changes with new architectures and capabilities.

## Acknowledgments

This research has been partially funded by the Federal Ministry of Education and Research of Germany and the state of North-Rhine Westphalia as part of the Lamarr-Institute for Machine Learning and Artificial Intelligence. Google Gemini 2.5 Pro [44] and GPT-4o [45] were employed to assist in refining and improving the text throughout all sections of this paper. The authors retain full responsibility for the accuracy, integrity, and originality of the work.

## References

[1] Rushabh Doshi, Kanhai S Amin, Pavan Khosla, Simar S Bajaj, Sophie Chheang, and Howard P Forman. Quantitative evaluation of large language models to streamline radiology report impressions: a multimodal retrospective analysis. Radiology, 2024.

[2] Jiaxi Cui, Munan Ning, Zongjian Li, Bohua Chen, Yang Yan, Hao Li, Bin Ling, Yonghong Tian, and Li Yuan. Chatlaw: A multi-agent collaborative legal assistant with knowledge graph enhanced mixture-of-experts large language model, 2024. URL https://arxiv.org/abs/2306.16092.

[3] Tobias Deußer, David Leonhard, Lars Hillebrand, Armin Berger, Mohamed Khaled, Sarah Heiden, Tim Dilmaghani, Bernd Kliem, Rüdiger Loitz, Christian Bauckhage, et al. Uncovering inconsistencies and contradictions in financial reports using large language models. In Proc. BigData, 2023.

[4] Armin Berger, Ali Hamza Bashir, David Berghaus, Nazia Afsan, Lorenz Grigull, Lara Fendrich, Henriette Högl, Gundula Ernst, Ralf Schmidt, David Bascom, et al. Optimizing rare disease patient matching with large language models. In Proc. BigData, 2024.

[5] Lars Hillebrand, Armin Berger, Tobias Deußer, Tim Dilmaghani, Mohamed Khaled, Bernd Kliem, Rüdiger Loitz, Maren Pielka, David Leonhard, Christian Bauckhage, et al. Improving zero-shot text matching for financial auditing with large language models. In Proc. DocEng, 2023.

[6] Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, and Yue Zhang. A survey on large language model (llm) security and privacy: The good, the bad, and the ugly. High-Confidence Computing, 2024.

[7] Badhan Chandra Das, M Hadi Amini, and Yanzhao Wu. Security and privacy challenges of large language models: A survey. ACM Computing Surveys, 2025.

[8] David Biesner, Rajkumar Ramamurthy, Robin Stenzel, Max Lübbering, Lars Hillebrand, Anna Ladi, Maren Pielka, Rüdiger Loitz, Christian Bauckhage, and Rafet Sifa. Anonymization of german financial documents using neural

network-based language models with contextual word representations. International Journal ofData Science and Analytics, 2022.

[9] Ali Sepas, Ali Haider Bangash, Omar Alraoui, Khaled El Emam, and Alaa El-Hussuna. Algorithms to anonymize structured medical and healthcare data: A systematic review. Frontiers in Bioinformatics, 2022.

[10] Tobias Deußer, Max Hahnbück, Tobias Uelwer, Cong Zhao, Christian Bauckhage, and Rafet Sifa. Resourceefficient anonymization of textual data via knowledge distillation from large language models. In Proc. COLING, 2025.

[11] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https: //arxiv.org/abs/1803.05457.

[12] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 2021.

[13] Samuel J. Paech. Eq-bench: An emotional intelligence benchmark for large language models, 2024. URL https://arxiv.org/abs/2312.06281.

[14] Ondˇrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, et al. Findings of the 2014 workshop on statistical machine translation. In Proc. WMT, 2014.

[15] Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

[16] OpenAI. Gpt-4o mini: Advancing cost-efficient intelligence. OpenAI Blog, 2024. URL https://openai.com/ index/gpt-4o-mini-advancing-cost-efficient-intelligence/. Accessed: 27/05/2025.

[17] Pierre Lison, Ildikó Pilán, David Sánchez, Montserrat Batet, and Lilja Øvrelid. Anonymisation models for text data: State of the art, challenges and future directions. In Proc. ACL-IJCNLP, 2021.

[18] Arttu Oksanen, Eero Hyvönen, Minna Tamper, Jouni Tuominen, Henna Ylimaa, Katja Löytynoja, Matti Kokkonen, and Aki Hietanen. An anonymization tool for open data publication of legal documents. In International Workshop on Artificial Intelligence Technologiesfor Legal Documents, 2022.

[19] Bennett Kleinberg, Toby Davies, and Maximilian Mozes. Textwash – automated open-source text anonymisation, 2022. URL https://arxiv.org/abs/2208.13081.

[20] Robin Staab, Mark Vero, Mislav Balunovic, and Martin Vechev. Language models are advanced anonymizers. In Proc. ICLR, 2025.

[21] Zhengliang Liu, Yue Huang, Xiaowei Yu, Lu Zhang, Zihao Wu, Chao Cao, Haixing Dai, Lin Zhao, Yiwei Li, Peng Shu, Fang Zeng, Lichao Sun, Wei Liu, Dinggang Shen, Quanzheng Li, Tianming Liu, Dajiang Zhu, and Xiang Li. DeID-GPT: Zero-shot medical text de-identification by GPT-4. arXiv Preprint, March 2023. URL https://arxiv.org/abs/2303.11032.

[22] Constantinos Patsakis and Nikolaos Lykousas. Man vs the machine in the struggle for effective text anonymisation in the age of large language models. Scientific Reports, 2023.

[23] Iyadh Ben Cheikh Larbi, Aljoscha Burchardt, and Roland Roller. Clinical text anonymization, its influence on downstream NLP tasks and the risk of re-identification. In Elisa Bassignana, Matthias Lindemann, and Alban Petit, editors, Proceedings ofthe 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics: Student Research Workshop, pages 105–111, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi:10.18653/v1/2023.eacl-srw.11. URL https://aclanthology.org/2023. eacl-srw.11/.

[24] Thomas Vakili, Aron Henriksson, and Hercules Dalianis. End-to-end pseudonymization of fine-tuned clinical bert models: Privacy preservation with maintained data utility. BMC Medical Informatics and Decision Making, 2024.

[25] Priyam Basu, Tiasa Singha Roy, Rakshit Naidu, and Zumrut Muftuoglu. Privacy enabled financial text classification using differential privacy and federated learning. In Proc. Workshop ECONLP, 2021.

[26] Oleksandr Yermilov, Vipul Raheja, and Artem Chernodub. Privacy- and utility-preserving NLP with anonymized data: A case study of pseudonymization. In Proce. Workshop TrustNLP, 2023.

[27] Sakib Shahriar and Rozita Dara. Priv-iq: A benchmark and comparative evaluation of large multimodal models on privacy competencies. AI, 2025.

[28] Jinglin Sun, Basem Suleiman, Imdad Ullah, and Imran Razzak. Effectiveness of privacy-preserving algorithms in llms: A benchmark and empirical analysis. In Proc. WWW, 2025.

[29] Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. Safetybench: Evaluating the safety of large language models. In Proc. ACL, 2024.

[30] Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, J Zico Kolter, Matt Fredrikson, Yarin Gal, and Xander Davies. Agentharm: A benchmark for measuring harmfulness of LLM agents. In Proc. ICLR, 2025.

[31] Tobias Deußer, Lorenz Sparrenberg, Armin Berger, Max Hahnbück, Christian Bauckhage, and Rafet Sifa. A survey on current trends and recent advances in text anonymization. In Proc. DSAA, 2025.

[32] Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. Open llm leaderboard v2. https://huggingface.co/spaces/open-llm-leaderboard/open\_llm\_leaderboard, 2024.

[33] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. Transactions on Machine Learning Research, 2023.

[34] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIG-bench tasks and whether chain-ofthought can solve them. In Findings ofthe ACL: ACL, 2023.

[35] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proc. ACL, 2019.

[36] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models, 2023. URL https://arxiv.org/abs/ 2311.07911.

[37] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. MMLU-Pro: A more robust and challenging multi-task language understand ing benchmark. In Proc. NeurIPS, 2024.

[38] Zayne Rea Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. Musr: Testing the limits of chainof-thought with multistep soft reasoning. In The Twelfth International Conference on Learning Representations, 2024.

[39] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. In Proc. AAAI, 2024.

[40] Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Proc. ACL, 2022

[41] Mehdi Ali, Michael Fromm, Klaudia Thellmann, Jan Ebert, Alexander Arno Weber, Richard Rutmann, Charvi Jain, Max Lübbering, Daniel Steinigen, Johannes Leveling, et al. Teuken-7b-base & teuken-7b-instruct: Towards european llms, 2024. URL https://arxiv.org/abs/2410.03730.

[42] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models, 2024. URL https: //arxiv.org/abs/2407.21783.

[43] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv.org/abs/2408.00118.

[44] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models, 2025.

[45] OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card, 2024.