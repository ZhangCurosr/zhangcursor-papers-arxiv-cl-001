# Abstracts or full text? Evaluating input scope and computational efficiency in biomedical lay summarization

Egecan C¸ elik Evgin<sup>1Y\*</sup>, <sup>˙</sup>Ilknur Karadeniz<sup>2Y</sup>, Olcay Taner Yıldız<sup>1,3Y</sup>

1 Department of Computer Science, Ozye˘gin University, <sup>¨</sup> <sup>˙</sup>Istanbul, T¨urkiye

2 Department of Computer Engineering, Galatasaray University, <sup>˙</sup>Istanbul, T¨urkiye

3 Department of Artificial Intelligence and Data Engineering, Ozye˘gin University, <sup>¨</sup> <sup>˙</sup>Istanbul, T¨urkiye

YAll authors contributed equally to this work.

\* egecan.evgin@ozu.edu.tr; ikaradeniz@gsu.edu.tr; olcay.yildiz@ozyegin.edu.tr

## Abstract

Biomedical research articles are mainly written for experts and often contain terms that are difficult for the general public to understand. Automated lay summarization has become a solution to make biomedical knowledge easier for patients and the public to understand. However, generating lay summaries from full-text articles requires more computing power because the articles are long and costly to process. Fine-tuning large language models for this task also requires substantial resources. This study asks two questions: (1) whether abstracts contain enough information to produce lay summaries as good as those based on full-text articles, and (2) whether few-shot learning performs as well as task-specific fine-tuning. The study used two public biomedical lay summarization datasets and two recent models, evaluated for readability, factual consistency, semantic similarity, and runtime efficiency. Results show that abstract-based summarization performs better overall. Across both datasets and models, abstract fine-tuning is 4.70 times faster, while abstract zero- and few-shot prompting is 5.31 times faster than full-text methods. These results show that abstract-based prompting reduces runtime cost while maintaining lay summary quality.

## Introduction

Biomedical research articles are important means of communicating research findings in the life sciences [1]. These articles are highly structured and often use dense technical language that can be difficult for non-expert readers to understand [2, 3]. Lay summaries have become a useful way to make biomedical research more accessible to these readers [4, 5]. Unlike standard summaries, lay summaries prioritise readability and use simpler language, reducing domain-specific terminology and enabling non-expert readers to understand the main findings [6, 7]. However, automatically generating lay summaries is often more challenging, as it requires models to improve readability while keeping the summaries relevant and factual [8, 9]. For example, Figure 1 illustrates the differences between technical and lay summaries of the same biomedical research article.

<table><tr><td rowspan=1 colspan=1>Technical abstract</td><td rowspan=1 colspan=1>Lay summary</td></tr><tr><td rowspan=3 colspan=1>Loss of cd59 in zebrafish embryos causesexcessive Schwann cell proliferation anddisrupts myelination. Mutant nerves can exhibitreduced myelin volume, abnormal myelinultrastructure, and altered node of Ranvierassembly. Complement activity is elevated inthese embryos. Pharmacological inhibition ofinflammation    restores    Schwann    cellproliferation, myelin volume, and nodalorganization to wildtype levels, implicatingCd59 in the regulation of developmentalinflammation and peripheral nerve formation.</td><td rowspan=3 colspan=1>Nerves carry messages through the body. Cellscalled Schwann cells wrap nerves in a fattycovering called myelin, which helps thesemessages travel quickly. Researchers studiedyoung zebrafish without the cd59 gene. Thesefish developed too many Schwann cells, andtheir nerve coverings did not form properly. Adrug that reduced inflammation helped restorenormal cell numbers and nerve coverings. Thefindings show how inflammation can affect theway nerves develop.</td></tr><tr><td rowspan=1 colspan=1>ir n</td></tr><tr><td rowspan=1 colspan=1>eir r</td></tr></table>

Adapted from Wiltbank et al., eLife (2022). DOl: 10.7554/eLife.76640

Fig 1. Technical and lay descriptions of biomedical research. Shortened versions adapted from the abstract and eLife digest of Wiltbank et al. [10].

The community-wide BioLaySumm shared task initiative focuses on automatic biomedical lay summarisation and has attracted many researchers since 2023. The BioLaySumm shared task series encourages participants to develop and evaluate their systems within a common framework, and to compare their performance with other approaches [11–13]. The shared task comprised two subtasks, standard lay summarisation and summarisation with external knowledge, and introduced the PLOS and eLife datasets, which pair biomedical articles with written lay summaries. Although both data sets were curated from full-text journal articles, they differ in authorship and length: PLOS uses short, author-written lay summaries, while eLife is based on longer summaries, approximately twice the length, written by expert

editors. [14].

The length of the text increases the computational cost of training and inference, as longer articles require more tokens to be processed [15, 16]. In automatic lay summarisation, this aspect can become more critical, depending on the content provided as input to the model. For instance, an eLife abstract requires an average of 217 tokens, whereas the corresponding full-text article requires an average of 14,046 tokens, as shown in Table 2.

The existing literature demonstrates considerable variation in how much source information is provided to biomedical lay summarization models. Some systems rely exclusively on abstracts [17–20], whereas others use abstracts supplemented with retrieved full-text passages [21], abstracts combined with conclusions [22], selected sections [23], salient sentences [24, 25], or the complete article [26, 27]. In addition, studies differ in their model architectures, prompting strategies, fine-tuning procedures, preprocessing methods, and evaluation settings. Consequently, direct conclusions about the effect of input scope remain difficult to draw.

Although several studies have compared or combined different input configurations, these comparisons are often embedded within different modeling frameworks or evaluated under different experimental conditions. For example, abstract-based methods may employ few-shot prompting [17–20], whereas full-text approaches may rely on fine-tuning or long-document architectures [26, 27]. Similarly, content-selection systems [23, 24] introduce additional variables, which are the qualities of the selected sentences and the dependents of the extraction method. These differences make it difficult to determine whether improvements arise from the additional source information itself or from the particular model or pre-processing strategy used to exploit it.

A particularly relevant question is therefore how much of a biomedical article is actually necessary for high-quality lay summarization. An abstract provides a concise representation of the article while it may omit methodological details, findings, or contextual information that could improve factuality and informativeness. Conversely, full-text input provides substantially more information, on the other hand it increases sequence length, computational cost, and the amount of potentially irrelevant content that the model must process. Although selected-content approaches attempt to address this trade-off, they introduce an additional content-selection component.

The present work investigates the input-scope question explained above by systematically examining different amounts of source information under controlled experimental conditions. Rather than treating abstract-based and full-text summarization as fundamentally different tasks, we investigate them as alternative input configurations for the same general summarization objective. This enables a more direct analysis of the extent to which additional article-level information contributes to the quality of biomedical lay summarization, while also considering the associated information. In this study, we reported the

experimental results on two different datasets with two different models and tested the statistical significance of the experimental results. Interestingly, our results show the opposite trend, with abstract-based approaches performing better than full-text approaches.

Based on these issues, this study addresses two research questions:

• RQ1: Is an abstract alone sufficient to generate lay summaries that are comparable to those generated from the full article in terms of relevance, readability, and factual consistency?

• RQ2: Can few-shot prompting achieve performance comparable to task-specific LoRA fine-tuning while requiring less computational time?

The hypotheses of this study are as follows:

1. H1: Abstract-based LORA fine-tuning achieves performance comparable to full-text LORA fine-tuning while requiring less computational time.

2. H2: Three-shot prompting using abstract–summary example pairs achieves better overall performance compared with full-text LoRA fine-tuning while requiring less computational time.

3. H3: Without fine-tuning, few-shot prompting with abstracts achieves better overall performance compared with few-shot prompting with full-text inputs while requiring less computational time.

The main contribution of this study is a systematic, controlled evaluation of the sufficiency of abstracts for biomedical lay summarization, examining the trade-off between summary quality and computational efficiency. The results demonstrate that abstract-based LoRA fine-tuning achieves superior overall performance, as measured by a composite score integrating readability, relevance, and factual consistency, while reducing fine-tuning runtime by a factor of 4.70 compared with full-text LoRA fine-tuning. These findings challenge the assumption that access to full-text articles is necessary to produce high-quality biomedical lay summaries and suggest that abstracts may provide a more computationally efficient alternative without compromising, and potentially improving, overall summary quality.

Beyond fine-tuning, this study systematically investigates few-shot prompting with abstracts, broadening the comparison to include a parameter-efficient training approach and a training-free alternative. Furthermore, it expands the evaluation framework used in previous BioLaySumm studies by incorporating FENICE [28] and LongDocFACTScore [29] alongside established evaluation metrics. This multidimensional assessment provides a more comprehensive basis for comparing summarization strategies in terms of linguistic quality, relevance, factual consistency, and computational cost.

The remainder of this paper is organized as follows: Related Work, Methods, Results, Discussion, and Conclusion.

## Related work

Automatic summarization has been extensively studied across news, scientific, and biomedical domains. Scientific summarization presents particular challenges due to the length of research articles, the complexity of their content, and the need to preserve essential information [30]. To address the computational demands of long-document processing, models such as Longformer and Longformer Encoder-Decoder (LED) were developed to handle longer input sequences [31], while PEGASUS was designed specifically for abstractive summarization [32]. Biomedical lay summarization introduces an additional challenge: generated summaries must convey scientifically accurate information in language accessible to non-expert readers. An important methodological consideration is therefore how much of the source article is needed to generate summaries that balance relevance, readability, and factual consistency. Accordingly, this section reviews previous work from three perspectives based on input strategy: abstract-based methods, content-selection methods, and full-text methods.

## Abstract-based strategies

Several studies have compared abstract input with longer article input for biomedical lay summarization. Basu et al. compared abstract-only and full-text fine-tuning on the eLife validation set and found trade-offs between relevance, readability, factuality, and inference time [33]. However, the comparison was limited to a single eLife split, so it remains unclear whether the same pattern holds on PLOS. Goldsack et al. reported a similar comparison for zero-shot prompting, again on eLife [19]. Together, these studies suggest that abstracts can be competitive with full text, but the evidence remains limited across datasets and adaptation methods.

Other studies extended abstract-based prompting to both PLOS and eLife. LaySummX compared abstract-only and full-text zero-shot prompting and found that abstract input gave better factuality scores on both datasets, while relevance and readability varied between them [21]. The 5cNLP team also tested zero- and few-shot prompting with abstracts and different article sections [20]. These studies covered a wider range of prompting settings, but did not compare the same abstract and full-text conditions across both prompting and fine-tuning.

Several studies have combined abstract-based fine-tuning with prompting. Saama Technologies used LoRA fine-tuning on abstracts together with zero- and six-shot generation [18], while SINAI compared two-shot prompting with abstract-based fine-tuning [34]. DeakinNLP also compared fine-tuning with zeroand few-shot prompting, although its fine-tuning input included both abstracts and conclusions [35]. Earlier work by MDC similarly compared abstract-based fine-tuning with three-shot prompting [17]. These studies compared several adaptation methods, but did not apply the same methods to both abstract and full-text input.

Other work reduced the amount of article text used during fine-tuning. TLPIQ trained on the abstract and conclusion sections of PLOS and eLife [22], while MetninOzU limited each article to its first ten sentences [24]. Both approaches showed that shorter inputs can support competitive lay summarization, but neither directly compared abstract-only and full-text fine-tuning under the same conditions.

## Content selection strategies

Several studies have explored strategies for reducing the amount of source text processed during lay summary generation. RainCityNLP employed TF–IDF to select a subset of sentences from full-text articles prior to training summarization models [23, 36]. Similarly, MIRAGES adopted an extract-then-summarize approach, selecting salient sentences from full-text articles and fine-tuning Llama-3-8B-Instruct using low-rank adaptation (LoRA) [37]. These approaches aim to reduce the volume of input text processed by the summarization model while retaining relevant information from the full article.

Other systems have adopted alternative strategies to reduce the input length before summary generation. UIUC BioNLP extracted salient content from articles before fine-tuning GPT-3.5 [38], whereas BDA-UC3M used extracted content in conjunction with compact models on the PLOS and eLife datasets [39]. In contrast, Aard divided full-text articles into smaller chunks before performing zero-shot generation [40]. Although these methods differ in their implementation, they share the objective of avoiding the processing of an entire article as a single input.

More recent work has investigated content selection explicitly as a means of improving summarization efficiency. Sava¸slı et al. proposed GraphXtract, a graph-based approach that represents biomedical articles as graphs and selects salient sentences before LLM-based summary generation [25]. Evaluated on the PLOS dataset, the method was designed to reduce the volume of source text processed during summarization. Collectively, these approaches offer an alternative to strategies that rely exclusively on abstracts or process entire articles, enabling summarization models to operate on selected portions of the source text that are considered most informative.

## Full-text-based strategies

Full-text input has also been explored in biomedical lay summarization. Early BioLaySumm systems investigated long-document models, including BART and LED, to process full-text articles [26]. Similarly, ISIKSumm employed text chunking to accommodate article length and generated separate summaries using abstract and full-text inputs [27]. These studies explored strategies for incorporating information from longer biomedical documents, although the extent to which input scope affects summarization performance was not the primary focus.

CTyun AI adopted a related approach, using truncation and chunking to prepare full-text articles for fine-tuning [41]. Rather than processing each article as a single sequence, the system shortened or divided the input to accommodate the constraints of long-document processing. Although these strategies offer practical solutions for handling lengthy biomedical texts, the study did not include a matched abstract-only condition to directly assess the effect of input scope.

More recent research has investigated the potential of smaller language models for biomedical lay summarization. Nursal and Mitchell evaluated models with fewer than 8 billion parameters in a zero-shot setting using full-text articles [42]. Their findings indicated that smaller models could generate informative lay summaries, with some outputs also receiving favorable human evaluations. However, their evaluation did not extend to abstract-only input, few-shot prompting, or fine-tuning. Consequently, the relative effectiveness of different input scopes and adaptation strategies remains insufficiently characterized across these approaches.

## Methods

The methods are organized into three parts: dataset, fine-tuning and inference, and evaluation. Fig 2 shows the overall workflow of the study.

## Dataset

The PLOS and eLife datasets consist of biomedical research articles paired with expert-written lay summaries [14]. During preprocessing, abstracts were extracted from the article text and used as input for the abstract-only condition. The datasets were obtained from BioLaySumm 2025 [13]. Since gold-standard lay summaries were unavailable for the original test sets, new test sets were sampled from the training data. Each new test set comprised approximately 5% of the corresponding original training set, yielding a split close to $9 0 / 5 / 5$ for training, validation, and testing, respectively (Table 1).

![](images/bbfe8aa7de2aea6d81e4282ae49190160e1e556d1fcb4ec5c4ec5ccacc5ec510.jpg)  
Fig 2. Overall workflow of the study. Biomedical articles from the PLOS and eLife datasets were used in abstract-based and full-text settings. The study compared LoRA fine-tuning and zero-/few-shot prompting, followed by evaluation using relevance, readability, factuality, and computational efficiency metrics.

Table 1. Dataset split sizes for eLife and PLOS.
<table><tr><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td><td>Prompt set</td><td>Few-shot samples</td></tr><tr><td>eLife</td><td>4,066 (89.68%)</td><td>241 (5.32%)</td><td>227 (5.01%)</td><td>25</td><td>3</td></tr><tr><td>PLOS</td><td>23,413 (89.73%)</td><td>1,376 (5.27%)</td><td>1,305 (5.00%)</td><td>25</td><td>3</td></tr></table>

All sampling was performed before the experiments using a fixed random seed of 42, without stratification. The newly constructed test sets, the prompt-selection set, and the few-shot samples were held out from model training and fine-tuning to prevent data leakage. The prompt-selection set comprised 25 samples from each dataset, while three samples from each dataset were used for few-shot prompting. The number of few-shot samples was limited to three because some eLife articles exceed 35,000 tokens, potentially exceeding the model’s context-length limit.

Table 2. Token statistics for eLife and PLOS.
<table><tr><td>Dataset</td><td>Avg. article tokens</td><td>Max. article tokens</td><td>Min. article tokens</td><td>Avg. abstract tokens</td><td>Abstract-article token overlap</td></tr><tr><td>eLife</td><td>14,046.1</td><td>38,940</td><td>3,020</td><td>217.39</td><td>21%</td></tr><tr><td>PLOS</td><td>8,902.0</td><td>24,522</td><td>1,195</td><td>348.74</td><td>49%</td></tr></table>

Abstract–article token overlap was calculated for each article from the tokens shared by the abstract and the full article, and then averaged for each dataset. Token counts were obtained using the tokenizer provided with the Qwen/Qwen3.5-0.8B checkpoint.

As reported in previous work [14], PLOS articles are shorter on average than eLife articles, a difference also reflected in the token statistics reported in Table 2.

## Fine-tuning and inference

The experiments were conducted using Qwen3.5-0.8B [43] and MobileLLM-Pro-1B [44]. These models were selected because their context lengths were sufficient for the articles in the datasets and their computational requirements were compatible with the available hardware. All experiments were conducted in the same Google Colab environment using an NVIDIA L4 GPU.

Prompt selection Ten prompt formulations were evaluated on the prompt-selection set using both models with greedy decoding. The prompt with the highest mean score across the same 11 evaluation metrics used in the main experiments was selected for subsequent experiments. The selected prompt was “Summarize the following in layman’s terms:” and was used throughout the remaining experiments. The complete set of prompt formulations and the corresponding prompt-selection results are provided in the Supplementary Material.

Fine-tuning configuration All fine-tuning experiments used identical Low-Rank Adaptation (LoRA) settings. LoRA was applied to the q proj and v proj layers, with a rank ((r)) of 4, an alpha value of 8, and a dropout rate of 0.05. The rank was set to 4 in accordance with the original LoRA study, which demonstrated competitive performance using low-rank adaptation of query and value projections [15]. Models were trained for two epochs using a learning rate of $( 2 \times 1 0 ^ { - 4 } )$ , a batch size of 2, and eight gradient accumulation steps. Weight decay was set to 0.01, with 50 warm-up steps. Training was performed using bfloat16 precision, gradient checkpointing, and scaled dot-product attention (SDPA).

Fine-tuning was conducted under two input conditions: full-article input and abstract-only input. As reported in Table 2, abstracts contain approximately 25.5 times fewer tokens than full articles in PLOS and 64.6 times fewer in eLife. Each fine-tuning condition was trained once, starting from the same base model, without intermediate checkpoint selection.

Inference configuration The maximum number of newly generated tokens was set to 800 for all summaries. This limit was chosen to exceed the average lay-summary lengths reported for both PLOS and eLife [14]. Sampling was disabled using do sample=False, and generation employed greedy decoding.

Consequently, the temperature, top k, and top p parameters did not affect generation. The end-of-sequence token was also used as the padding token. Identical inference settings were applied to zero-shot, one-shot, three-shot, and fine-tuned generation.

## Evaluation

The generated summaries were evaluated along three dimensions: relevance, readability, and factuality. Relevance was assessed using ROUGE [45], BLEU [46], METEOR [47], and BERTScore [48]. ROUGE, BLEU, and METEOR quantify lexical similarity between generated summaries and their reference summaries, whereas BERTScore assesses semantic similarity using contextual embeddings. Readability was evaluated using the Flesch–Kincaid Grade Level (FKGL) [49], the Dale–Chall Readability Score (DCRS) [50], the SMOG index [51], and the Coleman–Liau Index (CLI) [52]. Lower values indicate greater readability, or easier-to-read text, for all four metrics. Factual consistency was assessed using SummaC [53], FENICE [28], and LongDocFactScore [29]. These metrics assess the consistency of generated content with the source text, providing complementary signals of potential factual errors.

For the normalization and aggregation of these three metrics, no explicit weighting was assigned to the three evaluation dimensions. Instead, all 11 metrics contributed equally to the overall score. Consequently, relevance and readability each accounted for (4/11) (36.36%) of the aggregate, while factuality accounted for (3/11) (27.27%). Before aggregation, each metric was normalized using min–max scaling. For metrics in which higher values indicate better performance, normalization was performed as follows:

$$
x ^ { \prime } = { \frac { x - \operatorname* { m i n } ( x ) } { \operatorname* { m a x } ( x ) - \operatorname* { m i n } ( x ) } } .\tag{1}
$$

For metrics in which lower values indicate better performance, the normalized score was calculated as:

$$
x ^ { \prime } = 1 - { \frac { x - \operatorname* { m i n } ( x ) } { \operatorname* { m a x } ( x ) - \operatorname* { m i n } ( x ) } } .\tag{2}
$$

These transformations map the metric values to a common scale on which higher normalized scores consistently indicate better performance. The overall score was calculated as the arithmetic mean of the 11 normalized metric scores.

A limited manual assessment was conducted to examine potential hallucinations in outputs with the lowest factuality scores. These outputs were inspected for unsupported or contradictory claims. The observations were consistent with the low factuality scores assigned to these outputs.

For statistical analysis, each evaluation metric was calculated for every prediction in the test set. Statistical significance between paired experimental conditions was assessed using a two-sided Wilcoxon signed-rank test [54], with scores paired by the corresponding test articles. The significance threshold was set to (α = 0.05). The rank-biserial correlation was reported as the effect size.

## Results

The results are presented according to the run time and the three hypotheses.

## Runtime

Table 3 shows the total runtimes of each setting across the two models and two datasets. All experiments were run in the same Google Colab environment with an NVIDIA L4 GPU.

Table 3. Total runtimes of each setting across two models and two datasets.
<table><tr><td>Dataset</td><td>Setting</td><td>Model</td><td>Fine-tuning duration</td><td>Inference duration</td><td>Total duration</td></tr><tr><td>eLife</td><td>Abstract Zero-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>28 min 41 s</td><td>28 min 41 s</td></tr><tr><td>eLife</td><td>Abstract One-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>35 min 12 s</td><td>35 min 12 s</td></tr><tr><td>eLife</td><td>Abstract Three-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>48 min 58 s</td><td>48 min 58 s</td></tr><tr><td>eLife</td><td>Full-Text Zero-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>1 h 19 min 11 s</td><td>1 h 19 min 11 s</td></tr><tr><td>eLife</td><td>Full-Text One-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>1 h 31 min 56 s</td><td>1 h 31 min 56 s</td></tr><tr><td>eLife</td><td>Full-Text Three-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>1 h 50 min 04s</td><td>1h 50 min04s</td></tr><tr><td>eLife</td><td>Abstract Zero-Shot (Fine-Tuned)</td><td>Qwen3.5-0.8B</td><td>24 min 15 s</td><td>1 h 19 min 00 s</td><td>1 h 43 min 15 s</td></tr><tr><td>eLife</td><td>Full-Text Zero-Shot (Fine-Tuned)</td><td>Qwen3.5-0.8B</td><td>2 h 48 min 04 s</td><td>5 h 29 min 00 s</td><td>8 h 17 min 04s</td></tr><tr><td>eLife</td><td>Abstract Zero-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>30 min 21 s</td><td>30 min 21 s</td></tr><tr><td>eLife</td><td>Abstract One-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>44 min 34 s</td><td>44 min 34 s</td></tr><tr><td>eLife</td><td>Abstract Three-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>56 min 04s</td><td>56 min 04 s</td></tr><tr><td>eLife</td><td>Full-Text Zero-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>1 h 28 min 54s</td><td>1 h 28 min 54 s</td></tr><tr><td>eLife</td><td>Full-Text One-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>1 h 46 min 42 s</td><td>1h 46 min 42s</td></tr><tr><td>eLife</td><td>Full-Text Three-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>2 h 08 min 51 s</td><td>2 h 08 min 51 s</td></tr><tr><td>eLife</td><td>Abstract Zero-Shot (Fine-Tuned)</td><td>MobileLLM-Pro (1B)</td><td>28 min 48 s</td><td>1h 24 min 57s</td><td>1 h 53 min 45 s</td></tr><tr><td>eLife</td><td>Full-Text Zero-Shot (Fine-Tuned)</td><td>MobileLLM-Pro (1B)</td><td>2 h 55 min 10 s</td><td>5 h 36 min 46 s</td><td>8 h 31 min 56 s</td></tr><tr><td>PLOS</td><td>Abstract Zero-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>3 h 03 min 41 s</td><td>3 h 03 min 41 s</td></tr><tr><td>PLOS</td><td>Abstract One-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>4h43 min 35s</td><td>4h43min 35 s</td></tr><tr><td>PLOS</td><td>Abstract Three-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>5 h 51 min 01 s</td><td>5 h 51 min 01 s</td></tr><tr><td>PLOS</td><td>Full-Text Zero-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>34 h 41 min 09 s</td><td>34 h 41 min 09 s</td></tr><tr><td>PLOS</td><td>Full-Text One-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>37 h 42 min 30 s</td><td>37 h 42 min 30 s</td></tr><tr><td>PLOS</td><td>Full-Text Three-Shot</td><td>Qwen3.5-0.8B</td><td></td><td>40 h 35 min 23 s</td><td>40 h 35 min 23 s</td></tr><tr><td>PLOS</td><td>Abstract Zero-Shot (Fine-Tuned)</td><td>Qwen3.5-0.8B</td><td>2 h 25 min 51 s 16 h 32 min 43 s</td><td>6 h 41 min 00 s</td><td>9 h 06 min 51 s</td></tr><tr><td>PLOS</td><td>Full-Text Zero-Shot (Fine-Tuned)</td><td>Qwen3.5-0.8B</td><td></td><td>28 h 05 min 00 s</td><td>44 h 37 min 43 s</td></tr><tr><td>PLOS</td><td>Abstract Zero-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>3 h 39 min 18 s</td><td>3 h 39 min 18 s</td></tr><tr><td>PLOS</td><td>Abstract One-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>5h 06 min 04s</td><td>5 h 06 min 04s</td></tr><tr><td>PLOS</td><td>Abstract Three-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>6 h 17 min 07 s</td><td>6 h 17 min 07s</td></tr><tr><td>PLOS</td><td>Full-Text Zero-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>36 h 01 min 52 s</td><td>36 h 01 min 52 s</td></tr><tr><td>PLOS</td><td>Full-Text One-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>39 h 23 min 51 s</td><td>39 h 23 min 51 s</td></tr><tr><td>PLOS</td><td>Full-Text Three-Shot</td><td>MobileLLM-Pro (1B)</td><td></td><td>44 h 53 min 21 s</td><td>44 h 53 min 21 s</td></tr><tr><td>PLOS</td><td>Abstract Zero-Shot (Fine-Tuned)</td><td>MobileLLM-Pro (1B)</td><td>2 h 37 min 29 s</td><td>7 h 34 min 50 s</td><td>10 h 12 min 19 s</td></tr><tr><td>PLOS</td><td>Full-Text Zero-Shot (Fine-Tuned)</td><td>MobileLLM-Pro (1B)</td><td>17 h 14 min 31 s</td><td>29 h 17 min 58 s</td><td>46 h 32 min 29 s</td></tr></table>

All experiments used a Google Colab L4 GPU with 24 GB of VRAM and the same parameter settings.

## H1: Abstract-based vs. full-text LoRA fine-tuning

H1 examines the effects of input scope on summary quality and fine-tuning runtime by comparing abstract-based and full-text LoRA fine-tuning. The comparison was conducted separately on the eLife and PLOS datasets using both models.

## H1: PLOS

Table 4 presents the results of abstract-based and full-text LoRA fine-tuning on PLOS. Both models achieved higher overall scores with abstract-based fine-tuning than with full-text fine-tuning (Qwen3.5-0.8B: Holm-adjusted $p < . 0 0 1 , r _ { r b } = . 6 8 $ MobileLLM-Pro-1B: Holm-adjusted $p < . 0 0 1 , r _ { r b } = . 7 6 )$

Table 4. PLOS fine-tuning results on the test set.
<table><tr><td></td><td colspan="4">Relevance</td><td colspan="4">Readability</td><td colspan="4">Factuality</td><td></td></tr><tr><td></td><td colspan="9">METOR</td><td></td><td>ENINCE</td><td></td><td></td><td></td></tr><tr><td>Settting</td><td>OUUGEY</td><td>LEUN</td><td></td><td>BERre</td><td>RKI]</td><td>DCS</td><td></td><td>MOG</td><td>4 CCI</td><td>←umma</td><td></td><td></td><td>Ionact</td><td>M←ean</td></tr><tr><td>MobileLLM-Pro Abstract Fine-Tuning</td><td>0.4052</td><td>0.1238</td><td>0.2257</td><td>0.8583</td><td>14.71</td><td></td><td>12.37</td><td>14.49</td><td>14.03</td><td>0.7654</td><td>0.9256</td><td>0.6234</td><td></td><td>0.7840</td></tr><tr><td>Qwen Abstract Fine-Tuning</td><td>0.2699</td><td>0.1119</td><td>0.2999</td><td>0.8633</td><td>13.86</td><td></td><td>11.77</td><td>14.89</td><td>17.11</td><td>0.7236</td><td>0.9249</td><td></td><td>0.7262</td><td>0.7262</td></tr><tr><td>Qwen Full-Text Fine-Tuning</td><td>0.4077</td><td>0.1197</td><td>0.1921</td><td>0.8448</td><td>14.66</td><td></td><td>13.39</td><td>15.34</td><td>15.95</td><td>0.7863</td><td>0.9482</td><td>0.5743</td><td></td><td>0.5526</td></tr><tr><td>MobileLLM-Pro Full-Text Fine-Tuning</td><td>0.0854</td><td>0.0090</td><td>0.1410</td><td>0.7919</td><td>16.31</td><td></td><td>11.79</td><td>15.19</td><td>15.35</td><td>0.7125</td><td>0.8067</td><td></td><td>0.5897</td><td>0.1671</td></tr></table>

Abstract-based fine-tuning also yielded lower scores on three of the four readability metrics for both models, indicating improvements in readability on those measures. As reported in Table 3, abstract-based fine-tuning was 4.90 times faster for Qwen3.5-0.8B and 4.56 times faster for MobileLLM-Pro-1B than full-text fine-tuning. These findings support H1 on PLOS in terms of both overall summary quality and runtime.

## H1: eLife

Table 5 presents the corresponding results for eLife. As on PLOS, abstract-based fine-tuning achieved higher overall scores than full-text fine-tuning for both models (Qwen3.5-0.8B: Holm-adjusted $p = . 0 0 4 , r _ { r b } = . 6 2 ;$ MobileLLM-Pro-1B: Holm-adjusted $p < . 0 0 1 , r _ { r b } = . 8 1 )$ . Among the reported eLife fine-tuning conditions, abstract-based fine-tuning with MobileLLM-Pro-1B achieved the highest overall score. Under full-text fine-tuning, Qwen3.5-0.8B achieved a higher overall score than MobileLLM-Pro-1B.

Table 5. eLife fine-tuning results on the test set.
<table><tr><td></td><td colspan="4">Relevance</td><td colspan="4">Readability</td><td colspan="4">Factuality</td><td></td></tr><tr><td></td><td colspan="9"></td><td colspan="4">←umma</td><td></td></tr><tr><td>Seting</td><td>OUUGE</td><td>BLEUN</td><td>METOR</td><td>BERSCCre</td><td>KILI]</td><td>DCRS</td><td>SM</td><td></td><td></td><td></td><td>ENICEY</td><td>Iogcty</td><td>M←ean</td></tr><tr><td>MobileLLM-Pro Abstract Fine-Tuning</td><td>0.3360</td><td>0.0797</td><td>0.1746</td><td>0.8378</td><td>12.21</td><td>10.99</td><td>12.20</td><td></td><td>14.38</td><td>0.8745</td><td>0.8634</td><td>0.6457</td><td>0.8149</td></tr><tr><td>Qwen Abstract Fine-Tuning</td><td>0.2357</td><td>0.0505</td><td>0.2482</td><td>0.8456</td><td>13.27</td><td>11.99</td><td>15.00</td><td>13.70</td><td>0.6669</td><td></td><td>0.7142</td><td>0.6295</td><td>0.5780</td></tr><tr><td>Qwen Full-Text Fine-Tuning</td><td>0.3483</td><td>0.0705</td><td>0.1638</td><td>0.8269</td><td>15.69</td><td>13.62</td><td>16.66</td><td>16.66</td><td>0.9911</td><td></td><td>0.9594</td><td>0.6001</td><td>0.4377</td></tr><tr><td>MobileLLM-Pro Full-Text Fine-Tuning</td><td>0.1098</td><td>0.0095</td><td>0.1331</td><td>0.7877</td><td>14.89</td><td>12.01</td><td>10.85</td><td>14.01</td><td></td><td>0.9032</td><td>0.8423</td><td>0.6055</td><td>0.3732</td></tr></table>

Qwen refers to Qwen3.5-0.8B, and MobileLLM-Pro refers to the MobileLLM-Pro-1B model.

The runtime results in Table 3 show that abstract-based fine-tuning was 4.83 times faster for Qwen3.5-0.8B and 4.49 times faster for MobileLLM-Pro-1B than full-text fine-tuning. Thus, the eLife results also support H1 in terms of overall summary quality and runtime.

## H2: Three-shot abstract prompting vs. full-text LoRA fine-tuning

H2 compares three-shot abstract prompting with full-text LoRA fine-tuning in terms of summary quality and runtime. The comparison was conducted separately on the eLife and PLOS datasets using both models.

## H2: PLOS

Table 6 presents the results for the best-performing prompting and fine-tuning strategies on PLOS.

Table 6. Best settings for each model on the PLOS test set.
<table><tr><td></td><td colspan="4">Relevance</td><td colspan="3">Readability</td><td colspan="4">Factuality</td><td></td></tr><tr><td></td><td colspan="9">MEEOR</td><td>ECE</td><td></td><td></td></tr><tr><td>Setting</td><td>OUUGEY</td><td>LEU</td><td></td><td>BERSCOr</td><td>I</td><td>DCRS</td><td>SMG</td><td>II</td><td>←ummay</td><td></td><td>Ioact</td><td>M←ean</td></tr><tr><td>Qwen 3-Shot Abstract</td><td>0.3471</td><td>0.0649</td><td>0.1678</td><td>0.8413</td><td>10.49</td><td>10.63</td><td>10.67</td><td>12.79</td><td>0.7712</td><td>0.8808</td><td>0.5930</td><td>0.6588</td></tr><tr><td>MobileLLM-Pro 3-Shot Abstract</td><td>0.3537</td><td>0.0812</td><td>0.1881</td><td>0.8417</td><td>13.14</td><td>11.31</td><td>12.37</td><td>14.98</td><td>0.8743</td><td>0.9246</td><td>0.5594</td><td>0.6108</td></tr><tr><td>MobileLLM-Pro Abstract Fine-Tuning</td><td>0.4052</td><td>0.1238</td><td>0.2257</td><td>0.8583</td><td>14.71</td><td>12.37</td><td>14.49</td><td>14.03</td><td>0.7654</td><td>0.9256</td><td>0.6234</td><td>0.5954</td></tr><tr><td>Qwen Abstract Fine-Tuning</td><td>0.2699</td><td>0.1119</td><td>0.2999</td><td>0.8633</td><td>13.86</td><td>11.77</td><td>14.89</td><td>17.11</td><td>0.7236</td><td>0.9249</td><td>0.7262</td><td>0.5890</td></tr><tr><td>Qwen Full-Text Fine-Tuning</td><td>0.4077</td><td>0.1197</td><td>0.1921</td><td>0.8448</td><td>14.66</td><td>13.39</td><td>15.34</td><td>15.95</td><td>0.7863</td><td>0.9482</td><td>0.5743</td><td>0.4659</td></tr><tr><td>MobileLLM-Pro Full-Text Fine-Tuning</td><td>0.0854</td><td>0.0090</td><td>0.1410</td><td>0.7919</td><td>16.31</td><td>11.79</td><td>15.19</td><td>15.35</td><td>0.7125</td><td>0.8067</td><td>0.5897</td><td>0.1093</td></tr></table>

Qwen refers to Qwen3.5-0.8B, and MobileLLM-Pro refers to the MobileLLM-Pro-1B model.

Three-shot abstract prompting achieved higher overall scores than full-text fine-tuning for both models (Qwen3.5-0.8B: $p = . 0 4 2$ $r _ { r b } = . 5 1 7 ;$ MobileLLM-Pro-1B: $p < . 0 0 1 , r _ { r b } = . 6 9 )$ . It also yielded lower scores on all four readability metrics for both models, indicating easier-to-read summaries across these measures.

In terms of runtime, three-shot abstract prompting was 7.63 times faster for Qwen3.5-0.8B and 7.41 times faster for MobileLLM-Pro-1B than full-text fine-tuning (Table 3). Moreover, three-shot abstract prompting achieved a higher overall mean score than abstract-based fine-tuning for both models. These results support H2 on PLOS in terms of overall summary quality and runtime.

## H2: eLife

Table 7 presents the results for the best-performing prompting and fine-tuning strategies on eLife.

Three-shot abstract prompting achieved higher overall scores than full-text fine-tuning for both models (Qwen3.5-0.8B: $p < . 0 0 1$ $r _ { r b } = . 9 5$ ; MobileLLM-Pro-1B: $p < . 0 0 1 , r _ { r b } = . 6 0 3 )$ . However, the results for relevance, readability, and factuality varied across the two models, indicating that the overall advantage did not translate into a uniform improvement across all evaluation dimensions.

Table 7. Best settings for each model on the eLife test set.
<table><tr><td></td><td colspan="4">Relevance</td><td colspan="4">Readability</td><td colspan="4">Factuality</td><td></td></tr><tr><td></td><td colspan="9">BLEEU</td><td></td><td>ENNICEY</td><td></td><td></td><td></td></tr><tr><td>Setting</td><td>ROUUGE</td><td></td><td>MMETOR</td><td>BERSCCr</td><td>KLI]</td><td></td><td>DCRS</td><td>SMMG</td><td></td><td>←umma</td><td></td><td></td><td>Ionact</td><td>M←ean</td></tr><tr><td>MobileLLM-Pro Abstract Fine-Tuning</td><td>0.3360</td><td>0.0797</td><td>0.1746</td><td>0.8378</td><td>12.21</td><td></td><td>10.99</td><td>12.20</td><td>14.38</td><td>0.8745</td><td>0.8634</td><td></td><td>0.6457</td><td>0.7472</td></tr><tr><td>MobileLLM-Pro 3-Shot Abstract</td><td>0.2929</td><td>0.0632</td><td>0.1535</td><td>0.8297</td><td>11.90</td><td></td><td>11.83</td><td>13.04</td><td>15.65</td><td>0.9054</td><td></td><td>0.9145</td><td>0.6585</td><td>0.6585</td></tr><tr><td>Qwen 3-Shot Abstract</td><td>0.3643</td><td>0.0605</td><td>0.1581</td><td>0.8373</td><td>10.56</td><td></td><td>11.06</td><td>11.75</td><td>12.69</td><td>0.7938</td><td></td><td>0.6995</td><td>0.5901</td><td>0.6372</td></tr><tr><td>Qwen Abstract Fine-Tuning</td><td>0.2357</td><td>0.0505</td><td>0.2482</td><td>0.8456</td><td>13.27</td><td></td><td>11.99</td><td>15.00</td><td>13.70</td><td>0.6669</td><td></td><td>0.7142</td><td>0.6295</td><td>0.5303</td></tr><tr><td>Qwen Full-Text Fine-Tuning</td><td>0.3483</td><td>0.0705</td><td>0.1638</td><td>0.8269</td><td>15.69</td><td></td><td>13.62</td><td>16.66</td><td>16.66</td><td>0.9911</td><td></td><td>0.9594</td><td>0.6001</td><td>0.4452</td></tr><tr><td>MobileLLM-Pro Full-Text Fine-Tuning</td><td>0.1098</td><td>0.0095</td><td>0.1331</td><td>0.7877</td><td></td><td>14.89</td><td>12.01</td><td>10.85</td><td>14.01</td><td>0.9032</td><td></td><td>0.8423</td><td>0.6055</td><td>0.3579</td></tr></table>

Qwen refers to Qwen3.5-0.8B, and MobileLLM-Pro refers to MobileLLM-Pro-1B.

Three-shot abstract prompting was 10.14 times faster for Qwen3.5-0.8B and 9.14 times faster for MobileLLM-Pro-1B than full-text fine-tuning (Table 3). Compared with abstract-based fine-tuning, three-shot abstract prompting achieved a higher overall mean score for Qwen3.5-0.8B, whereas abstract-based fine-tuning achieved a higher overall mean score for MobileLLM-Pro-1B. Nevertheless, the comparison with full-text fine-tuning supports H2 on eLife in terms of overall summary quality and runtime.

## H3: Abstract vs. full-text few-shot prompting

H3 investigates the effect of input scope on few-shot summarization by comparing abstract-based and full-text prompting under zero-shot, one-shot, and three-shot settings. The comparisons were conducted separately on the eLife and PLOS datasets using both models, with summary quality and runtime as the primary outcomes.

## H3: PLOS

Table 8 presents the few-shot inference results for both models on PLOS.

Abstract-based prompting achieved higher overall scores than full-text prompting across the evaluated zero-shot, one-shot, and three-shot settings for both models (Qwen3.5-0.8B: p = .002, r<sub>rb</sub> = .80; MobileLLM-Pro-1B: p < .001, r<sub>rb</sub> = .74). Among the evaluated prompting configurations, three-shot abstract prompting achieved the highest overall mean score for both models.

Abstract-based prompting was also computationally more efficient, being 8.28 times faster for Qwen3.5-0.8B and 8.00 times faster for MobileLLM-Pro-1B than full-text prompting (Table 3). These findings support H3 on PLOS in terms of overall summary quality and runtime.

Table 8. Few-shot inference results on the PLOS test set.
<table><tr><td></td><td colspan="4">Relevance</td><td colspan="4">Readability</td><td colspan="4">Factuality</td><td></td></tr><tr><td>Setting</td><td>OUUGE</td><td>BLEU</td><td>METOR</td><td>BERSr</td><td>LI]</td><td>CRS</td><td>SMOG</td><td>III</td><td></td><td>umma</td><td>ENNCE</td><td>IOoact</td><td>M←ean</td></tr><tr><td>Qwen 3-Shot Abstract</td><td>0.3471</td><td>0.0649</td><td>0.1678</td><td>0.8413</td><td></td><td>10.49</td><td>10.63</td><td>10.67</td><td>12.79</td><td>0.7712</td><td>0.8808</td><td>0.5930</td><td>0.7366</td></tr><tr><td>Qwen 1-Shot Abstract</td><td>0.3351</td><td>0.0700</td><td>0.1789</td><td>0.8426</td><td>10.73</td><td></td><td>10.73</td><td>10.74</td><td>12.65</td><td>0.6628</td><td>0.8788</td><td>0.5965</td><td>0.7364</td></tr><tr><td>Qwen 0-Shot Abstract</td><td>0.3508</td><td>0.0695</td><td>0.1828</td><td>0.8419</td><td>11.25</td><td>10.72</td><td></td><td>10.77</td><td>13.30</td><td>0.5204</td><td>0.8890</td><td>0.5953</td><td>0.7060</td></tr><tr><td>MobileLLM-Pro 3-Shot Abstract</td><td>0.3537</td><td>0.0812</td><td>0.1881</td><td>0.8417</td><td>13.14</td><td></td><td>11.31</td><td>12.37</td><td>14.98</td><td>0.8743</td><td>0.9246</td><td>0.5594</td><td>0.6954</td></tr><tr><td>MobileLLM-Pro 1-Shot Abstract</td><td>0.3497</td><td>0.0860</td><td>0.1909</td><td>0.8455</td><td>13.31</td><td></td><td>11.48</td><td>12.76</td><td>15.10</td><td>0.8545</td><td>0.9133</td><td>0.5657</td><td>0.6909</td></tr><tr><td>Qwen 1-Shot</td><td>0.3583</td><td>0.0811</td><td>0.1707</td><td>0.8283</td><td>12.07</td><td></td><td>11.76</td><td>12.11</td><td>14.10</td><td>0.9724</td><td>0.9519</td><td>0.5133</td><td>0.6465</td></tr><tr><td>Qwen 0-Shot</td><td>0.3199</td><td>0.0655</td><td>0.1558</td><td>0.8216</td><td>10.58</td><td></td><td>11.06</td><td>10.68</td><td>12.73</td><td>0.9257</td><td>0.9647</td><td>0.4832</td><td>0.6363</td></tr><tr><td>Qwen 3-Shot</td><td>0.3696</td><td>0.1031</td><td>0.1777</td><td>0.8283</td><td>13.16</td><td></td><td>12.57</td><td>13.83</td><td>14.91</td><td>0.8036</td><td>0.9479</td><td>0.5197</td><td>0.5666</td></tr><tr><td>MobileLLM-Pro 0-Shot Abstract</td><td>0.3336</td><td>0.0873</td><td>0.1906</td><td>0.8449</td><td>15.15</td><td></td><td>12.50</td><td>14.99</td><td>16.60</td><td>0.8454</td><td>0.9045</td><td>0.5871</td><td>0.5466</td></tr><tr><td>MobileLLM-Pro 0-Shot</td><td>0.2079</td><td>0.0426</td><td>0.2034</td><td>0.8470</td><td>15.72</td><td></td><td>12.60</td><td>14.81</td><td>17.29</td><td>0.8124</td><td>0.8832</td><td>0.4658</td><td>0.3320</td></tr><tr><td>MobileLLM-Pro 1-Shot</td><td>0.1855</td><td>0.0334</td><td>0.1701</td><td>0.8423</td><td>16.20</td><td></td><td>12.96</td><td>15.40</td><td>17.56</td><td>0.8352</td><td>0.8965</td><td>0.4966</td><td>0.2284</td></tr><tr><td>MobileLLM-Pro 3-Shot</td><td>0.1981</td><td>0.0684</td><td>0.1711</td><td>0.8306</td><td>15.91</td><td></td><td>12.94</td><td>15.81</td><td>17.47</td><td>0.8024</td><td>0.8532</td><td>0.4917</td><td>0.1951</td></tr></table>

Qwen refers to Qwen3.5-0.8B, and MobileLLM-Pro refers to the MobileLLM-Pro-1B model.

## H3: eLife

Table 9 presents the few-shot inference results for both models on eLife.

Table 9. Few-shot baseline results on the eLife test set.
<table><tr><td></td><td colspan="4">Relevance</td><td colspan="4">Readability</td><td colspan="4">Factuality</td><td></td></tr><tr><td></td><td>OUGEY</td><td></td><td>MMETOR</td><td>BERSr)</td><td>LI</td><td>DCRS</td><td>MMG</td><td></td><td>SUuma</td><td></td><td>ENICEY</td><td>Ioact</td><td></td></tr><tr><td>Setting</td><td></td><td>LTEU</td><td></td><td></td><td></td><td></td><td></td><td></td><td>II</td><td></td><td></td><td></td><td>M←ean</td></tr><tr><td>Qwen 3-Shot Abstract</td><td>0.3643</td><td>0.0605</td><td>0.1581</td><td>0.8373</td><td>10.56</td><td></td><td>11.06</td><td>11.75</td><td>12.69</td><td>0.7938</td><td>0.6995</td><td>0.5901</td><td>0.7441</td></tr><tr><td>Qwen 1-Shot Abstract</td><td>0.3404</td><td>0.0658</td><td>0.1514</td><td>0.8372</td><td>10.87</td><td>11.50</td><td></td><td>12.72</td><td>13.60</td><td>0.7167</td><td>0.7364</td><td>0.6028</td><td>0.6639</td></tr><tr><td>MobileLLM-Pro 3-Shot Abstract</td><td>0.2929</td><td>0.0632</td><td>0.1535</td><td>0.8297</td><td>11.90</td><td>11.83</td><td></td><td>13.04</td><td>15.65</td><td>0.9054</td><td>0.9145</td><td>0.6585</td><td>0.6589</td></tr><tr><td>Qwen 0-Shot Abstract</td><td>0.3052</td><td>0.0623</td><td>0.1527</td><td>0.8359</td><td>10.68</td><td></td><td>11.18</td><td>11.51</td><td>12.80</td><td>0.5161</td><td>0.7093</td><td>0.5776</td><td>0.6319</td></tr><tr><td>Qwen 1-Shot</td><td>0.3967</td><td>0.0786</td><td>0.1636</td><td>0.8173</td><td>12.10</td><td></td><td>12.00</td><td>14.28</td><td>14.99</td><td>0.8351</td><td>0.9472</td><td>0.4965</td><td>0.6230</td></tr><tr><td>Qwen 0-Shot</td><td>0.3979</td><td>0.0820</td><td>0.1725</td><td>0.8220</td><td>11.94</td><td></td><td>12.13</td><td>12.45</td><td>12.92</td><td>0.6233</td><td>0.6556</td><td>0.4825</td><td>0.6033</td></tr><tr><td>Qwen 3-Shot</td><td>0.3848</td><td>0.0742</td><td>0.1636</td><td>0.8147</td><td>12.90</td><td></td><td>12.68</td><td>14.56</td><td>14.48</td><td>0.9108</td><td>0.9325</td><td>0.4932</td><td>0.5721</td></tr><tr><td>MobileLLM-Pro 1-Shot Abstract</td><td>0.2435</td><td>0.0547</td><td>0.1392</td><td>0.8294</td><td>12.10</td><td></td><td>13.17</td><td>14.30</td><td>17.25</td><td>0.9054</td><td>0.9143</td><td>0.6326</td><td>0.4497</td></tr><tr><td>MobileLLM-Pro 0-Shot Abstract</td><td>0.2275</td><td>0.0513</td><td>0.1372</td><td>0.8308</td><td>14.97</td><td></td><td>12.62</td><td>15.22</td><td>16.58 15.94</td><td>0.8944</td><td>0.9327 0.7252</td><td>0.7004</td><td>0.4427</td></tr><tr><td>MobileLLM-Pro 0-Shot</td><td>0.2611</td><td>0.0519</td><td>0.1578</td><td>0.8203</td><td>15.01 17.53</td><td>12.04</td><td>12.22</td><td>13.34 12.97</td><td>16.89</td><td>0.7453 0.6943</td><td>0.6852</td><td>0.5459 0.5173</td><td>0.3873</td></tr><tr><td>MobileLLM-Pro 1-Shot</td><td>0.2507</td><td>0.0422</td><td>0.1539 0.1323</td><td>0.8125 0.8230</td><td>15.64</td><td>12.86</td><td></td><td>15.18</td><td>17.46</td><td>0.5238</td><td>0.6744</td><td>0.5005</td><td>0.2366</td></tr><tr><td>MobileLLM-Pro 3-Shot</td><td>0.2167</td><td>0.0435</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0954</td></tr></table>

Qwen refers to Qwen3.5-0.8B, and MobileLLM-Pro refers to the MobileLLM-Pro-1B model.

As observed on PLOS, abstract-based prompting achieved higher overall scores than full-text prompting across the evaluated zero-shot, one-shot, and three-shot settings for both models $( \mathrm { Q w e n 3 . 5 – 0 . 8 B } \colon p = . 0 1 8$ $r _ { r b } = . 5 4 ; \mathrm { M o b i l e L L M - P r o - 1 B : } \ p < . 0 0 1 , \ r _ { r b } = . 7 8 2 )$ . Three-shot abstract prompting achieved the highest overall mean score for both models.

The runtime advantage of abstract-based prompting was smaller on eLife than on PLOS. Nevertheless, abstract-based prompting was 2.49 times faster for Qwen3.5-0.8B and 2.48 times faster for

MobileLLM-Pro-1B than full-text prompting (Table 3). These findings support H3 on eLife in terms of overall summary quality and runtime.

## Discussion

This study investigated the effects of input scope on biomedical lay summarization by comparing abstract-based and full-text-based approaches across two models and two datasets. The discussion is organized around the three hypotheses, focusing on the implications of the findings for summary quality and computational efficiency.

## H1: Abstract-based vs. full-text LoRA fine-tuning

Abstract-based LoRA fine-tuning achieved higher overall summary quality scores than full-text fine-tuning across both datasets and models, while reducing fine-tuning runtime by an average factor of 4.7. These findings suggest that using abstracts as fine-tuning input can offer a favorable balance between summary quality and computational cost in the evaluated settings.

This result differs from Basu et al. [33], who reported better performance with full-text fine-tuning on several metrics using the eLife validation set. Their study also employed the larger Llama 3.1 Instruct 8B model. Differences in model architecture, model size, and experimental configuration may contribute to the contrasting findings, although the available evidence does not establish which factor accounts for the difference.

One possible explanation for the advantage of abstract-based fine-tuning is that abstracts provide a concise representation of the article’s principal findings and conclusions. Their shorter length also reduces the amount of input processed during training. Full-text articles may contain additional details that are useful for generating comprehensive summaries, but they also require substantially more computation. The present results indicate that, under the evaluated conditions, this additional input did not translate into higher overall scores. Thus, abstract-based fine-tuning represents a computationally efficient alternative to full-text fine-tuning without sacrificing overall performance in these experiments.

## H2: Three-shot abstract prompting vs. full-text LoRA fine-tuning

Three-shot prompting with abstract–lay-summary examples achieved higher overall scores than full-text LoRA fine-tuning and was, on average, 8.58 times faster across the two datasets and models. This finding suggests that providing a small number of examples of the desired summarization style can be an effective alternative to fine-tuning on full-text articles.

The results differ from those of DeakinNLP [35], where fine-tuning outperformed few-shot prompting. However, the methods were evaluated using different model families for the two approaches, limiting direct comparability. In the present study, three-shot prompting and LoRA fine-tuning were evaluated using the same models within a consistent experimental setup, providing a more controlled comparison of these approaches under the selected conditions.

A potential advantage of three-shot abstract prompting is that it demonstrates the desired lay-summary style through examples without requiring additional model training. Furthermore, using abstracts rather than full articles substantially reduces the amount of input processed during inference. Together, these characteristics may explain the observed computational advantage. The findings suggest that three-shot abstract prompting is a practical alternative to full-text LoRA fine-tuning when overall summary quality and runtime are considered jointly. However, the relative performance of prompting and fine-tuning may depend on the model, dataset, and evaluation criteria.

## H3: Abstract vs. full-text few-shot prompting

Abstract-based prompting achieved higher overall scores than full-text prompting across the evaluated zero-shot, one-shot, and three-shot settings, while being, on average, 5.31 times faster across both datasets and models. These results suggest that restricting the input to abstracts can improve computational efficiency while maintaining or improving overall summary quality in the evaluated prompting configurations.

Previous work, including 5cNLP [20], has investigated zero-shot and few-shot prompting with different demonstration settings. However, it did not examine the same matched comparison between abstract-based and full-text-based input. The present study therefore provides complementary evidence on how input scope affects prompting-based biomedical lay summarization.

The observed advantage of abstract-based prompting may be related to the concise nature of abstracts, which typically foreground the principal objectives, findings, and conclusions of a study. In contrast, full-text articles contain extensive methodological details and additional information that may not be necessary for producing a concise lay summary. Processing the full article also increases the input length and associated inference cost. Nevertheless, the current results do not establish whether the performance differences arise from reduced input length, differences in information content, or both. Further investigation would be needed to disentangle these factors.

Overall, the findings across H1–H3 indicate that abstract-based approaches achieved higher overall scores and lower computational costs than their full-text counterparts under the experimental conditions examined. This pattern was observed for both fine-tuning and prompting, suggesting that the advantages of abstract-based input are not limited to a single adaptation strategy. However, these conclusions should be interpreted within the scope of the two datasets and two models evaluated in this study.

## Conclusion

This study aimed to address a research gap by systematically comparing abstract-based fine-tuning and few-shot prompting with their full-text-based counterparts under a consistent experimental setup. The primary contribution is the finding that abstract-based approaches can achieve better overall lay summarization performance while substantially reducing runtime.

All three hypotheses were supported across both datasets and models. Abstract-based LoRA fine-tuning achieved higher overall lay summarization scores and was, on average, 4.70 times faster than full-text-based LoRA fine-tuning. Similarly, three-shot prompting using abstracts outperformed full-text-based LoRA fine-tuning while providing an 8.58-fold speedup. Furthermore, abstract-based prompting consistently outperformed full-text-based prompting across zero-shot, one-shot, and three-shot settings, with an average speedup of 5.31-fold.

These findings suggest that abstract-based approaches can offer an effective and computationally efficient alternative to full-text-based summarization, particularly when abstracts adequately capture the key information presented in the corresponding articles. By reducing the amount of input text, these approaches can lower computational costs and accelerate summary generation without compromising overall lay summary quality under the experimental conditions examined in this study.

This study has several limitations. First, the experiments were restricted to relatively small language models because of the computational demands associated with evaluating numerous experimental configurations, limited GPU resources, and model context-length constraints. Consequently, the extent to which these findings generalize to larger language models remains to be investigated. Second, the evaluation relied primarily on automatic metrics, and no clinician-based assessment was conducted. Future work should therefore investigate larger models and incorporate expert evaluations to assess the factual accuracy, clinical relevance, completeness, and readability of the generated lay summaries. Such evaluations would help establish whether the observed efficiency and performance advantages of abstract-based approaches extend to broader model settings and practical biomedical communication scenarios.

## Supporting information

S1 File. Supplementary material. A separate supplementary document is provided with this article. It contains the complete prompt formulations and prompt-selection results. Three articles were randomly selected as few-shot examples from eLife [55–57], and three from PLOS [58–60].

## References

1. Forero DA, Lopez-Leon S, Perry G. A brief guide to the science and art of writing manuscripts in biomedicine. Journal of Translational Medicine. 2020;18:425. doi:10.1186/s12967-020-02596-2.

2. Sollaci LB, Pereira MG. The introduction, methods, results, and discussion (IMRAD) structure: a fifty-year survey. Journal of the Medical Library Association. 2004;92(3):364-7.

3. Devaraj A, Marshall I, Wallace BC, Li JJ. Paragraph-level Simplification of Medical Texts. In: Proceedings of NAACL-HLT; 2021. p. 4972-84. doi:10.18653/v1/2021.naacl-main.395.

4. Nunn E, Pinfield S. Lay summaries of open access journal articles: Engaging with the general public on medical research. Learned Publishing. 2014;27(3):173-84. doi:10.1087/20140303.

5. King SRF, Pewsey E, Shailes S. Plain-language Summaries of Research: An inside guide to eLife digests. eLife. 2017;6:e25410. doi:10.7554/eLife.25410.

6. Rosenberg A, Bar´onikov´a S, Feighery L, Gattrell W, Olsen RE, Watson A, et al. Open Pharma recommendations for plain language summaries of peer-reviewed medical journal publications. Current Medical Research and Opinion. 2021;37(11):2015-6. doi:10.1080/03007995.2021.1971185.

7. Wen J, Yi L. Are plain language summaries more readable than scientific abstracts? Evidence from six biomedical and life sciences journals. Public Understanding of Science. 2025;34(1):114-26. doi:10.1177/09636625241252565.

8. Al-Hussaini I, Wu A, Mitchell C. Pathology Dynamics at BioLaySumm: The Trade-off between Readability, Relevance, and Factuality in Lay Summarization. In: Proceedings of the 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks; 2023. p. 592-601. doi:10.18653/v1/2023.bionlp-1.63.

9. Rosati D. GRASUM at BioLaySumm Task 1: Background Knowledge Grounding for Readable, Relevant, and Factual Biomedical Lay Summaries. In: Proceedings of the 22nd Workshop on

Biomedical Natural Language Processing and BioNLP Shared Tasks; 2023. p. 483-90.   
doi:10.18653/v1/2023.bionlp-1.46.

10. Wiltbank AT, Steinson ER, Criswell SJ, Piller M, Kucenas S. Cd59 and inflammation regulate Schwann cell development. eLife. 2022;11:e76640. doi:10.7554/eLife.76640.

11. Goldsack T, Luo Z, Xie Q, Scarton C, Shardlow M, Ananiadou S, et al. Overview of the BioLaySumm 2023 Shared Task on Lay Summarization of Biomedical Research Articles. In: Proceedings of the 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks. Toronto, Canada: Association for Computational Linguistics; 2023. p. 468-77. Available from: https://aclanthology.org/2023.bionlp-1.44/. doi:10.18653/v1/2023.bionlp-1.44.

12. Goldsack T, Scarton C, Shardlow M, Lin C. Overview of the BioLaySumm 2024 shared task on the lay summarization of biomedical research articles. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. Bangkok, Thailand; 2024. p. 122-31. Available from: https://aclanthology.org/2024.bionlp-1.10/. doi:10.18653/v1/2024.bionlp-1.10.

13. Xiao C, Zhao K, Wang X, Wu S, Yan S, Goldsack T, et al. Overview of the BioLaySumm 2025 Shared Task on Lay Summarization of Biomedical Research Articles and Radiology Reports. In: Proceedings of the 24th Workshop on Biomedical Language Processing. Vienna, Austria: Association for Computational Linguistics; 2025. p. 365-77. Available from: https://aclanthology.org/2025.bionlp-1.31/. doi:10.18653/v1/2025.bionlp-1.31.

14. Goldsack T, Zhang Z, Lin C, Scarton C. Making Science Simple: Corpora for the Lay Summarisation of Scientific Literature. In: Proceedings of the 2022 Conference on Empirical Methods in Natura Language Processing. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics; 2022. p. 10589-604. Available from: https://aclanthology.org/2022.emnlp-main.724/. doi:10.18653/v1/2022.emnlp-main.724.

15. Hu EJ, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, et al. LoRA: Low-Rank Adaptation of Large Language Models. In: International Conference on Learning Representations; 2022. Available from: https://openreview.net/forum?id=nZeVKeeFYf9. doi:10.48550/arXiv.2106.09685.

16. Tay Y, Dehghani M, Bahri D, Metzler D. Efficient Transformers: A Survey. ACM Computing Surveys. 2022;55(6):1-28. doi:10.1145/3530811.

17. Turbitt O, Bevan R, Aboshokor M. MDC at BioLaySumm Task 1: Evaluating GPT Models for Biomedical Lay Summarization. In: Proceedings of the 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks. Toronto, Canada: Association for Computational Linguistics; 2023. p. 611-9. Available from: https://aclanthology.org/2023.bionlp-1.65/. doi:10.18653/v1/2023.bionlp-1.65.

18. Kim H, Kanakarajan KR, Sankarasubbu M. Saama Technologies at BioLaySumm: Abstract-Based Fine-Tuned Models with LoRA. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 786-92. Available from: https://aclanthology.org/2024.bionlp-1.72/. doi:10.18653/v1/2024.bionlp-1.72.

19. Goldsack T, Scarton C, Lin C. Leveraging Large Language Models for Zero-shot Lay Summarisation in Biomedicine and Beyond. arXiv preprint arXiv:250105224. 2025. Available from: https://arxiv.org/abs/2501.05224. doi:10.48550/arXiv.2501.05224.

20. Lossio-Ventura JA, Chan C, Basavaraj A, Alatrista-Salas H, Pereira F, Inkpen D. 5cNLP at BioLaySumm2025: Prompts, Retrieval, and Multimodal Fusion. In: Soni S, Demner-Fushman D, editors. Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 215-31. Available from: https://aclanthology.org/2025.bionlp-share.27/. doi:10.18653/v1/2025.bionlp-share.27.

21. Lin F, Yu D. LaySummX at BioLaySumm: Retrieval-Augmented Fine-Tuning for Biomedical Lay Summarization Using Abstracts and Retrieved Full-Text Context. In: Soni S, Demner-Fushman D, editors. Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 202-14. Available from: https://aclanthology.org/2025.bionlp-share.26/. doi:10.18653/v1/2025.bionlp-share.26.

22. Bechler M, Crowther C, Luedke E, Schimka N, Sharaf I. TLPIQ at BioLaySumm: Hide and Seq, a FLAN-T5 Model for Biomedical Summarization. In: Soni S, Demner-Fushman D, editors. Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 196-201. Available from: https://aclanthology.org/2025.bionlp-share.25/. doi:10.18653/v1/2025.bionlp-share.25.

23. Wilson J, Pollack M, Edwards R, Bellamy A, Salgi H. RainCityNLP at BioLaySumm2025: Extract then Summarize at Home. In: Soni S, Demner-Fushman D, editors. Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational

Linguistics; 2025. p. 190-5. Available from: https://aclanthology.org/2025.bionlp-share.24/. doi:10.18653/v1/2025.bionlp-share.24.

24. Evgin E, Karadeniz I, Yıldız OT. MetninOzU at BioLaySumm2025: Text Summarization with Reverse Data Augmentation and Injecting Salient Sentences. In: Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 179-84. Available from: https://aclanthology.org/2025.bionlp-share.22/. doi:10.18653/v1/2025.bionlp-share.22.

25. Sava¸slı AC¸ , Sefer E, Karadeniz I. GraphXtract: Adaptive Graph-Based Sentence Selection for Resource-Efficient Lay Summarization with Large Language Models. In: Proceedings of the 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics. ACM; 2026. p. 97:1-97:6. doi:10.1145/3807503.3819376.

26. Liu Q, Ren X, Vydiswaran VGV. LHS712EE at BioLaySumm 2023: Using BART and LED to summarize biomedical research articles. In: Proceedings of the 22nd Workshop on Biomedical Natura Language Processing and BioNLP Shared Tasks. Toronto, Canada: Association for Computational Linguistics; 2023. p. 620-4. doi:10.18653/v1/2023.bionlp-1.66.

27. Colak C, Karadeniz L. ISIKSumm at BioLaySumm Task 1: BART-Based Summarization System Enhanced with Bio-Entity Labels. In: Proceedings of the 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks. Toronto, Canada: Association for Computational Linguistics; 2023. p. 636-40. Available from: https://aclanthology.org/2023.bionlp-1.69/. doi:10.18653/v1/2023.bionlp-1.69.

28. Zhou Y, Li H, Kumar S. FENICE: evaluating factual consistency in text generation using natural language inference. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing; 2024. p. 1120-32.

29. Nguyen T, Gupta R, Zhang Y. LongDocFACT: measuring factual consistency in long document summarization. Transactions of the Association for Computational Linguistics. 2024;12:450-62.

30. Cohan A, Dernoncourt F, Kim DS, Bui T, Kim S, Chang W, et al. A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents. In: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. vol. 2; 2018. p. 615-21. doi:10.18653/v1/N18-2097.

31. Beltagy I, Peters ME, Cohan A. Longformer: The Long-Document Transformer. arXiv preprint arXiv:200405150. 2020.

32. Zhang J, Zhao Y, Saleh M, Liu PJ. PEGASUS: Pre-training with Extracted Gap-Sentences for Abstractive Summarization. In: Proceedings of the 37th International Conference on Machine Learning. PMLR; 2020. p. 11328-39. Available from: https://proceedings.mlr.press/v119/zhang20ae.html.

33. Basu P, Cols J, Jarvis D, Park Y, Rodabaugh D. SUWMIT at BioLaySumm2025: Instruction-based Summarization with Contrastive Decoding. In: Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 240-8. Available from: https://aclanthology.org/2025.bionlp-share.29/. doi:10.18653/v1/2025.bionlp-share.29.

34. Chizhikova M, D´ıaz-Galiano MC, Ure˜na-L´opez LA, Mart´ın-Valdivia MT. SINAI at BioLaySumm: Self-Play Fine-Tuning of Large Language Models for Biomedical Lay Summarisation. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 804-9. Available from: https://aclanthology.org/2024.bionlp-1.74/. doi:10.18653/v1/2024.bionlp-1.74.

35. To HQ, Liu M, Huang G. DeakinNLP at BioLaySumm: Evaluating Fine-Tuning Longformer and GPT-4 Prompting for Biomedical Lay Summarization. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 744-52. Available from: https://aclanthology.org/2024.bionlp-1.67/. doi:10.18653/v1/2024.bionlp-1.67.

36. Sparck Jones K. A statistical interpretation of term specificity and its application in retrieval. Journal of Documentation. 1972;28(1):11-21.

37. Pong B, Chen JH, Jiang J, Jimenez A, Vahadi M. MIRAGES at BioLaySumm2025: The Impact of Search Terms and Data Curation for Biomedical Lay Summarization. In: Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 232-9. Available from: https://aclanthology.org/2025.bionlp-share.28/. doi:10.18653/v1/2025.bionlp-share.28.

38. You Z, Radhakrishna S, Ming S, Kilicoglu H. UIUC BioNLP at BioLaySumm: An Extract-then-Summarize Approach Augmented with Wikipedia Knowledge for Biomedical Lay

Summarization. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 132-43. Available from: https://aclanthology.org/2024.bionlp-1.11/. doi:10.18653/v1/2024.bionlp-1.11.

39. Ramzi I, Bedmar I. BDA-UC3M @ BioLaySumm: Efficient Lay Summarization with Small-Scale SoTA LLMs. In: Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 249-55. Available from: https://aclanthology.org/2025.bionlp-share.30/. doi:10.18653/v1/2025.bionlp-share.30.

40. Gupta A, Krishnamurthy P. Shared Task at BioLaySumm2025: Extract then Summarize Approach Augmented with UMLS-based Definition Retrieval for Lay Summary Generation. In: Soni S, Demner-Fushman D, editors. Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks). Vienna, Austria: Association for Computational Linguistics; 2025. p. 185-9. Available from: https://aclanthology.org/2025.bionlp-share.23/. doi:10.18653/v1/2025.bionlp-share.23.

41. Bao S, Zhao R, Zhang S, Zhang J, Wang W, Ru Y. Ctyun AI at BioLaySumm: Enhancing Lay Summaries of Biomedical Articles Through Large Language Models and Data Augmentation. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 837-44. Available from: https://aclanthology.org/2024.bionlp-1.79/. doi:10.18653/v1/2024.bionlp-1.79.

42. Nursal B, Mitchell CS. Understanding LLMs’ Summarization Capabilities: An Analysis of Biomedical Abstract and Lay Summary Generation. In: Findings of the Association for Computational Linguistics: ACL 2026. San Diego, California, United States: Association for Computational Linguistics; 2026. p. 11393-417. Available from: https://aclanthology.org/2026.findings-acl.554/. doi:10.18653/v1/2026.findings-acl.554.

43. Qwen Team. Qwen3.5: Towards Native Multimodal Agents [Internet]. Qwen Team; 2026. Available from: https://qwen.ai/blog?id=qwen3.5.

44. Huber P, Chang E, Wen W, Fedorov I, Elgamal T, Huang H, et al. MobileLLM-Pro Technical Report. arXiv preprint arXiv:251106719. 2025. Available from: https://arxiv.org/abs/2511.06719. doi:10.48550/arXiv.2511.06719.

45. Lin CY. ROUGE: a package for automatic evaluation of summaries. In: Text Summarization Branches Out: Proceedings of the ACL-04 Workshop. Association for Computational Linguistics; 2004. p. 74-81.

46. Papineni K, Roukos S, Ward T, Zhu WJ. BLEU: a method for automatic evaluation of machine translation. In: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics; 2002. p. 311-8.

47. Banerjee S, Lavie A. METEOR: an automatic metric for MT evaluation with improved correlation with human judgments. In: Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization. Association for Computational Linguistics; 2005. p. 65-72.

48. Zhang T, Kishore V, Wu F, Weinberger KQ, Artzi Y. BERTScore: Evaluating Text Generation with BERT. In: International Conference on Learning Representations; 2020. Available from: https://openreview.net/forum?id=SkeHuCVFDr. doi:10.48550/arXiv.1904.09675.

49. Kincaid JP, Fishburne Jr RP, Rogers RL, Chissom BS. Derivation of new readability formulas (automated readability index, fog count and flesch reading ease formula) for navy enlisted personnel. Chief of Naval Technical Training, Naval Air Station Memphis; 1975. Research Branch Report 8-75.

50. Chall JS, Dale E. Readability revisited: the new Dale–Chall readability formula. Cambridge, MA: Brookline Books; 1995.

51. McLaughlin GH. SMOG grading — a new readability formula. Journal of Reading. 1969;12(8):639-46.

52. Coleman M, Liau TL. A computer readability formula designed for machine scoring. Journal of Applied Psychology. 1975;60(2):283-4.

53. Laban P, Schnabel T, Bennett PN, Hearst MA. SummaC: re-visiting NLI-based models for inconsistency detection in summarization. arXiv preprint arXiv:211109525. 2021. doi:10.48550/arXiv.2111.09525.

54. Wilcoxon F. Individual Comparisons by Ranking Methods. Biometrics Bulletin. 1945;1(6):80-3. doi:10.2307/3001968.

55. Marcon L, Diego X, Sharpe J, M¨uller P. High-throughput mathematical analysis identifies Turing networks for patterning with equally diffusing signals. eLife. 2016;5:e14022. doi:10.7554/eLife.14022.

56. Clipman SJ, Mehta SH, Srikrishnan AK, Zook KJC, Duggal P, Mohapatra S, et al. Role of direct and indirect social and spatial ties in the diffusion of HIV and HCV among people who inject drugs: a cross-sectional community-based network analysis in New Delhi, India. eLife. 2021;10:e69174. doi:10.7554/eLife.69174.

57. Taylor MS, Altukhov I, Molloy KR, Mita P, Jiang H, Adney EM, et al. Dissection of affinity captured LINE-1 macromolecular complexes. eLife. 2018;7:e30094. doi:10.7554/eLife.30094.

58. B¨onsch C, Kempf C, Mueller I, Manning L, Laman M, Davis TME, et al. Chloroquine and its derivatives exacerbate B19V-associated anemia by promoting viral replication. PLOS Neglected Tropical Diseases. 2010;4(4):e669. doi:10.1371/journal.pntd.0000669.

59. Lacombe J, Hanley O, Jung H, Philippidou P, Surmeli G, Grinstein J, et al. Genetic and functional modularity of Hox activities in the specification of limb-innervating motor neurons. PLOS Genetics. 2013;9(1):e1003184. doi:10.1371/journal.pgen.1003184.

60. Sutherland TE, R¨uckerl D, Logan N, Duncan S, Wynn TA, Allen JE. Ym1 induces RELMα and rescues IL-4Rα deficiency in lung repair during nematode infection. PLOS Pathogens. 2018;14(11):e1007423. doi:10.1371/journal.ppat.1007423.