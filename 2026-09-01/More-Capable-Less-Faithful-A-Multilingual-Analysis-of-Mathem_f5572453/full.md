# More Capable, Less Faithful: A Multilingual Analysis of Mathematical (Un)Solvability Detection in LLMs

Maria-Eleni Zoumpoulidi\* Nikolaos Xiros\* Georgios Paraskevopoulos

Institute for Language and Speech Processing, Athena Research Center, Greece mzoumpoulidi@gmail.com {n.xiros,g.paraskevopoulos}@athenarc.gr

## Abstract

Solvability detection is one of the most challenging aspects of mathematical reasoning for Large Language Models (LLMs). While prior work has studied this capability extensively, these analyses have been limited to English. Consequently, it remains unclear whether multilingual failures arise from differences in internal Solvability Belief or from languagedependent failures to express it. To address this gap, we introduce the first multilingual benchmark of paired solvable and unsolvable mathematical problems, extending ReliableMath to French and Greek. Using this, we train multilingual probes predicting Solvability Belief and analyze the solvability detection capabilities of state-of-the-art LLMs behaviorally, representationally, and in terms of faithfulness. We find that Solvability Belief is encoded as a largely universal, language-agnostic feature, and that higher-resource languages such as English, despite achieving stronger mathematical reasoning performance, exhibit lower solvability-detection faithfulness.

## 1 Introduction

Studying the mathematical capabilities of Large Language Models (LLMs) across multiple languages is an important and growing research direction, aimed at improving performance across languages.

Research on multilingual mathematical reasoning has focused primarily on accuracy over solvable problems. Existing work spans benchmarks (e.g., Shi et al. (2023), Xu et al. (2026)), training approaches (Chen et al., 2024), and prompting strategies such as Shi et al. (2023), and Barua et al. (2026). Representation-level studies investigate the encoding of problem difficulty Civelli et al. (2026). However, multilingual solvability detection remains underexplored.

The picture is different for solvability detection in English. Prior work covers benchmarks and incontext learning (e.g., Xue et al. (2025), Sun et al. (2024)) as well as investigations of models’ internal representations. More specifically, Xiros et al. (2026) and Sanyal et al. (2025) disentangle Solvability Belief and verbalization in LLMs, showing that they correspond to distinct, geometrically decoupled directions. Furthermore, Liu et al. (2026) show that LLMs often internally recognize unsolvable problems despite failing to abstain.

However, none of these works investigates whether models’ internal Solvability Belief transfers faithfully across languages or whether this belief is verbalized consistently in multilingual settings. Consequently, it remains unknown whether multilingual mathematical reasoning failures stem from differences in internal knowledge or from language-dependent failures to express it. To address this gap, we construct a multilingual version of ReliableMath (Xue et al., 2025) by translating both its solvable problems and their unsolvable counterparts into French and Greek. Using this benchmark, we investigate how language affects both models’ internal representations of solvability and the faithfulness with which these representations are reflected in their generated responses.

We aim to answer the following research questions: RQ1: How does LLM performance in mathematical reasoning and unsolvability detection vary across languages? RQ2: How does language affect the faithfulness between internal solvability representations and textual solvability judgments?

Our main contributions are:

• We introduce the first multilingual benchmark for paired solvable-unsolvable mathematical problems, extending ReliableMath to French and Greek. Our code and data will be available under the Apache 2.0 license.

• We provide a behavioral analysis of multilingual mathematical reasoning and unsolvability detection across state-of-the-art LLMs.

• We present the first multilingual representational analysis of Solvability Belief and its faithfulness to textual outputs, showing that stronger languages tend to exhibit lower solvability-detection faithfulness.

## 2 Methodology

Our methodology comprises four stages: benchmark translation, chain-of-thought (CoT) generation and hidden-state extraction, probing of solvability representations and LLM-based annotation of the output text’s solvability verdict. Each stage is described in detail below.

Benchmark Translation: We translate the ReliableMath benchmark (Xue et al., 2025) (the prompt used can be found in Appendix A), including both solvable problems and their unsolvable counterparts, into French and Greek using Claude Sonnet 5 (Anthropic, 2026). We perform human validation to ensure that mathematical content, numerical values, and solvability type are preserved (Appendix G).

CoT generation and hidden-state extraction: For each language, we generate Chain-of-Thought (CoT) responses using language-specific prompt templates (Appendix B). We consider two prompting settings: standard, which instructs the model to solve the problem step by step, and aware, which additionally allows it to explicitly state that a problem is unsolvable. We use the hidden states extracted from the layer yielding the highest probing performance (Appendix D), with each sequence represented by 20 uniformly sampled token vectors.

Probing: We probe Solvability Belief (SB), using the ground-truth solvability label of the problem as a proxy for the model’s internal belief. Hidden-state representations from the standard and solvability-aware prompting settings are pooled to train a single L1-regularized logistic regression probe for predicting SB.

Textual Solvability Verdict Annotation: Determining the textual solvability verdict is non-trivial, as reasoning traces for unsolvable problems often contain shifts in the model’s assessment throughout the generation process. Rather than relying on stricter alternatives, such as matching predefined phrases, we adopt an LLM-as-a-judge approach using Llama-3.3-70B-Instruct (Grattafiori et al., 2024). The judge infers the dominant solvability verdict conveyed by the reasoning trace, distinguishing between solution-oriented reasoning and recognition of unsolvability (the corresponding prompt can be found in Appendix C). The judge was validated through human annotation on a subset of 100 samples, achieving a high agreement score (see Appendix F).

## 3 Experiments

Models: For our experiments, we query Qwen3- 30B-A3B-Instruct-2507, Qwen3-4B-Instruct-2507, (Yang et al., 2025), Llama-3.1-8B-Instruct (Grattafiori et al., 2024), Gemma4-31B-IT (Google DeepMind, 2026). Additionally, we include two language-specialized variants of Llama-3.1-8B-Instruct: Llama-Krikri-8B (Roussis et al., 2025), adapted to Greek, and French-Alpaca (Pacifico, 2024), adapted to French. For these two models, we evaluate only on English and their respective specialization language.

Datasets: We use our translated version of ReliableMath, which comprises 313 solvable problems drawn from MATH (Hendrycks et al., 2021) (highschool level), MinervaMath (Math-AI, 2025) (competitive college level), and AIME24/AMC (AI-MO, 2024a,b) (Olympiad level), together with 1,102 unsolvable variants obtained by removing or contradicting information essential to the solution.

## 4 Results

## 4.1 Models Achieve Strongest Mathematical Reasoning On Their Dominant Language

We first benchmark how accurately each model solves problems on the solvable split of our dataset, per language, under a standard prompt, in order to measure raw task capability across languages.

Table 1 shows a clear capability hierarchy that is largely independent of language: Gemma-4-31B-it and both Qwen3 variants substantially outperform the Llama-based models across all three languages. In nearly every model, performance is best in English and degrades as we move to French and further to Greek. This pattern is consistent with our expectation that models reason more effectively in the language dominating their pretraining data.

<table><tr><td>Model</td><td>English</td><td>French</td><td>Greek</td></tr><tr><td>Gemma-4-31B-it</td><td>61.3</td><td>55.6</td><td>56.5</td></tr><tr><td>Qwen3-4B-Instruct</td><td>52.1</td><td>53.3</td><td>43.5</td></tr><tr><td>Qwen3-30B-Instruct</td><td>54.3</td><td>53.7</td><td>49.8</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>23.3</td><td>15.0</td><td>14.4</td></tr><tr><td>French-Alpaca-Llama3-8B</td><td>7.7</td><td>7.4</td><td></td></tr><tr><td>Llama-Krikri-8B-Instruct</td><td>17.6</td><td>一</td><td>14.7</td></tr></table>

Table 1: Accuracy (%) on the solvable split under the standard prompt, by model and language.

## 4.2 Internal Solvability Belief Is Strongest In The Dominant Language, But Highly Universal

Moving from raw accuracy on solving problems, we next assess the validation AUC of the Solvability Belief (SB) probe, which measures how sharply a model’s hidden states encode the solvability of a problem internally. We evaluate this across each of the three languages individually, using both per-language probes and a pooled universal probe trained jointly on all available languages.

Solvability Belief is less encoded in underrepresented languages Figure 1 plots validation AUC by language for both probe types. Across all six models, AUC point estimates are highest in English and decline as we move to French and further to Greek, for both the per-language and universal probes. Notably, the drop also persists for the two language-specialized models (Llama-Krikri-8B and French-Alpaca-8B), suggesting that the dominant pretraining language outweighs the language-specific fine-tuning. All Llama-based models, having weaker baseline reasoning capability as shown before, sit lowest throughout. The sharper encoding of Solvability Belief in English is consistent with the stronger reasoning performance observed for English in Section 4.1. Significance testing details are provided in Appendix E.1.

Solvability Belief Probe is highly universal As shown in Figure 1, pooling all available languages into a single universal probe matches or exceeds the per-language probe performance for every model. A bootstrap check confirms that in most cases this improvement is statistically significant (Appendix E.1). This indicates the Solvability Belief signal is largely encoded in a shared, only mildly language-specific subspace. Thus, we adopt the universal probe for our analysis.

![](images/7a83262be9ab83f28c32ac858f0c80b9a93366ee464c1f292153781a3413d2e8.jpg)  
Figure 1: Validation AUC by language for the Solvability Belief probe. Probe AUC degrades monotonically from English to Greek for every model. \*=languagespecialized.

## 4.3 Language Adaptation Raises Abstention

Having studied how models encode solvability internally, we now measure the behavioral abstention profiles across models and languages. Figure 2 reports the percentage of true abstentions on the unsolvable split alongside false-abstentions (false refusals) on the solvable split, under both prompt settings.

Under the standard prompt (Figure 2a), abstention is low for most models: only the two Qwen3 variants abstain at substantial rates (∼25%), with Llama-Krikri-8B barely abstaining at all. Notably, under the standard prompt, the two Qwen3 variants perform better in the underrepresented languages than in English: both Qwen3-4B and Qwen3-30B show a higher hit rate on the unsolvable split and a lower false-abstentions rate on the solvable split in Greek than in English.

The aware prompt raises abstention sharply for every model (Figure 2b). The natively multilingual models (Gemma-4-31B, Qwen3 variants) flag unsolvable problems at a comparable rate across all three languages, but their false-abstentions rate rises as we move from English to French and Greek. Llama-3.1-8B, with virtually no Greek pretraining, abstains markedly better in English than in French or Greek. Its language-adapted variants reverse this: French-Alpaca-8B abstains most in French and Llama-Krikri-8B in Greek. Crucially, in both models, this gain is not free: the surge of true abstentions in the target language is accompanied by a spike in false-abstentions concentrated in that same language. Thus, under aware prompting, language adaptation appears to lower the abstention threshold globally for the target language, increasing both true hits and false refusals.

![](images/a909e593617c8457578c1db6058b8fbe60ac83ef0c4596ef82eb76950ff9284b.jpg)  
(a) Standard prompt

![](images/53f177088e2713bd0319445c966e092c8f65c12d8d5effdd4a354c919d1b90f8.jpg)  
(b) Aware prompt  
Figure 2: Abstention rates (%) across English, French, and Greek for various language models. The segments distinguish between the hit rate (true abstention) on the unsolvable split and the false-abstentions rate on the solvable split for the standard prompt (left) and the aware prompt (right).

## 4.4 Models Are Least Faithful In Their Dominant Language

Having found that models reason more capably in their dominant language (Section 4.1) and encode Solvability Belief more sharply in that same language (Section 4.2), yet do not verbalize that belief any more reliably (Section 4.3), we are motivated to directly measure faithfulness across languages. For each sample, we compare the universal Solvability Belief probe’s prediction against the textual verdict assigned by the LLM judge, and define their agreement rate as the faithfulness rate.

Table 2, which reports the overall faithfulness rate across all prompt styles and solvability splits, shows a striking reversal of the pattern observed for capability and internal belief: for the natively multilingual models (Gemma and Qwen), faithfulness is higher in the underrepresented languages than in English, whereas the least multilingual Llama-3.1-8B shows the opposite ordering, with faithfulness highest in English and lowest in Greek. The two language-specialized models follow the same underrepresented-language pattern: both Krikri-8B and Alpaca-8B are more faithful in their respective specialization language (Greek and French) than in English.

To better understand this effect, we restrict our faithfulness analysis to the unsolvable split and compare standard→aware prompt pairs (Table 3). As shown, the aware prompt raises faithfulness for every model, but far more so in underrepresented languages: the gain is smallest in English and largest in French or Greek across all six models. This suggests that explicitly prompting models to consider unsolvability closes the gap between internal belief and verbalized output more effectively in lower-resource languages than in English. Significance testing details are provided in Appendix E.2.

<table><tr><td>Model</td><td>En</td><td>Fr</td><td>Gr</td></tr><tr><td>Gemma-4-31B</td><td>0.612</td><td>0.615</td><td>0.626</td></tr><tr><td>Qwen3-4B</td><td>0.691</td><td>0.728</td><td>0.728</td></tr><tr><td>Qwen3-30B</td><td>0.712</td><td>0.730</td><td>0.723</td></tr><tr><td>Llama-3.1-8B</td><td>0.544</td><td>0.529</td><td>0.498</td></tr><tr><td>Alpaca-8B</td><td>0.473</td><td>0.498</td><td></td></tr><tr><td>Krikri-8B</td><td>0.500</td><td></td><td>0.510</td></tr></table>

Table 2: Faithfulness rate, pooled across prompt styles and solvability types.

<table><tr><td>Model</td><td>En</td><td>Fr</td><td>Gr</td></tr><tr><td>Gemma-4-31B</td><td></td><td>0.271→0.762 0.253→0.791 0.287→0.787</td><td></td></tr><tr><td>Qwen3-4B</td><td></td><td>0.484→0.773 0.532→0.815 0.535→0.832</td><td></td></tr><tr><td>Qwen3-30B</td><td></td><td>0.509→0.796 0.524→0.819 0.506→0.829</td><td></td></tr><tr><td>Llama-3.1-8B</td><td>0.400→0.5100.374→0.514 0.305→0.523</td><td></td><td></td></tr><tr><td>Alpaca-8B</td><td>0.330→0.434 0.308→0.558</td><td></td><td></td></tr><tr><td>Krikri-8B</td><td>0.375→0.413</td><td></td><td>0.273→0.556</td></tr></table>

Table 3: Faithfulness rate on unsolvable problems, standard→aware prompt.

## 5 Conclusions

We introduced the first multilingual version of ReliableMath, extended to French and Greek, and used it to study solvability detection both behaviorally and through models’ internal representations. We find a consistent capability hierarchy, with models performing best in English and worst in Greek, but the opposite pattern for faithfulness: multilingual models are more faithful to their internal Solvability Belief in underrepresented languages than in English. Our findings suggest that a model’s dominant language yields stronger raw capability but not more trustworthy self-reports.

## Limitations

We acknowledge that, while our analysis provides informative insights, it would benefit from broader language coverage. Future work will extend our analysis to additional high-resource languages (e.g., Chinese) and typologically diverse, lower-resource languages. We also plan to leverage these insights to develop methods that better align textual judgments with models’ internal Solvability Beliefs, improving the reliability of multilingual solvability detection.

## Ethical Considerations

This work is purely analytical: we evaluate and probe existing publicly available models and do not release any new model, nor propose any method intended for deployment. Our benchmark is a direct translation of ReliableMath (Xue et al., 2025) into French and Greek; the underlying problems are mathematical in nature and contain no personal, sensitive, or offensive content, and the translation preserves the original items without adding, removing, or reweighting any of them.

## References

AI-MO. 2024a. Aimo validation aime.

AI-MO. 2024b. Aimo validation amc.

Anthropic. 2026. Introducing claude sonnet 5.

Josh Barua, Seun Eisape, Kayo Yin, and Alane Suhr. 2026. Long chain-of-thought reasoning across languages. Preprint, arXiv:2508.14828.

Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Dongmei Zhang, and Jia Li. 2024. Breaking language barriers in multilingual mathematical reasoning: Insights and observations. Preprint, arXiv:2310.20246.

Stefano Civelli, Pietro Bernardelle, Nicolò Brunello, and Gianluca Demartini. 2026. A shared geometry of difficulty in multilingual language models. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pages 796–807, San Diego, California, United States. Association for Computational Linguistics.

Google DeepMind. 2026. Gemma 4 model card. https://ai.google.dev/gemma/docs/ core/model\_card\_4. Google AI for Developers.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh

Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. NeurIPS.

Yi Liu, Xiangyu Liu, Zequn Sun, and Wei Hu. 2026. Answering the unanswerable is to err knowingly: Analyzing and mitigating abstention failures in large reasoning models. Preprint, arXiv:2508.18760.

Math-AI. 2025. Minervamath. https: //huggingface.co/datasets/math-ai/ minervamath. Hugging Face dataset.

Jonathan Pacifico. 2024. French-alpacallama3-8b-instruct-v1.0. https: //huggingface.co/jpacifico/ French-Alpaca-Llama3-8B-Instruct-v1.0.

Dimitris Roussis, Leon Voukoutis, Georgios Paraskevopoulos, Sokratis Sofianopoulos, Prokopis Prokopidis, Vassilis Papavassileiou, Athanasios Katsamanis, Stelios Piperidis, and Vassilis Katsouros. 2025. Krikri: Advancing open large language models for Greek. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 5012–5033, Suzhou, China. Association for Computational Linguistics.

Debdeep Sanyal, Manya Pandey, Dhruv Kumar, Saurabh Deshpande, and Murari Mandal. 2025. Confidence is not competence. Preprint, arXiv:2510.24772.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations.

YuHong Sun, Zhangyue Yin, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Hui Zhao. 2024. Benchmarking hallucination in large language models based on unanswerable math word problem. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2178–2188, Torino, Italia. ELRA and ICCL.

Nikolaos Xiros, Maria-Eleni Zoumpoulidi, and Georgios Paraskevopoulos. 2026. Knowledge knows, verbalization tells: Disentangling latent directions for mathematical solvability in llms. Preprint, arXiv:2607.05013.

Tianyi Xu, Kosei Uemura, Alfred Malengo Kondoro, Tadesse Destaw Belay, Catherine Nana Nyaah Essuman, Ifeoma Okoh, Ganiyat Afolabi, Ayodele Awokoya, and David Ifeoluwa Adelani. 2026.

Mgsm-pro: A simple strategy for robust multilingual mathematical reasoning evaluation. Preprint, arXiv:2601.21225.

Boyang Xue, Qi Zhu, Rui Wang, Sheng Wang, Hongru Wang, Minda Hu, Fei Mi, Yasheng Wang, Lifeng Shang, Qun Liu, and Kam-Fai Wong. 2025. Reliablemath: Benchmark of reliable mathematical reasoning on large language models. Preprint, arXiv:2507.03133.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

## A Translation Prompt

The prompt used for the dataset translation can be found in Table 7.

## B CoT Prompts

We present the prompts used for chain-of-thought (CoT) generation across all three languages, for both the standard and unsolvable-aware prompting settings. The standard prompts can be found in Table 9, while the unsolvable-aware prompts in Table 10.

## C Judge Prompt

The prompt used by the LLM judge to annotate models’ verbal responses is shown in Table 11.

## D Layer selection

In Table 4, we report the selected layer for each model, which corresponds to the one achieving the highest probe AUC. For all models, these correspond to the middle or middle-to-late layers.

<table><tr><td>Model</td><td>Layer</td></tr><tr><td>Gemma-4-31B-it</td><td>36</td></tr><tr><td>Qwen3-4B-Instruct</td><td>18</td></tr><tr><td>Qwen3-30B-Instruct</td><td>36</td></tr><tr><td>Llama-3.1-8B-Instruct/ KriKri / Alpaca</td><td>16</td></tr></table>

Table 4: Selected layer used for our experiments. For each model, we use the layer with the highest probing performance.

## E Significance testing details

## E.1 Significance of Solvability Belief Results

We compute problem-level bootstrap confidence intervals (1000 resamples, 95% level) for the two claims in Section 4.2. Since English, French, and Greek data are parallel translations of the same problems, we use a paired bootstrap, resampling the same problems across languages for direct comparison. All effect sizes below are AUC differences reported ×100.

English vs. other languages The English advantage over Greek is significant for Qwen3-4B (+3.8, CI [1.3, 6.5]), Qwen3-30B (+2.8, CI [0.5, 5.2]), Gemma-4-31B (+3.8, CI [1.8, 6.0]), and Llama-Krikri-8B (+7.9, CI [4.8, 10.7]). Notably, Krikri has the largest gap we observe, despite being Greekadapted. The advantage over French is significant only for Qwen3-4B (+4.2, CI [2.1, 6.5]) and Gemma-4-31B (+2.8, CI [1.0, 4.7]). For Llama-3.1-8B and French-Alpaca-8B, point estimates favor English throughout (+0.8 to +2.8) but all CIs cross zero. French vs. Greek is never significant for any model (largest gap +2.1 for Qwen3-30B, CI [−0.1, 4.1]).

Universal vs. per-language probes The universal probe is never significantly worse than the per-language probe across any of the 16 modellanguage combinations tested. It is significantly better for Gemma-4-31B in all three languages (+2.0 to +2.2), for Qwen3-4B, Qwen3-30B, and Llama-3.1-8B in both English and French (+0.9 to +2.3), and for Llama-Krikri-8B in Greek (+2.1, CI [0.8, 3.5]). The remaining comparisons do not reach significance, but every point estimate stays positive (largest such: +1.2 for French-Alpaca-8B in French, CI [−0.8, 3.5]). Pooling is thus at worst neutral and at best beneficial throughout.

## E.2 Significance testing for faithfulness claims

We test both faithfulness comparisons with a twoproportion z-test on agreement counts. For the pooled comparison, Qwen3-4B’s English-worse effect is significant $( p ~ < ~ 0 . 0 0 1 )$ , as is Llama-3.1-8B’s reversed, English-better effect $( p \_ { } <$ 0.01); Qwen3-30B, Gemma-4-31B, Krikri-8B, and Alpaca-8B trend in the same English-worse direction but do not reach significance (p between 0.07 and 0.49). For the aware-prompt, unsolvable-only comparison, the non-English advantage is highly significant for Krikri-8B in Greek $( p < 1 0 ^ { - 1 0 } )$ ,

Alpaca-8B in French $( p < 1 0 ^ { - 7 } ) .$ , and Qwen3-4B in both French $( p < 0 . 0 5 )$ and Greek $( p < 0 . 0 0 1 )$ ; Qwen3-30B, Gemma-4-31B, and Llama-3.1-8B show the same direction in both languages without reaching significance (p between 0.05 and 0.86).

## F Human Validation of the LLM Judge for Solvability Verdict Annotation

To validate the reliability of the LLM-as-a-judge used to annotate models’ verbal solvability verdicts (i.e., whether a reasoning trace ultimately treats a problem as solvable or unsolvable), one of the authors manually annotated a balanced subset of 100 reasoning traces according to the same criterion and prompt used by the judge. The subset was balanced with respect to language (English, French, and Greek), model, and solvability type (solvable vs. unsolvable). Specifically, each reasoning trace was labeled based on whether the model’s final verbal behavior was to attempt to solve the problem or to conclude that the problem is unsolvable. Table 5 reports the agreement between the human annotations and the LLM judge.

<table><tr><td>Annotator Pair</td><td>Agreement</td></tr><tr><td>LLM – Human</td><td>93.0</td></tr></table>

Table 5: Agreement (%) between the LLM judge and a human annotator on the validation subset of 100 reasoning traces.

## G Human Validation of the Translation

To validate the quality of the French and Greek translations of ReliableMath, two of the authors manually reviewed the translated problems, with each author annotating 100 problems for one target language (100 French and 100 Greek; 200 total). Each translation was compared against its original English source to assess whether the mathematical content was faithfully preserved and whether the translation was fluent and natural in the target language. Annotations were collected through a dedicated interface that displays the original English problem alongside its translation and allows the annotator to mark it as accurate or inaccurate, as shown in Figure 3. Table 6 reports the results of this review. These scores indicate that the translations are of high quality, with only a small fraction of problems flagged as inaccurate in either language.

<table><tr><td>Language</td><td>Overall agreement (%)</td></tr><tr><td>French</td><td>97.0</td></tr><tr><td>Greek</td><td>95.0</td></tr></table>

Table 6: Human validation agreement for the French and Greek translations on a subset of 100 problems per language.

## H Translation Example

Table 8 shows an example of a contradictory (unsolvable) problem alongside its Greek and French translations.

![](images/d51b56516d5300ce628337004b87832ce02cb762c5f44b2736f73fcdab19848e.jpg)  
Figure 3: Annotation interface used for translation validation. The original English problem and its translation are shown side by side, and the annotator marks whether the translation preserves the meaning and mathematica content of the source.

![](images/fe54bb114ffe201bba2d4fe7555e9e09846bb53ca64fb7c34fbeb58297e58aed.jpg)  
Table 7: Prompt used to translate mathematical problems between languages.

<table><tr><td>Language</td><td>Problem</td></tr><tr><td>English</td><td>Among the 1000 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.</td></tr><tr><td>Greek</td><td>Mετα των 1000 ατοων του Aiville, vπχον 195 που ατχον να δαχτυλδι ε διαντ, 367 πυ ατχν να σετ λφ, α 562 π ατχν να φτυρι π. ιλν, αθνα α τ 90 ατ ατχει α σλα ε αρλε σε σχα ι. πχν 437 - τι π τχν αι  α τ τα πατα,  234 τ π τχν αι τα α ατ τα πρατα. ρετε τν αι των ατων τ vi π ατχν α τα τσσερα απ αντá τα πργατα.</td></tr><tr><td>French</td><td>Parmi les 1000 résidents d’Aimeville, il y en a 195 qui possèdent une bague en diamant, 367 qui possèdent un ensemble de clubs de golf, et 562 qui possèdent une bêche de jardin. De plus, chacun des 900 résidents possède un sachet de bonbons en forme de cœur. Il y a 437 résidents qui possèdent exactement deux de ces objets, et 234 résidents qui possèdent exactement trois de ces objets. Trouver le nombre de résidents d’Aimeville qui possèdent les quatre objets.</td></tr></table>

Table 8: Example of an unsolvable (contradictory) problem and its translations into Greek and French. The contradiction is introduced by stating there are 1000 residents while also referring to “each of the 900 residents,” making the problem underdetermined.

<table><tr><td>Language</td><td>Prompt</td></tr><tr><td>English</td><td>Solve the following math problem step by step. Put your final answer within \boxed{}. Problem: {question}</td></tr><tr><td>Greek</td><td>Λσε τ παατω αατιó πληα α π α. Γρψε την τελικ σου απντηση μσα σε \bxd{}. ∏ρóβλημα: {question}</td></tr><tr><td>French</td><td>Résous le problème de mathématiques suivant étape par étape. Indique ta réponse finale dans \boxed{}. Problème : {question}</td></tr></table>

Table 9: Standard prompting templates used in our experiments.

<table><tr><td>Language</td><td>Prompt</td></tr><tr><td>English</td><td>Examine whether the following math problem is solvable with the information provided. If it is solvable, solve it step by step. Otherwise, state that it is unsolvable. Put your final answer in exactly one of these formats: - If the problem is solvable: \boxed{&lt;answer&gt;} - If the problem cannot be solved with the information provided: \boxed{unsolvable}</td></tr><tr><td>Greek</td><td>ξτασε αν τ παατω αηατι πληα ενα επιλσι ε τι πληφε πυ δívoνται. Aν εναι επιλσι, λσε τ βα π βα. ∆ιαφετι, δλωσε óτι εναι µη επιλúσιμο. ε την τελι συ απντηη σε αι μα απó αυτ τι φ: - Aν το πρóβληµα εíναι επιλσιµο: \bοxed{&lt;απντηση&gt;} - Aν τo πρóβληµα δεν μπορεí να λυθεí µε τι δoθεiσε πληροφορíε: \boxed{unsolvable}</td></tr><tr><td>French</td><td>Examine si le problème de mathématiques suivant est résoluble avec les informations fournies. S&#x27;il est résoluble, résous-le étape par étape. Sinon, indique qu&#x27;il est insoluble. Indique ta réponse finale dans exactement l&#x27;un de ces formats : - Si le problème est résoluble : \boxed{&lt;réponse&gt;} - Si le problème ne peut pas être résolu avec les informations fournies : \boxed{unsolvable}</td></tr></table>

Table 10: Unsolvable-aware prompting templates used in our experiments.

![](images/9c0fd0327caecf6ae90837f25f125dc6d86b7044f5ef632cbea049895738f37f.jpg)  
Table 11: Prompt used by the LLM judge to annotate whether a model ultimately treated a problem as solvable or unsolvable.