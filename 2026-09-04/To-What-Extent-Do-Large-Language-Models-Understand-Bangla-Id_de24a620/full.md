# To What Extent Do Large Language Models Understand Bangla Idioms?

Mousumi Akter<sup>1,2</sup>, Md. Faiyaz Abdullah Sayeedi<sup>3</sup>, Nurul Labib Sayeedi<sup>4</sup>, Swakkhar Shatabda<sup>3</sup> <sup>1</sup>Technical University Dortmund <sup>2</sup>Research Center Trustworthy Data Science and Security <sup>3</sup>BRAC University <sup>4</sup>United International University Correspondence: mousumi.akter@tu-dortmund.de

## Abstract

Idiomatic expressions are an integral part of natural language, reflecting cultural nuances and posing unique challenges for computational models, particularly in low-resource languages. In this paper, we present the first largescale benchmark dataset of Bangla idioms, complemented by a synthetic multiple-choice question (MCQ) dataset for idiom meaning identification. We conduct a comprehensive evaluation of recent large language models (LLMs) across three idiom-related tasks: paraphrasing, idiom span detection, and meaning identification, leveraging zero-shot and fewshot prompting strategies. Our results reveal substantial variability in model performance, with no single LLM consistently outperform ing others across all tasks. Notably, Phi-4- mini-instruct excels in paraphrasing, Kimi-K2- 32b-instruct in span detection, and Gemini-2.5- flash in meaning identification. We believe that our datasets and analyses will provide valuable resources to guide future research in improving LLM comprehension of idiomatic expressions, particularly in Bangla and other low-resource languages.<sup>1</sup>

## 1 Introduction

Idioms are multiword expressions whose meanings often diverge from their literal interpretation. Rooted in cultural and historical contexts, idioms are unique to specific linguistic communities. Bangla, the seventh most widely spoken language globally, is particularly rich in idiomatic expressions (an example is illustrated in Figure 1). However, the scarcity of annotated resources and the lack of a large-scale benchmark dataset for Bangla idioms significantly hinder progress in developing and evaluating robust NLP models (Joshi et al., 2020a; Chakraborty et al., 2021).

In this work, we present the first large-scale benchmark dataset of Bangla idioms, comprising

![](images/4341696344a36bc5bf136a00329177b06d5942e311c71f91c86b7edbb42fdae3.jpg)  
Figure 1: Example of the Bangla idiom “আঙুল ফুেল কলাগাছ” (Suddenly becoming rich or powerful), showing its literal translation vs. its true meaning.

10,822 entries with their corresponding meanings, of which 4,772 additionally include usage examples. We further construct a synthetic multiplechoice question (MCQ) dataset for idiom meaning detection, consisting of 10,913 samples with single correct answers and 38,688 samples with multiple correct answers.

We additionally investigate the performance of গাrecent large language models (LLMs) on three idiom-related tasks: paraphrasing, idiom span detection, and MCQ-based meaning detection. Our experiments cover several LLMs, including Kimi-K2-32B-Instruct (Bai et al., 2025), LLaMA-4- Scout-17B-Instruct (Meta, 2025), DeepSeek-R1- Distill-LLaMA-70B (Guo et al., 2025), GPT-OSS-20B, GPT-OSS-120B (OpenAI, 2025), Phi-4-Mini-Instruct (Abdin et al., 2024), Qwen3- 32B (Team, 2024), and Gemini-2.5-Flash (Comanici et al., 2025). The results show that Phi-4- mini-instruct excels in paraphrasing, Kimi-K2-32b-<sup>রা</sup> <sup>জ্ঞা</sup> <sup>ছে</sup> instruct in idiom span detection, and Gemini-2.5- flash in meaning identification.

In Summary:

• We introduce the first large-scale benchmark dataset of Bangla idioms, consisting of 10,822 entries with meanings and 4,772 usage examples.

• We construct a large-scale synthetic MCQ dataset for idiom meaning detection with both single-answer (10,913 samples) and multipleanswer (38,688 samples) settings.

• We provide a comprehensive evaluation ofrecent large language models on three idiomrelated tasks in Bangla, highlighting their strengths and limitations in low-resource idiomatic language understanding.

## 2 Related Works

Idiom Understanding in NLP. Idiomatic expressions, often represented as multiword expressions (MWEs), have been widely studied in NLP. Early work surveyed MWE processing (Constant et al., 2017) and released annotated corpora for Hindi and Marathi (Singh et al., 2016). Later resources include English (Zhou et al., 2021), multilingual Id10M (Tedeschi et al., 2022), and Chinese idiom datasets (Qiang et al., 2023). Idiomatic translation remains challenging (Dankers et al., 2022; Rezaeimanesh et al., 2025), and recent studies have analyzed idiomaticity in LLMs through sensitivity tests (Phelps et al., 2022; Klubička et al., 2023; Liu and Lareau, 2024; Yayavaram et al., 2024), hybrid identification (Alves et al., 2024), and commercial system evaluation (Phelps et al., 2024). Multilingual datasets exist for Indian languages (Agrawal et al., 2018) and other languages such as Catalan, Italian, and Russian (Sentsova et al., 2025). More recently, the PARSEME 2.0 Shared Task (Scholivet et al., 2026) has promoted multilingual idiom identification and paraphrasing across 14 languages.

Bangla NLP and Our Contribution. Bangla, spoken by over 300 million people, remains underrepresented in NLP (Joshi et al., 2020b), though recent datasets cover summarization (Hasan et al., 2021), QA (Bhattacharjee et al., 2022; Ekram et al., 2022), paraphrasing (Akil et al., 2022), back-transliteration (Fahim et al., 2024), and NER (Mahtab et al., 2025), etc. Recently, there has been small scale effort for figurative understanding of Bangla Idioms (Das et al., 2026; Sakhawat et al., 2026). To our knowledge, no prior work has systematically evaluated LLMs on Bangla idiomrelated tasks (i.e., paraphrasing, span detection and

MCQ meaning detection). We address this gap by introducing a large-scale Bangla idiom dataset and a synthetic MCQ corpus for meaning detection, and by benchmarking recent LLMs on paraphrasing, idiom span detection, and meaning identification, providing insights into their performance in a low-resource setting.

## 3 Bangla Idiom Dataset

We present a curated benchmark of Bangla idioms, manually collected from open-access and permissive sources under fair dealing provisions— including the National Curriculum and Text Book (NCTB) (বাংলা ভাষার বয্করন), Bangla Newspapers, Bangla Wikipedia<sup>2</sup>, and online idiom lexicons<sup>3</sup>. More details are in Appendix A.2. The dataset underwent systematic cleaning to remove duplicates and apply minimal normalization (e.g., spacing corrections), while retaining authentic spelling variants without transliteration. Native Banglaspeaking authors manually verified that all examples reflect genuine idiomatic usage with semantically coherent meanings. To ensure legal compliance and transparency, a full source-by-source licensing mapping is detailed in Appendix A.1. We will distribute the benchmark dataset under a CC BY-SA 4.0 license upon acceptance.

<table><tr><td>Field</td><td>Example</td></tr><tr><td>idiom</td><td>(moon of the new moon night)</td></tr><tr><td rowspan="3">meaning</td><td>可何招，T招，對可</td></tr><tr><td>(something seldom seen; a rare thing; seen very</td></tr><tr><td>rarely)</td></tr><tr><td rowspan="5">sentence</td><td>可G,</td></tr><tr><td>可</td></tr><tr><td>?(We hardly see you these</td></tr><tr><td>days, is getting a job make you</td></tr><tr><td>moon of the new moon night?)</td></tr></table>

Table 1: An example from the Idiom dataset

Each entry in the dataset comprises three fields: (a) idiom — the surface form of the Bangla idiom, (b) meaning — a set of Bangla glosses capturing both primary and secondary interpretations, and (c) sentence — a Bangla sentence illustrating the idiom in natural usage. The final dataset contains 10,822 unique idiom forms, of which 2,624 exhibit multiple meanings. In addition, 4,772 entries are accompanied by contextual sentences. Table 1 presents a representative example.

![](images/93144236cfb96e2180e0973a37db8781def24e499af36fa4ab251d4a74958c6a.jpg)  
Figure 2: Word-count length distributions

Building on 10,822 idiomatic examples, we construct a multiple-choice (MCQ) dataset for Bangla idiom meaning detection, where each instance requires selecting the correct meaning of a given idiom from four candidate options. Since the dataset provides each idiom paired with its ground-truth meaning(s), we generate distractor options by randomly sampling meanings from other idioms and shuffling them together with the correct option(s) to form balanced MCQ sets. This randomization ensures variability while preventing positional bias across options. As several idioms admit multiple valid interpretations, we further formulate a multi-answer setting in which more than one option may be correct for a given item. The resulting dataset consists of 10,913 single-answer instances and 38,688 multi-answer instances. Table 3 summarizes the key statistics and characteristics of the proposed Bangla idiom dataset. As shown in Figure 2, word-count distributions separate cleanly across tasks: answer meanings remain concise (2– 8 words), MCQ prompts cluster around formulaic templates (10–14 words), and usage sentences display the widest syntactic variation (8–35 words).

## 4 Experimental Setup

We define three tasks to evaluate the understanding of Bangla idioms by large language models (LLMs): (1) paraphrasing, (2) idiom span detection, and (3) multiple-choice (MCQ) meaning detection. The overarching goal is to assess the extent to which LLMs comprehend idiomatic expressions in Bangla—a low-resource language. We experimented with eight LLMs, including Kimi-K2-32B-Instruct (Bai et al., 2025), LLaMA-4-

Scout-17B-Instruct (Meta, 2025), DeepSeek-R1- Distill-LLaMA-70B (Guo et al., 2025), GPT-OSS-20B, GPT-OSS-120B (OpenAI, 2025), Phi-4-Mini-Instruct (Abdin et al., 2024), Qwen3- 32B (Team, 2024), and Gemini-2.5-Flash (Comanici et al., 2025), accessed through the Groq AI<sup>4</sup> API. The prompt templates used for each task are presented in Table 2. We evaluated the models on a subset of 1,000 samples from the dataset.

Paraphrasing For the paraphrasing task, we provided each model with a sentence containing an idiom and instructed it to generate an alternative version that preserves the original meaning. As ground truth, we constructed reference sentences by replacing the idiomatic span with its literal meaning. The model outputs were evaluated against these references using lexical overlap metrics (ROUGE) (Lin, 2004), semantic similarity metrics (BERTScore) (Zhang et al., 2020), and cosine similarity computed over embeddings from OpenAI, Gemini, and multilingual models such as LaBSE (Feng et al., 2022) and LASER (Artetxe and Schwenk, 2019).

Idiom Span Detection In the idiom span detection task, models were asked to identify the idiomatic portion within a given sentence. The dataset provides gold-standard idiom spans, which were used as ground truth. Performance was evaluated using n-gram overlap percentage and Levenshtein distance (Li and Liu, 2007) as similaritybased metrics.

MCQ Meaning Detection The MCQ meaning detection task involved two experimental setups: one where each question contained a single correct meaning, and another where multiple correct meanings were possible. This task was designed to assess whether LLMs can recognize all valid interpretations of idiomatic expressions. Evaluation was performed using accuracy as the metric.

## 5 Results and Analysis

## 5.1 Quantitative Analysis

Our first set of results pertains to the idiomatic dataset presented in Table 1, where large language models (LLMs) were evaluated under two prompting configurations: zero-shot and five-shot. We chose a 5-shot setup motivated by (Beauchemin et al., 2026), who showed that a 5-shot configuration serves as a highly stable proxy for evaluating an LLM’s multiword reasoning boundaries. To prevent in-context contamination, the few-shot examples were drawn exclusively from a held-out pool that is entirely disjoint from the test set used in all evaluations. As expected, overall performance improved under the five-shot setting, reflecting the advantage of in-context examples for this task. As detailed in Table 4 for the paraphrasing evaluation, Phi-4-mini-instruct achieved the highest lexical overlap scores under the five-shot setup (ROUGE-1 of 0.63 and ROUGE-2 of 0.50), indicating its strong capacity to generate paraphrases that closely approximate the reference texts.

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Prompt</td></tr><tr><td rowspan=1 colspan=1>Paraphrasing</td><td rowspan=1 colspan=1>Paraphrase the following idiomatic Bengali sentence while keeping the original meaning unchanged:&quot;,.Just output the paraphrased sentence andnothing more than that. No explanations needed at all.</td></tr><tr><td rowspan=1 colspan=1>Span-detection</td><td rowspan=1 colspan=1>Detect idiom from thisBengali sentence:&quot;?&quot;.Justoutput the extracted idiom and nothing more than that. No explanations needed at all.</td></tr><tr><td rowspan=1 colspan=1>mcq-single</td><td rowspan=1 colspan=1>Which one is the correct meaning of the Bengali idiom &quot;&#x27;?Find just one single option:&quot;(1) (2)(3) 可(4) .Just output thecorrect option number and nothing more than that.</td></tr><tr><td rowspan=1 colspan=1>mcq-multiple</td><td rowspan=1 colspan=1>Which are the correct meanings of the Bengali idiom &#x27; &#x27;? Output all the correct options:(1) -G(2) (3) 物(4).Just output the correct option numbers sepa-rated by comma and nothing more than that.</td></tr></table>

Table 2: Prompt designs for various tasks. Correct answers are underlined for clarity but not provided.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total idioms</td><td>10,822 4,772 (44.1%)</td></tr><tr><td>Entries with usage sentence Entries with multiple meaning</td><td>2,624</td></tr><tr><td>Mean meanings per entry</td><td>1.23</td></tr><tr><td>Max meanings per entry</td><td>16</td></tr><tr><td>MCQ with single answers</td><td>10,913</td></tr><tr><td>MCQ with multiple answers</td><td>38,688</td></tr></table>

Table 3: Dataset statistics

For semantic similarity–based metrics, performance varied across embedding configurations. This variation aligns with prior research (Mahajan et al., 2024) highlighting the limitations of surface sentence embeddings in fully capturing idiomatic equivalence. Notably, GPT-OSS-20b and Phi-4-mini-instruct delivered the strongest overall performance across metrics, while Gemini-2.5- Flash obtained the highest similarity score when OpenAI embeddings (text-emb-3-large) were applied (0.84). Considering both lexical overlap and embedding-based measures, Phi-4-mini-instruct achieves the highest overall average score (0.73), making it the most effective model for idiomatic sentence paraphrasing in Bangla.

Extending this analysis, we next examined LLM performance on the idiom span detection task under the same prompting conditions. Evaluation was based on two primary metrics: the percentage of unigram overlap within the identified idiomatic spans and the Levenshtein distance (LD) between predicted and reference spans. As summarized in Table 5, Kimi-K2-32b-instruct achieved the highest performance in the five-shot setting, yielding a unigram overlap of48.25% and the lowest Levenshtein distance of 6.27. Consistent with the paraphrasing task, most models benefited from in-context demonstrations, further indicating that LLMs leverage example-based prompting effectively for extracting idiomatic spans.

Finally, we assessed model performance on idiom meaning identification using our synthetic MCQ dataset. In this task, models were prompted to select the correct meaning for each idiomatic expression, evaluated under both single-meaning and multi-meaning setup conditions. The detailed results are shown in Table 6. Gemini-2.5-flash achieved the highest accuracy across both subtasks, scoring 0.76 on single-answer detection and 0.55 on multi-answer detection. These results suggest its strong capability in capturing fine-grained semantic interpretations and disambiguating idiomatic senses in low-resource settings.

Overall, our analysis highlights notable variability in LLM performance across idiom-related tasks. While Phi-4-mini-instruct demonstrates strong proficiency in paraphrasing idiomatic sentences, Kimi-K2-32b-instruct excels in idiom span detection, and Gemini-2.5-Flash performs best in meaning identification for Bangla. These results suggest that no single model uniformly excels across all subtasks, underscoring the multifaceted nature of idiomatic understanding in Bangla.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Fewshot Setting</td><td colspan="4"></td><td rowspan="2">text-emb gemini-emb -3-large</td><td colspan="3">Cosine Similarity</td><td rowspan="2">AVG</td></tr><tr><td>ROUGE-1</td><td>ROUGE-2</td><td>ROUGE-L</td><td>BERTScore</td><td>-001</td><td>LaBSE</td><td>LASER</td></tr><tr><td rowspan="2">Kimi-K2-32b-instruct</td><td>0-shot</td><td>0.38</td><td>0.09</td><td>0.38 0.37</td><td>0.76</td><td>0.57</td><td>0.81</td><td>0.73</td><td>0.80</td><td>0.57</td></tr><tr><td>5-shot</td><td>0.40</td><td>0.16</td><td></td><td>0.78</td><td>0.63</td><td>0.82</td><td>0.80</td><td>0.83</td><td>0.60</td></tr><tr><td rowspan="2">LLaMA-4-17b-instruct</td><td>0-shot</td><td>0.41</td><td>0.22</td><td>0.37</td><td>0.85</td><td>0.60</td><td>0.81</td><td>0.76</td><td>0.77</td><td>0.60</td></tr><tr><td>5-shot</td><td>0.47</td><td>0.25</td><td>0.39</td><td>0.80</td><td>0.62</td><td>0.87</td><td>0.79</td><td>0.78</td><td>0.62</td></tr><tr><td rowspan="2">DeepSeek-r1-70b</td><td>0-shot</td><td>0.47</td><td>0.17</td><td>0.44</td><td>0.74</td><td>0.70</td><td>0.81</td><td>0.70</td><td>0.71</td><td>0.59</td></tr><tr><td>5-shot</td><td>0.51</td><td>0.19</td><td>0.51</td><td>0.80</td><td>0.61</td><td>0.76</td><td>0.76</td><td>0.75</td><td>0.61</td></tr><tr><td rowspan="2">GPT-OSS-20b</td><td>0-shot</td><td>0.60</td><td>0.30</td><td>0.56</td><td>0.88</td><td>0.77</td><td>0.85</td><td>0.88</td><td>0.85</td><td>0.71</td></tr><tr><td>5-shot</td><td>0.52</td><td>0.40</td><td>0.53</td><td>0.88</td><td>0.75</td><td>0.92</td><td>0.88</td><td>0.86</td><td>0.72</td></tr><tr><td rowspan="2">GPT-OSS-120b</td><td>0-shot</td><td>0.54</td><td>0.18</td><td>0.43</td><td>0.83</td><td>0.72</td><td>0.88</td><td>0.84</td><td>0.81</td><td>0.65</td></tr><tr><td>5-shot</td><td>0.55</td><td>0.30</td><td>0.47</td><td>0.87</td><td>0.73</td><td>0.90</td><td>0.80</td><td>0.84</td><td>0.68</td></tr><tr><td rowspan="2">Phi-4-mini-instruct</td><td>0-shot</td><td>0.22</td><td>0.02</td><td>0.26</td><td>0.74</td><td>0.68</td><td>0.79</td><td>0.68</td><td>0.73</td><td>0.52</td></tr><tr><td>5-shot</td><td>0.63</td><td>0.50</td><td>0.56</td><td>0.83</td><td>0.70</td><td>0.90</td><td>0.82</td><td>0.89</td><td>0.73</td></tr><tr><td rowspan="2">Qwen3-32b</td><td>0-shot</td><td>0.55</td><td>0.24</td><td>0.46</td><td>0.74</td><td>0.75</td><td>0.80</td><td>0.69</td><td>0.68</td><td>0.61</td></tr><tr><td>5-shot</td><td>0.58</td><td>0.31</td><td>0.55</td><td>0.76</td><td>0.76</td><td>0.74</td><td>0.80</td><td>0.76</td><td>0.66</td></tr><tr><td rowspan="2">Gemini-2.5-flash</td><td>0-shot</td><td>0.41</td><td>0.16</td><td>0.40</td><td>0.77</td><td>0.80</td><td>0.80</td><td>0.73</td><td>0.77</td><td>0.61</td></tr><tr><td>5-shot</td><td>0.57</td><td>0.26</td><td>0.54</td><td>0.80</td><td>0.84</td><td>0.84</td><td>0.84</td><td>0.87</td><td>0.70</td></tr></table>

Table 4: Paraphrasing task results across various LLMs under 0-shot and 5-shot settings. The highest score for each metric is highlighted in bold.

<table><tr><td>Model</td><td>Fewshot Setting</td><td>% Unigram Overlap</td><td>Levenshtein Distance (LD)</td></tr><tr><td rowspan="2">Kimi-K2-32b-instruct</td><td>0-shot</td><td>37.66</td><td>8.21</td></tr><tr><td>5-shot</td><td>48.25</td><td>6.27</td></tr><tr><td rowspan="2">LLaMA-4-17b-instruct</td><td>0-shot</td><td>32.12</td><td>13.34</td></tr><tr><td>5-shot</td><td>47.61</td><td>13.78</td></tr><tr><td rowspan="2">DeepSeek-r1-70b</td><td>0-shot</td><td>26.54</td><td>10.47</td></tr><tr><td>5-shot</td><td>38.56</td><td>8.36</td></tr><tr><td rowspan="2">GPT-OSS-20b</td><td>0-shot</td><td>34.23</td><td>11.36</td></tr><tr><td>5-shot</td><td>37.04</td><td>8.54</td></tr><tr><td rowspan="2">GPT-OSS-120b</td><td>0-shot</td><td>36.07</td><td>9.62</td></tr><tr><td>5-shot</td><td>44.20</td><td>8.25</td></tr><tr><td rowspan="2">Phi-4-mini-instruct</td><td>0-shot</td><td>11.00</td><td>9.30</td></tr><tr><td>5-shot</td><td>21.08</td><td>9.93</td></tr><tr><td rowspan="2">Qwen3-32b</td><td>0-shot</td><td>39.39</td><td>7.55</td></tr><tr><td>5-shot</td><td>40.97</td><td>6.64</td></tr><tr><td rowspan="2">Gemini-2.5-flash</td><td>0-shot</td><td>35.23</td><td>9.90</td></tr><tr><td>5-shot</td><td>42.68</td><td>7.91</td></tr></table>

Table 5: Results across various LLMs for the idiomatic span detection task under 0-shot and 5-shot settings. The best scores per metric are highlighted in bold.

Finding: Idiomatic understanding in Bangla is a multifaceted challenge. LLM performance on Bangla idiom tasks varies substantially across subtasks, with different models excelling in different areas, indicating that no single model consistently performs best across paraphrasing, span detection, and meaning identification.

## 5.2 Human Evaluation

To rigorously evaluate the contextual nuance and semantic preservation of Bangla idioms, we perform a manual human evaluation focused exclusively on the paraphrasing task. Automated evaluation metrics are often insufficient to capture the linguistic, cultural, and idiomatic subtleties inherent in Bangla. Accordingly, three highly qualified annotators (native speakers as well) assess a randomly sampled subset of 25 idiomatic expressions per model across eight models, under both zeroshot and five-shot settings (400 samples in total). The outputs are rated on a continuous scale from 0 to 2, where 0 indicates complete loss of idiomatic meaning or nonsensical output, 1 denotes a partially correct or overly literal paraphrase, and 2 corresponds to a fully faithful and natural-sounding

<table><tr><td>Model</td><td>Single Detection</td><td>Multiple Detection</td></tr><tr><td rowspan="2">Kimi-K2-32b-instruct LLaMA-4-17b-instruct</td><td>0.71</td><td>0.52</td></tr><tr><td>0.74</td><td>0.48</td></tr><tr><td rowspan="2">DeepSeek-r1-70b</td><td>0.19</td><td>0.08</td></tr><tr><td>0.60</td><td>0.23</td></tr><tr><td>GPT-OSS-20b GPT-OSS-120b</td><td>0.63</td><td>0.24</td></tr><tr><td>Phi-4-mini-instruct</td><td>0.29</td><td>0.26</td></tr><tr><td>Qwen3-32b</td><td>0.49</td><td>0.28</td></tr><tr><td>Gemini-2.5-flash</td><td>0.76</td><td>0.55</td></tr></table>

Table 6: Accuracy across various LLMs for the MCQ meaning detection task.

<table><tr><td>Model</td><td>A1 vs A2</td><td>A2 vs A3</td><td>A1 vs A3</td><td>Score</td></tr><tr><td>Gemini-2.5-flash</td><td>0.644</td><td>0.614</td><td>0.668</td><td>0.642</td></tr><tr><td>DeepSeek-r1-70b</td><td>0.608</td><td>0.635</td><td>0.636</td><td>0.626</td></tr><tr><td>LLaMA-4-17b-instruct</td><td>0.667</td><td>0.651</td><td>0.563</td><td>0.627</td></tr><tr><td>Kimi-K2-instruct</td><td>0.585</td><td>0.687</td><td>0.601</td><td>0.624</td></tr><tr><td>GPT-OSS-20b</td><td>0.589</td><td>0.783</td><td>0.501</td><td>0.624</td></tr><tr><td>GPT-OSS-120b</td><td>0.645</td><td>0.620</td><td>0.648</td><td>0.638</td></tr><tr><td>Qwen3-32b</td><td>0.631</td><td>0.754</td><td>0.824</td><td>0.737</td></tr><tr><td>Phi-4-mini-instruct</td><td>0.633</td><td>0.755</td><td>0.639</td><td>0.676</td></tr><tr><td>Average</td><td>0.625</td><td>0.687</td><td>0.635</td><td>0.65</td></tr></table>

Table 7: Inter-annotator Agreement (Cohen’s Kappa) across models.

![](images/81036cce145204d5cd174c80ad7fd6655df1373fa14a7db477e4d08ec969b5c0.jpg)  
Figure 3: Pearson Correlation Matrix illustrating the linear relationship between automated evaluation metrics and human annotation scores.

paraphrase.

## 5.2.1 Inter-annotator Agreement

To ensure the validity of our manual grading process, we calculated the inter-annotator agreement using pairwise weighted Cohen’s Kappa scores across all three annotators (A1 vs. A2, A2 vs. A3, and A1 vs. A3). The annotations exhibited substantial agreement across the board. The total overall average Kappa score across all models and annotators stood at a robust 0.65.

Highest Agreement. The evaluators showed the highest consensus when grading the outputs of qwen3-32b, achieving an overall Kappa score of 0.737. This indicates that the outputs generated by this model were straightforward and unambiguous to grade.

Lowest Agreement. Conversely, kimi-k2- instruct and gpt-oss-20b tied for the lowest annotator agreement at 0.624, suggesting their paraphrased variations were slightly more ambiguous or prone to subjective interpretation by the native speakers.

## 5.2.2 Human Assessment Scores

The raw human annotation scores reveal distinct differences in model capabilities regarding Bengali idiom comprehension. We averaged the hu-

man scores for each model out of the maximum possible score of 2.0.
<table><tr><td>Model</td><td>0-Shot</td><td>5-Shot</td><td>Average</td></tr><tr><td>Gemini-2.5-flash</td><td>1.417</td><td>1.333</td><td>1.375</td></tr><tr><td>DeepSeek-r1-70b</td><td>0.625</td><td>0.736</td><td>0.681</td></tr><tr><td>LLaMA-4-17b-instruct</td><td>0.917</td><td>0.917</td><td>0.917</td></tr><tr><td>Kimi-K2-32b-instruct</td><td>1.542</td><td>1.333</td><td>1.438</td></tr><tr><td>GPT-OSS-20b</td><td>0.875</td><td>0.889</td><td>0.882</td></tr><tr><td>GPT-OSS-120b</td><td>0.736</td><td>0.903</td><td>0.819</td></tr><tr><td>Qwen3-32b</td><td>0.458</td><td>0.708</td><td>0.583</td></tr><tr><td>Phi-4-mini-instruct</td><td>1.056</td><td>1.236</td><td>1.146</td></tr></table>

Table 8: Average human annotation scores across different models by different Prompt Settings.

Top Performers. The kimi-k2-instruct model achieved the highest overall average score of 1.438, closely followed by gemini-2.5-flash at 1.375. Interestingly, both of these leading models peaked during the zero-shot prompting phase (scoring 1.542 and 1.417, respectively) and slightly regressed when provided with five-shot examples (1.333 for both).

Few-Shot Beneficiaries. Conversely, several smaller or distilled models benefited heavily from the in-context learning provided by the five-shot prompts. phi-4-mini-instruct improved from 1.056 (zero-shot) to 1.236 (five-shot), and deepseek-r1-distill-llama-70b jumped from 0.625 to 0.736.

Lowest Performers. The qwen3-32b model struggled the most to capture the nuanced meaning of the idioms, returning the lowest overall average score of 0.583.

Finding: Model performance varies across prompting settings, with leading models performing best in zero-shot, smaller models benefiting from few-shot examples, and Qwen3- 32B showing the weakest overall performance.

## 5.2.3 Validating Automatic Metrics against Human Judgement

To determine which automated metrics are the most reliable proxies for true human evaluation in Bengali paraphrase tasks, we computed the Pearson Correlation Coefficient (r) between the automated scores and the segment-level human annotations.

Our correlation analysis highlights a significant disparity between traditional surface-level metrics and modern embedding-based evaluators: Strongest Correlations. Embedding-based metrics demonstrated the highest linear alignment with human judgment. BERTScore Precision (BERT P) yielded the strongest overall correlation (r = 0.80), followed by the Gemini Embedding similarity (r = 0.74) and BERTScore Recall (BERT R) (r = 0.72). LaBSE also proved to be a highly reliable multilingual metric, achieving a Pearson correlation of r = 0.72.

Weakest Correlations. Traditional n-gram overlap metrics proved highly ineffective at capturing the semantic preservation of idioms. ROUGE-1 (r = 0.15) and ROUGE-3 (r = 0.21) showed virtually no meaningful correlation with human preference, emphasizing that surface-level lexical matching fails to reward models that accurately paraphrase the meaning of an idiom using entirely different vocabulary.

Finding: The evaluation of Bengali idiomatic paraphrasing should prioritize embeddingbased semantic metrics (e.g., BERTScore, LaBSE) over lexical metrics (e.g., ROUGE), as they better align with human judgments.

6 Error Analysis and Limitations of Open-Source LLMs for Bangla Idiom Understanding

## 6.1 Failures in Bangla Pre-trained LLMs

We initially evaluated several Bangla pre-trained large language models (LLMs), including TituLLM-3B (Nahin et al., 2025), BanglaLLaMA3- 8B (Zehady, 2025), and OdiaGenAI-Bengali-LLaMA7B (Parida et al., 2023). However, these models consistently failed to produce coherent or contextually relevant responses even for simple conversational prompts. For example, when prompted with “আিম িক দয়া কের জানেত পাির আপিন েক বলেছন?” (May I please know who are you?), one model generated unrelated and repetitive text such as “আিম একটা েছাট্ েদাকােন ঢুকলাম। েসখােন একটা সুন্দর সাদা রেঙর েটিবল িছল।...” (I entered a small shop. There was a beautiful white table there...). Notably, this failure occurred outside the idiom understanding task itself and instead reflected a broader inability to maintain semantic coherence in general Bangla text generation. Due to such persistent irrelevance and instability in outputs, we excluded these Bangla-specific pre-trained models from the final benchmarking suite.

Finding: Existing Bangla pre-trained LLMs struggle to maintain semantic coherence even in simple conversational settings, limiting their applicability to downstream idiom understanding tasks.

## 6.2 Idiom Span Detection: Zero-Shot vs. Few-Shot Prompting

Few-shot prompting is commonly expected to improve model performance by providing structured demonstrations. However, in our Bangla idiom span detection experiments, we observed that additional examples occasionally introduced hallucinations and degraded performance. This behavior was particularly evident in open-source models such as Qwen3-32B.

For instance, given the sentence “‘ভৰ্ূিবলাস েশেখ নাই কারা েসই নারী’ - রবীন্দৰনাথ ঠাকুর,” (Who is the woman who has not learned the play of eyebrows? – Rabindranath Tagore), the zero-shot setup extracted “ভৰ্ূিবলাস েশেখ নাই” (has not learned the play of eyebrows), which overextended the span but still included the correct idiom “ভৰ্ূিবলাস” (play of eyebrows). In contrast, the 5-shot prompting setup identified only “েশেখ নাই” (has not learned), omitting the core idiomatic expression entirely. Although the zero-shot prediction remained imperfect, it was substantially closer to the ground truth than the few-shot variant. This suggests that, in certain settings, few-shot demonstrations may inadvertently confuse models toward structurally similar yet semantically irrelevant phrases.

Finding: Few-shot prompting can sometimes degrade Bangla idiom span detection performance by introducing hallucinated or semantically irrelevant spans.

## 6.3 Over-Extraction in Idiom Span Detection

A recurring limitation across open-source LLMs in the span-based Bangla idiom detection task is the tendency to extract overly long spans that include non-idiomatic surrounding words. This over-extraction issue becomes especially prominent when the idiom appears within a broader noun phrase or figurative construction.

For example, in the sentence “রাজনীিতর ময়দােন শুধু েঘােড়লেদর েঘারােফরা” (In the field of politics, only the clever ones roam around), the correct idiomatic expression is “েঘােড়ল” (clever). However, several open-source models, including Qwen3- 32B, Meta LLaMA-4-17B, DeepSeek-R1-Distill-LLaMA-70B, and GPT-OSS-20B, incorrectly extracted the extended span “েঘােড়লেদর েঘারােফরা” (the roaming of the clever ones).

In contrast, the closed-source model Gemini-2.5-Flash successfully identified the precise idiomatic boundaries in such cases, demonstrating stronger fine-grained semantic understanding. These findings indicate that although open-source LLMs show promising capabilities, they still lag behind proprietary models in accurately capturing idiomatic spans and figurative language phenomena in low-resource languages such as Bangla.

Finding: Open-source LLMs frequently overextract Bangla idiomatic spans, whereas proprietary models demonstrate stronger boundarylevel semantic understanding.

## 6.4 Difficulty in Multi-Label MCQ Reasoning

The multiple-choice question (MCQ) task in our benchmark requires models to identify all valid meanings of a Bangla idiom from four candidate options. Unlike standard single-answer MCQs, this task involves multi-label semantic reasoning, where multiple options may simultaneously be correct, and models are expected to output only the indices of the correct choices in a comma-separated format.

For example, given the idiom “অকমর্ার ধািড়” (a useless person), the candidate meanings are: (1) িববাহকােল শুভদৃিষ্ট (The auspicious exchange of glances during the wedding), (2) ধানকােঠর মই ( a useless person), (3) েখাদার খািস (A carefree person), and (4) অগাকান্ত (ignorant person). The correct answers are options 2, 3, and 4. However, none ofthe evaluated open-source models successfully identified the complete set of correct answers, with option (2) ধানকােঠর মই, being most frequently omitted where the literal translation is paddy straw ladder. By contrast, Gemini-2.5-Flash correctly selected all valid options while also adhering to the required output format. This highlights a substantial gap between open-source and proprietary models in handling nuanced multi-label semantic reasoning tasks in Bangla, particularly when meanings involve culturally grounded interpretations.

Finding: Open-source LLMs struggle with multi-label semantic reasoning in Bangla idiom MCQs, especially when multiple culturally grounded meanings must be identified simultaneously.

## 7 Conclusion

In this paper, we introduced the first large-scale benchmark dataset of Bangla idioms, along with an additional synthetic multiple-choice question (MCQ) dataset designed for idiom meaning detection. We conducted a series of experiments using recent LLMs across multiple idiom-related tasks, including paraphrasing, idiom span detection, and meaning identification, employing various prompting strategies. Our findings reveal notable variability in LLM performance, indicating that no single model performs optimally across all tasks. We believe our dataset and analyses provide a foundation for advancing idiomatic understanding in LLMs, particularly for low-resource languages like Bangla. Further linguistic analyses enabled by the dataset, such as translation challenges and vocabulary complexity, are beyond the scope of this work and left for future research.

## Limitations

While the dataset is broad and carefully curated, some idioms may have been missed unintentionally, and future work can further expand the dataset. Due to computational constraints, experiments were conducted on a subset of 1,000 samples per setting, which may reduce evaluation granularity but still captures overall performance trends. Additionally, only 4,772 out of 10,822 idiomatic expressions (44.1%) are accompanied by contextual usage sentences, limiting their applicability for context-dependent tasks. However, these sentences were directly collected from original sources and represent human-written examples, while missing contexts were unavailable in the source materials. The synthetic MCQ benchmark also relies on randomly sampled meanings as distractors, which may allow models to solve questions through elimination; future work can explore more challenging distractor construction strategies. Finally, budget constraints limited the inclusion of some commercially available or high-cost LLMs in our evaluation. These limitations do not diminish our central contribution: a large-scale Bangla idiom dataset and a comprehensive evaluation of LLM capabilities on idiom-related tasks, including paraphrasing, span identification and meaning detection.

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. 2024. Phi-4 technical report. arXiv preprint arXiv:2412.08905.

Ruchit Agrawal, Vighnesh Chenthil Kumar, Vignesh waran Muralidaran, and Dipti Misra Sharma. 2018. No more beating about the bush : A step towards idiom handling for indian language NLP. In LREC. European Language Resources Association (ELRA).

Ajwad Akil, Najrin Sultana, Abhik Bhattacharjee, and Rifat Shahriyar. 2022. Banglaparaphrase: A highquality bangla paraphrase dataset. In AACL/IJCNLP (2), pages 261–272. Association for Computational Linguistics.

Diego Alves, Stefan Fischer, Stefania Degaetano-Ortlieb, and Elke Teich. 2024. Multi-word expressions in English scientific writing. In Proceedings of the 8th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2024), pages 67–76, St. Julians, Malta. Association for Computational Linguistics.

Mikel Artetxe and Holger Schwenk. 2019. Massively multilingual sentence embeddings for zeroshot cross-lingual transfer and beyond. Trans. Assoc. Comput. Linguistics, 7:597–610.

Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, and Haiming Wang. 2025. Kimi K2: open agentic intelligence. CoRR, abs/2507.20534.

David Beauchemin, Yan Tremblay, Mohamed Amine Youssef, and Richard Khoury. 2026. Idiom understanding as a tool to measure the dialect gap. In Findings of the Association for Computational Linguistics: ACL 2026, pages 505–522.

Abhik Bhattacharjee, Tahmid Hasan, Wasi Uddin Ahmad, Kazi Samin Mubasshir, Md Saiful Islam, Anindya Iqbal, M. Sohel Rahman, and Rifat Shahriyar. 2022. Banglabert: Language model pretraining and benchmarks for low-resource language understanding evaluation in bangla. In NAACL-HLT (Findings), pages 1318–1327. Association for Computational Linguistics.

Susmoy Chakraborty, Mir Tafseer Nayeem, and Wasi Uddin Ahmad. 2021. Simple or complex? learning to predict readability of bengali texts. Proceedings ofthe AAAI Conference on Artificial Intelligence, 35(14):12621–12629.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Mathieu Constant, Gülşen Eryigit, Johanna Monti, Lonneke van der Plas, Carlos Ramisch, Michael Rosner,

and Amalia Todirascu. 2017. Survey: Multiword expression processing: A Survey. Computational Linguistics, 43(4):837–892.

Verna Dankers, Christopher Lucas, and Ivan Titov. 2022. Can transformer be too compositional? analysing idiom processing in neural machine translation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3608–3626, Dublin, Ireland. Association for Computational Linguistics.

Sarmistha Das, Shreyas Guha, Suvrayan Bandyopadhyay, Salisa Phosit, Kitsuchart Pasupa, and Sriparna Saha. 2026. When meaning isn’t literal: Exploring idiomatic meaning across languages and modalities. arXiv preprint arXiv:2604.10787.

Syed Mohammed Sartaj Ekram, Adham Arik Rahman, Md. Sajid Altaf, Mohammed Saidul Islam, Mehrab Mustafy Rahman, Md Mezbaur Rahman, Md Azam Hossain, and Abu Raihan Mostofa Kamal. 2022. BanglaRQA: A benchmark dataset for underresourced Bangla language reading comprehensionbased question answering with diverse questionanswer types. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 2518–2532, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Md Fahim, Fariha Tanjim Shifat, Fabiha Haider, Deeparghya Dutta Barua, MD Sakib Ul Rahman Sourove, Md Farhan Ishmam, and Md Farhad Alam Bhuiyan. 2024. BanglaTLit: A benchmark dataset for back-transliteration of Romanized Bangla. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14656–14672, Miami, Florida, USA. Association for Computational Linguistics.

Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Languageagnostic BERT sentence embedding. In ACL (1), pages 878–891. Association for Computational Linguistics.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Tahmid Hasan, Abhik Bhattacharjee, Md. Saiful Islam, Kazi Samin Mubasshir, Yuan-Fang Li, Yong-Bin Kang, M. Sohel Rahman, and Rifat Shahriyar. 2021. Xl-sum: Large-scale multilingual abstractive summarization for 44 languages. In ACL/IJCNLP (Findings), volume ACL/IJCNLP 2021 of Findings ofACL, pages 4693–4703. Association for Computational Linguistics.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020a. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics,

pages 6282–6293, Online. Association for Computational Linguistics.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020b. The state and fate of linguistic diversity and inclusion in the NLP world. In ACL, pages 6282–6293. Association for Computational Linguistics.

Filip Klubička, Vasudevan Nedumpozhimana, and John Kelleher. 2023. Idioms, probing and dangerous things: Towards structural probing for idiomaticity in vector space. In Proceedings of the 19th Workshop on Multiword Expressions (MWE 2023), pages 45–57, Dubrovnik, Croatia. Association for Computational Linguistics.

Yujian Li and Bi Liu. 2007. A normalized levenshtein distance metric. IEEE Trans. Pattern Anal. Mach. Intell., 29(6):1091–1095.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Li Liu and Francois Lareau. 2024. Assessing BERT’s sensitivity to idiomaticity. In Proceedings of the Joint Workshop on Multiword Expressions and Universal Dependencies (MWE-UD) @ LREC-COLING 2024, pages 14–23, Torino, Italia. ELRA and ICCL.

Yash Mahajan, Naman Bansal, Eduardo Blanco, and Santu Karmaker. 2024. ALIGN-SIM: A task-free test bed for evaluating and interpreting sentence embeddings through semantic similarity alignment. In EMNLP (Findings), pages 7393–7428. Association for Computational Linguistics.

Md. Motahar Mahtab, Faisal Ahamed Khan, Md. Ekramul Islam, Md. Shahad Mahmud Chowdhury, Labib Imam Chowdhury, Sadia Afrin, Hazrat Ali, Mohammad Mamun Or Rashid, Nabeel Mohammed, and Mohammad Ruhul Amin. 2025. BanNERD: A benchmark dataset and context-driven approach for Bangla named entity recognition. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 6807–6828, Albuquerque, New Mexico. Association for Computational Linguistics.

Meta. 2025. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai.meta.com/blog/ llama-4-multimodal-intelligence/. Accessed: 2025-09-07.

Shahriar Kabir Nahin, Rabindra Nath Nandi, Sagor Sarker, Quazi Sarwar Muhtaseem, Md Kowsher, Apu Chandraw Shill, Md Ibrahim, Mehadi Hasan Menon, Tareq Al Muntasir, and Firoj Alam. 2025. Titullms: A family of bangla llms with comprehensive benchmarking. arXiv preprint arXiv:2502.11187.

OpenAI. 2025. Introducing gpt-oss. https:// openai.com/index/introducing-gpt-oss/. Accessed: 2025-09-28.

Shantipriya Parida, Sambit Sekhar, Guneet Singh Kohli, Arghyadeep Sen, and Shashikanta Sahoo. 2023. Bengali instruction-tuning model. https: //huggingface.co/OdiaGenAI.

Dylan Phelps, Xuan-Rui Fan, Edward Gow-Smith, Harish Tayyar Madabushi, Carolina Scarton, and Aline Villavicencio. 2022. Sample efficient approaches for idiomaticity detection. In Proceedings of the 18th Workshop on Multiword Expressions @LREC2022, pages 105–111, Marseille, France. European Language Resources Association.

Dylan Phelps, Thomas M. R. Pickard, Maggie Mi, Edward Gow-Smith, and Aline Villavicencio. 2024. Sign of the times: Evaluating the use of large language models for idiomaticity detection. In Proceedings of the Joint Workshop on Multiword Expressions and Universal Dependencies (MWE-UD) @ LREC-COLING 2024, pages 178–187, Torino, Italia. ELRA and ICCL.

Jipeng Qiang, Yang Li, Chaowei Zhang, Yun Li, Yi Zhu, Yunhao Yuan, and Xindong Wu. 2023. Chinese idiom paraphrasing. Trans. Assoc. Comput. Linguistics, 11:740–754.

Sara Rezaeimanesh, Faezeh Hosseini, and Yadollah Yaghoobzadeh. 2025. Large language models for Persian-English idiom translation. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7974–7985, Albuquerque, New Mexico. Association for Computational Linguistics.

Adib Sakhawat, Shamim Ara Parveen, Md Ruhul Amin, Shamim Al Mahmud, Md Saiful Islam, and Tahera Khatun. 2026. When words don’t mean what they say: Figurative understanding in bengali idioms. arXiv preprint arXiv:2602.12921.

Manon Scholivet, Agata Savary, Carlos Ramisch, Eric Bilinski, Takuya Nakamura, Maria Mitrofan, and Vasile Pais. 2026. Edition 2.0 of the PARSEME shared task on multilingual identification and paraphrasing of multiword expressions. In Proceedings of the 22nd Workshop on Multiword Expressions (MWE 2026), pages 254–275, Rabat, Marocco. Association for Computational Linguistics.

Uliana Sentsova, Debora Ciminari, JosefVan Genabith, and Cristina España-Bonet. 2025. MultiCoPIE: A multilingual corpus of potentially idiomatic expressions for cross-lingual PIE disambiguation. In Proceedings ofthe 21st Workshop on Multiword Expressions (MWE 2025), pages 67–81, Albuquerque, New Mexico, U.S.A. Association for Computational Linguistics.

Dhirendra Singh, Sudha Bhingardive, and Pushpak Bhattacharyya. 2016. Multiword expressions dataset for Indian languages. In Proceedings of

the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 2331– 2335, Portorož, Slovenia. European Language Resources Association (ELRA).

Qwen Team. 2024. Qwen2.5: A party of foundation models. https://qwen.ai/research.

Simone Tedeschi, Federico Martelli, and Roberto Navigli. 2022. ID10M: Idiom identification in 10 languages. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 2715– 2726, Seattle, United States. Association for Computational Linguistics.

Arnav Yayavaram, Siddharth Yayavaram, Prajna Devi Upadhyay, and Apurba Das. 2024. BERT-based idiom identification using language translation and word cohesion. In Proceedings of the Joint Workshop on Multiword Expressions and Universal Dependencies (MWE-UD) @ LREC-COLING 2024, pages 220–230, Torino, Italia. ELRA and ICCL.

Abdullah Khan Zehady. 2025. Banglallama-3-8b-bnwiki-instruct. https: //huggingface.co/BanglaLLM/ BanglaLLama-3-8b-BnWiki-Instruct. Hugging Face model repository.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with BERT. In ICLR. Open-Review.net.

Jianing Zhou, Hongyu Gong, and Suma Bhat. 2021. PIE: A parallel idiomatic expression corpus for idiomatic sentence generation and paraphrasing. In Proceedings of the 17th Workshop on Multiword Expressions (MWE 2021), pages 33–48, Online. Association for Computational Linguistics.

## A Appendix

## A.1 Data Licensing and Source Redistribution Permissions

All text sources used to construct the Bangla idiom benchmark are distributed under open, public, or Creative Commons frameworks, permitting their use, adaptation, and redistribution for educational and non-commercial research purposes. The extracted source annotations, vocabulary phrases, and contextual sentences fall under permissive reuse guidelines for academic evaluation tasks. A comprehensive breakdown of the licensing framework for each source category is summarized in Table 9. Specifically, our data collection pipeline draws from four primary categories:

<table><tr><td>Data Source Category</td><td>Primary Content Ori- gin</td><td>Applicable Distribution License / Term</td><td>Permissible Academic Use</td></tr><tr><td>Online Encyclopedias</td><td>Bangla Wikipedia</td><td>Creative Commons Attribution- ShareAlike (CC BY-SA 4.0)</td><td>Copying, remixing, and redistribu- tion permitted with attribution un- der the same license.</td></tr><tr><td>Online Lexicons</td><td>Public Reference Dictio- naries</td><td>Open Educational Resources (OER) Terms / Public Domain</td><td>Permitted for vocabulary mapping, non-commercial research, and text parsing evaluation.</td></tr><tr><td>National Curriculum</td><td>NCTB Grammar Text- books</td><td>Public Distribution (Free Govern- ment Textbooks)</td><td>Openly accessible and allowed for non-commercial educational analy- sis and replication.</td></tr><tr><td>Print Media</td><td>Bangla Newspapers</td><td>Fair Use / Fair Dealing (Section 52 of Bangladesh Copyright Act)</td><td>Permitted for non-commercial lin- guistic evaluation, syntactic parsing, and text benchmarking.</td></tr></table>

Table 9: Source licensing and distribution permission mapping for the benchmark dataset.

Online Encyclopedias: Entries sourced from Bangla Wikipedia are licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license, which permits copying, remixing, and redistribution under the same terms with proper attribution.

Online Lexicons: Vocabulary mappings and reference definitions sourced from public reference dictionaries fall under Open Educational Resources (OER) terms and public domain guidelines, allowing non-commercial research and computational parsing.

National Curriculum: Idiomatic examples and grammar rules derived from National Curriculum and Textbook Board (NCTB) grammar textbooks are freely distributed government educational materials, permitted for open academic analysis and reproducibility.

Print Media: Natural usage sentences extracted from Bangla newspapers are utilized under Fair Use and Fair Dealing provisions (e.g., Section 52 of the Bangladesh Copyright Act), which explicitly allow the reproduction of published works for non-commercial educational instruction, linguistic evaluation, and criticism.

## A.2 Data Collection Strategy

Our data collection strategy followed a two-stage digitization and curation framework:

Digital/Online Sourcing: For natively digital platforms like Bangla Wikipedia and established online lexicons, data was manually collated to build the core electronic lexicon.

Print/Offline Sourcing: For physical text resources, specifically the National Curriculum and Textbook Board (NCTB) grammar books (বাংলা ভাষার বয্াকরণ) and contemporary print newspapers, we captured high-resolution source images. These images were processed through an Optical Character Recognition (OCR) pipeline optimized for the Bengali script to convert print text into editable strings.

Following extraction, the authors (Bangla native speakers) thoroughly audited all entries to fix spelling inconsistencies and filter out duplicates. Human annotators also identified idioms with multiple senses, which are stored as a single list under the corresponding idiom entry. To maximize usability for future benchmarking, all verified artifacts were structured into a standardized JSON schema.