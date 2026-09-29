# Pass or Fail? Evaluating LLMs on Two Greek Examination Benchmarks

Panagiota Kyriazi, Eleni Kasoura, Prokopis Prokopidis Institute for Language and Speech Processing / Athena RC {p.kyriazi, eleni.kasoura, prokopis}@athenarc.gr

## Abstract

The rapid advancement of Large Language Models (LLMs) imposes a thorough evaluation of their linguistic and analytical capabilities as well as constraints, particularly for a lan guage with limited benchmark coverage such as Greek. To address the limited availability of comprehensive benchmarks in this domain, we introduce Prot-Ex and Pan-Ex, two bench marks consisting of questions from entrance exams for Greek Model and Experimental schools as well as the Panhellenic exams (the Greek national university entrance examinations). These benchmarks are employed to assess the performance of text-only LLMs—including the Greek-adapted KriKri-8B-Instruct, Llama-3.1- 8B, Gemma-4-26B, and Qwen-3-32B—across diverse academic disciplines (Modern Greek, Mathematics, Physics, etc.) and task formats (closed, structured, and open-ended), including textualized visual context (i.e., image descriptions). Our findings indicate the localized KriKri-8B significantly outperforms its base model, successfully rivalling much larger LLMs in linguistically demanding humani ties tasks. By leveraging an LLM-as-a-Judge methodology, we expose the inadequacy of traditional lexical metrics for evaluating complex reasoning. Crucially, we uncover a few-shot prompting paradox: while synthetic examples improve accuracy in closed-ended questions, they severely overload the context window of 8B models in structured tasks, causing signifi cant performance degradation. Ultimately, this study suggests targeted linguistic adaptation offsets lower parameter counts in specialized domains, despite the fragility of smaller models to prompt verbosity.

## 1 Introduction

Evaluation tasks are crucial tools in the field of natural language processing (NLP) to keep track of the progress in machine learning and communicate the potential issues, biases, or needs that may arise from these evaluations (Biderman et al., 2024). To enable this process, benchmarks are deployed as reference points for Large Language Models (LLMs) to assess their performance on an equal footing for comparison purposes. These mainly consist of one or multiple datasets along with relevant metrics and a standardized methodology to evaluate their performance (Ruder, 2021).

Well-known benchmarks, such as General Language Understanding Evaluation (GLUE) (Wang et al., 2018) and Cross-lingual TRansfer Evaluation of Multilingual Encoders (XTREME) (Hu et al., 2020), are collections of tasks used for training, evaluation, and analysis of monolingual and/or multilingual language models in a series of NLU and NLP tasks. These benchmarks serve as an objective lens to observe each model’s suitability for a specific task as well as the field’s progression through remarkable findings.

However, issues regarding the transparency and reproducibility of LLM evaluations conducted by independent researchers have been noted. Specifically, concerns related to data contamination, where models may have seen test data during training, pose a major challenge to the validity of results (Sainz et al., 2023; Zhou et al., 2023). This highlights the need for a unified framework for result reproduction and novel evaluations on any LLM using already supported benchmarks (Biderman et al., 2024; Siddiq et al., 2025).

To this end, Gao et al. (2023) built the Language Model Evaluation Harness—known as lm-eval— which serves as an open-source research library for LLM evaluation. This infrastructure is publicly available, offering a unified framework to test generative language models on a large number of different evaluation tasks and compare the results across models in a standardized way.

In addition to lm-eval, the Inspect AI framework (UK AI Security Institute, 2024) is a core evaluation platform, developed and open-sourced by the UK AI Security Institute (UK AISI). This tool aims to test the capabilities and security of frontier LLMs as well as to evaluate open-ended and complicated tasks, requiring human reasoning and critical thinking. It is publicly available, and it provides reproducible and structured evaluations by enabling the user to handle its infrastructure and create custom functions based on the task needs.

Building upon the need for standardized assessment in non-English contexts, this project introduces the creation and evaluation of two novel Greek benchmarks: greek-protipa-exams (hereafter Prot-Ex) and panhellenic-exams (hereafter Pan-Ex). These datasets comprise questions and answers from official examinations for admission to model and experimental schools, as well as universities in Greece, respectively. Evaluations were conducted using both the lm-eval and Inspect AI frameworks.

Protipa exam topics, as published online by the Greek Government from 2013 to 2026, include topics related to the subjects of Greek Language, Mathematics, Physics, and Religious Studies. Moreover, they cover levels of secondary education; specifically, the topics are divided into middle school and high school entrance exams.

The Panellinies dataset includes exam topics from 2020 to 2026, and the involved subjects are the following: Ancient Greek, Mathematics, Biology, Chemistry, Economics, Computer Science, History, Latin, Physics, and Greek Language. It consists of questions originating only from the general high school exams, offering a broad academic curriculum while preparing students for university entrance.

By leveraging the benefits of the lm-eval infrastructure combined with the Inspect AI framework, we evaluate the answers given by Llama-KriKri-8B-Instruct (Roussis et al., 2025) along with three other LLMs on both benchmarks to address the following research questions:

• RQ1: How does LLM performance differ (i) between open-ended and closed-ended tasks within the same subject and (ii) across different subjects?

• RQ2: Are the classic evaluation metrics, such as BERTScore, reliable in comparison with more up-to-date LLM-as-a-Judge approaches when evaluating Greek educational data?

• RQ3: In which type of tasks does the exploitation of few-shot examples have the most significant effect as opposed to baseline performance?

## 2 Related Work

LLM evaluation has evolved from early multimodal question-answering datasets to large-scale multisubject suites such as MMLU (Hendrycks et al., 2021), which benchmarks zero- and few-shot performance across 57 academic subjects spanning STEM, humanities, and social sciences, establishing a widely adopted standard for broad academic assessment.

While a plethora of multilingual questionanswering benchmarks like Belebele (Bandarkar et al., 2024) and XNLI (Conneau et al., 2018) exist, their scope, question formats, and multimodal resources for the Greek language remain limited. Existing Greek benchmarks focus on tasks that include dialect identification (Chatzikyriakidis et al., 2025), Sign Language Translation (Voskou et al., 2023), and Ancient to Modern Greek machine translation (Mavromatis et al., 2026), alongside speech processing evaluation resources spanning podcasts and regional dialects (Paraskevopoulos et al., 2024; Tsoukala et al., 2026).

The ecosystem has been steadily expanding to encompass physical commonsense reasoning within broad cross-lingual initiatives such as Global PIQA (Chang et al., 2026), domain-specific and educational evaluation suites covering medical exam datasets (Papavassiliou and Prokopidis, 2024), financial NLP with Plutus (Peng et al., 2025), legal reasoning with GreekBarBench (Chlapanis et al., 2025), social-media-based QA with DemosQA (Mastrokostas et al., 2026), empathetic support conversations for exam stress (Kyriazi and Prokopidis, 2026), and broad multi-subject multiple-choice evaluation through native-sourced GreekMMLU (Zhang et al., 2026).

While GreekMMLU provides an excellent breadth of disciplines, it is confined to the multiplechoice format, measuring only discriminative accuracy. To contribute to the ecosystem and address this gap, we introduce the Prot-Ex and Pan-Ex benchmarks, derived from Greek Model and Experimental schools, as well as the national Panhellenic entrance exams. These datasets capture the rigorous nature of the national secondary education curriculum and extend beyond traditional closedended questions (CEQs) by featuring structured queries (SQs) and demanding open-ended generation tasks (OEQs) across diverse disciplines (e.g., Modern Greek, Mathematics, Physics). Furthermore, to support models in reasoning over visual information, we provide supplementary textualized visual contexts (image descriptions and transcriptions) for geometry diagrams and scientific figures. Finally, by integrating these benchmarks into established infrastructures like lm-eval and Inspect AI—and deploying an LLM-as-a-Judge pipeline to evaluate complex reasoning—we ensure transparent, standardized, and easily reproducible evaluation pipelines addressing the methodological limitations of prior studies.

## 3 Methodology

## 3.1 Data Collection and Processing

The data collection and processing tasks were identical for both datasets; we gathered the corpora of the publicly available exams of the experimental school and Panhellenic topics and organized them according to subject and year. To ensure consistency, raw files were renamed using standardized conventions, which were subsequently used for generating unique IDs for each QA pair.

Each exam entry was processed and converted into a structured format. Specifically, questions were parsed into JSON files, while the corresponding solutions were extracted into Markdown files. The resulting datasets preserve essential information for LLM training and evaluation through the following keys: id (unique identifier), question (task description), input (passages or context), choices (candidate answers for closed tasks), answer (the ground truth), image (path and metadata for multimodal entries), and mark (grading score).

It is important to mention that the image\_description and image\_transcription fields were LLM-generated, and specifically, by Gemini 3.1 Pro. The images were provided as input to the LLM, along with an instructional prompt to analytically describe each image and transcribe any visual input that consisted of text. Subsequently, the generated descriptions and transcriptions were manually checked by the authors to ensure accuracy. This process was designed to assist non-multimodal LLMs in comprehending visual elements, providing them with the necessary textualized context in order to give the correct answer (see Appendix A).

## 3.2 Dataset Statistics and Taxonomy

To ensure rigorous evaluation and prevent data contamination, both benchmarks are partitioned into publicly accessible subsets and withheld private

test sets.

The Prot-Ex dataset comprises a total of 1,766 entries, out of which 64 entries from the 2019 examinations are retained as a private evaluation set. We classified the total entries into two primary categories based on the required response type:

• CEQs (1,484 items): This category includes Multiple-Choice, True/False, and Fill-in-thegaps.

• OEQs (282 items): This category consists of Open questions, matching, and Fill-in-thegaps tasks without provided choices.

The Pan-Ex dataset comprises a total of 1,540 entries, with 222 entries from the recent 2026 examinations strictly withheld to serve as a private evaluation set. The total entries are divided into the two respective categories as the aforementioned dataset:

• CEQs (454 items): This category includes Multiple-Choice and True/False tasks.

• OEQs (1,086 items): This category consists of Open questions, matching, and Fill-in-thegaps tasks without provided choices.

The distribution varies across subjects: most of the subjects include all task types, and Physics from protipa exams contains only open questions, while Religious Studies consists solely of closed (multiple-choice) items. A detailed subject-wise breakdown of these task formats for the public subsets is provided in Appendix B and for the private in Appendix C. Furthermore, the dataset supports multimodality, indicating entries that require visual context (diagrams, geometric shapes, maps) for their resolution, including text-based descriptions and transcriptions.

## 3.3 Evaluation Setup

For the evaluation process, we utilized the lm-evalharness (Gao et al., 2023) and the inspect-ai (UK AI Security Institute, 2024) frameworks. We used default settings and parameters for lm-eval-harness, along with a temperature of 0.1, a top-p of 0.9, and a top-k of 40 for inspect-ai in all evaluations described below.

We assessed the performance of four LLMs:

1. Llama-KriKri-8B-Instruct (Roussis et al., 2025): A model fine-tuned specifically for the Greek language.

<table><tr><td rowspan="2">Subject</td><td colspan="3">KriKri-8B</td><td colspan="3">Llama-8B</td><td colspan="3">Gemma-4-26B</td><td colspan="3">Qwen3-32B</td></tr><tr><td>Closed</td><td>Structured</td><td>Open-ended</td><td>Closed</td><td>Structured</td><td>Open-ended</td><td>Closed</td><td>Structured</td><td>Open-ended</td><td>Closed</td><td>Structured</td><td>Open-ended</td></tr><tr><td>Greek Language</td><td>56.45</td><td>21.67</td><td>77.54</td><td>51.08</td><td>7.08</td><td>56.15</td><td>75.94</td><td>35.28</td><td>75.90</td><td>68.28</td><td>30.35</td><td>74.75</td></tr><tr><td>Mathematics</td><td>30.88</td><td>=</td><td>66.89</td><td>30.72</td><td>一</td><td>41.06</td><td>56.26</td><td>一</td><td>87.02</td><td>52.09</td><td>一</td><td>88.91</td></tr><tr><td>Physics</td><td></td><td>一</td><td>69.44</td><td></td><td>一</td><td>47.22</td><td></td><td>一</td><td>75.00</td><td></td><td>一</td><td>69.44</td></tr><tr><td>Religious Studies</td><td>77.78</td><td>=</td><td></td><td>72.22</td><td>一</td><td>1</td><td>78.89</td><td>=</td><td>1</td><td>80.00</td><td>一</td><td>1</td></tr><tr><td>Aggregate</td><td>47.10</td><td>21.67</td><td>69.93</td><td>43.89</td><td>7.08</td><td>45.48</td><td>67.90</td><td>35.28</td><td>83.46</td><td>62.25</td><td>30.35</td><td>84.21</td></tr></table>

Table 1: Zero-shot performance comparison (%) for the Prot-Ex benchmark. We use the LM-Eval framework for the closed and structured question formats, and Inspect AI for open-ended questions. Note: The aggregate scores for Structured tasks reflect only the Greek Language subject, as other subjects do not contain questions in thisformat.

<table><tr><td rowspan="2">Subject</td><td colspan="3">KriKri-8B</td><td colspan="3">Llama-8B</td><td colspan="3">Gemma-4-26B</td><td colspan="3">Qwen3-32B</td></tr><tr><td>Closed</td><td>Structured</td><td>Open-ended</td><td>Closed</td><td>Structured</td><td>Open-ended</td><td>Closed</td><td>Structured</td><td>Open-ended</td><td>Closed</td><td>Structured</td><td>Open-ended</td></tr><tr><td>Ancient Greek</td><td>51.11</td><td>21.04</td><td>57.06</td><td>51.11</td><td>20.00</td><td>35.31</td><td>68.89</td><td>46.04</td><td>65.53</td><td>75.56</td><td>28.75</td><td>64.62</td></tr><tr><td>Economics</td><td>54.76</td><td>=</td><td>56.71</td><td>40.48</td><td>一</td><td>24.32</td><td>78.57</td><td>一</td><td>76.78</td><td>59.52</td><td>=</td><td>74.59</td></tr><tr><td>Physics</td><td>48.65</td><td>-</td><td>51.37</td><td>41.89</td><td>一</td><td>21.23</td><td>66.22</td><td>一</td><td>70.55</td><td>55.41</td><td>一</td><td>74.66</td></tr><tr><td>History</td><td>63.33</td><td></td><td>68.98</td><td>53.33</td><td>1</td><td>44.44</td><td>50.00</td><td>1</td><td>66.20</td><td>53.33</td><td>1</td><td>63.33</td></tr><tr><td>Latin</td><td>73.33</td><td>12.31</td><td>51.68</td><td>66.67</td><td>6.15</td><td>32.72</td><td>66.67</td><td>15.38</td><td>64.03</td><td>73.33</td><td>15.38</td><td>66.14</td></tr><tr><td>Mathematics</td><td>54.84</td><td>一</td><td>61.47</td><td>61.29</td><td>一</td><td>19.47</td><td>87.10</td><td>1</td><td>91.58</td><td>70.97</td><td>1</td><td>93.42</td></tr><tr><td>Greek Language</td><td>96.67</td><td></td><td>82.02</td><td>83.33</td><td></td><td>56.55</td><td>86.67</td><td>1</td><td>82.02</td><td>86.67</td><td></td><td>79.88</td></tr><tr><td>Computer Science</td><td>56.67</td><td>7.50</td><td>71.35</td><td>56.67</td><td>2.50</td><td>41.35</td><td>93.33</td><td>12.50</td><td>77.88</td><td>86.67</td><td>12.50</td><td>86.54</td></tr><tr><td>Chemistry</td><td>48.08</td><td></td><td>55.33 63.38</td><td>40.38</td><td></td><td>23.17</td><td>76.92</td><td>35.00</td><td>84.47</td><td>63.46</td><td></td><td>88.50</td></tr><tr><td>Biology</td><td>51.35</td><td>20.48</td><td></td><td>32.43</td><td>16.90</td><td>30.30</td><td>56.76</td><td></td><td>69.75</td><td>56.76</td><td>54.52</td><td>70.86</td></tr><tr><td>Aggregate</td><td>56.74</td><td>15.57</td><td>59.33</td><td>49.48</td><td>11.89</td><td>30.81</td><td>72.54</td><td>28.70</td><td>74.01</td><td>66.06</td><td>23.86</td><td>75.49</td></tr></table>

Table 2: Zero-shot performance comparison (%) for the Pan-Ex benchmark. We use the LM-Eval framework for the closed and structured question formats, and Inspect AI for open-ended questions.

2. Llama-3.1-8B (Grattafiori et al., 2024): The foundation model of the Greek-focused Llama-KriKri-8B-Instruct, with the same size to compare their capabilities.

3. Gemma-4-26B (Gemma Team et al., 2026): An efficient instruction-tuned multimodal model with 25.2B total parameters from the Google DeepMind Gemma.

4. Qwen3-32B (Qwen Team, 2025): A 32.8B parameter language model from the Qwen3 series, optimized for both complex reasoning and efficient dialogue.

For accuracy purposes, we designed three different task configurations within the evaluation harness:

• CEQ Tasks: Evaluated using the lm-eval harness framework. The evaluation prompt explicitly instructs the model to output a specific identifier (e.g., A, B, 0, 1) corresponding to the correct choice. Performance is strictly measured using Exact Match accuracy.

• SQ Tasks (Structured): This category consists of questions requiring constrained outputs—such as a single word or short phrase rather than extended generation. Evaluation is conducted via lm-eval utilizing a custom targeted metric (structured accuracy). To prevent false negatives caused by model verbosity, raw outputs undergo rigorous normalization (e.g., stripping Markdown syntax, filtering newlines, and truncating explanatory text) before being evaluated against the ground truth using format-specific matching rules.

• OEQ Tasks: Deploys generative prompts that require the model to produce comprehensive, full-text responses or detailed reasoning. To overcome the limitations of traditional string-matching metrics, we employ a dualevaluation strategy. First, BERTScore is utilized to capture semantic similarity against the reference answers. Second, the Inspect AI framework is deployed to implement an LLMas-a-Judge evaluation paradigm, where we specifically used the Gemma-3-27b-it model as the grader.

Computational Infrastructure and Compute Footprint: All evaluation experiments were executed across a distributed computing setup: a local GPU server (NVIDIA GB10 GPU) for the 8B models and Gemma-3-27B-it judge scoring, a European HPC provider (NVIDIA A100 nodes) for closed/structured evaluation runs, a \$20 Open-Router budget for Qwen3-32B and Gemma-4-26B, and Gemini 3.1 Pro for visual context preprocessing. Across all zero-shot and few-shot evaluation passes, total compute overhead is estimated at approximately 24 GPU hours.

## 3.4 Data Availability Limitations

During the data collection phase, certain inconsistencies were encountered with the source exam files. For the Prot-Ex benchmark, all files from 2015 were unavailable or corrupted, resulting in a gap in the dataset. Additionally, no solutions were provided for the Greek language subject for high schools in 2014, nor for the subjects of Greek and Math in 2018. Moreover, essay components of the Greek Language exams were excluded from the main benchmarks as no official answers are provided.

Regarding the Pan-Ex benchmark, the solutions were published by the OEFE (Federation of Private Education Tutors of Greece) and are made publicly available as part of this dataset for scientific and research purposes.

To mitigate potential data leakage for future community evaluations, the publicly released versions of our datasets exclude the 2026 data from Pan-Ex and the 2019 data from Prot-Ex. Consequently, our primary experimental results reported in Section 4 are evaluated on these public datasets. To assess whether model performance remains consistent on unseen data, we additionally conducted experiments on the withheld “private” test sets (2019 for Prot-Ex and 2026 for Pan-Ex), with results reported in Appendix D. Overall model rankings and relative task-format performance patterns on the private test sets closely mirror the public evaluation findings.

## 4 Experimental Results and Analysis

We present the evaluation results for the involved LLMs on the Prot-Ex and Pan-Ex benchmarks. The analysis is structured around the three research questions defined in the introduction, aiming to assess the models’ capabilities across different task formats, evaluation metrics, and in-context learning.

## 4.1 Comparative Performance Across Exercise Types and Subjects

As depicted in Tables 1 and 2, the performance of the evaluated LLMs varies significantly based on both the task modality and the specific academic subject. The evaluation spans two distinct benchmarks: the Prot-Ex benchmark, focusing on four core subjects, and the expansive Pan-Ex benchmark, covering ten diverse disciplines across humanities and STEM. The tasks across these benchmarks are categorized into closed, structured, and open-ended generation.

Addressing the first sub-question (i) regarding performance differences between task types within the same subject, a clear pattern emerges across both benchmarks where SQ tasks consistently yield the lowest performance for all models.

In the Prot-Ex benchmark, models unexpectedly score higher in OEQ tasks compared to CEQs in specific subjects. In Mathematics, Qwen3-32B achieves 88.91% in OEQs versus 52.09% in CEQ tasks. This may reflect the capacity of larger models to articulate reasoning steps correctly, even if they struggle to map their output to a strict multiplechoice format. Furthermore, the dataset design inherently dictates task availability; Religious Studies consists purely of CEQ items—where models demonstrate high knowledge recall (e.g., Qwen scoring 80.00%)—while Physics contains solely OEQs.

Focusing on the Pan-Ex benchmark, the relationship between closed and open-ended performance is highly subject-dependent. For instance, in Ancient Greek, Qwen3-32B scores 75.56% in CEQ tasks and 64.62% in OEQs, but drops significantly to 28.75% in SQ tasks. Conversely, in the Greek Language subject, performance in CEQ tasks is exceptionally high (KriKri scoring 96.67%) and generally exceeds open-ended scores (82.02%).

Concerning the second sub-question (ii) regarding performance across different subjects, larger parameter models (Gemma-4-26B and Qwen3-32B) predictably outperform the 8B models (KriKri and Llama) in aggregate, particularly in subjects demanding technical and scientific reasoning.

In the Prot-Ex benchmark, the localized KriKri-8B demonstrates a competitive edge in the Greek Language subject. In OEQ tasks, KriKri-8B scores 77.54%, outperforming both the larger Gemma-4- 26B (75.90%) and Qwen3-32B (74.75%) models, reinforcing the importance of targeted training.

In the Pan-Ex benchmark, Qwen3-32B achieves outstanding open-ended scores in STEM subjects such as Mathematics (93.42%), Chemistry (88.50%), and Computer Science (86.54%), establishing a substantial gap over its 8B counterparts. Furthermore, Gemma-4-26B records exceptionally high closed-ended scores in Computer Science (93.33%) and Mathematics (87.10%). However, a significant exception is observed in the Greek Language subject, where KriKri-8B achieves the highest closed-ended score (96.67%) and ties with

Gemma in OEQs (82.02%). This highlights the profound impact of language-specific adaptation over sheer parameter count when processing linguistically demanding humanities subjects.

## 4.2 BERTScore vs. LLM-as-a-judge

Tables 3 and 4 present zero-shot performance on OEQs using BERTScore and Inspect AI for both benchmarks. Concerning the LLM-as-a-Judge methodology, we utilized Gemma-3-27B-it as the evaluator, guided by subject-specific rubric prompts that assigned a distinct persona (e.g., a strict Greek national examiner) alongside granular grading rules on a 0.0 to 1.0 scale (detailed in Appendix E).

While this framework provides a nuanced assessment of the models’ reasoning capabilities, qualitative analysis revealed slight evaluator leniency. The judge model occasionally awarded partial credit (0.25) for mere effort on incorrect answers, as seen in a Pan-Ex Ancient Greek task (see Appendix F.1). Exploring stricter negative-constraint prompting or alternative judges remains for future work.

Results highlight a notable contrast between traditional lexical metrics and reasoning-based evaluations. BERTScore remains relatively steady across both benchmarks (65% and 70%), as it relies primarily on lexical overlap and contextual embeddings rather than on factual correctness or logical flow. Consequently, it often assigns high scores to incorrect outputs; for instance, it awarded a 0.70 to a completely wrong Prot-Ex Mathematics answer, whereas the LLM judge correctly assigned a score of 0.0 (see Appendix F.2).

The data demonstrate a compression effect in BERTScore outputs. BERTScore aggregates range between 64% and 71% across models and benchmarks. The metric over-reports the performance of Llama-8B and under-reports the performance of Gemma-4-26B. In the Pan-Ex benchmark, BERTScore evaluates Llama-8B at 65.68%, while the LLM-Judge evaluates it at 30.81%. In the Prot-Ex benchmark, BERTScore evaluates Gemma-4- 26B at 69.36%, while the LLM-Judge evaluates it at 83.46%.

Overall, we observe that Qwen3-32B consistently demonstrated the highest proficiency and accuracy across both benchmarks, achieving aggregate LLM-Judge scores of 84.21% in Prot-Ex and 75.49% in Pan-Ex. Notably, the Greek-focused KriKri-8B significantly outperformed its base foundation model, Llama-8B. For instance, it nearly doubled Llama-8B’s aggregate score in the Pan-Ex benchmark (59.33% vs. 30.81%) and reached 82.02% in the Pan-Ex Greek Language subject. A representative example occurred in a Prot-Ex Modern Greek task, where KriKri-8B correctly generated the gold answer (scoring 1.0), while Llama-8B yielded a partially accurate response scoring only 0.50 (see Appendix F.3). This performance gap extends to STEM disciplines; in the Pan-Ex Mathematics open-ended evaluation, Llama-8B scores 19.47%, whereas KriKri-8B achieves 61.47%. This 42-point increase may be attributed to the impact of Greek-specific continued pretraining.

## 4.3 Performance Across Baseline and Few-shot examples

As shown in tables 5 and 6, we analyzed the aggregate impact of 5-shot prompting across different task formats, revealing a distinct pattern. It is observed that few-shot examples consistently improved performance in CEQ tasks across all models and both benchmarks. The most notable remark was made in the Pan-Ex benchmark, where KriKri-8B and Llama-8B improved by 8.5 and 6.7 percentage points, respectively. This suggests that providing synthetic examples effectively aligns the models with the expected objective formats (e.g., multiple-choice; see Appendix G).

Conversely, OEQ tasks exhibited remarkable stability in both benchmarks. The inclusion of few-shot examples yielded minimal performance changes across the board. This indicates that for open-ended generation, zero-shot instructions—combined with a strong system prompt—are largely sufficient for the models to understand the reasoning and formatting requirements, rendering additional context redundant.

Interestingly, SQ tasks experienced a negative impact from few-shot prompting, particularly for the 8B parameter models. KriKri-8B saw a significant drop in both benchmarks (e.g., from 21.67% to 8.61% in Prot-Ex), while Llama-8B collapsed almost entirely in Pan-Ex, diving from 11.89% to a mere 0.41% (see Appendix H). Notably, larger models like Gemma and Qwen survived this fewshot SQ collapse, validating the capacity overload hypothesis for 8B models. This degradation implies that filling the context window with complex synthetic examples might overwhelm smaller models, causing them to lose track of the specific output constraints or to become distracted by the lengthy and verbose prompt.

<table><tr><td rowspan="2">Subject</td><td colspan="2">KriKri-8B</td><td colspan="2">Llama-8B</td><td colspan="2">Gemma-4-26B</td><td colspan="2">Qwen3-32B</td></tr><tr><td>BERTScore</td><td>LLM-Judge</td><td>BERTScore</td><td>LLM-Judge</td><td>BERTScore</td><td>LLM-Judge</td><td>BERTScore</td><td>LLM-Judge</td></tr><tr><td>Greek Language</td><td>68.17</td><td>77.54</td><td>69.56</td><td>56.15</td><td>72.26</td><td>75.90</td><td>70.57</td><td>74.75</td></tr><tr><td>Mathematics</td><td>68.15</td><td>66.89</td><td>65.65</td><td>41.06</td><td>67.56</td><td>87.02</td><td>71.22</td><td>88.91</td></tr><tr><td>Physics</td><td>71.83</td><td>69.44</td><td>75.11</td><td>47.22</td><td>79.82</td><td>75.00</td><td>77.55</td><td>69.44</td></tr><tr><td>Aggregate</td><td>68.31</td><td>69.93</td><td>67.11</td><td>45.48</td><td>69.36</td><td>83.46</td><td>71.29</td><td>84.21</td></tr></table>

Table 3: Comparison of evaluation metrics (%) for zero-shot open-ended tasks in the Prot-Ex benchmark. The LLM-as-a-Judge approach utilizes Gemma-3-27B via the Inspect AI framework.

<table><tr><td rowspan="2">Subject</td><td colspan="2">KriKri-8B</td><td colspan="2">Llama-8B</td><td colspan="2">Gemma-4-26B</td><td colspan="2">Qwen3-32B</td></tr><tr><td>BERTScore</td><td>LLM-Judge</td><td>BERTScore</td><td>LLM-Judge</td><td>BERTScore</td><td>LLM-Judge</td><td>BERTScore</td><td>LLM-Judge</td></tr><tr><td>Ancient Greek</td><td>64.37</td><td>57.06</td><td>67.71</td><td>35.31</td><td>67.69</td><td>65.53</td><td>61.89</td><td>64.62</td></tr><tr><td>Economics</td><td>65.24</td><td>56.71</td><td>63.70</td><td>24.32</td><td>66.13</td><td>76.78</td><td>63.34</td><td>74.59</td></tr><tr><td>Physics</td><td>67.06</td><td>51.37</td><td>65.77</td><td>21.23</td><td>67.96</td><td>70.55</td><td>69.38</td><td>74.66</td></tr><tr><td>History</td><td>70.17</td><td>68.98</td><td>69.61</td><td>44.44</td><td>70.33</td><td>66.20</td><td>70.43</td><td>63.33</td></tr><tr><td>Latin</td><td>55.20</td><td>51.68</td><td>62.11</td><td>32.72</td><td>62.63</td><td>64.03</td><td>53.29</td><td>66.14</td></tr><tr><td>Mathematics</td><td>72.47</td><td>61.47</td><td>69.94</td><td>19.47</td><td>71.67</td><td>91.58</td><td>73.88</td><td>93.42</td></tr><tr><td>Greek Language</td><td>70.47</td><td>82.02</td><td>68.88</td><td>56.55</td><td>70.54</td><td>82.02</td><td>69.73</td><td>79.88</td></tr><tr><td>Computer Science</td><td>64.60</td><td>71.35</td><td>65.20</td><td>41.35</td><td>64.83</td><td>77.88</td><td>65.84</td><td>86.54</td></tr><tr><td>Chemistry</td><td>62.70</td><td>55.33</td><td>61.04</td><td>23.17</td><td>64.29</td><td>84.47</td><td>63.79</td><td>88.50</td></tr><tr><td>Biology</td><td>67.58</td><td>63.38</td><td>68.18</td><td>30.30</td><td>68.20</td><td>69.75</td><td>62.09</td><td>70.86</td></tr><tr><td>Aggregate</td><td>64.77</td><td>59.33</td><td>65.68</td><td>30.81</td><td>66.88</td><td>74.01</td><td>63.86</td><td>75.49</td></tr></table>

Table 4: Comparison of evaluation metrics (%) for zero-shot open-ended tasks in the Pan-Ex benchmark. The LLM-as-a-Judge approach utilizes Gemma-3-27B via the Inspect AI framework.

## 5 Discussion

Regarding RQ1, performance across both benchmarks heavily depends on task modality and subject, with SQ tasks consistently yielding the lowest scores. Interestingly, larger models (e.g., Qwen3- 32B) sometimes excel in open-ended generation (88.91% in Prot-Ex Mathematics) while struggling with CEQs, likely favoring reasoning articulation over strict option mapping. Furthermore, while larger models predictably dominate STEM, the localized KriKri-8B outperforms them in humanities (Greek Language subject), reaching 96.67% in Pan-Ex CEQ tasks. This proves that targeted linguistic pre-training can effectively offset lower parameter counts in specialized domains.

Addressing RQ2, traditional lexical metrics like BERTScore fall short for open-ended reasoning, statically measuring lexical overlap rather than factual correctness and reasoning. Conversely, the LLM-as-a-Judge methodology (via Inspect AI) provides a more nuanced and qualitative assessment, though occasional judge leniency towards incorrect answers may compromise evaluation strictness. Overall, Qwen3-32B achieved the highest open-ended aggregate score (84.21% in Prot-Ex). Meanwhile, the Greek-adapted KriKri-8B consistently received higher evaluations in OEQs than its base model, Llama-8B (a ∼20% Pan-Ex gap), from the Gemma-3-27B judge.

Concerning RQ3, 5-shot prompting primarily benefits CEQs, especially for smaller models (KriKri-8B and Llama-8B), which improved by ∼7 percentage points in Pan-Ex. Conversely, OEQs showed minimal fluctuations, indicating that zeroshot instructions and strong system prompts are sufficient for accurate text generation. Notably, few-shot prompting negatively impacted SQs, causing a sharp decline in 8B models; Llama-8B plummeted from 11.89% to 0.41% in Pan-Ex, exhibiting erratic behavior in matching and fill-in-the-gaps tasks. This suggests overloading the context window with complex synthetic examples overwhelms smaller models, distracting them from strict output constraints.

## 6 Resources

We release Prot-Ex and Pan-Ex benchmarks as open-source resources on Hugging Face (available at https://huggingface.co/ datasets/ilsp/greek-protipa-exams and https://huggingface.co/datasets/ilsp/ panellinies-exams-dataset, respectively). Additionally, the accompanying codebase, including evaluation scripts and management commands to use the given datasets, is publicly available in our GitHub repositories (https://github.com/PK9811-hub/protipa\_ exams\_dataset and https://github.com/ PK9811-hub/panellinies\_exams\_dataset).

<table><tr><td rowspan="2">Model</td><td colspan="2">Closed</td><td colspan="2">Structured</td><td colspan="2">Open-ended</td></tr><tr><td>Zero-Shot</td><td>Few-Shot</td><td>Zero-Shot</td><td>Few-Shot</td><td>Zero-Shot</td><td>Few-Shot</td></tr><tr><td>KriKri-8B</td><td>47.10</td><td>52.20</td><td>21.67</td><td>8.61</td><td>69.93</td><td>68.03</td></tr><tr><td>Llama-8B</td><td>43.89</td><td>49.06</td><td>7.08</td><td>8.61</td><td>45.48</td><td>44.30</td></tr><tr><td>Gemma-4-26B</td><td>67.90</td><td>69.43</td><td>35.28</td><td>36.94</td><td>83.46</td><td>83.64</td></tr><tr><td>Qwen3-32B</td><td>62.25</td><td>62.81</td><td>30.35</td><td>32.85</td><td>84.21</td><td>84.05</td></tr></table>

Table 5: Aggregate impact of few-shot prompting on model performance (%) across task types in the Prot-Ex benchmark. Note: The aggregate scores for Structured tasks reflect only the Greek Language subject, as other subjects do not contain questions in thisformat.

<table><tr><td rowspan="2">Model</td><td colspan="2">Closed</td><td colspan="2">Structured</td><td colspan="2">Open-ended</td></tr><tr><td>Zero-Shot</td><td>Few-Shot</td><td>Zero-Shot</td><td>Few-Shot</td><td>Zero-Shot</td><td>Few-Shot</td></tr><tr><td>KriKri-8B</td><td>56.74</td><td>65.28</td><td>15.57</td><td>4.88</td><td>59.33</td><td>60.13</td></tr><tr><td>Llama-8B</td><td>49.48</td><td>56.22</td><td>11.89</td><td>0.41</td><td>30.81</td><td>32.49</td></tr><tr><td>Gemma-4-26B</td><td>72.54</td><td>78.76</td><td>28.70</td><td>30.77</td><td>74.01</td><td>70.34</td></tr><tr><td>Qwen3-32B</td><td>66.06</td><td>71.76</td><td>23.86</td><td>22.11</td><td>75.49</td><td>75.71</td></tr></table>

Table 6: Aggregate impact of few-shot prompting on model performance (%) across task types in the Pan-Ex benchmark.

## 7 Conclusions

This study demonstrates that while large-scale models predictably excel in complex STEM reasoning, parameter size is not the sole determinant of success. Particularly for a language with limited benchmark coverage such as Greek, smaller but localized models, such as KriKri-8B, can outperform their massive counterparts in linguistically demanding humanities subjects. This highlights that targeted linguistic adaptation and focused domain training can effectively offset lower parameter counts, offering an efficient paradigm for specialized educational applications.

Furthermore, our evaluation exposes the limitations of traditional lexical metrics like BERTScore in capturing factual correctness and logical flow during open-ended reasoning. Adopting an LLMas-a-Judge methodology provides a more nuanced qualitative assessment. However, this approach is not entirely infallible; we observed instances of evaluator leniency where the judge model awarded partial credit for flawed answers. This underscores that while automated LLM evaluation is superior to static metrics, it requires further refinement via strict negative-constraint prompting.

Finally, our findings reveal that few-shot prompting is not a panacea. While providing synthetic examples clearly benefits CEQs, it yields negligible improvements in OEQs and actively degrades performance in SQ tasks, especially for smaller models. Overloading the context window causes 8B models to lose structural focus, leading to erratic behavior. Ultimately, prompt engineering must be carefully tailored to both the specific task modality and the architectural constraints of each model.

In terms of future work, a key direction involves extending our evaluation paradigm to native Vision Large Language Models (VLMs). Since our current methodology relies on textualized visual contexts, testing VLMs directly on the raw image inputs will allow us to assess their inherent multimodal reasoning capabilities. This will provide critical insights into whether processing visual data introduces performance decline or improvements in multimodal QAs compared to text-only alternatives.

## 8 Limitations

Due to the unavailability of source exam data, our study faces certain limitations regarding data composition. The Prot-Ex benchmark contains temporal discontinuities (e.g., missing files or official solutions for the years 2014, 2015, and 2018) and excludes essay-based components, thus preventing the assessment of the models’ extensive writing capabilities. Additionally, the Physics subject comprises only 9 questions; consequently, performance metrics for this specific domain lack statistical robustness and should be interpreted with caution.

A persistent challenge in evaluating on educational benchmarks is the potential risk of data contamination. Although these original materials were released in noisy, unstructured formats (e.g., raw PDFs and Word documents), making direct memorization of structured question-answer pairs highly unlikely, contamination cannot be entirely ruled out. To address this, we constructed a private test set holdout, withholding the 2019 exams for Prot-Ex and the 2026 exams for Pan-Ex from public releases. This serves as a robust safeguard against evaluation leakage and ensures the integrity of our baseline measurements.

Furthermore, while we observe a severe performance degradation in Structured Questions (SQ) under few-shot settings for smaller models, a detailed ablation study to definitively isolate contextwindow overload from prompt-format confusion was deferred due to computational budget constraints. We plan to incorporate these extended ablations in the final version of this work.

## 9 Ethical Considerations

The Prot-Ex and Pan-Ex benchmarks comprise content derived from official educational bodies, including the Greek Ministry of Education, the Governing Body of Model and Experimental schools, and the OEFE organization. We explicitly acknowledge that all original exam materials remain the intellectual property of these respective entities. Our use of this data is strictly limited to noncommercial, academic research, in accordance with European text and data mining exceptions for scientific purposes and standard fair use principles.

From a broader ethical perspective, releasing these resources addresses the ongoing disparity in LLM evaluation, which predominantly focuses on high-resource languages like English. By providing robust benchmarks for Greek, we aim to support the development of linguistically inclusive and unbiased AI models.

While every effort has been made to ensure the accuracy and completeness of these datasets, any errors, omissions, or formatting issues are the result of processing and transformation pipelines and are not related to the original sources.

## References

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), page 749–775. Association for Computational Linguistics.

Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, Anthony DiPofi, Julen Etxaniz, Benjamin Fattori, Jessica Zosa Forde, Charles Foster, Jeffrey Hsu, Mimansa Jaiswal, Wilson Y. Lee, Haonan Li, and 11 others. 2024. Lessons from the trenches on reproducible evaluation of language models. Preprint, arXiv:2405.14782.

Tyler A. Chang, Catherine Arnett, Abdelrahman Sadallah, Abdelrahman Eldesokey, Abeer Kashar, Abolade Daud, Abosede Grace Olanihun, Adamu Labaran Mohammed, Adeyemi Praise, Adhikarimayum Meerajita Sharma, Aditi Gupta, Adril Putra Merin, Adwoa Bremang, Afitab Iyigun, Afonso Simplício, Ahmed Essouaied, Aicha Chorana, Akhil Eppa, Akintunde Oladipo, and 361 others. 2026. Global PIQA: Evaluating commonsense reasoning across 100+ languages and cultures. Preprint.

Stergios Chatzikyriakidis, Chatrine Qwaider, Ilias Kolokousis, Christina Koula, Dimitris Papadakis, and Efthymia Sakellariou. 2025. GRDD: A Dataset for Greek Dialectal NLP. Preprint, arXiv:2308.00802.

Odysseas S. Chlapanis, Dimitrios Galanis, Nikolaos Aletras, and Ion Androutsopoulos. 2025. GreekBar-Bench: A challenging benchmark for free-text legal reasoning and citations. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, pages 25099–25119, Suzhou, China. Association for Computational Linguistics.

Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and

5 others. 2023. A framework for few-shot language model evaluation.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle˘ Casbon, Mayank Chaturvedi, Aditya Chawla, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, and 304 others. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. Preprint, arXiv:2009.03300.

Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalization. Preprint, arXiv:2003.11080.

Panagiota Kyriazi and Prokopis Prokopidis. 2026. Empathy in Greek exam-related support conversations: A comparative evaluation of LLM responses. In Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026), pages 2682– 2697, Palma, Mallorca, Spain. European Language Resources Association (ELRA).

Charalampos Mastrokostas, Nikolaos Giarelis, and Nikos Karacapilidis. 2026. Evaluating monolingual and multilingual large language models for Greek question answering: The DemosQA benchmark. Preprint, arXiv:2602.16811.

Spyridon Mavromatis, Sokratis Sofianopoulos, Prokopis Prokopidis, and Maria Giagkou. 2026. Ancient Greek to Modern Greek Machine Translation: A Novel Benchmark and Fine-Tuning Experiments on LLMs and NMT Models. In Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026), pages 8685–8698, Palma, Mallorca, Spain. European Language Resources Association (ELRA).

Vassilis Papavassiliou and Prokopis Prokopidis. 2024. Greek Medical Multiple Choice QA. Hugging-Face https://huggingface.co/datasets/ilsp/ medical\_mcqa\_greek.

Georgios Paraskevopoulos, Chara Tsoukala, Athanasios Katsamanis, and Vassilis Katsouros. 2024. The Greek podcast corpus: Competitive speech models for low-resourced languages with weakly supervised data. In Interspeech 2024, pages 4728–4732.

Xueqing Peng, Triantafillos Papadopoulos, Efstathia Soufleri, Polydoros Giannouris, Ruoyu Xiang, Yan Wang, Lingfei Qian, Jimin Huang, Qianqian Xie, and Sophia Ananiadou. 2025. Plutus: Benchmarking large language models in low-resource Greek finance. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 30176–30202, Suzhou, China. Association for Computational Linguistics.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Dimitris Roussis, Leon Voukoutis, Georgios Paraskevopoulos, Sokratis Sofianopoulos, Prokopis Prokopidis, Vassilis Papavassileiou, Athanasios Katsamanis, Stelios Piperidis, and Vassilis Katsouros. 2025. Krikri: Advancing open large language models for Greek. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 5012–5033, Suzhou, China. Association for Computational Linguistics.

Sebastian Ruder. 2021. Challenges and Opportunities in NLP Benchmarking. https://www.ruder.io/ nlp-benchmarking/. Blog post.

Oscar Sainz, Jon Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. 2023. NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10776–10787, Singapore. Association for Computational Linguistics.

Mohammed Latif Siddiq, Arvin Islam-Gomes, Natalie Sekerak, and Joanna C. S. Santos. 2025. Large language models for software engineering: A reproducibility crisis. Preprint, arXiv:2512.00651.

Chara Tsoukala, Stavros Bompolas, Antigoni Margariti, Konstantina Panagiotou, Maria Elisavet Plaiti, Nefeli Tzanakaki, Petros Karatsareas, Angela Ralli, Antonios Anastasopoulos, and Stella Markantonatou. 2026. Extending ASR evaluation resources for Modern Greek dialects. In Proceedings of the 13th Workshop on NLPfor Similar Languages, Varieties and Dialects, pages 210–222, Rabat, Morocco. Association for Computational Linguistics.

UK AI Security Institute. 2024. Inspect AI: Framework for Large Language Model Evaluations. https: //inspect.aisi.org.uk/. Software. MIT License. GitHub repository: https://github.com/ UKGovernmentBEIS/inspect\_ai.

Andreas Voskou, Konstantinos P. Panousis, Harris Partaourides, Kyriakos Tolias, and Sotirios Chatzis. 2023. A New Dataset for End-to-End Sign Language Translation: The Greek Elementary School Dataset. Preprint, arXiv:2310.04753.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings ofthe

![](images/bbd08c4a583d2f50a9db5178021323d784faa70eeab3f7b22a018a348362f4c2.jpg)  
(b) Prot-Ex (Mathematics 2025) example.  
Figure 1: Examples with LLM-generated image descriptions and transcriptions from the Pan-Ex (Physics 2022) and Prot-Ex (Mathematics 2025) benchmarks.

Figure 1 illustrates two representative examples of the LLM-generated visual context used in our benchmarks. For each image entry, Gemini 3.1 Pro was prompted to produce (i) a detailed textual description of the visual content and (ii) a transcription of any text visible in the image. These textualized representations serve as the sole visual input to the text-only LLMs evaluated in this study, enabling them to reason over image-based questions without native multimodal capabilities.

## B Prot-Ex and Pan-Ex Public Benchmarks: Subject and Task Format Distribution

<table><tr><td>Subject</td><td>CEQ</td><td>SQ</td><td>OEQ</td><td>Total Items</td></tr><tr><td colspan="3">Prot-Ex</td><td></td><td></td></tr><tr><td>Greek Language</td><td>735</td><td>57</td><td>61</td><td>853</td></tr><tr><td>Mathematics</td><td>599</td><td>0</td><td>151</td><td>750</td></tr><tr><td>Religious Studies</td><td>90</td><td>0</td><td>0</td><td>90</td></tr><tr><td>Physics</td><td>0</td><td>0</td><td>9</td><td>9</td></tr><tr><td>Total</td><td>1,424</td><td>57</td><td>221</td><td>1,702</td></tr><tr><td colspan="3">Pan-Ex</td><td></td><td></td></tr><tr><td>Ancient Greek</td><td>45</td><td>16</td><td>131</td><td>192</td></tr><tr><td>Latin</td><td>15</td><td>13</td><td>149</td><td>177</td></tr><tr><td>Chemistry</td><td>52</td><td>0</td><td>123</td><td>175</td></tr><tr><td>Physics</td><td>74</td><td>0</td><td>73</td><td>147</td></tr><tr><td>Biology</td><td>37</td><td>4</td><td>99</td><td>140</td></tr><tr><td>Mathematics</td><td>31</td><td>0</td><td>95</td><td>126</td></tr><tr><td>Economics</td><td>42</td><td>0</td><td>73</td><td>115</td></tr><tr><td>Computer Science</td><td>30</td><td>8</td><td>52</td><td>90</td></tr><tr><td>History</td><td>30</td><td>0</td><td>54</td><td>84</td></tr><tr><td>Greek Language</td><td>30</td><td>0</td><td>42</td><td>72</td></tr><tr><td>Total</td><td>386</td><td>41</td><td>891</td><td>1,318</td></tr></table>

Table 7: Subject-wise distribution across Closed-Ended Questions (CEQ), Structured Questions (SQ), and Open-Ended Questions (OEQ) for the public Prot-Ex and Pan-Ex benchmarks.

## C Prot-Ex and Pan-Ex Private Test Sets: Subject and Task Format Distribution

<table><tr><td rowspan=1 colspan=1>Subject            CEQ SQ OEQ  Total Items</td></tr><tr><td rowspan=1 colspan=1>Prot-Ex Private</td></tr><tr><td rowspan=1 colspan=1>Greek Language      35    5      0           40Mathematics          16    0      8           24</td></tr><tr><td rowspan=1 colspan=1>Total                 51    5      8           64</td></tr><tr><td rowspan=1 colspan=1>Pan-Ex Private</td></tr><tr><td rowspan=1 colspan=1>Ancient Greek          7    0     23           30</td></tr><tr><td rowspan=1 colspan=1>Latin                   5    7     29           41</td></tr><tr><td rowspan=1 colspan=1>Chemistry             12    0     20           32</td></tr><tr><td rowspan=1 colspan=1>Physics               12    0     11           23</td></tr><tr><td rowspan=1 colspan=1>Biology                5    1     11           17</td></tr><tr><td rowspan=1 colspan=1>Mathematics            5    0     16           21</td></tr><tr><td rowspan=1 colspan=1>Economics             7    0     11           18</td></tr><tr><td rowspan=1 colspan=1>Computer Science      5    2      6           13</td></tr><tr><td rowspan=1 colspan=1>History                 5    0      8           13Greek Language        5    0      9           14</td></tr><tr><td rowspan=1 colspan=1>Total                  68   10   144          222</td></tr></table>

Table 8: Subject-wise distribution across Closed-Ended Questions (CEQ), Structured Questions (SQ), and Open-Ended Questions (OEQ) for the private Prot-Ex and Pan-Ex test sets.

## D Results on the Prot-Ex and Pan-Ex Private Test Sets

This appendix reports detailed model performance metrics on the withheld private test sets for both benchmarks (2019 split for Prot-Ex, N = 64; 2026 split for Pan-Ex, N = 222). Table 9 presents the zero-shot and few-shot results across closed, structured, and open-ended question formats for Prot-Ex, while Table 10 details the corresponding performance evaluation on Pan-Ex. Across both private test sets, overall model rankings and relative format behaviors closely mirror the primary public dataset results reported in Section 4.

<table><tr><td>Model</td><td>Setup</td><td>Closed</td><td>Struct.</td><td>Open</td></tr><tr><td>KriKri-8B</td><td>Zero-Shot Few-Shot</td><td>64.71 60.78</td><td>36.00 56.00</td><td>56.25 84.38</td></tr><tr><td>Llama-8B</td><td>Zero-Shot Few-Shot</td><td>54.90 56.86</td><td>32.00 56.00</td><td>31.25 43.75</td></tr><tr><td>Gemma-4-26B</td><td>Zero-Shot Few-Shot</td><td>68.63 64.71</td><td>60.00 80.00</td><td>90.62 96.88</td></tr><tr><td>Qwen3-32B</td><td>Zero-Shot Few-Shot</td><td>72.55 70.59</td><td>12.00 60.00</td><td>93.75 96.88</td></tr></table>

Table 9: Evaluation results for the private Prot-Ex (2019) test set. Scores represent aggregate accuracy for Closed and Structured question formats under the LM-Eval framework, alongside Inspect AI evaluations for Open-Ended questions.

<table><tr><td>Model</td><td>Setup</td><td>Closed</td><td>Struct.</td><td>Open</td></tr><tr><td rowspan="2">KriKri-8B</td><td>Zero-Shot</td><td>72.06</td><td>13.00</td><td>57.26</td></tr><tr><td>Few-Shot</td><td>79.41</td><td>0.00</td><td>56.18</td></tr><tr><td>Llama-8B</td><td>Zero-Shot Few-Shot</td><td>57.35 63.24</td><td>6.50 0.00</td><td>31.08 29.34</td></tr><tr><td rowspan="2">Gemma-4-26B</td><td>Zero-Shot</td><td>80.88</td><td>23.50</td><td>73.02</td></tr><tr><td>Few-Shot</td><td>79.41</td><td>26.00</td><td>68.68</td></tr><tr><td rowspan="2">Qwen3-32B</td><td>Zero-Shot</td><td>79.41</td><td>24.00</td><td></td></tr><tr><td>Few-Shot</td><td>77.94</td><td>16.00</td><td>75.14 78.23</td></tr></table>

Table 10: Evaluation results for the private Pan-Ex (2026) test set. Scores represent aggregate accuracy for Closed and Structured question formats under the LM-Eval framework, alongside Inspect AI evaluations for Open-Ended questions.

## E LLM-as-a-Judge Evaluation Prompts

In this section, we provide representative examples of the system instructions and grading rubrics utilized for the LLM-as-a-Judge evaluation methodology. For each prompt, we present the original

Greek text provided to the model, followed by its English translation.

## E.1 Modern Greek Language (Pan-Ex)

## System Instruction (Original Greek):

Είσαι ένας 18χρονος τελειόφοιτος Λυκείου που απαντά σε διαγώνισμα Πανελλαδικών στη Νεοελληνική Γλώσσα και Λογοτεχνία. Απάντησε στο ερώτημα συγκροτημένα, με πλού- σιο λεξιλόγιο, σωστή δομή, άρτια γραμματική και συντακτικό.

## System Instruction (English Translation):

You are an 18-year-old high school senior taking the Panhellenic national exam in Modern Greek Language and Literature. Answer the question coherently, with rich vocabulary, proper structure, and flawless grammar and syntax.

## Rubric (Original Greek):

Είσαι ένας αυστηρός ΄Ελληνας βαθμολογητής Πανελλαδικών Εξετάσεων που διορθώνει το γραπτό Νεοελληνικής Γλώσσας ενός 18χρονου τελειόφοιτου Λυκείου. Αξιολόγησε την απάντηση του μαθητή (Submission) συγ- κρίνοντάς τη με την πρότυπη λύση (Criterion). Χρησιμοποίησε κλίμακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0. Κανόνες:

• 1. Εστίασε στην ορθογραφία, τη γραμ- ματική, το συντακτικό, την ακρίβεια του λεξιλογίου και την πλήρη απόδοση του νοή- <sub>ματος</sub>.

• 2. Δώσε 1.0 αν η απάντηση είναι άψογη νοηματικά και συντακτικά, πλήρως τεκμηρι- ωμένη και στοχευμένη.

• 3. Δώσε 0.75 αν βρήκε το σωστό νόημα, αλλά έκανε κάποιο ελαφρύ εκφραστικό, συντακτικό ή ορθογραφικό λάθος.

• 4. Δώσε 0.50 αν βρήκε μέρος της απάντησης ή αν η διατύπωση είναι ασαφής, άκομψη ή δημιουργεί πλεονασμούς.

• 5. Δώσε 0.25 αν η απάντηση είναι ελλιπής ή μερικώς εκτός θέματος, αλλά περιέχει τουλάχιστον ένα σωστό σημείο αναφοράς.

• 6. Δώσε 0.0 αν η απάντηση είναι εντελώς εκτός θέματος, λανθασμένη ή παρουσιάζει σοβαρότατα πραγματολογικά λάθη.

## Rubric (English Translation):

You are a strict Greek national examiner grading the Modern Greek Language exam paper of an 18- year-old high school senior. Evaluate the student’s answer (Submission) by comparing it with the gold standard solution (Criterion). Use the following grading scale: 0.0, 0.25, 0.5, 0.75, or 1.0. Rules:

• 1. Focus on spelling, grammar, syntax, vocabulary accuracy, and the complete rendering of the meaning.

• 2. Provide a 1.0 score if the answer is conceptually and syntactically flawless,fully substantiated, and targeted.

• 3. Provide a 0.75 score if the correct meaning is captured, but there is a minor expressive, syntactic, or spelling error.

• 4. Provide a 0.50 score if part of the answer is correct or if the phrasing is vague, awkward, or creates redundancies.

• 5. Provide a 0.25 score if the answer is incomplete or partially off-topic, but contains at least one correct reference point.

• 6. Provide a 0.0 score if the answer is completely off-topic, incorrect, or presents severe factual errors.

## E.2 Mathematics (Prot-Ex)

## System Instruction (Original Greek):

Είσαι ένας 12χρονος ΄Ελληνας μαθητής που απαντά σε διαγώνισμα Μαθηματικών. Λύσε το πρόβλημα βήμα-βήμα, δείχνοντας τις πράξ- εις σου απλά, και γράψε το τελικό αριθμητικό αποτέλεσμα καθαρά στο τέλος.

## System Instruction (English Translation):

You are a 12-year-old Greek student taking a Mathematics exam. Solve the problem step-by-step, showing your operations simply, and write the final numerical result clearly at the end.

## Rubric (Original Greek):

Είσαι ένας αυστηρός ΄Ελληνας εκπαιδευτικός που βαθμολογεί το γραπτό Μαθηματικών ενός 12χρονου μαθητή. Αξιολόγησε την απάντηση του μαθητή (Submission) συγκρίνοντάς τη με την πρότυπη λύση (Criterion). Χρησιμοποίησε κλί- μακα βαθμολόγησης: 0.0, 0.25, 0.5, 0.75, ή 1.0. Κανόνες:

• 1. Δώσε 1.0 αν η μεθοδολογία είναι σωστή και το τελικό αποτέλεσμα ταυτίζεται από- λυτα με το Criterion.

• 2. Δώσε 0.75 αν η μεθοδολογία είναι ολόσωστη αλλά υπάρχει ένα μικρό αρι- θμητικό λάθος στο τελικό αποτέλεσμα.

• 3. Δώσε 0.50 αν ο μαθητής ακολούθησε τα σωστά βήματα μέχρι τη μέση ή βρήκε μόνο μέρος της λύσης (π.χ. τη μία από τις δύο λύσεις μιας εξίσωσης).

• 4. Δώσε 0.25 αν η μεθοδολογία είναι λαν- θασμένη ή ατελής, αλλά εφάρμοσε σωστά κάποιον βασικό τύπο ή έκανε μια σωστή αρ- χική σκέψη.

• 5. Δώσε 0.0 αν και η λογική και το αποτέλεσμα είναι εντελώς λανθασμένα ή δεν υπάρχει καμία προσπάθεια λύσης.

## Rubric (English Translation):

You are a strict Greek educator grading the Mathematics exam paper ofa 12-year-old student. Evaluate the student’s answer (Submission) by comparing it with the gold standard solution (Criterion). Use the following grading scale: 0.0, 0.25, 0.5, 0.75, or 1.0. Rules:

• 1. Provide a 1.0 score if the methodology is correct and thefinal result matches the Criterion perfectly.

• 2. Provide a 0.75 score if the methodology is entirely correct but there is a minor arithmetic error in the final result.

• 3. Provide a 0.50 score ifthe studentfollowed the correct steps halfway or found only part of the solution (e.g., one of two solutions to an equation).

• 4. Provide a 0.25 score ifthe methodology is incorrect or incomplete, but the student correctly applied a basicformula or made a correct initial reasoning step.

• 5. Provide a 0.0 score ifboth the logic and the result are completely incorrect, or if there is no attempt to solve the problem.

## F LLM-as-a-Judge Examples from Evaluation Logs

## F.1 Evaluator Leniency: Pan-Ex Ancient Greek

This example demonstrates a case where the judge model (Gemma-3-27B-it) awarded partial credit to a model (Gemma-4-26B) for an incorrect answer. The judge recognized the student’s effort to explain the broader context, despite the failure to address the specific grammatical question.

Question ID: ancient\_greek\_gel\_2020\_Α1.β.1

## Question:

«<sub>σφ</sub>εῖ<sub>ς</sub> ἐ<sub>ντι</sub>θέ<sub>ναι</sub>...»: To which word in the ancient text does the pronoun «<sub>σφε</sub>ῖ<sub>ς</sub>» refer?

## Target:

The word in the ancient text to which the pronoun «σφεῖς» refers is: «ἐπαγγελλόμενοί».

## Student Answer (Gemma-4-26B):

The pronoun «<sub>σφε</sub>ῖ<sub>ς</sub>» (meaning "only", "simply") refers to the concept of ἐπιστήμης [knowledge/science]. In this specific part of the text, Plato reproduces the view of certain people who argue that education is not the actual acquisition of knowledge, but simply the addition of a "science" to the soul, without it residing within it, exactly as adding sight to blind eyes does not make them capable ofvision.

## Judge Explanation (Gemma-3-27B-it):

The student’s answer incorrectly interprets the reference of the pronoun "<sub>σφε</sub>ῖ<sub>ς</sub>". Instead of identifying the word it refers to (i.e., "ἐπαγγελλόμενοί"), it focuses on the concept of the word "ἐπιστήμης" and attempts an interpretation ofthe excerpt. The understanding of the excerpt is partially correct, but the answer does notfocus on what the question asked. Therefore, it receives a low grade.

Score: 0.25

## F.2 Metric Discrepancy: Prot-Ex Mathematics

This example highlights the discrepancy between lexical metrics and reasoning-based evaluation. The student model (Llama-8B) provides a completely incorrect final answer without demonstrating any logical steps. While the judge model correctly identifies the error and assigns a score of 0.0, BERTScore assigns a high score of ∼0.71 due to high vocabulary and numerical overlap with the target solution.

## Question ID: math\_gym\_2016\_1\_10.2

## Question:

Orpheus and Hypatia have a total of 27 candies together. Orpheus gave some of his candies to Hypatia. Hypatia now has twice as many as she had,

while Orpheus now has three more than Hypatia. b) How many candies did each have initially?

## Target:

12 : 2 = 6 candies for Hypatia, 27 - 6 = 21 candies for Orpheus.

Alternatively (for a and b):

2 parts Hypatia and 2 parts + 3 candies Orpheus.   
27 - 3 = 24, 24 : 4 = 6.

Now: Hypatia 2 · 6 = 12 candies and Orpheus 2 · 6 + 3 = 15 candies.

Initially: Hypatia 6 candies and Orpheus 27 - 6 = 21 candies.

Student Answer (Llama-8B):   
Answer: 15 candies for Orpheus and 12 candies for Hypatia.

Judge Explanation (Gemma-3-27B-it): The answer is completely incorrect. The student does not seem to have understood the description of the problem and has not followed any logical path to find the solution. The answer 15 and 12 does not satisfy any ofthe conditions given in the problem.

LLM Judge Score: 0.0   
BERTScore: 0.71

## F.3 Model Comparison: Prot-Ex Modern Greek

This example illustrates the performance gap between the Greek-adapted KriKri-8B and its base foundation model, Llama-8B. While KriKri-8B correctly identifies a correct synonym that fits the semantic context, Llama-8B struggles with the nuances of the Greek language, selecting a contextually inappropriate verb that alters the meaning of the sentence, resulting in a lower score from the LLM-as-a-Judge (Gemma-3-27B-it).

Question ID: greek\_language\_gym\_2016\_1\_2.4

## Question:

In the following phrase, replace the underlined word with another word without changing the meaning: "The operating hours of’Hellenic Cosmos’ may change [<sub>α</sub>λλάξ<sub>ει</sub>] without notice."

## Target:

The operating hours of ’Hellenic Cosmos’ may be altered/modified [μεταβληθεί/τροποποιηθεί] without notice.

Student Answer (KriKri-8B): The phrase with the replacement of the underlined word is: "The operating hours of ’Hellenic Cosmos’ may be modified [<sub>τροποποιη</sub>θ<sub>ε</sub>ί] without notice." The word "modified" maintains the meaning of the possibility ofaltering/modifying the schedule without prior notice.

## Judge Explanation for KriKri-8B:

The answer is flawless. The student selected the correct synonym ("<sub>τροποποιη</sub>θεί") for the underlined word, fully maintaining the meaning of the sentence. There are no spelling or syntax errors. Excellentjob!

Score (KriKri-8B): 1.0

## Student Answer (Llama-8B):

Answer: "The operating hours of ’Hellenic Cosmos’ may change without notice."

Replacement: "The operating hours of ’Hellenic Cosmos’ may transit/move [<sub>μετα</sub>β<sub>ε</sub>ί] without notice."

Explanation: The word "transit/move" means to change location, time, etc., and is appropriate for the required case.

## Judge Explanation for Llama-8B:

The answer, although attempting to provide an alternative word, is not the most appropriate. The word "μεταβεί" [transit/move] has a different nuance and does not exactly convey the meaning of changing the schedule. The use of words like "μεταβληθεί" or "τροποποιηθεί" [altered/modified] would be preferable. Therefore, the answer is part of the solution, but not fully satisfactory.

Score (Llama-8B): 0.50

## G Synthetic Examples used in Few-Shot Prompt

## G.1 Prot-Ex: Modern Greek Language (Matching)

## Question (Original Greek):

Κείμενο 2: Το όνειρο του ΄Αρη. Ο ΄Αρης είναι το βασικό στήριγμα στην τετραμελή οικογένειά του. Οι γονείς του διατηρούν έναν παραδοσιακό φούρνο στο χωριό και καθημερινά αναλαμβάνει τον ρόλο του ταμία για να τους εξυπηρετεί. Φρον- τίζει ενεργά για τις παραδόσεις των παραγγελιών, μιλάει με ευγένεια στους πελάτες και, παράλληλα, πηγαίνει στις προπονήσεις του. Εκεί, ο προ- πονητής του αντιλαμβάνεται τις δυνατότητές του στις ταχύτητες, τον παροτρύνει να ενταχθεί στην τοπική ομάδα και σιγά-σιγά τον καθοδ- ηγεί να βελτιώσει τους χρόνους του. Καθώς οι επιδόσεις του εξελίσσονται, ο προπονητής τού παρουσιάζει μία διαφορετική πρόταση που δεν είχε τολμήσει να σκεφτεί ποτέ. Του προτείνε- ται να συμμετάσχει στο πανελλήνιο πρωτάθλημα στην Αθήνα, όπου η διάκριση θα μπορούσε να του προσφέρει μια θέση σε μεγάλο σύλλογο και μια λαμπρή καριέρα στον αθλητισμό. Ο ΄Αρης θέλει να κάνει το όνειρό του πραγματικότητα, αλλά δε νιώθει έτοιμος να αποχωριστεί τους δικούς του, που βασίζονται τόσο πολύ πάνω του. Γράψε τις φράσεις (1-5) στη στήλη (Α-Γ) στην οποία ταιριάζει η καθεμιά, σύμφωνα με το κείμενο 2:

Α. Οικογενειακή επιχείρηση Β. Αθλητική δραστηριότητα Γ. Μελλοντική σταδιοδρομία

1. αναλαμβάνει τον ρόλο του ταμία

2. Φροντίζει ενεργά για τις παραδόσεις των παραγγελιών

3. τον παροτρύνει να ενταχθεί στην τοπική ομάδα   
4. λαμπρή καριέρα στον αθλητισμό

5. δε νιώθει έτοιμος να αποχωριστεί τους δικούς του

Target: A-1, A-2, A-5, B-3, Γ-4

## Question (English Translation):

Text 2: Aris’s dream. Aris is the main pillar ofhis four-memberfamily. His parents run a traditional bakery in the village, and every day he takes on the role of cashier to help them. He actively takes care of order deliveries, speaks politely to customers, and, at the same time, goes to his training sessions. There, his coach recognizes his potential in sprinting, encourages him to join the local team, and gradually guides him to improve his times. As his performance evolves, the coach presents him with a different proposal he had never dared to think about. He is suggested to participate in the national championship in Athens, where a distinction could offer him a position in a major club and a brilliant career in sports. Aris wants to make his dream come true, but he doesn’tfeel ready to part with hisfamily, who rely on him so much.

Write the phrases (1-5) in the column (A-C) they   
match, according to Text 2:   
A. Family business   
B. Sports activity   
C. Future career

1. takes on the role of cashier   
2. actively takes care oforder deliveries   
3. encourages him to join the local team   
4. brilliant career in sports   
5. doesn’tfeel ready to part with hisfamily

Target (English Translation): A-1, A-2, A-5, B-3, C-4

## G.2 Prot-Ex: Mathematics (Multiple Choice)

## Question (Original Greek):

Η Μαρία στα διαγωνίσματα της Ιστορίας έχει πάρει τις εξής βαθμολογίες: 14, 17, 15, 16. Πόσο πρέπει να πάρει στο 5ο διαγώνισμα για να βγάλει <sub>μ</sub>έ<sub>σο</sub> ό<sub>ρο</sub> 16; Α. 15, Β. 16, Γ. 18, Δ. 19, Ε. 20

Target: Γ

## Question (English Translation):

Maria has received the following grades in her History exams: 14, 17, 15, 16. What score must she get on the 5th exam to achieve an average of 16?   
A. 15, B. 16, C. 18, D. 19, E. 20

Target (English Translation): C

## G.3 Pan-Ex: Ancient Greek (Fill in the gaps)

## Question (Original Greek):

Να συμπληρώσετε τις παρακάτω περιόδους λό- γου με ουσιαστικά ετυμολογικά συγγενή (απλά ή σύνθετα) της μετοχής «λαμβάνοντας» ώστε να ολοκληρωθεί σωστά το νόημά τους: Η ......... του νέου εργαστηριακού εξοπλισμού θα γίνει την ερχόμενη Δευτέρα.

Target: παραλαβή

## Question (English Translation):

Fill in the following sentences with nouns etymologically related (simple or compound) to the participle "λαμβάνοντας" (receiving) so that their meaning is correctly completed: The ......... of the new laboratory equipment will take place next Monday.

Target (English Translation): παραλαβή (receipt)

## G.4 Pan-Ex: Computer Science (Fill in the gaps)

## Question (Original Greek):

Δίνεται τετραγωνικός πίνακας ακεραίων A[50, 50]. Το παρακάτω τμήμα αλγορίθμου ελέγχει αν ο πίνακας είναι συμμετρικός ως προς την κύρια διαγώνιό του (δηλαδή αν για κάθε στοιχείο <sub>τ</sub>ου ι<sub>σ</sub>χύει A[i, j] = A[j, i]) χ<sub>ρησ</sub>ιμο<sub>π</sub>οιών<sub>τ</sub>ας μια λογική μεταβλητή. Αν βρεθεί έστω και ένα ζευγάρι στοιχείων που να παραβιάζει αυτή τη συνθήκη, η διαδικασία του ελέγχου διακόπτεται. Να γράψετε στο τετράδιό σας τους αριθμούς (1) έως (5) που αντιστοιχούν στα κενά του τμήματος αλγορίθμου και δίπλα ό,τι πρέπει να συμπληρω- θεί, έτσι ώστε να επιτελεί τη λειτουργία που περιγράφηκε.

Συ<sub>μμ</sub>ε<sub>τρ</sub>ικός ← ...(1)...   
i ← 2   
ΟΣΟ i ≤ 50 ΚΑΙ Συμμετρικός = ...(2)... ΕΠΑΝΑΛΑΒΕ   
j ← 1   
ΟΣΟ j < ...(3)... ΚΑΙ Συμμετρικός = ΑΛΗΘΗΣ ΕΠΑΝΑΛΑΒΕ   
ΑΝ A[i, j] ̸= A[...(4)...] ΤΟΤΕ   
Συμμετρικός ← ...(5)...   
ΑΛΛΙΩΣ   
j ← j + 1   
ΤΕΛΟΣ\_ΑΝ   
ΤΕΛΟΣ\_ΕΠΑΝΑΛΗΨΗΣ   
i ← i + 1   
ΤΕΛΟΣ\_ΕΠΑΝΑΛΗΨΗΣ   
Target:   
(1) ΑΛΗΘΗΣ, (2) ΑΛΗΘΗΣ, (3) i, (4) j, i, (5)   
ΨΕΥΔΗΣ

## Question (English Translation):

An integer square matrix A[50, 50] is given. The following algorithm snippet checks ifthe matrix is symmetric with respect to its main diagonal (i.e., if for every element A[i, j] = A[j, i]) using a boolean variable. Ifeven one pair ofelements violates this condition, the checking process stops. Write in your notebook the numbers (1) to (5) corresponding to the blanks in the algorithm snippet and next to them what needs to befilled in, so that it performs the describedfunction.

Symmetric ← ...(1)...   
i ← 2   
WHILE i ≤ 50 AND Symmetric = ...(2)... DO j ← 1   
WHILEj < ...(3)... AND Symmetric = TRUE DO IF A[i, j] ̸= A[...(4)...] THEN   
Symmetric ← ...(5)...   
ELSE   
j ← j + 1   
END\_IF   
END\_WHILE   
i ← i + 1   
END\_WHILE

Target (English Translation): (1) TRUE, (2) TRUE, (3) i, (4) j, i, (5) FALSE

## H Few-Shot Degradation in Structured Tasks (Llama-8B)

## H.1 Pan-Ex Latin

This example demonstrates the model completely losing its output language consistency. Despite the prompt and all 5 few-shot examples being strictly in Greek, the model hallucinates and answers in English. Furthermore, it incorrectly identifies the grammatical case (identifying it as genitive instead of dative).

Question ID: latin\_gel\_2025\_Δ1.<sub>α</sub>.3

Question (Original Greek): Να αναγνωρίσετε συντακτικά τον παρακάτω όρο: ancillae: <sub>ε</sub>ί<sub>ναι</sub> ................. <sub>στο</sub> ...........

Target: έμμεσο αντικείμενο στο ρήμα credidi

Model Output (Llama-8B 5-shot): genitive

Question (English Translation): Syntactically identify the following term: ancillae: it is ................. to the .................

Target (English Translation): indirect object to the verb credidi

Model Output (English): genitive

## H.2 Pan-Ex Biology

In this matching task, the model fails to process the logical constraints of the prompt. Instead of correctly mapping the items, it falls into a repetitive generation loop (babbling), assigning the choice "<sub>γ</sub>" to almost all items in the array.

Question ID: biology\_gel\_2021\_B1

Target Format: ["1-α", "2-γ", "3-β", "4-α", "5-γ", "6-β", "7-α"]

## Model Output (Llama-8B 5-shot):

["1-γ", "2-β", "3-α", "4-γ", "5-γ", "6-γ", "7-γ"]

## H.3 Pan-Ex Ancient Greek

In this fill-in-the-gaps task, the model is explicitly instructed to provide a word etymologically related to the participle "<sub>φ</sub>έ<sub>ροντας</sub>" (carrying/bearing). Overwhelmed by the few-shot examples in the context, it completely ignores this primary semantic constraint and simply guesses a contextually plausible but unrelated word ("<sub>φ</sub>όβ<sub>ο</sub>" instead of "<sub>φ</sub>έ<sub>ρ-</sub> <sub>σιμο</sub>").

## Question ID: ancient\_greek\_gel\_2025\_B3.4

## Question (Original Greek):

Να συμπληρώσετε τις παρακάτω περιόδους λό- γου με ουσιαστικά ετυμολογικά συγγενή (απλά ή σύνθετα) της μετοχής «φέροντας» ώστε να ολοκληρωθεί σωστά το νόημά τους: Με στενο- χώρησε το .......... του. ΄Ηταν φοβερά προσβλ- ητικό.

## Target:

φέρσιμο

## Model Output (Llama-8B 5-shot):

φόβο

## Question (English Translation):

Fill in the following sentences with nouns etymologically related (simple or compound) to the participle "<sub>φ</sub>έ<sub>ροντας</sub>" (bearing/carrying) so that their meaning is correctly completed: His .......... saddened me. It was terribly offensive.

Target (English Translation):

<sub>φ</sub>έ<sub>ρσιμο</sub> (behavior)

## Model Output (English Translation):

φόβο (fear)