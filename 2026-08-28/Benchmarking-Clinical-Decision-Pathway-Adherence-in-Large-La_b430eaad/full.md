# Benchmarking Clinical Decision Pathway Adherence in Large Language Models

Nuo Chen<sup>1,6</sup> Xinyang Jiang<sup>7</sup> Zilong Wang<sup>7∗</sup> Zhifei Zhang<sup>1</sup> Xiaoye Qu<sup>2</sup> Jiajun Deng<sup>3</sup> Yulan Guo<sup>4,6</sup> Cairong Zhao<sup>1,5∗</sup>

<sup>1</sup>Tongji University <sup>2</sup>Shanghai Artificial Intelligence Laboratory <sup>3</sup>Tongji University School of Medicine <sup>4</sup>Sun Yat-sen University <sup>5</sup>Fuzhou University <sup>6</sup>Shenzhen Loop Area Institute <sup>7</sup>Independent Researcher

## Abstract

Following clinical decision pathways (CDPs) defined by clinical practice guidelines is essential for safe and reliable medical decision-making. However, existing medical large language model (LLM) benchmarks mainly evaluate final-answer accuracy, providing limited evaluation of models’ ability to adhere to guidelines. To address this gap, we introduce MEGA-CDP, a benchmark for evaluating whether medical LLMs can generate guideline-adherent CDPs using provided guidelines as references. MEGA-CDP is constructed from 2,274 English and Chinese clinical practice guidelines through a guidelineto-case pipeline, yielding 42,353 clinical cases with explicit reference CDPs. It supports both single-turn vignette and multi-turn interactive settings, and introduces a CDP-oriented evaluation framework for measuring pathway consistency. Experiments on 16 representative LLMs show that reliable clinical decision support remains challenging for current models, demonstrating the need for CDP-oriented evaluation and the value of MEGA-CDP for advancing guideline adherence in medical LLMs.

## 1 Introduction

Large language models (LLMs) have shown strong potential in medical applications, motivating a growing body of benchmarks for evaluating their medical knowledge and reasoning capabilities (Jin et al. 2021; Pal, Umapathi, and Sankarasubbu 2022; Jin et al. 2019). Meanwhile, LLMs are increasingly being explored for clinical decision-making tasks, including diagnosis, treatment planning, and clinical decision support (Hou et al. 2026; Vatsal, Dubey, and Singh 2026). Unlike isolated question answering, clinical decision-making requires multi-step reasoning over heterogeneous patient evidence. This diference highlights the need to evaluate not only final decisions, but also the processes by which they are produced.

In clinical practice, safe and reliable decision-making requires strict adherence to clinical decision pathways (CDPs) defined by established guidelines (Tso et al. 2017). As illustrated in Figure 1(a), a CDP consists of ordered intermediate steps whose omission or misordering can lead to decisions that appear correct but are clinically unsafe, non-explainable, or dificult to review (Ng et al. 2025; Mienye et al. 2024). This highlights the necessity of CDP adherence for medical LLMs. Moreover, guideline-defined CDPs may vary across regions, healthcare resources, and clinical contexts, and they are frequently updated as medical evidence evolves (Steinberg et al. 2011), requiring models to reason with provided guidelines as references rather than relying solely on internal parametric knowledge. However, existing works largely overlook these two aspects, focusing primarily on final outcomes or general rationale quality. This motivates our goal of evaluating whether LLMs can use provided guidelines as references to generate clinical decisions that faithfully follow guideline-defined CDPs.

To address this limitation, we introduce the Medical Guideline-Adherent Clinical Decision Pathway (MEGA-CDP), a large-scale benchmark for systematically evaluating the ability of LLMs to generate clinical decisions that adhere to guidelines. Specifically, MEGA-CDP comprises 2,274 real-world clinical practice guidelines collected from publicly available English and Chinese sources between 2020 and 2025. We develop a rigorous and automated pipeline that transforms the textual content of each guideline into a reasoning tree, from which guideline-defined CDPs are extracted. The pipeline then synthesizes corresponding clinical cases by traversing each CDP. In this design, each case can be correctly resolved using the full guideline text, which provides all necessary information for decision-making without requiring external knowledge. This formulation yields structured, pathway-aligned data that supports fine-grained evaluation of guideline adherence, while remaining scalable for continuous expansion and updates as real-world guidelines evolve. The resulting dataset contains over 42,353 curated CDPs and corresponding clinical cases.

Evaluating predicted CDPs is challenging because they may difer in length from the reference CDPs, intermediate steps can be expressed in diverse natural language forms, and models may omit necessary steps, introduce redundant steps, or reorder steps, potentially violating the intended decision logic. Currently, there is no standardized method for assessing whether model-generated CDPs align with the groundtruth reference. To tackle this challenge, we propose a CDPoriented evaluation framework that first estimates step-level consistency between predicted and reference steps and then aggregates them to compute pathway-level consistency using dynamic time warping. As shown in Figure 1(c), we applied this evaluation framework to 16 representative LLMs under two task settings: (i) a single-turn vignette setting, where the model produces a full CDP from a summarized clinical case, and (ii) a multi-turn interactive setting, where the model incrementally acquires information and updates its decisions. This evaluation provides fine-grained insights into the models’ ability to produce guideline-adherent CDPs.

![](images/73a0cf213dc0b2f67499194d2b481cec52c3c4ef330a1b08c36440362b2d56df.jpg)  
Figure 1: Overview of MEGA-CDP. (a) Comparison between existing benchmarks and MEGA-CDP. (b) Feature comparison between MEGA-CDP and representative clinical decision-making benchmarks. (c) Representative model performance under the two settings.

## 2 Related Work

Prior work has introduced a range of benchmark datasets for evaluating medical question answering systems. Representative examples include MedQA (Jin et al. 2021), MedMCQA (Pal, Umapathi, and Sankarasubbu 2022), and PubMedQA (Jin et al. 2019), which are formulated as exam-style tasks. Other benchmarks extend this setting to more context-rich scenarios. For instance, emrQA (Pampari et al. 2018) introduces question answering over electronic medical records, while MedDialog (Zeng et al. 2020) incorporates large-scale doctor–patient conversations. CliMedBench (Ouyang et al. 2024) further expands the scope by covering multiple clinical scenarios, tasks, and evaluation metrics. These benchmarks evaluate medical domain knowledge through multiple-choice or open-ended question formats.

Recent eforts have explored the evaluation of clinical decision-making and reasoning processes through dedicated benchmarks. DR.BENCH (Gao et al. 2023) introduces diagnostic reasoning tasks for assessing clinical decision-making. MedJourney (Wu et al. 2024) evaluates decision-making in multi-stage scenarios with progressively revealed patient information, capturing the dynamic nature of clinical decision processes. LLMEval-Med (Zhang et al. 2025) adopts an LLM-as-a-Judge (Zheng et al. 2023) paradigm to enable automated evaluation of reasoning quality. Building on this trend toward reasoning-focused evaluation, subsequent studies have emphasized guideline-based clinical assessment.

MIMIC-CDM (Hager et al. 2024) evaluates an LLM’s ability to generate decisions consistent with authoritative clinical guidelines, while CPGBench (Tan et al. 2026) examines whether LLMs can provide guideline-adherent recommendations in multi-turn conversations. More closely related to our work, MedGUIDE (Li et al. 2025) measures adherence to CDPs explicitly defined by existing guideline flowcharts.

Current studies have begun to examine multi-step reasoning and sequential decision-making in clinical settings, exploring methods that simulate diagnostic workflows, treatment planning, or agent-driven reasoning (Hou et al. 2026; Tang et al. 2024; Yue et al. 2024; Shao et al. 2026). These investigations underscore the critical role of stepwise information integration and decision alignment in realistic medical scenarios. Yet, most existing evaluation frameworks do not explicitly encode guideline-based decision structures, nor do they systematically assess whether generated reasoning follows CDPs. As shown in Figure 1(b), addressing this limitation, MEGA-CDP provides a CDP-oriented benchmark that represents guidelines as CDPs, linking them to structured case vignettes and canonical reasoning sequences, which enables rigorous process assessment of guideline adherence and the fidelity of model-generated reasoning.

## 3 MEGA-CDP

## 3.1 Data Curation Pipeline

We design a two-stage data curation pipeline that first extracts guideline-defined CDPs from clinical practice guidelines and then synthesizes CDP-aligned case vignettes for downstream evaluation.

Guideline Collection. We collected clinical practice guidelines published between 2020 and 2025 from publicly available English and Chinese sources. Text-based guidelines were prioritized, while those dominated by explicit flowcharts or diagram-based representations were excluded, resulting in 2,274 retained guidelines. To facilitate downstream construction, we converted the guideline PDFs into structured Markdown format using PaddleOCR (Cui et al. 2026) for text extraction and Ovis2.5-9B (Lu et al. 2025) for layout understanding and formatting.

![](images/fb2c33800f4579eba830caaf8253806fe647a6bcdbf9b9b9d9bdc6cf216a1dfd.jpg)  
Figure 2: Framework of MEGA-CDP. The framework consists of guideline collection, guideline-to-case construction, downstream task settings, and CDP-oriented evaluation. Clinical guidelines are transformed into reasoning trees, guideline-defined CDPs, and case vignettes to support both single-turn vignette and multi-turn interactive settings. Model-generated CDPs are then evaluated through a two-stage procedure for measuring pathway consistency.

Guideline-to-Case Construction. As shown in Figure 2, we construct the MEGA-CDP dataset through a multi-stage automated pipeline implemented with GPT-5.2 (OpenAI 2026b). The construction process consists of two steps:

(1) Clinical Decision Pathway (CDP) Extraction. Given the raw guideline text, we use LLMs with structured prompting templates to extract guideline-defined CDPs. The extracted decision logic is represented as a tree structure, where each non-leaf node corresponds to a clinical decision condition, each branch represents a possible outcome of the corresponding condition and leads to subsequent decision steps, and each leaf node represents a final decision outcome. We then enumerate all root-to-leaf paths, yielding a total of 42,353 guideline-defined CDPs. Each CDP contains an ordered sequence of decision conditions leading to a specific outcome, and therefore serves as the reference CDP for subsequent evaluation. These CDPs represent guidelinedefined reasoning pathways rather than an exhaustive set of all possible clinical workflows, and are designed to evaluate whether models can reproduce the decision logic specified by guidelines. By converting free-text guideline recommendations into explicit CDPs, the pipeline provides a structured basis for constructing cases whose correct decisions can be traced back to guideline-defined reasoning steps.

(2) Case Vignette Synthesis. Given each extracted CDP, we synthesize a corresponding clinical case vignette with

LLMs. The prompt instructs the model to transform the CDP into a case that is consistent with the pathway and leads to a uniquely determined terminal outcome. The resulting case vignette integrates the information required by the CDP into a natural clinical narrative without explicitly exposing the underlying decision structure. In this way, each vignette is aligned with one reference CDP and contains the necessary information for reaching its associated decision outcome.

Data Quality Control. To ensure the reliability of the constructed data, we conducted quality control through LLMbased sampling inspection and human review. Specifically, generated instances were sampled and examined for consistency among the reasoning tree, CDP, and case vignette. Cases with obvious contradictions, unsupported decision logic, or inconsistencies between the case description and the corresponding pathway were excluded from the dataset. Across the CDPs and case vignettes derived from 500 sampled guidelines, fewer than 0.5% of the CDPs and fewer than 1% of the case vignettes were identified as problematic.

## 3.2 Downstream Task Settings

To evaluate the decision-making processes of LLMs, as shown in Figure 2, we propose two downstream task settings.

Single-Turn Vignette Setting. In the single-turn vignette setting, the input consists of a clinical guideline and a corresponding case vignette. The model is then instructed to follow the guideline and produce a step-by-step predicted CDP together with the final decision outcome. To ensure a consistent descriptive style for CDP comparison, we convert the reference CDP into a sequence of concise reasoning narratives in this setting. This setting evaluates whether the model can derive a complete guideline-adherent CDP from summarized clinical information.

Multi-Turn Interactive Setting. In the multi-turn interactive setting, the model maintains access to the clinical guideline and dialogue history while interacting with a patient-side environment. At each turn, the model asks one question to acquire additional patient information, after which the environment responds according to the clinical case. The interaction continues until the model produces a final decision outcome. Afterwards, the sequence of generated questions is treated as the predicted CDP for evaluation. This setting evaluates whether the model can actively acquire patient information in a way that follows the guideline-defined CDP.

## 3.3 CDP-Oriented Evaluation

We propose a CDP-oriented evaluation method that compares the predicted CDPs generated in downstream task settings with the reference CDPs constructed during data construction. As illustrated in Figure 2, we first assess steplevel consistency between predicted and reference CDP steps, and subsequently aggregate the resulting scores to measure pathway-level consistency.

Step-Level Consistency Scoring. We evaluate consistency at the step level using an LLM-as-a-judge paradigm. Given the task instruction, the predicted CDP, and the reference CDP, the judge model compares each predicted step with the reference steps by assessing whether it covers the same key decision point and preserves the corresponding guidelinedefined logic. Following G-Eval (Liu et al. 2023), the judge model assigns a score on a 5-point Likert scale, and the scores are averaged over five repeated runs to improve reliability. We then normalize the averaged scores to the range of [0, 1] for subsequent pathway-level measurement. Before formal evaluation, we iteratively refined the scoring prompt to improve its agreement with human review. The resulting step-level scores capture fine-grained alignment between the predicted and reference CDPs and serve as the basis for pathway-level consistency measurement.

We considered three candidate judge models, including GPT-5.4 Mini (OpenAI 2026a), Qwen3-8B (Yang et al. 2025), and BERTScore (Zhang et al. 2020), and conducted experiments to compare their scoring behavior. Based on the threshold analysis in Section 4.4, GPT-5.4 Mini showed better alignment with human judgments on the sampled instances, and was therefore selected as the judge model for the main evaluation. Subsequently, we conducted a correlation analysis to assess the agreement among the candidate judge models.

Pathway-Level Consistency Measurement. Building on the step-level consistency scores, we compute a Pathway Consistency Cost (PCC) between the predicted CDP and the reference CDP to assess the extent to which the model’s decisionmaking process adheres to the guideline. We design three candidate methods for pathway-level consistency measurement, including Dynamic Time Warping (DTW), Optimal Transport (OT), and Longest Non-Decreasing Subsequence (LNDS). Based on empirical performance, we select DTW as the primary metric for the main evaluation. The algorithmic details of the alternative metrics, OT and LNDS, are provided in the supplementary materials.

In our benchmark setting, we evaluate whether models can follow guideline-defined CDPs. A model may omit intermediate decisions, introduce additional steps, or deviate from the decision order specified by the guideline, resulting in misalignments between the predicted and reference pathways. Therefore, we formulate pathway comparison as a sequence alignment problem and employ the DTW method, which identifies the optimal alignment between two ordered sequences of potentially diferent lengths while preserving their sequential order. Let $m$ and n denote the number of steps in the reference and predicted CDPs, respectively. After step-level consistency scoring, the scores form a consistency matrix $C \in \mathbb { R } ^ { m \times n }$ , where each entry $C _ { i , j }$ denotes the consistency score between the i-th reference step and the j-th predicted step. We then convert the consistency matrix into a distance matrix $D \in \mathbb { R } ^ { m \times n }$ , which is used as the cost matrix in the DTW formulation, by defining $D _ { i , j } = 1 - C _ { i , j }$ so that larger values correspond to higher alignment costs.

Using this cost matrix, the alignment is represented by a path $\mathcal { P } \overset {  } { = } \{ ( i _ { k } , j _ { k } ) \} _ { k = 1 } ^ { K }$ , where each pair $( i _ { k } , j _ { k } )$ indicates that the $i _ { k } { \cdot } \mathrm { t h }$ reference step is aligned with the $j _ { k } { \cdot } \mathrm { t h }$ predicted step. Here, K denotes the total length of the alignment path. The alignment path satisfies the standard DTW constraints, including boundary conditions, monotonicity, and continuity, so that the relative order of decision steps is preserved. The DTW objective is to find the alignment path $\mathcal { P }$ that minimizes the accumulated alignment cost:

$$
\mathcal { P } ^ { * } = \underset { \mathcal { P } } { \operatorname { a r g m i n } } \sum _ { k = 1 } ^ { K } D _ { i _ { k } , j _ { k } } .\tag{1}
$$

The DTW algorithm enables flexible alignment under sequence variation, allowing one-to-many or many-to-one correspondences between steps while preserving the global ordering of the CDP. As the final metric, we normalize the accumulated alignment cost by the length of the alignment path as

$$
\mathrm { P C C } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } D _ { i _ { k } , j _ { k } } .\tag{2}
$$

The PCC reflects the loss of order alignment between the reference CDP and the predicted CDP, with lower values indicating better alignment.

We analyze the agreement between the primary DTWbased metric and the two alternative metrics in Section 4.5. Additionally, a human preference study is conducted to evaluate whether PCC aligns with human judgments, with details provided in the supplementary materials.

## 4 Experiments

## 4.1 Experimental Setup

To evaluate LLMs’ ability to perform guideline-adherent clinical decision-making, we selected three types of models: (1) proprietary models, including the GPT series (OpenAI 2026c), Claude series (Anthropic 2026), and Gemini series (Google 2026); (2) open-source models, including the DeepSeek series (DeepSeek-AI 2025) , Qwen series (Yang et al. 2025), gpt-oss series (OpenAI 2025), and LLaMA series (Grattafiori et al. 2024); and (3) specialized medical models, including the HuatuoGPT series (Chen et al. 2024) and Baichuan series (Team et al. 2026). Given that MEGA-CDP involves long clinical guidelines in multiple languages, we restricted our evaluation to models that support both English and Chinese inputs and provide a context window of at least 40K tokens. All models were evaluated with their default configurations.

<table><tr><td>Model</td><td>Single-Turn PCC (%) ↓ Acc (%) ↑</td><td>PCC (%) ↓</td><td>Multi-Turn Acc (%) ↑</td><td>SR (%) ↑</td></tr><tr><td colspan="5">Proprietary Models</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>44.61 65.14 28.41</td><td>65.32</td><td>28.73</td><td>78.17</td></tr><tr><td>GPT-5.4</td><td>63.06 32.42</td><td>67.43</td><td>23.73</td><td>74.77</td></tr><tr><td>Claude-Sonnet-4.6</td><td>75.38</td><td>55.92</td><td>45.40</td><td>99.22</td></tr><tr><td colspan="5">Open-source Models</td></tr><tr><td>Qwen3-8B</td><td>47.88 60.94 44.31</td><td>64.34</td><td>26.70</td><td>97.65</td></tr><tr><td>Qwen3-32B Qwen3-235B-A22B</td><td>65.53 41.23 75.81</td><td>60.45 63.66</td><td>31.98 26.64</td><td>99.24 95.44</td></tr><tr><td>Qwen3-30B-A3B-Instruct-2507</td><td>62.97</td><td>61.95</td><td>16.61</td><td>97.44</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507</td><td>34.29 50.17 62.98</td><td>72.61</td><td>22.67</td><td></td></tr><tr><td>Qwen3-Next-80B-A3B-Instruct</td><td>37.26 63.94</td><td>58.27</td><td>22.03</td><td>76.21</td></tr><tr><td>Qwen3-Next-80B-A3B-Thinking</td><td>66.21</td><td>69.15</td><td>26.30</td><td>98.71</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-32B</td><td>51.31 67.01</td><td>82.12</td><td>12.75</td><td>80.01 58.69</td></tr><tr><td>gpt-oss-120b</td><td>36.98 63.10</td><td>64.78</td><td>33.08</td><td></td></tr><tr><td>LLaMA-3.3-70B-Instruct</td><td>39.24 34.71</td><td></td><td>25.58</td><td>91.85</td></tr><tr><td></td><td>62.10</td><td>61.04</td><td></td><td>99.77</td></tr><tr><td colspan="5">Specialized Medical Models 57.56</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>46.56 67.72</td><td>91.73</td><td>4.07</td><td>24.05</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>33.87 66.69</td><td>70.28</td><td>16.61</td><td>74.68</td></tr><tr><td>Baichuan-M3-235B</td><td>42.35</td><td>68.27</td><td>35.10</td><td>98.87</td></tr></table>

Table 1: Performance of evaluated LLMs on single-turn vignette and multi-turn interactive settings. PCC denotes the pathway consistency cost, Acc denotes the outcome accuracy, and SR denotes the success rate. Arrows indicate whether higher (↑) or lower (↓) values are better.

We constructed the test set from 440 clinical practice guidelines sampled from the collected corpus, including 210 Chinese guidelines and 230 English guidelines, yielding 7,458 cases for evaluation. Each test instance consisted of a clinical guideline, a case vignette, the corresponding final decision outcome, and reference CDPs for evaluation. We used the same underlying test instances for both the single-turn vignette setting and the multi-turn interactive setting, while adapting the input format and task instruction to each setting. In addition to PCC, we report outcome accuracy for all models to measure the correctness of the final clinical decision. For the multi-turn setting, we also report the success rate, defined as the proportion of cases in which the model follows the required interaction format and successfully completes the multi-turn dialogue.

## 4.2 Main Results

Table 1 summarizes the performance of all evaluated models under the single-turn vignette and multi-turn interactive settings. Overall, model performance varies substantially across model families and task settings. Proprietary models generally achieve the strongest overall performance, while general open-source models provide acceptable alternatives. In contrast, specialized medical models do not consistently show advantages in understanding and following medical guidelines. Notably, although Baichuan-M3-235B is fine-tuned from Qwen3-235B-A22B and has been further trained to model clinical decision-making processes, its PCC performance degrades under both settings. This suggests that medical specialization does not necessarily translate into better guideline-adherent reasoning and may even compromise the model’s underlying reasoning ability. We also observe that, under the same model family and parameter scale, thinking variants may exhibit weaker pathway consistency than their instruct counterparts, as seen in Qwen3-30B-A3B and Qwen3-Next-80B-A3B. These results provide new insights into the generation of guideline-adherent CDPs in clinical decision-making. Additional results for the candidate metrics and judge models are reported in supplementary materials.

Analysis of the Single-Turn Vignette Setting. In the singleturn vignette setting, GPT-5.4 achieves the best pathway consistency with the lowest PCC of 28.41%. Qwen3-235B-A22B obtains the highest outcome accuracy of75.81%, while Claude-Sonnet-4.6 ofers the best overall trade-of, with a PCC of 32.42% and an accuracy of 75.38%. More broadly, pathway consistency and outcome accuracy do not always change consistently across models. Some models achieve high accuracy despite relatively higher PCC, showing that correct final decisions do not necessarily imply correct reasoning processes. This highlights the necessity of processlevel evaluation beyond outcome accuracy.

We further observe that model scale, model version, and medical fine-tuning all afect single-turn performance. Larger models within the same family generally perform better, but improvements across generations are not strictly monotonic. Medical fine-tuning can help in some cases, but its efect is not consistent across medical models. These findings suggest that guideline-adherent reasoning depends on multiple factors beyond final-answer capability.

![](images/4365cd17e0cbe8bdb9eb6f5037622e726ed930f185a6c871eb97660c8041b900.jpg)  
Figure 3: Example of pathway-level inconsistency in guideline-adherent reasoning. Although the model reaches the correct final decision, its predicted CDP still deviates from the reference CDP through reversal, skipping, omission, and redundancy.

Analysis of the Multi-Turn Interactive Setting. The multiturn interactive setting is substantially more challenging than the single-turn vignette setting. Success rates vary considerably across models, reflecting diferences in their ability to follow interaction instructions. Compared with the singleturn setting, PCC increases by 26.98 percentage points on average, while outcome accuracy decreases by 40.51 percentage points. This overall performance degradation is consistent with prior findings on the dificulty of multi-turn clinical decision-making (Hager et al. 2024; Laban et al. 2025). Claude-Sonnet-4.6 achieves the best overall performance in this setting, with the lowest PCC of 55.92% and the highest outcome accuracy of 45.40%.

Model-generated CDPs show more severe process deviations in the multi-turn setting. Figure 3 shows an example from Claude-Sonnet-4.6, the best-performing model in the multi-turn setting. Although the model reaches the correct final decision, its predicted CDP still deviates from the reference CDP. In this example, the predicted CDP exhibits four common types of process deviations, including omission, redundancy, skipping, and reversal. These errors indicate that multi-turn clinical inquiry remains challenging, as current LLMs may reach the correct outcome while still failing to produce a guideline-adherent CDP.

Specialized medical models also show inconsistent efects in this setting. Baichuan-M3-235B achieves better outcome accuracy and success rate than similar-scale general models, but its PCC is worse. This may reflect a mismatch between its clinical decision-making fine-tuning and our evaluation setup, where models must generate CDPs by using the provided guideline as a reference. In contrast, HuatuoGPT-o1 degrades more broadly across multi-turn metrics. These results suggest that current medical fine-tuning does not consistently improve the generation of guideline-adherent CDPs.

## 4.3 Agreement between PCC and Human Preferences

To assess whether PCC aligns with human judgments, we conduct a pairwise human preference study. Given a reference pathway and two model-generated pathways, two human annotators independently select the pathway more consistent with the reference. We compare aggregated human preferences with PCC-based pairwise decisions and calculate agreement rates.

We randomly sample 100 pathway pairs from the multiturn setting results across diferent models. Since low-quality model outputs may introduce ambiguity into human judgments, we additionally sample another 100 pairs from two relatively strong-performing models, Claude-Sonnet-4.6 and Qwen3-Next-80B-A3B-Instruct. Both groups contain equal numbers of English and Chinese cases.

Our human preference study shows strong agreement between PCC and human judgments, with agreement rates of 88.5% and 91.5% for the two groups, respectively, and an overall agreement rate of 90.0% across all 200 pairs. The agreement between the two human annotators is also 90.0%, indicating that PCC achieves a level of consistency comparable to human agreement. These results suggest that PCC provides a reliable measure of pathway consistency that is aligned with human judgments.

![](images/808c0f51b62f238cf85053c529875ed09106d1d9a7bb58533b4a86430283938f.jpg)  
Figure 4: Threshold experiment results for step-level matching. The seven models labeled A through G in the figure are, respectively, Qwen3-Next-80B-A3B-Instruct, Qwen3- Next-80B-A3B-Thinking, Qwen3-8B, Qwen3-235B-A22B, HuatuoGPT-o1-7B, HuatuoGPT-o1-72B, and Baichuan-M3- 235B.

## 4.4 Threshold Analysis for Step-Level Consistency Scores

Step-level consistency scores provide a graded measure of how well a predicted step matches a reference step. However, diferent judge models may exhibit diferent calibration behaviors when assessing consistency, making their scores not directly comparable under a single universal threshold, even though the scoring criteria are explicitly defined in the prompt. We therefore design an analysis to determine appropriate thresholds for step-level matching that align judge model scores with human judgments.

Specifically, we sample prediction results from seven representative models and perform human annotation. For each reference step in the model predictions, we obtain human judgments on whether its best-matched predicted step corresponds to a correct match. We then compute the F1 score of the binary matching results under a range of candidate thresholds at fixed intervals. Around the threshold that achieves the highest F1 score, we identify a stable region as the candidate threshold range. The results are shown in Figure 4. Considering both the optimal threshold and the common stable region, we set the matching thresholds to 0.7 for GPT-5.4 Mini, 0.9 for Qwen3-8B, and 0.2 for BERTScore.

## 4.5 Correlation Analysis of Metrics and Judge Models

To validate the generality of our evaluation framework, we perform a correlation analysis across diferent pathway-level consistency metrics and judge models. Since LNDS has opposing directionality relative to DTW and OT, its scores are inverted. Table 2 reports Pearson correlation r, Spearman rank correlation $\rho ,$ and Kendall rank correlation τ for pairwise comparisons, as well as overall agreement measured by Kendall’s W.

When fixing the metric, GPT-5.4 Mini and Qwen3-8B, as candidate LLM-as-a-judge models, exhibit strong correlations across all metrics, whereas BERTScore shows moderate agreement with the other two. When fixing the judge model, DTW and OT, which leverage global alignment information, show higher pairwise consistency, while LNDS also demonstrates reasonable agreement. Both groups achieve high Kendall’s W, suggesting that the evaluation results remain largely consistent across metric and judge model choices.

<table><tr><td>Setting</td><td>Pair</td><td>r</td><td>ρ</td><td>T</td><td>W</td></tr><tr><td rowspan="2">DTW</td><td>GPT vs Qwen</td><td>0.79</td><td>0.79</td><td>0.60</td><td rowspan="2">0.70</td></tr><tr><td>BERT vs GPT Qwen vs BERT</td><td>0.48 0.39</td><td>0.46 0.38</td><td>0.32 0.26</td></tr><tr><td rowspan="2">OT</td><td>GPT vs Qwen</td><td>0.80</td><td>0.79</td><td>0.60</td><td rowspan="2">0.71</td></tr><tr><td>BERT vs GPT Qwen vs BERT</td><td>0.50 0.43</td><td>0.49</td><td>0.34</td></tr><tr><td>LNDS</td><td>GPT vs Qwen BERT vs GPT Qwen vs BERT</td><td>0.72 0.52</td><td>0.43 0.72 0.53</td><td>0.29 0.58 0.41</td><td>0.71</td></tr><tr><td>GPT</td><td>DTW vs OT LNDS vs DTW OT vs LNDS</td><td>0.47 0.88 0.82 0.76</td><td>0.47 0.86 0.82</td><td>0.36 0.70 0.66</td><td>0.87</td></tr><tr><td>Qwen</td><td>DTW vs OT LNDS vs DTW OT vs LNDS</td><td>0.86 0.73 0.71</td><td>0.75 0.83 0.73 0.71</td><td>0.59 0.66 0.57 0.55</td><td>0.84</td></tr><tr><td>BERT</td><td>DTW vs OT LNDS vs DTW OT vs LNDS</td><td>0.94 0.71 0.69</td><td>0.93 0.71 0.68</td><td>0.79 0.54 0.51</td><td>0.84</td></tr></table>

Table 2: Correlation analysis of metrics and judge models. Pearson $r ,$ Spearman $\rho ,$ Kendall $\tau ,$ and Kendall’s W are reported for pairwise comparisons. The table is divided into groups fixing either the metric or the judge model.

These results demonstrate the feasibility and robustness of the selected evaluation components. Based on this analysis, we choose GPT-5.4 Mini as the default judge model and DTW as the primary metric, while also providing alternative judge models and metrics for experimental use.

## 5 Conclusion

In this paper, we introduced MEGA-CDP, a large-scale benchmark for evaluating whether medical LLMs can generate guideline-adherent CDPs. MEGA-CDP includes an automated guideline-to-case construction pipeline that extracts guideline-defined CDPs from real-world guidelines and synthesizes pathway-aligned clinical cases. Built on this pipeline, MEGA-CDP supports both single-turn vignette and multi-turn interactive settings with a CDP-oriented evaluation framework. Experiments on 16 representative LLMs show that outcome accuracy and pathway consistency capture diferent aspects of clinical decision-making, that both task settings remain substantially challenging, and that specialized medical models do not consistently outperform general-purpose models. These findings highlight the importance of CDP-oriented evaluation and demonstrate the value of MEGA-CDP for developing more reliable and guidelineadherent medical LLMs.

## Acknowledgments

This work was supported by the Shanghai Municipal Special Program for Basic Research on General AI Foundation Models (Grant No. 2025SHZDZX025G10), in collaboration with Shanghai Artificial Intelligence Laboratory. This work was supported by the Shenzhen Loop Area Institute under grant FPF10120260001

## References

Anthropic. 2026. System Card: Claude Sonnet 4.6. https://www-cdn.anthropic.com/ 78073f739564e986f3e28522761a7a0b4484f84.pdf. Accessed: 2026-05-19.

Chen, J.; Cai, Z.; Ji, K.; Wang, X.; Liu, W.; Wang, R.; Hou, J.; and Wang, B. 2024. HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs. arXiv:2412.18925.

Chizat, L.; Peyré, G.; Schmitzer, B.; and Vialard, F.-X. 2017. Scaling Algorithms for Unbalanced Transport Problems. arXiv:1607.05816.

Cui, C.; Sun, T.; Liang, S.; Gao, T.; Zhang, Z.; Liu, J.; Wang, X.; Zhou, C.; Liu, H.; Lin, M.; Zhang, Y.; Zhang, Y.; Liu, Y.; Yu, D.; and Ma, Y. 2026. PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing. arXiv:2601.21957.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

Frogner, C.; Zhang, C.; Mobahi, H.; Araya, M.; and Poggio, T. A. 2015. Learning with a Wasserstein Loss. Advances in Neural Information Processing Systems, 28.

Gao, Y.; Dligach, D.; Miller, T.; Caskey, J.; Sharma, B.; Churpek, M. M.; and Afshar, M. 2023. DR.BENCH: Diagnostic Reasoning Benchmark for Clinical Natural Language Processing. Journal ofBiomedical Informatics, 138: 104286.

Google. 2026. Gemini 3.1 Pro Model Card. https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf. Accessed: 2026-05-19.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The Llama 3 Herd of Models. arXiv:2407.21783.

Hager, P.; Jungmann, F.; Holland, R.; Bhagat, K.; Hubrecht, I.; Knauer, M.; Vielhauer, J.; Makowski, M.; Braren, R.; Kaissis, G.; et al. 2024. Evaluation and mitigation of the limitations of large language models in clinical decisionmaking. Nature Medicine, 30(9): 2613–2622.

Hou, R.; Xue, D.; Sun, H.; He, P.; Zhang, W.; and Ruan, T. 2026. CDAFlow: Enhancing LLM clinical decision-making through agentic workflow. Expert Systems with Applications, 316: 131806.

Jin, D.; Pan, E.; Oufattole, N.; Weng, W.-H.; Fang, H.; and Szolovits, P. 2021. What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences, 11(14): 6421.

Jin, Q.; Dhingra, B.; Liu, Z.; Cohen, W.; and Lu, X. 2019. PubMedQA: A Dataset for Biomedical Research Question Answering. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2567–2577.

Laban, P.; Hayashi, H.; Zhou, Y.; and Neville, J. 2025. LLMs Get Lost In Multi-Turn Conversation. arXiv:2505.06120.

Li, X.; Gao, M.; Hao, Y.; Li, T.; Wan, G.; Wang, Z.; and Wang, Y. 2025. MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models. arXiv:2505.11613.

Liu, Y.; Iter, D.; Xu, Y.; Wang, S.; Xu, R.; and Zhu, C. 2023. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2511–2522.

Lu, S.; Li, Y.; Xia, Y.; Hu, Y.; Zhao, S.; Ma, Y.; Wei, Z.; Li, Y.; Duan, L.; Zhao, J.; Han, Y.; Li, H.; Chen, W.; Tang, J.; Hou, C.; Du, Z.; Zhou, T.; Zhang, W.; Ding, H.; Li, J.; Li, W.; Hu, G.; Gu, Y.; Yang, S.; Wang, J.; Sun, H.; Wang, Y.; Sun, H.; Huang, J.; He, Y.; Shi, S.; Zhang, W.; Zheng, G.; Jiang, J.; Gao, S.; Wu, Y.-F.; Chen, S.; Chen, Y.; Chen, Q.-G.; Xu, Z.; Luo, W.; and Zhang, K. 2025. Ovis2.5 Technical Report. arXiv:2508.11737.

Mienye, I. D.; Obaido, G.; Jere, N.; Mienye, E.; Aruleba, K.; Emmanuel, I. D.; and Ogbuokiri, B. 2024. A survey of explainable artificial intelligence in healthcare: Concepts, applications, and challenges. Informatics in Medicine Unlocked, 51: 101587.

Ng, I. K.; Goh, W. G.; Teo, D. B.; Chong, K. M.; Tan, L. F.; and Teoh, C. M. 2025. Clinical reasoning in real-world practice: a primer for medical trainees and practitioners. Postgraduate Medical Journal, 101(1191): 68–75.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925.

OpenAI. 2026a. GPT-5.4 mini. https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/. Accessed: 2026-05- 19.

OpenAI. 2026b. Introducing GPT-5.2. https://openai.com/ index/introducing-gpt-5-2/. Accessed: 2026-05-19.

OpenAI. 2026c. Introducing GPT-5.4. https://openai.com/ index/introducing-gpt-5-4/. Accessed: 2026-05-19.

Ouyang, Z.; Qiu, Y.; Wang, L.; De Melo, G.; Zhang, Y.; Wang, Y.; and He, L. 2024. CliMedBench: A Large-Scale Chinese Benchmark for Evaluating Medical Large Language Models in Clinical Scenarios. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 8428–8438.

Pal, A.; Umapathi, L. K.; and Sankarasubbu, M. 2022. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering. In Proceedings of the Conference on Health, Inference, and Learning, 248–260. PMLR.

Pampari, A.; Raghavan, P.; Liang, J.; and Peng, J. 2018. emrQA: A Large Corpus for Question Answering on Electronic Medical Records. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, 2357– 2368.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou, F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou, J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.; Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men, R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren, X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Wang, Z.; Cui, Z.; Zhang, Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Shao, Y.; Tang, X.; Sohn, J.; Chen, J.; Liao, Y.; Zhang, J.; Xiang, J.; Wu, F.; Zhao, Y.; Wu, C.; Shi, W.; Cohan, A.; and Gerstein, M. 2026. MedicalAgentsBench for Complex Medical Reasoning: Comparing Internalized Reasoning Models versus Externalized Agent-based Frameworks. arXiv:2503.07459.

Steinberg, E.; Greenfield, S.; Wolman, D. M.; Mancher, M.; and Graham, R. 2011. Clinical Practice Guidelines We Can Trust. National Academies Press.

Tan, A.; Dai, S.; Wang, J.; Zhou, F.; Lu, Y.; Wang, X.; Chen, Y.; Yang, C.; Liu, S.; and Chen, H. 2026. A Decade-Scale Benchmark Evaluating LLMs’ Clinical Practice Guidelines Detection and Adherence in Multi-turn Conversations. arXiv:2603.25196.

Tang, X.; Zou, A.; Zhang, Z.; Li, Z.; Zhao, Y.; Zhang, X.; Cohan, A.; and Gerstein, M. 2024. MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning. In Findings of the Association for Computational Linguistics: ACL 2024, 599–621.

Team, M.; Dou, C.; Yang, F.; Li, F.; Jia, J.; Ju, Q.; Wang, S.; Li, T.; Zeng, X.; Zhou, Y.; Zhang, H.; Tai, J.; Sun, L.; Guo, P.; Mo, Y.; Wang, X.; Cui, H.; and Zhang, Z. 2026. Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making. arXiv:2602.06570.

Tso, G. J.; Tu, S. W.; Oshiro, C.; Martins, S.; Ashcraft, M.; Yuen, K. W.; Wang, D.; Robinson, A.; Heidenreich, P. A.; and Goldstein, M. K. 2017. Automating Guidelines for Clinical Decision Support: Knowledge Engineering and Implementation. In AMIA Annual Symposium Proceedings, volume 2016, 1189.

Vatsal, S.; Dubey, H.; and Singh, A. 2026. Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents. IEEE Access, 14: 4840–4863.

Wu, X.; Zhao, Y.; Zhang, Y.; Wu, J.; Zhu, Z.; Zhang, Y.; Ouyang, Y.; Zhang, Z.; Wang, H.; Lin, Z.; et al. 2024. MedJourney: Benchmark and Evaluation of Large Language Models over Patient Clinical Journey. Advances in Neural Information Processing Systems, 37: 87621–87646.

Yue, L.; Xing, S.; Chen, J.; and Fu, T. 2024. ClinicalAgent: Clinical Trial Multi-Agent System with Large Language Model-based Reasoning. In Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, 1–10.

Zeng, G.; Yang, W.; Ju, Z.; Yang, Y.; Wang, S.; Zhang, R.;Zhou, M.; Zeng, J.; Dong, X.; Zhang, R.; et al. 2020. Med-

Dialog: Large-scale Medical Dialogue Datasets. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 9241–9250.

Zhang, M.; Shen, Y.; Li, Z.; Sha, H.; Hu, B.; Wang, Y.; Huang, C.; Liu, S.; Tong, J.; Jiang, C.; Chai, M.; Xi, Z.; Dou, S.; Gui, T.; Zhang, Q.; and Huang, X. 2025. LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation. In Christodoulopoulos, C.; Chakraborty, T.; Rose, C.; and Peng, V., eds., Findings of the Association for Computational Linguistics: EMNLP 2025, 4888–4914. Suzhou, China: Association for Computational Linguistics. ISBN 979-8-89176-335-7.

Zhang, T.; Kishore, V.; Wu, F.; Weinberger, K. Q.; and Artzi, Y. 2020. BERTScore: Evaluating Text Generation with BERT. In International Conference on Learning Representations.

Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E.; et al. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36: 46595–46623.

<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Specification</td></tr><tr><td rowspan=1 colspan=2>Hardware Infrastructure</td></tr><tr><td rowspan=1 colspan=1>GPUCPUMemory</td><td rowspan=1 colspan=1>8× NVIDIA H100 80GBIntel Xeon Platinum 8468V1920 GB</td></tr><tr><td rowspan=1 colspan=2>Software Environmen</td></tr><tr><td rowspan=1 colspan=1>Operating SystemCUDAPythonPyTorchSGLang</td><td rowspan=1 colspan=1>Ubuntu 20.04 LTS12.6Python 3.102.9.10.5.10.post1</td></tr></table>

Table 3: Computing infrastructure and software environment used for experiments.

## A Reproducibility Details

All experiments were conducted using a combination of APIbased inference and local model inference. Closed-source models were accessed through their oficial APIs. Opensource models were deployed locally using GPU-based inference. Table 3 summarizes the hardware and software configurations used for local inference and evaluation. For local experiments involving stochastic components, random seeds were fixed to 42 to ensure reproducibility.

## B Alternative Pathway-Level Metrics

## B.1 OT-based Measurement

We model pathway alignment as a global matching problem using OT, which establishes correspondences between steps by considering their overall matching cost without enforcing strict sequential alignment. Given a step-level consistency matrix $\boldsymbol { \dot { C } } \in \mathbb { R } ^ { m \times n }$ , we convert it into a consistency-based distance matrix $D ^ { c o n } \in \mathbb { R } ^ { m \times n }$ by defining $D _ { i , j } ^ { c o n } = 1 \mathrm { ~ - ~ }$ $C _ { i , j }$ . Since the OT formulation does not explicitly model the ordering of steps, we introduce a relative positional penalty $D ^ { p o s } \in \mathbb { R } ^ { m \times \bar { n } }$ to encourage order-aware alignment. This penalty is defined as the absolute diference between the normalized positions of the i-th reference step and the j-th predicted step:

$$
D _ { i , j } ^ { p o s } = \left| \frac { i - 0 . 5 } { m } - \frac { j - 0 . 5 } { n } \right| .\tag{3}
$$

We then combine the consistency-based distance and positional penalty into a unified transport cost matrix:

$$
D _ { i , j } = \frac { \alpha D _ { i , j } ^ { c o n } + \beta D _ { i , j } ^ { p o s } } { \alpha + \beta } ,\tag{4}
$$

where $\alpha = 1 . 0$ controls the contribution of the consistencybased distance, while $\beta$ controls the contribution of positional alignment and is set to 0.2 for GPT-5.4 Mini and Qwen3-8B, and 0.1 for BERTScore.

We define a transport plan $T \in \mathbb { R } ^ { m \times n }$ , where each entry $T _ { i , j }$ represents the amount of mass transported from the i-th reference step to the j-th predicted step. Let $\mathbf { a } \in \mathbb { R } ^ { m }$ and b $\in \mathbb { R } ^ { n }$ denote the marginal distributions over reference and predicted steps, respectively, which we set to uniform distributions, i.e., $\begin{array} { r } { \mathbf { a } _ { i } \ = \ \frac { 1 } { m } } \end{array}$ and $\begin{array} { r } { \mathbf { b } _ { j } ~ = ~ \frac { 1 } { n } } \end{array}$ . Instead of standard balanced OT, we adopt an unbalanced OT formulation, which relaxes the exact marginal constraints and allows partial mass variation during transport, making it more suitable for pathway alignment. The resulting optimization objective is to find a transport plan that minimizes the total transport cost while softly enforcing the marginal constraints. Specifically, we adopt a KL-regularized formulation (Frogner et al. 2015; Chizat et al. 2017):

$$
\begin{array} { r l } & { \underset { T \geq 0 } { \operatorname* { m i n } } \langle T , D \rangle _ { F } + \lambda H ( T ) } \\ & { \qquad + \tau _ { a } \mathrm { K L } ( T \mathbf { 1 } _ { n } , \mathbf { a } ) + \tau _ { b } \mathrm { K L } ( T ^ { \top } \mathbf { 1 } _ { m } , \mathbf { b } ) , } \end{array}\tag{5}
$$

where $H ( T )$ is an entropic regularization term, the KL divergence terms softly penalize deviations from the marginal distributions a and b, and $\lambda , \tau _ { a }$ and $\tau _ { b }$ control the strength of entropy regularization and marginal relaxation.

This formulation captures global alignment between pathways by allowing flexible many-to-many correspondences while softly encouraging positional consistency. As the final metric, we compute the pathway consistency cost induced by the optimal transport plan as

$$
\operatorname { P C C } _ { \mathrm { O T } } = \frac { \langle T , D \rangle _ { F } } { \| T \| _ { 1 } } .\tag{6}
$$

Here, $\langle T , D \rangle _ { F }$ denotes the total transport cost weighted by the transport plan, and $\| T \| _ { 1 }$ normalizes the cost by the total transported mass. Lower $\mathrm { P C C _ { O T } }$ indicates better alignment between the reference CDP and the predicted CDP.

## B.2 LNDS-based Measurement

In addition to global alignment strategies, we consider a lightweight approach that directly evaluates order consistency using the Longest Non-Decreasing Subsequence (LNDS). Unlike DTW and OT, which allow flexible matching between steps, this approach focuses on identifying the largest subset of matched steps that can be aligned while strictly preserving the decision order.

Let $s _ { i } ^ { * }$ and $s _ { j }$ denote the i-th reference step and the j-th predicted step from the reference CDP and predicted CDP, respectively. Starting from the step-level consistency matrix $C \in \mathbb { R } ^ { m \times n }$ , we derive a candidate alignment for each reference step $s _ { i } ^ { * }$ by selecting the predicted step index $j _ { i }$ with the highest consistency score. As described in Section 4.4, we introduce a threshold $\gamma$ to avoid spurious matches caused by low-confidence alignments, and retain only matches satisfying $C _ { i , j _ { i } } \geq \gamma$ . Reference steps below the threshold are treated as unmatched. This produces a sequence of matched predicted-step indices ordered by the reference CDP. We then compute the longest non-decreasing subsequence of this sequence, which represents the largest subset of matched steps that preserves the expected decision order.

The resulting subsequence length reflects how consistently the predicted CDP follows the correct decision order. As the final metric, we normalize the subsequence length as

$$
\mathrm { P C S } _ { \mathrm { L N D S } } = \frac { m ^ { \prime } } { m } ,\tag{7}
$$

![](images/b53c4e55c865482aacf90150f5ca836e826a2ceacf6b59e549d0c4bb81487970.jpg)

Figure 5: Distribution of reference CDP lengths. The inset shows a magnified view of the long-tail region.
<table><tr><td rowspan=1 colspan=1>Statistic</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Non-Test</td></tr><tr><td rowspan=1 colspan=1>Mean</td><td rowspan=1 colspan=1>6.51</td><td rowspan=1 colspan=1>6.12</td><td rowspan=1 colspan=1>6.59</td></tr><tr><td rowspan=1 colspan=1>Std. Dev.</td><td rowspan=1 colspan=1>3.74</td><td rowspan=1 colspan=1>3.18</td><td rowspan=1 colspan=1>3.84</td></tr><tr><td rowspan=1 colspan=1>Min</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>P5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>P25</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Median</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>P75</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>P95</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>13</td></tr><tr><td rowspan=1 colspan=1>Max</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>62</td></tr></table>

Table 4: Summary statistics of guideline-defined CDP lengths across dataset splits.

where $m ^ { \prime }$ denotes the length of the longest non-decreasing subsequence and m denotes the number of reference steps. A higher $\mathrm { P C S } _ { \mathrm { L N D S } }$ indicates better adherence to the guidelinedefined decision order.

## C Statistics of CDP Lengths

We analyze the statistics and distribution of reference CDP lengths to characterize the structural properties of the guideline-defined CDPs. Here, CDP length is defined as the number of non-leaf nodes along a root-to-leaf pathway, corresponding to the number of clinical decision conditions excluding the final decision outcome. Figure 5 presents the distribution of reference CDP lengths across the full CDP set and its test and non-test subsets, with the long-tail region magnified in the inset. Table 4 summarizes the descriptive statistics of reference CDP lengths, including the mean, standard deviation, percentile statistics, and range. The test and non-test subsets exhibit similar CDP length statistics, suggesting that the held-out test set maintains similar structural characteristics to the overall CDP set.

## D Additional Evaluation Results

Tables 5, 6, and 7 report additional pathway consistency results under diferent pathway-level metrics andjudge models, with 95% confidence intervals estimated using 10,000 bootstrap resampling iterations. Across diferent judge models, similar model rankings and relative performance trends are generally preserved, suggesting that the pathway consistency evaluation is robust to variations in judge models. Similarly, DTW-PCC, OT-PCC, and LNDS-PCS exhibit consistent relative trends across models under both task settings, indicating that diferent pathway-level metrics provide broadly consistent assessments despite their diferent mathematical formulations. Moreover, the narrow confidence intervals further indicate that the estimated pathway consistency scores are stable.

## E Prompt Templates

This section provides the English prompt templates used in MEGA-CDP. For readability, we use placeholders such as {guideline} and {case\_vignette} to indicate taskspecific inputs. Tables 8–16 present the complete prompt templates in their original wording.

<table><tr><td>Model</td><td>GPT-5.4 Mini</td><td>DTW-PCC (%) ↓ Qwen3-8B</td><td>BERTScore</td></tr><tr><td colspan="4">Single-Turn Vignette Setting</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>44.61 (44.07–45.16)</td><td>33.67 (33.16–34.19) 18.49 (18.08–18.88)</td><td>71.37 (71.13–71.62) 62.88 (62.61–63.14)</td></tr><tr><td>GPT-5.4 Claude-Sonnet-4.6</td><td>28.41 (27.94–28.86) 32.42 (31.90–32.93)</td><td>24.17 (23.72–24.64)</td><td>68.35 (68.09–68.61)</td></tr><tr><td>Qwen3-8B</td><td>47.88 (47.41–48.35)</td><td>33.66 (33.24–34.09)</td><td>74.07 (73.84–74.29)</td></tr><tr><td>Qwen3-32B</td><td>44.31 (43.81–44.81)</td><td>30.59 (30.15–31.03)</td><td>77.35 (77.13–77.57)</td></tr><tr><td>Qwen3-235B-A22B</td><td>41.23 (40.73–41.74)</td><td>29.14 (28.71–29.58)</td><td>77.51 (77.27–77.75)</td></tr><tr><td>Qwen3-30B-A3B-Instruct-2507</td><td>34.29 (33.82–34.77)</td><td>22.99 (22.60–23.40)</td><td>70.91 (70.68–71.14)</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507</td><td>50.17 (49.63–50.72)</td><td>37.37 (36.84–37.89)</td><td>73.23 (72.99–73.47)</td></tr><tr><td>Qwen3-Next-80B-A3B-Instruct</td><td>37.26 (36.78–37.74)</td><td>25.20 (24.79–25.60)</td><td>71.49 (71.24–71.74)</td></tr><tr><td>Qwen3-Next-80B-A3B-Thinking</td><td></td><td></td><td></td></tr><tr><td></td><td>51.31 (50.76–51.85)</td><td>39.26 (38.72–39.81)</td><td>74.65 (74.40–74.90)</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-32B</td><td>36.98 (36.49–37.45)</td><td>25.95 (25.53–26.36)</td><td>72.03 (71.78–72.27)</td></tr><tr><td>gpt-oss-120b</td><td>39.24 (38.74–39.74)</td><td>26.63 (26.17–27.09)</td><td>74.45 (74.19–74.71)</td></tr><tr><td>LLaMA-3.3-70B-Instruct</td><td>34.71 (34.24–35.17)</td><td>25.24 (24.82–25.66)</td><td>70.79 (70.60–70.98)</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>46.56 (46.09–47.01)</td><td>32.82 (32.40–33.24)</td><td>78.97 (78.79–79.16)</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>33.87 (33.43–34.30)</td><td>23.32 (22.94–23.70)</td><td>71.09 (70.86–71.32)</td></tr><tr><td>Baichuan-M3-235B</td><td>42.35 (41.83–42.88)</td><td>29.80 (29.32–30.28)</td><td>73.91 (73.66–74.17)</td></tr><tr><td colspan="4">Multi-Turn Interactive Setting</td></tr><tr><td>Gemini-3.1-Pro-Preview GPT-5.4</td><td>65.32 (64.67–65.96) 67.43 (66.79–68.06)</td><td>58.93 (58.22–59.66)</td><td>75.47 (75.03–75.90)</td></tr><tr><td>Claude-Sonnet-4.6</td><td>55.92 (55.38–56.46)</td><td>61.42 (60.70–62.14) 46.15 (45.58–46.74)</td><td>77.61 (77.19–78.03)</td></tr><tr><td>Qwen3-8B</td><td>64.34 (63.85–64.82)</td><td></td><td>70.52 (70.18–70.86)</td></tr><tr><td>Qwen3-32B</td><td></td><td>53.63 (53.07–54.17)</td><td>73.84 (73.54–74.14)</td></tr><tr><td>Qwen3-235B-A22B</td><td>60.45 (59.95–60.95)</td><td>47.91 (47.37–48.43)</td><td>74.22 (73.92–74.53)</td></tr><tr><td></td><td>63.66 (63.12–64.19)</td><td>54.69 (54.10–55.28)</td><td>74.64 (74.32–74.97)</td></tr><tr><td>Qwen3-30B-A3B-Instruct-2507</td><td>61.95 (61.44–62.45)</td><td>48.81 (48.28–49.35)</td><td>74.15 (73.85–74.45)</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507</td><td>72.61 (72.07–73.14)</td><td>65.50 (64.85–66.14)</td><td>80.02 (79.65–80.40)</td></tr><tr><td>Qwen3-Next-80B-A3B-Instruct</td><td>58.27 (57.77–58.76)</td><td>47.30 (46.77–47.85)</td><td>75.88 (75.58–76.18)</td></tr><tr><td>Qwen3-Next-80B-A3B-Thinking</td><td>69.15 (68.58–69.73)</td><td>59.45 (58.77–60.11)</td><td>79.96 (79.61–80.32)</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-32B</td><td>82.12 (81.62–82.62)</td><td>78.80 (78.23–79.38)</td><td>88.31 (88.01–88.61)</td></tr><tr><td>gpt-oss-120b</td><td>64.78 (64.25–65.30)</td><td>56.81 (56.22–57.40)</td><td>77.50 (77.19–77.82)</td></tr><tr><td>LLaMA-3.3-70B-Instruct</td><td>61.04 (60.57–61.52)</td><td>46.84 (46.34–47.32)</td><td>82.32 (82.10–82.53)</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>91.73 (91.32–92.15)</td><td>91.44 (91.00–91.88)</td><td>96.78 (96.60–96.96)</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>70.28 (69.70–70.85)</td><td>63.12 (62.44–63.79)</td><td>83.27 (82.92–83.62)</td></tr><tr><td>Baichuan-M3-235B</td><td>68.27 (67.84–68.70)</td><td>54.12 (53.66–54.60)</td><td>81.16 (80.96–81.36)</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 5: DTW-based PCC results with bootstrap confidence intervals. Results are reported across diferent judge models under both single-turn vignette and multi-turn interactive settings. Each entry shows the mean PCC value with its 95% bootstrap confidence interval. Lower values indicate better pathway consistency.

<table><tr><td rowspan=1 colspan=9>Model</td><td rowspan=1 colspan=4>OT-PCC (%) ↓GPT-5.4 Mini          Qwen3-8B           BERTScore</td></tr><tr><td rowspan=1 colspan=8></td><td></td><td rowspan=1 colspan=4>Single-Turn Vignette Setting</td></tr><tr><td rowspan=1 colspan=8>Gemini-3.1-Pro-Preview</td><td></td><td rowspan=1 colspan=4>39.37 (38.88–39.86)  31.41 (30.96–31.87)  73.10 (72.86–73.35)</td></tr><tr><td rowspan=3 colspan=1>GPT-5.4</td><td rowspan=5 colspan=8>Claude-Sonnet-4.6Qwen3-8B</td><td rowspan=3 colspan=1></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=4>22.02 (21.62–22.40)  15.35 (15.02–15.66)  64.44 (64.21–64.69)</td></tr><tr><td></td><td rowspan=2 colspan=4>36.94 (36.56–37.33)  25.72 (25.38–26.06)  74.22 (74.00–74.44)</td></tr><tr><td></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=3>Qwen3-32B</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4>34.51 (34.10–34.91)  24.07 (23.72–24.42)  78.27 (78.06–78.47)</td></tr><tr><td rowspan=5 colspan=8>Qwen3-235B-A22BQwen3-30B-A3B-Instruct-2507</td><td rowspan=2 colspan=1>2B</td><td rowspan=2 colspan=1>-2507</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4>33.94 (33.55–34.35)  24.92 (24.56–25.28)  78.89 (78.65–79.12)</td></tr><tr><td rowspan=1 colspan=3>struct-25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>28.73 (28.35–29.12)  19.99 (19.67–20.33)  71.49 (71.28–71.70)</td></tr><tr><td rowspan=4 colspan=8>Qwen3-Next-80B-A3B-ThinkingDeepSeek-R1-Distill-Qwen-32B</td><td rowspan=1 colspan=1>2507</td><td rowspan=1 colspan=1>07</td><td rowspan=1 colspan=3>41.57 (41.09–42.07)  31.10 (30.64–31.53)  73.98 (73.75–74.21)</td></tr><tr><td></td><td rowspan=1 colspan=4>29.99 (29.60–30.37)  20.63 (20.32–20.95)  71.57 (71.33–71.79)</td></tr><tr><td></td><td rowspan=1 colspan=2>42.75 (42.25–43.25)  32.82 (32.35–33.30)</td><td rowspan=1 colspan=2>75.83 (75.58–76.09)</td></tr><tr><td></td><td rowspan=1 colspan=2>30.35 (29.96–30.75)  21.52 (21.19–21.87)</td><td rowspan=1 colspan=2>73.78 (73.54–74.02)</td></tr><tr><td rowspan=3 colspan=8>gpt-oss-120bLLaMA-3.3-70B-InstructHuatuoGPT-o1-7B</td><td></td><td rowspan=1 colspan=1>31.18 (30.76–31.61)</td><td rowspan=1 colspan=1>21.58 (21.21–21.96)</td><td rowspan=1 colspan=2>76.04 (75.78–76.29)</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>29.23 (28.84–29.63)</td><td rowspan=1 colspan=1>22.23 (21.86–22.59)</td><td rowspan=1 colspan=2>72.60 (72.42–72.78)</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>39.64 (39.24–40.04)</td><td rowspan=1 colspan=1>28.24 (27.88–28.61)</td><td rowspan=1 colspan=2>80.86 (80.68–81.04)</td></tr><tr><td rowspan=4 colspan=8>HuatuoGPT-o1-72BBaichuan-M3-235B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1>27.41 (27.07–27.76)</td><td rowspan=1 colspan=1>19.44 (19.15–19.73)</td><td rowspan=1 colspan=2>72.24 (72.03–72.43)</td></tr><tr><td></td><td rowspan=1 colspan=1>34.52 (34.08–34.98)</td><td rowspan=1 colspan=1>24.73 (24.34–25.13)</td><td rowspan=1 colspan=2>74.84 (74.59–75.08)</td></tr><tr><td rowspan=1 colspan=8></td><td></td><td rowspan=1 colspan=2>Multi-Turn Interactive Setting</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=2 colspan=8>Gemini-3.1-Pro-PreviewGPT-5.4</td><td></td><td rowspan=1 colspan=1>58.12 (57.39–58.85)</td><td rowspan=1 colspan=1>53.28 (52.51–54.05)</td><td rowspan=1 colspan=1>75.15 (74.69–75.60)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>60.41 (59.69–61.12)</td><td rowspan=1 colspan=1>56.13 (55.35–56.89)</td><td rowspan=1 colspan=1>77.14 (76.69–77.57)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=9>Claude-Sonnet-4.6Qwen3-8BQwen3-32B</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>46.63 (46.06–47.19)</td><td rowspan=1 colspan=1>38.95 (38.36–39.55)</td></tr><tr><td rowspan=1 colspan=1>55.19 (54.67–55.73)</td><td rowspan=1 colspan=1>46.13 (45.54–46.72)</td><td rowspan=1 colspan=1>73.50 (73.18–73.83)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>50.91 (50.38–51.45)</td><td rowspan=1 colspan=1>40.08 (39.52–40.63)</td><td rowspan=1 colspan=1>73.98 (73.65–74.30)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=9>Qwen3-235B-A22B</td><td rowspan=1 colspan=1>55.69 (55.11–56.26)</td><td rowspan=1 colspan=1>47.90 (47.27–48.51)</td><td rowspan=1 colspan=1>74.43 (74.08–74.78)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=9>Qwen3-30B-A3B-Instruct-2507Qwen3-30B-A3B-Thinking-2507</td><td rowspan=1 colspan=1>52.40 (51.83–52.96)</td><td rowspan=1 colspan=1>40.81 (40.24–41.36)</td><td rowspan=1 colspan=1>74.56 (74.24–74.88)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>65.82 (65.20–66.43)</td><td rowspan=1 colspan=1>60.00 (59.27–60.72)</td><td rowspan=1 colspan=1>79.80 (79.40–80.20)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=9>Qwen3-Next-80B-A3B-InstructQwen3-Next-80B-A3B-ThinkingDeepSeek-R1-Distill-Qwen-32Bgpt-oss-120b</td><td rowspan=1 colspan=1>48.47 (47.96–48.99)</td><td rowspan=1 colspan=1>39.46 (38.90–40.02)</td><td rowspan=1 colspan=1>75.90 (75.57–76.21)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>61.78 (61.13–62.44)</td><td rowspan=1 colspan=1>53.25 (52.51–53.97)</td><td rowspan=1 colspan=1>79.93 (79.56–80.31)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>78.70 (78.11–79.27)</td><td rowspan=1 colspan=1>75.61 (74.96–76.27)</td><td rowspan=1 colspan=1>89.41 (89.11–89.70)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>56.01 (55.44–56.58)</td><td rowspan=1 colspan=1>49.52 (48.88–50.16)</td><td rowspan=1 colspan=1>77.62 (77.29–77.95)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=9>LLaMA-3.3-70B-InstructHuatuoGPT-o1-7BHuatuoGPT-o1-72B</td><td rowspan=1 colspan=1>50.51 (50.00–51.02)</td><td rowspan=1 colspan=1>37.85 (37.33–38.35)</td><td rowspan=1 colspan=1>84.01 (83.78–84.24)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>90.59 (90.12–91.06)</td><td rowspan=1 colspan=1>90.55 (90.07–91.04)</td><td rowspan=1 colspan=1>97.34 (97.17–97.50)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>63.91 (63.26–64.56)</td><td rowspan=1 colspan=1>57.93 (57.18–58.67)</td><td rowspan=1 colspan=1>84.06 (83.71–84.43)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=9>Baichuan-M3-235B</td><td rowspan=1 colspan=1>56.35 (55.87–56.83)</td><td rowspan=1 colspan=1>43.05 (42.58–43.55)</td><td rowspan=1 colspan=1>82.62 (82.40–82.84)</td><td rowspan=1 colspan=1></td></tr></table>

Table 6: OT-based PCC results with bootstrap confidence intervals. Results are reported across diferent judge models under both single-turn vignette and multi-turn interactive settings. Each entry shows the mean PCC value with its 95% bootstrap confidence interval. Lower values indicate better pathway consistency.

<table><tr><td rowspan="2">Model</td><td colspan="3">LNDS-PCS (%) ↑</td></tr><tr><td>GPT-5.4 Mini</td><td>Qwen3-8B</td><td>BERTScore</td></tr><tr><td colspan="4">Single-Turn Vignette Setting</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>53.01 (52.37–53.65)</td><td>50.97 (50.35–51.58) 75.86 (75.32–76.40)</td><td>67.09 (66.48–67.68) 81.73 (81.28–82.17)</td></tr><tr><td>GPT-5.4 Claude-Sonnet-4.6</td><td>74.09 (73.55–74.65) 67.92 (67.32–68.52)</td><td>64.87 (64.30–65.45)</td><td>74.17 (73.64–74.71)</td></tr><tr><td>Qwen3-8B</td><td>56.20 (55.65–56.76)</td><td>57.58 (57.05–58.11)</td><td>67.22 (66.66–67.77)</td></tr><tr><td>Qwen3-32B</td><td>61.72 (61.14–62.28)</td><td>62.64 (62.09–63.17)</td><td>65.52 (64.92–66.11)</td></tr><tr><td>Qwen3-235B-A22B</td><td></td><td></td><td></td></tr><tr><td>Qwen3-30B-A3B-Instruct-2507</td><td>59.99 (59.39–60.59)</td><td>58.56 (58.00–59.11)</td><td>59.89 (59.22–60.56)</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507</td><td>68.03 (67.46–68.58)</td><td>67.62 (67.08–68.17)</td><td>74.30 (73.80–74.80)</td></tr><tr><td></td><td>47.48 (46.85–48.08)</td><td>49.60 (49.04–50.18)</td><td>63.42 (62.83–64.00)</td></tr><tr><td>Qwen3-Next-80B-A3B-Instruct</td><td>65.28 (64.71–65.85)</td><td>65.62 (65.07–66.15)</td><td>71.97 (71.43–72.51)</td></tr><tr><td>Qwen3-Next-80B-A3B-Thinking</td><td>45.97 (45.35–46.58)</td><td>47.53 (46.93–48.13)</td><td>58.35 (57.72–59.00)</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-32B</td><td>62.89 (62.32–63.50)</td><td>60.98 (60.44–61.55)</td><td>69.41 (68.84–70.01)</td></tr><tr><td>gpt-oss-120b</td><td>59.69 (59.10–60.27)</td><td>63.09 (62.52–63.66)</td><td>64.39 (63.77–65.02)</td></tr><tr><td>LLaMA-3.3-70B-Instruct</td><td>67.53 (66.97–68.11)</td><td>62.50 (61.92–63.09)</td><td>76.22 (75.75–76.70)</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>60.69 (60.11–61.27)</td><td>56.73 (56.17–57.31)</td><td>65.33 (64.72–65.93)</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>72.97 (72.45–73.48)</td><td>71.75 (71.24–72.25)</td><td>78.62 (78.15–79.07)</td></tr><tr><td>Baichuan-M3-235B</td><td>58.18 (57.55–58.78)</td><td>59.19 (58.61–59.78)</td><td>65.11 (64.49–65.72)</td></tr><tr><td colspan="4">Multi-Turn Interactive Setting</td></tr><tr><td>Gemini-3.1-Pro-Preview GPT-5.4</td><td>35.96 (35.24–36.68) 31.85 (31.16–32.55)</td><td>34.54 (33.80–35.29) 30.83 (30.12–31.56)</td><td>49.06 (48.25–49.89) 45.27 (44.46–46.09)</td></tr><tr><td>Claude-Sonnet-4.6</td><td>45.15 (44.51–45.81)</td><td>44.57 (43.88–45.23)</td><td>61.88 (61.25–62.52)</td></tr><tr><td>Qwen3-8B</td><td>36.90 (36.30–37.48)</td><td>37.26 (36.63–37.89)</td><td>57.69 (57.04–58.33)</td></tr><tr><td>Qwen3-32B</td><td>43.82 (43.19–44.44)</td><td>45.12 (44.47–45.76)</td><td></td></tr><tr><td>Qwen3-235B-A22B</td><td>36.74 (36.11–37.38)</td><td></td><td>59.43 (58.79–60.04)</td></tr><tr><td>Qwen3-30B-A3B-Instruct-2507</td><td></td><td>35.32 (34.68–35.97)</td><td>54.73 (54.03–55.41)</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507</td><td>44.57 (43.95–45.21)</td><td>48.17 (47.53–48.81)</td><td>60.07 (59.48–60.68)</td></tr><tr><td></td><td>25.90 (25.32–26.49)</td><td>25.84 (25.23–26.47)</td><td>41.66 (40.91–42.43)</td></tr><tr><td>Qwen3-Next-80B-A3B-Instruct</td><td>47.60 (46.98–48.24)</td><td>47.11 (46.42–47.79)</td><td>57.49 (56.83–58.14)</td></tr><tr><td>Qwen3-Next-80B-A3B-Thinking</td><td>32.98 (32.31–33.66)</td><td>35.16 (34.46–35.87)</td><td>45.61 (44.86–46.38)</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-32B</td><td>16.45 (15.92–17.01)</td><td>14.24 (13.72–14.74)</td><td>27.51 (26.74–28.29)</td></tr><tr><td>gpt-oss-120b</td><td>34.23 (33.63–34.84)</td><td>32.89 (32.29–33.52)</td><td>49.52 (48.82–50.22)</td></tr><tr><td>LLaMA-3.3-70B-Instruct</td><td>47.70 (47.09–48.31)</td><td>50.42 (49.80–51.04)</td><td>49.64 (49.01–50.28)</td></tr><tr><td>HuatuoGPT-o1-7B</td><td>7.73 (7.29–8.17)</td><td>5.37 (5.01–5.74)</td><td>6.84 (6.39–7.29)</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>33.17 (32.46–33.88)</td><td>30.94 (30.23–31.66)</td><td>38.82 (38.04–39.60)</td></tr><tr><td>Baichuan-M3-235B</td><td>52.89 (52.29–53.50)</td><td>57.37 (56.75–57.98)</td><td>58.18 (57.60–58.77)</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 7: LNDS-based PCS results with bootstrap confidence intervals. Results are reported across diferent judge models under both single-turn vignette and multi-turn interactive settings. Each entry shows the mean PCS value with its 95% bootstrap confidence interval. Higher values indicate better pathway consistency.

## Prompt 1: Reasoning Tree Extraction

You are a medical expert. You will receive a clinical guideline as input. Your task is to read the input, understand each paragraph, extract all content related to clinical decision-making, and convert it into a decision tree represented as a JSON object. Each non-leaf node must branch into two paths, "yes" and "no", and represent a criteria-check question about patient-specific clinical data or findings (e.g., “Is symptom A present?”, “Is lab value B ≥ threshold?”, “Did event C occur within time window D?”); each leaf node should represent a specific conclusion.

Before constructing the decision tree, analyze the entire guideline and determine whether the primary clinical focus for the selected disease is diagnosis, classification, screening, or management. Report this explicitly in the final JSON under "primary\_clinical\_focus". Use this determination to guide node construction, but do not represent it as a decision node. The decision tree must represent a complete and clinically meaningful pathway. It should reflect the main logical architecture of the selected domain, typically with multiple hierarchical levels, unless the guideline itself is inherently simple.

## When constructing the decision tree:

\- Replace vague or implicit conditions with explicit, objective, and clinically actionable criteria (e.g., symptoms, laboratory thresholds, imaging findings, or time-based conditions), strictly based on the guideline text.

\- Each decision node should assess only one independent clinical variable or threshold. Avoid merging multiple distinct criteria into a single question. When multiple criteria are required by the guideline, represent them as sequential hierarchical nodes rather than compound logical expressions.

\- Do NOT introduce external medical knowledge or infer criteria not explicitly stated in the guideline. If the guideline text is ambiguous, preserve the ambiguity without speculation.

\- Do NOT include nodes that pose questions about clinical goals, tasks, intentions, scopes, or diagnostic objectives (e.g., whether the clinical task is to diagnose, classify, screen, or manage a condition). All decision nodes must be based solely on patient-specific clinical data or findings.

\- Do NOT create decision nodes that ask whether a diagnosis/classification/screening result/management decision is already established or “clear/confirmed/likely” (e.g., “Is the diagnosis clear?”, “Has the patient been diagnosed with. . . ?”, “Is the patient classified as. . . ?”, “Should treatment be started?”). Decision nodes must test only patient-level observable criteria (symptoms, signs, labs, imaging, events) rather than the existence of a clinical conclusion.

\- Do NOT create redundant or logically equivalent nodes. If a condition has already been fully determined by a previous yes/no branch (e.g., the "no" branch of "Is the patient pregnant?" already implies the patient is not pregnant), do not restate the complementary condition as a new decision node.

\- If the guideline uses conclusion-like phrasing (e.g., “diagnose X if. . . ”, “classify as Y when. . . ”, “screen positive if. . . ”, “treat if. . . ”), convert it into explicit, verifiable criteria nodes (the “if/when” conditions), and place the conclusion itself only in a leaf node.

In addition to the decision tree, identify the guideline evidence supporting each node:

\- Assign a unique numeric index to every node in the decision tree.

\- For each node, identify and quote the exact sentence(s) from the guideline that justify the node. Quoted evidence must consist of complete declarative sentences containing clinical statements or recommendations from the guideline body, and must not be drawn from section titles, headings, subheadings, interrogative statements, table/figure captions or other non-informative content.

\- All quoted sentences must be reproduced in full, without omission or truncation.

\- All quoted guideline sentences must pertain specifically to the selected disease and population. Do not cite general background statements or recommendations that are not explicitly tied to the selected focus.

\- Allow one-to-many or many-to-one mappings between nodes and guideline sentences.

Note: The guideline text is automatically converted from a PDF into Markdown format. As a result, it may contain structural inconsis tencies, incorrect heading levels, formatting artifacts, duplicated text, table fragments, or other non-informative content. Do NOT rely solely on heading hierarchy or formatting to determine logical structure. Instead, identify and extract clinically meaningful, decisionrelevant statements based on their semantic content. Ignore irrelevant, repetitive, or non-clinical material that does not contribute to patient-specific clinical decision-making. Do not assume that structurally well-formatted or clearly segmented subsections represent the primary clinical objective of the guideline. Selection of the focus must be based on semantic centrality and scope rather than formatting clarity or local structural neatness.

Do not include any explanations, commentary, or content beyond the specified JSON output.

\## Guideline:

{Guideline}

\## Example of a Leaf-Node JSON Format:

"conclusion": "a specific conclusion"

```perl
Prompt 1: Reasoning Tree Extraction (continued)
## Example of a Non-Leaf-Node JSON Format:
{
"index": 1,
"description": "description of a specific decision step",
$" y e s " : \{ \} .$
$" \mathrm { { n o " } : \{ j \} }$
}
## Output Format:
{
"primary_clinical_focus": "diagnosis | classification | screening | management",
"decision_tree": {
$\ " \mathrm { i n d e x } \mathrm { " : } 1 ,$
...
},
"node_guideline_mapping": {
$" 1 " : |$ [
"Quoted guideline sentence $1 " ,$
"Quoted guideline sentence $2 "$
],
$" 2 " :$ [
...
]
}
}
## Your Output:
```  
Table 8: Prompt template for reasoning tree extraction.

![](images/8590b02eb5d5c87a5b0dcae43f6dd4a36f10ae5669ff247e12fd1780659e234c.jpg)  
Table 9: Prompt template for case vignette synthesis.

![](images/2e31bf7c817bbe4985bc4dd699f44225777dc0b94518645013e0cb56fbe6608a.jpg)  
Table 10: Prompt template for reasoning narrative conversion.

![](images/48acea5e5c2361171db0a20e6b36e42cbf3256afc3195b7cca46680402484c24.jpg)  
Table 11: Prompt template for single-turn inference.

## Prompt 5: Single-Turn Step-Level Consistency Scoring

You are a medical expert.

You will receive two clinical decision pathways:

\- Reference Clinical Pathway: a reference sequence of clinical reasoning steps

\- Candidate Clinical Pathway: a pathway to be evaluated against the reference

Each pathway is provided as a list of strings, where each string represents one clinical decision step.

In the Reference Clinical Pathway, the steps are labeled sequentially from R1 to R{Reference\_Pathway\_Length}, for a total of {Reference\_Pathway\_Length} steps.

In the Candidate Clinical Pathway, the steps are labeled sequentially from C1 to C{Candidate\_Pathway\_Length}, for a total of {Candidate\_Pathway\_Length} steps.

Your task is to compare every step in the Reference Clinical Pathway with every step in the Candidate Clinical Pathway and evaluate their consistency.

## Evaluation Criteria:

Consistency (1–5) – the degree to which the Candidate Step represents the same clinical decision or logical judgment as the Reference Step.

## Scoring Guide:

First read and understand each clinical pathway as a whole. The meaning of each step should be interpreted in the context of its pathway. For each pair (Ri, Cj):

Identify the key decision points in the Reference Step (such as symptoms, test results, thresholds, clinical conditions, or required actions) and understand the decision logic expressed in that step within the context of the pathway.

Then evaluate whether the Candidate Step, when interpreted within its own pathway context, preserves the same decision meaning by covering these key decision points in a way that supports the same decision logic.

Matching concepts alone is not suficient for a high score. A high score requires both strong coverage of the Reference Step’s key decision points and the same decision logic.

## Score Definitions:

5: The Candidate Step covers all key decision points in the Reference Step and expresses the same decision logic correctly.

4: The Candidate Step covers most key decision points in the Reference Step and preserves essentially the same decision logic, missing only minor details.

3: The Candidate Step overlaps with the Reference Step on some key decision points, but misses important information or only partially preserves the decision logic.

2: The Candidate Step mentions related concepts or partially related information, but represents a diferent decision logic, reasoning stage, or type of judgment.

1: The Candidate Step is unrelated to the key decision points and decision logic in the Reference Step, or expresses a clearly incorrect or contradictory condition.

## Output Format:

For each Reference step Ri, output all corresponding Candidate comparisons in ONE line.

Within each line, list all pairs (Ri, Cj) using the format, separated by commas:

R<i>-C1: <score>, R<i>-C2: <score>, ..., R<i>-C{Candidate\_Pathway\_Length}: <score>

For example:

R1-C1: 5, R1-C2: 2

R2-C1: 3, R2-C2: 5

Output Requirements:

\- Output all {Reference\_Pathway\_Length} × {Candidate\_Pathway\_Length} step pairs exactly once.

\- Output exactly {Reference\_Pathway\_Length} lines (one line per Reference step).

\- Each line must contain exactly {Candidate\_Pathway\_Length} pairs, covering C1 to C{Candidate\_Pathway\_Length} in order.

\- Each pair must follow the format R<i>-C<j>: <score>.

\- Use only the provided labels: R1 to R{Reference\_Pathway\_Length} and C1 to C{Candidate\_Pathway\_Length}. Do not use any labe outside these ranges.

\- Within each line, pairs must be separated by a comma.

\- Output only the pairwise scores. Do not include explanations, comments, headings, or additional text.

\- Before producing the final output, verify that all possible Reference–Candidate pairs are included and that no pair is missing or duplicated.

Prompt 5: Single-Turn Step-Level Consistency Scoring (continued)   
## Reference Clinical Pathway:   
{Reference\_Pathway}   
## Candidate Clinical Pathway:   
{Candidate\_Pathway}   
## Your Output:  
Table 12: Prompt template for single-turn step-level consistency scoring.

![](images/301244a9fc79131642f30a616fc8468b367ddcb94a0d6d32fcfc98e081767594.jpg)  
Table 13: Prompt template for multi-turn inference.

![](images/f3143d61a771696cfeefd7c4f6e2c71f533ea348ce1b52d3c6ebd0e5ff38f353.jpg)  
Table 14: Prompt template for multi-turn patient environment.

## Prompt 8: Multi-Turn Step-Level Consistency Scoring

You are a medical expert.

You will receive two clinical questioning pathways:

\- Reference Clinical Questioning Pathway: a reference sequence of clinician questions used to gather patient information

\- Candidate Clinical Questioning Pathway: a questioning pathway to be evaluated against the reference

Each pathway is provided as a list of strings, where each string represents one clinical question.

In the Reference Clinical Questioning Pathway, the questions are labeled sequentially from R1 to R{Reference\_Pathway\_Length}, for a total of {Reference\_Pathway\_Length} questions.

In the Candidate Clinical Questioning Pathway, the questions are labeled sequentially from C1 to C{Candidate\_Pathway\_Length}, for a total of {Candidate\_Pathway\_Length} questions.

Your task is to compare every question in the Reference Clinical Questioning Pathway with every question in the Candidate Clinical Questioning Pathway and evaluate their consistency.

## Evaluation Criteria:

Consistency (1–5) – the degree to which the Candidate Question reflects the same clinical intent, information need, or reasoning purpose as the Reference Question.

## Scoring Guide:

First read and understand each questioning pathway as a whole. Each question should be interpreted in the context of its pathway, including what has already been asked and what information is being pursued.

For each pair (Ri, Cj):

Identify the key information targets in the Reference Question (such as symptoms, duration, severity, location, timing, risk factors, medical history, test results, or clarifications being requested), and understand the intent of the question within the pathway.

Then evaluate whether the Candidate Question, when interpreted within its own pathway context, seeks the same information and serves the same clinical reasoning purpose.

Consider whether the two questions contribute to the same clinical reasoning process or information-gathering goal within their respective pathways.

Superficial keyword overlap is not suficient. A high score requires both alignment in the information being requested and consistency in the clinical reasoning purpose.

## Score Definitions:

5: The Candidate Question covers all key information targets in the Reference Question and expresses the same questioning intent and clinical reasoning purpose.

4: The Candidate Question covers most key information targets and preserves essentially the same intent and reasoning purpose, missing only minor details.

3: The Candidate Question overlaps with the Reference Question on some information targets, but misses important aspects or only partially aligns with the intent.

2: The Candidate Question mentions related concepts or partially related information, but represents a diferent questioning intent, reasoning stage, or purpose.

1: The Candidate Question is unrelated to the information targets and intent of the Reference Question, or reflects a clearly diferent or inappropriate line of questioning.

## Output Format:

For each Reference question Ri, output all corresponding Candidate comparisons in ONE line.

Within each line, list all pairs (Ri, Cj) using the format, separated by commas:

R<i>-C1: <score>, R<i>-C2: <score>, ..., R<i>-C{Candidate\_Pathway\_Length}: <score>

For Example:

R1-C1: 5, R1-C2: 2

R2-C1: 3, R2-C2: 5

Output Requirements:

\- Output all {Reference\_Pathway\_Length} × {Candidate\_Pathway\_Length} question pairs exactly once.

\- Output exactly {Reference\_Pathway\_Length} lines (one line per Reference question).

\- Each line must contain exactly {Candidate\_Pathway\_Length} pairs, covering C1 to C{Candidate\_Pathway\_Length} in order. - ach pair must follow the format R<i>-C<j>: <score>.

\- Use only the provided labels: R1 to R{Reference\_Pathway\_Length} and C1 to C{Candidate\_Pathway\_Length}. Do not use any label outside these ranges.

\- Within each line, pairs must be separated by a comma.

![](images/ac1b446efef4e5f575c5510830e49384f7856e6da5ba19d622c797b5e778de99.jpg)  
Table 15: Prompt template for multi-turn step-level consistency scoring.

![](images/8dff395b00f111c5e3028fb33c8a6ba45073abe4b759bd0056096a6d13ae4159.jpg)  
Table 16: Prompt template for outcome accuracy scoring.