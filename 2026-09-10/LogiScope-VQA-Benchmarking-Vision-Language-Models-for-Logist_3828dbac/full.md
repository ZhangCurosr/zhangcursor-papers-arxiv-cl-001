# LogiScope-VQA : Benchmarking Vision-Language Models for Logistics Hazard Identification in Industrial Scenarios

Hanjing Zhou<sup>1,\*</sup>, Mingze Yin<sup>2,\*</sup>, Ying Lian<sup>1</sup>, Jun Ma<sup>1</sup>, Chang-Yu Hsieh<sup>2,†</sup> and Yanbing Zhou<sup>1,†</sup>

<sup>1</sup>Cainiao Group, Alibaba Group, <sup>2</sup>Zhejiang University,

<sup>\*</sup>Equal contributions. E-mail: {zhj85393, mzyin256}@gmail.com

<sup>†</sup>Corresponding authors. E-mail: kimhsieh@zju.edu.cn, tonychou.zyb@cainiao.com

Large Multimodal Models (LMMs) large-scale deployment in industrial warehouse settings specifically necessitates that models exhibit human-expert-level hazard-oriented perception, understanding, and reasoning capabilities. However, the scarcity of real industrial data, tightly coupled to commercial terms, significantly hampers further advancement. To bridge this gap, we curate LogiScope-VQA to investigate the practical applicability of mainstream LMMs in real-world logistics operations. LogiScope-VQA comprises 2,476 images and 2,918 videos primarily sourced from real-world logistics parks, along with 10,274 VQAs meticulously curated and validated by human annotators. Grounded in 18 core objects and 20 risk types, we devise 39 subtasks aligned with three principal themes: industrial element perception, warehouse knowledge understanding, and potential risk reasoning. Furthermore, we incorporate dynamic thinking-budget configurations and dual-dimensional risk bias analyses to elucidate the properties of LMMs. Extensive experiments unveil that even powerful proprietary models, including GPT-5.5, Gemini-3.1-Pro, and Claude-Opus-4.7, exhibit a significant gap relative to human performance. The unique challenge of jointly integrating perception, understanding, and reasoning for hazard identification poses substantial headroom for further improvement on LogiScope-VQA. We additionally reveal the pervasive security bias issue that impedes LLMs’ practical deployment in real-world settings. The industrial dataset is publicly available under the CC BY-NC-SA 4.0 license.

Keywords: Industrial Benchmark, Logistics Hazard Identification, Visual Question Answering

## 1. Introduction

Large Multimodal Models (LMMs) have achieved remarkable success on a broad spectrum of questionanswering tasks in naturalistic settings, showcasing strong perceptual and reasoning abilities as well as extensive embedded knowledge (Li et al., 2025; Liu et al., 2026; Zhou et al., 2025a). Nevertheless, the limited evaluation of LMMs in industrial scenarios has hindered the conversion of this potential into practically deployable systems and tools (Dou et al., 2026; Yue et al., 2024). Industrial environments pose unique challenges that are largely underexplored in current benchmarks, particularly due to safety-oriented task requirements and the relatively dense spatial distribution of targets.

Logistics warehousing is a representative industrial scenario in which continuous monitoring and timely safety risk recognition are crucial for accident prevention, regulatory compliance, and eficient warehouse operations. Previous studies have begun to develop multimodal benchmarks for industrial logistics, but their applicability to real-world assessment remains constrained by two inherent obstacles: (i) In terms of evaluation depth, industrial logistics environments naturally raise concerns regarding commercial cost, privacy, safety, and liability. Consequently, IndustryEQA instead relies on physically simulated data produced within virtual environments. Collecting data from real-world industrial environments for systematic evaluation remains a significant gap. (ii) In terms of evaluation breadth, practical deployment of LMMs involves diverse safety-critical dimensions, including object perception, spatial reasoning, path planning, and action recognition. However, ARMBench and iSafetyBench are limited to evaluating specific tasks under a single question format. Comprehensive multi-task evaluation of logistics safety-oriented perception and reasoning capabilities has yet to be established. Therefore, the safety-oriented assessment of mainstream LMMs in real-world logistics scenarios remains an open question.

![](images/a003d59c8e571ce653e6b99c642e84644263035a931d6c9057c6acbbce6995fa.jpg)  
Figure 1 | Overview diagram the LogiScope-VQA, evaluating LMMs in real-world industrial scenarios across three key competency dimensions: industrial element perception, warehouse knowledge understanding, and potential risk reasoning. (The full task taxonomy is provided in Appendix D.3.)

To systematically evaluate LMMs for logistics safety inspection, we introduce LogiScope-VQA, an industrial benchmark consisting of 2,476 images, 2,918 videos, 1,0274 VQAs, 20 risk factors, and 18 logistics objects. As illustrated in Fig. 1, we adopt a progressive curriculum to broadly assess the deployment value of LMMs in real-world industrial scenarios. For industrial element perception, 3.5 million surveillance clips collected from the global intelligent logistics company Cainiao capture the dynamics and heterogeneous operational scenarios within logistics parks. For warehouse knowledge understanding, recruited logistics experts collectively devote 52 person-days to annotating essential factors, including spatial locations, object attributes, hazardous actions, etc. For potential risk reasoning, we additionally incorporate open-ended problems and an LLM-as-a-Judge evaluation protocol to assess LMMs’ reliability and fairness in sophisticated thinking. Coupled evaluation tasks pose fundamentally new challenges to LMMs in satisfying the demands of industrial logistics deployment.

![](images/0fd8a878228e9278eb2504129b8f98c37391cb5a40a9e2d6b2f985c888764914.jpg)  
Figure 2 | Core statistical distributions in LogiScope-VQA, comprehensively covering 5,394 visual samples, 20 safety risks, and 18 core objects.

By conducting a holistic evaluation of 20 mainstream open-source and advanced proprietary LMMs, we summarize the key findings as follows: (a) As exemplified by Qwen3.5-Plus, open-source LMMs closely track proprietary performance and achieve parity with GPT-5.5 and Claude-Opus-4.7. Nevertheless, LMMs still have a pronounced performance disparity relative to human experts, indicating substantial room for improvement. (b) Industrial scenarios pose unprecedented challenges to fine-grained visual perception. LMMs consistently perform below expectations on the fine-grained industrial element perception tasks. Open-source models attain average accuracies of 0.39 on FG-S and 0.36 on FG-C. (c) Tasks involving industrial element perception and warehouse knowledge understanding are predominantly driven by visual cues. In contrast, potential risk reasoning requires models proficient in sophisticated thinking to identify risk factors. (d) Mainstream LMMs exhibit a pervasive risk-averse bias when responding to risk-prediction queries. Proprietary commercial models tend to adopt conservative responses, whereas open-source models provide more permissive and candid reports. Only 25% of LMMs pass our fairness-criteria audit. Collectively, these findings emphasize the challenges intrinsic to the LogiScope-VQA benchmark and delineate directions for future research and model improvement.

The curated benchmark bridges the last-mile from well-trained, simulation-validated models to practical application, providing actionable guidance for real-world readiness. For LMMs developed for large-scale industrial deployment, achieving strong performance on LogiScope-VQA is necessary to demonstrate holistic security expertise and expert-level perception and reasoning in operational environments. We hope the unique visual perception challenges, risk reasoning characteristics, and response bias tendencies introduced by LogiScope-VQA will provide useful insights for future work.

![](images/5725b672b06b7762845fe79dfe3462b133cf8a7c45a55f7e4ad589053e4f9372.jpg)  
Figure 3 | The curation pipeline of LogiScope-VQA, rigorously presenting the operational protocol alongside example outputs.

## 2. LogiScope-VQA

## 2.1. Overview of Logistics Benchmark

We introduce the LogiScope-VQA benchmark, a novel benchmark meticulously curated to evaluate the perception and reasoning capabilities of foundation LMMs for logistics safety in real-world industrial scenarios. The visual corpus is derived from the warehouse surveillance of the global intelligent logistics company Cainiao, comprising 3.5 million raw clips that, after rigorous cleaning, filtering and augmentation, are distilled into 5,394 high-quality visual samples. Logistics questions are automatically generated from human-annotated object attributes and hand-crafted question templates, comprehensively covering single-choice, multiple-choice, and open-ended formats. Corresponding answers are authored, annotated, and cross-validated by recruited logistics experts through a hierarchical workflow, resulting in 10,274 high-quality VQA instances. Collectively, the benchmark’s thorough data release and holistic experimental evaluation are underpinned by one year of surveillance across industrial warehouses worldwide, 52 person-days of annotation by logistics experts, and 324 million tokens consumed during LMM API inference.

Grounded in diverse visual content and granular attribute annotations, our benchmark encompasses single-choice, multiple-choice, and open-ended VQA problems. As illustrated in Fig. 2, we achieve comprehensive coverage of key elements in industrial logistics scenarios, spanning 18 core objects and 20 risk types. We aim to directly challenge LMMs to perceive logistics objects and integrate vertical knowledge to uncover latent industrial hazards. Eventually, LogiScope-VQA is designed to rigorously evaluate the essential skills of LMMs, spanning industrial element perception, warehouse knowledge understanding, and potential risk reasoning.

Our benchmark proposes four key challenges for multimodal foundation models. Across diverse tasks, it requires LMMs to (i) perceive dense object distributions in industrial environments, (ii) acquire and operationalize logistics-specific domain knowledge, (iii) integrate multimodal information to perform reasoning, and (iv) anticipate and identify potential safety risks via sophisticated thinking. We aim for the curated benchmark to serve as a comprehensive LMM testbed for identifying logistics safety risks in industrial scenarios, thereby precisely evaluating the real-world deployment potential of mainstream foundation models.

## 2.2. Distinctive Positioning of LogiScope-VQA

Distinct from existing benchmarks, we require LMMs to perform visual question-answering grounded in industrial contexts, enabling an intuitive assessment of practical operational value. The most closely related prior eforts are IndustryEQA (Li et al., 2026) and iSafetyBench (Abdullah et al., 2025). However, IndustryEQA derives industrial data from simulated virtual environments, while iSafetyBench focuses solely on hazardous action recognition, constraining the breadth of visual elements and task coverage. In contrast, LogiScope-VQA primarily sources data from real-world logistics warehouses, a nearly tenfold size over the previous dataset, and conducts a comprehensive multi-dimensional evaluation across perception, understanding, and reasoning.

## 2.3. Dataset Curation Pipeline

The construction of LogiScope-VQA followed a hierarchical curation pipeline that focused on data coverage and annotation quality, as illustrated in Fig. 3. In step 1, we aggregated one year of surveillance video data from Cainiao’s global warehouse parks, amounting to 3.5 million raw clips. This massive corpus was then cleaned through video segmentation, camera-zone grouping, object detection pre-labeling and deduplication, yielding more than 7K high-quality visual samples as the initial seed data. Targeted editing is performed on approximately 900 samples with Nano Banana 2 (Gemini, 2023) to rebalance the distribution of rare risk types. (The detailed image editing prompt is presented in Appendix D.4.) These over 7K seed samples were subsequently filtered in step 4 based on VQA pair quality, retaining the final 5,394 visual samples. Concurrently, we identified core objects of interest within logistics safety and transformed their associated unstructured safety manuals into a structured schema. In step 2, basic annotators spent 12 person-days annotating 18 core objects and their basic attributes. Specifically, they first drew bounding boxes to identify designated objects (e.g., persons, electric forklifts) on the keyframes, and then labeled the corresponding attributes (e.g., frame location, gender). To ensure accuracy, a third-party audit was conducted, with substandard annotations returned for iterative correction. In step 3, the labeled object attributes were populated into the placeholders of the pre-defined 15 question templates to batch-generate initial VQA pairs, followed by a refinement operation. Such operation not only resolves referential ambiguity by adding specific modifiers and descriptive constraints to object names, but also improves sentence fluency by ensuring grammatical correctness and natural phrasing. In step 4, four specialized annotators leveraged their domain expertise to annotate ground-truth answers and simultaneously filter out low-quality VQA pairs. This three-round cross-validation process, which took 40 person-days in total, comprised: (i) initial answer annotation and filtering of ill-formed pairs (i.e., those with ambiguous referents, unnatural phrasing, or missing options), which retained 5,394 visual samples from the initial 7K seed samples; (ii) independent re-annotation by a senior expert to accept concordant cases and adjudicate discordant cases with reference to the original labels to determine preferred answers; and (iii) a stratified review that prioritized verifying cases with prior inter-round inconsistency. Consequently, approximately 90% of the data was retained as high-quality VQA pairs. In step 5, to ensure dataset diversity, we balanced the distribution of core objects, risk types, and hallucination rates by removing over-represented categories. Ultimately, the entire pipeline produced LogiScope-VQA, a comprehensive dataset comprising 10,274 high-quality VQA pairs. (Thorough details of the data processing pipeline are provided in Appendix D)

## 3. Experiment

## 3.1. Evaluation Setups

LMM Baselines We evaluate the dificulty of LogiScope-VQA using diverse state-of-the-art LMMs as baselines and establish robust reference points for future research. For proprietary LMMs, six leading commercial models are selected: Claude-Sonnet-4.6 (Team., 2026b), Claude-Opus-4.7 (Team., 2026a), Gemini-3.1-Pro (Gemini, 2023), GPT-5.4, GPT-5.5 (OpenAI, 2025), and Qwen3.7-Plus QwenTeam (2026). These models represent the current upper bound of closed-source multimodal capabilities and serve as strong performance ceilings for LogiScope-VQA. For open-source LMMs, models spanning multiple architectural families and parameter scales are included: MiMo-VL-7B-RL (Team, 2025), GLM-5.2 (Z.ai, 2026), GLM-4.7 (Z.ai, 2025), Kimi-K2-Thinking (Kimi AI, 2025), Kimi-K2.6 (Kimi AI, 2026), InternVL-3.5 (InternVL, 2025), LLaVA-v1.6-7B (LLaVA, 2024), Qwen3.5-Plus (QwenTeam, 2026), and multiple Qwen3-VL variants (8B, 32B, 235B-A22B, Plus) (Bai et al., 2025). Moreover, we incorporate the Random Choice and Frequent Guess baselines to provide chance-level and prior-biased reference points. While computed in the standard manner for multiple-choice questions, the Random Choice accuracy is treated as zero for open-ended questions, given the extremely low probability that a solution randomly sampled from the vast candidate space would be consistent with the ground-truth answer. Meanwhile, the Frequent Guess baseline uses the most frequent canonical answer within each task category as the prediction for all open-ended questions.

Table 1 | Main results on LogiScope-VQA across 10 evaluation tasks. Abbr., FP-S: Fine-grained Perception of Single-instance, FP-C: Fine-grained Perception of Cross-instance, CP: Coarse-grained Perception; WC: Warehouse Commonsense, SR: Spatial Relation, OR: Operator Role; PAC: Perimeter Access Control, FM: Fire Monitoring, PSD: Personnel Safety Duty, EOC: Equipment Operation Compliance. The optimal proprietary and open-source models are highlighted in purple and yellow , respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Overall</td><td colspan="3">Industrial Element Perception</td><td colspan="3">Warehouse Knowledge Understanding</td><td colspan="3">Potential Risk Reasoning</td></tr><tr><td>FP-S</td><td>FP-C</td><td>CP</td><td>WC SR</td><td>OR</td><td>PAC</td><td>FM</td><td>PSD</td><td>EOC</td></tr><tr><td colspan="10">Heuristics Baselines</td></tr><tr><td>富 Random Choice</td><td>0.33</td><td>0.24</td><td>0.22</td><td>0.32</td><td>0.31</td><td>0.15</td><td>0.07 0.37</td><td>0.46</td><td>0.46</td><td>0.47</td></tr><tr><td> Frequent Guess</td><td>0.47</td><td>0.32</td><td>0.27</td><td>0.63 0.39</td><td>0.24</td><td>0.40</td><td>0.74</td><td>0.49</td><td>0.64</td><td>0.80</td></tr><tr><td colspan="10">Proprietary Large Multimodal Models</td></tr><tr><td>米Claude-Sonnet-4.6</td><td>0.64</td><td>0.63</td><td>0.64</td><td>0.86</td><td>0.90 0.58</td><td>0.20</td><td>0.49</td><td>0.76</td><td>0.58</td><td>0.62</td></tr><tr><td>米Claude-Opus-4.7</td><td>0.70</td><td>0.64</td><td>0.61</td><td>0.88</td><td>0.89 0.60</td><td>0.30</td><td>0.74</td><td>0.82</td><td>0.65</td><td>0.79</td></tr><tr><td>Gemini-3.1-Pro</td><td>0.68</td><td>0.72</td><td>0.70</td><td>0.91</td><td>0.92</td><td>0.66</td><td>0.30</td><td>0.48 0.82</td><td>0.58</td><td>0.61</td></tr><tr><td>GPT-5.4</td><td>0.69</td><td>0.59</td><td>0.58</td><td>0.90</td><td>0.93</td><td>0.59</td><td>0.25 0.58</td><td>0.84</td><td>0.69</td><td>0.78</td></tr><tr><td>GPT-5.5</td><td>0.71</td><td>0.62</td><td>0.68</td><td>0.78</td><td>0.95</td><td>0.59 0.20</td><td>0.79</td><td>0.82</td><td>0.65</td><td>0.81</td></tr><tr><td>Qwen3.7-Plus</td><td>0.66</td><td>0.67</td><td>0.64</td><td>0.89</td><td>0.95 0.56</td><td>0.40</td><td>0.48</td><td>0.83</td><td>0.53</td><td>0.69</td></tr><tr><td colspan="10">Open-source Large Multimodal Models</td></tr><tr><td>KKimi-K2-Thinking Kimi-K2.6</td><td>0.49</td><td>0.45</td><td>0.54</td><td>0.72 0.96</td><td>0.18</td><td>0.05</td><td>0.15 0.57</td><td>0.30 0.85</td><td>0.29 0.59</td><td>0.14 0.66</td></tr><tr><td>K Qwen3-VL-8B</td><td>0.62 0.50</td><td>0.61 0.48</td><td>0.60 0.89 0.44 0.89</td><td>0.93 0.87</td><td>0.45 0.09</td><td>0.20 0.05</td><td>0.42</td><td>0.73</td><td>0.57</td><td>0.65</td></tr><tr><td>Qwen3-VL-32B</td><td>0.55</td><td>0.55</td><td>0.54</td><td>0.83 0.92</td><td>0.27</td><td>0.15</td><td>0.46</td><td>0.74</td><td>0.60</td><td>0.59</td></tr><tr><td>Qwen3-VL-235B</td><td>0.64</td><td>0.64</td><td>0.59</td><td>0.89</td><td>0.91 0.55</td><td>0.25</td><td>0.43</td><td>0.73</td><td>0.60</td><td>0.67</td></tr><tr><td>Qwen3-VL-Plus</td><td>0.34</td><td>0.53</td><td>0.45</td><td>0.76</td><td>0.91 0.15</td><td>0.05</td><td>0.20</td><td>0.57</td><td>0.30</td><td>0.13</td></tr><tr><td>Qwen3.5-Plus</td><td>0.70</td><td>0.66</td><td>0.65</td><td></td><td></td><td></td><td>0.48</td><td></td><td></td><td></td></tr><tr><td>InternVL3.5</td><td>0.33</td><td>0.48</td><td></td><td>0.90</td><td>0.93 0.61</td><td>0.45</td><td>0.44</td><td>0.84 0.83</td><td>0.59</td><td>0.77</td></tr><tr><td>LLaVA-v1.6-7B</td><td>0.50</td><td></td><td>0.39</td><td>0.88</td><td>0.69 0.04</td><td>0.05</td><td></td><td></td><td>0.42</td><td>0.17</td></tr><tr><td></td><td></td><td>0.34</td><td>0.30</td><td>0.48</td><td>0.60 0.21</td><td>0.05</td><td>0.85</td><td>0.62</td><td>0.69</td><td>0.82</td></tr><tr><td>MiMo-VL-7B-RL 2</td><td>0.44</td><td>0.46</td><td>0.39</td><td>0.65</td><td>0.32 0.05</td><td>0.10</td><td>0.44</td><td>0.74</td><td>0.70</td><td>0.60</td></tr><tr><td>GLM-4.7</td><td>0.48</td><td>0.40</td><td>0.41</td><td>0.70</td><td>0.95 0.20</td><td>0.15</td><td>0.06</td><td>0.40</td><td>0.29</td><td>0.16</td></tr><tr><td>Z GLM-5.2</td><td>0.53</td><td>0.44</td><td>0.45</td><td>0.72</td><td>0.95 0.20</td><td>0.15</td><td>0.41</td><td>0.45</td><td>0.37</td><td>0.38</td></tr><tr><td>MiniMax-M2.1</td><td>0.43</td><td>0.47</td><td>0.49</td><td>0.54</td><td>0.93</td><td>0.19</td><td>0.15 0.50</td><td>0.43</td><td>0.40</td><td>0.60</td></tr><tr><td>MiniMax-M2.5</td><td>0.46</td><td>0.50</td><td>0.54</td><td>0.61</td><td></td><td></td><td></td><td>0.41</td><td>0.43</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.93</td><td>0.18</td><td>0.15</td><td>0.42</td><td></td><td>0.59</td></tr><tr><td colspan="10">Human Performance</td></tr><tr><td>豆 Novice Student</td><td>0.73</td><td></td><td></td><td></td><td>0.76</td><td>0.58</td><td>0.75</td><td>0.75</td><td>0.80</td><td>0.67</td></tr><tr><td>Logistics Expert 8</td><td>0.95</td><td>0.80 0.97</td><td>0.81 0.97</td><td>0.85 0.96</td><td>1.00</td><td>0.99</td><td>0.95</td><td>0.93 0.95</td><td>0.70 0.95</td><td>0.91</td></tr></table>

Human Performance To assess holistic human performance, we recruited ten college students with no expertise in industrial logistics to represent the novice level, and one senior logistics specialist with over four years of on-site warehouse safety experience to represent the expert level. Human participants completed all questions independently based solely on professional knowledge, without external references or model outputs. This setup ensures that the reported performance reflects genuine domain expertise rather than test-taking strategies or external aids, serving as a reliable upper bound for model evaluation. (The detailed employment terms for human evaluators are provided in the Appendix E.2.)

Decoding Settings For all problem instances, we use a unified prompt template that explicitly instructs the models to produce answers in a standardized format. For single-choice and multiplechoice questions, correctness is determined via direct keyword matching. For open-ended questions, we develop an LLM-as-a-Judge workflow to extract the key conclusion from each response for answer matching. Eventually, we adopt micro-averaged accuracy as the evaluation metric. All evaluated models are accessed via their oficial APIs or publicly available checkpoints. For each task, we repeat each test three times and report the average performance to ensure reliability and reproducibility. The sampling temperature for each model is set to its recommended or default value. (Detailed prompts and decoding setups are in Appendix E.1.)

## 3.2. Overall Results

## 3.2.1. Main Performance Analysis

As reported in Tab. 1, proprietary models maintain an advantage, yet the gap with open-source models has essentially closed at the top, where the best open-source model now ties the strongest proprietary systems. These leading models have edged past the untrained human (novice student) while remaining far below the logistics expert, confirming that our benchmark measures learnable professional expertise rather than common intuition. The key finding, however, is that fine-grained perception in logistics safety scenarios constitutes a shared, industry-wide bottleneck. All top models cluster within a narrow, low band on FG-S and FG-C, falling short of even the novice human and trailing the expert by a wide margin. This is partly because surveillance images in these settings are typically wide-angle, low-resolution, and densely cluttered with numerous targets and backgrounds, posing a far greater challenge than the clean, high-quality images common in standard benchmarks. In stark contrast, on knowledge-intensive tasks such as AR and EOC, models already outperform untrained humans. This contrast reveals an interesting structural complementarity between human and machine capabilities. Humans naturally excel at spotting fine-grained visual details in logistics safety scenarios even without training, whereas models, despite their vast knowledge, often behave like a well-read scholar with poor eyesight.

## 3.2.2. In-depth Thinking Analysis

As shown in Table 2 (evaluated on a subset of LogiScope-VQA), red, orange and blue cells present notable increases, moderate increases and noticeable drops, respectively. We can see that enabling thinking mode yields a mixed efect rather than a uniform improvement. Across models, the overall ranking remains largely stable, but the performance shift is highly task-dependent, suggesting that thinking mode interacts diferently with perception-heavy and reasoning-heavy tasks. Industrial Element Perception is the least responsive dimension, while Warehouse Knowledge Understanding shows modest gains, which may be constrained by the perceptual layer itself. However, Potential Risk Reasoning benefits the most, especially for stronger models such as Qwen3.5-Plus. This is because the process of observing the scene, matching safety rules and drawing inferences is reasoning-dominant, making it particularly well aligned with explicit chain-of-thought enabled by thinking mode.

Building on this observation, we further probe whether the reasoning advantage persists as the thinking time budget increases for the same Potential Risk Reasoning VQA samples. Figure 4 indicates a clear saturation pattern across all models. At low budgets, performance degrades severely due to output truncation, while further gains become marginal beyond the standard budget. Therefore, to balance performance and token consumption, thinking mode should be selectively enabled depending on the task’s primary reliance on visual perception or reasoning. And the time budget should be set near the saturation point.

Table 2 | Cross-task performance comparison with and without thinking mode. Colored cells indicate significant changes.
<table><tr><td rowspan="2">Model</td><td colspan="2">Industrial Element Perception</td><td colspan="2">Warehouse Knowledge Understanding</td><td colspan="2">Potential Risk Reasoning</td></tr><tr><td>non-think</td><td>think</td><td>non-think</td><td>think</td><td>non-think</td><td>think</td></tr><tr><td>LLaVA-v1.6-7B</td><td>0.28</td><td>0.30 ↑7.1%</td><td>0.27</td><td>0.31 ↑14.8%</td><td>0.65</td><td>0.71 ↑9.2%</td></tr><tr><td>Kimi-K2.6</td><td>0.66</td><td>0.66</td><td>0.71</td><td>0.57 ↓19.7%</td><td>0.68</td><td>0.74 ↑8.8%</td></tr><tr><td>MiniMax-M2.1</td><td>0.14</td><td>0.10 ↓28.6%</td><td>0.38</td><td>0.38</td><td>0.62</td><td>0.63</td></tr><tr><td>GLM-5.2</td><td>0.15</td><td>0.12 ↓20.0%</td><td>0.39</td><td>0.39</td><td>0.50</td><td>0.54 ↑8.0%</td></tr><tr><td>Qwen3-VL-235B</td><td>0.63</td><td>0.64</td><td>0.66</td><td>0.69</td><td>0.70</td><td>0.71</td></tr><tr><td>Qwen3.5-Plus</td><td>0.69</td><td>0.68 ↓1.5%</td><td>0.72</td><td>0.74</td><td>0.72</td><td>0.80 ↑11.1%</td></tr><tr><td>Qwen3.7-Plus</td><td>0.69</td><td>0.69</td><td>0.71</td><td>0.68 ↓4.2%</td><td>0.69</td><td>0.77 ↑11.6%</td></tr></table>

![](images/4524577fb9d76c8ce448f40dc98636903560d68eb27c868531dcb20691178f38.jpg)  
Figure 4 | Impact of thinking time budget on reasoning-driven tasks.

## 3.2.3. Safety Risk Bias Analysis

We investigate whether models’ safety risk judgments exhibit systematic bias or random noise with two experiments, as shown in Fig. 5. In panel (a), we quantify global directional skew via the indicator RBS, defined as Risk Bias Score $\begin{array} { r } { = \operatorname { t a n h } \Bigl ( \frac { \log \left( ( \operatorname { F N R } + \epsilon ) / ( \operatorname { F P R } + \epsilon ) \right) } { c } \Bigr ) } \end{array}$ , where FNR and FPR denote the false negative rate and false positive rate, and $\epsilon = 1 0 ^ { - 8 }$ and $c = 2$ . Positive, negative, and near-zero RBS indicate optimistic, conservative, and neutral judgments, respectively. Our evaluation reveals two critical findings. First, conservative bias dominates, with most models exhibiting a negative RBS. This systematic over-reporting tendency likely reflects a safety-first alignment strategy that prioritizes high recall, penalizing missed detections more heavily than false alarms. Second, proprietary and open-source models exhibit a marked performance divergence, suggesting a cognitive divide between genuine risk comprehension and superficial pattern matching. While top proprietary models cluster near the unbiased neutral zone, open-source ones scatter across the entire score range. This pattern implies that elite proprietary models may leverage event-level risk reasoning to accurately distinguish benign events from hazards, whereas less capable models lack such depth. Consequently, the latter likely rely on defensive over-reporting to mitigate uncertainty, manifesting as systemic conservative bias.

Building on this global picture, we conduct a paired-image probe to examine how such bias behaves under explicit risk removal. Specifically, we evaluate models on 100 surveillance samples containing genuine hazards and a corresponding set of risk-free samples, captured by the same cameras at a diferent time period, to measure recall and false alarm rates, respectively. As depicted in panel (b), most models maintain a risk recall above 60% but their false alarm rates vary substantially, confirming the prevalence of conservative bias. Moreover, The models are distributed across three quadrants, with those in quadrant 2 exhibiting excessive sensitivity and high false alarm rates, whereas those in quadrant 4 underperform by missing genuine threats. In contrast, models in quadrant 1 achieve the optimal trade-of between recall and false alarm rates, yet they still remain far from human expert performance. Collectively, our results underscore that safety risk bias is a pervasive systemic issue, which may be attributed to an interplay between internal cognitive limitations and external factors such as training data and alignment protocols.

![](images/c71c05dd98a9a3a08f36ba37e9a7ca0e188c77c20d32201afcf00a292e21ba56.jpg)  
(a) Statistical analysis of safety risk bias  
(b) Dual-dimensional comparison of risk recall  
Figure 5 | Empirical analysis of safety risk biases in mainstream LMMs. Panel (a) depicts holistic characterization of deviations in risk bias scores. Panel (b) provides detailed comparison across risk-recall and false-alarm dimensions.

## 4. Challenges and Future Directions

Evaluations on LogiScope-VQA expose two key limitations of current models. Firstly, fine-grained perception in logistics safety scenarios remains a major bottleneck, where models particularly struggle with low-resolution, wide-angle, heavily occluded and densely cluttered surveillance footage. To address this, future eforts should aim to boost models’ intrinsic visual capacity, achievable through strategies like optimizing pre-training for dense and low-resolution targets, or exploiting temporal information across video frames. Secondly, safety risk bias emerges as a significant yet overlooked issue, characterized by a prevalent unfairness, with most models heavily inclined to over-report risks. This problem deserves more attention from model developers, as industrial application models require fairness as much as accuracy. For future work, a crucial direction is to prevent such bias at the source, for example through balanced data curation or bias-aware training objectives.

## 5. Related Work

## 5.1. LMM Industrial Assessment

Industrial logistics safety assessment provides a compelling testbed for studying the practical applicability of LMMs. Unlike natural-scene evaluation (Liu et al., 2024; Majumdar et al., 2024), industrial environments (Abdullah et al., 2025; Mitash et al., 2023) impose the integration of sophisticated object recognition, precise spatial reasoning, and safety-critical hazard identification. However, this research area remains heavily underexplored, as industrial data source are often constrained by commercial confidentiality. IndustryEQA (Li et al., 2026) is a pioneering study on logistics safety evaluation in warehouse scenarios. However, it contains only approximately 1.3 thousand physicssimulated samples generated in virtual environments. To fully alleviate this challenge, we release LogiScope-VQA, which consists of 10,274 VQA instances annotated by logistics experts. We expect the constructed benchmark to provide a direct and practical testbed for assessing the potential of LMMs in industrial applications.

## 5.2. Practical Application Benchmarks

LMMs have emerged as a major frontier in AI research, ofering a unified modeling framework that integrates visual perception, language modeling, and sophisticated reasoning (Alayrac et al., 2022; Liu et al., 2023; Radford et al., 2021). Through vision-language pre-training, instruction tuning, and reinforcement learning for reasoning, cutting-edge research has progressively advanced model capabilities across a broad spectrum of vertical domains. To keep pace with these advances, recent benchmarks have introduced targeted evaluations for reasoning-centric mathematical problem solving (Lu et al., 2024; Wang et al., 2026; Yue et al., 2024), knowledge-intensive medical diagnosis (Chen et al., 2024; Liu et al., 2025; Yin et al., 2026; Zhou et al., 2025b), and visually-grounded remote sensing tasks (Danish et al., 2025; Wang et al., 2025a,b). Logistics safety evaluation focuses on workflows, equipment states, human behaviors and safety regulations, framing an embodied risk-oriented QA task in real industrial settings with distinctive domain-specific challenges. LogiScope-VQA is the first dedicated evaluation suite for logistics scenarios, filling a critical gap in existing research.

## 6. Conclusion

The proposed LogiScope-VQA represents a significant advance in assessing the capabilities of LMMs for logistics hazard identification in industrial scenarios. Concretely, we comprehensively evaluate the basic skills of industrial element perception, warehouse knowledge understanding, and potential risk reasoning. Extensive empirical evaluations reveal that LMMs fall short of expectations in industrial visual perception, and efective hazard identification hinges on coherent sophisticated reasoning. Moreover, our investigation uncovers a pervasive issue of security-related bias in current LMMs. These discrepant findings ofer valuable insights for foundation model construction and practical deployment. In future work, we will further enrich the visual question-answering corpus and extend the benchmark for agentic industrial operations, continuously contributing robust resources to the multimodal evaluation research community.

## Acknowledgments

We thank Yuan Zhou and other experts for their professional guidance on logistics safety knowledge and data annotation, and Qian Xu and her colleagues for their support in data annotation. We also thank the Qwen team for an insightful technical exchange, which deepened our understanding of the scarcity of logistics surveillance data for training foundation models. We have since shared a portion of our surveillance data with the team, in the hope that it may serve as a useful resource for future model development.

## References

R. Abdullah, Y. S. Rawat, and S. Vyas. iSafetyBench: A video-language benchmark for safety in industrial environment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1433–1442, 2025.

J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025, 2025.

P. Chen, J. Ye, G. Wang, Y. Li, Z. Deng, W. Li, T. Li, H. Duan, Z. Huang, Y. Su, et al. GMAI-MMBench: A comprehensive multimodal evaluation benchmark towards general medical ai. Advances in Neural Information Processing Systems, 37:94327–94427, 2024.

M. Danish et al. GEOBench-VLM: Benchmarking vision-language models for geospatial tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

S. Dou, M. Zhang, Z. Yin, C. Huang, Y. Shen, J. Wang, J. Chen, Y. Ni, J. Ye, C. Zhang, et al. Cl-Bench: A benchmark for context learning. arXiv preprint arXiv:2602.03587, 2026.

T. Gemini. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023, 2023.

T. InternVL. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265, 2025.

Kimi AI. Introducing Kimi K2 thinking. https://www.kimi.com/blog/kimi-k2-thinking, 2025.

Kimi AI. Kimi K2.6 tech blog: Advancing open-source coding. https://www.kimi.com/blog/ki mi-k2-6, 2026.

Y. Li, Z. Liu, Z. Li, X. Zhang, Z. Xu, X. Chen, H. Shi, S. Jiang, X. Wang, J. Wang, et al. Perception, reason, think, and plan: A survey on large multimodal reasoning models. arXiv preprint arXiv:2505.04921, 2025.

Y. Li, Y. Chen, A. Dao, L. Li, Z. Cai, Z. Tan, T. Chen, and Y. Kong. IndustryEQA: Pushing the frontiers of embodied question answering in industrial scenarios. In Advances in Neural Information Processing Systems, volume 38, 2026.

B. Liu, K. Zou, L.-M. Zhan, Z. Lu, X. Dong, Y. Chen, C. Xie, J. Cao, X.-M. Wu, and H. Fu. GEMeX: A large-scale, groundable, and explainable medical vqa benchmark for chest x-ray diagnosis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21310–21320, 2025.

H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. MMBench: Is your multi-modal model an all-around player? In Proceedings of the European Conference on Computer Vision, pages 216–233. Springer, 2024.

Y. Liu, T. Qu, Z. Zhong, B. Peng, S. Liu, B. Yu, and J. Jia. VisionReasoner: Unified reasoning-integrated visual perception via reinforcement learning. In International Conference on Learning Representations, volume 2026, pages 94069–94086, 2026.

T. LLaVA. LLaVA: v1.6. https://ollama.com/library/llava:v1.6, 2024.

P. Lu, L. Xu, S. Mishra, L. Xia, Y. Nie, C. Zhu, M. Zhang, T. Goldstein, W. Y. Wang, and H. Hajishirzi. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024.

A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaf, S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud, K. Yadav, Q. Li, B. Newman, M. Sharma, V. Berges, S. Zhang, P. Agrawal, Y. Bisk, D. Batra, M. Kalakrishnan, F. Meier, C. Paxton, S. Sax, and A. Rajeswaran. OpenEQA: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

C. Mitash, F. Wang, S. Lu, V. Terhuja, T. Garaas, F. Polido, and M. Nambi. ARMBench: An object-centric benchmark dataset for robotic manipulation. arXiv preprint arXiv:2303.16382, 2023.

OpenAI. OpenAI GPT-5 system sard. arXiv preprint arXiv:2601.03267, 2025.

QwenTeam. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, 2026.

QwenTeam. Qwen3.7: Multimodal agent intelligence. https://qwen.ai/blog?id=qwen3.7-p lus, 2026.

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021.

A. C. Team. Introducing claude opus 4.7. https://www.anthropic.com/news/claude-opu s-4-7, 2026a.

A. C. Team. Introducing claude sonnet 4.6. https://www.anthropic.com/news/claude-son net-4-6, 2026b.

X. L.-C. Team. MiMo-VL technical report. arXiv preprint arXiv:2506.03569, 2025.

F. Wang, M. Chen, X. He, Y.-F. Zhang, Y. Li, F. Liu, Z. Guo, Z. Hu, J. Wang, J. Xu, et al. OmniEarth-Bench: Towards holistic evaluation of earth’s six spheres and cross-spheres interactions with multimodal observational earth data. arXiv preprint arXiv:2505.23522, 2025a.

F. Wang et al. XLRS-Bench: Could your multimodal llms understand extremely large ultra-highresolution remote sensing imagery? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025b.

X. Wang, M. Yin, Y. Zhao, G. Liu, and D. Li. LiveK12Bench: Have large multimodal models truly conquered high school-level examinations? arXiv preprint arXiv:2605.26781, 2026.

M. Yin, Y. Zhu, J. Wu, J. Ma, H. Zhou, M. Li, Y. Zhou, J. Chen, T. Hou, J. Ye, et al. Caduceus: MoE foundation models for unifying biological and natural language. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 12715–12726, 2026.

X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

Z.ai. GLM-4.7: Advancing the coding capability. https://z.ai/blog/glm-4.7, 2025.

Z.ai. GLM-5.2: Built for long-horizon tasks. https://z.ai/blog/glm-5.2, 2026.

H. Zhou, M. Yin, D. Chen, J. Wu, and J. Chen. Group-On: Boosting one-shot segmentation with supportive query. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2025a.

H. Zhou, M. Yin, W. Wu, M. Li, K. Fu, J. Chen, J. Wu, and Z. Wang. ProtCLIP: Function-informed protein multi-modal learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 22937–22945, 2025b.

## A. Large Language Model Usage

The conceptual innovation and motivation underlying this study are conceived independently by the human authors, without cognitive input from large language models. The manuscript is originally written by the human authors, and large language models are engaged only at the final stage to assist in polishing key academic terminology. During dataset construction, we primarily relied on proprietary real-world data, while approximately 20.5% of the dataset is generated by the large language model (i.e., Nano Banana 2 (Gemini, 2023)) to systematically enrich the diversity of risk factors. For experimental evaluation, we employ large language models exclusively through the oficial API endpoints provided by respective vendors. All human–AI interactions are conducted in full compliance with all applicable terms of service and licensing conditions.

## B. Ethics Statement

We are committed to responsible AI research and strict adherence to data privacy standards. For the private data in LogiScope-VQA, we have implemented rigorous anonymization protocols. Specifically, all privacy-sensitive content—including clearly visible human faces, specific warehouse names and commercial identifiers—has been blurred or masked. This process combines automated detection with manual verification, ensuring that no individual or proprietary entity can be identified. In contrast, since the synthetic data contains no real-world identities and the open-source data is already publicly available, both are released in their original format without additional anonymization. We confirm that the construction and release of LogiScope-VQA comply with relevant ethical guidelines and data protection regulations.

## C. Data Usage and Licensing

LogiScope-VQA is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). It is strictly intended for academic and non-commercial research purposes only. Users are prohibited from using this data for any commercial activities, including but not limited to training commercial models, selling derived products, or integrating it into profit-generating services. Furthermore, any derivative works based on this dataset must be distributed under the same license. By accessing this dataset, users agree to adhere to these terms.

## D. Curation Details

## D.1. Hierarchical Curation Pipeline

Visual Data Acquisition The raw surveillance corpus comprises 3.5 million video clips collected from Cainiao’s global warehouse parks between April 15, 2025 and April 10, 2026. The data spans diverse operational scenarios, including high rack areas, loading docks, storage zones, light-duty picking shelves, ofice areas, etc. Notably, we declare that no surveillance data from the United States was included in this collection. To transform this massive raw corpus into a usable foundation, we implemented a rigorous cleaning process. Continuous videos were first segmented into clips or keyframes via fixed-interval sampling or risk-triggered alarm retrieval. These segments were then grouped by camera zones to align with specific operational areas. Subsequently, a pre-trained object detection model identified core objects, allowing us to filter out irrelevant background frames and balance object category distributions. The process concluded with deduplication, yielding over 7,000 high-quality, unique visual samples that serve as the foundational seed data for benchmark construction. With this visual foundation in place, a structured knowledge base from unstructured safety manuals became essential. Specifically, logistics specifications are parsed into hierarchical JSON schemas based on semantic tuples (Object,Attribute,Value). In this formulation, Object denotes one of the 18 core entities critical to warehouse safety monitoring, such as person, electric carrier and security scanner. Attribute captures either the intrinsic state of an object (e.g., posture, PPE compliance) or its interaction relationship with other entities (e.g., human-truck proximity). Value consists of predefined enumerations derived from domain regulations, strictly constraining the attribute space. For instance, a valid tuple might be (person, attrs\_hoi\_rack, climbing), which explicitly encodes the specific interaction of a person climbing a rack. Some examples of the JSON schema are shown in Fig. 6.

![](images/9c6ad1cde8601a852a2c40f8c75bb626c0a6180ff7531884c8f39578d6649b7b.jpg)  
Figure 6 | JSON schema examples for person, truck and electric forklift.

![](images/4562c8de297a59978f038b3e88e1cb2d8dd5743775007a71b6fa3fce11c8706e.jpg)  
Figure 7 | Question templates for generating question stems.

Object Attribute Annotation In Object Attribute Annotation, visual grounding was established via lightweight, cost-efective manual labeling. Specifically, annotators followed a "box-then-attribute" protocol: they first drew bounding boxes to localize core objects, and then marked basic attributes (e.g., frame location, occlusion status). For video samples, this process was applied exclusively to the single keyframe exhibiting the most salient risk characteristics, thereby maximizing annotation eficiency. Furthermore, to enrich semantic details, the "person" object received extended tags such as gender and upper-garment color. Crucially, all labels were constrained to predefined values aligned with the logistics knowledge base. To ensure final precision, a third-party quality audit was conducted, involving iterative sampling and correction cycles to rectify any remaining inaccuracies.

Automated Question Generation This step transformed object attribute annotations into natural language queries through two sequential operations. First, we leveraged a library of 15 predefined question templates (Fig. 7) to generate raw question stems. Specifically, twelve multiple-choice templates were designed to assess cognitive understanding across dimensions such as holistic scene interpretation, single-object recognition, and multi-object interaction. The remaining three openended templates evaluated capabilities in safety risk perception, consequence prediction, hazard rectification, and emergency response planning. Based on these templates, we performed deterministic population by mapping extracted core objects and attribute values into the corresponding slots. Second, we applied rule-driven linguistic refinement to enhance naturalness and precision. This refinement operation served two purposes. It synthesized multi-attribute constraints into natural pre-nominal modifiers to disambiguate and precisely localize target objects (e.g., refining “person” with attributes to “the woman in a blue upper garment in frame 3”). Meanwhile, it made minimal grammatical adjustments (e.g., quantifiers, prepositions or verbs) only when necessary, thereby preserving the structural integrity of the majority of queries.

<table><tr><td rowspan=1 colspan=7>Risk Name</td><td rowspan=1 colspan=6>Definition</td></tr><tr><td rowspan=1 colspan=7>Warehouse Access Control</td><td rowspan=7 colspan=6>In accordance with the warehouse safety regulation &quot;Warehouse Access Control&quot;During the daily power-off and lockdown period, all personnel must finish their work in advance and evacuate the warehousearea orderly; no one may remain behind or continue warehousing, inspection, or debugging activities for any reason.In accordance with the warehouse safety regulation “Key Post Attendance&quot;:A dedicated staff member must monitor the security scanner and goods in real time. The attendant must not leave the post, swap duties,</td></tr><tr><td rowspan=1 colspan=7>Presence After Night Lockdown</td></tr><tr><td rowspan=1 colspan=7>Key Post Attendance</td></tr><tr><td rowspan=2 colspan=6>Unattended Security Scanner Post</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=7></td><td rowspan=4 colspan=6>standards-compliant safety helmet with the chin strap fastened; this area poses risks of falling goods and beam impacts, and anyonewithout protective equipment is strictly forbidden to enter.</td></tr><tr><td rowspan=1 colspan=7>Personal Protective Equipment</td></tr><tr><td rowspan=1 colspan=7>No Helmet in High-Rack Area</td></tr><tr><td rowspan=1 colspan=7></td></tr><tr><td rowspan=2 colspan=7>Elevated Work #1Non-Designated Climbing Tool</td><td rowspan=8 colspan=6>In accordance with the warehouse safety regulation &quot;Elevated Work&quot;:All elevated work must use designated, inspection-approved tools like aerial work platforms, inventory cages, order pickers, A-frameladders, or mobile ladders; stepping on racks, cargo boxes, forklift forks, or other non-purpose-built structures at height is prohibited.In accordance with the warehouse safety regulation &quot;Elevated Work&quot;:Dedicated elevating devices such as lifting cages and order pickers are limited to a single occupant; carrying two or more personssimultaneously is strictly prohibited, as overloading significantly reduces equipment stability and increases the risk of tipping and falls.In accordance with the warehouse safety regulation “Elevated Work&quot;:When working at height on an A-frame ladder or mobile ladder, a safety helmet must be worn properly; when using mobile platformssuch as aerial work platforms, inventory cages, or order pickers, a protective helmet is required and a safety harness must be secured toa reliable anchor point at all times.</td></tr><tr><td rowspan=1 colspan=6>Non-Designated Climbing Tool</td></tr><tr><td rowspan=1 colspan=7>Elevated Work #2</td></tr><tr><td rowspan=1 colspan=6>Multiple Riders on Climbing</td><td></td></tr><tr><td rowspan=1 colspan=6>Device</td><td></td></tr><tr><td rowspan=2 colspan=5>Elevated Work #3</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=4>Elevated Work</td><td rowspan=1 colspan=5>Elevated Work Without Protection</td></tr><tr><td rowspan=1 colspan=7>Fire Monitoring</td><td rowspan=1 colspan=6>In accordance with the warehouse safety regulation &quot;Fire Monitoring&quot;:In warehouse and operation areas, any open flame, visible firelight, flying sparks, or smoke emission constitutes a serious safety hazard</td></tr><tr><td rowspan=1 colspan=7>Open Flame or Smoke</td><td rowspan=2 colspan=6>and is treated as a violation upon discovery. Unpermitted hot work—such as welding, cutting, or grinding that produces high-temperature sparks—is strictly prohibited.</td></tr><tr><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=7>Dock Loading &amp; Unloading #1</td><td rowspan=3 colspan=6>In accordance with the warehouse safety regulation “Dock Loading &amp; Unloading”:While the truck is moving (including starting/reversing), no one may enter the danger zone: 3m front/rear and 1m sides. This areacovers blind spots and poses severe collision/crushing risks. Drivers must check mirrors, cameras, and aids, proceeding slowly onlyafter confirming the area is clear.</td></tr><tr><td rowspan=1 colspan=7>Personnel in Truck Danger Zone</td></tr><tr><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=7>Dock Loading &amp; Unloading #2</td><td rowspan=2 colspan=6>In accordance with the warehouse safety regulation &quot;Dock Loading &amp; Unloading&quot;:No one may remain in or on the cargo bed while the truck is moving. After loading/unloading, personnel must evacuate to a safe area,and the driver may move the vehicle only after confirming the bed is empty.</td></tr><tr><td rowspan=1 colspan=7>Personnel in Moving Truck Bed</td></tr><tr><td rowspan=1 colspan=7>Dock Loading &amp; Unloading #3</td><td rowspan=2 colspan=6>In accordance with the warehouse safety regulation “Dock Loading &amp; Unloading”:Personnel must get on and off the dock only via designated passages such as dedicated stairs, ramps, or dock levelers; climbing dockguardrails, crossing edge structures, or jumping from the dock down to the ground or vehicles is strictly prohibited. Such behavior caneasily lead to missteps, slips, and falls, causing serious personal injury.</td></tr><tr><td rowspan=1 colspan=7>Dock Climbing or Jumping</td></tr><tr><td rowspan=1 colspan=7></td><td rowspan=5 colspan=6>In accordance with the warehouse safety regulation &quot;Forklift Driving Compliance&quot;:When powered forklifts (electric forklifts, electric pallet trucks, semi-electric pallet trucks, pallet stackers, or clamp trucks) travel with aload, the stacked cargo height must not exceed 1.8 meters or the upper limit of the mast, so as to prevent rollover from an excessivelyhigh center of gravity or collisions caused by blocked vision; cargo stability must be checked before loading to ensure safe operation.</td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #1</td></tr><tr><td rowspan=1 colspan=7>Overheight Load on Forklift</td><td rowspan=1 colspan=1>load, the s</td></tr><tr><td rowspan=1 colspan=7></td><td rowspan=3 colspan=6>load, the forks must face backward with the cargo held tight against the mast, and the truck must travel in reverse to ensure clear visionand a stable center of gravity; driving forward with a load is strictly prohibited.</td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #2</td></tr><tr><td rowspan=1 colspan=7>Loaded Forklift Not Reversing</td></tr><tr><td rowspan=1 colspan=7></td><td rowspan=4 colspan=6>In accordance with the warehouse safety regulation &quot;Forklift Driving Compliance&quot;:During powered forklift operations (electric forklifts, electric pallet trucks, semi-electric pallet trucks, pallet stackers, or clamp trucks),n-operators must maintain a safe distance. If boundary lights are present, do not enter the warning zone. Otherwise, keep clear of 2mfront/rear and 1m sides.</td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #3</td></tr><tr><td rowspan=1 colspan=7>Insufficient Forklift Safety Distance no</td></tr><tr><td rowspan=1 colspan=7></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #4</td><td rowspan=5 colspan=6>In accordance with the warehouse safety regulation “Forklift Driving Compliance”&quot;:Using powered forklifts (electric forklifts, electric pallet trucks, semi-electric pallet trucks, pallet stackers, or clamp trucks) to pushcargo or obstacles along the ground is strictly prohibited, as such operation can easily cause equipment loss of control, cargo tipping, orpersonnel being run over; to move heavy objects, use dedicated handling tools or place the cargo securely on the forks before transport.In accordance with the warehouse safety regulation &quot;Forklift Driving Compliance&quot;:Powered forklifts (electric forklifts, electric pallet trucks, semi-electric pallet trucks, pallet stackers, or clamp trucks) may only beoperated by a single licensed driver, and carrying any other person is strictly prohibited; the forks, fork carriage, or any part of the truckbody must never be used as a platform for carrying people.</td></tr><tr><td rowspan=1 colspan=7>Forklift Pushing Cargo on Ground</td></tr><tr><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #5</td></tr><tr><td rowspan=1 colspan=7>Extra Riders on Forklift</td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #6Forklift Forks Not Lowered</td><td rowspan=4 colspan=6>In accordance with the warehouse safety regulation “Forklift Driving Compliance&quot;:While powered forklifts (electric forklifts, pallet stackers, or clamp trucks) are traveling with or without a load, the forks must belowered to a safe height of about 15 cm above the ground; traveling with raised forks is strictly prohibited.In accordance with the warehouse safety regulation &quot;Forklift Driving Compliance&quot;:While a powered forklift (electric forklift, clamp truck, etc.) is in motion, the operator must stay inside the cab at all times and must notextend the head, hands, torso, or any other body part out of the cab. Such behavior can easily cause scraping collisions with racks, doorframes, pipes, or other equipment, resulting in serious personal injury.</td></tr><tr><td rowspan=2 colspan=7>Forklift Driving Compliance #7</td></tr><tr><td rowspan=1 colspan=2>While a powere</td></tr><tr><td rowspan=1 colspan=7>Operator Leaning Out of Cabin</td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #8Dismounting a Moving Forklift</td><td rowspan=2 colspan=6>In accordance with the warehouse safety regulation &quot;Forklift Driving Compliance&quot;:Operators must not leave the seat until any powered forklift (e.g., electric, pallet, stacker, clamp) is fully stopped with brakes applied.For temporary absence, stop, apply the handbrake, lower forks to the ground, and ensure the equipment is safe before leavingIn accordance with the warehouse safety regulation &quot;Forklift Driving Compliance&quot;:When an inventory cage is mounted on the forklift forks, the cage is limited to a single occupant; carrying more than one person isstrictly prohibited, as overloading significantly reduces equipment stability and increases the risk of tipping and falls.</td></tr><tr><td rowspan=1 colspan=7>Forklift Driving Compliance #9Overloaded Inventory Cage</td></tr><tr><td rowspan=2 colspan=7>Conveyor Operation ComplianceStanding on Running Conveyor</td><td rowspan=2 colspan=6>In accordance with the warehouse safety regulation &quot;Conveyor Operation Compliance&quot;:While the conveyor is running, being on its surface is strictly prohibited to prevent entanglement and injury; for maintenance, theconveyor must be stopped, powered off, and locked out, and personnel may enter only after confirming it is stationary.</td></tr><tr><td rowspan=1 colspan=2>Standi</td><td rowspan=1 colspan=1>ding or</td></tr></table>

Figure 8 | Definitions of 20 risk in potential risk reasoning.

![](images/cd29195bca5e59c9fa94db1378fe9a73713bc2a68b9b6db8121cca4f453601f1.jpg)  
Figure 9 | Full task taxonomy of LogiScope-VQA.

![](images/444622ae3225f6bfbf508a639b42938b62ca7263c377c677ff21310047c79485.jpg)  
Figure 10 | Prompt template and example for targeted image editing.

Question-Answering Annotation We performed Question-Answering Annotation to assign groundtruth answers and simultaneously filter out low-quality QA pairs. This process involved four specialized annotators and totaled 40 person-days of efort at a cost of \$1,000. In the first round, two specialists divided the dataset (containing over 7,000 visual samples) equally and independently annotated their respective halves, producing initial labels for each VQA pair. During this round, they explicitly discarded VQA pairs with ambiguous referents, unnatural phrasing, or missing correct options, retaining only well-formed and verifiable pairs (corresponding to 5,394 final visual samples) for subsequent verification. In the second round, one senior expert independently re-annotated every VQA pair and compared the result with the first-round label. If the labels matched, the pair was tentatively accepted; if they disagreed, the expert re-evaluated the pair to adjudicate the discrepancy, either upholding the initial label or correcting it. In the third round, VQA pairs with consistent labels from the previous two rounds underwent a rapid check, while significant attention was devoted to verifying those pairs with prior disagreements. This intensive cross-validation ensured the precision of the final ground-truth answers.

Dataset Assembly As is common in real-world scenarios, logistics safety surveillance data also exhibited a long-tail distribution, where the occurrences of core objects and safety hazards were highly uneven. To achieve a trade-of between diversity and balance, we rebalanced the VQA pairs across multiple dimensions, such as hallucination proportions, core object types and hazard categories, ultimately yielding LogiScope-VQA.

## D.2. Detailed Risk Definitions

We provide precise definitions for 20 specific risks in Fig. 8, which are involved in the potential risk reasoning tasks.

## D.3. Full Task Taxonomy

Fig. 9 presents the full task taxonomy.

## D.4. Prompt for Targeted Image Editing

As illustrated in Fig. 10, our prompt template for targeted image editing employs multiple placeholders, accompanied by a concrete example demonstrating its usage. The diversity of resulting prompts stems from the rich set of allowable values for each placeholder, which we comprehensively enumerate in Fig. 11.

<table><tr><td rowspan=1 colspan=1>Element</td><td></td><td rowspan=1 colspan=2>Available Value</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=16 colspan=2>① Climbing barehanded up a loaded rack, feet on beams, hands gripping column, with no climbing aids nearby② Standing on scissor lift guardrail edge, leaning out, one hand holding rail while other reaches for distant goods③ Standing on electric forklift tines without cage, one hand holding mast, other searching for items on shelf④ Standing on top A-ladder step above safety line, body overreaching laterally to grab goods from adjacent racking⑤ Straddling racking beam with legs dangling, one hand gripping column, other retrieving item from adjacent location⑥ standing on top of the operator cabin of an electric forklift, leaning toward the racking to inspect goods on the shelves⑦ standing on a handcart taller than a person&#x27;s height, reaching for goods on the shelf⑧ standing on a wheeled mobile ladder platform with casters, retrieving goods from the shelf9 standing alone on the platform of a single-occupancy order picker (man-up forklift)10 standing side by another worker, overcrowding the platform of a single-occupancy order picker (man-up forklift) standing inside a bright yellow, grid-walled inventory cage mounted directly onto the forks of an electric forklift12 standing on a high-level order picker vehicle, inspecting goods on the racking shelves13 standing side by another worker, overcrowding the cab of the same order picker vehicle14 standing on top of stacked cargo boxes, reaching for goods on the shelf15 using three empty pallets stacked vertically as a makeshift step ladder, standing on the top pallet and tiptoeing to retrieve an item16 standing on the conveyor belt of an operating production line17 climbing over an operating production line conveyor belt18 a truck reversing toward the loading dock from 2-3 meters away, a worker standing directly behind the truck between the dock andthe truck, only upper body visible, unaware of the reversing truck, at risk of being crushed19 a truck reversing toward the loading dock from 2-3 meters away, a worker passing through the gap between the dock and the truck,only upper body visible, at risk of being crushed@ a truck moving forward, a worker chasing the truck from behind to close the rear cargo door, at risk of being crushed</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>④ Stand</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>action</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=2 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=5 colspan=2>① back facing the camera                                ⑨ squatting position facing left② side profile facing left                                 ⑩ squatting position facing right③ side profile facing right                                11 standing with hands on hips④ slightly looking up                                   12 standing with one hand on waist⑤ facing the camera with head down                       13 body tilting slightly to the left6 looking down at feet                                  14 body tilting slightly to the right⑦ side profile facing left, leaning forward slightly             15 head tilted upward looking up⑧ side profile facing right, leaning backward slightly           16 head down focused on the ground</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>orientation</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>clothing</td><td rowspan=2 colspan=2>① wearing a blue warehouse uniform                       ④ wearing gray workwear② wearing a yellow reflective vest                         ⑤ wearing a black hoodie and a fluorescent yellow reflective vest③ wearing an orange safety jacket                         ⑥ wearing a blue reflective vest</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=4 colspan=2>① wearing a yellow hard hat but NO safety harness② wearing a red hard hat with safety harness unhooked (dangling)③ wearing both a hard hat and a safety harness④ wearing a safety harness but NO hard hat, head fully exposed⑤ without any hard hat, head fully exposed, and NO safety harness on body</td></tr><tr><td rowspan=1 colspan=2>safety gear</td></tr><tr><td rowspan=1 colspan=2>bareey gear</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=11 colspan=4>① electric forklift — a common logistics transport vehicle with two forks at the front for moving palletized goods; open-type drivercabin; metal mesh protective roof over the cabin② handcart — a manually pushed logistics handling tool, usually with two or four wheels, similar to a large supermarket shopping cart③ scissor lift — a logistics lift with an X-shaped hydraulic scissor mechanism and four-sided guardrails on all four sides④ mobile ladder — a non-folding, straight wheeled ladder with guardrail handholds and casters at the bottom⑤ A-frame ladder — a foldable A-shaped ladder, without wheels⑥ racking shelf— a steel structure for 3D storage, composed of upright columns, horizontal beams and shelves, for multi-layerstacking of pallets or turnover boxestool          ⑦ order picker — a dedicated industrial vehicle for order picking, with an operator platform and linked lifting device; the rider is thedriver, no dedicated cabin, has a frame for safety harness, lifting column, and large motor, no forks⑧ cargo — goods, packages or items to be transported, stored or handled in logistics, usually in cardboard or plastic boxes on pallets⑨ empty pallet — a standard pallet without cargo, usually made of wood, plastic or metal, for subsequent loading and forklift handling10 inventory cage — a cage with four-sided protective mesh mounted on forklift forks, with insertion holes at the bottom for forks,about half-person height to waist level1 conveyor belt — a continuous transport system in logistics for moving goods along a production line12 loading dock — the height-difference platform between the warehouse floor and the truck parking area13 truck — a cargo vehicle parked at the loading dock area for loading and unloading① far end, very high position, height above ground &gt; 4m        ⑥ dock area below (truck parking zone)② near end, very high position, height above ground &gt; 4m       ⑦ far end, low position, height above ground ≤ 2m③ middle position, very high position, height above ground &gt; 4m ⑧ near end, low position, height above ground ≤ 2m④ far end, high position, height above ground &gt; 2m            ⑨ middle position, low position, height above ground ≤ 2m⑤ near end, high position, height above ground &gt; 2m           ⑩ conveyor belt marked area (indicated by red bounding box)</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>tool</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>121</td></tr><tr><td rowspan=1 colspan=1>position</td></tr></table>

Figure 11 | Values of elements for targeted image editing.

![](images/85462465f5de53811e2ad3c1ea6643233a960273db9e310e1cd60eea0944006a.jpg)  
(a) Prompt for solving questions

![](images/e7966fe5561a221c31b89df886be6ba47d82e1df38036ec3bef8b8fad0dcc295.jpg)  
(b) Prompt for evaluating open-ended questions  
Figure 12 | Prompts for solving and evaluating questions.

## E. Evaluation Details

## E.1. More Decoding Settings

Fig. 12 illustrates two evaluation prompt templates. The left one is designed for solving multi-choices and open-ended questions, while the right one is utilized in the LLM-as-a-Judge workflow for a thirdparty LMM to assess the correctness of the model’s responses (outputting “correct” or “incorrect”). In this judging workflow, the ground truth for short-answer questions typically comprises several core key points. A response is judged as “correct” if it contains all these essential elements, regardless of any additional elaboration; conversely, the omission of any single key point results in an “incorrect” classification. This zero-tolerance policy for omissions is designed to prioritize high recall in hazard detection, given the severe consequences of missed risks in industrial environments. Such a rigorous criterion serves as a stringent test, efectively validating a model’s true capability and reliability in logistics safety surveillance. For improved transparency and reproducibility, Tab. 4 includes links to oficial documentation or model checkpoints of the evaluated LMMs, along with detailed inference configurations and the evaluation timeframe.

## E.2. Human Evaluation Protocol

To obtain comprehensive human performance, we recruited two distinct groups of participants with contrasting expertise levels. The first group consisted of 10 undergraduate student volunteers who served as generalist evaluators. While possessing strong general common sense and basic visual recognition capabilities, they lacked specialized knowledge in logistics operations or safety protocols. These students collectively covered the entire benchmark by answering approximately 1,000 questions each over a period of two weeks, receiving a stipend of \$15 per person. The second group comprised a single professional domain expert with at least four years of hands-on experience in logistics safety and surveillance. Unlike the generalists, this expert leveraged deep domain knowledge (e.g., familiarity with industrial equipment and safety protocols) to independently answer all questions in the benchmark. This expert evaluation also spanned two weeks and incurred a total cost of \$700. The results labeled “Novice Student” and “Logistics Expert” in Tab. 1 correspond to the performance of the student volunteers and logistics expert, respectively.

## E.3. Reliability of LLM-as-Judge Evaluation

A reliable LLM-as-Judge evaluation must be both stable and aligned with human expert judgments. We therefore randomly sampled 100 questions from the open-ended VQA set and examined the judge along the following two dimensions:

• Intra-judge: The same LLM judge evaluates each question 3× with temperature = 0.7. We report the agreement rate across the three runs.

• Inter-judge: Human experts independently judge the same subset. We report the agreement rate between the LLM judge’s decisions and the human judgments.

Table 3 | Stability and human alignment of LLM-as-a-Judge evaluation on 100 VQA pairs.
<table><tr><td>Dimension</td><td>Metric</td><td>PAC</td><td>FM</td><td>PSD</td><td>EOC</td><td>Overall</td></tr><tr><td>Intra-judge Inter-judge</td><td>Agreement rate Agreement rate</td><td>92.9% 85.3%</td><td>95.1% 89.4%</td><td>93.8% 87.9%</td><td>91.6% 85.1%</td><td>93.3% 87.2%</td></tr></table>

As illustrated in Tab. 3, intra-judge repeatability exceeds 93% and the LLM judge’s agreement with human experts exceeds 87%. These results prove that the LLM judge is stable and aligned with human judgment.

## F. Visualization

To facilitate a more intuitive understanding, we provide additional visualization examples. Fig. 13 displays visual-only examples (raw images).

![](images/f63dbb11b22a570d6796615e102baf50ed3b12f8943aafd7cbfd6ee53dd4b89f.jpg)

![](images/e2d34ff26074c3d77108931c79f44656ffea42ecc0f61dc94a49d7af6d0d7ab8.jpg)

![](images/513b829e0c0a875121f47507e0988389819c00326159959de883a23df67b8b5a.jpg)

![](images/d9c11f0aa4f90fa4bb78253a3e295ac964a1fb50be93dda089f43f33f7467c4e.jpg)

![](images/6e18cffe18a2fd0a367406bc93fd8477a6f6170ab49a5458c6607672bb51311b.jpg)

![](images/7bd1672b30b453f2fae3015a3ad7f44da24e931f395a0c28fef76f2c027ad0a0.jpg)

![](images/d7860795ebeaf8ee4ecbe9ce46395f8f895811126f4f0c626e90769a1bc8fa6f.jpg)

![](images/c71c1ac2c229c3ee29d1805b2eb3e937d8303ee2870686efe7ab6671b8f3380c.jpg)  
(a) Real video: dock operations area

![](images/d38709ddee5ee77a4d68c77e99ef32c59b14de175665c325df14f0c1c854c161.jpg)

![](images/a8ed03e74610eeb8933f0226b333a2058457f9b2bc437e3a6b5b21756d019015.jpg)

![](images/827373f7d057aa23f6574af4697e06d504e1a731d82b1778a271ad0d1e7b5b58.jpg)

![](images/7f185dabe815b74c3a757b3a30d39e482d81f1eac50c5b2b3cf4a832936e6299.jpg)

![](images/13c061778390802c5670c6603a26b59f244b459c70623837c18b5caa5a17bc7f.jpg)

![](images/2f84041ca31154eba9de28307b6f372a019cf9046844eabc0af9badeca574914.jpg)

![](images/af3cb35b2684bd588f9d5dc442bbcdf96658853dde5bf3376d5fb7ca127ba4b1.jpg)  
(b) Real video: storage area

![](images/e6f30cbdb586d8af2d20441cd67516a46632fab40eba67997287a5c79b05abcd.jpg)

![](images/214d986512c6ef5f86e673d1baf97fd22aa7be7564974b539fbc13e68e24b4e1.jpg)

![](images/c4be4fe34ebad7f2dca306a31248ec8473954bd19e4424d2d861bc286f6fa576.jpg)

![](images/24d892fff31b0ff1c81c7f2d41f5e3af7a8528a838f0b4f60b05be5ae43da7cd.jpg)

![](images/0a0a57869d5d391bdc05282eb27ad23fd7d0fdc342dba947bd7a7f45447de3bb.jpg)

![](images/9797a6e203c71accdd45bdaf3401ebb0604c5fe61e31bb07f3a8f810b733599b.jpg)

![](images/53ce5ed589e27d2d451d389723fce60e2409fb2c34e4a03c4eada57ff463f28c.jpg)

![](images/c842211bcf2844b281391f84017efe1820ac55874fd5a5f407e1b56a060b267e.jpg)  
(c) Real video: bulk storage area

![](images/49ea01440d4e67826e8eb7b77a8e2a1868ba43acd4e7dcfcda23da532ef9abf1.jpg)

![](images/5182c2cc10b96d729742279e44b4a433eb1bddd17911b0bf63e7ab0b7c840f29.jpg)

![](images/69c16dd8f7096facf044e7f300eacf0b83bb5c130b5e7ad153b6f75bec4d2257.jpg)

![](images/23558d9066adca36a5f5d992018aa42862d3c00b297d190067b4d0c3a377bd2b.jpg)

![](images/6623dbce34418e2b1699020f1406c7ea72c2f5c08031b2a7947b5d9fe50eec80.jpg)

![](images/766c7f4bd7804febfead8b4e44ffa291cd0f408d68bddfe120d82965583d6486.jpg)

![](images/c370220497b9bd5243730b81e90d65d1ea699ee337ac8d1b701ba3f5d397da55.jpg)  
(d) Real video: nighttime scene

![](images/cd5cbb4d882858085a439601dd5aee74a7ffb02ae3cd1a7ff64756cdb5cd4ce2.jpg)

![](images/a0de47ec7a0c88263ca77e326ac89ea46a4be6d0cddaee4c105e47cf8dee877d.jpg)

![](images/c44dfc3d0daaac75e3b0cda8aec6da88d73d08ada84eefe2674b6b9e5be8c99f.jpg)

![](images/e860f15d002ce95701f255145ded6045175b233d50f61660b9ff990cb9b6dc6e.jpg)

![](images/39ce88d79c3cf142280e3840acb6e4217ae27501aa74d7b0a3369c3c974afbdc.jpg)

![](images/2f35a462f9e35fa0103845c8724d136f69ec48981ddf6401557ae0b63e05ef61.jpg)  
(e) Real video: high rack area

![](images/4e12679fff125b9d8afa3170f8d1b83be8f734279dc358b08765973087502125.jpg)

![](images/7e223e3f23a4636363ee57887f7c33c70cc2e03f7ddaf6db93efa84a7b744580.jpg)

![](images/bc802705d154830db453af87ffe8b7ecea5d17a1d57585dacbf0f0d049161c3d.jpg)

![](images/10fd841c2603f6bdde1a639b5cc810a367dfa4a8864110a4ec1b4a54f5e042fe.jpg)

![](images/0b256e520778709a06f4694fab59190f169ff9e449aae02c9058909e494516ae.jpg)  
(f) Real video: security screening area

![](images/a55421e5b28fd2ef8a73bd78637bdedec6fe621d08e3f99a49cb9921e8e5f02a.jpg)

![](images/52ad19f11d9577f51f42ae94e08a93454f4c8cffc0b56415dfb891d554c330db.jpg)

![](images/d0def96fd7e649c2f14bdc15b45d932c794014dd9882f7ff8a19160debe7de0e.jpg)

![](images/1504afa5cea51ddde3a1dd5a755afcf5e17ac105329c324496845867ef7a0d4f.jpg)  
(g) Synthetic video: dock operations area

![](images/c030db35dcec4a172e1fe6b2d91634959c186e539e56c81c71c01eff7472e75e.jpg)

![](images/de85c29b545ba639e017477625420fc9ff35444f69c5ee62616282dc5577a64c.jpg)

![](images/bc390433d52da6117eaad7ec0fb22536dd5da83642ea067b209b0cdeee4564e2.jpg)

![](images/4cc8227f48b95dfab4626372e09551d5b1436ec8a660d38a4d8621742975d3bf.jpg)

![](images/3a1bef25ab234c533b7d66952ad36568d3a2efba0ecee4e685849fe779e2dd2b.jpg)  
(h) Synthetic video: conveyor area

![](images/594218e020a565fbe0eca64ab11a8a6f76cd6e87e7e0a9b583410420e78729c2.jpg)

![](images/a1d586580b4687e53e5384c671da7b8485cd397d2b8e9bd3188b84fbd6188a5f.jpg)

![](images/61be20dfe11acc4d1b08dd7dbab3d14e6c5e35cf07c4b8f509c9579018a9b902.jpg)

![](images/1b394dd7c854f39e60e7cabaa9d1aaad83d94f43d2631d8de76887b597c3f74a.jpg)  
(i) Real images

![](images/c958edae3b9f2e643e7d79255c335da02aca2df4b6d359ebce13d6c5dd42d993.jpg)

![](images/8784a5e4b8bf968aca669747f37e9075e1e904d8bcf5377e8e790a0fc784e89d.jpg)

![](images/8b733bbe05afb2de46f5fe682c9a5063f644b8f457a0f2f09c0a9474034faec4.jpg)  
(j) Synthetic images

![](images/bc0ed124b7c09ad24c134e9f60cd360c70440ab4a45a24437481e1bd40f45f70.jpg)

![](images/e7e829b37adb9045faa517a9d1630ed1f884a5407ad5ae3c915e29fee085a9e3.jpg)  
Figure 13 | Examples of real and synthetic visual data, covering both daytime and nighttime scenes across various zones within logistics warehouses.

```yaml
Link: https://huggingface.co/OpenGVLab/InternVL3_5-8B
Max len: 32� | Batch size: 1 | backend: Pytorch
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: July 10-20, 2026
```

Table 4 | Inference configurations for proprietary (purple) and open-source (green) LMMs.  
Link: https://openai.com/index/introducing-gpt-5-5/   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: June 30-July 15, 2026

Sonnet Access: https://www.anthropic.com/claude/sonnet   
Opus Access: https://www.anthropic.com/claude/opus   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: June 30-July 15, 2026

Link: https://deepmind.google/models/model-cards/gemini-3-1-pro/   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: June 30-July 15, 2026

Link: https://qwen.ai/blog?id=qwen3.7-plus   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: June 30-July 22, 2026

Link: https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL   
Max len: 32� | Batch size: 1 | backend: Pytorch   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: July 10-15, 2026

Kimi-K2-Thinking & Kimi-K2.6

Link: https://huggingface.co/moonshotai/Kimi-K2.6   
Max len: 32� | Batch size: 1 | backend: Pytorch   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: June 25-July 22, 2026

Link: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf   
Max len: 32� | Batch size: 1 | backend: Pytorch   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: July 10-20, 2026

Qwen Open-Source Suite   
Models: Qwen3.5-Plus, Qwen3.6-Flash, Qwen3-VL (8B, 30B-A3B, 32B, 235B-A22B, Plus)   
Qwen3 Access: https://huggingface.co/collections/Qwen/qwen3   
Qwen3.5 Access: https://huggingface.co/collections/Qwen/qwen35   
Qwen3.6 Access: https://huggingface.co/collections/Qwen/qwen36   
Max len: 32� | Batch size: 1 | backend: Pytorch   
Temperature: 0.2 | Max new tokens (think/non-think): 4096 / 256 | Eval date: June 25-July 22, 2026