# LEGALREWARDBENCH: Benchmarking Reward Models for Legal Preference Evaluation

Hui Huang<sup>1∗</sup>, Jie Zhou<sup>2∗</sup>, Hongli Zhou<sup>1</sup>, Qundong Shi<sup>2</sup>, Shuyuan Zheng<sup>3</sup>, Yini Su<sup>2</sup>, Shaochun Wang<sup>2</sup>, Biyuan Lin<sup>2</sup>, Muyun Yang<sup>1B</sup>, Tiejun Zhao<sup>1</sup>

<sup>1</sup>Harbin Institute of Technology, <sup>2</sup>ModelBest Inc., <sup>3</sup>The University of Osaka {zhoujie, shiqundong, linbiyuan}@modelbest.cn, huanghui@stu.hit.edu.cn, yangmuyun@hit.edu.cn

## Abstract

Reward models are central to preference alignment for legal tasks, yet existing reward model benchmarks are still primarily centered on general-domain preferences. In this work, we introduce LEGALREWARDBENCH, the first human-verified benchmark for evaluating reward models in the legal domain. The benchmark organizes legal preference evaluation<sup>1</sup> into five capability levels: knowledge, understanding, analysis, decision, and writing, and preference examples are constructed through rubric-guided automatic judging combined with human verification. We evaluate three families of judges, covering both general and fine-tuned models, and generative and discriminative architectures. Experimental results show that strong general-purpose generative models are currently the most robust legal reward models, while existing fine-tuned reward models transfer less reliably to legal pref erence evaluation. We also conduct detailed comparisons of input organizations and training strategies.

## 1 Introduction

The rapid advancement of large language models (LLMs) has spurred growing interest in applying them to the legal domain (Cui et al., 2023; Colombo et al., 2024). Existing work has explored a broad spectrum of tasks, including legal judgment prediction (Zhong et al., 2020), legal fact prediction (Liu et al., 2025), and court opinion generation (Dahl et al., 2024). Despite significant progress, benchmarking results consistently reveal that even stateof-the-art LLMs fall short of satisfactory performance (Fei et al., 2024; Guha et al., 2023), underscoring the need for principled methods to improve LLM capabilities in the legal domain.

![](images/5083be1c6a54b929ede6bec49f9543428c5dad6f9dd40474177bf16f7e24af47.jpg)  
Figure 1: Performance of best generative RMs across the five LEGALREWARDBENCH capability levels.

Reinforcement learning from human feedback (RLHF) has emerged as a standard paradigm for LLM post-training (Ouyang et al., 2022; Bai et al., 2022). At the heart of RLHF lies the reward model (RM), which serves as a learned proxy for human preferences and guides the policy model toward preferred behaviors (Zhong et al., 2025; Zhou et al., 2026). Since legal reasoning depends on the theoretical and experiential preferences of legal experts, RLHF holds strong potential for strengthening LLM performance on legal tasks. However, benchmarks for evaluating how well RMs capture legal preferences remain underexplored, leaving the field without a reliable means to measure advances in this direction.

To address this gap, we introduce LEGALRE-WARDBENCH, the first human-verified benchmark specifically designed to evaluate RMs on legal preference assessment. LEGALREWARDBENCH covers five core dimensions of legal preference— knowledge, understanding, analysis, decision, and writing. We leverage a rubric-guided preference annotation strategy, combining both automatic scoring and human verification to ensure annotation precision.

Based on LEGALREWARDBENCH, we evaluate three families of RMs: general LLMs, finetuned generative RMs, and fine-tuned discriminative RMs. The results show that strong generalpurpose LLMs are currently the most robust choice for legal reward modeling, while many fine-tuned RMs transfer only weakly to legal preference evaluation. Ablation experiments further show that reference answers substantially improve legal reward modeling. We further compare training methods, showing that legal RM training brings limited and inconsistent gains. These findings collectively demonstrate that legal reward modeling should be evaluated as a domain-specific problem rather than assumed to transfer from general preference benchmarks.

Our contributions are threefold:

• We introduce LEGALREWARDBENCH, the first human-verified benchmark for legal reward modeling, via a rubric-guided preference annotation strategy;

• We conduct a systematic evaluation of existing reward-model families, showing that general models mostly outperform fine-tuned RMs;

• We conduct detailed ablation studies on inference and training strategies, offering guidance for reward modeling in the legal domain.

## 2 Data Construction

The construction pipeline of LEGALREWARD-BENCH has three stages: 1) query collection, 2) response generation, and 3) preference annotation.

## 2.1 Query Collection

LEGALREWARDBENCH is organized around five legal capability levels inspired by Bloom’s taxonomy (Krathwohl, 2002) and by prior legal evaluation benchmarks. As shown in Table 1, we prioritize source tasks with relatively clear evaluation criteria<sup>2</sup>, categorized into five levels:

1. Knowledge: evaluates whether a model can recall, identify, and apply determinate legal knowledge in legally grounded contexts.

<table><tr><td>Level</td><td>Representative Tasks</td></tr><tr><td>L1 Knowledge</td><td>Statute Recitation; Legal Rule Recall; Applicable Law Identification; Legal</td></tr><tr><td>L2</td><td>Knowledge Error Detection.. Case Event Timeline Extraction; Le-</td></tr><tr><td>Understanding</td><td>gal Element Extraction; Long-Form Legal Document Understanding...</td></tr><tr><td>L3 Analysis</td><td>Fact-Finding Analysis; Disputed Is- sue Identification; Evidence Analy-</td></tr><tr><td>L4</td><td>sis... Sentence Prediction; Judgment Out-</td></tr><tr><td>Decision</td><td>come Prediction; Legal Task Plan- ning...</td></tr><tr><td>L5</td><td>Procuratorial Recommendation</td></tr><tr><td>Writing</td><td>Drafting; Defense Pleading Drafting; Indictment Drafting...</td></tr></table>

Table 1: Representative tasks for the five capability levels in LEGALREWARDBENCH.

2. Understanding: evaluates whether a model can read, preserve, and reorganize legal information from structured or unstructured legal texts.

3. Analysis: evaluates whether a model can decompose legal problems and reason over facts, evidence, arguments, and legal categories.

4. Decision: evaluates whether a model can make or simulate decision-oriented legal judgments under explicit task constraints.

5. Writing: evaluates whether a model can produce structured legal documents under professional legal-writing conventions.

This hierarchy enables fine-grained analysis of RMs across task types. It also makes the final benchmark more diagnostic: a model can be strong at identifying surface-level legal facts while still failing on long-form writing.

The collected queries mainly come from three sources: 1) in-house legal task data; 2) existing legal benchmarks; 3) previous legal exams. Each query is required to have a reference answer.

## 2.2 Response Generation

To expose reward models to realistic response distributions, we ask a diverse set of models to generate responses for each legal task, and then construct preference pairs from the generated candidates. The candidate generation stage covers roughly 30 models, including open and closed models from more than 10 model families, with model sizes ranging from 3B to 1T parameters<sup>3</sup>. This design increases response diversity, exposes a wider range of legal error types, and avoids overly homogeneous preference pairs.

<table><tr><td rowspan="2">Level</td><td rowspan="2">Samples</td><td colspan="3">Average Length</td><td colspan="2">Human-Human Agree.</td><td colspan="2">Human-Model Agree.</td></tr><tr><td>Query</td><td>Chosen</td><td>Rejected</td><td>w/ Tie</td><td>w/o Tie</td><td>w/ Tie</td><td>w/o Tie</td></tr><tr><td>L1 Knowledge</td><td>306</td><td>234.5</td><td>449.9</td><td>435.9</td><td>76.19</td><td>76.19</td><td>81.69</td><td>86.31</td></tr><tr><td>L2 Understanding</td><td>228</td><td>1497.9</td><td>670.5</td><td>566.9</td><td>70.00</td><td>77.78</td><td>73.54</td><td>78.68</td></tr><tr><td>L3 Analysis</td><td>377</td><td>1292.3</td><td>661.7</td><td>446.2</td><td>64.71</td><td>64.71</td><td>77.71</td><td>81.15</td></tr><tr><td>L4 Decision</td><td>296</td><td>566.9</td><td>648.4</td><td>345.5</td><td>76.19</td><td>88.89</td><td>82.35</td><td>88.05</td></tr><tr><td>L5 Writing</td><td>124</td><td>6355.7</td><td>995.0</td><td>874.7</td><td>96.15</td><td>96.15</td><td>90.00</td><td>91.67</td></tr><tr><td>Overall</td><td>1331</td><td>1394.7</td><td>642.6</td><td>482.0</td><td>79.20</td><td>83.90</td><td>79.93</td><td>86.12</td></tr></table>

Table 2: Data statistics and annotation agreement by capability level on LEGALREWARDBENCH.

Many legal queries originally ask only for a final answer, but in reward modeling the reasoning process often reveals errors that are invisible in the final answer alone. We therefore rewrite queries of all levels (except for writing) to ask the model to produce responses in a constrained format with analysis followed by the final answer. This preserves final-answer checkability while making legal reasoning available to preference annotation.

## 2.3 Preference Annotation

The main challenge in legal preference annotation is to make judgments stable, interpretable, and auditable. We therefore introduce a Rubric-guided Annotation scheme<sup>4</sup>, with the following steps:

1. Rubric Construction: we first ask legal-domain annotators to construct rubrics based on the reference answer, task requirements, and legal background knowledge. The rubric clearly defines the essential criteria required to constitute a comprehensive and legally sound response.

2. Automatic Scoring: for verifiable tasks, such as sentence prediction, we design rules to score the candidate responses. For non-verifiable queries, we use a proprietary model to score candidate responses according to the rubrics.

3. Pair Selection: based on the automatic scoring, we construct pairwise examples whose score gap is sufficiently clear while avoiding overly trivial pairs, using the criterion that the chosen score should exceed the rejected score by at least 10% and should itself rank in the top 50%.

4. Human Verification: each pair is assigned to an annotator to evaluate the queries and candidate responses, including their reasoning processes against the predefined rubrics to validate the correctness of the preference. We retain only those samples where the human verification aligns with the automatic preference.

The data statistics are presented in the next section. Compared with purely subjective preference labels, this rubric-based design helps constrain the evaluation around legal criteria.

## 2.4 Data Statistics

As shown in Table 2, we obtain 1,331 verified preference pairs across five capability levels as the final LEGALREWARDBENCH, with each sample consisting of a quadruplet with (query, chosen response, rejected response, reference answer)<sup>5</sup>. The substantial length of queries and responses demands that reward models perform robust comparisons under long-context and information-dense conditions.

To assess annotation reliability, we conduct an additional double-blind check on 5% of the preference pairs, as shown in Table 2. As can be seen, both double-blind human agreement and human agreement with automatic annotation reach an agreement around 80%, verifying the effectiveness of the rubric-guided annotation scheme.

We further analyze the model distribution of chosen and rejected responses in Figure 2. As can be seen, strong closed-source models account for a larger share of chosen responses, while mediumscale models appear more often among rejected responses. At the same time, the overall model distribution remains diverse, indicating the comprehensiveness of the responses.

![](images/759095827501f1df5a7821d59a22ec7bf35c11503b8c2bb1f97b90c85c76fa7a.jpg)  
Figure 2: Model distribution of candidate responses.

## 3 Benchmarking

## 3.1 Evaluation Setup

We conduct evaluation on LEGALREWARDBENCH with three families of RMs<sup>6</sup>:

1. General-purpose Generative Reward Models (General GenRM): These refer to general LLMs (e.g., Qwen-3 (Yang et al., 2025)) that perform evaluation via zero-shot prompting. We apply a standardized prompt template across this category to ensure a fair comparison.

2. Fine-tuned Generative Reward Models (Finetuned GenRM): These models are explicitly fine-tuned to produce textual judgment or scores such as Mahan et al. (2024). We evaluate them using their native templates to align with their specific fine-tuning designs.

3. Fine-tuned Discriminative Reward Models (Fine-tuned DisRM): These models are trained to output scalar scores with a newly initialized classification head such as Liu et al. (2024). They typically process instruction-response pairs directly without evaluation prompts.

We use accuracy on pairwise preference prediction as the primary metric<sup>7</sup>. To mitigate position bias (Shi et al., 2025; Zhou et al., 2027) for generative models, we evaluate each pair twice with swapped response order and average the two results.

## 3.2 Main Results

Table 3 reports the results on LEGALREWARD-BENCH. We summarize our findings as follows:

Finding 1: General GenRMs substantially outperform most fine-tuned reward models. Topperforming models are mainly from the Claude, GPT, Gemini, and Qwen3 families, where the most updated Claude-Opus-4.7 achieves nearly 87 points. On the other hand, fine-tuned generative and discriminative RMs perform more modestly, and neither family clearly dominates the other. This differs from previous RM benchmarks, where strong discriminative RMs can achieve top performance (Lambert et al., 2025; Malik et al., 2025). This likely stems from the lack of high-quality legal preference data, while general-domain preferences hardly transfer to the legal domain.

Finding 2: Legal information extraction and generation remain difficult for legal preference annotation. RMs perform comparatively better on knowledge and analysis tasks, likely benefiting from the knowledge and reasoning abilities of recent LLMs (Sun et al., 2025). On the other hand, RMs struggle on understanding and writing related tasks. This demonstrates that the current models still struggle on information extraction and generation tasks, which cannot be easily improved by scaled model parameters or reasoning length.

Finding 3: Legal-specialized LLMs do not necessarily become strong legal RMs. LegalOne-8B (Li et al., 2026), for example, does not show a clear advantage in this benchmark. This suggests that solving legal tasks and evaluating legal responses are not necessarily related capabilities.

## 3.3 Analysis

## 3.3.1 Effect of Reference Answers

As detailed in Section 2, we curate a reference answer for every query in our benchmark. To evaluate the impact of these references, we compare the baseline setting (where the RM receives only the query and candidate responses) against the reference-augmented setting (where the reference is included in the input), as shown in Table 4.

<table><tr><td>Model</td><td>Type</td><td>TK</td><td>Overall</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td></tr><tr><td>Claude-Opus-4.7</td><td>General LLM</td><td></td><td>86.85</td><td>91.83</td><td>81.36</td><td>87.00</td><td>87.50</td><td>82.66</td></tr><tr><td>Claude-Opus-4.5</td><td>General LLM</td><td>√</td><td>85.01</td><td>91.01</td><td>76.75</td><td>86.60</td><td>84.12</td><td>82.66</td></tr><tr><td>GPT-5.5</td><td>General LLM</td><td>√</td><td>83.66</td><td>92.97</td><td>78.73</td><td>88.06</td><td>72.97</td><td>81.85</td></tr><tr><td>Gemini-3-Flash-Preview</td><td>General LLM</td><td></td><td>82.95</td><td>89.22</td><td>74.12</td><td>84.88</td><td>82.09</td><td>79.84</td></tr><tr><td>Qwen3-235B-A22B</td><td>General LLM</td><td></td><td>81.52</td><td>88.40</td><td>73.25</td><td>84.08</td><td>77.87</td><td>80.65</td></tr><tr><td>Gemini-2.5-Pro</td><td>General LLM</td><td>X</td><td>81.14</td><td>88.40</td><td>75.22</td><td>83.55</td><td>79.56</td><td>70.56</td></tr><tr><td>Kimi-K2.5</td><td>General LLM</td><td></td><td>80.65</td><td>87.91</td><td>71.27</td><td>83.82</td><td>80.07</td><td>71.77</td></tr><tr><td>GPT-5.2</td><td>General LLM</td><td></td><td>80.54</td><td>84.15</td><td>75.44</td><td>83.02</td><td>78.89</td><td>77.42</td></tr><tr><td>GLM-4.5</td><td>General LLM</td><td></td><td>79.90</td><td>82.19</td><td>71.93</td><td>83.95</td><td>79.73</td><td>77.02</td></tr><tr><td>Qwen3-30B-A3B</td><td>General LLM</td><td>√</td><td>79.30</td><td>78.92</td><td>74.78</td><td>82.49</td><td>79.90</td><td>77.42</td></tr><tr><td>GPT-4.1</td><td>General LLM</td><td>X</td><td>78.96</td><td>77.94</td><td>72.15</td><td>83.42</td><td>83.11</td><td>70.56</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>General LLM</td><td>√</td><td>78.55</td><td>84.80</td><td>73.68</td><td>80.77</td><td>78.38</td><td>65.73</td></tr><tr><td>Gemini-2.5-Flash</td><td>General LLM</td><td>√</td><td>78.51</td><td>81.54</td><td>73.68</td><td>83.42</td><td>76.01</td><td>70.97</td></tr><tr><td>DeepSeek-R1</td><td>General LLM</td><td>√</td><td>78.29</td><td>80.23</td><td>73.25</td><td>82.76</td><td>77.53</td><td>70.97</td></tr><tr><td>Qwen3-Nemotron-8B-BRRM</td><td>Ft. GenRM</td><td></td><td>78.25</td><td>78.43</td><td>73.46</td><td>83.16</td><td>76.52</td><td>75.81</td></tr><tr><td>Skywork-Reward-V2-Qwen3-8B</td><td>Ft. DisRM</td><td></td><td>78.14</td><td>77.12</td><td>70.61</td><td>84.08</td><td>79.73</td><td>72.58</td></tr><tr><td>DeepSeek-v3.2</td><td>General LLM</td><td></td><td>77.84</td><td>80.39</td><td>71.05</td><td>80.77</td><td>77.70</td><td>75.40</td></tr><tr><td>Qwēn3-14B</td><td>General LLM</td><td>√</td><td>77.84</td><td>76.63</td><td>70.39</td><td>85.01</td><td>76.69</td><td>75.40</td></tr><tr><td>Qwen3-8B</td><td>General LLM</td><td>√</td><td>75.69</td><td>76.47</td><td>68.86</td><td>79.97</td><td>76.86</td><td>70.56</td></tr><tr><td>Qwen3.5-27B</td><td>General LLM</td><td>√</td><td>75.69</td><td>81.70</td><td>65.35</td><td>81.17</td><td>75.84</td><td>62.90</td></tr><tr><td>Claude-Sonnet-4</td><td>General LLM</td><td>X</td><td>75.51</td><td>77.61</td><td>71.71</td><td>78.91</td><td>72.80</td><td>73.39</td></tr><tr><td>Qwen3-4B</td><td>General LLM</td><td>√</td><td>74.87</td><td>71.90</td><td>70.18</td><td>82.23</td><td>74.66</td><td>68.95</td></tr><tr><td>LegalOne-8B</td><td>General LLM</td><td>X</td><td>74.68</td><td>83.50</td><td>65.79</td><td>80.11</td><td>73.82</td><td>54.84</td></tr><tr><td>GPT-4o</td><td>General LLM</td><td>X</td><td>74.61</td><td>73.53</td><td>69.74</td><td>79.18</td><td>76.18</td><td>68.55</td></tr><tr><td>Kimi-K2</td><td>General LLM</td><td>√</td><td>74.08</td><td>77.61</td><td>65.35</td><td>78.65</td><td>72.97</td><td>70.16</td></tr><tr><td>Gemma-3-12B-IT</td><td>General LLM</td><td>X</td><td>72.35</td><td>69.61</td><td>68.42</td><td>78.65</td><td>72.80</td><td>66.13</td></tr><tr><td>Skywork-Reward-V2-Llama-3.2-3B</td><td>Ft. DisRM</td><td></td><td>71.37</td><td>65.69</td><td>71.49</td><td>77.19</td><td>71.62</td><td>66.94</td></tr><tr><td>Skywork-Reward-V2-Llama-3.1-8B</td><td>Ft. DisRM</td><td></td><td>70.85</td><td>68.30</td><td>64.91</td><td>75.60</td><td>75.00</td><td>63.71</td></tr><tr><td>RRM-7B</td><td>Ft. GenRM</td><td></td><td>69.20</td><td>64.87</td><td>63.60</td><td>75.07</td><td>73.14</td><td>62.90</td></tr><tr><td>Llama-3.1-8B-Instruct-RM-RB2</td><td>Ft. DisRM</td><td></td><td>68.90</td><td>60.46</td><td>67.11</td><td>75.07</td><td>72.97</td><td>64.52</td></tr><tr><td>RM-R1-Qwen2.5-Instruct-7B</td><td>Ft. GenRM</td><td></td><td>68.22</td><td>66.67</td><td>66.45</td><td>69.10</td><td>70.61</td><td>66.94</td></tr><tr><td>Qwen3-1.7B Llama-3.1-8B-Instruct</td><td>General LLM</td><td>x</td><td>67.66</td><td>63.24</td><td>61.40</td><td>71.22</td><td>74.49</td><td>62.90</td></tr><tr><td>Skywork-Reward-Llama-3.1-8B-v0.2</td><td>General LLM</td><td>X</td><td>67.36</td><td>68.30</td><td>64.69</td><td>68.30</td><td>68.07</td><td>65.32</td></tr><tr><td></td><td>Ft. DisRM</td><td></td><td>67.02</td><td>61.44</td><td>62.28</td><td>67.90</td><td>72.64</td><td>73.39</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>General LLM</td><td>X</td><td>65.78</td><td>62.58</td><td>60.75</td><td>67.11</td><td>69.59</td><td>69.76</td></tr><tr><td>InternLM2-1.8B-Reward</td><td>Ft. DisRM</td><td>一</td><td>62.58</td><td>58.82</td><td>64.04</td><td>68.44</td><td>63.18</td><td>50.00</td></tr><tr><td>Llama-3.2-3B-Instruct Qwen3-0.6B</td><td>General LLM General LLM</td><td>x √</td><td>56.20 55.41</td><td>60.29 56.21</td><td>58.11 51.97</td><td>55.97 50.13</td><td>56.93 63.51</td><td>41.53 56.45</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Main results on LEGALREWARDBENCH. Values are accuracies in percentage points.

Finding 4: Reference answers significantly improve legal reward modeling. This improvement is particularly pronounced in knowledge, understanding, and decision tasks. This aligns with the inherent characteristics of the legal tasks: unlike general open-ended tasks that lack a single standard response, legal queries are typically grounded in specific statutes and precedents, leading to more determinate reference answers. Therefore, reference answers should be leveraged whenever available in legal reward modeling.

## 3.3.2 Effect of Thinking Enhancement

Legal tasks inherently require extensive reasoning. We therefore compare Think and No Think settings for representative models while keeping other settings fixed<sup>8</sup>. We also test whether the language of reasoning process affects results by explicitly specifying the reasoning language. The results are presented in Tables 5 and 6.

Finding 5: Thinking enhancement usually improves performance on LEGALREWARDBENCH. This suggests that explicit deliberation helps the RM compare legal reasoning, detect unsupported claims, and avoid being misled by surface fluency. On the other hand, no consistent advantage is seen for thinking in a specific language. Although the dataset is in Chinese, legal reward modeling appears to depend more on the precision of the thinking process rather than the thinking language.

<table><tr><td rowspan="2">Model</td><td colspan="2">Overall</td><td colspan="2">L1</td><td colspan="2">L2</td><td colspan="2">L3</td><td colspan="2">L4</td><td colspan="2">L5</td></tr><tr><td>Ori</td><td>Ref</td><td>Ori</td><td>Ref</td><td>Ori</td><td>Ref</td><td>Ori</td><td>Ref</td><td>Ori</td><td>Ref</td><td>Ori</td><td>Ref</td></tr><tr><td>Claude-Opus-4.7</td><td>86.85</td><td>91.81</td><td>91.83</td><td>94.93</td><td>81.36</td><td>90.13</td><td>87.00</td><td>88.99</td><td>87.50</td><td>96.11</td><td>82.66</td><td>85.48</td></tr><tr><td>GPT-40</td><td>74.61</td><td>79.30</td><td>73.53</td><td>84.15</td><td>69.74</td><td>76.10</td><td>79.18</td><td>79.97</td><td>76.18</td><td>81.76</td><td>68.55</td><td>65.32</td></tr><tr><td>GPT-5.2</td><td>80.54</td><td>88.73</td><td>84.15</td><td>90.52</td><td>75.44</td><td>84.43</td><td>83.02</td><td>87.53</td><td>78.89</td><td>93.92</td><td>77.42</td><td>83.47</td></tr><tr><td>Kimi-K2.5</td><td>80.65</td><td>83.43</td><td>87.91</td><td>89.05</td><td>71.27</td><td>73.90</td><td>83.82</td><td>85.68</td><td>80.07</td><td>89.19</td><td>71.77</td><td>66.53</td></tr><tr><td>Qwen3-14B</td><td>77.84</td><td>86.14</td><td>76.63</td><td>90.69</td><td>70.39</td><td>82.46</td><td>85.01</td><td>87.80</td><td>76.69</td><td>85.47</td><td>75.40</td><td>78.23</td></tr></table>

Table 4: Averaged accuracy on LEGALREWARDBENCH with reference included or excluded.
<table><tr><td rowspan="2">Model</td><td colspan="2">Overall</td><td colspan="2">L1</td><td colspan="2">L2</td><td colspan="2">L3</td><td colspan="2">L4</td><td colspan="2">L5</td></tr><tr><td>Tk</td><td>NoTk</td><td>Tk</td><td>NoTk</td><td>Tk</td><td>NoTk</td><td>Tk</td><td>NoTk</td><td>Tk</td><td>NoTk</td><td>Tk</td><td>NoTk</td></tr><tr><td>Claude-Opus-4.7</td><td>86.85</td><td>86.18</td><td>91.83</td><td>90.52</td><td>81.36</td><td>81.58</td><td>87.00</td><td>86.60</td><td>87.50</td><td>86.66</td><td>82.66</td><td>81.45</td></tr><tr><td>Gemini-3-Flash</td><td>82.95</td><td>82.34</td><td>89.22</td><td>88.73</td><td>74.12</td><td>73.25</td><td>84.88</td><td>84.22</td><td>82.09</td><td>82.26</td><td>79.84</td><td>77.82</td></tr><tr><td>Qwen3-14B</td><td>77.84</td><td>76.97</td><td>76.63</td><td>74.67</td><td>70.39</td><td>72.15</td><td>85.01</td><td>80.64</td><td>76.69</td><td>77.53</td><td>75.40</td><td>79.03</td></tr><tr><td>Qwen3-4B</td><td>74.87</td><td>73.14</td><td>71.90</td><td>70.26</td><td>70.18</td><td>69.08</td><td>82.23</td><td>81.70</td><td>74.66</td><td>72.30</td><td>68.95</td><td>63.71</td></tr><tr><td>Qwen3-8B</td><td>75.69</td><td>76.60</td><td>76.47</td><td>75.49</td><td>68.86</td><td>71.93</td><td>79.97</td><td>82.49</td><td>76.86</td><td>76.01</td><td>70.56</td><td>71.37</td></tr></table>

Table 5: Averaged accuracy on LEGALREWARDBENCH with thinking mode on or off.
<table><tr><td rowspan="2">Model</td><td colspan="2">Overall</td><td colspan="2">L1</td><td colspan="2">L2</td><td colspan="2">L3</td><td colspan="2">L4</td><td colspan="2">L5</td></tr><tr><td>Zh</td><td>En</td><td>Zh</td><td>En</td><td>Zh</td><td>En</td><td>Zh</td><td>En</td><td>Zh</td><td>En</td><td>Zh</td><td>En</td></tr><tr><td>Claude-Opus-4.7</td><td>86.85</td><td>85.54</td><td>91.83</td><td>89.71</td><td>81.36</td><td>81.58</td><td>87.00</td><td>85.94</td><td>87.50</td><td>84.97</td><td>82.66</td><td>82.66</td></tr><tr><td>GPT-40</td><td>74.61</td><td>74.98</td><td>73.53</td><td>71.73</td><td>69.74</td><td>70.61</td><td>79.18</td><td>79.44</td><td>76.18</td><td>79.05</td><td>68.55</td><td>67.74</td></tr><tr><td>GPT-5.2</td><td>80.54</td><td>80.09</td><td>84.15</td><td>83.82</td><td>75.44</td><td>75.66</td><td>83.02</td><td>82.49</td><td>78.89</td><td>77.36</td><td>77.42</td><td>78.23</td></tr><tr><td>Kimi-K2.5</td><td>80.65</td><td>78.36</td><td>87.91</td><td>85.46</td><td>71.27</td><td>69.52</td><td>83.82</td><td>80.77</td><td>80.07</td><td>79.22</td><td>71.77</td><td>67.74</td></tr><tr><td>Qwen3-14B</td><td>77.84</td><td>78.44</td><td>76.63</td><td>77.78</td><td>70.39</td><td>74.12</td><td>85.01</td><td>85.28</td><td>76.69</td><td>76.18</td><td>75.40</td><td>72.58</td></tr></table>

Table 6: Averaged accuracy on LEGALREWARDBENCH with different thinking languages.
<table><tr><td rowspan="2">Model</td><td colspan="2">Overall</td><td colspan="2">L1</td><td colspan="2">L2</td><td colspan="2">L3</td><td colspan="2">L4</td><td colspan="2">L5</td></tr><tr><td>Ori</td><td>Rub</td><td>Ori</td><td>Rub</td><td>Ori</td><td>Rub</td><td>Ori</td><td>Rub</td><td>Ori</td><td>Rub</td><td>Ori</td><td>Rub</td></tr><tr><td>Claude-Opus-4.7</td><td>86.85</td><td>85.99</td><td>91.83</td><td>87.75</td><td>81.36</td><td>80.48</td><td>87.00</td><td>87.14</td><td>87.50</td><td>88.85</td><td>82.66</td><td>81.45</td></tr><tr><td>GPT-40</td><td>74.61</td><td>75.88</td><td>73.53</td><td>72.55</td><td>69.74</td><td>71.49</td><td>79.18</td><td>81.70</td><td>76.18</td><td>79.22</td><td>68.55</td><td>66.53</td></tr><tr><td>GPT-5.2</td><td>80.54</td><td>80.73</td><td>84.15</td><td>82.19</td><td>75.44</td><td>76.32</td><td>83.02</td><td>81.56</td><td>78.89</td><td>81.76</td><td>77.42</td><td>80.24</td></tr><tr><td>Kimi-K2.5</td><td>80.65</td><td>71.83</td><td>87.91</td><td>79.74</td><td>71.27</td><td>62.94</td><td>83.82</td><td>74.40</td><td>80.07</td><td>75.00</td><td>71.77</td><td>53.23</td></tr><tr><td>Qwen3-14B</td><td>77.84</td><td>78.21</td><td>76.63</td><td>76.14</td><td>70.39</td><td>73.46</td><td>85.01</td><td>82.89</td><td>76.69</td><td>78.55</td><td>75.40</td><td>77.02</td></tr></table>

Table 7: Averaged accuracy on LEGALREWARDBENCH with automatic rubric generation.

## 3.3.3 Effect of Rubric Generation

As discussed in Section 2.3, the integration of rubrics is instrumental in enhancing annotation consistency. Therefore, we further investigate whether autonomous rubric generation can also improve RM performance. Specifically, we ask the generative LLMs to first write a brief rubric for each query and then compare the candidate responses based on the rubric. This turns the evaluation into a two-step process. The results are in Table 7.

Finding 6: Model-generated rubrics do not consistently improve performance. A likely explanation is that human-written legal rubrics are highly task-specific, while model-generated rubrics tend to remain generic, emphasizing broad criteria such as clarity and completeness. Such generic rubrics may fail to capture concrete legal issues. Despite the potential of rubrics to further improve RM accuracy (Gunjal et al., 2025), it is non-trivial to obtain effective rubrics for legal reward modeling.

## 3.3.4 Best-of-n Validation

We further evaluate whether scores on LEGAL-REWARDBENCH serve as a reliable predictor for downstream performance gains when an RM is employed for model alignment. Specifically, we conduct a Best-of-n validation on LawBench (Fei et al., 2024) and LexEval (Li et al., 2024). We construct a 50-sample subset for each external benchmark and compare representative RMs in a best-of-4 setting due to computational constraints. The overall results are reported in Table 8.

<table><tr><td>Rank</td><td>Ori_Rank</td><td>Base Model</td><td>Sampler</td><td>Average</td><td>LawBench</td><td>LexEval</td></tr><tr><td>1</td><td>1</td><td>Qwen3-4B</td><td>Claude-Opus-4.5</td><td>54.29</td><td>52.12</td><td>56.46</td></tr><tr><td>2</td><td>3</td><td>Qwen3-4B</td><td>GPT-5.2</td><td>54.14</td><td>51.26</td><td>57.02</td></tr><tr><td>3</td><td>2</td><td>Qwen3-4B</td><td>Gemini-3-Flash-Preview</td><td>54.00</td><td>51.26</td><td>56.74</td></tr><tr><td>4</td><td>4</td><td>Qwen3-4B</td><td>GPT-4.1</td><td>52.70</td><td>49.65</td><td>55.75</td></tr><tr><td>5</td><td>5</td><td>Qwen3-4B</td><td>Skywork-Reward-V2-Qwen3-8B</td><td>52.38</td><td>49.23</td><td>55.52</td></tr><tr><td>6</td><td>7</td><td>Qwen3-4B</td><td>GPT-40</td><td>52.19</td><td>49.54</td><td>54.84</td></tr><tr><td>7</td><td>6</td><td>Qwen3-4B</td><td>Qwen3-8B</td><td>51.87</td><td>49.43</td><td>54.32</td></tr><tr><td>8</td><td>8</td><td>Qwen3-4B</td><td>Skywork-Reward-V2-Llama-3.2-3B</td><td>51.47</td><td>48.27</td><td>54.67</td></tr><tr><td>9</td><td>9</td><td>Qwen3-4B</td><td>Skywork-Reward-V2-Llama-3.1-8B</td><td>51.28</td><td>48.68</td><td>53.88</td></tr><tr><td>一</td><td>一</td><td>Qwen3-4B</td><td>Best-of-1 Baseline</td><td>51.02</td><td>48.19</td><td>53.85</td></tr><tr><td>10</td><td>10</td><td>Qwen3-4B</td><td>Skywork-Reward-Llama-3.1-8B-v0.2</td><td>50.94</td><td>49.21</td><td>52.67</td></tr></table>

Table 8: Best-of-4 sampling validation on LawBench and LexEval, with Qwen3-4B as the base model and different RMs as the sampler. Ori\_Rank denotes the RM’s original rank on LEGALREWARDBENCH.

Finding 7: Performance on LEGALREWARD-BENCH strongly correlates with downstream sampling effectiveness. The trends are highly consistent across both benchmarks, with a Spearman correlation of 0.9758 between LEGALRE-WARDBENCH ranking and downstream selection gains with the 10 selected RMs. This supports the practical validity of the benchmark: it measures differences among legal RMs that are strongly reflected in actual response selection.

## 4 Model Training

Despite the underperformance of general-domain fine-tuned RMs on LEGALREWARDBENCH, in this section, we investigate whether we can train a legaldomain reward model that transfers more effectively to legal preference judgments.

## 4.1 Training Setup

To enable reliable, reference-based preference annotation, we prioritize legal queries accompanied by authoritative reference answers. Concretely, we aggregate raw data from two primary sources: legal evaluation data from LegalDelta (Dai et al., 2025) and in-house judicial examination data.

We further implement a rigorous preprocessing pipeline: 1) Remove instances with low-quality queries and reference answers. 2) Filter out examples highly similar to LEGALREWARDBENCH to prevent evaluation data leakage. 3) Rewrite all queries into a unified format that instructs the model to explicitly reason before answering. 4) Reduce the proportion of simplistic objective tasks such as multi-choice questions to less than 30%.

Following preprocessing, we construct preference pairs from the cleaned query-reference dataset.

For each query, we sample 6 responses from two distinct base models and evaluate them using GPT-5.2 under a reference-based scoring protocol<sup>9</sup>. The highest- and lowest-scoring responses are designated as the chosen and rejected instances, respectively. This pipeline yields a final training set of approximately 39K high-quality preference pairs, consisting of 33K exam data-based pairs and 6K pairs derived from LegalDelta.

Using Qwen3-8B as the backbone, we compare two prominent RM training paradigms:

1. Discriminative RM: This approach trains a sequence classifier with a scalar scoring head. Specifically, we incorporate Bradley-Terry objective (Liu et al., 2024) to learn preference scoring based on pairwise data.

2. Generative RM: This approach reformulates reward modeling as a generative preference optimization task (Huang et al., 2026). Specifically, we leverage a GRPO-based objective (Shao et al., 2024), utilizing the correctness of the preference prediction as the reward signal.

## 4.2 Results and Analysis

Finding 8: It is non-trivial to fine-tune a legaldomain reward model. As shown in Table 9, RM training brings limited and inconsistent gains. GenRM training yields only marginal improvements over the prompt-based judge, while DisRM training can even underperform it. These results suggest that simply converting reference-scored legal data into preference pairs is insufficient for cultivating a superior legal RM. Notably, the LegalDelta data was successfully applied for LLM post-training with substantial improvements (Dai et al., 2025), which largely eliminates poor query quality as the primary bottleneck for our RM’s underperformance.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Data</td><td colspan="6">Qwen3-8B</td><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Overall</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>Overall</td></tr><tr><td>Prompt w/ reference</td><td></td><td>76.47 88.42</td><td>68.86</td><td>79.97</td><td>76.86</td><td>70.56</td><td>75.69</td><td>62.58</td><td>60.75</td><td>67.11</td><td>69.59</td><td>69.76</td><td>65.78</td></tr><tr><td rowspan="2">DisRM</td><td></td><td></td><td>78.98</td><td>83.38</td><td>82.61</td><td>69.01</td><td>82.26</td><td>76.31</td><td>64.91</td><td>70.03</td><td>72.64</td><td>69.35</td><td>71.11</td></tr><tr><td>LegalDelta 6K Combined 39K</td><td>72.55 78.43</td><td>60.96</td><td>65.78</td><td>73.65</td><td>67.74</td><td>68.44</td><td>76.14</td><td>73.25</td><td>73.47</td><td>73.31</td><td>62.90</td><td>73.03</td></tr><tr><td rowspan="2">GenRM</td><td></td><td></td><td>64.91</td><td>72.41</td><td>77.03</td><td>66.13</td><td>72.95</td><td>73.86</td><td>64.47</td><td>74.01</td><td>76.35</td><td>70.97</td><td>72.58</td></tr><tr><td>LegalDelta 6K Combined 39K</td><td>74.02 74.51</td><td>69.74 76.10</td><td>82.49 83.69</td><td>77.87 81.25</td><td>69.76 75.40</td><td>76.15 78.96</td><td>65.69 70.10</td><td>56.58 68.20</td><td>62.07 74.27</td><td>72.13 71.45</td><td>58.06 65.32</td><td>63.82 70.81</td></tr></table>

Table 9: Comparison of different RM training methods on LEGALREWARDBENCH.

In summary, effective legal reward modeling extends far beyond merely aggregating domainspecific data. In contrast, simply introducing reference answers yields a performance boost of nearly 7 percentage points without the risk of compromising the model’s inherent generalization capabilities (Huang et al., 2025). Therefore, for practical deployments, directly leveraging strong off-the-shelf generative models coupled with prompting techniques presents a more robust alternative.

## 5 Related Work

## 5.1 Reward Model Benchmarks

With the prevalence of RM-based reinforcement learning during LLM alignment process (Dong et al., 2024), systematic evaluation of reward models has grown rapidly in recent years. Reward-Bench (Lambert et al., 2025) established the first open evaluation framework for comparisons among different reward-model types. RewardBench 2 (Malik et al., 2025) further introduced more difficult examples and broader skill coverage.

Beyond general RM evaluation, recent work has extended RM benchmarks to specialized settings. IF-RewardBench focuses on instruction-following evaluation (Wen et al., 2026); Multimodal Reward-Bench extends reward modeling to multimodal scenarios (Yasunaga et al., 2025); RAG-RewardBench studies multi-hop reasoning, citation faithfulness, and robustness in retrieval-augmented generation (Jin et al., 2024); and Agent-RewardBench evaluates reward models in agentic tasks (Lù et al., 2025). These benchmarks show that RM evaluation is highly dependent on task and domain structure. In the legal domain, LRB (Franzone et al., 2026) constructs retrieval-grounded preference pairs labeled solely by LLM judges, but a human-verified

RM benchmark is still lacking.

## 5.2 Reward Modeling in the Legal Domain

As a critical application domain for natural language processing (NLP), various legal NLP benchmarks have been established. LawBench (Fei et al., 2024) serves as a comprehensive benchmark for assessing LLMs in the Chinese legal field, evaluating their ability from three dimensions. Similarly, LexEval (Li et al., 2024) provides a framework for measuring the cognitive and practical capabilities of LLMs across six dimensions. However, existing works primarily focus on whether a model can perform legal tasks instead of reward modeling.

Recent work has also explored reinforcement learning (RL) based alignment for legal models. LegalDelta improves legal reasoning through chainof-thought-guided information gain and reward signals from verifiable legal tasks (Dai et al., 2025). Unilaw-R1 conducts reinforcement learning combining format rewards, accuracy rewards, and legaleffect rewards (Cai et al., 2025). These works demonstrate that RL-based legal post-training is useful and necessary, but the design of more effective RMs for legal alignment remains a gap.

## 6 Conclusion

In this work, we introduce LEGALREWARD-BENCH, the first human-verified benchmark for reward modeling in the legal domain. Our evaluation shows that strong general-purpose LLMs are currently the most robust RMs for the legal domain, while existing fine-tuned RMs transfer less reliably to legal tasks, and it is non-trivial to finetune legal-domain RMs. Considering the lack of high-quality preference data and the knowledgeand reasoning-intensive nature of legal tasks, we strongly recommend using existing general models with knowledge and reasoning enhancement for legal reward modeling.

## Limitations

This work still has several limitations:

First, the current version mainly covers Chinese legal tasks due to practical constraints: high-quality legal annotation requires annotators who understand both the language and the relevant legal system, and it is difficult to obtain reliable annotation resources for multiple jurisdictions at the same time. Future versions should extend LEGALRE-WARDBENCH to more languages and legal systems.

Second, although we discuss preliminary directions for legal reward-model training, this paper does not introduce a new training method or report complete training results. Future work should investigate stronger legal reward-model systems, including knowledge-enhanced judges, higher-quality legal rubrics, reference-aware reward models, and training data designed specifically for legal preference judgment.

## Ethics Considerations

This work involves several ethical considerations:

Data and Consent Evaluation queries of the benchmark primarily originate from three sources: in-house legal task data, existing legal benchmarks, and historical judicial examination materials. The training phase additionally incorporates LegalDelta and in-house exam materials. We process publicsource data strictly when permitted by the original platform or publisher for research, crawling, or redistribution, ensuring full compliance with respective data-use agreements, platform terms, and open-source licenses. For internal data, our research team ensures legitimate authorization and strictly controlled data circulation.

Furthermore, we recognize copyright and licensing constraints, acknowledging that not all legal texts, examination materials, or benchmark data are exempt from reuse restrictions.

Human Annotators and Crowdsourcing The benchmark construction involves a rigorous humanverification stage where annotators with domain expertise participate in rubric creation and pairwise preference validation. All annotators were fully informed that their contributions would be used for academic research, ensuring clear comprehension of the task, workload, and data usage. Compensation was fairly commensurate with the time and effort required, strictly adhering to reasonable hourly wage standards to prevent exploitation.

In addition, we acknowledge that human annotation inherently risks introducing subjectivity and institutional bias. Although we mitigate these risks via rubric-guided evaluation, hybrid automatichuman review, and double-blind agreement checks, we emphasize that human annotations represent constrained research preference signals rather than absolute, definitive legal judgments.

Potential Harms and Broader Impact The primary objective of this work is to benchmark reward models within the legal domain. However, advanced legal reward models could potentially be integrated into downstream applications for posttraining or answer filtering. This poses a risk of misuse in high-stakes decision-making scenarios, potentially generating plausible yet legally unsound outputs that amplify user over-reliance.

Additionally, bias and fairness remain unavoidable challenges. Legal corpora, historical exams, reference answers, rubrics, and annotator judgments may implicitly harbor institutional biases or stylistic inclinations. Therefore, we explicitly position this work as a rigorous academic evaluation tool, rather than a foundation for autonomous decision-making in real-world legal practices.

## References

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Hua Cai, Shuang Zhao, Liang Zhang, Xuli Shen, Qing Xu, Weilin Shen, Zihao Wen, and Tianke Ban. 2025. Unilaw-R1: A large language model for legal reasoning with reinforcement learning and iterative inference. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 18128–18142.

Pierre Colombo, Telmo Pires, Malik Boudiaf, Dominic Culver, Rui Melo, Caio Corro, Andre F. T. Martins, Fabrizio Esposito, Vera Lúcia Raposo, Sofia Morgado, et al. 2024. SaulLM-7B: A pioneering large language model for law. arXiv preprint arXiv:2403.03883.

Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan. 2023. ChatLaw: Open-source legal large language model with integrated external knowledge bases. arXiv preprint arXiv:2306.16092.

Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E. Ho. 2024. Large legal fictions: Profiling le-

gal hallucinations in large language models. Journal ofLegal Analysis, 16(1):64–93.

Xin Dai, Buqiang Xu, Zhenghao Liu, Yukun Yan, Huiyuan Xie, Xiaoyuan Yi, Shuo Wang, and Ge Yu. 2025. Legal δ: Enhancing legal reasoning in LLMs via reinforcement learning with chain-ofthought guided information gain. arXiv preprint arXiv:2508.12281.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. 2024. RLHF workflow: From reward modeling to online RLHF. Preprint, arXiv:2405.07863.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, et al. 2024. LawBench: Benchmarking legal knowledge of large language models. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 7933–7962.

Rilton Franzone, Valentin Noël, Puyu Wang, Philip Torr, and Fabio J. Fehr. 2026. Building legal reward models for grounding and abstention. Preprint, arXiv:2609.14739.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, et al. 2023. Legal-Bench: A collaboratively built benchmark for measuring legal reasoning in large language models. arXiv preprint arXiv:2308.11462.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. 2025. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746.

Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang, Bing Xu, and Tiejun Zhao. 2025. An empirical study of LLM-as-a-judge for LLM evaluation: Fine-tuned judge model is not a general substitute for GPT-4. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 5880–5895.

Hui Huang, Yancheng He, Hongli Zhou, Rui Zhang, Wei Liu, Weixun Wang, Jiaheng Liu, and Wenbo Su. 2026. Think-J: Learning to think for generative LLMas-a-judge. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 31158– 31166.

Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. 2024. RAG-RewardBench: Benchmarking reward models in retrieval augmented generation for preference alignment. arXiv preprint arXiv:2412.13746.

David R Krathwohl. 2002. A revision of Bloom’s taxonomy: An overview. Theory into practice, 41(4):212– 218.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, Lester James Validad Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. 2025. RewardBench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1755–1797.

Haitao Li, Yifan Chen, Shuo Miao, Qian Dong, Jia Chen, Yiran Hu, Junjie Chen, Minghao Qin, Yueyue Wu, Yujia Zhou, et al. 2026. LegalOne: a family of foundation models for reliable legal reasoning. arXiv preprint arXiv:2602.00642.

Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, and Yiqun Liu. 2024. LexEval: A comprehensive Chinese legal benchmark for evaluating large language models. Advances in Neural Information Processing Systems, 37:25061–25094.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. 2024. Skywork-Reward: Bag of tricks for reward modeling in LLMs. arXiv preprint arXiv:2410.18451.

Junkai Liu, Yujie Tong, Hui Huang, Bowen Zheng, Yiran Hu, Peicheng Wu, Chuan Xiao, Makoto Onizuka, Muyun Yang, and Shuyuan Zheng. 2025. Legal fact prediction: the missing piece in legal judgment prediction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6345–6360.

Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stanczak, Peter Shaw, Christopher J.´ Pal, and Siva Reddy. 2025. AgentRewardBench: Evaluating automatic evaluations of web agent trajectories. Preprint, arXiv:2504.08942.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. 2024. Generative reward models. arXiv preprint arXiv:2410.12832.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A Smith, Hannaneh Hajishirzi, and Nathan Lambert. 2025. RewardBench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. 2025. Judging the judges: A systematic study of position bias in LLMas-a-judge. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics, pages 292–314.

Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, et al. 2025. A survey of reasoning with foundation models: Concepts, methodologies, and outlook. ACM Computing Surveys.

Koki Wataoka, Tsubasa Takahashi, and Ryokan Ri. 2024. Self-preference bias in LLM-as-a-judge. In NeurIPS Safe Generative AI Workshop.

Bosi Wen, Yilin Niu, Cunxiang Wang, Xiaoying Ling, Ying Zhang, Pei Ke, Hongning Wang, and Minlie Huang. 2026. IF-RewardBench: Benchmarking judge models for instruction-following evaluation. arXiv preprint arXiv:2603.04738.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Michihiro Yasunaga, Luke Zettlemoyer, and Marjan Ghazvininejad. 2025. Multimodal RewardBench: Holistic evaluation of reward models for vision language models. Preprint, arXiv:2502.14191.

Haoxi Zhong, Yuzhong Wang, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun. 2020. Iteratively questioning and answering for interpretable legal judgment prediction. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 1250–1257.

Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen, Yang Zhang, Wei Zhou, Jinjie Gu, and Lei Zou. 2025. A comprehensive survey of reward models: Taxonomy, applications, challenges, and future. arXiv preprint arXiv:2504.12328.

Hongli Zhou, Hui Huang, Wei Liu, Chenglong Wang, Xingyuan Bu, Lvyuan Han, Fuhai Song, Muyun Yang, Wenhao Jiang, Hailong Cao, and Tiejun Zhao. 2026. RM-Distiller: Exploiting generative LLM for reward model distillation. In Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, IJCAI-26, pages 6118–6126. International Joint Conferences on Artificial Intelligence Organization. Main Track.

Hongli Zhou, Hui Huang, Rui Zhang, Kehai Chen, Bing Xu, Conghui Zhu, Tiejun Zhao, and Muyun Yang. 2027. Toward robust LLM-based judges: Taxonomic bias evaluation and debiasing optimization. Information Fusion, 139:104801.

## A Model Catalog

Table 10 lists all models involved in LEGALRE-WARDBENCH, including both candidate response generation and reward modeling evaluation. For each model, we report its reward-model type when applicable, whether it is evaluated with thinking enhancement (TK), and whether it is used for response generation or reward-model evaluation. This catalog shows that our benchmark covers models from diverse families, architectures, and scales across both construction and benchmarking.

## B Task Distribution

Table 11 summarizes the task composition of LEGALREWARDBENCH. For each level, we list the task name, sample count in the final benchmark, data source, and scoring type. The final benchmark contains 41 task types in 5 capability levels.

## C Annotation Details

As explained in Section 2.3, our rubric-based annotation pipeline consists of two stages: automatic scoring and human verification.

For automatic scoring, we use two scoring protocols. For tasks that can be easily verified, we use rule-based scoring. Specifically, for statute recitation task, we compute ROUGE-L for the final output and the reference answer based on characterlevel segmentation, and map the score to a 0–100 scale. For the two sentence-prediction tasks, we start from 100 points and deduct one point for each month of absolute difference between the predicted and reference sentence terms, with a lower bound of 0. All remaining tasks use modelbased scoring with Gemini-3.1-Pro-Preview as the LLM judge, and the automatic-scoring prompt (Prompt 1) asks an LLM judge to assign an integer score to each candidate response according to task-specific rubrics. This design separates tasks with deterministic reference-based metrics from open-ended legal reasoning or writing tasks that require rubric-based model judging.

For human verification, we hired 5 Chinese master’s students in law. The human annotation prompts in Prompt 3 and Prompt 4 then ask annotators to verify pairwise preferences, using the rubrics as the primary comparison criterion while still checking for clear legal-quality issues. For L5 Writing tasks, we additionally provide the reasoning produced by the automatic scorer as auxiliary evidence, because these queries often contain extensive background materials and are more difficult for annotators to compare directly. The whole process of human verification takes 560 human-hours.

## D Inference Details

As explained in Section 3.1, we evaluate three families of reward models under inference settings that match their model interfaces. General Gen-RMs are prompted as pairwise legal judges and are asked to output a verdict between two candidate responses, using the template shown in Prompt 5. Fine-tuned GenRMs are evaluated with their native generative judgment templates whenever available, so that inference remains aligned with their supervised or preference-tuned training format. Finetuned DisRMs are used as scalar scorers: each query–response pair is scored independently, and the response with the higher score is selected as the model’s pairwise preference.

For generative models, unless otherwise specified by the model provider, we use temperature 0.2, top-p 1.0, top-k disabled, a maximum generation length of 4096 new tokens, and a maximum context length of 32768 tokens. For Qwen3-series models, we follow the inference guideline with temperature 0.6, top-p 0.95, and top-k 20.

## E Self-Bias of Reward Models

A common concern in RM evaluation is self-bias: if a model both generates candidate responses and acts as the RM, it may prefer its own style, reasoning, or wording (Wataoka et al., 2024). We therefore examine cases where one candidate response was generated by the RM itself. We record how often the RM selects its own response (RM Prefer), how often the gold label prefers that response (Gold Prefer), and define Self-bias as their difference.

As shown in Figure 3, all listed self-bias values are negative, indicating that these RMs do not systematically over-prefer their own outputs relative to the gold labels. This suggests that self-bias is not a dominant source of noise in the current LEGALREWARDBENCH setting.

## F Benchmark Examples

To make the benchmark format concrete, we present three representative examples from LEGAL-REWARDBENCH: an L1 statute-recitation item from LawBench, an L3 judicial-exam item, and an L4 sentence-prediction item from LawBench. Each example follows the same structure: task instruction, query, rubrics, the chosen and rejected responses, and the reference answer. Note that due to space limitations, we omit the reasoning process of each candidate response.

<table><tr><td>Model</td><td>Type</td><td>TK</td><td>Size</td><td>Response Gen.</td><td>RM</td></tr><tr><td>Open-weight General LLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-397B-A17B</td><td>General LLM</td><td></td><td>397B (17B)</td><td></td><td></td></tr><tr><td>Qwen3.5-122B-A10B</td><td>General LLM</td><td></td><td>122B (10B)</td><td></td><td>X</td></tr><tr><td>Qwen3.5-27B</td><td>General LLM</td><td></td><td>27B</td><td></td><td></td></tr><tr><td>Qwen3-235B-A22B</td><td>General LLM</td><td></td><td>235B (22B)</td><td>√</td><td></td></tr><tr><td>Qwen3-32B</td><td>General LLM</td><td></td><td>32B</td><td></td><td>X</td></tr><tr><td>Qwen3-30B-A3B</td><td>General LLM</td><td></td><td>30B (3B)</td><td>x</td><td></td></tr><tr><td>Qwen3-14B</td><td>General LLM</td><td></td><td>14B</td><td>√</td><td></td></tr><tr><td>Qwen3-8B</td><td>General LLM</td><td></td><td>8B</td><td></td><td></td></tr><tr><td>Qwen3-4B</td><td>General LLM</td><td></td><td>4B</td><td>√</td><td></td></tr><tr><td>Qwen3-1.7B</td><td>General LLM</td><td></td><td>1.7B</td><td>x</td><td></td></tr><tr><td>Qwen3-0.6B</td><td>General LLM</td><td></td><td>0.6B</td><td>X</td><td></td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>General LLM</td><td>X</td><td>7B</td><td></td><td></td></tr><tr><td>DeepSeek-v3.2</td><td>General LLM</td><td></td><td>671B (37B)</td><td></td><td></td></tr><tr><td>DeepSeek-R1</td><td>General LLM</td><td></td><td>671B (37B)</td><td></td><td></td></tr><tr><td>Kimi-K2.5</td><td>General LLM</td><td></td><td>1T (32B)</td><td></td><td></td></tr><tr><td>Kimi-K2</td><td>General LLM</td><td></td><td>1T (32B)</td><td></td><td></td></tr><tr><td>GLM-5</td><td>General LLM</td><td></td><td>744B (40B)</td><td></td><td>X</td></tr><tr><td>GLM-4.5</td><td>General LLM</td><td></td><td>355B (32B)</td><td></td><td></td></tr><tr><td>MiniCPM4-8B</td><td>General LLM</td><td>x</td><td>8B</td><td></td><td></td></tr><tr><td>MiniCPM3-4B</td><td>General LLM</td><td>x</td><td>4B</td><td></td><td>X</td></tr><tr><td>Minimax-M2.7</td><td>General LLM</td><td>√</td><td>230B (10B)</td><td></td><td>X</td></tr><tr><td>Gemma-3-12B-IT</td><td>General LLM</td><td>x</td><td>12B</td><td>√</td><td></td></tr><tr><td>LegalOne-8B</td><td>General LLM</td><td>x</td><td>8B</td><td>x</td><td></td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>General LLM</td><td>x</td><td>8B</td><td>V</td><td></td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>General LLM</td><td>x</td><td>3B</td><td>1</td><td></td></tr><tr><td>Closed-source General LLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude-Opus-4.7</td><td>General LLM</td><td></td><td></td><td>x</td><td></td></tr><tr><td>Claude-Opus-4.5</td><td>General LLM</td><td></td><td></td><td></td><td></td></tr><tr><td>Claude-Sonnet-4</td><td>General LLM</td><td>x</td><td></td><td></td><td></td></tr><tr><td>Claude-3-5-Sonnet</td><td>General LLM</td><td>X</td><td></td><td></td><td></td></tr><tr><td>Gemini-3-Flash-Preview</td><td>General LLM</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Gemini-2.5-Pro</td><td>General LLM</td><td>x</td><td></td><td></td><td></td></tr><tr><td>Gemini-2.5-Flash</td><td>General LLM</td><td>√</td><td></td><td></td><td></td></tr><tr><td>GPT-5.5</td><td>General LLM</td><td>V</td><td></td><td></td><td></td></tr><tr><td>GPT-5.2</td><td>General LLM</td><td>√</td><td></td><td></td><td></td></tr><tr><td>GPT-4.1</td><td>General LLM</td><td>x</td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>General LLM</td><td>x</td><td></td><td>L</td><td></td></tr><tr><td>Grok-4</td><td>General LLM</td><td>x</td><td></td><td></td><td>X</td></tr><tr><td>Fine-tuned generative reward models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-Nemotron-8B-BRRM</td><td>Ft. GenRM</td><td></td><td>8B</td><td>X</td><td></td></tr><tr><td>RM-R1-Qwen2.5-Instruct-7B</td><td>Ft. GenRM</td><td></td><td>7B</td><td>x</td><td></td></tr><tr><td>RRM-7B</td><td>Ft. GenRM</td><td></td><td>7B</td><td>X</td><td></td></tr><tr><td>Fine-tuned discriminative reward models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Skywork-Reward-V2-Qwen3-8B</td><td>Ft. DisRM</td><td></td><td>8B</td><td>X</td><td></td></tr><tr><td>Skywork-Reward-V2-Llama-3.1-8B</td><td>Ft. DisRM</td><td></td><td>8B</td><td>x</td><td></td></tr><tr><td>Llama-3.1-8B-Instruct-RM-RB2</td><td>Ft. DisRM</td><td></td><td>8B</td><td>x</td><td></td></tr><tr><td>Skywork-Reward-Llama-3.1-8B-v0.2</td><td>Ft. DisRM</td><td></td><td>8B</td><td>x</td><td></td></tr><tr><td>Skywork-Reward-V2-Llama-3.2-3B</td><td>Ft. DisRM</td><td></td><td>3B</td><td>x</td><td></td></tr><tr><td>InternLM2-1.8B-Reward</td><td>Ft. DisRM</td><td></td><td>1.8B</td><td>x</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 10: Model catalog for LEGALREWARDBENCH. TK denotes whether the model is evaluated in thinking mode; – indicates that the distinction is not applicable to the model family.

<table><tr><td>Level</td><td>Task</td><td>Samples</td><td>Source</td><td>Scoring</td></tr><tr><td rowspan="8">L1 (Knowledge) L2 (Understanding)</td><td>Statute Recitation (法条背诵)</td><td>45</td><td>LawBench (1-1)</td><td>Rule-based</td></tr><tr><td>Legal Rule Recall (法条规则记忆)</td><td>37</td><td>In-house</td><td>Model-based</td></tr><tr><td>Legal Knowledge Error Detection (法律知识幻觉)</td><td>31</td><td>In-house</td><td>Model-based</td></tr><tr><td>Applicable Law Identification (适用法律推荐)</td><td>35</td><td>In-house</td><td>Model-based</td></tr><tr><td>Statutory Evolution Analysis (立法沿革分析)</td><td>57</td><td>In-house</td><td>Model-based</td></tr><tr><td>Judicial Procedure QA (司法程序问答)</td><td>49</td><td>In-house</td><td>Model-based</td></tr><tr><td>Legal Policy QA (法律政策解答)</td><td>10</td><td>In-house</td><td>Model-based</td></tr><tr><td>Statute/regulation QA (法律法规问答)</td><td>42</td><td>In-house</td><td>Model-based</td></tr><tr><td rowspan="14">L3 (Analysis)</td><td>Legal Document Format Checking</td><td></td><td></td><td></td></tr><tr><td>(法律文书格式检查) Legal Grammar Checking</td><td>13</td><td>In-house</td><td>Model-based</td></tr><tr><td>(法律专业语法检查) Case Event Timeline Extraction</td><td>25 25</td><td>In-house</td><td>Model-based</td></tr><tr><td>(案情大事记梳理) Sensitive Information Redaction (去除敏感信息)</td><td>27</td><td>In-house In-house</td><td>Model-based Model-based</td></tr><tr><td>Judgment Reasoning Summarization (判决书审理思路总结)</td><td>20</td><td>In-house</td><td>Model-based</td></tr><tr><td>Forensic Report Analysis (司法专业报告分析)</td><td>19</td><td>In-house</td><td>Model-based</td></tr><tr><td>Legal Document Summarization (法律文书摘要)</td><td>20</td><td>In-house</td><td>Model-based</td></tr><tr><td>Legal Literature Review (法律文献综述)</td><td>22</td><td>In-house</td><td>Model-based</td></tr><tr><td>Legal Translatión (法律翻译)</td><td>23</td><td>In-house</td><td></td></tr><tr><td>Legal Element Extraction (法律要素还原)</td><td>19</td><td></td><td>Model-based</td></tr><tr><td>Long-Form Legal Document Understanding</td><td></td><td>In-house</td><td>Model-based</td></tr><tr><td>(法律长文本分析总结) Judicial Exam QA</td><td>15</td><td>In-house Legal exams</td><td>Model-based</td></tr><tr><td>(法律知识问答：司法考试题) Master-of-Law Exam QA (法律知识问答：考研)</td><td>44</td><td>(Judicial Exam, 24-25) Legal exáms</td><td>Model-based</td></tr><tr><td>Legal Characterization (法律定性判断)</td><td>10</td><td>(Master-of-Law Exam, 24-25)</td><td>Model-based</td></tr><tr><td>Fact-Finding Analysis</td><td>41</td><td>In-house</td><td>Model-based</td></tr><tr><td>(事实认定分析) Disputed Issue Identification</td><td>37</td><td>In-house</td><td>Model-based</td></tr><tr><td>(双方抗辩点识别) Copyright Infringément Text Comparison</td><td>43</td><td>In-house</td><td>Model-based</td></tr><tr><td>(著作权侵权文本比对) Evidence Analysis</td><td>43</td><td>In-house</td><td>Model-based</td></tr><tr><td>(证据分析) Trial-Focus Recommendation</td><td>35</td><td>In-house</td><td>Model-based</td></tr><tr><td>(审理要点推荐) Legal Rationale Generation</td><td>8</td><td>In-house</td><td>Model-based</td></tr><tr><td>(释法说理生成) Comparative Law Analysis</td><td>15</td><td>In-house</td><td>Model-based</td></tr><tr><td>(比较法分析) Bid Qualification Extraction</td><td>19</td><td>In-house</td><td>Model-based</td></tr><tr><td>(招投标文件资质提取)</td><td>38</td><td>In-house</td><td>Model-based</td></tr><tr><td>Commonsense Inconsistency Analysis (违背常情常理分析)</td><td>44</td><td>In-house</td><td>Model-based</td></tr><tr><td rowspan="4">L4 (Decision)</td><td>Sentence Prediction Without Statutes (刑期预测：无法条内容)</td><td>69</td><td>LawBench (3-4)</td><td>Rule-based</td></tr><tr><td>Šentence Prediction With Statutes (刑期预测：给定法条内容)</td><td>68</td><td>LawBench (3-5)</td><td>Rule-based</td></tr><tr><td>Judgment Outcome Prediction (裁判结果预测) Public-Öpinion Risk Assessment</td><td>41</td><td>In-house</td><td>Model-based</td></tr><tr><td>(审判结果舆情预测) Èxpert Reasoning Outline Generation</td><td>40</td><td>In-house</td><td>Model-based</td></tr><tr><td rowspan="3"></td><td>(法律专家思路生成) Legal Task Planning</td><td>36</td><td>In-house</td><td>Model-based</td></tr><tr><td>(法律任务规划) Procuratorial Recommendation Drafting</td><td>42</td><td>In-house</td><td>Model-based</td></tr><tr><td>(检察建议书生成) Defense Pleading Drafting</td><td>11</td><td>In-house</td><td>Model-based</td></tr><tr><td rowspan="3">L5 (Writing)</td><td>(答辩状生成) Similar-Case Retrieval Report Drafting</td><td>43</td><td>In-house</td><td>Model-based</td></tr><tr><td>(类案检索报告生成) Indictment Drafting</td><td>28</td><td>In-house</td><td>Model-based</td></tr><tr><td>(起诉书生成)</td><td>42</td><td>In-house</td><td>Model-based</td></tr></table>

Table 11: Task descriptions in LEGALREWARDBENCH (L1–L3). For different tasks, we use either rule-based or model-based automatic scoring methods based on the nature of the task.

Table 12: Task descriptions in LEGALREWARDBENCH (L4–L5, continued).

![](images/cab37d02239babda52d6f1f5fa1a2b8de5f7cbc8b485e107aa5f3f31a05e4c7d.jpg)  
Figure 3: Self-bias analysis for judge models. All values are reported as percentages.

<table><tr><td>PROMPT 1: AUTOMATIC SCORING PROMPT</td></tr><tr><td>You are a professional judge for legal and related tasks. Please score the candidate model response according to the task requirement, rubrics, and the reference answer if provided. Use an integer scale from 0 to 100, where 100 is the full score. ### Task Requirement / Query {instruction} ### Rubrics {score_point}</td></tr><tr><td>{trunc_block}{ref_block}</td></tr><tr><td>### Candidate Model Response {shown} Please output your response in the following format: {&quot;reason&quot;: &quot;Detailed reasoning process in Chinese explaining the reasons for deduction&quot;, &quot;score&quot;:</td></tr></table>

![](images/dc43803a4e4b7b67d7eb4dada0923c9f600a1e5152bcd04438f96ce5bbb229dd.jpg)

PROMPT 2: TRAINING-DATA SCORING PROMPT   
You are a legal expert evaluator. Please score the model response according to the following information, using an integer   
score from 1 to 10:   
### Task Requirement {instruction}   
### Query {question}   
### Reference Answer {reference\_answer}{analysis\_block}   
### Model Response {model\_answer}   
Scoring criteria:   
- 1–3 points: The response is seriously incorrect or completely deviates from the task   
- 4–6 points: The response is partially correct but contains clear defects or omissions.   
- 7–8 points: The response is mostly correct, with only minor flaws.   
- 9–10 points: The response is fully correct, logically clear, and accurately expressed.   
Please output only one JSON object:   
{"score": <an integer from 1 to 10>, "reason": "<brief reason>"}

## PROMPT 3: ANNOTATION GUIDELINE FOR SHORT OUTPUTS

score\_point but one response has a clear quality problem, such as a serious professional error, you may annotate according to that quality issue.

3. Only when the final-output parts are close in quality should you compare the reasoning process before Final Output:. 4. If a response does not provide a reasoning process before the final response, it fails to follow the query instruction and should be penalized accordingly.

6. If you think some rubrics in score\_point are problematic, please explain this in the notes.

## PROMPT 4: ANNOTATION GUIDELINE FOR LONG OUTPUTS

PROMPT 5: GENERATIVE JUDGE INFERENCE PROMPT

Please act as an impartial legal expert judge and evaluate the quality of the responses provided by two AI assistants to the user’s legal inquiry displayed below.

After providing your explanation, output your final verdict by strictly following this format: [[A]] if assistant A is better, [[B]] if assistant B is better.

Sample F.1: L1 Knowledge – Statute Recitation

回答以下问题，请先写出简要推理，换行后在新的一行以「最终输出：」开头直接给出法条内容。请你回答：

English: Answer the following question. First provide a brief reasoning, and then, on a new line beginning with “Final Output:”, directly provide the text of the relevant statutory provision.

社会法家庭教育促进法第二十二条的内容是什么？

English: What does Article 22 of the Family Education Promotion Law provide?

## Rubrics

最终输出应准确复述《中华人民共和国家庭教育促进法》第二十二条：未成年人的父母或者其他监护人应当合理安排未成年人学习、休息、娱乐和体育锻炼的时间，避免加重未成年人学习负担，预防未成年人沉迷网络。

English: The final output should accurately reproduce Article 22 of the Family Education Promotion Law of the People’s Republic of China: Parents or other guardians of minors shall reasonably arrange the minors’ time for study, rest, recreation, and physical exercise, avoid increasing their study burden, and prevent minors from becoming addicted to the Internet.

## Chosen Response

最终输出：未成年人的父母或者其他监护人应当合理安排未成年人学习、休息、娱乐和体育锻炼的时间，避免加重未成年人学习负担，预防未成年人沉迷网络。

English: Final Output: Parents or other guardians of minors shall reasonably arrange the minors’ time for study, rest, recreation, and physical exercise, avoid increasing their study burden, and prevent minors from becoming addicted to the Internet.

## Rejected Response

最终输出：父母或者其他监护人应当树立家庭是第一个课堂、家长是第一任老师的责任意识，承担对未成年子女实施家庭教育的主体责任。父母或者其他监护人应当亲自养育未成年子女，加强亲子陪伴。父母或者其他监护人应当共同参与家庭教育，关注未成年子女的生理、心理、智力发展状况，尊重其参与相关家庭事务和发表意见的权利。父母或者其他监护人不得实施家庭暴力，不得因性别、身体状况、智力状况等歧视未成年子女，不得胁迫、引诱、教唆、纵容、包庇未成年人实施违法犯罪行为，不得放任、教唆或者帮助未成年人自杀、自残、欺凌、暴力等行为。

English: Final Output: Parents or other guardians shall recognize that the family is the first classroom and that parents are the first teachers, and shall assume primary responsibility for providing family education to their minor children. Parents or other guardians shall personally raise their minor children and strengthen parent-child companionship. Parents or other guardians shall jointly participate in family education, pay attention to the physical, psychological, and intellectual development of their minor children, and respect their right to participate in relevant family matters and express their opinions. Parents or other guardians shall not commit domestic violence or discriminate against their minor children on the basis of sex, physical condition, intellectual condition, or other factors; coerce, entice, instigate, tolerate, or shelter minors in committing unlawful or criminal acts; or allow, instigate, or assist minors in engaging in suicide, self-harm, bullying, violence, or other such acts.

## Reference Answer

未成年人的父母或者其他监护人应当合理安排未成年人学习、休息、娱乐和体育锻炼的时间，避免加重未成年人学习负担，预防未成年人沉迷网络。

English: Parents or other guardians of minors shall reasonably arrange the minors’ time for study, rest, recreation, and physical exercise, avoid increasing their study burden, and prevent minors from becoming addicted to the Internet.

2. If the company fails to pay the fine on time, the district emergency management bureau shall impose an additional fine at a rate of three percent of the fine amount per day, notify the company of this standard, and ensure that the additional fine does not exceed the original fine amount.

Sample F.2: L3 Analysis – Judicial Exam QA

## Task Instruction

题干与具体问题见用户消息。请先进行分析或推理，然后在新的一行「最终输出：」开头给出你的作答。

English: The case facts and specific question are provided below. First analyze or reason through the problem, and then provide your answer on a new line beginning with “Final Output:”.

## Query

【案情】某应急管理局因为某公司的焊接工人没有取得焊接证，责令其临时停业，取得证件后再进行生产。后逾期没有取得许可证，管理局拟作出行政处罚决定，责令停业，罚款<sub>7</sub>万元。告知其<sub>5</sub>日内可以进行申诉等等。但是该公司<sub>5</sub>日内没有申诉，管理局就做出行政处罚。后该公司觉得处罚不合理，申请复议。复议机关作出维持决定。该公司如果逾期不缴纳罚款，丙区应急管理局应该如何处理？

English: Case facts: An emergency management bureau ordered a company to temporarily suspend operations because its welding workers had not obtained the required welding certificates, and allowed it to resume production only after the certificates were obtained. The company subsequently failed to obtain the required permit within the prescribed period, and the bureau proposed an administrative penalty requiring it to suspend operations and pay a fine of 70,000 yuan. The company was notified that it could file an appeal within five days, but did not do so within that period, and the bureau subsequently issued the administrative penalty. The company later considered the penalty unreasonable and applied for administrative reconsideration. The reconsideration authority upheld the decision.

If the company fails to pay the fine within the prescribed period, how should the emergency management bureau of District C handle the matter?

## Rubrics

<sub>1.</sub> 说明应当书面催告，并载明金额、给付方式和期限；

<sub>2.</sub> 说明逾期不缴纳罚款的，应按罚款数额每日百分之三加处罚款，并告知该标准，且加处罚款的数额不得超过罚款数额；

<sub>3.</sub> 说明若在法定期限内不申请行政复议或者提起行政诉讼，又不履行行政决定的，因该局无权直接执行罚款，应自期限届满之日起三个月内向本机关所在地法院申请强制执行。

English: 1. State that the bureau should issue a written demand specifying the amount, the method of payment, and the time limit.

2. State that if the fine is not paid on time, an additional fine of three percent of the original fine amount per day shall be imposed, that this standard should be communicated to the company, and that the additional fine may not exceed the original fine amount.

3. State that if the company neither applies for administrative reconsideration nor brings an administrative lawsuit within the statutory period, and also fails to comply with the administrative decision, the bureau, which has no authority to directly enforce the fine, should apply to the court at the bureau’s location for compulsory enforcement within three months after the statutory period expires.

## Chosen Response

最终输出：丙区应急管理局可以每日按罚款数额的<sub>3%</sub>加处罚款（加处罚款额不得超过<sub>7</sub>万元罚款本数）；在申请强制执行前应当依法书面催告；若该公司在法定期限内不提起诉讼又不履行，应急管理局应当申请人民法院强制执行。

English: Final Output: The emergency management bureau of District C may impose an additional fine at a rate of 3% of the original fine amount per day, provided that the additional fine does not exceed the original fine of 70,000 yuan. Before applying for compulsory enforcement, it should issue a written demand in accordance with the law. If the company neither brings a lawsuit within the statutory period nor complies with the decision, the bureau should apply to a people’s court for compulsory enforcement.

## Rejected Response

最终输出：丙区应急管理局应当按下列步骤处理：

<sub>1.</sub> 催告缴纳：向该公司送达缴纳罚款的催告或通知，明确补缴期限、缴纳方式及逾期 后果。

<sub>2.</sub> 加处罚款：公司逾期仍不缴纳的，可以依法对逾期未缴部分按日加处罚款，通常为按日加收不超过罚款数额一定比例的加处罚款。

<sub>3.</sub> 申请法院强制执行：经催告后仍不履行的，应当向有管辖权的人民法院申请强制执行，而不是自行强制扣划或查封，除非有明确法律授权。

English: Final Output: The emergency management bureau of District C should proceed as follo

1. Demand payment: Serve the company with a written demand or notice to pay the fine, specifying the time limit for payment, the method of payment, and the consequences of late payment.

2. Impose an additional fine: If the company still fails to pay on time, an additional daily fine may be imposed on the unpaid amount in accordance with the law, generally at a rate not exceeding a specified proportion of the original fine.

3. Apply for compulsory enforcement: If the company still fails to comply after receiving the demand, the bureau should apply to the competent people’s court for compulsory enforcement rather than directly freezing or seizing the company’s assets, unless expressly authorized to do so by law.

## Reference Answer

若该公司在法定期限内不申请行政复议或者提起行政诉讼，又不履行行政决定的，因为该区应急管理局无权直接执行罚款，其可以自期限届满之日起三个月内，向本机关所在地法院申请强制执行。

<sub>1.</sub> 书面催告，载明金额、给付方式和期限。

<sub>2.</sub> 该公司如果逾期不缴纳罚款的，该区应急管理局应按罚款数额的每日百分之三加处罚款，告知标准且加处罚款的数额不得超过罚款的数额。

Sample F.3: L4 Decision – Sentence Prediction Without Statutes

## Task Instruction

根据下列事实、罪名和刑法法条预测判决刑期。请先进行分析或推理，然后给出判决刑期为多少月，请将答案填在<sub>[</sub>刑期<sub>]</sub>与<sub><eoa></sub>之间。例如<sub>[</sub>刑期<sub>]12</sub>月<sub><eoa></sub>。请你回答：

English: Based on the following facts, charge, and provision of the Criminal Law, predict the length of the sentence in months. First analyze or reason through the case, and then place the predicted <sub>sentence</sub> <sub>term</sub> <sub>between</sub> <sub>[</sub>刑期<sub>]</sub> <sub>and</sub> <sub><eoa>.</sub> <sub>For</sub> <sub>example,</sub> <sub>[</sub>刑期<sub>]12</sub>月<sub><eoa>.</sub>

## Query

事实：新沂市人民检察院指控： 年 月至 年间，被告人潘某某在新沂市＊＊镇＊＊村＊组＊＊号其家中开设诊所，因未取得《医疗机构执业许可证》及《执业医师资格证》行医，分别于<sub>2010</sub>年<sub>6</sub>月<sub>1</sub>日、<sub>2011</sub>年5月<sub>27</sub>日、<sub>2012</sub>年5月<sub>20</sub>日被新沂市卫生局行政处罚。后被告人潘某某继续在其诊所内无证行医，并于<sub>201</sub>5年<sub>10</sub>月<sub>30</sub>日正在行医时被新沂市公安局马陵山派出所查获。被告人潘某某于<sub>2015</sub>年<sub>10</sub>月<sub>30</sub>日被新沂市公安局马陵山派出所民警依法传唤到案，其如实供述上述犯罪事实。罪名：非法行医。法条：刑法<sup>第</sup>336<sup>条。</sup>

English: Facts: The Xinyi Municipal People’s Procuratorate charged that, from April 2010 to 2015, defendant Pan operated a clinic at his home in Xinyi without obtaining a Medical Institution Practice License or a Physician Qualification Certificate. The Xinyi Municipal Health Bureau imposed administrative penalties on him on June 1, 2010, May 27, 2011, and May 20, 2012. Pan subsequently continued to practice medicine without a license at his clinic and was apprehended while practicing medicine on October 30, 2015, by officers from the Malingshan Police Station of the Xinyi Public Security Bureau. On the same day, Pan was lawfully summoned to appear before officers from the Malingshan Police Station and truthfully confessed to the above criminal conduct. Charge: illegal practice of medicine. Provision: Article 336 of the Criminal Law.

## Rubrics

参考刑期为16个月。从100分起，预测刑期与参考刑期每相差1个月扣1分，分数下限为0。

English: The reference sentence is 16 months. Starting from a score of 100, one point is deducted for each month of absolute difference between the predicted sentence and the reference sentence, with a minimum score of 0.

## Chosen Response

## Rejected Response

<sub>[</sub>刑期<sub>]18</sub>月<sub><eoa></sub>

<sub>[</sub>刑期<sub>]30</sub>月<sub><eoa></sub>

<sub>Eng</sub>l<sub>is</sub>h<sub>:</sub> <sub>[</sub>刑期<sub>]18</sub>月<sub><eoa></sub>

<sub>Eng</sub>l<sub>is</sub>h<sub>: [</sub>刑期<sub>]30</sub>月<sub><eoa></sub>

## Reference Answer

刑期：16个月。

English: Sentence term: 16 months.