# ToolLoop: Closed-Loop Tool-Use Data Synthesis via Decomposed Generation and Dynamic Self-Feedback

Min Zeng, Yuzhou Liu, Zhenyu Cao, Hanxiu Chen, Heng Li, Caiquan Liu,

Yafei Wen, Xiaoxin Chen

vivo AI Lab

zengmin325@163.com

## Abstract

High-quality tool-use data is critical for training language models to interact effectively with external tools. However, existing synthetic approaches typically follow a generate-then-filter paradigm with static post-hoc verification, often yielding inefficient data with imbalanced feature distributions. We propose ToolLoop, a closed-loop framework that decomposes synthesis into three progressive stages: (1) sampling function name combinations as ground truth; (2) backward derivation of user queries; and (3) forward derivation of tool calls. At each stage, dynamic self-feedback iteratively guides the model toward high-quality generation, realizing a transition from generate-thenfilter to generate-verify-refine. On the Berkeley Function Calling Leaderboard (BFCL), a 4B parameter model trained with our 11K synthetic examples achieves 86.40% accuracy in nonreasoning mode, while an Isolate variant that removes BFCL-overlapping candidate functions still reaches 86.07%. Cross-benchmark evaluation on ACEBench further demonstrates strong generalization, with 72.1% overall accuracy using only 18.3% of baseline training data.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation (Agashe et al., 2024; Antoniades et al., 2024; Hu et al., 2025; Bahdanau et al., 2024). However, they remain fundamentally constrained by their static knowledge and limited ability to interact with the external world (Matarazzo and Torlone, 2025; Li et al., 2025). To address these limitations, recent research has adopted agent-based paradigms in which autonomous LLM-powered agents not only generate text but also interpret user intent and invoke external tools (e.g., APIs for weather queries, web search, or other services) to fulfill user requests, effectively bridging language understanding with real-world actions (Xu et al., 2025a; Wang et al., 2025a; Ferrag et al., 2025). Despite this progress, training models to effectively leverage such tools requires high-quality tool-use datasets that accurately capture the relationship between user instruction and tool selection — a resource that remains extremely scarce in current datasets (Yang et al., 2025; Ye et al., 2025).

Existing synthetic methods for tool-use data primarily follow a “generate-then-filter” paradigm, where complete samples are generated in a single step and subsequently filtered using rule-based or model-based verification (Xu et al., 2025b; Liu et al., 2024a,b). While straightforward, this approach suffers from three fundamental limitations. First, due to the diversity and complexity of tools, selecting the correct and most efficient tool from a large candidate set in a generation is highly challenging (Kim et al., 2025). Second, static post-hoc filtering operates as a binary accept/reject mechanism that discards invalid instances without providing corrective feedback, thereby leading to biased feature distributions in the resulting dataset. Third, the absence of intermediate supervision during generation implies that the model receives no guidance on logical consistency between stages of generation (e.g., whether the generated tool call actually resolves the user query), leaving certain error modes unmitigated (Wang et al., 2025b).

To address these challenges, as illustrated in Figure 1, we propose ToolLoop, a closed-loop data synthesis framework that rethinks the generation process through decomposed generation and dynamic self-feedback. ToolLoop decomposes synthesis into three progressive stages with explicit intermediate representations. At each stage, we integrate a dynamic self-verification mechanism that detects problematic outputs and feeds identified issues back to the language model as explicit guidance for regeneration. This closed-loop feedback enables the model to learn from its mistakes during synthesis and improves both efficiency and quality of retained samples. We conduct comprehensive evaluations on BFCL (Patil et al., 2025) and ACEBench (Chen et al., 2025), demonstrating strong improvements over existing approaches on both benchmarks.

![](images/3cbcbd814ec0e94e2296dab4ea187b55d9b6a317b9b9c6393bb6160048d28929.jpg)  
Figure 1: Comparison between existing generate-then-filter paradigm and our ToolLoop closed-loop synthesis framework with generate-verify-refine approach.

Our contributions can be summarized as follows:

• We introduce ToolLoop, a novel closed-loop framework that decomposes synthesis into three stages and integrates dynamic selffeedback to guide iterative optimization.

• We demonstrate through comprehensive experiments on BFCL and ACEBench that Tool-Loop achieves strong downstream task performance with substantially less training data.

• We demonstrate the practical utility of Tool-Loop by constructing an 11K-example synthetic tool-use dataset covering four representative function-calling scenarios.

## 2 Related Work

Synthetic Data for LLMs. Recent advances in tool-augmented language models have focused on improving both model capabilities and training data quality. Early synthetic-data methods primarily focused on general instruction tuning. Self-Instruct (Wang et al., 2023) enabled language models to generate instruction–input–output examples, followed by filtering and deduplication, while Stanford Alpaca (Taori et al., 2023) built on this paradigm to improve synthesis diversity and efficiency. Subsequent work extended synthetic data generation to more diverse tasks and data types. Persona Hub (Ge et al., 2024) introduced a persona-driven methodology for synthesizing reasoning data; APIGen (Liu et al., 2024b) automated the generation of verifiable function-calling data; SynthAgent (Wang et al., 2025c) improved synthetic data quality through the joint refinement of tasks and trajectories derived from web elements; and APIGen-MT (Prabhakar et al., 2025) proposed a two-phase framework for generating verifiable and diverse multi-turn agent data.

LLMs as Automatic Evaluators / Judges. LLMs are increasingly used as automatic evaluators (LLM-as-a-Judge) of model outputs. Laskar et al. (Laskar et al., 2025) explore using LLMs as judges to evaluate biomedical relation extraction systems, demonstrating that structured output formatting can improve LLM-judge performance relative to traditional metrics. Similarly, Bavaresco et al. (Bavaresco et al., 2025) provide a comprehensive empirical study across 20 NLP tasks, showing that LLMs can replicate some aspects of human annotation but also reveal substantial variability depending on task and model choice. D’Souza et al. (D’Souza et al., 2025) introduce YESciEval, a robust LLM-as-a-Judge framework for scientific question answering with rubric-based assessment, supporting scalable evaluation without human feedback. Beyond systematic benchmarks, Yamauchi et al. (Yamauchi et al., 2025) empirically analyze how design choices in LLM-based evaluation affect alignment with human judgments, reporting important factors such as decoding and criteria formulation. Finally, Liu et al. (Liu et al., 2025) propose Judge-Consistency for improving consistency of LLM judgments on retrieval-augmented generation.

![](images/82d5a6ca04364cc2c352b7537c798d7d250fcbecfca40916f8aa47e52f187080.jpg)  
Figure 2: ToolLoop Framework: Three-Stage Decomposed Generation with Dynamic Self-Feedback

These works suggest that LLM-based assessment can serve as an effective, scalable alternative to traditional evaluation protocols in many scenarios.

## 3 Methods

## 3.1 Scope of Category

To ensure diversity and comprehensiveness in our synthesized data, we define a set of tool-call<sup>1</sup> scenarios that span different levels of complexity and patterns of use. These include: Simple, where a single provided function is invoked to answer the query; Multiple, where several functions are available but only one is needed; Parallel, where a single function must be called multiple times within one response and each call is independent; and Parallel Multiple, where the model selects one or more relevant functions from multiple candidates and may invoke them multiple times in parallel. Together, these scenarios cover the majority of singleturn tool-call cases, from straightforward singlefunction calls to complex multi-function parallel invocations.

## 3.2 Candidate Function Construction

Synthetic data requires constructing candidate functions for different tasks under each scenario. While random sampling and direct LLM-based filtering are potential approaches, they either fail to capture semantic relationships between tools or are limited by context window constraints when dealing with large function libraries. Based on this, we employ a method that combines semantic clustering with LLM-guided sampling. As shown in

Figure 2, we first use a vector model to convert the textual description of each function into dense semantic embeddings. We then apply K-means clustering to partition the embedding space into K clusters, where each cluster contains semantically related functions that typically share the same domain or usage scenario. Finally, we sample functions belonging to the same usage scenario from each cluster via LLM as candidate functions for each Function invocation.

## 3.3 Decomposed Data Synthesis with Dynamic Self-Feedback

Given the substantial difficulty of directly synthesizing valid tool-use data in a single pass, we decompose the synthesis pipeline into three progressive stages: ground truth generation, user query derivation, and tool call instantiation. Each stage incorporates dynamic self-feedback to ensure quality before proceeding. Detailed prompt templates for all three stages with integrated self-feedback mechanisms are provided in Appendix B. We describe each stage in detail below.

## 3.3.1 Ground Truth Sampling

A ground truth represents the sequence of function names that will be invoked to address a user’s request. For parallel scenarios, we prompt an LLM to select function names from the candidate set that can potentially produce parallel invocations within a coherent usage context. For non-parallel scenarios, we employ random sampling to select function as the target invocation sequence.

We enforce the following properties during ground truth generation:

• Parallelizability: Functions in the ground truth can be invoked concurrently without dependencies—no function requires the execution results of another.

• Coherence: The combination of functions should enable a reasonable user request within a unified scenario context.

• Cardinality: For parallel and parallelmultiple scenarios, the ground truth contains 2–4 function invocations, where the same function name may appear multiple times to support parallel calls. For simple and multiple scenarios, the ground truth contains a single target invocation.

## 3.3.2 Backward Derivation of User Query

Given a generated ground truth and the associated candidate functions, we perform backward reasoning to derive a natural user query. This reverse generation ensures tight alignment between user intent and the ground truth. We guide query generation with the following desiderata:

• Precision: The implicit tool calls requirements in the query must correspond exactly to the ground truth—neither more nor fewer function calls should be implied.

• Consistency: Parameters mentioned in the query must conform to the function signatures and maintain format consistency across the query.

• Completeness: The query must provide all necessary information to satisfy the parameter requirements of the functions in the ground truth.

• Naturalness: The query should emulate realistic, conversational user expressions rather than mechanistic command-like instructions.

## 3.3.3 Forward Derivation of Tool Calls

In the final stage, we generate the concrete tool calls with fully specified parameters. Taking the user query and candidate functions as input, we prompt an LLM to produce the complete tool calls sequence in a structured format. We enforce the following constraints:

• Schema Compliance: Generated tool calls must strictly adhere to the function signatures, with correct parameter names, types, and structures.

• Functional Accuracy: Tool calls must accurately fulfill the user’s request as expressed in the query, with appropriate parameter values.

• Format Specification: We adopt the OpenAI function-calling format for standardized tool call representation, ensuring compatibility with existing evaluation frameworks.

## 3.3.4 Dynamic Self-Feedback

Critically, we integrate a dynamic self-feedback mechanism at each of the three stages described above. This mechanism combines three validation strategies: an LLM verifier assesses semantic correctness and logical coherence; deterministic rules check format compliance, structural constraints, and data types; and AST parsing detects syntactic errors in generated tool calls that would cause execution failures at runtime.

The feedback is stage-specific rather than a generic quality score. In ground truth sampling, the verifier is designed to assess whether the selected function names form a coherent intent, especially for parallel cases where calls should be independent but semantically related. In backward query derivation, it checks whether the user request is aligned with the intended function sequence and provides sufficient information for required arguments. In forward derivation, the checks become more concrete: the generated tool calls should follow the target schema and instantiate arguments using values supported by the user query. This stage-local design helps prevent early mistakes from being hidden until final filtering, where they are harder to diagnose.

When validation fails at any stage, we construct a refinement prompt that includes: (1) the original generation prompt, (2) the failed output as a negative example, and (3) specific issues identified by the validators with actionable feedback. For instance, if AST parsing detects a syntax error in a tool call, the feedback explicitly states "Syntax error: unmatched closing bracket in arguments field" rather than simply marking the sample as invalid. This feedback-augmented prompt guides the model to correct its mistakes in subsequent generation attempts.

This design also preserves useful diversity. A pure filtering pipeline tends to discard difficult samples, which can bias the retained data toward short queries, simple schemas, or overly explicit user requests. After a valid function combination has been selected, later-stage refinements attempt to repair the query or tool calls around that target rather than immediately discarding the sample. Only samples that remain invalid after the maximum number of refinement attempts are discarded. In this way, the feedback loop acts as a targeted correction mechanism rather than a coarse accept/reject gate.

The refinement process continues iteratively until either: (a) all validation checks pass, allowing progression to the next stage, or (b) a maximum of three retries is reached, at which point the sample is discarded. Thus, a sample may receive one initial generation and up to three feedbackguided regeneration attempts. This design balances quality improvement with computational efficiency—empirically, we find that most correctable errors are resolved within one or two retries, while samples requiring more retries often have fundamental semantic issues that are unlikely to be resolved through additional refinement.

## 4 Experiments

## 4.1 Experimental Setup

Models. To evaluate the quality of the synthetic data generated by our approach, we adopt Qwen3-4B-Instruct-2507 <sup>2</sup> (Qwen, 2025) as the base model. This non-deliberative variant exhibits strong general-purpose capabilities and serves as a stable foundation for our experiments. We benchmark our method against a wide range of stateof-the-art models, including open-source variants such as Qwen3 series, LLaMA 4 (Meta, 2025), Gemma3 (Team et al., 2025), and GLM 4.6 (Zhipu, 2025), as well as commercial models such as GPT-5.2, Gemini-3-Pro, Claude-Opus-v4.5, Grok 4.1, and Amazon-Nova-2.

API Source. Following the methodologies outlined in APIGen (Liu et al., 2024b) and Magnet (Yin et al., 2025), we randomly sample a subset of functions from ToolBench (Qin et al., 2023) and BFCL to synthesize data. We collected a total of 5,281 executable APIs. For the leakage-control setting, ToolLoop-4B-Isolate filters candidate functions that overlap with BFCL evaluation candidate functions before synthesis, producing a 10K training set. These APIs were partitioned into 26 distinct semantic groups using K-means clustering applied to API descriptions encoded by the Qwen-

3-Embedding-8B <sup>3</sup> (Zhang et al., 2025). We set the number of clusters to K = 26 to ensure that each cluster contains approximately 200 data points; this size strikes a balance between effectively covering a rich diversity of data and staying within the text length constraints required for model inference. Figure 3 demonstrates the distribution of the candidate API pool across diverse domains.

Benchmark. We evaluate the trained models on two complementary benchmarks:

BFCL-v4. The Berkeley Function Calling Leaderboard (Patil et al., 2025) contains 2,501 test instances across four scenarios: Simple (single function), Multiple (selecting from candidates), and Parallel (repeated invocations), Parallel Multiple (combined).

ACEBench. ACEBench (Chen et al., 2025) is a comprehensive benchmark evaluating tool-use across five dimensions: Atom (atomic operations), Single Turn (complete task execution), Similar API (distinguishing similar tools), Profile (personalized selection), and Overall (aggregate performance). This provides a holistic view of model capabilities beyond function-calling accuracy.

Hyperparameters. All experiments are conducted on a multi-node cluster using the swift framework (Zhao et al., 2025), where each node is equipped with four NVIDIA L40s GPUs (48 GB VRAM each). The maximum sequence length for training data is set to 16k tokens. To mitigate overfitting, we train the models on synthetic datasets for two epochs.

For our ToolLoop variants, training uses the same base model, formatting convention, and non-reasoning inference setting, so the comparison between ToolLoop-4B and ToolLoop-4B-Isolate mainly reflects the effect of filtering BFCL-overlapping candidate functions rather than changes in decoding or model capacity. During evaluation, the model is required to output tool calls directly in the benchmark-compatible functioncalling format, without additional chain-of-thought traces. This setting is intentionally strict: it measures whether the synthesized data teaches the model to map user intent to valid tool invocations, rather than whether the model can recover through verbose reasoning at inference time.

![](images/11c48d216b003dd7816b207723d80e584ef501262e0aec14b298c0f8f10b957a.jpg)  
Figure 3: Distribution of the executable APIs.

## 4.2 Main Results

Considering that agentic systems require high time efficiency during operation, we train and evaluate our model in non-reasoning mode without additional chain-of-thought overhead. Table 1 compares ToolLoop with commercial, open-source, and data-centric baselines on BFCL. ToolLoop-4B, trained on only 11K synthetic examples (Appendix A), achieves the best overall accuracy of 86.40%, improving over APIGen-4B (83.11%, 60K examples) by 3.29 points and ToolMind-4B (83.53%, 55K examples) by 2.87 points. The leakage-control variant, ToolLoop-4B-Isolate, removes BFCL-overlapping candidate functions and still reaches 86.07%, only 0.33 points below the full model, suggesting that the gains are not driven by candidate-function overlap.

In non-live scenarios, ToolLoop achieves 91.29%, surpassing the strongest non-ToolLoop baseline, APIGen-4B (89.90%), by 1.39 points. The advantage is visible in complex selection and parallelization settings: ToolLoop obtains 96.50% on Multiple and 94.50% on Parallel\_Multiple, while the Isolate variant even reaches 97.00% on Multiple. These results support the value of decomposing synthesis around explicit function combinations rather than relying on one-shot sample generation.

In live scenarios, ToolLoop remains competitive with the strongest commercial and open-source systems, reaching 81.50% overall and clearly exceeding the data-centric baselines APIGen-4B (76.31%) and ToolMind-4B (77.57%). In Live-Parallel\_Multiple, ToolLoop scores 79.17% versus APIGen-4B’s 83.33%, a gap that amounts to a single sample difference on this 24-instance split. Given the small split size, this result has high sampling uncertainty and does not support a strong category-level conclusion.

The small gap between ToolLoop-4B and ToolLoop-4B-Isolate further indicates that Tool-Loop does not simply benefit from memorizing benchmark-specific function schemas. After removing BFCL-overlapping candidate functions, the model retains nearly the same overall performance and even improves on the non-live Multiple category. This suggests that the main benefit comes from the synthesis procedure itself: clustering related tools, constructing explicit target function combinations, and using feedback-guided regeneration to align user queries with executable tool calls. In other words, ToolLoop improves the model’s general tool-selection and argument-grounding behavior rather than only adapting it to a particular function pool.

## 4.3 Ablation Study

To investigate the contribution of dynamic selffeedback, we compare three variants under identical training settings, as shown in Table 2. The w/ Final Filtering variant uses the same synthesis framework, verifier, and initial generation budget as ToolLoop; its key difference is that invalid outputs are discarded rather than revised using verifier feedback, making it a controlled generate-thenfilter baseline. Removing the feedback mechanism entirely (w/o Feedback) yields only 79.97% overall accuracy—a 2.17-point drop from the base model (82.14%), suggesting that naively generated synthetic data without quality control introduces noisy training signals that actively harm model performance. Replacing iterative refinement with static post-hoc filtering partially recovers performance (82.56%), yet still falls far short of the full system, as discarding invalid samples without corrective guidance leaves systematic generation errors unaddressed. In contrast, the full ToolLoop framework achieves 86.40% overall (91.29% non-live, 81.50% live), demonstrating that the key to high-quality synthetic data lies not in generation alone, but in the iterative generate-verify-refine loop enabled by dynamic self-feedback.

<table><tr><td rowspan="3">Models</td><td colspan="4">Non-Live</td><td colspan="4">Live</td><td colspan="3">Overall</td></tr><tr><td>Simple</td><td>Multiple</td><td>Parallel</td><td>Parallel Multiple</td><td>Simple</td><td>Multiple</td><td>Parallel</td><td>Parallel Multiple</td><td>Non-Live</td><td>Live</td><td>Overall</td></tr><tr><td>GPT-5.2-2025-12-11</td><td>72.92</td><td>88.00</td><td>89.00</td><td>77.50</td><td>71.71</td><td>70.37</td><td>68.75</td><td>58.33</td><td>81.85</td><td>70.39</td><td>76.12</td></tr><tr><td>Gemini-3-Pro-Preview</td><td>75.50</td><td>94.00</td><td>91.00</td><td>82.50</td><td>87.60</td><td>80.44</td><td>75.00</td><td>79.17</td><td>85.75</td><td>81.72</td><td>83.74</td></tr><tr><td>Grok-4-1-fast</td><td>77.58</td><td>93.00</td><td>92.50</td><td>90.00</td><td>84.11</td><td>77.30</td><td>75.00</td><td>70.83</td><td>88.27</td><td>78.46</td><td>83.37</td></tr><tr><td>Claude-Opus-4-5-20251101</td><td>76.83</td><td>95.50</td><td>93.50</td><td>88.50</td><td>86.43</td><td>78.16</td><td>87.50</td><td>75.00</td><td>88.58</td><td>79.79</td><td>84.19</td></tr><tr><td>Amazon-Nova-2-Lite-v1:0</td><td>76.33</td><td>94.00</td><td>91.50</td><td>86.00</td><td>83.33</td><td>80.15</td><td>87.50</td><td>79.17</td><td>86.96</td><td>80.83</td><td>83.90</td></tr><tr><td>Qwen3-32B</td><td>75.58</td><td>94.50</td><td>93.50</td><td>91.50</td><td>89.53</td><td>80.91</td><td>81.25</td><td>50.00</td><td>88.77</td><td>82.01</td><td>85.39</td></tr><tr><td>Qwen3-4B-Instruct-2507</td><td>75.50</td><td>93.50</td><td>92.50</td><td>90.00</td><td>79.07</td><td>76.16</td><td>62.50</td><td>66.67</td><td>87.88</td><td>76.39</td><td>82.14</td></tr><tr><td>Llama-4-Scout-17B-16E</td><td>79.00</td><td>94.00</td><td>94.00</td><td>90.50</td><td>81.78</td><td>72.74</td><td>81.25</td><td>79.17</td><td>89.38</td><td>74.69</td><td>82.04</td></tr><tr><td>Gemma-3-27b-it</td><td>77.67</td><td>92.50</td><td>89.00</td><td>89.50</td><td>84.50</td><td>72.46</td><td>93.75</td><td>45.83</td><td>87.17</td><td>74.54</td><td>80.86</td></tr><tr><td>GLM-4.6</td><td>74.25</td><td>95.00</td><td>91.50</td><td>89.50</td><td>89.53</td><td>78.92</td><td>81.25</td><td>75.00</td><td>87.56</td><td>80.90</td><td>84.23</td></tr><tr><td>APIGen-4B (60K)</td><td>79.58</td><td>94.50</td><td>95.00</td><td>90.50</td><td>74.81</td><td>76.83</td><td>56.25</td><td>83.33</td><td>89.90</td><td>76.31</td><td>83.11</td></tr><tr><td>ToolMind-4B (55K)</td><td>77.92</td><td>96.50</td><td>93.50</td><td>90.00</td><td>84.50</td><td>76.35</td><td>62.50</td><td>66.67</td><td>89.48</td><td>77.57</td><td>83.53</td></tr><tr><td>ToolLoop-4B-Isolate (10K)</td><td>79.33</td><td>97.00</td><td>94.50</td><td>93.50</td><td>86.05</td><td>79.87</td><td>81.25</td><td>79.17</td><td>91.08</td><td>81.05</td><td>86.07</td></tr><tr><td>ToolLoop-4B (11K)</td><td>79.67</td><td>96.50</td><td>94.50</td><td>94.50</td><td>86.82</td><td>80.34</td><td>75.00</td><td>79.17</td><td>91.29</td><td>81.50</td><td>86.40</td></tr></table>

Table 1: Comparison on the BFCL (last updated on 2025-12-16) across non-live and live scenarios. The best result within each category is highlighted in bold.

<table><tr><td>Model Variant</td><td>Non-Live</td><td>Live</td><td>Overall</td></tr><tr><td>Qwen3-4B-Instruct (Base)</td><td>87.88</td><td>76.39</td><td>82.14</td></tr><tr><td>w/o Feedback</td><td>85.02</td><td>74.97</td><td>79.97</td></tr><tr><td>w/ Final Filtering</td><td>88.81</td><td>76.31</td><td>82.56</td></tr><tr><td>w/ Feedback (ToolLoop)</td><td>91.29</td><td>81.50</td><td>86.40</td></tr></table>

Table 2: Ablation study of ToolLoop components on BFCL.

This ablation highlights a practical distinction between filtering and refinement. Final filtering can remove malformed samples, but it cannot recover cases where the user query, target function sequence, and final arguments are individually plausible yet mutually inconsistent. By providing feedback at each intermediate stage, ToolLoop corrects these alignment errors before they propagate to later steps, which explains why the full framework improves both non-live and live performance rather than merely increasing the number of accepted samples.

## 4.4 Generalization to Broader Tool-Use Benchmarks

To assess cross-benchmark generalization, we evaluate ToolLoop on ACEBench, which covers atomic operations, single-turn interactions, API similarity, and profile-based tool selection. To keep the comparison controlled, Table 3 focuses on the base model and two data-centric baselines, APIGen-4B and ToolMind-4B.

ToolLoop-4B achieves the best overall score of 72.1%, improving over APIGen-4B by 5.1 points while using only 18.3% of its training data, and over ToolMind-4B by 1.9 points while using one fifth of its data. It also performs best on Atom (84.0%) and Similar API (78.0%), indicating that ToolLoop improves both basic tool-call semantics and fine-grained tool discrimination. These gains show that the closed-loop synthesis process transfers beyond BFCL-specific formatting.

The category-level results also reveal useful limits. In Single Turn, ToolLoop reaches 66.5%, outperforming APIGen-4B and the base model but trailing ToolMind-4B (69.5%), making single-turn intent following a useful direction for improvement. In Profile, all fine-tuned models underperform the base model, suggesting that synthetic toolcall fine-tuning can weaken personalized selection when user preferences are not explicitly modeled. ToolLoop nevertheless scores 60.0%, higher than APIGen-4B and ToolMind-4B (both 54.0%).

<table><tr><td>Model</td><td>Overall</td><td>Atom</td><td>Single Turn</td><td>Similar API</td><td>Profile</td></tr><tr><td>Qwen3-4B-Instruct-2507</td><td>64.9</td><td>68.0</td><td>59.5</td><td>68.0</td><td>64.0</td></tr><tr><td>APIGen-4B (60K)</td><td>67.0</td><td>76.0</td><td>62.0</td><td>76.0</td><td>54.0</td></tr><tr><td>ToolMind-4B (55K)</td><td>70.2</td><td>83.3</td><td>69.5</td><td>74.0</td><td>54.0</td></tr><tr><td>ToolLoop-4B (11K)</td><td>72.1</td><td>84.0</td><td>66.5</td><td>78.0</td><td>60.0</td></tr></table>

Table 3: Performance comparison on ACEBench across five evaluation categories.

## 5 Analysis

## 5.1 Synthesis Efficiency

Table 4 presents the distribution of refinement iterations across the three progressive stages of Tool-Loop synthesis. Stage 2 (Backward Query Derivation) has the highest refinement demand, with 18.1% of samples requiring at least one retry. Compared with generate-then-filter pipelines, where failed samples are discarded and regenerated from scratch, ToolLoop concentrates its extra cost on the small fraction of samples requiring two or three retries: 1.72% in Stage 1, 8.05% in Stage 2, and 4.25% in Stage 3. This suggests that closed-loop refinement adds modest overhead while substantially improving data quality.

Across the 11,024 retained examples, we estimate that synthesis consumed 26.13M input tokens and 7.28M output tokens, totaling 33.41M tokens as measured with the Qwen tokenizer. The cost varies by category because simple and multiple examples bypass LLM-based ground-truth sampling in Stage 1, and the Stage 2 query generator receives only the functions involved in the sampled tool chain rather than the full candidate pool. Parallel examples account for the largest share of the total cost because they require LLM-based tool-chain construction and typically contain longer function combinations. This accounting quantifies the computational trade-off of refinement: additional inference is concentrated on invalid intermediate outputs instead of regenerating every rejected example from scratch.

The higher retry rate in Stage 2 is expected because query generation is the point where symbolic function plans must be translated into natural language. A query can be fluent but still invalid if it omits a required parameter, implies an extra tool call, or describes a scenario that no longer matches the sampled function set. These errors are difficult for rule-based checks alone because they require semantic comparison between the query and the target function sequence. The relatively low retry rate in Stage 3 suggests that once the query is well aligned with the ground truth, producing schemacompliant tool calls becomes easier, especially with deterministic checks guarding argument names and structural validity.

This pattern also clarifies why ToolLoop gains more in complex scenarios. Multiple and parallelmultiple cases require the model to distinguish relevant tools from distractors while maintaining a consistent user intent across several calls. If the synthetic query is even slightly underspecified, the final trained model may learn ambiguous toolselection behavior. By resolving such issues during data synthesis, ToolLoop increases the density of training examples that expose the model to hard selection decisions without introducing inconsistent supervision.

<table><tr><td rowspan="2">Stage</td><td colspan="4">Number of Retries</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Stage 1</td><td>10720</td><td>114</td><td>161</td><td>29</td></tr><tr><td>Stage 2</td><td>9029</td><td>1108</td><td>530</td><td>357</td></tr><tr><td>Stage 3</td><td>10257</td><td>299</td><td>298</td><td>170</td></tr></table>

Table 4: Distribution of refinement iterations per stage during ToolLoop synthesis.

We also examined the 280 samples that remained invalid after the maximum of three retries. Among them, 192 repeatedly failed LLM-based semantic verification, 83 failed deterministic rule-based validation, and 5 failed both types of checks. Most discarded cases therefore exhibited persistent semantic inconsistencies rather than only isolated formatting errors. This discarded set is small relative to the 11,024 retained examples, suggesting that the retry limit primarily removes unresolved supervision signals while iterative correction preserves most initially invalid samples.

## 5.2 Verifier Reliability

All three stages employ the same verifier model, Qwen-Max, as the LLM judge for semantic validation. To assess the reliability of this automated verifier, we manually annotated a random sample of 100 instances and compared the results against the model’s judgments, yielding a human-agreement rate of 94%. We treat this result as a sanity check for the semantic verifier rather than as a full validation of every error type or synthesis stage.

We use this manual annotation only as an overall reliability check rather than as a fine-grained error taxonomy. Among the few disagreement cases, errors were mainly associated with parameter-level constraints: for example, when a function schema required an integer argument, the verifier occasionally failed to flag cases where the generated tool call supplied a floating-point value with a fractional part. Such type-related errors can typically be captured by deterministic schema or AST-based checks. We also observed a small number of semantic grounding errors, such as treating a geographic location name as a city when the tool expected a city-level argument; unlike structural or type errors, these cases are harder to detect with AST parsing alone. These disagreements were infrequent and did not indicate a systematic failure pattern. The 94% agreement rate supports the use of Qwen-Max as a semantic verifier, while rule-based and AST-based checks remain necessary for objective constraints such as JSON validity, argument names, data types, and schema conformance. Although this sample size does not provide a complete validation of all error categories, it serves as a sanity check that the verifier is sufficiently reliable for semantic screening.

## 6 Conclusion

This paper presents ToolLoop, a closed-loop framework for synthesizing high-quality tool-use data through decomposed generation and dynamic selffeedback. By decomposing synthesis into ground truth generation, user query derivation, and tool call instantiation, ToolLoop moves tool-use synthesis from generate-then-filter toward generateverify-refine. Experiments on BFCL show that a 4B model trained on only 11K ToolLoop examples achieves 86.40% accuracy in non-reasoning mode, while the BFCL-overlap-filtered Isolate variant still reaches 86.07%. On ACEBench, ToolLoop achieves the best overall score among data-centric methods, suggesting that the synthesized data can transfer beyond the primary benchmark.

Taken together, our results suggest that the main value of ToolLoop lies in improving the internal consistency of synthetic examples rather than simply scaling the amount of supervision. By jointly verifying the intended function sequence, the derived user intent, and the final executable tool calls, ToolLoop produces training examples with better alignment across intermediate components. These findings support a simple takeaway: effective tooluse data should be constructed through an iterative generate–verify–refine process rather than generated as a single query–answer pair and filtered only at the end. ToolLoop provides one practical implementation of this process and offers a starting point for building more reliable tool-use training data.

## Limitations

The absence of real environment feedback means we cannot verify whether our synthesized data adequately prepares models for handling practical challenges such as timeout errors, malformed API responses, or cascading failures in ground truths. Furthermore, real-world tool-use often involves iterative refinement based on execution results, a capability not assessed in current static evaluation protocols. Future work should establish interactive testbeds with executable tool environments to validate the practical applicability of synthetic training data.

ToolLoop also uses Qwen-Max as the semantic verifier across all three synthesis stages. Although the verifier achieves 94% agreement with human annotations on a random sample of 100 instances and is complemented by deterministic schema and AST checks, this evaluation does not rule out verifier-specific or correlated semantic biases. Future work should compare independent verifier models, report stage-level calibration, and incorporate executable environment feedback.

## Ethics Statement

This work focuses on synthetic data generation for tool-use in language models and does not collect personally identifiable information or conduct user studies. The manual verifier-reliability assessment annotates only synthetic examples. All experiments use publicly available API specifications and benchmarks.

## References

Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. 2024. Agent s: An open agentic framework that uses computers like a human. arXiv preprint arXiv:2410.08164.

Antonis Antoniades, Albert Örwall, Kexun Zhang, Yuxi Xie, Anirudh Goyal, and William Wang. 2024. Swesearch: Enhancing software agents with monte carlo tree search and iterative refinement. arXiv preprint arXiv:2410.20285.

Dzmitry Bahdanau, Nicolas Gontier, Gabriel Huang, Ehsan Kamalloo, Rafael Pardinas, Alex Piché, Torsten Scholak, Oleh Shliazhko, Jordan Prince Tremblay, Karam Ghanem, and 1 others. 2024. Tapeagents: a holistic framework for agent development and optimization. arXiv preprint arXiv:2412.08445.

Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fernández, and et al. 2025. Llms instead of human judges? a large scale empirical study across 20 nlp evaluation tasks. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Short Papers).

Chen Chen, Xinlong Hao, Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Yuefeng Huang, Xiangcheng Liu, Wang Xinzhi, and 1 others. 2025. Acebench: A comprehensive evaluation of llm tool usage. Findings ofthe Associationfor Computational Linguistics: EMNLP, pages 12970–12998.

Jennifer D’Souza, Hamed Babaei Giglou, and Quentin Münch. 2025. Yescieval: Robust llm-as-a-judge for scientific question answering. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics.

Mohamed Amine Ferrag, Norbert Tihanyi, and Merouane Debbah. 2025. From llm reasoning to autonomous ai agents: A comprehensive review. arXiv preprint arXiv:2504.19678.

Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. 2024. Scaling synthetic data creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094.

Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Qiguang Chen, and 1 others. 2025. Owl: Optimized workforce learning for general multiagent assistance in real-world task automation. arXiv preprint arXiv:2505.23885.

Seungone Kim, Juyoung Suk, Xiang Yue, Vijay Viswanathan, Seongyun Lee, Yizhong Wang, Kiril Gashteovski, Carolin Lawrence, Sean Welleck, and Graham Neubig. 2025. Evaluating language models as synthetic data generators. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6385–6403.

Md Tahmid Rahman Laskar, Israt Jahan, Elham Dolatabadi, Chun Peng, Enamul Hoque, and Jimmy Huang. 2025. Improving automatic evaluation of large language models (llms) in biomedical relation extraction via llms-as-the-judge. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics.

Jiawei Li, Yang Gao, Yizhe Yang, Yu Bai, Xiaofeng Zhou, Yinghao Li, Huashan Sun, Yuhang Liu, Xingpeng Si, Yuhao Ye, and 1 others. 2025. Fundamental capabilities and applications of large language models: A survey. ACM Computing Surveys.

Shuliang Liu, Xinze Li, Zhenghao Liu, Yukun Yan, Cheng Yang, and et al. 2025. Judge as a judge: Improving the evaluation of retrieval-augmented generation through the judge-consistency of large language models. In Findings ofthe ACL.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, and 1 others. 2024a. Toolace: Winning the points of llm function calling. arXiv preprint arXiv:2409.00920.

Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh RN, and 1 others. 2024b. Apigen: Automated pipeline for generating verifiable and diverse function-calling datasets. Advances in Neural Information Processing Systems, 37:54463–54482.

Andrea Matarazzo and Riccardo Torlone. 2025. A survey on large language models with some insights on their capabilities and limitations. arXiv preprint arXiv:2501.04040.

AI Meta. 2025. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai. meta. com/blog/llama-4-multimodalintelligence/, checked on, page 2025.

Shishir G. Patil, Huanzhi Mao, Charlie Cheng-Jie Ji, Fanjia Yan, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. 2025. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning.

Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, and 1 others. 2025. Apigen-mt: Agentic pipeline for multi-turn data generation via simulated agenthuman interplay. arXiv preprint arXiv:2504.03601.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, and 1 others. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Team Qwen. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford\_alpaca.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, and 1 others. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Jingwei Wang, Zai Zhang, Hao Qian, Chunjing Gan, Binbin Hu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Bin Shi, and Bo Dong. 2025a. Enhancing llm tool use with high-quality instruction data from knowledge graph. arXiv preprint arXiv:2506.21071.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 13484–13508.

Zezhong Wang, Xingshan Zeng, Weiwen Liu, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu, and Kam-Fai Wong. 2025b. Toolflow: Boosting llm tool-calling through natural and coherent dialogue synthesis. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4246–4263.

Zhaoyang Wang, Yiming Liang, Xuchao Zhang, Qianhui Wu, Siwei Han, Anson Bastos, Rujia Wang, Chetan Bansal, Baolin Peng, Jianfeng Gao, and 1 others. 2025c. Adapting web agents with synthetic supervision. arXiv preprint arXiv:2511.06101.

Weikai Xu, Chengrui Huang, Shen Gao, and Shuo Shang. 2025a. Llm-based agents for tool learning: A survey: W. xu et al. Data Science and Engineering, pages 1–31.

Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, and Rameswar Panda. 2025b. Toucan: Synthesizing 1.5 m tool-agentic data from real-world mcp environments. arXiv preprint arXiv:2510.01179.

Yusuke Yamauchi, Taro Yano, and Masafumi Oyamada. 2025. An empirical study of llm-as-a-judge: How design choices impact evaluation reliability. arXiv.

Chen Yang, Ran Le, Yun Xing, Zhenwei An, Zongchao Chen, Wayne Xin Zhao, Yang Song, and Tao Zhang. 2025. Toolmind technical report: A large-scale, reasoning-enhanced tool-use dataset. arXiv preprint arXiv:2511.15718.

Junjie Ye, Guanyu Li, Songyang Gao, Caishuang Huang, Yilong Wu, Sixian Li, Xiaoran Fan, Shihan Dou, Tao

Ji, Qi Zhang, and 1 others. 2025. Tooleyes: Finegrained evaluation for tool learning capabilities of large language models in real-world scenarios. In Proceedings of the 31st international conference on computational linguistics, pages 156–187.

Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long Le, Kai-Wei Chang, Chen-Yu Lee, and 1 others. 2025. Magnet: Multiturn tool-use data synthesis and distillation via graph translation. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32600–32616.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, and 1 others. 2025. Swift: a scalable lightweight infrastructure for fine-tuning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 29733–29735.

AI Zhipu. 2025. Glm-4.6: Advanced agentic, reasoning and coding capabilities.

## A The distribution of synthetic data

Through the ToolLoop approach, we synthesized 11K data points, whose distribution across four query categories is shown in Figure 4. Out of the 11,024 total examples, the dataset consists of 4,453 simple (40.4%), 3,634 parallel (33.0%), 1,783 multiple (16.2%), and 1,154 parallel-multiple (10.5%) instances.

![](images/f0335d0435544609108e01c1a855e6a67785957eb558367bdb161511388218bf.jpg)  
Figure 4: Distribution of 11K synthetic training examples across four categories.

## B Prompt Design

To ensure reproducibility and transparency of our approach, we provide the complete prompt templates used in each stage of ToolLoop’s decomposed generation process. Figures 5, 6, and 7 present the detailed prompts for Stage 1 (Ground Truth Sampling), Stage 2 (Backward Derivation of User Query), and Stage 3 (Forward Derivation of Tool Calls), respectively. Each prompt incorporates our dynamic self-feedback mechanism through the History of Feedback section, which guides iterative refinement based on identified issues from previous attempts.

![](images/36ad37be57bbe39dc2ccd431ae122d2ec6a9d2a43ad0d02762c16d7ff2a39bd6.jpg)  
Figure 5: Prompt template for Stage 1: Ground Truth Sampling with dynamic self-feedback.

![](images/6561eeacad0163db1009d8e8f9fc416444c4ae092cc8ed91264e7ba950f8876c.jpg)  
Figure 6: Prompt template for Stage 2: Backward Derivation of User Query with dynamic self-feedback.

![](images/d56310f263065b37204297b1f0014f2c41010bf49a572841d34fb06efe6d1b6f.jpg)  
Figure 7: Prompt template for Stage 3: Forward Derivation of Tool Calls with dynamic self-feedback.