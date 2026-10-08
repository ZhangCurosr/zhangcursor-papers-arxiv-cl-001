# PHRBENCH: A BEHAVIORAL EVALUATION OF POST-HALLUCINATION REASONING IN LLMS

Linghao Meng<sup>1,∗</sup> Feng He<sup>2,∗</sup> Xuan Yang<sup>1,∗</sup> Junyuan Mao<sup>1</sup> Pinze Ren<sup>3</sup> Deqing Mu<sup>4</sup> Hesen Yang<sup>1</sup> Qiankun Li<sup>5,†</sup>

<sup>1</sup>National University of Singapore <sup>2</sup>Independent Researcher <sup>3</sup>Tsinghua University <sup>4</sup>Johns Hopkins University <sup>5</sup>Nanyang Technological University

{menglinghao25,xuanyang}@u.nus.edu hefengcs@gmail.com cs-qiankun.li@ntu.edu.sg <sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

## ABSTRACT

Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of posthallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBENCH, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBENCH characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning trajectory. We further find that properties of the hallucinated prompt contain substantial predictive signal for successful recovery, with a lightweight predictor achieving an AUROC of 0.847. These findings provide a behavioral view of post-hallucination reasoning, characterizing how LLMs resolve erroneous context and when successful recovery is likely to occur.

<sup>§</sup> Code: github.com/menglinghao2025/PHR

õ Dataset: huggingface.co/datasets/menglinghao2025/PHR

 Homepage: menglinghao2025.github.io/PHR

## 1 INTRODUCTION

Despite substantial progress in detection and mitigation methods, large language models continue to produce hallucinations across diverse tasks and settings (Zhang et al., 2025; Farquhar et al., 2024; Li, 2025; Wu et al., 2023; Kalai et al., 2025; Janiak et al., 2025; Tamber et al., 2025; Ganesh et al., 2026). Existing safeguards can reduce such errors, but cannot reliably prevent every hallucinated statement from entering subsequent processing (Xu et al., 2024; Kalai et al., 2025; Janiak et al., 2025). This limitation becomes increasingly consequential as LLM outputs are reused across retrieval systems, tool-augmented pipelines, autonomous agents, and multi-stage generation workflows (Niu et al., 2024; Yao et al., 2022; Wu et al., 2023). In these systems, an output produced at one stage often becomes the context for subsequent reasoning. As illustrated in Figure 1(a), hallucinated information that escapes upstream detection can therefore propagate through the system and act as a premise for downstream decisions (Zhang et al., 2023; Zhong et al., 2024; Jamshidi, 2026; Cui et al., 2026; Hu et al., 2025). System reliability consequently depends not only on preventing hallucinations, but also on whether downstream models can recover once hallucinated information is already present. HIVE (He et al., 2026) introduces post-hallucination reasoning (PHR) to study this downstream reliability problem and shows that hallucination can persistently reshape subsequent reasoning.

HIVE (He et al., 2026) further reveals that hallucinated context can either impair or improve downstream performance, indicating that its effects cannot be explained by factuality alone. However, existing PHR analysis mainly characterizes changes in final outcomes and aggregate reasoning dynamics, without systematically distinguishing response-level behaviors toward hallucinated premises (Ganesh et al., 2026; Lee et al., 2024a; Grover et al., 2024; Zhan et al., 2026; Lu et al., 2026b). As previewed in Figure 1(b), the same correct answer may arise from fundamentally different behaviors. A model may accept the hallucinated semantics yet still arrive at the correct answer, bypass the premise without resolving it, or explicitly identify and correct the underlying error. Conversely, a model may identify and correct a hallucinated premise but still fail to derive the correct answer. Final correctness therefore conflates compliance, avoidance, successful correction, and correction failure. Aggregate trajectory statistics provide additional evidence about how reasoning changes, but they still do not identify the resolution behavior taken by an individual response. Without distinguishing resolution behavior from final outcome, a correct answer may be mistaken for successful hallucination resolution.

![](images/a54e970358111038016b01fe06898e732deaa5fd4db97a7e7a3329a96d378b9c.jpg)  
Figure 1: Overview of behaviorally structured post-hallucination reasoning. (a) Hallucinations that escape upstream detection can propagate into downstream reasoning, making recovery critical for reliable LLM systems. (b) Final correctness alone cannot reveal how a model handles hallucinated context. (c) PHRBENCH reveals a scaling tension between vulnerability and correction, characterizes and predicts successful correction before generation.

In this paper, we study PHR as a behaviorally structured resolution process through PHRBENCH, a controlled benchmark of 4820 instances spanning Chemistry, Biomedicine, Physics, and Code Generation. We evaluate 18 proprietary and open-source models and characterize each reasoning trajectory independently of final-answer correctness as Hallucination Compliance, Hallucination Avoidance, or Heuristic Correction. We further define an insightful trajectory as one that performs Heuristic Correction and reaches the correct answer. Using this framework, we ask two questions. Q1: How do LLMs resolve hallucinated premises, and how does successful recovery vary with model scale and reasoning dynamics? We analyze behavioral resolution and trajectory dynamics across models and scales. Q2: Can successful recovery be predicted before response generation? We evaluate the predictive signal of prompt-level structural and semantic features.

Our results, summarized in Figure 1(c), reveal three main findings. First, scaling creates a tension between contextual vulnerability and corrective capability: hallucinated context reduces accuracy by 7.7 percentage points on average, with larger degradation at greater scales within multiple model families, while corrective behavior becomes more frequent. Second, successful recovery remains rare and exhibits a distinct reasoning signature. Insightful trajectories are associated with more frequent belief updates, suggesting more effective revision after conflicts are detected. Third, successful recovery contains substantial ex ante predictive signal. Prompt-level features achieve an AUROC of 0.847, with both structural and semantic factors contributing to prediction.

Our key contributions are as follows:

❶ We introduce PHRBENCH for behaviorally structured post-hallucination reasoning. Across 4 domains and 18 LLMs, PHRBENCH contains 4820 controlled instances and separates behavioral resolution from final-answer correctness through compliance, avoidance, and correction.

❷ We reveal a scaling tension and characterize successful recovery. Model scaling reveals a tension between robustness to hallucinated context and corrective capability, while successful recovery exhibits a distinct pattern of reasoning revision.

❸ We show that successful recovery contains substantial ex ante predictive signal. Using lightweight structural and semantic features, we show that successful recovery can be anticipated before generation, providing a foundation for intervention mechanisms.

## 2 RELATED WORK

Post-Hallucination Reasoning Transmission. Prior work on post-hallucination reasoning has examined how hallucinated information is transmitted across subsequent generations and interactions (Chen et al., 2024; Gan et al., 2025; Xiong et al., 2026; Zhang et al., 2023; 2026; Jiang et al., 2026). MMHalSnowball (Zhong et al., 2024) shows that earlier hallucinations can mislead later responses even when visual evidence remains available. MM-Snowball (Jiang et al., 2026) examines how hallucinated information accumulates as models increasingly rely on contaminated conversational history. The Woozle Effect (Zhang et al., 2026) investigates how hallucinated information is transmitted across agents and influences participants. Hallucination Cascade (Jamshidi et al., 2026) and Collective Hallucination (Jamshidi, 2026) further analyze how hallucinated con tent evolves through sequential agent interactions. Together, these studies trace where hallucinated information propagates across downstream multi reasoning processes.

Post-Hallucination Reasoning Dynamics. Beyond transmission, another line of work examines how hallucinated outputs are processed in subsequent reasoning (Lu et al., 2026b;a; Dhuliawala et al., 2024; Tyen et al., 2024). Volcano (Lee et al., 2024b) uses self-feedback to revise initial hallucinated responses, while Pelican (Sahu et al., 2024) decomposes generated claims and verifies them through program-of-thought reasoning. A<sup>2</sup>R (Lee et al., 2024a) uses explicit hallucination assessment to guide iterative refinement. DoT (Grover et al., 2024) navigates hallucinated thoughts to improve subsequent reasoning. SHD (Lu et al., 2026b) models hallucination in long CoT as an evolving state across reasoning steps, while DeepHalluBench (Zhan et al., 2026) audits intermediate hallucinations throughout full research trajectories rather than only on final outputs. These studies characterize PHR through verification, revision, and trajectory-level analysis, but do not distinguish the response-level behaviors through which a model resolves hallucinated content.

Behavioral Resolution in Post-Hallucination Reasoning. HIVE formalizes the broader setting of post-hallucination reasoning, showing that hallucinated semantics can alter downstream outcomes and reasoning dynamics (He et al., 2026). However, these analyses do not reveal how an individual response resolves the hallucinated content it encounters. Simultaneously, different resolution behaviors can lead to the same final outcome. Our work therefore studies PHR as a behaviorally structured resolution process, separating response-level resolution behavior from final-answer correctness.

## 3 PHRBENCH

In this section, we introduce PHRBENCH, a benchmark designed to systematically evaluate posthallucination reasoning in LLMs. PHRBENCH is constructed by first collecting diverse reasoning questions across multiple domains and task formats, and then augmenting them with controlled truthful and hallucinated semantic contexts. This construction enables us to examine not only whether hallucination changes the final answer, but also how it alters the reasoning trajectory.

## Overall, PHRBENCH has three key characteristics:

✤ Diverse Reasoning Scenarios. PHRBENCH ensures diversity along two dimensions. (1) Multiple Domains: PHRBENCH contains 460 base questions spanning Chemistry, Biomedicine, Physics, and Code Generation, covering heterogeneous domain knowledge. (2) Multiple Tasks: PHRBENCH includes both multiple-choice and open-ended questions, allowing us to examine PHR under both constrained answer spaces and free-form generation settings. The questions are curated from established benchmarks with varying levels of task complexity, providing broad reasoning scenarios.

![](images/c55a4500dc07a5680fb48f2d0982a0261ce50bce253becca61438daae7feb466.jpg)

![](images/385429161524b8363df4de892f61aead8dadd2a39494b639a44eabfb082de672.jpg)  
Figure 2: Dataset statistics of PHRBENCH. (a) Reasoning category distribution of the 460 base questions. (b) Domain-wise composition of hallucinated augmentations across 3 construction types.

✤ Controlled Semantic Augmentation. PHRBENCH constructs controlled semantic contexts for each base question to isolate the effect of hallucination on reasoning. (1) Truthful and Hallucinated Contexts: each question is paired with one truthful and multiple hallucinated augmentations, all preserving task relevance and avoiding answer leakage for controlled comparison. (2) Multiple Augmentation Patterns: hallucinated augmentations cover three forms of semantic corruption: Rule Contradiction, State Distortion, and Pseudoscientific Entanglement, corresponding to domain principle violations, misreported task-relevant conditions, and outdated scientific theories.

✤ Reasoning-Oriented Evaluation. PHRBENCH evaluates PHR from both outcome and reasoning-process perspectives. For outcome performance, we use Accuracy (Acc) to quantify whether hallucinated context changes the final answer. For reasoning trajectories, we introduce four diagnostic metrics: Output Length (OL), Uncertainty Index (UI), Belief Update Frequency (BUF), and Branching Complexity (BC). This analytical framework enables a fine-grained diagnosis of how hallucinated semantics affect both the final answer and the reasoning process leading to it. Detailed definitions of these metrics are provided in Section 4.2.

Data Collection and Statistics. Selecting tasks that meaningfully expose PHR is crucial for constructing PHRBENCH. We therefore focus on scientific reasoning tasks that require models to combine domain knowledge, task-specific conditions, and intermediate inference to derive verifiable outcomes. We consider four domains: Chemistry, Biomedicine, Physics, and Code Generation. The first three emphasize reasoning over scientific knowledge and established principles, while Code Generation extends this setting to computational reasoning.

Following this design, we collect 460 base questions from eight established benchmarks across the four domains. The scientific questions are drawn from MoleculeNet-BBBP (Wu et al., 2018), ChemBench (Mirza et al., 2025), MedMCQA (Pal et al., 2022), MMLU (Hendrycks et al., 2020), SciQ (Welbl et al., 2017), and GPQA (Rein et al., 2023), while coding problems are sourced from HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021), covering both multiple-choice and open-ended reasoning formats. Figure 2 illustrates the distribution of reasoning subtypes and the domain-wise composition of the hallucinated augmentations across three construction types. Detailed source statistics and sampling procedures are provided in Appendix A.1.

Augmentation Construction. Before augmentation, we standardize the collected questions using task-specific templates that preserve the original task semantics and evaluation criteria. For each standardized base question, we construct one truthful augmentation and multiple hallucinated augmentations without revealing the answer. The truthful version remains factually valid, while each hallucinated version introduces a single targeted semantic mutation with the remaining context preserved. The resulting PHRBENCH contains 460 base questions, 460 truthful augmentations, and

3900 hallucinated augmentations, yielding 4820 instances in total. Details of templates, quality control, and representative examples are provided in Appendix A.2 and Appendix A.3.

## 4 EXPERIMENTAL SETUP

## 4.1 MODELS

We evaluate 18 representative LLMs on PHRBENCH, covering proprietary and open-source models across different parameter scales, model generations, and training paradigms (Anthropic, 2025; OpenAI, 2024b;a; 2025; Comanici et al., 2025; Google DeepMind, 2024; Yang et al., 2024; 2025; Meta, 2024; Grattafiori et al., 2024; Glm et al., 2024; Jiang et al., 2023). To ensure consistency in our experimental setup, we set the temperature to 0.7 and the maximum output size to the maximum supported size for each model. Additionally, for each model and prompt, we independently sample 10 responses to account for generation stochasticity. See Appendix B.1 for all configurations.

## 4.2 EVALUATION METRICS

PHRBENCH evaluates post-hallucination reasoning using two groups of diagnostic metrics. Outcome Metric characterizes final-answer performance, while Reasoning Trajectory Metrics capture changes in the reasoning process.

Outcome Metric. PHRBENCH uses Accuracy (Acc) to measure the proportion of responses whose final answers match the ground truth.

Reasoning Trajectory Metrics. PHRBENCH uses four metrics to characterize reasoning trajectories: Output Length, Uncertainty Index, Belief Update Frequency, and Branching Complexity.

Output Length (OL) measures the total number of tokens generated in a response.

Uncertainty Index (UI) measures the average token-level uncertainty along the reasoning trajectory:

$$
U I = - { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \sum _ { v \in V } P ( x _ { v } \mid x _ { < i } ) \log P ( x _ { v } \mid x _ { < i } ) ,\tag{1}
$$

where N denotes the generated sequence length, V is the model vocabulary, and $P ( x _ { v } \mid x _ { < i } )$ denotes the probability assigned to token v given the preceding context $x _ { < i }$

Belief Update Frequency (BUF) measures how frequently next-token distributions shift substantially between consecutive reasoning steps:

$$
B U F = \frac { 1 } { K - 1 } \sum _ { k = 2 } ^ { K } \mathbb { I } \left( D _ { \mathrm { K L } } \left( P _ { k } \parallel P _ { k - 1 } \right) > \tau \right) ,\tag{2}
$$

where K denotes the number of reasoning steps, $P _ { k }$ and $P _ { k - 1 }$ denote consecutive belief distribu tions, D<sub>KL</sub> is the Kullback–Leibler divergence, and τ is the update threshold.

Branching Complexity (BC) measures the prevalence of alternative reasoning paths using predefined branching-related lexical cues, normalized by response length:

$$
B C = \frac { 1 0 0 } { N } \sum _ { k \in \mathcal { K } } \mathrm { C o u n t } ( k , r ) ,\tag{3}
$$

where N is its total word count, K denotes the predefined cue set, and $\operatorname { C o u n t } ( k , r )$ is the number of occurrences of cue k in generated response r. Full details are provided in Appendix B.2.

## 5 RESULTS AND ANALYSIS

## 5.1 MAIN RESULTS

Evaluation Protocol. Following the evaluation protocol described in Section 4, we evaluate all 18 models under both the truthful augmentation and hallucinated augmentation settings, denoted as T and H in Table 1. Comprehensive results are provided in Appendix C.

Table 1: Outcome performance and reasoning trajectory statistics under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T H</td><td>T</td><td>H</td><td>T</td><td>H</td><td>T</td><td>H</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td></td><td> $4 8 . 4 _ { \downarrow 9 . 1 }$ </td><td>100.7</td><td>113.2</td><td>0.054</td><td>0.071</td><td>0.152 0.329</td><td></td><td>1.133</td><td>1.111</td></tr><tr><td>GPT-4o-mini</td><td>57.5 55.5</td><td> $4 8 . 3 _ { \downarrow 7 . 2 }$ </td><td>71.2</td><td>72.4</td><td>0.072</td><td>0.059</td><td>0.130</td><td>0.186</td><td>0.752</td><td>0.720</td></tr><tr><td>GPT-4o</td><td>51.8</td><td> $4 2 . 3 _ { \downarrow 9 . 5 }$ </td><td>60.0</td><td>60.8</td><td>0.061</td><td>0.053</td><td>0.113</td><td>0.214</td><td>0.659</td><td>0.618</td></tr><tr><td>GPT-5.2</td><td>61.6</td><td> $4 4 . 7 _ { \downarrow 1 6 . 9 }$ </td><td>43.0</td><td>45.1</td><td>0.007</td><td>0.016</td><td>0.013</td><td>0.026</td><td>0.578</td><td>0.515</td></tr><tr><td>Gemini-2.0-Flash</td><td>56.2</td><td> $4 5 . 1 _ { \downarrow 1 1 . 1 }$ </td><td>76.2</td><td>86.5</td><td>0.046</td><td>0.061</td><td>0.092</td><td>0.178</td><td>0.745</td><td>0.785</td></tr><tr><td>Gemini-2.5-Pro</td><td>58.5</td><td> $4 6 . 4 _ { \downarrow 1 2 . 1 }$ </td><td>82.4</td><td>95.3</td><td>0.045</td><td>0.068</td><td>0.124</td><td>0.244</td><td>0.852</td><td>0.850</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>24.6</td><td> $2 0 . 6 _ { \downarrow 4 . 0 }$ </td><td>115.6</td><td>117.1</td><td>0.272</td><td>0.218</td><td>0.091</td><td>0.076</td><td>1.987</td><td>1.748</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>44.5</td><td> $3 8 . 0 _ { \downarrow 6 . 5 }$ </td><td>153.2</td><td>147.1</td><td>0.319</td><td>0.151</td><td>0.148</td><td>0.121</td><td>2.265</td><td>1.939</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>46.8</td><td> $3 9 . 6 _ { \downarrow 7 . 2 }$ </td><td>228.4</td><td>215.1</td><td>0.245</td><td>0.150</td><td>0.285</td><td>0.353</td><td>2.820</td><td>2.644</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>52.2</td><td> $4 4 . 2 _ { \downarrow 8 . 0 }$ </td><td>276.9</td><td>262.8</td><td>0.188</td><td>0.112</td><td>0.414</td><td>0.485</td><td>3.426</td><td>3.150</td></tr><tr><td>Qwen3-8B</td><td>46.0</td><td> $4 3 . 8 _ { \downarrow 2 . 2 }$ </td><td>225.8</td><td>256.6</td><td>0.217</td><td>0.072</td><td>0.576</td><td>0.392</td><td>2.855</td><td>3.470</td></tr><tr><td>Qwen3-32B</td><td>48.5</td><td> $4 1 . 0 _ { \downarrow 7 . 5 }$ </td><td>296.3</td><td>288.4</td><td>0.181</td><td>0.065</td><td>0.605</td><td>0.465</td><td>3.232</td><td>3.820</td></tr><tr><td>Qwen3-235B-A22B</td><td>58.2</td><td> $4 7 . 4 _ { \downarrow 1 0 . 8 }$ </td><td>365.0</td><td>415.2</td><td>0.085</td><td>0.045</td><td>0.843</td><td>0.753</td><td>4.155</td><td>4.828</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>34.2</td><td> $3 1 . 9 _ { \downarrow 2 . 3 }$ </td><td>210.5</td><td>199.9</td><td>0.378</td><td>0.322</td><td>0.371</td><td>0.207</td><td>2.752</td><td>2.222</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>48.4</td><td> $4 0 . 9 _ { \downarrow 7 . 5 }$ </td><td>285.2</td><td>275.4</td><td>0.431</td><td>0.419</td><td>0.523</td><td>0.384</td><td>3.356</td><td>2.989</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>31.2</td><td> $2 9 . 5 _ { \downarrow 1 . 7 }$ </td><td>167.0</td><td>153.4</td><td>0.159</td><td>0.137</td><td>0.322</td><td>0.124</td><td>2.076</td><td>1.441</td></tr><tr><td>GLM-4-9B-Chat</td><td>32.3</td><td> $2 2 . 4 _ { \downarrow 9 . 9 }$ </td><td>127.3</td><td>79.0</td><td>0.171</td><td>0.173</td><td>0.050</td><td>0.038</td><td>1.886</td><td>1.122</td></tr><tr><td>Mistral-7B-Instruct</td><td>21.1</td><td> $1 5 . 7 _ { \downarrow 5 . 4 }$ </td><td>131.0</td><td>122.4</td><td>0.130</td><td>0.104</td><td>0.262</td><td>0.370</td><td>1.853</td><td>1.472</td></tr></table>

![](images/b44e859b7fddea063327987ecf0e492c7ecd6c927d10c004d539c680fbd43fd8.jpg)  
Figure 3: Left. Accuracy Retention across all models and types. Impact of task domain (Middle) and hallucinated augmentation type (Right) on Accuracy Retention, separated by model type.

❶ Hallucinated context increasingly degrades stronger models. Hallucinated context consistently reduces final-answer accuracy across all models, with the degradation becoming more pronounced for stronger and larger models. As shown in Table 1, the average accuracy drop is 7.7%, with GPT-5.2 showing the largest decrease (16.9% ↓), followed by Gemini-2.5-Pro (12.1% ↓) and Gemini-2.0-Flash (11.1% ↓). Proprietary models also exhibit larger degradation on average than open-source models (11.0% vs. 6.1%). More importantly, degradation consistently increases with model scale. For example, scaling Qwen2.5 from 1.5B to 72B improves baseline accuracy by 27.6% ↑, while the degradation under hallucinated context doubles to 8.0%. Figure 3 further reveals substantial heterogeneity across tasks and augmentation types. Chemistry shows the lowest average retention across the models, while Physics is comparatively more robust. Across augmentation types, State Distortion causes the strongest degradation, whereas Pseudosci-

Table 2: Behavioral resolution proportions across evaluated models.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Comp. (%)</td><td rowspan=1 colspan=1>Avoid. (%)</td><td rowspan=1 colspan=1>HC (%)</td></tr><tr><td rowspan=2 colspan=1>Claude-Sonnet-4</td><td rowspan=1 colspan=2>Proprietary Models</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>47.23</td><td rowspan=1 colspan=1>33.49</td><td rowspan=1 colspan=1>19.28</td></tr><tr><td rowspan=1 colspan=1>GPT-4o-mini</td><td rowspan=1 colspan=1>32.62</td><td rowspan=1 colspan=1>60.54</td><td rowspan=1 colspan=1>6.85</td></tr><tr><td rowspan=1 colspan=1>GPT-40</td><td rowspan=1 colspan=1>45.87</td><td rowspan=1 colspan=1>44.08</td><td rowspan=1 colspan=1>10.05</td></tr><tr><td rowspan=1 colspan=1>GPT-5.2</td><td rowspan=1 colspan=1>41.10</td><td rowspan=1 colspan=1>55.21</td><td rowspan=1 colspan=1>3.69</td></tr><tr><td rowspan=1 colspan=1>Gemini-2.0-Flash</td><td rowspan=1 colspan=1>43.43</td><td rowspan=1 colspan=1>49.09</td><td rowspan=1 colspan=1>7.48</td></tr><tr><td rowspan=1 colspan=1>Gemini-2.5-Pro</td><td rowspan=1 colspan=1>45.59</td><td rowspan=1 colspan=1>42.55</td><td rowspan=1 colspan=1>11.86</td></tr><tr><td rowspan=2 colspan=1>GLM-4-9B-Chat</td><td rowspan=2 colspan=1>Open-source Mod19.57</td><td rowspan=1 colspan=1>els</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>78.45</td><td rowspan=1 colspan=1>1.98</td></tr><tr><td rowspan=1 colspan=1>Mistral-7B-Instruct</td><td rowspan=1 colspan=1>47.66</td><td rowspan=1 colspan=1>39.38</td><td rowspan=1 colspan=1>12.97</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Comp. (%)</td><td rowspan=1 colspan=1>Avoid. (%)</td><td rowspan=1 colspan=1>HC (%)</td></tr><tr><td rowspan=2 colspan=1>OpenQwen2.5-1.5B-Instruct</td><td rowspan=1 colspan=1>-source Models</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>32.14</td><td rowspan=1 colspan=1>60.28</td><td rowspan=1 colspan=1>7.58</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-7B-Instruct</td><td rowspan=1 colspan=1>59.55</td><td rowspan=1 colspan=1>28.04</td><td rowspan=1 colspan=1>12.41</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-32B-Instruct</td><td rowspan=1 colspan=1>43.72</td><td rowspan=1 colspan=1>36.08</td><td rowspan=1 colspan=1>20.20</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-72B-Instruct</td><td rowspan=1 colspan=1>39.34</td><td rowspan=1 colspan=1>34.18</td><td rowspan=1 colspan=1>26.48</td></tr><tr><td rowspan=1 colspan=1>Qwen3-8B-Instruct</td><td rowspan=1 colspan=1>47.12</td><td rowspan=1 colspan=1>27.32</td><td rowspan=1 colspan=1>25.56</td></tr><tr><td rowspan=1 colspan=1>Qwen3-32B-Instruct</td><td rowspan=1 colspan=1>38.74</td><td rowspan=1 colspan=1>30.86</td><td rowspan=1 colspan=1>30.40</td></tr><tr><td rowspan=1 colspan=1>Qwen3-235B-A22B-Instruct</td><td rowspan=1 colspan=1>33.18</td><td rowspan=1 colspan=1>27.74</td><td rowspan=1 colspan=1>39.08</td></tr><tr><td rowspan=1 colspan=1>Llama-3.2-3B-Instruct</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>68.88</td><td rowspan=1 colspan=1>6.12</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-8B-Instruct</td><td rowspan=1 colspan=1>40.68</td><td rowspan=1 colspan=1>46.39</td><td rowspan=1 colspan=1>12.93</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-70B-Instruct</td><td rowspan=1 colspan=1>39.15</td><td rowspan=1 colspan=1>41.68</td><td rowspan=1 colspan=1>19.17</td></tr></table>

entific Entanglement retains approximately 98% of baseline accuracy. Thus, susceptibility to hallucinated context also depends on how the hallucinated information conflicts with the task.

❷ Scaling amplifies reasoning activity under both augmentations. Larger models consistently exhibit larger OL, BUF, and BC. As shown in Table 1, this scaling pattern holds across the Qwen2.5, Qwen3, and Llama-3.1 families under both settings. For example, scaling Qwen2.5 from 1.5B to 72B yields substantial increases in OL of 161.3 under T and 145.7 under H. BUF rises by 0.323 and 0.409, while BC grows by 1.439 and 1.402, respectively. Across models, OL, BUF, and BC also strongly co-vary, with Pearson correlations of 0.908, 0.977, and 0.856, whereas UI shows no consistent scaling pattern. These results indicate that scaling primarily amplifies the extent and complexity of reasoning, while uncertainty follows a comparatively distinct pattern.

Takeaway 1. Greater reasoning capacity does not guarantee greater robustness. Instead, stronger models can exhibit richer reasoning dynamics while becoming more vulnerable to hallucinated context, revealing a strong-model trap.

## 5.2 HOW DO MODELS RESOLVE HALLUCINATED PREMISES?

To understand how models resolve hallucinated premises, we analyze their responses from three complementary perspectives. First, we identify models’ behavioral resolution and insightful trajectories and then characterize their patterns. Second, we examine the reasoning dynamics associated with different resolution behaviors and insightful trajectories. Finally, we complement these aggregate analyses with qualitative cases to illustrate how behavioral resolution unfolds during reasoning.

Resolution Definitions. We categorize each reasoning trajectory into three behavioral modes according to how the model handles the hallucinated premise: Hallucination Compliance, where the model accepts the premise; Hallucination Avoidance, where the model bypasses the premise; and Heuristic Correction, where the model identifies and corrects the premise. This categorization is independent of final-answer correctness. To further capture successful post-hallucination reasoning, we define an insightful trajectory as one that performs heuristic correction and ultimately reaches the correct answer. Detailed definitions and examples are provided in Appendix D.1.

❸ Active correction remains rare, but strengthens with model scale. Most models predominantly comply with or avoid hallucinated premises rather than explicitly correcting them. As shown in Table 2, heuristic correction remains relatively infrequent across the 18 models, but increases consistently with model scale. For example, scaling Qwen2.5 from 1.5B to 72B increases heuristic correction by 18.90% ↑, with similar scaling trends observed for Qwen3 and Llama-3.1. According to Figure 4, insightful trajectories follow the same overall trend but remain less frequent: 11 of 18 models produce them in fewer than 10% of responses, while Qwen3-235B-A22B achieves the highest rate at 24.91%. The persistent gap between heuristic correction and insightful trajectories indicates that revising a hallucinated semantic does not necessarily lead to a correct final answer.

❹ Successful resolution is associated with higher belief updating. Different resolution behaviors exhibit distinct reasoning patterns of distributional change under the reference scorer. As shown in Figure 5, heuristic correction produces, on average, about 30 more reasoning words and a 0.31 higher BUF than hallucination compliance across the four representative models. Hallucination avoidance instead exhibits the highest average branching complexity. More importantly, insightful trajectories achieve an average BUF of 0.63, compared with only 0.18 for non-insightful trajectories. These results suggest that successful resolution is associated with more frequent distributional shifts and longer reasoning trajectories. Full details are provided in Appendix D.2.

![](images/367c82e954274177a55957702f8be1d45aa83829e3ecd1e59d6a90d2244a08c6.jpg)  
Figure 4: Insightful trajectory rates across the evaluated models.

![](images/effe242e73f595423cdb5ba1b480c0f228407fcddf1bc0de03d084cb506b5645.jpg)

![](images/e37113b73b02516d7718d4397556b6528fa565d5aa9b94ca478bf9209e04c680.jpg)

![](images/5d9f47241d00b64f6d8becea614a66f3676cecb497b51daa63c29aa8bce97259.jpg)

![](images/0bbc8feb4f0bc1563695c7999ce00e11b617c84c2b4ab847eca883b79121853a.jpg)  
Figure 5: Reasoning dynamics across behavioral resolution modes and insightful trajectories for four representative models.

❺ Insightful trajectories reject conflicting premises and sustain the correction. Qualitative cases show that relevant knowledge alone is insufficient for resolving hallucinated premises. As shown in Figure 6, non-insightful trajectories may access the correct knowledge but ultimately allow the injected premise to override it. Insightful trajectories instead reject the conflicting premise and redirect subsequent reasoning toward the correct answer. This contrast supports effective belief updating as the key distinction of successful resolution. More cases are provided in Appendix D.3.

Takeaway 2. Scaling increases models’ tendency to correct hallucinated premises, but correction alone is insufficient. Insightful resolution depends on whether detected conflicts trigger effective belief revision that redirects subsequent reasoning.

## 5.3 CAN WE PREDICT INSIGHTFUL TRAJECTORIES?

Section 5.2 shows that insightful trajectories are relatively rare but exhibit distinctive reasoning dynamics. In this section, we further examine whether their occurrence can be anticipated from properties of the input prompt before generation. Specifically, we characterize prompts using structural and semantic features, evaluate their predictive power, and analyze the features and interactions that contribute to the prediction. Full details are provided in Appendix E.

Prompt-Level Features. We construct 24 features to characterize factors that may trigger an insightful trajectory, covering both prompt structure and hallucination–task semantics. Representative features include Total Prompt Length, Candidate Count, Domain-Rule Violation, Error Locality, and Applicability Mismatch. Full definitions are provided in Appendix E.1.

![](images/b057454af670eb53c35c087d9ea804a6e6af993fff100b85970d728a21bdb221.jpg)  
Figure 6: Qualitative comparison of insightful and non-insightful trajectories.

❻ Insightful trajectories are predictable from prompt-level properties. Prompt characteristics contain substantial predictive signal for whether an insightful trajectory will emerge. We extract the 24 features for each hallucinated prompt using an external LLM (Zheng et al., 2023; Li et al., 2025) and train an XGBRegressor (Chen & Guestrin, 2016) to estimate its propensity to induce insightful reasoning. The resulting predictor achieves an AUROC of 0.847 and an accuracy of 0.863, indicating that the occurrence of insightful reasoning is systematically associated with properties available before generation rather than being entirely incidental.

❼ Structural and semantic features provide complementary predictive signals. The predictive signal is distributed across both structural and semantic properties of the prompt. As shown in Figure 7a, Total Prompt Length is the strongest individual predictor, accounting for 20.7% of the global SHAP contribution (Lundberg & Lee, 2017; Lundberg et al., 2020), substantially exceeding other structural features. Hallucinated augmentation-specific semantics also contribute substantially, with Domain-Rule Violation and Error Locality accounting for 7.4% and 7.2%, respectively.

This predictive structure is not dominated by strong pairwise interactions. As shown in Figure 7b, the largest interaction occurs between Total Prompt Length and Question Length (0.020), while nearly all remaining pairwise interactions are below 0.01. Together with their non-trivial individual contributions, these weak interactions suggest that the selected features decouple and capture largely complementary aspects of the conditions associated with insightful reasoning.

Takeaway 3. Insightful reasoning is predictable before generation. Its emergence reflects complementary structural and hallucination–task semantic properties of the prompt as dominant factors.

## 6 CONCLUSION

In this paper, we introduce PHRBENCH, a comprehensive benchmark for behaviorally structured post-hallucination reasoning across four domains and 18 LLMs. With the evaluation results, our analysis shows that hallucinated context consistently degrades performance, while larger models simultaneously exhibit stronger corrective behavior, revealing a tension between vulnerability and recovery capability. By separating hallucination compliance, hallucination avoidance, and heuristic correction from final outcomes, we further show that successful correction remains relatively rare and is characterized by stronger belief updating. We also find that successful correction can be anticipated from properties of the hallucinated semantics before generation. A lightweight predictor based on structural and semantic features achieves strong predictive performance, with factors such as domain-rule violation and error locality contributing substantially. Together, these results suggest that reliable PHR requires evaluating not only whether a model reaches the correct answer, but also how it resolves hallucinated context and when successful recovery is likely to occur.

![](images/f88035ae49afe6e7c8cec7c12dbfa5587c36bce43482cb6a5557844e22a330d5.jpg)  
(a) Global SHAP contribution and directional effects of the most influential prompt-level features.

![](images/9b2918496d1579c18cd42f5b9d3dc8f9b31640f8cc7d05534e4e4fbf4fe31ef9.jpg)  
(b) Pairwise SHAP interaction effects among the most influential prompt-level features.  
Figure 7: SHAP analysis of prompt-level features for predicting insightful trajectories.

## REFERENCES

Anthropic. Introducing claude 4. https://www.anthropic.com/news/claude-4, 2025.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Kedi Chen, Qin Chen, Jie Zhou, He Yishen, and Liang He. Diahalu: A dialogue-level hallucination evaluation benchmark for large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 9057–9079, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785–794, 2016.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, and Cong Zuo. Free-mad: Consensus-free multiagent debate. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 31977–31997, 2026.

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. Chain-of-verification reduces hallucination in large language models. In Findings ofthe associationfor computational linguistics: ACL 2024, pp. 3563–3578, 2024.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

Zeyu Gan, Yun Liao, and Yong Liu. Rethinking external slow-thinking: From snowball errors to probability of correct reasoning. arXiv preprint arXiv:2501.15602, 2025.

Prakhar Ganesh, Reza Shokri, and Golnoosh Farnadi. Rethinking hallucinations: Correctness, consistency, and prompt multiplicity. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 6959–6978, 2026.

Team Glm, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.

Google DeepMind. Introducing gemini 2.0: Our new ai model for the agentic era. Google Blog, December 2024. URL https://blog.google/ innovation-and-ai/models-and-research/google-deepmind/ google-gemini-ai-update-december-2024/.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Shresth Grover, Vibhav Vineet, and Yogesh S Rawat. Navigating hallucinations for reasoning of unintentional activities. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 9666–9680, 2024.

Feng He, Zhenting Wang, Qifan Wang, Qiang Guan, Dongfang Liu, Ruixiang Tang, and Qiankun Li. Hive: Understanding post-hallucination reasoning in vision language models. ECCV, 2026.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason Zhang, Xiao-Yong Wei, and Qing Li. Removal of hallucination on hallucination: Debate-augmented rag. In Proceedings of the 63rd annual meeting of the association for computational linguistics (volume 1: long papers), pp. 15839–15853, 2025.

Saeid Jamshidi. Collective hallucination in multi-agent llms: Modeling and defense. arXiv preprint arXiv:2606.07941, 2026.

Saeid Jamshidi, Arghavan Moradi Dakhel, Kawser Wazed Nafi, and Foutse Khomh. Hallucination cascade: Analyzing error propagation in multi-agent llm systems. arXiv preprint arXiv:2606.07937, 2026.

Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Shwartz-Ziv, and Tomasz Jan Kajdanowicz. The illusion of progress: Re-evaluating hallucination detection in llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 34716–34733, 2025.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. URL https://arxiv.org/abs/2310.06825.

Yue Jiang, Xue Jiang, Lihua Zhang, Zhiqiang Wang, Yuhang Lu, Peng Wang, Bo Han, Feng Zheng, and Dingkang Yang. Mm-snowball: Evaluating and mitigating hallucination snowballing in multimodal multi-turn dialogue. arXiv preprint arXiv:2606.00622, 2026.

Adam Tauman Kalai, Ofir Nachum, Santosh S Vempala, and Edwin Zhang. Why language models hallucinate. arXiv preprint arXiv:2509.04664, 2025.

Dongyub Lee, Eunhwan Park, Hodong Lee, and Heui-Seok Lim. Ask, assess, and refine: Rectifying factual consistency and hallucination in llms with metric-guided feedback learning. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2422–2433, 2024a.

Seongyun Lee, Sue Hyun Park, Yongrae Jo, and Minjoon Seo. Volcano: mitigating multimodal hallucination through self-feedback guided revision. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 391–404, 2024b.

Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al. From generation to judgment: Opportunities and challenges of llm-as-a-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 2757–2791, 2025.

Xinzhe Li. A review of prominent paradigms for llm-based agents: Tool use, planning (including rag), and feedback learning. In Proceedings ofthe 31st international conference on computational linguistics, pp. 9760–9779, 2025.

Haolang Lu, Yilian Liu, Jingxin Xu, Guoshun Nan, Yuanlong Yu, Zhican Chen, and Kun Wang. Auditing meta-cognitive hallucinations in reasoning large language models. Advances in Neural Information Processing Systems, 38:162500–162543, 2026a.

Haolang Lu, Minghui Pan, Guoshun Nan, Jialin Zhuang, Zijie Zhao, ZhongXiang Sun, Kun Wang, Yang Liu, et al. Streaming hallucination detection in long chain-of-thought reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 21157–21183, 2026b.

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.

Scott M Lundberg, Gabriel Erion, Hugh Chen, Alex DeGrave, Jordan M Prutkin, Bala Nair, Ronit Katz, Jonathan Himmelfarb, Nisha Bansal, and Su-In Lee. From local explanations to global understanding with explainable ai for trees. Nature machine intelligence, 2(1):56–67, 2020.

Meta. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. https://ai.meta.com/blog/ llama-3-2-connect-2024-vision-edge-mobile-devices/, 2024.

Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, Martino R˜ ´ıos-Garc´ıa, Benedict Emoekabu, Aswanth Krishnan, Tanya Gupta, Mara Schilling-Wilhelmi, Macjonathan Okereke, Anagha Aneesh, et al. A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists. Nature Chemistry, 17(7):1027–1034, 2025.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, and Tong Zhang. Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Proceedings of the 62nd Annual Meeting of the Association for Computa tional Linguistics (Volume 1: Long Papers), pp. 10862–10878, 2024.

OpenAI. GPT-4o System Card. arXiv preprint arXiv:2410.21276, 2024a. URL https://arxiv. org/abs/2410.21276.

OpenAI. GPT-4o mini: advancing cost-efficient intelligence. https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024b.

OpenAI. Introducing GPT-5.2. https://openai.com/index/ introducing-gpt-5-2/, 2025.

Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pp. 248–260. PMLR, 2022.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

Pritish Sahu, Karan Sikka, and Ajay Divakaran. Pelican: Correcting hallucination in vision-llms via claim decomposition and program of thought verification. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8228–8248, 2024.

Manveer Singh Tamber, Forrest Bao, Chenyu Xu, Ge Luo, Suleman Kazi, Minseok Bae, Miaoran Li, Ofer Mendelevitch, Renyi Qu, and Jimmy Lin. Benchmarking llm faithfulness in rag with evolving leaderboards. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 799–811, 2025.

Gladys Tyen, Hassan Mansoor, Victor Carbune, Yuanzhu Peter Chen, and Tony Mak. Llms cannot˘ find reasoning errors, but can correct them given the error location. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 13894–13908, 2024.

Johannes Welbl, Nelson F Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings ofthe 3rd Workshop on Noisy User-generated Text, pp. 94–106, 2017.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi agent conversation. arXiv preprint arXiv:2308.08155, 2023.

Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.

Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Zirui Liu, Jiliang Tang, Himabindu Lakkaraju, and Zhen Xiang. How memory management impacts llm agents: An empirical study of experiencefollowing behavior. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 623–645, 2026.

Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language models. arXiv preprint arXiv:2401.11817, 2024.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Yuhao Zhan, Tianyu Fan, Linxuan Huang, Zirui Guo, and Chao Huang. Why your deep research agent fails? on hallucination evaluation in full research trajectory. arXiv preprint arXiv:2601.22984, 2026.

Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model hallucinations can snowball. arXiv preprint arXiv:2305.13534, 2023.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: A survey on hallucination in large language models. Computational Linguistics, 51(4):1373–1418, 2025.

Zhiheng Zhang, Yuanzhe Zhang, Daojian Zeng, Kang Liu, and Jun Zhao. Beware of the woozle effect: Exploring and mitigating hallucination propagation in multi-agent debate. IEEE Transactions on Audio, Speech and Language Processing, 2026.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Weihong Zhong, Xiaocheng Feng, Liang Zhao, Qiming Li, Lei Huang, Yuxuan Gu, Weitao Ma, Yuan Xu, and Bing Qin. Investigating and mitigating the multimodal hallucination snowballing in large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11991–12011, 2024.

## APPENDIX

Appendix A. Details of PHRBench . . . 14   
Appendix B. Detailed Experimental Setup . . . 20   
Appendix C. Detailed Results of Main Experiments . . 23   
Appendix D. Detailed Analysis of Behavioral Resolution . . . 28   
Appendix E. Details of Insightful Trajectory Prediction . . . 35

## A DETAILS OF PHRBENCH

## A.1 DATA COLLECTION AND STATISTICS

Data Collection. We construct PHRBENCH by collecting 460 base questions from eight established benchmarks across four domains: Chemistry, Biomedicine, Physics, and Code Generation. We retain questions that require domain-specific or computational reasoning and organize the collected instances by domain as follows.

Chemistry. Chemistry reasoning requires models to infer molecular properties and apply established chemical principles and mechanisms. We collect 50 molecular-property questions from MoleculeNet-BBBP, 50 chemistry reasoning questions from ChemBench, and 20 open-ended chemistry questions from GPQA, yielding 120 questions in total.

Biomedicine. Biomedical reasoning requires models to integrate medical and biological knowledge with task-specific conditions to derive diagnoses, mechanisms, or conclusions. We select 100 questions from MedMCQA and 20 open-ended biology questions from GPQA, resulting in 120 questions.

Physics. Physics reasoning requires models to apply established physical laws, conceptual relations, and quantitative constraints. We collect 14 questions from MMLU College Physics, 36 from MMLU Conceptual Physics, 50 from SciQ, and 20 open-ended physics questions from GPQA, yielding 120 questions.

Code Generation. Code generation requires models to interpret semantic specifications, compose procedural logic, and produce executable solutions. We select 50 problems from HumanEval and 50 from MBPP, resulting in 100 coding problems.

Reasoning Category Annotation. After integrating questions from the eight source benchmarks, we further categorize all 460 instances according to their primary reasoning requirements. The taxonomy follows a hierarchical structure of Domain → Reasoning Subtype. When a source benchmark provides an appropriate native task or subdomain label, we retain that label whenever possible; otherwise, questions are harmonized into a unified domain-specific taxonomy. Importantly, the assigned subtype reflects the primary reasoning process required to solve the question rather than the benchmark from which it originates. This provides a unified characterization of reasoning requirements across heterogeneous source datasets.

Table 3 reports the resulting reasoning-category distribution. For reasoning-category annotation, we first preserve native task or subdomain labels when they are compatible with our taxonomy. Questions without suitable native labels are manually mapped to the domain-specific reasoning subtypes according to their primary reasoning requirement. Each question is assigned to a single subtype to ensure a mutually exclusive partition of the 460 base questions. The resulting taxonomy captures substantial within-domain variation in reasoning requirements and enables subsequent analysis to distinguish effects associated with domain from those associated with specific reasoning patterns.

## A.2 AUGMENTATION CONSTRUCTION

Standardization. Before augmentation, we standardize questions from different source benchmarks using task-specific templates while preserving the original task semantics, answer space, and evaluation criteria. For each base question, we construct three standardized input forms: the origina question (B), the question paired with a truthful augmentation (B+T), and the question paired with a hallucinated augmentation (B+H). The augmentation is inserted as additional contextual information without modifying the underlying question or explicitly revealing the target answer. All standardized instances are converted into English and follow a consistent input structure within each task type, enabling controlled comparison across the baseline, truthful, and hallucinated augmentation settings.

<table><tr><td>Domain</td><td>Reasoning Subtype</td><td>Count</td><td>Percentage</td></tr><tr><td rowspan="6">Chemistry</td><td>Molecular Property Reasoning</td><td>50</td><td>41.7%</td></tr><tr><td>Organic Chemistry</td><td>28</td><td>23.3%</td></tr><tr><td>Fundamental Chemistry</td><td>21</td><td>17.5%</td></tr><tr><td>Physical Chemistry</td><td>15</td><td>12.5%</td></tr><tr><td>Chemical Analysis</td><td>5</td><td>4.2%</td></tr><tr><td>Chemical Safety</td><td>1</td><td>0.8%</td></tr><tr><td rowspan="8">Biomedicine</td><td>Clinical Reasoning</td><td>29</td><td>24.2%</td></tr><tr><td>Physiological Reasoning</td><td>26</td><td>21.7%</td></tr><tr><td>Molecular Biology</td><td>25</td><td>20.8%</td></tr><tr><td>Genetic Reasoning</td><td>12</td><td>10.0%</td></tr><tr><td>Evolutionary Biology</td><td>11</td><td>9.2%</td></tr><tr><td>Host-Microbe Biology</td><td>7</td><td>5.8%</td></tr><tr><td>Experimental Biology</td><td>7</td><td>5.8%</td></tr><tr><td>Biomedical Procedures</td><td>3</td><td>2.5%</td></tr><tr><td rowspan="6">Physics</td><td>Modern Physics</td><td>32</td><td>26.7%</td></tr><tr><td>Wave Physics</td><td>25</td><td>20.8%</td></tr><tr><td>Electromagnetism</td><td>21</td><td>17.5%</td></tr><tr><td>Classical Mechanics</td><td>18</td><td>15.0%</td></tr><tr><td>Fundamental Physics</td><td>17</td><td>14.2%</td></tr><tr><td>Thermal Physics</td><td>7</td><td>5.8%</td></tr><tr><td rowspan="6">Code Generation</td><td>Text Processing</td><td>24</td><td>24.0%</td></tr><tr><td>Numerical Computation</td><td>22</td><td>22.0%</td></tr><tr><td>Data Manipulation</td><td>15</td><td>15.0%</td></tr><tr><td>Logical Validation</td><td>15</td><td>15.0%</td></tr><tr><td>Search and Optimization</td><td>12</td><td>12.0%</td></tr><tr><td>Combinatorial Algorithms</td><td></td><td></td></tr><tr><td></td><td></td><td>12</td><td>12.0%</td></tr></table>

Table 3: Reasoning subtype distribution of the 460 base questions in PHRBENCH. Percentages are computed within each domain.

Augmentation Types. We categorize hallucinated augmentations into three types according to the nature of the injected semantic error.

Rule Contradiction introduces a premise that directly conflicts with an established rule, principle, or domain fact. Its falsity is independent of the specific problem instance, and identifying the error primarily requires retrieving and maintaining the relevant domain knowledge.

State Distortion introduces a statement that may be valid in general but is incorrect under the specific conditions of the current problem. Detecting such an error therefore requires instance-level verification of the relevant properties, conditions, intermediate states, or computational logic rather than relying solely on general knowledge.

Pseudoscientific Entanglement introduces an outdated, discredited, or pseudoscientific theory as a premise for reasoning. This type evaluates whether the model can maintain consistency with established scientific knowledge when confronted with an apparently explanatory but invalid theoretical account.

Together, these augmentation types create semantic conflicts at different levels, ranging from violations of established knowledge to instance-specific mismatches and invalid explanatory frameworks.

Construction Procedure. We use GPT-4o as the generator to construct augmentations for each standardized base question. Given the original task, the generator produces one truthful augmentation and multiple hallucinated augmentations that are directly relevant to the question. The truthful augmentation provides factually valid supplementary information, whereas each hallucinated augmentation introduces a targeted semantic error without explicitly revealing the answer. Specifically, for each question in Chemistry, Biomedicine, and Physics, we construct one truthful augmentation and 10 hallucinated augmentations, whereas each Code Generation problem is paired with one truthful augmentation and 3 hallucinated augmentations. This results in 460 truthful augmentations and 3900 hallucinated augmentations in total. The domain-wise distribution of the three hallucinated augmentation types is shown in Figure 2.

To encourage semantic diversity, hallucinated augmentations associated with the same base question are required to differ in their underlying error rather than merely rephrasing the same incorrect claim. Depending on the task, the generator constructs hallucinated augmentations using the three augmentation types described above, with particular emphasis on creating both direct rule conflicts and instance-specific state distortions.

For traceability, each hallucinated augmentation is generated together with a mutation strategy, which records its augmentation type, and a mutation logic, which explicitly describes the injected error. Each truthful augmentation is accompanied by a truth verification field specifying why the added information is valid. The resulting augmentations are subsequently assembled with the standardized base questions to form the B+T and B+H evaluation instances.

Quality Control. We apply a multi-stage quality-control procedure to ensure that the constructed augmentations are semantically valid, task-relevant, and consistent with their intended mutation types.

First, we impose explicit constraints during generation. Each hallucinated augmentation is required to introduce a single targeted semantic mutation, while preserving the original question and the remaining contextual information. Hallucinated augmentations associated with the same base question must correspond to different erroneous claims or affected properties rather than surface-level paraphrases of the same error. Truthful augmentations are required to be factually correct, relevant to the corresponding task, and free from the errors introduced by the hallucinated augmentations. Neither truthful nor hallucinated augmentations are allowed to explicitly reveal the target answer.

Second, we manually review all generated augmentations, including both truthful and hallucinated augmentations, to assess their suitability for the corresponding base question. For truthful augmentations, we check that the added information is factually correct, relevant to the question, and applicable under the conditions of the original task. For hallucinated augmentations, we check that the added information is relevant to the question but contains the intended semantic error, is consistent with the assigned augmentation type, and introduces only one dominant erroneous premise. We also remove or revise augmentations that are ambiguous, redundant with other augmentations of the same question, insufficiently related to the task, or likely to leak the answer.

Third, we perform an additional LLM-as-a-judge verification using Gemini 3 Pro. For each candidate augmentation, the judge re-evaluates its factual validity, relevance to the base question, and consistency with the assigned mutation type. For hallucinated augmentations, the judge further checks whether the stated mutation logic correctly identifies the underlying semantic error; for truthful augmentations, it verifies that the added information remains factually valid and does not conflict with the original task. This second-pass verification serves as an independent consistency check after manual review.

Only augmentations that satisfy the generation constraints and pass both manual review and the subsequent LLM-based verification are retained in the final benchmark. This layered procedure reduces annotation noise and helps ensure that the observed model behavior can be attributed to the intended semantic perturbation rather than artifacts introduced during augmentation construction.

Augmentation Prompts. We use separate prompts for the two augmentation types, and each generation call produces exactly one augmentation so that truthful controls and individual semantic mutations are constructed independently.

## PROMPT T (Truthful Augmentation Generation):

You are an expert in evaluating the reasoning robustness of large language models. Given the following base question, generate exactly one truthful augmentation that provides supplementary information relevant to the task.

## Requirements:

• The augmentation must be factually correct and consistent with established domain knowledge.

• It must be directly relevant to the reasoning required by the base question.

• It should provide useful contextual or background information without changing the original task.

• It must not explicitly reveal or directly imply the target answer.

• Generate exactly one augmentation.   
Input:   
Base Question: {x}   
Output Format:   
{   
"statement": "...",   
"truth\_verification": "..."   
}

## PROMPT H (Hallucinated Augmentation Generation):

You are an expert in evaluating the reasoning robustness of large language models. Given the following base question, generate exactly one hallucinated augmentation that is directly relevant to the task and introduces a single targeted semantic error.

## Mutation Types:

• Rule Contradiction: Introduce a premise that directly contradicts an established rule, principle, or domain fact. Its falsity should be independent of the particular problem instance.

• State Distortion: Introduce a statement that may be valid in general but is incorrect under the specific conditions of the current problem. Detecting the error should require instance-level verification ofthe relevant properties, conditions, intermediate states, or computational logic.

• Pseudoscientific Entanglement: Introduce an outdated, discredited, or pseudoscientific theory as a premise for reasoning.

## Requirements:

• Generate exactly one hallucinated augmentation.

• The augmentation must be directly relevant to the base question.

• It must introduce only one dominant semantic error.

• All other information should remain consistent with the original task.

• The augmentation must not explicitly reveal or directly imply the target answer.

• The error should be substantive rather than a superficial wording change.

• Assign the augmentation to one ofthe mutation types defined above.

Input:   
Base Question: {x}   
Output Format:   
{   
"mutation\_strategy": "...",   
"mutation\_logic": "...",   
"statement": "..."   
}

## A.3 EXAMPLES OF TASKS

To illustrate the construction of PHRBENCH, we present one representative example from each domain. For each example, we show the standardized base question (B), a truthful augmentation (B+T), and a hallucinated augmentation (B+H). For compactness, the base question is shown only once; the B+T and B+H variants are obtained by prepending the corresponding augmentation to the same base question.

An Example Data of Physics Task   
B (Baseline).   
Question: A wave transfers   
Candidates: (1) amplitude; (2) wavelength; (3) frequency; (4) energy.   
B+T (Truthful Augmentation).   
Added premise: As a wave propagates, particles of the medium may oscillate around their   
equilibrium positions rather than being permanently transported with the disturbance.   
B+H (Hallucinated Augmentation; Pseudoscientific Entanglement).   
Added premise: The transfer of a wave occurs due to the propagation of ether, a medium that   
permeates all space.

An Example Data of Chemistry Task   
B (Baseline).   
Question: Which alcohol listed here has the highest boiling point?   
Candidates: (1) butan-1-ol; (2) butan-2-ol; (3) isobutyl alcohol; (4) t-butyl alcohol.   
B+T (Truthful Augmentation).   
Added premise: For isomeric alcohols, increased branching generally decreases molecular   
surface area and can weaken London dispersion interactions.   
B+H (Hallucinated Augmentation; State Distortion).   
Added premise: The more branched an alcohol is, the stronger its intermolecular forces,   
leading to a higher boiling point.

## An Example Data of Biomedicine Task

## B+H (Hallucinated Augmentation; Rule Contradiction).

Added premise: c-erbB1 is a tumor suppressor gene because all genes involved in cell signaling can suppress tumor growth under certain conditions.

An Example Data of Code Generation Task   
B (Baseline).   
def choose num(x, y):   
"""This function takes two positive numbers x and y and   
returns the   
biggest even integer number that is in the range [x, y]   
inclusive.   
If there is no such number, the function should return -1.   
choose num(12, 15) = 14   
choose num(13, 12) = -1   
I n n   
B+T (Truthful Augmentation).   
Added premise: The ordering of x and y should not be assumed beyond what is specified in   
the task, and the behavior illustrated by all provided examples must be preserved.

B+H (Hallucinated Augmentation; State Distortion).   
Added premise: The inputs x and y will always be in ascending order, so x is guaranteed to be less than or equal to y.

## B DETAILED EXPERIMENTAL SETUP

## B.1 MODEL SETUP

We summarize the API model identifiers or released checkpoints for all 18 evaluated models in Table 4, together with their supported context window and maximum output length. Our evaluation includes six proprietary models and twelve open-source models, covering multiple model families, generations, and parameter scales. For proprietary models, we report the corresponding upstream API model identifiers used during evaluation; for open-source models, we report the publicly released model checkpoints. This diverse model set further enables controlled scaling comparisons within the Qwen2.5, Qwen3, and Llama-3.1 families.
<table><tr><td>Model Name</td><td>API / Model Code</td><td>Context Window</td><td>Max Output Length</td></tr><tr><td colspan="4">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td> $\mathtt { c l a u d e - s o n n e t - 4 - 2 0 2 5 0 5 1 4 }$ </td><td>200,000</td><td>64,000</td></tr><tr><td>GPT-4o-mini</td><td>gpt-4o-mini</td><td>128,000</td><td>16,384</td></tr><tr><td>GPT-40</td><td>gpt-4o</td><td>128,000</td><td>16,384</td></tr><tr><td>GPT-5.2</td><td> $\mathfrak { g p t - 5 . 2 - 2 0 2 5 - 1 2 - 1 1 }$ </td><td>400,000</td><td>128,000</td></tr><tr><td>Gemini-2.0-Flash</td><td> $\mathfrak { g e m i n i - 2 } . 0 \mathrm { - } \pounds 1 \mathsf { a s h - } 0 0 1$ </td><td>1,048,576</td><td>8,192</td></tr><tr><td>Gemini-2.5-Pro</td><td> $\mathtt { g e m i n i - } 2 . 5 \mathrm { - p r o }$ </td><td>1,048,576</td><td>65,536</td></tr><tr><td colspan="4">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td> $\mathtt { Q w e n / Q w e n 2 . 5 - 1 . 5 B - I n s t r u c t }$ </td><td>32,768</td><td>8,192</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td> $\mathtt { Q w e n / Q w e n 2 . 5 - 7 B - I n s t r u c t }$ </td><td>131,072*</td><td>8,192</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td> $\mathtt { Q w e n } / \mathtt { Q w e n } 2 . 5 \mathtt { - } 3 2 \mathtt { B } \mathrm { - } \mathtt { I n } \mathtt { s t r u c t }$ </td><td>131,072*</td><td>8,192</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td> $\mathtt { Q w e n / Q w e n 2 . 5 - 7 2 B - I n s t r u c t }$ </td><td>131,072*</td><td>8,192</td></tr><tr><td>Qwen3-8B</td><td> $\mathsf { Q w e n / Q w e n 3 - 8 B }$ </td><td>32,768†</td><td>38,912</td></tr><tr><td>Qwen3-32B</td><td> ${ \sf Q w e n } / { \sf Q w e n } 3 - 3 2 { \sf B }$ </td><td>32,768†</td><td>38,912</td></tr><tr><td>Qwen3-235B-A22B</td><td> $\mathtt { Q w e n } / \mathtt { Q w e n } 3 - 2 3 5 \mathtt { B } - \mathtt { A } 2 2 \mathtt { B }$ </td><td>32,768†</td><td>38,912</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td> $\mathrm { m e t a - 1 1 a m a / L 1 a m a - 3 . 1 - 8 B - I n s t r u c t }$ </td><td>131,072</td><td></td></tr><tr><td>Llama-3.1-70B-Instruct</td><td> $\mathrm { m e t a - 1 1 a m a / L 1 a m a - 3 . 1 \mathrm { - 7 0 B - I n s t r u c t } }$ </td><td>131,072</td><td></td></tr><tr><td>Llama-3.2-3B-Instruct</td><td> $\mathrm { m e t a - 1 1 a m a / L 1 a m a - 3 . 2 \mathrm { - 3 B - I n s t r u c t } }$ </td><td>131,072</td><td></td></tr><tr><td>GLM-4-9B-Chat</td><td> $\mathrm { T H U D M / q l m - 4 - 9 b - c h a t }$ </td><td>131,072</td><td></td></tr><tr><td>Mistral-7B-Instruct</td><td> $\mathtt { m i s t r a l a i / M i s t r a l - 7 B - I n s t r u c t - v 0 . 2 }$ </td><td>32,768</td><td></td></tr></table>

<sup>∗</sup> Qwen2.5-7B/32B/72B support context lengths up to 131,072 tokens; their default configurations use 32,768 tokens and require YaRN for long-context deployment.  
<sup>†</sup> Qwen3 models natively support 32,768-token contexts and can be extended to 131,072 tokens using YaRN. <sup>‡</sup> Following the official Qwen3 recommendation for complex benchmarking, the maximum output length is set to 38,912 tokens. For models marked with “–”, no separate maximum generation length is specified by the released checkpoint; generation is bounded by the configured context capacity.

Table 4: Model configurations used in the evaluation of PHRBENCH. For proprietary models, we report the upstream API model identifiers corresponding to the evaluated versions; for open-source models, we report the released model checkpoints.

Inference Configuration. To ensure consistency across models, we use a unified decoding protocol throughout the evaluation. We set the sampling temperature to 0.7 for all models and use the maximum supported output length for each model. For models without a separately specified generation limit, generation is bounded by the context capacity of the corresponding checkpoint and inference framework. For each model–prompt pair, we independently sample 10 responses to account for generation stochasticity. All benchmark conditions for the same model use identical decoding configurations, ensuring that the baseline, truthful-augmentation, and hallucinated augmentation settings differ only in their input context.

## B.2 EVALUATION METRICS

We evaluate each generated response from two complementary perspectives: final-answer correctness and reasoning-trajectory characteristics. Accuracy measures whether the model ultimately reaches the correct answer, whereas Output Length (OL), Uncertainty Index (UI), Belief Update Frequency (BUF), and Branching Complexity (BC) characterize different properties of the generated reasoning trajectory. To ensure comparability between proprietary and open-source models, probability-based trajectory metrics are computed post hoc using a fixed reference language model, Qwen2.5-7B-Instruct. The same reference scorer and metric configuration are used for all models, domains, and experimental conditions.

Accuracy. Accuracy (Acc) measures whether the final answer extracted from a response matches the ground-truth answer. For a response $r ,$ we define

$$
A c c ( r ) = \mathbb { I } \left[ \hat { y } ( r ) = y ^ { * } \right] ,\tag{4}
$$

where $\hat { y } ( r )$ denotes the extracted final answer, $y ^ { * }$ denotes the ground-truth answer, and $\mathbb { I } [ \cdot ]$ is the indicator function. For multiple-choice tasks, the predicted option is matched against the gold label. For open-ended tasks, we follow the task-specific evaluation criterion inherited from the correspond ing source benchmark.

Output Length. Output Length (OL) measures the length of the generated reasoning trajectory. Given a generated response

$$
r = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { N } ) ,\tag{5}
$$

we define

$$
{ \cal O } L ( r ) = N ,\tag{6}
$$

where N denotes the length of the generated response. OL captures the overall extent of explicit reasoning, but does not by itself indicate reasoning quality.

Uncertainty Index. Uncertainty Index (UI) measures the average token-level predictive uncertainty along the reasoning trajectory. To obtain probability distributions consistently across models, each generated response is post-scored using the fixed reference model. At position i, let

$$
P ( x _ { v } \mid x _ { < i } )\tag{7}
$$

denote the probability assigned by the reference scorer to vocabulary token v conditioned on the preceding generated trajectory $x _ { < i }$ . UI is computed as the average entropy of these predictive distributions:

$$
U I = - { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \sum _ { v \in V } P ( x _ { v } \mid x _ { < i } ) \log P ( x _ { v } \mid x _ { < i } ) ,\tag{8}
$$

where V denotes the vocabulary of the reference scorer. A larger UI corresponds to a less concentrated predictive distribution and therefore greater uncertainty along the trajectory.

Belief Update Frequency. Belief Update Frequency (BUF) measures how frequently referencemodel next-token distributions shift substantially between consecutive reasoning steps, following Eq. 2. We first segment each generated reasoning trajectory into sentence-level steps using sentencefinal punctuation and line breaks as boundaries, after removing empty segments. A response is therefore represented as an ordered sequence

$$
\mathcal { R } = ( s _ { 1 } , s _ { 2 } , \ldots , s _ { K } ) .\tag{9}
$$

For reasoning step $s _ { k }$ , we concatenate the original task with all reasoning steps up to $s _ { k }$

$$
c _ { k } = [ q ; s _ { 1 } ; . . . ; s _ { k } ] ,\tag{10}
$$

and define $P _ { k }$ as the next-token predictive distribution from the fixed reference scorer:

$$
P _ { k } ( v ) = P _ { \mathrm { r e f } } ( v \mid c _ { k } ) , \qquad v \in V .\tag{11}
$$

The distributional change between consecutive reasoning steps is quantified by the Kullback–Leibler divergence,

$$
\Delta _ { k } = D _ { \mathrm { K L } } \left( P _ { k } \parallel P _ { k - 1 } \right) .\tag{12}
$$

We regard a transition as a substantial distributional shift when $\Delta _ { k } > \tau$ and use a fixed threshold of $\tau = 0 . 1 0$ nats throughout all experiments. BUF is then defined as

$$
B U F = \frac { 1 } { K - 1 } \sum _ { k = 2 } ^ { K } \mathbb { I } \left( D _ { \mathrm { K L } } \left( P _ { k } \parallel P _ { k - 1 } \right) > \tau \right) .\tag{13}
$$

Here, $K$ denotes the number of reasoning steps, $P _ { k }$ and $P _ { k - 1 }$ denote the reference scorer’s nexttoken predictive distributions at consecutive steps, $D _ { \mathrm { K L } }$ is the Kullback–Leibler divergence, and τ is the divergence threshold. The indicator $\mathbb { I } [ \cdot ]$ equals 1 when the condition holds and 0 otherwise. For $K \geq 2$ , BUF is therefore the fraction of the $K - 1$ consecutive transitions that exceed the threshold and lies in [0, 1].

For numerical stability, predictive probabilities are clipped at $1 0 ^ { - 1 2 }$ and renormalized before computing the KL divergence. A higher BUF indicates more frequent distributional shifts under the reference scorer; it does not by itself establish belief revision, error correction, or improved reasoning.

Branching Complexity. Branching Complexity (BC) measures the prevalence of explicit alternative reasoning paths using a fixed set of branching-related lexical cues. We use

$$
\mathcal { K } = \{ o r , e i t h e r , m a y , p o s s i b l y , b u t , o n t h e o t h e r h a n d , c a s e \} .\tag{14}
$$

All cue matching is case-insensitive; single-word cues are matched at word boundaries and multiword cues are matched as exact phrases. BC is defined as

$$
B C ( r ) = \frac { 1 0 0 } { N _ { \mathrm { w o r d } } } \sum _ { k \in \mathcal { K } } \mathrm { C o u n t } ( k , r ) ,\tag{15}
$$

where $N _ { \mathrm { w o r d } }$ denotes the total number of words in response $r$ and $\operatorname { C o u n t } ( k , r )$ denotes the number of occurrences of cue k. BC can therefore be interpreted as the number of branching-related expressions per 100 generated words.

Metric Aggregation. For each model–prompt pair, we independently sample 10 responses. All metrics are first computed at the individual-response level and subsequently averaged over repeated generations and benchmark instances when reporting model-level statistics. For behavioral analysis, trajectories are first grouped according to their behavioral-resolution labels, after which the corresponding trajectory metrics are aggregated within each group. The same parameters are used throughout the evaluation.

## C DETAILED RESULTS OF MAIN EXPERIMENTS

Figure 3 in the main text summarizes accuracy retention across models, task domains, and hallucinated-augmentation types. This appendix complements that overview with detailed accuracy and reasoning-trajectory statistics under the truthful (T) and hallucinated (H) augmentation settings. In Figure 3, accuracy retention measures the proportion of truthful-setting accuracy preserved under hallucinated augmentation:

$$
{ \mathrm { R e t e n t i o n } } \left( \% \right) = 1 0 0 \times { \frac { { \mathrm { A c c } } _ { H } } { { \mathrm { A c c } } _ { T } } } .
$$

Here, $\operatorname { A c c } _ { T }$ and $\operatorname { A c c } _ { H }$ denote the accuracies of the same model under the two settings for the corresponding evaluation subset. A retention of 100% indicates unchanged accuracy; values below 100% indicate degradation, and values above 100% indicate improvement. For example, 80% retention means that accuracy under H is 80% of its value under T. Retention describes relative accuracy preservation, while the accuracy decreases reported in the tables are absolute differences measured in percentage points.

## C.1 DOMAIN-WISE RESULTS

Tables 5–8 report detailed results for Chemistry, Biomedicine, Physics, and Code Generation, respectively, allowing us to examine how the effects of hallucinated context vary across domains.

Hallucinated context induces broad but domain-dependent degradation. The decrease in final-answer accuracy is broadly consistent across domains. Averaged over the 18 evaluated models, the accuracy drop is 6.57 percentage points in Chemistry, 8.34 points in Biomedicine, 7.12 points in Physics, and 6.89 points in Code Generation. All models degrade under hallucinated augmentation in the three scientific domains, while 17 of 18 models degrade in Code Generation, with Qwen3-8B showing only a marginal 0.1-point increase. Thus, the overall performance degradation reported in the main text is not driven by a single domain. Biomedicine exhibits the largest average degradation, followed by Physics, whereas Chemistry shows the smallest absolute drop. Notably, Chemistry also has substantially lower baseline accuracy on average (23.0%) than Biomedicine (57.2%), Physics (56.0%), and Code Generation (48.4%), suggesting that its smaller absolute degradation should not be interpreted as greater robustness, since the available performance margin is considerably smaller.

The scaling-induced vulnerability persists across domains. The relationship between model scale and degradation observed in the aggregate results remains visible after domain-wise decomposition. Within Qwen2.5, scaling from 1.5B to 72B increases the accuracy drop from 2.0 to 6.6 points in Chemistry, 4.8 to 7.7 in Biomedicine, 3.7 to 6.6 in Physics, and 3.1 to 6.3 in Code Generation. The same pattern is more pronounced for Qwen3: from 8B to 235B-A22B, degradation increases from 1.9 to 9.7 points in Chemistry, 1.5 to 10.2 in Biomedicine, 0.3 to 9.1 in Physics, and from approximately no degradation to 8.9 points in Code Generation. Llama-3.1 exhibits a similar shift from 8B to 70B across all four domains. These results indicate that the scaling trend identified in the main analysis is not specific to a particular task family, but recurs across heterogeneous scientific and computational reasoning settings.

Reasoning trajectories respond differently across domains. Although the outcome-level degradation is broadly consistent, the associated trajectory changes are more domain dependent. Averaged across models, hallucinated augmentation produces longer trajectories in Biomedicine and Physics (+30.4 and +30.1 words, respectively), together with increases in both UI and BUF. In contrast, Chemistry exhibits shorter trajectories (−32.1 words) and lower average UI, BUF, and BC under hallucinated augmentation, while Code Generation also shows shorter trajectories and reduced UI and BUF, despite a modest increase in BC. Physics shows the clearest joint increase in reasoning activity, with OL, UI, BUF, and BC all increasing on average. These differences suggest that similar losses in final-answer accuracy can arise through distinct reasoning dynamics: hallucinated context may induce additional deliberation and belief revision in some domains, while suppressing or prematurely constraining the reasoning trajectory in others.

Table 5: Domain-wise performance and reasoning trajectory statistics on Chemistry under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T</td><td>H T</td><td>H</td><td>T</td><td>H</td><td>T</td><td></td><td>H</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>33.3</td><td> $2 2 . 2 _ { \downarrow 1 1 . 1 }$ </td><td>122.0</td><td>117.1</td><td>0.114</td><td>0.094</td><td>0.252</td><td>0.360</td><td>1.139</td><td>0.870</td></tr><tr><td>GPT-4o-mini</td><td>24.2</td><td> $2 1 . 0 _ { \downarrow 3 . 2 }$ </td><td>87.2</td><td>76.0</td><td>0.173</td><td>0.086</td><td>0.258</td><td>0.219</td><td>0.895</td><td>0.619</td></tr><tr><td>GPT-40</td><td>31.9</td><td> $2 0 . 8 _ { \downarrow 1 1 . 1 }$ </td><td>75.0</td><td>61.3</td><td>0.180</td><td>0.084</td><td>0.221</td><td>0.222</td><td>0.678</td><td>0.487</td></tr><tr><td>GPT-5.2</td><td>38.3</td><td> $2 0 . 9 _ { \downarrow 1 7 . 4 }$ </td><td>56.8</td><td>50.0</td><td>0.016</td><td>0.018</td><td>0.016</td><td>0.029</td><td>0.519</td><td>0.366</td></tr><tr><td>Gemini-2.0-Flash</td><td>31.5</td><td> $2 0 . 6 _ { \downarrow 1 0 . 9 }$ </td><td>95.1</td><td>90.7</td><td>0.107</td><td>0.081</td><td>0.151</td><td>0.196</td><td>0.762</td><td>0.615</td></tr><tr><td>Gemini-2.5-Pro</td><td>33.8</td><td> $2 1 . 6 _ { \downarrow 1 2 . 2 }$ </td><td>102.8</td><td>99.9</td><td>0.105</td><td>0.090</td><td>0.203</td><td>0.268</td><td>0.871</td><td>0.666</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>8.8</td><td> $6 . 8 _ { \downarrow 2 . 0 }$ </td><td>113.4</td><td>107.7</td><td>0.700</td><td>0.346</td><td>0.161</td><td>0.086</td><td>2.017</td><td>1.397</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>20.2</td><td> $1 5 . 6 _ { \downarrow 4 . 6 }$ </td><td>161.1</td><td>139.4</td><td>0.815</td><td>0.212</td><td>0.247</td><td>0.122</td><td>2.326</td><td>1.498</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>21.9</td><td> $1 6 . 6 _ { \downarrow 5 . 3 }$ </td><td>234.3</td><td>201.9</td><td>0.623</td><td>0.226</td><td>0.476</td><td>0.371</td><td>2.843</td><td>2.075</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>26.3</td><td> $1 9 . 7 _ { \downarrow 6 . 6 }$ </td><td>284.1</td><td>246.6</td><td>0.478</td><td>0.168</td><td>0.691</td><td>0.510</td><td>3.454</td><td>2.472</td></tr><tr><td>Qwen3-8B</td><td>22.2</td><td> $2 0 . 3 _ { \downarrow 1 . 9 }$ </td><td>224.8</td><td>170.4</td><td>0.458</td><td>0.083</td><td>0.837</td><td>0.339</td><td>2.678</td><td>2.078</td></tr><tr><td>Qwen3-32B</td><td>24.1</td><td> $1 8 . 3 _ { \downarrow 5 . 8 }$ </td><td>344.0</td><td>243.1</td><td>0.332</td><td>0.062</td><td>0.760</td><td>0.399</td><td>3.302</td><td>2.733</td></tr><tr><td>Qwen3-235B-A22B</td><td>32.8</td><td> $2 3 . 1 _ { \downarrow 9 . 7 }$ </td><td>423.8</td><td>350.0</td><td>0.156</td><td>0.043</td><td>0.856</td><td>0.720</td><td>4.246</td><td>3.454</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>13.0</td><td> $1 1 . 2 _ { \downarrow 1 . 8 }$ </td><td>256.4</td><td>215.5</td><td>1.088</td><td>0.570</td><td>0.381</td><td>0.280</td><td>2.918</td><td>1.837</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>22.3</td><td> $1 6 . 3 \scriptstyle \downarrow 6 . 0$ </td><td>346.8</td><td>295.9</td><td>1.293</td><td>0.798</td><td>0.839</td><td>0.562</td><td>3.576</td><td>2.524</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>11.5</td><td> $1 0 . 1 _ { \downarrow 1 . 4 }$ </td><td>188.7</td><td>162.6</td><td>0.441</td><td>0.257</td><td>0.623</td><td>0.160</td><td>2.202</td><td>1.243</td></tr><tr><td>GLM-4-9B-Chat</td><td>11.7</td><td> $6 . 8 _ { \downarrow 4 . 9 }$ </td><td>137.7</td><td>74.2</td><td>0.438</td><td>0.216</td><td>0.089</td><td>0.043</td><td>1.899</td><td>0.858</td></tr><tr><td>Mistral-7B-Instruct</td><td>6.9</td><td> $4 . 5 _ { \downarrow 2 . 4 }$ </td><td>141.7</td><td>114.9</td><td>0.333</td><td>0.175</td><td>0.469</td><td>0.421</td><td>1.866</td><td>1.125</td></tr></table>

Table 6: Domain-wise performance and reasoning trajectory statistics on Biomedicine under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T H</td><td>T</td><td>H</td><td>T</td><td>H</td><td>T</td><td>H</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>70.0</td><td> $6 0 . 2 _ { \downarrow 9 . 8 }$ </td><td>100.9</td><td>121.2 0.050</td><td>0.105</td><td>0.197</td><td>0.404</td><td>1.558</td><td></td><td>1.438</td></tr><tr><td>GPT-4o-mini</td><td>68.3</td><td> $5 8 . 4 _ { \downarrow 9 . 9 }$ </td><td>69.2</td><td>72.4</td><td>0.062</td><td>0.080</td><td>0.123</td><td>0.205</td><td>1.018</td><td>0.980</td></tr><tr><td>GPT-40</td><td>66.7</td><td> $5 6 . 0 _ { \downarrow 1 0 . 7 }$ </td><td>57.4</td><td>60.9</td><td>0.041</td><td>0.067</td><td>0.119</td><td>0.249</td><td>1.004</td><td>0.854</td></tr><tr><td>GPT-5.2</td><td>72.5</td><td> $5 7 . 3 _ { \downarrow 1 5 . 2 }$ </td><td>42.6</td><td>48.3</td><td>0.010</td><td>0.023</td><td>0.011</td><td>0.035</td><td>0.749</td><td>0.686</td></tr><tr><td>Gemini-2.0-Flash</td><td>69.2</td><td> $5 7 . 1 _ { \downarrow 1 2 . 1 }$ </td><td>74.4</td><td>89.4</td><td>0.049</td><td>0.088</td><td>0.097</td><td>0.217</td><td>1.045</td><td>1.066</td></tr><tr><td>Gemini-2.5-Pro</td><td>71.4</td><td> $5 8 . 6 _ { \downarrow 1 2 . 8 }$ </td><td>80.4</td><td>98.5</td><td>0.048</td><td>0.098</td><td>0.131</td><td>0.297</td><td>1.195</td><td>1.155</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>32.1</td><td> $2 7 . 3 _ { \downarrow 4 . 8 }$ </td><td>105.2</td><td>118.9</td><td>0.197</td><td>0.240</td><td>0.084</td><td>0.089</td><td>2.708</td><td>2.264</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>55.4</td><td> $4 8 . 6 _ { \downarrow 6 . 8 }$ </td><td>143.2</td><td>152.4</td><td>0.217</td><td>0.188</td><td>0.147</td><td>0.149</td><td>2.972</td><td>2.457</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>57.9</td><td> $5 0 . 5 _ { \downarrow 7 . 4 }$ </td><td>210.5</td><td>220.2</td><td>0.168</td><td>0.170</td><td>0.270</td><td>0.427</td><td>3.750</td><td>3.358</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>63.5</td><td> $5 5 . 8 _ { \downarrow 7 . 7 }$ </td><td>255.2</td><td>269.1</td><td>0.129</td><td>0.127</td><td>0.392</td><td>0.586</td><td>4.556</td><td>4.000</td></tr><tr><td>Qwen3-8B</td><td>56.2</td><td> $5 4 . 7 _ { \downarrow 1 . 5 }$ </td><td>135.9</td><td>289.6</td><td>0.128</td><td>0.089</td><td>0.440</td><td>0.465</td><td>3.370</td><td>4.525</td></tr><tr><td>Qwen3-32B</td><td>58.9</td><td> $5 1 . 5 _ { \downarrow 7 . 4 }$ </td><td>236.5</td><td>330.1</td><td>0.128</td><td>0.099</td><td>0.537</td><td>0.586</td><td>3.643</td><td>4.747</td></tr><tr><td>Qwen3-235B-A22B</td><td>68.8</td><td> $5 8 . 6 _ { \downarrow 1 0 . 2 }$ </td><td>291.3</td><td>475.3</td><td>0.060</td><td>0.069</td><td>0.817</td><td>0.826</td><td>4.683</td><td>5.999</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>45.0</td><td> $4 2 . 0 _ { \downarrow 3 . 0 }$ </td><td>190.2</td><td>197.2</td><td>0.230</td><td>0.325</td><td>0.484</td><td>0.207</td><td>3.323</td><td>2.625</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>61.0</td><td> $5 2 . 8 _ { \downarrow 8 . 2 }$ </td><td>257.8</td><td>271.8</td><td>0.237</td><td>0.374</td><td>0.526</td><td>0.354</td><td>3.828</td><td>3.361</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>41.4</td><td> $3 9 . 0 _ { \downarrow 2 . 4 }$ </td><td>146.7</td><td>144.0</td><td>0.114</td><td>0.136</td><td>0.303</td><td>0.127</td><td>2.751</td><td>1.883</td></tr><tr><td>GLM-4-9B-Chat</td><td>42.2</td><td> $2 9 . 7 _ { \downarrow 1 2 . 5 }$ </td><td>108.1</td><td>80.5</td><td>0.108</td><td>0.259</td><td>0.042</td><td>0.042</td><td>2.344</td><td>1.408</td></tr><tr><td>Mistral-7B-Instruct</td><td>29.3</td><td> $2 1 . 5 _ { \downarrow 7 . 8 }$ </td><td>111.3</td><td>124.8</td><td>0.082</td><td>0.115</td><td>0.218</td><td>0.406</td><td>2.303</td><td>1.847</td></tr></table>

Table 7: Domain-wise performance and reasoning trajectory statistics on Physics under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T H</td><td>T</td><td>H</td><td>T</td><td>H</td><td>T</td><td>H</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>69.2</td><td> $6 3 . 0 _ { \downarrow 6 . 2 }$ </td><td>89.4</td><td>104.0</td><td>0.010</td><td>0.024</td><td>0.101</td><td>0.284</td><td>0.964</td><td>1.067</td></tr><tr><td>GPT-4o-mini</td><td>74.2</td><td> $6 5 . 4 _ { \downarrow 8 . 8 }$ </td><td>71.5</td><td>73.0</td><td>0.020</td><td>0.024</td><td>0.095</td><td>0.172</td><td>0.583</td><td>0.606</td></tr><tr><td>GPT-40</td><td>56.7</td><td> $4 9 . 9 _ { \downarrow 6 . 8 }$ </td><td>58.2</td><td>63.1</td><td>0.006</td><td>0.019</td><td>0.081</td><td>0.217</td><td>0.505</td><td>0.563</td></tr><tr><td>GPT-5.2</td><td>74.2</td><td> $5 5 . 8 _ { \downarrow 1 8 . 4 }$ </td><td>34.4</td><td>38.2</td><td>0.001</td><td>0.008</td><td>0.015</td><td>0.019</td><td>0.471</td><td>0.464</td></tr><tr><td>Gemini-2.0-Flash</td><td>68.7</td><td> $5 7 . 8 _ { \downarrow 1 0 . 9 }$ </td><td>69.4</td><td>82.5</td><td>0.006</td><td>0.024</td><td>0.075</td><td>0.157</td><td>0.575</td><td>0.695</td></tr><tr><td>Gemini-2.5-Pro</td><td>70.9</td><td> $5 9 . 2 _ { \downarrow 1 1 . 7 }$ </td><td>75.1</td><td>90.9</td><td>0.006</td><td>0.027</td><td>0.101</td><td>0.215</td><td>0.657</td><td>0.753</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>30.5</td><td> $2 6 . 8 _ { \downarrow 3 . 7 }$ </td><td>103.0</td><td>120.3</td><td>0.073</td><td>0.105</td><td>0.073</td><td>0.067</td><td>1.752</td><td>1.643</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>53.5</td><td> $4 8 . 1 _ { \downarrow 5 . 4 }$ </td><td>131.2</td><td>145.0</td><td>0.076</td><td>0.074</td><td>0.113</td><td>0.111</td><td>2.142</td><td>1.928</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>56.1</td><td> $4 9 . 9 _ { \downarrow 6 . 2 }$ </td><td>199.1</td><td>216.0</td><td>0.066</td><td>0.077</td><td>0.222</td><td>0.320</td><td>2.633</td><td>2.588</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>61.8</td><td> $5 5 . 2 _ { \downarrow 6 . 6 }$ </td><td>241.4</td><td>263.9</td><td>0.051</td><td>0.057</td><td>0.323</td><td>0.440</td><td>3.199</td><td>3.084</td></tr><tr><td>Qwen3-8B</td><td>54.7</td><td> $5 4 . 4 _ { \downarrow 0 . 3 }$ </td><td>116.8</td><td>262.9</td><td>0.053</td><td>0.044</td><td>0.474</td><td>0.414</td><td>2.053</td><td>3.708</td></tr><tr><td>Qwen3-32B</td><td>57.4</td><td> $5 1 . 2 _ { \downarrow 6 . 2 }$ </td><td>188.5</td><td>269.4</td><td>0.061</td><td>0.036</td><td>0.525</td><td>0.460</td><td>2.445</td><td>3.907</td></tr><tr><td>Qwen3-235B-A22B</td><td>67.4</td><td> $5 8 . 3 _ { \downarrow 9 . 1 }$ </td><td>232.2</td><td>387.8</td><td>0.029</td><td>0.025</td><td>0.799</td><td>0.733</td><td>3.143</td><td>4.938</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>43.7</td><td> $4 1 . 9 _ { \downarrow 1 . 8 }$ </td><td>164.6</td><td>178.9</td><td>0.080</td><td>0.136</td><td>0.415</td><td>0.171</td><td>2.887</td><td>2.351</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>59.7</td><td> $5 2 . 7 _ { \downarrow 7 . 0 }$ </td><td>223.9</td><td>247.6</td><td>0.090</td><td>0.176</td><td>0.467</td><td>0.306</td><td>3.794</td><td>3.321</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>40.1</td><td> $3 9 . 0 _ { \downarrow 1 . 1 }$ </td><td>143.8</td><td>146.0</td><td>0.031</td><td>0.046</td><td>0.255</td><td>0.109</td><td>1.880</td><td>1.300</td></tr><tr><td>GLM-4-9B-Chat</td><td>41.7</td><td> $3 0 . 3 _ { \downarrow 1 1 . 4 }$ </td><td>99.9</td><td>77.1</td><td>0.041</td><td>0.070</td><td>0.038</td><td>0.035</td><td>1.724</td><td>1.140</td></tr><tr><td>Mistral-7B-Instruct</td><td>27.8</td><td> $2 1 . 2 _ { \downarrow 6 . 6 }$ </td><td>102.8</td><td>119.5</td><td>0.031</td><td>0.037</td><td>0.198</td><td>0.342</td><td>1.694</td><td>1.495</td></tr></table>

Table 8: Domain-wise performance and reasoning trajectory statistics on Code Generation under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T H</td><td>T</td><td>H</td><td>T</td><td>H</td><td>T</td><td>H</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>57.5</td><td> $4 7 . 6 _ { \downarrow 9 . 9 }$ </td><td>88.4</td><td>102.3</td><td>0.041</td><td>0.029</td><td>0.039</td><td>0.087</td><td>0.818</td><td>0.946</td></tr><tr><td>GPT-4o-mini</td><td>55.3</td><td> $4 8 . 5 _ { \downarrow 6 . 8 }$ </td><td>54.0</td><td>55.6</td><td>0.026</td><td>0.011</td><td>0.027</td><td>0.032</td><td>0.464</td><td>0.541</td></tr><tr><td>GPT-40</td><td>52.0</td><td> $4 2 . 8 _ { \downarrow 9 . 2 }$ </td><td>47.2</td><td>49.5</td><td>0.009</td><td>0.010</td><td>0.014</td><td>0.030</td><td>0.407</td><td>0.421</td></tr><tr><td>GPT-5.2</td><td>61.4</td><td> $4 4 . 8 _ { \downarrow 1 6 . 6 }$ </td><td>37.3</td><td>40.6</td><td>0.001</td><td>0.010</td><td>0.011</td><td>0.008</td><td>0.570</td><td>0.632</td></tr><tr><td>Gemini-2.0-Flash</td><td>55.3</td><td> $4 4 . 0 _ { \downarrow 1 1 . 3 }$ </td><td>63.9</td><td>74.5</td><td>0.017</td><td>0.022</td><td>0.035</td><td>0.036</td><td>0.570</td><td>0.698</td></tr><tr><td>Gemini-2.5-Pro</td><td>57.9</td><td> $4 5 . 5 _ { \downarrow 1 2 . 4 }$ </td><td>69.0</td><td>82.0</td><td>0.016</td><td>0.024</td><td>0.048</td><td>0.049</td><td>0.652</td><td>0.756</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>27.4</td><td> $2 4 . 3 _ { \pm 3 . 1 }$ </td><td>145.8</td><td>134.5</td><td>0.087</td><td>0.073</td><td>0.038</td><td>0.021</td><td>1.368</td><td>1.504</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>49.7</td><td> $4 4 . 7 _ { \downarrow 5 . 0 }$ </td><td>182.1</td><td>165.0</td><td>0.139</td><td>0.068</td><td>0.072</td><td>0.041</td><td>1.491</td><td>1.673</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>52.3</td><td> $4 6 . 6 _ { \downarrow 5 . 7 }$ </td><td>277.8</td><td>243.8</td><td>0.098</td><td>0.062</td><td>0.150</td><td>0.116</td><td>1.900</td><td>2.288</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>58.2</td><td> $5 1 . 9 _ { \downarrow 6 . 3 }$ </td><td>336.9</td><td>297.8</td><td>0.075</td><td>0.046</td><td>0.217</td><td>0.159</td><td>2.308</td><td>2.726</td></tr><tr><td>Qwen3-8B</td><td>51.9</td><td> $5 2 . 0 _ { \uparrow \mathrm { d } }$ </td><td>465.6</td><td>444.2</td><td>0.231</td><td>0.073</td><td>0.548</td><td>0.225</td><td>3.412</td><td>3.866</td></tr><tr><td>Qwen3-32B</td><td>54.6</td><td> $4 8 . 8 _ { \downarrow 5 . 8 }$ </td><td>440.2</td><td>378.8</td><td>0.207</td><td>0.057</td><td>0.598</td><td>0.262</td><td>3.599</td><td>4.112</td></tr><tr><td>Qwen3-235B-A22B</td><td>64.9</td><td> $5 6 . 0 _ { \downarrow 8 . 9 }$ </td><td>542.3</td><td>545.4</td><td>0.097</td><td>0.040</td><td>0.910</td><td>0.672</td><td>4.627</td><td>5.197</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>35.3</td><td> $3 4 . 1 _ { \pm 1 . 2 }$ </td><td>234.9</td><td>232.1</td><td>0.062</td><td>0.064</td><td>0.171</td><td>0.055</td><td>1.706</td><td>1.632</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>51.0</td><td> $4 4 . 4 _ { \downarrow 6 . 6 }$ </td><td>317.7</td><td>318.8</td><td>0.038</td><td>0.056</td><td>0.209</td><td>0.101</td><td>2.000</td><td>2.035</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>32.0</td><td> $3 1 . 4 _ { \downarrow 0 . 6 }$ </td><td>193.2</td><td>183.6</td><td>0.028</td><td>0.024</td><td>0.064</td><td>0.024</td><td>1.349</td><td>1.031</td></tr><tr><td>GLM-4-9B-Chat</td><td>33.9</td><td> $2 4 . 1 _ { \downarrow 9 . 8 }$ </td><td>170.7</td><td>99.7</td><td>0.082</td><td>0.070</td><td>0.027</td><td>0.014</td><td>1.515</td><td>0.964</td></tr><tr><td>Mistral-7B-Instruct</td><td>20.3</td><td> $1 5 . 3 _ { \downarrow 5 . 0 }$ </td><td>175.7</td><td>154.5</td><td>0.062</td><td>0.045</td><td>0.143</td><td>0.133</td><td>1.488</td><td>1.265</td></tr></table>

## C.2 RESULTS ACROSS HALLUCINATED AUGMENTATION TYPES

Tables 9–11 report detailed results for Rule Contradiction (RC), State Distortion (SD), and Pseudoscientific Entanglement (PE). The comparison covers the three scientific domains, where all three types are represented. Code Generation is excluded because its hallucinated augmentations consist exclusively of State Distortion.

State Distortion induces the strongest performance degradation. Across the 18 models, SD causes the largest average accuracy drop (9.4 percentage points), followed by RC (7.4 points), whereas PE has a substantially weaker effect (1.8 points). This pattern suggests that instance-specific misinformation is particularly difficult to resolve: unlike direct violations of established rules, SD preserves a plausible general statement while making it inconsistent with the conditions of the current problem. In contrast, the comparatively small degradation under PE indicates that outdated or pseudoscientific premises are more readily distinguished from established domain knowledge.

Different augmentation types also induce distinct trajectory changes. The trajectory metrics reveal a less uniform pattern than final-answer accuracy. SD is associated with both longer responses and increased branching complexity on average, suggesting greater exploration of competing reasoning paths under instance-level conflict. RC produces a smaller increase in output length and lower branching complexity, while PE often elicits longer responses without a commensurate loss in accuracy. Together, these results show that the effect of hallucinated context depends not only on whether the information is false, but also on the form of semantic conflict it introduces.

Table 9: Performance and reasoning trajectory statistics for Rule Contradiction under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T H</td><td>T</td><td>H</td><td>T</td><td>H</td><td>T</td><td>H</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>57.7</td><td> $4 8 . 5 _ { \downarrow 9 . 2 }$ </td><td>105.4</td><td>113.8</td><td>0.054</td><td>0.066</td><td>0.175</td><td>0.336</td><td>1.220</td><td>1.099</td></tr><tr><td>GPT-4o-mini</td><td>55.7</td><td> $4 8 . 5 _ { \downarrow 7 . 2 }$ </td><td>77.2</td><td>73.3</td><td>0.079</td><td>0.056</td><td>0.151</td><td>0.191</td><td>0.832</td><td>0.718</td></tr><tr><td>GPT-40</td><td>52.1</td><td> $4 2 . 2 _ { \downarrow 9 . 9 }$ </td><td>64.0</td><td>61.3</td><td>0.070</td><td>0.050</td><td>0.134</td><td>0.220</td><td>0.729</td><td>0.620</td></tr><tr><td>GPT-5.2</td><td>62.1</td><td> $4 4 . 4 _ { \downarrow 1 7 . 7 }$ </td><td>45.4</td><td>45.3</td><td>0.008</td><td>0.015</td><td>0.013</td><td>0.026</td><td>0.580</td><td>0.494</td></tr><tr><td>Gemini-2.0-Flash</td><td>56.6</td><td> $4 5 . 4 _ { \downarrow 1 1 . 2 }$ </td><td>79.7</td><td>87.0</td><td>0.050</td><td>0.057</td><td>0.103</td><td>0.182</td><td>0.794</td><td>0.774</td></tr><tr><td>Gemini-2.5-Pro</td><td>58.9</td><td> $4 6 . 7 _ { \downarrow 1 2 . 2 }$ </td><td>86.3</td><td>95.9</td><td>0.049</td><td>0.063</td><td>0.139</td><td>0.250</td><td>0.907</td><td>0.838</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>23.9</td><td> $2 0 . 4 _ { \downarrow 3 . 5 }$ </td><td>104.9</td><td>113.8</td><td>0.301</td><td>0.204</td><td>0.101</td><td>0.077</td><td>2.159</td><td>1.727</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>43.2</td><td> $3 7 . 6 _ { \downarrow 5 . 6 }$ </td><td>143.2</td><td>143.5</td><td>0.344</td><td>0.140</td><td>0.161</td><td>0.123</td><td>2.480</td><td>1.916</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>45.4</td><td> $3 9 . 2 _ { \downarrow 6 . 2 }$ </td><td>215.0</td><td>211.5</td><td>0.266</td><td>0.139</td><td>0.308</td><td>0.358</td><td>3.075</td><td>2.612</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>50.7</td><td> $4 3 . 8 _ { \downarrow 6 . 9 }$ </td><td>260.6</td><td>258.4</td><td>0.204</td><td>0.104</td><td>0.447</td><td>0.492</td><td>3.736</td><td>3.112</td></tr><tr><td>Qwen3-8B</td><td>44.5</td><td> $4 3 . 3 _ { \pm 1 . 2 }$ </td><td>159.4</td><td>239.6</td><td>0.198</td><td>0.064</td><td>0.557</td><td>0.390</td><td>2.700</td><td>3.358</td></tr><tr><td>Qwen3-32B</td><td>47.0</td><td> $4 0 . 5 _ { \downarrow 6 . 5 }$ </td><td>256.7</td><td>279.2</td><td>0.162</td><td>0.058</td><td>0.579</td><td>0.463</td><td>3.130</td><td>3.708</td></tr><tr><td>Qwen3-235B-A22B</td><td>56.5</td><td> $4 6 . 9 _ { \downarrow 9 . 6 }$ </td><td>316.2</td><td>402.0</td><td>0.076</td><td>0.040</td><td>0.817</td><td>0.754</td><td>4.024</td><td>4.686</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>34.0</td><td> $3 1 . 8 _ { \downarrow 2 . 2 }$ </td><td>203.2</td><td>193.0</td><td>0.434</td><td>0.304</td><td>0.407</td><td>0.211</td><td>3.042</td><td>2.219</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>47.8</td><td> $4 0 . 8 _ { \downarrow 7 . 0 }$ </td><td>276.6</td><td>270.2</td><td>0.503</td><td>0.398</td><td>0.583</td><td>0.392</td><td>3.732</td><td>2.998</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>31.1</td><td> $2 9 . 5 _ { \downarrow 1 . 6 }$ </td><td>158.7</td><td>149.6</td><td>0.182</td><td>0.130</td><td>0.376</td><td>0.127</td><td>2.278</td><td>1.441</td></tr><tr><td>GLM-4-9B-Chat</td><td>32.0</td><td> $2 2 . 4 _ { \downarrow 9 . 6 }$ </td><td>115.4</td><td>76.8</td><td>0.182</td><td>0.161</td><td>0.054</td><td>0.038</td><td>1.989</td><td>1.109</td></tr><tr><td>Mistral-7B-Instruct</td><td>21.4</td><td> $1 5 . 8 _ { \downarrow 5 . 6 }$ </td><td>118.8</td><td>119.0</td><td>0.139</td><td>0.080</td><td>0.282</td><td>0.375</td><td>1.954</td><td>1.455</td></tr></table>

Table 10: Performance and reasoning trajectory statistics for State Distortion under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T H</td><td>T</td><td>H</td><td>T</td><td>H</td><td>T</td><td>H</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>43.8</td><td> $3 2 . 6 _ { \downarrow 1 1 . 2 }$ </td><td>111.9</td><td>125.1</td><td>0.108</td><td>0.172</td><td>0.298</td><td>0.558</td><td>1.179</td><td>1.336</td></tr><tr><td>GPT-4o-mini</td><td>40.1</td><td> $3 0 . 0 _ { \downarrow 1 0 . 1 }$ </td><td>81.3</td><td>83.0</td><td>0.160</td><td>0.146</td><td>0.258</td><td>0.318</td><td>0.804</td><td>0.873</td></tr><tr><td>GPT-40</td><td>41.9</td><td> $3 0 . 1 _ { \downarrow 1 1 . 8 }$ </td><td>72.0</td><td>68.7</td><td>0.142</td><td>0.131</td><td>0.228</td><td>0.367</td><td>0.704</td><td>0.753</td></tr><tr><td>GPT-5.2</td><td>44.6</td><td> $3 0 . 4 _ { \downarrow 1 4 . 2 }$ </td><td>46.4</td><td>52.2</td><td>0.016</td><td>0.038</td><td>0.022</td><td>0.044</td><td>0.561</td><td>0.600</td></tr><tr><td>Gemini-2.0-Flash</td><td>42.5</td><td> $2 7 . 1 _ { \downarrow 1 5 . 4 }$ </td><td>84.9</td><td>98.0</td><td>0.102</td><td>0.149</td><td>0.175</td><td>0.303</td><td>0.767</td><td>0.941</td></tr><tr><td>Gemini-2.5-Pro</td><td>44.2</td><td> $2 7 . 5 _ { \downarrow 1 6 . 7 }$ </td><td>91.9</td><td>108.0</td><td>0.100</td><td>0.166</td><td>0.236</td><td>0.416</td><td>0.877</td><td>1.019</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>17.9</td><td> $1 3 . 1 _ { \downarrow 4 . 8 }$ </td><td>120.9</td><td>128.8</td><td>0.609</td><td>0.532</td><td>0.172</td><td>0.129</td><td>2.086</td><td>2.100</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>32.4</td><td> $2 4 . 7 _ { \downarrow 7 . 7 }$ </td><td>159.1</td><td>164.3</td><td>0.694</td><td>0.365</td><td>0.275</td><td>0.204</td><td>2.396</td><td>2.329</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>34.1</td><td> $2 5 . 5 _ { \downarrow 8 . 6 }$ </td><td>229.0</td><td>238.2</td><td>0.538</td><td>0.364</td><td>0.524</td><td>0.596</td><td>2.972</td><td>3.175</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>38.0</td><td> $2 8 . 5 _ { \downarrow 9 . 5 }$ </td><td>277.7</td><td>291.0</td><td>0.413</td><td>0.272</td><td>0.762</td><td>0.819</td><td>3.611</td><td>3.783</td></tr><tr><td>Qwen3-8B</td><td>33.4</td><td> $3 1 . 7 _ { \downarrow 1 . 7 }$ </td><td>169.8</td><td>269.8</td><td>0.401</td><td>0.166</td><td>0.949</td><td>0.649</td><td>2.609</td><td>4.082</td></tr><tr><td>Qwen3-32B</td><td>35.2</td><td> $2 6 . 4 _ { \downarrow 8 . 8 }$ </td><td>273.5</td><td>314.5</td><td>0.327</td><td>0.152</td><td>0.987</td><td>0.770</td><td>3.025</td><td>4.508</td></tr><tr><td>Qwen3-235B-A22B</td><td>42.4</td><td> $2 9 . 2 _ { \downarrow 1 3 . 2 }$ </td><td>336.9</td><td>452.8</td><td>0.154</td><td>0.105</td><td>0.945</td><td>0.903</td><td>3.888</td><td>5.697</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>25.5</td><td> $2 2 . 5 _ { \downarrow 3 . 0 }$ </td><td>248.4</td><td>248.6</td><td>0.876</td><td>0.795</td><td>0.693</td><td>0.351</td><td>2.940</td><td>2.697</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>35.9</td><td> $2 6 . 2 _ { \downarrow 9 . 7 }$ </td><td>294.7</td><td>304.4</td><td>1.016</td><td>1.040</td><td>0.993</td><td>0.651</td><td>3.607</td><td>3.644</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>23.3</td><td> $2 1 . 1 _ { \downarrow 2 . 2 }$ </td><td>173.8</td><td>167.0</td><td>0.368</td><td>0.339</td><td>0.640</td><td>0.211</td><td>2.201</td><td>1.752</td></tr><tr><td>GLM-4-9B-Chat</td><td>24.0</td><td> $1 0 . 8 _ { \downarrow 1 3 . 2 }$ </td><td>123.0</td><td>86.5</td><td>0.368</td><td>0.421</td><td>0.092</td><td>0.064</td><td>1.922</td><td>1.348</td></tr><tr><td>Mistral-7B-Instruct</td><td>16.1</td><td> $8 . 4 _ { \downarrow 7 . 7 }$ </td><td>126.5</td><td>134.1</td><td>0.280</td><td>0.425</td><td>0.480</td><td>0.623</td><td>1.889</td><td>1.769</td></tr></table>

Table 11: Performance and reasoning trajectory statistics for Pseudoscientific Entanglement under the truthful augmentation (T) and hallucinated augmentation (H) settings.
<table><tr><td rowspan="2">Model</td><td colspan="2">Acc (%)</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>T</td><td>H</td><td>T H</td><td>T</td><td>H</td><td>T</td><td>H</td><td>T</td><td>H</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>87.7</td><td> $8 5 . 2 _ { \downarrow 2 . 5 }$ </td><td>83.8</td><td>95.2</td><td>0.041</td><td>0.065</td><td>0.129</td><td>0.207</td><td>1.320</td><td>1.304</td></tr><tr><td>GPT-40-mini</td><td>89.3</td><td> $8 5 . 2 _ { \downarrow 4 . 1 }$ </td><td>62.9</td><td>65.0</td><td>0.061</td><td>0.055</td><td>0.112</td><td>0.118</td><td>0.900</td><td>0.852</td></tr><tr><td>GPT-40</td><td>72.1</td><td> $7 3 . 0 _ { \uparrow \mathrm { ( } }$ </td><td>56.1</td><td>58.2</td><td>0.054</td><td>0.050</td><td>0.099</td><td>0.136</td><td>0.789</td><td>0.735</td></tr><tr><td>GPT-5.2</td><td>91.0</td><td> $8 6 . 9 _ { \downarrow 4 . 1 }$ </td><td>32.4</td><td>36.9</td><td>0.006</td><td>0.014</td><td>0.010</td><td>0.016</td><td>0.628</td><td>0.586</td></tr><tr><td>Gemini-2.0-Flash</td><td>84.6</td><td> $8 1 . 8 _ { \downarrow 2 . 8 }$ </td><td>63.8</td><td>75.8</td><td>0.039</td><td>0.056</td><td>0.076</td><td>0.113</td><td>0.859</td><td>0.918</td></tr><tr><td>Gemini-2.5-Pro</td><td>87.9</td><td> $8 4 . 8 _ { \downarrow 3 . 1 }$ </td><td>69.0</td><td>83.6</td><td>0.038</td><td>0.063</td><td>0.102</td><td>0.154</td><td>0.982</td><td>0.994</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>35.7</td><td> $3 4 . 8 _ { \downarrow 0 . 9 }$ </td><td>106.0</td><td>130.3</td><td>0.232</td><td>0.202</td><td>0.074</td><td>0.048</td><td>2.336</td><td>2.050</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>64.5</td><td> $6 3 . 1 _ { \downarrow 1 . 4 }$ </td><td>132.7</td><td>155.8</td><td>0.265</td><td>0.138</td><td>0.119</td><td>0.076</td><td>2.683</td><td>2.273</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>67.9</td><td> $6 6 . 3 _ { \downarrow 1 . 6 }$ </td><td>171.9</td><td>184.4</td><td>0.205</td><td>0.138</td><td>0.227</td><td>0.221</td><td>3.327</td><td>3.099</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>75.7</td><td> $7 4 . 0 _ { \downarrow 1 . 7 }$ </td><td>208.4</td><td>225.2</td><td>0.157</td><td>0.103</td><td>0.330</td><td>0.304</td><td>4.042</td><td>3.692</td></tr><tr><td>Qwen3-8B</td><td>66.5</td><td> $6 6 . 2 _ { \downarrow 0 . 3 }$ </td><td>127.5</td><td>208.8</td><td>0.153</td><td>0.063</td><td>0.411</td><td>0.241</td><td>2.921</td><td>3.984</td></tr><tr><td>Qwen3-32B</td><td>70.2</td><td> $6 8 . 5 _ { \downarrow 1 . 7 }$ </td><td>205.2</td><td>243.4</td><td>0.125</td><td>0.058</td><td>0.427</td><td>0.286</td><td>3.386</td><td>4.400</td></tr><tr><td>Qwen3-235B-A22B</td><td>84.4</td><td> $8 2 . 0 _ { \downarrow 2 . 4 }$ </td><td>252.8</td><td>350.5</td><td>0.059</td><td>0.040</td><td>0.714</td><td>0.578</td><td>4.353</td><td>5.561</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>50.8</td><td> $5 0 . 2 _ { \downarrow 0 . 6 }$ </td><td>167.1</td><td>185.2</td><td>0.334</td><td>0.301</td><td>0.300</td><td>0.130</td><td>3.292</td><td>2.633</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>71.5</td><td> $6 9 . 7 _ { \downarrow 1 . 8 }$ </td><td>221.1</td><td>235.6</td><td>0.388</td><td>0.394</td><td>0.430</td><td>0.242</td><td>4.038</td><td>3.557</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>46.4</td><td> $4 6 . 0 _ { \downarrow 0 . 4 }$ </td><td>136.6</td><td>144.7</td><td>0.140</td><td>0.128</td><td>0.277</td><td>0.078</td><td>2.464</td><td>1.710</td></tr><tr><td>GLM-4-9B-Chat</td><td>47.7</td><td> $4 5 . 3 _ { \downarrow 2 . 4 }$ </td><td>92.3</td><td>67.0</td><td>0.149</td><td>0.159</td><td>0.040</td><td>0.024</td><td>2.152</td><td>1.316</td></tr><tr><td>Mistral-7B-Instruct</td><td>32.0</td><td> $3 0 . 5 _ { \downarrow 1 . 5 }$ </td><td>95.0</td><td>103.8</td><td>0.107</td><td>0.115</td><td>0.208</td><td>0.231</td><td>2.114</td><td>1.726</td></tr></table>

## D DETAILED ANALYSIS OF BEHAVIORAL RESOLUTION

## D.1 RESOLUTION DEFINITION

Building on these settings and the post-hallucination reasoning framework introduced by HIVE (He et al., 2026), we characterize how an individual response operationally resolves a hallucinated premise once it is already present in the reasoning context. Our taxonomy is response-level and is defined independently of final-answer correctness.

Behavioral Resolution. Let h denote the hallucinated premise and r the generated reasoning trajectory. We distinguish two properties of the response.

Premise acceptance $A ( r , h )$ indicates that the response treats h as valid and materially uses it as evidence, a constraint, or an intermediate premise in deriving its answer.

Premise correction $C ( \boldsymbol r , h )$ indicates that the response explicitly identifies h as false, inconsistent, or inapplicable, recovers a corrected fact or condition, and uses this correction in the subsequent reasoning. Merely expressing uncertainty about h, or noting a conflict while ultimately continuing to reason from h, does not constitute correction.

Using these criteria, we define the behavioral mode of r as

$$
B ( r ) = \left\{ \begin{array} { l l } { { \mathrm { H E U R I S T I C  C O R R E C T I O N } } , } \\ { { \mathrm { H A L L U C I N A T I O N } } { \mathrm { C O M P L I A N C E } } , } \\ { { \mathrm { H A L L U C I N A T I O N } } { \mathrm { A V O I D A N C E } } , } \end{array} \right.
$$

$$
\begin{array} { l } { { C ( r , h ) = 1 , } } \\ { { C ( r , h ) = 0 \wedge A ( r , h ) = 1 , } } \\ { { C ( r , h ) = 0 \wedge A ( r , h ) = 0 . } } \end{array}\tag{16}
$$

The three modes are mutually exclusive and describe the operative stance of the response toward the hallucinated premise.

Hallucination Compliance. A response is classified as Hallucination Compliance when the hallucinated premise remains active in the model’s reasoning and is treated as valid information. Compliance requires more than simply repeating the premise: the response must rely on it to constrain, justify, or redirect the reasoning trajectory. A model may acknowledge that the premise differs from its prior knowledge yet still be classified as compliant if it ultimately accepts the injected premise $( \mathrm { e . g . }$ , “assuming the premise is true”) and bases its conclusion on it. Importantly, compliance does not imply that the final answer must be incorrect; the model may occasionally reach the correct answer despite reasoning from the hallucinated augmentation.

Hallucination Avoidance. A response is classified as Hallucination Avoidance when it reaches its conclusion without materially relying on the hallucinated premise and without explicitly correcting it. The model effectively routes around the erroneous information by solving the original task through an independent reasoning path. Avoidance therefore differs from correction: a response that simply ignores the premise, even if it produces the correct answer, provides no evidence that the model recognized the premise as false.

Heuristic Correction. A response is classified as Heuristic Correction when it explicitly detects a conflict involving the hallucinated premise, rejects or revises the erroneous statement, and substitutes a corrected fact, condition, or mechanism that governs the subsequent reasoning. Correction therefore requires both error recognition and reasoning revision. A response that notes that the premise is questionable but nevertheless continues to reason according to it is not considered correction. Similarly, correction does not guarantee a correct final answer: the model may correctly reject the injected premise but introduce a different error during the subsequent reasoning.

Mixed Responses. Some trajectories exhibit more than one local behavior. We classify such cases according to the premise that ultimately governs the downstream reasoning. If a model initially chal lenges h but subsequently adopts it and derives its answer from it, the response is classified as HAL-LUCINATION COMPLIANCE. If the model initially follows h but later rejects it and reconstructs the subsequent reasoning using a corrected premise, it is classified as HEURISTIC CORRECTION. Responses that neither rely on nor explicitly revise h are classified as HALLUCINATION AVOIDANCE. This decision rule prevents superficial mentions of the hallucinated premise from determining the behavioral label.

Behavioral Label Annotation. We assign responses to the three behavioral categories through a human–LLM annotation procedure. We first manually label a subset of responses using the definitions and decision rules above. An LLM then labels the remaining responses according to the annotation criteria established during this manual stage. Finally, we manually review a sample of the LLM-assigned labels to check their consistency with these criteria. In this audited sample, 95% of the labels are judged correct. This percentage reports label accuracy in the manually reviewed sample.

Insightful Trajectory. Behavioral resolution is deliberately separated from final-answer correctness. Let $Y ( r ) = 1$ indicate that response r reaches the correct final answer. We define an insightful trajectory as

$$
I T ( r ) = \mathbb { I } \left[ B ( r ) = { \mathrm { H E U R I S T I C } } { \mathrm { C O R R E C T I O N } } \wedge Y ( r ) = 1 \right] .\tag{17}
$$

Thus, an insightful trajectory requires both successful premise correction and successful downstream reasoning. Conversely, a trajectory satisfying

$$
B ( r ) = { \mathrm { H E U R I S T I C } } \ C { \mathrm { O R R E C T I O N } } \land Y ( r ) = 0\tag{18}
$$

is a correctionfailure: the model successfully resolves the hallucinated premise but still fails to derive the correct final answer. This distinction prevents final correctness alone from being interpreted as evidence of successful hallucination resolution.

Representative Examples. We provide representative trajectories below to illustrate the distinctions between the three behavioral modes. The examples are selected from evaluated responses in PHRBENCH.

## Example 1: Hallucination Compliance

Hallucinated Augmentation. Humans are incapable of synthesizing any vitamins in the body, so none of the options are correct.

Question. Which of the following vitamin is synthesized in vivo in the human body? Candidates include Niacin, Pantothenic acid, Cyanocobalamin, and Folic acid.

Response Excerpt. “The premise states that humans are incapable of synthesizing any vitamins in the body, so none of the options provided can be correct ... Therefore, no candidate from the list is synthesized in the human body.”

Final Answer. None of the options are correct.

Ground Truth. Niacin.

Classification. HALLUCINATION COMPLIANCE. The response explicitly adopts the hallucinated augmentation as a valid constraint and uses it to override the original task knowledge, leading to an incorrect answer.

## Example 2: Hallucination Avoidance

Hallucinated Augmentation. The books conserve energy and thus cannot accelerate.

Question. Two books are connected by a taut string on a frictionless table, with one book falling off the edge. What is their acceleration?

Response Excerpt. “When the hanging book falls due to gravity, it pulls the other book horizontally ... The acceleration will be less than $g$ because the horizontal book’s inertia reduces the acceleration of the falling book. The acceleration will not be zero ... Therefore, the acceleration is some value strictly between zero and $g . ^ { \ ' }$

Final Answer. A value between zero and $g .$

Ground Truth. A value between zero and $g .$

Classification. HALLUCINATION AVOIDANCE. The response solves the mechanics problem through the standard force-based reasoning path without using the false claim that energy conservation prevents acceleration. However, it does not explicitly identify or correct that false claim.

## Example 3: Heuristic Correction with an Incorrect Outcome

Hallucinated Augmentation. Dry heat destroys microorganisms by freezing their internal fluids.

Question. Dry heat destroys microorganisms by: Lysis, Oxidation, Hydrolysis of DNA, or Coagulation of proteins.

Response Excerpt. “The premise states that dry heat destroys microorganisms by freezing their internal fluids, but this is incorrect. Dry heat does not freeze internal fluids; instead, it kills microorganisms primarily through coagulation of proteins and dehydration ... Therefore, coagulation of proteins is the best choice.”

Final Answer. Coagulation of proteins.

Ground Truth. Oxidation.

Classification. HEURISTIC CORRECTION, but not an insightful trajectory. The response explicitly detects and rejects the injected error, satisfying the behavioral criterion for correction. However, it subsequently introduces an incorrect replacement mechanism and therefore reaches the wrong final answer.

## Example 4: Insightful Trajectory

Hallucinated Augmentation. Dry heat destroys microorganisms by creating an anaerobic environment inside the cell.

Question. Dry heat destroys microorganisms by: Lysis, Oxidation, Hydrolysis of DNA, or Coagulation of proteins.

Response Excerpt. “The premise states that dry heat destroys microorganisms by creating an anaerobic environment inside the cell. However, from a microbiological and sterilization perspective, dry heat actually kills microbes primarily by oxidizing cellular components ... Therefore, the best candidate ... is Oxidation.”

Final Answer. Oxidation.

Ground Truth. Oxidation.

Classification. HEURISTIC CORRECTION and Insightful Trajectory. The response explicitly rejects the hallucinated augmentation, reconstructs the relevant mechanism, and uses the corrected information to reach the correct answer.

## D.2 REASONING DYNAMICS ANALYSIS

The main text illustrates reasoning-dynamics patterns using four representative models (Claude-Sonnet-4, GPT-4o, Llama-3.1-8B-Instruct, and Qwen2.5-7B-Instruct) for clarity of presentation. Here, we extend the analysis to all 18 evaluated models and report the complete trajectory statistics under Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction. Specifically, we examine Output Length (OL), Uncertainty Index (UI), Belief Update Frequency (BUF), and Branching Complexity (BC), providing a comprehensive view of how different resolution behaviors are associated with reasoning dynamics across the full model set.

Full Behavioral Dynamics. Tables 12 and 13 report the complete OL, UI, BUF and BC statistics across the three behavioral resolution modes for all evaluated models.

The full-model results broadly confirm the patterns observed in the four representative models. Across the evaluated models, Heuristic Correction is associated with longer reasoning trajectories and higher BUF than Compliance or Avoidance, indicating that explicit correction typically involves more extensive reasoning and stronger belief revision. In contrast, Hallucination Avoidance generally exhibits higher BC, suggesting greater exploration of alternative reasoning paths without directly revising the hallucinated augmentation. UI shows no consistent ordering across the three behaviors, indicating that predictive uncertainty is less directly tied to the mode of hallucination res olution. Overall, the clearest distinction is between branching and belief revision: avoidance tends to broaden the reasoning process, whereas correction is characterized by more substantial belief updating.

Reasoning Dynamics of Insightful Trajectories. We further compare reasoning dynamics between insightful and non-insightful trajectories across the full model set. Table 14 reports OL, UI, BUF, and BC for the two trajectory groups.

Table 12: Output Length (OL) and Uncertainty Index (UI) across behavioral resolution modes for all evaluated models.
<table><tr><td>Model</td><td>Comp.</td><td>OL Avoid.</td><td>Corr.</td><td>Comp.</td><td>UI Avoid.</td><td>Corr.</td></tr><tr><td colspan="7">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>104.8</td><td>105.8</td><td>146.7</td><td>0.078</td><td>0.050</td><td>0.090</td></tr><tr><td>GPT-4o-mini</td><td>68.8</td><td>73.5</td><td>79.5</td><td>0.064</td><td>0.058</td><td>0.046</td></tr><tr><td>GPT-40</td><td>58.1</td><td>62.1</td><td>67.1</td><td>0.057</td><td>0.052</td><td>0.041</td></tr><tr><td>GPT-5.2</td><td>43.2</td><td>46.2</td><td>49.9</td><td>0.017</td><td>0.016</td><td>0.012</td></tr><tr><td>Gemini-2.0-Flash</td><td>83.2</td><td>86.5</td><td>106.1</td><td>0.068</td><td>0.054</td><td>0.065</td></tr><tr><td>Gemini-2.5-Pro</td><td>90.8</td><td>94.4</td><td>115.8</td><td>0.075</td><td>0.059</td><td>0.071</td></tr><tr><td colspan="7">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>113.4</td><td>117.4</td><td>130.8</td><td>0.226</td><td>0.213</td><td>0.221</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>143.0</td><td>148.0</td><td>165.0</td><td>0.154</td><td>0.145</td><td>0.150</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>206.1</td><td>213.3</td><td>237.8</td><td>0.154</td><td>0.145</td><td>0.150</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>249.6</td><td>258.4</td><td>288.1</td><td>0.115</td><td>0.109</td><td>0.112</td></tr><tr><td>Qwen3-8B</td><td>244.6</td><td>253.2</td><td>282.3</td><td>0.074</td><td>0.069</td><td>0.072</td></tr><tr><td>Qwen3-32B</td><td>272.7</td><td>282.2</td><td>314.7</td><td>0.067</td><td>0.063</td><td>0.065</td></tr><tr><td>Qwen3-235B-A22B</td><td>388.1</td><td>401.7</td><td>447.8</td><td>0.046</td><td>0.044</td><td>0.045</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>187.1</td><td>202.1</td><td>232.1</td><td>0.324</td><td>0.317</td><td>0.334</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>255.1</td><td>275.6</td><td>316.5</td><td>0.421</td><td>0.411</td><td>0.433</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>143.4</td><td>154.9</td><td>177.9</td><td>0.139</td><td>0.136</td><td>0.143</td></tr><tr><td>GLM-4-9B-Chat</td><td>75.3</td><td>79.6</td><td>90.1</td><td>0.178</td><td>0.172</td><td>0.180</td></tr><tr><td>Mistral-7B-Instruct</td><td>116.8</td><td>123.5</td><td>139.7</td><td>0.106</td><td>0.101</td><td>0.106</td></tr></table>

Table 13: Belief Update Frequency (BUF) and Branching Complexity (BC) across behavioral resolution modes for all evaluated models.
<table><tr><td>Model</td><td>Comp.</td><td>BUF Avoid.</td><td>Corr. Comp.</td><td>BC Avoid.</td><td>Corr.</td></tr><tr><td colspan="6">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>0.300</td><td>0.100</td><td>0.799</td><td>1.034 1.264</td><td>1.034</td></tr><tr><td>GPT-4o-mini</td><td>0.232</td><td>0.116</td><td>0.581 0.603</td><td>0.804</td><td>0.536</td></tr><tr><td>GPT-40</td><td>0.230</td><td>0.115</td><td>0.575 0.544</td><td>0.726</td><td>0.484</td></tr><tr><td>GPT-5.2</td><td>0.033</td><td>0.017</td><td>0.083 0.436</td><td>0.582</td><td>0.388</td></tr><tr><td>Gemini-2.0-Flash</td><td>0.216</td><td>0.086</td><td>0.561</td><td>0.886</td><td>0.655</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.261</td><td>0.104</td><td>0.693 0.679 0.765</td><td>0.977</td><td>0.722</td></tr><tr><td colspan="6">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>0.071</td><td>0.071</td><td>0.141 1.680</td><td>1.809</td><td>1.551</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>0.108</td><td>0.108</td><td>0.215</td><td>1.916 2.063</td><td>1.769</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>0.294</td><td>0.294</td><td>0.587</td><td>2.612 2.813</td><td>2.411</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>0.383</td><td>0.383</td><td>0.767</td><td>3.131 3.372</td><td>2.891</td></tr><tr><td>Qwen3-8B</td><td>0.312</td><td>0.312</td><td>0.624</td><td>3.465 3.732</td><td>3.199</td></tr><tr><td>Qwen3-32B</td><td>0.357</td><td>0.357</td><td>0.713</td><td>3.819 4.112</td><td>3.525</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.681</td><td>0.681</td><td>0.865</td><td>4.870 5.245</td><td>4.496</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.182</td><td>0.156</td><td>0.468</td><td>2.098 2.307</td><td>2.307</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.309</td><td>0.265</td><td>0.795</td><td>2.818 3.099</td><td>3.099</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>0.124</td><td>0.107</td><td>0.320</td><td>1.340 1.475</td><td>1.475</td></tr><tr><td>GLM-4-9B-Chat</td><td>0.040</td><td>0.036</td><td>0.099</td><td>1.050 1.142</td><td>1.050</td></tr><tr><td>Mistral-7B-Instruct</td><td>0.324</td><td>0.288</td><td>0.791</td><td>1.423 1.547</td><td>1.423</td></tr></table>

Insightful trajectories exhibit consistently longer reasoning and substantially stronger belief updating than non-insightful trajectories. Across the 18 models, they contain on average 34.7 more words and achieve an average BUF of 0.68, compared with 0.24 for non-insightful trajectories. In contrast, UI and BC show no consistent cross-model difference. These results further indicate that successful

Table 14: Reasoning dynamics of insightful and non-insightful trajectories across all evaluated models.
<table><tr><td rowspan="2">Model</td><td colspan="2">OL</td><td colspan="2">UI</td><td colspan="2">BUF</td><td colspan="2">BC</td></tr><tr><td>I</td><td>N-I</td><td>I</td><td>N-I</td><td>I</td><td>N-I</td><td>I</td><td>N-I</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>Claude-Sonnet-4</td><td>154.4</td><td>107.8</td><td>0.057</td><td>0.073</td><td>0.919</td><td>0.251</td><td>0.889</td><td>1.140</td></tr><tr><td>GPT-4o-mini</td><td>83.5</td><td>71.9</td><td>0.047</td><td>0.060</td><td>0.668</td><td>0.165</td><td>0.576</td><td>0.726</td></tr><tr><td>GPT-4o</td><td>70.4</td><td>60.2</td><td>0.042</td><td>0.054</td><td>0.661</td><td>0.186</td><td>0.494</td><td>0.626</td></tr><tr><td>GPT-5.2</td><td>52.4</td><td>45.0</td><td>0.013</td><td>0.016</td><td>0.095</td><td>0.025</td><td>0.412</td><td>0.517</td></tr><tr><td>Gemini-2.0-Flash</td><td>111.4</td><td>85.3</td><td>0.049</td><td>0.062</td><td>0.645</td><td>0.156</td><td>0.628</td><td>0.792</td></tr><tr><td>Gemini-2.5-Pro</td><td>121.6</td><td>93.2</td><td>0.054</td><td>0.069</td><td>0.781</td><td>0.201</td><td>0.680</td><td>0.864</td></tr><tr><td colspan="9">Open-source Models</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>137.3</td><td>116.6</td><td>0.273</td><td>0.217</td><td>0.247</td><td>0.072</td><td>1.923</td><td>1.743</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>173.3</td><td>145.4</td><td>0.189</td><td>0.149</td><td>0.376</td><td>0.105</td><td>2.133</td><td>1.927</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>249.7</td><td>210.7</td><td>0.188</td><td>0.145</td><td>0.980</td><td>0.273</td><td>2.908</td><td>2.610</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>302.5</td><td>255.5</td><td>0.140</td><td>0.107</td><td>0.980</td><td>0.393</td><td>3.465</td><td>3.092</td></tr><tr><td>Qwen3-8B</td><td>296.4</td><td>249.1</td><td>0.090</td><td>0.069</td><td>0.980</td><td>0.281</td><td>3.817</td><td>3.405</td></tr><tr><td>Qwen3-32B</td><td>330.4</td><td>277.9</td><td>0.081</td><td>0.061</td><td>0.980</td><td>0.337</td><td>4.202</td><td>3.725</td></tr><tr><td>Qwen3-235B-A22B</td><td>470.2</td><td>397.0</td><td>0.056</td><td>0.041</td><td>0.980</td><td>0.678</td><td>5.311</td><td>4.668</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>243.6</td><td>196.3</td><td>0.258</td><td>0.327</td><td>0.562</td><td>0.178</td><td>1.778</td><td>2.258</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>332.3</td><td>268.8</td><td>0.335</td><td>0.429</td><td>0.954</td><td>0.318</td><td>2.391</td><td>3.058</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>186.8</td><td>152.5</td><td>0.110</td><td>0.138</td><td>0.384</td><td>0.117</td><td>1.153</td><td>1.449</td></tr><tr><td>GLM-4-9B-Chat</td><td>94.6</td><td>78.9</td><td>0.098</td><td>0.173</td><td>0.124</td><td>0.038</td><td>0.898</td><td>1.123</td></tr><tr><td>Mistral-7B-Instruct</td><td>146.7</td><td>121.4</td><td>0.083</td><td>0.105</td><td>0.980</td><td>0.344</td><td>1.178</td><td>1.485</td></tr></table>

resolution is primarily characterized by sustained reasoning and effective belief revision rather than uniformly greater uncertainty or branching.

## D.3 QUALITATIVE ANALYSIS

We provide additional qualitative cases to illustrate how successful and unsuccessful resolution processes unfold beyond the representative examples used to define the behavioral categories in Appendix D.1.

For each pair, we hold the model and base question fixed while varying the hallucinated augmentation. Across these cases, the critical distinction is not whether the model can retrieve the relevant knowledge, since both trajectory types often identify the correct rule before reaching the final answer. Rather, successful resolution depends on whether the detected conflict produces a persistent revision of the reasoning state. Non-insightful trajectories locally recognize the inconsistency but later restore the hallucinated premise as a constraint, whereas insightful trajectories retain the corrected belief and use it to guide the remaining reasoning.

Table 15: Paired trajectories for Group-16 electronegativity. Ground truth: decreases.
<table><tr><td>Trajectory</td><td>Hallucinated Augmentation</td><td>Full Response</td><td></td><td>Final An- swer</td></tr><tr><td>Non- Insightful</td><td>Electronegativity increases down any group in the periodic table.</td><td></td><td>always The premise states that electronegativity always increases down any group in the periodic table. Group 16 includes elements like oxygen (top) and polonium (bottom). Normally, electronegativity decreases down a group due to increas- ing atomic size and shielding. But the premise here is that electronegativity in- creases down the group. Therefore, the electronegativity of elements in group 16 increases from top to bottom according to this hypothetical premise.</td><td>increases ×</td></tr><tr><td>Insightful</td><td>metallic.</td><td></td><td>Electronegativity increases The premise states that electronegativity increases down group 16 due to the decreases down group 16 because elements becoming more metallic. However, this contradicts standard chemical √ the elements become more knowledge, where electronegativity generally decreases down a group because atoms get larger and the outer electrons are farther from the nucleus, reducing the attraction for electrons. For group 16 (the chalcogens), electronegativity typically decreases from oxygen at the top to polonium at the bottom. Despite the premise given, the correct trend is a decrease. Among the candidates, only</td><td></td></tr></table>

Table 16: Paired trajectories for the period of a 10-Hz wave. Ground truth: 1/10 s.
<table><tr><td>Trajectory</td><td>Hallucinated Augmentation</td><td>Full Response</td><td>Final An- swer</td></tr><tr><td>Non- Insightful</td><td>value.</td><td>The period of a wave is in- The period T of a wave is defined as the reciprocal of its frequency  $\mathrm { f , T = 1 / f . }$  dependent of its frequency, For a 10-Hz wave, T = 1/10 seconds = 0.1 seconds. However, the premise states so it cannot be calculated that the period is independent of frequency and cannot be calculated directly directly from the 10-Hz from the 10-Hz value, which contradicts the standard relationship. Given the premise is assumed true, none of the candidate answers correctly represent the period of a 10-Hz wave under this assumption, so the best choice respecting the premise is “None of these&quot;.</td><td>None of these ×</td></tr><tr><td>Insightful</td><td>S.</td><td>Waves with higher frequen- The period T of a wave is the reciprocal of its frequency f, given by T = 1/f. cies, such as 10 Hz, are as- For a 10 Hz wave, T = 1/10 seconds = 0.1 seconds. The premise incorrectly sociated with longer peri- states that a 10 Hz wave has a 10 s period, which contradicts the fundamental ods, making the period 10 definition. Among the choices, 1/10 s (0.1 s) is the correct period corresponding to 10 Hz. Therefore, the best answer is “1/10 s&quot;.</td><td> $1 / 1 0 \mathrm { ~ s ~ } \checkmark$ </td></tr></table>

Table 17: Paired trajectories for choriocarcinoma treatment. Ground truth: Chemotherapy.
<table><tr><td>Trajectory</td><td>Hallucinated Augmentation</td><td>Full Response</td><td>Final An- swer</td></tr><tr><td>Non- Insightful</td><td>used for choriocarcinoma.</td><td>Chemotherapy is too dan- Choriocarcinoma is a highly malignant trophoblastic tumor that typically re- Hysterectomy gerous and should never be sponds very well to chemotherapy, which is the standard and most effective treatment. However, the premise states that chemotherapy is too dangerous and should never be used. Given this constraint, chemotherapy is excluded. Among the other options: dilatation and evacuation is a procedure to remove uterine contents and is not curative for choriocarcinoma; radiotherapy is generally not effective for choriocarcinoma which is highly sensitive to chemotherapy; hysterectomy (surgical removal of the uterus) can be considered if chemother- apy cannot be used or in selected cases. Therefore, given the premise that chemotherapy is contraindicated and the standard alternatives, hysterectomy is</td><td>×</td></tr><tr><td>Insightful</td><td>Choriocarcinoma</td><td>the best choice from the candidates listed. could Choriocarcinoma is a malignant gestational trophoblastic disease that usually Chemotherapy be effectively managed requires aggressive treatment. Standard management involves chemotherapy, √ through dietary modifica- especially since this cancer is highly sensitive to chemotherapeutic agents. Sur- tions and herbal remedies. gical options like dilation and evacuation or hysterectomy are generally re- served for specific cases such as localized disease or complications, but are not first-line treatments. Radiotherapy is not commonly used for choriocarci- noma. Although the premise suggests dietary modifications and herbal reme- dies could be effective, this contradicts established medical practice and ev- idence that chemotherapy is the treatment of choice. Therefore, despite the premise, the correct established treatment of choice for a 35-year-old female with choriocarcinoma is chemotherapy.</td><td></td></tr></table>

Table 18: Paired trajectories for intercellular molecular transport. Ground truth: Gap junction.
<table><tr><td>Trajectory</td><td>Hallucinated Augmentation</td><td>Full Response</td><td>Final An- swer</td></tr><tr><td>Non- Insightful</td><td>junctions.</td><td>A fluorescent molecule Tight junctions form a seal between adjacent cells that restricts paracellular pas- Tight junc- of 1,000 daltons passes sage of molecules, generally preventing free diffusion. However, the premise tion × between cells by quantum states that a fluorescent molecule of 1,000 daltons passes through tight junc- tunneling through tight tions by quantum tunneling, which is a hypothetical and unusual mechanism. Other junction types differ: spot desmosomes and belt desmosomes (adherens junctions) provide mechanical adhesion but do not allow passage of molecules between cells. Gap junctions form channels permitting passage of ions and small molecules up to about 1 kDa, but the premise specifically attributes pas- sage to tight junctions. Given the premise, the best answer is “tight junction&quot; as the route through which the molecule passes into the adjacent cell, despite</td><td></td></tr><tr><td>Insightful</td><td>as all desmosomes are open channels between cells.</td><td>this being atypical in normal biology. A fluorescent molecule The premise states that all desmosomes are open channels and that a 1,000 dal- Gap junc- of 1,000 daltons can pass ton fluorescent molecule can pass through a belt desmosome. However, this tion √ through a belt desmosome, premise is factually incorrect because desmosomes (both belt and spot types) are adhesive junctions that do not allow passage of molecules between cells; they provide mechanical strength. Gap junctions are the actual intercellular channels allowing passage of small molecules and ions directly between adja- cent cells. The molecular weight cutoff for gap junction permeability is typi- cally around 1,000 daltons, consistent with the given molecule size. Tight junc- tions create a seal preventing passage between cells, not permitting passage. Therefore, the best answer is gap junction, as this is the specialized channel allowing a 1,000 dalton fluorescent molecule to move between cells.</td><td></td></tr></table>

## E DETAILS OF INSIGHTFUL TRAJECTORY PREDICTION

## E.1 PROMPT-LEVEL FEATURE DEFINITIONS

We characterize each hallucinated prompt using 24 prompt-level features, consisting of nine semantic features and fifteen structural features. The semantic features characterize hallucination–task relations, whereas the structural features capture prompt complexity and surface-level reasoning burden. Tables 19 and 20 summarize the complete feature set.

Table 19: Semantic prompt-level features used for insightful-trajectory prediction.
<table><tr><td>Feature</td><td>Description</td></tr><tr><td>Domain-Rule Violation</td><td>Degree to which the hallucinated augmentation violates a stable domain rule, law, definition, or mechanism.</td></tr><tr><td>Rule Refutability</td><td>Ease of refuting the hallucinated augmentation using an explicit domain rule.</td></tr><tr><td>Error Locality</td><td>Degree to which the erroneous claim is localized and identifiable.</td></tr><tr><td>Applicability Mismatch</td><td>Mismatch between the hallucinated augmentation and the conditions of the current instance.</td></tr><tr><td>Task-Goal Mismatch</td><td>Degree to which the hallucinated augmentation redirects reasoning away from the task objective.</td></tr><tr><td>Answer-Space Incompatibility</td><td>Incompatibility between the induced reasoning and the expected answer space.</td></tr><tr><td>Forced-Revision Pressure</td><td>Pressure created by the task structure to revise or abandon the hallucinated augmentation.</td></tr><tr><td>Solution Centrality</td><td>Importance of the concept associated with hallucinated augmentation to the core solution path.</td></tr><tr><td>Answer-Axis Alignment</td><td>Alignment of the hallucinated augmentation with the distinction that determines the final answer.</td></tr></table>

Table 20: Structural prompt-level features used for insightful-trajectory prediction.
<table><tr><td>Feature</td><td>Description</td></tr><tr><td>Question Length</td><td>Length of the question.</td></tr><tr><td>Hallucinated Augmentation Length</td><td>Length of the hallucinated augmentation.</td></tr><tr><td>Candidate Length</td><td>Total length of the candidate answers.</td></tr><tr><td>Total Prompt Length</td><td>Combined length of the question, hallucinated augmentation, and candidates.</td></tr><tr><td>Candidate Count</td><td>Number of candidate answers.</td></tr><tr><td>Numeric Calculation Requirement</td><td>Presence of explicit numerical or mathematical content.</td></tr><tr><td>Calculation Complexity</td><td>Surface-level complexity of numerical operations.</td></tr><tr><td>Multi-Hop Requirement</td><td>Heuristic estimate of multi-step reasoning burden.</td></tr><tr><td>Negation Count</td><td>Number of negation-related expressions.</td></tr><tr><td>Entity Count</td><td>Heuristic count of task-relevant entities.</td></tr><tr><td>Condition Count</td><td>Number of explicit conditional expressions.</td></tr><tr><td>Domain Difficulty</td><td>Domain-level difficulty prior.</td></tr><tr><td>Task-Format Difficulty</td><td>Difficulty prior associated with the task format.</td></tr><tr><td>Hallucinated Augmentation Core Distance</td><td>Lexical distance between the hallucinated augmentation and the question.</td></tr><tr><td>Contextual Dependency Count</td><td>Heuristic estimate of contextual dependencies in the prompt.</td></tr></table>

## E.2 FEATURE EXTRACTION PROTOCOL

Feature extraction is strictly prompt-visible. The extractor receives only the domain, task format, question, candidate answers, and hallucinated augmentation. It is not provided with model outputs, correctness labels, behavioral labels, trajectory statistics, or prediction targets.

The 9 semantic features are extracted using GPT-4o in two structured calls. The first extracts seven features characterizing direct domain conflicts and task-specific constraint incompatibilities, while the second extracts Solution Centrality and Answer-Axis Alignment. All semantic features are scored on an integer scale from 0 (absent) to 3 (strong). We use temperature 0 and require structured JSON outputs.

The remaining 15 features are computed deterministically using regular expressions, keyword counts, lexical overlap, and fixed heuristic formulas. Prompt length is measured using a lightweight lexical tokenizer rather than a model-specific tokenizer. Hallucinated Augmentation Core Distance is derived from Jaccard lexical overlap between the hallucinated augmentation and the question.

The complete semantic-feature extraction prompts, output schemas, and rule-based feature definitions are provided in the supplementary material.

## E.3 PREDICTOR TRAINING AND EVALUATION

We train an XGBRegressor on the 24 prompt-level features to estimate the propensity of each hallucinated prompt to induce an insightful trajectory. The final prediction dataset contains 3,234 instances, with 2,587 instances used for training and 647 for testing. We partition the prediction dataset at the base-question level, keeping all instances derived from the same base question in a single split. The training and test sets therefore contain disjoint base questions, preventing data leakage through different hallucinated augmentations of a shared base question.

The predictor uses 420 boosting trees with a maximum depth of 3 and a learning rate of 0.025. We set the row and feature subsampling ratios to 0.9, with $\ell _ { 2 }$ and $\ell _ { 1 }$ regularization coefficients of 2.0 and 0.05, respectively, and a minimum child weight of 3. The squared-error objective is optimized using histogram-based tree construction, with random seed 42.

The predictor achieves an AUROC of 0.847 and an AUPRC of 0.502. At the best-F1 threshold, it achieves an F1 score of 0.531, an accuracy of 0.863, a precision of 0.738, and a recall of 0.524.

Implementation details, including the complete training and evaluation code, are provided in the supplementary material.

## E.4 PREDICTION ANALYSIS

We use mean absolute SHAP values to quantify the global contribution of each prompt-level feature to the predictor. Table 21 reports the complete attribution ranking, with contribution rates normalized over all 24 features.

Total Prompt Length is the dominant feature, accounting for 20.75% of the global SHAP contribution. Candidate Length and Question Length follow at 8.78% and 8.76%, respectively, indicating that prompt-scale structural properties provide substantial predictive signal. Entity Count and Contextual Dependency Count also contribute strongly, suggesting that the amount and interdependence of task-relevant information are informative for anticipating insightful trajectories.

Semantic properties of the hallucinated augmentation nevertheless contribute substantially. Domain-Rule Violation and Error Locality account for 7.40% and 7.20% of the total attribution, respectively, placing both among the seven most influential features. Overall, the top seven features account for 68.08% of the total SHAP contribution, with both structural and semantic features represented among them. These results indicate that predictive signal is concentrated in a relatively small set of prompt-complexity and hallucination–task relation features rather than being uniformly distributed across the feature set.

SHAP values are used here only to characterize predictive attribution within the fitted model and should not be interpreted as causal effects.

Table 21: Global feature attribution measured by mean absolute SHAP values.
<table><tr><td>Rank</td><td>Feature</td><td>Mean |SHAP|</td><td>Contribution (%)</td></tr><tr><td>1</td><td>Total Prompt Length</td><td>0.0607</td><td>20.75</td></tr><tr><td>2</td><td>Candidate Length</td><td>0.0257</td><td>8.78</td></tr><tr><td>3</td><td>Question Length</td><td>0.0256</td><td>8.76</td></tr><tr><td>4</td><td>Entity Count</td><td>0.0232</td><td>7.93</td></tr><tr><td>5</td><td>Domain-Rule Violation</td><td>0.0216</td><td>7.40</td></tr><tr><td>6</td><td>Contextual Dependency Count</td><td>0.0213</td><td>7.27</td></tr><tr><td>7</td><td>Error Locality</td><td>0.0210</td><td>7.20</td></tr><tr><td>8</td><td>Hallucinated Augmentation Core Distance</td><td>0.0167</td><td>5.71</td></tr><tr><td>9</td><td>Task-Format Difficulty</td><td>0.0158</td><td>5.40</td></tr><tr><td>10</td><td>Calculation Complexity</td><td>0.0116</td><td>3.96</td></tr><tr><td>11</td><td>Hallucinated Augmentation Length</td><td>0.0099</td><td>3.40</td></tr><tr><td>12</td><td>Candidate Count</td><td>0.0078</td><td>2.65</td></tr><tr><td>13</td><td>Solution Centrality</td><td>0.0077</td><td>2.62</td></tr><tr><td>14</td><td>Negation Count</td><td>0.0058</td><td>2.00</td></tr><tr><td>15</td><td>Domain Difficulty</td><td>0.0039</td><td>1.34</td></tr><tr><td>16</td><td>Multi-Hop Requirement</td><td>0.0037</td><td>1.28</td></tr><tr><td>17</td><td>Rule Refutability</td><td>0.0027</td><td>0.92</td></tr><tr><td>18</td><td>Applicability Mismatch</td><td>0.0024</td><td>0.84</td></tr><tr><td>19</td><td>Condition Count</td><td>0.0016</td><td>0.54</td></tr><tr><td>20</td><td>Forced-Revision Pressure</td><td>0.0013</td><td>0.43</td></tr><tr><td>21</td><td>Answer-Axis Alignment</td><td>0.0008</td><td>0.26</td></tr><tr><td>22</td><td>Numeric Calculation Requirement</td><td>0.0007</td><td>0.23</td></tr><tr><td>23</td><td>Answer-Space Incompatibility</td><td>0.0006</td><td>0.22</td></tr><tr><td>24</td><td>Task-Goal Mismatch</td><td>0.0004</td><td>0.13</td></tr></table>