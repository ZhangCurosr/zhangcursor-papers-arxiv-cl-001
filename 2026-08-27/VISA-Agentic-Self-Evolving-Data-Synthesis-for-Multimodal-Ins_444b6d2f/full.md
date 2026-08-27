# VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following

Min Zeng, Guanxin Tan, Libin Cen, Yawei Wen Rui Hu, Liuyang Bian, Xiaolong Chen, Xiaoxin Chen vivo AI Lab zengmin325@163.com

## Abstract

Multimodal instruction-following models re quire training data that is accurate, diverse, verifiable, and challenging. Existing synthesis pipelines typically follow a one-pass generateand-filter paradigm, discarding feedback from failed samples, verifier outcomes, and targetmodel errors. We present VISA (Visual Instruction Synthesis Agent), an agentic framework that reformulates multimodal instruction synthesis as a self-evolving loop. At each round, VISA analyzes an image to filter incompatible constraints and discover new verifiable ones, samples diversity- and difficulty-aware constraint sets from persistent memory, generates candidate instructions, and verifies the resulting samples with executable tools and structured large language model judges. Failed samples trigger diagnostic-guided recovery, while accepted samples are probed against the target model to estimate difficulty. The resulting verifier signals and target-model failure profiles are written back to memory, allowing subsequent rounds to adaptively expand the constraint space, reduce template repetition, and focus on unresolved model weaknesses. The same verifier contracts further provide reward signals for reinforcement learning without a separately trained reward model. Experiments on MM-IFEval show that VISA consistently improves multimodal instruction following over strong baselines, while preserving general multimodal capability across seven public benchmarks.

## 1 Introduction

Multimodal large language models (MLLMs) have achieved remarkable progress in visual understanding and reasoning abilities (Liu et al., 2023; Team, 2026; Bai et al., 2025; Li et al., 2024). Beyond perception, real-world applications increasingly require MLLMs to precisely follow complex user instructions under diverse constraints, such as output formatting, keyword inclusion, style control, structured generation, and visually grounded reasoning (Zhou et al., 2023; Qian et al., 2025). Building MLLMs that can reliably satisfy such multimodal constraints has therefore become an important problem for multimodal alignment and instruction tuning.

Recent works improve multimodal instruction following by synthesizing constraint-rich data and designing specialized evaluation benchmarks (Ding et al., 2025; He et al., 2026). However, most synthesis pipelines are still one-pass systems. They generate instructions from templates or prompts, filter the outputs, and then add the accepted samples to a training set. Feedback from failed samples, repeated templates, missing verifiers, or target-model errors is often used only locally, if at all. As a result, the synthesis process lacks an agent loop: it does not reflect on its past outputs, update its internal state, or use that state to plan.

In this paper, we present VISA (Visual Instruction Synthesis Agent), a self-evolving agentic framework that treats multimodal instruction synthesis as a closed-loop optimization problem rather than a one-shot generation task. At its core, VISA runs an agent loop between a planning and execution stage, which composes image-grounded instructions and generates responses, and a reflection stage, which evaluates each sample through a unified dual-track verifier combining executable code tools and structured large language model (LLM) judges. When a sample fails verification, the loop triggers diagnostic-guided recovery; when it passes, the loop probes the target model to estimate difficulty. Crucially, the outcomes of every iteration—accepted samples, constraint–verifier bindings, diversity statistics, and per-constraint failure rates—are written into a persistent memory that reshapes the next round’s constraint sampling and instruction planning. Through this iterative accumulation, VISA continuously steers synthesis toward data that is more accurate, more diverse, and more challenging to the current target model, without additional human annotation during self-evolution.

Our contributions are three-fold: (1) We propose a self-evolving framework for multimodal instruction data synthesis that turns static generation into an iterative feedback-driven process; (2) We introduce a unified verification and recovery framework that supports dynamic verifier binding, bounded agentic repair, and target-aware difficulty probing during synthesis, while naturally producing verifiable reward signals that can be readily extended to reinforcement learning (RL); (3) We demonstrate through extensive experiments on multiple multimodal instruction-following benchmarks that VISA consistently improves instruction adherence while preserving strong general multimodal capabilities.

## 2 Related Work

Existing multimodal instruction synthesis pipelines primarily scale data diversity and response quality through large-scale synthetic generation (Huang et al., 2025; Zhang et al., 2024), with recent works such as MM-IFEngine (Ding et al., 2025) further introducing structured constraints and hybrid verification to improve instruction-following quality. Selfevolving approaches such as Self-Instruct (Wang et al., 2023), WizardLM (Xu et al., 2023), and MMEvol (Luo et al., 2025) improve instruction complexity through iterative rewriting or evolutionary generation. However, all of these methods remain fundamentally open-loop: they optimize what is generated but leave the synthesis process itself static, with no mechanism to adapt based on what the target model has yet to master. VISA instead treats synthesis as a closed-loop optimization process, where verification outcomes, difficulty signals, and accumulated memory continuously reshape subsequent rounds of data generation.

Prior works on instruction-following evaluation (Zhou et al., 2023; Qian et al., 2025) demonstrate that many constraints admit precise programmatic verification—an insight VISA extends from evaluation into synthesis by binding each constraint to a verifier at generation time. Agentic LLM systems (Yao et al., 2022; Hong et al., 2024; Qin et al., 2024) similarly decompose tasks into perception, planning, and execution stages, but typically operate within isolated episodes without persistent state. VISA maintains a persistent memory across synthesis rounds, so that reflection outcomes directly reshape future planning rather than being discarded at the end of each episode.

## 3 Method

VISA casts multimodal instruction synthesis as a closed-loop process with four stages: Perception, Planning & Execution, Reflection, and Memory Update. As shown in Figure 1, each round generates, verifies, and repairs candidate samples, then writes accepted samples and difficulty signals into persistent memory to guide the next round.

## 3.1 Problem Formulation

We frame synthesis as image-conditioned constraint satisfaction. Given an image pool I and a dynamically evolving constraint-type registry $\tau$ , VISA constructs a multimodal instructionfollowing dataset

$$
\mathcal { D } = \{ ( I _ { i } , Q _ { i } , R _ { i } , C _ { i } , v _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{1}
$$

where $I _ { i } \in \mathcal { Z }$ is an input image, $Q _ { i }$ is an instruction grounded in $I _ { i } , C _ { i } \subset \tau$ is the set of constraints embedded in $Q _ { i } , R _ { i }$ is the corresponding response expected to satisfy every $c \in C _ { i }$ , and $v _ { i }$ denotes the verification procedure—either a code-based verifier or a structured LLM judge—bound to the sample at synthesis time. The registry $\tau$ is not fixed: new constraint types may be proposed and validated during Perception, allowing the synthesized data to expand beyond its initial scope without human annotation.

## 3.2 Perception

A fundamental limitation of static synthesis pipelines is that a hand-crafted constraint registry, however carefully designed, cannot anticipate every instruction requirement that naturally arises from a diverse image collection. The Perception stage addresses this by first analyzing each image with the synthesis model to identify which constraint types in $\tau$ are meaningful for the current scene, returning a compatible subset $\smash {  { S _ { I } } \subseteq \tau }$ by pruning types whose requirements cannot be grounded in the visual content—a spatial\_relation constraint, for instance, is semantically vacuous for a close-up portrait. Restricting downstream planning to $ { \boldsymbol { S } } _ { I }$ ensures that every selected constraint is both verifiable and visually meaningful, preventing the planner from composing instructions that are technically valid but disconnected from the image.

![](images/5e4eadad6ae905698782a09733184900bc576d687f2d4eaf89b17109695ff18c.jpg)  
Figure 1: Overview of VISA. The framework iteratively performs perception, planning and execution, reflection, and memory update, allowing verifier feedback and target-model difficulty signals to guide subsequent synthesis rounds.

Beyond filtering, the Perception stage actively encourages the synthesis model to propose constraint types that are uniquely suited to each image, even when the current registry already provides adequate coverage. This image-driven discovery enriches the constraint space with fine-grained, visually grounded requirements that a static registry would not express, directly improving the diversity of synthesized instructions. Each proposed type is paired with a candidate verification function and must pass a sandboxed validation test before admission to $\tau ,$ ensuring that every newly discovered constraint remains reliably verifiable in subsequent iterations.

## 3.3 Planning & Execution

Given the image-compatible constraint subset ${ \cal { S } } _ { I } ,$ a naive planner would repeatedly sample similar constraint combinations and generate near-duplicate instructions across iterations—a form of data homogenization that limits the diversity and informativeness of the training corpus. VISA addresses this by jointly managing diversity at two levels: the constraint level, guided by memory signals, and the instruction level, guided by embedding-space distance. Specifically, constraint sampling weights each type in $ { \boldsymbol { S } } _ { I }$ by two signals maintained in persistent memory—a coverage signal that down-weights types already well-represented in the accepted corpus, and a difficulty signal that up-weights types whose constraints the target model has historically struggled to satisfy (derived from the difficulty probing described in reflection)—concentrating synthesis on combinations that are simultaneously underrepresented and challenging.

To alleviate data homogenization, VISA separates instruction planning from response generation. For each image, the planner first generates k candidate instructions with sampled constraint sets, without immediately producing the corresponding responses. Each candidate instruction is then embedded and compared with the global embedding pool $\mathcal { P }$ of previously accepted instructions stored in memory. Following a greedy diversity selection strategy, VISA selects the candidate whose nearest neighbor in $\mathcal { P }$ is farthest away:

$$
i ^ { * } = \arg \operatorname* { m a x } _ { i \in [ k ] } \operatorname* { m i n } _ { p \in \mathcal { P } } \mathrm { d i s t } ( e _ { Q _ { i } } , e _ { p } ) ,
$$

where $e _ { Q _ { i } }$ denotes the embedding of candidate instruction $Q _ { i } , e _ { p }$ denotes an instruction embedding in ${ \mathcal P } ,$ , and dist $( \cdot , \cdot )$ measures the distance between two embeddings. Only the selected instruction $Q _ { i ^ { * } }$ is passed to the response generator to produce $R _ { i }$

## 3.4 Reflection

After planning and execution, the Reflection stage determines whether a generated sample should be accepted, repaired, or discarded. Given a sample $( I , Q , R , C )$ , VISA evaluates each constraint $c \in C$ through its bound verifier: deterministic constraints—word count, keyword inclusion, output format—are checked by executable code tools, while semantic and visually grounded constraints are assessed by structured LLM judges, both sharing a unified output interface that reports a binary pass/fail verdict alongside diagnostic information localizing each failure. When a sample fails, VISA uses these per-constraint diagnostics as structured feedback to select the most appropriate recovery strategy—applying local response edits for surfacelevel violations, or escalating to instruction rewrites and constraint simplification when the failure reflects a deeper incompatibility between the instruction and the image—rather than discarding the sample outright. If the verification score stops improving across consecutive attempts, VISA discards the sample once the budget is exhausted, preventing the degenerate behavior in which repeated selfcorrection converges to an incorrect solution.

For samples that pass verification, VISA estimates their informativeness by probing the target model M: given the accepted instruction $Q$ and image I, M generates a response that is then evaluated against the same constraint set used during synthesis. The per-constraint outcomes are aggregated into a difficulty score $d \in [ 0 , 1 ]$ , where a lower score indicates that M already satisfies most constraints and thus gains little from the sample, while a higher score signals unresolved failures that provide more informative supervision. Both the aggregate score and the per-constraint failure pattern are written to memory, subsequently shifting the constraint sampling weights in the Planning stage toward types that remain challenging for M and directly linking each synthesis round to the current capability profile of the target model.

## 3.5 Memory

A defining characteristic of most data synthesis pipelines is that they are stateless: each sample is generated independently, and the synthesis process itself does not improve as more data is produced. VISA breaks this assumption by maintaining a persistent memory M that accumulates the outcomes of every synthesis iteration and makes them available to all subsequent rounds. Specifically, M stores the accepted samples together with their constraint sets and verifier handles, the instruction embeddings that populate the diversity pool $\mathcal { P } _ { : }$ , the per-constraint difficulty profiles derived from target-model probing, and the constraint types discovered during Perception along with their associated verification functions.

This accumulated state feeds directly back into every stage of the next iteration, giving rise to VISA’s self-evolving behavior. The instruction embedding pool steers the diversity selector away from redundant instructions; the difficulty profiles shift constraint sampling weights toward types that remain unresolved for the target model; and newly discovered constraint types become immediately available for future synthesis. As a result, each iteration begins with strictly more information than the last—about what has already been generated, where the target model still struggles, and what constraints the system can express—so that subsequent rounds can exploit increasingly informative synthesis state without external intervention.

## 4 Reinforcement Learning with Verifiable Rewards

Recent studies on reinforcement learning with verifiable rewards (RLVR) show that rule-based or programmatically checkable signals can guide policy optimization without relying on separately trained reward models (Wen et al., 2025; Cai et al., 2025). This idea has also been adopted in recent reasoning models such as DeepSeek-R1 (Guo et al., 2025), where Group Relative Policy Optimization (GRPO) optimizes model responses using verifiable reward signals for tasks with clearly defined correctness criteria (Shao et al., 2024). These methods suggest that verifier-derived rewards can provide a more transparent and task-aligned alternative to separately learned reward models when the desired behavior can be checked by explicit rules or reliable evaluators.

VISA naturally fits this paradigm because the agent binds each accepted instruction to explicit verifier contracts during execution. The same verification logic used to accept and repair synthetic data can therefore be reused during RL. Given an image I, an instruction $Q ,$ a model response $R ,$ and the constraint set $C = \{ c _ { 1 } , \ldots , c _ { m } \}$ embedded in the instruction, each constraint $c _ { j }$ is evaluated by its bound verifier $v _ { j }$ . We let $v _ { j } ( \bar { I } , Q , R ) \in \{ 0 , 1 \}$ denote whether the response satisfies $c _ { j }$ , and define the reward as the constraint pass rate:

$$
r ( I , Q , R , C ) = \frac { 1 } { | C | } \sum _ { j = 1 } ^ { | C | } v _ { j } ( I , Q , R ) .
$$

This reward directly measures the fraction of instruction constraints followed by the model response, aligning RL optimization with multimodal instruction following. Since the reward is computed from verifier contracts already maintained by the synthesis agent, VISA does not require an additional reward model, and failed constraints can be traced to specific sources such as formatting errors, missing keywords, or incorrect visually grounded descriptions.

## 5 Experimental Setup

## 5.1 Generated Training Data

We synthesize a dataset of approximately 15k samples using VISA, based on the image pool provided by MM-IFInstruct<sup>1</sup>. Each generated sample includes an image, a corresponding instruction, the model’s response, and the associated verification method. This dataset is used for fine-tuning of multimodal instruction-following models in our experiments.

## 5.2 Models

We use the Qwen3.5<sup>2</sup> series as the primary model family in our experiments. Specifically, Qwen3.5- 27B serves as the synthesis agent in VISA, while Qwen3.5-4B is used as the target model for supervised fine-tuning (SFT) and RL. We also include Qwen3.5-9B as a stronger same-family reference model. For comparison with contemporary opensource MLLMs, we report results of MiniCPM-V-4.5 (Yu et al., 2025) and InternVL3.5 (Wang et al., 2025) at comparable model scales.

## 5.3 Training Details

All experiments are conducted on a cluster of 128 L40s GPUs. The visual encoder layers are frozen, and only the LLM backbone and the alignment module are fine-tuned. We adopt a learning rate of $1 \times 1 0 ^ { - 5 }$ and perform training in bfloat16 precision to balance memory efficiency and numerical stability. To efficiently scale across multiple GPUs, we utilize DeepSpeed (Rajbhandari et al., 2020) ZeRO-3 for parallelized training, enabling effective model updates across the large distributed setup. SFT is implemented using swift (Zhao et al., 2025), while RL stages leverage verl (Sheng et al., 2024), with rollout size set to 8 and GPU memory utilization at 0.8.

## 5.4 Evaluation Benchmarks

We evaluate VISA from two perspectives: multimodal instruction following and general multimodal capability. For instruction following, we use MM-IFEval (Ding et al., 2025), which evaluates both compose-level (C-Level) constraints on output responses and perception-level (P-Level) cases grounded in input images. We report C-Level, P-Level, and average accuracy following the original protocol. For general capability evaluation, we report results on MMBench (Liu et al., 2024a), MM-Star (Chen et al., 2024), MM-Vet (Yu et al., 2023), HallusionBench (Guan et al., 2024), MathVista (Lu et al., 2024), OCRBench (Liu et al., 2024b), and AI2D (Kembhavi et al., 2016). These benchmarks cover broad multimodal perception and reasoning, integrated vision-language abilities, hallucination robustness, visual mathematical reasoning, optical character recognition (OCR)-centric understanding and scientific diagram understanding. To ensure a fair comparison with the base model and avoid confounding improvements from explicit reasoning traces, all benchmark evaluations are conducted in the non-thinking mode.

<table><tr><td>Model / Training</td><td>Params</td><td>MM-IFEval C-Level P-Level</td><td>Avg.</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>63.9 60.0</td><td>63.0</td></tr><tr><td>MiniCPM-V-4.5</td><td>8B</td><td>68.8</td><td>44.0 62.6</td></tr><tr><td>InternVL3.5-8B</td><td>8B</td><td>61.8</td><td>39.0 56.1</td></tr><tr><td>InternVL3.5-4B</td><td>4B</td><td>57.0</td><td>33.0 51.0</td></tr><tr><td>Qwen3.5-4B</td><td>4B</td><td>62.7</td><td>55.0 60.8</td></tr><tr><td colspan="4">Qwen3.5-4B</td></tr><tr><td>+ MM-IFInstruct-23k</td><td>4B</td><td>65.7</td><td>46.0 60.8</td></tr><tr><td>+ MM-IFDPO-23k</td><td>4B</td><td>65.0 52.0</td><td>61.7</td></tr><tr><td>+ VISA-SFT-15k</td><td>4B</td><td>67.2</td><td>54.0 63.9</td></tr><tr><td>+ VISA-RL-15k</td><td>4B</td><td>66.9</td><td>59.0 64.9</td></tr></table>

Table 1: Main results on MM-IFEval. We report C-Level accuracy, P-Level accuracy, and their average score; higher values indicate better instructionfollowing performance.

## 6 Results

## 6.1 Main Results

Table 1 presents the main results on MM-IFEval. Among the baselines, Qwen3.5-9B achieves an average score of 63.0, while the 4B-scale models (MiniCPM-V-4.5, InternVL3.5-4B, and Qwen3.5- 4B) score 62.6, 51.0, and 60.8, respectively. Training Qwen3.5-4B on MM-IFInstruct-23k brings only limited improvement, suggesting that simply increasing data volume without targeted synthesis is insufficient. In contrast, VISA-SFT-15k achieves an average score of 63.9, surpassing the 9B reference model on the average score with less than half the parameters and using only 15k training samples. VISA-RL-15k further improves the average score to 64.9, with a notable gain on P-Level (59.0), reducing the performance gap between constraintoriented and perception-oriented cases.

<table><tr><td>Method (Qwen3.5-4B)</td><td>C-Level</td><td>P-Level</td><td>Avg.</td></tr><tr><td>Base</td><td>62.7</td><td>55.0</td><td>60.8</td></tr><tr><td>Static Pipeline</td><td>64.9</td><td>46.0</td><td>60.2</td></tr><tr><td>+ Reflection</td><td>65.6</td><td>48.0</td><td>61.2</td></tr><tr><td>+ Reflection + Memory</td><td>67.1</td><td>51.0</td><td>63.1</td></tr><tr><td>Full VISA</td><td>67.2</td><td>54.0</td><td>63.9</td></tr></table>

Table 2: Ablation results on MM-IFEval with Qwen3.5- 4B under SFT, showing the incremental contribution of reflection, memory, and the full VISA pipeline.

MM-IFEval contains 300 C-Level samples that assess multi-constraint satisfaction in model outputs, and 100 P-Level samples that require visually grounded understanding akin to visual question answering (VQA), where instruction-following ability is less likely to be the only bottleneck. As a result, P-Level gains are less directly attributable to instruction-following supervision alone, since they also depend on the model’s underlying visual perception and grounding ability. Nevertheless, VISA-RL-15k attains a P-Level score of 59.0, outperforming all 4B-scale baselines and approaching Qwen3.5-9B (60.0), suggesting that difficultyaware constraint sampling and verifier-based RL provide useful supervision even for perceptionheavy instructions.

## 6.2 Ablation Study

To understand the contribution of each component in VISA, we conduct an ablation study on MM-IFEval using Qwen3.5-4B as the target backbone under the SFT setting. As shown in Table 2, the static pipeline improves C-Level but noticeably degrades P-Level, leading to a lower average score than the base model. This suggests that naively generating constraint-rich data can improve outputconstraint adherence while harming perceptionoriented behavior. Adding the Reflection module, which performs diagnostic-guided repair of failed samples, improves the average score from 60.2 to 61.2, indicating that bounded recovery enhances per-sample reliability. Incorporating persistent memory further raises the average score to 63.1, with consistent gains on both C-Level and P-Level, confirming that coverage and difficulty signals accumulated across rounds help steer synthesis toward underrepresented and challenging constraints. The full VISA system achieves the best performance, reaching 67.2 on C-Level, 54.0 on P-Level, and 63.9 on average. Compared with the static pipeline, the largest numerical recovery appears on P-Level, suggesting that reflection and memory help reduce the perception-side degradation introduced by naive synthesis. However, since P-Level cases are more perception-oriented and closer to VQA-style evaluation, the C-Level gains provide more direct evidence that VISA improves compositional instruction following. These results show that the components contribute complementarily: Reflection improves local sample quality, Memory guides global constraint selection, and the remaining modules further enhance instruction diversity and difficulty targeting.

## 6.3 General Capability Evaluation

Table 3 reports results across seven general multimodal benchmarks. VISA-SFT-15k improves the average score from 70.5 to 71.0, with gains on MMBench (+2.8) and MMStar (+1.5), and only a minor drop on HallusionBench (−1.9). These results suggest that instruction tuning on VISA data broadly preserves, and in some cases slightly enhances, general multimodal perception and reasoning. VISA-RL-15k achieves the strongest overall performance with an average score of 72.9, improving over the base model on five out of seven benchmarks. The largest gain appears on MathVista (+9.1), which may partly reflect the alignment between verifier-derived optimization and structured visual reasoning tasks, as MathVista emphasizes mathematical reasoning in visual contexts. OCR-Bench and AI2D also improve under RL, while HallusionBench also recovers to 57.5 compared with the SFT variant, suggesting that RL does not exacerbate the mild hallucination-related drop observed after SFT. Together, these results show that VISA improves instruction adherence without sacrificing general multimodal capability, and that the verifier-derived reward signal supports transferable improvements across diverse evaluation settings.

## 7 Analysis

## 7.1 Dynamic Constraint Expansion

Figure 2 analyzes how the agent’s constraintverifier registry grows during data synthesis. We use constraint category to denote a high-level group such as attribute, spatial, or format, and constraint type to denote a fine-grained requirement within a category, such as word\_count, json\_format, or spatial\_relation. Starting from 76 hand-crafted constraint types, VISA expands the registry during perception. For each image, the system analyzes what kinds of requirements the image can support and may propose new constraint types when the current state does not cover a useful image-grounded requirement. Each new type is linked to either a deterministic code verifier or a structured LLM judge before it is added to the registry. Across 126 self-evolution rounds, this process adds 234 new constraint types, producing a final library of 310 types, a 4.1× increase without additional human annotation. Appendix B lists the corresponding constraint categories and representative types.

<table><tr><td>Model / Training</td><td>MMBench MMStar MM-Vet HallusionBench MathVista OCRBench AI2D</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td></tr><tr><td>Qwen3.5-4B</td><td>78.4</td><td>61.7</td><td>47.6</td><td>56.0</td><td>78.8</td><td>84.9</td><td>86.2</td><td>70.5</td></tr><tr><td>+ VISA-SFT-15k</td><td>81.2</td><td>63.2</td><td>49.1</td><td>54.1</td><td>77.8</td><td>85.2</td><td>86.4</td><td>71.0</td></tr><tr><td>+ VISA-RL-15k</td><td>78.8</td><td>62.5</td><td>50.5</td><td>57.5</td><td>87.9</td><td>85.6</td><td>87.6</td><td>72.9</td></tr></table>

Table 3: General multimodal capability results on Qwen3.5-4B. VISA preserves or improves broad multimodal performance across seven public benchmarks, with all evaluations conducted in non-thinking mode.

![](images/452732529e2c9679145f73ce019619cdd2b4c97c3ea0c2feaaab073d90d6d34d.jpg)

![](images/12ebe1fbd570adb0a54d031f5507767516e13d64ecd2ee078d8e464fa339e39f.jpg)  
Figure 2: Evolution of the constraint-verifier registry in VISA. Left: cumulative growth of constraint types across self-evolution rounds, showing that most discoveries occur early and gradually saturate. Right: category distribution of the final 310 constraint types.

The left panel of Figure 2 shows that most new constraint types are found in the early rounds. After this initial stage, the number of newly added types drops quickly, and the cumulative curve becomes almost flat. This pattern suggests that the agent first discovers common visual requirements, such as object attributes, spatial relations, and response structure, and then only occasionally adds rare cases in later rounds. In other words, the registry grows quickly at the beginning but does not keep expanding without limit. The right panel shows that the final registry is mainly composed of attribute, spatial, and structural categories. This means that the newly added types are mostly related to visual content and response organization, rather than simple text-only rules such as keyword matching or output format.

## 7.2 Diversity and Difficulty

We analyze the 15,668 samples produced by VISA from the perspectives of diversity and difficulty. Figure 3 shows that the data covers a broad joint constraint space. All 90 off-diagonal category pairs appear at least once, and the strongest cooccurrences follow intuitive visual dependencies: object\_grounded, counting, and spatial cooccur with attribute in 85–92% of samples, while causal pairs more often with format (87%) and structural (61%) than with spatial (49%). The instruction embedding view further shows a continuous cloud rather than a few isolated template clusters, with a mean pairwise cosine distance of 0.977. This supports the role of the embedding pool in planning: new instructions are selected against previous accepted instructions, reducing repeated surface forms. The full constraint-count distribution is provided in Figure 4 in Appendix A; it is centered around four to six constraints per sample (mean 4.87, median 5, standard deviation 2.12), confirming that most examples combine several jointly satisfiable requirements instead of testing a single rule.

![](images/9b2c0b45b38c3b48f29b77a6056f6a2ab13df1ce94362710ea94bb2626fe556b.jpg)

![](images/35016e1b2cbcc61a6f8514229eb889787d060050a5bc6bd0c57724226a2caa4c.jpg)  
Figure 3: Diversity analysis of VISA-synthesized instructions. Left: conditional co-occurrence among the ten most frequent constraint categories, where each cell reports $P ( j \mid i )$ as a percentage. Right: t-distributed stochastic neighbor embedding (t-SNE) projection of high-dimensional instruction embeddings, colored by dominant constraint category.

We also use the target-model probe to estimate how much training signal these samples contain. Across all samples, 82.6% are labeled easy, 15.9% medium, and 1.5% hard, with 5.6% of constraint instances flagged as failures by the verifier track. This skewed distribution is expected, since the target model already possesses strong basic multimodal instruction-following ability, making many verified samples easy under the probing protocol. Rather than enforcing a balanced sample-level difficulty distribution, VISA uses difficulty signals to identify the remaining failure modes on top of this strong base capability. As a result, easy cases preserve broad coverage and stabilize common behaviors, while medium and hard cases contribute roughly 2,730 examples with at least one verifier-detected failure that can further improve the model’s upper-bound performance through SFT and verifier-derived RL. These failures are concentrated in counting (11.6%), lexical (10.2%), and format (8.1%), whereas style (0.16%) and conditional (0%) are almost always satisfied. This gives later synthesis rounds a clear signal about which constraint types remain difficult, without sacrificing broad coverage or verifier reliability. While VISA introduces additional synthesis-time computation compared with static pipelines, this cost is incurred offline during data construction and is partly amortized by parallel candidate generation and batched verification; we discuss this trade-off

in Appendix C.

## 8 Conclusion

We presented VISA, an agentic framework for self-evolving multimodal instruction data synthesis. By coupling perception, planning and execution, reflection, and memory update, VISA turns static generate-and-filter pipelines into adaptive feedback-driven synthesis. Its persistent memory accumulates verifier outcomes, accepted samples, diversity signals, and target-model difficulty profiles, allowing later rounds to expand the constraint space and focus on unresolved weaknesses.

Experiments with Qwen3.5 target models demonstrate that VISA improves multimodal instruction following while preserving broad multimodal capabilities. VISA-SFT-15k surpasses both the base model and data-synthesis baselines on the average MM-IFEval score, while VISA-RL-15k further improves the average MM-IFEval score to 64.9. Across seven general multimodal benchmarks, VISA-RL-15k improves the average score from 70.5 to 72.9, showing that stronger instruction adherence does not come at the cost of general perception and reasoning ability. The same verifier contracts also serve directly as RL reward signals, requiring no separately trained reward model. We hope VISA provides a useful step toward adaptive, feedback-driven data synthesis for multimodal instruction-following models.

## Limitations

VISA primarily focuses on improving multimodal instruction synthesis through agentic iteration. As a result, the current framework prioritizes synthesis quality and difficulty adaptation over aggressive efficiency optimization. Although most synthesis stages can be parallelized across samples and constraints, state updates, iterative recovery, and target-aware probing still introduce additional overhead compared with static one-shot pipelines. We view this trade-off as acceptable in offline data construction, where synthesis quality is the primary objective.

## Ethics Statement

This work studies synthetic data generation for multimodal instruction following. The generated data may inherit biases, unsafe associations, or privacysensitive content from the source images and the MLLM used for synthesis. Although VISA uses image-aware constraint selection, verification, and reflection feedback to improve constraint satisfaction, these mechanisms are not complete safety or fairness filters. VISA is designed for research on multimodal instruction tuning and agentic data synthesis; models trained with its data should be audited before deployment, especially in high-stakes settings, and all source images and generated artifacts should be used in accordance with the licenses of the underlying datasets.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, and Masashi Sugiyama. 2025. Reinforcement learning with verifiable yet noisy rewards under imperfect verifiers. arXiv preprint arXiv:2510.00915.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and 1 others. 2024. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087.

Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. 2025. Mmifengine: Towards multimodal instruction following.

In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1099–1109.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, and 1 others. 2024. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14375–14385.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Weilei He, Feng Ju, Zhiyuan Fan, Rui Min, Minhao Cheng, and Yi R Fung. 2026. Empowering reliable visual-centric instruction following in mllms. arXiv preprint arXiv:2601.03198.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Steven Yau, Zijuan Lin, Liyang Zhou, and 1 others. 2024. Metagpt: Meta programming for a multiagent collaborative framework. In International Conference on Learning Representations, volume 2024, pages 23247–23275.

Xin Huang, Jihao Liu, Jinliang Zheng, Boxiao Liu, Jia Wang, Yu Liu, Hongsheng Li, and Osamu Yoshie. 2025. Mm-instruct: Generated visual instructions for large multimodal model alignment. Neurocomputing, page 131164.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and 1 others. 2024. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024a. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. 2024b. Ocrbench: on the hidden mystery of ocr in large multi-

modal models. Science China Information Sciences, 67(12):220102.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2024. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, volume 2024, pages 23439–23554.

Run Luo, Haonan Zhang, Longze Chen, Ting-En Lin, Xiong Liu, Yuchuan Wu, Min Yang, Yongbin Li, Minzheng Wang, Pengpeng Zeng, and 1 others. 2025. Mmevol: Empowering multimodal large language models with evol-instruct. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19655–19682.

Yusu Qian, Hanrong Ye, Jean-Philippe Fauconnier, Peter Grasch, Yinfei Yang, and Zhe Gan. 2025. Miabench: Towards better instruction following evaluation of multimodal llms. In International Conference on Learning Representations, volume 2025, pages 35145–35165.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, and 1 others. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. In International Conference on Learning Representations, volume 2024, pages 9695–9717.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: Memory optimizations toward training trillion parameter models. In SC20: international conference for high performance computing, networking, storage and analysis, pages 1–16. IEEE.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Qwen Team. 2026. Qwen3. 5-omni technical report. arXiv preprint arXiv:2604.15804.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, and 1 others. 2025. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language

models with self-generated instructions. In Proceedings ofthe 61st annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 13484–13508.

Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, and 1 others. 2025. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, and 1 others. 2025. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.

Wenqi Zhang, Zhenglin Cheng, Yuanyu He, Mengna Wang, Yongliang Shen, Zeqi Tan, Guiyang Hou, Mingqian He, Yanna Ma, Weiming Lu, and 1 others. 2024. Multimodal self-instruct: Synthetic abstract image and visual reasoning instruction using language model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19228–19252.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, and 1 others. 2025. Swift: a scalable lightweight infrastructure for fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 29733–29735.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

## A Additional Analysis

Figure 4 shows the full distribution of the number of constraints attached to each synthesized sample.

![](images/28a3ef634e9832118f5ac0b4d50c9b80e1fed8f77ff388cf0dfd4836b4fe3364.jpg)  
Figure 4: Distribution of the number of constraints per synthesized sample, showing that most accepted samples combine multiple jointly satisfiable requirements rather than a single constraint.

## B Constraint Details

The constraint registry is organized at two granularities. Constraint categories provide coarse groups for analysis and diversity control, while constraint types are the atomic units sampled by the planner and bound to verifiers. Table 4 summarizes the final registry after self-evolution. “Seed” counts the hand-crafted types available at initialization, and “+Disc.” counts types added through dynamic discovery after verifier binding. The largest expansions appear in visually grounded and structural categories, reflecting VISA’s emphasis on imagecompatible requirements and response organization.

The seed constraint registry is initialized with common instruction-following constraint families studied in prior IF benchmarks, including IFEval and MM-IFEngine/MM-IFEval. VISA extends this initial registry through image-conditioned discovery and verifier binding. Unlike a fixed benchmark taxonomy, VISA maintains the registry as an evolving synthesis-time object whose entries can be filtered, discovered, validated, and reused across self-evolution rounds.

## C Discussion on Synthesis Efficiency

The main cost of VISA comes from replacing single-pass generation with an agentic loop. In our implementation, the planner generates four candidate instructions for each image during the greedy selection stage, and we set the multi-threaded concurrency to 16. With a single synthesis model deployed on L40s GPUs, the end-to-end synthesis time is approximately 20 seconds per accepted sample on average.

Several design choices help amortize this cost. Perception is performed once per image; the four candidate instructions are generated in parallel before the more expensive response-generation stage; code-based verifiers are lightweight and parallelizable; and LLM-track constraints are packed into batched judge calls whenever possible. The bounded recovery mechanism also avoids unbounded self-correction. Among the 15,668 accepted samples, 10,230 samples (65.3%) pass verification without recovery, 2,241 samples (14.3%) require one recovery attempt, 2,174 samples (13.9%) require two attempts, and 1,023 samples (6.5%) reach three attempts. This shows that most accepted samples incur no recovery overhead, while only a small fraction requires the maximum repair budget.

The efficiency trade-off is therefore best understood per accepted and target-informative sample rather than per raw generated sample. A static pipeline may generate more candidates per unit time, but many candidates can be redundant, incompatible with the image, unverifiable, or already easy for the target model. In contrast, VISA spends more computation before admission, but each accepted sample is associated with explicit constraint sets, verifier handles, difficulty scores, and recovery diagnostics. These metadata are not discarded: they update the agent state and guide subsequent synthesis rounds. To assess the reliability of the LLMbased judge track, we further conduct a manual audit on a sampled subset of judge decisions and observe approximately 95% agreement with human annotations, suggesting that the verifier feedback used by the agent is largely consistent with human

<table><tr><td>Category</td><td colspan="4">Representative constraint types</td><td>Seed +Disc.</td><td>Total</td></tr><tr><td>format</td><td>word_count, sentence_count, xml_format, yaml_format, punctuation_control,</td><td>paragraph_count, markdown_format, number_precision,</td><td>json_format, list_format, quotation,</td><td>23</td><td>+1</td><td>24</td></tr><tr><td>lexical</td><td colspan="4">placeholder_count, flexible_layout must_include_keyword, must_exclude_keyword, keyword_frequency, keyword_in_first_sentence, alliteration, acronym_constraint, word_diversity, synonym_substitution</td><td>+0</td><td>16</td></tr><tr><td>structural</td><td colspan="4">section_headers, prefix_suffix, paragraph_start, paragraph_end, sentence_pattern, logical_connector, summary,</td><td>+53</td><td>66</td></tr><tr><td>style</td><td colspan="4">argument_structure, postscript tone, perspective, language, tense, rhetorical_device, 11 audience_adaptation, role, scenario_framing</td><td>+2</td><td>13</td></tr><tr><td>attribute</td><td colspan="4">attribute_describe, scene_mood, size_compare, visual_include, 5 visual_exclude</td><td>+105</td><td>110</td></tr><tr><td>spatial</td><td colspan="4">spatial_relation</td><td>+70 +1</td><td>71 2</td></tr><tr><td>counting</td><td colspan="4">object_count</td><td>+0</td><td>1</td></tr><tr><td>object_grounded</td><td colspan="4">object_describe</td><td>+1</td><td>2</td></tr><tr><td>ocr_grounded</td><td colspan="4">ocr_content</td><td>+1</td><td>3</td></tr><tr><td>causal</td><td colspan="4">causal_explain, temporal_sequence</td><td>+0</td><td>2</td></tr><tr><td>conditional</td><td colspan="4">conditional_if, hypothetical</td><td></td><td>310</td></tr><tr><td colspan="4">Total constraint types</td><td>76</td><td>+234</td><td></td></tr></table>

Table 4: Constraint categories in the final VISA registry. Representative types are illustrative examples; “Seed” and “+Disc.” denote initially defined and dynamically discovered constraint types, respectively.

judgment.

## D Tool Registry Interface

Each verification tool in VISA’s code-verifier track follows a compact interface contract: a metadata block links the tool to a constraint category and type, a verify function maps a response and typed parameters to a standardized pass/fail record, and a small set of test cases is executed before the tool enters the registry. The same contract is used for both seed verifiers and code verifiers attached to dynamically discovered constraint types. The schematic below illustrates the interface with a shortened JSON-format verifier; implementation details are omitted because they are tool-specific and not central to the framework.

```python
TOOL_META = {
"tool_id": "verify_json_format",
"description": "Check JSON format",
"constraint_categories": ["format"],
"constraint_types": ["json_format"],
"input_schema": {
"response": {
"type": "str",
"required": True,
"desc": "Response text"
},
"required_keys": {
"type": "list",
"required": False,
"desc": "Required JSON keys"
}
},
"output_schema": {
"pass": {"type": "bool"},
"expected": {"type": "str"},
```

"actual": {"type": "str"},   
"detail": {"type": "str"}   
}   
}   
def verify(response: str, \*\*params) -> dict:   
return {   
"pass": <bool>,   
"expected": <str>,   
"actual": <str>,   
"detail": <str>   
}   
TEST\_CASES = [   
{"input": {"response": '{"a": 1}'},   
"expected\_pass": True},   
{"input": {"response": "not json"},   
"expected\_pass": False},   
]

## E Prompt Examples

This appendix provides abbreviated prompt templates for the main agent calls in VISA. We omit implementation-specific formatting instructions and long field lists; placeholders such as {...} are filled by the controller at runtime.

Perception and constraint-type discovery. Issued at the beginning of every iteration to the multimodal teacher together with the input image. It returns both the perception profile of the image and, when warranted, proposals for new constraint types.

tured profile. Summarize the visible scene, estimate its complexity, and mark any visiongrounded constraint typesfrom {registered types} that the image cannot support.

If the registered library misses a useful requirementfor this image-conditioned instruction, propose up to {max\_discover} new constraint types with their category, verifier track, and short description. Do not propose a new type when an existing parameterized type already covers the requirement.

Constraint selection. Given the perception profile and the candidate constraint types compatible with the image, the planner selects a budgeted, mutually consistent constraint set.

Select {constraint\_budget} compatible constraint types from {candidate types}. Use the priority weightsfrom the agent state as a soft guide, while avoiding redundant choices from a single category.

The selected constraints must be jointly satisfiable and verifiable. For each selected type, write a short description: vision-grounded constraints should refer to objects by role or category without revealing the answer, and text-only constraints should state the required response property directly.

Instruction composition. Once a constraint set is fixed, the agent rewrites it as a single naturallanguage user instruction.

Write a single natural user requestfor the image. The request should be fluent and self-contained, and it should integrate all selected constraints without listing them mechanically.

Do not reveal visual evidence that the model should inferfrom the image. Ifa constraint specifies the response format, ask for that format in natural language rather than providing an answer skeleton.

Verifier parameter extraction. For constraints bound to a code verifier, a lightweight extraction prompt fills the verifier’s typed parameters from the natural-language description.

For each code-verifiable constraint, extract only the parameters explicitly supported by its verifier schema. Preserve the required data types and return one parameter object per constraint.

Ifa value is not specified or not relevant, set it to null; do not infer missing valuesfrom common sense orfrom the image.

Coverage self-check. Before full verification, a judge LLM checks whether the composed instruction explicitly references every selected constraint, and feeds any missing items back to the recovery loop.

Check whether the composed instruction explicitly covers every selected constraint. For each missing constraint, return its identifier and a brief repair suggestion.

Do not count a constraint as covered merely because it is implicit, conventional, or likely to be satisfied by default.