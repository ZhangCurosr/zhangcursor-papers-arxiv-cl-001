# Collaborative Reasoning Distillation via Cross-Feedback and Coherent Curation

Taehoon Kim<sup>1</sup>, Seunggeun Cho<sup>1</sup> and Dongsu Han<sup>1</sup> <sup>1</sup>KAIST

Reasoning capabilities are critical for advancing Large Language Models, yet current approaches either require massive computational budgets or struggle to efectively distill reasoning to smaller models. Standard distillation methods rely on outcome-based rewards, failing to distinguish between sound reasoning and lucky guesses. We propose Collaborative Reasoning Distillation (CRD), a framework that enhances reasoning in compact models through three innovations: (1) interactive cross-feedback where teachers iteratively critique each other’s reasoning, (2) fine-grained step-wise quality assessment capturing logical validity independent of final answers, and (3) coherence-aware step stitching that synthesizes complementary strengths. Students are trained via Reasoning Quality Optimization (RQO) with budget constraints. Our model, CRD-4B, achieves 97.3% on MATH-500 and 70.3% on AIME’25, surpassing baselines while using only 50K training examples, up to 12 times smaller than the datasets of comparable models.

## 1. Introduction

Large Language Models (LLMs) have exhibited impressive reasoning capabilities (Wei et al., 2023). However, advancing reasoning abilities requires massive computational investment. Cutting-edge models demand thousands of GPUs (Jiang et al., 2024), pushing estimated hardware costs into the \$800 million range (Cottier et al., 2025). Although training smaller models from scratch appears cost-efective, the resulting systems almost always trail far behind in reasoning quality despite significant resource outlays (Fu et al., 2023; Magister et al., 2023).

Knowledge Distillation (KD) (Hinton et al., 2015) ofers a practical framework for transferring capabilities from large models to compact students. While KD has been widely applied to vision and general NLP tasks, transferring complex reasoning capabilities through distillation remains challenging (Li et al., 2022; Tian et al., 2024). The student’s reasoning ability is inherently upper-bounded by the teacher, and any limitations or reasoning flaws in the teacher are inevitably inherited by the student, creating a ceiling efect that constrains achievable performance.

Fortunately, the growing ecosystem of diverse LLMs creates an opportunity. Diferent models exhibit distinct strengths across reasoning domains. Some excel at mathematical deduction, others at symbolic manipulation, logical analysis, or domain-specific reasoning. While recent work has explored leveraging multiple teachers for distillation (Tian et al., 2024), existing approaches rely on simple aggregation or ensemble voting, without mechanisms to identify and combine high-quality reasoning steps across teachers.

The core challenge is that standard distillation methods derive learning signals from binary correctness or format compliance, ignoring the logical validity of intermediate steps (Stanton et al., 2021). This allows reasoning shortcuts and logical hallucinations to persist as long as they coincidentally yield correct answers. While Chain-of-Thought (CoT) prompting (Wei et al., 2023) provides explicit reasoning paths useful for distillation (Li et al., 2022), single-teacher CoT sufers from limited diversity, bias propagation, and inconsistent quality. Recent approaches like Group Relative Policy Optimization (GRPO) (Shao et al., 2024) have made progress in aligning model outputs. Advanced reasoning models such as DeepSeek-R1 (Guo et al., 2025) demonstrate that extended test-time computation enhances reasoning capabilities. However, longer reasoning chains substantially increase inference latency and computational cost, limiting deployment in latency-sensitive applications.

To address this limitation, we propose a complementary approach. Rather than requiring extensive test-time computation, we distill insights from teacher reasoning by assessing the logical quality of intermediate steps and synthesizing high-quality reasoning patterns across multiple teachers. This allows student models to achieve comparable reasoning quality with far fewer final training trajectories and shorter reasoning at inference.

![](images/f5f2bdf0d42930510a430e46ff255a316ab8e3ca832d0c34cfda50675caf979b.jpg)  
Figure 1: Overview of the CRD Framework. Multiple teachers iteratively critique and refine each other’s reasoning through Cross-Feedback. An LLM-as-a-Judge then assigns step-wise quality scores, and the Thinking Path algorithm stitches high-quality steps across teachers while pruning flawed steps. The curated dataset is used to train students via RQO with Optimal Budget curriculum.

To achieve this vision, we introduce Collaborative Reasoning Distillation (CRD), a framework combining multiple teachers with complementary strengths, assessing step-level reasoning quality, and synthesizing high-quality reasoning patterns. Our key contributions are:

∙ Interactive Multi-Teacher Cross-Feedback: An iterative mechanism where diverse teachers critique and refine each other’s reasoning, producing a richer candidate pool with corrected errors and complementary solution strategies.

∙ Fine-grained Step-wise Quality Assessment: A dual-judge system (JudgeLM and xVerify) that evaluates the logical validity of each reasoning step independent of the final answer. Unlike binary outcome-based rewards, this system assigns granular quality scores to each step, enabling the identification of high-quality reasoning patterns even within incorrect solutions.

∙ Coherence-Aware Step-Level Stitching: A Viterbi-style algorithm that constructs hybrid reasoning chains by selecting high-quality steps across multiple teachers, enabling solutions no single teacher could produce alone.

Together, these three components produce a curated training dataset rich in diverse, high-quality reasoning examples with explicit step-level quality signals. We train students on this dataset using Reasoning Quality Optimization (RQO) with curriculum-based budget constraints to achieve both reasoning quality and inference eficiency.

CRD substantially improves reasoning across mathematical, scientific, and programming benchmarks. On the DeepSeek-R1-Distill-Qwen-1.5B student, CRD-1.5B achieves +7.2% improvements on MATH-500, +6.8% on GPQA Diamond, and +10.3% on LiveCodeBench. On Qwen3 students, CRD-1.7B gains +2.0% on AIME’24 and +5.4% on AIME’25 over same-parameter baselines. Crucially, CRD achieves these results using only 50K training examples, demonstrating that high-quality step-level curation can make reasoning distillation substantially more data-eficient.

## 2. Methodology

Current reasoning distillation methods primarily rely on imitation learning from a single teacher or outcomebased reinforcement learning (e.g., DeepSeek-R1, GRPO). While efective for final answer accuracy, these approaches treat reasoning as a monolithic sequence, failing to capture the granular quality of intermediate steps.

We introduce Collaborative Reasoning Distillation (CRD), a framework that shifts from imitation to collaborative synthesis. As illustrated in Figure 1, CRD transforms multiple teacher outputs into a curated training dataset through three integrated components. First, diverse teachers generate initial reasoning chains through iterative cross-feedback. Rather than treating each teacher independently, teachers critique and refine each other’s outputs, producing high-quality candidates that reflect complementary strengths. Second, we assess the logical validity of each step using a multi-judge system. Fine-grained quality scores capture reasoning quality beyond binary correctness, identifying high-quality patterns even in failed attempts. Third, we construct hybrid reasoning chains via coherence-aware step stitching, combining high-quality steps from diferent teachers while maintaining logical flow. Finally, we train students via Reasoning Quality Optimization (RQO) with curriculum-based budget constraints, enabling eficient reasoning within token limits. In our experiments, we refer to these components as Cross-Feedback, Judging, Thinking Path, and Optimal Budget.

## 2.1. Interactive Multi-Teacher Cross-Feedback

Recent studies reveal that LLMs exhibit complementary error profiles. Models like DeepSeek-R1 are prone to computational errors and statement skipping, omitting intermediate steps (Abdollahi et al., 2025), while models like Gemini exhibit shallow reasoning with prediction bias (Gao et al., 2025). A student trained on a single teacher is often biased toward that teacher’s error profile, motivating approaches that leverage these asymmetric weaknesses for mutual correction (Du et al., 2023).

We introduce a cross-feedback loop where � distinct teachers (we use $N = 2 ,$ DeepSeek-R1 and Gemini 2.5 Flash Thinking) engage in iterative critique and refinement. Rather than simple ensemble voting, each teacher reviews peer outputs, identifies logical gaps and errors, and generates improved reasoning by synthesizing complementary insights. This interactive process produces reasoning chains of higher logical depth and diversity than any single teacher could generate alone.

For the initial round $( k = 0 )$ , each teacher $T _ { i }$ independently generates a baseline reasoning chain $R _ { i } ^ { ( 0 ) }$ for the given problem �. In subsequent refinement rounds $( k \in \{ 1 , 2 , \ldots , K \} )$ , each teacher $T _ { i }$ receives a structured input consisting of the original problem � and the reasoning chains from all other teachers in the previous round $\{ R _ { j } ^ { ( k - 1 ) } \mid j \neq i \}$ . A carefully designed review prompt instructs the teacher to examine these peer chains, identify logical gaps and errors, suggest corrections, and synthesize insights from multiple perspectives (see Appendix A for the complete prompt). The teacher then generates an improved reasoning chain $R _ { i } ^ { ( k ) }$ that incorporates this feedback.

After � rounds of cross-feedback refinement, we collect the complete set of reasoning chains including both initial and refined outputs from all teachers. Our experiments show that performance converges at $K = 2$ to 3 rounds, with additional iterations providing diminishing returns (Appendix G).

Cross-feedback substantially improves dataset quality. Comparing initial and refined outputs, mathematical reasoning accuracy increased from 64.1% to 83.5%, and code generation success rate improved from 52.3% to 73.9%. Our ablation study shows this translates to +3.4%–7.1% student performance improvements across benchmarks (Section 3.3).

## 2.2. Step-wise Reasoning Quality Assessment

Current models like DeepSeek-R1 (Guo et al., 2025) primarily rely on outcome-based rewards, where intermediate steps receive identical signals regardless of logical validity. This sparse reward structure fails to distinguish between eficient reasoning and verbose or lucky guesses, treating all correct-answer paths equivalently.

To address this limitation, we introduce a fine-grained step-wise quality assessment system that directly evaluates the logical validity of each intermediate step, independent of the final answer. Instead of deriving rewards from format compliance or outcome correctness, we assign continuous scalar scores to every step, capturing logical validity and reasoning eficiency (Lightman et al., 2023; Li and Li, 2025). This enables students to learn the distinction between high-quality reasoning paths and low-quality paths that coincidentally reach the correct answer, and to internalize the nuanced diferences among multiple correct solutions.

We employ two specialized judge models to evaluate each step: JudgeLM (Zhu et al., 2025), which achieves 16× speedup over GPT-4o with comparable accuracy, and xVerify (Chen et al., 2025), a lightweight 0.5B model with exceptional eficiency.

To capture algorithmic uncertainty, each judge evaluates every step twice, yielding a set of four independent scores. To ensure robustness against stochastic outliers, we apply a 25% trimmed mean aggregation strategy: the minimum and maximum scores are discarded, and the remaining two intermediate scores are averaged to produce the final step quality score $v _ { t }$ . In our ablations, using both judges provides larger gains than using either judge alone, suggesting that JudgeLM and xVerify contribute complementary quality signals rather than redundant scores (Appendix H).

After step-wise scoring, we perform domain-specific verification combining LLM evaluation with specialized tools. For code, we execute solutions and penalize steps after failures. For math, we compare against ground truth and penalize steps after divergence points. For logic, we verify inference rule correctness.

## 2.3. Coherence-Aware Step Stitching

The third component of CRD is Coherence-Aware Step Stitching, which we call the Thinking Path algorithm. Standard distillation selects the best complete chain from a single teacher, inheriting that teacher’s limitations even when other teachers excel at specific steps. In contrast, CRD enables step-level synthesis: combining Teacher 1’s novel problem setup with Teacher 2’s rigorous derivation with Teacher 1’s verification, even if neither teacher alone produces a complete correct solution.

This capability is possible because we have both fine-grained quality scores (Section 2.2) and multiple diverse reasoning chains (Section 2.1). The challenge is selecting steps that maintain logical coherence. Simply picking the highest-scoring step at each position creates Frankenstein chains where steps do not flow naturally into each other.

We formulate this as a coherence-aware path-finding problem on a directed acyclic graph (DAG) (Yao et al., 2023), where nodes represent reasoning steps $s _ { t } ^ { ( i ) }$ from teacher $T _ { i }$ at position �. We define the optimal hybrid path �<sup>ˆ</sup> as the sequence that maximizes both step-wise quality and semantic coherence between consecutive steps.

To handle granularity diferences across teachers, we align steps using embedding similarity with lookahead merging: if consecutive steps from one teacher better match a single step from another, they are merged into one node. When similarity falls below a threshold, steps remain as parallel branches rather than forced alignments.

Let ${ \cal S } _ { t } = \{ s _ { t } ^ { ( 1 ) } , \dots , s _ { t } ^ { ( N ) } \}$ be the set of candidate steps at position �. The score for transitioning from step $u \in S _ { t - 1 }$ to step $v \in S _ { t }$ balances two objectives:

$$
\operatorname { S c o r e } ( u \to v ) = \lambda \cdot \underbrace { \mathcal { V } ( v ) } _ { \mathrm { Q u a l i t y } } + ( 1 - \lambda ) \cdot \underbrace { \log ( \epsilon + C ( u , v ) ) } _ { \mathrm { C o h e r e n c e } }\tag{1}
$$

where $C ( u , v )$ is the transition-coherence score between consecutive steps, computed from the cosine similarity of their embeddings and rescaled to [0, 1], �(�) is the normalized step-quality score, and � is a small constant for numerical stability. We use $\lambda = 0 . 7$ based on the sensitivity analysis in Figure 4.

We find the globally optimal path using a Viterbi-style dynamic programming algorithm:

$$
\hat { R } = \underset { s _ { 1 } , \ldots , s _ { L } } { \arg \operatorname* { m a x } } \sum _ { t = 1 } ^ { L } { \mathrm { S c o r e } ( s _ { t - 1 }  s _ { t } ) }\tag{2}
$$

This enables emergent solutions that transcend individual teacher limitations. A problem might require Teacher 1’s creative insight for setup, Teacher 2’s careful arithmetic for calculation, and Teacher 1’s verification logic for confirmation. By combining these complementary strengths at the step level, the resulting trajectory represents the collective best reasoning that no single teacher could produce in isolation. The complete algorithm with pseudocode is provided in Appendix C. Among 2,694 tasks with incorrect pre-stitch trajectories, 528 (19.6%) become correct after stitching (Appendix I).

## 2.4. Reasoning Transfer via RQO

Once the reasoning trajectories are synthesized via step-level stitching and assigned fine-grained quality scores, we transfer these capabilities to the student model �<sub>�</sub> using a reinforcement learning approach that directly leverages the step-wise quality signals.

Reasoning Quality Optimization (RQO). Standard preference-based methods like DPO treat all preference pairs equally, ignoring the magnitude of quality diferences. In contrast, RQO directly utilizes the fine-grained quality scores from Section 2.2 to weight learning signals. For each reasoning chain, we aggregate the step-level quality scores to obtain a trajectory-level quality signal $\begin{array} { r } { { \boldsymbol { v } } _ { R } = \sum _ { t } { \boldsymbol { v } } _ { t } . } \end{array}$ , where $v _ { t }$ is the quality score of step �. For chain pairs $( R _ { w i n } , v _ { w i n } )$ and $( R _ { l o s e } , v _ { l o s e } )$ where $v _ { w i n } > v _ { l o s e } ,$ , we compute the quality margin $\Delta v = v _ { w i n } - v _ { l o s e }$ and apply exponential weighting:

$$
w ( \Delta v ) = { \frac { e ^ { \beta \Delta v } - 1 } { Z } }\tag{3}
$$

where $\beta$ controls the sensitivity to quality diferences and $Z$ is a normalization constant. The loss function becomes:

$$
\mathcal { L } _ { \mathrm { R Q O } } = - \mathbb { E } \left[ w ( \Delta v ) \cdot \log \sigma \left( \pi _ { \theta } ( R _ { w i n } | q ) - \pi _ { \theta } ( R _ { l o s e } | q ) \right) \right]\tag{4}
$$

This formulation ensures that chains with larger quality margins receive stronger learning signals, enabling the student to learn not just from correct vs. incorrect distinctions, but from the nuanced quality diferences captured by step-level assessment.

Dynamic Length Regularization. Training with strict token constraints from the start fails due to the learnability gap (Ding et al., 2025): students cannot learn complex reasoning patterns when forced to be concise prematurely. Following the Long-to-Short paradigm (Wu et al., 2025; Hammoud et al., 2025), we implement a curriculum-based length penalty that progressively constrains token budgets (32K→10K→1K). The efective reward includes an exponential penalty for exceeding the current budget �:

$$
R _ { \mathrm { b u d g e t } } ( l ) = R _ { \mathrm { b a s e } } \cdot \exp ( - \alpha \operatorname* { m a x } ( 0 , l - B ) )\tag{5}
$$

where � is the response length and � controls penalty severity. This soft constraint (Li et al., 2025a) enables the model to first learn correct reasoning at longer lengths, then gradually compress into eficient solutions without performance degradation.

## 3. Evaluation

Training Data. Our training dataset comprises 50K reasoning instances from four diverse sources: Deep-ScaleR (Luo et al., 2025) for mathematical reasoning, Eurus-2-RL (Cui et al., 2025) for code generation, Reasoning Gym (Stojanovski et al., 2025) for logical reasoning, and SCP-116K (Lu et al., 2025) for STEM problems (Appendix D).

Recent findings show that reasoning distillation exhibits a performance valley between 1K–30K examples (He et al., 2025). Our 50K dataset size avoids this valley while remaining significantly smaller than prior methods (220K–800K examples), enabled by multi-stage quality filtering that ensures high-quality reasoning across the dataset.

Contamination Analysis. We systematically verified training-evaluation separation. For DeepScaleR, we identified 11 overlapping problems with MATH-500 (2.2%), which were excluded during our 50% subsampling; AIME’24 and AIME’25 showed zero overlap. For Eurus-2-RL (455K math, 25K code problems), we sampled 15K code problems for training and found no contamination across all code benchmarks. LiveCodeBench v5 problems (August 2024–February 2025) post-date all code training sources: APPS (2021), CodeContests (2022), and TACO (2023).

Benchmarks. We evaluate on mathematics (MATH-500 (Hendrycks et al., 2021), AIME’24 (Mathematical Association of America, 2024), AIME’25 (Mathematical Association of America, 2025)), code (LiveCodeBench (Jain et al., 2024)), and graduate-level STEM reasoning (GPQA-Diamond (Rein et al., 2023)). Detailed descriptions are in Appendix E.

Models. We use DeepSeek-R1 and Gemini 2.5 Flash Thinking as teachers, running � = 2 cross-feedback rounds to generate refined reasoning chains. For students, we evaluate DeepSeek-R1-Distill-Qwen (1.5B) and Qwen3 (1.7B, 4B, 30B-A3B MoE) to demonstrate generalization across dense and mixture-of-experts architectures.

## 3.1. Performance Across Model Sizes

A key question in reasoning distillation is whether small models can achieve strong performance through higher-quality supervision rather than relying solely on parameter scaling. Table 1 demonstrates that CRD answers this afirmatively through step-wise quality assessment and coherence-aware synthesis, achieving competitive performance across all model sizes.

Table 1: Performance comparison across benchmarks (Pass@1). We report accuracy (%) on mathematical, coding, and general reasoning benchmarks. The best results are highlighted in bold. Gray rows indicate DeepSeek-R1 and distilled models included as reference points for parameter eficiency comparison.
<table><tr><td></td><td colspan="3">Math</td><td>Code</td><td>Graduate-Level STEM</td></tr><tr><td>Model</td><td>MATH-500</td><td>AIME&#x27;24</td><td>AIME&#x27;25</td><td>LCB</td><td>GPQA-Diamond</td></tr><tr><td colspan="6">Tiny Scale (&lt;2B)</td></tr><tr><td>DeepSeek-R1-Distill-7B</td><td>92.8</td><td>55.5</td><td></td><td>37.6</td><td>49.1</td></tr><tr><td>Gemini-Flash-Distill-1.5B</td><td>85.3</td><td>32.5</td><td>24.6</td><td>17.2</td><td>31.6</td></tr><tr><td>DeepSeek-R1-Distill-1.5B</td><td>83.9</td><td>28.9</td><td>22.0</td><td>16.9</td><td>32.9</td></tr><tr><td>ProRL-1.5B†</td><td>91.9</td><td>48.1</td><td>33.3</td><td>23.8</td><td>41.7</td></tr><tr><td>CRD-1.5B†</td><td>91.1</td><td>46.9</td><td>35.1</td><td>27.2</td><td>39.7</td></tr><tr><td>DeepSeek-R1-Distill-7B</td><td>92.8</td><td>55.5</td><td></td><td>37.6</td><td>49.1</td></tr><tr><td>Qwen3-1.7B†</td><td>93.4</td><td>48.3</td><td>36.8</td><td>33.2</td><td>40.1</td></tr><tr><td>CRD-1.7B†</td><td>94.5</td><td>50.3</td><td>42.2</td><td>37.3</td><td>42.9</td></tr><tr><td colspan="6">Small Scale (≤4B)</td></tr><tr><td>DeepSeek-R1-Distill-32B</td><td>94.3</td><td>72.6</td><td>60.0</td><td>57.2</td><td>62.1</td></tr><tr><td>Qwen3-4B†</td><td>97.0</td><td>73.8</td><td>65.6</td><td>54.2</td><td>55.9</td></tr><tr><td>CRD-4B†</td><td>97.3</td><td>76.2</td><td>70.3</td><td>57.4</td><td>58.7</td></tr><tr><td colspan="6">Large Scale (≥30B)</td></tr><tr><td>EXAONE Deep 32B</td><td>95.7</td><td>72.1</td><td>65.8</td><td>59.5</td><td>66.1</td></tr><tr><td>Hermes 4 70B</td><td>95.5</td><td>73.5</td><td>67.5</td><td>50.5</td><td>66.1</td></tr><tr><td>DeepSeek-R1 (671B, A37B)</td><td>97.3</td><td>79.8</td><td>70.0</td><td>63.5</td><td>71.5</td></tr><tr><td>Qwen3-30B-A3B†</td><td>98.0</td><td>80.4</td><td>70.9</td><td>62.6</td><td>65.8</td></tr><tr><td>CRD-30B-A3B†</td><td>98.3</td><td>82.5</td><td>73.2</td><td>65.9</td><td>67.2</td></tr></table>

<sup>†</sup> Models marked with † share the same baseline model; our method generally outperforms them across benchmarks.

At the tiny scale (1.5B), CRD-1.5B achieves 91.1% on MATH-500 compared to the baseline DeepSeek-R1-Distill’s 83.9%. On AIME’25, CRD-1.5B reaches 35.1%, slightly exceeding ProRL-1.5B (Liu et al., 2025) (33.3%), the current state-of-the-art at the 1.5B scale, while using 2.7× less training dataset with a similar RL-based approach. The gray rows contextualize this result: CRD-1.5B approaches the 7B distilled model on MATH-500 (91.1 vs. 92.8, 4.7× smaller), while CRD-1.7B (94.5%) surpasses the 14B distilled model (93.9%) on MATH-500 (8.2× smaller). This suggests that higher-quality supervision can partly substitute for parameter scaling at small model sizes.

At the small scale (4B), CRD-4B achieves 97.3% on MATH-500 and 70.3% on AIME’25, approaching saturation on standard benchmarks. CRD-4B exceeds Hermes 4 70B (Teknium et al., 2025) (17× smaller) and EXAONE Deep 32B (Bae et al., 2026) on AIME’25 and outperforms 32B distilled model (8× smaller) on all math and code benchmarks, demonstrating that CRD provides an alternative to parameter scaling. Figure 2a visualizes this parameter eficiency across model scales.

At the large scale (30B), CRD-30B-A3B improves across diverse reasoning domains. It improves mathematical reasoning on AIME’25 from 70.9% to 73.2% (+2.3%), code generation on LiveCodeBench from 62.6% to 65.9% (+3.3%), and STEM reasoning on GPQA-Diamond from 65.8% to 67.2% (+1.4%). These results show that CRD consistently improves performance across all model scales and problem types. Remarkably, CRD-30B-A3B surpasses its own teacher DeepSeek-R1 on math and code benchmarks, despite using 12× fewer active parameters (3B vs 37B). This demonstrates that CRD enables students to exceed teacher capabilities through cross-teacher synthesis.

Comparison with Multi-Teacher Approaches. We compare CRD against existing multi-teacher distillation methods to isolate the benefit of step-level quality curation. Table 2 reports results using the same Qwen2.5-1.5B and Llama-3.1-8B student backbones and the same pool of CRD-generated teacher candidates, isolating the efect of the distillation and curation strategy.

![](images/983a09619e8b7689a1867a6b755b939907bd9e846cdc742719c6ac3680f5e273.jpg)

Table 2: Multi-teacher comparison on Qwen-2.5- 1.5B, Llama-3.1-8B.  
Table 3: Ablation study on CRD components. Base model is DeepSeek-R1-Distill-Qwen-1.5B.
<table><tr><td colspan="4">Method GSM8K MATH-500 HumanEval LCB</td></tr><tr><td>Qwen-2.5-1.5B (Base)</td><td>73.2</td><td>55.3</td><td>61.6 14.8</td></tr><tr><td>TinyLLM</td><td>81.8</td><td>68.1</td><td>64.3 15.7</td></tr><tr><td>FAIR</td><td>82.3</td><td>73.7</td><td>65.8 16.3</td></tr><tr><td>MAGDi</td><td>85.9</td><td>72.6</td><td>64.9 18.4</td></tr><tr><td>CRD (Ours)</td><td>95.2</td><td>91.1</td><td>72.3 27.2</td></tr><tr><td>Llama-3.1-8B (Base)</td><td>84.5</td><td>51.9</td><td>72.6 16.2</td></tr><tr><td>FAIR</td><td>86.2</td><td>54.8 75.4</td><td>18.5</td></tr><tr><td>MAGDi</td><td>86.9</td><td>56.1 76.2</td><td>18.8</td></tr><tr><td>CRD (Ours)</td><td>92.6</td><td>62.3</td><td>85.1 22.4</td></tr></table>

<table><tr><td>Configuration</td><td>MATH-500 AIME&#x27;25</td><td></td><td>LCB</td><td>GPQA-Diamond</td></tr><tr><td>Base Model</td><td>83.9</td><td>22.0</td><td>16.9</td><td>32.9</td></tr><tr><td>+ Cross-Feedback</td><td>87.3 +3.4</td><td>29.1 +7.1 22.8 +5.9</td><td></td><td> $3 6 . 7 \ + 3 . 8 $ </td></tr><tr><td>+ Thinking Path</td><td>89.4 +2.1</td><td> $3 2 . 6 ~ + 3 . 5 ~ 2 5 . 4 ~ + 2 . 6$ </td><td></td><td> $3 7 . 8 \ + 1 . 1$ </td></tr><tr><td>+ Optimal Budget</td><td>89.7 +0.3</td><td> $3 3 . 2 \ + 0 . 6 \ 2 5 . 8 \ + 0 . 4$ </td><td></td><td> $3 8 . 0 \ \substack { + 0 . 2 }$ </td></tr><tr><td>+ Judging (CRD)</td><td> $9 1 . 1 \ + 1 . 4$ </td><td> $\mathbf { 3 5 . 1 _ { \delta + 1 . 9 } 2 7 . 2 _ { \delta + 1 . 4 } }$ </td><td></td><td> $3 9 . 7 \ + 1 . 7 $ </td></tr><tr><td>Total Gain</td><td>↑7.2</td><td>↑13.1</td><td>↑10.3</td><td>↑6.8</td></tr></table>

![](images/22e5a0f4c3d7f14b2a1d541e8dee12bbe1c42da69a4f83ad7300de80f475b0fc.jpg)  
(a) Parameter eficiency comparison.

![](images/0001263ebc84ba8edabc148bdfe695994e265ef994445a615bdac4fa5eff3d33.jpg)  
(b) Data eficiency comparison.  
Figure 2: (a) CRD achieves comparable performance to DeepSeek-R1-Distill models with 4.7–8× fewer parameters. CRD-1.5B approaches the 7B model on MATH-500 (4.7×), while CRD-4B surpasses the 32B model (8×). (b) CRD achieves superior performance with 12× less data than DeepSeek-R1-Distill (600K reasoning samples) and 2.7× less than ProRL (136K), while outperforming both on AIME’25 and LCB.

∙ TinyLLM (Tian et al., 2024) uses teacher-forcing CoT to transfer reasoning from multi-teachers, but does not perform step-level quality scoring.

∙ FAIR (Li et al., 2025b) employs peer-review between teachers to identify student mistakes and filter rationales via acceptance threshold.

∙ MAGDi (Chen et al., 2024) represents multi-agent interactions as graphs and uses contrastive learning between correct and incorrect reasoning.

CRD outperforms all baselines by substantial margins: +18.5% over MAGDi on MATH-500, +9.3% on GSM8K, and +8.8% on LiveCodeBench. This gap stems from CRD’s step-level granularity: while TinyLLM, FAIR, and MAGDi operate at the chain level, selecting or weighting complete solutions, CRD evaluates each reasoning step independently. A chain with one flawed step is entirely discarded by chain-level methods, but CRD extracts its valid steps and combines them with correct steps from other teachers. This synthesis is particularly impactful on multi-step reasoning (MATH-500: +18.5%), where a single error propagates to incorrect answers.

Generalization Beyond the Training Distribution. Table 4 reports two held-out evaluations. On BoxNet, an out-of-distribution planning task used by ProRL and excluded from our training data, CRD-1.5B is comparable to ProRL at pass@1 and higher at pass@2 and pass@4. On the text-only subset of HLE (Phan et al., 2026), CRD-30B-A3B improves over its Qwen3-30B-A3B base by 3.23 points.

Table 4: Generalization beyond the training distribution. Left: BoxNet planning (100 procedurally generated instances, our re-evaluation of all models, temperature 0.6, top-� 0.95, 32K tokens). Right: HLE text-only subset (oficial protocol).
<table><tr><td>BoxNet</td><td>pass@1</td><td>pass@2</td><td>pass@4</td><td></td><td>Text-only</td></tr><tr><td>DeepSeek-R1-Distill-1.5B</td><td>0.0</td><td>0.0</td><td>0.0</td><td>Qwen3-30B-A3B</td><td>11.31</td></tr><tr><td>ProRL-1.5B</td><td>7.1</td><td>13.8</td><td>22.4</td><td>CRD-30B-A3B</td><td>14.54</td></tr><tr><td>CRD-1.5B</td><td>6.8</td><td>14.6</td><td>25.3</td><td></td><td></td></tr></table>

## 3.2. Data Eficiency: Quality Over Quantity

Figure 2b demonstrates CRD’s data eficiency. At 50K samples, CRD-1.5B reaches 35.1% on AIME’25, surpassing DeepSeek-R1-Distill (22.0% on 600K reasoning samples) and ProRL (33.3% on 136K samples). CRD achieves this superior performance using 12× and 2.7× less data, respectively. Notably, CRD-1.5B exhibits consistent data scaling behavior within our curated dataset, improving monotonically from 28.1% at 10K to 35.1% at 50K. This curve is shifted upward by quality-aware curation, so CRD achieves with 10K samples what baselines require 600K to match. DeepSeek-R1-Distill uses 800K total samples (600K reasoning, 200K non-reasoning); we compare against the reasoning portion for fair evaluation.

This eficiency stems mainly from curation, since naive SFT on the same 50K trajectories already reaches 90.5% on MATH-500 (Table 5). RQO adds a consistent further gain by weighting preference pairs with their step-derived quality margins.

CRD improves reasoning distillation eficiency by replacing scale-driven supervision with quality-aware data selection. Instead of relying on a large pool of distillation corpus, CRD constructs a compact 50K-example training set using the same teacher models, yielding a 12× smaller final trajectory set and a 2.7× smaller dataset than ProRL’s 136K examples. Although CRD generates multiple candidate chains per prompt through cross-feedback, only high-quality trajectories are retained for student learning. As a result, the 1.5B student requires only 1,632 training steps and 8.8 H100 GPU-hours, compared with 7,800 steps and 42 H100 GPUhours for the 600K distillation baseline and more than 2K steps for ProRL. This supports our main claim that improving the quality and coherence of reasoning supervision can reduce the amount of final training data required by the student.

## 3.3. Ablation Study: Understanding Each Component

Table 3 isolates the contribution of each CRD component, revealing that they address distinct limitations in reasoning distillation. At the step level, algebra step correctness improves from 85.8% to 92.2% (Appendix F), and cross-feedback converges after two rounds (Appendix G). Appendix J shows representative student outputs.

Cross-Feedback provides the largest individual gain (+3.4–7.1%), with particularly strong improvements on complex benchmarks: +7.1% on AIME’25 and +5.9% on LiveCodeBench. This aligns with our motivation that teachers exhibit complementary error profiles. DeepSeek-R1 excels at creative problem formulation but makes computational errors. Gemini provides rigorous derivations but occasionally shallow reasoning. Cross-feedback enables mutual correction, producing refined candidates that repair errors found in individual teacher outputs.

Thinking Path contributes +1.1–3.5% by enabling step-level fusion across teachers. When Teacher 1’s novel approach fails due to a logic bug and Teacher 2’s standard approach fails due to a calculation error, Thinking Path stitches Teacher 1’s creative setup with Teacher 2’s correct computation, yielding a correct solution that neither teacher produced alone.

Optimal Budget and Judging provide incremental refinements (+0.2–0.6% and +1.4–1.9% respectively). Optimal Budget implements curriculum-based length constraints that help students learn eficient reasoning without premature truncation (Ding et al., 2025). Judging adds fine-grained quality signals that enable the student to distinguish between high-quality and mediocre reasoning paths, even when both reach correct answers.

Same-data SFT vs. RQO. Table 5 compares RQO with naive SFT trained on the same 50K Thinking Path trajectories from the same DeepSeek-R1-Distill-Qwen-1.5B checkpoint. Naive SFT already reaches 90.5% on

MATH-500, so most of the gain comes from the curated data. RQO adds consistent gains on all four benchmarks (+1.15 points on average).  
Table 5: Same-data comparison of training objectives (DeepSeek-R1-Distill-Qwen-1.5B, 50K Thinking Path trajectories).
<table><tr><td>Method</td><td>MATH-500</td><td>AIME&#x27;25</td><td>LCB</td><td>GPQA-Diamond</td></tr><tr><td>Naive SFT</td><td>90.5</td><td>33.4</td><td>25.7</td><td>38.9</td></tr><tr><td>RQO (CRD)</td><td>91.1</td><td>35.1</td><td>27.2</td><td>39.7</td></tr></table>

![](images/a3f7008a3cc643525b8022362360c7d9a3ec95fe8d2146cbec781ba144174d33.jpg)  
Figure 3: Single-teacher vs. multi-teacher comparison on Qwen2.5-1.5B models.

![](images/ee0aca028c1d787796750289143d0f532f94e9ef3da7b61ba53f3bbd5c9049aa.jpg)

![](images/45a34f4a4bebe3a273f5f7d25e711696dd09fc6e7ae876036ab9ce19edbf3a53.jpg)  
Figure 4: Sensitivity analysis of � (quality-coherence trade-of). Performance peaks at $\lambda = 0 . 7 ;$ Smaller � values over-prioritize coherence, while larger values produce less coherent chains.

Single-Teacher vs Multi-Teacher. Figure 3 illustrates the performance gap between single-teacher distillation and CRD’s multi-teacher approach. Both Gemini-Flash-Distill and DeepSeek-R1-Distill apply the same scoring and student-training pipeline to a single teacher’s outputs, but remove cross-feedback and cross-teacher step stitching. CRD consistently outperforms both across all benchmarks, with particularly large gains on AIME’24 (+14–18%) and LCB (+10%), showing that combining two heterogeneous teachers outperforms either teacher alone.

Lambda sensitivity. Figure 4 shows the sensitivity of the coherence-quality trade-of parameter � in Thinking Path. At � ≤ 0.5, excessive weight on transition coherence leads to selecting plausible-sounding but low-quality steps (MATH-500 drops to 76.0%). At � > 0.8, local quality dominates and coherence is underweighted, producing disconnected reasoning chains (82.3%). The optimal value, $\lambda = 0 . 7 ,$ balances high-quality step selection while maintaining logical flow.

## 4. Related Work

Reasoning and Distillation. Transferring reasoning capabilities from large models to smaller students via knowledge distillation is a key research area. Prior work has explored distilling Chain-of-Thought demonstrations (Wei et al., 2023), leveraging multiple models for enhanced knowledge transfer (Tian et al., 2024), and iterative refinement techniques (Saunders et al., 2022; Liang et al., 2024). Contrastive Representation Distillation (Tian et al., 2022) aligns student and teacher embeddings for finer-grained transfer. Recent work demonstrates that budget forcing techniques can yield significant reasoning improvements with small curated datasets (Muennighof et al., 2025). LIMO (Ye et al., 2025) shows that a few hundred carefully curated examples can elicit strong mathematical reasoning, while CRD improves each trajectory through cross-teacher feedback and step-level synthesis across mathematics, coding, logic, and STEM. Building upon these foundations, our Collaborative Reasoning Distillation (CRD) framework generates high-quality reasoning datasets through integrated multi-teacher generation, cross-feedback refinement, and fine-grained quality assessment using multi-judge evaluation.

Multi-Teacher Distillation Methods. Recent work has explored various approaches to leverage multiple teacher models for knowledge distillation. TinyLLM uses multi-task SFT where the student imitates both answers and teacher-specific rationales. FAIR uses peer review among teachers to filter rationales with an acceptance threshold. MAGDi distills multi-agent interaction graphs with a contrastive objective between correct and incorrect reasoning. Our CRD framework takes a complementary approach by introducing interactive refinement (cross-feedback) before distillation and explicit step-wise quality scoring during training. This enables the student to learn from the collaborative synthesis of multiple perspectives, resulting in 3.6–23% improvements over these methods when using identical training data (see Section 3.1).

Eficient Reasoning Distillation. Advanced reasoning models such as DeepSeek-R1 (Guo et al., 2025) have demonstrated that extended test-time computation can substantially enhance reasoning capabilities. CRD is complementary to test-time scaling: its primary goal is to improve the quality of distilled supervision, while the Optimal Budget curriculum can additionally encourage shorter student reasoning at inference time. This aligns with recent findings that concise reasoning can achieve comparable accuracy to verbose chains (Xu et al., 2025), and that reasoning structure often matters more than output length (He et al., 2025). Our Optimal Budget curriculum enables students to produce efective solutions using a fraction of the tokens required by teacher models, making advanced reasoning accessible for latency-sensitive deployments without sacrificing problem-solving capability.

LLM-as-a-Judge. LLM-as-a-Judge serves as a scalable alternative to human evaluation for assessing intermediate reasoning steps (Gu et al., 2025). Unlike outcome-based evaluation, step-level assessment provides dense quality signals that help models learn the distinction between logically sound reasoning and lucky guesses. EvalPlanner (Saha et al., 2025) demonstrates that learned evaluation criteria can efectively assess reasoning quality, providing a foundation for our multi-judge approach.

## 5. Conclusion

This paper presents Collaborative Reasoning Distillation (CRD), a framework that transfers reasoning capabilities from multiple foundation models to compact students through multi-teacher cross-feedback, fine-grained quality assessment, and coherence-aware step synthesis. CRD achieves strong reasoning performance in small models on the evaluated benchmarks: CRD-4B reaches 97.3% on MATH-500, 70.3% on AIME’25, and 57.4% on LiveCodeBench, using only 50K training examples, 12× fewer final training trajectories than standard distillation. CRD demonstrates that step-level synthesis enables solving problems where individual teachers fail.

## Limitations and Future Work

CRD relies on LLM teachers and judges and may inherit some of their biases, so plausible but incorrect steps can occasionally receive high scores. Stitching selects paths that score well under our quality and coherence criteria, which does not by itself guarantee that every selected chain is logically correct. We evaluate one teacher pair (DeepSeek-R1 and Gemini 2.5 Flash), and other teacher combinations remain to be explored. The reported training cost covers student optimization only and does not include teacher generation or curation. We compare RQO with naive SFT on the same data but not with DPO.

While our current rubric provides a robust baseline, reliance on static criteria limits scalability across diverse domains. A promising future direction is to transition from hand-crafted rules to learned quality models trained on preference data. Such models could dynamically adapt to domain-specific nuances and enable more scalable supervision without manual rubric engineering. Furthermore, extending our framework to handle heterogeneous teacher feedback could mitigate potential sycophancy, paving the way for more robust collaborative reasoning systems. Future work includes larger and all-open-weight teacher pools such as QwQ-32B and GPT-OSS, and stronger teachers for expert-level benchmarks such as HLE. We also plan controlled comparisons with on-policy distillation and SFT-based baselines, formal bounds on stitching failures, and a larger human audit of stitched chains.

## References

Mohammad Abdollahi, Khandaker Rifah Tasnia, Soumit Kanti Saha, Jinqiu Yang, Song Wang, and Hadi Hemmati. Demystifying errors in llm reasoning traces: An empirical study of code execution simulation, 2025. URL https://arxiv.org/abs/2512.00215.

Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Yemuk Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Kijeong Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Hyosang Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Youchul Kim, Edward Hwayoung Lee, Haeju Lee, Honglak Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Yongmin Park, Sihoon Yang, Heuiyeen Yeen, Sihyuk Yi, and Hyeongu Yun. Exaone deep: Reasoning enhanced language models, 2026. URL https://arxiv.org/abs/2503.12524.

Ding Chen, Qingchen Yu, Pengyuan Wang, Mengting Hu, Wentao Zhang, Zhengren Wang, Bo Tang, Feiyu Xiong, Xinchi Li, Chao Wang, Minchuan Yang, and Zhiyu Li. xverify: Eficient answer verifier for reasoning model evaluations, 2025. URL https://arxiv.org/abs/2504.10481.

Justin Chih-Yao Chen, Swarnadeep Saha, Elias Stengel-Eskin, and Mohit Bansal. Magdi: Structured distillation of multi-agent interaction graphs improves reasoning in smaller language models, 2024. URL https: //arxiv.org/abs/2402.01620.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

Ben Cottier, Robi Rahman, Loredana Fattorini, Nestor Maslej, Tamay Besiroglu, and David Owen. The rising costs of training frontier ai models, 2025. URL https://arxiv.org/abs/2405.21015.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. Process reinforcement through implicit rewards, 2025. URL https://arxiv.org/abs/2502.01456.

Dongyi Ding, Tiannan Wang, Chenghao Zhu, Meiling Tao, Yuchen Eleanor Jiang, and Wangchunshu Zhou. Micota: Bridging the learnability gap with intermediate cot and teacher assistants, 2025. URL https: //arxiv.org/abs/2507.01887.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate, 2023. URL https://arxiv.org/abs/2305.1 4325.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning, 2023. URL https://arxiv.org/abs/2301.12726.

Tianchen Gao, Jiashun Jin, Zheng Tracy Ke, and Gabriel Moryoussef. A comparison of deepseek and other llms, 2025. URL https://arxiv.org/abs/2502.03688.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on llm-as-a-judge, 2025. URL https://arxiv.org/abs/2411.15594.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yuku Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

Hasan Abed Al Kader Hammoud, Kumail Alhamoud, Abed Hammoud, Elie Bou-Zeid, Marzyeh Ghassemi, and Bernard Ghanem. Train long, think short: Curriculum learning for eficient reasoning, 2025. URL https://arxiv.org/abs/2508.08940.

Muyu He, Muhammad Ali Shafique, Anand Kumar, Tsach Mackey, and Nazneen Rajani. The valley of code reasoning: Scaling knowledge distillation of large language models, 2025. URL https://arxiv.org/ab s/2510.06101.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https: //arxiv.org/abs/2103.03874.

Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network, 2015. URL https://arxiv.org/abs/1503.02531.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. URL https://arxiv.org/abs/2403.07974.

Ziheng Jiang, Haibin Lin, Yinmin Zhong, Qi Huang, Yangrui Chen, Zhi Zhang, Yanghua Peng, Xiang Li, Cong Xie, Shibiao Nong, Yulu Jia, Sun He, Hongmin Chen, Zhihao Bai, Qi Hou, Shipeng Yan, Ding Zhou, Yiyao Sheng, Zhuo Jiang, Haohan Xu, Haoran Wei, Zhang Zhang, Pengfei Nie, Leqi Zou, Sida Zhao, Liang Xiang, Zherui Liu, Zhe Li, Xiaoying Jia, Jianxi Ye, Xin Jin, and Xin Liu. Megascale: Scaling large language model training to more than 10,000 gpus, 2024. URL https://arxiv.org/abs/2402.15627.

Shiyang Li, Jianshu Chen, Yelong Shen, Zhiyu Chen, Xinlu Zhang, Zekun Li, Hong Wang, Jing Qian, Baolin Peng, Yi Mao, Wenhu Chen, and Xifeng Yan. Explanations from large language models make small reasoners better, 2022. URL https://arxiv.org/abs/2210.06726.

Wendi Li and Yixuan Li. Process reward model with q-value rankings, 2025. URL https://arxiv.org/ab s/2410.11287.

Yanhao Li, Lu Ma, Jiaran Zhang, Lexiang Tang, Wentao Zhang, and Guibo Luo. Leash: Adaptive length penalty and reward shaping for eficient large reasoning model, 2025a. URL https://arxiv.org/abs/2512.2 1540.

Zhuochun Li, Yuelyu Ji, Rui Meng, and Daqing He. Learning from committee: Reasoning distillation from a mixture of teachers with peer-review, 2025b. URL https://arxiv.org/abs/2410.03663.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent debate, 2024. URL https://arxiv.org/abs/2305.19118.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023. URL https://arxiv.org/ab s/2305.20050.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models, 2025. URL https: //arxiv.org/abs/2505.24864.

Dakuan Lu, Xiaoyu Tan, Rui Xu, Tianchu Yao, Chao Qu, Wei Chu, Yinghui Xu, and Yuan Qi. Scp-116k: A high-quality problem-solution dataset and a generalized pipeline for automated extraction in the higher education science domain, 2025. URL https://arxiv.org/abs/2501.15587.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jefrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. URL https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-P review-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2. Notion Blog.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. Teaching small language models to reason, 2023. URL https://arxiv.org/abs/2212.08410.

Mathematical Association of America. AIME 2024 problems and solutions, 2024. URL https://artofprobl emsolving.com/wiki/index.php/2024\_AIME\_I.

Mathematical Association of America. AIME 2025 problems and solutions, 2025. URL https://artofprobl emsolving.com/wiki/index.php/2025\_AIME\_I.

Niklas Muennighof, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

Long Phan, Alice Gatti, Nathaniel Li, Adam Khoja, Ryan Kim, Richard Ren, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Dan Hendrycks, et al. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649(8099):1139–1146, 2026. doi: 10.1038/s41586-025-09962-4. URL https: //doi.org/10.1038/s41586-025-09962-4.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, and Tianlu Wang. Learning to plan & reason for evaluation with thinking-llm-as-a-judge, 2025. URL https://arxiv.org/abs/2501.18099.

William Saunders, Catherine Yeh, Jef Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Selfcritiquing models for assisting human evaluators, 2022. URL https://arxiv.org/abs/2206.05802.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, and Andrew Gordon Wilson. Does knowledge distillation really work?, 2021. URL https://arxiv.org/abs/2106.05945.

Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, and Andreas Köpf. Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards, 2025. URL https://arxiv.org/abs/2505.24760.

Ryan Teknium, Roger Jin, Jai Suphavadeeprasit, Dakota Mahan, Jefrey Quesnelle, Joe Li, Chen Guang, Shannon Sands, and Karan Malhotra. Hermes 4 technical report, 2025. URL https://arxiv.org/abs/2508.182 55.

Yijun Tian, Yikun Han, Xiusi Chen, Wei Wang, and Nitesh V. Chawla. Beyond answers: Transferring reasoning capabilities to smaller llms using multi-teacher knowledge distillation, 2024. URL https://arxiv.org/ abs/2402.04616.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation, 2022. URL https: //arxiv.org/abs/1910.10699.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https: //arxiv.org/abs/2201.11903.

Han Wu, Yuxuan Yao, Shuqi Liu, Zehua Liu, Xiaojin Fu, Xiongwei Han, Xing Li, Hui-Ling Zhen, Tao Zhong, and Mingxuan Yuan. Unlocking eficient long-to-short llm reasoning with model merging, 2025. URL https://arxiv.org/abs/2503.20641.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by writing less, 2025. URL https://arxiv.org/abs/2502.18600.

Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Thomas L. Grifiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023. URL https://arxiv.or g/abs/2305.10601.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning, 2025. URL https://arxiv.org/abs/2502.03387.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges, 2025. URL https://arxiv.org/abs/2310.17631.

## A. Cross-Feedback Review Prompt

This section provides the complete prompt template used in the cross-feedback refinement step (Section 2.1) to instruct teacher models to critically review peer reasoning chains across mathematical, coding, and logical reasoning tasks.

[ SYSTEM ]   
You are an expert reasoning evaluator specializing in mathematical problem - solving , code   
generation , and logical deduction . Your task is to critically review a reasoning chain from   
another model and generate an improved solution .   
Be rigorous and critical in your evaluation . Do not accept superficially correct answers --   
identify subtle logical gaps , unstated assumptions , and potential edge cases that the   
original solution may have overlooked .   
[ PROBLEM ]   
{ problem }   
[ DOMAIN ]   
{ domain } # One of: MATH , CODE , LOGIC   
[ PEER REASONING CHAIN ]   
The following is a reasoning chain from another teacher model . Your task is to review it   
critically and improve upon it .   
--- Other Teacher ’s Output ---   
{ other\_teacher\_reasoning\_chain }   
[ REVIEW INSTRUCTIONS ]   
Analyze the peer chain thoroughly . Your review must cover both general reasoning quality and   
domain - specific criteria .   
=== General Reasoning Criteria ===   
1. Logical Flow :   
- reasoning : coherence - Are transitions between steps justified ?   
- reasoning : completeness - Are all necessary steps present ?   
- reasoning : assumptions - Are implicit assumptions stated and valid ?   
2. Error Detection :   
- reasoning : contradiction - Does any step contradict previous ones ?   
- reasoning : circular - Is there circular reasoning ?   
reasoning : leap - Are there unjustified logical leaps ?   
3. Quality Assessment :   
reasoning : clarity - Is the reasoning clear and followable ?   
- reasoning : efficiency - Is the solution unnecessarily verbose ?   
reasoning : novelty - Does it offer creative insights ?   
=== Domain - Specific Criteria ===   
[If MATH ]:   
math : algebraic\_validity - Are algebraic manipulations correct ?   
math : arithmetic - Are calculations accurate ?   
math : constraint\_satisfaction - Are problem constraints respected ?   
math : case\_coverage - Are all cases ( including edge cases ) handled ?   
math : theorem\_applicability - Are theorems applied with valid preconditions ?   
math:unit consistency - Are units handled correctly (if applicable)?   
[If CODE ]:   
code : compilable - Does the code compile / run without syntax errors ?   
code : correctness - Does it produce correct output for given inputs ?   
code : edge\_cases - Are boundary conditions handled ?   
code : efficiency - Is time / space complexity reasonable ?   
code : readability - Is the code well - structured and readable ?   
code : error\_handling - Are potential runtime errors handled ?   
[If LOGIC ]:   
logic : premise\_validity - Are premises clearly stated and valid ?   
logic : inference\_rules - Are deduction rules correctly applied ?   
logic : quantifier\_scope - Are quantifiers (all , some , none ) used correctly ?   
logic : negation - Are negations handled properly ?   
logic : counterexample - Has the solution considered counterexamples ?   
logic : soundness - Is the argument logically sound ?   
[ TASK ]   
Based on your critical review :   
1. Identify ALL errors , gaps , and weaknesses (be thorough , not lenient )   
2. Note any strengths worth preserving   
3. Generate an improved reasoning chain that fixes identified issues   
[ OUTPUT FORMAT ]   
<review >

[ Critical analysis of the peer chain ]   
- Strengths : [ What to preserve ]   
- Weaknesses : [ Errors , gaps , issues found - be specific with tags ]   
- Verdict : [ ACCEPT\_WITH\_MINOR\_FIXES | NEEDS\_MAJOR\_REVISION | REJECT\_AND\_REDO ]   
</ review >   
<think >   
[ Your improved step -by - step reasoning that addresses all identified issues ]   
</think >   
<answer >   
[ Final answer ]   
</ answer >

## B. Mathematical Scoring Rubric (T-LCM)

This section provides the full scoring rubric used for mathematical reasoning evaluation. The T-LCM (Taxonomy-Guided Logic-Consistency Metric) protocol operates in two phases: domain classification at the problem level, followed by step-wise scoring with explicit decision boundaries.

## B.1. Score Range and Decision Boundaries

Table 6: T-LCM scoring rubric with decision boundaries for mathematical reasoning.
<table><tr><td>Score</td><td>Grade</td><td>Description</td><td>Decision Boundary</td></tr><tr><td>10.0</td><td>Rigorous</td><td>Flawless execution, all constraints satisfied, all cases covered.</td><td>N/A (Perfect)</td></tr><tr><td>8.0-9.9</td><td>Valid</td><td>Correct logic with minor inefficiency (redun- dancy, non-standard notation).</td><td>Does inefficiency obscure argument?</td></tr><tr><td>6.0-7.9</td><td>Minor Slip</td><td>Recoverable execution errors (calculation, sign, transcription, off-by-one).</td><td>Can error be fixed by single value change?</td></tr><tr><td>3.0-5.9</td><td>Structural Flaw</td><td>Significant reasoning errors (constraint viola- Is there any mathematical basis? tion, theorem misuse, missing cases).</td><td></td></tr><tr><td>0.0-2.9</td><td>Critical Failure</td><td>Fundamental logic breakdown (hallucination, Is reasoning entirely baseless? logical leap, circular reasoning).</td><td></td></tr></table>

## B.2. Error Taxonomy

We categorize errors into three types based on severity and recoverability:

Type I (Execution Errors, 6.0–7.9): Calculation errors where correct formula is applied but arithmetic is wrong; sign errors with correct equation structure; transcription errors copying values incorrectly; of-by-one boundary errors in discrete domains; unit oversight with approximately correct magnitude.

Type II (Reasoning Errors, 3.0–5.9): Constraint violations ignoring core problem conditions; conceptual misapplication using theorems where preconditions are not met; incomplete case analysis missing critical branches; variable scope errors reusing symbols for diferent meanings; premature approximation causing significant deviation.

Type III (Logic Breakdown, 0.0–2.9): Hallucination inventing non-existent theorems or values; logical leaps concluding without justification; circular reasoning using conclusions as premises; contradiction with previous steps; irrelevance to the problem.

## B.3. Domain-Specific Checklists

The scoring protocol is domain-adaptive. For algebra, we check sign consistency, factoring validity, inequality direction, and domain restrictions. For geometry, we verify diagram consistency, angle/length properties, and similarity/congruence conditions. For number theory, we validate modular arithmetic, divisibility claims, and integer constraints. For combinatorics, we examine counting methods, overcounting adjustments, and boundary conditions.

## B.4. Full Judge Prompt Template

Below is the complete prompt template used for mathematical reasoning evaluation, inspired by xVerify (Chen et al., 2025):

```yaml
[ SYSTEM ]
You are a Mathematics Olympiad Judge . Evaluate the given reasoning step using the T- LCM protocol
[ DOMAIN ] { domain }
[ RUBRIC ]
10.0 ( Rigorous ): Flawless deduction , all constraints satisfied , all cases covered .
- 8.0 -9.9 ( Valid ): Correct logic but contains redundancy or non - standard notation .
- 6.0 -7.9 ( Minor Slip ): Execution errors ( calculation , sign , transcription , off -by - one ) with
correct structure .
- 3.0 -5.9 ( Structural Flaw ): Reasoning errors ( constraint violation , theorem misuse , missing
cases ).
- 0.0 -2.9 ( Critical Failure ): Hallucination , logical leap , circular reasoning , contradiction .
[ DECISION BOUNDARIES ]
- 8.0 vs 7.9: Does inefficiency obscure the argument ? (Yes -> below 8.0)
- 6.0 vs 5.9: Can error be fixed by changing a single value ? ( Yes -> 6.0+)
- 3.0 vs 2.9: Is there any mathematical basis ? ( Yes -> 3.0+)
[ DOMAIN - SPECIFIC CHECKLIST : { domain }]
{ domain_checklist }
[ INPUT ]
Problem : { problem }
Previous Steps : { history }
Current Step : { current_step }
[ OUTPUT FORMAT ]
Score : [0.0 -10.0]
Error Type : [ None | Calculation | Sign | Transcription | Off -by - One | Constraint Violation |
Theorem Misuse | Missing Case | Hallucination | Logical Leap | Circular | Contradiction |
Irrelevance ]
Rationale : [ Brief explanation ]
```

## C. Coherence-Aware Thinking Path Algorithm

This section provides the detailed algorithm for Thinking Path, our coherence-aware pruning method that curates high-quality reasoning steps from multiple teacher outputs while preventing the Frankenstein efect, the selection of high-scoring but logically disconnected steps that form an incoherent chain.

## C.1. Algorithm Overview

Unlike naive greedy pruning that independently selects the highest-scoring step at each position, our Coherence-Aware Thinking Path algorithm optimizes a joint objective of Quality (from the aggregated JudgeLM and xVerify step scores) and Coherence (transition coherence between consecutive steps). This formulation is solved eficiently using Viterbi-style dynamic programming.

• Node Score $q ( s ) { \mathrm { ; } }$ : The aggregated step-quality score $v _ { t }$ (25% trimmed mean of the four JudgeLM and xVerify scores; Section 2.2), normalized to [0, 1].

• Edge Weight: The transition-coherence score between consecutive steps $s _ { t - 1 }$ and $s _ { t } ,$ computed from the cosine similarity of their embeddings and rescaled to [0, 1].

Complexity Analysis. The time complexity of our Coherence-Aware Path Finding is $O ( T \cdot N ^ { 2 } )$ , where $T$ is the maximum number of reasoning steps per chain and � is the number of teacher models. Since � is small $( N = 2$ in our experiments) and reasoning steps are bounded $( T < 2 0$ for most problems), the computational overhead is negligible compared to the LLM generation cost of initial generation and cross-feedback. The space complexity is $O ( T \cdot N )$ for storing the dynamic programming table and backpointers.

## C.2. Implementation

```python
def coherence_aware_thinking_path ( chains , lambda_ =0.7 , tau =7.0) :
chains = align_steps ( chains ) # embedding - based step alignment with lookahead merging
num_steps = max ( len (c) for c in chains )
num_teachers = len ( chains )
# DP Table : dp[t][i] = max cumulative score ending at teacher i, step t
dp = [[- float (’inf ’)] * num_teachers for _ in range ( num_steps )]
parent = [[ -1] * num_teachers for _ in range ( num_steps )]
# Initialization ( Step 0)
for i in range ( num_teachers ):
if len ( chains [i]) > 0:
dp [0][ i] = normalize ( chains [i ][0][ ’ score ’]) # Scale to 0-1
# Forward Pass ( Viterbi Search )
for t in range (1 , num_steps ) :
for curr_i in range ( num_teachers ):
if t >= len ( chains [ curr_i ]):
continue
# 1. Quality score ( aggregated JudgeLM / xVerify step score )
q_score = normalize ( chains [ curr_i ][t][’ score ’])
best_prev_val = -float (’inf ’)
best_prev_idx = -1
for prev_i in range ( num_teachers ):
if t -1 >= len ( chains [ prev_i ]):
continue
# 2. Coherence score C(u, v) ( transition coherence )
c_score = get_transition_coherence (
chains [ prev_i ][t -1][ ’ content ’],
chains [ curr_i ][t][’ content ’]
)
# 3. Joint Score : Quality + Coherence
transition_score = dp[t -1][ prev_i ] + \
( lambda_ * q_score ) + \
((1 - lambda_ ) * math .log ( c_score + 1e -9) )
if transition_score > best_prev_val :
best_prev_val = transition_score
best_prev_idx = prev_i
dp [t ][ curr_i ] = best_prev_val
parent [t][ curr_i ] = best_prev_idx
# Backward Pass (Path Reconstruction)
final_idx = num_steps - 1
best_end = max ( range ( num_teachers ), key = lambda i: dp[ final_idx ][i])
optimal_path = []
curr = best_end
for t in range ( final_idx , -1, -1):
step = chains [ curr ][t]
if step [’score ’] >= tau : # Final quality threshold
optimal_path . append ( step )
curr = parent [t][ curr ]
return optimal_path [:: -1] # Reverse to chronological order
def get_transition_coherence ( prev_content , curr_content ):
""" Compute C( prev , curr ) with rescaled embedding cosine similarity ."""
return (1 + cosine_similarity ( embed ( prev_content ), embed ( curr_content ))) / 2 # rescale [-1,
1] to [0, 1]
```

## C.3. Key Design Choices

The algorithm embodies several important design decisions that distinguish it from simple greedy selection.

The joint optimization of quality and coherence prevents the Frankenstein efect where high-scoring but disconnected steps form an invalid reasoning chain. The hyperparameter � controls this trade-of: higher values $( \lambda > 0 . 5 )$ prioritize step quality scores, while lower values emphasize smooth transitions between steps. In our experiments, � = 0.7 achieved the best balance, allocating 70% weight to quality and 30% to coherence.

The Viterbi-style dynamic programming ensures global optimality in $O ( T \cdot N ^ { 2 } )$ time, where � is the

maximum chain length and � is the number of teachers. This is eficient for typical settings $( N = 2 , T < 2 0 )$

## D. Training Data Composition

Table 7 lists the number of training instances drawn from each source.

Table 7: Composition of the CRD-50K training dataset.

<table><tr><td>Source Dataset</td><td>Domain</td><td>Samples</td></tr><tr><td>DeepScaleR</td><td>Math</td><td>20K</td></tr><tr><td>Eurus-2-RL</td><td>Code</td><td>15K</td></tr><tr><td>Reasoning Gym</td><td>Logical</td><td>10K</td></tr><tr><td>SCP-116K</td><td>STEM</td><td>5K</td></tr><tr><td>Total</td><td></td><td>50K</td></tr></table>

## E. Benchmark Descriptions

We evaluate CRD on nine benchmarks spanning mathematics, code generation, STEM reasoning, planning, and expert-level knowledge:

• MATH-500 (Lightman et al., 2023): Competition mathematics problems requiring multi-step reasoning across the seven MATH subjects (prealgebra, algebra, number theory, counting and probability, geometry, intermediate algebra, and precalculus).

• LiveCodeBench (Jain et al., 2024): Code generation benchmark with execution-based evaluation measuring functional correctness.

• GPQA-Diamond (Rein et al., 2023): Graduate-level questions in physics, biology, and chemistry requiring deep domain knowledge.

• AIME’24 (Mathematical Association of America, 2024) and AIME’25 (Mathematical Association of America, 2025): American Invitational Mathematics Examination problems spanning algebra, geometry, number theory, and combinatorics that demand creative multi-step solutions.

• GSM8K (Cobbe et al., 2021): Grade-school math word problems, used for the multi-teacher comparison (Table 2).

• HumanEval (Chen et al., 2021): Python function synthesis evaluated with unit tests, used for the multi-teacher comparison (Table 2).

• BoxNet (Stojanovski et al., 2025): Procedurally generated multi-step planning that moves colored boxes to their targets, used as an out-of-distribution test.

• HLE (Phan et al., 2026): Expert-level questions across many academic fields. We use the text-only subset.

## F. Detailed Analysis

## F.1. MATH-500 Subject Breakdown

MATH-500 covers seven subjects; we analyze algebra, precalculus, and number theory. To quantify the impact of collaborative feedback, we measured per-subject Chain-of-Thought accuracy by matching generated reasoning steps against oficial solution templates. Algebra shows the clearest improvement: on a 50-problem slice, the DeepSeek-R1-1.5B baseline attains 85.8% step correctness (482/562 steps), whereas CRD-1.5B reaches 92.2% (688/746 steps). Precalculus benefits by 6–8 percentage points in function analysis and limit computations, while number theory improves by 5–7 points in divisibility and modular-arithmetic reasoning.

## F.2. Scaling Analysis

Our results reveal consistent scaling patterns. On the DeepSeek-R1-Distill-Qwen-1.5B student, CRD improves MATH-500 by 7.2 points (83.9 → 91.1). For Qwen3 students, MATH-500 gains shrink as the base model

saturates $( 1 . 7 \mathrm { { B } } \colon + 1 . 1 , 4 \mathrm { { B } } \colon + 0 . 3 , 3 0 \mathrm { { B } } { \cdot } \mathrm { { A } } 3 \mathrm { { B } } \colon + 0 . 3 )$ , while gains on the harder AIME’25 persist $( + 5 . 4 , + 4 . 7 ,$ +2.3), suggesting that the marginal benefit of CRD diminishes on near-saturated benchmarks but remains on harder ones.

## G. Multi-Round Cross-Feedback Analysis

We analyze the efect of multiple cross-feedback rounds on final model performance.

Table 8: Impact of cross-feedback rounds on model performance. Evaluated on Qwen3-1.7B student model. Performance converges at round 2–3.
<table><tr><td>Round</td><td>MATH-500</td><td>GPQA-Diamond</td><td>LCB</td><td>Marginal Gain</td></tr><tr><td>Initial (0)</td><td>93.4</td><td>40.1</td><td>33.2</td><td></td></tr><tr><td>Round 1</td><td>93.8</td><td>41.5</td><td>34.7</td><td> $+ 0 . 4 ⁄ + 1 . 4 ⁄ + 1 . 5$ </td></tr><tr><td>Round 2</td><td>94.0</td><td>42.2</td><td>35.1</td><td> $+ 0 . 2 / + 0 . 7 / + 0 . 4$ </td></tr><tr><td>Round 3</td><td>94.0</td><td>42.2</td><td>35.0</td><td> $0 / 0 / - 0 . 1$ </td></tr></table>

Performance converges at round 2–3, with additional rounds providing no benefit. The marginal utility of cross-feedback diminishes rapidly, suggesting that 2 rounds achieve the optimal quality-eficiency trade-of.

## H. Judge Reliability

We reduce dependence on any single evaluator with two heterogeneous judges (JudgeLM and xVerify), two evaluations per judge, a 25% trimmed mean over the four scores, and domain-specific verification (execution checks for code and answer-equivalence checks for mathematics). Re-scoring all steps in our evaluation set with GPT-4o, given the same preceding history and current step, yields a mean absolute diference of 0.428 from our aggregated score on the 0–10 scale. On MATH-500, the full dual-judge configuration improves the student by 1.4 points, compared with 0.6 with xVerify alone, 0.4 with JudgeLM alone, and 0.6 with GPT-4o alone. Correlated evaluator biases may remain, and plausible but incorrect reasoning may still receive high scores.

## I. Stitched Training Example

Table 9 shows a stitched training trajectory for a problem that asks for the fraction of an equilateral triangle lying outside a circle tangent to two of its sides. The original chain derives the first two steps correctly but then assumes only an exterior-center configuration. Cross-feedback produces a corrected continuation, and Thinking Path selects $A _ { 1 }  A _ { 2 }  B _ { 3 }  B _ { 4 }  B _ { 5 }$

Table 9: A stitched training trajectory (�: original chain, �: cross-feedback continuation).
<table><tr><td>Step</td><td>Content</td><td>Score</td><td>Decision</td></tr><tr><td> $A _ { 1 }$ </td><td>Derives  $\angle B O C = 1 2 0 ^ { \circ }$ </td><td>9.3</td><td>Retain</td></tr><tr><td> $A _ { 2 }$ </td><td>Derives  $r = s / \sqrt { 3 }$ </td><td>9.0</td><td>Retain</td></tr><tr><td> $A _ { 3 }$ </td><td>Assumes only an exterior center</td><td>3.6</td><td>Reject</td></tr><tr><td> $A _ { 4 }$ </td><td>Applies an absolute value to repair a negative area</td><td>0.8</td><td>Reject</td></tr><tr><td> $B _ { 3 }$ </td><td>Identifies two feasible center configurations</td><td>9.4</td><td>Retain</td></tr><tr><td> $B _ { 4 }$ </td><td>Rejects the exterior case by contradiction</td><td>9.1</td><td>Retain</td></tr><tr><td> $B _ { 5 }$ </td><td>Computes the valid minor-segment area</td><td>8.9</td><td>Retain</td></tr></table>

## J. Student Output Examples

We compare DeepSeek-R1-Distill-Qwen-1.5B and CRD-1.5B on a held-out MATH-500 problem with identical prompts and decoding. The problem asks for $p ( 8 )$ , where $p ( x )$ is a polynomial of degree 5 such that $\textstyle p ( n ) = { \frac { n } { n ^ { 2 } - 1 } }$ for $n = 2 , 3 , \ldots , 7 .$

DeepSeek-R1-Distill-Qwen-1.5B (incorrect). The model defines $q ( x ) = x ^ { 2 } - 1 - p ( x )$ and asserts $q ( 2 ) =$ $\begin{array} { r } { \cdots = q ( 7 ) = 0 } \end{array}$ , which does not follow from the condition. It then writes this degree-5 polynomial as a product of six linear factors and outputs $p ( 8 ) = - 6 5 7$

CRD-1.5B (correct). The model constructs $Q ( x ) = ( x ^ { 2 } - 1 ) p ( x ) - x$ , notes that $Q ( n ) = 0$ for $n = 2 , \ldots , 7 .$ keeps the degree-7 structure, and uses $Q ( 1 ) = - 1$ and $Q ( - 1 ) = 1$ to determine the remaining linear factor. It obtains $Q ( 8 ) = - 2 5 9 / 5 6$ and $6 3 p ( 8 ) - 8 = - 2 5 9 / 5 6$ , and returns $p ( 8 ) = 3 / 5 6$