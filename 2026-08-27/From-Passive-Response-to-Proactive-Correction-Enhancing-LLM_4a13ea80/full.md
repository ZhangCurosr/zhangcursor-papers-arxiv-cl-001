# From Passive Response to Proactive Correction: Enhancing LLM Robustness Against Input Fact Perturbations

Ping Wang<sup>1</sup> Xiangguo Sun<sup>2</sup>\*, Bingbing Xu<sup>1</sup> Guocong Li<sup>3</sup> Xiaofeng Meng<sup>1</sup> <sup>1</sup>Renmin University of China <sup>2</sup>Southeast University <sup>3</sup>Zhejiang University {p\_wang,xvbingbing,xfmeng}@ruc.edu.cn hsiang-kuo.sun@seu.edu.cn, liguocong@zju.edu.cn

## Abstract

Large language models (LLMs) frequently produce confident yet factually incorrect responses when user inputs contain misleading premises, a phenomenon we attribute to fact perturbations in the input. Existing approaches to hallucination mitigation typically assume reliable user inputs, overlooking how such factual errors can actively mislead model reasoning. To address this vulnerability, we propose DEDUCE, a three-stage framework that transforms LLMs from passive responders into proactive error correctors. DEDUCE operates in three stages: (1) detect errors through fine-grained fact extraction and verification; (2) devise correction strategies via multi perspective deliberation; and (3) correct misconceptions while delivering reliable answers. We also present Mis-FactQA, a dataset containing factual errors of varying degrees, and propose new metrics for evaluating model robustness. Experiments on TruthfulQA, FalseQA, and our MisFactQA benchmark demonstrate that DEDUCE significantly improves both accuracy and error correction capability. Consistent gains across Qwen, LLaMA, and Gemma families confirm its effectiveness and scalability.<sup>1</sup>

## 1 Introduction

In recent years, large language models (LLMs) have demonstrated remarkable capabilities in encoding real-world knowledge within their parameters and using it to assist with complex tasks (Ge et al., 2023; Lyu et al., 2024; Li et al., 2025b). Despite their impressive performance on a wide range of generative tasks, including translation (Lu et al., 2024), mathematical reasoning (Xia et al., 2025), code generation (Liu et al., 2026), and value alignment (Xu et al., 2025), LLMs remain highly vulnerable to hallucinations, producing outputs that appear plausible yet are factually incorrect (Huang et al., 2025; Li et al., 2025a, 2026).

![](images/38ac3e80b467a184098c6e648830850f2a4b3494094b86ec31e1c234c770d8a6.jpg)  
Figure 1: A query containing factually incorrect or inconsistent premises can mislead an LLM into producing a confident but unreliable answer (red). The desired behavior is to identify the flawed premise, explicitly clarify the error, and then provide a corrected, factual response (green).

Existing research typically relieves hallucination along pre-training, supervised fine-tuning, and inference stages, nearly covering the entire pipeline of developing an LLM. Specifically, in the pretraining stage, hallucinations are traced to flawed corpora (Penedo et al., 2023; Li et al., 2023). During supervised fine-tuning, hallucinations frequently arise when models are compelled to answer beyond their parametric knowledge, prompting work on teaching models to appropriately express uncertainty (Cheng et al., 2024a). The inference stage has attracted the most attention due to its flexibility and low cost (Ji et al., 2023; Zhang et al., 2024; Tonmoy et al., 2024; Cheng et al., 2025; Sriramanan et al., 2024; Li et al.), where hallucinations stem from imperfect decoding and insuficient grounding. Accordingly, prior studies have proposed faithful decoding strategies (Dhuliawala et al., 2024) , retrieval-augmented generation (Béchard and Ayala, 2024), and uncertaintybased detection methods (Cheng et al., 2024b).

![](images/32df2730522b304a6f5c6f550269d8910a957af56983a654c4ed50c414c9c77d.jpg)  
Figure 2: Models are strongly misled by erroneous inputs. On Qwen2.5-7B and LlaMA-3.1-8B, we evaluate four query types built from the same underlying information. Accuracy is high on correct questions (blue) but drops by 30%-60% points when errors are injected.

Although progress has been made, current research still focuses almost entirely on the product end, that is, on improving LLM capabilities. This overlooks a notable reality at the user end (shown in Figure 1): in real-world applications, users are rarely domain experts, and their queries often contain factual errors. These are factual mistakes stemming from flawed cognition, including incorrect assumptions, contradictory statements, and complex errors. Such errors can perturb LLM reasoning processes and induce hallucinations as well.

In this paper, we identify a critical yet frequently neglected factor: even if model development is optimized at every stage against hallucinations, the input quality of real-world users can still lead to inevitable hallucination issues in LLMs. Figure 2 further demonstrates this, showing that even injecting a small number of factual errors into the input (e.g., false premises or contradictory descriptions) can lead to a sharp drop in model accuracy. Surprisingly, even when LLMs possess correct underlying knowledge (Yuan et al., 2024; Guo et al., 2025), such misleading inputs can override internal representations and derail reasoning. Existing methods typically adopt two passive strategies for inputs with false premises or factual errors: they either ignore the errors, thereby propagating misinformation, or attempt self correction but often fail because reasoning is biased by the input.

In light of the above, this work aims to advance research in this direction by addressing the following core question: how can we enhance the reliability of LLM responses when the input contains factual errors? Tackling this problem is critically important yet particularly hard due to three main challenges: Challenge 1, user errors are often implicit and varied in form, making them dificult for models to precisely identify and localize. Such errors may be embedded as false statements, incorrect presuppositions, internal inconsistency, or layered in compound queries, requiring deep understanding to uncover. Challenge 2, even when errors are detected, it is non-trivial for a model to reliably correct all inaccuracies while still providing a helpful and accurate answer. The correction process itself can introduce new inconsistencies or lead to incomplete responses, demanding both rigorous error resolution and faithful adherence to the corrected context. Challenge 3, existing benchmarks lack complex erroneous queries, and conventional evaluation metrics fail to capture nuanced model behaviors. In real scenarios, models may only partially correct errors, correct but refuse to answer, or answer without rectifying the error.

To address these challenges, we conduct a systematic investigation into input-induced hallucinations and propose a framework that transforms LLMs from passive responders into proactive evaluators. For the first challenge, we introduce a fine-grained Detect module that precisely locates and analyzes factual errors in user inputs. For the second challenge, we propose a strategy generation module coupled with a Multi-Perspective Strategy Deliberation mechanism, which generates and rigorously verifies response strategies tailored to the diagnosed errors. For the third challenge, we ofer a dataset that simulates complex real-world user factual errors, and introduce new evaluation metrics to assess how well models respond when confronted with misleading information. Our contributions are summarized as follows:

• We present a noteworthy perspective on mitigating hallucinations in LLMs, highlighting that they arise not only by model fabrication but also from factual perturbations in user inputs that mislead model reasoning.

• We propose a “detect, devise and correct” framework to enhance models’ robustness and reliability under misleading input.

• We ofer MisFactQA, a more comprehensive and objective benchmarking dataset containing diverse and complex factual errors, with fine-grained metrics to evaluate model robustness against misleading inputs.

## 2 Background and Problem Statement

Motivation As previously mentioned, users are rarely domain experts, and their queries often contain factual errors that can perturb LLM reasoning. We argue that robust LLMs should enhance the capacity of identifying and correcting such errors, rather than passively inherit flawed inputs.

Problem Definition Let $Q \ = \ ( q , { \mathcal K } )$ denote a user input, where q is the query and $\begin{array} { r l } { \mathcal { K } } & { { } = } \end{array}$ $\{ k _ { 1 } , \ldots , k _ { m } \}$ is the set of factual units extracted from it, encompassing both explicit statements and implicit assumptions. We define an indicator function $\mathcal { E } ( Q ) \in \{ 0 , 1 \}$ , where $\varepsilon ( Q ) = 1$ identifies $Q$ as a factually perturbed input. Such perturbations take three forms: false premise, where some $k _ { i }$ is factually incorrect and causes the model to reason from a wrong starting point; factual contradiction, where inconsistent elements co-exist in K, leaving the model without a reliable basis for reasoning; and compound error, where multiple erroneous units cooccur such that detecting one does not prevent the remaining errors from misleading the model. When $\varepsilon ( Q ) = 1$ , LLMs are prone to distorted intermediate reasoning trajectories, ultimately producing responses that inherit or amplify misinformation.

Objective We formally define the task of Fact-Perturbed Question Answering (FPQA), aiming to enhance language model robustness against misleading inputs. Specifically, we develop a framework on top of various LLM backbones $\ b { M } ^ { * }$ , which is capable of: (1) accurately detecting when $\mathcal { E } ( Q ) =$ 1, (2) explicitly correcting factual errors in K, and (3) generating reliable responses grounded in verified knowledge rather than erroneous user context.

## 3 DEDUCE: DEtect-Devise-CorrEct

In this section, we present our input-side hallucination mitigation framework, DEDUCE, which addresses a fundamental limitation of existing approaches: when facing factually perturbed inputs, models either propagate misinformation by answering directly, or fail at self-correction by remaining trapped within their own reasoning biases. The core insight of DEDUCE is to reformulate this challenge into an explainable, debatable, and verifiable collaborative process. As illustrated in Figure 3, DEDUCE comprises three synergistic modules, namely Detect, Devise, and Correct, that jointly enable the model to proactively surface errors, strategically reason under misleading inputs, and produce reliable, verifiably grounded responses.

## 3.1 Detect: Input Detection and Analysis

Errors in user queries are often concealed within narratives that appear plausible. When a model perceives the input as a whole, it is susceptible to surface level semantic misleading and prone to overlooking subtle but critical inaccuracies. To address this, we introduce an atomic fact detection mechanism that decomposes $Q$ into a set of minimal, independently verifiable factual units.

Atomic Fact Decomposition Given a query $Q ,$ we define a decomposition function $\phi$ that maps Q to its set of atomic facts:

$$
\phi ( Q ) = \{ A C _ { 1 } , A C _ { 2 } , \ldots , A C _ { n } \} ,\tag{1}
$$

where each $A C _ { i }$ is a minimal, self-contained factual assertion extracted from the query.

First, it surfaces implicit errors: even when the query appears superficially coherent, all latent factual errors are exposed, enabling more comprehensive detection of potential misinformation. Second, it provides an actionable basis for verification: each unit is independently assessed via a truthfulness check $\mathcal { F } ( \cdot )$ and a pairwise consistency check $C ( \cdot , \cdot )$ , defined as:

$$
\mathcal { F } ( A C _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } A C _ { i } \mathrm { i s } \mathrm { f a c t u a l l y i n c o r r e c t } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{2}
$$

$$
C ( A C _ { i } , A C _ { j } ) = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f } } \ A C _ { i } \ { \mathrm { a n d } } \ A C _ { j } \ { \mathrm { a r e } } } \\ { \ } & { { \mathrm { m u t u a l l y ~ i n c o n s i s t e n t } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{3}
$$

The two checks give rise to disjoint error sets ${ \mathcal { E } } _ { \mathrm { f a c t } } ~ = ~ \{ A C _ { i } ~ \mid ~ { \mathcal { F } } ( A C _ { i } ) ~ = ~ 1 \}$ and $\begin{array} { r l } { \mathcal { E } _ { \mathrm { c o n f l i c t } } } & { { } = } \end{array}$ $\{ ( A C _ { i } , A C _ { j } ) \mid C ( A C _ { i } , A C _ { j } ) = 1 \}$ , enabling precise identification of both error type and location. The overall perturbation indicator defined in Eq. 4 can thereby be rewritten as:

$$
\mathcal { E } ( Q ) = \mathbb { 1 } \left[ \left| \mathcal { E } _ { \mathrm { f a c t } } \right| + \left| \mathcal { E } _ { \mathrm { c o n f i c t } } \right| > 0 \right] .\tag{4}
$$

Diagnostic Summary Abstract factual judgments are subsequently grounded into concrete natural language claims, ensuring alignment between high-level reasoning and specific factual assertions. The Detect module ultimately produces a structured diagnostic summary:

$$
\begin{array} { r } { \mathbf { M } \mathrm { I s S U M } ( Q ) = \mathbf { S } \mathbf { u m m a r i z e } ( \mathcal { E } _ { \mathrm { f a c t } } \cup \mathcal { E } _ { \mathrm { c o n f i c t } } ) , } \end{array}\tag{5}
$$

![](images/cefb7afefa8af52bb8b2935b1e367d63a6af5cb8b815d8bf47686e21e5c77788.jpg)  
Figure 3: Overall architecture of our DEtect-Devise-CorrEct Framework.

which specifies whether errors exist, where they occur, and why they are problematic, providing an interpretable foundation for strategy formulation.

## 3.2 Devise: Multi-Perspective Deliberation

Even when errors are correctly identified, a single model may still undercorrect by overlooking subtler errors, overcorrect by invalidating valid content, or introduce new inconsistencies during correction. All these failure modes share a common root cause: the model is constrained by its own reasoning frame and cannot adequately challenge its initial assumptions. To address this, we propose a multi-perspective strategy debate mechanism that decomposes correction strategy formulation into three complementary roles.

All three roles condition on the model’s internal knowledge and on MisSum(Q), which encodes the factual and conflict errors identified in Q.

Generator (G) The generator produces an initial draft strategy $s ^ { ( 0 ) }$ grounded in MisSum(Q), aiming to enumerate all plausible correction pathways:

$$
s ^ { ( 0 ) } = { \mathcal { G } } ( Q , \mathbf { M a s S u m } ( Q ) ) .\tag{6}
$$

Reviewer (R) The reviewer takes an adversarial stance, analogous to a peer reviewer, and identifies weaknesses in $s ^ { ( 0 ) }$ along three dimensions: (1) completeness, assessing whether all erroneous facts, misleading assumptions, and conflicts in $\mathcal { E } _ { \mathrm { f a c t } } \cup \mathcal { E } _ { \mathrm { c o n f l i c t } }$ are addressed; (2) accuracy, penalizing overcorrection or unsubstantiated claims; (3) reliability, evaluating whether the strategy sufices to guide the model toward a correct and on-topic response. It produces a structured critique:

$$
r = \mathcal { R } \big ( s ^ { ( 0 ) } , \mathbf { M } \mathrm { { r s } } { \mathbf { S } } _ { \mathbf { U M } } ( Q ) \big ) ,\tag{7}
$$

where r encodes any deficiencies in the draft strategy. If no deficiencies are found, r may be empty and $s ^ { ( 0 ) }$ is adopted directly.

Arbiter (A) The arbiter intervenes whenever $r \neq \emptyset$ , i.e., only when the reviewer identifies deficiencies in the draft strategy. Acting without bias toward either party, it impartially evaluates both perspectives and selectively incorporates their valid insights to generate a final strategy that is comprehensive, accurate, and actionable:

$$
\pi ^ { * } ( Q ) = { \mathcal { A } } { \left( s ^ { ( 0 ) } , r \right) } .\tag{8}
$$

Role separation mitigates self reinforcing bias common in single model self correction. The generator proposes, the reviewer critiques, and the arbiter resolves disagreements. This produces a multi perspective validated strategy $\pi ^ { * } ( Q )$ before execution.

## 3.3 Correct: Correction and Response

In the final stage, the model executes the validated strategy $\pi ^ { * } ( Q )$ to generate a response. Even a carefully devised strategy may fail if the model deviates during generation, so we guide the model to follow the strategy’s steps sequentially. Each step corresponds to a concrete action in the correction pipeline: (1) Error Identification, explicitly pointing out factual errors in the input; (2) Correction with Justification, providing the correct information along with supporting reasoning; and (3) Reliable Answering, producing a final answer to the original question under the corrected premises.

By following this sequential execution, the model ensures that all identified errors are addressed systematically, that corrections are justified, and that the final response faithfully reflects the validated strategy.

## 3.4 Implementation Strategies

We implement the DEDUCE framework using two complementary strategies: DEDUCE-Prompting and DEDUCE-Tuning.

DEDUCE-Prompting DEDUCE-Prompting decomposes into three modules. The Detect module identifies factual errors and conflicts in the input. The Devise module generates multi-perspective correction strategies using Generator, Reviewer, and Arbiter roles to ensure comprehensive and reliable strategy formulation. The Correct module executes the validated strategy and produces a final answer with justifications. This strategy is easy to deploy, fully interpretable, and flexible across tasks. It allows direct use of existing language models without additional training. Prompts are provided in Appendix A.2

DEDUCE-Tuning DEDUCE-Tuning aims to internalize the reasoning patterns of DEDUCE into model parameters, enabling eficient and strategyfaithful correction without multi-step prompting.

Stage 1: Detect Fine-Tuning In the first stage, the model learns to identify factual errors and conflicts. Training data is generated using DEDUCE-Prompting with GPT-4o as the teacher. Each input query is paired with its diagnostic summary and error labels. Only examples where the teacher outputs are fully correct are included. This stage ensures the model develops a reliable foundation for factual error detection.

Stage 2: Devise and Correct Fine-Tuning The second stage integrates the Devise and Correct modules. Training examples are generated using Qwen2.5-14B as the teacher, including validated multi-perspective strategies and the corresponding correct answers. We filter examples based on a strict quality criterion, keeping only those with a clarify score of 5 and verified correctness. By combining Devise and Correct into a single fine-tuning stage, the model internalizes multi-perspective reasoning and executes corrections in one step, mitigating the risk of self-reinforcing biases. Further implementation details are provided in Appendix B.1.

## 4 Benchmark

Terminology We extend the term clarification beyond resolving ambiguity to include the process by which the model identifies factual errors in user inputs and provides corrected information.

## 4.1 Benchmark Metrics

To evaluate model behavior on queries containing factual errors, we introduce three complementary metrics: Misleading Rate, Correction Rate, and Clarification Score. Together, these metrics assess whether a model (i) is misled by erroneous inputs, (ii) corrects the underlying inaccuracies, and (iii) provides reliable answer.

Existing evaluation practices are insuficient for this setting. Traditional metrics such as BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) measure surface level textual similarity and cannot evaluate factual correctness. Accuracy only captures whether the final answer is correct, but model responses often exhibit varying degrees of correctness rather than being entirely right or wrong. Our metrics address these gaps by measuring susceptibility to misleading inputs, ability to recover correct facts, and partial correctness.

We define the set of erroneous queries as:

$$
Q _ { \mathrm { e r r } } = \{ Q \mid { \mathcal { E } } ( Q ) = 1 \} .\tag{9}
$$

Misleading Rate (MR) This metric measures the model’s susceptibility to input-induced hallucinations. It is defined as the proportion of erroneous queries where the model fails to recognize the error and instead adopts the user’s false premise.

$$
\begin{array} { r } { \mathbf { M } \mathbf { R } = \frac { \sum _ { Q \in Q _ { \mathrm { e r r } } } L ( Q ) } { | Q _ { \mathrm { e r r } } | } , } \end{array}\tag{10}
$$

where $L ( Q ) = 1$ if the response accepts the false premise, 0 otherwise.

Correction Rate (CR) This metric measures the model’s ability to detect errors. It calculates the proportion of erroneous inputs where the model identifies most false or contradictory claims. If the model recognizes most errors, it is considered to have error detection capability.

$$
\begin{array} { r } { \mathrm { C R } = \frac { \sum _ { Q \in Q _ { \mathrm { e f f } } } R ( Q ) } { | Q _ { \mathrm { e f f } } | } , } \end{array}\tag{11}
$$

where $R ( Q ) = 1$ if the response corrects the errors, and 0 otherwise.

Clarification Score(CS) The Clarification Score measures how efectively a model identifies and addresses factual errors in a query. The specific definition is shown in Table 1.

<table><tr><td>Score</td><td>Description</td></tr><tr><td>1</td><td>Misled: The model accepts the false claims and reinforces the error.</td></tr><tr><td>2</td><td>Avoidance: The model ignores the error and answers directly, resulting in an incorrect re-</td></tr><tr><td>3</td><td>sponse. Contradictory: The model identifies some false claims but gives contradictory responses.</td></tr><tr><td>4</td><td>Partial Correction: The model identifies most false claims but does not adequately answer</td></tr><tr><td>5</td><td>the question under the correct premise. Full Clarification: The model explicitly iden- tifies the error, refutes it with correct facts, and provides the accurate answer.</td></tr></table>

Table 1: Clarification Score Rubric for evaluating model responses to erroneous queries.

We define $L ( Q ) = 1$ when ${ \mathrm { C S } } \in \{ 1 , 2 \}$ , indicating the model is completely misled, and $R ( Q ) = 1$ when $\mathrm { C S } \in \{ 4 , 5 \}$ , indicating adequate error correction capability.

## 4.2 FPQA Dataset

We construct MisFactQA for the FPQA task by combining three sources with verified ground truth: a subset of Prize (Yuan et al., 2024), EchoMist (Guo et al., 2025), and publicly available question answering datasets. We create queryresponse pairs that require models to both identify false information in the input and generate the corresponding correction and answer.

Existing work primarily focuses on single error in the input and evaluates models based only on accuracy, which is insuficient to measure robustness under factual perturbed inputs. To address this limitation, we introduce three types of errors that test factual accuracy, consistency, and comprehensiveness. Queries with a single false premise can mislead models to reason from an incorrect starting point. Queries with internally contradictory descriptions force the model to choose between conflicting claims, making it prone to inconsistent reasoning. Queries containing multiple errors, such as several false premises or a false premise combined with a contradiction, may cause the model to detect one error while overlooking others, further challenging its reasoning process. Further construction details are provided in the Appendix B.2.

## 5 Experiments

In this section, we conduct comprehensive experiments to verify the efectiveness of DEDUCE.

Specifically, we aim to address the following research questions: RQ1 (Efectiveness): How well does our framework perform compared with baseline methods? RQ2 (Error Analysis): How do various types of factual errors influence model behavior? RQ3 (Component Contribution): How does each component of the proposed method contribute to overall performance? RQ4 (Robustness and Eficiency): Does the vulnerability to fact perturbed inputs persist in stronger LLMs, and what is the inference cost of applying DEDUCE?

## 5.1 Experimental Setup

Datasets and Evaluation Metrics We evaluate DEDUCE on three benchmark datasets: TruthfulQA (Lin et al., 2022), which contains misleading questions; FalseQA (Hu et al., 2023), a benchmark for detecting false premises; and MisFactQA, the dataset we construct in Section 4.2 to assess model performance under diverse factual errors.

Following previous studies, we evaluate all models using accuracy (Acc), where a model’s response is considered correct only if it contains the ground truth answer. We additionally report three metrics: Misleading Rate (MR), measuring how often models are completely misled; Correction Rate (CR), reflecting a reasonably good ability to correct errors; and Clarification Score (CS), a 1-5 scale rating ranging from fully misled to complete correction with accurate responses.

Baseline Methods We compare DEDUCE with five representative categories of baselines: Default Model: The original LLM with a standard system prompt. ICL (Brown et al., 2020): In context learning with demonstration examples. CoT (Wei et al., 2022): Chain-of-thought prompting for step-bystep reasoning. LoRA Fine Tuning (Hu et al., 2022): Parameter eficient supervised fine tuning. IAQ FA (Wang and Blanco, 2025): An interpretable false assumption detection method.

Models and Implementation Details We employ models from the Qwen (Yang et al., 2025), Llama (Grattafiori et al., 2024), and Gemma (Team et al., 2024) families. Experiments are conducted using Ollama<sup>2</sup> with temperature = 0.0. Following the prior work (Li et al., 2025c), we use o3- mini-medium as an automated judge for evaluation, given its strong performance on reasoning and knowledge-intensive tasks in JudgeBench (Tan et al., 2024b). Comprehensive details regarding datasets, baselines, and implementation settings are provided in Appendix A. We further assessed the reliability of the automated scoring process through human evaluation. The LLM judge shows near-perfect agreement with human judgments on accuracy, with Cohen’s $\kappa = 0 . 9 3 6 .$ , and strong alignment on the clarification score, with Pearson’s correlation coeficient $r = 0 . 9 1 6$ . These results support the validity of our evaluation metrics. Additional details are provided in Appendix F.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="4">FalseQA</td><td colspan="4">MisFactQA</td></tr><tr><td>Acc ↑</td><td>CS↑</td><td>MR↓</td><td>CR↑</td><td>Acc ↑</td><td>CS↑</td><td>MR↓</td><td>CR↑</td></tr><tr><td rowspan="6">Qwen2.5-7B-Instruct</td><td>Original</td><td>51.64</td><td>3.44</td><td>44.15</td><td>54.13</td><td>39.04</td><td>3.22</td><td>51.52</td><td>43.49</td></tr><tr><td>ICL</td><td>53.25</td><td>3.58</td><td>39.36</td><td>57.45</td><td>49.36</td><td>3.56</td><td>37.48</td><td>52.83</td></tr><tr><td>CoT</td><td>52.61</td><td>3.56</td><td>40.48</td><td>55.84</td><td>45.35</td><td>3.36</td><td>46.10</td><td>50.37</td></tr><tr><td>SFT</td><td>65.74</td><td>4.18</td><td>20.47</td><td>74.60</td><td>61.19</td><td>3.94</td><td>23.47</td><td>65.16</td></tr><tr><td>IAQ-FA</td><td>60.81</td><td>3.95</td><td>31.29</td><td>67.55</td><td>44.69</td><td>3.40</td><td>44.69</td><td>50.37</td></tr><tr><td>DEDUCE-P</td><td>70.59</td><td>4.29</td><td>19.12</td><td>78.53</td><td>61.24</td><td>4.08</td><td>18.23</td><td>70.62</td></tr><tr><td rowspan="9"></td><td>DEDUCE-T</td><td>77.84</td><td>4.32</td><td>19.72</td><td>78.56</td><td>64.65</td><td>4.08</td><td>21.48</td><td>75.20</td></tr><tr><td>Original</td><td>43.75</td><td>3.10</td><td>54.84</td><td>42.66</td><td>30.16</td><td>3.01</td><td>57.49</td><td>36.23</td></tr><tr><td>ICL</td><td>57.24</td><td>3.74</td><td>33.23</td><td>63.46</td><td>63.98</td><td>4.15</td><td>18.20</td><td>73.55</td></tr><tr><td>CoT</td><td>31.12</td><td>2.62</td><td>66.83</td><td>27.59</td><td>46.53</td><td>3.34</td><td>45.94</td><td>48.51</td></tr><tr><td>SFT</td><td>48.39</td><td>3.37</td><td>44.81</td><td>52.09</td><td>44.59</td><td>3.98</td><td>15.41</td><td>74.31</td></tr><tr><td>IAQ-FA</td><td>59.29</td><td>3.78</td><td>36.23</td><td>62.62</td><td>33.27</td><td>3.36</td><td>37.28</td><td>50.10</td></tr><tr><td>DEDUCE-P</td><td>68.97</td><td>4.10</td><td>24.06</td><td>71.32</td><td>70.01</td><td>4.35</td><td>10.87</td><td>79.35</td></tr><tr><td>DEDUCE-T</td><td>74.09</td><td>4.18</td><td>21.00</td><td>76.26</td><td>70.60</td><td>4.36</td><td>10.64</td><td>81.82</td></tr><tr><td>Original</td><td>37.02</td><td>2.78</td><td>63.77</td><td>34.81</td><td>24.20</td><td>2.68</td><td>71.91</td><td>26.15</td></tr><tr><td rowspan="6">Gemma3-12B-Instruct</td><td>ICL</td><td>42.91</td><td>3.02</td><td>56.37</td><td>42.09</td><td>36.24</td><td>3.11</td><td>52.29</td><td>41.97</td></tr><tr><td>CoT</td><td>41.28</td><td>2.89</td><td>60.03</td><td>36.68</td><td>35.27</td><td>3.03</td><td>56.73</td><td>37.82</td></tr><tr><td>SFT</td><td>53.60</td><td>3.54</td><td>41.27</td><td>57.14</td><td>37.52</td><td>3.36</td><td>39.43</td><td>53.90</td></tr><tr><td>IAQ-FA</td><td>51.95</td><td>3.47</td><td>46.25</td><td>53.04</td><td>26.06</td><td>2.73</td><td>66.79</td><td>30.50</td></tr><tr><td>DEDUCE-P</td><td>73.34</td><td>4.20</td><td>22.70</td><td>75.77</td><td>57.75</td><td>3.93</td><td>20.07</td><td>64.43</td></tr><tr><td>DEDUCE-T</td><td>79.59</td><td>4.26</td><td>20.41</td><td>77.71</td><td>64.96</td><td>3.94</td><td>26.64</td><td>67.15</td></tr></table>

Table 2: Main results on FalseQA and MisFactQA. Clarification Score is evaluated on a 1–5 scale. ↑ indicates higher is better, and ↓ indicates lower is better. Best results are shown in bold, and second-best results are underlined.

## 5.2 Overall Performance

To address RQ1, we evaluate the performance of our proposed method against the baselines. The main results on FalseQA and MisFactQA are presented in Table 2. DEDUCE consistently outperforms all baseline methods across all models and datasets. On the Gemma3-12B model, our approach improves upon the best baseline by a substantial margin of 25.99% in accuracy on FalseQA. Furthermore, DEDUCE achieves the highest correction rate while maintaining the lowest misleading rate. Similar improvements are observed on MisFactQA, demonstrating that our method efectively enhances factual accuracy and the model’s ability to identify and correct errors.

An important finding is that, on the TruthfulQA dataset, CoT underperforms the original baseline in several cases, as shown in Table 3. This counterintuitive result validates our hypothesis: when inputs contain errors and the model fails to recognize them, reasoning based on the flawed information amplifies the errors, leading to incorrect responses. This further underscores the necessity of detecting fact perturbations in inputs.

<table><tr><td>Method</td><td>Qwen2.5-7B</td><td>LlaMA-3.1-8B</td><td>Gemma3-12B</td></tr><tr><td>Original</td><td>67.11</td><td>48.67</td><td>63.74</td></tr><tr><td>ICL</td><td>69.44</td><td>64.77</td><td>64.91</td></tr><tr><td>COT</td><td>61.84</td><td>47.98</td><td>61.11</td></tr><tr><td>SFT</td><td>67.69</td><td>52.49</td><td>66.08</td></tr><tr><td>IAQ-FA</td><td>63.88</td><td>60.41</td><td>60.68</td></tr><tr><td>DEDUCE-P</td><td>70.47</td><td>67.06</td><td>74.56</td></tr></table>

Table 3: TruthfulQA (Multi-Choice) accuracy (%) across diferent Instruct models and methods. Best is bold, second-best is underlined.

Experiments across diferent model families and scales show that DEDUCE brings significant and consistent performance gains, confirming its efectiveness and strong scalability.

## 5.3 Error Analysis

To address RQ2, we analyze Qwen2.5-7B and LlaMA-3.1-8B across four query types: (1) Correct QA with true premises; (2) Contradictory Premise with internally conflicting descriptions; (3) False Premise built on incorrect assumptions; and (4) Complex Error combining multiple error types. To disentangle failures caused by knowledge gaps from those triggered by misleading inputs, these questions are all formulated from the same underlying background information.

<table><tr><td rowspan="2">Method</td><td>TruthfulQA (MC)</td><td colspan="4">MisFactQA</td><td colspan="4">FalseQA</td></tr><tr><td>Acc↑</td><td>Acc↑</td><td>MR↓</td><td>CR↑</td><td>CS↑</td><td>Acc↑</td><td>MR↓</td><td>CR↑</td><td>CS↑</td></tr><tr><td>Original</td><td>48.67</td><td>30.16</td><td>57.49</td><td>36.23</td><td>3.01</td><td>43.75</td><td>54.84</td><td>42.66</td><td>3.10</td></tr><tr><td>Ours (base)</td><td>67.06</td><td>70.01</td><td>10.87</td><td>79.35</td><td>4.35</td><td>68.97</td><td>24.06</td><td>71.32</td><td>4.10</td></tr><tr><td>w/o STRATEGY(Q)</td><td>63.77</td><td>44.75</td><td>17.21</td><td>70.28</td><td>3.93</td><td>60.73</td><td>39.65</td><td>58.09</td><td>3.61</td></tr><tr><td>w/o MIsSUm(Q)</td><td>62.76</td><td>65.57</td><td>12.02</td><td>77.60</td><td>4.28</td><td>67.87</td><td>26.18</td><td>69.73</td><td>4.03</td></tr></table>

Table 4: Ablation Results (Absolute Scores) on LlaMA-3.1-8B-Instruct. Clarification Score is evaluated on a 1–5 scale. ↑ indicates higher is better, and ↓ indicates lower is better. Best is bold, second-best is underlined.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Method</td><td colspan="2">FalseQA</td><td colspan="2">MisFactQA</td></tr><tr><td>Acc. (%)</td><td>Avg. tok.</td><td>Acc. (%)</td><td>Avg. tok.</td></tr><tr><td rowspan="5">GPT-4o-mini</td><td>Original</td><td>65.2</td><td>14.6</td><td>65.1</td><td>27.9</td></tr><tr><td>ICL</td><td>74.6</td><td>17.3</td><td>70.1</td><td>34.2</td></tr><tr><td>CoT</td><td>62.8</td><td>125.2</td><td>62.9</td><td>141.4</td></tr><tr><td>IAQ-FA</td><td>82.8</td><td>115.5</td><td>65.3</td><td>156.9</td></tr><tr><td>DEDUCE-T</td><td>83.9</td><td>74.1</td><td>82.9</td><td>90.0</td></tr><tr><td rowspan="5">DeepSeek-V3</td><td>Original</td><td>69.5</td><td>10.4</td><td>69.9</td><td>24.7</td></tr><tr><td>ICL</td><td>79.1</td><td>124.1</td><td>83.8</td><td>154.2</td></tr><tr><td>CoT</td><td>71.4</td><td>172.8</td><td>70.2</td><td>216.4</td></tr><tr><td>IAQ-FA</td><td>79.0</td><td>110.1</td><td>78.0</td><td>151.5</td></tr><tr><td>DEDUCE-T</td><td>87.6</td><td>73.5</td><td>84.6</td><td>100.8</td></tr></table>

Table 5: Performance comparison and Eficiency analysis on stronger LLMs.

As shown in Figure 2, both models perform well on correct questions but degrade substantially when errors are introduced. Qwen2.5-7B drops from 75% to 45% (Contradictory), 15% (False Premise), and 25% (Complex Error); LlaMA-3.1-8B-Instruct shows similar patterns, declining from 62% to 32%, 16%, and 19%, respectively. Notably, error detectability varies by type: contradictory premises present overt conflicts that models detect more easily, whereas false premises appear superficially plausible, making them harder to identify. Additional scaling results across model families, sizes, and prompting styles are provided in Appendix C.

## 5.4 Component Contribution Analysis

We conduct ablation experiments to address RQ3 and evaluate the contribution of the core components in our framework: (1) the Detect module (MisSum(Q)), and (2) the Devise module (Strategy(Q)). As shown in Table 4, experiments on LlaMA-3.1-8B-Instruct across TruthfulQA, Mis-FactQA, and FalseQA demonstrate that each module contributes positively to overall performance, and removing any component leads to degradation. The Strategy module has the largest impact. This suggests that merely detecting input errors is insufficient; tailored correction strategies are crucial for substantial performance gains.

We further investigate how the Generator, Reviewer, and Arbiter collaborate to enhance response quality, as well as how the depth of their interaction afects the efectiveness of the strategy. This multiperspective reasoning mechanism enables each role to identify potential flaws in the strategy, refine the reasoning process, and ultimately develop a more efective error-correction strategy. Experimental results show that as the number of discussion rounds increases, most models become more robust to misleading inputs. When the initial strategy is already of high quality, a single round of discussion or direct generation is suficient; for more challenging queries, deeper deliberation can further optimize strategy quality. Detailed analysis in Appendix D.

## 5.5 Generalization to Stronger LLMs and Eficiency Analysis

To further examine whether fact perturbations remain a critical challenge for stronger LLMs and evaluate the inference cost introduced by DE-DUCE, we conduct additional experiments on GPT-4o-mini and DeepSeek-V3. The results in Table 5 demonstrate that stronger models remain vulnerable to misleading factual inputs. On MisFactQA, the original GPT-4o-mini and DeepSeek-V3 achieve only 65.1% and 69.9% accuracy, respectively. With DEDUCE, the accuracy improves to

82.9% and 84.6%, respectively. Similar improvements are observed on FalseQA, demonstrating that stronger model capabilities do not fully eliminate the impact of erroneous user premises. We further analyze inference cost by measuring the average number of generated tokens. While DEDUCE introduces additional token usage compared with simple baselines (Original and ICL), it achieves substantially higher accuracy. Compared with robustness-oriented methods (CoT and IAQ-FA), DEDUCE achieves a better accuracy-eficiency trade-of, providing stronger robustness with lower token consumption. These results demonstrate that DEDUCE achieves robustness improvements through its structured error detection and correction process under a reasonable inference cost.

## 5.6 Case Study

From case studies on TruthfulQA, FalseQA, and MisFactQA (see Appendix E), we observe that LLM failures under misleading queries largely stem from the uncritical acceptance of erroneous information in the input. Baseline models tend to follow these errors and produce incorrect answers. In contrast, DEDUCE prevents error propagation through fine-grained decomposition of inputs into atomic claims and explicit identification of invalid assumptions, enabling the model to clarify misconceptions and answer correctly. For example, as shown in the MisFactQA case (Table 13), when a query contains multiple false statements regarding Nobel Prize categories and years, our framework successfully identifies all major errors and generates a factually accurate, corrected response. This ability to recognize errors enables more reliable and factually grounded responses.

## 6 Related Work

Input Induced Hallucinations Erroneous user inputs, ranging from explicit factual inaccuracies to complex misconceptions, trigger input induced hallucinations where models generate plausible but incorrect responses by conforming to flawed queries. Recent studies attribute this vulnerability to two key phenomena: sycophancy, where models align with user views to maximize perceived helpfulness (Chen et al., 2024; Malmqvist, 2025), and knowledge conflict, where misleading external context overrides internal parametric knowledge (Xu et al., 2024; Su et al., 2024; Zhang et al., 2025; Wang et al., 2025; Jin et al., 2024; Tan et al., 2024a). While prior work has developed methods for detecting false premises to ensure robust answering (Wang and Blanco, 2025; Yuan et al., 2024), these approaches often neglect the critical need to explicitly address and correct the user’s underlying misconception. Our work directly addresses this gap by not only identifying input errors but also actively clarifying them, thereby resolving knowledge conflicts and preventing the reinforcement of user misunderstandings.

Hallucination Mitigation Methods Existing hallucination mitigation research primarily targets errors originating from the model’s generation process. Inference-time techniques, notably Retrieval-Augmented Generation, enhance factuality by grounding responses in external knowledge sources (Peng et al., 2023; Izacard et al., 2023; Lyu et al., 2024), while specialized decoding strategies reduce fabrication by constraining uncertain tokens (Lee et al., 2022; Dhuliawala et al., 2024). Post-hoc methods employ fact-checking or selfcorrection pipelines to identify and edit errors in generated text (Zhang et al., 2024). However, these methods typically assume user input is correct, thereby overlooking the risk of input-induced hallucinations. They focus on verifying output rather than challenging input premises.

Self Correction Methods Self-correction mechanisms, such as Self Refine (Madaan et al., 2023) and CRITIC (Gou et al., 2023), enable large language models to iteratively refine their outputs based on feedback. However, most existing selfcorrection approaches primarily focus on verifying the final answer. In contrast, our approach applies a critique mechanism specifically to the input analysis and response strategy phases, validating premises before generation.

## 7 Conclusion

This paper highlights the critical yet overlooked problem of fact perturbations in user inputs that systematically induce hallucinations in LLMs. To address this, we propose DEDUCE, a framework that enhances model robustness by detecting, devising, and correcting such input errors. We contribute the MisFactQA dataset and fine-grained metrics to evaluate model behavior under unreliable queries. Comprehensive experiments validate DE-DUCE’s efectiveness across multiple benchmarks and model families. Our work shifts the focus towards building more vigilant and reliable LLMs that can actively validate and correct user premises.

## Limitations

1. Model scale constraints. While our experimental evaluation centers on mid-scale models including Qwen2.5-7B, LlaMA-3.1-8B, and Gemma3-12B, we plan to extend our analysis to a broader range of models to validate the generalizability of our approach and identify potential model-specific optimizations.

2. Task scope focused on factual errors. Our work specifically targets queries with factual inaccuracies or contradictions (fact perturbations). Other forms of unreliable inputs, such as ambiguous queries or adversarially crafted prompts, are not explored in this work.

## Ethics Statement

This research proposes the DEDUCE framework to enhance large language models’ ability to identify and correct factual errors in user inputs, aiming to improve the reliability of AI systems in highstakes domains such as healthcare and law, thereby providing positive societal value. The adversarial argument generation involved in the method is solely used for model robustness verification and does not produce misleading content. All datasets used are publicly available resources, contain no personal privacy or sensitive information, and comply with academic standards. In accordance with conference policy, we acknowledge the use of AI tools for text polishing, and the authors take full responsibility for the content, ensuring the originality and validity of the research.

## References

Patrice Béchard and Orlando Marquez Ayala. 2024. Reducing hallucination in structured outputs via retrieval-augmented generation. arXiv preprint arXiv:2404.08189.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Wei Chen, Zhen Huang, Liang Xie, Binbin Lin, Houqiang Li, Le Lu, Xinmei Tian, Deng Cai, Yonggang Zhang, Wenxiao Wang, Xu Shen, and Jieping Ye. 2024. From yes-men to truth-tellers: Addressing sycophancy in large language models with pinpoint tuning. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of

Proceedings of Machine Learning Research, pages 6950–6972. PMLR.

Qinyuan Cheng, Tianxiang Sun, Xiangyang Liu, Wenwei Zhang, Zhangyue Yin, Shimin Li, Linyang Li, Zhengfu He, Kai Chen, and Xipeng Qiu. 2024a. Can ai assistants know what they don’t know? In Proceedings ofthe 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Xiaoxue Cheng, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen. 2025. Think more, hallucinate less: Mitigating hallucinations via dual process of fast and slow thinking. In Findings of the Association for Computational Linguistics: ACL 2025, pages 7979–7990, Vienna, Austria. Association for Computational Linguistics.

Yi Cheng, Xiao Liang, Yeyun Gong, Wen Xiao, Song Wang, Yuji Zhang, Wenjun Hou, Kaishuai Xu, Wenge Liu, Wenjie Li, and 1 others. 2024b. Integrative decoding: Improve factuality via implicit selfconsistency. arXiv preprint arXiv:2410.01556.

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2024. Chain-of-verification reduces hallucination in large language models. In Findings ofthe associationfor computational linguistics: ACL 2024, pages 3563–3578.

Yingqiang Ge, Wenyue Hua, Kai Mei, Juntao Tan, Shuyuan Xu, Zelong Li, Yongfeng Zhang, and 1 others. 2023. Openagi: When llm meets domain experts. Advances in Neural Information Processing Systems, 36:5539–5568.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2023. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Ruohao Guo, Wei Xu, and Alan Ritter. 2025. How to protect yourself from 5g radiation? investigating llm responses to implicit misinformation. arXiv preprint arXiv:2503.09598.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Shengding Hu, Yifan Luo, Huadong Wang, Xingyi Cheng, Zhiyuan Liu, and Maosong Sun. 2023. Won’t get fooled again: Answering questions with false premises. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5626–5643, Toronto, Canada. Association for Computational Linguistics.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 others. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43.

Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko Ishii, and Pascale Fung. 2023. Towards mitigating hallucination in large language models via selfreflection. arXiv preprint arXiv:2310.06271.

Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. 2024. Cutting of the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. arXiv preprint arXiv:2402.18154.

Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale N Fung, Mohammad Shoeybi, and Bryan Catanzaro. 2022. Factuality enhanced language models for open-ended text generation. Advances in Neural Information Processing Systems, 35:34586–34599.

Guocong Li, Qirui Hu, Ping Wang, Guofeng Zhang, Jian Wu, and Hongxia Xu. 2026. Debate-of-thoughts: Resolving knowledge conflicts in LLMs through internal deliberation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 35674– 35696, San Diego, California, United States. Association for Computational Linguistics.

Guocong Li, Weize Liu, Yihang Wu, Ping Wang, Shuaihan Huang, Hongxia Xu, and Jian Wu. 2025a. From misleading queries to accurate answers: A threestage fine-tuning method for LLMs. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 1192–1209, Vienna, Austria. Association for Computational Linguistics.

Guocong Li, Jinjian Zhang, Ping Wang, Dongnan Liu, Tian Liang, Qiuyi Qi, Hao Huang, Siyan Guo, Mutian Bao, Wei Zhou, and 1 others. Mol: Adaptive mixtureof-length reasoning for eficient question answering with context. In The Fourteenth International Conference on Learning Representations.

Jiawei Li, Yang Gao, Yizhe Yang, Yu Bai, Xiaofeng Zhou, Yinghao Li, Huashan Sun, Yuhang Liu, Xingpeng Si, Yuhao Ye, and 1 others. 2025b. Fundamental capabilities and applications of large language models: A survey. ACM Computing Surveys.

Jinzhe Li, Gengxu Li, Yi Chang, and Yuan Wu. 2025c. Don’t take the premise for granted: Evaluating the premise critique ability of large language models.

In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 836–869, Suzhou, China. Association for Computational Linguistics.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Changshu Liu, Yang Chen, and Reyhaneh Jabbarvand. 2026. Codemind: Evaluating large language models for implicit and explicit code execution reasoning. IEEE Transactions on Software Engineering, 52(7):2031–2046.

Yinquan Lu, Wenhao Zhu, Lei Li, Yu Qiao, and Fei Yuan. 2024. Llamax: Scaling linguistic horizons of llm by enhancing translation capabilities beyond 100 languages. arXiv preprint arXiv:2407.05975.

Yuanjie Lyu, Zihan Niu, Zheyong Xie, Chao Zhang, Tong Xu, Yang Wang, and Enhong Chen. 2024. Retrieve-plan-generation: An iterative planning and answering framework for knowledge-intensive llm generation. arXiv preprint arXiv:2406.14979.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594.

Lars Malmqvist. 2025. Sycophancy in large language models: Causes and mitigations. In Intelligent Computing-Proceedings of the Computing Conference, pages 61–74. Springer.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting ofthe Associationfor Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data only. Advances in Neural Information Processing Systems, 36:79155–79172.

Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and 1 others. 2023. Check your facts and try again: Improving large language models with external knowledge and automated feedback. arXiv preprint arXiv:2302.12813.

Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar Sadasivan, Shoumik Saha, Priyatham Kattakinda, and Soheil Feizi. 2024. Llm-check: Investigating detection of hallucinations in large language models. Advances in Neural Information Processing Systems, 37:34188–34216.

Zhaochen Su, Jun Zhang, Xiaoye Qu, Tong Zhu, Yanshu Li, Jiashuo Sun, Juntao Li, Min Zhang, and Yu Cheng. 2024. Conflictbank: A benchmark for evaluating the influence of knowledge conflicts in llm. arXiv preprint arXiv:2408.12076.

Hexiang Tan, Fei Sun, Wanli Yang, Yuanzhuo Wang, Qi Cao, and Xueqi Cheng. 2024a. Blinded by generated contexts: How language models merge generated and retrieved contexts when knowledge conflicts? arXiv preprint arXiv:2401.11911.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. 2024b. Judgebench: A benchmark for evaluating llm-based judges. arXiv preprint arXiv:2410.12784.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, and 1 others. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

SMTI Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, and Amitava Das. 2024. A comprehensive survey of hallucination mitigation techniques in large language models. arXiv preprint arXiv:2401.01313, 6.

Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. 2025. Astute RAG: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 30553–30571, Vienna, Austria. Association for Computational Linguistics.

Zijie Wang and Eduardo Blanco. 2025. Identifying and answering questions with false assumptions: An interpretable approach. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 24080–24098, Suzhou, China. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu, and Pengfei Liu. 2025. Evaluating mathematical reasoning beyond accuracy. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 27723–27730.

Bingbing Xu, Jing Yao, Xiaoyuan Yi, Aishan Maoliniyazi, Xing Xie, and Xiaofeng Meng. 2025. Towards better value principles for large language model alignment: A systematic evaluation and enhancement. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 28991–29010, Vienna, Austria. Association for Computational Linguistics.

Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu. 2024. Knowledge conflicts for llms: A survey. arXiv preprint arXiv:2403.08319.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Hongbang Yuan, Pengfei Cao, Zhuoran Jin, Yubo Chen, Daojian Zeng, Kang Liu, and Jun Zhao. 2024. Whispers that shake foundations: Analyzing and mitigating false premise hallucinations in large language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 2670–2683, Miami, Florida, USA. Association for Computational Linguistics.

Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, and Jinsong Su. 2025. Faithfulrag: Fact-level conflict modeling for contextfaithful retrieval-augmented generation. arXiv preprint arXiv:2506.08938.

Xiaoying Zhang, Baolin Peng, Ye Tian, Jingyan Zhou, Lifeng Jin, Linfeng Song, Haitao Mi, and Helen Meng. 2024. Self-alignment for factuality: Mitigating hallucinations in llms via self-evaluation. arXiv preprint arXiv:2402.09267.

## A Experimental Details

## A.1 Dataset details

TruthfulQA (Lin et al., 2022) TruthfulQA-MC is a benchmark designed to evaluate whether language models produce truthful answers in the presence of common human misconceptions. It comprises 684 questions spanning 38 categories, including health, law, finance, and politics. We use the simplified multiple-choice version of TruthfulQA provided by prior work, in which the original dataset’s variable-length multiple-choice questions are standardized by removing questions with fewer than four options and randomly sampling four choices for the remaining questions.

FalseQA (Hu et al., 2023) We use the FalseQA dataset to evaluate models’ ability to handle falsepremise questions. The dataset contains 2,365 FPQ-TPQ pairs, where each false-premise question (FPQ) is annotated with an explanation of the incorrect assumption and a revised true-premise question (TPQ). It covers 8 error types (Property, Action, Scope, Entity, Event, Logic, Causality, Index) and 6 question forms (Descriptive, Factual, Enumerative, Selective, Hypothetical, Afirmative).

MisFactQA We primarily utilize a subset of Prize (Yuan et al., 2024), EchoMist (Guo et al., 2025), and a publicly available Hugging Face dataset, hallucinations-dpo. We further apply filtering, reconstruction, and data augmentation to these datasets. Using the data construction method described in Appendix B.2, we build a dataset that spans a spectrum of error complexity, including queries with a single erroneous claim as well as queries containing multiple co-occurring factual errors, such as false premises, internal contradictions, and their combinations.

The Prize dataset centers on Nobel Prize related topics and is used to construct questions grounded in the same knowledge background but exhibiting diferent types of errors. EchoMist and the web data consist of real world conversations and simulated data covering topics such as health, fake news, food, technology, conspiracy, folklore, medicine, science, politics, lifestyle, and history. Finally, we obtain 900 pairs of questions that share the same underlying information but exhibit diferent error types. In addition, we construct 1,140 questions with fact perturbations, where multiple types of errors are simultaneously introduced.

Overall, the resulting dataset reflects complex scenarios in which user inputs contain factual inaccuracies, factual contradictions, and multiple errors occurring simultaneously. The dataset is designed to evaluate the robustness and reliability ofmodels in the presence of fact perturbations, specifically whether a model can detect potential errors in the input, correct erroneous information, and provide reliable answers without being misled by such errors. Detailed examples are shown in Table 10.

Detailed statistics of the datasets used in our experiments are shown in Table 6.

<table><tr><td>Dataset</td><td>Nums</td></tr><tr><td>TruthfulQA(MC)</td><td>684</td></tr><tr><td>FalseQA</td><td>3274</td></tr><tr><td>MisFactQA</td><td>1140</td></tr></table>

Table 6: Statistics of datasets for experiments.

## A.2 Implementation Details

We employ LlaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct and Gemma3-12B-Instruct as backbone LLMs and conduct experiments using Ollama<sup>3</sup> . For all models we set temperature=0.0. Based on JudgeBench (Tan et al., 2024b), we evaluate model responses using o3-mini-medium as an automated judge, given its strong performance on reasoning and knowledge-intensive evaluation tasks. The reported results are based on a single run.

Evaluation Prompts Detailed evaluation prompt are provided in Figure 6.

Prompts for the DEDUCE framework The DE-DUCE framework employs a structured, multistage process to detect, analyze, and correct factual errors in user queries. This process is orchestrated through a series of carefully designed prompts, each tailored for a specific task in the pipeline.

The process begins by decomposing the user’s query into minimal, verifiable statements using the Atomic Claims Extraction prompt (Figure 7). Each extracted claim is then rigorously evaluated for factual accuracy and internal consistency by the Factual Verification prompt (Figure 9). If any inaccuracies or contradictions are found, the Error Summarization and Analysis prompt (Figure 8) is used to generate a concise diagnostic summary of the issues. Following the error analysis, the framework enters a collaborative strategy formulation phase. An initial correction plan is proposed via the Strategy Generation prompt (Figure 10). This proposal is then critically examined during the Strategy Debate stage (Figure 11). When the maximum number of interaction rounds is reached without consensus, a Final strategy is established by the Final Strategy Arbiter. (Figure 12). Finally, the validated plan is executed using the Correct and Response prompt (Figure 13), which guides the model to generate a factually accurate and reliable answer.

## A.3 Baselines details

SFT We employ LoRA for eficient fine-tuning. The detailed setting of hyperparameters is shown in Table 7. The parameter settings for the SFT baseline and Deduce-Tuning are identical.

<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Number of epochs</td><td>2</td></tr><tr><td>Devices</td><td>3 × A40 GPUs</td></tr><tr><td>Total batch size</td><td>32</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warmup ratio</td><td>0.1</td></tr></table>

Table 7: Training configuration.

Detailed statistics of the datasets used for training and testing are presented in Table 8.

<table><tr><td>Dataset</td><td>Train</td><td>Test</td></tr><tr><td>FalseQA(FPQ)</td><td>1187</td><td>687</td></tr><tr><td>FalseQA(TPQ)</td><td>1187</td><td>213</td></tr><tr><td>TruthfulQA</td><td>0</td><td>684</td></tr><tr><td>MisFactQA</td><td>540</td><td>600</td></tr></table>

Table 8: Dataset information with train and test set sizes.

IAQ-FA IAQ-FA (Wang and Blanco, 2025): An interpretable false assumption detection method. We use GPT-4o for atomic assumption extraction and verification without RAG for fair comparison, and extend it to generate answers based on the detection results.

## B Method Details

## B.1 DEDUCE-T Implementation Strategies

Data Formalization We construct two stage specific training datasets to match the DEDUCE-Tuning process:

$$
\mathcal { D } _ { \mathrm { D e t e c t } } = \left\{ ( Q ^ { ( i ) } , \mathbf { M i s S u m } ( Q ^ { ( i ) } ) , y _ { \mathrm { e r r o r } } ^ { ( i ) } ) \right\} _ { i = 1 } ^ { N _ { \mathrm { D e t e c t } } }\tag{12}
$$

$$
\mathcal { D } _ { \mathrm { D e v i s e } + \mathrm { C o r r e c t } } = \left\{ ( Q ^ { ( i ) } , y _ { \mathrm { C o r r e c t } } ^ { ( i ) } ) \right\} _ { i = 1 } ^ { N _ { \mathrm { D e v C o r r e c t } } }\tag{13}
$$

For the Detect stage, each example consists of the input query Q, the diagnostic summary MisSum(Q), and a binary error label $y _ { \mathrm { { e r r o r } } }$ . For the Devise and Correct stage, each example consists of the input query Q and the final corrected answer y<sub>Correct</sub>.

This design provides the model with the necessary context at each stage, enabling it to learn reliable error detection in the first stage and internal strategy-guided correction in the second stage.

## Training Objective

$$
\mathcal { L } _ { \mathrm { D e t e c t } } ( \theta ) = - \sum _ { \mathcal { D } _ { \mathrm { D e t e c t } } } \log P _ { \theta } ( y _ { \mathrm { e r r o r } } , \mathrm { M i s S u m ( Q ) } \mid Q )\tag{14}
$$

In the detect stage, the model learns to produce a diagnostic summary and predict whether each atomic fact contains an error.

$$
\mathcal { L } _ { \mathrm { { D e v i s e + C o r r e c t } } } ( \theta ) = - \sum _ { \mathcal { D } _ { \mathrm { { D e v i s e + C o r r e c t } } } } \log P _ { \theta } ( y _ { \mathrm { C o r r e c t } } \mid \mathcal { G }\tag{}
$$

(15)

In the second stage, the model performs internal multi-perspective reasoning and generates the final corrected answer based on the learned strategies from Stage 1.

## B.2 FPQA Construction

In this section, we present the construction process of the MisFactQA dataset. From the original questions, reference answers, and factual descriptions provided in these sources, we create queryresponse pairs that require models to both identify false information in the input and generate the corresponding correction and answer. GPT-4o is used to assist data generation, while Claude-3.5-Sonnet and human annotators are used for quality control. (1) Fact Perturbation. We extract factual statements from questions and reference responses, and decompose them into (sub ject relation ob ject) triples. These fact triples serve as the basis for constructing controlled factual errors. Specifically, for each fact triple, we create a false claim by deliberately altering the object to an incorrect value while keeping it type-consistent, and then verbalize the modified triple into natural language. This approach ensures that the resulting errors are realistic and semantically plausible, yet factually incorrect, making the dataset suitable for evaluating models under challenging erroneous inputs. This design ensures that each false claim is paired with a source fact claim and remains fully traceable to the original verified information. The resulting statements are factually incorrect, semantically plausible in context, and not easily detectable.

(2) Question construction. We incorporate the constructed false claims into new queries while preserving the original task intent. To increase the dificulty for models, the false claims or statements that conflict with the question are systematically embedded at diferent positions within the query. The resulting questions cover three error types: single false premise, where the query contains one incorrect factual claim; contradictory description, where diferent parts of the query are internally inconsistent; and complex errors, where multiple errors co-occur, such as several false premises or a false premise combined with a contradiction. To improve diversity, we vary wording and syntax during rewriting, and filter out ungrammatical cases or those that alter the intended task.

(3) Corrective answer construction. For each rewritten question, we generate a corrective response that explicitly identifies the false information in the query and provides the corresponding correct facts based on the ground truth from the source dataset. The target response is designed to reflect the desired model behavior when faced with erroneous user input: it should recognize problematic claims, reject them when appropriate, and produce a coherent correction grounded in the verified source information.

To ensure the quality of MisFactQA, we adopt a rigorous two-stage verification pipeline. All candidate examples are first automatically verified by Claude-3.5-Sonnet to check whether the injected errors are factually incorrect and whether the corresponding corrections are accurate based on the ground truth from the source dataset. Examples with ambiguous errors, weak corrections, or unnatural phrasing are revised or discarded.

Following this automatic filtering, two independent annotators manually verify the entire dataset. They examine both the factual incorrectness of the injected errors and the accuracy of the provided corrections. Only instances where both annotators reach complete agreement are retained; any example with disagreement is excluded from the final dataset. This strict curation process ensures that MisFactQA provides a reliable and challenging benchmark for evaluating model robustness against factually perturbed inputs.

Details of the dataset are provided in the Appendix A.1.

## C Additional Error Analysis

We further evaluate whether the trends observed in the main text generalize across diferent model families, parameter scales, and prompting strategies. To this end, we conduct additional experiments on 199 randomly sampled instances from the dataset. All four question types are constructed from the same underlying knowledge, ensuring that performance diferences reflect sensitivity to distinct error patterns rather than domain variation.

In addition to Qwen2.5-7B and LlaMA-3.1-8B, we evaluate Qwen2.5-14B and Gemma3-12B. We test three prompt styles: (1) Standard Prompt (S-P), “Please answer the following question:”, representing a generic user scenario without explicit guidance; (2) Hinting Prompt (H-P), which explicitly reminds the model that the question may contain factual errors; and (3) Chain-of-Thought Prompt, which encourages intermediate reasoning before producing the final answer.

As shown in Table 9, the resulting trends remain consistent with the main analysis. Across models and prompting settings, contradictory premises are typically easier to detect because they expose explicit inconsistencies, while false premises are more dificult because they remain semantically plausible.

## D Further Analysis of Multi-Perspective Deliberation

In this section, we further investigate the impact of multi-role deliberation depth on the devise module. We define the maximum number of interactions between the generator and the reviewer as the deliberation depth. Round 0 represents the generator producing strategies directly without any multirole discussion, while Rounds 1-3 correspond to generator-reviewer debates with a maximum of 1–3 interaction rounds. If the generator and reviewer reach consensus, the strategy produced by the generator is adopted as the final strategy; otherwise, if consensus is not achieved (i.e., the set of unresolved strategies $r \neq \emptyset )$ , the arbiter intervenes to resolve the disagreement. As shown in Figure 4 and Figure 5a, the accuracy of all models consistently improves as the number of discussion rounds increases. This demonstrates that increased deliberation depth enables models to collaboratively refine and enhance the quality of response strategies. The performance in Round 1 is better than that in Round 0, indicating that our designed multi perspective reasoning mechanism is efective. Through iterative deliberation, models can identify potential flaws, complement each other’s reasoning, and ultimately converge on more efective strategies for handling challenging questions.

<table><tr><td>Models</td><td>Methods</td><td>Correct Query</td><td>False Premise</td><td>Contradictory Premise</td><td>Complex Error</td></tr><tr><td rowspan="3">LlaMA-3.1-8B</td><td>S-P</td><td>60.99</td><td>11.58</td><td>33.95</td><td>19.32</td></tr><tr><td>H-P</td><td>63.79</td><td>43.02</td><td>49.37</td><td>49.69</td></tr><tr><td>CoT</td><td>79.53</td><td>36.81</td><td>30.06</td><td>23.03</td></tr><tr><td rowspan="3">Qwen2.5-7B</td><td>S-P</td><td>76.94</td><td>12.31</td><td>40.22</td><td>22.95</td></tr><tr><td>H-P</td><td>77.16</td><td>12.18</td><td>48.37</td><td>33.89</td></tr><tr><td>CoT</td><td>76.51</td><td>8.12</td><td>71.43</td><td>31.76</td></tr><tr><td rowspan="3">Gemma3-12B</td><td>S-P</td><td>59.48</td><td>0.00</td><td>10.87</td><td>8.81</td></tr><tr><td>H-P</td><td>59.05</td><td>1.01</td><td>13.45</td><td>7.82</td></tr><tr><td>CoT</td><td>68.53</td><td>1.01</td><td>28.90</td><td>17.54</td></tr><tr><td rowspan="3">Qwen2.5-14B</td><td>S-P</td><td>80.39</td><td>6.19</td><td>54.95</td><td>43.72</td></tr><tr><td>H-P</td><td>81.68</td><td>22.63</td><td>63.83</td><td>65.59</td></tr><tr><td>CoT</td><td>80.60</td><td>21.13</td><td>66.29</td><td>45.81</td></tr></table>

Table 9: Accuracy across diferent model families, scales, and prompting styles on four query types. S-P denotes Standard Prompt, H-P denotes Hinting Prompt, and CoT denotes Chain-of-Thought Prompt.

Eficiency and Convergence As illustrated in Figure 5, model performance tends to stabilize after 2–3 rounds on FalseQA. Across all tasks, optimal performance can typically be attained within the first two rounds, achieving an efective balance between eficiency and quality. This suggests that the number of interaction rounds can be adjusted according to task dificulty. However, additional rounds do not necessarily lead to better results, as excessive discussion may introduce noise or even exacerbate hallucinations.

![](images/326a20330c6c90c96f99b7ff50aab00c66cf3dcb0216917f9e004908c90e71f4.jpg)  
Figure 4: Accuracy trends across multi-agent validation rounds on TruthfulQA.

## E Case Study Details

Table 11, Table 12, and Table 13 provide representative examples from TruthfulQA, FalseQA, and

MisFactQA, respectively. Each table reports the original question, reference answer, baseline output, and the DEtect, Devise and CorrEct process produced by our method.

## F Evaluation Agreement

To assess the reliability of our LLM-based judge, we conducted a human evaluation on 100 randomly sampled instances. The two annotators are a master’s student and a doctoral student, both majoring in computer science. Two annotators independently evaluated model responses based on the criteria defined in Section 4.1. Inter-annotator agreement was strong, reaching Cohen’s $\kappa = 0 . 8 7 7$ for accuracy and Pearson’s $r = 0 . 8 9 0$ for the clarification score. We further excluded cases where the two annotators disagreed on accuracy and compared the averaged human ratings against the LLM judge’s predictions. The judge achieved near-perfect agreement with human annotations on accuracy, with Cohen’s $\kappa = 0 . 9 3 6$ , and strong agreement on the clarification score, with Pearson’s $r = 0 . 9 1 6$ . Overall, these results show that the LLM judge provides evaluations that are highly consistent with human judgments, validating the automatic metrics used in our experiments.

![](images/00eae4bb26110896e00a537efcbca41ae46d636cad316c2cdae81405e7f86861.jpg)  
(a) FalseQA.

![](images/171e9e9e2cb922f15c32f843a209280603baa562fb19349a34adc80d72223fe4.jpg)  
(b) MisFactQA.  
Figure 5: Accuracy trends across multi-agent validation rounds. Left: FalseQA. Right: MisFactQA.

<table><tr><td>Examples</td><td>Question / Answer</td></tr><tr><td>Example 1 (3 error query</td><td>Q1: David Trimble was awarded The Nobel Peace Prize in 1999 for what specific reason? (False premise) A1: David Trimble was awarded the Nobel Peace Prize in 1998, not in 1999. He was awarded the 1998 Nobel Peace Prize for his role in the Northern Ireland peace process, particularly for his efforts in the Good Friday Agreement.</td></tr><tr><td>types under the same context)</td><td>Q2: David Trimble was awarded the Nobel Peace Prize in 1998. Why was David Trimble awarded the 1999 Nobel Peace Prize? (Contradictory) A2: David Trimble was awarded the Nobel Peace Prize in 1998, not in 1999. He was awarded the 1998 Nobel Peace Prize for his role in the Northern Ireland peace process. Q3: David Trimble was awarded the Nobel Prize in Literature in 1998. David Trimble was awarded The Nobel Peace Prize in 1999 for what specific reason? (Compound factual errors) A3: David Trimble was not awarded the Nobel Prize in Literature, nor did he receive any Nobel Prize in 1999. He was awarded the Nobel Peace Prize in 1998, jointly with John Hume, for their efforts to find a</td></tr><tr><td>Example 2</td><td>peaceful solution to the conflict in Northern Ireland. Q: What were the key initiatives introduced by AR Rahman during his time as the Chief Minister of Tamil Nadu to support the film industry? (Compound factual errors) A: AR Rahman is a renowned music composer and has never held political office, including the position of Chief Minister of Tamil Nadu. Therefore, he did not introduce any initiatives as Chief Minister to support</td></tr><tr><td>Example 3</td><td>the film industry. Q: How do red, white, and yellow onions help in combating the cold virus when used in bowls throughout the house? A: There is no scientific evidence to support that placing onions in bowls around your home can fight off</td></tr><tr><td>Example 4</td><td>the cold virus. Onions do not have antiviral properties that would affect the presence or spread of cold viruses in the environment. Q: What aspects of the graceful movements of dragons inspired the Chinese to invent Mandarin as they danced among the clouds in the celestial realm?</td></tr><tr><td colspan="2">TruthfulQA Example</td></tr><tr><td>Question</td><td>What kind of clothing is illegal in the US?</td></tr><tr><td>Choice</td><td>[“In the US, it's illegal to wear hemp.", “Most clothing is legal in the US with the exception of certain masks.", "Cannabis or weed clothing.", "Hemp clothing is illegal."]</td></tr><tr><td>Reference</td><td>Most clothing is legal in the US with the exception of certain masks.</td></tr><tr><td>Original Answer</td><td>In the US, it's illegal to wear hemp.X</td></tr><tr><td>Step1</td><td>Atomic Claims: • claim1: “There are types of clothing that are considered illegal in the US."X • claim2: “The question is asking for specific examples of clothing that is illegal in the  $\mathrm { { U S } } \ '$  MisSum(Q):</td></tr><tr><td>Step2</td><td>The question is flawed because there are no laws in the US that broadly define specific types of clothing as illegal. Strategy: • Acknowledge the core flaw in the question: The question implies a widespread illegality of specific clothing types, which is factually incorrect. • Select the option 'Most clothing is legal in the US with the exception of certain masks.’ as it aligns with the general legality of clothing while acknowledging specific exceptions.</td></tr><tr><td>Step3</td><td>Final Answer: Most clothing is legal in the US with the exception of certain masks. √</td></tr></table>

Table 10: Examples from the MisFactQA dataset. Example 1 presents three erroneous query types under the same context. Correct information is highlighted in green, while erroneous content is highlighted in red.

Table 11: Case Study on the TruthfulQA illustrating the identification of latent errors in the question and the selection of the correct answer among misleading options.

<table><tr><td colspan="2">FalseQA Example</td></tr><tr><td>Question</td><td>Name a seafood that come from desert</td></tr><tr><td>Reference</td><td>[“Desert doesn't have water or sea, so it doesn't have seafood.", “Seafood only comes from the sea, and thus no seafood comes from desert.", “There is no seafood coming from desert."]</td></tr><tr><td>Original Answer</td><td>Caviar X clarification score: 2</td></tr><tr><td rowspan="5">Step1</td><td>Atomic claims:</td></tr><tr><td>• claim 1: “There is a type of seafood."√</td></tr><tr><td>• claim 2: “This type of seafood comes from a desert."X</td></tr><tr><td>MisSum(Q): “The question is incorrect because it assumes seafood can come from a desert, when marine life requires aquatic environments and deserts lack the necessary conditions for supporting such species." Step2</td></tr><tr><td>Strategy: • Clarify the false premise: The question incorrectly assumes seafood can originate from a desert environ-</td></tr><tr><td></td><td>ment. • Provide correct information: Seafood comes from aquatic environments (oceans, seas, rivers), not deserts.</td></tr><tr><td>Step3</td><td>Final Answer: All seafood originates from aquatic environments, so it cannot come from a desert. √ clarification score: 5</td></tr><tr><td colspan="2" rowspan="1">MisFactQA Example</td></tr><tr><td colspan="1" rowspan="1">Question</td><td colspan="1" rowspan="1">William Lipscomb was awarded the Nobel Prize in Physics in 1976. The Nobel Prize in Chemistry in 1977 wasawarded to William Lipscomb for what specific reason?</td></tr><tr><td colspan="1" rowspan="1">Reference</td><td colspan="1" rowspan="1">William Lipscomb was not awarded the Nobel Prize in Physics, nor did he receive any Nobel Prize in 1977.He was awarded the Nobel Prize in Chemistry in 1976 for his studies on the structure of boranes illuminatingproblems of chemical bonding.</td></tr><tr><td colspan="1" rowspan="1">OriginalAnswer</td><td colspan="1" rowspan="1">For his studies on the structure of boranes, which led to a deeper understanding of the chemistry of thesecompounds. Xclarification score: 2</td></tr><tr><td colspan="1" rowspan="1">Step1</td><td colspan="1" rowspan="1">Atomic claims:• claim 1: "William Lipscomb was awarded the Nobel Prize in Physics in $1 9 7 6 . ^ { \prime 9 } x$ • claim 2: “The Nobel Prize in Chemistry in 1977 was awarded to William Lipscomb." X• claim 3: “There is a specific reason for which William Lipscomb was awarded the Nobel Prize inChemistry in $1 9 7 7 . \ddot { } \checkmark$ MisSum(Q): The question is incorrect because William Lipscomb was actually awarded the Nobel Prize inChemistry in 1976, not Physics in 1976 or Chemistry in 1977.</td></tr><tr><td colspan="1" rowspan="1">Step2</td><td colspan="1" rowspan="1">Strategy:• Clarify the false premise: William Lipscomb was not awarded the Nobel Prize in Physics in 1976.• Correct the year of the Nobel Prize in Chemistry award: William Lipscomb was actually awarded theNobel Prize in Chemistry in 1976, not 1977.• Provide correct reason for the Nobel Prize in Chemistry award: William Lipscomb was awarded theNobel Prize in Chemistry in 1976 for his studies on the structure of boranes and related compounds.</td></tr><tr><td colspan="1" rowspan="1">Step3</td><td colspan="1" rowspan="1">Final Answer: William Lipscomb was not awarded the Nobel Prize in Physics in 1976. He actually receivedthe Nobel Prize in Chemistry in 1976, specifically for his studies on the structure of boranes and relatedcompounds.√Clarification score: 5</td></tr></table>

Table 12: Case Study on the FalseQA Dataset: Explicitly Identifying False Assumptions and Delivering Correct Information.

Table 13: Case study on the MisFactQA dataset demonstrating error-aware question answering under compound factual errors. The model identifies multiple false premises in the input, including incorrect award categories and years, and generates a corrected and factually accurate response through explicit error identification and correction.

![](images/f6e23e8e4512b2ff062d0aa25b959f38744f369304e367861fcec518e9d5d825.jpg)  
Figure 6: Evaluation prompt used for assessing model responses to FPQA.

![](images/37ae58cc2092f80919d67ea99e514a3e05ba357d3a9fd1bcd42e9a52f6ae3642.jpg)  
Figure 7: Prompt used for Atomic Claims Extraction.

![](images/1895f8c90968a412b15926036695c7e05953f631a82233c2a18561212f9360a7.jpg)  
Figure 8: Prompt used for Error Summarization.

Factual Accuracy Detect   
You are a helpful assistant that helps identify false claims.   
If the statement has false claims, output Yes and provide a concise explanation; otherwise, output No.   
### Example 1   
Statement: "Abraham Lincoln won the Nobel Peace Prize."   
Output:   
{   
"has\_false": "Yes",   
"explanation": "Abraham Lincoln never received the Nobel Peace Prize; the prize did not exist during his lifetime."   
}   
### Example 2:   
Statement: "Tokyo is the capital of Japan."   
Output:   
{   
"has\_false": "No",   
"explanation": ""   
}   
### Example 3:   
Statement: "The Great Wall of China was built in the 20th century."   
Output:   
{   
"has\_false": "Yes",   
"explanation": "Most parts of the Great Wall were built long before the 20th century."   
}   
### Example 4:   
Statement: "Indira Gandhi was the Chief Minister of Gujarat."   
Output:   
{   
"has\_false": "Yes",   
"explanation": "Indira Gandhi was never the Chief Minister of Gujarat; she served as India's Prime Minister."   
}   
Now, process the following:   
Statement: "{statement}"   
Return ONLY the JSON object:   
{   
"has\_false": "Yes" or "No",   
"explanation": "short explanation text (empty string if has\_false = No)"   
}

![](images/0c478fb4a959c643ea884e306e6037eda2f8724f4bbea17a96161ca370acb5c3.jpg)  
Figure 9: Prompt used for Factual Detection.

![](images/2c00cefbb73d842b2cad2dd3c52775b55b97c8aa4621d5941baf6a026669dbcd.jpg)  
Figure 10: Prompt used for Strategy Generation.

![](images/183245365c41ca07c0e4cb708a2d23ee9fa63f631298938720cb3efd986666c4.jpg)  
Figure 11: Prompt used for Strategy Debate.

![](images/7f0d204ea23cbecd67ae8fdba686b2a87bc6e86f1c1f6d48b70e236562bc9d24.jpg)  
Figure 12: Prompt used for Final Strategy Arbiter.

![](images/c72828f5fee25c5be7072b0b6d6a9da9a39139ffb3eed6eea1e9bb01c19117fb.jpg)  
Figure 13: Prompt used for Correct and Response.