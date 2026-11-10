# ChemInsight: A Comprehensive Benchmark for Evaluating and Diagnosing Chemistry Understanding and Reasoning in Large Language Models

Changyu Hu<sup>1,2#</sup>, Yiming Zou<sup>1,2#</sup>, Xin Tang<sup>1,3#</sup>, Yiru Pan<sup>4</sup>, Zicheng Huang<sup>5</sup>, Deng Zhao<sup>1</sup>, Qing Cui<sup>1</sup>, Zhenduo Zhang<sup>1</sup>∗, Wang Ren<sup>1</sup>∗, Xiao Li<sup>2</sup>∗

<sup>1</sup>Ant Group <sup>2</sup>Peking University <sup>3</sup>University of Electronic Science and Technology of China <sup>4</sup>Westlake University <sup>5</sup>Peking University Health Science Center

## Abstract

We introduce ChemInsight, a benchmark designed to evaluate large language models (LLMs) on chemistry tasks that require multi-step reasoning and structured chemical understanding. We propose a two-dimensional scoring system—scope and depth—to assess both task coverage and reasoning completeness. ChemInsight consists of 203 challenging chemistry questions organized into 50 bundles, which support the two-dimensional scoring evaluation scheme.

We evaluate ten representative LLMs using ChemInsight and observe consistently low performance across models, with scope scores below 30% and depth scores below 11%, substantially lower than human expert performance. We then proceed to ask (1) What factor limits the advancement of current LLMs in chemistry? (2) To what extent are statistically driven language-model training methods effective at modeling chemistry symbolic language such as SMILES? And (3) Can LLM performance in chemistry be improved through intensive training on large, domain-specific corpora?

## 1 Introduction

Benchmarks for large language models (LLMs) are designed to measure a model’s capability within a specific domain. Recent advances in LLMs have led to remarkable success in tasks such as code generation [Dong et al., 2025] and general question answering [Yue, 2025], substantially enhancing productivity across many industries [Shaw et al., 2023]. However, comparisons across standard tests and benchmark scores reveal that current LLMs are considerably weaker in chemistry than in domains such as history or programming, and they continue to underperform relative to human experts on chemistry tasks OpenAI [2025], Zhao et al. [2025]. This gap limits the reliability of LLMs for integration into rigorous chemistry research workflows and authentic educational settings.

Existing chemistry benchmarks provide only limited insight into model performance, typically reporting a single numerical score Rein et al. [2024], Mirza et al. [2025], Zhao et al. [2025]. Such a score obscures the underlying bottlenecks that constrain LLM performance in chemistry and offers little guidance for targeted model improvement. Before LLMs can be responsibly adopted or systematically enhanced for chemistry applications, a more fine-grained and diagnostic evaluation of their capabilities is required.

![](images/4f35a2eb19ef6c8b36acd68d097d8961bb1dfea629f27a77a750d3f85d830ebb.jpg)  
Figure 1. Overview of the ChemInsight evaluation framework. The Benchmark Set (left) contains 203 questions in 50 bundles covering six chemistry ability dimensions; the lower panels summarize answer formats and difficulty levels. The scope score measures overall performance, while the depth score rewards completion of interconnected questions within essence bundles. The Diagnosis Set (right) contains 50 single-step questions on IUPAC naming, SMILES generation, and property prediction to probe foundational chemistry skills.

“Measurement is the first step that leads to control and eventually to improvement.”

— H. James Harrington

In this work, we propose ChemInsight, a benchmark designed to support two-dimensional scoring and diagnostic analysis. ChemInsight consists of two components: a benchmark set and a diagnosis set. The benchmark set contains 203 challenging chemistry questions organized into 50 question bundles. Within each bundle, questions are interrelated and progressively increase in difficulty, resembling problems encountered in authentic chemistry research or advanced coursework. Successfully completing an entire bundle is treated as evidence of deep understanding and high task competency, and is therefore rewarded with an additional depth score. Model performance on the benchmark set is quantified using two dimensions: a scope score, defined as the total proportion of correctly answered questions, and a depth score, which captures how deeply a model can progress through a complex, multi-step chemistry task.

The diagnosis set comprises 50 single-step questions targeting three fundamental chemistry skills: nomenclature, SMILES generation, and property prediction. Each question isolates a single elementary skill. Unlike traditional benchmarks, this diagnostic component is explicitly designed to facilitate interpretation of model performance on the more complex, multi-skill questions in the benchmark set, enabling finer-grained analysis of model strengths and weaknesses.

We evaluate ten state-of-the-art LLMs using ChemInsight. The resulting scope scores range from 11.6% to 27.8%, while depth scores range from 2.3% to 10.7% (Figure 2). Gemini 3 Pro achieves the highest performance, with a scope score of 27.8% and a depth score of 10.7%. Overall, the consistently low scores in both dimensions indicate that current LLM performance in chemistry remains unsatisfactory, particularly given that eight of the evaluated models represent the latest reasoning models. We further analyze model behavior on the diagnosis set and conduct a detailed examination of model outputs. In addition to reporting scores, the goal of this work is to provide an insightful diagnostic assessment of LLM capabilities in the chemistry domain. By identifying the key factors that limit LLM performance, we aim to illuminate pathways for future improvements in LLMs applied to scientific domains.

## 2 Methodology

## 2.1 ChemInsight Benchmark Question Construction

To evaluate the chemistry-specific capabilities of large language models (LLMs) and to provide diagnostic insights into their strengths and limitations, we construct two complementary question sets: a Benchmark Set and an Diagnosis Set.

Benchmark Set. The Benchmark Set consists of 203 original, challenging, and search-engine–resistant questions designed to require multi-step reasoning. These questions are organized into 50 bundles, with a primary focus on organic chemistry. Each question may draw upon multiple chemical principles and demand complex reasoning and/or quantitative calculations, thereby reflecting the types of problem-solving tasks encountered in advanced chemistry coursework and authentic research settings.

The design of the Benchmark Set follows two key standards: (1) the complexity and difficulty are intended to be challenging even for domain experts, which in this work refers to chemistry majors at topranked universities; and (2) the answers cannot be directly retrieved from general or professional chemistry databases or search engines, such as SciFinder or Reaxys.

The questions mainly test the following six major categories of chemical ability: A. reaction-type recognition, B. structure generation in context, C. calculation and counting, D. structure-property reasoning, E. selectivity prediction, and F. spatial imagination and three-dimensional thinking.

Diagnosis Set. The Diagnosis Set comprises 50 single-step questions that are intentionally simpler. These questions are designed to isolate and assess individual elementary chemistry skills, thereby facilitating the interpretation of model performance on the more complex, multi-skill questions in the Benchmark Set. Specifically, 30 questions explicitly test structure recognition and 20 questions assess structure-property reasoning.

All questions in both sets require LLMs to generate free-form answers; no multiple-choice questions are included. Both the Benchmark Set and the Diagnosis Set were independently created and cross-validated by domain experts to ensure correctness, clarity, and appropriate difficulty.

## 2.2 Evaluation Method

Baseline Models. In this benchmark, we evaluate 10 representative Large Language Models (LLMs) that cover current industry standards. Eight are evaluated in thinking mode: GPT-5 [OpenAI, 2025], Gemini 3 Pro [Google DeepMind, 2026], Gemini 2.5 Pro [Comanici et al., 2025], Ring-1T [Ling Team et al., 2025b], Qwen3-235B-A22B-Thinking-2507 [Qwen Team, 2025], DeepSeek-V3.2 [DeepSeek-AI et al., 2025b], DeepSeek-V3.2-Speciale [DeepSeek-AI et al., 2025b], and DeepSeek-R1 [DeepSeek-AI et al., 2025a]. Additionally, we evaluate Ling-1T [Ling Team et al., 2025a] and DeepSeek-V3.1 [DeepSeek-AI, 2025] in non-thinking mode.

Response Generation Protocol. To ensure standardized assessment within complex problem scenarios, we designed a Task-Specific Structured Prompt to guide the baseline models (the full prompt is provided in Appendix B). To mitigate generation randomness and assess model stability, we implement an 8-shot sampling strategy, independently sampling 8 responses for each problem from every model.

Hybrid Evaluation Pipeline. Upon collecting the generated responses, we employ a hybrid evaluation mechanism tailored to different question types:

1) Rule-Based Validation. For questions requiring molecular structure outputs (SMILES representations), we utilize the RDKit library. Due to the non-uniqueness of SMILES syntax, all generated SMILES strings are first canonicalized. We then verify exact matches against the canonicalized reference answers to determine correctness.

2) LLM-as-a-Judge. To achieve high-quality, fine-grained evaluation for short-answer questions involving multi-step reasoning, we adopt the LLM-as-a-Judge paradigm. We utilize a strong large language model as an evaluator, guided by an Evaluation-Specific Structured Prompt (the full prompt is provided in Appendix B). To standardize scoring and reduce bias, the judge model follows a hierarchical evaluation process:

• Special Rule Enforcement & Keyword Matching: This phase serves as the primary filter. The judge first inspects the Note field to enforce hard constraints, where a violation of special instructions (e.g., incorrect number of answers) triggers an immediate zero score regardless of content quality. Subsequently, it performs keyword matching, assessing the response based on the semantic coverage of essential terms defined in the reference.

• Three-Level Scoring Rubric: In general scenarios, the judge applies a rigorous tiered rubric: (1) Full Credit is awarded for responses that are conceptually complete, scientifically accurate, and logically consistent with the reference; (2) Partial Credit reflects constrained partial understanding. Here, we enforce a strict Anti-Hallucination Constraint: partial scores are valid only if the response remains strictly relevant without introducing extraneous concepts not mentioned in the question or reference; (3) Zero Credit is assigned for factual errors, violations of chemical principles, format non-compliance, or the inclusion of irrelevant information.

Score Aggregation. The final score S for a model on a specific sub-question is calculated as the average of the scores obtained from the 8 sampling attempts:

$$
S = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } E ( a _ { k } , a _ { \mathrm { r e f } } )\tag{1}
$$

where $K = 8 , a _ { k }$ is the k-th generated answer, $a _ { \mathrm { r e f } }$ is the reference answer, and E denotes the evaluation function (either Rule-based or LLM-based depending on the question type).

## 2.3 Scoring System

We propose a two-dimensional evaluation framework consisting of a Scope Score and a Depth Score, designed to jointly characterize the breadth and depth of chemical understanding demonstrated by large language models (LLMs).

The Scope Score $( S _ { \mathrm { s c o p e } } )$ is defined as the raw accuracy percentage achieved by an LLM on the Benchmark Set. The benchmark contains a total of 203 questions, worth 262 points in total. The Scope Score is normalized to a maximum of 100. The more questions answered correctly, the higher the score that a model receives.

<table><tr><td></td><td>Inference</td><td>Scope</td><td>Depth</td><td colspan="6"></td></tr><tr><td>Model</td><td>Mode</td><td>Score</td><td>Score</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td></tr><tr><td>Gemini 3 Pro</td><td>Thinking</td><td>0.278</td><td>0.107</td><td>0.217</td><td>0.202</td><td>0.548</td><td>0.396</td><td>0.291</td><td>0.191</td></tr><tr><td>GPT-5</td><td>Thinking</td><td>0.215</td><td>0.069</td><td>0.076</td><td>0.078</td><td>0.275</td><td>0.450</td><td>0.360</td><td>0.207</td></tr><tr><td>Gemini 2.5 Pro</td><td>Thinking</td><td>0.212</td><td>0.059</td><td>0.100</td><td>0.106</td><td>0.371</td><td>0.421</td><td>0.304</td><td>0.181</td></tr><tr><td>DeepSeek-V3.2-Speciale</td><td>Thinking</td><td>0.208</td><td>0.067</td><td>0.120</td><td>0.120</td><td>0.370</td><td>0.354</td><td>0.279</td><td>0.163</td></tr><tr><td>DeepSeek-V3.2</td><td>Thinking</td><td>0.181</td><td>0.047</td><td>0.089</td><td>0.091</td><td>0.359</td><td>0.351</td><td>0.244</td><td>0.156</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507</td><td>Thinking</td><td>0.151</td><td>0.034</td><td>0.058</td><td>0.063</td><td>0.117</td><td>0.325</td><td>0.262</td><td>0.127</td></tr><tr><td>Ring-1T</td><td>Thinking</td><td>0.151</td><td>0.028</td><td>0.052</td><td>0.051</td><td>0.257</td><td>0.318</td><td>0.275</td><td>0.137</td></tr><tr><td>DeepSeek-R1</td><td>Thinking</td><td>0.144</td><td>0.021</td><td>0.056</td><td>0.061</td><td>0.225</td><td>0.292</td><td>0.225</td><td>0.130</td></tr><tr><td>Ling-1T</td><td>Non-thinking</td><td>0.122</td><td>0.021</td><td>0.009</td><td>0.003</td><td>0.295</td><td>0.320</td><td>0.124</td><td>0.155</td></tr><tr><td>DeepSeek-V3.1</td><td>Non-thinking</td><td>0.116</td><td>0.024</td><td>0.015</td><td>0.007</td><td>0.238</td><td>0.300</td><td>0.130</td><td>0.160</td></tr></table>

Table 1: Performance comparison of different models. Inference Mode indicates the mode used in this eval uation. Scope and Depth scores are normalized by 262 and 524 respectively. Columns A–F are normalized by 137, 172, 17, 78, 46, and 48.5 respectively.

The Depth Score $( S _ { \mathrm { d e p t h } } )$ captures the extent to which a model demonstrates coherent, multi-step understanding within problem groups. Unlike the scope score, the questions do not contribute equally to $S _ { \mathrm { d e p t h } }$ Instead, models earn a higher depth score by correctly answering multiple questions within the same essence bundle. Among the 50 question bundles in the benchmark, 23 are designated as essence bundles, in which questions are intentionally interconnected and arranged in increasing order of difficulty, thereby mimicking the progressive reasoning challenges encountered by human chemists in real-world scientific scenarios.

The depth score for an individual essence bundle $Q _ { i }$ is defined as:

$$
S _ { \mathrm { d e p t h } } ( Q _ { i } ) = 2 S _ { \mathrm { s c o p e } } ( Q _ { i } ) \sin ^ { 3 } \left( \frac { \pi \omega _ { i } } { 2 } \right) ,\tag{2}
$$

where $S _ { \mathrm { s c o p e } } ( Q _ { i } )$ denotes the Breadth Score contribution of the bundle $Q _ { i }$ , and $\omega _ { i }$ represents the completion rate of that bundle. Full completion of an essence bundle $( \omega _ { i } = 1 )$ indicates high competence in the corresponding chemical domain and yields the maximum depth score, which is twice the scope score contribution of that bundle.

The total Depth Score is computed as:

$$
S _ { \mathrm { d e p t h } } = \sum _ { i } S _ { \mathrm { d e p t h } } ( Q _ { i } ) .\tag{3}
$$

Because the Depth Score is designed to reward concentrated, multi-step mastery beyond isolated correctness, the combined score $S _ { \mathrm { s c o p e } } + S _ { \mathrm { d e p t h } }$ is not capped at 100.

## 3 Experiments

We investigate the following research questions (RQs):

RQ1: How do representative LLMs perform on ChemInsight in terms of both scope and depth?

RQ2: What limitations currently hinder the advancement of LLMs in the chemistry domain?

RQ3: To what extent are statistical language models effective at modeling chemistry symbolic language such as SMILES?

## 3.1 Model Performance on ChemInsight (RQ1)

The performance of the evaluated LLMs on the ChemInsight benchmark, averaged over eight independent runs, is summarized in Table 1 and visualized in Figure 2. Overall, the results indicate that all evaluated models achieve relatively low scores on this newly proposed benchmark. Scope scores range from 11.6% to 27.8%, with none exceeding 50%. Depth scores are consistently low as well, ranging from 2.3% to 10.7% in the ten models.

![](images/a27db345241886bb16615b6495b8d096c1ea1921460eaf5a427cc7e6716449eb.jpg)  
Figure 2. Scope and depth scores on the ChemInsight Benchmark Set. Scores are averaged over eight responses per question and normalized by 262 and 524 points, respectively.

![](images/cb2e332e00425787f120a344348eb383b31d9639260ccb65a587bfd3462c0acb.jpg)  
Figure 3. Benchmark performance by expert-assigned question difficulty. Scores generally decrease from easy to medium and hard questions.

Among all evaluated models, Gemini 3 Pro attains the highest performance in both scope and depth, with scores of 27.8% and 10.7%, respectively. Nevertheless, this performance remains substantially below that of human experts, who achieve a 36.3% scope score. Reasoning-oriented models generally outperform non-reasoning models, as evidenced by Ling-1T and DeepSeek-V3.1, which rank near the bottom in both scope and depth metrics. It is also observed from Figure 3 that scores decrease with increasing humanexpert-defined difficulty level.

The depth score introduces an additional dimension to evaluate and differentiate model performance. For example, GPT-5 and Gemini 2.5 Pro obtain nearly identical scope scores (21.5% and 21.2%, respectively). Based solely on scope, one might conclude that the two models exhibit comparable capabilities. However, the depth score reveals a meaningful distinction: GPT-5 achieves a higher depth score (6.9%) than Gemin 2.5 Pro (5.9%), suggesting that GPT-5 is able to engage more deeply with complex chemistry problems.

In this evaluation work, depth scores appear to be roughly proportional to scope scores and exhibit only limited variation. This behavior is primarily attributable to the overall poor performance of existing LLMs on ChemInsight. As artificial intelligence technology continues to advance, scope scores are expected to increase and eventually approach saturation. At that stage, the depth score will become a more informative and discriminative metric for distinguishing among high-performing models.

## 3.2 Shortcomings Limiting LLM Advancement in Chemistry (RQ2)

Despite their strong performance in domains such as programming and natural language processing, large language models (LLMs) exhibit markedly weaker capabilities in chemistry. This raises a fundamental question: what factors limit the effectiveness of LLMs in the chemical domain?

![](images/08073fc347c53491aa8a777e513d77e3922bc97c7b753ec3c1f7ff71a7e863af.jpg)  
Figure 4. Contributions of text, chemical-formula, and SMILES answers to the total scope score. Segment heights are percentage points of the normalized benchmark score.

![](images/07c2cb2d958999072267907a4b62a941e0d23cf4743b5a4ff5bd60f12c632956.jpg)  
Figure 5. Accuracy on the Diagnosis Set for IUPAC naming, SMILES generation, and property prediction. Performance is strongest on property prediction and weakest on SMILES generation.

Our evaluation results indicate that in-context comprehension and generation of SMILES representations constitute the primary bottleneck. The required answer formats in the ChemInsight benchmark fall into three categories: natural language text, chemical formulas, and structural representations expressed as SMILES. Performance across these categories is summarized in Figure 4. While most models achieve comparable scores on text-based and chemical-formula questions, substantial performance differences emerge on SMILES-format questions, which clearly distinguish among models.

Variation in SMILES performance accounts for approximately 82% of the variance in the overall benchmark score, highlighting its dominant influence on total model performance.

Although the benchmark questions are designed to be comprehensive, each question is annotated by its primary objective and grouped into six core chemical ability dimensions: (A) reaction-type recognition, (B) structure generation in context, (C) calculation and counting, (D) structure-property reasoning, (E) selectivity prediction, and (F) spatial imagination and three-dimensional thinking. The radar plot shown in Figure 6 reveals pronounced weaknesses across nearly all dimensions, with the notable exception of calculation and counting. Importantly, this is the only ability that does not rely heavily on chemical structure comprehension.

In the ChemInsight benchmark, chemical structures are predominantly represented using SMILES notation. The observed pattern of results therefore provides strong evidence that deficient SMILES comprehension and manipulation is a central limitation of current LLMs, and a key factor constraining their progress in chemistry-focused reasoning tasks.

## 3.3 Chemistry SMILES Representation Competency Diagnosis (RQ3)

Chemistry-domain corpora typically contain a mixture of natural language text and symbolic chemical representations.

In the Benchmark Set, questions involving SMILES are often coupled with additional requirements such as contextual understanding or multi-step reasoning. As a result, poor model performance on these questions cannot be confidently attributed to a single underlying competency. To address this limitation, we construct a Diagnosis Set designed to evaluate isolated chemistry skills in a controlled manner.

![](images/df5e6ba207fa23d981cd11fb99cd8bea4c7bcab446a0d81a8de25730e463dd30.jpg)  
Figure 6. Normalized performance across the six chemistry ability dimensions in ChemInsight: (A) reaction-type recognition, (B) structure generation in context, (C) calculation and counting, (D) structure-property reasoning, (E) selectivity prediction, and (F) spatial imagination and three-dimensional thinking. Each curve represents one model; higher values indicate stronger performance.

Importantly, the answers to these single-skill questions can be directly retrieved from authoritative chemistry databases. Human experts, state-of-the-art search engines, and professional software tools (e.g., Chem-Draw) can achieve near-perfect accuracy on these tasks. While such questions do not resemble real-world chemistry problem-solving scenarios, they are intentionally designed to diagnose specific foundational competencies of LLMs in chemical representation and reasoning.

The evaluation results (Figure 5) indicate that most assessed LLMs demonstrate strong competency in property prediction. Reasoning-oriented models achieve approximately 90% accuracy, while non-reasoning models reach around 76%.

These results suggest that once a chemical compound is recognized, LLMs are generally capable of associating it with relevant chemical properties—an important criterion for structural understanding.

In chemistry, a compound may be represented in multiple forms, including its common name, IU-PAC systematic name, structural diagram, and SMILES string. For example, the compound quinine can be referred to by its common name or by its IUPAC name, (R)-(6-methoxyquinolin-4-yl)((2S,4S,8R)-8- vinylquinuclidin-2-yl)methanol. It can also be represented as a structural formula illustrating atomic connectivity and spatial arrangement, or as a SMILES string used by professional software to render molecular structures.

The results show that model accuracy ranges from 44.5% to 78.8% when translating common names to IUPAC names, and drops further to 29.4%–56.6% when translating common names to SMILES. These findings indicate that competency in chemical symbolic language—particularly SMILES representation— remains a major technical bottleneck. This deficiency substantially limits the ability of LLMs to perform complex and multi-step chemistry tasks that require precise manipulation of molecular structures.

## 4 Related Work

Benchmarks on chemistry. Currently, there exists a significant body of work regarding chemistry benchmarks. GPQA [Rein et al., 2024] encompasses the three major fields of biology, physics, and chemistry, comprising 448 high-quality multiple-choice questions written and verified by PhD-level experts in these fields. It challenges the knowledge boundaries of large models in cutting-edge scientific domains. Chem-Bench [Guo et al., 2023] establishes a comprehensive benchmark suite containing eight chemistry tasks, categorizing the capabilities of LLMs in chemistry into three core domains: understanding, reasoning, and explanation. This provides a framework for evaluating the “logical depth” of LLMs in the scientific domain. GPT-4 achieved 71% and Claude-3.5 achieved 61% on ChemBench as reported. SUPERChem [Zhao et al., 2025] provides expert-authored standardized solution paths for each question, employing the Reasoning Path Fidelity (RPF) metric to evaluate whether the model’s logical process in solving the problem is correct. By combining RPF scores with accuracy, it enables an accurate assessment of the comprehensive reasoning capabilities of the models.

LLM in chemistry. With the continuous advancement of AI technology, the application of AI in chemistry has been extensively explored and developed. LLMs play a dual role in chemical applications [Ramos et al., 2025]: acting as chemical tools and functioning as drivers for autonomous agents. ChemLLM [Zhang et al., 2024] is a conversational large language model specifically designed for the field of chemistry, aiming to address the performance deficiencies of general-purpose large models when processing specialized chemical knowledge. Through the construction of chemistry-specific datasets and a comprehensive training pipeline, the research team achieved the integration of knowledge across all chemical disciplines within the large language model framework, endowing the model with the “chemical intuition” required to handle complex interdisciplinary problems.

## 5 Conclusion

In this work, we introduced ChemInsight, a rigorous benchmark designed to evaluate the depth and scope of Large Language Models (LLMs) in the domain of chemistry. Through extensive experiments on representative models, including Gemini 3 Pro and GPT-5, we quantified the significant gap that remains between current state-of-the-art LLMs and human experts. Our findings reveal that while models demonstrate reasonable proficiency in general chemical textual knowledge, they struggle significantly with complex multi-step reasoning and deep conceptual understanding, as evidenced by the consistently low Depth Scores.

Crucially, our analysis identifies the comprehension and generation of symbolic representations— specifically SMILES—as the primary bottleneck restricting model performance. The symbolic nature of chemical language demands a precision that current general-purpose LLMs lack. While the systematic training or fine-tuning of models on SMILES grammars and large-scale chemical corpora falls outside the scope of this benchmarking study, it represents a critical direction for future research. Addressing this limitation may require novel tokenization strategies or neuro-symbolic approaches to bridge the gap between natural language processing and chemical structure manipulation.

Furthermore, we acknowledge that chemistry is an inherently multi-modal discipline, relying heavily on molecular graphs, spectra, and reaction diagrams to convey information. Our current evaluation is predominantly based on language models processing text and serialized strings. However, a comprehensive understanding of chemistry necessitates the ability to perceive and interpret visual data. Future work should extend the ChemInsight framework to include Multimodal Large Language Models (MLLMs), investigating whether visual encoders can enhance the reasoning capabilities required for tasks involving spatial imagination and structural analysis.

## Limitations

Despite the comprehensive nature of ChemInsight, several limitations should be noted. First, as highlighted in the conclusion, our benchmark currently focuses on the textual and symbolic modalities (SMILES/InChI). This excludes a vast array of chemical problems that require visual interpretation, such as reading NMR spectra or analyzing steric hindrance from 3D molecular representations. Consequently, the performance reported here may not fully reflect a model’s potential utility in a laboratory setting where visual data is ubiquitous.

Second, although we employed a rigorous prompt structure and expert-verified answers, the evaluation of open-ended generation tasks by LLM-based judges inevitably carries a degree of uncertainty. While we mitigated this through structured rubrics and multiple evaluation runs, subtle correct reasoning paths that deviate from standard reference answers might still be under-penalized or over-penalized.

Finally, the field of LLMs is evolving rapidly. The “Scope” and “Depth” scores reported in this paper represent a snapshot of model capabilities at the time of writing. As models with stronger reasoning capabilities (e.g., “System 2” thinking modes) continue to emerge, the difficulty ceiling of the benchmark may need to be dynamically adjusted to maintain its discriminative power.

## Acknowledgments

This work was carried out by Changyu Hu, Yiming Zou and Xin Tang during their internship at Ant Group.

## References

Gheorghe Comanici et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.

DeepSeek-AI. DeepSeek-V3.1. Model card, 2025. URL https://huggingface.co/deepseek-ai/ DeepSeek-V3.1.

DeepSeek-AI et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, 2025a. URL https://arxiv.org/abs/2501.12948.

DeepSeek-AI et al. DeepSeek-V3.2: Pushing the frontier of open large language models, 2025b. URL https://arxiv.org/abs/2512.02556.

Yihong Dong et al. A survey on code generation with llm-based agents. arXiv preprint arXiv:2508.00083, 2025.

Google DeepMind. Gemini 3 Pro model card. Model card, 2026. URL https://deepmind.google/ models/model-cards/gemini-3-pro/. Model released November 2025; model card updated May 2026.

Taicheng Guo et al. What can large language models do in chemistry? a comprehensive benchmark on eight tasks. Advances in Neural Information Processing Systems, 36:59662–59688, 2023.

Ling Team et al. Every activation boosted: Scaling general reasoner to 1 trillion open language foundation, 2025a. URL https://arxiv.org/abs/2510.22115.

Ling Team et al. Every step evolves: Scaling reinforcement learning for trillion-scale thinking model, 2025b. URL https://arxiv.org/abs/2510.18855.

Adrian Mirza et al. A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists. Nature Chemistry, pages 1–8, 2025.

OpenAI. GPT-5 system card. System card, 2025. URL https://openai.com/index/ gpt-5-system-card/.

Qwen Team. Qwen3-235B-A22B-Thinking-2507. Model card, 2025. URL https://huggingface.co/ Qwen/Qwen3-235B-A22B-Thinking-2507.

Mayk Caldas Ramos, Christopher J Collison, and Andrew D White. A review of large language models and autonomous agents in chemistry. Chemical science, 2025.

David Rein et al. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

C. Shaw et al. Genai in higher education: Fall 2023 update time for class study. arXiv preprint arXiv:2512.01274, 2023.

Murong Yue. A survey of large language model agents for question answering. arXiv preprint arXiv:2503.19213, 2025.

Di Zhang et al. Chemllm: A chemical large language model. arXiv preprint arXiv:2402.06852, 2024.

Zehua Zhao et al. Superchem: A multimodal reasoning benchmark in chemistry. Tyton Partners: Boston, 2025.

## A Benchmark Instance Gallery

The Benchmark Set consists of 203 original, challenging, and search-engine–resistant questions designed to require multi-step reasoning. To facilitate a deeper understanding of the ChemInsight benchmark, Example A.1 and Example A.2 present two question bundles from the benchmark set. These examples demonstrate tasks that integrate natural language descriptions, chemical logic, and symbolic representations. Each example includes prior knowledge, the input question, the verified ground truth, and scoring metadata.

Example A.1: Essence Question Bundle   
Problem:   
<begin\_knowledge>   
Metal ions can activate alkynes and enhance the electrophilicity of the triple bond by   
coordinating with the carbon-carbon triple bond, making it easy for them to undergo nucleophilic   
addition reactions with nucleophiles in the system.   
<end\_knowledge>   
<begin\_problem>   
The carbon atom alpha to the carbonyl group often exists as a carbanion or its equivalent.   
However, using alkynes, it is possible to transform the alpha-carbonyl carbocation.   
1) One method of transformation involves activating the alkyne with a metal ion and then   
attacking it with a nucleophilic oxidant. In the following reaction, monovalent gold activates   
the triple bond, and the resulting product forms a new, more strained ring system different from   
that of the substrate.   
reaction:   
COC1=CC=C(C#CC2CC2)C=C1 + ([LAuCl] + AgNTf2 + 5eq Ph2SO) ----> A   
<question>1-a</question>   
Please write the SMILES code for the product A.   
2) If the structure of the substrate is changed, the reaction results will be greatly changed. It   
is known that the product B does not contain a new high-tension ring system different from the   
substrate:   
reaction:   
COC1=CC=C(C#CC2CCC2)C=C1 + ([LAuCl] +AgNTf2 + 1.2eq Ph2SO + MeNO2) ----> B   
<question>2-a</question>   
Please give the SMILES code of the product B.   
<question>3-a</question>   
Please explain why a slight change in the structure of the reaction substrate can result in   
completely different products.   
<end\_problem>   
Ground truth:   
[   
{   
"sub\_question\_id": "1-a",   
"answer": "O=C(C1=CCC1)C2=CC=C(OC)C=C2"   
},<sub>{</sub>   
"sub\_question\_id": "2-a",   
"answer": "O=C(C1=CC=C(OC)C=C1)C(C2CCC2)C3=CC=CC=C3SC4=CC=CC=C4"   
},   
{   
"sub\_question\_id": "3-a",   
"answer": "The three-membered ring has a large tension and a large driving force for ring   
expansion, so the ring expansion occurs before the [3,3]σ rearrangement has time to occur. The   
four-membered ring has a small tension and a small driving force for ring expansion, so the

Example A.1: Essence Question Bundle (continued)   
[3,3]σ rearrangement occurs first."   
}   
]   
Max Score: 6   
Note:   
Full credit is awarded for covering any single keyword.   
Key words:   
large tension; large driving force for ring expansion; before the [3,3]σ rearrangement ; small   
driving force for ring expansion; [3,3]σ rearrangement occurs first   
Capabilities:   
{   
'1-a': AB   
'2-a': AB   
'3-a': D, E   
}

## Example A.2: Ordinary Question Bundle

Problem:   
<begin\_knowledge>   
Hypervalent iodine reagents are oxidizing agents that can bind to nucleophilic sites in   
molecules. Due to the inherent instability of hypervalent iodine, they tend to leave with a pair   
of electrons, oxidizing the original molecule. During this process, various unstable positive   
ions may be generated, which in turn trigger some intramolecularreactions.   
<end\_knowledge>   
<begin\_problem>   
Iodine reagents can also achieve ring construction through other reaction modes,such as the   
following question:)   
O=IC1=CC=CC=C1 + C=CC[Si](C)(C)C + (Lewis acid) ----> A   
<question>1-a</question>   
Please give the SMILES code of the product A . (Hint: There is still iodine atom in the product.)   
<end\_problem>   
Ground Truth:   
{   
"sub\_question\_id": "1-a",   
"Answer": "IC1=CC=CC=C1CC=C"   
}   
Max Score: 2   
Note: No Special Rules   
Key Words: No Key Words   
Difficulty: Medium   
Capabilities: AB

## B Prompt Design

To standardize model outputs within complex chemical scenarios, we designed a Task-Specific Structured Prompt (Prompt B.1). For assessment, to achieve high-quality, fine-grained evaluation for short-answer questions involving multi-step reasoning, we adopt the LLM-as-a-Judge paradigm: we employ an Evaluation-Specific Structured Prompt (Prompt B.2) that guides the judge model through a hierarchical protocol.

```markdown
Prompt B.1: Task-Specific Structured Prompt
You are an expert chemistry AI assistant. Your task is to answer the chemistry problem by
performing accurate mechanistic reasoning and providing a structured json response.
### Instructions
1. Read and understand the content in `<begin_knowledge><end_knowledge>` and
`<begin_problem><end_problem>`.
2. Identify all sub-questions marked by `<question>...</question>` tags (e.g., `1-a`, `2-b`).
3. For each sub-question, perform accurate chemical reasoning based on reaction mechanisms and
reagent behavior.
4. Generate a structured JSON response using the exact schema below.
### Output Schema
{"responses": [
{
"question_id": "string identifier of the sub-question (e.g., '1-a', '2-b')",
"answer_type": "type of the answer (only two capitalized options: 'SMILES' or 'TEXT')",
"answer": "answer value"
},
]
}
### Requirements
- Answer **every** sub-question defined by `<question>...</question>` in the order they appear.
- Use **standard SMILES notation** for molecular structures.
- The `question_id` must **exactly match** the text inside `<question>...</question>` (e.g.,
"1-a", "2-b") — preserve case and formatting.
- Do not add, remove, or modify any `question_id`.
- Ensure the JSON is syntactically valid (proper escaping, commas, quotes) and can be directly
parsed by software.
Begin now.
$question
```

```markdown
Prompt B.2: Evaluation-Specific Structured Prompt
You are a chemistry assessment expert responsible for accurately scoring responses to chemistry
questions.
## Input Data Description
You will be given the following inputs:
- **Question**: The full text of the chemistry question.
- **Question ID**: Identifier of the sub-question (e.g., "1-a", "2-b") — used as output key.
- **Reference Answer**: A natural language text representing the correct, expected answer.
- **Model Response**: A natural language response generated by a large model that needs to be
evaluated.
- **Scoring range**: The scoring range of this question must strictly follow this scoring range
and must not exceed or fall below this range.
- **Note**: Special scoring instructions. If not "No special rules", you must strictly follow the
rule described in this field.
- **Key words**: The key words you need to pay attention to when scoring the model's answer.
Your task is to assess the scientific accuracy, completeness, and conceptual alignment of the
model's response, and assign a score based on both the reference answer and any special rules.
## Scoring Rules and Criteria
### General Text Scoring Principles
Scores should reflect how closely the model response matches the reference answer in terms of key
concepts, scientific accuracy, and logical completeness. Use the following guidelines when no
special rule applies:
### 1. Full Credit (score = $max_score )
- The response correctly explains **all key chemical concepts** required by the question
(typically 1-2 core ideas).
- Explanation is scientifically accurate and logically complete, leading to the same conclusion
as reference.
- Paraphrasing or alternative wording is acceptable as long as meaning matches the reference.
- No irrelevant, incorrect, or extraneous concepts are introduced.
### 2. Partial Credit (score = up to 0.5 $max_score , typically 0.3-0.5 $max_score )
- Only **some** of the key concepts are correctly explained.
- The answer shows partial understanding but **omits or misexplains one major point**.
- Must NOT contain any concept that was **not mentioned or implied in the question**.
- Do NOT award more than half credit for incomplete or partially correct answers.
### 3. Zero Credit (score = 0.0)
- The response **introduces concepts not present or suggested in both the original question and
the reference**, even if they are chemically true.
- No keyword or point was mention/raised and clearly explained.
- The answer is factually wrong, off-topic, misunderstanding the quetion or contradicts
scientific principles.
- The answer does not conform to the format required by the question. (e.g., the question asks
for the chemical formula, but the answer is the name of the substance; the question requires a
fixed format, but the answer is not followed)
- When in doubt between partial and zero: if an unasked-for mechanism or term is added to justify
the answer, default to 0.
### Special Rule Handling (Highest Priority!)
- If **note = "No special rules"**, then:
- You **must** follow the instruction the special rules exactly.
- If a special rule in note conflicts with a general rule, **the special rule takes
precedence**.
```

```markdown
Prompt B.2: Evaluation-Specific Structured Prompt (continued)
- If no keyword or point in **note was **mention/raised** and **clearly explained with no
ambiguous or indicative description**, you **must** default 0 credit
- (Important!)Though you think the Model Response is reasonable, if it **goes against** the
rules above, **follow the rules above** and **don't begrudge to score partial even 0**
- Examples:
- `note: "The order of the answers is not important. Give 1 point for each correct answer and
<sup>zero</sup> <sup>points</sup> <sup>for</sup> <sup>the</sup> <sup>wrong</sup> <sup>number</sup> <sup>of</sup> <sup>answers."`</sup> → <sup>if</sup> <sup>number</sup> <sup>of</sup> <sup>answers</sup> <sup>is</sup> <sup>wrong,</sup> <sup>give</sup> <sup>zero</sup> <sup>point.</sup>
- If **Key words = "No key words"**, then when scoring, you need to pay attention to whether the
model's answer includes keywords or content with similar meanings. If so, then the answer is
considered to have touched upon that keyword. The more keywords the model's answer involves, the
higher the score. If it involves all keywords, then the model's answer will receive a perfect
score. Conversely, if no keywords are involved, then the answer will receive zero points.
## Scoring Process
1. **Check for Special Rule**: If it's not "No special rules", apply it immediately.
2. **Verify Question ID Alignment**: Confirm that the current evaluation is being performed for
the exact sub-question identified by **question_id**.
3. **Understand Context**: Read the full question to understand what is being asked.
4. **Identify Key Concepts**: Extract essential ideas from the reference answer.
5. **Semantic Comparison**: Evaluate whether the model response conveys the same meaning with
scientific correctness.
6. **Score Assignment**: Assign a float score strictly following the scoring range.
## Output Requirements
Return the result in strict JSON format:
{
"sub_question_id": "string identifier of the sub-question (e.g., '1-a', '2-b')",
"score": 2.7 "Numeric float value (e.g., 1.0, 1.8)"
}
## Begin Evaluation
Please score the following item according to the above rules:
### Question
$question
### Sub-Question ID
$sub_question_id
### Reference Answer
$answer
### Model Response
$answer_llm
### Scoring range
0.0 to $max_score
### Note
$note
### Key words
$key_words
Please output the evaluation result:
```