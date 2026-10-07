![](images/cfd363f5eb65b2d6abc2c77a7938e6f924e6d013d2a9f991d2b26bd595e5b94f.jpg)

# IdeaAnchor: Teaching LLMs to Turn Literature into Research Ideas

Ziyu Chen<sup>C</sup> Yilun Zhao<sup>Y</sup> Jiashuo Sun<sup>I</sup> Yiling Ma<sup>Y</sup>

Manasi Patwardhan<sup>T</sup> Arman Cohan<sup>Y</sup>

<sup>Y</sup> Yale University <sup>C</sup>The Unversity of Chicago

<sup>I</sup>University of Illinois Urbana-Champaign <sup>T</sup> Tata Consultancy Services

Homepage Code Dataset

## Abstract

Scientific research often begins by synthesizing ideas from a set of related papers to identify gaps and formulate new directions. However, training language models to perform this form of literature-grounded ideation remains challenging, as existing approaches based on prompting or feedback lack structured supervision for how papers should be synthesized. We introduce IdeaAnchor, a paradigm for training LLMs to perform research ideation, using structured specifications as privileged signals. Each IdeaAnchor instance encodes how each input paper should be synthesized into a successful idea, including their functional roles, relationships, and target synthesis criteria. We build this paradigm by mining instances from published papers, capturing how real ideas emerge from prior literature. We then train models via demonstration, self-distillation, and reinforcement learning, and further enhance generation with retrieval at inference time. Experiments show consistent improvements in ideation quality. Our analysis reveals a functional decomposition: anchor-based training strengthens creative synthesis, retrieval enhances detail elaboration, and combining both yields the best performance.

![](images/444ffb4951bf43fdd1079c60bc160d80715162e41fb52e597409f5515095b9de.jpg)  
Figure 1: Overview of IdeaAnchor. Starting from a set of papers, we extract IdeaAnchor instances from published papers by identifying the functional roles of prior work, their cross-paper relationships, and the target synthesis criteria. These structured anchors make otherwise implicit synthesis logic explicit, providing privileged supervision for demonstration learning, self-distillation, and reinforcement learning. At inference time, the framework uses either related papers or a topic query as input. It can also incorporate retrieval augmentation, which complements the learned synthesis capability by supplying factual and methodological details for elaborating generated research ideas.

## 1 Introduction

Scientific discoveries rarely emerge from a blank page. Before any novel finding, researchers read a cluster of related papers, identify what each one enables, locate what remains unresolved, and crystallize a research direction that is both new and defensible [Uzzi et al., 2013, Park et al., 2023, Pu et al., 2025]. This process decomposes naturally into two capabilities. (i) creative synthesis: analyzing prior work at a high level, recognizing gaps across multiple contributions, and formulating a research direction that no single paper addresses; (ii) detail elaboration: fleshing out the envisioned approach with concrete methodological specifics drawn from the technical substance of prior work. A strong research idea requires both a compelling direction grounded in literature and enough technical depth to be actionable.

LLMs have demonstrated impressive downstream scientific assistance, from summarizing and reasoning over papers [Zhao et al., 2025, Chen et al., 2026a] to executing well-defined experiments [Boiko et al., 2023, Nathani et al., 2025, Novikov et al., 2025, Wang et al., 2023]. Even so, generating novel and actionable research ideas remains challenging. Given some related papers, LLMs tend to produce fluent overviews or shallow concept blends, which often struggle to identify meaningful research gaps and develop concrete methods grounded in existing scholarly evidence [Gupta and Pruthi, 2025a, Si et al., 2024]. Comparing ideas generated from the same prior works, Chen et al. [2026b] further find a consistent distributional gap from human research taste: LLM ideas concentrate on bridge-like opportunities and synthesis methods, whereas human papers frame gaps and construct contributions in far more diverse ways. Recent efforts address this through goal-conditional plan generation with rubric rewards [Goel et al., 2025], execution feedback in sandboxed environments [Jansen et al., 2025, Lu et al., 2024], or single-paper hypothesis inversion [O’Neill et al., 2025]. However, these approaches either assume the research goal is already given, require domain-specific execution environments, or do not support multi-paper synthesis.

We argue that the bottleneck of ideation is the absence of training signals that capture what makes a good research idea emergefrom prior work. To address this, we introduce IdeaAnchor, per-instance and literature-derived specifications that encode the functional roles each input paper should play, the gaps that should be identified, and the patterns that should be leveraged in the synthesis. IdeaAnchor serves as a unified data interface for improving research ideation via 3 distinct training paradigms: (1) Anchor-Guided Demonstration: By prompting a strong external LLM with the input paper list with corresponding IdeaAnchor instance, we can produce high-quality demonstrations that the target model learns from via supervised fine-tuning (SFT). IdeaAnchor steer the teacher model to better utilize input literature, yielding higher-quality demonstrations for training. (2) Anchor-Based Self-Distillation: In the absence of an external teacher, the target model itself with IdeaAnchor can self-distill [Zhao et al., 2026]. The model’s anchor-conditioned outputs, which benefit from seeing the structured specification, serve as self-training targets. This creates a self-improvement loop: the model distills the knowledge embedded in IdeaAnchor into its own parameters. (3) Anchor-Privileged Reinforcement Learning: Privileged information in IdeaAnchor can be used for RL reward. [Christiano et al., 2017, Bai et al., 2022]. During GRPO [Shao et al., 2024], the reward judge checks each output against anchor-defined criteria. Through iterative reward optimization, the policy model gradually learns to generate contents that align with standards from real publications.

To construct training data for this paradigm at scale, we retrospectively reconstruct how published contributions build on prior work [Jurgens et al., 2018, Lo et al., 2020]. Specifically, we assemble a corpus of approximately 14K papers spanning machine learning and natural science. Leveraging LLMs, we first extract the core intellectual contribution of each work, then retrospectively trace its most influential prior studies, and further reverse-engineer the implicit anchor linking foundational literature to subsequent research innovations. By exploiting the inherent provenance structure of academic publications, this pipeline produces high-quality structured training signals, and a study with the original authors confirms that the mined prior works closely reflect the literature they credit for their ideas.

To determine whether anchor-derived training signals actually teach models to synthesize research ideas from prior work, evaluation must mirror the same provenance structure. However, existing ideation benchmarks are not designed for this question. For example, Si et al. [2024] evaluate open-ended idea generation with generic rubrics across four fixed dimensions; IdeaBench [Guo et al., 2025] ranks ideas against reference papers but without structured multi-paper input sets; LiveIdeaBench [Ruan et al., 2026] measures divergent thinking from single-keyword prompts; and SciMON [Wang et al., 2024] optimizes novelty given a single background context. These benchmarks typically rate open-ended ideas or single-paper transformations, without expert-verified prior-work sets and instance-specific criteria for multi-paper synthesis. We therefore build a benchmark where domain experts analyze academic papers, pinpoint their pivotal prior studies, and refine standardized evaluation metrics for literature-grounded ideation quality. Our experiments focus on creative synthesis: whether a model can recognize non-obvious gaps and formulate novel directions from multiple papers. We evaluate three anchor-driven training paradigms and inference-time retrieval augmentation with both LLM-based judging and human expert assessment [Liu et al., 2023, Zheng et al., 2023]. Across settings, training improves ideation quality, with RL producing the largest gains in creative synthesis; retrieval mainly enhances detailed content elaboration. Our results uncover a functional decomposition: gap recognition and direction formulation need to be internalized via training, whereas grounding conceived ideas into methodological details can be enhanced by inference-time retrieval.

Our primary contributions can be summarized as follows:

• We adopt three anchor-driven training paradigms, alongside inference-time retrieval augmentation to boost research ideation capability in LLMs.

• We propose a pipeline to mine the intellectual genealogy of academic publications and extract IdeaAnchor instances via LLMs, constructing a structured corpus to address the lack of dedicated training signals for research ideation tasks. We also build an expert-annotated benchmark with evaluation rubrics to systematically assess the quality of research ideation from prior works.

• Through extensive LLM-based and human evaluation, we verify the effectiveness of our framework and uncover a functional decomposition of research ideation, offering guidance for future research on AI-assisted scientific innovation.

## 2 IdeaAnchor

This section first defines the task of literature-conditional research ideation and then introduces IdeaAnchor, the structured specifications that form the backbone of our training framework.

## 2.1 Literature-Conditional Research Ideation

Let $\mathcal { P } = \{ p _ { 1 } , . . . , p _ { N } \}$ denote a set of N related papers within a research topic, where each paper $p _ { i } = ( t _ { i } , c _ { i } )$ consists of a title $t _ { i }$ and a brief summary $c _ { i }$ . A model π<sub>θ</sub> is required to produce a structured output $y = ( \mathcal { T } , \mathcal { M } , \mathcal { R } )$ consisting of:

• Thinking Trace $\tau { : }$ a structured analysis of each input paper’s functional role and inter-paper relationships, articulating the synthesis logic that leads to a new research direction.

• Research Motivation M: a novel problem definition derived from identifying gaps, limitations, or unexplored combinations across the input papers.

• Research Plan R: a detailed methodology that synthesizes techniques from the input papers to address the identified problem.

Here, the research motivation M must be discovered from the literature rather than given as input, and the plan R must be grounded in the specific papers provided.

## 2.2 IdeaAnchor Specification

An IdeaAnchor $A _ { D }$ for a training instance is a structured specification derived from a ground-truth paper D and its input literature P. It encodes three types of information:

• Functional Role Assignments $\{ \rho _ { i } \} _ { i = 1 } ^ { N }$ : Each prior work $p _ { i }$ is assigned one of four roles that characterize its relationship to $D \colon$ Direct Predecessor (the method most immediately extended or improved upon), Inspiration Source (a technique or idea from a different context that sparked the approach), Gap & Motivation (work whose limitations define the research problem), and Methodological Ingredient (a specific technical component incorporated into the proposed method) [Cohan et al., 2019, Jurgens et al., 2018]. Each role assignment is accompanied by a learned-insight summary: what the authors of D learned from $p _ { i }$ and how it influenced the contribution.

• Relationship Analysis $S _ { D } \colon \mathbf { A }$ structured analysis that articulates the synthesis logic connecting the prior works $\mathcal { P }$ to $D ' s$ contribution:

– Per-paper analysis: For each prior work, what it achieved, what limitation remains, and why that limitation matters for the eventual contribution.

– Cross-paper relationship analysis: How the works relate to each other, including whose ideas address whose limitations, what combinations open new possibilities, and how the logical chain across papers points toward an unexplored direction.

• Checkable Criteria $\mathcal { U } _ { D } = \{ u _ { 1 } , . . . , u _ { K } \}$ : A set of verifiable items that a successful idea should satisfy, spanning three dimensions:

– Motivation criteria: Does the output identify the key gap or limitation that D addresses?

– Method criteria: Does the proposed approach incorporate details from the prior works?

![](images/7771397752e669d255c6c1ccf42dbe651b923affa036297a9dca4b5bc616b80e.jpg)  
Figure 2: A sample IdeaAnchor instance derived from a paper on Byzantine-resilient distributed learning. The anchor encodes per-paper role assignments with learned insights (top), a cross-paper relationship analysis identifying the synthesized gap (bottom-left), and literature-grounded checkable criteria (bottom-right).

– Overall criteria: Is the synthesis coherent, and does the proposed direction logically follow from the cross-paper analysis?

The value of IdeaAnchor is that they are instance-specific and literature-grounded: rather than generic quality criteria (e.g., Is the idea novel?), each anchor is derived from the particular way a real set of papers led to a real published contribution. This makes anchors a form of privileged information [Vapnik and Vashist, 2009], knowledge available at training time (from the ground-truth paper) but not at inference, when the model must synthesize without knowing the answer. The anchor concept unifies the structured reasoning process, the rewards signal for RL, and the specification for high-quality demonstrations. By framing all three as aspects of a single underlying specification, we enable a comparison of different training strategies that exploit the same information source.

## 2.3 Mining IdeaAnchor from Research Papers

We construct IdeaAnchor at scale by reverse-engineering the intellectual genealogy of existing research papers. The pipeline is automated, using a sample creator model $\mathcal { M } _ { C }$ to progressively build the structured specification from a ground-truth paper D in three stages. (i) Prior work extraction. Given the full text of D, $\mathcal { M } _ { C }$ extracts the core idea of D and identifies 5–7 prior works that substantively shaped D’s contribution [Beltagy et al., 2019, Lo et al., 2020]. Each prior work $p _ { i }$ is assigned a functional role $\rho _ { i }$ and a learned-insight annotation as described in Section 2.2. (ii) Literature enrichment. The extracted prior works are queried against scholarly search engines to retrieve their abstracts [Priem et al., 2022], providing factual grounding beyond how D describes them. (iii) Criteria generation. Given the prior works, corresponding analysis, and reference proposal, $\mathcal { M } _ { C }$ generates several checkable criteria $\boldsymbol { \mathcal { U } } _ { D }$ spanning motivation, method, and overall dimensions, ensuring that evaluation criteria are anchored in the specific intellectual genealogy of D. The full prompts for each stage are provided in Appendix A.

Author Validation. Beyond being topically related to D, the mined prior works are expected to be the ones that actually shaped its core idea. To verify this, we sent authors of benchmark papers the extracted prior works together with the reconstructed idea-formation path, and asked whether these reflect how their idea formed, which important works are missing, and which included works were not important. We received responses from the authors of 20 papers, 16 of whom were satisfied with the mined set. Five authors named one to three missing works (9 in total), so the mined sets cover 93.7% of the works the authors credit; two authors flagged included works as unimportant, amounting to only 3 of the 134 extracted works (2.2%). The mined sets therefore closely reflect the prior works that authors themselves credit for their ideas.

Paper Corpus. We apply the mining pipeline to two domains. (i) Machine Learning. We collect 7,494 accepted papers from ICLR, ICML, and NeurIPS, spanning from 2023 to 2025. (ii) Natural Science. 6,689 papers published from 2023 to 2025 on Nature Communications, spanning 71 major scientific subject such as physics, chemistry, and neuroscience.

Evaluation Benchmark. To assess generalization, we construct a temporally held-out evaluation set from 924 papers accepted at ICLR 2026, including all the 224 oral presentations. For each paper, we apply the same pipeline to generate candidate prior works and rubric criteria, which are then reviewed and refined by expert annotators. Annotators verify the correctness of extracted prior works, adjust role assignments where necessary, and edit criteria items to ensure they are unambiguous and faithfully reflect the paper’s contribution. We describe the annotation protocol and inter-annotator agreement in detail in Appendix D.

## 3 Boosting Research Ideation with IdeaAnchor

We leverage IdeaAnchor in boosting research ideation. During training (Section 3.1), three complementary paradigms exploit anchors as prompting context, privileged input, and reward signal. At inference (Section 3.2), two enhancements extend the model beyond abstract-only reasoning: retrieval-augmented generation with extended-text depth, and automated topic-driven literature discovery [Izacard and Grave, 2021, Lewis et al., 2020, Nakano et al., 2021].

## 3.1 Anchor-Driven Training

All three strategies treat $A _ { D }$ as privileged information [Vapnik and Vashist, 2009], which is available during training but absent at deployment, and differ in how they convert it into learning signal.

Anchor-Guided Demonstration. Direct use of privileged information is to steer a strong teacher model toward higher-quality demonstrations [Hinton et al., 2015, Hsieh et al., 2023]. Without structured guidance, teacher outputs might be fluent but often fail to deeply engage with the input literature [Goel et al., 2025]; providing a specification alongside $\mathcal { P }$ yields demonstrations that are more grounded. For each instance $( \mathcal { P } , \mathcal { A } _ { D } )$ , we prompt an external LLM $\mathcal { M } _ { \mathrm { e x t } }$ with both inputs: $\boldsymbol { y } _ { \mathrm { d e m o } } = \mathcal { M } _ { \mathrm { e x t } } ( \mathcal { P } , \mathcal { A } _ { D } )$ . The target model $\pi _ { \theta }$ is trained on the expert output with the anchor removed:

$$
\mathcal { L } _ { \mathrm { d e m o } } = - \mathbb { E } _ { ( \mathcal { P } , y _ { \mathrm { d e m o } } ) } \left[ \log \pi _ { \theta } ( y _ { \mathrm { d e m o } } \mid \mathcal { P } ) \right] .\tag{1}
$$

The anchor’s influence is thus embedded in the demonstration: $\pi _ { \theta }$ learns to satisfy anchor criteria without ever observing $A _ { D }$

Anchor-Based Self-Distillation. Self-distillation removes the external-teacher requirement by exploiting the asymmetry between a model’s output quality with versus without privileged context. The privileged-conditioned model serves as its own teacher while the unprivileged model is the student [Zhao et al., 2026]. We sample $y _ { \mathrm { s e l f } } = \pi _ { \boldsymbol { \theta } } ( \mathcal { P } , \mathcal { A } _ { D } )$ and train $\pi _ { \theta }$ on its own anchor-conditioned outputs with $A _ { D }$ removed:

$$
\mathcal { L } _ { \mathrm { s e l f } } = - \mathbb { E } _ { ( \mathcal { P } , \mathit { y } _ { \mathrm { s e l f } } ) } \left[ \log \pi _ { \theta } ( \boldsymbol { y } _ { \mathrm { s e l f } } \mid \mathcal { P } ) \right] .\tag{2}
$$

We use system persona to prevent the model from revealing privileged information during thinking, avoiding training illusions [Kim et al., 2026]. We also compared different anti-leak strategies in Appendix H.

Anchor-Privileged Reinforcement Learning. Demonstration-based methods optimize a proxy (matching teacher output) rather than directly optimizing what defines a good idea. RL closes this gap: instance-specific privileged knowledge supplies a reward that is both more informative and harder to game than generic criteria. A judge $\theta _ { r }$ receives $\left( \mathcal { P } , y , A _ { D } \right)$ and scores each checkable item $u \in \mathcal { U } _ { D }$

$$
r _ { \mathrm { a n c h o r } } = \frac { 1 } { | \mathcal { U } _ { D } | } \sum _ { u \in \mathcal { U } _ { D } } \mathbb { I } [ \theta _ { r } ( \mathcal { P } , y , u ) = \mathrm { s a t i s f i e d } ] .\tag{3}
$$

Each $u _ { i }$ is deemed satisfied only if no general quality guideline Γ (specificity, soundness, feasibility) is violated. We optimize with GRPO [Shao et al., 2024]: for each $\mathcal { P }$ the policy samples G candidates, scored by rewar $1 ( y ) =$ $r _ { \mathrm { a n c h o r } } ( y ) - \lambda \cdot \mathbb { I } \{$ {format violation}, and updates toward higher-scoring ones. Training is initialized from the base model, while the judge is flexible to be either frozen or evolving during training.

## 3.2 Inference-Time Enhancements

Training internalizes creative synthesis, gap recognition and direction formulation. These capabilities let the model effectively conceptualize promising research directions, but lack specific details to translate ideas into actionable plans. Producing actionable proposals requires detail elaboration, which we address with two inference-time mechanisms.

Algorithm 1: Ideation Pipeline   
Require: Papers P or keywords $q ;$ Search, Retrieve, π<sub>θ</sub>   
1: if input is q then   
2: $\dot { \mathcal { P } } \gets \dot { \mathsf { S e a r c h } } ( q , N )$   
3: $\mathcal { T }  \pi _ { \theta } . \mathrm { t h i n k } ( \mathcal { P } )$ ▷ role & relationship analysis   
4: for $p _ { i } \in \mathcal { P }$ with full-text access do   
5: $\mathsf { \bar { d } } _ { i } \gets \mathsf { R e t r i e v e } ( p _ { i }$ , ExtractRole $( \mathcal { T } , p _ { i } ) )$   
6: return π<sub>θ</sub>.generate $\big ( \{ ( p _ { i } , d _ { i } ) \} , \mathcal { T } \big )$

Role-Aware Retrieval Augmentation. When only abstracts are available, the model lacks methodological depth for tangible plans. We address this via role-aware retrieval: $\pi _ { \theta }$ first generates a thinking trace T that assigns each $p _ { i }$ a functional role $\rho _ { i }$ . The system then fetches targeted full-text sections, including methods for $K e y$ Methodology, results for Primary Baseline, and problem statements for Gap & Motivation, and produces the final proposal from the augmented set:

$$
( \mathcal { M } , \mathcal { R } ) = \pi _ { \boldsymbol { \theta } } \big ( \{ ( p _ { i } , d _ { i } ) \} _ { i = 1 } ^ { N } , ~ \mathcal { T } \big ) ,\tag{4}
$$

where $d _ { i } = \mathsf { R e t r i e v e } ( p _ { i } , \rho _ { i } )$ denotes role-guided passages.

Automated Topic-Driven Ideation. The model also accepts a high-level topic $q$ as an alternative to specific papers. It queries a search API to retrieve N candidate papers, then applies the same synthesis pipeline [Priem et al., 2022, Yao et al., 2022]. This pipeline can be optionally combined with full-text retrieval to generate a proposal in an end-to-end manner. Algorithm 1 unifies the two inference modes.

## 4 Experiments

We evaluate whether IdeaAnchor-driven training improves literature-conditional research ideation beyond prompting alone. Our experiments ask three questions: (i) whether specialized training improves over the Qwen3-8B base model and approaches strong proprietary models; (ii) how retrieval-augmented inference complements training; and (iii) whether the learned synthesis behavior transfers across training scale, domains, and open-ended topic-driven ideation.

## 4.1 Experimental Setup

Models and Training Data. All trained policies are initialized from Qwen3-8B and Qwen3.5-9B [Yang et al., 2025, Qwen Team, 2026]. We use the mined IdeaAnchor corpus described in Section 2.3, each training instance consists of 5 to 7 prior papers represented by titles and abstracts, together with the privileged IdeaAnchor extracted from the target paper. The three training paradigms in Section 3.1 differ only in how this privileged information is converted into learning signal. For anchor-guided demonstrations, we prompt GPT-5.4-mini to get the structured proposal for fine-tuning. For self-distillation, the trained model itself is prompted with the anchor and then trained on its own anchor-conditioned output under the unprivileged input. For RL, candidate proposals are scored item-by-item against the anchor criteria by GPT-5.4-mini judge, and the scores are used as rewards to optimize the policy. All model outputs follow the same structured format: $y = ( \mathcal { T } , \mathcal { M } , \mathcal { R } )$ . Prompts and details in Appendix B.

Retrieval-Augmented Inference. At inference time we evaluate both abstract-only generation and two retrieval variants. RAG-Full follows the method in Section 3.2: the model first assigns functional roles to input papers during thinking, then retrieves role-relevant passages and generates from the augmented context. RAG-Summary uses the same retrieved passages but compresses each paper into a targeted summary by merging the abstract with the retrieved evidence. This second variant keeps input length and style close to the training distribution while injecting more task-directed methodological information.

Automated Evaluation. For each instance in our evaluation benchmark described in Section 2.3, GPT-5.4 judges the generated proposal against the expert-refined checkable criteria. Each criterion is marked as satisfied only if the proposal puts forward a concrete, literature-grounded claim that meets the criterion while maintaining basic soundness and feasibility. We report the criteria satisfaction rate (CSR), averaged across all rubric items in the benchmark.

Human Evaluation. While automated rubrics support large-scale assessment, they cannot fully capture subjective nuances in research ideation. We conduct blind pairwise human evaluation to complement automatic metrics. We compare proposals from Qwen3-8B base and corresponding trained variants under identical input. Experts rank overall strength and assess literature grounding, synthetic novelty, methodological specificity, and feasibility. Presentation order is randomized, with ties allowed. Full annotation guidelines and subset statistics are provided in Appendix G.

![](images/d6ffad7465ce2511a0db4792ed996694fc1704a112fda809f4695a93f78c3bc5.jpg)  
Figure 4: Pairwise preference over the base model. Each stacked bar compares a trained Qwen3-8B variant with the base model. Colored segments denote trained-model wins, gray segments denote base wins, and the center segment denotes ties. Hatched bars are GPT-5.4 judgments and solid bars are human evaluations.

![](images/fe205cbccd62460e7ecf97069e89639db22ba3ed41bda83b0719e12244a3a896.jpg)  
Figure 5: Training dynamics of anchor-driven ideation. CSR is measured on the ICLR benchmark across training stages for SFT, Self-Distillation, and RL over Qwen3-8B. Solid curves use machine-learning anchors, while dashed curves use only natural-science anchors and are evaluated on the same benchmark.

## 4.2 Main Results

Training with IdeaAnchor Improves Literature-Grounded Ideation. Figure 3 reports CSR on 924 ICLR instances, with proprietary models as upper references. Every training paradigm improves its base model. On Qwen3-8B, self-distillation reaches 16.3% (+5.4), SFT 21.7% (+10.8), and RL 24.6% (+13.7), versus 10.9% for th base. On Qwen3.5-9B, SFT and RL raise CSR from 31.1% to 48.7% and 54.2%, respectively, with RL surpassing the references. Anchor supervision remains effective when the starting model is substantially stronger, rather than compensating for limited base-model capacity. RL gives the largest gain at both scales even though its reward contains only instance-specific criteria, suggesting that the policy learns a general pattern of gap identification and cross-paper synthesis instead of surface rubric matching. Retrieval adds smaller improvements, 25.6% for RL with RAG-Full and 22.9% for SFT with RAG-Summary, so access to more evidence cannot replace learned synthesis. We examine this difference in Section 4.3.

Pairwise Ranking Confirms Holistic Quality Gains. Beyond item-wise rubric satisfaction, we conduct pairwise preference ranking on the same outputs. The judge receives two proposals, one from a trained variant and one from Qwen3-8B base, in randomized order and selects the one that presents better. Unlike CSR, which decomposes quality into independent checklist items, pairwise comparison captures holistic synthesis quality: whether a proposal constitutes a more coherent and actionable research direction given the same literature context. Figure 4 reports win rates against the base under three judge configurations: LLM judge GPT-5.4, and human experts. Despite variation in absolute magnitudes across paradigms and judges, the overall trend holds: all three training paradigms achieve >75% pairwise win rate against the base model. RL leads, followed closely

![](images/3fa79a66ee2a84c39b39e0e8159804d606c92df947525065f15252a1a68ad4b0.jpg)  
Figure 3: Main automated evaluation on the ICLR 2026 benchmark with 924 instances. Bars report CSR (%) for the base, self-distillation (SSD), SFT, and RL variants of Qwen3-8B and Qwen3.5-9B. Horizontal lines show proprietary model references.

by SFT. Even self-distillation, which requires no external teacher, attains substantial preference gains, reinforcing that the anchor signal itself is the primary driver of improvement.

## 4.3 Analysis and Case Studies

Retrieval Mostly Improves Methodological Detail. The two variants test an inference-time axis orthogonal to anchor training: RAG-Full exposes the model to role-relevant full text, whereas RAG-Summary compresses that evidence into an abstract-style input. Table 1 shows that both are strongly preferred to abstractonly generation across all training paradigms, despite their modest CSR gains. We manually inspect 100 paired outputs to resolve this discrepancy. Retrieved passages add method choices, experimental settings, baselines, and implementation constraints, making proposals more actionable and improving holistic preference. They rarely change the high-level gap, motivation, or direction established earlier during creative synthesis. Because CSR primarily rewards alignment with that direction, extra technical detail cannot recover a proposal whose synthesis is wrong.

Table 1: Pairwise preference of RAG over abstract-only outputs. Each cell reports the win rate of the retrieval variant against the corresponding abstract-only output under a GPT-5.4 judge.
<table><tr><td>Variant</td><td>RAG-Fu11</td><td>RAG-Sum.</td></tr><tr><td>SFT</td><td>73.6</td><td>74.2</td></tr><tr><td>SSD</td><td>68.7</td><td>70.3</td></tr><tr><td>RL</td><td>76.8</td><td>75.5</td></tr></table>

This distinction also explains two automated trends. Retrieval slightly hurts the base model because unreliable role assignment can surface evidence for a weak premise and amplify it. Conversely, RAG-Full slightly outperforms the distribution-matched RAG-Summary for the strongest RL policy: once the model chooses a sound direction, richer full-text evidence becomes more useful than avoiding distribution shift. Retrieval is therefore best at elaborating a direction the model has already formulated, not deciding that direction from scratch. Accordingly, CSR and pairwise preference measure complementary stages of the pipeline: the former emphasizes selecting the right research direction, while the latter also rewards how fully that direction is operationalized.

Cross-Domain Generalization and Scaling. Figure 5 traces CSR throughout training for all three anchor-driven paradigms, with solid curves trained on machine-learning papers and dashed curves trained only on natural-science papers but evaluated on the same ICLR benchmark. In-domain training improves steadily with scale: SFT and RL exhibit large gains over the base model, while self-distillation yields a smaller but consistent increase, suggesting that additional anchor supervision strengthens literature-grounded synthesis rather than merely fitting the evaluation set. The models trained on natural-science IdeaAnchor also improve on the machine-learning ideation quality across all paradigms, despite never observing ML anchors during training. This transfer indicates that IdeaAnchor teach a domain-general ideation procedure: identifying a gap in prior work, relating evidence across papers, and formulating a motivated resolution, rather than memorizing domain-specific topics or surface patterns.

From-Scratch Ideation. Finally, we test whether the trained model can be used when no curated input papers are provided. We run a small topic-driven study that asks a narrower question: can the full pipeline move from an open research area to a literature-grounded proposal that experts find actionable? We select six broad machine-learning topics, prompt the model to generate search queries for each topic, retrieve recent papers through Semantic Scholar API [Kinney et al., 2023], and let the model select 6 to 8 papers per topic from the returned candidates based on topical relevance and diversity. The RL policy with RAG-Full then reasoning over the retrieved papers, generates multiple candidate proposals, and uses a lightweight pairwise ranker to select the final output. Three ML researchers conduct blind review on a 1 to 5 scale for novelty, literature grounding, methodological specificity, feasibility, and overall quality.

Table 2 suggests that anchor training remains useful beyond the benchmark setting. The full system improves most on literature grounding and methodological specificity, while maintaining feasibility. Qualitative reviews indicate that the main failure mode is not lack of retrieved evidence but weak selection among competing directions: lower-scoring proposals often combine individually relevant papers without committing to a sharp gap. In contrast, the stronger outputs first identify a non-obvious tension across the retrieved literature and then use full-text evidence to instantiate an experiment plan. This aligns with our benchmark analysis.

Table 2: Human evaluation of topic-driven ideation. Scores are averaged over proposals from six open ML topics.
<table><tr><td>Ideation System</td><td></td><td>Novelty Grounding Specificity Feasibility</td><td></td><td></td><td>Overall</td></tr><tr><td>Qwen3-8B</td><td>3.1</td><td>2.8</td><td>3.2</td><td>3.0</td><td>2.9</td></tr><tr><td>Qwen3-8B-RL w/ RAG-Ful1</td><td>3.5</td><td>3.9</td><td>3.8</td><td>3.2</td><td>3.7</td></tr></table>

## 5 Related Work

AI for Scientific Discovery. LLMs are increasingly deployed across the scientific research pipeline [Wang et al., 2023], assisting with literature search and synthesis [Skarlinski et al., 2024, Zhao et al., 2025], research code generation [Wijk et al., 2024, Nathani et al., 2025], and automated peer review [D’Arcy et al., 2024, Liang et al., 2024]. Agents perform end-to-end experiment execution by optimizing objectives in sandboxed environments [Boiko et al., 2023, Jansen et al., 2025, Novikov et al., 2025], while multi-agent systems aim to automate workflows from hypothesis generation to paper writing [Gottweis et al., 2025, Schmidgall and Moor, 2025, Schmidgall et al., 2025, Yamada et al., 2025]. For idea generation, Si et al. [2024] provide the first large-scale human evaluation, finding LLM-generated ideas novel but weaker on feasibility. Follow-up systems refine ideas with reviewing agents over academic graphs [Baek et al., 2025], retrieve prior-paper inspirations and optimize novelty [Wang et al., 2024], search diverse plans [Hu et al., 2024], evolve and compose idea facets [Pu et al., 2025], recombine extracted facets with human-in-the-loop novelty verification [Radensky et al., 2026], or invert single-concept assumptions via structured schemas [O’Neill et al., 2025]. AInstein [Mishra et al., 2025] and follow-up studies [Gupta and Pruthi, 2025b] further assess feasibility and quality. More recently, CHIMERA [Sternlicht and Hope, 2026] mines a large-scale knowledge base of pairwise concept recombinations from literature and trains a hypothesis generation model on it. However, most of these methods primarily rely on prompting strategies or single-paper manipulation at inference time, without explicit training for multi-paper synthesis.

Training with Privileged Information. IdeaAnchor builds on the principle of learning using privileged information: additional information available only during training can guide learning [Vapnik and Vashist, 2009]. In the context of LLMs, RL with LLM-graded rubrics extends training beyond verifiable domains [Ouyang et al., 2022]. Goel et al. [2025] use goal-specific rubrics as privileged information for self-grader to train research plan generators. LDC [Li et al., 2024] trains idea generators via SFT on paper-derived pairs and controllable RL with multi-dimensional reward models. Extensions include domain-specific rubric rewards [Gunjal et al., 2025], evolving rubrics that co-evolve with the policy [Shao et al., 2025], rubrics as dual-purpose exploration scaffolding and rewards [Zhou et al., 2025], and rubric refinement via recursive decomposition [Shen et al., 2026]. Recent work explores self-improvement where teacher and student share the same weights but differ in input access [Zhao et al., 2026]. π-Distill [Penaloza et al., 2026] jointly trains a privileged-conditioned teacher and an unconditioned student; GATES [Stein et al., 2026] gates distillation on consensus among privileged rollouts; HDPO [Ding, 2026] augments RL with privileged self-distillation on failures.

## 6 Conclusion and Discussion

We presented IdeaAnchor, a framework that mines structured, instance-specific specifications from published papers and uses them as privileged training signals for literature-grounded research ideation. Three complementary paradigms: demonstration, self-distillation, and reinforcement learning exploit these specifications, while role-aware retrieval augmentation enhances generation at inference time. Our analysis reveals a functional decomposition: creative synthesis benefits from training, whereas detail elaboration is driven by retrieval.

Our results also suggest several directions for future work. (i) Selecting the research gap. Anchor-based training improves ideation, whereas retrieval mainly adds methodological details without changing the gap a proposal targets. In the from-scratch study, low-scoring proposals combined relevant papers without committing to a clear gap. Training models to propose multiple candidate gaps and rank them is therefore a promising next step. (ii) Multiple valid ideas. Each instance uses one published paper as its target, but the same prior works might lead to several valid ideas. The anchor guides the model toward one well-grounded direction, but this does not mean that other directions are worse. Using several papers that build on the same prior works as targets, and measuring the diversity of generated ideas, would be promising [Chen et al., 2026b, Deng et al., 2026]. (iii) Execution-level validation. Our evaluation focuses on written proposals, while LLM ideas can lose more score than human ideas once they are executed [Si et al., 2026]. Implementing generated ideas and evaluating their experimental results is thus an important direction, although such validation is not feasible in every domain.

## Acknowledgments

This work was supported in part by the U.S. National Science Foundation under award No. 2541654 and the Tata Consultancy Services.

## References

Brian Uzzi, Satyam Mukherjee, Michael Stringer, and Ben Jones. Atypical combinations and scientific impact. Science, 342(6157):468–472, 2013.

Michael Park, Erin Leahey, and Russell J Funk. Papers and patents are becoming less disruptive over time. Nature, 613(7942):138–144, 2023.

Kevin Pu, KJ Kevin Feng, Tovi Grossman, Tom Hope, Bhavana Dalvi Mishra, Matt Latzke, Jonathan Bragg, Joseph Chee Chang, and Pao Siangliulue. Ideasynth: Iterative research idea development through evolving and composing idea facets with literature-grounded feedback. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems, pages 1–31, 2025.

Yilun Zhao, Kaiyan Zhang, Tiansheng Hu, Sihong Wu, Ronan Le Bras, Taira Anderson, Jonathan Bragg, Joseph Chee Chang, Jesse Dodge, Matt Latzke, et al. Sciarena: An open evaluation platform for foundation models in scientific literature tasks. arXiv preprint arXiv:2507.01001, 2025.

Ziyu Chen, Yilun Zhao, Chengye Wang, Rilyn R. Han, Manasi Patwardhan, and Arman Cohan. SciMDR: Advancing scientific multimodal document reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 44718–44742, 2026a.

Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023.

Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, et al. Mlgym: A new framework and benchmark for advancing ai research agents. arXiv preprint arXiv:2502.14499, 2025.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey˜ Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the age of artificial intelligence. Nature, 620(7972): 47–60, 2023.

Tarun Gupta and Danish Pruthi. All that glitters is not novel: Plagiarism in ai generated research. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25721–25738, 2025a.

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers. arXiv preprint arXiv:2409.04109, 2024.

Ziyu Chen, Yilun Zhao, and Arman Cohan. Measuring the gap between human and LLM research ideas. arXiv preprint arXiv:2607.01233, 2026b.

Shashwat Goel, Rishi Hazra, Dulhan Jayalath, Timon Willi, Parag Jain, William F Shen, Ilias Leontiadis, Francesco Barbieri, Yoram Bachrach, Jonas Geiping, et al. Training ai co-scientists using rubric rewards. arXiv preprint arXiv:2512.23707, 2025.

Peter Jansen, Oyvind Tafjord, Marissa Radensky, Pao Siangliulue, Tom Hope, Bhavana Dalvi, Bodhisattwa Prasad Majumder, Daniel S Weld, and Peter Clark. Codescientist: End-to-end semi-automated scientific discovery with code-based experimentation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 13370–13467, 2025.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

Charles O’Neill, Tirthankar Ghosal, Roberta Raileanu, Mike Walmsley, Thang Bui, Kevin Schawinski, and Ioana˘ Ciuca. Sparks of science: Hypothesis generation using structured paper data.˘ arXiv preprint arXiv:2504.12976, 2025.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

David Jurgens, Srijan Kumar, Raine Hoover, Dan McFarland, and Dan Jurafsky. Measuring the evolution of a scientific field through citation frames. Transactions ofthe Associationfor Computational Linguistics, 6:391–406, 2018.

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel S Weld. S2orc: The semantic scholar open research corpus. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 4969–4983, 2020.

Sikun Guo, Amir Hassan Shariatmadari, Guangzhi Xiong, Albert Huang, Eric Xie, Stefan Bekiranov, and Aidong Zhang. Ideabench: Benchmarking large language models for research idea generation. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025.

Kai Ruan, Xuan Wang, Jixiang Hong, Peng Wang, Yang Liu, and Hao Sun. Evaluating llms’ divergent thinking capabilities for scientific idea generation with minimal context. Nature Communications, 2026.

Qingyun Wang, Doug Downey, Heng Ji, and Tom Hope. Scimon: Scientific inspiration machines optimized for novelty. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 279–299, 2024.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 2511–2522, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Arman Cohan, Waleed Ammar, Madeleine Van Zuylen, and Field Cady. Structural scaffolds for citation intent classification in scientific publications. In Proceedings ofthe 2019 conference ofthe North American chapter of the Association for Computational Linguistics: human language technologies, volume 1 (long and short papers), pages 3586–3596, 2019.

Vladimir Vapnik and Akshay Vashist. A new learning paradigm: Learning using privileged information. Neural networks, 22(5-6):544–557, 2009.

Iz Beltagy, Kyle Lo, and Arman Cohan. Scibert: A pretrained language model for scientific text. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 3615–3620, 2019.

Jason Priem, Heather Piwowar, and Richard Orr. Openalex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv preprint arXiv:2205.01833, 2022.

Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings ofthe 16th conference ofthe european chapter ofthe associationfor computational linguistics: main volume, pages 874–880, 2021.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8003–8017, 2023.

Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of llms? arXiv preprint arXiv:2603.24472, 2026.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

Rodney Kinney, Chloe Anastasiades, Russell Authur, Iz Beltagy, Jonathan Bragg, Alexandra Buraczynski, Isabel Cachola, Stefan Candra, Yoganand Chandrasekhar, Arman Cohan, et al. The semantic scholar open data platform. arXiv preprint arXiv:2301.10140, 2023.

Michael D Skarlinski, Sam Cox, Jon M Laurent, James D Braza, Michaela Hinks, Michael J Hammerling, Manvitha Ponnapati, Samuel G Rodriques, and Andrew D White. Language agents achieve superhuman synthesis of scientific knowledge. arXiv preprint arXiv:2409.13740, 2024.

Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, et al. Re-bench: Evaluating frontier ai r&d capabilities of language model agents against human experts. arXiv preprint arXiv:2411.15114, 2024.

Mike D’Arcy, Tom Hope, Larry Birnbaum, and Doug Downey. Marg: Multi-agent review generation for scientific papers. arXiv preprint arXiv:2401.04259, 2024.

Weixin Liang, Yuhui Zhang, Hancheng Cao, Binglu Wang, Daisy Yi Ding, Xinyu Yang, Kailas Vodrahalli, Siyu He, Daniel Scott Smith, Yian Yin, et al. Can large language models provide useful feedback on research papers? a large-scale empirical analysis. NEJM AI, 1(8):AIoa2400196, 2024.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an ai co-scientist. arXiv preprint arXiv:2502.18864, 2025.

Samuel Schmidgall and Michael Moor. Agentrxiv: Towards collaborative autonomous research. arXiv preprint arXiv:2503.18102, 2025.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents as research assistants. Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 5977–6043, 2025.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. Researchagent: Iterative research idea generation over scientific literature with large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6709–6738, 2025.

Xiang Hu, Hongyu Fu, Jinge Wang, Yifeng Wang, Zhikun Li, Renjun Xu, Yu Lu, Yaochu Jin, Lili Pan, and Zhenzhong Lan. Nova: An iterative planning and search approach to enhance novelty and diversity of llm generated ideas. arXiv preprint arXiv:2410.14255, 2024.

Marissa Radensky, Simra Shahid, Raymond Fok, Pao Siangliulue, Tom Hope, and Daniel S Weld. Scideator: Humanllm compound system for scientific ideation through facet recombination and novelty evaluation. In Proceedings ofthe ACM Conference on AI and Agentic Systems, 2026.

Shambhavi Mishra, Gaurav Sahu, Marco Pedersoli, Laurent Charlin, Jose Dolz, and Christopher Pal. Ainstein: Assessing the feasibility of ai-generated approaches to research problems. arXiv preprint arXiv:2510.05432, 2025.

Tarun Gupta and Danish Pruthi. All that glitters is not novel: Plagiarism in ai generated research. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25721–25738, 2025b.

Noy Sternlicht and Tom Hope. CHIMERA: A knowledge base of scientific idea recombinations for research analysis and ideation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, 2026.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Ruochen Li, Liqiang Jing, Chi Han, Jiawei Zhang, and Arman Cohan. LDC: Learning to generate research idea with dynamic control. arXiv preprint arXiv:2412.14626, 2024.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G Finlayson, David Sontag, et al. Dr tulu: Reinforcement learning with evolving rubrics for deep research. arXiv preprint arXiv:2511.19399, 2025.

Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Kongcheng Zhang, Jiale Zhao, Jingwen Yang, Yihe Zhou, Jianwei Lv, Tongya Zheng, et al. Breaking the exploration bottleneck: Rubric-scaffolded reinforcement learning for general llm reasoning. arXiv preprint arXiv:2508.16949, 2025.

William F Shen, Xinchi Qiu, Chenxi Whitehouse, Lisa Alazraki, Shashwat Goel, Francesco Barbieri, Timon Willi, Akhil Mathur, and Ilias Leontiadis. Rethinking rubric generation for improving llm judge and reward modeling for open-ended tasks. arXiv preprint arXiv:2602.05125, 2026.

Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. Privileged information distillation for language models. arXiv preprint arXiv:2602.04942, 2026.

Alex Stein, Furong Huang, and Tom Goldstein. Gates: Self-distillation under privileged context with consensus gating. arXiv preprint arXiv:2602.20574, 2026.

Ken Ding. Hdpo: Hybrid distillation policy optimization via privileged self-distillation. arXiv preprint arXiv:2603.23871, 2026.

Yuting Deng, Melanie Brucks, and Olivier Toubia. Examining and addressing barriers to diversity in LLM-generated ideas. arXiv preprint arXiv:2602.20408, 2026.

Chenglei Si, Tatsunori Hashimoto, and Diyi Yang. The ideation-execution gap: Execution outcomes of LLMgenerated versus human research ideas. In The Fourteenth International Conference on Learning Representations, 2026.

## A Prompts for IdeaAnchor Mining

This section provides the full prompts used in the IdeaAnchor mining pipeline described in Section 2.3. The pipeline consists of three prompt-driven stages: (Section A.1) extracting prior works with role assignments and learned insights from a ground-truth paper, (Section A.2) generating a synthesis narrative and reference proposal that reconstructs the reasoning from prior works to the new idea, and (Section A.3) generating checkable evaluation criteria grounded in the prior works and reference proposal.

## A.1 Stage 1: Prior Work Extraction & Role Assignment

Given the full text of a ground-truth paper D, the creator model $\mathcal { M } _ { C }$ extracts the core idea and identifies 5–7 prior works with functional role assignments and learned-insight annotations. The prompt enforces a recency principle: selected papers should be dominated by recent works (within ∼3 years of D), filtering out classic foundational citations that serve as background rather than proximal inspiration. Each candidate paper must pass three tests— counterfactual (would the idea exist without it?), specificity (what concrete element fed into D?), and proximity (is this the most recent version of the idea?).

## Prompt: Prior Work Extraction & Role Assignment

You are an expert AI research analyst specializing in tracing the proximal intellectual lineage of research ideas. Your task is to read a paper and reconstruct the recent reading list that led the authors to their core idea—identifying the specific recent papers they were actively engaging with when the idea took shape.

The Recency Principle. Research ideas are born from a researcher’s recent reading, not from textbook knowledge. When a researcher writes a new paper, the creative process typically looks like this: (1) they read a cluster of recent papers (typically published within the last 1–3 years); (2) they notice gaps, opportunities, or combinable ideas across those recent papers; (3) they synthesize a new contribution that builds on, extends, or recombines elements from that recent cluster. Classic/foundational papers (e.g., the original Transformer, GANs, ResNet) are background knowledge—they are cited for completeness, but they are NOT the proximal triggers of the new idea. Your output should be dominated by recent papers—ideally the majority published within ∼3 years before the current paper.

Step 1: Determine the Paper’s Publication Context. Identify (or estimate) the publication year of the current paper. Use this to calibrate your selection.

Step 2: Extract the Core Idea. Articulate: (a) What is the paper’s core innovation? (b) What existing approach does it improve upon or depart from? (c) What is the key insight that makes this work non-obvious?

Step 3: Trace the Proximal Intellectual Lineage. Think about which recent papers the authors were actively reading and engaging with. For each candidate prior work, apply all three tests:

1. Counterfactual test: If the authors had NOT read this paper, would they still have come up with this specific idea? If yes, do not include it.

2. Specificity test: What SPECIFIC element (technique, insight, finding, limitation) from this prior work fed into the current paper? Must be concrete.

3. Proximity test: Is this the most recent/proximal version of this idea the authors engaged with? Always prefer the newer, more proximal paper.

## DO NOT INCLUDE:

• Classic foundational works that are common knowledge in the field.

• Any paper whose main contribution has been fully absorbed into common practice.

• Generic infrastructure/tools (PyTorch, CUDA, standard libraries).

• Papers that share the domain but don’t directly influence the core idea.

• Papers only mentioned in passing or used only as evaluation baselines.

• “Grandparent” papers—if the current paper builds on Paper A (2023) which builds on Paper B (2018), include A, not B.

## DO INCLUDE:

• The most direct recent predecessor—the specific recent system/method this work directly extends or improves upon (typically within 1–2 years).

• Recent papers whose LIMITATIONS or GAPS the current paper explicitly addresses.

• Recent papers whose specific IDEAS or TECHNIQUES are directly extended, adapted, or combined.

• Papers from potentially different subfields whose recent application of an idea was “borrowed” or adapted here.

## Role Classifications (assign ONE per paper):

1. Direct Predecessor: The most closely related recent prior work—the specific system/method this paper directly extends or improves upon. Typically 1–2 papers.

2. Inspiration Source: A recent paper whose specific idea or technique (possibly from a different context) directly sparked the current paper’s innovation.

3. Gap & Motivation: A recent paper whose explicit limitations, failures, or unsolved challenges directly motivated this research.

4. Methodological Ingredient: A paper that introduced a specific technique or formulation that the current paper directly uses or adapts as a building block. Should be the most proximal source.

Distribution guidance: Aim for roughly 1–2 Direct Predecessors, 2–3 Inspiration Sources, 1–2 Gap & Motivation papers, and 0–1 Methodological Ingredients.

cite\_id Convention. Each prior work must have a cite\_id in the format: firstauthorlastname+year+keyword, all lowercase, no spaces or punctuation (e.g., chen2020simclr, radford2021clip).

Final Recency Check. Before finalizing: (1) Are the majority of selected papers from within ∼3 years? (2) For each paper older than 5 years: is there truly no more recent paper that conveyed the same insight? (3) Am I including any “textbook knowledge” papers? Remove them.

Output (JSON): For each prior work (5–7 papers), provide: role, cite\_id, title, authors, year, and a one-sentence description of what the authors learned and how it informed the current work.

## A.2 Stage 2: Synthesis Narrative & Reference Proposal

Given the extracted prior works (with abstracts, roles, and learned insights), the core idea, and access to the original paper, M generates a synthesis narrative that reconstructs the reasoning from prior works to the new idea, and a structured research proposal (motivation + method). The synthesis narrative serves as the relationship analysis S<sub>D</sub> component of the IdeaAnchor, and the reference proposal provides grounding for criteria generation in Stage 3.

## Prompt: Synthesis Narrative & Reference Proposal

You are an expert research analyst and proposal writer. Your task is to produce TWO things given a set of prior works (with their abstracts), the core idea of a paper, and the original paper content: (1) a synthesis narrative that reconstructs the reasoning process from prior works to the new idea, and (2) a research proposal that presents the motivation and method.

## Part 1: Synthesis Narrative (400–600 words)

Write this narrative in the first person, as the researcher who wrote this paper. You are reconstructing their reasoning process—how they analyzed these prior works, identified gaps and opportunities, and synthesized a new approach. Reference each prior work using its @cite\_id. Only mention papers that appear in the provided prior works list.

Critical rules: This is a structured record of analytical reasoning—how specific observations about prior work led to specific design decisions. Prioritize clarity and depth of reasoning over narrative polish. Use precise, direct analytical language. Avoid dramatic or emotional expressions. The narrative consists of exactly three paragraphs, separated by blank lines. Do NOT include phase labels or headers.

1. Paragraph 1—Analysis of individual prior works (∼100–150 words): Analyze each key prior work individually. For each paper, state: (a) what it concretely achieved or proposed, (b) what technical limitation, gap, or opportunity it exposes, and (c) why that observation matters for the current research direction.

2. Paragraph 2—Cross-paper synthesis (∼200–250 words): This paragraph should contain NO new per-paper analysis— instead, focus entirely on how the observations from Paragraph 1 logically connect to form a new approach. Make the reasoning chain explicit: Which specific idea from one paper directly addresses a limitation identified in another? What new possibility emerges when techniques from different works are combined? Why had this particular combination not been explored before?

3. Paragraph 3—Problem and approach summary (∼200–250 words): Distill the reasoning into two clear points: (1) the research problem—identifying the specific gaps or limitations revealed by prior work, analyzing their underlying causes, and defining the precise research questions; (2) the proposed approach at a high level—what the core idea is, which insights from the above analysis it draws on, and why this design is expected to work.

## Part 2: Research Proposal

The proposal has exactly two sections:

Section 1—Research Motivation & Problem Definition (2–3 paragraphs): Define the specific research problem being addressed. Identify the concrete gap or limitation in existing approaches, citing which prior works (@cite\_id) reveal this gap. Explain why this gap matters and what opportunity it creates.

Section 2—Core Idea & Methodology: An overview paragraph followed immediately by a numbered component list.

• Overview paragraph: A concise high-level description of the proposed approach. State the core idea in 2–3 sentences, then briefly explain why this formulation addresses the identified gap.

• Numbered method components: Break the method into 3–5 key technical components. Each component follows the format: N. [Component Name]: [Description of what this component does, why it is needed, and how it draws from or extends prior work. Reference relevant @cite\_id.]

After the numbered list, add 1–2 sentences explaining why these components work together as a coherent system.

Writing guidelines: Use @cite\_id throughout. Be technically precise. Write in a propositional tone (“We propose. . . ”).   
Keep the proposal to approximately 400–600 words.

Output (JSON): synthesis\_narrative, motivation, method.

## A.3 Stage 3: Checkable Criteria Generation

Given the extracted prior works, synthesis narrative, and reference proposal, M generates checkable criteria U spanning motivation, method, and overall dimensions. The criteria are designed to admit multiple valid proposals— they check whether a proposal meaningfully engages with the gaps and opportunities revealed by the prior works, rather than whether it reproduces the specific method from the reference proposal.

## Prompt: Checkable Criteria Generation

You are an expert in designing evaluation criteria for research proposals. Your task is to generate a set of rubric items that can be used to score the quality of a research proposal generated from a given set of prior works.

Context. You are given: (1) prior works—a set of papers with their roles, cite\_ids, and what was learned from each; (2) a synthesis narrative—a reasoning process showing how insights from the prior works connect; (3) a reference proposal—a ground-truth proposal that serves as ONE valid example but NOT the only valid answer.

The rubrics will be used in reinforcement learning to score model-generated proposals. The model receives the same set of prior works (titles + abstracts) and must produce a research proposal with motivation and method.

## Critical Design Principles:

1. Ground rubrics in the prior works, not in the specific reference proposal. Rubrics should check whether the proposal meaningfully engages with the gaps, limitations, and opportunities revealed by the prior works. They should NOT check whether the proposal reproduces the exact method from the reference proposal.

2. Allow multiple valid solutions. The same set of prior works can inspire different research problems and different methods. Your rubrics should admit multiple high-scoring proposals. A proposal with a completely different method than the reference can still score full marks if it is well-reasoned and grounded.

3. Motivation rubrics can be moderately specific; method rubrics must be high-level. Motivation rubrics may check that the proposal identifies specific gaps that the prior works clearly reveal. Method rubrics should check for general properties (e.g., “addresses the encoding challenge”) rather than specific techniques (e.g., “uses Fourier features”).

4. Each rubric must be independently evaluable as pass/fail. A reward model will check each rubric against the generated proposal. The rubric must be clear enough that pass/fail can be determined from reading the proposal alone.

## Rubric Categories:

Motivation Rubrics (2–3 items): Does the proposal identify the core limitation or gap that the “Gap & Motivation” papers reveal? Does it explain why this gap matters? Does it articulate what opportunity the combination of prior works creates? Method Rubrics (2–3 items): Frame these as “what the method should achieve or address” rather than “what specific technique it should use.” Does the method address the identified gap in a principled way? Does it draw on or extend ideas from the provided prior works? Does it handle specific technical challenges that the prior works surface? Overall Quality Rubrics (1–2 items): Is the proposal internally consistent (motivation leads to method)? Does the proposal present a feasible and complete research plan?

## Writing guidelines for individual rubrics:

• Each criterion is ONE sentence. Be precise but not overlong.

• For motivation rubrics: reference which prior work(s) reveal the gap using @cite\_id.

• For method rubrics: describe what the method should achieve or address, not what specific technique it should use.

• Avoid rubrics that can only be satisfied by one specific approach.

• Make sure the full set of rubrics, taken together, does NOT uniquely determine a single method.

Output (JSON): Generate 5–7 rubric items total, each with: id, category (motivation / method / overall), and criterion.

## B Prompts for Training Data Synthesis

This section provides the full prompts used in the three anchor-driven training paradigms described in Section 3.1.

## B.1 Demonstration Generation Prompt (Shared by SFT and Self-Distillation)

Anchor-guided demonstration (SFT) and anchor-based self-distillation (SSD) share the same generation prompt: both paradigms condition on the input papers P together with the IdeaAnchor A and produce a structured proposal comprising a thinking trace, research motivation, and research plan. The two paradigms differ only in which model

executes the prompt—SFT uses the external teacher M (GPT-5.4-mini), while SSD uses the target model π<sub>θ</sub> (Qwen3-8B) itself—and in the anti-leakage system persona prepended for SSD (see Appendix H).

Prompt: Anchor-Conditioned Demonstration Generation   
System Message:   
You are a research scientist skilled at synthesizing ideas from existing literature into novel research proposals. You are given   
a set of related research papers (with titles and abstracts). Your task is to analyze these papers, identify research gaps and   
opportunities, and propose a novel research idea. When referencing prior works, use its @cite\_id (e.g., @chen2020simclr).   
[Anchor-derived context inserted here — presented directly as rubric criteriafor SFT; dissolved into persona knowledge   
for SSD (see below)]   
Analyze the papers below and write a research proposal with your own original thinking. Structure your response as:   
1. <motivation> and </motivation> tags — explain the research gap   
2. <method> and </method> tags — describe your proposed approach   
Place your actual content directly inside each tag pair.   
User Message:   
# {Paper Title 1} (@cite\_id\_1)   
## Summary: {Abstract 1}   
# {Paper Title 2} (@cite\_id\_2)   
## Summary: {Abstract 2}

For self-distillation, the following system persona is prepended to prevent the model from leaking privileged anchor content into its thinking trace or proposal:

System Persona: Anti-Leakage Constraint (SSD only)

The anchor criteria A<sub>D</sub> are not presented as explicit rubrics to the model. Instead, each criterion is rewritten as a first-person   
observation that the model has “already made” from reading the literature, avoiding all meta-language (e.g., “criteria”,   
“guidance”, “rubric”). The dissolved form is injected into the system prompt as follows:   
From your reading of the recent literature, you have observed several important trends and gaps: {motivation insight 1}.   
Furthermore, {motivation insight 2}. Furthermore, {motivation insight 3}.   
Based on these observations, you believe a promising research direction would involve a method that would {method insight   
1}; {method insight 2}; {method insight 3}.   
Each motivation-category rubric is stripped of prescriptive prefixes (e.g., “The proposal should . . . ”) and verb phrases (e.g.,   
“identify that . . . ”), lowercased, and concatenated with “Furthermore, . . . ” connectors. Each method-category rubric is   
similarly normalized and joined with semicolons. This prevents the model from reciting privileged anchor content verbatim   
during chain-of-thought reasoning.

## B.2 RL Reward Judging Prompt

For anchor-privileged RL, the judge model receives a generated proposal alongside the anchor criteria and scores each checkable item as satisfied or not.

Prompt: RL Reward Judge (Per-Criterion Scoring)   
System Message:   
You are a strict but fair evaluator of research proposals. You need to evaluate if the Proposed Research Plan satisfies the   
provided evaluation rubrics.   
Rules   
1. Evaluate each rubric independently.   
2. Be strict but reasonable — semantic equivalence counts.

3. Focus on substance, not style.   
4. Only score based on what is explicitly stated.   
5. Be skeptical, careful, and come up with valid criticisms. Be as strict as possible, while being unbiased and reasonable.   
6. Note that the plan should not just say it satisfies these desiderata, don’t be fooled by that. Check carefully WHETHER,   
HOW and WHY the proposed plan meets each desiderata for this rubric item one by one.   
Output Format   
Return ONLY a JSON object. For each rubric, give a hit (1) or miss (0) and a brief reason. Then give the total.   
Example with 3 rubrics:   
{   
" rubrics ": [   
{" id ": 1 , " reason ": " Proposal clearly identifies X ." , " hit ": 1} ,   
{" id ": 2 , " reason ": " No mention of Y ." , " hit ": 0} ,   
{" id ": 3 , " reason ": " Method addresses Z ." , " hit ": 1}   
] ,   
" total\_score ": 2,   
" max\_score ": 3   
}   
Do NOT output anything outside the JSON object. No markdown, no explanation before or after.   
User Message:   
Proposal   
{generated proposal text}   
Rubrics   
• Rubric id (category): criterion text   
• Rubric id (category): criterion text   
Evaluate each rubric and return the JSON.

## C Prompts for Evaluation

## C.1 Criteria-Based Evaluation Prompt

For automated evaluation, GPT-5.4 judges whether each criterion in the expert-refined checkable criteria U is satisfied by the generated proposal. Each criterion is marked as satisfied only if the proposal presents a concrete, literature-grounded claim that meets the criterion while maintaining soundness and feasibility.

## Prompt: Criteria-Based Evaluation

You are a rigorous expert peer reviewer evaluating research proposals against specific rubrics. Your evaluation must reward depth of understanding over breadth of keyword coverage.

## Core Principle

A rubric should be marked HIT only when the proposal demonstrates genuine technical understanding of the criterion — not merely surface-level mention of relevant terms. The most common failure mode is a proposal that lists many related techniques or concepts without clearly reasoning about them.

## Rules

1. Evaluate each rubric independently.

2. HIT (1): The proposal clearly and specifically addresses the criterion with concrete technical substance. The proposal must show it understands WHY the criterion matters and HOW it is addressed — not just THAT relevant terms appear.

3. MISS (0): The proposal fails to address the criterion, addresses it only superficially, or mentions relevant keywords without demonstrating understanding of the underlying technical point.

4. Kitchen-sink penalty: If a proposal lists many techniques, frameworks, or citations without clearly explaining each one’s role and connection to the criterion, this is a sign of superficial coverage. Do NOT reward scattershot inclusion of buzzwords or methods. A proposal that names 10 techniques but explains none in depth should score lower than one that deeply explains 3 relevant techniques.

5. Depth reward: If a proposal addresses the criterion through a different but clearly valid technical approach with sound reasoning, mark as HIT even if terminology differs from the rubric. Genuine insight expressed in different words is more valuable than exact keyword matching without understanding.

6. For motivation rubrics: the proposal must show it understands the specific gap or opportunity — not just mention the topic area. A well-reasoned argument identifying the core issue is worth more than a laundry list of related problems. If the proposal clearly explains WHY a gap exists and what its implications are, that demonstrates understanding even if framed differently.

7. For method rubrics: the proposal must describe a concrete, well-justified mechanism. A generic pipeline that happens to list a relevant technique does NOT satisfy the criterion — the proposal must explain how that technique specifically serves the rubric’s requirement. Conversely, a focused method that achieves the rubric’s goal through a different but well-explained approach DOES satisfy it.

8. For overall/coherence rubrics: a focused, well-integrated plan where each component is clearly motivated is superior to an ambitious plan with many loosely connected components. Penalize proposals that stack borrowed techniques without explaining integration. Reward proposals where the stated problem clearly drives the method design, even if the method is simpler.

9. When in doubt, mark as MISS.

## Output Format

Return ONLY a JSON object. For each rubric, give a hit (1) or miss (0) and a brief reason. Then give the total.

{   
" rubrics ": [   
{" id ": 1 , " reason ": " Proposal clearly identifies X ." , " hit ": 1} ,   
{" id ": 2 , " reason ": " No mention of Y ." , " hit ": 0} ,   
{" id ": 3 , " reason ": " Method addresses Z ." , " hit ": 1}   
] ,   
" total\_score ": 2 ,   
" max\_score ": 3   
}   
Do NOT output anything outside the JSON object. No markdown, no explanation before or after.

## D Annotation Protocol and Inter-Annotator Agreement

The evaluation benchmark consists of 924 papers accepted at ICLR 2026, including all 224 oral presentations. For each paper, our mining pipeline generates candidate prior works, role assignments, and checkable criteria, which are then reviewed and refined by expert annotators.

Annotator Recruitment. We recruit annotators who are graduate students or researchers in machine learning with at least two years of research experience. Each annotator has published at least one first-author paper at a top-tier ML venue.

Annotation Procedure. Each instance is independently reviewed by two annotators. Annotators perform three tasks: (i) verify the correctness of extracted prior works and remove irrelevant ones, (ii) adjust role assignments where the automated pipeline misclassifies a paper’s functional role, and (iii) edit criteria items to ensure they are unambiguous, non-trivial, and faithfully reflect the paper’s contribution. Disagreements are resolved through discussion, with a senior researcher adjudicating unresolved cases.

## E Experimental Configuration

Hardware. Experiments are conducted on a 4 × NVIDIA B200 GPUs node.

SFT / Self-Distillation Hyperparameters. Since anchor-guided demonstration (SFT) and anchor-based selfdistillation (SSD) differ only in the source model that generates the training data (see Section B.1), they share the same fine-tuning configuration. Table 3 summarizes the shared hyperparameters.

Inference Configuration. During RL rollout, completions are sampled with temperature 0.7, top-k = 50, top-p = 0.9, and a maximum generation length of 2048 tokens. At evaluation time, greedy decoding is used (temperature 0.0) with a maximum of 64 new tokens for SFT/SSD models.

## F Training and Evaluation Data Statistics

Training Corpus Statistics. Table 5 summarizes the training corpus across the two domains. Each instance contains 5–8 input prior works and 6–7 checkable criteria generated by the mining pipeline. The total corpus contains 14,183

Table 3: SFT / Self-distillation training configuration.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Base model</td><td>Qwen3-8B</td></tr><tr><td>Learning rate</td><td>1 × 10 5</td></tr><tr><td>LR scheduler</td><td>Cosine</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Number of epochs</td><td>5</td></tr><tr><td>Max sequence length</td><td>8192</td></tr><tr><td>Warmup ratio</td><td>0.1</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Precision</td><td>bfloat16</td></tr><tr><td colspan="2">SFT-specific: demonstration source</td></tr><tr><td>Teacher model</td><td>GPT-5.4-mini</td></tr><tr><td colspan="2">SSD-specific: demonstration source</td></tr><tr><td>Teacher model</td><td>Qwen3-8B</td></tr><tr><td>Anti-leakage</td><td>System persona constraint</td></tr></table>

Table 4: RL training configuration.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Base model Learning rate</td><td>Qwen3-8B  $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>LR scheduler Batch size Group size G</td><td>Cosine 512</td></tr><tr><td></td><td>8</td></tr><tr><td>Number of training steps</td><td>150</td></tr><tr><td>Max sequence length KL coefficient β Clip € Format penalty λ</td><td>8192 / 2048 0.0 0.2</td></tr></table>

instances with 96,667 criteria.

Table 5: Training corpus statistics.
<table><tr><td>Domain</td><td></td><td># Papers Avg. Prior Works Avg. Criteria</td><td></td><td>Total Criteria</td><td>Source Venues</td></tr><tr><td>Machine Learning</td><td>7,494</td><td>6.52</td><td>6.86</td><td>51,416</td><td>ICLR, ICML, NeurIPS (2023–2025)</td></tr><tr><td>Natural Science</td><td>6,689</td><td>5.90</td><td>6.76</td><td>45,251</td><td>Nature Comm. (2023–2025)</td></tr><tr><td>Total</td><td>14,183</td><td>6.23</td><td>6.82</td><td>96,667</td><td></td></tr></table>

Evaluation Benchmark Statistics. Table 6 provides detailed statistics for the ICLR 2026 evaluation benchmark.   
Each instance is annotated with 7–8 expert-refined criteria across three dimensions.

Table 6: ICLR 2026 evaluation benchmark statistics.
<table><tr><td colspan="2">Benchmark size</td><td colspan="2">Criteria</td><td colspan="2">Criteria type</td></tr><tr><td>Statistic</td><td>Value</td><td>Statistic</td><td>Value</td><td>Statistic</td><td>Value</td></tr><tr><td>Total instances</td><td>924</td><td>Total checkable criteria</td><td>6,667</td><td>Avg. motivation criteria</td><td>2.99</td></tr><tr><td>Oral presentations</td><td>224</td><td>Avg. criteria per instance</td><td>7.22</td><td>Avg. method criteria</td><td>3.22</td></tr><tr><td>Spotlight / poster</td><td>700</td><td>Instances with 7 criteria</td><td>725</td><td>Avg. overall criteria</td><td>1.00</td></tr><tr><td>Total input papers</td><td>6,108</td><td>Instances with 8 criteria</td><td>199</td><td></td><td></td></tr><tr><td>Avg. prior works per instance</td><td>6.61</td><td></td><td></td><td></td><td></td></tr><tr><td>Range</td><td>4-8</td><td></td><td></td><td></td><td></td></tr></table>

Criteria Distribution across Training Domains. Table 7 compares the distribution of criteria categories across the two training domains. Both domains show a balanced distribution across motivation and method criteria, with exactly one overall criterion per instance on average.

Criteria Count Distribution across Training Domains.

## G Human Evaluation Details

Evaluation Protocol. We conduct blind pairwise human evaluation to complement automatic metrics. For each evaluation instance, annotators receive the same set of input papers P and two proposals generated by different model variants (e.g., Qwen3-8B Base vs. a trained variant). The presentation order of the two proposals is randomized to eliminate position bias.

Evaluation Dimensions. Annotators assess proposals along five dimensions:

• Overall quality: Which proposal presents a stronger, more promising research direction?

Table 7: Per-category criteria distribution across training domains.
<table><tr><td>Domain</td><td>Avg. Motivation</td><td>Avg. Method</td><td>Avg. Overall</td></tr><tr><td>Machine Learning</td><td>2.97</td><td>2.89</td><td>1.01</td></tr><tr><td>Natural Science</td><td>2.71</td><td>3.00</td><td>1.06</td></tr><tr><td>Evaluation (ICLR 2026)</td><td>2.99</td><td>3.22</td><td>1.00</td></tr></table>

Table 8: Criteria count distribution. Number of instances with each rubric count.
<table><tr><td>Criteria count</td><td>ML</td><td>Natural Science</td><td>Evaluation</td></tr><tr><td>6 criteria</td><td>1,042</td><td>1,572</td><td>0</td></tr><tr><td>7 criteria</td><td>6,452</td><td>5,117</td><td>725</td></tr><tr><td>8 criteria</td><td>0</td><td>0</td><td>199</td></tr></table>

• Literature grounding: Which proposal better leverages the input papers and demonstrates deeper understanding of the prior work?

• Synthetic novelty: Which proposal identifies a more creative and non-obvious gap or research direction from the combination of input papers?

• Methodological specificity: Which proposal provides a more concrete and actionable research plan with specific technical details?

• Feasibility: Which proposal outlines a more realistic and executable research agenda?

For each dimension, annotators select one of three options: Proposal A is better, Proposal B is better, or Tie.   
Annotators are encouraged to provide brief justifications for their choices.

## H Self-Distillation Data Leakage Analysis

In anchor-based self-distillation (Section 3.1), the model is prompted with the IdeaAnchor as privileged context during data generation. A key risk is that the model may “leak” privileged information—directly copying anchor content into its thinking trace or proposal—rather than internalizing the synthesis pattern. If such leakage occurs, the resulting training data would contain references to ground-truth answers that are unavailable at inference time, creating a training–inference mismatch.

Anti-Leak Strategies. We compare six strategies that vary in how the anchor is presented and how the model is instructed to use it, against a naive baseline with no anti-leak protection:

• Naive (baseline): The anchor is placed directly in the prompt with no structural tags or anti-leak instructions. The model generates free-form output.

• Persona knowledge: The anchor is reframed as the model’s internalized domain expertise rather than an external reference, discouraging verbatim copying while preserving full semantic access.

• Bottleneck PI: The anchor is compressed into single-keyword dimensions, maximally restricting verbatim content while transmitting abstract directional signals.

• Quality characteristics: The anchor is dissolved into a set of desired output quality attributes, preserving directional information while removing copyable details.

• Critique–edit: A two-phase pipeline where the model first generates independently, then self-critiques against the anchor within a <think> block and revises.

• Uncertainty steering: A three-stage thinking structure (explore → reflect → refine) that channels the anchor through structured uncertainty rather than direct reference.

• Weighted SFT: Only emphasis weights over rubric dimensions are transmitted, without explicit content signal from the anchor.

Leakage Detection Metrics. We measure leakage using: (i) keyword pattern matching for telltale phrases that reference the anchor (e.g., “the rubric says,” “according to the criteria”), (ii) structural tag validation (whether the output correctly uses <motivation>/<method> tags), and (iii) manual inspection of all 50 samples per strategy. An output is marked as leaked if it directly references or quotes anchor content that would be unavailable at inference time.

Results. We evaluate all strategies on 50 samples (10 shared problems × 5 generations each) using Qwen3-8B with vLLM inference. We measure three complementary metrics: leakage rate, rubric compliance (via LLM-as-judge), and ROUGE-L against the ground-truth reference.

Table 9: Comparison of anti-leak strategies for self-distillation. Leakage = fraction of outputs that reference anchor content; Rubric Compliance = LLM-as-judge pass rate over motivation/method/overall criteria; ROUGE-L = overlap with ground-truth reference. <sup>†</sup>False positives: the word “criterion” used in normal academic context.
<table><tr><td>Strategy</td><td>Leakage ↓</td><td>Rubric Compl. ↑</td><td>ROUGE-L ↑</td><td>Rank</td></tr><tr><td>Naive (baseline)</td><td>100% (50/50)</td><td>0.836</td><td>0.147</td><td>5</td></tr><tr><td>Persona knowledge</td><td>0% (0/50)</td><td>1.000</td><td>0.193</td><td>2</td></tr><tr><td>Bottleneck PI</td><td>0% (0/50)</td><td>0.522</td><td>0.164</td><td>7</td></tr><tr><td>Quality characteristics</td><td>4% (2/50)</td><td>1.000</td><td>0.192</td><td>1</td></tr><tr><td>Critique-edit</td><td>4%† (2/50)</td><td>1.000</td><td>0.193</td><td>3</td></tr><tr><td>Uncertainty steering</td><td>4%† (2/50)</td><td>0.955</td><td>0.186</td><td>4</td></tr><tr><td>Weighted SFT</td><td>2% (1/50)</td><td>0.701</td><td>0.156</td><td>6</td></tr></table>

Discussion. Three findings emerge from Table 9:

• No anti-leak protection yields complete leakage. The naive baseline leaks in 100% of outputs. Without structural tags or instructions, the model freely copies and references anchor content, producing outputs that cannot be used as training data due to training–inference mismatch.

• Zero leakage is achievable without sacrificing quality—but not always. Persona knowledge achieves 0% leakage while maintaining perfect rubric compliance (1.000) and the highest ROUGE-L (0.193), demonstrating the ideal safety–quality balance. In contrast, bottleneck PI also achieves 0% leakage but at severe cost to rubric compliance (0.522), because compressing the anchor into single-keyword dimensions strips too much semantic content for the model to produce adequate outputs.

• Remaining leakage is largely spurious. For critique–edit and uncertainty steering, the detected 4% leakage (2/50 each) consists of false positives—the word “criterion” used in normal academic prose, not as a reference to anchor rubrics. After accounting for false positives, five of six strategies achieve effectively 0–2% true leakage, confirming that structured prompting reliably prevents information copying.

Among strategies with near-zero leakage, the quality–leakage frontier is dominated by three approaches: quality\_chars (best ROUGE-1 at 0.543, perfect compliance), persona\_knowledge (best safety profile with 0% leakage and 0 tag errors), and critique\_edit (perfect compliance with self-contained reasoning in <think>). The choice among them depends on whether the practitioner prioritizes absolute zero leakage (persona knowledge), maximal ground-truth similarity (quality characteristics), or interpretable self-critique traces (critique–edit).

## I Additional Judge Evaluation

To verify that our automated evaluation results are not dependent on a specific LLM judge, we repeat the criteria satisfaction rate (CSR) evaluation with alternative judge models.

Table 10: CSR evaluated by different LLM judges. All results on the 924-instance ICLR 2026 benchmark (abstract-only setting).
<table><tr><td>Judge</td><td>Base</td><td>SFT</td><td>SSD</td><td>RL</td></tr><tr><td>GPT-5.4</td><td>10.9</td><td>21.7</td><td>16.3</td><td>24.6</td></tr><tr><td>GPT-5.4-mini</td><td>10.4</td><td>20.9</td><td>15.8</td><td>23.7</td></tr><tr><td>GPT-5.3-chat</td><td>8.7</td><td>18.2</td><td>13.9</td><td>20.4</td></tr></table>

## J Case Studies

We present qualitative comparisons of model outputs given identical input papers. For each case, we show the input prior works, followed by the outputs from Qwen3-8B Base and our best trained variant (SFT). The SFT model produces structured reasoning traces that explicitly synthesize across papers using citation handles, while the base model tends to produce more generic, list-based proposals.

## J.1 Case Study 1: Object-Centric World Models with Latent Actions

## Input: Prior Works P

• DDLP: Unsupervised object-centric video prediction with deep dynamic latent particles (Tal Daniel et al., 2024). We propose a new object-centric video prediction algorithm based on the deep latent particle (DLP) representation. In comparison to existing slot- or patch-based representations, D...

• Unsupervised image representation learning with deep latent particles (Tal Daniel et al., 2022). We propose a new representation of visual data that disentangles object position from appearance. Our method, termed Deep Latent Particles (DLP), decomposes the visual input into l...

• Genie: Generative interactive environments (Jake Bruce et al., 2024). We introduce Genie, the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos. The model can be prompted to generate an endless...

• PlaySlot: Learning Controllable Object-Centric Dynamics with Latent Actions (Victor Villar-Corrales et al., 2025). Predicting future scene representations is a crucial task for enabling robots to understand and interact with the environment. However, most existing methods rely on videos and sim...

• AdaWorld: Learning Adaptable World Models with Latent Actions (Shenyuan Gao et al., 2025). World models aim to learn action-controlled future prediction and have proven essential for the development of intelligent agents. However, most existing world models rely heavily...

• Improving generative imagination in object-centric world models (Zhixuan Lin et al., 2020). The remarkable recent advances in object-centric generative world models raise a few questions. First, while many of the recent achievements are indispensable for making a general...

• Entity-centric reinforcement learning for object manipulation from pixels (Dan Haramati et al., 2024). Manipulating objects is a hallmark of human intelligence, and an important task in domains such as robotics. In principle, Reinforcement Learning (RL) offers a general approach to...

## Output: Qwen3-8B Base

## Thinking trace (excerpt):

Okay, let me try to work through this. The user provided several research papers and wants me to analyze them, find gaps, and propose a novel research idea. Let me start by understanding each paper’s contribution and then see where the gaps might be.

First, the DDLP paper introduces a method for object-centric video prediction using deep dynamic latent particles. They use DLPs which are interpretable and efficient. The other DLP paper from 2022 is about unsupervised image representation learning, decomposing images into particles with spatial and feature info. Then there’s Genie, a generative interactive environment trained unsupervised from internet videos, which can generate virtual worlds. PlaySlot is an object-centric model that uses unlabeled videos to infer object representations an [...]

## Proposal:

Research Gap and Novel Research Idea Gap: Current object-centric video prediction and world modeling approaches excel in representation learning, uncertainty estimation, and action inference but lack a unified framework that integrates interpretability, temporal uncertainty, latent action control, and environmental adaptability. Existing methods often prioritize one aspect (e.g., prediction accuracy or controllability) at the expense of others, limiting their applicability in dynamic, real-world scenarios requiring safe and interpretable planning. For example: - DDLP and DLP provide interpretable object representations but lack robust uncertainty quantification for dynamic interactions. - Genie and PlaySlot enable controllable environments but rely on labeled data or predefined action policies, limiting adaptability. - AdaWorld and G-SWM address adaptability and uncertainty but struggle to integrate object-centric representations with actionable latent actions. - Entity-Centric RL focuses on manipulation but does not explicitly model temporal uncertainty or generalize to novel environments.

Novel Research Idea: Uncertainty-Aware Adaptive Object-Centric World Models (UA-OCWM) Objective: Develop a unified framework that combines interpretable object-centric representations (DLPs), temporal uncertainty estimation (G-SWM), latent action inference (PlaySlot/AdaWorld), and environmental adaptability (AdaWorld) to enable safe, controllable, and interpretable world modeling for dynamic environments.

Key Innovations: 1. Interpretable Uncertainty Estimation: - Integrate DLPs with G-SWM’s multimodal uncertainty framework to quantify uncertainty in object positions, sizes, and interactions during video prediction. This enables “what-if” scenarios (DDLP) while providing confidence bounds for future states.

2. Adaptive Latent Action Control: - Leverage AdaWorld’s self-supervised latent action extraction and PlaySlot’s action-conditioned prediction to infer and control actions in novel environments. This allows the model to adapt to unseen dynamics while maintaining interpretability.

3. Dynamic Object Interaction Modeling: - Extend DLPs to explicitly model spatial-temporal relationships between objects (e.g., occlusions, collisions) using a graph-based particle interaction module, enabling robust prediction in complex scenes.

4. Efficient Transfer Learning: - Pre-train the model on unlabeled videos (as in Genie) and fine-tune it on novel environments with minimal interaction, using uncertainty-aware meta-learning to adapt to heterogeneous actions.

Applications: - Robotics: Safe manipulation tasks where uncertainty estimation ensures robustness to sensor noise and environmental changes. - Virtual Environments: Interactive simulations with interpretable physics and controllable agen

[...]

# Output: Qwen3-8B SFT

## Thinking trace (excerpt):

I began from the observation that object-centric latent dynamics had already been made practical by Deep Latent Particles. @daniel2022dlp showed that a VAE can represent a scene as a set of spatially localized latent particles with position and appearance features, and that this representation supports both reconstruction and downstream control. The limitation for my purpose was that DLPs are primarily a representation, not a full world model, so they do not solve action-conditioned future prediction. @daniel2024ddlp extended DLPs to video prediction, demonstrating that particle trajectories can be modeled autoregressively and used for controllable generation, but it still did not provide a learned latent action space for control. In parallel, @bruce2024genie established that latent action [...]

## Proposal:

## [Motivation]

Current world models face a specific technical tradeoff between object-centric structure and controllability. Object-centric methods identify and track entities, but they do not typically provide a learned latent action space that is both compact and directly usable for control. This limitation is visible in object-centric video prediction systems such as @daniel2024ddlp, which model particle trajectories but do not explicitly infer latent actions for controllable manipulation. It is also reflected in structured visual RL and world modeling work such as @haramati2024entitycentric and @lin2020gswm, which emphasize object identity and temporal imagination, yet still struggle to expose a representation that supports generalizable action-conditioned prediction.

At the same time, action-conditioned world models show that latent actions can be inferred from video dynamics without explicit action labels. @bruce2024genie demonstrates that temporal motion can be converted into a controllable latent action space, and @gao2025adaworld shows that such action-conditioned prediction can be learned from unlabeled video. However, these approaches operate on non-object-centric latent states, so they do not distinguish which entities should be acted upon or how actions should be organized across entities. This matters because manipulation and planning require actions to be attached to specific objects rather than to a monolithic latent state.

The resulting gap is therefore clear: there is no existing framework that combines a compact object-centric latent state with a learned latent action space that is explicitly aligned to those entities. This gap creates an opportunity to unify representation learning and action modeling in a single world model. If the action predictor is constrained to operate in the same particle space used by the world model, then the model can infer compact actions for controllable entities while preserving object identity across time. That formulation would directly address the limitations exposed by @villarcorrales2025playslot, @haramati2024entitycentric, and @lin2020gswm, while retaining the unsupervised action-learning advantages established by @bruce2024genie and @gao2025adaworld.

## [Method]

We propose a world model that represents each scene as a set of deep latent particles and learns a compact latent action space directly in that particle space. The core idea is to let the action predictor infer motion patterns from the current particle state, then condition future particle generation on those actions. This formulation addresses

the identified gap by coupling object-centric state representation with object-centric action prediction, so the model can generate controllable futures without relying on hand-designed action dimensions or dense action annotations.

1. Deep Latent Particle World Model: We represent each frame as a set of spatially localized latent particles with position and appearance featur

[...]

## Analysis

Gap identification. The SFT model identifies a precise technical gap: object-centric representations exist but lack a learned latent action space for control. The base model identifies a vague “unified framework” gap without specifying what is technically missing.

Cross-paper synthesis. The SFT model constructs an explicit logical chain: DLPs provide structure → latent actions from Genie/AdaWorld remove supervision bottleneck → combining both in a shared particle space enables entity-aligned control. The base model lists contributions independently without showing how they connect into a design.

Methodological specificity. The SFT model proposes five concrete components (particle-based dynamics, per-particle action predictor, entity-aligned control, unsupervised action extraction, generalization via particle-aligned actions), each justified by a specific prior work limitation. The base model proposes a generic “Uncertainty-Aware Object-Centric World Model” with a laundry list of applications and potential challenges but no clear technical mechanism.

## J.2 Case Study 2: GFlowNet-Based Formulaic Alpha Factor Mining

## Input: Prior Works P

• Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning (Shuo Yu et al., 2023). In the field of quantitative trading, it is common practice to transform raw historical stock data into indicative signals for the market trend. Such signals are called alpha facto...

• QuantFactor REINFORCE: Mining Steady Formulaic Alpha Factors with Variance-bounded REINFORCE (Junjie Zhao et al., 2024). Alpha factor mining aims to discover investment signals from the historical financial market data, which can be used to predict asset returns and gain excess profits. Powerful deep...

• Flow Network Based Generative Models for Non-Iterative Diverse Candidate Generation (Emmanuel Bengio et al., 2021). This paper is about the problem of learning a stochastic policy for generating an object (like a molecular graph) from a sequence of actions, such that the probability of generatin...

• Trajectory Balance: Improved Credit Assignment in GFlowNets (Nikolay Malkin et al., 2022). Generative flow networks (GFlowNets) are a method for learning a stochastic policy for generating compositional objects, such as graphs or strings, from a given unnormalized densit...

• Let the Flows Tell: Solving Graph Combinatorial Problems with GFlowNets (Dinghuai Zhang et al., 2023). Combinatorial optimization (CO) problems are often NP-hard and thus out of reach for exact algorithms, making them a tempting domain to apply machine learning methods. The highly s...

• Modeling Relational Data with Graph Convolutional Networks (Michael Schlichtkrull et al., 2018). Knowledge graphs enable a wide variety of applications, including question answering and information retrieval. Despite the great effort invested in their creation and maintenance,...

• AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors (Hao Shi et al., 2025). The complexity offinancial data, characterized by its variability and low signal-to-noise ratio, necessitates advanced methods in quantitative investment that prioritize both perf...

## Output: Qwen3-8B Base

## Thinking trace (excerpt):

Okay, let me try to work through this. The user provided several research papers related to alpha factor mining and generative models, and they want me to analyze them, identify gaps, and propose a novel research idea.

First, I need to understand each paper’s contribution. Thefirst paper, Yu2023, talks about using reinforcement learning (RL) to mine synergistic alpha factors by optimizing the downstream combination model’s performance. They use RL to explore the search space of formulaic alphas, which are interpretable and preferred in risksensitive markets. The second paper, Zhao2024, critiques PPO for alpha mining and proposes a modified REINFORCE algorithm with a dedicated baseline and information ratio as a reward. They mention improving

correlation with returns by 3.83% and better [...]

## Proposal:

Novel Research Idea: Synergistic Alpha Generation with Graph-Enhanced GFlowNets and Adaptive Reinforcement Learning

Motivation and Context The existing literature on alpha factor mining and combination (Yu2023, Zhao2024, Shi2025) emphasizes interpretability, dynamic weighting, and reinforcement learning (RL) for optimizing alpha sets. However, critical gaps remain: 1. Limited Diversity in Alpha Generation: While GFlowNets (Bengio2021, Zhang2023) excel at generating diverse solutions, they are not explicitly tailored for financial alphas, which require balancing interpretability and market adaptability. 2. Decoupled Generation and Combination: Most frameworks (Yu2023, Shi2025) treat alpha generation and combination as separate stages, missing opportunities for end-to-end optimization. 3. Neglect of Relational Market Dynamics: Financial markets involve complex relationships (e.g., sector correlations, sentiment networks), which are underexplored in alpha generation. 4. Temporal Adaptability: Current methods (Zhao2024, Shi2025) rely on static or semi-static weighting, failing to dynamically adjust to rapidly shifting market conditions.

Proposed Framework: Graph-Enhanced GFlowNets with Adaptive Reinforcement Learning (GE-GFlowRL) Core Idea: Integrate GFlowNets for diverse alpha generation, graph-based relational modeling (e.g., R-GCNs) to capture market dependencies, and adaptive RL for dynamic combination, creating a unified framework that addresses the above gaps.

Key Components 1. Graph-Enhanced Alpha Generation (GFlowNets + R-GCNs) - GFlowNets: Generate a diverse set of formulaic alphas by modeling the generation process as a flow network, ensuring coverage of high-reward (high-accuracy) and low-reward (diverse) solutions. - R-GCNs: Encode relational data (e.g., market entities, sector relationships, sentiment networks) to inform alpha generation. For example, R-GCNs can model how macroeconomic factors (nodes) influence stock returns (edges), enabling alphas that capture these interactions. - Hybrid Objective: Combine GFlowNet’s flow consistency with R-GCN’s graph-based rewards to prioritize alphas that are both high-performing and interpretable (e.g., leveraging graph edges as feature interactions).

2. Adaptive Reinforcement Learning for Dynamic Combination - State Representation: Use a graph embedding of the current market state (e.g., via R-GCNs) to contextualize alpha combinations. - Policy Optimization: Train a policy (e.g., PPO or modified REINFORCE) to dynamically adjust weights of generated alphas, with rewards based on: - Short-term Performance: Recent portfolio returns. - Risk-adjusted Metrics: Sharpe ratio, Value at Risk (VaR), or information ratio. - Diversity Constraints: Encourage exploration of novel alpha combinations to avoid overfitting to historical patterns.

[...]

## Output: Qwen3-8B SFT

## Thinking trace (excerpt):

I beganfrom the observation that reinforcement-learning-based alpha mining had already established a strong search paradigm, but only for single-factor discovery. @yu2023alphacollections showed that a policy can generateformulaic alphas and optimize them through downstream portfolio returns, which made RL the natural optimization backbonefor alpha generation. However, its objective still depends on evaluating completefactor sequences, so the search remains expensive. @zhao2024quantfactor reinforced this point by showing that even improved RL training objectivesfor alpha mining stillfundamentally rely on trajectory-level reward signals, exposing the inefficiency ofevaluating entirefactor sequences during training. In parallel, @bengio2021gflownet provided a different principle for s [...]

## Proposal:

## [Motivation]

Formulaic alpha factor mining requires discovering sequences of interpretable formulas whose combined portfolio return is maximized. Existing RL-based approaches have shown that this objective can indeed be optimized effectively: @yu2023alphacollections demonstrates that reinforcement learning can generate synergistic alpha sets by optimizing downstream portfolio returns, and @zhao2024quantfactor further shows that improved RL training can make alpha mining more stable. However, both lines of work still evaluate factor sequences only at the terminal stage and optimize return through trajectory-level rewards. This creates a concrete inefficiency: each generated factor sequence must be fully realized before its quality can be assessed, making training substantially more expensive than the trajectory-level evaluation used in standard RL.

A second limitation is that current alpha generators do not adequately model the structural dependencies among factors during generation. @shi2025alphaforge highlights the importance of diversity in factor discovery, but its search procedure still does not explicitly represent the compositional structure of factor sequences in the policy. Meanwhile, @schlichtkrull2018rgcn shows that graph-based encoders can reason over relational structure, suggesting an opportunity to inject structural information into alpha generation. This gap matters because alpha

quality is not determined by isolated formulas; it depends on how formulas interact across time and how the resulting factor sequence composes into a coherent trading signal.   
These observations create a clear opportunity. GFlowNets provide a principled alternative to return maximization by learning a policy whose terminal distribution matches an unnormalized reward distribution, as proposed in @bengio2021gflownet, and trajectory balance provides a stable training objective for long-horizon sequential generation in @malkin2022trajectorybalance. If alpha factor mining can be reformulated as sampling factor sequences whose probability is proportional to their portfolio return, then the search can be guided by a more informative intermediate signal than the final reward alone. The central problem is therefore to design a GFlowNet-based alpha generator that approximates the full return distribution efficiently while incorporating structural information about factor sequences.   
[Method]   
We propose to cast alpha factor mining as a generative flow problem in which the policy learns to sample factor sequences whose terminal distribution is proportional to their portfolio return. Instead of maximizing terminal reward directly, the model approximates the unnormalized reward with a discounted sum of immediate returns, allowing early actions to receive informative reward estimates during training. This formulation addresses the inefficiency of trajectory-level RL while preserving the objective of maximizing portfolio return.   
1. GFlowNet formulation:   
[...]

## Analysis

Gap identification. The SFT model identifies a precise inefficiency: trajectory-level reward evaluation in RL-based alpha mining is expensive because full factor sequences must be realized before quality assessment. The base model identifies a broad “decoupled generation and combination” gap without specifying why this matters computationally. Cross-paper synthesis. The SFT model builds a logical argument: RL alpha mining works but evaluates only at trajectory end → GFlowNets replace return maximization with distribution matching → trajectory balance stabilizes long-horizon generation → R-GCN supplies structural context for intermediate reward estimation. The base model proposes combining GFlowNets + R-GCNs + RL but does not explain the causal logic connecting these choices. Methodological specificity. The SFT model proposes a concrete mechanism: GFlowNet over factor sequences with discounted immediate-return approximation and R-GCN conditioning. Each component has a clear role (what it does, why it is needed, which prior work gap it addresses). The base model enumerates “key components” but presents them as a feature list without a unifying technical mechanism.

## J.3 Case Study 3: Safety-Aware Low-Rank Adaptation

## Input: Prior Works P

• LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2022). An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larg...

• Safety-Aligned LoRA (SaLoRA): Safety-Alignment Preserved Low-Rank Adaptation (Li et al., 2025). As advancements in large language models (LLMs) continue and the demand for personalized models increases, parameter-efficient fine-tuning (PEFT) methods (e.g., LoRA) will become e...

• Safe LoRA: The Silver Lining of Reducing Safety Risks When Finetuning Large Language Models (Hsu et al., 2024). While large language models (LLMs) such as Llama-2 or GPT-4 have shown impressive zero-shot performance, fine-tuning is still necessary to enhance their performance for customized...

• Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! (Qi et al., 2024). Optimizing large language models (LLMs) for downstream use cases often involves the customization of pre-trained LLMs through further fine-tuning. Meta’s open release of Llama mode...

• AlphaEdit: Null-Space Constrained Model Editing for Language Models (Fang et al., 2025). Large language models (LLMs) often exhibit hallucinations due to incorrect or outdated knowledge. Hence, model editing methods have emerged to enable targeted knowledge updates. To...

• PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models (Meng et al., 2024). To parameter-efficiently fine-tune (PEFT) large language models (LLMs), the low-rank adaptation (LoRA) method approximates the model changes ∆W ∈ R<sup>m×n</sup>...

• Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (Aghajanyan et al., 2021). Although pretrained language models can be fine-tuned to produce state-of-the-art results for a very wide range of language understanding tasks, the dynamics of this process are no...

## Output: Qwen3-8B Base

## Thinking trace (excerpt):

Okay, let’s see. The user provided several research papers related to fine-tuning large language models (LLMs) with a focus on parameter-efficient methods like LoRA and its variants. They also mentioned issues around safety alignment and the effectiveness of these methods. The task is to analyze these papers, identify gaps, and propose a novel research idea.

First, I need to understand each paper’s contribution. LoRA is about reducing trainable parameters by using low-rank matrices. Then SaLoRA and SafeLoRA address safety alignment issues duringfine-tuning, showing that even benign data can affect safety. AlphaEdit is about model editing without disrupting existing knowledge. PiSSA improves upon LoRA by using principal componentsforfaster convergence. The intrinsic dimensionality paper [...]

## Proposal:

Novel Research Idea: Intrinsic-Dimension-Guided Safety-Enhanced Low-Rank Adaptation (ID-SaLoRA)

Motivation & Context Recent advancements in parameter-efficient fine-tuning (PEFT) methods like LoRA have enabled efficient adaptation of large language models (LLMs) while preserving most of the pre-trained knowledge. However, safety alignment risks persist, as demonstrated by studies showing that even benign fine-tuning can degrade safety (e.g., Qi et al. 2024, SafeLoRA 2024). Meanwhile, the intrinsic dimensionality of LLMs (Aghajanyan et al. 2021) reveals that only a small subset of parameters is critical for performance, suggesting that safety and efficiency could be co-optimized within this low-dimensional subspace.

Research Gap Current safety-enhanced PEFT methods (e.g., SaLoRA, SafeLoRA) apply constraints globally or across the entire model, potentially sacrificing efficiency or performance. However, the intrinsic dimensionality framework suggests that safety-critical components may reside in a low-dimensional subspace, while other parameters are redundant. This gap highlights the need for a method that: 1. Leverages intrinsic dimensionality to identify safety-critical parameters. 2. Applies safety constraints selectively to this subspace, preserving efficiency. 3. Maintains performance on downstream tasks by avoiding disruption of non-critical parameters.

Proposed Approach ID-SaLoRA combines intrinsic dimensionality analysis with safety-preserving PEFT (e.g., SaLoRA) to achieve efficient and safe adaptation: 1. Intrinsic Subspace Identification: - Use singular value decomposition (SVD) or random projection to identify the low-dimensional subspace (as in PiSSA and Aghajanyan et al.) that captures most of the model’s performance. - This subspace corresponds to the “intrinsic” parameters critical for task performance and safety alignment.

2. Safety-Enhanced Low-Rank Adaptation: - Apply SaLoRA-style safety constraints only to the intrinsic subspace, freezing or regularizing parameters outside this subspace. - This ensures that safety alignment is preserved without over-constraining the model.

3. Efficient Training: - Use low-rank decomposition (as in LoRA) to adapt only the intrinsic subspace, reducing trainable parameters by orders of magnitude. - Combine with quantization (e.g., QPiSSA) to further reduce memory usage.

4. Theoretical Justification: - Leverage the intrinsic dimensionality framework to derive generalization bounds that are independent of full parameter counts, ensuring safety and efficiency are aligned.

Expected Contributions 1. First integration of intrinsic dimensionality with safety-aware PEFT, enabling targeted safety constraints without sacrificing efficiency. 2. Empirical validation on safety benchmarks (e.g., ja [...]

# Output: Qwen3-8B SFT

## Thinking trace (excerpt):

I startedfrom the observation that LoRA already provides a strong structural handle on adaptation: itfreezes the backbone and learns a low-rank update, so any safety intervention can be inserted directly into that update space rather than intofullfine-tuning @hu2022lora. The limitation is that the update space is unconstrained, and prior safety work had already shown that even benign downstream adaptation can degrade alignment @qi2024finetuning. SafeLoRA addressed this by projecting the LoRA update into a safety-aligned subspace using a reference model @hsu2024safelora, but its projection is a hard constraint that can suppress task learning. SaLoRA improved thatformulation by replacing the safety subspace with a null space, which preserves more task capacity @li2025salora, yet it stil [...]

## Proposal:

## [Motivation]

Fine-tuning aligned large language models with LoRA is attractive because it preserves the backbone and keeps adaptation lightweight, but recent evidence shows that this efficiency comes with a safety cost. Even benign downstream adaptation can degrade safety alignment, indicating that the LoRA update space can drift in harmful directions during optimization @qi2024finetuning. Existing safety-aware LoRA methods address this by constraining the update, but they do so through fixed projection operators. SafeLoRA projects the update into a safety-aligned subspace defined by reference models @hsu2024safelora, while SaLoRA uses a null-space formulation to preserve more task capacity @li2025salora. Both methods reveal the same structural limitation: safety is enforced by a static geometric constraint rather than by an estimate derived from the current task and model state.

This gap matters because hard constraints can be too rigid for downstream learning. If the task update already contains safety-preserving information, then forcing it into a fixed null space or a fixed safety subspace can remove exactly the directions needed for performance. The opportunity, suggested by @meng2024pissa, is to treat the LoRA update geometry itself as the object of analysis: if safety-relevant directions can be identified from the update space, then safety preservation can be enforced through a data-dependent constraint rather than a hand-designed one. At the same time, @aghajanyan2021intrinsic suggests that fine-tuning occupies a low-dimensional manifold, making it plausible that safety alignment can also be represented by a compact subspace. The problem we therefore address is: how can we estimate a task-specific safety subspace from the model’s own behavior, and how can we use that estimate to constrain LoRA updates so that safety is preserved without sacrificing downstream utility?

We define the research problem as constructing a safety-aware LoRA method that is adaptive to the downstream task while remaining lightweight and compatible with frozen backbones. The key technical challenge is to estimate a safety-relevant subspace from limited safety-aligned data and to convert that estimate into a trainable update constraint that suppresses harmful directions while leaving task-relevant directions available for learning.

## [Method]

We propose Safety LoRA, a safety-aware fine-tuning framework that estimates a safety subspace from the model’s own safety-aligned outputs and constrains the downstream LoRA update to remain in the null space of that subspace. Instead of projecting the update into a fixed safety-aligned or null space as in prior methods @hsu2024safelora @li2025salora, we estimate the dominant safety directions through SVD and remove only the most influential safety-irrelevant directions. This formulation directly addresses the rigidity of fixed constraints while preserving the low-rank adaptation efficiency introduced by @hu2022lo

## Analysis

Gap identification. The SFT model identifies a specific failure mode: fixed safety projection constraints can over-regularize because they treat safety as a static property rather than a task-conditioned geometric structure. The base model identifies a broader “global vs. selective constraint” gap without pinpointing why existing constraints fail.

Cross-paper synthesis. The SFT model traces a precise design path: LoRA provides the update space → safety degradation persists even under benign adaptation → existing projections (SafeLoRA, SaLoRA) are too rigid → PiSSA shows SVD can identify important directions → AlphaEdit shows null-space projection preserves behaviors → combining these yields data-dependent safety estimation. The base model mentions similar papers but does not construct a causal chain showing why their combination produces a specific new mechanism.

Methodological specificity. The SFT model proposes four tightly coupled steps: estimate safety subspace from model outputs, extract dominant directions via SVD, project LoRA updates into the null space of those directions, and train under this adaptive constraint. The base model proposes “ID-SaLoRA” but describes it at a higher level of abstraction without specifying how intrinsic dimensionality analysis concretely feeds into the safety constraint.