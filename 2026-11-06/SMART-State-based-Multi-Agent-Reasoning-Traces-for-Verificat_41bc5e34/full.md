# SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving

Sai Vallurupalli, Francis Ferraro Department of Computer Science and Electrical Engineering University of Maryland Baltimore County Baltimore, MD 21250 USA {kolli,ferraro}@umbc.edu

## Abstract

Complex problems with limited context, such as claims, require a means to obtain additional knowledge necessary for solution reasoning. We propose a multi-agent framework, SMART, to obtain knowledge through capturing the implicit state of participating entities in a problem context. SMART curates the state knowledge as diverse, complementary and solutionsalient premises using a variety of LLMs and a structured role-based prompt strategy. Using knowledge from these real-world verified curated premises, SMART’s knowledge guided solution reasoning improves validation performance up to 20% on complex scientific claims spanning 3 datasets. We show that grounding premises in real-world knowledge is necessary and improves performance up to 5%.

## 1 Introduction

Consider the problem of assessing the veracity of the following scientific claim,

Genetic deletion of JAM-A decreases gut permeability.

Determining whether this claim is supported, or refuted, requires reasoning about the different entities, concepts, or topics mentioned: JAM-A, and gut permeability and how they interact with one another, such as the impacts of genetic deletion (on JAM-A specifically).<sup>1</sup>

This reasoning is challenging in multiple ways. First, not all relevant entities may be specified in the claim: e.g., with appropriate background knowledge about properties of the (latent, unmentioned) entity intestinal epithelium, we can easily infer that this claim is refuted: (i) an increase in the paracellular transport of molecules across the intestinal epithelium decreases gut permeability, and (ii) genetic deletion of JAM-A increases this paracellular transport (Klingensmith et al., 2022; Fan et al.,

2022). Note that this chain of reasoning not only relied on states or properties of numerous entities, but it also required the introduction of a new entity.

Second, while the vast amount of knowledge that Large Language Models (LLMs) can obtain from pre-training can be impressive (Ajith et al., 2024; Ge et al., 2023; Collins et al., 2024), they are unable to fully leverage the relevant knowledge they possess in their reasoning (Turpin et al., 2023; Jullien et al., 2025). This can get especially challenging when the underlying, possibly niche, knowledge can regularly or rapidly advance, such as in specialized scientific domains. Further, LLMs’ propensity to reason over surface text (Jiang et al., 2024; Hooda et al., 2024; Wang et al., 2024) leads to biased (Suri et al., 2023; Talboy and Fuller, 2023) and hallucinated generations (Mahmoud et al., 2026; Simhi et al., 2025; Li et al., 2024; Rawte et al., 2023) limiting their ability to generate accurate solutions. Any solution must have a way of validating proposed premises of entities states or properties.

To address these challenges, we propose a framework, SMART, based on a structured rolebased prompting method, to obtain salient state knowledge that captures interactions between participating entities as conditional inferences, and uses the knowledge to reason about the solution. Our approach demonstrates how to elicit, verify, and use the state knowledge obtained from participating entities to guide solution-reasoning for the task of claim validation. With SMART, we encourage deeper reasoning using explicit state knowledge obtained from the problem context that is verifiable through external means. In Fig. 1 we outline the process of eliciting and using state knowledge: SMART dynamically identifies entities, concepts, or topics relevant to each claim, and assigns an expert to each entity to obtain deep and connected knowledge between them. To facilitate deeper reasoning, we examine contrastive knowledge: each expert generates contrastive premise pairs that capture opposing views of an entity property, state, constraint, rule, law, etc., in order to critically examine how the state knowledge may (or may not) support the problem. Next, SMART automatically curates these premises, by assessing their validity, identifying their stance alignment to the claim, and selecting a strong premise from each pair that is grounded in established real-world knowledge. Reasoning with these curated premises, SMART assesses the validity of the claim, providing a validation label (refuted, for Fig. 1), a selfrated confidence, and a rationale.

![](images/673aa58cb40822b0844a88d0dc921fb3e6a3469aa7cdc686887875ba9724503c.jpg)  
Figure 1: A complex problem’s implicit state-based reasoning using a scientific claim from SciFact (Wadden et al., 2020). We identify experts knowledgeable about participating entities and explore their state and obtain contrastive premises. We curate true premises of supporting or opposing stance and reason with these to infer the solution.

We examine SMART’s ability to obtain and combine salient domain knowledge to validate complex scientific contexts from 3 challenging datasets: SFBench (Costello et al., 2026), Scifact (Wadden et al., 2020) and Scifact-Open (Wadden et al., 2022). These claims require a variety of domain knowledge and depth of reasoning that SMART can provide. We demonstrate SMART’s usefulness with a variety of small to large, older to newer, open and closed LLMs as agents: Qwen 3 1.7b, 8b & 32b thinking & non-thinking models, Gemma 3 1b, 4b & 12b instruct models, Llama-3.1-Instruct 8b & 70b, GPT-5.4-mini and Claude Haiku 4.5 models. In particular, we find that SMART is especially helpful when the solution depends on knowledge that is not explicit in the problem context and needs to be obtained reliably. It is useful for problems where solution reasoning goes beyond simple verification ofsearched knowledge and requires sifting through the relatedness of available information. It leads to solutions that are based on deeper reasoning from exploring contrastive views of grounded knowledge minimizing models’ generic reasoning when they lack specific problem knowledge.

Across all methods and datasets, on average, SMART obtained state knowledge improves validation performance (BAccuracy) up to 20% over CoT when providing solution supporting information. Our structured prompting allows explicit control over knowledge used for solution reasoning and allows external verification; knowledge verification improves performance up to 19%. Use of contrastive knowledge minimizes generic reasoning seen with single premises and improves up to 6%. Our contributions are:

• we propose a novel framework for verification based solutions employing dynamic context-derived experts for curating verifiable problem-salient knowledge,

• we improve solution reasoning using focused problem specific background knowledge to handle incomplete and uncertain information in complex problems, and

• we show that our structured role-based prompting method is generic using an extensive set of models with multiple datasets.

We will release our code upon publication.

## 2 Related Work

Inspired by methods which obtain consistent solutions through exploring multiple diverse reasoning paths (Wang et al., 2023; Yao et al., 2023a,b), our work explores and reasons with implicit problem state identified as premises. Following techniques that use iterative approaches to improve problem output (Welleck et al., 2023; Madaan et al., 2023; Yao et al., 2023b; Shinn et al., 2023), we obtain premises iteratively and use them to guide solution reasoning. Due to space constraints, we expand on this section in Appendix §10.7.

## 3 Method

Reasoning about a complex problem solution, such as verifying the example claim in Fig. 1, Genetic deletion of JAM-A decreases gut permeability requires knowledge about entities that are not part of the claim context. Chain-of-thought (Wei et al., 2022) with Qwen3 8b model (assigns a wrong label of supported with the rationale: Genetic deletion ofJAM-A has been shown in studies to reduce gut permeability by modulating tight junction integrity. This supports the claim that JAM-A deletion decreases gut permeability. Qwen3 1b & Llama models reason similarly. These models likely consider a specific study from which this could be inferred without examining other latent interactions.

Knowledge about entity states that are salient to the problem outcome can lead to useful background knowledge (Funke, 2010; Vallurupalli and Ferraro, 2025). For example, eliciting the following background knowledge about entities, JAMA-A & gut permeability, such as JAM-A is involved in maintaining gut barrier integrity, JAM-A promotes theformation oftightjunctions in intestinal epithelial cells, etc. is useful for reasoning. Reasoning with these entity states and properties captures inter-relationships between them. However, not all state knowledge is relevant to the outcome and noisy information can lead to an incorrect outcome reasoning (Vallurupalli and Ferraro, 2025). Curating relevant knowledge leads to the correct label of refuted with the rationale: ... genetic deletion increases gut permeability and paracellular transport, both of which are associated with higher permeability. ... provide strong evidence against the claim, leading to a refute label.

Claim decomposition and Retrieval Augmented Generation (RAG) provide other alternatives, however, given the interacting entities needed to assess a claim, these methods also require reasoning beyond a simple lookup to capture distributed knowlege and subclaim relationships (Schlichtkrull et al., 2023; Kamoi et al., 2023).

Eliciting and curating relevant knowledge leads to a reliable solution, but requires deeper reasoning with multiple model calls. A benefit is that the curated knowledge is factually verifiable and produces verification based solutions. For systematically obtaining entity state knowledge and curating it for use in the final reasoning, we propose a multi-stage framework, SMART. We show the framework is generic by applying it on a variety of real-world claims which serve as our complex problems. Below, we explain how we obtain, curate and use the state knowledge from the claims.

Task Description: We validate a context x and generate a solution $y = ( y ^ { l } , y ^ { e } , y ^ { r } )$ consisting of a binary label $y ^ { l }$ , a rating $y ^ { r }$ from 1 to 10 to indicate confidence in the label and an explanation justifying the label and rating. Our proposed framework elicits additional knowledge s relating to entities in x and uses it for the solution reasoning, i.e., y|x, s.

Premise: Entities in x lead us to relevant implicit knowledge which we call a premise.<sup>2</sup> For the example in Fig. 1, the entity JAM-A leads to the premise, JAM-A promotes the formation of tight junctions in intestinal epithelial cells.

## 3.1 SMART Framework:

Details of the various stages shown in Fig. 1 are as follows (with an example in the appendix §10.1):

i. The Identify stage identifies an entity, topic, concept, etc. and assigns experts to these,

ii. In the Explore stage, agent experts obtain premises connecting the selected entities,

iii. The Curate stage selects real-world valid premises based on established knowledge,

iv. The Infer stage reasons with the selected premises to find the solution.

i. Identifying Experts: We identify k salient entities, concepts, topics etc. from x through prompting for entities with a modifiable state. We use k agent as experts where each agent $e _ { i }$ is assigned as an expert on all aspects relating to entity i.

ii. Exploring Problem Domain Knowledge We use all k experts to iteratively generate premise pairs after examining previously obtained pairs for n rounds. With each expert agent, we obtain domain knowledge as contrastive premises, $p _ { k }$ and $p _ { k } ^ { c }$ that capture alternate viewpoints of a property, state, constraint, rule or law relating to a salient entity k. One such premise pair is: JAM-A promotes theformation oftightjunctions .., JAM-A inhibits theformation oftightjunctions ... . These alternate view points allow us to counterfactually reason about which viewpoint is more salient to the problem. By allowing experts to view previously generated premises, we capture fine-grained entity relationships. These premises, $\{ ( p _ { k } , p _ { k } ^ { c } ) | x _  \}$ $( ( p _ { 1 } , p _ { 1 } ^ { c } ) , e _ { 1 } ) , . . . , ( ( p _ { k - 1 } , p _ { k - 1 } ^ { c } ) , e _ { k - 1 } ) \}$ capture the state connecting all the entities in the problem.

iii. Curating Salient Premises From the obtained premise pairs, we select true premises salient to the problem. Premises may not be faithful to the problem context due to LLM hallucinations. Even when faithful, refuted scientific claims can lead to untrustworthy premises that do not reflect accepted scientific knowledge. For these reasons, we verify the binary truth of each premise using two different methods: parametric knowledge and websearch, retrieving up to m documents as evidence. In our running example, both approaches can identify the premise “JAM-A inhibits the formation of tight junctions in intestinal epithelial cells” as untrue, though this is not always the case. Verification with websearch allows latest information from new scientific discoveries to factually verify premises; with parametric knowledge, models self-reflect on their knowledge related to the premise.

For salient premises, we obtain the stance of each premise towards the claim context, where the stance $s ~ = ~ ( s ^ { l } , s ^ { r } , s ^ { e } )$ , consists of a label $s ^ { l } \in \mathsf { \Gamma } ( { ^ \bullet \mathrm { s u p p o r t } } , \mathsf { ^ { \prime } \cdot { ^ \circ p p o s e } } , ^ { \prime }$ ‘unsure’), a numeric confidence rating $s ^ { r }$ ranging from 1 to 10 and an explanation $\boldsymbol { s } _ { k } ^ { e }$ justifying the label and rating. We consider only premises that strongly support or oppose the context and for each we select the premise with a higher confidence rating $s _ { k } ^ { r }$ which indicates how confident we are about the premise’s salience.

<table><tr><td rowspan=1 colspan=1>Prompt Type</td><td rowspan=1 colspan=1>Claim Prompt</td></tr><tr><td rowspan=1 colspan=1>General, Refute Filter</td><td rowspan=1 colspan=1>§10.6.1, §10.6.14</td></tr><tr><td rowspan=1 colspan=1>Direct &amp; Baselines</td><td rowspan=1 colspan=1>§10.6.2, §10.6.3</td></tr><tr><td rowspan=1 colspan=1>Identify Salient Entities</td><td rowspan=1 colspan=1>§10.6.4</td></tr><tr><td rowspan=1 colspan=1>Explore for Premise Pairs</td><td rowspan=1 colspan=1>§10.6.5</td></tr><tr><td rowspan=1 colspan=1>Verify Premise</td><td rowspan=1 colspan=1>§10.6.6,§10.6.7</td></tr><tr><td rowspan=1 colspan=1>Premise&#x27;s Stance</td><td rowspan=1 colspan=1>§10.6.8</td></tr><tr><td rowspan=1 colspan=1>Meta-Reasoners</td><td rowspan=1 colspan=1>§10.6.9, §10.6.10, §10.6.11,§10.6.12, §10.6.13</td></tr></table>

Table 1: The prompts we developed; see the Appendix.

We select the premise JAM-A promotes theformation of tight junctions ... which supports the claim.

iv. Solution Inference with a Meta-Reasoner We obtain the solution y using a Reasoning Judge to meta-reason over all truthful and salient premises after grouping them into supporting and opposing sets, $P _ { s } \ \& \ P _ { o } ,$ based on their stance labels $s ^ { l }$ This allows the premises to be interpreted together; model reasoning coherently connects the premises, inherently assigning different weights to them.

## 4 Implementation Details

In the Identify stage, we obtain $k \leq 5$ entities with a modifiable state that are most important to solving the problem. We use a system prompt to assign an agent as an expert on one of these entities. All prompts are listed in Table 1.

In the Explore stage, to obtain contrastive premises we use few-shot prompting with 3 examples chosen randomly from outcome causal contrastive states in the CoRE dataset (Vallurupalli and Ferraro, 2025). This dataset is based on ROC-Stories (Mostafazadeh et al., 2016) which contain general (non-domain specific) knowledge. These examples demonstrate the general notion of contrasting conditions based on participating entity states; they do not provide domain- or datasetspecific targeting. Since each claim may employ a different number of experts, we obtain $n * k = 1 0$ premise pairs for a fair comparison across models and datasets. We use a prompt guideline to encourage generating unique premise pairs and remove occasional duplicate generations (see §10.3.1).

In the Verify & Select stage, we verify premises using 2 methods: asking a model if the premise is true (PK) and Natural Language Inference with documents retrieved through web-search<sup>3</sup> (WS); see more details in Appendix §10.5. Our claim and stance rating uses a Likert score of 0 – 10. When all premises are untrue or their stance is unsure, we fallback to the direct reasoning prompt.

![](images/fb55106c1d9b1c366a2d5e175a0cfda9aded3845d8437a74d95296eed804b2a6.jpg)  
(a) SciFact Dataset

![](images/e8468818369a336c8d5b122ab8ef8d265061c30fae8f4d1413f787a293df8eee.jpg)  
(b) SFBench Dataset  
Figure 2: Comparison of SMART-PK performance with CoT and RAG on SciFact & SFBench datasets, after applying the Refute Filter (rationales are required to adequately support refute labels; see §5.2). SMART-PK improves over baselines; on avg. 10% on SFBench and 5% on Scifact over CoT (details in Appendix Table 4).

## 5 Methodology

Datasets: We use a total of 577 scientific claims which include the newer more complex SFBench dataset (183 claims) and the older complex SciFact (188 claims) & SciFact-Open (206 claims). We combine the latter two as their collection was a single effort. All claims were written and labeled manually; SciFact-Open, considered to have lowbias, used additional automatic methods to substantiate manual labels. SFBench labels are multi-class with values ranging from -2 to +2; we convert the -ve labels to refute and + labels to support. Labels in the SciFact collection are binary but assigned based on an evidence text; each claim is associated with multiple evidence texts and labels. We use claims with a single representative label. Language in these contexts is highly technical & scientific discourse (except for a few covid-related claims with simpler language). The claims are multi-dimensional and require reasoning that integrates several evidence sources from multiple documents. Evidence—whether correct and useful, or incorrect and misguiding—is abundantly available for these claims on the internet, requiring critical and complex reasoning. See Appendix §10.2 for more claim details.

## 5.1 Models:

We use the following open and closed models of varying sizes, both older & newer, from the Llama, Qwen, OpenAI and Anthropic families: Llama-3.1: 8b & 70b Instruct models, Qwen3: 1.7b, 8b & 32b non-thinking models, OpenAI: GPT-5-mini and Anthropic: Claude Haiku 4.5. We also tried models within the Gemma family (Gemma 3: 1b, 4b & 12b), but we found that all models underperformed across the board. For completeness, we provide more information on this in §10.9.

## 5.2 Baselines & Evaluation:

We compare three SMART settings, SMART-PK (using premises that are verified with parametric knowledge), SMART-WS (using with premises verified with web documents) and X-SMART-PK (a cross model approach where models use premises from a good generation model) with two baselines: Chain-of-Thought (CoT) (Wei et al., 2022) and a RAG using websearch. We compare with these baselines to show that SMART provides good performance along with providing solution supporting information. We compare with RAG and show that a simple websearch leads to noisy evidence with performance far lower than SMART-PK. With SMART-WS we demonstrate an external method for premise validation and show that even noisy websearch results can improve performance.

We report balanced accuracy of support & refute labels. All results in tables and figures are based on a average of 3 runs; we provide average values and standard deviation. We report instances where curation yields no premises (as Fallbacks to direct reasoning) and measure premise correctness as the fraction of curated premises that are supported by gold evidence (we verify if the premise is valid using gold evidence in the prompt §10.6.7).

## 6 Results

For SMART’s knowledge guided reasoning, a model should generate both supporting and refuting premises. More importantly, knowledge provided by the curated premises should adequately support a validation label, i.e., lack of knowledge should not be the reason for a refute label. Likewise, we expect CoT rationales to provide adequate support for refute labels. In this section, we explore various aspects of the SMART framework: premise generation, curation and reasoning, as related to performance on our datasets; we demonstrate its usefulness with several experiments and ablations.

![](images/cd6ba6c77e25fd7980ad42a3e5452e3f343ea49c94e80540a2f36e7e1aaf5fec.jpg)  
(a) SciFact Dataset

![](images/a87523a362052f8d5aa3d19cd949d125c8593520ad6c117a7dfdedfec3bbc0b7.jpg)  
(b) SFBench Dataset  
Figure 3: Improving SMART performance using external premise validation with websearch (SMART-WS) and cross model generation and reasoning (X-SMART-PK). For SciFact, X-SMART-PK uses Qwen3 8b generated premises & improves performance by 6.2%; for SFBench, GPT-5.4-mini generated premises lead to 4% improvement.

Enforcing Logical Reasoning. While a lack of information about some claim aspects can lead to a refute label, we notice that LLMs excessively use lack of supporting information as the rationale to refute claims, especially when models lack claim knowledge (or they are unable to leverage it in reasoning). CoT applies this line of reasoning on 23-78% of SciFact claims whereas SMART applies this on 11-46% (see details in Table 3), leading to solutions that are not logically sound. While RAG reasoning does not have this exact issue, retrieval of unrelated evidence leads a refute label. To ensure models provide more logically sound solutions, we apply a Refute Filter on rationales to verify whether refute labels are adequately supported. Using GPT-5.6-Sol as an LLM judge, we examine rationales using the prompt §10.6.14 for all claims with a gold refute label. We filter rationales which lack evidence to ensure our performance comparisons are fair.

SMART generates label-supporting rationales. We ensure model reasoning is logical with rationales providing adequate support for refute labels. As seen in Fig. 2, SMART’s premise guided reasoning works well across all models and datasets, improving up to 20%; on average 10% on SFBench and 5% on SciFact (averages exclude Llama3 8b due to numerous generation errors).

![](images/f3686dfc2209129912f4eee7a94aac35508304befcd7e820755d11333e3ad076.jpg)  
Figure 4: Comparing SMART-PK performance generating a single premise Vs. contrastive premises on SFBench claims. See Fig. 13 for SciFact claims. Contrastive premises improve performance by 3% across datasets and models. See results in Table 5

Limiting performance inflation. Unsupported refute labels lead to an unsound solution and inflate performance. Smaller models inflate performance more than larger models and SFBench claims inflate performance more than SciFact claims across all methods. Filtering unsupported refute labels reduced this performance inflation; BAccuracy was reduced by 3.5-46.3% for CoT and by 0.6-26.1% for SMART-PK. However, this inflation is smaller for SMART-PK, due to models leveraging premise knowledge in their reasoning (inflation is <3% for SciFact; <4% for SFBench for models larger than 8b). See more details in Appendix Table 4.

Dynamic knowledge improves SMART reasoning. We examine two approaches that add dynamic knowledge to improve SMART: external verification (SMART-WS) and different models for premise generation and reasoning (X-SMART-PK). As seen in Fig. 3, using different models for premise generation and reasoning improve performance by 4% on SFBench (using GPT-5.4-mini generated premises) and 6.2% on SciFact (using

![](images/c6e250c244a1ee64a89fcb612704f81a6408b07b9e77c926f08f5b1a0d3d0f0c.jpg)  
Figure 5: We compare the 5 meta-reasonsers for Sci-Fact claims: M-Majority Voting, R-Reasoning Judge, A-Aggregating Judge, C-Critical Judge and S-Supporting Judge. Reasoning Judge consistently improves performance across models and datasets: by 8% on SFBench and 5% on SciFact on average (See Fig. 9 for SFBench claims and results in Appendix Table 7).

Qwen3 8b generated premises). External verification helps some models but improvement is not consistent due to web-search being noisy. Overall, Llama3.1 70b performance is lower than the smaller Qwen3 32b, due to not generating enough opposing premises and poor curation; smaller models are worse at curation and meta-reasoning. SMART-WS on SciFact claims lead to guardrails errors due to bait-and-switch websites. We describe and classify all SMART errors (see §10.10) and provide examples in Appendix Table 13 & Table 12.

Contrastive premises lead to deeper reasoning. As seen in Fig. 4 and Appendix Fig. 13, all models benefit from deeper reasoning with contrastive premises. Not all models implicitly obtain single premises that explore both sides of a claim (support and refute). Single premises lead to more rationales that use lack of supporting evidence as the reason for a refute label. Applying the Refute Filter, leads to a performance improvement of up to 4% with contrastive premises over using a single premise; Qwen3 1.7b improves with single premises as it is worse at generating contrastive premises.

Collective reasoning improves performance. SMART’s premise curation leads to unique but related premises. As seen in Fig. 5 and Appendix Fig. 9, meta-reasoners, especially SMART’s Reasoning Judge meta-reasoner, improves performance over Majority Voting. Majority Voting, based on independently reasoning with each premise may not capture the collective knowledge in these premises. By interpreting the premises together the Reasoning Judge improves performance up to 8% on both datasets. See results in Appendix Table 7.

![](images/b94e519c180b743501fe8d72b748fa0b320066c6144801841e442759f4aff767.jpg)  
Type SMART−NV SMART−PK SMART−WS X−SMART−PK

Figure 6: Comparing SMART-NV where no verification is performed) with verification using self (SMART-PK), external knowledge with websearch (SMART-WS), and cross-model (X-SMART-PK) on SciFact claims. Verification improves performance an average by 2-7% across methods for both datasets, with a few exceptions (see SFBench in Fig. 10). See results in appendix Table 8.

![](images/4926a9ce1e595a0ee0269c51365151981b86be909d291471ca252fdc26e90f41.jpg)  
Figure 7: Distribution of GPT-5.4-mini curated stances, their alignment with the claim and their confidence ratings for the SciFact-Open dataset. Supporting and opposing premises are required for supporting claims that of the support and refute type respectively.

Verification is necessary. As seen in Fig. 6 and Appendix Fig. 10 premise verification improves performance up to 19% with self-verification using parametric knowledge (SMART-PK), external verification with websearch (SMART-WS) and cross-model verification (X-SMART-PK); averaging across models 2-7% per method with a few exceptions. See results in Appendix Table 8).

## 6.1 Premise Quality

Improved performance of X-SMART-PK is due to two factors relating to curated premises: salience and the coverage for a given claim context. Salience measures correctness of the premise for the claim whereas coverage addresses the completeness of solution knowledge in all premises.

![](images/9cd8ea6b9b04bf53ca81aed22c2fe2e2c8034a0f062aae8b6dbbf230e11df959.jpg)  
Figure 8: We compare the validation performance of SMART-PK with 3 sizes of Qwen and GPT-5.4-mini on SciFact-Open claims using premises generated and curated by one model and solution reasoning with another model. These comparisons show that both Qwen3 8b and GPT-5.4-mini generate good quality premises that other models can curate and use for solution-reasoning.

Measuring Premise Salience: We measure the correctness of curated premises as the fraction of premises supported by the gold evidence (per claim average), the percentage of claims where curation does not lead to any premises and the number of supporting and opposing premises in Appendix Table 9. While these measures provide some general markers, a better indicator for the correctness (and to some extent completeness) of curated premises is the performance; we conclude that premises used in SMART-PK are salient to the claim contexts. Quality of SMART-WS suffers due to not finding salient evidence to curate premises.

Orthogonally we examine correctly identified supported and refuted claims. As seen in Fig. 7 the stance of GPT-5.4-mini curated premises aligns with the claim type. Not generating a premise of the correct stance or not curating it (due to an inability to verify its truth or stance) can lead to missed premises that are required for a correct solution.

Premise Generation Vs Reasoning: SMART depends on both a model’s ability to generate both supporting and opposing premises and the ability to curate salient premises through reasoning. We demonstrate that better reasoning can overcome poor premises using 3 sizes of Qwen3 and GPT-5-mini on the SciFact-Open dataset (we use this dataset as labels are considered less-biased). As seen in Fig. 8 all models are worse at reasoning with their own curated premises and better at reasoning with premises curated by Qwen3 8b and GPT-5.4-mini. We leverage this finding for the X-SMART-PK setting. See premise comparisons in Appendix Fig. 14 and examples in Table 10.

## 6.2 Ablations

We examine choices we made in premise generation and curation: a) The number of entities was limited to 5 because across all datasets and models the average number of entities is ≈ 3.5. b) Obtaining 10 premise pairs was to use experts more than once for related premises. c) Curated premises require a meta-reasoner (Abl. 1). d) Using a stronger premise from each pair reduces conflicting knowledge (Abl. 2). e) premise generation involves performance vs. cost decisions for multiple contextderived experts vs. a single and iterative vs. noniterative (Abl. 3: Context Priming).

Abl. 1 - MetaReasoner: As seen in Fig. 5 and Appendix Fig. 9 the Reasoning Judge consistently performs the best among 4 other meta-reasoners described in §10.8. See results in Appendix Table 7.

Abl. 2 - Premise Selection: Logically, SMART’s use of a stronger premise to avoid information conflicts is sound, however, models are able to reason with both premises as seen in Appendix Fig. 11.

Abl. 3 - Context Priming: Appendix Fig. 12 compares performance of various premise generation configurations (see Appendix Fig. 15) with the SciFact collection. A simple low-resource method is to sample from top-p premise generations (shown as Non-Iter) using a single static (“science”) expert; this increases knowledge redundancy (useful for smaller models) but possible for all models: each API call to the Claude-Haiku-4.5 model provides a single generation and GPT-5.4-mini limits them to 8 (with these models multiple premises can be generated at once, which is not effective for other models). Iterative generation captures fine-grained, related and unique premises, especially, with contextderived experts, improving performance by 2%.

## 7 Conclusions

We propose a framework, SMART, to curate solution salient knowledge using a structured prompting method with several LLMs. Our multi-stage setup leads to a natural decomposition of a complex problem and allows multiple agents to reason over simpler problem aspects. Using a variety of LLMs on 3 complex claim datasets, we demonstrate that knowledge grounded in real-world evidence can guide reasoning and improve performance. We show external verification with a noisy websearch and leave improvement of search results for future.

## 8 Limitations

We acknowledge the pre-trained large language models in our experiments can echo biases and mis-information either implicitly or explicitly. Our work does not alleviate or resolve any of these biases knowingly.

While we believe our work does not introduce any additional biases, it is likely that our framework may explicate previously hidden implicit biases. Our premise generation may draw out implicit biases related to the claims and models are likely to hallucinate premises introducing new biases. Our premise verification is likely to suppress these biases especially when an external source such as the web search is used for verifying premises, unless the bias is present in the source documents as well.

We believe that a solution should be based on the knowledge at the time of examining the problem. However, websearch results are not reproducible unless retrieval snapshots or queries are cached. Critical domains will require employing such caching of trusted evidence or maintian domain-specific documents for reproducibility. Results can also change with slight alterations in prompts, however, as long as we are able to obtain and curate premises we can identify what information led to a particular solution.

## 9 Ethical Considerations

This work analyzes how Large Language Models (LLMs) can be used used for problem-solving. We acknowledge that LLM outputs can reflect biases from pretraining data and we do not make normative claims about models’ solutions. We use publicly available datasets and papers for this work and these datasets did not contain any personally identifiable information. Additionally, our work does not involve human subjects and does not contain any personally identifiable information. While our work attempts to control for brittleness in models solution reasoning, it may still produce confident yet ungrounded solutions for incomplete or underdefined problems.

## References

Anirudh Ajith, Mengzhou Xia, Alexis Chevalier, Tanya Goyal, Danqi Chen, and Tianyu Gao. 2024. Lit-Search: A retrieval benchmark for scientific literature search. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,

pages 15068–15083, Miami, Florida, USA. Association for Computational Linguistics.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michał Podstawski, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. 2024. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(16):17682–17690.

Katherine M. Collins, Albert Q. Jiang, Simon Frieder, Lionel Wong, Miri Zilka, Umang Bhatt, Thomas Lukasiewicz, Yuhuai Wu, Joshua B. Tenenbaum, William Hart, Timothy Gowers, Wenda Li, Adrian Weller, and Mateja Jamnik. 2024. Evaluating language models for mathematics through interactions. Proceedings of the National Academy of Sciences, 121(24):e2318124121.

Cash Costello, James Mayfield, Elsbeth Turcan, Christine Piatko, Christina K. Pikas, Justin Rokisky, Sam Scheck, Chris Ribaudo, Ritwik Bose, and Alex Memory. 2026. Sfbench: The scify scientific feasibility benchmark.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning.

Shuling Fan, Kevin Boerner, Muraleedharan Chithra, Asma Nusrat, Miguel Quiros, and Charles Parkos. 2022. Epithelial jam-a is fundamental for intestinal wound repair in vivo. Virtual. JCI Insight.

Joachim Funke. 2010. Complex problem solving: a case for complex cognition? Cognitive Processing, 11(2):133–142.

Yingqiang Ge, Wenyue Hua, Kai Mei, jianchao ji, Juntao Tan, Shuyuan Xu, Zelong Li, and Yongfeng Zhang. 2023. OpenAGI: When LLM meets domain experts. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Sayontan Ghosh, Mahnaz Koupaee, Isabella Chen, Francis Ferraro, Nathanael Chambers, and Niranjan Balasubramanian. 2023. PASTA: A dataset for modeling PArticipant STAtes in narratives. Transactions ofthe Association for Computational Linguistics, 11:1283– 1300.

WenYuan Gu, JiaLe Han, HaoWen Wang, Xiang Li, and Bo Cheng. 2025. Explain-analyze-generate: A sequential multi-agent collaboration method for complex reasoning. In Proceedings of the 31st International Conference on Computational Linguistics, pages 7127–7140, Abu Dhabi, UAE. Association for Computational Linguistics.

Ashish Hooda, Mihai Christodorescu, Miltos Allamanis, Aaron Wilson, Kassem Fawaz, and Somesh Jha. 2024.

Do large code models understand programming concepts? counterfactual analysis for code predicates. In International Conference on Machine Learning.

Bowen Jiang, Yangxinyu Xie, Zhuoqun Hao, Xiaomeng Wang, Tanwi Mallick, Weijie J Su, Camillo Jose Taylor, and Dan Roth. 2024. A peek into token bias: Large language models are not yet genuine reasoners. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4722–4756, Miami, Florida, USA. Association for Computational Linguistics.

Mael Jullien, Marco Valentino, and André Freitas. 2025. The knowledge-reasoning dissociation: Fundamental limitations of llms in clinical natural language inference. ArXiv, abs/2508.10777.

Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and Greg Durrett. 2023. WiCE: Real-world entailment for claims in Wikipedia. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 7561–7583, Singapore. Association for Computational Linguistics.

Nathan Klingensmith, Katherine Fay, David Swift, Julia Bazzano, John Lyons, Ching-wen Chen, Mei Meng, Kimberly Ramonell, Zhe Liang, Eileen Burd, Charles Parkos, Mandy Ford, and Craig Coopersmith. 2022. Junctional adhesion molecule-a deletion increases phagocytosis and improves survival in a murine model of sepsis. Virtual. JCI Insight.

Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2024. The dawn after the dark: An empirical study on factuality hallucination in large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 10879–10899, Bangkok, Thailand. Association for Computational Linguistics.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. 2023. Making language models better reasoners with step-aware verifier. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5315–5333, Toronto, Canada. Association for Computational Linguistics.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17889–17904, Miami, Florida, USA. Association for Computational Linguistics.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Thirty-seventh

Conference on Neural Information Processing Systems.

Omar Mahmoud, Ali Khalil, Thommen George Karimpanal, Buddhika Laknath Semage, and Santu Rana. 2026. The unintended trade-off of AI alignment: Balancing hallucination mitigation and safety in LLMs. In Findings of the Association for Computational Linguistics: EACL 2026, pages 1017–1037, Rabat, Morocco. Association for Computational Linguistics.

Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016. A corpus and cloze evaluation for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 839–849, San Diego, California. Association for Computational Linguistics.

Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar, S.M Towhidul Islam Tonmoy, Aman Chadha, Amit Sheth, and Amitava Das. 2023. The troubling emergence of hallucination in large language models - an extensive definition, quantification, and prescriptive remediations. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 2541–2573, Singapore. Association for Computational Linguistics.

Michael Schlichtkrull, Zhijiang Guo, and Andreas Vlachos. 2023. Averitec: A dataset for real-world claim verification with evidence from the web. In Advances in Neural Information Processing Systems, volume 36, pages 65128–65167. Curran Associates, Inc.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems.

Adi Simhi, Itay Itzhak, Fazl Barez, Gabriel Stanovsky, and Yonatan Belinkov. 2025. Trust me, I’m wrong: LLMs hallucinate with certainty despite knowing the answer. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 14665– 14688, Suzhou, China. Association for Computational Linguistics.

Gaurav Suri, Lily R. Slater, Ali Ziaee, and Morgan Nguyen. 2023. Do large language models show decision heuristics similar to humans? a case study using gpt-3.5.

Alaina N. Talboy and Elizabeth Fuller. 2023. Challenging the appearance of machine intelligence: Cognitive bias in llms and best practices for adoption.

Niket Tandon, Bhavana Dalvi, Keisuke Sakaguchi, Peter Clark, and Antoine Bosselut. 2019. WIQA: A dataset for “what if...” reasoning over procedural text.

In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6076– 6085, Hong Kong, China. Association for Computational Linguistics.

Niket Tandon, Keisuke Sakaguchi, Bhavana Dalvi, Dheeraj Rajagopal, Peter Clark, Michal Guerquin, Kyle Richardson, and Eduard Hovy. 2020. A dataset for tracking entities in open domain procedural text. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6408–6417, Online. Association for Computational Linguistics.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. 2023. Language models don't always say what they think: Unfaithful explanations in chain-ofthought prompting. In Advances in Neural Information Processing Systems, volume 36, pages 74952– 74965. Curran Associates, Inc.

Sai P Vallurupalli and Francis Ferraro. 2025. CoRE: Condition-based reasoning for identifying outcome variance in complex events. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 19381–19401, Vienna, Austria. Association for Computational Linguistics.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534–7550, Online. Association for Computational Linguistics.

David Wadden, Kyle Lo, Bailey Kuehl, Arman Cohan, Iz Beltagy, Lucy Lu Wang, and Hannaneh Hajishirzi. 2022. SciFact-open: Towards open-domain scientific claim verification. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 4719–4734, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Mengru Wang, Yunzhi Yao, Ziwen Xu, Shuofei Qiao, Shumin Deng, Peng Wang, Xiang Chen, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2024. Knowledge mechanisms in large language models: A survey and perspective. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7097–7135, Miami, Florida, USA. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. 2022.

Chain of thought prompting elicits reasoning in large language models. CoRR, abs/2201.11903.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. 2023. Generating sequences by learning to self-correct. In The Eleventh International Conference on Learning Representations.

Kai Xiong, Xiao Ding, Yixin Cao, Ting Liu, and Bing Qin. 2023. Examining inter-consistency of large language models collaboration: An in-depth analysis via debate. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 7572–7590, Singapore. Association for Computational Linguistics.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, volume 36, pages 11809–11822. Curran Associates, Inc.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023b. React: Synergizing reasoning and acting in language models. International Conference on Learning Representations (ICLR).

Wenting Zhao, Mor Geva, Bill Yuchen Lin, Michihiro Yasunaga, Aman Madaan, and Tao Yu. 2023. Complex reasoning in natural language. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts), pages 11–20, Toronto, Canada. Association for Computational Linguistics.

## 10 Appendix

No AI assistants were used for writing either the paper or any of the code written for producing the work in this paper. All writing is original and produced by the authors.

## 10.1 SMART Validation with an Example:

## A Scifact-open claim example:

Claim: CENP-B inhibits heterochromatin formation Gold Label: False

GPT-5-mini’s direct baseline label and reasoning: True. CENP-B binds alphoid DNA and promotes assembly/maintenance of CENP-A-containing centromeric chromatin and de novo centromere formation; experimental work shows loss of CENP-B impairs CENP-A assembly and can allow heterochromatin features to encroach on centromeric regions, consistent with CENP-B acting to antagonize or inhibit heterochromatin formation at centromeres.

SMART (using GPT-5.4-mini as agents) generated final label and reasoning: False. All provided opposing conditions show that CENP-B-containing centromeres bear canonical heterochromatin features (high H3K9me3, low H3K9 demethylase, low transcription, low

H3K9ac), which contradicts the claim; published data also link CENP-B with heterochromatin/HP1 recruitment rather than inhibition, so the claim is not supported.

SMART Framework Stages for the above claim: For the above final label and reasoning, SMART obtained contrastive premise pairs using GPT-5-mini expert agents. and curated them in the 4 framework stages.

Identify (Stage 1): The §10.6.4 identified the following 5 entities with modifiable states: CENP-B, histone H3K9 methylation, heterochromatin formation, HP1 recruitment to chromatin, centromeric heterochromatin levels

Explore (Stage 2): 5 agents are assigned to be experts, each assigned to be an expert one of the 5 entities obtained above, and prompted to generate the following contrastive premise pairs iteratively using §10.6.5. (We use an ampersand (&) to separate the two premises in a pair.)

1. Presence of CENP-B at centromeres is associated with low levels of the heterochromatin mark H3K9me3 & Presence of CENP-B at centromeres is associated with high levels of the heterochromatin mark H3K9me3

2. HP1 protein levels at centromeric chromatin are low & HP1 protein levels at centromeric chromatin are high

3. Centromeric chromatin is enriched in the histone variant CENP-A. & Centromeric chromatin is depleted of the histone variant CENP-A.

4. CENP-B physically interacts with the histone methyltransferase SUV39H1 & CENP-B does not physically interact with the histone methyltransferase SUV39H1

5. Centromeric DNA contains CENP-B box sequence motifs. & Centromeric DNA lacks CENP-B box sequence motifs.

6. Centromeric chromatin has high levels of the active chromatin mark H3K4me2. & Centromeric chromatin has low levels of the active chromatin mark H3K4me2.

7. Centromeric chromatin has high levels of H3K9-specific demethylase localized at centromeres. & Centromeric chromatin has low levels of H3K9-specific demethylase localized at centromeres.

8. Centromeric transcription levels are high. & Centromeric transcription levels are low.

9. Centromeric chromatin has high levels of H3K9 acetylation (H3K9ac). & Centromeric chromatin has low levels of H3K9 acetylation (H3K9ac).

10. CENP-B is DNA-binding competent & CENP-B is DNA-binding deficient (to be continued in the next post)

Verify & Select (Stage 3): After verifying the truth value (using §10.6.7 and identifying the stance of premises (using §10.6.8), the following supporting and opposing premises (each of these oppose the claim being “valid”) were curated for use in the final reasoning step. No supporting premises were left as the model was more confident about an opposing premise from each pair and assigned it a higher stance rating.

## Supporting premises: none

## Opposing premises:

1. Presence of CENP-B at centromeres is associated with high levels of the heterochromatin mark H3K9me3 Stance Explanation: If CENP-B presence at centromeres correlates with high levels of the heterochromatin mark H3K9me3, that contradicts the claim that CENP-B inhibits heterochromatin formation; however, association does not prove causation and contextual factors could confound the relationship, so confidence is high but not absolute.

2. Centromeric chromatin has low levels of H3K9-specific demethylase localized at centromeres. Stance Explanation: Low levels of H3K9-specific demethylase at centromeres would favor persistence of H3K9 methylation and heterochromatin formation, which contradicts the claim that CENP-B inhibits heterochromatin; although the condition does not mention CENP-B directly, it is inconsistent with an inhibition role, hence opposition with moderately high confidence.

3. Centromeric transcription levels are low. Stance Explanation: Low centromeric transcription is indicative of a heterochromatic, transcriptionally repressed state; this observation contradicts the claim that CENP-B inhibits heterochromatin formation because inhibition of heterochromatin would be expected to raise transcription. While other factors could cause low transcription independent of CENP-B, the condition primarily opposes the claim.

4. Centromeric chromatin has low levels of H3K9 acetylation (H3K9ac). Stance Explanation: Low H3K9 acetylation is a hallmark of heterochromatin (acetylation is associated with active/euchromatin), so centromeric chromatin showing low H3K9ac indicates heterochromatin is present at centromeres and thus contradicts the claim that CENP-B inhibits heterochromatin formation; however, other factors could modulate local chromatin so confidence is not absolute.

Infer (Stage 4): Using all the supporting and opposing premises the reasoning judge generated the above provided correct label and reasoning using §10.6.11.

## 10.2 Dataset Details

SFBench<sup>4</sup> (Costello et al., 2026) is a collection of novel material science claims manually created by subject matter specialists with 5 label categories using a Likert score from -2 to 2. We use all claims with a positive score as ‘Supported’ claims and claims with a negative score as ‘Refuted’ claims leading to 183 claims. We do not always have easy to use textual evidence for these claims, as a result we do not have gold evidence for these claims and we do not evaluate the quality of premises for this dataset.

## SFBench Data Example:

Claim: Lithium manganese iron phosphate can be coated on a carbon cloth current collector at a loading of 3 mAh/cm2 and a formulation of 94/3/3 active/carbon/PVDF binder and flexed repeatedly without cracking or delamination prior to calendering.   
Likert Score: -1   
Label: Contradict

SciFact<sup>5</sup> (Wadden et al., 2020) & SciFact-Open<sup>6</sup> (Wadden et al., 2022): SciFact is a collection of 1409 manual re-writes of citation sen-

<sup>4</sup>https://github.com/JHUAPL/SFBench <sup>5</sup>https:// github.com/allenai/scifact <sup>6</sup>https://github.com/dwadden/scifact-open tences sampled from well-regarded scientific journals covering basic sciences to clinical medicine. The dataset consists of a dev datasplit (referred to as SciFact) and a test datasplit (referred to as SciFact-Open). In SciFact, claim labels were manually identified after verifying each claim against the abstract of a cited article. Since the labels are based on a single evidence document which is derived from a small corpus of 5K documents there can be label bias. We use the claims where labels are either ‘support’ or ‘contradict’ for a total of 188 claims. We use the abstract on which the label is based as gold evidence for evaluating our premises. To reduce label bias, SciFact-Open obtained more evidence documents, a corpus of 50K documents, using a BM25 search and automatically generated labels from SOTA models using all matched evidence documents. Correlating these automatically generated labels with manually identified labels from the smaller corpus label bias is reduced. A label from each matched evidence document can lead to several labels for a claim which we consolidate to obtain 206 claims that are exclusively supported or refuted (we drop 73 claims that are both supported and refuted). We concatenate all the evidence abstracts for our gold evidence which use for evaluating the quality of premises.

SciFact Data Example:   
Claim: Pseudogene PTENP1 regulates the expression of   
PTEN by functioning as an miRNA decoy.   
Label: Support   
Evidence: A coding-independent function of gene and   
pseudogene mRNAs regulates tumour biology The canoni  
cal role of messenger RNA (mRNA) is to deliver protein  
coding information to sites of protein synthesis. ...

![](images/a3624765cc612a46e5eb17b40d256571efd2a865187f14443d4cfe1cdd1b0239.jpg)

## 10.3 Implementation Details & Prompts

In all the prompts we use ‘condition’ when refering to a ‘premise.’ It was easier to use the term ‘premise’ in the paper after we obtained the results.

## 10.3.1 Prompt Guidelines & Hyperparameters

## 10.4 Infrastructure

We used both RTX 8000 with 48GB of GPU memory and Nvidia A100 with 80GB GPU memory for inference with the open models. For the open models we use OpenAI’s API endpoint and for Claude we use LiteLLM. Inference time for each claim depends on the hardware used but ranges from 1-5 minutes with Qwen32b and Llama70b taking the longest time.

## 10.5 Websearch

We use LangChain’s implementation of Duck-DuckGo Search for obtaining search documents. We use each premise as the query to obtain relevant documents without any date restriction. With use the text from the top 3 documents along with the url (we do not scrape data from the url) as searched evidence. Using this information as evidence, we verify the premise’s truth. When no results are returned are returned, the claim the premise cannot be verified and is not used towards final reasoning. When the results are not relevant the premise is likely to be identified as false and will not be used in the final reasoning. When none of the premises are verified to be true, the final solution falls back to the direct reasoning baseline. See Table 9 for an average number of fall-backs (averaged across the 6 datasets). We don not cache the evidence across claims or across the 3 runs since its main purpose is to verify a condition before selecting it in final reasoning.

## 10.6 Hyperparameters & Premise Diversity

For simplicity, we use the same settings for all the models and datasets. We use nucleus sampling to get the top 10% of generations. These are closely related to the claim from which we obtain multiple sufficiently diverse conditions through repeated sampling. We provide a list of guidelines to follow as part of each prompt, such as, not generating duplicate premise pairs, considering the specificity and credibility of any numbers or percentages, assessing any biases and limitations, etc (see §10.6.1). With the Llama models, we encourage premise diversity using a temperature of 0.9, but for all other reasoning we use 0.6 a recommended value for Llama models. We use the default temperature for Qwen3, GPT-5.4-mini and Claude models.

```handlebars
Claim:
<claim>
{{claim}}
</claim>
Guidelines:
<guidelines>
1. Identify entities from the claim context and consider
```

## 10.6.1 Common Guidelines and Output

## Common Guidelines

- Provide a label decision of Y if the claim is supported and   
N if it is not supported.   
- Provide a rating between 0 and 10 to indicate your con  
fidence in the label decision, with 10 indicating a high   
confidence and 0 indicating no confidence.   
- Provide a clear reasoning for why the provided label deci  
sion and rating are correct.   
- Do not include any additional commentary or keys.

## Prompt Output Format

```xml
<example_format>
"decision": "Y, or N to indicate whether you think the
claim is supported or not.",
"rating": "Your rating indicating the confidence in the
above label decision on a scale from 0 to 10, 10 indicating
high confidence and 0 indicating no confidence.",
"reasoning": "Your brief explanation in one or two sen
tences for your reasoning on the provided label decision
and rating."
</example_format>
```

## 10.6.2 Direct Prompt

Direct Reasoning Prompt   
User Prompt: You are an AI agent designed to assess   
whether a scientific claim is supported by current scientific   
knowledge. The claim is given in the <claim></claim>   
tags. You must think and reason carefully, and provide   
output according to a certain format following the   
guidelines in the <guidelines></guidelines> tags.   
Claim:   
<claim>   
{{claim context}}   
</claim>   
Guidelines:   
<guidelines>   
1. Output a JSON object as shown in <exam  
ple\_format></example\_format> with exactly three keys:   
“decision,” “rating” and “reasoning”.   
{{common\_guidelines}}   
</guidelines>   
{{example\_format}}   
Now, please evaluate the claim above.

## 10.6.3 CoT and RAG Prompts

## CoT Reasoning Prompt

```handlebars
User Prompt: You are an AI agent designed to assess
whether a scientific claim is supported by current scientific
knowledge. The claim is given in the <claim></claim>
tags. You must think step by step and reason carefully, and
provide output according to a certain format following the
guidelines in the <guidelines></guidelines> tags.
Claim:
<claim>
{{claim context}}
```

</claim>   
Guidelines:   
<guidelines>   
1. Output a JSON object as shown in <exam  
ple\_format></example\_format> with exactly three keys:   
“decision,” “rating” and “reasoning”.   
{{common\_guidelines}}   
</guidelines>   
{{example\_format}}   
Now, please evaluate the claim above.

## RAG Reasoning Prompt

```handlebars
User Prompt: You are an AI agent designed to assess
whether a scientific claim is supported by the given
evidence. The claim is given in the <claim></claim> tags
and evidence for it is given in the <evidence></evidence>
tags. You must think and reason carefully with the
evidence, and provide output according to a certain format
following the guidelines in the <guidelines></guidelines>
tags.
Claim:
<claim>
{{claim context}}
</claim>
Evidence:
<evidence>
{{search evidence}}
</evidence>
Guidelines:
<guidelines>
Output a JSON object as shown in <exam
ple_format></example_format> with exactly three keys:
“decision,” “rating” and “reasoning”.
{{common_guidelines}}
</guidelines>
{{example_format}}
Now, please evaluate the claim above.
```

## 10.6.4 Identifying Salient Entities

User Prompt: You are given a claim in the <claim></claim> tags. Your job is to examine the claim and identify salient entities whose state is modifiable. If there are more than 5 such entities select the 5 most important entities whose state is most salient to the claim context. Then, generate these entities separated by a semicolon. You should also follow the guidelines provided in the <guidelines></guidelines> tags.

4. Make sure the condition pair is salient to the claim   
context.   
5. Identify conditions that are related to the   
previously identified conditions listed in <identi  
fied\_conditions></identified\_conditions>.   
6. Do not repeat any of the conditions listed in <identi  
fied\_conditions></identified\_conditions>.   
7. The conditions should be stative expressions. Do not   
describe activities.   
8. Output your response strictly as a JSON object as   
shown in <example\_format></example\_format> with   
exactly two keys: "condition" and "counterfactual". Do   
not include any additional commentary or keys.   
</guidelines>   
Example output format:   
<example\_format>   
"condition": "A property, state, attribute, constraint, rule   
or law salient to the claim context.",   
"counterfactual": "A property, state, attribute, constraint,   
rule or law contrastive to the condition."   
}   
</example\_format>   
Now, please generate a condition pair for the above claim.

whether their state is modifiable.   
2. Select upto 5 entities whose state is most salient to the   
claim context.   
3. You are allowed to use your internal scientific and   
linguistic knowledge.   
4. Do not list numbers or units by themselves without any   
associated modifiable state or entity.   
5. Do not number them or include any additional   
commentary or keys.   
6. Do not generate any other commentary or preamble,   
generate only the list of entities.   
</guidelines>   
Now, please generate a list of entities.

## 10.6.5 Claim Explore Prompt

User Prompt: You are given a claim in the <claim></claim> tags. You are an expert with knowledge of properties, states, constraints, rules and laws relating to {{entity}} . With your expert knowledge of {{entity}} , your job is to examine the claim and identify a pair of contrastive reasons or conditions, such as entity properties, states, constraints, rules and laws, that are salient to the claim context and its underlying state. These conditions should be such that they examine an aspect of the claim to identify if the claim is supported by current scientific knowledge. You should identify contrastive condition pairs that were not previously identified which are listed in the <identified\_conditions></identified\_conditions> tags. When you are identifying the first condition pair, the <identified\_conditions></identified\_conditions> tags will be empty. Each pair previously identified is given in the <contrastive\_pair></contrastive\_pair> tags and consists of an expert name in the <expert></expert> tags and the two conditions identified by the expert in the <condition1></condition1> tags and the <condition2></condition2> tags. Pay attention to which expert identified the conditions in a pair to know their viewpoint on the aspect they identified in the conditions. You should provide your output according to a certain format following the provided guidelines in the <guidelines></guidelines> tags.

## 10.6.6 Verify with Parametric Knowledge

<table><tr><td>above label decision on a scale from 0 to 10, 10 indicating high confidence and 0 indicating no confidence.&quot;, &quot;reasoning&quot;: &quot;Your brief explanation in one or two sentences for your reasoning on the provided label decision and rating.&#x27; 1 &lt;/example_format&gt;</td><td>sentences for your reasoning on the provided label decision and rating.&quot; 1 &lt;/example_format&gt; Now, please evaluate the condition pair above.</td></tr><tr><td>Now, please evaluate the condition pair above. 10.6.7 Verify with External Evidence</td><td>10.6.8 Obtaining a Premise&#x27;s Stance System Prompt: You are an expert at reasoning about conditions that support or oppose claims related to medical</td></tr><tr><td>This uses websearched documents as evidence. System Prompt: You are an expert with knowledge of properties, states, constraints, rules and laws and can</td><td>and science topics. User Prompt: You are given a claim in the &lt;claim&gt;&lt;/claim&gt; tags and a condition &lt;condi-</td></tr><tr><td>identify if a given property, state, constraint, rule or law is true. User Prompt: You are given a condition in the</td><td>tion&gt;&lt;/condition&gt; tags. Your job is to analyze the claim and the condition and decide whether the condition supports or opposes the claim using your general scientific knowledge. You should also follow the provided guidelines in the &lt;guidelines&gt;&lt;/guidelines&gt; tags.</td></tr><tr><td>&lt;condition&gt;&lt;/condition&gt; tags and some evidence in the &lt;evidence&gt;&lt;/evidence&gt; tags. The condition describes a property, state, constraint, rule or law governing an entity. Your job is to identify whether the condition is true based</td><td>Claim: &lt;claim&gt;</td></tr><tr><td>on the evidence. You should also follow the provided guidelines in the &lt;guidelines&gt;&lt;/guidelines&gt; tags.</td><td>{{claim}} &lt;/claim&gt;</td></tr><tr><td></td><td>Condition: &lt;condition&gt;</td></tr><tr><td>Condition: &lt;condition&gt;</td><td>{{condition}} &lt;/condition&gt;</td></tr><tr><td>{{condition}} &lt;/condition&gt;</td><td></td></tr><tr><td>Evidence:</td><td>Guidelines: &lt;guidelines&gt;</td></tr><tr><td>&lt;evidence&gt;</td><td>1. Analyze the context and scope of the claim. 2. Consider the specificity and credibility of any numbers</td></tr><tr><td>{{evidence}} &lt;/evidence&gt;</td><td>or percentages.</td></tr><tr><td></td><td>3. Assess any potential biases or limitations.</td></tr><tr><td></td><td>4. Evaluate using your general scientific knowledge whether the condition supports or opposes the claim.</td></tr><tr><td>Guidelines: &lt;guidelines&gt;</td><td>5. It is likely the condition both supports and opposes the</td></tr><tr><td>1. Analyze whether the evidence provides sufficient</td><td>claim on different aspects. For such a condition use the</td></tr><tr><td>context and scope for verifying whether the condition is</td><td>unsure label. 6. It is likely the condition is unrelated to the claim or</td></tr><tr><td>true.</td><td>related to the claim but neutral towards it. For such a</td></tr><tr><td>2. Evaluate whether the condition is true based on the</td><td>condition use the unsure label.</td></tr><tr><td>provided evidence and your scientific knowledge. 3. Output your response strictly as a JSON object as</td><td>7. Output your response strictly as a JSON object as</td></tr><tr><td>shown in &lt;example_format&gt;&lt;/example_format&gt; with</td><td>shown in &lt;example_format&gt;&lt;/example_format&gt; with exactly three keys: “decision,&quot; “rating&quot; and “reasoning&quot;.</td></tr><tr><td>exactly three keys: “decision,&quot; “rating&quot; and “reasoning&quot;.</td><td>8. Provide a label decision of Y if the condition supports</td></tr><tr><td>4. Provide a label decision of Y if the condition is true</td><td>the claim, N if the condition opposes the claim, and U for</td></tr><tr><td>and N if it is known to be untrue or it is not likely to be true based on the provided evidence and your scientific</td><td>all other situations. 9. Provide a rating between 0 and 10 to indicate your</td></tr><tr><td>knowledge.</td><td>confidence in the label decision, with 10 indicating a high</td></tr><tr><td>5. Provide a rating between 0 and 10 to indicate your confidence in the label decision, with 10 indicating a high</td><td>confidence and 0 indicating no confidence. 10. Provide a clear reasoning for why the provided label</td></tr><tr><td>confidence and 0 indicating no confidence. 6. Provide a clear reasoning for why the provided label</td><td>decision and rating are correct. 11. Do not include any additional commentary or keys.</td></tr><tr><td>decision and rating are correct.</td><td>&lt;/guidelines&gt;</td></tr><tr><td>7. Do not include any additional commentary or keys. &lt;/guidelines&gt;</td><td>Example output format:</td></tr><tr><td></td><td>&lt;example_format&gt;</td></tr><tr><td>Example output format:</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>&lt;example_format&gt;</td><td></td></tr><tr><td></td><td>&quot;decision&quot;: &quot;Y or N or U to indicate the condition&#x27;s</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>support for the claim.&quot;,</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>&quot;decision&quot;: &quot;Y or N to indicate if the condition is true.&quot;,</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>&quot;rating&quot;: &quot;Your rating indicating the confidence in the</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>&quot;rating&quot;: &quot;Your rating indicating the confidence in the</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>above label decision on a scale from 0 to 10, 10 indicating</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>above label decision on a scale from 0 to 10, 10 indicating</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>high confidence and 0 indicating no confidence.&quot;,</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>high confidence and 0 indicating no confidence.&quot;,</td><td>&quot;reasoning&quot;: &quot;Your brief explanation in one or two</td></tr></table>

decision and rating."   
}   
</example\_format>   
Now, please evaluate the claim and condition pair   
above.

## 10.6.9 Claim Aggregate Explanations Prompt

User Prompt: You are given a claim in the <claim></claim> tags and multiple conditions that support the claim in the <supporting\_conditions></supporting\_conditions> and multiple conditions that oppose the claim in the <opposing\_conditions></opposing\_conditions> tags. Each condition in the <condition></condition> tags is verified for its truth value and whether it supports or opposes the claim feasibility in the <explanation></explanation> tags. Summarize the justifications provided in all the <explanation></explanation> tags. The <supporting\_conditions></supporting\_conditions> tags or the <opposing\_conditions></opposing\_conditions> tags may be empty and in such a case state that no conditions were provided.

Claim: <claim> {{claim}} </claim>

Claim Opposing Conditions:   
<opposing\_conditions>   
{{<condition></condition><explanation></explanation>}} {{<condition></condition><explanation></explanation>}} {{...}}   
</opposing\_conditions>   
Now, please summarize the conditions above.

## 10.6.10 Claim Aggregate Judge Prompt

System Prompt: You are an expert at critically examining and deciding whether claims about medical and science topics are true.

User Prompt: You are a scientific analyst designed to assess whether a scientific claim is supported by current scientific knowledge. The claim is given in the <claim></claim> tags. You must think and reason logically using the additional information provided as supporting and opposing conditions to aid in your reasoning. Conditions that support the claim are given in the <supporting\_conditions></supporting\_conditions> tags and conditions that oppose the claim are given in the <opposing\_conditions></opposing\_conditions> tags. Each of the conditions is given in the <condition></condition> tags and their justification for supporting or opposing the claim is given in the <explanation></explanation> tags. Use these conditions to aid in your reasoning about whether the claim is supported. Provide your output according to a certain format following the guidelines in the <guidelines></guidelines> tags.

Claim: <claim> {{claim}} </claim>

Claim Supporting Conditions:   
<supporting\_conditions>   
{{<condition></condition><explanation></explanation>}}   
{{<condition></condition><explanation></explanation>}}   
{{...}}   
</supporting\_conditions>   
Claim Opposing Conditions:   
<opposing\_conditions>   
{{<condition></condition><explanation></explanation>}}   
{{<condition></condition><explanation></explanation>}}   
{{...}}   
</opposing\_conditions>

4. Examine all the supporting and opposing conditions and their explanations. Use these to aid your scientific knowledge and reason logically about whether the claim is supported.

## 10.6.11 Claim Reasoning Judge Prompt

System Prompt: You are an expert at critically examining and deciding whether claims about medical and science topics are true.

User Prompt: You are a scientific analyst designed to assess whether a scientific claim is supported by current scientific knowledge. The claim is given in the <claim><claim> tags. You must think and reason logically using the additional information provided as supporting and opposing conditions to aid in your reasoning. Conditions that support the claim are given in the <supporting\_conditions></supporting\_conditions> tags and conditions that oppose the claim are given in the <opposing\_conditions></opposing\_conditions> tags. Each of the conditions is given in the <condition></condition> tags. Use these conditions to aid in your reasoning about whether the claim is supported. Provide your output according to a certain format following the guidelines in the <guidelines></guidelines> tags.

Claim: <claim> {{claim}}

</claim>   
Claim Supporting Conditions:   
<supporting\_conditions>   
{{<condition></condition>}}   
{{<condition></condition>}}   
{{...}}   
</supporting\_conditions>   
Claim Opposing Conditions:   
<opposing\_conditions>   
{{<condition></condition>}}   
{{<condition></condition>}}   
{{...}}   
</opposing\_conditions>   
Guidelines:   
<guidelines>   
1. Analyze the claim using your scientific knowledge.   
2. Consider the specificity and credibility of any numbers   
or percentages.   
3. Assess any potential biases or limitations.   
4. Examine all the supporting and opposing conditions   
and use these to aid your scientific knowledge and reason   
logically about whether the claim is supported.   
5. Output a JSON object as shown in <exam  
ple\_format></example\_format> with exactly three keys:   
“decision,” “rating” and “reasoning”.   
{{common\_guidelines}}   
</guidelines>   
{{example\_format}}   
Now, please evaluate the claim above.

## 10.6.12 Claim Critical Judge Prompt

System Prompt: You are an expert at critically examining   
and deciding whether claims about medical and science   
topics are true.   
User Prompt: You are a scientific analyst de  
signed to assess whether a scientific claim is supported   
by current scientific knowledge. The claim is given   
in the <claim></claim> tags. You must think and   
reason logically using the additional information   
provided as opposing conditions to aid in your reasoning.   
Conditions that oppose the claim are given in the <op  
posing\_conditions></opposing\_conditions> tags where   
each condition is given in the <condition></condition>   
tags. Use these conditions to play a devil’s advocate and   
interrogate your knowledge relating to the claim. Provide   
your output according to a certain format following the   
guidelines in the <guidelines></guidelines> tags.   
Claim:   
<claim>   
{{claim}}   
</claim>   
Claim Opposing Conditions:   
<opposing\_conditions>   
{{<condition></condition>}}   
{{<condition></condition>}}   
{{...}}   
</opposing\_conditions>   
Guidelines:   
<guidelines>

1. Analyze the claim using your scientific knowledge.   
2. Consider the specificity and credibility of any numbers   
or percentages.   
3. Assess any potential biases or limitations.   
4. Examine all the opposing conditions and use them to   
critically reason about whether the claim is supported   
taking the stance of a devil’s advocate.   
5. Output a JSON object as shown in <exam  
ple\_format></example\_format> with exactly three keys:   
“decision,” “rating” and “reasoning”.   
{{common\_guidelines}}   
</guidelines>   
{{example\_format}}   
Now, please evaluate the claim above.

## 10.6.13 Claim Supportive Judge Prompt

System Prompt: You are an expert at critically examining   
and deciding whether claims about medical and science   
topics are true.   
User Prompt: You are a scientific analyst de  
signed to assess whether a scientific claim is supported   
by current scientific knowledge. The claim is given in   
the <claim></claim> tags. You must think and reason   
logically using the additional information provided   
as supporting conditions to aid in your reasoning.   
Conditions that support the claim are given in the <sup  
porting\_conditions></supporting\_conditions> tags where   
each condition is given in the <condition></condition>   
tags. Use these conditions to examine for contradictory   
information that prevents these these conditions from   
supporting the claim. Provide your output according   
to a certain format following the guidelines in the   
<guidelines></guidelines> tags.   
Claim:   
<claim>   
{{claim}}   
</claim>   
Claim Supporting Conditions:   
<supporting\_conditions>   
{{<condition></condition>}}   
{<condition></condition>}}   
{{...}}   
</supporting\_conditions>   
Guidelines:   
<guidelines>   
1. Analyze the claim using your scientific knowledge.   
2. Consider the specificity and credibility of any numbers   
or percentages.   
3. Assess any potential biases or limitations.   
4. Examine all the supporting conditions and use them to   
critically reason about whether the claim is supported.   
5. Output a JSON object as shown in <exam  
ple\_format></example\_format> with exactly three keys:   
“decision,” “rating” and “reasoning”.   
{{common\_guidelines}}   
</guidelines>   
{{example\_format}}   
Now, please evaluate the claim above.

## 10.6.14 Refute Filter

System Prompt: You are an expert at evaluating   
rationales.   
User Prompt: You are asked to assess whether   
the rationale below provides specific details for why the   
scientific claim is refuted or if it uses lack of evidence for   
supporting the claim as the reason to refute it. Think and   
reason carefully and provide your response strictly as a   
JSON object as shown in the output.   
Claim: {{claim}}   
Rationale: {{rationale}}   
Output:   
{   
"label": "provide Y if the rationale contains specific   
details, provide N if the rationale uses lack of evidence for   
supporting the claim as the reason to refute it",   
"rationale": "Provide a clear reasoning for the label in one   
or two sentences"   
}   
Now, please assess the rationale.

## 10.7 Related Work (expanded from main paper section §2 )

Below, we outline research related to the ideas we use in our work.

Implicit and Counterfactual States Causal relationships between implicit states and events have been previously studied using common everyday events (Tandon et al., 2020, 2019; Ghosh et al., 2023; Vallurupalli and Ferraro, 2025). Leveraging their idea of contrastive states, we obtain contrastive premises capturing implicit knowledge in the problem context.

Language Models & Complex Reasoning Complex reasoning requires models to go beyond exploiting surface level data patterns and reason with knowledge embedded in the context. This requires a model to connect distinct pieces of information and fill-in missing gaps. Zhao et al. (2023)’s tutorial covers a variety of methods and past works that examined various aspects of complex reasoning. We use some of these methods such as leveraging external knowledge to validate conditions and few-shot exemplars to obtain contrastive states.

Iterative and Diverse Reasoning Techniques such as Self-Correction (Welleck et al., 2023), Self-Refine (Madaan et al., 2023), ReAct (Yao et al., 2023b) and Reflexion (Shinn et al., 2023) use an iterative approach to improve problem output through incorporating feedback from previous reasoning steps. These methods use a iterative approach to post-hoc correct the solution while we obtain solution-salient knowledge prior to the solution. Techniques such as DiVeRSe (Li et al., 2023), Tree-of-thought (Yao et al., 2023a), Graphof-thought (Besta et al., 2024) examine diverse reasoning paths selecting optimal intermediate solutions. Our reasoning approach selects optimal premises in lieu of intermediate solutions. Multi agent debate frameworks leverage iterative and diverse reasoning paths (Du et al., 2024; Liang et al., 2024; Xiong et al., 2023), however, Gu et al. (2025) showed that instead of a debate multiple agents can also be leveraged to handle sub-problems. We take inspiration from the latter and decompose obtaining a problem’s implicit state into sub-problems.

## 10.8 Solution Inference with other Meta-Reasoners

We compare with 4 other meta-reasoners: majority voting, an aggregating judge, a critical judge and a supportive judge.

The Majority Voting meta-reasoner is arguably the simplest one: each stance label contributes equally to the solution and the solution is based on the higher average stance rating $s ^ { r }$ of premises in the two sets: $P _ { s }$ or $P _ { o }$ . The claim label $y ^ { l }$ is ’support’ if the average stance rating of premises in $P _ { s }$ is higher, else it is ’refute.’ The confidence rating for this label $y ^ { r }$ is equal to the average $s ^ { r }$ of all premises in the set that determined the label with $y ^ { e }$ summarizing all premises in $P _ { s } \ \& \ P _ { o }$

The Aggregating Judge, like the reasoning judge, selects y conditioned on all supporting and opposing premises. It additionally uses all the stance explanations, i.e., $\{ ( y | x , ( p _ { k } , s _ { k } ^ { e } ) ~ \in$ $( P _ { s } , P _ { o } ) \}$ , to provide additional context which can be useful to connect the premises especially when small changes in wording determine the claim’s veracity. However, these justifications can bias the final inference, as they are obtained through isolated reasoning with each premise.

The Critical Judge reasons only with the opposing premises playing the role of “the devil’s advocate,” critically considering whether each premise can discredit the claim. The solution is conditioned on opposing premises only, i.e., $\{ y | x , p _ { k } \in P _ { o } ) \}$

The Supportive Judge reasons with only the supporting premises, considering whether they can support the claim. The solution is conditioned on supporting premises only, i.e., $\{ y | x , p _ { k } \in P _ { s } ) \}$ contrasting the reasoning of the Critical Judge.

<table><tr><td>Model</td><td>CoT</td><td>SMART-PK</td><td>SMART-WS</td></tr><tr><td>Gemma 3 1b Gemma 3 4b</td><td> $\overline { { . 5 6 3 \pm . 0 0 3 } }$   $. 6 2 2 { \pm } . 0 0 1$ </td><td> $. \overline { { 5 6 2 \pm . 0 0 5 } }$   $. 5 9 4 \pm . 0 0 7$ </td><td> $. \overline { { 5 6 2 \pm . 0 0 5 } }$   $. 6 0 9 { \pm } . 0 1 1$ </td></tr></table>

Table 2: Gemma Models do not make for good SMART agents for two reasons: a) CoT performance is boosted by refuting claims for which there is no supporting evidence. b) Models are good at generating supporting premises but poor at generating contrastive premises that can lead to refuting evidence. This results in SMART meta-reasoning being overly positive with not having opposing reasons for a claim. As a result, we do not use these models in examining other data besides Sci-Fact. This behavior is also seen to a lesser extent with Llama3.1 70b model.

![](images/eb279c3752d396f8f9ef106398735476f2844deb7d06db59704358d0083d3593.jpg)  
Figure 9: We compare the 5 meta-reasonsers for SFBench claims: M-Majority Voting, R-Reasoning Judge, A-Aggregating Judge, C-Critical Judge and S-Supporting Judge. Reasoning Judge metareasoner is more consistent across models and datasets and improves performance, by 8% on SFBench and 5% on SciFact (See Fig. 5 for SciFact and results in Appendix Table 8).

## 10.9 Issues with the Gemma models

We found that Gemma family models underperformed comparably sized models from Llama and Qwen3. This was mainly because these model were unable to generate a mix of both opposing and supporting premises. Qualitatively, we noticed that Gemma models curate mostly supporting premises which improves the accuracy of valid claims but lowers the accuracy of invalid claims, leading to an overall poorer performance than the CoT baseline. In addition, CoT performance is boosted as lack of supporting evidence as these models use the lack of evidence to refute claims as seen in Table 2.

## 10.10 Error Types:

Examining errors from both claim sources, we find that incomplete model knowledge relating to a claim leads to errors. Some errors are silent where the model selects the correct label, either due to model positivity or inadequate evidence supporting a refute label. We filter the latter with the Refute Filter but filtering the former positivity errors is not simple as these rationales contain claim knowledge, although, the knowledge is incomplete.) We roughly classify these errors into the following types:

![](images/f9cf20df363f1bf3c27b3e8f88e24459ced82c2b70edbfec621c7da39b7c97a3.jpg)  
Type SMART−NV SMART−PK SMART−WS X−SMART−PK

Figure 10: Comparing SMART-PK, where premises are verified with parametric knowledge, with SMART-NV where no verification is performed on SFBench claims, we show that verification improves performance an average of 2% with a few exceptions (also true for SciFact claims in Fig. 6). Results are in appendix Table 7.  
![](images/e806f22ad3baf8fbe9bab10b1e0e36b38bb8ea85d5a51836a782bc7086fae996.jpg)  
Figure 11: Comparing SMART-PK performance using a stronger premise Vs. both premises. Logically using a stronger premise was a good decision (performance of both both GPT-5.4-mini and Claude-Haiku-4.5 reduces due to direct conflicts), however, model reasoning is able to use the additional nuanced information provided by both premises. Larger models are better at leveraging the nuanced information in both premises (Llama70b improves by 1.3% & Qwen3 32b by 6.7%); smaller models benefit from additional premises as they do not curate as many premises (Qwen3 1b improves by 1.5%).

a) When models do not have complete knowledge, CoT ignores unknown aspects and supports the claim (Model Positivity). SMART explores these unknowns with the premise knowledge and either leads to the correct refuted label [Type 1 (these situations are not errors)], or supports the claim with the same model positivity due to incomplete premise knowledge [Type 3]. There are a few Type 1 and a few Type 3 errors across all models.

![](images/7f53053c718e63f9a38160de3103b4a14f2f3dd8e91530722126b13d1cd9550f.jpg)  
Figure 12: We compare various premise generation methods using the SciFact collection. We obtain premises with a single expert in one-shot (Non-Iter) Vs an iterative generation strategy (Iter). We also compare SMART’s multiple context-derived experts with the iterative strategy. Claude-Haiku-4.5 does not allow obtaining more than one generation per API call; due to this we do not have any result for the single expert in one-shot (Non-Iter) strategy.

![](images/9ef2153ddb68a55a8435f6ab823a2737fcc79c3b214523480cd1b75cce8fc48b.jpg)  
Figure 13: Comparing SMART-PK performance generating a single premise Vs. contrastive premises on SciFact claims. See Fig. 4 for SFBench claims. Contrastive premises improve performance by 3% across datasets and models. See results in Table 5

b) Models have little to no claim related knowledge, yet, CoT correctly identifies with a refute label, with the rationale citing lack of support for the claim. [Type 2b]. Meanwhile, SMART reasoning either correctly reasons with the premise knowledge, or leads to the incorrect label when the knowledge provided by the premises is incomplete (a lack of opposing conditions either due to a model invalidating them from lack of knowledge or a model not generating them as with Llama3.1 70b are the main reasons for this situation) [Type 2a]. When lacking curated premises, SMART’s Fallback to direct-reasoning leads to the same Type 2b error seen with CoT.

Even though models identify the correct label, Type 2b are also errors due to not providing adequate support for the label. There are far more of these errors which artificially inflate performance and we handle these with a Refute Filter (see §6). This filtering removes a few correct labels, especially seen with GPT-5.4-mini and Claude-Haiku-4.5, where models correctly cite that there are no known reasons to support the claim. SMART-WS can correct both Types 2a and 2b errors with external evidence helping curate better premises. Models’ performance mainly differs in the number of these errors.

<table><tr><td>Dataset- -&gt;</td><td colspan="3">SFBench</td><td colspan="3">SciFact</td></tr><tr><td>Model</td><td>CoT</td><td>SMART-PK</td><td>SMART-S</td><td>CoT</td><td>SMART-PK</td><td>SMART-S</td></tr><tr><td>Qwen3 1.7b</td><td>56</td><td>43</td><td>58</td><td>78</td><td>46</td><td>48</td></tr><tr><td>Qwen3 8b</td><td>28</td><td>20</td><td>53</td><td>63</td><td>31</td><td>24</td></tr><tr><td>Qwen3 32b</td><td>12</td><td>09</td><td>26</td><td>67</td><td>32</td><td>11</td></tr><tr><td>Llama 8b</td><td>48</td><td>27</td><td>51</td><td>87</td><td>99</td><td>29</td></tr><tr><td>Llama 70b</td><td>13</td><td>12</td><td>47</td><td>61</td><td>27</td><td>18</td></tr><tr><td>GPT-5.4-mini</td><td>8</td><td>7</td><td>24</td><td>23</td><td>16</td><td>13</td></tr><tr><td>Cla. Haiku 4.5</td><td>22</td><td>7</td><td>48</td><td>51</td><td>11</td><td>19</td></tr></table>

Table 3: Comparing the percentage of rationales that lack support for a refute claim that are filtered by the Refute Filter for CoT reasoning, SMART-PK reasoning with contrastive premises and the single premise generation method (SMART-S). These lead to incorrect logical reasoning on 50% of SFBench claims and 40% of SciFact claims (claims with a gold refute). CoT leads to almost double the errors than SMART on smaller models. Single premise generation leads to more errors on SFBench (double the number than on SciFact) where as the reverse is true for contrastive premises. We discuss these results in §6.

d) Both CoT and SMART are incorrect due to a lack of knowledge. Similar to a) Model Positivity leads to errors in refuted labels. [Type 3]

Type 1 situations improve SMART performance over CoT; Type 2a & 2b lower the performance for both CoT and SMART after refute filtering; Type 3 lower the performance of both.

<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td colspan="3">Chain-of-Thought</td><td colspan="3">RAG with Web Search</td><td colspan="2">SMART-PK</td><td rowspan="2"></td><td rowspan="2">%SMART Improv.</td></tr><tr><td>Before RF</td><td>After RF</td><td>%Drop↓</td><td>Before RF</td><td> $\mathrm { A f t e r R F }$ </td><td>%Drop↓</td><td>Before RF  $\mathrm { A f t e r \ R F }$ </td><td>%Drop↓</td></tr><tr><td rowspan="7">SFBench</td><td>Qwen3 1.7b</td><td>.534±.000</td><td>.165±.006</td><td>36.9</td><td>.503±.008</td><td> $\overline { { . 1 0 0 { \pm } . 0 0 8 } }$ </td><td>40.3</td><td>.535±.023</td><td> $\overline { { . 2 7 4 \pm . 0 2 5 } }$ </td><td>26.1</td><td>10.9</td></tr><tr><td>Qwen3 8b</td><td>.610±.006</td><td>.369±.005</td><td>34.1</td><td>.500±.000</td><td> $. 0 0 0 { \scriptstyle \pm . 0 0 0 }$ </td><td>50.0</td><td> $. 5 6 8 { \pm } . 0 0 9$ </td><td> $. 4 3 5 { \pm } . 0 1 6$ </td><td>13.3</td><td>6.6</td></tr><tr><td>Qwen3 32b</td><td>.642±.000</td><td>.362±.005</td><td>28.0</td><td>.507±.068</td><td> $. 0 9 6 { \pm } . 0 2 2$ </td><td>43.9</td><td> $. 6 2 0 { \pm } . 0 1 9$ </td><td> $. 5 6 9 { \pm } . 0 3 2$ </td><td>5.1</td><td>20.7</td></tr><tr><td>Llama 8b</td><td>.523±.000</td><td>.243±.099</td><td>28.0</td><td>.514±.009</td><td> $. 0 3 8 { \pm } . 0 0 6$ </td><td>47.6</td><td> $. 5 7 9 2 . 0 0 4$ </td><td> $. 4 4 7 { \pm } . 0 1 0$ </td><td>13.2</td><td>20.4</td></tr><tr><td>Llama 70b</td><td>.643±.000</td><td>.514±.000</td><td>12.9</td><td>.519±.003</td><td> $. 0 4 0 { \pm } . 0 0 8 $ </td><td>47.9</td><td> $. 5 4 2 { \pm } . 0 2 3$ </td><td> $. 5 3 6 { \pm } . 0 3 0$ </td><td>0.6</td><td>2.2</td></tr><tr><td>GPT-5.4-mini</td><td>.713±.012</td><td>.600±.020</td><td>11.3</td><td>.503±.004</td><td> $. 0 0 3 { \pm } . 0 0 4$ </td><td>50.0</td><td> $. 7 5 0 { \pm } . 0 2 6$ </td><td> $. 7 0 2 { \pm } . 0 0 4$ </td><td>4.8</td><td>10.2</td></tr><tr><td>Cla. Haiku 4.5</td><td>.654±.004</td><td>.522±.008</td><td>13.2</td><td>.506±.016</td><td> $. 0 1 1 { \pm } . 0 0 8$ </td><td>49.5</td><td>.653±.012</td><td> $. 5 8 3 { \pm } . 0 0 4 $ </td><td>7.0</td><td>6.1</td></tr><tr><td rowspan="7">SciFact &amp; SciFact- Open</td><td>Qwen3 1.7b</td><td>.590±.000</td><td>.377±.007</td><td>21.3</td><td>.619±.027</td><td> $\overline { { 2 4 0 { \pm } . 0 5 0 } }$ </td><td>37.9</td><td>.646±.019</td><td> $\overline { { . 4 4 8 \pm . 0 4 4 } }$ </td><td>19.8</td><td>7.1</td></tr><tr><td>Qwen3 8b</td><td>.707±.000</td><td>.548±.011</td><td>15.9</td><td>.654±.037</td><td> $. 2 0 4 { \pm } . 0 6 5$ </td><td>45.0</td><td> $. 7 2 2 { \pm } . 0 1 4$ </td><td> $. 6 4 1 { \pm } . 0 0 7$ </td><td>8.1</td><td>9.3</td></tr><tr><td>Qwen3 32b</td><td>.652±.003</td><td>.616±.009</td><td>3.6</td><td>.656±.049</td><td> $. 1 9 2 { \pm } . 0 6 5$ </td><td>46.4</td><td> $. 7 3 1 { \pm } . 0 1 4$ </td><td> $. 6 9 6 { \pm } . 0 1 8$ </td><td>3.5</td><td>8.0</td></tr><tr><td>Llama 8b</td><td>.498±.000</td><td>.035±.011</td><td>46.3</td><td>.654±.045</td><td> $2 5 8 { \pm } . 0 8 3$ </td><td>39.6</td><td> $. 6 9 0 { \pm } . 0 2 5 $ </td><td> $. 5 9 5 { \pm } . 0 1 5$ </td><td>9.5</td><td>56.0</td></tr><tr><td>Llama 70b</td><td>.699±.004</td><td>.630±.000</td><td>6.9</td><td>.668±.057</td><td> $. 3 0 6 { \pm } . 0 9 3$ </td><td>36.2</td><td> $. 6 6 5 { \pm } . 0 1 2$ </td><td> $. 6 5 4 { \pm } . 0 1 4$ </td><td>1.1</td><td>2.4</td></tr><tr><td>GPT-5.4-mini</td><td>.826±.004</td><td>.750±.006</td><td>7.6</td><td>.663±.004</td><td> $. 2 0 2 { \pm } . 0 0 2$ </td><td>46.1</td><td> $. 7 9 8 { \pm } . 0 0 6$ </td><td> $. 7 6 5 { \pm } . 0 0 8$ </td><td>3.3</td><td>1.5</td></tr><tr><td>Cla. Haiku4.5</td><td>.796±.003</td><td>.736±.019</td><td>6.0</td><td>.589±.038</td><td> $. 1 0 7 { \pm } . 0 3 1$ </td><td>48.2</td><td> $. 7 8 2 { \pm } . 0 1 4$ </td><td> $. 7 4 9 { \pm } . 0 1 7$ </td><td>3.3</td><td>1.3</td></tr></table>

Table 4: Comparing the balanced accuracy of CoT, RAG and SMART-PK before and after applying the Refute Filter (refute labels that are not adequately supported are changed to ’unknown’, essentially dropping them from contributing to balanced accuracy). Smaller models either lack or are unable to leverage claim knowledge and select more refute labels, both with CoT and SMART reasoning (the %drop is higher for the smaller models). However, this incorrect performance boost is worse with CoT than SMART (the %drop is higher for CoT than SMART). It is also worse for SFBench’s Material Science knowledge than SciFact’s Biological Science knowledge (the %drop is higher SFBench than SciFact, Llama3 8b model on SciFact is an exception where the model does not generate most rationales). Web search based on a claim is noisy; up to half the RAG refute labels are based on not finding the appropriate evidence. Since RAG performance is lower than CoT, we provide % SMART improvement over CoT in the last column (SMART-PK improves performance up to 20%.) We discuss these results in §6.

<table><tr><td rowspan="2">Data Model</td><td rowspan="2"></td><td colspan="3">Single Premise SMART</td><td colspan="3">Contrastive Premises (SMART-PK)</td><td rowspan="2">%Contrast. Improv. After RF</td></tr><tr><td>Before RF</td><td>After RF</td><td>%Drop↓</td><td>Before RF After RF</td><td>%Drop↓</td><td>Before RF</td></tr><tr><td rowspan="7">SFBench</td><td>Qwen3 1.7b</td><td> $. 5 5 0 { \pm } . 0 0 5 $ </td><td> $3 3 0 { \pm } . 0 0 5$ </td><td>22.0</td><td> $\overline { { . 5 3 5 \pm . 0 2 3 } }$   $. 2 7 4 \pm . 0 2 5$ </td><td>26.1</td><td>-1.5</td><td>-5.6</td></tr><tr><td>Qwen3 8b</td><td> $. 5 6 6 { \pm } . 0 0 5$ </td><td> $. 3 9 2 { \pm } . 0 0 5$ </td><td>17.4</td><td>.568±.009  $. 4 3 5 { \pm } . 0 1 6$ </td><td>13.3</td><td>0.2</td><td>4.3</td></tr><tr><td>Qwen3 32b</td><td> $. 5 6 7 { \pm } . 0 1 2$ </td><td> $. 5 0 8 { \pm } . 0 0 5$ </td><td>5.9</td><td>.620±.019  $. 5 6 9 { \pm } . 0 3 2$ </td><td>5.1</td><td>5.3</td><td>6.1</td></tr><tr><td>Llama 8b</td><td> $. 6 1 1 { \pm } . 0 1 6$ </td><td> $. 4 4 2 { \pm } . 0 0 5$ </td><td>16.9</td><td>.579±.004  $. 4 4 7 { \pm } . 0 1 0$ </td><td>13.2</td><td>-3.2</td><td>0.5</td></tr><tr><td>Llama 70b</td><td> $. 6 4 0 { \pm } . 0 0 6$ </td><td> $. 4 9 2 { \pm } . 0 0 5$ </td><td>14.8</td><td>.542±.023  $. 5 3 6 { \pm } . 0 3 0$ </td><td>0.6</td><td>-9.8</td><td>4.4</td></tr><tr><td>GPT-5.4-mini</td><td> $. 7 6 0 { \scriptstyle \pm . 0 0 8 }$ </td><td> $. 6 5 9 { \pm } . 0 0 5$ </td><td>6.1</td><td>.750±.026  $. 7 0 2 { \pm } . 0 0 4$ </td><td>4.8</td><td>-1.0</td><td>0.3</td></tr><tr><td>Cla. Haiku 4.5</td><td> $. 6 5 2 { \pm } . 0 1 0$ </td><td> $. 5 3 3 { \pm } . 0 0 5$ </td><td>11.9</td><td>.653±.012 .583±.004</td><td>7.0</td><td>-0.1</td><td>5.0</td></tr><tr><td rowspan="7">SciFact &amp; SciFact- Open</td><td>Qwen3 1.7b</td><td> $\overline { { . 6 2 6 { \pm } . 0 1 0 } }$ </td><td> $\overline { { . 4 3 2 \pm . 0 0 5 } }$ </td><td>19.4</td><td>.646±.019 .448±.044</td><td>19.8</td><td>2.0</td><td>1.6</td></tr><tr><td>Qwen3 8b</td><td> $. 7 1 4 \pm . 0 0 2$ </td><td> $. 5 9 9 { \pm } . 0 0 5$ </td><td>11.5</td><td> $. 7 2 2 { \pm } . 0 1 4$   $. 6 4 1 { \pm } . 0 0 7$ </td><td>8.1</td><td>0.8</td><td>4.2</td></tr><tr><td>Qwen3 32b</td><td>.788±.005</td><td> $. 6 7 5 { \pm } . 0 0 5$ </td><td>11.3</td><td> $. 7 3 1 { \pm } . 0 1 4$   $. 6 9 6 { \pm } . 0 1 8$ </td><td>3.5</td><td>-5.7</td><td>2.1</td></tr><tr><td>Llama 8b</td><td>.650±.004</td><td> $. 5 4 7 { \pm } . 0 0 5$ </td><td>10.3</td><td>.690±.025  $. 5 9 5 { \pm } . 0 1 5$ </td><td>9.5</td><td>-5.5</td><td>4.8</td></tr><tr><td>Llama 70b</td><td>.658±.005</td><td>.620±.005</td><td>3.8</td><td>.665±.012  $. 6 5 4 \pm . 0 1 4$ </td><td>1.1</td><td>-0.4</td><td>3.4</td></tr><tr><td>GPT-5.4-mini</td><td>.804±.006</td><td> $. 7 4 3 { \pm } . 0 0 5$ </td><td>6.1</td><td>.798±.006  $. 7 6 5 { \pm } . 0 0 8$ </td><td>3.3</td><td>-0.6</td><td>2.2</td></tr><tr><td>Cla. Haiku4.5</td><td> $. 7 7 3 { \pm } . 0 0 8$ </td><td> $. 7 1 2 { \pm } . 0 0 5$ </td><td>6.1</td><td> $. 7 8 2 { \pm } . 0 1 4$   $. 7 4 9 { \pm } . 0 1 7$ </td><td>3.3</td><td>0.9</td><td>3.7</td></tr></table>

Table 5: Comparing the balanced accuracy of using Single Vs. Contrastive premises with the SMART framework. We examine performance before and after applying the Refute Filter (refute labels that are not adequately supported are changed to ’unknown’, essentially dropping them from contributing to balanced accuracy). Single premises often lack necessary claim knowledge that models select refute labels without adequately supporting them. Correcting for this with the Refute Filter shows that contrastive knowledge improves on average across models and datasets by 3%. The smallest model Qwen3 1.7b is worse at obtaining contrastive knowledge, but leads to better performance with single premises. We discuss these results in §6.

<table><tr><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>SMART-PK</td><td rowspan=1 colspan=1>SMART-WS  %Imp.↑</td><td rowspan=1 colspan=1>X-SMART-PK  % Imp.↑</td></tr><tr><td rowspan=7 colspan=1>SFBench</td><td rowspan=2 colspan=1>Qwen3 1.7bQwen3 8b</td><td rowspan=1 colspan=1> $\overline { { . 5 3 5 { \pm } . 0 2 3 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 5 4 4 \pm . 0 2 0 } }$       0.9</td><td rowspan=1 colspan=1>.620±.015        8.5</td></tr><tr><td rowspan=1 colspan=1> $. 5 6 8 { \pm } . 0 0 9$ </td><td rowspan=1 colspan=1> $. 5 9 8 { \pm } . 0 1 4$       3.0</td><td rowspan=1 colspan=1> $. 6 2 1 { \pm } . 0 1 3$         5.3</td></tr><tr><td rowspan=1 colspan=1>Qwen3 32b</td><td rowspan=1 colspan=1> $. 6 2 0 { \pm } . 0 1 9$ </td><td rowspan=1 colspan=1> $. 6 1 1 \pm . 0 1 9$       -0.9</td><td rowspan=1 colspan=1> $. 6 4 5 { \pm } . 0 2 1$         2.5</td></tr><tr><td rowspan=4 colspan=1>Llama3.1 8bLlama3.1 70bGPT-5.4-miniCla. Haiku 4.5</td><td rowspan=1 colspan=1> $. 5 7 9 { \pm } . 0 0 4$ </td><td rowspan=1 colspan=1> $. 5 4 5 { \pm } . 0 2 9$       -3.4</td><td rowspan=1 colspan=1> $. 5 7 8 { \pm } . 0 1 9$         0.0</td></tr><tr><td rowspan=2 colspan=1>.542±.023.750±.026</td><td rowspan=1 colspan=1> $. 6 1 0 { \pm } . 0 0 7$       6.8</td><td rowspan=1 colspan=1> $. 5 9 8 { \pm } . 0 1 9$        5.6</td></tr><tr><td rowspan=1 colspan=1> $. 7 3 6 { \pm } . 0 1 5$       -2.4</td><td rowspan=1 colspan=1>.750±.031</td></tr><tr><td rowspan=1 colspan=1>.653±.012</td><td rowspan=1 colspan=1> $. 6 5 8 { \pm } . 0 3 5$       0.5</td><td rowspan=1 colspan=1>.677±.000        2.4</td></tr><tr><td rowspan=7 colspan=1>SciFact &amp;SciFact-Open</td><td rowspan=4 colspan=1>Qwen3 1.7bQwen3 8bQwen3 32bLlama3.1 8b</td><td rowspan=1 colspan=1> $\overline { { . 6 4 6 { \pm } . 0 1 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { . 6 3 2 \pm . 0 1 5 } }$       -1.4</td><td rowspan=1 colspan=1>.692±.010        4.6</td></tr><tr><td rowspan=2 colspan=1> $. 7 2 2 { \pm } . 0 1 4$  $. 7 3 1 { \pm } . 0 1 4$ </td><td rowspan=1 colspan=1> $. 7 3 2 { \pm } . 0 0 9$       1.0</td><td rowspan=1 colspan=1> $. 7 2 2 { \pm } . 0 1 4$ </td></tr><tr><td rowspan=1 colspan=1> $. 7 3 4 \pm . 0 2 7$       0.3</td><td rowspan=1 colspan=1> $. 8 7 3 { \pm } . 0 1 0$        14.1</td></tr><tr><td rowspan=1 colspan=1> $. 6 9 0 { \pm } . 0 2 5 $ </td><td rowspan=1 colspan=1> $. 6 4 0 { \pm } . 0 1 3$       -5.0</td><td rowspan=1 colspan=1> $. 7 2 7 { \pm } . 0 1 4$        3.7</td></tr><tr><td rowspan=3 colspan=1>Llama3.1 70bGPT-5.4-miniCla. Haiku4.5</td><td rowspan=1 colspan=1> $. 6 6 5 { \pm } . 0 1 2$ </td><td rowspan=1 colspan=1> $. 6 8 2 { \pm } . 0 1 2$       1.7</td><td rowspan=1 colspan=1> $. 7 9 6 { \pm } . 0 1 2$        13.1</td></tr><tr><td rowspan=1 colspan=1> $. 7 9 8 { \pm } . 0 0 6$ </td><td rowspan=1 colspan=1> $. 7 8 8 { \pm } . 0 1 4$       -1.0</td><td rowspan=1 colspan=1> $. 8 6 6 { \pm } . 0 1 0$         6.5</td></tr><tr><td rowspan=1 colspan=1> $. 7 8 2 { \pm } . 0 1 4$ </td><td rowspan=1 colspan=1> $. 7 8 4 \pm . 0 0 4$       0.2</td><td rowspan=1 colspan=1> $. 8 0 9 { \pm } . 0 1 2$         8.4</td></tr></table>

Table 6: Comparing performance (BAccuracy) of SMART-WS and X-SMART-PK with SMART-PK. X-SMART-PK where models use premises from a better generating model: Qwen3 8b for SciFact and GPT-5.4-mini for SFBench. Since SMART-PK does not lead to as much of an artificial performance boost, we compare results without applying the Refute Filter. Performance on SciFact collection of claims improves by a 6.5-14.1%; and up to 8.5% for SFBench claims. We discuss these results and choice of curation model in §6.

<table><tr><td>Data</td><td>Model</td><td>SMART-PK</td><td>MajVot</td><td>AggJ</td><td>CriJ</td><td>SuppJ</td><td>%Improv.↑</td></tr><tr><td rowspan="6">SFBench</td><td>Qwen3 1.7b</td><td>.535±.023</td><td> $\overline { { . 5 3 6 \pm . 0 2 2 } }$ </td><td>.540±.009</td><td>.520±.005</td><td>.528±.010</td><td>-0.1</td></tr><tr><td>Qwen3 8b</td><td>.568±.009</td><td> $. 5 2 2 { \pm } . 0 1 4$ </td><td>.558±.007</td><td>.543±.016</td><td>.586±.021</td><td>4.6</td></tr><tr><td>Qwen3 32b</td><td>.620±.019</td><td> $. 5 3 0 { \pm } . 0 0 8$ </td><td>.606±.019</td><td>.575±.025</td><td>.594±.019</td><td>9.0</td></tr><tr><td>Llama3.1 8b</td><td>.579±.004</td><td> $. 4 6 4 \pm . 0 1 4$ </td><td>.534±.027</td><td>.529±.014</td><td>.568±.022</td><td>11.5</td></tr><tr><td>Llama3.1 70b</td><td>.542±.023</td><td> $. 5 0 6 { \pm } . 0 4 2$ </td><td>.552±.015</td><td>.553±.007</td><td>.523±.005</td><td>3.6</td></tr><tr><td>GPT-5.4-mini</td><td>.750±.026</td><td> $. 5 2 8 { \pm } . 0 1 1$ </td><td>.739±.012</td><td>.646±.023</td><td>.728±.030</td><td>22.2</td></tr><tr><td rowspan="6">SciFact &amp; SciFact- Open</td><td>Cla. Haiku 4.5 Qwen3 1.7b</td><td>.653±.012 .646±.019</td><td> $. 5 9 3 { \pm } . 0 1 2$   $\overline { { . 6 1 3 \pm . 0 2 4 } }$ </td><td>.688±.008 .634±.017</td><td>.685±.004</td><td>.678±.039</td><td>6.0 3.3</td></tr><tr><td>Qwen3 8b</td><td>.722±.015</td><td></td><td></td><td>.631±.020</td><td>.637±.008</td><td></td></tr><tr><td>Qwen3 32b</td><td>.731±.014</td><td> $. 7 0 4 \pm . 0 0 3$ </td><td>.722±.014</td><td>.685±.011</td><td>.725±.011</td><td>1.8</td></tr><tr><td>Llama3.1 8b</td><td>.690±.025</td><td> $. 6 7 4 \pm . 0 1 9$ </td><td>.730±.016 .691±.003</td><td>.686±.003</td><td>.692±.018</td><td>5.7</td></tr><tr><td>Llama3.1 70b</td><td> $. 6 6 5 { \pm } . 0 1 2$ </td><td> $. 6 3 1 { \pm } . 0 1 5$   $. 6 0 7 { \pm } . 0 1 0$ </td><td>.658±.015</td><td>.654±.020 .650±.017</td><td>.631±.011 .632±.002</td><td>5.9</td></tr><tr><td>GPT-5.4-mini</td><td>.798±.006</td><td> $. 7 2 0 { \pm } . 0 0 7$ </td><td>.795±.002</td><td>.757±.010</td><td>.796±.012</td><td>5.3 7.8</td></tr><tr><td rowspan="2"></td><td>Cla. Haiku 4.5</td><td>.782±.014</td><td> $. 7 2 3 { \pm } . 0 2 2$ </td><td>.788±.001</td><td>.778±.016</td><td>.733±.021</td><td>5.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: Comparing the balanced accuracy of SMART-PK (with Reasoning Judge as meta-reasoner) with 4 other meta-reasoners. We show the advantage of using the Reasoning Judge over Majority Voting in the %Improv. column.

<table><tr><td>Data</td><td>Model</td><td>SMART-NV</td><td>SMART-PK</td><td>%Improv↑</td><td>SMART-WS</td><td>%Improv↑</td><td>X-SMART-PK</td><td>%Improv↑</td></tr><tr><td rowspan="7">SFBench</td><td>Qwen3 1.7b</td><td>.520±.005</td><td>.535±.023</td><td>1.5</td><td>.544±.020</td><td>2.4</td><td>.620±.015</td><td>10.0</td></tr><tr><td>Qwen3 8b</td><td>.542±.014</td><td>.568±.009</td><td>2.6</td><td>.598±.014</td><td>5.6</td><td>.621±.013</td><td>7.9</td></tr><tr><td>Qwen3 32b</td><td>.564±.014</td><td>.620±.019</td><td>4.9</td><td>.611±.019</td><td>4.7</td><td>.645±.021</td><td>8.1</td></tr><tr><td>Llama3.1 8b</td><td>.572±.016</td><td>.579±.004</td><td>0.7</td><td>.545±.029</td><td>-2.7</td><td>.578±.019</td><td>0.6</td></tr><tr><td>Llama3.1 70b</td><td>.518±.004</td><td>.542±.023</td><td>2.4</td><td>.610±.007</td><td>9.2</td><td>.598±.019</td><td>8.0</td></tr><tr><td>GPT-5.4-mini</td><td>.753±.008</td><td>.750±.026</td><td>-0.3</td><td>.736±.015</td><td>-1.7</td><td>.750±.031</td><td>-0.3</td></tr><tr><td>Cla. Haiku 4.5</td><td>.650±.015</td><td>.653±.012</td><td>0.3</td><td>.658±.035</td><td>0.8</td><td>.677±.000</td><td>2.7</td></tr><tr><td rowspan="7">SciFact &amp; SciFact- Open</td><td>Qwen3 1.7b</td><td>.636±.012</td><td>.646±.019</td><td>1.0</td><td>.632±.015</td><td>-0.4</td><td>.692±.010</td><td>5.6</td></tr><tr><td>Qwen3 8b</td><td>.697±.002</td><td>.722±.015</td><td>2.5</td><td>.732±.009</td><td>3.5</td><td>.722±.014</td><td>2.5</td></tr><tr><td>Qwen3 32b</td><td>.684±.011</td><td>.731±.014</td><td>4.7</td><td>.734±.027</td><td>5.0</td><td>.873±.010</td><td>18.9</td></tr><tr><td>Llama3.1 8b</td><td>.697±.003</td><td>.690±.025</td><td>-0.7</td><td>.640±.013</td><td>-5.7</td><td>.727±.014</td><td>3.0</td></tr><tr><td>Llama3.1 70b</td><td>.645±.018</td><td>.665±.012</td><td>2.0</td><td>.682±.012</td><td>3.7</td><td>.796±.012</td><td>15.1</td></tr><tr><td>GPT-5.4-mini</td><td>.771±.005</td><td>.798±.006</td><td>2.7</td><td>.788±.014</td><td>1.7</td><td>.866±.010</td><td>9.5</td></tr><tr><td>Cla. Haiku 4.5</td><td>.765±.010</td><td>.782±.014</td><td>1.7</td><td>.784±.004</td><td>1.9</td><td>.809±.012</td><td>4.4</td></tr></table>

Table 8: Comparing the balanced accuracy of SMART without verification (SMART-NV) with self-verification (SMART-PK), external verification with websearch (SMART-WS) and cross-verification (X-SMART-PK). Improvement with each method is shown in %Improve (the difference in performance between SMART-NV and the method). Verification improves performance on average across models and datasets by by 2-7%. Grounding premises with real-world knowledge provided by websearch improves on average by 4-5% with some exceptions.

<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td rowspan="2">Model Calls</td><td colspan="4">SMART-PK Inference</td><td colspan="4">SMART-WS Inference</td><td>Premise</td></tr><tr><td>KTok</td><td>%Fback</td><td>Sup.</td><td>Opp.</td><td>KTok</td><td>%Fback</td><td>Sup.</td><td>Opp.</td><td>Quality</td></tr><tr><td rowspan="7">SFBench</td><td>Qwen3 1.7b</td><td>5</td><td>11</td><td>42</td><td>1</td><td>0</td><td>11</td><td>70</td><td>.5</td><td>0</td><td>NA</td></tr><tr><td>Qwen3 8b</td><td>11</td><td>12</td><td>25</td><td>1.5</td><td>1.2</td><td>11</td><td>83</td><td>.3</td><td>.1</td><td>NA</td></tr><tr><td>Qwen3 32b</td><td>19</td><td>13</td><td>3</td><td>4.1</td><td>1.9</td><td>12</td><td>23</td><td>2.0</td><td>.73</td><td>NA</td></tr><tr><td>Llama 8b</td><td>15</td><td>20</td><td>4</td><td>2.7</td><td>1.6</td><td>16</td><td>40</td><td>1.2</td><td>.7</td><td>NA</td></tr><tr><td>Llama 70b</td><td>18</td><td>14</td><td>0</td><td>5</td><td>1.9</td><td>11</td><td>63</td><td>.6</td><td>.2</td><td>NA</td></tr><tr><td>GPT-5.4-mini</td><td>19</td><td>13</td><td>4</td><td>2.5</td><td>2</td><td>12</td><td>10</td><td>2.0</td><td>1.5</td><td>NA</td></tr><tr><td>Cla. Haiku 4.5</td><td>20</td><td>14</td><td>0</td><td>3.8</td><td>3.9</td><td>12</td><td>12</td><td>2.3</td><td>1.6</td><td>NA</td></tr><tr><td rowspan="6">SciFact &amp; SciFact- Open</td><td>Qwen3 1.7b</td><td>4</td><td>11</td><td>49</td><td>.7</td><td>.1</td><td>11</td><td>67</td><td>.4</td><td>.1</td><td>.41</td></tr><tr><td>Qwen3 8b</td><td>11</td><td>12</td><td>19</td><td>1.7</td><td>1.3</td><td>12</td><td>42</td><td>1.0</td><td>.8</td><td>.36</td></tr><tr><td>Qwen3 32b</td><td>18</td><td>13</td><td>3</td><td>3.6</td><td>2.3</td><td>12</td><td>.25</td><td>1.8</td><td>1.3</td><td>.36</td></tr><tr><td>Llama 8b</td><td>14</td><td>13</td><td>2</td><td>3.3</td><td>1.5</td><td>13</td><td>14</td><td>2.6</td><td>1.2</td><td>.59</td></tr><tr><td>Llama 70b</td><td>17</td><td>14</td><td>1</td><td>5.1</td><td>1.1</td><td>11</td><td>47</td><td>1.7</td><td>.4</td><td>.32</td></tr><tr><td>GPT-5.4-mini</td><td>16</td><td>12</td><td>8</td><td>2.3</td><td>1.6</td><td>12</td><td>16</td><td>2.0</td><td>1.3</td><td>.28</td></tr><tr><td></td><td>Cla. Haiku4.5</td><td>20</td><td>14</td><td>1</td><td>5.6</td><td>2.9</td><td>13</td><td>4</td><td>3.6</td><td>2.2</td><td>.34</td></tr></table>

Table 9: Comparing inference cost for SMART-PK and SMART-WS in model calls and number of input + output tokens measured in K tokens (KTok). We report % fallbacks (Fback) to the direct reasoning prompt when there are no curated premises, the quality of curated premises (Qual) and their average numbers per stance (Sup. & Opp.). Both methods require the same # calls since we verify before obtaining stance; these do not include 10 calls that iteratively obtain premises as a single call could potentially be used to obtain all premises for GPT-5.4-mini and Claude-Haiku-4.5. All values are per claim averages. We report NA-Not applicable as the premise quality (Qual) for SFbench claims due to not having textual evidence. Average premise quality by coincidence is the same for both methods, so we report in a single column.

![](images/8f7a6eea4101d1fc29bd1436e3d29ec71d7cfc93a11eb311136bc7deb8200ed8.jpg)  
(a) Qwen3 1.7b curated premises.

![](images/ff5d09c3f67ce009edb4830e864b6fdd134e34a9f8ef8d0ff8d0ce1750e560f0.jpg)  
(b) Qwen3 8b curated premises.

![](images/e3b41b6f40c14b289b1f2828575504dee4bcb427a01706de0d92946e8cd63713.jpg)  
(c) Qwen3 32b curated premises.

Figure 14: Distribution of curated premise stances, their alignment with the claim and their confidence ratings for Qwen3 models. Correct identification of refuted claims requires opposing premises and supported claims require supporting premises. Qwen3 1.7b curation leads to few and noisy premises compared to Qwen8b. Qwen3 32 generates too many supporting premises increasing meta-reasoning difficulty. Supporting and opposing premises are required for supporting claims of support and refute labels respectively.

![](images/1a328bf0ff4804bfa701c63f694a6e75055a752e2df55b7af0ebc5c53290dcfc.jpg)  
Figure 15: We compare various configurations we used for generating premises. The performance of these different configurations is compared in Fig. 12.

<table><tr><td>Models</td><td>Contrastive premise pairs (separated by an &amp;; ... replaces repeating text from premise #1) Fig. 1 Example Claim: Genetic deletion of JAM-A decreases gut permeability.</td></tr><tr><td>Qwen3 1b</td><td>Premises are few and superficial. Genetic deletion of JAM-A occurs in a specific cellular context. &amp; Genetic deletion of JAM-A does not occur in a specific cellular context. Genetic deletion of JAM-A is associated with a specific cellular location. &amp; Genetic deletion of JAM-A is not associated with a specific cellular location.</td></tr><tr><td>Qwen3 8b</td><td>Premises are atomic and easy to validate. JAM-A is involved in maintaining gut barrier integrity. &amp; JAM-A is not ... integrity. JAM-A promotes the formation of tight junctions in intestinal epithelial cells. &amp; JAM-A inhibits the formation of tight junctions in intestinal epithelial cells. Genetic deletion of JAM-A reduces the expression of tight junction proteins. &amp; Genetic deletion of JAM-A increases the expression of tight junction proteins. JAM-A is a component of the intestinal epithelial cell membrane. &amp; JAM-A is not .. Genetic deletion of JAM-A disrupts the structural integrity of the intestinal epithelial barrier. &amp; Genetic deletion of JAM-A enhances the structural ... epithelial barrier. Genetic deletion of JAM-A decreases the expression of claudin proteins. &amp; Genetic deletion of JAM-A increases the expression of claudin proteins. Genetic deletion of JAM-A decreases the expression of occludin proteins.&amp; Genetic deletion of JAM-A increases the expression of occludin proteins. Genetic deletion of JAM-A increases the paracellular transport of molecules across the</td></tr><tr><td>Qwen3 32b</td><td>intestinal epithelium. &amp; Genetic deletion of JAM-A decreases ... intestinal epithelium. Premises are more nuanced, as result these are less atomic and difficult to validate. JAM-A is normally expressed in gut epithelial cells. &amp; JAM-A is not expressed ... The intestinal barrier is intact.&amp; The intestinal barrier is compromised. JAM-A deletion leads to reduced paracellular transport. &amp; JAM-A deletion does not ... JAM-A is required for tight junction integrity. &amp; JAM-A is not required .. JAM-A deletion increases paracellular permeability. &amp; JAM-A deletion decreases .. JAM-A deletion enhances intestinal barrier function. &amp; JAM-A deletion impairs ... JAM-A deletion reduces intestinal paracellular permeability. &amp; JAM-A deletion increa .. JAM-A is essential for maintaining gut epithelial barrier integrity. &amp; JAM-A is not . JAM-A deletion reduces the passage of molecules through the gut epithelium. &amp; JAM-A deletion increases the passage of molecules through the gut epithelium. JAM-A deletion is associated with decreased paracellular transport of luminal antigens. &amp;</td></tr><tr><td>GPT-5.4-mini</td><td>JAM-A deletion is associated with increased paracellular transport of luminal antigens. Premises are more scientific &amp; salient; not easy for smaller models to validate. JAM-A has a barrier-stabilizing role in intestinal epithelial tight junctions. &amp; JAM-A has a barrier-weakening role in intestinal epithelial tight junctions. Claudin-1 abundance is elevated in the intestinal epithelium. &amp; Claudin-1 abundance is not elevated in the intestinal epithelium. The JAM-A deletion is restricted to intestinal epithelial cells. &amp; The JAM-A deletion is not restricted to intestinal epithelial cells. The intestinal mucosa is inflamed. &amp; The intestinal mucosa is not inflamed. Occludin abundance is elevated in the intestinal epithelium. &amp; Occludin ... is not ... Intestinal epithelial apoptosis is increased. &amp; Intestinal epithelial apoptosis is not . Tricellular tight junction sealing is compromised in the intestinal epithelium. &amp; Tricellular tight junction sealing is intact in the intestinal epithelium. JAM-A is localized at intestinal epithelial tight junctions. &amp; JAM-A is not localized ... Claudin-2 abundance is reduced in the intestinal epithelium. &amp; Claudin-2 abundance is not reduced in the intestinal epithelium. Myosin light chain phosphorylation is reduced in the intestinal epithelium. &amp; Myosin light chain phosphorylation is not reduced in the intestinal epithelium.</td></tr><tr><td>Models</td><td>Contrastive premise pairs (separated by an &amp;; ... replaces repeating text from premise #1)</td></tr><tr><td>Fig. 1 Example Claim: Genetic deletion of JAM-A decreases gut permeability. Llama3.1 8b</td><td>Premises are limited to the context and do not explore new entities and concepts. JAM-A is expressed in the epithelial cells of the gut &amp; JAM-A is not expressed ... Gut permeability is not affected by JAM-A deletion &amp; Gut permeability is increased by ...</td></tr><tr><td>Llama3.1 70b</td><td>The mechanism of JAM-A's effect on gut permeability is well understood &amp; The mecha- nism of JAM-A's effect on gut permeability is not well understood Gut permeability decreases when JAM-A is not present. &amp; Gut permeability does not .. JAM-A deletion leads to a significant decrease in gut permeability &amp; JAM-A deletion has no effect on gut permeability JAM-A deletion results in the disruption of tight junctions in the gut epithelial cells &amp; JAM-A deletion does not affect the structure of tight junctions in the gut epithelial cells JAM-A deletion leads to a decrease in gut permeability is a well-established scientific fact &amp; JAM-A deletion leads to an increase in gut permeability JAM-A is a key regulator of tight junctions in the gut epithelial cells &amp; JAM-A is not involved in regulating tight junctions in the gut epithelial cells Premises are simple and atomic but most premises support the claim. JAM-A is involved in the regulation of tight junctions in the gut epithelium &amp; JAM-A is not involved in the regulation of tight junctions in the gut epithelium JAM-A is a crucial component of the intestinal epithelial barrier &amp; JAM-A is not a cruc The deletion of JAM-A does not affect the expression of other tight junction proteins in the gut epithelium &amp; The deletion of JAM-A affects the expression .. The gut epithelium of genetically deleted JAM-A mice has increased levels of pro- inflammatory cytokines.&amp; The gut epithelium ... deleted JAM-A mice does not... JAM-A plays a role in the maintenance of the gut epithelial barrier's physical structure &amp; JAM-A does not play a role in the maintenance of the gut epithelial barrier's ... JAM-A is involved in the regulation of gut epithelial cell apoptosis &amp; JAM-A is not . The genetically deleted JAM-A mice have reduced gut motility &amp; The genetically deleted</td></tr><tr><td>Claude-Haiku -4.5</td><td>JAM-A mice do not have reduced gut motility JAM-A deletion reduces the expression of occludin in the gut epithelium &amp; JAM-A deletion does not affect the expression of occludin ... Premises are more scientific; not easy for smaller models to validate. Tight junction protein complexes are intact and functional at the intestinal epithelial barrier. &amp; Tight junction protein complexes are disrupted or have compromised function ... JAM-A maintains adherens junction stability through interaction with VE-cadherin and catenin complexes. &amp; JAM-A does not contribute to adherens ... JAM-A regulates claudin-based tight junction strand formation and sealing function. &amp; JAM-A does not contribute to claudin-based tight junction strand formation or ... JAM-A is present and participates in leukocyte transepithelial migration regulation at the intestinal epithelial barrier. &amp; JAM-A is absent and cannot regulate leukocyte transepithe- lial migration, leading to increased immune cell infiltration and barrier destabilization. JAM-A is localized at tricellular junctions and forms sealing strands that prevent paracel- lular solute flux. &amp; JAM-A is absent from tricellular junctions and cannot form sealing strands, allowing increased paracellular solute permeability. JAM-A mediates homophilic interactions between adjacent epithelial cells that reinforce barrier cohesion and reduce paracellular permeability. &amp; JAM-A does not mediate ho- mophilic cell-cell interactions or such interactions are non-essential for maintain .. JAM-A negatively regulates paracellular transport through size-selective barrier function at tricellular corners. &amp; JAM-A does not regulate paracellular transport selectivity or plays a minor role in size-dependent barrier permeability. JAM-A deletion results in compensatory upregulation of alternative tight junction proteins that maintain barrier function. &amp; JAM-A deletion does not trigger compensatory upregula-</td></tr><tr><td>Error</td><td>Issue Description Claim:Fluorine-containing additives in a liquid electrolyte enable Li-ion battery chemistries to</td></tr><tr><td>Type 2b</td><td>cycle up to 10 V. CoT with Qwen3 8b: refuted label is correct. Rationale: While fluorine-containing additives in electrolytes can improve the performance and stability of Li-ion batteries, there is no widely accepted evidence that they enable cycling up to 10 V. The typical voltage range for Li-ion batteries is around 3.0–4.3 V, and achieving 10 V would require significant technological advancements beyond current understanding and practice. SMART with Qwen3 8b : supported label is incorrect. Rationale: The claim is supported by the evidence that fluorine-containing additives increase voltage capacity and enhance electro-</td></tr><tr><td>Type 1</td><td>Claim: A battery with a lithium titanate anode, a lithium manganese oxide cathode, and an electrolyte consisting of 1M lithium hexafluorophosphate in dimethyl carbonate can pass cold crank at -10'C CoT with Qwen3 8b: Lithium titanate (Li4Ti5O12) anodes and lithium manganese oxide (LiMn2O4) cathodes are known for their good low-temperature performance. The electrolyte described, 1M lithium hexafluorophosphate in dimethyl carbonate, is commonly used in lithium- ion batteries and can function effectively at low temperatures, including -10°C. Therefore, the claim is supported by current scientific knowledge. SMART with Qwen3 8b : The claim is not supported because the opposing conditions indicate</td></tr><tr><td>Type 3 Type 2a</td><td>that the electrolyte loses ionic conductivity, has high viscosity, and the anode has low electrical conductivity at low temperatures, all of which hinder cold crank performance at -10°C. CoT with Qwen3 32b: Lithium titanate anodes and lithium manganese oxide cathodes are known for their stability and performance at low temperatures. Carbon cloth as a current collector is flexible and suitable for wearable applications. The freezing point of the electrolyte (-50°C) and the use of polypropylene encapsulation suggest a design capable of withstanding -30°C. While the specific integration on a finger joint is novel, the materials and properties described are plausible for the claim, though experimental validation is not provided. SMART with Qwen3 32b : While the carbon cloth, lithium titanate anode, and polypropylene encapsulation suggest potential for cold performance and flexibility, the opposing conditions indicate significant challenges: loss of adhesion, rigidity, ionic conductivity, and mechanical durability at -30°C would likely prevent the battery from maintaining structural integrity and function on a moving finger joint. The claim is not strongly supported due to these critical performance limitations. CoT with Llama3.1 70b: infeasible label is correct. The claim is supported because the described battery components, including the lithium titanate anode, lithium manganese oxide cathode, and electrolyte with a freezing point of -50'C, are suitable for low-temperature operation,</td></tr><tr><td>Error</td><td>Issue Description</td></tr><tr><td>Type 1</td><td>Claim: Chenodeosycholic acid treatment reduces whole-body energy expenditure. CoT with Qwen3 32b: feasible label is incorrect. Chenodeoxycholic acid (CDCA) is a bile acid and FXR agonist, and FXR activation has been shown in experimental studies to suppress thermogenic pathways and lower whole-body energy expenditure. The evidence is mainly preclinical and context-dependent, so the support is good but not absolute. SMART with Qwen3 32b : infeasible label is correct. Chenodeoxycholic acid is a bile acid that can activate pathways such as TGR5, which promotes type 2 deiodinase activity in thermogenic tissue and tends to increase thermogenesis and whole-body energy expenditure. Given these</td></tr><tr><td>Type 2a</td><td>opposing conditions, the claim that it reduces energy expenditure is not supported and is likely opposite to the expected biological effect. Claim: Macrolides have no protective effect against myocardial infarction. CoT with Qwen3 32b: infeasible label is correct. Current scientific evidence does not strongly support the claim that macrolides protect against myocardial infarction. While some studies have suggested potential anti-inflammatory and immunomodulatory effects of macrolides, these effects have not been conclusively linked to a protective role against myocardial infarction in large, well-controlled clinical trials. SMART with Qwen3 32b : feasible label is incorrect. The claim that macrolides protect against myocardial infarction is supported by multiple plausible biological mechanisms, including anti- inflammatory, immunomodulatory, antioxidant, and anti-platelet effects, all of which are relevant to cardiovascular disease and myocardial infarction prevention. While there are no opposing</td></tr><tr><td>Type 3</td><td>conditions presented, the evidence is largely mechanistic and not based on strong clinical trial data showing direct protection against myocardial infarction, which limits the confidence to a high but not maximum level. CoT with Llama3.1 70b: feasible label is incorrect. Current scientific knowledge suggests that macrolides, a class of antibiotics, have been found to have neutral or conflicting effects on cardiovascular outcomes, including myocardial infarction. While some studies have investigated potential associations, there is no conclusive evidence to support a protective effect of macrolides against myocardial infarction, leading to a decision of'Y' with a moderate to high confidence rating. SMART with Llama3.1 70b: feasible label is incorrect. The claim that macrolides protect against myocardial infarction is supported due to their anti-inflammatory properties and the con- ditions provided, such as the patient having an inflammatory condition responsive to macrolides and high levels of CRP and IL-6 in the blood, which are associated with cardiovascular disease. However, the rating is not a 10 due to the limited scope of the conditions and the lack of direct</td></tr></table>

Table 10: Example premises with different Qwen3 and GPT-5.4-mini models. See Table 11 for other models.

Table 11: Example premises with Llama3 and Claude models. See Table 10 for other models.

Table 12: Errors seen in SFBench Data. With Qwen3 8b, 32b and Llama3.1 70b, there are more Type 2a & 2b situations than Type 1 leading to SMART-PK’s lower performance; SMART-WS improves upon some of the Type 2a & 2b errors through external premise validation. See error descriptions in §10.10 and Table 13 for SciFact errors.

Table 13: Errors seen in SciFact Data. With Llama3.1 70b, GPT-5.4-mini and Claude-Haiku-4.5, there are a more Type 2a situations than Type 1 leading to SMART-PK’s lower performance. X-SMART-PK reduces some of the Type 3 and Type 2a errors leading to improved performance. See error descriptions in §10.10 and Table 13 for SFBench errors.