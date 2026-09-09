# ReCite: Agentic Reasoning for Faithful Citation

Yuyang Huang<sup>1</sup>, Bobo Li<sup>2</sup>\*, Jiajia Song<sup>2</sup>, Yuzhe Ding<sup>1</sup>, Chong Teng<sup>1</sup>, Fei Li<sup>1</sup>, Donghong Ji<sup>1</sup>\*   
<sup>1</sup>School of Cyber Science and Engineering, Wuhan University <sup>2</sup>National University of Singapore {hyy279, dhji}@whu.edu.cn, {libobo, jjiasong}@nus.edu.sg  https://hyy279.github.io/ReCite/

## Abstract

Accurate citations are the foundation of academic writing, tracing intellectual origins and substantiating core claims. However, manually navigating the growing volume of scientific literature is increasingly difficult, prompting reliance on automatic citation recommendation. While modern retrieval-augmented architectures have largely mitigated the fabrication of non-existent papers, current systems relying on semantic similarity struggle with misattribution, often citing authentic papers that fail to logically support the author’s claim. To address this challenge, we argue that accurate citation requires a shift from similarity-based search to active, claim-level reasoning. We propose ReCite, a decoupled agentic framework that orchestrates location perception, intentaware query planning, and reflective verification. Trained on synthesized reasoning trajectories, our agent verifies claim-evidence consistency and triggers self-correction loops when retrieved candidates lack logical support. Experiments demonstrate that our lightweight framework outperforms state-of-the-art massive generative models in strict citation accuracy. By grounding literature matching in verifiable logic rather than semantic overlap, ReCite establishes a reliable foundation for automated academic writing.

## 1 Introduction

Accurate citations are the foundation of academic writing, serving to trace intellectual origins and substantiate core claims. However, as the volume of scientific literature grows exponentially, manually tracking relevant prior work and locating the most appropriate papers have become increasingly difficult (Bornmann and Mutz, 2015; Fortunato et al., 2018). To alleviate this growing cognitive burden, automatic citation recommendation (ACR) has been widely adopted as an essential research tool (Gao et al., 2023). As large language models (LLMs) increasingly draft academic texts, they introduce a severe risk of generating plausible but incorrect references (Ji et al., 2023; Sakai et al., 2026), forcing ACR to evolve from a passive writing aid into an active quality gatekeeper.

![](images/97163dd7e4c9fa67008fc7ca53cadfe8076b2a5ad9cab584ee17342a92954f6f.jpg)  
Figure 1: Comparison of citation paradigms: Traditional semantic retrieval versus our framework.

For years, citation recommendation has been modeled as a similarity search problem (Gu et al., 2022; Gao et al., 2023; Wang et al., 2025). Both early retrievers (Huang et al., 2014) and recent LLM-based agents (Çelik and Tekir, 2025; Zhang et al., 2026) equate citing with finding a textually relevant paper. In the LLM era, citation hallucinations (Algaba et al., 2025; Sakai et al., 2026) have crystallized into two distinct forms. Fabrication, inventing non-existent papers, is largely resolved by retrieval-augmented architectures; misattribution, citing authentic papers that fail to logically support the claim, exposes the core limit of similarity-based retrieval and remains open (Xu et al., 2025).

To address this, we argue that citation is fundamentally a reasoning task, not a retrieval one:

topical relevance does not guarantee logical support for a claim. For example, a claim about the Transformer’s success in NLP cannot be supported by the vision transformer paper (Dosovitskiy et al., 2021) despite their textual overlap; “Attention is All You Need” (Vaswani et al., 2017) is the correct logical pillar. Yet semantic retrievers (Cohan et al., 2020) ignore that different citation contexts, such as introducing a background versus comparing a method, demand distinct verification criteria (Cohan et al., 2019). Worse, current systems operate as unidirectional pipelines (Medic and Snajder´ , 2020), so a single faulty retrieval becomes an irrevocable error without opportunity for self-correction. Together, solving misattribution requires intent-aware retrieval, claim-level verification, and a reflective loop for error recovery.

To fulfill these requirements, we propose Re-Cite (Figure 1), an agentic reasoning framework for faithful citation, trained via supervised finetuning and reinforcement learning. It reshapes the citation process from a linear pipeline into an iterative architecture of reasoning, retrieval, verification, and reflection. Operating in an observethink-act format (Yao et al., 2023), the agent is built on Qwen3-4B (Yang et al., 2025), with key modules reinforced via Group Relative Policy Optimization (GRPO) (Shao et al., 2024). Within this framework, we instantiate three core mechanisms. First, claim-evidence verification tackles misattribution by cross-checking each candidate’s metadata against the inferred citation intent, accepting a citation only upon consistent evidence. Second, an intent-aware query planner addresses context misalignment by generating retrieval keywords conditioned on predicted citation intents. Third, a reflective re-retrieval loop overcomes single points of failure. Upon verification failure, the agent analyzes the cause, rewrites the query, and retries. Since retrieval queries only target authentic databases, hallucinating entirely non-existent papers is eliminated; ReCite then focuses entirely on resolving misattribution.

Experiments on the benchmark show that ReCite substantially outperforms strong generative baselines in strict citation accuracy. Ablations confirm that claim-evidence verification, intent-aware planning, and the reflective loop are indispensable to this gain. In summary, our main contributions are:

• We systematically decouple citation hallucinations into fabrication and misattribution, establishing misattribution as the core challenge in the LLM era and reframing citation as a reasoning task.

• We propose ReCite, the first closed-loop claim-evidence verification framework, integrating intent-aware query planning and a reflective re-retrieval loop.

• Experiment results highlight the inherent challenge of citation reasoning for current LLMs and demonstrate the effectiveness of ReCite.

## 2 Related Work

Local and Global Citation Recommendation. ACR has transitioned from global representations and two-stage pipelines (Beltagy et al., 2019; Medic and Snajder´ , 2020; Gu et al., 2022) to localized, context-aware generation (Çelik and Tekir, 2025; Buscaldi et al., 2024) and dynamic Retrieval-Augmented Generation (RAG) frameworks (Wang et al., 2025; Zhang et al., 2026). Although these generative models excel at contextual integration, relying on semantic similarity as a generative prior often bypasses strict logical entailment. Consequently, these systems frequently suffer from citation fidelity issues, including hallucinated references and misattributed claims (Algaba et al., 2025; Sakai et al., 2026; Çelik and Tekir, 2025).

Logical Verification and Agentic Reasoning. Standardized benchmarks such as ALCE (Gao et al., 2023) have begun to quantify citation reliability, prompting verification approaches based on Natural Language Inference (NLI) (Xu et al., 2025) or theorem provers (Pei et al., 2025). However, such static post-hoc filtering cannot correct errors that have propagated through a unidirectional generation pipeline. Autonomous frameworks (e.g., ReAct (Yao et al., 2023), Reflexion (Shinn et al., 2023)) and step-level reinforcement (e.g., GRPO (Li et al., 2026b)) introduce self-correction (Li et al., 2026a), but remain constrained by pre-trained priors and incur high context overhead. Current citation datasets only map local contexts to target papers; they omit the explicit reasoning traces required to train agents for this task. To build reasoning-driven citation models, the field requires a specialized dataset that supports location perception, intent reasoning, and closed-loop reflective trajectories.

![](images/c55e9c8204137e96af238789b2115d50d3c250258628bb4d2b5a095e89580597.jpg)  
Figure 2: The proportion distribution of academic conference sources and the various citation intent categories within the utilized citation dataset.

## 3 Dataset Construction

We construct a large-scale literature dataset for training an agentic citation workflow. Given a draft paragraph, the agent must (1) detect which positions require a citation, (2) infer each citation’s intent and generate targeted retrieval queries, and (3) verify whether retrieved candidates support the local claim, retrying with refined queries on failure. The dataset therefore provides three task-specific subsets that supply the supervision for the three corresponding modules of ReCite, namely CiteLocator, QueryPlanner, and Master Brain.

## 3.1 Data Source and Preparation

Our raw corpus consists of 10,893 LaTeX source packages crawled from arXiv, corresponding to papers published in 2024–2025 at top computer science venues, including ACL, EMNLP, CVPR, ICCV, and NeurIPS. Working at the LAT X source level preserves the explicit \cite commands and their bibliography keys, giving us ground-truth citation positions and references for downstream supervision.

We apply a four-step preparation pipeline: extracting introduction sections, normalizing heterogeneous citation macros (\cite, \citep, \citet, etc.) into a unified form, filtering non-textual noise, and removing paragraphs with abnormal citation density. Figure 2 shows the venue-level distribution of the resulting corpus. Compared with existing citation datasets such as RefSeer (Huang et al., 2014) and S2ORC (Lo et al., 2020), which provide only static context-to-reference mappings, our dataset additionally captures the location, intent, and reflective trajectories required to train an agentic citation workflow (Table 1).

<table><tr><td>Features</td><td>RefSeer</td><td>S2ORC</td><td>Ours</td></tr><tr><td>Local Context Mapping</td><td></td><td></td><td></td></tr><tr><td>Full-Text Structure</td><td>X</td><td></td><td></td></tr><tr><td>Location Perception</td><td>X</td><td>X</td><td></td></tr><tr><td>Intent Reasoning</td><td>X</td><td>X</td><td></td></tr><tr><td>Reflective Trajectories</td><td>X</td><td>X</td><td></td></tr></table>

Table 1: Comparison of our constructed trajectory dataset with existing citation datasets, including Ref-Seer (Huang et al., 2014) and S2ORC (Lo et al., 2020).
<table><tr><td colspan="2">Category Abbr. Description</td></tr><tr><td>Background BG</td><td>Provides background knowledge or infor- mation for the research domain.</td></tr><tr><td>Motivation MG &amp; Gap</td><td>Highlights research motivations or iden- tifies gaps in existing literature.</td></tr><tr><td>Comparison CMP</td><td>Compares the current work or other base- lines with specific prior methods.</td></tr><tr><td>Basis &amp; BS Support</td><td>Provides theoretical foundations or con- cepts supporting the current study.</td></tr><tr><td>Method &amp; MTD Data</td><td>Refers to specific methods, algorithms, tools, or datasets utilized in the research.</td></tr><tr><td>Validation VLD</td><td>Provides empirical evidence, results, or evaluation metrics to validate claims.</td></tr><tr><td>Evolution EV</td><td>Describes the historical evolution or de- velopment of a methodology.</td></tr><tr><td>Critique CRT</td><td>Expresses critiques, conflicts, or oppos- ing views regarding prior work.</td></tr></table>

Table 2: Detailed descriptions of the 8-category Citation Intent Taxonomy (CAP-8).

## 3.2 Citation Location Perception Data

We formulate citation location perception as a sequence restoration task. Given a paragraph with all original citations stripped, the task requires predicting a unified [#CITE#] marker at every legitimate citation position. Beyond position recovery, we use DeepSeek-V3 to annotate the necessity of each ground-truth marker, classifying it as either Mandatory (e.g., references to specific methods, algorithms, or datasets) or Optional (e.g., general domain background). By jointly supervising position and necessity, the dataset provides a more precise and context-aware location signal. To support this task, we construct 31,118 samples (comprising 28,006 for training and 3,112 for testing)

pairing the stripped input text with target outputsnreferenced Draft Input that mark every citation position together with its     tic model by optimizing relative advantages among sampled groups [#CITE#]... ... necessity label. A complete example is shown in <sub>Appendix C.1.</sub>O is ww<sub>PPO", "critic model", "advant</sub>

[Index 0]tle: Proximal Policy   
<sub><reasoning></sub>lgorithms (Schulman et   
The purpose of this citation is to provide an<sup>advantages across groups. refined</sup>   
example of how testing can reveal biases orOptimization", "eliminate critic"]milarity ScOore:is92w% (wHigh Overlap)   
spurious correlations learned by a model,   
supporting the claim about the value ofO is wwTitle: DeepSeekMath: Pushing the   
invariance testing. </reasoning><sup>Limits... (Shao et al.)</sup>Logical Mismatch   
<keywords>he draft critiques PPO   
[‘‘spurious correlations’’, ‘‘textO is wwFaithful Citation:Logically grou<sup>etriever</sup> <sup>cited PPO as the recent</sup>   
classification’’, ‘‘counterfactuals’’,   
<sub>‘‘robustness’’] </keywords></sub>Similarity-based RAG  
Listing 1: An example of the synthesized training data for QueryPlanner.

## 3.3 Query Planning Data

Moving beyond citation location, the QueryPlanner dataset focuses on semantic intent. Its objective is to bridge the vocabulary gap between a paragraph’s vague local context and the precise metadata of the target reference. Given a paragraph and a target citation position, the task is to generate both a reasoning chain that infers the citation intent and a keyword set that can retrieve the target reference. We isolate 7,079 high-density paragraphs (comprising 6,779 for training and 300 for testing) whose BibTeX entries contain complete titles and abstracts, and use DeepSeek-V3 in a posterior setting: with full access to the target reference’s metadata, it retroactively deduces a logical reasoning path (<reasoning>) and an optimal keyword set (<keywords>) for each citation (Listing 1). This posterior view yields keywords that effectively retrieve the target.

To verify keyword quality, we run a cross-model consistency check on 100 random samples. We substitute the keywords in 50 of these samples with keywords from unrelated instances to form negatives, and five LLMs binary-classify whether the keywords align with the target reference. The 90.6% average accuracy (Cohen’s κ = 0.805) confirms the reliability of the synthesized data. To analyze the logical distribution of training data, we run LLM-based motivation classification over 82,881 citation nodes; allowing secondary motivation labels expands the total to 93,403. This yields the 8-category Citation Intent Taxonomy (CAP-8), with category definitions in Table 2 and the corpus distribution in the intent panel of Figure 2.

![](images/840c1f4c4e02b57f7c845e4f01750dd24af8986158b3e031c7eab815db8583c8.jpg)  
Figure 3: Data Generation Pipeline for Trajectory Thinking Process.

## 3.4 Reflective Trajectory Synthesis

The third subset trains Master Brain to orchestrate the workflow and recover from retrieval failures. Each sample is a multi-turn trajectory in an observe-think-act format: the agent observes the environment state (<observe>), articulates a plan (<think>), and issues an action (<act>) such as a tool call or a final citation choice. To synthesize supervision, we fix <observe> and <act> from successful executions and use DeepSeek-V3 to retroactively deduce the connecting <think> (Figure 3). We construct 2,000 trajectories in total: 500 “gold” paths and 1,500 reflection-enhanced variants.

For the gold paths, we inject real distractors from the local database so that the model must state within <think> why it prefers the target reference over each alternative. For the reflection-enhanced variants, we inject vague initial keywords to simulate retrieval drift (e.g., returning irrelevant survey papers); the model must diagnose the failure, refine keywords from broad domains to specific entities, and issue a secondary retrieval before succeeding. A complete example is in Appendix C.2.

To guarantee robust generalization and prevent data leakage, we construct a separate test set for our end-to-end citation prediction evaluation. Specifically, we randomly sample 50 paragraphs from CVPR’25, EMNLP’24, ICCV’23, and NeurIPS’25, yielding 200 paragraphs in total. This evaluation set is strictly disjoint from our training data pool and contains 1,023 unique target test papers. An audit confirms that none of the 2,640 training citation decisions involved any of the 1,023 test papers.

## 4 Method

Figure 4 illustrates the framework of ReCite, and we detail each module below.

## 4.1 Logical Breakpoint Perception

CiteLocator predicts citation boundaries and their necessity labels (Mandatory or Optional). We finetune Qwen3-4B (Yang et al., 2025) via SFT on the location perception data, registering the unified [#CITE#] marker as a new tokenizer entry so that it is not fragmented during encoding. Because citation tokens are extremely sparse compared to regular text tokens, standard cross-entropy collapses to the majority class; we mitigate this with an asymmetric cross-entropy that places heavier loss weight on citation marker tokens, penalizing missed or hallucinated boundaries.

![](images/8fc5ee1e575a16b89c52875d9f7608cd1e19cac0ee3dba8594a145c469fe62b7.jpg)  
Figure 4: The overall architecture of our proposed ReCite framework. The system decouples the automatic citation process into three iterative stages: citation location perception (CiteLocator), intent-aware query planning (QueryPlanner), and task scheduling with reflective verification orchestrated by the Master Brain.

## 4.2 Intent-Aware Query Planning

QueryPlanner bridges the local context and precise retrieval metadata by generating a reasoning chain and a keyword set per citation position. We first fine-tune Qwen3-4B via SFT on the query planning data, then apply GRPO (Shao et al., 2024) to elicit deeper reasoning. For each query, GRPO samples a group of G = 4 candidate outputs and updates the policy with their relative advantages, eliminating the need for a separate value network. The composite reward combines three components: format compliance, which penalizes deviations from the required <reasoning>/<keywords> layout; entity recall, against the target reference; and intent alignment, against the ground-truth intent.

## 4.3 Task Orchestration

Master Brain is the central orchestrator of the agentic workflow. Like the other two modules, it is built on Qwen3-4B and trained via SFT on the reflective trajectory data to follow an observe-think-act state machine. At inference, Master Brain first invokes CiteLocator to identify all citation positions in the draft. For each position, it calls QueryPlanner to generate search keywords, queries the Semantic Scholar API to fetch candidate metadata (title, abstract, authors, year, citation count), and verifies whether the evidence supports the local claim. When no candidate provides consistent evidence, the agent enters a reflective re-retrieval loop: it emits a failure-attribution <think>, refines keywords from broad domains to specific entities, and issues a secondary retrieval. Because retrieval queries hit authentic databases, fabrication is impossible by construction; the residual challenge of misattribution is directly targeted by this verification and self-correction loop.

## 5 Experiments

## 5.1 Settings

Baselines The base model for all three modules is Qwen3-4B-Instruct (Yang et al., 2025). We compare our framework against three categories of baselines: (i) zero-shot LLM pipelines that mirror our workflow; (ii) General Agents (Claude Code (Anthropic, 2026), Hermes Agent (NousResearch, 2026), and OpenClaw (Steinberger, 2026)) running end-to-end on a Mimo-V2.5-Pro inference engine; and (iii) heuristic and human baselines for location-only evaluation (GM-s2orc, GM-s2orc-H, and human experts (Buscaldi et al., 2024)). The full LLM baseline list is in Appendix A; detailed system prompts are in Appendices C.3-C.5.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Size</td><td colspan="3">Overall-Strictly (%)</td><td colspan="3">Lenient Evaluation (%)</td><td colspan="3">Position-Only (%)</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>Models via zero-shot prompt pipeline</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GLM-4-Plus (GLM et al., 2024)</td><td></td><td>9.83</td><td>10.27</td><td>10.05</td><td>17.22</td><td>17.99</td><td>17.60</td><td>47.08</td><td>49.19</td><td>48.11</td></tr><tr><td>Mimo-V2.5-Pro (Xiaomi, 2026)</td><td></td><td>26.40</td><td>27.72</td><td>27.04</td><td>29.19</td><td>30.63</td><td>29.89</td><td>75.07</td><td>78.78</td><td>76.88</td></tr><tr><td>GPT-4o-mini (OpenAI et al., 2024)</td><td></td><td>7.20</td><td>8.65</td><td>7.86</td><td>12.73</td><td>15.29</td><td>13.89</td><td>44.89</td><td>53.90</td><td>48.98</td></tr><tr><td>GPT-5.1-Chat (Singh et al., 2025)</td><td></td><td>8.99</td><td>12.12</td><td>10.32</td><td>14.48</td><td>19.54</td><td>16.63</td><td>35.32</td><td>47.64</td><td>40.57</td></tr><tr><td>Kimi-k2-preview (Kimi et al., 2026)</td><td></td><td>19.06</td><td>21.24</td><td>20.09</td><td>24.32</td><td>27.10</td><td>25.64</td><td>55.86</td><td>62.24</td><td>58.88</td></tr><tr><td>Qwen3.6-Plus (Yang et al., 2025)</td><td></td><td>28.09</td><td>35.44</td><td>31.34</td><td>31.09</td><td>39.23</td><td>34.69</td><td>68.60</td><td>86.50</td><td>76.52</td></tr><tr><td>Qwen3.6-27B (Yang et al., 2025)</td><td>27B</td><td>22.89</td><td>29.62</td><td>25.83</td><td>25.58</td><td>33.08</td><td>28.85</td><td>62.34</td><td>80.48</td><td>70.26</td></tr><tr><td>DeepSeek-V4-Flash (DeepSeek-AI, 2026)</td><td>284B</td><td>31.23</td><td>35.21</td><td>33.10</td><td>34.52</td><td>38.89</td><td>36.57</td><td>72.95</td><td>82.11</td><td>77.26</td></tr><tr><td>DeepSeek-V4-Pro (DeepSeek-AI, 2026)</td><td>1.6T</td><td>31.29</td><td>32.28</td><td>31.77</td><td>35.25</td><td>36.34</td><td>35.79</td><td>75.97</td><td>78.32</td><td>77.13</td></tr><tr><td>General Agents (End to End via Mimo-V2.5-Pro)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Code (Anthropic, 2026)</td><td></td><td>17.73</td><td>27.26</td><td>21.49</td><td>21.35</td><td>32.82</td><td>25.87</td><td>25.51</td><td>39.23</td><td>30.92</td></tr><tr><td>Hermes Agent (NousResearch, 2026)</td><td></td><td>22.39</td><td>21.24</td><td>21.80</td><td>26.38</td><td>25.02</td><td>25.68</td><td>28.01</td><td>26.56</td><td>27.27</td></tr><tr><td>OpenClaw (Steinberger, 2026)</td><td></td><td>18.09</td><td>7.03</td><td>10.12</td><td>25.25</td><td>9.81</td><td>14.13</td><td>29.42</td><td>11.43</td><td>16.46</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReCite-Base</td><td>4B</td><td>36.92</td><td>35.75</td><td>36.33</td><td>43.46</td><td>42.08</td><td>42.76</td><td>89.80</td><td>86.11</td><td>87.53</td></tr><tr><td>ReCite-SFT</td><td>4B</td><td>37.48</td><td>37.45</td><td>37.47</td><td>47.14</td><td>47.07</td><td>47.10</td><td>89.80</td><td>89.66</td><td>89.73</td></tr><tr><td>ReCite-SFT(CAP-8)</td><td>4B</td><td>39.22</td><td>39.07</td><td>39.15</td><td>55.27</td><td>55.02</td><td>55.14</td><td>89.92</td><td>89.51</td><td>89.71</td></tr></table>

Table 3: End-to-end citation prediction evaluation results. We report Precision (P), Recall (R), and F1-score across three metrics: Strict End-to-End, Lenient Evaluation, and Position-Only. CAP-8 denotes the model fine-tuned with citation taxonomy information. Bold indicates the best result in each column.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td rowspan="2">Strategy</td><td colspan="3">Mandatory (%)</td><td colspan="3">Optional (%)</td><td colspan="3">Overall (%)</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>Baseline Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.1-Chat (Singh et al., 2025)</td><td></td><td>Direct</td><td>22.01</td><td>59.72</td><td>32.17</td><td>9.21</td><td>58.50</td><td>15.92</td><td>27.73</td><td>59.39</td><td>37.81</td></tr><tr><td>GPT-4o-mini (OpenAI et al., 2024)</td><td></td><td>Direct</td><td>24.62</td><td>53.61</td><td>33.75</td><td>11.34</td><td>57.17</td><td>18.92</td><td>31.25</td><td>54.57</td><td>39.74</td></tr><tr><td>Kimi-k2-preview (Kimi et al., 2026)</td><td></td><td>Direct</td><td>31.27</td><td>61.05</td><td>41.36</td><td>14.36</td><td>61.28</td><td>23.27</td><td>38.38</td><td>61.11</td><td>47.15</td></tr><tr><td>MiniMax-M2.5 (MiniMax-AI, 2026)</td><td></td><td>Direct</td><td>33.52</td><td>56.37</td><td>42.04</td><td>14.71</td><td>52.52</td><td>22.98</td><td>40.35</td><td>55.34</td><td>46.67</td></tr><tr><td>GLM-4-flash (GLM et al., 2024)</td><td></td><td>Direct</td><td>25.81</td><td>16.73</td><td>20.30</td><td>12.07</td><td>17.96</td><td>14.43</td><td>32.67</td><td>17.06</td><td>22.41</td></tr><tr><td>Qwen3-Max (Yang et al., 2025)</td><td></td><td>Direct</td><td>32.87</td><td>28.90</td><td>30.75</td><td>15.41</td><td>29.28</td><td>20.19</td><td>40.18</td><td>29.00</td><td>33.69</td></tr><tr><td>Qwen3-Max (Yang et al., 2025)</td><td></td><td>CoT</td><td>34.43</td><td>28.42</td><td>31.14</td><td>16.00</td><td>28.09</td><td>20.39</td><td>41.71</td><td>28.33</td><td>33.74</td></tr><tr><td>DeepSeek-V3 (DeepSeek-AI et al., 2025)</td><td>671B</td><td>Direct</td><td>30.91</td><td>62.66</td><td>41.40</td><td>13.91</td><td>61.65</td><td>22.70</td><td>37.85</td><td>62.39</td><td>47.12</td></tr><tr><td>DeepSeek-V3 (DeepSeek-AI et al., 2025)</td><td>671B</td><td>CoT</td><td>29.23</td><td>49.09</td><td>36.64</td><td>12.41</td><td>45.88</td><td>19.54</td><td>35.68</td><td>48.23</td><td>41.02</td></tr><tr><td>Qwen3.5-27B (Yang et al., 2025)</td><td>27B</td><td>Direct</td><td>36.13</td><td>55.86</td><td>43.88</td><td>15.83</td><td>50.56</td><td>24.11</td><td>42.98</td><td>54.44</td><td>48.04</td></tr><tr><td>Qwen3-4B (Yang et al., 2025)</td><td>4B</td><td>Direct</td><td>13.98</td><td>9.70</td><td>11.45</td><td>5.55</td><td>9.56</td><td>7.03</td><td>18.12</td><td>9.66</td><td>12.60</td></tr><tr><td>Qwen3-4B (Yang et al., 2025)</td><td>4B</td><td>CoT</td><td>8.98</td><td>3.99</td><td>5.52</td><td>2.59</td><td>2.92</td><td>2.74</td><td>11.12</td><td>3.70</td><td>5.55</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CiteLocator</td><td>4B</td><td>Direct</td><td>67.93</td><td>60.61</td><td>64.06</td><td>42.90</td><td>58.57</td><td>49.52</td><td>74.16</td><td>60.06</td><td>66.37</td></tr></table>

Table 4: Citation location prediction results on the test set (N = 3, 112). We report Precision (P), Recall (R), and F1-score for Mandatory, Optional, and Overall categories. Bold indicates the best result in each column.

Evaluation Metrics Metrics strictly align with our architecture. CiteLocator uses Precision (P), Recall (R), and F1-score for Mandatory and Optional citations. QueryPlanner uses Token F1 (R-F1) for reasoning, alongside Token F1/Recall (K-F1, K-TR), Phrase Recall (K-PR), Jaccard Similarity (K-Jac), and an LLM-as-a-judge mechanism for keyword quality. End-to-end workflows are evaluated at three strictness levels: Overall-Strictly (exact target match), Lenient (relevant background/baselines), and Position-Only. Details on the evaluation metric definitions and calculations are provided in Appendix B.

Parameters For SFT, we use an 8-bit Paged AdamW optimizer (learning rate $2 \times 1 0 ^ { - 5 } .$ , 3 epochs, cosine decay, 5% warmup). Maximum sequence lengths are dynamically set: 1,024 for CiteLocator, 4,096 for QueryPlanner, and 9,216 for the Master Brain to preserve multi-turn trajectories. In the GRPO phase for QueryPlanner, the learning rate is $5 \times 1 0 ^ { - 7 }$ with a sample group size of G = 4.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="2">Reasoning</td><td colspan="5">Keyword Extraction</td></tr><tr><td>R-F1 (%)</td><td>R-Jdg</td><td>K-F1 (%)</td><td>K-TR (%)</td><td>K-PR (%)</td><td>K-Jac (%)</td><td>K-Jdg</td></tr><tr><td colspan="10">Baseline Models</td></tr><tr><td>GPT-5.1-Chat (Singh et al., 2025)</td><td></td><td>52.25</td><td>7.48</td><td>39.05</td><td>44.03</td><td>12.16</td><td>7.87</td><td>6.26</td></tr><tr><td>MiniMax-M2.5 (MiniMax-AI, 2026)</td><td></td><td>53.88</td><td>7.86</td><td>41.19</td><td>47.19</td><td>13.73</td><td>8.75</td><td>6.53</td></tr><tr><td>Qwen3-Max (Yang et al., 2025)</td><td></td><td>53.91</td><td>7.33</td><td>37.01</td><td>36.56</td><td>15.38</td><td>10.83</td><td>5.95</td></tr><tr><td>GLM-4-flash (GLM et al., 2024)</td><td></td><td>21.73</td><td>3.04</td><td>16.09</td><td>19.95</td><td>6.65</td><td>3.98</td><td>2.63</td></tr><tr><td>Kimi-k2-preview (Kimi et al., 2026)</td><td></td><td>54.12</td><td>7.75</td><td>45.80</td><td>55.30</td><td>25.45</td><td>16.42</td><td>7.12</td></tr><tr><td>DeepSeek-V3 (DeepSeek-AI et al., 2025)</td><td>671B</td><td>58.63</td><td>7.88</td><td>46.25</td><td>49.18</td><td>22.59</td><td>15.13</td><td>6.65</td></tr><tr><td>Qwen3.5-27B (Yang et al., 2025)</td><td>27B</td><td>50.55</td><td>7.56</td><td>40.99</td><td>43.62</td><td>16.73</td><td>11.14</td><td>6.33</td></tr><tr><td colspan="9">Ours</td></tr><tr><td>QueryPlanner(SFT)</td><td>4B</td><td>56.12</td><td>7.21</td><td>37.74</td><td>43.56</td><td>16.87</td><td>10.38</td><td>5.93</td></tr><tr><td>QueryPlanner(SFT+GRPO)</td><td>4B</td><td>57.81</td><td>7.64</td><td>39.06</td><td>42.47</td><td>17.85</td><td>11.00</td><td>6.44</td></tr></table>

Table 5: Comprehensive evaluation results about intent-aware query planning on the test set (N = 300). Bold indicates the best result in each column.

## 5.2 Main Results

As shown in Table 3, ReCite-SFT(CAP-8) achieves the highest Strict F1 of 39.15%, outperforming every baseline including DeepSeek-V4-Pro and Qwen3.6-Plus, demonstrating the effectiveness of our decoupled agentic design. This margin is especially notable despite the larger baselines likely benefiting from data leakage, since their pretraining corpora almost certainly cover our 2024- 2025 test papers. In addition, adding CAP-8 over ReCite-SFT yields a clear gain on both Strict and Lenient F1, showing that fine-grained intent supervision substantially improves candidate verification. By comparison, General Agents (Claude Code, Hermes Agent, OpenClaw) fall well behind, indicating that black-box end-to-end agents cannot recover from retrieval drift, and that a task-specific agent design remains necessary in this setting.

While ReCite achieves state-of-the-art results, its Strict F1 (39.15%) is heavily constrained by the severity of exact-match evaluation, which requires perfect joint accuracy across sequential stages and is confounded by subjective author biases. Since scientific claims can often be validated by multiple appropriate papers, the 15.99% gain in Lenient F1 (55.14%) demonstrates ReCite’s ability to discover logically supportive alternatives that are semantically valid but not explicitly cited by the original authors (a detailed qualitative case study is provided in Appendix D).

## 5.3 Module-Level Evaluation

Beyond end-to-end performance, we also evaluate CiteLocator and QueryPlanner individually against module-specific baselines.

CiteLocator. On location prediction (Table 4), CiteLocator outperforms all LLM baselines by a wide margin, reaching Overall F1 of 66.37% versus 48.04% for the next-best baseline (Qwen3.5- 27B), with consistent gains across both Mandatory and Optional categories. On the out-of-domain BUSCALDI benchmark (Table 6), which categorizes tasks into coarse-grained (sentence-level necessity) and fine-grained (token-level placement) predictions, CiteLocator significantly narrows the gap to human experts on exact location prediction. By outperforming GM-s2orc-H on Fine-grained, it demonstrates that our learned rhetorical signals generalize effectively beyond the training corpus.

<table><tr><td>Type</td><td>Method</td><td>P(%)</td><td>R(%)</td><td>F1 (%)</td></tr><tr><td rowspan="4">Coarse- grained</td><td>Scientist</td><td>76.2</td><td>88.4</td><td>81.8</td></tr><tr><td>GM-s2orc</td><td>78.2</td><td>78.2</td><td>78.2</td></tr><tr><td>GM-s2orc-H</td><td>80.2</td><td>82.6</td><td>81.4</td></tr><tr><td>CiteLocator</td><td>66.7</td><td>87.0</td><td>75.5</td></tr><tr><td rowspan="4">Fine- grained</td><td>Scientist</td><td>57.5</td><td>66.6</td><td>61.7</td></tr><tr><td>GM-s2orc</td><td>44.9</td><td>44.9</td><td>44.9</td></tr><tr><td>GM-s2orc-H</td><td>49.2</td><td>50.7</td><td>50.0</td></tr><tr><td>CiteLocator</td><td>46.7</td><td>60.9</td><td>52.8</td></tr></table>

Table 6: Performance comparison with existing baselines on the citation prediction task. The “-H” suffix indicates methods utilizing extra heuristic rules.

QueryPlanner. On reasoning and keyword generation (Table 5), QueryPlanner-SFT alone reaches Token F1 of 56.12%, approaching the 58.63% of DeepSeek-V3 (671B) and clearly beating the 27B open-source baseline. Adding GRPO further lifts reasoning quality and most keyword metrics, with the composite reward (format + entity recall + intent alignment) pushing the model from surface copying to deeper entity grounding.

## 5.4 Ablation Study

To validate our decoupled framework, we individually replace CiteLocator, QueryPlanner, and the

![](images/fd3084d1fcb728b34fe7b1b5e291fd924de426246a43087061cd02266228bf37.jpg)  
Figure 5: Performance comparison of various models across different citation intent categories. The evaluation is based on the Recall metric within the CAP-8 taxonomy.

<table><tr><td>Method</td><td>P (%)</td><td>R(%)</td><td>F1 (%)</td></tr><tr><td colspan="4">Overall-Strictly</td></tr><tr><td>ReCite</td><td>37.48</td><td>37.45</td><td>37.47</td></tr><tr><td>w/o CiteLocator</td><td>2.98 (-34.50)</td><td>2.55 (-34.90)</td><td>2.75 (-34.72)</td></tr><tr><td>w/o QueryPlanner</td><td>15.79 (-21.69)</td><td>15.75 (-21.70)</td><td>15.77 (-21.70)</td></tr><tr><td>w/o Master Brain</td><td>36.92 (-0.56)</td><td>35.75 (-1.70)</td><td>36.33 (-1.14)</td></tr><tr><td colspan="4">Lenient Evaluation</td></tr><tr><td>ReCite</td><td>47.14</td><td>47.07</td><td>47.10</td></tr><tr><td>w/o CiteLocator</td><td>7.05 (-40.09)</td><td>6.02 (-41.05)</td><td>6.49 (-40.61)</td></tr><tr><td>w/o QueryPlanner</td><td>30.88 (-16.26)</td><td>30.81 (-16.26)</td><td>30.85 (-16.25)</td></tr><tr><td>w/o Master Brain</td><td>43.46 (-3.68)</td><td>42.08 (-4.99)</td><td>42.76 (-4.34)</td></tr></table>

Table 7: Ablation study on different components.

Master Brain with the Qwen3-4B-Instruct baseline (Table 7). Replacing CiteLocator causes a severe drop in Strict F1 to 2.75%, proving that logical breakpoint perception is the foundational prerequisite. Similarly, the absence of QueryPlanner lowers Strict F1 to 15.77%, highlighting the necessity of intent-aware query planning. Also, removing the Master Brain’s reflective verification degrades the Lenient F1 from 47.10% to 42.76%, indicating its crucial role in filtering out noise. Ultimately, the complete system achieves optimal performance, demonstrating that these modules are mutually reinforcing and indispensable.

## 5.5 In-depth Analysis

Analysis of Cross-Venue Data Robustness Figure 6 breaks down end-to-end F1 by the source conference of each test paragraph. ReCite stays balanced across all four sub-domains, while several general-purpose baselines fluctuate or skew toward specific venues. This suggests that ReCite captures domain-general rhetorical signals rather than surface vocabulary overlap, supporting crossdomain transfer within computer science.

Analysis of Intent-Level Recall Figure 5 reports recall across the CAP-8 intent categories. ReCite achieves higher recall than most baselines across categories, indicating that explicit intent supervision helps the agent align retrieval with author motivation rather than relying on semantic overlap.

![](images/9da39994f67611c1918c07a4d75b14b68c07f36c53c23a2513841051be8b34e5.jpg)  
Figure 6: Performance comparison of different models based on the Recall and F1-Score.

![](images/eee9e64efec6b9c3bf8dc10c2b88f4da88737ee418cf38655b333b52ccd65742.jpg)  
Figure 7: Performance variation of the proposed system and the baseline across different numbers of retrieved candidate papers (Top-k).

Analysis of Retrieval Candidate Quantity As k varies from 4 to 12 (Figure 7), the end-to-end F1 scores of both ReCite and the DeepSeek-V4-Flash baseline improve up to k = 8, after which further improvements are limited by context noise. Moreover, a larger k inflates input length and inference cost; we therefore fix k = 8 as the optimal trade-off between evidence sufficiency and latency.

## 6 Conclusion

In this paper, we introduced ReCite, a decoupled agentic reasoning framework for faithful citation. ReCite integrates citation location perception, intent-aware query planning, and reflective verification to address misattribution while eliminating fabricated references through retrieval from authentic academic databases. To train these capabilities, we constructed a large-scale reasoning-oriented dataset from LAT X sources and synthesized supervision for location prediction, citation intent inference, keyword planning, and self-corrective trajectories. Extensive experiments show that Re-Cite achieves stronger strict citation accuracy than substantially larger generative models and generalpurpose agents, demonstrating the effectiveness of our lightweight modules. Ablation and intent-level analyses confirm that each component contributes to robust citation grounding. By establishing citation as verifiable reasoning, ReCite provides a practical foundation for AI-assisted academic writing that is more faithful, accountable, and trustworthy.

## Limitations

While ReCite demonstrates strong performance in specific citation tasks, several limitations remain. First, our specialized lightweight model inherently trails massive general-purpose LLMs in open-ended reasoning outside the targeted workflow. Second, the system currently operates exclusively on uni-modal text. Since real-world citations often rely on charts or pseudocode, future work will integrate Vision-Language Models (VLMs) to support multimodal verification. Third, because the framework was trained entirely on computer science papers, its cross-disciplinary generalization (e.g., to the humanities or biomedical fields) remains unexplored and requires systematic evaluation.

## Ethical Considerations

Our research directly addresses a critical ethical issue in AI-assisted scientific writing: the fabrication of references, which severely misleads readers and undermines academic integrity. By enforcing a rigorous verification loop grounded in real-world academic databases, our framework mitigates the propagation of hallucinated knowledge.

We recognize the ethical concerns surrounding data privacy and intellectual property. Academic research often involves highly confidential, unpublished data, and relying on closed-source, cloudbased LLMs poses inherent compliance and data leakage risks. A core motivation for developing our framework on a lightweight 4B parameter model is to facilitate future model quantization and pruning. This paves the way for fully offline, on-device deployment, ensuring that researchers can leverage agentic citation assistance without compromising sensitive intellectual property.

## Acknowledgements

We acknowledge the use of AI assistants during the writing and coding phases of this research. These tools were utilized solely to refine linguistic phrasing and assist with coding, while all scientific concepts, experimental designs, and final analyses were developed entirely by the authors.

## References

Andres Algaba, Carmen Mazijn, Vincent Holst, Floriano Tori, Sylvia Wenmackers, and Vincent Ginis. 2025. Large language models reflect human citation patterns with a heightened citation bias. In Findings ofNAACL, pages 6844–6879.

Anthropic. 2026. Claude code.

Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. SciBERT: A pretrained language model for scientific text. In Proceedings ofEMNLP, pages 3615–3620.

Lutz Bornmann and Rüdiger Mutz. 2015. Growth rates of modern science: A bibliometric analysis based on the number of publications and cited references. Journal of the association for information science and technology, 66(11):2215–2222.

Davide Buscaldi, Danilo Dessí, Enrico Motta, Marco Murgia, Francesco Osborne, and Diego Reforgiato Recupero. 2024. Citation prediction by leveraging transformers and natural language processing heuristics. Information Processing & Management, 61(1):103583.

Ege Yigit Çelik and Selma Tekir. 2025. CiteBART:˘ Learning to generate citations for local citation recommendation. In Proceedings of EMNLP, pages 1703–1719.

Arman Cohan, Waleed Ammar, Madeleine van Zuylen, and Field Cady. 2019. Structural scaffolds for citation intent classification in scientific publications. In NAACL-HLT, pages 3586–3596.

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel Weld. 2020. SPECTER: Document-level representation learning using citation-informed transformers. In Proceedings of ACL, pages 2270–2282.

DeepSeek-AI. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, and 181 others. 2025. Deepseek-v3 technical report. Preprint, arXiv:2412.19437.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings ofICLR.

Santo Fortunato, Carl T Bergstrom, Katy Börner, James A Evans, Dirk Helbing, Staša Milojevic,´ Alexander M Petersen, Filippo Radicchi, Roberta Sinatra, Brian Uzzi, and 1 others. 2018. Science of science. Science, 359(6379):eaao0185.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings ofEMNLP, pages 6465–6488.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, and 39 others. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. Preprint, arXiv:2406.12793.

Nianlong Gu, Yingqiang Gao, and Richard HR Hahnloser. 2022. Local citation recommendation with hierarchical-attention text encoder and scibert-based reranking. In European conference on information retrieval, pages 274–288. Springer.

Wenyi Huang, Zhaohui Wu, Prasenjit Mitra, and C. Lee Giles. 2014. Refseer: A citation recommendation system. IEEE/ACM Joint Conference on Digital Libraries, pages 371–374.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12):248:1–248:38.

Team Kimi, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du,

and 181 others. 2026. Kimi k2: Open agentic intelligence. Preprint, arXiv:2507.20534.

Bobo Li, Rui Wu, Zibo Ji, Meishan Zhang, Hao Fei, Min Zhang, Mong-Li Lee, and Wynne Hsu. 2026a. Taming actor-observer asymmetry in agents via dialectical alignment. In Proceedings of ACL, pages 24068–24084.

Weijie Li, Jin Wang, Liang-Chih Yu, and Xuejie Zhang. 2026b. Step-grpo: Enhancing reasoning quality and efficiency via structured prm-based reinforcement learning. In Proceedings ofAAAI.

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Michael Kinney, and Daniel S. Weld. 2020. S2orc: The semantic scholar open research corpus. In Proceedings ofACL.

Zoran Medic and Jan Snajder. 2020. Improved local´ citation recommendation based on context enhanced with global information. In Proceedings ofthe First Workshop on Scholarly Document Processing, pages 97–103.

MiniMax-AI. 2026. Minimax m2.5 large language model.

NousResearch. 2026. Hermes agent documentation.

OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛adry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, and 400 others. 2024. Gpt-4o system card. Preprint, arXiv:2410.21276.

Yu Pei, Yongping Du, and Xingnan Jin. 2025. Fover: First-order logic verification for natural language reasoning. Trans. Assoc. Comput. Linguistics, 13:1340– 1359.

Yusuke Sakai, Hidetaka Kamigaito, and Taro Watanabe. 2026. Hallucitation matters: Revealing the impact of hallucinated references with 300 hallucinated papers in ACL conferences. CoRR, abs/2601.18724.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Advances in NeurIPS, pages 8634–8652.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry,

Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, and 465 others. 2025. Openai gpt-5 system card. Preprint, arXiv:2601.03267.

Peter Steinberger. 2026. Openclaw.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of NeurIPS, pages 5998– 6008.

Yubo Wang, Xueguang Ma, Ping Nie, Huaye Zeng, Zhiheng Lyu, Yuxuan Zhang, Benjamin Schneider, Yi Lu, Xiang Yue, and Wenhu Chen. 2025. Scholarcopilot: Training large language models for academic writing with accurate citations. Preprint, arXiv:2504.00824.

Xiaomi. 2026. Mimo-v2.5.

Yilong Xu, Jinhua Gao, Xiaoming Yu, Baolong Bi, Huawei Shen, and Xueqi Cheng. 2025. ALiiCE: Evaluating positional fine-grained citation generation. In Proceedings ofNAACL-HLT, pages 545–561.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In ICLR.

Yao Zhang, Yude Bai, Minhong Dong, Keqing Cen, Ji Zhang, Qiang Hu, Wei Ma, Yongqiang Lyu, Ruitao Feng, Xiaohong Li, Junjie Wang, Lingxiao Jiang, and Yang Liu. 2026. Limr: Intent-aware mashup api recommendation via llm-augmented multi-scale fusion. IEEE Transactions on Services Computing.

## A Experiment Settings

Zero-shot LLM Baselines. For end-to-end and module-level evaluation we benchmark against proprietary models (GPT-4o-mini (OpenAI et al., 2024), GPT-5.1-Chat (Singh et al., 2025), Kimi-k2- preview (Kimi et al., 2026), GLM-4-Plus, GLM-4- Flash (GLM et al., 2024), Qwen3.6-Plus, Qwen3- Max (Yang et al., 2025), Mimo-V2.5-Pro (Xiaomi, 2026), MiniMax-M2.5 (MiniMax-AI, 2026)) and open-source LLMs (Qwen3.5-27B, Qwen3- 4B (Yang et al., 2025), DeepSeek-V3 (DeepSeek-AI et al., 2025), DeepSeek-V4-Flash, DeepSeek-V4-Pro (DeepSeek-AI, 2026)), all evaluated zeroshot under a prompt pipeline mirroring our workflow.

General Agents. The three end-to-end agents (Claude Code, Hermes Agent, OpenClaw) receive a single end-to-end citation objective and orchestrate their own tool calls; for fair comparison, all three run on Mimo-V2.5-Pro.

Heuristic and Human Baselines. For locationonly evaluation, we compare against GM-s2orc, a GPT-2 based generative model fine-tuned on the s2orc dataset, and its variant GM-s2orc-H, which incorporates additional post-hoc NLP heuristics. Scientist denotes human annotator performance from (Buscaldi et al., 2024). To evaluate out-ofdomain generalization, we utilize their introduced gold standard dataset, which comprises 133 sentences randomly sampled from recent arXiv papers and strictly annotated by three senior researchers.

Implementation Details. We train and evaluate ReCite on NVIDIA RTX 4090 D and A800 GPUs, with DeepSpeed for distributed training and vLLM (PagedAttention) for high-throughput inference.

## B Details of Evaluation Metrics

We detail the evaluation metrics, strictly aligning with the three modules of the ReCite framework.

## B.1 Main Results

The end-to-end evaluation assesses the final citation mapping at three strictness levels:

• Overall-Strictly: Requires both correct identification of the citation position AND successful retrieval of the exact ground-truth paper.

• Lenient Evaluation: Accepts highly relevant substitute papers, acknowledging that multiple papers can logically support a claim. This is assessed via an LLM-as-a-Judge that outputs a binary score (1 for academically appropriate, 0 otherwise) based on local context and candidate metadata.

• Position-Only: Evaluates structural perception alone. A prediction is correct if it successfully identifies the logical breakpoint, regardless of the retrieved candidate.

## B.2 Citation Location Perception (CiteLocator)

We evaluate the generated [#CITE#] placeholders against ground-truth breakpoints using standard Precision (P), Recall (R), and F1-score (F1):

$$
P = { \frac { T P } { T P + F P } } , \quad R = { \frac { T P } { T P + F N } }\tag{1}
$$

$$
F 1 = \frac { 2 \times P \times R } { P + R }\tag{2}
$$

These metrics are computed separately for Mandatory (specific evidentiary backing) and Optional (general background) citations.

## B.3 Intent-Aware Query Planning (QueryPlanner)

To evaluate the QueryPlanner, we utilize the following metrics:

• Token Metrics (R-F1, K-F1, K-TR): Unigram F1 overlap for the generated <reasoning> (R-F1) and <keywords> (K-F1), along with the keyword Token Recall (K-TR), evaluated against the ground-truth.

• Phrase Recall (K-PR): The proportion of ground-truth core entities successfully extracted in the <keywords> list.

• Jaccard Similarity (K-Jac): The intersection over union of the predicted $( K _ { p r e d } )$ and ground-truth $( K _ { g t } )$ keyword sets:

$$
{ \mathrm { K - J a c } } = { \frac { | K _ { p r e d } \cap K _ { g t } | } { | K _ { p r e d } \cup K _ { g t } | } }\tag{3}
$$

• LLM-as-a-Judge: An advanced LLM scores the keyword quality (0-10) based on logical soundness and retrieval feasibility.

```ini
[System Prompt]
You are an academic paper selector. Your goal
is to match the exact foundational paper needed
for the specific citation context.
[User Prompt]
You need to select the best citation for the
marker [Index {index}] (which corresponds to
the {index+1}-th [#CITE#] marker in the text).
ORIGINAL CONTEXT:
{marked_text}
Your reasoning for this citation was:
{reasoning}
```

```markdown
You are an expert academic citation analyst.
### Task
Analyze the context of the given text. For each
citation marker [#CITE#], infer what keywords
should be used to search for the paper that
needs to be cited. Each [#CITE#] marker in the
input must correspond to one [Index n] in the
output in order.
### Output Components
1. <reasoning>: A 1-2 sentence logical analysis
of the context explaining the core role of the
citation. MUST start with the fixed phrase:
‘‘The purpose of this citation is to...’’
2. <keywords>: 3-5 precise keywords to retrieve
the target document.
### Constraints
- Sequential Indexing: Start with [Index 0] and
increment by 1 for each marker.
- Output ONLY the structured content. Strictly
NO conversational filler.
### Output Format
[Index X]
<reasoning>...</reasoning>
<keywords>[‘‘keyword 1’’, ‘‘keyword 2’’,
...]</keywords>
```

## C Data Examples and System Prompts

## C.1 Examples of Location Perception Data

Listing 2: Example input and output for citation location   
perception data.   
input: Most state-of-the-art FSRL methods adopt   
a meta-learning framework, where a base model   
is pre-trained... they predominantly rely on KG   
embedding models like TransE or DistMult to   
exploit relational information.   
output: Most state-of-the-art FSRL methods   
adopt a meta-learning framework [#CITE#], where   
a base model is pre-trained... they   
predominantly rely on KG embedding models like   
TransE [#CITE#] or DistMult [#CITE#] to exploit   
relational information.

## C.2 Example of Reflective Trajectory Data

Listing 3: Example of multi-turn reflective trajectory data synthesis, demonstrating self-correction.

=== Turn 3 Assistant ===   
<think>   
The initial intent analysis suggested keywords   
such as reinforcement learning feedback’’. I   
will use these to search for candidate papers   
for index 0.   
</think>   
<act>   
{name’’: search\_db\_single’’,arguments’’:   
{index’’: 0,keywords’’: [‘‘reinforcement   
learning feedback’’]}}   
</act>   
=== Turn 4 User ===   
<observe>   
Result from search\_db\_single: ... (No specific   
match for LLM alignment or RLHF found.)   
</observe>   
=== Turn 5 Assistant ===   
<think>   
The returned papers focus on robotics and   
driving, which are irrelevant to the current   
context of LLM alignment. This indicates the   
initial keywords were too general, leading to   
search drift. I will reformulate the search to   
be more precise.   
</think>   
<act>   
{name’’:search\_db\_single’’, arguments’’:   
{index’’: 0, keywords’’: [token-wise dense   
feedback RLHF’’, ‘‘RL alignment dense   
rewards’’]}}   
</act>

## C.3 System Prompts for Pipeline Baselines

To ensure reproducibility and fair comparison, we provide the exact zero-shot system prompts used for our three-stage pipeline baselines.

Listing 4: Zero-shot Prompt for Citation Location Per  
ception (CiteLocator Pipeline).

```ini
[System Prompt]
You are a professional academic copy-editor.
[User Prompt]
Task: Act as a professional academic
copy-editor. Insert the specified marker
[#CITE#] at places that require citations, such
as when mentioning prior works, existing
methods, algorithms, models, datasets or other
places where citations are needed.
Rules: Do not alter any original content. Do
not add extra preamble. Output only the
processed text.
Input Text:{text}
```

Listing 5: Zero-shot Prompt for Intent-Aware Query Planning (QueryPlanner Pipeline).

Listing 6: Zero-shot Prompt for Candidate Verification (Search and Submit Pipeline).

Here are the candidate papers from the database:   
{candidate papers}   
Evaluate carefully based on the original   
context. If you find the correct foundational   
paper, reply with ONLY its ID. If NONE of these   
papers match the context, reply with exactly   
‘‘NONE’’.

## C.4 System Prompts for General Agents

Listing 7: System Prompt for Autonomous General Agents (End-to-End via Mimo-V2.5-Pro). Agents (End-to-End via Mimo-V2.5-Pro).

You are an automatic citation assistant. Your   
capability is to identify places in given text   
paragraphs that require academic citations,   
autonomously construct keywords to retrieve   
real academic papers, and strictly follow the   
rules to add standard citations. It is strictly   
forbidden to search the original source article   
to copy existing real citations, or to generate   
fake papers.   
STRICT CITATION RULES:   
Citation Judgment: Identify positions that   
require citations, such as when mentioning   
prior works, existing methods, algorithms,   
models, datasets, or other places where   
citations are needed.   
Literature Retrieval: Actively use keywords to   
retrieve real literature for matching citations.   
Citation Placement Rule: Insert exactly one   
ordered citation mark (e.g., [1], [2], [3]...)   
for each independent sentence or academic   
claim. Multiple citations stacking at the same   
position are strictly prohibited.   
Bibliography Format: Append an exact line   
titled ‘‘Bibliography:’’ at the very end of the   
modified full text. Each bibliography entry   
follows the fixed format: [Number] First   
Author, ‘‘Exact Full Paper Title’’   
No Extra Content: Do not add any redundant   
explanations, comments, or modification notes   
in the output text.   
FIXED OUTPUT FORMAT (MANDATORY):   
Output a single-line valid JSON object for each   
processed entry, containing only two fields   
with no extra content:   
{‘‘text’’:‘‘Full modified text with inserted   
citations and final Bibliography section’’,   
‘‘bibliography’’:[‘‘First Author, \‘‘Exact Full   
Paper Title\’’]}   
HARD CONSTRAINTS:   
Never modify the original input file in any way.   
All cited literature must be real and   
retrievable; fake literature and copying   
original citations are strictly prohibited.   
Strictly follow the ‘‘one sentence, one   
citation’’ rule; no citation stacking or   
redundant citation marking.

## C.5 System Prompt for Master Brain Agent

Listing 8: System Prompt for the Master Brain Agent (Task Orchestration and Reflective Verification).

You are an Autonomous Academic Agent capable of   
reflection and self-correction. You process   
text by calling functions. You have access to   
the following functions: {functions}   
CRITICAL RULES:   
1. Sequential Workflow: mark\_citations ->   
analyze\_intents -> (Loop starts) ->   
search\_db\_single (Index 0) ->   
evaluate\_and\_submit\_single (Index 0) ->   
search\_db\_single (Index 1) ... ->   
finalize\_chunk.   
2. You MUST process each index one by one. Do   
NOT search for Index 1 until you have submitted   
Index 0.   
3. Once all indices reported by analyze\_intents   
are submitted, call finalize\_chunk.   
OUTPUT FORMAT:   
Every response MUST consist of two parts:   
1. A reasoning process wrapped in   
<think>...</think> tags.   
2. A function call wrapped in <act>...</act>   
tags.

## D Case Study on the Multi-Option Reality in Citations

To illustrate the gap between Strict and Lenient metrics, we analyze a representative case from our test set. Consider the context: “Compared with the global post-processing optimization used in previous work [#CITE#], our method learns the descent direction...” While the original authors cited GLAMR (Yuan et al., 2022), ReCite retrieved Hu-MoR (Rempe et al., 2021), a prominent framework that also utilizes a gradient-based global optimization loop for 3D human motion estimation. Both papers are academically appropriate and provide equivalent logical validation for the claim.

Under the Strict exact-match metric, retrieving HuMoR is penalized with a score of zero due to the string mismatch with the authors’ specific bibliographic key. In contrast, our Lenient Evaluation correctly identifies HuMoR as a valid semantic alternative. This case highlights how Strict F1 significantly underrepresents ReCite’s true citation faithfulness by ignoring acceptable multi-option academic realities and subjective author preferences.