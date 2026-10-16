# CARE-SD: Explainable Reasoning Architecture for Local SLM Self-Disclosure Analysis in Textual Data

Aisvarya Adeseye, Adeyemi Adeseye, Tahir Mohammad and Jouni Isoaho

Department of Computing, University of Turku, Turku, Finland

Abstract—Cybersecurity interviews uncover critical security ex periences, behaviors, and privacy concerns often not fully captured by conventional technical assessments. Consequently, identifying genuine self-disclosures is challenging because statements are context-dependent, categories overlap, and automated interpretations may introduce unsupported claims. Locally deployed Small Language Models (SLMs) protect sensitive participant data by keeping processing within controlled environments, but their limited reasoning capabilities make simple prompting insufficient for context-aware and factually reliable analysis. This study developed CARE-SD (Context-Adaptive Reasoning and Explanation for Self-Disclosure), a novel four-module architecture integrating context-adaptive processing, hierarchical disclosure reasoning, explanation-consistency verification, and cross-batch reconciliation. We evaluated three local SLMs, including Llama 1B, Qwen 1.5B, and Llama 3B, using 82 independently annotated cybersecurity interviews across five organizational sectors. Comparisons included conventional classifiers, three prompting strategies, larger 8B–32B models, and architectural ablations. Two independent reviewers assessed disclosure labels, factual claims, and explanation correctness. CARE-SD improved the accuracy and evidence reliability of SLMs, increasing Llama 3B Macro-F1 from 0.80 to 0.88 while reducing unsupported claims by 60%. Remarkably, the 3B model matched the 20B reference model in category Macro-F1, demonstrating the effectiveness of the architecture without requiring a larger model.

Index Terms—Self-disclosure analysis, small language models, explainable AI, cybersecurity, factual reliability

## I. INTRODUCTION

Organizational cybersecurity depends not only on technical safeguards but also on individuals’ security experiences, practices, and perceptions of privacy and trust. Frameworks such as the NIST Cybersecurity Framework support risk management [1], yet technical assessments alone may overlook why employees adopt unsafe practices, experience incidents, or distrust monitoring. Qualitative interviews complement these assessments by capturing personal incidents, protective behaviors, and privacy concerns. Such insights are particularly relevant to gamified workplaces, where engagement mechanisms can introduce concerns about data collection and surveillance [2], [3]. Distinguishing what participants personally experienced, did, or feared can therefore inform security awareness, organizational policy, and privacy-conscious system design.

However, cybersecurity self-disclosure analysis is more demanding than identifying security-related keywords. Participants may implicitly describe phishing incidents, refer to earlier interview questions, combine several personal concerns, or discuss hypothetical organizational risks without disclosing anything personally. A response may contain overlapping disclosures of security experience (C1), security/privacy behavior (C2), and security/privacy concern (C3). Failure to distinguish these categories can mischaracterise participants’ risk exposure or inflate the apparent prevalence of behaviors and incidents. Earlier language-model research established fine-grained selfdisclosure detection and taxonomies [4], but these findings do not directly establish accurate multi-label reasoning over context-dependent cybersecurity interview responses.

Large Language Models (LLMs) provide contextual interpretation, semantic classification, and human-readable explanations. Prior local-LLM studies involving cybersecurity interviews established the feasibility of qualitative privacy analysis [2] and investigated methods for reducing hallucination [3]. Nevertheless, fluent model-generated interpretations may attribute incidents to participants who discussed only hypothetical situations, infer unsupported motives, or present plausible explanations unsupported by transcript evidence. Such errors undermine qualitative cybersecurity conclusions, consistent with broader findings on generative hallucination [5]. Consequently, disclosure analysis requires more than plausible category predictions; generated claims and explanations must be assessed against their sources.

Local deployment offers an important way to process sensitive transcripts within researcher-controlled computing environments rather than routinely transmitting them to external inference services. Furthermore, large LLMs require substantial computing resources. Small Language Models (SLMs) of approximately 1B–3B parameters offer a more accessible alternative, as demonstrated in [6] work on small capable language models. However, their restricted capacity can impair contextual reasoning, multi-label decisions, and explanation fidelity. Language models also use long contexts inconsistently [7]; quantized-model studies have identified further reasons to introduce controlled verification [8]. Simply asking an SLM to detect disclosures, resolve references, assign overlapping categories, and justify each conclusion in one prompt leaves these subtasks entangled. Chain-of-thought prompting and iterative refinement help with reasoning [9], [10], but neither guarantees faithful evidence selection, and explanation quality must be independently assessed [11].

This study develops a modular and explainable architecture for local SLM-based cybersecurity self-disclosure analysis. The following research questions guided this work:

RQ1: To what extent does the proposed architecture improve self-disclosure detection and category classification compared with conventional classifiers and standard SLM prompting approaches?

RQ2: How effectively does the proposed architecture reduce unsupported interpretations and improve explanation accuracy while narrowing the performance gap with larger reference models?

The principal contribution of this study is CARE-SD, which integrates context-adaptive batch processing, hierarchical disclosure reasoning, explanation consistency verification, and crossbatch reconciliation. By separating SLM-based interpretation from deterministic context selection, validation, and aggregation, CARE-SD enables independently assessable predictions and supporting explanations. This study further establishes the classification and factual-reliability gains obtainable from this controlled architecture across locally deployable SLMs, offering a reproducible approach to evidence-grounded analysis under resource constraints.

## II. RELATED WORK

Detection and contextual classification: Self-disclosure detection has been studied in medical posts [26], social media and annotated spans [4], [27], [28], [34], and modality-dependent disclosures [21], [23]. Dou et al. [4] developed a 19-category taxonomy, while Yang et al. [27] studied contextual extraction of rationales. These advances address disclosure identification, but their evaluation units and categories differ from questiondependent, overlapping C1–C3 disclosures in cybersecurity interviews. Particularly, an extracted span does not by itself establish the correctness of a generated interpretation.

Privacy-sensitive text processing: Studies of personal disclosures in human–LLM conversations [29], context-aware privacy assistance [30], [31], and sensitive-information detection [16], [33] demonstrate important privacy applications. Locally deployable prompt reformulation [32], sanitization [18], synthetic disclosure data [17], and analyses of online communities [19], [20] address related but distinct tasks. Detecting sensitive phrases, supporting a user’s disclosure decision, or rewriting text does not establish whether category explanations generated by locally deployed SLMs faithfully represent the underlying interview evidence.

Prior Interview Analysis Research, Prior Foundations, and Research Gap. Research on self-disclosure in conversational AI [24], elicitation by interview agents [25], and semantic/thematic interpretation [22] provides relevant context for conversational settings. Also, prior studies by Adeseye et al. [2] examined LLM-assisted cybersecurity/privacy interviews, hallucination mitigation in qualitative privacy analysis [3], and multi-pass verification for quantized models [8], laying the foundation for this study. CARE-SD does not claim these underlying techniques or the interview setting as new. Its distinct contribution is the disclosure-specific integration: context-selected inputs, hierarchical multi-label decisions, quotation/explanation checks, and cross-window reconciliation, assessed jointly for category accuracy and claimlevel reliability.

## III. PROPOSED CARE-SD ARCHITECTURE

This study developed CARE-SD, a four-module architecture designed to improve the accuracy and factual reliability of local SLMs in cybersecurity-related textual disclosure analysis. As illustrated in Fig. 1, the architecture comprises (M1) context-adaptive batch processing, (M2) hierarchical disclosure reasoning, (M3) explainability and consistency verification, and (M4) cross-batch reconciliation and aggregation. The design combined context-management principles [7], structured language-model reasoning [9], disclosure classification [4], and iterative verification [10]. Unlike unconstrained multipass prompting, CARE-SD separates SLM-based semantic interpretation from deterministic context selection, validation, and aggregation.

## A. Architectural Modules

M1–Context-Adaptive Batch Processing: Long transcripts were divided into sentence-aligned windows of $W \ : = \ : 3 8 4$ tokens with O = 64 tokens of overlap. Each window preserved the participant response, corresponding interviewer question, participant identifiers, and original text offsets. Up to $K = 8$ independent windows were scheduled per inference batch without concatenating unrelated interviews. Because language models can overlook information within long contexts [7], the initial input was limited to the target participant response and its corresponding interviewer question. Context was expanded to the preceding P = 2 question–answer pairs when predefined cues (“yes”, “it”, “that”, or “as mentioned”) indicated contextual dependency or when M3 validation detected invalid output structure, inconsistent labels, or missing supporting quotations. A maximum input budget of $B = 2 0 4 8$ tokens was enforced to bound processing costs and limit irrelevant context while prioritizing the target participant response.

M2–Hierarchical Disclosure Reasoning: Self-disclosure is decomposed into binary detection followed by independent category classification, reducing the complexity of a single multi-label decision. This design was motivated by explicit disclosure taxonomies [4] and structured reasoning [9]. For text window i, classification was defined as

$$
\hat { y } _ { i c } = \hat { d } _ { i } \hat { z } _ { i c } , \quad c \in \{ 1 , 2 , 3 \} ,\tag{1}
$$

where $\hat { d } _ { i }$ indicates disclosure presence and $\hat { z } _ { i c }$ represents the predicted category: C1 (cybersecurity experiences), C2 (security/privacy behaviours), or C3 (personal security/privacy concerns). Multiple positive categories were permitted. C0 was assigned only when all category indicators were zero.

M3–Explainability and Consistency Verification: The SLM generated concise classification explanations and verbatim supporting quotations. Inspired by rationale evaluation [11] and iterative refinement [10], a deterministic validator checked JSON conformity, permitted labels, category consistency, quotation existence, and explanation length. Each positive category required a supporting quotation from the participant response. When validation failed, M3 triggered at most one corrective re-analysis using the expanded conversational context provided by M1. Outputs remaining invalid after this retry were flagged rather than assigned the C0 non-disclosure label. These checks enforced structural and quotation consistency, while semantic correctness and unsupported interpretations were independently evaluated by human reviewers.

![](images/8c72f505df8849bc2c1feca3c6b20d867d35961c1e86fc24af595b3d410e9635.jpg)  
Fig. 1. The proposed four-module CARE-SD architecture for context-adaptive, explainable self-disclosure analysis using local SLMs. The four-module design separates semantic reasoning from deterministic context management, verification, and aggregation, reducing the complexity of individual SLM tasks while improving classification accuracy, evidence reliability, and consistency across transcript segments.

M4–Cross-Batch Reconciliation and Aggregation: Overlapping windows were reconciled using participant identifiers and original text offsets. Duplicate quotations were removed, and valid category predictions were consolidated into one record per participant response:

$$
\hat { Y } _ { i c } = \mathbb { I } \left( \exists w \in \mathcal { W } _ { i } : v _ { w } = 1 \land \hat { y } _ { w c } = 1 \right) ,\tag{2}
$$

where W<sub>i</sub> represent the windows associated with response i, v<sub>w</sub> indicates successful validation, and I is the indicator function. This deterministic aggregation preserved valid disclosures across window boundaries. Responses with no valid predictions were flagged rather than automatically classified as nondisclosures.

## B. Algorithmic Implementation

Algorithm 1 presents the complete execution procedure. The controller explicitly defines segmentation, contextual triggers, inference order, validation conditions, retry limits, and aggregation. The SLM performs semantic detection and classification but does not control the processing workflow. All models used the same processing parameters and task definitions.

The inference function executed the detection system prompt S<sub>D</sub> first. When disclosure was detected, the classification system prompt S<sub>C</sub> independently assessed C1–C3; otherwise, C0 was assigned. The common user prompt U supplied the segment identifier, interviewer question, selected preceding context, and participant response. Context expansion and retry decisions were executed by the controller rather than generated by the SLM.

![](images/3679d297483f3b7d1ec6b49f7325fb11d5a4dc169d0c23c463a81c8c83af4163.jpg)

## C. System and User Prompt Specification

The architecture used two fixed system prompts and one common user template. The detection system prompt (S<sub>D</sub>) instructs the model to determine whether the participant personally disclosed a cybersecurity experience, security/privacy behaviour, or personal security/privacy concern, excluding general observations and hypothetical statements. It returns a Boolean disclosure decision in JSON format. The classification system prompt $( S _ { C } )$ defined C1–C3, permitted multiple categories, and required a verbatim quotation for each positive category and a concise explanation of at most 20 words. It explicitly prohibited invented experiences, motivations, or circumstances. The user prompt followed the fixed structure

TABLE I  
DATASET STATISTICS AND PARTICIPANT-DISJOINT PARTITIONS.
<table><tr><td>Split</td><td>Interviews</td><td>Annotated Responses</td><td>C0</td><td>C1</td><td>C2</td><td>C3</td></tr><tr><td>Training</td><td>58</td><td>1,028</td><td>240</td><td>360</td><td>450</td><td>420</td></tr><tr><td>Validation</td><td>12</td><td>225</td><td>53</td><td>80</td><td>95</td><td>88</td></tr><tr><td>Test</td><td>12</td><td>229</td><td>56</td><td>81</td><td>98</td><td>85</td></tr><tr><td>Total</td><td>82</td><td>1,482</td><td>349</td><td>521</td><td>643</td><td>593</td></tr></table>

$$
U _ { i } = \{ \mathrm { I D } _ { i } , Q _ { i } , C _ { i } , R _ { i } \} ,\tag{3}
$$

where $\mathrm { I D } _ { i }$ identifies the segment, $Q _ { i }$ is the interviewer question, $C _ { i }$ contains the controller-selected preceding context, and $R _ { i }$ is the target participant response. Both system prompts required JSON-only outputs. All prompt instructions, token limits, and correction rules remained fixed across the evaluated SLMs.

## IV. EXPERIMENTAL SETUP

The experiments evaluated whether CARE-SD improved disclosure-category accuracy and reduced unsupported interpretation. The evaluation combined independently annotated interviews, conventional and prompting baselines, larger-model references, architectural ablations, and blinded output review.

## A. Dataset and Human Gold Standard

The corpus comprised 82 completed researcher-led interviews on cybersecurity and privacy in gamified organizational environments. The participants represented NGOs (n = 35), companies $( n = 1 7 )$ , academia (n = 11), government (n = 10), and healthcare $( n ~ = ~ 9 )$ . Interviews employed open-ended questions in private virtual meetings. Participant responses were segmented for analysis while retaining their corresponding questions as context. Two independent reviewers annotated the entire corpus using three potentially overlapping self-disclosure categories: C1, cybersecurity experiences and incidents; C2, security and privacy behaviours; and C3, personal security and privacy concerns. C0 indicated the absence of all three categories. Pre-adjudication Cohen’s κ values were 0.90, 0.86, and 0.88 for C1–C3, respectively, yielding an unweighted mean of $\bar { \kappa } = 0 . 8 8 .$ . Disagreements were resolved through consensus to establish the human gold standard. The dataset contained 1,482 annotated participant responses, partitioned into training (58 interviews; 1,028 responses), validation (12; 225), and held-out testing (12; 229). All responses from each participant were assigned to a single partition to prevent participant-level information leakage. Table I summarizes the distribution. C0 is exclusive, whereas C1–C3 may overlap within individual responses.

Separately from gold-standard annotation, two reviewers, blinded to the model identity, independently assessed the generated outputs against the source transcripts. Each assessable factual claim was classified as supported, unsupported, or contradicted. The reviewers additionally evaluated whether the generated explanations correctly justified the assigned disclosure categories. Disagreements were resolved by consensus. Invalid outputs received one corrective retry. Predictions remaining invalid or flagged after reconciliation were retained in the held-out evaluation rather than excluded or automatically assigned C0. For detection and category F1-score, these predictions were counted as false negatives for gold-positive labels and false positives for gold-negative labels. Missing or invalid explanations were scored as incorrect whenever an explanation was required. The unsupported claim rate (UCR) and contradiction rate (CR) were computed as the respective proportions of unsupported and contradicted claims among all assessable generated factual claims. Outputs containing no assessable claims contributed nothing to these claim-level denominators and were recorded separately as flagged failures.

## B. Baselines and Reference Models

Conventional baselines consisted of TF-IDF features with a linear support vector machine (SVM) [12] and a fine-tuned DeBERTa-v3-base classifier [13]. They established lexical and contextual supervised reference performance, motivated by prior work on self-disclosure detection [4]. Both used freely available, permissively licensed implementations. The target SLMs were Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, and Llama-3.2-3B-Instruct. The Llama models enabled withinfamily comparison across parameter scales, while Qwen provided cross-family evaluation. Each was evaluated under (i) direct zero-shot instruction, (ii) few-shot demonstrations from the training set [14], (iii) zero-shot step-by-step reasoning [15], drawing on chain-of-thought prompting [9], and (iv) the proposed architecture. Larger, self-hosted references comprised Qwen3-8B, gpt-oss-20b, and Qwen3-32B. These supplied model-scale comparisons rather than ground-truth labels.

## C. Inference and Quantization

SLM inference used a local laptop with 64 GB of available memory. Larger models used a self-hosted AWS EC2 g6e.2xlarge instance with an NVIDIA L40S GPU (48 GB nominal VRAM) and 64 GiB system RAM. The specified configuration used 4-bit GGUF Q4\_K\_M weights for dense generative models and released native MXFP4 weights for gpt-oss-20b. Within-model comparisons held the checkpoint, quantization, decoding, and task specification fixed. Hardwaredependent efficiency metrics were excluded because the environments differed.

## D. Evaluation Matrix and Protocol

Evaluation procedure: Table II summarises the 5 experiments, their measurement definitions, and computation procedures. Classification metrics were calculated against the adjudicated gold standard, whereas hallucination and explanation metrics used consensus judgments from two independent reviewers. Macro-F1 was calculated across the three disclosure categories (C1–C3), treating each as an independent binary label. C0 represented the absence of all disclosure categories.

E1 and E2 established the primary accuracy and hallucination outcomes; E3 assessed individual component contributions; E4 examined model-scale differences; and E5 evaluated explanation correctness. Participant-disjoint splitting was used for point estimates. Confidence intervals are omitted because response-level paired predictions, participant identifiers, and claim adjudications required to independently rerun the reported bootstrap are not included in the available result records. Accordingly, differences are presented descriptively rather than as statistical-significance or equivalence findings.

## E. Experimental Comparisons

E1 tested architectural accuracy gains against direct, few-shot, and step-by-step prompting in the same SLM. E2 measured hallucination separately because a correct category could still have an unsupported explanation. E3 removed one architectural component at a time to attribute changes in Macro-F1 and UCR to its contribution. E4 compared architecture-enhanced SLMs with larger references using the same labels and test set. E5 evaluated human-judged explanation correctness [11].

## V. RESULTS AND DISCUSSION

Table III reports disclosure detection and category classification against the adjudicated gold standard, together with human-assessed factual reliability and explanation correctness. Detection F1 treated C1–C3 as disclosure and C0 as no disclosure; Macro-F1 was averaged over the three potentially overlapping categories. Unsupported Claim Rate (UCR) and Contradiction Rate (CR) were calculated over evaluable factual claims (lower is better), whereas Explanation Correctness Rate (ECR) measured human-judged correct explanations (higher is better). Figure 2 summarizes prompting and model-scale comparisons; component ablations appear in Table IV.

## A. E1: Architecture Effectiveness

CARE-SD improved category Macro-F1 over direct prompting from 0.72 to 0.84 for Llama 1B (+0.12), 0.75 to 0.85 for Qwen 1.5B (+0.10), and 0.80 to 0.88 for Llama 3B (+0.08), as shown in Fig. 2(a). Detection F1 also increased to 0.87, 0.88, and 0.90, respectively (Table III). The proposed configuration exceeded the strongest conventional baseline, DeBERTa-v3- base (Macro-F1 = 0.79), for all three SLMs. These comparisons isolated the gains within each underlying SLM, rather than attributing them to model scale alone.

## B. E2: Hallucination Reduction

Unsupported claims decreased under CARE-SD for all three SLMs: UCR fell from 0.21 to 0.10 (Llama 1B), 0.19 to 0.09 (Qwen 1.5B), and 0.15 to 0.06 (Llama 3B). The corresponding CR values declined from 0.09 to 0.04, 0.08 to 0.03, and 0.06 to 0.02. For Llama 3B, this represented a 60.0% relative UCR reduction (absolute difference: -0.09). Thus, gains in category classification were accompanied by fewer unsupported or contradicted statements in generated interpretations; the two outcomes were evaluated separately because correct labels could still have incorrect explanations.

(a) Prompting methods  
![](images/24f0447cf65ed2cc889dee33da201a6f07e18b8937cca1a76afaac567ae1e4cf.jpg)

(b) Accuracy vs factuality  
![](images/593e528ef15eed1208a8fdf851d8528997099105a2ccef1bcdd7216480df9310.jpg)  
Fig. 2. CARE-SD performance: (a) category Macro-F1 across prompting approaches; (b) Macro-F1 versus unsupported claim rate for three SLMs and three larger reference models. Ablation appears in Table IV.

## C. E3: Architectural Ablation

Single-module ablations were evaluated using Llama 3B (Table IV). Full CARE-SD achieved Macro-F1 = 0.88 and UCR = 0.06. Removing M1, M2, M3, and M4 reduced Macro-F1 by 4, 5, 3, and 2 percentage points, respectively. The corresponding UCR increases were 5, 6, 9, and 3 percentage points. M2 contributed most strongly to classification accuracy, whereas removing M3 increased UCR from 0.06 to 0.15, representing the largest deterioration in factual reliability (+9 percentage points; 150% relative increase). This finding highlights the contribution of quotation verification, consistency checking, and controlled re-analysis to limiting unsupported interpretations. However, the ablation did not isolate the individual effects of these verification operations or determine interactions between modules.

## D. E4: Larger-Model Comparison

Llama 3B + CARE-SD achieved Macro-F1 = 0.88, matching the reported two-decimal score of gpt-oss 20B, exceeding Qwen3 8B (0.87), and trailing Qwen3 32B (0.89) by one percentage point (Table III). This numerical agreement does not establish statistical equivalence. Its UCR of 0.06 equaled Qwen3 32B and was lower than Qwen3 8B (0.08) and gpt-oss 20B (0.07), as shown in Fig. 2(b). Llama 1B and Qwen 1.5B also achieved Macro-F1 values of 0.84 and 0.85, respectively. Computational efficiency was not directly compared because inference hardware differed.

TABLE II  
EXPERIMENTAL OBJECTIVES, METRICS, AND EVALUATION PROCEDURES.
<table><tr><td>Exp.</td><td>Objective</td><td>Metrics</td><td>Measurement and comparison</td></tr><tr><td>E1</td><td>Architecture effectiveness: detect and classify disclosures.</td><td> $\begin{array} { r } { F 1 = \frac { 2 T P } { 2 T P + F P + F N } ; } \end{array}$   $\begin{array} { r } { M F 1 = \frac { 1 } { 3 } \sum _ { c = 1 } ^ { 3 } F 1 _ { c } . } \end{array}$ </td><td>Against adjudicated labels: binary disclosure (C1–C3 versus C0) and mean binary F1 across C1-C3; compare prompting strategies and conventional classifiers.</td></tr><tr><td>E2</td><td>Hallucination reduction: limit unsupported and contradictory claims.</td><td> $U C R = U / N ;$   $C R = C / N ;$   $N = S + U + C .$ </td><td>Two blinded human reviewers adjudicated generated claims as supported (S), unsupported (U), or contradicted (C) against source transcripts.</td></tr><tr><td>E3</td><td>Component ablation: isolate module contributions.</td><td>MF1; UCR (defined above).</td><td>Remove each module individually (—j); compare with full CARE-SD, keeping the SLM and test set fixed.</td></tr><tr><td>E4</td><td>Model-scale comparison: assess smaller versus larger models.</td><td>MF1; UCR (defined above).</td><td>Evaluate CARE-SD-enhanced 1B-3B SLMs and 8B-32B references on identical held-out samples.</td></tr><tr><td>E5</td><td>Explanation correctness: assess the validity of explanations.</td><td> $E C R = E _ { \mathrm { o k } } / E _ { \mathrm { a l l } } .$ </td><td>Two independent reviewers adjudicate correct explanations  $( E _ { \mathrm { o k } } )$  among evaluated explanations  $( E _ { \mathrm { a l l } } ) .$  Sector-wise Macro-F1 was proposed but not available for reporting.</td></tr></table>

Definitions: TP, FP, FN: true positives, false positives, false negatives; $F 1 _ { c } \colon$ binary F1 for category C1, C2, or C3; N: total evaluable factual claims. All claim/explanation outcomes use consensus human-review judgments.

TABLE III  
DISCLOSURE CLASSIFICATION AND FACTUAL RELIABILITY. ↑: HIGHER IS BETTER; ↓: LOWER IS BETTER.
<table><tr><td>Det. F1 MF1 UCR CR ECR</td></tr><tr><td>Model / method</td></tr><tr><td>Conventional classifiers .71</td></tr><tr><td>TF-IDF + SVM .78 DeBERTa-v3-base .84</td></tr><tr><td>.79 1 一</td></tr><tr><td>Local small language models</td></tr><tr><td>Llama 1B / direct .79 .72 .21 .09 .75</td></tr><tr><td>Llama 1B / few-shot .81 .75 .19 .08 .78</td></tr><tr><td>Llama 1B / step-by-step .83 .78 .16 .07 .81</td></tr><tr><td>Llama 1B / CARE-SD .87 .84 .10 .04 .88</td></tr><tr><td>Qwen 1.5B / direct .81 .75 .19 .08 .78</td></tr><tr><td>Qwen 1.5B / few-shot .83 .78 .17 .07 .80</td></tr><tr><td>Qwen 1.5B / step-by-step .84 .80 .15 .06 .83</td></tr><tr><td>Qwen 1.5B / CARE-SD .88 .85 .09 .03 .90</td></tr><tr><td>Llama 3B / direct .84 .80 .15 .06 .84</td></tr><tr><td>.82</td></tr><tr><td>Llama 3B / few-shot .85 .14 .05 .86</td></tr><tr><td>Llama 3B / step-by-step .86 .83 .12 .05 .88</td></tr><tr><td>Llama 3B / CARE-SD .90 .88 .06 .02 .93</td></tr><tr><td>Larger reference models (CARE-SD)</td></tr><tr><td>Qwen3 8B .90 .87 .08 .03 .92</td></tr><tr><td>gpt-oss 20B .91 .88 .07 .02 .93</td></tr><tr><td>Qwen3 32B .92 .89 .06 .02 .94</td></tr></table>

Note: MF1 = category Macro-F1; UCR = unsupported claim rate; CR = contradicted claim rate; ECR = explanation correctness rate. Dashes mean not evaluated. Claim counts S/U/C (supported/unsupported/contradicted), flagged-output count F , and explanation counts $E _ { \mathrm { o k } } / E _ { \mathrm { a l l } }$ were not supplied with the aggregate result records; percentages cannot be independently recalculated without these denominators.

## E. E5: Explanation Correctness

Human-assessed explanation correctness (ECR) increased from 0.75 to 0.88 for Llama 1B, 0.78 to 0.90 for Qwen 1.5B, and 0.84 to 0.93 for Llama 3B under CARE-SD compared with direct prompting. Llama 3B matched the gpt-oss 20B reference (0.93) and approached Qwen3 32B (0.94). These improvements complemented the reductions in unsupported claims, indicating that the complete architecture improved both explanation correctness and factual reliability. However, the deterministic verification in M3 established structural and quotation consistency rather than semantic correctness, which was assessed independently by human reviewers.

## F. Discussion and Limitations

Across the reported comparisons, CARE-SD improved category classification and reduced unsupported interpretations without SLM fine-tuning. The ablations associated hierarchical reasoning most strongly with accuracy, and verification most strongly with factual reliability. Nevertheless, the findings were specific to 82 cybersecurity-oriented interviews. The held-out test comprises only 12 interviews, and sector sizes were unequal. No independently reproducible confidence intervals or sectorspecific scores were available from the supplied aggregates. The absence of claim-level numerators and denominators, flagged-output counts, and explanation-count denominators limits independent audit of UCR, CR, and ECR. Numerical equality in Macro-F1 between Llama 3B + CARE-SD and gpt-oss 20B does not establish statistical equivalence. The joint accuracy–factuality comparison was not treated as a Pareto frontier because some reference configurations dominated others on both plotted metrics.

## VI. CONCLUSION

This study developed and evaluated CARE-SD, a fourmodule reasoning architecture for accurate and explainable self-disclosure analysis using locally deployed small language models. The architecture integrated M1: Context-Adaptive Batch Processing, M2: Hierarchical Disclosure Reasoning, M3: Explainability and Consistency Verification, and M4: Cross-Batch Reconciliation to address contextual ambiguity and unsupported interpretations without modifying model parameters. Evaluation across 82 cybersecurity interviews demonstrated improvements in disclosure classification, factual reliability, and explanation correctness. With CARE-SD, Llama 3B improved category Macro-F1 from 0.80 to 0.88 and reduced the unsupported claim rate from 0.15 to 0.06, a 60% reduction. It also achieved detection F1 of 0.90 and explanation correctness of 0.93, numerically matching the 20B reference model’s Macro-F1. Ablation results identified M2 (Hierarchical Disclosure Reasoning) as the largest contributor to classification accuracy and M3 (Explainability and Consistency Verification) as the most influential module for reducing unsupported claims.

TABLE IV  
COMPONENT-WISE ABLATION OF CARE-SD WITH LLAMA-3.2-3B-INSTRUCT (SAME EVALUATION DATA; MF1 HIGHER AND UCR LOWER ARE BETTER).
<table><tr><td>Configuration</td><td>Disabled architectural functionality</td><td>Macro-F1 ↑</td><td>F1 loss (pp)↓</td><td>UCR ↓</td><td>UCR rise (pp)↓</td><td>Relative UCR rise</td></tr><tr><td>Full CARE-SD</td><td>All four modules enabled: adaptive context, hierarchical reasoning, explanation verification, and cross-batch reconciliation</td><td>0.88</td><td></td><td>0.06</td><td></td><td></td></tr><tr><td>Without M1</td><td>Context-adaptive input selection and targeted conversational context expansion</td><td>0.84</td><td>4</td><td>0.11</td><td>5</td><td>83.3%</td></tr><tr><td>Without M2</td><td>Hierarchical disclosure detection followed by independent multi-label classification of C1-C3</td><td>0.83</td><td>5</td><td>0.12</td><td>6</td><td>100.0%</td></tr><tr><td>Without M3</td><td>Explanation-consistency verification, supporting-quotation checks, and targeted validation of model outputs</td><td>0.85</td><td>3</td><td>0.15</td><td>9</td><td>150.0%</td></tr><tr><td>Without M4</td><td>Cross-window reconciliation, overlap handling, and aggregation of disclosure predictions</td><td>0.86</td><td>2</td><td>0.09</td><td>3</td><td>50.0%</td></tr></table>

Note: M1 = context-adaptive batch processing; M2 = hierarchical disclosure reasoning; M3 = explainability and consistency verification; M4 = cross-batch reconciliation.  
$\mathrm { F 1 ~ l o s s } = 1 0 0 ( 0 . 8 8 - F \mathrm { 1 } _ { \mathrm { a b l a t e d } } )$ percentage points.  
UCR rise = 100(UCR<sub>ablated</sub> − 0.06) percentage points.  
Relative UCR rise = 100(UCR<sub>ablated</sub> − 0.06)/0.06 percent.  
Each ablation removes one architectural module while retaining the remaining components.

These findings demonstrate that CARE-SD empowers resource-constrained SLMs to achieve classification performance comparable to a substantially larger model while improving factual reliability, strengthening their practical viability for privacy-sensitive self-disclosure analysis. Future work will evaluate CARE-SD on independent, larger, and more diverse disclosure datasets to establish cross-domain generalizability and robustness.

## REFERENCES

[1] National Institute of Standards and Technology, “The NIST Cybersecurity Framework (CSF) 2.0,” NIST CSWP 29, 2024, doi: 10.6028/NIST.CSWP.29.

[2] A. Adeseye, J. Isoaho, and T. Mohammad, “LLM-assisted qualitative data analysis: Security and privacy concerns in gamified workforce studies,” Procedia Computer Science, vol. 257, pp. 60–67, 2025, doi: 10.1016/j.procs.2025.03.011.

[3] A. Adeseye, J. Isoaho, and M. Tahir, “Insights on mitigating privacy concerns in gamification through LLM-assisted qualitative analysis with minimal hallucination,” ACM Transactions on Privacy and Security, vol. 29, no. 4, Art. no. 42, pp. 1–34, 2026, doi: 10.1145/3839358.

[4] Y. Dou et al., “Reducing privacy risks in online self-disclosures with language models,” in Proc. 62nd Annual Meeting of the Association for Computational Linguistics, pp. 13732–13754, 2024, doi: 10.18653/v1/2024.acl-long.741.

[5] Z. Ji et al., “Survey of hallucination in natural language generation,” ACM Computing Surveys, vol. 55, no. 12, Art. no. 248, pp. 1–38, 2023, doi: 10.1145/3571730.

[6] M. Abdin et al., “Phi-3 technical report: A highly capable language model locally on your phone,” arXiv:2404.14219, 2024.

[7] N. F. Liu et al., “Lost in the middle: How language models use long contexts,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 157–173, 2024, doi: 10.1162/tacl a 00638.

[8] A. Adeseye, J. Isoaho, and A. Adeseye, “Improving quantized model performance in qualitative analysis with multi-pass prompt verification,” arXiv:2605.20193, 2026.

[9] J. Wei et al., “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[10] A. Madaan et al., “Self-refine: Iterative refinement with self-feedback,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[11] J. DeYoung et al., “ERASER: A benchmark to evaluate rationalized NLP models,” in Proc. 58th Annual Meeting of the Association for Computational Linguistics, pp. 4443–4458, 2020, doi: 10.18653/v1/2020.aclmain.408.

[12] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

[13] P. He, J. Gao, and W. Chen, “DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing,” in Proc. International Conference on Learning Representations, 2023.

[14] T. B. Brown et al., “Language models are few-shot learners,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[15] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[16] C. Alfieri, J. Di Rocco, P. Inverardi, and P. T. Nguyen, “Exploring user privacy awareness on GitHub: An empirical study,” Empirical Software Engineering, vol. 29, Art. no. 156, 2024, doi: 10.1007/s10664-024-10544- 7.

[17] S. Jangra, S. De, N. Sastry, and S. Fadaei, “Protecting vulnerable voices: Synthetic dataset generation for selfdisclosure detection,” arXiv:2507.22930, 2025. [Online]. Available: https://arxiv.org/abs/2507.22930

[18] C. Alfieri, G. L. Scoccia, S. Ganesh, and N. M. Sadeh, “Applying large language models to sanitize self-disclosure in user-generated content,” Applied Soft Computing, vol. 181, Art. no. 113311, 2025, doi: 10.1016/j.asoc.2025.113311.

[19] Z. Wang and Y. Lu, “Unmasking self-disclosure in online depression communities: A novel framework integrating large language models and social network analysis,” in E-Business: Generative Artificial Intelligence and Management Transformation, 2025, pp. 74–88, doi: 10.1007/978-3- 031-94190-0 7.

[20] E.-U. Haq, S. Jangra, S. De, N. Sastry, and G. Tyson, “Unpacking the layers: Exploring self-disclosure norms, engagement dynamics, and privacy implications,” in Companion Proc. ACM Web Conference (WWW), 2025, doi: 10.1145/3701716.3717740.

[21] M. Grau, D. Sieber, N. Zierau, and I. Blohm, “Vocalizing user feedback: The impact of input modality on self-disclosure,” Proceedings of the ACM on Human-Computer Interaction, vol. 9, no. 7, Art. no. CSCW520, 2025, doi: 10.1145/3757701.

[22] S. Chiang, G. Laban, E. S. Cross, and H. Gunes, “Self-disclosure themes and semantics across human, robotic, and disembodied conversational partners,” in Proc. IEEE International Conference on Robot and Human Interactive Communication (RO-MAN), pp. 2012–2017, 2025, doi: 10.1109/RO-MAN63969.2025.11217701.

[23] K. Henderson, K. Omoomi, V. Varadarajan, A. Lahnala, and C. Welch, “Examining the utility of self-disclosure types for modeling annotators of social norms,” in Findings of the Association for Computational Linguis tics: EACL 2026, pp. 132–150, 2026, doi: 10.18653/v1/2026.findingseacl.8.

[24] H. Papneja and N. Yadav, “Self-disclosure to conversational AI: A literature review, emergent framework, and directions for future research,” Personal and Ubiquitous Computing, vol. 29, pp. 119–151, 2025, published online Aug. 2024, doi: 10.1007/s00779-024-01823-7.

[25] K. Hama, A. Otsuka, M. Mizukami, H. Sugiyama, and M. Naka, “RaPSIL: A preference-guided interview agent for rapport-aware self-disclosure,” in Proc. 26th Annual Meeting ofthe Special Interest Group on Discourse and Dialogue (SIGDIAL), pp. 355–369, 2025, doi: 10.18653/v1/2025.sigdial 1.29.

[26] M. Valizadeh, X. Qian, P. Ranjbar-Noiey, C. Caragea, and N. Parde, “What clued the AI doctor in? On the influence of data source and quality for transformer-based medical self-disclosure detection,” in Proc. 17th Conf. European Chapter of the ACL (EACL), pp. 1201–1216, 2023, doi: 10.18653/v1/2023.eacl-main.86.

[27] C. Yang, T. Chakrabarty, K. Hochstatter, M. Slavin, N. El-Bassel, and S. Muresan, “Identifying self-disclosures of use, misuse and addiction in community-based social media posts,” in Findings of the ACL: NAACL 2024, pp. 2507–2521, 2024, doi: 10.18653/v1/2024.findings-naacl.161.

[28] D. C. Lozoya, J. P. Mikal, Y. L. Wong, L. S. Hemmy, and M. Conway, “Truth in the noise: Unveiling authentic dementia self-disclosure statements in social media with LLMs,” in Proc. 22nd Workshop of the Australasian Language Technology Association, pp. 189–196, 2024.

[29] N. Mireshghallah, M. Antoniak, Y. More, Y. Choi, and G. Farnadi, “Trust no bot: Discovering personal disclosures in human–LLM conversations in the wild,” in Proc. Conf. on Language Modeling (COLM), 2024. [Online]. Available: https://arxiv.org/abs/2407.11438

[30] T. Du, J. Kim, A. Squicciarini, and S. Rajtmajer, “Toward context-aware privacy enhancing technologies for online self-disclosure,” Proc. AAAI Conf. Human Computation and Crowdsourcing, vol. 12, no. 1, pp. 44–54, 2024, doi: 10.1609/hcomp.v12i1.31599.

[31] I. Krsek et al., “Measuring, modeling, and helping people account for privacy risks in online self-disclosures with AI,” Proc. ACM Human Computer Interaction, vol. 9, no. 2, Art. no. CSCW131, 2025, doi: 10.1145/3711029.

[32] I. C. Ngong et al., “Protecting users from themselves: Safeguarding contextual privacy in interactions with conversational agents,” in Findings of the ACL: ACL 2025, pp. 26196–26220, 2025, doi: 10.18653/v1/2025.findings-acl.1343.

[33] H. Zeng, X. Liu, Y. Hu, C. Niu, F. Wu, S. Tang, and G. Chen, “Automated privacy information annotation in large language model interactions,” arXiv:2505.20910, 2025. [Online]. Available: https://arxiv.org/abs/2505.20910

[34] J. Luo et al., “Exploring and detecting self-disclosure in multi-modal posts on Chinese social media,” in Findings of the ACL: EMNLP 2025, pp. 21510–21527, 2025, doi: 10.18653/v1/2025.findings-emnlp.1173.