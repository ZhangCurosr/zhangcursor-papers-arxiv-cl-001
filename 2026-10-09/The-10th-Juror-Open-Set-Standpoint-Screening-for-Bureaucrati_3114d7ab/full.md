# The “10th Juror”: Open-Set Standpoint Screening for Bureaucratic Bias Detection

Yuchen Miao<sup>1</sup>, Zijun Wang<sup>1</sup>, Chang Han<sup>1</sup>, Yurui Shi<sup>3</sup>, Mingtai Zhang<sup>4</sup>, Siyang Xu<sup>2,\*</sup>

<sup>1</sup>Sydney Smart Technology College, Northeastern University, China <sup>2</sup>School of Computer and Communication Engineering, Northeastern University at Qinhuangdao, China <sup>3</sup>Taiyuan University of Technology <sup>4</sup>School of Resources, Environment and Materials, Guangxi University

## Abstract

Presupposing the boundaries of bias is itself a form of bias. We study closed-loop bias governance for Dutch government documents, where a system must detect biased language, ground decisions in legal and contextual evidence, rewrite problematic sentences when intervention is warranted, and verify that the rewrite mitigates harm without distorting meaning. Existing methods face three challenges: (i) discriminative classifiers capture surface regularities but lack normative grounding; (ii) zero-shot LLMs often adopt generic viewpoints and overflag ambiguous administrative language; and (iii) fixed taxonomies inherit the Closed-World Assumption, missing emerging local targets. We propose MARS-Gov, a standpoint-aware multi-agent framework that combines legal retrieval, open-set target screening, specialized jurors, conservative routing, and rewrite verification. When screening finds an uncovered group, MARS-Gov instantiates a dynamic “10th juror” to deliberate outside the fixed panel. On DGDB, MARS-Gov sets a new SOTA with 0.880 F1, outperforming the strongest zero-shot LLM detector by 20.2 points (29.8% relative) and the best supervised Dutch encoder by 6.8 points, while reducing unnecessary interventions to 2.5%. Leave-One-Category-Out (LOCO) evaluation recovers held-out categories with 85.1% Correct@1 and 93.8% Correct@3.

## 1 Introduction

Government documents are a primary interface between the state and its citizens. Their language therefore does more than transmit information: it marks who is treated as credible, risky, dependent, or entitled to care. Linguistic neutrality matters for institutional trust (Rizk and Lindgren, 2025; Eubanks, 2018), yet bias can persist even in restrained administrative prose (de Swart et al., 2025;

![](images/77799922264e142817b192e2dba6dba0f73f982b56a6caea71c86f76376ed087.jpg)  
Figure 1: Closed-set detection versus MARS-Gov open-set screening. Fixed taxonomies may miss local targets or force them into a known class. MARS-Gov retrieves legal evidence, detects standpoint gaps, adds a dynamic 10th juror when needed, and sends grounded reports to the Judge for an auditable verdict and minimal rewrite.

Benjamin, 2019). It often appears not as overt toxicity (Waseem and Hovy, 2016), but as exclusionary naming, negative framing, presupposition, or uneven standards of responsibility (Otmakhova et al., 2024). A sentence may be formally polite and still attach suspicion, incompetence, or burden to a protected or socio-economically vulnerable group.

This setting makes bias detection different from ordinary harmful-language classification. The relevant question is not only whether a phrase sounds negative, but whether it is normatively grounded, legally salient, and administratively faithful. A reliable system must identify the affected standpoint, distinguish a necessary eligibility condition from a stigmatizing proxy, and avoid rewriting official text in ways that change factual obligations. These requirements turn detection into a governance problem: the system must explain, intervene only when warranted, and verify that any intervention preserves meaning.

Existing approaches face three connected challenges. First, supervised encoders can be strong on closed benchmarks (Devlin et al., 2019; Chalkidis et al., 2020), but they mainly learn label regularities and provide limited evidence for why a decision is normatively justified (Blodgett et al., 2020). Second, zero-shot LLMs are more flexible (Gallegos et al., 2024), but they often adopt a generic fairness viewpoint, over-flag group mentions, and give plausible verdicts without showing which legal or administrative norm supports the decision (Resnik, 2025). Third, both paradigms usually assume that the ontology of bias is fixed before evaluation begins.

A deeper problem is therefore the fixed label space. DGDB contains 3,747 Dutch governmentdocument sentences with binary labels organized around nine bureaucratic-bias categories. As Figure 1 shows, a detector trained on these categories is evaluated as if all relevant categories were known in advance (Scheirer et al., 2013). Administrative language is less tidy. Digital illiteracy stigma, regional disadvantage, care dependency, or new socio-economic stereotypes may appear before an annotation scheme names them. Treating such cases as negative examples silently preserves the Closed-World Assumption; representing them only through the nearest fixed standpoint may preserve a binary alert while losing the affected group’s specific rationale. A governance detector therefore needs both known-category review and open-set target discovery.

We propose MARS-Gov (Multi-Agent Reasoning for Standpoint-aware Governance), a closedloop framework for government text and a new SOTA system on DGDB. For each sentence, MARS-Gov first retrieves legal and administrative context through Normative Evidence Contextualization (NEC). It then performs Open-Set Standpoint Screening (OSS): a Scout identifies potentially affected groups, fixed jurors inspect the nine known categories, and a dynamic “10th juror” is created when the Scout finds a target outside the panel. An Alert-based Routing Gate (ARG) conservatively clears low-risk sentences and escalates uncertain cases to a grounded Judge. If the Judge flags bias, a Restoration Agent proposes a minimal rewrite, and the system verifies the rewrite with the same detection criteria. The result is not a single classifier with a longer prompt, but an auditable decision loop for detection, mitigation, and review.

Our contributions are:

• We formulate bureaucratic bias detection as closed-loop governance: detection, evidence retrieval, minimal rewriting, and verification.

• We introduce MARS-Gov, a legal-retrieval and juror-based framework that sets a new SOTA on DGDB.

• We show open-set and cross-lingual robustness through a dynamic “10th juror” and governance metrics for mitigation quality and over-governance.

## 2 Related Work

Bias in government documents is shaped by institutional practice as well as model behavior (Eubanks, 2018; Noble, 2018; Benjamin, 2019; Rizk and Lindgren, 2025). Legal NLP resources such as LEGAL-BERT, LexGLUE, and LegalBench-RAG support this setting (Chalkidis et al., 2020, 2022; Pipitone and Houir Alami, 2024), while DGDB shows that bureaucratic bias often appears through framing and presupposition rather than slurs (de Swart et al., 2025). This view is consistent with discourse, framing, stereotyping, and intersectionality research, which treats bias as situated and socially produced (Fairclough, 1992; van Dijk, 1993; Goffman, 1974; Entman, 1993; Otmakhova et al., 2024; Allport, 1954; Fiske, 1998; Crenshaw, 1989). For administrative review, this matters because the same phrase can be harmless in a procedural context and discriminatory when it assigns risk, competence, or trustworthiness to a protected or stigmatized group.

Supervised and benchmark-based bias detection. Computational bias detection has relied on association tests, debiasing methods, and BERTstyle encoders (Caliskan et al., 2017; Bolukbasi et al., 2016; Devlin et al., 2019). Benchmarks such as StereoSet, BBQ, WinoBias, Social Bias Probing, and F2Bench probe stereotypes or fairness behavior with controlled examples (Nadeem et al., 2021; Parrish et al., 2022; Zhao et al., 2018; Marchiori Manerba et al., 2024; Lan et al., 2025). These resources are useful, and fine-tuned encoders remain strong closed-set baselines, but their fixed labels or demographic contrasts make them brittle for newly emerging administrative targets. They also give little support for the later governance question: what should be changed, and how can the system know that the change did not distort the original administrative meaning?

LLM-based detection, audits, and retrieval. LLMs can apply natural-language criteria without task-specific fine-tuning (Gallegos et al., 2024; Resnik, 2025). Audits nevertheless show hidden social associations, toxic continuations, long-form bias, and false positives around group mentions (Smith et al., 2022; Blodgett et al., 2020; Gehman et al., 2020; Jeung et al., 2025; Waseem and Hovy, 2016; Dixon et al., 2018). Retrieval can add legal evidence, but retrieval alone does not decide which standpoint should evaluate a sentence, how conflicting standpoints should be adjudicated, or whether a rewrite has repaired the problem. These unresolved steps are precisely where an apparently helpful LLM review can become hard to audit.

![](images/910cc223497c2eb708e8558a4b7e00bc41af3303b114ce5afe394ffbe12a1551.jpg)  
Figure 2: Overview of MARS-Gov. The framework retrieves normative evidence, gathers juror risk reports, routes only risky cases to a Judge, and verifies any minimal rewrite. Separating evidence, routing, adjudication, and restoration keeps the decision trace inspectable

Open-set and agentic governance. Open-set recognition and out-of-distribution detection study cases outside known categories (Scheirer et al., 2013; Hendrycks and Gimpel, 2017); recent debiasing work applies related ideas to LMs (Rani et al., 2025). Multi-agent LLM systems coordinate specialized roles for software, simulation, safety review, and factuality (Wu et al., 2023; Li et al., 2023; Qian et al., 2024; Hong et al., 2024; Park et al., 2023; He et al., 2025; Ning et al., 2025). Most such systems define agent roles in advance. We combine legal grounding, open-set target discovery, standpoint-specialized agents, and closedloop rewriting so that a newly affected group can receive a dedicated review standpoint instead of being represented only through the nearest fixed one.

## 3 Methodology

## 3.1 Task Definition: Closed-loop Bias Governance

We formulate closed-loop bias governance for a sentence x ∈ X from a government document or benchmark. In DGDB, x is a Dutch administrative sentence with an expert bias label. Unlike one-shot classification, the system may intervene and must verify the intervention. A pipeline D(·) returns an initial label $\hat { y } = D ( x ) \in \{ 0 , 1 \}$ , an optional minimal restoration $x ^ { \prime } = R ( x )$ , and a detection-only verification $\hat { y } ^ { \prime } = D _ { \mathrm { v e r i f y } } ( x ^ { \prime } )$ for non-empty restorations. The output also stores retrieved evidence and juror reports for audit. The task optimizes detection reliability, mitigation, and controlled side effects (§3.7). This framing treats a biased sentence as a governance failure only if the system can identify the issue, propose a faithful repair, and confirm that the repair is no longer judged biased.

Two constraints follow from this formulation. First, the system must explain why a sentence is problematic in terms that can be checked against external norms, rather than only against the model’s internal label boundary. Second, the rewrite must remain narrow: it should remove the biased implication without changing the administrative fact pattern. We therefore treat evidence retrieval, standpoint review, adjudication, restoration, and verification as separate artifacts in the trace.

## 3.2 Overview: Pipeline-defined Detector

Figure 2 shows MARS-Gov as a pipeline-defined detector rather than a single classifier. This design keeps evidence, standpoint assignment, routing, and intervention separate enough to audit. It first builds a neutral evidence summary $E ( x ) =$ $\mathrm { N E C } ( x ; K )$ from knowledge base $\kappa .$ , then uses a Scout and juror panel to produce reports ${ \mathcal { I } } ( x ) =$ $\mathrm { O S S } ( x , E ( x ) , { \mathcal { G } } ( x ) )$ ). ARG clears low-risk cases and sends the rest to a Judge; if restoration is produced, $D _ { \mathrm { v e r i f y } }$ applies the same detection criteria with rewriting disabled.

The trace records the retrieved evidence identifiers, the Scout’s affected-group hypotheses, all juror reports, the routing action, the Judge rationale, the proposed restoration, and the verification result. These records are not presented as legal authority, but they make the model’s path from sentence to intervention inspectable.

Algorithm 1: Inference in MARS-Gov.   
Input: Sentence x; knowledge base K; fixed jurors   
$\boldsymbol { B } _ { \mathrm { f i x } } = \left\{ B _ { 1 } , \ldots , B _ { 9 } \right\}$   
Output: Detection yˆ; optional restoration $x ^ { \prime } ;$   
verification $\hat { y } ^ { \prime } ;$ trace T.   
1 E ← NEC(x;K)   
2 G ← Scout(x, E)   
3 U ← Uncovered(G, B )   
4 ${ \mathcal { B } } _ { \mathrm { d y n } } \gets \{ \mathrm { I n s t a n t i a t e } ( g , x , E ) : g \in \mathcal { U } \}$   
5 $\mathcal { T }  \mathrm { O S } \mathrm { \bar { S } } ( x , E , \mathcal { B } _ { \mathrm { f x } } \cup \mathcal { B } _ { \mathrm { d y n } } ) ^ { \cdot } / \prime$ one report per   
fixed or dynamic juror   
6 a ← ARG(G, J) // rule-based routing gate   
7 $\hat { y } \gets 0 ; \ x ^ { \prime } \gets \emptyset ; \ \hat { y } ^ { \prime } \gets \emptyset$   
8 if a ̸= CLEAR then   
9 yˆ ← Judge(x, E, J )   
10 if yˆ = 1 then   
11 $x ^ { \prime } \gets R ( x , E , \mathcal { T } )$ // minimal   
restoration   
12 i $ { \mathbf { \hat { x } } } ^ { \prime } \neq \emptyset$ then   
13 $\dot { \hat { y } } ^ { \prime } \gets D _ { \mathrm { v e r i f y } } ( x ^ { \prime } ) \ / /$ same criteria;   
restoration disabled   
14 end   
15 end   
16 end   
17 $\mathcal { T }  \{ E , \mathcal { G } , \mathcal { T } , a , \hat { y } , x ^ { \prime } , \hat { y } ^ { \prime } \}$   
18 return $( \hat { y } , x ^ { \prime } , \hat { y } ^ { \prime } , \mathcal { T } )$

## 3.3 Normative Evidence Contextualization via RAG (NEC)

Governance decisions depend on policy definitions, protected attributes, and domain guidance, not only sentence semantics. NEC gives all downstream agents the same reference frame by applying three-stage RAG over K (Dutch equal-treatment law, antidiscrimination policy, accessibility guidance, online-harm and institutional-racism reports, inclusive-language guidance, and DGDB category notes): keyword retrieval, semantic retrieval, and chunk-level context expansion. Figure 3 summarizes these three retrieval stages. For input x, the retrieved evidence is

$$
C ( x ) = \mathrm { R e a d } \big ( \mathrm { R e t } _ { \mathrm { k w } } ( x ; { \mathcal K } ) , \mathrm { R e t } _ { \mathrm { s e m } } ( x ; { \mathcal K } ) ; { \mathcal K } \big ) .\tag{1}
$$

NEC then produces a neutral summary

$$
E ( x ) = \mathrm { S u m m } ( x , C ( x ) ) .\tag{2}
$$

The summarizer returns definitions, criteria, examples, and short caveats, but no verdict. This keeps NEC from becoming an implicit classifier and leaves later judgments traceable to $C ( x )$ rather than to unobserved model priors. In administrative text, this separation is important: the same word can be neutral in a rule description and harmful when attached to a group as a proxy for risk or competence.

![](images/e898513e998fe8f7da9ff9531e7ad07c06606224b3a893a877cea00d4856efb9.jpg)  
Figure 3: Three-layer retrieval in NEC.

## 3.4 Open-Set Standpoint Screening (OSS): Scout + Juror Panel

Bias governance is open-set because affected groups may not fit a fixed taxonomy. Given the input and evidence, the Scout identifies potentially implicated groups or sensitive attributes:

$$
\mathcal { G } ( x ) = \mathrm { S c o u t } ( x , E ( x ) ) .\tag{3}
$$

Let $B _ { \mathrm { f i x } } = \{ B _ { 1 } , \ldots , B _ { 9 } \}$ denote the fixed panel, which covers Disability, Migration, Colonialism, Religion, Gender, LGBTQ+, Culture, Education/Class, and Welfare. The Scout maps every target in $\mathcal { G } ( x )$ to this panel and collects the uncovered targets in

$$
\mathcal { U } ( x ) = \{ g \in \mathcal { G } ( x ) : \operatorname { C o v e r } ( g , \mathcal { B } _ { \mathrm { f i x } } ) = 0 \} .\tag{4}
$$

For every $g \in \mathcal { U } ( x )$ , OSS instantiates one temporary, evidence-conditioned juror:

$$
\begin{array} { r l } & { { \mathcal { B } } _ { \mathrm { d y n } } ( x , E ) = \{ B _ { g } : g \in \mathcal { U } ( x ) \} , } \\ & { \quad \quad \quad \quad \quad B _ { g } = \mathrm { I n s t a n t i a t e } ( g , x , E ) . } \end{array}\tag{5}
$$

Thus, a sentence with several uncovered standpoints receives a separate dynamic report for each standpoint rather than collapsing them into one generic analysis. DGDB predominantly assigns one primary category per sentence, so the common case creates a single additional juror, informally called the “10th juror”; this name is not an architectural limit. Dynamic jurors are not permanent ontology classes, but temporary review standpoints grounded in the current sentence and evidence.

Each fixed or dynamic juror returns an evidencegrounded report:

$$
\begin{array} { r l } & { \mathcal { B } ( x , E ) = \mathcal { B } _ { \mathrm { f i x } } \cup \mathcal { B } _ { \mathrm { d y n } } ( x , E ) , } \\ & { \mathcal { I } ( x , E ) = \{ J _ { B } : B \in \mathcal { B } ( x , E ) \} , } \\ & { \quad J _ { B } = B ( x , E ) . } \end{array}\tag{6}
$$

Here $J _ { B } \ = \ ( b _ { B } , p _ { B } , r _ { B } , \rho _ { B } )$ contains a binary verdict, confidence, severity, and a short rationale grounded in $E ( x )$ , optionally with supporting spans. These reports are passed to routing and restoration, giving the Judge several grounded views rather than one undifferentiated LLM answer.

This design also limits a common failure mode in open-set detection. The system may notice an unfamiliar affected group, but the final decision still passes through the same Judge, evidence, and verification steps as known categories. Open-set discovery therefore expands who can be reviewed, while the downstream standard for intervention remains unchanged.

## 3.5 Alert-based Routing Gate (ARG): Risk-sensitive Escalation

To reduce cost without lowering recall, MARS-Gov uses an alert-based routing gate (ARG)<sup>1</sup>. ARG maps $( \mathcal { G } , \mathcal { I } )$ to CLEAR or ESCALATE, clearing only when all risk signals are weak and wellformed. It escalates under three conditions: hard alerts, when Scout confidence exceeds $\tau _ { \mathrm { h i g h } }$ or a high-risk watchlist threshold; soft alerts, when multiple jurors give weaker but consistent signals; and fail-safe alerts, when any juror output is missing, malformed, or unparsable.

$$
\mathrm { A R G } ( \mathcal G , \mathcal T ) = \left\{ \begin{array} { l l } { \mathrm { C L E A R , } } & { \mathrm { n o \ a l e r t \ f i r e s , } } \\ { \mathrm { E S C A L A T E , } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{7}
$$

Escalated cases go to the Judge and, when needed, restoration. ARG is rule-based, so routing changes do not introduce extra backbone calls.

## 3.6 Decision and Restoration: Judge and Verify with Minimal Rewriting

For escalated cases, the Judge resolves juror conflicts using $x , E ( x )$ , and $\mathcal { I } ( x , E ( x ) )$ . It grounds

the final decision in explicit evidence and outputs a binary label

$$
\hat { y } \ = \ \mathrm { J u d g e } ( x , E ( x ) , \mathcal { I } ( x , E ( x ) ) ) \in \{ 0 , 1 \} .\tag{8}
$$

The Judge sees raw reports and evidence, but not the ARG verdict, which keeps adjudication separate from threshold tuning. This matters in practice because conservative routing should affect cost and recall, not the legal rationale of the final verdict.

If $\hat { y } = 1$ , a Restoration Agent generates a minimally edited sentence

$$
x ^ { \prime } = R ( x , E ( x ) , \mathcal { I } ( x , E ( x ) ) ) .\tag{9}
$$

The restoration removes biased framing while preserving factual content, meaning, and administrative register. It minimizes surface edits, returns exactly one revised sentence, and emits an empty string if no safe rewrite is possible. Empty restorations are treated as failed mitigation. The restored sentence is then verified:

$$
\hat { y } ^ { \prime } = D _ { \mathrm { v e r i f y } } ( x ^ { \prime } ) .\tag{10}
$$

A governance action succeeds when $\hat { y } = 1 , x ^ { \prime } \neq \varnothing$ and $\hat { y } ^ { \prime } = 0 .$ , so the same detector clears the nonempty restoration. This prevents a rewrite from being counted as successful merely because it was produced.

We deliberately avoid free-form paraphrasing. In pilot runs, broad rewrites often improved tone while changing who did what, which is unacceptable for administrative records. The restoration agent is therefore instructed to preserve named entities, dates, obligations, eligibility criteria, and material facts unless the biased wording itself must be replaced.

We measure meaning preservation by the cosine similarity between sentence embeddings of the original and restored text:

$$
\mathrm { S F } ( x , x ^ { \prime } ) = \cos \bigl ( \phi ( x ) , \phi ( x ^ { \prime } ) \bigr ) ,\tag{11}
$$

where $\phi ( \cdot )$ is a fixed text encoder.<sup>2</sup> If no restoration is produced, $\mathrm { S F } ( x , x ^ { \prime } ) = 0 $

## 3.7 Pipeline-aware Governance Metrics

F1 only assesses the initial decision on x. It does not show whether restoration resolves the problem or whether the system intervenes on neutral text. We therefore add metrics for the full detect– restore–verify loop. Given $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , let $\hat { y } _ { i } = D ( x _ { i } ) , x _ { i } ^ { \prime }$ be the restored sentence, $\hat { y } _ { i } ^ { \prime } =$ $D _ { \mathrm { v e r i f y } } ( x _ { i } ^ { \prime } )$ when $x _ { i } ^ { \prime } \neq \varnothing ,$ , and $m _ { i } = \mathbb { I } [ x _ { i } ^ { \prime } \neq \varnothing ]$ We also compute $\mathrm { S F } _ { i } = \mathrm { S F } ( x _ { i } , x _ { i } ^ { \prime } )$ , with $\mathrm { S F } _ { i } = 0$ when no rewrite is produced. Bias Mitigation Rate (BMR) measures the share of correctly detected biased cases that are restored and cleared:

$$
\mathrm { B M R } \ = \ \frac { \sum _ { i } m _ { i } \mathbb { I } [ y _ { i } = 1 \wedge \hat { y } _ { i } = 1 \wedge \hat { y } _ { i } ^ { \prime } = 0 ] } { \sum _ { i } \mathbb { I } [ y _ { i } = 1 \wedge \hat { y } _ { i } = 1 ] } .\tag{12}
$$

Semantic Fidelity (SF) averages meaning preservation over successfully mitigated cases:

$$
\overline { { \mathrm { S F } } } \ = \ \frac { \sum _ { i } m _ { i } \mathrm { S F } _ { i } \cdot \mathbb { I } [ y _ { i } = 1 \wedge \hat { y } _ { i } = 1 \wedge \hat { y } _ { i } ^ { \prime } = 0 ] } { \sum _ { i } m _ { i } \mathbb { I } [ y _ { i } = 1 \wedge \hat { y } _ { i } = 1 \wedge \hat { y } _ { i } ^ { \prime } = 0 ] } .\tag{13}
$$

Bias Governance Score (BGS) combines coverage, mitigation, and fidelity:

$$
{ \mathrm { B G S ~ } } = { \mathrm { ~ R e c a l l \times B M R \times { \overline { { S F } } } } } .\tag{14}
$$

Including Recall prevents a pipeline from scoring well by governing only easy positives. Over-Governance Rate (OGR) measures unnecessary intervention on non-biased inputs, equivalent to the false-positive rate on the initial detector:

$$
\mathrm { O G R } \ = \ { \frac { \sum _ { i } \mathbb { I } [ y _ { i } = 0 \wedge { \hat { y } } _ { i } = 1 ] } { \sum _ { i } \mathbb { I } [ y _ { i } = 0 ] } } .\tag{15}
$$

Together, BMR, SF, BGS, and OGR evaluate mitigation quality and side effects beyond one-shot classification. We report these components alongside F1 because a model can look strong under ordinary classification while over-rewriting neutral text, or while producing rewrites that do not survive verification.

## 4 Experiments

We organize the experiments around five questions: RQ1: Does MARS-Gov outperform fine-tuned, zero-shot, and agentic baselines?

RQ2: Can the Scout recover held-out bias categories outside the fixed juror panel?

RQ3: Which modules account for the gains in detection and governance quality?

RQ4: How much inference cost does ARG save without changing the detector?

RQ5: Does the review procedure remain useful across domains and languages?

## 4.1 Experimental Setup

Datasets.<sup>3</sup> We use DGDB (de Swart et al., 2025), a Dutch administrative benchmark with nine bias categories. To test vocabulary generalization, we evaluate on a held-out subset, DGDB-Unseen, containing new bias terms. We also use DALC (Caselli et al., 2021), a Dutch social-media offensivelanguage set; SBIC (Sap et al., 2020), an English social-bias inference corpus; and KoBBQ (Jin et al., 2024), a Korean BBQ-style bias benchmark.

Baselines. We compare fine-tuned PLMs (BERTje, RobBERT), zero-shot LLMs (GPT-4o, Grok-3, Gemini 2.5 Pro, DeepSeek-R1), and agentic controls: SA-RAG and Standard Debate. Two matched Gemini 2.5 Pro controls test whether stronger prompting can explain the gains. A fixed 12-shot CoT prompt uses nine biased demonstrations spanning the DGDB categories and three hard nonbiased demonstrations, all selected from non-test data with fixed IDs and order. A Single-Call Integrated Prompt receives the same retrieved NEC evidence, task definition, and output requirements as MARS-Gov, but performs affected-group identification, classification, rationale generation, minimal rewriting, and self-checking in one LLM call. For governance metrics, all generative controls use the same frozen $D _ { \mathrm { v e r i f y } }$ and fixed SF encoder; the Single-Call self-check does not determine BMR or BGS.

## 4.2 Overall Performance (RQ1)

Table 1 reports in-domain DGDB and DGDB-Unseen performance across four backbones<sup>4</sup>. DGDB-Unseen replaces selected DGDB bias terms with alternative unseen expressions while preserving labels and administrative context, testing whether detectors generalize beyond the original lexical cues.

❶ Architecture Trumps Generic Debate. Zeroshot models have high recall but plateau at F1 ≈ 0.63–0.68 because precision stays below 0.55. Standard Debate improves precision only modestly, and SA-RAG does not consistently surpass the best fine-tuned encoder. MARS-Gov reaches SOTA F1 (> 0.85) across all backbones, with Gemini 2.5 Pro obtaining the best mean F1 (0.880).

Matched prompting controls. The 12-shot CoT prompt raises Gemini 2.5 Pro F1 from 0.664 to

<table><tr><td colspan="10"></td></tr><tr><td>Fine-tuned</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>F10oD</td><td>∆Drop</td><td>OGR↓</td><td>BMR</td><td>SF</td><td>BGS</td></tr><tr><td colspan="10">BERTje (de Vries et al., 2019) 0.842 ±.017 0.784 ±.021 0.812 ±.014</td></tr><tr><td>RobBERT (Delobelle et al., 2020)</td><td>0.839 ±.0190.784 ±.0180.811 ±.013</td><td></td><td></td><td>0.554 ±.032 0.499 ±.034</td><td>-31.8% ±1.62 -38.5% ±1.94</td><td> $3 . 1 \% \pm 0 . 5 1$   $3 . 6 \% \pm 0 . 6 2$ </td><td></td><td></td><td></td></tr><tr><td colspan="10">I. Prompted LLMs (Zero-/Few-shot)</td></tr><tr><td>GPT-40</td><td></td><td> $0 . 5 1 4 \pm . 0 2 5 \ 0 . 8 8 2 \pm 0 2 1 \ 0 . 6 4 9 \pm . 0 1 7 \ 0 . 6 2 1 \pm . 0 2 4 \ - 4 . 3 \mathcal { G } \pm 1 . 0 7 \ | 1 6 . 4 \mathcal { W } \pm \pm 0 . 8 3 \ 0 . 8 7 2 \pm 0 1 6 \ 0 . 8 2 1 \pm . 0 2 1 \ \ 0 . 6 3 1 \pm . 0 2 1$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Grok-3</td><td></td><td> $0 . 4 8 4 \pm . 0 3 1 0 . 9 2 0 \pm . 0 1 7 0 . 6 3 4 \pm . 0 1 9 0 . 5 9 8 \pm . 0 2 9 - 5 . 7 \mathcal { H } \pm 1 . 1 8$ </td><td></td><td></td><td></td><td></td><td></td><td>19.1% ±0.940.864 ±.019 0.808 ±.0240.642 ±.022</td><td></td></tr><tr><td>Gemini 2.5 Pro</td><td></td><td> $0 . 5 2 4 \pm . 0 2 4 \ 0 . 9 0 7 \pm . 0 1 8 \ 0 . 6 6 4 \pm . 0 1 6 \ 0 . 6 3 6 \pm . 0 2 2 \ - 4 . 2 \% \pm 0 . 9 5$ </td><td></td><td></td><td></td><td></td><td></td><td> $1 5 . 2 \% \pm 0 . 7 2 0 . 8 8 1 \pm . 0 1 4 0 . 8 2 7 \pm . 0 1 8 0 . 6 6 1 \pm . 0 2 0$ </td><td></td></tr><tr><td>Gemini 2.5 Pro (12-shot CoT)</td><td></td><td> $0 . 6 4 2 \pm . 0 2 1 0 . 8 8 3 \pm . 0 1 9 0 . 7 4 3 \pm . 0 1 7 0 . 7 1 6 \pm . 0 2 1$ </td><td></td><td></td><td>-3.6%  $- 3 . 8 \% \pm 0 . 8 3$ </td><td></td><td></td><td> $9 . 6 \% \pm 0 . 6 1 0 . 9 1 4 \pm . 0 1 5 0 . 8 7 6 \pm . 0 1 8 0 . 7 0 7 \pm . 0 1 8$ </td><td> $1 3 . 4 \% \pm 0 . 6 5 0 . 8 9 3 \pm . 0 1 3 0 . 8 5 1 \pm . 0 1 6 0 . 6 7 4 \pm . 0 1 7$ </td></tr><tr><td colspan="10">DeepSeek-R1  $0 . 5 4 9 \pm . 0 2 3 ~ 0 . 8 8 7 \pm . 0 1 9 ~ 0 . 6 7 8 \pm . 0 1 5 ~ 0 . 6 5 2 \pm . 0 2 1$ </td></tr><tr><td colspan="10">II. Standard Debate (No RAG)  $0 . 5 8 7 \pm 0 2 6 \ 0 . 8 6 2 \pm 0 2 0 \ 0 . 6 9 8 \pm 0 1 8 \ 0 . 6 7 1 \pm 0 2 6 \ - 3 . 9 4 7 6 \pm 1 . 1 3 1 \ 1 2 . 1 \% \ 4 0 \pm 0 . 7 7 \ 0 . 9 0 2 \pm 0 1 7 \ 0 . 8 6 8 \pm 0 1 9 \ 0 . 6 7 4 \pm 0 . 0 2 0$ </td></tr><tr><td>Standard Debate (GPT-4o)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Standard Debate (Grok-3)</td><td></td><td> $0 . 5 4 9 \pm . 0 2 9 0 . 9 0 6 \pm . 0 1 9 0 . 6 8 4 \pm . 0 2 1 0 . 6 5 1 \pm . 0 3 1 - 4 . 8 \mathcal { H } \pm 1 . 4 3$ </td><td></td><td></td><td></td><td></td><td></td><td> $1 4 . 5 \% \pm 0 . 8 7 \ 0 . 8 7 8 \pm . 0 1 9 0 . 8 5 1 \pm . 0 2 2 \ 0 . 6 7 7 \pm . 0 2 4$ </td><td></td></tr><tr><td colspan="10">Standard Debate (Gemini 2.5 Pro)  $0 . 6 1 2 \pm . 0 2 3 0 . 8 8 6 \pm . 0 1 9 0 . 7 2 4 \pm . 0 1 8 0 . 6 9 6 \pm . 0 2 4 - 3 . 9 \mathcal { H } \pm 0 . 9 6$  Standard Debate (DeepSeek-R1)</td></tr><tr><td colspan="10">0.634 ±.0220.882 ±.018 0.738 ±.017 0.711 ±.023-3.7% ±0.86  $1 0 . 2 \% \pm 0 . 6 3 \ 0 . 9 2 1 \pm . 0 1 4 \ 0 . 8 9 2 \pm . 0 1 7 \ 0 . 7 2 4 \pm . 0 1 8$  III. Integrated and Agentic Controls</td></tr><tr><td colspan="10"></td></tr><tr><td>SA-RAG (GPT-40)  $\mathbf { S A { \mathrm { - } } R A G } \left( \mathbf { G r o k } { \mathrm { - } } 3 \right)$ </td><td></td><td> $0 . 6 6 7 \pm . 0 2 0 0 . 8 5 2 \pm . 0 1 9 0 . 7 4 8 \pm . 0 1 5 0 . 7 2 3 \pm . 0 2 1$ </td><td></td><td></td><td> $- 3 . 3 \% \pm 0 . 7 1$ </td><td> $8 . 9 \% \pm 0 . 5 9$ </td><td></td><td> $0 . 9 1 3 \pm . 0 1 5 0 . 8 7 6 \pm . 0 1 7 0 . 6 8 2 \pm . 0 1 7$ </td><td></td></tr><tr><td> $\mathbf { S A { \mathrm { - } } R A G } \ ( \mathbf { G e m i n i } \ 2 . 5 \ \mathbf { P r o } )$ </td><td></td><td> $0 . 6 1 9 \pm . 0 2 4 0 . 8 9 6 \pm . 0 1 7 0 . 7 3 2 \pm . 0 1 7 0 . 7 0 0 \pm . 0 2 4$ </td><td></td><td></td><td> $- 4 . 4 \% \pm 1 . 0 4$ </td><td> $1 1 . 1 \% \pm 0 . 7 1$ </td><td></td><td>0.889 ±.018 0.862 ±.021 0.687 ±.019</td><td></td></tr><tr><td>Single-Cail Integrated Prompt</td><td></td><td> $0 . 6 8 7 \pm . 0 1 9 0 . 8 7 6 \pm . 0 1 8 0 . 7 7 0 \pm . 0 1 4 0 . 7 4 8 \pm . 0 1 9$ </td><td></td><td></td><td> $- 2 . 9 \% \pm 0 . 6 2$ </td><td> $7 . 9 \% \pm 0 . 5 1$ </td><td></td><td> $0 . 9 1 8 \pm . 0 1 3 0 . 8 9 3 \pm . 0 1 4 0 . 7 1 8 \pm . 0 1 5$ </td><td></td></tr><tr><td> $\mathbf { S A { \mathrm { - R A G } } \Phi ( D e e p { \bar { \mathbf { S } } } e e k { \mathrm { - } } R 1 ) }$ </td><td></td><td> $\begin{array} { r } { 0 . 7 6 2 \pm . 0 1 9 \ 0 . 8 6 9 \pm . 0 1 8 \ 0 . 8 1 2 \pm . 0 1 5 \ \mathrm { ~  ~ { ~ - ~ } ~ } } \end{array}$   $0 . 7 1 0 \pm . 0 1 8 0 . 8 7 6 \pm . 0 1 7 0 . 7 8 4 \pm . 0 1 3 0 . 7 6 2 \pm . 0 1 8$ </td><td></td><td></td><td> $- 2 . 8 \% \pm 0 . 5 3$ </td><td> $6 . 7 \% \pm 0 . 5 5$   $7 . 2 \% \pm 0 . 4 7$ </td><td> $0 . 9 2 9 \pm . 0 1 2 0 . 9 0 2 \pm . 0 1 3 0 . 7 3 4 \pm . 0 1 3$ </td><td> $0 . 9 2 7 \pm . 0 1 5 0 . 9 1 2 \pm . 0 1 7 0 . 7 3 5 \pm . 0 1 9$ </td><td></td></tr><tr><td colspan="10"></td></tr><tr><td colspan="10"> $I V . O u r s \colon M A R S \ – G o \nu F r a m e w o r k$ </td></tr><tr><td>MARS-Gov (GPT-40)</td><td></td><td> $0 . 8 4 4 \pm . 0 1 7 \ 0 . 8 6 3 \pm . 0 1 5 \ 0 . 8 5 3 \pm . 0 1 3 \ 0 . 8 2 3 \pm . 0 1 9$ </td><td></td><td></td><td> $- 3 . 5 \% \pm 0 . 6 2$ </td><td> $3 . 5 \% \pm 0 . 3 2$ </td><td></td><td> $0 . 9 3 4 \pm . 0 1 1 0 . 9 3 9 \pm . 0 1 3 0 . 7 5 7 \pm . 0 1 4$ </td><td></td></tr><tr><td>MARS-Gov (Grok-3)</td><td></td><td> $0 . 8 3 6 \pm . 0 1 9 0 . 9 1 0 \pm . 0 1 3 0 . 8 7 1 \pm . 0 1 4 0 . 8 4 0 \pm . 0 2 2$ </td><td></td><td></td><td> $- 3 . 6 \% \pm 0 . 7 7$ </td><td> $3 . 9 \% \pm 0 . 4 1$   $2 . 5 \% \pm 0 . 2 8$ </td><td></td><td>0.922 ±.013 0.928 ±.015 0.779 ±.018</td><td></td></tr><tr><td> $\mathbf { M A R S - G o v } \left( \mathbf { G e m i n i } \ 2 . 5 \ \mathbf { P r o } \right)$ </td><td></td><td> $0 . 8 6 5 \pm . 0 1 5 0 . 8 9 5 \pm . 0 1 3 0 . 8 8 0 \pm . 0 1 2 0 . 8 5 3 \pm . 0 1 7$ </td><td></td><td></td><td> $- 3 . 1 \% \pm 0 . 5 1$ </td><td></td><td></td><td></td><td> $\mathbf { 0 . 9 5 1 \pm . 0 0 9 0 . 9 4 9 \pm . 0 1 1 0 . 8 0 8 \pm . 0 1 3 }$ </td></tr><tr><td>MARS-Gov (DeepSeek-R1)</td><td></td><td> ${ \bf 0 . 8 7 2 \pm . 0 1 6 0 . 8 8 4 \pm . 0 1 4 0 . 8 7 8 \pm . 0 1 3 0 . 8 5 7 \pm . 0 1 7 }$ </td><td></td><td></td><td> $- 2 . 4 \% \pm 0 . 4 7$ </td><td> $2 . 6 \% \pm 0 . 2 9$ </td><td></td><td>0.948 ±.011 0.962 ±.010 0.806 ±.015</td><td></td></tr></table>

Table 1: Main DGDB results. Darker green marks stronger performance; darker red marks larger ∆Drop or OGR. F1 is DGDB-Unseen F1; ∆Drop is the relative drop from DGDB F1. Single-Call was evaluated in-domain only. MARS-Gov gives the best mean F1/BGS with low governance intrusion.

0.743 and lowers OGR from 15.2% to 9.6%. Integrating all steps into one call further raises F1 to 0.812 and BGS to 0.735. The full staged procedure remains 6.8 F1 points and 7.3 BGS points higher than Single-Call, while reducing OGR from 6.7% to 2.5%. These comparisons isolate gains beyond demonstrations, prompt length, and retrieved context alone.

❷ DGDB-Unseen Exposes Lexical Fragility. Fine-tuned encoders look attractive on the original split: BERTje and RobBERT are compact and reach F1 ≈ 0.81, higher than most Standard Debate variants. Yet their $\operatorname { F l } _ { \mathrm { O O D } }$ falls to 0.554 and 0.499 (∆Drop = -31.8% and -38.5%). This is a serious weakness for bias detection, because social naming practices change and new stigmatizing terms appear after training. MARS-Gov is much more stable, keeping DGDB-Unseen F1 at 0.823–0.857 with ∆Drop between -3.6% and -2.4%.

❸ Improving the Precision-Recall Balance. Grok-3 gives high zero-shot recall (0.920) but high OGR (19.1%), showing that recall alone can produce excessive intervention. MARS-Gov (Gemini 2.5 Pro) keeps recall high (0.895), raises precision to 0.865, and lowers OGR to 2.5%. DeepSeek-R1 gives the highest MARS-Gov precision (0.872) and the lowest DGDB-Unseen drop (-2.4%).

Overall, the main gain comes from routing legal evidence and standpoint-specific reports into adjudication. Every backbone benefits more from MARS-Gov than from Standard Debate or SA-RAG, and the best configuration improves BGS to 0.808 while keeping unnecessary intervention low.

## 4.3 Controlled Open-set Evaluation with Held-out Categories (RQ2)

Open-set evaluation. DGDB-Unseen tests lexical robustness: whether a detector remains stable when familiar bias terms are replaced by unseen expressions. It does not show whether MARS-Gov can recover a bias category that is absent from the fixed juror panel. To test this stronger open-set claim, we design a leave-one-juror-out experiment over all 9 categories with Gemini 2.5 Pro; the setup is as follows.

Design. Each DGDB category is associated with a fixed juror and watchlist. We remove one juror– watchlist pair at a time and evaluate the corresponding test subset, excluding both the specialist and its lexical cues. We encode the Scout’s predicted group and rationale together with fixed category prototypes using clips/e5-base-trm-nl, and rank the categories by cosine similarity. Correct@k indicates whether the held-out category appears among the top k predictions. The prototypes are fixed before evaluation and are not tuned by category; Appendix C provides the full texts.

Results. Table 2 shows that, without the Scout, the remaining jurors achieve an average F1 of 0.807: they often detect broad negative framing but fail to identify the held-out category. Scout-guided dynamic review raises average F1 to 0.875 and recovers the held-out category with 85.1% Correct@1 and 93.8% Correct@3. The gains are pronounced for categories with indirect cues, including Colonialism and Education/Class. This distinction matters for rewriting: a neighboring juror may flag the sentence as biased, but recovering the held-out standpoint provides the category-specific basis for revision.

<table><tr><td>Config.</td><td>Metric</td><td>B1 (Disability)</td><td>B2  $( \mathbf { M i g r a t i o n } )$ </td><td>B3  $\mathbf { ( C o l o n i a l i s m ) }$ </td><td>B4 (Religion)</td><td>B5  $( \mathrm { G e n d e r } )$ </td><td>B6  $\mathrm { ( L G B T Q + ) }$ </td><td>B7  $( { \mathrm { { C u l t u r e } } } )$ </td><td>B8  $( \mathrm { E d u c a t i o n } / \mathrm { C l a s s } )$ </td><td>B9 (Welfare)</td></tr><tr><td rowspan="2">w/o Scout</td><td>Rec.</td><td> $0 . 7 9 4 \pm . 0 2 7$ </td><td> $0 . 8 1 4 \pm . 0 2 4$ </td><td> $0 . 7 7 8 \pm . 0 3 1$ </td><td> $0 . 7 7 6 \pm . 0 2 8$ </td><td> $0 . 8 2 1 \pm . 0 2 3$ </td><td> $0 . 7 8 8 \pm . 0 2 6$ </td><td> $0 . 8 0 7 \pm . 0 2 4$ </td><td> $0 . 7 6 4 \pm . 0 2 9$ </td><td> $0 . 8 1 2 \pm . 0 2 2$ </td></tr><tr><td>F1</td><td> $0 . 8 0 4 \pm . 0 2 1$ </td><td> $0 . 8 2 6 \pm . 0 1 8$ </td><td> $0 . 7 9 4 \pm . 0 2 5$ </td><td> $0 . 7 8 9 \pm . 0 2 2$ </td><td>0.831 ±.017</td><td> $0 . 8 0 6 \pm . 0 1 9$ </td><td> $0 . 8 1 3 \pm . 0 1 8$ </td><td> $0 . 7 8 2 \pm . 0 2 3$ </td><td> $0 . 8 1 9 \pm . 0 1 7$ </td></tr><tr><td rowspan="3">w/ Scout (Ours)</td><td>Rec.</td><td> $0 . 8 8 2 \pm . 0 1 8$ </td><td> $0 . 9 0 1 \pm . 0 1 5$ </td><td> $0 . 8 7 4 \pm . 0 2 1$ </td><td> $0 . 8 7 1 \pm . 0 1 9$ </td><td> $0 . 8 9 4 \pm . 0 1 4$ </td><td> $0 . 8 8 1 \pm . 0 1 8$ </td><td> $0 . 8 8 9 \pm . 0 1 6$ </td><td> $0 . 8 6 6 \pm . 0 2 1$ </td><td> $0 . 8 9 6 \pm . 0 1 5$ </td></tr><tr><td>F1</td><td> $\mathbf { 0 . 8 7 3 \pm . 0 1 5 }$ </td><td> $\mathbf { 0 . 8 9 1 } \pm . 0 1 2$ </td><td> $\mathbf { 0 . 8 6 7 \pm . 0 1 7 }$ </td><td> $\mathbf { 0 . 8 6 4 \pm . 0 1 5 }$ </td><td> $\mathbf { 0 . 8 8 6 \bot } . 0 1 1$ </td><td> $\mathbf { 0 . 8 7 2 \pm . 0 1 4 }$ </td><td> $\mathbf { 0 . 8 8 1 \pm . 0 1 3 }$ </td><td> $\mathbf { 0 . 8 5 4 \ : \pm . 0 1 7 }$ </td><td> $\mathbf { 0 . 8 8 4 \pm . 0 1 2 }$ </td></tr><tr><td>Correct@1 (%) Correct@3 (%)</td><td> $8 3 . 4 \pm 2 . 4$   $9 2 . 5 \pm 1 . 5$ </td><td> $8 7 . 6 \pm 1 . 9$   $9 6 . 3 \pm 0 . 9$ </td><td> $8 2 . 7 \pm 2 . 7$   $9 1 . 8 \pm 1 . 7$ </td><td>86.5 ±2.1  $9 4 . 7 \pm 1 . 2 $ </td><td> $8 8 . 9 \pm 1 . 7$   $9 5 . 6 \pm 1 . 1$ </td><td> $8 4 . 3 \pm 2 . 2$   $9 3 . 4 \pm 1 . 4$ </td><td> $8 1 . 6 \pm 2 . 6 $   $9 2 . 7 \pm 1 . 6 $ </td><td> $8 3 . 2 \pm 2 . 5$   $9 2 . 1 \pm 1 . 6 $ </td><td> $8 7 . 4 \pm 1 . 8$   $9 5 . 2 \pm 1 . 1$ </td></tr></table>

Table 2: Leave-One-Category-Out (LOCO) evaluation on Gemini 2.5 Pro. Best F1 scores are bold. Significance: al w/ Scout gains over w/o Scout remain significant after Bonferroni correction over 9 categories (two-sided paired t-test over 3 runs; max corrected $p = 2 . 4 { \times } 1 0 ^ { - 2 } )$ .

<table><tr><td rowspan="2">Configuration</td><td colspan="2">Detection Quality</td><td rowspan="2">Governance BGS</td><td rowspan="2">Efficiency Tokens/Doc</td></tr><tr><td>Prec. Rec.</td><td>F1</td></tr><tr><td>Full MARS-Gov</td><td>0.865 ±.015 0.895 ±.013 0.880 ±.012</td><td></td><td>0.808 ±.013</td><td> $^ { 2 , 4 5 0 \pm 5 8 }$ </td></tr><tr><td>w/o Verify Loop</td><td>0.865 ±.0150.895 ±.013 0.880 ±.012</td><td></td><td> $0 . 6 4 6 \pm . 0 1 9 ( \downarrow 2 0 . 0 \% ) ^ { * }$ </td><td> $2 , 1 0 0 \pm 4 9$ </td></tr><tr><td>w/o Reasoning</td><td>0.842 ±.0190.870 ±.0170.856 ±.016</td><td></td><td> $0 . 7 6 5 \pm . 0 2 1 ( \downarrow . 5 . 3 \% ) ^ { * }$ </td><td> $^ { 2 , 3 8 0 \pm 6 2 }$ </td></tr><tr><td>w/o NEC</td><td>0.802 ±.021 0.847 ±.0190.824 ±.018</td><td></td><td> $0 . 7 1 0 \pm . 0 2 2 ( \downarrow 1 2 . 1 \% ) ^ { * + }$ </td><td> $2 { , } 2 5 0 \pm 5 4$ </td></tr><tr><td>w/o Scout (10th)</td><td> $| 0 . 8 5 1 \pm . 0 1 7 0 . 8 4 0 \pm . 0 1 9 0 . 8 4 5 \pm . 0 1 5 $ </td><td></td><td>0.760 ±.018 (↓ 5.9%)*</td><td> $2 , 1 5 0 \pm 4 4$ </td></tr><tr><td>w/o ARG</td><td> $\left| 0 . 8 6 5 \pm . 0 1 5 \ 0 . 8 9 5 \pm . 0 1 3 \ 0 . 8 8 0 \pm . 0 1 2 \right|$ </td><td></td><td> $0 . 8 0 8 \pm . 0 1 3$ </td><td>6,100 ±131 (↑ 149%)</td></tr></table>

Table 3: Ablation on DGDB with Gemini 2.5 Pro, averaged over 3 runs. Significance: marked BGS drops vs. Full MARS-Gov remain significant after Bonferroni correction (two-sided paired t-test; $^ { * * } p < 0 . 0 1 , { } ^ { * } p < 0 . 0 5 $ max corrected $p = 2 . 7 { \times } 1 0 ^ { - 2 } )$ . w/o ARG is cost-only.

## 4.4 Ablation & Efficiency Analysis (RQ3–RQ4)

Detection modules. Table 3 shows that Scoutguided dynamic juror review prevents static-juror consensus errors (F1 ↓ 0.035 without it and recall ↓ 5.5%). Removing NEC reduces F1 to 0.824, showing that legal grounding helps separate neutral bureaucratic phrasing from normative violation, especially in cases that are lexically neutral but procedurally stigmatizing.

Governance & efficiency. Removing verification drops BGS from 0.808 to 0.646 and inflates BMR to 98%, because rewrites are no longer checked against the same detector.<sup>5</sup> ARG filters about 65% of documents as CLEAR, reducing token use by roughly 60%. Figure 4 shows a stable elbow at $\tau =$ 2, where over-governance falls with little recall loss across backbones.

![](images/5833eff50ac68cf3b9492b45f4addf774a2d1ca34b4335ee63be0caaab8095a1.jpg)  
Figure 4: Recall vs. Over-Governance Rate (OGR) trade-off as threshold τ increases.

<table><tr><td>Configuration</td><td>Calls</td><td>Lat.(s)</td><td>Clear%</td><td>$/1k</td></tr><tr><td colspan="5">Baselines (Gemini 2.5 Pro)</td></tr><tr><td>Zero-shot LLM Standard Debate SA-RAG</td><td>1.0 4.0 1.0</td><td>1.8 (0.3) 7.2 (0.9) 3.4 (0.5)</td><td></td><td>0.85 4.62 2.46</td></tr><tr><td colspan="5">MARS-Gov (with ARG)</td></tr><tr><td>GPT-40 Grok-3 Gemini 2.5 Pro</td><td>12.8 (0.7) 18.7 (2.1) 63.5 (1.9) 12.4 (0.5)</td><td>12.6 (0.6) 20.5 (2.3) 64.2 (1.7) 10.04 )19.2 (2.2) 64.8 (1.6)</td><td></td><td>13.39 7.35 4.25</td></tr><tr><td colspan="5">DeepSeek-R1 12.5 (0.5) 31.6 (3.2) 65.1 (1.8)</td></tr><tr><td>w/o ARG (Gemini) 28.6 (0.8) 41.5 (3.7)</td><td colspan="3"></td><td>18.30</td></tr></table>

Table 4: Per-sentence inference cost on DGDB (1,000 sentences, 3 runs; mean with σ). Calls counts all backbone calls; Clear% is the early-stop share.

Table 4 reports cost. The first-pass detector uses NEC, the Scout, and nine fixed jurors; dynamic jurors are called only for uncovered targets, one per target, and the rule-based gate adds no LLM call. With Gemini 2.5 Pro, ARG clears 64.8% of inputs early and reduces calls from 28.6 to 12.4, latency from 41.5s to 19.2s, and cost from \$18.30 to \$7.35 per 1,000 sentences. Since removing the gate leaves detection unchanged but roughly doubles cost, ARG acts as an efficiency component rather than a hidden classifier.

<table><tr><td rowspan="2">Configuration</td><td colspan="3">DALC (Dutch, Social Media)</td><td colspan="3">SBIC (English, Social Media)</td><td colspan="3">KoBBQ (Korean, General)</td></tr><tr><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>Zero-shot LLMs</td><td> $0 . 5 1 3 \pm . 0 2 8$ </td><td> $0 . 8 7 3 \pm . 0 2 1$ </td><td> $0 . 6 4 6 \pm . 0 1 8$ </td><td> $0 . 4 8 6 \pm . 0 3 3$ </td><td> $0 . 8 4 4 \pm . 0 2 3$ </td><td> $0 . 6 1 7 \pm . 0 1 9$ </td><td> $0 . 4 5 1 \pm . 0 3 6$ </td><td> $0 . 8 0 3 \pm . 0 2 7$ </td><td> $0 . 5 7 8 \pm . 0 2 4$ </td></tr><tr><td>Standard Debate</td><td> $0 . 6 0 7 \pm . 0 2 5$ </td><td> $0 . 8 5 1 \pm . 0 2 0$ </td><td> $0 . 7 0 9 \pm . 0 1 9$ </td><td> $0 . 5 8 3 \pm . 0 2 7$ </td><td> $0 . 8 1 9 \pm . 0 2 1$ </td><td> $0 . 6 8 1 \pm . 0 2 1$ </td><td> $0 . 5 5 3 \pm . 0 2 9$ </td><td> $0 . 8 1 1 \pm . 0 2 4$ </td><td> $0 . 6 5 8 \pm . 0 2 3$ </td></tr><tr><td>SA-RAG</td><td> $0 . 6 5 3 \pm . 0 2 1$ </td><td> $0 . 8 6 2 \pm . 0 1 8$ </td><td> $0 . 7 4 3 \pm . 0 1 6$ </td><td> $0 . 6 2 6 \pm . 0 2 3$ </td><td> $0 . 8 3 6 \pm . 0 1 9$ </td><td> $0 . 7 1 6 \pm . 0 1 8$ </td><td> $0 . 5 9 3 \pm . 0 2 7$ </td><td> $0 . 8 2 1 \pm . 0 2 1$ </td><td> $0 . 6 8 9 \pm . 0 1 9$ </td></tr><tr><td>Ours: MARS-Gov</td><td> $\mathbf { 0 . 8 2 3 \pm . 0 1 5 }$  </td><td> $\mathbf { 0 . 8 8 6 \pm . 0 1 3 }$ </td><td> $\mathbf { 0 . 8 5 3 \pm . 0 1 2 }$ </td><td> $\mathbf { 0 . 8 0 3 \pm . 0 1 8 }$ </td><td>一  $\mathbf { 0 . 8 4 7 \pm . 0 1 6 }$ </td><td> $\mathbf { 0 . 8 2 4 \bot } 0 1 5$ </td><td> $\mathbf { 0 . 7 7 6 \pm . 0 2 1 }$  </td><td> $\mathbf { 0 . 8 2 9 \pm . 0 1 8 }$  </td><td> $\mathbf { 0 . 8 0 2 \pm . 0 1 7 }$ </td></tr></table>

Table 5: Generalization on DALC, SBIC, and KoBBQ using Gemini 2.5 Pro. Results average 3 runs; best F1 scores are bold. Significance: MARS-Gov beats SA-RAG after Bonferroni correction over 3 datasets (two-sided paired t-test; max corrected $p = 1 . 1 { \times } 1 0 ^ { - 2 } )$ .

## 4.5 Cross-Domain and Cross-Lingual Generalization (RQ5)

We test transfer on DALC, SBIC, and KoBBQ using balanced 1,000-instance samples and Gemini 2.5 Pro. Table 5 shows that MARS-Gov remains strongest across all three datasets. The gain mainly comes from precision: the review loop rejects unsupported group-risk inferences left by zero-shot detection, adding 20–30 precision points while keeping recall comparable. Absolute F1 remains below in-domain DGDB because the external benchmarks shift genre, label assumptions, and language.

These results support the portability of the review procedure, not of Dutch law itself: deployment in another jurisdiction would still need a local legal and policy evidence base.

## 5 Conclusion

We presented MARS-Gov, a standpoint-aware framework for closed-loop bias governance in Dutch government documents. Rather than treating bias detection as one-shot classification over a fixed taxonomy, MARS-Gov combines legal retrieval, open-set target screening, juror-based adjudication, conservative routing, and rewrite verification. Experiments show stronger performance than fine-tuned, zero-shot, and generic debate baselines, with higher open-set recovery and BGS, better cross-domain and cross-lingual transfer, and lower over-governance. These results suggest that reliable governance NLP requires more than stronger classifiers: it requires legally grounded evidence, explicit standpoint reasoning, and verification after interventions. The DGDB-Unseen results sharpen this point: compact encoders remain competitive on familiar terms, but lose reliability when public language shifts. Future work will extend the framework to additional jurisdictions and reduce inference cost for broader evaluation and lowercost deployment.

## Limitations

Our framework remains dependent on domainspecific normative resources. Although the external experiments suggest that the core mechanism can transfer across languages and domains, MARS-Gov’s strongest setting is Dutch administrative text because its evidence base is built around Dutch anti-discrimination law and administrative guidance. Applying the framework to other jurisdictions or platforms therefore requires localized legal and policy resources, not prompt translation alone; otherwise, the system may produce justifications that are linguistically plausible but normatively incorrect.

A second limitation is inference cost. The multiagent pipeline is more expensive and slower than lightweight discriminative models, even with the routing gate. This cost is acceptable for high-stakes public-sector review, where false positives and false negatives can both have institutional consequences, but it limits use in high-throughput or real-time settings. Reducing the number of model calls without weakening legal grounding or verification remains an important direction for future work.

Finally, BGS is an aggregate governance metric rather than a complete diagnostic. Because BGS = Recall×BMR×SF (Eq. 14), different systems can obtain similar scores through different trade-offs between detection coverage, mitigation success, and meaning preservation. We therefore report the individual components alongside BGS in Table 1. BGS should be interpreted as a compact summary of governance performance, not as a substitute for component-level analysis.

## Ethics Statement

This work addresses a socially beneficial but highstakes NLP setting: identifying potentially biased language in Dutch government documents and proposing minimally edited alternatives. MARS-Gov is not intended to act as an autonomous decision-maker or to automatically rewrite official records. In any real deployment, it should be used only as a human-in-the-loop decision-support system: it may flag potentially problematic passages, present evidence and rationales, and suggest candidate rewrites for review by qualified experts such as legal professionals, policy officers, or ethics reviewers. The original text must remain preserved in the auditable record, especially in historical or contested cases, so that affected individuals retain access to recourse and institutions remain accountable for past wording. We also do not claim that neutralized wording eliminates discriminatory practice. A rewrite should be treated as a signal for substantive institutional review, including review of policy rules, risk indicators, and administrative procedures.

The normative resources used by MARS-Gov, including watchlists, criteria, and protected-group standpoints, are not fixed truths. They should be transparently documented, versioned, periodically audited, and revised with input from affected communities, domain experts, and legal stakeholders. Since LLMs can reproduce social stereotypes, model confidence should not be treated as correctness, and high-risk outputs require manual verification, error analysis, and monitoring for disparate impact.

Finally, the system has dual-use risks. An unrestricted rewriting tool could be misused to sanitize discriminatory policy language without changing the underlying policy. For this reason, we release the code and prompts as research artifacts rather than as a turnkey tool for live administrative deployment, and we do not recommend use in fully automated government decision pipelines.

## Acknowledgments

This work was supported by the National College Student Innovation and Entrepreneurship Training Program under Project No. 202619145026; the Open Project Program of Marine Ecological Restoration and Smart Ocean Engineering Research Center of Hebei Province under Grant No. HBMESO2507; the Shijiazhuang– Northeastern University Science and Technology Cooperation Special Project under Grant No. NEUS2025-01-003; and the Scientific Research Project of Hebei Education Department under Grant No. QN2024167.

## References

Gordon W. Allport. 1954. The Nature of Prejudice. Addison-Wesley.

Ruha Benjamin. 2019. Race After Technology: Abolitionist Tools for the New Jim Code. Polity.

Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. 2020. Language (technology) is power: A critical survey of “bias” in NLP. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics. Association for Computational Linguistics.

Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam T. Kalai. 2016. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In Advances in Neural Information Processing Systems.

Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. 2017. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183–186.

Tommaso Caselli, Arjan Schelhaas, Marieke Weultjes, Folkert Leistra, Hylke Van Der Veen, Gerben Timmerman, and Malvina Nissim. 2021. DALC: the Dutch abusive language corpus. In Proceedings of the 5th Workshop on Online Abuse and Harm (WOAH 2021), pages 54–66. Association for Computational Linguistics.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. LEGAL-BERT: The muppets straight out of law school. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2898– 2904, Online. Association for Computational Linguistics.

Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Katz, and Nikolaos Aletras. 2022. LexGLUE: A benchmark dataset for legal language understanding in English. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland. Association for Computational Linguistics.

Kimberlé Crenshaw. 1989. Demarginalizing the intersection of race and sex: A black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. University ofChicago Legal Forum.

Milena de Swart, Floris den Hengst, and Jieying Chen. 2025. Detecting linguistic bias in government documents using large language models. In WWW 2025: Proceedings of the ACM on Web Conference 2025, pages 5034–5044. Association for Computing Machinery.

Wietse de Vries, Andreas van Cranenburgh, Arianna Bisazza, Tommaso Caselli, Gertjan van Noord, and Malvina Nissim. 2019. Bertje: A dutch bert model. arXiv preprint arXiv:1912.09582.

Pieter Delobelle, Thomas Winters, and Bettina Berendt. 2020. RobBERT: a dutch RoBERTa-based language model. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 3255–3265, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long and Short Papers), Minneapolis, Minnesota. Association for Computational Linguistics.

Lucas Dixon, John Li, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2018. Measuring and mitigating unintended bias in text classification. In Proceedings ofthe 2018 AAAI/ACM Conference on AI, Ethics, and Society, pages 67–73. ACM.

Robert M. Entman. 1993. Framing: Toward clarification of a fractured paradigm. Journal ofCommunication.

Virginia Eubanks. 2018. Automating Inequality: How High-Tech Tools Profile, Police, and Punish the Poor. St. Martin’s Press.

Norman Fairclough. 1992. Discourse and Social Change. Polity Press, Cambridge, UK.

Susan T. Fiske. 1998. Stereotyping, prejudice, and discrimination. In Daniel T. Gilbert, Susan T. Fiske, and Gardner Lindzey, editors, The Handbook of Social Psychology, 4 edition, pages 357–411. McGraw-Hill.

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md. M. Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Rui Zhang, and Nesreen K. Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics.

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020. Association for Computational Linguistics.

Erving Goffman. 1974. Frame Analysis: An Essay on the Organization ofExperience. Harper & Row.

Yuxiang He, Jian Zhao, Yuchen Yuan, Tianle Zhang, Wei Cai, Haojie Cheng, Ziyan Shi, Ming Zhu, Haichuan Tang, Chi Zhang, and Xuelong Li. 2025. Aetheria: A multimodal interpretable content safety framework based on multi-agent debate and collaboration. Preprint, arXiv:2512.02530.

Dan Hendrycks and Kevin Gimpel. 2017. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations.

Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. Metagpt: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations.

Wonje Jeung, Dongjae Jeon, Ashkan Yousefpour, and Jonghyun Choi. 2025. Large language models still exhibit bias in long text. In Findings of the Association for Computational Linguistics: ACL 2025, pages 26147–26169, Vienna, Austria. Association for Computational Linguistics.

Jiho Jin, Jiseon Kim, Nayeon Lee, Haneul Yoo, Alice Oh, and Hwaran Lee. 2024. KoBBQ: Korean bias benchmark for question answering. Transactions ofthe Associationfor Computational Linguistics, 12:507–524.

Tian Lan, Jiang Li, Yemin Wang, Xu Liu, Xiangdong Su, and Guanglai Gao. 2025. F²Bench: An openended fairness evaluation benchmark for LLMs with factuality considerations. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2031–2046, Suzhou, China. Association for Computational Linguistics.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: Communicative agents for “mind” exploration of large language model society. arXiv preprint arXiv:2303.17760.

Marta Marchiori Manerba, Karolina Stanczak, Riccardo Guidotti, and Isabelle Augenstein. 2024. Social bias probing: Fairness benchmarking for language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 14653–14671, Miami, Florida, USA. Association for Computational Linguistics.

Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. Stereoset: Measuring stereotypical bias in pretrained language models. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics.

Yucheng Ning, Xixun Lin, Fang Fang, and Yanan Cao. 2025. Mad-fact: A multi-agent debate framework for long-form factuality evaluation in llms. Preprint, arXiv:2510.22967.

Safiya Umoja Noble. 2018. Algorithms of Oppression: How Search Engines Reinforce Racism. NYU Press.

Yulia Otmakhova, Shima Khanehzar, and Lea Frermann. 2024. Media framing: A typology and survey of computational approaches across disciplines. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15407–15428, Bangkok, Thailand. Association for Computational Linguistics.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442.

Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, Samuel R. Bowman, and Alex Warstadt. 2022. BBQ: A hand-built bias benchmark for question answering. In Findings of the Association for Computational Linguistics: ACL 2022. Association for Computational Linguistics.

Nicholas Pipitone and Ghita Houir Alami. 2024. LegalBench-RAG: A benchmark for retrievalaugmented generation in the legal domain. arXiv preprint arXiv:2408.10343.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Chatdev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics.

Arti Rani, Shweta Singh, Nihar Ranjan Sahoo, and Gaurav Kumar Nayak. 2025. Open-DeBias: Toward mitigating open-set bias in language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 25027–25051, Suzhou, China. Association for Computational Linguistics.

Philip Resnik. 2025. Large language models are biased because they are large language models. Computational Linguistics, 51(3):885–906.

Aya Rizk and Ida Lindgren. 2025. Automated decisionmaking in public administration: Changing the decision space between public officials and citizens. Government Information Quarterly, 42(3):102061.

Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A. Smith, and Yejin Choi. 2020. Social bias frames: Reasoning about social and power implications of language. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5477–5490, Online. Association for Computational Linguistics.

Walter J. Scheirer, Anderson Rocha, Archie Sapkota, and Terrance E. Boult. 2013. Toward open set recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7):1757–1772.

Eric Michael Smith, Melissa Hall, Melanie Kambadur, Eleonora Presani, and Adina Williams. 2022. “I’m sorry to hear that”: Finding new biases in language models with a holistic descriptor dataset. In Proceed ings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9180–9211. Association for Computational Linguistics.

Teun A. van Dijk. 1993. Principles of critical discourse analysis. Discourse & Society.

Zeerak Waseem and Dirk Hovy. 2016. Hateful symbols or hateful people? predictive features for hate speech detection on twitter. In Proceedings of the NAACL Student Research Workshop. Association for Computational Linguistics.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang. 2023. Autogen: Enabling next-gen LLM applications via multi-agent conversation. arXiv preprint arXiv:2308.08155.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers). Association for Computational Linguistics.

## Appendix Index

1. Dataset Overview (p. 12)   
2. Extended Benchmarks across Model Families   
(p. 12)   
• Analysis of Extended Results (p. 13)   
• Cross-Lingual Reasoning (p. 13)   
3. LOCO Prototypes (p. 13)   
• Fixed Prototype Texts for Deterministic   
Mapping (p. 13)   
4. Representative End-to-End Decision Traces   
(p. 14)   
5. Knowledge Base Construction and RAG   
Strategies (p. 15)   
• DGDB NEC Retrieval Setting (p. 15)   
• Knowledge Base Composition (p. 15)   
• Implementation Parameters (p. 15)   
6. ARG Hyperparameter Sensitivity Grid (p. 15)   
• Performance Heatmaps (F1 vs. OGR)   
(p. 15)   
7. English and Dutch Prompt Templates (p. 16) 7. English and Dutch Prompt Templates (p. 16)

## A Dataset Overview

Tables 6 and 7 summarize the datasets used in the experiments.

## B Extended Benchmarks across Model Families

This appendix tests whether MARS-Gov’s gains depend on a single frontier backbone. We evaluate

<table><tr><td>Statistics</td><td>DGDB</td><td>DGDB-Unseen</td></tr><tr><td>Language</td><td>Dutch</td><td>Dutch</td></tr><tr><td>Domain</td><td>Gov. docs</td><td>Gov. docs</td></tr><tr><td>Unit</td><td>Sentence</td><td>Sentence</td></tr><tr><td>Corpus size</td><td>3,747</td><td>DGDB-derived</td></tr><tr><td>Evaluation size</td><td>Test split</td><td>Held-out subset</td></tr><tr><td>Construction</td><td>Original split</td><td>Bias-term substitu- tion</td></tr><tr><td>Evaluation balance Original split</td><td></td><td>Label-preserving</td></tr><tr><td>Targets / labels</td><td>9 bias cats.</td><td>Same 9 bias cats.</td></tr></table>

Table 6: Statistical overview of the DGDB-derived datasets.
<table><tr><td>Statistics</td><td>DALC</td><td>SBIC</td><td>KoBBQ</td></tr><tr><td>Language</td><td>Dutch</td><td>English</td><td>Korean</td></tr><tr><td>Domain</td><td>Twitter</td><td>Social</td><td>QA</td></tr><tr><td>Unit</td><td>Tweet</td><td>Post / tuple</td><td>MCQ</td></tr><tr><td>Corpus size</td><td>8,156</td><td>44,671/147,139</td><td>76,048</td></tr><tr><td>Evaluation size</td><td>1,000</td><td>1,000</td><td>1,000</td></tr><tr><td>Evaluation balance 500 / 500</td><td></td><td>500 / 500</td><td>500 / 500</td></tr><tr><td>Targets / labels</td><td>Abusive binary</td><td>Bias tuple</td><td>Bias contexts</td></tr></table>

Table 7: Statistical overview of the external evaluation datasets.

18 LLMs, grouped into Frontier SOTA, Strong Generalist, and Mid/Efficiency tiers. Table 8 reports the corresponding zero-shot and MARS-Gov results on DGDB.

## B.1 Analysis of Extended Results

The extended results show broad gains across backbones, but the final ceiling still depends on reasoning capacity. Models at or above the GPT-4 Turbo / Llama-3-70B tier reliably exceed the fine-tuned BERTje baseline, including open-weight options such as Qwen-2.5-72B and DeepSeek-V3.

## B.2 Cross-Lingual Reasoning

Native vs. English Reasoning. Figure 5 compares Dutch prompting with English prompting after DeepL translation. All scores are averaged over 3 independent runs.

Dutch prompting gives a small but consistent F1 advantage across all four backbones (+0.8% to +2.3%), mainly because administrative terms such as "afstand tot de arbeidsmarkt" and "leefvorm" lose part of their meaning after translation.

## C LOCO Prototypes

To support reproducibility of the deterministic open-set evaluation, we provide the full fixed prototype texts e(c) used for cosine-similarity mapping. LOCO uses clips/e5-base-trm-nl for prototype ranking, whereas NEC uses the separate retrieval embedding model described in Appendix E.3.

![](images/5ec39a6cde9be08e1b0e73f689f2971f49c3039ae34060662cad52f50e99c63c.jpg)  
Figure 5: Comparison of F1 scores between Native Dutch and Translated English prompts. Native prompting consistently outperforms English across all backbones, avoiding the semantic loss associated with translating specific administrative terminology (p < 0.05, paired Student’s t-test over 3 independent runs).

## C.1 Fixed Prototype Texts for Deterministic Mapping

The following prototype vectors were fixed before the experiment based on dataset statistics and definitions. The Scout’s generated group and rationale were mapped against these to calculate Correct@1 and Correct@3.

• B1 (Disability): "beperkingen, handicap, rolstoelgebonden, dwerg, doventolk, medische toestand, fysieke of mentale beperking."

• B2 (Migration): "migratie, allochtoon, vluchteling, nieuwkomer, asielzoeker, integratie, westers, niet-westers."

• B3 (Colonialism): "kolonialisme, slavernijverleden, inheems, etnisch, blank, zwart, ras, exotisch, primitief."

• B4 (Religion): "geloof, religie, christen, moslim, islam, joods, hoofddoek, extremisme."

• B5 (Gender): "gender, geslacht, man, vrouw, meisje, jongen, seksisme, traditionele rolpatronen."

• B6 (LGBTQ+): "seksualiteit, geaardheid, homo, hetero, queer, transgender, non-binair, lhbtiq+."

• B7 (Culture): "cultuur, traditie, bi-cultureel, tussenpositie, culturele achtergrond, minderheden."

• B8 (Education/Class): "onderwijs, klasse, achterstandsleerling, mbo, laagopgeleid, probleemwijk, sociale status."

• B9 (Welfare): "algemeen, armoede, machtsongelijkheid,fobie, sociaal-economische kwetsbaarheid."

<table><tr><td rowspan="2">Model Family</td><td rowspan="2">Model Name</td><td colspan="3">Standard Zero-shot</td><td colspan="3">Ours: MARS-Gov</td><td rowspan="2">Governance BGS</td></tr><tr><td>Prec</td><td>Rec</td><td>F1</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td rowspan="4">Frontier SOTA</td><td>Gemini 2.5 Pro DeepSeek-R1</td><td> $0 . 5 2 4 \pm . 0 2 4$   $0 . 5 4 9 \pm . 0 2 3$ </td><td> $0 . 9 0 7 \pm . 0 1 8$   $0 . 8 8 7 \pm . 0 1 9$ </td><td> $0 . 6 6 4 \pm . 0 1 6$   $0 . 6 7 8 \pm . 0 1 5$ </td><td> $\mathbf { 0 . 8 6 5 \pm . 0 1 5 }$   $\mathbf { 0 . 8 7 2 \pm . 0 1 6 }$ </td><td> $0 . 8 9 5 { \scriptstyle \pm . 0 1 3 }$   $0 . 8 8 4 \pm . 0 1 4$ </td><td> $\mathbf { 0 . 8 8 0 \pm . 0 1 2 }$   $0 . 8 7 8 \pm . 0 1 3$ </td><td>0.808 ±.013  $0 . 8 0 6 \pm . 0 1 5$ </td></tr><tr><td>Claude 3.5 Sonnet</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Grok-3</td><td> $\mathbf { 0 . 5 6 2 \pm . 0 2 2 }$ </td><td> $0 . 8 7 2 { \scriptstyle \pm . 0 2 0 }$ </td><td> $\mathbf { 0 . 6 8 3 \pm . 0 1 6 }$ </td><td> $0 . 8 6 0 \pm . 0 1 7$ </td><td> $0 . 8 8 2 \pm . 0 1 5$ </td><td> $0 . 8 7 1 \pm . 0 1 3$ </td><td> $0 . 7 8 7 \pm . 0 1 6$ </td></tr><tr><td>GPT-40</td><td> $0 . 4 8 4 \pm . 0 3 1$   $0 . 5 1 4 \pm . 0 2 5$ </td><td> $\mathbf { 0 . 9 2 0 } \pm . 0 1 7$   $0 . 8 8 2 \pm . 0 2 1$ </td><td> $0 . 6 3 4 \pm . 0 1 9$   $0 . 6 4 9 \pm . 0 1 7$ </td><td> $0 . 8 3 6 \pm . 0 1 9$   $0 . 8 4 4 \pm . 0 1 7$ </td><td> $\mathbf { 0 . 9 1 0 \bot } . 0 1 3$   $0 . 8 6 3 \pm . 0 1 5$ </td><td> $0 . 8 7 1 \pm . 0 1 4$   $0 . 8 5 3 \pm . 0 1 3$ </td><td> $0 . 7 7 9 \pm . 0 1 8$   $0 . 7 5 7 \pm . 0 1 4$ </td></tr><tr><td rowspan="6">Strong Generalist</td><td>Claude 3 Opus</td><td> $0 . 5 3 1 \pm . 0 2 6$ </td><td> $0 . 8 7 2 \pm . 0 2 1$ </td><td> $0 . 6 6 0 \pm . 0 1 8$ </td><td> $0 . 8 5 2 \pm . 0 1 8$ </td><td> $0 . 8 6 7 \pm . 0 1 6$ </td><td> $0 . 8 5 9 \pm . 0 1 4$ </td><td> $0 . 7 4 9 \pm . 0 1 8$ </td></tr><tr><td>GPT-4 Turbo</td><td> $0 . 4 9 4 \pm . 0 3 1$ </td><td> $0 . 8 6 2 \pm . 0 2 0$ </td><td> $0 . 6 2 8 \pm . 0 2 1$ </td><td> $0 . 8 3 6 \pm . 0 1 9$ </td><td> $0 . 8 5 2 \pm . 0 1 8$ </td><td> $0 . 8 4 4 \pm . 0 1 5$ </td><td> $0 . 7 1 7 \pm . 0 1 7$ </td></tr><tr><td>DeepSeek-V3</td><td> $0 . 4 8 2 \pm . 0 3 3$ </td><td> $0 . 8 4 6 \pm . 0 2 3$ </td><td> $0 . 6 1 4 \pm . 0 2 0$ </td><td> $0 . 8 1 1 \pm . 0 2 2$ </td><td> $0 . 8 3 2 { \scriptstyle \pm . 0 1 9 }$ </td><td> $0 . 8 2 1 \pm . 0 1 7$ </td><td> $0 . 6 8 2 \pm . 0 2 1$ </td></tr><tr><td>Qwen-2.5-72B</td><td> $0 . 4 7 2 \pm . 0 3 6$ </td><td> $0 . 8 5 2 \pm . 0 2 4$ </td><td> $0 . 6 0 8 \pm . 0 2 3$ </td><td> $0 . 8 1 9 \pm . 0 2 1$ </td><td> $0 . 8 3 9 \pm . 0 1 9$ </td><td> $0 . 8 2 9 { \scriptstyle \pm . 0 1 6 }$ </td><td> $0 . 6 9 6 \pm . 0 1 9$ </td></tr><tr><td>Mistral Large 2</td><td> $0 . 4 9 1 \pm . 0 3 4$ </td><td> $0 . 8 2 8 \pm . 0 2 6$ </td><td> $0 . 6 1 6 \pm . 0 2 2$ </td><td> $0 . 8 0 6 \pm . 0 2 2$ </td><td> $0 . 8 2 6 \pm . 0 1 9$ </td><td> $0 . 8 1 6 \pm . 0 1 8$ </td><td> $0 . 6 5 4 \pm . 0 2 2$ </td></tr><tr><td>Llama-3-70B</td><td> $0 . 4 6 3 \pm . 0 3 7$ </td><td> $0 . 8 3 7 \pm . 0 2 4$ </td><td> $0 . 5 9 6 \pm . 0 2 3$ </td><td> $0 . 8 0 0 \pm . 0 2 3 $ </td><td> $0 . 8 2 0 \pm . 0 2 0$ </td><td> $0 . 8 1 0 { \scriptstyle \pm . 0 1 7 }$ </td><td> $0 . 6 6 8 \pm . 0 2 3$ </td></tr><tr><td colspan="10">Performance Threshold: Fine-tuned BERTje  $\overset { - } { ( F I = 0 . 8 I 2 ) } -$ </td></tr><tr><td rowspan="7">Mid/Efficiency</td><td colspan="10">Gemini 1.5 Pro</td></tr><tr><td>Qwen-2.5-32B</td><td> $0 . 4 7 8 \pm . 0 3 8$   $0 . 4 2 2 \pm . 0 4 1$ </td><td> $0 . 8 2 1 \pm . 0 2 7$   $0 . 7 9 8 \pm . 0 2 8$ </td><td> $0 . 6 0 4 \pm . 0 2 4$   $0 . 5 5 2 \pm . 0 2 6$ </td><td> $0 . 7 8 1 \pm . 0 2 4$   $0 . 7 4 2 \pm . 0 2 6$ </td><td> $0 . 7 9 8 \pm . 0 2 1$   $0 . 7 6 2 \pm . 0 2 3$ </td><td> $0 . 7 8 9 \pm . 0 1 8$   $0 . 7 5 2 \pm . 0 1 9$ </td><td> $0 . 6 1 2 \pm . 0 2 2$   $0 . 5 7 5 \pm . 0 2 4$ </td></tr><tr><td>Claude 3 Haiku</td><td> $0 . 4 1 2 \pm . 0 4 3$ </td><td> $0 . 7 7 8 \pm . 0 3 1$ </td><td> $0 . 5 3 9 \pm . 0 2 9$ </td><td> $0 . 7 1 8 \pm . 0 2 7$ </td><td> $0 . 7 3 8 \pm . 0 2 3$ </td><td> $0 . 7 2 8 \pm . 0 2 1$ </td><td> $0 . 5 3 9 \pm . 0 2 7$ </td></tr><tr><td>Gemini 1.5 Flash</td><td>0.401 ±.042</td><td> $0 . 7 8 9 \pm . 0 3 0$ </td><td> $0 . 5 3 2 \pm . 0 2 7$ </td><td> $0 . 7 0 3 \pm . 0 2 7$ </td><td> $0 . 7 2 3 \pm . 0 2 5$ </td><td> $0 . 7 1 3 \pm . 0 2 1$ </td><td> $0 . 5 2 1 \pm . 0 2 7$ </td></tr><tr><td>Mixtral 8x7B</td><td>0.382 ±.045</td><td> $0 . 7 6 2 \pm . 0 3 3$ </td><td> $0 . 5 0 9 \pm . 0 3 1$ </td><td> $0 . 6 7 1 \pm . 0 2 9$ </td><td> $0 . 6 9 4 \pm . 0 2 6$ </td><td> $0 . 6 8 2 \pm . 0 2 3$ </td><td> $0 . 4 8 6 \pm . 0 2 9$ </td></tr><tr><td>GPT-3.5 Turbo</td><td> $0 . 3 5 2 \pm . 0 4 8$ </td><td> $0 . 7 4 8 \pm . 0 3 6$ </td><td> $0 . 4 7 9 \pm . 0 3 3$ </td><td> $0 . 6 3 3 \pm . 0 3 2$ </td><td> $0 . 6 5 4 \pm . 0 2 7$ </td><td> $0 . 6 4 3 \pm . 0 2 4$ </td><td> $0 . 4 4 8 \pm . 0 3 1$ </td></tr><tr><td>Llama-3-8B</td><td> $0 . 3 2 1 \pm . 0 5 1$ </td><td> $0 . 7 1 2 \pm . 0 3 7$ </td><td> $0 . 4 4 2 \pm . 0 3 5$ </td><td> $0 . 5 9 1 \pm . 0 3 3$ </td><td> $0 . 6 1 1 \pm . 0 2 9$ </td><td> $0 . 6 0 1 \pm . 0 2 6$ </td><td> $0 . 3 9 7 \pm . 0 3 3$ </td></tr></table>

Table 8: Extended Leaderboard on DGDB (In-Domain). Models are ranked by their MARS-Gov F1 score. The red dashed line marks the performance threshold of the fine-tuned BERTje baseline $( F 1 = 0 . 8 1 2 )$ . Observation: While MARS-Gov improves performance across model families (avg. $\Delta \mathrm { F } 1 + 0 . 2 0 $ , larger for frontier backbones than for smaller ones), a minimum reasoning capability threshold (approx. GPT-4 Turbo level) is required to surpass supervised baselines.

## D Representative End-to-End Decision Traces

We provide three representative traces that illustrate the main decision paths: severe intersectional bias, false inference from language proficiency, and a neutral procedural case. These traces explain the decision mechanism; aggregate detection and ablation results are reported in Tables 1 and 3. The full prompt definitions used to produce these traces are available in agent\_bias/common/prompts.py.

## Case T1: Intersectional welfare-fraud profiling

Input. "Om uitkeringsfraude tegen te gaan, moet de gemeente strengere huisbezoeken uitvoeren bij allochtone alleenstaande moeders in de Schilderswijk, aangezien deze groep historisch gezien een gebrek aan administratief inzicht heeft."

cators.

Trace. The evidence layer retrieves antidiscrimination and profiling guidance. The Scout identifies migration, gender, and neighborhood/class proxies. Jurors for migration, gender, class, and welfare flag the sentence because it links protected and socio-economic attributes to fraud risk and administrative incapacity.

Decision. The Judge returns BIASED; the rewrite preserves the fraud-control goal but replaces group targeting with objective risk indi-

## Case T2: Language proficiency as intent inference

Input. "De aanvrager toonde tijdens het gesprek een gebrekkige beheersing van de Nederlandse taal; dit duidt op een gebrek aan integratie en een verhoogde kans op opzettelijke misleiding."

Trace. The Scout identifies non-native speakers as the affected group. Jurors flag the unsupported causal chain from language ability to integration failure and intentional deception.

Decision. The Judge returns BIASED; the rewrite reframes the observation as a communication-support need while preserving the administrative fact.

## Case T3: Objective income threshold

Input. "Om in aanmerking te komen voor de huurtoeslag, mag het gezamenlijke toetsingsinkomen niet hoger zijn dan C32.500 per jaar."

Trace. The evidence layer identifies a statutory eligibility threshold. The Scout finds no protected or stigmatized group target beyond the general applicant population. Jurors treat the sentence as a neutral criterion.

Decision. The Judge returns NON-BIASED, and restoration is skipped.

## E Knowledge Base Construction and RAG Strategies

This appendix summarizes the NEC knowledge base K and retrieval settings used by MARS-Gov.

## E.1 DGDB NEC Retrieval Setting

For DGDB, NEC retrieval uses the source inventory released with the code. It combines legal and policy foundations, accessibility and administrative communication guidance, reports on online discrimination and institutional racism, inclusivelanguage resources, and DGDB category reasoning notes.

## E.2 Knowledge Base Composition

The curated repository K is structured into three functional layers:

• Legal and Policy Core (Grondslag): Grondwet Chapter 1, AWGB core articles, OCW antidiscrimination agenda material, the Dutch national programme against discrimination and racism, and ECRI Netherlands extracts.

• Administrative and Accessibility Guides (Overheidsrichtlijnen): accessible voting plans, election communication guidance, and official inclusive and accessible writing guides.

• Sociocultural and Inclusive-Language Resources (Inclusiviteit): online-discrimination and harmful-behavior reports, institutionalracism summaries, emancipation monitoring, DEBIAS vocabulary entries, inclusive communication glossaries, cultural-sector language guidance, and DGDB reasoning entries.

Table 9 summarizes how this inventory supports the DGDB retrieval setting.

## E.3 Implementation Parameters

NEC utilizes text-embedding-3-small for RAG dense retrieval. This is separate from the clips/e5-base-trm-nl encoder used for semantic fidelity and LOCO prototype ranking. The released implementation first retrieves keyword matches, then dense semantic matches, and finally expands selected hits to adjacent chunks from the same source. Long policy documents use persource character windows, while DGDB reasoning entries are split by markdown headings. At inference, the default setting selects $k = 5$ seed fragments before local context expansion, ensuring the Summary Agent synthesizes a contextually grounded, neutral evidence base.

<table><tr><td>Dimension</td><td>DGDB NEC retrieval</td></tr><tr><td>Primary Objective</td><td>Detecting implicit bias in official policy texts.</td></tr><tr><td>Core Domain</td><td>Legal and political administrative text (high- stakes).</td></tr><tr><td>Inventory</td><td>Dutch constitutional and equal-treatment law; antidiscrimination policy; accessibil- ity and election communication guidance; online-discrimination, institutional-racism, emancipation, and inclusive-language re- sources; DGDB reasoning entries.</td></tr><tr><td>Selection Rationale</td><td>Grounds the review in legal norms while covering administrative, accessibility, and inclusive-language contexts that commonly shape implicit bias.</td></tr></table>

Table 9: NEC retrieval construction for DGDB. MARS-Gov prioritizes normative sources and official writing guidance for administrative bias review.

The three retrieval stages are complementary: keyword matching preserves precise legal and administrative terminology, dense retrieval recovers paraphrases, and local expansion restores nearby definitions or caveats. Source identifiers and chunk positions are retained in the decision trace, allowing the retrieved evidence to be inspected independently of the verdict. The Summary Agent condenses this material but cannot classify the sentence, keeping retrieval errors separate from later reasoning errors.

## F ARG Hyperparameter Sensitivity Grid

We tune ARG on the validation set with Gemini 2.5 Pro. Five parameters govern escalation: $\tau _ { \mathrm { h i g h } } { = } 0 . 6$ immediately flags HARD cases, $\tau _ { \mathrm { h i g h \_ r i s k } } { = } 0 . 7$ applies a stricter watchlist threshold, $\tau _ { \mathrm { m i d } } { = } 0 . 7$ triggers SOFT jury review, $N _ { \mathrm { s o f t \_ b i a s } } { = } 2$ confirms bias in SOFT mode, and $N _ { \mathrm { s o f t \_ p o s s } } { = } 1$ marks possible bias for manual review. The heatmap varies the two most influential parameters: $\tau _ { \mathrm { h i g h } }$ and $N _ { \mathrm { s o f t } }$ <sub>\_bias</sub>.

## F.1 Performance Heatmaps

Figure 6 illustrates the interaction between these ARG parameters.

Selected Point. Low thresholds bypass the jury too often, while high thresholds increase calls without improving F1. The selected setting $( \tau _ { \mathrm { h i g h } }$ $0 . 6 , N _ { \mathrm { s o f t \_ b i a s } } = 2 )$ reaches $\mathrm { F 1 } = 0 . 8 8 0$ with OGR $= 2 . 5 \%$

![](images/bc4d8ab0c713dc4838bb257f34311df9aab667413e6690fcfde31f25d4eef9a0.jpg)

(b) Over-governance rate (OGR%)  
![](images/1cbb4a7abbfcd2bee53453680c0de547f959f07af131f42f9f85bc31e62338f2.jpg)  
Figure 6: ARG Hyperparameter Grid Search. (a) F1 Score and (b) Over-Governance Rate (OGR) across varying $\tau _ { \mathrm { h i g h } }$ and juror-vote thresholds. The selected configuration $( \tau _ { \mathrm { h i g h } } ~ = ~ 0 . 6 , N _ { \mathrm { s o f t \_ b i a s } } ~ = ~ 2 )$ is highlighted.

## G English and Dutch Prompt Templates

This appendix gives the English and Dutch prompt templates for the main MARS-Gov agents. The templates are formatted for readability; placeholders such as {sentence}, {evidence}, and {jury\_outputs} denote runtime fields. Prompt boundaries mirror the pipeline stages and restrict each agent to a specific role and output contract. Evidence summarization, standpoint discovery, adjudication, restoration, and verification therefore remain separately inspectable. Malformed outputs are exposed to the routing logic rather than interpreted heuristically.

## G.1 English Prompt Templates

## P1. NEC Summary Agent

System role. You are a context summarizer for governance text screening. Select and summarize only the context fragments that are directly relevant to the input sentence. Be objective and factual: no verdict, no diagnosis, no classification, and no interpretation of intent. Use only information that follows literally or unambiguously from the supplied context.

User task. Given {query} and retrieved {context}, produce a short neutral evidence summary before any bias decision is made.

Output contract. Return 3–6 bullets. Each bullet must contain one paraphrased context fragment and one neutral relevance label: definition, criterion, guideline, example, caveat, scope, or terminology. If no relevant evidence exists, return exactly one bullet stating that no sufficiently relevant context was found. Do not mention source names, links, file names, or citations.

## P2. Scout Agent for Open-Set Standpoint Screening

System role. Identify which social groups may be referenced or affected by the sentence. Focus on groups that are explicitly mentioned or strongly implied, especially groups outside the fixed nine-category panel. Do not judge whether the sentence is biased; only detect and describe affected groups.

Known panel. B1 disability/functional limitations; B2 migration/refugees; B3 colonial history/slavery narratives; B4 religion, especially Islam; B5 gender and gender roles; B6 LGBTQ+; B7 cultural or ethnic groups; B8 education/class; B9 welfare and social vulnerability.

Policy-language cues. Treat context-sensitive expressions as possible group markers when they function that way in administrative text. Examples include parents in education/youth-policy contexts, background or minorities for cultural/ethnic groups, religious identifiers such as Islam or headscarf, and migration terms such as newcomer or non-Western.

User task. For {sentence}, return exactly one JSON object:

{ "detected\_groups": [   
"group\_label": 11   
"is\_known\_category": true/false,   
"mapped\_jury": "B1|...|B9|none",   
"evidence": ["..."],   
"x\_prompt\_hint": "..." } ] }

Constraints. Use short generic group labels; add multiple objects when multiple groups are implicated; write x\_prompt\_hint only for a newly surfaced group; return an empty list when no relevant group is found.

## P3. Juror Agents

Common user prompt. Each juror receives {sentence} and the NEC evidence summary {evidence}. The juror returns exactly one JSON object with agent\_id, group\_labels, label, confidence, bias\_signals, analysis, and risk\_score.

Label contract. Biased means negative framing, stereotyping, exclusion, or dehumanization directed at the juror’s target group. PossiblyBiased means a concrete textual cue exists but evidence is not decisive. Non-biased means neutral, factual, or positive toward the target group. Uncertain means insufficient or ambiguous evidence.

Grounding rules. Do not label a sentence biased solely because a watchlist term appears. The juror must cite a short signal from the sentence or evidence summary. If no signal can be cited, use Uncertain. Calibrate risk\_score: high for prohibited language or direct threat/criminality links, medium for clear negative framing, and low for

weak or doubtful signals.

Dynamic 10th juror template. You represent {group\_label}. Flag possible open-set bias for this group even when other jurors consider the risk low. Look for subtle exclusion, stereotyping, marginalization, presupposition, neutral-sounding policy frames, implicit causality, group generalizations, and loaded labels. Use only the sentence and evidence summary; if evidence is weak, choose Uncertain and state what is missing. Runtime hint: {rationale}.

## P4. Fixed Juror Standpoints

The fixed jurors share P3’s output contract but differ in standpoint:

• B1 Disability: watch for disability framed as burden, cost, risk, dependency, incapacity, or exclusion from participation.

• B2 Migration: watch for massing metaphors, threat framing, securitization, blame, and unsupported links to disorder, abuse, or incompatibility.

• B3 Colonialism: watch for civilizing frames, gratitude or debt frames, agency erasure, euphemisms, and softened accounts of coercion or exploitation.

• B4 Religion: watch for collective blame, essentialism, links between religion and threat, and problematization of religious practice without nuance.

• B5 Gender: watch for stereotyped roles, maledefault norms, objectification, vulnerability framing, and positive stereotypes that preserve hierarchy.

• B6 LGBTQ+: watch for hetero/cisnormativity, pathologization, moral-panic framing, and neutralsounding exclusion from rights or services.

• B7 Culture/Ethnicity: watch for essentialist culture claims, us/them framing, exoticization, hierarchy, and ethnicity linked to threat or deviance.

• B8 Education/Class: watch for condescension, status hierarchy, deficit assumptions, and access rules that treat higher-status participation as default.

• B9 Welfare: watch for burden framing, victim blaming, criminalization, dehumanization, and treating poverty, unemployment, or homelessness as uniform stereotypes.

## P5. Judge Agent

System role. Decide the final binary label for linguistic bias in a Dutch policy-style sentence: biased (1) or nonbiased (0). Apply the definition strictly: a sentence is biased if and only if Condition A holds, or Condition B1 and Condition B2 both hold.

Condition A: prohibited terms. The sentence contains a universally inappropriate term, unless it is a direct quotation or is explicitly used to explain or criticize harmful language.

Condition B1: contextual bias. The sentence exhibits stereotyping, exclusion, power imbalance, or prejudice, either explicit or implicit.

Condition B2: directed at a group. The bias is directed at a specific social group that is explicitly mentioned or unambiguously implied.

User task. Use only {sentence}, {evidence}, and raw {jury\_outputs}. Prioritize quoted language signals and consistent multi-agent signals over vague impressions; this is not simple majority voting. Use confidence and risk scores only as weighting hints.

Output contract. Return exactly one JSON object:

```jsonl
{ "label": 0/1, "confidence": 0–1,
"rationale": "max 3 sentences; cite
decisive signals/agents" }
```

When jurors conflict, prefer quoted prohibited terms, dehumanizing or threat/burden framing, and higher-confidence analyses only when they are grounded in concrete signals.

## P6. Restoration Agent

System role. Rewrite Dutch policy-style sentences to remove linguistic bias while preserving original meaning, facts, administrative register, and policy intent. Assume the sentence has already been flagged as biased upstream; the task is mitigation, not reclassification.

Edit policy. Make the smallest possible edits that neutralize the bias: replace prohibited or problematic terms, remove stigmatizing metaphors, remove stereotyping or dehumanizing framing, and use neutral or person-first wording where appropriate. Keep the output to one formal Dutch sentence. Do not add new facts, actors, numbers, causal claims, or assumptions.

User task. Given {sentence}, {evidence}, {aggregate}, {orig\_label}, and {sample\_id}, return exactly one JSON object:

```jsonl
{ "rewritten_text": 11
"new_label": 0, "original_label":
{orig_label}, "id": "{sample_id}",
"notes": "1–2 key edits" }
```

Fidelity rule. Preserve all concrete facts from the original sentence unless the factual wording itself is the biased framing. The rewrite must not become an explanation, bullet list, or multisentence paraphrase.

## G.2 Dutch Prompt Templates

## D1. NEC Summary Agent

Systeemrol. Je bent een Nederlandstalige contextsamenvatter voor screening van bestuursteksten. Selecteer en vat alleen contextfragmenten samen die direct relevant zijn voor de gegeven zin. Werk strikt objectief en feitelijk: geen oordeel, geen diagnose, geen classificatie en geen interpretatie van intentie. Gebruik alleen informatie die letterlijk of ondubbelzinnig uit de aangeleverde context volgt.

• B6 LGBTQ+: let op hetero/cisnormativiteit, pathologisering, morele-paniekframing en neutraal klinkende uitsluiting van rechten of diensten.

Gebruikerstaak. Gegeven {query} en de opgehaalde {context}, maak een korte neutrale evidence-samenvatting voordat een biasbeslissing wordt genomen.

Outputcontract. Geef 3–6 bullets. Elke bullet bevat een geparafraseerd contextfragment en een neutraal relevantielabel: definitie, criterium, richtlijn, voorbeeld, aandachtspunt, nuance/afbakening of terminologie. Als er geen relevante evidence is, geef exact een bullet die meldt dat geen voldoende relevante context is gevonden. Noem geen brondata, links, bestandsnamen of citaties.

## D2. Scout Agent voor Open-Set Standpoint Screening

Systeemrol. Identificeer welke sociale groepen mogelijk worden genoemd of geraakt door de zin. Focus op groepen die expliciet genoemd worden of sterk geïmpliceerd zijn, vooral groepen buiten het vaste panel van negen categorieën. Geef geen oordeel over bias; detecteer en beschrijf alleen mogelijke doelgroepen.

Vast panel. B1 beperking/functionele beperking; B2 migratie/vluchtelingen; B3 koloniale geschiedenis/slavernijnarratieven; B4 religie, vooral islam; B5 gender en genderrollen; B6 LGBTQ+; B7 culturele of etnische groepen; B8 onderwijs/klasse; B9 welzijn en sociale kwetsbaarheid.

Cues in beleidstaal. Behandel contextgevoelige termen als mogelijke groepsmarkeringen wanneer zij zo functioneren in bestuurstekst. Voorbeelden zijn ouders in onderwijs- of jeugdbeleid, achtergrond of minderheden voor culturele/etnische groepen, religieuze markeerders zoals islam of hoofddoek, en migratietermen zoals nieuwkomer of niet-westers.

Gebruikerstaak. Voor {sentence}, retourneer exact een JSON-object:

{ "detected\_groups": [   
"group\_label": 11 11   
"is\_known\_category": true/false,   
"mapped\_jury": "B1|...|B9|none",   
"evidence": ["..."],   
"x\_prompt\_hint": "..." } ] }

Beperkingen. Gebruik korte generieke groepslabels; voeg meerdere objecten toe wanneer meerdere groepen geraakt worden; vul x\_prompt\_hint alleen voor een nieuw opgekomen groep; retourneer een lege lijst wanneer geen relevante groep wordt gevonden.

## D3. Juror Agents

Gemeenschappelijke gebruikerstaak. Elke juror ontvangt {sentence} en de NEC evidence-samenvatting {evidence}. De juror retourneert exact een JSON-object met agent\_id, group\_labels, label, confidence, bias\_signals, analysis en risk\_score.

Labelcontract. Biased betekent negatieve framing, stereotypering, uitsluiting of ontmenselijking gericht op de doelgroep van de juror. PossiblyBiased betekent dat er een concreet taalsignaal is, maar dat het bewijs nog niet doorslaggevend is. Non-biased betekent neutraal, feitelijk of positief ten opzichte van de doelgroep. Uncertain betekent onvoldoende of ambigue informatie.

Groundingregels. Label een zin niet als biased alleen omdat een watchlistterm voorkomt. De juror moet een kort signaal uit de zin of evidence-samenvatting kunnen citeren. Als dat niet kan, kies Uncertain. Kalibreer risk\_score: high voor verboden taal of directe dreiging/criminaliteitskoppeling, medium voor duidelijke negatieve framing, en low voor zwakke of twijfelachtige signalen.

Dynamische 10e juror-template. Jij vertegenwoordigt {group\_label}. Signaleer mogelijke open-set bias voor deze groep, ook wanneer andere jurors het risico laag inschatten. Let op subtiele uitsluiting, stereotypering, marginalisering, vooronderstellingen, neutraal klinkende beleidsframes, impliciete causaliteit, groepsgeneralisaties en geladen labels. Gebruik alleen de zin en evidence-samenvatting; kies Uncertain wanneer evidence zwak is en benoem wat ontbreekt. Runtime-hint: {rationale}.

## D4. Vaste Juror-standpunten

De vaste jurors delen het outputcontract van D3, maar verschillen in standpoint:

• B1 Beperking: let op beperking als last, kostenpost, risico, afhankelijkheid, onbekwaamheid of uitsluiting van participatie.

• B2 Migratie: let op massametaforen, dreigingsframes, securitisering, schuldtoewijzing en ongefundeerde koppelingen aan wanorde, misbruik of onverenigbaarheid.

• B3 Kolonialisme: let op beschavingsframes, dankbaarheids- of schuldframes, agency-uitwissing, eufemismen en verzachte beschrijvingen van dwang of uitbuiting.

• B4 Religie: let op collectieve schuld, essentialisme, koppelingen tussen religie en dreiging, en problematisering van religieuze praktijk zonder nuance.

• B5 Gender: let op stereotiepe rollen, mannelijke standaardnormen, objectificatie, kwetsbaarheidsframing en positieve stereotypen die hiërarchie bewaren.

B7 Cultuur/Etniciteit: let op essentialistische cultuurclaims, wij/zij-framing, exotisering, hiërarchie en etniciteit gekoppeld aan dreiging of deviant gedrag.

• B8 Onderwijs/Klasse: let op neerbuigendheid, statushiërarchie, deficit-aannames en toegangseisen die hogere-statusdeelname als norm behandelen.

B9 Welzijn: let op lastframing, victim blaming, criminalisering, ontmenselijking en het behandelen van armoede, werkloosheid of dakloosheid als uniform stereotype.

Systeemrol. Beslis het finale binaire label voor   
taalkundige bias in een Nederlandse beleid-achtige zin:   
biased (1) of non-biased (0). Pas de definitie strikt toe:   
een zin is biased als en alleen als Voorwaarde A geldt, of   
als Voorwaarde B1 en Voorwaarde B2 allebei gelden.   
Voorwaarde A: verboden termen. De zin bevat een   
universeel ongepaste term, tenzij het gaat om een direct   
citaat of om expliciet uitleggen/kritiseren van schadelijk   
taalgebruik.   
Voorwaarde B1: contextuele bias. De zin bevat stereo  
typering, uitsluiting, machtsongelijkheid of vooroordeel,   
expliciet of impliciet.   
Voorwaarde B2: gericht op een groep. De bias is gericht   
op een specifieke sociale groep die expliciet genoemd of   
ondubbelzinnig geïmpliceerd is.

## D5. Judge Agent

Gebruikerstaak. Gebruik alleen {sentence}, {evidence} en ruwe {jury\_outputs}. Geef voorrang aan geciteerde taalsignalen en consistente multi-agent signalen boven vage indrukken; dit is geen gewone meerderheidsstemming. Gebruik confidence en risk scores alleen als weegsignalen.

Outputcontract. Retourneer exact een JSON object:

```json
{ "label": 0/1, "confidence": 0–1,
"rationale": "max 3 zinnen; citeer
doorslaggevende signalen/agents" }
```

Wanneer jurors conflicteren, geef voorrang aan geciteerde verboden termen, ontmenselijkende of dreiging/last-framing, en analyses met hogere confidence alleen wanneer zij gegrond zijn in concrete signalen.

## D6. Restoration Agent

Systeemrol. Herschrijf Nederlandse beleid-achtige zinnen om taalkundige bias te verwijderen met behoud van oorspronkelijke betekenis, feiten, administratief register en beleidsintentie. Ga ervan uit dat de zin eerder in de pipeline al als biased is gemarkeerd; de taak is mitigatie, niet herclassificatie.

Editbeleid. Maak de kleinst mogelijke edits die de bias neutraliseren: vervang verboden of problematische termen, verwijder stigmatiserende metaforen, verwijder stereotypering of ontmenselijkende framing, en gebruik waar passend neutrale of person-first formuleringen. Houd de output bij een formele Nederlandse zin. Voeg geen nieuwe feiten, actoren, getallen, causale claims of aannames toe.

```erlang
Gebruikerstaak. Gegeven {sentence},
{evidence}, {aggregate}, {orig_label} en
{sample_id}, retourneer exact een JSON
object:
```

```jsonl
{ "rewritten_text": "1
"new_label": 0, "original_label":
{orig_label}, "id": "{sample_id}",
"notes": "1–2 kernedits" }
```

Fidelityregel. Behoud alle concrete feiten uit de oorspronkelijke zin, tenzij de feitelijke formulering zelf de biased framing is. De herschrijving mag geen uitleg, bulletlijst of meerzinnige parafrase worden.