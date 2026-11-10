# TrialDesignBench: Benchmarking Language Models for Phase-I Trial Design Against Clinical Knowledge

Yihan Hank Tang<sup>1∗</sup> Peng Yang<sup>2∗</sup> Jiayi Xin<sup>1∗</sup> Tianqi Shang<sup>1</sup> Xuyang Chen<sup>1</sup> Ying Yuan<sup>2</sup> Yi Lian<sup>1†</sup> Qi Long<sup>1†</sup>

<sup>1</sup>University of Pennsylvania <sup>2</sup>The University of Texas MD Anderson Cancer Center

## Abstract

AI is used in much of the clinical trial pipeline. However, selecting the statistical design of a trial, a crucial step that determines how drug doses are tested on humans and evaluated, has seen almost no AI progress. LLMs appear to be a natural fit given their strengths in biomedical knowledge and statistical reasoning. Nevertheless, there is currently no appropriate way to evaluate such abilities because recent regulatory and methodological shifts, most notably FDA’s Project Optimus, have redefined what constitutes a reasonable design and rendered many previous designs obsolete. To lay the groundwork for AI-assisted trial design, we introduce TRIALDESIGNBENCH, a benchmark of 617 Phase-I/I-II oncology dose-finding trials, each paired with an expert-validated reference design that is consistent with the latest FDA regulatory initiatives. This reference standard is produced through a formalization of the expert design process. We evaluate five frontier LLMs, none of which matches the reference standard. Our ablation experiments trace this performance gap to two main causes, inference of latent, unrecorded design determinants from trial sources (supplying them raises class-wise balanced accuracy by 0.22) and structured decision-tree reasoning (+0.24). TRIALDESIGNBENCH offers a reproducible, contemporary-standard objective for evaluating and enhancing AI-assisted trial design.

Data and Code Availability This study uses the public Aggregate Analysis of ClinicalTrials.gov (AACT) database [9]. Code and data are available at https://github.com/hanktang-researc h/TrialDesignBench.

## 1 Introduction

Clinical trials are the gold standard for validating new therapeutics [4]. However, clinical trials are costly and often fail. It is estimated that the entire development process from discovery to bringing a drug to market costs over \$2.5 billion [2]. There are generally three phases in clinical trials [4]. More than 32% of trials fail at Phase I alone [13]. Statistical design and planning are critical components of a clinical trial, determining how doses are selected and how safety and efficacy are evaluated throughout a study [4]. The chosen statistical design can substantially affect the accuracy of these assessments and the number of patients required to reach actionable conclusions, ultimately influencing development timelines, costs, and likelihood of success. In this paper, we focus on the design method selection for phase I oncology trials, where the primary objective is to identify an appropriate dose for subsequent studies while minimizing patient exposure to subtherapeutic or overly toxic doses [16].

Although AI is applied across different parts of the clinical trial pipeline, it has made little progress in the statistical design components of clinical trials, particularly the statistical method selection. Existing AI applications in clinical trial pipelines include patient-to-trial matching [6] and trial outcome prediction [5]. The development of these tools benefits from readily available training and benchmark data derived from historical trials, where documentations of past decisions, outcomes, or annotations can often serve as supervision signals. These signals are largely invariant to changes in regulatory guidelines.

By contrast, statistical design decisions cannot rely on historical trial designs as ground truth. Trial design selection is influenced by available statistical methods and more importantly, regulatory guidelines [4]. The design adopted in a historical study is not necessarily the design that would be recommended today. Statistical methodologies have been evolving from the traditional 3+3 design [8], to Continual Reassessment Method [7], and to the Bayesian Optimal Interval Design [14].

Project Optimus [10], a landmark clinical trial design initiative published by the FDA, fundamentally breaks the traditional design paradigm. Since it is issued by the official regulatory agency, Project Optimus sets the standard for how current oncology trials are designed. It moves early-phase trials away from pursuing the maximum tolerated dose (MTD) toward the optimal biological dose (OBD), which is to focus on both safety and efficacy rather than safety alone [3]. This objective defines what a reasonable trial design is. A design method well suited to identifying an MTD is not necessarily appropriate for identifying an OBD. The methodological standard has evolved alongside the objective. Regulatory recognition of contemporary Bayesian adaptive dose-finding designs was granted more recently, with BOIN receiving FDA Fit-for-Purpose designation in 2021 [11] and Bayesian design methods gaining broader recognition in the FDA’s 2026 guidance [12]. Bayesian adaptive trial designs take advantage of the Bayes’ theorem and incoming clinical data to continuously update probabilities of treatment outcomes and alter the specifics of trial implementation [12]. Since both the design objective and the accepted methodology have changed fundamentally, the overwhelming majority of recorded trials can no longer serve as ground truth or training data for models expected to meet the latest clinical trial guidelines.

We introduce TRIALDESIGNBENCH, the first benchmark for AI-assisted statistical design recommendation in clinical trials. Our contributions are as follows:

1. We curate a benchmark of 617 early-phase oncology dose-finding trials, each annotated with rich clinical information fields and paired with an expert-validated reference-standard design over 11 trial design method classes. The reference standard is produced through a formalization of the expert design process and reflects Project Optimus guidelines.

2. We evaluate five frontier LLMs under tiered input information configurations that isolate the effects of clinical information availability and expert-derived design principles. No model matches the reference standard, demonstrating a substantial performance gap.

3. Through ablations, we identify two consistent bottlenecks in LLM-based design recommendation, recovering unrecorded, latent design information from source text (+0.22 class-wise balanced accuracy when supplied) and structured decision-tree reasoning (+0.24).

Together, TRIALDESIGNBENCH establishes a reproducible, modern-standard foundation for robust evaluation and future improvements in AI systems for statistical trial design.

## 2 Benchmark Construction

## 2.1 Data Collection

We draw our sampling frame from the AACT database [9], comprising approximately 70,810 interventional trial records (fetched on 28 May 2026). We restrict the selection to Phase I and combined Phase I & II dose-finding interventional oncology trials with a linked publication on PubMed and a posted protocol and, where available, a statistical analysis plan (SAP). The registry reports information such as trial phase, conditions, and interventions. Protocols and SAPs contain more crucial information, such as the dose-limiting toxicity (DLT) definition, the DLT assessment window, and the endpoints used to select the dose, for biostatisticians to perform trial design recommendation (Section 4.1). As illustrated in Figure 1, after applying these filters, we derive 617 trials.

![](images/5ca1d6d31e1d0103a4e8469d3058a626e415cf4ccc0371d9049710c2c92079fe.jpg)  
Figure 1: Corpus selection process.

## 2.2 Benchmark Inputs

For each trial, we curate three levels of trial attributes that reflect progressively richer information available during statistical design selection, enabling controlled evaluation of LLMs’ ability to recommend design methods by varying the amount of trial knowledge provided as input.

Basic Trial Context High-level trial information, including condition, intervention, phase, and objectives. These are typically communicated to biostatisticians during initial discussions with physician trialists. For benchmark curation, Basic Trial Context is extracted from ClinicalTrials.gov records.

Trial Attributes More fine-grained trial information relevant to operational and planning details, including endpoint definition, treatment schedules, and other design-related characteristics enumerated in Appendix C. These details are typically documented during the very early phases of protocol development and before final decision on any statistical design selection. For benchmark curation, Trial Attributes are extracted from protocols and SAPs.

Design Factors Design-relevant attributes that are not explicitly described in protocols and SAPs but can be inferred. Examples include treatment regimen type (single agent vs. combination), inferred accrual rate, primary endpoint type (acute vs. delayed) [1], and others (Appendix C). These factors approximate information that trialists and biostatisticians often derive from domain knowledge when evaluating candidate statistical designs. For benchmark curation, such factors were inferred and validated based on basic trial context and trial attributes.

## 2.3 Reference-standard Label

The reference-standard recommendation for each trial is produced based on contemporary phase I dose-finding design principles developed and adopted by experienced trial methodologists and biostatisticians. Figure 2 summarizes this workflow as a decision tree adapted from modern dosefinding methodology [15, 16]. The tree sequentially evaluates key Basic Trial Context, Trial Attributes, and Design Factors, a subset of the benchmark inputs described in Section 2.2, ultimately leading to a recommended design method.

The benchmark output is a single reference-standard design recommendation. Each recommendation corresponds to a specific dose-finding method, such as BOIN, 3+3, or CRM. Although a single method is assigned to each trial, multiple methods may represent reasonable choices to the same dose-finding problem. Methods grouped in the same leaf node of the decision workflow (Figure 2) are grouped into a class, reflecting shared design objectives and methodological characteristics.

The resulting reference-standard recommendations were subsequently reviewed and validated by experienced trial methodologists and biostatisticians to ensure consistency with contemporary methodological standards rather than historical practice. The node-resolution rules, in particular the late-onset determination (Appendix E.2), were calibrated against case sets labeled by these experts, and the experts re-verified the single-agent-versus-combination node for every trial initially labeled with a combination design, which reassigned 31 of 51 such trials to single-agent designs. The complete decision rules are provided in Appendix E, and the label distribution is reported in Appendix B.

![](images/d6224bfc2552cef11ca67316f6de85af3dc6d0d66ee139c334d9a21e35d30c4f.jpg)  
Figure 2: Decision workflow for Phase I dose-finding interventional oncology trial design (compact overview). A full-size, legible version, split by primary endpoint into the MTD and OBD branches, is provided in Figure 3 (Appendix E).

## 3 Experiments

We evaluate frontier LLMs on TRIALDESIGNBENCH to establish baseline performance for AIassisted statistical trial design recommendation. In addition, we investigate how recommendation quality changes as progressively richer trial information becomes available and whether expert-guided decision-tree reasoning improves performance. Our experiments aim to (1) quantify the current capabilities of LLMs on statistical trial design recommendation, (2) assess whether LLM performance improves with increasingly detailed trial information and access to expert-derived design principles encoded in Figure 2, and (3) identify common failure modes in model reasoning.

## 3.1 Evaluation Metric

Each model recommendation and each reference-standard label correspond to a specific design method. Since multiple design methods may be admissible for the same trial, we intentionally relax evaluation from exact reference-standard matching to class-level agreement. Let $c ( \cdot )$ denote the mapping from a design method to its class. A recommendation m is considered correct whenever $c ( m ) = c ( m ^ { \star } )$ , where $m ^ { \star }$ is the reference-standard recommendation. For $N$ evaluated trials with recommendations $m _ { i }$ and reference-standard recommendations $m _ { i } ^ { \star }$ , class-wise accuracy is then

$$
\operatorname { A c c } _ { c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left[ c ( m _ { i } ) = c ( m _ { i } ^ { \star } ) \right] .\tag{1}
$$

We report it together with class-wise balanced accuracy, macro-F1, and Cohen’s κ (Section 4) throughout. The complete class taxonomy is provided in Appendix A.

## 3.2 Model Selection

To comprehensively evaluate the performance of popular LLMs on trial design recommendation, we select Claude Opus 5, GPT-5, OpenAI o3, OpenAI gpt-oss-120b, and Llama 4 Maverick.

## 3.3 Input Configurations

To understand the information and reasoning requirements of statistical trial design recommendation, we construct a series of input configurations that progressively enrich the trial and design information available to the model. The configurations form a tiered information hierarchy crossed with an expert-derived design workflow:

Table 1: Class-wise metrics averaged over five models, by input configuration. Input configurations form an information ladder (A→B→C) crossed with a reasoning scaffold $( \mathrm { B \to D , C \to E } )$
<table><tr><td>Input Configuration</td><td> $\operatorname { A c c } _ { c _ { c } }$ </td><td> $\mathbf { B a l A c c } _ { c }$ </td><td> $\mathrm { F } 1 _ { c }$ </td><td> $\kappa _ { c }$ </td></tr><tr><td>A (basic trial context)</td><td>0.436</td><td>0.225</td><td>0.099</td><td>0.059</td></tr><tr><td>B (A + attributes)</td><td>0.614</td><td>0.307</td><td>0.135</td><td>0.131</td></tr><tr><td>C (B + design factors)</td><td>0.806</td><td>0.531</td><td>0.293</td><td>0.406</td></tr><tr><td>D (B + decision workflow)</td><td>0.715</td><td>0.547</td><td>0.298</td><td>0.272</td></tr><tr><td>E (C + decision workflow)</td><td>0.868</td><td>0.728</td><td>0.506</td><td>0.542</td></tr></table>

A. Basic Trial Context. Basic Trial Context only.

B. A + Trial Attributes. Basic Trial Context and Trial Attributes.

C. B + Design Factors. Basic Trial Context, Trial Attributes, and Design Factors.

D. B + Workflow. Basic Trial Context, Trial Attributes, and the decision workflow (Figure 2).

E. C + Workflow. Basic Trial Context, Trial Attributes, Design Factors, and the decision workflow.

Configurations A→B→C evaluate whether recommendation performance improves as progressively richer trial information, particularly design factors that required judgment by experienced trialists, becomes available. Configurations D and E additionally provide the design workflow, allowing the contributions of trial information and design principles to be examined separately. Because configuration E supplies both the design factors and the decision workflow from which the reference labels are derived, it can be viewed as an oracle-workflow condition.

## 3.4 Implementation Details

All five models were accessed through a single unified serving layer with pinned versions and identical prompting and decoding settings, so that differences across models reflect capability. For each of the 617 trials under input configurations A, B, and C, every model returns a direct design class recommendation. For input configurations D and E, every model returns a structured JSON response, per-node binary answers with rationales, which a deterministic, code-based tree walker maps to a single leaf design class. Predictions are then scored against the expert validated labels using the class-level accuracy defined in Section 3.1. For configuration E, the design-factor input at the combination-target node was corrected after review to match the field used by the reference rules, and the 20 affected combination trials were re-run. All E results reflect this correction.

## 4 Results

We report four class-wise metrics, accuracy $( \operatorname { A c c } _ { c } ) .$ , balanced accuracy $( { \mathrm { B a l A c c } } _ { c } )$ , macro-F1 $( \mathrm { F } 1 _ { c } )$ and Cohen’s $\kappa \left( \kappa _ { c } \right)$ , each computed after mapping the recommended and validated methods to their design-method classes (Section 3.1). Balanced accuracy is the mean per-class recall, macro-F1 the unweighted mean of per-class F1, and κ Cohen’s chance-corrected agreement. The subscript c marks computation over design-method classes rather than raw methods. Table 1 summarizes performance averaged over the five models. Table 2 gives the per-model balanced accuracy underlying the two findings below.

## 4.1 Model Performance Degrades Without Inferred Design Factors

Supplying the inferred design factors (input configuration C vs. B) improves every class-wise metric for all five models. Averaged over models, class-wise accuracy rises by +0.192 (0.614 → 0.806), balanced accuracy by +0.224 (0.307 → 0.531), macro-F1 by $+ 0 . 1 5 8 \ \mathrm { ( 0 . 1 3 5  0 . 2 9 3 ) }$ , and κ by $+ 0 . 2 7 5 \ : ( 0 . 1 3 1  0 . 4 0 6 )$ . The same performance gap exists under the reasoning scaffold (E vs. D, +0.153 class-wise accuracy). Therefore, the bottleneck in frontier LLM recommendation appears to be the translation of trial information into latent design factors that influence design selection.

Table 2: Class-wise balanced accuracy per model. Both effects are 5/5 consistent (C>B and D>B on all four metrics; Appendix D).
<table><tr><td>Model</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td></tr><tr><td>Claude Opus 5</td><td>0.152</td><td>0.395</td><td>0.558</td><td>0.499</td><td>0.764</td></tr><tr><td>GPT-5</td><td>0.259</td><td>0.327</td><td>0.662</td><td>0.570</td><td>0.787</td></tr><tr><td>03</td><td>0.219</td><td>0.314</td><td>0.408</td><td>0.552</td><td>0.761</td></tr><tr><td>gpt-oss-120b</td><td>0.276</td><td>0.254</td><td>0.508</td><td>0.450</td><td>0.480</td></tr><tr><td>Llama 4 Maverick</td><td>0.219</td><td>0.246</td><td>0.522</td><td>0.663</td><td>0.850</td></tr><tr><td>Average</td><td>0.225</td><td>0.307</td><td>0.531</td><td>0.547</td><td>0.728</td></tr></table>

For example, when a trial combines an investigational agent with a standard-of-care therapy of known dose and toxicity profile, models may incorrectly formulate the problem as a two-dimensional dose-finding study and recommend a combination design.

A node-level breakdown of the tree walks (Appendix D, Table 5) localizes where this inference fails. Deviations from the reference concentrate at the late-onset node, the accrual-to-window latent, which alone accounts for more incorrect node answers than all other nodes combined, followed by the singleversus-combination node. The single-versus-combination node shows a deceptively low deviation rate. This is because the dataset is dominated by the single-agent class trials. True-Combination recall at this node ranges from 20% to 45% across models.

## 4.2 Structured Reasoning Recovers Minority-Class Performance

Requiring the model to traverse the expert-codified decision tree, answering each node with a binary decision and rationale rather than naming a design directly, improves performance. Adding the reasoning scaffold to the structured inputs (D vs. B) raises model-averaged balanced accuracy by +0.240 (0.307 → 0.547), macro-F1 by +0.163, κ by +0.141, and class-wise accuracy by +0.101. These four improvements are consistent across models. The benefit is concentrated in the imbalancesensitive metrics. On balanced accuracy, the scaffold alone (D, 0.547) matches the hand-resolvedlatent input configurations (C, 0.531), which means that structured reasoning is capable of making up for most of the value of the expert-supplied latents. When those latents are already given, the additional scaffold gain shrinks (E vs. C: +0.197 balanced accuracy, positive for 4/5 models). Structured reasoning, through serializing and visualizing the structured reasoning process, makes it straightforward to identify reasoning errors and implement fixes.

## 5 Limitations and Future Work

Our benchmark has several limitations. First, we recommend a single reference-standard trial design when multiple design methods are often admissible. We address this by grouping multiple design methods into the same class in the workflow and running experiments within defined design-method classes. Second, to construct the first benchmark dataset, we restrict the scope to a more homogeneous set of Phase I/I–II oncology dose-finding trials to ensure high-quality extraction of trial information and design method recommendations. For the same purpose, we select trials with a linked publication and a posted protocol so that trial information can be more accurately derived without significant loss of representativeness or generalizability for the purpose of benchmarking.

We will expand to a broader range of phase I–III trials, with or without protocol documents. A major challenge is that much design-relevant information is not explicitly reported in clinicaltrials.gov registry records. The current TRIALDESIGNBENCH may serve as few-shot examples for automated inference of latent design factors, facilitating scalable benchmark development.

## 6 Conclusion

We introduce TRIALDESIGNBENCH, a benchmark of 617 Phase I and Phase I/II oncology dosefinding trials paired with a rule-derived reference standard for the recommended statistical design. The workflow that produces the benchmark is based on contemporary dose-finding methodology rather than historically adopted practice. The benchmark reframes trial-design recommendation as an evaluable task. It enables us to measure a suite of contemporary LLMs against it under common class-wise criteria.

By modeling the trial design recommendation process as sequential and having models decide on nodes before traversing the decision tree, human experts are able to localize where recommendations break down. Models are limited considerably by inferring the latent, structurally unrecorded fields of trial design such as regimen modality and the late-onset accrual-to-window ratio. Without those resolved fields, models perform considerably worse on every metric. In addition, models that do not follow the tree-guided, expert-codified reasoning scaffold also show a considerable drop in performance. No model matches the rule-derived reference standard, leaving clear headroom.

Beyond ranking today’s systems, TRIALDESIGNBENCH provides a reusable, Project-Optimusaligned target for AI-assisted trial design and a diagnostic that separates extraction from higher-level inference. We see it as a step toward tools that help investigators move from basic trial information to a statistically sound design, and ultimately to trial protocol development.

## Responsible-Use Statement

TRIALDESIGNBENCH is an evaluation benchmark, not a clinical decision tool. Its reference-standard designs are a rule-based formalization of one specified expert workflow for contemporary Phase-I dose-finding methodology. Although it is expert-validated, it is not a unique clinical gold standard and not a substitute for a qualified biostatistician. No results here should inform a real trial without expert review. Every evaluated model in this paper fails to match the reference standard and can be confidently wrong. Therefore, unsupervised use risks propagating unsafe or inefficient designs. The corpus is built from public AACT/ClinicalTrials.gov and PubMed-linked protocols and may inherit their selection and documentation biases. The corpus largely consists of completed and well-documented oncology trials. We release processed data to facilitate future research on AI for clinical trial design. We recommend a domain expert in the loop with calibrated uncertainty for any downstream use.

## References

[1] Kai Chen, Ting-Yu Chen, Yiming Zhang, Ruitao Lin, and Ying Yuan. Practical considerations for using the TITE-BOIN design to handle late-onset toxicity or fast accrual in phase I trials. Clinical Cancer Research, 31(13):2573–2580, 2025.

[2] David B Fogel. Factors associated with clinical trials that fail and opportunities for improving the likelihood of success: a review. Contemporary clinical trials communications, 11:156–164, 2018.

[3] Jeanne Fourie Zirkelbach, Mirat Shah, Jonathon Vallejo, Joyce Cheng, Amal Ayyoub, Jiang Liu, Rachel Hudson, Rajeshwari Sridhara, Gwynn Ison, Laleh Amiri-Kordestani, et al. Improving dose-optimization processes used in oncology drug development to minimize toxicity and maximize benefit to patients. Journal ofClinical Oncology, 40(30):3489–3500, 2022.

[4] Lawrence M. Friedman, Curt D. Furberg, David L. DeMets, David M. Reboussin, and Christopher B. Granger. Fundamentals ofClinical Trials. Springer, Cham, 5th edition, 2015.

[5] Tianfan Fu, Kexin Huang, Cao Xiao, Lucas M. Glass, and Jimeng Sun. HINT: Hierarchical interaction network for clinical-trial-outcome predictions. Patterns, 3(4):100445, 2022. doi: 10.1016/j.patter.2022.100445.

[6] Qiao Jin, Zifeng Wang, Charalampos S Floudas, Fangyuan Chen, Changlin Gong, Dara Bracken-Clarke, Elisabetta Xue, Yifan Yang, Jimeng Sun, and Zhiyong Lu. Matching patients to clinical trials with large language models. Nature communications, 15(1):9074, 2024.

[7] John O’Quigley, Margaret Pepe, and Lloyd Fisher. Continual reassessment method: a practical design for phase 1 clinical trials in cancer. Biometrics, pages 33–48, 1990.

[8] Barry E Storer. Design and analysis of phase I clinical trials. Biometrics, pages 925–937, 1989.

[9] Asba Tasneem, Laura Aberle, Hari Ananth, Swati Chakraborty, Karen Chiswell, Brian J McCourt, and Ricardo Pietrobon. The database for aggregate analysis of clinicaltrials.gov (AACT) and subsequent regrouping by clinical specialty. PloS one, 7(3):e33677, 2012.

[10] U.S. Food and Drug Administration. Project Optimus: Reforming the dose optimization and dose selection paradigm in oncology. https://www.fda.gov/about-fda/oncology-c enter-excellence/project-optimus, 2024. Oncology Center of Excellence. Accessed September 10, 2026.

[11] U.S. Food and Drug Administration. Drug development tools: Fit-for-purpose initiative, 2025. URL https://www.fda.gov/drugs/development-approval-process-drugs/drug-d evelopment-tools-fit-purpose-initiative. Accessed September 10, 2026.

[12] U.S. Food and Drug Administration. Use of Bayesian methodology in clinical trials of drug and biological products, 2026. URL https://www.fda.gov/regulatory-information /search-fda-guidance-documents/use-bayesian-methodology-clinical-trial s-drug-and-biological-products. Accessed September 10, 2026.

[13] Chi Heem Wong, Kien Wei Siah, and Andrew W. Lo. Estimation of clinical trial success rates and related parameters. Biostatistics, 20(2):273–286, 2019.

[14] Ying Yuan, Kenneth R. Hess, Susan G. Hilsenbeck, and Mark R. Gilbert. Bayesian optimal interval design: A simple and well-performing design for phase I oncology trials. Clinical Cancer Research, 22(17):4291–4301, 2016.

[15] Ying Yuan, Ruitao Lin, and J. Jack Lee. Model-Assisted Bayesian Designs for Dose Finding and Optimization: Methods and Applications. CRC Press, 2022.

[16] Ying Yuan, Heng Zhou, and Suyu Liu. Statistical and practical considerations in planning and conduct of dose-optimization trials. Clinical Trials, 21(3):273–286, 2024.

## Appendix

## A Design-Method Class Taxonomy

Recommendations and reference-standard labels are grouped into eleven classes, the leaves of the decision tree (Figure 3). Class-wise credit $( c ( m ) = c ( m ^ { \star } )$ , Section 3.1) is given within Classes 1–10; Class 11 is a residual catch-all of off-tree designs and receives credit only on an exact-method match.

1. Standard single-agent MTD: BOIN, Keyboard, 3+3, CRM, BMA-CRM, EWOC, BLRM.

2. Time-to-event single-agent MTD: TITE-BOIN, TITE-Keyboard, TITE-CRM.

3. Multiple / Continuous toxicity: MT-BOIN, gBOIN, MC-CRM.

4. Informative-prior: iBOIN, iKeyboard, iCRM.

5. Combination single-MTD: BOIN Comb, Keyboard Comb.

6. Combination MTD-contour: Waterfall.

7. OBD time-to-event: TITE-BOIN12.

8. OBD without utility: BOIN-ET, STEIN.

9. OBD utility-based: BOIN12, EffTox.

10. OBD with MTD as secondary: U-BOIN.

11. Other (off-tree): TPI, mTPI, mTPI-2, TEQR, Rolling-6, Storer two-stage, accelerated titration.

## B Reference-Standard Label Distribution

Table 3 reports the design-method label distribution of the reference standard across the 617 trials.

Table 3: Reference-standard label distribution (617 trials).
<table><tr><td>Label</td><td>Count</td><td>%</td></tr><tr><td>BOIN</td><td>366</td><td>59.3</td></tr><tr><td>Out of scope</td><td>142</td><td>23.0</td></tr><tr><td>3+3</td><td>59</td><td>9.6</td></tr><tr><td>TITE-BOIN</td><td>24</td><td>3.9</td></tr><tr><td>BOIN Comb</td><td>20</td><td>3.2</td></tr><tr><td>BOIN-ET</td><td>4</td><td>0.6</td></tr><tr><td>TITE-BOIN12</td><td>1</td><td>0.2</td></tr><tr><td>BOIN12</td><td>1</td><td>0.2</td></tr><tr><td>Total</td><td>617</td><td>100.0</td></tr></table>

Within the standard single-agent MTD class (Class 1, Appendix A), trials were initially labelled BOIN. However, a trial is relabeled 3+3 when it prespecifies at most two dose levels. Two things are worth noting here:

• The fact that 3+3 methods are recommended here is primarily grounded in operational simplicity.

• The prevalence of BOIN and 3+3 over other design methods reflects statistical soundness and operational consideration.

## C Input Configurations

Every input configuration identifies the trial by a case ID only. The originally adopted design, reported results, and design-revealing fields are withheld (Appendix G).

A (protocol narrative). The protocol’s background, rationale, and objectives excerpt only. No structured fields.

B (structured fields). Fourteen fields: brief title, phase, conditions, interventions, enrollment, start date, Phase I objective, endpoint nature, DLT (Dose-limiting Toxicity) definition, CTCAE (Common Terminology Criteria for Adverse Events) version, DLT assessment window, interpatient interval, interval basis, and safety-stagger floor, together with protocol excerpts in A.

C (structured + resolved latents). All of B, along with seven resolved latent fields, primary endpoint type, toxicity multiplicity, prior-information availability, regimen modality, combination target, OBD basis, MTD-as-secondary, and the resolved late-onset quantities (the accrual-to-window ratio W/I, the escalation gate, and the interval basis).

D, E. Identical field inputs to B and C respectively, presented under the tree-walk prompt (Appendix G).

## D Full Per-Model Results and Scoring Counts

Table 4 reports every model’s class-wise accuracy, macro-F1, and Cohen’s κ across the five input configurations (balanced accuracy is in Table 2). It substantiates the 5/5 consistency of C>B and D>B stated in the main text.

Table 4: Per-model class-wise metrics by input configuration (five models).
<table><tr><td>Model</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td></tr><tr><td colspan="6">Class-wise accuracy</td></tr><tr><td>Claude Opus 5</td><td>0.248</td><td>0.598</td><td>0.907</td><td>0.838</td><td>0.955</td></tr><tr><td>GPT-5</td><td>0.264</td><td>0.552</td><td>0.860</td><td>0.639</td><td>0.895</td></tr><tr><td>03</td><td>0.436</td><td>0.601</td><td>0.725</td><td>0.653</td><td>0.794</td></tr><tr><td>gpt-oss-120b</td><td>0.482</td><td>0.554</td><td>0.720</td><td>0.620</td><td>0.829</td></tr><tr><td>Llama 4 Maverick</td><td>0.749</td><td>0.767</td><td>0.818</td><td>0.823</td><td>0.864</td></tr><tr><td>Mean</td><td>0.436</td><td>0.614</td><td>0.806</td><td>0.715</td><td>0.868</td></tr><tr><td colspan="6">Class-wise macro-F1</td></tr><tr><td>Claude Opus 5 GPT-5</td><td>0.113</td><td>0.135</td><td>0.397</td><td>0.273</td><td>0.562</td></tr><tr><td>03</td><td>0.064 0.108</td><td>0.119 0.175</td><td>0.349 0.231</td><td>0.355 0.384</td><td>0.620</td></tr><tr><td>gpt-oss-120b</td><td>0.087</td><td>0.095</td><td>0.153</td><td>0.223</td><td>0.597 0.313</td></tr><tr><td>Llama 4 Maverick</td><td>0.121</td><td>0.150</td><td>0.332</td><td>0.252</td><td>0.440</td></tr><tr><td>Mean</td><td>0.099</td><td>0.135</td><td>0.293</td><td>0.298</td><td>0.506</td></tr><tr><td colspan="6">Class-wise κ</td></tr><tr><td>Claude Opus 5</td><td>0.040</td><td>0.140</td><td>0.586</td><td>0.382</td><td>0.778</td></tr><tr><td>GPT-5</td><td>0.045</td><td>0.141</td><td>0.518</td><td>0.204</td><td>0.598</td></tr><tr><td>03</td><td>0.037</td><td>0.138</td><td>0.294</td><td>0.193</td><td>0.435</td></tr><tr><td>gpt-oss-120b</td><td>0.083</td><td>0.097</td><td>0.270</td><td>0.188</td><td>0.392</td></tr><tr><td>Llama 4 Maverick</td><td>0.087</td><td>0.142</td><td>0.361</td><td>0.392</td><td>0.506</td></tr><tr><td>Mean</td><td>0.059</td><td>0.131</td><td>0.406</td><td>0.272</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.542</td></tr></table>

Table 5: Node-level deviation rate for the tree-walk configurations (D and E), averaged over the five models: the fraction of trials whose model node answer disagrees with the reference at each node.
<table><tr><td>Node</td><td>D</td><td>E</td><td>Pooled n</td></tr><tr><td>mtd_late_onset</td><td>0.262</td><td>0.118</td><td>3,919</td></tr><tr><td>mtd_comb_target</td><td>0.011</td><td>0.000</td><td>172</td></tr><tr><td>mtd_modality</td><td>0.029</td><td>0.027</td><td>4,627</td></tr></table>

## E Decision Rules

We walk the decision tree node by node. When a model resolves a node, it is deterministically routed to a deeper node in the tree. The leaf reached is mapped to a design-method class. Please refer to Figure 3.

![](images/4a467ea5e891379494c946a2326ca3ce2082d8f2c667f6aad6fb5b2152439069.jpg)  
(a) MTD branch (primary endpoint: toxicity).

![](images/96dc65e5af1c28a8988f0acfb2af35bd4408b0eadb9ed12d39e6c6ae2164a931.jpg)  
(b) OBD branch (primary endpoint: safety and efficacy).  
Figure 3: Full decision workflow for Phase I dose-finding interventional oncology trial design, split by primary endpoint: (a) MTD branch and (b) OBD branch. <sup>\*</sup>model-assisted, <sup>^</sup>model-based, <sup>+</sup>algorithm-based.

## E.1 Node-by-node resolution

1. In scope of dose-finding? A trial is in scope unless it is not dose-finding (fixed-dose or Recommended Phase 2 Dose confirmation, vaccine-immunogenicity, or expansion-only) or has no escalation dimension. Out-of-scope trials receive no design.

2. MTD or OBD? A dose-finding trial adopts either a Maximum Tolerated Dose (MTD) or an Optimal Biological Dose (OBD) objective [16]. Read from the Phase I objective.

3. Single agent or combination? Whether adjacent dose levels follow a non-monotonic toxicity order. A partially ordered (non-monotonic) dose space indicates a combination design, while a monotonic one indicates a single agent.

4. Combination target (combination only). If the design delivers a single recommended dose pair → BOIN-Comb. If it delivers the MTD curve → Waterfall. This is a leaf node. The single-agent late-onset node is not on the combination path.

5. Informative prior? For simplicity of design, this study resolves this node to negative.

6. Single or multiple/continuous DLT? Whether the trial monitors one binary DLT endpoint, or generalizes to either multiple concurrent toxicity types or a graded (non-binary) toxicity outcome.

7. Late onset? (BOIN vs. TITE-BOIN) Whether each patient’s dose-limiting toxicity is fully observed within a fixed window before the next escalation decision [15]. First, if the escalation gate of a trial opens before one cycle of dose-limiting toxicity (DLT) window closes, this trial qualifies for late onset. Otherwise, we compute W/I, the DLT assessment window divided by the interpatient accrual interval, and decide. Full decision rules are detailed in §E.2.

8. 3 levels of OBD branch nodes. Resolve utility vs. independent admissibility, MTD-assecondary, and whether the dose-selection outcomes are late-onset, yielding BOIN-ET, U-BOIN, BOIN12/EffTox, or TITE-BOIN12.

9. Leaf → label. The leaf is a set of admissible designs, ranked into Primary/Secondary/Acceptable tiers; the reported label is the Primary design.

## E.2 Late-onset determination

10. W. DLT assessment window (days). The registry primary-outcome time frame when it is the DLT/MTD/safety endpoint. Otherwise the protocol DLT-definition clause. Else cycle length × number of DLT cycles. Guards: assessment cadence is not the window (“every 3 wk for 9 wk” gives W =63, not 21). A response/survival time frame is not a DLT window. If the DLT is defined on a combination begun after a single-agent run-in, W starts with the combination. For multi-stage designs W is the governing (longest, latest-stage) window.

11. I. Interpatient interval, as a range [fast, slow]. Thefast bound is a protocol-mandated stagger if stated (I ≥ W for cohort-gated designs). A short “dose x days apart” safety floor counts here, not as the accrual pace. The slow bound is the registry accrual,

$$
I _ { \mathrm { s l o w } } = { \frac { t _ { \mathrm { c o m p l e t e } } - t _ { \mathrm { s t a r t } } - W } { N _ { \mathrm { a r m } } - 1 } } , N _ { \mathrm { a r m } } = N _ { \mathrm { t o t a l } } / k ,
$$

where k is the number of parallel dose-finding cohorts. W is subtracted because the completion date trails the last enrollment by about one window.

12. Two-bound ratio. The window-to-interval ratio spans $W / I _ { \mathrm { s l o w } }$ (lower bound, slowest accrual) to $W / I _ { \mathrm { f a s t } }$ (upper bound).

13. Decision (escalation gate is primary). (i) The protocol opens the next dose before the window closes → TITE-BOIN. (ii) Escalation waits for the full window (cohort-complete, sentinel, or single-patient full-window) → BOIN, regardless of registry accrual. Therefore, an expansion cohort that inflates whole-trial accrual cannot force a false TITE. (iii) No gate stated → fall back to accrual: $W / I _ { \mathrm { s l o w } } \geq 4 $ TITE-BOIN, else BOIN.

## E.3 OBD variant

The OBD branch has two windows, toxicity $W _ { T }$ and the dose-selection efficacy endpoint $W _ { E }$ (binarized ORR, a PD biomarker, or landmark PFS, not long-term survival). max $( W _ { T } , \bar { W } _ { E } ) / I \geq$ cutoff → TITE-BOIN12, else BOIN12/EffTox.

## F Model List and Inference Configuration

All models were accessed through a single unified serving layer (Databricks Model Serving / Foundation Model APIs) to standardize access and pin versions across providers. Decoding was deterministic with identical settings for every model and input configuration. Model details are listed in Table 6.

Table 6: Serving endpoints for the five evaluated models.
<table><tr><td>Model</td><td>Serving endpoint</td></tr><tr><td>Claude Opus 5</td><td>databricks-claude-opus-5</td></tr><tr><td>GPT-5</td><td>openai-gpt-5</td></tr><tr><td>OpenAI  $^ { 0 3 }$ </td><td>openai-gpt-o3</td></tr><tr><td>gpt-oss-120b</td><td>databricks-gpt-oss-120b</td></tr><tr><td>Llama 4 Maverick</td><td>databricks-llama-4-maverick</td></tr></table>

## G Prompts

Every input configuration shares a system block defining the role, an anti-lookup instruction, a rule for out-of-scope situation, and the class taxonomy (Appendix A). The tree input configurations (D, E) additionally include the decision tree and a node-by-node walk instruction. The per-inputconfiguration user block carries the fields of Appendix C. It is followed by representative excerpts of the tree-walk system prompt. Full verbatim prompts for every input configuration are released with the code.

You are a professional clinical trial biostatistician specializing in Phase I and Phase I/II oncology dose-finding trial design. $I . . . J$ Do not attempt to identify this specific trial or look up related materials. $I . . . J$ Base your recommendation solely on the information given here [. . . ], not on recalledfacts about any specific real trial.

If the trial described is out of scope [. . . ] (for example a PK/PD-only, imaging, single fixed-dose, or vaccine-immunogenicity-only study), mark it out of scope and do not force a recommendation.

[Tree input configurations] Follow the tree closely $I . . . J ,$ answering each node you pass through, in order, with a brief rationale, until you reach a leaf. Record every node you visit in node\_answers. [. . . ] The distinction between single-agent and combination dose-escalation is whether the dose-LEVEL sequence has a joint (non-monotonic) toxicity order — NOT how many agents are administered or how many agents’ doses vary.

Redaction. Model-visible inputs exclude the originally adopted design, reported results, and any design-revealing registry field. The case is identified by an anonymized case ID only.