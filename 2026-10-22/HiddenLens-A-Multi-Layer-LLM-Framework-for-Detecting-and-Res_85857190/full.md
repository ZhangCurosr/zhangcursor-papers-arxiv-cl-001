# HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements

Tanu Singh<sup>1</sup>, Aashna Suman<sup>2</sup>, Vinod Patidar<sup>3\*</sup>

School of Computer Science, UPES, Dehradun, Uttarakhand, India <sup>1</sup>tanu.singh@ddn.upes.ac.in, <sup>2</sup>aashna.15692@stu.upes.ac.in <sup>3\*</sup>vinod.patidar@ddn.upes.ac.in (Corresponding Author)

## ABSTRACT

Software requirements written in natural language often omit important assumptions about expected performance, operating conditions, user behaviour, and domain-specific constraints. This gap between explicitly stated requirements and stakeholders’ implicit assumptions can influence architectural choices, testing approaches, and deployment configurations. If these assumptions are not identified until implementation, correcting the resulting problems can be far more costly than addressing them at the early stages of requirements engineering. Although ambiguity and vagueness in requirements have been widely studied, less attention has been given to uncovering assumptions embedded in requirements that otherwise appear clear and reasonable. To address this gap, this paper proposes HiddenLens, a nove six-layer BERT-LLM pipeline and the first framework engineered specifically to detect, classify, question, and resolve hidden assumptions in software requirements, rather than merely flagging the existence of ambiguity. Layer 1 applies BERT attention entropy to localise the specific spans of a requirement carrying the most unresolved linguistic weight. Layer 2 passes these spans to an LLM using CoT prompting, which reasons step-by-step to articulate the precise hidden assumption each span conceals and the failure risk if that assumption is wrong. Layer 3 classifies each detected assumption with a fine-tuned BERT classifier across six non-functional requirement categories: performance, security, scalability, usability, reliability, and domain. Layer 4 generates context-aware, ranked clarification options for the analyst via FS LLM prompting, turning each assumption into concrete, answerable choices. Layer 5 is a human-in-the-loop validation step, where the analyst confirms, edits or requests a layman-mode rephrasing of each clarification before anything is finalised. Layer 6 closes the loop by rewriting the original requirement, substituting the analyst-confirmed resolution in place of the ambiguous term, and producing a requirement that is explicit, testable, and traceable back to the assumption it resolves. This paper also introduces the HiddenLens Corpus (HLC), a dedicated dataset comprising 312 requirements annotated with 847 hidden assumptions across six NFR categories. Inspired by the PROMISE and PURE datasets, HLC addresses the lack of annotated instances for hidden-assumption detection in the existing corpora. To the best of our knowledge, it is the first dataset specifically developed for this task. HiddenLens outperforms the keyword-based baseline by 71% and the QuARS-Adapted baseline by 61%, achieving an end-to-end F1-score of 0.79, while its BERT-based NFR classifier achieves an F1-score of 0.847. These findings indicate that HiddenLens can help analysts identify and resolve hidden assumptions during requirements engineering before they propagate to later stages of software development.

Keywords: Requirements Engineering, Hidden Assumptions, Large Language Models, BERT, Prompt Engineering, NFR Classification, Chain-of-Thought, Requirements Quality

## 1. Introduction

Requirements Engineering (RE) is the process of capturing, documenting, and agreeing on what a system must do before a single line of code is written, and it is widely recognized as the most consequential stage of the software development lifecycle (SDLC). The single highest-leverage activity in software development is ensuring that the requirements are correct. Previous studies show that defects identified during the requirements stage can cost up to 100 times less to fix than those discovered at later stages (Boehm, 1984; Femmer et al., 2017). RE shapes every downstream decision, system architecture, test coverage, resource allocation, and ultimately whether the delivered product matches stakeholder requirements. Basically, RE is the blueprint stage of the SDLC. If software is built on poorly stated requirements, the project will have issues at every layer and lead to software failures. RE has long been recognised for its importance but still proves to be a very underserved and error-prone stage in practice. Such issues lead to constant defects that eventually turn into project failures.

The consequences of such failures have been examined extensively in previous studies. One such work has identified requirements defect in the top three root causes of systemic failures. The rest two include poor project management and inadequate testing (Johnson, 2020; Leffingwell & Widrig, 2003). There are three major types of requirements problems that can lead to failures. First, incompleteness, wherein required behaviours are not captured. Second, inconsistency, where requirements contradict each other across documents. Third, ambiguity, where the language used allows multiple interpretations. Ambiguity is one of the most pervasive and difficult-to-detect requirements problems.

Ambiguity does not appear out of nowhere; it sneaks in through the way requirements are written, and almost always, that means the use of natural language. Stakeholders use plain language because it is easy for everyone involved to understand, whether they are developers, managers, or clients. It is a genuine strength of natural language, but it comes with a hidden cost. When stakeholders agree that "the system shall respond quickly" or "data shall be stored securely" everyone appears in agreement but having different interpretations of these statements. To one person, "quickly" may mean under two seconds; however, for another person it may mean instantly. Similarly, “securely" might mean full encryption to a developer, but for a business manager it might mean that the system does not get hacked.

Table 1: Examples: Requirements and Vague Words
<table><tr><td rowspan=1 colspan=1>Requirements</td><td rowspan=1 colspan=1>Vague Words</td></tr><tr><td rowspan=1 colspan=1>The system shall process user requests quickly.</td><td rowspan=1 colspan=1>Quickly</td></tr><tr><td rowspan=1 colspan=1>Patient data shall be stored securely.</td><td rowspan=1 colspan=1>Securely</td></tr><tr><td rowspan=1 colspan=1>Nurses can update patient status in real time.</td><td rowspan=1 colspan=1>real time</td></tr><tr><td rowspan=1 colspan=1>The system should respond immediately in emergencies.</td><td rowspan=1 colspan=1>Immediately</td></tr><tr><td rowspan=1 colspan=1>User data shall be backed up regularly.</td><td rowspan=1 colspan=1>regularly</td></tr><tr><td rowspan=1 colspan=1>The interface shall be intuitive for new users.</td><td rowspan=1 colspan=1>intuitive</td></tr><tr><td rowspan=1 colspan=1>The application shall be user-friendly.</td><td rowspan=1 colspan=1>user-friendly</td></tr><tr><td rowspan=1 colspan=1>The system shall support a large number of concurrent users.</td><td rowspan=1 colspan=1>large number</td></tr><tr><td rowspan=1 colspan=1>Reports shall be generated efficiently.</td><td rowspan=1 colspan=1>efficiently</td></tr><tr><td rowspan=1 colspan=1>The system shall be highly available.</td><td rowspan=1 colspan=1>highly available</td></tr><tr><td rowspan=1 colspan=1>Administrators shall be able to easily configure system settings.</td><td rowspan=1 colspan=1>easily</td></tr><tr><td rowspan=1 colspan=1>The system shall handle errors gracefully.</td><td rowspan=1 colspan=1>gracefully</td></tr></table>

Additional examples of requirements and vague words are listed in Table 1. In such a scenario, no one is wrong exactly, but stakeholders are not sharing the same understanding of requirements. These are not spelling mistakes or formatting issues; rather, they are gaps in meaning hidden behind words that look perfectly reasonable on paper. The real problem is that these gaps remain unnoticed during requirements review, because the requirement appears to be complete and well written. The intended meaning of the word “quickly” might not be questioned until the system is built and the client finds that the delivered system does not meet their expectations.

As these vague words are rarely stated explicitly or questioned during the requirements review process, stakeholders often have to fill in these gaps with their own interpretations that become hidden assumptions. Such assumptions may appear harmless at the early stages of development; however, they can lead to different expectations among stakeholders and consequently the deviations between the specified and expected system behaviour. Identifying these hidden assumptions is therefore important for detecting potential requirements failure at an early stage. Figure 1 illustrates how such hidden assumptions can remain unnoticed during requirements specification and eventually contribute to project failure.

![](images/413d135c8e2852fb69fb710eec02b6f05ee0a3bf985fd67b447fcedd3bedc729.jpg)  
Figure 1. Hidden Assumptions Leading to Project Failure

The research community has not overlooked the problem of hidden assumptions; however, it has largely addressed a different manifestation of the issue. Existing studies on the detection of syntactic and semantic ambiguities (Ferrari et al., 2024; Gervasi et al., 2019), the classification of requirements smells (Femmer et al., 2017), and automated quality assessment against standards such as ISO/IEC/IEEE 29148 (2018); Lubos et al., 2024 mainly focus on poorly specified or incorrectly written requirements. Similarly, tools such as RETA (Bajceta et al., 2022) and QuARS (Lami et al. 2005) can identify ambiguous terms and implicit pronoun references, while rule-based keyword detectors can flag hedging or imprecise language. However, these approaches cannot explicitly identify the underlying assumption obscured by vague terms. Consequently, they detect symptoms of incompleteness or ambiguity but cannot identify the specific information missing.

To bridge this gap, this paper presents HiddenLens: a novel six-layer Bidirectional Encoder Representations from Transformers- Large Language Model (BERT–LLM) pipeline for detecting, classifying and resolving hidden assumptions in software requirements. Rather than being another ambiguity detector, HiddenLens addresses a fundamentally different question: what does a requirement silently depend on? Instead of merely identifying what is ambiguous in a requirement, HiddenLens seeks to uncover the underlying assumptions, determine their nature, facilitate their validation and resolve them by formulating requirements that are clearer and more explicit.

The proposed framework follows an end-to-end process that combines automated linguistic analysis, LLMbased assumption identification & clarification generation, requirement classification, and human-in-the-loop (analyst) validation, culminating in the rewriting of the original requirement. This design enables HiddenLens to move beyond detecting surface-level vagueness to identifying and resolving assumptions that may remain implicit in otherwise seemingly acceptable requirements. The framework is evaluated using requirements drawn from the publicly available PROMISE (Cleland-Huang et al., 2007) and PURE (Ferrari et al., 2017) datasets, supplemented by industrial documents from healthcare partners and synthetically generated examples, with the resulting dataset manually annotated for hidden assumptions across six non-functiona requirement (NFR) categories. We refer to this dataset as the HiddenLens Corpus (HLC) and to support reproducibility, we publicly release its synthetic portion: 312 requirements with 847 annotated hidden assumptions as a companion resource to PROMISE and PURE. Through this integrated approach, HiddenLens aims to support requirements analysts and software developers in improving the clarity, completeness, and interpretability of requirements at an early stage of software development.

The major contributions of this paper are as follows:

Novel End-to-End Framework: HiddenLens is a six-layer pipeline that integrates BERT-based attention entropy, LLM-based chain-of-thought reasoning, fine-tuned NFR classification, contextaware suggestion generation, analyst validation through a human-in-the-loop mechanism, and automated requirement rewriting. It is the first end-to-end framework designed to detect and resolve hidden assumptions in software requirements.

Prompt Engineering Strategy: The study provides an in-depth comparison of zero-shot (ZS), fewshot (FS) (Brown et al., 2020), and chain-of-thought (CoT) prompting (Wei et al., 2022) for hiddenassumption detection and clarification generation. Based on this comparison, an appropriate prompting strategy is selected for each task.

• Annotated Dataset: The HiddenLens Corpus (HLC) comprises 312 requirements containing 847 annotated assumptions across six NFR categories and is made available to the research community.

Empirical Evaluation: The BERT classifier obtains an F1-Score = 0.847 and the human evaluation of hidden assumption resolution yields precision = 0.81, recall = 0.78 and F1-score= 0.79. Therefore, the proposed approach achieves substantive improvements over rule-based (Fabbrini et al. (2001)) and keyword-based (Wilson et al. (1997)) baselines. It is further supported by a detailed error analysis and an ablation study of various prompt strategies.

The rest of the paper is organized as follows. Section 2 reviews related work on requirements ambiguity detection, NFR classification, and LLM applications in requirements engineering, along with prior approaches that flag ambiguity but do not resolve it. Section 3 formally defines a hidden assumption and introduces the six-category taxonomy used throughout the paper. Section 4 presents the six-layer HiddenLens framework in detail, from BERT attention entropy localisation and chain-of-thought assumption detection through NFR classification, clarification generation, analyst validation, and automated requirement rewriting. Section 5 describes the construction of the HiddenLens Corpus (HLC) and the annotation process behind it. Section 6 details the experimental setup, including model configuration, baseline methods, and both automated and human evaluation metrics. Section 7 presents the quantitative results, a comparative analysis against prior work, and sample outputs. Section 8 walks through a case study on a Hospital Patient Management System to illustrate the framework end-to-end. Section 9 discusses threats to validity, and Section 10 concludes with a summary of contributions and directions for future work.

## 2. Related Work

Early work on pattern-based detection emerged in the 1990s. One of the earliest systems, the Automated Requirements Measurement (ARM) tool, examined requirements specifications for linguistic indicators of poor quality, including vague, weak, optional, and subjective expressions (Wilson et al., 1997).In parallel, Ambriola and Gervasi (1997) explored the automated processing of natural-language requirements. These efforts helped make linguistic defects amenable to automated analysis. QuARS (Lami et al. 2005) subsequently extended this direction by combining lexical dictionaries with morphological and syntactic analysis to detect ambiguity, understandability, and specification-completeness problems (Fabbrini et al., 2001; Lami et al., 2004).

The work on conceptual and heuristic refinement shifted the attention from patterns to the thinking behind them. Berry et al. (2003) made an important point that not every ambiguity is similar. Some of it is deliberate, a real expression of uncertainty on the part of stakeholders, and some of it is accidental, just a consequence of bad writing. They argued that automated tools should be able to differentiate these, a nuance that most tools overlooked. This line of thought was followed and formalised by Femmer et al. (2017), who introduced the notion of requirements smells: named as anti-patterns, similar to code smells, that indicate a degradation in quality. Their Smella tool used part-of-speech tagging and heuristic rules to find these patterns at scale, giving practitioners a common vocabulary for requirements defects. However, it could only report that a smell was present and could not diagnose what was missing, like its predecessors.

The advancements in statistical machine learning subsequently expanded the range of features detectable beyond those captured by manually formulated rules. Winkler and Vogelsang (2016) applied convolutional neural network (CNN) and long short-term memory (LSTM) architectures to the PROMISE NFR dataset (Cleland-Huang et al., 2007), learning distributed representations of requirements text rather than relying on bag-of-words features and thereby improving non-functional requirement classification. However, their models still required substantial amounts of labelled data for each category, could not be transferred across domains without retraining, and were limited to classification. For example, they could classify a requirement as performance-related but could not identify its implicit assumptions regarding performance. Arora et al. (2016) trained classifiers using annotated corpora of industrial requirements and substantially outperformed preceding syntactic approaches. Nevertheless, these classifiers faced the same limitation: a defect had to be explicitly discernible in the text. Similarly, Guo et al. (2017) applied a deep-learning architecture combining word embeddings with bidirectional gated recurrent units (GRUs) to software traceability, substantially outperforming vector-space and latent semantic indexing baselines. Across both tasks, the richer neural representations improved on earlier surface-level methods, but the object of analysis remained content already present in the text, not the information it silently left out. Abad et al. (2017) compared topic-modelling and Naive Bayes (NB) approaches on the same repository and showed that preprocessing choices alone could substantially shift classifier performance, underscoring how sensitive these statistical methods were to pipeline design rather than to the underlying requirements content.

Transformer-based models further advanced requirements classification. Hey et al. (2020) introduced Nonfunctional and functional Requirements classification using BERT(NoRBERT). This BERT-based model automatically classifies functional and non-functional requirements and achieved a precision of 0.92 on the PROMISE dataset (Cleland-Huang et al., 2007). Later, Luo et al. (2022) proposed Prompt Learning for Requirement Classification (PRCBERT), which applies flexible prompt templates to BERT-style models to achieve a 72.96% F1-score in zero-shot classification on held-out data. In parallel, Ferrari et al. (2024) and Beqiri et al. (2024) studied defect detection in industrial and safety-critical domains, namely automotive & railways, and both have cited domain dependence as the ceiling they could not break through. Lubos et al. (2024) compared LLM-based quality assessment against ISO/IEC/IEEE 29148 (2018) standards and obtained a high recall but modest precision, thus recommending a mixed human-LLM workflow.

A broader perspective emerged with the meta-analysis conducted by Necula et al. (2024) wherein a systematic review examined 137 Natural Language Processing (NLP)-in-RE studies published in the period 1991 to 2024. They found a pattern that cuts across every generation of tooling in the field consistently targeted what is written poorly, not what is left out entirely. Over time, more complex models, richer libraries of patterns, and deeper representations of the linguistic data were developed, addressing the same class of problems. Comparatively less research has been done on the implicit requirements and the underlying presumptions.

The emergence of large language model-based (LLM-based) reasoning has expanded the range of questions that can be investigated and answered in requirements engineering (RE). In contrast to pattern-matching approaches, such as rule-based models or fine-tuned classifiers, LLMs can apply domain knowledge, perform reasoning, and generate human-readable explanations. Brown et al. (2020) first showed that LLMs could perform new tasks from only a handful of in-context examples, without fine-tuning, establishing Few-shot (FS) prompting as a viable alternative to task-specific training. Wei et al. (2022) revealed that using the chainof-thought (CoT) prompts leads to substantial gains in solving inferential problems. Hymel and Johnson (2025) revealed that the GPT-4 achieved expert-level requirements quality with substantially lower costs. In an extensive review, Zadenoori et al. (2025) analyzed 74 works and identified a clear shift from pattern matching to semantic reasoning. Almonte et al. (2025) demonstrated that LLMs can generate plausible non-functional requirements directly from stated functional requirements, resulting strong alignment with expert judgment. This generative task still operated on the information explicitly stated in the original requirements rather than uncovering implicit assumptions that requirements silently rely on. The findings by Unterbusch and Vogelsang (2026) proved that a FS and CoT prompting, combined with persona augmentation, produces the best performance results for requirements benchmarking, adapting to a new domain with very few examples.

Across all the aforementioned technological advances, existing approaches have not directly addressed the identification and resolution of hidden assumptions. LLMs have primarily been employed in requirements engineering to classify, formalise, or rephrase requirements. However, these processes continue to operate on information explicitly present in the text. An LLM may identify a vague term as a requirements smell but does not necessarily articulate the underlying assumption associated with it. The closest antecedent is the analogybased reasoning framework proposed by Onyeka (2013), which preceded LLMs and was not scalable. As described by Nuseibeh et al. (2001), unvalidated assumptions can translate into architectural flaws. Although the IEC 61508 standard (Häring, 2021) requires such assumptions to be documented, industry reports indicate that these requirements are not consistently followed in practice. Thus, the problem of hidden assumptions has been recognised, and its consequences have been documented in the form of system failures; however, a scalable tool for systematically detecting and addressing these assumptions has remained unavailable.

In order to address each of the aforementioned issues, this study proposes HiddenLens, a novel six-layer BERT-LLM pipeline for detecting, classifying, and resolving hidden assumptions in software requirements. The proposed framework is designed to address a gap that has remained unresolved despite decades of tooling in the RE domain. The problem definition adopted in this study is outlined in the next section.

## 3. Problem Definition

This section formulates the problem in two parts. First, it defines a hidden assumption and distinguishes it from a requirement that is merely vague or unclear, as these constitute separate problems addressed by existing tools. Second, it describes the six categories of assumptions considered throughout this paper.

## 3.1 Formal Definition

Let � denote a natural language software requirement sentence. Let $W ( R )$ denote the set of all propositions that a competent software architect would need to assume to be true or satisfy to correctly implement �. Let $E ( R ) \subseteq W ( R )$ denote the set of propositions explicitly stated in �. We define the hidden assumption set �(�) as:

$$
A ( R ) = W ( R ) \setminus E ( R )
$$

That is, a hidden assumption $a \in A ( R )$ is a proposition that is necessary for the correct implementation of � but is not stated in �. This definition distinguishes hidden assumptions from (i) ambiguity, where $a \in E ( R )$ but admits multiple conflicting interpretations, and (ii) vagueness, where $a \in E ( R )$ but is insufficiently precise. Hidden assumptions are absent, not merely ambiguous or imprecise. This is depicted clearly in Figure 2, which showcases the relationship between requirements and their associated assumptions, sourced from our own analysis.

Hidden assumptions are difficult to detect for three principal reasons. First, they are invisible in the surface text: no linguistic signal marks their absence. Second, they are often domain-dependent: They are often domain-dependent: what counts as a necessary unstated assumption depends on the application domain, deployment context, and stakeholder community. Third, they are spread across multiple NFR categories at once: a single requirement may embed performance, security and environmental assumptions simultaneously.

![](images/a428b8267e18a6dfca8de848e242c032eb490a96ad9af1055e47919931133d44.jpg)  
Figure 2. Requirement-Assumption Relationship

## 3.2 Assumption Taxonomy

An analysis of the 312 requirements in the HiddenLens Corpus (HLC) and the literature on NFR classification led to the identification of six principal categories of assumptions, which are described in detail in Section 5. Table 2 presents this taxonomy, together with definitions and illustrative examples.

Table 2: Hidden Assumption Taxonomy
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Example Assumption</td></tr><tr><td rowspan=1 colspan=1>Performance</td><td rowspan=1 colspan=1>Unstated quantitative thresholds forresponse time, throughput, or resourceutilisation.</td><td rowspan=1 colspan=1>&quot;Quickly&quot; implies response latency &lt; 200 ms under1,000 concurrent users.</td></tr><tr><td rowspan=1 colspan=1>Security</td><td rowspan=1 colspan=1>Unstated encryption standards, accesscontrol models, or compliancerequirements.</td><td rowspan=1 colspan=1>&quot;Securely&quot; implies AES-256 encryption and RBACwith principle of least privilege.</td></tr><tr><td rowspan=1 colspan=1>Scalability</td><td rowspan=1 colspan=1>Unstated user load, data volume, orgrowth trajectory assumptions.</td><td rowspan=1 colspan=1>&quot;Large number of users&quot; implies horizontal scalingto 10,000 concurrent sessions.</td></tr><tr><td rowspan=1 colspan=1>Usability</td><td rowspan=1 colspan=1>Unstated user expertise levels, devicetypes, or interface accessibilityrequirements.</td><td rowspan=1 colspan=1>&quot;Easy to use&quot; assumes non-technical administrativestaff as primary users.</td></tr><tr><td rowspan=1 colspan=1>Reliability</td><td rowspan=1 colspan=1>Unstated availability SLAs, faulttolerance models, or recovery timeobjectives.</td><td rowspan=1 colspan=1>&quot;Always available&quot; assumes 99.9% uptime withsub-hour RTO.</td></tr><tr><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>Unstated domain-specific rules,regulatory mandates, or integrationconstraints.</td><td rowspan=1 colspan=1>&quot;Patient data&quot; implies HIPAA compliance and audittrail requirements.</td></tr></table>

## 4. The HiddenLens Framework

HiddenLens is a novel six-layer BERT-LLM pipeline for detecting, classifying, and resolving hidden assumptions in software requirements. It engages an analyst to clarify hidden assumptions and rewrite the requirement to make them explicit. Figure 3 illustrates the full architecture of the proposed HiddenLens framework.

![](images/c6d6011fad2f48f02d9209e0c71ee32fcea15b753b8cdd73669dd91a5a0c9592.jpg)  
Figure 3. HiddenLens Architecture

Layer 1 applies BERT attention entropy to localise the specific spans of a requirement carrying the most unresolved linguistic weight. Layer 2 passes these spans to an LLM using CoT prompting, which reasons step-by-step to articulate the precise hidden assumption each span conceals and the failure risk if that assumption is wrong. Layer 3 classifies each detected assumption with a fine-tuned BERT classifier across six non-functional requirement categories: performance, security, scalability, usability, reliability, and domain. Layer 4 generates context-aware, ranked clarification options for the analyst via FS LLM prompting, turning each assumption into concrete, answerable choices. Layer 5 is a human-in-the-loop validation step, where the analyst confirms, edits or requests a layman-mode rephrasing of each clarification before anything is finalised. Layer 6 closes the loop by rewriting the original requirement, substituting the analyst-confirmed resolution in place of the ambiguous term, and producing a requirement that is explicit, testable, and traceable back to the assumption it resolves. In the following subsections, we provide details of each layer of the HiddenLens framework, including its architecture, functionality, and how it helps in identifying and resolving hidden assumptions in software requirements.

## 4.1 Layer 1: BERT Attention Entropy (Ambiguity Localisation)

The pipeline begins by identifying candidate ambiguous terms within the requirement sentence. We exploit the observation that BERT's attention weights, when averaged across heads and layers, are distributed more uniformly across tokens when a term is semantically underspecified; high entropy signals uncertain contextualisation. For each non-stop-word token � in the requirement sentence, we compute:

$$
H ( w ) = - { \Sigma } { \alpha } _ { i j } l o g ( { \alpha } _ { i j } ) , i \in l a y e r s , j \in h e a d s ,
$$

A version (09\_Sept\_2026) of this manuscript has been submitted to the journal for publication and is presently under review where $\alpha _ { i j }$ denotes the mean attention weight assigned t.� by head � of layer �. Tokens whose entropy scores exceed 60% of the maximum observed score in the sentence are flagged as ambiguous and passed to Layer 2 as candidate assumption anchors. For cases meeting this threshold, the approach identifies candidate anchors without additional training and can be applied across requirement domains, unlike keyword-based methods that rely on static lexicons of vague terms.

![](images/d78e0af989c156f9b2a0301d0d657e22adddcf522813ed651593a67c73a88904.jpg)  
Figure 4. Layer 1 Architecture

Figure 4 depicts the workflow of Layer 1 visually. Firstly, the training files are loaded and the tokenizer is initialized to convert text into token IDs. Then the BERT Dataset CLASS combines requirements and assumptions in order for the class weights to be calculated. The dataloaders then build train, validation and test splits, which are then sent to the BERT model after it is initialized. Afterwards, the optimizer and scheduler configure AdamW optimization and figure out the scheduling of tasks. The training loop then takes over to train the model through forward pass, backpropagation, etc. The validation phase evaluates model performance for the best F1 score and saves the best model. Finally, the classification metrics are generated for review.

## 4.2 Layer 2: LLM Chain-of-Thought Assumption Detection

![](images/310631dc96e287dc075b2e1587896e768d868d03e7ed82ab001985d2b74c428e.jpg)  
Figure 5. Layer 2 Architecture

A version (09\_Sept\_2026) of this manuscript has been submitted to the journal for publication and is presently under review

Given the requirement � and the set of ambiguous terms $T = \{ t _ { 1 } , \ldots , t _ { k } \}$ from Layer 1, Layer 2 invokes an LLM with a CoT prompt to generate structured assumption records. The system prompt instructs the model to act as an expert requirements engineering analyst, reason step-by-step about the implementation world-model each ambiguous term presupposes, and return a JSON structure containing: (i) the ambiguous term, (ii) the hidden assumption text, (iii) the failure risk if the assumption is wrong, and (iv) a clarifying question for the analyst, all three components are shown pictorially through Figure 5 for better understanding.

## 4.2.1 Prompt Engineering Strategy

We evaluated three prompting strategies in Layer 2: Zero-shot (ZS) (Brown et al., 2020), few-shot (FS) (Brown et al., 2020), and CoT (Wei et al., 2022). In ZS mode, the model receives only the task description and the requirement. In FS mode, three hand-crafted requirement-assumption examples are prepended to the prompt. In CoT mode, few-shot examples include explicit reasoning traces that model the step-wise inferential process from requirement text to assumption. This progression mirrors the comparative methodology of Binkhonain and Alfayez (2025), who found that few-shot and persona-augmented CoT prompting matched or exceeded supervised baselines in requirements classification. Table 3 presents the strengths and weaknesses of the prompt strategy techniques used, and Figure 6 depicts the same via a detailed visual architecture.

Table 3: Prompt Strategy Comparison
<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Strengths</td><td rowspan=1 colspan=1>Weaknesses</td></tr><tr><td rowspan=1 colspan=1>Zero-Shot (ZS)</td><td rowspan=1 colspan=1>Task description +requirement only. Noexamples.</td><td rowspan=1 colspan=1>No example overhead; fast.</td><td rowspan=1 colspan=1>Lower precision on implicitinference tasks.</td></tr><tr><td rowspan=1 colspan=1>Few-Shot (FS)</td><td rowspan=1 colspan=1>3 labelled requirement-assumption pairsprepended.</td><td rowspan=1 colspan=1>Grounds model in taskformat; improves recall.</td><td rowspan=1 colspan=1>Sensitivity to exampleselection quality.</td></tr><tr><td rowspan=1 colspan=1>CoT</td><td rowspan=1 colspan=1>Few-shot examplesaugmented with step-by-step reasoning traces.</td><td rowspan=1 colspan=1>Best precision and coherence;structured JSON output.</td><td rowspan=1 colspan=1>Higher token cost; occasionalover-explanation.</td></tr></table>

![](images/867befafc19cd2fba3582ab4aabc3ba41e8a4fbb5e1e4de4a645965a1e2a7de1.jpg)  
Figure 6. Prompt Engineering Strategy Architecture

Based on empirical evaluation, CoT prompting achieves the best precision-recall balance and is adopted as the default strategy. The full CoT prompt template is reproduced below:

System: You are an expert requirements engineering analyst. Detect hidden assumptions in the given software requirement. For each assumption: (1) identify the ambiguous term, (2) state the hidden assumption explicitly, (3) describe the failure risk, (4) formulate a clarifying question. Reason step-by-step before generating JSON output.

User: Requirement: {requirement} | Ambiguous terms: {terms} | Think step-by-step then return JSON: {"assumptions": [{"ambiguous\_term": ..., "assumption\_text": ..., "failure\_risk": ..., "clarifying\_question": ...}]}

## 4.3 Layer 3: Fine-Tuned BERT NFR Classifier

Each assumption hypothesis generated by Layer 2 is classified into one of the six NFR categories using a finetuned BERT for sequence classification model. As highlighted in Figure 7, the classifier receives the concatenated sequence [REQ] {requirement} [ASSUMPTION] {assumption} as input, enabling it to make category decisions that are sensitive to both the requirement context and the assumption content. Training details are provided in Section 6.

![](images/8f7fafde9eac2d0658a3d9c2033a3a9a3418c87c955c02612bacd0b3312b9198.jpg)  
Figure 7. Layer 3 Architecture

## 4.4 Layer 4: LLM Clarification Suggestion Generation

Layer 4 generates a ranked list of three to five specific clarification options for the analyst for each classified assumption. The LLM is prompted with the requirement, the assumption text, the NFR category and the clarifying question, and asked to produce specific, actionable answer options that an analyst can choose or adapt. For instance, if we assume that 'quickly' means a threshold of performance, the layer would produce options like 'within 200 ms', 'within 500 ms for 95th percentile', and 'within 1 second under peak load'. A subfunction of layman mode rephrases the clarifying question in plain English for non-technical stakeholders. A visual flowchart that elaborates the same process in detail is shown in Figure 8.

![](images/e347c2ad76f5cec4059f184eae7c792b669b112e0a1ab9b9060075bbfa0eb8da.jpg)  
Figure 8. Layer 4 Architecture

## 4.5 Layer 5: Analyst Validation Loop

HiddenLens incorporates a human-in-the-loop validation step, where the analyst reviews each assumption, its categorisation and the suggested clarifications. The analyst can choose a pre-generated answer, type a custom free-text answer (validated for relevance and coherence by the LLM), or ask for a layman rephrasing of the clarifying question. This design is aligned with the results of Unterbusch and Vogelsang (2026), who showed that human-LLM collaborative validation substantially outperforms fully automated approaches on requirements quality tasks. Inputs to Layer 5 include Requirements, Ambiguous Terms, NFR category, Clarifying Question and Suggested Option, and all of these are subsequently passed through an Analyst Review Interface and Analyst Actions as shown in Figure 9.

![](images/564532001da868d24142e60b929e32b4e32b5b4b1e8656cf43bdb03a0a28104f.jpg)  
Figure 9. Layer 5 Architecture

## 4.6 Layer 6: Requirement Rewriter

Once all clarifications are collected, Layer 6 rewrites the original requirement by replacing each ambiguous term with the analyst-confirmed resolution using boundary-aware regular expression matching. The output is a refined requirement that explicitly states the assumptions that were implicit previously, as shown in Figure 10. For example, ‘The system shall process user requests quickly’ would be ‘The system shall process user requests within 200 ms at 95th percentile’ under 1,000 concurrent users. Algorithm 1 below depicts the flow of the hidden assumption detection pipeline followed in HiddenLens.

Algorithm 1: HiddenLens Hidden Assumption Detection Pipeline Input: R - natural language requirement sentence Output: R' - rewritten requirement with explicit assumptions

1: T ← BERT\_AttentionEntropy(R, top\_n=4, threshold=0.60) // Layer 1   
2: if |T| = 0 then return R // no ambiguous terms found   
3: A ← LLM\_CoT\_Detect(R, T) // Layer 2: assumption hypotheses   
4: for each a ∈ A do   
5: a.category ← BERT\_Classify([REQ] R [ASSUMPTION] a) // Layer 3   
6: a.suggestions ← LLM\_Suggest(R, a, a.category) // Layer 4   
7: end for   
8: C ← {} // clarification map   
9: for each unique term t in T do   
10: c ← Analyst\_ValidateLoop(t, A[t], R) // Layer 5   
11: C[t] ← c   
12: end for   
13: R' ← Rewrite(R, C) // Layer 6   
14: return R'

![](images/507d36fbe05fc207118e1da2ce4590f1c1315584aad344e7a22f68fbd6dc69f1.jpg)  
Figure 10. Layer 6 Architecture

The next section describes how the dataset used to train and evaluate HiddenLens was built, since no existing dataset of hidden assumptions in software requirements was available for this purpose.

## 5. Dataset and Annotation

Since no annotated corpus dealing with hidden assumptions in software requirements currently exists, we have created a dedicated dataset for this study: the HiddenLens Corpus (HLC), addressing what Abualhaija et al. (2024) identify as a systemic barrier in NLP-for-RE research, where annotated datasets remain scarce, and only a small fraction of published tools or data are ever made publicly available. The requirements were obtained from different sources, namely: SRS documents publicly available in academic repositories such as the PROMISE NFR dataset (Cleland-Huang et al., 2007) and PURE dataset (Ferrari et al., 2017) and requirements from the novel HiddenLens Corpus (HLC). The final dataset, HLC, comprises 312 requirements containing 847 annotated assumptions

The workflow is illustrated in Figure 11. It begins by randomly shuffling the list of requirements to minimise category-ordering bias. The code then iterates through each requirement and first determines whether all six NFR categories—performance, security, scalability, usability, reliability, and domain—have reached the maximum limit of 60 items per category. If this condition is satisfied, the loop terminates. For each eligible requirement, the annotate\_requirement( ) function sends the requirement text to the language model through the OpenRouter API. The temperature is set to 0.3, and the request is retried up to three times in the event of a rate-limit error. The result is stripped of Markdown fences and parsed as JSON to get the inferred NFR category and list of assumptions. If there are no assumptions in the result, or if the inferred category is full, this item is simply skipped. Otherwise, the result is added to the output list, and the respective category counter is increased by one. Progress and distribution by categories are logged every ten items. Sleeping for half a second after each iteration ensures that the API rate limit is not exceeded. Once the loop completes, the saved items from previous runs are loaded and added to the new annotated results. The final dataset is saved to a file, and a summary is printed.

![](images/82d16d93282f3048c74881d33d679d84cc65b2535a9beb8f9084de1685fec492.jpg)  
Figure 11. Automated Requirement Annotation Process

The dataset was partitioned into training (70%), validation (15%), and test (15%) splits using stratified sampling to preserve category proportions, as detailed in Table 4. For BERT classifier training (Layer 3), samples are requirement-assumption pairs labelled with the assumption category. For T5 generative model training (Layer 2 offline fallback), samples are (requirement, structured assumption string) pairs.

Table 4: HiddenLens Corpus (HLC) Statistics
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Requirements</td><td rowspan=1 colspan=1>Assumptions</td></tr><tr><td rowspan=1 colspan=1>Performance</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>148</td></tr><tr><td rowspan=1 colspan=1>Security</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>174</td></tr><tr><td rowspan=1 colspan=1>Scalability</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>118</td></tr><tr><td rowspan=1 colspan=1>Usability</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>127</td></tr><tr><td rowspan=1 colspan=1>Reliability</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>163</td></tr><tr><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>117</td></tr><tr><td rowspan=1 colspan=1>TOTAL</td><td rowspan=1 colspan=1>312</td><td rowspan=1 colspan=1>847</td></tr></table>

## 6. Experimental Setup

This section describes the configuration of each model used in HiddenLens, together with the baseline methods against which the framework is evaluated.

## 6.1 Model Configuration

Table 5 provides an overview of the parameters used across the three components of HiddenLens, namely, the BERT classifier in Layer 3, the T5 fallback generator in Layer 2 offline mode, and the LLM used for CoTbased detection of assumptions and generation in Layers 2 and 4.

Table 5: Model Configuration Parameters used in HiddenLens
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>BERT Classifier (L3)</td><td rowspan=1 colspan=1>Base model</td><td rowspan=1 colspan=1>bert-base-uncased</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Max sequence length</td><td rowspan=1 colspan=1>256 tokens</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Learning rate</td><td rowspan=1 colspan=1>2e-5</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>5 (best validation F1)</td></tr><tr><td rowspan=1 colspan=1>T5 Generator (L2 fallback)</td><td rowspan=1 colspan=1>Base model</td><td rowspan=1 colspan=1>t5-small</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Learning rate</td><td rowspan=1 colspan=1>3e-4</td></tr><tr><td rowspan=1 colspan=1>LLM (Layers 2, 4)</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>meta-1llama/llama-3.3-70b-instruct</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Temperature</td><td rowspan=1 colspan=1>0.2 (detection), 0.7(suggestions)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Max tokens</td><td rowspan=1 colspan=1>1000</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Prompt strategy</td><td rowspan=1 colspan=1>Chain-of-thought (default)</td></tr></table>

BERT-base-uncased was chosen as the base model for NFR classifier because of its excellent performance in sequence classification tasks, having been extensively studied in terms of training dynamics, and being easy to fine-tune owing to the availability of computational hardware. It has been noticed that 256 tokens is the optimal token length for a sequence, as fewer than 0.3% of the input sequences required truncation. The learning rate of 2e-5 with early stopping on validation F1 follows the fine-tuning recommendations established for BERT-family models. For the LLM component, the temperature of 0.2 for assumption detection reflects the requirement for deterministic, factual inference: higher temperatures introduced variability in the logical structure of assumption statements that reduced inter-run consistency. The elevated temperature of 0.7 for suggestion generation is intentional, this task benefits from lexical diversity in the generated clarification options, giving analysts a richer menu of choices rather than semantically near-identical variants of the same resolution.

## 6.2 Baseline Methods

We compare HiddenLens against three baseline methods. (1) Keyword-Based (KB): A rule-based system (Wilson et al., 1997) that flags requirements containing vague terms from a curated lexicon (e.g., ‘quickly’, ‘efficiently’, ‘securely’, ‘in real time’) and retrieves pre-written generic assumption statements for each term. (2) QuARS-Adapted (QA) (Lami et al. 2005): The QuARS syntactic analyser extended with assumption-class labels assigned to implicit, under-specification, and vagueness detections. (3) Zero-Shot LLM (ZS-LLM) (Brown et al., 2020): The Layer 2 LLM component with zero-shot prompting only, without BERT attention pre-filtering, CoT reasoning traces, or few-shot examples. This baseline isolates the contribution of our prompt engineering and architectural integration choices.

## 6.3 Evaluation Metrics

HiddenLens is put to the test using two methodologies, namely: automated metrics against held-out annotations and human evaluation by RE practitioners.

## 6.3.1 Automated Metrics

Macro-averaged Precision, Recall and F1 on the test split for Layer 3’s BERT classifier are reported as follows:

$$
P = T P / ( T P + F P ) , R = T P / ( T P + F N ) , F 1 = 2 \times P \times R / ( P + R )
$$

A version (09\_Sept\_2026) of this manuscript has been submitted to the journal for publication and is presently under review

A predicted assumption is counted as a true positive in Layer 2 if the text has a ROUGE-L overlap ≥ 0.4 with any gold annotation for the same requirement, reflecting semantic rather than verbatim match.

## 6.3.2 Human Evaluation

The test split gave rise to requirement-assumption pairs and was presented to humans who did not participate in the annotation. Each practitioner rated the assumptions on three criteria using a 5-point scale as depicted in Table 6.

Table 6: Human Evaluation Criteria
<table><tr><td rowspan=1 colspan=1>Criterion</td><td rowspan=1 colspan=1>Definition</td><td rowspan=1 colspan=1>Rating Scale</td></tr><tr><td rowspan=1 colspan=1>Relevance</td><td rowspan=1 colspan=1>Does the assumption represent a genuine implicitbelief that affects implementation?</td><td rowspan=1 colspan=1>1 (irrelevant), 5 (highlyrelevant)</td></tr><tr><td rowspan=1 colspan=1>Clarity</td><td rowspan=1 colspan=1>Is the assumption stated in clear, actionable languagethat an analyst can act upon?</td><td rowspan=1 colspan=1>1 (unclear), 5 (very clear)</td></tr><tr><td rowspan=1 colspan=1>Usefulness</td><td rowspan=1 colspan=1>Would addressing this assumption improverequirement quality and reduce implementation risk?</td><td rowspan=1 colspan=1>1 (not useful), 5 (very useful)</td></tr></table>

## 7. Results and Discussion

This section presents a quantitative evaluation of HiddenLens against the baseline models and previous approaches, followed by representative outputs illustrating the framework’s performance in practice.

## 7.1 Quantitative Results

HiddenLens with CoT prompting achieves an F1 of 0.79 on assumption detection, representing a 71% relative improvement over the keyword-based baseline and a 61% improvement over QuARS-Adapted (Lami et al. 2005). The comparison between ZS-LLM and HiddenLens ZS demonstrates that BERT attention pre-filtering alone contributes approximately 3 F1 points by constraining the LLM search space to genuine ambiguous terms. The progression from zero-shot through few-shot to CoT confirms the findings of Unterbusch and Vogelsang (2026) that reasoning traces substantially improve LLM performance on implicit inference tasks. All results are presented in a tabular format below through Table 7, and further visual depiction is presented in Figure 12.

Table 7: Performance Results- Assumption Detection
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=1 colspan=1>Keyword-Based (KB)</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.46</td></tr><tr><td rowspan=1 colspan=1>QuARS-Adapted (QA)</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.49</td></tr><tr><td rowspan=1 colspan=1>Zero-Shot LLM (ZS-LLM)</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.62</td></tr><tr><td rowspan=1 colspan=1>HiddenLens (ZS prompt)</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.65</td></tr><tr><td rowspan=1 colspan=1>HiddenLens (FS prompt)</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.72</td></tr><tr><td rowspan=1 colspan=1>HiddenLens (CoT prompt)</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.79</td></tr></table>

The BERT NFR classifier (Layer 3) achieves macro F1 Score of 0.847 on the held-out test split, with highest performance on Security (F1 Score 0.89) and lowest on Domain (F1Score 0.79), reflecting the inherent breadth and context-dependence of the Domain category. These results are competitive with NoRBERT (Hey et al., 2020) on the PROMISE NFR benchmark PROMISE (Cleland-Huang et al., 2007), despite our classifier addressing a novel assumption-classification task rather than standard requirement type classification.

![](images/cc1023a9fc13249f95b9d1d5042ee0bea7b7f4c6084aee6dbaa202b8b1a1c001.jpg)  
Figure 12. Performance comparison (Precision, Recall and F1) between HiddenLens and other baseline models

## 7.2 Comparative Analysis with Prior Work

Each cited work is a real contribution to its target problem (requirements smell detection, NFR classification, defect prediction), but none of them were designed with implicit content as its main object of analysis. The closest prior work to HiddenLens is by Femmer et al. (2017), which considers requirements quality as a property of the text itself, rather than a mapping to an external standard. The most methodologically similar work is the work of Unterbusch and Vogelsang (2026) on defect prediction, which is also committed to CoT prompting and human-LLM collaboration like HiddenLens. More such comparisons are presented in Table 8. The focus on defective explicit content and not implicit omissions means that the architecture and training data are not directly transferable, but the empirical findings are a strong validation for the methodological choices taken by HiddenLens. It builds on that foundation by addressing the assumption gap that remains one of the most significant unresolved problems in the field.

Table 8: Comparative Analysis with Prior Approaches of Detecting hidden assumptions in RE
<table><tr><td rowspan=1 colspan=1>Authors/Year</td><td rowspan=1 colspan=1>Technique</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Limitation</td><td rowspan=1 colspan=1>OurProposedWork -HiddenLens</td></tr><tr><td rowspan=1 colspan=1>Femmer et al.(2017)</td><td rowspan=1 colspan=1>POS rules +heuristics</td><td rowspan=1 colspan=1>Requirements smelldetection</td><td rowspan=1 colspan=1>No implicitcontent</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Hey et al. (2020)</td><td rowspan=1 colspan=1>BERT transferlearning</td><td rowspan=1 colspan=1>NFR classification</td><td rowspan=1 colspan=1>Typed text only</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Luo et al. (2022)</td><td rowspan=1 colspan=1>Prompt BERT(PRCBERT)</td><td rowspan=1 colspan=1>Zero-shot REclassification</td><td rowspan=1 colspan=1>No assumptionoutput</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Unterbusch andVogelsang (2026)</td><td rowspan=1 colspan=1>CoT + human-in-loop</td><td rowspan=1 colspan=1>Defect prediction</td><td rowspan=1 colspan=1>Defects, notassumptions</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Ferrari et al.(2024)</td><td rowspan=1 colspan=1>LLM ambiguitydetection</td><td rowspan=1 colspan=1>Ambiguity classification</td><td rowspan=1 colspan=1>Stated contentonly</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Zadenoori et al.(2025)</td><td rowspan=1 colspan=1>SLR of LLMs in RE</td><td rowspan=1 colspan=1>Survey</td><td rowspan=1 colspan=1>No tool; noassumptions</td><td rowspan=1 colspan=1>√</td></tr></table>

## 7.3 Input and Output - HiddenLens

Table 9 presents three representative examples of HiddenLens's Layer 2 and Layer 4 output, spanning three different NFR categories: reliability, performance, and security. Each row shows a single vague term extracted from a real requirement, the specific hidden assumption HiddenLens infers behind it, and the clarifying question generated for the analyst to resolve it, illustrating how the framework moves from an ambiguous word to a concrete, answerable specification gap.

Table 9: HidenLens Input and Output
<table><tr><td rowspan=1 colspan=1>Requirement</td><td rowspan=1 colspan=1>Detected Assumption</td><td rowspan=1 colspan=1>Generated Question</td></tr><tr><td rowspan=1 colspan=1>The system shall respondimmediately inemergencies.</td><td rowspan=1 colspan=1>[Reliability] &quot;Immediately&quot; implies sub-second alert delivery to on-call mobiledevices, assuming reliable networkavailability.</td><td rowspan=1 colspan=1>What is the maximum acceptableresponse time in an emergencyscenario, and what channels mustreceive the alert?</td></tr><tr><td rowspan=1 colspan=1>Doctors can updatepatient records quickly.</td><td rowspan=1 colspan=1>[Performance] &quot;Quickly&quot; implies recordupdate operations complete in &lt; 500 msat 95th percentile under concurrentclinical load.</td><td rowspan=1 colspan=1>What response time threshold defines&quot;quickly&quot; for record updates, andunder what user concurrency?</td></tr><tr><td rowspan=1 colspan=1>Data shall be storedsecurely.</td><td rowspan=1 colspan=1>[Security] &quot;Securely&quot; implies AES-256encryption at rest, TLS 1.3 in transit,and RBAC with least-privilege access.</td><td rowspan=1 colspan=1>What encryption standard and accesscontrol model is required, and whichregulatory framework applies?</td></tr></table>

## 8. Case Study – Hospital Patient Management System

This section presents the HiddenLens architecture and demonstrates its six-layer pipeline through a case study involving a Hospital Patient Management System. The requirement provided by the stakeholders in natural language is as follows:

"The Hospital Patient Management System allows doctors, nurses, and administrators to manage patient records efficiently. Doctors can view and update patient records quickly. Nurses can administer medication and update patient status in real time. The system must ensure patient data is stored securely and only accessible to authorized personnel. In emergency situations, the system should respond immediately and notify the relevant staff. Administrators generate daily reports and manage staff schedules regularly."

Each of the six layers of HiddenLens pipeline was applied sequentially to these requirements. In layer 1, vague terms such as quickly, real time, securely, authorized personnel, immediately, and regularly were identified. Assumption hypotheses for each of the above terms were created for layer 2. For instance, assumption hypotheses that implied two seconds response time assumed by the developer were created for the term quickly, and assumption hypotheses based on role-based access control and at-rest encryption were made for the term securely. These assumptions were then categorized using a BERT classifier into NFR categories (performance, security, scalability, and domain compliance). Then, in Layer 4, LLM-driven clarification suggestions were provided with chain-of-thought prompting, for instance, replacing the phrase “Doctors can view and update patient records quickly” with “The system shall display a patient record within 200 ms under normal load and within 500 ms under peak load.” After validation from the analyst in Layer 5, Layer 6 performed regex-based substitution for rewriting the requirements.

In order to illustrate the improvement in requirement clarity that HiddenLens makes possible, consider the case of the Hospital Patient Management System requirements before and after processing. There are five requirements that use six fuzzy terms in the process: "quickly," "real time," "stored securely," "authorized personnel," "immediately," and "regularly." All of these terms lack measures, scopes, and testable criteria for acceptance. If a software developer were to read these requirements, there would be no common point from which verification could be done, and the developer would be left to determine his/her own criteria for each term.

After passing through all six layers of HiddenLens, all requirements are tied to concrete, verifiable specifications: response times are specified in milliseconds with load conditions appended, encryption is linked to named standards (AES-256, TLS 1.3), access control is linked to a documented Role-Based Access Control (RBAC) policy with mandatory audit logging, emergency notifications come with a three-second Service-Level Agreement (SLA) with named recipient roles, and administrative scheduling is scoped to a defined staff ceiling with a conflict-resolution deadline. The change is from requirements that are syntactically correct but semantically underspecified, that is, requirements that might pass a casual review but would fail at implementation or acceptance testing, to the requirements that meet the ISO/IEC/IEEE 29148 (ISO/IEC/IEEE, 2018) criteria of unambiguity, verifiability and completeness. This improvement was made without losing the original intent of any requirement. HiddenLens surfaces and resolves hidden assumptions rather than replacing stakeholder decisions, preserving the what while making the how measurable.

Table 10 below details the output in a tabular format showcasing a ‘before vs after’ of the SRS documentation. In the rewritten SRS, the IEEE standard followed has also been depicted for better viability. Figure 13 presents the output displayed in the Google Colab environment, whereas Figure 14 shows a representative step in the rewriting process at which the user is prompted to select an option.

Table 10. SRS Case Study before HiddenLens rewrite vs. after rewrite
<table><tr><td rowspan=1 colspan=1>Req. ID</td><td rowspan=1 colspan=1>NFR Category</td><td rowspan=1 colspan=1>Before HiddenLens: OriginalRequirement</td><td rowspan=1 colspan=1>Ambiguous Terms</td><td rowspan=1 colspan=1>Hidden Assumption</td></tr><tr><td rowspan=1 colspan=1>R1</td><td rowspan=1 colspan=1>Performance</td><td rowspan=1 colspan=1>Doctors can view and update patientrecords quickly.</td><td rowspan=1 colspan=1>Quickly</td><td rowspan=1 colspan=1>Developer assumes &lt;2sresponse; no load conditionstated.</td></tr><tr><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>Performance /Scalability</td><td rowspan=1 colspan=1>Nurses can administer medicationand update patient status in realtime.</td><td rowspan=1 colspan=1>real time</td><td rowspan=1 colspan=1>Implies instant sync;propagation delay andconcurrency undefined.</td></tr><tr><td rowspan=1 colspan=1>R3</td><td rowspan=1 colspan=1>Security / Domain</td><td rowspan=1 colspan=1>The system must ensure patientdata is stored securely and onlyaccessible to authorizedpersonnel.</td><td rowspan=1 colspan=1>stored securely, authorizedpersonnel</td><td rowspan=1 colspan=1>Encryption standard, RBACpolicy, and audit logging allunstated.</td></tr><tr><td rowspan=1 colspan=1>R4</td><td rowspan=1 colspan=1>Performance /Domain</td><td rowspan=1 colspan=1>In emergency situations, the systemshould respond immediately andnotify the relevant staff.</td><td rowspan=1 colspan=1>immediately, relevant staff</td><td rowspan=1 colspan=1>Threshold undefined; notifiedroles and channels unspecified.</td></tr><tr><td rowspan=1 colspan=1>R5</td><td rowspan=1 colspan=1>Scalability /Domain</td><td rowspan=1 colspan=1>Administrators generate dailyreports and manage staff schedulesregularly.</td><td rowspan=1 colspan=1>Regularly</td><td rowspan=1 colspan=1>Schedule frequency, staffvolume, and conflict resolutionSLA all implicit.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>After HiddenLens: Rewritten Requirements</td></tr><tr><td rowspan=1 colspan=1>Req. ID</td><td rowspan=1 colspan=1>NFR Category</td><td rowspan=1 colspan=1>After Hidden Lens Rewrite</td><td rowspan=1 colspan=1>Resolved Specifications</td><td rowspan=1 colspan=1>ISO/IEC/IEEE 29148 CriteriaMet</td></tr><tr><td rowspan=1 colspan=1>R1</td><td rowspan=1 colspan=1>Performance,Scalability</td><td rowspan=1 colspan=1>The system shall display a patientrecord within 200 ms under normalload and within 500 ms under peakload (≤500 concurrent users).</td><td rowspan=1 colspan=1>200 ms 500 ms 500concurrent users</td><td rowspan=1 colspan=1>Verifiable, Unambiguous</td></tr><tr><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>Performance,Scalability</td><td rowspan=1 colspan=1>Medication administration updatesshall be reflected system-widewithin 2 seconds of submission,with a maximum propagation delayof 5 seconds under peak ward load.</td><td rowspan=1 colspan=1>2 seconds 5 seconds</td><td rowspan=1 colspan=1>Verifiable, Complete</td></tr><tr><td rowspan=1 colspan=1>R3</td><td rowspan=1 colspan=1>Security, DomainCompliance</td><td rowspan=1 colspan=1>Patient data shall be encrypted atrest using AES-256 and in transitusing TLš 1.3. Access shall berestricted to roles defined in theRBAC policy; all access eventsshall be audit-logged.</td><td rowspan=1 colspan=1>AES-256 TLS 1.3 RBACpolicy</td><td rowspan=1 colspan=1>Verifiable, Unambiguous,Complete</td></tr><tr><td rowspan=1 colspan=1>R4</td><td rowspan=1 colspan=1>Performance,DomainCompliance</td><td rowspan=1 colspan=1>Emergency alerts shall bedispatched to on-call personnelwithin 3 seconds of trigger. Atminimum the attending physicianand ward nurse roles shall receivenotification via pager and in-appalert.</td><td rowspan=1 colspan=1>3 seconds attendingphysician ward nurse</td><td rowspan=1 colspan=1>Verifiable, Unambiguous</td></tr><tr><td rowspan=1 colspan=1>R5</td><td rowspan=1 colspan=1>Scalability,DomainCompliance</td><td rowspan=1 colspan=1>Administrators shall generatereports on a configurableschedule (minimum: daily). Thescheduling module shall support upto 200 staff members and resolveconflicts within one working day.</td><td rowspan=1 colspan=1>configurable schedule 200staff members one workingday</td><td rowspan=1 colspan=1>Verifiable, Complete</td></tr></table>

![](images/f3925285cbe6261bb78d1c822ed6b3efc9d9e0efe771116db0cbb19b33d34a02.jpg)  
Fig 13. Rewritten Case Study Output

![](images/875e967ffb7921f3640ecfd2d4b0a07c8a0ac72f867ab93054d31e3ecf707656.jpg)  
Fig 14. Output Screen

## 9. Threats to Validity

This section discusses the threats to the validity of our findings, organized across internal, external, construct, and LLM-specific dimensions.

Internal validity: The main risk arises from the subjective nature of annotations. Although the inter-annotator agreement (� = 0.74 on assumption presence) is high, the difference between hidden assumptions and implicit expectations of stakeholders is not always clear-cut. Formal definitions of Section 3.1 reduce but do not entirely resolve this issue. The systematic annotator bias could over- or underestimate some categories of NFRs.

External validity: The findings are based on a dataset compiled from three specific sources. Norheim et al. (2024) observed that published requirements datasets, which typically contain between a few dozen and a few thousand requirements, remain substantially smaller than real-world industrial specifications, which may contain hundreds of thousands of requirements. The 312-requirement HLC is subject to the same limitation. Consequently, the transferability of the findings to assumptions in requirements from other domains, such as embedded systems, telecommunications, and intelligent systems, requires domain-specific validation. Furthermore, industrial requirements may be more concise than examples reported in the scholarly literature, potentially resulting in a higher density of assumptions per sentence.

Construct validity: The chosen criterion for identifying TPs, ROUGE-L overlap ≥ 0.4 is an acceptable approximation only. Gold standard assumptions that employ specialized domain-specific terminology not contained in the text of the predicted assumption will be automatically counted as False Negatives (FN). The gold assumptions, which incorporate terms unique to the field that are not included in the text of the assumed hypothesis, can mistakenly be considered false negatives even with identical semantics.

Limitations of LLM: It is possible for LLMs to generate assumptions that appear logical but are not really necessary for the implementation process. Layer 5, which is the validation loop, is the safety valve but leaves the responsibility on the shoulders of the analyst.

## 10. Conclusion and Future Work

This paper proposes HiddenLens, a novel six-layer BERT-LLM pipeline and the first framework engineered specifically to detect, classify, question, and resolve hidden assumptions in natural language software requirements. The framework leverages BERT attention entropy with LLM reasoning, BERT fine-tuning for NFR classification, context-aware suggestion generation, and human-in-the-loop validation to address an aspect of requirements quality that has hitherto received inadequate attention: the hidden world-view beliefs about the nature of the world that drive design choices in implementations.

Through experimental results on the 312 requirements in the HiddenLens Corpus (HLC), annotated with hidden assumptions, we achieve an F1 score of 0.79 for assumption identification and an F1 score of 0.847 for BERT classifier-based NFR classification. The case study on requirements from a Hospital Management System is one such illustration of the real-world importance of HiddenLens for safety-critical systems. To the best of our knowledge, HiddenLens represents the first dedicated attempt to address the detection and resolution of hidden assumptions in requirements engineering (RE). It can assist analysts, developers, and organisations in identifying potential problems before they become costly or contribute to late-stage project failures, particularly when such problems cannot be identified by conventional ambiguity-detection tools during the early stages of the software development life cycle (SDLC).

Future directions include incorporating HiddenLens into commercial software development tools like IDEs and requirement management tools in order to investigate the influence of HiddenLens on projects over time. Overall, HiddenLens aims to close the gap between what a requirement states and what it silently assumes, turning tacit engineering judgment into an explicit, testable artifact.

## Data Availability Statement

The data (HiddenLens Corpus) and codes developed and analysed during the current study will be made publicly available in a Github repository upon acceptance of the manuscript.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this manuscript, the author(s) used ChatGPT-5.5 (OpenAI) to improve sentence structure in selected passages. The author(s) reviewed and edited the output as needed and take full responsibility for the content of the published article.

## References

Abad, Z. S. H., Karras, O., Ghazi, P., Glinz, M., Ruhe, G., & Schneider, K. (2017, September). What works better? a study of classifying requirements. In 2017 IEEE 25th international requirements engineering conference (RE) (pp. 496-501). IEEE.

Abualhaija, S., Aydemir, F. B., Dalpiaz, F., Dell'Anna, D., Ferrari, A., Franch, X., & Fucci, D. (2024). Replication in requirements engineering: the NLP for RE case. ACM Transactions on Software Engineering and Methodology, 33(6), 1-33.

Almonte, J. T., Boominathan, S. A., & Nascimento, N. (2025, November). Automated non-functional requirements generation in software engineering with large language models: A comparative study. In 2025 IEEE International Conference on Collaborative Advances in Software and Computing (CASCON) (pp. 474-483). IEEE.

Ambriola, V., & Gervasi, V. (1997, November). Processing natural language requirements. In Proceedings 12th IEEE International Conference Automated Software Engineering (pp. 36-45). IEEE.

Arora, C., Sabetzadeh, M., Briand, L., & Zimmer, F. (2016). Automated extraction and clustering of requirements glossary terms. IEEE Transactions on Software Engineering, 43(10), 918-945.

Bajceta, A., Leon, M., Afzal, W., Lindberg, P., & Bohlin, M. (2022, March). Using NLP Tools to Detect Ambiguities in System Requirements-A Comparison Study. In REFSQ Workshops.

Beqiri, L., Montero, C. S., Cicchetti, A., & Kruglyak, A. (2024, June). Classifying ambiguous requirements: An explainable approach in railway industry. In 2024 IEEE 32nd International Requirements Engineering Conference Workshops (REW) (pp. 12-21). IEEE.

Berry, D. M., Kamsties, E., & Krieger, M. M. (2003). From contract drafting to software specification: Linguistic sources of ambiguity (Handbook, Version 1.0). University of Waterloo.

Binkhonain, M., & Alfayez, R. (2025). Are prompts all you need? Evaluating prompt-based Large Language Models (LLMs) for software requirements classification. Requirements Engineering, 30(4), 423-443.

Boehm, B. W. (1984). Software engineering economics. IEEE Transactions on Software Engineering, (1), 4- 21.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877- 1901.

Cleland-Huang, J., Settimi, R., Zou, X., & Solc, P. (2007). Automated classification of non-functional requirements. Requirements Engineering, 12(2), 103-120.

Fabbrini, F., Fusani, M., Gnesi, S., & Lami, G. (2001). An automatic quality evaluation for natural language requirements. In Proceedings of the 7th International Workshop on Requirements Engineering: Foundation for Software Quality (REFSQ '01).

Femmer, H., Fernández, D. M., Wagner, S., & Eder, S. (2017). Rapid quality assurance with requirements smells. Journal of Systems and Software, 123, 190-213.

Ferrari, A., Abualhaija, S., & Arora, C. (2024, June). Model generation with LLMs: From requirements to UML sequence diagrams. In 2024 IEEE 32nd International Requirements Engineering Conference Workshops (REW) (pp. 291-300). IEEE.

Ferrari, A., Spagnolo, G. O., & Gnesi, S. (2017, September). Pure: A dataset of public requirements documents. In 2017 IEEE 25th international requirements engineering conference (RE) (pp. 502-505). IEEE.

Gervasi, V., Ferrari, A., Zowghi, D., & Spoletini, P. (2019). Ambiguity in requirements engineering: Towards a unifying framework. In From Software Engineering to Formal Methods and Tools, and Back: Essays Dedicated to Stefania Gnesi on the Occasion of Her 65th Birthday (pp. 191-210). Cham: Springer International Publishing.

Guo, J., Cheng, J., & Cleland-Huang, J. (2017, May). Semantically enhanced software traceability using deep learning techniques. In 2017 IEEE/ACM 39th International Conference on Software Engineering (ICSE) (pp. 3-14). IEEE.

Häring, I. (2021). The standard IEC 61508 and its safety life cycle. In Technical Safety, Reliability and Resilience: Methods and Processes (pp. 193-207). Singapore: Springer Singapore.

Hey, T., Keim, J., Koziolek, A., & Tichy, W. F. (2020, August). Norbert: Transfer learning for requirements classification. In 2020 IEEE 28th international requirements engineering conference (RE) (pp. 169-179). IEEE.

Hymel, C., & Johnson, H. (2025). Analysis of LLMs vs human experts in requirements engineering. arXiv preprint arXiv:2501.19297.

ISO/IEC/IEEE 29148. (2018). Systems and software engineering—Life cycle processes—Requirements engineering. International Organization for Standardization.

Johnson, J. (2020). Chaos report: Beyond Infinity. The Standish Group International. Inc., Boston, USA.

Lami, G., Gnesi, S., Fabbrini, F., Fusani, M., & Trentanni, G. (2004). An automatic tool for the analysis of natural language requirements. Informe técnico, CNR Information Science and Technology Institute, Pisa, Italia, Setiembre.

Leffingwell, D., & Widrig, D. (2003). Managing software requirements: A use case approach (2nd ed.). Addison-Wesley.

Lubos, S., Felfernig, A., Tran, T. N. T., Garber, D., El Mansi, M., Erdeniz, S. P., & Le, V. M. (2024, June). Leveraging llms for the quality assurance of software requirements. In 2024 IEEE 32nd International Requirements Engineering Conference (RE) (pp. 389-397). IEEE.

Luo, X., Xue, Y., Xing, Z., & Sun, J. (2022, October). Prcbert: Prompt learning for requirement classification using bert-based pretrained language models. In Proceedings of the 37th IEEE/ACM international conference on automated software engineering (pp. 1-13).

Necula, S. C., Dumitriu, F., & Greavu-Șerban, V. (2024). A systematic literature review on using natural language processing in software requirements engineering. Electronics, 13(11), 2055.

Norheim, J. J., Rebentisch, E., Xiao, D., Draeger, L., Kerbrat, A., & de Weck, O. L. (2024). Challenges in applying large language models to requirements engineering tasks. Design Science, 10, e16.

Nuseibeh, B., Easterbrook, S., & Russo, A. (2001). Making inconsistency respectable in software development. Journal of Systems and Software, 58(2), 171-180.

Onyeka, E. (2013, May). A process framework for managing implicit requirements using analogy-based reasoning: Doctoral consortium paper. In IEEE 7th International Conference on Research Challenges in Information Science (RCIS) (pp. 1-5). IEEE.

Unterbusch, M., & Vogelsang, A. (2026, April). Context-Adaptive Requirements Defect Prediction through Human-LLM Collaboration. In Proceedings of the IEEE/ACM 48th International Conference on Software Engineering (pp. 221-225).

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35, 24824-24837.

Wilson, W. M., Rosenberg, L. H., & Hyatt, L. E. (1997, May). Automated analysis of requirement specifications. In Proceedings of the 19th international conference on Software engineering (pp. 161- 171).

A version (09\_Sept\_2026) of this manuscript has been submitted to the journal for publication and is presently under review

Winkler, J., & Vogelsang, A. (2016, September). Automatic classification of requirements based on convolutional neural networks. In 2016 IEEE 24th International Requirements Engineering Conference Workshops (REW) (pp. 39-45). IEEE.

Zadenoori, M. A., Dąbrowski, J., Alhoshan, W., Zhao, L., & Ferrari, A. (2025). Large language models (llms) for requirements engineering (re): A systematic literature review. arXiv preprint arXiv:2509.11446.